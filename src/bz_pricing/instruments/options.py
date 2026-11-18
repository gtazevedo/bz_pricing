import datetime
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from bz_pricing.core.calendars import get_calendar
from bz_pricing.core.curves import Curve

try:
    from bz_pricing.core.options_core import (
        asian_arithmetic_tw_price as _cy_asian_arith,
        asian_geometric_price as _cy_asian_geom,
        bachelier_price_greeks as _cy_bachelier,
        barrier_bsm_price as _cy_barrier_bsm,
        black76_price_greeks as _cy_black76,
        bsm_price_greeks as _cy_bsm,
        corrado_su_price as _cy_corrado_su,
        crr_american_price as _cy_crr_american,
        crr_american_price_greeks as _cy_crr_american_greeks,
        implied_vol_bachelier as _cy_iv_bachelier,
        implied_vol_black76 as _cy_iv_black76,
        implied_vol_bsm as _cy_iv_bsm,
        price_black76_batch as _cy_price_b76_batch,
        price_options_batch as _cy_price_batch,
    )
except ImportError:
    _cy_bsm = None
    _cy_black76 = None
    _cy_bachelier = None
    _cy_barrier_bsm = None
    _cy_asian_geom = None
    _cy_asian_arith = None
    _cy_crr_american = None
    _cy_crr_american_greeks = None
    _cy_corrado_su = None
    _cy_iv_bsm = None
    _cy_iv_black76 = None
    _cy_iv_bachelier = None
    _cy_price_b76_batch = None
    _cy_price_batch = None


def _norm_cdf_py(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf_py(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class _BaseOption:
    """Classe base para contratos de opções listadas e de balcão (padrão B3 / Bloomberg)."""

    def __init__(
        self,
        ticker: str,
        maturity_date: Union[datetime.date, str],
        strike: float,
        option_type: str = "CALL",
        exercise: str = "EUROPEAN",
        calendar_name: str = "B3",
        multiplier: float = 1.0,
        holidays: Any = None,
    ):
        self.ticker = ticker.strip().upper()
        self.maturity_date = self._normalize_date(maturity_date)
        self.strike = float(strike)
        opt_t = option_type.strip().upper()
        if opt_t not in ("CALL", "PUT"):
            raise ValueError(f"option_type inválido: {option_type}. Use 'CALL' ou 'PUT'.")
        self.option_type = opt_t

        ex_t = exercise.strip().upper()
        if ex_t not in ("EUROPEAN", "AMERICAN"):
            raise ValueError(f"exercise inválido: {exercise}. Use 'EUROPEAN' ou 'AMERICAN'.")
        self.exercise = ex_t
        self.calendar = get_calendar(calendar_name, holidays=holidays)
        self.multiplier = float(multiplier)

    @staticmethod
    def _normalize_date(d: Union[datetime.date, str]) -> datetime.date:
        if isinstance(d, datetime.datetime):
            return d.date()
        if isinstance(d, datetime.date):
            return d
        return datetime.date.fromisoformat(str(d).strip()[:10])

    @staticmethod
    def _get_rate_for_du(rate_or_curve: Union[float, Curve], du: int, dc: int = 0) -> float:
        if isinstance(rate_or_curve, Curve):
            interp = getattr(rate_or_curve, "interpolator", None)
            basis = getattr(rate_or_curve, "day_count_basis", None)
            if basis is None and interp is not None:
                basis = getattr(interp, "day_count_basis", None)
            if basis is not None and abs(float(basis) - 252.0) > 1e-6:
                t_query = float(dc) if dc > 0 else float(du)
                return float(rate_or_curve.get_rate(max(1.0, t_query), float(dc)))
            return float(rate_or_curve.get_rate(max(1, int(du)), max(0, int(dc))))
        return float(rate_or_curve)

    @staticmethod
    def _resolve_cont_rate(
        rate_or_curve: Union[float, Curve],
        du: int,
        dc: int = 0,
        default_conv: str = "exp_252",
    ) -> float:
        if du <= 0:
            return 0.0
        t_du = du / 252.0
        if isinstance(rate_or_curve, Curve):
            interp = getattr(rate_or_curve, "interpolator", None)
            meta = (
                rate_or_curve
                if hasattr(rate_or_curve, "day_count_basis")
                else (interp if interp is not None and hasattr(interp, "day_count_basis") else None)
            )
            if meta is not None:
                basis = float(meta.day_count_basis)
                comp = str(meta.compounding).strip().lower()
                freq = int(getattr(meta, "freq", 1))
                if abs(basis - 252.0) > 1e-6:
                    dc_eff = float(dc) if dc > 0 else float(du) * (basis / 252.0)
                    r_val = float(rate_or_curve.get_rate(max(1.0, dc_eff), float(dc_eff)))
                    tau = dc_eff / basis
                else:
                    r_val = float(rate_or_curve.get_rate(max(1, int(du)), max(0, int(dc))))
                    tau = t_du
                if comp in ("continuous", "cont"):
                    df = math.exp(-r_val * tau)
                elif comp in ("simple", "simple_linear", "linear"):
                    df = 1.0 / max(1e-12, 1.0 + r_val * tau)
                elif comp in ("exponential", "exponential_b3", "exp_252", "b3"):
                    df = (1.0 + max(-0.99, r_val)) ** (-tau)
                else:
                    k = float(max(1, freq))
                    df = (1.0 + max(-0.99, r_val) / k) ** (-k * tau)
                return -math.log(max(1e-12, df)) / t_du

        r_raw = _BaseOption._get_rate_for_du(rate_or_curve, du, dc)
        if default_conv == "simple_360":
            dc_eff = float(dc) if dc > 0 else float(du)
            return math.log(max(1e-8, 1.0 + r_raw * (dc_eff / 360.0))) / t_du
        return math.log(1.0 + max(-0.99, r_raw))

    def _resolve_vol(self, vol_or_surface: Any, du: int, forward: float) -> float:
        if hasattr(vol_or_surface, "get_vol"):
            return float(vol_or_surface.get_vol(max(1, int(du)), strike=self.strike, forward=float(forward)))
        return float(vol_or_surface)

    def du(self, base_date: Union[datetime.date, str]) -> int:
        base_dt = self._normalize_date(base_date)
        return max(0, int(self.calendar.bizdays(base_dt, self.maturity_date)))

    def dc(self, base_date: Union[datetime.date, str]) -> int:
        base_dt = self._normalize_date(base_date)
        return max(0, (self.maturity_date - base_dt).days)

    def time_to_maturity(self, base_date: Union[datetime.date, str]) -> float:
        return self.du(base_date) / 252.0


class EquityOption(_BaseOption):
    """
    Opção sobre ação, ETF ou índice (Manual de Apreçamento de Opções B3 / Bloomberg):
    - r_n = ln(1 + T_PreDI1_n)
    - q_n = ln(1 + T_CY_n) (custo de carrego / dividend yield contínuo)
    - Exercício europeu por Black-Scholes-Merton ou americano via árvore binomial CRR em Cython.
    """

    def _resolve_params(
        self,
        base_date: Union[datetime.date, str],
        pre_rate_or_curve: Union[float, Curve],
        carry_or_div_yield: Union[float, Curve] = 0.0,
    ):
        du_n = self.du(base_date)
        dc_n = self.dc(base_date)
        t_n = du_n / 252.0
        r_cont = self._resolve_cont_rate(pre_rate_or_curve, du_n, dc_n, default_conv="exp_252")
        q_cont = self._resolve_cont_rate(carry_or_div_yield, du_n, dc_n, default_conv="exp_252")
        return du_n, t_n, r_cont, q_cont

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        spot: float,
        pre_rate_or_curve: Union[float, Curve],
        vol: float,
        carry_or_div_yield: Union[float, Curve] = 0.0,
        crr_steps: int = 200,
    ) -> Dict[str, float]:
        du_n, t_n, r_cont, q_cont = self._resolve_params(base_date, pre_rate_or_curve, carry_or_div_yield)
        is_call = self.option_type == "CALL"

        if _cy_bsm is None:
            raise RuntimeError("Extensão Cython 'options_core' não compilada.")

        fwd_est = float(spot) * math.exp((r_cont - q_cont) * t_n)
        vol_val = self._resolve_vol(vol, du_n, fwd_est)

        price_eu, delta, gamma, vega, theta, rho, vanna, volga = _cy_bsm(
            float(spot), self.strike, r_cont, q_cont, t_n, vol_val, is_call
        )
        if self.exercise == "AMERICAN":
            if _cy_crr_american_greeks is not None:
                price, delta, gamma, vega, theta, rho = _cy_crr_american_greeks(
                    float(spot), self.strike, r_cont, q_cont, t_n, vol_val, is_call, int(crr_steps)
                )
            else:
                price = _cy_crr_american(
                    float(spot), self.strike, r_cont, q_cont, t_n, vol_val, is_call, int(crr_steps)
                )
        else:
            price = price_eu

        return {
            "du": float(du_n),
            "T": t_n,
            "r_cont": r_cont,
            "q_cont": q_cont,
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "vega_1pct": float(vega * 0.01),
            "theta": float(theta),
            "theta_du": float(theta / 252.0),
            "rho": float(rho),
            "vanna": float(vanna),
            "volga": float(volga),
        }

    def price(
        self,
        base_date: Union[datetime.date, str],
        spot: float,
        pre_rate_or_curve: Union[float, Curve],
        vol: float,
        carry_or_div_yield: Union[float, Curve] = 0.0,
        crr_steps: int = 200,
    ) -> float:
        return self.risk_metrics(base_date, spot, pre_rate_or_curve, vol, carry_or_div_yield, crr_steps)["price"]

    def implied_volatility(
        self,
        base_date: Union[datetime.date, str],
        market_price: float,
        spot: float,
        pre_rate_or_curve: Union[float, Curve],
        carry_or_div_yield: Union[float, Curve] = 0.0,
    ) -> float:
        _, t_n, r_cont, q_cont = self._resolve_params(base_date, pre_rate_or_curve, carry_or_div_yield)
        is_call = self.option_type == "CALL"
        if self.exercise == "EUROPEAN" and _cy_iv_bsm is not None:
            return float(_cy_iv_bsm(float(market_price), float(spot), self.strike, r_cont, q_cont, t_n, is_call))
        from scipy.optimize import brentq

        def _obj(s: float) -> float:
            return self.price(base_date, spot, pre_rate_or_curve, s, carry_or_div_yield) - float(market_price)

        return float(brentq(_obj, 1e-4, 5.0, xtol=1e-8))


class FXOption(_BaseOption):
    """
    Opção sobre taxa de câmbio BRL/USD (Garman-Kohlhagen, padrão B3):
    - r_n = ln(1 + T_PreDI1_n)
    - q_n = (252 / DU_n) * ln(1 + T_PreDDI_n * DC_n / 360)
    - Suporte opcional ao ajuste de volatilidade de fixing t-1: sigma_adj = sigma * sqrt(DU / (DU + 1)).
    """

    def _resolve_fx_params(
        self,
        base_date: Union[datetime.date, str],
        pre_rate_or_curve: Union[float, Curve],
        cupom_limpo_or_curve: Union[float, Curve],
        vol: Any,
        fixing_vol_adjust: bool = False,
        spot_ptax: Optional[float] = None,
    ):
        du_n = self.du(base_date)
        dc_n = self.dc(base_date)
        t_n = du_n / 252.0
        r_cont = self._resolve_cont_rate(pre_rate_or_curve, du_n, dc_n, default_conv="exp_252")
        q_cont = self._resolve_cont_rate(cupom_limpo_or_curve, du_n, dc_n, default_conv="simple_360")
        fwd_est = float(spot_ptax) * math.exp((r_cont - q_cont) * t_n) if spot_ptax is not None else self.strike
        vol_val = self._resolve_vol(vol, du_n, fwd_est)
        if du_n > 0:
            sigma_eff = vol_val * math.sqrt(du_n / (du_n + 1.0)) if fixing_vol_adjust else vol_val
        else:
            sigma_eff = vol_val
        return du_n, dc_n, t_n, r_cont, q_cont, sigma_eff

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        spot_ptax: float,
        pre_rate_or_curve: Union[float, Curve],
        cupom_limpo_or_curve: Union[float, Curve],
        vol: Any,
        fixing_vol_adjust: bool = False,
    ) -> Dict[str, float]:
        du_n, dc_n, t_n, r_cont, q_cont, sigma_eff = self._resolve_fx_params(
            base_date,
            pre_rate_or_curve,
            cupom_limpo_or_curve,
            vol,
            fixing_vol_adjust=fixing_vol_adjust,
            spot_ptax=spot_ptax,
        )
        is_call = self.option_type == "CALL"
        price, delta, gamma, vega, theta, rho, vanna, volga = _cy_bsm(
            float(spot_ptax), self.strike, r_cont, q_cont, t_n, sigma_eff, is_call
        )
        return {
            "du": float(du_n),
            "dc": float(dc_n),
            "T": t_n,
            "r_cont": r_cont,
            "q_cont": q_cont,
            "vol_eff": sigma_eff,
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "vega_1pct": float(vega * 0.01),
            "theta": float(theta),
            "theta_du": float(theta / 252.0),
            "rho": float(rho),
            "vanna": float(vanna),
            "volga": float(volga),
        }

    def price(
        self,
        base_date: Union[datetime.date, str],
        spot_ptax: float,
        pre_rate_or_curve: Union[float, Curve],
        cupom_limpo_or_curve: Union[float, Curve],
        vol: float,
        fixing_vol_adjust: bool = False,
    ) -> float:
        return self.risk_metrics(
            base_date, spot_ptax, pre_rate_or_curve, cupom_limpo_or_curve, vol, fixing_vol_adjust
        )["price"]

    def implied_volatility(
        self,
        base_date: Union[datetime.date, str],
        market_price: float,
        spot_ptax: float,
        pre_rate_or_curve: Union[float, Curve],
        cupom_limpo_or_curve: Union[float, Curve],
        fixing_vol_adjust: bool = False,
    ) -> float:
        du_n, _, t_n, r_cont, q_cont, _ = self._resolve_fx_params(
            base_date,
            pre_rate_or_curve,
            cupom_limpo_or_curve,
            0.15,
            fixing_vol_adjust=False,
            spot_ptax=spot_ptax,
        )
        iv_eff = float(
            _cy_iv_bsm(
                float(market_price),
                float(spot_ptax),
                self.strike,
                r_cont,
                q_cont,
                t_n,
                self.option_type == "CALL",
            )
        )
        if fixing_vol_adjust and du_n > 0 and not math.isnan(iv_eff):
            return iv_eff / math.sqrt(du_n / (du_n + 1.0))
        return iv_eff


class IDIOption(_BaseOption):
    """
    Opção sobre Índice de Taxa Média de Depósitos Interfinanceiros (IDI - B3):
    - S_n = IDI_0 * exp(r_n * T_n)
    - Black-76 descontado por exp(-r_n * T_n).
    """

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        idi_spot: float,
        pre_rate_or_curve: Union[float, Curve],
        vol: float,
    ) -> Dict[str, float]:
        du_n = self.du(base_date)
        dc_n = self.dc(base_date)
        t_n = du_n / 252.0
        r_cont = self._resolve_cont_rate(pre_rate_or_curve, du_n, dc_n, default_conv="exp_252")
        idi_fwd = float(idi_spot) * math.exp(r_cont * t_n)
        vol_val = self._resolve_vol(vol, du_n, idi_fwd)

        price, delta, gamma, vega, theta, rho, vanna, volga = _cy_black76(
            idi_fwd, self.strike, r_cont, t_n, vol_val, self.option_type == "CALL"
        )
        return {
            "du": float(du_n),
            "T": t_n,
            "idi_forward": idi_fwd,
            "r_cont": r_cont,
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "vega_1pct": float(vega * 0.01),
            "theta": float(theta),
            "rho": float(rho),
            "vanna": float(vanna),
            "volga": float(volga),
        }

    def price(
        self,
        base_date: Union[datetime.date, str],
        idi_spot: float,
        pre_rate_or_curve: Union[float, Curve],
        vol: float,
    ) -> float:
        return self.risk_metrics(base_date, idi_spot, pre_rate_or_curve, vol)["price"]

    def implied_volatility(
        self,
        base_date: Union[datetime.date, str],
        market_price: float,
        idi_spot: float,
        pre_rate_or_curve: Union[float, Curve],
    ) -> float:
        du_n = self.du(base_date)
        dc_n = self.dc(base_date)
        t_n = du_n / 252.0
        r_cont = self._resolve_cont_rate(pre_rate_or_curve, du_n, dc_n, default_conv="exp_252")
        idi_fwd = float(idi_spot) * math.exp(r_cont * t_n)
        return float(
            _cy_iv_black76(float(market_price), idi_fwd, self.strike, r_cont, t_n, self.option_type == "CALL")
        )


class DI1Option(_BaseOption):
    """
    Opção sobre Contrato Futuro de DI1 (Black-76 adaptado ao FRA de DI, padrão B3):
    - S' = (PU_C / PU_L - 1) / (T_L_dc - T_C_dc)
    - K' = ((1 + K)^(T_L_du - T_C_du) - 1) / (T_L_dc - T_C_dc)
    - delta_annuity = PU_L * (T_L_dc - T_C_dc) / (1 + K' * (T_L_dc - T_C_dc))
    """

    def __init__(
        self,
        ticker: str,
        maturity_date: Union[datetime.date, str],
        underlying_maturity_date: Union[datetime.date, str],
        strike_rate: float,
        option_type: str = "CALL",
        calendar_name: str = "B3",
        dc_basis: float = 360.0,
        holidays: Any = None,
    ):
        super().__init__(
            ticker=ticker,
            maturity_date=maturity_date,
            strike=strike_rate,
            option_type=option_type,
            exercise="EUROPEAN",
            calendar_name=calendar_name,
            holidays=holidays,
        )
        self.underlying_maturity_date = self._normalize_date(underlying_maturity_date)
        self.dc_basis = float(dc_basis)

    def _resolve_di1_params(
        self,
        base_date: Union[datetime.date, str],
        pu_c: float,
        pu_l: float,
    ):
        if self.underlying_maturity_date <= self.maturity_date:
            raise ValueError("underlying_maturity_date deve ser estritamente posterior a maturity_date.")
        base_dt = self._normalize_date(base_date)
        du_c = max(0, int(self.calendar.bizdays(base_dt, self.maturity_date)))
        du_l = max(du_c + 1, int(self.calendar.bizdays(base_dt, self.underlying_maturity_date)))
        dc_c = max(0, (self.maturity_date - base_dt).days)
        dc_l = max(dc_c + 1, (self.underlying_maturity_date - base_dt).days)

        t_c_dc = dc_c / self.dc_basis
        t_l_dc = dc_l / self.dc_basis
        dt_dc = t_l_dc - t_c_dc
        dt_du = (du_l - du_c) / 252.0

        s_prime = ((float(pu_c) / float(pu_l)) - 1.0) / dt_dc
        k_prime = (math.pow(1.0 + self.strike, dt_du) - 1.0) / dt_dc
        delta_annuity = float(pu_l) * dt_dc / (1.0 + k_prime * dt_dc)
        return t_c_dc, s_prime, k_prime, delta_annuity

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        pu_c: float,
        pu_l: float,
        vol: float,
    ) -> Dict[str, float]:
        t_c_dc, s_prime, k_prime, delta_annuity = self._resolve_di1_params(base_date, pu_c, pu_l)
        undisc_price, delta, gamma, vega, theta, _, vanna, volga = _cy_black76(
            s_prime, k_prime, 0.0, t_c_dc, float(vol), self.option_type == "CALL"
        )
        return {
            "T_dc": t_c_dc,
            "S_prime": s_prime,
            "K_prime": k_prime,
            "delta_annuity": delta_annuity,
            "price": float(delta_annuity * undisc_price),
            "delta": float(delta_annuity * delta),
            "gamma": float(delta_annuity * gamma),
            "vega": float(delta_annuity * vega),
            "vega_1pct": float(delta_annuity * vega * 0.01),
            "theta": float(delta_annuity * theta),
            "vanna": float(delta_annuity * vanna),
            "volga": float(delta_annuity * volga),
        }

    def price(
        self,
        base_date: Union[datetime.date, str],
        pu_c: float,
        pu_l: float,
        vol: float,
    ) -> float:
        return self.risk_metrics(base_date, pu_c, pu_l, vol)["price"]

    def implied_volatility(
        self,
        base_date: Union[datetime.date, str],
        market_price: float,
        pu_c: float,
        pu_l: float,
    ) -> float:
        t_c_dc, s_prime, k_prime, delta_annuity = self._resolve_di1_params(base_date, pu_c, pu_l)
        if delta_annuity <= 0.0:
            raise ValueError("delta_annuity deve ser positivo para inversão de volatilidade implícita.")
        undisc_target = float(market_price) / delta_annuity
        return float(
            _cy_iv_black76(undisc_target, s_prime, k_prime, 0.0, t_c_dc, self.option_type == "CALL")
        )


class CorradoSuOption(_BaseOption):
    """
    Opção europeia precificada pelo modelo de Corrado & Su (1996) com expansão de
    Gram-Charlier para incorporar assimetria (kappa3) e curtose (kappa4).
    """

    def price(
        self,
        base_date: Union[datetime.date, str],
        spot: float,
        pre_rate_or_curve: Union[float, Curve],
        vol: float,
        kappa3: float = 0.0,
        kappa4: float = 3.0,
        carry_or_div_yield: Union[float, Curve] = 0.0,
    ) -> float:
        du_n = self.du(base_date)
        dc_n = self.dc(base_date)
        t_n = du_n / 252.0
        r_cont = self._resolve_cont_rate(pre_rate_or_curve, du_n, dc_n, default_conv="exp_252")
        q_cont = self._resolve_cont_rate(carry_or_div_yield, du_n, dc_n, default_conv="exp_252")
        fwd_est = float(spot) * math.exp((r_cont - q_cont) * t_n)
        vol_val = self._resolve_vol(vol, du_n, fwd_est)
        return float(
            _cy_corrado_su(
                float(spot),
                self.strike,
                r_cont,
                q_cont,
                t_n,
                vol_val,
                float(kappa3),
                float(kappa4),
                self.option_type == "CALL",
            )
        )


def price_options_portfolio(
    positions: List[Dict[str, Any]],
    base_date: Union[datetime.date, str],
    crr_steps: int = 150,
) -> pd.DataFrame:
    """
    Precifica em lote carteiras mistas de opções (`EquityOption`, `FXOption`, `IDIOption`, `DI1Option`
    e `CorradoSuOption`) via núcleos Cython (`nogil`), retornando prêmio unitário, MtM e gregas.
    """
    if not positions:
        return pd.DataFrame()
    if _cy_price_batch is None or _cy_price_b76_batch is None:
        raise RuntimeError("Extensão Cython 'options_core' não compilada.")

    n = len(positions)
    bsm_idx: List[int] = []
    b76_idx: List[int] = []
    cs_idx: List[int] = []

    tickers = []
    classes = []
    opt_types = []
    exercises = []
    qty_arr = np.empty(n, dtype=np.float64)

    p_out = np.empty(n, dtype=np.float64)
    d_out = np.empty(n, dtype=np.float64)
    g_out = np.empty(n, dtype=np.float64)
    v_out = np.empty(n, dtype=np.float64)
    t_out = np.empty(n, dtype=np.float64)
    r_out = np.empty(n, dtype=np.float64)

    bsm_params: List[tuple] = []
    b76_params: List[tuple] = []

    for i, pos in enumerate(positions):
        opt = pos["instrument"]
        tickers.append(opt.ticker)
        classes.append(opt.__class__.__name__)
        opt_types.append(opt.option_type)
        exercises.append(opt.exercise)
        qty_arr[i] = float(pos.get("quantity", 1.0)) * float(opt.multiplier)

        if isinstance(opt, FXOption):
            spot_ptax = float(pos["spot_ptax"])
            du_n, _, t_n, r_c, q_c, sig_eff = opt._resolve_fx_params(
                base_date,
                pos["pre_rate_or_curve"],
                pos["cupom_limpo_or_curve"],
                pos["vol"],
                fixing_vol_adjust=bool(pos.get("fixing_vol_adjust", False)),
                spot_ptax=spot_ptax,
            )
            bsm_idx.append(i)
            bsm_params.append((1 if opt.option_type == "CALL" else 0, 0, spot_ptax, opt.strike, r_c, q_c, t_n, sig_eff))
        elif isinstance(opt, IDIOption):
            du_n = opt.du(base_date)
            dc_n = opt.dc(base_date)
            t_n = du_n / 252.0
            r_c = opt._resolve_cont_rate(pos["pre_rate_or_curve"], du_n, dc_n, default_conv="exp_252")
            fwd = float(pos["idi_spot"]) * math.exp(r_c * t_n)
            sig = opt._resolve_vol(pos["vol"], du_n, fwd)
            b76_idx.append(i)
            b76_params.append((1 if opt.option_type == "CALL" else 0, fwd, opt.strike, r_c, t_n, sig, 1.0))
        elif isinstance(opt, DI1Option):
            t_c_dc, s_prime, k_prime, delta_ann = opt._resolve_di1_params(base_date, pos["pu_c"], pos["pu_l"])
            du_n = opt.du(base_date)
            sig = opt._resolve_vol(pos["vol"], du_n, s_prime)
            b76_idx.append(i)
            b76_params.append((1 if opt.option_type == "CALL" else 0, s_prime, k_prime, 0.0, t_c_dc, sig, delta_ann))
        elif isinstance(opt, CorradoSuOption):
            cs_idx.append(i)
            p_cs = opt.price(
                base_date,
                pos["spot"],
                pos["pre_rate_or_curve"],
                pos["vol"],
                kappa3=pos.get("kappa3", 0.0),
                kappa4=pos.get("kappa4", 3.0),
                carry_or_div_yield=pos.get("carry_or_div_yield", 0.0),
            )
            p_out[i] = p_cs
            d_out[i] = np.nan
            g_out[i] = np.nan
            v_out[i] = np.nan
            t_out[i] = np.nan
            r_out[i] = np.nan
        elif isinstance(opt, EquityOption):
            du_n = opt.du(base_date)
            dc_n = opt.dc(base_date)
            t_n = du_n / 252.0
            r_c = opt._resolve_cont_rate(pos["pre_rate_or_curve"], du_n, dc_n, default_conv="exp_252")
            q_c = opt._resolve_cont_rate(pos.get("carry_or_div_yield", 0.0), du_n, dc_n, default_conv="exp_252")
            S_val = float(pos["spot"])
            fwd_est = S_val * math.exp((r_c - q_c) * t_n)
            sig = opt._resolve_vol(pos["vol"], du_n, fwd_est)
            bsm_idx.append(i)
            bsm_params.append((1 if opt.option_type == "CALL" else 0, 1 if opt.exercise == "AMERICAN" else 0, S_val, opt.strike, r_c, q_c, t_n, sig))
        else:
            raise TypeError(f"Instrumento não suportado em price_options_portfolio: {type(opt)}")

    if bsm_idx:
        ic_a = np.array([x[0] for x in bsm_params], dtype=np.int32)
        ia_a = np.array([x[1] for x in bsm_params], dtype=np.int32)
        S_a = np.array([x[2] for x in bsm_params], dtype=np.float64)
        K_a = np.array([x[3] for x in bsm_params], dtype=np.float64)
        r_a = np.array([x[4] for x in bsm_params], dtype=np.float64)
        q_a = np.array([x[5] for x in bsm_params], dtype=np.float64)
        T_a = np.array([x[6] for x in bsm_params], dtype=np.float64)
        sg_a = np.array([x[7] for x in bsm_params], dtype=np.float64)
        pb, db, gb, vb, tb, rb = _cy_price_batch(ic_a, ia_a, S_a, K_a, r_a, q_a, T_a, sg_a, int(crr_steps))
        p_out[bsm_idx] = pb
        d_out[bsm_idx] = db
        g_out[bsm_idx] = gb
        v_out[bsm_idx] = vb
        t_out[bsm_idx] = tb
        r_out[bsm_idx] = rb

    if b76_idx:
        ic_a = np.array([x[0] for x in b76_params], dtype=np.int32)
        F_a = np.array([x[1] for x in b76_params], dtype=np.float64)
        K_a = np.array([x[2] for x in b76_params], dtype=np.float64)
        r_a = np.array([x[3] for x in b76_params], dtype=np.float64)
        T_a = np.array([x[4] for x in b76_params], dtype=np.float64)
        sg_a = np.array([x[5] for x in b76_params], dtype=np.float64)
        sc_a = np.array([x[6] for x in b76_params], dtype=np.float64)
        pb, db, gb, vb, tb, rb = _cy_price_b76_batch(ic_a, F_a, K_a, r_a, T_a, sg_a, sc_a)
        p_out[b76_idx] = pb
        d_out[b76_idx] = db
        g_out[b76_idx] = gb
        v_out[b76_idx] = vb
        t_out[b76_idx] = tb
        r_out[b76_idx] = rb

    return pd.DataFrame({
        "ticker": tickers,
        "classe": classes,
        "option_type": opt_types,
        "exercise": exercises,
        "quantity": qty_arr,
        "price": p_out,
        "mtm": p_out * qty_arr,
        "delta": d_out,
        "gamma": g_out,
        "vega_1pct": v_out * 0.01,
        "theta_du": t_out / 252.0,
        "rho": r_out,
    })


def price_fx_options_portfolio(positions: List[Dict[str, Any]], base_date: Union[datetime.date, str]) -> pd.DataFrame:
    """Atalho vetorizado em Cython para carteiras de `FXOption`."""
    return price_options_portfolio(positions, base_date)


def price_idi_options_portfolio(positions: List[Dict[str, Any]], base_date: Union[datetime.date, str]) -> pd.DataFrame:
    """Atalho vetorizado em Cython para carteiras de `IDIOption`."""
    return price_options_portfolio(positions, base_date)


def price_di1_options_portfolio(positions: List[Dict[str, Any]], base_date: Union[datetime.date, str]) -> pd.DataFrame:
    """Atalho vetorizado em Cython para carteiras de `DI1Option`."""
    return price_options_portfolio(positions, base_date)


class Swaption(_BaseOption):
    """
    Swaption europeia sobre taxa de swap DI x Pré (modelos Black-76 lognormal ou Normal de Bachelier).
    Suporta convenção periódica (`swap_convention='periodic'`) ou swap zero-cupom composto B3 em base 252
    (`swap_convention='b3_zero'`), além de parâmetros diretos (`forward_swap_rate`, `annuity`) em modo offline.
    """

    def __init__(
        self,
        ticker: str,
        expiry_date: Union[datetime.date, str],
        swap_maturity_date: Optional[Union[datetime.date, str]] = None,
        strike_rate: float = 0.13,
        option_type: str = "PAYER",
        notional: float = 1_000_000.0,
        payment_dates: Optional[List[Union[datetime.date, str]]] = None,
        swap_convention: str = "periodic",
        calendar_name: str = "B3",
        holidays: Any = None,
    ):
        opt_norm = option_type.strip().upper()
        if opt_norm in ("PAYER", "CALL"):
            mapped = "CALL"
        elif opt_norm in ("RECEIVER", "PUT"):
            mapped = "PUT"
        else:
            raise ValueError(f"option_type inválido para Swaption: {option_type}. Use 'PAYER'/'CALL' ou 'RECEIVER'/'PUT'.")
        conv_norm = swap_convention.strip().lower()
        if conv_norm not in ("periodic", "b3_zero"):
            raise ValueError(f"swap_convention inválido: {swap_convention}. Use 'periodic' ou 'b3_zero'.")
        super().__init__(
            ticker=ticker,
            maturity_date=expiry_date,
            strike=strike_rate,
            option_type=mapped,
            exercise="EUROPEAN",
            calendar_name=calendar_name,
            holidays=holidays,
        )
        self.swaption_side = "PAYER" if mapped == "CALL" else "RECEIVER"
        self.swap_maturity_date = self._normalize_date(swap_maturity_date) if swap_maturity_date else None
        self.notional = float(notional)
        self.payment_dates = [self._normalize_date(d) for d in payment_dates] if payment_dates else None
        self.swap_convention = conv_norm

    def _resolve_swap_params(
        self,
        base_date: Union[datetime.date, str],
        pre_rate_or_curve: Union[float, Curve],
        forward_swap_rate: Optional[float] = None,
        annuity: Optional[float] = None,
    ):
        base_dt = self._normalize_date(base_date)
        du_exp = self.du(base_dt)
        t_exp = du_exp / 252.0

        if forward_swap_rate is not None and annuity is not None:
            return du_exp, t_exp, float(forward_swap_rate), float(annuity)

        if self.payment_dates:
            pay_dates = sorted(d for d in self.payment_dates if d > self.maturity_date)
        elif self.swap_maturity_date is not None:
            pay_dates = [self.swap_maturity_date]
        else:
            raise ValueError("Informe 'swap_maturity_date', 'payment_dates' ou ('forward_swap_rate', 'annuity').")

        if not pay_dates:
            raise ValueError("Não há datas de pagamento do swap posteriores ao vencimento da swaption.")

        r_exp = self._get_rate_for_du(pre_rate_or_curve, max(1, du_exp))
        df_exp = (1.0 + r_exp) ** (-du_exp / 252.0) if du_exp > 0 else 1.0

        if self.swap_convention == "b3_zero":
            du_end = max(du_exp + 1, int(self.calendar.bizdays(base_dt, pay_dates[-1])))
            r_end = self._get_rate_for_du(pre_rate_or_curve, du_end)
            df_end = (1.0 + r_end) ** (-du_end / 252.0)
            tau_swap = max(1e-12, (du_end - du_exp) / 252.0)
            fwd_calc = math.pow(df_exp / max(1e-15, df_end), 1.0 / tau_swap) - 1.0
            fwd_rate = float(forward_swap_rate) if forward_swap_rate is not None else fwd_calc
            ann_calc = df_exp * tau_swap / max(1e-8, 1.0 + fwd_rate)
            ann_eff = float(annuity) if annuity is not None else ann_calc
            return du_exp, t_exp, fwd_rate, ann_eff

        prev_du = du_exp
        ann_val = 0.0
        for pdt in pay_dates:
            du_i = max(prev_du + 1, int(self.calendar.bizdays(base_dt, pdt)))
            tau_i = (du_i - prev_du) / 252.0
            r_i = self._get_rate_for_du(pre_rate_or_curve, du_i)
            df_i = (1.0 + r_i) ** (-du_i / 252.0)
            ann_val += tau_i * df_i
            prev_du = du_i

        r_end = self._get_rate_for_du(pre_rate_or_curve, prev_du)
        df_end = (1.0 + r_end) ** (-prev_du / 252.0)

        fwd_rate = float(forward_swap_rate) if forward_swap_rate is not None else ((df_exp - df_end) / max(1e-12, ann_val))
        ann_eff = float(annuity) if annuity is not None else ann_val
        return du_exp, t_exp, fwd_rate, ann_eff

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        pre_rate_or_curve: Union[float, Curve],
        vol: Any,
        model: str = "black76",
        forward_swap_rate: Optional[float] = None,
        annuity: Optional[float] = None,
    ) -> Dict[str, float]:
        du_exp, t_exp, fwd_rate, ann_eff = self._resolve_swap_params(
            base_date, pre_rate_or_curve, forward_swap_rate=forward_swap_rate, annuity=annuity
        )
        vol_val = self._resolve_vol(vol, du_exp, fwd_rate)
        is_call = self.option_type == "CALL"
        scale = self.notional * ann_eff
        mdl = model.strip().lower()

        if mdl == "bachelier":
            p_u, d_u, g_u, v_u, th_u = _cy_bachelier(fwd_rate, self.strike, 1.0, t_exp, vol_val, is_call)
            return {
                "du": float(du_exp),
                "T": t_exp,
                "forward_swap_rate": fwd_rate,
                "annuity": ann_eff,
                "price": float(scale * p_u),
                "delta": float(scale * d_u),
                "gamma": float(scale * g_u),
                "vega": float(scale * v_u),
                "vega_1bp": float(scale * v_u * 1e-4),
                "theta": float(scale * th_u),
            }
        if mdl == "black76":
            p_u, d_u, g_u, v_u, th_u, _, _, _ = _cy_black76(fwd_rate, self.strike, 0.0, t_exp, vol_val, is_call)
            return {
                "du": float(du_exp),
                "T": t_exp,
                "forward_swap_rate": fwd_rate,
                "annuity": ann_eff,
                "price": float(scale * p_u),
                "delta": float(scale * d_u),
                "gamma": float(scale * g_u),
                "vega": float(scale * v_u),
                "vega_1pct": float(scale * v_u * 0.01),
                "theta": float(scale * th_u),
            }
        raise ValueError(f"Modelo inválido para Swaption: {model}. Use 'black76' ou 'bachelier'.")

    def price(
        self,
        base_date: Union[datetime.date, str],
        pre_rate_or_curve: Union[float, Curve],
        vol: Any,
        model: str = "black76",
        forward_swap_rate: Optional[float] = None,
        annuity: Optional[float] = None,
    ) -> float:
        return self.risk_metrics(base_date, pre_rate_or_curve, vol, model, forward_swap_rate, annuity)["price"]

    def implied_volatility(
        self,
        base_date: Union[datetime.date, str],
        market_price: float,
        pre_rate_or_curve: Union[float, Curve],
        model: str = "black76",
        forward_swap_rate: Optional[float] = None,
        annuity: Optional[float] = None,
    ) -> float:
        _, t_exp, fwd_rate, ann_eff = self._resolve_swap_params(
            base_date, pre_rate_or_curve, forward_swap_rate=forward_swap_rate, annuity=annuity
        )
        target_u = float(market_price) / max(1e-12, self.notional * ann_eff)
        is_call = self.option_type == "CALL"
        mdl = model.strip().lower()
        if mdl == "bachelier":
            return float(_cy_iv_bachelier(target_u, fwd_rate, self.strike, 1.0, t_exp, is_call))
        if mdl == "black76":
            return float(_cy_iv_black76(target_u, fwd_rate, self.strike, 0.0, t_exp, is_call))
        raise ValueError(f"Modelo inválido para Swaption: {model}.")


class CapFloor(_BaseOption):
    """
    Cap ou Floor de taxa de juros (cesta de caplets/floorlets sobre DI1 / FRA),
    precificado pelos modelos Black-76 lognormal ou Normal de Bachelier.
    """

    def __init__(
        self,
        ticker: str,
        fixing_dates: List[Union[datetime.date, str]],
        payment_dates: List[Union[datetime.date, str]],
        strike_rate: float,
        option_type: str = "CAP",
        notional: float = 1_000_000.0,
        calendar_name: str = "B3",
        holidays: Any = None,
    ):
        if len(fixing_dates) != len(payment_dates) or not fixing_dates:
            raise ValueError("fixing_dates e payment_dates devem ser não vazios e de mesmo tamanho.")
        opt_norm = option_type.strip().upper()
        if opt_norm in ("CAP", "CALL"):
            mapped = "CALL"
        elif opt_norm in ("FLOOR", "PUT"):
            mapped = "PUT"
        else:
            raise ValueError(f"option_type inválido para CapFloor: {option_type}. Use 'CAP' ou 'FLOOR'.")
        self.fixing_dates = [self._normalize_date(d) for d in fixing_dates]
        self.payment_dates = [self._normalize_date(d) for d in payment_dates]
        super().__init__(
            ticker=ticker,
            maturity_date=max(self.payment_dates),
            strike=strike_rate,
            option_type=mapped,
            exercise="EUROPEAN",
            calendar_name=calendar_name,
            holidays=holidays,
        )
        self.capfloor_type = "CAP" if mapped == "CALL" else "FLOOR"
        self.notional = float(notional)

    def cashflow_table(
        self,
        base_date: Union[datetime.date, str],
        pre_rate_or_curve: Union[float, Curve],
        vol: Any,
        model: str = "black76",
        forward_rates: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        base_dt = self._normalize_date(base_date)
        is_call = self.option_type == "CALL"
        mdl = model.strip().lower()
        rows = []
        for idx, (f_dt, p_dt) in enumerate(zip(self.fixing_dates, self.payment_dates)):
            if p_dt <= base_dt:
                continue
            du_fix = max(0, int(self.calendar.bizdays(base_dt, f_dt)))
            du_pay = max(du_fix + 1, int(self.calendar.bizdays(base_dt, p_dt)))
            t_fix = du_fix / 252.0
            tau = (du_pay - du_fix) / 252.0

            r_fix = self._get_rate_for_du(pre_rate_or_curve, max(1, du_fix))
            df_fix = (1.0 + r_fix) ** (-du_fix / 252.0) if du_fix > 0 else 1.0
            r_pay = self._get_rate_for_du(pre_rate_or_curve, du_pay)
            df_pay = (1.0 + r_pay) ** (-du_pay / 252.0)

            fwd_i = float(forward_rates[idx]) if forward_rates is not None else (((df_fix / df_pay) - 1.0) / max(1e-12, tau))
            vol_i = self._resolve_vol(vol, max(1, du_fix), fwd_i)

            if mdl == "bachelier":
                p_u, d_u, _, v_u, _ = _cy_bachelier(fwd_i, self.strike, df_pay, t_fix, vol_i, is_call)
            elif mdl == "black76":
                p_u, d_u, _, v_u, _, _, _, _ = _cy_black76(fwd_i, self.strike, 0.0, t_fix, vol_i, is_call)
                p_u *= df_pay
                d_u *= df_pay
                v_u *= df_pay
            else:
                raise ValueError(f"Modelo inválido para CapFloor: {model}. Use 'black76' ou 'bachelier'.")

            scale = self.notional * tau
            rows.append({
                "fixing_date": f_dt,
                "payment_date": p_dt,
                "du_fix": du_fix,
                "du_pay": du_pay,
                "tau": tau,
                "forward_rate": fwd_i,
                "df_pay": df_pay,
                "vol": vol_i,
                "caplet_pv": float(scale * p_u),
                "delta": float(scale * d_u),
                "vega": float(scale * v_u),
            })
        return pd.DataFrame(rows)

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        pre_rate_or_curve: Union[float, Curve],
        vol: Any,
        model: str = "black76",
        forward_rates: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        df = self.cashflow_table(base_date, pre_rate_or_curve, vol, model=model, forward_rates=forward_rates)
        if df.empty:
            return {"price": 0.0, "delta": 0.0, "vega": 0.0, "caplets": 0}
        return {
            "price": float(df["caplet_pv"].sum()),
            "delta": float(df["delta"].sum()),
            "vega": float(df["vega"].sum()),
            "caplets": int(len(df)),
        }

    def price(
        self,
        base_date: Union[datetime.date, str],
        pre_rate_or_curve: Union[float, Curve],
        vol: Any,
        model: str = "black76",
        forward_rates: Optional[List[float]] = None,
    ) -> float:
        return self.risk_metrics(base_date, pre_rate_or_curve, vol, model=model, forward_rates=forward_rates)["price"]


class BarrierOption(_BaseOption):
    """
    Opção europeia com barreira (Reiner & Rubinstein, 1991):
    `down-and-out`, `down-and-in`, `up-and-out` e `up-and-in`.
    Quando `monitoring_steps` é informado (> 0), aplica a correção de continuidade
    de Broadie, Glasserman & Kou (1997) para monitoramento discreto.
    """

    def __init__(
        self,
        ticker: str,
        maturity_date: Union[datetime.date, str],
        strike: float,
        barrier: float,
        option_type: str = "CALL",
        barrier_type: str = "down-and-out",
        rebate: float = 0.0,
        monitoring_steps: Optional[int] = None,
        calendar_name: str = "B3",
        holidays: Any = None,
    ):
        super().__init__(
            ticker=ticker,
            maturity_date=maturity_date,
            strike=strike,
            option_type=option_type,
            exercise="EUROPEAN",
            calendar_name=calendar_name,
            holidays=holidays,
        )
        bt = barrier_type.strip().lower()
        if bt not in ("down-and-out", "down-and-in", "up-and-out", "up-and-in"):
            raise ValueError(f"barrier_type inválido: {barrier_type}")
        self.barrier = float(barrier)
        self.barrier_type = bt
        self.is_up = bt.startswith("up")
        self.is_out = bt.endswith("out")
        self.rebate = float(rebate)
        self.monitoring_steps = int(monitoring_steps) if monitoring_steps is not None else None

    def price(
        self,
        base_date: Union[datetime.date, str],
        spot: float,
        pre_rate_or_curve: Union[float, Curve],
        vol: Any,
        carry_or_div_yield: Union[float, Curve] = 0.0,
    ) -> float:
        du_n = self.du(base_date)
        dc_n = self.dc(base_date)
        t_n = du_n / 252.0
        r_cont = self._resolve_cont_rate(pre_rate_or_curve, du_n, dc_n, default_conv="exp_252")
        q_cont = self._resolve_cont_rate(carry_or_div_yield, du_n, dc_n, default_conv="exp_252")
        fwd_est = float(spot) * math.exp((r_cont - q_cont) * t_n)
        vol_val = self._resolve_vol(vol, du_n, fwd_est)

        h_eff = self.barrier
        if self.monitoring_steps is not None and self.monitoring_steps > 0 and t_n > 0.0:
            spot_f = float(spot)
            if (self.is_up and spot_f < self.barrier) or ((not self.is_up) and spot_f > self.barrier):
                beta_bgk = 0.5825971579
                dt_mon = t_n / float(self.monitoring_steps)
                shift = beta_bgk * vol_val * math.sqrt(dt_mon)
                h_eff = self.barrier * math.exp(shift if self.is_up else -shift)

        return float(
            _cy_barrier_bsm(
                float(spot),
                self.strike,
                h_eff,
                r_cont,
                q_cont,
                t_n,
                vol_val,
                self.option_type == "CALL",
                self.is_up,
                self.is_out,
                self.rebate,
            )
        )


class AsianOption(_BaseOption):
    """
    Opção asiática europeia de média recém-iniciada (contínua ou discreta em `n_fixings` datas):
    - `average_type='geometric'`: fórmula fechada de Kemna & Vorst (1990) / média geométrica discreta.
    - `average_type='arithmetic'`: aproximação lognormal de dois momentos de Levy (1992) / Turnbull & Wakeman (1991).
    """

    def __init__(
        self,
        ticker: str,
        maturity_date: Union[datetime.date, str],
        strike: float,
        option_type: str = "CALL",
        average_type: str = "arithmetic",
        n_fixings: Optional[int] = None,
        calendar_name: str = "B3",
        holidays: Any = None,
    ):
        super().__init__(
            ticker=ticker,
            maturity_date=maturity_date,
            strike=strike,
            option_type=option_type,
            exercise="EUROPEAN",
            calendar_name=calendar_name,
            holidays=holidays,
        )
        avg = average_type.strip().lower()
        if avg not in ("arithmetic", "geometric"):
            raise ValueError("average_type deve ser 'arithmetic' ou 'geometric'.")
        self.average_type = avg
        self.n_fixings = int(n_fixings) if n_fixings is not None else None

    def price(
        self,
        base_date: Union[datetime.date, str],
        spot: float,
        pre_rate_or_curve: Union[float, Curve],
        vol: Any,
        carry_or_div_yield: Union[float, Curve] = 0.0,
    ) -> float:
        du_n = self.du(base_date)
        dc_n = self.dc(base_date)
        t_n = du_n / 252.0
        r_cont = self._resolve_cont_rate(pre_rate_or_curve, du_n, dc_n, default_conv="exp_252")
        q_cont = self._resolve_cont_rate(carry_or_div_yield, du_n, dc_n, default_conv="exp_252")
        fwd_est = float(spot) * math.exp((r_cont - q_cont) * t_n)
        vol_val = self._resolve_vol(vol, du_n, fwd_est)
        n_fix = self.n_fixings if (self.n_fixings is not None and self.n_fixings > 0) else 0
        fn = _cy_asian_geom if self.average_type == "geometric" else _cy_asian_arith
        return float(fn(float(spot), self.strike, r_cont, q_cont, t_n, vol_val, self.option_type == "CALL", n_fix))

