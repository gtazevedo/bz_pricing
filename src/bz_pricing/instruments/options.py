import datetime
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from bz_pricing.core.calendars import get_calendar
from bz_pricing.core.curves import Curve

try:
    from bz_pricing.core.options_core import (
        black76_price_greeks as _cy_black76,
        bsm_price_greeks as _cy_bsm,
        corrado_su_price as _cy_corrado_su,
        crr_american_price as _cy_crr_american,
        implied_vol_black76 as _cy_iv_black76,
        implied_vol_bsm as _cy_iv_bsm,
        price_options_batch as _cy_price_batch,
    )
except ImportError:
    _cy_bsm = None
    _cy_black76 = None
    _cy_crr_american = None
    _cy_corrado_su = None
    _cy_iv_bsm = None
    _cy_iv_black76 = None
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
        self.calendar = get_calendar(calendar_name)
        self.multiplier = float(multiplier)

    @staticmethod
    def _normalize_date(d: Union[datetime.date, str]) -> datetime.date:
        if isinstance(d, datetime.datetime):
            return d.date()
        if isinstance(d, datetime.date):
            return d
        return datetime.date.fromisoformat(str(d).strip()[:10])

    @staticmethod
    def _get_rate_for_du(rate_or_curve: Union[float, Curve], du: int) -> float:
        if isinstance(rate_or_curve, Curve):
            return float(rate_or_curve.get_rate(max(1, int(du))))
        return float(rate_or_curve)

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
        t_n = du_n / 252.0
        r_eff = self._get_rate_for_du(pre_rate_or_curve, du_n)
        q_eff = self._get_rate_for_du(carry_or_div_yield, du_n)
        r_cont = math.log(1.0 + max(-0.99, r_eff))
        q_cont = math.log(1.0 + max(-0.99, q_eff))
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

        price_eu, delta, gamma, vega, theta, rho, vanna, volga = _cy_bsm(
            float(spot), self.strike, r_cont, q_cont, t_n, float(vol), is_call
        )
        if self.exercise == "AMERICAN":
            price = _cy_crr_american(
                float(spot), self.strike, r_cont, q_cont, t_n, float(vol), is_call, int(crr_steps)
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
        vol: float,
        fixing_vol_adjust: bool = False,
    ):
        du_n = self.du(base_date)
        dc_n = self.dc(base_date)
        t_n = du_n / 252.0
        r_pre = self._get_rate_for_du(pre_rate_or_curve, du_n)
        r_ddi = self._get_rate_for_du(cupom_limpo_or_curve, du_n)
        r_cont = math.log(1.0 + max(-0.99, r_pre))
        if du_n > 0:
            q_cont = (252.0 / du_n) * math.log(max(1e-8, 1.0 + r_ddi * (dc_n / 360.0)))
            sigma_eff = float(vol) * math.sqrt(du_n / (du_n + 1.0)) if fixing_vol_adjust else float(vol)
        else:
            q_cont = 0.0
            sigma_eff = float(vol)
        return du_n, dc_n, t_n, r_cont, q_cont, sigma_eff

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        spot_ptax: float,
        pre_rate_or_curve: Union[float, Curve],
        cupom_limpo_or_curve: Union[float, Curve],
        vol: float,
        fixing_vol_adjust: bool = False,
    ) -> Dict[str, float]:
        du_n, dc_n, t_n, r_cont, q_cont, sigma_eff = self._resolve_fx_params(
            base_date, pre_rate_or_curve, cupom_limpo_or_curve, vol, fixing_vol_adjust
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
    ) -> float:
        _, _, t_n, r_cont, q_cont, _ = self._resolve_fx_params(
            base_date, pre_rate_or_curve, cupom_limpo_or_curve, 0.15, False
        )
        return float(
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
        t_n = du_n / 252.0
        r_eff = self._get_rate_for_du(pre_rate_or_curve, du_n)
        r_cont = math.log(1.0 + max(-0.99, r_eff))
        idi_fwd = float(idi_spot) * math.exp(r_cont * t_n)

        price, delta, gamma, vega, theta, rho, vanna, volga = _cy_black76(
            idi_fwd, self.strike, r_cont, t_n, float(vol), self.option_type == "CALL"
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
        t_n = du_n / 252.0
        r_eff = self._get_rate_for_du(pre_rate_or_curve, du_n)
        r_cont = math.log(1.0 + max(-0.99, r_eff))
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
    ):
        super().__init__(
            ticker=ticker,
            maturity_date=maturity_date,
            strike=strike_rate,
            option_type=option_type,
            exercise="EUROPEAN",
            calendar_name=calendar_name,
        )
        self.underlying_maturity_date = self._normalize_date(underlying_maturity_date)
        self.dc_basis = float(dc_basis)

    def _resolve_di1_params(
        self,
        base_date: Union[datetime.date, str],
        pu_c: float,
        pu_l: float,
    ):
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
        undisc_target = float(market_price) / delta_annuity
        return float(
            _cy_iv_black76(undisc_target, s_prime, k_prime, 0.0, t_c_dc, self.option_type == "CALL")
        )


class CorradoSuOption(_BaseOption):
    """
    Opção precificada pelo modelo de Corrado & Su (1996) com expansão de Gram-Charlier
    para assimetria (kappa3) e curtose (kappa4), aplicado pela B3 a séries ilíquidas.
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
        t_n = du_n / 252.0
        r_cont = math.log(1.0 + max(-0.99, self._get_rate_for_du(pre_rate_or_curve, du_n)))
        q_cont = math.log(1.0 + max(-0.99, self._get_rate_for_du(carry_or_div_yield, du_n)))
        return float(
            _cy_corrado_su(
                float(spot),
                self.strike,
                r_cont,
                q_cont,
                t_n,
                float(vol),
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
    Precifica em lote uma carteira de `EquityOption` (europeias e americanas) via Cython (`nogil`),
    retornando prêmio unitário, MtM financeiro e gregas por posição.
    """
    if not positions:
        return pd.DataFrame()
    if _cy_price_batch is None:
        raise RuntimeError("Extensão Cython 'options_core' não compilada.")

    n = len(positions)
    is_call_arr = np.empty(n, dtype=np.int32)
    is_amer_arr = np.empty(n, dtype=np.int32)
    S_arr = np.empty(n, dtype=np.float64)
    K_arr = np.empty(n, dtype=np.float64)
    r_arr = np.empty(n, dtype=np.float64)
    q_arr = np.empty(n, dtype=np.float64)
    T_arr = np.empty(n, dtype=np.float64)
    sig_arr = np.empty(n, dtype=np.float64)
    qty_arr = np.empty(n, dtype=np.float64)
    tickers = []
    opt_types = []
    exercises = []

    for i, pos in enumerate(positions):
        opt = pos["instrument"]
        tickers.append(opt.ticker)
        opt_types.append(opt.option_type)
        exercises.append(opt.exercise)
        is_call_arr[i] = 1 if opt.option_type == "CALL" else 0
        is_amer_arr[i] = 1 if opt.exercise == "AMERICAN" else 0
        S_arr[i] = float(pos["spot"])
        K_arr[i] = float(opt.strike)
        du_n = opt.du(base_date)
        T_arr[i] = du_n / 252.0
        r_eff = opt._get_rate_for_du(pos["pre_rate_or_curve"], du_n)
        q_eff = opt._get_rate_for_du(pos.get("carry_or_div_yield", 0.0), du_n)
        r_arr[i] = math.log(1.0 + max(-0.99, r_eff))
        q_arr[i] = math.log(1.0 + max(-0.99, q_eff))
        sig_arr[i] = float(pos["vol"])
        qty_arr[i] = float(pos.get("quantity", 1.0)) * float(opt.multiplier)

    p_arr, d_arr, g_arr, v_arr, t_arr, rho_arr = _cy_price_batch(
        is_call_arr, is_amer_arr, S_arr, K_arr, r_arr, q_arr, T_arr, sig_arr, int(crr_steps)
    )

    return pd.DataFrame({
        "ticker": tickers,
        "option_type": opt_types,
        "exercise": exercises,
        "quantity": qty_arr,
        "price": p_arr,
        "mtm": p_arr * qty_arr,
        "delta": d_arr,
        "gamma": g_arr,
        "vega_1pct": v_arr * 0.01,
        "theta_du": t_arr / 252.0,
        "rho": rho_arr,
    })

