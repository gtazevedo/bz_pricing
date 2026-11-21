import datetime
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from bz_pricing.core.calendars import get_calendar
from bz_pricing.core.curves import Curve
from bz_pricing.core.derivatives_core import (
    dv01_dap,
    dv01_ddi,
    dv01_di1,
    fair_dol,
    fair_ind,
    future_daily_settlement,
    ndf_mtm,
    price_di1_batch,
    pu_dap,
    pu_ddi,
    pu_di1,
    rate_from_pu_di1,
    swap_pre_di_mtm,
)
from bz_pricing.data.loaders import B3DataLoader
from bz_pricing.instruments.offshore_bonds import (
    _ShiftedCurve,
    _generate_schedule_dates,
    _normalize_day_count,
    _resolve_discount_factor,
    _year_fraction_bond,
)


def _to_date(d: Union[datetime.date, str]) -> datetime.date:
    if isinstance(d, datetime.date) and not isinstance(d, datetime.datetime):
        return d
    return pd.to_datetime(d, dayfirst="/" in str(d)).date()


def _resolve_rate(rate_or_curve: Union[float, Curve], du: int, dc: int = 0) -> float:
    if isinstance(rate_or_curve, Curve):
        return float(rate_or_curve.get_rate(du, dc))
    return float(rate_or_curve)


class _BaseFuture:
    """Classe base para contratos futuros listados na B3."""

    ASSET_CODE = ""
    MULTIPLIER = 1.0

    def __init__(
        self,
        ticker: str,
        maturity_date: Union[datetime.date, str],
        calendar_name: str = "B3",
        contract_multiplier: Optional[float] = None,
        holidays: Any = None,
    ):
        self.ticker = ticker.strip().upper()
        self.maturity_date = _to_date(maturity_date)
        self.calendar = get_calendar(calendar_name, holidays=holidays)
        if contract_multiplier is not None:
            self.MULTIPLIER = float(contract_multiplier)

    def _du(self, base_date: Union[datetime.date, str]) -> int:
        return max(0, int(self.calendar.bizdays(_to_date(base_date), self.maturity_date)))

    def _dc(self, base_date: Union[datetime.date, str]) -> int:
        return max(0, int((self.maturity_date - _to_date(base_date)).days))

    @classmethod
    def from_b3(
        cls,
        ticker: str,
        ref_date: Union[datetime.date, str],
        loader: Optional[B3DataLoader] = None,
        calendar_name: str = "B3",
    ) -> Dict:
        """
        Carrega os dados oficiais de fechamento/ajuste do contrato futuro na B3
        (arquivo SPRD / BVBG.187.01) e retorna instância e métricas oficiais.
        """
        dt = _to_date(ref_date)
        ldr = loader or B3DataLoader()
        asst = cls.ASSET_CODE or ticker[:3].upper()
        df = ldr.get_curve_instruments(dt, asst=asst)
        row = df[df["TckrSymb"].astype(str).str.upper() == ticker.strip().upper()]
        if row.empty:
            raise KeyError(f"Contrato '{ticker}' não encontrado no boletim B3 de {dt}.")
        rec = row.iloc[0]
        inst = cls(ticker=ticker, maturity_date=rec["XprtnDt"], calendar_name=calendar_name)
        return {
            "instrument": inst,
            "adjstd_qt": float(rec["AdjstdQt"]) if pd.notna(rec.get("AdjstdQt")) else None,
            "adjstd_qt_tax": (float(rec["AdjstdQtTax"]) / 100.0) if pd.notna(rec.get("AdjstdQtTax")) else None,
            "open_interest": float(rec["OpnIntrst"]) if pd.notna(rec.get("OpnIntrst")) else None,
        }

    def daily_settlement(
        self,
        pu_atual: float,
        pu_anterior: float,
        contracts: float = 1.0,
        direction: int = 1,
    ) -> float:
        """Calcula o ajuste diário B3 em moeda do contrato."""
        return float(
            future_daily_settlement(
                float(pu_atual),
                float(pu_anterior),
                float(self.MULTIPLIER),
                float(contracts),
                float(direction),
            )
        )


class DI1Future(_BaseFuture):
    """Contrato Futuro de Taxa Média de Depósitos Interfinanceiros de Um Dia (DI1 - B3)."""

    ASSET_CODE = "DI1"
    MULTIPLIER = 1.0

    def pu(self, base_date: Union[datetime.date, str], rate_or_curve: Union[float, Curve], round_2: bool = True) -> float:
        du = self._du(base_date)
        rate = _resolve_rate(rate_or_curve, du)
        return float(pu_di1(rate, du, round_2))

    def implied_rate(self, base_date: Union[datetime.date, str], pu_val: float) -> float:
        du = self._du(base_date)
        return float(rate_from_pu_di1(float(pu_val), du))

    def dv01(self, base_date: Union[datetime.date, str], rate_or_curve: Union[float, Curve], contracts: float = 1.0) -> float:
        du = self._du(base_date)
        rate = _resolve_rate(rate_or_curve, du)
        return float(dv01_di1(rate, du) * float(contracts))


class DDIFuture(_BaseFuture):
    """Contrato Futuro de Cupom Cambial (DDI - B3, base linear ACT/360 em USD)."""

    ASSET_CODE = "DDI"
    MULTIPLIER = 0.5  # USD 50.000 por contrato (face 100.000 pontos -> USD 0,50 por ponto)

    def pu(self, base_date: Union[datetime.date, str], rate_or_curve: Union[float, Curve], round_2: bool = True) -> float:
        du = self._du(base_date)
        dc = self._dc(base_date)
        rate = _resolve_rate(rate_or_curve, du, dc)
        return float(pu_ddi(rate, dc, round_2))

    def dv01(self, base_date: Union[datetime.date, str], rate_or_curve: Union[float, Curve], contracts: float = 1.0) -> float:
        du = self._du(base_date)
        dc = self._dc(base_date)
        rate = _resolve_rate(rate_or_curve, du, dc)
        return float(dv01_ddi(rate, dc) * self.MULTIPLIER * float(contracts))


class DAPFuture(_BaseFuture):
    """Contrato Futuro de Cupom de IPCA (DAP - B3, base exponencial 252 DU)."""

    ASSET_CODE = "DAP"
    MULTIPLIER = 0.00025  # R$ 0,00025 * VNA_IPCA por ponto (ou multiplicador em pontos de VNA)

    def pu(self, base_date: Union[datetime.date, str], rate_or_curve: Union[float, Curve], round_2: bool = True) -> float:
        du = self._du(base_date)
        rate = _resolve_rate(rate_or_curve, du)
        return float(pu_dap(rate, du, round_2))

    def dv01(self, base_date: Union[datetime.date, str], rate_or_curve: Union[float, Curve], contracts: float = 1.0) -> float:
        du = self._du(base_date)
        rate = _resolve_rate(rate_or_curve, du)
        return float(dv01_dap(rate, du) * float(contracts))

    def daily_settlement(
        self,
        pu_atual: float,
        pu_anterior: float,
        contracts: float = 1.0,
        direction: int = 1,
        vna_ntnb: float = 1.0,
    ) -> float:
        """Ajuste diário do DAP (em R$ quando `vna_ntnb` é informado, ou em pontos de VNA se `vna_ntnb=1.0`)."""
        return float(super().daily_settlement(pu_atual, pu_anterior, contracts=contracts, direction=direction) * float(vna_ntnb))


class DOLFuture(_BaseFuture):
    """Contrato Futuro de Taxa de Câmbio USD/BRL (DOL cheio: USD 50k; WDO mini: USD 10k)."""

    ASSET_CODE = "DOL"

    def __init__(
        self,
        ticker: str,
        maturity_date: Union[datetime.date, str],
        calendar_name: str = "B3",
        mini: Optional[bool] = None,
        contract_multiplier: Optional[float] = None,
    ):
        super().__init__(ticker, maturity_date, calendar_name=calendar_name)
        is_mini = (self.ticker.startswith("WDO")) if mini is None else bool(mini)
        self.MULTIPLIER = float(contract_multiplier) if contract_multiplier is not None else (10.0 if is_mini else 50.0)

    def fair_price(
        self,
        base_date: Union[datetime.date, str],
        spot_ptax: float,
        pre_rate_or_curve: Union[float, Curve],
        cupom_rate_or_curve: Union[float, Curve],
    ) -> float:
        du = self._du(base_date)
        dc = self._dc(base_date)
        r_pre = _resolve_rate(pre_rate_or_curve, du, dc)
        r_cup = _resolve_rate(cupom_rate_or_curve, du, dc)
        return float(fair_dol(float(spot_ptax), r_pre, du, r_cup, dc))


class INDFuture(_BaseFuture):
    """Contrato Futuro de Ibovespa (IND cheio: R$ 1,00/ponto; WIN mini: R$ 0,20/ponto)."""

    ASSET_CODE = "IND"

    def __init__(
        self,
        ticker: str,
        maturity_date: Union[datetime.date, str],
        calendar_name: str = "B3",
        mini: Optional[bool] = None,
        contract_multiplier: Optional[float] = None,
    ):
        super().__init__(ticker, maturity_date, calendar_name=calendar_name)
        is_mini = (self.ticker.startswith("WIN")) if mini is None else bool(mini)
        self.MULTIPLIER = float(contract_multiplier) if contract_multiplier is not None else (0.20 if is_mini else 1.00)

    def fair_price(
        self,
        base_date: Union[datetime.date, str],
        spot_ibov: float,
        pre_rate_or_curve: Union[float, Curve],
        div_yield: float = 0.0,
    ) -> float:
        du = self._du(base_date)
        r_pre = _resolve_rate(pre_rate_or_curve, du)
        return float(fair_ind(float(spot_ibov), r_pre, du, float(div_yield)))


class SwapDIxPRE:
    """
    Contrato de Swap DI x Pré (B3).
    `fixed_direction = 1`: ativo na ponta Pré (recebe Pré, paga DI).
    `fixed_direction = -1`: ativo na ponta DI (recebe DI, paga Pré).
    """

    def __init__(
        self,
        notional: float,
        start_date: Union[datetime.date, str],
        maturity_date: Union[datetime.date, str],
        fixed_rate: float,
        fixed_direction: int = 1,
        calendar_name: str = "B3",
        holidays: Any = None,
    ):
        self.notional = float(notional)
        self.start_date = _to_date(start_date)
        self.maturity_date = _to_date(maturity_date)
        self.fixed_rate = float(fixed_rate)
        self.fixed_direction = 1 if fixed_direction >= 0 else -1
        self.calendar = get_calendar(calendar_name, holidays=holidays)

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        pre_rate_or_curve: Union[float, Curve],
        di_accrued_factor: float = 1.0,
    ) -> Dict[str, float]:
        base_dt = _to_date(base_date)
        du_rem = max(0, int(self.calendar.bizdays(base_dt, self.maturity_date)))
        du_total = max(0, int(self.calendar.bizdays(self.start_date, self.maturity_date)))
        r_mkt = _resolve_rate(pre_rate_or_curve, du_rem)

        mtm, pv_pre, pv_di, dv01_pre = swap_pre_di_mtm(
            self.notional,
            self.fixed_rate,
            r_mkt,
            du_total,
            float(di_accrued_factor),
            float(self.fixed_direction),
            du_rem,
        )

        return {
            "mtm": float(mtm),
            "pv_pre": float(pv_pre),
            "pv_di": float(pv_di),
            "dv01": float(dv01_pre),
            "du_rem": du_rem,
        }

    def mtm(
        self,
        base_date: Union[datetime.date, str],
        pre_rate_or_curve: Union[float, Curve],
        di_accrued_factor: float = 1.0,
    ) -> float:
        return self.risk_metrics(base_date, pre_rate_or_curve, di_accrued_factor)["mtm"]


class SwapDIxIPCA:
    """Swap DI x IPCA (base 252 DU) com marcação a mercado pelo cupom de IPCA."""

    def __init__(
        self,
        notional: float,
        start_date: Union[datetime.date, str],
        maturity_date: Union[datetime.date, str],
        ipca_coupon_rate: float,
        ipca_direction: int = 1,
        calendar_name: str = "B3",
        holidays: Any = None,
    ):
        self.notional = float(notional)
        self.start_date = _to_date(start_date)
        self.maturity_date = _to_date(maturity_date)
        self.ipca_coupon_rate = float(ipca_coupon_rate)
        self.ipca_direction = 1 if ipca_direction >= 0 else -1
        self.calendar = get_calendar(calendar_name, holidays=holidays)

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        cupom_ipca_rate_or_curve: Union[float, Curve],
        ipca_accrued_factor: float = 1.0,
        di_accrued_factor: float = 1.0,
    ) -> Dict[str, float]:
        base_dt = _to_date(base_date)
        du_rem = max(0, int(self.calendar.bizdays(base_dt, self.maturity_date)))
        du_total = max(0, int(self.calendar.bizdays(self.start_date, self.maturity_date)))
        r_cup = _resolve_rate(cupom_ipca_rate_or_curve, du_rem)

        df_cup = math.pow(1.0 + r_cup, du_rem / 252.0) if du_rem > 0 else 1.0
        fv_ipca = self.notional * float(ipca_accrued_factor) * (math.pow(1.0 + self.ipca_coupon_rate, du_total / 252.0) if du_total > 0 else 1.0)
        pv_ipca = fv_ipca / df_cup
        pv_di = self.notional * float(di_accrued_factor)
        mtm = float(self.ipca_direction) * (pv_ipca - pv_di)

        df_up = math.pow(1.0 + r_cup + 0.0001, du_rem / 252.0) if du_rem > 0 else 1.0
        dv01 = float(self.ipca_direction) * ((fv_ipca / df_up) - pv_ipca)
        return {"mtm": float(mtm), "pv_ipca": float(pv_ipca), "pv_di": float(pv_di), "dv01": float(dv01), "du_rem": du_rem}


class NDFContract:
    """Contrato a Termo de Moeda Sem Entrega Física (NDF USD/BRL)."""

    def __init__(
        self,
        notional_usd: float,
        maturity_date: Union[datetime.date, str],
        strike_brl: float,
        direction: int = 1,
        calendar_name: str = "B3",
        holidays: Any = None,
    ):
        self.notional_usd = float(notional_usd)
        self.maturity_date = _to_date(maturity_date)
        self.strike_brl = float(strike_brl)
        self.direction = 1 if direction >= 0 else -1
        self.calendar = get_calendar(calendar_name, holidays=holidays)

    def mtm(
        self,
        base_date: Union[datetime.date, str],
        spot_ptax: float,
        pre_rate_or_curve: Union[float, Curve],
        cupom_rate_or_curve: Union[float, Curve],
    ) -> float:
        base_dt = _to_date(base_date)
        du = max(0, int(self.calendar.bizdays(base_dt, self.maturity_date)))
        dc = max(0, int((self.maturity_date - base_dt).days))
        r_pre = _resolve_rate(pre_rate_or_curve, du, dc)
        r_cup = _resolve_rate(cupom_rate_or_curve, du, dc)
        return float(
            ndf_mtm(
                self.notional_usd,
                self.strike_brl,
                float(spot_ptax),
                r_pre,
                du,
                r_cup,
                dc,
                float(self.direction),
            )
        )


def price_di1_portfolio(
    positions: List[Dict],
    base_date: Union[datetime.date, str],
    pre_curve: Optional[Union[float, Curve]] = None,
) -> pd.DataFrame:
    """
    Precifica em lote uma carteira de contratos futuros DI1 usando o núcleo Cython
    `price_di1_batch`.
    """
    if not positions:
        return pd.DataFrame(columns=["ticker", "du", "rate", "contracts", "pu", "dv01"])

    base_dt = _to_date(base_date)
    tickers = []
    dus = []
    rates = []
    contracts = []
    for pos in positions:
        inst: DI1Future = pos["instrument"]
        du = inst._du(base_dt)
        rc = pos.get("rate", pre_curve)
        if rc is None:
            raise ValueError(f"Informe 'rate' ou 'pre_curve' para {inst.ticker}.")
        tickers.append(inst.ticker)
        dus.append(du)
        rates.append(_resolve_rate(rc, du))
        contracts.append(float(pos.get("contracts", pos.get("quantity", 1.0))))

    pus, dv01s = price_di1_batch(
        np.asarray(rates, dtype=np.float64),
        np.asarray(dus, dtype=np.int32),
        np.asarray(contracts, dtype=np.float64),
    )
    return pd.DataFrame({
        "ticker": tickers,
        "du": dus,
        "rate": rates,
        "contracts": contracts,
        "pu": pus,
        "dv01": dv01s,
    })


_OIS_INDEX_DEFAULTS = {
    "SOFR": ("ACT/360", "US"),
    "FEDFUNDS": ("ACT/360", "US"),
    "ESTR": ("ACT/360", "TARGET"),
    "EURIBOR": ("ACT/360", "TARGET"),
    "SONIA": ("ACT/365", "UK"),
    "TONA": ("ACT/365", "ACTUAL"),
}


def _bump_rate_or_curve(rate_or_curve: Union[float, Curve], shift: float, default_basis: float) -> Union[float, Curve]:
    if isinstance(rate_or_curve, Curve):
        return _ShiftedCurve(rate_or_curve, shift, default_basis=default_basis)
    return float(rate_or_curve) + float(shift)


class OISSwap:
    """
    Swap de taxa de juros overnight offshore (Overnight Indexed Swap: SOFR, ESTR, SONIA, TONA, FEDFUNDS).
    A ponta fixa paga taxa periódica simples segundo a convenção do contrato, enquanto a ponta
    flutuante acumula a composição diária overnight projetada pela curva de forwarding.
    `fixed_direction = 1` recebe a ponta fixa e paga a ponta flutuante; `-1` inverte a posição.
    """

    def __init__(
        self,
        notional: float,
        start_date: Union[datetime.date, str],
        maturity_date: Union[datetime.date, str],
        fixed_rate: float,
        fixed_direction: int = 1,
        payment_freq_months: int = 12,
        day_count: Optional[str] = None,
        index_name: str = "SOFR",
        calendar_name: Optional[str] = None,
        holidays: Any = None,
        payment_dates: Optional[List[Union[datetime.date, str]]] = None,
        spread: float = 0.0,
        date_adjustment: str = "modified_following",
        rate_compounding: str = "auto",
    ):
        self.notional = float(notional)
        self.start_date = _to_date(start_date)
        self.maturity_date = _to_date(maturity_date)
        if self.maturity_date <= self.start_date:
            raise ValueError("maturity_date deve ser posterior a start_date.")
        self.fixed_rate = float(fixed_rate)
        self.fixed_direction = 1 if fixed_direction >= 0 else -1
        self.payment_freq_months = int(payment_freq_months) if payment_freq_months is not None else 12
        self.index_name = str(index_name).strip().upper()
        def_dc, def_cal = _OIS_INDEX_DEFAULTS.get(self.index_name, ("ACT/360", "US"))
        self.day_count = _normalize_day_count(day_count) if day_count is not None else def_dc
        cal_use = calendar_name if calendar_name is not None else def_cal
        self.calendar = get_calendar(cal_use, holidays=holidays)
        self.spread = float(spread)
        self.date_adjustment = str(date_adjustment).strip().lower()
        self.rate_compounding = str(rate_compounding).strip().lower()
        if payment_dates is not None:
            pts = sorted({_to_date(d) for d in payment_dates if _to_date(d) > self.start_date})
            self.schedule_dates = [self.start_date] + pts
        else:
            self.schedule_dates = _generate_schedule_dates(
                self.start_date,
                self.maturity_date,
                self.payment_freq_months,
                cal=self.calendar,
                date_adjustment=self.date_adjustment,
            )

    def _period_rows(
        self,
        base_dt: datetime.date,
        discount_curve: Union[float, Curve],
        forwarding_curve: Optional[Union[float, Curve]] = None,
        accrued_overnight_factor: float = 1.0,
    ) -> List[Dict[str, Any]]:
        fwd_src = forwarding_curve if forwarding_curve is not None else discount_curve
        rows: List[Dict[str, Any]] = []
        for idx in range(1, len(self.schedule_dates)):
            d_start = self.schedule_dates[idx - 1]
            d_end = self.schedule_dates[idx]
            if d_end <= base_dt:
                continue
            alpha = _year_fraction_bond(d_start, d_end, self.day_count, freq=1)
            _, df_disc = _resolve_discount_factor(
                discount_curve, base_dt, d_end, self.calendar, self.day_count, self.rate_compounding
            )
            _, df_fwd_end = _resolve_discount_factor(
                fwd_src, base_dt, d_end, self.calendar, self.day_count, self.rate_compounding
            )
            if d_start <= base_dt:
                df_fwd_start = 1.0
                acc_fac = float(accrued_overnight_factor)
            else:
                _, df_fwd_start = _resolve_discount_factor(
                    fwd_src, base_dt, d_start, self.calendar, self.day_count, self.rate_compounding
                )
                acc_fac = 1.0

            comp_factor = acc_fac * (df_fwd_start / max(1e-15, df_fwd_end))
            fwd_simple = (comp_factor - 1.0) / max(1e-12, alpha)
            cf_fix = self.notional * self.fixed_rate * alpha
            cf_flt = self.notional * ((comp_factor - 1.0) + self.spread * alpha)
            rows.append({
                "period_start": d_start,
                "payment_date": d_end,
                "dc_pay": (d_end - base_dt).days,
                "alpha": alpha,
                "df_discount": df_disc,
                "df_forward_start": df_fwd_start,
                "df_forward_end": df_fwd_end,
                "forward_rate": fwd_simple,
                "cf_fixed": cf_fix,
                "cf_floating": cf_flt,
                "pv_fixed": cf_fix * df_disc,
                "pv_floating": cf_flt * df_disc,
            })
        return rows

    def cashflow_table(
        self,
        base_date: Union[datetime.date, str],
        discount_curve: Union[float, Curve],
        forwarding_curve: Optional[Union[float, Curve]] = None,
        accrued_overnight_factor: float = 1.0,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self._period_rows(_to_date(base_date), discount_curve, forwarding_curve, accrued_overnight_factor)
        )

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        discount_curve: Union[float, Curve],
        forwarding_curve: Optional[Union[float, Curve]] = None,
        accrued_overnight_factor: float = 1.0,
    ) -> Dict[str, float]:
        base_dt = _to_date(base_date)
        rows = self._period_rows(base_dt, discount_curve, forwarding_curve, accrued_overnight_factor)
        if not rows:
            return {"mtm": 0.0, "pv_fixed": 0.0, "pv_floating": 0.0, "annuity": 0.0, "par_rate": 0.0, "dv01": 0.0}

        pv_fix = sum(r["pv_fixed"] for r in rows)
        pv_flt = sum(r["pv_floating"] for r in rows)
        annuity = sum(r["alpha"] * r["df_discount"] for r in rows)
        par_rate = (pv_flt - self.notional * self.spread * annuity) / (self.notional * annuity) if annuity > 0.0 else 0.0
        mtm_val = float(self.fixed_direction) * (pv_fix - pv_flt)

        basis_def = 365.0 if "365" in self.day_count else 360.0
        d_up = _bump_rate_or_curve(discount_curve, 1e-4, basis_def)
        f_up = _bump_rate_or_curve(forwarding_curve, 1e-4, basis_def) if forwarding_curve is not None else None
        rows_up = self._period_rows(base_dt, d_up, f_up, accrued_overnight_factor)
        pv_fix_up = sum(r["pv_fixed"] for r in rows_up)
        pv_flt_up = sum(r["pv_floating"] for r in rows_up)
        mtm_up = float(self.fixed_direction) * (pv_fix_up - pv_flt_up)

        return {
            "mtm": float(mtm_val),
            "pv_fixed": float(pv_fix),
            "pv_floating": float(pv_flt),
            "annuity": float(annuity),
            "par_rate": float(par_rate),
            "dv01": float(mtm_up - mtm_val),
        }

    def par_rate(
        self,
        base_date: Union[datetime.date, str],
        discount_curve: Union[float, Curve],
        forwarding_curve: Optional[Union[float, Curve]] = None,
        accrued_overnight_factor: float = 1.0,
    ) -> float:
        return self.risk_metrics(base_date, discount_curve, forwarding_curve, accrued_overnight_factor)["par_rate"]

    def mtm(
        self,
        base_date: Union[datetime.date, str],
        discount_curve: Union[float, Curve],
        forwarding_curve: Optional[Union[float, Curve]] = None,
        accrued_overnight_factor: float = 1.0,
    ) -> float:
        return self.risk_metrics(base_date, discount_curve, forwarding_curve, accrued_overnight_factor)["mtm"]

    def dv01(
        self,
        base_date: Union[datetime.date, str],
        discount_curve: Union[float, Curve],
        forwarding_curve: Optional[Union[float, Curve]] = None,
        accrued_overnight_factor: float = 1.0,
    ) -> float:
        return self.risk_metrics(base_date, discount_curve, forwarding_curve, accrued_overnight_factor)["dv01"]


class SOFRFuture:
    """
    Contrato futuro CME sobre taxa SOFR:
    - 'SR3' (3 meses composto): multiplicador padrão de USD 2.500 por ponto de índice (USD 25 por bp).
    - 'SR1' (1 mês médio aritmético/contínuo): multiplicador padrão de USD 4.167 por ponto (USD 41,67 por bp).
    """

    def __init__(
        self,
        ticker: str,
        start_date: Union[datetime.date, str],
        end_date: Union[datetime.date, str],
        contract_type: str = "SR3",
        contract_multiplier: Optional[float] = None,
        calendar_name: str = "US",
        holidays: Any = None,
        rate_compounding: str = "auto",
    ):
        self.ticker = ticker.strip().upper()
        self.start_date = _to_date(start_date)
        self.end_date = _to_date(end_date)
        if self.end_date <= self.start_date:
            raise ValueError("end_date deve ser posterior a start_date.")
        ctype = str(contract_type).strip().upper()
        if ctype not in ("SR3", "SR1"):
            raise ValueError("contract_type inválido. Use 'SR3' ou 'SR1'.")
        self.contract_type = ctype
        default_mult = 2500.0 if ctype == "SR3" else 4167.0
        self.MULTIPLIER = float(contract_multiplier) if contract_multiplier is not None else default_mult
        self.calendar = get_calendar(calendar_name, holidays=holidays)
        self.rate_compounding = str(rate_compounding).strip().lower()

    def fair_rate(
        self,
        base_date: Union[datetime.date, str],
        sofr_curve: Union[float, Curve],
        accrued_factor: float = 1.0,
        convexity_adj: float = 0.0,
    ) -> float:
        base_dt = _to_date(base_date)
        tau = float((self.end_date - self.start_date).days) / 360.0
        if self.end_date <= base_dt:
            return (float(accrued_factor) - 1.0) / max(1e-12, tau)
        _, df_end = _resolve_discount_factor(
            sofr_curve, base_dt, self.end_date, self.calendar, "ACT/360", self.rate_compounding
        )
        if base_dt < self.start_date:
            _, df_start = _resolve_discount_factor(
                sofr_curve, base_dt, self.start_date, self.calendar, "ACT/360", self.rate_compounding
            )
            ratio = df_start / max(1e-15, df_end)
        else:
            ratio = float(accrued_factor) / max(1e-15, df_end)

        if self.contract_type == "SR1":
            base_rate = math.log(max(1e-12, ratio)) / max(1e-12, tau)
        else:
            base_rate = (ratio - 1.0) / max(1e-12, tau)
        return float(base_rate + float(convexity_adj))

    def fair_price(
        self,
        base_date: Union[datetime.date, str],
        sofr_curve: Union[float, Curve],
        accrued_factor: float = 1.0,
        convexity_adj: float = 0.0,
    ) -> float:
        r = self.fair_rate(base_date, sofr_curve, accrued_factor=accrued_factor, convexity_adj=convexity_adj)
        return 100.0 * (1.0 - r)

    @staticmethod
    def implied_rate(price: float) -> float:
        return (100.0 - float(price)) / 100.0

    def dv01(self, contracts: float = 1.0) -> float:
        return float(self.MULTIPLIER * 0.01 * float(contracts))

    def daily_settlement(
        self,
        price_atual: float,
        price_anterior: float,
        contracts: float = 1.0,
        direction: int = 1,
    ) -> float:
        sgn = 1.0 if direction >= 0 else -1.0
        return float(sgn * float(contracts) * self.MULTIPLIER * (float(price_atual) - float(price_anterior)))


class CrossCurrencySwap:
    """
    Swap de moedas entre Brasil (DI ou Pré em BUS/252) e moeda estrangeira (USD/EUR em ACT/360 ou ACT/365),
    com troca de principal opcional e cálculo integralmente offline de MtM, DV01 por ponta, delta FX,
    taxa/spread par doméstico e basis spread par na ponta externa.
    """

    def __init__(
        self,
        notional_foreign: float,
        start_date: Union[datetime.date, str],
        maturity_date: Union[datetime.date, str],
        initial_fx_spot: Optional[float] = None,
        notional_brl: Optional[float] = None,
        brl_direction: int = 1,
        brl_leg_type: str = "DI",
        foreign_leg_type: str = "FLOAT",
        brl_rate: float = 0.0,
        foreign_rate: float = 0.0,
        payment_freq_months: int = 6,
        day_count_foreign: str = "ACT/360",
        include_notional_exchange: bool = True,
        calendar_brl: str = "B3",
        calendar_foreign: str = "US",
        holidays_brl: Any = None,
        holidays_foreign: Any = None,
        payment_dates: Optional[List[Union[datetime.date, str]]] = None,
        foreign_rate_compounding: str = "auto",
    ):
        self.notional_foreign = float(notional_foreign)
        self.start_date = _to_date(start_date)
        self.maturity_date = _to_date(maturity_date)
        if self.maturity_date <= self.start_date:
            raise ValueError("maturity_date deve ser posterior a start_date.")
        self.initial_fx_spot = float(initial_fx_spot) if initial_fx_spot is not None else None
        self.notional_brl = float(notional_brl) if notional_brl is not None else None
        self.brl_direction = 1 if brl_direction >= 0 else -1
        b_type = str(brl_leg_type).strip().upper()
        f_type = str(foreign_leg_type).strip().upper()
        if b_type not in ("DI", "PRE"):
            raise ValueError("brl_leg_type deve ser 'DI' ou 'PRE'.")
        if f_type not in ("FLOAT", "FIXED"):
            raise ValueError("foreign_leg_type deve ser 'FLOAT' ou 'FIXED'.")
        self.brl_leg_type = b_type
        self.foreign_leg_type = f_type
        self.brl_rate = float(brl_rate)
        self.foreign_rate = float(foreign_rate)
        self.payment_freq_months = int(payment_freq_months) if payment_freq_months is not None else 6
        self.day_count_foreign = _normalize_day_count(day_count_foreign)
        self.include_notional_exchange = bool(include_notional_exchange)
        self.calendar_brl = get_calendar(calendar_brl, holidays=holidays_brl)
        self.calendar_foreign = get_calendar(calendar_foreign, holidays=holidays_foreign)
        self.foreign_rate_compounding = str(foreign_rate_compounding).strip().lower()
        if payment_dates is not None:
            pts = sorted({_to_date(d) for d in payment_dates if _to_date(d) > self.start_date})
            self.schedule_dates = [self.start_date] + pts
        else:
            self.schedule_dates = _generate_schedule_dates(
                self.start_date,
                self.maturity_date,
                self.payment_freq_months,
                cal=self.calendar_foreign,
                date_adjustment="modified_following",
            )

    def _effective_notional_brl(self, fx_spot: float) -> float:
        if self.notional_brl is not None:
            return self.notional_brl
        ref_fx = self.initial_fx_spot if self.initial_fx_spot is not None else float(fx_spot)
        return self.notional_foreign * ref_fx

    def _evaluate_legs(
        self,
        base_dt: datetime.date,
        fx_spot: float,
        pre_curve: Union[float, Curve],
        foreign_discount_curve: Union[float, Curve],
        foreign_forward_curve: Optional[Union[float, Curve]] = None,
        di_accrued_factor: float = 1.0,
        foreign_accrued_factor: float = 1.0,
        brl_rate_override: Optional[float] = None,
        foreign_rate_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        n_brl = self._effective_notional_brl(fx_spot)
        n_fx = self.notional_foreign
        b_rate = self.brl_rate if brl_rate_override is None else float(brl_rate_override)
        f_rate = self.foreign_rate if foreign_rate_override is None else float(foreign_rate_override)
        fx_fwd_crv = foreign_forward_curve if foreign_forward_curve is not None else foreign_discount_curve

        future_idx = [i for i in range(1, len(self.schedule_dates)) if self.schedule_dates[i] > base_dt]
        if not future_idx:
            return {"pv_brl": 0.0, "pv_foreign": 0.0, "fx_annuity": 0.0, "brl_annuity": 0.0, "rows": []}

        last_idx = future_idx[-1]
        pv_brl = 0.0
        pv_fx = 0.0
        fx_annuity = 0.0
        brl_annuity = 0.0
        rows: List[Dict[str, Any]] = []

        for idx in future_idx:
            d_start = self.schedule_dates[idx - 1]
            d_end = self.schedule_dates[idx]
            d_end_brl = self.calendar_brl.following(d_end)
            d_start_brl = self.calendar_brl.following(d_start) if d_start > base_dt else d_start

            # Ponta BRL (BUS/252)
            du_end = max(1, int(self.calendar_brl.bizdays(base_dt, d_end_brl)))
            _, df_brl_end = _resolve_discount_factor(pre_curve, base_dt, d_end_brl, self.calendar_brl, "BUS/252", "exponential_252")
            if d_start <= base_dt:
                du_p = max(1, int(self.calendar_brl.bizdays(d_start_brl, d_end_brl)))
                df_brl_start = 1.0
                acc_di = float(di_accrued_factor)
            else:
                du_p = max(1, int(self.calendar_brl.bizdays(d_start_brl, d_end_brl)))
                _, df_brl_start = _resolve_discount_factor(pre_curve, base_dt, d_start_brl, self.calendar_brl, "BUS/252", "exponential_252")
                acc_di = 1.0

            tau_brl_p = float(du_p) / 252.0
            if self.brl_leg_type == "PRE":
                cf_brl_cpn = n_brl * (math.pow(1.0 + b_rate, tau_brl_p) - 1.0)
            else:
                di_fac_p = acc_di * (df_brl_start / max(1e-15, df_brl_end))
                spr_fac_p = math.pow(1.0 + b_rate, tau_brl_p)
                cf_brl_cpn = n_brl * (di_fac_p * spr_fac_p - 1.0)
            cf_brl_princ = n_brl if (self.include_notional_exchange and idx == last_idx) else 0.0
            cf_brl_tot = cf_brl_cpn + cf_brl_princ
            pv_brl_i = cf_brl_tot * df_brl_end
            pv_brl += pv_brl_i
            brl_annuity += tau_brl_p * df_brl_end

            # Ponta estrangeira (ACT/360 ou ACT/365)
            alpha_fx = _year_fraction_bond(d_start, d_end, self.day_count_foreign, freq=1)
            _, df_fx_disc = _resolve_discount_factor(
                foreign_discount_curve, base_dt, d_end, self.calendar_foreign, self.day_count_foreign, self.foreign_rate_compounding
            )
            _, df_fx_fwd_end = _resolve_discount_factor(
                fx_fwd_crv, base_dt, d_end, self.calendar_foreign, self.day_count_foreign, self.foreign_rate_compounding
            )
            if d_start <= base_dt:
                df_fx_fwd_start = 1.0
                acc_fx = float(foreign_accrued_factor)
            else:
                _, df_fx_fwd_start = _resolve_discount_factor(
                    fx_fwd_crv, base_dt, d_start, self.calendar_foreign, self.day_count_foreign, self.foreign_rate_compounding
                )
                acc_fx = 1.0

            if self.foreign_leg_type == "FIXED":
                cf_fx_cpn = n_fx * f_rate * alpha_fx
            else:
                fac_fx_p = acc_fx * (df_fx_fwd_start / max(1e-15, df_fx_fwd_end))
                cf_fx_cpn = n_fx * ((fac_fx_p - 1.0) + f_rate * alpha_fx)
            cf_fx_princ = n_fx if (self.include_notional_exchange and idx == last_idx) else 0.0
            cf_fx_tot = cf_fx_cpn + cf_fx_princ
            pv_fx_i = cf_fx_tot * df_fx_disc
            pv_fx += pv_fx_i
            fx_annuity += alpha_fx * df_fx_disc

            rows.append({
                "period_start": d_start,
                "payment_date": d_end,
                "cf_brl": cf_brl_tot,
                "df_brl": df_brl_end,
                "pv_brl": pv_brl_i,
                "cf_foreign": cf_fx_tot,
                "df_foreign": df_fx_disc,
                "pv_foreign": pv_fx_i,
            })

        return {
            "pv_brl": float(pv_brl),
            "pv_foreign": float(pv_fx),
            "fx_annuity": float(fx_annuity),
            "brl_annuity": float(brl_annuity),
            "n_brl": float(n_brl),
            "rows": rows,
        }

    def cashflow_table(
        self,
        base_date: Union[datetime.date, str],
        fx_spot: float,
        pre_curve: Union[float, Curve],
        foreign_discount_curve: Union[float, Curve],
        foreign_forward_curve: Optional[Union[float, Curve]] = None,
        di_accrued_factor: float = 1.0,
        foreign_accrued_factor: float = 1.0,
    ) -> pd.DataFrame:
        ev = self._evaluate_legs(
            _to_date(base_date),
            float(fx_spot),
            pre_curve,
            foreign_discount_curve,
            foreign_forward_curve,
            di_accrued_factor,
            foreign_accrued_factor,
        )
        return pd.DataFrame(ev["rows"])

    def par_basis_spread(
        self,
        base_date: Union[datetime.date, str],
        fx_spot: float,
        pre_curve: Union[float, Curve],
        foreign_discount_curve: Union[float, Curve],
        foreign_forward_curve: Optional[Union[float, Curve]] = None,
        di_accrued_factor: float = 1.0,
        foreign_accrued_factor: float = 1.0,
    ) -> float:
        """Retorna a taxa ou spread da ponta externa que iguala os valores presentes das duas pontas."""
        base_dt = _to_date(base_date)
        ev0 = self._evaluate_legs(
            base_dt,
            float(fx_spot),
            pre_curve,
            foreign_discount_curve,
            foreign_forward_curve,
            di_accrued_factor,
            foreign_accrued_factor,
            foreign_rate_override=0.0,
        )
        denom = self.notional_foreign * ev0["fx_annuity"]
        if denom <= 0.0:
            return 0.0
        target_pv_fx = ev0["pv_brl"] / float(fx_spot)
        return float((target_pv_fx - ev0["pv_foreign"]) / denom)

    def par_brl_rate(
        self,
        base_date: Union[datetime.date, str],
        fx_spot: float,
        pre_curve: Union[float, Curve],
        foreign_discount_curve: Union[float, Curve],
        foreign_forward_curve: Optional[Union[float, Curve]] = None,
        di_accrued_factor: float = 1.0,
        foreign_accrued_factor: float = 1.0,
    ) -> float:
        """Retorna a taxa ou spread doméstico em base BUS/252 que zera o MtM do swap."""
        from scipy.optimize import brentq

        base_dt = _to_date(base_date)

        def _err(r_b: float) -> float:
            ev = self._evaluate_legs(
                base_dt,
                float(fx_spot),
                pre_curve,
                foreign_discount_curve,
                foreign_forward_curve,
                di_accrued_factor,
                foreign_accrued_factor,
                brl_rate_override=r_b,
            )
            return ev["pv_brl"] - float(fx_spot) * ev["pv_foreign"]

        return float(brentq(_err, -0.50, 1.50, xtol=1e-11))

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        fx_spot: float,
        pre_curve: Union[float, Curve],
        foreign_discount_curve: Union[float, Curve],
        foreign_forward_curve: Optional[Union[float, Curve]] = None,
        di_accrued_factor: float = 1.0,
        foreign_accrued_factor: float = 1.0,
    ) -> Dict[str, float]:
        base_dt = _to_date(base_date)
        spot = float(fx_spot)
        ev = self._evaluate_legs(
            base_dt, spot, pre_curve, foreign_discount_curve, foreign_forward_curve, di_accrued_factor, foreign_accrued_factor
        )
        mtm_brl = float(self.brl_direction) * (ev["pv_brl"] - spot * ev["pv_foreign"])
        mtm_fx = mtm_brl / spot
        fx_delta = -float(self.brl_direction) * ev["pv_foreign"]

        # DV01 ponta BRL (+1 bp na curva Pré)
        pre_up = _bump_rate_or_curve(pre_curve, 1e-4, 252.0)
        ev_b_up = self._evaluate_legs(
            base_dt, spot, pre_up, foreign_discount_curve, foreign_forward_curve, di_accrued_factor, foreign_accrued_factor
        )
        mtm_b_up = float(self.brl_direction) * (ev_b_up["pv_brl"] - spot * ev_b_up["pv_foreign"])

        # DV01 ponta externa (+1 bp nas curvas externas)
        basis_fx = 365.0 if "365" in self.day_count_foreign else 360.0
        f_d_up = _bump_rate_or_curve(foreign_discount_curve, 1e-4, basis_fx)
        f_f_up = _bump_rate_or_curve(foreign_forward_curve, 1e-4, basis_fx) if foreign_forward_curve is not None else None
        ev_f_up = self._evaluate_legs(
            base_dt, spot, pre_curve, f_d_up, f_f_up, di_accrued_factor, foreign_accrued_factor
        )
        mtm_f_up = float(self.brl_direction) * (ev_f_up["pv_brl"] - spot * ev_f_up["pv_foreign"])
        dv01_fx_brl = mtm_f_up - mtm_brl

        return {
            "mtm_brl": float(mtm_brl),
            "mtm_foreign": float(mtm_fx),
            "pv_brl": float(ev["pv_brl"]),
            "pv_foreign": float(ev["pv_foreign"]),
            "fx_delta": float(fx_delta),
            "dv01_brl": float(mtm_b_up - mtm_brl),
            "dv01_foreign_brl": float(dv01_fx_brl),
            "dv01_foreign": float(dv01_fx_brl / spot),
            "par_basis_spread": self.par_basis_spread(
                base_dt, spot, pre_curve, foreign_discount_curve, foreign_forward_curve, di_accrued_factor, foreign_accrued_factor
            ),
        }

    def mtm(
        self,
        base_date: Union[datetime.date, str],
        fx_spot: float,
        pre_curve: Union[float, Curve],
        foreign_discount_curve: Union[float, Curve],
        foreign_forward_curve: Optional[Union[float, Curve]] = None,
        di_accrued_factor: float = 1.0,
        foreign_accrued_factor: float = 1.0,
    ) -> float:
        return self.risk_metrics(
            base_date, fx_spot, pre_curve, foreign_discount_curve, foreign_forward_curve, di_accrued_factor, foreign_accrued_factor
        )["mtm_brl"]


