import datetime
import math
from typing import Dict, List, Optional, Union

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
    ):
        self.ticker = ticker.strip().upper()
        self.maturity_date = _to_date(maturity_date)
        self.calendar = get_calendar(calendar_name)
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
    ):
        self.notional = float(notional)
        self.start_date = _to_date(start_date)
        self.maturity_date = _to_date(maturity_date)
        self.fixed_rate = float(fixed_rate)
        self.fixed_direction = 1 if fixed_direction >= 0 else -1
        self.calendar = get_calendar(calendar_name)

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
    ):
        self.notional = float(notional)
        self.start_date = _to_date(start_date)
        self.maturity_date = _to_date(maturity_date)
        self.ipca_coupon_rate = float(ipca_coupon_rate)
        self.ipca_direction = 1 if ipca_direction >= 0 else -1
        self.calendar = get_calendar(calendar_name)

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
    ):
        self.notional_usd = float(notional_usd)
        self.maturity_date = _to_date(maturity_date)
        self.strike_brl = float(strike_brl)
        self.direction = 1 if direction >= 0 else -1
        self.calendar = get_calendar(calendar_name)

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

