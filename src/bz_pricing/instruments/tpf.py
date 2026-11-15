import datetime
import math
from typing import List, Union, Optional, Tuple
import numpy as np
import pandas as pd

from bz_pricing.core.calendars import get_calendar
from bz_pricing.core.curves import Curve
from bz_pricing.core.bonds import (
    LTN as _LTN_C,
    NTNF as _NTNF_C,
    LFT as _LFT_C,
    NTNB as _NTNB_C,
    NTNBP as _NTNBP_C,
    NTNC as _NTNC_C
)


class TituloPublico:
    """
    Classe base para instrumentos de títulos públicos federais (TPF).
    """
    def __init__(self, maturity_date: Union[datetime.date, str], calendar_name: str = 'B3'):
        if isinstance(maturity_date, str):
            maturity_date = pd.to_datetime(maturity_date).date()
        self.nominal_maturity_date = maturity_date
        self.calendar = get_calendar(calendar_name)
        self.maturity_date = self.calendar.following(maturity_date)

    def _normalize_date(self, d: Union[datetime.date, str]) -> datetime.date:
        if isinstance(d, str):
            return pd.to_datetime(d).date()
        return d


class LTN(TituloPublico):
    """
    Letra do Tesouro Nacional (título prefixado zero-cupom).
    Valor nominal no vencimento de R$ 1.000,00.
    """
    def __init__(self, maturity_date: Union[datetime.date, str], face_value: float = 1000.0, calendar_name: str = 'B3'):
        super().__init__(maturity_date, calendar_name)
        self.face_value = face_value

    def cashflow_table(self, base_date: Union[datetime.date, str], curve: Curve) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _LTN_C(du, self.face_value)
        details = bond_c.cashflow_details(curve)
        rows = []
        for du_val, fl_nom, rate, df, pv in details:
            rows.append({
                'data_fluxo': self.maturity_date,
                'du': du_val,
                'tipo': 'Principal',
                'fluxo_nominal': fl_nom,
                'taxa': rate,
                'fator_desconto': df,
                'valor_presente': pv
            })
        return pd.DataFrame(rows)

    def present_value(self, base_date: Union[datetime.date, str], curve: Curve, return_cashflows: bool = False) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            pu = self.face_value
            if return_cashflows:
                df = pd.DataFrame([{
                    'data_fluxo': self.maturity_date,
                    'du': 0,
                    'tipo': 'Principal',
                    'fluxo_nominal': self.face_value,
                    'taxa': 0.0,
                    'fator_desconto': 1.0,
                    'valor_presente': self.face_value
                }])
                return pu, df
            return pu

        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _LTN_C(du, self.face_value)
        pu = bond_c.present_value(curve)
        if return_cashflows:
            return pu, self.cashflow_table(base_date, curve)
        return pu

    def duration(self, base_date: Union[datetime.date, str], curve: Curve = None) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        return float(self.calendar.bizdays(base_date, self.maturity_date))

    def dv01(self, base_date: Union[datetime.date, str], curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _LTN_C(du, self.face_value)
        return bond_c.dv01(curve)


class NTNF(TituloPublico):
    """
    Nota do Tesouro Nacional - Série F (título prefixado com cupons semestrais).
    Taxa de cupom editalícia de 10,00% a.a. e cupons pagos em 01/01 e 01/07.
    """
    def __init__(self, maturity_date: Union[datetime.date, str], coupon_rate: float = 0.10, face_value: float = 1000.0, calendar_name: str = 'B3'):
        super().__init__(maturity_date, calendar_name)
        self.coupon_rate = coupon_rate
        self.face_value = face_value

    def get_coupon_dates(self, base_date: Union[datetime.date, str]) -> List[datetime.date]:
        base_date = self._normalize_date(base_date)
        dates = []
        curr_year = base_date.year

        while True:
            d1 = datetime.date(curr_year, 1, 1)
            d2 = datetime.date(curr_year, 7, 1)

            for d in [d1, d2]:
                adj_d = self.calendar.following(d)
                if base_date < adj_d <= self.maturity_date:
                    if adj_d not in dates:
                        dates.append(adj_d)

            if curr_year >= self.nominal_maturity_date.year:
                break
            curr_year += 1

        mat_adj = self.maturity_date
        if mat_adj not in dates and mat_adj > base_date:
            dates.append(mat_adj)

        return sorted(dates)

    def cashflow_table(self, base_date: Union[datetime.date, str], curve: Curve) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNF_C(dus, self.coupon_rate, self.face_value)
        details = bond_c.cashflow_details(curve)
        rows = []
        for i, (du_val, fl_nom, rate, df, pv) in enumerate(details):
            tipo = 'Cupom + Principal' if i == len(details) - 1 else 'Cupom'
            rows.append({
                'data_fluxo': coupon_dates[i],
                'du': du_val,
                'tipo': tipo,
                'fluxo_nominal': fl_nom,
                'taxa': rate,
                'fator_desconto': df,
                'valor_presente': pv
            })
        return pd.DataFrame(rows)

    def present_value(self, base_date: Union[datetime.date, str], curve: Curve, return_cashflows: bool = False) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            pu = self.face_value
            if return_cashflows:
                return pu, pd.DataFrame([{
                    'data_fluxo': self.maturity_date,
                    'du': 0,
                    'tipo': 'Principal',
                    'fluxo_nominal': self.face_value,
                    'taxa': 0.0,
                    'fator_desconto': 1.0,
                    'valor_presente': self.face_value
                }])
            return pu

        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNF_C(dus, self.coupon_rate, self.face_value)
        pu = bond_c.present_value(curve)
        if return_cashflows:
            return pu, self.cashflow_table(base_date, curve)
        return pu

    def duration(self, base_date: Union[datetime.date, str], curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNF_C(dus, self.coupon_rate, self.face_value)
        return bond_c.duration(curve)

    def modified_duration(self, base_date: Union[datetime.date, str], curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNF_C(dus, self.coupon_rate, self.face_value)
        return bond_c.modified_duration(curve)

    def dv01(self, base_date: Union[datetime.date, str], curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNF_C(dus, self.coupon_rate, self.face_value)
        return bond_c.dv01(curve)


class LFT(TituloPublico):
    """
    Letra Financeira do Tesouro (título pós-fixado atrelado à taxa Selic).
    Precificada a partir da cotação de ágio/deságio aplicada sobre o VNA da LFT.
    """
    def __init__(self, maturity_date: Union[datetime.date, str], calendar_name: str = 'B3'):
        super().__init__(maturity_date, calendar_name)

    def cashflow_table(self, base_date: Union[datetime.date, str], agio_curve: Curve, vna: float = None) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _LFT_C(du)
        vna_val = float(vna) if vna is not None else 0.0
        details = bond_c.cashflow_details(agio_curve, vna=vna_val)
        rows = []
        for du_val, fl_base, rate, df, cot, v, pu in details:
            rows.append({
                'data_fluxo': self.maturity_date,
                'du': du_val,
                'tipo': 'Resgate VNA',
                'fluxo_base': fl_base,
                'taxa_ágio': rate,
                'fator_desconto': df,
                'cotação': cot,
                'vna': v,
                'valor_presente': pu
            })
        return pd.DataFrame(rows)

    def cotacao(self, base_date: Union[datetime.date, str], agio_curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 100.0
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _LFT_C(du)
        return bond_c.cotacao(agio_curve)

    def present_value(self, base_date: Union[datetime.date, str], agio_curve: Curve, vna: float = None,
                      selic_curve: Curve = None, return_cashflows: bool = False) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _LFT_C(du)
        vna_val = float(vna) if vna is not None else 0.0
        pu = bond_c.present_value(agio_curve, vna=vna_val, selic_curve=selic_curve)
        if return_cashflows:
            effective_vna = vna_val
            if effective_vna <= 0.0 and selic_curve is not None:
                selic_r = selic_curve.get_rate(du)
                effective_vna = math.trunc((1000.0 * math.trunc((1.0 + selic_r) ** (du / 252.0) * 1e14) / 1e14) * 1e6) / 1e6
            return pu, self.cashflow_table(base_date, agio_curve, vna=effective_vna)
        return pu

    def duration(self, base_date: Union[datetime.date, str], agio_curve: Curve = None) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        return float(self.calendar.bizdays(base_date, self.maturity_date))

    def dv01(self, base_date: Union[datetime.date, str], agio_curve: Curve, vna: float) -> float:
        base_date = self._normalize_date(base_date)
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _LFT_C(du)
        return bond_c.dv01(agio_curve, float(vna))


class NTNB(TituloPublico):
    """
    Nota do Tesouro Nacional - Série B (título indexado ao IPCA com cupons semestrais).
    Taxa de cupom editalícia de 6,00% a.a. aplicada sobre o Valor Nominal Atualizado (VNA).
    """
    def __init__(self, maturity_date: Union[datetime.date, str], coupon_rate: float = 0.06, calendar_name: str = 'B3'):
        super().__init__(maturity_date, calendar_name)
        self.coupon_rate = coupon_rate

    def get_coupon_dates(self, base_date: Union[datetime.date, str]) -> List[datetime.date]:
        base_date = self._normalize_date(base_date)
        dates = []
        m1 = self.nominal_maturity_date.month
        m2 = (m1 + 6) if m1 <= 6 else (m1 - 6)
        curr_year = base_date.year

        while curr_year <= self.nominal_maturity_date.year:
            for m in sorted([m1, m2]):
                d = self.calendar.following(datetime.date(curr_year, m, 15))
                if base_date < d <= self.maturity_date:
                    if d not in dates:
                        dates.append(d)
            curr_year += 1

        mat_adj = self.maturity_date
        if mat_adj not in dates and mat_adj > base_date:
            dates.append(mat_adj)

        return sorted(dates)

    def cashflow_table(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None,
                       inflation_curve: Curve = None, base_vna: float = None, base_vna_date: Union[datetime.date, str] = None) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)

        vna_val = float(vna) if vna is not None else 0.0
        base_vna_val = float(base_vna) if base_vna is not None else 0.0
        if vna_val <= 0.0 and inflation_curve is not None and base_vna_val > 0.0 and base_vna_date is not None:
            base_vna_date = self._normalize_date(base_vna_date)
            du_inf = self.calendar.bizdays(base_vna_date, base_date)
            ipca_r = inflation_curve.get_rate(du_inf)
            ipca_f = math.trunc(math.pow(1.0 + ipca_r, du_inf / 252.0) * 1e14) / 1e14
            vna_val = math.trunc((base_vna_val * ipca_f) * 1e6) / 1e6

        details = bond_c.cashflow_details(real_curve, vna=vna_val)
        rows = []
        for i, (du_val, fl_cot, rate, df, pv_cot, pv_fin) in enumerate(details):
            tipo = 'Cupom + Principal' if i == len(details) - 1 else 'Cupom'
            rows.append({
                'data_fluxo': coupon_dates[i],
                'du': du_val,
                'tipo': tipo,
                'fluxo_cotação': fl_cot,
                'taxa_real': rate,
                'fator_desconto': df,
                'cotação_descontada': pv_cot,
                'vna': vna_val,
                'valor_presente': pv_fin
            })
        return pd.DataFrame(rows)

    def cotacao(self, base_date: Union[datetime.date, str], real_curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 100.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)
        return bond_c.cotacao(real_curve)

    def present_value(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None,
                      inflation_curve: Curve = None, base_vna: float = None, base_vna_date: Union[datetime.date, str] = None,
                      return_cashflows: bool = False) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)

        vna_val = float(vna) if vna is not None else 0.0
        base_vna_val = float(base_vna) if base_vna is not None else 0.0
        du_inf = 0
        if base_vna_date is not None and inflation_curve is not None:
            base_vna_date = self._normalize_date(base_vna_date)
            du_inf = self.calendar.bizdays(base_vna_date, base_date)

        pu = bond_c.present_value(real_curve, vna=vna_val, inflation_curve=inflation_curve, base_vna=base_vna_val, du_inflation=du_inf)
        if return_cashflows:
            df_flows = self.cashflow_table(
                base_date=base_date,
                real_curve=real_curve,
                vna=vna,
                inflation_curve=inflation_curve,
                base_vna=base_vna,
                base_vna_date=base_vna_date
            )
            return pu, df_flows
        return pu

    def duration(self, base_date: Union[datetime.date, str], real_curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)
        return bond_c.duration(real_curve)

    def modified_duration(self, base_date: Union[datetime.date, str], real_curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)
        return bond_c.modified_duration(real_curve)

    def dv01(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)
        vna_val = float(vna) if vna is not None else 0.0
        return bond_c.dv01(real_curve, vna=vna_val)


class NTNBPrincipal(TituloPublico):
    """
    Nota do Tesouro Nacional - Série B Principal (título indexado ao IPCA sem cupons intermediários).
    """
    def __init__(self, maturity_date: Union[datetime.date, str], calendar_name: str = 'B3'):
        super().__init__(maturity_date, calendar_name)

    def cashflow_table(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _NTNBP_C(du)
        vna_val = float(vna) if vna is not None else 0.0
        details = bond_c.cashflow_details(real_curve, vna=vna_val)
        rows = []
        for du_val, fl_base, rate, df, cot, v, pu in details:
            rows.append({
                'data_fluxo': self.maturity_date,
                'du': du_val,
                'tipo': 'Principal',
                'fluxo_cotação': fl_base,
                'taxa_real': rate,
                'fator_desconto': df,
                'cotação': cot,
                'vna': v,
                'valor_presente': pu
            })
        return pd.DataFrame(rows)

    def cotacao(self, base_date: Union[datetime.date, str], real_curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 100.0
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _NTNBP_C(du)
        return bond_c.cotacao(real_curve)

    def present_value(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None,
                      inflation_curve: Curve = None, base_vna: float = None, base_vna_date: Union[datetime.date, str] = None,
                      return_cashflows: bool = False) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _NTNBP_C(du)

        vna_val = float(vna) if vna is not None else 0.0
        base_vna_val = float(base_vna) if base_vna is not None else 0.0
        du_inf = 0
        if base_vna_date is not None and inflation_curve is not None:
            base_vna_date = self._normalize_date(base_vna_date)
            du_inf = self.calendar.bizdays(base_vna_date, base_date)

        pu = bond_c.present_value(real_curve, vna=vna_val, inflation_curve=inflation_curve, base_vna=base_vna_val, du_inflation=du_inf)
        if return_cashflows:
            effective_vna = vna_val
            if effective_vna <= 0.0 and inflation_curve is not None and base_vna_val > 0.0 and du_inf > 0:
                ipca_r = inflation_curve.get_rate(du_inf)
                ipca_f = math.trunc(math.pow(1.0 + ipca_r, du_inf / 252.0) * 1e14) / 1e14
                effective_vna = math.trunc((base_vna_val * ipca_f) * 1e6) / 1e6
            return pu, self.cashflow_table(base_date, real_curve, vna=effective_vna)
        return pu

    def duration(self, base_date: Union[datetime.date, str], real_curve: Curve = None) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        return float(self.calendar.bizdays(base_date, self.maturity_date))

    def modified_duration(self, base_date: Union[datetime.date, str], real_curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _NTNBP_C(du)
        return bond_c.modified_duration(real_curve)

    def dv01(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None) -> float:
        base_date = self._normalize_date(base_date)
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _NTNBP_C(du)
        vna_val = float(vna) if vna is not None else 0.0
        return bond_c.dv01(real_curve, vna=vna_val)


class NTNC(TituloPublico):
    """
    Nota do Tesouro Nacional - Série C (título indexado ao IGP-M com cupons semestrais).
    Taxa de cupom editalícia de 6,00% ou 12,00% a.a. aplicada sobre o VNA do IGP-M.
    """
    def __init__(self, maturity_date: Union[datetime.date, str], coupon_rate: float = 0.06, calendar_name: str = 'B3'):
        super().__init__(maturity_date, calendar_name)
        self.coupon_rate = coupon_rate

    def get_coupon_dates(self, base_date: Union[datetime.date, str]) -> List[datetime.date]:
        base_date = self._normalize_date(base_date)
        dates = []
        m1 = self.nominal_maturity_date.month
        m2 = (m1 + 6) if m1 <= 6 else (m1 - 6)
        curr_year = base_date.year

        while curr_year <= self.nominal_maturity_date.year:
            for m in sorted([m1, m2]):
                d = self.calendar.following(datetime.date(curr_year, m, 1))
                if base_date < d <= self.maturity_date:
                    if d not in dates:
                        dates.append(d)
            curr_year += 1

        mat_adj = self.maturity_date
        if mat_adj not in dates and mat_adj > base_date:
            dates.append(mat_adj)

        return sorted(dates)

    def cashflow_table(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None,
                       inflation_curve: Curve = None, base_vna: float = None, base_vna_date: Union[datetime.date, str] = None) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)

        vna_val = float(vna) if vna is not None else 0.0
        base_vna_val = float(base_vna) if base_vna is not None else 0.0
        if vna_val <= 0.0 and inflation_curve is not None and base_vna_val > 0.0 and base_vna_date is not None:
            base_vna_date = self._normalize_date(base_vna_date)
            du_inf = self.calendar.bizdays(base_vna_date, base_date)
            igpm_r = inflation_curve.get_rate(du_inf)
            igpm_f = math.trunc(math.pow(1.0 + igpm_r, du_inf / 252.0) * 1e14) / 1e14
            vna_val = math.trunc((base_vna_val * igpm_f) * 1e6) / 1e6

        details = bond_c.cashflow_details(real_curve, vna=vna_val)
        rows = []
        for i, (du_val, fl_cot, rate, df, pv_cot, pv_fin) in enumerate(details):
            tipo = 'Cupom + Principal' if i == len(details) - 1 else 'Cupom'
            rows.append({
                'data_fluxo': coupon_dates[i],
                'du': du_val,
                'tipo': tipo,
                'fluxo_cotação': fl_cot,
                'taxa_real': rate,
                'fator_desconto': df,
                'cotação_descontada': pv_cot,
                'vna': vna_val,
                'valor_presente': pv_fin
            })
        return pd.DataFrame(rows)

    def cotacao(self, base_date: Union[datetime.date, str], real_curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 100.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)
        return bond_c.cotacao(real_curve)

    def present_value(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None,
                      inflation_curve: Curve = None, base_vna: float = None, base_vna_date: Union[datetime.date, str] = None,
                      return_cashflows: bool = False) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)

        vna_val = float(vna) if vna is not None else 0.0
        base_vna_val = float(base_vna) if base_vna is not None else 0.0
        du_inf = 0
        if base_vna_date is not None and inflation_curve is not None:
            base_vna_date = self._normalize_date(base_vna_date)
            du_inf = self.calendar.bizdays(base_vna_date, base_date)

        pu = bond_c.present_value(real_curve, vna=vna_val, inflation_curve=inflation_curve, base_vna=base_vna_val, du_inflation=du_inf)
        if return_cashflows:
            df_flows = self.cashflow_table(
                base_date=base_date,
                real_curve=real_curve,
                vna=vna,
                inflation_curve=inflation_curve,
                base_vna=base_vna,
                base_vna_date=base_vna_date
            )
            return pu, df_flows
        return pu

    def duration(self, base_date: Union[datetime.date, str], real_curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)
        return bond_c.duration(real_curve)

    def modified_duration(self, base_date: Union[datetime.date, str], real_curve: Curve) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)
        return bond_c.modified_duration(real_curve)

    def dv01(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None) -> float:
        base_date = self._normalize_date(base_date)
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)
        vna_val = float(vna) if vna is not None else 0.0
        return bond_c.dv01(real_curve, vna=vna_val)
