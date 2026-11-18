import datetime
import math
from typing import Any, List, Union, Optional, Tuple
import numpy as np
import pandas as pd

from bz_pricing.core.calendars import get_calendar
from bz_pricing.core.curves import Curve
from bz_pricing.core.vna_engine import (
    LFT_BASE_DATE,
    LFT_BASE_VALUE,
    NTNC_BASE_DATE,
    NTNC_BASE_VALUE,
    VNAEngine,
    calculate_lft_vna,
    calculate_ntnb_vna,
    calculate_ntnc_vna,
    calculate_vna_from_vne,
)
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
    Suporta agenda padrão editalícia ou fluxos customizados fornecidos pelo usuário.
    """
    def __init__(
        self,
        maturity_date: Optional[Union[datetime.date, str]] = None,
        calendar_name: str = 'B3',
        custom_cashflows: Any = None,
        holidays: Any = None,
    ):
        self.calendar = get_calendar(calendar_name, holidays=holidays)
        self.custom_cashflows = custom_cashflows
        if maturity_date is None and custom_cashflows is not None:
            parsed = self._parse_custom_cashflows_raw(custom_cashflows)
            if not parsed:
                raise ValueError("custom_cashflows não pode ser vazio quando maturity_date é None.")
            maturity_date = max(d for d, _, _, _ in parsed)
        if isinstance(maturity_date, str):
            maturity_date = pd.to_datetime(maturity_date).date()
        self.nominal_maturity_date = maturity_date
        self.maturity_date = self.calendar.following(maturity_date)

    def _normalize_date(self, d: Union[datetime.date, str]) -> datetime.date:
        if isinstance(d, pd.Timestamp):
            return d.date()
        if isinstance(d, datetime.datetime):
            return d.date()
        if isinstance(d, str):
            return pd.to_datetime(d).date()
        return d

    def _parse_custom_cashflows_raw(self, custom_cashflows: Any) -> List[Tuple[datetime.date, str, float, bool]]:
        """
        Normaliza fluxos customizados em lista de (data, tipo, valor, is_cotacao_base100).
        Aceita DataFrame, Series, dict {data: valor} ou lista de tuplas/dicionários.
        """
        if custom_cashflows is None:
            return []
        rows: List[Tuple[datetime.date, str, float, bool]] = []
        if isinstance(custom_cashflows, pd.DataFrame):
            if custom_cashflows.empty:
                return []
            d_col = next(
                (c for c in custom_cashflows.columns if str(c).lower() in ("data_fluxo", "data_pagamento", "date", "data", "vencimento")),
                custom_cashflows.columns[0],
            )
            cot_col = next((c for c in custom_cashflows.columns if str(c).lower() in ("fluxo_cotação", "fluxo_cotacao", "cotacao")), None)
            nom_col = next(
                (c for c in custom_cashflows.columns if str(c).lower() in ("fluxo_nominal", "cashflow", "fluxo", "valor", "amount")),
                None,
            )
            is_cot = (nom_col is None and cot_col is not None)
            val_col = nom_col if nom_col is not None else (cot_col if cot_col is not None else custom_cashflows.columns[-1])
            tipo_col = next((c for c in custom_cashflows.columns if str(c).lower() in ("tipo", "evento", "type")), None)
            for _, r in custom_cashflows.iterrows():
                dt = self._normalize_date(r[d_col])
                tipo = str(r[tipo_col]) if tipo_col is not None else "Fluxo"
                rows.append((dt, tipo, float(r[val_col]), is_cot))
        elif isinstance(custom_cashflows, pd.Series):
            for k, v in custom_cashflows.items():
                rows.append((self._normalize_date(k), "Fluxo", float(v), False))
        elif isinstance(custom_cashflows, dict):
            for k, v in custom_cashflows.items():
                rows.append((self._normalize_date(k), "Fluxo", float(v), False))
        elif isinstance(custom_cashflows, (list, tuple)):
            for item in custom_cashflows:
                if isinstance(item, dict):
                    dt = self._normalize_date(item.get("data_fluxo", item.get("data_pagamento", item.get("date", item.get("data")))))
                    is_cot = ("fluxo_cotacao" in item or "fluxo_cotação" in item) and not any(
                        k in item for k in ("fluxo_nominal", "cashflow", "fluxo", "valor", "amount")
                    )
                    val = item.get("fluxo_nominal", item.get("cashflow", item.get("fluxo", item.get("valor", item.get("amount", item.get("fluxo_cotacao", item.get("fluxo_cotação", 0.0)))))))
                    tipo = str(item.get("tipo", item.get("evento", "Fluxo")))
                    rows.append((dt, tipo, float(val), is_cot))
                else:
                    dt = self._normalize_date(item[0])
                    val = float(item[1])
                    tipo = str(item[2]) if len(item) > 2 else "Fluxo"
                    rows.append((dt, tipo, val, False))
        else:
            raise TypeError(f"Formato não suportado para custom_cashflows: {type(custom_cashflows)}")
        return sorted(rows, key=lambda x: x[0])

    def _evaluate_custom_cashflows(
        self,
        base_date: datetime.date,
        curve: Curve,
        custom_cashflows: Any,
        vna: Optional[float] = None,
        rate_col_name: str = "taxa",
    ) -> pd.DataFrame:
        raw = self._parse_custom_cashflows_raw(custom_cashflows)
        rows = []
        vna_val = float(vna) if vna is not None else 0.0
        for dt, tipo, val, is_cot in raw:
            if dt <= base_date:
                continue
            if is_cot and vna_val <= 0.0:
                raise ValueError("Informe 'vna' positivo para avaliar custom_cashflows em base de cotação (`fluxo_cotação`).")
            du = int(self.calendar.bizdays(base_date, dt))
            rate = float(curve.get_rate(du)) if du > 0 else 0.0
            df_comp = math.trunc(math.pow(1.0 + rate, du / 252.0) * 1e14) / 1e14 if du > 0 else 1.0
            if is_cot:
                pv_cot = round((val / df_comp) * 1e10) / 1e10
                pv = math.trunc((vna_val * (pv_cot / 100.0)) * 1e6) / 1e6
                rows.append({
                    "data_fluxo": dt,
                    "du": du,
                    "tipo": tipo,
                    "fluxo_cotação": val,
                    rate_col_name: rate,
                    "fator_desconto": df_comp,
                    "cotação_descontada": pv_cot,
                    "vna": vna_val,
                    "valor_presente": pv,
                })
            else:
                pv = round((val / df_comp) * 1e9) / 1e9
                rows.append({
                    "data_fluxo": dt,
                    "du": du,
                    "tipo": tipo,
                    "fluxo_nominal": val,
                    rate_col_name: rate,
                    "fator_desconto": df_comp,
                    "valor_presente": pv,
                })
        return pd.DataFrame(rows)

    def _custom_metrics(self, base_date: datetime.date, curve: Curve, custom_cashflows: Any, vna: Optional[float] = None) -> dict:
        df = self._evaluate_custom_cashflows(base_date, curve, custom_cashflows, vna=vna)
        if df.empty:
            return {"df": df, "pu": 0.0, "duration": 0.0, "modified_duration": 0.0, "dv01": 0.0}
        vna_val = float(vna) if vna is not None else 0.0
        if "cotação_descontada" in df.columns and vna_val > 0.0:
            cot_total = math.trunc(float(df["cotação_descontada"].sum()) * 1e4) / 1e4
            pu = math.trunc((vna_val * (cot_total / 100.0)) * 1e6) / 1e6
            weights = df["cotação_descontada"]
        else:
            pu = math.trunc(float(df["valor_presente"].sum()) * 1e6) / 1e6
            weights = df["valor_presente"]
        w_sum = float(weights.sum())
        if pu <= 0.0 or w_sum <= 0.0:
            return {"df": df, "pu": pu, "duration": 0.0, "modified_duration": 0.0, "dv01": 0.0}
        rate_col = "taxa" if "taxa" in df.columns else "taxa_real"
        dur = float((df["du"] * weights).sum() / w_sum)
        mod_dur = float(((df["du"] / 252.0) / (1.0 + df[rate_col]) * weights).sum() / w_sum)
        dv01 = float(mod_dur * pu * 1e-4)
        return {"df": df, "pu": pu, "duration": dur, "modified_duration": mod_dur, "dv01": dv01}

    def _resolve_ntnb_vna(
        self,
        base_date: datetime.date,
        vna: Optional[float] = None,
        inflation_curve: Optional[Curve] = None,
        base_vna: Optional[float] = None,
        base_vna_date: Optional[Union[datetime.date, str]] = None,
        ipca_proj: Optional[float] = None,
        vna_engine: Optional[VNAEngine] = None,
        allow_download: bool = True,
        ipca_history: Any = None,
        ipca_projections: Any = None,
    ) -> float:
        if vna is not None:
            return float(vna)
        if inflation_curve is not None and base_vna is not None and base_vna_date is not None:
            b_dt = self._normalize_date(base_vna_date)
            du_inf = self.calendar.bizdays(b_dt, base_date)
            ipca_r = inflation_curve.get_rate(du_inf)
            ipca_f = math.trunc(math.pow(1.0 + ipca_r, du_inf / 252.0) * 1e14) / 1e14
            return float(math.trunc((float(base_vna) * ipca_f) * 1e6) / 1e6)
        if vna_engine is not None:
            return float(vna_engine.calculate_vna(base_date, cal=self.calendar, anniversary_day=15, ipca_proj=ipca_proj, truncate_monthly=False))
        return float(
            calculate_ntnb_vna(
                base_date,
                ipca_proj=ipca_proj,
                ipca_history=ipca_history,
                ipca_projections=ipca_projections,
                cal=self.calendar,
                allow_download=allow_download,
                truncate_monthly=False,
            )
        )


class LTN(TituloPublico):
    """
    Letra do Tesouro Nacional (título prefixado zero-cupom).
    Valor nominal no vencimento de R$ 1.000,00 (ou fluxos customizados do usuário).
    """
    def __init__(self, maturity_date: Optional[Union[datetime.date, str]] = None, face_value: float = 1000.0,
                 calendar_name: str = 'B3', custom_cashflows: Any = None, holidays: Any = None):
        super().__init__(maturity_date, calendar_name, custom_cashflows=custom_cashflows, holidays=holidays)
        self.face_value = face_value

    def cashflow_table(self, base_date: Union[datetime.date, str], curve: Curve, custom_cashflows: Any = None) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._evaluate_custom_cashflows(base_date, curve, cf)
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

    def present_value(self, base_date: Union[datetime.date, str], curve: Curve, return_cashflows: bool = False,
                      custom_cashflows: Any = None) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            res = self._custom_metrics(base_date, curve, cf)
            return (res["pu"], res["df"]) if return_cashflows else res["pu"]
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

    def duration(self, base_date: Union[datetime.date, str], curve: Curve = None, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            if curve is None:
                raise ValueError("Informe 'curve' para calcular duration de fluxos customizados.")
            return self._custom_metrics(base_date, curve, cf)["duration"]
        if base_date >= self.maturity_date:
            return 0.0
        return float(self.calendar.bizdays(base_date, self.maturity_date))

    def dv01(self, base_date: Union[datetime.date, str], curve: Curve, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, curve, cf)["dv01"]
        if base_date >= self.maturity_date:
            return 0.0
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _LTN_C(du, self.face_value)
        return bond_c.dv01(curve)


class NTNF(TituloPublico):
    """
    Nota do Tesouro Nacional - Série F (título prefixado com cupons semestrais).
    Taxa de cupom editalícia de 10,00% a.a. e cupons pagos em 01/01 e 01/07 (ou fluxos customizados).
    """
    def __init__(self, maturity_date: Optional[Union[datetime.date, str]] = None, coupon_rate: float = 0.10,
                 face_value: float = 1000.0, calendar_name: str = 'B3', custom_cashflows: Any = None,
                 holidays: Any = None):
        super().__init__(maturity_date, calendar_name, custom_cashflows=custom_cashflows, holidays=holidays)
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

    def cashflow_table(self, base_date: Union[datetime.date, str], curve: Curve, custom_cashflows: Any = None) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._evaluate_custom_cashflows(base_date, curve, cf)
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

    def present_value(self, base_date: Union[datetime.date, str], curve: Curve, return_cashflows: bool = False,
                      custom_cashflows: Any = None) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            res = self._custom_metrics(base_date, curve, cf)
            return (res["pu"], res["df"]) if return_cashflows else res["pu"]
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

    def duration(self, base_date: Union[datetime.date, str], curve: Curve, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, curve, cf)["duration"]
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNF_C(dus, self.coupon_rate, self.face_value)
        return bond_c.duration(curve)

    def modified_duration(self, base_date: Union[datetime.date, str], curve: Curve, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, curve, cf)["modified_duration"]
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNF_C(dus, self.coupon_rate, self.face_value)
        return bond_c.modified_duration(curve)

    def dv01(self, base_date: Union[datetime.date, str], curve: Curve, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, curve, cf)["dv01"]
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
    def __init__(self, maturity_date: Optional[Union[datetime.date, str]] = None, calendar_name: str = 'B3',
                 custom_cashflows: Any = None, holidays: Any = None):
        super().__init__(maturity_date, calendar_name, custom_cashflows=custom_cashflows, holidays=holidays)

    def cashflow_table(self, base_date: Union[datetime.date, str], agio_curve: Curve, vna: float = None,
                       custom_cashflows: Any = None) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._evaluate_custom_cashflows(base_date, agio_curve, cf, vna=vna, rate_col_name='taxa_ágio')
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

    def calculate_vna(
        self,
        base_date: Union[datetime.date, str],
        vne: float = LFT_BASE_VALUE,
        base_vna_date: Union[datetime.date, str] = LFT_BASE_DATE,
        daily_selic: Any = None,
        allow_download: bool = True,
    ) -> float:
        """Calcula o VNA da LFT a partir do VNE e da taxa Selic acumulada."""
        base_dt = self._normalize_date(base_date)
        return calculate_lft_vna(
            base_dt,
            vne=vne,
            base_date=base_vna_date,
            daily_selic=daily_selic,
            allow_download=allow_download,
        )

    def present_value(self, base_date: Union[datetime.date, str], agio_curve: Curve, vna: float = None,
                      selic_curve: Curve = None, return_cashflows: bool = False,
                      custom_cashflows: Any = None, daily_selic: Any = None,
                      allow_download: bool = True) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        vna_val = float(vna) if vna is not None else 0.0
        if vna_val <= 0.0 and selic_curve is None:
            vna_val = self.calculate_vna(base_date, daily_selic=daily_selic, allow_download=allow_download)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            res = self._custom_metrics(base_date, agio_curve, cf, vna=vna_val)
            return (res["pu"], res["df"]) if return_cashflows else res["pu"]
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _LFT_C(du)
        pu = bond_c.present_value(agio_curve, vna=vna_val, selic_curve=selic_curve)
        if return_cashflows:
            effective_vna = vna_val
            if effective_vna <= 0.0 and selic_curve is not None:
                selic_r = selic_curve.get_rate(du)
                effective_vna = math.trunc((1000.0 * math.trunc((1.0 + selic_r) ** (du / 252.0) * 1e14) / 1e14) * 1e6) / 1e6
            return pu, self.cashflow_table(base_date, agio_curve, vna=effective_vna)
        return pu

    def duration(self, base_date: Union[datetime.date, str], agio_curve: Curve = None, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            if agio_curve is None:
                raise ValueError("Informe 'agio_curve' para calcular duration de fluxos customizados.")
            return self._custom_metrics(base_date, agio_curve, cf)["duration"]
        if base_date >= self.maturity_date:
            return 0.0
        return float(self.calendar.bizdays(base_date, self.maturity_date))

    def dv01(self, base_date: Union[datetime.date, str], agio_curve: Curve, vna: float = None,
             daily_selic: Any = None, allow_download: bool = True, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        vna_val = float(vna) if vna is not None else self.calculate_vna(base_date, daily_selic=daily_selic, allow_download=allow_download)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, agio_curve, cf, vna=vna_val)["dv01"]
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _LFT_C(du)
        return bond_c.dv01(agio_curve, vna_val)


class NTNB(TituloPublico):
    """
    Nota do Tesouro Nacional - Série B (título indexado ao IPCA com cupons semestrais).
    Taxa de cupom editalícia de 6,00% a.a. aplicada sobre o Valor Nominal Atualizado (VNA).
    """
    def __init__(self, maturity_date: Optional[Union[datetime.date, str]] = None, coupon_rate: float = 0.06,
                 calendar_name: str = 'B3', custom_cashflows: Any = None, holidays: Any = None):
        super().__init__(maturity_date, calendar_name, custom_cashflows=custom_cashflows, holidays=holidays)
        self.coupon_rate = coupon_rate

    def calculate_vna(
        self,
        base_date: Union[datetime.date, str],
        ipca_proj: Optional[float] = None,
        vna_engine: Optional[VNAEngine] = None,
        inflation_curve: Optional[Curve] = None,
        base_vna: Optional[float] = None,
        base_vna_date: Optional[Union[datetime.date, str]] = None,
        allow_download: bool = True,
        ipca_history: Any = None,
        ipca_projections: Any = None,
    ) -> float:
        """Calcula o VNA oficial/projetado da NTN-B na data de referência."""
        base_dt = self._normalize_date(base_date)
        return self._resolve_ntnb_vna(
            base_dt,
            inflation_curve=inflation_curve,
            base_vna=base_vna,
            base_vna_date=base_vna_date,
            ipca_proj=ipca_proj,
            vna_engine=vna_engine,
            allow_download=allow_download,
            ipca_history=ipca_history,
            ipca_projections=ipca_projections,
        )

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
                       inflation_curve: Curve = None, base_vna: float = None, base_vna_date: Union[datetime.date, str] = None,
                       ipca_proj: Optional[float] = None, vna_engine: Optional[VNAEngine] = None,
                       allow_download: bool = True, custom_cashflows: Any = None,
                       ipca_history: Any = None, ipca_projections: Any = None) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        vna_val = self._resolve_ntnb_vna(
            base_date, vna=vna, inflation_curve=inflation_curve, base_vna=base_vna,
            base_vna_date=base_vna_date, ipca_proj=ipca_proj, vna_engine=vna_engine,
            allow_download=allow_download, ipca_history=ipca_history, ipca_projections=ipca_projections,
        )
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._evaluate_custom_cashflows(base_date, real_curve, cf, vna=vna_val, rate_col_name='taxa_real')

        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)
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
                      ipca_proj: Optional[float] = None, vna_engine: Optional[VNAEngine] = None,
                      allow_download: bool = True, return_cashflows: bool = False,
                      custom_cashflows: Any = None, ipca_history: Any = None,
                      ipca_projections: Any = None) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        vna_val = self._resolve_ntnb_vna(
            base_date, vna=vna, inflation_curve=inflation_curve, base_vna=base_vna,
            base_vna_date=base_vna_date, ipca_proj=ipca_proj, vna_engine=vna_engine,
            allow_download=allow_download, ipca_history=ipca_history, ipca_projections=ipca_projections,
        )
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            res = self._custom_metrics(base_date, real_curve, cf, vna=vna_val)
            return (res["pu"], res["df"]) if return_cashflows else res["pu"]

        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)

        pu = bond_c.present_value(real_curve, vna=vna_val)
        if return_cashflows:
            df_flows = self.cashflow_table(
                base_date=base_date,
                real_curve=real_curve,
                vna=vna_val,
                inflation_curve=inflation_curve,
                base_vna=base_vna,
                base_vna_date=base_vna_date,
                ipca_proj=ipca_proj,
                vna_engine=vna_engine,
                allow_download=allow_download,
                ipca_history=ipca_history,
                ipca_projections=ipca_projections,
            )
            return pu, df_flows
        return pu

    def duration(self, base_date: Union[datetime.date, str], real_curve: Curve, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, real_curve, cf, vna=100.0)["duration"]
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)
        return bond_c.duration(real_curve)

    def modified_duration(self, base_date: Union[datetime.date, str], real_curve: Curve, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, real_curve, cf, vna=100.0)["modified_duration"]
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)
        return bond_c.modified_duration(real_curve)

    def dv01(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None,
             ipca_proj: Optional[float] = None, vna_engine: Optional[VNAEngine] = None,
             allow_download: bool = True, custom_cashflows: Any = None,
             ipca_history: Any = None, ipca_projections: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        vna_val = self._resolve_ntnb_vna(
            base_date, vna=vna, ipca_proj=ipca_proj, vna_engine=vna_engine,
            allow_download=allow_download, ipca_history=ipca_history, ipca_projections=ipca_projections,
        )
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, real_curve, cf, vna=vna_val)["dv01"]
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNB_C(dus, self.coupon_rate)
        return bond_c.dv01(real_curve, vna=vna_val)


class NTNBPrincipal(TituloPublico):
    """
    Nota do Tesouro Nacional - Série B Principal (título indexado ao IPCA sem cupons intermediários).
    """
    def __init__(self, maturity_date: Optional[Union[datetime.date, str]] = None, calendar_name: str = 'B3',
                 custom_cashflows: Any = None, holidays: Any = None):
        super().__init__(maturity_date, calendar_name, custom_cashflows=custom_cashflows, holidays=holidays)

    def calculate_vna(
        self,
        base_date: Union[datetime.date, str],
        ipca_proj: Optional[float] = None,
        vna_engine: Optional[VNAEngine] = None,
        inflation_curve: Optional[Curve] = None,
        base_vna: Optional[float] = None,
        base_vna_date: Optional[Union[datetime.date, str]] = None,
        allow_download: bool = True,
        ipca_history: Any = None,
        ipca_projections: Any = None,
    ) -> float:
        """Calcula o VNA oficial/projetado da NTN-B Principal na data de referência."""
        base_dt = self._normalize_date(base_date)
        return self._resolve_ntnb_vna(
            base_dt,
            inflation_curve=inflation_curve,
            base_vna=base_vna,
            base_vna_date=base_vna_date,
            ipca_proj=ipca_proj,
            vna_engine=vna_engine,
            allow_download=allow_download,
            ipca_history=ipca_history,
            ipca_projections=ipca_projections,
        )

    def cashflow_table(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None,
                       inflation_curve: Curve = None, base_vna: float = None, base_vna_date: Union[datetime.date, str] = None,
                       ipca_proj: Optional[float] = None, vna_engine: Optional[VNAEngine] = None,
                       allow_download: bool = True, custom_cashflows: Any = None,
                       ipca_history: Any = None, ipca_projections: Any = None) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        vna_val = self._resolve_ntnb_vna(
            base_date, vna=vna, inflation_curve=inflation_curve, base_vna=base_vna,
            base_vna_date=base_vna_date, ipca_proj=ipca_proj, vna_engine=vna_engine,
            allow_download=allow_download, ipca_history=ipca_history, ipca_projections=ipca_projections,
        )
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._evaluate_custom_cashflows(base_date, real_curve, cf, vna=vna_val, rate_col_name='taxa_real')
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _NTNBP_C(du)
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
                      ipca_proj: Optional[float] = None, vna_engine: Optional[VNAEngine] = None,
                      allow_download: bool = True, return_cashflows: bool = False,
                      custom_cashflows: Any = None, ipca_history: Any = None,
                      ipca_projections: Any = None) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        vna_val = self._resolve_ntnb_vna(
            base_date, vna=vna, inflation_curve=inflation_curve, base_vna=base_vna,
            base_vna_date=base_vna_date, ipca_proj=ipca_proj, vna_engine=vna_engine,
            allow_download=allow_download, ipca_history=ipca_history, ipca_projections=ipca_projections,
        )
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            res = self._custom_metrics(base_date, real_curve, cf, vna=vna_val)
            return (res["pu"], res["df"]) if return_cashflows else res["pu"]

        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _NTNBP_C(du)
        pu = bond_c.present_value(real_curve, vna=vna_val)
        if return_cashflows:
            return pu, self.cashflow_table(base_date, real_curve, vna=vna_val)
        return pu

    def duration(self, base_date: Union[datetime.date, str], real_curve: Curve = None, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            if real_curve is None:
                raise ValueError("Informe 'real_curve' para calcular duration de fluxos customizados.")
            return self._custom_metrics(base_date, real_curve, cf, vna=100.0)["duration"]
        if base_date >= self.maturity_date:
            return 0.0
        return float(self.calendar.bizdays(base_date, self.maturity_date))

    def modified_duration(self, base_date: Union[datetime.date, str], real_curve: Curve, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, real_curve, cf, vna=100.0)["modified_duration"]
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _NTNBP_C(du)
        return bond_c.modified_duration(real_curve)

    def dv01(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None,
             ipca_proj: Optional[float] = None, vna_engine: Optional[VNAEngine] = None,
             allow_download: bool = True, custom_cashflows: Any = None,
             ipca_history: Any = None, ipca_projections: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        vna_val = self._resolve_ntnb_vna(
            base_date, vna=vna, ipca_proj=ipca_proj, vna_engine=vna_engine,
            allow_download=allow_download, ipca_history=ipca_history, ipca_projections=ipca_projections,
        )
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, real_curve, cf, vna=vna_val)["dv01"]
        du = self.calendar.bizdays(base_date, self.maturity_date)
        bond_c = _NTNBP_C(du)
        return bond_c.dv01(real_curve, vna=vna_val)


class NTNC(TituloPublico):
    """
    Nota do Tesouro Nacional - Série C (título indexado ao IGP-M com cupons semestrais).
    Taxa de cupom editalícia de 6,00% ou 12,00% a.a. (padrão 12% para série 2031) aplicada sobre o VNA do IGP-M.
    """
    def __init__(self, maturity_date: Optional[Union[datetime.date, str]] = None, coupon_rate: Optional[float] = None,
                 calendar_name: str = 'B3', custom_cashflows: Any = None, holidays: Any = None):
        super().__init__(maturity_date, calendar_name, custom_cashflows=custom_cashflows, holidays=holidays)
        if coupon_rate is None:
            if self.nominal_maturity_date is not None and self.nominal_maturity_date.year == 2031:
                self.coupon_rate = 0.12
            else:
                self.coupon_rate = 0.06
        else:
            self.coupon_rate = float(coupon_rate)

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

    def calculate_vna(
        self,
        base_date: Union[datetime.date, str],
        vne: float = NTNC_BASE_VALUE,
        base_vna_date: Union[datetime.date, str] = NTNC_BASE_DATE,
        igpm_proj: Optional[float] = None,
        vna_engine: Optional[VNAEngine] = None,
        inflation_curve: Optional[Curve] = None,
        allow_download: bool = True,
        igpm_history: Any = None,
        igpm_projections: Any = None,
        base_index: Optional[float] = None,
        current_index: Optional[float] = None,
    ) -> float:
        """Calcula o VNA da NTN-C a partir do VNE/data-base pelo IGP-M (aniversário dia 1)."""
        base_dt = self._normalize_date(base_date)
        if inflation_curve is not None:
            b_dt = self._normalize_date(base_vna_date)
            du_inf = self.calendar.bizdays(b_dt, base_dt)
            igpm_r = inflation_curve.get_rate(du_inf)
            igpm_f = math.trunc(math.pow(1.0 + igpm_r, du_inf / 252.0) * 1e14) / 1e14
            return float(math.trunc((float(vne) * igpm_f) * 1e6) / 1e6)
        if vna_engine is not None:
            return float(vna_engine.calculate_vna(base_dt, cal=self.calendar, anniversary_day=1, ipca_proj=igpm_proj, truncate_monthly=False))
        return float(
            calculate_ntnc_vna(
                base_dt,
                vne=vne,
                base_date=base_vna_date,
                igpm_proj=igpm_proj,
                igpm_history=igpm_history,
                igpm_projections=igpm_projections,
                cal=self.calendar,
                allow_download=allow_download,
                base_index=base_index,
                current_index=current_index,
            )
        )

    def cashflow_table(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None,
                       inflation_curve: Curve = None, base_vna: float = None, base_vna_date: Union[datetime.date, str] = None,
                       igpm_proj: Optional[float] = None, vna_engine: Optional[VNAEngine] = None,
                       allow_download: bool = True, custom_cashflows: Any = None,
                       igpm_history: Any = None, igpm_projections: Any = None,
                       base_index: Optional[float] = None, current_index: Optional[float] = None) -> pd.DataFrame:
        base_date = self._normalize_date(base_date)
        if vna is not None:
            vna_val = float(vna)
        else:
            vna_val = self.calculate_vna(
                base_date,
                vne=float(base_vna) if base_vna is not None else NTNC_BASE_VALUE,
                base_vna_date=base_vna_date if base_vna_date is not None else NTNC_BASE_DATE,
                igpm_proj=igpm_proj,
                vna_engine=vna_engine,
                inflation_curve=inflation_curve,
                allow_download=allow_download,
                igpm_history=igpm_history,
                igpm_projections=igpm_projections,
                base_index=base_index,
                current_index=current_index,
            )
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._evaluate_custom_cashflows(base_date, real_curve, cf, vna=vna_val, rate_col_name='taxa_real')

        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)
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
                      igpm_proj: Optional[float] = None, vna_engine: Optional[VNAEngine] = None,
                      allow_download: bool = True, return_cashflows: bool = False,
                      custom_cashflows: Any = None, igpm_history: Any = None,
                      igpm_projections: Any = None, base_index: Optional[float] = None,
                      current_index: Optional[float] = None) -> Union[float, Tuple[float, pd.DataFrame]]:
        base_date = self._normalize_date(base_date)
        vna_val = (
            float(vna)
            if vna is not None
            else self.calculate_vna(
                base_date,
                vne=float(base_vna) if base_vna is not None else NTNC_BASE_VALUE,
                base_vna_date=base_vna_date if base_vna_date is not None else NTNC_BASE_DATE,
                igpm_proj=igpm_proj,
                vna_engine=vna_engine,
                inflation_curve=inflation_curve,
                allow_download=allow_download,
                igpm_history=igpm_history,
                igpm_projections=igpm_projections,
                base_index=base_index,
                current_index=current_index,
            )
        )
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            res = self._custom_metrics(base_date, real_curve, cf, vna=vna_val)
            return (res["pu"], res["df"]) if return_cashflows else res["pu"]

        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)
        pu = bond_c.present_value(real_curve, vna=vna_val)
        if return_cashflows:
            df_flows = self.cashflow_table(
                base_date=base_date,
                real_curve=real_curve,
                vna=vna_val,
                inflation_curve=inflation_curve,
                base_vna=base_vna,
                base_vna_date=base_vna_date,
                igpm_proj=igpm_proj,
                vna_engine=vna_engine,
                allow_download=allow_download,
                igpm_history=igpm_history,
                igpm_projections=igpm_projections,
                base_index=base_index,
                current_index=current_index,
            )
            return pu, df_flows
        return pu

    def duration(self, base_date: Union[datetime.date, str], real_curve: Curve, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, real_curve, cf, vna=100.0)["duration"]
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)
        return bond_c.duration(real_curve)

    def modified_duration(self, base_date: Union[datetime.date, str], real_curve: Curve, custom_cashflows: Any = None) -> float:
        base_date = self._normalize_date(base_date)
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, real_curve, cf, vna=100.0)["modified_duration"]
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)
        return bond_c.modified_duration(real_curve)

    def dv01(self, base_date: Union[datetime.date, str], real_curve: Curve, vna: float = None,
             igpm_proj: Optional[float] = None, vna_engine: Optional[VNAEngine] = None,
             allow_download: bool = True, custom_cashflows: Any = None,
             igpm_history: Any = None, igpm_projections: Any = None,
             base_index: Optional[float] = None, current_index: Optional[float] = None) -> float:
        base_date = self._normalize_date(base_date)
        vna_val = (
            float(vna)
            if vna is not None
            else self.calculate_vna(
                base_date,
                igpm_proj=igpm_proj,
                vna_engine=vna_engine,
                allow_download=allow_download,
                igpm_history=igpm_history,
                igpm_projections=igpm_projections,
                base_index=base_index,
                current_index=current_index,
            )
        )
        cf = custom_cashflows if custom_cashflows is not None else self.custom_cashflows
        if cf is not None:
            return self._custom_metrics(base_date, real_curve, cf, vna=vna_val)["dv01"]
        if base_date >= self.maturity_date:
            return 0.0
        coupon_dates = self.get_coupon_dates(base_date)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        bond_c = _NTNC_C(dus, self.coupon_rate)
        return bond_c.dv01(real_curve, vna=vna_val)
