import datetime
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from bz_pricing.core.calendars import get_calendar
from bz_pricing.core.curves import Curve
from bz_pricing.data.debentures import DebenturesClient


class Debenture:
    """
    Instrumento de crédito privado estruturado (Debênture) conforme metodologia ANBIMA e SND.
    Suporta ativos indexados ao IPCA, IGP-M, DI (+ spread ou % do DI) e Prefixados,
    com amortizações ordinárias/extraordinárias e fluxos de caixa periódicos.
    """

    def __init__(self,
                 ticker: str,
                 maturity_date: Union[datetime.date, str] = None,
                 indexador: str = "IPCA",
                 coupon_rate: float = 0.0,
                 issue_date: Union[datetime.date, str] = None,
                 cashflows_df: Optional[pd.DataFrame] = None,
                 nominal_value: float = 1000.0,
                 calendar_name: str = "B3"):
        self.ticker = ticker.strip().upper()
        self.indexador = indexador.strip().upper()
        self.coupon_rate = coupon_rate
        self.nominal_value = nominal_value
        self.calendar = get_calendar(calendar_name)

        if maturity_date:
            self.nominal_maturity_date = self._normalize_date(maturity_date)
            self.maturity_date = self.calendar.following(self.nominal_maturity_date)
        else:
            self.nominal_maturity_date = None
            self.maturity_date = None

        self.issue_date = self._normalize_date(issue_date) if issue_date else None
        if cashflows_df is not None and not cashflows_df.empty:
            df_copy = cashflows_df.copy()
            for c in ["data_evento", "data_pagamento"]:
                if c in df_copy.columns:
                    df_copy[c] = pd.to_datetime(df_copy[c]).dt.date
            self.cashflows_df = df_copy
        else:
            self.cashflows_df = cashflows_df

    def _normalize_date(self, d: Union[datetime.date, str]) -> datetime.date:
        if isinstance(d, str):
            return pd.to_datetime(d, dayfirst=True).date()
        return d

    @classmethod
    def from_ticker(cls, ticker: str, client: Optional[DebenturesClient] = None,
                    calendar_name: str = "B3") -> "Debenture":
        """
        Constrói uma instância de Debenture a partir dos dados extraídos do SND
        (agenda de eventos futuros, eventos passados liquidados e características cadastrais).
        """
        if client is None:
            client = DebenturesClient()

        carac = client.get_characteristics(ticker)
        future_cf = client.get_cashflows(ticker)
        past_events = client.get_event_pus(ticker)

        # Mescla eventos passados e futuros para compor o ciclo integral do ativo
        combined_rows = []
        if not past_events.empty:
            for _, r in past_events.iterrows():
                combined_rows.append({
                    "data_evento": r["data"],
                    "data_pagamento": r["data"],
                    "emissor": carac.get("Empresa", ""),
                    "ativo": ticker,
                    "evento": r["tipo_evento"],
                    "tipo": "",
                    "taxa_percentual": 0.0,
                    "liquidado": r.get("liquidacao", "LIQUIDADO")
                })

        if not future_cf.empty:
            for _, r in future_cf.iterrows():
                combined_rows.append({
                    "data_evento": r["data_evento"],
                    "data_pagamento": r["data_pagamento"],
                    "emissor": r.get("emissor", carac.get("Empresa", "")),
                    "ativo": ticker,
                    "evento": r["evento"],
                    "tipo": r.get("tipo", ""),
                    "taxa_percentual": r.get("taxa_percentual", 0.0),
                    "liquidado": r.get("liquidado", "")
                })

        if combined_rows:
            all_cf_df = pd.DataFrame(combined_rows)
            for col in ["data_evento", "data_pagamento"]:
                all_cf_df[col] = pd.to_datetime(all_cf_df[col]).dt.date
            all_cf_df = (
                all_cf_df.drop_duplicates(subset=["data_pagamento", "evento"])
                .sort_values(by=["data_pagamento", "data_evento"])
                .reset_index(drop=True)
            )
        else:
            all_cf_df = future_cf

        vcto_str = carac.get("Data de Vencimento") or carac.get("Data de Saida / Novo Vencimento")
        emissao_str = carac.get("Data de Emissao")
        indice = carac.get("indice", "IPCA")

        # Taxa de juros do cupom cadastrado
        taxa_str = carac.get("Juros Criterio Novo - Taxa") or "0"
        try:
            taxa_num = float(str(taxa_str).replace(".", "").replace(",", ".")) / 100.0
        except ValueError:
            taxa_num = 0.0

        vne_str = carac.get("Valor Nominal na Emissao") or "1000"
        try:
            vne_num = float(str(vne_str).replace(".", "").replace(",", "."))
        except ValueError:
            vne_num = 1000.0

        return cls(
            ticker=ticker,
            maturity_date=vcto_str,
            indexador=indice if indice else "IPCA",
            coupon_rate=taxa_num,
            issue_date=emissao_str,
            cashflows_df=all_cf_df,
            nominal_value=vne_num,
            calendar_name=calendar_name
        )

    def _get_rate_for_du(self, discount_rate_or_curve: Union[float, Curve], du: int) -> float:
        if isinstance(discount_rate_or_curve, (int, float)):
            return float(discount_rate_or_curve)
        elif hasattr(discount_rate_or_curve, "get_rate"):
            return discount_rate_or_curve.get_rate(float(du))
        raise TypeError("A taxa de desconto deve ser um float ou um objeto compilado de Curve.")

    def pu_par(self, base_date: Union[datetime.date, str], vna: float,
               last_coupon_date: Union[datetime.date, str] = None) -> float:
        """
        Calcula o PU Par na data de referência:
        PU Par = VNA * (1 + Juros_Acruados)
        """
        base_date = self._normalize_date(base_date)
        vna = float(vna)

        if self.coupon_rate <= 0.0:
            return vna

        # Identifica a data do último pagamento de cupom anterior a base_date
        if last_coupon_date is not None:
            last_date = self._normalize_date(last_coupon_date)
        elif self.cashflows_df is not None and not self.cashflows_df.empty:
            past_coupons = self.cashflows_df[
                (self.cashflows_df["data_pagamento"] <= base_date) &
                (self.cashflows_df["evento"].str.contains("Juros", case=False, na=False))
            ]
            if not past_coupons.empty:
                last_date = past_coupons["data_pagamento"].iloc[-1]
            else:
                last_date = self.issue_date if self.issue_date else base_date
        else:
            last_date = base_date

        du_passado = self.calendar.bizdays(last_date, base_date)
        if du_passado <= 0:
            return vna

        # Juros acruados pró-rata dia útil (A-9)
        f_juros = math.pow(1.0 + self.coupon_rate, du_passado / 252.0) - 1.0
        juros_val = math.trunc((vna * f_juros) * 1e6) / 1e6
        return vna + juros_val

    def cashflow_table(self, base_date: Union[datetime.date, str],
                       discount_rate_or_curve: Union[float, Curve],
                       vna: float,
                       last_coupon_date: Union[datetime.date, str] = None) -> pd.DataFrame:
        """
        Retorna a decomposição analítica fluxo a fluxo dos eventos futuros de juros e amortização,
        com as respectivas taxas, fatores de desconto T-14 e valores presentes.
        """
        base_date = self._normalize_date(base_date)
        vna = float(vna)

        if self.cashflows_df is None or self.cashflows_df.empty:
            raise ValueError(f"Não há agenda de fluxos carregada para a debênture {self.ticker}.")

        # Filtra os eventos com liquidação estritamente posterior à data-base
        future_df = self.cashflows_df[self.cashflows_df["data_pagamento"] > base_date].copy()
        if future_df.empty:
            return pd.DataFrame(columns=[
                "data_evento", "data_pagamento", "du", "evento", "tipo",
                "taxa_cupom", "fluxo_nominal", "taxa_desconto", "fator_desconto", "valor_presente"
            ])

        if last_coupon_date is not None:
            prev_event_date = self._normalize_date(last_coupon_date)
        else:
            past_events = self.cashflows_df[self.cashflows_df["data_pagamento"] <= base_date]
            if not past_events.empty:
                prev_event_date = past_events["data_pagamento"].iloc[-1]
            else:
                prev_event_date = self.issue_date if self.issue_date else base_date

        rows = []
        saldo_vna = vna

        for _, row in future_df.iterrows():
            d_ev = row["data_evento"]
            d_pg = row["data_pagamento"]
            evento = str(row["evento"]).strip()
            tipo = str(row["tipo"]).strip()
            perc = float(row.get("taxa_percentual", 0.0))

            du_desc = self.calendar.bizdays(base_date, d_pg)
            r_desc = self._get_rate_for_du(discount_rate_or_curve, du_desc)

            # Fator de desconto T-14
            df_val = math.trunc(math.pow(1.0 + r_desc, du_desc / 252.0) * 1e14) / 1e14

            if "Juros" in evento:
                # O período contratual de contagem de juros ocorre entre as datas de liquidação (d_pg)
                du_period = self.calendar.bizdays(prev_event_date, d_pg)
                r_cupom = (perc / 100.0) if perc > 0.0 else self.coupon_rate
                f_cupom = math.trunc((math.pow(1.0 + r_cupom, du_period / 252.0) - 1.0) * 1e9) / 1e9
                valor_fluxo = saldo_vna * f_cupom
                pv_val = valor_fluxo / df_val

                rows.append({
                    "data_evento": d_ev,
                    "data_pagamento": d_pg,
                    "du": du_desc,
                    "evento": "Juros",
                    "tipo": tipo,
                    "taxa_cupom": r_cupom,
                    "fluxo_nominal": valor_fluxo,
                    "taxa_desconto": r_desc,
                    "fator_desconto": df_val,
                    "valor_presente": pv_val
                })
                prev_event_date = d_pg

            elif evento in ["Amortização", "Vencimento"]:
                if evento == "Vencimento" or perc <= 0.0 or perc >= 100.0:
                    valor_fluxo = saldo_vna
                    saldo_vna = 0.0
                else:
                    valor_fluxo = saldo_vna * (perc / 100.0)
                    saldo_vna -= valor_fluxo

                pv_val = valor_fluxo / df_val
                rows.append({
                    "data_evento": d_ev,
                    "data_pagamento": d_pg,
                    "du": du_desc,
                    "evento": evento,
                    "tipo": tipo,
                    "taxa_cupom": 0.0,
                    "fluxo_nominal": valor_fluxo,
                    "taxa_desconto": r_desc,
                    "fator_desconto": df_val,
                    "valor_presente": pv_val
                })

        return pd.DataFrame(rows)

    def present_value(self, base_date: Union[datetime.date, str],
                      discount_rate_or_curve: Union[float, Curve],
                      vna: float,
                      last_coupon_date: Union[datetime.date, str] = None,
                      return_cashflows: bool = False) -> Union[float, Tuple[float, pd.DataFrame]]:
        """
        Calcula o Valor Presente (PU da Operação marcado a mercado) da debênture.
        Se return_cashflows=True, retorna a tupla (pu, df_fluxos).
        """
        df_flows = self.cashflow_table(
            base_date=base_date,
            discount_rate_or_curve=discount_rate_or_curve,
            vna=vna,
            last_coupon_date=last_coupon_date
        )

        if df_flows.empty:
            pu = float(vna)
        else:
            pu = math.trunc(df_flows["valor_presente"].sum() * 1e6) / 1e6

        if return_cashflows:
            return pu, df_flows
        return pu

    def cotacao(self, base_date: Union[datetime.date, str],
                discount_rate_or_curve: Union[float, Curve],
                vna: float,
                last_coupon_date: Union[datetime.date, str] = None) -> float:
        """
        Calcula a cotação percentual sobre o PU Par (% PU Par):
        Cotação = (PU_Operacao / PU_Par) * 100
        """
        pu_par_val = self.pu_par(base_date, vna, last_coupon_date=last_coupon_date)
        if pu_par_val <= 0.0:
            return 100.0
        pu_val = self.present_value(base_date, discount_rate_or_curve, vna, last_coupon_date=last_coupon_date)
        return math.trunc((pu_val / pu_par_val) * 1e6) / 1e4

    def duration(self, base_date: Union[datetime.date, str],
                 discount_rate_or_curve: Union[float, Curve],
                 vna: float,
                 last_coupon_date: Union[datetime.date, str] = None) -> float:
        """
        Calcula a Macaulay Duration em dias úteis (DU).
        """
        df_flows = self.cashflow_table(
            base_date=base_date,
            discount_rate_or_curve=discount_rate_or_curve,
            vna=vna,
            last_coupon_date=last_coupon_date
        )
        if df_flows.empty:
            return 0.0

        total_pv = df_flows["valor_presente"].sum()
        if total_pv <= 0.0:
            return 0.0

        dur = (df_flows["du"] * df_flows["valor_presente"]).sum() / total_pv
        return round(float(dur), 2)
