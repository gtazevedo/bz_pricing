import datetime
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from bz_pricing.core.calendars import get_calendar
from bz_pricing.core.curves import Curve
from bz_pricing.core.vna_engine import VNAEngine
from bz_pricing.data.cricra import CRICRAClient
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
                 calendar_name: str = "B3",
                 multiplicador_di: float = 1.0,
                 amortizacao_sobre_vne: bool = False,
                 coupon_day_count: str = "BUS/252",
                 defasagem_di: int = 0,
                 anniversary_day: Optional[int] = None,
                 vna_engine: Optional[VNAEngine] = None):
        self.ticker = ticker.strip().upper()
        self.indexador = indexador.strip().upper()
        self.coupon_rate = 0.0 if (coupon_rate is None or (isinstance(coupon_rate, float) and math.isnan(coupon_rate))) else float(coupon_rate)
        self.nominal_value = float(nominal_value)
        self.calendar = get_calendar(calendar_name)
        self.multiplicador_di = float(multiplicador_di)
        self.amortizacao_sobre_vne = bool(amortizacao_sobre_vne)
        self.coupon_day_count = coupon_day_count.strip().upper()
        self.defasagem_di = int(defasagem_di)
        self.vna_engine = vna_engine

        if maturity_date:
            self.nominal_maturity_date = self._normalize_date(maturity_date)
            self.maturity_date = self.calendar.following(self.nominal_maturity_date)
        else:
            self.nominal_maturity_date = None
            self.maturity_date = None

        self.issue_date = self._normalize_date(issue_date) if issue_date else None
        if anniversary_day is not None:
            self.anniversary_day = int(anniversary_day)
        elif self.issue_date is not None:
            self.anniversary_day = self.issue_date.day
        elif self.nominal_maturity_date is not None:
            self.anniversary_day = self.nominal_maturity_date.day
        else:
            self.anniversary_day = 15
        if cashflows_df is not None and not cashflows_df.empty:
            df_copy = cashflows_df.copy()
            for c in ["data_evento", "data_pagamento"]:
                if c in df_copy.columns:
                    df_copy[c] = pd.to_datetime(df_copy[c]).dt.date
            self.cashflows_df = df_copy
        else:
            self.cashflows_df = cashflows_df

    @staticmethod
    def _parse_br_float(value, default: float = 0.0) -> float:
        """Converte strings numéricas no padrão brasileiro ou floats para float."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if not s or s.lower() == "nan":
            return default
        if "," in s:
            s = s.replace(".", "").replace(",", ".")
        try:
            val = float(s)
            return default if math.isnan(val) else val
        except ValueError:
            return default

    def _is_di_linked(self) -> bool:
        """Indica se a debênture possui remuneração atrelada ao DI."""
        return "DI" in self.indexador and "IGP" not in self.indexador and "IPCA" not in self.indexador

    def _is_pct_di(self) -> bool:
        """Indica se a debênture é remunerada exclusivamente por percentual do DI."""
        return self._is_di_linked() and self.coupon_rate <= 0.0 and self.multiplicador_di != 1.0

    def _coupon_year_fraction(self, start_date: datetime.date, end_date: datetime.date) -> float:
        """Calcula a fração de ano contratual para apuração de juros do cupom."""
        if start_date >= end_date:
            return 0.0
        dc = self.coupon_day_count
        if dc in ("BUS/252", "DU/252"):
            return self.calendar.bizdays(start_date, end_date) / 252.0
        if dc in ("ACT/360", "DC/360"):
            return (end_date - start_date).days / 360.0
        if dc in ("ACT/365", "DC/365"):
            return (end_date - start_date).days / 365.0
        if dc == "30/360":
            d1 = min(start_date.day, 30)
            d2 = 30 if (end_date.day == 31 and d1 == 30) else end_date.day
            days_30 = (end_date.year - start_date.year) * 360 + (end_date.month - start_date.month) * 30 + (d2 - d1)
            return days_30 / 360.0
        raise ValueError(f"Convenção de contagem de dias '{self.coupon_day_count}' não suportada.")

    def _fetch_di_factor(self, start_date: datetime.date, end_date: datetime.date,
                         multiplicador: float = 1.0, round_factor_8: bool = False) -> float:
        """
        Calcula o fator de DI acumulado no intervalo [start_date, end_date)
        via série SGS 12 do Banco Central (taxa DI diária), respeitando defasagem_di.
        """
        if start_date >= end_date:
            return 1.0
        if self.defasagem_di > 0:
            start_date = self.calendar.offset(start_date, -self.defasagem_di)
            end_date = self.calendar.offset(end_date, -self.defasagem_di)
        from bcb import sgs

        df_cdi = sgs.get(
            {"CDI": 12},
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d")
        )
        if df_cdi is None or df_cdi.empty:
            raise ValueError(f"Série SGS 12 (CDI) indisponível para o período {start_date} a {end_date}.")

        dates = pd.to_datetime(df_cdi.index).date
        mask = (dates >= start_date) & (dates < end_date)
        rates = df_cdi.loc[mask, "CDI"].dropna()

        fator = 1.0
        for r in rates:
            fator *= (1.0 + (float(r) / 100.0) * multiplicador)
        if round_factor_8:
            fator = round(fator, 8)
        return float(fator)

    def _normalize_date(self, d: Union[datetime.date, str]) -> datetime.date:
        if isinstance(d, str):
            return pd.to_datetime(d, dayfirst="/" in d).date()
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

        # Taxa de juros do cupom cadastrado, VNE e multiplicador DI
        taxa_num = cls._parse_br_float(carac.get("Juros Criterio Novo - Taxa"), default=0.0) / 100.0
        vne_num = cls._parse_br_float(carac.get("Valor Nominal na Emissao"), default=1000.0)
        mult_num = cls._parse_br_float(carac.get("Percentual Multiplicador/Rentabilidade"), default=100.0) / 100.0
        if mult_num <= 0.0:
            mult_num = 1.0

        tipo_amort = str(carac.get("Tipo de Amortizacao") or "").lower()
        amort_vne = "emiss" in tipo_amort

        return cls(
            ticker=ticker,
            maturity_date=vcto_str,
            indexador=indice if indice else "IPCA",
            coupon_rate=taxa_num,
            issue_date=emissao_str,
            cashflows_df=all_cf_df,
            nominal_value=vne_num,
            calendar_name=calendar_name,
            multiplicador_di=mult_num,
            amortizacao_sobre_vne=amort_vne
        )

    def _get_rate_for_du(self, discount_rate_or_curve: Union[float, Curve], du: int) -> float:
        if isinstance(discount_rate_or_curve, (int, float)):
            return float(discount_rate_or_curve)
        elif hasattr(discount_rate_or_curve, "get_rate"):
            return discount_rate_or_curve.get_rate(float(du))
        raise TypeError("A taxa de desconto deve ser um float ou um objeto compilado de Curve.")

    def resolve_vna(self, base_date: Union[datetime.date, str],
                    vna: Optional[float] = None,
                    vna_engine: Optional[VNAEngine] = None) -> float:
        """
        Retorna o VNA na data de referência. Se não for informado explicitamente,
        utiliza o valor nominal para papéis não corrigidos por inflação ou calcula
        via VNAEngine (IPCA/IGP-M) a partir da data de emissão e aniversário.
        """
        if vna is not None:
            return float(vna)
        if "IPCA" not in self.indexador and "IGP" not in self.indexador:
            return float(self.nominal_value)

        base_dt = self._normalize_date(base_date)
        engine = vna_engine or self.vna_engine
        if engine is None:
            if self.issue_date is None:
                raise ValueError(
                    f"Para calcular o VNA automático de {self.ticker} ({self.indexador}), "
                    "informe 'issue_date' ou passe 'vna' / 'vna_engine' explicitamente."
                )
            engine = VNAEngine.from_bcb(
                base_vna_date=self.issue_date,
                base_vna_value=self.nominal_value,
                indicator=self.indexador,
                ref_date=base_dt,
            )
            self.vna_engine = engine
        return float(engine.calculate_vna(base_dt, self.calendar, anniversary_day=self.anniversary_day))

    def pu_par(self, base_date: Union[datetime.date, str], vna: Optional[float] = None,
               last_coupon_date: Union[datetime.date, str] = None,
               fator_di_acumulado: Optional[float] = None) -> float:
        """
        Calcula o PU Par na data de referência:
        PU Par = VNA * (1 + Juros_Acruados)
        """
        base_date = self._normalize_date(base_date)
        vna = self.resolve_vna(base_date, vna=vna)

        if not self._is_di_linked() and self.coupon_rate <= 0.0:
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

        if self._is_pct_di():
            f_di = (
                float(fator_di_acumulado)
                if fator_di_acumulado is not None
                else self._fetch_di_factor(last_date, base_date, multiplicador=self.multiplicador_di, round_factor_8=False)
            )
            return math.trunc((vna * f_di) * 1e6) / 1e6

        if self._is_di_linked():
            f_di = (
                float(fator_di_acumulado)
                if fator_di_acumulado is not None
                else self._fetch_di_factor(last_date, base_date, multiplicador=1.0, round_factor_8=True)
            )
            f_spr = round(math.pow(1.0 + self.coupon_rate, du_passado / 252.0), 9)
            f_juros = math.trunc(f_di * f_spr * 1e9) / 1e9
            return math.trunc((vna * f_juros) * 1e6) / 1e6

        # Juros acruados pró-rata (A-9) para IPCA, IGP-M e Prefixado
        yf_passado = self._coupon_year_fraction(last_date, base_date)
        f_juros = math.pow(1.0 + self.coupon_rate, yf_passado) - 1.0
        juros_val = math.trunc((vna * f_juros) * 1e6) / 1e6
        return vna + juros_val

    def cashflow_table(self, base_date: Union[datetime.date, str],
                       discount_rate_or_curve: Union[float, Curve],
                       vna: Optional[float] = None,
                       last_coupon_date: Union[datetime.date, str] = None,
                       di_curve: Optional[Union[float, Curve]] = None,
                       pu_par_val: Optional[float] = None,
                       fator_di_acumulado: Optional[float] = None) -> pd.DataFrame:
        """
        Retorna a decomposição analítica fluxo a fluxo dos eventos futuros de juros e amortização,
        com as respectivas taxas, fatores de desconto T-14 e valores presentes.
        Para debêntures indexadas ao DI (DI + Spread ou % do DI), exige o parâmetro 'di_curve'.
        """
        base_date = self._normalize_date(base_date)
        vna = self.resolve_vna(base_date, vna=vna)
        is_di = self._is_di_linked()
        is_pct_di = self._is_pct_di()

        if is_di and di_curve is None:
            raise ValueError(f"Para debêntures indexadas ao DI ({self.ticker}), informe 'di_curve'.")

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

        fator_acruado_di = 1.0
        if is_di:
            pu_par_atual = (
                float(pu_par_val)
                if pu_par_val is not None
                else self.pu_par(base_date, vna, last_coupon_date=last_coupon_date, fator_di_acumulado=fator_di_acumulado)
            )
            fator_acruado_di = (pu_par_atual / vna) if vna > 0.0 else 1.0

        rows = []
        saldo_vna = vna
        prev_du_di = 0
        prev_f_di = 1.0

        for _, row in future_df.iterrows():
            d_ev = row["data_evento"]
            d_pg = row["data_pagamento"]
            evento = str(row["evento"]).strip()
            evento_lower = evento.lower()
            tipo = str(row["tipo"]).strip()
            perc = float(row.get("taxa_percentual", 0.0))

            du_desc = self.calendar.bizdays(base_date, d_pg)
            r_desc = self._get_rate_for_du(discount_rate_or_curve, du_desc)

            if is_di:
                r_di = self._get_rate_for_du(di_curve, du_desc)
                f_di_acum = math.pow(1.0 + r_di, du_desc / 252.0)
                if is_pct_di:
                    di_d = math.pow(1.0 + r_di, 1.0 / 252.0) - 1.0
                    df_val = math.trunc(math.pow(1.0 + di_d * r_desc, du_desc) * 1e14) / 1e14
                else:
                    df_val = math.trunc((f_di_acum * math.pow(1.0 + r_desc, du_desc / 252.0)) * 1e14) / 1e14
                taxa_desconto_eff = (math.pow(df_val, 252.0 / du_desc) - 1.0) if du_desc > 0 else 0.0
            else:
                # Fator de desconto T-14
                df_val = math.trunc(math.pow(1.0 + r_desc, du_desc / 252.0) * 1e14) / 1e14
                taxa_desconto_eff = r_desc

            if "juros" in evento_lower:
                if is_di:
                    if is_pct_di:
                        if prev_du_di == 0:
                            di_d_rem = math.pow(f_di_acum, 1.0 / du_desc) - 1.0
                            f_rem = math.pow(1.0 + di_d_rem * self.multiplicador_di, du_desc)
                            f_cupom = (fator_acruado_di * f_rem) - 1.0
                        else:
                            du_p = du_desc - prev_du_di
                            f_di_p = f_di_acum / prev_f_di
                            di_d_p = math.pow(f_di_p, 1.0 / du_p) - 1.0
                            f_cupom = math.pow(1.0 + di_d_p * self.multiplicador_di, du_p) - 1.0
                        r_cupom = self.multiplicador_di
                    else:
                        r_cupom = (perc / 100.0) if perc > 0.0 else self.coupon_rate
                        if prev_du_di == 0:
                            f_spr_rem = math.pow(1.0 + r_cupom, du_desc / 252.0)
                            f_cupom = (fator_acruado_di * f_di_acum * f_spr_rem) - 1.0
                        else:
                            du_p = du_desc - prev_du_di
                            f_di_p = f_di_acum / prev_f_di
                            f_spr_p = math.pow(1.0 + r_cupom, du_p / 252.0)
                            f_cupom = (f_di_p * f_spr_p) - 1.0
                    prev_du_di = du_desc
                    prev_f_di = f_di_acum
                else:
                    # O período contratual de contagem de juros ocorre entre as datas de liquidação (d_pg)
                    yf_period = self._coupon_year_fraction(prev_event_date, d_pg)
                    r_cupom = (perc / 100.0) if perc > 0.0 else self.coupon_rate
                    f_cupom = math.trunc((math.pow(1.0 + r_cupom, yf_period) - 1.0) * 1e9) / 1e9

                # Critério padrão SND: valor unitário dos juros truncado em 6 casas decimais (T-6)
                valor_fluxo = math.trunc((saldo_vna * f_cupom) * 1e6) / 1e6
                pv_val = valor_fluxo / df_val

                rows.append({
                    "data_evento": d_ev,
                    "data_pagamento": d_pg,
                    "du": du_desc,
                    "evento": "Juros",
                    "tipo": tipo,
                    "taxa_cupom": r_cupom,
                    "fluxo_nominal": valor_fluxo,
                    "taxa_desconto": taxa_desconto_eff,
                    "fator_desconto": df_val,
                    "valor_presente": pv_val
                })
                prev_event_date = d_pg

            elif "amort" in evento_lower or "vencimento" in evento_lower:
                if "vencimento" in evento_lower or perc <= 0.0 or (not self.amortizacao_sobre_vne and perc >= 100.0):
                    valor_fluxo = math.trunc(saldo_vna * 1e6) / 1e6
                    saldo_vna = 0.0
                else:
                    base_amort = vna if self.amortizacao_sobre_vne else saldo_vna
                    valor_fluxo = min(saldo_vna, math.trunc((base_amort * (perc / 100.0)) * 1e6) / 1e6)
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
                    "taxa_desconto": taxa_desconto_eff,
                    "fator_desconto": df_val,
                    "valor_presente": pv_val
                })

        return pd.DataFrame(rows)

    def present_value(self, base_date: Union[datetime.date, str],
                      discount_rate_or_curve: Union[float, Curve],
                      vna: Optional[float] = None,
                      last_coupon_date: Union[datetime.date, str] = None,
                      return_cashflows: bool = False,
                      di_curve: Optional[Union[float, Curve]] = None,
                      pu_par_val: Optional[float] = None,
                      fator_di_acumulado: Optional[float] = None) -> Union[float, Tuple[float, pd.DataFrame]]:
        """
        Calcula o Valor Presente (PU da Operação marcado a mercado) da debênture.
        Se return_cashflows=True, retorna a tupla (pu, df_fluxos).
        """
        vna_atual = self.resolve_vna(base_date, vna=vna)
        df_flows = self.cashflow_table(
            base_date=base_date,
            discount_rate_or_curve=discount_rate_or_curve,
            vna=vna_atual,
            last_coupon_date=last_coupon_date,
            di_curve=di_curve,
            pu_par_val=pu_par_val,
            fator_di_acumulado=fator_di_acumulado
        )

        if df_flows.empty:
            pu = vna_atual
        else:
            pu = math.trunc(df_flows["valor_presente"].sum() * 1e6) / 1e6

        if return_cashflows:
            return pu, df_flows
        return pu

    def cotacao(self, base_date: Union[datetime.date, str],
                discount_rate_or_curve: Union[float, Curve],
                vna: Optional[float] = None,
                last_coupon_date: Union[datetime.date, str] = None,
                di_curve: Optional[Union[float, Curve]] = None,
                pu_par_val: Optional[float] = None,
                fator_di_acumulado: Optional[float] = None) -> float:
        """
        Calcula a cotação percentual sobre o PU Par (% PU Par):
        Cotação = (PU_Operacao / PU_Par) * 100
        """
        vna_atual = self.resolve_vna(base_date, vna=vna)
        pu_par_atual = (
            float(pu_par_val)
            if pu_par_val is not None
            else self.pu_par(base_date, vna_atual, last_coupon_date=last_coupon_date, fator_di_acumulado=fator_di_acumulado)
        )
        if pu_par_atual <= 0.0:
            return 100.0
        pu_val = self.present_value(
            base_date, discount_rate_or_curve, vna_atual,
            last_coupon_date=last_coupon_date,
            di_curve=di_curve,
            pu_par_val=pu_par_atual,
            fator_di_acumulado=fator_di_acumulado
        )
        return math.trunc((pu_val / pu_par_atual) * 1e6) / 1e4

    def duration(self, base_date: Union[datetime.date, str],
                 discount_rate_or_curve: Union[float, Curve],
                 vna: Optional[float] = None,
                 last_coupon_date: Union[datetime.date, str] = None,
                 di_curve: Optional[Union[float, Curve]] = None,
                 pu_par_val: Optional[float] = None,
                 fator_di_acumulado: Optional[float] = None) -> float:
        """
        Calcula a Macaulay Duration em dias úteis (DU).
        """
        df_flows = self.cashflow_table(
            base_date=base_date,
            discount_rate_or_curve=discount_rate_or_curve,
            vna=vna,
            last_coupon_date=last_coupon_date,
            di_curve=di_curve,
            pu_par_val=pu_par_val,
            fator_di_acumulado=fator_di_acumulado
        )
        if df_flows.empty:
            return 0.0

        total_pv = df_flows["valor_presente"].sum()
        if total_pv <= 0.0:
            return 0.0

        dur = (df_flows["du"] * df_flows["valor_presente"]).sum() / total_pv
        # Critério ANBIMA: truncamento em 2 casas decimais (T-2)
        return math.trunc(float(dur) * 100.0) / 100.0

    def modified_duration(self, base_date: Union[datetime.date, str],
                          discount_rate_or_curve: Union[float, Curve],
                          vna: Optional[float] = None,
                          last_coupon_date: Union[datetime.date, str] = None,
                          di_curve: Optional[Union[float, Curve]] = None,
                          pu_par_val: Optional[float] = None,
                          fator_di_acumulado: Optional[float] = None) -> float:
        """
        Calcula a Duration Modificada em anos (base BUS/252):
        D_mod = (1 / P) * sum( (t_k / (1 + y_k)) * PV_k ), onde t_k = DU_k / 252.
        """
        df_flows = self.cashflow_table(
            base_date=base_date,
            discount_rate_or_curve=discount_rate_or_curve,
            vna=vna,
            last_coupon_date=last_coupon_date,
            di_curve=di_curve,
            pu_par_val=pu_par_val,
            fator_di_acumulado=fator_di_acumulado
        )
        if df_flows.empty:
            return 0.0

        total_pv = df_flows["valor_presente"].sum()
        if total_pv <= 0.0:
            return 0.0

        t_anos = df_flows["du"] / 252.0
        d_mod = ((t_anos / (1.0 + df_flows["taxa_desconto"])) * df_flows["valor_presente"]).sum() / total_pv
        return float(d_mod)

    def dv01(self, base_date: Union[datetime.date, str],
             discount_rate_or_curve: Union[float, Curve],
             vna: Optional[float] = None,
             last_coupon_date: Union[datetime.date, str] = None,
             di_curve: Optional[Union[float, Curve]] = None,
             pu_par_val: Optional[float] = None,
             fator_di_acumulado: Optional[float] = None) -> float:
        """
        Calcula o DV01 (Dollar Value of 01), variação monetária do PU para 1 ponto base (0,01% a.a.)
        de deslocamento paralelo na taxa de desconto:
        DV01 = PU * D_mod * 0.0001.
        """
        pu_val = self.present_value(
            base_date, discount_rate_or_curve, vna,
            last_coupon_date=last_coupon_date,
            di_curve=di_curve,
            pu_par_val=pu_par_val,
            fator_di_acumulado=fator_di_acumulado
        )
        d_mod = self.modified_duration(
            base_date, discount_rate_or_curve, vna,
            last_coupon_date=last_coupon_date,
            di_curve=di_curve,
            pu_par_val=pu_par_val,
            fator_di_acumulado=fator_di_acumulado
        )
        return float(pu_val * d_mod * 0.0001)

    def convexity(self, base_date: Union[datetime.date, str],
                  discount_rate_or_curve: Union[float, Curve],
                  vna: Optional[float] = None,
                  last_coupon_date: Union[datetime.date, str] = None,
                  di_curve: Optional[Union[float, Curve]] = None,
                  pu_par_val: Optional[float] = None,
                  fator_di_acumulado: Optional[float] = None) -> float:
        """
        Calcula a Convexidade Analítica em anos^2 (base BUS/252):
        C = (1 / P) * sum( (t_k * (t_k + 1) / (1 + y_k)^2) * PV_k ), onde t_k = DU_k / 252.
        """
        df_flows = self.cashflow_table(
            base_date=base_date,
            discount_rate_or_curve=discount_rate_or_curve,
            vna=vna,
            last_coupon_date=last_coupon_date,
            di_curve=di_curve,
            pu_par_val=pu_par_val,
            fator_di_acumulado=fator_di_acumulado
        )
        if df_flows.empty:
            return 0.0

        total_pv = df_flows["valor_presente"].sum()
        if total_pv <= 0.0:
            return 0.0

        t_anos = df_flows["du"] / 252.0
        f_conv = (t_anos * (t_anos + 1.0)) / np.power(1.0 + df_flows["taxa_desconto"], 2.0)
        conv = (f_conv * df_flows["valor_presente"]).sum() / total_pv
        return float(conv)

    def risk_metrics(self, base_date: Union[datetime.date, str],
                     discount_rate_or_curve: Union[float, Curve],
                     vna: Optional[float] = None,
                     last_coupon_date: Union[datetime.date, str] = None,
                     di_curve: Optional[Union[float, Curve]] = None,
                     pu_par_val: Optional[float] = None,
                     fator_di_acumulado: Optional[float] = None) -> Dict[str, float]:
        """
        Consolida as métricas de precificação e sensibilidade da debênture na data de referência.
        """
        vna_atual = self.resolve_vna(base_date, vna=vna)
        pu_par_atual = (
            float(pu_par_val)
            if pu_par_val is not None
            else self.pu_par(base_date, vna_atual, last_coupon_date=last_coupon_date, fator_di_acumulado=fator_di_acumulado)
        )
        pu_val, df_flows = self.present_value(
            base_date, discount_rate_or_curve, vna_atual,
            last_coupon_date=last_coupon_date,
            return_cashflows=True,
            di_curve=di_curve,
            pu_par_val=pu_par_atual,
            fator_di_acumulado=fator_di_acumulado
        )
        cot_val = math.trunc((pu_val / pu_par_atual) * 1e6) / 1e4 if pu_par_atual > 0.0 else 100.0

        if df_flows.empty or df_flows["valor_presente"].sum() <= 0.0:
            return {
                "pu_par": pu_par_atual,
                "pu": pu_val,
                "cotacao": cot_val,
                "duration_du": 0.0,
                "duration_anos": 0.0,
                "modified_duration": 0.0,
                "dv01": 0.0,
                "convexity": 0.0
            }

        total_pv = df_flows["valor_presente"].sum()
        t_anos = df_flows["du"] / 252.0
        dur_du = math.trunc(((df_flows["du"] * df_flows["valor_presente"]).sum() / total_pv) * 100.0) / 100.0
        dur_anos = float((t_anos * df_flows["valor_presente"]).sum() / total_pv)
        d_mod = float(((t_anos / (1.0 + df_flows["taxa_desconto"])) * df_flows["valor_presente"]).sum() / total_pv)
        dv01_val = float(pu_val * d_mod * 0.0001)
        conv_val = float(((t_anos * (t_anos + 1.0)) / np.power(1.0 + df_flows["taxa_desconto"], 2.0) * df_flows["valor_presente"]).sum() / total_pv)

        return {
            "pu_par": pu_par_atual,
            "pu": pu_val,
            "cotacao": cot_val,
            "duration_du": dur_du,
            "duration_anos": dur_anos,
            "modified_duration": d_mod,
            "dv01": dv01_val,
            "convexity": conv_val
        }


class CertificadoRecebiveis(Debenture):
    """
    Classe base para Certificados de Recebíveis (CRI e CRA) conforme metodologia ANBIMA.
    Suporta convenções de contagem de dias contratuais no cupom (BUS/252, ACT/360,
    ACT/365, 30/360), defasagem de DI no cálculo do PU Par, amortizações sobre VNE
    ou saldo remanescente e desconto a mercado em base 252 dias úteis.
    """

    DEFAULT_LASTRO = ""

    def __init__(self,
                 ticker: str,
                 maturity_date: Union[datetime.date, str] = None,
                 indexador: str = "IPCA",
                 coupon_rate: float = 0.0,
                 issue_date: Union[datetime.date, str] = None,
                 cashflows_df: Optional[pd.DataFrame] = None,
                 nominal_value: float = 1000.0,
                 calendar_name: str = "B3",
                 multiplicador_di: float = 1.0,
                 amortizacao_sobre_vne: bool = False,
                 coupon_day_count: str = "BUS/252",
                 defasagem_di: int = 0,
                 securitizadora: str = "",
                 lastro: str = "",
                 serie: str = "",
                 emissao: str = "",
                 anniversary_day: Optional[int] = None,
                 vna_engine: Optional[VNAEngine] = None):
        super().__init__(
            ticker=ticker,
            maturity_date=maturity_date,
            indexador=indexador,
            coupon_rate=coupon_rate,
            issue_date=issue_date,
            cashflows_df=cashflows_df,
            nominal_value=nominal_value,
            calendar_name=calendar_name,
            multiplicador_di=multiplicador_di,
            amortizacao_sobre_vne=amortizacao_sobre_vne,
            coupon_day_count=coupon_day_count,
            defasagem_di=defasagem_di,
            anniversary_day=anniversary_day,
            vna_engine=vna_engine
        )
        self.securitizadora = securitizadora.strip()
        self.lastro = (lastro or self.DEFAULT_LASTRO).strip()
        self.serie = str(serie).strip()
        self.emissao = str(emissao).strip()

    @classmethod
    def from_schedule(cls,
                      ticker: str,
                      issue_date: Union[datetime.date, str],
                      maturity_date: Union[datetime.date, str],
                      indexador: str = "IPCA",
                      coupon_rate: float = 0.0,
                      coupon_freq: str = "mensal",
                      amort_type: str = "bullet",
                      amort_start_date: Optional[Union[datetime.date, str]] = None,
                      custom_amort_schedule: Optional[List[Tuple[Union[datetime.date, str], float]]] = None,
                      nominal_value: float = 1000.0,
                      calendar_name: str = "B3",
                      multiplicador_di: float = 1.0,
                      amortizacao_sobre_vne: bool = False,
                      coupon_day_count: str = "BUS/252",
                      defasagem_di: int = 0,
                      securitizadora: str = "",
                      lastro: str = "",
                      serie: str = "",
                      emissao: str = "",
                      anniversary_day: Optional[int] = None,
                      vna_engine: Optional[VNAEngine] = None):
        """
        Constrói um Certificado de Recebíveis (CRI ou CRA) gerando automaticamente
        a agenda de eventos financeiros a partir dos parâmetros do termo de securitização.
        """
        df_cf = CRICRAClient.build_schedule(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_freq=coupon_freq,
            amort_type=amort_type,
            amort_start_date=amort_start_date,
            custom_amort_schedule=custom_amort_schedule,
            calendar_name=calendar_name
        )
        return cls(
            ticker=ticker,
            maturity_date=maturity_date,
            indexador=indexador,
            coupon_rate=coupon_rate,
            issue_date=issue_date,
            cashflows_df=df_cf,
            nominal_value=nominal_value,
            calendar_name=calendar_name,
            multiplicador_di=multiplicador_di,
            amortizacao_sobre_vne=amortizacao_sobre_vne,
            coupon_day_count=coupon_day_count,
            defasagem_di=defasagem_di,
            securitizadora=securitizadora,
            lastro=lastro,
            serie=serie,
            emissao=emissao,
            anniversary_day=anniversary_day,
            vna_engine=vna_engine
        )


class CRI(CertificadoRecebiveis):
    """
    Certificado de Recebíveis Imobiliários (CRI) conforme metodologia ANBIMA.
    """

    DEFAULT_LASTRO = "Imobiliário"


class CRA(CertificadoRecebiveis):
    """
    Certificado de Recebíveis do Agronegócio (CRA) conforme metodologia ANBIMA.
    """

    DEFAULT_LASTRO = "Agronegócio"
