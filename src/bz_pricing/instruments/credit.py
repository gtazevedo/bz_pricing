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

try:
    from bz_pricing.core.credit_core import compute_credit_cashflows as _cy_compute_credit_cashflows
    from bz_pricing.core.credit_core import evaluate_gross_up_pv as _cy_evaluate_gross_up_pv
except ImportError:
    _cy_compute_credit_cashflows = None
    _cy_evaluate_gross_up_pv = None


class Debenture:
    """
    Instrumento de crédito privado estruturado (Debênture) conforme metodologia ANBIMA e SND.
    Suporta ativos indexados ao IPCA, IGP-M, DI (+ spread ou % do DI) e Prefixados,
    com amortizações ordinárias/extraordinárias e fluxos de caixa periódicos.
    """

    _value_scale: float = 1e6
    _is_cricra: bool = False

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
                 vna_engine: Optional[VNAEngine] = None,
                 is_pct_di: Optional[bool] = None):
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
        self._is_pct_di_override = is_pct_di

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
        if self._is_pct_di_override is not None:
            return bool(self._is_pct_di_override)
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
            f_dia = 1.0 + (float(r) / 100.0) * multiplicador
            if self._is_cricra and multiplicador != 1.0:
                f_dia = math.trunc(f_dia * 1e16) / 1e16
            fator *= f_dia
        if round_factor_8 or self._is_cricra:
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
        base_dt = self._normalize_date(base_date)
        if "IPCA" not in self.indexador and "IGP" not in self.indexador:
            saldo = float(self.nominal_value)
            if self.cashflows_df is not None and not self.cashflows_df.empty:
                past_amorts = self.cashflows_df[
                    (self.cashflows_df["data_pagamento"] <= base_dt) &
                    (self.cashflows_df["evento"].str.contains("amort", case=False, na=False))
                ]
                for _, r in past_amorts.iterrows():
                    perc = float(r.get("taxa_percentual", 0.0) or 0.0)
                    if perc > 0.0:
                        base_a = float(self.nominal_value) if self.amortizacao_sobre_vne else saldo
                        saldo = max(0.0, saldo - math.trunc((base_a * (perc / 100.0)) * self._value_scale) / self._value_scale)
            return saldo
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
            return math.trunc((vna * f_di) * self._value_scale) / self._value_scale

        if self._is_di_linked():
            f_di = (
                float(fator_di_acumulado)
                if fator_di_acumulado is not None
                else self._fetch_di_factor(last_date, base_date, multiplicador=1.0, round_factor_8=True)
            )
            f_spr = round(math.pow(1.0 + self.coupon_rate, du_passado / 252.0), 9)
            f_juros = round(f_di * f_spr, 9) if self._is_cricra else math.trunc(f_di * f_spr * 1e9) / 1e9
            return math.trunc((vna * f_juros) * self._value_scale) / self._value_scale

        # Juros acruados pró-rata para IPCA, IGP-M e Prefixado
        yf_passado = self._coupon_year_fraction(last_date, base_date)
        f_juros = math.pow(1.0 + self.coupon_rate, yf_passado) - 1.0
        juros_val = math.trunc((vna * f_juros) * self._value_scale) / self._value_scale
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

        if _cy_compute_credit_cashflows is not None:
            d_ev_list = future_df["data_evento"].tolist()
            d_pg_list = future_df["data_pagamento"].tolist()
            ev_raw = future_df["evento"].astype(str).str.strip().tolist()
            tipo_list = future_df["tipo"].astype(str).str.strip().tolist()
            perc_raw = pd.to_numeric(future_df["taxa_percentual"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

            keep_idx = []
            du_list = []
            yf_list = []
            ev_codes = []
            ev_labels = []
            r_desc_list = []
            r_di_list = []
            prev_dt = prev_event_date

            for idx, (ev_str, d_pg) in enumerate(zip(ev_raw, d_pg_list)):
                ev_low = ev_str.lower()
                if "juros" in ev_low:
                    code = 0
                    label = "Juros"
                elif "amort" in ev_low or "vencimento" in ev_low:
                    code = 2 if "vencimento" in ev_low else 1
                    label = ev_str
                else:
                    continue

                du_d = self.calendar.bizdays(base_date, d_pg)
                yf_p = self._coupon_year_fraction(prev_dt, d_pg) if (code == 0 and not is_di) else 0.0
                if code == 0:
                    prev_dt = d_pg

                keep_idx.append(idx)
                du_list.append(du_d)
                yf_list.append(yf_p)
                ev_codes.append(code)
                ev_labels.append(label)
                r_desc_list.append(self._get_rate_for_du(discount_rate_or_curve, du_d))
                r_di_list.append(self._get_rate_for_du(di_curve, du_d) if is_di else 0.0)

            if not keep_idx:
                return pd.DataFrame(columns=[
                    "data_evento", "data_pagamento", "du", "evento", "tipo",
                    "taxa_cupom", "fluxo_nominal", "taxa_desconto", "fator_desconto", "valor_presente"
                ])

            du_arr = np.asarray(du_list, dtype=np.int32)
            yf_arr = np.asarray(yf_list, dtype=np.float64)
            perc_arr = perc_raw[keep_idx]
            ev_arr = np.asarray(ev_codes, dtype=np.int32)
            r_desc_arr = np.asarray(r_desc_list, dtype=np.float64)
            r_di_arr = np.asarray(r_di_list, dtype=np.float64)

            taxa_cupom_arr, fluxo_arr, taxa_desc_arr, df_arr, pv_arr, _ = _cy_compute_credit_cashflows(
                du_arr,
                yf_arr,
                perc_arr,
                ev_arr,
                r_desc_arr,
                r_di_arr,
                float(vna),
                float(fator_acruado_di),
                float(self.coupon_rate),
                float(self.multiplicador_di),
                float(self._value_scale),
                bool(self._is_cricra),
                bool(is_di),
                bool(is_pct_di),
                bool(self.amortizacao_sobre_vne),
            )

            return pd.DataFrame({
                "data_evento": [d_ev_list[i] for i in keep_idx],
                "data_pagamento": [d_pg_list[i] for i in keep_idx],
                "du": du_arr,
                "evento": ev_labels,
                "tipo": [tipo_list[i] for i in keep_idx],
                "taxa_cupom": taxa_cupom_arr,
                "fluxo_nominal": fluxo_arr,
                "taxa_desconto": taxa_desc_arr,
                "fator_desconto": df_arr,
                "valor_presente": pv_arr,
            })

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
                        f_pct_acum = math.pow(1.0 + di_d * self.multiplicador_di, du_desc)
                        if prev_du_di == 0:
                            f_cupom = (fator_acruado_di * f_pct_acum) - 1.0
                        else:
                            f_cupom = (f_pct_acum / prev_f_di) - 1.0
                        r_cupom = self.multiplicador_di
                        prev_du_di = du_desc
                        prev_f_di = f_pct_acum
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
                    f_cupom_raw = math.pow(1.0 + r_cupom, yf_period) - 1.0
                    f_cupom = round(f_cupom_raw, 9) if self._is_cricra else math.trunc(f_cupom_raw * 1e9) / 1e9

                # Critério SND (T-6) para debêntures e ANBIMA (T-8) para CRI/CRA
                valor_fluxo = math.trunc((saldo_vna * f_cupom) * self._value_scale) / self._value_scale
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
                    valor_fluxo = math.trunc(saldo_vna * self._value_scale) / self._value_scale
                    saldo_vna = 0.0
                else:
                    base_amort = vna if self.amortizacao_sobre_vne else saldo_vna
                    valor_fluxo = min(saldo_vna, math.trunc((base_amort * (perc / 100.0)) * self._value_scale) / self._value_scale)
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
            pu = math.trunc(df_flows["valor_presente"].sum() * self._value_scale) / self._value_scale

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

    @staticmethod
    def _ir_regressive_rate(calendar_days: int) -> float:
        """Retorna a alíquota da tabela regressiva de IR conforme prazo em dias corridos."""
        if calendar_days <= 180:
            return 0.225
        if calendar_days <= 360:
            return 0.200
        if calendar_days <= 720:
            return 0.175
        return 0.150

    def gross_up_rate(self, base_date: Union[datetime.date, str],
                      discount_rate: float,
                      vna: Optional[float] = None,
                      last_coupon_date: Union[datetime.date, str] = None,
                      di_curve: Optional[Union[float, Curve]] = None,
                      pu_par_val: Optional[float] = None,
                      fator_di_acumulado: Optional[float] = None) -> float:
        """
        Calcula a taxa nominal com gross-up de IR (Lei 12.431 / ativos incentivados)
        conforme metodologia ANBIMA de Z-Spread: retorna a alíquota regressiva de IR
        sobre cada parcela futura de juros e encontra a taxa que preserva o PU indicativo.
        """
        from scipy.optimize import brentq

        base_dt = self._normalize_date(base_date)
        vna_atual = self.resolve_vna(base_dt, vna=vna)
        pu_target, df_flows = self.present_value(
            base_dt,
            float(discount_rate),
            vna=vna_atual,
            last_coupon_date=last_coupon_date,
            return_cashflows=True,
            di_curve=di_curve,
            pu_par_val=pu_par_val,
            fator_di_acumulado=fator_di_acumulado,
        )
        if df_flows.empty or pu_target <= 0.0:
            return float(discount_rate)

        adjusted_flows = []
        du_arr = df_flows["du"].to_numpy(dtype=float)
        for _, row in df_flows.iterrows():
            fluxo = float(row["fluxo_nominal"])
            if "juros" in str(row["evento"]).lower():
                d_pg = self._normalize_date(row["data_pagamento"])
                dc = max(1, (d_pg - base_dt).days)
                aliq = self._ir_regressive_rate(dc)
                adjusted_flows.append(fluxo / (1.0 - aliq))
            else:
                adjusted_flows.append(fluxo)

        flows_arr = np.asarray(adjusted_flows, dtype=np.float64)
        du_i32 = du_arr.astype(np.int32)

        def _pv_diff(rate: float) -> float:
            if _cy_evaluate_gross_up_pv is not None:
                return float(_cy_evaluate_gross_up_pv(du_i32, flows_arr, float(rate)) - pu_target)
            disc = np.power(1.0 + rate, du_arr / 252.0)
            return float(np.sum(flows_arr / disc) - pu_target)

        low = max(-0.90, float(discount_rate) - 0.05)
        high = float(discount_rate) + 0.50
        return float(brentq(_pv_diff, low, high, xtol=1e-10))

    def z_spread(self, base_date: Union[datetime.date, str],
                 discount_rate: float,
                 risk_free_rate: Optional[float] = None,
                 ettj_pre_rate_or_curve: Optional[Union[float, Curve]] = None,
                 incentivada: bool = False,
                 vna: Optional[float] = None,
                 last_coupon_date: Union[datetime.date, str] = None,
                 di_curve: Optional[Union[float, Curve]] = None,
                 pu_par_val: Optional[float] = None,
                 fator_di_acumulado: Optional[float] = None) -> float:
        """
        Calcula o Z-Spread conforme metodologia oficial ANBIMA (Agosto/2025):
        - CDI + spread: o Z-spread é a própria taxa indicativa (`discount_rate`).
        - % do DI: ((J * T) + 1) / (J + 1) - 1, usando a taxa `J` da ETTJ prefixada
          na duration de Macaulay do ativo.
        - IPCA / IGP-M / Prefixado: (1 + i_eff) / (1 + P) - 1, onde `P` é a taxa do
          ativo livre de risco de duration mais próxima (NTN-B ou ETTJ) e `i_eff`
          considera o gross-up de IR quando `incentivada=True`.
        """
        if self._is_pct_di():
            if ettj_pre_rate_or_curve is None:
                raise ValueError(
                    "Para calcular o Z-Spread de ativos '% do DI', informe 'ettj_pre_rate_or_curve' "
                    "(taxa ou curva ETTJ prefixada)."
                )
            dur_du = self.duration(
                base_date=base_date,
                discount_rate_or_curve=discount_rate,
                vna=vna,
                last_coupon_date=last_coupon_date,
                di_curve=di_curve,
                pu_par_val=pu_par_val,
                fator_di_acumulado=fator_di_acumulado,
            )
            j_rate = self._get_rate_for_du(ettj_pre_rate_or_curve, dur_du)
            t_pct = float(discount_rate)
            return float(((j_rate * t_pct) + 1.0) / (j_rate + 1.0) - 1.0)

        if self._is_di_linked():
            return float(discount_rate)

        if risk_free_rate is None:
            raise ValueError(
                "Para calcular o Z-Spread de ativos IPCA/IGP-M/Prefixados, informe 'risk_free_rate' "
                "(taxa da NTN-B ou vértice livre de risco de duration mais próxima)."
            )

        i_eff = (
            self.gross_up_rate(
                base_date=base_date,
                discount_rate=float(discount_rate),
                vna=vna,
                last_coupon_date=last_coupon_date,
                di_curve=di_curve,
                pu_par_val=pu_par_val,
                fator_di_acumulado=fator_di_acumulado,
            )
            if incentivada
            else float(discount_rate)
        )
        return float((1.0 + i_eff) / (1.0 + float(risk_free_rate)) - 1.0)

    def risk_metrics(self, base_date: Union[datetime.date, str],
                     discount_rate_or_curve: Union[float, Curve],
                     vna: Optional[float] = None,
                     last_coupon_date: Union[datetime.date, str] = None,
                     di_curve: Optional[Union[float, Curve]] = None,
                     pu_par_val: Optional[float] = None,
                     fator_di_acumulado: Optional[float] = None,
                     risk_free_rate: Optional[float] = None,
                     ettj_pre_rate_or_curve: Optional[Union[float, Curve]] = None,
                     incentivada: bool = False) -> Dict[str, float]:
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

        out: Dict[str, float] = {
            "pu_par": pu_par_atual,
            "pu": pu_val,
            "cotacao": cot_val,
            "duration_du": dur_du,
            "duration_anos": dur_anos,
            "modified_duration": d_mod,
            "dv01": dv01_val,
            "convexity": conv_val,
        }

        if not isinstance(discount_rate_or_curve, Curve):
            r_scalar = float(discount_rate_or_curve)
            if self._is_di_linked() and not self._is_pct_di():
                out["z_spread"] = r_scalar
            elif self._is_pct_di() and ettj_pre_rate_or_curve is not None:
                j_rate = self._get_rate_for_du(ettj_pre_rate_or_curve, dur_du)
                out["z_spread"] = float(((j_rate * r_scalar) + 1.0) / (j_rate + 1.0) - 1.0)
            elif not self._is_di_linked() and risk_free_rate is not None:
                i_eff = (
                    self.gross_up_rate(
                        base_date=base_date,
                        discount_rate=r_scalar,
                        vna=vna_atual,
                        last_coupon_date=last_coupon_date,
                        di_curve=di_curve,
                        pu_par_val=pu_par_atual,
                        fator_di_acumulado=fator_di_acumulado,
                    )
                    if incentivada
                    else r_scalar
                )
                if incentivada:
                    out["taxa_gross_up"] = i_eff
                out["z_spread"] = float((1.0 + i_eff) / (1.0 + float(risk_free_rate)) - 1.0)

        return out

    def ytm(self, base_date: Union[datetime.date, str],
            target_pu: float,
            vna: Optional[float] = None,
            last_coupon_date: Union[datetime.date, str] = None,
            di_curve: Optional[Union[float, Curve]] = None,
            pu_par_val: Optional[float] = None,
            fator_di_acumulado: Optional[float] = None,
            xtol: float = 1e-10) -> float:
        """
        Calcula o Yield to Maturity (ou Yield to Call em Letras Financeiras subordinadas)
        implícito a partir de um PU de mercado (`target_pu`), na convenção de cotação do
        indexador do ativo (taxa nominal PRÉ, taxa real IPCA/IGP-M, spread sobre DI ou
        multiplicador percentual do DI).
        """
        from scipy.optimize import brentq

        pu_obj = float(target_pu)
        if pu_obj <= 0.0:
            raise ValueError("'target_pu' deve ser estritamente positivo.")

        base_dt = self._normalize_date(base_date)
        vna_atual = self.resolve_vna(base_dt, vna=vna)
        pu_par_atual = pu_par_val
        if pu_par_atual is None and self._is_di_linked():
            pu_par_atual = self.pu_par(
                base_dt,
                vna_atual,
                last_coupon_date=last_coupon_date,
                fator_di_acumulado=fator_di_acumulado,
            )

        def _obj(rate: float) -> float:
            pu_val = self.present_value(
                base_dt,
                float(rate),
                vna=vna_atual,
                last_coupon_date=last_coupon_date,
                di_curve=di_curve,
                pu_par_val=pu_par_atual,
                fator_di_acumulado=fator_di_acumulado,
            )
            return float(pu_val - pu_obj)

        low = 1e-4 if self._is_pct_di() else -0.90
        high = 5.0 if self._is_pct_di() else 2.0
        f_low = _obj(low)
        f_high = _obj(high)
        while f_low * f_high > 0.0 and high < 100.0:
            high *= 2.0
            f_high = _obj(high)
        if f_low * f_high > 0.0:
            raise ValueError(f"Não foi possível isolar a raiz de YTM para {self.ticker} com target_pu={pu_obj}.")
        return float(brentq(_obj, low, high, xtol=xtol))

    def tir(self, base_date: Union[datetime.date, str],
            target_pu: float,
            vna: Optional[float] = None,
            last_coupon_date: Union[datetime.date, str] = None,
            di_curve: Optional[Union[float, Curve]] = None,
            pu_par_val: Optional[float] = None,
            fator_di_acumulado: Optional[float] = None,
            xtol: float = 1e-10) -> float:
        """
        Calcula a Taxa Interna de Retorno (TIR) efetiva anual em base 252 dias úteis
        sobre os fluxos futuros projetados do ativo para o PU informado (`target_pu`).
        Em títulos indexados ao DI, retorna a taxa total implícita (DI + spread).
        """
        from scipy.optimize import brentq

        pu_obj = float(target_pu)
        if pu_obj <= 0.0:
            raise ValueError("'target_pu' deve ser estritamente positivo.")

        base_dt = self._normalize_date(base_date)
        vna_atual = self.resolve_vna(base_dt, vna=vna)
        pu_par_atual = pu_par_val
        if pu_par_atual is None and self._is_di_linked():
            pu_par_atual = self.pu_par(
                base_dt,
                vna_atual,
                last_coupon_date=last_coupon_date,
                fator_di_acumulado=fator_di_acumulado,
            )

        df_flows = self.cashflow_table(
            base_dt,
            0.0,
            vna=vna_atual,
            last_coupon_date=last_coupon_date,
            di_curve=di_curve,
            pu_par_val=pu_par_atual,
            fator_di_acumulado=fator_di_acumulado,
        )
        if df_flows.empty:
            raise ValueError(f"Não há fluxos futuros após {base_dt} para o cálculo de TIR de {self.ticker}.")

        du_arr = df_flows["du"].to_numpy(dtype=np.float64)
        du_i32 = du_arr.astype(np.int32)
        flows_arr = df_flows["fluxo_nominal"].to_numpy(dtype=np.float64)

        def _obj(rate: float) -> float:
            if _cy_evaluate_gross_up_pv is not None:
                return float(_cy_evaluate_gross_up_pv(du_i32, flows_arr, float(rate)) - pu_obj)
            return float(np.sum(flows_arr / np.power(1.0 + rate, du_arr / 252.0)) - pu_obj)

        low, high = -0.90, 5.0
        f_low = _obj(low)
        f_high = _obj(high)
        while f_low * f_high > 0.0 and high < 100.0:
            high *= 2.0
            f_high = _obj(high)
        if f_low * f_high > 0.0:
            raise ValueError(f"Não foi possível isolar a raiz de TIR para {self.ticker} com target_pu={pu_obj}.")
        return float(brentq(_obj, low, high, xtol=xtol))


class CertificadoRecebiveis(Debenture):
    """
    Classe base para Certificados de Recebíveis (CRI e CRA) conforme metodologia ANBIMA.
    Suporta convenções de contagem de dias contratuais no cupom (BUS/252, ACT/360,
    ACT/365, 30/360), defasagem de DI no cálculo do PU Par, amortizações sobre VNE
    ou saldo remanescente e desconto a mercado em base 252 dias úteis.
    """

    DEFAULT_LASTRO = ""
    _value_scale: float = 1e8
    _is_cricra: bool = True

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
                 vna_engine: Optional[VNAEngine] = None,
                 is_pct_di: Optional[bool] = None):
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
            vna_engine=vna_engine,
            is_pct_di=is_pct_di
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

    @classmethod
    def from_ticker(cls, ticker: str, client: Optional[CRICRAClient] = None,
                    calendar_name: str = "B3") -> "CertificadoRecebiveis":
        """
        Constrói automaticamente um CRI ou CRA a partir do portal ANBIMA Data,
        carregando características cadastrais e agenda de eventos financeiros.
        """
        cli = client or CRICRAClient()
        carac = cli.get_asset_characteristics(ticker)
        df_cf = cli.get_asset_cashflows(ticker)

        idx_obj = carac.get("indexador") or {}
        indice = idx_obj.get("nome", "IPCA") if isinstance(idx_obj, dict) else str(idx_obj)
        remun = str(carac.get("remuneracao") or "").upper()
        taxa_emissao = cls._parse_br_float(carac.get("taxa_emissao"), default=0.0)

        is_pct_di_remun = ("% DI" in remun or "% DO DI" in remun) and "+" not in remun
        if is_pct_di_remun:
            mult_di = taxa_emissao / 100.0 if taxa_emissao > 0.0 else 1.0
            coupon_rate = 0.0
        else:
            mult_di = 1.0
            coupon_rate = taxa_emissao / 100.0

        vne_num = cls._parse_br_float(carac.get("vne"), default=1000.0)
        expressao = str(carac.get("expressao_papel") or "252").upper()
        if "360" in expressao:
            day_count = "ACT/360"
        elif "365" in expressao:
            day_count = "ACT/365"
        else:
            day_count = "BUS/252"

        emissao_obj = carac.get("emissao") or {}
        emissor_obj = emissao_obj.get("emissor") if isinstance(emissao_obj, dict) else {}
        securitizadora = emissor_obj.get("nome", "") if isinstance(emissor_obj, dict) else ""

        target_cls = cls
        if cls is CertificadoRecebiveis:
            tipo_ativo = str(carac.get("tipo") or "").upper()
            if tipo_ativo == "CRI":
                target_cls = CRI
            elif tipo_ativo == "CRA":
                target_cls = CRA

        return target_cls(
            ticker=ticker,
            maturity_date=carac.get("data_vencimento"),
            indexador=indice,
            coupon_rate=coupon_rate,
            issue_date=carac.get("data_inicio_rentabilidade"),
            cashflows_df=df_cf,
            nominal_value=vne_num,
            calendar_name=calendar_name,
            multiplicador_di=mult_di,
            coupon_day_count=day_count,
            securitizadora=securitizadora,
            serie=str(carac.get("numero_serie") or ""),
            is_pct_di=is_pct_di_remun,
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


class CotaFIDC(CertificadoRecebiveis):
    """
    Cota de Fundo de Investimento em Direitos Creditórios (FIDC) conforme
    metodologia ANBIMA (T-8 para cota atualizada, amortizações, fluxos e cota
    de mercado; T-16/A-8 para DI; A-9 para fator de juros/spread; T-2 para duration).
    """

    def __init__(self,
                 ticker: str,
                 maturity_date: Union[datetime.date, str] = None,
                 indexador: str = "DI",
                 coupon_rate: float = 0.0,
                 issue_date: Union[datetime.date, str] = None,
                 cashflows_df: Optional[pd.DataFrame] = None,
                 nominal_value: float = 1000.0,
                 cota_emissao: Optional[float] = None,
                 calendar_name: str = "B3",
                 multiplicador_di: float = 1.0,
                 amortizacao_sobre_vne: bool = False,
                 coupon_day_count: str = "BUS/252",
                 defasagem_di: int = 0,
                 classe_cota: str = "SÊNIOR",
                 cnpj_fundo: str = "",
                 administrador: str = "",
                 custodiante: str = "",
                 anniversary_day: Optional[int] = None,
                 vna_engine: Optional[VNAEngine] = None,
                 is_pct_di: Optional[bool] = None):
        vne = float(cota_emissao) if cota_emissao is not None else float(nominal_value)
        super().__init__(
            ticker=ticker,
            maturity_date=maturity_date,
            indexador=indexador,
            coupon_rate=coupon_rate,
            issue_date=issue_date,
            cashflows_df=cashflows_df,
            nominal_value=vne,
            calendar_name=calendar_name,
            multiplicador_di=multiplicador_di,
            amortizacao_sobre_vne=amortizacao_sobre_vne,
            coupon_day_count=coupon_day_count,
            defasagem_di=defasagem_di,
            anniversary_day=anniversary_day,
            vna_engine=vna_engine,
            is_pct_di=is_pct_di,
        )
        self.classe_cota = classe_cota.strip().upper()
        self.cnpj_fundo = cnpj_fundo.strip()
        self.administrador = administrador.strip()
        self.custodiante = custodiante.strip()

    @classmethod
    def from_schedule(cls,
                      ticker: str,
                      issue_date: Union[datetime.date, str],
                      maturity_date: Union[datetime.date, str],
                      indexador: str = "DI",
                      coupon_rate: float = 0.0,
                      coupon_freq: str = "mensal",
                      amort_type: str = "bullet",
                      amort_start_date: Optional[Union[datetime.date, str]] = None,
                      custom_amort_schedule: Optional[List[Tuple[Union[datetime.date, str], float]]] = None,
                      nominal_value: float = 1000.0,
                      cota_emissao: Optional[float] = None,
                      calendar_name: str = "B3",
                      multiplicador_di: float = 1.0,
                      amortizacao_sobre_vne: bool = False,
                      coupon_day_count: str = "BUS/252",
                      defasagem_di: int = 0,
                      classe_cota: str = "SÊNIOR",
                      cnpj_fundo: str = "",
                      administrador: str = "",
                      custodiante: str = "",
                      anniversary_day: Optional[int] = None,
                      vna_engine: Optional[VNAEngine] = None):
        """Constrói uma CotaFIDC gerando automaticamente a agenda de eventos financeiros."""
        df_cf = CRICRAClient.build_schedule(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_freq=coupon_freq,
            amort_type=amort_type,
            amort_start_date=amort_start_date,
            custom_amort_schedule=custom_amort_schedule,
            calendar_name=calendar_name,
        )
        return cls(
            ticker=ticker,
            maturity_date=maturity_date,
            indexador=indexador,
            coupon_rate=coupon_rate,
            issue_date=issue_date,
            cashflows_df=df_cf,
            nominal_value=nominal_value,
            cota_emissao=cota_emissao,
            calendar_name=calendar_name,
            multiplicador_di=multiplicador_di,
            amortizacao_sobre_vne=amortizacao_sobre_vne,
            coupon_day_count=coupon_day_count,
            defasagem_di=defasagem_di,
            classe_cota=classe_cota,
            cnpj_fundo=cnpj_fundo,
            administrador=administrador,
            custodiante=custodiante,
            anniversary_day=anniversary_day,
            vna_engine=vna_engine,
        )


FIDC = CotaFIDC


class LetraFinanceira(CertificadoRecebiveis):
    """
    Letra Financeira (LF Sênior, Subordinada ou Complementar) conforme
    metodologia ANBIMA.

    Para LF Sênior, considera o vencimento final (yield to maturity).
    Para LF Subordinada ou Complementar com `call_date` informada, adota
    a data de possível recompra pelo emissor (yield to call) no fluxo.
    """

    def __init__(self,
                 ticker: str,
                 maturity_date: Union[datetime.date, str] = None,
                 indexador: str = "DI",
                 coupon_rate: float = 0.0,
                 issue_date: Union[datetime.date, str] = None,
                 cashflows_df: Optional[pd.DataFrame] = None,
                 nominal_value: float = 1000.0,
                 calendar_name: str = "B3",
                 multiplicador_di: float = 1.0,
                 amortizacao_sobre_vne: bool = False,
                 coupon_day_count: str = "BUS/252",
                 defasagem_di: int = 0,
                 classe: str = "SENIOR",
                 emissor: str = "",
                 call_date: Optional[Union[datetime.date, str]] = None,
                 anniversary_day: Optional[int] = None,
                 vna_engine: Optional[VNAEngine] = None,
                 is_pct_di: Optional[bool] = None):
        self.classe = classe.strip().upper()
        self.emissor = emissor.strip()
        self.call_date = self._normalize_date(call_date) if call_date is not None else None
        effective_maturity = (
            self.call_date
            if (self.call_date is not None and self.classe in ("SUBORDINADA", "COMPLEMENTAR"))
            else maturity_date
        )
        super().__init__(
            ticker=ticker,
            maturity_date=effective_maturity,
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
            vna_engine=vna_engine,
            is_pct_di=is_pct_di,
        )

    @classmethod
    def from_bullet(cls,
                    ticker: str,
                    issue_date: Union[datetime.date, str],
                    maturity_date: Union[datetime.date, str],
                    indexador: str = "DI",
                    coupon_rate: float = 0.0,
                    nominal_value: float = 1000.0,
                    calendar_name: str = "B3",
                    multiplicador_di: float = 1.0,
                    classe: str = "SENIOR",
                    emissor: str = "",
                    call_date: Optional[Union[datetime.date, str]] = None,
                    is_pct_di: Optional[bool] = None) -> "LetraFinanceira":
        """Constrói uma Letra Financeira bullet (sem cupons intermediários)."""
        cls_norm = classe.strip().upper()
        horizon = call_date if (call_date is not None and cls_norm in ("SUBORDINADA", "COMPLEMENTAR")) else maturity_date
        horizon_dt = horizon if isinstance(horizon, datetime.date) else pd.to_datetime(horizon, dayfirst="/" in str(horizon)).date()
        cal = get_calendar(calendar_name)
        pg_dt = cal.adjust_next(horizon_dt)
        df_cf = pd.DataFrame([
            {"data_evento": horizon_dt, "data_pagamento": pg_dt, "evento": "Juros", "tipo": "Juros", "taxa_percentual": 0.0},
            {"data_evento": horizon_dt, "data_pagamento": pg_dt, "evento": "Vencimento", "tipo": "Vencimento", "taxa_percentual": 100.0},
        ])
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
            classe=classe,
            emissor=emissor,
            call_date=call_date,
            is_pct_di=is_pct_di,
        )

    @staticmethod
    def indicative_spread_from_anbima(
        emissor: str,
        prazo_meses: float,
        tipo_taxa: str = "Indicativa",
        client=None,
    ) -> float:
        """
        Consulta a matriz aberta de Letras Financeiras no ANBIMA Data e interpola
        linearmente o spread indicativo (em decimal a.a.) para o emissor e prazo informados.
        """
        from bz_pricing.data.anbima_data import ANBIMADataClient

        cli = client or ANBIMADataClient()
        payload = cli.get_letras_financeiras()
        content = payload.get("content", []) if isinstance(payload, dict) else []
        if not content:
            raise ValueError("Matriz de Letras Financeiras vazia no ANBIMA Data.")

        emissor_up = emissor.strip().upper()
        tipo_up = tipo_taxa.strip().upper()
        vertices: List[Tuple[float, float]] = []

        for block in content:
            for item in block.get("emissores", []):
                if str(item.get("emissor", "")).strip().upper() == emissor_up and str(item.get("tipo_taxa", "")).strip().upper() == tipo_up:
                    for v in item.get("vertices", []):
                        taxa_val = Debenture._parse_br_float(v.get("taxa"), default=float("nan"))
                        if not math.isnan(taxa_val):
                            vertices.append((float(v["mes"]), taxa_val / 100.0))

        if not vertices:
            raise KeyError(f"Emissor '{emissor}' ({tipo_taxa}) não encontrado na matriz ANBIMA de Letras Financeiras.")

        vertices.sort(key=lambda x: x[0])
        if prazo_meses <= vertices[0][0]:
            return vertices[0][1]
        if prazo_meses >= vertices[-1][0]:
            return vertices[-1][1]

        for (m0, r0), (m1, r1) in zip(vertices[:-1], vertices[1:]):
            if m0 <= prazo_meses <= m1:
                if m1 == m0:
                    return r0
                w = (prazo_meses - m0) / (m1 - m0)
                return r0 + w * (r1 - r0)
        return vertices[-1][1]


def price_credit_portfolio(
    positions: List[Dict],
    base_date: Union[datetime.date, str],
    di_curve: Optional[Union[float, Curve]] = None,
) -> pd.DataFrame:
    """
    Precifica em lote uma carteira de ativos de crédito privado (Debêntures, CRI,
    CRA, FIDC e Letras Financeiras) utilizando o núcleo Cython, retornando um
    DataFrame consolidado com PU, valor financeiro, duration, DV01, convexidade
    e Z-Spread por ativo.

    Cada item de `positions` é um dicionário com:
    - `instrument`: instância de Debenture / CRI / CRA / CotaFIDC / LetraFinanceira
    - `discount_rate`: taxa indicativa ou curva de desconto
    - `quantity` (opcional, padrão 1.0): quantidade de títulos na carteira
    - `vna`, `pu_par_val`, `fator_di_acumulado`, `di_curve`, `risk_free_rate`,
      `ettj_pre_rate_or_curve`, `incentivada` (todos opcionais).
    """
    rows = []
    for pos in positions:
        inst: Debenture = pos["instrument"]
        qty = float(pos.get("quantity", 1.0))
        metrics = inst.risk_metrics(
            base_date=base_date,
            discount_rate_or_curve=pos["discount_rate"],
            vna=pos.get("vna"),
            last_coupon_date=pos.get("last_coupon_date"),
            di_curve=pos.get("di_curve", di_curve),
            pu_par_val=pos.get("pu_par_val"),
            fator_di_acumulado=pos.get("fator_di_acumulado"),
            risk_free_rate=pos.get("risk_free_rate"),
            ettj_pre_rate_or_curve=pos.get("ettj_pre_rate_or_curve"),
            incentivada=bool(pos.get("incentivada", False)),
        )
        rows.append({
            "ticker": inst.ticker,
            "classe": inst.__class__.__name__,
            "indexador": inst.indexador,
            "quantity": qty,
            "pu_par": metrics["pu_par"],
            "pu": metrics["pu"],
            "valor_financeiro": math.trunc((metrics["pu"] * qty) * 100.0) / 100.0,
            "cotacao": metrics["cotacao"],
            "duration_du": metrics["duration_du"],
            "modified_duration": metrics["modified_duration"],
            "dv01": metrics["dv01"],
            "dv01_financeiro": metrics["dv01"] * qty,
            "convexity": metrics["convexity"],
            "z_spread": metrics.get("z_spread"),
        })
    return pd.DataFrame(rows)
