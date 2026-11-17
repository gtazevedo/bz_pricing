import calendar
import datetime
import math
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


NTNB_BASE_DATE = datetime.date(2000, 7, 15)
NTNB_BASE_VALUE = 1000.0
LFT_BASE_DATE = datetime.date(2000, 7, 1)
LFT_BASE_VALUE = 1000.0
NTNC_BASE_DATE = datetime.date(2000, 7, 1)
NTNC_BASE_VALUE = 1000.0


def _fetch_sgs_chunked(series_id: int, start_date: datetime.date, end_date: datetime.date, chunk_years: int = 9) -> pd.Series:
    """Consulta uma série do BCB SGS em janelas de até `chunk_years` anos (contornando limite de 10 anos em séries diárias)."""
    from bcb import sgs

    parts = []
    cur_start = start_date
    while cur_start <= end_date:
        cur_end = min(end_date, datetime.date(cur_start.year + chunk_years, cur_start.month, min(28, cur_start.day)))
        if cur_end < cur_start:
            cur_end = end_date
        df_part = sgs.get({"VAL": series_id}, start=cur_start.strftime("%Y-%m-%d"), end=cur_end.strftime("%Y-%m-%d"))
        if df_part is not None and not df_part.empty:
            parts.append(df_part["VAL"].dropna())
        cur_start = cur_end + datetime.timedelta(days=1)
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts)
    s.index = pd.to_datetime(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _coerce_monthly_series(data: Any) -> pd.Series:
    """Converte dict, Series ou DataFrame ('Date'/'Data', 'Value'/'Valor') em Series mensal em decimal."""
    if data is None:
        return pd.Series(dtype=float)
    if isinstance(data, pd.Series):
        s = data.copy()
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
        return s.astype(float).sort_index()
    if isinstance(data, dict):
        if not data:
            return pd.Series(dtype=float)
        idx = pd.to_datetime(list(data.keys())).to_period("M").to_timestamp()
        return pd.Series(list(data.values()), index=idx, dtype=float).sort_index()
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        d_col = next((c for c in df.columns if str(c).lower() in ("date", "data", "ref", "mes")), df.columns[0])
        v_col = next((c for c in df.columns if str(c).lower() in ("value", "valor", "ipca", "taxa")), df.columns[-1])
        idx = pd.to_datetime(df[d_col]).dt.to_period("M").dt.to_timestamp()
        return pd.Series(df[v_col].astype(float).values, index=idx).sort_index()
    raise TypeError(f"Formato não suportado para série mensal de IPCA: {type(data)}")


class VNAEngine:
    """
    Engine para estimar e projetar o Valor Nominal Atualizado (VNA) da ANBIMA / Tesouro Nacional
    com base no IPCA/IGP-M histórico (SGS ou informado pelo usuário) e projeções (Focus ou explícita).
    """
    def __init__(self, base_vna_date: datetime.date, base_vna_value: float, data_dir: str = None):
        self.base_vna_date = pd.to_datetime(base_vna_date).date()
        self.base_vna_value = float(base_vna_value)
        self.data_dir = data_dir
        self.ipca_history = pd.Series(dtype=float)
        self.ipca_projections = pd.Series(dtype=float)

    def load_ipca_history(self, csv_path: str):
        """
        Carrega histórico de IPCA mensal.
        O CSV deve ter pelo menos 'Date' e 'Value' (em %).
        """
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
        df = df.set_index('Date').sort_index()
        if 'Value' in df.columns:
            self.ipca_history = df['Value'] / 100.0
            
    def load_ipca_projections(self, csv_path: str):
        """
        Carrega projeções do Focus.
        O CSV deve ter 'Date' (mês de referência) e 'Value' (mediana projetada em %).
        """
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
        df = df.set_index('Date').sort_index()
        if 'Value' in df.columns:
            self.ipca_projections = df['Value'] / 100.0

    def get_ipca_for_month(self, target_month: datetime.date) -> float:
        """
        Retorna o IPCA (histórico ou projetado) para o mês alvo.
        """
        target_ts = pd.to_datetime(target_month).replace(day=1)
        if target_ts in self.ipca_history.index and pd.notna(self.ipca_history.loc[target_ts]):
            return float(self.ipca_history.loc[target_ts])
        elif target_ts in self.ipca_projections.index and pd.notna(self.ipca_projections.loc[target_ts]):
            return float(self.ipca_projections.loc[target_ts])
        else:
            raise ValueError(f"Faltam dados de IPCA (histórico ou projetado) para {target_ts.strftime('%Y-%m')}")

    @classmethod
    def from_bcb(cls, base_vna_date: datetime.date, base_vna_value: float,
                 indicator: str = "IPCA", ref_date: datetime.date = None) -> "VNAEngine":
        """
        Constrói um VNAEngine populado com a série histórica oficial do BCB (SGS)
        e as expectativas mensais do Relatório Focus.
        """
        from bcb import sgs
        from bz_pricing.data.scrapers import B3Scraper

        ind_upper = indicator.upper().strip()
        series_id = 189 if "IGP" in ind_upper else 433
        focus_ind = "IGP-M" if "IGP" in ind_upper else "IPCA"

        base_dt = pd.to_datetime(base_vna_date).date()
        end_dt = pd.to_datetime(ref_date).date() if ref_date else datetime.date.today()
        start_hist = (base_dt.replace(day=1) - pd.DateOffset(months=3)).date()

        engine = cls(base_vna_date=base_dt, base_vna_value=base_vna_value)
        try:
            df_hist = sgs.get({"VAL": series_id}, start=start_hist.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
            if df_hist is not None and not df_hist.empty:
                s_hist = df_hist["VAL"].dropna() / 100.0
                s_hist.index = pd.to_datetime(s_hist.index).to_period("M").to_timestamp()
                engine.ipca_history = s_hist[~s_hist.index.duplicated(keep="last")].sort_index()
        except Exception:
            pass

        try:
            scraper = B3Scraper()
            df_focus = scraper.download_focus_expectations(indicator=focus_ind, limit=200, base_date=end_dt)
            if df_focus is not None and not df_focus.empty:
                df_f = df_focus.copy()
                if "DataReferencia" in df_f.columns and "Mediana" in df_f.columns:
                    df_f["RefTS"] = pd.to_datetime(df_f["DataReferencia"], format="%m/%Y", errors="coerce")
                    df_f = df_f.dropna(subset=["RefTS", "Mediana"]).drop_duplicates(subset=["RefTS"], keep="first")
                    s_proj = pd.Series(df_f["Mediana"].astype(float).values / 100.0, index=df_f["RefTS"]).sort_index()
                    engine.ipca_projections = s_proj
        except Exception:
            pass

        return engine

    @classmethod
    def for_ntnb(
        cls,
        ref_date: Optional[Union[datetime.date, str]] = None,
        ipca_history: Any = None,
        ipca_projections: Any = None,
        allow_download: bool = True,
    ) -> "VNAEngine":
        """
        Constrói um VNAEngine configurado para NTN-B (data-base 15/07/2000, VNA base R$ 1.000,00).
        Aceita séries mensais offline de IPCA (`ipca_history` e `ipca_projections` em decimal)
        ou baixa automaticamente do BCB/Focus quando `allow_download=True`.
        """
        if ipca_history is not None:
            engine = cls(base_vna_date=NTNB_BASE_DATE, base_vna_value=NTNB_BASE_VALUE)
            engine.ipca_history = _coerce_monthly_series(ipca_history)
            if ipca_projections is not None:
                engine.ipca_projections = _coerce_monthly_series(ipca_projections)
            return engine

        if not allow_download:
            raise ValueError(
                "Informe 'ipca_history' para calcular o VNA da NTN-B offline (allow_download=False)."
            )

        engine = cls.from_bcb(
            base_vna_date=NTNB_BASE_DATE,
            base_vna_value=NTNB_BASE_VALUE,
            indicator="IPCA",
            ref_date=ref_date,
        )
        if ipca_projections is not None:
            engine.ipca_projections = _coerce_monthly_series(ipca_projections)
        return engine

    def calculate_vna(
        self,
        target_date: Union[datetime.date, str],
        cal: Any = None,
        anniversary_day: int = 15,
        ipca_proj: Optional[float] = None,
        truncate_monthly: bool = True,
    ) -> float:
        """
        Calcula o VNA projetado para `target_date`, partindo de `base_vna_date`.
        Se `cal` for omitido, utiliza o calendário padrão B3/ANBIMA.
        `ipca_proj` permite informar explicitamente a projeção mensal (decimal) para o período pró-rata aberto.
        """
        if cal is None:
            from bz_pricing.core.calendars import get_calendar
            cal = get_calendar("B3")

        target_date = pd.to_datetime(target_date).date()
        if target_date == self.base_vna_date:
            return self.base_vna_value

        if target_date < self.base_vna_date:
            raise NotImplementedError("Cálculo retroativo de VNA não implementado ainda.")

        current_date = self.base_vna_date
        current_vna = float(self.base_vna_value)

        while current_date < target_date:
            if current_date.day < anniversary_day:
                last_dom = calendar.monthrange(current_date.year, current_date.month)[1]
                next_anniv = current_date.replace(day=min(anniversary_day, last_dom))
            else:
                next_m = (current_date.replace(day=1) + pd.DateOffset(months=1)).date()
                last_dom = calendar.monthrange(next_m.year, next_m.month)[1]
                next_anniv = next_m.replace(day=min(anniversary_day, last_dom))

            ref_month = (next_anniv.replace(day=1) - pd.DateOffset(months=1))

            if target_date <= next_anniv and ipca_proj is not None:
                ipca_rate = float(ipca_proj)
            else:
                try:
                    ipca_rate = self.get_ipca_for_month(ref_month)
                except ValueError:
                    if not self.ipca_projections.empty:
                        ipca_rate = float(self.ipca_projections.iloc[-1])
                    else:
                        raise

            if target_date == next_anniv:
                fator = 1.0 + ipca_rate
                current_vna = current_vna * fator
                if truncate_monthly:
                    current_vna = math.trunc(current_vna * 1000000.0) / 1000000.0
                break
            elif target_date < next_anniv:
                try:
                    du_calc = cal.bizdays(current_date, target_date)
                    du_total = cal.bizdays(current_date, next_anniv)
                except Exception:
                    du_calc = int(np.busday_count(current_date, target_date))
                    du_total = int(np.busday_count(current_date, next_anniv))
                fator = (1.0 + ipca_rate) ** (du_calc / du_total) if du_total > 0 else 1.0
                current_vna = math.trunc((current_vna * fator) * 1000000.0) / 1000000.0
                break
            else:
                fator = (1.0 + ipca_rate)
                current_vna = current_vna * fator
                if truncate_monthly:
                    current_vna = math.trunc(current_vna * 1000000.0) / 1000000.0
                current_date = next_anniv

        return float(math.trunc(current_vna * 1000000.0) / 1000000.0)


def calculate_ntnb_vna(
    target_date: Union[datetime.date, str],
    ipca_proj: Optional[float] = None,
    ipca_history: Any = None,
    ipca_projections: Any = None,
    cal: Any = None,
    allow_download: bool = True,
    truncate_monthly: bool = False,
) -> float:
    """
    Calcula o Valor Nominal Atualizado (VNA) da NTN-B na data `target_date`
    (base R$ 1.000,00 em 15/07/2000), aceitando séries offline ou baixando do BCB/Focus.
    """
    target_dt = pd.to_datetime(target_date).date()
    engine = VNAEngine.for_ntnb(
        ref_date=target_dt,
        ipca_history=ipca_history,
        ipca_projections=ipca_projections,
        allow_download=allow_download,
    )
    return engine.calculate_vna(
        target_dt,
        cal=cal,
        anniversary_day=15,
        ipca_proj=ipca_proj,
        truncate_monthly=truncate_monthly,
    )


def calculate_selic_or_di_vna(
    vne: float,
    base_date: Union[datetime.date, str],
    target_date: Union[datetime.date, str],
    indicator: str = "SELIC",
    daily_rates: Any = None,
    allow_download: bool = True,
) -> float:
    """
    Calcula o VNA a partir do VNE acumulando fatores diários da Selic (SGS 11) ou DI (SGS 12)
    no intervalo [base_date, target_date - 1 dia], truncado em 6 casas decimais.
    `daily_rates` (opcional) aceita dict/Series/DataFrame com taxas diárias em decimal.
    """
    base_dt = pd.to_datetime(base_date).date()
    target_dt = pd.to_datetime(target_date).date()
    if target_dt <= base_dt:
        return float(vne)

    end_acc = target_dt - datetime.timedelta(days=1)
    if daily_rates is not None:
        if isinstance(daily_rates, pd.Series):
            s = daily_rates.copy()
            s.index = pd.to_datetime(s.index)
        elif isinstance(daily_rates, dict):
            s = pd.Series(list(daily_rates.values()), index=pd.to_datetime(list(daily_rates.keys())), dtype=float)
        elif isinstance(daily_rates, pd.DataFrame):
            d_col = next((c for c in daily_rates.columns if str(c).lower() in ("date", "data")), daily_rates.columns[0])
            v_col = next((c for c in daily_rates.columns if str(c).lower() in ("value", "valor", "rate", "taxa")), daily_rates.columns[-1])
            s = pd.Series(daily_rates[v_col].astype(float).values, index=pd.to_datetime(daily_rates[d_col]))
        else:
            raise TypeError(f"Formato não suportado para daily_rates: {type(daily_rates)}")
        s = s.sort_index()
        s = s[(s.index.date >= base_dt) & (s.index.date <= end_acc)]
    else:
        if not allow_download:
            raise ValueError(f"Informe 'daily_rates' para calcular VNA {indicator} offline (allow_download=False).")
        series_id = 12 if "DI" in indicator.upper() and "SELIC" not in indicator.upper() else 11
        s = _fetch_sgs_chunked(series_id, base_dt, end_acc) / 100.0

    factor = float((1.0 + s.astype(float)).prod()) if len(s) > 0 else 1.0
    return float(math.trunc((float(vne) * factor) * 1000000.0) / 1000000.0)


def calculate_vna_from_vne(
    vne: float,
    base_date: Union[datetime.date, str],
    target_date: Union[datetime.date, str],
    indexador: str = "IPCA",
    anniversary_day: Optional[int] = None,
    proj: Optional[float] = None,
    history: Any = None,
    projections: Any = None,
    cal: Any = None,
    allow_download: bool = True,
    truncate_monthly: bool = False,
) -> float:
    """
    Calcula o VNA na data `target_date` a partir do Valor Nominal de Emissão (`vne`)
    e data-base (`base_date`) para qualquer indexador compatível:
    IPCA, IGP-M, SELIC, CDI/DI ou PRE/PREFIXADO.
    """
    idx = indexador.upper().strip()
    if idx in ("PRE", "PREFIXADO", "PRÉ", "FIXED"):
        return float(vne)
    if "SELIC" in idx or idx in ("DI", "CDI"):
        return calculate_selic_or_di_vna(
            vne=vne,
            base_date=base_date,
            target_date=target_date,
            indicator=idx,
            daily_rates=history,
            allow_download=allow_download,
        )

    base_dt = pd.to_datetime(base_date).date()
    target_dt = pd.to_datetime(target_date).date()
    anniv = int(anniversary_day) if anniversary_day is not None else (1 if "IGP" in idx else 15)

    if history is not None:
        engine = VNAEngine(base_vna_date=base_dt, base_vna_value=float(vne))
        engine.ipca_history = _coerce_monthly_series(history)
        if projections is not None:
            engine.ipca_projections = _coerce_monthly_series(projections)
    else:
        if not allow_download:
            raise ValueError(f"Informe 'history' para calcular VNA {idx} offline (allow_download=False).")
        engine = VNAEngine.from_bcb(base_vna_date=base_dt, base_vna_value=float(vne), indicator=idx, ref_date=target_dt)
        if projections is not None:
            engine.ipca_projections = _coerce_monthly_series(projections)

    return engine.calculate_vna(
        target_dt,
        cal=cal,
        anniversary_day=anniv,
        ipca_proj=proj,
        truncate_monthly=truncate_monthly,
    )


def calculate_lft_vna(
    target_date: Union[datetime.date, str],
    vne: float = LFT_BASE_VALUE,
    base_date: Union[datetime.date, str] = LFT_BASE_DATE,
    daily_selic: Any = None,
    allow_download: bool = True,
) -> float:
    """Calcula o VNA oficial da LFT (base R$ 1.000,00 em 01/07/2000) pela Selic acumulada."""
    return calculate_selic_or_di_vna(
        vne=vne,
        base_date=base_date,
        target_date=target_date,
        indicator="SELIC",
        daily_rates=daily_selic,
        allow_download=allow_download,
    )


def calculate_ntnc_vna(
    target_date: Union[datetime.date, str],
    vne: float = NTNC_BASE_VALUE,
    base_date: Union[datetime.date, str] = NTNC_BASE_DATE,
    igpm_proj: Optional[float] = None,
    igpm_history: Any = None,
    igpm_projections: Any = None,
    cal: Any = None,
    allow_download: bool = True,
    truncate_monthly: bool = False,
) -> float:
    """Calcula o VNA da NTN-C a partir do VNE/data-base pelo IGP-M (aniversário dia 1)."""
    return calculate_vna_from_vne(
        vne=vne,
        base_date=base_date,
        target_date=target_date,
        indexador="IGP-M",
        anniversary_day=1,
        proj=igpm_proj,
        history=igpm_history,
        projections=igpm_projections,
        cal=cal,
        allow_download=allow_download,
        truncate_monthly=truncate_monthly,
    )
