import datetime
import io
import os
import re
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import requests

from bz_pricing.core.calendars import get_calendar


class DebenturesClient:
    """
    Cliente e API para extração automatizada de características cadastrais,
    agenda de fluxos de eventos financeiros (juros e amortizações), série histórica
    de PU Par e arquivos oficiais diários de mercado secundário da ANBIMA.
    """

    BASE_URL_SND = "https://www.debentures.com.br/exploreosnd/consultaadados"
    BASE_URL_ANBIMA_MERC_SEC = "https://www.anbima.com.br/informacoes/merc-sec-debentures/arqs"

    def __init__(self, data_dir: str = "dados/debentures", timeout: int = 15):
        self.data_dir = data_dir
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        })
        os.makedirs(self.data_dir, exist_ok=True)

    def _normalize_date(self, d: Union[datetime.date, str]) -> datetime.date:
        if isinstance(d, str):
            return pd.to_datetime(d).date()
        return d

    def _format_ticker(self, ticker: str) -> str:
        clean = ticker.strip().upper()
        # SND exige preenchimento com espaços ou '+' na URL da consulta
        return clean.ljust(10)

    def get_cashflows(self, ticker: str, force_download: bool = False) -> pd.DataFrame:
        """
        Extrai o cronograma completo de eventos financeiros (juros, amortizações e vencimento)
        para o ativo a partir do Sistema Nacional de Debêntures (SND).
        """
        ticker_clean = ticker.strip().upper()
        cache_file = os.path.join(self.data_dir, f"agenda_{ticker_clean}.csv")

        if not force_download and os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=["data_evento", "data_pagamento"])

        ticker_padded = ticker_clean.ljust(10).replace(" ", "+")
        url = f"{self.BASE_URL_SND}/eventosfinanceiros/agenda_e.asp?ativo={ticker_padded}"

        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        # O arquivo exportado contém metadados nas duas primeiras linhas
        raw_text = resp.text
        lines = [l for l in raw_text.splitlines() if "\t" in l]
        if not lines:
            raise ValueError(f"Não foram encontrados fluxos de eventos para o ativo {ticker_clean}.")

        df = pd.read_csv(io.StringIO(raw_text), sep="\t", skiprows=2, index_col=False, encoding="latin1")

        # Renomeação e higienização das colunas
        rename_map = {
            "Data do Evento": "data_evento",
            "Data do Pagamento": "data_pagamento",
            "Emissor": "emissor",
            "Ativo": "ativo",
            "Evento": "evento",
            "Tipo": "tipo",
            "Taxa/Percentual": "taxa_percentual",
            "Liquidaçăo": "liquidado"
        }
        df = df.rename(columns=rename_map)

        for col in ["data_evento", "data_pagamento"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce").dt.date

        if "taxa_percentual" in df.columns:
            df["taxa_percentual"] = (
                df["taxa_percentual"]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df["taxa_percentual"] = pd.to_numeric(df["taxa_percentual"], errors="coerce").fillna(0.0)

        for col in ["emissor", "ativo", "evento", "tipo"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        df = df.sort_values(by=["data_pagamento", "data_evento"]).reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        return df

    def get_characteristics(self, ticker: str, force_download: bool = False) -> Dict[str, any]:
        """
        Recupera os parâmetros cadastrais e termos de emissão da debênture
        (emissor, indexador, taxa de juros, tipo de amortização, datas de início e vencimento).
        """
        ticker_clean = ticker.strip().upper()
        cache_file = os.path.join(self.data_dir, f"carac_{ticker_clean}.csv")

        if not force_download and os.path.exists(cache_file):
            df_cached = pd.read_csv(cache_file, encoding="utf-8")
            return df_cached.iloc[0].to_dict()

        url = f"{self.BASE_URL_SND}/emissoesdedebentures/caracteristicas_e.asp?Ativo={ticker_clean}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text), sep="\t", skiprows=3, encoding="latin1")
        if df.empty:
            raise ValueError(f"Não foram encontradas características cadastrais para o ativo {ticker_clean}.")

        row = df.iloc[0].to_dict()
        clean_dict = {}
        for k, v in row.items():
            k_clean = str(k).strip()
            if pd.isna(v):
                clean_dict[k_clean] = None
            elif isinstance(v, str):
                clean_dict[k_clean] = v.strip()
            else:
                clean_dict[k_clean] = v

        pd.DataFrame([clean_dict]).to_csv(cache_file, index=False, encoding="utf-8")
        return clean_dict

    def get_pu_history(self, ticker: str, start_date: Union[datetime.date, str] = None,
                       end_date: Union[datetime.date, str] = None,
                       force_download: bool = False) -> pd.DataFrame:
        """
        Extrai o histórico oficial diário de Preço Unitário Par (PU Par),
        Valor Nominal Atualizado (VNA) e juros acruados apurados pelo SND.
        """
        ticker_clean = ticker.strip().upper()
        s_dt = self._normalize_date(start_date) if start_date else datetime.date(2024, 1, 1)
        e_dt = self._normalize_date(end_date) if end_date else datetime.date(2026, 12, 31)

        dt_ini_str = s_dt.strftime("%d/%m/%Y")
        dt_fim_str = e_dt.strftime("%d/%m/%Y")

        ticker_padded = ticker_clean.ljust(10).replace(" ", "+")
        url = (
            f"{self.BASE_URL_SND}/emissoesdedebentures/puhistorico_e.asp"
            f"?op_exc=False&ativo={ticker_padded}&dt_ini={dt_ini_str}&dt_fim={dt_fim_str}"
        )

        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text), sep="\t", skiprows=2, index_col=False, encoding="latin1")

        rename_map = {
            "Data do PU": "data",
            "Ativo": "ativo",
            "Valor Nominal": "vna",
            "Juros": "juros",
            "Pręmio": "premio",
            "Preço Unitário": "pu_par",
            "Critério de Cálculo": "criterio",
            "Situaçăo": "situacao"
        }
        df = df.rename(columns=rename_map)

        if "data" in df.columns:
            df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce").dt.date

        for num_col in ["vna", "juros", "pu_par"]:
            if num_col in df.columns:
                df[num_col] = (
                    df[num_col]
                    .astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0.0)

        df = df.sort_values(by="data").reset_index(drop=True)
        return df

    def get_event_pus(self, ticker: str, force_download: bool = False) -> pd.DataFrame:
        """
        Extrai o histórico de eventos financeiros já liquidados (juros e amortizações passadas)
        apurados pelo SND (arquivo pudeeventos_e.asp).
        """
        ticker_clean = ticker.strip().upper()
        cache_file = os.path.join(self.data_dir, f"pueventos_{ticker_clean}.csv")

        if not force_download and os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=["data"])

        ticker_padded = ticker_clean.ljust(10).replace(" ", "+")
        url = f"{self.BASE_URL_SND}/eventosfinanceiros/pudeeventos_e.asp?ativo={ticker_padded}"

        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        raw_text = resp.text
        lines = [l for l in raw_text.splitlines() if "\t" in l]
        if not lines:
            return pd.DataFrame(columns=["data", "ativo", "tipo_evento", "pu_evento", "situacao", "liquidacao"])

        df = pd.read_csv(io.StringIO(raw_text), sep="\t", skiprows=2, index_col=False, encoding="latin1")

        rename_map = {
            "Data": "data",
            "Ativo": "ativo",
            "Tipo de Evento": "tipo_evento",
            "PU   de Evento": "pu_evento",
            "Situação": "situacao",
            "Liquidação": "liquidacao"
        }
        df = df.rename(columns=rename_map)

        if "data" in df.columns:
            df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce").dt.date

        if "pu_evento" in df.columns:
            df["pu_evento"] = (
                df["pu_evento"]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df["pu_evento"] = pd.to_numeric(df["pu_evento"], errors="coerce").fillna(0.0)

        for c in ["ativo", "tipo_evento", "situacao", "liquidacao"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        df = df.sort_values(by="data").reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        return df

    def get_market_prices(self, date: Union[datetime.date, str], force_download: bool = False) -> pd.DataFrame:
        """
        Baixa e consolida a tabela de mercado secundário de debêntures divulgada
        diariamente pela ANBIMA (arquivo oficial dbYYMMDD.txt).
        """
        dt = self._normalize_date(date)
        date_str = dt.strftime("%y%m%d")
        cache_file = os.path.join(self.data_dir, f"db{date_str}.txt")

        if force_download or not os.path.exists(cache_file):
            url = f"{self.BASE_URL_ANBIMA_MERC_SEC}/db{date_str}.txt"
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                raise FileNotFoundError(
                    f"Arquivo ANBIMA para {dt.strftime('%d/%m/%Y')} não localizado (HTTP {resp.status_code})."
                )
            with open(cache_file, "wb") as f:
                f.write(resp.content)

        df = pd.read_csv(cache_file, sep="@", skiprows=2, encoding="latin1")

        rename_map = {
            "Código": "codigo",
            "Nome": "nome",
            "Repac./  Venc.": "vencimento",
            "Índice/ Correção": "indice_correcao",
            "Taxa de Compra": "taxa_compra",
            "Taxa de Venda": "taxa_venda",
            "Taxa Indicativa": "taxa_indicativa",
            "Desvio Padrão": "desvio_padrao",
            "Intervalo Indicativo Minimo": "intervalo_min",
            "Intervalo Indicativo Máximo": "intervalo_max",
            "PU": "pu_mercado",
            "% PU Par / % VNE": "perc_pu_par",
            "Duration": "duration",
            "% Reune": "perc_reune",
            "Referência NTN-B": "ntnb_referencia"
        }
        df = df.rename(columns=rename_map)

        for str_col in ["codigo", "nome", "indice_correcao", "ntnb_referencia"]:
            if str_col in df.columns:
                df[str_col] = df[str_col].astype(str).str.strip().str.replace(r"\s*\(\*\).*", "", regex=True)

        if "vencimento" in df.columns:
            df["vencimento"] = pd.to_datetime(df["vencimento"], format="%d/%m/%Y", errors="coerce").dt.date

        num_cols = ["taxa_compra", "taxa_venda", "taxa_indicativa", "desvio_padrao",
                    "intervalo_min", "intervalo_max", "pu_mercado", "perc_pu_par", "duration"]
        for num_col in num_cols:
            if num_col in df.columns:
                df[num_col] = (
                    df[num_col]
                    .astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        return df.reset_index(drop=True)


DebenturesAPI = DebenturesClient
