import datetime
import io
import json
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple, Union
from dateutil.relativedelta import relativedelta
import pandas as pd
import requests

from bz_pricing.core.calendars import get_calendar


class CRICRAClient:
    """
    Cliente para extração, leitura de arquivos locais do ANBIMA Data e
    geração de agendas de fluxos de caixa de CRI e CRA.
    """

    ANBIMA_OAUTH_URL = "https://api.anbima.com.br/oauth/access-token"
    ANBIMA_FEED_URL = "https://api.anbima.com.br/feed/precos-indices/v1/cri-cra/mercado-secundario"
    ANBIMA_PUBLIC_URL = (
        "https://www.anbima.com.br/pt_br/informar/precos-e-indices/precos/"
        "taxas-de-cri-e-cra/taxas-de-cri-e-cra.htm"
    )

    _FREQ_MONTHS = {
        "mensal": 1,
        "monthly": 1,
        "trimestral": 3,
        "quarterly": 3,
        "semestral": 6,
        "semiannual": 6,
        "anual": 12,
        "annual": 12,
        "bullet": 0,
    }

    def __init__(self,
                 data_dir: str = "dados/cricra",
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 timeout: int = 15):
        self.data_dir = data_dir
        self.client_id = client_id or os.environ.get("ANBIMA_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("ANBIMA_CLIENT_SECRET")
        self.timeout = timeout
        self.session = requests.Session()
        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def _normalize_date(d: Union[datetime.date, str]) -> datetime.date:
        if isinstance(d, str):
            return pd.to_datetime(d, dayfirst=True).date() if "/" in d else pd.to_datetime(d).date()
        if isinstance(d, datetime.datetime):
            return d.date()
        return d

    def _get_access_token(self) -> str:
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Credenciais ANBIMA Developers ausentes. Informe 'client_id' e 'client_secret' "
                "ou defina as variáveis de ambiente ANBIMA_CLIENT_ID e ANBIMA_CLIENT_SECRET."
            )
        resp = self.session.post(
            self.ANBIMA_OAUTH_URL,
            auth=(self.client_id, self.client_secret),
            json={"grant_type": "client_credentials"},
            timeout=self.timeout
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Falha na autenticação OAuth2 ANBIMA (HTTP {resp.status_code}): {resp.text}")
        token = resp.json().get("access_token")
        if not token:
            raise RuntimeError("Resposta OAuth2 ANBIMA não retornou 'access_token'.")
        return token

    def get_anbima_public_prices(self, force_download: bool = False) -> pd.DataFrame:
        """
        Extrai a tabela diária pública de taxas indicativas, PUs e durations de CRI/CRA
        divulgada no portal da ANBIMA (taxas-de-cri-e-cra.htm), com cache local em CSV.
        """
        today_str = datetime.date.today().strftime("%y%m%d")
        cache_file = os.path.join(self.data_dir, f"cricra_public_{today_str}.csv")
        if not force_download and os.path.exists(cache_file):
            return self.load_market_file(cache_file)

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = self.session.get(self.ANBIMA_PUBLIC_URL, headers=headers, timeout=max(self.timeout, 25))
        if resp.status_code != 200:
            raise RuntimeError(f"Erro ao acessar página pública da ANBIMA (HTTP {resp.status_code}).")

        tables = pd.read_html(io.StringIO(resp.text), decimal=",", thousands=".")
        target_df = None
        for tbl in tables:
            cols_str = " ".join(str(c) for c in tbl.columns)
            if "Código" in cols_str and "PU" in cols_str:
                target_df = tbl
                break
        if target_df is None or target_df.empty:
            raise ValueError("Tabela de preços CRI/CRA não localizada na página pública da ANBIMA.")

        df = self._standardize_market_df(target_df)
        df.to_csv(cache_file, index=False)
        return df

    def get_anbima_data_asset(self, ticker: str, force_download: bool = False) -> Dict[str, dict]:
        """
        Extrai características cadastrais, preços indicativos/PU histórico e agenda completa
        de eventos de um CRI ou CRA diretamente do portal ANBIMA Data, com cache em JSON.
        """
        ticker_clean = ticker.strip().upper()
        cache_file = os.path.join(self.data_dir, f"anbima_data_{ticker_clean}.json")
        if not force_download and os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        captured: Dict[str, dict] = {}
        errors: List[str] = []

        def _run():
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()

                    def on_response(resp):
                        u = resp.url
                        prefix = f"web-bff/v1/certificado-recebiveis/{ticker_clean}"
                        if prefix in u and resp.status == 200:
                            try:
                                suffix = u.split(prefix, 1)[1] or "resumo"
                                suffix = suffix.lstrip("/").split("?")[0] or "resumo"
                                if suffix == "precos/pu-historico":
                                    captured.setdefault("pu_historico_pages", []).append(resp.json())
                                else:
                                    captured[suffix] = resp.json()
                            except Exception:
                                pass

                    page.on("response", on_response)
                    for tab in ("caracteristicas", "precos", "agenda"):
                        url = f"https://data.anbima.com.br/certificado-de-recebiveis/{ticker_clean}/{tab}"
                        page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_timeout(3000)
                    browser.close()
            except Exception as exc:
                errors.append(str(exc))

        t = threading.Thread(target=_run)
        t.start()
        t.join(timeout=120)

        if errors:
            raise RuntimeError(f"Erro ao extrair {ticker_clean} no ANBIMA Data: {errors[0]}")
        if not captured:
            raise FileNotFoundError(f"Nenhum dado encontrado no ANBIMA Data para o ativo {ticker_clean}.")

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(captured, f, ensure_ascii=False, indent=2)
        return captured

    def get_asset_characteristics(self, ticker: str, force_download: bool = False) -> Dict:
        """Retorna as características cadastrais do CRI/CRA no ANBIMA Data."""
        data = self.get_anbima_data_asset(ticker, force_download=force_download)
        return data.get("caracteristicas", data.get("resumo", {}))

    def get_asset_prices(self, ticker: str, force_download: bool = False) -> Dict:
        """Retorna PU histórico, PU indicativo e histórico de preços do ativo no ANBIMA Data."""
        data = self.get_anbima_data_asset(ticker, force_download=force_download)
        return data.get("precos", {})

    def get_asset_cashflows(self, ticker: str, force_download: bool = False) -> pd.DataFrame:
        """
        Retorna a agenda de eventos financeiros (juros, amortizações e vencimento)
        extraída do ANBIMA Data no formato padrão da calculadora.
        """
        data = self.get_anbima_data_asset(ticker, force_download=force_download)
        agenda = data.get("agenda", {})
        content = agenda.get("content", []) if isinstance(agenda, dict) else []
        if not content:
            return pd.DataFrame(columns=["data_evento", "data_pagamento", "evento", "tipo", "taxa_percentual", "status", "valor"])

        rows = []
        for item in content:
            ev_raw = str(item.get("evento_arc") or item.get("evento") or "").strip()
            ev_desc = str(item.get("evento") or "").upper()
            if "VENCIMENTO" in ev_desc or "RESGATE" in ev_desc:
                ev_std = "Vencimento"
            elif "AMORT" in ev_raw.upper() or "AMORT" in ev_desc:
                ev_std = "Amortização"
            elif "JURO" in ev_raw.upper() or "JURO" in ev_desc:
                ev_std = "Juros"
            else:
                ev_std = ev_raw

            taxa_val = item.get("taxa")
            try:
                taxa_pct = float(taxa_val) if taxa_val is not None else 0.0
            except (TypeError, ValueError):
                taxa_pct = 0.0

            status_obj = item.get("status")
            status_str = status_obj.get("status", "") if isinstance(status_obj, dict) else str(status_obj or "")

            rows.append({
                "data_evento": pd.to_datetime(item.get("data_evento")).date(),
                "data_pagamento": pd.to_datetime(item.get("data_liquidacao") or item.get("data_evento")).date(),
                "evento": ev_std,
                "tipo": "Regular",
                "taxa_percentual": taxa_pct,
                "status": status_str,
                "valor": item.get("valor"),
            })

        df = pd.DataFrame(rows)
        order_map = {"Juros": 0, "Amortização": 1, "Vencimento": 2}
        df["_ord"] = df["evento"].map(lambda x: order_map.get(x, 9))
        return df.sort_values(["data_pagamento", "_ord"]).drop(columns=["_ord"]).reset_index(drop=True)

    def get_market_prices(self, date: Optional[Union[datetime.date, str]] = None,
                          force_download: bool = False) -> pd.DataFrame:
        """
        Consulta os preços e taxas indicativas de CRI/CRA divulgados pela ANBIMA.
        Utiliza a API ANBIMA Developers quando há credenciais configuradas e 'date'
        informada; caso contrário, extrai a tabela pública diária da ANBIMA.
        """
        if date is None or not (self.client_id and self.client_secret):
            return self.get_anbima_public_prices(force_download=force_download)

        dt = self._normalize_date(date)
        date_iso = dt.strftime("%Y-%m-%d")
        cache_file = os.path.join(self.data_dir, f"cricra_{dt.strftime('%y%m%d')}.csv")

        if not force_download and os.path.exists(cache_file):
            return self.load_market_file(cache_file)

        token = self._get_access_token()
        headers = {
            "client_id": self.client_id,
            "access_token": token,
            "Accept": "application/json",
        }
        resp = self.session.get(
            self.ANBIMA_FEED_URL,
            headers=headers,
            params={"data": date_iso},
            timeout=self.timeout
        )
        if resp.status_code != 200:
            raise FileNotFoundError(
                f"Dados de CRI/CRA para {dt.strftime('%d/%m/%Y')} indisponíveis na API ANBIMA (HTTP {resp.status_code})."
            )

        payload = resp.json()
        records = payload if isinstance(payload, list) else payload.get("data", payload.get("content", []))
        df = pd.DataFrame(records)
        if not df.empty:
            df = self._standardize_market_df(df)
            df.to_csv(cache_file, index=False)
        return df

    def load_market_file(self, file_path: str) -> pd.DataFrame:
        """
        Lê e padroniza arquivo exportado manualmente do portal ANBIMA Data (CSV, TXT, XLS ou XLSX).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(file_path)
        elif ext == ".txt":
            df = pd.read_csv(file_path, sep="@", encoding="latin1")
        else:
            df = pd.read_csv(file_path, encoding="utf-8")

        return self._standardize_market_df(df)

    @staticmethod
    def _standardize_market_df(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Código": "codigo",
            "Codigo": "codigo",
            "codigo_ativo": "codigo",
            "Ativo": "tipo_ativo",
            "Tipo": "tipo_ativo",
            "Securitizadora": "securitizadora",
            "Emissor": "securitizadora",
            "emissor": "securitizadora",
            "Risco de Crédito": "risco_credito",
            "Data de Referência": "data_referencia",
            "Série": "serie",
            "Emissão": "emissao",
            "Vencimento": "vencimento",
            "Data de Vencimento": "vencimento",
            "data_vencimento": "vencimento",
            "Índice/ Correção": "indice_correcao",
            "Índice / Correção": "indice_correcao",
            "Indexador": "indice_correcao",
            "indexador": "indice_correcao",
            "Taxa de Compra": "taxa_compra",
            "Taxa Compra": "taxa_compra",
            "Taxa de Venda": "taxa_venda",
            "Taxa Venda": "taxa_venda",
            "Taxa Indicativa": "taxa_indicativa",
            "taxa_indicativa": "taxa_indicativa",
            "PU": "pu_mercado",
            "pu": "pu_mercado",
            "% PU Par": "perc_pu_par",
            "% PU Par / % VNE": "perc_pu_par",
            "Duration": "duration",
            "duration": "duration",
            "Referência NTN-B": "ntnb_referencia",
            "Referência NTNB": "ntnb_referencia",
        }
        out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()

        for col in ("codigo", "tipo_ativo", "securitizadora", "indice_correcao", "ntnb_referencia"):
            if col in out.columns:
                out[col] = out[col].astype(str).str.strip()

        if "vencimento" in out.columns:
            sample_v = str(out["vencimento"].dropna().iloc[0]) if not out["vencimento"].dropna().empty else ""
            out["vencimento"] = pd.to_datetime(
                out["vencimento"], dayfirst="/" in sample_v, errors="coerce"
            ).dt.date

        num_cols = ["taxa_compra", "taxa_venda", "taxa_indicativa", "pu_mercado", "perc_pu_par", "duration"]
        for col in num_cols:
            if col in out.columns and out[col].dtype == object:
                out[col] = (
                    out[col].astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                out[col] = pd.to_numeric(out[col], errors="coerce")

        return out

    @classmethod
    def build_schedule(cls,
                       issue_date: Union[datetime.date, str],
                       maturity_date: Union[datetime.date, str],
                       coupon_freq: str = "mensal",
                       amort_type: str = "bullet",
                       amort_start_date: Optional[Union[datetime.date, str]] = None,
                       custom_amort_schedule: Optional[Sequence[Tuple[Union[datetime.date, str], float]]] = None,
                       calendar_name: str = "B3") -> pd.DataFrame:
        """
        Constrói a agenda padronizada de eventos financeiros (Juros, Amortização e Vencimento)
        para CRI e CRA, ajustando as datas de liquidação para o próximo dia útil (following).
        """
        d_ini = cls._normalize_date(issue_date)
        d_fim = cls._normalize_date(maturity_date)
        if d_fim <= d_ini:
            raise ValueError("A data de vencimento deve ser posterior à data de emissão.")

        freq_key = coupon_freq.strip().lower()
        if freq_key not in cls._FREQ_MONTHS:
            raise ValueError(f"Frequência de cupom '{coupon_freq}' inválida. Opções: {list(cls._FREQ_MONTHS.keys())}")

        cal = get_calendar(calendar_name)
        step_months = cls._FREQ_MONTHS[freq_key]

        # Gera datas nominais de cupom entre d_ini e d_fim
        coupon_dates: List[datetime.date] = []
        if step_months == 0:
            coupon_dates = [d_fim]
        else:
            k = 1
            while True:
                dt = d_ini + relativedelta(months=step_months * k)
                if dt >= d_fim:
                    coupon_dates.append(d_fim)
                    break
                coupon_dates.append(dt)
                k += 1

        rows: List[Dict[str, object]] = []
        for dt in coupon_dates:
            rows.append({
                "data_evento": dt,
                "data_pagamento": cal.following(dt),
                "evento": "Juros",
                "tipo": "Regular",
                "taxa_percentual": 0.0,
            })

        amort_key = amort_type.strip().lower()
        if amort_key == "bullet":
            rows.append({
                "data_evento": d_fim,
                "data_pagamento": cal.following(d_fim),
                "evento": "Vencimento",
                "tipo": "Regular",
                "taxa_percentual": 100.0,
            })
        elif amort_key == "sac":
            d_start_amort = cls._normalize_date(amort_start_date) if amort_start_date else coupon_dates[0]
            amort_dates = [dt for dt in coupon_dates if dt >= d_start_amort]
            if not amort_dates:
                amort_dates = [d_fim]
            n_amort = len(amort_dates)
            for idx, dt in enumerate(amort_dates):
                is_last = idx == (n_amort - 1)
                # Percentual sobre saldo remanescente para amortização linear uniforme (SAC)
                perc_rem = 100.0 if is_last else round(100.0 / (n_amort - idx), 6)
                rows.append({
                    "data_evento": dt,
                    "data_pagamento": cal.following(dt),
                    "evento": "Vencimento" if is_last else "Amortização",
                    "tipo": "Regular",
                    "taxa_percentual": perc_rem,
                })
        elif amort_key == "custom":
            if not custom_amort_schedule:
                raise ValueError("Para amort_type='custom', informe 'custom_amort_schedule'.")
            for idx, (dt_raw, perc) in enumerate(custom_amort_schedule):
                dt = cls._normalize_date(dt_raw)
                is_last = idx == (len(custom_amort_schedule) - 1)
                rows.append({
                    "data_evento": dt,
                    "data_pagamento": cal.following(dt),
                    "evento": "Vencimento" if (is_last and dt == d_fim) else "Amortização",
                    "tipo": "Regular",
                    "taxa_percentual": float(perc),
                })
        else:
            raise ValueError(f"Tipo de amortização '{amort_type}' inválido. Use 'bullet', 'sac' ou 'custom'.")

        df = pd.DataFrame(rows)
        # Garante ordem determinística: Juros antes de Amortização/Vencimento na mesma data
        order_map = {"Juros": 0, "Amortização": 1, "Vencimento": 2}
        df["_ord"] = df["evento"].map(lambda x: order_map.get(x, 9))
        df = df.sort_values(["data_pagamento", "_ord"]).drop(columns=["_ord"]).reset_index(drop=True)
        return df

