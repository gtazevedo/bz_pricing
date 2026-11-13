import json
import os
import re
import threading
import urllib3
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import unquote

import pandas as pd
import requests

from .cricra import CRICRAClient


class ANBIMADataClient:
    """
    Cliente de acesso aberto ao portal ANBIMA Data (data.anbima.com.br) sem
    necessidade de credenciais pagas ou autenticação prévia.

    Reúne consultas de debêntures, CRI/CRA, Letras Financeiras, listas públicas
    de mercado e download dos manuais oficiais de precificação.
    """

    BASE_URL = "https://data.anbima.com.br"
    TRANSPARENCIA_URL = "https://data.anbima.com.br/transparencia-e-governanca"

    def __init__(self, data_dir: str = "dados/anbima_data", manuals_dir: str = "dados/manuais_anbima"):
        self.data_dir = data_dir
        self.manuals_dir = manuals_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.manuals_dir, exist_ok=True)
        self.cricra = CRICRAClient(data_dir=data_dir)

    @staticmethod
    def _run_playwright(worker: Callable[[Any], None], timeout_s: int = 120) -> None:
        """Executa uma rotina Playwright em thread isolada para compatibilidade com Jupyter/asyncio."""
        errors: List[str] = []

        def _target():
            try:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    try:
                        page = browser.new_page()
                        worker(page)
                    finally:
                        browser.close()
            except Exception as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=_target)
        thread.start()
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise TimeoutError(f"Tempo limite de {timeout_s}s excedido na consulta ao ANBIMA Data.")
        if errors:
            raise RuntimeError(f"Erro na consulta ao ANBIMA Data: {errors[0]}")

    def _capture_urls(
        self,
        urls: List[str],
        matcher: Callable[[str], Optional[str]],
        wait_ms: int = 3000,
    ) -> Dict[str, Any]:
        """Navega pelas URLs informadas e captura respostas JSON 200 conforme a função matcher."""
        captured: Dict[str, Any] = {}

        def _worker(page):
            def on_response(resp):
                if resp.status != 200:
                    return
                key = matcher(resp.url)
                if key is None:
                    return
                try:
                    captured[key] = resp.json()
                except Exception:
                    pass

            page.on("response", on_response)
            for url in urls:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(wait_ms)

        self._run_playwright(_worker)
        return captured

    def download_manuals(
        self,
        output_dir: Optional[str] = None,
        overwrite: bool = False,
    ) -> List[str]:
        """
        Baixa todos os manuais oficiais de metodologia e documentos de governança
        publicados na página aberta de Transparência e Governança do ANBIMA Data.
        """
        target_dir = output_dir or self.manuals_dir
        os.makedirs(target_dir, exist_ok=True)
        doc_urls: set = set()

        def _extract_links(obj: Any) -> None:
            if isinstance(obj, dict):
                for val in obj.values():
                    _extract_links(val)
            elif isinstance(obj, list):
                for item in obj:
                    _extract_links(item)
            elif isinstance(obj, str) and (".pdf" in obj.lower() or ".xlsx" in obj.lower()):
                match = re.findall(r"https?://[^\s\"'<>]+?\.(?:pdf|xlsx)", obj, flags=re.IGNORECASE)
                for url in match:
                    doc_urls.add(url)
                if obj.startswith("/uploads/") and (obj.lower().endswith(".pdf") or obj.lower().endswith(".xlsx")):
                    doc_urls.add(f"https://data-strapi.prd.anbima.com.br{obj}")

        def _worker(page):
            def on_response(resp):
                if "data-strapi" in resp.url and resp.status == 200:
                    try:
                        _extract_links(resp.json())
                    except Exception:
                        pass

            page.on("response", on_response)
            page.goto(self.TRANSPARENCIA_URL, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(4000)

            hrefs = page.eval_on_selector_all("a", "els => els.map(e => e.href)")
            for href in hrefs:
                if href and (".pdf" in href.lower() or ".xlsx" in href.lower()):
                    doc_urls.add(href)

        self._run_playwright(_worker)

        if not doc_urls:
            raise RuntimeError("Nenhum documento metodológico localizado em Transparência e Governança.")

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        saved_paths: List[str] = []

        for url in sorted(doc_urls):
            filename = unquote(url.split("/")[-1].split("?")[0])
            dest = os.path.join(target_dir, filename)
            if os.path.exists(dest) and not overwrite:
                saved_paths.append(dest)
                continue

            resp = requests.get(url, timeout=60, verify=False)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            saved_paths.append(dest)

        return saved_paths

    def get_debenture(self, ticker: str, force_download: bool = False) -> Dict[str, Any]:
        """
        Extrai características, preços e agenda de uma debênture no ANBIMA Data
        via web-bff aberto, com cache local em JSON.
        """
        ticker_clean = ticker.strip().upper()
        cache_file = os.path.join(self.data_dir, f"anbima_debenture_{ticker_clean}.json")
        if not force_download and os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        prefix = f"web-bff/v1/debentures/{ticker_clean}"

        def _matcher(url: str) -> Optional[str]:
            if prefix not in url:
                return None
            suffix = url.split(prefix, 1)[1].lstrip("/").split("?")[0]
            return suffix or "resumo"

        urls = [
            f"{self.BASE_URL}/debentures/{ticker_clean}/caracteristicas",
            f"{self.BASE_URL}/debentures/{ticker_clean}/precos",
            f"{self.BASE_URL}/debentures/{ticker_clean}/agenda",
        ]
        captured = self._capture_urls(urls, _matcher)
        if not captured:
            raise FileNotFoundError(f"Nenhum dado encontrado no ANBIMA Data para a debênture {ticker_clean}.")

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(captured, f, ensure_ascii=False, indent=2)
        return captured

    def get_debenture_characteristics(self, ticker: str, force_download: bool = False) -> Dict[str, Any]:
        """Retorna as características cadastrais de uma debênture no ANBIMA Data."""
        data = self.get_debenture(ticker, force_download=force_download)
        return data.get("caracteristicas", data.get("resumo", {}))

    def get_debenture_prices(self, ticker: str, force_download: bool = False) -> Dict[str, Any]:
        """Retorna os preços indicativos e histórico de PU de uma debênture no ANBIMA Data."""
        data = self.get_debenture(ticker, force_download=force_download)
        return data.get("precos", {})

    def get_debenture_cashflows(self, ticker: str, force_download: bool = False) -> Dict[str, Any]:
        """Retorna a agenda de eventos financeiros de uma debênture no ANBIMA Data."""
        data = self.get_debenture(ticker, force_download=force_download)
        return data.get("agenda", {})

    def get_cri_cra(self, ticker: str, force_download: bool = False) -> Dict[str, Any]:
        """Extrai características, preços e agenda de um CRI ou CRA no ANBIMA Data."""
        return self.cricra.get_anbima_data_asset(ticker, force_download=force_download)

    def get_public_cri_cra_table(self, force_download: bool = False) -> pd.DataFrame:
        """Retorna a tabela pública diária de taxas e PUs indicativos de CRI/CRA da ANBIMA."""
        return self.cricra.get_anbima_public_prices(force_download=force_download)

    def get_letras_financeiras(self, force_download: bool = False) -> Dict[str, Any]:
        """
        Extrai as taxas e parâmetros abertos de Letras Financeiras publicados em
        data.anbima.com.br/letras-financeiras.
        """
        cache_file = os.path.join(self.data_dir, "anbima_letras_financeiras.json")
        if not force_download and os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        def _matcher(url: str) -> Optional[str]:
            if "web-bff/v1/letras-financeiras" in url:
                return "letras_financeiras"
            return None

        captured = self._capture_urls([f"{self.BASE_URL}/letras-financeiras"], _matcher)
        payload = captured.get("letras_financeiras")
        if not payload:
            raise RuntimeError("Nenhum dado aberto de Letras Financeiras retornado pelo ANBIMA Data.")

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    def list_debentures(self, force_download: bool = False) -> pd.DataFrame:
        """
        Consulta a listagem aberta inicial de debêntures do ANBIMA Data
        (`web-bff/v1/debentures?view=caracteristicas`).
        """
        cache_file = os.path.join(self.data_dir, "anbima_debentures_lista.json")
        if not force_download and os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            def _matcher(url: str) -> Optional[str]:
                if "web-bff/v1/debentures?" in url and "view=caracteristicas" in url:
                    return "lista"
                return None

            captured = self._capture_urls([f"{self.BASE_URL}/debentures"], _matcher)
            payload = captured.get("lista")
            if not payload:
                raise RuntimeError("Não foi possível obter a listagem aberta de debêntures do ANBIMA Data.")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        content = payload.get("content", []) if isinstance(payload, dict) else []
        return pd.DataFrame(content)

