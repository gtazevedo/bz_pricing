import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bz_pricing.data import ANBIMADataClient


def test_anbima_data_client_uses_debenture_cache(tmp_path):
    client = ANBIMADataClient(data_dir=str(tmp_path), manuals_dir=str(tmp_path / "manuais"))
    payload = {
        "caracteristicas": {"codigo_b3": "VALEA0", "remuneracao": "IPCA + 6,00%"},
        "precos": {"pu_indicativo": 1050.12},
        "agenda": {"content": [{"evento": "Juros", "data_pagamento": "2026-08-15"}]},
    }
    cache_path = tmp_path / "anbima_debenture_VALEA0.json"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    assert client.get_debenture_characteristics("valea0")["codigo_b3"] == "VALEA0"
    assert client.get_debenture_prices("VALEA0")["pu_indicativo"] == 1050.12
    assert len(client.get_debenture_cashflows("VALEA0")["content"]) == 1


def test_anbima_data_client_uses_lf_and_list_cache(tmp_path):
    client = ANBIMADataClient(data_dir=str(tmp_path), manuals_dir=str(tmp_path / "manuais"))

    lf_cache = tmp_path / "anbima_letras_financeiras.json"
    lf_cache.write_text(json.dumps({"vertices": [{"du": 252, "taxa": 12.5}]}), encoding="utf-8")
    assert client.get_letras_financeiras()["vertices"][0]["du"] == 252

    list_cache = tmp_path / "anbima_debentures_lista.json"
    list_cache.write_text(json.dumps({"content": [{"codigo_b3": "VALEA0"}, {"codigo_b3": "ACRC21"}]}), encoding="utf-8")
    df = client.list_debentures()
    assert isinstance(df, pd.DataFrame)
    assert list(df["codigo_b3"]) == ["VALEA0", "ACRC21"]


def test_anbima_data_client_delegates_cricra(tmp_path):
    client = ANBIMADataClient(data_dir=str(tmp_path), manuals_dir=str(tmp_path / "manuais"))
    expected = {"caracteristicas": {"codigo_b3": "CRA026002BD"}}
    with patch.object(client.cricra, "get_anbima_data_asset", return_value=expected) as mock_get:
        res = client.get_cri_cra("CRA026002BD")
        mock_get.assert_called_once_with("CRA026002BD", force_download=False)
        assert res == expected


def test_anbima_data_client_download_manuals_writes_files(tmp_path):
    client = ANBIMADataClient(data_dir=str(tmp_path), manuals_dir=str(tmp_path / "manuais"))

    def fake_run_playwright(worker, timeout_s=120):
        page = MagicMock()
        page.eval_on_selector_all.return_value = [
            "https://data-strapi.prd.anbima.com.br/uploads/Metodologia_Teste_123.pdf",
            "https://data-strapi.prd.anbima.com.br/uploads/Retificacoes_Teste_456.xlsx",
        ]
        worker(page)

    fake_resp = MagicMock()
    fake_resp.content = b"%PDF-1.4 mock"
    fake_resp.raise_for_status.return_value = None

    with patch.object(client, "_run_playwright", side_effect=fake_run_playwright), \
         patch("bz_pricing.data.anbima_data.requests.get", return_value=fake_resp) as mock_req:
        saved = client.download_manuals(overwrite=True)

    assert len(saved) == 2
    assert mock_req.call_count == 2
    for p in saved:
        assert (tmp_path / "manuais" / p.split("\\")[-1].split("/")[-1]).exists()

