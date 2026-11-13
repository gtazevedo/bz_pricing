import datetime
import math
import numpy as np
import pandas as pd
import pytest

from bz_pricing.data.debentures import DebenturesClient, DebenturesAPI
from bz_pricing.instruments.credit import Debenture


def test_debentures_client_offline_cache():
    client = DebenturesClient(data_dir="dados/debentures")
    assert client.data_dir == "dados/debentures"

    # Test market prices parser for 30/10/2026
    df_mkt = client.get_market_prices(datetime.date(2026, 10, 30))
    assert not df_mkt.empty
    assert "codigo" in df_mkt.columns
    assert "pu_mercado" in df_mkt.columns
    assert "taxa_indicativa" in df_mkt.columns

    valea0_mkt = df_mkt[df_mkt["codigo"] == "VALEA0"]
    assert not valea0_mkt.empty
    assert abs(valea0_mkt["pu_mercado"].iloc[0] - 1022.352694) < 1e-4
    assert abs(valea0_mkt["taxa_indicativa"].iloc[0] - 7.6229) < 1e-4


def test_debenture_valea0_pricing():
    client = DebenturesClient(data_dir="dados/debentures")
    deb = Debenture.from_ticker("VALEA0", client=client)

    assert deb.ticker == "VALEA0"
    assert deb.indexador == "IPCA"
    assert abs(deb.coupon_rate - 0.063837) < 1e-5

    base_date = datetime.date(2026, 10, 30)
    vna_oficial = 1096.791130

    # 1. PU Par oficial SND em 30/10/2026: 1099.757794
    pu_par = deb.pu_par(base_date, vna=vna_oficial)
    assert abs(pu_par - 1099.757794) < 1e-5

    # 2. Decomposição de fluxos
    taxa_ind = 0.076229
    pu_calc, df_cf = deb.present_value(base_date, taxa_ind, vna=vna_oficial, return_cashflows=True)
    assert len(df_cf) == 17
    assert set(["data_evento", "data_pagamento", "du", "evento", "fluxo_nominal", "valor_presente"]).issubset(df_cf.columns)
    assert abs(df_cf["valor_presente"].sum() - pu_calc) < 1e-4

    # 3. Medidas de risco e cotação (aderência exata ao SND / ANBIMA)
    cot = deb.cotacao(base_date, taxa_ind, vna=vna_oficial)
    dur = deb.duration(base_date, taxa_ind, vna=vna_oficial)
    assert abs(pu_calc - 1022.352694) < 1e-6
    assert abs(cot - 92.9616) < 1e-4
    assert abs(dur - 1579.24) < 1e-2

    # 4. Métricas de sensibilidade (Modified Duration, DV01 e Convexidade)
    metrics = deb.risk_metrics(base_date, taxa_ind, vna=vna_oficial)
    assert abs(metrics["modified_duration"] - 5.822975) < 1e-4
    assert abs(metrics["dv01"] - 0.595313) < 1e-4
    assert abs(metrics["convexity"] - 44.6096) < 1e-2


def test_debenture_aalm12_pricing():
    client = DebenturesClient(data_dir="dados/debentures")
    deb = Debenture.from_ticker("AALM12", client=client)

    assert deb.ticker == "AALM12"
    assert deb.indexador == "DI"

    base_date = datetime.date(2026, 10, 30)
    vna = 1000.00
    taxa_ind = 0.008877

    pu_calc, df_cf = deb.present_value(base_date, taxa_ind, vna=vna, return_cashflows=True)
    assert len(df_cf) > 0
    assert "Amortização" in df_cf["evento"].values
    assert "Juros" in df_cf["evento"].values

    dur = deb.duration(base_date, taxa_ind, vna=vna)
    assert 400.0 < dur < 700.0

