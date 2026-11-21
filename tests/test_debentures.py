import datetime
import math
import numpy as np
import pandas as pd
import pytest

from bz_pricing.curves.manager import CurveManager
from bz_pricing.data.debentures import DebenturesClient, DebenturesAPI
from bz_pricing.instruments.credit import Debenture


@pytest.mark.market_data("dados/debentures/db261030.txt")
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


@pytest.mark.market_data(
    "dados/debentures/carac_VALEA0.csv",
    "dados/debentures/agenda_VALEA0.csv",
)
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


@pytest.mark.market_data(
    "dados/debentures/carac_AEGPA0.csv",
    "dados/debentures/agenda_AEGPA0.csv",
)
def test_debenture_aegpa0_prefixado_pricing():
    client = DebenturesClient(data_dir="dados/debentures")
    deb = Debenture.from_ticker("AEGPA0", client=client)

    assert deb.ticker == "AEGPA0"
    assert deb.indexador in ["PRÉ", "PRE"]
    assert abs(deb.coupon_rate - 0.16762) < 1e-6

    base_date = datetime.date(2026, 10, 30)
    vna = 1000.00
    taxa_ind = 0.179668

    metrics = deb.risk_metrics(base_date, taxa_ind, vna=vna)
    assert abs(metrics["pu_par"] - 1074.600743) < 1e-5
    assert abs(metrics["pu"] - 1052.691191) < 1e-4
    assert abs(metrics["cotacao"] - 97.9611) < 1e-4
    assert abs(metrics["duration_du"] - 504.78) < 1e-2


@pytest.mark.online
@pytest.mark.market_data(
    "dados/debentures/carac_AALM12.csv",
    "dados/debentures/agenda_AALM12.csv",
    "dados/b3/BVBG_SPRD_20261030.csv",
)
def test_debenture_aalm12_di_spread_pricing():
    client = DebenturesClient(data_dir="dados/debentures")
    deb = Debenture.from_ticker("AALM12", client=client)

    assert deb.ticker == "AALM12"
    assert deb.indexador == "DI"
    assert abs(deb.coupon_rate - 0.016) < 1e-6
    assert abs(deb.multiplicador_di - 1.0) < 1e-6

    base_date = datetime.date(2026, 10, 30)
    curva_pre = CurveManager(allow_download=True).get_curve("PRE", base_date)
    vna = 1000.00
    taxa_ind = 0.008877

    # PU Par via SGS 12 e marcação a mercado via curva PRE
    pu_par = deb.pu_par(base_date, vna=vna)
    assert abs(pu_par - 1010.903086) < 1e-4

    pu_calc, df_cf = deb.present_value(
        base_date, taxa_ind, vna=vna, di_curve=curva_pre, return_cashflows=True
    )
    assert len(df_cf) > 0
    assert "Amortização" in df_cf["evento"].values
    assert "Juros" in df_cf["evento"].values

    metrics = deb.risk_metrics(base_date, taxa_ind, vna=vna, di_curve=curva_pre)
    assert abs(metrics["pu"] - 1025.332013) < 0.01
    assert abs(metrics["cotacao"] - 101.4273) < 0.01
    assert abs(metrics["duration_du"] - 508.56) < 0.1
    assert metrics["dv01"] > 0.0


@pytest.mark.online
@pytest.mark.market_data(
    "dados/debentures/carac_KLBNA2.csv",
    "dados/debentures/agenda_KLBNA2.csv",
    "dados/b3/BVBG_SPRD_20261030.csv",
)
def test_debenture_klbna2_pct_di_pricing():
    client = DebenturesClient(data_dir="dados/debentures")
    deb = Debenture.from_ticker("KLBNA2", client=client)

    assert deb.ticker == "KLBNA2"
    assert deb.indexador == "DI"
    assert abs(deb.multiplicador_di - 1.1465) < 1e-6
    assert deb.amortizacao_sobre_vne is True

    base_date = datetime.date(2026, 10, 30)
    curva_pre = CurveManager(allow_download=True).get_curve("PRE", base_date)
    vna = 10000.00
    taxa_ind = 1.03285  # 103.285% do DI

    pu_par = deb.pu_par(base_date, vna=vna)
    assert abs(pu_par - 10164.3278) < 1e-3

    metrics = deb.risk_metrics(base_date, taxa_ind, vna=vna, di_curve=curva_pre)
    assert abs(metrics["pu"] - 10360.376432) < 0.05
    assert abs(metrics["cotacao"] - 101.9288) < 0.01
    assert abs(metrics["duration_du"] - 309.77) < 0.1
    assert metrics["dv01"] > 0.0


@pytest.mark.online
@pytest.mark.market_data(
    "dados/debentures/db261030.txt",
    "dados/b3/BVBG_SPRD_20261030.csv",
)
@pytest.mark.parametrize("ticker", ["ACRC21", "ADPA21", "RADL14", "AEGPA8", "BTEL13"])
def test_debentures_amostra_diversificada(ticker):
    client = DebenturesClient(data_dir="dados/debentures")
    base_date = datetime.date(2026, 10, 30)
    df_mkt = client.get_market_prices(base_date)
    row = df_mkt[df_mkt["codigo"] == ticker].iloc[0]

    deb = Debenture.from_ticker(ticker, client=client)
    hist = client.get_pu_history(ticker)
    hist_dt = hist[hist["data"] == base_date]
    vna_val = float(hist_dt["vna"].iloc[0]) if (not hist_dt.empty and pd.notna(hist_dt["vna"].iloc[0])) else None

    curva_pre = CurveManager(allow_download=True).get_curve("PRE", base_date) if deb._is_di_linked() else None
    metrics = deb.risk_metrics(
        base_date,
        float(row["taxa_indicativa"]) / 100.0,
        vna=vna_val,
        di_curve=curva_pre,
    )
    assert abs(metrics["pu"] - float(row["pu_mercado"])) < 0.02
    assert abs(metrics["duration_du"] - float(row["duration"])) < 0.1


def test_debentures_client_parser_and_pricing_offline(tmp_path):
    # 1. Simula cache db261030.txt da ANBIMA em diretório isolado
    db_path = tmp_path / "db261030.txt"
    db_content = (
        "ANBIMA - Associação Brasileira das Entidades dos Mercados Financeiro e de Capitais\n"
        "\n"
        "Código@Nome@Repac./  Venc.@Índice/ Correção@Taxa de Compra@Taxa de Venda@Taxa Indicativa@"
        "Desvio Padrão@Intervalo Indicativo Minimo@Intervalo Indicativo Máximo@PU@% PU Par / % VNE@"
        "Duration@% Reune@Referência NTN-B\n"
        "TSTB01@EMISSORA TESTE S.A. (*)@15/10/2029@IPCA + 6,5%@6,9500@6,8000@6,8500@"
        "0,0200@6,8300@6,8700@1042,150000@99,2500@580,50@--@B29\n"
    )
    db_path.write_text(db_content, encoding="latin1")

    # 2. Simula arquivos cacheados do SND (carac, agenda e pueventos) para TSTB01
    pd.DataFrame(
        [
            {
                "Codigo do Ativo": "TSTB01",
                "Empresa": "EMISSORA TESTE S.A.",
                "Data de Emissao": "15/10/2025",
                "Data de Vencimento": "15/10/2029",
                "Data de Saida / Novo Vencimento": "15/10/2029",
                "indice": "IPCA",
                "Valor Nominal na Emissao": "1000",
                "Percentual Multiplicador/Rentabilidade": "0",
                "Tipo de Amortizacao": "",
                "Juros Criterio Novo - Taxa": "6,5",
            }
        ]
    ).to_csv(tmp_path / "carac_TSTB01.csv", index=False, encoding="utf-8")

    pd.DataFrame(
        [
            {
                "data_evento": "2027-04-15",
                "data_pagamento": "2027-04-15",
                "emissor": "EMISSORA TESTE S.A.",
                "ativo": "TSTB01",
                "evento": "Juros",
                "tipo": "IPCAIPCA",
                "taxa_percentual": 6.5,
                "liquidado": "",
            },
            {
                "data_evento": "2027-10-15",
                "data_pagamento": "2027-10-15",
                "emissor": "EMISSORA TESTE S.A.",
                "ativo": "TSTB01",
                "evento": "Juros",
                "tipo": "IPCAIPCA",
                "taxa_percentual": 6.5,
                "liquidado": "",
            },
            {
                "data_evento": "2028-04-15",
                "data_pagamento": "2028-04-17",
                "emissor": "EMISSORA TESTE S.A.",
                "ativo": "TSTB01",
                "evento": "Juros",
                "tipo": "IPCAIPCA",
                "taxa_percentual": 6.5,
                "liquidado": "",
            },
            {
                "data_evento": "2028-10-15",
                "data_pagamento": "2028-10-16",
                "emissor": "EMISSORA TESTE S.A.",
                "ativo": "TSTB01",
                "evento": "Juros",
                "tipo": "IPCAIPCA",
                "taxa_percentual": 6.5,
                "liquidado": "",
            },
            {
                "data_evento": "2029-04-15",
                "data_pagamento": "2029-04-16",
                "emissor": "EMISSORA TESTE S.A.",
                "ativo": "TSTB01",
                "evento": "Juros",
                "tipo": "IPCAIPCA",
                "taxa_percentual": 6.5,
                "liquidado": "",
            },
            {
                "data_evento": "2029-10-15",
                "data_pagamento": "2029-10-15",
                "emissor": "EMISSORA TESTE S.A.",
                "ativo": "TSTB01",
                "evento": "Juros",
                "tipo": "IPCAIPCA",
                "taxa_percentual": 6.5,
                "liquidado": "",
            },
            {
                "data_evento": "2029-10-15",
                "data_pagamento": "2029-10-15",
                "emissor": "EMISSORA TESTE S.A.",
                "ativo": "TSTB01",
                "evento": "Vencimento",
                "tipo": "--",
                "taxa_percentual": 0.0,
                "liquidado": "",
            },
        ]
    ).to_csv(tmp_path / "agenda_TSTB01.csv", index=False, encoding="utf-8")

    pd.DataFrame(
        [
            {
                "data": "2026-10-15",
                "ativo": "TSTB01",
                "tipo_evento": "Juros",
                "pu_evento": 32.5,
                " Situação": "Registrado",
                "liquidacao": "LIQUIDADO",
            }
        ]
    ).to_csv(tmp_path / "pueventos_TSTB01.csv", index=False, encoding="utf-8")

    client = DebenturesClient(data_dir=str(tmp_path))
    base_date = datetime.date(2026, 10, 30)
    df_mkt = client.get_market_prices(base_date)
    assert len(df_mkt) == 1
    row = df_mkt.iloc[0]
    assert row["codigo"] == "TSTB01"
    assert abs(float(row["taxa_indicativa"]) - 6.85) < 1e-6
    assert abs(float(row["pu_mercado"]) - 1042.15) < 1e-6

    deb = Debenture.from_ticker("TSTB01", client=client)
    assert deb.indexador == "IPCA"
    assert abs(deb.coupon_rate - 0.065) < 1e-8

    vna = 1050.0
    taxa_ind = float(row["taxa_indicativa"]) / 100.0
    pu_calc, df_cf = deb.present_value(base_date, taxa_ind, vna=vna, return_cashflows=True)
    dur_calc = deb.duration(base_date, taxa_ind, vna=vna)
    metrics = deb.risk_metrics(base_date, taxa_ind, vna=vna)

    assert abs(df_cf["valor_presente"].sum() - pu_calc) < 1e-4
    assert pu_calc > 1000.0
    assert dur_calc > 400.0
    assert abs(metrics["pu"] - pu_calc) < 1e-6
    assert metrics["dv01"] > 0.0 and metrics["convexity"] > 0.0

