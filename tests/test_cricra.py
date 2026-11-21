import datetime
import math
import pandas as pd
import pytest

from bz_pricing.core.curves import FlatCurve
from bz_pricing.core.vna_engine import VNAEngine
from bz_pricing.curves.manager import CurveManager
from bz_pricing.data.cricra import CRICRAClient
from bz_pricing.instruments.credit import CRI, CRA, CertificadoRecebiveis


def test_cricra_schedule_builder_sac_and_bullet():
    df_sac = CRICRAClient.build_schedule(
        issue_date="2026-01-15",
        maturity_date="2026-07-15",
        coupon_freq="mensal",
        amort_type="sac",
    )
    assert not df_sac.empty
    juros = df_sac[df_sac["evento"] == "Juros"]
    amorts = df_sac[df_sac["evento"].isin(["Amortização", "Vencimento"])]
    assert len(juros) == 6
    assert len(amorts) == 6
    assert abs(amorts["taxa_percentual"].iloc[-1] - 100.0) < 1e-6

    df_bullet = CRICRAClient.build_schedule(
        issue_date="2026-01-15",
        maturity_date="2028-01-15",
        coupon_freq="semestral",
        amort_type="bullet",
    )
    assert len(df_bullet[df_bullet["evento"] == "Juros"]) == 4
    assert len(df_bullet[df_bullet["evento"] == "Vencimento"]) == 1


def test_cri_ipca_act360_sac_pricing():
    cri = CRI.from_schedule(
        ticker="26A0001234",
        issue_date="2026-01-15",
        maturity_date="2028-01-15",
        indexador="IPCA",
        coupon_rate=0.075,
        coupon_freq="mensal",
        amort_type="sac",
        coupon_day_count="ACT/360",
        securitizadora="VIRGO",
    )
    assert isinstance(cri, CertificadoRecebiveis)
    assert cri.lastro == "Imobiliário"
    assert cri.coupon_day_count == "ACT/360"

    base_date = datetime.date(2026, 10, 30)
    vna = 1042.50
    pu_par = cri.pu_par(base_date, vna=vna)
    assert pu_par > vna

    # Desconto a mercado sempre em base BUS/252 conforme manual ANBIMA
    metrics = cri.risk_metrics(base_date, discount_rate_or_curve=0.082, vna=vna)
    assert 900.0 < metrics["pu"] < 1100.0
    assert metrics["duration_du"] > 0.0
    assert metrics["dv01"] > 0.0

    pu, df_cf = cri.present_value(base_date, 0.082, vna=vna, return_cashflows=True)
    assert abs(df_cf["valor_presente"].sum() - pu) < 1e-4
    # No SAC com 24 parcelas mensais de 1000/24 = 41.6667, soma das amortizações futuras deve igualar VNA
    total_amort = df_cf[df_cf["evento"].isin(["Amortização", "Vencimento"])]["fluxo_nominal"].sum()
    assert abs(total_amort - vna) < 0.05


def test_cra_di_spread_and_pct_di_pricing():
    base_date = datetime.date(2026, 10, 30)
    di_flat = FlatCurve(rate=0.1490)
    di_dates = pd.bdate_range("2026-04-15", "2026-10-30")
    r_day = (1.0 + 0.1490) ** (1.0 / 252.0) - 1.0
    di_daily_rates = pd.Series(r_day, index=di_dates)

    cra_spread = CRA.from_schedule(
        ticker="CRA026001",
        issue_date="2026-04-15",
        maturity_date="2028-04-15",
        indexador="DI",
        coupon_rate=0.015,
        coupon_freq="semestral",
        amort_type="bullet",
        defasagem_di=1,
        securitizadora="ECOAGRO",
        di_daily_rates=di_daily_rates,
        allow_download=False,
    )
    assert cra_spread.lastro == "Agronegócio"
    assert cra_spread.defasagem_di == 1

    pu_par_s = cra_spread.pu_par(base_date, vna=1000.0)
    assert pu_par_s > 1000.0

    metrics_s = cra_spread.risk_metrics(base_date, 0.012, vna=1000.0, di_curve=di_flat)
    assert metrics_s["pu"] > metrics_s["pu_par"]  # spread de desconto (1.2%) < spread contratual (1.5%)
    assert metrics_s["duration_du"] > 0.0
    assert metrics_s["dv01"] > 0.0

    cra_pct = CRA.from_schedule(
        ticker="CRA026002",
        issue_date="2026-04-15",
        maturity_date="2028-04-15",
        indexador="DI",
        coupon_rate=0.0,
        multiplicador_di=1.08,
        coupon_freq="semestral",
        amort_type="bullet",
        di_daily_rates=di_daily_rates,
        allow_download=False,
    )
    metrics_p = cra_pct.risk_metrics(base_date, 1.05, vna=1000.0, di_curve=di_flat)
    assert metrics_p["pu"] > metrics_p["pu_par"]  # 105% < 108% -> acima do par


def test_cri_auto_vna_integration():
    engine = VNAEngine(base_vna_date=datetime.date(2026, 8, 15), base_vna_value=1000.0)
    engine.ipca_history = pd.Series(
        [0.0040, 0.0050, 0.0045],
        index=pd.to_datetime(["2026-08-01", "2026-09-01", "2026-10-01"])
    )
    cri = CRI.from_schedule(
        ticker="CRI_AUTO_VNA",
        issue_date="2026-08-15",
        maturity_date="2028-08-15",
        indexador="IPCA",
        coupon_rate=0.07,
        coupon_freq="mensal",
        amort_type="bullet",
        anniversary_day=15,
        vna_engine=engine,
    )
    base_date = datetime.date(2026, 10, 30)
    # Sem passar vna explicitamente, deve resolver via VNAEngine
    vna_calc = cri.resolve_vna(base_date)
    assert vna_calc > 1009.0
    metrics = cri.risk_metrics(base_date, 0.075)
    assert metrics["pu_par"] > vna_calc
    assert metrics["pu"] > 0.0


@pytest.mark.online
def test_cricra_anbima_public_prices():
    client = CRICRAClient(data_dir="dados/cricra")
    df_pub = client.get_anbima_public_prices()
    assert not df_pub.empty
    assert {"codigo", "vencimento", "indice_correcao", "pu_mercado", "taxa_indicativa"}.issubset(df_pub.columns)
    assert len(df_pub) > 500


@pytest.mark.online
@pytest.mark.market_data("dados/cricra/anbima_data_CRA026002BD.json")
def test_cra026002bd_real_anbima_data():
    client = CRICRAClient(data_dir="dados/cricra")
    cra = CRA.from_ticker("CRA026002BD", client=client)
    assert cra.ticker == "CRA026002BD"
    assert cra.indexador == "DI"
    assert abs(cra.multiplicador_di - 1.015) < 1e-6

    prices = client.get_asset_prices("CRA026002BD")
    pu_ind = prices["pu_indicativo"]
    ref_dt = datetime.date.fromisoformat(pu_ind["data_referencia"])
    taxa_ind = float(pu_ind["taxa_indicativa"]) / 100.0
    pu_oficial = float(pu_ind["pu_indicativo"])
    dur_oficial = float(pu_ind["duration"])

    curva_pre = CurveManager(allow_download=True).get_curve("PRE", ref_dt)
    metrics = cra.risk_metrics(ref_dt, taxa_ind, di_curve=curva_pre)
    assert abs(metrics["pu"] - pu_oficial) < 0.05
    assert abs(metrics["duration_du"] - dur_oficial) < 1.0


@pytest.mark.online
@pytest.mark.market_data(
    "dados/cricra/anbima_data_CRA019003JY.json",
    "dados/cricra/anbima_data_22D1289009.json",
    "dados/cricra/anbima_data_CRA024004H7.json",
)
@pytest.mark.parametrize("ticker", ["CRA019003JY", "22D1289009", "CRA024004H7"])
def test_cricra_amostra_diversificada_anbima_data(ticker):
    client = CRICRAClient(data_dir="dados/cricra")
    asset = CertificadoRecebiveis.from_ticker(ticker, client=client)
    prices = client.get_asset_prices(ticker)
    pu_ind = prices["pu_indicativo"]
    pu_hist = prices.get("pu_historico") or {}

    ref_dt = datetime.date.fromisoformat(pu_ind["data_referencia"])
    taxa_ind = float(pu_ind["taxa_indicativa"]) / 100.0
    pu_oficial = float(pu_ind["pu_indicativo"])
    dur_oficial = float(pu_ind["duration"])
    vna_of = float(pu_hist["vna"]) if pu_hist.get("vna") else None

    curva_pre = CurveManager(allow_download=True).get_curve("PRE", ref_dt) if asset._is_di_linked() else None
    metrics = asset.risk_metrics(ref_dt, taxa_ind, vna=vna_of, di_curve=curva_pre)
    assert abs(metrics["pu"] - pu_oficial) < 0.10
    assert abs(metrics["duration_du"] - dur_oficial) < 0.5


def test_cricra_client_cache_and_market_file_offline(tmp_path):
    import json

    client = CRICRAClient(data_dir=str(tmp_path))
    cache_path = tmp_path / "anbima_data_CRAOFFLINE1.json"
    payload = {
        "resumo": {"codigo_b3": "CRAOFFLINE1", "tipo": "CRA"},
        "pu_indicativo": {"pu_indicativo": "1005.12", "taxa_indicativa": "101.5"},
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = client.get_anbima_data_asset("CRAOFFLINE1")
    assert loaded["resumo"]["codigo_b3"] == "CRAOFFLINE1"
    assert float(loaded["pu_indicativo"]["pu_indicativo"]) == 1005.12

    mkt_csv = tmp_path / "cricra_mkt.csv"
    mkt_csv.write_text(
        "Código,Ativo,Securitizadora,Vencimento,Índice/ Correção,Taxa Indicativa,PU,Duration\n"
        "CRAOFFLINE1,CRA,ECOAGRO,15/10/2030,IPCA + 7%,\"7,1520\",\"1.005,12\",\"612,40\"\n",
        encoding="utf-8",
    )
    df_mkt = client.load_market_file(str(mkt_csv))
    assert len(df_mkt) == 1
    assert df_mkt["codigo"].iloc[0] == "CRAOFFLINE1"
    assert df_mkt["vencimento"].iloc[0] == datetime.date(2030, 10, 15)
    assert abs(float(df_mkt["taxa_indicativa"].iloc[0]) - 7.1520) < 1e-6
    assert abs(float(df_mkt["pu_mercado"].iloc[0]) - 1005.12) < 1e-6
    assert abs(float(df_mkt["duration"].iloc[0]) - 612.40) < 1e-6

