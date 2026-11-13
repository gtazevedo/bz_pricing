import datetime
import math
import pandas as pd
import pytest

from bz_pricing.core.curves import FlatCurve
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
    )
    metrics_p = cra_pct.risk_metrics(base_date, 1.05, vna=1000.0, di_curve=di_flat)
    assert metrics_p["pu"] > metrics_p["pu_par"]  # 105% < 108% -> acima do par
