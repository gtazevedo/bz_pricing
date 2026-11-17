import datetime
import math
from unittest.mock import MagicMock, patch

import pytest

from bz_pricing.data import ANBIMADataClient
from bz_pricing.instruments import Debenture


def test_z_spread_cdi_spread():
    deb = Debenture(
        ticker="DEB-DI",
        maturity_date="2028-01-03",
        indexador="DI",
        coupon_rate=0.015,
        issue_date="2025-01-02",
    )
    zs = deb.z_spread(datetime.date(2026, 7, 1), discount_rate=0.018516)
    assert abs(zs - 0.018516) < 1e-12


def test_z_spread_ipca_nao_incentivada_exemplo_anbima():
    # Exemplo oficial do manual Z-spread ANBIMA (p. 4):
    # i = 7,3348%, P = 6,4904% -> Z-spread = 0,7929%
    deb = Debenture(
        ticker="DEB-IPCA",
        maturity_date="2035-05-15",
        indexador="IPCA",
        coupon_rate=0.06,
        issue_date="2025-05-15",
    )
    zs = deb.z_spread(
        datetime.date(2026, 7, 1),
        discount_rate=0.073348,
        risk_free_rate=0.064904,
        incentivada=False,
    )
    assert abs(zs * 100.0 - 0.7929) < 1e-4


def test_z_spread_pct_di_exemplo_anbima():
    # Exemplo oficial do manual Z-spread ANBIMA (p. 6):
    # J = 14,8196%, T = 104,1168% -> Z-spread = 0,5313%
    deb = Debenture.from_schedule(
        ticker="DEB-PCTDI",
        issue_date="2025-01-02",
        maturity_date="2028-01-03",
        indexador="DI",
        coupon_rate=0.0,
        multiplicador_di=1.02,
        is_pct_di=True,
    ) if hasattr(Debenture, "from_schedule") else None
    if deb is None:
        import pandas as pd
        df_cf = pd.DataFrame([
            {"data_evento": datetime.date(2028, 1, 3), "data_pagamento": datetime.date(2028, 1, 3), "evento": "Juros", "tipo": "Juros", "taxa_percentual": 0.0},
            {"data_evento": datetime.date(2028, 1, 3), "data_pagamento": datetime.date(2028, 1, 3), "evento": "Vencimento", "tipo": "Vencimento", "taxa_percentual": 100.0},
        ])
        deb = Debenture(
            ticker="DEB-PCTDI",
            maturity_date="2028-01-03",
            indexador="DI",
            coupon_rate=0.0,
            issue_date="2025-01-02",
            cashflows_df=df_cf,
            multiplicador_di=1.02,
            is_pct_di=True,
        )
    zs = deb.z_spread(
        datetime.date(2026, 7, 1),
        discount_rate=1.041168,
        ettj_pre_rate_or_curve=0.148196,
        di_curve=0.148196,
        pu_par_val=1050.0,
    )
    assert abs(zs * 100.0 - 0.5313) < 1e-4


def test_z_spread_ipca_incentivada_gross_up():
    import pandas as pd
    df_cf = pd.DataFrame([
        {"data_evento": datetime.date(2027, 7, 15), "data_pagamento": datetime.date(2027, 7, 15), "evento": "Juros", "tipo": "Juros", "taxa_percentual": 0.0},
        {"data_evento": datetime.date(2028, 7, 17), "data_pagamento": datetime.date(2028, 7, 17), "evento": "Juros", "tipo": "Juros", "taxa_percentual": 0.0},
        {"data_evento": datetime.date(2028, 7, 17), "data_pagamento": datetime.date(2028, 7, 17), "evento": "Vencimento", "tipo": "Vencimento", "taxa_percentual": 100.0},
    ])
    deb = Debenture(
        ticker="DEB-INC",
        maturity_date="2028-07-17",
        indexador="IPCA",
        coupon_rate=0.065,
        issue_date="2025-07-15",
        cashflows_df=df_cf,
    )
    m = deb.risk_metrics(
        datetime.date(2026, 7, 1),
        discount_rate_or_curve=0.075,
        vna=1100.0,
        risk_free_rate=0.065,
        incentivada=True,
    )
    assert m["taxa_gross_up"] > 0.075
    assert m["z_spread"] > (1.075 / 1.065 - 1.0)


def test_anbima_data_client_get_ettj_parses_csv(tmp_path):
    cli = ANBIMADataClient(data_dir=str(tmp_path), manuals_dir=str(tmp_path / "manuais"))
    csv_body = (
        "12/11/2026;Beta 1;Beta 2\n"
        "PREFIXADOS;0,14;0,01\n\n"
        "Vertices;ETTJ IPCA;ETTJ PREF;Inflação Implícita\n"
        "126;4,5000;14,8000;9,8565\n"
        "252;5,1000;14,5000;8,9439\n"
    )
    fake_resp = MagicMock()
    fake_resp.content = csv_body.encode("latin-1")
    fake_resp.raise_for_status.return_value = None

    with patch("bz_pricing.data.anbima_data.requests.get", return_value=fake_resp):
        df = cli.get_ettj(force_download=True)

    assert len(df) == 2
    assert list(df["du"]) == [126, 252]
    assert abs(df.loc[0, "ettj_pref"] - 0.148) < 1e-9
    assert abs(df.loc[1, "ettj_ipca"] - 0.051) < 1e-9


def test_price_credit_portfolio_cython_speed():
    import time
    from bz_pricing.instruments import CotaFIDC, LetraFinanceira, price_credit_portfolio

    fidc = CotaFIDC.from_schedule(
        ticker="FIDC-1",
        issue_date="2026-01-02",
        maturity_date="2028-01-03",
        indexador="IPCA",
        coupon_rate=0.07,
        coupon_freq="semestral",
    )
    lf = LetraFinanceira.from_bullet(
        ticker="LF-1",
        issue_date="2026-01-02",
        maturity_date="2028-01-03",
        indexador="DI",
        coupon_rate=0.005,
        nominal_value=50000.0,
    )
    positions = [
        {"instrument": fidc, "discount_rate": 0.075, "quantity": 100, "vna": 1050.0, "risk_free_rate": 0.065},
        {"instrument": lf, "discount_rate": 0.006, "quantity": 10, "vna": 50000.0, "pu_par_val": 51000.0, "di_curve": 0.135},
    ] * 100  # 200 ativos na carteira

    t0 = time.perf_counter()
    df = price_credit_portfolio(positions, base_date=datetime.date(2026, 7, 1))
    elapsed = time.perf_counter() - t0

    assert len(df) == 200
    assert df["pu"].notna().all()
    assert df["z_spread"].notna().all()
    assert elapsed < 2.0


def test_credit_ytm_and_tir_all_indexers():
    from bz_pricing.instruments import CotaFIDC, LetraFinanceira, CRI

    base_dt = datetime.date(2026, 7, 1)

    # 1. Prefixado (YTM == TIR)
    deb_pre = CRI.from_schedule(
        ticker="CRI-PRE",
        issue_date="2026-01-02",
        maturity_date="2028-01-03",
        indexador="PRE",
        coupon_rate=0.12,
        coupon_freq="semestral",
    )
    pu_pre = deb_pre.present_value(base_dt, 0.135)
    assert abs(deb_pre.ytm(base_dt, pu_pre) - 0.135) < 1e-6
    assert abs(deb_pre.tir(base_dt, pu_pre) - 0.135) < 1e-6

    # 2. IPCA (YTM == TIR real)
    fidc_ipca = CotaFIDC.from_schedule(
        ticker="FIDC-IPCA",
        issue_date="2026-01-02",
        maturity_date="2028-01-03",
        indexador="IPCA",
        coupon_rate=0.07,
        coupon_freq="semestral",
    )
    pu_ipca = fidc_ipca.present_value(base_dt, 0.082, vna=1050.0)
    assert abs(fidc_ipca.ytm(base_dt, pu_ipca, vna=1050.0) - 0.082) < 1e-6
    assert abs(fidc_ipca.tir(base_dt, pu_ipca, vna=1050.0) - 0.082) < 1e-6

    # 3. DI + Spread (YTM = spread, TIR = DI + spread efetivo)
    lf_di = LetraFinanceira.from_bullet(
        ticker="LF-DI",
        issue_date="2026-01-02",
        maturity_date="2028-01-03",
        indexador="DI",
        coupon_rate=0.005,
        nominal_value=50000.0,
    )
    pu_di = lf_di.present_value(base_dt, 0.012, vna=50000.0, pu_par_val=51000.0, di_curve=0.135)
    ytm_di = lf_di.ytm(base_dt, pu_di, vna=50000.0, pu_par_val=51000.0, di_curve=0.135)
    tir_di = lf_di.tir(base_dt, pu_di, vna=50000.0, pu_par_val=51000.0, di_curve=0.135)
    assert abs(ytm_di - 0.012) < 1e-6
    assert abs(tir_di - ((1.135 * 1.012) - 1.0)) < 1e-5

    # 4. % do DI (YTM = multiplicador % do DI, TIR = taxa efetiva anual)
    deb_pct = CotaFIDC.from_schedule(
        ticker="FIDC-PCT",
        issue_date="2026-01-02",
        maturity_date="2028-01-03",
        indexador="% DI",
        multiplicador_di=1.05,
        coupon_freq="semestral",
    )
    pu_pct = deb_pct.present_value(base_dt, 1.12, vna=1000.0, pu_par_val=1020.0, di_curve=0.14)
    ytm_pct = deb_pct.ytm(base_dt, pu_pct, vna=1000.0, pu_par_val=1020.0, di_curve=0.14)
    tir_pct = deb_pct.tir(base_dt, pu_pct, vna=1000.0, pu_par_val=1020.0, di_curve=0.14)
    assert abs(ytm_pct - 1.12) < 1e-6
    assert 0.14 < tir_pct < 0.18


def test_credit_custom_cashflows_and_debenture_schedule():
    from bz_pricing.instruments.credit import Debenture

    base_dt = datetime.date(2026, 10, 30)

    # 1. Debenture construída diretamente com fluxos customizados do usuário
    custom_flows = [
        (datetime.date(2027, 4, 30), 60.0, "Juros"),
        (datetime.date(2027, 10, 29), 560.0, "Amortização + Juros"),
        (datetime.date(2028, 4, 28), 530.0, "Vencimento"),
    ]
    deb_cf = Debenture.from_cashflows("DEB-CUSTOM", custom_flows, indexador="PRE")
    pu = deb_cf.present_value(base_dt, 0.13)
    assert pu > 0.0
    assert abs(deb_cf.ytm(base_dt, pu) - 0.13) < 1e-6
    assert abs(deb_cf.tir(base_dt, pu) - 0.13) < 1e-6
    assert deb_cf.duration(base_dt, 0.13) > 0.0
    assert deb_cf.dv01(base_dt, 0.13) > 0.0

    # 2. Debenture.from_schedule (sem necessidade de CSVs da ANBIMA)
    deb_sch = Debenture.from_schedule(
        ticker="DEB-SCH",
        issue_date="2026-01-02",
        maturity_date="2028-01-03",
        indexador="PRE",
        coupon_rate=0.12,
        coupon_freq="semestral",
        amort_type="bullet",
    )
    pu_sch = deb_sch.present_value(base_dt, 0.125)
    assert pu_sch > 0.0
    assert abs(deb_sch.ytm(base_dt, pu_sch) - 0.125) < 1e-6


def test_hazard_rate_curve_and_risky_credit_pricing():
    from bz_pricing.instruments import Debenture, HazardRateCurve

    base_dt = datetime.date(2026, 10, 30)
    hz = HazardRateCurve.from_credit_spreads([252, 504], [0.015, 0.020], recovery_rate=0.40)
    assert abs(hz.hazard_rate(252) - (math.log(1.015) / 0.60)) < 1e-12
    assert abs(hz.survival_prob(0) - 1.0) < 1e-12
    assert 0.90 < hz.survival_prob(252) < 1.0
    assert hz.default_prob(504) > hz.default_prob(252)

    deb = Debenture.from_cashflows(
        "DEB-HZ",
        [(datetime.date(2027, 4, 30), 60.0, "Juros"), (datetime.date(2027, 10, 29), 1060.0, "Vencimento")],
        indexador="PRE",
    )
    pu_rf_direct = deb.present_value(base_dt, 0.12)
    res = deb.risky_present_value(base_dt, 0.12, hazard_curve=hz)
    assert abs(res["pv_risk_free"] - pu_rf_direct) < 1e-4
    assert 0.0 < res["pv_risky"] < res["pv_risk_free"]
    assert abs(res["cva"] - (res["pv_risk_free"] - res["pv_risky"])) < 1e-8

    # Inversão de hazard rate implícito a partir de pv_risky com lambda constante
    res_flat = deb.risky_present_value(base_dt, 0.12, hazard_rate=0.03, recovery_rate=0.40)
    lam_imp = deb.implied_hazard_rate(base_dt, res_flat["pv_risky"], 0.12, recovery_rate=0.40)
    assert abs(lam_imp - 0.03) < 1e-6

    # Acima do valor livre de risco retorna NaN; no valor exato retorna 0.0
    assert abs(deb.implied_hazard_rate(base_dt, res["pv_risk_free"], 0.12, recovery_rate=0.40) - 0.0) < 1e-10
    assert math.isnan(deb.implied_hazard_rate(base_dt, res["pv_risk_free"] + 25.0, 0.12, recovery_rate=0.40))

    # Amortização reduz saldo exposto à recuperação vs bullet
    deb_sac = Debenture.from_schedule("DEB-SAC", "2026-10-30", "2029-10-30", indexador="PRE", coupon_rate=0.10, coupon_freq="anual", amort_type="sac")
    deb_bul = Debenture.from_schedule("DEB-BUL", "2026-10-30", "2029-10-30", indexador="PRE", coupon_rate=0.10, coupon_freq="anual", amort_type="bullet")
    res_sac = deb_sac.risky_present_value(base_dt, 0.10, hazard_rate=0.05, recovery_rate=0.40)
    res_bul = deb_bul.risky_present_value(base_dt, 0.10, hazard_rate=0.05, recovery_rate=0.40)
    assert abs(res_sac["pv_risk_free"] - deb_sac.present_value(base_dt, 0.10)) < 1e-4
    assert abs(res_bul["pv_risk_free"] - deb_bul.present_value(base_dt, 0.10)) < 1e-4
    assert 0.0 < res_sac["cva"] < res_bul["cva"]

    cdv01 = deb.credit_dv01(base_dt, 0.12, hazard_curve=hz)
    assert cdv01 > 0.0


def test_credit_offline_di_and_resolve_vna_consistency():
    from bz_pricing.instruments import Debenture

    base_dt = datetime.date(2026, 10, 30)
    daily_di = {
        "2026-10-27": 0.00055,
        "2026-10-28": 0.00055,
        "2026-10-29": 0.00055,
        "2026-10-30": 0.00055,
    }
    deb_di = Debenture.from_schedule(
        "DEB-DI-OFF",
        issue_date="2026-10-27",
        maturity_date="2028-10-30",
        indexador="DI",
        coupon_rate=0.015,
    )
    pu_par = deb_di.pu_par(base_dt, di_daily_rates=daily_di, allow_download=False)
    assert abs(pu_par - 1000.0 * round((1.00055 ** 3) * round(1.015 ** (3 / 252.0), 9), 9)) < 0.05



