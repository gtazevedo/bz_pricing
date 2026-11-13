import datetime
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

