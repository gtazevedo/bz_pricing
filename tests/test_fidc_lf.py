import datetime
import json
import math

import pytest

from bz_pricing.data import ANBIMADataClient
from bz_pricing.instruments import CotaFIDC, FIDC, LetraFinanceira


def test_cota_fidc_di_spread_pricing():
    fidc = CotaFIDC.from_schedule(
        ticker="FIDC-SEN1",
        issue_date="2026-01-02",
        maturity_date="2027-01-04",
        indexador="DI",
        coupon_rate=0.02,
        coupon_freq="semestral",
        amort_type="bullet",
        nominal_value=1000.0,
    )
    assert fidc._value_scale == 1e8
    assert FIDC is CotaFIDC

    ref_dt = datetime.date(2026, 7, 1)
    m = fidc.risk_metrics(
        ref_dt,
        discount_rate_or_curve=0.025,
        vna=1000.0,
        di_curve=0.135,
        pu_par_val=1010.0,
    )
    assert m["pu"] > 990.0
    assert m["duration_du"] > 0.0
    # T-8 check: no more than 8 decimal places
    assert abs(m["pu"] - math.trunc(m["pu"] * 1e8) / 1e8) < 1e-12


def test_cota_fidc_ipca_pricing():
    fidc = FIDC.from_schedule(
        ticker="FIDC-IPCA",
        issue_date="2025-01-15",
        maturity_date="2028-01-17",
        indexador="IPCA",
        coupon_rate=0.07,
        coupon_freq="semestral",
        amort_type="bullet",
        nominal_value=1000.0,
    )
    m = fidc.risk_metrics(datetime.date(2026, 7, 1), discount_rate_or_curve=0.075, vna=1100.0)
    assert m["pu"] > 1000.0
    assert m["dv01"] > 0.0


def test_letra_financeira_senior_ytm_and_subordinada_ytc():
    lf_sen = LetraFinanceira.from_bullet(
        ticker="LF-ITAU-24M",
        issue_date="2026-01-02",
        maturity_date="2028-01-03",
        indexador="DI",
        coupon_rate=0.004,
        nominal_value=50000.0,
        classe="SENIOR",
        emissor="ITAU",
    )
    assert lf_sen.maturity_date == datetime.date(2028, 1, 3)

    lf_sub = LetraFinanceira.from_bullet(
        ticker="LFSN-ABC-84M",
        issue_date="2026-01-02",
        maturity_date="2033-01-03",
        indexador="DI",
        coupon_rate=0.012,
        nominal_value=50000.0,
        classe="SUBORDINADA",
        emissor="ABC",
        call_date="2031-01-02",
    )
    # Yield to call: effective maturity in cashflows uses call_date
    assert lf_sub.maturity_date == datetime.date(2031, 1, 2)
    m = lf_sub.risk_metrics(
        datetime.date(2026, 7, 1),
        discount_rate_or_curve=0.015,
        vna=50000.0,
        di_curve=0.135,
        pu_par_val=50500.0,
    )
    assert m["pu"] > 45000.0
    assert m["duration_du"] > 900.0


def test_letra_financeira_indicative_spread_from_anbima(tmp_path):
    cache = tmp_path / "anbima_letras_financeiras.json"
    cache.write_text(
        json.dumps(
            {
                "content": [
                    {
                        "letra_financeira": "LF",
                        "data_referencia": "12/11/2026",
                        "emissores": [
                            {
                                "emissor": "ABC",
                                "tipo_taxa": "Indicativa",
                                "vertices": [
                                    {"mes": 12, "taxa": "0,2484"},
                                    {"mes": 24, "taxa": "0,3194"},
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cli = ANBIMADataClient(data_dir=str(tmp_path), manuals_dir=str(tmp_path / "manuais"))
    s12 = LetraFinanceira.indicative_spread_from_anbima("ABC", 12, client=cli)
    s18 = LetraFinanceira.indicative_spread_from_anbima("ABC", 18, client=cli)
    assert abs(s12 - 0.002484) < 1e-9
    assert abs(s18 - (0.002484 + 0.003194) / 2.0) < 1e-9

