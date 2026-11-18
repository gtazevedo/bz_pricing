import datetime
import math
import time

import pytest

from bz_pricing.data.loaders import B3DataLoader
from bz_pricing.instruments import (
    DAPFuture,
    DDIFuture,
    DI1Future,
    DOLFuture,
    INDFuture,
    NDFContract,
    SwapDIxIPCA,
    SwapDIxPRE,
    price_di1_portfolio,
)


def test_di1_future_pu_implied_rate_dv01_and_settlement():
    di1 = DI1Future("DI1F28", "2028-01-03")
    ref = datetime.date(2026, 7, 1)
    du = di1._du(ref)
    rate = 0.135
    pu = di1.pu(ref, rate)
    expected = round(100000.0 / math.pow(1.0 + rate, du / 252.0), 2)
    assert pu == expected

    imp = di1.implied_rate(ref, pu)
    assert abs(imp - rate) < 1e-5

    dv01 = di1.dv01(ref, rate, contracts=10)
    assert dv01 > 0.0
    assert abs(di1.daily_settlement(pu + 15.0, pu, contracts=10, direction=1) - 150.0) < 1e-9


def test_ddi_and_dap_futures():
    ref = datetime.date(2026, 7, 1)
    ddi = DDIFuture("DDIN27", "2027-07-01")
    dc = ddi._dc(ref)
    pu_ddi = ddi.pu(ref, 0.055)
    assert pu_ddi == round(100000.0 / (1.0 + 0.055 * (dc / 360.0)), 2)
    assert ddi.dv01(ref, 0.055, contracts=4) > 0.0

    dap = DAPFuture("DAPQ28", "2028-08-15")
    du = dap._du(ref)
    pu_dap = dap.pu(ref, 0.068)
    assert pu_dap == round(100000.0 / math.pow(1.0 + 0.068, du / 252.0), 2)
    assert dap.dv01(ref, 0.068) > 0.0


def test_dol_and_ind_futures():
    ref = datetime.date(2026, 7, 1)
    dol = DOLFuture("DOLQ26", "2026-08-03")
    wdo = DOLFuture("WDOQ26", "2026-08-03")
    assert dol.MULTIPLIER == 50.0
    assert wdo.MULTIPLIER == 10.0

    f_dol = dol.fair_price(ref, spot_ptax=5.50, pre_rate_or_curve=0.135, cupom_rate_or_curve=0.055)
    assert f_dol > 5.50
    assert abs(wdo.daily_settlement(f_dol + 2.0, f_dol, contracts=5, direction=1) - 100.0) < 1e-9

    ind = INDFuture("INDQ26", "2026-08-19")
    win = INDFuture("WINQ26", "2026-08-19")
    assert ind.MULTIPLIER == 1.0
    assert win.MULTIPLIER == 0.20
    f_ind = ind.fair_price(ref, spot_ibov=130000.0, pre_rate_or_curve=0.135, div_yield=0.03)
    assert f_ind > 130000.0


def test_swaps_and_ndf():
    ref = datetime.date(2026, 7, 1)
    sw_pre = SwapDIxPRE(
        notional=1_000_000.0,
        start_date="2026-01-02",
        maturity_date="2027-01-04",
        fixed_rate=0.14,
        fixed_direction=1,
    )
    m_pre = sw_pre.risk_metrics(ref, pre_rate_or_curve=0.13, di_accrued_factor=1.06)
    assert "mtm" in m_pre and m_pre["dv01"] < 0.0

    sw_ipca = SwapDIxIPCA(
        notional=1_000_000.0,
        start_date="2026-01-02",
        maturity_date="2028-01-03",
        ipca_coupon_rate=0.065,
    )
    m_ipca = sw_ipca.risk_metrics(ref, cupom_ipca_rate_or_curve=0.06, ipca_accrued_factor=1.02, di_accrued_factor=1.06)
    assert m_ipca["pv_ipca"] > 1_000_000.0

    ndf = NDFContract(notional_usd=100_000.0, maturity_date="2026-09-01", strike_brl=5.40, direction=1)
    mtm_ndf = ndf.mtm(ref, spot_ptax=5.55, pre_rate_or_curve=0.135, cupom_rate_or_curve=0.055)
    assert mtm_ndf > 0.0


def test_price_di1_portfolio_cython_batch():
    ref = datetime.date(2026, 7, 1)
    di1 = DI1Future("DI1F28", "2028-01-03")
    positions = [{"instrument": di1, "rate": 0.135, "contracts": 10}] * 500
    t0 = time.perf_counter()
    df = price_di1_portfolio(positions, ref)
    elapsed = time.perf_counter() - t0
    assert len(df) == 500
    assert df["pu"].notna().all()
    assert elapsed < 0.5


def test_di1_from_b3_validation():
    ref = datetime.date(2026, 11, 12)
    ldr = B3DataLoader()
    df_di1 = ldr.get_curve_instruments(ref, asst="DI1")
    valid = df_di1.dropna(subset=["AdjstdQt", "AdjstdQtTax"])
    valid = valid[valid["AdjstdQt"] > 0]
    sample_ticker = str(valid.iloc[0]["TckrSymb"])

    info = DI1Future.from_b3(sample_ticker, ref, loader=ldr)
    pu_calc = info["instrument"].pu(ref, info["adjstd_qt_tax"])
    assert abs(pu_calc - info["adjstd_qt"]) < 1.0


def test_ois_swap_sofr_future_and_cross_currency_swap():
    from bz_pricing.core.curves import FlatCurve
    from bz_pricing.curves.offshore import bootstrap_ois_curve
    from bz_pricing.instruments import CrossCurrencySwap, OISSwap, SOFRFuture

    base_dt = datetime.date(2026, 10, 30)
    sofr_crv = bootstrap_ois_curve(
        [30.0, 90.0, 180.0, 360.0, 720.0, 1080.0, 1800.0],
        [0.0410, 0.0415, 0.0422, 0.0430, 0.0442, 0.0452, 0.0465],
        convention="ACT/360",
    )

    # 1. OISSwap (SOFR ACT/360, ESTR ACT/360, SONIA ACT/365) single & dual curve
    ois_sofr = OISSwap(10_000_000.0, "2026-10-30", "2028-10-30", fixed_rate=0.0442, fixed_direction=1, index_name="SOFR")
    par_r = ois_sofr.par_rate(base_dt, sofr_crv)
    ois_at_par = OISSwap(10_000_000.0, "2026-10-30", "2028-10-30", fixed_rate=par_r, fixed_direction=1, index_name="SOFR")
    assert abs(ois_at_par.mtm(base_dt, sofr_crv)) < 1e-6

    # Curva dupla e contrato seasoned com fator overnight já acruado
    fwd_crv = bootstrap_ois_curve(
        [30.0, 90.0, 180.0, 360.0, 720.0, 1080.0],
        [0.0420, 0.0425, 0.0432, 0.0440, 0.0452, 0.0462],
        convention="ACT/360",
    )
    ois_seasoned = OISSwap(5_000_000.0, "2026-07-30", "2027-07-30", fixed_rate=0.043, fixed_direction=1, index_name="SOFR")
    m_seas = ois_seasoned.risk_metrics(base_dt, discount_curve=sofr_crv, forwarding_curve=fwd_crv, accrued_overnight_factor=1.011)
    assert m_seas["dv01"] < 0.0 and len(ois_seasoned.cashflow_table(base_dt, sofr_crv, fwd_crv, 1.011)) == 1

    sonia_crv = bootstrap_ois_curve([90.0, 180.0, 365.0, 730.0], [0.044, 0.0445, 0.045, 0.0455], convention="ACT/365")
    ois_sonia = OISSwap(1_000_000.0, "2026-10-30", "2028-10-30", fixed_rate=0.0455, index_name="SONIA")
    assert ois_sonia.day_count == "ACT/365"
    assert abs(ois_sonia.mtm(base_dt, sonia_crv)) < 2500.0

    # 2. SOFRFuture SR3 (3M, $25/bp) e SR1 (1M, $41.67/bp)
    sr3 = SOFRFuture("SR3H27", start_date="2027-03-17", end_date="2027-06-16", contract_type="SR3")
    r_sr3 = sr3.fair_rate(base_dt, sofr_crv)
    px_sr3 = sr3.fair_price(base_dt, sofr_crv)
    assert abs(SOFRFuture.implied_rate(px_sr3) - r_sr3) < 1e-12
    assert sr3.dv01(contracts=2.0) == 50.0
    assert abs(sr3.daily_settlement(px_sr3 + 0.05, px_sr3, contracts=4.0) - 500.0) < 1e-9

    sr1 = SOFRFuture("SR1F27", start_date="2027-01-01", end_date="2027-02-01", contract_type="SR1")
    assert sr1.dv01(contracts=1.0) == 41.67
    assert 0.03 < sr1.fair_rate(base_dt, sofr_crv) < 0.06

    # 3. CrossCurrencySwap (USD/BRL DI x SOFR + basis spread)
    pre_crv = FlatCurve(0.140)
    xccy = CrossCurrencySwap(
        notional_foreign=1_000_000.0,
        start_date="2026-10-30",
        maturity_date="2028-10-30",
        initial_fx_spot=5.70,
        brl_direction=1,
        brl_leg_type="DI",
        foreign_leg_type="FLOAT",
        brl_rate=0.0,
        foreign_rate=0.0,
        payment_freq_months=6,
    )
    rm_xccy = xccy.risk_metrics(base_dt, fx_spot=5.70, pre_curve=pre_crv, foreign_discount_curve=sofr_crv)
    # Sem spread nem markup de câmbio no início, PV_brl == N_brl e PV_fx == N_fx => MtM == 0
    assert abs(rm_xccy["mtm_brl"]) < 1e-5
    assert rm_xccy["fx_delta"] < 0.0
    assert abs(rm_xccy["par_basis_spread"]) < 1e-9

    # Em swap Pré BRL x USD Fixed, par_brl_rate zera o MtM exato
    xccy_fix = CrossCurrencySwap(
        notional_foreign=1_000_000.0,
        start_date="2026-10-30",
        maturity_date="2028-10-30",
        initial_fx_spot=5.70,
        brl_leg_type="PRE",
        foreign_leg_type="FIXED",
        brl_rate=0.145,
        foreign_rate=0.045,
        payment_freq_months=6,
    )
    r_par_brl = xccy_fix.par_brl_rate(base_dt, fx_spot=5.70, pre_curve=pre_crv, foreign_discount_curve=sofr_crv)
    xccy_par = CrossCurrencySwap(
        notional_foreign=1_000_000.0,
        start_date="2026-10-30",
        maturity_date="2028-10-30",
        initial_fx_spot=5.70,
        brl_leg_type="PRE",
        foreign_leg_type="FIXED",
        brl_rate=r_par_brl,
        foreign_rate=0.045,
        payment_freq_months=6,
    )
    assert abs(xccy_par.mtm(base_dt, fx_spot=5.70, pre_curve=pre_crv, foreign_discount_curve=sofr_crv)) < 1e-4


