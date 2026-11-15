import datetime
import math
import time
import numpy as np

from bz_pricing.instruments import (
    EquityOption,
    FXOption,
    IDIOption,
    DI1Option,
    CorradoSuOption,
    price_options_portfolio,
)
from bz_pricing.volatility import (
    SVISlice,
    SABRSlice,
    VolatilitySurface,
    delta_to_strike,
    strike_to_delta,
)


def test_equity_option_bsm_american_and_iv():
    base_dt = datetime.date(2026, 7, 1)
    mat_dt = datetime.date(2027, 1, 4)
    spot = 38.50
    strike = 40.00
    r_pre = 0.14
    q_div = 0.04
    vol = 0.32

    call_eu = EquityOption("PETRC400", mat_dt, strike, "CALL", "EUROPEAN")
    put_eu = EquityOption("PETRO400", mat_dt, strike, "PUT", "EUROPEAN")
    put_am = EquityOption("PETRO400A", mat_dt, strike, "PUT", "AMERICAN")
    call_am_nodiv = EquityOption("PETRC400A", mat_dt, strike, "CALL", "AMERICAN")

    mc = call_eu.risk_metrics(base_dt, spot, r_pre, vol, q_div)
    mp = put_eu.risk_metrics(base_dt, spot, r_pre, vol, q_div)

    # Paridade Put-Call: C - P = S * exp(-qT) - K * exp(-rT)
    t_val = mc["T"]
    rhs = spot * math.exp(-mc["q_cont"] * t_val) - strike * math.exp(-mc["r_cont"] * t_val)
    assert abs((mc["price"] - mp["price"]) - rhs) < 1e-8
    assert abs((mc["delta"] - mp["delta"]) - math.exp(-mc["q_cont"] * t_val)) < 1e-8
    assert abs(mc["gamma"] - mp["gamma"]) < 1e-10
    assert abs(mc["vega"] - mp["vega"]) < 1e-10

    # Inversão de volatilidade implícita
    iv_c = call_eu.implied_volatility(base_dt, mc["price"], spot, r_pre, q_div)
    iv_p = put_eu.implied_volatility(base_dt, mp["price"], spot, r_pre, q_div)
    assert abs(iv_c - vol) < 1e-7
    assert abs(iv_p - vol) < 1e-7

    # Americana: Put americana >= Put europeia; Call americana sem dividendo == Call europeia
    p_am = put_am.price(base_dt, spot, r_pre, vol, q_div, crr_steps=300)
    assert p_am >= mp["price"] - 1e-4

    c_eu_0 = call_eu.price(base_dt, spot, r_pre, vol, 0.0)
    c_am_0 = call_am_nodiv.price(base_dt, spot, r_pre, vol, 0.0, crr_steps=300)
    assert abs(c_am_0 - c_eu_0) < 0.03


def test_fx_option_garman_kohlhagen_and_fixing_adjust():
    base_dt = datetime.date(2026, 7, 1)
    mat_dt = datetime.date(2026, 10, 1)
    spot = 5.75
    strike = 5.80
    r_pre = 0.145
    cupom_ddi = 0.055
    vol = 0.16

    call_fx = FXOption("DOLC580", mat_dt, strike, "CALL")
    put_fx = FXOption("DOLP580", mat_dt, strike, "PUT")

    mc = call_fx.risk_metrics(base_dt, spot, r_pre, cupom_ddi, vol, fixing_vol_adjust=False)
    mp = put_fx.risk_metrics(base_dt, spot, r_pre, cupom_ddi, vol, fixing_vol_adjust=False)
    t_val = mc["T"]
    rhs = spot * math.exp(-mc["q_cont"] * t_val) - strike * math.exp(-mc["r_cont"] * t_val)
    assert abs((mc["price"] - mp["price"]) - rhs) < 1e-8

    mc_adj = call_fx.risk_metrics(base_dt, spot, r_pre, cupom_ddi, vol, fixing_vol_adjust=True)
    assert mc_adj["vol_eff"] < vol
    assert mc_adj["price"] < mc["price"]

    iv = call_fx.implied_volatility(base_dt, mc["price"], spot, r_pre, cupom_ddi)
    assert abs(iv - vol) < 1e-7


def test_idi_and_di1_options_and_corrado_su():
    base_dt = datetime.date(2026, 7, 1)
    opt_mat = datetime.date(2026, 10, 1)
    fut_mat = datetime.date(2027, 7, 1)

    idi_call = IDIOption("IDIC250", opt_mat, 250000.0, "CALL")
    p_idi = idi_call.price(base_dt, idi_spot=242000.0, pre_rate_or_curve=0.14, vol=0.05)
    iv_idi = idi_call.implied_volatility(base_dt, p_idi, idi_spot=242000.0, pre_rate_or_curve=0.14)
    assert p_idi > 0.0
    assert abs(iv_idi - 0.05) < 1e-7

    di1_call = DI1Option("DI1C14", opt_mat, fut_mat, strike_rate=0.14, option_type="CALL")
    p_di1 = di1_call.price(base_dt, pu_c=96500.0, pu_l=87000.0, vol=0.18)
    iv_di1 = di1_call.implied_volatility(base_dt, p_di1, pu_c=96500.0, pu_l=87000.0)
    assert p_di1 > 0.0
    assert abs(iv_di1 - 0.18) < 1e-7

    cs_opt = CorradoSuOption("VALEC60", opt_mat, 60.0, "CALL")
    eq_opt = EquityOption("VALEC60", opt_mat, 60.0, "CALL")
    p_cs0 = cs_opt.price(base_dt, 58.0, 0.14, 0.30, kappa3=0.0, kappa4=3.0)
    p_eq0 = eq_opt.price(base_dt, 58.0, 0.14, 0.30)
    assert abs(p_cs0 - p_eq0) < 1e-8
    p_cs_fat = cs_opt.price(base_dt, 58.0, 0.14, 0.30, kappa3=-0.4, kappa4=4.5)
    assert p_cs_fat != p_eq0


def test_svi_sabr_and_volatility_surface_no_arbitrage():
    fwd = 100.0
    k_grid = np.linspace(80.0, 120.0, 15)
    log_m = np.log(k_grid / fwd)

    # SVI sintético livre de arbitragem em dois vencimentos (63 DU e 126 DU)
    sl_1 = SVISlice(a=0.015, b=0.04, rho=-0.35, m=0.0, sigma=0.12, T=63 / 252.0)
    sl_2 = SVISlice(a=0.032, b=0.07, rho=-0.35, m=0.0, sigma=0.14, T=126 / 252.0)
    vols_1 = sl_1.implied_vol(log_m)
    fit_1 = SVISlice.fit(log_m, vols_1, T=63 / 252.0)
    assert np.max(np.abs(fit_1.implied_vol(log_m) - vols_1)) < 1e-4

    surf_svi = VolatilitySurface.from_svi_slices({63: sl_1, 126: sl_2})
    v_90 = surf_svi.get_vol(90, strike=95.0, forward=fwd)
    assert sl_1.implied_vol(math.log(95.0 / fwd)) < v_90 + 0.05
    arb = surf_svi.check_no_arbitrage(forward_by_du=fwd, r_cont_by_du=0.13)
    assert arb["passed"] is True

    # SABR para juros
    sabr_true = SABRSlice(alpha=0.035, beta=0.5, rho=-0.25, nu=0.35, F=0.14, T=0.5)
    k_ir = np.linspace(0.11, 0.17, 9)
    v_ir = sabr_true.implied_vol(k_ir)
    sabr_fit = SABRSlice.fit(0.14, k_ir, v_ir, T=0.5, beta=0.5)
    assert np.max(np.abs(sabr_fit.implied_vol(k_ir) - v_ir)) < 1e-4

    # Grid por Delta + conversão Delta <-> Strike
    deltas = [0.10, 0.25, 0.50, 0.75, 0.90]
    vols_mat = [
        [0.29, 0.26, 0.24, 0.25, 0.27],
        [0.30, 0.27, 0.25, 0.26, 0.28],
    ]
    surf_d = VolatilitySurface.from_delta_grid([63, 126], deltas, vols_mat)
    assert abs(surf_d.get_vol(63, delta=0.50) - 0.24) < 1e-8
    k_25 = delta_to_strike(0.25, fwd, 0.26, 63 / 252.0)
    d_back = strike_to_delta(k_25, fwd, 0.26, 63 / 252.0)
    assert abs(d_back - 0.25) < 1e-6


def test_price_options_portfolio_cython_speed():
    base_dt = datetime.date(2026, 7, 1)
    opt_eu = EquityOption("PETRC40", "2026-10-01", 40.0, "CALL", "EUROPEAN")
    opt_am = EquityOption("PETRO38", "2026-10-01", 38.0, "PUT", "AMERICAN")

    positions = [
        {"instrument": opt_eu, "spot": 39.0, "pre_rate_or_curve": 0.14, "vol": 0.30, "quantity": 100},
        {"instrument": opt_am, "spot": 39.0, "pre_rate_or_curve": 0.14, "vol": 0.32, "quantity": -50},
    ] * 500  # 1.000 opções

    t0 = time.perf_counter()
    df = price_options_portfolio(positions, base_dt, crr_steps=100)
    elapsed = time.perf_counter() - t0
    assert len(df) == 1000
    assert df["price"].notna().all()
    assert elapsed < 1.5

