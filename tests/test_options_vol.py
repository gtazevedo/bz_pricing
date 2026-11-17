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


def test_b3_loader_options_real_data():
    import os
    from bz_pricing.data.loaders import B3DataLoader

    if not os.path.exists("dados/b3/BVBG_IN_20261030.csv") or not os.path.exists("dados/b3/BVBG_SPRD_20261030.csv"):
        return

    loader = B3DataLoader(data_dir="dados/b3")
    ref_dt = datetime.date(2026, 10, 30)

    cat_dol = loader.load_options_catalog(ref_dt, asst="DOL", auto_download=False)
    assert len(cat_dol) > 0
    assert set(cat_dol["option_type"].dropna().unique()).issubset({"call", "put"})
    assert "european" in set(cat_dol["option_style"].dropna().unique())

    dol_priced = loader.load_options(ref_dt, asst="DOL", only_priced=True, auto_download=False)
    assert len(dol_priced) >= 10
    assert (dol_priced["RefPric"] > 0).all()

    idi_priced = loader.load_options(ref_dt, asst="IDI", only_priced=True, auto_download=False)
    assert len(idi_priced) >= 10


def test_surface_advanced_validation_breeden_dupire_and_oos():
    from bz_pricing.core.options_core import black76_price_greeks
    from bz_pricing.volatility import validate_slice_out_of_sample

    fwd = 5.55
    r_c = 0.14
    sl_1 = SVISlice(a=0.008, b=0.025, rho=-0.30, m=0.0, sigma=0.10, T=63 / 252.0)
    sl_2 = SVISlice(a=0.018, b=0.045, rho=-0.30, m=0.0, sigma=0.12, T=126 / 252.0)
    surf = VolatilitySurface.from_svi_slices({63: sl_1, 126: sl_2})

    wing = surf.wing_slope_check()
    assert wing["passed"] is True

    bl = surf.breeden_litzenberger_density(63, forward=fwd, r_cont=r_c)
    assert bl["passed"] is True
    assert bl["min_density"] >= -1e-6
    assert 0.95 <= bl["integral"] <= 1.02

    dup = surf.dupire_local_vol_grid(forward_by_du=fwd, r_cont_by_du=r_c)
    assert dup["passed"] is True
    assert dup["negative_nodes"] == 0
    assert dup["min_local_var"] > 0.0

    # Validação out-of-sample (SVI e SABR) sobre preços sintéticos exatos
    strikes = np.linspace(4.9, 6.2, 11)
    opt_types = ["put" if k < fwd else "call" for k in strikes]
    t_val = 63 / 252.0
    vols = sl_1.implied_vol(np.log(strikes / fwd))
    prices = [
        black76_price_greeks(fwd, float(k), r_c, t_val, float(v), ot == "call")[0]
        for k, v, ot in zip(strikes, vols, opt_types)
    ]

    oos_svi = validate_slice_out_of_sample(strikes, prices, opt_types, fwd, r_c, 63, model="svi")
    assert oos_svi["rmse_vol_bps"] < 5.0
    assert oos_svi["rmse_price"] < 1e-3

    oos_sabr = validate_slice_out_of_sample(strikes, prices, opt_types, fwd, r_c, 63, model="sabr")
    assert oos_sabr["rmse_vol_bps"] < 50.0


def test_options_and_surface_edge_case_fixes():
    from bz_pricing.core.options_core import sabr_implied_vol, implied_vol_bsm

    # 1. SABR com rho nos extremos não pode dar ZeroDivisionError ou NaN
    v_extreme = sabr_implied_vol(0.14, 0.12, 0.5, 0.035, 0.5, 1.0, 0.35)
    assert math.isfinite(v_extreme) and v_extreme > 0.0

    # 2. Corrado-Su gaussiano (kappa3=0, kappa4=3) deve coincidir exatamente com BSM
    base_dt = datetime.date(2026, 7, 1)
    mat_dt = datetime.date(2027, 1, 4)
    eq_c = EquityOption("PETRC40", mat_dt, 40.0, "CALL", "EUROPEAN")
    cs_c = CorradoSuOption("PETRC40", mat_dt, 40.0, "CALL", "EUROPEAN")
    p_eq = eq_c.price(base_dt, 39.0, 0.14, 0.30)
    p_cs = cs_c.price(base_dt, 39.0, 0.14, 0.30, kappa3=0.0, kappa4=3.0)
    assert abs(p_cs - p_eq) < 1e-10

    # 3. Gregas CRR consistentes para put americana deep ITM (delta mais próximo de -1 que BSM europeu)
    am_put = EquityOption("VALEO50", mat_dt, 55.0, "PUT", "AMERICAN")
    eu_put = EquityOption("VALEO50", mat_dt, 55.0, "PUT", "EUROPEAN")
    rm_am = am_put.risk_metrics(base_dt, 38.0, 0.14, 0.25, crr_steps=200)
    rm_eu = eu_put.risk_metrics(base_dt, 38.0, 0.14, 0.25)
    assert rm_am["price"] > rm_eu["price"]
    assert rm_am["delta"] < rm_eu["delta"]

    # 4. FXOption.implied_volatility com fixing_vol_adjust=True recupera a vol original
    fx_c = FXOption("DOLC580", mat_dt, 5.80, "CALL")
    p_fx_adj = fx_c.price(base_dt, 5.65, 0.14, 0.05, 0.18, fixing_vol_adjust=True)
    iv_fx_adj = fx_c.implied_volatility(base_dt, p_fx_adj, 5.65, 0.14, 0.05, fixing_vol_adjust=True)
    assert abs(iv_fx_adj - 0.18) < 1e-6

    # 5. implied_vol_bsm retorna NaN acima do limite superior de não arbitragem
    iv_nan = implied_vol_bsm(100.0, 50.0, 40.0, 0.10, 0.0, 1.0, True)
    assert math.isnan(iv_nan)

    # 6. Instrumentos aceitam VolatilitySurface diretamente
    sl = SVISlice(a=0.015, b=0.04, rho=-0.30, m=0.0, sigma=0.12, T=eq_c.du(base_dt) / 252.0)
    surf = VolatilitySurface.from_svi_slices({eq_c.du(base_dt): sl})
    p_surf = eq_c.price(base_dt, 39.0, 0.14, surf)
    assert math.isfinite(p_surf) and p_surf > 0.0


def test_swaption_capfloor_barrier_and_asian():
    from bz_pricing.instruments import Swaption, CapFloor, BarrierOption, AsianOption

    base_dt = datetime.date(2026, 10, 30)
    exp_dt = datetime.date(2027, 4, 30)
    swap_mat = datetime.date(2028, 4, 28)

    # 1. Swaption (Black-76 e Bachelier Normal + inversão de vol implícita)
    swp = Swaption("SWP-DI-28", exp_dt, swap_mat, strike_rate=0.138, option_type="PAYER", notional=1_000_000.0)
    p_b76 = swp.price(base_dt, 0.140, 0.18, model="black76")
    iv_b76 = swp.implied_volatility(base_dt, p_b76, 0.140, model="black76")
    assert p_b76 > 0.0 and abs(iv_b76 - 0.18) < 1e-6

    p_bach = swp.price(base_dt, 0.140, 0.0150, model="bachelier")
    iv_bach = swp.implied_volatility(base_dt, p_bach, 0.140, model="bachelier")
    assert p_bach > 0.0 and abs(iv_bach - 0.0150) < 1e-6

    # 2. CapFloor (Cap = soma dos caplets; Black-76 e Bachelier)
    fix_dates = [datetime.date(2027, 1, 4), datetime.date(2027, 4, 1)]
    pay_dates = [datetime.date(2027, 4, 1), datetime.date(2027, 7, 1)]
    cap = CapFloor("CAP-DI", fix_dates, pay_dates, strike_rate=0.135, option_type="CAP", notional=1_000_000.0)
    df_cap = cap.cashflow_table(base_dt, 0.142, 0.20, model="black76")
    assert len(df_cap) == 2
    assert abs(cap.price(base_dt, 0.142, 0.20, model="black76") - df_cap["caplet_pv"].sum()) < 1e-8
    assert cap.price(base_dt, 0.142, 0.015, model="bachelier") > 0.0

    # 3. Paridade In-Out em BarrierOption (down-and-in + down-and-out == vanilla; up-and-in + up-and-out == vanilla)
    van_c = EquityOption("VALE-C", exp_dt, 60.0, "CALL", "EUROPEAN").price(base_dt, 62.0, 0.14, 0.30, 0.02)
    do_c = BarrierOption("VALE-DO", exp_dt, 60.0, 52.0, "CALL", "down-and-out").price(base_dt, 62.0, 0.14, 0.30, 0.02)
    di_c = BarrierOption("VALE-DI", exp_dt, 60.0, 52.0, "CALL", "down-and-in").price(base_dt, 62.0, 0.14, 0.30, 0.02)
    assert abs((do_c + di_c) - van_c) < 1e-8

    van_p = EquityOption("VALE-P", exp_dt, 60.0, "PUT", "EUROPEAN").price(base_dt, 62.0, 0.14, 0.30, 0.02)
    uo_p = BarrierOption("VALE-UO", exp_dt, 60.0, 72.0, "PUT", "up-and-out").price(base_dt, 62.0, 0.14, 0.30, 0.02)
    ui_p = BarrierOption("VALE-UI", exp_dt, 60.0, 72.0, "PUT", "up-and-in").price(base_dt, 62.0, 0.14, 0.30, 0.02)
    assert abs((uo_p + ui_p) - van_p) < 1e-8

    # 4. AsianOption: média geométrica <= média aritmética (para call ATM) <= europeia vanilla
    ag_c = AsianOption("PETR-AG", exp_dt, 40.0, "CALL", "geometric").price(base_dt, 40.0, 0.14, 0.30, 0.0)
    aa_c = AsianOption("PETR-AA", exp_dt, 40.0, "CALL", "arithmetic").price(base_dt, 40.0, 0.14, 0.30, 0.0)
    ev_c = EquityOption("PETR-EV", exp_dt, 40.0, "CALL", "EUROPEAN").price(base_dt, 40.0, 0.14, 0.30, 0.0)
    assert 0.0 < ag_c <= aa_c <= ev_c


def test_batch_portfolios_for_fx_idi_di1_and_mixed():
    from bz_pricing.instruments import (
        price_fx_options_portfolio,
        price_idi_options_portfolio,
        price_di1_options_portfolio,
    )

    base_dt = datetime.date(2026, 7, 1)
    mat_dt = datetime.date(2027, 1, 4)

    eq = EquityOption("PETRC40", mat_dt, 40.0, "CALL", "EUROPEAN")
    fx = FXOption("DOLC580", mat_dt, 5.80, "CALL")
    idi = IDIOption("IDIC600K", mat_dt, 600000.0, "CALL")
    di1 = DI1Option("DI1C140", mat_dt, datetime.date(2028, 1, 3), 0.14)
    cs = CorradoSuOption("VALEC60", mat_dt, 60.0, "CALL")

    df_mix = price_options_portfolio(
        [
            {"instrument": eq, "spot": 39.0, "pre_rate_or_curve": 0.14, "vol": 0.30, "quantity": 10},
            {"instrument": fx, "spot_ptax": 5.65, "pre_rate_or_curve": 0.14, "cupom_limpo_or_curve": 0.05, "vol": 0.18, "quantity": 5},
            {"instrument": idi, "idi_spot": 560000.0, "pre_rate_or_curve": 0.14, "vol": 0.025, "quantity": 2},
            {"instrument": di1, "pu_c": 93500.0, "pu_l": 81500.0, "vol": 0.15, "quantity": 1},
            {"instrument": cs, "spot": 58.0, "pre_rate_or_curve": 0.14, "vol": 0.30, "kappa3": -0.3, "kappa4": 4.0, "quantity": 3},
        ],
        base_dt,
    )
    assert len(df_mix) == 5
    assert df_mix["price"].notna().all()
    assert abs(df_mix.loc[1, "price"] - fx.price(base_dt, 5.65, 0.14, 0.05, 0.18)) < 1e-8
    assert abs(df_mix.loc[2, "price"] - idi.price(base_dt, 560000.0, 0.14, 0.025)) < 1e-8
    assert abs(df_mix.loc[3, "price"] - di1.price(base_dt, 93500.0, 81500.0, 0.15)) < 1e-8

    assert len(price_fx_options_portfolio([{"instrument": fx, "spot_ptax": 5.65, "pre_rate_or_curve": 0.14, "cupom_limpo_or_curve": 0.05, "vol": 0.18}], base_dt)) == 1
    assert len(price_idi_options_portfolio([{"instrument": idi, "idi_spot": 560000.0, "pre_rate_or_curve": 0.14, "vol": 0.025}], base_dt)) == 1
    assert len(price_di1_options_portfolio([{"instrument": di1, "pu_c": 93500.0, "pu_l": 81500.0, "vol": 0.15}], base_dt)) == 1


def test_b3_loader_equity_options_cotahist():
    import os
    from bz_pricing.data.loaders import B3DataLoader

    if not os.path.exists("dados/b3/COTAHIST_D20261030.csv"):
        return

    loader = B3DataLoader(data_dir="dados/b3")
    ref_dt = datetime.date(2026, 10, 30)
    bova = loader.load_equity_options(ref_dt, underlying="BOVA11", only_priced=True, auto_download=False)
    assert len(bova) > 100
    assert "spot_price" in bova.columns and bova["spot_price"].iloc[0] > 100.0
    assert set(bova["option_type"].unique()).issubset({"call", "put"})

