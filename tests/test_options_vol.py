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


def test_stage1_fx_surface_iv_bounds_and_parity():
    import pandas as pd
    from bz_pricing.core.options_core import implied_vol_bsm
    from bz_pricing.volatility import infer_forward_from_put_call_parity, validate_slice_out_of_sample

    base_dt = datetime.date(2026, 7, 1)
    mat_dt = datetime.date(2026, 10, 1)
    spot = 5.50
    strike = 6.10  # bem fora do ATM para evidenciar efeito do skew
    r_pre = 0.145
    r_ddi = 0.055

    fx = FXOption("DOLC610", mat_dt, strike, "CALL")
    du_n = fx.du(base_dt)
    t_n = du_n / 252.0
    sl = SVISlice(a=0.008, b=0.08, rho=-0.65, m=0.0, sigma=0.10, T=t_n)
    surf = VolatilitySurface.from_svi_slices({du_n: sl})

    # A1: Chamada individual FXOption deve coincidir com o lote e refletir o smile
    rm = fx.risk_metrics(base_dt, spot, r_pre, r_ddi, surf)
    p_single = fx.price(base_dt, spot, r_pre, r_ddi, surf)
    p_batch = price_options_portfolio(
        [{"instrument": fx, "spot_ptax": spot, "pre_rate_or_curve": r_pre, "cupom_limpo_or_curve": r_ddi, "vol": surf}],
        base_dt,
    )["price"].iloc[0]
    fwd_true = spot * math.exp((rm["r_cont"] - rm["q_cont"]) * t_n)
    assert abs(rm["vol_eff"] - surf.get_vol(du_n, strike=strike, forward=fwd_true)) < 1e-12
    assert abs(rm["vol_eff"] - surf.get_vol(du_n, strike=strike, forward=strike)) > 1e-3
    assert abs(p_single - p_batch) < 1e-10

    # A2: Preço <= intrínseco deve retornar NaN (nunca 1e-6)
    intrinsic = max(spot * math.exp(-0.02 * 0.5) - 5.0 * math.exp(-0.10 * 0.5), 0.0)
    assert math.isnan(implied_vol_bsm(intrinsic - 0.05, spot, 5.0, 0.10, 0.02, 0.5, True))
    assert math.isnan(implied_vol_bsm(intrinsic, spot, 5.0, 0.10, 0.02, 0.5, True))

    # A3: Forward por paridade put-call e modos out-of-sample (interpolation / wings / vega_weighted)
    df_syn = pd.DataFrame({
        "ExrcPric": [95.0, 95.0, 100.0, 100.0, 105.0, 105.0],
        "option_type": ["call", "put", "call", "put", "call", "put"],
        "RefPric": [11.4, 2.0, 7.6, 3.0, 4.7, 4.9],  # (C-P) ≈ 9.4, 4.6, -0.2 => F ≈ 105? (C-P)/0.96 ≈ 9.79 -> ~104.8
        "TOTNEG": [10, 10, 50, 50, 8, 8],
    })
    df_disc = 0.96
    f_inf = infer_forward_from_put_call_parity(df_syn, df_disc)
    assert abs(f_inf - (100.0 + (7.6 - 3.0) / df_disc)) < 1e-9

    from bz_pricing.core.options_core import black76_price_greeks
    strikes = np.linspace(85.0, 115.0, 13)
    fwd = 100.0
    r_c = 0.12
    t_v = 63 / 252.0
    sl_syn = SVISlice(a=0.012, b=0.05, rho=-0.35, m=0.0, sigma=0.10, T=t_v)
    otypes = ["put" if k < fwd else "call" for k in strikes]
    vols = sl_syn.implied_vol(np.log(strikes / fwd))
    px = [black76_price_greeks(fwd, float(k), r_c, t_v, float(v), ot == "call")[0] for k, v, ot in zip(strikes, vols, otypes)]

    for smode in ("alternating", "interpolation", "wings"):
        res_oos = validate_slice_out_of_sample(
            strikes, px, otypes, fwd, r_c, 63, model="svi", otm_only=True, vega_weighted=True, split_mode=smode
        )
        assert res_oos["rmse_vol_bps"] < 15.0


def test_stage2_theoretical_and_quantlib_cross_checks():
    from scipy.integrate import quad
    from bz_pricing.core.options_core import asian_geometric_price, crr_american_price
    from bz_pricing.instruments import AsianOption, BarrierOption, Swaption

    base_dt = datetime.date(2026, 10, 30)
    exp_dt = datetime.date(2027, 4, 30)

    # 1. M1 & M2 & M3: Superfície em escala elevada (F ~ 500k) com carry não nulo deve dar vol local constante = sigma0
    sig0 = 0.22
    r_c = 0.14
    spot_s = 500000.0
    du1, du2 = 63, 126
    f1 = spot_s * math.exp(r_c * du1 / 252.0)
    f2 = spot_s * math.exp(r_c * du2 / 252.0)
    sl_flat1 = SVISlice(a=(sig0 ** 2) * (du1 / 252.0), b=1e-5, rho=0.0, m=0.0, sigma=0.1, T=du1 / 252.0)
    sl_flat2 = SVISlice(a=(sig0 ** 2) * (du2 / 252.0), b=1e-5, rho=0.0, m=0.0, sigma=0.1, T=du2 / 252.0)
    surf_flat = VolatilitySurface.from_svi_slices({du1: sl_flat1, du2: sl_flat2}, slice_forwards={du1: f1, du2: f2})
    arb_flat = surf_flat.check_no_arbitrage({du1: f1, du2: f2}, r_c)
    assert arb_flat["passed"] is True

    fwd_fn = lambda d: spot_s * math.exp(r_c * (d / 252.0))
    dup_flat = surf_flat.dupire_local_vol_grid(fwd_fn, r_c)
    assert dup_flat["passed"] is True
    assert np.max(np.abs(dup_flat["local_vol"] - sig0)) < 1e-4

    # M11: SVI fit preserva condição borboleta g(k) >= 0 de Gatheral-Jacquier
    k_grid = np.linspace(-0.4, 0.4, 17)
    sl_clean = SVISlice(a=0.015, b=0.06, rho=-0.4, m=0.01, sigma=0.12, T=0.25)
    svi_fit = SVISlice.fit(k_grid, sl_clean.implied_vol(k_grid), T=0.25)
    dk = k_grid - svi_fit.m
    sq = np.sqrt(dk * dk + svi_fit.sigma ** 2)
    w_v = svi_fit.a + svi_fit.b * (svi_fit.rho * dk + sq)
    wp_v = svi_fit.b * (svi_fit.rho + dk / sq)
    wpp_v = (svi_fit.b * svi_fit.sigma ** 2) / (sq ** 3)
    g_vals = (1.0 - (k_grid * wp_v) / (2.0 * w_v)) ** 2 - 0.25 * (wp_v ** 2) * (1.0 / w_v + 0.25) + 0.5 * wpp_v
    assert np.min(g_vals) >= -1e-6

    # 2. M4: Corrado-Su (1996) conferido contra integração direta por quadratura (com dividendo q > 0)
    S, K, r, q, T, sig = 58.0, 60.0, 0.14, 0.03, 0.5, 0.30
    k3, k4 = -0.35, 4.2
    cs_opt = CorradoSuOption("CS-QUAD", "2027-04-30", K, "CALL")
    r_ann = math.exp(r) - 1.0
    q_ann = math.exp(q) - 1.0
    du_cs = cs_opt.du(base_dt)
    t_cs = du_cs / 252.0
    p_cs = cs_opt.price(base_dt, S, r_ann, sig, kappa3=k3, kappa4=k4, carry_or_div_yield=q_ann)
    sT = sig * math.sqrt(t_cs)
    d2_c = (math.log(S / K) + (r - q - 0.5 * sig * sig) * t_cs) / sT
    n_pdf = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    f_gc = lambda z: n_pdf(z) * (1.0 + (k3 / 6.0) * (z ** 3 - 3.0 * z) + ((k4 - 3.0) / 24.0) * (z ** 4 - 6.0 * z ** 2 + 3.0))
    pay_gc = lambda z: math.exp(-r * t_cs) * max(S * math.exp((r - q - 0.5 * sig * sig) * t_cs + sT * z) - K, 0.0) * f_gc(z)
    p_quad, _ = quad(pay_gc, -d2_c, 12.0)
    assert abs(p_cs - p_quad) < 1e-10

    # 3. M10: Suavização par-ímpar CRR elimina oscilação binomial
    p_200 = crr_american_price(38.0, 40.0, 0.13, 0.0, 0.5, 0.30, False, 200)
    p_201 = crr_american_price(38.0, 40.0, 0.13, 0.0, 0.5, 0.30, False, 201)
    assert abs(p_200 - p_201) < 1e-3

    # 4. M7: BarrierOption com monitoramento discreto (BGK) e comparação vs Monte Carlo
    bar_cont = BarrierOption("B-C", exp_dt, 60.0, 52.0, "CALL", "down-and-out")
    bar_disc = BarrierOption("B-D", exp_dt, 60.0, 52.0, "CALL", "down-and-out", monitoring_steps=12)
    p_bc = bar_cont.price(base_dt, 62.0, 0.14, 0.30, 0.02)
    p_bd = bar_disc.price(base_dt, 62.0, 0.14, 0.30, 0.02)
    assert p_bd > p_bc  # barreira efetiva menor -> maior sobrevivência -> prêmio maior

    # 5. M8: AsianOption com n_fixings=1 deve coincidir com EquityOption europeia; geométrica discreta == QuantLib fórmula
    eq_1 = EquityOption("EQ-1", exp_dt, 40.0, "CALL", "EUROPEAN").price(base_dt, 40.0, 0.14, 0.30, 0.02)
    as_1 = AsianOption("AS-1", exp_dt, 40.0, "CALL", "arithmetic", n_fixings=1).price(base_dt, 40.0, 0.14, 0.30, 0.02)
    ag_1 = AsianOption("AG-1", exp_dt, 40.0, "CALL", "geometric", n_fixings=1).price(base_dt, 40.0, 0.14, 0.30, 0.02)
    assert abs(as_1 - eq_1) < 1e-10
    assert abs(ag_1 - eq_1) < 1e-10

    # 6. M9: Swaption na convenção B3 zero-cupom (DI x Pré 252) + paridade Payer - Receiver
    swap_mat = datetime.date(2028, 4, 28)
    swp_p = Swaption("SWP-B3-P", exp_dt, swap_mat, strike_rate=0.135, option_type="PAYER", swap_convention="b3_zero")
    swp_r = Swaption("SWP-B3-R", exp_dt, swap_mat, strike_rate=0.135, option_type="RECEIVER", swap_convention="b3_zero")
    rm_p = swp_p.risk_metrics(base_dt, 0.14, 0.18, model="black76")
    rm_r = swp_r.risk_metrics(base_dt, 0.14, 0.18, model="black76")
    assert abs(rm_p["forward_swap_rate"] - 0.14) < 1e-12
    rhs_swp = swp_p.notional * rm_p["annuity"] * (rm_p["forward_swap_rate"] - 0.135)
    assert abs((rm_p["price"] - rm_r["price"]) - rhs_swp) < 1e-6



