import datetime
import math
import time
import numpy as np

from bz_pricing.instruments import (
    AsianOption,
    BarrierOption,
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


def test_stage2_theoretical_cross_checks():
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

    # 5. M8: AsianOption com n_fixings=1 deve coincidir com EquityOption europeia
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


def test_bbg_part1_filtering_deamericanization_kalman_and_blended_dividends():
    import pandas as pd
    from bz_pricing.core.options_core import crr_american_price
    from bz_pricing.volatility import (
        DividendCurveBBG,
        KalmanDividendFilter,
        deamericanize_and_infer_forward_bbg,
        filter_option_quotes_bbg,
    )

    # Filtragem de cotações (spread <= 0,09% do spot para T <= 1 ano)
    raw = pd.DataFrame([
        {"strike": 90.0, "option_type": "call", "bid": 14.17, "ask": 14.23, "mid": 14.20},
        {"strike": 95.0, "option_type": "call", "bid": 10.17, "ask": 10.23, "mid": 10.20},
        {"strike": 100.0, "option_type": "call", "bid": 11.67, "ask": 11.73, "mid": 11.70},
        {"strike": 105.0, "option_type": "call", "bid": 4.37, "ask": 4.43, "mid": 4.40},
        {"strike": 90.0, "option_type": "put", "bid": 0.0001, "ask": 0.05, "mid": 0.025},
        {"strike": 95.0, "option_type": "put", "bid": 1.20, "ask": 5.00, "mid": 3.10},
        {"strike": 100.0, "option_type": "put", "bid": 4.27, "ask": 4.33, "mid": 4.30},
    ])
    flt = filter_option_quotes_bbg(raw, spot=100.0, maturity_years=0.5)
    assert len(flt[flt["option_type"] == "call"]) == 3
    assert 100.0 not in flt[flt["option_type"] == "call"]["strike"].values
    assert len(flt[flt["option_type"] == "put"]) == 1

    # Desamericanização iterativa e recuperação do forward
    spot, r, q_true, T, sig = 100.0, 0.10, 0.03, 0.5, 0.25
    ks = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    c_am = [crr_american_price(spot, k, r, q_true, T, sig, True, 120) for k in ks]
    p_am = [crr_american_price(spot, k, r, q_true, T, sig, False, 120) for k in ks]
    out_am = deamericanize_and_infer_forward_bbg(ks, c_am, p_am, spot=spot, r_cont=r, T=T, exercise="AMERICAN", crr_steps=120)
    f_true = spot * math.exp((r - q_true) * T)
    assert abs(out_am["forward"] - f_true) < 0.15

    # Filtro de Kalman com piso e aceleração em violação bid/ask
    kf = KalmanDividendFilter(alpha_over_sigma=1.0)
    kf.update(0.020, d_bid=0.015, d_ask=0.025)
    u2 = kf.update(0.021, d_bid=0.015, d_ask=0.025)
    assert u2["gain"] >= 0.03 and not u2["violation"]
    u3 = kf.update(0.032, d_bid=0.026, d_ask=0.035)
    assert u3["violation"] and abs(u3["gain"] - 0.5) < 1e-12
    u4 = kf.update(0.010, d_bid=0.005, d_ask=0.015, elapsed_days=8.0)
    assert abs(u4["dividend"] - 0.010) < 1e-12

    # Sazonalidade, horário estendido e Blended Dividends
    dc_bbg = DividendCurveBBG(is_index=False)
    wts = dc_bbg.fit_seasonality(
        ["2024-03-10", "2024-09-10", "2025-03-12", "2025-09-12"],
        [1.0, 3.0, 1.0, 3.0],
    )
    assert abs(np.sum(wts) - 1.0) < 1e-12 and wts[8] > wts[2]
    s_ets = dc_bbg.extended_trading_hours_spot(5000.0, 5050.0, 0.05, 5040.0, 0.05, 0.25)
    assert abs(s_ets - 5000.0 * (5050.0 / 5040.0)) < 1e-8

    bd = dc_bbg.decompose_blended_dividends([0.5, 1.5, 3.0, 11.0], spot_f0=100.0, fwd_T=96.0, r_cont=0.05, T=11.5)
    assert abs(bd["w_d"][0] - 1.0) < 1e-12 and abs(bd["w_q"][3] - 1.0) < 1e-12
    f_cur = 100.0
    t_prev = 0.0
    for ti, dc_i, yp_i in zip(bd["times"], bd["d_cash"], bd["y_prop"]):
        f_cur = f_cur * math.exp((0.05 - bd["q"]) * (ti - t_prev)) * (1.0 - yp_i) - dc_i
        t_prev = ti
    f_terminal = f_cur * math.exp((0.05 - bd["q"]) * (11.5 - t_prev))
    assert abs(f_terminal - 96.0) < 1e-8


def test_bbg_part2_mixture_fig15_1_and_2d_surface():
    from scipy.integrate import quad
    from bz_pricing.volatility import BloombergMixtureSurface, LognormalMixtureSlice

    sl_15 = LognormalMixtureSlice(weights=[0.5, 0.5], xis=[0.95, 1.05], sigmas_total=[0.40, 0.20], forward=1.0, T=1.0)
    area, _ = quad(lambda s: sl_15.pdf(s), 1e-6, 10.0)
    mean, _ = quad(lambda s: s * sl_15.pdf(s), 1e-6, 10.0)
    assert abs(area - 1.0) < 1e-4
    assert abs(mean - 1.0) < 1e-4

    p_vec = np.array([0.30, 0.45, 0.20])
    xi_vec = LognormalMixtureSlice._reconstruct_xis(p_vec, np.log([1.1, 1.15]))
    sl_def = LognormalMixtureSlice(p_vec, xi_vec, [0.42, 0.24, 0.18], forward=100.0, T=0.5, k_min_damp=90.0)
    for k_t in [65.0, 89.9, 90.0, 100.0, 130.0]:
        c_val = sl_def.option_price(k_t, r_cont=0.05, is_call=True)
        p_val = sl_def.option_price(k_t, r_cont=0.05, is_call=False)
        assert abs((c_val - p_val) - math.exp(-0.05 * 0.5) * (100.0 - k_t)) < 1e-10

    _, _, q_far = sl_def.damped_parameters(45.0)
    _, _, q_edge = sl_def.damped_parameters(89.99)
    assert q_far < q_edge <= sl_def.Q_default + 1e-12

    v_ext_l = sl_15.implied_vol(0.05)
    v_ext_r = sl_15.implied_vol(8.0)
    assert math.isfinite(v_ext_l) and v_ext_l > 0.0
    assert math.isfinite(v_ext_r) and v_ext_r > 0.0

    sl_1y = LognormalMixtureSlice(p_vec, xi_vec, [0.55, 0.34, 0.25], forward=105.0, T=1.0)
    bbg_surf = BloombergMixtureSurface({126: sl_def, 252: sl_1y})
    v_short = bbg_surf.get_vol(21, 100.0, forward=100.0)
    v_mid = bbg_surf.get_vol(189, 102.5, forward=102.5)
    v_long = bbg_surf.get_vol(756, 105.0, forward=105.0)
    assert 0.05 < v_short < 1.0 and 0.05 < v_mid < 1.0 and 0.05 < v_long < 1.0

    vs = VolatilitySurface.from_bloomberg_mixture(bbg_surf)
    assert abs(vs.get_vol(189, strike=102.5, forward=102.5) - v_mid) < 1e-12


def test_bbg_surface_real_b3_bova11_calibration():
    import pandas as pd
    from bz_pricing.core.calendars import get_calendar
    from bz_pricing.curves import CurveManager
    from bz_pricing.data.loaders import B3DataLoader
    from bz_pricing.volatility import (
        BloombergMixtureSurface,
        filter_surface_quotes_bbg,
        validate_slice_out_of_sample,
    )

    ref = datetime.date(2026, 10, 30)
    ldr = B3DataLoader(data_dir="dados/b3")
    cal = get_calendar("B3")
    pre = CurveManager(loader=ldr, allow_download=True).get_curve("PRE", ref)
    bova = ldr.load_equity_options(ref, underlying="BOVA11", only_priced=True, auto_download=False)
    spot = float(bova["spot_price"].iloc[0])

    b_sub = bova[(bova["TOTNEG"] >= 10) & (bova["ExrcPric"].between(0.85 * spot, 1.15 * spot))].copy()
    exps = ["2026-11-19", "2026-12-18"]
    fwds = {}
    rs = {}
    dus = {}
    for e_s in exps:
        du_i = cal.bizdays(ref, pd.to_datetime(e_s).date())
        r_i = math.log(1.0 + pre.get_rate(du_i))
        fwds[e_s] = spot * math.exp(r_i * (du_i / 252.0))
        rs[e_s] = r_i
        dus[e_s] = du_i

    cleaned = filter_surface_quotes_bbg(
        b_sub[b_sub["XprtnDt"].dt.strftime("%Y-%m-%d").isin(exps)],
        fwds,
        rs,
        du_by_expiry=dus,
        use_bid_ask=False,
    )
    assert not cleaned.empty

    slices_data = {}
    for e_s in exps:
        g = cleaned[cleaned["XprtnDt"] == e_s].sort_values("_k")
        slices_data[dus[e_s]] = {
            "strikes": g["_k"].to_numpy(),
            "market_prices": g["_price"].to_numpy(),
            "option_types": g["option_type"].to_numpy(),
            "forward": fwds[e_s],
            "r_cont": rs[e_s],
        }
    surf_bbg = BloombergMixtureSurface.fit(slices_data, n_components=4, allow_default=True)
    vs_price = VolatilitySurface.from_bloomberg_mixture(surf_bbg)
    arb_price = vs_price.check_no_arbitrage(
        forward_by_du={d: slices_data[d]["forward"] for d in surf_bbg.dus},
        r_cont_by_du={d: slices_data[d]["r_cont"] for d in surf_bbg.dus},
    )
    assert arb_price["passed"] is True

    surf_vega = BloombergMixtureSurface.fit(slices_data, n_components=4, allow_default=True, weighting="vega")
    vs_vega = VolatilitySurface.from_bloomberg_mixture(surf_vega)
    arb_vega = vs_vega.check_no_arbitrage(
        forward_by_du={d: slices_data[d]["forward"] for d in surf_vega.dus},
        r_cont_by_du={d: slices_data[d]["r_cont"] for d in surf_vega.dus},
    )
    assert arb_vega["passed"] is True
    for du_v in surf_vega.dus:
        assert surf_vega.slices[du_v].fit_diagnostics["rmse_vol_bps"] < 65.0

    oos = validate_slice_out_of_sample(
        strikes=slices_data[dus["2026-12-18"]]["strikes"],
        market_prices=slices_data[dus["2026-12-18"]]["market_prices"],
        option_types=slices_data[dus["2026-12-18"]]["option_types"],
        forward=fwds["2026-12-18"],
        r_cont=rs["2026-12-18"],
        du=dus["2026-12-18"],
        model="bloomberg_mixture",
        otm_only=True,
    )
    assert oos["rmse_vol_bps"] < 120.0
    assert math.isfinite(surf_bbg.get_vol(dus["2026-11-19"] + 10, spot))


def test_fx_option_offshore_dc_and_linear360_curves_e3():
    from bz_pricing.core.calendars import get_calendar
    from bz_pricing.core.curves import FlatCurve, SimpleCurve
    from bz_pricing.core.interpolators import Linear360Interpolator
    from bz_pricing.curves.offshore import bootstrap_ois_curve

    base_dt = datetime.date(2026, 10, 30)
    mat_dt = datetime.date(2027, 4, 30)
    cal = get_calendar("B3")
    du_n = cal.bizdays(base_dt, mat_dt)
    dc_n = (mat_dt - base_dt).days

    # 1. Curva Linear360 com t_sec_nodes (DC) distintos por prazo: _get_rate_for_du deve passar dc_n
    t_du = np.array([21.0, 126.0, 252.0], dtype=np.float64)
    t_dc = np.array([30.0, 182.0, 365.0], dtype=np.float64)
    y_cup = np.array([0.042, 0.050, 0.054], dtype=np.float64)
    doc_curve = SimpleCurve(t_du, y_cup, Linear360Interpolator(0, 0), t_sec_nodes=t_dc)

    opt = FXOption("DOLK27C", mat_dt, strike=5.65, option_type="CALL")
    pre_crv = FlatCurve(0.145)
    m_curve = opt.risk_metrics(base_dt, spot_ptax=5.60, pre_rate_or_curve=pre_crv, cupom_limpo_or_curve=doc_curve, vol=0.14)
    r_ddi_exact = doc_curve.get_rate(du_n, dc_n)
    q_expected_b3 = (252.0 / du_n) * math.log(1.0 + r_ddi_exact * (dc_n / 360.0))
    assert abs(m_curve["q_cont"] - q_expected_b3) < 1e-12

    # 2. Curva offshore contínua ACT/360 (SOFR OIS): DF_for = exp(-q_cont * T_du) == exp(-r_sofr * dc_n / 360)
    sofr_crv = bootstrap_ois_curve([30.0, 90.0, 180.0, 360.0], [0.043, 0.0435, 0.044, 0.045], convention="ACT/360")
    m_off = opt.risk_metrics(base_dt, spot_ptax=5.60, pre_rate_or_curve=pre_crv, cupom_limpo_or_curve=sofr_crv, vol=0.14)
    r_sofr = sofr_crv.get_rate(float(dc_n))
    df_for_expected = math.exp(-r_sofr * (dc_n / 360.0))
    df_for_actual = math.exp(-m_off["q_cont"] * (du_n / 252.0))
    assert abs(df_for_actual - df_for_expected) < 1e-12
    assert m_off["price"] > 0.0

    # Custom holidays também isolados em opção
    opt_cust = FXOption("DOLK27C", mat_dt, strike=5.65, holidays=["2026-11-16"])
    assert opt_cust.du(base_dt) == du_n - 1


def test_swaption_and_capfloor_offshore_conventions():
    from bz_pricing.core.curves import SimpleCurve
    from bz_pricing.core.interpolators import LogLinearDiscountInterpolator
    from bz_pricing.curves.offshore import bootstrap_ois_curve
    from bz_pricing.instruments import CapFloor, Swaption

    base_dt = datetime.date(2026, 10, 30)
    sofr = bootstrap_ois_curve([90.0, 180.0, 360.0, 720.0, 1080.0, 1800.0], [0.041, 0.0415, 0.042, 0.043, 0.044, 0.045], convention="ACT/360")

    # 1. Swaption offshore SOFR ACT/360 anual gerando agenda automaticamente via payment_freq_months=12
    swpt_pay = Swaption(
        "SWPT-SOFR-1Y2Y-P",
        expiry_date="2027-10-29",
        swap_maturity_date="2029-10-30",
        strike_rate=0.044,
        option_type="PAYER",
        notional=10_000_000.0,
        calendar_name="US",
        day_count="ACT/360",
        payment_freq_months=12,
    )
    swpt_rec = Swaption(
        "SWPT-SOFR-1Y2Y-R",
        expiry_date="2027-10-29",
        swap_maturity_date="2029-10-30",
        strike_rate=0.044,
        option_type="RECEIVER",
        notional=10_000_000.0,
        calendar_name="US",
        day_count="ACT/360",
        payment_freq_months=12,
    )
    m_p = swpt_pay.risk_metrics(base_dt, sofr, vol=0.22, model="black76")
    m_r = swpt_rec.risk_metrics(base_dt, sofr, vol=0.22, model="black76")
    # Paridade Payer - Receiver = Annuity * (F - K) * Notional
    parity = 10_000_000.0 * m_p["annuity"] * (m_p["forward_swap_rate"] - 0.044)
    assert abs((m_p["price"] - m_r["price"]) - parity) < 1e-6

    m_bac = swpt_pay.risk_metrics(base_dt, sofr, vol=0.0085, model="bachelier")
    assert m_bac["price"] > 0.0

    # 2. Cap vs Floor offshore ACT/360 vs paridade Cap - Floor = Swap
    fixings = [datetime.date(2027, 4, 30), datetime.date(2027, 10, 29)]
    payments = [datetime.date(2027, 10, 29), datetime.date(2028, 4, 28)]
    cap = CapFloor("CAP-SOFR", fixings, payments, strike_rate=0.043, option_type="CAP", notional=5_000_000.0, calendar_name="US", day_count="ACT/360")
    flr = CapFloor("FLR-SOFR", fixings, payments, strike_rate=0.043, option_type="FLOOR", notional=5_000_000.0, calendar_name="US", day_count="ACT/360")
    df_c = cap.cashflow_table(base_dt, sofr, vol=0.20)
    df_f = flr.cashflow_table(base_dt, sofr, vol=0.20)
    swap_from_caplets = (5_000_000.0 * df_c["tau"] * df_c["df_pay"] * (df_c["forward_rate"] - 0.043)).sum()
    assert abs((df_c["caplet_pv"].sum() - df_f["caplet_pv"].sum()) - swap_from_caplets) < 1e-6


def test_equity_option_discrete_and_prop_dividends():
    base_dt = datetime.date(2026, 10, 30)
    mat_dt = "2027-04-30"
    r_annual = 0.12
    spot = 40.0
    strike = 38.0
    vol = 0.30
    disc_divs = {"2026-12-15": 1.60, "2027-03-15": 1.20}
    prop_divs = [(0.18, 0.025)]

    # 1. Europeia com dividendos discretos em reais e proporcionais: gregas vs diferenças finitas
    for opt_t in ("CALL", "PUT"):
        opt_eu = EquityOption(
            "PETR4",
            mat_dt,
            strike=strike,
            option_type=opt_t,
            exercise="EUROPEAN",
            discrete_dividends=disc_divs,
            prop_dividends=prop_divs,
        )
        rm = opt_eu.risk_metrics(base_dt, spot, r_annual, vol)
        p0 = opt_eu.price(base_dt, spot, r_annual, vol)
        assert abs(rm["price"] - p0) < 1e-12

        eps_s = 1e-3
        p_up = opt_eu.price(base_dt, spot + eps_s, r_annual, vol)
        p_dn = opt_eu.price(base_dt, spot - eps_s, r_annual, vol)
        fd_delta = (p_up - p_dn) / (2.0 * eps_s)
        fd_gamma = (p_up - 2.0 * p0 + p_dn) / (eps_s ** 2)
        assert abs(rm["delta"] - fd_delta) < 1e-5
        assert abs(rm["gamma"] - fd_gamma) < 1e-4

        eps_v = 1e-4
        fd_vega = (opt_eu.price(base_dt, spot, r_annual, vol + eps_v) - opt_eu.price(base_dt, spot, r_annual, vol - eps_v)) / (2.0 * eps_v)
        assert abs(rm["vega"] - fd_vega) < 1e-4

        # Rho em relação à taxa contínua r_cont (incluindo efeito do PV dos dividendos)
        r_cont0 = math.log(1.0 + r_annual)
        eps_r = 1e-5
        r_up = math.exp(r_cont0 + eps_r) - 1.0
        r_dn = math.exp(r_cont0 - eps_r) - 1.0
        fd_rho = (opt_eu.price(base_dt, spot, r_up, vol) - opt_eu.price(base_dt, spot, r_dn, vol)) / (2.0 * eps_r)
        assert abs(rm["rho"] - fd_rho) < 1e-4

        iv_rt = opt_eu.implied_volatility(base_dt, p0, spot, r_annual)
        assert abs(iv_rt - vol) < 1e-6

    # 2. Americana com dividendo discreto elevado: exercício antecipado gera prêmio >= europeia
    big_div = {"2026-12-15": 3.50}
    eu_c = EquityOption("PETR4", mat_dt, strike=36.0, option_type="CALL", exercise="EUROPEAN", discrete_dividends=big_div)
    am_c = EquityOption("PETR4", mat_dt, strike=36.0, option_type="CALL", exercise="AMERICAN", discrete_dividends=big_div)
    eu_p = EquityOption("PETR4", mat_dt, strike=42.0, option_type="PUT", exercise="EUROPEAN", discrete_dividends=big_div, prop_dividends=prop_divs)
    am_p = EquityOption("PETR4", mat_dt, strike=42.0, option_type="PUT", exercise="AMERICAN", discrete_dividends=big_div, prop_dividends=prop_divs)

    p_eu_c = eu_c.price(base_dt, spot, r_annual, vol)
    p_am_c = am_c.price(base_dt, spot, r_annual, vol, crr_steps=400)
    assert p_am_c >= p_eu_c - 1e-6

    rm_am_c = am_c.risk_metrics(base_dt, spot, r_annual, vol, crr_steps=400)
    rm_am_p = am_p.risk_metrics(base_dt, spot, r_annual, vol, crr_steps=400)
    assert rm_am_p["price"] >= eu_p.price(base_dt, spot, r_annual, vol) - 1e-6
    assert 0.0 < rm_am_c["delta"] < 1.0
    assert -1.0 < rm_am_p["delta"] < 0.0
    for key in ("gamma", "vega", "theta", "rho"):
        assert np.isfinite(rm_am_c[key])
        assert np.isfinite(rm_am_p[key])

    # Checa delta da americana contra bump de spot em malha fina
    fd_am_delta = (
        am_c.price(base_dt, spot + 0.1, r_annual, vol, crr_steps=400)
        - am_c.price(base_dt, spot - 0.1, r_annual, vol, crr_steps=400)
    ) / 0.2
    assert abs(rm_am_c["delta"] - fd_am_delta) < 0.02

    iv_am = am_c.implied_volatility(base_dt, am_c.price(base_dt, spot, r_annual, vol), spot, r_annual)
    assert abs(iv_am - vol) < 1e-5


def test_exotic_options_greeks_and_portfolio():
    base_dt = datetime.date(2026, 10, 30)
    mat_dt = "2027-04-30"
    spot = 100.0
    r_annual = 0.11
    q_annual = 0.02
    vol = 0.28

    # Corrado-Su (assimetria e curtose não gaussianas)
    cs_opt = CorradoSuOption("IBOV-CS", mat_dt, strike=102.0, option_type="CALL")
    rm_cs = cs_opt.risk_metrics(base_dt, spot, r_annual, vol, kappa3=-0.35, kappa4=4.2, carry_or_div_yield=q_annual)
    p_cs = cs_opt.price(base_dt, spot, r_annual, vol, kappa3=-0.35, kappa4=4.2, carry_or_div_yield=q_annual)
    assert abs(rm_cs["price"] - p_cs) < 1e-12
    fd_cs_delta = (
        cs_opt.price(base_dt, spot + 1e-3, r_annual, vol, kappa3=-0.35, kappa4=4.2, carry_or_div_yield=q_annual)
        - cs_opt.price(base_dt, spot - 1e-3, r_annual, vol, kappa3=-0.35, kappa4=4.2, carry_or_div_yield=q_annual)
    ) / 2e-3
    fd_cs_vega = (
        cs_opt.price(base_dt, spot, r_annual, vol + 1e-4, kappa3=-0.35, kappa4=4.2, carry_or_div_yield=q_annual)
        - cs_opt.price(base_dt, spot, r_annual, vol - 1e-4, kappa3=-0.35, kappa4=4.2, carry_or_div_yield=q_annual)
    ) / 2e-4
    assert abs(rm_cs["delta"] - fd_cs_delta) < 1e-4
    assert abs(rm_cs["vega"] - fd_cs_vega) < 1e-3

    # Barreira Down-and-Out (contínua e discreta com monitoring_steps=52)
    bar_cont = BarrierOption("PETR-KO", mat_dt, strike=100.0, barrier=82.0, option_type="CALL", barrier_type="down-and-out", rebate=1.5)
    bar_disc = BarrierOption("PETR-KOD", mat_dt, strike=100.0, barrier=82.0, option_type="CALL", barrier_type="down-and-out", monitoring_steps=52)
    rm_bc = bar_cont.risk_metrics(base_dt, spot, r_annual, vol, carry_or_div_yield=q_annual)
    assert abs(rm_bc["price"] - bar_cont.price(base_dt, spot, r_annual, vol, carry_or_div_yield=q_annual)) < 1e-12
    fd_bc_delta = (
        bar_cont.price(base_dt, spot + 1e-3, r_annual, vol, carry_or_div_yield=q_annual)
        - bar_cont.price(base_dt, spot - 1e-3, r_annual, vol, carry_or_div_yield=q_annual)
    ) / 2e-3
    assert abs(rm_bc["delta"] - fd_bc_delta) < 1e-4
    rm_bd = bar_disc.risk_metrics(base_dt, spot, r_annual, vol, carry_or_div_yield=q_annual)
    assert rm_bd["barrier_eff"] < bar_disc.barrier
    assert np.isfinite(rm_bd["vega"]) and rm_bd["vega"] != 0.0

    # Asiática aritmética e geométrica
    as_art = AsianOption("DOL-ASA", mat_dt, strike=98.0, option_type="CALL", average_type="arithmetic")
    as_geo = AsianOption("DOL-ASG", mat_dt, strike=98.0, option_type="PUT", average_type="geometric")
    for as_opt in (as_art, as_geo):
        rm_a = as_opt.risk_metrics(base_dt, spot, r_annual, vol, carry_or_div_yield=q_annual)
        p_a = as_opt.price(base_dt, spot, r_annual, vol, carry_or_div_yield=q_annual)
        assert abs(rm_a["price"] - p_a) < 1e-12
        fd_a_delta = (
            as_opt.price(base_dt, spot + 1e-3, r_annual, vol, carry_or_div_yield=q_annual)
            - as_opt.price(base_dt, spot - 1e-3, r_annual, vol, carry_or_div_yield=q_annual)
        ) / 2e-3
        fd_a_vega = (
            as_opt.price(base_dt, spot, r_annual, vol + 1e-4, carry_or_div_yield=q_annual)
            - as_opt.price(base_dt, spot, r_annual, vol - 1e-4, carry_or_div_yield=q_annual)
        ) / 2e-4
        assert abs(rm_a["delta"] - fd_a_delta) < 1e-4
        assert abs(rm_a["vega"] - fd_a_vega) < 1e-3

    # Carteira consolidada: todas as posições exóticas e com dividendos devem retornar gregas finitas
    eq_div = EquityOption("PETR4-DIV", mat_dt, strike=100.0, option_type="PUT", exercise="AMERICAN", discrete_dividends={"2026-12-15": 2.0})
    eq_plain = EquityOption("VALE3", mat_dt, strike=95.0, option_type="CALL", exercise="EUROPEAN")
    df_port = price_options_portfolio(
        [
            {"instrument": eq_plain, "quantity": 100.0, "spot": spot, "pre_rate_or_curve": r_annual, "vol": vol},
            {"instrument": eq_div, "quantity": 50.0, "spot": spot, "pre_rate_or_curve": r_annual, "vol": vol},
            {"instrument": cs_opt, "quantity": 10.0, "spot": spot, "pre_rate_or_curve": r_annual, "vol": vol, "kappa3": -0.35, "kappa4": 4.2, "carry_or_div_yield": q_annual},
            {"instrument": bar_cont, "quantity": 20.0, "spot": spot, "pre_rate_or_curve": r_annual, "vol": vol, "carry_or_div_yield": q_annual},
            {"instrument": as_art, "quantity": 15.0, "spot": spot, "pre_rate_or_curve": r_annual, "vol": vol, "carry_or_div_yield": q_annual},
        ],
        base_dt,
    )
    assert len(df_port) == 5
    for col in ("price", "mtm", "delta", "gamma", "vega_1pct", "theta_du", "rho"):
        assert df_port[col].notna().all()
    assert abs(float(df_port.loc[df_port["ticker"] == "IBOV-CS", "price"].iloc[0]) - p_cs) < 1e-12
    assert abs(float(df_port.loc[df_port["ticker"] == "PETR-KO", "delta"].iloc[0]) - rm_bc["delta"]) < 1e-12


def test_idi_di1_normal_bachelier_and_revalidation_2d():
    import os
    import pandas as pd
    from bz_pricing.core.calendars import get_calendar
    from bz_pricing.curves import CurveManager
    from bz_pricing.data.loaders import B3DataLoader
    from bz_pricing.volatility import infer_forward_from_put_call_parity, validate_slice_out_of_sample

    # 1. IDIOption e DI1Option em Black-76 e Bachelier Normal
    base_dt = datetime.date(2026, 10, 30)
    idi_spot = 539229.27
    fwd_jan = 550844.05
    r_pre = 0.1364
    sigma_n_idi = 384.28
    idi_c = IDIOption("IDIF27C550600", "2027-01-04", 550600.0, "CALL")
    rm_in = idi_c.risk_metrics(base_dt, idi_spot, r_pre, sigma_n_idi, model="bachelier", idi_forward=fwd_jan)
    iv_in_rt = idi_c.implied_volatility(base_dt, rm_in["price"], idi_spot, r_pre, model="bachelier", idi_forward=fwd_jan)
    assert rm_in["model"] == "bachelier"
    assert abs(rm_in["idi_forward"] - fwd_jan) < 1e-9
    assert abs(iv_in_rt - sigma_n_idi) < 1e-7

    # Verifica delta normal de IDIOption por diferença finita (passo pequeno h=0.01)
    h_fwd = 0.01
    p_up = idi_c.price(base_dt, idi_spot, r_pre, sigma_n_idi, model="bachelier", idi_forward=fwd_jan + h_fwd)
    p_dn = idi_c.price(base_dt, idi_spot, r_pre, sigma_n_idi, model="bachelier", idi_forward=fwd_jan - h_fwd)
    assert abs(rm_in["delta"] - (p_up - p_dn) / (2.0 * h_fwd)) < 1e-6

    # DI1Option em modelo Normal de Bachelier
    di1_c = DI1Option("DI1C14", "2027-02-01", "2028-01-03", strike_rate=0.145, option_type="CALL")
    rm_dn = di1_c.risk_metrics(base_dt, pu_c=97100.0, pu_l=85800.0, vol=0.012, model="bachelier")
    iv_dn_rt = di1_c.implied_volatility(base_dt, rm_dn["price"], pu_c=97100.0, pu_l=85800.0, model="bachelier")
    assert rm_dn["model"] == "bachelier" and rm_dn["price"] > 0.0
    assert abs(iv_dn_rt - 0.012) < 1e-8

    # 2. Revalidação 2D real de IDI (2026-10-30): Black-76 e Normal com ancoragem ATM
    if not (os.path.exists("dados/b3/BVBG_IN_20261030.csv") and os.path.exists("dados/b3/BVBG_SPRD_20261030.csv")):
        return

    loader = B3DataLoader(data_dir="dados/b3")
    curva_pre = CurveManager(loader=loader, allow_download=True).get_curve("PRE", base_dt)
    idi_opts = loader.load_options(base_dt, asst="IDI", only_priced=True, auto_download=False)
    cal = get_calendar()

    du_jan = cal.bizdays(base_dt, datetime.date(2027, 1, 4))
    r_jan = math.log(1.0 + curva_pre.get_rate(du_jan))
    df_jan = math.exp(-r_jan * (du_jan / 252.0))
    fwd_jan_mkt = infer_forward_from_put_call_parity(
        idi_opts[idi_opts["XprtnDt"] == "2027-01-04"], discount_factor=df_jan
    )
    spot_mkt = fwd_jan_mkt * df_jan

    fwds, rs, strikes_by_du = {}, {}, {}
    m_band = np.linspace(0.996, 1.003, 15)
    for vt in ("lognormal", "normal"):
        slices_svi, slices_sabr = {}, {}
        for exp_str in ("2027-01-04", "2027-04-01"):
            sub = idi_opts[idi_opts["XprtnDt"] == exp_str].sort_values("ExrcPric").reset_index(drop=True)
            du_i = cal.bizdays(base_dt, pd.to_datetime(exp_str).date())
            r_i = math.log(1.0 + curva_pre.get_rate(du_i))
            fwd_i = spot_mkt * math.exp(r_i * (du_i / 252.0))
            fwds[du_i] = fwd_i
            rs[du_i] = r_i
            strikes_by_du[du_i] = np.unique(sub["ExrcPric"].values)

            oos_svi = validate_slice_out_of_sample(
                sub["ExrcPric"].values,
                sub["RefPric"].values,
                sub["option_type"].values,
                fwd_i,
                r_i,
                du_i,
                model="svi",
                otm_only=True,
                vega_weighted=True,
                split_mode="interpolation",
                vol_type=vt,
                anchor_atm=True,
            )
            oos_sabr = validate_slice_out_of_sample(
                sub["ExrcPric"].values,
                sub["RefPric"].values,
                sub["option_type"].values,
                fwd_i,
                r_i,
                du_i,
                model="sabr",
                otm_only=True,
                vega_weighted=True,
                split_mode="interpolation",
                vol_type=vt,
                anchor_atm=True,
            )
            assert oos_svi["rmse_vol_bps"] < 2.0
            assert oos_sabr["rmse_vol_bps"] < 2.5
            assert abs(float(oos_svi["slice"].implied_vol(0.0)) - float(oos_svi["slice"].atm_vol)) < 1e-12
            assert abs(float(oos_sabr["slice"].implied_vol(fwd_i)) - float(oos_sabr["slice"].atm_vol)) < 1e-12
            slices_svi[du_i] = oos_svi["slice"]
            slices_sabr[du_i] = oos_sabr["slice"]

        surf_sabr = VolatilitySurface.from_sabr_slices(slices_sabr, slice_forwards=fwds)
        assert surf_sabr.vol_type == vt
        arb_sabr = surf_sabr.check_no_arbitrage(fwds, rs, strikes=strikes_by_du, moneyness_grid=m_band)
        assert arb_sabr["passed"] is True
        if vt == "lognormal":
            surf_svi = VolatilitySurface.from_svi_slices(slices_svi, slice_forwards=fwds)
            arb_svi = surf_svi.check_no_arbitrage(fwds, rs, strikes=strikes_by_du, moneyness_grid=m_band)
            assert arb_svi["passed"] is True


def test_normal_vol_anchoring_and_surface_persistence(tmp_path):
    from bz_pricing.volatility import BloombergMixtureSurface, LognormalMixtureSlice

    # 1. Ancoragem ATM exata em SABR Normal e SVI Normal
    fwd = 550000.0
    t_val = 42.0 / 252.0
    ks = np.linspace(548000.0, 552000.0, 9)
    true_n = SABRSlice(alpha=400.0, beta=0.0, rho=-0.25, nu=1.8, F=fwd, T=t_val, vol_type="normal")
    vols_n = true_n.implied_vol(ks)
    atm_forced = 415.0

    fit_sabr_n = SABRSlice.fit(fwd, ks, vols_n, t_val, beta=0.0, vol_type="normal", anchor_atm=True, atm_vol=atm_forced)
    assert abs(float(fit_sabr_n.implied_vol(fwd)) - atm_forced) < 1e-12

    fit_svi_n = SVISlice.fit(np.log(ks / fwd), vols_n, t_val, vol_type="normal", anchor_atm=True, atm_vol=atm_forced)
    assert abs(float(fit_svi_n.implied_vol(0.0)) - atm_forced) < 1e-12

    # 2. Persistência completa (to_dict/from_dict, to_json/from_json, save/load)
    # 2a. VolatilitySurface SABR Normal
    s_n2 = SABRSlice.fit(fwd * 1.02, ks * 1.02, vols_n * 1.2, t_val * 2.0, beta=0.0, vol_type="normal", anchor_atm=True)
    surf_n = VolatilitySurface.from_sabr_slices({42: fit_sabr_n, 84: s_n2}, slice_forwards={42: fwd, 84: fwd * 1.02})
    path_sabr = tmp_path / "surf_sabr_normal.json"
    surf_n.save(path_sabr)
    surf_n_ld = VolatilitySurface.load(path_sabr)
    assert surf_n_ld.vol_type == "normal"
    for d_q, k_q in ((42, 550000.0), (63, 555000.0), (84, 561000.0)):
        assert abs(surf_n.get_vol(d_q, strike=k_q) - surf_n_ld.get_vol(d_q, strike=k_q)) < 1e-12

    # 2b. VolatilitySurface SVI Lognormal
    svi_l = SVISlice(0.01, 0.04, -0.35, 0.01, 0.12, 0.25, vol_type="lognormal", atm_vol=0.22)
    surf_svi = VolatilitySurface.from_svi_slices({63: svi_l}, slice_forwards={63: 100.0})
    surf_svi_rt = VolatilitySurface.from_json(surf_svi.to_json())
    assert abs(surf_svi.get_vol(63, strike=95.0, forward=100.0) - surf_svi_rt.get_vol(63, strike=95.0, forward=100.0)) < 1e-12
    assert abs(surf_svi_rt._slices[63].atm_vol - 0.22) < 1e-12

    # 2c. BloombergMixtureSurface & VolatilitySurface(bloomberg_mixture)
    sl_mix1 = LognormalMixtureSlice([0.60, 0.35], [0.88, 1.24], [0.22, 0.14], forward=100.0, T=0.25)
    sl_mix2 = LognormalMixtureSlice([0.55, 0.38], [0.85, 1.32], [0.30, 0.20], forward=104.0, T=0.50)
    bbg_surf = BloombergMixtureSurface({63: sl_mix1, 126: sl_mix2})
    path_bbg = tmp_path / "bbg_surf.json"
    bbg_surf.save(path_bbg)
    bbg_ld = BloombergMixtureSurface.load(path_bbg)
    assert abs(bbg_surf.get_vol(90, strike=95.0, forward=102.0) - bbg_ld.get_vol(90, strike=95.0, forward=102.0)) < 1e-12
    surf_bbg_w = VolatilitySurface.from_bloomberg_mixture(bbg_surf)
    surf_bbg_rt = VolatilitySurface.from_json(surf_bbg_w.to_json())
    assert abs(surf_bbg_w.get_vol(90, strike=95.0, forward=102.0) - surf_bbg_rt.get_vol(90, strike=95.0, forward=102.0)) < 1e-12

    # 2d. Grid strike & grid delta
    surf_grid = VolatilitySurface.from_strike_grid([63, 126], [90.0, 100.0, 110.0], [[0.30, 0.25, 0.28], [0.31, 0.26, 0.29]])
    surf_grid_rt = VolatilitySurface.from_dict(surf_grid.to_dict())
    assert abs(surf_grid.get_vol(90, strike=95.0) - surf_grid_rt.get_vol(90, strike=95.0)) < 1e-12


def test_cython_mandatory_no_import_fallbacks():
    import pathlib
    import re
    from bz_pricing.volatility.surface import exp_smile_interp, monotone_cubic_interp

    src_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "bz_pricing"
    offenders = []
    for py_file in sorted(src_root.rglob("*.py")):
        txt = py_file.read_text(encoding="utf-8")
        if re.search(r"try:\s*\n(?:.*\n)*?.*from\s+bz_pricing\.core\..*\n(?:.*\n)*?except\s+ImportError", txt):
            offenders.append(str(py_file))
    assert not offenders, f"Encontrados try/except ImportError proibidos para Cython: {offenders}"

    val_mc = monotone_cubic_interp([1.0, 2.0, 3.0], [0.20, 0.22, 0.25], 1.5)
    val_exp = exp_smile_interp([1.0, 3.0], [0.20, 0.30], 2.0)
    assert 0.20 < val_mc < 0.22
    assert abs(val_exp - math.sqrt(0.20 * 0.30)) < 1e-12


def test_bbg_cython_kernels_wings_jacobian_and_american_crr():
    from bz_pricing.core.options_core import (
        bloomberg_mixture_prices_and_jac,
        bloomberg_mixture_prices_vec,
        crr_american_discrete_div_price_greeks,
        implied_vol_american_crr,
        longest_monotonic_subsequence,
    )
    from bz_pricing.volatility import LognormalMixtureSlice

    # 1. Subsequência monótona mais longa em Cython (crescente e decrescente)
    vals_dec = np.array([15.0, 12.0, 14.0, 9.5, 9.5, 4.0, 1.0], dtype=np.float64)
    idx_dec = longest_monotonic_subsequence(vals_dec, True)
    kept_dec = vals_dec[idx_dec]
    assert len(kept_dec) == 6
    assert np.all(np.diff(kept_dec) <= 1e-12)

    vals_inc = np.array([1.0, 3.0, 2.0, 5.0, 7.0], dtype=np.float64)
    idx_inc = longest_monotonic_subsequence(vals_inc, False)
    kept_inc = vals_inc[idx_inc]
    assert len(kept_inc) == 4
    assert np.all(np.diff(kept_inc) >= -1e-12)

    # 2. Asas extremas da Fig. 15.1 sem cliff numérico de fallback em BSM
    sl_15 = LognormalMixtureSlice(weights=[0.5, 0.5], xis=[0.95, 1.05], sigmas_total=[0.40, 0.20], forward=1.0, T=1.0)
    wing_ks = np.array([0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0], dtype=np.float64)
    ivs_raw = np.asarray(sl_15.implied_vol(wing_ks, r_cont=0.0, use_wing_extrap=False), dtype=np.float64)
    assert np.all(np.isfinite(ivs_raw))
    assert ivs_raw[0] > 0.39 and ivs_raw[1] > 0.38
    assert ivs_raw[-1] > 0.37 and ivs_raw[4] < 0.31
    assert np.all(np.diff(ivs_raw[:5]) < 0.0)
    assert np.all(np.diff(ivs_raw[4:]) > 0.0)

    # 3. Jacobiano analítico exato vs diferenças finitas centrais
    ks_j = np.array([75.0, 90.0, 100.0, 115.0, 130.0], dtype=np.float64)
    ic_j = np.array([0, 0, 1, 1, 1], dtype=np.uint8)
    for allow_def, fixed_q, vec_base in [
        (True, 0.01, np.array([0.4, -0.3, 0.12, 0.14, 0.34, 0.25, 0.19, 0.035], dtype=np.float64)),
        (False, 0.0, np.array([0.3, -0.2, 0.10, 0.12, 0.32, 0.24, 0.18], dtype=np.float64)),
    ]:
        px, jac = bloomberg_mixture_prices_and_jac(ks_j, ic_j, 100.0, 0.05, 0.5, vec_base, 3, allow_def, fixed_q)
        eps = 1e-6
        jac_fd = np.zeros_like(jac)
        for p_idx in range(len(vec_base)):
            vp = vec_base.copy()
            vm = vec_base.copy()
            vp[p_idx] += eps
            vm[p_idx] -= eps
            pp, _ = bloomberg_mixture_prices_and_jac(ks_j, ic_j, 100.0, 0.05, 0.5, vp, 3, allow_def, fixed_q)
            pm, _ = bloomberg_mixture_prices_and_jac(ks_j, ic_j, 100.0, 0.05, 0.5, vm, 3, allow_def, fixed_q)
            jac_fd[:, p_idx] = (pp - pm) / (2.0 * eps)
        assert np.max(np.abs(jac - jac_fd)) < 1e-6
        assert np.all(np.isfinite(px)) and np.all(px > 0.0)

    px_vec = bloomberg_mixture_prices_vec(ks_j, ic_j, 100.0, 0.05, 0.5, sl_15.p, sl_15.xi, sl_15.Sigma, sl_15.Q_default, 1.0)
    assert np.all(np.isfinite(px_vec))

    # 4. implied_vol_american_crr com dividendos discretos em dinheiro e proporcionais
    dt_d = np.array([0.20, 0.55], dtype=np.float64)
    da_d = np.array([1.40, 1.10], dtype=np.float64)
    pt_d = np.array([0.35], dtype=np.float64)
    py_d = np.array([0.015], dtype=np.float64)
    for is_c, strike_a in [(True, 96.0), (False, 104.0)]:
        p_am = crr_american_discrete_div_price_greeks(
            100.0, strike_a, 0.10, 0.01, 0.80, 0.30, is_c, 150, dt_d, da_d, pt_d, py_d
        )[0]
        iv_am = implied_vol_american_crr(
            p_am, 100.0, strike_a, 0.10, 0.01, 0.80, is_c, 150, dt_d, da_d, pt_d, py_d
        )
        assert abs(iv_am - 0.30) < 1e-6
    iv_floor = implied_vol_american_crr(3.5, 100.0, 95.0, 0.10, 0.01, 0.80, True, 150, dt_d, da_d, pt_d, py_d)
    assert not math.isfinite(iv_floor)


def test_bbg_part1_filters_reconciliation_kalman_and_blended():
    import pandas as pd
    from bz_pricing.core.options_core import black76_price_greeks, crr_american_discrete_div_price_greeks
    from bz_pricing.volatility import (
        DividendCurveBBG,
        KalmanDividendFilter,
        deamericanize_and_infer_forward_bbg,
        filter_option_quotes_bbg,
        reconcile_multi_maturity_dividends_bbg,
    )

    # 1. filter_option_quotes_bbg agrupado por vencimento
    rows = []
    for exp_s, mult in [("2026-12-18", 1.0), ("2027-03-19", 0.98)]:
        rows.extend([
            {"XprtnDt": exp_s, "strike": 90.0, "option_type": "call", "bid": 13.97 * mult, "ask": 14.03 * mult, "mid": 14.00 * mult},
            {"XprtnDt": exp_s, "strike": 95.0, "option_type": "call", "bid": 10.07 * mult, "ask": 10.13 * mult, "mid": 10.10 * mult},
            {"XprtnDt": exp_s, "strike": 100.0, "option_type": "call", "bid": 12.97 * mult, "ask": 13.03 * mult, "mid": 13.00 * mult},  # quebra monotonicidade
            {"XprtnDt": exp_s, "strike": 105.0, "option_type": "call", "bid": 4.47 * mult, "ask": 4.53 * mult, "mid": 4.50 * mult},
            {"XprtnDt": exp_s, "strike": 100.0, "option_type": "put", "bid": 4.17 * mult, "ask": 4.23 * mult, "mid": 4.20 * mult},
            {"XprtnDt": exp_s, "strike": 105.0, "option_type": "put", "bid": 5.00 * mult, "ask": 5.50 * mult, "mid": 5.25 * mult},  # spread > 0.0009 * spot
        ])
    flt_m = filter_option_quotes_bbg(
        pd.DataFrame(rows),
        spot=100.0,
        maturity_years={"2026-12-18": 0.25, "2027-03-19": 0.50},
    )
    for exp_s in ("2026-12-18", "2027-03-19"):
        g = flt_m[flt_m["XprtnDt"] == exp_s]
        assert len(g[g["option_type"] == "call"]) == 3
        assert len(g[g["option_type"] == "put"]) == 1

    # 2. deamericanize_and_infer_forward_bbg com dividendos discretos, deamericanize_all e fallback bid/ask cruzado
    spot, r_c, q_c, T_am, sig_am = 100.0, 0.10, 0.015, 0.60, 0.26
    dt_d = np.array([0.25], dtype=np.float64)
    da_d = np.array([2.50], dtype=np.float64)
    pt_d = np.array([0.40], dtype=np.float64)
    py_d = np.array([0.015], dtype=np.float64)
    ks_am = np.array([88.0, 94.0, 100.0, 106.0, 112.0], dtype=np.float64)
    c_am = np.array([crr_american_discrete_div_price_greeks(spot, k, r_c, q_c, T_am, sig_am, True, 120, dt_d, da_d, pt_d, py_d)[0] for k in ks_am])
    p_am = np.array([crr_american_discrete_div_price_greeks(spot, k, r_c, q_c, T_am, sig_am, False, 120, dt_d, da_d, pt_d, py_d)[0] for k in ks_am])
    fwd_true = (spot - 2.50 * math.exp(-r_c * 0.25)) * (1.0 - 0.015) * math.exp((r_c - q_c) * T_am)
    out_da = deamericanize_and_infer_forward_bbg(
        ks_am, c_am, p_am,
        spot=spot, r_cont=r_c, T=T_am, exercise="AMERICAN",
        call_bids=c_am - 0.05, call_asks=c_am + 0.05,
        put_bids=p_am - 0.05, put_asks=p_am + 0.05,
        discrete_dividends=[(0.25, 2.50)],
        prop_dividends=[(0.40, 0.015)],
        crr_steps=120,
        deamericanize_all=True,
    )
    assert abs(out_da["forward"] - fwd_true) < 0.15
    assert np.all(np.isfinite(out_da["implied_vols_call"]))
    assert out_da["forward_bid"] <= out_da["forward_ask"]
    df_r = math.exp(-r_c * T_am)
    parity_da = (out_da["european_calls"] - out_da["european_puts"]) - df_r * (out_da["forward"] - ks_am)
    assert np.max(np.abs(parity_da[2:3])) < 0.10

    # Intervalos bid/ask cruzados entre os dois strikes centrais -> fallback para strike mais próximo
    ks_cr = np.array([98.0, 102.0], dtype=np.float64)
    c_m = np.array([black76_price_greeks(100.0, k, 0.0, 0.5, 0.25, True)[0] for k in ks_cr])
    p_m = np.array([black76_price_greeks(100.0, k, 0.0, 0.5, 0.25, False)[0] for k in ks_cr])
    out_cr = deamericanize_and_infer_forward_bbg(
        ks_cr, c_m, p_m, spot=100.0, r_cont=0.0, T=0.5,
        call_bids=[c_m[0] + 0.30, c_m[1] - 0.25],
        call_asks=[c_m[0] + 0.35, c_m[1] - 0.20],
        put_bids=[p_m[0] + 0.01, p_m[1] + 0.20],
        put_asks=[p_m[0] + 0.02, p_m[1] + 0.25],
    )
    assert out_cr["forward_bid"] <= out_cr["forward_ask"] + 1e-12

    # 3. Reconciliação de dividendos em múltiplos vencimentos
    rec_ok = reconcile_multi_maturity_dividends_bbg([0.25, 0.50], [1.10, 1.15], [1.30, 1.25])
    assert rec_ok["consistent"] is True and abs(rec_ok["dividend"] - 1.20) < 1e-12
    rec_bad = reconcile_multi_maturity_dividends_bbg([0.25, 0.50], [1.00, 1.40], [1.20, 1.60])
    assert rec_bad["consistent"] is False and abs(rec_bad["dividend"] - 1.10) < 1e-12
    assert abs(rec_bad["borrow_discrepancies"][0.50] - 0.40) < 1e-12

    # 4. KalmanDividendFilter com snapshot ausente e ex-dividendo de índice
    kf = KalmanDividendFilter(alpha_over_sigma=1.0)
    kf.update(0.030, d_bid=0.025, d_ask=0.035)
    v_before = kf.variance
    miss = kf.update(None, dt_steps=2.0)
    assert miss["missing"] is True and miss["gain"] == 0.0 and abs(miss["dividend"] - 0.030) < 1e-12
    assert miss["variance"] > v_before
    kf.apply_ex_dividend_index(spot=5000.0, ex_dividend_points=25.0, maturity_years=0.5)
    assert kf.state < 0.030

    # 5. DividendCurveBBG (cum_seasonality, ex-dates, short-chain, spot implícito e blended weights)
    dc = DividendCurveBBG(is_index=False)
    dc.seasonality_weights = np.array([2.0, 1.0, 1.0, 0.5, 0.5, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0]) / 12.0
    t_mid_v = dc.interpolate_cash_dividend(0.0, 0.0, 1.0, 12.0, 0.5)
    assert abs(t_mid_v - 12.0 * (6.0 / 12.0)) < 1e-12
    alloc = dc.allocate_to_ex_dates(0.0, 0.0, 1.0, 3.0, ex_times=[0.25, 0.75], r_cont=0.10)
    assert abs(alloc[0] * math.exp(-0.10 * 0.25) + alloc[1] * math.exp(-0.10 * 0.75) - 3.0) < 1e-10
    y_ext = DividendCurveBBG.extrapolate_long_forward_yield(
        [0.25, 0.50], [101.0, 102.0], spot=100.0, r_cont=0.10, t_eval=2.0, augment_div_pv=2.0
    )
    assert y_ext > 0.0
    cash_10y = DividendCurveBBG.extrapolate_equity_cash_dividends(
        [0.5, 1.5], [102.0, 104.5], spot=100.0, r_cont=0.10, target_div_times=[0.5, 2.0, 5.0, 10.0, 12.0]
    )
    assert len(cash_10y["discrete_cash_dividends"]) == 4 and cash_10y["cutoff_years"] == 10.0
    assert abs(DividendCurveBBG.option_implied_spot(102.0, 0.10, 0.5, forecast_div_pv=1.5) - (102.0 * math.exp(-0.05) + 1.5)) < 1e-12

    bd_w = DividendCurveBBG.decompose_blended_dividends(
        [0.5, 1.5, 3.0, 11.0], spot_f0=100.0, fwd_T=96.0, r_cont=0.05, T=11.5, div_weights=[1.0, 1.5, 1.2, 0.8]
    )
    assert abs(bd_w["y_prop"][0]) < 1e-12 and abs(bd_w["y_prop"][-1]) < 1e-12 and abs(bd_w["d_cash"][-1]) < 1e-12
    f_cur, t_prev = 100.0, 0.0
    for ti, dci, ypi in zip(bd_w["times"], bd_w["d_cash"], bd_w["y_prop"]):
        f_cur = f_cur * math.exp((0.05 - bd_w["q"]) * (ti - t_prev)) * (1.0 - ypi) - dci
        t_prev = ti
    assert abs(f_cur * math.exp((0.05 - bd_w["q"]) * (11.5 - t_prev)) - 96.0) < 1e-8


def test_bbg_part2_surface_calibration_extrap_reference_and_serialization():
    import pandas as pd
    from bz_pricing.core.options_core import black76_price_greeks, crr_american_discrete_div_price_greeks
    from bz_pricing.volatility import (
        BloombergMixtureSurface,
        LognormalMixtureSlice,
        SVISlice,
        VolatilitySurface,
        filter_surface_quotes_bbg,
    )

    # 1. filter_surface_quotes_bbg: min_options_per_maturity=3, age_days ignorado em bid/ask e americana
    ks_f = np.array([85.0, 92.0, 100.0, 108.0, 115.0], dtype=np.float64)
    spot_f, r_f, t_f = 100.0, 0.08, 0.5
    fwd_f = spot_f * math.exp(r_f * t_f)
    rows_s = []
    for k_i in ks_f:
        ot_i = "CALL" if k_i >= fwd_f else "PUT"
        p_am = crr_american_discrete_div_price_greeks(spot_f, k_i, r_f, 0.01, t_f, 0.28, (ot_i == "CALL"), 100, None, None, None, None)[0]
        rows_s.append({
            "XprtnDt": "2027-04-30",
            "ExrcPric": k_i,
            "option_type": ot_i,
            "bid": max(0.002, p_am - 0.05),
            "ask": p_am + 0.05,
            "age_days": 10,  # ignorado quando bid/ask ativo
        })
    # Vencimento com exatamente 3 opcoes válidas e um outlier de spread
    for k_i, sp_i in [(92.0, 0.04), (100.0, 0.04), (108.0, 0.04), (112.0, 0.04)]:
        ot_i = "CALL" if k_i >= fwd_f else "PUT"
        p_eu = black76_price_greeks(fwd_f, k_i, r_f, 0.25, 0.26, ot_i == "CALL")[0]
        rows_s.append({
            "XprtnDt": "2027-01-29",
            "ExrcPric": k_i,
            "option_type": ot_i,
            "bid": max(0.002, p_eu - 0.5 * sp_i),
            "ask": p_eu + 0.5 * sp_i,
            "age_days": 1,
        })
    df_surf_in = pd.DataFrame(rows_s)
    flt_am = filter_surface_quotes_bbg(
        df_surf_in,
        forward_by_expiry={"2027-01-29": fwd_f, "2027-04-30": fwd_f},
        r_cont_by_expiry={"2027-01-29": r_f, "2027-04-30": r_f},
        t_by_expiry={"2027-01-29": 0.25, "2027-04-30": t_f},
        use_bid_ask=True,
        min_options_per_maturity=3,
        exercise="AMERICAN",
        spot=spot_f,
        crr_steps=100,
    )
    assert set(flt_am["XprtnDt"].unique()) == {"2027-01-29", "2027-04-30"}
    assert np.all(np.isfinite(flt_am["_iv"]))

    # 2. LognormalMixtureSlice.fit: weighting ('price','vega','bid_ask'), preset e distressed Q
    t_s = 0.5
    w_dist = np.array([0.95 * 0.45, 0.95 * 0.55])
    xi_dist = LognormalMixtureSlice._reconstruct_xis(w_dist, np.array([math.log(1.15)]))
    sl_true = LognormalMixtureSlice(w_dist, xi_dist, np.array([0.35 * math.sqrt(t_s), 0.22 * math.sqrt(t_s)]), forward=100.0, T=t_s)
    ks_fit = np.linspace(55.0, 145.0, 19)
    ot_fit = ["PUT" if k < 100.0 else "CALL" for k in ks_fit]
    ps_fit = np.array([sl_true.option_price(k, r_cont=0.05, is_call=(ot == "CALL"), apply_hazard_damping=False) for k, ot in zip(ks_fit, ot_fit)])
    spreads_fit = np.full(len(ks_fit), 0.15)

    for wm in ("price", "vega", "bid_ask"):
        sl_f = LognormalMixtureSlice.fit(
            ks_fit, ps_fit, ot_fit, 100.0, 0.05, t_s, n_components=4, allow_default=True,
            weighting=wm, bid_ask_spreads=spreads_fit if wm == "bid_ask" else None,
        )
        assert sl_f.fit_diagnostics["weighting"] == wm
        assert sl_f.fit_diagnostics["rmse_vol_bps"] < 20.0
        assert abs(sl_f.Q_default - 0.05) < 0.015
        if wm == "bid_ask":
            assert sl_f.fit_diagnostics["bid_ask_hit_rate"] == 1.0

    # Zero butterfly sem hazard damping vs damping explícito opt-in
    vs_dist = VolatilitySurface.from_bloomberg_mixture(BloombergMixtureSurface({126: sl_f}))
    arb_dist = vs_dist.check_no_arbitrage({126: 100.0}, {126: 0.05})
    assert arb_dist["butterfly_violations"] == 0
    _, _, q_far = sl_f.damped_parameters(40.0)
    assert q_far < sl_f.Q_default

    sl_idx = LognormalMixtureSlice.fit(ks_fit, ps_fit, ot_fit, 100.0, 0.05, t_s, preset="index_deep_otm")
    assert sl_idx.N == 5 and sl_idx.Q_default < 1e-12

    # 3. Extrapolação de curto e longo prazo, 2-pillar Heston, referência, bid/ask, basis 365 e JSON
    sl_1 = LognormalMixtureSlice([0.55 * 0.99, 0.45 * 0.99], LognormalMixtureSlice._reconstruct_xis(np.array([0.55 * 0.99, 0.45 * 0.99]), [math.log(1.12)]), [0.22 * math.sqrt(0.25), 0.15 * math.sqrt(0.25)], forward=100.0, T=0.25)
    sl_2 = LognormalMixtureSlice([0.55 * 0.98, 0.45 * 0.98], LognormalMixtureSlice._reconstruct_xis(np.array([0.55 * 0.98, 0.45 * 0.98]), [math.log(1.18)]), [0.25 * math.sqrt(0.50), 0.17 * math.sqrt(0.50)], forward=102.0, T=0.50)

    surf_2 = BloombergMixtureSurface(
        {63: sl_1, 126: sl_2},
        short_atm_anchor={15: 0.34},
        spread_grid={63: (np.array([80.0, 100.0, 120.0]), np.array([0.012, 0.008, 0.012])), 126: (np.array([80.0, 100.0, 120.0]), np.array([0.010, 0.006, 0.010]))},
    )
    assert surf_2.heston_params[1] >= 0.20 * min(sl_1.implied_vol(100.0) ** 2, sl_2.implied_vol(102.0) ** 2) - 1e-12
    assert abs(surf_2.get_vol(15, 100.0, forward=100.0) - 0.34) < 1e-6
    assert abs(surf_2.get_vol(15, 89.99, forward=100.0) - surf_2.get_vol(15, 90.01, forward=100.0)) < 0.01

    vb, vm, va = surf_2.get_bid_ask_vol(90, 95.0)
    assert vb < vm < va

    # Estágio 2/3 de longo prazo atingindo target_atm estressado
    surf_stress = BloombergMixtureSurface({126: sl_2}, heston_params=(0.65, 0.65, 1.0))
    v_long_stress = surf_stress.get_vol(1260, sl_2.F, forward=sl_2.F)
    assert abs(v_long_stress - math.sqrt(0.65)) < 1e-3

    # Superfície de referência (ATM ausente, asas estreitas e prazo longo de 10 anos) + basis=365
    ref_s = VolatilitySurface.from_svi_slices({
        180: SVISlice(0.03, 0.05, -0.35, 0.0, 0.15, 180.0 / 365.0),
        365: SVISlice(0.035, 0.05, -0.35, 0.0, 0.16, 1.0),
        3650: SVISlice(0.04, 0.05, -0.30, 0.0, 0.20, 10.0),
    })
    ks_no_atm = np.array([75.0, 82.0, 118.0, 125.0])
    ps_no_atm = [black76_price_greeks(100.0, k, 0.05, 180.0 / 365.0, ref_s.get_vol(180, strike=k, forward=100.0) + 0.02, k >= 100.0)[0] for k in ks_no_atm]
    ks_narrow = np.array([92.0, 96.0, 100.0, 104.0, 108.0])
    ps_narrow = [black76_price_greeks(103.0, k, 0.05, 1.0, ref_s.get_vol(365, strike=k, forward=103.0) + 0.02, k >= 103.0)[0] for k in ks_narrow]
    surf_ref_fit = BloombergMixtureSurface.fit(
        {
            180: {"strikes": ks_no_atm, "market_prices": ps_no_atm, "option_types": ["PUT", "PUT", "CALL", "CALL"], "forward": 100.0, "r_cont": 0.05},
            365: {"strikes": ks_narrow, "market_prices": ps_narrow, "option_types": ["PUT", "PUT", "CALL", "CALL", "CALL"], "forward": 103.0, "r_cont": 0.05},
        },
        day_count_basis=365.0,
        reference_surface=ref_s,
        reference_long_tenors=[1825, 3650],
    )
    assert 3650 in surf_ref_fit.slices and surf_ref_fit.day_count_basis == 365.0
    surf_rt = BloombergMixtureSurface.from_json(surf_2.to_json())
    assert abs(surf_rt.get_vol(90, 95.0) - surf_2.get_vol(90, 95.0)) < 1e-12
    assert surf_rt.get_bid_ask_vol(90, 95.0) == surf_2.get_bid_ask_vol(90, 95.0)







