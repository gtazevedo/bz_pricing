import pytest
import numpy as np
import math
from bz_pricing.core.interpolators import (
    Exponential252Interpolator,
    FlatForward252Interpolator,
    FlatForward252LinearInterpolator,
    Interpolation360Interpolator,
    Linear360Interpolator,
    PriceInterpolator
)
from bz_pricing.core.curves import SimpleCurve

def test_exponential_252():
    t_nodes = np.array([21.0, 42.0], dtype=np.float64)
    y_nodes = np.array([0.10, 0.12], dtype=np.float64)
    
    interp = Exponential252Interpolator()
    curve = SimpleCurve(t_nodes, y_nodes, interp)
    
    # Midpoint t = 31.5
    t_target = 31.5
    rate = curve.get_rate(t_target)
    
    # Manual calculation
    f1 = 1.0 + 0.10
    f2 = 1.0 + 0.12
    expected = f1 * math.pow(f2 / f1, (31.5 - 21.0) / (42.0 - 21.0)) - 1.0
    
    assert math.isclose(rate, expected, rel_tol=1e-9)
    assert rate > 0.10 and rate < 0.12

def test_flat_forward_252():
    t_nodes = np.array([21.0, 42.0], dtype=np.float64)
    y_nodes = np.array([0.10, 0.12], dtype=np.float64)
    
    interp = FlatForward252Interpolator()
    curve = SimpleCurve(t_nodes, y_nodes, interp)
    
    t_target = 31.5
    rate = curve.get_rate(t_target)
    
    # Manual calculation
    f1 = math.pow(1.0 + 0.10, 21.0 / 252.0)
    f2 = math.pow(1.0 + 0.12, 42.0 / 252.0)
    forward = f1 * math.pow(f2 / f1, (t_target - 21.0) / (42.0 - 21.0))
    expected = math.pow(forward, 252.0 / t_target) - 1.0
    
    assert math.isclose(rate, expected, rel_tol=1e-9)

def test_flat_forward_252_linear():
    t_nodes = np.array([21.0, 42.0], dtype=np.float64) # DU
    y_nodes = np.array([0.10, 0.12], dtype=np.float64)
    t_sec_nodes = np.array([30.0, 60.0], dtype=np.float64) # DC
    
    interp = FlatForward252LinearInterpolator()
    curve = SimpleCurve(t_nodes, y_nodes, interp, t_sec_nodes=t_sec_nodes)
    
    t_target_du = 31.5
    t_target_dc = 45.0
    
    rate = curve.get_rate(t_target_du, t_target_dc)
    
    # Manual calculation
    f1 = 1.0 + (0.10 * 30.0) / 360.0
    f2 = 1.0 + (0.12 * 60.0) / 360.0
    forward = f1 * math.pow(f2 / f1, (t_target_du - 21.0) / (42.0 - 21.0))
    expected = (forward - 1.0) * (360.0 / t_target_dc)
    
    assert math.isclose(rate, expected, rel_tol=1e-9)

def test_linear_360():
    t_nodes = np.array([30.0, 60.0], dtype=np.float64)
    y_nodes = np.array([0.10, 0.12], dtype=np.float64)
    
    interp = Linear360Interpolator()
    curve = SimpleCurve(t_nodes, y_nodes, interp)
    
    t_target = 45.0
    rate = curve.get_rate(t_target)
    
    # Manual calculation
    f1 = (0.10 * 30.0) / 360.0
    f2 = (0.12 * 60.0) / 360.0
    forward = f1 + (f2 - f1) * ((t_target - 30.0) / (60.0 - 30.0))
    expected = (forward * 360.0) / t_target
    
    assert math.isclose(rate, expected, rel_tol=1e-9)

def test_price_interpolator():
    t_nodes = np.array([30.0, 60.0], dtype=np.float64)
    y_nodes = np.array([1000.0, 1020.0], dtype=np.float64)
    
    interp = PriceInterpolator()
    curve = SimpleCurve(t_nodes, y_nodes, interp)
    
    t_target = 45.0
    price = curve.get_rate(t_target) # get_rate returns the interpolated price
    
    # Manual calculation
    expected = 1000.0 * math.pow(1020.0 / 1000.0, (45.0 - 30.0) / (60.0 - 30.0))
    
    assert math.isclose(price, expected, rel_tol=1e-9)

def test_extrapolation_flat():
    t_nodes = np.array([21.0, 42.0], dtype=np.float64)
    y_nodes = np.array([0.10, 0.12], dtype=np.float64)
    
    # extrap_left=0 (EXTRAP_FLAT), extrap_right=0 (EXTRAP_FLAT) by default
    interp = Exponential252Interpolator(0, 0)
    curve = SimpleCurve(t_nodes, y_nodes, interp)
    
    # Left extrapolation
    assert math.isclose(curve.get_rate(10.0), 0.10, rel_tol=1e-9)
    # Right extrapolation
    assert math.isclose(curve.get_rate(50.0), 0.12, rel_tol=1e-9)

def test_extrapolation_forward():
    t_nodes = np.array([21.0, 42.0], dtype=np.float64)
    y_nodes = np.array([0.10, 0.12], dtype=np.float64)
    
    # extrap_left=1 (EXTRAP_FLAT_FORWARD), extrap_right=1 (EXTRAP_FLAT_FORWARD)
    interp = Exponential252Interpolator(1, 1)
    curve = SimpleCurve(t_nodes, y_nodes, interp)
    
    # Left extrapolation (should mathematically project backwards from 21 and 42)
    f1 = 1.0 + 0.10
    f2 = 1.0 + 0.12
    t_target = 10.0
    expected_left = f1 * math.pow(f2 / f1, (t_target - 21.0) / (42.0 - 21.0)) - 1.0
    assert math.isclose(curve.get_rate(t_target), expected_left, rel_tol=1e-9)
    
    # Right extrapolation
    t_target = 60.0
    expected_right = f1 * math.pow(f2 / f1, (t_target - 21.0) / (42.0 - 21.0)) - 1.0
    assert math.isclose(curve.get_rate(t_target), expected_right, rel_tol=1e-9)


def test_single_node_zero_t_and_complex_curve_rounding():
    from bz_pricing.core.curves import ComplexCurve
    from bz_pricing.core.interpolators import FlatForward252LogInterpolator

    t_single = np.array([21.0], dtype=np.float64)
    y_single = np.array([0.135], dtype=np.float64)
    for cls in (Exponential252Interpolator, FlatForward252Interpolator, FlatForward252LogInterpolator, Linear360Interpolator):
        c = SimpleCurve(t_single, y_single, cls(1, 1))
        assert math.isclose(c.get_rate(1.0), 0.135, rel_tol=1e-12)
        assert math.isclose(c.get_rate(252.0), 0.135, rel_tol=1e-12)

    # t == 0.0 em FlatForward252Interpolator com extrapolação à esquerda não pode dividir por zero
    c2 = SimpleCurve(np.array([21.0, 42.0], dtype=np.float64), np.array([0.10, 0.12], dtype=np.float64), FlatForward252Interpolator(1, 1))
    r0 = c2.get_rate(0.0)
    assert math.isfinite(r0) and math.isclose(r0, 0.10, rel_tol=1e-12)

    # ComplexCurve aplica round_decimals também na extrapolação e expõe t_nodes / y_nodes
    cc = ComplexCurve(
        np.array([21.0, 42.0, 63.0], dtype=np.float64),
        np.array([0.10123456, 0.11123456, 0.12123456], dtype=np.float64),
        [FlatForward252Interpolator(0, 0), FlatForward252Interpolator(0, 0)],
        round_decimals=4,
    )
    assert math.isclose(cc.get_rate(5.0), 0.1012, rel_tol=1e-12)
    assert math.isclose(cc.get_rate(100.0), 0.1212, rel_tol=1e-12)
    assert len(cc.t_nodes) == 3 and len(cc.y_nodes) == 3


def test_offshore_interpolators_parametric_and_bootstrapping():
    from bz_pricing import (
        CurveManager,
        LogCubicDiscountInterpolator,
        LogLinearDiscountInterpolator,
        MonotoneConvexInterpolator,
        NelsonSiegelCurve,
        SvenssonCurve,
        bootstrap_ois_curve,
        bootstrap_treasury_curve,
    )

    tenors = [30, 90, 180, 365, 730, 1095, 1825]
    z_rates = [0.0420, 0.0430, 0.0442, 0.0455, 0.0468, 0.0476, 0.0488]
    t_arr = np.asarray(tenors, dtype=float)
    y_arr = np.asarray(z_rates, dtype=float)

    ll_interp = LogLinearDiscountInterpolator(0, 1, 365.0, "continuous")
    lc_interp = LogCubicDiscountInterpolator(0, 1, 365.0, "continuous")
    mc_interp = MonotoneConvexInterpolator(0, 1, 365.0, "continuous")
    crv_ll = SimpleCurve(t_arr, y_arr, ll_interp)
    crv_lc = SimpleCurve(t_arr, y_arr, lc_interp)
    crv_mc = SimpleCurve(t_arr, y_arr, mc_interp)

    for d in [15, 45, 120, 270, 540, 900, 1460, 2200]:
        assert ll_interp.forward_rate(float(d), t_arr, y_arr) > 0.0
        assert lc_interp.forward_rate(float(d), t_arr, y_arr) > 0.0
        assert mc_interp.forward_rate(float(d), t_arr, y_arr) > 0.0
        assert 0.02 < crv_ll.get_rate(float(d)) < 0.08
        assert 0.02 < crv_lc.get_rate(float(d)) < 0.08

    for d, r in zip(tenors, z_rates):
        assert abs(crv_mc.get_rate(float(d)) - r) < 1e-14

    ois_dcs = [30.0, 90.0, 180.0, 360.0, 720.0, 1080.0, 1800.0]
    ois_pars = [0.0432, 0.0436, 0.0442, 0.0450, 0.0462, 0.0471, 0.0484]
    bz_ois_ll = bootstrap_ois_curve(ois_dcs, ois_pars, convention="ACT/360", payment_freq_months=12, method="log_linear")
    bz_ois_mc = bootstrap_ois_curve(ois_dcs, ois_pars, convention="ACT/360", payment_freq_months=12, method="monotone_convex")
    for dc_m, y_par in zip(ois_dcs, ois_pars):
        if dc_m <= 360.0 + 1e-6:
            pv = (1.0 + y_par * (dc_m / 360.0)) * math.exp(-bz_ois_ll.get_rate(dc_m) * (dc_m / 360.0))
        else:
            n_c = int(round(dc_m / 360.0))
            pv = sum((y_par + (1.0 if j == n_c else 0.0)) * math.exp(-bz_ois_ll.get_rate(360.0 * j) * j) for j in range(1, n_c + 1))
        assert abs(pv - 1.0) < 1e-11
        assert bz_ois_mc.get_rate(dc_m) > 0.0

    ust_dcs = [90.0, 182.5, 365.0, 730.0, 1095.0, 1825.0, 3650.0]
    ust_pars = [0.0440, 0.0448, 0.0455, 0.0462, 0.0470, 0.0482, 0.0495]
    ust_mc = bootstrap_treasury_curve(ust_dcs, ust_pars, coupon_freq=2, day_count_basis=365.0, method="monotone_convex")
    for dc_m, y_par in zip(ust_dcs, ust_pars):
        if dc_m <= 182.5 + 1e-6:
            pv = (1.0 + y_par * (dc_m / 365.0)) * math.exp(-ust_mc.get_rate(dc_m) * (dc_m / 365.0))
        else:
            step = 182.5
            pts = []
            c_cur = dc_m
            while c_cur > 1e-6:
                pts.append(c_cur)
                c_cur -= step
            pts.reverse()
            pv = 0.0
            prev = 0.0
            for j, pt in enumerate(pts):
                amt = y_par * ((pt - prev) / 365.0) + (1.0 if j == len(pts) - 1 else 0.0)
                pv += amt * math.exp(-ust_mc.get_rate(pt) * (pt / 365.0))
                prev = pt
        assert abs(pv - 1.0) < 1e-11

    sv_true = SvenssonCurve(0.052, -0.015, -0.012, 0.018, 1.2, 4.5, day_count_basis=365.0)
    dcs_sv = np.array([30.0, 90.0, 180.0, 365.0, 730.0, 1095.0, 1825.0, 2555.0, 3650.0, 7300.0])
    z_sv = np.array([sv_true.get_rate(float(d)) for d in dcs_sv])
    sv_fit = SvenssonCurve.fit(dcs_sv, z_sv, day_count_basis=365.0)
    for d in dcs_sv:
        assert abs(sv_fit.get_rate(float(d)) - sv_true.get_rate(float(d))) < 1e-6
        assert abs(sv_fit.discount_factor(float(d)) - sv_true.discount_factor(float(d))) < 1e-5

    ns_crv = NelsonSiegelCurve.fit(ust_dcs, [ust_mc.get_rate(d) for d in ust_dcs], day_count_basis=365.0)
    assert ns_crv.instantaneous_forward(365.0) > 0.0

    cm = CurveManager(allow_download=False)
    mgr_sofr = cm.get_curve("SOFR_OIS", data=dict(zip(ois_dcs, ois_pars)), method="monotone_convex")
    assert abs(mgr_sofr.get_rate(ois_dcs[-1]) - bz_ois_mc.get_rate(ois_dcs[-1])) < 1e-14


def test_left_extrapolation_non_continuous_compounding_e5():
    from bz_pricing.core.curves import SimpleCurve
    from bz_pricing.core.interpolators import (
        LogCubicDiscountInterpolator,
        LogLinearDiscountInterpolator,
        MonotoneConvexInterpolator,
    )

    t_nodes = np.array([90.0, 180.0, 360.0], dtype=float)
    y_nodes = np.array([0.045, 0.048, 0.050], dtype=float)

    ll_flat = LogLinearDiscountInterpolator(0, 1, 360.0, "simple")
    ll_fwd = LogLinearDiscountInterpolator(1, 1, 360.0, "simple")
    ll_cont = LogLinearDiscountInterpolator(0, 1, 360.0, "continuous")

    crv_flat = SimpleCurve(t_nodes, y_nodes, ll_flat)
    crv_fwd = SimpleCurve(t_nodes, y_nodes, ll_fwd)
    crv_cont = SimpleCurve(t_nodes, y_nodes, ll_cont)

    # Em t = 45 < t_0 = 90: extrapolação flat simples preserva 0.045 exato;
    # flat-forward simples mantém f instantânea constante em (0, t_0], logo a taxa simples equivalente difere ligeiramente de y_0.
    assert abs(crv_flat.get_rate(45.0) - 0.045) < 1e-12
    assert abs(crv_cont.get_rate(45.0) - 0.045) < 1e-12
    tau0 = 90.0 / 360.0
    f0 = math.log(1.0 + 0.045 * tau0) / tau0
    tau_mid = 45.0 / 360.0
    expected_fwd_simple = (math.exp(f0 * tau_mid) - 1.0) / tau_mid
    assert abs(crv_fwd.get_rate(45.0) - expected_fwd_simple) < 1e-12
    assert abs(crv_fwd.get_rate(45.0) - 0.045) > 1e-5

    # LogCubic e MonotoneConvex também avaliam (0, t_0] via curva de desconto em regime simples (sem salto para y_0)
    for cls in (LogCubicDiscountInterpolator, MonotoneConvexInterpolator):
        interp_s = cls(0, 1, 360.0, "simple")
        interp_c = cls(0, 1, 360.0, "continuous")
        crv_s = SimpleCurve(t_nodes, y_nodes, interp_s)
        crv_c = SimpleCurve(t_nodes, y_nodes, interp_c)
        assert abs(crv_s.get_rate(90.0) - 0.045) < 1e-12
        assert abs(crv_c.get_rate(90.0) - 0.045) < 1e-12
        r_s_45 = crv_s.get_rate(45.0)
        assert 0.040 < r_s_45 < 0.048
        assert crv_s.get_rate(0.0) == 0.045

