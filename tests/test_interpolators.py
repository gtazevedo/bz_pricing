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
