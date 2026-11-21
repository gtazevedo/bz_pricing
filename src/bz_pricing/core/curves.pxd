# cython: language_level=3

from bz_pricing.core.interpolators cimport Interpolator

cdef class Curve:
    cdef int _round_decimals
    cdef bint _is_trunc
    cpdef double get_rate(self, double t, double t_sec=*) except *

cdef class FlatCurve(Curve):
    cdef double _rate
    
cdef class AdditiveCurve(Curve):
    cdef list _curves
    
cdef class MultiplicativeCurve(Curve):
    cdef Curve _curve
    cdef double _scalar
    
cdef class SimpleCurve(Curve):
    cdef double[:] _t_nodes
    cdef double[:] _y_nodes
    cdef double[:] _t_sec_nodes
    cdef int _n_nodes
    cdef Interpolator _interpolator
    
cdef class ComplexCurve(Curve):
    cdef double[:] _t_nodes
    cdef double[:] _y_nodes
    cdef double[:] _t_sec_nodes
    cdef int _n_nodes
    cdef list _interpolators # Lista de objetos Interpolator para cada tranche

cdef class InflationCurve(Curve):
    cdef Curve _nominal_curve
    cdef Curve _real_curve

cdef class ConvenienceYieldCurve(Curve):
    cdef double _spot_price
    cdef Curve _futures_curve
    cdef Curve _pre_curve
    cdef SimpleCurve _interp_curve

cdef class SpreadCurve(Curve):
    cdef Curve _curve_a
    cdef Curve _curve_b

cdef class NelsonSiegelCurve(Curve):
    cdef double _beta0
    cdef double _beta1
    cdef double _beta2
    cdef double _lambda1
    cdef double _day_count_basis
    cdef int _compounding
    cdef int _freq
    cpdef double instantaneous_forward(self, double t) except *
    cpdef double discount_factor(self, double t) except *

cdef class SvenssonCurve(Curve):
    cdef double _beta0
    cdef double _beta1
    cdef double _beta2
    cdef double _beta3
    cdef double _lambda1
    cdef double _lambda2
    cdef double _day_count_basis
    cdef int _compounding
    cdef int _freq
    cpdef double instantaneous_forward(self, double t) except *
    cpdef double discount_factor(self, double t) except *

cdef class BloombergConvenienceYieldCurve(Curve):
    cdef double _spot_price
    cdef Curve _pre_curve
    cdef double[:] _t_nodes
    cdef double[:] _fwd_nodes
    cdef double[:] _r_cont_nodes
    cdef double[:] _d_pv_nodes
    cdef double[:] _y_cont_nodes
    cdef double[:] _season_weights
    cdef double[:] _ex_div_times
    cdef double[:] _ex_div_weights
    cdef double[:] _ex_div_r_cont
    cdef double[:] _decl_div_times
    cdef double[:] _decl_div_amounts
    cdef double[:] _decl_div_r_cont
    cdef int _n_nodes
    cdef int _n_bins
    cdef int _n_ex
    cdef int _n_decl
    cdef bint _is_index
    cdef double _start_bin_frac
    cdef double _day_count_basis
    cdef int _compounding
    cdef int _pre_compounding
    cdef double _long_fwd_yield
    cdef dict _diagnostics
    cdef double _get_pre_cont(self, double t, double t_sec, double tau) except *
    cpdef double forward_price(self, double t, double t_sec=*) except *
    cpdef double cash_dividend_pv(self, double t, double t_sec=*) except *
    cpdef double cash_dividend_fv(self, double t, double t_sec=*) except *
    cpdef double borrow_cost_rate(self, double t, double t_sec=*) except *

cpdef double cum_seasonality_exact(double[:] weights, double t_a, double t_b, double start_bin_frac=*)

cpdef list extract_linear360_knot_indices(double[:] dcs, double[:] rates, double tolerance=*)

