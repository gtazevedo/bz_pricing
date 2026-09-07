# cython: language_level=3

from bz_pricing.core.interpolators cimport Interpolator

cdef class Curve:
    cdef int _round_decimals
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

cdef class TBFCurve(Curve):
    cdef Curve _pre_curve
    cdef double _media_tbf

cdef class TRCurve(Curve):
    cdef Curve _tbf_curve
