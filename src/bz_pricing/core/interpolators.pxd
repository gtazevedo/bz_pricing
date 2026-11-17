# cython: language_level=3
from bz_pricing.core.conventions cimport DayCountConvention, CompoundingType

cdef enum ExtrapolationType:
    EXTRAP_FLAT = 0
    EXTRAP_FLAT_FORWARD = 1
    EXTRAP_LINEAR = 2

cdef class Interpolator:
    cdef int _extrap_left
    cdef int _extrap_right
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=*, double *t_sec_nodes=*) except *


cdef class LogLinearDiscountInterpolator(Interpolator):
    cdef double _day_count_basis
    cdef int _compounding
    cdef int _freq
    cdef tuple _eval_ln_df_and_fwd(self, double t, double *t_nodes, double *y_nodes, int n_nodes)
    cpdef double forward_rate(self, double t, double[:] t_nodes, double[:] y_nodes) except *


cdef class LogCubicDiscountInterpolator(Interpolator):
    cdef double _day_count_basis
    cdef int _compounding
    cdef int _freq
    cdef tuple _eval_ln_df_and_fwd(self, double t, double *t_nodes, double *y_nodes, int n_nodes)
    cpdef double forward_rate(self, double t, double[:] t_nodes, double[:] y_nodes) except *


cdef class MonotoneConvexInterpolator(Interpolator):
    cdef double _day_count_basis
    cdef int _compounding
    cdef int _freq
    cdef bint _enforce_positivity
    cdef tuple _eval_ln_df_and_fwd(self, double t, double *t_nodes, double *y_nodes, int n_nodes)
    cpdef double forward_rate(self, double t, double[:] t_nodes, double[:] y_nodes) except *
