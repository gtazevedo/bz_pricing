# cython: language_level=3

from bz_pricing.core.curves cimport Curve

cdef class Bond:
    cpdef double present_value(self, Curve curve) except *

cdef class GenericCashflowBond(Bond):
    cdef int[:] _dus
    cdef double[:] _cashflows
    cdef int _n_flows

cdef class LTN(Bond):
    cdef int _du
    cdef double _face_value
    
cdef class NTNF(Bond):
    cdef int[:] _coupon_dus
    cdef int _n_coupons
    cdef double _coupon_rate
    cdef double _face_value
