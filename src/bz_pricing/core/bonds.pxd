# cython: language_level=3

from bz_pricing.core.curves cimport Curve

cdef class Bond:
    cpdef double present_value(self, Curve curve) except *
    cpdef list cashflow_details(self, Curve curve)

cdef class GenericCashflowBond(Bond):
    cdef int[:] _dus
    cdef double[:] _cashflows
    cdef int _n_flows
    cpdef double present_value(self, Curve curve) except *
    cpdef double duration(self, Curve curve) except *
    cpdef list cashflow_details(self, Curve curve)

cdef class LTN(Bond):
    cdef int _du
    cdef double _face_value
    cpdef double present_value(self, Curve curve) except *
    cpdef double duration(self, Curve curve) except *
    cpdef double dv01(self, Curve curve) except *
    cpdef list cashflow_details(self, Curve curve)

cdef class NTNF(Bond):
    cdef int[:] _coupon_dus
    cdef int _n_coupons
    cdef double _coupon_rate
    cdef double _face_value
    cpdef double present_value(self, Curve curve) except *
    cpdef double duration(self, Curve curve) except *
    cpdef double modified_duration(self, Curve curve) except *
    cpdef double dv01(self, Curve curve) except *
    cpdef list cashflow_details(self, Curve curve)

cdef class LFT(Bond):
    cdef int _du
    cpdef double cotacao(self, Curve agio_curve) except *
    cpdef double present_value(self, Curve agio_curve, double vna=*, Curve selic_curve=*) except *
    cpdef double duration(self, Curve agio_curve) except *
    cpdef double dv01(self, Curve agio_curve, double vna) except *
    cpdef list cashflow_details(self, Curve agio_curve, double vna=*)

cdef class NTNB(Bond):
    cdef int[:] _coupon_dus
    cdef int _n_coupons
    cdef double _coupon_rate
    cpdef double cotacao(self, Curve real_curve) except *
    cpdef double present_value(self, Curve real_curve, double vna=*, Curve inflation_curve=*, double base_vna=*, int du_inflation=*) except *
    cpdef double duration(self, Curve real_curve) except *
    cpdef double modified_duration(self, Curve real_curve) except *
    cpdef double dv01(self, Curve real_curve, double vna=*) except *
    cpdef list cashflow_details(self, Curve real_curve, double vna=*)

cdef class NTNBP(Bond):
    cdef int _du
    cpdef double cotacao(self, Curve real_curve) except *
    cpdef double present_value(self, Curve real_curve, double vna=*, Curve inflation_curve=*, double base_vna=*, int du_inflation=*) except *
    cpdef double duration(self, Curve real_curve) except *
    cpdef double modified_duration(self, Curve real_curve) except *
    cpdef double dv01(self, Curve real_curve, double vna=*) except *
    cpdef list cashflow_details(self, Curve real_curve, double vna=*)

cdef class NTNC(Bond):
    cdef int[:] _coupon_dus
    cdef int _n_coupons
    cdef double _coupon_rate
    cpdef double cotacao(self, Curve real_curve) except *
    cpdef double present_value(self, Curve real_curve, double vna=*, Curve inflation_curve=*, double base_vna=*, int du_inflation=*) except *
    cpdef double duration(self, Curve real_curve) except *
    cpdef double modified_duration(self, Curve real_curve) except *
    cpdef double dv01(self, Curve real_curve, double vna=*) except *
    cpdef list cashflow_details(self, Curve real_curve, double vna=*)
