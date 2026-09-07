# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from bz_pricing.core.curves cimport Curve
from libc.math cimport pow, trunc

cdef class Bond:
    cpdef double present_value(self, Curve curve) except *:
        raise NotImplementedError("Método base present_value")

cdef class LTN(Bond):
    """
    Letra do Tesouro Nacional (Zero-Coupon)
    """
    def __init__(self, int du, double face_value=1000.0):
        self._du = du
        self._face_value = face_value
        
    cpdef double present_value(self, Curve curve) except *:
        if self._du == 0:
            return self._face_value
            
        cdef double rate = curve.get_rate(self._du)
        # O Fator de desconto da LTN utiliza a taxa ao ano, descontada por (DU/252)
        cdef double discount_factor = pow(1.0 + rate, self._du / 252.0)
        
        # A B3/ANBIMA tipicamente calcula o PU final truncando em 6 casas
        return trunc((self._face_value / discount_factor) * 1000000.0) / 1000000.0

cdef class NTNF(Bond):
    """
    Nota do Tesouro Nacional - Série F
    """
    def __init__(self, int[:] coupon_dus, double coupon_rate, double face_value=1000.0):
        self._coupon_dus = coupon_dus
        self._n_coupons = coupon_dus.shape[0]
        self._coupon_rate = coupon_rate
        self._face_value = face_value
        
    cpdef double present_value(self, Curve curve) except *:
        cdef double total_pv = 0.0
        cdef int i
        cdef int du
        cdef double rate
        cdef double discount_factor
        
        # O cupom da NTN-F é de 10% a.a. => O semestral é (1.10)^0.5 - 1
        # Pelas regras da B3/ANBIMA, o valor nominal do cupom é truncado em 5 casas decimais
        cdef double coupon_amount = self._face_value * (pow(1.0 + self._coupon_rate, 0.5) - 1.0)
        cdef double rounded_coupon = trunc(coupon_amount * 100000.0) / 100000.0
        
        cdef double pv_fluxo = 0.0
        
        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            
            if du <= 0:
                continue
                
            rate = curve.get_rate(du)
            discount_factor = pow(1.0 + rate, du / 252.0)
            
            if i == self._n_coupons - 1:
                # O último fluxo é Cupom + Principal
                pv_fluxo = (rounded_coupon + self._face_value) / discount_factor
            else:
                # Demais fluxos são apenas Cupom
                pv_fluxo = rounded_coupon / discount_factor
                
            # Pelas regras da ANBIMA, cada fluxo descontado (VPL do Vértice) é truncado em 8 casas decimais
            # antes do somatório final.
            pv_fluxo = trunc(pv_fluxo * 100000000.0) / 100000000.0
            
            total_pv += pv_fluxo
                
        # PU Final da NTN-F é truncado em 6 casas decimais
        return trunc(total_pv * 1000000.0) / 1000000.0
