# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from bz_pricing.core.curves cimport Curve
from libc.math cimport pow, trunc, round

cdef class Bond:
    """
    Classe base para instrumentos de renda fixa e títulos públicos federais.
    """
    cpdef double present_value(self, Curve curve) except *:
        raise NotImplementedError("Método base present_value não implementado")

    cpdef list cashflow_details(self, Curve curve):
        raise NotImplementedError("Método base cashflow_details não implementado")

    cpdef list cashflow_details(self, Curve curve) except *:
        raise NotImplementedError("Método base cashflow_details não implementado")

cdef class GenericCashflowBond(Bond):
    """
    Título genérico baseado em fluxo de caixa com prazos e valores explícitos.
    """
    def __init__(self, int[:] dus, double[:] cashflows):
        if dus.shape[0] != cashflows.shape[0]:
            raise ValueError("Arrays dus e cashflows devem possuir a mesma dimensão")
        self._dus = dus
        self._cashflows = cashflows
        self._n_flows = dus.shape[0]

    cpdef double present_value(self, Curve curve) except *:
        cdef double total_pv = 0.0
        cdef int i, du
        cdef double rate, df, pv_fluxo

        for i in range(self._n_flows):
            du = self._dus[i]
            if du <= 0:
                continue

            rate = curve.get_rate(du)
            df = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0
            pv_fluxo = self._cashflows[i] / df
            pv_fluxo = trunc(pv_fluxo * 100000000.0) / 100000000.0
            total_pv += pv_fluxo

        return trunc(total_pv * 1000000.0) / 1000000.0

    cpdef double duration(self, Curve curve) except *:
        cdef double total_weighted_pv = 0.0
        cdef double total_pv = 0.0
        cdef int i, du
        cdef double rate, df, pv_fluxo

        for i in range(self._n_flows):
            du = self._dus[i]
            if du <= 0:
                continue

            rate = curve.get_rate(du)
            df = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0
            pv_fluxo = self._cashflows[i] / df
            total_weighted_pv += du * pv_fluxo
            total_pv += pv_fluxo

        if total_pv <= 0.0:
            return 0.0
        return total_weighted_pv / total_pv

    cpdef list cashflow_details(self, Curve curve):
        cdef list flows = []
        cdef int i, du
        cdef double rate, df, pv_fluxo

        for i in range(self._n_flows):
            du = self._dus[i]
            if du <= 0:
                continue

            rate = curve.get_rate(du)
            df = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0
            pv_fluxo = self._cashflows[i] / df
            pv_fluxo = trunc(pv_fluxo * 100000000.0) / 100000000.0
            flows.append((du, self._cashflows[i], rate, df, pv_fluxo))

        return flows

    cpdef list cashflow_details(self, Curve curve) except *:
        cdef list flows = []
        cdef int i, du
        cdef double rate, df, pv_fluxo

        for i in range(self._n_flows):
            du = self._dus[i]
            if du <= 0:
                continue

            rate = curve.get_rate(du)
            df = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0
            pv_fluxo = self._cashflows[i] / df
            pv_fluxo = trunc(pv_fluxo * 100000000.0) / 100000000.0
            flows.append((du, self._cashflows[i], rate, df, pv_fluxo))

        return flows

cdef class LTN(Bond):
    """
    Letra do Tesouro Nacional (título prefixado zero-cupom).
    Valor nominal de resgate no vencimento de R$ 1.000,00.
    """
    def __init__(self, int du, double face_value=1000.0):
        self._du = du
        self._face_value = face_value

    cpdef double present_value(self, Curve curve) except *:
        if self._du == 0:
            return self._face_value

        cdef double rate = curve.get_rate(self._du)
        cdef double discount_factor = trunc(pow(1.0 + rate, self._du / 252.0) * 100000000000000.0) / 100000000000000.0
        return trunc((self._face_value / discount_factor) * 1000000.0) / 1000000.0

    cpdef double duration(self, Curve curve) except *:
        return <double>self._du

    cpdef double dv01(self, Curve curve) except *:
        cdef double rate = curve.get_rate(self._du)
        cdef double mod_dur = (self._du / 252.0) / (1.0 + rate)
        cdef double pu = self.present_value(curve)
        return mod_dur * pu * 0.0001

    cpdef list cashflow_details(self, Curve curve):
        cdef double rate = curve.get_rate(self._du)
        cdef double df = trunc(pow(1.0 + rate, self._du / 252.0) * 100000000000000.0) / 100000000000000.0
        cdef double pu = trunc((self._face_value / df) * 1000000.0) / 1000000.0
        return [(self._du, self._face_value, rate, df, pu)]

    cpdef list cashflow_details(self, Curve curve) except *:
        cdef double rate = curve.get_rate(self._du)
        cdef double df = trunc(pow(1.0 + rate, self._du / 252.0) * 100000000000000.0) / 100000000000000.0
        cdef double pu = trunc((self._face_value / df) * 1000000.0) / 1000000.0
        return [(self._du, self._face_value, rate, df, pu)]

cdef class NTNF(Bond):
    """
    Nota do Tesouro Nacional - Série F (título prefixado com cupons semestrais).
    Taxa de cupom editalícia de 10,00% a.a. e valor nominal no vencimento de R$ 1.000,00.
    """
    def __init__(self, int[:] coupon_dus, double coupon_rate=0.10, double face_value=1000.0):
        self._coupon_dus = coupon_dus
        self._n_coupons = coupon_dus.shape[0]
        self._coupon_rate = coupon_rate
        self._face_value = face_value

    cpdef double present_value(self, Curve curve) except *:
        cdef double total_pv = 0.0
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_amount = self._face_value * (pow(1.0 + self._coupon_rate, 0.5) - 1.0)
        cdef double rounded_coupon = round(coupon_amount * 100000.0) / 100000.0
        cdef double pv_fluxo

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                pv_fluxo = (rounded_coupon + self._face_value) / discount_factor
            else:
                pv_fluxo = rounded_coupon / discount_factor

            pv_fluxo = round(pv_fluxo * 1000000000.0) / 1000000000.0
            total_pv += pv_fluxo

        return trunc(total_pv * 1000000.0) / 1000000.0

    cpdef double duration(self, Curve curve) except *:
        cdef double total_weighted = 0.0
        cdef double total_pv = 0.0
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_amount = self._face_value * (pow(1.0 + self._coupon_rate, 0.5) - 1.0)
        cdef double rounded_coupon = round(coupon_amount * 100000.0) / 100000.0
        cdef double pv_fluxo

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                pv_fluxo = (rounded_coupon + self._face_value) / discount_factor
            else:
                pv_fluxo = rounded_coupon / discount_factor

            pv_fluxo = round(pv_fluxo * 1000000000.0) / 1000000000.0
            total_weighted += du * pv_fluxo
            total_pv += pv_fluxo

        if total_pv <= 0.0:
            return 0.0
        return total_weighted / total_pv

    cpdef double modified_duration(self, Curve curve) except *:
        cdef double dur_du = self.duration(curve)
        cdef int last_du = self._coupon_dus[self._n_coupons - 1]
        cdef double last_rate = curve.get_rate(last_du)
        return (dur_du / 252.0) / (1.0 + last_rate)

    cpdef double dv01(self, Curve curve) except *:
        cdef double mod_dur = self.modified_duration(curve)
        cdef double pu = self.present_value(curve)
        return mod_dur * pu * 0.0001

    cpdef list cashflow_details(self, Curve curve):
        cdef list flows = []
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_amount = self._face_value * (pow(1.0 + self._coupon_rate, 0.5) - 1.0)
        cdef double rounded_coupon = round(coupon_amount * 100000.0) / 100000.0
        cdef double fluxo, pv_fluxo

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                fluxo = rounded_coupon + self._face_value
            else:
                fluxo = rounded_coupon

            pv_fluxo = round((fluxo / discount_factor) * 1000000000.0) / 1000000000.0
            flows.append((du, fluxo, rate, discount_factor, pv_fluxo))

        return flows

    cpdef list cashflow_details(self, Curve curve) except *:
        cdef list flows = []
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_amount = self._face_value * (pow(1.0 + self._coupon_rate, 0.5) - 1.0)
        cdef double rounded_coupon = round(coupon_amount * 100000.0) / 100000.0
        cdef double fluxo, pv_fluxo

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                fluxo = rounded_coupon + self._face_value
            else:
                fluxo = rounded_coupon

            pv_fluxo = round((fluxo / discount_factor) * 1000000000.0) / 1000000000.0
            flows.append((du, fluxo, rate, discount_factor, pv_fluxo))

        return flows

cdef class LFT(Bond):
    """
    Letra Financeira do Tesouro (título pós-fixado atrelado à taxa Selic).
    Precificada a partir da cotação de ágio/deságio aplicada sobre o VNA.
    """
    def __init__(self, int du):
        self._du = du

    cpdef double cotacao(self, Curve agio_curve) except *:
        if self._du <= 0:
            return 100.0

        cdef double rate = agio_curve.get_rate(self._du)
        cdef double discount_factor = trunc(pow(1.0 + rate, self._du / 252.0) * 100000000000000.0) / 100000000000000.0
        cdef double cot = 100.0 / discount_factor
        return trunc(cot * 10000.0) / 10000.0

    cpdef double present_value(self, Curve agio_curve, double vna=0.0, Curve selic_curve=None) except *:
        cdef double cot = self.cotacao(agio_curve)
        cdef double selic_rate = 0.0
        cdef double selic_factor = 1.0

        if vna <= 0.0 and selic_curve is not None:
            selic_rate = selic_curve.get_rate(self._du)
            selic_factor = trunc(pow(1.0 + selic_rate, self._du / 252.0) * 100000000000000.0) / 100000000000000.0
            vna = trunc((1000.0 * selic_factor) * 1000000.0) / 1000000.0

        return trunc((vna * (cot / 100.0)) * 1000000.0) / 1000000.0

    cpdef double duration(self, Curve agio_curve) except *:
        return <double>self._du

    cpdef double dv01(self, Curve agio_curve, double vna) except *:
        cdef double rate = agio_curve.get_rate(self._du)
        cdef double pu = self.present_value(agio_curve, vna)
        cdef double mod_dur = (self._du / 252.0) / (1.0 + rate)
        return mod_dur * pu * 0.0001

    cpdef list cashflow_details(self, Curve agio_curve, double vna=0.0):
        cdef double rate = agio_curve.get_rate(self._du)
        cdef double discount_factor = trunc(pow(1.0 + rate, self._du / 252.0) * 100000000000000.0) / 100000000000000.0
        cdef double cot = trunc((100.0 / discount_factor) * 10000.0) / 10000.0
        cdef double pu = 0.0
        if vna > 0.0:
            pu = trunc((vna * (cot / 100.0)) * 1000000.0) / 1000000.0
        return [(self._du, 100.0, rate, discount_factor, cot, vna, pu)]

    cpdef list cashflow_details(self, Curve agio_curve, double vna=0.0) except *:
        cdef double rate = agio_curve.get_rate(self._du)
        cdef double discount_factor = trunc(pow(1.0 + rate, self._du / 252.0) * 100000000000000.0) / 100000000000000.0
        cdef double cot = trunc((100.0 / discount_factor) * 10000.0) / 10000.0
        cdef double pu = 0.0
        if vna > 0.0:
            pu = trunc((vna * (cot / 100.0)) * 1000000.0) / 1000000.0
        return [(self._du, 100.0, rate, discount_factor, cot, vna, pu)]

cdef class NTNB(Bond):
    """
    Nota do Tesouro Nacional - Série B (título indexado ao IPCA com cupons semestrais).
    Taxa de cupom editalícia de 6,00% a.a. aplicada sobre o Valor Nominal Atualizado (VNA).
    """
    def __init__(self, int[:] coupon_dus, double coupon_rate=0.06):
        self._coupon_dus = coupon_dus
        self._n_coupons = coupon_dus.shape[0]
        self._coupon_rate = coupon_rate

    cpdef double cotacao(self, Curve real_curve) except *:
        cdef double total_cot = 0.0
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_factor = round((pow(1.0 + self._coupon_rate, 0.5) - 1.0) * 100.0 * 1000000.0) / 1000000.0
        cdef double fluxo, pv_fluxo

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = real_curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                fluxo = 100.0 + coupon_factor
            else:
                fluxo = coupon_factor

            pv_fluxo = round((fluxo / discount_factor) * 10000000000.0) / 10000000000.0
            total_cot += pv_fluxo

        return trunc(total_cot * 10000.0) / 10000.0

    cpdef double present_value(self, Curve real_curve, double vna=0.0, Curve inflation_curve=None, double base_vna=0.0, int du_inflation=0) except *:
        cdef double cot = self.cotacao(real_curve)
        cdef double ipca_rate = 0.0
        cdef double ipca_factor = 1.0

        if vna <= 0.0 and inflation_curve is not None and base_vna > 0.0 and du_inflation > 0:
            ipca_rate = inflation_curve.get_rate(du_inflation)
            ipca_factor = trunc(pow(1.0 + ipca_rate, du_inflation / 252.0) * 100000000000000.0) / 100000000000000.0
            vna = trunc((base_vna * ipca_factor) * 1000000.0) / 1000000.0

        return trunc((vna * (cot / 100.0)) * 1000000.0) / 1000000.0

    cpdef double duration(self, Curve real_curve) except *:
        cdef double total_weighted = 0.0
        cdef double total_cot = 0.0
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_factor = round((pow(1.0 + self._coupon_rate, 0.5) - 1.0) * 100.0 * 1000000.0) / 1000000.0
        cdef double fluxo, pv_fluxo

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = real_curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                fluxo = 100.0 + coupon_factor
            else:
                fluxo = coupon_factor

            pv_fluxo = round((fluxo / discount_factor) * 10000000000.0) / 10000000000.0
            total_weighted += du * pv_fluxo
            total_cot += pv_fluxo

        if total_cot <= 0.0:
            return 0.0
        return total_weighted / total_cot

    cpdef double modified_duration(self, Curve real_curve) except *:
        cdef double dur_du = self.duration(real_curve)
        cdef int last_du = self._coupon_dus[self._n_coupons - 1]
        cdef double last_rate = real_curve.get_rate(last_du)
        return (dur_du / 252.0) / (1.0 + last_rate)

    cpdef double dv01(self, Curve real_curve, double vna=0.0) except *:
        cdef double mod_dur = self.modified_duration(real_curve)
        cdef double pu = self.present_value(real_curve, vna=vna)
        return mod_dur * pu * 0.0001

    cpdef list cashflow_details(self, Curve real_curve, double vna=0.0):
        cdef list flows = []
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_factor = round((pow(1.0 + self._coupon_rate, 0.5) - 1.0) * 100.0 * 1000000.0) / 1000000.0
        cdef double fluxo_cot, pv_cot, pv_fin

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = real_curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                fluxo_cot = 100.0 + coupon_factor
            else:
                fluxo_cot = coupon_factor

            pv_cot = round((fluxo_cot / discount_factor) * 10000000000.0) / 10000000000.0
            pv_fin = 0.0
            if vna > 0.0:
                pv_fin = trunc((vna * (pv_cot / 100.0)) * 1000000.0) / 1000000.0

            flows.append((du, fluxo_cot, rate, discount_factor, pv_cot, pv_fin))

        return flows

    cpdef list cashflow_details(self, Curve real_curve, double vna=0.0) except *:
        cdef list flows = []
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_factor = round((pow(1.0 + self._coupon_rate, 0.5) - 1.0) * 100.0 * 1000000.0) / 1000000.0
        cdef double fluxo_cot, pv_cot, pv_fin

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = real_curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                fluxo_cot = 100.0 + coupon_factor
            else:
                fluxo_cot = coupon_factor

            pv_cot = round((fluxo_cot / discount_factor) * 10000000000.0) / 10000000000.0
            pv_fin = 0.0
            if vna > 0.0:
                pv_fin = trunc((vna * (pv_cot / 100.0)) * 1000000.0) / 1000000.0

            flows.append((du, fluxo_cot, rate, discount_factor, pv_cot, pv_fin))

        return flows

cdef class NTNBP(Bond):
    """
    Nota do Tesouro Nacional - Série B Principal (título indexado ao IPCA sem cupons intermediários).
    """
    def __init__(self, int du):
        self._du = du

    cpdef double cotacao(self, Curve real_curve) except *:
        if self._du <= 0:
            return 100.0

        cdef double rate = real_curve.get_rate(self._du)
        cdef double discount_factor = trunc(pow(1.0 + rate, self._du / 252.0) * 100000000000000.0) / 100000000000000.0
        cdef double cot = 100.0 / discount_factor
        return trunc(cot * 10000.0) / 10000.0

    cpdef double present_value(self, Curve real_curve, double vna=0.0, Curve inflation_curve=None, double base_vna=0.0, int du_inflation=0) except *:
        cdef double cot = self.cotacao(real_curve)
        cdef double ipca_rate = 0.0
        cdef double ipca_factor = 1.0

        if vna <= 0.0 and inflation_curve is not None and base_vna > 0.0 and du_inflation > 0:
            ipca_rate = inflation_curve.get_rate(du_inflation)
            ipca_factor = trunc(pow(1.0 + ipca_rate, du_inflation / 252.0) * 100000000000000.0) / 100000000000000.0
            vna = trunc((base_vna * ipca_factor) * 1000000.0) / 1000000.0

        return trunc((vna * (cot / 100.0)) * 1000000.0) / 1000000.0

    cpdef double duration(self, Curve real_curve) except *:
        return <double>self._du

    cpdef double modified_duration(self, Curve real_curve) except *:
        cdef double rate = real_curve.get_rate(self._du)
        return (self._du / 252.0) / (1.0 + rate)

    cpdef double dv01(self, Curve real_curve, double vna=0.0) except *:
        cdef double mod_dur = self.modified_duration(real_curve)
        cdef double pu = self.present_value(real_curve, vna=vna)
        return mod_dur * pu * 0.0001

    cpdef list cashflow_details(self, Curve real_curve, double vna=0.0):
        cdef double rate = real_curve.get_rate(self._du)
        cdef double discount_factor = trunc(pow(1.0 + rate, self._du / 252.0) * 100000000000000.0) / 100000000000000.0
        cdef double cot = trunc((100.0 / discount_factor) * 10000.0) / 10000.0
        cdef double pu = 0.0
        if vna > 0.0:
            pu = trunc((vna * (cot / 100.0)) * 1000000.0) / 1000000.0
        return [(self._du, 100.0, rate, discount_factor, cot, vna, pu)]

    cpdef list cashflow_details(self, Curve real_curve, double vna=0.0) except *:
        cdef double rate = real_curve.get_rate(self._du)
        cdef double discount_factor = trunc(pow(1.0 + rate, self._du / 252.0) * 100000000000000.0) / 100000000000000.0
        cdef double cot = trunc((100.0 / discount_factor) * 10000.0) / 10000.0
        cdef double pu = 0.0
        if vna > 0.0:
            pu = trunc((vna * (cot / 100.0)) * 1000000.0) / 1000000.0
        return [(self._du, 100.0, rate, discount_factor, cot, vna, pu)]

cdef class NTNC(Bond):
    """
    Nota do Tesouro Nacional - Série C (título indexado ao IGP-M com cupons semestrais).
    Taxa de cupom editalícia de 6,00% ou 12,00% a.a. aplicada sobre o VNA do IGP-M.
    """
    def __init__(self, int[:] coupon_dus, double coupon_rate=0.06):
        self._coupon_dus = coupon_dus
        self._n_coupons = coupon_dus.shape[0]
        self._coupon_rate = coupon_rate

    cpdef double cotacao(self, Curve real_curve) except *:
        cdef double total_cot = 0.0
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_factor = round((pow(1.0 + self._coupon_rate, 0.5) - 1.0) * 100.0 * 1000000.0) / 1000000.0
        cdef double fluxo, pv_fluxo

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = real_curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                fluxo = 100.0 + coupon_factor
            else:
                fluxo = coupon_factor

            pv_fluxo = round((fluxo / discount_factor) * 10000000000.0) / 10000000000.0
            total_cot += pv_fluxo

        return trunc(total_cot * 10000.0) / 10000.0

    cpdef double present_value(self, Curve real_curve, double vna=0.0, Curve inflation_curve=None, double base_vna=0.0, int du_inflation=0) except *:
        cdef double cot = self.cotacao(real_curve)
        cdef double igpm_rate = 0.0
        cdef double igpm_factor = 1.0

        if vna <= 0.0 and inflation_curve is not None and base_vna > 0.0 and du_inflation > 0:
            igpm_rate = inflation_curve.get_rate(du_inflation)
            igpm_factor = trunc(pow(1.0 + igpm_rate, du_inflation / 252.0) * 100000000000000.0) / 100000000000000.0
            vna = trunc((base_vna * igpm_factor) * 1000000.0) / 1000000.0

        return trunc((vna * (cot / 100.0)) * 1000000.0) / 1000000.0

    cpdef double duration(self, Curve real_curve) except *:
        cdef double total_weighted = 0.0
        cdef double total_cot = 0.0
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_factor = round((pow(1.0 + self._coupon_rate, 0.5) - 1.0) * 100.0 * 1000000.0) / 1000000.0
        cdef double fluxo, pv_fluxo

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = real_curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                fluxo = 100.0 + coupon_factor
            else:
                fluxo = coupon_factor

            pv_fluxo = round((fluxo / discount_factor) * 10000000000.0) / 10000000000.0
            total_weighted += du * pv_fluxo
            total_cot += pv_fluxo

        if total_cot <= 0.0:
            return 0.0
        return total_weighted / total_cot

    cpdef double modified_duration(self, Curve real_curve) except *:
        cdef double dur_du = self.duration(real_curve)
        cdef int last_du = self._coupon_dus[self._n_coupons - 1]
        cdef double last_rate = real_curve.get_rate(last_du)
        return (dur_du / 252.0) / (1.0 + last_rate)

    cpdef double dv01(self, Curve real_curve, double vna=0.0) except *:
        cdef double mod_dur = self.modified_duration(real_curve)
        cdef double pu = self.present_value(real_curve, vna=vna)
        return mod_dur * pu * 0.0001

    cpdef list cashflow_details(self, Curve real_curve, double vna=0.0):
        cdef list flows = []
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_factor = round((pow(1.0 + self._coupon_rate, 0.5) - 1.0) * 100.0 * 1000000.0) / 1000000.0
        cdef double fluxo_cot, pv_cot, pv_fin

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = real_curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                fluxo_cot = 100.0 + coupon_factor
            else:
                fluxo_cot = coupon_factor

            pv_cot = round((fluxo_cot / discount_factor) * 10000000000.0) / 10000000000.0
            pv_fin = 0.0
            if vna > 0.0:
                pv_fin = trunc((vna * (pv_cot / 100.0)) * 1000000.0) / 1000000.0

            flows.append((du, fluxo_cot, rate, discount_factor, pv_cot, pv_fin))

        return flows

    cpdef list cashflow_details(self, Curve real_curve, double vna=0.0) except *:
        cdef list flows = []
        cdef int i, du
        cdef double rate, discount_factor
        cdef double coupon_factor = round((pow(1.0 + self._coupon_rate, 0.5) - 1.0) * 100.0 * 1000000.0) / 1000000.0
        cdef double fluxo_cot, pv_cot, pv_fin

        for i in range(self._n_coupons):
            du = self._coupon_dus[i]
            if du <= 0:
                continue

            rate = real_curve.get_rate(du)
            discount_factor = trunc(pow(1.0 + rate, du / 252.0) * 100000000000000.0) / 100000000000000.0

            if i == self._n_coupons - 1:
                fluxo_cot = 100.0 + coupon_factor
            else:
                fluxo_cot = coupon_factor

            pv_cot = round((fluxo_cot / discount_factor) * 10000000000.0) / 10000000000.0
            pv_fin = 0.0
            if vna > 0.0:
                pv_fin = trunc((vna * (pv_cot / 100.0)) * 1000000.0) / 1000000.0

            flows.append((du, fluxo_cot, rate, discount_factor, pv_cot, pv_fin))

        return flows
