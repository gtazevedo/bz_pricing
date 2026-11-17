# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.math cimport pow, round
import numpy as np
cimport numpy as np


cpdef double pu_di1(double rate, int du, bint round_2=True) noexcept nogil:
    """PU teórico do contrato futuro de DI1 (base 252 DU, face R$ 100.000)."""
    if du <= 0:
        return 100000.0
    cdef double pu = 100000.0 / pow(1.0 + rate, du / 252.0)
    if round_2:
        return round(pu * 100.0) / 100.0
    return pu


cpdef double rate_from_pu_di1(double pu, int du) noexcept nogil:
    """Taxa implícita (decimal a.a., 252 DU) a partir do PU do DI1."""
    if du <= 0 or pu <= 0.0:
        return 0.0
    return pow(100000.0 / pu, 252.0 / du) - 1.0


cpdef double dv01_di1(double rate, int du) noexcept nogil:
    """DV01 unitário por contrato de DI1 (variação do PU para +1 bp na taxa)."""
    if du <= 0:
        return 0.0
    return pu_di1(rate, du, False) - pu_di1(rate + 0.0001, du, False)


cpdef double pu_ddi(double rate, int dc, bint round_2=True) noexcept nogil:
    """PU teórico do futuro de Cupom Cambial DDI (base linear ACT/360, face USD 100.000)."""
    if dc <= 0:
        return 100000.0
    cdef double pu = 100000.0 / (1.0 + rate * (dc / 360.0))
    if round_2:
        return round(pu * 100.0) / 100.0
    return pu


cpdef double dv01_ddi(double rate, int dc) noexcept nogil:
    """DV01 unitário em USD do contrato DDI para +1 bp na taxa de cupom."""
    if dc <= 0:
        return 0.0
    return pu_ddi(rate, dc, False) - pu_ddi(rate + 0.0001, dc, False)


cpdef double pu_dap(double rate, int du, bint round_2=True) noexcept nogil:
    """PU teórico do futuro de Cupom de IPCA DAP (base 252 DU exponencial, face 100.000)."""
    if du <= 0:
        return 100000.0
    cdef double pu = 100000.0 / pow(1.0 + rate, du / 252.0)
    if round_2:
        return round(pu * 100.0) / 100.0
    return pu


cpdef double dv01_dap(double rate, int du) noexcept nogil:
    """DV01 unitário (em pontos de VNA) do contrato DAP para +1 bp."""
    if du <= 0:
        return 0.0
    return pu_dap(rate, du, False) - pu_dap(rate + 0.0001, du, False)


cpdef double fair_dol(
    double spot_ptax,
    double r_pre,
    int du_pre,
    double r_cupom,
    int dc_cupom,
) noexcept nogil:
    """Preço teórico do futuro de dólar (DOL/WDO) por paridade de juros (pontos/USD 1.000)."""
    cdef double f_pre = pow(1.0 + r_pre, du_pre / 252.0) if du_pre > 0 else 1.0
    cdef double f_cup = (1.0 + r_cupom * (dc_cupom / 360.0)) if dc_cupom > 0 else 1.0
    return spot_ptax * (f_pre / f_cup)


cpdef double fair_ind(
    double spot_ibov,
    double r_pre,
    int du_pre,
    double div_yield=0.0,
) noexcept nogil:
    """Preço teórico do futuro de Ibovespa (IND/WIN) por custo de carregamento 252 DU."""
    if du_pre <= 0:
        return spot_ibov
    return spot_ibov * pow((1.0 + r_pre) / (1.0 + div_yield), du_pre / 252.0)


cpdef double future_daily_settlement(
    double pu_atual,
    double pu_anterior,
    double multiplier,
    double contracts,
    double direction=1.0,
) noexcept nogil:
    """Ajuste diário B3 de contratos futuros."""
    return (pu_atual - pu_anterior) * multiplier * contracts * direction


cpdef tuple swap_pre_di_mtm(
    double notional,
    double fixed_rate,
    double market_pre_rate,
    int du,
    double di_accrued_factor=1.0,
    double fixed_direction=1.0,
    int du_rem=-1,
):
    """
    Marcação a mercado de Swap DI x Pré (B3).
    `du`: prazo total em DU entre início e vencimento do swap.
    `du_rem`: prazo remanescente em DU da data-base até o vencimento (padrão = `du`).
    `fixed_direction = 1.0` -> ativo na ponta Pré (recebe Pré, paga DI).
    `fixed_direction = -1.0` -> ativo na ponta DI (recebe DI, paga Pré).
    Retorna (mtm, pv_pre, pv_di, dv01_pre).
    """
    cdef int eff_du_rem = du if du_rem < 0 else du_rem
    cdef double df = pow(1.0 + market_pre_rate, eff_du_rem / 252.0) if eff_du_rem > 0 else 1.0
    cdef double fv_pre = notional * pow(1.0 + fixed_rate, du / 252.0) if du > 0 else notional
    cdef double pv_pre = fv_pre / df
    cdef double pv_di = notional * di_accrued_factor
    cdef double mtm = fixed_direction * (pv_pre - pv_di)

    cdef double df_up = pow(1.0 + market_pre_rate + 0.0001, eff_du_rem / 252.0) if eff_du_rem > 0 else 1.0
    cdef double pv_pre_up = fv_pre / df_up
    cdef double dv01_pre = fixed_direction * (pv_pre_up - pv_pre)
    return (mtm, pv_pre, pv_di, dv01_pre)


cpdef double ndf_mtm(
    double notional_usd,
    double strike_brl,
    double spot_ptax,
    double r_pre,
    int du_pre,
    double r_cupom,
    int dc_cupom,
    double direction=1.0,
) noexcept nogil:
    """
    Marcação a mercado de NDF (USD/BRL).
    `direction = 1.0` -> comprado em USD (recebe PTAX - Strike).
    """
    cdef double fwd = fair_dol(spot_ptax, r_pre, du_pre, r_cupom, dc_cupom)
    cdef double df = pow(1.0 + r_pre, du_pre / 252.0) if du_pre > 0 else 1.0
    return direction * notional_usd * (fwd - strike_brl) / df


cpdef tuple price_di1_batch(
    double[:] rates,
    int[:] dus,
    double[:] contracts,
) :
    """Precifica em lote um vetor de contratos DI1 em Cython puro."""
    cdef Py_ssize_t n = rates.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] pus = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] dv01s = np.zeros(n, dtype=np.float64)
    cdef Py_ssize_t i
    cdef double p, d
    with nogil:
        for i in range(n):
            p = pu_di1(rates[i], dus[i], True)
            d = dv01_di1(rates[i], dus[i])
            pus[i] = p
            dv01s[i] = d * contracts[i]
    return pus, dv01s
