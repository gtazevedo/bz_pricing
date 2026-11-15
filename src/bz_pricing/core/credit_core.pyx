# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.math cimport pow, trunc, round
import numpy as np
cimport numpy as np


cdef inline double _min_d(double a, double b) noexcept nogil:
    return a if a < b else b


cpdef tuple compute_credit_cashflows(
    int[:] du_desc,
    double[:] yf_period,
    double[:] perc,
    int[:] event_code,
    double[:] r_desc_arr,
    double[:] r_di_arr,
    double vna,
    double fator_acruado_di,
    double coupon_rate,
    double multiplicador_di,
    double value_scale,
    bint is_cricra,
    bint is_di,
    bint is_pct_di,
    bint amortizacao_sobre_vne,
):
    """
    Núcleo Cython para projeção de fluxos, desconto T-14 e métricas de risco
    de Debêntures, CRI, CRA, FIDC e Letras Financeiras conforme normas ANBIMA/SND.

    event_code: 0 = Juros, 1 = Amortização, 2 = Vencimento.
    Retorna (taxa_cupom_arr, fluxo_arr, taxa_desc_arr, df_arr, pv_arr, metrics_tuple).
    """
    cdef Py_ssize_t n = du_desc.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] out_taxa_cupom = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_fluxo = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_taxa_desc = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_df = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_pv = np.zeros(n, dtype=np.float64)

    cdef Py_ssize_t i
    cdef int du, du_p, prev_du_di = 0, ev
    cdef double r_desc, r_di, f_di_acum, di_d, df_val, taxa_desc_eff
    cdef double prev_f_di = 1.0, saldo_vna = vna
    cdef double di_d_rem, f_pct_acum, f_cupom, f_cupom_raw, r_cupom
    cdef double f_di_p, f_spr_rem, f_spr_p
    cdef double valor_fluxo, pv_val, base_amort, p_val
    cdef double total_pv = 0.0, sum_du_pv = 0.0, sum_t_pv = 0.0
    cdef double sum_dmod_pv = 0.0, sum_conv_pv = 0.0, t_ano

    with nogil:
        for i in range(n):
            du = du_desc[i]
            ev = event_code[i]
            p_val = perc[i]
            r_desc = r_desc_arr[i]

            if is_di:
                r_di = r_di_arr[i]
                f_di_acum = pow(1.0 + r_di, du / 252.0)
                if is_pct_di:
                    di_d = pow(1.0 + r_di, 1.0 / 252.0) - 1.0
                    df_val = trunc(pow(1.0 + di_d * r_desc, du) * 1e14) / 1e14
                else:
                    df_val = trunc((f_di_acum * pow(1.0 + r_desc, du / 252.0)) * 1e14) / 1e14
                if du > 0:
                    taxa_desc_eff = pow(df_val, 252.0 / du) - 1.0
                else:
                    taxa_desc_eff = 0.0
            else:
                df_val = trunc(pow(1.0 + r_desc, du / 252.0) * 1e14) / 1e14
                taxa_desc_eff = r_desc

            if ev == 0:
                if is_di:
                    if is_pct_di:
                        f_pct_acum = pow(1.0 + di_d * multiplicador_di, du)
                        if prev_du_di == 0:
                            f_cupom = (fator_acruado_di * f_pct_acum) - 1.0
                        else:
                            f_cupom = (f_pct_acum / prev_f_di) - 1.0
                        r_cupom = multiplicador_di
                        prev_du_di = du
                        prev_f_di = f_pct_acum
                    else:
                        r_cupom = (p_val / 100.0) if p_val > 0.0 else coupon_rate
                        if prev_du_di == 0:
                            f_spr_rem = pow(1.0 + r_cupom, du / 252.0)
                            f_cupom = (fator_acruado_di * f_di_acum * f_spr_rem) - 1.0
                        else:
                            du_p = du - prev_du_di
                            f_di_p = f_di_acum / prev_f_di
                            f_spr_p = pow(1.0 + r_cupom, du_p / 252.0)
                            f_cupom = (f_di_p * f_spr_p) - 1.0
                        prev_du_di = du
                        prev_f_di = f_di_acum
                else:
                    r_cupom = (p_val / 100.0) if p_val > 0.0 else coupon_rate
                    f_cupom_raw = pow(1.0 + r_cupom, yf_period[i]) - 1.0
                    if is_cricra:
                        f_cupom = round(f_cupom_raw * 1e9) / 1e9
                    else:
                        f_cupom = trunc(f_cupom_raw * 1e9) / 1e9

                valor_fluxo = trunc((saldo_vna * f_cupom) * value_scale) / value_scale
            else:
                r_cupom = 0.0
                if ev == 2 or p_val <= 0.0 or ((not amortizacao_sobre_vne) and p_val >= 100.0):
                    valor_fluxo = trunc(saldo_vna * value_scale) / value_scale
                    saldo_vna = 0.0
                else:
                    base_amort = vna if amortizacao_sobre_vne else saldo_vna
                    valor_fluxo = _min_d(saldo_vna, trunc((base_amort * (p_val / 100.0)) * value_scale) / value_scale)
                    saldo_vna -= valor_fluxo

            pv_val = valor_fluxo / df_val

            out_taxa_cupom[i] = r_cupom
            out_fluxo[i] = valor_fluxo
            out_taxa_desc[i] = taxa_desc_eff
            out_df[i] = df_val
            out_pv[i] = pv_val

            total_pv += pv_val
            t_ano = du / 252.0
            sum_du_pv += du * pv_val
            sum_t_pv += t_ano * pv_val
            sum_dmod_pv += (t_ano / (1.0 + taxa_desc_eff)) * pv_val
            sum_conv_pv += ((t_ano * (t_ano + 1.0)) / ((1.0 + taxa_desc_eff) * (1.0 + taxa_desc_eff))) * pv_val

    cdef double pu = vna if n == 0 else (trunc(total_pv * value_scale) / value_scale)
    cdef double dur_du = 0.0
    cdef double dur_anos = 0.0
    cdef double d_mod = 0.0
    cdef double dv01 = 0.0
    cdef double conv = 0.0

    if total_pv > 0.0:
        dur_du = trunc((sum_du_pv / total_pv) * 100.0) / 100.0
        dur_anos = sum_t_pv / total_pv
        d_mod = sum_dmod_pv / total_pv
        dv01 = pu * d_mod * 0.0001
        conv = sum_conv_pv / total_pv

    return (
        out_taxa_cupom,
        out_fluxo,
        out_taxa_desc,
        out_df,
        out_pv,
        (pu, dur_du, dur_anos, d_mod, dv01, conv),
    )


cpdef double evaluate_gross_up_pv(
    int[:] du_arr,
    double[:] flows_arr,
    double rate,
) noexcept nogil:
    """Avalia o valor presente de um fluxo ajustado com gross-up de IR em C puro."""
    cdef Py_ssize_t n = du_arr.shape[0]
    cdef Py_ssize_t i
    cdef double pv = 0.0
    for i in range(n):
        pv += flows_arr[i] / pow(1.0 + rate, du_arr[i] / 252.0)
    return pv

