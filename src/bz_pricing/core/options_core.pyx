# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, log, sqrt, erf, erfc, fabs, pow, fmax, fmin, NAN, isnan, floor
from libc.stdlib cimport malloc, free

cdef double INV_SQRT_2PI = 0.3989422804014327
cdef double INV_SQRT_2 = 0.7071067811865475


cdef inline double _norm_pdf(double x) noexcept nogil:
    return INV_SQRT_2PI * exp(-0.5 * x * x)


cdef inline double _norm_cdf(double x) noexcept nogil:
    if x < 0.0:
        return 0.5 * erfc(-x * INV_SQRT_2)
    return 0.5 * (1.0 + erf(x * INV_SQRT_2))


cpdef double norm_cdf(double x) noexcept nogil:
    return _norm_cdf(x)


cpdef double norm_pdf(double x) noexcept nogil:
    return _norm_pdf(x)


cpdef double norm_ppf(double p) noexcept nogil:
    """Inversa da normal padrão (aproximação racional de Acklam, erro < 1.15e-9)."""
    if p <= 0.0:
        return -1e12
    if p >= 1.0:
        return 1e12

    cdef double a1 = -3.969683028665376e+01
    cdef double a2 =  2.209460984245205e+02
    cdef double a3 = -2.759285104469687e+02
    cdef double a4 =  1.383577518672690e+02
    cdef double a5 = -3.066479806614716e+01
    cdef double a6 =  2.506628277459239e+00

    cdef double b1 = -5.447609879822406e+01
    cdef double b2 =  1.615858368580409e+02
    cdef double b3 = -1.556989798598866e+02
    cdef double b4 =  6.680131188771972e+01
    cdef double b5 = -1.328068155288572e+01

    cdef double c1 = -7.784894002430293e-03
    cdef double c2 = -3.223964580411365e-01
    cdef double c3 = -2.400758277161838e+00
    cdef double c4 = -2.549732539343734e+00
    cdef double c5 =  4.374664141464968e+00
    cdef double c6 =  2.938163982698783e+00

    cdef double d1 =  7.784695709041462e-03
    cdef double d2 =  3.224671290700398e-01
    cdef double d3 =  2.445134137142996e+00
    cdef double d4 =  3.754408661907416e+00

    cdef double p_low = 0.02425
    cdef double p_high = 1.0 - p_low
    cdef double q, r, num, den

    if p < p_low:
        q = sqrt(-2.0 * log(p))
        num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return num / den
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        return num / den
    else:
        q = sqrt(-2.0 * log(1.0 - p))
        num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return -num / den


cdef void _bsm_core(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call,
    double* out_price,
    double* out_delta,
    double* out_gamma,
    double* out_vega,
    double* out_theta,
    double* out_rho,
    double* out_vanna,
    double* out_volga,
) noexcept nogil:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        if is_call:
            out_price[0] = fmax(S - K, 0.0)
            out_delta[0] = 1.0 if S > K else 0.0
        else:
            out_price[0] = fmax(K - S, 0.0)
            out_delta[0] = -1.0 if K > S else 0.0
        out_gamma[0] = 0.0
        out_vega[0] = 0.0
        out_theta[0] = 0.0
        out_rho[0] = 0.0
        out_vanna[0] = 0.0
        out_volga[0] = 0.0
        return

    cdef double sqrt_T = sqrt(T)
    cdef double sig_sqrt_T = sigma * sqrt_T
    cdef double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sig_sqrt_T
    cdef double d2 = d1 - sig_sqrt_T

    cdef double df_q = exp(-q * T)
    cdef double df_r = exp(-r * T)
    cdef double pdf_d1 = _norm_pdf(d1)
    cdef double cdf_d1, cdf_d2, cdf_md1, cdf_md2

    if is_call:
        cdf_d1 = _norm_cdf(d1)
        cdf_d2 = _norm_cdf(d2)
        out_price[0] = S * df_q * cdf_d1 - K * df_r * cdf_d2
        out_delta[0] = df_q * cdf_d1
        out_theta[0] = (
            -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * df_r * cdf_d2
            + q * S * df_q * cdf_d1
        )
        out_rho[0] = K * T * df_r * cdf_d2
    else:
        cdf_md1 = _norm_cdf(-d1)
        cdf_md2 = _norm_cdf(-d2)
        out_price[0] = K * df_r * cdf_md2 - S * df_q * cdf_md1
        out_delta[0] = -df_q * cdf_md1
        out_theta[0] = (
            -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * df_r * cdf_md2
            - q * S * df_q * cdf_md1
        )
        out_rho[0] = -K * T * df_r * cdf_md2

    out_gamma[0] = (df_q * pdf_d1) / (S * sig_sqrt_T)
    out_vega[0] = S * df_q * pdf_d1 * sqrt_T
    out_vanna[0] = -df_q * pdf_d1 * d2 / sigma
    out_volga[0] = out_vega[0] * d1 * d2 / sigma


cpdef tuple bsm_price_greeks(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call=True,
):
    cdef double price, delta, gamma, vega, theta, rho, vanna, volga
    with nogil:
        _bsm_core(
            S, K, r, q, T, sigma, is_call,
            &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga,
        )
    return (price, delta, gamma, vega, theta, rho, vanna, volga)


cpdef tuple black76_price_greeks(
    double F,
    double K,
    double r,
    double T,
    double sigma,
    bint is_call=True,
):
    """Black-76 sobre futuro F (equivalente a BSM com S=F e q=r)."""
    cdef double price, delta, gamma, vega, theta, rho, vanna, volga
    with nogil:
        _bsm_core(
            F, K, r, r, T, sigma, is_call,
            &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga,
        )
    return (price, delta, gamma, vega, theta, rho, vanna, volga)


cdef double _crr_american_core(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call,
    int steps,
    double* out_delta,
    double* out_gamma,
    double* out_theta,
) noexcept nogil:
    """Núcleo CRR em C puro: calcula preço americano e delta/gamma/theta diretamente da árvore."""
    if T <= 0.0 or sigma <= 0.0 or steps <= 0 or S <= 0.0 or K <= 0.0:
        out_delta[0] = (1.0 if S > K else 0.0) if is_call else (-1.0 if K > S else 0.0)
        out_gamma[0] = 0.0
        out_theta[0] = 0.0
        return fmax(S - K, 0.0) if is_call else fmax(K - S, 0.0)

    cdef double* values = <double*>malloc((steps + 1) * sizeof(double))
    if values == NULL:
        out_delta[0] = 0.0
        out_gamma[0] = 0.0
        out_theta[0] = 0.0
        return 0.0

    cdef double dt = T / steps
    cdef double u = exp(sigma * sqrt(dt))
    cdef double d = 1.0 / u
    cdef double d_sq = d * d
    cdef double disc = exp(-r * dt)
    cdef double p = (exp((r - q) * dt) - d) / (u - d)
    if p < 0.0 or p > 1.0:
        free(values)
        out_delta[0] = NAN
        out_gamma[0] = NAN
        out_theta[0] = NAN
        return NAN
    cdef double one_minus_p = 1.0 - p

    cdef int i, j
    cdef double spot_i = S * pow(u, steps)
    for i in range(steps + 1):
        values[i] = fmax(spot_i - K, 0.0) if is_call else fmax(K - spot_i, 0.0)
        spot_i *= d_sq

    cdef double v20 = 0.0, v21 = 0.0, v22 = 0.0
    cdef double v10 = 0.0, v11 = 0.0
    cdef double spot_row_start = S * pow(u, steps - 1)
    cdef double hold, exercise

    for j in range(steps - 1, -1, -1):
        spot_i = spot_row_start
        for i in range(j + 1):
            hold = disc * (p * values[i] + one_minus_p * values[i + 1])
            exercise = (spot_i - K) if is_call else (K - spot_i)
            values[i] = hold if hold > exercise else exercise
            spot_i *= d_sq
        spot_row_start *= d
        if j == 2:
            v20 = values[0]
            v21 = values[1]
            v22 = values[2]
        elif j == 1:
            v10 = values[0]
            v11 = values[1]

    cdef double val = values[0]
    free(values)

    cdef double su = S * u
    cdef double sd = S * d
    if steps >= 1 and fabs(su - sd) > 1e-12:
        out_delta[0] = (v10 - v11) / (su - sd)
    else:
        out_delta[0] = (1.0 if S > K else 0.0) if is_call else (-1.0 if K > S else 0.0)

    cdef double suu, sdd, d1_u, d1_d
    if steps >= 2 and fabs(su - sd) > 1e-12:
        suu = S * u * u
        sdd = S * d * d
        d1_u = (v20 - v21) / (suu - S)
        d1_d = (v21 - v22) / (S - sdd)
        out_gamma[0] = (d1_u - d1_d) / (0.5 * (suu - sdd))
        out_theta[0] = (v21 - val) / (2.0 * dt)
    else:
        out_gamma[0] = 0.0
        out_theta[0] = 0.0

    return val


cpdef double crr_american_price(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call=True,
    int steps=200,
):
    """Árvore binomial Cox-Ross-Rubinstein (CRR) em C puro para opções americanas (média par-ímpar N e N+1)."""
    cdef double d_tmp, g_tmp, t_tmp, v1, v2
    with nogil:
        v1 = _crr_american_core(S, K, r, q, T, sigma, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
        if steps >= 2:
            v2 = _crr_american_core(S, K, r, q, T, sigma, is_call, steps + 1, &d_tmp, &g_tmp, &t_tmp)
            return 0.5 * (v1 + v2)
        return v1


cpdef tuple crr_american_price_greeks(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call=True,
    int steps=200,
):
    """Preço americano CRR com suavização par-ímpar e gregas consistentes."""
    cdef double price, delta, gamma, theta, vega, rho
    cdef double p1, d1_t, g1_t, t1_t, p2, d2_t, g2_t, t2_t
    cdef double d_tmp, g_tmp, t_tmp
    cdef double dv = 1e-4
    cdef double dr = 1e-4
    cdef double p_v_up, p_v_dn, p_r_up, p_r_dn

    with nogil:
        p1 = _crr_american_core(S, K, r, q, T, sigma, is_call, steps, &d1_t, &g1_t, &t1_t)
        if steps >= 2:
            p2 = _crr_american_core(S, K, r, q, T, sigma, is_call, steps + 1, &d2_t, &g2_t, &t2_t)
            price = 0.5 * (p1 + p2)
            delta = 0.5 * (d1_t + d2_t)
            gamma = 0.5 * (g1_t + g2_t)
            theta = 0.5 * (t1_t + t2_t)
        else:
            price = p1
            delta = d1_t
            gamma = g1_t
            theta = t1_t

        if T > 0.0 and sigma > dv and S > 0.0 and K > 0.0:
            p1 = _crr_american_core(S, K, r, q, T, sigma + dv, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
            p_v_up = 0.5 * (p1 + _crr_american_core(S, K, r, q, T, sigma + dv, is_call, steps + 1, &d_tmp, &g_tmp, &t_tmp)) if steps >= 2 else p1
            p1 = _crr_american_core(S, K, r, q, T, sigma - dv, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
            p_v_dn = 0.5 * (p1 + _crr_american_core(S, K, r, q, T, sigma - dv, is_call, steps + 1, &d_tmp, &g_tmp, &t_tmp)) if steps >= 2 else p1
            vega = (p_v_up - p_v_dn) / (2.0 * dv)

            p1 = _crr_american_core(S, K, r + dr, q, T, sigma, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
            p_r_up = 0.5 * (p1 + _crr_american_core(S, K, r + dr, q, T, sigma, is_call, steps + 1, &d_tmp, &g_tmp, &t_tmp)) if steps >= 2 else p1
            p1 = _crr_american_core(S, K, r - dr, q, T, sigma, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
            p_r_dn = 0.5 * (p1 + _crr_american_core(S, K, r - dr, q, T, sigma, is_call, steps + 1, &d_tmp, &g_tmp, &t_tmp)) if steps >= 2 else p1
            rho = (p_r_up - p_r_dn) / (2.0 * dr)
        else:
            vega = 0.0
            rho = 0.0

    return (price, delta, gamma, vega, theta, rho)


cdef double _crr_american_discrete_core(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call,
    int steps,
    double *div_t,
    double *div_a,
    int n_div,
    double *prop_t,
    double *prop_y,
    int n_prop,
) noexcept nogil:
    cdef int k, j, i
    cdef double d_tmp, g_tmp, t_tmp
    if n_div == 0 and n_prop == 0:
        return _crr_american_core(S, K, r, q, T, sigma, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
    if T <= 0.0 or sigma <= 0.0 or steps <= 0 or S <= 0.0 or K <= 0.0:
        return fmax(S - K, 0.0) if is_call else fmax(K - S, 0.0)

    cdef double pv_div0 = 0.0
    for k in range(n_div):
        if 0.0 < div_t[k] <= T + 1e-12 and div_a[k] > 0.0:
            pv_div0 += div_a[k] * exp(-r * div_t[k])

    cdef double S_star = fmax(1e-8, S - pv_div0)
    cdef double dt = T / (<double>steps)
    cdef double u = exp(sigma * sqrt(dt))
    cdef double d = 1.0 / u
    cdef double d_sq = d * d
    cdef double growth = exp((r - q) * dt)
    cdef double disc = exp(-r * dt)
    cdef double p = (growth - d) / (u - d)
    if p < 0.0 or p > 1.0:
        return NAN
    cdef double one_minus_p = 1.0 - p

    cdef double *values = <double *>malloc((steps + 1) * sizeof(double))
    cdef double *step_pv = <double *>malloc((steps + 1) * sizeof(double))
    cdef double *step_prop = <double *>malloc((steps + 1) * sizeof(double))
    if values == NULL or step_pv == NULL or step_prop == NULL:
        free(values); free(step_pv); free(step_prop)
        return NAN

    cdef double t_j, pv_j, pr_j, y_clip
    for j in range(steps + 1):
        t_j = (<double>j) * dt
        pv_j = 0.0
        for k in range(n_div):
            if div_t[k] > t_j + 1e-12 and div_t[k] <= T + 1e-12 and div_a[k] > 0.0:
                pv_j += div_a[k] * exp(-r * (div_t[k] - t_j))
        pr_j = 1.0
        for k in range(n_prop):
            if 0.0 < prop_t[k] <= t_j + 1e-12 and prop_y[k] > 0.0:
                y_clip = prop_y[k] if prop_y[k] < 0.999 else 0.999
                pr_j *= (1.0 - y_clip)
        step_pv[j] = pv_j
        step_prop[j] = pr_j

    cdef double spot_i = S_star * step_prop[steps] * pow(u, steps)
    cdef double s_node
    for i in range(steps + 1):
        s_node = spot_i + step_pv[steps]
        values[i] = fmax(s_node - K, 0.0) if is_call else fmax(K - s_node, 0.0)
        spot_i *= d_sq

    cdef double spot_row_base = S_star * pow(u, steps - 1)
    cdef double hold, exercise
    for j in range(steps - 1, -1, -1):
        spot_i = spot_row_base * step_prop[j]
        for i in range(j + 1):
            hold = disc * (p * values[i] + one_minus_p * values[i + 1])
            s_node = spot_i + step_pv[j]
            exercise = (s_node - K) if is_call else (K - s_node)
            values[i] = hold if hold > exercise else exercise
            spot_i *= d_sq
        spot_row_base *= d

    cdef double val = values[0]
    free(values)
    free(step_pv)
    free(step_prop)
    return val


cdef double _crr_disc_smoothed(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call,
    int steps,
    double *div_t,
    double *div_a,
    int n_div,
    double *prop_t,
    double *prop_y,
    int n_prop,
) noexcept nogil:
    cdef double v1 = _crr_american_discrete_core(
        S, K, r, q, T, sigma, is_call, steps, div_t, div_a, n_div, prop_t, prop_y, n_prop
    )
    if steps >= 2:
        return 0.5 * (
            v1 + _crr_american_discrete_core(
                S, K, r, q, T, sigma, is_call, steps + 1, div_t, div_a, n_div, prop_t, prop_y, n_prop
            )
        )
    return v1


cpdef tuple crr_american_discrete_div_price_greeks(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call,
    int steps,
    double[:] div_times,
    double[:] div_amounts,
    double[:] prop_div_times,
    double[:] prop_div_yields,
):
    """Árvore CRR para opções americanas com dividendos discretos em dinheiro e/ou proporcionais."""
    cdef int n_div = <int>div_times.shape[0]
    cdef int n_prop = <int>prop_div_times.shape[0]
    cdef double *dt_ptr = &div_times[0] if n_div > 0 else NULL
    cdef double *da_ptr = &div_amounts[0] if n_div > 0 else NULL
    cdef double *pt_ptr = &prop_div_times[0] if n_prop > 0 else NULL
    cdef double *py_ptr = &prop_div_yields[0] if n_prop > 0 else NULL

    cdef double price = 0.0, delta = 0.0, gamma = 0.0, vega = 0.0, theta = 0.0, rho = 0.0
    cdef double p_s_up, p_s_dn, p_v_up, p_v_dn, p_r_up, p_r_dn, p_t_dn
    cdef double dS = fmax(1e-3, S * fmax(0.01, 0.5 * sigma * sqrt(fmax(1e-6, T) / fmax(4.0, <double>steps))))
    cdef double dv = 1e-4
    cdef double dr = 1e-4
    cdef double dt_th = fmin(1.0 / 252.0, 0.1 * T)
    cdef int k
    cdef double *dt_sh = NULL
    cdef double *pt_sh = NULL

    with nogil:
        price = _crr_disc_smoothed(S, K, r, q, T, sigma, is_call, steps, dt_ptr, da_ptr, n_div, pt_ptr, py_ptr, n_prop)
        if T > 0.0 and sigma > dv and S > dS and K > 0.0:
            p_s_up = _crr_disc_smoothed(S + dS, K, r, q, T, sigma, is_call, steps, dt_ptr, da_ptr, n_div, pt_ptr, py_ptr, n_prop)
            p_s_dn = _crr_disc_smoothed(S - dS, K, r, q, T, sigma, is_call, steps, dt_ptr, da_ptr, n_div, pt_ptr, py_ptr, n_prop)
            delta = (p_s_up - p_s_dn) / (2.0 * dS)
            gamma = (p_s_up - 2.0 * price + p_s_dn) / (dS * dS)

            p_v_up = _crr_disc_smoothed(S, K, r, q, T, sigma + dv, is_call, steps, dt_ptr, da_ptr, n_div, pt_ptr, py_ptr, n_prop)
            p_v_dn = _crr_disc_smoothed(S, K, r, q, T, sigma - dv, is_call, steps, dt_ptr, da_ptr, n_div, pt_ptr, py_ptr, n_prop)
            vega = (p_v_up - p_v_dn) / (2.0 * dv)

            p_r_up = _crr_disc_smoothed(S, K, r + dr, q, T, sigma, is_call, steps, dt_ptr, da_ptr, n_div, pt_ptr, py_ptr, n_prop)
            p_r_dn = _crr_disc_smoothed(S, K, r - dr, q, T, sigma, is_call, steps, dt_ptr, da_ptr, n_div, pt_ptr, py_ptr, n_prop)
            rho = (p_r_up - p_r_dn) / (2.0 * dr)

            if T > dt_th + 1e-8:
                if n_div > 0:
                    dt_sh = <double *>malloc(n_div * sizeof(double))
                    for k in range(n_div):
                        dt_sh[k] = dt_ptr[k] - dt_th
                if n_prop > 0:
                    pt_sh = <double *>malloc(n_prop * sizeof(double))
                    for k in range(n_prop):
                        pt_sh[k] = pt_ptr[k] - dt_th
                p_t_dn = _crr_disc_smoothed(
                    S, K, r, q, T - dt_th, sigma, is_call, steps, dt_sh, da_ptr, n_div, pt_sh, py_ptr, n_prop
                )
                theta = (p_t_dn - price) / dt_th
                if dt_sh != NULL:
                    free(dt_sh)
                if pt_sh != NULL:
                    free(pt_sh)
        else:
            delta = (1.0 if S > K else 0.0) if is_call else (-1.0 if K > S else 0.0)

    return (price, delta, gamma, vega, theta, rho)


cpdef double corrado_su_price(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    double kappa3,
    double kappa4,
    bint is_call=True,
) noexcept nogil:
    """Modelo de Corrado & Su (1996) com expansão de Gram-Charlier (assimetria e curtose)."""
    if T <= 0.0 or sigma <= 0.0:
        return fmax(S - K, 0.0) if is_call else fmax(K - S, 0.0)

    cdef double price_bs, delta, gamma, vega, theta, rho, vanna, volga
    _bsm_core(S, K, r, q, T, sigma, True, &price_bs, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga)

    cdef double df_q = exp(-q * T)
    cdef double df_r = exp(-r * T)
    cdef double sig_sqrt_T = sigma * sqrt(T)
    cdef double sig2_T = sigma * sigma * T
    cdef double exc_kurt = kappa4 - 3.0
    cdef double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sig_sqrt_T
    cdef double d2 = d1 - sig_sqrt_T
    cdef double nd1 = _norm_pdf(d1)
    cdef double Nd1 = _norm_cdf(d1)

    cdef double q3 = (1.0 / 6.0) * S * df_q * sig_sqrt_T * ((2.0 * sig_sqrt_T - d1) * nd1 + sig2_T * Nd1)
    cdef double q4 = (1.0 / 24.0) * S * df_q * sig_sqrt_T * (
        (d1 * d1 - 1.0 - 3.0 * sig_sqrt_T * d2) * nd1 + (sig2_T * sig_sqrt_T) * Nd1
    )

    cdef double call_cs = price_bs + kappa3 * q3 + exc_kurt * q4
    if call_cs < 0.0:
        call_cs = 0.0
    if is_call:
        return call_cs

    cdef double put_cs = call_cs - S * df_q + K * df_r
    return put_cs if put_cs > 0.0 else 0.0


cpdef double implied_vol_bsm(
    double target_price,
    double S,
    double K,
    double r,
    double q,
    double T,
    bint is_call=True,
    double tol=1e-9,
    int max_iter=100,
) noexcept nogil:
    """Inversão de volatilidade implícita via Newton-Raphson com tolerância relativa e bisseção segura."""
    if T <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0

    cdef double df_q = exp(-q * T)
    cdef double df_r = exp(-r * T)
    cdef double intrinsic = fmax(S * df_q - K * df_r, 0.0) if is_call else fmax(K * df_r - S * df_q, 0.0)
    cdef double upper_bound = (S * df_q) if is_call else (K * df_r)
    if intrinsic > 0.0:
        if target_price <= intrinsic + 1e-12 * fmax(1.0, intrinsic):
            return NAN
    else:
        if target_price <= 0.0:
            return NAN
    if target_price >= upper_bound - 1e-12 * fmax(1.0, upper_bound):
        return NAN

    cdef double low = 1e-6
    cdef double high = 10.0
    cdef double price, delta, gamma, vega, theta, rho, vanna, volga, diff
    _bsm_core(S, K, r, q, T, low, is_call, &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga)
    if target_price < price:
        return NAN
    _bsm_core(S, K, r, q, T, high, is_call, &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga)
    if target_price > price:
        return NAN

    cdef double price_tol = fmax(1e-300, tol * fmin(1.0, target_price))
    cdef double sigma = 0.25
    cdef int it

    for it in range(max_iter):
        _bsm_core(S, K, r, q, T, sigma, is_call, &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga)
        diff = price - target_price
        if fabs(diff) <= price_tol:
            return sigma
        if diff < 0.0 and sigma > low:
            low = sigma
        elif diff > 0.0 and sigma < high:
            high = sigma

        if vega > 1e-300:
            sigma = sigma - diff / vega
            if fabs(diff) <= tol * vega and fabs(diff) <= 1e-6 * fmax(target_price, 1e-300):
                return sigma if (low <= sigma <= high) else 0.5 * (low + high)
            if sigma <= low or sigma >= high:
                sigma = 0.5 * (low + high)
        else:
            sigma = 0.5 * (low + high)
        if (high - low) < tol * 0.1:
            return 0.5 * (low + high)

    for it in range(80):
        sigma = 0.5 * (low + high)
        _bsm_core(S, K, r, q, T, sigma, is_call, &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga)
        diff = price - target_price
        if fabs(diff) <= price_tol or (high - low) < tol * 0.1:
            return sigma
        if diff < 0.0:
            low = sigma
        else:
            high = sigma
    return 0.5 * (low + high)


cpdef double implied_vol_black76(
    double target_price,
    double F,
    double K,
    double r,
    double T,
    bint is_call=True,
    double tol=1e-9,
    int max_iter=100,
) noexcept nogil:
    return implied_vol_bsm(target_price, F, K, r, r, T, is_call, tol, max_iter)


cpdef double sabr_normal_implied_vol(
    double F,
    double K,
    double T,
    double alpha,
    double beta,
    double rho,
    double nu,
) noexcept nogil:
    """Fórmula assintótica de Hagan et al. (2002) para volatilidade implícita Normal (Bachelier) no modelo SABR."""
    if T <= 0.0 or alpha <= 0.0:
        return 0.0
    if beta > 0.0 and (F <= 0.0 or K <= 0.0):
        return 0.0

    cdef double rho_clamped = rho
    if rho_clamped > 0.999999:
        rho_clamped = 0.999999
    elif rho_clamped < -0.999999:
        rho_clamped = -0.999999

    cdef double term1, term2, term3, fk_pow, pref, z, x_z, factor_b, arg_log
    cdef double one_minus_beta = 1.0 - beta
    cdef double f1b, k1b, int_inv_u, log_fk

    if fabs(F - K) < 1e-12:
        if fabs(beta) < 1e-12:
            return alpha * (1.0 + ((2.0 - 3.0 * rho_clamped * rho_clamped) * nu * nu / 24.0) * T)
        fk_pow = pow(F, one_minus_beta)
        term1 = (-beta * (2.0 - beta) * alpha * alpha) / (24.0 * fk_pow * fk_pow)
        term2 = (rho_clamped * beta * nu * alpha) / (4.0 * fk_pow)
        term3 = ((2.0 - 3.0 * rho_clamped * rho_clamped) * nu * nu) / 24.0
        return alpha * pow(F, beta) * (1.0 + (term1 + term2 + term3) * T)

    if fabs(beta) < 1e-12:
        pref = alpha
        z = (nu / alpha) * (F - K)
        arg_log = fmax(1e-15, (sqrt(1.0 - 2.0 * rho_clamped * z + z * z) + z - rho_clamped) / (1.0 - rho_clamped))
        x_z = log(arg_log)
        factor_b = z / x_z if fabs(x_z) > 1e-12 else 1.0
        term3 = ((2.0 - 3.0 * rho_clamped * rho_clamped) * nu * nu) / 24.0
        return pref * factor_b * (1.0 + term3 * T)

    if fabs(one_minus_beta) < 1e-12:
        log_fk = log(F / K)
        pref = alpha * (F - K) / log_fk
        z = (nu / alpha) * log_fk
        fk_pow = 1.0
    else:
        f1b = pow(F, one_minus_beta)
        k1b = pow(K, one_minus_beta)
        int_inv_u = (f1b - k1b) / one_minus_beta
        pref = alpha * (F - K) / int_inv_u
        z = (nu / alpha) * int_inv_u
        fk_pow = pow(F * K, 0.5 * one_minus_beta)

    arg_log = fmax(1e-15, (sqrt(1.0 - 2.0 * rho_clamped * z + z * z) + z - rho_clamped) / (1.0 - rho_clamped))
    x_z = log(arg_log)
    factor_b = z / x_z if fabs(x_z) > 1e-12 else 1.0

    term1 = (-beta * (2.0 - beta) * alpha * alpha) / (24.0 * fk_pow * fk_pow)
    term2 = (rho_clamped * beta * nu * alpha) / (4.0 * fk_pow)
    term3 = ((2.0 - 3.0 * rho_clamped * rho_clamped) * nu * nu) / 24.0
    return pref * factor_b * (1.0 + (term1 + term2 + term3) * T)


cpdef double sabr_alpha_from_atm_vol(
    double F,
    double T,
    double sigma_atm,
    double beta,
    double rho,
    double nu,
    bint is_normal=False,
) noexcept nogil:
    """Resolve o parâmetro alpha de Hagan et al. (2002) tal que sigma_SABR(F, F) == sigma_atm."""
    if sigma_atm <= 0.0:
        return 1e-8
    cdef double rho_c = rho
    if rho_c > 0.999999:
        rho_c = 0.999999
    elif rho_c < -0.999999:
        rho_c = -0.999999

    cdef double one_minus_beta = 1.0 - beta
    cdef double fk_pow = 1.0 if fabs(one_minus_beta) < 1e-12 else pow(fmax(1e-12, F), one_minus_beta)
    cdef double f_scale = pow(fmax(1e-12, F), beta) if is_normal else (1.0 / fk_pow)
    cdef double term3 = ((2.0 - 3.0 * rho_c * rho_c) * nu * nu) / 24.0

    if is_normal and fabs(beta) < 1e-12:
        return sigma_atm / fmax(1e-12, 1.0 + term3 * T)

    if T <= 0.0:
        return sigma_atm / fmax(1e-12, f_scale)

    cdef double c1 = (
        (-beta * (2.0 - beta)) / (24.0 * fk_pow * fk_pow)
        if is_normal
        else (one_minus_beta * one_minus_beta) / (24.0 * fk_pow * fk_pow)
    )
    cdef double c2 = (rho_c * beta * nu) / (4.0 * fk_pow)
    cdef double alpha = sigma_atm / fmax(1e-12, f_scale)
    cdef double fx, dfx, step, corr
    cdef int it

    for it in range(25):
        corr = 1.0 + (c1 * alpha * alpha + c2 * alpha + term3) * T
        fx = f_scale * alpha * corr - sigma_atm
        if fabs(fx) < 1e-14 * fmax(1.0, sigma_atm):
            return alpha
        dfx = f_scale * (1.0 + (3.0 * c1 * alpha * alpha + 2.0 * c2 * alpha + term3) * T)
        if fabs(dfx) < 1e-15:
            break
        step = fx / dfx
        alpha = fmax(1e-12, alpha - step)
        if fabs(step) < 1e-14 * fmax(1.0, alpha):
            return alpha
    return alpha


cpdef double svi_total_variance(
    double k,
    double a,
    double b,
    double rho,
    double m,
    double sigma,
) noexcept nogil:
    """Parametrização SVI (Gatheral / Manual B3): w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    cdef double dx = k - m
    return a + b * (rho * dx + sqrt(dx * dx + sigma * sigma))


cpdef np.ndarray[np.float64_t, ndim=1] svi_eval_vec(
    double[:] k_arr,
    double a,
    double b,
    double rho,
    double m,
    double sigma,
):
    cdef Py_ssize_t n = k_arr.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[:] out_v = out
    with nogil:
        for i in range(n):
            out_v[i] = svi_total_variance(k_arr[i], a, b, rho, m, sigma)
    return out


cpdef double sabr_implied_vol(
    double F,
    double K,
    double T,
    double alpha,
    double beta,
    double rho,
    double nu,
) noexcept nogil:
    """Fórmula assintótica de Hagan et al. (2002) para volatilidade implícita SABR."""
    if F <= 0.0 or K <= 0.0 or T <= 0.0 or alpha <= 0.0:
        return 0.0

    cdef double one_minus_beta = 1.0 - beta
    cdef double fk_pow, log_fk, z, x_z, factor_a, factor_b, term1, term2, term3

    if fabs(F - K) < 1e-12:
        fk_pow = pow(F, one_minus_beta)
        term1 = (one_minus_beta * one_minus_beta * alpha * alpha) / (24.0 * fk_pow * fk_pow)
        term2 = (rho * beta * nu * alpha) / (4.0 * fk_pow)
        term3 = ((2.0 - 3.0 * rho * rho) * nu * nu) / 24.0
        return (alpha / fk_pow) * (1.0 + (term1 + term2 + term3) * T)

    log_fk = log(F / K)
    fk_pow = pow(F * K, 0.5 * one_minus_beta)
    z = (nu / alpha) * fk_pow * log_fk
    cdef double rho_clamped = rho
    if rho_clamped > 0.999999:
        rho_clamped = 0.999999
    elif rho_clamped < -0.999999:
        rho_clamped = -0.999999
    cdef double arg_log = fmax(1e-15, (sqrt(1.0 - 2.0 * rho_clamped * z + z * z) + z - rho_clamped) / (1.0 - rho_clamped))
    x_z = log(arg_log)

    cdef double log2 = log_fk * log_fk
    cdef double log4 = log2 * log2
    cdef double denom_fk = fk_pow * (
        1.0 + (one_minus_beta * one_minus_beta / 24.0) * log2
        + (pow(one_minus_beta, 4.0) / 1920.0) * log4
    )
    factor_a = alpha / denom_fk
    factor_b = z / x_z if fabs(x_z) > 1e-12 else 1.0

    term1 = (one_minus_beta * one_minus_beta * alpha * alpha) / (24.0 * fk_pow * fk_pow)
    term2 = (rho * beta * nu * alpha) / (4.0 * fk_pow)
    term3 = ((2.0 - 3.0 * rho * rho) * nu * nu) / 24.0
    return factor_a * factor_b * (1.0 + (term1 + term2 + term3) * T)


cpdef np.ndarray[np.float64_t, ndim=1] sabr_eval_vec(
    double F,
    double[:] K_arr,
    double T,
    double alpha,
    double beta,
    double rho,
    double nu,
    bint is_normal=False,
):
    cdef Py_ssize_t n = K_arr.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[:] out_v = out
    with nogil:
        for i in range(n):
            if is_normal:
                out_v[i] = sabr_normal_implied_vol(F, K_arr[i], T, alpha, beta, rho, nu)
            else:
                out_v[i] = sabr_implied_vol(F, K_arr[i], T, alpha, beta, rho, nu)
    return out


cpdef tuple price_options_batch(
    int[:] is_call_arr,
    int[:] is_american_arr,
    double[:] S_arr,
    double[:] K_arr,
    double[:] r_arr,
    double[:] q_arr,
    double[:] T_arr,
    double[:] sigma_arr,
    int crr_steps=150,
):
    """Precifica um lote de opções em Cython, retornando preços e gregas."""
    cdef Py_ssize_t n = S_arr.shape[0]
    cdef Py_ssize_t i

    cdef np.ndarray[np.float64_t, ndim=1] out_price = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_delta = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_gamma = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_vega = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_theta = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_rho = np.empty(n, dtype=np.float64)

    cdef double[:] pv = out_price
    cdef double[:] dv = out_delta
    cdef double[:] gv = out_gamma
    cdef double[:] vv = out_vega
    cdef double[:] tv = out_theta
    cdef double[:] rv = out_rho

    cdef double price, delta, gamma, vega, theta, rho, vanna, volga
    cdef double d_tmp, g_tmp, t_tmp, p_v_up, p_v_dn, p_r_up, p_r_dn
    cdef double p1_am, d1_am, g1_am, t1_am, p2_am, d2_am, g2_am, t2_am
    cdef double d_vol = 1e-4
    cdef double d_rate = 1e-4

    cdef bint call_flag
    with nogil:
        for i in range(n):
            call_flag = (is_call_arr[i] != 0)
            if is_american_arr[i]:
                if crr_steps >= 2:
                    p1_am = _crr_american_core(
                        S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i],
                        call_flag, crr_steps, &d1_am, &g1_am, &t1_am,
                    )
                    p2_am = _crr_american_core(
                        S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i],
                        call_flag, crr_steps + 1, &d2_am, &g2_am, &t2_am,
                    )
                    pv[i] = 0.5 * (p1_am + p2_am)
                    dv[i] = 0.5 * (d1_am + d2_am)
                    gv[i] = 0.5 * (g1_am + g2_am)
                    tv[i] = 0.5 * (t1_am + t2_am)
                else:
                    pv[i] = _crr_american_core(
                        S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i],
                        call_flag, crr_steps, &delta, &gamma, &theta,
                    )
                    dv[i] = delta
                    gv[i] = gamma
                    tv[i] = theta
                if T_arr[i] > 0.0 and sigma_arr[i] > d_vol and S_arr[i] > 0.0 and K_arr[i] > 0.0:
                    p_v_up = _crr_american_core(S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i] + d_vol, call_flag, crr_steps, &d_tmp, &g_tmp, &t_tmp)
                    p_v_dn = _crr_american_core(S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i] - d_vol, call_flag, crr_steps, &d_tmp, &g_tmp, &t_tmp)
                    if crr_steps >= 2:
                        p_v_up = 0.5 * (p_v_up + _crr_american_core(S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i] + d_vol, call_flag, crr_steps + 1, &d_tmp, &g_tmp, &t_tmp))
                        p_v_dn = 0.5 * (p_v_dn + _crr_american_core(S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i] - d_vol, call_flag, crr_steps + 1, &d_tmp, &g_tmp, &t_tmp))
                    vv[i] = (p_v_up - p_v_dn) / (2.0 * d_vol)
                    p_r_up = _crr_american_core(S_arr[i], K_arr[i], r_arr[i] + d_rate, q_arr[i], T_arr[i], sigma_arr[i], call_flag, crr_steps, &d_tmp, &g_tmp, &t_tmp)
                    p_r_dn = _crr_american_core(S_arr[i], K_arr[i], r_arr[i] - d_rate, q_arr[i], T_arr[i], sigma_arr[i], call_flag, crr_steps, &d_tmp, &g_tmp, &t_tmp)
                    if crr_steps >= 2:
                        p_r_up = 0.5 * (p_r_up + _crr_american_core(S_arr[i], K_arr[i], r_arr[i] + d_rate, q_arr[i], T_arr[i], sigma_arr[i], call_flag, crr_steps + 1, &d_tmp, &g_tmp, &t_tmp))
                        p_r_dn = 0.5 * (p_r_dn + _crr_american_core(S_arr[i], K_arr[i], r_arr[i] - d_rate, q_arr[i], T_arr[i], sigma_arr[i], call_flag, crr_steps + 1, &d_tmp, &g_tmp, &t_tmp))
                    rv[i] = (p_r_up - p_r_dn) / (2.0 * d_rate)
                else:
                    vv[i] = 0.0
                    rv[i] = 0.0
            else:
                _bsm_core(
                    S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i],
                    call_flag,
                    &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga,
                )
                pv[i] = price
                dv[i] = delta
                gv[i] = gamma
                vv[i] = vega
                tv[i] = theta
                rv[i] = rho

    return (out_price, out_delta, out_gamma, out_vega, out_theta, out_rho)


cdef void _bachelier_core(
    double F,
    double K,
    double df_r,
    double T,
    double sigma_n,
    bint is_call,
    double* out_price,
    double* out_delta,
    double* out_gamma,
    double* out_vega,
    double* out_theta,
) noexcept nogil:
    if T <= 0.0 or sigma_n <= 0.0:
        if is_call:
            out_price[0] = df_r * fmax(F - K, 0.0)
            out_delta[0] = df_r if F > K else 0.0
        else:
            out_price[0] = df_r * fmax(K - F, 0.0)
            out_delta[0] = -df_r if K > F else 0.0
        out_gamma[0] = 0.0
        out_vega[0] = 0.0
        out_theta[0] = 0.0
        return

    cdef double sqrt_T = sqrt(T)
    cdef double std_dev = sigma_n * sqrt_T
    cdef double d = (F - K) / std_dev
    cdef double pdf_d = _norm_pdf(d)
    cdef double cdf_d = _norm_cdf(d)

    if is_call:
        out_price[0] = df_r * ((F - K) * cdf_d + std_dev * pdf_d)
        out_delta[0] = df_r * cdf_d
    else:
        out_price[0] = df_r * ((K - F) * (1.0 - cdf_d) + std_dev * pdf_d)
        out_delta[0] = -df_r * (1.0 - cdf_d)

    out_gamma[0] = df_r * pdf_d / std_dev
    out_vega[0] = df_r * sqrt_T * pdf_d
    out_theta[0] = -df_r * sigma_n * pdf_d / (2.0 * sqrt_T)


cpdef tuple bachelier_price_greeks(
    double F,
    double K,
    double df_r,
    double T,
    double sigma_n,
    bint is_call=True,
):
    """Modelo Normal de Bachelier para taxas/swaptions/caplets."""
    cdef double price, delta, gamma, vega, theta
    with nogil:
        _bachelier_core(F, K, df_r, T, sigma_n, is_call, &price, &delta, &gamma, &vega, &theta)
    return (price, delta, gamma, vega, theta)


cpdef double implied_vol_bachelier(
    double target_price,
    double F,
    double K,
    double df_r,
    double T,
    bint is_call=True,
    double tol=1e-9,
    int max_iter=100,
) noexcept nogil:
    """Inversão de volatilidade normal (Bachelier) via Newton-Raphson com fallback por bisseção."""
    if T <= 0.0 or df_r <= 0.0:
        return 0.0
    cdef double intrinsic = df_r * (fmax(F - K, 0.0) if is_call else fmax(K - F, 0.0))
    if target_price < intrinsic - 1e-10:
        return NAN
    if target_price <= intrinsic + 1e-12:
        return 0.0

    cdef double sqrt_T = sqrt(T)
    cdef double time_val = fmax(1e-15, target_price - intrinsic)
    cdef double sigma_0 = fmax(
        1e-4,
        time_val / fmax(1e-15, df_r * sqrt_T * INV_SQRT_2PI) + 0.35 * fabs(F - K) / sqrt_T,
    )
    cdef double low = 1e-10
    cdef double high = fmax(1.0, 2.0 * sigma_0)
    cdef double price, delta, gamma, vega, theta, diff
    cdef int it

    for it in range(60):
        _bachelier_core(F, K, df_r, T, high, is_call, &price, &delta, &gamma, &vega, &theta)
        if price >= target_price or high > 1e12:
            break
        high *= 2.0

    cdef double sigma_n = sigma_0
    if sigma_n <= low or sigma_n >= high:
        sigma_n = 0.5 * (low + high)
    cdef double eff_tol = fmin(tol, 1e-10 * fmax(1.0, target_price))

    for it in range(max_iter):
        _bachelier_core(F, K, df_r, T, sigma_n, is_call, &price, &delta, &gamma, &vega, &theta)
        diff = price - target_price
        if fabs(diff) < eff_tol:
            return sigma_n
        if diff < 0.0 and sigma_n > low:
            low = sigma_n
        elif diff > 0.0 and sigma_n < high:
            high = sigma_n
        if vega > 1e-15:
            sigma_n = sigma_n - diff / vega
            if sigma_n <= low or sigma_n >= high:
                sigma_n = 0.5 * (low + high)
        else:
            sigma_n = 0.5 * (low + high)

    for it in range(80):
        sigma_n = 0.5 * (low + high)
        _bachelier_core(F, K, df_r, T, sigma_n, is_call, &price, &delta, &gamma, &vega, &theta)
        diff = price - target_price
        if fabs(diff) < eff_tol or (high - low) < 1e-10 * fmax(1.0, sigma_n):
            return sigma_n
        if diff < 0.0:
            low = sigma_n
        else:
            high = sigma_n
    return 0.5 * (low + high)


cpdef double barrier_bsm_price(
    double S,
    double K,
    double H,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call=True,
    bint is_up=False,
    bint is_out=True,
    double rebate=0.0,
) noexcept nogil:
    """
    Precificação analítica de opções europeias com barreira contínua (Reiner & Rubinstein, 1991).
    Suporta down-and-out, down-and-in, up-and-out e up-and-in (call e put) com rebate opcional.
    """
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0 or H <= 0.0:
        return 0.0

    cdef double df_r = exp(-r * T)
    cdef double df_q = exp(-q * T)
    cdef double p_bsm, d_tmp, g_tmp, v_tmp, th_tmp, rh_tmp, va_tmp, vo_tmp

    # Verifica violação imediata da barreira na data-base
    if (is_up and S >= H) or ((not is_up) and S <= H):
        if is_out:
            return rebate
        _bsm_core(S, K, r, q, T, sigma, is_call, &p_bsm, &d_tmp, &g_tmp, &v_tmp, &th_tmp, &rh_tmp, &va_tmp, &vo_tmp)
        return p_bsm

    cdef double b = r - q
    cdef double sig2 = sigma * sigma
    cdef double sig_sqrt_T = sigma * sqrt(T)
    cdef double mu = (b - 0.5 * sig2) / sig2
    cdef double lam = sqrt(mu * mu + 2.0 * r / sig2)

    cdef double x1 = log(S / K) / sig_sqrt_T + (1.0 + mu) * sig_sqrt_T
    cdef double x2 = log(S / H) / sig_sqrt_T + (1.0 + mu) * sig_sqrt_T
    cdef double y1 = log((H * H) / (S * K)) / sig_sqrt_T + (1.0 + mu) * sig_sqrt_T
    cdef double y2 = log(H / S) / sig_sqrt_T + (1.0 + mu) * sig_sqrt_T
    cdef double z = log(H / S) / sig_sqrt_T + lam * sig_sqrt_T

    cdef double phi = 1.0 if is_call else -1.0
    cdef double eta = -1.0 if is_up else 1.0

    cdef double A = phi * (S * df_q * _norm_cdf(phi * x1) - K * df_r * _norm_cdf(phi * (x1 - sig_sqrt_T)))
    cdef double B = phi * (S * df_q * _norm_cdf(phi * x2) - K * df_r * _norm_cdf(phi * (x2 - sig_sqrt_T)))
    cdef double C = phi * (
        S * df_q * pow(H / S, 2.0 * (mu + 1.0)) * _norm_cdf(eta * y1)
        - K * df_r * pow(H / S, 2.0 * mu) * _norm_cdf(eta * (y1 - sig_sqrt_T))
    )
    cdef double D = phi * (
        S * df_q * pow(H / S, 2.0 * (mu + 1.0)) * _norm_cdf(eta * y2)
        - K * df_r * pow(H / S, 2.0 * mu) * _norm_cdf(eta * (y2 - sig_sqrt_T))
    )
    cdef double E = rebate * df_r * (
        _norm_cdf(eta * (x2 - sig_sqrt_T)) - pow(H / S, 2.0 * mu) * _norm_cdf(eta * (y2 - sig_sqrt_T))
    )
    cdef double F = rebate * (
        pow(H / S, mu + lam) * _norm_cdf(eta * z)
        + pow(H / S, mu - lam) * _norm_cdf(eta * (z - 2.0 * lam * sig_sqrt_T))
    )

    cdef double val = 0.0
    if is_call and (not is_up):  # Down Call
        if is_out:
            val = (A - C + F) if K >= H else (B - D + F)
        else:
            val = (C + E) if K >= H else (A - B + D + E)
    elif is_call and is_up:  # Up Call
        if is_out:
            val = F if K >= H else (A - B + C - D + F)
        else:
            val = (A + E) if K >= H else (B - C + D + E)
    elif (not is_call) and (not is_up):  # Down Put
        if is_out:
            val = (A - B + C - D + F) if K >= H else F
        else:
            val = (B - C + D + E) if K >= H else (A + E)
    else:  # Up Put
        if is_out:
            val = (B - D + F) if K >= H else (A - C + F)
        else:
            val = (A - B + D + E) if K >= H else (C + E)

    return val if val > 0.0 else 0.0


cpdef double asian_geometric_price(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call=True,
    int n_fixings=0,
) noexcept nogil:
    """
    Opção asiática de média geométrica (Kemna & Vorst, 1990).
    Quando `n_fixings > 0`, utiliza momentos exatos para fixings discretos em (0, T].
    """
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return fmax(S - K, 0.0) if is_call else fmax(K - S, 0.0)
    cdef double b = r - q
    cdef double sig2 = sigma * sigma
    cdef double m_f, t_bar, var_sum, sig_g, b_g, q_g
    if n_fixings > 0:
        m_f = <double>n_fixings
        t_bar = T * (m_f + 1.0) / (2.0 * m_f)
        var_sum = sig2 * T * (m_f + 1.0) * (2.0 * m_f + 1.0) / (6.0 * m_f * m_f)
        sig_g = sqrt(var_sum / T)
        b_g = ((b - 0.5 * sig2) * t_bar + 0.5 * var_sum) / T
        q_g = r - b_g
    else:
        sig_g = sigma / sqrt(3.0)
        b_g = 0.5 * (b - sig2 / 6.0)
        q_g = r - b_g
    cdef double price, d_tmp, g_tmp, v_tmp, th_tmp, rh_tmp, va_tmp, vo_tmp
    _bsm_core(S, K, r, q_g, T, sig_g, is_call, &price, &d_tmp, &g_tmp, &v_tmp, &th_tmp, &rh_tmp, &va_tmp, &vo_tmp)
    return price


cpdef double asian_arithmetic_tw_price(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call=True,
    int n_fixings=0,
) noexcept nogil:
    """
    Opção asiática de média aritmética recém-iniciada via aproximação lognormal de dois momentos
    (Levy, 1992 / Turnbull & Wakeman, 1991). Quando `n_fixings > 0`, calcula M1 e M2 para
    observações discretas igualmente espaçadas em (0, T].
    """
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return fmax(S - K, 0.0) if is_call else fmax(K - S, 0.0)

    cdef double b = r - q
    cdef double sig2 = sigma * sigma
    cdef double m1 = 0.0
    cdef double m2 = 0.0
    cdef double dt_fix, t_i, s_i, suffix_sum, sum_m1, sum_m2
    cdef int i

    if n_fixings > 0:
        dt_fix = T / (<double>n_fixings)
        suffix_sum = 0.0
        sum_m1 = 0.0
        sum_m2 = 0.0
        for i in range(n_fixings, 0, -1):
            t_i = (<double>i) * dt_fix
            s_i = exp(b * t_i)
            sum_m1 += s_i
            sum_m2 += exp((b + sig2) * t_i) * (s_i + 2.0 * suffix_sum)
            suffix_sum += s_i
        m1 = sum_m1 / (<double>n_fixings)
        m2 = sum_m2 / ((<double>n_fixings) * (<double>n_fixings))
    else:
        if fabs(b) < 1e-8:
            m1 = 1.0
            if sig2 * T > 1e-8:
                m2 = (2.0 * exp(sig2 * T) - 2.0 * (1.0 + sig2 * T)) / (sig2 * sig2 * T * T)
            else:
                m2 = 1.0
        elif fabs(b + sig2) < 1e-8:
            m1 = (exp(b * T) - 1.0) / (b * T)
            m2 = (2.0 * (exp(b * T) * (b * T - 1.0) + 1.0)) / (b * b * T * T)
        elif fabs(2.0 * b + sig2) < 1e-8:
            m1 = (exp(b * T) - 1.0) / (b * T)
            m2 = (2.0 / ((b + sig2) * T * T)) * (T - (exp(b * T) - 1.0) / b)
        else:
            m1 = (exp(b * T) - 1.0) / (b * T)
            m2 = (
                (2.0 * exp((2.0 * b + sig2) * T)) / ((b + sig2) * (2.0 * b + sig2) * T * T)
                + (2.0 / (b * T * T)) * ((1.0 / (2.0 * b + sig2)) - (exp(b * T) / (b + sig2)))
            )

    cdef double b_eff = log(fmax(1e-12, m1)) / T
    cdef double var_eff = log(fmax(1.0 + 1e-12, m2 / (m1 * m1)))
    cdef double sig_eff = sqrt(var_eff / T)
    cdef double q_eff = r - b_eff

    cdef double price, d_tmp, g_tmp, v_tmp, th_tmp, rh_tmp, va_tmp, vo_tmp
    _bsm_core(S, K, r, q_eff, T, sig_eff, is_call, &price, &d_tmp, &g_tmp, &v_tmp, &th_tmp, &rh_tmp, &va_tmp, &vo_tmp)
    return price


cpdef tuple price_black76_batch(
    int[:] is_call_arr,
    double[:] F_arr,
    double[:] K_arr,
    double[:] r_arr,
    double[:] T_arr,
    double[:] sigma_arr,
    double[:] scale_arr,
):
    """Precifica em lote opções Black-76 (IDI, DI1, Swaptions) em Cython (`nogil`)."""
    cdef Py_ssize_t n = F_arr.shape[0]
    cdef Py_ssize_t i

    cdef np.ndarray[np.float64_t, ndim=1] out_price = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_delta = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_gamma = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_vega = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_theta = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_rho = np.empty(n, dtype=np.float64)

    cdef double[:] pv = out_price
    cdef double[:] dv = out_delta
    cdef double[:] gv = out_gamma
    cdef double[:] vv = out_vega
    cdef double[:] tv = out_theta
    cdef double[:] rv = out_rho

    cdef double price, delta, gamma, vega, theta, rho, vanna, volga, sc
    cdef bint call_flag

    with nogil:
        for i in range(n):
            call_flag = (is_call_arr[i] != 0)
            sc = scale_arr[i]
            _bsm_core(
                F_arr[i], K_arr[i], r_arr[i], r_arr[i], T_arr[i], sigma_arr[i],
                call_flag, &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga,
            )
            pv[i] = price * sc
            dv[i] = delta * sc
            gv[i] = gamma * sc
            vv[i] = vega * sc
            tv[i] = theta * sc
            rv[i] = rho * sc

    return (out_price, out_delta, out_gamma, out_vega, out_theta, out_rho)


cpdef tuple corrado_su_price_greeks(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    double kappa3,
    double kappa4,
    bint is_call=True,
):
    """Preço e gregas numéricas centrais de Corrado & Su (1996) em Cython (`nogil`)."""
    cdef double price = 0.0, delta = 0.0, gamma = 0.0, vega = 0.0, theta = 0.0, rho = 0.0
    cdef double dS = fmax(1e-5, 1e-4 * S)
    cdef double dv = 1e-4
    cdef double dr = 1e-4
    cdef double dT = fmin(1e-4, 0.1 * T)
    cdef double p_su, p_sd, p_vu, p_vd, p_ru, p_rd, p_td

    with nogil:
        price = corrado_su_price(S, K, r, q, T, sigma, kappa3, kappa4, is_call)
        if T > 0.0 and sigma > dv and S > dS and K > 0.0:
            p_su = corrado_su_price(S + dS, K, r, q, T, sigma, kappa3, kappa4, is_call)
            p_sd = corrado_su_price(S - dS, K, r, q, T, sigma, kappa3, kappa4, is_call)
            delta = (p_su - p_sd) / (2.0 * dS)
            gamma = (p_su - 2.0 * price + p_sd) / (dS * dS)

            p_vu = corrado_su_price(S, K, r, q, T, sigma + dv, kappa3, kappa4, is_call)
            p_vd = corrado_su_price(S, K, r, q, T, sigma - dv, kappa3, kappa4, is_call)
            vega = (p_vu - p_vd) / (2.0 * dv)

            p_ru = corrado_su_price(S, K, r + dr, q, T, sigma, kappa3, kappa4, is_call)
            p_rd = corrado_su_price(S, K, r - dr, q, T, sigma, kappa3, kappa4, is_call)
            rho = (p_ru - p_rd) / (2.0 * dr)

            if T > dT:
                p_td = corrado_su_price(S, K, r, q, T - dT, sigma, kappa3, kappa4, is_call)
                theta = (p_td - price) / dT
        else:
            delta = (1.0 if S > K else 0.0) if is_call else (-1.0 if K > S else 0.0)

    return (price, delta, gamma, vega, theta, rho)


cpdef tuple barrier_bsm_price_greeks(
    double S,
    double K,
    double H,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call=True,
    bint is_up=False,
    bint is_out=True,
    double rebate=0.0,
):
    """Preço e gregas numéricas centrais de opções com barreira (Reiner-Rubinstein) em Cython."""
    cdef double price = 0.0, delta = 0.0, gamma = 0.0, vega = 0.0, theta = 0.0, rho = 0.0
    cdef double dist = fabs(S - H)
    cdef double dS = fmin(1e-4 * S, 0.25 * dist) if dist > 1e-6 else 1e-4 * S
    cdef double dv = 1e-4
    cdef double dr = 1e-4
    cdef double dT = fmin(1e-4, 0.1 * T)
    cdef double p_su, p_sd, p_vu, p_vd, p_ru, p_rd, p_td

    with nogil:
        price = barrier_bsm_price(S, K, H, r, q, T, sigma, is_call, is_up, is_out, rebate)
        if T > 0.0 and sigma > dv and S > dS and K > 0.0 and dist > 1e-6:
            p_su = barrier_bsm_price(S + dS, K, H, r, q, T, sigma, is_call, is_up, is_out, rebate)
            p_sd = barrier_bsm_price(S - dS, K, H, r, q, T, sigma, is_call, is_up, is_out, rebate)
            delta = (p_su - p_sd) / (2.0 * dS)
            gamma = (p_su - 2.0 * price + p_sd) / (dS * dS)

            p_vu = barrier_bsm_price(S, K, H, r, q, T, sigma + dv, is_call, is_up, is_out, rebate)
            p_vd = barrier_bsm_price(S, K, H, r, q, T, sigma - dv, is_call, is_up, is_out, rebate)
            vega = (p_vu - p_vd) / (2.0 * dv)

            p_ru = barrier_bsm_price(S, K, H, r + dr, q, T, sigma, is_call, is_up, is_out, rebate)
            p_rd = barrier_bsm_price(S, K, H, r - dr, q, T, sigma, is_call, is_up, is_out, rebate)
            rho = (p_ru - p_rd) / (2.0 * dr)

            if T > dT:
                p_td = barrier_bsm_price(S, K, H, r, q, T - dT, sigma, is_call, is_up, is_out, rebate)
                theta = (p_td - price) / dT

    return (price, delta, gamma, vega, theta, rho)


cpdef tuple asian_price_greeks(
    double S,
    double K,
    double r,
    double q,
    double T,
    double sigma,
    bint is_call=True,
    int n_fixings=0,
    bint is_geometric=False,
):
    """Preço e gregas centrais em Cython para opções asiáticas geométricas e aritméticas."""
    cdef double price = 0.0, delta = 0.0, gamma = 0.0, vega = 0.0, theta = 0.0, rho = 0.0
    cdef double dS = fmax(1e-5, 1e-4 * S)
    cdef double dv = 1e-4
    cdef double dr = 1e-4
    cdef double dT = fmin(1e-4, 0.1 * T)
    cdef double p_su, p_sd, p_vu, p_vd, p_ru, p_rd, p_td

    with nogil:
        if is_geometric:
            price = asian_geometric_price(S, K, r, q, T, sigma, is_call, n_fixings)
            if T > 0.0 and sigma > dv and S > dS and K > 0.0:
                p_su = asian_geometric_price(S + dS, K, r, q, T, sigma, is_call, n_fixings)
                p_sd = asian_geometric_price(S - dS, K, r, q, T, sigma, is_call, n_fixings)
                p_vu = asian_geometric_price(S, K, r, q, T, sigma + dv, is_call, n_fixings)
                p_vd = asian_geometric_price(S, K, r, q, T, sigma - dv, is_call, n_fixings)
                p_ru = asian_geometric_price(S, K, r + dr, q, T, sigma, is_call, n_fixings)
                p_rd = asian_geometric_price(S, K, r - dr, q, T, sigma, is_call, n_fixings)
                p_td = asian_geometric_price(S, K, r, q, T - dT, sigma, is_call, n_fixings) if T > dT else price
        else:
            price = asian_arithmetic_tw_price(S, K, r, q, T, sigma, is_call, n_fixings)
            if T > 0.0 and sigma > dv and S > dS and K > 0.0:
                p_su = asian_arithmetic_tw_price(S + dS, K, r, q, T, sigma, is_call, n_fixings)
                p_sd = asian_arithmetic_tw_price(S - dS, K, r, q, T, sigma, is_call, n_fixings)
                p_vu = asian_arithmetic_tw_price(S, K, r, q, T, sigma + dv, is_call, n_fixings)
                p_vd = asian_arithmetic_tw_price(S, K, r, q, T, sigma - dv, is_call, n_fixings)
                p_ru = asian_arithmetic_tw_price(S, K, r + dr, q, T, sigma, is_call, n_fixings)
                p_rd = asian_arithmetic_tw_price(S, K, r - dr, q, T, sigma, is_call, n_fixings)
                p_td = asian_arithmetic_tw_price(S, K, r, q, T - dT, sigma, is_call, n_fixings) if T > dT else price

        if T > 0.0 and sigma > dv and S > dS and K > 0.0:
            delta = (p_su - p_sd) / (2.0 * dS)
            gamma = (p_su - 2.0 * price + p_sd) / (dS * dS)
            vega = (p_vu - p_vd) / (2.0 * dv)
            rho = (p_ru - p_rd) / (2.0 * dr)
            if T > dT:
                theta = (p_td - price) / dT

    return (price, delta, gamma, vega, theta, rho)


cpdef double monotone_cubic_eval(
    double[:] x_nodes,
    double[:] y_nodes,
    double x_eval,
) noexcept nogil:
    """Interpolação por spline cúbico monótono de Fritsch-Carlson em C puro."""
    cdef int n = <int>x_nodes.shape[0]
    if n <= 0:
        return NAN
    if n == 1 or x_eval <= x_nodes[0]:
        return y_nodes[0]
    if x_eval >= x_nodes[n - 1]:
        return y_nodes[n - 1]

    cdef double *d = <double *>malloc((n - 1) * sizeof(double))
    cdef double *m = <double *>malloc(n * sizeof(double))
    if d == NULL or m == NULL:
        free(d); free(m)
        return NAN

    cdef int i
    cdef double dx, alpha, beta, tau, denom_fc
    for i in range(n - 1):
        dx = x_nodes[i + 1] - x_nodes[i]
        d[i] = (y_nodes[i + 1] - y_nodes[i]) / dx if fabs(dx) > 1e-15 else 0.0

    m[0] = d[0]
    m[n - 1] = d[n - 2]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] > 0.0:
            m[i] = 0.5 * (d[i - 1] + d[i])
        else:
            m[i] = 0.0

    for i in range(n - 1):
        if fabs(d[i]) < 1e-15:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            alpha = m[i] / d[i]
            beta = m[i + 1] / d[i]
            denom_fc = 3.0 * (alpha + beta - 2.0)
            if not (
                (alpha + beta - 2.0 <= 0.0)
                or (2.0 * alpha + beta - 3.0 <= 0.0)
                or (alpha + 2.0 * beta - 3.0 <= 0.0)
                or (fabs(denom_fc) > 1e-15 and alpha - ((2.0 * alpha + beta - 3.0) * (2.0 * alpha + beta - 3.0)) / denom_fc >= 0.0)
            ):
                tau = 3.0 / sqrt(alpha * alpha + beta * beta)
                m[i] = tau * alpha * d[i]
                m[i + 1] = tau * beta * d[i]

    cdef int lo = 0
    cdef int hi = n - 2
    cdef int mid
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if x_nodes[mid] <= x_eval:
            lo = mid
        else:
            hi = mid - 1

    cdef double h = x_nodes[lo + 1] - x_nodes[lo]
    cdef double t = (x_eval - x_nodes[lo]) / h if fabs(h) > 1e-15 else 0.0
    cdef double t2 = t * t
    cdef double t3 = t2 * t
    cdef double res = (
        y_nodes[lo] * (2.0 * t3 - 3.0 * t2 + 1.0)
        + h * m[lo] * (t3 - 2.0 * t2 + t)
        + y_nodes[lo + 1] * (-2.0 * t3 + 3.0 * t2)
        + h * m[lo + 1] * (t3 - t2)
    )
    free(d)
    free(m)
    return res


cpdef double exp_smile_eval(
    double[:] x_nodes,
    double[:] y_nodes,
    double x_eval,
) noexcept nogil:
    """Interpolação exponencial por partes com busca binária O(log n) em C puro."""
    cdef int n = <int>x_nodes.shape[0]
    if n <= 0:
        return NAN
    if n == 1 or x_eval <= x_nodes[0]:
        return y_nodes[0]
    if x_eval >= x_nodes[n - 1]:
        return y_nodes[n - 1]

    cdef int lo = 0
    cdef int hi = n - 2
    cdef int mid
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if x_nodes[mid] <= x_eval:
            lo = mid
        else:
            hi = mid - 1

    cdef double dx = x_nodes[lo + 1] - x_nodes[lo]
    cdef double w = (x_eval - x_nodes[lo]) / dx if fabs(dx) > 1e-15 else 0.0
    return y_nodes[lo] * pow(y_nodes[lo + 1] / y_nodes[lo], w)


cpdef np.ndarray[np.int64_t, ndim=1] longest_monotonic_subsequence(
    double[:] values,
    bint non_increasing=True,
):
    """Retorna os índices da maior subsequência monótona em C puro preservando o desempate determinístico."""
    cdef int n = <int>values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    cdef int* dp = <int *>malloc(n * sizeof(int))
    cdef int* parent = <int *>malloc(n * sizeof(int))
    if dp == NULL or parent == NULL:
        if dp != NULL:
            free(dp)
        if parent != NULL:
            free(parent)
        raise MemoryError("Falha ao alocar buffers em longest_monotonic_subsequence.")

    cdef int i, j
    cdef int best_len = 1
    cdef int best_idx = 0
    cdef double vi, vj
    cdef bint cond

    with nogil:
        for i in range(n):
            dp[i] = 1
            parent[i] = -1
            vi = values[i]
            for j in range(i):
                vj = values[j]
                if non_increasing:
                    cond = vi <= vj + 1e-12
                else:
                    cond = vi >= vj - 1e-12
                if cond and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
            if dp[i] >= best_len:
                best_len = dp[i]
                best_idx = i

    cdef np.ndarray[np.int64_t, ndim=1] out = np.empty(best_len, dtype=np.int64)
    cdef int cur = best_idx
    cdef int pos = best_len - 1
    while cur != -1 and pos >= 0:
        out[pos] = cur
        cur = parent[cur]
        pos -= 1

    free(dp)
    free(parent)
    return out


cpdef double implied_vol_american_crr(
    double target_price,
    double S,
    double K,
    double r,
    double q,
    double T,
    bint is_call,
    int steps=150,
    object div_times=None,
    object div_amounts=None,
    object prop_div_times=None,
    object prop_div_yields=None,
    double tol=1e-7,
    int max_iter=60,
):
    """Inversão de volatilidade implícita americana via árvore CRR suavizada em Cython."""
    if T <= 0.0 or S <= 0.0 or K <= 0.0 or target_price <= 0.0:
        return NAN

    cdef double intrinsic = fmax(S - K, 0.0) if is_call else fmax(K - S, 0.0)
    cdef double upper_bound = S if is_call else K
    if target_price <= intrinsic + 1e-12 * fmax(1.0, intrinsic):
        return NAN
    if target_price >= upper_bound - 1e-12 * fmax(1.0, upper_bound):
        return NAN

    cdef np.ndarray[np.float64_t, ndim=1] dt_arr
    cdef np.ndarray[np.float64_t, ndim=1] da_arr
    cdef np.ndarray[np.float64_t, ndim=1] pt_arr
    cdef np.ndarray[np.float64_t, ndim=1] py_arr
    cdef int n_div = 0
    cdef int n_prop = 0
    cdef double* dt_ptr = NULL
    cdef double* da_ptr = NULL
    cdef double* pt_ptr = NULL
    cdef double* py_ptr = NULL

    if div_times is not None and len(div_times) > 0:
        dt_arr = np.ascontiguousarray(div_times, dtype=np.float64)
        da_arr = np.ascontiguousarray(div_amounts, dtype=np.float64)
        n_div = <int>dt_arr.shape[0]
        if <int>da_arr.shape[0] != n_div:
            raise ValueError("div_times e div_amounts devem possuir o mesmo tamanho.")
        dt_ptr = &dt_arr[0]
        da_ptr = &da_arr[0]

    if prop_div_times is not None and len(prop_div_times) > 0:
        pt_arr = np.ascontiguousarray(prop_div_times, dtype=np.float64)
        py_arr = np.ascontiguousarray(prop_div_yields, dtype=np.float64)
        n_prop = <int>pt_arr.shape[0]
        if <int>py_arr.shape[0] != n_prop:
            raise ValueError("prop_div_times e prop_div_yields devem possuir o mesmo tamanho.")
        pt_ptr = &pt_arr[0]
        py_ptr = &py_arr[0]

    cdef int eff_steps = steps if steps >= 4 else 4
    cdef double dt_step = T / <double>eff_steps
    cdef double vol_floor = fmax(1e-3, fabs(r - q) * sqrt(dt_step) * 1.25)
    cdef double grid[9]
    grid[0] = vol_floor
    grid[1] = fmax(vol_floor * 2.0, 0.02)
    grid[2] = 0.05
    grid[3] = 0.15
    grid[4] = 0.35
    grid[5] = 0.75
    grid[6] = 1.5
    grid[7] = 3.0
    grid[8] = 6.0

    cdef int i, it
    cdef double v_prev = NAN
    cdef double p_prev = NAN
    cdef double v_cur, p_cur
    cdef double v_lo = NAN, v_hi = NAN
    cdef double p_lo_val = NAN, p_hi_val = NAN
    cdef bint first_valid = False
    cdef double first_p = NAN
    cdef double v_mid, p_mid, f_lo, f_hi, f_mid, v_sec

    with nogil:
        for i in range(9):
            v_cur = grid[i]
            if v_cur < vol_floor:
                continue
            if first_valid and v_cur <= v_prev + 1e-12:
                continue
            p_cur = _crr_disc_smoothed(
                S, K, r, q, T, v_cur, is_call, eff_steps,
                dt_ptr, da_ptr, n_div, pt_ptr, py_ptr, n_prop,
            )
            if isnan(p_cur):
                continue
            if not first_valid:
                first_valid = True
                first_p = p_cur
                if target_price < first_p - 1e-12:
                    return NAN
                if fabs(p_cur - target_price) <= tol:
                    return v_cur
            else:
                if (p_prev - target_price) * (p_cur - target_price) <= 0.0:
                    v_lo = v_prev
                    p_lo_val = p_prev
                    v_hi = v_cur
                    p_hi_val = p_cur
                    break
            v_prev = v_cur
            p_prev = p_cur

        if isnan(v_lo) or isnan(v_hi):
            return NAN

        f_lo = p_lo_val - target_price
        f_hi = p_hi_val - target_price
        if fabs(f_lo) <= tol:
            return v_lo
        if fabs(f_hi) <= tol:
            return v_hi

        for it in range(max_iter):
            if fabs(f_hi - f_lo) > 1e-15:
                v_sec = v_hi - f_hi * (v_hi - v_lo) / (f_hi - f_lo)
            else:
                v_sec = 0.5 * (v_lo + v_hi)

            # Mantém passo estritamente interior para evitar estagnação de secante
            if v_sec <= v_lo + 0.05 * (v_hi - v_lo) or v_sec >= v_hi - 0.05 * (v_hi - v_lo):
                v_mid = 0.5 * (v_lo + v_hi)
            else:
                v_mid = v_sec

            p_mid = _crr_disc_smoothed(
                S, K, r, q, T, v_mid, is_call, eff_steps,
                dt_ptr, da_ptr, n_div, pt_ptr, py_ptr, n_prop,
            )
            if isnan(p_mid):
                v_mid = 0.5 * (v_lo + v_hi)
                p_mid = _crr_disc_smoothed(
                    S, K, r, q, T, v_mid, is_call, eff_steps,
                    dt_ptr, da_ptr, n_div, pt_ptr, py_ptr, n_prop,
                )
                if isnan(p_mid):
                    return NAN

            f_mid = p_mid - target_price
            if fabs(f_mid) <= tol or (v_hi - v_lo) <= tol:
                return v_mid
            if (f_lo < 0.0 and f_mid < 0.0) or (f_lo > 0.0 and f_mid > 0.0):
                v_lo = v_mid
                f_lo = f_mid
            else:
                v_hi = v_mid
                f_hi = f_mid

        return 0.5 * (v_lo + v_hi)


cpdef np.ndarray[np.float64_t, ndim=1] bloomberg_mixture_prices_vec(
    double[:] strikes,
    np.ndarray[np.uint8_t, ndim=1] is_call,
    double F,
    double r,
    double T,
    double[:] p_weights,
    double[:] xis,
    double[:] sigmas_total,
    double q_default,
    double beta_sigma=1.0,
):
    """Avaliação vetorizada em C puro dos prêmios europeus sob a mistura de lognormais com default."""
    cdef int m = <int>strikes.shape[0]
    cdef int n = <int>p_weights.shape[0]
    if <int>is_call.shape[0] != m or <int>xis.shape[0] != n or <int>sigmas_total.shape[0] != n:
        raise ValueError("Dimensões incompatíveis em bloomberg_mixture_prices_vec.")

    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(m, dtype=np.float64)
    cdef double t_eff = T if T > 1e-8 else 1e-8
    cdef double sqrt_T = sqrt(t_eff)
    cdef double df_r = exp(-r * t_eff)
    cdef int i, l
    cdef bint ic
    cdef double kf, tot, vol_l, price, delta, gamma, vega, theta, rho, vanna, volga

    with nogil:
        for i in range(m):
            kf = strikes[i]
            ic = is_call[i] != 0
            tot = 0.0
            for l in range(n):
                vol_l = (beta_sigma * sigmas_total[l]) / sqrt_T
                if vol_l < 1e-8:
                    vol_l = 1e-8
                _bsm_core(
                    xis[l] * F, kf, r, r, t_eff, vol_l, ic,
                    &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga,
                )
                tot += p_weights[l] * price
            if (not ic) and q_default > 0.0:
                tot += df_r * q_default * kf
            out[i] = tot
    return out


cpdef tuple bloomberg_mixture_prices_and_jac(
    double[:] strikes,
    np.ndarray[np.uint8_t, ndim=1] is_call,
    double F,
    double r,
    double T,
    double[:] vec,
    int n_components,
    bint allow_default,
    double fixed_q_default=0.0,
):
    """Avalia prêmios e Jacobiano analítico exato da mistura de lognormais em relação a (u, eta, Sigma, Q)."""
    cdef int n = n_components
    cdef int m = <int>strikes.shape[0]
    cdef int n_params = (3 * n - 1) if allow_default else (3 * n - 2)
    if n < 2 or <int>vec.shape[0] != n_params or <int>is_call.shape[0] != m:
        raise ValueError("Dimensões incompatíveis em bloomberg_mixture_prices_and_jac.")

    cdef np.ndarray[np.float64_t, ndim=1] prices_out = np.empty(m, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] jac_out = np.empty((m, n_params), dtype=np.float64)

    cdef double t_eff = T if T > 1e-8 else 1e-8
    cdef double sqrt_T = sqrt(t_eff)
    cdef double inv_sqrt_T = 1.0 / sqrt_T
    cdef double df_r = exp(-r * t_eff)

    cdef double* logits = <double *>malloc(n * sizeof(double))
    cdef double* s_w = <double *>malloc(n * sizeof(double))
    cdef double* p_w = <double *>malloc(n * sizeof(double))
    cdef double* rel_m = <double *>malloc(n * sizeof(double))
    cdef double* xis = <double *>malloc(n * sizeof(double))
    cdef double* sig_t = <double *>malloc(n * sizeof(double))
    cdef double* b_vals = <double *>malloc(n * sizeof(double))
    cdef double* delta_vals = <double *>malloc(n * sizeof(double))
    cdef double* vega_vals = <double *>malloc(n * sizeof(double))
    if (
        logits == NULL or s_w == NULL or p_w == NULL or rel_m == NULL
        or xis == NULL or sig_t == NULL or b_vals == NULL or delta_vals == NULL or vega_vals == NULL
    ):
        free(logits); free(s_w); free(p_w); free(rel_m); free(xis)
        free(sig_t); free(b_vals); free(delta_vals); free(vega_vals)
        raise MemoryError("Falha ao alocar buffers da mistura lognormal.")

    cdef int i, l, j, k
    cdef double q_def = vec[3 * n - 2] if allow_default else fixed_q_default
    if q_def < 0.0:
        q_def = 0.0
    if q_def > 0.999:
        q_def = 0.999
    cdef double p_surv = 1.0 - q_def
    cdef double p_surv_safe = p_surv if p_surv > 1e-12 else 1e-12
    cdef double max_l, sum_exp, denom_xi, cum_eta, d_inv
    cdef double kf, vol_l, pr, dl, gm, vg, th, rh, vn, vo
    cdef double sum_pB, A, tail_px, sk, pk
    cdef bint ic

    with nogil:
        logits[0] = 0.0
        max_l = 0.0
        for l in range(1, n):
            logits[l] = vec[l - 1]
            if logits[l] > max_l:
                max_l = logits[l]
        sum_exp = 0.0
        for l in range(n):
            s_w[l] = exp(logits[l] - max_l)
            sum_exp += s_w[l]
        for l in range(n):
            s_w[l] /= sum_exp
            p_w[l] = p_surv * s_w[l]

        cum_eta = 0.0
        rel_m[0] = 1.0
        for l in range(1, n):
            cum_eta += vec[(n - 1) + (l - 1)]
            rel_m[l] = exp(cum_eta)

        denom_xi = 0.0
        for l in range(n):
            denom_xi += p_w[l] * rel_m[l]
        d_inv = 1.0 / (denom_xi if denom_xi > 1e-12 else 1e-12)
        for l in range(n):
            xis[l] = rel_m[l] * d_inv
            sig_t[l] = vec[(2 * n - 2) + l]

        for i in range(m):
            kf = strikes[i]
            ic = is_call[i] != 0
            sum_pB = 0.0
            A = 0.0
            for l in range(n):
                vol_l = sig_t[l] * inv_sqrt_T
                if vol_l < 1e-8:
                    vol_l = 1e-8
                _bsm_core(
                    xis[l] * F, kf, r, r, t_eff, vol_l, ic,
                    &pr, &dl, &gm, &vg, &th, &rh, &vn, &vo,
                )
                b_vals[l] = pr
                delta_vals[l] = dl
                vega_vals[l] = vg
                sum_pB += p_w[l] * pr
                A += p_w[l] * xis[l] * F * dl

            prices_out[i] = sum_pB + (0.0 if ic else df_r * q_def * kf)

            for k in range(1, n):
                sk = s_w[k]
                pk = p_w[k]
                jac_out[i, k - 1] = pk * b_vals[k] - sk * sum_pB - (pk * xis[k] - sk) * A

            for j in range(n - 1):
                tail_px = 0.0
                sk = 0.0
                for l in range(j + 1, n):
                    pk = p_w[l] * xis[l]
                    tail_px += pk * F * delta_vals[l]
                    sk += pk
                jac_out[i, (n - 1) + j] = tail_px - sk * A

            for l in range(n):
                jac_out[i, (2 * n - 2) + l] = p_w[l] * vega_vals[l] * inv_sqrt_T

            if allow_default:
                jac_out[i, 3 * n - 2] = (A - sum_pB) / p_surv_safe + (0.0 if ic else df_r * kf)

    free(logits); free(s_w); free(p_w); free(rel_m); free(xis)
    free(sig_t); free(b_vals); free(delta_vals); free(vega_vals)
    return (prices_out, jac_out)

