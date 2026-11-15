# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, log, sqrt, erf, fabs, pow, fmax, NAN
from libc.stdlib cimport malloc, free

cdef double INV_SQRT_2PI = 0.3989422804014327
cdef double INV_SQRT_2 = 0.7071067811865475


cdef inline double _norm_pdf(double x) noexcept nogil:
    return INV_SQRT_2PI * exp(-0.5 * x * x)


cdef inline double _norm_cdf(double x) noexcept nogil:
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
    cdef double cdf_d1 = _norm_cdf(d1)
    cdef double cdf_d2 = _norm_cdf(d2)

    if is_call:
        out_price[0] = S * df_q * cdf_d1 - K * df_r * cdf_d2
        out_delta[0] = df_q * cdf_d1
        out_theta[0] = (
            -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * df_r * cdf_d2
            + q * S * df_q * cdf_d1
        )
        out_rho[0] = K * T * df_r * cdf_d2
    else:
        out_price[0] = K * df_r * (1.0 - cdf_d2) - S * df_q * (1.0 - cdf_d1)
        out_delta[0] = df_q * (cdf_d1 - 1.0)
        out_theta[0] = (
            -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * df_r * (1.0 - cdf_d2)
            - q * S * df_q * (1.0 - cdf_d1)
        )
        out_rho[0] = -K * T * df_r * (1.0 - cdf_d2)

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
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
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
    """Árvore binomial Cox-Ross-Rubinstein (CRR) em C puro para opções americanas."""
    cdef double d_tmp, g_tmp, t_tmp, val
    with nogil:
        val = _crr_american_core(S, K, r, q, T, sigma, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
    return val


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
    """Preço americano CRR e gregas consistentes (delta/gamma/theta na árvore; vega/rho por diferenças finitas)."""
    cdef double price, delta, gamma, theta, vega, rho
    cdef double d_tmp, g_tmp, t_tmp
    cdef double dv = 1e-4
    cdef double dr = 1e-4
    cdef double p_v_up, p_v_dn, p_r_up, p_r_dn

    with nogil:
        price = _crr_american_core(S, K, r, q, T, sigma, is_call, steps, &delta, &gamma, &theta)
        if T > 0.0 and sigma > dv and S > 0.0 and K > 0.0:
            p_v_up = _crr_american_core(S, K, r, q, T, sigma + dv, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
            p_v_dn = _crr_american_core(S, K, r, q, T, sigma - dv, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
            vega = (p_v_up - p_v_dn) / (2.0 * dv)

            p_r_up = _crr_american_core(S, K, r + dr, q, T, sigma, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
            p_r_dn = _crr_american_core(S, K, r - dr, q, T, sigma, is_call, steps, &d_tmp, &g_tmp, &t_tmp)
            rho = (p_r_up - p_r_dn) / (2.0 * dr)
        else:
            vega = 0.0
            rho = 0.0

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

    cdef double sig_sqrt_T = sigma * sqrt(T)
    cdef double sig2_T = sigma * sigma * T
    cdef double exc_kurt = kappa4 - 3.0
    cdef double w = (kappa3 / 6.0) * sig_sqrt_T * sig2_T + (exc_kurt / 24.0) * sig2_T * sig2_T
    cdef double denom = fmax(1e-8, 1.0 + w)
    cdef double d = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T - log(denom)) / sig_sqrt_T
    cdef double nd = _norm_pdf(d)

    cdef double q3 = (1.0 / (6.0 * denom)) * S * sig_sqrt_T * (2.0 * sig_sqrt_T - d) * nd
    cdef double q4 = (1.0 / (24.0 * denom)) * S * sig_sqrt_T * (
        d * d - 3.0 * d * sig_sqrt_T + 3.0 * sig2_T - 1.0
    ) * nd

    cdef double call_cs = price_bs + kappa3 * q3 + exc_kurt * q4
    if call_cs < 0.0:
        call_cs = 0.0
    if is_call:
        return call_cs

    cdef double put_cs = call_cs - S * exp(-q * T) + K * exp(-r * T)
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
    """Inversão de volatilidade implícita via Newton-Raphson com fallback por bisseção."""
    if T <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0

    cdef double df_q = exp(-q * T)
    cdef double df_r = exp(-r * T)
    cdef double intrinsic = fmax(S * df_q - K * df_r, 0.0) if is_call else fmax(K * df_r - S * df_q, 0.0)
    cdef double upper_bound = (S * df_q) if is_call else (K * df_r)
    if target_price <= intrinsic + 1e-12:
        return 1e-6
    if target_price >= upper_bound - 1e-12:
        return NAN

    cdef double low = 1e-5
    cdef double high = 5.0
    cdef double sigma = 0.25
    cdef double price, delta, gamma, vega, theta, rho, vanna, volga, diff
    cdef int it

    for it in range(max_iter):
        _bsm_core(S, K, r, q, T, sigma, is_call, &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga)
        diff = price - target_price
        if fabs(diff) < tol:
            return sigma
        if diff < 0.0 and sigma > low:
            low = sigma
        elif diff > 0.0 and sigma < high:
            high = sigma

        if vega > 1e-10:
            sigma = sigma - diff / vega
            if sigma <= low or sigma >= high:
                sigma = 0.5 * (low + high)
        else:
            sigma = 0.5 * (low + high)

    for it in range(60):
        sigma = 0.5 * (low + high)
        _bsm_core(S, K, r, q, T, sigma, is_call, &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga)
        diff = price - target_price
        if fabs(diff) < tol or (high - low) < tol:
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
    cdef double d_vol = 1e-4
    cdef double d_rate = 1e-4

    cdef bint call_flag
    with nogil:
        for i in range(n):
            call_flag = (is_call_arr[i] != 0)
            if is_american_arr[i]:
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
                    vv[i] = (p_v_up - p_v_dn) / (2.0 * d_vol)
                    p_r_up = _crr_american_core(S_arr[i], K_arr[i], r_arr[i] + d_rate, q_arr[i], T_arr[i], sigma_arr[i], call_flag, crr_steps, &d_tmp, &g_tmp, &t_tmp)
                    p_r_dn = _crr_american_core(S_arr[i], K_arr[i], r_arr[i] - d_rate, q_arr[i], T_arr[i], sigma_arr[i], call_flag, crr_steps, &d_tmp, &g_tmp, &t_tmp)
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

