# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, log, sqrt, erf, fabs, pow, fmax
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
    if T <= 0.0 or sigma <= 0.0 or steps <= 0:
        return fmax(S - K, 0.0) if is_call else fmax(K - S, 0.0)

    cdef double dt, u, d, disc, p, one_minus_p, spot_i, exercise, hold, val
    cdef int i, j
    cdef double* values = <double*>malloc((steps + 1) * sizeof(double))
    if values == NULL:
        raise MemoryError("Falha ao alocar memória para árvore CRR.")

    with nogil:
        dt = T / steps
        u = exp(sigma * sqrt(dt))
        d = 1.0 / u
        disc = exp(-r * dt)
        p = (exp((r - q) * dt) - d) / (u - d)
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        one_minus_p = 1.0 - p

        for i in range(steps + 1):
            spot_i = S * pow(u, steps - 2 * i)
            if is_call:
                values[i] = fmax(spot_i - K, 0.0)
            else:
                values[i] = fmax(K - spot_i, 0.0)

        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                hold = disc * (p * values[i] + one_minus_p * values[i + 1])
                spot_i = S * pow(u, j - 2 * i)
                exercise = (spot_i - K) if is_call else (K - spot_i)
                values[i] = hold if hold > exercise else exercise

        val = values[0]
        free(values)
    return val


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
    cdef double w = (kappa3 / 6.0) * sig_sqrt_T * sig2_T + (kappa4 / 24.0) * sig2_T * sig2_T
    cdef double d = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T - log(1.0 + w)) / sig_sqrt_T
    cdef double nd = _norm_pdf(d)
    cdef double denom = 1.0 + w

    cdef double q3 = (1.0 / (6.0 * denom)) * S * sig_sqrt_T * (2.0 * sig_sqrt_T - d) * nd
    cdef double q4 = (1.0 / (24.0 * denom)) * S * sig_sqrt_T * (
        d * d - 3.0 * d * sig_sqrt_T + 3.0 * sig2_T - 1.0
    ) * nd

    cdef double call_cs = price_bs + kappa3 * q3 + (kappa4 - 3.0) * q4
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
    if target_price <= intrinsic + 1e-12:
        return 1e-6

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
    x_z = log((sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))

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

    for i in range(n):
        if is_american_arr[i]:
            pv[i] = crr_american_price(
                S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i],
                bool(is_call_arr[i]), crr_steps,
            )
            _bsm_core(
                S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i],
                bool(is_call_arr[i]),
                &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga,
            )
            dv[i] = delta
            gv[i] = gamma
            vv[i] = vega
            tv[i] = theta
            rv[i] = rho
        else:
            _bsm_core(
                S_arr[i], K_arr[i], r_arr[i], q_arr[i], T_arr[i], sigma_arr[i],
                bool(is_call_arr[i]),
                &price, &delta, &gamma, &vega, &theta, &rho, &vanna, &volga,
            )
            pv[i] = price
            dv[i] = delta
            gv[i] = gamma
            vv[i] = vega
            tv[i] = theta
            rv[i] = rho

    return (out_price, out_delta, out_gamma, out_vega, out_theta, out_rho)

