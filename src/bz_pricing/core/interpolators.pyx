# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport pow, log, exp, fabs, sqrt
from libc.stdlib cimport malloc, free

from bz_pricing.core.conventions cimport (
    DayCountConvention,
    CompoundingType,
    EXPONENTIAL_B3,
    SIMPLE_LINEAR,
    COMPOUND_ANNUAL,
    CONTINUOUS,
)
from bz_pricing.core.interpolators cimport ExtrapolationType, EXTRAP_FLAT, EXTRAP_FLAT_FORWARD, EXTRAP_LINEAR


cdef inline int _find_interval(double t, double *t_nodes, int n_nodes, int extrap_left, int extrap_right) noexcept nogil:
    """
    Retorna:
      -1 se deve retornar y_nodes[0] (n_nodes <= 1 ou extrapolação flat à esquerda)
      -2 se deve retornar y_nodes[n_nodes - 1] (extrapolação flat à direita)
      i >= 0 indicando o intervalo [t_nodes[i], t_nodes[i+1]] via busca binária O(log n).
    """
    if n_nodes <= 1:
        return -1
    if t <= t_nodes[0]:
        if extrap_left == EXTRAP_FLAT:
            return -1
        return 0
    if t >= t_nodes[n_nodes - 1]:
        if extrap_right == EXTRAP_FLAT:
            return -2
        return n_nodes - 2

    cdef int lo = 0
    cdef int hi = n_nodes - 2
    cdef int mid
    while lo <= hi:
        mid = (lo + hi) >> 1
        if t < t_nodes[mid]:
            hi = mid - 1
        elif t > t_nodes[mid + 1]:
            lo = mid + 1
        else:
            return mid
    return lo

cdef class Interpolator:
    """
    Classe base para todas as funções de interpolação.
    Recebe as diretrizes de extrapolação (Esquerda e Direita) no construtor.
    EXTRAP_FLAT (0) = Trava no último/primeiro valor
    EXTRAP_FLAT_FORWARD (1) / EXTRAP_LINEAR (2) = Usa os dois primeiros/últimos vértices 
    para projetar a mesma fórmula de interpolação para fora da curva
    """
    def __init__(self, int extrap_left=0, int extrap_right=0):
        self._extrap_left = extrap_left
        self._extrap_right = extrap_right

    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        raise NotImplementedError("Método base")


cdef class Exponential252Interpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        cdef int i = _find_interval(t, t_nodes, n_nodes, self._extrap_left, self._extrap_right)
        if i == -1:
            return y_nodes[0]
        if i == -2:
            return y_nodes[n_nodes - 1]
                    
        cdef double t1 = t_nodes[i]
        cdef double t2 = t_nodes[i+1]
        cdef double y1 = y_nodes[i]
        cdef double y2 = y_nodes[i+1]
        
        if t1 == t2:
            return y1
            
        cdef double f1 = 1.0 + y1
        cdef double f2 = 1.0 + y2
        cdef double factor = f1 * pow(f2 / f1, (t - t1) / (t2 - t1))
        return factor - 1.0


cdef class FlatForward252Interpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        cdef int i = _find_interval(t, t_nodes, n_nodes, self._extrap_left, self._extrap_right)
        if i == -1:
            return y_nodes[0]
        if i == -2:
            return y_nodes[n_nodes - 1]
                    
        cdef double t1 = t_nodes[i]
        cdef double t2 = t_nodes[i+1]
        cdef double y1 = y_nodes[i]
        cdef double y2 = y_nodes[i+1]
        
        if t1 == t2 or fabs(t) < 1e-12:
            return y1
            
        cdef double f1 = pow(1.0 + y1, t1 / 252.0)
        cdef double f2 = pow(1.0 + y2, t2 / 252.0)
        cdef double forward = f1 * pow(f2 / f1, (t - t1) / (t2 - t1))
        return pow(forward, 252.0 / t) - 1.0


cdef class FlatForward252LinearInterpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        if t_sec_nodes == NULL:
            raise ValueError("t_sec_nodes (DC) é obrigatório para FlatForward252LinearInterpolator")
            
        cdef int i = _find_interval(t, t_nodes, n_nodes, self._extrap_left, self._extrap_right)
        if i == -1:
            return y_nodes[0]
        if i == -2:
            return y_nodes[n_nodes - 1]
                    
        cdef double du1 = t_nodes[i]
        cdef double du2 = t_nodes[i+1]
        cdef double dc1 = t_sec_nodes[i]
        cdef double dc2 = t_sec_nodes[i+1]
        cdef double y1 = y_nodes[i]
        cdef double y2 = y_nodes[i+1]
        
        if du1 == du2:
            return y1
            
        cdef double f1 = 1.0 + (y1 * dc1) / 360.0
        cdef double f2 = 1.0 + (y2 * dc2) / 360.0
        
        cdef double forward = f1 * pow(f2 / f1, (t - du1) / (du2 - du1))
        
        if t_sec == 0.0:
            t_sec = dc1 + (dc2 - dc1) * ((t - du1) / (du2 - du1))
            if t_sec == 0.0:
                t_sec = 1.0
            
        return (forward - 1.0) * (360.0 / t_sec)


cdef class Interpolation360Interpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        # Busca o intervalo em t_nodes e interpola exponencialmente sobre os dias corridos (t_sec_nodes)
        if t_sec_nodes == NULL:
            raise ValueError("t_sec_nodes (DC) é obrigatório para Interpolation360Interpolator")
            
        cdef int i = _find_interval(t, t_nodes, n_nodes, self._extrap_left, self._extrap_right)
        if i == -1:
            return y_nodes[0]
        if i == -2:
            return y_nodes[n_nodes - 1]
                    
        cdef double dc1 = t_sec_nodes[i]
        cdef double dc2 = t_sec_nodes[i+1]
        cdef double y1 = y_nodes[i]
        cdef double y2 = y_nodes[i+1]
        
        if dc1 == dc2:
            return y1
            
        cdef double f1 = 1.0 + (y1 * dc1) / 360.0
        cdef double f2 = 1.0 + (y2 * dc2) / 360.0
        
        if t_sec == 0.0:
            t_sec = dc1 + (dc2 - dc1) * ((t - t_nodes[i]) / (t_nodes[i+1] - t_nodes[i]))
            if t_sec == 0.0:
                return y1
                
        cdef double forward = f1 * pow(f2 / f1, (t_sec - dc1) / (dc2 - dc1))
        return (forward - 1.0) * (360.0 / t_sec)


cdef class Linear360Interpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        cdef int i = _find_interval(t, t_nodes, n_nodes, self._extrap_left, self._extrap_right)
        if i == -1:
            return y_nodes[0]
        if i == -2:
            return y_nodes[n_nodes - 1]
                    
        cdef double dc1
        cdef double dc2
        if t_sec_nodes != NULL:
            dc1 = t_sec_nodes[i]
            dc2 = t_sec_nodes[i+1]
            if t_sec == 0.0:
                t_sec = dc1 + (dc2 - dc1) * ((t - t_nodes[i]) / (t_nodes[i+1] - t_nodes[i]))
        else:
            # Caso a curva seja pura em Dias Corridos (DC), t_nodes é o próprio vetor de DC
            dc1 = t_nodes[i]
            dc2 = t_nodes[i+1]
            if t_sec == 0.0:
                t_sec = t
                
        cdef double y1 = y_nodes[i]
        cdef double y2 = y_nodes[i+1]
        
        if dc1 == dc2:
            return y1
            
        cdef double f1 = (y1 * dc1) / 360.0
        cdef double f2 = (y2 * dc2) / 360.0
        
        if t_sec == 0.0:
            t_sec = 1.0 # Previne divisão por zero extrema
                
        # Ponderação linear pelos dias corridos (ACT/360)
        cdef double forward = f1 + (f2 - f1) * ((t_sec - dc1) / (dc2 - dc1))
            
        return (forward * 360.0) / t_sec


cdef class PriceInterpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        cdef int i = _find_interval(t, t_nodes, n_nodes, self._extrap_left, self._extrap_right)
        if i == -1:
            return y_nodes[0]
        if i == -2:
            return y_nodes[n_nodes - 1]
                    
        cdef double t1 = t_nodes[i]
        cdef double t2 = t_nodes[i+1]
        cdef double p1 = y_nodes[i]
        cdef double p2 = y_nodes[i+1]
        
        if t1 == t2:
            return p1
        return p1 * pow(p2 / p1, (t - t1) / (t2 - t1))


cdef class FlatForward252LogInterpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        cdef int i = _find_interval(t, t_nodes, n_nodes, self._extrap_left, self._extrap_right)
        if i == -1:
            return y_nodes[0]
        if i == -2:
            return y_nodes[n_nodes - 1]
                    
        cdef double t1 = t_nodes[i]
        cdef double t2 = t_nodes[i+1]
        cdef double y1 = y_nodes[i]
        cdef double y2 = y_nodes[i+1]
        
        if t1 == t2 or fabs(t) < 1e-12:
            return y1
            
        cdef double ln_f1 = (t1 / 252.0) * log(1.0 + y1)
        cdef double ln_f2 = (t2 / 252.0) * log(1.0 + y2)
        cdef double ln_forward = ln_f1 + (ln_f2 - ln_f1) * ((t - t1) / (t2 - t1))
        
        return exp(ln_forward * 252.0 / t) - 1.0


cdef inline int _coerce_compounding(object compounding) except -1:
    if isinstance(compounding, int):
        return <int>compounding
    cdef str c = str(compounding).strip().lower()
    if c in ("continuous", "cont"):
        return CONTINUOUS
    if c in ("simple", "simple_linear", "linear"):
        return SIMPLE_LINEAR
    if c in ("exponential", "exponential_b3", "exp_252", "b3"):
        return EXPONENTIAL_B3
    if c in ("compounded", "compound", "compound_annual", "annual", "semiannual"):
        return COMPOUND_ANNUAL
    raise ValueError(f"Regime de capitalização inválido: {compounding}")


cdef inline double _rate_to_ln_df(double rate, double tau, int compounding, int freq) noexcept nogil:
    if tau <= 0.0:
        return 0.0
    cdef double k
    if compounding == CONTINUOUS:
        return -rate * tau
    if compounding == SIMPLE_LINEAR:
        if 1.0 + rate * tau <= 1e-12:
            return -rate * tau
        return -log(1.0 + rate * tau)
    if compounding == EXPONENTIAL_B3:
        if 1.0 + rate <= 1e-12:
            return -rate * tau
        return -tau * log(1.0 + rate)
    k = <double>(freq if freq > 0 else 1)
    if 1.0 + rate / k <= 1e-12:
        return -rate * tau
    return -(k * tau) * log(1.0 + rate / k)


cdef inline double _ln_df_to_rate(double ln_df, double tau, int compounding, int freq) noexcept nogil:
    if tau <= 1e-12:
        return 0.0
    cdef double df, k
    if compounding == CONTINUOUS:
        return -ln_df / tau
    if compounding == SIMPLE_LINEAR:
        df = exp(ln_df)
        return (1.0 / df - 1.0) / tau
    if compounding == EXPONENTIAL_B3:
        return exp(-ln_df / tau) - 1.0
    k = <double>(freq if freq > 0 else 1)
    return k * (exp(-ln_df / (k * tau)) - 1.0)


cdef class LogLinearDiscountInterpolator(Interpolator):
    """Interpolação log-linear nos fatores de desconto (taxa a termo constante por partes)."""

    def __init__(
        self,
        int extrap_left=0,
        int extrap_right=0,
        double day_count_basis=365.0,
        object compounding="continuous",
        int freq=1,
    ):
        super().__init__(extrap_left, extrap_right)
        if day_count_basis <= 0.0:
            raise ValueError("day_count_basis deve ser positivo.")
        self._day_count_basis = day_count_basis
        self._compounding = _coerce_compounding(compounding)
        self._freq = max(1, freq)

    @property
    def day_count_basis(self):
        return self._day_count_basis

    @property
    def compounding(self):
        if self._compounding == EXPONENTIAL_B3:
            return "exponential"
        if self._compounding == SIMPLE_LINEAR:
            return "simple"
        if self._compounding == COMPOUND_ANNUAL:
            return "compound"
        return "continuous"

    @property
    def freq(self):
        return self._freq

    cdef tuple _eval_ln_df_and_fwd(self, double t, double *t_nodes, double *y_nodes, int n_nodes):
        if n_nodes <= 0:
            raise ValueError("Nós de interpolação vazios.")
        cdef double tau = t / self._day_count_basis
        cdef double t0 = t_nodes[0]
        cdef double tau0 = t0 / self._day_count_basis if t0 > 0.0 else 1.0
        cdef double ln0 = _rate_to_ln_df(y_nodes[0], tau0, self._compounding, self._freq) if t0 > 0.0 else 0.0
        cdef double f0, f_left, slope, t1, t2, ln1, ln2, ln_t, ta, tb, lna, lnb
        cdef int idx = 0
        cdef int lo, hi, mid

        if n_nodes == 1 or t <= 0.0:
            ln0 = _rate_to_ln_df(y_nodes[0], tau0, self._compounding, self._freq)
            f0 = -ln0 / tau0
            return (0.0 if t <= 0.0 else -f0 * tau, f0)

        if t <= t0:
            if self._extrap_left == EXTRAP_FLAT:
                ln_t = _rate_to_ln_df(y_nodes[0], tau, self._compounding, self._freq)
                return (ln_t, -ln_t / tau if tau > 1e-12 else 0.0)
            f_left = (-ln0 / tau0) if tau0 > 1e-12 else y_nodes[0]
            return (-f_left * tau, f_left)

        if t >= t_nodes[n_nodes - 1]:
            if self._extrap_right == EXTRAP_FLAT:
                ln_t = _rate_to_ln_df(y_nodes[n_nodes - 1], tau, self._compounding, self._freq)
                return (ln_t, -ln_t / tau if tau > 1e-12 else 0.0)
            t1 = t_nodes[n_nodes - 2]
            t2 = t_nodes[n_nodes - 1]
            ln1 = _rate_to_ln_df(y_nodes[n_nodes - 2], t1 / self._day_count_basis, self._compounding, self._freq) if t1 > 0.0 else 0.0
            ln2 = _rate_to_ln_df(y_nodes[n_nodes - 1], t2 / self._day_count_basis, self._compounding, self._freq)
            slope = (ln2 - ln1) / ((t2 - t1) / self._day_count_basis) if t2 > t1 else (-ln2 / (t2 / self._day_count_basis))
            ln_t = ln2 + slope * ((t - t2) / self._day_count_basis)
            return (ln_t, -slope)

        lo = 0
        hi = n_nodes - 2
        while lo <= hi:
            mid = (lo + hi) >> 1
            if t < t_nodes[mid]:
                hi = mid - 1
            elif t > t_nodes[mid + 1]:
                lo = mid + 1
            else:
                idx = mid
                break

        ta = t_nodes[idx] / self._day_count_basis
        tb = t_nodes[idx + 1] / self._day_count_basis
        lna = _rate_to_ln_df(y_nodes[idx], ta, self._compounding, self._freq) if ta > 0.0 else 0.0
        lnb = _rate_to_ln_df(y_nodes[idx + 1], tb, self._compounding, self._freq)
        if tb == ta:
            return (lna, 0.0)
        slope = (lnb - lna) / (tb - ta)
        return (lna + slope * (tau - ta), -slope)

    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        if n_nodes <= 1 or t <= 0.0:
            return y_nodes[0]
        if t <= t_nodes[0] and self._extrap_left == EXTRAP_FLAT and self._compounding == CONTINUOUS:
            return y_nodes[0]
        if t >= t_nodes[n_nodes - 1] and self._extrap_right == EXTRAP_FLAT:
            return y_nodes[n_nodes - 1]
        cdef tuple res = self._eval_ln_df_and_fwd(t, t_nodes, y_nodes, n_nodes)
        cdef double ln_df = res[0]
        return _ln_df_to_rate(ln_df, t / self._day_count_basis, self._compounding, self._freq)

    cpdef double forward_rate(self, double t, double[:] t_nodes, double[:] y_nodes) except *:
        cdef tuple res = self._eval_ln_df_and_fwd(t, &t_nodes[0], &y_nodes[0], t_nodes.shape[0])
        return <double>res[1]


cdef class LogCubicDiscountInterpolator(Interpolator):
    """Interpolação cúbica monótona de Fritsch-Carlson sobre o logaritmo do fator de desconto."""

    def __cinit__(self):
        self._cached_n = 0
        self._cached_m = 0
        self._cached_t = NULL
        self._cached_y = NULL
        self._xs = NULL
        self._ys = NULL
        self._h = NULL
        self._d = NULL
        self._slopes = NULL

    def __dealloc__(self):
        if self._cached_t != NULL:
            free(self._cached_t)
            self._cached_t = NULL
        if self._cached_y != NULL:
            free(self._cached_y)
            self._cached_y = NULL
        if self._xs != NULL:
            free(self._xs)
            self._xs = NULL
        if self._ys != NULL:
            free(self._ys)
            self._ys = NULL
        if self._h != NULL:
            free(self._h)
            self._h = NULL
        if self._d != NULL:
            free(self._d)
            self._d = NULL
        if self._slopes != NULL:
            free(self._slopes)
            self._slopes = NULL

    def __init__(
        self,
        int extrap_left=0,
        int extrap_right=0,
        double day_count_basis=365.0,
        object compounding="continuous",
        int freq=1,
    ):
        super().__init__(extrap_left, extrap_right)
        if day_count_basis <= 0.0:
            raise ValueError("day_count_basis deve ser positivo.")
        self._day_count_basis = day_count_basis
        self._compounding = _coerce_compounding(compounding)
        self._freq = max(1, freq)

    @property
    def day_count_basis(self):
        return self._day_count_basis

    @property
    def compounding(self):
        if self._compounding == EXPONENTIAL_B3:
            return "exponential"
        if self._compounding == SIMPLE_LINEAR:
            return "simple"
        if self._compounding == COMPOUND_ANNUAL:
            return "compound"
        return "continuous"

    @property
    def freq(self):
        return self._freq

    cdef void _ensure_cache(self, double *t_nodes, double *y_nodes, int n_nodes) except *:
        cdef int i, offset, m
        cdef bint match = (self._cached_n == n_nodes and self._cached_t != NULL and self._cached_y != NULL)
        if match:
            for i in range(n_nodes):
                if self._cached_t[i] != t_nodes[i] or self._cached_y[i] != y_nodes[i]:
                    match = False
                    break
        if match:
            return

        if self._cached_t != NULL: free(self._cached_t); self._cached_t = NULL
        if self._cached_y != NULL: free(self._cached_y); self._cached_y = NULL
        if self._xs != NULL: free(self._xs); self._xs = NULL
        if self._ys != NULL: free(self._ys); self._ys = NULL
        if self._h != NULL: free(self._h); self._h = NULL
        if self._d != NULL: free(self._d); self._d = NULL
        if self._slopes != NULL: free(self._slopes); self._slopes = NULL
        self._cached_n = 0
        self._cached_m = 0

        cdef bint prepend_zero = (t_nodes[0] > 0.0)
        m = n_nodes + (1 if prepend_zero else 0)
        cdef double *ct = <double *>malloc(n_nodes * sizeof(double))
        cdef double *cy = <double *>malloc(n_nodes * sizeof(double))
        cdef double *xs = <double *>malloc(m * sizeof(double))
        cdef double *ys = <double *>malloc(m * sizeof(double))
        cdef double *h = <double *>malloc(m * sizeof(double))
        cdef double *d = <double *>malloc(m * sizeof(double))
        cdef double *slopes = <double *>malloc(m * sizeof(double))
        if ct == NULL or cy == NULL or xs == NULL or ys == NULL or h == NULL or d == NULL or slopes == NULL:
            free(ct); free(cy); free(xs); free(ys); free(h); free(d); free(slopes)
            raise MemoryError()

        for i in range(n_nodes):
            ct[i] = t_nodes[i]
            cy[i] = y_nodes[i]

        offset = 1 if prepend_zero else 0
        if prepend_zero:
            xs[0] = 0.0
            ys[0] = 0.0
        for i in range(n_nodes):
            xs[i + offset] = t_nodes[i] / self._day_count_basis
            ys[i + offset] = _rate_to_ln_df(y_nodes[i], xs[i + offset], self._compounding, self._freq)

        cdef double w1, w2, alpha, beta, r2, sc
        if m == 2:
            d[0] = (ys[1] - ys[0]) / max(1e-12, xs[1] - xs[0])
            slopes[0] = d[0]
            slopes[1] = d[0]
        elif m > 2:
            for i in range(m - 1):
                h[i] = xs[i + 1] - xs[i]
                d[i] = (ys[i + 1] - ys[i]) / h[i] if h[i] > 1e-12 else 0.0

            # Inclinações iniciais ponderadas (método harmônico de Kruger para grade não uniforme)
            for i in range(1, m - 1):
                if d[i - 1] * d[i] <= 0.0:
                    slopes[i] = 0.0
                else:
                    w1 = 2.0 * h[i] + h[i - 1]
                    w2 = h[i] + 2.0 * h[i - 1]
                    slopes[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

            slopes[0] = ((2.0 * h[0] + h[1]) * d[0] - h[0] * d[1]) / max(1e-12, h[0] + h[1])
            if slopes[0] * d[0] <= 0.0:
                slopes[0] = 0.0
            elif fabs(slopes[0]) > 3.0 * fabs(d[0]):
                slopes[0] = 3.0 * d[0]

            slopes[m - 1] = ((2.0 * h[m - 2] + h[m - 3]) * d[m - 2] - h[m - 2] * d[m - 3]) / max(1e-12, h[m - 2] + h[m - 3])
            if slopes[m - 1] * d[m - 2] <= 0.0:
                slopes[m - 1] = 0.0
            elif fabs(slopes[m - 1]) > 3.0 * fabs(d[m - 2]):
                slopes[m - 1] = 3.0 * d[m - 2]

            # Filtro de monotonicidade de Fritsch-Carlson
            for i in range(m - 1):
                if fabs(d[i]) < 1e-15:
                    slopes[i] = 0.0
                    slopes[i + 1] = 0.0
                else:
                    alpha = slopes[i] / d[i]
                    beta = slopes[i + 1] / d[i]
                    r2 = alpha * alpha + beta * beta
                    if r2 > 9.0:
                        sc = 3.0 / sqrt(r2)
                        slopes[i] = sc * alpha * d[i]
                        slopes[i + 1] = sc * beta * d[i]

        self._cached_t = ct
        self._cached_y = cy
        self._xs = xs
        self._ys = ys
        self._h = h
        self._d = d
        self._slopes = slopes
        self._cached_n = n_nodes
        self._cached_m = m

    cdef tuple _eval_ln_df_and_fwd(self, double t, double *t_nodes, double *y_nodes, int n_nodes):
        if n_nodes <= 0:
            raise ValueError("Nós de interpolação vazios.")
        self._ensure_cache(t_nodes, y_nodes, n_nodes)
        cdef int m = self._cached_m
        cdef double *xs = self._xs
        cdef double *ys = self._ys
        cdef double *h = self._h
        cdef double *d = self._d
        cdef double *slopes = self._slopes

        if m == 1:
            return (0.0, 0.0)
        cdef double tau = t / self._day_count_basis
        if m == 2:
            return (ys[0] + d[0] * (tau - xs[0]), -d[0])

        if tau <= 0.0:
            return (0.0, -slopes[0])
        cdef double ln_v, dlndf
        if tau >= xs[m - 1]:
            if self._extrap_right == EXTRAP_FLAT:
                ln_v = _rate_to_ln_df(y_nodes[n_nodes - 1], tau, self._compounding, self._freq)
                return (ln_v, -ln_v / tau)
            ln_v = ys[m - 1] + slopes[m - 1] * (tau - xs[m - 1])
            return (ln_v, -slopes[m - 1])

        cdef int lo = 0
        cdef int hi = m - 2
        cdef int mid
        cdef int idx = m - 2
        while lo <= hi:
            mid = (lo + hi) >> 1
            if tau < xs[mid]:
                hi = mid - 1
            elif tau > xs[mid + 1]:
                lo = mid + 1
            else:
                idx = mid
                break

        cdef double h_seg = h[idx]
        cdef double u = (tau - xs[idx]) / h_seg if h_seg > 1e-12 else 0.0
        cdef double u2 = u * u
        cdef double u3 = u2 * u
        cdef double h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        cdef double h10 = u3 - 2.0 * u2 + u
        cdef double h01 = -2.0 * u3 + 3.0 * u2
        cdef double h11 = u3 - u2

        cdef double dh00 = (6.0 * u2 - 6.0 * u) / h_seg if h_seg > 1e-12 else 0.0
        cdef double dh10 = (3.0 * u2 - 4.0 * u + 1.0)
        cdef double dh01 = (-6.0 * u2 + 6.0 * u) / h_seg if h_seg > 1e-12 else 0.0
        cdef double dh11 = (3.0 * u2 - 2.0 * u)

        ln_v = ys[idx] * h00 + h_seg * slopes[idx] * h10 + ys[idx + 1] * h01 + h_seg * slopes[idx + 1] * h11
        dlndf = ys[idx] * dh00 + h_seg * slopes[idx] * dh10 + ys[idx + 1] * dh01 + h_seg * slopes[idx + 1] * dh11
        return (ln_v, -dlndf)

    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        if n_nodes <= 1 or t <= 0.0:
            return y_nodes[0]
        if t >= t_nodes[n_nodes - 1] and self._extrap_right == EXTRAP_FLAT:
            return y_nodes[n_nodes - 1]
        cdef tuple res = self._eval_ln_df_and_fwd(t, t_nodes, y_nodes, n_nodes)
        return _ln_df_to_rate(<double>res[0], t / self._day_count_basis, self._compounding, self._freq)

    cpdef double forward_rate(self, double t, double[:] t_nodes, double[:] y_nodes) except *:
        cdef tuple res = self._eval_ln_df_and_fwd(t, &t_nodes[0], &y_nodes[0], t_nodes.shape[0])
        return <double>res[1]


cdef class MonotoneConvexInterpolator(Interpolator):
    """Interpolação Monotone Convex de Hagan e West (2006) para taxas a termo e curvas soberanas/OIS."""

    def __cinit__(self):
        self._cached_n = 0
        self._cached_m = 0
        self._cached_t = NULL
        self._cached_y = NULL
        self._taus = NULL
        self._zcum = NULL
        self._fd = NULL
        self._fnodes = NULL

    def __dealloc__(self):
        if self._cached_t != NULL:
            free(self._cached_t)
            self._cached_t = NULL
        if self._cached_y != NULL:
            free(self._cached_y)
            self._cached_y = NULL
        if self._taus != NULL:
            free(self._taus)
            self._taus = NULL
        if self._zcum != NULL:
            free(self._zcum)
            self._zcum = NULL
        if self._fd != NULL:
            free(self._fd)
            self._fd = NULL
        if self._fnodes != NULL:
            free(self._fnodes)
            self._fnodes = NULL

    def __init__(
        self,
        int extrap_left=0,
        int extrap_right=0,
        double day_count_basis=365.0,
        object compounding="continuous",
        int freq=1,
        bint enforce_positivity=True,
    ):
        super().__init__(extrap_left, extrap_right)
        if day_count_basis <= 0.0:
            raise ValueError("day_count_basis deve ser positivo.")
        self._day_count_basis = day_count_basis
        self._compounding = _coerce_compounding(compounding)
        self._freq = max(1, freq)
        self._enforce_positivity = enforce_positivity

    @property
    def day_count_basis(self):
        return self._day_count_basis

    @property
    def compounding(self):
        if self._compounding == EXPONENTIAL_B3:
            return "exponential"
        if self._compounding == SIMPLE_LINEAR:
            return "simple"
        if self._compounding == COMPOUND_ANNUAL:
            return "compound"
        return "continuous"

    @property
    def freq(self):
        return self._freq

    cdef void _ensure_cache(self, double *t_nodes, double *y_nodes, int n_nodes) except *:
        cdef int i, start_idx, m
        cdef bint match = (self._cached_n == n_nodes and self._cached_t != NULL and self._cached_y != NULL)
        if match:
            for i in range(n_nodes):
                if self._cached_t[i] != t_nodes[i] or self._cached_y[i] != y_nodes[i]:
                    match = False
                    break
        if match:
            return

        if self._cached_t != NULL: free(self._cached_t); self._cached_t = NULL
        if self._cached_y != NULL: free(self._cached_y); self._cached_y = NULL
        if self._taus != NULL: free(self._taus); self._taus = NULL
        if self._zcum != NULL: free(self._zcum); self._zcum = NULL
        if self._fd != NULL: free(self._fd); self._fd = NULL
        if self._fnodes != NULL: free(self._fnodes); self._fnodes = NULL
        self._cached_n = 0
        self._cached_m = 0

        start_idx = 1 if (n_nodes > 1 and t_nodes[0] <= 0.0) else 0
        m = (n_nodes - start_idx) + 1
        cdef double *ct = <double *>malloc(n_nodes * sizeof(double))
        cdef double *cy = <double *>malloc(n_nodes * sizeof(double))
        cdef double *taus = <double *>malloc(m * sizeof(double))
        cdef double *zcum = <double *>malloc(m * sizeof(double))
        cdef double *fd = <double *>malloc(m * sizeof(double))
        cdef double *fnodes = <double *>malloc(m * sizeof(double))
        if ct == NULL or cy == NULL or taus == NULL or zcum == NULL or fd == NULL or fnodes == NULL:
            free(ct); free(cy); free(taus); free(zcum); free(fd); free(fnodes)
            raise MemoryError()

        for i in range(n_nodes):
            ct[i] = t_nodes[i]
            cy[i] = y_nodes[i]

        cdef double min_fd = 1e99
        cdef double w1, w2, span
        taus[0] = 0.0
        zcum[0] = 0.0
        fd[0] = 0.0
        fnodes[0] = 0.0
        for i in range(1, m):
            taus[i] = t_nodes[start_idx + i - 1] / self._day_count_basis
            zcum[i] = -_rate_to_ln_df(y_nodes[start_idx + i - 1], taus[i], self._compounding, self._freq)
            span = taus[i] - taus[i - 1]
            fd[i] = (zcum[i] - zcum[i - 1]) / span if span > 1e-12 else 0.0
            if fd[i] < min_fd:
                min_fd = fd[i]

        if m > 2:
            for i in range(1, m - 1):
                span = taus[i + 1] - taus[i - 1]
                if span > 1e-12:
                    w1 = (taus[i] - taus[i - 1]) / span
                    w2 = (taus[i + 1] - taus[i]) / span
                    fnodes[i] = w1 * fd[i + 1] + w2 * fd[i]
                else:
                    fnodes[i] = fd[i]

            fnodes[0] = fd[1] - 0.5 * (fnodes[1] - fd[1])
            fnodes[m - 1] = fd[m - 1] - 0.5 * (fnodes[m - 2] - fd[m - 1])

            if self._enforce_positivity and min_fd >= 0.0:
                if fnodes[0] < 0.0:
                    fnodes[0] = 0.0
                elif fnodes[0] > 2.0 * fd[1]:
                    fnodes[0] = 2.0 * fd[1]

                for i in range(1, m - 1):
                    w1 = 2.0 * fd[i] if fd[i] < fd[i + 1] else 2.0 * fd[i + 1]
                    if fnodes[i] < 0.0:
                        fnodes[i] = 0.0
                    elif fnodes[i] > w1:
                        fnodes[i] = w1

                if fnodes[m - 1] < 0.0:
                    fnodes[m - 1] = 0.0
                elif fnodes[m - 1] > 2.0 * fd[m - 1]:
                    fnodes[m - 1] = 2.0 * fd[m - 1]

        self._cached_t = ct
        self._cached_y = cy
        self._taus = taus
        self._zcum = zcum
        self._fd = fd
        self._fnodes = fnodes
        self._cached_n = n_nodes
        self._cached_m = m

    cdef tuple _eval_ln_df_and_fwd(self, double t, double *t_nodes, double *y_nodes, int n_nodes):
        if n_nodes <= 0:
            raise ValueError("Nós de interpolação vazios.")
        self._ensure_cache(t_nodes, y_nodes, n_nodes)
        cdef int m = self._cached_m
        cdef double *taus = self._taus
        cdef double *zcum = self._zcum
        cdef double *fd = self._fd
        cdef double *fnodes = self._fnodes

        cdef double tau = t / self._day_count_basis
        if m <= 1:
            return (0.0, 0.0)
        if m == 2:
            return (-fd[1] * tau, fd[1])
        if tau <= 0.0:
            return (0.0, fnodes[0])

        cdef double cum_int, inst_fwd, h_seg, x, g0, g1, g_x, G_x, eta, u, A, denom, fd_seg
        if tau >= taus[m - 1]:
            if self._extrap_right == EXTRAP_FLAT:
                cum_int = -_rate_to_ln_df(y_nodes[n_nodes - 1], tau, self._compounding, self._freq)
                return (-cum_int, fnodes[m - 1])
            cum_int = zcum[m - 1] + fnodes[m - 1] * (tau - taus[m - 1])
            return (-cum_int, fnodes[m - 1])

        cdef int lo = 1
        cdef int hi = m - 1
        cdef int mid
        cdef int seg_idx = m - 1
        while lo <= hi:
            mid = (lo + hi) >> 1
            if tau < taus[mid - 1]:
                hi = mid - 1
            elif tau > taus[mid]:
                lo = mid + 1
            else:
                seg_idx = mid
                break

        h_seg = taus[seg_idx] - taus[seg_idx - 1]
        if h_seg <= 1e-12:
            return (-zcum[seg_idx], fnodes[seg_idx])

        x = (tau - taus[seg_idx - 1]) / h_seg
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0

        fd_seg = fd[seg_idx]
        g0 = fnodes[seg_idx - 1] - fd_seg
        g1 = fnodes[seg_idx] - fd_seg

        if fabs(g0) < 1e-15 and fabs(g1) < 1e-15:
            g_x = 0.0
            G_x = 0.0
        elif (g0 > 0.0 and -2.0 * g0 <= g1 and g1 <= -0.5 * g0) or (g0 < 0.0 and -0.5 * g0 <= g1 and g1 <= -2.0 * g0):
            g_x = g0 + (-4.0 * g0 - 2.0 * g1) * x + (3.0 * g0 + 3.0 * g1) * x * x
            G_x = g0 * x + (-2.0 * g0 - g1) * x * x + (g0 + g1) * x * x * x
        elif (g0 > 0.0 and -0.5 * g0 < g1 and g1 < 0.0) or (g0 < 0.0 and 0.0 < g1 and g1 < -0.5 * g0):
            eta = (3.0 * g1) / (g1 - g0)
            if eta < 1e-12:
                g_x = g1
                G_x = 0.0
            elif x < eta:
                u = (eta - x) / eta
                g_x = g1 + (g0 - g1) * u * u
                G_x = g1 * x + (eta / 3.0) * (g0 - g1) * (1.0 - u * u * u)
            else:
                g_x = g1
                G_x = g1 * x + (eta / 3.0) * (g0 - g1)
        elif (g0 > 0.0 and g1 < -2.0 * g0) or (g0 < 0.0 and g1 > -2.0 * g0):
            eta = (g1 + 2.0 * g0) / (g1 - g0)
            if eta > 1.0 - 1e-12:
                g_x = g0
                G_x = 0.0
            elif x <= eta:
                g_x = g0
                G_x = g0 * x
            else:
                u = (x - eta) / (1.0 - eta)
                g_x = g0 + (g1 - g0) * u * u
                G_x = g0 * x + ((1.0 - eta) / 3.0) * (g1 - g0) * u * u * u
        else:
            denom = g0 + g1
            if fabs(g0) > 1e-15 and fabs(g1) > 1e-15 and fabs(denom) > 1e-15:
                eta = g1 / denom
            else:
                eta = 0.5
            if eta < 0.001:
                eta = 0.001
            elif eta > 0.999:
                eta = 0.999
            A = -0.5 * (eta * g0 + (1.0 - eta) * g1)
            if x <= eta:
                u = (eta - x) / eta
                g_x = A + (g0 - A) * u * u
                G_x = A * x + (eta / 3.0) * (g0 - A) * (1.0 - u * u * u)
            else:
                u = (x - eta) / (1.0 - eta)
                g_x = A + (g1 - A) * u * u
                G_x = A * x + (eta / 3.0) * (g0 - A) + ((1.0 - eta) / 3.0) * (g1 - A) * u * u * u

        cum_int = zcum[seg_idx - 1] + h_seg * (fd_seg * x + G_x)
        inst_fwd = fd_seg + g_x
        return (-cum_int, inst_fwd)

    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        if n_nodes <= 1 or t <= 0.0:
            return y_nodes[0]
        if t >= t_nodes[n_nodes - 1] and self._extrap_right == EXTRAP_FLAT:
            return y_nodes[n_nodes - 1]
        cdef tuple res = self._eval_ln_df_and_fwd(t, t_nodes, y_nodes, n_nodes)
        return _ln_df_to_rate(<double>res[0], t / self._day_count_basis, self._compounding, self._freq)

    cpdef double forward_rate(self, double t, double[:] t_nodes, double[:] y_nodes) except *:
        cdef tuple res = self._eval_ln_df_and_fwd(t, &t_nodes[0], &y_nodes[0], t_nodes.shape[0])
        return <double>res[1]
