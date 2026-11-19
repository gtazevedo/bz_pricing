# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from bz_pricing.core.interpolators cimport Interpolator
import numpy as np
from libc.math cimport round, pow, trunc, exp, log, fabs, floor, fmax
from bz_pricing.core.conventions cimport EXPONENTIAL_B3, SIMPLE_LINEAR, COMPOUND_ANNUAL, CONTINUOUS

cdef class Curve:
    """
    Interface base para todas as curvas do b3quant.
    O método get_rate resolve a taxa no tempo t (geralmente DU).
    t_sec (geralmente DC) pode ser providenciado caso o modelo matemático exija ambos.
    """
    def __init__(self, int round_decimals=-1, str round_mode='round'):
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')

    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        raise NotImplementedError("Método base get_rate")

    @property
    def round_decimals(self):
        return self._round_decimals

    @property
    def round_mode(self):
        return 'trunc' if self._is_trunc else 'round'

cdef class FlatCurve(Curve):
    """
    Estrutura Tipo 1: Flat
    Retorna uma taxa constante cravada para todos os vértices.
    """
    def __init__(self, double rate, int round_decimals=-1, str round_mode='round'):
        self._rate = rate
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')
        
    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        cdef double rate = self._rate
        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
        return rate

cdef class AdditiveCurve(Curve):
    """
    Estrutura Tipo 2: Aditiva
    Soma os rates de um conjunto de curvas injetadas.
    """
    def __init__(self, list curves, int round_decimals=-1, str round_mode='round'):
        self._curves = curves
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')
        
    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        cdef double total = 0.0
        cdef Curve c
        for c in self._curves:
            total += c.get_rate(t, t_sec)
        
        if self._round_decimals >= 0:
            if self._is_trunc:
                total = trunc(total * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                total = round(total * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            
        return total

cdef class MultiplicativeCurve(Curve):
    """
    Estrutura Tipo 3: Multiplicativa
    Multiplica os rates de uma curva de referência por um escalar.
    """
    def __init__(self, Curve curve, double scalar, int round_decimals=-1, str round_mode='round'):
        self._curve = curve
        self._scalar = scalar
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')
        
    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        cdef double rate = self._scalar * self._curve.get_rate(t, t_sec)
        
        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            
        return rate

cdef class SimpleCurve(Curve):
    """
    Estrutura Tipo 4: Construção Simples
    Aplica um objeto Interpolator (Injeção de Dependência) em toda a curva, 
    utilizando a grade de nós fornecida.
    """
    def __init__(self, double[:] t_nodes, double[:] y_nodes, Interpolator interpolator, double[:] t_sec_nodes=None, int round_decimals=-1, str round_mode='round'):
        if t_nodes.shape[0] != y_nodes.shape[0]:
            raise ValueError("t_nodes e y_nodes devem ter o mesmo tamanho")
        if t_sec_nodes is not None and t_sec_nodes.shape[0] != t_nodes.shape[0]:
            raise ValueError("t_sec_nodes deve ter o mesmo tamanho que t_nodes se informado")
            
        self._t_nodes = t_nodes
        self._y_nodes = y_nodes
        self._t_sec_nodes = t_sec_nodes
        self._n_nodes = t_nodes.shape[0]
        self._interpolator = interpolator
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')
        
    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        cdef double * t_sec_ptr = NULL
        if self._t_sec_nodes is not None:
            t_sec_ptr = &self._t_sec_nodes[0]
            
        cdef double rate = self._interpolator.interpolate(t, &self._t_nodes[0], &self._y_nodes[0], self._n_nodes, t_sec, t_sec_ptr)
        
        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            
        return rate

    @property
    def t_nodes(self):
        import numpy as np
        return np.asarray(self._t_nodes)

    @property
    def y_nodes(self):
        import numpy as np
        return np.asarray(self._y_nodes)

    @property
    def t_sec_nodes(self):
        import numpy as np
        if self._t_sec_nodes is None:
            return None
        return np.asarray(self._t_sec_nodes)

    @property
    def interpolator(self):
        return self._interpolator


cdef class ComplexCurve(Curve):
    """
    Estrutura Tipo 5: Construção Complexa
    Aplica um Interpolator DIFERENTE para cada tranche/seção de vértices da curva.
    """
    def __init__(self, double[:] t_nodes, double[:] y_nodes, list interpolators, double[:] t_sec_nodes=None, int round_decimals=-1, str round_mode='round'):
        if t_nodes.shape[0] != y_nodes.shape[0]:
            raise ValueError("t_nodes e y_nodes devem ter o mesmo tamanho")
        
        self._t_nodes = t_nodes
        self._y_nodes = y_nodes
        self._t_sec_nodes = t_sec_nodes
        self._n_nodes = t_nodes.shape[0]
        self._interpolators = interpolators
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')
        
    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        cdef int i, lo, hi, mid
        cdef Interpolator interp
        cdef double * t_sec_ptr = NULL
        cdef double rate
        
        if self._n_nodes <= 1:
            rate = self._y_nodes[0]
        else:
            if self._t_sec_nodes is not None:
                t_sec_ptr = &self._t_sec_nodes[0]
                
            if t <= self._t_nodes[0]:
                interp = <Interpolator>self._interpolators[0]
            elif t >= self._t_nodes[self._n_nodes - 1]:
                interp = <Interpolator>self._interpolators[self._n_nodes - 2]
            else:
                lo = 0
                hi = self._n_nodes - 2
                i = 0
                while lo <= hi:
                    mid = (lo + hi) >> 1
                    if t < self._t_nodes[mid]:
                        hi = mid - 1
                    elif t > self._t_nodes[mid + 1]:
                        lo = mid + 1
                    else:
                        i = mid
                        break
                interp = <Interpolator>self._interpolators[i]
            rate = interp.interpolate(t, &self._t_nodes[0], &self._y_nodes[0], self._n_nodes, t_sec, t_sec_ptr)
        
        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            
        return rate

    @property
    def t_nodes(self):
        return np.asarray(self._t_nodes)

    @property
    def y_nodes(self):
        return np.asarray(self._y_nodes)

    @property
    def t_sec_nodes(self):
        if self._t_sec_nodes is None:
            return None
        return np.asarray(self._t_sec_nodes)

cdef class InflationCurve(Curve):
    """
    Estrutura Tipo 8: Derivativa de Inflação Implícita (Ex: DIC, DIM)
    Calculada como (1 + Nominal) / (1 + Real) - 1
    Ambas as curvas subjacentes devem estar anualizadas na base 252.
    """
    def __init__(self, Curve nominal_curve, Curve real_curve, int round_decimals=-1, str round_mode='round'):
        self._nominal_curve = nominal_curve
        self._real_curve = real_curve
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')
        
    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        cdef double pre_rate = self._nominal_curve.get_rate(t, t_sec)
        cdef double dap_rate = self._real_curve.get_rate(t, t_sec)
        
        cdef double rate = ((1.0 + pre_rate) / (1.0 + dap_rate)) - 1.0
        
        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            
        return rate

cdef class ConvenienceYieldCurve(Curve):
    """
    Estrutura a termo de custo de carrego (Convenience Yield):
    - Vértice de 1 dia útil (DU = 1): taxa de 1 dia útil da curva prefixada (PRE).
    - Vértices de vencimento futuro (DU_n):
      TxCY_n = (((Spot * (1 + PRE_n)^(DU_n/252)) / Futures_n)^(252 / DU_n) - 1)
    - Prazos intermediários: interpolação por taxa a termo exponencial em base 252.
    - Extrapolação: constante (Flat) antes do primeiro vencimento (1 < DU < DU_1) e após o último vencimento (DU > DU_k).
    As taxas são retornadas em formato decimal (% / 100).
    """
    def __init__(self, double spot_price, Curve futures_curve, Curve pre_curve, int round_decimals=-1, str round_mode='round'):
        from bz_pricing.core.interpolators import FlatForward252Interpolator
        self._spot_price = spot_price
        self._futures_curve = futures_curve
        self._pre_curve = pre_curve
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')
        self._interp_curve = None

        cdef SimpleCurve sc
        cdef int i
        cdef double du_n, fut_n, pre_n, cy_n
        cdef list v_t = []
        cdef list v_y = []

        if isinstance(futures_curve, SimpleCurve) and spot_price > 0.0:
            sc = <SimpleCurve>futures_curve
            for i in range(sc._n_nodes):
                du_n = sc._t_nodes[i]
                # Filtra o Spot (DU=1) e a âncora auxiliar de preço (DU=2 quando há vencimentos > 2)
                if du_n <= 1.0:
                    continue
                if du_n == 2.0 and sc._n_nodes > 2 and sc._t_nodes[sc._n_nodes - 1] > 2.0:
                    continue
                fut_n = sc._y_nodes[i]
                if fut_n <= 0.0:
                    continue
                pre_n = pre_curve.get_rate(du_n)
                cy_n = pow((spot_price * pow(1.0 + pre_n, du_n / 252.0)) / fut_n, 252.0 / du_n) - 1.0
                if self._round_decimals >= 0:
                    if self._is_trunc:
                        cy_n = trunc(cy_n * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
                    else:
                        cy_n = round(cy_n * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
                v_t.append(du_n)
                v_y.append(cy_n)

            if len(v_t) > 0:
                self._interp_curve = SimpleCurve(
                    np.array(v_t, dtype=np.float64),
                    np.array(v_y, dtype=np.float64),
                    FlatForward252Interpolator(extrap_left=0, extrap_right=0),
                    round_decimals=round_decimals,
                    round_mode=round_mode
                )

    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        cdef double rate
        cdef double fut_price
        cdef double pre_rate

        if t <= 1.0 or self._spot_price <= 0.0:
            rate = self._pre_curve.get_rate(1.0, t_sec)
        elif self._interp_curve is not None:
            return self._interp_curve.get_rate(t, t_sec)
        else:
            fut_price = self._futures_curve.get_rate(t, t_sec)
            pre_rate = self._pre_curve.get_rate(t, t_sec)
            if fut_price <= 0.0:
                rate = pre_rate
            else:
                rate = pow((self._spot_price * pow(1.0 + pre_rate, t / 252.0)) / fut_price, 252.0 / t) - 1.0

        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)

        return rate

    @property
    def t_nodes(self):
        if self._interp_curve is not None:
            return self._interp_curve.t_nodes
        return np.array([], dtype=np.float64)

    @property
    def y_nodes(self):
        if self._interp_curve is not None:
            return self._interp_curve.y_nodes
        return np.array([], dtype=np.float64)

cdef class SpreadCurve(Curve):
    """
    Estrutura de Curvas de Spread de Taxas:
    Spread = Rate_A - Rate_B
    """
    def __init__(self, Curve curve_a, Curve curve_b, int round_decimals=-1, str round_mode='round'):
        self._curve_a = curve_a
        self._curve_b = curve_b
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')

    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        cdef double r_a = self._curve_a.get_rate(t, t_sec)
        cdef double r_b = self._curve_b.get_rate(t, t_sec)
        cdef double rate = r_a - r_b

        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)

        return rate


cpdef list extract_linear360_knot_indices(double[:] dcs, double[:] rates, double tolerance=0.001):
    """
    Identifica os índices dos nós base de interpolação Linear 360,
    detectando mudanças de inclinação na taxa linear acumulada.
    """
    cdef int n = dcs.shape[0]
    if n <= 2:
        return list(range(n))

    cdef list knots = [0]
    cdef int curr = 0
    cdef int best_next, cand, m
    cdef double d1, r1, d2, r2, f1, f2, dm, f, val
    cdef bint ok

    while curr < n - 1:
        best_next = curr + 1
        for cand in range(curr + 1, n):
            d1 = dcs[curr]
            r1 = rates[curr]
            d2 = dcs[cand]
            r2 = rates[cand]
            f1 = (r1 * d1) / 360.0
            f2 = (r2 * d2) / 360.0
            ok = True
            for m in range(curr + 1, cand):
                dm = dcs[m]
                f = f1 + (f2 - f1) * ((dm - d1) / (d2 - d1))
                val = round((f * 360.0) / dm * 100.0) / 100.0
                if abs(val - rates[m]) > tolerance:
                    ok = False
                    break
            if ok:
                best_next = cand
            else:
                break
        knots.append(best_next)
        curr = best_next
        if curr >= n - 1:
            break
    return knots



cpdef tuple apply_cython_smoothing(double[:] t_nodes, double[:] y_nodes, double smoothing):
    """
    Realiza o tratamento otimizado de suavização de curvas.
    Filtra vértices únicos e gera o grid diário em C,
    utilizando o UnivariateSpline do SciPy por baixo.
    """
    cdef int n = t_nodes.shape[0]
    cdef int i, j
    cdef double current_t, current_y
    cdef bint found
    
    # Pré-aloca os arrays únicos com o tamanho máximo n
    cdef double[:] unique_t = np.empty(n, dtype=np.float64)
    cdef double[:] unique_y = np.empty(n, dtype=np.float64)
    cdef int u_count = 0
    
    for i in range(n):
        current_t = t_nodes[i]
        current_y = y_nodes[i]
        
        found = False
        for j in range(u_count):
            if unique_t[j] == current_t:
                found = True
                break
                
        if not found:
            unique_t[u_count] = current_t
            unique_y[u_count] = current_y
            u_count += 1
            
    # Cria views contíguas apenas do tamanho real preenchido
    cdef object t_np = np.asarray(unique_t[:u_count])
    cdef object y_np = np.asarray(unique_y[:u_count])
    
    from scipy.interpolate import UnivariateSpline
    cdef object spline = UnivariateSpline(t_np, y_np, s=smoothing)
    
    # Determina o prazo máximo
    cdef int max_du = 0
    for j in range(u_count):
        if unique_t[j] > max_du:
            max_du = <int>unique_t[j]
            
    # Cria o grid diário e evalua o spline
    cdef double[:] new_t = np.arange(1, max_du + 1, dtype=np.float64)
    cdef double[:] new_y = spline(np.asarray(new_t))
    cdef double[:] new_dc = np.maximum(1.0, np.round(np.asarray(new_t) * (365.0 / 252.0)))
    
    return (np.asarray(new_t), np.asarray(new_y), np.asarray(new_dc))


cdef inline int _coerce_curve_compounding(object compounding) except -1:
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


cdef inline double _eval_svensson_zero_cont(
    double tau,
    double beta0,
    double beta1,
    double beta2,
    double beta3,
    double lam1,
    double lam2,
) noexcept nogil:
    cdef double x1, f1, e1, x2, f2, e2
    if tau <= 1e-8:
        return beta0 + beta1
    x1 = tau / lam1
    e1 = exp(-x1)
    f1 = (1.0 - e1) / x1 if x1 > 1e-7 else (1.0 - 0.5 * x1 + (x1 * x1) / 6.0)
    x2 = tau / lam2
    e2 = exp(-x2)
    f2 = (1.0 - e2) / x2 if x2 > 1e-7 else (1.0 - 0.5 * x2 + (x2 * x2) / 6.0)
    return beta0 + beta1 * f1 + beta2 * (f1 - e1) + beta3 * (f2 - e2)


cdef inline double _eval_svensson_fwd_cont(
    double tau,
    double beta0,
    double beta1,
    double beta2,
    double beta3,
    double lam1,
    double lam2,
) noexcept nogil:
    cdef double x1, x2
    if tau <= 0.0:
        return beta0 + beta1
    x1 = tau / lam1
    x2 = tau / lam2
    return beta0 + beta1 * exp(-x1) + beta2 * x1 * exp(-x1) + beta3 * x2 * exp(-x2)


cdef inline double _cont_to_target_rate(double z_cont, double tau, int compounding, int freq) noexcept nogil:
    cdef double ln_df = -z_cont * tau
    cdef double k
    if compounding == CONTINUOUS or tau <= 1e-12:
        return z_cont
    if compounding == SIMPLE_LINEAR:
        return (exp(-ln_df) - 1.0) / tau
    if compounding == EXPONENTIAL_B3:
        return exp(z_cont) - 1.0
    k = <double>(freq if freq > 0 else 1)
    return k * (exp(z_cont / k) - 1.0)


def _target_to_cont_rates(t_arr, y_arr, day_count_basis, compounding, freq):
    taus = np.maximum(1e-8, np.asarray(t_arr, dtype=np.float64) / float(day_count_basis))
    ys = np.asarray(y_arr, dtype=np.float64)
    c_int = _coerce_curve_compounding(compounding)
    if c_int == CONTINUOUS:
        return taus, ys
    if c_int == SIMPLE_LINEAR:
        return taus, np.log1p(ys * taus) / taus
    if c_int == EXPONENTIAL_B3:
        return taus, np.log1p(ys)
    k = float(max(1, int(freq)))
    return taus, k * np.log1p(ys / k)


def _design_matrix_ns_sv(taus, lam1, lam2=None):
    x1 = np.maximum(1e-8, taus / float(lam1))
    e1 = np.exp(-x1)
    f1 = np.where(x1 > 1e-7, (1.0 - e1) / x1, 1.0 - 0.5 * x1 + (x1 * x1) / 6.0)
    cols = [np.ones_like(taus), f1, f1 - e1]
    if lam2 is not None:
        x2 = np.maximum(1e-8, taus / float(lam2))
        e2 = np.exp(-x2)
        f2 = np.where(x2 > 1e-7, (1.0 - e2) / x2, 1.0 - 0.5 * x2 + (x2 * x2) / 6.0)
        cols.append(f2 - e2)
    return np.column_stack(cols)


cdef class NelsonSiegelCurve(Curve):
    """Curva paramétrica de Nelson-Siegel (1987) para estrutura a termo de taxas de juros."""

    def __init__(
        self,
        double beta0,
        double beta1,
        double beta2,
        double lambda1,
        double day_count_basis=365.0,
        object compounding="continuous",
        int freq=1,
        int round_decimals=-1,
        str round_mode='round',
    ):
        super().__init__(round_decimals, round_mode)
        if lambda1 <= 0.0 or day_count_basis <= 0.0:
            raise ValueError("lambda1 e day_count_basis devem ser estritamente positivos.")
        self._beta0 = beta0
        self._beta1 = beta1
        self._beta2 = beta2
        self._lambda1 = lambda1
        self._day_count_basis = day_count_basis
        self._compounding = _coerce_curve_compounding(compounding)
        self._freq = max(1, freq)

    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        cdef double tau = (t if t > 1e-8 else 1e-8) / self._day_count_basis
        cdef double z_c = _eval_svensson_zero_cont(tau, self._beta0, self._beta1, self._beta2, 0.0, self._lambda1, 1.0)
        cdef double rate = _cont_to_target_rate(z_c, tau, self._compounding, self._freq)
        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
        return rate

    cpdef double instantaneous_forward(self, double t) except *:
        cdef double tau = (t if t > 0.0 else 0.0) / self._day_count_basis
        return _eval_svensson_fwd_cont(tau, self._beta0, self._beta1, self._beta2, 0.0, self._lambda1, 1.0)

    cpdef double discount_factor(self, double t) except *:
        if t <= 0.0:
            return 1.0
        cdef double tau = t / self._day_count_basis
        cdef double z_c = _eval_svensson_zero_cont(tau, self._beta0, self._beta1, self._beta2, 0.0, self._lambda1, 1.0)
        return exp(-z_c * tau)

    @classmethod
    def fit(
        cls,
        t_nodes,
        y_nodes,
        day_count_basis: float = 365.0,
        compounding="continuous",
        freq: int = 1,
    ):
        """Ajusta os parâmetros de Nelson-Siegel por projeção de variáveis sobre vértices de mercado."""
        from scipy.optimize import minimize_scalar

        taus, z_targets = _target_to_cont_rates(t_nodes, y_nodes, day_count_basis, compounding, freq)

        def _loss(lam1):
            X = _design_matrix_ns_sv(taus, lam1)
            betas, _, _, _ = np.linalg.lstsq(X, z_targets, rcond=None)
            err = X @ betas - z_targets
            return float(np.dot(err, err))

        best_res = minimize_scalar(_loss, bounds=(0.05, 15.0), method="bounded")
        lam_opt = float(best_res.x)
        X_opt = _design_matrix_ns_sv(taus, lam_opt)
        betas, _, _, _ = np.linalg.lstsq(X_opt, z_targets, rcond=None)
        return cls(
            float(betas[0]), float(betas[1]), float(betas[2]), lam_opt,
            day_count_basis=day_count_basis, compounding=compounding, freq=freq,
        )

    @property
    def params(self):
        return {
            "beta0": self._beta0,
            "beta1": self._beta1,
            "beta2": self._beta2,
            "lambda1": self._lambda1,
        }

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


cdef class SvenssonCurve(Curve):
    """Curva paramétrica de Svensson (1994), padrão oficial de bancos centrais (Fed, BCE) e ANBIMA."""

    def __init__(
        self,
        double beta0,
        double beta1,
        double beta2,
        double beta3,
        double lambda1,
        double lambda2,
        double day_count_basis=365.0,
        object compounding="continuous",
        int freq=1,
        int round_decimals=-1,
        str round_mode='round',
    ):
        super().__init__(round_decimals, round_mode)
        if lambda1 <= 0.0 or lambda2 <= 0.0 or day_count_basis <= 0.0:
            raise ValueError("lambda1, lambda2 e day_count_basis devem ser estritamente positivos.")
        self._beta0 = beta0
        self._beta1 = beta1
        self._beta2 = beta2
        self._beta3 = beta3
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._day_count_basis = day_count_basis
        self._compounding = _coerce_curve_compounding(compounding)
        self._freq = max(1, freq)

    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        cdef double tau = (t if t > 1e-8 else 1e-8) / self._day_count_basis
        cdef double z_c = _eval_svensson_zero_cont(
            tau, self._beta0, self._beta1, self._beta2, self._beta3, self._lambda1, self._lambda2
        )
        cdef double rate = _cont_to_target_rate(z_c, tau, self._compounding, self._freq)
        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
        return rate

    cpdef double instantaneous_forward(self, double t) except *:
        cdef double tau = (t if t > 0.0 else 0.0) / self._day_count_basis
        return _eval_svensson_fwd_cont(
            tau, self._beta0, self._beta1, self._beta2, self._beta3, self._lambda1, self._lambda2
        )

    cpdef double discount_factor(self, double t) except *:
        if t <= 0.0:
            return 1.0
        cdef double tau = t / self._day_count_basis
        cdef double z_c = _eval_svensson_zero_cont(
            tau, self._beta0, self._beta1, self._beta2, self._beta3, self._lambda1, self._lambda2
        )
        return exp(-z_c * tau)

    @classmethod
    def fit(
        cls,
        t_nodes,
        y_nodes,
        day_count_basis: float = 365.0,
        compounding="continuous",
        freq: int = 1,
    ):
        """Ajusta os seis parâmetros de Svensson por projeção de variáveis com múltiplos chutes iniciais."""
        from scipy.optimize import least_squares

        taus, z_targets = _target_to_cont_rates(t_nodes, y_nodes, day_count_basis, compounding, freq)

        def _res(lams):
            X = _design_matrix_ns_sv(taus, lams[0], lams[1])
            betas, _, _, _ = np.linalg.lstsq(X, z_targets, rcond=None)
            return X @ betas - z_targets

        starts = [(0.7, 2.5), (1.5, 5.0), (0.3, 1.5), (2.0, 8.0)]
        best_res = None
        for l1_0, l2_0 in starts:
            cur = least_squares(_res, [l1_0, l2_0], bounds=([0.05, 0.05], [15.0, 20.0]), max_nfev=300)
            if best_res is None or cur.cost < best_res.cost:
                best_res = cur

        lam1_opt, lam2_opt = float(best_res.x[0]), float(best_res.x[1])
        X_opt = _design_matrix_ns_sv(taus, lam1_opt, lam2_opt)
        betas, _, _, _ = np.linalg.lstsq(X_opt, z_targets, rcond=None)
        return cls(
            float(betas[0]), float(betas[1]), float(betas[2]), float(betas[3]),
            lam1_opt, lam2_opt,
            day_count_basis=day_count_basis, compounding=compounding, freq=freq,
        )

    @property
    def params(self):
        return {
            "beta0": self._beta0,
            "beta1": self._beta1,
            "beta2": self._beta2,
            "beta3": self._beta3,
            "lambda1": self._lambda1,
            "lambda2": self._lambda2,
        }

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


cdef double _cum_season_exact_c(
    const double* weights,
    int n_bins,
    double t_a,
    double t_b,
    double start_bin_frac,
) noexcept nogil:
    """Integral exata da densidade sazonal por partes em [t_a, t_b] (em anos); para pesos uniformes retorna t_b - t_a."""
    if t_b <= t_a or n_bins <= 0 or weights == NULL:
        return 0.0
    cdef double pos = start_bin_frac + t_a * (<double>n_bins)
    cdef double u_end = start_bin_frac + t_b * (<double>n_bins)
    cdef double acc = 0.0
    cdef double nxt
    cdef int k_floor, bin_idx
    while pos < u_end - 1e-13:
        k_floor = <int>floor(pos + 1e-12)
        nxt = <double>(k_floor + 1)
        if nxt > u_end:
            nxt = u_end
        bin_idx = k_floor % n_bins
        if bin_idx < 0:
            bin_idx += n_bins
        acc += weights[bin_idx] * (nxt - pos)
        pos = nxt
    return acc


cpdef double cum_seasonality_exact(double[:] weights, double t_a, double t_b, double start_bin_frac=0.0):
    """Integração analítica exata por faixas sazonais mensais (12) ou semanais (52)."""
    cdef int n_bins = <int>weights.shape[0]
    if n_bins <= 0:
        return 0.0
    cdef double[::1] w_c = np.ascontiguousarray(weights, dtype=np.float64)
    return _cum_season_exact_c(&w_c[0], n_bins, t_a, t_b, start_bin_frac)


cdef int _infer_pre_compounding(Curve pre_curve, int default_comp, object pre_compounding) except -1:
    cdef str pc_str = str(pre_compounding).strip().lower() if pre_compounding is not None else "auto"
    cdef str interp_name
    if pc_str != "auto":
        return _coerce_curve_compounding(pre_compounding)
    if hasattr(pre_curve, "compounding"):
        return _coerce_curve_compounding(pre_curve.compounding)
    if isinstance(pre_curve, SimpleCurve):
        interp_name = pre_curve.interpolator.__class__.__name__
        if interp_name in (
            "MonotoneConvexInterpolator",
            "LogLinearDiscountInterpolator",
            "LogCubicDiscountInterpolator",
            "LinearZeroInterpolator",
            "CubicSplineDiscountInterpolator",
        ):
            return CONTINUOUS
    return default_comp


cdef inline double _curve_pre_cont(Curve pre_curve, int pre_comp, double t, double t_sec, double tau) except *:
    cdef double r_raw = pre_curve.get_rate(t, t_sec)
    if pre_comp == CONTINUOUS:
        return r_raw
    if pre_comp == SIMPLE_LINEAR:
        if tau <= 1e-12:
            return r_raw
        return log(fmax(1e-12, 1.0 + r_raw * tau)) / tau
    return log(fmax(1e-12, 1.0 + r_raw))


cdef class BloombergConvenienceYieldCurve(Curve):
    """
    Estrutura a termo de Convenience Yield e Dividendos Implícitos segundo o modelo Bloomberg:
    - Interpola ações em valor presente acumulado de dividendos (D_pv) respeitando sazonalidade
      mensal ou datas ex-dividendo discretas com capitalização pelos juros sem risco;
    - Interpola índices em rendimento contínuo acumulado com sazonalidade semanal (52 semanas);
    - Extrapola o longo prazo pelo rendimento a termo anualizado do par mais longo com janela >= 1 ano
      (ou aumentado por previsões até 1 ano);
    - Permite isolar o custo de aluguel (borrow cost) a partir de dividendos declarados.
    """

    def __init__(
        self,
        double spot_price,
        Curve pre_curve,
        object t_nodes,
        object fwd_nodes,
        bint is_index=False,
        object seasonality_weights=None,
        double start_bin_frac=0.0,
        double day_count_basis=252.0,
        object compounding="exponential",
        object pre_compounding="auto",
        object ex_div_times=None,
        object ex_div_weights=None,
        object declared_div_times=None,
        object declared_div_amounts=None,
        object long_fwd_yield=None,
        object augment_yield=None,
        object augment_div_pv=None,
        double augment_horizon=1.0,
        int round_decimals=-1,
        str round_mode="round",
        dict diagnostics=None,
    ):
        super().__init__(round_decimals, round_mode)
        if spot_price <= 0.0 or day_count_basis <= 0.0:
            raise ValueError("spot_price e day_count_basis devem ser estritamente positivos.")
        self._spot_price = spot_price
        self._pre_curve = pre_curve
        self._is_index = is_index
        self._start_bin_frac = start_bin_frac
        self._day_count_basis = day_count_basis
        self._compounding = _coerce_curve_compounding(compounding)
        self._pre_compounding = _infer_pre_compounding(pre_curve, self._compounding, pre_compounding)

        t_arr = np.asarray(t_nodes, dtype=np.float64).ravel()
        f_arr = np.asarray(fwd_nodes, dtype=np.float64).ravel()
        if t_arr.shape[0] != f_arr.shape[0]:
            raise ValueError("t_nodes e fwd_nodes devem ter o mesmo tamanho.")
        if t_arr.shape[0] > 0:
            mask = (t_arr > 0.0) & (f_arr > 0.0)
            t_arr = t_arr[mask]
            f_arr = f_arr[mask]
            order = np.argsort(t_arr)
            t_arr = t_arr[order]
            f_arr = f_arr[order]

        self._n_nodes = <int>t_arr.shape[0]
        self._t_nodes = np.ascontiguousarray(t_arr, dtype=np.float64)
        self._fwd_nodes = np.ascontiguousarray(f_arr, dtype=np.float64)

        cdef int default_bins = 52 if is_index else 12
        if seasonality_weights is None:
            w_arr = np.full(default_bins, 1.0 / default_bins, dtype=np.float64)
        else:
            w_arr = np.maximum(0.0, np.asarray(seasonality_weights, dtype=np.float64).ravel())
            w_sum = float(np.sum(w_arr))
            if w_sum > 0.0:
                w_arr = w_arr / w_sum
            else:
                w_arr = np.full(len(w_arr) if len(w_arr) > 0 else default_bins, 1.0 / (len(w_arr) if len(w_arr) > 0 else default_bins), dtype=np.float64)
        self._n_bins = <int>w_arr.shape[0]
        self._season_weights = np.ascontiguousarray(w_arr, dtype=np.float64)

        r_c_nodes = np.empty(self._n_nodes, dtype=np.float64)
        d_pv_nodes = np.empty(self._n_nodes, dtype=np.float64)
        y_c_nodes = np.empty(self._n_nodes, dtype=np.float64)
        cdef int i
        cdef double t_i, tau_i, r_i, fwd_i
        for i in range(self._n_nodes):
            t_i = self._t_nodes[i]
            tau_i = t_i / self._day_count_basis
            fwd_i = self._fwd_nodes[i]
            r_i = self._get_pre_cont(t_i, 0.0, tau_i)
            r_c_nodes[i] = r_i
            d_pv_nodes[i] = spot_price - fwd_i * exp(-r_i * tau_i)
            y_c_nodes[i] = (r_i * tau_i - log(fwd_i / spot_price)) / tau_i

        self._r_cont_nodes = r_c_nodes
        self._d_pv_nodes = d_pv_nodes
        self._y_cont_nodes = y_c_nodes

        # Datas ex-dividendo discretas para interpolação de ações por evento
        if ex_div_times is not None and len(ex_div_times) > 0:
            ex_t = np.asarray(ex_div_times, dtype=np.float64).ravel()
            ex_keep = ex_t > 0.0
            ex_t = ex_t[ex_keep]
            ex_ord = np.argsort(ex_t)
            ex_t = ex_t[ex_ord]
            if ex_div_weights is not None:
                ex_w = np.maximum(0.0, np.asarray(ex_div_weights, dtype=np.float64).ravel()[ex_keep][ex_ord])
            else:
                bins_idx = np.floor(self._start_bin_frac + ex_t * self._n_bins).astype(int) % self._n_bins
                ex_w = w_arr[bins_idx]
            ex_r = np.array(
                [self._get_pre_cont(float(et) * self._day_count_basis, 0.0, float(et)) for et in ex_t],
                dtype=np.float64,
            )
            self._n_ex = <int>ex_t.shape[0]
            self._ex_div_times = np.ascontiguousarray(ex_t, dtype=np.float64)
            self._ex_div_weights = np.ascontiguousarray(ex_w, dtype=np.float64)
            self._ex_div_r_cont = np.ascontiguousarray(ex_r, dtype=np.float64)
        else:
            self._n_ex = 0
            self._ex_div_times = np.empty(0, dtype=np.float64)
            self._ex_div_weights = np.empty(0, dtype=np.float64)
            self._ex_div_r_cont = np.empty(0, dtype=np.float64)

        # Dividendos declarados para separação do custo de aluguel (borrow cost)
        if declared_div_times is not None and len(declared_div_times) > 0:
            dc_t = np.asarray(declared_div_times, dtype=np.float64).ravel()
            dc_a = np.asarray(declared_div_amounts, dtype=np.float64).ravel()
            dc_keep = dc_t > 0.0
            dc_t = dc_t[dc_keep]
            dc_a = dc_a[dc_keep]
            dc_ord = np.argsort(dc_t)
            dc_t = dc_t[dc_ord]
            dc_a = dc_a[dc_ord]
            dc_r = np.array(
                [self._get_pre_cont(float(dt_v) * self._day_count_basis, 0.0, float(dt_v)) for dt_v in dc_t],
                dtype=np.float64,
            )
            self._n_decl = <int>dc_t.shape[0]
            self._decl_div_times = np.ascontiguousarray(dc_t, dtype=np.float64)
            self._decl_div_amounts = np.ascontiguousarray(dc_a, dtype=np.float64)
            self._decl_div_r_cont = np.ascontiguousarray(dc_r, dtype=np.float64)
        else:
            self._n_decl = 0
            self._decl_div_times = np.empty(0, dtype=np.float64)
            self._decl_div_amounts = np.empty(0, dtype=np.float64)
            self._decl_div_r_cont = np.empty(0, dtype=np.float64)

        # Extrapolação de longo prazo: rendimento a termo sobre janela >= 1 ano (ou aumentada)
        cdef int i_end, i_start
        cdef double tau_end, tau_start, y_end, tau_h, d_pv_tot, r_h, f_h, y_cum_h
        if long_fwd_yield is not None:
            self._long_fwd_yield = float(long_fwd_yield)
        elif self._n_nodes == 0:
            self._long_fwd_yield = 0.0
        else:
            i_end = self._n_nodes - 1
            tau_end = self._t_nodes[i_end] / self._day_count_basis
            y_end = self._y_cont_nodes[i_end]
            i_start = -1
            for i in range(i_end - 1, -1, -1):
                if tau_end - (self._t_nodes[i] / self._day_count_basis) >= 1.0 - 1e-9:
                    i_start = i
                    break
            if i_start >= 0:
                tau_start = self._t_nodes[i_start] / self._day_count_basis
                self._long_fwd_yield = (
                    y_end * tau_end - self._y_cont_nodes[i_start] * tau_start
                ) / (tau_end - tau_start)
            elif tau_end < 1.0 - 1e-9 and (augment_div_pv is not None or augment_yield is not None):
                tau_h = augment_horizon if augment_horizon > tau_end + 1e-6 else 1.0
                if tau_h < 1.0:
                    tau_h = 1.0
                if augment_div_pv is not None:
                    d_pv_tot = self._d_pv_nodes[i_end] + float(augment_div_pv)
                    r_h = self._get_pre_cont(tau_h * self._day_count_basis, 0.0, tau_h)
                    f_h = fmax(1e-12, spot_price - d_pv_tot) * exp(r_h * tau_h)
                    y_cum_h = r_h * tau_h - log(f_h / spot_price)
                else:
                    y_cum_h = y_end * tau_end + float(augment_yield) * (tau_h - tau_end)
                self._long_fwd_yield = y_cum_h / tau_h
            elif self._n_nodes >= 2:
                tau_start = self._t_nodes[0] / self._day_count_basis
                self._long_fwd_yield = (y_end * tau_end - self._y_cont_nodes[0] * tau_start) / (tau_end - tau_start)
            else:
                self._long_fwd_yield = y_end

        self._diagnostics = dict(diagnostics) if diagnostics is not None else {}

    cdef double _get_pre_cont(self, double t, double t_sec, double tau) except *:
        return _curve_pre_cont(self._pre_curve, self._pre_compounding, t, t_sec, tau)

    cpdef double forward_price(self, double t, double t_sec=0.0) except *:
        """Retorna o preço forward implícito na maturidade `t` (mesma unidade da curva)."""
        if t <= 0.0:
            return self._spot_price
        cdef double tau = t / self._day_count_basis
        cdef double r_c = self._get_pre_cont(t, t_sec, tau)
        if self._n_nodes == 0:
            return self._spot_price * exp(r_c * tau)

        cdef int i_end = self._n_nodes - 1
        cdef double tau_end = self._t_nodes[i_end] / self._day_count_basis
        cdef double y_end_cum = self._y_cont_nodes[i_end] * tau_end
        cdef double y_cum_t

        if tau > tau_end + 1e-12:
            y_cum_t = y_end_cum + self._long_fwd_yield * (tau - tau_end)
            return self._spot_price * exp(r_c * tau - y_cum_t)

        cdef int i, lo, hi, mid, j
        for i in range(self._n_nodes):
            if fabs(t - self._t_nodes[i]) <= 1e-9:
                return self._fwd_nodes[i]

        cdef double tau_a, tau_b, d_a, d_b, y_a, y_b
        if tau <= self._t_nodes[0] / self._day_count_basis:
            tau_a = 0.0
            d_a = 0.0
            y_a = 0.0
            tau_b = self._t_nodes[0] / self._day_count_basis
            d_b = self._d_pv_nodes[0]
            y_b = self._y_cont_nodes[0] * tau_b
        else:
            lo = 0
            hi = self._n_nodes - 2
            i = 0
            while lo <= hi:
                mid = (lo + hi) >> 1
                if t < self._t_nodes[mid]:
                    hi = mid - 1
                elif t > self._t_nodes[mid + 1]:
                    lo = mid + 1
                else:
                    i = mid
                    break
            tau_a = self._t_nodes[i] / self._day_count_basis
            tau_b = self._t_nodes[i + 1] / self._day_count_basis
            d_a = self._d_pv_nodes[i]
            d_b = self._d_pv_nodes[i + 1]
            y_a = self._y_cont_nodes[i] * tau_a
            y_b = self._y_cont_nodes[i + 1] * tau_b

        cdef double span = tau_b - tau_a
        if span <= 1e-12:
            return self._spot_price * exp(r_c * tau - y_a)

        cdef double w_tot, w_part, frac
        if self._is_index:
            w_tot = _cum_season_exact_c(&self._season_weights[0], self._n_bins, tau_a, tau_b, self._start_bin_frac)
            if w_tot > 1e-12:
                w_part = _cum_season_exact_c(&self._season_weights[0], self._n_bins, tau_a, tau, self._start_bin_frac)
                frac = w_part / w_tot
            else:
                frac = (tau - tau_a) / span
            y_cum_t = y_a + (y_b - y_a) * frac
            return self._spot_price * exp(r_c * tau - y_cum_t)

        # Ações: aloca incremento de valor presente de dividendos às datas ex-dividendo ou sazonalidade mensal
        cdef double pv_w_tot = 0.0
        cdef double pv_w_part = 0.0
        cdef double ex_t, pv_w
        if self._n_ex > 0:
            for j in range(self._n_ex):
                ex_t = self._ex_div_times[j]
                if ex_t > tau_a + 1e-12 and ex_t <= tau_b + 1e-12:
                    pv_w = self._ex_div_weights[j] * exp(-self._ex_div_r_cont[j] * ex_t)
                    pv_w_tot += pv_w
                    if ex_t <= tau + 1e-12:
                        pv_w_part += pv_w
            if pv_w_tot > 1e-14:
                frac = pv_w_part / pv_w_tot
            else:
                frac = (tau - tau_a) / span
        else:
            w_tot = _cum_season_exact_c(&self._season_weights[0], self._n_bins, tau_a, tau_b, self._start_bin_frac)
            if w_tot > 1e-12:
                w_part = _cum_season_exact_c(&self._season_weights[0], self._n_bins, tau_a, tau, self._start_bin_frac)
                frac = w_part / w_tot
            else:
                frac = (tau - tau_a) / span

        cdef double d_pv_t = d_a + (d_b - d_a) * frac
        return fmax(1e-12, self._spot_price - d_pv_t) * exp(r_c * tau)

    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        """Retorna a taxa de convenience yield anualizada na convenção da curva."""
        cdef double t_eff = t if t > 1e-4 * self._day_count_basis else 1e-4 * self._day_count_basis
        cdef double tau_eff = t_eff / self._day_count_basis
        cdef double fwd = self.forward_price(t_eff, t_sec)
        cdef double r_c = self._get_pre_cont(t_eff, t_sec, tau_eff)
        cdef double y_c = (r_c * tau_eff - log(fmax(1e-12, fwd) / self._spot_price)) / tau_eff
        cdef double rate = _cont_to_target_rate(y_c, tau_eff, self._compounding, 1)
        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
        return rate

    cpdef double cash_dividend_pv(self, double t, double t_sec=0.0) except *:
        """Valor presente acumulado de dividendos + carrego líquido até a maturidade `t`."""
        if t <= 0.0:
            return 0.0
        cdef double tau = t / self._day_count_basis
        cdef double fwd = self.forward_price(t, t_sec)
        cdef double r_c = self._get_pre_cont(t, t_sec, tau)
        return self._spot_price - fwd * exp(-r_c * tau)

    cpdef double cash_dividend_fv(self, double t, double t_sec=0.0) except *:
        """Valor futuro acumulado S*exp(r*T) - F(T) até a maturidade `t`."""
        if t <= 0.0:
            return 0.0
        cdef double tau = t / self._day_count_basis
        cdef double fwd = self.forward_price(t, t_sec)
        cdef double r_c = self._get_pre_cont(t, t_sec, tau)
        return self._spot_price * exp(r_c * tau) - fwd

    cpdef double borrow_cost_rate(self, double t, double t_sec=0.0) except *:
        """Isola o custo de aluguel anualizado implícito descontando o valor presente dos dividendos declarados."""
        cdef double t_eff = t if t > 1e-4 * self._day_count_basis else 1e-4 * self._day_count_basis
        cdef double tau_eff = t_eff / self._day_count_basis
        cdef double fwd = self.forward_price(t_eff, t_sec)
        cdef double r_c = self._get_pre_cont(t_eff, t_sec, tau_eff)
        cdef double y_c = (r_c * tau_eff - log(fmax(1e-12, fwd) / self._spot_price)) / tau_eff
        cdef double pv_decl = 0.0
        cdef int j
        cdef double dt_j
        if self._n_decl > 0:
            for j in range(self._n_decl):
                dt_j = self._decl_div_times[j]
                if 0.0 < dt_j <= tau_eff + 1e-12:
                    pv_decl += self._decl_div_amounts[j] * exp(-self._decl_div_r_cont[j] * dt_j)
        cdef double q_decl = -log(fmax(1e-12, 1.0 - pv_decl / self._spot_price)) / tau_eff
        cdef double b_c = y_c - q_decl
        return _cont_to_target_rate(b_c, tau_eff, self._compounding, 1)

    def ex_dividend_schedule(self):
        """Retorna os valores em caixa alocados a cada data ex-dividendo discreta até o último vértice."""
        if self._n_ex == 0 or self._n_nodes == 0:
            return []
        out = []
        cdef int n = self._n_nodes
        cdef double tau_a, tau_b, d_a, d_b, pv_w_tot, scale, d_j, ex_t, pv_j
        cdef int k, j
        for k in range(n):
            tau_a = 0.0 if k == 0 else (self._t_nodes[k - 1] / self._day_count_basis)
            d_a = 0.0 if k == 0 else self._d_pv_nodes[k - 1]
            tau_b = self._t_nodes[k] / self._day_count_basis
            d_b = self._d_pv_nodes[k]
            pv_w_tot = 0.0
            for j in range(self._n_ex):
                ex_t = self._ex_div_times[j]
                if ex_t > tau_a + 1e-12 and ex_t <= tau_b + 1e-12:
                    pv_w_tot += self._ex_div_weights[j] * exp(-self._ex_div_r_cont[j] * ex_t)
            if pv_w_tot <= 1e-14:
                continue
            scale = (d_b - d_a) / pv_w_tot
            for j in range(self._n_ex):
                ex_t = self._ex_div_times[j]
                if ex_t > tau_a + 1e-12 and ex_t <= tau_b + 1e-12:
                    d_j = scale * self._ex_div_weights[j]
                    pv_j = d_j * exp(-self._ex_div_r_cont[j] * ex_t)
                    out.append({
                        "t_years": float(ex_t),
                        "t_curve": float(ex_t * self._day_count_basis),
                        "cash_dividend": float(d_j),
                        "pv": float(pv_j),
                    })
        return out

    def extrapolated_discrete_dividends(self, div_times, double cash_cutoff_years=10.0):
        """
        Converte os forwards da curva (incluindo extrapolação de longo prazo) em pagamentos
        discretos em caixa para t <= 10 anos e retorna o rendimento contínuo residual além desse prazo.
        """
        t_list = sorted([float(dt_v) for dt_v in div_times if float(dt_v) > 0.0])
        cash_items = []
        cdef double pv_prev = 0.0
        cdef double pv_curr, d_cash_j, r_j, d_pv_j
        for t_j in t_list:
            if t_j <= cash_cutoff_years + 1e-12:
                pv_curr = self.cash_dividend_pv(t_j * self._day_count_basis)
                d_pv_j = pv_curr - pv_prev
                r_j = self._get_pre_cont(t_j * self._day_count_basis, 0.0, t_j)
                d_cash_j = d_pv_j * exp(r_j * t_j)
                cash_items.append({"t_years": float(t_j), "cash_dividend": float(d_cash_j), "pv": float(d_pv_j)})
                pv_prev = pv_curr
        return {
            "discrete_cash_dividends": cash_items,
            "residual_long_yield": float(self._long_fwd_yield),
            "cutoff_years": float(cash_cutoff_years),
        }

    def blended_decomposition(self, double t, object div_times, object div_weights=None):
        """Aplica a decomposição híbrida Bloomberg (caixa / proporcional / contínuo) até `t`."""
        from bz_pricing.volatility.bloomberg_surface import DividendCurveBBG

        tau = t / self._day_count_basis
        r_c = self._get_pre_cont(t, 0.0, tau)
        fwd_t = self.forward_price(t)
        return DividendCurveBBG.decompose_blended_dividends(
            div_times=div_times,
            spot_f0=self._spot_price,
            fwd_T=fwd_t,
            r_cont=r_c,
            T=tau,
            is_index=self._is_index,
            div_weights=div_weights,
        )

    @classmethod
    def from_futures(
        cls,
        double spot_price,
        object futures_curve_or_data,
        Curve pre_curve,
        bint is_index=False,
        object seasonality_weights=None,
        double start_bin_frac=0.0,
        double day_count_basis=252.0,
        object compounding="exponential",
        object pre_compounding="auto",
        object ex_div_times=None,
        object ex_div_weights=None,
        object declared_div_times=None,
        object declared_div_amounts=None,
        **kwargs,
    ):
        """Constrói `BloombergConvenienceYieldCurve` a partir de preços futuros (SimpleCurve ou dict)."""
        if isinstance(futures_curve_or_data, SimpleCurve):
            sc = <SimpleCurve>futures_curve_or_data
            t_raw = np.asarray(sc.t_nodes, dtype=np.float64)
            f_raw = np.asarray(sc.y_nodes, dtype=np.float64)
            keep = t_raw > 1.0 + 1e-9
            if int(np.sum(keep)) > 1:
                max_t = float(np.max(t_raw))
                if max_t > 2.0 + 1e-9:
                    keep = keep & (np.abs(t_raw - 2.0) > 1e-9)
            t_arr = t_raw[keep]
            f_arr = f_raw[keep]
        elif isinstance(futures_curve_or_data, dict):
            items = sorted((float(k), float(v)) for k, v in futures_curve_or_data.items() if float(k) > 0.0)
            t_arr = np.array([k for k, _ in items], dtype=np.float64)
            f_arr = np.array([v for _, v in items], dtype=np.float64)
        else:
            t_arr, f_arr = futures_curve_or_data
            t_arr = np.asarray(t_arr, dtype=np.float64)
            f_arr = np.asarray(f_arr, dtype=np.float64)

        return cls(
            spot_price=spot_price,
            pre_curve=pre_curve,
            t_nodes=t_arr,
            fwd_nodes=f_arr,
            is_index=is_index,
            seasonality_weights=seasonality_weights,
            start_bin_frac=start_bin_frac,
            day_count_basis=day_count_basis,
            compounding=compounding,
            pre_compounding=pre_compounding,
            ex_div_times=ex_div_times,
            ex_div_weights=ex_div_weights,
            declared_div_times=declared_div_times,
            declared_div_amounts=declared_div_amounts,
            **kwargs,
        )

    @classmethod
    def from_options_chain(
        cls,
        object df_options,
        double spot_price,
        Curve pre_curve,
        object base_date=None,
        str exercise="EUROPEAN",
        bint is_index=False,
        object discrete_dividends=None,
        object prop_dividends=None,
        object kalman_filters=None,
        object seasonality_dates=None,
        object seasonality_amounts=None,
        object seasonality_weights=None,
        double start_bin_frac=0.0,
        double day_count_basis=252.0,
        object compounding="exponential",
        object pre_compounding="auto",
        object ex_div_times=None,
        object ex_div_weights=None,
        object declared_div_times=None,
        object declared_div_amounts=None,
        object expiry_col="XprtnDt",
        object du_col=None,
        int crr_steps=120,
        bint filter_quotes=True,
        double max_spread_short=0.0009,
        double max_spread_long=0.021,
        **kwargs,
    ):
        """
        Infere uma `BloombergConvenienceYieldCurve` diretamente de cotações de opções via paridade
        put-call, desamericanização binomial (com dividendos discretos opcionais), suavização por
        Filtro de Kalman e sazonalidade histórica/prevista.
        """
        import pandas as pd
        from bz_pricing.core.calendars import get_calendar
        from bz_pricing.volatility.bloomberg_surface import (
            DividendCurveBBG,
            deamericanize_and_infer_forward_bbg,
            filter_option_quotes_bbg,
        )

        if df_options is None or len(df_options) == 0:
            raise ValueError("df_options não pode ser vazio em from_options_chain.")
        df = df_options.copy()

        k_col = next((c for c in ("strike", "ExrcPric", "Strike", "K") if c in df.columns), None)
        tp_col = next((c for c in ("option_type", "type", "OptnTp", "Type") if c in df.columns), None)
        p_col = next((c for c in ("mid", "price", "RefPric", "LastPric") if c in df.columns), None)
        if k_col is None or tp_col is None:
            raise ValueError("df_options deve conter colunas de strike e tipo de opção.")
        has_ba = ("bid" in df.columns) and ("ask" in df.columns)
        if p_col is None and has_ba:
            df["_mid_calc"] = 0.5 * (pd.to_numeric(df["bid"], errors="coerce") + pd.to_numeric(df["ask"], errors="coerce"))
            p_col = "_mid_calc"
        if p_col is None:
            raise ValueError("df_options deve conter coluna de preço médio ou par bid/ask.")

        # Resolve prazo em unidades da curva (t_curve) por linha/vencimento
        exp_key_col = expiry_col if expiry_col in df.columns else (du_col if (du_col and du_col in df.columns) else None)
        if exp_key_col is None:
            exp_key_col = next((c for c in ("expiry", "maturity", "du", "DU", "dc", "DC") if c in df.columns), None)
        if exp_key_col is None:
            raise ValueError("df_options deve conter coluna de vencimento ou prazo.")

        cal = get_calendar("B3") if abs(day_count_basis - 252.0) < 1e-6 else None
        base_dt = pd.to_datetime(base_date).date() if base_date is not None else None

        def _resolve_t_curve(val):
            if du_col and exp_key_col == du_col:
                return float(val)
            if exp_key_col.lower() in ("du", "dc", "t", "maturity_days"):
                return float(val)
            dt_v = pd.to_datetime(val).date()
            if base_dt is None:
                raise ValueError("base_date é obrigatório quando vencimentos são informados como datas.")
            if cal is not None:
                return float(cal.bizdays(base_dt, dt_v))
            return float((dt_v - base_dt).days)

        t_by_row = np.array([_resolve_t_curve(v) for v in df[exp_key_col]], dtype=np.float64)
        df["_t_curve"] = t_by_row
        df = df[df["_t_curve"] > 0.0].copy()

        # Prepara sazonalidade caso datas e valores sejam informados
        s_weights = seasonality_weights
        if s_weights is None and seasonality_dates is not None and seasonality_amounts is not None:
            dc_tmp = DividendCurveBBG(is_index=bool(is_index))
            s_weights = dc_tmp.fit_seasonality(seasonality_dates, seasonality_amounts)

        pre_comp_int = _infer_pre_compounding(pre_curve, _coerce_curve_compounding(compounding), pre_compounding)

        def _r_cont_at(t_v):
            return _curve_pre_cont(pre_curve, pre_comp_int, float(t_v), 0.0, float(t_v) / day_count_basis)

        # Prepara dividendos discretos em anos para desamericanização e borrow cost
        disc_pairs = None
        if discrete_dividends is not None:
            if isinstance(discrete_dividends, dict):
                disc_pairs = []
                for dk, dv in discrete_dividends.items():
                    t_y = float(dk) if isinstance(dk, (int, float)) else _resolve_t_curve(dk) / day_count_basis
                    if t_y > 0.0:
                        disc_pairs.append((t_y, float(dv)))
            else:
                disc_pairs = [(float(x[0]), float(x[1])) for x in discrete_dividends if float(x[0]) > 0.0]

        t_nodes_list = []
        fwd_nodes_list = []
        expiry_diag = []

        for t_val, grp in df.groupby("_t_curve", sort=True):
            t_cur = float(t_val)
            tau_cur = t_cur / day_count_basis
            if filter_quotes:
                g_work = filter_option_quotes_bbg(
                    grp,
                    spot=spot_price,
                    maturity_years=tau_cur,
                    max_spread_short=max_spread_short,
                    max_spread_long=max_spread_long,
                ).copy()
            else:
                g_work = grp.copy()
            if g_work.empty:
                continue

            g_work["_k"] = pd.to_numeric(g_work[k_col], errors="coerce")
            g_work["_p"] = pd.to_numeric(g_work[p_col], errors="coerce")
            g_work["_is_c"] = g_work[tp_col].astype(str).str.lower().str.startswith("c")

            calls_g = g_work[g_work["_is_c"]].groupby("_k", as_index=True).first()
            puts_g = g_work[~g_work["_is_c"]].groupby("_k", as_index=True).first()
            paired = calls_g.join(puts_g, how="inner", lsuffix="_c", rsuffix="_p")
            if paired.empty:
                continue

            ks_p = paired.index.to_numpy(dtype=np.float64)
            cs_p = paired["_p_c"].to_numpy(dtype=np.float64)
            ps_p = paired["_p_p"].to_numpy(dtype=np.float64)
            c_bids = paired["bid_c"].to_numpy(dtype=np.float64) if "bid_c" in paired.columns else None
            c_asks = paired["ask_c"].to_numpy(dtype=np.float64) if "ask_c" in paired.columns else None
            p_bids = paired["bid_p"].to_numpy(dtype=np.float64) if "bid_p" in paired.columns else None
            p_asks = paired["ask_p"].to_numpy(dtype=np.float64) if "ask_p" in paired.columns else None

            r_c_eval = _r_cont_at(t_cur)
            res_da = deamericanize_and_infer_forward_bbg(
                ks_p, cs_p, ps_p,
                spot=spot_price,
                r_cont=r_c_eval,
                T=tau_cur,
                exercise=exercise,
                call_bids=c_bids,
                call_asks=c_asks,
                put_bids=p_bids,
                put_asks=p_asks,
                discrete_dividends=disc_pairs,
                prop_dividends=prop_dividends,
                crr_steps=crr_steps,
                deamericanize_all=True,
            )

            fwd_use = float(res_da["forward"])
            if kalman_filters is not None:
                kf_key = min(kalman_filters.keys(), key=lambda k: abs(float(k) - t_cur)) if len(kalman_filters) > 0 else None
                if kf_key is not None and abs(float(kf_key) - t_cur) < 1e-3:
                    kf = kalman_filters[kf_key]
                    if is_index:
                        kf_res = kf.update(
                            float(res_da["dividend_yield"]),
                            d_bid=float(res_da["dividend_yield_bid"]),
                            d_ask=float(res_da["dividend_yield_ask"]),
                        )
                        y_sm = float(kf_res["dividend"])
                        fwd_use = spot_price * exp((r_c_eval - y_sm) * tau_cur)
                    else:
                        kf_res = kf.update(
                            float(res_da["cash_dividend"]),
                            d_bid=float(res_da["cash_dividend_bid"]),
                            d_ask=float(res_da["cash_dividend_ask"]),
                        )
                        d_sm = float(kf_res["dividend"])
                        fwd_use = spot_price * exp(r_c_eval * tau_cur) - d_sm

            t_nodes_list.append(t_cur)
            fwd_nodes_list.append(fwd_use)
            res_item = dict(res_da)
            res_item["t_curve"] = t_cur
            res_item["T_years"] = tau_cur
            res_item["forward_smoothed"] = fwd_use
            expiry_diag.append(res_item)

        if not t_nodes_list:
            raise ValueError("Nenhum par call/put válido restou após a filtragem das cotações.")

        decl_t = declared_div_times
        decl_a = declared_div_amounts
        if decl_t is None and disc_pairs is not None and len(disc_pairs) > 0:
            decl_t = [x[0] for x in disc_pairs]
            decl_a = [x[1] for x in disc_pairs]
        ex_t_eff = ex_div_times
        if ex_t_eff is None and (not is_index) and disc_pairs is not None and len(disc_pairs) > 0:
            ex_t_eff = [x[0] for x in disc_pairs]

        return cls(
            spot_price=spot_price,
            pre_curve=pre_curve,
            t_nodes=np.array(t_nodes_list, dtype=np.float64),
            fwd_nodes=np.array(fwd_nodes_list, dtype=np.float64),
            is_index=is_index,
            seasonality_weights=s_weights,
            start_bin_frac=start_bin_frac,
            day_count_basis=day_count_basis,
            compounding=compounding,
            pre_compounding=pre_compounding,
            ex_div_times=ex_t_eff,
            ex_div_weights=ex_div_weights,
            declared_div_times=decl_t,
            declared_div_amounts=decl_a,
            diagnostics={"expiries": expiry_diag},
            **kwargs,
        )

    @classmethod
    def from_futures_and_options(
        cls,
        double spot_price,
        object futures_curve_or_data,
        object df_options,
        Curve pre_curve,
        double tol_duplicate_t=1e-3,
        **kwargs,
    ):
        """Funde pilares líquidos de contratos futuros com forwards implícitos de opções."""
        fut_keys = {
            "is_index", "seasonality_weights", "start_bin_frac", "day_count_basis",
            "compounding", "pre_compounding", "ex_div_times", "ex_div_weights",
            "declared_div_times", "declared_div_amounts", "long_fwd_yield",
            "augment_yield", "augment_div_pv", "augment_horizon", "round_decimals", "round_mode",
        }
        fut_kwargs = {k: v for k, v in kwargs.items() if k in fut_keys}
        fut_cy = cls.from_futures(spot_price, futures_curve_or_data, pre_curve, **fut_kwargs)
        opt_cy = cls.from_options_chain(df_options, spot_price, pre_curve, **kwargs)

        merged = {}
        sources = {}
        for t_o, f_o in zip(opt_cy.t_nodes, opt_cy.fwd_nodes):
            merged[float(t_o)] = float(f_o)
            sources[float(t_o)] = "option_implied"
        for t_f, f_f in zip(fut_cy.t_nodes, fut_cy.fwd_nodes):
            matched_k = next((k for k in list(merged.keys()) if abs(k - float(t_f)) <= tol_duplicate_t), None)
            if matched_k is not None and matched_k != float(t_f):
                del merged[matched_k]
                del sources[matched_k]
            merged[float(t_f)] = float(f_f)
            sources[float(t_f)] = "futures"

        t_sorted = sorted(merged.keys())
        f_sorted = [merged[k] for k in t_sorted]
        diag = dict(opt_cy.diagnostics)
        diag["sources"] = {k: sources[k] for k in t_sorted}

        return cls(
            spot_price=spot_price,
            pre_curve=pre_curve,
            t_nodes=np.array(t_sorted, dtype=np.float64),
            fwd_nodes=np.array(f_sorted, dtype=np.float64),
            is_index=opt_cy.is_index,
            seasonality_weights=opt_cy.seasonality_weights,
            start_bin_frac=opt_cy.start_bin_frac,
            day_count_basis=opt_cy.day_count_basis,
            compounding=kwargs.get("compounding", "exponential"),
            pre_compounding=kwargs.get("pre_compounding", "auto"),
            ex_div_times=kwargs.get("ex_div_times", None) if "ex_div_times" in kwargs else (opt_cy.ex_div_times if len(opt_cy.ex_div_times) > 0 else None),
            ex_div_weights=kwargs.get("ex_div_weights", None),
            declared_div_times=kwargs.get("declared_div_times", None) if "declared_div_times" in kwargs else (opt_cy.decl_div_times if len(opt_cy.decl_div_times) > 0 else None),
            declared_div_amounts=kwargs.get("declared_div_amounts", None) if "declared_div_amounts" in kwargs else (opt_cy.decl_div_amounts if len(opt_cy.decl_div_amounts) > 0 else None),
            diagnostics=diag,
        )

    @property
    def spot_price(self):
        return self._spot_price

    @property
    def t_nodes(self):
        return np.asarray(self._t_nodes)

    @property
    def fwd_nodes(self):
        return np.asarray(self._fwd_nodes)

    @property
    def y_cont_nodes(self):
        return np.asarray(self._y_cont_nodes)

    @property
    def d_pv_nodes(self):
        return np.asarray(self._d_pv_nodes)

    @property
    def seasonality_weights(self):
        return np.asarray(self._season_weights)

    @property
    def ex_div_times(self):
        return np.asarray(self._ex_div_times)

    @property
    def decl_div_times(self):
        return np.asarray(self._decl_div_times)

    @property
    def decl_div_amounts(self):
        return np.asarray(self._decl_div_amounts)

    @property
    def is_index(self):
        return self._is_index

    @property
    def start_bin_frac(self):
        return self._start_bin_frac

    @property
    def day_count_basis(self):
        return self._day_count_basis

    @property
    def long_fwd_yield(self):
        return self._long_fwd_yield

    @property
    def diagnostics(self):
        return self._diagnostics
