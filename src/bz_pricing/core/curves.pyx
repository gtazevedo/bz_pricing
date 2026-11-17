# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from bz_pricing.core.interpolators cimport Interpolator
import numpy as np
from libc.math cimport round, pow, trunc, exp, log, fabs
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
