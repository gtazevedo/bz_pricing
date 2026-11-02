# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from bz_pricing.core.interpolators cimport Interpolator
from cpython cimport array
import array
import array
import numpy as np
from libc.math cimport round, pow, trunc

cdef class Curve:
    """
    Interface base para todas as curvas do b3quant.
    O método get_rate resolve a taxa no tempo t (geralmente DU).
    t_sec (geralmente DC) pode ser providenciado caso o modelo matemático exija ambos.
    """
    cpdef double get_rate(self, double t, double t_sec=0.0) except *:
        raise NotImplementedError("Método base get_rate")

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
        cdef int i
        cdef Interpolator interp
        cdef double * t_sec_ptr = NULL
        
        if self._t_sec_nodes is not None:
            t_sec_ptr = &self._t_sec_nodes[0]
            
        if t <= self._t_nodes[0]:
            interp = <Interpolator>self._interpolators[0]
            return interp.interpolate(t, &self._t_nodes[0], &self._y_nodes[0], self._n_nodes, t_sec, t_sec_ptr)
        
        if t >= self._t_nodes[self._n_nodes - 1]:
            interp = <Interpolator>self._interpolators[self._n_nodes - 2]
            return interp.interpolate(t, &self._t_nodes[0], &self._y_nodes[0], self._n_nodes, t_sec, t_sec_ptr)
            
        for i in range(self._n_nodes - 1):
            if self._t_nodes[i] <= t and t <= self._t_nodes[i+1]:
                break
                
        interp = <Interpolator>self._interpolators[i]
        cdef double rate = interp.interpolate(t, &self._t_nodes[0], &self._y_nodes[0], self._n_nodes, t_sec, t_sec_ptr)
        
        if self._round_decimals >= 0:
            if self._is_trunc:
                rate = trunc(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            else:
                rate = round(rate * pow(10.0, self._round_decimals)) / pow(10.0, self._round_decimals)
            
        return rate

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
    cdef double[:] new_dc = np.ones(max_du, dtype=np.float64)
    
    return (np.asarray(new_t), np.asarray(new_y), np.asarray(new_dc))
