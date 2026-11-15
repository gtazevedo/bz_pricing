# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport pow, log, exp, fabs

from bz_pricing.core.conventions cimport DayCountConvention, CompoundingType
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
