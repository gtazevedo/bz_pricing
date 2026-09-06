# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport pow, log, exp

from bz_pricing.core.conventions cimport DayCountConvention, CompoundingType
from bz_pricing.core.interpolators cimport ExtrapolationType, EXTRAP_FLAT, EXTRAP_FLAT_FORWARD, EXTRAP_LINEAR

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
        cdef int i
        # TRATAMENTO DE EXTRAPOLAÇÃO
        if t <= t_nodes[0]: # Extrapolação (Início)
            if self._extrap_left == EXTRAP_FLAT:
                return y_nodes[0]
            # Se não for Flat, forçamos o índice 0. O código pegará t_nodes[0] e t_nodes[1]
            # e projetará a fórmula matematicamente para trás (t < t_nodes[0]).
            i = 0
        elif t >= t_nodes[n_nodes - 1]: # Extrapolação (Fim)
            if self._extrap_right == EXTRAP_FLAT:
                return y_nodes[n_nodes - 1]
            # Se não for Flat, forçamos os dois últimos índices. O código projetará 
            # a fórmula matematicamente para a frente, cumprindo o Flat Forward 252 (Fim).
            i = n_nodes - 2
        else:
            for i in range(n_nodes - 1):
                if t_nodes[i] <= t and t <= t_nodes[i+1]:
                    break
                    
        cdef double t1 = t_nodes[i]
        cdef double t2 = t_nodes[i+1]
        cdef double y1 = y_nodes[i]
        cdef double y2 = y_nodes[i+1]
        
        if t1 == t2:
            return y1
            
        cdef double f1 = 1.0 + y1
        cdef double f2 = 1.0 + y2
        """
        y = f1 * (f2/f1) ^((t - t1) / (t2 - t1)) - 1
        onde:
            f1 e o fator do vertice anterior
            f2 e o fator do vertice posterior
            t e o vertice atual
            t1 e o vertice anterior
            t2 e o vertice psoterior
        O resultado e y, em formato decimal
        """
        cdef double factor = f1 * pow(f2 / f1, (t - t1) / (t2 - t1))
        return factor - 1.0


cdef class FlatForward252Interpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        cdef int i
        if t <= t_nodes[0]:
            if self._extrap_left == EXTRAP_FLAT:
                return y_nodes[0]
            i = 0
        elif t >= t_nodes[n_nodes - 1]:
            if self._extrap_right == EXTRAP_FLAT:
                return y_nodes[n_nodes - 1]
            i = n_nodes - 2
        else:
            for i in range(n_nodes - 1):
                if t_nodes[i] <= t and t <= t_nodes[i+1]:
                    break
                    
        cdef double t1 = t_nodes[i]
        cdef double t2 = t_nodes[i+1]
        cdef double y1 = y_nodes[i]
        cdef double y2 = y_nodes[i+1]
        
        if t1 == t2:
            return y1
            
        cdef double f1 = pow(1.0 + y1, t1 / 252.0)
        cdef double f2 = pow(1.0 + y2, t2 / 252.0)

        """
        y = (f1 ^(t1/252) * ((f2^(t2/252))/(f1^(t1/252)))^((t-t1)/(t2-t1)))^(252/t)-1
        Onde:
            f1 e o fator do vertice anterior
            f2 e o fator do vertice posterior
            t e o vertice atual
            t1 e o vertice anterior
            t2 e o vertice psoterior
        O resultado e y, em formato decimal
        """        
        cdef double forward = f1 * pow(f2 / f1, (t - t1) / (t2 - t1))
        return pow(forward, 252.0 / t) - 1.0


cdef class FlatForward252LinearInterpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        if t_sec_nodes == NULL:
            raise ValueError("t_sec_nodes (DC) é obrigatório para FlatForward252LinearInterpolator")
            
        cdef int i
        if t <= t_nodes[0]:
            if self._extrap_left == EXTRAP_FLAT:
                return y_nodes[0]
            i = 0
        elif t >= t_nodes[n_nodes - 1]:
            if self._extrap_right == EXTRAP_FLAT:
                return y_nodes[n_nodes - 1]
            i = n_nodes - 2
        else:
            for i in range(n_nodes - 1):
                if t_nodes[i] <= t and t <= t_nodes[i+1]:
                    break
                    
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
        """
        y = ((1+y1*dc1/360)*((1+y2*dc2/360)/(1+y1*dc1/360))^((du-du1)/(du2-du1))-1)*360/dc
        Onde:
            f1 e o fator do vertice anterior
            f2 e o fator do vertice posterior
            du e o vertice atual (Dias Uteis)
            dc e o vertice atual (Dias Corridos)
            du1, dc1 sao os prazos do vertice anterior
            du2, dc2 sao os prazos do vertice posterior
        O resultado e y, em formato decimal
        """
        cdef double forward = f1 * pow(f2 / f1, (t - du1) / (du2 - du1))
        
        if t_sec == 0.0:
            raise ValueError("t_sec (DC atual) deve ser informado não-zero")
            
        return (forward - 1.0) * (360.0 / t_sec)


cdef class Interpolation360Interpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        # Para esta, t é DC
        cdef int i
        if t <= t_nodes[0]:
            if self._extrap_left == EXTRAP_FLAT:
                return y_nodes[0]
            i = 0
        elif t >= t_nodes[n_nodes - 1]:
            if self._extrap_right == EXTRAP_FLAT:
                return y_nodes[n_nodes - 1]
            i = n_nodes - 2
        else:
            for i in range(n_nodes - 1):
                if t_nodes[i] <= t and t <= t_nodes[i+1]:
                    break
                    
        cdef double t1 = t_nodes[i]
        cdef double t2 = t_nodes[i+1]
        cdef double y1 = y_nodes[i]
        cdef double y2 = y_nodes[i+1]
        
        if t1 == t2:
            return y1
            
        cdef double f1 = pow(1.0 + y1, t1 / 360.0)
        cdef double f2 = pow(1.0 + y2, t2 / 360.0)
        
        cdef double forward = f1 * pow(f2 / f1, (t - t1) / (t2 - t1))
        """
        y = (f1 ^(t1/360) * ((f2^(t2/360))/(f1^(t1/360)))^((t-t1)/(t2-t1)))^(360/t)-1
        Onde:
            f1 e o fator do vertice anterior
            f2 e o fator do vertice posterior
            t e o vertice atual
            t1 e o vertice anterior
            t2 e o vertice psoterior
        O resultado e y, em formato decimal
        """       
        return pow(forward, 360.0 / t) - 1.0


cdef class Linear360Interpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        cdef int i
        if t <= t_nodes[0]:
            if self._extrap_left == EXTRAP_FLAT:
                return y_nodes[0]
            i = 0
        elif t >= t_nodes[n_nodes - 1]:
            if self._extrap_right == EXTRAP_FLAT:
                return y_nodes[n_nodes - 1]
            i = n_nodes - 2
        else:
            for i in range(n_nodes - 1):
                if t_nodes[i] <= t and t <= t_nodes[i+1]:
                    break
                    
        cdef double t1 = t_nodes[i]
        cdef double t2 = t_nodes[i+1]
        cdef double y1 = y_nodes[i]
        cdef double y2 = y_nodes[i+1]
        
        if t1 == t2:
            return y1
            
        cdef double f1 = (y1 * t1) / 360.0
        cdef double f2 = (y2 * t2) / 360.0
        
        cdef double forward = f1 + (f2 - f1) * ((t - t1) / (t2 - t1))
        """
        y = ((y1*t1/360) + (y2*t2/360 - y1*t1/360) * ((t-t1)/(t2-t1))) * 360/t
        Onde:
            y1 e a taxa do vertice anterior
            y2 e a taxa do vertice posterior
            t e o vertice atual
            t1 e o vertice anterior
            t2 e o vertice psoterior
        O resultado e y, em formato decimal
        """
        return (forward * 360.0) / t


cdef class PriceInterpolator(Interpolator):
    cdef double interpolate(self, double t, double *t_nodes, double *y_nodes, int n_nodes, double t_sec=0.0, double *t_sec_nodes=NULL) except *:
        # 1.4.5 Interpolação de Preços
        cdef int i
        if t <= t_nodes[0]:
            if self._extrap_left == EXTRAP_FLAT:
                return y_nodes[0]
            i = 0
        elif t >= t_nodes[n_nodes - 1]:
            if self._extrap_right == EXTRAP_FLAT:
                return y_nodes[n_nodes - 1]
            i = n_nodes - 2
        else:
            for i in range(n_nodes - 1):
                if t_nodes[i] <= t and t <= t_nodes[i+1]:
                    break
                    
        cdef double t1 = t_nodes[i]
        cdef double t2 = t_nodes[i+1]
        cdef double p1 = y_nodes[i]
        cdef double p2 = y_nodes[i+1]
        
        if t1 == t2:
            return p1
            
        """
        p = p1 * ((p2/p1)^((t-t1)/(t2-t1)))
        Onde:
            p1 e o preço do vertice anterior
            p2 e o preço do vertice posterior
            t e o vertice atual
            t1 e o vertice anterior
            t2 e o vertice psoterior
        O resultado e p, em formato decimal
        """
        return p1 * pow(p2 / p1, (t - t1) / (t2 - t1))

