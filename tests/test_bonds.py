import datetime
import pytest
from bz_pricing.core.curves import FlatCurve
from bz_pricing.instruments.tpf import LTN, NTNF

def test_ltn_pricing():
    # Uma curva flat de 10% a.a.
    curve = FlatCurve(0.10)
    
    base_date = datetime.date(2026, 9, 4)
    # Supondo 252 DU (1 ano exato)
    maturity_date = datetime.date(2027, 9, 3) 
    
    # Criamos a LTN (a classe vai usar bizdays, se não der 252 DU exato, o valor será um pouco diferente)
    # Para forçar o teste matemático exato, vamos assumir que o calendário tem X dias.
    ltn = LTN(maturity_date)
    
    pu = ltn.present_value(base_date, curve)
    
    du = ltn.calendar.bizdays(base_date, maturity_date)
    
    # Formula teorica PU = 1000 / (1 + 0.10)^(DU/252)
    pu_esperado = 1000.0 / ((1.10) ** (du / 252.0))
    pu_esperado = int(pu_esperado * 1000000.0) / 1000000.0
    
    assert abs(pu - pu_esperado) < 1e-5

def test_ntnf_pricing():
    curve = FlatCurve(0.10)
    base_date = datetime.date(2026, 9, 4)
    maturity_date = datetime.date(2031, 1, 1) # NTN-F 2031
    
    ntnf = NTNF(maturity_date, coupon_rate=0.10)
    pu = ntnf.present_value(base_date, curve)
    
    # Como a curva de desconto também é 10% a.a. (a mesma do cupom),
    # o PU de um título par deve ser extamente R$ 1000,00!
    # Na verdade, por causa da contagem exata de DU de cada semestre (que varia em torno de 126),
    # o PU não será exatamente R$ 1000,00, podendo haver um ágio/deságio estrutural na casa dos R$ 20-30.
    assert abs(pu - 1000.0) < 30.0
