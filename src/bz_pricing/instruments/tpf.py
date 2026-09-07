import datetime
import bizdays
import numpy as np
import pandas as pd

from bz_pricing.core.curves import SimpleCurve
from bz_pricing.core.bonds import LTN as _LTN_C, NTNF as _NTNF_C

class TituloPublico:
    def __init__(self, maturity_date: datetime.date, calendar_name: str = 'B3'):
        self.maturity_date = maturity_date
        
        if calendar_name == 'B3':
            self.calendar = bizdays.Calendar.load('B3')
        else:
            self.calendar = bizdays.Calendar(weekdays=[], name='Actual')
        
    def present_value(self, base_date: datetime.date, curve: SimpleCurve) -> float:
        raise NotImplementedError

class LTN(TituloPublico):
    """
    Letra do Tesouro Nacional (Zero-Coupon)
    """
    def __init__(self, maturity_date: datetime.date, face_value: float = 1000.0, calendar_name: str = 'B3'):
        super().__init__(maturity_date, calendar_name)
        self.face_value = face_value
        
    def present_value(self, base_date: datetime.date, curve: SimpleCurve) -> float:
        if base_date >= self.maturity_date:
            return self.face_value
            
        du = self.calendar.bizdays(base_date, self.maturity_date)
        
        # Chama a engine C++
        bond_c = _LTN_C(du, self.face_value)
        return bond_c.present_value(curve)

class NTNF(TituloPublico):
    """
    Nota do Tesouro Nacional - Série F (Cupons Semestrais Fixos)
    """
    def __init__(self, maturity_date: datetime.date, coupon_rate: float, face_value: float = 1000.0, calendar_name: str = 'B3'):
        super().__init__(maturity_date, calendar_name)
        self.coupon_rate = coupon_rate
        self.face_value = face_value
        
    def _get_coupon_dates(self, base_date: datetime.date):
        """
        Gera as datas de pagamento de cupons da NTN-F (01/01 e 01/07) a partir da data base
        até a data de vencimento.
        """
        dates = []
        curr_year = base_date.year
        
        while True:
            d1 = datetime.date(curr_year, 1, 1)
            d2 = datetime.date(curr_year, 7, 1)
            
            for d in [d1, d2]:
                adj_d = self.calendar.following(d)
                if base_date < adj_d <= self.maturity_date:
                    if adj_d not in dates:
                        dates.append(adj_d)
                        
            if curr_year >= self.maturity_date.year:
                break
                
            curr_year += 1
            
        # Garante que o vencimento está na lista se por acaso for um dia diferente
        mat_adj = self.calendar.following(self.maturity_date)
        if mat_adj not in dates and mat_adj > base_date:
            dates.append(mat_adj)
            
        return sorted(dates)

    def present_value(self, base_date: datetime.date, curve: SimpleCurve) -> float:
        if base_date >= self.maturity_date:
            return self.face_value
            
        coupon_dates = self._get_coupon_dates(base_date)
        
        # Converte as datas para DU (Dias Úteis)
        dus = np.array([self.calendar.bizdays(base_date, d) for d in coupon_dates], dtype=np.int32)
        
        # Chama a engine C++
        bond_c = _NTNF_C(dus, self.coupon_rate, self.face_value)
        return bond_c.present_value(curve)
