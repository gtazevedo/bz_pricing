import numpy as np
import pandas as pd
import datetime

from bz_pricing.core.bonds import GenericCashflowBond, LTN
from bz_pricing.data.loaders import B3DataLoader
from bz_pricing.core.vna_engine import VNAEngine

class TPFBuilder:
    """
    Fábrica de Títulos Públicos (Data-Driven).
    """
    def __init__(self, data_loader: B3DataLoader, cal):
        self.loader = data_loader
        self.cal = cal

    def build_ltn(self, base_date: datetime.date, exp_date: datetime.date, face_value: float = 1000.0) -> LTN:
        """
        LTN não tem cupom, então não precisamos de dados históricos de fluxo.
        Basta o DU até o vencimento.
        """
        du = self.cal.bizdays(base_date, exp_date)
        return LTN(du, face_value=face_value)

    def build_from_empirical_cashflows(self, 
                                       base_date: datetime.date, 
                                       titulo: str, 
                                       vencimento: datetime.date,
                                       vna_engine: VNAEngine = None) -> GenericCashflowBond:
        """
        Constrói o título usando as datas determinísticas de pagamento e projeta 
        os valores futuros (incluindo VNA para NTN-B).
        Pode cruzar com o CSV do Tesouro Transparente caso haja fluxos listados.
        """
        try:
            df_cupons = self.loader.load_tesouro_transparente_cupons()
        except FileNotFoundError:
            df_cupons = pd.DataFrame()

        current_date = pd.to_datetime(base_date).date()
        end_date = pd.to_datetime(vencimento).date()
        
        dus = []
        cashflows = []
        
        # Mapeamento Tesouro Transparente e meses de cupom
        if titulo.startswith("NTN-F") or titulo == "Tesouro Prefixado com Juros Semestrais":
            coupon_rate = 0.10
            coupon_amount = 1000.0 * ((1.0 + coupon_rate)**0.5 - 1.0)
            rounded_coupon = np.trunc(coupon_amount * 100000.0) / 100000.0
            coupon_months = [1, 7]
            has_vna = False
            
        elif titulo.startswith("NTN-B") or titulo == "Tesouro IPCA+ com Juros Semestrais":
            coupon_rate = 0.06
            coupon_amount_factor = ((1.0 + coupon_rate)**0.5 - 1.0)
            # NTN-B paga em Maio e Novembro, exceto as vezes em Fev/Ago dependendo da emissão, mas o padrão é 5 e 11
            # O vencimento dita o ciclo. Se vence no mês M, paga em M e M-6/M+6
            m1 = end_date.month
            m2 = m1 - 6 if m1 > 6 else m1 + 6
            coupon_months = sorted([m1, m2])
            has_vna = True
            if vna_engine is None:
                raise ValueError("VNAEngine é obrigatório para precificar NTN-B.")
        else:
            raise ValueError(f"Título não suportado para geração de fluxo: {titulo}")
            
        y = current_date.year
        while True:
            for m in coupon_months:
                # O dia é o mesmo do vencimento (geralmente dia 15 para NTN-B, dia 1 para NTN-F)
                dt = datetime.date(y, m, end_date.day)
                if dt > current_date and dt <= end_date:
                    # Ajusta para dia útil
                    dt_adj = self.cal.following(dt) if hasattr(self.cal, 'following') else self.cal.adjust_next(dt)
                    dus.append(self.cal.bizdays(current_date, dt_adj))
                    
                    if has_vna:
                        vna_proj = vna_engine.calculate_vna(dt_adj, self.cal)
                        c_val = np.trunc((vna_proj * coupon_amount_factor) * 100000.0) / 100000.0
                        cashflows.append(c_val)
                    else:
                        cashflows.append(rounded_coupon)
            y += 1
            if datetime.date(y, 1, 1) > end_date:
                break
                
        if len(cashflows) > 0:
            if has_vna:
                vna_final = vna_engine.calculate_vna(end_date, self.cal)
                cashflows[-1] += vna_final
            else:
                cashflows[-1] += 1000.0 # Principal no final
                
        # Converte para arrays tipados numpy para enviar ao Cython
        dus_arr = np.array(dus, dtype=np.int32)
        cf_arr = np.array(cashflows, dtype=np.float64)
        
        return GenericCashflowBond(dus_arr, cf_arr)
