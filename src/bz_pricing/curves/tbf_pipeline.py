import datetime
import pandas as pd
from bcb import sgs

from bz_pricing.curves.builder import B3CurveBuilder
from bz_pricing.data.loaders import B3DataLoader
from bz_pricing.core.curves import TBFCurve

def get_last_business_days(calendar, base_date: datetime.date, count: int) -> list:
    """
    Retorna os últimos `count` dias úteis anteriores a `base_date` (exclusive).
    """
    days = []
    current_date = base_date
    while len(days) < count:
        current_date = current_date - datetime.timedelta(days=1)
        if calendar.isbizday(current_date):
            days.append(current_date)
    return days

def compute_tbf_factor(tbf_rate: float, pre30_rate: float, du30: int = 30, du_mes: int = 21) -> float:
    """
    Calcula o Fator TBF para um determinado dia no passado.
    Fórmula B3 (Adaptada do input do usuário):
    FatorTBF_t = ((1 + TBF_t) ** (DU30 / DU_mes) - 1) / ((1 + PRE30_t) ** (DU30 / 252) - 1)
    
    tbf_rate e pre30_rate devem estar em decimal (ex: 0.10 para 10%).
    """
    num = (1.0 + tbf_rate) ** (du30 / du_mes) - 1.0
    den = (1.0 + pre30_rate) ** (du30 / 252.0) - 1.0
    
    if den == 0.0:
        return 0.0
        
    return num / den

def build_tbf_pipeline(base_date: datetime.date, data_loader: B3DataLoader, builder: B3CurveBuilder) -> TBFCurve:
    """
    Orquestra a construção da TBFxPRE (Tipo 6 - Cython):
    1. Extrai a TBF (Série 11 BCB) dos últimos 5 DU.
    2. Constrói a curva PRE dos últimos 5 DU para achar o PRE30.
    3. Calcula o FatorTBF médio.
    4. Constrói a curva PRE em t0 e envelopa tudo na `TBFCurve`.
    """
    
    # 1. Obter últimos 5 dias úteis ANTES de base_date
    past_days = get_last_business_days(builder.cal, base_date, 5)
    
    start_date = min(past_days)
    end_date = max(past_days)
    
    # Extrai TBF do BCB (Série 11)
    # A TBF no BCB geralmente vem em % ao mês.
    print(f"Extraindo TBF do BCB (Série 11) de {start_date} a {end_date}...")
    df_tbf = sgs.get({'TBF': 11}, start=start_date, end=end_date)
    
    fatores = []
    
    for d in past_days:
        # A TBF série 11 é mensal. O índice pode não bater diariamente exato, 
        # mas vamos tentar buscar o mais próximo ou assumir fallback caso o BCB mude o formato.
        # Converter d para pd.Timestamp
        d_ts = pd.Timestamp(d)
        
        if d_ts not in df_tbf.index:
            print(f"[Aviso] TBF não encontrada para a data {d}. Usaremos fallback de 1.0% ao mês para teste.")
            tbf_rate = 0.01
        else:
            tbf_rate = df_tbf.loc[d_ts, 'TBF'] / 100.0
            
        # Constrói a Curva PRE do passado (t-i)
        try:
            df_di1 = data_loader.get_curve_instruments(d, asst="DI1")
            if len(df_di1) == 0:
                print(f"[Aviso] Contratos DI1 não encontrados para {d}. Fator pulado.")
                continue
                
            pre_curve_past = builder.build_from_config(df_di1, "PRE", d, v1_rate=0.10) # CDI de fallback 10%
            
            # Pega a taxa do vértice 30 da curva interpolada
            pre30_rate = pre_curve_past.get_rate(30.0)
            
            # Por simplificação assumiremos 21 DUs no mes.
            fator = compute_tbf_factor(tbf_rate, pre30_rate, du30=30, du_mes=21)
            fatores.append(fator)
        except Exception as e:
            print(f"[Erro] Falha ao processar histórico para {d}: {e}")
            continue
        
    if not fatores:
        raise ValueError("Não foi possível calcular nenhum Fator TBF no histórico de 5 dias.")
        
    media_tbf = sum(fatores) / len(fatores)
    print(f"Média do Fator TBF calculada: {media_tbf:.6f}")
    
    # Finalmente, constrói a PRE_t0 para ser envelopada
    df_di1_t0 = data_loader.get_curve_instruments(base_date, asst="DI1")
    pre_t0 = builder.build_from_config(df_di1_t0, "PRE", base_date, v1_rate=0.1050)
    
    # Envelopa no motor Cython
    return builder.build_tbf_curve(pre_t0, media_tbf, round_decimals=5)
