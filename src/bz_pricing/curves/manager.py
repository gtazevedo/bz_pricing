import datetime
import pandas as pd
from bcb import sgs

from bz_pricing.curves.builder import B3CurveBuilder
from bz_pricing.data.loaders import B3DataLoader
from bz_pricing.core.curves import Curve
from bz_pricing.curves.configs import CURVE_CONFIGS

class CurveManager:
    """
    Orquestrador genérico para construção de curvas de juros.
    Abstrai a lógica de extração de dados (SGS, B3, ANBIMA) e a construção de curvas derivadas.
    """
    
    def __init__(self, loader: B3DataLoader = None, builder: B3CurveBuilder = None):
        self.loader = loader if loader is not None else B3DataLoader()
        self.builder = builder if builder is not None else B3CurveBuilder()
        
    def get_curve(self, curve_name: str, base_date: datetime.date) -> Curve:
        """
        Retorna uma Curva (SimpleCurve, InflationCurve, TBFCurve, TRCurve) 
        baseada nas definições do configs.py
        """
        if curve_name not in CURVE_CONFIGS:
            raise ValueError(f"Configuração da curva '{curve_name}' não encontrada no configs.py")
            
        config = CURVE_CONFIGS[curve_name]
        curve_type = config.get("type", "simple")
        
        if curve_type == "simple":
            return self._build_simple_curve(curve_name, config, base_date)
        elif curve_type == "derived":
            return self._build_derived_curve(curve_name, config, base_date)
        elif curve_type == "flat":
            return self._build_flat_curve(curve_name, config, base_date)
        else:
            raise NotImplementedError(f"Tipo de curva '{curve_type}' não suportado.")

    def _build_simple_curve(self, curve_name: str, config: dict, base_date: datetime.date) -> Curve:
        # Extrair Dataset
        asset = config.get("asset", curve_name)
        
        # Loader especial para LTN usando arquivo da ANBIMA MS
        if curve_name == "LTN":
            df = self.loader.get_anbima_bonds(base_date, "LTN")
        else:
            df = self.loader.get_curve_instruments(base_date, asst=asset)
            
        if len(df) == 0:
            raise ValueError(f"Dataset vazio para a curva '{curve_name}' (Ativo: {asset}) na data {base_date}.")
            
        # Extrair Vértice 1 (Se necessário)
        v1_rate = None
        v1_source = config.get("v1_source", None)
        
        if v1_source == "CDI":
            df_bcb = sgs.get({'CDI': 12}, start=base_date, end=base_date)
            if not df_bcb.empty:
                v1_rate = df_bcb['CDI'].iloc[0] / 100.0
                v1_rate = (1.0 + v1_rate) ** 252.0 - 1.0 # Anualiza
            else:
                v1_rate = 0.1040 # Fallback
                
        elif v1_source == "SELIC":
            df_bcb = sgs.get({'SELIC': 4389}, start=base_date, end=base_date)
            if not df_bcb.empty:
                v1_rate = df_bcb['SELIC'].iloc[0] / 100.0
                # A SELIC (SGS 4389) já é uma taxa anualizada base 252, não deve ser elevada a 252!

            else:
                v1_rate = 0.1050 # Fallback
                
        # Constrói a curva via builder (o interpolador já estará na config)
        # Suavização Cython pode ser passada como arg extra (hardcoded para DI1/PRE para testes)
        smoothing = 0.0001 if curve_name == "PRE" else None
        
        return self.builder.build_from_config(
            df=df,
            config_name=curve_name,
            base_date=base_date,
            v1_rate=v1_rate,
            smoothing=smoothing,
            use_cython_smoothing=False
        )
        
    def _build_derived_curve(self, curve_name: str, config: dict, base_date: datetime.date) -> Curve:
        # Se for TBF_PRE, precisa da base_curve e histórico BCB
        if curve_name == "TBF_PRE":
            base_curve_name = config.get("base_curve", "PRE")
            return self._build_tbf_pipeline(base_date, base_curve_name)
            
        # Se for TR_PRE ou TR_LTN, puxa a base TBF
        elif curve_name in ["TR_PRE", "TR_LTN"]:
            base_curve_name = config.get("base_curve")
            tbf_curve = self.get_curve(base_curve_name, base_date)
            round_decimals = config.get("round_decimals", -1)
            return self.builder.build_tr_curve(tbf_curve, round_decimals=round_decimals)
            
        elif curve_name == "TBF_LTN":
            ltn_curve = self.get_curve("LTN", base_date)
            return self.builder.build_tbf_curve(ltn_curve, 0.93, round_decimals=config.get("round_decimals", -1))
            
        # Se for Curva de Inflação (DIC/DIM)
        elif curve_name in ["DIC", "DIM"]:
            nominal_curve_name = config.get("nominal_curve", "PRE")
            real_curve_name = config.get("real_curve", "DAP" if curve_name == "DIC" else "DGI")
            
            nominal_curve = self.get_curve(nominal_curve_name, base_date)
            real_curve = self.get_curve(real_curve_name, base_date)
            return self.builder.build_inflation_curve(nominal_curve, real_curve, config_name=curve_name)
            
        else:
            raise NotImplementedError(f"Lógica de Curva Derivada não implementada para {curve_name}")

    def _build_flat_curve(self, curve_name: str, config: dict, base_date: datetime.date) -> Curve:
        # Lógica genérica de flat curves
        source = config.get("source")
        if source == "BCB":
            series = config.get("series_id")
            df_bcb = sgs.get({curve_name: series}, start=base_date, end=base_date)
            if not df_bcb.empty:
                rate = df_bcb[curve_name].iloc[0] / 100.0
            else:
                rate = 0.05
            return self.builder.build_flat_curve(rate, config_name=curve_name)
        else:
            raise NotImplementedError("Source de flat curve não implementado.")

    def _build_tbf_pipeline(self, base_date: datetime.date, base_curve_name: str) -> Curve:
        """
        Substitui o tbf_pipeline isolado. Resolve a TBF iterando os 5 dias passados recursivamente.
        """
        def get_last_business_days(calendar, d: datetime.date, count: int) -> list:
            days = []
            curr = d
            while len(days) < count:
                curr = curr - datetime.timedelta(days=1)
                if calendar.isbizday(curr):
                    days.append(curr)
            return days

        past_days = get_last_business_days(self.builder.cal, base_date, 5)
        start_date = min(past_days)
        end_date = max(past_days)
        
        # A série 253 é % a.m.
        df_tbf = sgs.get({'TBF': 253}, start=start_date, end=end_date)
        fatores = []
        
        for d in past_days:
            d_ts = pd.Timestamp(d)
            if d_ts not in df_tbf.index:
                continue
                
            tbf_am = df_tbf.loc[d_ts, 'TBF'] / 100.0
                
            try:
                # Chama recursivamente para pegar a curva base no dia D-i
                pre_curve_past = self.get_curve(base_curve_name, d)
                pre30_rate = pre_curve_past.get_rate(30.0)
                
                # DU_30 é os dias úteis correspondentes ao vértice de 30 dias corridos (tipicamente 21)
                # DU_mes é os dias úteis do mês da TBF (tipicamente 21)
                # Portanto DU_30 / DU_mes = 1.0
                tbf_30 = tbf_am
                # A taxa PRE_30 (vértice 30 dias corridos = 21 DU) deve ser elevada a 21/252
                pre_30 = (1.0 + pre30_rate) ** (21.0 / 252.0) - 1.0
                
                fator = tbf_30 / pre_30 if pre_30 != 0.0 else 0.0
                fatores.append(fator)
            except Exception:
                continue
                
        if not fatores:
            media_tbf = 0.93
        else:
            media_tbf = sum(fatores) / len(fatores)
        
        # Gera a curva t0
        pre_t0 = self.get_curve(base_curve_name, base_date)
        config = CURVE_CONFIGS["TBF_PRE"]
        return self.builder.build_tbf_curve(pre_t0, media_tbf, round_decimals=config.get("round_decimals", -1))
