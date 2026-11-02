import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from bcb import sgs

from bz_pricing.curves.builder import B3CurveBuilder
from bz_pricing.data.loaders import B3DataLoader
from bz_pricing.data.scrapers import B3Scraper
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
        self.scraper = B3Scraper(output_dir=self.loader.data_dir)
        
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

    def _fetch_sgs_rate(self, series_id: int, base_date: datetime.date, fallback: float) -> float:
        """
        Busca a última taxa disponível na série SGS do BCB usando janela de 7 dias corridos
        (evita erro na API do BCB quando dataInicial == dataFinal).
        Recebe start e end date formatados como string '%Y-%m-%d'.
        """
        start_str = (base_date - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        end_str = base_date.strftime("%Y-%m-%d")
        try:
            df_bcb = sgs.get({'VAL': series_id}, start=start_str, end=end_str)
            if not df_bcb.empty:
                return float(df_bcb['VAL'].dropna().iloc[-1]) / 100.0
        except Exception:
            pass
        return fallback

    def _fetch_focus_monthly(self, indicator: str, base_date: datetime.date) -> dict:
        """
        Extrai o dicionário {(ano, mes): taxa_decimal} da Mediana do Relatório Focus Mensal
        disponível na data de cálculo base_date.
        """
        monthly = {}
        try:
            df_focus = self.scraper.download_focus_expectations(indicator=indicator, limit=200, base_date=base_date)
            if not df_focus.empty:
                if 'baseCalculo' in df_focus.columns:
                    df_focus = df_focus[df_focus['baseCalculo'] == 0]
                latest_dt = df_focus['Data'].iloc[0]
                df_latest = df_focus[df_focus['Data'] == latest_dt]
                for _, row in df_latest.iterrows():
                    m_str, y_str = str(row['DataReferencia']).split('/')
                    monthly[(int(y_str), int(m_str))] = float(row['Mediana']) / 100.0
        except Exception:
            pass
        return monthly

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
            daily_cdi = self._fetch_sgs_rate(12, base_date, fallback=0.0)
            if daily_cdi > 0.0:
                v1_rate = round(((1.0 + daily_cdi) ** 252.0 - 1.0) * 10000.0) / 10000.0
            else:
                v1_rate = 0.1040 # Fallback
                
        elif v1_source == "SELIC":
            # Taxa Selic diária (SGS 11) apurada pelo BCB (% a.d.), anualizada base 252
            daily_selic = self._fetch_sgs_rate(11, base_date, fallback=0.0)
            if daily_selic > 0.0:
                v1_rate = round(((1.0 + daily_selic) ** 252.0 - 1.0) * 10000.0) / 10000.0
            else:
                v1_rate = 0.1040 # Fallback

        elif v1_source == "FED":
            try:
                v1_rate = self.scraper.download_sofr_overnight(base_date) / 100.0
            except Exception:
                v1_rate = 0.0530

        elif v1_source == "BANXICO":
            try:
                v1_rate = self.scraper.download_banxico_tiie(base_date) / 100.0
            except Exception:
                v1_rate = 0.1125
                
        # Constrói a curva via builder (o interpolador já estará na config)
        # Seção 2.1 do Manual de Curvas B3 utiliza FlatForward 252 exato nos vértices sem suavização PCHIP
        smoothing = config.get("smoothing", None)
        
        return self.builder.build_from_config(
            df=df,
            config_name=curve_name,
            base_date=base_date,
            v1_rate=v1_rate,
            smoothing=smoothing,
            use_cython_smoothing=False
        )
        
    def _build_derived_curve(self, curve_name: str, config: dict, base_date: datetime.date) -> Curve:
        # Se for TBF_PRE ou TBF_LTN, precisa da base_curve e histórico BCB (Seção 2.4 Manual B3)
        if curve_name in ["TBF_PRE", "TBF_LTN"]:
            base_curve_name = config.get("base_curve", "PRE")
            return self._build_tbf_pipeline(base_date, base_curve_name)
            
        # Se for TR_PRE ou TR_LTN, puxa a base TBF (Seção 2.5 Manual B3 / Res. CMN 5.124 BCB)
        elif curve_name in ["TR_PRE", "TR_LTN"]:
            base_curve_name = config.get("base_curve")
            tbf_curve = self.get_curve(base_curve_name, base_date)
            round_decimals = config.get("round_decimals", -1)
            return self.builder.build_tr_curve(tbf_curve, round_decimals=round_decimals)

        # Seção 3.2 Manual B3: DI x IPCA Cupom Limpo (DPL)
        elif curve_name == "DPL":
            pre_curve = self.get_curve(config.get("nominal_curve", "PRE"), base_date)
            df_dap = self.loader.get_curve_instruments(base_date, asst=config.get("real_curve", "DAP"))
            focus_ipca = self._fetch_focus_monthly("IPCA", base_date)
            prev_m = base_date - relativedelta(months=1)
            eipca_0 = focus_ipca.get((prev_m.year, prev_m.month), focus_ipca.get((base_date.year, base_date.month), None))
            return self.builder.build_dpl_curve(
                pre_curve=pre_curve,
                dap_df=df_dap,
                base_date=base_date,
                eipca_0=eipca_0,
                round_decimals=config.get("round_decimals", 4),
                round_mode=config.get("round_mode", "round")
            )
            
        # Seções 3.1 (DIC) e 3.3 (DIM) Manual B3: Curvas de Cupom Sujo de Inflação (DI x IPCA e DI x IGP-M)
        elif curve_name in ["DIC", "DIM"]:
            nominal_curve_name = config.get("nominal_curve", "PRE")
            indicator = config.get("indicator", "IPCA" if curve_name == "DIC" else "IGP-M")
            pre_curve = self.get_curve(nominal_curve_name, base_date)
            focus_monthly = self._fetch_focus_monthly(indicator, base_date)
            try:
                dpl_curve = self.get_curve("DPL", base_date)
            except Exception:
                dpl_curve = None
            return self.builder.build_dirty_inflation_coupon_curve(
                pre_curve=pre_curve,
                base_date=base_date,
                indicator=indicator,
                focus_monthly=focus_monthly,
                dpl_curve=dpl_curve,
                round_decimals=config.get("round_decimals", 4),
                round_mode=config.get("round_mode", "round")
            )
            
        else:
            raise NotImplementedError(f"Lógica de Curva Derivada não implementada para {curve_name}")

    def _build_flat_curve(self, curve_name: str, config: dict, base_date: datetime.date) -> Curve:
        # Lógica genérica de flat curves
        source = config.get("source")
        if source == "BCB":
            series = config.get("series_id")
            rate = self._fetch_sgs_rate(series, base_date, fallback=0.05)
            return self.builder.build_flat_curve(rate, config_name=curve_name)
        else:
            raise NotImplementedError("Source de flat curve não implementado.")

    def _build_tbf_pipeline(self, base_date: datetime.date, base_curve_name: str) -> Curve:
        """
        Implementa o cálculo da curva TBF x PRE (TFP) conforme Seção 2.4 do Manual de Curvas B3 (págs. 14-16):
        1. Apura TBF_{fator,t} para cada um dos 5 dias úteis anteriores à data de cálculo (exclusive):
           TBF_{fator,t} = (((1 + TBF_t/100)^(DU_30 / DU_mes) - 1) * 100) / (((1 + PRE_30/100)^(DU_30 / 252) - 1) * 100)
           onde:
           - DU_30 = dias úteis correspondentes ao vértice de 30 dias corridos da curva DI x PRE na data t;
           - DU_mes = dias úteis correspondentes a exatamente o mesmo dia no mês seguinte;
           - PRE_30 = taxa do vértice de 30 dias corridos da curva DI x PRE na data t (arredondada em 2 casas decimais).
        2. Extrai a média aritmética simples dos 5 fatores (MediaTBF_{fator,t}).
        3. Calcula os vértices da curva TFP aplicando MediaTBF_{fator,t} sobre PRE_{i+21,t}.
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
        start_date_str = (min(past_days) - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
        end_date_str = max(past_days).strftime("%Y-%m-%d")
        
        # A série 253 é % a.m. (recebe start e end date como string '%Y-%m-%d')
        try:
            df_tbf = sgs.get({'TBF': 253}, start=start_date_str, end=end_date_str)
        except Exception:
            df_tbf = pd.DataFrame()
        fatores = []
        
        for d in past_days:
            d_ts = pd.Timestamp(d)
            if df_tbf.empty or d_ts not in df_tbf.index:
                continue
                
            tbf_am = float(df_tbf.loc[d_ts, 'TBF']) / 100.0
                
            try:
                # Chama recursivamente para pegar a curva base no dia D-i
                pre_curve_past = self.get_curve(base_curve_name, d)
                
                # DU_mes: dias úteis no período de vigência da TBF (dataFim da série BCB 253 ou mesmo dia do mês seguinte;
                # conforme regra do BCB, períodos iniciados no dia 31 cujo mês seguinte tem 30 dias encerram-se no dia 1º do mês subsequente)
                if 'dataFim' in df_tbf.columns and pd.notna(df_tbf.loc[d_ts, 'dataFim']):
                    dt_fim_raw = datetime.datetime.strptime(str(df_tbf.loc[d_ts, 'dataFim']).strip(), "%d/%m/%Y").date()
                    dt_mes = self.builder.cal.following(dt_fim_raw)
                else:
                    next_m = d + relativedelta(months=1)
                    if d.day == 31 and next_m.day == 30:
                        next_m = next_m + datetime.timedelta(days=1)
                    dt_mes = self.builder.cal.following(next_m)
                if hasattr(dt_mes, 'date'):
                    dt_mes = dt_mes.date()
                du_mes = float(self.builder.cal.bizdays(d, dt_mes))
                
                # DU_30: dias úteis correspondentes ao vértice de 30 dias corridos na data d
                dt_30 = self.builder.cal.following(d + datetime.timedelta(days=30))
                if hasattr(dt_30, 'date'):
                    dt_30 = dt_30.date()
                du_30 = float(self.builder.cal.bizdays(d, dt_30))
                
                # PRE_30 da curva publicada (2 casas decimais em %)
                pre30_rate = round(pre_curve_past.get_rate(du_30) * 100.0, 2) / 100.0
                
                tbf_30 = (1.0 + tbf_am) ** (du_30 / du_mes) - 1.0
                pre_30 = (1.0 + pre30_rate) ** (du_30 / 252.0) - 1.0
                
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
