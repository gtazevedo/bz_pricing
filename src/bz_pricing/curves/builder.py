import numpy as np
import pandas as pd
import datetime
import bizdays

from bz_pricing.core.interpolators import (
    Interpolator,
    FlatForward252Interpolator,
    FlatForward252LinearInterpolator,
    Exponential252Interpolator,
    Linear360Interpolator,
    Interpolation360Interpolator
)
from bz_pricing.core.curves import SimpleCurve, FlatCurve, TBFCurve, TRCurve, Curve

class B3CurveBuilder:
    def __init__(self, calendar_name: str = "B3"):
        self.calendar_name = calendar_name
        self._init_calendar()

    def _init_calendar(self):
        """
        Carrega os feriados do pandas_market_calendars e joga na engine super rápida do bizdays.
        """
        if self.calendar_name == "B3":
            self.cal = bizdays.Calendar.load('B3')
        else:
            # Fallback para calendário atual (dias corridos sem feriados)
            self.cal = bizdays.Calendar(weekdays=[], name='Actual')

    def build_curve(self,
                    dataset: pd.DataFrame,
                    base_date: datetime.date,
                    interpolator_class: type,
                    extrap_left: int = 0,
                    extrap_right: int = 0,
                    v1_rate: float = None,
                    v1_du: int = 1,
                    v1_dc: int = 1,
                    rate_column: str = 'AdjstdQtTax',
                    exp_date: str = 'XprtnDt',
                    use_dc: bool = False,
                    round_decimals: int = -1,
                    smoothing: float = None,
                    use_cython_smoothing: bool = False) -> SimpleCurve:
        """
        Pipeline principal para construir a Curva Cython a partir do DataFrame pandas.
        """
        df = dataset.copy()

        # Filtro Básico (Garantir que a coluna alvo exista e sem NAs)
        if rate_column not in df.columns:
            raise ValueError(f"Coluna {rate_column} não encontrada no dataset.")
        df = df.dropna(subset=[rate_column, exp_date]).copy()

        # Cálculo dos dias (DU e DC)       
        # Converte o vencimento para date
        df['XprtnDt_date'] = pd.to_datetime(df[exp_date]).dt.date
        df['DU'] = [self.cal.bizdays(base_date, x) for x in df['XprtnDt_date']]
        df['DC'] = [(x - base_date).days for x in df['XprtnDt_date']]
        
        # Filtra prazos negativos ou zero
        df = df[df['DU'] > 0]
        
        # Preparação dos arrays
        t_list = df['DU'].tolist()
        y_list = (df[rate_column] / 100.0).tolist() # Converte taxa de % para decimal
        dc_list = df['DC'].tolist()

        # Injeção do Vértice 1 (ex: CDI 1 dia)
        if v1_rate is not None:
            # Garante que o Vértice 1 venha antes inserindo no índice 0
            if v1_du not in t_list:
                t_list.insert(0, v1_du)
                y_list.insert(0, v1_rate)
                dc_list.insert(0, v1_dc)
            else:
                # Se já existe o vértice no dataset, sobrescrevemos com a taxa v1_rate
                idx = t_list.index(v1_du)
                y_list[idx] = v1_rate

        if len(t_list) == 0:
            raise ValueError("Não há vértices válidos para construir a curva.")

        # Aplica Suavização (opcional)
        if smoothing is not None and smoothing > 0:
            # 1. Constrói curva base (Original) para ancorar a ponta longa (Smart Smoothing)
            base_t = np.array(t_list, dtype=np.float64)
            base_y = np.array(y_list, dtype=np.float64)
            base_dc = np.array(dc_list, dtype=np.float64)
            base_interp = interpolator_class(extrap_left, extrap_right)
            
            if use_dc:
                base_curve = SimpleCurve(base_t, base_y, base_interp, t_sec_nodes=base_dc, round_decimals=-1)
            else:
                base_curve = SimpleCurve(base_t, base_y, base_interp, round_decimals=-1)
                
            anchor_t = 10080.0 # Vértice âncora de 40 anos para travar a taxa forward
            anchor_y = base_curve.get_rate(anchor_t)
            anchor_dc = 10080.0 if use_dc else 1.0
            
            if use_cython_smoothing:
                from bz_pricing.core.curves import apply_cython_smoothing
                t_arr = np.array(t_list, dtype=np.float64)
                y_arr = np.array(y_list, dtype=np.float64)
                
                new_t, new_y, new_dc = apply_cython_smoothing(t_arr, y_arr, smoothing)
                
                t_list = new_t.tolist()
                y_list = new_y.tolist()
                dc_list = new_dc.tolist()
            else:
                from scipy.interpolate import UnivariateSpline
                
                unique_t = []
                unique_y = []
                for t, y in zip(t_list, y_list):
                    if t not in unique_t:
                        unique_t.append(t)
                        unique_y.append(y)
                        
                # s is smoothing factor. Default heuristic if simply given as a boolean-like float, let scipy handle it or use the passed float.
                spline = UnivariateSpline(unique_t, unique_y, s=smoothing)
                
                # Generate dense grid (daily)
                max_du = int(max(unique_t))
                new_t_list = list(range(1, max_du + 1))
                new_y_list = spline(new_t_list).tolist()
                
                t_list = new_t_list
                y_list = new_y_list
                # Default DC to 1 when smoothing since exact day counting isn't as relevant for smoothed curves 
                # if we are not evaluating complex instruments against the smoothed curve precisely.
                dc_list = [1] * len(t_list)
                
            # Adiciona a âncora para forçar o interpolador a fazer o blend da suavizada com a base
            t_list.append(anchor_t)
            y_list.append(anchor_y)
            dc_list.append(anchor_dc)

        # Converte para arrays contíguos numpy (float64 C-contiguous)
        t_nodes = np.array(t_list, dtype=np.float64)
        y_nodes = np.array(y_list, dtype=np.float64)
        dc_nodes = np.array(dc_list, dtype=np.float64)

        # Instanciação do Interpolador Cython
        interp = interpolator_class(extrap_left, extrap_right)

        # Construção da Curva Cython
        if use_dc:
            # Modelos como FlatForward252LinearInterpolator exigem Dias Corridos (DC) como sec_nodes
            curve = SimpleCurve(t_nodes, y_nodes, interp, t_sec_nodes=dc_nodes, round_decimals=round_decimals)
        else:
            curve = SimpleCurve(t_nodes, y_nodes, interp, round_decimals=round_decimals)

        return curve

    # -------------------------------------------------------------
    # Factory Methods usando Configurações (Mapeamento)
    # -------------------------------------------------------------

    def build_from_config(self, df: pd.DataFrame, config_name: str, base_date: datetime.date, v1_rate: float = None, v1_du: int = 1, v1_dc: int = 1, smoothing: float = None, use_cython_smoothing: bool = False) -> SimpleCurve:
        """
        Constrói uma curva baseada nas configurações predefinidas no arquivo configs.py.
        """
        from .configs import CURVE_CONFIGS
        
        if config_name not in CURVE_CONFIGS:
            raise ValueError(f"Configuração de curva '{config_name}' não encontrada.")
            
        config = CURVE_CONFIGS[config_name]
        
        dataset = df.copy()
        
        # Filtro de Liquidez (opcional)
        if config.get("liquidity_filter", False):
            has_trades = dataset.get('RglrTxsQty', pd.Series(0, index=dataset.index)) > 0
            has_oi = dataset.get('OpnIntrst', pd.Series(0, index=dataset.index)) > 0
            dataset = dataset[has_trades | has_oi].copy()
            
        return self.build_curve(
            dataset=dataset,
            base_date=base_date,
            interpolator_class=config["interpolator_class"],
            extrap_left=config.get("extrap_left", 0),
            extrap_right=config.get("extrap_right", 0),
            v1_rate=v1_rate,
            v1_du=v1_du,
            v1_dc=v1_dc,
            rate_column=config.get("rate_column", "AdjstdQtTax"),
            exp_date=config.get("exp_date", "XprtnDt"),
            use_dc=config.get("use_dc", False),
            round_decimals=config.get("round_decimals", -1),
            smoothing=smoothing,
            use_cython_smoothing=use_cython_smoothing
        )

    def build_flat_curve(self, rate: float, config_name: str = None) -> FlatCurve:
        """
        Constrói uma FlatCurve estática (ex: TJLP).
        """
        round_decimals = -1
        if config_name:
            from .configs import CURVE_CONFIGS
            if config_name in CURVE_CONFIGS:
                round_decimals = CURVE_CONFIGS[config_name].get("round_decimals", -1)
                
        # rate já deve estar em decimal (ex: 0.10)
        return FlatCurve(rate, round_decimals=round_decimals)
        
    def build_tbf_curve(self, pre_curve: Curve, media_tbf: float, round_decimals: int = -1) -> TBFCurve:
        """
        Envelopa a Curva PRE em uma matemática Derivativa TBFxPRE no Cython.
        """
        return TBFCurve(pre_curve, media_tbf, round_decimals=round_decimals)

    def build_tr_curve(self, tbf_curve: Curve, round_decimals: int = -1) -> TRCurve:
        """
        Envelopa a Curva TBFxPRE em uma matemática TRxPRE no Cython (Res. 5.124 BCB).
        """
        return TRCurve(tbf_curve, round_decimals=round_decimals)
