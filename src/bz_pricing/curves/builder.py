import numpy as np
import pandas as pd
import datetime
import bizdays
from bz_pricing.core.calendars import get_calendar

from bz_pricing.core.interpolators import (
    Interpolator,
    FlatForward252Interpolator,
    FlatForward252LinearInterpolator,
    FlatForward252LogInterpolator,
    Exponential252Interpolator,
    Linear360Interpolator,
    Interpolation360Interpolator
)
from bz_pricing.core.curves import SimpleCurve, FlatCurve, Curve

class B3CurveBuilder:
    def __init__(self, calendar_name: str = "B3"):
        self.calendar_name = calendar_name
        self._init_calendar()

    def _init_calendar(self):
        self.cal = get_calendar(self.calendar_name)

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
                    round_mode: str = 'round',
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
                base_curve = SimpleCurve(base_t, base_y, base_interp, t_sec_nodes=base_dc, round_decimals=-1, round_mode=round_mode)
            else:
                base_curve = SimpleCurve(base_t, base_y, base_interp, round_decimals=-1, round_mode=round_mode)
                
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
                from scipy.interpolate import PchipInterpolator
                
                unique_t = []
                unique_y = []
                for t, y in zip(t_list, y_list):
                    if t not in unique_t:
                        unique_t.append(t)
                        unique_y.append(y)
                        
                # Utiliza PchipInterpolator para garantir monotonicidade (evita a curva suavizada 'feia' do Spline)
                spline = PchipInterpolator(unique_t, unique_y)
                
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
            curve = SimpleCurve(t_nodes, y_nodes, interp, t_sec_nodes=dc_nodes, round_decimals=round_decimals, round_mode=round_mode)
        else:
            curve = SimpleCurve(t_nodes, y_nodes, interp, round_decimals=round_decimals, round_mode=round_mode)

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
            round_mode=config.get("round_mode", "round"),
            smoothing=smoothing,
            use_cython_smoothing=use_cython_smoothing
        )

    def build_flat_curve(self, rate: float, config_name: str = None) -> FlatCurve:
        """
        Constrói uma FlatCurve estática (ex: TJLP).
        """
        round_decimals = -1
        round_mode = 'round'
        if config_name:
            from .configs import CURVE_CONFIGS
            if config_name in CURVE_CONFIGS:
                round_decimals = CURVE_CONFIGS[config_name].get("round_decimals", -1)
                round_mode = CURVE_CONFIGS[config_name].get("round_mode", "round")
                
        # rate já deve estar em decimal (ex: 0.10)
        return FlatCurve(rate, round_decimals=round_decimals, round_mode=round_mode)
        
    def build_tbf_curve(self, pre_curve: Curve, media_tbf: float, round_decimals: int = -1, round_mode: str = 'round') -> SimpleCurve:
        """
        Constrói a curva TBF x PRE (TFP) aplicando o fator médio histórico da TBF
        aos vértices (t + 21) da curva DI x PRE.
        TFP_{i} = ((((1 + PRE_{i+21})^(21/252) - 1) * media_tbf + 1)^(252/21)) - 1
        """
        from bz_pricing.core.interpolators import FlatForward252Interpolator
        from bz_pricing.core.curves import SimpleCurve
        import numpy as np
        
        t_nodes = pre_curve.t_nodes
        new_y = []
        
        for t in t_nodes:
            # Pega o valor da curva PRE interpolado no dia t + 21
            pre_t21 = pre_curve.get_rate(t + 21.0)
            
            # Taxa mensal (21 DU) da PRE
            pre_21_month = ((1.0 + pre_t21) ** (21.0 / 252.0)) - 1.0
            
            # Aplica o fator de spread (media_tbf)
            tbf_month = pre_21_month * media_tbf
            
            # Reanualiza para 252
            tbf_anual = ((1.0 + tbf_month) ** (252.0 / 21.0)) - 1.0
            
            new_y.append(tbf_anual)
            
        new_y_nodes = np.array(new_y, dtype=np.float64)
        
        # B3 usa FlatForward para curvas derivadas
        interp = FlatForward252Interpolator(0, 1)
        return SimpleCurve(t_nodes, new_y_nodes, interp, round_decimals=round_decimals, round_mode=round_mode)

    def build_tr_curve(self, tbf_curve: Curve, round_decimals: int = -1, round_mode: str = 'round') -> SimpleCurve:
        """
        Constrói a curva TR aplicando a regra do Redutor (Res. 5.124 BCB) sobre os vértices da Curva TBFxPRE.
        Para evitar o formato distorcido "quadrado", a matemática é aplicada apenas nos vértices,
        e a curva resultante é interpolada por FlatForward (como a B3 faz).
        """
        from bz_pricing.core.interpolators import FlatForward252Interpolator
        from bz_pricing.core.curves import SimpleCurve
        
        t_nodes = tbf_curve.t_nodes
        y_nodes = tbf_curve.y_nodes
        
        new_y = []
        for i in range(len(t_nodes)):
            tbf_anual = y_nodes[i]
            # A curva TBFxPRE (agora "TBF Verdadeira") JÁ tem o 0.93 aplicado.
            # 1. Desanualizamos a taxa TBF para 1 mês (21 dias úteis)
            tfp_am = ((1.0 + tbf_anual) ** (21.0 / 252.0)) - 1.0

            # Res. 5.124 do BCB (Tabela de Redutores TR)
            if tfp_am >= 0.013134:
                a = 1.00500000; b = 0.48000000
            elif tfp_am >= 0.009856:
                a = 0.99869130; b = 0.96034591
            elif tfp_am >= 0.008462:
                a = 1.00500000; b = 0.32000000
            elif tfp_am >= 0.007323:
                a = 1.00010877; b = 0.89838243
            else:
                a = 1.00500000; b = 0.23000000
                
            redutor = a + b * tfp_am
            tr_am_fator = (1.0 + tfp_am) / redutor
            
            # Reanualiza a TR encontrada
            rate = (tr_am_fator ** (252.0 / 21.0)) - 1.0
            
            # Piso para não ficar negativo
            if rate <= 0.0:
                rate = 0.0001
                
            new_y.append(rate)
            
        import numpy as np
        new_y_nodes = np.array(new_y, dtype=np.float64)
        
        # B3 usa FlatForward para a TR
        interp = FlatForward252Interpolator(0, 1)
        return SimpleCurve(t_nodes, new_y_nodes, interp, round_decimals=round_decimals, round_mode=round_mode)

    def build_inflation_curve(self, nominal_curve: Curve, real_curve: Curve, config_name: str = None) -> Curve:
        """
        Constrói a Curva de Inflação Implícita derivativamente (Ex: DIC e DIM).
        Inflacao = (1 + Nominal) / (1 + Real) - 1
        """
        from bz_pricing.core.curves import InflationCurve
        round_decimals = -1
        round_mode = 'round'
        if config_name:
            from .configs import CURVE_CONFIGS
            if config_name in CURVE_CONFIGS:
                round_decimals = CURVE_CONFIGS[config_name].get("round_decimals", -1)
                round_mode = CURVE_CONFIGS[config_name].get("round_mode", "round")
                
        return InflationCurve(nominal_curve, real_curve, round_decimals=round_decimals, round_mode=round_mode)

    def build_currency_forward_curve(self, 
                                     pre_curve: Curve, 
                                     foreign_curve: Curve, 
                                     spot_rate: float, 
                                     du_nodes: np.ndarray, 
                                     dc_nodes: np.ndarray, 
                                     config_name: str) -> SimpleCurve:
        """
        Constrói a Curva de Forward de Moedas (PTX, EUR, JPY) 
        projetando o spot rate através do diferencial de juros (PRE vs Cupom Estrangeiro).
        Fwd_t = Spot * (1 + PRE_t)^(DU/252) / (1 + Cupom_t)^(DC/360)
        """
        from .configs import CURVE_CONFIGS
        
        if config_name not in CURVE_CONFIGS:
            raise ValueError(f"Configuração '{config_name}' não encontrada.")
            
        config = CURVE_CONFIGS[config_name]
        
        n = len(du_nodes)
        y_nodes = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            du = du_nodes[i]
            dc = dc_nodes[i]
            
            # Para o Vértice 1 (overnight), DU e DC podem ser 0. A curva de Forward retorna Spot.
            if du <= 0 or dc <= 0:
                y_nodes[i] = spot_rate
                continue
                
            rate_pre = pre_curve.get_rate(du, dc)
            rate_for = foreign_curve.get_rate(du, dc) # Passa du como primário e dc como secundário
            
            fator_pre = (1.0 + rate_pre) ** (du / 252.0)
            
            # Cupom Cambial Onshore (DOL/DDI/FRC) e Offshores usam juros simples (Linear) DC/360
            if config.get("foreign_compounding", "linear") == "linear":
                fator_for = 1.0 + rate_for * (dc / 360.0)
            else:
                fator_for = (1.0 + rate_for) ** (dc / 360.0)
            
            fwd = spot_rate * (fator_pre / fator_for)
            y_nodes[i] = fwd

        interp_class = config["interpolator_class"]
        interp = interp_class(config.get("extrap_left", 0), config.get("extrap_right", 0))
        
        round_decimals = config.get("round_decimals", -1)
        round_mode = config.get("round_mode", "round")
        
        use_dc = config.get("use_dc", False)
        if use_dc:
            curve = SimpleCurve(du_nodes, y_nodes, interp, t_sec_nodes=dc_nodes, round_decimals=round_decimals, round_mode=round_mode)
        else:
            curve = SimpleCurve(du_nodes, y_nodes, interp, round_decimals=round_decimals, round_mode=round_mode)
            
        return curve
