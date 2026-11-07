import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import bizdays
from bz_pricing.core.calendars import get_calendar

from bz_pricing.core.interpolators import (
    Interpolator,
    FlatForward252Interpolator,
    FlatForward252LinearInterpolator,
    FlatForward252LogInterpolator,
    Exponential252Interpolator,
    Linear360Interpolator,
    Interpolation360Interpolator,
    PriceInterpolator
)
from bz_pricing.core.curves import SimpleCurve, FlatCurve, Curve, InflationCurve, ConvenienceYieldCurve, SpreadCurve


def _resolve_dc_from_knots(t: float, t_sec: float, t_nodes: np.ndarray, t_sec_nodes: np.ndarray) -> float:
    if t_sec > 0.0:
        return float(t_sec)
    if t_nodes is not None and t_sec_nodes is not None and len(t_nodes) >= 2 and len(t_sec_nodes) == len(t_nodes):
        if t <= t_nodes[0]:
            return float(t_sec_nodes[0])
        if t >= t_nodes[-1]:
            du_span = t_nodes[-1] - t_nodes[-2]
            dc_span = t_sec_nodes[-1] - t_sec_nodes[-2]
            slope = (dc_span / du_span) if du_span > 0 else (365.0 / 252.0)
            return float(t_sec_nodes[-1] + slope * (t - t_nodes[-1]))
        return float(np.interp(t, t_nodes, t_sec_nodes))
    return max(1.0, float(t) * 365.0 / 252.0)


class CrossCouponCurve(Curve):
    """
    Estrutura a termo de cupom cambial cruzado (EUC e YCL):
    TC_n = ((1 + DOC_n * DC_n / 36000) * (1 + Spread_n * DC_n / 36000) - 1) * 36000 / DC_n
    Avalia dinamicamente a paridade a partir das curvas DOC e Spread interpoladas em seus respectivos vértices base.
    """
    def __init__(self, doc_curve: Curve, spread_curve: Curve, round_decimals: int = 6, round_mode: str = 'round'):
        self.doc_curve = doc_curve
        self.spread_curve = spread_curve
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')

        pairs = {}
        for c in (doc_curve, spread_curve):
            if hasattr(c, 't_nodes') and c.t_nodes is not None:
                c_du = np.asarray(c.t_nodes, dtype=np.float64)
                c_dc = np.asarray(c.t_sec_nodes, dtype=np.float64) if (hasattr(c, 't_sec_nodes') and c.t_sec_nodes is not None) else np.round(c_du * 365.0 / 252.0)
                for du, dc in zip(c_du, c_dc):
                    pairs[float(du)] = float(dc)
        kn_du = sorted(pairs.keys()) if pairs else [1.0]
        self._t_nodes = np.array(kn_du, dtype=np.float64)
        self._t_sec_nodes = np.array([pairs.get(k, max(1.0, round(k * 365.0 / 252.0))) for k in kn_du], dtype=np.float64)
        self._y_nodes = None

    def get_rate(self, t: float, t_sec: float = 0.0) -> float:
        dc = _resolve_dc_from_knots(t, t_sec, self._t_nodes, self._t_sec_nodes)
        if dc <= 0.0:
            dc = 1.0
        doc_pct = self.doc_curve.get_rate(float(t), float(dc)) * 100.0
        spr_pct = self.spread_curve.get_rate(float(t), float(dc)) * 100.0
        tc_pct = (((1.0 + (doc_pct * dc) / 36000.0) * (1.0 + (spr_pct * dc) / 36000.0)) - 1.0) * (36000.0 / dc)
        rate = tc_pct / 100.0
        if self._round_decimals >= 0:
            factor = 10.0 ** self._round_decimals
            if self._is_trunc:
                rate = float(np.trunc(rate * factor) / factor)
            else:
                rate = round(rate, self._round_decimals)
        return rate

    @property
    def t_nodes(self):
        return self._t_nodes

    @property
    def t_sec_nodes(self):
        return self._t_sec_nodes

    @property
    def y_nodes(self):
        if self._y_nodes is None:
            self._y_nodes = np.array([self.get_rate(float(du), float(dc)) for du, dc in zip(self._t_nodes, self._t_sec_nodes)], dtype=np.float64)
        return self._y_nodes


class CurrencyForwardCurve(Curve):
    """
    Estrutura a termo de câmbio a termo (PTX, EUR e JPY):
    - Vértice 1 (DU = 1) para PTX: Ptax_{t0}
    - Demais vértices: TC_n = ((1 + PRE_n)^(DU_n/252) / (1 + Cupom_n * DC_n / 360)) * Spot_ref
    Avalia dinamicamente a paridade entre a curva prefixada (PRE) e o cupom cambial interpolados em seus respectivos vértices base.
    """
    def __init__(self, pre_curve: Curve, foreign_curve: Curve, spot_t0: float, spot_ref: float,
                 is_ptx: bool = False, linear_comp: bool = True,
                 du_nodes: np.ndarray = None, dc_nodes: np.ndarray = None,
                 round_decimals: int = 7, round_mode: str = 'trunc'):
        self.pre_curve = pre_curve
        self.foreign_curve = foreign_curve
        self.spot_t0 = float(spot_t0)
        self.spot_ref = float(spot_ref)
        self.is_ptx = bool(is_ptx)
        self.linear_comp = bool(linear_comp)
        self._round_decimals = round_decimals
        self._is_trunc = (round_mode == 'trunc')

        pairs = {}
        if du_nodes is not None and dc_nodes is not None and len(du_nodes) == len(dc_nodes):
            for du, dc in zip(du_nodes, dc_nodes):
                pairs[float(du)] = float(dc)
        for c in (foreign_curve, pre_curve):
            if hasattr(c, 't_nodes') and c.t_nodes is not None:
                c_du = np.asarray(c.t_nodes, dtype=np.float64)
                c_dc = np.asarray(c.t_sec_nodes, dtype=np.float64) if (hasattr(c, 't_sec_nodes') and c.t_sec_nodes is not None) else None
                if c_dc is not None and len(c_dc) == len(c_du):
                    for du, dc in zip(c_du, c_dc):
                        pairs[float(du)] = float(dc)
        kn_du = sorted(pairs.keys()) if pairs else [1.0]
        self._t_nodes = np.array(kn_du, dtype=np.float64)
        self._t_sec_nodes = np.array([pairs.get(k, max(1.0, round(k * 365.0 / 252.0))) for k in kn_du], dtype=np.float64)
        self._y_nodes = None

    def get_rate(self, t: float, t_sec: float = 0.0) -> float:
        if t <= 0.0 or (self.is_ptx and abs(t - 1.0) < 1e-9):
            rate = self.spot_t0
        else:
            dc = _resolve_dc_from_knots(t, t_sec, self._t_nodes, self._t_sec_nodes)
            if dc <= 0.0:
                dc = 1.0
            rate_pre = self.pre_curve.get_rate(float(t), float(dc))
            rate_for = self.foreign_curve.get_rate(float(t), float(dc))
            fator_pre = (1.0 + rate_pre) ** (float(t) / 252.0)
            if self.linear_comp:
                fator_for = 1.0 + rate_for * (float(dc) / 360.0)
            else:
                fator_for = (1.0 + rate_for) ** (float(dc) / 360.0)
            rate = self.spot_ref * (fator_pre / fator_for)

        if self._round_decimals >= 0:
            factor = 10.0 ** self._round_decimals
            if self._is_trunc:
                rate = float(np.trunc(rate * factor) / factor)
            else:
                rate = round(rate, self._round_decimals)
        return rate

    @property
    def t_nodes(self):
        return self._t_nodes

    @property
    def t_sec_nodes(self):
        return self._t_sec_nodes

    @property
    def y_nodes(self):
        if self._y_nodes is None:
            self._y_nodes = np.array([self.get_rate(float(du), float(dc)) for du, dc in zip(self._t_nodes, self._t_sec_nodes)], dtype=np.float64)
        return self._y_nodes


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
                    index_by_dc: bool = False,
                    round_decimals: int = -1,
                    round_mode: str = 'round',
                    smoothing: float = None,
                    use_cython_smoothing: bool = False) -> SimpleCurve:
        """
        Pipeline principal para construir a Curva Cython a partir do DataFrame pandas.
        """
        # Suporte a formatos externos variados (Series, dict, list, ou DataFrame flexível)
        if isinstance(dataset, pd.Series):
            df = dataset.reset_index()
            df.columns = ['DU', 'rate']
        elif isinstance(dataset, dict):
            df = pd.DataFrame(list(dataset.items()), columns=['DU', 'rate'])
        elif isinstance(dataset, (list, tuple)):
            if len(dataset) > 0 and isinstance(dataset[0], (list, tuple)):
                if len(dataset[0]) >= 3:
                    df = pd.DataFrame(dataset, columns=['DU', 'DC', 'rate'])
                else:
                    df = pd.DataFrame(dataset, columns=['DU', 'rate'])
            else:
                df = pd.DataFrame(dataset)
        else:
            df = dataset.copy()

        # Identifica se possui colunas de vencimento ou se já possui prazos em dias
        has_exp_date = exp_date in df.columns or any(c.lower() in ['xprtndt', 'maturity', 'vencimento'] for c in df.columns)
        has_day_nodes = any(c in df.columns for c in ['DU', 'day252', 'DC', 'day360', 'prazo', 'dias'])

        if has_exp_date and not (has_day_nodes and rate_column not in df.columns):
            actual_exp = exp_date if exp_date in df.columns else [c for c in df.columns if c.lower() in ['xprtndt', 'maturity', 'vencimento']][0]
            actual_rate = rate_column if rate_column in df.columns else [c for c in df.columns if c.lower() in ['adjstdqttax', 'rate', 'taxa', 'yield']][0]

            df = df.dropna(subset=[actual_rate, actual_exp]).copy()
            df['XprtnDt_date'] = pd.to_datetime(df[actual_exp]).dt.date
            df['DU'] = [self.cal.bizdays(base_date, x) for x in df['XprtnDt_date']]
            df['DC'] = [(x - base_date).days for x in df['XprtnDt_date']]
            
            # Filtra prazos negativos ou zero e ordena por DU/DC crescente
            sort_col = 'DC' if index_by_dc else 'DU'
            df = df[df[sort_col] > 0].sort_values(sort_col).copy()
            
            t_list = df['DC' if index_by_dc else 'DU'].astype(float).tolist()
            raw_y = df[actual_rate].astype(float).values
            if np.nanmax(np.abs(raw_y)) > 1.0:
                y_list = (raw_y / 100.0).tolist()
            else:
                y_list = raw_y.tolist()
            dc_list = df['DC'].astype(float).tolist()
        else:
            # Já possui prazos em dias (DU, day252, DC, day360)
            col_du = 'DU' if 'DU' in df.columns else ('day252' if 'day252' in df.columns else None)
            col_dc = 'DC' if 'DC' in df.columns else ('day360' if 'day360' in df.columns else None)
            
            if col_du is None and col_dc is None:
                col_du = df.columns[0]
                
            target_t_col = (col_dc if (index_by_dc and col_dc is not None) else (col_du or col_dc))
            rate_candidates = [c for c in df.columns if c.lower() in ['rate', 'taxa', 'adjstdqttax', 'yield', 'preco']]
            col_rate = rate_column if rate_column in df.columns else (rate_candidates[0] if rate_candidates else df.columns[-1])
            
            df = df.dropna(subset=[target_t_col, col_rate]).copy()
            df = df.sort_values(target_t_col).copy()
            
            t_list = df[target_t_col].astype(float).tolist()
            raw_y = df[col_rate].astype(float).values
            if np.nanmax(np.abs(raw_y)) > 1.0:
                y_list = (raw_y / 100.0).tolist()
            else:
                y_list = raw_y.tolist()
                
            if col_dc is not None and col_dc in df.columns:
                dc_list = df[col_dc].astype(float).tolist()
            else:
                dc_list = [max(1.0, round(float(t) * 365.0 / 252.0)) for t in t_list]

        # Injeção do Vértice 1 (ex: CDI 1 dia)
        if v1_rate is not None:
            if v1_dc == 1 and v1_du == 1:
                try:
                    next_bday = self.cal.offset(base_date, 1)
                    if hasattr(next_bday, 'date'):
                        next_bday = next_bday.date()
                    v1_dc = max(1, (next_bday - base_date).days)
                except Exception:
                    v1_dc = 1
            v1_t = float(v1_dc if index_by_dc else v1_du)
            # Garante que o Vértice 1 venha antes inserindo no índice 0
            if v1_t not in t_list:
                t_list.insert(0, v1_t)
                y_list.insert(0, v1_rate)
                dc_list.insert(0, float(v1_dc))
            else:
                # Se já existe o vértice no dataset, sobrescrevemos com a taxa v1_rate
                idx = t_list.index(v1_t)
                y_list[idx] = v1_rate
                dc_list[idx] = float(v1_dc)

        if len(t_list) == 0:
            raise ValueError("Não há vértices válidos para construir a curva.")

        # Aplica Suavização (opcional)
        if smoothing is not None and smoothing > 0:
            # 1. Constrói curva base (Original) para ancorar a ponta longa (Smart Smoothing)
            base_t = np.array(t_list, dtype=np.float64)
            base_y = np.array(y_list, dtype=np.float64)
            base_dc = np.array(dc_list, dtype=np.float64)
            base_interp = interpolator_class(extrap_left, extrap_right)
            
            if use_dc and not index_by_dc:
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
        if use_dc and not index_by_dc:
            # Modelos como FlatForward252LinearInterpolator exigem Dias Úteis (DU) em t_nodes e Dias Corridos (DC) em t_sec_nodes
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
            index_by_dc=config.get("index_by_dc", False),
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
        
        t_nodes_list = list(pre_curve.t_nodes)
        if max(t_nodes_list) < 8463.0:
            # Estende com a grade de vértices longos padronizados até 8463 DU
            long_vertices = [3780, 4032, 4284, 4536, 4788, 5040, 5292, 5544, 5796, 6048, 6300, 7146, 7404, 8463]
            for lv in long_vertices:
                if float(lv) > max(t_nodes_list):
                    t_nodes_list.append(float(lv))
        t_nodes = np.array(sorted(t_nodes_list), dtype=np.float64)
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
            
            if round_decimals >= 0:
                tbf_anual = round(tbf_anual, round_decimals)
                
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
                
            if round_decimals >= 0:
                rate = round(rate, round_decimals)
                
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

    def build_dpl_curve(self,
                        pre_curve: Curve,
                        dap_df: pd.DataFrame,
                        base_date: datetime.date,
                        eipca_0: float = None,
                        v1_rate: float = None,
                        rate_column: str = 'AdjstdQtTax',
                        exp_date: str = 'XprtnDt',
                        round_decimals: int = 4,
                        round_mode: str = 'round') -> Curve:
        """
        Constrói a curva DI x IPCA Cupom Limpo (DPL):
        1. Calcula o vértice de 1 dia útil (DPL_{0,t}) a partir da taxa DI e da prévia de inflação EIPCA_0 / DU_{15a,15p}.
        2. Nos vértices de vencimento de DAP, extrai a inflação implícita:
           implicita_{n,t} = (1 + PRE_{n,t}) / (1 + DPL_{n,t}) - 1.
        3. Interpola a inflação implícita por taxa a termo exponencial em base 252.
        4. Recompõe o cupom limpo DPL para qualquer prazo:
           DPL_{n,t} = (1 + PRE_{n,t}) / (1 + implicita_{n,t}) - 1.
        """
        if isinstance(dap_df, pd.Series):
            df = dap_df.reset_index()
            df.columns = ['DU', 'rate'] if len(df.columns) == 2 else df.columns
        elif isinstance(dap_df, dict):
            df = pd.DataFrame(list(dap_df.items()), columns=['DU', 'rate'])
        elif isinstance(dap_df, (list, tuple)):
            df = pd.DataFrame(dap_df, columns=['DU', 'rate'] if len(dap_df) > 0 and isinstance(dap_df[0], (list, tuple)) and len(dap_df[0]) == 2 else None)
        else:
            df = dap_df.copy()

        actual_rate = rate_column if rate_column in df.columns else ([c for c in df.columns if c.lower() in ['adjstdqttax', 'rate', 'taxa', 'yield']][0] if any(c.lower() in ['adjstdqttax', 'rate', 'taxa', 'yield'] for c in df.columns) else df.columns[-1])
        has_exp = exp_date in df.columns or any(c.lower() in ['xprtndt', 'maturity', 'vencimento'] for c in df.columns)
        if has_exp:
            actual_exp = exp_date if exp_date in df.columns else [c for c in df.columns if c.lower() in ['xprtndt', 'maturity', 'vencimento']][0]
            df = df.dropna(subset=[actual_rate, actual_exp]).copy()
            df['XprtnDt_date'] = pd.to_datetime(df[actual_exp]).dt.date
            df['DU'] = [self.cal.bizdays(base_date, x) for x in df['XprtnDt_date']]
        else:
            du_col = 'DU' if 'DU' in df.columns else ('day252' if 'day252' in df.columns else df.columns[0])
            df = df.dropna(subset=[actual_rate, du_col]).copy()
            df['DU'] = df[du_col].astype(float)

        df = df[df['DU'] > 0].sort_values('DU').copy()

        t_list = df['DU'].astype(float).tolist()
        raw_dpl = df[actual_rate].astype(float).values
        dpl_list = (raw_dpl / 100.0 if np.nanmax(np.abs(raw_dpl)) > 1.0 else raw_dpl).tolist()

        # Cálculo do vértice de 1 dia útil
        dpl_v1 = None
        if v1_rate is not None:
            dpl_v1 = float(v1_rate) / 100.0 if abs(float(v1_rate)) > 1.0 else float(v1_rate)
        elif eipca_0 is not None:
            # Determina 15a (15 anterior corrigido para dia útil) e 15p (15 posterior corrigido para dia útil)
            dt_15_curr = self.cal.following(datetime.date(base_date.year, base_date.month, 15))
            if hasattr(dt_15_curr, 'date'):
                dt_15_curr = dt_15_curr.date()
            if base_date < dt_15_curr:
                prev_m = base_date - relativedelta(months=1)
                dt_15a = self.cal.following(datetime.date(prev_m.year, prev_m.month, 15))
                dt_15p = dt_15_curr
            else:
                next_m = base_date + relativedelta(months=1)
                dt_15a = dt_15_curr
                dt_15p = self.cal.following(datetime.date(next_m.year, next_m.month, 15))
            if hasattr(dt_15a, 'date'):
                dt_15a = dt_15a.date()
            if hasattr(dt_15p, 'date'):
                dt_15p = dt_15p.date()
            du_15a_15p = max(1, self.cal.bizdays(dt_15a, dt_15p))
            cdi_rate = pre_curve.get_rate(1.0)
            f_cdi = (1.0 + cdi_rate) ** (1.0 / 252.0)
            f_ipca = (1.0 + eipca_0) ** (1.0 / float(du_15a_15p))
            dpl_v1 = ((f_cdi / f_ipca) ** 252.0) - 1.0

        if dpl_v1 is not None:
            if 1.0 not in t_list:
                t_list.insert(0, 1.0)
                dpl_list.insert(0, dpl_v1)
            else:
                idx = t_list.index(1.0)
                dpl_list[idx] = dpl_v1

        # Extração da inflação implícita nos vértices base: implicita_{n,t} = (1 + PRE_{n,t}) / (1 + DPL_{n,t}) - 1
        imp_list = []
        for du_val, dpl_val in zip(t_list, dpl_list):
            pre_val = pre_curve.get_rate(float(du_val))
            imp_val = ((1.0 + pre_val) / (1.0 + dpl_val)) - 1.0
            imp_list.append(imp_val)

        # Interpolação Flat Forward 252 sobre a inflação implícita
        imp_curve = SimpleCurve(
            np.array(t_list, dtype=np.float64),
            np.array(imp_list, dtype=np.float64),
            FlatForward252Interpolator(0, 1),
            round_decimals=-1
        )

        # Recomposição do cupom limpo DPL: DPL_{n,t} = (1 + PRE_{n,t}) / (1 + implicita_{n,t}) - 1
        return InflationCurve(pre_curve, imp_curve, round_decimals=round_decimals, round_mode=round_mode)

    def build_dirty_inflation_coupon_curve(self,
                                           pre_curve: Curve,
                                           base_date: datetime.date,
                                           indicator: str = "IPCA",
                                           focus_monthly: dict = None,
                                           focus_annual: dict = None,
                                           dpl_curve: Curve = None,
                                           informants_df: pd.DataFrame = None,
                                           round_decimals: int = 4,
                                           round_mode: str = 'round') -> SimpleCurve:
        """
        Constrói as curvas de cupom sujo de inflação (DIC para IPCA e DIM para IGP-M).
        Aplica a contagem discreta de divulgações do índice (k_i) e o fator acumulado
        1 + E(Indice)_{k_i,t} nos vértices referenciais da janela de divulgação,
        interpolando os demais prazos via FlatForward252Interpolator.
        """
        # 1. Construção a partir da amostra de informantes (CpInformantes) e curva PRE
        if informants_df is not None:
            if isinstance(informants_df, pd.Series):
                inf_df = informants_df.reset_index()
                inf_df.columns = ['DU', 'rate'] if len(inf_df.columns) == 2 else inf_df.columns
            elif isinstance(informants_df, dict):
                inf_df = pd.DataFrame(list(informants_df.items()), columns=['DU', 'rate'])
            elif isinstance(informants_df, (list, tuple)):
                inf_df = pd.DataFrame(informants_df, columns=['DU', 'rate'] if len(informants_df) > 0 and isinstance(informants_df[0], (list, tuple)) and len(informants_df[0]) == 2 else None)
            else:
                inf_df = informants_df.copy()

            if not inf_df.empty:
                if any(c.lower() in ['xprtndt', 'maturity', 'vencimento', 'data'] for c in inf_df.columns) and 'DU' not in inf_df.columns and 'day252' not in inf_df.columns:
                    exp_col = [c for c in inf_df.columns if c.lower() in ['xprtndt', 'maturity', 'vencimento', 'data']][0]
                    inf_df['DU'] = [self.cal.bizdays(base_date, pd.to_datetime(x).date()) for x in inf_df[exp_col]]

                inf_df['DU_val'] = inf_df['DU'].astype(int) if 'DU' in inf_df.columns else inf_df['day252'].astype(int)
                inf_df = inf_df.sort_values('DU_val').reset_index(drop=True)

                e_col_scale = 1.0
                if 'e_inf' in inf_df.columns and not inf_df['e_inf'].dropna().empty:
                    if float(np.nanmax(np.abs(inf_df['e_inf'].dropna().values.astype(float)))) < 0.15:
                        e_col_scale = 100.0

                kn_dict = {1: round(pre_curve.get_rate(1.0), round_decimals) if round_decimals >= 0 else pre_curve.get_rate(1.0)}
                prev_du = 1
                for _, row in inf_df.iterrows():
                    du_e = int(row['DU_val'])
                    if du_e <= 1:
                        prev_du = 1
                        continue
                    if 'du_start' in row and not pd.isna(row['du_start']):
                        du_s = int(row['du_start'])
                    else:
                        du_s = (prev_du + 1) if du_e <= 265 else du_e

                    r_dec = None
                    if 'rate' in row and not pd.isna(row['rate']):
                        r_raw = float(row['rate'])
                        r_dec = r_raw / 100.0 if abs(r_raw) > 1.0 else r_raw

                    e_inf_raw = row.get('e_inf', np.nan)
                    e_inf_pct = (float(e_inf_raw) * e_col_scale) if not pd.isna(e_inf_raw) else np.nan
                    if pd.isna(e_inf_pct) and r_dec is not None and du_e >= du_s and du_e <= 265:
                        # Expectativa de inflação acumulada arredondada em 2 casas decimais em percentual
                        pre_pct = round(pre_curve.get_rate(float(du_e)) * 100.0, 2)
                        cp_pct = r_dec * 100.0
                        e_inf_pct = round((((1.0 + pre_pct / 100.0) ** (du_e / 252.0)) / ((1.0 + cp_pct / 100.0) ** (du_e / 252.0)) - 1.0) * 100.0, 2)

                    if not pd.isna(e_inf_pct) and du_e >= du_s and du_e <= 265:
                        # Cálculo nos vértices referenciais do patamar k_i [du_s .. du_e]
                        for du_p in range(max(1, du_s), du_e + 1):
                            pre_pct = round(pre_curve.get_rate(float(du_p)) * 100.0, 2)
                            f_pre = (1.0 + pre_pct / 100.0) ** (float(du_p) / 252.0)
                            cp_pct = round(((f_pre / (1.0 + e_inf_pct / 100.0)) ** (252.0 / float(du_p)) - 1.0) * 100.0, 2)
                            kn_dict[du_p] = cp_pct / 100.0
                    elif r_dec is not None:
                        kn_dict[du_e] = round(r_dec, round_decimals) if round_decimals >= 0 else r_dec
                    elif not pd.isna(e_inf_pct):
                        pre_pct = round(pre_curve.get_rate(float(du_e)) * 100.0, 2)
                        f_pre = (1.0 + pre_pct / 100.0) ** (float(du_e) / 252.0)
                        cp_pct = round(((f_pre / (1.0 + e_inf_pct / 100.0)) ** (252.0 / float(du_e)) - 1.0) * 100.0, 2)
                        kn_dict[du_e] = cp_pct / 100.0
                    prev_du = du_e

                kn_dus = sorted(kn_dict.keys())
                interp = FlatForward252Interpolator(0, 1)
                return SimpleCurve(
                    np.array(kn_dus, dtype=np.float64),
                    np.array([kn_dict[k] for k in kn_dus], dtype=np.float64),
                    interp,
                    round_decimals=round_decimals,
                    round_mode=round_mode
                )

        # 2. Alternativa via projeções mensais de inflação + curva DPL
        focus_monthly = focus_monthly or {}
        focus_annual = focus_annual or {}
        max_du = int(max(pre_curve.t_nodes)) if len(pre_curve.t_nodes) > 0 else 10080

        t_knots = []
        y_knots = []

        if indicator.upper() == "IPCA":
            # Vértice de 1 dia útil (k_i = 0): E(IPCA)_{0,t} = 0 => Cp_{1,t} = PRE_{1,t}
            t_knots.append(1.0)
            y_knots.append(pre_curve.get_rate(1.0))

            F_acc = 1.0
            curr_month = datetime.date(base_date.year, base_date.month, 1)

            dt_15_curr = self.cal.following(datetime.date(curr_month.year, curr_month.month, 15))
            if hasattr(dt_15_curr, 'date'):
                dt_15_curr = dt_15_curr.date()
            du_15_curr = self.cal.bizdays(base_date, dt_15_curr)

            prev_month_curr = curr_month - relativedelta(months=1)
            ref_key_curr = (prev_month_curr.year, prev_month_curr.month)
            pi_m_curr = focus_monthly.get(ref_key_curr, focus_monthly.get((curr_month.year, curr_month.month), 0.0))

            if du_15_curr > 1:
                for du_p in range(2, du_15_curr):
                    pro_rata = float(du_p - 1) / float(du_15_curr - 1)
                    E_pr = pi_m_curr * pro_rata
                    r_pre = pre_curve.get_rate(float(du_p))
                    f_pre = (1.0 + r_pre) ** (float(du_p) / 252.0)
                    cp = ((f_pre / (1.0 + E_pr)) ** (252.0 / float(du_p))) - 1.0
                    if round_decimals >= 0:
                        cp = round(cp, round_decimals)
                    t_knots.append(float(du_p))
                    y_knots.append(cp)

            for m_idx in range(1, 500):
                dt_15 = self.cal.following(datetime.date(curr_month.year, curr_month.month, 15))
                if hasattr(dt_15, 'date'):
                    dt_15 = dt_15.date()
                next_month = curr_month + relativedelta(months=1)
                dt_01_next = self.cal.following(next_month)
                if hasattr(dt_01_next, 'date'):
                    dt_01_next = dt_01_next.date()

                du_15 = self.cal.bizdays(base_date, dt_15)
                du_01_next = self.cal.bizdays(base_date, dt_01_next)
                if du_15 <= 1:
                    curr_month = next_month
                    continue
                if du_15 > max_du:
                    break

                prev_month = curr_month - relativedelta(months=1)
                ref_key = (prev_month.year, prev_month.month)
                if m_idx <= 18 and ref_key in focus_monthly:
                    pi_m = focus_monthly[ref_key]
                    F_acc *= (1.0 + pi_m)
                elif dpl_curve is not None:
                    r_pre_15 = pre_curve.get_rate(float(du_15))
                    r_dpl_15 = dpl_curve.get_rate(float(du_15))
                    F_acc = ((1.0 + r_pre_15) / (1.0 + r_dpl_15)) ** (float(du_15) / 252.0)
                else:
                    ann_rate = focus_annual.get(prev_month.year, 0.038)
                    pi_m = (1.0 + ann_rate) ** (1.0 / 12.0) - 1.0
                    F_acc *= (1.0 + pi_m)

                # Cálculo nos vértices referenciais do patamar k_i = m_idx [du_15 .. du_01_next]
                for du_p in range(du_15, min(du_01_next, max_du) + 1):
                    if du_p > t_knots[-1]:
                        r_pre = pre_curve.get_rate(float(du_p))
                        f_pre = (1.0 + r_pre) ** (float(du_p) / 252.0)
                        cp = ((f_pre / F_acc) ** (252.0 / float(du_p))) - 1.0
                        if round_decimals >= 0:
                            cp = round(cp, round_decimals)
                        t_knots.append(float(du_p))
                        y_knots.append(cp)

                curr_month = next_month
        else:
            # Antes da 1ª divulgação mensal do IGP-M (até 10 DU): k_i = 0 => E(IGPM) = 0 => DIM = PRE
            curr_month = datetime.date(base_date.year, base_date.month, 1)
            dt_15_raw = self.cal.following(datetime.date(curr_month.year, curr_month.month, 15))
            dt_15_date = dt_15_raw.date() if hasattr(dt_15_raw, 'date') else dt_15_raw
            du_pre_range = 10 if base_date.day >= 15 else max(1, self.cal.bizdays(base_date, dt_15_date))
            for du_p in range(1, du_pre_range + 1):
                t_knots.append(float(du_p))
                y_knots.append(pre_curve.get_rate(float(du_p)))

            F_acc = 1.0
            for m_idx in range(1, 500):
                next_month = curr_month + relativedelta(months=1)
                dt_end = self.cal.preceding(next_month - datetime.timedelta(days=1))
                if hasattr(dt_end, 'date'):
                    dt_end = dt_end.date()
                dt_15_next = self.cal.following(datetime.date(next_month.year, next_month.month, 15))
                if hasattr(dt_15_next, 'date'):
                    dt_15_next = dt_15_next.date()

                du_end = self.cal.bizdays(base_date, dt_end)
                du_15_next = self.cal.bizdays(base_date, dt_15_next)
                if du_end <= du_pre_range:
                    curr_month = next_month
                    continue
                if du_end > max_du:
                    break

                ref_key = (curr_month.year, curr_month.month)
                if m_idx <= 15 and ref_key in focus_monthly:
                    pi_m = focus_monthly[ref_key]
                elif dpl_curve is not None:
                    r_pre_end = pre_curve.get_rate(float(du_end))
                    r_dpl_end = dpl_curve.get_rate(float(du_end))
                    pi_m = (((1.0 + r_pre_end) / (1.0 + r_dpl_end)) ** (float(du_end) / 252.0) / F_acc) - 1.0
                else:
                    ann_rate = focus_annual.get(curr_month.year, 0.038)
                    pi_m = (1.0 + ann_rate) ** (1.0 / 12.0) - 1.0

                last_du = int(t_knots[-1])
                if du_end > last_du + 1:
                    span = float(du_end - last_du)
                    for du_p in range(last_du + 1, du_end):
                        pro_rata = float(du_p - last_du) / span
                        E_pr = (F_acc * (1.0 + pi_m * pro_rata)) - 1.0
                        r_pre = pre_curve.get_rate(float(du_p))
                        f_pre = (1.0 + r_pre) ** (float(du_p) / 252.0)
                        cp = ((f_pre / (1.0 + E_pr)) ** (252.0 / float(du_p))) - 1.0
                        if round_decimals >= 0:
                            cp = round(cp, round_decimals)
                        t_knots.append(float(du_p))
                        y_knots.append(cp)

                F_acc *= (1.0 + pi_m)

                # Cálculo nos vértices referenciais do patamar k_i = m_idx [du_end .. du_15_next]
                for du_p in range(du_end, min(du_15_next, max_du) + 1):
                    if not t_knots or du_p > t_knots[-1]:
                        r_pre = pre_curve.get_rate(float(du_p))
                        f_pre = (1.0 + r_pre) ** (float(du_p) / 252.0)
                        cp = ((f_pre / F_acc) ** (252.0 / float(du_p))) - 1.0
                        if round_decimals >= 0:
                            cp = round(cp, round_decimals)
                        t_knots.append(float(du_p))
                        y_knots.append(cp)

                curr_month = next_month

        # Interpolação por taxa a termo exponencial em base 252 para os demais prazos
        interp = FlatForward252Interpolator(0, 1)
        return SimpleCurve(
            np.array(t_knots, dtype=np.float64),
            np.array(y_knots, dtype=np.float64),
            interp,
            round_decimals=round_decimals,
            round_mode=round_mode
        )

    def build_clean_fx_coupon_curve(self,
                                    ddi_df: pd.DataFrame,
                                    base_date: datetime.date,
                                    ratio_cl_tm1: float = None,
                                    v1_rate: float = None,
                                    v1_dc: int = None,
                                    dol_cl: float = None,
                                    ptax_tm1: float = None,
                                    casado_1d: float = None,
                                    cdi_rate: float = None,
                                    rate_column: str = 'AdjstdQtTax',
                                    exp_date: str = 'XprtnDt',
                                    round_decimals: int = 4,
                                    round_mode: str = 'round') -> SimpleCurve:
        """
        Constrói a estrutura a termo de cupom cambial limpo (DOC e DCL)
        a partir dos vencimentos do contrato futuro de DDI e da razão DolCL_{t0} / Ptax_{t-1}:
        - Vértice 1: v1_rate explícito ou calculado via DolCL_{d1} = DolCL_{t0} + Casado_{1d}/1000
          e TC_1 = (((1 + CDI_1/100)^(1/252) / (DolCL_{d1} / DolCL_{t0})) - 1) * (36000 / DC_1)
        - Vértices de DDI: TC_n = ((1 + DDI_n * DC_n / 36000) * (DolCL_{t0} / Ptax_{t-1}) - 1) * 36000 / DC_n
        Interpola e extrapola os demais prazos via FlatForward252LinearInterpolator.
        """
        if ratio_cl_tm1 is None:
            if dol_cl is not None and ptax_tm1 is not None and float(ptax_tm1) > 0:
                ratio_cl_tm1 = float(dol_cl) / float(ptax_tm1)
            else:
                ratio_cl_tm1 = 1.0

        if isinstance(ddi_df, pd.Series):
            df = ddi_df.reset_index()
            df.columns = ['DU', 'rate'] if len(df.columns) == 2 else df.columns
        elif isinstance(ddi_df, dict):
            df = pd.DataFrame(list(ddi_df.items()), columns=['DU', 'rate'])
        elif isinstance(ddi_df, (list, tuple)):
            df = pd.DataFrame(ddi_df)
        else:
            df = ddi_df.copy()

        actual_rate = rate_column if rate_column in df.columns else ([c for c in df.columns if c.lower() in ['adjstdqttax', 'rate', 'taxa', 'yield']][0] if any(c.lower() in ['adjstdqttax', 'rate', 'taxa', 'yield'] for c in df.columns) else df.columns[-1])
        has_exp = exp_date in df.columns or any(c.lower() in ['xprtndt', 'maturity', 'vencimento'] for c in df.columns)
        if has_exp:
            actual_exp = exp_date if exp_date in df.columns else [c for c in df.columns if c.lower() in ['xprtndt', 'maturity', 'vencimento']][0]
            df = df.dropna(subset=[actual_rate, actual_exp]).copy()
            df['XprtnDt_date'] = pd.to_datetime(df[actual_exp]).dt.date
            df['DU'] = [self.cal.bizdays(base_date, x) for x in df['XprtnDt_date']]
            df['DC'] = [(x - base_date).days for x in df['XprtnDt_date']]
        else:
            col_du = 'DU' if 'DU' in df.columns else ('day252' if 'day252' in df.columns else df.columns[0])
            col_dc = 'DC' if 'DC' in df.columns else ('day360' if 'day360' in df.columns else None)
            df = df.dropna(subset=[actual_rate, col_du]).copy()
            df['DU'] = df[col_du].astype(float)
            df['DC'] = df[col_dc].astype(float) if col_dc is not None else np.round(df['DU'] * 365.0 / 252.0)

        df = df[df['DU'] > 0].sort_values('DU').copy()

        if v1_dc is None:
            next_bday = self.cal.offset(base_date, 1)
            if hasattr(next_bday, 'date'):
                next_bday = next_bday.date()
            v1_dc = max(1, (next_bday - base_date).days)

        if v1_rate is None and dol_cl is not None and casado_1d is not None and cdi_rate is not None:
            # Cálculo de TC_1 a partir do Casado de 1 dia útil e taxa DI
            cdi_dec = float(cdi_rate) / 100.0 if abs(float(cdi_rate)) > 1.0 else float(cdi_rate)
            dol_cl_d1 = float(dol_cl) + float(casado_1d) / 1000.0
            v1_pct = ((((1.0 + cdi_dec) ** (1.0 / 252.0)) / (dol_cl_d1 / float(dol_cl))) - 1.0) * (36000.0 / float(v1_dc))
            v1_dec = round(v1_pct, 2) / 100.0
        elif v1_rate is not None:
            v1_dec = float(v1_rate) / 100.0 if abs(float(v1_rate)) > 1.0 else float(v1_rate)
        else:
            v1_dec = None

        t_list = []
        dc_list = []
        y_list = []
        if v1_dec is not None:
            t_list.append(1.0)
            dc_list.append(float(v1_dc))
            y_list.append(v1_dec)

        col_is_pct = bool(np.nanmax(np.abs(df[actual_rate].values.astype(float))) > 1.0) if len(df) > 0 else True
        for _, row in df.iterrows():
            du = float(row['DU'])
            dc = float(row['DC'])
            if (du <= 1.0 and v1_dec is not None) or dc <= 0.0:
                continue
            raw_ddi = float(row[actual_rate])
            ddi_pct = raw_ddi if col_is_pct else raw_ddi * 100.0
            tc_pct = (((1.0 + (ddi_pct * dc) / 36000.0) * float(ratio_cl_tm1)) - 1.0) * (36000.0 / dc)
            tc_dec = round(tc_pct, 2) / 100.0
            t_list.append(du)
            dc_list.append(dc)
            y_list.append(tc_dec)

        interp = FlatForward252LinearInterpolator(0, 1)
        return SimpleCurve(
            np.array(t_list, dtype=np.float64),
            np.array(y_list, dtype=np.float64),
            interp,
            t_sec_nodes=np.array(dc_list, dtype=np.float64),
            round_decimals=round_decimals,
            round_mode=round_mode
        )

    def build_currency_forward_curve(self, 
                                     pre_curve: Curve, 
                                     foreign_curve: Curve, 
                                     spot_rate: float, 
                                     du_nodes: np.ndarray, 
                                     dc_nodes: np.ndarray, 
                                     config_name: str,
                                     spot_rate_tm1: float = None) -> Curve:
        """
        Constrói a estrutura a termo de câmbio a termo (PTX, EUR e JPY):
        - Para PTX:
          Vértice 1 (DU = 1): Ptax_{t0} (spot_rate)
          Demais vértices (DU > 1): TC_n = ((1 + PRE_n/100)^(DU_n/252) / (1 + DOL_n * DC_n / 36000)) * Ptax_{t-1}
        - Para EUR e JPY:
          Todos os vértices: TC_n = ((1 + PRE_n/100)^(DU_n/252) / (1 + Cupom_n * DC_n / 36000)) * Tx_{Reais x Moeda}
        """
        from .configs import CURVE_CONFIGS
        
        if config_name not in CURVE_CONFIGS:
            raise ValueError(f"Configuração '{config_name}' não encontrada.")
            
        config = CURVE_CONFIGS[config_name]
        round_decimals = config.get("round_decimals", 7)
        round_mode = config.get("round_mode", "trunc")
        linear_comp = (config.get("foreign_compounding", "linear") == "linear")
        is_ptx = (config_name.upper() == "PTX")
        ref_spot = float(spot_rate_tm1) if (is_ptx and spot_rate_tm1 is not None and spot_rate_tm1 > 0) else float(spot_rate)
        spot_t0 = float(spot_rate)

        return CurrencyForwardCurve(
            pre_curve=pre_curve,
            foreign_curve=foreign_curve,
            spot_t0=spot_t0,
            spot_ref=ref_spot,
            is_ptx=is_ptx,
            linear_comp=linear_comp,
            du_nodes=np.asarray(du_nodes, dtype=np.float64),
            dc_nodes=np.asarray(dc_nodes, dtype=np.float64),
            round_decimals=round_decimals,
            round_mode=round_mode
        )

    def build_equity_index_curve(self,
                                 dataset: pd.DataFrame,
                                 base_date: datetime.date,
                                 spot_index: float,
                                 config_name: str = "INP",
                                 price_column: str = "AdjstdQt",
                                 exp_date: str = "XprtnDt",
                                 round_decimals: int = None) -> SimpleCurve:
        """
        Constrói curvas de índices de ações e futuros de ações (BRP, INP, SML, MBR, VIX e ações individuais):
        - Vértice 1 (DU = 1): Spot_t
        - Vértices anteriores ao 1º vencimento (1 < DU_v < DU_1º):
          Spot_t * (Fut_1º / Spot_t)^(DU_v / DU_1º)
        - Demais vértices: interpolação exponencial de preços (PriceInterpolator) e extrapolação constante no fim.
        """
        from .configs import CURVE_CONFIGS
        config = CURVE_CONFIGS.get(config_name, {})

        if isinstance(dataset, pd.Series):
            df = dataset.reset_index()
            df.columns = ['DU', price_column]
        elif isinstance(dataset, dict):
            df = pd.DataFrame(list(dataset.items()), columns=['DU', price_column])
        elif isinstance(dataset, (list, tuple)):
            df = pd.DataFrame(dataset, columns=['DU', price_column])
        else:
            df = dataset.copy()

        if 'DU' in df.columns or 'day252' in df.columns:
            du_col = 'DU' if 'DU' in df.columns else 'day252'
            df = df.dropna(subset=[price_column, du_col]).copy()
            df['DU'] = df[du_col].astype(int)
        else:
            df = df.dropna(subset=[price_column, exp_date]).copy()
            df['XprtnDt_date'] = pd.to_datetime(df[exp_date]).dt.date
            df['DU'] = [self.cal.bizdays(base_date, x) for x in df['XprtnDt_date']]
        df = df[df['DU'] > 0].sort_values('DU').copy()

        t_list = df['DU'].astype(float).tolist()
        p_list = df[price_column].astype(float).tolist()

        # Vértice 1: Spot do índice no dia t
        if spot_index is not None and spot_index > 0:
            if 1.0 not in t_list:
                t_list.insert(0, 1.0)
                p_list.insert(0, float(spot_index))
            else:
                idx = t_list.index(1.0)
                p_list[idx] = float(spot_index)

        # Se round_decimals não for explicitamente informado, usa o do config
        if round_decimals is None:
            round_decimals = config.get("round_decimals", 2)

        # Para vértices anteriores ao 1º vencimento (1 < DU_v < DU_1º), insere-se a âncora em DU = 2.0
        # com valor Spot * (F_1 / Spot)^(2 / DU_1), de modo que PriceInterpolator reproduza analiticamente
        # Spot * (F_1 / Spot)^(DU_v / DU_1) para qualquer DU_v no intervalo [2, DU_1].
        if len(t_list) >= 2 and spot_index is not None and spot_index > 0:
            first_du = t_list[1]
            first_fut = p_list[1]
            if first_du > 2.0:
                val_2 = float(spot_index) * ((float(first_fut) / float(spot_index)) ** (2.0 / float(first_du)))
                t_list.insert(1, 2.0)
                p_list.insert(1, val_2)

        t_nodes = np.array(t_list, dtype=np.float64)
        p_nodes = np.array(p_list, dtype=np.float64)

        interp_cls = config.get("interpolator_class", PriceInterpolator)
        extrap_left = config.get("extrap_left", 0)
        extrap_right = config.get("extrap_right", 0)
        interp = interp_cls(extrap_left, extrap_right)

        round_mode = config.get("round_mode", "round")

        return SimpleCurve(t_nodes, p_nodes, interp, round_decimals=round_decimals, round_mode=round_mode)

    def build_convenience_yield_curve(self,
                                      spot_price: float,
                                      futures_curve: Curve,
                                      pre_curve: Curve,
                                      config_name: str = "CYI") -> ConvenienceYieldCurve:
        """
        Constrói a curva de Custo de Carrego / Convenience Yield:
        TxCY_n = (((Spot * (1 + PRE_n)^(DU/252)) / Futures_n)^(252 / DU) - 1)
        """
        from .configs import CURVE_CONFIGS
        config = CURVE_CONFIGS.get(config_name, {})
        round_decimals = config.get("round_decimals", 6)
        round_mode = config.get("round_mode", "round")
        return ConvenienceYieldCurve(spot_price, futures_curve, pre_curve, round_decimals=round_decimals, round_mode=round_mode)

    def build_spread_curve(self,
                           curve_a: Curve,
                           curve_b: Curve,
                           config_name: str = "SDE") -> SpreadCurve:
        """
        Constrói a curva de Spread de Taxas:
        Spread = Curve_A - Curve_B
        """
        from .configs import CURVE_CONFIGS
        config = CURVE_CONFIGS.get(config_name, {})
        round_decimals = config.get("round_decimals", 4)
        round_mode = config.get("round_mode", "round")
        return SpreadCurve(curve_a, curve_b, round_decimals=round_decimals, round_mode=round_mode)

    def build_di_x_tr_curve(self, pre_curve: Curve, tp_curve: Curve, round_decimals: int = 5, round_mode: str = 'round') -> SimpleCurve:
        """
        Constrói a curva DI x TR:
        TR_t = ((1 + PRE_t) / (1 + TP_t) - 1)
        """
        from bz_pricing.core.interpolators import FlatForward252Interpolator
        from bz_pricing.core.curves import SimpleCurve
        t_nodes = list(pre_curve.t_nodes)
        y_nodes = []
        for t in t_nodes:
            pre_val = pre_curve.get_rate(t)
            tp_val = tp_curve.get_rate(t)
            tr_val = ((1.0 + pre_val) / (1.0 + tp_val)) - 1.0
            if round_decimals >= 0:
                tr_val = round(tr_val, round_decimals)
            y_nodes.append(tr_val)
        interp = FlatForward252Interpolator(extrap_left=0, extrap_right=1)
        return SimpleCurve(np.array(t_nodes, dtype=np.float64), np.array(y_nodes, dtype=np.float64), interp, round_decimals=round_decimals, round_mode=round_mode)

    def build_euc_curve(self, doc_curve: Curve, sde_curve: Curve, round_decimals: int = 6, round_mode: str = 'round') -> Curve:
        """
        Constrói a Curva de Cupom de Euro / Iene (EUC e YCL):
        TC_n = ((1 + DOC_n * DC_n / 36000) * (1 + Spread_n * DC_n / 36000) - 1) * 36000 / DC_n
        """
        return CrossCouponCurve(
            doc_curve=doc_curve,
            spread_curve=sde_curve,
            round_decimals=round_decimals,
            round_mode=round_mode
        )

    def build_zero_coupon_curve(self,
                                forward_curve: Curve = None,
                                round_decimals: int = 5,
                                round_mode: str = 'round',
                                discount_factors: any = None) -> SimpleCurve:
        """
        Constrói a Curva Zero Cupom Internacional (ZMX, ZUS e ZEU):
        - Caso `discount_factors` (FD_JUR por prazo DC) seja fornecido:
          TxLinJUR(DC_v) = (1 / FD_JUR(DC_v) - 1) * (360 / DC_v)
        - Caso contrário, constrói a partir dos fatores de desconto acumulados da curva forward:
          FD(t) = prod_{i=1}^n 1 / (1 + f_i * delta_DC_i / 360)
          TxLinJUR(t) = (1 / FD(t) - 1) * (360 / DC_t)
        Interpolação linear em base 360 com extrapolação constante no fim.
        """
        from bz_pricing.core.interpolators import Linear360Interpolator
        from bz_pricing.core.curves import SimpleCurve

        if discount_factors is not None:
            dc_list = []
            fd_list = []
            if isinstance(discount_factors, dict):
                for k, v in sorted(discount_factors.items(), key=lambda x: float(x[0])):
                    dc_list.append(float(k))
                    fd_list.append(float(v))
            elif isinstance(discount_factors, pd.Series):
                for k, v in discount_factors.sort_index().items():
                    dc_list.append(float(k))
                    fd_list.append(float(v))
            elif isinstance(discount_factors, pd.DataFrame):
                df_fd = discount_factors.copy()
                dc_col = next((c for c in ["DC", "dc", "day360", "DU", "du"] if c in df_fd.columns), df_fd.columns[0])
                fd_col = next((c for c in ["FD", "fd", "discount_factor", "df", "value", "rate"] if c in df_fd.columns and c != dc_col), df_fd.columns[1])
                df_fd = df_fd.sort_values(dc_col)
                dc_list = [float(x) for x in df_fd[dc_col].values]
                fd_list = [float(x) for x in df_fd[fd_col].values]

            dense_dcs = []
            y_nodes = []
            for dc_v, fd_v in zip(dc_list, fd_list):
                if dc_v <= 0 or fd_v <= 0:
                    continue
                z_rate = (1.0 / fd_v - 1.0) * (360.0 / dc_v)
                if round_decimals >= 0:
                    if round_mode == 'trunc':
                        z_rate = np.trunc(z_rate * (10 ** round_decimals)) / (10 ** round_decimals)
                    else:
                        z_rate = round(z_rate, round_decimals)
                dense_dcs.append(dc_v)
                y_nodes.append(z_rate)

            interp = Linear360Interpolator(extrap_left=0, extrap_right=0)
            return SimpleCurve(
                np.array(dense_dcs, dtype=np.float64),
                np.array(y_nodes, dtype=np.float64),
                interp,
                round_decimals=round_decimals,
                round_mode=round_mode
            )

        # Extrai nós de dias corridos da curva forward
        if hasattr(forward_curve, 't_nodes') and len(forward_curve.t_nodes) > 0:
            dcs = sorted(list(set(forward_curve.t_nodes)))
        else:
            dcs = [1.0, 7.0, 30.0, 60.0, 90.0, 180.0, 270.0, 360.0, 540.0, 720.0, 1080.0, 1440.0, 1800.0, 2520.0, 3600.0]
        
        dense_dcs = [1.0]
        for dc in dcs:
            if dc > 1.0 and dc not in dense_dcs:
                dense_dcs.append(float(dc))
        dense_dcs.sort()
        
        y_nodes = []
        fd = 1.0
        prev_dc = 0.0
        
        for dc in dense_dcs:
            delta = dc - prev_dc
            # Taxa forward aplicável ao intervalo
            mid_dc = (dc + prev_dc) / 2.0
            fwd_rate = forward_curve.get_rate(mid_dc)
            fd *= 1.0 / (1.0 + fwd_rate * (delta / 360.0))
            z_rate = (1.0 / fd - 1.0) * (360.0 / dc)
            if round_decimals >= 0:
                if round_mode == 'trunc':
                    z_rate = np.trunc(z_rate * (10 ** round_decimals)) / (10 ** round_decimals)
                else:
                    z_rate = round(z_rate, round_decimals)
            y_nodes.append(z_rate)
            prev_dc = dc
            
        interp = Linear360Interpolator(extrap_left=0, extrap_right=0)
        return SimpleCurve(
            np.array(dense_dcs, dtype=np.float64),
            np.array(y_nodes, dtype=np.float64),
            interp,
            round_decimals=round_decimals,
            round_mode=round_mode
        )


