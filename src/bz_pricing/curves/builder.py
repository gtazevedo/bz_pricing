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
    Interpolation360Interpolator
)
from bz_pricing.core.curves import SimpleCurve, FlatCurve, Curve, InflationCurve

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
        
        # Filtra prazos negativos ou zero e ordena por DU crescente
        df = df[df['DU'] > 0].sort_values('DU').copy()
        
        # Preparação dos arrays
        t_list = df['DU'].tolist()
        y_list = (df[rate_column] / 100.0).tolist() # Converte taxa de % para decimal
        dc_list = df['DC'].tolist()

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
            # Garante que o Vértice 1 venha antes inserindo no índice 0
            if v1_du not in t_list:
                t_list.insert(0, v1_du)
                y_list.insert(0, v1_rate)
                dc_list.insert(0, v1_dc)
            else:
                # Se já existe o vértice no dataset, sobrescrevemos com a taxa v1_rate
                idx = t_list.index(v1_du)
                y_list[idx] = v1_rate
                dc_list[idx] = v1_dc

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
                        rate_column: str = 'AdjstdQtTax',
                        exp_date: str = 'XprtnDt',
                        round_decimals: int = 4,
                        round_mode: str = 'round') -> Curve:
        """
        Constrói a curva DI x IPCA Cupom Limpo (DPL) conforme Seção 3.2 do Manual de Curvas B3 (págs. 32-42).
        1. Calcula o Vértice 1 dia útil (DPL_{0,t}) usando CDI e EIPCA_0 / DU_{15a,15p} (pág. 33).
        2. Nos vértices coincidentes com vencimentos de DAP, extrai a inflação implícita:
           implicita_{n,t} = (1 + PRE_{n,t}) / (1 + DPL_{n,t}) - 1 (pág. 35, item a).
        3. Interpola a inflação implícita via FlatForward 252 (pág. 35, item b).
        4. Transforma novamente no cupom limpo DPL para todos os vértices n:
           DPL_{n,t} = (1 + PRE_{n,t}) / (1 + implicita_{n,t}) - 1 (pág. 35, item c).
        """
        df = dap_df.dropna(subset=[rate_column, exp_date]).copy()
        df['XprtnDt_date'] = pd.to_datetime(df[exp_date]).dt.date
        df['DU'] = [self.cal.bizdays(base_date, x) for x in df['XprtnDt_date']]
        df = df[df['DU'] > 0].sort_values('DU').copy()

        t_list = df['DU'].astype(float).tolist()
        dpl_list = (df[rate_column] / 100.0).tolist()

        # Vértice 1 dia útil (Seção 3.2, pág. 33)
        if eipca_0 is not None:
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
            if 1.0 not in t_list:
                t_list.insert(0, 1.0)
                dpl_list.insert(0, dpl_v1)
            else:
                idx = t_list.index(1.0)
                dpl_list[idx] = dpl_v1

        # Passo a (pág. 35): implicita_{n,t} = (1 + PRE_{n,t}) / (1 + DPL_{n,t}) - 1
        imp_list = []
        for du_val, dpl_val in zip(t_list, dpl_list):
            pre_val = pre_curve.get_rate(float(du_val))
            imp_val = ((1.0 + pre_val) / (1.0 + dpl_val)) - 1.0
            imp_list.append(imp_val)

        # Passo b (pág. 35): interpolação Flat Forward 252 sobre a inflação implícita
        imp_curve = SimpleCurve(
            np.array(t_list, dtype=np.float64),
            np.array(imp_list, dtype=np.float64),
            FlatForward252Interpolator(0, 1),
            round_decimals=-1
        )

        # Passo c (pág. 35): DPL_{n,t} = (1 + PRE_{n,t}) / (1 + implicita_{n,t}) - 1
        return InflationCurve(pre_curve, imp_curve, round_decimals=round_decimals, round_mode=round_mode)

    def build_dirty_inflation_coupon_curve(self,
                                           pre_curve: Curve,
                                           base_date: datetime.date,
                                           indicator: str = "IPCA",
                                           focus_monthly: dict = None,
                                           focus_annual: dict = None,
                                           dpl_curve: Curve = None,
                                           round_decimals: int = 4,
                                           round_mode: str = 'round') -> SimpleCurve:
        """
        Constrói as curvas de cupom sujo de inflação:
        - DIC: DI x IPCA Cupom Sujo (Seção 3.1 do Manual de Curvas B3, págs. 29-32)
        - DIM: DI x IGP-M Cupom Sujo (Seção 3.3 do Manual de Curvas B3, págs. 43-47)

        Aplica a contagem discreta de divulgações do índice (k_i) e o fator acumulado
        1 + E(Indice)_{k_i,t} nos vértices referenciais da curva intermediária (Equações 3.1.3 e 3.3.3),
        interpolando os demais vértices por Interpolation252Year (FlatForward252Interpolator).
        """
        focus_monthly = focus_monthly or {}
        focus_annual = focus_annual or {}
        max_du = int(max(pre_curve.t_nodes)) if len(pre_curve.t_nodes) > 0 else 10080

        t_knots = []
        y_knots = []

        if indicator.upper() == "IPCA":
            # Seção 3.1 (DIC): Antes da 1ª divulgação do IPCA (k_i = 0), E(IPCA)_{0,t} = 0 => Cp_{1,t} = PRE_{1,t}
            t_knots.append(1.0)
            y_knots.append(pre_curve.get_rate(1.0))

            F_acc = 1.0
            curr_month = datetime.date(base_date.year, base_date.month, 1)
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

                # Equação 3.1.3 nos vértices referenciais do patamar k_i = m_idx [du_15 .. du_01_next]
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
            # Seção 3.3 (DIM - IGP-M): Antes da 1ª divulgação do IGP-M no fim do mês (até dia 15 útil), k_i = 0 => E(IGPM) = 0
            curr_month = datetime.date(base_date.year, base_date.month, 1)
            dt_15_first = self.cal.following(datetime.date(curr_month.year, curr_month.month, 15))
            if hasattr(dt_15_first, 'date'):
                dt_15_first = dt_15_first.date()
            du_15_first = max(1, self.cal.bizdays(base_date, dt_15_first))
            for du_p in range(1, du_15_first + 1):
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
                if du_end <= 1:
                    curr_month = next_month
                    continue
                if du_end > max_du:
                    break

                ref_key = (curr_month.year, curr_month.month)
                if m_idx <= 15 and ref_key in focus_monthly:
                    pi_m = focus_monthly[ref_key]
                    F_acc *= (1.0 + pi_m)
                elif dpl_curve is not None:
                    r_pre_end = pre_curve.get_rate(float(du_end))
                    r_dpl_end = dpl_curve.get_rate(float(du_end))
                    F_acc = ((1.0 + r_pre_end) / (1.0 + r_dpl_end)) ** (float(du_end) / 252.0)
                else:
                    ann_rate = focus_annual.get(curr_month.year, 0.038)
                    pi_m = (1.0 + ann_rate) ** (1.0 / 12.0) - 1.0
                    F_acc *= (1.0 + pi_m)

                # Equação 3.3.3 nos vértices referenciais do patamar k_i = m_idx [du_end .. du_15_next]
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

        # Interpolação Interpolation252Year (FlatForward252Interpolator) para os demais vértices (págs. 32 e 47)
        interp = FlatForward252Interpolator(0, 1)
        return SimpleCurve(
            np.array(t_knots, dtype=np.float64),
            np.array(y_knots, dtype=np.float64),
            interp,
            round_decimals=round_decimals,
            round_mode=round_mode
        )

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
