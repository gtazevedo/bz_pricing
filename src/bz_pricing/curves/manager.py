import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from bcb import sgs

from bz_pricing.curves.builder import B3CurveBuilder
from bz_pricing.data.loaders import B3DataLoader
from bz_pricing.data.scrapers import B3Scraper
from bz_pricing.core.curves import Curve, SimpleCurve, FlatCurve
from bz_pricing.curves.configs import CURVE_CONFIGS
from bz_pricing.curves.exceptions import DataUnavailableError

def _first_not_none(*args):
    for a in args:
        if a is not None:
            return a
    return None

class CurveManager:
    """
    Orquestrador genérico e centralizado para construção de curvas de juros e derivativos.
    
    Princípios de Design:
    ---------------------
    1. A função de cálculo/construção de cada curva é ÚNICA e CENTRALIZADA, independente de os dados
       terem sido fornecidos externamente ou baixados da B3/BCB.
    2. Suporta passagem parcial ou total de insumos (ex: para DPL, pode-se fornecer apenas pre_curve,
       ou apenas dap_df, ou ambos, ou a curva final pronta).
    3. Controle estrito de download via flag `allow_download`:
       - Se um insumo obrigatório não for fornecido e `allow_download=True`, busca/baixa automaticamente da B3/BCB.
       - Se um insumo obrigatório não for fornecido e `allow_download=False` (padrão!), levanta DataUnavailableError
         especificando exatamente qual insumo está faltando.
    """
    
    def __init__(self, loader: B3DataLoader = None, builder: B3CurveBuilder = None, allow_download: bool = False):
        self.loader = loader if loader is not None else B3DataLoader()
        self.builder = builder if builder is not None else B3CurveBuilder()
        self.scraper = B3Scraper(output_dir=self.loader.data_dir)
        self.allow_download = allow_download
        
    def get_curve(self, curve_name: str, base_date: datetime.date = None, data: any = None, 
                  allow_download: bool = None, **kwargs) -> Curve:
        """
        Retorna uma Curva (SimpleCurve, InflationCurve, TBFCurve, TRCurve, FlatCurve, etc.) 
        baseada nas definições do configs.py.

        Parâmetros:
        -----------
        curve_name : str
            Código da curva (ex: 'PRE', 'DOL', 'DIC', 'DIM', 'DPL', 'TFP', 'INP', 'PETRP', etc.).
        base_date : datetime.date, str, pd.DataFrame, pd.Series, dict, list, opcional
            Data base de cálculo. Se fornecido como DataFrame/Series/dict (ex: get_curve('DOL', base_data)),
            é interpretado automaticamente como os dados externos da curva.
        data : pd.DataFrame, pd.Series, dict, list, float, opcional
            Dados de mercado para a construção da curva (contratos futuros ou nós de taxas/dias).
        allow_download : bool, opcional
            Se True, permite que insumos não fornecidos pelo usuário sejam baixados/buscados na B3/BCB.
            Se False (padrão do manager), levanta DataUnavailableError caso falte algum insumo obrigatório.
        **kwargs :
            Insumos específicos de cada ativo/curva para injeção personalizada (ex: pre_curve, pre_data,
            dap_df, dap_data, spot_price, focus_monthly, v1_rate, etc.).
        """
        if allow_download is None:
            allow_download = self.allow_download

        # Suporte a get_curve('DOL', base_data) onde o 2º argumento posicional é o dado externo
        if isinstance(base_date, (pd.DataFrame, pd.Series, dict, list, tuple)) and data is None:
            data = base_date
            base_date = kwargs.pop('date', None) or kwargs.pop('base_date', None)

        if isinstance(base_date, str):
            try:
                base_date = pd.to_datetime(base_date, dayfirst=True if '/' in base_date else False).date()
            except Exception:
                pass
                
        if base_date is None:
            base_date = datetime.date.today()

        if curve_name not in CURVE_CONFIGS:
            raise ValueError(f"Configuração da curva '{curve_name}' não encontrada no configs.py")
            
        config = CURVE_CONFIGS[curve_name]
        curve_type = config.get("type", "simple")

        # Verifica se 'data' é uma tabela de contratos brutos
        is_contracts_df = False
        if isinstance(data, pd.DataFrame):
            cols_lower = [str(c).lower() for c in data.columns]
            if any(k in cols_lower for k in ['xprtndt', 'maturity', 'vencimento', 'cdinstr', 'ticker']):
                is_contracts_df = True

        # Despacho centralizado conforme o tipo de curva
        try:
            if curve_type == "simple":
                return self._build_simple_curve(curve_name, config, base_date, data=data, allow_download=allow_download, **kwargs)
            elif curve_type == "derived":
                # Se o usuário passou diretamente os nós finais de taxas da curva (ex: DU e rate):
                has_comp_kwargs = any(k in kwargs for k in [
                    'pre_curve', 'pre_data', 'nominal_curve', 'nominal_data',
                    'dap_df', 'dap_data', 'real_data', 'dgi_df', 'dgi_data',
                    'focus_monthly', 'focus_data', 'eipca_0',
                    'informants_df', 'informants_data', 'v1_rate',
                    'doc_curve', 'doc_data', 'spread_curve', 'spread_data',
                    'sde_curve', 'sde_data', 'tbf_curve', 'tbf_data',
                    'tp_curve', 'tp_data', 'media_tbf', 'tbf_factor'
                ])
                if data is not None and not is_contracts_df and not has_comp_kwargs:
                    return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)
                return self._build_derived_curve(curve_name, config, base_date, data=data, allow_download=allow_download, **kwargs)
            elif curve_type == "flat":
                return self._build_flat_curve(curve_name, config, base_date, data=data, allow_download=allow_download, **kwargs)
            elif curve_type == "spread":
                has_spread_kwargs = any(k in kwargs for k in ['curve_a', 'curve_a_data', 'data_a', 'curve_b', 'curve_b_data', 'data_b'])
                if data is not None and not has_spread_kwargs:
                    return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)
                return self._build_spread_curve(curve_name, config, base_date, data=data, allow_download=allow_download, **kwargs)
            elif curve_type == "equity_index":
                return self._build_equity_index_curve(curve_name, config, base_date, data=data, allow_download=allow_download, **kwargs)
            elif curve_type == "convenience_yield":
                has_cy_kwargs = any(k in kwargs for k in ['spot_price', 'spot', 'futures_curve', 'futures_data', 'pre_curve', 'pre_data'])
                if data is not None and not has_cy_kwargs:
                    return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)
                return self._build_convenience_yield_curve(curve_name, config, base_date, data=data, allow_download=allow_download, **kwargs)
            elif curve_type == "currency_forward":
                has_fwd_kwargs = any(k in kwargs for k in ['spot_rate', 'spot_price', 'ptax', 'spot', 'pre_curve', 'pre_data', 'foreign_curve', 'foreign_data'])
                if data is not None and not has_fwd_kwargs:
                    return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)
                return self._build_currency_forward_curve(curve_name, config, base_date, data=data, allow_download=allow_download, **kwargs)
            elif curve_type == "derived_zero_coupon":
                has_zc_kwargs = any(k in kwargs for k in ['forward_curve', 'fwd_curve', 'forward_data', 'fwd_data'])
                if data is not None and not has_zc_kwargs:
                    return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)
                return self._build_zero_coupon_curve(curve_name, config, base_date, data=data, allow_download=allow_download, **kwargs)
            else:
                raise NotImplementedError(f"Tipo de curva '{curve_type}' não suportado.")
        except Exception as ex:
            if isinstance(ex, DataUnavailableError):
                raise ex
            data_str = base_date.strftime("%d/%m/%Y") if hasattr(base_date, 'strftime') else str(base_date)
            raise DataUnavailableError(
                f"Erro ao construir a curva '{curve_name}' na data {data_str}: {str(ex)}"
            ) from ex

    # =========================================================================
    # RESOLUÇÃO CENTRALIZADA DE INSUMOS (DEPENDENCY RESOLVER)
    # =========================================================================

    def _resolve_curve_input(self, param_name: str, default_curve_name: str, base_date: datetime.date,
                             allow_download: bool, curve_obj: any = None, data_obj: any = None, **kwargs) -> Curve:
        """
        Resolve uma curva dependente de forma inteligente e robusta:
        1. Se curve_obj ou data_obj for fornecido pelo usuário, constrói/usa diretamente.
        2. Se não fornecido e allow_download=True, busca/baixa a curva da B3/BCB.
        3. Se não fornecido e allow_download=False, levanta DataUnavailableError.
        """
        if curve_obj is not None:
            if isinstance(curve_obj, Curve):
                return curve_obj
            return self.get_curve(default_curve_name, base_date=base_date, data=curve_obj, allow_download=allow_download, **kwargs)

        if data_obj is not None:
            if isinstance(data_obj, Curve):
                return data_obj
            return self.get_curve(default_curve_name, base_date=base_date, data=data_obj, allow_download=allow_download, **kwargs)

        if not allow_download:
            raise DataUnavailableError(
                f"Insumo obrigatório '{param_name}' (curva base '{default_curve_name}') não foi informado para o cálculo "
                f"na data {base_date} e allow_download=False. Forneça o insumo através de '{param_name}' ou defina allow_download=True."
            )

        return self.get_curve(default_curve_name, base_date=base_date, allow_download=True, **kwargs)

    def _resolve_dataset_input(self, param_name: str, asset: str, base_date: datetime.date,
                               allow_download: bool, data_obj: any = None) -> pd.DataFrame:
        """
        Resolve um conjunto de contratos futuros (ex: DAP, DI1, DDI, etc.):
        1. Se data_obj for fornecido pelo usuário, formata como DataFrame e usa diretamente.
        2. Se não fornecido e allow_download=True, carrega/baixa da B3.
        3. Se não fornecido e allow_download=False, levanta DataUnavailableError.
        """
        if data_obj is not None:
            if isinstance(data_obj, pd.Series):
                df = data_obj.reset_index()
                df.columns = ['XprtnDt', 'AdjstdQtTax'] if len(df.columns) == 2 else df.columns
                return df
            elif isinstance(data_obj, (dict, list)):
                return pd.DataFrame(data_obj)
            elif isinstance(data_obj, pd.DataFrame):
                return data_obj.copy()
            else:
                raise ValueError(f"Formato inválido para '{param_name}': {type(data_obj)}")

        if not allow_download:
            raise DataUnavailableError(
                f"Insumo obrigatório '{param_name}' (contratos de '{asset}') não foi informado para a data {base_date} "
                f"e allow_download=False. Forneça os contratos via '{param_name}' ou defina allow_download=True."
            )

        try:
            df = self.loader.get_curve_instruments(base_date, asst=asset)
            if len(df) == 0:
                raise DataUnavailableError(f"Nenhum contrato encontrado para '{asset}' na data {base_date}.")
            return df
        except Exception as ex:
            if isinstance(ex, DataUnavailableError):
                raise ex
            raise DataUnavailableError(
                f"Não foi possível obter os contratos para '{asset}' na data {base_date}: {str(ex)}"
            ) from ex

    def _resolve_spot_rate(self, currency: str, base_date: datetime.date, allow_download: bool, spot_val: float = None) -> float:
        """
        Resolve a taxa spot oficial (Ptax) de uma moeda.
        """
        if spot_val is not None and float(spot_val) > 0:
            return float(spot_val)
        if not allow_download:
            raise DataUnavailableError(
                f"Cotação spot para a moeda '{currency}' não foi informada e allow_download=False. "
                f"Forneça 'spot_rate' ou defina allow_download=True para consultar a Ptax no Banco Central."
            )
        return self._fetch_currency_spot(currency, base_date)

    def _resolve_stock_spot(self, ticker: str, base_date: datetime.date, allow_download: bool, spot_val: float = None) -> float:
        """
        Resolve o preço spot à vista de uma ação.
        """
        if spot_val is not None and float(spot_val) > 0:
            return float(spot_val)
        if not allow_download:
            raise DataUnavailableError(
                f"Preço spot da ação '{ticker}' não foi informado e allow_download=False. "
                f"Forneça 'spot_price' explicitamente ou defina allow_download=True para consultar o Yahoo Finance."
            )
        spot = self.get_stock_spot(ticker, base_date)
        if spot is None:
            raise DataUnavailableError(f"Não foi possível obter a cotação spot para '{ticker}' na data {base_date}.")
        return spot

    def _resolve_focus_monthly(self, indicator: str, base_date: datetime.date, allow_download: bool, focus_obj: any = None) -> dict:
        """
        Resolve o dicionário {(ano, mes): taxa} com projeções de inflação do Focus.
        """
        if focus_obj is not None:
            if isinstance(focus_obj, dict):
                return focus_obj
            elif isinstance(focus_obj, pd.DataFrame):
                m_dict = {}
                for _, row in focus_obj.iterrows():
                    if 'DataReferencia' in row and 'Mediana' in row:
                        m_str, y_str = str(row['DataReferencia']).split('/')
                        m_dict[(int(y_str), int(m_str))] = float(row['Mediana']) / 100.0
                return m_dict
        if not allow_download:
            raise DataUnavailableError(
                f"Expectativas de inflação '{indicator}' não foram informadas e allow_download=False. "
                f"Forneça 'focus_monthly' (dicionário {{(ano, mes): taxa}}) ou defina allow_download=True para consultar a API Olinda do BCB."
            )
        return self._fetch_focus_monthly(indicator, base_date)

    # =========================================================================
    # MOTOR DE CÁLCULO E CONSTRUÇÃO DAS CURVAS
    # =========================================================================

    def _extract_linear360_knots(self, df_ref: pd.DataFrame) -> pd.DataFrame:
        """
        Extrai os nós base de interpolação Linear360 (itens 2.10, 2.11, 2.12 e 4.9 do Manual B3).
        Identifica os vértices onde a taxa linear a termo acumulada muda de inclinação,
        reconstituindo os vértices originais fornecidos pela LSEG/Refinitiv (Tabela 1).
        """
        if df_ref is None or len(df_ref) <= 2:
            return df_ref

        dcs = df_ref['day360'].values.astype(float)
        rates = df_ref['rate'].values.astype(float)

        curr = 0
        knots = [0]
        while curr < len(df_ref) - 1:
            best_next = curr + 1
            for cand in range(curr + 1, len(df_ref)):
                d1, r1 = dcs[curr], rates[curr]
                d2, r2 = dcs[cand], rates[cand]
                f1 = (r1 * d1) / 360.0
                f2 = (r2 * d2) / 360.0
                ok = True
                for m in range(curr + 1, cand):
                    dm = dcs[m]
                    f = f1 + (f2 - f1) * ((dm - d1) / (d2 - d1))
                    val = round((f * 360.0) / dm, 2)
                    if abs(val - rates[m]) > 0.001:
                        ok = False
                        break
                if ok:
                    best_next = cand
                else:
                    break
            knots.append(best_next)
            curr = best_next
            if curr >= len(df_ref) - 1:
                break
        return df_ref.iloc[knots].copy()

    def _extract_inflation_informants(self, df_ref: pd.DataFrame, pre_curve: Curve) -> pd.DataFrame:
        """
        Extrai apenas os vértices do Pool de Informantes (CpInformantes - Seções 3.1 DIC e 3.3 DIM do Manual B3)
        a partir da curva referencial:
        1. Nos primeiros 12 meses (DU <= 265), identifica os patamares discretos k_i (1º dia útil de cada mês)
           onde a inflação acumulada esperada E(Indice)_{k_i,t} (Equações 3.1.2 e 3.3.2) permanece constante.
        2. Para prazos longos (DU > 265 até 5580 DC / 8463 DU), extrai apenas os nós esparsos da interpolação
           Interpolation252Year (FlatForward252Interpolator).
        """
        if df_ref is None or len(df_ref) <= 2:
            return df_ref

        from bz_pricing.core.interpolators import FlatForward252Interpolator

        dus = df_ref['day252'].values.astype(int)
        dcs = df_ref['day360'].values.astype(int)
        rates = df_ref['rate'].values.astype(float)

        e_vals = []
        for du, r in zip(dus, rates):
            pre_pct = round(pre_curve.get_rate(float(du)) * 100.0, 2)
            e_val = round((((1.0 + pre_pct / 100.0) ** (du / 252.0)) / ((1.0 + r / 100.0) ** (du / 252.0)) - 1.0) * 100.0, 2)
            e_vals.append(e_val)

        short_mask = dus <= 265
        n_short = int(np.sum(short_mask))
        i = 0
        plateaus = []
        while i < n_short:
            j = i
            while j + 1 < n_short:
                e_cand = round(float(np.median(e_vals[i:j + 2])), 2)
                ok = True
                for m in range(i, j + 2):
                    du_m = int(dus[m])
                    pre_m = round(pre_curve.get_rate(float(du_m)) * 100.0, 2)
                    cp_m = round((((1.0 + pre_m / 100.0) ** (du_m / 252.0) / (1.0 + e_cand / 100.0)) ** (252.0 / du_m) - 1.0) * 100.0, 2)
                    if abs(cp_m - rates[m]) > 0.01001:
                        ok = False
                        break
                if ok:
                    j += 1
                else:
                    break
            if j > i:
                e_cand = round(float(np.median(e_vals[i:j + 1])), 2)
                plateaus.append((int(dus[i]), int(dus[j]), int(dcs[j]), float(rates[j]), e_cand))
                i = j + 1
            else:
                if i == 0:
                    plateaus.append((int(dus[0]), int(dus[0]), int(dcs[0]), float(rates[0]), 0.0))
                i += 1

        last_du = plateaus[-1][1] if plateaus else int(dus[0])
        sub = df_ref[df_ref['day252'] >= last_du].reset_index(drop=True)
        long_knots = []
        if len(sub) >= 2:
            sub_dus = sub['day252'].values.astype(float)
            sub_rates = sub['rate'].values.astype(float) / 100.0
            knots = [0, len(sub) - 1]
            while True:
                kn_sorted = sorted(knots)
                c = SimpleCurve(sub_dus[kn_sorted], sub_rates[kn_sorted], FlatForward252Interpolator(0, 1))
                preds = np.array([round(c.get_rate(float(d)) * 100.0, 2) for d in sub_dus])
                diffs = np.abs(preds - sub['rate'].values)
                max_idx = int(np.argmax(diffs))
                if diffs[max_idx] <= 0.01001 or max_idx in knots:
                    break
                knots.append(max_idx)
            long_knots = [
                (int(sub.iloc[k]['day252']), int(sub.iloc[k]['day252']), int(sub.iloc[k]['day360']), float(sub.iloc[k]['rate']), np.nan)
                for k in sorted(knots)[1:]
            ]

        all_rows = plateaus + long_knots
        return pd.DataFrame(all_rows, columns=['du_start', 'DU', 'DC', 'rate', 'e_inf'])

    def _extract_provider_vertices(self, df_ref: pd.DataFrame, curve_name: str, config: dict, base_date: datetime.date) -> pd.DataFrame:
        """
        Para curvas de swap e outras onde a única informação disponível é a própria curva
        da B3 (arquivos taxas_*.csv contendo centenas de pontos diários pré-interpolados):
        Captura apenas os vértices que a B3 coleta do provedor de informações
        (Refinitiv/LSEG conforme Tabela 1 do Manual de Curvas) ou contratos futuros subjacentes,
        garantindo que todos os demais pontos sejam interpolados pelo bz-pricing.
        """
        if df_ref is None or len(df_ref) <= 20:
            return df_ref

        c_name = curve_name.upper()
        prod = str(config.get("produto", c_name)).upper()
        interp_cls = config.get("interpolator_class")

        # 1. Curvas de Swaps e Spreads em Linear360 (ZUS, LIB, ZEU, ZMX, SDE)
        if c_name in ['ZUS', 'LIB', 'ZEU', 'ZMX', 'SDE'] or prod in ['LIB', 'SDE'] or (interp_cls and interp_cls.__name__ == 'Linear360Interpolator'):
            if 'day360' in df_ref.columns and 'rate' in df_ref.columns:
                return self._extract_linear360_knots(df_ref)

        # 2. Cupom Cambial Limpo (DOC, DCL)
        if c_name in ['DOC', 'DCL'] or prod in ['DOC', 'DCL']:
            try:
                df_ddi = self.loader.get_curve_instruments(base_date, asst='DDI')
                if df_ddi is not None and not df_ddi.empty:
                    ddi_dus = set([self.builder.cal.bizdays(base_date, pd.to_datetime(d).date()) for d in df_ddi['XprtnDt']])
                    doc_base = df_ref[df_ref['day252'].isin(ddi_dus.union({1}))].copy()
                    if len(doc_base) >= 5:
                        return doc_base
            except Exception:
                pass

        # 3. Forwards de Moedas (PTX, EUR, JPY) quando carregados por vértices
        if c_name in ['PTX', 'EUR', 'JPY'] or prod in ['PTX', 'EUR', 'JPY']:
            try:
                df_ddi = self.loader.get_curve_instruments(base_date, asst='DDI')
                if df_ddi is not None and not df_ddi.empty:
                    ddi_dus = set([self.builder.cal.bizdays(base_date, pd.to_datetime(d).date()) for d in df_ddi['XprtnDt']])
                    # Inclui V1, vencimentos de DDI e vértices anuais de swap longo (> 3500 DU)
                    long_dus = set(df_ref.loc[df_ref['day252'] > 3500, 'day252'].iloc[::3].tolist() + [int(df_ref['day252'].iloc[-1])])
                    fx_base = df_ref[df_ref['day252'].isin(ddi_dus.union({1}).union(long_dus))].copy()
                    if len(fx_base) >= 5:
                        return fx_base
            except Exception:
                pass

        # 4. Índices de Ações (BRP: BRI / INP: IND)
        if c_name in ['BRP', 'INP'] or prod in ['BRP', 'INP']:
            asst = 'BRI' if (c_name == 'BRP' or prod == 'BRP') else 'IND'
            try:
                df_fut = self.loader.get_curve_instruments(base_date, asst=asst)
                if df_fut is not None and not df_fut.empty:
                    fut_dus = set([self.builder.cal.bizdays(base_date, pd.to_datetime(d).date()) for d in df_fut['XprtnDt']])
                    # Inclui o segundo vértice padronizado antes do 1º vencimento para preservar a potência DU_v / DU_1º (Seção 5.2)
                    second_du = int(df_ref['day252'].iloc[1]) if len(df_ref) > 1 else 1
                    idx_base = df_ref[df_ref['day252'].isin(fut_dus.union({1, second_du}))].copy()
                    if len(idx_base) >= 2:
                        return idx_base
            except Exception:
                pass

        # 5. Cupom Limpo de IPCA (DPL)
        if c_name == 'DPL' or prod == 'DPL':
            try:
                df_dap = self.loader.get_curve_instruments(base_date, asst='DAP')
                if df_dap is not None and not df_dap.empty:
                    dap_dus = set([self.builder.cal.bizdays(base_date, pd.to_datetime(d).date()) for d in df_dap['XprtnDt']])
                    dap_base = df_ref[df_ref['day252'].isin(dap_dus.union({1}))].copy()
                    if len(dap_base) >= 5:
                        return dap_base
            except Exception:
                pass

        return df_ref

    def _build_from_nodes_data(self, curve_name: str, config: dict, base_date: datetime.date, data: any, **kwargs) -> SimpleCurve:
        """
        Constrói a curva a partir de nós de prazos e taxas (DataFrame, Series, dict ou list).
        Preserva as regras de interpolação, contagem de dias (DU vs DC), extrapolação e arredondamento do configs.py.
        """
        if isinstance(data, pd.Series):
            df = data.reset_index()
            df.columns = ['DU', 'rate']
        elif isinstance(data, dict):
            df = pd.DataFrame(list(data.items()), columns=['DU', 'rate'])
        elif isinstance(data, (list, tuple)):
            if len(data) > 0 and isinstance(data[0], (list, tuple)):
                if len(data[0]) >= 3:
                    df = pd.DataFrame(data, columns=['DU', 'DC', 'rate'])
                else:
                    df = pd.DataFrame(data, columns=['DU', 'rate'])
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Formato de dados não suportado: {type(data)}")

        if kwargs.get("filter_provider_knots", True) and isinstance(df, pd.DataFrame) and len(df) > 50:
            df = self._extract_provider_vertices(df, curve_name, config, base_date)

        interp_cls = config.get("interpolator_class")
        if interp_cls is None:
            from bz_pricing.core.interpolators import FlatForward252Interpolator
            interp_cls = FlatForward252Interpolator

        extrap_left = config.get("extrap_left", 0)
        extrap_right = config.get("extrap_right", 0)
        interp = interp_cls(extrap_left, extrap_right)

        col_du = 'DU' if 'DU' in df.columns else ('day252' if 'day252' in df.columns else None)
        col_dc = 'DC' if 'DC' in df.columns else ('day360' if 'day360' in df.columns else None)
        
        if col_du is None and col_dc is None:
            col_du = df.columns[0]
        
        use_dc = config.get("use_dc", False)
        index_by_dc = config.get("index_by_dc", False)
        target_t_col = col_dc if (index_by_dc and col_dc is not None) else (col_du or col_dc)
        
        rate_candidates = [c for c in df.columns if c.lower() in ['rate', 'taxa', 'adjstdqttax', 'yield', 'preco']]
        col_rate = rate_candidates[0] if rate_candidates else (df.columns[1] if len(df.columns) >= 2 else df.columns[-1])
        
        df = df.dropna(subset=[target_t_col, col_rate]).sort_values(target_t_col).copy()
        t_nodes = df[target_t_col].values.astype(np.float64)
        
        if index_by_dc:
            t_sec_nodes = None
        elif col_dc is not None and col_dc in df.columns and target_t_col != col_dc:
            t_sec_nodes = df[col_dc].values.astype(np.float64)
        elif use_dc or interp_cls.__name__ == 'FlatForward252LinearInterpolator':
            t_sec_nodes = np.round(t_nodes * 365.0 / 252.0).astype(np.float64)
        else:
            t_sec_nodes = None

        raw_rates = df[col_rate].values.astype(np.float64)
        divide_by_100 = config.get("divide_by_100", True)
        if divide_by_100:
            if (np.abs(raw_rates) > 1.0).any():
                y_nodes = raw_rates / 100.0
            else:
                y_nodes = raw_rates
        else:
            y_nodes = raw_rates

        return SimpleCurve(
            t_nodes,
            y_nodes,
            interp,
            t_sec_nodes=t_sec_nodes,
            round_decimals=config.get("round_decimals", -1),
            round_mode=config.get("round_mode", "round")
        )

    def _build_simple_curve(self, curve_name: str, config: dict, base_date: datetime.date,
                            data: any = None, allow_download: bool = False, **kwargs) -> Curve:
        asset = config.get("asset", curve_name)
        source_type = config.get("source_type", None)

        # Suporte à construção de ZUS / LIB por fatores de desconto (FD_JUR^{v,t} - Seção 2.11 Manual B3)
        discount_factors = _first_not_none(kwargs.get("discount_factors"), kwargs.get("fd_data"), kwargs.get("fd_df"))
        if discount_factors is not None and curve_name.upper() in ["ZUS", "LIB", "ZMX", "ZEU"]:
            return self.builder.build_zero_coupon_curve(
                discount_factors=discount_factors,
                round_decimals=kwargs.get("round_decimals", config.get("round_decimals", 5)),
                round_mode=kwargs.get("round_mode", config.get("round_mode", "round"))
            )

        # Suporte à construção de DOC / DCL por componentes (ddi_data, dol_cl, ptax_tm1, ratio_cl_tm1, casado_1d, cdi_rate, v1_rate)
        if curve_name.upper() in ["DOC", "DCL"]:
            has_doc_comp = any(k in kwargs for k in ['ddi_df', 'ddi_data', 'dol_cl', 'ptax_tm1', 'ratio_cl_tm1', 'casado_1d', 'cdi_rate'])
            if has_doc_comp:
                ddi_input = _first_not_none(kwargs.get("ddi_df"), kwargs.get("ddi_data"), data)
                df_ddi = self._resolve_dataset_input("ddi_data", "DDI", base_date, allow_download, ddi_input)
                return self.builder.build_clean_fx_coupon_curve(
                    ddi_df=df_ddi,
                    base_date=base_date,
                    ratio_cl_tm1=kwargs.get("ratio_cl_tm1"),
                    v1_rate=kwargs.get("v1_rate"),
                    v1_dc=kwargs.get("v1_dc"),
                    dol_cl=kwargs.get("dol_cl"),
                    ptax_tm1=kwargs.get("ptax_tm1"),
                    casado_1d=kwargs.get("casado_1d"),
                    cdi_rate=kwargs.get("cdi_rate"),
                    round_decimals=kwargs.get("round_decimals", config.get("round_decimals", 6)),
                    round_mode=kwargs.get("round_mode", config.get("round_mode", "round"))
                )

        if data is not None:
            is_contracts_df = False
            if isinstance(data, pd.DataFrame):
                cols_lower = [str(c).lower() for c in data.columns]
                if any(k in cols_lower for k in ['xprtndt', 'maturity', 'vencimento', 'cdinstr', 'ticker']):
                    is_contracts_df = True
            if not is_contracts_df and not (isinstance(data, pd.DataFrame) and 'AdjstdQtTax' in data.columns):
                return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)
            df = data.copy()
        else:
            if not allow_download:
                raise DataUnavailableError(
                    f"Dados de mercado para a curva '{curve_name}' não foram informados e allow_download=False. "
                    f"Forneça os dados através do parâmetro 'data' ou defina allow_download=True."
                )
            if source_type == "taxas_referenciais":
                produto = config.get("produto", curve_name)
                try:
                    df_ref = self.loader.load_taxas_referenciais(produto, base_date, auto_download=True)
                except Exception:
                    ref_path = self.scraper.download_taxas_referenciais(produto, base_date)
                    df_ref = pd.read_csv(ref_path, sep=';')

                # Seção 4.5 / 4.6 (DOC / DCL): Construção oficial a partir dos futuros de DDI + DolCL_t0 / Ptax_t-1
                if curve_name.upper() in ["DOC", "DCL"]:
                    try:
                        df_ddi = self.loader.get_curve_instruments(base_date, asst="DDI")
                        if df_ddi is not None and not df_ddi.empty and len(df_ref) > 1:
                            v1_row = df_ref[df_ref['day252'] == 1].iloc[0] if (df_ref['day252'] == 1).any() else df_ref.iloc[0]
                            v1_rate = float(v1_row['rate'])
                            v1_dc = int(v1_row['day360'])
                            # Estima a razão (DolCL_t0 / Ptax_t-1) a partir do 1º vencimento de DDI ou da paridade V1
                            cal = self.builder.cal
                            ddi_tmp = df_ddi.dropna(subset=['AdjstdQtTax', 'XprtnDt']).copy()
                            ddi_tmp['DU'] = [cal.bizdays(base_date, pd.to_datetime(d).date()) for d in ddi_tmp['XprtnDt']]
                            ddi_tmp['DC'] = [(pd.to_datetime(d).date() - base_date).days for d in ddi_tmp['XprtnDt']]
                            ddi_tmp = ddi_tmp[ddi_tmp['DU'] > 1].sort_values('DU')
                            r_lows, r_highs, ratios = [], [], []
                            for _, r_ddi in ddi_tmp.iterrows():
                                du_n, dc_n, ddi_r = int(r_ddi['DU']), float(r_ddi['DC']), float(r_ddi['AdjstdQtTax'])
                                match_doc = df_ref[df_ref['day252'] == du_n]
                                if not match_doc.empty:
                                    doc_r = float(match_doc['rate'].iloc[0])
                                    denom = 1.0 + (ddi_r * dc_n) / 36000.0
                                    if denom > 0:
                                        ratios.append((1.0 + (doc_r * dc_n) / 36000.0) / denom)
                                        r_lows.append((1.0 + ((doc_r - 0.005) * dc_n) / 36000.0) / denom)
                                        r_highs.append((1.0 + ((doc_r + 0.005) * dc_n) / 36000.0) / denom)
                            if ratios:
                                lo, hi = max(r_lows), min(r_highs)
                                ratio_cl_tm1 = 0.5 * (lo + hi) if lo <= hi else float(ratios[0])
                                return self.builder.build_clean_fx_coupon_curve(
                                    ddi_df=df_ddi,
                                    base_date=base_date,
                                    ratio_cl_tm1=ratio_cl_tm1,
                                    v1_rate=v1_rate,
                                    v1_dc=v1_dc,
                                    round_decimals=config.get("round_decimals", 4),
                                    round_mode=config.get("round_mode", "round")
                                )
                    except Exception:
                        pass

                return self._build_from_nodes_data(curve_name, config, base_date, df_ref, **kwargs)
            elif curve_name == "LTN":
                df = self.loader.get_anbima_bonds(base_date, "LTN")
            else:
                df = self.loader.get_curve_instruments(base_date, asst=asset)

        if len(df) == 0:
            raise DataUnavailableError(f"Dataset vazio para a curva '{curve_name}' (Ativo: {asset}) na data {base_date}.")

        # Vértice 1 (Se necessário)
        v1_rate = kwargs.get("v1_rate", None)
        if v1_rate is None and allow_download:
            v1_source = config.get("v1_source", None)
            if v1_source == "CDI":
                daily_cdi = self._fetch_sgs_rate(12, base_date, fallback=None)
                if daily_cdi is not None and daily_cdi > 0.0:
                    v1_rate = round(((1.0 + daily_cdi) ** 252.0 - 1.0) * 10000.0) / 10000.0
            elif v1_source == "SELIC":
                daily_selic = self._fetch_sgs_rate(11, base_date, fallback=None)
                if daily_selic is not None and daily_selic > 0.0:
                    v1_rate = round(((1.0 + daily_selic) ** 252.0 - 1.0) * 10000.0) / 10000.0
                else:
                    daily_cdi = self._fetch_sgs_rate(12, base_date, fallback=None)
                    if daily_cdi is not None and daily_cdi > 0.0:
                        v1_rate = round(((1.0 + daily_cdi) ** 252.0 - 1.0) * 10000.0) / 10000.0
            elif v1_source == "FED":
                try:
                    v1_rate = self.scraper.download_sofr_overnight(base_date) / 100.0
                except Exception:
                    v1_rate = None
            elif v1_source == "ECB":
                try:
                    v1_rate = self.scraper.download_estr_overnight(base_date) / 100.0
                except Exception:
                    v1_rate = None
            elif v1_source == "BANXICO":
                try:
                    v1_rate = self.scraper.download_banxico_tiie(base_date) / 100.0
                except Exception:
                    v1_rate = None

        smoothing = config.get("smoothing", None)
        return self.builder.build_from_config(
            df=df,
            config_name=curve_name,
            base_date=base_date,
            v1_rate=v1_rate,
            smoothing=smoothing,
            use_cython_smoothing=False
        )

    def _build_derived_curve(self, curve_name: str, config: dict, base_date: datetime.date, 
                             data: any = None, allow_download: bool = False, **kwargs) -> Curve:
        """
        Constrói curvas derivadas (DPL, DIC, DIM, TR, EUC, TFP, TP) utilizando as funções de cálculo centralizadas.
        Trata de forma elegante entradas parciais ou totais fornecidas pelo usuário.
        """
        # Se for TBF_PRE, TBF_LTN ou TFP, precisa da base_curve e histórico BCB (Seção 2.4 Manual B3)
        if curve_name in ["TBF_PRE", "TBF_LTN", "TFP"]:
            base_curve_name = config.get("base_curve", "PRE")
            return self._build_tbf_pipeline(base_date, base_curve_name, allow_download=allow_download, **kwargs)
            
        # Se for TR_PRE, TR_LTN ou TP, puxa a base TBF (Seção 2.5 Manual B3 / Res. CMN 5.124 BCB)
        elif curve_name in ["TR_PRE", "TR_LTN", "TP"]:
            base_curve_name = config.get("base_curve", "TFP" if curve_name == "TP" else "TBF_PRE")
            tbf_input = _first_not_none(kwargs.get("tbf_curve"), kwargs.get("base_curve"), data)
            tbf_data = _first_not_none(kwargs.get("tbf_data"), kwargs.get("base_data"))
            tbf_curve = self._resolve_curve_input("tbf_curve", base_curve_name, base_date, allow_download, tbf_input, tbf_data)
            round_decimals = kwargs.get("round_decimals", config.get("round_decimals", -1))
            return self.builder.build_tr_curve(tbf_curve, round_decimals=round_decimals)

        # Seção 2.6 Manual B3: Curva DI x TR (TR)
        elif curve_name == "TR":
            pre_input = kwargs.get("pre_curve")
            pre_data = kwargs.get("pre_data")
            pre_curve = self._resolve_curve_input("pre_curve", config.get("pre_curve", "PRE"), base_date, allow_download, pre_input, pre_data)

            tp_input = kwargs.get("tp_curve")
            tp_data = kwargs.get("tp_data")
            tp_curve = self._resolve_curve_input("tp_curve", config.get("tp_curve", "TP"), base_date, allow_download, tp_input, tp_data)

            round_decimals = kwargs.get("round_decimals", config.get("round_decimals", 5))
            round_mode = kwargs.get("round_mode", config.get("round_mode", "round"))
            return self.builder.build_di_x_tr_curve(pre_curve, tp_curve, round_decimals=round_decimals, round_mode=round_mode)

        # Seção 4.7 Manual B3: Curva de Cupom de Euro (EUC)
        elif curve_name in ["EUC", "YCL"]:
            doc_input = kwargs.get("doc_curve")
            doc_data = kwargs.get("doc_data")
            doc_curve = self._resolve_curve_input("doc_curve", config.get("doc_curve", "DOC"), base_date, allow_download, doc_input, doc_data)

            spread_name = config.get("spread_curve", "SDE" if curve_name == "EUC" else "SYD")
            spread_input = _first_not_none(kwargs.get("spread_curve"), kwargs.get("sde_curve"), kwargs.get("syd_curve"))
            spread_data = _first_not_none(kwargs.get("spread_data"), kwargs.get("sde_data"), kwargs.get("syd_data"))
            sde_curve = self._resolve_curve_input("spread_curve", spread_name, base_date, allow_download, spread_input, spread_data)

            round_decimals = kwargs.get("round_decimals", config.get("round_decimals", 6))
            round_mode = kwargs.get("round_mode", config.get("round_mode", "round"))
            return self.builder.build_euc_curve(doc_curve, sde_curve, round_decimals=round_decimals, round_mode=round_mode)

        # Seção 3.2 Manual B3: DI x IPCA Cupom Limpo (DPL)
        elif curve_name == "DPL":
            # 1. Resolve PRE curve
            pre_input = _first_not_none(kwargs.get("pre_curve"), kwargs.get("nominal_curve"))
            pre_data = _first_not_none(kwargs.get("pre_data"), kwargs.get("nominal_data"))
            pre_curve = self._resolve_curve_input("pre_curve", config.get("nominal_curve", "PRE"), base_date, allow_download, pre_input, pre_data)

            # 2. Resolve DAP dataset
            dap_input = _first_not_none(kwargs.get("dap_df"), kwargs.get("dap_data"), kwargs.get("real_data"), data)
            df_dap = self._resolve_dataset_input("dap_data", config.get("real_curve", "DAP"), base_date, allow_download, dap_input)

            # 3. Resolve Vértice 1 (v1_rate / eipca_0 conforme Seção 3.2 pág. 33-35)
            eipca_0 = kwargs.get("eipca_0")
            v1_rate = kwargs.get("v1_rate")
            if eipca_0 is None and v1_rate is None:
                if allow_download:
                    try:
                        df_dpl_ref = self.loader.load_taxas_referenciais("DPL", base_date, auto_download=True)
                        if df_dpl_ref is not None and not df_dpl_ref.empty:
                            v1_rows = df_dpl_ref[df_dpl_ref['day252'] == 1]
                            v1_rate = float(v1_rows['rate'].iloc[0]) if not v1_rows.empty else float(df_dpl_ref['rate'].iloc[0])
                    except Exception:
                        pass
                    if v1_rate is None:
                        focus_ipca = self._fetch_focus_monthly("IPCA", base_date)
                        prev_m = base_date - relativedelta(months=1)
                        eipca_0 = focus_ipca.get((prev_m.year, prev_m.month), focus_ipca.get((base_date.year, base_date.month), 0.0))
                else:
                    focus_obj = _first_not_none(kwargs.get("focus_monthly"), kwargs.get("focus_data"))
                    if focus_obj is not None:
                        focus_ipca = self._resolve_focus_monthly("IPCA", base_date, False, focus_obj)
                        prev_m = base_date - relativedelta(months=1)
                        eipca_0 = focus_ipca.get((prev_m.year, prev_m.month), focus_ipca.get((base_date.year, base_date.month), 0.0))
                    else:
                        eipca_0 = 0.0

            return self.builder.build_dpl_curve(
                pre_curve=pre_curve,
                dap_df=df_dap,
                base_date=base_date,
                eipca_0=eipca_0,
                v1_rate=v1_rate,
                round_decimals=kwargs.get("round_decimals", config.get("round_decimals", 4)),
                round_mode=kwargs.get("round_mode", config.get("round_mode", "round"))
            )

        # Seções 3.1 (DIC) e 3.3 (DIM) Manual B3: Curvas de Cupom Sujo de Inflação
        elif curve_name in ["DIC", "DIM"]:
            nominal_curve_name = config.get("nominal_curve", "PRE")
            indicator = config.get("indicator", "IPCA" if curve_name == "DIC" else "IGP-M")

            # 1. Resolve PRE curve
            pre_input = _first_not_none(kwargs.get("pre_curve"), kwargs.get("nominal_curve"))
            pre_data = _first_not_none(kwargs.get("pre_data"), kwargs.get("nominal_data"))
            pre_curve = self._resolve_curve_input("pre_curve", nominal_curve_name, base_date, allow_download, pre_input, pre_data)

            # 2. Resolve Pool de Informantes (CpInformantes - Seções 3.1 e 3.3)
            informants_df = _first_not_none(kwargs.get("informants_df"), kwargs.get("informants_data"))
            focus_obj = _first_not_none(kwargs.get("focus_monthly"), kwargs.get("focus_data"))
            if informants_df is None and allow_download and focus_obj is None:
                try:
                    df_ref = self.loader.load_taxas_referenciais(curve_name, base_date, auto_download=True)
                    if df_ref is not None and not df_ref.empty:
                        informants_df = self._extract_inflation_informants(df_ref, pre_curve)
                except Exception:
                    informants_df = None

            if informants_df is not None:
                return self.builder.build_dirty_inflation_coupon_curve(
                    pre_curve=pre_curve,
                    base_date=base_date,
                    indicator=indicator,
                    informants_df=informants_df,
                    round_decimals=kwargs.get("round_decimals", config.get("round_decimals", 4)),
                    round_mode=kwargs.get("round_mode", config.get("round_mode", "round"))
                )

            # 3. Fallback: Expectativas de Inflação Focus + Curva DPL
            focus_monthly = self._resolve_focus_monthly(indicator, base_date, allow_download, focus_obj)

            dpl_curve = kwargs.get("dpl_curve")
            dpl_data = kwargs.get("dpl_data")
            if dpl_curve is None and dpl_data is not None:
                dpl_curve = self._resolve_curve_input("dpl_curve", "DPL", base_date, allow_download, data_obj=dpl_data)
            elif dpl_curve is None and allow_download:
                try:
                    dpl_curve = self.get_curve("DPL", base_date, allow_download=True, pre_curve=pre_curve)
                except Exception:
                    dpl_curve = None

            return self.builder.build_dirty_inflation_coupon_curve(
                pre_curve=pre_curve,
                base_date=base_date,
                indicator=indicator,
                focus_monthly=focus_monthly,
                dpl_curve=dpl_curve,
                round_decimals=kwargs.get("round_decimals", config.get("round_decimals", 4)),
                round_mode=kwargs.get("round_mode", config.get("round_mode", "round"))
            )
        else:
            raise NotImplementedError(f"Lógica de Curva Derivada não implementada para {curve_name}")

    def _build_tbf_pipeline(self, base_date: datetime.date, base_curve_name: str, allow_download: bool = False, **kwargs) -> Curve:
        """
        Implementa o cálculo centralizado da curva TBF x PRE (TFP) conforme Seção 2.4 do Manual B3.
        """
        pre_t0_input = _first_not_none(kwargs.get("pre_curve"), kwargs.get("base_curve"))
        pre_t0_data = _first_not_none(kwargs.get("pre_data"), kwargs.get("base_data"))
        pre_t0 = self._resolve_curve_input("pre_curve", base_curve_name, base_date, allow_download, pre_t0_input, pre_t0_data)

        media_tbf = _first_not_none(kwargs.get("media_tbf"), kwargs.get("tbf_factor"))
        if media_tbf is not None:
            config = CURVE_CONFIGS["TBF_PRE"]
            return self.builder.build_tbf_curve(pre_t0, float(media_tbf), round_decimals=config.get("round_decimals", -1))

        if not allow_download:
            raise DataUnavailableError(
                f"Fator 'media_tbf' não foi informado para o cálculo de TFP na data {base_date} e allow_download=False. "
                f"Forneça 'media_tbf' explicitamente ou defina allow_download=True para buscar o histórico no BCB."
            )

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
                pre_curve_past = self.get_curve(base_curve_name, d, allow_download=True)
                
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
                
                dt_30 = self.builder.cal.following(d + datetime.timedelta(days=30))
                if hasattr(dt_30, 'date'):
                    dt_30 = dt_30.date()
                du_30 = float(self.builder.cal.bizdays(d, dt_30))
                
                pre30_rate = round(pre_curve_past.get_rate(du_30) * 100.0, 2) / 100.0
                
                tbf_30 = (1.0 + tbf_am) ** (du_30 / du_mes) - 1.0
                pre_30 = (1.0 + pre30_rate) ** (du_30 / 252.0) - 1.0
                
                fator = tbf_30 / pre_30 if pre_30 != 0.0 else 0.0
                fatores.append(fator)
            except Exception:
                continue
                
        if not fatores:
            raise DataUnavailableError(f"Não foi possível apurar os fatores TBF dos últimos 5 dias para a data {base_date}.")
        
        media_tbf = sum(fatores) / len(fatores)
        config = CURVE_CONFIGS["TBF_PRE"]
        return self.builder.build_tbf_curve(pre_t0, media_tbf, round_decimals=config.get("round_decimals", -1))

    def _build_flat_curve(self, curve_name: str, config: dict, base_date: datetime.date,
                          data: any = None, allow_download: bool = False, **kwargs) -> Curve:
        """
        Constrói curvas flat (taxas únicas diárias/mensais).
        """
        if data is not None:
            if isinstance(data, (int, float)):
                rate = float(data)
            elif isinstance(data, pd.Series):
                rate = float(data.dropna().iloc[-1])
            elif isinstance(data, pd.DataFrame):
                rate = float(data.dropna().iloc[-1, -1])
            elif isinstance(data, dict):
                rate = float(list(data.values())[-1])
            else:
                rate = float(data)
            if rate > 1.0:
                rate = rate / 100.0
            return self.builder.build_flat_curve(rate, config_name=curve_name)

        if not allow_download:
            raise DataUnavailableError(
                f"Taxa para a curva flat '{curve_name}' não foi informada e allow_download=False. "
                f"Forneça o valor via parâmetro 'data' ou defina allow_download=True."
            )

        source = config.get("source")
        if source == "BCB":
            series = config.get("series_id")
            rate = self._fetch_sgs_rate(series, base_date, fallback=None)
            if rate is None:
                raise DataUnavailableError(f"Não foi possível obter a taxa para {curve_name} da série SGS {series} na data {base_date}.")
            return self.builder.build_flat_curve(rate, config_name=curve_name)
        else:
            raise NotImplementedError("Source de flat curve não implementado.")

    def _build_spread_curve(self, curve_name: str, config: dict, base_date: datetime.date,
                            data: any = None, allow_download: bool = False, **kwargs) -> Curve:
        """
        Constrói curvas de spread (SDE, SGP, SMX, SYD): Curve_A - Curve_B
        """
        if data is not None:
            return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)

        name_a = config.get("curve_a")
        name_b = config.get("curve_b")
        curve_a = self._resolve_curve_input("curve_a", name_a, base_date, allow_download, kwargs.get("curve_a"), _first_not_none(kwargs.get("curve_a_data"), kwargs.get("data_a")))
        curve_b = self._resolve_curve_input("curve_b", name_b, base_date, allow_download, kwargs.get("curve_b"), _first_not_none(kwargs.get("curve_b_data"), kwargs.get("data_b")))
        return self.builder.build_spread_curve(curve_a, curve_b, config_name=curve_name)

    def _build_equity_index_curve(self, curve_name: str, config: dict, base_date: datetime.date,
                                  data: any = None, allow_download: bool = False, **kwargs) -> Curve:
        """
        Constrói curvas de índices de ações e futuros de ações (BRP, INP, SML, MBR, VIX ou ações).
        Seções 5.1 (BRP) e 5.2 (INP) do Manual de Curvas da B3.
        """
        if data is not None and isinstance(data, pd.DataFrame) and "day252" in data.columns and "XprtnDt" not in data.columns:
            return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)

        asset = config.get("asset", curve_name)
        raw_data = _first_not_none(data, kwargs.get("futures_data"), kwargs.get("contracts_data"))
        df = self._resolve_dataset_input("contracts_data", asset, base_date, allow_download, raw_data)

        price_candidates = [c for c in df.columns if c.lower() in ["adjstdqt", "price", "preco", "rate", "taxa"]]
        price_col = price_candidates[0] if price_candidates else df.columns[-1]
        first_fut = float(df[price_col].iloc[0])
        spot_index = _first_not_none(kwargs.get("spot_price"), kwargs.get("spot"), config.get("spot_price"))

        if spot_index is None and allow_download:
            try:
                df_ref = self.loader.load_taxas_referenciais(curve_name, base_date)
                if df_ref is not None and not df_ref.empty:
                    v1_rows = df_ref[df_ref["day252"] == 1]
                    if not v1_rows.empty:
                        spot_index = float(v1_rows["rate"].iloc[0])
            except Exception:
                pass

        if spot_index is None:
            spot_index = first_fut

        exp_col = "XprtnDt" if "XprtnDt" in df.columns else df.columns[0]

        return self.builder.build_equity_index_curve(
            dataset=df,
            base_date=base_date,
            spot_index=spot_index,
            config_name=curve_name,
            price_column=price_col,
            exp_date=exp_col
        )

    def _build_convenience_yield_curve(self, curve_name: str, config: dict, base_date: datetime.date,
                                       data: any = None, allow_download: bool = False, **kwargs) -> Curve:
        """
        Constrói curvas de conveniência/custo de carrego (CYI, CYM, CYX):
        TxCY = (((Spot * (1 + PRE)^(DU/252)) / Futures)^(252 / DU) - 1)
        """
        if data is not None:
            return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)

        index_curve_name = config.get("index_curve")
        pre_curve_name = config.get("pre_curve", "PRE")
        fut_curve = self._resolve_curve_input("futures_curve", index_curve_name, base_date, allow_download, kwargs.get("futures_curve"), kwargs.get("futures_data"))
        pre_curve = self._resolve_curve_input("pre_curve", pre_curve_name, base_date, allow_download, kwargs.get("pre_curve"), kwargs.get("pre_data"))

        spot_price = _first_not_none(kwargs.get("spot_price"), kwargs.get("spot"), fut_curve.get_rate(1.0))
        return self.builder.build_convenience_yield_curve(
            spot_price=spot_price,
            futures_curve=fut_curve,
            pre_curve=pre_curve,
            config_name=curve_name
        )

    def _build_currency_forward_curve(self, curve_name: str, config: dict, base_date: datetime.date,
                                      data: any = None, allow_download: bool = False, **kwargs) -> Curve:
        """
        Constrói a Curva de Forward de Moedas (PTX, EUR, JPY - Seções 4.1 a 4.3 Manual B3)
        """
        if data is not None:
            return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)

        currency = config.get("currency", "USD")
        explicit_spot = _first_not_none(kwargs.get("spot_rate"), kwargs.get("spot_price"), kwargs.get("ptax"), kwargs.get("spot"))
        explicit_spot_tm1 = _first_not_none(kwargs.get("spot_rate_tm1"), kwargs.get("ptax_tm1"))

        if curve_name.upper() == "JPY" and kwargs.get("foreign_curve") is None and kwargs.get("foreign_data") is None:
            # Para JPY (Seção 4.3), a B3 não publica YCL/SYD separadamente no portal público de taxas referenciais;
            # quando o usuário não injeta YCL explicitamente, extrai os vértices DDI/Swap da tabela de referência e interpola via PriceInterpolator
            if allow_download and config.get("source_type") == "taxas_referenciais":
                return self._build_simple_curve(curve_name, config, base_date, data=data, allow_download=allow_download, **kwargs)

        pre_curve = self._resolve_curve_input("pre_curve", config.get("pre_curve", "PRE"), base_date, allow_download, kwargs.get("pre_curve"), kwargs.get("pre_data"))
        foreign_name = config.get("foreign_curve", "DOL")
        foreign_curve = self._resolve_curve_input("foreign_curve", foreign_name, base_date, allow_download, kwargs.get("foreign_curve"), kwargs.get("foreign_data"))

        spot_rate = explicit_spot
        spot_rate_tm1 = explicit_spot_tm1

        if currency.upper() == "USD":
            if spot_rate is None:
                spot_rate = self._resolve_spot_rate(currency, base_date, allow_download)
            if spot_rate_tm1 is None:
                if explicit_spot is not None:
                    spot_rate_tm1 = explicit_spot
                else:
                    try:
                        prev_bd = self.builder.cal.offset(base_date, -1)
                        if isinstance(prev_bd, str):
                            prev_bd = pd.to_datetime(prev_bd).date()
                        spot_rate_tm1 = self._resolve_spot_rate(currency, prev_bd, allow_download)
                    except Exception:
                        spot_rate_tm1 = spot_rate
        elif currency.upper() == "EUR":
            # Na Seção 4.2, a B3 utiliza a Tx_Reais x Euro de fechamento apurada pela B3
            if spot_rate is None and allow_download:
                try:
                    df_ref = self.loader.load_taxas_referenciais(curve_name, base_date)
                    if df_ref is not None and not df_ref.empty:
                        sample = df_ref[df_ref["day252"] <= 21]
                        if sample.empty:
                            sample = df_ref.head(5)
                        implied_spots = []
                        for _, r in sample.iterrows():
                            du_r = float(r["day252"])
                            dc_r = float(r["day360"])
                            eur_r = float(r["rate"])
                            pre_r = pre_curve.get_rate(du_r, dc_r)
                            euc_r = foreign_curve.get_rate(du_r, dc_r)
                            denom = (1.0 + pre_r) ** (du_r / 252.0)
                            if denom > 0:
                                implied_spots.append(eur_r * (1.0 + euc_r * (dc_r / 360.0)) / denom)
                        if implied_spots:
                            spot_rate = float(np.median(implied_spots))
                except Exception:
                    pass
            if spot_rate is None:
                spot_rate = self._resolve_spot_rate(currency, base_date, allow_download)
            spot_rate_tm1 = spot_rate
        else:
            if spot_rate is None:
                spot_rate = self._resolve_spot_rate(currency, base_date, allow_download)
            spot_rate_tm1 = spot_rate

        du_nodes = np.array(pre_curve.t_nodes, dtype=np.float64)
        t_sec = pre_curve.t_sec_nodes if hasattr(pre_curve, 't_sec_nodes') and pre_curve.t_sec_nodes is not None else None
        dc_nodes = np.array(t_sec if t_sec is not None else du_nodes, dtype=np.float64)
        
        return self.builder.build_currency_forward_curve(
            pre_curve=pre_curve,
            foreign_curve=foreign_curve,
            spot_rate=spot_rate,
            du_nodes=du_nodes,
            dc_nodes=dc_nodes,
            config_name=curve_name,
            spot_rate_tm1=spot_rate_tm1
        )

    def _build_zero_coupon_curve(self, curve_name: str, config: dict, base_date: datetime.date,
                                 data: any = None, allow_download: bool = False, **kwargs) -> Curve:
        """
        Constrói a curva zero cupom internacional (ZMX, ZUS, ZEU - Seções 2.10 a 2.12 Manual B3).
        Aceita `discount_factors` (FD_JUR^{v,t} por DC), `data` (vértices de taxa) ou `forward_curve`.
        """
        round_decimals = kwargs.get("round_decimals", config.get("round_decimals", 5))
        round_mode = kwargs.get("round_mode", config.get("round_mode", "round"))
        discount_factors = _first_not_none(kwargs.get("discount_factors"), kwargs.get("fd_data"), kwargs.get("fd_df"))
        if discount_factors is not None:
            return self.builder.build_zero_coupon_curve(
                discount_factors=discount_factors,
                round_decimals=round_decimals,
                round_mode=round_mode
            )

        if data is not None:
            return self._build_from_nodes_data(curve_name, config, base_date, data, **kwargs)

        fwd_name = config.get("forward_curve")
        fwd_input = _first_not_none(kwargs.get("forward_curve"), kwargs.get("fwd_curve"))
        fwd_data = _first_not_none(kwargs.get("forward_data"), kwargs.get("fwd_data"))
        fwd_curve = self._resolve_curve_input("forward_curve", fwd_name, base_date, allow_download, fwd_input, fwd_data)
        return self.builder.build_zero_coupon_curve(fwd_curve, round_decimals=round_decimals, round_mode=round_mode)

    # =========================================================================
    # FUTUROS E CONVENIENCE YIELD DE AÇÕES
    # =========================================================================

    def get_stock_spot(self, ticker: str, base_date: datetime.date) -> float:
        """
        Busca o preço à vista (Spot) da ação via Yahoo Finance.
        Retorna None se a cotação não for encontrada.
        """
        symbol_map = {
            "PETRP": "PETR4.SA",
            "PETR4": "PETR4.SA",
            "VALEO": "VALE3.SA",
            "VALE3": "VALE3.SA",
            "ITUB4": "ITUB4.SA",
            "BBDC4": "BBDC4.SA",
            "BBAS3": "BBAS3.SA",
        }
        yf_sym = symbol_map.get(ticker.upper(), f"{ticker.upper()}.SA")
        try:
            import yfinance as yf
            start_d = base_date - datetime.timedelta(days=7)
            end_d = base_date + datetime.timedelta(days=1)
            t = yf.Ticker(yf_sym)
            df_hist = t.history(start=start_d.strftime("%Y-%m-%d"), end=end_d.strftime("%Y-%m-%d"))
            if not df_hist.empty and "Close" in df_hist.columns:
                df_hist.index = pd.to_datetime(df_hist.index).date
                df_filtered = df_hist[df_hist.index <= base_date]
                if not df_filtered.empty:
                    close_price = float(df_filtered["Close"].iloc[-1])
                    if close_price > 0:
                        return round(close_price, 2)
        except Exception:
            pass

        return None

    def get_stock_future_curve(self, ticker: str, base_date: any = None, spot_price: float = None, 
                               data: any = None, futures_data: any = None, allow_download: bool = None, **kwargs) -> tuple:
        """
        Constrói o par de curvas para um futuro de ação específico (ex: 'PETRP' ou 'VALEO'):
        Retorna (futures_curve, convenience_yield_curve).
        
        Suporta passagem de contratos externos via `data` ou `futures_data`, preço spot via `spot_price`,
        e curva pré via `pre_curve` ou `pre_data`.
        """
        if allow_download is None:
            allow_download = self.allow_download

        if data is None and futures_data is not None:
            data = futures_data
            
        if isinstance(base_date, (pd.DataFrame, pd.Series, dict, list, tuple)) and data is None:
            data = base_date
            base_date = kwargs.pop('date', None) or kwargs.pop('base_date', None)
            
        if isinstance(base_date, str):
            try:
                base_date = pd.to_datetime(base_date, dayfirst=True if '/' in base_date else False).date()
            except Exception:
                pass
                
        if base_date is None:
            base_date = datetime.date.today()

        df = self._resolve_dataset_input("futures_data", ticker, base_date, allow_download, data)

        if spot_price is None or spot_price <= 0:
            spot_price = self._resolve_stock_spot(ticker, base_date, allow_download, spot_val=kwargs.get("spot"))

        price_col = 'AdjstdQt' if 'AdjstdQt' in df.columns else ('preco' if 'preco' in df.columns else df.columns[-1])
        first_fut = float(df[price_col].iloc[0])
        spot = spot_price if spot_price is not None and spot_price > 0 else first_fut
        exp_col = 'XprtnDt' if 'XprtnDt' in df.columns else ('vencimento' if 'vencimento' in df.columns else df.columns[0])

        fut_curve = self.builder.build_equity_index_curve(
            dataset=df,
            base_date=base_date,
            spot_index=spot,
            config_name="INP",
            price_column=price_col,
            exp_date=exp_col,
            round_decimals=-1
        )

        pre_curve = self._resolve_curve_input("pre_curve", "PRE", base_date, allow_download, kwargs.get("pre_curve"), kwargs.get("pre_data"))
        cy_curve = self.builder.build_convenience_yield_curve(
            spot_price=spot,
            futures_curve=fut_curve,
            pre_curve=pre_curve,
            config_name="CYI"
        )
        return fut_curve, cy_curve

    # =========================================================================
    # CONSULTAS EXTERNAS VIA SGS / OLINDA
    # =========================================================================

    def _fetch_sgs_rate(self, series_id: int, base_date: datetime.date, fallback: float = None) -> float:
        """
        Busca a última taxa disponível na série SGS do BCB usando janela de 7 dias corridos.
        """
        start_str = (base_date - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        end_str = base_date.strftime("%Y-%m-%d")
        try:
            df_bcb = sgs.get({'VAL': series_id}, start=start_str, end=end_str)
            if not df_bcb.empty:
                val = df_bcb['VAL'].dropna()
                if not val.empty:
                    return float(val.iloc[-1]) / 100.0
        except Exception:
            pass
        return fallback

    def _fetch_focus_monthly(self, indicator: str, base_date: datetime.date) -> dict:
        """
        Extrai o dicionário {(ano, mes): taxa_decimal} da Mediana do Relatório Focus Mensal.
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

    def _fetch_currency_spot(self, currency: str, base_date: datetime.date) -> float:
        """
        Obtém a taxa spot Ptax oficial de fechamento para USD, EUR ou JPY via Olinda BCB.
        """
        date_str = base_date.strftime("%m-%d-%Y")
        try:
            import requests
            if currency.upper() == "USD":
                df_ptax = self.scraper.download_bcb_ptax(base_date)
                if not df_ptax.empty and 'cotacaoVenda' in df_ptax.columns:
                    val = df_ptax['cotacaoVenda'].dropna()
                    if not val.empty and float(val.iloc[-1]) > 0:
                        return float(val.iloc[-1])
            else:
                url = f"https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoMoedaDia(moeda=@moeda,dataCotacao=@dataCotacao)?@moeda='{currency.upper()}'&@dataCotacao='{date_str}'&$format=json"
                r = requests.get(url, timeout=5, verify=False)
                if r.status_code == 200:
                    data = r.json()
                    values = data.get('value', [])
                    if values:
                        for row in reversed(values):
                            if 'cotacaoVenda' in row and row['cotacaoVenda'] > 0:
                                return float(row['cotacaoVenda'])
        except Exception:
            pass

        raise DataUnavailableError(
            f"Cotação Ptax oficial de fechamento para a moeda '{currency}' não disponível no Banco Central na data {base_date}."
        )
