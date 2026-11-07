import datetime
import numpy as np
import pandas as pd
import pytest

from bz_pricing.curves.manager import CurveManager
from bz_pricing.curves.builder import B3CurveBuilder
from bz_pricing.data.loaders import B3DataLoader

@pytest.fixture(scope="module")
def curve_manager():
    loader = B3DataLoader(data_dir="dados/b3")
    builder = B3CurveBuilder()
    return CurveManager(loader=loader, builder=builder, allow_download=True)

def test_pre_and_selic_curves(curve_manager):
    base_date = datetime.date(2026, 9, 4)
    curve_pre = curve_manager.get_curve("PRE", base_date)
    curve_slp = curve_manager.get_curve("SELIC_PRE", base_date)
    
    assert curve_pre is not None
    assert curve_slp is not None
    
    # Vertice 1 (1 DU = 4 DC no fim de semana prolongado de 7 de setembro)
    v1_pre = curve_pre.get_rate(1.0, 4.0) * 100.0
    v1_slp = curve_slp.get_rate(1.0, 4.0) * 100.0
    assert abs(v1_pre - 13.90) < 1e-4
    assert abs(v1_slp - 13.90) < 1e-4

def test_tbf_and_tr_curves_adherence(curve_manager):
    base_date = datetime.date(2026, 9, 4)
    curve_tbf = curve_manager.get_curve("TBF_PRE", base_date)
    curve_tr = curve_manager.get_curve("TR_PRE", base_date)
    
    df_tfp = pd.read_csv("dados/b3/taxas_TFP_20260904.csv", sep=";")
    df_tp = pd.read_csv("dados/b3/taxas_TP_20260904.csv", sep=";")
    
    calc_tbf = np.array([curve_tbf.get_rate(float(r['day252']), float(r['day360'])) * 100.0 for _, r in df_tfp.iterrows()])
    err_tbf = np.abs(calc_tbf - df_tfp['rate'].values)
    # Aderencia extrema ao Manual de Curvas B3 Secao 2.4 (MAE < 0.01%)
    assert err_tbf.mean() < 0.01
    assert err_tbf.max() < 0.05
    
    calc_tr = np.array([curve_tr.get_rate(float(r['day252']), float(r['day360'])) * 100.0 for _, r in df_tp.iterrows()])
    err_tr = np.abs(calc_tr - df_tp['rate'].values)
    # Aderencia extrema a Res. CMN 5.124 e Secao 2.5 (MAE < 0.02%)
    assert err_tr.mean() < 0.02
    assert err_tr.max() < 0.05

def test_inflation_curves(curve_manager):
    base_date = datetime.date(2026, 9, 4)
    curve_dic = curve_manager.get_curve("DIC", base_date)
    curve_dpl = curve_manager.get_curve("DPL", base_date)
    curve_dim = curve_manager.get_curve("DIM", base_date)
    
    assert curve_dic is not None
    assert curve_dpl is not None
    assert curve_dim is not None
    
    # Vertice 1 (1 DU) de DIC e DIM e PRE coincidem conforme Secao 3.1 e 3.3
    curve_pre = curve_manager.get_curve("PRE", base_date)
    v1_pre = curve_pre.get_rate(1.0)
    assert abs(curve_dic.get_rate(1.0) - v1_pre) < 1e-4
    assert abs(curve_dim.get_rate(1.0) - v1_pre) < 1e-4

def test_external_data_support(curve_manager):
    from bz_pricing.curves.manager import DataUnavailableError
    base_date = datetime.date(2026, 10, 30)
    
    # 1. External DataFrame with DU and rate
    df_custom = pd.DataFrame({
        'DU': [1, 21, 42, 63, 126, 252],
        'rate': [13.75, 13.80, 13.85, 13.90, 14.00, 14.20]
    })
    
    # Formato A: get_curve('DOL', df_custom) - 2º argumento posicional
    c_pos = curve_manager.get_curve('DOL', df_custom)
    assert abs(c_pos.get_rate(21) - 0.1380) < 1e-4
    assert abs(c_pos.get_rate(252) - 0.1420) < 1e-4
    
    # Formato B: get_curve('DOL', base_date, df_custom)
    c_date_pos = curve_manager.get_curve('DOL', base_date, df_custom)
    assert abs(c_date_pos.get_rate(21) - 0.1380) < 1e-4
    
    # Formato C: get_curve('DOL', base_date, data=df_custom)
    c_kw = curve_manager.get_curve('DOL', base_date, data=df_custom)
    assert abs(c_kw.get_rate(63) - 0.1390) < 1e-4

    # 2. External pd.Series
    s_custom = pd.Series([13.75, 13.80, 13.90], index=[1, 21, 63])
    c_series = curve_manager.get_curve('PRE', data=s_custom)
    assert abs(c_series.get_rate(21) - 0.1380) < 1e-4

    # 3. External dict
    dict_custom = {1: 13.75, 21: 13.80, 42: 13.85}
    c_dict = curve_manager.get_curve('DIC', data=dict_custom)
    assert abs(c_dict.get_rate(21) - 0.1380) < 1e-4

    # 4. External Flat Curve
    c_flat = curve_manager.get_curve('TJLP', data=0.075)
    assert abs(c_flat.get_rate(100) - 0.075) < 1e-4

    # 5. External Stock Future Data
    df_stock = pd.DataFrame({
        'XprtnDt': [datetime.date(2026, 11, 16), datetime.date(2026, 12, 18)],
        'AdjstdQt': [53.10, 53.60]
    })
    fut_curve, cy_curve = curve_manager.get_stock_future_curve('PETRP', base_date=base_date, spot_price=52.71, data=df_stock)
    assert fut_curve is not None
    assert cy_curve is not None
    assert fut_curve.get_rate(1.0) == 52.71
    # Verifica extrapolação Flat (Fim) da curva de Convenience Yield após o último vencimento (DU=33)
    assert abs(cy_curve.get_rate(33.0) - cy_curve.get_rate(34.0)) < 1e-8
    assert abs(cy_curve.get_rate(33.0) - cy_curve.get_rate(50.0)) < 1e-8

def test_data_unavailable_error(curve_manager):
    from bz_pricing.curves.manager import DataUnavailableError
    # Data no passado remoto que não existe localmente nem no histórico disponível da B3
    ancient_date = datetime.date(1991, 1, 1)
    
    with pytest.raises(DataUnavailableError) as exc_info:
        curve_manager.get_curve('DOL', ancient_date)
    
    msg = str(exc_info.value)
    assert "DOL" in msg

def test_allow_download_and_partial_inputs():
    from bz_pricing.curves.manager import CurveManager, DataUnavailableError
    mgr = CurveManager() # default allow_download=False
    assert mgr.allow_download is False
    base_date = datetime.date(2026, 9, 4)

    # 1. Sem dados e allow_download=False -> Erro
    with pytest.raises(DataUnavailableError) as exc:
        mgr.get_curve('PRE', base_date)
    assert "allow_download=False" in str(exc.value)

    # 2. DPL com apenas DAP fornecido e allow_download=False -> Erro indicando pre_curve faltante
    df_dap = pd.DataFrame({
        'XprtnDt': [datetime.date(2027, 8, 15), datetime.date(2028, 8, 15)],
        'AdjstdQtTax': [6.45, 6.55]
    })
    with pytest.raises(DataUnavailableError) as exc_dpl:
        mgr.get_curve('DPL', base_date, dap_df=df_dap)
    assert "pre_curve" in str(exc_dpl.value)
    assert "allow_download=False" in str(exc_dpl.value)

    # 3. DPL com PRE e DAP fornecidos externamente -> Sucesso 100% offline
    df_pre = pd.DataFrame({
        'DU': [1, 21, 252, 504],
        'rate': [13.75, 13.80, 14.00, 14.10]
    })
    c_dpl = mgr.get_curve('DPL', base_date, pre_data=df_pre, dap_df=df_dap, eipca_0=0.002)
    assert c_dpl is not None
    assert c_dpl.get_rate(252) > 0.0

    # 4. DIC com PRE e Focus fornecidos externamente -> Sucesso 100% offline
    c_dic = mgr.get_curve('DIC', base_date, pre_data=df_pre, focus_monthly={(2026, 9): 0.002, (2026, 10): 0.003})
    assert c_dic is not None
    assert c_dic.get_rate(21) > 0.0

def test_provider_knots_and_interpolation(curve_manager):
    """
    Testa se para curvas como ZUS e SDE, onde a fonte é a tabela da B3 (taxas_*.csv),
    o manager captura apenas os nós fornecidos pelo provedor (Refinitiv conforme Tabela 1),
    e interpola os demais vértices com alta precisão.
    """
    base_date = datetime.date(2026, 10, 30)
    curve_zus = curve_manager.get_curve('ZUS', base_date)
    assert curve_zus is not None
    
    # Verifica que filtrou os 278 vértices diários para os ~73 nós originais do provedor
    assert len(curve_zus.t_nodes) < 100
    assert len(curve_zus.t_nodes) == len(curve_zus.y_nodes)
    
    # Verifica interpolação entre nós (ex: DC=45, intermediário entre 32 e 47/66)
    r45 = curve_zus.get_rate(45.0)
    assert 0.0420 < r45 < 0.0435

def test_all_b3_curves_vertex_adherence_20261030(curve_manager):
    """
    Valida que todas as curvas da análise (PRE, DIC, DPL, DIM, DOL, DOC, EUC, SDE, PTX, EUR, JPY, INP, BRP)
    são construídas a partir de vértices esparsos e possuem aderência extrema às curvas oficiais da B3.
    """
    base_date = datetime.date(2026, 10, 30)
    loader = curve_manager.loader

    specs = [
        ("PRE", False, 60, 0.005, 0.015),
        ("DIC", False, 160, 0.005, 0.015),
        ("DPL", False, 30, 0.005, 0.015),
        ("DIM", False, 160, 0.005, 0.015),
        ("DOL", False, 60, 0.005, 0.015),
        ("DOC", False, 60, 0.005, 0.015),
        ("EUC", False, 100, 0.005, 0.015),
        ("SDE", False, 80, 0.005, 0.015),
        ("PTX", True, 60, 0.005, 0.015),
        ("EUR", True, 100, 0.005, 0.025),
        ("JPY", True, 70, 0.005, 0.015),
        ("INP", True, 25, 0.01, 0.05),
        ("BRP", True, 10, 0.01, 0.05),
    ]

    for name, is_price, max_knots, max_mae, max_diff in specs:
        c = curve_manager.get_curve(name, base_date)
        df_b3 = loader.load_taxas_referenciais(name, base_date)
        mult = 1.0 if is_price else 100.0
        if hasattr(c, "t_nodes") and c.t_nodes is not None:
            assert len(c.t_nodes) <= max_knots, f"{name} tem {len(c.t_nodes)} nós (limite {max_knots})"
        diffs = [
            abs(c.get_rate(float(r["day252"]), float(r["day360"])) * mult - float(r["rate"]))
            for _, r in df_b3.iterrows()
        ]
        assert np.mean(diffs) <= max_mae, f"{name} MAE={np.mean(diffs):.5f} excedeu {max_mae}"
        assert np.max(diffs) <= max_diff, f"{name} MaxDiff={np.max(diffs):.5f} excedeu {max_diff}"


def test_user_component_vertices_offline():
    """
    Testa a construção 100% offline (allow_download=False) passando os pontos
    dos componentes para DIC, DPL, DOC, EUC, PTX e INP.
    """
    mgr = CurveManager(allow_download=False)
    base_date = datetime.date(2026, 10, 30)

    # 1. Curva PRE base (por vértices)
    pre_df = pd.DataFrame({
        'DU': [1, 10, 21, 31, 52, 126, 252, 504],
        'rate': [13.65, 13.70, 13.72, 13.75, 13.80, 13.90, 14.00, 14.10]
    })
    c_pre = mgr.get_curve('PRE', base_date, data=pre_df)

    # 2. DIC via informantes com 'rate' (CpInformantes nos vértices de divulgação + longo prazo)
    inf_rate = pd.DataFrame({
        'DU': [10, 31, 52, 252, 504],
        'rate': [10.85, 8.83, 8.95, 7.80, 7.65]
    })
    c_dic_rate = mgr.get_curve('DIC', base_date, pre_curve=c_pre, informants_df=inf_rate)
    assert abs(c_dic_rate.get_rate(1.0) - c_pre.get_rate(1.0)) < 1e-4
    # No patamar [2..10], a taxa em DU=5 reflete o degrau de inflação acumulada de DU=10 (Eq. 3.1.3)
    assert c_dic_rate.get_rate(5.0) != c_dic_rate.get_rate(10.0)
    assert abs(c_dic_rate.get_rate(10.0) - 0.1085) < 1e-3

    # 3. DIC via informantes com 'e_inf' (expectativa acumulada de inflação em %)
    inf_einf = pd.DataFrame({
        'DU': [10, 31, 52, 252],
        'e_inf': [0.45, 0.92, 1.35, 4.50]
    })
    c_dic_einf = mgr.get_curve('DIC', base_date, pre_curve=c_pre, informants_df=inf_einf)
    assert c_dic_einf.get_rate(10.0) < c_pre.get_rate(10.0)

    # 4. DPL via vértices DAP em DU + eipca_0
    dap_df = pd.DataFrame({
        'DU': [14, 135, 252, 504],
        'rate': [9.62, 6.80, 7.65, 7.80]
    })
    c_dpl = mgr.get_curve('DPL', base_date, pre_curve=c_pre, dap_df=dap_df, eipca_0=0.45)
    assert abs(c_dpl.get_rate(14.0) - 0.0962) < 1e-4
    assert abs(c_dpl.get_rate(252.0) - 0.0765) < 1e-4

    # 5. DOC via vértices DDI + dol_cl + ptax_tm1 + casado_1d + cdi_rate (Eq. 4.5.1 a 4.5.3)
    ddi_df = pd.DataFrame({
        'DU': [1, 21, 42, 126, 252],
        'DC': [4, 31, 62, 182, 365],
        'rate': [-73.16, 1.07, 3.16, 4.85, 5.25]
    })
    c_doc = mgr.get_curve(
        'DOC', base_date,
        ddi_data=ddi_df,
        dol_cl=5.7520,
        ptax_tm1=5.7610,
        casado_1d=1.20,
        cdi_rate=13.65
    )
    assert c_doc.get_rate(1.0, 4.0) > 0.0
    assert c_doc.get_rate(252.0, 365.0) > 0.05

    # 6. EUC via DOC + SDE (Eq. 4.7.1)
    sde_dict = {30: -1.45, 90: -1.55, 180: -1.65, 360: -1.75}
    c_euc = mgr.get_curve('EUC', base_date, doc_curve=c_doc, sde_data=sde_dict)
    assert c_euc.get_rate(126.0, 182.0) < c_doc.get_rate(126.0, 182.0)

    # 7. PTX via PRE + DOC + spot_rate + spot_rate_tm1 (Eq. 4.1.1 / 4.1.2)
    c_ptx = mgr.get_curve(
        'PTX', base_date,
        pre_curve=c_pre,
        foreign_curve=c_doc,
        spot_rate=5.7520,
        spot_rate_tm1=5.7610
    )
    assert c_ptx.get_rate(252.0, 365.0) > 5.7520

    # 8. INP via spot_price + futuros IND + dividend yield CYI (Eq. 5.1.1)
    ind_df = pd.DataFrame({
        'DU': [31, 71, 114],
        'price': [131200.0, 133500.0, 136100.0]
    })
    c_inp = mgr.get_curve(
        'INP', base_date,
        pre_curve=c_pre,
        spot_price=129500.0,
        futures_data=ind_df,
        cy_data={31: 3.10, 71: 3.25, 114: 3.40}
    )
    assert abs(c_inp.get_rate(1.0) - 129500.0) < 1e-4
    assert c_inp.get_rate(50.0) > 129500.0

    # 9. JPY via PRE + YCL + spot_rate (Eq. 4.3.1) e YCL via DOC + SYD (Eq. 4.8.1)
    syd_dict = {30: -4.10, 90: -4.20, 180: -4.30, 360: -4.40}
    c_ycl = mgr.get_curve('YCL', base_date, doc_curve=c_doc, syd_data=syd_dict)
    c_jpy = mgr.get_curve('JPY', base_date, pre_curve=c_pre, foreign_curve=c_ycl, spot_rate=0.0385)
    assert c_jpy.get_rate(252.0, 365.0) > 0.0385

    # 10. ZUS / ZMX / ZEU via fatores de desconto FD_JUR (Seções 2.10 a 2.12 Manual B3)
    fd_dict = {30: 0.9965, 90: 0.9895, 180: 0.9790, 360: 0.9585}
    c_zus_fd = mgr.get_curve('ZUS', base_date, discount_factors=fd_dict)
    expected_90 = (1.0 / 0.9895 - 1.0) * (360.0 / 90.0)
    assert abs(c_zus_fd.get_rate(90.0) - expected_90) < 1e-4






