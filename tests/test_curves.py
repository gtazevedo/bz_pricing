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
    return CurveManager(loader=loader, builder=builder)

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
    # Aderencia extrema a Res. CMN 5.124 e Secao 2.5 (MAE < 0.005%)
    assert err_tr.mean() < 0.005
    assert err_tr.max() < 0.02

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
