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

@pytest.mark.online
@pytest.mark.market_data("dados/b3/BVBG_SPRD_20260904.csv", "dados/b3/BVBG_IN_20260904.csv")
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


@pytest.mark.online
@pytest.mark.market_data(
    "dados/b3/BVBG_SPRD_20260904.csv",
    "dados/b3/BVBG_IN_20260904.csv",
    "dados/b3/taxas_TFP_20260904.csv",
    "dados/b3/taxas_TP_20260904.csv",
)
def test_tbf_and_tr_curves_adherence(curve_manager):
    base_date = datetime.date(2026, 9, 4)
    curve_tbf = curve_manager.get_curve("TBF_PRE", base_date)
    curve_tr = curve_manager.get_curve("TR_PRE", base_date)
    
    df_tfp = pd.read_csv("dados/b3/taxas_TFP_20260904.csv", sep=";")
    df_tp = pd.read_csv("dados/b3/taxas_TP_20260904.csv", sep=";")
    
    calc_tbf = np.array([curve_tbf.get_rate(float(r['day252']), float(r['day360'])) * 100.0 for _, r in df_tfp.iterrows()])
    err_tbf = np.abs(calc_tbf - df_tfp['rate'].values)
    # Verifica aderência da curva TBF x PRE (MAE < 0.01%)
    assert err_tbf.mean() < 0.01
    assert err_tbf.max() < 0.05
    
    calc_tr = np.array([curve_tr.get_rate(float(r['day252']), float(r['day360'])) * 100.0 for _, r in df_tp.iterrows()])
    err_tr = np.abs(calc_tr - df_tp['rate'].values)
    # Verifica aderência da curva TR x PRE (MAE < 0.02%)
    assert err_tr.mean() < 0.02
    assert err_tr.max() < 0.05


@pytest.mark.online
@pytest.mark.market_data("dados/b3/BVBG_SPRD_20260904.csv", "dados/b3/BVBG_IN_20260904.csv")
def test_inflation_curves(curve_manager):
    base_date = datetime.date(2026, 9, 4)
    curve_dic = curve_manager.get_curve("DIC", base_date)
    curve_dpl = curve_manager.get_curve("DPL", base_date)
    curve_dim = curve_manager.get_curve("DIM", base_date)
    
    assert curve_dic is not None
    assert curve_dpl is not None
    assert curve_dim is not None
    
    # Vértice de 1 dia útil (DU = 1) de DIC e DIM coincide com PRE(1)
    curve_pre = curve_manager.get_curve("PRE", base_date)
    v1_pre = curve_pre.get_rate(1.0)
    assert abs(curve_dic.get_rate(1.0) - v1_pre) < 1e-4
    assert abs(curve_dim.get_rate(1.0) - v1_pre) < 1e-4


def test_external_data_support():
    mgr = CurveManager(allow_download=False)
    base_date = datetime.date(2026, 10, 30)
    
    # 1. External DataFrame with DU and rate
    df_custom = pd.DataFrame({
        'DU': [1, 21, 42, 63, 126, 252],
        'rate': [13.75, 13.80, 13.85, 13.90, 14.00, 14.20]
    })
    
    # Formato A: get_curve('DOL', df_custom) - 2º argumento posicional
    c_pos = mgr.get_curve('DOL', df_custom)
    assert abs(c_pos.get_rate(21) - 0.1380) < 1e-4
    assert abs(c_pos.get_rate(252) - 0.1420) < 1e-4
    
    # Formato B: get_curve('DOL', base_date, df_custom)
    c_date_pos = mgr.get_curve('DOL', base_date, df_custom)
    assert abs(c_date_pos.get_rate(21) - 0.1380) < 1e-4
    
    # Formato C: get_curve('DOL', base_date, data=df_custom)
    c_kw = mgr.get_curve('DOL', base_date, data=df_custom)
    assert abs(c_kw.get_rate(63) - 0.1390) < 1e-4

    # 2. External pd.Series
    s_custom = pd.Series([13.75, 13.80, 13.90], index=[1, 21, 63])
    c_series = mgr.get_curve('PRE', data=s_custom)
    assert abs(c_series.get_rate(21) - 0.1380) < 1e-4

    # 3. External dict
    dict_custom = {1: 13.75, 21: 13.80, 42: 13.85}
    c_dict = mgr.get_curve('DIC', data=dict_custom)
    assert abs(c_dict.get_rate(21) - 0.1380) < 1e-4

    # 4. External Flat Curve
    c_flat = mgr.get_curve('TJLP', data=0.075)
    assert abs(c_flat.get_rate(100) - 0.075) < 1e-4

    # 5. External Stock Future Data (100% offline passando pre_curve)
    df_stock = pd.DataFrame({
        'XprtnDt': [datetime.date(2026, 11, 16), datetime.date(2026, 12, 18)],
        'AdjstdQt': [53.10, 53.60]
    })
    fut_curve, cy_curve = mgr.get_stock_future_curve(
        'PETRP', base_date=base_date, spot_price=52.71, data=df_stock, pre_curve=c_series
    )
    assert fut_curve is not None
    assert cy_curve is not None
    assert fut_curve.get_rate(1.0) == 52.71
    # Verifica extrapolação Flat (Fim) da curva de Convenience Yield após o último vencimento (DU=33)
    assert abs(cy_curve.get_rate(33.0) - cy_curve.get_rate(34.0)) < 1e-8
    assert abs(cy_curve.get_rate(33.0) - cy_curve.get_rate(50.0)) < 1e-8


def test_data_unavailable_error(tmp_path):
    from bz_pricing.curves.manager import DataUnavailableError
    mgr = CurveManager(
        loader=B3DataLoader(data_dir=str(tmp_path)),
        builder=B3CurveBuilder(),
        allow_download=False,
    )
    # Data no passado remoto ausente localmente em modo offline
    ancient_date = datetime.date(1991, 1, 1)
    
    with pytest.raises(DataUnavailableError) as exc_info:
        mgr.get_curve('DOL', ancient_date)
    
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

@pytest.mark.online
@pytest.mark.market_data("dados/b3")
def test_provider_knots_and_interpolation(curve_manager):
    """
    Testa se para curvas de swaps offshore como ZUS e SDE,
    o gerenciador extrai apenas os nós base de mudança de inclinação
    e interpola os demais vértices por Linear 360.
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


@pytest.mark.online
@pytest.mark.market_data(
    "dados/b3/BVBG_SPRD_20261030.csv",
    "dados/b3/BVBG_IN_20261030.csv",
    "dados/b3/taxas_PRE_20261030.csv",
)
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
    # No patamar [2..10], a taxa em DU=5 reflete o degrau de inflação acumulada de DU=10
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

    # 5. DOC via vértices DDI + dol_cl + ptax_tm1 + casado_1d + cdi_rate
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

    # 6. EUC via DOC + SDE
    sde_dict = {30: -1.45, 90: -1.55, 180: -1.65, 360: -1.75}
    c_euc = mgr.get_curve('EUC', base_date, doc_curve=c_doc, sde_data=sde_dict)
    assert c_euc.get_rate(126.0, 182.0) < c_doc.get_rate(126.0, 182.0)

    # 7. PTX via PRE + DOC + spot_rate + spot_rate_tm1
    c_ptx = mgr.get_curve(
        'PTX', base_date,
        pre_curve=c_pre,
        foreign_curve=c_doc,
        spot_rate=5.7520,
        spot_rate_tm1=5.7610
    )
    assert c_ptx.get_rate(252.0, 365.0) > 5.7520

    # 8. INP via spot_price + futuros IND + dividend yield CYI
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

    # 9. JPY via PRE + YCL + spot_rate e YCL via DOC + SYD
    syd_dict = {30: -4.10, 90: -4.20, 180: -4.30, 360: -4.40}
    c_ycl = mgr.get_curve('YCL', base_date, doc_curve=c_doc, syd_data=syd_dict)
    c_jpy = mgr.get_curve('JPY', base_date, pre_curve=c_pre, foreign_curve=c_ycl, spot_rate=0.0385)
    assert c_jpy.get_rate(252.0, 365.0) > 0.0385

    # 10. ZUS / ZMX / ZEU via fatores de desconto por prazo DC
    fd_dict = {30: 0.9965, 90: 0.9895, 180: 0.9790, 360: 0.9585}
    c_zus_fd = mgr.get_curve('ZUS', base_date, discount_factors=fd_dict)
    expected_90 = (1.0 / 0.9895 - 1.0) * (360.0 / 90.0)
    assert abs(c_zus_fd.get_rate(90.0) - expected_90) < 1e-4


def test_offline_calendars_ois_schedule_configs_and_provider_guard():
    import math
    from bz_pricing.core.calendars import get_calendar
    from bz_pricing.curves.configs import CURVE_CONFIGS
    from bz_pricing.curves.offshore import (
        _coupon_schedule_dc,
        bootstrap_ois_curve,
        bootstrap_treasury_curve,
    )

    # O1 / E2: calendários embutidos offline e custom holidays sem mutar singleton B3
    cal_b3 = get_calendar("B3", allow_download=False)
    cal_us = get_calendar("US", allow_download=False)
    cal_tgt = get_calendar("TARGET", allow_download=False)
    cal_uk = get_calendar("UK", allow_download=False)
    cal_act = get_calendar("ACTUAL", allow_download=False)
    assert cal_b3 is not None and cal_us is not None and cal_tgt is not None and cal_uk is not None and cal_act is not None

    # 2026-11-02 (Finados BR) é feriado na B3, mas dia útil nos EUA; 2026-07-03 (obs. 4 de Julho) é feriado US
    assert not cal_b3.isbizday(datetime.date(2026, 11, 2))
    assert cal_us.isbizday(datetime.date(2026, 11, 2))
    assert not cal_us.isbizday(datetime.date(2026, 7, 3))
    assert not cal_tgt.isbizday(datetime.date(2026, 12, 26))
    assert not cal_uk.isbizday(datetime.date(2026, 8, 31))
    assert cal_act.bizdays(datetime.date(2026, 10, 30), datetime.date(2026, 11, 2)) == 3

    du_std = cal_b3.bizdays(datetime.date(2026, 10, 30), datetime.date(2026, 12, 1))
    cal_custom = get_calendar("B3", holidays=["2026-11-16"])
    assert cal_custom.bizdays(datetime.date(2026, 10, 30), datetime.date(2026, 12, 1)) == du_std - 1
    assert cal_b3.bizdays(datetime.date(2026, 10, 30), datetime.date(2026, 12, 1)) == du_std

    # E1: espaçamento anual civil (365, 730, 1095 DC) sob ACT/360 em OIS e Treasury reproduz PV par = 1.0
    ois_cal_dcs = [90.0, 180.0, 365.0, 730.0, 1095.0, 1825.0]
    ois_cal_pars = [0.0430, 0.0438, 0.0446, 0.0458, 0.0467, 0.0480]
    ois_crv = bootstrap_ois_curve(
        ois_cal_dcs, ois_cal_pars, convention="ACT/360", payment_freq_months=12, method="monotone_convex"
    )
    for dc_m, y_par in zip(ois_cal_dcs, ois_cal_pars):
        pts = _coupon_schedule_dc(dc_m, 1.0, 360.0)
        pv = 0.0
        prev = 0.0
        for j, pt in enumerate(pts):
            df = math.exp(-ois_crv.get_rate(pt) * (pt / 360.0))
            amt = y_par * ((pt - prev) / 360.0) + (1.0 if j == len(pts) - 1 else 0.0)
            pv += amt * df
            prev = pt
        assert abs(pv - 1.0) < 1e-11

    ust_cal_dcs = [90.0, 182.0, 365.0, 730.0, 1095.0]
    ust_cal_pars = [0.0440, 0.0446, 0.0454, 0.0462, 0.0470]
    ust_crv = bootstrap_treasury_curve(ust_cal_dcs, ust_cal_pars, coupon_freq=2, day_count_basis=365.25, method="monotone_convex")
    assert ust_crv.get_rate(730.0) > 0.0

    # E4: configuração de moedas offshore SGP (SONIA), SYD (TONA) e TONA_OIS
    assert CURVE_CONFIGS["SGP"]["curve_a"] == "SONIA_OIS"
    assert CURVE_CONFIGS["SYD"]["curve_a"] == "TONA_OIS"
    assert "TONA_OIS" in CURVE_CONFIGS and CURVE_CONFIGS["TONA_OIS"]["convention"] == "ACT/365"

    # O4: _extract_provider_vertices não toca self.loader quando allow_download=False
    class _BrokenLoader:
        data_dir = "dados/b3"
        def get_curve_instruments(self, *args, **kwargs):
            raise AssertionError("Não deve consultar loader com allow_download=False")

    mgr_off = CurveManager(loader=_BrokenLoader(), builder=B3CurveBuilder(), allow_download=False)
    df_many = pd.DataFrame({
        "day252": np.arange(1, 70, dtype=int) * 5,
        "day360": np.arange(1, 70, dtype=int) * 7,
        "rate": np.linspace(12.5, 13.8, 69),
    })
    for c_name in ("PRE", "DAP", "DOC"):
        crv_o = mgr_off.get_curve(c_name, datetime.date(2026, 10, 30), data=df_many, allow_download=False)
        assert crv_o.get_rate(25.0, 35.0) > 0.0


def test_bloomberg_convenience_yield_equity_discrete_and_seasonality():
    import math
    from bz_pricing.core.curves import BloombergConvenienceYieldCurve, FlatCurve

    # 1. Tabela XYZ (r=0): alocação de incrementos D_pv nas datas ex-dividendo discretas
    spot = 100.0
    r0_curve = FlatCurve(0.0)
    cy_xyz = BloombergConvenienceYieldCurve(
        spot_price=spot,
        pre_curve=r0_curve,
        t_nodes=[63.0, 126.0],
        fwd_nodes=[99.0, 97.5],
        ex_div_times=[0.10, 0.20, 0.40],
        ex_div_weights=[1.0, 1.0, 1.0],
    )
    assert abs(cy_xyz.forward_price(63.0) - 99.0) < 1e-10
    assert abs(cy_xyz.forward_price(126.0) - 97.5) < 1e-10
    sched = cy_xyz.ex_dividend_schedule()
    assert len(sched) == 3
    assert abs(sched[0]["cash_dividend"] - 0.5) < 1e-10
    assert abs(sched[1]["cash_dividend"] - 0.5) < 1e-10
    assert abs(sched[2]["cash_dividend"] - 1.5) < 1e-10
    # Preço a termo cai apenas na data ex-dividendo quando r=0
    assert abs(cy_xyz.forward_price(0.05 * 252.0) - 100.0) < 1e-10
    assert abs(cy_xyz.forward_price(0.15 * 252.0) - 99.5) < 1e-10

    # 2. CDI positivo (14,5% a.a. exponencial 252): capitalização exata de D_pv
    pre_cdi = FlatCurve(0.145)
    r_c = math.log(1.145)
    cy_cdi = BloombergConvenienceYieldCurve(
        spot_price=spot,
        pre_curve=pre_cdi,
        t_nodes=[63.0, 126.0],
        fwd_nodes=[101.0, 103.0],
        ex_div_times=[0.10, 0.20, 0.40],
        ex_div_weights=[1.0, 1.0, 1.0],
    )
    assert abs(cy_cdi.forward_price(63.0) - 101.0) < 1e-9
    assert abs(cy_cdi.forward_price(126.0) - 103.0) < 1e-9
    pv_expected = spot - 101.0 * math.exp(-r_c * 0.25)
    assert abs(cy_cdi.cash_dividend_pv(63.0) - pv_expected) < 1e-9
    sched_cdi = cy_cdi.ex_dividend_schedule()
    pv_alloc_1 = sum(item["cash_dividend"] * math.exp(-r_c * item["t_years"]) for item in sched_cdi[:2])
    assert abs(pv_alloc_1 - pv_expected) < 1e-9

    # 3. Sazonalidade mensal em 12 faixas + extrapolação >= 1 ano + dividendos em caixa até 10 anos
    seas = np.array([0.05] * 3 + [0.45] + [0.05] * 8, dtype=np.float64)
    cy_seas = BloombergConvenienceYieldCurve(
        spot_price=100.0,
        pre_curve=pre_cdi,
        t_nodes=[63.0, 252.0, 504.0],
        fwd_nodes=[102.0, 108.0, 117.0],
        seasonality_weights=seas,
    )
    # Incremento de D_pv em [0.25, 0.35] concentra a maior parte no mês 4 (peso 0.45 vs 0.05)
    dpv_before = cy_seas.cash_dividend_pv(0.25 * 252.0)
    dpv_after_m4 = cy_seas.cash_dividend_pv((4.0 / 12.0) * 252.0)
    dpv_end = cy_seas.cash_dividend_pv(1.0 * 252.0)
    assert (dpv_after_m4 - dpv_before) > 0.35 * (dpv_end - dpv_before)

    # Extrapolação longa sobre janela >= 1 ano (252 a 504 DU = 1Y..2Y)
    tau1, tau2 = 1.0, 2.0
    yc1 = cy_seas.y_cont_nodes[1]
    yc2 = cy_seas.y_cont_nodes[2]
    assert abs(cy_seas.long_fwd_yield - ((yc2 * tau2 - yc1 * tau1) / (tau2 - tau1))) < 1e-12

    ext_divs = cy_seas.extrapolated_discrete_dividends(div_times=[0.5, 2.0, 5.0, 10.0, 12.0], cash_cutoff_years=10.0)
    assert len(ext_divs["discrete_cash_dividends"]) == 4
    assert abs(ext_divs["residual_long_yield"] - cy_seas.long_fwd_yield) < 1e-12

    # Aumento de série curta (< 1 ano) via augment_div_pv
    cy_short = BloombergConvenienceYieldCurve(
        spot_price=100.0,
        pre_curve=FlatCurve(0.10),
        t_nodes=[63.0, 126.0],
        fwd_nodes=[101.0, 102.0],
        augment_div_pv=2.5,
        augment_horizon=1.0,
    )
    assert cy_short.long_fwd_yield > 0.02


def test_bloomberg_convenience_yield_index_borrow_and_offshore():
    import math
    from bz_pricing.core.curves import BloombergConvenienceYieldCurve, FlatCurve, SimpleCurve
    from bz_pricing.core.interpolators import LogLinearDiscountInterpolator

    # 1. Índice com sazonalidade em 52 semanas e integração analítica exata em Cython
    seas_52 = np.ones(52, dtype=np.float64)
    seas_52[10:20] = 3.0
    pre_252 = FlatCurve(0.12)
    cy_idx = BloombergConvenienceYieldCurve(
        spot_price=1000.0,
        pre_curve=pre_252,
        t_nodes=[21.0, 126.0, 252.0],
        fwd_nodes=[1005.0, 1030.0, 1062.0],
        is_index=True,
        seasonality_weights=seas_52,
    )
    assert abs(cy_idx.forward_price(126.0) - 1030.0) < 1e-9
    y_slope_in_season = cy_idx.get_rate(85.0)
    assert y_slope_in_season > 0.0

    # 2. Recuperação analítica exata da taxa de aluguel (borrow_cost_rate)
    spot = 100.0
    r_c = math.log(1.12)
    t_decl = [0.25, 0.75]
    a_decl = [1.50, 2.00]
    pv_decl = sum(a * math.exp(-r_c * t) for t, a in zip(t_decl, a_decl))
    b_true_cont = 0.035
    fwd_1y = (spot - pv_decl) * math.exp((r_c - b_true_cont) * 1.0)
    cy_btc = BloombergConvenienceYieldCurve(
        spot_price=spot,
        pre_curve=pre_252,
        t_nodes=[252.0],
        fwd_nodes=[fwd_1y],
        declared_div_times=t_decl,
        declared_div_amounts=a_decl,
        compounding="exponential",
    )
    expected_btc_exp = math.exp(b_true_cont) - 1.0
    assert abs(cy_btc.borrow_cost_rate(252.0) - expected_btc_exp) < 1e-12

    # 3. Modo offshore ACT/365 contínuo (multimercado sem 252 hardcoded)
    pre_off = SimpleCurve(
        np.array([30.0, 180.0, 365.0], dtype=np.float64),
        np.array([0.045, 0.045, 0.045], dtype=np.float64),
        LogLinearDiscountInterpolator(0, 0, 365.0, "continuous"),
    )
    fwd_off_1y = 100.0 * math.exp((0.045 - 0.02) * 1.0)
    cy_off = BloombergConvenienceYieldCurve(
        spot_price=100.0,
        pre_curve=pre_off,
        t_nodes=[182.5, 365.0],
        fwd_nodes=[100.0 * math.exp((0.045 - 0.02) * 0.5), fwd_off_1y],
        day_count_basis=365.0,
        compounding="continuous",
    )
    assert abs(cy_off.get_rate(365.0) - 0.02) < 1e-10
    assert abs(cy_off.forward_price(365.0) - fwd_off_1y) < 1e-9

    # 4. Decomposição híbrida a partir da curva reconstrói o forward terminal
    bd = cy_btc.blended_decomposition(252.0, div_times=[0.25, 0.75])
    f_cur, t_prev = 100.0, 0.0
    for ti, dci, ypi in zip(bd["times"], bd["d_cash"], bd["y_prop"]):
        f_cur = f_cur * math.exp((r_c - bd["q"]) * (ti - t_prev)) * (1.0 - ypi) - dci
        t_prev = ti
    assert abs(f_cur * math.exp((r_c - bd["q"]) * (1.0 - t_prev)) - fwd_1y) < 1e-8


def test_bloomberg_convenience_yield_from_options_and_manager():
    import math
    from bz_pricing.core.calendars import get_calendar
    from bz_pricing.core.curves import BloombergConvenienceYieldCurve, FlatCurve
    from bz_pricing.core.options_core import black76_price_greeks, crr_american_discrete_div_price_greeks
    from bz_pricing.volatility import KalmanDividendFilter

    base_dt = datetime.date(2026, 10, 30)
    pre_crv = FlatCurve(0.12)
    r_c = math.log(1.12)
    spot = 100.0

    # 1. from_options_chain europeia e americana com dividendos discretos + Kalman
    rows_eu, rows_am = [], []
    dt_d = np.array([0.20], dtype=np.float64)
    da_d = np.array([2.00], dtype=np.float64)
    kfs = {}
    for du_v, q_v in [(63, 0.03), (126, 0.035)]:
        t_v = du_v / 252.0
        fwd_eu = spot * math.exp((r_c - q_v) * t_v)
        fwd_am = max(1e-6, spot - 2.00 * math.exp(-r_c * 0.20) if t_v >= 0.20 else spot) * math.exp((r_c - 0.01) * t_v)
        kfs[du_v] = KalmanDividendFilter(alpha_over_sigma=1.0)
        for k in (90.0, 95.0, 100.0, 105.0, 110.0):
            for ic in (True, False):
                ot = "CALL" if ic else "PUT"
                p_eu = black76_price_greeks(fwd_eu, k, r_c, t_v, 0.25, ic)[0]
                rows_eu.append({"DU": du_v, "strike": k, "option_type": ot, "bid": max(0.002, p_eu - 0.03), "ask": p_eu + 0.03, "mid": p_eu})
                p_am = crr_american_discrete_div_price_greeks(spot, k, r_c, 0.01, t_v, 0.25, ic, 120, dt_d, da_d, None, None)[0]
                rows_am.append({"DU": du_v, "strike": k, "option_type": ot, "bid": max(0.002, p_am - 0.03), "ask": p_am + 0.03, "mid": p_am})

    cy_opt_eu = BloombergConvenienceYieldCurve.from_options_chain(
        pd.DataFrame(rows_eu), spot_price=spot, pre_curve=pre_crv, du_col="DU", exercise="EUROPEAN", kalman_filters=kfs
    )
    assert abs(cy_opt_eu.forward_price(63.0) - spot * math.exp((r_c - 0.03) * 0.25)) < 1e-4

    cy_opt_am = BloombergConvenienceYieldCurve.from_options_chain(
        pd.DataFrame(rows_am), spot_price=spot, pre_curve=pre_crv, du_col="DU", exercise="AMERICAN", discrete_dividends=[(0.20, 2.00)]
    )
    assert abs(cy_opt_am.forward_price(126.0) - (spot - 2.00 * math.exp(-r_c * 0.20)) * math.exp((r_c - 0.01) * 0.5)) < 0.15

    # 2. from_futures_and_options prioriza pilares de futuros sobre opções coincidentes
    cy_fused = BloombergConvenienceYieldCurve.from_futures_and_options(
        spot_price=spot,
        futures_curve_or_data={63.0: 102.5},
        df_options=pd.DataFrame(rows_eu),
        pre_curve=pre_crv,
        du_col="DU",
    )
    assert abs(cy_fused.forward_price(63.0) - 102.5) < 1e-9
    assert cy_fused.diagnostics["sources"][63.0] == "futures"
    assert cy_fused.diagnostics["sources"][126.0] == "option_implied"

    # 3. Integração B3CurveBuilder & CurveManager (cy_model='b3' preservado vs 'bloomberg')
    mgr = CurveManager(allow_download=False)
    cal = get_calendar("B3")
    du1, du2 = 63, 126
    exp1 = cal.offset(base_dt, du1)
    exp2 = cal.offset(base_dt, du2)
    df_fut = pd.DataFrame({"XprtnDt": [exp1.strftime("%Y-%m-%d"), exp2.strftime("%Y-%m-%d")], "AdjstdQt": [102.1, 104.5]})

    _, cy_b3 = mgr.get_stock_future_curve("PETRP", base_dt, data=df_fut, spot_price=100.0, pre_curve=pre_crv, cy_model="b3")
    _, cy_bbg_m = mgr.get_stock_future_curve("PETRP", base_dt, data=df_fut, spot_price=100.0, pre_curve=pre_crv, cy_model="bloomberg", ex_div_times=[0.15, 0.35])
    assert type(cy_b3).__name__ == "ConvenienceYieldCurve"
    assert isinstance(cy_bbg_m, BloombergConvenienceYieldCurve)
    assert abs(cy_bbg_m.forward_price(du2) - 104.5) < 1e-9

    cy_mgr_imp = mgr.get_implied_convenience_yield_curve(
        base_dt, spot_price=spot, pre_curve=pre_crv, options_data=pd.DataFrame(rows_eu), futures_data={63.0: 102.5}, du_col="DU"
    )
    assert abs(cy_mgr_imp.forward_price(63.0) - 102.5) < 1e-9







