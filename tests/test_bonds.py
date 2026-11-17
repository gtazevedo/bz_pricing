import datetime
import math
import numpy as np
import pytest

from bz_pricing.core.curves import FlatCurve, SimpleCurve
from bz_pricing.core.interpolators import FlatForward252Interpolator
from bz_pricing.instruments.tpf import LTN, NTNF, LFT, NTNB, NTNBPrincipal, NTNC
from bz_pricing.curves.builder import B3CurveBuilder
from bz_pricing.curves.manager import CurveManager


def test_ltn_pricing():
    base_date = datetime.date(2026, 10, 30)
    maturity_date = datetime.date(2027, 4, 1)

    # Vértices e curva prefixada
    t_nodes = np.array([1.0, 102.0, 252.0], dtype=np.float64)
    y_nodes = np.array([0.1415, 0.137844, 0.1420], dtype=np.float64)
    curve_pre = SimpleCurve(t_nodes, y_nodes, FlatForward252Interpolator(0, 0))

    ltn = LTN(maturity_date)
    pu = ltn.present_value(base_date, curve_pre)
    dur = ltn.duration(base_date, curve_pre)
    dv01 = ltn.dv01(base_date, curve_pre)

    # PU oficial divulgado pela ANBIMA para LTN 2027-04-01 em 30/10/2026
    pu_oficial = 949.073505
    assert abs(pu - pu_oficial) < 1e-5
    assert dur == 102.0
    assert dv01 > 0.0


def test_ntnf_pricing():
    base_date = datetime.date(2026, 10, 30)
    maturity_date = datetime.date(2027, 1, 1)

    t_nodes = np.array([1.0, 42.0, 252.0], dtype=np.float64)
    y_nodes = np.array([0.1415, 0.135793, 0.1420], dtype=np.float64)
    curve_pre = SimpleCurve(t_nodes, y_nodes, FlatForward252Interpolator(0, 0))

    ntnf = NTNF(maturity_date, coupon_rate=0.10)
    pu = ntnf.present_value(base_date, curve_pre)
    dur = ntnf.duration(base_date, curve_pre)
    mod_dur = ntnf.modified_duration(base_date, curve_pre)
    dv01 = ntnf.dv01(base_date, curve_pre)

    pu_oficial = 1026.785700
    assert abs(pu - pu_oficial) < 1e-5
    assert dur == 42.0
    assert mod_dur > 0.0
    assert dv01 > 0.0


def test_lft_pricing():
    base_date = datetime.date(2026, 10, 30)
    maturity_date = datetime.date(2027, 3, 1)

    # Curva de ágio/deságio
    curve_agio = FlatCurve(0.000232)
    lft = LFT(maturity_date)

    cotacao = lft.cotacao(base_date, curve_agio)
    assert cotacao == 99.9926

    # VNA oficial de 20201.049997 em 30/10/2026
    vna_oficial = 20201.049997
    pu = lft.present_value(base_date, curve_agio, vna=vna_oficial)
    pu_oficial = 20199.555119
    assert abs(pu - pu_oficial) < 1e-5


def test_ntnb_pricing():
    base_date = datetime.date(2026, 10, 30)
    maturity_date = datetime.date(2027, 5, 15)

    # Curva real com taxa indicativa oficial de 4.6629% em DU=133
    t_nodes = np.array([10.0, 133.0, 252.0], dtype=np.float64)
    y_nodes = np.array([0.046629, 0.046629, 0.0500], dtype=np.float64)
    curve_real = SimpleCurve(t_nodes, y_nodes, FlatForward252Interpolator(0, 0))

    ntnb = NTNB(maturity_date, coupon_rate=0.06)
    cotacao = ntnb.cotacao(base_date, curve_real)
    assert cotacao == 103.4603

    vna_oficial = 4771.575461
    pu = ntnb.present_value(base_date, curve_real, vna=vna_oficial)
    pu_oficial = 4936.686286
    assert abs(pu - pu_oficial) < 1e-5

    dur = ntnb.duration(base_date, curve_real)
    mod_dur = ntnb.modified_duration(base_date, curve_real)
    dv01 = ntnb.dv01(base_date, curve_real, vna=vna_oficial)
    assert 120.0 < dur < 133.0
    assert mod_dur > 0.0
    assert dv01 > 0.0


def test_ntnb_principal_pricing():
    base_date = datetime.date(2026, 10, 30)
    maturity_date = datetime.date(2027, 5, 15)

    curve_real = FlatCurve(0.046629)
    ntnb_p = NTNBPrincipal(maturity_date)

    cotacao = ntnb_p.cotacao(base_date, curve_real)
    vna_oficial = 4771.575440
    pu = ntnb_p.present_value(base_date, curve_real, vna=vna_oficial)

    du = ntnb_p.calendar.bizdays(base_date, ntnb_p.maturity_date)
    cot_esperada = math.trunc((100.0 / ((1.0 + 0.046629) ** (du / 252.0))) * 1e4) / 1e4
    pu_esperado = math.trunc((vna_oficial * (cot_esperada / 100.0)) * 1e6) / 1e6

    assert cotacao == cot_esperada
    assert abs(pu - pu_esperado) < 1e-5


def test_ntnc_pricing():
    base_date = datetime.date(2026, 10, 30)
    maturity_date = datetime.date(2031, 1, 1)

    curve_real = FlatCurve(0.077739)
    # Série 2031 possui cupom editalício de 12% a.a. quando coupon_rate é omitido
    ntnc_12 = NTNC(maturity_date)
    assert ntnc_12.coupon_rate == 0.12
    assert ntnc_12.cotacao(base_date, curve_real) == 118.2632
    pu_12 = ntnc_12.present_value(base_date, curve_real, vna=6765.022425)
    assert abs(pu_12 - 8000.532000) < 1e-5

    # Sobrescrita explícita de cupom (ex.: séries antigas de 6% a.a.) continua suportada
    ntnc = NTNC(maturity_date, coupon_rate=0.06)
    cotacao = ntnc.cotacao(base_date, curve_real)
    assert cotacao == 96.1378

    vna_oficial = 8321.942046
    pu = ntnc.present_value(base_date, curve_real, vna=vna_oficial)
    pu_oficial = 8000.532000
    assert abs(pu - pu_oficial) < 1e-5

    dur = ntnc_12.duration(base_date, curve_real)
    mod_dur = ntnc_12.modified_duration(base_date, curve_real)
    dv01 = ntnc_12.dv01(base_date, curve_real, vna=6765.022425)
    assert dur > 0.0
    assert mod_dur > 0.0
    assert dv01 > 0.0


def test_tpf_multi_curve_pricing():
    base_date = datetime.date(2026, 10, 30)
    maturity_date = datetime.date(2027, 5, 15)

    # Objeto 1: Curva de Taxa Real de IPCA para desconto dos cupons e principal
    curve_real = FlatCurve(0.046629)

    # Objeto 2: Curva de Inflação Implícita (DIC) para projetar o VNA a partir do dia 15
    curve_dic = FlatCurve(0.0450)

    ntnb = NTNB(maturity_date, coupon_rate=0.06)

    # Precificação fornecendo ambos os objetos de curvas
    base_vna_date = datetime.date(2026, 9, 15)
    base_vna = 4750.00
    pu_multi = ntnb.present_value(
        base_date,
        real_curve=curve_real,
        inflation_curve=curve_dic,
        base_vna=base_vna,
        base_vna_date=base_vna_date
    )

    assert pu_multi > 0.0
    assert abs(pu_multi - 4941.909732) < 1e-4


def test_curve_rounding_truncation_anbima():
    t_nodes = np.array([1.0, 102.0, 252.0], dtype=np.float64)
    y_nodes = np.array([0.1415, 0.13784429, 0.1420], dtype=np.float64)

    # Curva padrão contínua (round_decimals = -1)
    curve_cont = SimpleCurve(t_nodes, y_nodes, FlatForward252Interpolator(0, 0))
    assert curve_cont.round_decimals == -1

    # Curva parametrizada com truncamento oficial ANBIMA (6 decimais unitárias / 4 decimais percentuais)
    curve_anbima = SimpleCurve(t_nodes, y_nodes, FlatForward252Interpolator(0, 0), round_decimals=6, round_mode='trunc')
    assert curve_anbima.round_decimals == 6
    assert curve_anbima.round_mode == 'trunc'

    rate_cont = curve_cont.get_rate(50)
    rate_anbima = curve_anbima.get_rate(50)
    assert rate_anbima == math.trunc(rate_cont * 1e6) / 1e6

    # LTN precificada com ambas as curvas
    ltn = LTN(datetime.date(2027, 4, 1))
    pu_cont = ltn.present_value(datetime.date(2026, 10, 30), curve_cont)
    pu_anbima = ltn.present_value(datetime.date(2026, 10, 30), curve_anbima)
    assert pu_cont > 0.0
    assert pu_anbima > 0.0


def test_build_from_vertices():
    builder = B3CurveBuilder()
    manager = CurveManager()

    t_nodes = [10, 50, 100]
    y_nodes = [0.10, 0.11, 0.12]

    # Construção direta por vértices
    c1 = builder.build_from_vertices(t_nodes, y_nodes)
    c2 = manager.build_from_vertices(t_nodes, y_nodes, round_decimals=6, round_mode='trunc')

    assert c1.round_decimals == -1
    assert c2.round_decimals == 6
    assert c2.round_mode == 'trunc'
    assert c1.get_rate(25) > 0.0
    assert c2.get_rate(25) > 0.0


def test_bond_cashflow_tables():
    base_date = datetime.date(2026, 10, 30)

    # LTN
    curve_ltn = FlatCurve(0.137844)
    ltn = LTN(datetime.date(2027, 4, 1))
    pu_ltn, df_ltn = ltn.present_value(base_date, curve_ltn, return_cashflows=True)
    assert len(df_ltn) == 1
    assert "data_fluxo" in df_ltn.columns
    assert "valor_presente" in df_ltn.columns
    assert abs(df_ltn["valor_presente"].iloc[0] - pu_ltn) < 1e-5

    # NTN-F
    curve_ntnf = FlatCurve(0.149501)
    ntnf = NTNF(datetime.date(2029, 1, 1), coupon_rate=0.10)
    pu_ntnf, df_ntnf = ntnf.present_value(base_date, curve_ntnf, return_cashflows=True)
    assert len(df_ntnf) == 5
    assert set(["data_fluxo", "du", "tipo", "fluxo_nominal", "taxa", "fator_desconto", "valor_presente"]).issubset(df_ntnf.columns)
    assert abs(df_ntnf["valor_presente"].sum() - pu_ntnf) < 1e-4

    # NTN-B
    curve_ntnb = FlatCurve(0.046629)
    ntnb = NTNB(datetime.date(2027, 5, 15), coupon_rate=0.06)
    pu_ntnb, df_ntnb = ntnb.present_value(base_date, curve_ntnb, vna=4771.575461, return_cashflows=True)
    assert len(df_ntnb) == 2
    assert set(["data_fluxo", "du", "tipo", "fluxo_cotação", "taxa_real", "fator_desconto", "cotação_descontada", "vna", "valor_presente"]).issubset(df_ntnb.columns)
    assert abs(df_ntnb["valor_presente"].sum() - pu_ntnb) < 1e-2


def test_all_anbima_bonds_20261030():
    import os
    import pandas as pd
    from pathlib import Path

    csv_path = Path(__file__).resolve().parents[1] / "dados" / "b3" / "ANBIMA_MS_20261030.csv"
    if not csv_path.exists():
        pytest.skip("Arquivo ANBIMA_MS_20261030.csv não encontrado no repositório.")

    df = pd.read_csv(csv_path, sep="@", encoding="latin1")
    base_date = datetime.date(2026, 10, 30)

    # VNAs oficiais vigentes em 30/10/2026
    vna_lft = 20201.049997
    vna_ntnb = 4771.575461
    vna_ntnc = 6765.022425

    tested_count = 0
    for _, row in df.iterrows():
        tit = str(row["Titulo"]).strip()
        dt_vcto_str = str(int(row["Data Vencimento"]))
        dt_vcto = datetime.date(int(dt_vcto_str[:4]), int(dt_vcto_str[4:6]), int(dt_vcto_str[6:8]))

        tx_str = str(row["Tx. Indicativas"]).replace(".", "").replace(",", ".")
        tx = float(tx_str) / 100.0

        pu_ofic_str = str(row["PU"]).replace(".", "").replace(",", ".")
        pu_ofic = float(pu_ofic_str)

        curve = FlatCurve(tx)

        if tit == "LTN":
            bond = LTN(dt_vcto)
            pu_calc, cf_df = bond.present_value(base_date, curve, return_cashflows=True)
        elif tit == "NTN-F":
            bond = NTNF(dt_vcto, coupon_rate=0.10)
            pu_calc, cf_df = bond.present_value(base_date, curve, return_cashflows=True)
        elif tit == "LFT":
            bond = LFT(dt_vcto)
            pu_calc, cf_df = bond.present_value(base_date, curve, vna=vna_lft, return_cashflows=True)
        elif tit == "NTN-B":
            bond = NTNB(dt_vcto, coupon_rate=0.06)
            pu_calc, cf_df = bond.present_value(base_date, curve, vna=vna_ntnb, return_cashflows=True)
        elif tit == "NTN-C":
            bond = NTNC(dt_vcto)
            pu_calc, cf_df = bond.present_value(base_date, curve, vna=vna_ntnc, return_cashflows=True)
        else:
            continue

        assert abs(pu_calc - pu_ofic) < 0.01, f"Divergência em {tit} {dt_vcto}: calc={pu_calc}, ofic={pu_ofic}"
        assert abs(pu_calc - pu_ofic) < 1e-4, f"Divergência superior a 1e-4 em {tit} {dt_vcto}"
        assert len(cf_df) > 0
        tested_count += 1

    assert tested_count == 50


def test_ntnb_vna_calculation_offline_and_online():
    from bz_pricing.core.vna_engine import VNAEngine, calculate_ntnb_vna, NTNB_BASE_DATE, NTNB_BASE_VALUE

    # 1. Cálculo offline determinístico a partir de série de IPCA mensal (base 15/07/2000 = R$ 1.000,00)
    ipca_hist = {
        "2000-07-01": 0.0161,
        "2000-08-01": 0.0131,
        "2000-09-01": 0.0023,
    }
    eng = VNAEngine.for_ntnb(ipca_history=ipca_hist, allow_download=False)
    vna_15oct = eng.calculate_vna(datetime.date(2000, 10, 15), truncate_monthly=False)
    expected_15oct = math.trunc((NTNB_BASE_VALUE * 1.0161 * 1.0131 * 1.0023) * 1e6) / 1e6
    assert abs(vna_15oct - expected_15oct) < 1e-6

    # Pró-rata DU com ipca_proj explícito entre 15/09/2000 e 15/10/2000
    vna_pro = calculate_ntnb_vna(
        datetime.date(2000, 9, 29),
        ipca_proj=0.0040,
        ipca_history={"2000-07-01": 0.0161, "2000-08-01": 0.0131},
        allow_download=False,
    )
    assert vna_pro > NTNB_BASE_VALUE * 1.0161 * 1.0131

    # 2. NTNB calcula automaticamente o VNA sem exigir parâmetro vna manual (via vna_engine offline)
    eng_2026 = VNAEngine(datetime.date(2026, 9, 15), 4750.0)
    eng_2026.ipca_history = eng_2026.ipca_history.copy()
    import pandas as _pd
    eng_2026.ipca_history.loc[_pd.Timestamp("2026-09-01")] = 0.0044
    ntnb = NTNB(datetime.date(2027, 5, 15), coupon_rate=0.06)
    vna_ntnb = ntnb.calculate_vna(datetime.date(2026, 10, 30), ipca_proj=0.0020, vna_engine=eng_2026)
    assert vna_ntnb > 4750.0

    pu_auto, df_auto = ntnb.present_value(
        datetime.date(2026, 10, 30),
        FlatCurve(0.06),
        ipca_proj=0.0020,
        vna_engine=eng_2026,
        return_cashflows=True,
    )
    assert pu_auto > 0.0
    assert abs(df_auto["vna"].iloc[0] - vna_ntnb) < 1e-6

    # 3. Cálculo online oficial (BCB SGS 433) na data-base 30/10/2026 com projeção de mercado ~0.20%
    vna_2026 = calculate_ntnb_vna(datetime.date(2026, 10, 30), ipca_proj=0.001972)
    assert abs(vna_2026 - 4771.575461) < 0.05


def test_tpf_custom_cashflows():
    import pandas as pd
    base_dt = datetime.date(2026, 10, 30)
    curve = FlatCurve(0.12)

    # Fluxos customizados passados via lista de tuplas (data, valor, tipo)
    custom = [
        (datetime.date(2027, 1, 4), 50.0, "Cupom"),
        (datetime.date(2027, 7, 1), 1050.0, "Cupom + Principal"),
    ]
    ntnf_custom = NTNF(custom_cashflows=custom)
    pu_c, df_c = ntnf_custom.present_value(base_dt, curve, return_cashflows=True)
    assert len(df_c) == 2
    assert abs(pu_c - df_c["valor_presente"].sum()) < 1e-9
    assert ntnf_custom.duration(base_dt, curve) > 0.0
    assert ntnf_custom.modified_duration(base_dt, curve) > 0.0
    assert ntnf_custom.dv01(base_dt, curve) > 0.0

    # Mesmo suporte em LTN (dict) e NTNB (DataFrame com fluxo_cotação + VNA)
    ltn_custom = LTN(custom_cashflows={datetime.date(2027, 7, 1): 1000.0})
    assert abs(ltn_custom.present_value(base_dt, curve) - LTN(datetime.date(2027, 7, 1)).present_value(base_dt, curve)) < 1e-4

    df_nb = pd.DataFrame({
        "data_fluxo": [datetime.date(2027, 5, 15)],
        "fluxo_cotação": [103.0],
        "tipo": ["Principal"],
    })
    ntnb_custom = NTNB(datetime.date(2027, 5, 15), custom_cashflows=df_nb)
    pu_nb = ntnb_custom.present_value(base_dt, FlatCurve(0.05), vna=4771.575461)
    assert pu_nb > 4700.0


def test_vna_from_vne_all_titles():
    from bz_pricing.core.vna_engine import calculate_lft_vna, calculate_vna_from_vne
    from bz_pricing.instruments.credit import Debenture

    # 1. LFT oficial (base R$ 1.000,00 em 01/07/2000) na data 30/10/2026 = 20201.049997
    vna_lft = calculate_lft_vna(datetime.date(2026, 10, 30))
    assert abs(vna_lft - 20201.049997) < 1e-6

    lft = LFT(datetime.date(2029, 3, 1))
    pu_lft = lft.present_value(datetime.date(2026, 10, 30), FlatCurve(0.001))
    assert pu_lft > 20000.0

    # 2. Genérico a partir de VNE offline para SELIC/CDI, IGP-M, IPCA e PRE
    vna_pre = calculate_vna_from_vne(1000.0, "2026-01-02", "2026-10-30", indexador="PRE")
    assert vna_pre == 1000.0

    daily = {"2026-10-28": 0.0005, "2026-10-29": 0.0005}
    vna_cdi = calculate_vna_from_vne(1000.0, "2026-10-28", "2026-10-30", indexador="CDI", history=daily, allow_download=False)
    assert abs(vna_cdi - 1001.00025) < 1e-6

    vna_igpm = calculate_vna_from_vne(
        1000.0, "2026-08-01", "2026-10-01", indexador="IGP-M",
        history={"2026-08-01": 0.0050, "2026-09-01": 0.0040},
        allow_download=False,
    )
    assert abs(vna_igpm - math.trunc((1000.0 * 1.005 * 1.004) * 1e6) / 1e6) < 1e-6

    # 3. Debênture / Crédito calculando VNA a partir do VNE
    deb = Debenture("DEB-VNE", maturity_date="2028-10-15", indexador="IPCA", issue_date="2026-08-15", nominal_value=1000.0)
    vna_deb = deb.calculate_vna(
        "2026-10-15",
        history={"2026-08-01": 0.0030, "2026-09-01": 0.0020},
        allow_download=False,
    )
    assert abs(vna_deb - math.trunc((1000.0 * 1.003 * 1.002) * 1e6) / 1e6) < 1e-6


def test_infer_anbima_vna_and_ntnc_index_ratio():
    from bz_pricing.core.vna_engine import calculate_ntnc_vna, infer_anbima_vna_and_projection

    # Extrai VNA oficial e projeção implícita da NTN-C 2031 (12%) em 30/10/2026
    res_ntnc = infer_anbima_vna_and_projection(
        pu_anbima=8000.532,
        cotacao_anbima=118.2632,
        vna_last_anniv=6671.488370,
        du_calc=20,
        du_total=20,
    )
    assert abs(res_ntnc["vna_oficial"] - 6765.022425) < 1e-6
    assert abs(res_ntnc["proj_implicita"] - 0.014020) < 1e-5

    # Fórmula por razão de Números-Índice FGV offline (com pró-rata)
    vna_idx = calculate_ntnc_vna(
        datetime.date(2026, 10, 15),
        vne=1000.0,
        igpm_proj=0.014,
        base_index=100.0,
        current_index=667.148837,
        allow_download=False,
    )
    assert vna_idx > 6671.488370 and vna_idx < 6765.0


def test_tpf_modified_duration_sloped_curve():
    base_date = datetime.date(2026, 10, 30)
    # Estrutura inclinada acentuada para testar ponderação fluxo a fluxo em M5
    t_nodes = np.array([21.0, 252.0, 504.0, 1260.0], dtype=np.float64)
    y_nodes = np.array([0.08, 0.11, 0.14, 0.17], dtype=np.float64)
    curve_slope = SimpleCurve(t_nodes, y_nodes, FlatForward252Interpolator(0, 0))

    ntnf = NTNF(datetime.date(2031, 1, 1), coupon_rate=0.10)
    mod_dur = ntnf.modified_duration(base_date, curve_slope)
    # Valor de referência fluxo a fluxo (sem usar taxa do último vértice como aproximação)
    cf = ntnf.cashflow_table(base_date, curve_slope)
    expected_md = float(((cf["du"] / 252.0) / (1.0 + cf["taxa"]) * cf["valor_presente"]).sum() / cf["valor_presente"].sum())
    assert abs(mod_dur - expected_md) < 1e-6

    # Em NTN-B inclinada também deve coincidir com o fluxo a fluxo exato
    t_r = np.array([21.0, 252.0, 1008.0], dtype=np.float64)
    y_r = np.array([0.04, 0.06, 0.09], dtype=np.float64)
    curve_real = SimpleCurve(t_r, y_r, FlatForward252Interpolator(0, 0))
    ntnb = NTNB(datetime.date(2030, 8, 15), coupon_rate=0.06)
    cf_b = ntnb.cashflow_table(base_date, curve_real, vna=4771.575461)
    expected_md_b = float(((cf_b["du"] / 252.0) / (1.0 + cf_b["taxa_real"]) * cf_b["valor_presente"]).sum() / cf_b["valor_presente"].sum())
    assert abs(ntnb.modified_duration(base_date, curve_real) - expected_md_b) < 1e-6



