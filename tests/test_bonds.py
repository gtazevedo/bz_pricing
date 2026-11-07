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
    ntnc = NTNC(maturity_date, coupon_rate=0.06)

    cotacao = ntnc.cotacao(base_date, curve_real)
    assert cotacao == 96.1378

    vna_oficial = 8321.942046
    pu = ntnc.present_value(base_date, curve_real, vna=vna_oficial)
    pu_oficial = 8000.532000
    assert abs(pu - pu_oficial) < 1e-5

    dur = ntnc.duration(base_date, curve_real)
    mod_dur = ntnc.modified_duration(base_date, curve_real)
    dv01 = ntnc.dv01(base_date, curve_real, vna=vna_oficial)
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
    vna_ntnc = 8321.942046

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
            bond = NTNC(dt_vcto, coupon_rate=0.06)
            pu_calc, cf_df = bond.present_value(base_date, curve, vna=vna_ntnc, return_cashflows=True)
        else:
            continue

        assert abs(pu_calc - pu_ofic) < 0.01, f"Divergência em {tit} {dt_vcto}: calc={pu_calc}, ofic={pu_ofic}"
        assert abs(pu_calc - pu_ofic) < 1e-4, f"Divergência superior a 1e-4 em {tit} {dt_vcto}"
        assert len(cf_df) > 0
        tested_count += 1

    assert tested_count == 50

