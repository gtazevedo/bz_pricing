import datetime
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _extract_python_blocks(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def test_exemplos_casos_uso_blocks_execute():
    blocks = _extract_python_blocks(ROOT / "docs" / "exemplos_casos_uso.md")
    assert len(blocks) == 7

    # Bloco 1 (Curvas offline), Bloco 3 (Crédito/Z-Spread/Hazard offline),
    # Bloco 4 (Derivativos lineares offline), Bloco 5 (Opções/Superfície offline),
    # Bloco 6 (Curvas offshore e modelo Bloomberg offline) e Bloco 7 (Offshore Bonds, Swaps e Opções c/ Dividendos/Exóticas)
    for idx in (0, 2, 3, 4, 5, 6):
        ns = {}
        exec(compile(blocks[idx], f"docs/exemplos_casos_uso.md#block_{idx+1}", "exec"), ns)

    # Bloco 2: TPF com custom_cashflows (executa completo localmente se BCB disponível, senão trecho offline)
    block2 = blocks[1]
    offline_part = (
        "import datetime\nfrom bz_pricing import NTNF\nfrom bz_pricing.core.curves import FlatCurve\n"
        "base_date = datetime.date(2026, 10, 30)\n"
        "# 2.3" + block2.split("# 2.3")[1]
    )
    is_ci_or_offline = (
        os.environ.get("CI", "").strip().lower() in ("1", "true", "yes")
        or os.environ.get("BZ_PRICING_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    )
    if is_ci_or_offline:
        ns2 = {}
        exec(compile(offline_part, "docs/exemplos_casos_uso.md#block_2_offline", "exec"), ns2)
    else:
        try:
            ns2 = {}
            exec(compile(block2, "docs/exemplos_casos_uso.md#block_2", "exec"), ns2)
        except Exception:
            ns2 = {}
            exec(compile(offline_part, "docs/exemplos_casos_uso.md#block_2_offline", "exec"), ns2)


def test_guia_uso_api_offline_blocks_execute():
    blocks = _extract_python_blocks(ROOT / "docs" / "guia_uso_api.md")
    assert len(blocks) >= 10

    # Blocos estritamente offline em docs/guia_uso_api.md:
    # 1.1 (idx 1), 3 (idx 5), 4 (idx 6), 4.2/builder (idx 7), 4.3 (idx 8), 6 (idx 12), 7 (idx 13), 8 (idx 14, 15), 9 (idx 16, 17)
    for idx in (1, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17):
        ns = {}
        exec(compile(blocks[idx], f"docs/guia_uso_api.md#block_{idx}", "exec"), ns)

    # Trecho offline do bloco 5.3 (idx 11: custom_cashflows, CRI, Z-Spread, lote crédito)
    block53_lines = [ln for ln in blocks[11].splitlines() if "calculate_lft_vna(" not in ln and "calculate_ntnb_vna(" not in ln]
    ns53 = {}
    exec(compile("\n".join(block53_lines), "docs/guia_uso_api.md#block_11_offline", "exec"), ns53)


def test_readme_offline_blocks_execute():
    blocks = _extract_python_blocks(ROOT / "README.md")
    assert len(blocks) >= 6

    # Blocos offline: Seção 5.5 (idx 3), Seção 5.6 (idx 4) e Seção 5.7 (idx 5)
    for idx in (3, 4, 5):
        ns_opt = {}
        exec(compile(blocks[idx], f"README.md#block_{idx}", "exec"), ns_opt)

    # Bloco 5.3 TPFs (parte determinística sem downloads BCB)
    b_tpf_lines = [ln for ln in blocks[1].splitlines() if "calculate_lft_vna(" not in ln and "calculate_ntnb_vna(" not in ln and "present_value(base_date, curva_real, ipca_proj" not in ln]
    ns_tpf = {}
    exec(compile("\n".join(b_tpf_lines), "README.md#block_tpf_offline", "exec"), ns_tpf)


def test_stage3_low_severity_fixes_b2_to_b8():
    from bz_pricing.core.curves import FlatCurve, SimpleCurve
    from bz_pricing.core.derivatives_core import swap_pre_di_mtm
    from bz_pricing.core.interpolators import FlatForward252Interpolator
    from bz_pricing.core.options_core import asian_arithmetic_tw_price, crr_american_price
    from bz_pricing.instruments import DAPFuture, Debenture, SwapDIxPRE, price_di1_portfolio, DI1Future
    from bz_pricing.instruments.tpf import LTN, NTNB, NTNC, NTNF

    base_dt = datetime.date(2026, 10, 30)

    # B2: swap_pre_di_mtm com du_total != du_rem coincide exatamente com SwapDIxPRE.risk_metrics
    sw = SwapDIxPRE(notional=1_000_000.0, start_date="2026-01-02", maturity_date="2028-01-03", fixed_rate=0.135)
    rm = sw.risk_metrics(base_dt, pre_rate_or_curve=0.142, di_accrued_factor=1.10)
    du_total = sw.calendar.bizdays(sw.start_date, sw.maturity_date)
    du_rem = sw.calendar.bizdays(base_dt, sw.maturity_date)
    mtm_cy, pv_pre_cy, pv_di_cy, dv01_cy = swap_pre_di_mtm(
        1_000_000.0, 0.135, 0.142, du_total, 1.10, 1.0, du_rem
    )
    assert abs(rm["mtm"] - mtm_cy) < 1e-9
    assert abs(rm["dv01"] - dv01_cy) < 1e-9

    # B3 & B4: custom_cashflows reproduce PU, Duration, ModDur e DV01 exatos em TPF e exige VNA > 0 em fluxo_cotação
    t_s = np.array([21.0, 252.0, 756.0], dtype=np.float64)
    y_s = np.array([0.12, 0.135, 0.15], dtype=np.float64)
    curva_slope = SimpleCurve(t_s, y_s, FlatForward252Interpolator(0, 0))
    ntnf = NTNF(datetime.date(2029, 1, 1), coupon_rate=0.10)
    pu_ref, df_ref = ntnf.present_value(base_dt, curva_slope, return_cashflows=True)
    ntnf_c = NTNF(custom_cashflows=df_ref[["data_fluxo", "fluxo_nominal", "tipo"]])
    assert abs(ntnf_c.present_value(base_dt, curva_slope) - pu_ref) < 1e-6
    assert abs(ntnf_c.modified_duration(base_dt, curva_slope) - ntnf.modified_duration(base_dt, curva_slope)) < 1e-8
    assert abs(ntnf_c.dv01(base_dt, curva_slope) - ntnf.dv01(base_dt, curva_slope)) < 1e-7

    ntnb = NTNB(datetime.date(2028, 8, 15), coupon_rate=0.06)
    vna_b = 4771.575461
    c_real = FlatCurve(0.062)
    pu_b, df_b = ntnb.present_value(base_dt, c_real, vna=vna_b, return_cashflows=True)
    ntnb_c = NTNB(custom_cashflows=df_b[["data_fluxo", "fluxo_cotação", "tipo"]])
    assert abs(ntnb_c.present_value(base_dt, c_real, vna=vna_b) - pu_b) < 1e-6
    assert abs(ntnb_c.dv01(base_dt, c_real, vna=vna_b) - ntnb.dv01(base_dt, c_real, vna=vna_b)) < 1e-7
    assert ntnb_c.duration(base_dt, c_real) > 0.0

    # B4: fluxo_cotação sem VNA positivo deve lançar ValueError
    ntnf_bad = NTNF(custom_cashflows=df_b[["data_fluxo", "fluxo_cotação", "tipo"]])
    with pytest.raises(ValueError):
        ntnf_bad.present_value(base_dt, c_real)

    # B5: contract_multiplier customizado, vna_ntnb no DAPFuture e alias 'quantity' em price_di1_portfolio
    dap = DAPFuture("DAPQ28", "2028-08-15", contract_multiplier=0.0005)
    assert dap.MULTIPLIER == 0.0005
    adj_pts = dap.daily_settlement(90100.0, 90000.0, contracts=2.0)
    adj_brl = dap.daily_settlement(90100.0, 90000.0, contracts=2.0, vna_ntnb=4771.575461)
    assert abs(adj_brl - adj_pts * 4771.575461) < 1e-8

    df_di1 = price_di1_portfolio([{"instrument": DI1Future("DI1F28", "2028-01-03"), "rate": 0.14, "quantity": 5}], base_dt)
    assert df_di1["contracts"].iloc[0] == 5.0

    # B6: gross_up_rate em título indexado ao DI
    deb_di = Debenture.from_schedule("DEB-DI-GU", "2026-10-30", "2028-10-30", indexador="DI", coupon_rate=0.015)
    gu = deb_di.gross_up_rate(base_dt, discount_rate=0.018, di_curve=0.14)
    assert gu > 0.018

    # B7: Limites singulares da fórmula asiática contínua (b=0, b=-sig^2, 2b=-sig^2)
    sig = 0.30
    p_b0 = asian_arithmetic_tw_price(100.0, 100.0, 0.10, 0.10, 1.0, sig, True, 0)
    p_bs2 = asian_arithmetic_tw_price(100.0, 100.0, 0.01, 0.01 + sig * sig, 1.0, sig, True, 0)
    p_2bs2 = asian_arithmetic_tw_price(100.0, 100.0, 0.05, 0.05 + 0.5 * sig * sig, 1.0, sig, True, 0)
    p_bs2_disc = asian_arithmetic_tw_price(100.0, 100.0, 0.01, 0.01 + sig * sig, 1.0, sig, True, 2000)
    p_2bs2_disc = asian_arithmetic_tw_price(100.0, 100.0, 0.05, 0.05 + 0.5 * sig * sig, 1.0, sig, True, 2000)
    assert math.isfinite(p_b0) and p_b0 > 0.0
    assert abs(p_bs2 - p_bs2_disc) < 0.01
    assert abs(p_2bs2 - p_2bs2_disc) < 0.01

    # B8: Violação de não arbitragem CRR (p fora de [0, 1]) retorna NaN
    p_bad_crr = crr_american_price(100.0, 100.0, 0.80, 0.0, 4.0, 0.02, True, 1)
    assert math.isnan(p_bad_crr)
