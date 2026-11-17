# Casos de Uso e Exemplos Práticos (`bz-pricing`)

Esta página reúne casos completos de utilização do pacote `bz-pricing` para construção de curvas de juros, cálculo de VNA a partir do VNE, precificação de títulos públicos e crédito privado com fluxos customizados, derivativos lineares, opções e validação de superfícies de volatilidade.

---

## 1. Curvas de Juros e Inflação (Modo *Offline* e Automático)

Todas as curvas podem ser construídas em memória a partir de vértices ou componentes fornecidos pelo usuário (`allow_download=False`), sem acesso à rede nem leitura de arquivos externos.

```python
import datetime
import pandas as pd
from bz_pricing.curves import CurveManager

base_date = datetime.date(2026, 10, 30)
manager = CurveManager(allow_download=False)

# 1.1 Curva DI x Pré (PRE) construída a partir de vértices em dias úteis (DU) e taxas (% a.a.)
df_pre = pd.DataFrame({
    "DU": [1, 21, 42, 63, 126, 252, 504],
    "rate": [14.90, 14.88, 14.85, 14.80, 14.70, 14.55, 14.30],
})
curva_pre = manager.get_curve("PRE", base_date, data=df_pre)
print(f"PRE 126 DU: {curva_pre.get_rate(126.0) * 100:.4f}% a.a.")

# 1.2 Cupom Limpo de IPCA (DPL) por componentes: PRE + futuros DAP + prévia do IPCA
df_dap = pd.DataFrame({
    "XprtnDt": [datetime.date(2026, 11, 16), datetime.date(2027, 5, 17), datetime.date(2028, 5, 15)],
    "AdjstdQtTax": [9.20, 7.60, 7.35],
})
curva_dpl = manager.get_curve("DPL", base_date, pre_curve=curva_pre, dap_df=df_dap, eipca_0=0.0020)
print(f"DPL 252 DU: {curva_dpl.get_rate(252.0) * 100:.4f}% a.a.")
```

---

## 2. Títulos Públicos Federais, VNA a partir do VNE e Fluxos Customizados

O motor `bz_pricing.core.vna_engine` apura o VNA de qualquer título a partir do Valor Nominal de Emissão (`vne`) e data-base (`base_date`) para IPCA, IGP-M, Selic, CDI e prefixados, com suporte tanto a séries *offline* quanto à extração no BCB. Todos os instrumentos em [`tpf.py`](../src/bz_pricing/instruments/tpf.py) também aceitam fluxos próprios via `custom_cashflows`.

```python
import datetime
from bz_pricing import LTN, NTNF, LFT, NTNB, calculate_lft_vna, calculate_ntnb_vna, calculate_vna_from_vne
from bz_pricing.core.curves import FlatCurve

base_date = datetime.date(2026, 10, 30)

# 2.1 Cálculo de VNA oficial da LFT (Selic acumulada desde 01/07/2000) e da NTN-B (IPCA desde 15/07/2000)
vna_lft = calculate_lft_vna(base_date)
vna_ntnb = calculate_ntnb_vna(base_date, ipca_proj=0.001972)
print(f"VNA LFT: R$ {vna_lft:.6f} | VNA NTN-B: R$ {vna_ntnb:.6f}")

# 2.2 Precificação automática de NTN-B e LFT sem informar VNA manualmente
ntnb = NTNB(datetime.date(2027, 5, 15), coupon_rate=0.06)
pu_ntnb, df_ntnb = ntnb.present_value(base_date, FlatCurve(0.046629), ipca_proj=0.001972, return_cashflows=True)
lft = LFT(datetime.date(2029, 3, 1))
pu_lft = lft.present_value(base_date, FlatCurve(0.0005))

# 2.3 Precificação com fluxos customizados do usuário em TPF
fluxos_usuario = [
    (datetime.date(2027, 1, 4), 48.808848, "Cupom"),
    (datetime.date(2027, 7, 1), 48.808848, "Cupom"),
    (datetime.date(2028, 1, 3), 1048.808848, "Cupom + Principal"),
]
ntnf_custom = NTNF(custom_cashflows=fluxos_usuario)
pu_custom = ntnf_custom.present_value(base_date, FlatCurve(0.1450))
dv01_custom = ntnf_custom.dv01(base_date, FlatCurve(0.1450))
print(f"NTN-F Custom PU: R$ {pu_custom:.6f} | DV01: R$ {dv01_custom:.6f}")
```

---

## 3. Crédito Privado, *Z-Spread*, YTM/TIR e Carteira em Lote

Os instrumentos `Debenture`, `CRI`, `CRA`, `CotaFIDC` e `LetraFinanceira` suportam construção por código SND/ANBIMA (`from_ticker`), por cronograma (`from_schedule`) ou por fluxos explícitos (`from_cashflows` / `custom_cashflows`), além de cálculo de VNA a partir do VNE (`calculate_vna`) e avaliação em lote (`price_credit_portfolio`).

```python
import datetime
from bz_pricing.instruments import Debenture, CRI, LetraFinanceira, price_credit_portfolio

base_date = datetime.date(2026, 10, 30)

# 3.1 Debênture construída diretamente por fluxos customizados do usuário
deb_cf = Debenture.from_cashflows(
    "DEB-USER",
    [
        (datetime.date(2027, 4, 30), 60.0, "Juros"),
        (datetime.date(2027, 10, 29), 1060.0, "Vencimento"),
    ],
    indexador="PRE",
)
pu_deb = deb_cf.present_value(base_date, 0.1325)
ytm_deb = deb_cf.ytm(base_date, pu_deb)
print(f"Debênture Custom PU: R$ {pu_deb:.6f} | YTM: {ytm_deb * 100:.4f}% a.a.")

# 3.2 CRI IPCA+ incentivado com cronograma SAC e Z-Spread (gross-up tributário)
cri = CRI.from_schedule(
    ticker="CRI-EXEMPLO",
    issue_date="2026-01-02",
    maturity_date="2029-01-02",
    indexador="IPCA",
    coupon_rate=0.078,
    coupon_freq="mensal",
    amort_type="sac",
)
vna_cri = cri.calculate_vna(
    base_date,
    history={"2026-01-01": 0.0040, "2026-02-01": 0.0035, "2026-03-01": 0.0030},
    projections={"2026-10-01": 0.0025},
    allow_download=False,
)
pu_cri = cri.present_value(base_date, 0.0820, vna=vna_cri)
z_sprd = cri.z_spread(base_date, discount_rate=0.0820, risk_free_rate=0.0720, vna=vna_cri, incentivada=True)

# 3.3 Letra Financeira DI+ e avaliação de carteira em lote
lf = LetraFinanceira.from_bullet("LF-EXEMPLO", "2026-01-02", "2028-01-03", indexador="DI", coupon_rate=0.006, nominal_value=50000.0)
df_port = price_credit_portfolio(
    [
        {"instrument": deb_cf, "discount_rate": 0.1325, "quantity": 100},
        {"instrument": cri, "discount_rate": 0.0820, "vna": vna_cri, "quantity": 50},
        {"instrument": lf, "discount_rate": 0.0080, "vna": 50000.0, "pu_par_val": 51200.0, "di_curve": 0.145, "quantity": 5},
    ],
    base_date=base_date,
)
print(df_port[["ticker", "classe", "pu", "valor_financeiro", "duration_du", "dv01"]])

# 3.4 Curva de Hazard Rate, PV com risco de crédito (CVA) e Credit DV01 (modo offline)
from bz_pricing.instruments import HazardRateCurve

hz = HazardRateCurve.from_credit_spreads([252, 504], [0.015, 0.020], recovery_rate=0.40)
res_risky = deb_cf.risky_present_value(base_date, risk_free_rate_or_curve=0.12, hazard_curve=hz)
lam_imp = deb_cf.implied_hazard_rate(base_date, market_pu=res_risky["pv_risky"], risk_free_rate_or_curve=0.12, recovery_rate=0.40)
cdv01 = deb_cf.credit_dv01(base_date, risk_free_rate_or_curve=0.12, hazard_curve=hz)
print(f"PV Risky: R$ {res_risky['pv_risky']:.2f} | CVA: R$ {res_risky['cva']:.2f} | Hazard implícito: {lam_imp*100:.2f}% | Credit DV01: R$ {cdv01:.4f}")
```

---

## 4. Derivativos Lineares (Futuros, *Swaps* e NDF)

O módulo [`derivatives.py`](../src/bz_pricing/instruments/derivatives.py) precifica futuros listados da B3, *swaps* de taxa de juros e contratos NDF com cálculo de PU, valor justo, ajuste financeiro por variação de taxa e DV01.

```python
import datetime
from bz_pricing.instruments import DI1Future, DOLFuture, SwapDIxPRE, SwapDIxIPCA, NDFContract, price_di1_portfolio

base_date = datetime.date(2026, 10, 30)

# 4.1 Futuro de DI1 e avaliação em lote
di1 = DI1Future("DI1F28", "2028-01-03")
print(f"DI1F28 PU: R$ {di1.pu(base_date, 0.1385):.2f} | DV01: R$ {di1.dv01(base_date, 0.1385):.4f}")

df_di1 = price_di1_portfolio(
    [
        {"instrument": DI1Future("DI1F27", "2027-01-04"), "rate": 0.1490, "contracts": 50},
        {"instrument": di1, "rate": 0.1385, "contracts": -30},
    ],
    base_date=base_date,
)

# 4.2 Swap DI x Pré e NDF USD/BRL
swap_pre = SwapDIxPRE(notional=1_000_000.0, start_date="2026-01-02", maturity_date="2028-01-03", fixed_rate=0.1350)
mtm_swap = swap_pre.mtm(base_date, pre_rate_or_curve=0.1410, di_accrued_factor=1.025)

ndf = NDFContract(notional_usd=250_000.0, maturity_date="2027-01-29", strike_brl=5.62)
mtm_ndf = ndf.mtm(base_date, spot_ptax=5.55, pre_rate_or_curve=0.1480, cupom_rate_or_curve=0.0520)
print(f"MtM Swap: R$ {mtm_swap:,.2f} | MtM NDF: R$ {mtm_ndf:,.2f}")
```

---

## 5. Opções, Superfícies de Volatilidade e Validação Empírica

Os módulos [`options.py`](../src/bz_pricing/instruments/options.py) e [`surface.py`](../src/bz_pricing/volatility/surface.py) integram a precificação de opções listadas e de balcão em Cython à calibração e auditoria de superfícies de volatilidade implícita.

```python
import datetime
import numpy as np
from bz_pricing.instruments import EquityOption, FXOption, IDIOption, DI1Option, CorradoSuOption, price_options_portfolio
from bz_pricing.volatility import SVISlice, SABRSlice, VolatilitySurface, validate_slice_out_of_sample

base_date = datetime.date(2026, 10, 30)

# 5.1 Opção americana sobre ação (árvore binomial CRR com gregas consistentes) e lote vetorizado
opt_am = EquityOption("PETRO38", "2027-01-04", 38.0, "PUT", "AMERICAN")
rm_am = opt_am.risk_metrics(base_date, spot=37.20, pre_rate_or_curve=0.1480, vol=0.32, crr_steps=200)
print(f"PETRO38 Americana -> Preço: R$ {rm_am['price']:.4f} | Delta: {rm_am['delta']:.4f}")

# 5.2 Superfície SVI, auditoria de não arbitragem, Breeden-Litzenberger e Dupire
sl_63 = SVISlice(a=0.012, b=0.035, rho=-0.30, m=0.0, sigma=0.11, T=63 / 252.0)
sl_126 = SVISlice(a=0.025, b=0.060, rho=-0.30, m=0.0, sigma=0.13, T=126 / 252.0)
surf = VolatilitySurface.from_svi_slices({63: sl_63, 126: sl_126})

arb = surf.check_no_arbitrage(forward_by_du=5.55, r_cont_by_du=0.138)
wing = surf.wing_slope_check()
bl = surf.breeden_litzenberger_density(63, forward=5.55, r_cont=0.138)
dup = surf.dupire_local_vol_grid(forward_by_du=5.55, r_cont_by_du=0.138)
print(f"Não arbitragem: {arb['passed']} | Lee wings: {wing['passed']} | BL: {bl['passed']} | Dupire: {dup['passed']}")

# 5.3 Precificação de FXOption acoplada diretamente à VolatilitySurface e ajuste de fixing
fx_call = FXOption("DOLF27C560", "2027-01-04", 5.60, "CALL")
preco_fx = fx_call.price(
    base_date,
    spot_ptax=5.52,
    pre_rate_or_curve=0.1480,
    cupom_limpo_or_curve=0.0520,
    vol=surf,
    fixing_vol_adjust=True,
)
print(f"FX Call (com superfície e fixing adjust): R$ {preco_fx:.4f}")

# 5.4 Swaption (Black-76 e Bachelier), CapFloor, BarrierOption e AsianOption
from bz_pricing.instruments import Swaption, CapFloor, BarrierOption, AsianOption

swp = Swaption("SWP-DI-28", "2027-04-30", "2028-04-28", strike_rate=0.138, option_type="PAYER", notional=1_000_000.0)
preco_swp = swp.price(base_date, pre_rate_or_curve=0.140, vol=0.0150, model="bachelier")

cap = CapFloor(
    "CAP-DI",
    fixing_dates=[datetime.date(2027, 1, 4), datetime.date(2027, 4, 1)],
    payment_dates=[datetime.date(2027, 4, 1), datetime.date(2027, 7, 1)],
    strike_rate=0.135,
    option_type="CAP",
    notional=1_000_000.0,
)
preco_cap = cap.price(base_date, pre_rate_or_curve=0.142, vol=0.20, model="black76")

bar_do = BarrierOption("VALE-DO", "2027-04-30", strike=60.0, barrier=52.0, option_type="CALL", barrier_type="down-and-out")
asian_ar = AsianOption("PETR-AA", "2027-04-30", strike=40.0, option_type="CALL", average_type="arithmetic")
print(f"Swaption: R$ {preco_swp:,.2f} | Cap: R$ {preco_cap:,.2f} | Barreira DO: R$ {bar_do.price(base_date, 62.0, 0.14, 0.30):.4f} | Asiática: R$ {asian_ar.price(base_date, 40.0, 0.14, 0.30):.4f}")

# 5.5 Precificação em lote Cython para carteiras mistas (Equity, FX, IDI, DI1 e Corrado-Su)
df_lote_opt = price_options_portfolio(
    [
        {"instrument": opt_am, "spot": 37.20, "pre_rate_or_curve": 0.1480, "vol": 0.32, "quantity": 100},
        {"instrument": fx_call, "spot_ptax": 5.52, "pre_rate_or_curve": 0.1480, "cupom_limpo_or_curve": 0.0520, "vol": surf, "quantity": 10},
    ],
    base_date=base_date,
)
print(df_lote_opt[["ticker", "classe", "price", "mtm", "delta", "vega_1pct"]])
```

---

## 6. Curvas Internacionais (`SOFR_OIS`, `UST_PAR`) e Superfície de Mistura Lognormal Bloomberg

Todos os interpoladores *offshore*, *bootstrappers* de OIS e Treasuries e ferramentas do modelo Bloomberg (desamericanização binomial, filtro de Kalman para dividendos, decomposição híbrida de dividendos e mistura de lognormais com estado de *default*) operam de forma integralmente *offline* com dados do usuário:

```python
import datetime
import math
import numpy as np
from bz_pricing.curves.offshore import bootstrap_ois_curve, bootstrap_treasury_curve
from bz_pricing.core.curves import SvenssonCurve
from bz_pricing.volatility import (
    deamericanize_and_infer_forward_bbg,
    KalmanDividendFilter,
    DividendCurveBBG,
    LognormalMixtureSlice,
    BloombergMixtureSurface,
)

# 6.1 Bootstrapping offline de OIS SOFR (ACT/360) e Treasuries EUA (Monotone Convex e Svensson)
sofr_crv = bootstrap_ois_curve(
    pillars_dc=[30, 90, 180, 365, 730, 1825],
    par_rates=[0.0432, 0.0428, 0.0420, 0.0405, 0.0392, 0.0385],
    convention="ACT/360",
    method="monotone_convex",
)
ust_mc = bootstrap_treasury_curve(
    pillars_dc=[91, 182, 365, 730, 1825, 3650],
    par_yields=[0.0435, 0.0430, 0.0422, 0.0410, 0.0415, 0.0425],
    method="monotone_convex",
)
t_nodes = np.array([91.0, 182.0, 365.0, 730.0, 1825.0, 3650.0])
y_nodes = np.array([ust_mc.get_rate(t) for t in t_nodes])
ust_svensson = SvenssonCurve.fit(t_nodes, y_nodes, day_count_basis=365.0)
print(f"SOFR 1Y zero: {sofr_crv.get_rate(365.0)*100:.4f}% | UST Svensson 5Y fwd: {ust_svensson.instantaneous_forward(1825.0)*100:.4f}%")

# 6.2 Part I Bloomberg: inferência de forward/dividendo com desamericanização, filtro de Kalman e decomposição híbrida
strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
T_opt = 0.50
r_c = 0.12
fwd_true = 100.0 * math.exp((r_c - 0.04) * T_opt)
sl_ref = LognormalMixtureSlice([0.65, 0.35], [0.96, 1.0742857], [0.16, 0.24], forward=fwd_true, T=T_opt)
c_eu = np.array([sl_ref.option_price(k, r_cont=r_c, is_call=True) for k in strikes])
p_eu = np.array([sl_ref.option_price(k, r_cont=r_c, is_call=False) for k in strikes])
res_fwd = deamericanize_and_infer_forward_bbg(strikes, c_eu, p_eu, spot=100.0, r_cont=r_c, T=T_opt, exercise="EUROPEAN")

kf = KalmanDividendFilter(alpha_over_sigma=1.0)
kf_out = kf.update(res_fwd["cash_dividend"])
decomp = DividendCurveBBG.decompose_blended_dividends([0.25, 0.75], spot_f0=100.0, fwd_T=102.0, r_cont=0.05, T=1.0)
print(f"Forward inferido: {res_fwd['forward']:.4f} | Div. filtrado: {kf_out['dividend']:.4f} | q_res: {decomp['q']:.4f}")

# 6.3 Part II Bloomberg: mistura de lognormais com estado de default e superfície 2D
sl_mix = LognormalMixtureSlice.fit(strikes, np.where(strikes < fwd_true, p_eu, c_eu), np.where(strikes < fwd_true, "PUT", "CALL"), fwd_true, r_c, T_opt, n_components=2)
surf_bbg = BloombergMixtureSurface({126: sl_mix})
print(f"Q_default={sl_mix.Q_default:.4f} | Vol ATM (126 DU): {surf_bbg.get_vol(126, fwd_true)*100:.2f}%")
```
