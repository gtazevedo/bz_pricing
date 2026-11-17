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
z_sprd = cri.z_spread(base_date, market_pu=pu_cri, ettj_pre_rate_or_curve=0.0720, vna=vna_cri, incentivada=True)

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
print(df_port[["ticker", "classe", "pu", "mtm", "duration_du", "dv01"]])
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
        {"instrument": DI1Future("DI1F27", "2027-01-04"), "rate": 0.1490, "quantity": 50},
        {"instrument": di1, "rate": 0.1385, "quantity": -30},
    ],
    base_date=base_date,
)

# 4.2 Swap DI x Pré e NDF USD/BRL
swap_pre = SwapDIxPRE("SWAP-PRE-28", "2028-01-03", fixed_rate=0.1350, notional=1_000_000.0)
mtm_swap = swap_pre.mtm(base_date, pre_curve=0.1410, cdi_accrued_factor=1.025)

ndf = NDFContract("NDF-USD", "2027-01-29", forward_rate=5.62, notional_usd=250_000.0)
mtm_ndf = ndf.mtm(base_date, spot_ptax=5.55, pre_rate_or_curve=0.1480, cupom_limpo_or_curve=0.0520)
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
