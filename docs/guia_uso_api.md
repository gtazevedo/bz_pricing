# Guia de Referência da API (`bz-pricing`)

Este documento detalha as interfaces programáticas dos módulos de gerenciamento de curvas, construção algébrica, coleta de dados e precificação de instrumentos de renda fixa do pacote `bz-pricing`.

---

## 1. Orquestrador de Estruturas a Termo (`CurveManager`)

A classe [`CurveManager`](../src/bz_pricing/curves/manager.py) centraliza a instanciação de curvas *onshore* e *offshore* e gerencia a validação de insumos fornecidos pelo usuário (`allow_download=False` por padrão) ou a resolução automática a partir de bases locais e provedores públicos (`allow_download=True`).

```python
import datetime
from bz_pricing.curves.manager import CurveManager

# Inicializa o orquestrador com resolução automática habilitada
manager = CurveManager(allow_download=True)
base_date = datetime.date(2026, 10, 30)

# 1. Curva DI x Pré (PRE)
curva_pre = manager.get_curve("PRE", base_date)
taxa_21_du = curva_pre.get_rate(21.0)
taxa_252_du = curva_pre.get_rate(252.0)

print(f"PRE 21 DU:  {taxa_21_du * 100:.4f}% a.a.")
print(f"PRE 252 DU: {taxa_252_du * 100:.4f}% a.a.")
```

### 1.1 Construção a partir de Vértices e Componentes Externos (`allow_download=False`)

Todas as estruturas suportadas podem ser instanciadas de forma integralmente *offline* mediante o fornecimento de tabelas de vértices ou dos componentes primários de cálculo. Caso um insumo obrigatório esteja ausente sob `allow_download=False`, a exceção [`DataUnavailableError`](../src/bz_pricing/curves/exceptions.py) é lançada com a indicação explícita do componente faltante.

```python
import datetime
import pandas as pd
from bz_pricing.curves.manager import CurveManager, DataUnavailableError

manager = CurveManager(allow_download=False)
base_date = datetime.date(2026, 10, 30)

# 1. Passagem de DataFrame com prazos em dias úteis (DU) e taxas (% a.a.)
df_pre = pd.DataFrame({
    "DU": [1, 21, 42, 63, 126, 252],
    "rate": [13.65, 13.70, 13.75, 13.85, 13.95, 14.10]
})
curva_pre = manager.get_curve("PRE", base_date, data=df_pre)

# 2. Passagem via pd.Series ou dicionário {DU: taxa}
serie_pre = pd.Series([13.65, 13.75, 13.90], index=[1, 21, 63])
curva_pre_s = manager.get_curve("PRE", data=serie_pre)

# 3. Construção de Cupom Limpo de IPCA (DPL) por componentes (PRE + futuros DAP + prévia IPCA)
df_dap = pd.DataFrame({
    "XprtnDt": [datetime.date(2026, 11, 16), datetime.date(2027, 5, 17)],
    "AdjstdQtTax": [9.62, 6.80]
})
curva_dpl = manager.get_curve("DPL", base_date, pre_curve=curva_pre, dap_df=df_dap, eipca_0=0.0045)

# 4. Futuros de Ações e Convenience Yield com cotações informadas pelo usuário
df_petr = pd.DataFrame({
    "XprtnDt": ["2026-11-16", "2026-12-18"],
    "AdjstdQt": [53.10, 53.60]
})
fut_petr, cy_petr = manager.get_stock_future_curve(
    "PETRP",
    base_date=base_date,
    spot_price=52.71,
    data=df_petr,
    pre_curve=curva_pre
)
```

### 1.2 Consulta de Curvas de Inflação, Câmbio, Ações e Taxas Internacionais

```python
import datetime
from bz_pricing.curves.manager import CurveManager

manager = CurveManager(allow_download=True)
base_date = datetime.date(2026, 10, 30)

# Curvas de Inflação (IPCA e IGP-M)
curva_dic = manager.get_curve("DIC", base_date)
curva_dpl = manager.get_curve("DPL", base_date)
curva_dim = manager.get_curve("DIM", base_date)

# Curvas de Cupom Cambial e Moedas a Termo
curva_dol = manager.get_curve("DOL", base_date)
curva_doc = manager.get_curve("DOC", base_date)
curva_euc = manager.get_curve("EUC", base_date)
curva_ptx = manager.get_curve("PTX", base_date)

# Curvas Internacionais (ACT/360)
curva_sfr = manager.get_curve("SFR", base_date)
curva_zus = manager.get_curve("ZUS", base_date)
```

---

## 2. Ingestão e Extração de Dados (`B3DataLoader` e `B3Scraper`)

### 2.1 Carregamento Local (`B3DataLoader`)

A classe [`B3DataLoader`](../src/bz_pricing/data/loaders.py) realiza a leitura, filtragem e padronização de arquivos de contratos derivativos (`BVBG_SPRD_*.csv`), títulos públicos (`anbima_tpf_*.csv`) e tabelas referenciais (`taxas_*.csv`):

```python
import datetime
from bz_pricing.data.loaders import B3DataLoader

loader = B3DataLoader(data_dir="dados/b3")
base_date = datetime.date(2026, 10, 30)

# Contratos futuros de DI1 negociados na data-base
df_di1 = loader.get_curve_instruments(base_date, asst="DI1")

# Títulos públicos federais do mercado secundário (LTN, NTN-F, NTN-B, LFT)
df_ltn = loader.get_anbima_bonds(base_date, bond_type="LTN")

# Tabelas referenciais para validação empírica
df_pre_ref = loader.load_taxas_referenciais("PRE", base_date)
```

### 2.2 Coleta Automatizada (`B3Scraper`)

A classe [`B3Scraper`](../src/bz_pricing/data/scrapers.py) automatiza a extração de séries históricas junto aos sistemas públicos da B3, ANBIMA, Banco Central do Brasil, Federal Reserve, ECB e Banxico:

```python
import datetime
from bz_pricing.data.scrapers import B3Scraper

scraper = B3Scraper(output_dir="dados/b3")
base_date = datetime.date(2026, 10, 30)

# Expectativas mensais de inflação do Relatório Focus (BCB Olinda API)
df_focus = scraper.download_focus_expectations(indicator="IPCA", base_date=base_date)

# Download de arquivo de taxas referenciais para auditoria
csv_path = scraper.download_taxas_referenciais(produto="PRE", data=base_date)
```

---

## 3. Construtor de Curvas e Interpoladores em Cython (`B3CurveBuilder` e `SimpleCurve`)

O módulo [`bz_pricing.core.curves`](../src/bz_pricing/core/curves.pyx) disponibiliza estruturas compiladas em Cython que podem ser instanciadas diretamente sobre vetores NumPy `float64`:

```python
import numpy as np
from bz_pricing.core.curves import SimpleCurve
from bz_pricing.core.interpolators import FlatForward252Interpolator

t_nodes = np.array([1.0, 21.0, 63.0, 126.0, 252.0], dtype=np.float64)
y_nodes = np.array([0.1365, 0.1370, 0.1380, 0.1395, 0.1410], dtype=np.float64)

# Instancia o interpolador em Cython (0 = Flat à esquerda, 1 = Flat Forward à direita)
interp = FlatForward252Interpolator(extrap_left=0, extrap_right=1)
curva = SimpleCurve(t_nodes, y_nodes, interp, round_decimals=6, round_mode="round")

print(f"Taxa interpolada em 100 DU: {curva.get_rate(100.0) * 100:.4f}% a.a.")
```

---

## 4. Precificação de Títulos Públicos Federais Baseada em Objetos de Curvas

Os módulos [`bz_pricing.instruments.tpf`](../src/bz_pricing/instruments/tpf.py) e [`bz_pricing.core.bonds`](../src/bz_pricing/core/bonds.pyx) implementam a precificação analítica, a cotação percentual e as medidas de sensibilidade (*Duration* de Macaulay, *Duration* Modificada e DV01) para **LTN**, **NTN-F**, **LFT**, **NTN-B**, **NTN-B Principal** e **NTN-C** conforme os padrões oficiais da ANBIMA e do Tesouro Nacional.

As funções de precificação operam sobre instâncias compiladas de [`Curve`](../src/bz_pricing/core/curves.pyx) e admitem curvas construídas diretamente a partir de nós customizados de mercado ou recuperadas via [`CurveManager`](../src/bz_pricing/curves/manager.py). Instrumentos que exigem indexadores compostos recebem múltiplas estruturas a termo de forma simultânea (multi-curvas).

```python
import datetime
import numpy as np
from bz_pricing.core.curves import SimpleCurve, FlatCurve
from bz_pricing.core.interpolators import FlatForward252Interpolator
from bz_pricing.instruments.tpf import LTN, NTNF, LFT, NTNB, NTNBPrincipal, NTNC

base_date = datetime.date(2026, 10, 30)

# 1. Construção da curva prefixada a partir de vértices empíricos
t_pre = np.array([1.0, 42.0, 102.0, 252.0], dtype=np.float64)
y_pre = np.array([0.1415, 0.135793, 0.137844, 0.1420], dtype=np.float64)
curve_pre = SimpleCurve(t_pre, y_pre, FlatForward252Interpolator(0, 0))

# 2. Construção da curva de juros reais (IPCA) a partir de vértices
t_real = np.array([10.0, 133.0, 252.0, 504.0], dtype=np.float64)
y_real = np.array([0.0450, 0.046629, 0.0520, 0.0580], dtype=np.float64)
curve_real = SimpleCurve(t_real, y_real, FlatForward252Interpolator(0, 0))

# 3. Precificação de LTN (Zero-Cupom Prefixado)
ltn = LTN(maturity_date=datetime.date(2027, 4, 1))
pu_ltn = ltn.present_value(base_date, curve_pre)
dur_ltn = ltn.duration(base_date, curve_pre)
dv01_ltn = ltn.dv01(base_date, curve_pre)
print(f"LTN 2027: PU = R$ {pu_ltn:.6f} | Duration = {dur_ltn:.1f} DU | DV01 = R$ {dv01_ltn:.6f}")

# 4. Precificação de NTN-F (Cupom Semestral de 10,00% a.a.)
ntnf = NTNF(maturity_date=datetime.date(2027, 1, 1), coupon_rate=0.10)
pu_ntnf = ntnf.present_value(base_date, curve_pre)
mod_ntnf = ntnf.modified_duration(base_date, curve_pre)
print(f"NTN-F 2027: PU = R$ {pu_ntnf:.6f} | Duration Modificada = {mod_ntnf:.4f} anos")

# 5. Precificação de LFT (Curva de Ágio/Deságio + VNA)
curve_agio = FlatCurve(0.000232)
lft = LFT(maturity_date=datetime.date(2027, 3, 1))
cot_lft = lft.cotacao(base_date, curve_agio)
pu_lft = lft.present_value(base_date, curve_agio, vna=20201.049997)
print(f"LFT 2027: Cotação = {cot_lft:.4f}% | PU = R$ {pu_lft:.6f}")

# 6. Precificação de NTN-B (Curva de Juros Reais + VNA Oficial)
ntnb = NTNB(maturity_date=datetime.date(2027, 5, 15), coupon_rate=0.06)
cot_ntnb = ntnb.cotacao(base_date, curve_real)
pu_ntnb = ntnb.present_value(base_date, curve_real, vna=4771.575461)
print(f"NTN-B 2027: Cotação = {cot_ntnb:.4f}% | PU = R$ {pu_ntnb:.6f}")

# 7. Precificação Multi-Curva da NTN-B (Curva Real + Curva de Inflação Implícita para Projeção de VNA)
curve_dic = FlatCurve(0.0450)
pu_ntnb_multi = ntnb.present_value(
    base_date=base_date,
    real_curve=curve_real,
    inflation_curve=curve_dic,
    base_vna=4750.00,
    base_vna_date=datetime.date(2026, 9, 15)
)
print(f"NTN-B 2027 Multi-Curva: PU = R$ {pu_ntnb_multi:.6f}")
```

### 4.1 Padrões de Arredondamento e Truncamento Normativos da ANBIMA

O motor de cálculo segue com rigor os critérios de arredondamento e truncamento padronizados pela ANBIMA, Banco Central do Brasil e Secretaria do Tesouro Nacional para liquidação no Selic:

| Variável | Operação e Precisão | Instrumentos Aplicáveis |
|---|---|---|
| **Taxa de Retorno (% a.a.)** | Truncada em 4 casas percentuais (`T-4`) | LTN, NTN-F, NTN-B, NTN-C, LFT |
| **Exponencial de Dias** | Truncado em 14 casas decimais (`T-14`): $\text{trunc}_{14}\left((1 + y)^{\frac{\text{DU}}{252}}\right)$ | LTN, NTN-F, NTN-B, NTN-C, LFT |
| **Juros Semestrais (%)** | Arredondado em 5 casas (`A-5`) para NTN-F ($4{,}88088\%$); arredondado em 6 casas (`A-6`) para NTN-B e NTN-C ($2{,}956301\%$) | NTN-F, NTN-B, NTN-C |
| **Fluxos Descontados** | Arredondados em 9 casas (`A-9`) para NTN-F; arredondados em 10 casas (`A-10`) para NTN-B e NTN-C | NTN-F, NTN-B, NTN-C |
| **Cotação (% do VNA)** | Truncada em 4 casas decimais (`T-4`) | NTN-B, NTN-B Principal, NTN-C, LFT |
| **Valor Nominal Atualizado (VNA)** | Truncado em 6 casas decimais (`T-6`) | NTN-B, NTN-B Principal, NTN-C, LFT |
| **Preço Unitário (PU)** | Truncado em 6 casas decimais (`T-6`) | Todos os TPFs |

### 4.2 Ajuste Paramétrico de Truncamento na Interpolação de Curvas

Todas as estruturas de curva compiladas em Cython ([`SimpleCurve`](../src/bz_pricing/core/curves.pyx), [`FlatCurve`](../src/bz_pricing/core/curves.pyx), [`ComplexCurve`](../src/bz_pricing/core/curves.pyx), [`InflationCurve`](../src/bz_pricing/core/curves.pyx)) suportam os parâmetros opcionais `round_decimals` e `round_mode`:

- Se omitidos ou quando `round_decimals = -1` (comportamento padrão), a curva preserva a representação analítica exata em ponto flutuante de dupla precisão nativa (64 bits), sem truncamentos intermediários.
- Se configurados explicitamente (por exemplo, `round_decimals=6, round_mode='trunc'` para taxas unitárias equivalentes a 4 casas decimais percentuais), a curva aplica o truncamento normativo a cada avaliação de `get_rate`.

```python
from bz_pricing.curves.builder import B3CurveBuilder
from bz_pricing.curves.manager import CurveManager

builder = B3CurveBuilder()

# Curva contínua nativa (round_decimals = -1)
curva_continua = builder.build_from_vertices(
    t_nodes=[1, 42, 102, 252],
    y_nodes=[0.1415, 0.135793, 0.137844, 0.1420]
)

# Curva com truncamento em 6 casas decimais unitárias (4 casas decimais em % a.a.)
curva_anbima = builder.build_from_vertices(
    t_nodes=[1, 42, 102, 252],
    y_nodes=[0.1415, 0.135793, 0.137844, 0.1420],
    round_decimals=6,
    round_mode='trunc'
)
```

### 4.3 Decomposição Fluxo a Fluxo e Tabela de Amortizações

Os instrumentos públicos federais suportam a decomposição fluxo a fluxo por meio do método `cashflow_table(base_date, ...)` e do argumento booleano `return_cashflows: bool = False` disponível em `present_value(...)`.

A chamada com `return_cashflows=False` preserva o retorno do escalar numérico do Preço Unitário (`pu`). Quando o parâmetro recebe o valor `True`, o método retorna a tupla `(pu, df_fluxos)`, na qual `df_fluxos` é uma estrutura `pandas.DataFrame` contendo os elementos determinísticos de cada pagamento:

- `data_fluxo`: Data de liquidação financeira de cada evento (ajustada para o dia útil subsequente caso a data nominal coincida com dia não útil).
- `du`: Prazo decorrido em dias úteis entre a data de referência e o fluxo.
- `tipo`: Classificação do evento (`Cupom`, `Principal` ou `Cupom + Principal`).
- `fluxo_nominal` ou `fluxo_cotação`: Valor contratual do evento (em reais para títulos prefixados ou em pontos percentuais para títulos com cotação indexada).
- `taxa` ou `taxa_real`: Taxa anual obtida na curva correspondente para o prazo `du`.
- `fator_desconto`: Fator financeiro composto truncated em 14 casas decimais ($T-14$), definido por $\text{trunc}_{14}\left((1 + y)^{\frac{\text{DU}}{252}}\right)$.
- `valor_presente`: Valor presente individualizado de cada fluxo.

```python
import datetime
from bz_pricing.core.curves import FlatCurve
from bz_pricing.instruments.tpf import NTNF, NTNB

base_date = datetime.date(2026, 10, 30)

# Extração de fluxos detalhados para NTN-F com vencimento em 2029
curva_pre = FlatCurve(0.149501)
ntnf = NTNF(maturity_date=datetime.date(2029, 1, 1), coupon_rate=0.10)
pu_ntnf, df_ntnf = ntnf.present_value(base_date, curva_pre, return_cashflows=True)

print(f"PU NTN-F: R$ {pu_ntnf:.6f}")
print(df_ntnf[["data_fluxo", "du", "tipo", "fluxo_nominal", "valor_presente"]])

# Extração de fluxos detalhados para NTN-B com vencimento em 2027
curva_real = FlatCurve(0.046629)
ntnb = NTNB(maturity_date=datetime.date(2027, 5, 15), coupon_rate=0.06)
pu_ntnb, df_ntnb = ntnb.present_value(
    base_date,
    curva_real,
    vna=4771.575461,
    return_cashflows=True
)

print(f"\nPU NTN-B: R$ {pu_ntnb:.6f}")
print(df_ntnb[["data_fluxo", "du", "tipo", "fluxo_cotação", "cotação_descontada", "valor_presente"]])
```

---

## 5. Extração e Precificação de Crédito Privado (`DebenturesClient` e `Debenture`)

O módulo de crédito privado fornece ferramentas integradas para coleta de eventos em fontes públicas e avaliação numérica de debêntures com amortizações periódicas e indexadores variados (IPCA, CDI e prefixados).

### 5.1 Cliente de Extração de Dados (`DebenturesClient`)

A classe [`DebenturesClient`](../src/bz_pricing/data/debentures.py) encapsula consultas aos dados estruturados do Sistema Nacional de Debêntures (SND / `debentures.com.br`) e aos arquivos oficiais de fechamento diário do mercado secundário da ANBIMA, dispondo de mecanismo automático de armazenamento em cache local no diretório `dados/debentures/`:

```python
import datetime
from bz_pricing.data.debentures import DebenturesClient

client = DebenturesClient(cache_dir="dados/debentures")

# 1. Agenda completa de eventos futuros (juros, amortizações e prêmios)
df_eventos = client.get_cashflows("VALEA0")

# 2. Cadastro de características da emissão (indexador, cupom emissor, datas base)
info_cadastrais = client.get_characteristics("VALEA0")

# 3. Histórico de PU Par, VNA e juros acumulados
df_pu_par = client.get_pu_history("VALEA0", datetime.date(2026, 10, 1), datetime.date(2026, 10, 30))

# 4. Arquivo oficial de mercado secundário de debêntures ANBIMA
df_mercado = client.get_market_prices(datetime.date(2026, 10, 30))
print(f"Total de debêntures com preços de mercado apurados: {len(df_mercado)}")
```

### 5.2 Avaliação Numérica e Decomposição de Fluxos (`Debenture`)

A classe [`Debenture`](../src/bz_pricing/instruments/credit.py) modela o instrumento de crédito privado e integra o histórico de liquidações para cômputo do saldo devedor e os fluxos contratuais futuros:

```python
import datetime
from bz_pricing.data.debentures import DebenturesClient
from bz_pricing.instruments.credit import Debenture

client = DebenturesClient()
base_date = datetime.date(2026, 10, 30)

# Instanciação automática a partir do código do ativo
deb = Debenture.from_ticker("VALEA0", client=client)

# 1. PU Par teórico na data de referência
vna_data = 1096.791130
pu_par = deb.pu_par(base_date, vna=vna_data)
print(f"VALEA0 - PU Par Teórico: R$ {pu_par:.6f}")

# 2. Precificação a mercado, duration, YTM e TIR por taxa indicativa ANBIMA
taxa_mercado = 0.066548
pu_mercado, df_fluxos = deb.present_value(
    base_date=base_date,
    discount_rate_or_curve=taxa_mercado,
    vna=vna_data,
    return_cashflows=True
)
dur_mercado = deb.duration(base_date=base_date, discount_rate_or_curve=taxa_mercado, vna=vna_data)
ytm_calc = deb.ytm(base_date, pu_mercado, vna=vna_data)

print(f"VALEA0 - PU Mercado:     R$ {pu_mercado:.6f}")
print(f"VALEA0 - Duration:       {dur_mercado:.2f} DU")
print(f"VALEA0 - YTM:            {ytm_calc * 100:.4f}% a.a.")
print(df_fluxos[["data_pagamento", "du", "evento", "fluxo_nominal", "fator_desconto", "valor_presente"]])
```

### 5.3 Fluxos Customizados, Securitização (`CRI`/`CRA`/`FIDC`), `LetraFinanceira` e `Z-Spread`

Todos os instrumentos de renda fixa (`LTN`, `NTNF`, `LFT`, `NTNB`, `NTNBPrincipal`, `NTNC`, `Debenture`, `CRI`, `CRA`, `CotaFIDC`, `LetraFinanceira`) aceitam fluxos próprios via `custom_cashflows` (ou `Debenture.from_cashflows` / `from_schedule`), e o módulo [`bz_pricing.core.vna_engine`](../src/bz_pricing/core/vna_engine.py) calcula o VNA a partir do VNE para qualquer indexador compatível:

```python
import datetime
from bz_pricing.core.vna_engine import calculate_lft_vna, calculate_ntnb_vna, calculate_vna_from_vne
from bz_pricing.instruments import NTNF, Debenture, CRI, CotaFIDC, LetraFinanceira, price_credit_portfolio

base_date = datetime.date(2026, 10, 30)

# 1. Cálculo de VNA a partir do VNE (online via BCB/Focus ou offline com séries do usuário)
vna_lft = calculate_lft_vna(base_date)
vna_ntnb = calculate_ntnb_vna(base_date, ipca_proj=0.001972)
vna_deb = calculate_vna_from_vne(
    vne=1000.0,
    base_date="2026-08-15",
    target_date="2026-10-15",
    indexador="IPCA",
    history={"2026-08-01": 0.0030, "2026-09-01": 0.0020},
    allow_download=False,
)

# 2. Precificação por fluxos customizados em TPF e Crédito Privado
ntnf_user = NTNF(custom_cashflows=[(datetime.date(2027, 1, 4), 50.0, "Cupom"), (datetime.date(2027, 7, 1), 1050.0, "Principal")])
deb_user = Debenture.from_cashflows("DEB-CUSTOM", [(datetime.date(2027, 4, 30), 1060.0, "Vencimento")], indexador="PRE")

# 3. Securitização, Z-Spread com gross-up tributário e avaliação em lote
cri = CRI.from_schedule("CRI-01", "2026-01-02", "2028-01-03", indexador="IPCA", coupon_rate=0.075, coupon_freq="mensal")
z_sprd = cri.z_spread(base_date, market_pu=1015.0, ettj_pre_rate_or_curve=0.065, vna=1035.0, incentivada=True)
df_carteira = price_credit_portfolio([{"instrument": deb_user, "discount_rate": 0.13, "quantity": 10}], base_date)
```

---

## 6. Derivativos Lineares (`bz_pricing.instruments.derivatives`)

O módulo [`derivatives.py`](../src/bz_pricing/instruments/derivatives.py) implementa futuros listados (`DI1Future`, `DDIFuture`, `DAPFuture`, `DOLFuture`, `INDFuture`), *swaps* (`SwapDIxPRE`, `SwapDIxIPCA`) e contratos a termo de moeda (`NDFContract`), além de precificação vetorizada em lote (`price_di1_portfolio`):

```python
import datetime
from bz_pricing.instruments import DI1Future, SwapDIxPRE, NDFContract

base_date = datetime.date(2026, 10, 30)
di1 = DI1Future("DI1F28", "2028-01-03")
pu_di1 = di1.pu(base_date, 0.1385)
dv01_di1 = di1.dv01(base_date, 0.1385)

swap = SwapDIxPRE("SWAP-PRE", "2028-01-03", fixed_rate=0.1350, notional=1_000_000.0)
ndf = NDFContract("NDF-USD", "2027-01-29", forward_rate=5.65, notional_usd=100_000.0)
```

---

## 7. Opções e Superfícies de Volatilidade (`options` e `volatility`)

Os módulos [`options.py`](../src/bz_pricing/instruments/options.py) e [`surface.py`](../src/bz_pricing/volatility/surface.py) disponibilizam precificação analítica e numérica em Cython (`EquityOption`, `FXOption`, `IDIOption`, `DI1Option`, `CorradoSuOption`, `Swaption`, `CapFloor`, `BarrierOption`, `AsianOption`, além dos lotes `price_options_portfolio`, `price_fx_options_portfolio`, `price_idi_options_portfolio` e `price_di1_options_portfolio`), parametrizações de *smiles* (`SVISlice`, `SABRSlice`), superfície `VolatilitySurface`, auditoria de não arbitragem, densidade neutra ao risco (`breeden_litzenberger_density`), volatilidade local (`dupire_local_vol_grid`), validação *out-of-sample* (`validate_slice_out_of_sample`) e leitura de séries reais B3 (`B3DataLoader.load_options`):

```python
import datetime
from bz_pricing.instruments import EquityOption, FXOption
from bz_pricing.volatility import SVISlice, VolatilitySurface

base_date = datetime.date(2026, 10, 30)
opt_am = EquityOption("PETRO38", "2027-01-04", 38.0, "PUT", "AMERICAN")
rm = opt_am.risk_metrics(base_date, spot=37.5, pre_rate_or_curve=0.14, vol=0.30)

sl = SVISlice(a=0.015, b=0.04, rho=-0.30, m=0.0, sigma=0.12, T=63 / 252.0)
surf = VolatilitySurface.from_svi_slices({63: sl})
arb = surf.check_no_arbitrage(forward_by_du=5.55, r_cont_by_du=0.13)
bl = surf.breeden_litzenberger_density(63, forward=5.55, r_cont=0.13)
```

