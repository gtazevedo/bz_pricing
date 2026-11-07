# Guia de Referência da API (`bz-pricing`)

Este documento detalha as interfaces programáticas dos módulos de gerenciamento de curvas, construção algébrica, coleta de dados e precificação de instrumentos de renda fixa do pacote `bz-pricing`.

---

## 1. Orquestrador de Estruturas a Termo (`CurveManager`)

A classe [`CurveManager`](../src/bz_pricing/curves/manager.py) centraliza a instanciação de curvas *onshore* e *offshore*, gerenciando a validação de insumos fornecidos pelo usuário (`allow_download=False` por padrão) ou a resolução automática a partir de bases locais e provedores públicos (`allow_download=True`).

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

## 4. Precificação de Títulos Públicos Federais (`TPF` e `BondBuilder`)

Os módulos [`bz_pricing.instruments.tpf`](../src/bz_pricing/instruments/tpf.py) e [`bz_pricing.core.bonds`](../src/bz_pricing/core/bonds.pyx) implementam a precificação determinística, o cálculo de Valor Nominal Atualizado (`VNAEngine`), cotação em base 100 e medidas de sensibilidade (*Duration* de Macaulay, *Duration* Modificada, DV01 e Convexidade) para **LTN**, **NTN-F**, **NTN-B** e **LFT**:

```python
import datetime
from bz_pricing.instruments.tpf import TPF

base_date = datetime.date(2026, 9, 4)

# 1. Precificação de LTN (Zero Cupom Prefixado, Valor Nominal = R$ 1.000,00)
ltn = TPF(
    bond_type="LTN",
    maturity_date=datetime.date(2027, 1, 1),
    base_date=base_date
)
pu_ltn = ltn.price_from_ytm(ytm=0.1350)
metrics_ltn = ltn.risk_metrics(ytm=0.1350)

print(f"PU LTN: R$ {pu_ltn:.6f} | Macaulay Duration: {metrics_ltn['macaulay_duration']:.4f} anos")

# 2. Precificação de NTN-B (Cupom Semestral de 6,00% a.a. sobre VNA de IPCA)
ntnb = TPF(
    bond_type="NTN-B",
    maturity_date=datetime.date(2035, 5, 15),
    base_date=base_date,
    vna_override=4325.684120
)
pu_ntnb = ntnb.price_from_ytm(ytm=0.0645)
print(f"PU NTN-B: R$ {pu_ntnb:.6f}")
```
