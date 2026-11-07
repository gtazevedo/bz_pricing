# Guia de Uso das Funções e Módulos (API Reference)

Este guia apresenta exemplos práticos de como utilizar as principais classes e funções do pacote `bz-pricing`.

---

## 1. Orquestrador de Curvas: `CurveManager`

A classe [`CurveManager`](../src/bz_pricing/curves/manager.py) é o ponto de entrada principal para a obtenção de qualquer curva de juros onshore ou offshore. Ela cuida automaticamente do download, carregamento de dados e cálculo de curvas derivadas.

```python
import datetime
from bz_pricing.curves.manager import CurveManager

# Inicializa o gerenciador
manager = CurveManager()
base_date = datetime.date(2026, 9, 4)

# 1. Curva DI x PRE
curva_pre = manager.get_curve("PRE", base_date)

# Consulta de taxas em dias úteis (DU)
taxa_21_du = curva_pre.get_rate(21.0)    # Taxa para 1 mês útil (decimal)
taxa_252_du = curva_pre.get_rate(252.0)  # Taxa para 1 ano útil (decimal)

print(f"PRE 21 DU: {taxa_21_du * 100:.4f}% a.a.")
print(f"PRE 252 DU: {taxa_252_du * 100:.4f}% a.a.")

### 1.1 Fornecimento de Dados Externos (`data`) e Tratamento de Erros

Todas as funções de cálculo aceitam dados externos informados pelo usuário. Quando o parâmetro de dados é fornecido, a curva é calculada diretamente a partir da série informada, sem depender de arquivos locais ou downloads da B3.

Caso os dados **não** sejam fornecidos, o sistema busca automaticamente nos arquivos locais ou realiza o download na B3. Se os dados não estiverem disponíveis na B3 para a data solicitada, um erro amigável [`DataUnavailableError`](../src/bz_pricing/curves/manager.py) é lançado.

#### Exemplo 1: Passagem Posicional `get_curve(curve_name, base_data)`
```python
import pandas as pd
from bz_pricing.curves.manager import CurveManager

manager = CurveManager()

# Série customizada de prazos (DU) e taxas (% a.a.)
meus_dados = pd.DataFrame({
    'DU': [1, 21, 42, 63, 126, 252],
    'rate': [13.75, 13.80, 13.85, 13.90, 14.00, 14.20]
})

# Passagem direta dos dados como 2º argumento posicional
curva_dol = manager.get_curve("DOL", meus_dados)
print("DOL 21 DU:", curva_dol.get_rate(21) * 100, "% a.a.")
```

#### Exemplo 2: Formatos Flexíveis (Series, Dicts, DataFrames com Vencimentos)
```python
# Via pd.Series
serie_pre = pd.Series([13.75, 13.80, 13.90], index=[1, 21, 63])
curva_pre = manager.get_curve("PRE", data=serie_pre)

# Via dicionário de nós {DU: taxa}
dict_dic = {1: 13.75, 21: 13.80, 42: 13.85}
curva_dic = manager.get_curve("DIC", data=dict_dic)

# Futuros de Ações com dataset customizado
df_petr = pd.DataFrame({
    'XprtnDt': ['2026-11-16', '2026-12-18'],
    'AdjstdQt': [53.10, 53.60]
})
fut_petr, cy_petr = manager.get_stock_future_curve("PETRP", spot_price=52.71, data=df_petr)
```

#### Exemplo 3: Tratamento de Dados ou Data Indisponível
```python
from bz_pricing.curves.manager import DataUnavailableError

try:
    # Tentativa de buscar data que não existe no histórico disponibilizado pela B3
    curva = manager.get_curve("DOL", datetime.date(1991, 1, 1))
except DataUnavailableError as e:
    print(f"Alerta amigável capturado: {e}")
    # Saída: Dado ou data indisponível: não foi possível obter os dados para a curva 'DOL' na data 01/01/1991.
    # Forneça os dados através do parâmetro 'data' ou utilize uma data disponível no histórico da B3.
```

# 2. Curvas de Inflação (IPCA e IGP-M)
curva_dic = manager.get_curve("DIC", base_date) # Cupom Sujo IPCA
curva_dpl = manager.get_curve("DPL", base_date) # Cupom Limpo IPCA (DAP)
curva_dim = manager.get_curve("DIM", base_date) # Cupom Sujo IGP-M

print(f"DIC 6 DU: {curva_dic.get_rate(6.0) * 100:.2f}% a.a.")
print(f"DIM 17 DU: {curva_dim.get_rate(17.0) * 100:.2f}% a.a.")
print(f"DIM Longo Prazo (8000 DU): {curva_dim.get_rate(8000.0) * 100:.2f}% a.a.")

# 3. Curvas de SOFR (Internacional)
curva_sfr = manager.get_curve("SFR", base_date) # Futuros B3
curva_zus = manager.get_curve("ZUS", base_date) # Swaps Refinitiv (Oficial B3 LIB)

print(f"SFR Futuros 30 DC: {curva_sfr.get_rate(30.0) * 100:.2f}% a.a.")
print(f"ZUS Swaps 360 DC: {curva_zus.get_rate(360.0) * 100:.2f}% a.a.")
```

---

## 2. Coleta e Carregamento de Dados: `B3DataLoader` e `B3Scraper`

### 2.1 [`B3DataLoader`](../src/bz_pricing/data/loaders.py)
Responsável por carregar arquivos locais brutos (CSVs da B3 e ANBIMA):

```python
import datetime
from bz_pricing.data.loaders import B3DataLoader

loader = B3DataLoader(data_dir="dados/b3")
data = datetime.date(2026, 10, 30)

# Carrega contratos futuros de DI1 negociados
df_di1 = loader.get_curve_instruments(data, asst="DI1")

# Carrega títulos públicos do mercado secundário da ANBIMA (ex: LTN ou NTN-B)
df_ltn = loader.get_anbima_bonds(data, bond_type="LTN")

# Carrega taxas referenciais oficiais da B3 (ex: LIB, PRE, DIC, DIM, DOL, INP)
df_lib = loader.load_taxas_referenciais("LIB", data)
```

### 2.2 [`B3Scraper`](../src/bz_pricing/data/scrapers.py)
Executa o download de cotações, relatórios Focus e taxas da B3, ANBIMA e Banco Central:

```python
import datetime
from bz_pricing.data.scrapers import B3Scraper

scraper = B3Scraper(output_dir="dados/b3")
data = datetime.date(2026, 10, 30)

# Download do Relatório Focus (Expectativas mensais de IPCA ou IGP-M com cache local)
df_focus = scraper.download_focus_expectations(indicator="IPCA", base_date=data)

# Download de taxas referenciais diárias da B3 via Playwright headless
caminho_csv = scraper.download_taxas_referenciais(produto="PRE", data=data)
```

---

## 3. Construtor Customizado de Curvas: [`B3CurveBuilder`](../src/bz_pricing/curves/builder.py)

Permite construir curvas com parâmetros customizados de interpolação, suavização ou nós manuais:

```python
import datetime
import numpy as np
from bz_pricing.curves.builder import B3CurveBuilder
from bz_pricing.core.interpolators import FlatForward252Interpolator, Linear360Interpolator

builder = B3CurveBuilder(calendar_name="B3")
base_date = datetime.date(2026, 10, 30)

# Exemplo: Construir curva personalizada com nós definidos
t_nodes = np.array([1.0, 21.0, 42.0, 126.0, 252.0], dtype=np.float64)
y_nodes = np.array([0.1365, 0.1360, 0.1358, 0.1355, 0.1350], dtype=np.float64)

# Instancia o interpolador Cython (0 = Flat à esquerda, 1 = Flat Forward à direita)
interp = FlatForward252Interpolator(0, 1)

from bz_pricing.core.curves import SimpleCurve
curva_custom = SimpleCurve(t_nodes, y_nodes, interp, round_decimals=4)

print(f"Taxa calculada em 100 DU: {curva_custom.get_rate(100.0) * 100:.2f}% a.a.")
```

---

## 4. Precificação de Títulos de Renda Fixa: `bz_pricing.bonds`

Precifica títulos públicos federais seguindo a metodologia oficial da Secretaria do Tesouro Nacional (STN) e da ANBIMA:

```python
import datetime
from bz_pricing.bonds.fixed_rate import FixedRateBond # LTN e NTN-F
from bz_pricing.bonds.inflation import InflationBond   # NTN-B

base_date = datetime.date(2026, 10, 30)
vencimento = datetime.date(2029, 1, 1)

# 1. Precificação de LTN (Título Pré Zero Cupom com VNA = R$ 1.000,00)
ltn = FixedRateBond(
    issue_date=datetime.date(2025, 1, 1),
    maturity_date=vencimento,
    coupon_rate=0.0 # Zero cupom
)
# Calcula o Preço Unitário (PU) para uma taxa de 13.50% a.a.
pu_ltn = ltn.price(base_date=base_date, yield_rate=0.1350)
print(f"PU LTN: R$ {pu_ltn:.6f}")

# 2. Precificação de NTN-B (Cupom Semestral de 6% a.a. indexado ao IPCA)
ntnb = InflationBond(
    issue_date=datetime.date(2024, 8, 15),
    maturity_date=datetime.date(2035, 5, 15),
    coupon_rate=0.06 # 6.00% a.a. cupom semestral
)
vna_atual = 4325.68 # Valor Nominal Atualizado pelo IPCA
pu_ntnb = ntnb.price(base_date=base_date, yield_rate=0.0725, vna=vna_atual)
print(f"PU NTN-B: R$ {pu_ntnb:.6f}")
```

---

## 5. Notebook de Análise Completa

Para visualizar todas as curvas plotadas e confrontadas contra os dados oficiais da B3, execute o notebook:
- [`notebooks/analise_curvas.ipynb`](../notebooks/analise_curvas.ipynb)

