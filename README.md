# bz-pricing

> **Motor Quantitativo em Cython e Python para Construção de Estruturas a Termo de Taxas de Juros, Moedas, Inflação e Precificação de Renda Fixa Onshore e Offshore**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Cython 3.0+](https://img.shields.io/badge/cython-3.0%2B-orange.svg)](https://cython.org/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-purple.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://docs.pytest.org/)

O **bz-pricing** compreende uma biblioteca computacional e quantitativa concebida em **Cython** e **Python** para a modelagem de estruturas a termo de taxas de juros, inflação implícita, cupons cambiais, moedas a termo, índices de ações e precificação de títulos públicos federais nos mercados doméstico (*onshore*) e internacional (*offshore*).

A arquitetura do pacote foi projetada para permitir que pesquisadores, gestores e mesas de operações construam e interpolem curvas a partir de seus próprios vértices ou dos insumos primários de cada componente (contratos futuros, expectativas de inflação, fatores de desconto de *swaps* ou cotações à vista) de modo integralmente *offline*, além de dispor de coletores automatizados para validação empírica frente às curvas referenciais divulgadas pela B3, ANBIMA e Banco Central do Brasil.

---

## 1. Arquitetura Computacional e Escopo Funcional

O pacote estrutura-se em três camadas desacopladas que separam a ingestão de dados, a orquestração algébrica dos vértices e o núcleo de avaliação numérica compilado em C/Cython:

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                 Camada 1: Ingestão ou Injeção de Vértices                    │
│   Insumos do Usuário (DataFrame / Series / Dict de Vértices e Componentes)   │
│   Coletores Públicos Opcionais: B3 (SPRD / Taxas), ANBIMA, BCB (SGS / Focus) │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ Vértices Esparsos Saneados (DU / DC)
┌──────────────────────────────────────▼───────────────────────────────────────┐
│              Camada 2: Orquestração e Construção de Estruturas               │
│   CurveManager & B3CurveBuilder: Resolução de V1, Paridades Cambiais,        │
│   Degraus Discretos de Inflação, Redutor Regulatório da TR e Carrego (CYI)   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ Vetores Contíguos C (double[:])
┌──────────────────────────────────────▼───────────────────────────────────────┐
│           Camada 3: Núcleo Numérico em Cython (Interpolação e Bonds)         │
│   Interpoladores: FlatForward252, FlatForward252Linear, Linear360, Price     │
│   Precificação Determinística de TPF (LTN, NTN-F, NTN-B, LFT), VNA e Risco   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Famílias de Estruturas a Termo e Instrumentos Suportados

1. **Juros Prefixados Domésticos (`BUS/252`):** Curva DI x Pré (`PRE` sobre futuros `DI1`), Curva Selic x Pré (`SELIC_PRE` / `SLP` sobre futuros `OC1`), Curva de Títulos Públicos Prefixados (`LTN`) e Taxa de Juros de Longo Prazo (`TJLP`);
2. **Taxas Regulatórias Derivadas (`BUS/252`):** Taxa Básica Financeira (`TFP` / `TBF_PRE`), Taxa Referencial (`TP` / `TR_PRE`) com aplicação do redutor paramétrico por faixas e Curva DI x TR (`TR`);
3. **Estruturas de Inflação (`BUS/252`):** Cupom Sujo de IPCA (`DIC`), Cupom Sujo de IGP-M (`DIM`) e Cupom Limpo de IPCA (`DPL` sobre contratos futuros `DAP`);
4. **Cupons Cambiais, Spreads e Moedas a Termo (`BUS/252` e `ACT/360`):** Cupom Cambial Sujo (`DOL` sobre futuros `DDI`), Cupom Cambial Limpo (`DOC` e `DCL`), Cupons Cruzados de Euro (`EUC`) e Iene (`YCL`), Spreads de Taxas Offshore (`SDE`, `SGP`, `SMX`, `SYD`) e Estruturas a Termo de Câmbio (`PTX`, `EUR` e `JPY`);
5. **Curvas Internacionais Offshore (`ACT/360`):** Curvas a termo de futuros de taxas *overnight* (`SFR` para SOFR, `EST` para €STR, `TIE` para F-TIIE) e curvas zero cupom de *Overnight Indexed Swaps* (`ZUS` / `LIB`, `ZEU` e `ZMX`);
6. **Renda Variável e Convenience Yield (`BUS/252`):** Estruturas de preços futuros de índices (`INP` para Ibovespa, `BRP` para IBrX-50, `SML`, `MBR`, `VIX`), futuros de ações individuais e curvas de custo de carrego (`CYI`, `CYM`, `CYX`);
7. **Títulos Públicos Federais (`TPF`):** Precificação exata (PU, cotação em base 100, VNA projetado e truncado em 6 casas decimais) e métricas de sensibilidade (*Duration* de Macaulay, *Duration* Modificada, DV01 e Convexidade) para **LTN**, **NTN-F**, **NTN-B** e **LFT**.

---

## 2. Fundamentação Metodológica e Formulações Matemáticas

As formulações completas encontram-se detalhadas em [`docs/metodologia_curvas.md`](docs/metodologia_curvas.md). Apresentam-se a seguir as equações centrais implementadas no motor em Cython.

### 2.1 Operadores de Interpolação em Cython (`bz_pricing.core.interpolators`)

- **Taxa a Termo Exponencial em Base 252 (`FlatForward252Interpolator`):** Para um prazo $\text{DU} \in [\text{DU}_k, \text{DU}_{k+1}]$ entre dois vértices conhecidos $(\text{DU}_k, y_k)$ e $(\text{DU}_{k+1}, y_{k+1})$:

$$y(\text{DU}) = \left[ \left(1 + y_k\right)^{\frac{\text{DU}_k}{252}} \times \left( \frac{\left(1 + y_{k+1}\right)^{\frac{\text{DU}_{k+1}}{252}}}{\left(1 + y_k\right)^{\frac{\text{DU}_k}{252}}} \right)^{\frac{\text{DU} - \text{DU}_k}{\text{DU}_{k+1} - \text{DU}_k}} \right]^{\frac{252}{\text{DU}}} - 1$$

- **Taxa a Termo Exponencial sobre Regime Linear (`FlatForward252LinearInterpolator`):** Aplicada às curvas de cupom cambial (`DOL`, `DOC`, `DCL`), nas quais o fator de capitalização é linear em dias corridos ($\text{DC}/360$) e a interpolação ocorre exponencialmente em dias úteis ($\text{DU}$):

$$F_k = 1 + y_k \times \frac{\text{DC}_k}{360}, \quad F(\text{DU}) = F_k \times \left( \frac{F_{k+1}}{F_k} \right)^{\frac{\text{DU} - \text{DU}_k}{\text{DU}_{k+1} - \text{DU}_k}}, \quad y(\text{DU}, \text{DC}) = \left( F(\text{DU}) - 1 \right) \times \frac{360}{\text{DC}}$$

- **Interpolação Linear em Base 360 (`Linear360Interpolator`):** Aplicada às estruturas *offshore* (`SFR`, `EST`, `TIE`, `ZUS`, `ZEU`, `ZMX`, `SDE`):

$$y(\text{DC}) = \left[ \frac{y_k \cdot \text{DC}_k}{360} + \left( \frac{y_{k+1} \cdot \text{DC}_{k+1}}{360} - \frac{y_k \cdot \text{DC}_k}{360} \right) \times \frac{\text{DC} - \text{DC}_k}{\text{DC}_{k+1} - \text{DC}_k} \right] \times \frac{360}{\text{DC}}$$

- **Interpolação Exponencial de Preços (`PriceInterpolator`):** Aplicada às estruturas de preços futuros de índices e ações (`INP`, `BRP` e ações individuais):

$$P(\text{DU}) = P_k \times \left( \frac{P_{k+1}}{P_k} \right)^{\frac{\text{DU} - \text{DU}_k}{\text{DU}_{k+1} - \text{DU}_k}}$$

### 2.2 Estruturas de Inflação: Cupom Sujo (`DIC`, `DIM`) e Cupom Limpo (`DPL`)

- **Degraus Discretos de Divulgação no Cupom Sujo (`DIC` e `DIM`):** Nos primeiros 12 meses, o número acumulado de divulgações do índice permanece constante entre duas datas consecutivas de apuração ($k_i$). A expectativa acumulada $E_t[\text{I}_{k_i}]$ extraída no vértice $\text{DU}_{k_i}$ define a taxa de cupom sujo $\text{Cp}_{p,t}$ em todos os dias úteis $\text{DU}_p$ pertencentes ao patamar $[\text{DU}_{\text{início}, k_i}, \text{DU}_{k_i}]$:

$$E_t[\text{I}_{k_i}] = \frac{\left(1 + \text{PRE}_{k_i,t}\right)^{\frac{\text{DU}_{k_i}}{252}}}{\left(1 + \text{Cp}_{k_i,t}\right)^{\frac{\text{DU}_{k_i}}{252}}} - 1, \qquad \text{Cp}_{p,t} = \left[ \frac{\left(1 + \text{PRE}_{p,t}\right)^{\frac{\text{DU}_p}{252}}}{1 + E_t[\text{I}_{k_i}]} \right]^{\frac{252}{\text{DU}_p}} - 1$$

- **Inflação Implícita e Cupom Limpo (`DPL`):** O vértice de 1 dia útil ($\text{DU} = 1$) desconta da taxa $\text{PRE}_{1,t}$ a prévia corrente $E_t[\text{IPCA}_0]$ proporcional ao número de dias úteis entre os dias 15 anterior e posterior ($\text{DU}_{15a,15p}$). Nos vencimentos de `DAP`, extrai-se a inflação implícita $\pi_{n,t}$, interpolada via `FlatForward252Interpolator(0, 1)` para recompor o cupom limpo:

$$\text{DPL}_{1,t} = \left[ \frac{\left(1 + \text{PRE}_{1,t}\right)^{\frac{1}{252}}}{\left(1 + E_t[\text{IPCA}_0]\right)^{\frac{1}{\text{DU}_{15a,15p}}}} \right]^{252} - 1, \qquad \text{DPL}(\text{DU}) = \frac{1 + \text{PRE}(\text{DU})}{1 + \pi(\text{DU})} - 1$$

### 2.3 Cupons Cambiais (`DOC`, `EUC`), Moedas a Termo (`PTX`, `EUR`) e Convenience Yield (`CYI`)

- **Ajuste de Cupom Limpo (`DOC`) e Paridade Cruzada (`EUC`):**

$$\text{DOC}_n = \left[ \left( 1 + \text{DDI}_n \times \frac{\text{DC}_n}{360} \right) \times \frac{\text{DolCL}_{t_0}}{\text{Ptax}_{t-1}} - 1 \right] \times \frac{360}{\text{DC}_n}$$

$$\text{EUC}_n = \left[ \left( 1 + \text{DOC}_n \times \frac{\text{DC}_n}{360} \right) \times \left( 1 + \text{SDE}_n \times \frac{\text{DC}_n}{360} \right) - 1 \right] \times \frac{360}{\text{DC}_n}$$

- **Moedas a Termo (`PTX`, `EUR`, `JPY`) e Custo de Carrego de Índices/Ações (`CYI`):**

$$\text{Fwd}_n = \text{Spot}_{\text{ref}} \times \frac{\left(1 + \text{PRE}_n\right)^{\frac{\text{DU}_n}{252}}}{1 + \text{TC}_n \times \frac{\text{DC}_n}{360}}, \qquad \text{TxCY}_n = \left[ \frac{\text{Spot}_t \times \left(1 + \text{PRE}_n\right)^{\frac{\text{DU}_n}{252}}}{\text{Fut}_n} \right]^{\frac{252}{\text{DU}_n}} - 1$$

---

## 3. Estrutura do Pacote

```text
bz_pricing/
├── docs/                                # Documentação Técnica e Metodológica
│   ├── README.md                        # Índice geral da documentação
│   ├── metodologia_curvas.md            # Formulações matemáticas e referências normativas
│   ├── guia_uso_api.md                  # Exemplos de uso da API (offline e online)
│   └── calendarios_convencoes.md        # Regras de contagem de dias e extrapolações
│
├── src/bz_pricing/
│   ├── core/                            # Extensões compiladas em Cython / C
│   │   ├── conventions.pxd              # Enums em C para DayCountConvention e CompoundingType
│   │   ├── interpolators.pxd / .pyx     # Interpoladores nativos em Cython (nogil)
│   │   ├── curves.pxd / .pyx            # Estruturas de curvas compiladas (SimpleCurve, ConvenienceYieldCurve, etc.)
│   │   ├── bonds.pxd / .pyx             # Motor Cython de fluxos de caixa, PU, YTM e métricas de risco
│   │   ├── calendars.py                 # Integração com calendários de dias úteis (bizdays)
│   │   └── vna_engine.py                # Cálculo e projeção de Valor Nominal Atualizado (IPCA / Selic)
│   │
│   ├── curves/                          # Camada de Orquestração e Construção de Curvas
│   │   ├── configs.py                   # Especificações de interpolação, arredondamento e convenções por curva
│   │   ├── builder.py                   # Construtor algébrico (B3CurveBuilder, CrossCouponCurve, CurrencyForwardCurve)
│   │   ├── manager.py                   # Fachada central de resolução de insumos (CurveManager)
│   │   └── exceptions.py                # Tratamento estruturado de exceções (DataUnavailableError)
│   │
│   ├── instruments/                     # Instrumentos de Renda Fixa Soberana
│   │   └── tpf.py                       # Precificação de LTN, NTN-F, NTN-B e LFT
│   │
│   ├── bonds/                           # Construtores auxiliares de carteiras de títulos
│   │   └── builder.py                   # BondBuilder integrado a dados secundários da ANBIMA
│   │
│   └── data/                            # Ingestão e Extração de Dados Públicos
│       ├── loaders.py                   # Leitura e filtragem de arquivos SPRD, ANBIMA e taxas referenciais
│       └── scrapers.py                  # Coletores automatizados (B3, ANBIMA, BCB SGS/Olinda, NY Fed, ECB, Banxico)
│
├── notebooks/                           # Cadernos de Validação Empírica
│   ├── analise_curvas.ipynb             # Auditoria visual e numérica de todas as famílias de curvas
│   └── validacao_tpfs_anbima.ipynb      # Validação de PU e métricas de risco contra o mercado secundário ANBIMA
│
└── tests/                               # Suíte Automatizada de Testes Unitários e de Integração
    ├── test_interpolators.py            # Testes dos operadores de interpolação e extrapolação em Cython
    ├── test_curves.py                   # Testes de aderência por vértices e construção 100% offline por componentes
    └── test_bonds.py                    # Testes de precificação de títulos públicos federais
```

---

## 4. Instalação e Compilação das Extensões Cython

### 4.1 Pré-requisitos
- Python >= 3.10 e < 3.13
- Compilador C/C++ compatível (MSVC Build Tools no Windows ou GCC/Clang no Linux/macOS)
- Poetry (`pip install poetry`)

### 4.2 Procedimento de Instalação

```bash
# 1. Clonagem do repositório
git clone https://github.com/gtazevedo/bz_pricing.git
cd bz_pricing

# 2. Instalação das dependências e compilação dos módulos Cython via Poetry
poetry install
poetry run python setup.py build_ext --inplace

# 3. Execução da suíte de testes automatizados
poetry run pytest -v
```

---

## 5. Exemplo de Uso (Construção por Vértices e Componentes)

O pacote permite tanto a construção *offline* a partir de vértices informados pelo usuário (`allow_download=False`) quanto a resolução automatizada de insumos de mercado (`allow_download=True`):

```python
import datetime
import pandas as pd
from bz_pricing.curves.manager import CurveManager

# 1. Construção 100% offline passando vértices e componentes
manager_offline = CurveManager(allow_download=False)
base_date = datetime.date(2026, 10, 30)

df_pre = pd.DataFrame({
    "DU": [1, 21, 63, 126, 252, 504],
    "rate": [13.65, 13.70, 13.80, 13.90, 14.00, 14.10]
})
df_dap = pd.DataFrame({
    "DU": [14, 135, 252, 504],
    "rate": [9.62, 6.80, 7.65, 7.80]
})

curva_pre = manager_offline.get_curve("PRE", base_date, data=df_pre)
curva_dpl = manager_offline.get_curve("DPL", base_date, pre_curve=curva_pre, dap_df=df_dap, eipca_0=0.0045)

print(f"PRE (180 DU): {curva_pre.get_rate(180.0) * 100:.4f}% a.a.")
print(f"DPL (180 DU): {curva_dpl.get_rate(180.0) * 100:.4f}% a.a.")

# 2. Resolução com dados de mercado locais/baixados (allow_download=True)
manager_b3 = CurveManager(allow_download=True)
curva_euc = manager_b3.get_curve("EUC", base_date)
curva_inp = manager_b3.get_curve("INP", base_date)
curva_cyi = manager_b3.get_curve("CYI", base_date)

print(f"Cupom de Euro EUC (252 DU): {curva_euc.get_rate(252.0) * 100:.4f}% a.a.")
print(f"Ibovespa Futuro INP (31 DU): {curva_inp.get_rate(31.0):.2f} pts | CYI: {curva_cyi.get_rate(31.0) * 100:.2f}% a.a.")
```

---

## 6. Validação Empírica por Vértices

Para validar a exatidão dos interpoladores em Cython e das formulações algébricas sem recorrer à cópia de grades diárias pré-interpoladas, a suíte de testes (`tests/test_curves.py`) e o caderno [`notebooks/analise_curvas.ipynb`](notebooks/analise_curvas.ipynb) reconstroem cada curva estritamente a partir de seus vértices esparsos (contratos futuros, nós de *swaps* ou datas de divulgação de inflação) e confrontam os valores interpolados contra todos os pontos divulgados pela B3 na data-base `30/10/2026`:

| Código | Estrutura a Termo | Nº de Vértices Base | Total de Pontos Avaliados | Erro Médio Absoluto (MAE) | Erro Máximo Absoluto |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `PRE` | Taxa Prefixada (DI x Pré — Futuros `DI1`) | 39 | 262 | 0,0009% a.a. | 0,0100% a.a. |
| `SLP` | Selic x Pré (Futuros `OC1`) | 19 | 242 | 0,0006% a.a. | 0,0100% a.a. |
| `DIC` | Cupom Sujo de IPCA (Degraus + Vértices Longos) | 150 | 262 | 0,0009% a.a. | 0,0100% a.a. |
| `DPL` | Cupom Limpo de IPCA (Futuros `DAP`) | 20 | 262 | 0,0017% a.a. | 0,0100% a.a. |
| `DIM` | Cupom Sujo de IGP-M (Degraus + Vértices Longos) | 145 | 262 | 0,0011% a.a. | 0,0100% a.a. |
| `DOL` | Cupom Cambial Sujo (Futuros `DDI`) | 45 | 262 | 0,0008% a.a. | 0,0100% a.a. |
| `DOC` | Cupom Cambial Limpo (V1 + Futuros `DDI`) | 45 | 262 | 0,0010% a.a. | 0,0100% a.a. |
| `EUC` | Cupom Cambial de Euro (`DOC` + `SDE`) | 94 | 262 | 0,0014% a.a. | 0,0100% a.a. |
| `SDE` | Spread Euro x Dólar Offshore (`TEU - TUS`) | 65 | 262 | 0,0006% a.a. | 0,0100% a.a. |
| `ZUS` | Zero Cupom SOFR Swaps Offshore (`LIB`) | 73 | 278 | 0,0003% a.a. | 0,0100% a.a. |
| `PTX` | Taxa de Câmbio a Termo Reais x Dólar | 45 | 262 | R\$ 0,0001 | R\$ 0,0011 |
| `EUR` | Taxa de Câmbio a Termo Reais x Euro | 94 | 262 | R\$ 0,0005 | R\$ 0,0021 |
| `INP` | Índice Bovespa Futuro (Spot + Futuros `IND`) | 15 | 259 | 0,0000 pt | 0,0000 pt |
| `BRP` | Índice IBrX-50 Futuro (Spot + Futuros `BRI`) | 3 | 250 | 0,0000 pt | 0,0000 pt |

---

## 7. Referências Normativas e Bibliográficas

- **B3 S.A. – Brasil, Bolsa, Balcão.** *Manual de Curvas da B3: Metodologia de Apuração das Estruturas a Termo de Taxas de Juros, Moedas e Índices*. São Paulo: B3, 2024. Disponível em: <https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-de-derivativos/precos-referenciais/manuais-de-curvas/>.
- **B3 S.A. – Brasil, Bolsa, Balcão.** *Preços Referenciais – Taxas Referenciais BM&FBOVESPA*. São Paulo: B3. Disponível em: <https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-de-derivativos/precos-referenciais/taxas-referenciais-bm-fbovespa/>.
- **ANBIMA – Associação Brasileira das Entidades dos Mercados Financeiro e de Capitais.** *Metodologia de Precificação de Títulos Públicos Federais e Estrutura a Termo das Taxas de Juros (ETTJ)*. Rio de Janeiro: ANBIMA. Disponível em: <https://www.anbima.com.br/pt_br/informar/precos-e-indices/precos/titulos-publicos.htm>.
- **Banco Central do Brasil (BCB) / Conselho Monetário Nacional (CMN).** *Resolução CMN nº 5.124, de 28 de março de 2024: Metodologia de cálculo da Taxa Básica Financeira (TBF) e da Taxa Referencial (TR)*. Brasília: BCB, 2024. Disponível em: <https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo?tipo=Resolu%C3%A7%C3%A3o%20CMN&numero=5124>.
- **Banco Central do Brasil (BCB).** *Sistema Gerenciador de Séries Temporais (SGS) e Sistema Expectativas de Mercado (Focus/Olinda)*. Brasília: BCB. Disponível em: <https://www3.bcb.gov.br/sgspub/>.
- **Secretaria do Tesouro Nacional (STN).** *Regras e Fórmulas de Cálculo dos Títulos Públicos Federais no Mercado Secundário*. Brasília: Ministério da Fazenda. Disponível em: <https://www.tesourodireto.com.br/titulos/historico-de-precos-e-taxas.htm>.
- **Federal Reserve Bank of New York.** *Secured Overnight Financing Rate (SOFR) Data & Calculation Methodology*. New York: FRBNY. Disponível em: <https://www.newyorkfed.org/markets/reference-rates/sofr>.
- **European Central Bank (ECB).** *Euro Short-Term Rate (€STR) Methodology and Policies*. Frankfurt: ECB. Disponível em: <https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_short-term_rate/html/index.en.html>.

---

## 8. Licença e Isenção de Responsabilidade Legal (*Disclaimer*)

Este software e todos os artefatos analíticos correlatos são distribuídos sob a licença de código aberto **MIT** (consulte o arquivo [`LICENSE`](LICENSE) na raiz deste repositório para os termos integrais).

> **Aviso de Isenção Legal e Financeira:** O conteúdo deste repositório destina-se estritamente a propósitos científicos, acadêmicos e de desenvolvimento quantitativo. Os modelos de interpolação, curvas de juros, projeções de inflação e precificações de ativos aqui implementados **não constituem assessoria financeira, recomendação de investimento ou prestação de serviços de análise de valores mobiliários** regulados pela Comissão de Valores Mobiliários (CVM), pela ANBIMA ou pelo Banco Central do Brasil.
