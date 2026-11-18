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
5. **Curvas Internacionais Offshore (`ACT/360` e `ACT/365`):** Curvas a termo de futuros de taxas *overnight* (`SFR`, `EST`, `TIE`), curvas zero cupom (`ZUS`, `ZEU`, `ZMX`) e *bootstrapping* em anos civis para `SOFR_OIS`, `ESTR_OIS`, `SONIA_OIS`, `TONA_OIS` e `UST_PAR` via `MonotoneConvexInterpolator` (Tesouro EUA), `LogLinearDiscountInterpolator` (CME/LCH), `LogCubicDiscountInterpolator` (ambos com cache nativo em C e busca binária $O(\log n)$), `NelsonSiegelCurve` e `SvenssonCurve` (Fed/BCE);
6. **Renda Variável e Convenience Yield (`BUS/252`):** Estruturas de preços futuros de índices (`INP` para Ibovespa, `BRP` para IBrX-50, `SML`, `MBR`, `VIX`), futuros de ações individuais e curvas de custo de carrego (`CYI`, `CYM`, `CYX`);
7. **Títulos Públicos Federais (`TPF`) e Renda Fixa Offshore (`offshore_bonds`):** Precificação determinística de **LTN**, **NTN-F**, **NTN-B**, **NTN-B Principal**, **NTN-C** e **LFT** com VNA calculado a partir do VNE (`calculate_vna_from_vne`, `calculate_ntnb_vna`, `calculate_lft_vna`, `calculate_ntnc_vna`), `TPFBuilder(..., real_discount=True)` e fluxos customizados (`custom_cashflows`); além de títulos norte-americanos e corporativos internacionais (`USTreasuryBill`, `USTreasuryNoteBond`, `TIPS`, `OffshoreBond`) com `Z-Spread`, `G-Spread`, *hazard rate* e operação integralmente *offline*;
8. **Crédito Privado (`Debenture`, `CRI`, `CRA`, `CotaFIDC`, `LetraFinanceira`, `HazardRateCurve`):** Extração estruturada no SND e ANBIMA, construção por cronograma (`from_schedule`) ou fluxos customizados (`from_cashflows` / `custom_cashflows`), PU Par, PU a mercado, *Z-Spread*, *gross-up* tributário, YTM, TIR, modelagem de *hazard rate* (`risky_present_value`, `implied_hazard_rate`, `credit_dv01`) e precificação em lote (`price_credit_portfolio`);
9. **Derivativos Lineares Onshore e Offshore (`DI1Future`, `DDIFuture`, `DAPFuture`, `DOLFuture`, `INDFuture`, `SwapDIxPRE`, `SwapDIxIPCA`, `NDFContract`, `OISSwap`, `SOFRFuture`, `CrossCurrencySwap`):** Precificação de futuros de juros, cupom cambial, cupom de IPCA, dólar, Ibovespa, futuros `SOFR` (`SR3` e `SR1`), *swaps* domésticos e *OIS*, *cross-currency swaps* (`BRL` x `USD`) e NDFs com cálculo de PU, taxa par, ajuste financeiro, DV01 e avaliação em lote (`price_di1_portfolio`);
10. **Opções e Superfícies de Volatilidade (`EquityOption`, `FXOption`, `IDIOption`, `DI1Option`, `CorradoSuOption`, `Swaption`, `CapFloor`, `BarrierOption`, `AsianOption`, `VolatilitySurface`):** Modelos Black-Scholes-Merton, Garman-Kohlhagen (com ajuste de *fixing* e convenção `ACT/360`), Black-76 e Bachelier para opções de IDI, DI1, `Swaption` e `CapFloor` multiconvenção (`BUS/252`, `ACT/360`, `ACT/365`, `30/360`), árvore binomial CRR em Cython para exercício americano (incluindo `discrete_dividends` e `prop_dividends`), Corrado & Su, barreiras contínuas/discretas e asiáticas com gregas completas (`risk_metrics` e `price_options_portfolio`), parametrizações `SVISlice`, `SABRSlice`, mistura de lognormais com estado de *default* (`LognormalMixtureSlice` e `BloombergMixtureSurface`), desamericanização/inferência de dividendos com filtro de Kalman (`deamericanize_and_infer_forward_bbg`, `KalmanDividendFilter`, `DividendCurveBBG`), auditoria de não arbitragem, densidade de Breeden-Litzenberger e volatilidade local de Dupire.

---

## 2. Fundamentação Metodológica e Formulações Matemáticas

As formulações completas encontram-se detalhadas em [`docs/metodologia_curvas.md`](docs/metodologia_curvas.md) e na [Wiki do projeto](https://github.com/gtazevedo/bz_pricing/wiki). Apresentam-se a seguir as equações centrais implementadas no motor em Cython.

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
│   │   ├── interpolators.pxd / .pyx     # Interpoladores nativos em Cython (busca binária O(log n))
│   │   ├── curves.pxd / .pyx            # Estruturas de curvas compiladas (SimpleCurve, ComplexCurve, ConvenienceYieldCurve)
│   │   ├── bonds.pxd / .pyx             # Motor Cython de fluxos de caixa, PU, YTM e métricas de risco de TPF
│   │   ├── credit_core.pyx              # Motor Cython para crédito privado, gross-up e Z-Spread
│   │   ├── derivatives_core.pyx         # Motor Cython para futuros, swaps e NDF
│   │   ├── options_core.pyx             # Motor Cython para BSM, Black-76, CRR (discreto/proporcional), Corrado-Su, barreiras, asiáticas, SVI e SABR
│   │   ├── _anbima_holidays.py          # Tabelas embutidas em memória para calendários B3, US, TARGET e UK
│   │   ├── calendars.py                 # Integração com calendários de dias úteis em modo offline/online (bizdays)
│   │   └── vna_engine.py                # Cálculo e projeção de VNA a partir do VNE (IPCA, IGP-M, Selic, CDI)
│   │
│   ├── curves/                          # Camada de Orquestração e Construção de Curvas
│   │   ├── configs.py                   # Especificações de interpolação, arredondamento e convenções por curva
│   │   ├── offshore.py                  # Bootstrapping de curvas OIS e Treasuries (Monotone Convex, Log-Linear, Log-Cubic, Svensson)
│   │   ├── builder.py                   # Construtor algébrico (B3CurveBuilder, CrossCouponCurve, CurrencyForwardCurve)
│   │   ├── manager.py                   # Fachada central de resolução de insumos (CurveManager)
│   │   └── exceptions.py                # Tratamento estruturado de exceções (DataUnavailableError)
│   │
│   ├── instruments/                     # Instrumentos de Renda Fixa, Derivativos e Opções
│   │   ├── tpf.py                       # Precificação de LTN, NTN-F, NTN-B, NTN-B Principal, NTN-C e LFT (com VNA e custom_cashflows)
│   │   ├── offshore_bonds.py            # Títulos dos EUA e globais offline (USTreasuryBill, USTreasuryNoteBond, TIPS, OffshoreBond)
│   │   ├── credit.py                    # Debenture, CRI, CRA, CotaFIDC, LetraFinanceira, Z-Spread, YTM/TIR e fluxos customizados
│   │   ├── derivatives.py               # Futuros (DI1, DDI, DAP, DOL, IND, SOFR SR3/SR1), Swaps (DIxPré, DIxIPCA, OIS, CCS) e NDF
│   │   └── options.py                   # EquityOption (com dividendos), FXOption, IDIOption, DI1Option, CorradoSu, Swaption, CapFloor, Barrier, Asian
│   │
│   ├── volatility/                      # Superfícies de Volatilidade Implícita e Validação
│   │   ├── bloomberg_surface.py         # Modelo Bloomberg (desamericanização, Kalman, mistura de lognormais e superfície 2D)
│   │   └── surface.py                   # SVISlice, SABRSlice, VolatilitySurface, Breeden-Litzenberger, Dupire e OOS
│   │
│   ├── bonds/                           # Construtores auxiliares de carteiras de títulos
│   │   └── builder.py                   # TPFBuilder integrado a dados secundários e VNAEngine
│   │
│   └── data/                            # Ingestão e Extração de Dados Públicos
│       ├── loaders.py                   # Leitura de SPRD, IN (séries de opções), ANBIMA e taxas referenciais
│       ├── anbima_data.py               # Cliente unificado ANBIMA Data (ETTJ, debêntures, CRI/CRA, LF)
│       ├── cricra.py                    # Cliente e gerador de cronogramas de CRI/CRA
│       ├── debentures.py                # Cliente de dados do SND (debentures.com.br) e mercado ANBIMA
│       └── scrapers.py                  # Coletores automatizados (B3, ANBIMA, BCB SGS/Olinda, NY Fed, ECB, Banxico)
│
├── notebooks/                           # Cadernos de Validação Empírica
│   ├── analise_curvas.ipynb             # Auditoria visual e numérica de todas as famílias de curvas
│   ├── validacao_tpfs.ipynb             # Validação de 50 TPFs ANBIMA, VNA a partir do VNE e fluxos customizados
│   ├── validacao_debentures.ipynb       # Extração de dados SND, PU Par e apreçamento de debêntures
│   ├── validacao_cricra.ipynb           # Validação de CRI e CRA contra preços indicativos ANBIMA
│   ├── validacao_fidc_lf_zspread.ipynb  # Validação de FIDC, Letras Financeiras, Z-Spread, YTM e TIR
│   ├── validacao_derivativos_lineares.ipynb # Validação de futuros, swaps e NDF frente aos manuais B3
│   └── validacao_opcoes_superficies.ipynb   # Validação de opções, SVI/SABR, Breeden-Litzenberger, Dupire e DOLZ26
│
└── tests/                               # Suíte Automatizada de Testes (Pytest)
    ├── test_interpolators.py            # Testes dos operadores de interpolação, cache de splines, busca binária e casos de borda
    ├── test_curves.py                   # Testes de aderência por vértices, curvas offshore e construção 100% offline por componentes
    ├── test_bonds.py                    # Testes de TPFs, títulos offshore (UST, TIPS, OffshoreBond), VNA e fluxos customizados
    ├── test_debentures.py               # Testes de extração SND e cálculo de debêntures (IPCA, CDI e Pré)
    ├── test_cricra.py                   # Testes de CRI e CRA
    ├── test_fidc_lf.py                  # Testes de FIDC e Letras Financeiras
    ├── test_zspread.py                  # Testes de Z-Spread, gross-up, YTM, TIR, hazard rate e fluxos customizados
    ├── test_anbima_data.py              # Testes de clientes e cache ANBIMA Data
    ├── test_derivatives.py              # Testes de futuros (DI1, SOFR), swaps (DIxPré, DIxIPCA, OIS, CCS) e NDF
    ├── test_options_vol.py              # Testes de opções, dividendos discretos/proporcionais, exóticas, SVI/SABR, Bloomberg e Dupire
    └── test_docs_examples.py            # Execução determinística offline de todos os exemplos da documentação
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

### 5.3 Precificação de Títulos Públicos Federais (TPFs) Baseada em Objetos de Curvas

A precificação de títulos públicos federais (ANBIMA / STN) é executada em Cython através das classes de instrumentos, as quais recebem os objetos de curvas de desconto e projeção:

```python
import datetime
from bz_pricing.core.curves import SimpleCurve, FlatCurve
from bz_pricing.core.interpolators import FlatForward252Interpolator
from bz_pricing.instruments.tpf import LTN, NTNF, LFT, NTNB, NTNBPrincipal, NTNC
from bz_pricing.curves.builder import B3CurveBuilder

base_date = datetime.date(2026, 10, 30)
builder = B3CurveBuilder()

# Curva prefixada construída a partir de nós customizados
curva_pre = builder.build_from_vertices(
    t_nodes=[1, 42, 102, 252],
    y_nodes=[0.1415, 0.135793, 0.137844, 0.1420],
    round_decimals=6,
    round_mode='trunc'
)

# 1. LTN (Zero-Cupom Prefixado)
ltn = LTN(maturity_date=datetime.date(2027, 4, 1))
pu_ltn = ltn.present_value(base_date, curva_pre)
dur_ltn = ltn.duration(base_date, curva_pre)
print(f"LTN 2027: PU = R$ {pu_ltn:.6f} | Duration = {dur_ltn:.1f} DU")

# 2. NTN-F (Cupom Semestral de 10,00% a.a.)
ntnf = NTNF(maturity_date=datetime.date(2027, 1, 1), coupon_rate=0.10)
pu_ntnf = ntnf.present_value(base_date, curva_pre)
print(f"NTN-F 2027: PU = R$ {pu_ntnf:.6f}")

# 3. LFT (Curva de Ágio/Deságio + VNA)
curva_agio = FlatCurve(0.000232)
lft = LFT(maturity_date=datetime.date(2027, 3, 1))
pu_lft = lft.present_value(base_date, curva_agio, vna=20201.049997)
print(f"LFT 2027: PU = R$ {pu_lft:.6f}")

# 4. NTN-B (Curva de Juros Reais + VNA de IPCA)
curva_real = FlatCurve(0.046629)
ntnb = NTNB(maturity_date=datetime.date(2027, 5, 15), coupon_rate=0.06)
pu_ntnb = ntnb.present_value(base_date, curva_real, vna=4771.575461)
print(f"NTN-B 2027: PU = R$ {pu_ntnb:.6f}")

# 5. NTN-C (Indexada ao IGP-M com Cupons Semestrais — série 2031 paga 12% a.a.)
ntnc = NTNC(maturity_date=datetime.date(2031, 1, 1))
pu_ntnc = ntnc.present_value(base_date, FlatCurve(0.077739), vna=6765.022425)
print(f"NTN-C 2031: PU = R$ {pu_ntnc:.6f}")

# 6. Precificação Multi-Curva (Taxa Real + Projeção de Inflação Implícita)
curva_dic = FlatCurve(0.0450)
pu_multi = ntnb.present_value(
    base_date=base_date,
    real_curve=curva_real,
    inflation_curve=curva_dic,
    base_vna=4750.00,
    base_vna_date=datetime.date(2026, 9, 15)
)
print(f"NTN-B 2027 Multi-Curva: PU = R$ {pu_multi:.6f}")
# 7. Cálculo automático de VNA a partir do VNE e fluxos customizados do usuário
from bz_pricing.core.vna_engine import calculate_lft_vna, calculate_ntnb_vna, calculate_vna_from_vne

vna_lft_calc = calculate_lft_vna(base_date)                  # R$ 20.201,049997 em 30/10/2026
vna_ntnb_calc = calculate_ntnb_vna(base_date, ipca_proj=0.001972)
pu_ntnb_auto = ntnb.present_value(base_date, curva_real, ipca_proj=0.001972)

ntnf_custom = NTNF(custom_cashflows=[(datetime.date(2027, 1, 4), 50.0, "Cupom"), (datetime.date(2027, 7, 1), 1050.0, "Principal")])
pu_custom = ntnf_custom.present_value(base_date, FlatCurve(0.1495))
```

### 5.4 Precificação de Debêntures, Crédito Privado e Fluxos Customizados

A precificação de debêntures e títulos de crédito privado (`Debenture`, `CRI`, `CRA`, `CotaFIDC`, `LetraFinanceira`) integra a extração de cronogramas contratuais no Sistema Nacional de Debêntures (SND / `debentures.com.br`), cotações de mercado secundário da ANBIMA, cálculo de VNA a partir do VNE e construção por fluxos customizados:

```python
import datetime
from bz_pricing.data.debentures import DebenturesClient
from bz_pricing.instruments.credit import Debenture

client = DebenturesClient(cache_dir="dados/debentures")
base_date = datetime.date(2026, 10, 30)

# 1. Carregamento analítico do ativo via código SND (ex.: VALEA0 - IPCA + 6,3837% a.a.)
deb = Debenture.from_ticker("VALEA0", client=client)

# 2. Cômputo do PU Par teórico com base no VNA oficial e juros acumulados da emissão
vna_data = 1096.791130
pu_par = deb.pu_par(base_date, vna=vna_data)
print(f"VALEA0 - PU Par Teórico: R$ {pu_par:.6f}")

# 3. Precificação a mercado por taxa indicativa, métricas de risco e Z-Spread
taxa_mercado = 0.066548
pu_mercado, df_fluxos = deb.present_value(
    base_date=base_date,
    discount_rate_or_curve=taxa_mercado,
    vna=vna_data,
    return_cashflows=True
)
dur_mercado = deb.duration(base_date=base_date, discount_rate_or_curve=taxa_mercado, vna=vna_data)
ytm_calc = deb.ytm(base_date, pu_mercado, vna=vna_data)

print(f"VALEA0 - PU Mercado: R$ {pu_mercado:.6f} | Duration: {dur_mercado:.2f} DU | YTM: {ytm_calc*100:.4f}% a.a.")
print(df_fluxos[["data_pagamento", "du", "evento", "fluxo_nominal", "valor_presente"]])

# 4. Ativo construído diretamente com fluxos próprios do usuário
deb_user = Debenture.from_cashflows(
    "DEB-CUSTOM",
    [(datetime.date(2027, 4, 30), 60.0, "Juros"), (datetime.date(2027, 10, 29), 1060.0, "Vencimento")],
    indexador="PRE"
)
print(f"Debênture Custom PU: R$ {deb_user.present_value(base_date, 0.13):.6f}")
```

### 5.5 Derivativos Lineares, Opções e Superfícies de Volatilidade

```python
import datetime
from bz_pricing.instruments import DI1Future, SwapDIxPRE, EquityOption, FXOption
from bz_pricing.volatility import SVISlice, VolatilitySurface

base_date = datetime.date(2026, 10, 30)

# Futuro de DI1 e Swap DI x Pré
di1 = DI1Future("DI1F28", "2028-01-03")
pu_di1 = di1.pu(base_date, 0.1385)
swap = SwapDIxPRE(notional=1_000_000.0, start_date="2026-10-30", maturity_date="2028-01-03", fixed_rate=0.1350)
mtm_swap = swap.mtm(base_date, pre_rate_or_curve=0.1410, di_accrued_factor=1.0)

# Opções e superfície SVI livre de arbitragem
opt_am = EquityOption("PETRO38", "2027-01-04", 38.0, "PUT", "AMERICAN")
rm = opt_am.risk_metrics(base_date, spot=37.5, pre_rate_or_curve=0.14, vol=0.30)

sl = SVISlice(a=0.015, b=0.04, rho=-0.30, m=0.0, sigma=0.12, T=63 / 252.0)
surf = VolatilitySurface.from_svi_slices({63: sl})
fx_opt = FXOption("DOLC560", "2027-01-29", 5.60, "CALL")
preco_fx = fx_opt.price(base_date, spot_ptax=5.55, pre_rate_or_curve=0.14, cupom_limpo_or_curve=0.05, vol=surf)
```

### 5.6 Curvas Internacionais (`SOFR_OIS`, `UST_PAR`) e Superfície de Mistura Bloomberg

```python
from bz_pricing.curves import bootstrap_ois_curve, bootstrap_treasury_curve
from bz_pricing.volatility import LognormalMixtureSlice, BloombergMixtureSurface

sofr = bootstrap_ois_curve([90, 180, 365, 730], [0.0428, 0.0420, 0.0405, 0.0392], convention="ACT/360", method="monotone_convex")
ust = bootstrap_treasury_curve([182, 365, 730, 1825], [0.0430, 0.0422, 0.0410, 0.0415], method="svensson")
sl_mix = LognormalMixtureSlice([0.57, 0.38], [0.84, 1.3715789473684212], [0.35, 0.25], forward=100.0, T=1.0)
surf_mix = BloombergMixtureSurface({252: sl_mix})
print(f"SOFR 1Y: {sofr.get_rate(365.0)*100:.4f}% | UST 5Y: {ust.get_rate(1825.0)*100:.4f}% | Mix Vol(K=90): {surf_mix.get_vol(252, 90.0)*100:.2f}%")
```

### 5.7 Renda Fixa Offshore, Swaps OIS/Cross-Currency e Opções com Dividendos e Gregas

```python
import datetime
import numpy as np
from bz_pricing.core.curves import FlatCurve, SimpleCurve
from bz_pricing.core.interpolators import LogLinearDiscountInterpolator
from bz_pricing.curves import bootstrap_ois_curve
from bz_pricing.instruments import (
    BarrierOption,
    CrossCurrencySwap,
    EquityOption,
    OISSwap,
    OffshoreBond,
    SOFRFuture,
    TIPS,
    USTreasuryNoteBond,
    price_options_portfolio,
)

base_date = datetime.date(2026, 10, 30)
sofr_crv = bootstrap_ois_curve([90, 180, 365, 730, 1825], [0.0432, 0.0425, 0.0415, 0.0408, 0.0412], convention="ACT/360")

tnote = USTreasuryNoteBond("2031-05-15", coupon_rate=0.04125)
tips = TIPS("2031-01-15", coupon_rate=0.0175, issue_cpi=305.0)
eurobond = OffshoreBond("2031-10-30", coupon_rate=0.0550, day_count="30/360")
z_sprd_bps = eurobond.z_spread(base_date, target_price=eurobond.clean_price(base_date, ytm=0.0575), zero_curve=sofr_crv) * 1e4

ois = OISSwap(10_000_000.0, base_date, "2028-10-30", fixed_rate=0.0410)
sr3 = SOFRFuture("SR3M27", "2027-06-16", "2027-09-15", contract_type="SR3")
ccs = CrossCurrencySwap(1_000_000.0, base_date, "2028-10-30", initial_fx_spot=5.65)
cupom_crv = SimpleCurve(np.array([30.0, 1825.0]), np.array([0.05, 0.05]), LogLinearDiscountInterpolator(0, 1, 360.0, "simple"))
par_basis = ccs.par_basis_spread(base_date, fx_spot=5.65, pre_curve=FlatCurve(0.138), foreign_discount_curve=cupom_crv, foreign_forward_curve=sofr_crv)

opt_div = EquityOption("PETR4", "2027-04-30", strike=38.0, option_type="PUT", exercise="AMERICAN", discrete_dividends={"2026-12-15": 1.60})
bar = BarrierOption("VALE-KO", "2027-04-30", strike=40.0, barrier=32.0, barrier_type="down-and-out", monitoring_steps=52)
df_g = price_options_portfolio(
    [
        {"instrument": opt_div, "spot": 40.0, "pre_rate_or_curve": 0.12, "vol": 0.30, "quantity": 100},
        {"instrument": bar, "spot": 40.0, "pre_rate_or_curve": 0.12, "vol": 0.30, "quantity": 50},
    ],
    base_date=base_date,
)
print(f"T-Note Clean: {tnote.clean_price(base_date, 0.042):.4f} | TIPS: {tips.adjusted_clean_price(base_date, 0.019, 318.5):.4f} | Z-Sprd: {z_sprd_bps:.1f} bps | OIS: {ois.par_rate(base_date, sofr_crv)*100:.3f}% | SR3: {sr3.fair_price(base_date, sofr_crv):.3f} | CCS: {par_basis*1e4:.1f} bps")
print(df_g[["ticker", "price", "delta", "gamma", "vega_1pct"]])
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

### 6.1 Validação de Títulos Públicos Federais (ANBIMA — Data-Base 30/10/2026)

O teste automatizado em lote (`test_all_anbima_bonds_20261030` em `tests/test_bonds.py`) confronta os preços calculados pelo motor analítico contra todos os 50 títulos públicos federais ativos divulgados pela ANBIMA no arquivo oficial `ANBIMA_MS_20261030.csv`:

| Família de Títulos | Quantidade de Títulos Vigentes | Vencimentos Analisados | Divergência Máxima Observada |
| :---: | :---: | :---: | :---: |
| **LTN** | 11 | 2027 a 2032 | R\$ 0,000000 |
| **NTN-F** | 6 | 2027 a 2037 | R\$ 0,000000 |
| **LFT** | 18 | 2027 a 2032 | R\$ 0,000000 |
| **NTN-B** | 14 | 2027 a 2060 | R\$ 0,000000 |
| **NTN-C** | 1 | 2031 | R\$ 0,000000 |
| **Total Consolidado** | **50** | **100% dos títulos do mercado** | **R\$ 0,000000** |

### 6.2 Validação de Debêntures no Mercado Secundário (SND / ANBIMA)

A consistência dos fluxos e do PU Par é verificada no caderno [`notebooks/validacao_debentures.ipynb`](notebooks/validacao_debentures.ipynb) e nos testes automatizados (`tests/test_debentures.py`). O cômputo do PU Par teórico para a debênture `VALEA0` em 30/10/2026 reproduz o valor divulgado pelo SND até a sexta casa decimal:

| Ativo | Emissor | Indexador | VNA Apurado | Juros Acumulados | PU Par Teórico | PU Par SND | Divergência |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `VALEA0` | Vale S.A. | IPCA + 6,40% a.a. | R\$ 1.096,791130 | 0,270486% | R\$ 1.099,757794 | R\$ 1.099,757794 | R\$ 0,000000 |

O arquivo oficial de mercado secundário de debêntures da ANBIMA (`db261030.txt`), com 1.272 ativos negociados, é utilizado para apreçamento a mercado e cálculo de duration individual e agregada.

---

## 7. Referências Normativas e Bibliográficas

- **B3 S.A. – Brasil, Bolsa, Balcão.** *Manual de Apreçamento de Opções da B3*. São Paulo: B3. Disponível em: <https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-de-derivativos/precos-referenciais/manuais-de-aprecamento/>.
- **B3 S.A. – Brasil, Bolsa, Balcão.** *Manual de Curvas da B3: Metodologia de Apuração das Estruturas a Termo de Taxas de Juros, Moedas e Índices*. São Paulo: B3, 2024. Disponível em: <https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-de-derivativos/precos-referenciais/manuais-de-curvas/>.
- **B3 S.A. – Brasil, Bolsa, Balcão.** *Preços Referenciais – Taxas Referenciais BM&FBOVESPA*. São Paulo: B3. Disponível em: <https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-de-derivativos/precos-referenciais/taxas-referenciais-bm-fbovespa/>.
- **ANBIMA – Associação Brasileira das Entidades dos Mercados Financeiro e de Capitais.** *Metodologia de Precificação de Títulos Públicos Federais e Estrutura a Termo das Taxas de Juros (ETTJ)*. Rio de Janeiro: ANBIMA. Disponível em: <https://www.anbima.com.br/pt_br/informar/precos-e-indices/precos/titulos-publicos.htm>.
- **ANBIMA – Associação Brasileira das Entidades dos Mercados Financeiro e de Capitais.** *Mercado Secundário de Debêntures e Metodologia de Precificação de Títulos Privados*. Rio de Janeiro: ANBIMA. Disponível em: <https://www.anbima.com.br/informacoes/merc-sec-debentures/>.
- **SND – Sistema Nacional de Debêntures.** *Consultas e Metodologia de Eventos Financeiros e PU Par*. Rio de Janeiro: B3 / ANBIMA. Disponível em: <https://www.debentures.com.br/>.
- **Banco Central do Brasil (BCB) / Conselho Monetário Nacional (CMN).** *Resolução CMN nº 5.124, de 28 de março de 2024: Metodologia de cálculo da Taxa Básica Financeira (TBF) e da Taxa Referencial (TR)*. Brasília: BCB, 2024. Disponível em: <https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo?tipo=Resolu%C3%A7%C3%A3o%20CMN&numero=5124>.
- **Banco Central do Brasil (BCB).** *Sistema Gerenciador de Séries Temporais (SGS) e Sistema Expectativas de Mercado (Focus/Olinda)*. Brasília: BCB. Disponível em: <https://www3.bcb.gov.br/sgspub/>.
- **Secretaria do Tesouro Nacional (STN).** *Regras e Fórmulas de Cálculo dos Títulos Públicos Federais no Mercado Secundário*. Brasília: Ministério da Fazenda. Disponível em: <https://www.tesourodireto.com.br/titulos/historico-de-precos-e-taxas.htm>.
- **Federal Reserve Bank of New York.** *Secured Overnight Financing Rate (SOFR) Data & Calculation Methodology*. New York: FRBNY. Disponível em: <https://www.newyorkfed.org/markets/reference-rates/sofr>.
- **European Central Bank (ECB).** *Euro Short-Term Rate (€STR) Methodology and Policies*. Frankfurt: ECB. Disponível em: <https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_short-term_rate/html/index.en.html>.
- **Gürkaynak, R. S., Sack, B., & Wright, J. H.** *The U.S. Treasury Yield Curve: 1961 to the Present*. Federal Reserve Board, 2006. Disponível em: <https://www.federalreserve.gov/econres/feds/the-us-treasury-yield-curve-1961-to-the-present.htm>.
- **U.S. Department of the Treasury.** *Treasury Coupon-Issue and Corporate Bond Yield Curve Methodology*. Washington, D.C.: Office of Debt Management. Disponível em: <https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics>.
- **Hagan, P. S., & West, G.** *Interpolation Methods for Curve Construction*. Applied Mathematical Finance, v. 13, n. 2, p. 89–129, 2006.

---

## 8. Licença e Isenção de Responsabilidade Legal (*Disclaimer*)

Este software e todos os artefatos analíticos correlatos são distribuídos sob a licença de código aberto **MIT** (consulte o arquivo [`LICENSE`](LICENSE) na raiz deste repositório para os termos integrais).

> **Aviso de Isenção Legal e Financeira:** O conteúdo deste repositório destina-se estritamente a propósitos científicos, acadêmicos e de desenvolvimento quantitativo. Os modelos de interpolação, curvas de juros, projeções de inflação e precificações de ativos aqui implementados **não constituem assessoria financeira, recomendação de investimento ou prestação de serviços de análise de valores mobiliários** regulados pela Comissão de Valores Mobiliários (CVM), pela ANBIMA ou pelo Banco Central do Brasil.
