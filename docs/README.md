# Documentação Técnica — `bz-pricing`

O pacote `bz-pricing` constitui uma biblioteca quantitativa de alta performance em Python e Cython destinada à construção de estruturas a termo de taxas de juros, inflação implícita, cupons cambiais, moedas a termo e índices de ações, bem como à precificação determinística e análise de risco de títulos públicos e privados nos mercados doméstico (*onshore*) e internacional (*offshore*).

---

## 1. Estrutura da Documentação

1. **[Metodologia de Construção de Estruturas a Termo e Formulações Matemáticas](metodologia_curvas.md)**
   - Operadores de interpolação em Cython (`FlatForward252Interpolator`, `FlatForward252LinearInterpolator`, `Linear360Interpolator` e `PriceInterpolator`) e splines em cache (`LogCubicDiscountInterpolator` e `MonotoneConvexInterpolator`);
   - Curvas de juros prefixadas domésticas (`PRE`, `SELIC_PRE` e `LTN`);
   - Curvas de Taxa Básica Financeira (`TFP`) e Taxa Referencial (`TP` e `TR`);
   - Curvas internacionais de futuros e *swaps* OIS (`SFR`, `EST`, `TIE`, `ZUS`, `ZEU`, `ZMX`, `SOFR_OIS`, `ESTR_OIS`, `SONIA_OIS`, `TONA_OIS`, `UST_PAR`) e modelos paramétricos (`NelsonSiegelCurve`, `SvenssonCurve`);
   - Estruturas a termo de inflação em cupom sujo (`DIC` e `DIM`) e cupom limpo (`DPL`);
   - Cupons cambiais (`DOL`, `DOC`, `DCL`, `EUC` e `YCL`), spreads externos (`SDE`, `SGP`, `SMX`, `SYD`) e moedas a termo (`PTX`, `EUR` e `JPY`);
   - Curvas de índices de ações (`INP`, `BRP`), futuros de ações individuais e *Convenience Yield* (`CYI`);
   - Construção integralmente *offline* de curvas, títulos públicos, crédito privado, derivativos e opções com insumos do usuário;
   - Referências normativas e bibliográficas.
2. **[Guia de Referência da API](guia_uso_api.md)**
   - Orquestrador central de estruturas a termo (`CurveManager`, com `allow_download=False` para operação *offline*) e *bootstrapping offshore* (`bootstrap_ois_curve`, `bootstrap_treasury_curve`);
   - Construção por vértices customizados e tratamento de exceções (`DataUnavailableError`);
   - Extração e coleta automatizada de dados (`B3DataLoader` e `B3Scraper`, com suporte a séries de opções via `load_options`);
   - Instanciação direta de curvas e interpoladores em Cython (`B3CurveBuilder` e `SimpleCurve`);
   - Precificação de títulos públicos federais (`TPF`), cálculo de VNA a partir do VNE (`VNAEngine`, `calculate_vna_from_vne`, `calculate_ntnb_vna`, `calculate_lft_vna`), `TPFBuilder(..., real_discount=True)` e fluxos customizados (`custom_cashflows`);
   - Renda fixa *offshore* e títulos do Tesouro dos EUA (`USTreasuryBill`, `USTreasuryNoteBond`, `TIPS`, `OffshoreBond`, com Z-Spread, G-Spread, YTM e *hazard rate*);
   - Crédito privado (`Debenture`, `CRI`, `CRA`, `CotaFIDC`, `LetraFinanceira`), *Z-Spread*, YTM, TIR, avaliação com probabilidade de *default* (`risky_present_value`) e lotes (`price_credit_portfolio`);
   - Derivativos lineares domésticos e *offshore* (`DI1Future`, `DDIFuture`, `DAPFuture`, `DOLFuture`, `INDFuture`, `SwapDIxPRE`, `SwapDIxIPCA`, `NDFContract`, `OISSwap`, `SOFRFuture`, `CrossCurrencySwap`);
   - Opções listadas, *offshore* e exóticas (`EquityOption` com dividendos discretos e proporcionais, `FXOption`, `IDIOption`, `DI1Option`, `CorradoSuOption`, `Swaption`, `CapFloor`, `BarrierOption`, `AsianOption`, avaliação de carteiras e gregas completas via `price_options_portfolio`);
   - Superfícies de volatilidade (`SVISlice`, `SABRSlice`, `LognormalMixtureSlice`, `BloombergMixtureSurface`, `KalmanDividendFilter`, `DividendCurveBBG`, `VolatilitySurface`).
3. **[Calendários, Convenções de Contagem de Dias e Regimes de Extrapolação](calendarios_convencoes.md)**
   - Convenções `BUS/252`, `ACT/360`, `ACT/365`, `ACT/ACT_ICMA` e `30/360`;
   - Calendários embutidos em memória (`B3`, `US`, `TARGET`, `UK` e `ACTUAL`), isolamento de feriados customizados e rolagem de datas (`following`, `preceding`, `modified_following`);
   - Regimes de extrapolação constante (`EXTRAP_FLAT`) e por taxa a termo constante (`EXTRAP_FLAT_FORWARD`).
4. **[Casos de Uso e Exemplos Práticos](exemplos_casos_uso.md)**
   - Construção de curvas *onshore* e *offshore* (*offline* e automático, incluindo SOFR OIS e Treasuries EUA);
   - Títulos públicos federais, cálculo de VNA a partir do VNE (`calculate_ntnb_vna`, `calculate_lft_vna`, `calculate_vna_from_vne`) e fluxos customizados (`custom_cashflows`);
   - Renda fixa *offshore* (`USTreasuryBill`, `USTreasuryNoteBond`, `TIPS`, `OffshoreBond`), *swaps* OIS/SOFR/Cross-Currency e crédito privado;
   - Opções com dividendos discretos, opções exóticas com gregas analíticas e numéricas, e validação de superfícies de volatilidade (SVI, SABR e modelo de mistura lognormal Bloomberg).

---

## 2. Compilação e Execução de Testes

Para compilar as extensões nativas em C/Cython no ambiente virtual do projeto:

```bash
poetry install
poetry run python setup.py build_ext --inplace
```

Para executar a suíte de testes automatizados:

```bash
poetry run pytest tests/ -v
```
