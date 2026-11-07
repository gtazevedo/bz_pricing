# Documentação Técnica — `bz-pricing`

O pacote `bz-pricing` constitui uma biblioteca quantitativa de alta performance em Python e Cython destinada à construção de estruturas a termo de taxas de juros, inflação implícita, cupons cambiais, moedas a termo e índices de ações, bem como à precificação determinística e análise de risco de títulos públicos e privados nos mercados doméstico (*onshore*) e internacional (*offshore*).

---

## 1. Estrutura da Documentação

1. **[Metodologia de Construção de Estruturas a Termo e Formulações Matemáticas](metodologia_curvas.md)**
   - Operadores de interpolação em Cython (`FlatForward252Interpolator`, `FlatForward252LinearInterpolator`, `Linear360Interpolator` e `PriceInterpolator`);
   - Curvas de juros prefixadas domésticas (`PRE`, `SELIC_PRE` e `LTN`);
   - Curvas de Taxa Básica Financeira (`TFP`) e Taxa Referencial (`TP` e `TR`);
   - Curvas internacionais de futuros e *swaps* OIS (`SFR`, `EST`, `TIE`, `ZUS`, `ZEU` e `ZMX`);
   - Estruturas a termo de inflação em cupom sujo (`DIC` e `DIM`) e cupom limpo (`DPL`);
   - Cupons cambiais (`DOL`, `DOC`, `DCL`, `EUC` e `YCL`), spreads externos (`SDE`, `SGP`, `SMX`, `SYD`) e moedas a termo (`PTX`, `EUR` e `JPY`);
   - Curvas de índices de ações (`INP`, `BRP`), futuros de ações individuais e *Convenience Yield* (`CYI`);
   - Construção *offline* de curvas a partir de vértices e componentes informados pelo usuário;
   - Referências normativas e bibliográficas.
2. **[Guia de Referência da API](guia_uso_api.md)**
   - Orquestrador central de estruturas a termo (`CurveManager`);
   - Construção por vértices customizados e tratamento de exceções (`DataUnavailableError`);
   - Extração e coleta automatizada de dados (`B3DataLoader` e `B3Scraper`);
   - Instanciação direta de curvas e interpoladores em Cython (`B3CurveBuilder` e `SimpleCurve`);
   - Precificação e métricas de risco de títulos públicos federais (`TPF` e `VNAEngine`).
3. **[Calendários, Convenções de Contagem de Dias e Regimes de Extrapolação](calendarios_convencoes.md)**
   - Convenções `BUS/252`, `ACT/360`, `ACT/365` e `30/360`;
   - Calendário de feriados nacionais e regras de rolagem de datas (`following`, `preceding`, `modified_following`);
   - Regimes de extrapolação constante (`EXTRAP_FLAT`) e por taxa a termo constante (`EXTRAP_FLAT_FORWARD`).

---

## 2. Compilação e Execução de Testes

Para compilar as extensões nativas em C/Cython no ambiente virtual do projeto:

```bash
poetry install
poetry run python setup.py build_ext --inplace
```

Para executar a suíte de testes automatizados:

```bash
poetry run pytest -v
```
