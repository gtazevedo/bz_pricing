# Metodologia de Construção de Estruturas a Termo e Formulações Matemáticas

Este documento apresenta a especificação matemática, as regras de contagem de prazos e os algoritmos de interpolação implementados no pacote `bz-pricing` para a construção de estruturas a termo de taxas de juros, inflação implícita, cupons cambiais, moedas a termo e índices de renda variável nos mercados doméstico (*onshore*) e internacional (*offshore*).

---

## 1. Operadores de Interpolação e Extrapolação em Cython

As classes de interpolação compiladas em Cython no módulo [`bz_pricing.core.interpolators`](../src/bz_pricing/core/interpolators.pyx) operam sobre vetores contíguos de prazos `t_nodes` (em dias úteis $\text{DU}$ ou dias corridos $\text{DC}$), taxas ou preços `y_nodes` e, quando aplicável, prazos secundários `t_sec_nodes`.

### 1.1 Interpolação Exponencial por Taxa a Termo Constante (`FlatForward252Interpolator`)

Para qualquer prazo em dias úteis $\text{DU}$ situado entre dois vértices adjacentes $(\text{DU}_k, y_k)$ e $(\text{DU}_{k+1}, y_{k+1})$, assume-se taxa a termo composta constante na convenção $\text{BUS}/252$. O fator de capitalização acumulado em $\text{DU}$ é definido por:

$$\left(1 + y(\text{DU})\right)^{\frac{\text{DU}}{252}} = \left(1 + y_k\right)^{\frac{\text{DU}_k}{252}} \times \left(1 + f_{k,k+1}\right)^{\frac{\text{DU} - \text{DU}_k}{252}}$$

na qual a taxa a termo anualizada $f_{k,k+1}$ vigente no intervalo $[\text{DU}_k, \text{DU}_{k+1}]$ corresponde a:

$$1 + f_{k,k+1} = \left[ \frac{\left(1 + y_{k+1}\right)^{\frac{\text{DU}_{k+1}}{252}}}{\left(1 + y_k\right)^{\frac{\text{DU}_k}{252}}} \right]^{\frac{252}{\text{DU}_{k+1} - \text{DU}_k}}$$

A expressão analítica equivalente implementada em Cython para a taxa spot interpolada $y(\text{DU})$ assume a forma:

$$y(\text{DU}) = \left[ \left(1 + y_k\right)^{\frac{\text{DU}_k}{252}} \times \left( \frac{\left(1 + y_{k+1}\right)^{\frac{\text{DU}_{k+1}}{252}}}{\left(1 + y_k\right)^{\frac{\text{DU}_k}{252}}} \right)^{\frac{\text{DU} - \text{DU}_k}{\text{DU}_{k+1} - \text{DU}_k}} \right]^{\frac{252}{\text{DU}}} - 1$$

### 1.2 Interpolação Exponencial Base 252 sobre Taxas Lineares (`FlatForward252LinearInterpolator`)

Aplicada às estruturas de cupom cambial (`DOL`, `DOC` e `DCL`), nas quais a cotação da taxa nominal $y_k$ obedece ao regime de capitalização linear em dias corridos ($\text{ACT}/360$), enquanto a interpolação entre vencimentos ocorre em dias úteis ($\text{DU}$):

$$F_k = 1 + y_k \times \frac{\text{DC}_k}{360}, \quad F_{k+1} = 1 + y_{k+1} \times \frac{\text{DC}_{k+1}}{360}$$

O fator interpolado $F(\text{DU})$ para $\text{DU} \in [\text{DU}_k, \text{DU}_{k+1}]$ e prazo corrido associado $\text{DC}$ é obtido via ponderação exponencial em dias úteis e posterior reconversão para taxa linear em base 360:

$$F(\text{DU}) = F_k \times \left( \frac{F_{k+1}}{F_k} \right)^{\frac{\text{DU} - \text{DU}_k}{\text{DU}_{k+1} - \text{DU}_k}}, \quad y(\text{DU}, \text{DC}) = \left( F(\text{DU}) - 1 \right) \times \frac{360}{\text{DC}}$$

### 1.3 Interpolação Linear em Base 360 (`Linear360Interpolator`)

Utilizada para curvas de futuros e *swaps* de taxas de juros internacionais (`SFR`, `EST`, `TIE`, `ZUS`, `ZEU`, `ZMX` e `SDE`). Dados dois nós adjacentes $(\text{DC}_k, y_k)$ e $(\text{DC}_{k+1}, y_{k+1})$, interpola-se linearmente a taxa acumulada no período:

$$A(\text{DC}) = \frac{y_k \cdot \text{DC}_k}{360} + \left( \frac{y_{k+1} \cdot \text{DC}_{k+1}}{360} - \frac{y_k \cdot \text{DC}_k}{360} \right) \times \frac{\text{DC} - \text{DC}_k}{\text{DC}_{k+1} - \text{DC}_k}$$

A taxa anualizada em convenção $\text{ACT}/360$ para o prazo $\text{DC}$ resulta de:

$$y(\text{DC}) = A(\text{DC}) \times \frac{360}{\text{DC}}$$

### 1.4 Interpolação Exponencial de Preços (`PriceInterpolator`)

Empregada na construção de curvas de preços futuros de índices de ações (`INP`, `BRP`, `SML`, `MBR`, `VIX`), contratos futuros de ações individuais e cotações de moedas a termo interpoladas por vértices. Para dois preços adjacentes $P_k$ e $P_{k+1}$ nos prazos $\text{DU}_k$ e $\text{DU}_{k+1}$:

$$P(\text{DU}) = P_k \times \left( \frac{P_{k+1}}{P_k} \right)^{\frac{\text{DU} - \text{DU}_k}{\text{DU}_{k+1} - \text{DU}_k}}$$

---

## 2. Curvas de Juros Prefixadas Domésticas (`PRE`, `SELIC_PRE` e `LTN`)

### 2.1 Curva de DI x Pré (`PRE`)

Construída sobre as taxas de ajuste dos contratos futuros de DI de 1 dia (`DI1`) submetidos a filtro de liquidez (existência de negócios regulares `RglrTxsQty > 0` ou contratos em aberto `OpnIntrst > 0`).

- **Vértice de 1 Dia Útil ($\text{DU} = 1$):** Corresponde à taxa DI *overnight* ($\text{CDI}_t$) divulgada na data-base $t$:

$$\text{PRE}_{1,t} = \text{CDI}_t$$

- **Vértices de Vencimento Futuro ($\text{DU}_n > 1$):** Taxas de ajuste `AdjstdQtTax` dos contratos `DI1` ativos.
- **Interpolação e Extrapolação:** `FlatForward252Interpolator(extrap_left=0, extrap_right=1)`, com arredondamento na quarta casa decimal da taxa percentual (`round_decimals = 6` em representação decimal).

### 2.2 Curva de Selic x Pré (`SELIC_PRE` / `SLP`)

Construída a partir dos contratos futuros de Taxa Média de Operações Compromissadas de 1 Dia (`OC1`). O vértice inicial ($\text{DU} = 1$) utiliza a taxa Selic diária apurada pelo Banco Central do Brasil (série temporal SGS 11, $i_{\text{dia}}$), anualizada sob regime exponencial de 252 dias úteis:

$$\text{SELIC}_{1,t} = \left(1 + i_{\text{dia}}\right)^{252} - 1$$

---

## 3. Curvas de Taxa Básica Financeira (`TFP` / `TBF_PRE`) e Taxa Referencial (`TP` / `TR_PRE` e `TR`)

### 3.1 Estrutura a Termo de TBF x Pré (`TFP` / `TBF_PRE`)

A Taxa Básica Financeira proyectada combina a estrutura a termo prefixada (`PRE`) avaliada em $\text{DU}_i + 21$ com a média aritmética dos fatores históricos de proporcionalidade apurados nos 5 dias úteis imediatamente anteriores à data-base ($\bar{F}_{\text{TBF},t}$):

$$\text{TFP}_{i,t} = \left[ \left( \left(1 + \text{PRE}_{\text{DU}_i + 21, t}\right)^{\frac{21}{252}} - 1 \right) \times \bar{F}_{\text{TBF},t} + 1 \right]^{\frac{252}{21}} - 1$$

Em cada um dos 5 dias úteis anteriores $d$, o fator diário $F_{\text{TBF},d}$ é obtido pela razão entre a TBF mensal divulgada pelo Banco Central (série SGS 253) ajustada para 30 dias corridos e a taxa `PRE` de 30 dias corridos na respectiva data $d$. Os vértices calculados são interpolados por `FlatForward252Interpolator(0, 1)`.

### 3.2 Redutor da Taxa Referencial (`TP` / `TR_PRE`) e Curva DI x TR (`TR`)

A partir da taxa mensal equivalente da curva de TBF nos vértices base, definida por:

$$\text{TFP}_{\text{am},i} = \left(1 + \text{TFP}_{i,t}\right)^{\frac{21}{252}} - 1$$

aplica-se o redutor paramétrico $R = a + b \cdot \text{TFP}_{\text{am},i}$ para a determinação da TR mensal $\text{TR}_{\text{am},i}$ e sua respectiva taxa anualizada $\text{TP}_{i,t}$:

$$\text{TR}_{\text{am},i} = \max\left(0, \; \frac{1 + \text{TFP}_{\text{am},i}}{a + b \cdot \text{TFP}_{\text{am},i}} - 1\right), \quad \text{TP}_{i,t} = \left(1 + \text{TR}_{\text{am},i}\right)^{\frac{252}{21}} - 1$$

Os parâmetros regulatórios $a$ e $b$ obedecem ao escalonamento por faixas de nível da TBF mensal:

| Faixa da TBF Mensal ($\text{TFP}_{\text{am}}$) | Parâmetro $a$ | Parâmetro $b$ | Expressão do Redutor $R$ |
| :--- | :---: | :---: | :--- |
| $\text{TFP}_{\text{am}} \ge 1,3134\%$ | $1,00500000$ | $0,48000000$ | $1,00500000 + 0,48000000 \cdot \text{TFP}_{\text{am}}$ |
| $0,9856\% \le \text{TFP}_{\text{am}} < 1,3134\%$ | $0,99869130$ | $0,96034591$ | $0,99869130 + 0,96034591 \cdot \text{TFP}_{\text{am}}$ |
| $0,8462\% \le \text{TFP}_{\text{am}} < 0,9856\%$ | $1,00500000$ | $0,32000000$ | $1,00500000 + 0,32000000 \cdot \text{TFP}_{\text{am}}$ |
| $0,7323\% \le \text{TFP}_{\text{am}} < 0,8462\%$ | $1,00010877$ | $0,89838243$ | $1,00010877 + 0,89838243 \cdot \text{TFP}_{\text{am}}$ |
| $\text{TFP}_{\text{am}} < 0,7323\%$ | $1,00500000$ | $0,23000000$ | $1,00500000 + 0,23000000 \cdot \text{TFP}_{\text{am}}$ |

A estrutura a termo de **DI x TR (`TR`)** é obtida nos vértices da curva prefixada pelo quociente de fatores e interpolada via `FlatForward252Interpolator(0, 1)`:

$$\text{TR}_{i,t} = \frac{1 + \text{PRE}_{i,t}}{1 + \text{TP}_{i,t}} - 1$$

---

## 4. Curvas Internacionais de Taxas de Juros (`SFR`, `EST`, `TIE`, `ZUS`, `ZEU` e `ZMX`)

As estruturas a termo internacionais operam sob contagem de dias corridos na convenção $\text{ACT}/360$ (`index_by_dc = True`), com precisão de cinco casas decimais na taxa percentual (`round_decimals = 7` em representação decimal).

### 4.1 Curvas a Termo de Contratos Futuros (`SFR`, `EST` e `TIE`)

Construídas a partir dos contratos futuros de taxas *overnight* negociados em bolsa (`SFR` para SOFR dos Estados Unidos, `EST` para €STR da Zona do Euro e `TIE` para F-TIIE do México):

- **Vértice Inicial ($\text{DC} = 1$):** Taxa *overnight* divulgada pelo respectivo banco central emissor (Federal Reserve Bank of New York, European Central Bank ou Banco de México).
- **Vértices de Vencimento:** Taxas de ajuste dos contratos futuros.
- **Interpolação e Extrapolação:** `Linear360Interpolator(extrap_left=0, extrap_right=0)`.

### 4.2 Curvas Zero Cupom de Swaps Offshore (`ZUS` / `LIB`, `ZEU` e `ZMX`)

Representam a taxa linear spot acumulada desde a data-base até o prazo corrido $\text{DC}_v$. Dado o fator de desconto $\text{FD}_v$ de um *Overnight Indexed Swap* (OIS) de prazo $\text{DC}_v$, a taxa zero cupom linear na base 360 é determinada por:

$$y_{\text{zero}}(\text{DC}_v) = \left( \frac{1}{\text{FD}_v} - 1 \right) \times \frac{360}{\text{DC}_v}$$

Quando derivadas de uma grade de taxas a termo $f_k$ com intervalos $\Delta \text{DC}_k$, o fator de desconto acumulado resulta do produtório:

$$\text{FD}_n = \prod_{k=1}^{n} \left( 1 + f_k \times \frac{\Delta \text{DC}_k}{360} \right)^{-1}$$

Na construção a partir de tabelas referenciais de *swaps*, o método `_extract_linear360_knots` identifica os nós primários onde ocorre mudança de inclinação da taxa linear acumulada e interpola os prazos intermediários via `Linear360Interpolator(0, 0)`.

---

## 5. Estruturas a Termo de Inflação: Cupom Sujo (`DIC` e `DIM`) e Cupom Limpo (`DPL`)

### 5.1 Cupom Sujo de IPCA (`DIC`) e Cupom Sujo de IGP-M (`DIM`)

As curvas de cupom sujo de inflação combinam a curva nominal prefixada (`PRE`) com a amostra de expectativas de inflação nos vértices discretos de divulgação do índice ($k_i$):

1. **Vértice de 1 Dia Útil ($\text{DU} = 1$):** Assume o valor do primeiro vértice da curva prefixada ($\text{DIC}_{1,t} = \text{DIM}_{1,t} = \text{PRE}_{1,t}$), uma vez que não há divulgação de índice no próprio dia ($E_t[\text{I}_0] = 0$).
2. **Expectativa Acumulada de Inflação nos Vértices de Divulgação ($k_i$):** A partir da taxa de cupom sujo $\text{Cp}_{k_i,t}$ informada para a data de divulgação $k_i$ (prazo $\text{DU}_{k_i}$), extrai-se a inflação acumulada esperada $E_t[\text{I}_{k_i}]$:

$$E_t[\text{I}_{k_i}] = \frac{\left(1 + \text{PRE}_{k_i,t}\right)^{\frac{\text{DU}_{k_i}}{252}}}{\left(1 + \text{Cp}_{k_i,t}\right)^{\frac{\text{DU}_{k_i}}{252}}} - 1$$

3. **Degraus de Divulgação nos Primeiros 12 Meses ($\text{DU}_{\text{início}, k_i} \le \text{DU}_p \le \text{DU}_{k_i}$):** Como o número acumulado de divulgações do índice permanece constante entre duas datas consecutivas de apuração, o fator $1 + E_t[\text{I}_{k_i}]$ é mantido fixo ao longo de toda a janela $[\text{DU}_{\text{início}, k_i}, \text{DU}_{k_i}]$, o que define o cupom sujo em cada dia útil $\text{DU}_p$ do patamar:

$$\text{Cp}_{p,t} = \left[ \frac{\left(1 + \text{PRE}_{p,t}\right)^{\frac{\text{DU}_p}{252}}}{1 + E_t[\text{I}_{k_i}]} \right]^{\frac{252}{\text{DU}_p}} - 1$$

4. **Vértices de Longo Prazo ($\text{DU} > 265$):** Interpolados e extrapolados a partir dos nós longos de cupom via `FlatForward252Interpolator(0, 1)`.

### 5.2 Cupom Limpo de IPCA (`DPL` / `DAP`)

Construída a partir da curva `PRE`, dos contratos futuros de Cupom de IPCA (`DAP`) e da prévia de inflação corrente $E_t[\text{IPCA}_0]$:

1. **Vértice de 1 Dia Útil ($\text{DU} = 1$):** Desconta da taxa $\text{PRE}_{1,t}$ a variação pro-rata de 1 dia útil da prévia do IPCA entre o dia 15 anterior corrigido para dia útil ($15_a$) e o dia 15 posterior corrigido para dia útil ($15_p$), cujo intervalo soma $\text{DU}_{15a,15p}$ dias úteis:

$$\text{DPL}_{1,t} = \left[ \frac{\left(1 + \text{PRE}_{1,t}\right)^{\frac{1}{252}}}{\left(1 + E_t[\text{IPCA}_0]\right)^{\frac{1}{\text{DU}_{15a,15p}}}} \right]^{252} - 1$$

2. **Extração e Interpolação da Inflação Implícita nos Vencimentos de `DAP`:** Em cada vértice $\text{DU}_n$ (incluindo $\text{DU} = 1$ e os vencimentos de `DAP`), calcula-se a taxa de inflação implícita $\pi_{n,t}$:

$$\pi_{n,t} = \frac{1 + \text{PRE}_{n,t}}{1 + \text{DPL}_{n,t}} - 1$$

3. **Recomposição do Cupom Limpo (`InflationCurve`):** Os vértices $\pi_{n,t}$ são interpolados por `FlatForward252Interpolator(0, 1)`, e a taxa `DPL` para qualquer prazo $\text{DU}$ é recomposta analiticamente por:

$$\text{DPL}(\text{DU}) = \frac{1 + \text{PRE}(\text{DU})}{1 + \pi(\text{DU})} - 1$$

---

## 6. Curvas de Moedas, Cupons Cambiais e Spreads Internacionais

### 6.1 Cupons Cambiais de Dólar: Sujo (`DOL`) e Limpo (`DOC` e `DCL`)

- **Cupom Cambial Sujo (`DOL`):** Construído diretamente a partir dos vencimentos do contrato futuro de Cupom Cambial (`DDI`), indexado simultaneamente em dias úteis ($\text{DU}$) e dias corridos ($\text{DC}$), e interpolado via `FlatForward252LinearInterpolator(0, 1)`.
- **Cupom Cambial Limpo (`DOC` e `DCL`):**Corrige a defasagem da PTAX de referência do contrato `DDI` ($\text{Ptax}_{t-1}$) em relação à cotação spot do dólar cupom limpo no dia $t_0$ ($\text{DolCL}_{t_0}$):
  - **Vértice 1 ($\text{DU} = 1$, prazo corrido $\text{DC}_1$):** Quando calculado a partir da cotação do casado de 1 dia útil ($\text{Casado}_{1\text{d}}$, em pontos de milésimo de real) e da taxa $\text{CDI}_t$:

$$\text{DolCL}_{d_1} = \text{DolCL}_{t_0} + \frac{\text{Casado}_{1\text{d}}}{1000}, \quad \text{DOC}_1 = \left[ \frac{\left(1 + \text{CDI}_t\right)^{\frac{1}{252}}}{\text{DolCL}_{d_1} / \text{DolCL}_{t_0}} - 1 \right] \times \frac{360}{\text{DC}_1}$$

  - **Vértices de Vencimento de `DDI` ($\text{DU}_n > 1$):**

$$\text{DOC}_n = \left[ \left( 1 + \text{DDI}_n \times \frac{\text{DC}_n}{360} \right) \times \frac{\text{DolCL}_{t_0}}{\text{Ptax}_{t-1}} - 1 \right] \times \frac{360}{\text{DC}_n}$$

  Os vértices resultantes são interpolados e extrapolados por `FlatForward252LinearInterpolator(0, 1)`.

### 6.2 Spreads de Taxas Offshore (`SDE`, `SGP`, `SMX`, `SYD`) e Cupons Cruzados (`EUC` e `YCL`)

- **Curvas de Spread Offshore:** Representam o diferencial linear entre a taxa externa da moeda de referência e a taxa externa em dólar americano na base 360:

$$\text{Spread}_n = y_{\text{Moeda}, n} - y_{\text{USD}, n}$$

- **Curvas de Cupom Cambial de Euro (`EUC`) e Iene (`YCL`):** Implementadas na classe `CrossCouponCurve`, compõem o fator de cupom limpo de dólar (`DOC`) com o respectivo spread internacional (`SDE` para Euro, `SYD` para Iene) em cada prazo $(\text{DU}_n, \text{DC}_n)$:

$$\text{TC}_n = \left[ \left( 1 + \text{DOC}_n \times \frac{\text{DC}_n}{360} \right) \times \left( 1 + \text{Spread}_n \times \frac{\text{DC}_n}{360} \right) - 1 \right] \times \frac{360}{\text{DC}_n}$$

### 6.3 Estruturas a Termo de Moedas (`PTX`, `EUR` e `JPY`)

Implementadas na classe `CurrencyForwardCurve`, projetam a taxa de câmbio a termo pela relação de arbitragem entre o fator de capitalização prefixado doméstico (`PRE`, em $\text{BUS}/252$) e o fator de cupom cambial da moeda estrangeira (em $\text{ACT}/360$), com truncamento na sétima casa decimal (`round_decimals = 7`, `round_mode = "trunc"`):

- **Dólar a Termo (`PTX`):** No primeiro vértice ($\text{DU} = 1$), assume a cotação $\text{Ptax}_{t_0}$ do próprio dia. Para todos os demais prazos ($\text{DU}_n > 1$), aplica a paridade sobre a cotação $\text{Ptax}_{t-1}$ do dia útil anterior e o cupom sujo `DOL`:

$$\text{PTX}_1 = \text{Ptax}_{t_0}, \quad \text{PTX}_n = \text{Ptax}_{t-1} \times \frac{\left(1 + \text{PRE}_n\right)^{\frac{\text{DU}_n}{252}}}{1 + \text{DOL}_n \times \frac{\text{DC}_n}{360}} \quad (\text{DU}_n > 1)$$

- **Euro a Termo (`EUR`) e Iene a Termo (`JPY`):** Aplicam em todos os vértices a cotação à vista de fechamento $\text{Spot}_{t_0}$ combinada com a curva `PRE` e o respectivo cupom cambial limpo cruzado (`EUC` ou `YCL`):

$$\text{Fwd}_n = \text{Spot}_{t_0} \times \frac{\left(1 + \text{PRE}_n\right)^{\frac{\text{DU}_n}{252}}}{1 + \text{TC}_n \times \frac{\text{DC}_n}{360}}$$

---

## 7. Curvas de Índices de Ações, Futuros de Ações e Convenience Yield

### 7.1 Estruturas a Termo de Preços de Índices e Ações (`INP`, `BRP`, `SML`, `MBR` e `VIX`)

Construídas a partir da cotação à vista de fechamento do índice ($\text{Spot}_t$) e dos preços de ajuste dos contratos futuros (`IND` para Ibovespa, `BRI` para IBrX-50, além dos contratos futuros de ações individuais):

1. **Vértice Inicial ($\text{DU} = 1$):** Igual à cotação à vista $\text{Spot}_t$.
2. **Prazos Anteriores ao Primeiro Vencimento Futuro ($1 < \text{DU}_v < \text{DU}_1$):** Evoluem exponencialmente desde a data-base ($t = 0$) até o primeiro vencimento futuro ($\text{DU}_1$, com preço $\text{Fut}_1$):

$$P(\text{DU}_v) = \text{Spot}_t \times \left( \frac{\text{Fut}_1}{\text{Spot}_t} \right)^{\frac{\text{DU}_v}{\text{DU}_1}}$$

No método `B3CurveBuilder.build_equity_index_curve`, a inserção de uma âncora exata em $\text{DU} = 2$ com valor $\text{Spot}_t \cdot (\text{Fut}_1 / \text{Spot}_t)^{2 / \text{DU}_1}$ permite que o `PriceInterpolator(0, 0)` reproduza analiticamente a expressão acima em todo o intervalo $[2, \text{DU}_1]$, preservando os nós esparsos dos vencimentos futuros.
3. **Prazos entre Vencimentos e Após o Último Futuro:** Interpolados exponencialmente por `PriceInterpolator(extrap_left=0, extrap_right=0)`, mantendo o último preço futuro constante após o derradeiro vencimento listado.

### 7.2 Estruturas a Termo de Custo de Carrego / Convenience Yield (`CYI`, `CYM`, `CYX` e Ações)

Implementadas em Cython na classe [`ConvenienceYieldCurve`](../src/bz_pricing/core/curves.pyx):

1. **Vértice de 1 Dia Útil ($\text{DU} = 1$):** Recebe a taxa prefixada de 1 dia útil ($\text{PRE}_{1,t} = \text{CDI}_t$).
2. **Vértices de Vencimento Futuro ($\text{DU}_n$):** Em cada vencimento de contrato futuro com preço $\text{Fut}_n$, apura-se a taxa anualizada de carrego implícito frente à curva `PRE`:

$$\text{TxCY}_n = \left[ \frac{\text{Spot}_t \times \left(1 + \text{PRE}_n\right)^{\frac{\text{DU}_n}{252}}}{\text{Fut}_n} \right]^{\frac{252}{\text{DU}_n}} - 1$$

3. **Prazos Intermediários e Extrapolação:** Os vértices $(\text{DU}_n, \text{TxCY}_n)$ são interpolados por `FlatForward252Interpolator(extrap_left=0, extrap_right=0)`, mantendo patamar constante antes do primeiro vencimento ($2 \le \text{DU} < \text{DU}_1$) e após o último vencimento ($\text{DU} > \text{DU}_{\text{último}}$).

---

## 8. Construção de Curvas a partir de Vértices e Componentes do Usuário

O `bz-pricing` opera como um motor quantitativo independente de arquivos externos quando os insumos dos componentes são informados diretamente pelo usuário (`allow_download=False`). A seguir especificam-se as assinaturas de `CurveManager.get_curve(...)` para a injeção de componentes por família de curvas.

### 8.1 Curvas de Inflação (`DIC`, `DIM` e `DPL`)

- **Cupom Sujo (`DIC` e `DIM`):** Requerem a curva nominal (`pre_curve` ou `pre_data`) associada a uma amostra de vértices de expectativas (`informants_df` com colunas `DU`, `rate` ou `e_inf`, e opcionalmente `du_start`) ou a um dicionário de projeções mensais (`focus_monthly` no formato `{(ano, mes): taxa_decimal}`):

```python
import datetime
import pandas as pd
from bz_pricing.curves.manager import CurveManager

manager = CurveManager(allow_download=False)
base_date = datetime.date(2026, 10, 30)

df_pre = pd.DataFrame({
    "DU": [1, 10, 31, 52, 126, 252, 504],
    "rate": [13.65, 13.70, 13.75, 13.80, 13.90, 14.00, 14.10]
})
df_informantes = pd.DataFrame({
    "du_start": [1, 11, 32, 252, 504],
    "DU":       [1, 31, 52, 252, 504],
    "rate":     [13.65, 8.83, 8.95, 7.80, 7.65]
})

curve_dic = manager.get_curve("DIC", base_date, pre_data=df_pre, informants_df=df_informantes)
```

- **Cupom Limpo (`DPL`):** Recebe `pre_curve` (ou `pre_data`), `dap_df` (contratos futuros ou vértices `(DU, rate)`) e a prévia de inflação `eipca_0` (ou diretamente a taxa do primeiro vértice `v1_rate`):

```python
df_dap = pd.DataFrame({
    "DU": [14, 135, 252, 504],
    "rate": [9.62, 6.80, 7.65, 7.80]
})
curve_dpl = manager.get_curve("DPL", base_date, pre_data=df_pre, dap_df=df_dap, eipca_0=0.0022)
```

### 8.2 Cupons Cambiais, Moedas a Termo e Zero Cupom Offshore (`DOC`, `EUC`, `YCL`, `PTX`, `EUR`, `JPY`, `ZUS`)

```python
# 1. Cupom Cambial Limpo (DOC) a partir dos vértices de DDI e cotações de referência
df_ddi = pd.DataFrame({
    "DU": [1, 21, 42, 126, 252],
    "DC": [4, 31, 62, 182, 365],
    "rate": [-73.16, 1.07, 3.16, 4.85, 5.25]
})
curve_doc = manager.get_curve(
    "DOC", base_date,
    ddi_data=df_ddi,
    dol_cl=5.7520,
    ptax_tm1=5.7610,
    casado_1d=1.20,
    cdi_rate=13.65
)

# 2. Cupom de Euro (EUC) e Dólar a Termo (PTX)
sde_vertices = {30: -1.45, 90: -1.55, 180: -1.65, 360: -1.75}
curve_euc = manager.get_curve("EUC", base_date, doc_curve=curve_doc, sde_data=sde_vertices)
curve_ptx = manager.get_curve(
    "PTX", base_date,
    pre_data=df_pre,
    foreign_curve=curve_doc,
    spot_rate=5.7520,
    spot_rate_tm1=5.7610
)

# 3. Curva Zero Cupom Internacional (ZUS) a partir de fatores de desconto OIS por DC
fd_sofr = {30: 0.9965, 90: 0.9895, 180: 0.9790, 360: 0.9585}
curve_zus = manager.get_curve("ZUS", base_date, discount_factors=fd_sofr)
```

### 8.3 Índices de Ações, Convenience Yield e Injeção Direta de Vértices

```python
# 1. Índice Bovespa Futuro (INP) e Convenience Yield (CYI)
df_ind = pd.DataFrame({"DU": [31, 72, 111], "AdjstdQt": [170165.0, 173803.0, 177269.0]})
curve_inp = manager.get_curve("INP", base_date, data=df_ind, spot_price=167938.10)
curve_cyi = manager.get_curve("CYI", base_date, spot_price=167938.10, futures_curve=curve_inp, pre_data=df_pre)

# 2. Interpolação direta de vértices arbitrários em qualquer estrutura configurada
curve_pre_custom = manager.get_curve("PRE", data={1: 13.65, 21: 13.72, 63: 13.85, 252: 14.10})
```

---

## 9. Referências Normativas e Bibliográficas

As formulações matemáticas, convenções de contagem de prazos e regras de arredondamento implementadas neste pacote seguem as especificações técnicas das instituições reguladoras e autorreguladoras do mercado financeiro e de capitais:

- **B3 S.A. – Brasil, Bolsa, Balcão.** *Manual de Curvas da B3: Metodologia de Apuração das Estruturas a Termo de Taxas de Juros, Moedas e Índices*. São Paulo: B3, 2024. Disponível em: <https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-de-derivativos/precos-referenciais/manuais-de-curvas/>.
- **B3 S.A. – Brasil, Bolsa, Balcão.** *Preços Referenciais – Taxas Referenciais BM&FBOVESPA*. São Paulo: B3. Disponível em: <https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-de-derivativos/precos-referenciais/taxas-referenciais-bm-fbovespa/>.
- **ANBIMA – Associação Brasileira das Entidades dos Mercados Financeiro e de Capitais.** *Metodologia de Precificação de Títulos Públicos Federais e Estrutura a Termo das Taxas de Juros (ETTJ)*. Rio de Janeiro: ANBIMA. Disponível em: <https://www.anbima.com.br/pt_br/informar/precos-e-indices/precos/titulos-publicos.htm>.
- **Banco Central do Brasil (BCB) / Conselho Monetário Nacional (CMN).** *Resolução CMN nº 5.124, de 28 de março de 2024: Metodologia de cálculo da Taxa Básica Financeira (TBF) e da Taxa Referencial (TR)*. Brasília: BCB, 2024. Disponível em: <https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo?tipo=Resolu%C3%A7%C3%A3o%20CMN&numero=5124>.
- **Banco Central do Brasil (BCB).** *Sistema Gerenciador de Séries Temporais (SGS)*. Brasília: BCB. Disponível em: <https://www3.bcb.gov.br/sgspub/>.
- **Secretaria do Tesouro Nacional (STN).** *Regras e Fórmulas de Cálculo dos Títulos Públicos Federais no Mercado Secundário*. Brasília: Ministério da Fazenda. Disponível em: <https://www.tesourodireto.com.br/titulos/historico-de-precos-e-taxas.htm>.
- **Federal Reserve Bank of New York.** *Secured Overnight Financing Rate (SOFR) Data & Calculation Methodology*. New York: FRBNY. Disponível em: <https://www.newyorkfed.org/markets/reference-rates/sofr>.
- **European Central Bank (ECB).** *Euro Short-Term Rate (€STR) Methodology and Policies*. Frankfurt: ECB. Disponível em: <https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_short-term_rate/html/index.en.html>.
