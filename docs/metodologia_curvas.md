# Metodologia de Construção de Curvas e Fundamentação Normativa

Esta documentação detalha as regras de negócio, a formulação matemática e o respaldo regulatório das curvas de juros construídas pelo `bz-pricing`, com base no **Manual de Curvas da B3 (V21)** e nas resoluções do **Banco Central do Brasil (BCB)**.

---

## 1. Curva de DI x Pré-Fixado (`PRE`)

- **Referência:** *Manual de Curvas B3, Seção 2.1 (págs. 7–9)*
- **Ativo-Objeto:** Contrato Futuro de DI de 1 Dia (`DI1`).
- **Base de Cálculo:** $DU / 252$ (Dias Úteis / Feriados B3).

### 1.1 Formulação do Vértice 1 (Overnight)
O primeiro vértice da curva corresponde ao CDI de 1 dia útil ($DU = 1$), apurado e divulgado pela B3:
$$PRE_{1,t} = CDI_t$$

### 1.2 Interpolação: Taxa a Termo Constante (Flat Forward 252)
Para qualquer prazo útil $DU$ situado entre dois nós conhecidos $t_1$ e $t_2$, a taxa a termo composta contínua é constante:

$$\left(1 + \frac{PRE(DU)}{100}\right)^{\frac{DU}{252}} = \left(1 + \frac{y_1}{100}\right)^{\frac{t_1}{252}} \times \left(1 + \frac{f_{1,2}}{100}\right)^{\frac{DU - t_1}{252}}$$

onde a taxa forward $f_{1,2}$ é dada por:

$$1 + \frac{f_{1,2}}{100} = \left[ \frac{\left(1 + \frac{y_2}{100}\right)^{\frac{t_2}{252}}}{\left(1 + \frac{y_1}{100}\right)^{\frac{t_1}{252}}} \right]^{\frac{252}{t_2 - t_1}}$$

---

## 2. Curva de Selic x Pré-Fixado (`SELIC_PRE`)

- **Referência:** *Manual de Curvas B3, Seção 2.2 (págs. 9–11)*
- **Ativo-Objeto:** Contrato Futuro de Taxa Selic (`OC1`).

### 2.1 Vértice 1 — Selic Diária (BCB SGS Série 11)
A taxa Selic diária apurada pelo Banco Central em operações compromissadas com títulos públicos federais é divulgada em termos percentuais ao dia ($i_{\text{dia}}$) pela série temporal **SGS 11**.

A anualização exponencial em base 252 dias úteis é dada por:

$$SELIC_{\text{anual}} = (1 + i_{\text{dia}})^{252} - 1$$

> **Nota Técnica:** Não se utiliza a série SGS 4389 (que é uma taxa anualizada nominal indicativa com convenção de corte), mas sim a série primária SGS 11 capitalizada dia a dia.

---

## 3. Curva de TBF x Pré (`TBF_PRE`) e TR x Pré (`TR_PRE`)

- **Referência:** *Manual de Curvas B3, Seções 2.4 e 2.5 (págs. 14–18)*
- **Normativo:** *Resolução CMN nº 5.124 / Comunicado BCB nº 31.751*

### 3.1 Curva TBF x Pré
A Taxa Básica Financeira (TBF) é obtida a partir do fator médio dos últimos 5 dias úteis de apuração ($MediaTBF_{fator,t}$):

$$TFP_{i,t} = \left( \left[ \left( \left( 1 + \frac{PRE_{i+21,t}}{100} \right)^{\frac{21}{252}} - 1 \right) \times MediaTBF_{fator,t} + 1 \right]^{\frac{252}{21}} \right) - 1$$

### 3.2 Redutor da Taxa Referencial (TR)
A TR mensal é calculada a partir da TBF mensal ($TFP_{am}$) aplicando o Redutor $R$:

$$TR_{am} = \frac{1 + TFP_{am}}{R} - 1$$

Onde o redutor é escalonado em função da TBF mensal conforme a **Resolução CMN nº 5.124**:

| Faixa da TBF Mensal ($TFP_{am}$) | Parâmetro $a$ | Parâmetro $b$ | Redutor $R = a + b \cdot TFP_{am}$ |
|:---|:---:|:---:|:---|
| $TFP_{am} \ge 1.3134\%$ | $1.00500000$ | $0.48000000$ | $1.005 + 0.48 \cdot TFP_{am}$ |
| $0.9856\% \le TFP_{am} < 1.3134\%$ | $0.99869130$ | $0.96034591$ | $0.99869130 + 0.96034591 \cdot TFP_{am}$ |
| $0.8462\% \le TFP_{am} < 0.9856\%$ | $1.00500000$ | $0.32000000$ | $1.005 + 0.32 \cdot TFP_{am}$ |
| $0.7323\% \le TFP_{am} < 0.8462\%$ | $1.00010877$ | $0.89838243$ | $1.00010877 + 0.89838243 \cdot TFP_{am}$ |
| $TFP_{am} < 0.7323\%$ | $1.00500000$ | $0.23000000$ | $1.005 + 0.23 \cdot TFP_{am}$ |

---

## 4. Curvas de SOFR: Futuros (`SFR`) vs. Swaps Zero Cupom (`ZUS` / `LIB`)

A B3 publica duas estruturas a termo de SOFR com mercados e metodologias completamente distintas:

### 4.1 Curva Forward de Futuros SOFR (`SFR` — Seção 2.8, pág. 21)
- **Fonte Primária:** Contratos Futuros de SOFR negociados na B3 (`SFRU26`, `SFRZ26`, etc.).
- **Vértice 1:** Taxa SOFR *Overnight* apurada e divulgada pelo Federal Reserve Bank of New York (NY Fed).
- **Interpolação:** Linear na base $DC / 360$ (Dias Corridos):
  $$Fator(DC) = 1 + \left( \frac{Taxa}{100} \times \frac{DC}{360} \right)$$

### 4.2 Curva Zero Cupom de Swaps SOFR (`ZUS` / `LIB` — Seção 2.11, pág. 26)
- **Fonte Primária:** Cotações de Swaps OIS de SOFR offshore capturadas via Refinitiv/LSEG (RICs `USDSROISONZ=R` a `USDSROIS15YZ=R`).
- **Arquivo Oficial B3:** `taxas_LIB_AAAAMMDD.csv`.
- **Horizonte de Avaliação:** Até 30 anos ($DC = 12.400$), com taxa flat após o último vértice disponível.

---

## 5. Curvas de Inflação: IPCA (`DIC` e `DPL`) e IGP-M (`DIM`)

- **Referência:** *Manual de Curvas B3, Seções 3.1, 3.2 e 3.3 (págs. 29–47)*

### 5.1 Cupom Sujo de IPCA (`DIC` — Seção 3.1) e Cupom Sujo de IGP-M (`DIM` — Seção 3.3)
As curvas de cupom sujo de inflação (`DIC` e `DIM`) são construídas a partir da curva `DI x Pré` (`PRE`) e da **Amostra de Expectativas de Inflação do Pool de Informantes da B3** (~18 a 23 vértices base $k_i$ nas datas de divulgação do índice pelo IBGE/FGV):

1. **Vértice 1 ($DU = 1$):** Igual à taxa de 1 dia útil da curva `DI x Pré` ($DIC_{1,t} = DIM_{1,t} = PRE_{1,t}$).
2. **Vértices dos Informantes ($k_i$):** A partir do cupom informado em cada data de divulgação $k_i$, obtém-se o fator acumulado de inflação esperada (Eq. 3.1.2 e Eq. 3.3.2):
   $$E(\text{Índice})_{k_i,t} = \frac{\left(1 + \frac{PRE_{k_i,t}}{100}\right)^{\frac{du_{k_i}}{252}}}{\left(1 + \frac{Cp_{k_i,t}}{100}\right)^{\frac{du_{k_i}}{252}}} - 1$$
3. **Vértices Intermediários em Degraus de Divulgação ($du_{k_i} \le du_i < du_{k_{i+1}}$):** Como o fator acumulado de inflação esperada $E(\text{Índice})_{k_i,t}$ permanece constante até a próxima data de divulgação $k_{i+1}$, o cupom sujo em cada dia útil $du_i$ dentro da janela é dado diretamente pelas Eq. 3.1.3 e Eq. 3.3.3:
   $$Cp_{i,t} = \left( \left[ \frac{\left(1 + \frac{PRE_{i,t}}{100}\right)^{\frac{du_i}{252}}}{1 + E(\text{Índice})_{k_i,t}} \right]^{\frac{252}{du_i}} - 1 \right) \times 100$$
4. **Interpolação e Extrapolação de Longo Prazo (Itens 3.1.3, 3.1.4, 3.3.3 e 3.3.4):** Entre os vértices longos dos informantes e após o último vértice disponível, aplica-se a **Interpolação e Extrapolação Exponencial Base 252 (Taxa a Termo Constante — `FlatForward252Interpolator(0, 1)`, item 1.4.2)**.

### 5.2 Cupom Limpo de IPCA (`DAP` / `DPL` — Seção 3.2)
- **Vértice 1 ($DU = 1$):** Calculado pela relação entre o primeiro vértice da curva `DI x Pré` (`PRE_1`) e a prévia diária de inflação do IPCA ($E(\text{IPCA})_{0,t}$ — ANBIMA `IPCA_prévia`), conforme Eq. 3.2.1 (pág. 36):
  $$DPL_{1,t} = \left( \left[ \frac{\left(1 + \frac{PRE_{1,t}}{100}\right)^{\frac{1}{252}}}{1 + E(\text{IPCA})_{0,t}} \right]^{252} - 1 \right) \times 100$$
- **Demais Vértices:** Preços de ajuste dos contratos futuros de Cupom de IPCA (`DAP`), com vencimento no dia 15 do mês (ou primeiro dia útil subsequente).
- **Interpolação e Extrapolação:** Taxa a Termo Exponencial Base 252 (`FlatForward252Interpolator(0, 1)` — item 1.4.2).

---

## 6. Curvas Internacionais: ESTR (EST / ZEU), F-TIIE (TIE / ZMX) e SOFR (SFR / ZUS)

- **Referência:** *Manual de Curvas B3, Seções 2.7 a 2.12 (págs. 19–28)*
- **Convenção de Prazo:** $DC / 360$ (Dias Corridos / Base ACT/360, com `index_by_dc = True`).
- **Arredondamento:** Taxa anualizada % com 5 casas decimais (`round_decimals = 7` em taxa decimal).

### 6.1 Curvas Forward Marginais de Futuros (EST, TIE e SFR)
- **Definição:** Refletem a taxa de juros a termo acordada entre os vencimentos dos contratos futuros negociados na B3.
- **Insumos:**
  - **Vértice 1:** Taxa overnight oficial mais recente do banco central emissor (ECB para ESTR, Banxico para TIIE, Fed NY para SOFR).
  - **Vértices Futuros:** Taxas de ajuste dos contratos futuros negociados na B3 (`EST`, `TIE`, `SFR`).
- **Interpolação:** Linear 360 (`Linear360Interpolator(0, 0)`) com Extrapolação Flat Fim no último vencimento.

### 6.2 Curvas Zero Cupom Offshore (ZEU, ZMX e ZUS)
- **Definição:** Representam a estrutura a termo da taxa spot acumulada desde $t=0$ até a data futura $DC$.
- **Cálculo da Taxa Linear 360 (Seções 2.10, 2.11 e 2.12):**
  A partir dos fatores de desconto acumulados $FD(DC)$:
  $$FD(DC_n) = \prod_{k=1}^n \frac{1}{1 + F_k \times \frac{\Delta DC_k}{36000}}$$
  $$TxLinJUR(DC_n) = \left( \frac{1}{FD(DC_n)} - 1 \right) \times \frac{36000}{DC_n}$$
- **Extração de Vértices do Provedor (`ZUS` / `LIB`):** Quando carregada a partir de `taxas_LIB_*.csv`, o `CurveManager` extrai apenas os ~73 vértices originais de swap fornecidos pela LSEG/Refinitiv (Tabela 1 do Manual B3) onde a taxa linear a termo muda de inclinação, interpolando todos os demais prazos via `Linear360Interpolator(0, 0)`.

---

## 7. Curvas de Moedas, Cupons Cambiais e Spreads (Capítulo 4 B3)

- **Referência:** *Manual de Curvas B3, Seções 4.1 a 4.12 (págs. 48–66)*

### 7.1 Forwards Cambiais: PTX (Dólar), EUR (Euro) e JPY (Iene)
Projetam a taxa de câmbio a termo pela paridade da taxa pré interna (`PRE`) contra o cupom cambial da moeda estrangeira (implementadas na classe analítica `CurrencyForwardCurve` sobre as curvas base interpoladas por vértices):

- **PTX (`PTX` — Seção 4.1, págs. 48–49):**
  - **Vértice 1 ($DU = 1$):** Igual à cotação $\text{Ptax}_{t0}$ de fechamento divulgada pelo BCB no dia $t_0$.
  - **Demais Vértices ($n \ge 2$, $DU_n > 1$):** Utilizam a cotação $\text{Ptax}_{t-1}$ (fechamento do dia útil anterior) combinada com as curvas `PRE` (interpolada sobre os futuros de `DI1`) e `DOL` (interpolada sobre os futuros de `DDI`):
    $$PTX_n = \frac{\left(1 + \frac{PRE_n}{100}\right)^{\frac{du_n}{252}}}{1 + \frac{DOL_n}{100} \times \frac{dc_n}{360}} \times \text{Ptax}_{t-1}$$
  - **Truncamento:** 7ª casa decimal (`round_decimals = 7`, `round_mode = "trunc"`).
- **EUR (`EUR` — Seção 4.2, págs. 50–51):** Forward Reais x Euro. Em todos os vértices utiliza a cotação de fechamento $\text{Tx}_{\text{Reais x Euro}}$ apurada pela B3 no dia $t_0$ combinada com as curvas `PRE` e `EUC`:
  $$EUR_n = \frac{\left(1 + \frac{PRE_n}{100}\right)^{\frac{du_n}{252}}}{1 + \frac{EUC_n}{100} \times \frac{dc_n}{360}} \times \text{Tx}_{\text{Reais x Euro}}$$
- **JPY (`JPY` — Seção 4.3, pág. 52):** Forward Reais x Iene. Utiliza a cotação $\text{Tx}_{\text{Reais x Iene}}$ e o Cupom de Iene (`YCL`), ou interpolação sobre os vértices de preços via `PriceInterpolator(0, 1)` (item 1.4.5).

### 7.2 Cupons Cambiais de Dólar: Sujo (`DOL`) e Limpo (`DOC` e `DCL`)
- **DOL (`DOL` — Seção 4.4):** Cupom Cambial Sujo construído a partir dos vencimentos dos contratos futuros de `DDI` via `FlatForward252LinearInterpolator(0, 1)` (item 1.4.3), com `t_nodes = DU` e `t_sec_nodes = DC`.
- **DOC e DCL (`DOC` e `DCL` — Seções 4.5 e 4.6, págs. 57–59):** Cupons Cambiais Limpos construídos a partir do Vértice 1 ($DU = 1$, apurado com base em informantes / casado) e dos vencimentos do Contrato Futuro de `DDI` ajustados pela razão $\text{DolCL}_{t0} / \text{Ptax}_{t-1}$ (Eq. 4.5.3 e 4.6.3):
  $$DOC_n = \left( \left[ \left(1 + \frac{DDI_n}{100} \times \frac{dc_n}{360}\right) \times \frac{\text{DolCL}_{t0}}{\text{Ptax}_{t-1}} \right] - 1 \right) \times \frac{36000}{dc_n}$$
  Os vértices intermediários e posteriores ao último vencimento de `DDI` são interpolados e extrapolados via `FlatForward252LinearInterpolator(0, 1)` (item 1.4.3), indexados em `t_nodes = DU` e `t_sec_nodes = DC`.

### 7.3 Curva de Cupom de Euro (`EUC` — Seção 4.7):
Implementada na classe `CrossCouponCurve`, avalia analiticamente a paridade entre as curvas `DOC` (interpolada a partir dos 45 vértices de V1 + `DDI`) e `SDE` (interpolada a partir dos 65 nós de swaps `TEU - TUS`):
$$EUC_n = \left[ \left(1 + DOC_n \times \frac{DC_n}{36000}\right) \times \left(1 + SDE_n \times \frac{DC_n}{36000}\right) - 1 \right] \times \frac{36000}{DC_n}$$
Arredondada na 4ª casa decimal da taxa percentual (`round_decimals = 6` em decimal).

### 7.4 Curva de Spread Libor/OIS Euro x Dólar (`SDE` — Seção 4.9):
Reflete o diferencial de taxas de juros offshore de Euro vs USD:
$$TC_n = TEU_n - TUS_n$$
Construída extraindo os ~65 vértices base de swaps da LSEG/Refinitiv (Tabela 1 do Manual B3) e interpolando/extrapolando os demais prazos via `Linear360Interpolator(0, 0)` (item 1.4.4), mantendo `t_nodes = DU` e `t_sec_nodes = DC`.

---

## 8. Curvas de Índices de Ações, Carrego e Futuros de Ações (Capítulo 5 B3)

- **Referência:** *Manual de Curvas B3, Seções 5.1 a 5.8 (págs. 67–77)*

### 8.1 Estrutura de Preços e Vértices Anteriores ao 1º Vencimento (`BRP` e `INP`)
- **Vértice 1 ($DU = 1$):** Cotação spot de fechamento do índice no dia $t$ ($\text{IBrX-50}_t$ para `BRP`, $\text{Ibovespa}_t$ para `INP`).
- **Vértices Anteriores ao 1º Vencimento ($1 < DU_v < DU_1$):** Conforme as Seções 5.1 (pág. 67) e 5.2 (pág. 69), para todos os vértices anteriores ao primeiro vencimento futuro:
  $$\text{Índice}_v = \text{Spot}_t \times \left( \frac{\text{Fut}_1}{\text{Spot}_t} \right)^{\frac{DU_v}{DU_1}}$$
  O `B3CurveBuilder` insere uma única âncora em $DU = 2$ com valor $\text{Spot}_t \cdot (\text{Fut}_1 / \text{Spot}_t)^{2 / DU_1}$, permitindo que o `PriceInterpolator` reproduza exatamente a potência $DU_v / DU_1$ em todos os dias úteis $1 < DU_v < DU_1$ mantendo apenas os vértices esparsos dos contratos futuros (`BRI` e `IND`).
- **Interpolação e Extrapolação:** Ambas as curvas (`BRP` — Seção 5.1 e `INP` — Seção 5.2) utilizam a **Interpolação de Preços (item 1.4.5 — `PriceInterpolator(0, 0)`)** entre os vencimentos futuros e extrapolação flat após o último vencimento.

### 8.2 Curvas de Custo de Carrego / Convenience Yield (`CYI`, `CYM`, `CYX`)
Calculam a taxa de carrego anualizada implícita no preço do futuro contra a curva nominal pré:
$$TxCY_n = \left[ \frac{Spot \times \left(1 + \frac{PRE_n}{100}\right)^{\frac{DU_n}{252}}}{Futures_n} \right]^{\frac{252}{DU_n}} - 1$$

- Implementada em Cython de alta performance através da classe `ConvenienceYieldCurve`.
- Aplicável tanto aos índices oficiais (`CYI`, `CYM`, `CYX`) quanto aos contratos futuros de ações individuais (ex: `PETRP`, `VALEO`, `BBASO`).

---

## 9. Construção de Curvas a partir de Vértices e Componentes Fornecidos pelo Usuário

O `bz-pricing` foi projetado como um motor quantitativo genérico: **o usuário pode fornecer seus próprios vértices ou os insumos de cada componente** (totalmente *offline*, com `allow_download=False`), ou combinar insumos próprios com dados baixados da B3/BCB (`allow_download=True`). A comparação com os arquivos `taxas_*.csv` da B3 tem finalidade estritamente de validação metodológica.

Abaixo estão detalhados os formatos e parâmetros aceitos por `manager.get_curve(...)` para cada família de curvas quando o usuário passa os pontos dos componentes:

### 9.1 Curvas de Inflação (`DIC`, `DIM` e `DPL` — Seções 3.1, 3.2 e 3.3)

#### A. `DIC` (Cupom Sujo IPCA — Seção 3.1) e `DIM` (Cupom Sujo IGP-M — Seção 3.3)
O usuário pode construir `DIC` ou `DIM` passando a curva nominal `PRE` (`pre_curve` já instanciada ou `pre_data` como DataFrame/Series/dict de vértices `(DU, rate)` ou contratos `DI1`) combinada com **uma das duas formas de expectativas de inflação**:

1. **Via Amostra de Informantes (`informants_df` ou `informants_data`):**
   - Aceita `pd.DataFrame`, `pd.Series`, `dict` ou `list` contendo os vértices de divulgação $k_i$:
     - Colunas/chaves de prazo: `DU` (ou `day252`) ou datas `XprtnDt` / `vencimento`.
     - Coluna de valor: `rate` (taxa de cupom sujo informada $Cp_{k_i}$ em `%` ou decimal) **ou** `e_inf` (inflação acumulada esperada $E(\text{Índice})_{k_i}$ em `%`).
     - Opcionalmente, a coluna `du_start` delimita o início do degrau de divulgação $[du_{\text{start}} \dots DU]$ onde $E(\text{Índice})_{k_i}$ permanece constante (aplicando as Eq. 3.1.2/3.1.3 e 3.3.2/3.3.3).
   ```python
   # Exemplo 1: Passando vértices de PRE e vértices de informantes (DU e cupom %)
   df_pre = pd.DataFrame({'DU': [1, 21, 63, 126, 252, 504], 'rate': [13.65, 13.70, 13.80, 13.90, 14.00, 14.10]})
   df_inf = pd.DataFrame({
       'du_start': [1, 11, 32, 252, 504],
       'DU':       [1, 31, 52, 252, 504],
       'rate':     [13.65, 8.96, 8.70, 7.85, 7.60]
   })
   curve_dic = manager.get_curve('DIC', base_date, pre_data=df_pre, informants_df=df_inf)
   ```

2. **Via Projeções Mensais de Inflação (`focus_monthly` ou `focus_data`):**
   - Aceita um dicionário `{(ano, mes): taxa_mensal_decimal}` (ou DataFrame no padrão Bacen Focus com `DataReferencia` `'MM/AAAA'` e `Mediana`).
   - Opcionalmente, recebe `dpl_curve` (ou `dpl_data`) para ancorar a inflação implícita nos prazos longos acima de 18 meses.
   ```python
   # Exemplo 2: Passando vértices de PRE e projeções mensais de inflação
   focus_ipca = {(2026, 10): 0.0045, (2026, 11): 0.0038, (2026, 12): 0.0050}
   curve_dic = manager.get_curve('DIC', base_date, pre_data=df_pre, focus_monthly=focus_ipca)
   ```

#### B. `DPL` (Cupom Limpo IPCA — Seção 3.2)
Recebe a curva `PRE` (`pre_curve` ou `pre_data`), os contratos/vértices de `DAP` (`dap_df` ou `dap_data`) e o insumo do Vértice 1:
- **`pre_curve` ou `pre_data`:** Objeto `Curve` ou DataFrame/Series/dict de `PRE`.
- **`dap_df` ou `dap_data`:** DataFrame com `XprtnDt` e `AdjstdQtTax` (contratos futuros de `DAP`) **ou** DataFrame/Series/dict com `(DU, rate)`.
- **`eipca_0` ou `v1_rate`:**
  - Se `eipca_0` for informado (prévia diária de IPCA $E(\text{IPCA})_{0,t}$ em decimal, ex: `0.0022`), o Vértice 1 ($DU = 1$) é calculado pela Eq. 3.2.1 descontando `eipca_0` proporcional aos dias úteis entre os dias 15 (`du_15a_15p`) da taxa `PRE(1)`.
  - Alternativamente, `v1_rate` permite informar diretamente a taxa do Vértice 1 ($DU = 1$).
```python
df_dap = pd.DataFrame({
    'XprtnDt': [datetime.date(2026, 12, 15), datetime.date(2027, 5, 17), datetime.date(2028, 5, 15)],
    'AdjstdQtTax': [7.45, 7.60, 7.35]
})
curve_dpl = manager.get_curve('DPL', base_date, pre_data=df_pre, dap_df=df_dap, eipca_0=0.0022)
```

---

### 9.2 Curvas de Câmbio, Cupons Cambiais e Spreads (`DOL`, `DOC`, `EUC`, `PTX`, `EUR`)

#### A. `DOC` / `DCL` (Cupom Cambial Limpo — Seções 4.5 e 4.6)
Pode ser construída passando os contratos/vértices de `DDI` e os parâmetros de ajuste do dólar cupom limpo:
- **`ddi_df` ou `ddi_data`:** Contratos futuros de `DDI` (`XprtnDt` e `AdjstdQtTax`) ou vértices `(DU, DC, rate)`.
- **Ajuste dos Vértices de DDI (Eq. 4.5.3):** Informe `dol_cl` ($\text{DolCL}_{t0}$) e `ptax_tm1` ($\text{Ptax}_{t-1}$), **ou** diretamente a razão `ratio_cl_tm1` ($\text{DolCL}_{t0} / \text{Ptax}_{t-1}$).
- **Vértice 1 ($DU = 1$):** Informe `v1_rate` diretamente **ou** os insumos primários `dol_cl`, `casado_1d` (pontos do casado de 1 dia útil) e `cdi_rate` para cálculo exato pelas Eq. 4.5.1 e 4.5.2.
```python
curve_doc = manager.get_curve(
    'DOC', base_date,
    ddi_data=df_ddi,
    dol_cl=5.4180,
    ptax_tm1=5.3744,
    casado_1d=1.25,
    cdi_rate=13.65
)
```

#### B. `EUC` (Cupom de Euro — Seção 4.7) e `YCL` (Cupom de Iene — Seção 4.8)
Recebem os dois componentes da paridade cruzada:
- **`doc_curve` ou `doc_data`:** Curva `DOC` (objeto `Curve` ou DataFrame com `DU`, `DC`, `rate`).
- **`spread_curve` / `sde_curve` ou `spread_data` / `sde_data`:** Curva de spread `SDE` (ou `SYD` para `YCL`).
```python
curve_euc = manager.get_curve('EUC', base_date, doc_curve=curve_doc, sde_data=df_sde_knots)
```

#### C. `PTX`, `EUR` e `JPY` (Forwards de Moedas — Seções 4.1 a 4.3)
Recebem as curvas de juros doméstico e cupom cambial e as cotações à vista:
- **`pre_curve` ou `pre_data`:** Curva `PRE`.
- **`foreign_curve` ou `foreign_data`:** Curva `DOL` (para `PTX`), `EUC` (para `EUR`) ou `YCL` (para `JPY`).
- **`spot_rate`:** Cotação à vista do dia $t_0$ ($\text{Ptax}_{t0}$ no Vértice 1 do `PTX`; $\text{Tx}_{\text{Reais x Euro}}$ em todos os vértices do `EUR`).
- **`spot_rate_tm1` (ou `ptax_tm1`):** Cotação $\text{Ptax}_{t-1}$ do dia útil anterior (aplicada aos vértices $n \ge 2$ do `PTX` conforme Seção 4.1).
```python
curve_ptx = manager.get_curve(
    'PTX', base_date,
    pre_curve=curve_pre,
    foreign_curve=curve_dol,
    spot_rate=5.4212,
    spot_rate_tm1=5.3744
)
```

#### D. `ZMX`, `ZUS` e `ZEU` (Curvas Zero Cupom Internacionais — Seções 2.10 a 2.12)
Podem ser construídas a partir dos fatores de desconto OIS $FD_{\text{JUR}^{v,t}}$ por prazo $DC$ (`discount_factors`) conforme a fórmula oficial $\text{TxLin}_{\text{JUR}^{v,t}} = \left(\frac{1}{FD_{\text{JUR}^{v,t}}} - 1\right) \times \frac{36000}{DC_v}$, ou a partir da curva forward (`forward_curve` / `TIE`, `SFR`, `EST`):
```python
fd_sofr = {30: 0.9965, 90: 0.9895, 180: 0.9790, 360: 0.9585}
curve_zus = manager.get_curve('ZUS', base_date, discount_factors=fd_sofr)
```

---

### 9.3 Curvas de Índices de Ações e Custo de Carrego (`INP`, `BRP`, `CYI` — Capítulo 5)
- **Para `INP` e `BRP` (Seções 5.1 e 5.2):** Informe `data` (contratos futuros de `IND` ou `BRI` com `XprtnDt` e `AdjstdQt`) e `spot_price` (cotação à vista de fechamento do índice no dia $t_0$).
- **Para `CYI`, `CYM` e `CYX` (Seções 5.4, 5.6 e 5.8):** Informe `spot_price`, `futures_curve` (ou `futures_data`) e `pre_curve` (ou `pre_data`).
```python
curve_inp = manager.get_curve('INP', base_date, data=df_ind_futures, spot_price=144519.69)
curve_cyi = manager.get_curve('CYI', base_date, spot_price=144519.69, futures_curve=curve_inp, pre_curve=curve_pre)
```

---

### 9.4 Passagem Direta dos Vértices Finais de Qualquer Curva
Além da construção pelos componentes acima, qualquer curva do pacote também pode ser instanciada passando diretamente uma tabela esparsa de vértices `(DU, rate)` ou `(DU, DC, rate)` via argumento `data` (DataFrame, Series, dicionário ou lista de tuplas). O `CurveManager` aplicará automaticamente o interpolador Cython oficial, a regra de contagem de dias (`DU` vs `DC`), o regime de extrapolação e o arredondamento configurados em [`configs.py`](file:///d:/projects/bz_pricing/src/bz_pricing/curves/configs.py).
```python
# Interpola vértices customizados usando o interpolador oficial configurado para cada curva
custom_pre = manager.get_curve('PRE', data={1: 13.65, 21: 13.72, 63: 13.85, 252: 14.10})
custom_dol = manager.get_curve('DOL', data=pd.DataFrame({'DU': [1, 20, 42], 'DC': [4, 32, 66], 'rate': [-73.16, -12.10, -2.87]}))
```

