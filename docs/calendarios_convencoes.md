# Calendários, Convenções de Contagem de Dias e Regimes de Extrapolação

Este documento especifica as convenções de contagem de prazos, as regras de feriados e os regimes de extrapolação implementados nos módulos [`bz_pricing.core.calendars`](../src/bz_pricing/core/calendars.py) e [`bz_pricing.core.interpolators`](../src/bz_pricing/core/interpolators.pyx).

---

## 1. Convenções de Contagem de Dias (`DayCountConvention`)

As estruturas de dados em Cython suportam múltiplos regimes de contagem de prazos e anualização, definidos no cabeçalho [`conventions.pxd`](../src/bz_pricing/core/conventions.pxd):

| Convenção | Expressão da Fração de Ano ($\tau$) | Aplicações no Pacote |
| :--- | :---: | :--- |
| **`BUS/252`** | $\tau = \frac{\text{DU}}{252}$ | Curvas prefixadas domésticas (`PRE`, `SELIC_PRE`, `LTN`), curvas de TBF e TR (`TFP`, `TP`, `TR`), inflação (`DIC`, `DIM`, `DPL`), índices de ações (`INP`, `BRP`, `CYI`) e títulos públicos federais. |
| **`ACT/360`** | $\tau = \frac{\text{DC}}{360}$ | Cupons cambiais (`DOL`, `DOC`, `DCL`, `EUC`, `YCL`), futuros e *swaps* de taxas *overnight* internacionais (`SFR`, `EST`, `TIE`, `ZUS`, `ZEU`, `ZMX`) e spreads de moedas (`SDE`, `SGP`, `SMX`, `SYD`). |
| **`ACT/365`** | $\tau = \frac{\text{DC}}{365}$ | Instrumentos do mercado monetário britânico e derivativos internacionais específicos. |
| **`30/360`** | $\tau = \frac{360 \Delta Y + 30 \Delta M + \Delta D}{360}$ | Títulos corporativos internacionais (*Eurobonds*) e obrigações de taxa fixa *offshore*. |

---

## 2. Calendários Embutidos, Modo Offline e Regras de Rolagem (`bizdays`)

A função [`get_calendar`](../src/bz_pricing/core/calendars.py) opera com `allow_download=False` por padrão e disponibiliza calendários embutidos em memória sem dependência de rede ou de arquivos locais:

- `'B3'`, `'ANBIMA'`, `'BR'`, `'BRAZIL'`: feriados nacionais brasileiros consolidados pela ANBIMA (2001–2099, incluindo a Lei Federal nº 14.759/2023 referente a 20 de novembro); caso exista `dados/b3/feriados_nacionais.xls` ou `allow_download=True`, o carregador lê a planilha local ou externa apenas quando solicitado.
- `'US'`, `'USA'`, `'SIFMA'`, `'NYSE'`, `'FED'`: calendário do mercado norte-americano (SIFMA / títulos do Tesouro dos EUA).
- `'TARGET'`, `'EUR'`, `'EU'`, `'ECB'`: calendário do sistema TARGET2 na Zona do Euro.
- `'UK'`, `'GBP'`, `'SONIA'`, `'LCH'`: calendário bancário de Londres.
- `'ACTUAL'`, `'DC'`, `'CALENDAR'`: dias corridos sem exclusão de feriados nem fins de semana.

Quando o parâmetro `holidays` é informado, a função constrói uma instância dedicada com os feriados informados (ou combinados ao calendário base, exceto para `'CUSTOM'`, `'USER'` e `'NONE'`), preservando isolado o cache global das praças padrão.

As regras de ajuste de vencimento em dias não úteis abrangem:

- `following`: desloca a data para o primeiro dia útil subsequente.
- `preceding`: recua a data para o dia útil imediatamente anterior.
- `modified_following`: desloca para o primeiro dia útil subsequente, exceto quando o ajuste implica mudança de mês civil, hipótese em que recua para o dia útil anterior.

---

## 3. Regimes de Extrapolação nos Interpoladores Cython

Todas as classes derivadas de [`Interpolator`](../src/bz_pricing/core/interpolators.pyx) recebem os parâmetros inteiros `extrap_left` (para prazos $t < t_1$) e `extrap_right` (para prazos $t > t_n$):

### 3.1 Extrapolação Constante (`EXTRAP_FLAT = 0`)

Mantém inalterada a taxa ou o preço do vértice extremo adjacente:

$$y(t) = y_1 \quad (\forall \, t < t_1), \qquad y(t) = y_n \quad (\forall \, t > t_n)$$

Esse regime é aplicado à esquerda ($t < t_1$) em todas as curvas do pacote e à direita ($t > t_n$) nas curvas internacionais (`SFR`, `EST`, `TIE`, `ZUS`, `ZEU`, `ZMX`, `SDE`, `SGP`, `SYD`), nas curvas de preços de índices e ações (`INP`, `BRP`, `SML`, `MBR`, `VIX`) e nas curvas de custo de carrego (`CYI`, `CYM`, `CYX`). Nos interpoladores sobre fatores de desconto em capitalização `simple` ou `compounded` (`LogLinearDiscountInterpolator`, `LogCubicDiscountInterpolator` e `MonotoneConvexInterpolator`), a avaliação no intervalo inicial $(0, t_1]$ preserva a convenção exata informada no construtor para que $\text{DF}(t_1)$ coincida sem descontinuidade.

### 3.2 Extrapolação por Taxa a Termo Constante (`EXTRAP_FLAT_FORWARD = 1`)

Preserva a taxa a termo (ou a inclinação exponencial) estimada no último segmento $[\text{DU}_{n-1}, \text{DU}_n]$ para qualquer horizonte $t > \text{DU}_n$:

$$y(t) = \left[ \left(1 + y_{n-1}\right)^{\frac{\text{DU}_{n-1}}{252}} \times \left( \frac{\left(1 + y_n\right)^{\frac{\text{DU}_n}{252}}}{\left(1 + y_{n-1}\right)^{\frac{\text{DU}_{n-1}}{252}}} \right)^{\frac{t - \text{DU}_{n-1}}{\text{DU}_n - \text{DU}_{n-1}}} \right]^{\frac{252}{t}} - 1$$

Esse regime é utilizado à direita ($t > t_n$) nas estruturas prefixadas (`PRE`, `SELIC_PRE`, `LTN`), nas curvas de inflação (`DIC`, `DIM`, `DPL`), nas curvas de TBF e TR (`TFP`, `TP`, `TR`) e nos cupons cambiais (`DOL`, `DOC`, `DCL`). Nos interpoladores baseados em fatores de desconto em capitalização `simple` ou `compounded`, a extrapolação à esquerda com `EXTRAP_FLAT_FORWARD` mantém a taxa instantânea inicial constante em $(0, t_1]$ antes de reconverter o desconto para o regime nominal configurado.
