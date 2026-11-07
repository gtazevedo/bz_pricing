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

## 2. Calendário de Dias Úteis e Regras de Rolagem (`bizdays`)

A função [`get_calendar`](../src/bz_pricing/core/calendars.py) instancia o calendário oficial de negociação e liquidação com base no pacote `bizdays`:

- **Dias Não Úteis:** Sábados, domingos e feriados nacionais brasileiros (incluindo as alterações introduzidas pela Lei Federal nº 14.759/2023 referente ao dia 20 de novembro).
- **Ajustes de Vencimento em Dia Não Útil:**
  - `following`: Desloca a data para o primeiro dia útil subsequente.
  - `preceding`: Recua a data para o dia útil imediatamente anterior.
  - `modified_following`: Desloca para o primeiro dia útil subsequente, exceto quando o ajuste implica mudança de mês civil, hipótese em que recua para o dia útil anterior.

---

## 3. Regimes de Extrapolação nos Interpoladores Cython

Todas as classes derivadas de [`Interpolator`](../src/bz_pricing/core/interpolators.pyx) recebem os parâmetros inteiros `extrap_left` (para prazos $t < t_1$) e `extrap_right` (para prazos $t > t_n$):

### 3.1 Extrapolação Constante (`EXTRAP_FLAT = 0`)

Mantém inalterada a taxa ou o preço do vértice extremo adjacente:

$$y(t) = y_1 \quad (\forall \, t < t_1), \qquad y(t) = y_n \quad (\forall \, t > t_n)$$

Esse regime é aplicado à esquerda ($t < t_1$) em todas as curvas do pacote e à direita ($t > t_n$) nas curvas internacionais (`SFR`, `EST`, `TIE`, `ZUS`, `ZEU`, `ZMX`, `SDE`), nas curvas de preços de índices e ações (`INP`, `BRP`, `SML`, `MBR`, `VIX`) e nas curvas de custo de carrego (`CYI`, `CYM`, `CYX`).

### 3.2 Extrapolação por Taxa a Termo Constante (`EXTRAP_FLAT_FORWARD = 1`)

Preserva a taxa a termo (ou a inclinação exponencial) estimada no último segmento $[\text{DU}_{n-1}, \text{DU}_n]$ para qualquer horizonte $t > \text{DU}_n$:

$$y(t) = \left[ \left(1 + y_{n-1}\right)^{\frac{\text{DU}_{n-1}}{252}} \times \left( \frac{\left(1 + y_n\right)^{\frac{\text{DU}_n}{252}}}{\left(1 + y_{n-1}\right)^{\frac{\text{DU}_{n-1}}{252}}} \right)^{\frac{t - \text{DU}_{n-1}}{\text{DU}_n - \text{DU}_{n-1}}} \right]^{\frac{252}{t}} - 1$$

Esse regime é utilizado à direita ($t > t_n$) nas estruturas prefixadas (`PRE`, `SELIC_PRE`, `LTN`), nas curvas de inflação (`DIC`, `DIM`, `DPL`), nas curvas de TBF e TR (`TFP`, `TP`, `TR`) e nos cupons cambiais (`DOL`, `DOC`, `DCL`).
