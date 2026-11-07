# Calendários, Contagens de Dias e Extrapolações

Este documento formaliza as regras de convenções de prazos, contagem de dias e tratamento de caudas de curvas implementadas no `bz-pricing`.

---

## 1. Convenções de Contagem de Dias

O `bz-pricing` suporta as principais convenções de contagem do mercado financeiro nacional e internacional:

| Convenção | Descrição | Aplicação Principal no Pacote |
|:---|:---|:---|
| **BUS/252** | Dias úteis ponderados por ano de 252 dias úteis | Curvas DI, PRE, Selic, TBF, TR, Títulos Públicos (LTN, NTN-B, NTN-F, LFT) |
| **ACT/360** | Dias corridos ponderados por ano bancário de 360 dias | Cupom Cambial Onshore (DOL, DDI), SOFR, Libor, Swaps Offshore |
| **ACT/365** | Dias corridos sobre ano civil de 365 dias | Derivativos de Moedas Internacionais (GBP, etc.) |
| **30/360** | Convenção americana de meses comerciais com 30 dias | Eurobonds e títulos de renda fixa offshore |

---

## 2. Calendário de Feriados Nacionais (B3 / ANBIMA)

A classe [`get_calendar`](../src/bz_pricing/core/calendars.py) encapsula a biblioteca `bizdays` com os feriados bancários e da B3:

- **Dias Não-Úteis:** Finais de semana (sábado e domingo) e feriados nacionais brasileiros (definidos pela Lei Federal nº 10.607/2002 e alterações, incluindo o Dia Nacional de Zumbi e da Consciência Negra em 20 de novembro).
- **Regras de Ajuste de Data:**
  - `following`: Se a data cair em feriado ou fim de semana, avança para o primeiro dia útil subsequente (padrão B3).
  - `preceding`: Recua para o dia útil imediatamente anterior.
  - `modified_following`: Avança para o próximo dia útil, a menos que mude de mês, caso em que recua para o dia útil anterior.

---

## 3. Modos de Extrapolação

Os interpoladores em Cython implementados em [`src/bz_pricing/core/interpolators.pyx`](../src/bz_pricing/core/interpolators.pyx) aceitam flags para controlar o comportamento fora do intervalo dos nós conhecidos $[t_1, t_n]$:

### 3.1 Extrapolação à Esquerda ($t < t_1$)
- `extrap_left = 0` (**EXTRAP_FLAT**): Retorna a taxa do primeiro nó ($y_1$).
- `extrap_left = 1` (**EXTRAP_FLAT_FORWARD**): Projeta a taxa a termo do primeiro intervalo para trás.

### 3.2 Extrapolação à Direita ($t > t_n$)
- `extrap_right = 0` (**EXTRAP_FLAT**): Congela a taxa no último nó conhecido ($y_n$).
  - *Utilizado em:* Curva de SOFR Swaps (`ZUS`), Curva de Cupom de IGP-M (`DIM` para $DU > 8.000$).
- `extrap_right = 1` (**EXTRAP_FLAT_FORWARD**): Mantém constante a taxa a termo forward apurada no último intervalo $[t_{n-1}, t_n]$.
  - *Utilizado em:* Curva DI x PRE (`PRE`), Curva Selic x PRE (`SELIC_PRE`).

