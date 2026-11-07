# bz-pricing: Precificação e Curvas de Juros Onshore e Offshore

`bz-pricing` é um pacote quantitativo de alta performance desenvolvido em Python e Cython para a construção de curvas de juros, interpolações financeiras, precificação de títulos de renda fixa (NTN-B, LTN, NTN-F, LFT, Debêntures) e derivativos negociados na B3 e no mercado internacional.

Toda a formulação matemática e as regras de negócio do pacote são rigorosamente respaldadas nas documentações técnicas oficiais da **B3** (Bolsa de Valores do Brasil), do **Banco Central do Brasil (BCB)** e da **ANBIMA**.

---

## 📚 Índice da Documentação

1. **[Fundamentos Matemáticos e Manuais B3/BCB](metodologia_curvas.md)**
   - Curvas de Juros Pré-Fixadas (DI x PRE - Seção 2.1)
   - Selic x PRE e SGS Série 11 (Seção 2.2)
   - TBF x PRE e Redutor da TR (Resolução CMN nº 5.124 / Seções 2.4 e 2.5)
   - Curvas Internacionais: SOFR Futuros (`SFR` - Seção 2.8) vs Swaps Zero Cupom (`ZUS`/`LIB` - Seção 2.11)
   - Curvas de Inflação Implícita e Cupom Sujo: DIC (IPCA - Seção 3.1) e DIM (IGP-M - Seção 3.3)
   - Cupom Limpo de IPCA: DAP / DPL (Seção 3.2)
2. **[Guia de Uso das Funções e APIs](guia_uso_api.md)**
   - Orquestrador de Curvas (`CurveManager`)
   - Construtor Especializado (`B3CurveBuilder`)
   - Extração e Download de Dados (`B3DataLoader` e `B3Scraper`)
   - Interpolação e Métodos em Cython (`SimpleCurve`, `FlatForward252Interpolator`, `Linear360Interpolator`)
   - Precificação de Títulos Públicos e Privados (`FixedRateBond`, `InflationBond`, `FloatingBond`)
3. **[Convenções de Contagem de Dias e Calendários](calendarios_convencoes.md)**
   - Base 252 (Dias Úteis - Feriados Nacionais B3/ANBIMA)
   - Base 360 / ACT (Dias Corridos - Mercados Offshore e Cupom Cambial)
   - Tratamento de Extrapolações (Flat, Flat Forward e Exponencial)

---

## ⚡ Instalação e Compilação

Para compilar as extensões Cython em ambiente de desenvolvimento:

```bash
# Ativar o ambiente virtual
.\.venv\Scripts\Activate.ps1

# Compilar as extensões Cython no local
pip install -e . --no-build-isolation
```

Para rodar os testes automatizados:

```bash
pytest -v
```

---

## 🚀 Exemplo Rápido de Uso

```python
import datetime
from bz_pricing.curves.manager import CurveManager

# Inicializa o orquestrador
manager = CurveManager()
base_date = datetime.date(2026, 9, 4)

# 1. Curva DI x PRE (B3)
curva_pre = manager.get_curve("PRE", base_date)
taxa_1_ano = curva_pre.get_rate(252.0)
print(f"Taxa Pré 252 DU: {taxa_1_ano * 100:.2f}% a.a.")

# 2. Curvas de Inflação: DIC (Sujo) e DPL (Limpo / DAP)
curva_dic = manager.get_curve("DIC", base_date)
curva_dpl = manager.get_curve("DPL", base_date)
print(f"DIC 6 DU: {curva_dic.get_rate(6.0) * 100:.2f}% a.a.")
print(f"DPL 252 DU: {curva_dpl.get_rate(252.0) * 100:.2f}% a.a.")

# 3. Curva de SOFR Swaps Offshore (ZUS / LIB - ACT/360)
curva_zus = manager.get_curve("ZUS", base_date)
print(f"SOFR Swaps 360 DC: {curva_zus.get_rate(360.0) * 100:.2f}% a.a.")
```

