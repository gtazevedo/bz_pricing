# bz_pricing

Biblioteca Quantitativa para precificação de Renda Fixa e Derivativos (Onshore e Offshore) baseada em Cython. Focada em performance institucional e integração simplificada.

## 🚀 Funcionalidades

- **Data Pipeline Integrado**: Scrapers e Loaders construídos para consumir as bases massivas da B3 (SPRD e IN) em XML/CSV diretamente da B3 ou da API JSON de Taxas Referenciais.
- **Motor Cython Ultra Rápido**: 
  - Cálculo de Curvas (Interpolações e Extrapolações Flat/FlatForward) nativas em C++.
  - Precificação determinística de Títulos Públicos (LTN e NTN-F).
  - Regras institucionais estritas da B3/ANBIMA (Truncamento e Arredondamento).
- **Curvas de Juros Suportadas**:
  - **DI x PRE**: Curva base do mercado construída com Futuros de DI1 (Exponential252).
  - **SELIC x PRE**: Curva construída com Futuros de OC1 (FlatForward252).
  - **TBF x PRE**: Construção matemática complexa *on-the-fly* via Cython baseada no Fator TBF Histórico.
  - **TJLP**: Integração direta com BCB (SGS).

## 📦 Instalação (Em breve no PyPI)
O projeto utiliza `poetry` e exige um compilador C++ para a compilação local (build_ext).

```bash
git clone https://github.com/gtazevedo/bz_pricing.git
cd bz_pricing
poetry install
poetry run python setup.py build_ext --inplace
```

## 📊 Notebooks e Análises
As análises e comparações de curvas geradas pelo motor contra as taxas oficias da B3 estão disponíveis na pasta `notebooks/`.

---
*Disclaimer: Este projeto tem propósitos de pesquisa quantitativa. Verifique as metodologias e documentações oficias da B3 para validação em produção.*
