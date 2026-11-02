from bz_pricing.core.interpolators import (
    FlatForward252Interpolator,
    Exponential252Interpolator,
    FlatForward252LinearInterpolator,
    FlatForward252LogInterpolator,
    Linear360Interpolator,
    Interpolation360Interpolator
)

# Mapeamento oficial de configurações de construção de Curvas
# Define qual interpolador, extrapolação e colunas usar para cada produto
CURVE_CONFIGS = {
    "DI1": {
        "description": "Curva de Juros DI (FlatForward 252)",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,  # EXTRAP_FLAT
        "extrap_right": 1, # EXTRAP_FLAT_FORWARD
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": False
    },
    "PRE": {
        "description": "Curva de Juros Pré-Fixada (FlatForward 252)",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,  # EXTRAP_FLAT
        "extrap_right": 1, # EXTRAP_FLAT_FORWARD (Conforme manual da B3)
        "round_decimals": 5, # Arredonda em 3 casas decimais a taxa percentual
        "liquidity_filter": True, # Exclui contratos DI1 sem liquidez
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": False,
        "asset": "DI1",
        "v1_source": "CDI"
    },
    "SELIC_PRE": {
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,  # EXTRAP_FLAT
        "extrap_right": 1, # EXTRAP_FLAT_FORWARD
        "round_decimals": 5,
        "liquidity_filter": False,
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": False,
        "asset": "OC1",
        "v1_source": "SELIC"
    },
    "LTN": {
        "description": "Curva de Títulos Públicos Pré-fixados (Letra do Tesouro Nacional)",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,  # EXTRAP_FLAT
        "extrap_right": 1, # EXTRAP_FLAT_FORWARD
        "round_decimals": 5,
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": False,
        "asset": "LTN",
        "v1_source": "SELIC"
    },
    "TJLP": {
        "type": "flat",
        "source": "BCB",
        "series_id": 256
    },
    "TBF_PRE": {
        "type": "derived",
        "base_curve": "PRE",
        "tbf_source": "BCB",
        "tbf_series_id": 253,
        "round_decimals": 4
    },
    "TR_PRE": {
        "type": "derived",
        "base_curve": "TBF_PRE",
        "round_decimals": 4
    },
    "TBF_LTN": {
        "type": "derived",
        "base_curve": "LTN",
        "round_decimals": 4
    },
    "TR_LTN": {
        "type": "derived",
        "base_curve": "TBF_LTN",
        "round_decimals": 4
    },
    "DAP": {
        "description": "Curva de Juros Cupom de IPCA (FlatForward 252)",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": False,
        "asset": "DAP"
    },
    "DPL": {
        "description": "Curva DI x IPCA Cupom Limpo (Seção 3.2 Manual B3)",
        "type": "derived",
        "nominal_curve": "PRE",
        "real_curve": "DAP",
        "round_decimals": 4,
        "round_mode": "round"
    },
    "CUPOM_LIMPO": {
        "description": "Curva de Cupom Cambial Limpo (FlatForward 252 Linear)",
        "interpolator_class": FlatForward252LinearInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": True
    },
    "TIE": {
        "description": "Curva Forward F-TIIE do México (Linear 360)",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0, # Flat para extrapolação posterior ao vencimento do contrato
        "round_decimals": 5,
        "round_mode": "round",
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": True,
        "asset": "TIE",
        "v1_source": "BANXICO"
    },
    "SFR": {
        "description": "Curva Forward SOFR (Linear 360)",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0, # Extrapolação flat
        "round_decimals": 5,
        "round_mode": "round",
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": True,
        "asset": "SFR",
        "v1_source": "FED"
    },
    "EST": {
        "description": "Curva Forward ESTR (Linear 360)",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 5,
        "round_mode": "round",
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": True,
        "asset": "EST",
        "v1_source": "ECB"
    },
    "ZMX": {
        "description": "Curva Zero Coupon F-TIIE (Linear 360 - Seção 2.10 Manual B3)",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 5,
        "round_mode": "round",
        "use_dc": True
    },
    "ZUS": {
        "description": "Curva Zero Coupon SOFR (Linear 360 - Seção 2.11 Manual B3)",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 5,
        "round_mode": "round",
        "use_dc": True
    },
    "ZEU": {
        "description": "Curva Zero Coupon ESTR (Linear 360 - Seção 2.12 Manual B3)",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 5,
        "round_mode": "round",
        "use_dc": True
    },
    "DIC": {
        "description": "Curva DI x IPCA Cupom Sujo (Seção 3.1 Manual B3)",
        "type": "derived",
        "nominal_curve": "PRE",
        "real_curve": "DAP",
        "indicator": "IPCA",
        "round_decimals": 4,
        "round_mode": "round"
    },
    "DIM": {
        "description": "Curva DI x IGP-M Cupom Sujo (Seção 3.3 Manual B3)",
        "type": "derived",
        "nominal_curve": "PRE",
        "real_curve": "DGI",
        "indicator": "IGP-M",
        "round_decimals": 4,
        "round_mode": "round"
    },
    "DGI": {
        "description": "Curva de Juro Real (DI x IGP-M) - Base para DIM",
        "interpolator_class": Exponential252Interpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": -1,
        "use_dc": False,
        "asset": "DGI"
    },
    "PTX": {
        "description": "Curva Forward Reais x Dolar (Construção Simples / Flat Forward)",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 7,
        "round_mode": "trunc",
        "use_dc": False
    },
    "EUR": {
        "description": "Curva Forward Reais x Euro (Construção Simples / Flat Forward)",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 7,
        "round_mode": "trunc",
        "use_dc": False
    },
    "JPY": {
        "description": "Curva Forward Reais x Iene (Construção Simples / Flat Forward)",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 7,
        "round_mode": "trunc",
        "use_dc": False
    },
    "DOL": {
        "description": "Curva Cupom Sujo de Dólares",
        "interpolator_class": FlatForward252LinearInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 5,
        "round_mode": "trunc",
        "use_dc": True
    },
    "DOC": {
        "description": "Curva Cupom Sujo de Dólares (Alternativa)",
        "interpolator_class": FlatForward252LinearInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 5,
        "round_mode": "trunc",
        "use_dc": True
    },
    "DCL": {
        "description": "Curva Cupom Limpo de Dólares (Alternativa)",
        "interpolator_class": FlatForward252LinearInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 5,
        "round_mode": "trunc",
        "use_dc": True
    },
    "EUC": {
        "description": "Curva Cupom Sujo de Euros",
        "interpolator_class": FlatForward252LinearInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 5,
        "round_mode": "trunc",
        "use_dc": True
    }
}
