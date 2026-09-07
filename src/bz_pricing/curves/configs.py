from bz_pricing.core.interpolators import (
    FlatForward252Interpolator,
    Exponential252Interpolator,
    FlatForward252LinearInterpolator,
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
    "TJLP": {
        "type": "flat",
        "source": "BCB",
        "series_id": 256
    },
    "TBF_PRE": {
        "type": "derived",
        "base_curve": "PRE",
        "tbf_source": "BCB",
        "tbf_series_id": 11,
        "round_decimals": 4
    },
    "TR_PRE": {
        "type": "derived",
        "base_curve": "TBF_PRE",
        "round_decimals": 4
    },
    "DAP": {
        "description": "Curva de Juros Cupom de IPCA (FlatForward 252)",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": False
    },
    "CUPOM_LIMPO": {
        "description": "Curva de Cupom Cambial Limpo (FlatForward 252 Linear)",
        "interpolator_class": FlatForward252LinearInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": True
    }
}
