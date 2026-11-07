from bz_pricing.core.interpolators import (
    FlatForward252Interpolator,
    Exponential252Interpolator,
    FlatForward252LinearInterpolator,
    FlatForward252LogInterpolator,
    Linear360Interpolator,
    Interpolation360Interpolator,
    PriceInterpolator
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
        "round_decimals": 6, # 4 casas decimais na taxa percentual (Seção 2.1 Manual B3)
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
        "round_decimals": 6,
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
        "source_type": "taxas_referenciais",
        "produto": "DPL",
        "interpolator_class": FlatForward252Interpolator,
        "round_decimals": 6,
        "round_mode": "round",
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
        "index_by_dc": True,
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
        "index_by_dc": True,
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
        "index_by_dc": True,
        "asset": "EST",
        "v1_source": "ECB"
    },
    "ZMX": {
        "description": "Curva Zero Coupon F-TIIE (Linear 360 - Seção 2.10 Manual B3)",
        "type": "derived_zero_coupon",
        "forward_curve": "TIE",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 5,
        "round_mode": "round",
        "use_dc": True,
        "index_by_dc": True
    },
    "ZUS": {
        "description": "Curva Zero Coupon SOFR (Linear 360 - Seção 2.11 Manual B3)",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 5,
        "round_mode": "round",
        "use_dc": True,
        "index_by_dc": True,
        "source_type": "taxas_referenciais",
        "produto": "LIB"
    },
    "LIB": {
        "description": "Curva Zero Coupon SOFR / Libor legada (Linear 360 - Seção 2.11 Manual B3)",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 5,
        "round_mode": "round",
        "use_dc": True,
        "index_by_dc": True,
        "source_type": "taxas_referenciais",
        "produto": "LIB"
    },
    "ZEU": {
        "description": "Curva Zero Coupon ESTR (Linear 360 - Seção 2.12 Manual B3)",
        "type": "derived_zero_coupon",
        "forward_curve": "EST",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 5,
        "round_mode": "round",
        "use_dc": True,
        "index_by_dc": True
    },
    "DIC": {
        "description": "Curva DI x IPCA Cupom Sujo (Seção 3.1 Manual B3)",
        "type": "derived",
        "nominal_curve": "PRE",
        "indicator": "IPCA",
        "source_type": "taxas_referenciais",
        "produto": "DIC",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 6,
        "round_mode": "round",
        "use_dc": False
    },
    "DIM": {
        "description": "Curva DI x IGP-M Cupom Sujo (Seção 3.3 Manual B3)",
        "type": "derived",
        "nominal_curve": "PRE",
        "indicator": "IGP-M",
        "source_type": "taxas_referenciais",
        "produto": "DIM",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 6,
        "round_mode": "round",
        "use_dc": False
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
        "description": "Forward de Reais x Dólar (Seção 4.1 Manual B3)",
        "type": "currency_forward",
        "currency": "USD",
        "pre_curve": "PRE",
        "foreign_curve": "DOL",
        "source_type": "taxas_referenciais",
        "produto": "PTX",
        "interpolator_class": PriceInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 7,
        "round_mode": "trunc",
        "use_dc": True,
        "divide_by_100": False
    },
    "EUR": {
        "description": "Forward de Reais x Euro (Seção 4.2 Manual B3)",
        "type": "currency_forward",
        "currency": "EUR",
        "pre_curve": "PRE",
        "foreign_curve": "EUC",
        "source_type": "taxas_referenciais",
        "produto": "EUR",
        "interpolator_class": PriceInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 7,
        "round_mode": "trunc",
        "use_dc": True,
        "divide_by_100": False
    },
    "JPY": {
        "description": "Forward de Reais x JPY (Seção 4.3 Manual B3)",
        "type": "currency_forward",
        "source_type": "taxas_referenciais",
        "produto": "JPY",
        "currency": "JPY",
        "pre_curve": "PRE",
        "foreign_curve": "YCL",
        "interpolator_class": PriceInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 7,
        "round_mode": "trunc",
        "use_dc": True,
        "divide_by_100": False
    },
    "DOL": {
        "description": "Curva de Cupom de Dólar Sujo (Seção 4.4 Manual B3)",
        "interpolator_class": FlatForward252LinearInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 7,
        "round_mode": "round",
        "use_dc": True,
        "asset": "DDI",
        "rate_column": "AdjstdQtTax"
    },
    "DOC": {
        "description": "Curva de Cupom de Dólar Limpo (Seção 4.5 Manual B3)",
        "source_type": "taxas_referenciais",
        "produto": "DOC",
        "asset": "DDI",
        "interpolator_class": FlatForward252LinearInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 6,
        "round_mode": "round",
        "use_dc": True
    },
    "DCL": {
        "description": "Curva de Cupom de Dólar Limpo (Seção 4.6 Manual B3)",
        "source_type": "taxas_referenciais",
        "produto": "DOC",
        "asset": "DDI",
        "interpolator_class": FlatForward252LinearInterpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 6,
        "round_mode": "round",
        "use_dc": True
    },
    "EUC": {
        "description": "Curva de Cupom de Euro (Seção 4.7 Manual B3)",
        "type": "derived",
        "doc_curve": "DOC",
        "spread_curve": "SDE",
        "round_decimals": 6,
        "round_mode": "round"
    },
    "YCL": {
        "description": "Curva de Cupom de JPY (Seção 4.8 Manual B3)",
        "type": "derived",
        "doc_curve": "DOC",
        "spread_curve": "SYD",
        "round_decimals": 9,
        "round_mode": "round"
    },
    "SDE": {
        "description": "Curva de Spread de Euro (Seção 4.9 Manual B3: TEU - TUS)",
        "source_type": "taxas_referenciais",
        "produto": "SDE",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 6,
        "round_mode": "round",
        "use_dc": True
    },
    "SGP": {
        "description": "Curva de Spread de GBP (Seção 4.10 Manual B3: TGP - TUS)",
        "type": "spread",
        "curve_a": "ZEU",
        "curve_b": "ZUS",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "use_dc": True,
        "index_by_dc": True,
        "round_decimals": 4,
        "round_mode": "round"
    },
    "SMX": {
        "description": "Curva de Spread de MXN (Seção 4.11 Manual B3: TMX - TUS)",
        "type": "spread",
        "curve_a": "ZMX",
        "curve_b": "ZUS",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "use_dc": True,
        "index_by_dc": True,
        "round_decimals": 4,
        "round_mode": "round"
    },
    "SYD": {
        "description": "Curva de Spread de JPY (Seção 4.12 Manual B3: TJP - TUS)",
        "type": "spread",
        "curve_a": "ZEU",
        "curve_b": "ZUS",
        "interpolator_class": Linear360Interpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "use_dc": True,
        "index_by_dc": True,
        "round_decimals": 4,
        "round_mode": "round"
    },
    "BRP": {
        "description": "Curva IBrX50 x PRE (Seção 5.1 Manual B3)",
        "type": "equity_index",
        "asset": "BRI",
        "source_type": "taxas_referenciais",
        "produto": "BRP",
        "interpolator_class": PriceInterpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 2,
        "round_mode": "round",
        "use_dc": False,
        "divide_by_100": False
    },
    "INP": {
        "description": "Curva Índice Bovespa (Seção 5.2 Manual B3)",
        "type": "equity_index",
        "asset": "IND",
        "source_type": "taxas_referenciais",
        "produto": "INP",
        "interpolator_class": PriceInterpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 2,
        "round_mode": "round",
        "use_dc": False,
        "divide_by_100": False
    },
    "SML": {
        "description": "Curva Índice Small Caps (Seção 5.3 Manual B3)",
        "type": "equity_index",
        "asset": "SML",
        "interpolator_class": PriceInterpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 2,
        "round_mode": "round"
    },
    "CYI": {
        "description": "Curva de Custo de Carrego dos Futuros de Ibovespa (Seção 5.4 Manual B3)",
        "type": "convenience_yield",
        "index_curve": "INP",
        "pre_curve": "PRE",
        "round_decimals": 6,
        "round_mode": "round"
    },
    "MBR": {
        "description": "Curva Índice Bovespa B3 BR+ (Seção 5.5 Manual B3)",
        "type": "equity_index",
        "asset": "MBR",
        "interpolator_class": PriceInterpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 2,
        "round_mode": "round"
    },
    "CYM": {
        "description": "Curva de Custo de Carrego dos Futuros Bovespa B3 BR+ (Seção 5.6 Manual B3)",
        "type": "convenience_yield",
        "index_curve": "MBR",
        "pre_curve": "PRE",
        "round_decimals": 6,
        "round_mode": "round"
    },
    "VIX": {
        "description": "Curva Índice S&P/B3 Bovespa VIX (Seção 5.7 Manual B3)",
        "type": "equity_index",
        "asset": "VIX",
        "interpolator_class": PriceInterpolator,
        "extrap_left": 0,
        "extrap_right": 0,
        "round_decimals": 2,
        "round_mode": "round"
    },
    "CYX": {
        "description": "Curva de Custo de Carrego dos Futuros de VIX (Seção 5.8 Manual B3)",
        "type": "convenience_yield",
        "index_curve": "VIX",
        "pre_curve": "PRE",
        "round_decimals": 6,
        "round_mode": "round"
    },
    # Aliases oficiais dos produtos da B3
    "SLP": {
        "description": "Curva Selic x Pré (Contratos OC1 - Seção 2.2 Manual B3)",
        "interpolator_class": FlatForward252Interpolator,
        "extrap_left": 0,
        "extrap_right": 1,
        "round_decimals": 5,
        "liquidity_filter": False,
        "rate_column": "AdjstdQtTax",
        "exp_date": "XprtnDt",
        "use_dc": False,
        "asset": "OC1",
        "v1_source": "SELIC"
    },
    "TFP": {
        "description": "Curva TBF x Pré (Seção 2.4 Manual B3)",
        "type": "derived",
        "base_curve": "PRE",
        "tbf_source": "BCB",
        "tbf_series_id": 253,
        "round_decimals": 4
    },
    "TP": {
        "description": "Curva TR x Pré (Seção 2.5 Manual B3)",
        "type": "derived",
        "base_curve": "TFP",
        "round_decimals": 4
    },
    "TR": {
        "description": "Curva DI x TR (Seção 2.6 Manual B3)",
        "type": "derived",
        "pre_curve": "PRE",
        "tp_curve": "TP",
        "round_decimals": 5,
        "round_mode": "round"
    }
}
