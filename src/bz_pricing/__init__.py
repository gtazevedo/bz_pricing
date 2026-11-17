"""
bz-pricing
Biblioteca (Cython/Python) para precificação Onshore (B3/ANBIMA) e Offshore.
"""

__version__ = "0.0.1"

# Exportando as principais ferramentas do pacote
from bz_pricing.data.scrapers import B3Scraper
from bz_pricing.data.loaders import B3DataLoader
from bz_pricing.curves.builder import B3CurveBuilder
from bz_pricing.curves.exceptions import DataUnavailableError
from bz_pricing.curves.manager import CurveManager
from bz_pricing.instruments.tpf import LTN, NTNF, LFT, NTNB, NTNBPrincipal, NTNC
from bz_pricing.core.vna_engine import (
    VNAEngine,
    calculate_vna_from_vne,
    calculate_ntnb_vna,
    calculate_lft_vna,
    calculate_ntnc_vna,
)

__all__ = [
    "B3Scraper",
    "B3DataLoader",
    "B3CurveBuilder",
    "CurveManager",
    "DataUnavailableError",
    "LTN",
    "NTNF",
    "LFT",
    "NTNB",
    "NTNBPrincipal",
    "NTNC",
    "VNAEngine",
    "calculate_vna_from_vne",
    "calculate_ntnb_vna",
    "calculate_lft_vna",
    "calculate_ntnc_vna",
]
