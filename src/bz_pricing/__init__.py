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
from bz_pricing.instruments.tpf import LTN, NTNF

__all__ = [
    "B3Scraper",
    "B3DataLoader",
    "B3CurveBuilder",
    "CurveManager",
    "DataUnavailableError",
    "LTN",
    "NTNF"
]
