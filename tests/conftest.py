import os
from pathlib import Path
import pytest


def is_offline_or_ci() -> bool:
    """Retorna True quando a execução ocorre em CI ou com modo offline habilitado."""
    ci = os.environ.get("CI", "").strip().lower() in ("1", "true", "yes")
    offline = os.environ.get("BZ_PRICING_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    return ci or offline


def pytest_runtest_setup(item):
    """Pula testes online ou dependentes de cache local quando em CI ou sem os arquivos."""
    offline = is_offline_or_ci()
    if item.get_closest_marker("online") is not None and offline:
        pytest.skip("Teste online (BCB/ANBIMA/B3) ignorado em CI/modo offline.")

    for marker in item.iter_markers(name="market_data"):
        if offline:
            pytest.skip("Teste empírico com dados locais ignorado em CI/modo offline.")
        for rel_path in marker.args:
            if not Path(rel_path).exists():
                pytest.skip(f"Arquivo/diretório local de mercado ausente: {rel_path}")
