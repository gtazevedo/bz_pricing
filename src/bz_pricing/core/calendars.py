import datetime
import os
from typing import Dict, Optional, Sequence

import bizdays
import pandas as pd

from bz_pricing.core._anbima_holidays import (
    ANBIMA_HOLIDAYS_ISO,
    build_target_holidays,
    build_uk_holidays,
    build_us_holidays,
)

_URL_ANBIMA_FERIADOS = "https://www.anbima.com.br/feriados/arqs/feriados_nacionais.xls"

# Cache segregado por identificador de calendário para evitar colisões globais.
_CALENDARS_CACHE: Dict[str, bizdays.Calendar] = {}


def _load_embedded_anbima_holidays():
    return [datetime.date.fromisoformat(s) for s in ANBIMA_HOLIDAYS_ISO]


def get_calendar(
    calendar_name: str = "B3",
    data_dir: str = "dados/b3",
    holidays: Optional[Sequence] = None,
    allow_download: bool = False,
) -> bizdays.Calendar:
    """
    Retorna o calendário `bizdays` solicitado sem acessar rede por padrão.

    Calendários embutidos em memória:
    - 'B3', 'ANBIMA', 'BR', 'BRAZIL': feriados nacionais ANBIMA (2001–2099);
    - 'US', 'USA', 'SIFMA', 'NYSE', 'FED': mercado financeiro norte-americano;
    - 'TARGET', 'EUR', 'EU', 'ECB': sistema TARGET2 da Zona do Euro;
    - 'UK', 'GBP', 'SONIA', 'LCH': praça financeira de Londres;
    - 'ACTUAL', 'DC', 'CALENDAR': dias corridos sem exclusão de feriados.

    Se `holidays` for informado, constrói um calendário dedicado com essa lista
    sem sobrescrever o cache dos calendários padrão.
    """
    cal_key = str(calendar_name).strip().upper()
    if holidays is not None:
        hol_dates = [pd.to_datetime(d).date() for d in holidays]
        if cal_key in ("CUSTOM", "USER", "NONE"):
            return bizdays.Calendar(
                holidays=hol_dates,
                weekdays=["Saturday", "Sunday"],
                startdate=datetime.date(1970, 1, 1),
                enddate=datetime.date(2100, 12, 31),
                name=str(calendar_name),
            )
        base_cal = get_calendar(
            calendar_name=calendar_name,
            data_dir=data_dir,
            holidays=None,
            allow_download=allow_download,
        )
        merged = sorted(set(base_cal.holidays) | set(hol_dates))
        return bizdays.Calendar(
            holidays=merged,
            weekdays=list(base_cal.weekdays),
            startdate=datetime.date(1970, 1, 1),
            enddate=datetime.date(2100, 12, 31),
            name=f"{calendar_name}_CUSTOM",
        )
    if cal_key in ("B3", "ANBIMA", "BR", "BRAZIL"):
        cache_key = f"B3::{os.path.abspath(data_dir)}"
        if cache_key in _CALENDARS_CACHE:
            return _CALENDARS_CACHE[cache_key]

        xls_path = os.path.join(data_dir, "feriados_nacionais.xls")
        holidays_dates = None
        if os.path.exists(xls_path):
            try:
                df = pd.read_excel(xls_path)
                df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
                df = df.dropna(subset=["Data"])
                holidays_dates = [d.date() for d in df["Data"]]
            except Exception:
                holidays_dates = None

        if holidays_dates is None and allow_download:
            try:
                import requests
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                os.makedirs(data_dir, exist_ok=True)
                response = requests.get(_URL_ANBIMA_FERIADOS, verify=False, timeout=5)
                if response.status_code == 200:
                    with open(xls_path, "wb") as f:
                        f.write(response.content)
                    df = pd.read_excel(xls_path)
                    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
                    df = df.dropna(subset=["Data"])
                    holidays_dates = [d.date() for d in df["Data"]]
            except Exception:
                holidays_dates = None

        if holidays_dates is None:
            holidays_dates = _load_embedded_anbima_holidays()

        cal = bizdays.Calendar(holidays=holidays_dates, weekdays=["Saturday", "Sunday"], name="B3")
        _CALENDARS_CACHE[cache_key] = cal
        return cal

    if cal_key in _CALENDARS_CACHE:
        return _CALENDARS_CACHE[cal_key]

    if cal_key in ("US", "USA", "SIFMA", "NYSE", "FED"):
        cal = bizdays.Calendar(holidays=build_us_holidays(), weekdays=["Saturday", "Sunday"], name="US")
    elif cal_key in ("TARGET", "EUR", "EU", "ECB"):
        cal = bizdays.Calendar(holidays=build_target_holidays(), weekdays=["Saturday", "Sunday"], name="TARGET")
    elif cal_key in ("UK", "GBP", "SONIA", "LCH"):
        cal = bizdays.Calendar(holidays=build_uk_holidays(), weekdays=["Saturday", "Sunday"], name="UK")
    else:
        cal = bizdays.Calendar(weekdays=[], name="Actual")

    _CALENDARS_CACHE[cal_key] = cal
    return cal
