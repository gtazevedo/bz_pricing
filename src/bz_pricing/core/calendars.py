import os
import datetime
import pandas as pd
import requests
import bizdays

# Fallback robusto nativo
_CAL_B3_NATIVE = "B3"
_URL_ANBIMA_FERIADOS = "https://www.anbima.com.br/feriados/arqs/feriados_nacionais.xls"

# Singleton
_CALENDAR_INSTANCE = None

def get_calendar(calendar_name: str = 'B3', data_dir: str = 'dados/b3') -> bizdays.Calendar:
    """
    Retorna o calendário bizdays da B3 com extensão até 2099 baseado na ANBIMA.
    Implementa um singleton para evitar recarregar e fazer download do Excel várias vezes.
    """
    global _CALENDAR_INSTANCE
    
    if _CALENDAR_INSTANCE is not None and _CALENDAR_INSTANCE.name == calendar_name:
        return _CALENDAR_INSTANCE

    if calendar_name != 'B3':
        _CALENDAR_INSTANCE = bizdays.Calendar(weekdays=[], name='Actual')
        return _CALENDAR_INSTANCE

    os.makedirs(data_dir, exist_ok=True)
    xls_path = os.path.join(data_dir, "feriados_nacionais.xls")
    
    try:
        # Se não existe, baixa da ANBIMA
        if not os.path.exists(xls_path):
            print(f"Baixando feriados da ANBIMA: {_URL_ANBIMA_FERIADOS}")
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(_URL_ANBIMA_FERIADOS, verify=False)
            if response.status_code == 200:
                with open(xls_path, 'wb') as f:
                    f.write(response.content)
            else:
                raise ValueError(f"HTTP {response.status_code}")
                
        # Lê o Excel
        df = pd.read_excel(xls_path)
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        df = df.dropna(subset=['Data'])
        holidays_dates = [d.date() for d in df['Data']]
        
        # O arquivo da ANBIMA muitas vezes não tem finais de semana, 
        # mas bizdays resolve passando os finais de semana.
        cal = bizdays.Calendar(holidays=holidays_dates, weekdays=['Saturday', 'Sunday'], name='B3')
        _CALENDAR_INSTANCE = cal
        
    except Exception as e:
        print(f"Falha ao carregar calendário expandido da ANBIMA ({e}). Realizando fallback para bizdays nativo B3 (limite 2026).")
        try:
            _CALENDAR_INSTANCE = bizdays.Calendar.load('B3')
        except Exception:
            # Fallback final em caso de o load('B3') falhar
            _CALENDAR_INSTANCE = bizdays.Calendar(weekdays=['Saturday', 'Sunday'], name='B3')

    return _CALENDAR_INSTANCE
