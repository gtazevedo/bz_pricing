import calendar
import datetime
import numpy as np
import pandas as pd

class VNAEngine:
    """
    Engine para estimar e projetar o Valor Nominal Atualizado (VNA) da ANBIMA 
    com base no IPCA histórico (SGS) e projeções (Focus).
    """
    def __init__(self, base_vna_date: datetime.date, base_vna_value: float, data_dir: str = None):
        self.base_vna_date = pd.to_datetime(base_vna_date).date()
        self.base_vna_value = base_vna_value
        self.data_dir = data_dir
        self.ipca_history = pd.Series(dtype=float)
        self.ipca_projections = pd.Series(dtype=float)

    def load_ipca_history(self, csv_path: str):
        """
        Carrega histórico de IPCA mensal.
        O CSV deve ter pelo menos 'Date' e 'Value' (em %).
        """
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        # Ajuste para ser o índice de mês (1º dia do mês)
        df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
        df = df.set_index('Date').sort_index()
        if 'Value' in df.columns:
            self.ipca_history = df['Value'] / 100.0
            
    def load_ipca_projections(self, csv_path: str):
        """
        Carrega projeções do Focus.
        O CSV deve ter 'Date' (mês de referência) e 'Value' (mediana projetada em %).
        """
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
        df = df.set_index('Date').sort_index()
        if 'Value' in df.columns:
            self.ipca_projections = df['Value'] / 100.0

    def get_ipca_for_month(self, target_month: datetime.date) -> float:
        """
        Retorna o IPCA (histórico ou projetado) para o mês alvo.
        """
        target_ts = pd.to_datetime(target_month).replace(day=1)
        if target_ts in self.ipca_history.index and pd.notna(self.ipca_history.loc[target_ts]):
            return self.ipca_history.loc[target_ts]
        elif target_ts in self.ipca_projections.index and pd.notna(self.ipca_projections.loc[target_ts]):
            return self.ipca_projections.loc[target_ts]
        else:
            raise ValueError(f"Faltam dados de IPCA (histórico ou projetado) para {target_ts.strftime('%Y-%m')}")

    @classmethod
    def from_bcb(cls, base_vna_date: datetime.date, base_vna_value: float,
                 indicator: str = "IPCA", ref_date: datetime.date = None) -> "VNAEngine":
        """
        Constrói um VNAEngine populado com a série histórica oficial do BCB (SGS)
        e as expectativas mensais do Relatório Focus.
        """
        from bcb import sgs
        from bz_pricing.data.scrapers import B3Scraper

        ind_upper = indicator.upper().strip()
        series_id = 189 if "IGP" in ind_upper else 433
        focus_ind = "IGP-M" if "IGP" in ind_upper else "IPCA"

        base_dt = pd.to_datetime(base_vna_date).date()
        end_dt = pd.to_datetime(ref_date).date() if ref_date else datetime.date.today()
        start_hist = (base_dt.replace(day=1) - pd.DateOffset(months=3)).date()

        engine = cls(base_vna_date=base_dt, base_vna_value=base_vna_value)
        try:
            df_hist = sgs.get({"VAL": series_id}, start=start_hist.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
            if df_hist is not None and not df_hist.empty:
                s_hist = df_hist["VAL"].dropna() / 100.0
                s_hist.index = pd.to_datetime(s_hist.index).to_period("M").to_timestamp()
                engine.ipca_history = s_hist[~s_hist.index.duplicated(keep="last")].sort_index()
        except Exception:
            pass

        try:
            scraper = B3Scraper()
            df_focus = scraper.download_focus_expectations(indicator=focus_ind, limit=200, base_date=end_dt)
            if df_focus is not None and not df_focus.empty:
                df_f = df_focus.copy()
                if "DataReferencia" in df_f.columns and "Mediana" in df_f.columns:
                    df_f["RefTS"] = pd.to_datetime(df_f["DataReferencia"], format="%m/%Y", errors="coerce")
                    df_f = df_f.dropna(subset=["RefTS", "Mediana"]).drop_duplicates(subset=["RefTS"], keep="first")
                    s_proj = pd.Series(df_f["Mediana"].astype(float).values / 100.0, index=df_f["RefTS"]).sort_index()
                    engine.ipca_projections = s_proj
        except Exception:
            pass

        return engine

    def calculate_vna(self, target_date: datetime.date, cal, anniversary_day: int = 15) -> float:
        """
        Calcula o VNA projetado para target_date, partindo de base_vna_date.
        `cal` é a instância do bizdays.Calendar(B3).
        """
        target_date = pd.to_datetime(target_date).date()
        if target_date == self.base_vna_date:
            return self.base_vna_value

        if target_date < self.base_vna_date:
            raise NotImplementedError("Cálculo retroativo de VNA não implementado ainda.")

        current_date = self.base_vna_date
        current_vna = self.base_vna_value

        while current_date < target_date:
            if current_date.day < anniversary_day:
                last_dom = calendar.monthrange(current_date.year, current_date.month)[1]
                next_anniv = current_date.replace(day=min(anniversary_day, last_dom))
            else:
                next_m = (current_date.replace(day=1) + pd.DateOffset(months=1)).date()
                last_dom = calendar.monthrange(next_m.year, next_m.month)[1]
                next_anniv = next_m.replace(day=min(anniversary_day, last_dom))

            # Mês de referência do índice é sempre o mês anterior ao aniversário
            ref_month = (next_anniv.replace(day=1) - pd.DateOffset(months=1))

            try:
                ipca_rate = self.get_ipca_for_month(ref_month)
            except ValueError:
                if not self.ipca_projections.empty:
                    ipca_rate = self.ipca_projections.iloc[-1]
                else:
                    raise

            # Se target_date for antes ou igual ao próximo aniversário, aplica pró-rata DU
            if target_date <= next_anniv:
                du_calc = cal.bizdays(current_date, target_date)
                du_total = cal.bizdays(current_date, next_anniv)
                fator = (1.0 + ipca_rate) ** (du_calc / du_total) if du_total > 0 else 1.0
                current_vna = np.trunc((current_vna * fator) * 1000000.0) / 1000000.0
                break
            else:
                fator = (1.0 + ipca_rate)
                current_vna = np.trunc((current_vna * fator) * 1000000.0) / 1000000.0
                current_date = next_anniv

        return float(current_vna)
