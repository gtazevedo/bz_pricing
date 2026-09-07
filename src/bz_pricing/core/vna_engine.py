import pandas as pd
import numpy as np
import datetime

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

    def calculate_vna(self, target_date: datetime.date, cal) -> float:
        """
        Calcula o VNA projetado para target_date, partindo de base_vna_date.
        `cal` é a instância do bizdays.Calendar(B3)
        """
        target_date = pd.to_datetime(target_date).date()
        if target_date == self.base_vna_date:
            return self.base_vna_value
            
        if target_date < self.base_vna_date:
            # Implementar decálculo se necessário
            raise NotImplementedError("Cálculo retroativo de VNA não implementado ainda.")

        current_date = self.base_vna_date
        current_vna = self.base_vna_value
        
        while current_date < target_date:
            # Encontra o dia 15 do mês seguinte
            next_month = current_date.replace(day=1) + pd.DateOffset(months=1)
            next_15 = next_month.replace(day=15).date()
            
            # Mês de referência do IPCA é sempre o mês anterior ao mês do dia 15
            ref_month = next_15.replace(day=1) - pd.DateOffset(months=1)
            
            try:
                ipca_rate = self.get_ipca_for_month(ref_month)
            except ValueError:
                # Se não temos a projeção desse mês, paramos ou repetimos a última?
                # Vamos simplificar repetindo a última projeção disponível se não encontrar
                if not self.ipca_projections.empty:
                    ipca_rate = self.ipca_projections.iloc[-1]
                else:
                    raise
            
            # Se target_date for ANTES do próximo dia 15, usamos a regra pro-rata
            if target_date <= next_15:
                du_calc = cal.bizdays(current_date, target_date)
                du_total = cal.bizdays(current_date, next_15)
                # Fórmula pro-rata da ANBIMA
                fator = (1.0 + ipca_rate) ** (du_calc / du_total)
                current_vna = np.trunc((current_vna * fator) * 1000000.0) / 1000000.0
                break
            else:
                # Se vai passar do dia 15, calcula o VNA cheio do dia 15
                du_total = cal.bizdays(current_date, next_15)
                fator = (1.0 + ipca_rate)
                current_vna = np.trunc((current_vna * fator) * 1000000.0) / 1000000.0
                current_date = next_15
                
        return current_vna
