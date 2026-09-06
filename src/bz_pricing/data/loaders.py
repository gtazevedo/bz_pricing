import os
import pandas as pd
import datetime

class B3DataLoader:
    """
    Carrega e consolida os dados de negociação (SPRD) com o cadastro (IN).
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def load_derivatives(self, data: datetime.date) -> pd.DataFrame:
        """
        Lê os arquivos BVBG_SPRD e BVBG_IN de uma data específica,
        e faz um merge para trazer informações como Vencimento (XprtnDt)
        para a tabela de preços.
        """
        date_str = data.strftime("%Y%m%d")
        
        path_sprd = os.path.join(self.data_dir, f"BVBG_SPRD_{date_str}.csv")
        path_in = os.path.join(self.data_dir, f"BVBG_IN_{date_str}.csv")
        
        if not os.path.exists(path_sprd):
            raise FileNotFoundError(f"Arquivo de negociação não encontrado: {path_sprd}")
        if not os.path.exists(path_in):
            raise FileNotFoundError(f"Arquivo de cadastro não encontrado: {path_in}")
            
        # Lê os CSVs
        # low_memory=False é usado devido à grande quantidade de colunas no IN
        df_sprd = pd.read_csv(path_sprd, sep=';', low_memory=False)
        df_in = pd.read_csv(path_in, sep=';', low_memory=False)
        
        # Filtra apenas as colunas essenciais do cadastro para não poluir
        # XprtnDt: Data de Vencimento
        # Asst: Ativo Objeto (ex: DI1, BGI, etc)
        # Sgnl: Sinal (para identificar se é compra/venda, não tão usado para curva geral)
        in_cols_to_keep = ['TckrSymb', 'XprtnDt', 'Asst', 'AsstDesc']
        # Interseção com as colunas que realmente existem no df_in
        in_cols_to_keep = [c for c in in_cols_to_keep if c in df_in.columns]
        
        df_in_sub = df_in[in_cols_to_keep].copy()
        
        # Merge (Left Join: mantém todos os preços do SPRD e traz as infos do IN)
        df_merged = pd.merge(df_sprd, df_in_sub, on='TckrSymb', how='left')
        
        # Converte a coluna XprtnDt para datetime
        if 'XprtnDt' in df_merged.columns:
            df_merged['XprtnDt'] = pd.to_datetime(df_merged['XprtnDt'], errors='coerce')
            
        # Limpeza de preços: o formato do XML pode trazer valores nulos ou vazios
        # O principal campo para curvas geralmente é AdjstdQt (PU de Ajuste) 
        # ou AdjstdQtTax (Taxa de Ajuste). Vamos garantir que sejam numéricos.
        for col in ['AdjstdQt', 'AdjstdQtTax', 'LastPric', 'OpnIntrst', 'TradAvrgPric']:
            if col in df_merged.columns:
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
                
        return df_merged
        
    def get_curve_instruments(self, data: datetime.date, asst: str = "DI1") -> pd.DataFrame:
        """
        Retorna apenas os instrumentos de um determinado ativo-objeto (ex: DI1, DAP, BGI)
        já preparados para a construção de curvas de juros.
        """
        df = self.load_derivatives(data)
        
        if 'Asst' not in df.columns:
            # Fallback: tentar filtrar pelo Ticker (ex: DI1F25 começa com DI1)
            df_filtered = df[df['TckrSymb'].str.startswith(asst, na=False)].copy()
        else:
            df_filtered = df[df['Asst'] == asst].copy()
            
        # Filtra apenas os que possuem Taxa de Ajuste, PU de ajuste ou Último Preço
        # E que possuam data de vencimento válida
        if 'XprtnDt' in df_filtered.columns:
            df_filtered = df_filtered.dropna(subset=['XprtnDt'])
            
            # Se a taxa existir, usamos. Caso contrário, usamos o preço de ajuste ou last price
            if 'AdjstdQtTax' in df_filtered.columns:
                df_filtered = df_filtered.dropna(subset=['AdjstdQtTax'])
            elif 'AdjstdQt' in df_filtered.columns:
                df_filtered = df_filtered.dropna(subset=['AdjstdQt'])
                
            # Ordena por vencimento
            df_filtered = df_filtered.sort_values(by='XprtnDt')
            
        return df_filtered
