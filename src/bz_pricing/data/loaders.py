import os
import pandas as pd
import datetime

class B3DataLoader:
    """
    Carrega e consolida os dados de negociação (SPRD) com o cadastro (IN).
    """
    def __init__(self, data_dir: str = "dados/b3"):
        self.data_dir = data_dir
        
    def load_derivatives(self, data: datetime.date, auto_download: bool = True) -> pd.DataFrame:
        """
        Lê os arquivos BVBG_SPRD e BVBG_IN de uma data específica,
        e faz um merge para trazer informações como Vencimento (XprtnDt)
        para a tabela de preços. Se os arquivos não existirem e auto_download=True,
        tenta baixá-los automaticamente da B3 via scraper.
        """
        date_str = data.strftime("%Y%m%d")
        
        path_sprd = os.path.join(self.data_dir, f"BVBG_SPRD_{date_str}.csv")
        path_in = os.path.join(self.data_dir, f"BVBG_IN_{date_str}.csv")
        
        if not os.path.exists(path_sprd) or not os.path.exists(path_in):
            if auto_download:
                try:
                    from bz_pricing.data.scrapers import B3Scraper
                    scraper = B3Scraper(output_dir=self.data_dir)
                    if not os.path.exists(path_sprd):
                        scraper.download_bvbg_187_01(data)
                    if not os.path.exists(path_in):
                        scraper.download_instruments(data)
                except Exception:
                    pass

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
        Retorna apenas os instrumentos de um determinado ativo-objeto (ex: DI1, DAP, BGI, IND, PETRP)
        já preparados para a construção de curvas de juros ou preços.
        """
        df = self.load_derivatives(data)
        
        if 'Asst' not in df.columns:
            # Fallback: tentar filtrar pelo Ticker (ex: DI1F25 começa com DI1)
            df_filtered = df[df['TckrSymb'].str.startswith(asst, na=False)].copy()
        else:
            df_filtered = df[df['Asst'] == asst].copy()
            
        # Filtra apenas os que possuem Taxa de Ajuste, Preço de Ajuste ou Último Preço
        # E que possuam data de vencimento válida
        if 'XprtnDt' in df_filtered.columns:
            df_filtered = df_filtered.dropna(subset=['XprtnDt'])
            
            # Se for contrato com taxa (DI1, DDI, DAP, SFR, EST, TIE), prioriza AdjstdQtTax
            # Se for contrato com preço/pontos (IND, WIN, BRI, SML, VIX, Ações), usa AdjstdQt
            if 'AdjstdQtTax' in df_filtered.columns and df_filtered['AdjstdQtTax'].notna().any():
                df_filtered = df_filtered.dropna(subset=['AdjstdQtTax'])
            elif 'AdjstdQt' in df_filtered.columns and df_filtered['AdjstdQt'].notna().any():
                df_filtered = df_filtered.dropna(subset=['AdjstdQt'])
            elif 'LastPric' in df_filtered.columns and df_filtered['LastPric'].notna().any():
                df_filtered = df_filtered.dropna(subset=['LastPric'])
                
            # Ordena por vencimento
            df_filtered = df_filtered.sort_values(by='XprtnDt')
            
        return df_filtered

    def load_anbima_ms(self, data: datetime.date, auto_download: bool = True) -> pd.DataFrame:
        """
        Lê o arquivo do Mercado Secundário (MS) da ANBIMA e formata as colunas essenciais.
        Se não existir e auto_download=True, tenta baixá-lo automaticamente via scraper.
        """
        date_str = data.strftime("%Y%m%d")
        path_csv = os.path.join(self.data_dir, f"ANBIMA_MS_{date_str}.csv")
        
        if not os.path.exists(path_csv) and auto_download:
            try:
                from bz_pricing.data.scrapers import B3Scraper
                scraper = B3Scraper(output_dir=self.data_dir)
                scraper.download_anbima_ms(data)
            except Exception:
                pass

        if not os.path.exists(path_csv):
            raise FileNotFoundError(f"Arquivo MS ANBIMA não encontrado: {path_csv}. Execute scraper.download_anbima_ms(data) primeiro.")
            
        df = pd.read_csv(path_csv, sep='@', decimal=',', parse_dates=['Data Referencia', 'Data Base/Emissao', 'Data Vencimento'])

        # Limpeza e padronização dos nomes das colunas
        df.rename(columns={
            'Titulo': 'Titulo',
            'Codigo SELIC': 'CodigoSelic',
            'Data Vencimento': 'DataVencimento',
            'Data Base/Emissao': 'DataEmissao',
            'Tx. Indicativas': 'TaxaIndicativa',
            'PU': 'PU'
        }, inplace=True)
        return df

    def load_taxas_referenciais(self, produto: str, data: datetime.date, auto_download: bool = True) -> pd.DataFrame:
        """
        Carrega o arquivo CSV de taxas referenciais da B3 (ex: LIB, PRE, DIC, DIM)
        já baixado no diretório de dados. Se não existir e auto_download=True, tenta
        baixá-lo via scraper.
        """
        date_str = data.strftime("%Y%m%d")
        path_csv = os.path.join(self.data_dir, f"taxas_{produto}_{date_str}.csv")
        if not os.path.exists(path_csv) and auto_download:
            try:
                from bz_pricing.data.scrapers import B3Scraper
                scraper = B3Scraper(output_dir=self.data_dir)
                scraper.download_taxas_referenciais(produto, data)
            except Exception:
                pass

        if not os.path.exists(path_csv):
            raise FileNotFoundError(f"Arquivo de taxas referenciais não encontrado: {path_csv}")
        return pd.read_csv(path_csv, sep=';')

    def get_anbima_bonds(self, data: datetime.date, bond_type: str = 'LTN') -> pd.DataFrame:
        """
        Retorna os títulos públicos (ex: LTN, NTN-B) do Mercado Secundário da ANBIMA,
        mascarados no formato B3 (XprtnDt, AdjstdQtTax) para serem usados
        diretamente no B3CurveBuilder.
        """
        df = self.load_anbima_ms(data)
        
        # Filtra pelo tipo de título
        df_filtered = df[df['Titulo'] == bond_type].copy()
        
        # A B3CurveBuilder espera as colunas XprtnDt (Data) e AdjstdQtTax (Taxa em %)
        # A Tx. Indicativa do MS ANBIMA já está em % (ex: 13.76) e já foi lida como float
        
        # Cria as colunas esperadas pelo Builder
        if pd.api.types.is_datetime64_any_dtype(df_filtered['DataVencimento']):
            df_filtered['XprtnDt'] = df_filtered['DataVencimento']
        else:
            df_filtered['XprtnDt'] = pd.to_datetime(df_filtered['DataVencimento'], format='%Y%m%d', errors='coerce')
            
        df_filtered['AdjstdQtTax'] = df_filtered['TaxaIndicativa']
        
        # Retira NAs e ordena por vencimento
        df_filtered = df_filtered.dropna(subset=['XprtnDt', 'AdjstdQtTax'])
        df_filtered = df_filtered.sort_values(by='XprtnDt')
        
        return df_filtered

    def load_tesouro_transparente_cupons(self) -> pd.DataFrame:
        """
        Lê o CSV de pagamentos de cupons do Tesouro Direto.
        """
        path_csv = os.path.join(self.data_dir, "Pagamento_Cupom_Juros_TD.csv")
        
        df = pd.read_csv(path_csv, sep=';')
        
        # Converte as datas que vem no formato DD/MM/YYYY
        df['Vencimento do Titulo'] = pd.to_datetime(df['Vencimento do Titulo'], format='%d/%m/%Y', errors='coerce')
        df['Data Resgate'] = pd.to_datetime(df['Data Resgate'], format='%d/%m/%Y', errors='coerce')
        
        # Converte o PU que tem vírgula
        if 'PU' in df.columns and df['PU'].dtype == object:
            df['PU'] = df['PU'].str.replace(',', '.').astype(float)
            
        return df
