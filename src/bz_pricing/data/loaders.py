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

    def load_options_catalog(
        self,
        data: datetime.date,
        asst: str = None,
        auto_download: bool = True,
    ) -> pd.DataFrame:
        """
        Carrega o cadastro de séries de opções do arquivo BVBG_IN para a data especificada,
        normalizando tipo (call/put), estilo (european/american), strike e vencimento.
        """
        date_str = data.strftime("%Y%m%d")
        path_in = os.path.join(self.data_dir, f"BVBG_IN_{date_str}.csv")

        if not os.path.exists(path_in) and auto_download:
            try:
                from bz_pricing.data.scrapers import B3Scraper
                scraper = B3Scraper(output_dir=self.data_dir)
                scraper.download_instruments(data)
            except Exception:
                pass

        if not os.path.exists(path_in):
            raise FileNotFoundError(f"Arquivo de cadastro não encontrado: {path_in}")

        df_in = pd.read_csv(path_in, sep=';', low_memory=False)
        if 'OptnTp' not in df_in.columns:
            return pd.DataFrame()

        opts = df_in[df_in['OptnTp'].notna()].copy()
        if asst is not None and 'Asst' in opts.columns:
            opts = opts[opts['Asst'] == asst].copy()

        keep_cols = [
            'TckrSymb', 'Asst', 'AsstDesc', 'SctyCtgy',
            'OptnTp', 'OptnStyle', 'ExrcStyle', 'ExrcPric',
            'XprtnDt', 'CtrctMltplr', 'AsstQtnQty',
        ]
        keep_cols = [c for c in keep_cols if c in opts.columns]
        opts = opts[keep_cols].drop_duplicates(subset=['TckrSymb']).copy()

        opts['option_type'] = (
            opts['OptnTp'].astype(str).str.upper().map({'CALL': 'call', 'PUTT': 'put', 'PUT': 'put'})
        )

        style_raw = pd.Series(index=opts.index, dtype=object)
        if 'OptnStyle' in opts.columns:
            style_raw = opts['OptnStyle']
        if 'ExrcStyle' in opts.columns:
            style_raw = style_raw.fillna(opts['ExrcStyle'])
        opts['option_style'] = (
            style_raw.astype(str).str.upper().map({'EURO': 'european', 'AMER': 'american'})
        )

        if 'ExrcPric' in opts.columns:
            opts['ExrcPric'] = pd.to_numeric(opts['ExrcPric'], errors='coerce')
        if 'CtrctMltplr' in opts.columns:
            opts['CtrctMltplr'] = pd.to_numeric(opts['CtrctMltplr'], errors='coerce')
        if 'XprtnDt' in opts.columns:
            opts['XprtnDt'] = pd.to_datetime(opts['XprtnDt'], errors='coerce')

        return opts.reset_index(drop=True)

    def load_options(
        self,
        data: datetime.date,
        asst: str = None,
        only_priced: bool = False,
        auto_download: bool = True,
    ) -> pd.DataFrame:
        """
        Consolida as séries de opções do cadastro (BVBG_IN) com os preços e liquidez
        do boletim de negociação (BVBG_SPRD) para uma data de referência.
        """
        date_str = data.strftime("%Y%m%d")
        path_sprd = os.path.join(self.data_dir, f"BVBG_SPRD_{date_str}.csv")

        if not os.path.exists(path_sprd) and auto_download:
            try:
                from bz_pricing.data.scrapers import B3Scraper
                scraper = B3Scraper(output_dir=self.data_dir)
                scraper.download_bvbg_187_01(data)
            except Exception:
                pass

        if not os.path.exists(path_sprd):
            raise FileNotFoundError(f"Arquivo de negociação não encontrado: {path_sprd}")

        df_catalog = self.load_options_catalog(data, asst=asst, auto_download=auto_download)
        if df_catalog.empty:
            return df_catalog

        df_sprd = pd.read_csv(path_sprd, sep=';', low_memory=False)
        df_merged = pd.merge(df_sprd, df_catalog, on='TckrSymb', how='inner')

        num_cols = [
            'AdjstdQt', 'PrvsAdjstdQt', 'LastPric', 'TradAvrgPric',
            'FrstPric', 'MinPric', 'MaxPric', 'OpnIntrst', 'RglrTxsQty',
        ]
        for col in num_cols:
            if col in df_merged.columns:
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

        ref_pric = pd.Series(float('nan'), index=df_merged.index, dtype=float)
        for col in ('AdjstdQt', 'LastPric', 'TradAvrgPric'):
            if col in df_merged.columns:
                ref_pric = ref_pric.fillna(df_merged[col])
        df_merged['RefPric'] = ref_pric

        if only_priced:
            df_merged = df_merged[df_merged['RefPric'].notna() & (df_merged['RefPric'] > 0)].copy()

        sort_cols = [c for c in ('XprtnDt', 'ExrcPric', 'option_type') if c in df_merged.columns]
        if sort_cols:
            df_merged = df_merged.sort_values(by=sort_cols).reset_index(drop=True)

        return df_merged

    def load_cotahist(self, data: datetime.date, auto_download: bool = True) -> pd.DataFrame:
        """Carrega o arquivo COTAHIST diário da B3 (vista e opções de ações/ETFs)."""
        ymd = data.strftime("%Y%m%d")
        csv_path = os.path.join(self.data_dir, f"COTAHIST_D{ymd}.csv")
        if not os.path.exists(csv_path) and auto_download:
            from bz_pricing.data.scrapers import B3Scraper
            scraper = B3Scraper(output_dir=self.data_dir)
            scraper.download_cotahist(data)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo COTAHIST não encontrado: {csv_path}")
        return pd.read_csv(csv_path, sep=";", low_memory=False)

    def load_equity_options(
        self,
        data: datetime.date,
        underlying: str = None,
        only_priced: bool = True,
        auto_download: bool = True,
    ) -> pd.DataFrame:
        """
        Carrega séries líquidas de opções sobre ações/ETFs do COTAHIST da B3,
        enriquecidas com estilo (europeia/americana) do cadastro BVBG_IN (quando disponível)
        e preço spot do ativo-objeto na mesma data.
        """
        df = self.load_cotahist(data, auto_download=auto_download)
        if df.empty:
            return df

        spots = df[df["TPMERC"] == 10][["CODNEG", "PREULT"]].set_index("CODNEG")["PREULT"].to_dict()
        opts = df[df["TPMERC"].isin([70, 80])].copy()
        opts["option_type"] = opts["TPMERC"].map({70: "call", 80: "put"})
        opts["TckrSymb"] = opts["CODNEG"].astype(str).str.strip()
        opts["ExrcPric"] = pd.to_numeric(opts["PREEXE"], errors="coerce")
        opts["XprtnDt"] = pd.to_datetime(opts["DATVEN"].astype(str), format="%Y%m%d", errors="coerce")
        opts["LastPric"] = pd.to_numeric(opts["PREULT"], errors="coerce")
        opts["MidBidAsk"] = 0.5 * (pd.to_numeric(opts["PREOFC"], errors="coerce") + pd.to_numeric(opts["PREOFV"], errors="coerce"))
        opts["RefPric"] = opts["LastPric"].where(opts["LastPric"] > 0, opts["MidBidAsk"])

        # Enriquecimento com cadastro BVBG_IN (se presente)
        try:
            cat = self.load_options_catalog(data, auto_download=False)
            if not cat.empty:
                cat_sub = cat[["TckrSymb", "Asst", "option_style", "CtrctMltplr"]].drop_duplicates("TckrSymb")
                opts = pd.merge(opts, cat_sub, on="TckrSymb", how="left")
        except FileNotFoundError:
            pass

        if "Asst" not in opts.columns:
            opts["Asst"] = opts["TckrSymb"].str[:4]
        else:
            opts["Asst"] = opts["Asst"].fillna(opts["TckrSymb"].str[:4])

        if "option_style" not in opts.columns:
            opts["option_style"] = opts["option_type"].map({"call": "american", "put": "european"})

        if underlying is not None:
            und_up = underlying.strip().upper()
            prefix = und_up[:4]
            opts = opts[(opts["Asst"] == und_up) | (opts["Asst"] == prefix) | (opts["TckrSymb"].str.startswith(prefix))].copy()
            spot_val = spots.get(und_up)
            if spot_val is not None:
                opts["spot_price"] = float(spot_val)

        if only_priced:
            opts = opts[opts["RefPric"].notna() & (opts["RefPric"] > 0) & (opts["TOTNEG"] > 0)].copy()

        sort_cols = [c for c in ("XprtnDt", "ExrcPric", "option_type") if c in opts.columns]
        if sort_cols:
            opts = opts.sort_values(by=sort_cols).reset_index(drop=True)
        return opts
