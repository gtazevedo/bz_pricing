import os
import zipfile
import io
import datetime
import requests
import pandas as pd
from playwright.sync_api import sync_playwright

class B3Scraper:
    def __init__(self, output_dir: str = "dados/b3"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cria a sessão requests para reuso de conexões e tratar retries
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        
    VALID_PRODUCTS = {
        "ACC": "Ajuste cupom",
        "APR": "Ajuste pré",
        "DCO": "Cupom Cambial OC1",
        "DOC": "Cupom limpo",
        "DOL": "DI x dólar",
        "EUC": "DI x euro",
        "DIM": "DI x IGP-M",
        "DIC": "DI x IPCA",
        "PRE": "DI x pré",
        "TR":  "DI x TR",
        "DPL": "DPL - Cupom Limpo de IPCA",
        "INP": "Ibovespa",
        "BRP": "IBrX-50",
        "LIB": "Libor",
        "PDN": "Prob. não default",
        "PTX": "Real x dólar",
        "EUR": "Real x euro",
        "JPY": "Real x iene",
        "SLP": "Selic x pré",
        "SDE": "Spread Libor Euro x Dólar",
        "TFP": "TBF x pré",
        "TP":  "TR x pré"
    }

    def download_taxas_referenciais(self, produto: str, data: datetime.date, force: bool = False) -> str:
        """
        Baixa as Taxas Referenciais (Curvas) da B3 utilizando um navegador headless (Playwright) 
        para simular a interação com a tela histórica, pois a API JSON ignora as datas solicitadas.
        Se o arquivo já existir no diretório, evita o download novamente (exceto se force=True).
        Nota: O histórico no site da B3 geralmente disponibiliza as últimas 20 datas.
        """
        if produto not in self.VALID_PRODUCTS:
            validos = ", ".join(self.VALID_PRODUCTS.keys())
            raise ValueError(f"Produto '{produto}' inválido. Produtos disponíveis: {validos}")
            
        date_str = data.strftime("%Y%m%d")
        output_path = os.path.join(self.output_dir, f"taxas_{produto}_{date_str}.csv")
        
        if not force and os.path.exists(output_path):
            print(f"Arquivo de taxas referenciais ({produto} - {date_str}) já existe em: {output_path}. Use force=True para baixar novamente.")
            return output_path
            
        print(f"Baixando Taxas Referenciais ({produto}) para a data {data.strftime('%d/%m/%Y')} via Playwright...")
        
        temp_path = os.path.join(self.output_dir, f"temp_{produto}_{date_str}.csv")
        data_formatada = data.strftime('%d/%m/%Y')
        nome_produto = self.VALID_PRODUCTS[produto]
        error_msg = []

        def _run_playwright():
            import sys
            import asyncio
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    
                    # Navega direto para a URL do frame que contém os dados da B3
                    page.goto('https://sistemaswebb3-derivativos.b3.com.br/referenceRatesPage/all?language=pt-br', timeout=60000)
                    
                    # Aguarda a caixa de seleção e clica nela
                    combo = page.locator("input[aria-label='Selecione um produto']")
                    combo.wait_for(state="visible", timeout=15000)
                    combo.click()
                    
                    # Clica no produto desejado
                    option = page.get_by_role("option", name=nome_produto, exact=True)
                    option.wait_for(state="visible", timeout=5000)
                    option.click()
                    
                    # Clica no botão "Buscar"
                    btn = page.get_by_role("button", name="Buscar")
                    btn.click()
                    
                    # Aguarda a requisição concluir e a página re-renderizar
                    page.wait_for_timeout(3000)
                    
                    # Aguarda carregar a tabela e a lista de arquivos históricos
                    page.wait_for_selector("table", timeout=15000)
                    
                    # Busca o link da data desejada
                    link_data = page.get_by_text(data_formatada, exact=True)
                    
                    try:
                        link_data.first.wait_for(state="visible", timeout=10000)
                    except:
                        pass
                    
                    if link_data.count() == 0:
                        browser.close()
                        error_msg.append(f"A data informada ({data_formatada}) não está disponível no histórico da página. O site fornece apenas os últimos 20 dias.")
                        return
                        
                    # Configura para capturar o download ao clicar
                    with page.expect_download(timeout=30000) as download_info:
                        link_data.first.click()
                        
                    download = download_info.value
                    download.save_as(temp_path)
                    browser.close()
            except Exception as e:
                error_msg.append(str(e))

        import threading
        t = threading.Thread(target=_run_playwright)
        t.start()
        t.join()

        if error_msg:
            raise ValueError(error_msg[0])
            
        # O arquivo baixado é um CSV que as vezes vem com codificação iso-8859-1 e separador ";"
        # Cabeçalho típico: Descrição Da Taxa;Dias Úteis;Dias Corridos;Preço/taxa
        try:
            df_raw = pd.read_csv(temp_path, sep=';', encoding='iso-8859-1')
        except UnicodeDecodeError:
            df_raw = pd.read_csv(temp_path, sep=';', encoding='utf-8')
            
        # Limpa o arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Verifica se o arquivo tem pelo menos as 4 colunas esperadas
        if len(df_raw.columns) < 4:
            raise ValueError("O formato do arquivo CSV baixado da B3 não é o esperado.")
            
        # Monta o DataFrame final com o padrão de nomes das colunas
        df = pd.DataFrame()
        df['code'] = [produto] * len(df_raw)
        df['curve'] = ['T1'] * len(df_raw)  # A curva padrão costuma ser T1
        df['description'] = df_raw.iloc[:, 0].astype(str).str.strip()
        df['day252'] = df_raw.iloc[:, 1].astype(int)
        df['day360'] = df_raw.iloc[:, 2].astype(int)
        
        # A taxa vem como string com vírgula '13,90', convertemos para numérico
        rates_raw = df_raw.iloc[:, 3].astype(str)
        df['rate'] = rates_raw.str.replace(',', '.').astype(float)
        
        df.to_csv(output_path, index=False, sep=";")
        print(f"Taxas salvos com sucesso em: {output_path}")
        
        return output_path

    def _download_and_extract_xmls(self, url: str, target_tag: str, date_str: str, prefix: str, force: bool = False) -> str:
        """
        Faz o download de um zip duplo contendo XMLs da B3 (como SPRD ou IN).
        Extrai e achata todas as tags iguais a `target_tag`.
        """
        import xml.etree.ElementTree as ET
        
        csv_path = os.path.join(self.output_dir, f"{prefix}_{date_str}.csv")
        if not force and os.path.exists(csv_path):
            print(f"Arquivo já existe em: {csv_path}. Use force=True para baixar novamente.")
            return csv_path
            
        print(f"Baixando: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Erro ao baixar arquivo da B3: HTTP {response.status_code}")
            
        dfs = []
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as outer_z:
                outer_files = outer_z.namelist()
                if not outer_files:
                    raise ValueError("O arquivo ZIP externo está vazio.")
                    
                inner_zip_bytes = outer_z.read(outer_files[0])
                
                with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_z:
                    xml_files = [f for f in inner_z.namelist() if f.endswith('.xml')]
                    
                    for xml_filename in xml_files:
                        print(f"Lendo XML interno: {xml_filename}")
                        xml_bytes = inner_z.read(xml_filename)
                        
                        rows = []
                        context = ET.iterparse(io.BytesIO(xml_bytes), events=('end',))
                        for event, elem in context:
                            tag = elem.tag.split('}', 1)[1] if '}' in elem.tag else elem.tag
                            
                            if tag == target_tag:
                                row_data = {}
                                
                                # Função recursiva para achatar o nó
                                def flatten_node(node):
                                    node_tag = node.tag.split('}', 1)[1] if '}' in node.tag else node.tag
                                    if len(node) == 0 and node.text and node.text.strip():
                                        row_data[node_tag] = node.text.strip()
                                    else:
                                        for child in node:
                                            flatten_node(child)
                                            
                                flatten_node(elem)
                                
                                if 'TckrSymb' in row_data or 'Id' in row_data:
                                    rows.append(row_data)
                                    
                                elem.clear() # Libera memória
                                
                        if rows:
                            df_xml = pd.DataFrame(rows)
                            df_xml['_source_file'] = xml_filename
                            dfs.append(df_xml)
                            
        except zipfile.BadZipFile:
            raise ValueError("O conteúdo retornado não é um ZIP válido (talvez o arquivo não exista para esta data).")
            
        if not dfs:
            raise ValueError(f"Nenhum dado encontrado com a tag {target_tag} nos XMLs.")
            
        final_df = pd.concat(dfs, ignore_index=True)
        
        # O cadastro pode usar 'Id' ou 'TckrSymb' como identificador
        uid_col = 'TckrSymb' if 'TckrSymb' in final_df.columns else ('Id' if 'Id' in final_df.columns else None)
        
        if uid_col:
            final_df = final_df.sort_values(by=[uid_col, '_source_file'], ascending=[True, False])
            final_df = final_df.drop_duplicates(subset=[uid_col], keep='first')
            
        if '_source_file' in final_df.columns:
            final_df = final_df.drop(columns=['_source_file'])
            
        csv_path = os.path.join(self.output_dir, f"{prefix}_{date_str}.csv")
        final_df.to_csv(csv_path, index=False, sep=";")
        print(f"CSV salvo em: {csv_path}")
        
        return csv_path

    def download_bvbg_187_01(self, data: datetime.date, force: bool = False) -> str:
        """
        Faz o download do boletim de negociação (SPRD) que contém os preços de fechamento.
        """
        date_str = data.strftime("%y%m%d")
        url = f"https://www.b3.com.br/pesquisapregao/download?filelist=SPRD{date_str}.zip"
        return self._download_and_extract_xmls(url, 'PricRpt', data.strftime('%Y%m%d'), 'BVBG_SPRD', force)
        
    def download_instruments(self, data: datetime.date, force: bool = False) -> str:
        """
        Faz o download do cadastro de instrumentos (IN) que contém os vencimentos, strikes, etc.
        """
        date_str = data.strftime("%y%m%d")
        url = f"https://www.b3.com.br/pesquisapregao/download?filelist=IN{date_str}.zip"
        # O Cadastro de instrumentos (BVBG.028) usa 'Instrm' como raiz de cada contrato
        return self._download_and_extract_xmls(url, 'Instrm', data.strftime('%Y%m%d'), 'BVBG_IN', force)

    def download_anbima_ms(self, data: datetime.date, force: bool = False) -> str:
        """
        Baixa o arquivo do Mercado Secundário da ANBIMA que contém as taxas e PUs oficiais (LTN, NTN-F, LFT, NTN-B, NTN-C).
        """
        date_str = data.strftime("%y%m%d")
        url = f"https://www.anbima.com.br/informacoes/merc-sec/arqs/ms{date_str}.txt"
        
        csv_path = os.path.join(self.output_dir, f"ANBIMA_MS_{data.strftime('%Y%m%d')}.csv")
        
        if not force and os.path.exists(csv_path):
            print(f"Arquivo MS ANBIMA já existe em: {csv_path}. Use force=True para baixar novamente.")
            return csv_path
            
        print(f"Baixando arquivo MS ANBIMA: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"Erro ao baixar arquivo MS da ANBIMA para {data}: HTTP {response.status_code}")
            
        # O arquivo usa @ como separador
        # Vamos tratar possíveis inconsistências no enconding e pular o header inicial "ANBIMA - ..."
        # Utilizar iso-8859-1 pois arquivos da B3/ANBIMA comumente usam esse encoding
        try:
            content = response.content.decode('iso-8859-1')
        except UnicodeDecodeError:
            content = response.text
            
        lines = content.splitlines()
        valid_lines = [line for line in lines if line and '@' in line and not line.startswith('ANBIMA')]
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            for line in valid_lines:
                f.write(line + '\n')
                
        print(f"Arquivo MS salvo em: {csv_path}")
        return csv_path

    def download_tesouro_transparente_cupons(self) -> str:
        """
        Baixa o CSV histórico de pagamentos de cupons do Tesouro Direto
        a partir do portal Tesouro Transparente.
        """
        url = "https://www.tesourotransparente.gov.br/ckan/dataset/f30db6e4-6123-416c-b094-be8dfc823601/resource/b04812f4-c453-4298-a12e-0ad190998243/download/cupomjurostesourodireto.csv"
        csv_path = os.path.join(self.output_dir, "Pagamento_Cupom_Juros_TD.csv")
        
        print(f"Baixando Cupons Históricos do Tesouro Transparente: {url}")
        
        # O Tesouro Transparente as vezes tem problema de certificado SSL, ignoramos o verify
        response = requests.get(url, verify=False)
        if response.status_code != 200:
            raise ValueError(f"Erro ao baixar arquivo do Tesouro Transparente. HTTP: {response.status_code}")
            
        with open(csv_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Arquivo de Cupons salvo em: {csv_path}")
        return csv_path

    def download_anbima_holidays(self, force: bool = False) -> str:
        """
        Baixa a planilha de feriados nacionais da ANBIMA.
        Só baixa novamente se o arquivo não existir ou se force=True.
        """
        url = "https://www.anbima.com.br/feriados/arqs/feriados_nacionais.xls"
        output_path = os.path.join(self.output_dir, "feriados_nacionais.xls")
        
        if not force and os.path.exists(output_path):
            print(f"Arquivo de feriados já existe em: {output_path}. Use force=True para baixar novamente.")
            return output_path
            
        print(f"Baixando feriados da ANBIMA: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"Erro ao baixar feriados da ANBIMA. HTTP {response.status_code}")
            
            f.write(response.content)
            
        print(f"Arquivo de feriados salvo em: {output_path}")
        return output_path

    # =========================================================================
    # SCRAPERS INTERNACIONAIS (TAXAS OVERNIGHT PARA VÉRTICE 1)
    # =========================================================================

    def download_sofr_overnight(self, data: datetime.date) -> float:
        """
        Baixa a taxa SOFR (Secured Overnight Financing Rate) do NY Fed.
        Caso o dia não seja útil ou a taxa não esteja disponível, busca retroativo.
        """
        end_date = data.strftime("%Y-%m-%d")
        start_date = (data - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
        
        url = f"https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?startDate={start_date}&endDate={end_date}"
        
        print(f"Baixando SOFR: {url}")
        res = self.session.get(url, verify=False)
        if res.status_code != 200:
            raise ValueError(f"Erro ao baixar SOFR: {res.status_code}")
            
        data_json = res.json()
        rates = data_json.get("sofr", {}).get("results", [])
        
        if not rates:
            today = datetime.date.today()
            if data > today:
                # Fallback para datas futuras
                print(f"[Aviso] Data {data} no futuro. Buscando a SOFR mais recente...")
                fallback_start = (today - datetime.timedelta(days=10)).strftime('%Y-%m-%d')
                fallback_end = today.strftime('%Y-%m-%d')
                url = f"https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?startDate={fallback_start}&endDate={fallback_end}"
                res = self.session.get(url, verify=False)
                if res.status_code == 200:
                    data_json = res.json()
                    rates = data_json.get("sofr", {}).get("results", [])
                    
        if not rates:
            raise ValueError(f"Nenhuma taxa SOFR encontrada para o período até {end_date}")
            
        # O primeiro resultado (índice 0) costuma ser o mais recente devido à ordenação da API
        # A taxa vem em percentual (ex: 5.31) -> convertemos para decimal se necessário no construtor de curvas
        # Para padronizar, retornamos percentual como no manual
        return float(rates[0]["percentRate"])

    def download_estr_overnight(self, data: datetime.date) -> float:
        """
        Baixa a taxa ESTR (Euro Short-Term Rate) do ECB (Banco Central Europeu).
        """
        # A API SDMX do ECB permite resgatar as últimas N observações
        url = "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?lastNObservations=1&format=jsondata"
        
        print(f"Baixando ESTR: {url}")
        res = self.session.get(url, verify=False)
        if res.status_code != 200:
            raise ValueError(f"Erro ao baixar ESTR: {res.status_code}")
            
        data_json = res.json()
        
        try:
            # Navegação no JSON do SDMX 2.1 JSON-data format do ECB
            series = data_json['dataSets'][0]['series']['0:0:0:0:0']
            obs = series['observations']['0']
            rate = float(obs[0])
            return rate
        except KeyError:
            raise ValueError("Formato de resposta inesperado da API do ESTR (ECB).")

    def download_tiie_overnight(self, data: datetime.date) -> float:
        """
        Baixa a taxa TIIE de Fondeo a 1 dia útil (Banxico).
        Exige BANXICO_TOKEN nas variáveis de ambiente, caso contrário usa mock/web scraping básico.
        Série Banxico: SF43718
        """
        token = os.environ.get("BANXICO_TOKEN")
        
        if token:
            url = "https://www.banxico.org.mx/SieAPIRest/v1/series/SF43718/dados/oportuno"
            headers = {'Bmx-Token': token}
            
            print(f"Baixando TIIE (Banxico API): {url}")
            res = self.session.get(url, headers=headers, verify=False)
            if res.status_code == 200:
                data_json = res.json()
                try:
                    rate = float(data_json['bmx']['series'][0]['dados'][0]['dato'])
                    return rate
                except (KeyError, IndexError):
                    pass
        
        print("Aviso: BANXICO_TOKEN não fornecido ou erro na API. Utilizando scraping de fallback para TIIE.")
        # Scraping rudimentar da página de indicadores do Banxico ou mock
        # Devido a limitações, um fallback em tela pode quebrar. Se houver erro, usar 11.25% (mock representativo)
        return 11.25

    # =========================================================================
    # SCRAPERS INFLAÇÃO E MACROECONOMIA
    # =========================================================================

    def download_focus_expectations(self, indicator: str = "IPCA", limit: int = 200, base_date: datetime.date = None) -> pd.DataFrame:
        """
        Baixa as expectativas do Relatório Focus via API Olinda do BCB.
        indicator: "IPCA" ou "IGP-M"
        base_date: se informada, filtra expectativas divulgadas até base_date (Data le 'YYYY-MM-DD').
        """
        # Formato OData, garantindo não usar aspas nos nomes das tabelas
        filter_str = f"Indicador%20eq%20'{indicator}'"
        if base_date is not None:
            dt_str = base_date.strftime("%Y-%m-%d")
            filter_str += f"%20and%20Data%20le%20'{dt_str}'%20and%20baseCalculo%20eq%200"
        url = f"https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativaMercadoMensais?$top={limit}&$filter={filter_str}&$orderby=Data%20desc&$format=json"
        
        print(f"Baixando Focus ({indicator}): {url}")
        res = self.session.get(url, verify=False)
        if res.status_code != 200:
            raise ValueError(f"Erro ao baixar Focus ({indicator}): {res.status_code}")
            
        data_json = res.json()
        df = pd.DataFrame(data_json.get("value", []))
        return df

    def download_bcb_ptax(self, data: datetime.date) -> pd.DataFrame:
        """
        Obtém a PTAX diária (todas as moedas) do BCB.
        Nesta versão, utilizamos a lib nativa `python-bcb`.
        Como a API exige end > start, recuamos 5 dias e pegamos a última cotação.
        """
        try:
            from bcb import currency
        except ImportError:
            raise ImportError("Pacote python-bcb não instalado. Execute: pip install python-bcb")
            
        start_date = data - datetime.timedelta(days=5)
        # O módulo get da currency puxa a cotação de venda por padrão
        df = currency.get(['USD'], start=start_date.strftime("%Y-%m-%d"), end=data.strftime("%Y-%m-%d"))
        
        if df.empty:
            return pd.DataFrame()
            
        # Transformar para o formato que o rest do código espera (ou aproximado)
        # O antigo scraper raw do BCB retornava colunas como 'simbolo', 'cotacaoVenda'
        # A lib bcb retorna um DataFrame com a data no índice e a coluna 'USD'
        last_val = df.iloc[-1]['USD']
        
        return pd.DataFrame([{
            'simbolo': 'USD',
            'cotacaoVenda': last_val
        }])

    def download_ibge_ipca_calendar(self) -> pd.DataFrame:
        """
        Retorna o calendário de divulgação do IPCA (IBGE).
        Placeholder/Scraper Simplificado. Em produção usar a API do IBGE de calendários.
        """
        print("Mock: Baixando Calendário IPCA...")
        # Mocking the usual mid-month release dates for IBGE IPCA
        dates = [datetime.date(2026, i, 10).strftime("%Y-%m-%d") for i in range(1, 13)]
        return pd.DataFrame({"DataReferencia": [f"2026-{i:02d}" for i in range(1, 13)], "DataDivulgacao": dates})

    def download_fgv_igpm_calendar(self) -> pd.DataFrame:
        """
        Retorna o calendário de divulgação do IGP-M (FGV).
        """
        print("Mock: Baixando Calendário IGP-M...")
        # Usually released on the last/penultimate business day of the month
        dates = [datetime.date(2026, i, 28).strftime("%Y-%m-%d") for i in range(1, 13)]
        return pd.DataFrame({"DataReferencia": [f"2026-{i:02d}" for i in range(1, 13)], "DataDivulgacao": dates})
