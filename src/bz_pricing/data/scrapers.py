import os
import zipfile
import io
import datetime
import requests
import pandas as pd
from playwright.sync_api import sync_playwright

class B3Scraper:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
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

    def download_taxas_referenciais(self, produto: str, data: datetime.date) -> str:
        """
        Baixa as Taxas Referenciais direto da API JSON da B3 (Muito mais rápido e robusto que Scraping via tela).
        """
        if produto not in self.VALID_PRODUCTS:
            validos = ", ".join(self.VALID_PRODUCTS.keys())
            raise ValueError(f"Produto '{produto}' inválido. Produtos disponíveis: {validos}")
            
        import base64
        import json
        
        date_str = data.strftime("%Y-%m-%d")
        
        # A API da B3 limita o pageSize (geralmente em 100), então precisamos paginar
        all_results = []
        page = 1
        
        print(f"Baixando Taxas Referenciais ({produto}) via API...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        
        while True:
            payload = {
                "language": "pt-br",
                "id": produto,
                "pageNumber": page,
                "pageSize": 100,
                "date": date_str
            }
            
            # Formata o JSON garantindo que não há espaços sobrando (separators) para a B3 não rejeitar
            b64_payload = base64.b64encode(json.dumps(payload, separators=(',', ':')).encode('utf-8')).decode('utf-8')
            url = f"https://sistemaswebb3-derivativos.b3.com.br/referenceRatesProxy/Search/GetList/{b64_payload}"
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise ValueError(f"Erro ao baixar da API B3. Status: {response.status_code}")
                
            data_json = response.json()
            results = data_json.get('results', [])
            
            if not results:
                break
                
            all_results.extend(results)
            page += 1
            
        if not all_results:
            raise ValueError(f"Nenhum dado retornado da B3 para o produto {produto} na data {date_str}.")
            
        df = pd.DataFrame(all_results)
        
        # O formato brasileiro de taxa devolvido na API vem como '13,90', convertemos para '13.90' e depois numérico
        if 'rate' in df.columns:
            df['rate'] = df['rate'].str.replace(',', '.').astype(float)
            
        # Garante a formatação do dia útil
        if 'day252' in df.columns:
            df['day252'] = df['day252'].astype(int)
            
        output_path = os.path.join(self.output_dir, f"taxas_{produto}_{data.strftime('%Y%m%d')}.csv")
        df.to_csv(output_path, index=False, sep=";")
        
        return output_path

    def _download_and_extract_xmls(self, url: str, target_tag: str, date_str: str, prefix: str) -> str:
        """
        Faz o download de um zip duplo contendo XMLs da B3 (como SPRD ou IN).
        Extrai e achata todas as tags iguais a `target_tag`.
        """
        import xml.etree.ElementTree as ET
        
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

    def download_bvbg_187_01(self, data: datetime.date) -> str:
        """
        Faz o download do boletim de negociação (SPRD) que contém os preços de fechamento.
        """
        date_str = data.strftime("%y%m%d")
        url = f"https://www.b3.com.br/pesquisapregao/download?filelist=SPRD{date_str}.zip"
        return self._download_and_extract_xmls(url, 'PricRpt', data.strftime('%Y%m%d'), 'BVBG_SPRD')
        
    def download_instruments(self, data: datetime.date) -> str:
        """
        Faz o download do cadastro de instrumentos (IN) que contém os vencimentos, strikes, etc.
        """
        date_str = data.strftime("%y%m%d")
        url = f"https://www.b3.com.br/pesquisapregao/download?filelist=IN{date_str}.zip"
        # O Cadastro de instrumentos (BVBG.028) usa 'Instrm' como raiz de cada contrato
        return self._download_and_extract_xmls(url, 'Instrm', data.strftime('%Y%m%d'), 'BVBG_IN')

    def download_anbima_ms(self, data: datetime.date) -> str:
        """
        Baixa o arquivo do Mercado Secundário da ANBIMA que contém as taxas e PUs oficiais (LTN, NTN-F, LFT, NTN-B, NTN-C).
        """
        date_str = data.strftime("%y%m%d")
        url = f"https://www.anbima.com.br/informacoes/merc-sec/arqs/ms{date_str}.txt"
        
        print(f"Baixando arquivo MS ANBIMA: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"Erro ao baixar arquivo MS da ANBIMA para {data}: HTTP {response.status_code}")
            
        csv_path = os.path.join(self.output_dir, f"ANBIMA_MS_{data.strftime('%Y%m%d')}.csv")
        
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
