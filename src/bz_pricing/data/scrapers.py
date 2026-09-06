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
        
    def download_taxas_referenciais(self, produto: str, data: datetime.date) -> str:
        """
        Usa o Playwright para emular um usuário, selecionando o produto 
        e clicando no link de download do histórico para a data fornecida.
        """
        date_str = data.strftime("%d/%m/%Y")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            
            # Vai para a página
            page.goto("https://sistemaswebb3-derivativos.b3.com.br/referenceRatesPage/all?language=pt-br")
            
            # Aguarda o select carregar e seleciona o produto (ex: 'PRE')
            page.wait_for_selector('select#product', timeout=15000)
            page.select_option('select#product', value=produto)
            
            # Aguarda a tabela atualizar (clique em buscar se existir ou apenas aguardar API)
            # A B3 carrega a tabela logo após o select
            page.wait_for_timeout(3000) # delay para render
            
            # Navega até o final (histórico de arquivos)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
            # No frontend atual, o histórico normalmente carrega arquivos disponíveis
            # Como a interface da B3 tem o link direto para download de XLS/CSV:
            # O texto do link de download costuma conter a data
            link_selector = f"a:has-text('{date_str}')"
            
            try:
                # Vamos esperar o link aparecer (se disponível no histórico do frontend)
                page.wait_for_selector(link_selector, timeout=10000)
                
                # Aguardamos o evento de download
                with page.expect_download() as download_info:
                    page.click(link_selector)
                
                download = download_info.value
                
                # Define o caminho de salvamento
                ext = download.suggested_filename.split('.')[-1]
                output_path = os.path.join(self.output_dir, f"taxas_{produto}_{data.strftime('%Y%m%d')}.{ext}")
                download.save_as(output_path)
                
                # Se for CSV mas o usuário pediu um formato específico, apenas garantimos a extensão
                # (A B3 geralmente envia como CSV ou XLS)
                
            except Exception as e:
                browser.close()
                raise ValueError(f"Não foi possível encontrar/clicar no link para {produto} na data {date_str}. Erro: {e}")
                
            browser.close()
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
