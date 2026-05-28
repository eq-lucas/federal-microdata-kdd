import shutil
from src.constantes import pasta_data_02_staging_microdata_fies,pasta_data_02_staging_microdata_fies_errors,pasta_data_01_raw_microdata_fies,pasta_data_01_raw_microdata_inep,pasta_data_02_staging_microdata_inep
import pandas as pd
import numpy as np
import os
from pathlib import Path

def limpeza_avancada_staging_fies():
    print("\n" + "="*90)
    print("🧹 INICIANDO LIMPEZA MASSIVA E UNIVERSAL (STAGING) - 140+ COLUNAS")
    print("="*90)

    # 1. Função Blindada para converter os números bizarros do MEC (Renda, Notas, Mensalidades)
    def converter_br_para_float(valor):
        if pd.isna(valor):
            return np.nan
        
        val_str = str(valor).strip().upper()
        if val_str in ['NAN', 'NONE', 'NULL', '', '-']:
            return np.nan
            
        # Se tem vírgula, trata o padrão BR (ex: 84.692,00 ou 480,38)
        if ',' in val_str:
            val_str = val_str.replace('.', '')  # Remove ponto de milhar
            val_str = val_str.replace(',', '.') # Transforma vírgula decimal em ponto
        
        try:
            return float(val_str)
        except:
            return np.nan

    # Varre a pasta buscando os arquivos CSV que já foram deduplicados
    arquivos_fies = list(pasta_data_02_staging_microdata_fies.glob('*.csv'))

    for caminho in arquivos_fies:
        # Pula arquivos de log/erro
        if 'erro' in caminho.name.lower() or 'resultado' in caminho.name.lower():
            continue

        print(f"[*] Aplicando higienização profunda em: {caminho.name}")
        try:
            df = pd.read_csv(str(caminho), low_memory=False)
            
            # --- FASE 1: EXTERMÍNIO DE COLUNAS FANTASMAS ---
            cols_unnamed = [c for c in df.columns if 'Unnamed' in str(c)]
            if cols_unnamed:
                df.drop(columns=cols_unnamed, inplace=True)
            
            # --- FASE 2: VARREDURA UNIVERSAL DE TEXTOS E TIPAGEM ---
            # Não importa se tem 60 ou 140 colunas, o laço varre TODAS mantendo o nome original.
            for col in df.columns:
                col_lower = col.lower()

                # A. TRATAMENTO UNIVERSAL PARA TEXTOS (Strip, Upper e NaNs reais)
                if df[col].dtype == 'object':
                    # Transforma tudo em string temporariamente, tira espaços das pontas e joga pra Maiúsculo
                    df[col] = df[col].astype(str).str.strip().str.upper()
                    
                    # Substitui lixos textuais por vazio matemático real (np.nan)
                    lixos = ['NAN', 'NONE', 'NULL', '', '-', 'NÃO INFORMADO', 'NAO INFORMADO']
                    df[col] = df[col].replace(lixos, np.nan)

                # B. COLUNAS FINANCEIRAS E DE NOTAS (Conversão de Float)
                is_float_col = any(k in col_lower for k in [
                    'renda', 'nota', 'média', 'media', 'redação', 'redacao', 'tecnologias', 'tec', 
                    'percentual', 'valor bruto', 'semestre bruto', 'valor do curso', 
                    'índice', 'indice', 'semestre fies'
                ])
                
                # C. COLUNAS INTEIRAS (Anos, Semestres, Vagas, IDs)
                is_int_col = any(k in col_lower for k in [
                    'ano', 'semestre do', 'semestre de', 'id ', 'código', 'codigo', 
                    'vagas', 'qtde', 'nº', 'classificação', 'classificacao', 'cod.'
                ])

                # Aplica as lógicas matemáticas
                if is_float_col:
                    df[col] = df[col].apply(converter_br_para_float)
                elif is_int_col and not is_float_col:
                    # O tipo 'Int64' (com I maiúsculo) do Pandas permite ter Inteiros perfeitos (2019) 
                    # convivendo com NaNs na mesma coluna, sem forçar tudo a virar Float (2019.0)
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

            # --- FASE 3: SOBRESCRITA DO ARQUIVO LIMPO ---
            df.to_csv(str(caminho), index=False, encoding='utf-8')
            print(f"   ✅ [OK] {len(df.columns)} colunas perfeitamente formatadas!")
            
        except Exception as e:
            print(f"   [!] ERRO CRÍTICO ao processar {caminho.name}: {e}")

    print("\n" + "="*90)
    print("🏁 LIMPEZA AVANÇADA CONCLUÍDA! O STAGING AGORA É UM ESPELHO PERFEITO.")
    print("="*90 + "\n")



def staging_fies():


    print(f"\nBuscando arquivos originais do INEP em: {pasta_data_01_raw_microdata_fies}\n")

    # 2. Dicionário Mestre: 'Nome Feio Original' -> ('Nome Bonito', 'Pasta Destino', 'Usa Pandas?')
    mapa_arquivos = {
        # --- 2019 ---
        'relatorio_inscricao_dados_abertos_fies_12019.csv': ('fies_1_inscricao_2019_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),
        'relatorio_inscricao_dados_abertos_fies_22019.csv': ('fies_2_inscricao_2019_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),
        'relatorio_dados_abertos_oferta_12019_18102021.csv': ('fies_1_ofertas_2019_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),
        'relatorio_dados_abertos_oferta_22019_18102021.csv': ('fies_2_ofertas_2019_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),

        # --- 2020 ---
        'relatorio_inscricao_dados_abertos_fies_12020.csv': ('fies_1_inscricao_2020_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),
        'relatorio_inscricao_dados_abertos_fies_22020.csv': ('fies_2_inscricao_2020_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),
        'relatorio_dados_abertos_oferta_12020_18102021.csv': ('fies_1_ofertas_2020_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),
        'relatorio_dados_abertos_oferta_22020_18102021.csv': ('fies_2_ofertas_2020_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),

        # --- 2021 ---
        'relatorio_inscricao_dados_abertos_fies_12021.csv': ('fies_1_inscricao_2021_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),
        'relatorio_inscricao_dados_abertos_fies_22021.csv': ('fies_2_inscricao_2021_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),
        'relatorio_dados_abertos_oferta_12021_18102021.csv': ('fies_1_ofertas_2021_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),
        'relatorio_dados_abertos_oferta_22021_18102021.csv': ('fies_2_ofertas_2021_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),

        # --- 2022 ---
        'relatorio_dados_abertos_oferta_12022_15072022.csv': ('fies_1_ofertas_2022_sem_duplicata.csv', pasta_data_02_staging_microdata_fies, True),

        # --- Arquivo Especial (Resultado/Erro) ---
        'relatorio_resultado_fies_12021.csv': ('resultado_fies_2020_1.csv', pasta_data_02_staging_microdata_fies_errors, False)
    }

    arquivos_processados = 0

    # 3. Executa a limpeza inicial para o staging ( etapa de preparação )
    for nome_original, (novo_nome, pasta_destino, usa_pandas) in mapa_arquivos.items():
        caminho_bruto = pasta_data_01_raw_microdata_fies / nome_original
        caminho_limpo = pasta_destino / novo_nome

        if caminho_bruto.exists():
            print(f"[*] Processando: '{nome_original}' \n    -> Salvando como: '{novo_nome}'")
            
            if usa_pandas:
                try:
                    # Lê o bruto da pasta 01_raw
                    df = pd.read_csv(str(caminho_bruto), sep=';', encoding='latin-1', decimal=',', low_memory=False)
                    # Remove duplicatas
                    df_limpo = df.drop_duplicates()
                    # Salva direto na pasta 02_staging com o nome bonito
                    df_limpo.to_csv(str(caminho_limpo), index=False)
                    arquivos_processados += 1
                except Exception as e:
                    print(f"    [!] ERRO no Pandas: {e}")
            else:
                try:
                    # O arquivo de resultado não precisa de Pandas, então só transferimos e renomeamos
                    shutil.copy2(str(caminho_bruto), str(caminho_limpo))
                    arquivos_processados += 1
                except Exception as e:
                    print(f"    [!] ERRO ao transferir: {e}")
        else:
            print(f"[AVISO] '{nome_original}' não encontrado na pasta 01_raw. Pulando.")

    print(f"Limpeza Concluída! {arquivos_processados} de {len(mapa_arquivos)} arquivos salvos em 02_staging.")
    limpeza_avancada_staging_fies()








def staging_inep():

    # 2. Padrões de NOME/CAMINHO
    data_subfolder = 'dados'

    # 3. Nomes dos ficheiros de ORIGEM
    source_filenames_templates = [
        'MICRODADOS_CADASTRO_CURSOS_{year}.CSV',
        'MICRODADOS_CADASTRO_IES_{year}.CSV'
    ]

    # 4. Anos a processar
    start_year = 2016
    end_year = 2024
    years_to_process = range(start_year, end_year + 1)

    # --- FIM DA CONFIGURAÇÃO ---


    print("\n--- 1. INICIANDO CÓPIA E RENAME (Usando os.listdir) ---")

    copied_count = 0
    error_count = 0

    try:
        # Lista todo o conteúdo da pasta '01_raw'
        all_items_in_base = os.listdir(pasta_data_01_raw_microdata_inep)
    except Exception as e:
        print(f"ERRO: Não foi possível ler o conteúdo de '{pasta_data_01_raw_microdata_inep}'. Erro: {e}")
        return

    # Loop pelos anos
    for year in years_to_process:
        print(f"\n--- Processando ano: {year} ---")
        
        year_str = str(year)
        
        # Encontra a pasta do ano usando 'in' (contém)
        matching_folders = [
            item for item in all_items_in_base 
            if year_str in item and os.path.isdir(os.path.join(pasta_data_01_raw_microdata_inep, item))
        ]

        if not matching_folders:
            print(f"  AVISO: Nenhuma pasta encontrada para o ano {year}. Pulando.")
            continue
        
        # Assumimos que a primeira pasta encontrada é a correta para o ano
        current_year_folder_name = matching_folders[0] 
        
        # MÁSCARA PRO PRINT NÃO TRAVAR O TERMINAL: Mantemos a variável original intacta para o os.path, 
        # e criamos uma versão "limpa" apenas para mostrar no ecrã.
        nome_seguro_para_print = current_year_folder_name.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')
        print(f"  Pasta encontrada: {nome_seguro_para_print}")

        # Caminho completo da PASTA RAIZ do ano
        root_folder_to_delete = os.path.join(pasta_data_01_raw_microdata_inep, current_year_folder_name)
        
        # Caminho completo da subpasta 'dados'
        full_data_subfolder_path = os.path.join(root_folder_to_delete, data_subfolder)

        # Loop pelos dois ficheiros
        for source_template in source_filenames_templates:
            source_filename = source_template.format(year=year)
            full_source_path = os.path.join(full_data_subfolder_path, source_filename)
            
            # Monta o destino (renomeando para .csv minúsculo)
            destination_filename = source_filename.replace('.CSV', '.csv')
            full_destination_path = os.path.join(pasta_data_02_staging_microdata_inep, destination_filename)

            # Verifica se o ficheiro de ORIGEM existe
            if os.path.exists(full_source_path):
                print(f"  Ficheiro encontrado: {source_filename}")
                try:
                    # COPIA o ficheiro e renomeia-o (sem apagar o original)
                    shutil.copy2(full_source_path, full_destination_path)
                    print(f"  >>> SUCESSO: Copiado para: {full_destination_path}")
                    copied_count += 1
                except Exception as e:
                    print(f"  !!! ERRO: Falha ao copiar/renomear {source_filename}. Erro: {e}")
                    error_count += 1
            else:
                print(f"  AVISO: Ficheiro de origem não encontrado: {source_filename}. Pulando.")
            
    # --- RESUMO FINAL ---

    print("\n--- PROCESSO COMPLETO CONCLUÍDO ---")
    print(f"Ficheiros copiados e renomeados: {copied_count}")
    print(f"Total de erros encontrados: {error_count}")