import pandas as pd
import os
import re
from pathlib import Path
from src.constantes import pasta_raiz_projeto,pasta_data_02_staging_microdata_fies,pasta_data_01_raw_microdata_fies,pasta_data_02_staging_microdata_inep,pasta_data_03_transform_inep,pasta_data_03_transform_fies



def transform_inep():
    print("\n" + "="*50)
    print("Iniciando Transformação: INEP (Dataset Mestre de Cursos)")
    print("="*50)

    # --- 2. COLUNAS ALVO (FILTRO DE MEMÓRIA) ---
    colunas_cine = [
        'NU_ANO_CENSO',
        'NO_CURSO',
        'CO_CURSO',
        'CO_CINE_AREA_GERAL',
        'NO_CINE_AREA_GERAL',
    ]

    lista_dfs = []
    arquivos_processados = 0

    print(f"Buscando arquivos na Staging: {pasta_data_02_staging_microdata_inep}")

    # --- 3. LOOP DE LEITURA E CONCATENAÇÃO ---
    for ano in range(2016, 2025): # Vai de 2016 até 2024
        nome_arquivo = f'MICRODADOS_CADASTRO_CURSOS_{ano}.csv'
        caminho_csv = pasta_data_02_staging_microdata_inep / nome_arquivo

        if caminho_csv.exists():
            print(f"  [*] Lendo e filtrando: {nome_arquivo}")
            try:

                # O Censo usa o padrão Windows (latin-1) e separador ponto e vírgula
                df_externo_cine = pd.read_csv(
                    str(caminho_csv),
                    encoding='latin-1',
                    sep=';',
                    low_memory=False,
                    decimal=',',
                    usecols=colunas_cine
                )
                
                lista_dfs.append(df_externo_cine)
                arquivos_processados += 1
                
            except Exception as e:
                print(f"    [!] ERRO ao processar '{nome_arquivo}': {e}")
        else:
            print(f"  [AVISO] Arquivo não encontrado: {nome_arquivo}. Pulando.")

    # --- 4. UNIFICAÇÃO E SALVAMENTO (CHECKPOINT EM PARQUET) ---
    if lista_dfs:
        print("\n[*] Empilhando os Censos num único Dataset Mestre...")
        df_concat = pd.concat(lista_dfs, ignore_index=True)
        
        nome_arquivo_saida = 'df_mestre_cadastro_cursos_2016_2024.parquet'
        caminho_saida = pasta_data_03_transform_inep / nome_arquivo_saida
        
        # Salva em Parquet para garantir que os tipos (especialmente códigos com zeros à esquerda) não quebrem no Merge final
        df_concat.to_parquet(str(caminho_saida), index=False)
        
        print(f"\n--- Processo Concluído! ---")
        print(f"Total de {arquivos_processados} arquivos unificados do INEP.")
        print(f"Dataset Mestre salvo em: {caminho_saida}")
        print(f"Total de linhas na tabela dimensional: {len(df_concat)}")
    else:
        print("\n[!] Nenhum arquivo do INEP foi processado. Verifique a pasta 02_staging.")










def validar_qualidade_inscritos():
    print("\n" + "="*50)
    print(" INICIANDO VALIDAÇÃO DE QUALIDADE: INSCRITOS")
    print("="*50)

    # Configurações do Pandas para visualização no Jupyter
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None) 

    # Fallback inteligente: se rodar no terminal, usa print. Se rodar no Jupyter, usa o display real.
    try:
        from IPython.display import display
    except ImportError:
        display = print

 
    arquivo_transform = pasta_raiz_projeto / 'data' / '03_transform' / 'fies' / 'fies_inscritos_unificado.parquet'

    # Definição das Chaves Primárias (PKs) em cada etapa
    pk_bruto = ['ID do estudante', 'Opções de cursos da inscrição']
    pk_transform = ['id_estudante_inscricao', 'opcoes_cursos_inscricao_inscricao']

    def analisar_df(df, nome_etapa, chaves_pk):
        print(f"\n" + "-"*40)
        print(f"📊 Análise da Etapa: {nome_etapa}")
        print("-"*40)
        
        qtde_linhas_antes = df.shape[0]

        # Verifica se as chaves existem no DF antes de agrupar
        chaves_validas = [col for col in chaves_pk if col in df.columns]
        if len(chaves_validas) == len(chaves_pk):
            df_agrupando_por_pk = df.groupby(by=chaves_validas, as_index=False).count()
            qtde_linhas_depois = df_agrupando_por_pk.shape[0]

            print(f'Total de linhas reais do dataset : {qtde_linhas_antes}')
            print(f'Total de chaves primárias únicas : {qtde_linhas_depois}')

            if qtde_linhas_antes != qtde_linhas_depois:
                print('\n⚠️ ERRO PROVÁVEL: DUPLICAÇÃO DE LINHA ENCONTRADA!')
                linhas_duplicadas = df.duplicated(keep='first').sum()
                print(f'Total de linhas 100% duplicadas: {linhas_duplicadas}')

                if linhas_duplicadas == 0:
                    print('\n⚠️ ERRO PROVÁVEL: LINHA COMPLETAMENTE VAZIA:')
                    df_sem_vazia = df.dropna(how='all')
                    print(f'Total de linhas após remover vazias: {df_sem_vazia.shape[0]}')
            else:
                print('✅ Nenhuma duplicação de PK encontrada. Integridade perfeita.')
        else:
            print(f"⚠️ Aviso: As chaves {chaves_pk} não foram encontradas neste dataset.")

        print(f"\n[ VISUALIZAÇÃO DO DATASET - {nome_etapa} ]")
        # Exibimos apenas as 5 primeiras linhas para não travar o navegador/terminal com milhões de registros
        display(df.head()) 
        
        print(f"\n[ ESTATÍSTICAS DESCRITIVAS - {nome_etapa} ]")
        display(df.describe())


    # --- 1. LENDO E ANALISANDO A CAMADA RAW ---
    print("\n⏳ Carregando dados da camada 01_RAW...")
    arquivos_raw = list(pasta_data_01_raw_microdata_fies.glob('*inscricao*.csv'))
    if arquivos_raw:
        # Lê lidando com a formatação brasileira do governo
        df_raw = pd.concat([pd.read_csv(str(f), sep=';', encoding='latin-1', decimal=',', low_memory=False) for f in arquivos_raw], ignore_index=True)
        analisar_df(df_raw, "01_RAW (Bruto Original)", pk_bruto)

    # --- 2. LENDO E ANALISANDO A CAMADA STAGING ---
    print("\n⏳ Carregando dados da camada 02_STAGING...")
    arquivos_staging = list(pasta_data_02_staging_microdata_fies.glob('*inscricao*.csv'))
    if arquivos_staging:
        # No staging já está limpo em UTF-8 e separado por vírgula
        df_staging = pd.concat([pd.read_csv(str(f), low_memory=False) for f in arquivos_staging], ignore_index=True)
        analisar_df(df_staging, "02_STAGING (Sem Duplicatas)", pk_bruto)

    # --- 3. LENDO E ANALISANDO A CAMADA TRANSFORM ---
    print("\n⏳ Carregando dados da camada 03_TRANSFORM...")
    if arquivo_transform.exists():
        df_transform = pd.read_parquet(str(arquivo_transform))
        analisar_df(df_transform, "03_TRANSFORM (Parquet Unificado)", pk_transform)
    else:
        print(f"⚠️ Arquivo {arquivo_transform.name} não encontrado.")

    print("\n" + "="*50)
    print("✅ VALIDAÇÃO DE QUALIDADE CONCLUÍDA")
    print("="*50)










def validar_qualidade_ofertas():
    print("\n" + "="*50)
    print("🔍 INICIANDO VALIDAÇÃO DE QUALIDADE: OFERTAS")
    print("="*50)

    # Configurações do Pandas para visualização no Jupyter
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None) 

    # Fallback inteligente para o display
    try:
        from IPython.display import display
    except ImportError:
        display = print



    arquivo_transform = pasta_raiz_projeto / 'data' / '03_transform' / 'fies' / 'fies_ofertas_unificado.parquet'

    # Definição das Chaves Primárias (PKs) em cada etapa
    pk_bruto = [
        'Código e-MEC da Mantenedora', 
        'Código do Local de Oferta', 
        'Código do Grupo de Preferência', 
        'Código do Curso', 
        'Turno'
    ]
    
    pk_transform = [
        'codigo_e_mec_mantenedora_ofertas',
        'codigo_local_oferta_ofertas',
        'codigo_grupo_preferencia_ofertas',
        'codigo_curso_ofertas',
        'turno_ofertas'
    ]

    def analisar_df(df, nome_etapa, chaves_pk):
        print(f"\n" + "-"*40)
        print(f"📊 Análise da Etapa: {nome_etapa}")
        print("-"*40)
        
        qtde_linhas_antes = df.shape[0]

        # Verifica se as chaves existem no DF antes de agrupar
        chaves_validas = [col for col in chaves_pk if col in df.columns]
        if len(chaves_validas) == len(chaves_pk):
            df_agrupando_por_pk = df.groupby(by=chaves_validas, as_index=False).count()
            qtde_linhas_depois = df_agrupando_por_pk.shape[0]

            print(f'Total de linhas reais do dataset : {qtde_linhas_antes}')
            print(f'Total de chaves primárias únicas : {qtde_linhas_depois}')

            if qtde_linhas_antes != qtde_linhas_depois:
                print('\n⚠️ ERRO PROVÁVEL: DUPLICAÇÃO DE LINHA ENCONTRADA!')
                linhas_duplicadas = df.duplicated(keep='first').sum()
                print(f'Total de linhas 100% duplicadas: {linhas_duplicadas}')

                if linhas_duplicadas == 0:
                    print('\n⚠️ ERRO PROVÁVEL: LINHA COMPLETAMENTE VAZIA:')
                    df_sem_vazia = df.dropna(how='all')
                    print(f'Total de linhas após remover vazias: {df_sem_vazia.shape[0]}')
            else:
                print('✅ Nenhuma duplicação de PK encontrada. Integridade perfeita.')
        else:
            print(f"⚠️ Aviso: As chaves {chaves_pk} não foram encontradas neste dataset. Colunas disponíveis: {df.columns.tolist()[:10]}...")

        print(f"\n[ VISUALIZAÇÃO DO DATASET - {nome_etapa} ]")
        display(df.head()) 
        
        print(f"\n[ ESTATÍSTICAS DESCRITIVAS - {nome_etapa} ]")
        display(df.describe())


    # --- 1. LENDO E ANALISANDO A CAMADA RAW ---
    print("\n⏳ Carregando dados da camada 01_RAW...")
    arquivos_raw = list(pasta_data_01_raw_microdata_fies.glob('*ofertas*.csv'))
    if arquivos_raw:
        df_raw = pd.concat([pd.read_csv(str(f), sep=';', encoding='latin-1', decimal=',', low_memory=False) for f in arquivos_raw], ignore_index=True)
        analisar_df(df_raw, "01_RAW (Bruto Original)", pk_bruto)

    # --- 2. LENDO E ANALISANDO A CAMADA STAGING ---
    print("\n⏳ Carregando dados da camada 02_STAGING...")
    arquivos_staging = list(pasta_data_02_staging_microdata_fies.glob('*ofertas*.csv'))
    if arquivos_staging:
        df_staging = pd.concat([pd.read_csv(str(f), low_memory=False) for f in arquivos_staging], ignore_index=True)
        analisar_df(df_staging, "02_STAGING (Sem Duplicatas)", pk_bruto)

    # --- 3. LENDO E ANALISANDO A CAMADA TRANSFORM ---
    print("\n⏳ Carregando dados da camada 03_TRANSFORM...")
    if arquivo_transform.exists():
        df_transform = pd.read_parquet(str(arquivo_transform))
        analisar_df(df_transform, "03_TRANSFORM (Parquet Unificado)", pk_transform)
    else:
        print(f"⚠️ Arquivo {arquivo_transform.name} não encontrado.")

    print("\n" + "="*50)
    print("✅ VALIDAÇÃO DE QUALIDADE CONCLUÍDA")
    print("="*50)











def transform_ofertas():
    print("\n" + "="*50)
    print("Iniciando Transformação e Unificação: FIES Ofertas")
    print("="*50)

 
    caminho_mestre_inep = pasta_raiz_projeto / 'data' / '03_transform' / 'inep' / 'df_mestre_cadastro_cursos_2016_2024.parquet'


    # --- 2. MAPA DE PADRONIZAÇÃO ---
    MAPA_RENOMEAR_OFERTAS = {
        'Ano': 'ano',
        'Semestre': 'semestre',
        'Nome Mantenedora': 'nome_mantenedora',
        'Código e-MEC da Mantenedora': 'codigo_e_mec_mantenedora',
        'CNPJ da mantenedora': 'cnpj_mantenedora',
        'Nome da IES': 'nome_ies',
        'Código e-MEC da IES': 'codigo_e_mec_ies',
        'Organização Acadêmica da IES': 'organizacao_academica_ies',
        'UF da IES': 'uf_ies',
        'Município da IES': 'municipio_ies',
        'Nome do Local de oferta': 'nome_local_oferta',
        'Código do Local de Oferta': 'codigo_local_oferta',
        'Município do Local de Oferta': 'municipio_local_oferta',
        'UF do Local de Oferta': 'uf_local_oferta',
        'Nome da Microrregião': 'nome_microrregiao',
        'Código da Microrregião': 'codigo_microrregiao',
        'Código da Mesorregião': 'codigo_mesorregiao',
        'Nome da Mesorregião': 'nome_mesorregiao',
        'Área do conhecimento': 'area_conhecimento',
        'Subárea do conhecimento': 'subarea_conhecimento',
        'Código do Grupo de Preferência': 'codigo_grupo_preferencia',
        'Nota de Corte Grupo Preferência': 'nota_corte_grupo_preferencia',
        'Código do Curso': 'codigo_curso',
        'Nome do Curso': 'nome_curso',
        'Turno': 'turno',
        'Grau': 'grau',
        'Conceito': 'conceito',
        'Vagas autorizadas e-mec': 'vagas_autorizadas_e_mec',
        'Vagas ofertadas FIES': 'vagas_ofertadas_fies',
        'Vagas além da Oferta': 'vagas_alem_da_oferta',
        'Vagas ocupadas': 'vagas_ocupadas',
        'Participa do P-FIES': 'participa_p_fies',
        'Vagas Ofertadas P-FIES': 'vagas_ofertadas_p_fies',
        'BANCO NORDESTE BRASIL (004)': 'banco_nordeste_brasil_004', 
        'ITAU UNIBANCO (PRAVALER)(341)': 'itau_unibanco_pravaler_341',
        'BV FINANCEIRA (PRAVALER)(455)': 'bv_financeira_pravaler_455',
        'BANCO ANDBANK (PRAVALER)(65)': 'banco_andbank_pravaler_65',
        'BANCO DA AMAZONIA S.A. (003)': 'banco_amazonia_sa_003', 
        'Valor bruto do curso': 'valor_bruto_curso',
        'Índice de correção - IPCA': 'indice_correcao_ipca',
        'Valor do curso para FIES': 'valor_curso_fies', 
        'Valor do curso para o FIES': 'valor_curso_fies', 
        **{f'{i} Semestre Bruto': f'semestre_{i}_bruto' for i in range(1, 13)},
        **{f'{i} Semestre FIES': f'semestre_{i}_fies' for i in range(1, 13)},
    }
    # --- 3. LOOP DE PROCESSAMENTO ---
    arquivos_ofertas = list(pasta_data_02_staging_microdata_fies.glob('*ofertas*.csv'))
    
    if not arquivos_ofertas:
        print(f"[AVISO] Nenhum arquivo de ofertas encontrado em {pasta_data_02_staging_microdata_fies}")
        return

    lista_dfs = []
    arquivos_processados = 0

    print(f"Buscando arquivos em: {pasta_data_02_staging_microdata_fies}")
    
    for caminho_csv in arquivos_ofertas:
        print(f"  [*] Lendo e padronizando: {caminho_csv.name}")
        try:
            df_temp = pd.read_csv(str(caminho_csv), low_memory=False) 
            
            novas_colunas = []
            for col in df_temp.columns:
                col_limpa = str(col).strip() 
                col_limpa = re.sub(r'\s+', ' ', col_limpa) 
                novas_colunas.append(col_limpa)
            df_temp.columns = novas_colunas 
            
            colunas_para_manter = [c for c in df_temp.columns if 'unnamed' not in c.lower()]
            df_temp = df_temp[colunas_para_manter] 
            
            df_temp.rename(columns=MAPA_RENOMEAR_OFERTAS, inplace=True)
            df_temp.columns = [f"{col}_ofertas" for col in df_temp.columns]
            
            lista_dfs.append(df_temp)
            arquivos_processados += 1

        except pd.errors.EmptyDataError:
            print(f"    [AVISO] O arquivo '{caminho_csv.name}' está vazio e foi ignorado.")
        except Exception as e:
            print(f"    [!] ERRO ao processar '{caminho_csv.name}': {e}")

    # --- 4. UNIFICAÇÃO E ORDENAÇÃO ---
    if lista_dfs:
        print("\n[*] Empilhando todos os semestres num único Dataset...")
        df_concat = pd.concat(lista_dfs, ignore_index=True)
        
        print("[*] Ordenando o Dataset por Ano e Semestre...")
        ordem_colunas = ['ano_ofertas', 'semestre_ofertas']
        df_final = df_concat.sort_values(by=ordem_colunas)

        # BLINDAGEM: Padroniza Turno e P-FIES nas ofertas para garantir Merge futuro (Layer 3)
        if 'turno_ofertas' in df_final.columns:
            df_final['turno_ofertas'] = df_final['turno_ofertas'].astype(str).str.strip().str.upper()
        if 'participa_p_fies_ofertas' in df_final.columns:
            df_final['participa_p_fies_ofertas'] = df_final['participa_p_fies_ofertas'].astype(str).str.strip().str.upper()

        # --- 5. ENRIQUECIMENTO (MERGE COM INEP) ---
        print("\n[*] Iniciando Merge com o Dataset Mestre do INEP (Áreas CINE)...")
        if caminho_mestre_inep.exists():
            df_mestre = pd.read_parquet(str(caminho_mestre_inep))
            colunas_cine = ['NO_CURSO', 'CO_CURSO', 'CO_CINE_AREA_GERAL', 'NO_CINE_AREA_GERAL']
            
            df_mestre_dedup = df_mestre.sort_values(by='NU_ANO_CENSO', ascending=True)
            df_mestre_dedup = df_mestre_dedup.drop_duplicates(subset=['CO_CURSO'], keep='last')
            df_mestre_dedup = df_mestre_dedup[colunas_cine]

            # BLINDAGEM DO MERGE: Força AMBOS OS LADOS a serem numéricos exatos
            df_mestre_dedup['CO_CURSO'] = pd.to_numeric(df_mestre_dedup['CO_CURSO'], errors='coerce')
            df_final['codigo_curso_ofertas'] = pd.to_numeric(df_final['codigo_curso_ofertas'], errors='coerce')

            df_final = pd.merge(
                df_final,
                df_mestre_dedup,
                how='left',
                left_on='codigo_curso_ofertas',
                right_on='CO_CURSO',
                suffixes=['', '_cine']
            )
            print("  -> Merge concluído com sucesso! Colunas do INEP adicionadas.")
        else:
            print(f"  [!] AVISO: Dataset mestre não encontrado em {caminho_mestre_inep}.")

        # --- 6. SALVAMENTO (CHECKPOINT EM PARQUET) E VALIDAÇÃO ---
        nome_arquivo_saida = 'fies_ofertas_unificado.parquet'
        caminho_saida = pasta_data_03_transform_fies / nome_arquivo_saida
        
        df_final.to_parquet(str(caminho_saida), index=False)
        
        print(f"\n--- Processo Concluído! ---")
        print(f"Total de {arquivos_processados} arquivos unificados e enriquecidos.")
        print(f"Checkpoint salvo em: {caminho_saida}")
        print(f"Total de linhasx colunas no dataset final: {df_final.shape}")
        
        validar_qualidade_ofertas()











def transform_inscritos():
    print("\n" + "="*50)
    print("Iniciando Transformação e Unificação: FIES Inscritos")
    print("="*50)

    caminho_mestre_inep = pasta_raiz_projeto / 'data' / '03_transform' / 'inep' / 'df_mestre_cadastro_cursos_2016_2024.parquet'

    # --- 2. MAPA DE PADRONIZAÇÃO ---
    MAPA_RENOMEAR_INSCRICAO = {
        'Ano do processo seletivo': 'ano_processo_seletivo',
        'Semestre do processo seletivo': 'semestre_processo_seletivo',
        'Cod. do Grupo de preferência': 'codigo_grupo_preferencia',
        'Classificação': 'classificacao',
        'ID do estudante': 'id_estudante',
        'Sexo': 'sexo',
        'Data de Nascimento': 'data_nascimento',
        'UF de residência': 'uf_residencia',
        'Municipio de residência': 'municipio_residencia',
        'Etnia/Cor': 'etnia_cor',
        'Pessoa com deficiência?': 'pessoa_com_deficiencia',
        'Concluiu ensino médio escola pública': 'concluiu_ensino_medio_escola_publica',
        'Ano conclusão ensino médio': 'ano_conclusao_ensino_medio',
        'Concluiu curso superior?': 'concluiu_curso_superior',
        'Beneficiado pelo Creduc ou Fies': 'beneficiado_creduc_ou_fies',
        'Professor rede pública ensino?': 'professor_rede_publica_ensino',
        'Nº de membros Grupo Familiar': 'numero_membros_grupo_familiar',
        'Renda familiar mensal bruta': 'renda_familiar_mensal_bruta',
        'Renda mensal bruta per capita': 'renda_mensal_bruta_per_capita',
        'Região grupo de preferência': 'regiao_grupo_preferencia',
        'UF': 'uf_grupo_preferencia',
        'Cod.Microrregião': 'codigo_microrregiao',
        'Microrregião': 'microrregiao',
        'Cod.Mesorregião': 'codigo_mesorregiao',
        'Mesorregião': 'mesorregiao',
        'Conceito de curso do GP': 'conceito_curso_gp',
        'Área do conhecimento': 'area_conhecimento',
        'Subárea do conhecimento': 'subarea_conhecimento',
        'Nota Corte Grupo Preferência': 'nota_corte_grupo_preferencia',
        'Opções de cursos da inscrição': 'opcoes_cursos_inscricao',
        'Nome mantenedora': 'nome_mantenedora',
        'Natureza Jurídica Mantenedora': 'natureza_juridica_mantenedora',
        'CNPJ da mantenedora': 'cnpj_mantenedora',
        'Código e-MEC da Mantenedora': 'codigo_e_mec_mantenedora',
        'Nome da IES': 'nome_ies',
        'Código e-MEC da IES': 'codigo_e_mec_ies',
        'Organização Acadêmica da IES': 'organizacao_academica_ies',
        'Município da IES': 'municipio_ies',
        'UF da IES': 'uf_ies',
        'Nome do Local de oferta': 'nome_local_oferta',
        'Código do Local de Oferta': 'codigo_local_oferta',
        'Munícipio do Local de Oferta': 'municipio_local_oferta',
        'UF do Local de Oferta': 'uf_local_oferta',
        'Código do curso': 'codigo_curso',
        'Nome do curso': 'nome_curso',
        'Turno': 'turno',
        'Grau': 'grau',
        'Conceito': 'conceito_curso',
        'Média nota Enem': 'media_nota_enem',
        'Ano do Enem': 'ano_enem',
        'Redação': 'nota_redacao',
        'Matemática e suas Tecnologias': 'nota_matematica',
        'Linguagens, Códigos e suas Tec': 'nota_linguagens',
        'Ciências Natureza e suas Tec': 'nota_ciencias_natureza',
        'Ciências Humanas e suas Tec': 'nota_ciencias_humanas',
        'Situação Inscrição Fies': 'situacao_inscricao_fies',
        'Percentual de financiamento': 'percentual_financiamento',
        'Semestre do financiamento': 'semestre_financiamento',
        'Qtde semestre financiado': 'qtde_semestre_financiado'
    }

    # --- 3. LOOP DE PROCESSAMENTO ---
    arquivos_inscricao = list(pasta_data_02_staging_microdata_fies.glob('*inscricao*.csv'))
    
    if not arquivos_inscricao:
        print(f"[AVISO] Nenhum arquivo de inscrição encontrado em {pasta_data_02_staging_microdata_fies}")
        return

    lista_dfs = []
    arquivos_processados = 0

    print(f"Buscando arquivos em: {pasta_data_02_staging_microdata_fies}")
    
    for caminho_csv in arquivos_inscricao:
        print(f"  [*] Lendo e padronizando: {caminho_csv.name}")
        try:
            df_temp = pd.read_csv(str(caminho_csv), low_memory=False) 
            
            novas_colunas = []
            for col in df_temp.columns:
                col_limpa = str(col).strip() 
                col_limpa = re.sub(r'\s+', ' ', col_limpa) 
                novas_colunas.append(col_limpa)
            df_temp.columns = novas_colunas 
            
            colunas_para_manter = [c for c in df_temp.columns if 'unnamed' not in c.lower()]
            df_temp = df_temp[colunas_para_manter] 
            
            df_temp.rename(columns=MAPA_RENOMEAR_INSCRICAO, inplace=True)
            df_temp.columns = [f"{col}_inscricao" for col in df_temp.columns]
            
            lista_dfs.append(df_temp)
            arquivos_processados += 1

        except pd.errors.EmptyDataError:
            print(f"    [AVISO] O arquivo '{caminho_csv.name}' está vazio e foi ignorado.")
        except Exception as e:
            print(f"    [!] ERRO ao processar '{caminho_csv.name}': {e}")

    # --- 4. UNIFICAÇÃO E ORDENAÇÃO ---
    if lista_dfs:
        print("\n[*] Empilhando todos os semestres num único Dataset...")
        df_concat = pd.concat(lista_dfs, ignore_index=True)
        
        print("[*] Ordenando o Dataset por Ano e Semestre...")
        ordem_colunas = ['ano_processo_seletivo_inscricao', 'semestre_processo_seletivo_inscricao']
        df_final = df_concat.sort_values(by=ordem_colunas)

        # BLINDAGEM: Padroniza Turno nas inscrições
        if 'turno_inscricao' in df_final.columns:
            df_final['turno_inscricao'] = df_final['turno_inscricao'].astype(str).str.strip().str.upper()

        # --- 5. ENRIQUECIMENTO (MERGE COM INEP) ---
        print("\n[*] Iniciando Merge com o Dataset Mestre do INEP (Áreas CINE)...")
        if caminho_mestre_inep.exists():
            df_mestre = pd.read_parquet(str(caminho_mestre_inep))
            colunas_cine = ['NO_CURSO', 'CO_CURSO', 'CO_CINE_AREA_GERAL', 'NO_CINE_AREA_GERAL']
            
            df_mestre_dedup = df_mestre.sort_values(by='NU_ANO_CENSO', ascending=True)
            df_mestre_dedup = df_mestre_dedup.drop_duplicates(subset=['CO_CURSO'], keep='last')
            df_mestre_dedup = df_mestre_dedup[colunas_cine]

            # BLINDAGEM DO MERGE: Força AMBOS OS LADOS a serem numéricos exatos
            df_mestre_dedup['CO_CURSO'] = pd.to_numeric(df_mestre_dedup['CO_CURSO'], errors='coerce')
            df_final['codigo_curso_inscricao'] = pd.to_numeric(df_final['codigo_curso_inscricao'], errors='coerce')

            df_final = pd.merge(
                df_final,
                df_mestre_dedup,
                how='left',
                left_on='codigo_curso_inscricao',
                right_on='CO_CURSO',
                suffixes=['', '_cine']
            )
            print("  -> Merge concluído com sucesso! Colunas do INEP adicionadas.")
        else:
            print(f"  [!] AVISO: Dataset mestre não encontrado em {caminho_mestre_inep}.")

        # --- 5.5 LIMPEZA DE TIPOS (Evita erro de conversão do PyArrow) ---
        print("[*] Padronizando tipos numéricos para o Parquet...")
        colunas_conversaao = [
            'ano_enem_inscricao', 
            'ano_conclusao_ensino_medio_inscricao',
            'media_nota_enem_inscricao',
            'nota_redacao_inscricao',
            'nota_matematica_inscricao',
            'nota_linguagens_inscricao',
            'nota_ciencias_natureza_inscricao',
            'nota_ciencias_humanas_inscricao'
        ]

        for col in colunas_conversaao:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

        colunas_anos = ['ano_enem_inscricao', 'ano_conclusao_ensino_medio_inscricao']
        for col in colunas_anos:
            if col in df_final.columns:
                df_final[col] = df_final[col].astype('Int64')

        # --- 6. SALVAMENTO (CHECKPOINT EM PARQUET) E VALIDAÇÃO ---
        nome_arquivo_saida = 'fies_inscritos_unificado.parquet'
        caminho_saida = pasta_data_03_transform_fies / nome_arquivo_saida
        
        df_final.to_parquet(str(caminho_saida), index=False)
        
        print(f"\n--- Processo Concluído! ---")
        print(f"Total de {arquivos_processados} arquivos unificados e enriquecidos.")
        print(f"Checkpoint salvo em: {caminho_saida}")
        print(f"Total de linhas no dataset final: {len(df_final)}")
        
        validar_qualidade_inscritos()
















def auditoria_cine_inscritos():
    print("\n" + "="*80)
    print("📊 AUDITORIA CINE E NaNs: FIES INSCRITOS")
    print("="*80)

    # Configuração de Caminhos
    arquivo_parquet = pasta_raiz_projeto / 'data' / '03_transform' / 'fies' / 'fies_inscritos_unificado.parquet'
    coluna_para_analisar = 'NO_CINE_AREA_GERAL' 

    if not arquivo_parquet.exists():
        print(f"ERRO: O arquivo unificado não existe: {arquivo_parquet}")
        return

    # Lê o dataset unificado
    df = pd.read_parquet(str(arquivo_parquet))

    if coluna_para_analisar not in df.columns:
        print(f"ERRO: A coluna '{coluna_para_analisar}' não foi encontrada!")
        return

    col_ano = 'ano_processo_seletivo_inscricao'
    col_semestre = 'semestre_processo_seletivo_inscricao'

    # Agrupa por ano e semestre para mostrar a análise individualizada de cada período
    grupos = df.groupby([col_ano, col_semestre])

    for (ano, semestre), df_temp in grupos:
        print(f"\n{'-'*80}")
        print(f"--- Iniciando Análise do Período: Ano {ano} | Semestre {semestre} ---")
        print(f"{'-'*80}")

        print(f"\n  Contagem de valores para '{coluna_para_analisar}':")
        contagem = df_temp[coluna_para_analisar].value_counts(dropna=False)
        print(contagem)

        print("\n  --- Resumo dos NaNs (para este período) ---")
        total_linhas = len(df_temp)
        total_nans = df_temp[coluna_para_analisar].isnull().sum()

        if total_linhas > 0:
            percentual_nans = (total_nans / total_linhas) * 100
            print(f"  Total de inscritos (linhas): {total_linhas}")
            print(f"  Total de NaNs (cursos não encontrados): {total_nans}")
            print(f"  Percentual de NaNs: {percentual_nans:.2f}%")
        else:
            print("  Sem dados para este período.")

    # Resumo Geral Final
    print(f"\n{'='*80}")
    print("--- RESUMO GERAL (TODOS OS ANOS) ---")
    total_geral = len(df)
    nans_geral = df[coluna_para_analisar].isnull().sum()
    print(f"Total Geral de Inscritos: {total_geral}")
    print(f"Total Geral de NaNs: {nans_geral} ({(nans_geral/total_geral)*100:.2f}%)")
    print(f"{'='*80}\n")










def auditoria_cine_ofertas():
    print("\n" + "="*80)
    print("📊 AUDITORIA CINE E NaNs: FIES OFERTAS")
    print("="*80)

    # Configuração de Caminhos
    arquivo_parquet = pasta_raiz_projeto / 'data' / '03_transform' / 'fies' / 'fies_ofertas_unificado.parquet'
    coluna_para_analisar = 'NO_CINE_AREA_GERAL' 

    if not arquivo_parquet.exists():
        print(f"ERRO: O arquivo unificado não existe: {arquivo_parquet}")
        return

    # Lê o dataset unificado
    df = pd.read_parquet(str(arquivo_parquet))

    if coluna_para_analisar not in df.columns:
        print(f"ERRO: A coluna '{coluna_para_analisar}' não foi encontrada!")
        return

    col_ano = 'ano_ofertas'
    col_semestre = 'semestre_ofertas'

    # Agrupa por ano e semestre
    grupos = df.groupby([col_ano, col_semestre])

    for (ano, semestre), df_temp in grupos:
        print(f"\n{'-'*80}")
        print(f"--- Iniciando Análise do Período: Ano {ano} | Semestre {semestre} ---")
        print(f"{'-'*80}")

        print(f"\n  Contagem de valores para '{coluna_para_analisar}':")
        contagem = df_temp[coluna_para_analisar].value_counts(dropna=False)
        print(contagem)

        print("\n  --- Resumo dos NaNs (para este período) ---")
        total_linhas = len(df_temp)
        total_nans = df_temp[coluna_para_analisar].isnull().sum()

        if total_linhas > 0:
            percentual_nans = (total_nans / total_linhas) * 100
            print(f"  Total de ofertas (linhas): {total_linhas}")
            print(f"  Total de NaNs (cursos não encontrados): {total_nans}")
            print(f"  Percentual de NaNs: {percentual_nans:.2f}%")
        else:
            print("  Sem dados para este período.")

    # Resumo Geral Final
    print(f"\n{'='*80}")
    print("--- RESUMO GERAL (TODOS OS ANOS) ---")
    total_geral = len(df)
    nans_geral = df[coluna_para_analisar].isnull().sum()
    print(f"Total Geral de Ofertas: {total_geral}")
    print(f"Total Geral de NaNs: {nans_geral} ({(nans_geral/total_geral)*100:.2f}%)")
    print(f"{'='*80}\n")















def verificar_colunas_inep():
    print("\n" + "="*50)
    print("🕵️ AUDITORIA DE METADADOS: INEP (Censo Superior)")
    print("="*50)



    # Dicionário para guardar o conjunto (set) de colunas de cada ano
    colunas_por_ano = {}

    print(f"Buscando arquivos na Staging: {pasta_data_02_staging_microdata_inep}\n")

    # --- ETAPA 1: Ler os cabeçalhos de cada arquivo ---
    for ano in range(2016, 2025): # 2016 até 2024
        nome_arquivo = f'MICRODADOS_CADASTRO_CURSOS_{ano}.csv'
        caminho_csv = pasta_data_02_staging_microdata_inep / nome_arquivo
        
        print(f"--- Processando {ano} ---")
        if caminho_csv.exists():
            try:
                # Usamos nrows=0 para ler APENAS o cabeçalho.
                # É instantâneo e não consome memória.
                df_header = pd.read_csv(
                    str(caminho_csv),
                    encoding='latin-1',
                    sep=';',
                    decimal=',',
                    nrows=0 # Lê só a linha 0 (nomes das colunas)
                )
                
                # Armazena as colunas como um 'set' para facilitar a comparação matemática
                colunas_por_ano[ano] = set(df_header.columns)
                print(f"  > OK: Arquivo {ano} lido. Total: {len(df_header.columns)} colunas.")
                
            except Exception as e:
                print(f"  [!] ERRO ao ler o arquivo {ano}: {e}")
                colunas_por_ano[ano] = None
        else:
            print(f"  [AVISO] Arquivo não encontrado: {nome_arquivo}")
            colunas_por_ano[ano] = None

    print("\n--- Leitura de Cabeçalhos Concluída ---")

    # --- ETAPA 2: Comparar os conjuntos de colunas ---
    anos_validos = [ano for ano, colunas in colunas_por_ano.items() if colunas is not None]

    if len(anos_validos) >= 2:
        print("\n--- 🔍 Comparando Esquemas de Colunas ---")
        
        # Pega o primeiro ano lido com sucesso como nossa referência (baseline)
        ano_referencia = anos_validos[0]
        colunas_referencia = colunas_por_ano[ano_referencia]
        
        print(f"Referência: {ano_referencia} ({len(colunas_referencia)} colunas).")
        
        # Compara os outros anos com a referência
        for i in range(1, len(anos_validos)):
            ano_comparar = anos_validos[i]
            colunas_comparar = colunas_por_ano[ano_comparar]
            
            print(f"\nComparando {ano_referencia} vs {ano_comparar}:")
            
            # Colunas que estão na referência mas não no outro
            diferenca_1 = colunas_referencia - colunas_comparar
            if diferenca_1:
                print(f"  [!] Colunas em {ano_referencia} que NÃO estão em {ano_comparar}: {diferenca_1}")
            
            # Colunas que estão no outro mas não na referência
            diferenca_2 = colunas_comparar - colunas_referencia
            if diferenca_2:
                print(f"  [!] Colunas em {ano_comparar} que NÃO estão em {ano_referencia}: {diferenca_2}")
            
            if not diferenca_1 and not diferenca_2:
                print("  ✅ Os conjuntos de colunas são EXATAMENTE idênticos.")
    else:
        print("\n[!] Não foi possível comparar os arquivos (menos de 2 arquivos lidos com sucesso).")

    # --- ETAPA 3: Listagem completa (para depuração) ---
    print("\n\n--- 📜 Listagem Bruta das Colunas ---")
    for ano, colunas in colunas_por_ano.items():
        if colunas is not None:
            print(f"\n====================\nCOLUNAS {ano}\n====================")
            # Imprime uma por linha, em ordem alfabética, para facilitar a leitura
            for col in sorted(list(colunas)):
                print(col)
        else:
            print(f"\n====================\nCOLUNAS {ano}\n====================\n(Falha na leitura)")
    auditoria_cine_inscritos()
    auditoria_cine_ofertas()

