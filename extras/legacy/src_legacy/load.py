import pandas as pd
from src.constantes import pasta_data_03_transform_fies,pasta_data_01_raw_microdata_fies, pasta_data_02_staging_microdata_fies, pasta_data_04_load_database
import numpy as np
def load_inscritos():

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None) 

    print("\n" + "="*60)
    print("📦 INICIANDO LAYER 4 (LOAD): PADRONIZAÇÃO FINAL - INSCRITOS")
    print("="*60)

    # 1. Caminhos
    arquivo_entrada = pasta_data_03_transform_fies / 'fies_inscritos_com_modalidade_final.parquet'
    
    arquivo_saida_parquet = pasta_data_04_load_database / 'inscritos_final_limpo.parquet'
    arquivo_saida_csv = pasta_data_04_load_database / 'inscritos_final_limpo.csv' # Bônus: Salva em CSV para PowerBI se quiser

    if not arquivo_entrada.exists():
        print(f"[!] ERRO: Arquivo da Layer 3 não encontrado: {arquivo_entrada}")
        return

    # 2. O teu Mapa de Nomes Curto
    mapa_nomes_curto = {
        # Chaves de processo
        'ano_processo_seletivo_inscricao': 'ano',
        'semestre_processo_seletivo_inscricao': 'semestre',
        'codigo_grupo_preferencia_inscricao': 'codigo_grupo_preferencia',
        'classificacao_inscricao': 'classificacao',
        'opcoes_cursos_inscricao_inscricao': 'opcao_curso',
        
        # Dados do Estudante
        'id_estudante_inscricao': 'id_estudante',
        'sexo_inscricao': 'sexo',
        'data_nascimento_inscricao': 'data_nascimento',
        'uf_residencia_inscricao': 'uf_residencia',
        'municipio_residencia_inscricao': 'municipio_residencia',
        'etnia_cor_inscricao': 'etnia_cor',
        'pessoa_com_deficiencia_inscricao': 'pessoa_com_deficiencia',
        
        # Escolaridade
        'concluiu_ensino_medio_escola_publica_inscricao': 'ensino_medio_escola_publica',
        'ano_conclusao_ensino_medio_inscricao': 'ano_conclusao_em',
        'concluiu_curso_superior_inscricao': 'concluiu_curso_superior',
        'professor_rede_publica_ensino_inscricao': 'professor_rede_publica',
        
        # Renda e Grupo Familiar
        'numero_membros_grupo_familiar_inscricao': 'membros_grupo_familiar',
        'renda_familiar_mensal_bruta_inscricao': 'renda_familiar_bruta',
        'renda_mensal_bruta_per_capita_inscricao': 'renda_per_capita',
        
        # Detalhes da Vaga (Grupo de Preferência)
        'regiao_grupo_preferencia_inscricao': 'regiao_gp',
        'uf_grupo_preferencia_inscricao': 'uf_gp',
        'codigo_microrregiao_inscricao': 'codigo_microrregiao_gp',
        'microrregiao_inscricao': 'microrregiao_gp',
        'codigo_mesorregiao_inscricao': 'codigo_mesorregiao_gp',
        'mesorregiao_inscricao': 'mesorregiao_gp',
        'conceito_curso_gp_inscricao': 'conceito_curso_gp',
        'nota_corte_grupo_preferencia_inscricao': 'nota_corte_gp',
        
        # Detalhes da IES
        'nome_mantenedora_inscricao': 'nome_mantenedora',
        'natureza_juridica_mantenedora_inscricao': 'natureza_juridica_mantenedora',
        'cnpj_mantenedora_inscricao': 'cnpj_mantenedora',
        'codigo_e_mec_mantenedora_inscricao': 'codigo_mec_mantenedora',
        'nome_ies_inscricao': 'nome_ies',
        'codigo_e_mec_ies_inscricao': 'codigo_mec_ies',
        'organizacao_academica_ies_inscricao': 'organizacao_academica_ies',
        'municipio_ies_inscricao': 'municipio_ies',
        'uf_ies_inscricao': 'uf_ies',
        
        # Detalhes do Local de Oferta
        'nome_local_oferta_inscricao': 'nome_local_oferta',
        'codigo_local_oferta_inscricao': 'codigo_local_oferta',
        'municipio_local_oferta_inscricao': 'municipio_local_oferta',
        'uf_local_oferta_inscricao': 'uf_local_oferta',
        
        # Detalhes do Curso
        'codigo_curso_inscricao': 'codigo_curso',
        'nome_curso_inscricao': 'nome_curso',
        'turno_inscricao': 'turno',
        'grau_inscricao': 'grau',
        'conceito_curso_inscricao': 'conceito_curso',
        'area_conhecimento_inscricao': 'area_conhecimento',
        'subarea_conhecimento_inscricao': 'subarea_conhecimento',

        # Notas ENEM
        'media_nota_enem_inscricao': 'media_enem',
        'ano_enem_inscricao': 'ano_enem',
        'nota_redacao_inscricao': 'nota_redacao',
        'nota_matematica_inscricao': 'nota_matematica',
        'nota_linguagens_inscricao': 'nota_linguagens',
        'nota_ciencias_natureza_inscricao': 'nota_ciencias_natureza',
        'nota_ciencias_humanas_inscricao': 'nota_ciencias_humanas',
        
        # Financiamento
        'beneficiado_creduc_ou_fies_inscricao': 'beneficiado_creduc_fies',
        'situacao_inscricao_fies_inscricao': 'situacao_fies',
        'percentual_financiamento_inscricao': 'percentual_financiamento',
        'semestre_financiamento_inscricao': 'semestre_financiamento',
        'qtde_semestre_financiado_inscricao': 'qtde_semestre_financiado',
        
        # Colunas CINE
        'NO_CURSO': 'nome_curso_cine',
        'CO_CURSO': 'codigo_curso_cine',
        'CO_CINE_AREA_GERAL': 'codigo_cine_area_geral',
        'NO_CINE_AREA_GERAL': 'nome_cine_area_geral',
        
        # Nota: A coluna 'modalidade_fies' já está com o nome perfeito.
    }

    # 3. Processamento
    print(f"[*] Lendo dataset pós-peneira: {arquivo_entrada.name}")
    df = pd.read_parquet(str(arquivo_entrada))

    print("[*] Renomeando colunas para o padrão do Banco de Dados...")
    df.rename(columns=mapa_nomes_curto, inplace=True)


    # Mapeamento UF -> Região
    mapa_uf_regiao = {
        'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
        'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
        'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
        'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
        'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
    }

    print("[*] Tratando strings de UF e aplicando Fallback Inteligente (IES -> Local de Oferta)...")
    
    # Substitui os textos de lixo por nulo matemático real para a lógica de Fallback funcionar
    df['uf_ies'] = df['uf_ies'].replace({'NAN': np.nan, 'nan': np.nan, '': np.nan, 'None': np.nan})
    df['uf_local_oferta'] = df['uf_local_oferta'].replace({'NAN': np.nan, 'nan': np.nan, '': np.nan, 'None': np.nan})

    # A MÁGICA: Pega a UF da IES. Onde for nulo (como em 2019.2), preenche com a UF do Local de Oferta.
    df['uf_ies_corrigida'] = df['uf_ies'].combine_first(df['uf_local_oferta'])

    # Limpa espaços invisíveis e garante maiúsculas
    df['uf_residencia'] = df['uf_residencia'].astype(str).str.strip().str.upper()
    df['uf_ies_corrigida'] = df['uf_ies_corrigida'].astype(str).str.strip().str.upper()
    
    # Aplica o mapeamento da Região usando a coluna blindada
    df['regiao_morar'] = df['uf_residencia'].map(mapa_uf_regiao)
    df['regiao_ies_alvo'] = df['uf_ies_corrigida'].map(mapa_uf_regiao)

    print("[*] Garantindo que Ano e Semestre sejam números Inteiros...")
    df['ano'] = df['ano'].astype(int)
    df['semestre'] = df['semestre'].astype(int)

    # 4. Salvamento
    print(f"[*] Salvando versão final (Parquet) em: {pasta_data_04_load_database.name}/...")

    # 4. Salvamento
    print(f"[*] Salvando versão final (Parquet) em: {pasta_data_04_load_database.name}/...")
    df.to_parquet(str(arquivo_saida_parquet), index=False)
    
    # Opcional: Descomenta a linha abaixo se quiseres gerar um CSV limpo para ler no Excel/PowerBI
    # df.to_csv(str(arquivo_saida_csv), index=False, encoding='utf-8')

    print(f"\n[OK] Dataset de Inscritos PRONTO PARA ANÁLISE.")
    print(f"Colunas Finais: {len(df.columns)}")
    print("="*60)










def load_ofertas():
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None) 

    print("\n" + "="*60)
    print("📦 INICIANDO LAYER 4 (LOAD): PADRONIZAÇÃO FINAL - OFERTAS")
    print("="*60)

    # 1. Caminhos
    # Lê do ficheiro pós-correção de NaNs (Layer 2)
    arquivo_entrada = pasta_data_03_transform_fies / 'fies_ofertas_unificado_corrigido.parquet'
    
    pasta_data_04_load_database.mkdir(parents=True, exist_ok=True)
    arquivo_saida_parquet = pasta_data_04_load_database / 'ofertas_final_limpo.parquet'

    if not arquivo_entrada.exists():
        print(f"[!] ERRO: Arquivo de ofertas corrigido não encontrado: {arquivo_entrada}")
        return

    # 2. O teu Mapa de Nomes Curto para Ofertas
    mapa_nomes_curto = {
        'ano_ofertas': 'ano',
        'semestre_ofertas': 'semestre',
        'nome_mantenedora_ofertas': 'nome_mantenedora',
        'codigo_e_mec_mantenedora_ofertas': 'codigo_mec_mantenedora',
        'cnpj_mantenedora_ofertas': 'cnpj_mantenedora',
        'nome_ies_ofertas': 'nome_ies',
        'codigo_e_mec_ies_ofertas': 'codigo_mec_ies',
        'organizacao_academica_ies_ofertas': 'organizacao_academica_ies',
        'uf_ies_ofertas': 'uf_ies',
        'municipio_ies_ofertas': 'municipio_ies',
        'nome_local_oferta_ofertas': 'nome_local_oferta',
        'codigo_local_oferta_ofertas': 'codigo_local_oferta',
        'municipio_local_oferta_ofertas': 'municipio_local_oferta',
        'uf_local_oferta_ofertas': 'uf_local_oferta',
        'nome_microrregiao_ofertas': 'nome_microrregiao',
        'codigo_microrregiao_ofertas': 'codigo_microrregiao',
        'codigo_mesorregiao_ofertas': 'codigo_mesorregiao',
        'nome_mesorregiao_ofertas': 'nome_mesorregiao',
        'area_conhecimento_ofertas': 'area_conhecimento',
        'subarea_conhecimento_ofertas': 'subarea_conhecimento',
        'codigo_grupo_preferencia_ofertas': 'codigo_grupo_preferencia',
        'nota_corte_grupo_preferencia_ofertas': 'nota_corte_gp',
        'codigo_curso_ofertas': 'codigo_curso',
        'nome_curso_ofertas': 'nome_curso',
        'turno_ofertas': 'turno',
        'grau_ofertas': 'grau',
        'conceito_ofertas': 'conceito_curso',
        'vagas_autorizadas_e_mec_ofertas': 'vagas_autorizadas_mec',
        'vagas_ofertadas_fies_ofertas': 'vagas_fies',
        'vagas_alem_da_oferta_ofertas': 'vagas_alem_oferta',
        'vagas_ocupadas_ofertas': 'vagas_ocupadas',
        'participa_p_fies_ofertas': 'participa_p_fies',
        'vagas_ofertadas_p_fies_ofertas': 'vagas_p_fies',
        
        # Agentes Financeiros
        'banco_nordeste_brasil_004_ofertas': 'ag_banco_nordeste_004',
        'itau_unibanco_pravaler_341_ofertas': 'ag_itau_pravaler_341',
        'bv_financeira_pravaler_455_ofertas': 'ag_bv_pravaler_455',
        'banco_andbank_pravaler_65_ofertas': 'ag_andbank_pravaler_65',
        'banco_amazonia_sa_003_ofertas': 'ag_banco_amazonia_003',
        
        # Valores Brutos
        'valor_bruto_curso_ofertas': 'valor_bruto_curso',
        'semestre_1_bruto_ofertas': 'sem_1_bruto',
        'semestre_2_bruto_ofertas': 'sem_2_bruto',
        'semestre_3_bruto_ofertas': 'sem_3_bruto',
        'semestre_4_bruto_ofertas': 'sem_4_bruto',
        'semestre_5_bruto_ofertas': 'sem_5_bruto',
        'semestre_6_bruto_ofertas': 'sem_6_bruto',
        'semestre_7_bruto_ofertas': 'sem_7_bruto',
        'semestre_8_bruto_ofertas': 'sem_8_bruto',
        'semestre_9_bruto_ofertas': 'sem_9_bruto',
        'semestre_10_bruto_ofertas': 'sem_10_bruto',
        'semestre_11_bruto_ofertas': 'sem_11_bruto',
        'semestre_12_bruto_ofertas': 'sem_12_bruto',
        
        # Valores FIES
        'valor_curso_fies_ofertas': 'valor_curso_fies',
        'indice_correcao_ipca_ofertas': 'indice_correcao_ipca',
        'semestre_1_fies_ofertas': 'sem_1_fies',
        'semestre_2_fies_ofertas': 'sem_2_fies',
        'semestre_3_fies_ofertas': 'sem_3_fies',
        'semestre_4_fies_ofertas': 'sem_4_fies',
        'semestre_5_fies_ofertas': 'sem_5_fies',
        'semestre_6_fies_ofertas': 'sem_6_fies',
        'semestre_7_fies_ofertas': 'sem_7_fies',
        'semestre_8_fies_ofertas': 'sem_8_fies',
        'semestre_9_fies_ofertas': 'sem_9_fies',
        'semestre_10_fies_ofertas': 'sem_10_fies',
        'semestre_11_fies_ofertas': 'sem_11_fies',
        'semestre_12_fies_ofertas': 'sem_12_fies',
        
        # Colunas CINE
        'NO_CURSO': 'nome_curso_cine',
        'CO_CURSO': 'codigo_curso_cine',
        'CO_CINE_AREA_GERAL': 'codigo_cine_area_geral',
        'NO_CINE_AREA_GERAL': 'nome_cine_area_geral'
    }

    # Mapeamento UF -> Região
    mapa_uf_regiao = {
        'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
        'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
        'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
        'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
        'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
    }

    # 3. Processamento
    print(f"[*] Lendo dataset de ofertas: {arquivo_entrada.name}")
    df_ofertas = pd.read_parquet(str(arquivo_entrada))

    print("[*] Renomeando colunas...")
    df_ofertas.rename(columns=mapa_nomes_curto, inplace=True)

    print("[*] Tratando string de UF e criando nova coluna 'regiao_ies'...")
    
    # Limpa espaços invisíveis e garante maiúsculas antes de mapear
    df_ofertas['uf_ies'] = df_ofertas['uf_ies'].astype(str).str.strip().str.upper()
    
    # Aplica o mapeamento
    df_ofertas['regiao_ies'] = df_ofertas['uf_ies'].map(mapa_uf_regiao)

    # Verifica NaNs na nova coluna
    num_nan = df_ofertas['regiao_ies'].isna().sum()
    if num_nan > 0:
        print(f"   [AVISO] {num_nan} ofertas tinham UF inválida ou faltante.")
    else:
        print("   -> Coluna 'regiao_ies' adicionada com sucesso e sem valores inválidos.")

    print("[*] Removendo linhas fantasmas (sem ano) e forçando Inteiros...")
    
    # Exclui qualquer linha que não tenha Ano ou Semestre preenchido (mata o fantasma)
    df_ofertas = df_ofertas.dropna(subset=['ano', 'semestre'])
    
    # Agora sim, converte com segurança para Inteiro sem precisar de fillna(0)
    df_ofertas['ano'] = df_ofertas['ano'].astype(int)
    df_ofertas['semestre'] = df_ofertas['semestre'].astype(int)

    # 4. Salvamento
    print(f"[*] Salvando versão final (Parquet) em: {pasta_data_04_load_database.name}/...")
    df_ofertas.to_parquet(str(arquivo_saida_parquet), index=False)
    
    print(f"\n[OK] Fase LOAD concluída! Dataset de Ofertas PRONTO PARA ANÁLISE.")
    print(f"Colunas Finais: {len(df_ofertas.columns)}")
    print("="*60)












import random

def auditoria_inscritos_carregados():
    from src.constantes import pasta_data_01_raw_microdata_fies, pasta_data_02_staging_microdata_fies, pasta_data_04_load_database
    import pandas as pd

    print("\n" + "="*90)
    print("🕵️  AUDITORIA FORENSE E RASTREABILIDADE PROFUNDA: INSCRITOS")
    print("="*90)

    path_final = pasta_data_04_load_database / 'inscritos_final_limpo.parquet'
    
    if not path_final.exists():
        print(f"!!! ERRO: Arquivo final não encontrado em: {path_final}")
        return

    # 1. CARREGAMENTO DOS DADOS (Com busca segura de arquivos)
    print("[*] Carregando as 3 camadas de dados (Raw, Staging, Final) para acareação...")
    df_final = pd.read_parquet(str(path_final))
    
    arquivos_raw = [f for f in pasta_data_01_raw_microdata_fies.iterdir() if 'inscricao' in f.name.lower() and f.is_file()]
    if arquivos_raw:
        df_raw = pd.concat([pd.read_csv(str(f), sep=';', encoding='latin-1', decimal=',', low_memory=False) for f in arquivos_raw], ignore_index=True)
    else:
        df_raw = pd.DataFrame()
        print("  [AVISO] Nenhum arquivo RAW de inscritos encontrado para comparação.")

    arquivos_staging = [f for f in pasta_data_02_staging_microdata_fies.iterdir() if 'inscricao' in f.name.lower() and f.is_file()]
    if arquivos_staging:
        df_staging = pd.concat([pd.read_csv(str(f), low_memory=False) for f in arquivos_staging], ignore_index=True)
    else:
        df_staging = pd.DataFrame()

    # 2. AUDITORIA VOLUMÉTRICA E DE CHAVES
    print("\n" + "-"*60)
    print("1. AUDITORIA VOLUMÉTRICA E PERDA DE DADOS")
    print("-"*60)
    print(f"Linhas em RAW     (Bruto c/ duplicatas) : {len(df_raw)}")
    print(f"Linhas em STAGING (Deduplicadas)        : {len(df_staging)}")
    print(f"Linhas em FINAL   (Carregadas no BD)    : {len(df_final)}")
    
    if not df_raw.empty:
        pk_bruto = ['ID do estudante', 'Opções de cursos da inscrição']
        pk_final = ['id_estudante', 'opcao_curso']
        
        chaves_raw = df_raw.dropna(subset=pk_bruto).drop_duplicates(subset=pk_bruto).shape[0]
        chaves_final = df_final.dropna(subset=pk_final).drop_duplicates(subset=pk_final).shape[0]
        
        print(f"-> Integridade de PKs únicas: RAW ({chaves_raw}) vs FINAL ({chaves_final})")
        if chaves_raw == chaves_final:
            print("   ✅ MATEMATICAMENTE PERFEITO: Nenhuma inscrição real foi perdida no pipeline.")
        else:
            print("   ⚠️ AVISO: Houve alteração no número de chaves únicas.")

        # 3. TESTE DE RASTREABILIDADE PROFUNDA (COLUNA A COLUNA)
        print("\n" + "-"*60)
        print("2. TESTE DE RASTREABILIDADE DE VALORES (RAW vs FINAL)")
        print("-"*60)
        try:
            # Sorteia um candidato aleatório
            amostra = df_final.sample(1).iloc[0]
            id_teste = amostra['id_estudante']
            opcao_teste = amostra['opcao_curso']

            print(f"Sorteando candidato aleatório para acareação:\nID: {id_teste} | Opção: {opcao_teste}\n")

            # Busca a mesma linha exata no RAW
            raw_match = df_raw[(df_raw['ID do estudante'] == id_teste) & (df_raw['Opções de cursos da inscrição'] == opcao_teste)].iloc[0]

            # Mapeamento do nome no RAW vs Nome no FINAL
            mapa_comparacao = [
                ("Sexo", "Sexo", "sexo"),
                ("Data de Nasc.", "Data de Nascimento", "data_nascimento"),
                ("UF Residência", "UF de residência", "uf_residencia"),
                ("Renda Per Capita", "Renda mensal bruta per capita", "renda_per_capita"),
                ("Nota Redação", "Redação", "nota_redacao"),
                ("Média ENEM", "Média nota Enem", "media_enem"),
                ("Nome do Curso", "Nome do curso", "nome_curso"),
                ("Turno", "Turno", "turno")
            ]

            for nome_exibicao, col_raw, col_final in mapa_comparacao:
                val_raw = str(raw_match.get(col_raw, 'N/A')).strip().replace(',', '.')
                val_final = str(amostra.get(col_final, 'N/A')).strip()
                
                # Para evitar falsos positivos com floats (ex: 998.0 vs 998)
                val_raw_short = val_raw[:6]
                val_final_short = val_final[:6]
                
                status = "✅ IGUAL" if val_raw_short == val_final_short else "⚠️ DIVERGE"
                print(f"  {nome_exibicao.ljust(18)} | RAW: {val_raw[:15].ljust(15)} | FINAL: {val_final[:15].ljust(15)} -> {status}")

        except Exception as e:
            print(f"  [!] Não foi possível realizar o teste de rastreabilidade profundo: {e}")

    # 4. AUDITORIA CINE (PEDIDOS 1 E 2 DO REPOSITÓRIO ANTIGO)
    print("\n" + "-"*60)
    print("3. ANÁLISE PROFUNDA DE ÁREAS CINE (NaNs RESTANTES)")
    print("-"*60)
    
    col_nome_curso = 'nome_curso'
    col_nome_cine = 'nome_cine_area_geral'
    col_cod_cine = 'codigo_curso_cine'

    filtro_cod_cine_nan = df_final[col_cod_cine].isna()
    cursos_sem_cod_cine = df_final[filtro_cod_cine_nan][col_nome_curso].dropna().unique()

    print(f"-> Total de inscrições com CÓDIGO CINE = NaN: {filtro_cod_cine_nan.sum()}")
    if len(cursos_sem_cod_cine) > 0:
        print(f"   (Isso afeta {len(cursos_sem_cod_cine)} nomes de cursos únicos. Ex: {cursos_sem_cod_cine[:3]})")
    else:
        print("   ✅ 100% dos cursos possuem Código CINE mapeado!")

    filtro_duplo_cine_nan = df_final[col_cod_cine].isna() & df_final[col_nome_cine].isna()
    total_duplo_cine_nan = filtro_duplo_cine_nan.sum()

    print(f"\n-> Total de inscrições TOTALMENTE ÓRFÃOS (Cód + Nome CINE ausentes): {total_duplo_cine_nan}")
    if total_duplo_cine_nan > 0:
        print("   [!] Estes cursos precisam ser adicionados ao dicionário manual na Layer 2:")
        print(df_final[filtro_duplo_cine_nan][col_nome_curso].value_counts().head())
    else:
        print("   ✅ Nenhuma inscrição está com o CINE totalmente em branco.")

    print("="*90 + "\n")
    del df_raw, df_staging, df_final


def auditoria_ofertas_carregadas():
    from src.constantes import pasta_data_01_raw_microdata_fies, pasta_data_02_staging_microdata_fies, pasta_data_04_load_database
    import pandas as pd

    print("\n" + "="*90)
    print("🕵️  AUDITORIA FORENSE E RASTREABILIDADE PROFUNDA: OFERTAS")
    print("="*90)

    path_final = pasta_data_04_load_database / 'ofertas_final_limpo.parquet'
    
    if not path_final.exists():
        print(f"!!! ERRO: Arquivo final não encontrado em: {path_final}")
        return

    # 1. CARREGAMENTO DOS DADOS (Com busca segura de arquivos)
    print("[*] Carregando as 3 camadas de dados (Raw, Staging, Final) para acareação...")
    df_final = pd.read_parquet(str(path_final))
    
    arquivos_raw = [f for f in pasta_data_01_raw_microdata_fies.iterdir() if 'oferta' in f.name.lower() and f.is_file()]
    if arquivos_raw:
        df_raw = pd.concat([pd.read_csv(str(f), sep=';', encoding='latin-1', decimal=',', low_memory=False) for f in arquivos_raw], ignore_index=True)
    else:
        df_raw = pd.DataFrame()
        print("  [AVISO] Nenhum arquivo RAW de ofertas encontrado para comparação.")

    arquivos_staging = [f for f in pasta_data_02_staging_microdata_fies.iterdir() if 'oferta' in f.name.lower() and f.is_file()]
    if arquivos_staging:
        df_staging = pd.concat([pd.read_csv(str(f), low_memory=False) for f in arquivos_staging], ignore_index=True)
    else:
        df_staging = pd.DataFrame()

    # 2. AUDITORIA VOLUMÉTRICA E LIMPEZA DA LINHA FANTASMA
    print("\n" + "-"*60)
    print("1. AUDITORIA VOLUMÉTRICA E PERDA DE DADOS")
    print("-"*60)
    
    # O arquivo original do MEC vem com uma linha em branco no final, vamos limpar para a prova matemática
    df_raw_limpo = df_raw.dropna(how='all') if not df_raw.empty else df_raw
    
    print(f"Linhas em RAW     (Bruto original)      : {len(df_raw)}")
    print(f"Linhas em RAW     (Sem linha fantasma)  : {len(df_raw_limpo)}")
    print(f"Linhas em STAGING (Deduplicadas)        : {len(df_staging)}")
    print(f"Linhas em FINAL   (Carregadas no BD)    : {len(df_final)}")
    
    if not df_raw_limpo.empty:
        pk_bruto = ['Código e-MEC da Mantenedora', 'Código do Local de Oferta', 'Código do Curso', 'Turno']
        pk_final = ['codigo_mec_mantenedora', 'codigo_local_oferta', 'codigo_curso', 'turno']
        
        chaves_raw = df_raw_limpo.dropna(subset=pk_bruto).drop_duplicates(subset=pk_bruto).shape[0]
        chaves_final = df_final.dropna(subset=pk_final).drop_duplicates(subset=pk_final).shape[0]
        
        print(f"-> Integridade de PKs únicas: RAW ({chaves_raw}) vs FINAL ({chaves_final})")
        if chaves_raw == chaves_final:
            print("   ✅ MATEMATICAMENTE PERFEITO: Nenhuma oferta real foi perdida.")
        else:
            print("   ⚠️ AVISO: Houve alteração no número de chaves únicas.")

        # 3. TESTE DE RASTREABILIDADE PROFUNDA (COLUNA A COLUNA)
        print("\n" + "-"*60)
        print("2. TESTE DE RASTREABILIDADE DE VALORES (RAW vs FINAL)")
        print("-"*60)
        try:
            # Sorteia uma oferta aleatória
            amostra = df_final.sample(1).iloc[0]
            cod_ies = amostra['codigo_mec_mantenedora']
            cod_curso = amostra['codigo_curso']
            turno = amostra['turno']

            print(f"Sorteando oferta aleatória para acareação:\nIES: {cod_ies} | Curso: {cod_curso} | Turno: {turno}\n")

            # Busca a mesma linha exata no RAW (protegendo a formatação de texto do Turno)
            raw_match = df_raw_limpo[
                (df_raw_limpo['Código e-MEC da Mantenedora'] == cod_ies) & 
                (df_raw_limpo['Código do Curso'] == cod_curso) & 
                (df_raw_limpo['Turno'].astype(str).str.upper().str.strip() == str(turno).upper().strip())
            ].iloc[0]

            # Mapeamento do nome no RAW vs Nome no FINAL (Para Ofertas)
            mapa_comparacao = [
                ("Nome IES", "Nome da IES", "nome_ies"),
                ("UF IES", "UF da IES", "uf_ies"),
                ("Nome do Curso", "Nome do Curso", "nome_curso"),
                ("Turno", "Turno", "turno"),
                ("Vagas Mec", "Vagas autorizadas e-mec", "vagas_autorizadas_mec"),
                ("Vagas FIES", "Vagas ofertadas FIES", "vagas_fies"),
                ("Participa P-FIES", "Participa do P-FIES", "participa_p_fies"),
                ("Valor Bruto", "Valor bruto do curso", "valor_bruto_curso")
            ]

            for nome_exibicao, col_raw, col_final in mapa_comparacao:
                val_raw = str(raw_match.get(col_raw, 'N/A')).strip()
                val_final = str(amostra.get(col_final, 'N/A')).strip()
                
                # Arredondamentos visuais para checagem exata
                status = "✅ IGUAL" if val_raw[:6].upper() == val_final[:6].upper() else "⚠️ DIVERGE"
                print(f"  {nome_exibicao.ljust(18)} | RAW: {val_raw[:15].ljust(15)} | FINAL: {val_final[:15].ljust(15)} -> {status}")

        except Exception as e:
            print(f"  [!] Não foi possível realizar o teste de rastreabilidade profundo: {e}")

    # 4. AUDITORIA CINE (PEDIDOS 1 E 2 DO REPOSITÓRIO ANTIGO)
    print("\n" + "-"*60)
    print("3. ANÁLISE PROFUNDA DE ÁREAS CINE (NaNs RESTANTES)")
    print("-"*60)
    
    col_nome_curso = 'nome_curso'
    col_nome_cine = 'nome_cine_area_geral'
    col_cod_cine = 'codigo_curso_cine'

    filtro_cod_cine_nan = df_final[col_cod_cine].isna()
    cursos_sem_cod_cine = df_final[filtro_cod_cine_nan][col_nome_curso].dropna().unique()

    print(f"-> Total de vagas com CÓDIGO CINE = NaN: {filtro_cod_cine_nan.sum()}")
    if len(cursos_sem_cod_cine) > 0:
        print(f"   (Afeta {len(cursos_sem_cod_cine)} cursos únicos. Ex: {cursos_sem_cod_cine[:3]})")
    else:
        print("   ✅ 100% das ofertas possuem Código CINE mapeado!")

    filtro_duplo_cine_nan = df_final[col_cod_cine].isna() & df_final[col_nome_cine].isna()
    total_duplo_cine_nan = filtro_duplo_cine_nan.sum()

    print(f"\n-> Total de ofertas TOTALMENTE ÓRFÃS (Cód + Nome CINE ausentes): {total_duplo_cine_nan}")
    if total_duplo_cine_nan > 0:
        print("   [!] Estes cursos precisam de inclusão no dicionário de Ofertas (Layer 2):")
        print(df_final[filtro_duplo_cine_nan][col_nome_curso].value_counts().head())
    else:
        print("   ✅ Nenhuma oferta está com o CINE totalmente em branco.")

    print("="*90 + "\n")
    del df_raw, df_raw_limpo, df_staging, df_final
















from sqlalchemy import create_engine

def exportar_para_sqlite():

    print("\n" + "="*80)
    print("💾 INICIANDO EXPORTAÇÃO PARA BANCO DE DADOS SQLITE")
    print("="*80)


    caminho_db = pasta_data_04_load_database / 'fies_database.db'
    engine = create_engine(f'sqlite:///{caminho_db}')




    # --- 1. EXPORTANDO INSCRITOS ---
    path_inscritos = pasta_data_04_load_database / 'inscritos_final_limpo.parquet'
    if path_inscritos.exists():
        print(f"[*] Lendo Parquet de Inscritos e criando tabela 'inscritos' no BD...")
        df_inscritos = pd.read_parquet(str(path_inscritos))


        # O chunksize evita estouro de memória da RAM ao salvar milhões de linhas de uma vez
        df_inscritos.to_sql('inscritos', con=engine, if_exists='replace', index=False, chunksize=50000)
        print("   ✅ Tabela 'inscritos' criada/atualizada com sucesso!")
    else:
        print("   [!] Arquivo Parquet de inscritos não encontrado para exportação.")



    # --- 2. EXPORTANDO OFERTAS ---
    path_ofertas = pasta_data_04_load_database / 'ofertas_final_limpo.parquet'
    if path_ofertas.exists():
        print(f"\n[*] Lendo Parquet de Ofertas e criando tabela 'ofertas' no BD...")
        df_ofertas = pd.read_parquet(str(path_ofertas))
        df_ofertas.to_sql('ofertas', con=engine, if_exists='replace', index=False, chunksize=50000)
        print("   ✅ Tabela 'ofertas' criada/atualizada com sucesso!")
    else:
        print("   [!] Arquivo Parquet de ofertas não encontrado para exportação.")

    print(f"\n[OK] BANCO DE DADOS PRONTO! Caminho: {caminho_db}")
    print("="*80 + "\n")