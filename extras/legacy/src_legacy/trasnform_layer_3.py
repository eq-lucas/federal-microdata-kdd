import pandas as pd
import numpy as np
from src.constantes import (
    pasta_data_02_staging_microdata_fies, 
    pasta_data_03_transform_fies
)

def processar_modalidades_e_peneira():
    print("\n" + "="*80)
    print("🧠 INICIANDO LAYER 3: CLASSIFICAÇÃO DE MODALIDADES E PENEIRA SOCIOECONÔMICA")
    print("="*80)

    arquivo_inscritos = pasta_data_03_transform_fies / 'fies_inscritos_unificado_corrigido.parquet'
    arquivo_ofertas = pasta_data_03_transform_fies / 'fies_ofertas_unificado_corrigido.parquet'
    
    # Destino final da Layer 3
    arquivo_saida = pasta_data_03_transform_fies / 'fies_inscritos_com_modalidade_final.parquet'

    if not arquivo_inscritos.exists() or not arquivo_ofertas.exists():
        print("[!] ERRO: Arquivos da Layer 2 não encontrados. Execute a Transform Layer 2 primeiro.")
        return

    # --- 1. CARREGAMENTO DOS DADOS ---
    print("[*] Carregando datasets da Layer 2...")
    df = pd.read_parquet(str(arquivo_inscritos))
    dfo = pd.read_parquet(str(arquivo_ofertas))

    # =========================================================================
    # AUDITORIA 1: COMPARAÇÃO VOLUMÉTRICA STAGING vs LAYER 2
    # =========================================================================
    print("\n" + "-"*60)
    print("📊 1. AUDITORIA VOLUMÉTRICA: STAGING vs LAYER 2")
    print("-"*60)
    print("[*] Varrendo pasta 02_staging para somar volume original de inscrições...")
    
    arquivos_staging = list(pasta_data_02_staging_microdata_fies.glob('*inscricao*.csv'))
    linhas_staging = 0
    for f in arquivos_staging:
        # Lê apenas a primeira coluna para ser incrivelmente rápido e não estourar a RAM
        df_temp = pd.read_csv(str(f), usecols=[0], low_memory=False)
        linhas_staging += len(df_temp)
    
    linhas_layer2 = len(df)
    print(f"  -> Total de inscrições no Staging (Pós-deduplicação): {linhas_staging}")
    print(f"  -> Total de inscrições na Layer 2 (Atual)           : {linhas_layer2}")
    
    if linhas_staging == linhas_layer2:
        print("  ✅ SUCESSO: 100% de integridade volumétrica mantida. Nenhum aluno perdido.")
    else:
        print(f"  ⚠️ AVISO: Divergência de {linhas_staging - linhas_layer2} linhas encontrada!")

    # =========================================================================
    # REGRAS DE NEGÓCIO E LIMITES FINANCEIROS
    # =========================================================================
    print("\n" + "-"*60)
    print("💰 2. APLICAÇÃO DAS REGRAS DE NEGÓCIO (CORTES DE RENDA E REGIÃO)")
    print("-"*60)

    salario_minimo_ano = {2019: 998.00, 2020: 1045.00, 2021: 1100.00}
    limite_3sm = {ano: sm * 3 for ano, sm in salario_minimo_ano.items()}
    limite_5sm = {ano: sm * 5 for ano, sm in salario_minimo_ano.items()}

    for ano, sm in salario_minimo_ano.items():
        print(f"  [Ano {ano}] Salário Mínimo: R$ {sm:7.2f} | Limite Mod I (3 SM): R$ {limite_3sm[ano]:7.2f} | Limite Mod II/III (5 SM): R$ {limite_5sm[ano]:7.2f}")

    mapa_uf_regiao = {
        'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
        'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
        'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
        'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
        'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
    }

    # Prepara as colunas
    col_ano = 'ano_processo_seletivo_inscricao'
    col_renda = 'renda_mensal_bruta_per_capita_inscricao'
    col_uf = 'uf_residencia_inscricao'

    df[col_renda] = pd.to_numeric(df[col_renda].astype(str).str.replace(',', '.'), errors='coerce')
    df[col_ano] = pd.to_numeric(df[col_ano], errors='coerce').fillna(0).astype(int)

    df['limite_3sm_ano'] = df[col_ano].map(limite_3sm)
    df['limite_5sm_ano'] = df[col_ano].map(limite_5sm)
    df['regiao_residencia'] = df[col_uf].map(mapa_uf_regiao)

    # =========================================================================
    # CLASSIFICAÇÃO DE MODALIDADE (COM AUDITORIA VETORIZADA)
    # =========================================================================
    print("\n[*] Executando a Peneira Socioeconômica Inicial...")

    # Máscaras Booleanas Isoladas para Auditoria
    mask_mod_I = (df[col_renda].notna()) & (df[col_renda] <= df['limite_3sm_ano'])
    
    mask_mod_II = (df[col_renda].notna()) & (df[col_renda] > df['limite_3sm_ano']) & \
                  (df[col_renda] <= df['limite_5sm_ano']) & \
                  (df['regiao_residencia'].isin(['Norte', 'Nordeste', 'Centro-Oeste']))
                  
    mask_mod_III = (df[col_renda].notna()) & (df[col_renda] > df['limite_3sm_ano']) & \
                   (df[col_renda] <= df['limite_5sm_ano']) & \
                   (df['regiao_residencia'].isin(['Sul', 'Sudeste']))

    # Atribuição
    df['modalidade_fies'] = 'eliminado' # Default para quem não bater as máscaras
    df.loc[mask_mod_I, 'modalidade_fies'] = 'Modalidade I'
    df.loc[mask_mod_II, 'modalidade_fies'] = 'Modalidade II'
    df.loc[mask_mod_III, 'modalidade_fies'] = 'Modalidade III (P-FIES)'

    qtd_mod_I = mask_mod_I.sum()
    qtd_mod_II = mask_mod_II.sum()
    qtd_mod_III = mask_mod_III.sum()
    qtd_eliminados_renda = len(df) - (qtd_mod_I + qtd_mod_II + qtd_mod_III)

    print(f"  -> Aprovados na Modalidade I   (Renda <= 3SM)       : {qtd_mod_I}")
    print(f"  -> Aprovados na Modalidade II  (3-5SM, N/NE/CO)     : {qtd_mod_II}")
    print(f"  -> Aprovados na Modalidade III (3-5SM, Sul/Sudeste) : {qtd_mod_III}")
    print(f"  -> Eliminados PRELIMINARMENTE  (Renda Alta ou Nula) : {qtd_eliminados_renda}")

    # =========================================================================
    # VALIDAÇÃO CRUZADA COM OFERTAS (FILTRO P-FIES)
    # =========================================================================
    print("\n" + "-"*60)
    print("🔄 3. VALIDAÇÃO DE CONTRADIÇÃO DO MEC (P-FIES)")
    print("-"*60)
    print("[*] Verificando se os candidatos da Modalidade III aplicaram para vagas que realmente aceitam P-FIES...")
    
    colunas_pk_ofertas = [
        'ano_ofertas', 'semestre_ofertas', 'codigo_e_mec_mantenedora_ofertas',
        'codigo_local_oferta_ofertas', 'codigo_grupo_preferencia_ofertas',
        'codigo_curso_ofertas', 'turno_ofertas'
    ]

    colunas_pk_inscritos = [
        'ano_processo_seletivo_inscricao', 'semestre_processo_seletivo_inscricao',
        'codigo_e_mec_mantenedora_inscricao', 'codigo_local_oferta_inscricao',
        'codigo_grupo_preferencia_inscricao', 'codigo_curso_inscricao', 'turno_inscricao'
    ]

    dfo_slim = dfo[colunas_pk_ofertas + ['participa_p_fies_ofertas']].copy()
    dfo_slim.drop_duplicates(subset=colunas_pk_ofertas, keep='first', inplace=True)

    # Padronização de chaves
    for col_insc, col_of in zip(colunas_pk_inscritos, colunas_pk_ofertas):
        if 'turno' in col_insc:
            df[col_insc] = df[col_insc].astype(str).str.strip().str.upper()
            dfo_slim[col_of] = dfo_slim[col_of].astype(str).str.strip().str.upper()
        else:
            df[col_insc] = pd.to_numeric(df[col_insc], errors='coerce')
            dfo_slim[col_of] = pd.to_numeric(dfo_slim[col_of], errors='coerce')

    dfo_slim['participa_p_fies_ofertas'] = dfo_slim['participa_p_fies_ofertas'].astype(str).str.strip().str.upper()

    df_merged = df.merge(
        dfo_slim,
        left_on=colunas_pk_inscritos,
        right_on=colunas_pk_ofertas,
        how='left'
    )

    # Detecta Contradição: O aluno passou na renda da Mod III, mas aplicou para uma faculdade que diz NÃO pro P-FIES
    filtro_contradicao_pfies = (df_merged['modalidade_fies'] == 'Modalidade III (P-FIES)') & \
                               (df_merged['participa_p_fies_ofertas'].isin(['NAO', 'NÃO', 'NAN', 'NONE', '']))
    
    qtd_rejeitados_pfies = filtro_contradicao_pfies.sum()
    
    # Rebaixamento da categoria
    df_merged.loc[filtro_contradicao_pfies, 'modalidade_fies'] = 'eliminado'

    print(f"  -> Candidatos que passaram na renda Mod III         : {qtd_mod_III}")
    print(f"  -> Ofertas de faculdades que rejeitavam P-FIES      : {qtd_rejeitados_pfies} (Eliminados do sistema)")
    print(f"  -> Candidatos DEFINITIVOS na Modalidade III         : {qtd_mod_III - qtd_rejeitados_pfies}")

    # =========================================================================
    # FINALIZAÇÃO E LIMPEZA
    # =========================================================================
    cols_para_remover = ['limite_3sm_ano', 'limite_5sm_ano', 'regiao_residencia', 'participa_p_fies_ofertas']
    df_final = df_merged.drop(columns=[c for c in cols_para_remover if c in df_merged.columns])
    
    for col in colunas_pk_ofertas:
        if col in df_final.columns:
            df_final.drop(columns=[col], inplace=True)

    print("\n" + "-"*60)
    print("📊 4. RESUMO FORENSE DA LAYER 3 (MATRIZ FINAL)")
    print("-"*60)
    
    # Pivot Table para mostrar um resumo lindo no terminal (Anos vs Modalidades)
    tabela_resumo = df_final.pivot_table(
        index=col_ano, 
        columns='modalidade_fies', 
        aggfunc='size', 
        fill_value=0
    )
    
    # Exibe garantindo uma formatação espaçada
    print(tabela_resumo.to_string())

    print("\n[*] Salvando dataset final e injetando na próxima camada...")
    df_final.to_parquet(str(arquivo_saida), index=False)
    
    print(f"✅ LAYER 3 CONCLUÍDA! Salvo em: {arquivo_saida.name}")
    print("="*80 + "\n")