def gerar_dataset_candidatos_unicos_por_Prioridade_inicial():
    #analise_001 trata-se apenas de criaca do dataset load de inscritos
    #  mas apenas inscritos dado prioridade inicial
    #  e outro dataset tbm mas com apenas certas colunas

    import pandas as pd
    import os
    from src.constantes import pasta_data_04_load_inscritos, pasta_data_04_load_ofertas, pasta_data_05_processed

    pd.set_option('display.max_columns', None)#type: ignore
    pd.set_option('display.max_rows', None) #type: ignore

    # ==========================================
    # 1. CARREGAMENTO DOS DADOS E CAMINHOS
    # ==========================================
    path = str(pasta_data_04_load_inscritos)
    df = pd.read_parquet(path)

    path_save = str(pasta_data_05_processed)
    nome = os.path.join(path_save, 'candidatos_unicos_por_prioridade_inicial.parquet')
    nome_agregado = os.path.join(path_save, 'candidatos_unicos_por_prioridade_inicial_por_cine_e_regiao_moradia.parquet')

    # ==========================================
    # 2. REGRAS DE NEGÓCIO (FUNIL DE PRIORIDADE)
    # ==========================================
    ordem_de_prioridade = [
        'CONTRATADA',                            # 1. Ganhou a vaga.
        'INSCRIÇÃO POSTERGADA',                  # 2. Ganhou a vaga pro semestre que vem.
        'PRÉ-SELECIONADO',                       # 3. Está vivo no processo agora mesmo! Ainda não enviou os docs.
        'NÃO CONTRATADO',                        # 4. Chegou no banco, mas falhou, ou não enviou docs à CPSA.
        'REJEITADA PELA CPSA',                   # 5. Chegou na faculdade, mas a documentação falhou.
        'OPÇÃO NÃO CONTRATADA',                  # 6. Status de sistema (a outra opção dele deu certo).
        'PARTICIPACAO CANCELADA PELO CANDIDATO', # 7. Desistência voluntária.
        'LISTA DE ESPERA'                        # 8. Nunca passou na nota, o menor progresso possível.
    ]

    chave_agrupamento_inscritos = [
        'ano',
        'semestre',
        'regiao_morar',
        #'regiao_ies_alvo',
        'nome_cine_area_geral',
        "uf_local_oferta",
        'situacao_fies',
    ]

    # A prioridade ('situacao_fies') vem antes da 'opcao_curso' para garantir o melhor status no desempate
    ordem_sort_opcao_em_ultimo = [
        'ano',
        'semestre',
        'id_estudante',
        'situacao_fies',
        'opcao_curso',
    ]

    subset_drop = [
        'ano',
        'semestre',
        'id_estudante'
    ]

    # ==========================================
    # 3. PROCESSAMENTO E EXTRAÇÃO DE ÚNICOS
    # ==========================================
    df_convertido = df.copy()

    # Tipagem categórica para aplicar a ordem customizada
    df_convertido['situacao_fies'] = pd.Categorical(
        df_convertido['situacao_fies'],
        categories=ordem_de_prioridade,
        ordered=True
    )

    # Ordenação
    df_situacaoInscricaoOrdenadaPorPrioridades = (df_convertido
                                                .sort_values(
                                                by=ordem_sort_opcao_em_ultimo,
                                                ascending=True))

    # Remoção de duplicatas (Candidato Único por Semestre/Ano mantendo a melhor situação)
    df_candidatos_unicos = (df_situacaoInscricaoOrdenadaPorPrioridades
                        .drop_duplicates(subset=subset_drop, keep='first'))

    # ==========================================
    # 4. AGRUPAMENTO BASE (O DATASET AGREGADO)
    # ==========================================
    # O parâmetro dropna=False garante que os 11 alunos com região "NaN" sejam contados!
    df_candidatos_unicos_agrupados = (df_candidatos_unicos
                                    .groupby(chave_agrupamento_inscritos, as_index=False, observed=True, dropna=True)['id_estudante']
                                    .count())

                                    #dropna=False no gorupby para que SE uamdas coluans de CHAVE AGRUPAMENTO for false, ele NAO deleta, entao TRUE...

    # Renomeia a coluna para um nome simples e direto e ordena a tabela
    nome_coluna_qtde = 'qtde_candidatos'
    df_candidatos_unicos_agrupados = (df_candidatos_unicos_agrupados
                                    .rename(columns={'id_estudante': nome_coluna_qtde})
                                    .sort_values(['ano', 'semestre', 'uf_local_oferta']))

    # ==========================================
    # 5. EXIBIÇÃO DOS RESULTADOS (DISPLAYS)#type: ignore
    # ==========================================

    print('--- 1. QTDE TOTAL DE CANDIDATOS ÚNICOS DE 2019-1 ATÉ 2021-2 ---')
    qtde_total = df_candidatos_unicos_agrupados[nome_coluna_qtde].sum()
    display(qtde_total) #type: ignore
    print('\n')


    print('--- 2. QTDE DE CANDIDATOS POR ÁREA CINE ---')
    # Agrupa apenas por Área CINE e soma a quantidade de candidatos
    df_qtde_area_cine = (df_candidatos_unicos_agrupados
                        .groupby('nome_cine_area_geral', as_index=False)[[nome_coluna_qtde]]
                        .sum()
                        .sort_values(by=nome_coluna_qtde, ascending=False))

    # Renomeia a coluna especificamente para esta visão ficar clara
    df_qtde_area_cine = df_qtde_area_cine.rename(columns={nome_coluna_qtde: 'qtde_candidatos_por_area_cine'})
    display(df_qtde_area_cine.head(50)) #type: ignore
    print('\n')


    print('--- 3. QTDE DE CANDIDATOS UNICOS EM CADA SITUACAO FIES, DE MESMA REGIAO, POR AREA CINE EM CADA UF DA IES ALVO, POR ANO/SEMESTRE ---')
    # Mostra a base agregada que contém Ano, Semestre, Área, UF, Situação e Qtde
    display(df_candidatos_unicos_agrupados.head(50)) #type: ignore
    print('\n')


    print('--- 4. QTDE DE CANDIDATOS POR ANO E SITUAÇÃO DO FIES ---')
    # Usa observed=True para não gerar FutureWarning e .sum() para somar os alunos reais
    df_situacao_ano = (df_candidatos_unicos_agrupados
                    .groupby(['ano', 'situacao_fies'], observed=True)
                    .agg(qtde_candidatos=(nome_coluna_qtde, 'sum'))
                    .reset_index()
                    .sort_values(by=['ano', 'qtde_candidatos'], ascending=[True, False]))

    display(df_situacao_ano) #type: ignore
    print('\n')






    print('--- 5. QTDE DE CANDIDATOS QUE RESIDEM NA MESMA REGIAO, POR ÁREA CINE ---')

    # Agrupa pela Área CINE e pela Região que JÁ EXISTE e sobreviveu no df agregado
    df_qtde_area_regiao = (df_candidatos_unicos_agrupados
                        .groupby(['nome_cine_area_geral', 'regiao_morar'], as_index=False)[[nome_coluna_qtde]]
                        .sum()
                        .sort_values(by=['nome_cine_area_geral', nome_coluna_qtde], ascending=[True, False]))

    # Renomeia para fazer sentido com ESTE display
    df_qtde_area_regiao = df_qtde_area_regiao.rename(columns={nome_coluna_qtde: 'qtde_candidatos_por_area_cine_e_regiao'})

    # Exibe os resultados
    display(df_qtde_area_regiao.head(50)) #type: ignore
    print('\n')



    print('--- 6. QTDE DE CANDIDATOS QUE RESIDEM NA MESMA REGIAO, POR ÁREA CINE POR ANO e SEMESTRE ---')

    # Agrupa pela Área CINE e pela Região que JÁ EXISTE e sobreviveu no df agregado
    df_qtde_area_regiao_ano_e_semestre = (df_candidatos_unicos_agrupados
                        .groupby(['nome_cine_area_geral', 'regiao_morar','ano','semestre'], as_index=False)[[nome_coluna_qtde]]
                        .sum()
                        .sort_values(by=['ano','semestre', nome_coluna_qtde], ascending=True))

    # Renomeia para fazer sentido com ESTE display
    df_qtde_area_regiao_ano_e_semestre = df_qtde_area_regiao_ano_e_semestre.rename(columns={nome_coluna_qtde: 'qtde_candidatos_por_area_cine_e_regiao_ano_e_semestre '})

    # Exibe os resultados
    display(df_qtde_area_regiao_ano_e_semestre.head(50)) #type: ignore
    print('\n')


    print('salvando os 2 dataset, inscritos normais porem somente inscritos que sao candidatos unicos e o apenas com colunas de interesse (display 3)')
    df_candidatos_unicos.to_parquet(nome,index=False)

    df_candidatos_unicos_agrupados.to_parquet(nome_agregado,index=False)

    print('SALVO')
    