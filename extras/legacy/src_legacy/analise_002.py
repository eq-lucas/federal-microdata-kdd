def eda_gerar_dataset_com_varias_analises():
    # ==============================================================================
    # CRIAÇÃO DO FUNIL ANALÍTICO (8 ETAPAS POR ANO/SEMESTRE E REGIÃO)
    # ==============================================================================
    # O objetivo deste script não é gerar um gráfico visual imediatamente, 
    # mas sim construir um dataset altamente enriquecido (Analytical Base Table - ABT).
    # Ele consolida métricas de Ofertas, Inscritos e Candidatos Únicos, 
    # achatando todas as etapas de conversão do FIES em uma única visão tabular 


    import pandas as pd
    import os
    import numpy as np
    from src.constantes import pasta_data_04_load_inscritos, pasta_data_05_processed,pasta_data_04_load_ofertas

    pd.set_option('display.max_columns', None)

    # --- 1. CARREGAMENTO DAS TRÊS FONTES DE DADOS ---
    caminho_ofertas = pasta_data_04_load_ofertas 
    caminho_inscricoes_geral = pasta_data_04_load_inscritos
    caminho_candidatos_validado = pasta_data_05_processed / 'candidatos_unicos_por_prioridade_inicial.parquet'


    caminho_salvar = pasta_data_05_processed / 'funil_por_regiao.parquet'

    dfo = pd.read_parquet(str(caminho_ofertas))
    dfi_geral = pd.read_parquet(str(caminho_inscricoes_geral))
    dfi_candidatos_unicos = pd.read_parquet(str(caminho_candidatos_validado))




    # para inscritos: 

    'regiao_morar'
    'regiao_ies_alvo'

    # para ofertas:

    'regiao_ies'



    # --- 2. CÁLCULO DE CADA ETAPA DO FUNIL ---
    chave_agrupamento_inscritos = ['ano', 'semestre', 'nome_cine_area_geral','regiao_ies_alvo']
    chave_agrupamento_ofertas = ['ano', 'semestre', 'nome_cine_area_geral','regiao_ies']

    ordem_sort = ['ano', 'semestre', 'id_estudante', 'situacao_fies']
    subset_drop = ['ano', 'semestre', 'id_estudante']



    # --- Etapas de Ofertas ---

    # O que acontece aqui:
    # 1. Agrupamos por Ano, Semestre, Área CINE e Região.
    # 2. Somamos o valor numérico que está dentro da coluna 'vagas_fies' para cada grupo (sendo suas pk = chave_agrupamento_ofertas e olhamos para as outras colunas. 
    #    No caso, 'vagas_fies', e faremos o quanto tem em cada grupinho = a soma das vagas ofertadas de cada linha).
    # 3. O as_index=False garante que o resultado seja um DataFrame com 5 colunas:
    #    (ano, semestre, nome_cine_area_geral, regiao_ies, vagas_fies). Nenhuma coluna some!
    df_vagas_ofertadas = dfo.groupby(chave_agrupamento_ofertas, as_index=False)['vagas_fies'].sum()

    # Mesma lógica acima, mas agora somando os valores da coluna 'vagas_ocupadas'.
    # Retorna um DataFrame de 5 colunas com o total de vagas preenchidas por grupo, onde teremos a quinta coluna que é a 'vagas_ocupadas' e o valor lá dentro será a sumarização que foi a soma de tudo de cada grupo em cada linha.
    df_vagas_ocupadas = dfo.groupby(chave_agrupamento_ofertas, as_index=False)['vagas_ocupadas'].sum()


    # --- Etapas de Inscrições (Gerais) ---

    # O que acontece aqui (Contagem de Linhas):
    # Como cada linha da base original é 1 inscrito, nós agrupamos pela combinação da chave
    # (ex: 2019 - 1 - Computação - SUDESTE) e usamos o .size() para CONTAR quantas linhas caíram aqui.
    # Se caíram 107.956 linhas, significa que temos 107.956 inscritos nessa combinação.
    # O resultado é um DataFrame com as 4 colunas do grupo + 1 coluna chamada 'Inscritos_Geral'.
    df_inscritos_geral = (dfi_geral
                        .groupby(chave_agrupamento_inscritos, as_index=False)
                        .size()
                        .rename(columns={'size': 'Inscritos_Geral'}))

    # Logo aqui, como foi a mesma questão, se tivéssemos um caso sem o as_index=False, o comportamento seria o seguinte:
    # 1. As 4 colunas da nossa chave ('ano', 'semestre', 'nome_cine_area_geral', 'regiao_morar')
    #    deixariam de ser colunas normais e se transformariam no ÍNDICE MÚLTIPLO (MultiIndex) da tabela.
    # 2. O Pandas não retornaria um DataFrame (tabela 2D), mas sim uma Pandas Series (tabela 1D).
    #    A Series teria apenas os valores da contagem (ex: 107956), e o "rótulo" de cada linha 
    #    seria a combinação do índice: (2019, 1, 'Computação', 'SUDESTE').
    # 3. Como o pd.merge() precisa de colunas normais para cruzar as tabelas (on='chave'), 
    #    nós seríamos obrigados a usar um .reset_index(name='Inscritos_Geral') logo após o .size()
    #    para "rebaixar" o índice de volta para colunas normais e transformar a Series de volta em DataFrame.
    # Conclusão: usar as_index=False já nos poupa esse trabalho e entrega a tabela achatada pronta pro Merge!


    #logo aqui como foi a msm questao, se tivessemos u mcaso sem o as index logo seria o seuginte: 


    # --- Etapas de Candidatos Únicos (Gerais)


    df_candidatos_unicos_geral = (dfi_candidatos_unicos
                                .groupby(chave_agrupamento_inscritos, as_index=False)
                                .size()
                                .rename(columns={'size': 'Candidatos_Unicos_Geral'}))



    # --- Etapas "CONCORRENDO por nota enem > nota corte grupo" de múltiplas fontes ---

    # garantir que sao numeros media do enem e a nota de corte e padrao brasileiro
    for df_temp in [dfi_geral, dfi_candidatos_unicos]:

        df_temp['media_enem'] = (pd.to_numeric(df_temp['media_enem']
                                                .astype(str)
                                                .str
                                                .replace(',', '.'), errors='coerce'))
        

        df_temp['nota_corte_gp'] = (pd.to_numeric(df_temp['nota_corte_gp']
                                                            .astype(str)
                                                            .str
                                                            .replace(',', '.'), errors='coerce'))




    # --- Etapa: candidatos unicos com nota do enem suficiente( maior ou igual a do corte) (Candidatos Únicos)-

    FILTRO_TAL= dfi_candidatos_unicos['media_enem'] >= dfi_candidatos_unicos['nota_corte_gp']

    df_base_candidatos_unicos_nota_suficiente = dfi_candidatos_unicos[FILTRO_TAL].copy()



    df_candidatos_unicos_nota_suficiente = (df_base_candidatos_unicos_nota_suficiente
                            .groupby(chave_agrupamento_inscritos, as_index=False)
                            .size()
                            .rename(columns={'size': 'candidatos_unicos_com_nota_suficiente'}))






    # --- Etapa: Inscritos  com Nota Suficiente ---

    FILTRO_TAL= dfi_geral['media_enem'] >= dfi_geral['nota_corte_gp']

    df_inscritos_com_nota_base = dfi_geral[FILTRO_TAL].copy()

    df_inscritos_com_nota = (df_inscritos_com_nota_base
                            .groupby(chave_agrupamento_inscritos, as_index=False)
                            .size()
                            .rename(columns={'size': 'inscritos_com_nota_suficiente'}))



    # --- Etapa: Inscritos com Nota Muito Insuficiente (Gap <= -100) ---

    # DICA DE PERFORMANCE: Subtração vetorizada (instantânea) em vez de .apply()
    # Já criamos a coluna 'gap' para todos os inscritos.
    dfi_geral['gap_nota'] = dfi_geral['media_enem'] - dfi_geral['nota_corte_gp']

    # Filtramos quem teve o gap de -100 pontos ou pior (ex: -100, -150, -200)
    filtro_gap_insuficiente = dfi_geral['gap_nota'] <= -100

    df_inscritos_gap_base = dfi_geral[filtro_gap_insuficiente].copy()

    # Agrupamos e contamos o volume dessas pessoas por região/área
    df_inscritos_gap_menos_100 = (df_inscritos_gap_base
                                .groupby(chave_agrupamento_inscritos, as_index=False)
                                .size()
                                .rename(columns={'size': 'inscritos_gap_menos_100'}))




    # --- Etapa: Destrinchar a Situação do FIES (Pivot) ---

    # O pivot_table usa a nossa chave como índice, transforma os valores únicos da 
    # coluna 'situacao_fies' em novas colunas, e conta (size) quantas linhas caem em cada cruzamento.
    # O fill_value=0 garante que, se não houver 'CONTRATADA' naquele grupo, a coluna receba 0.

    # POR QUE USAR PIVOT_TABLE DIRETO EM VEZ DE GROUPBY + PIVOT?
    # Na teoria, poderíamos fazer um .groupby() nas chaves + 'situacao_fies' para contar as linhas,
    # e depois usar um .pivot() (ou .unstack()) para pegar essa coluna de situação e "tombar"
    # seus valores transformando-os em colunas.
    # O problema? Isso exige dois processamentos distintos na memória RAM e cria um MultiIndex intermediário chato de lidar.
    # O .pivot_table() é uma função otimizada (baseada em C++) que faz as duas operações (Agrupar + Tombar) 
    # EM UM ÚNICO PASSO, sendo muito mais performático para bases de 3 milhões de linhas.

    # O QUE ELE FAZ COM O NOSSO DATASET REAL DE 60 COLUNAS?
    # Ele ignora completamente as outras 55 colunas! O motor do Pandas olha estritamente 
    # para as colunas passadas nos parâmetros abaixo e descarta o resto virtualmente para esta operação.

    df_status_inscritos = (dfi_geral
                        .pivot_table(
                            # 1. index: As colunas que vão "segurar" a tabela (nossas 4 chaves padrão).
                            # Ele agrupa por elas. Para cada combinação única (ex: 2019, 1, Computação, Sul), 
                            # teremos exatamente 1 linha no DataFrame resultante.
                            index=chave_agrupamento_inscritos,
                            
                            # 2. columns: A coluna categórica que queremos "destrinchar" na horizontal.
                            # Ele vai ler os valores únicos da coluna original (CONTRATADA, LISTA DE ESPERA, etc) 
                            # e criar 8 NOVAS COLUNAS na tabela, uma para cada status.
                            columns='situacao_fies',
                            
                            # 3. aggfunc='size': Qual a matemática do agrupamento?
                            # Como cada linha original da nossa base de 60 colunas representa 1 inscrito,
                            # nós não queremos somar notas aqui, queremos apenas "contar" (size) quantas 
                            # linhas originais caíram no cruzamento exato da linha (index) com a coluna (status).
                            aggfunc='size',
                            
                            # 4. fill_value=0: O tratamento de NaNs nativo.
                            # Se, por exemplo, na região 'Norte' em '2019' ninguém ficou com o status 'CONTRATADA', 
                            # o cruzamento ficaria vazio (NaN). Com esse parâmetro, o Pandas já injeta um '0' inteiro 
                            # na hora da criação, economizando um .fillna(0) depois.
                            fill_value=0
                        )
                        .reset_index()) # O reset_index pega as 4 chaves (que viraram o "nome" da linha no pivot)
                                        # e as rebaixa de volta para colunas normais para podermos usar no pd.merge()


    # Padronização dos nomes das colunas novas geradas pelo Pivot:
    novos_nomes = {}
    for col in df_status_inscritos.columns:
        if col not in chave_agrupamento_inscritos:
            # Pega "LISTA DE ESPERA", transforma em "vol_inscritos_lista_de_espera"
            nome_limpo = (str(col).lower()
                        .replace(' ', '_')
                        .replace('-', '_')
                        .replace('ç', 'c')
                        .replace('ã', 'a')
                        .replace('õ', 'o')
                        .replace('é', 'e'))
            novos_nomes[col] = f"vol_inscritos_{nome_limpo}"

    df_status_inscritos = df_status_inscritos.rename(columns=novos_nomes)




    # --- 3. JUNÇÃO FINAL ---

    # antes da juncao o df_vagas ocuapdas precisa tratar o nome da sua coluna regiao_ies so para conseguir mergear
    # tal merge oq ele faz atraves de conexao ? adicioan as coluans do df bem em A, NUM  a MERGE b... retirando apenas as chaves de juncao (Nao sao repetidas)


    df_final = df_inscritos_geral

    lista_dfs_inscritos = [
    #df_inscritos_geral, nao precisa estar aqui pq df_final eh o primeiro! "df_final = df_vagas_ofertadas"
    df_inscritos_com_nota,
    df_candidatos_unicos_geral,
    df_candidatos_unicos_nota_suficiente,
    #df_vagas_ocupadas como dito, esta aqui nao tem a mesma chave de juncao!
    df_inscritos_gap_menos_100,
    df_status_inscritos,

    ]


    for df_etapa in lista_dfs_inscritos:
        df_final = pd.merge(df_final,
                            df_etapa,
                            on=chave_agrupamento_inscritos,
                            how='outer')

    df_final = df_final.fillna(0) # esta transformando cada linha q seja com NA em QUALQUER coluna, em um valor numerico zero...

    apenas_para_mergear_com_ofertas= {'regiao_ies_alvo':'regiao_ies'} 

    df_final= df_final.rename(columns=apenas_para_mergear_com_ofertas)


        
    lista_dfs_ofertas=[

        df_vagas_ofertadas,
        df_vagas_ocupadas,
    ]

    chave_merge_inscritos_com_ofertas = ['ano', 'semestre', 'nome_cine_area_geral','regiao_ies']

    for  df_etapa in lista_dfs_ofertas:
        df_final = pd.merge( left= df_final,
            right=df_etapa,
            left_on=chave_merge_inscritos_com_ofertas,
            right_on=chave_merge_inscritos_com_ofertas,
            how='outer',
            suffixes=('','_OFERTA')) # somenete para nao visualziar 2 colunas de mesmo nome!
                                        # as chaves em on, tbm caso tiverem o msm nome, ficara 2 colunas iguais, mas se qusier da pra manter somente como eh
                                        # chaves de on, manter so 1 vez elas ja q sao mei oq identicas SEMPRE e isso ja eh o padrao entao o suffixes serve so para as
                                        # outras colunas!

    df_final = df_final.fillna(0) 



    # duvida:

    #A sua lógica de que "uma IES oferece vaga para 20 e tem 100 inscritos (1 para N)"
    #estaria 100% correta se estivéssemos cruzando os microdados brutos linha a linha.
    #Mas lembre-se de um detalhe crucial: os seus DataFrames agora estão AGRUPADOS.
    #Neste momento do script, você não tem mais a linha do aluno João e a linha da UTFPR.
    #Você tem uma linha única que diz: "2019 - 1º Sem - Computação - Sul". É uma relação de 1 para 1.


    #Motivo 1: A Vaga Fantasma (Oferta sem Demanda)
    #
    #Imagine que uma IES no interior do Acre (Região Norte) abriu 10 vagas para "Física Nuclear" (df_vagas_ofertadas). Porém, o curso era tão específico que absolutamente nenhum aluno se inscreveu para ele lá (df_inscritos_geral não tem essa linha).
    #
    #    Se você usar INNER: O Pandas não acha correspondência, deleta a linha, e essas 10 vagas somem do seu relatório final. O seu somatório de "Total de Vagas do MEC" vai dar errado.
    #
    #    Se você usar OUTER: O Pandas mantém a linha da oferta. Como não acha inscritos, ele coloca NaN nos inscritos. O seu .fillna(0) entra em ação e transforma em 0. Você descobre um insight valioso: "Tivemos 10 vagas com 0 demanda".
    #
    #Motivo 2: O Aluno Perdido (Demanda sem Oferta)
    #
    #Imagine que 5 alunos se inscreveram para "Medicina" no "Centro-Oeste". Mas, por algum bug do MEC, cassação de liminar na justiça, ou erro no Censo do INEP, a base de Ofertas não tem nenhuma vaga registrada para essa exata combinação.
    #
    #    Se você usar INNER: O Pandas deleta esses 5 alunos do seu funil final. O seu total de inscritos não vai bater com o total oficial do governo.
    #
    #    Se você usar OUTER: O Pandas mantém os alunos, bota 0 nas vagas ofertadas, e você consegue auditar que existiu um erro de sistema no MEC para esses casos.



    display(df_final) #ignore: type



    df_final.to_parquet(str(caminho_salvar), index=False)