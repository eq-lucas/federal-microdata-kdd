def orquestrador_inicial_inscritos_nove_um():

    X, Y, X_teste, Y_teste, df_base = gerar_ABT()

    modelo = TreinarModelo('veio_treino', X, Y)

    prever_probabilidade_treino(
        'veio_treino', 'simm',
        modelo, X, Y, X_teste, Y_teste
    )

    analises_e_datasets(
        'veio_treino', 'simm',
        modelo, X, Y, X_teste, Y_teste,
        df_inicial=df_base
    )

    acuracia_e_previsao(
        'veio', 'simm',
        modelo, X, Y, X_teste, Y_teste
    )



def orquestrador_ja_rodado_inscritos_nove_um():
    import pandas as pd
    import joblib
    from src.constantes import (
        pasta_data_05_processed_analise_009_1_X_treino, 
        pasta_data_05_processed_analise_009_1_y_treino,
        pasta_modelo_analise_009_1,
        pasta_data_05_processed_analise_009_1_X_teste,
        pasta_data_05_processed_analise_009_1_y_teste,
        pasta_data_04_load_inscritos
    )

    X= pd.read_parquet(str(pasta_data_05_processed_analise_009_1_X_treino))
    Y= pd.read_parquet(str(pasta_data_05_processed_analise_009_1_y_treino))
    X_teste= pd.read_parquet(str(pasta_data_05_processed_analise_009_1_X_teste))
    Y_teste= pd.read_parquet(str(pasta_data_05_processed_analise_009_1_y_teste))
    df_base = pd.read_parquet(str(pasta_data_04_load_inscritos))

    modelo = joblib.load(str(pasta_modelo_analise_009_1))
    prever_probabilidade_treino('veio_treino', 'simm', modelo, X, Y, X_teste, Y_teste)
    acuracia_e_previsao('veio', 'simm', modelo, X, Y, X_teste, Y_teste)



def gerar_ABT():
    import pandas as pd
    from src.constantes import pasta_data_04_load_inscritos,pasta_data_05_processed_analise_009_1
    from sklearn.linear_model import LogisticRegression

    # Configuração de visualização
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 1. CARREGAR OS DADOS
    df_base = pd.read_parquet(str(pasta_data_04_load_inscritos))

    # 2. CRIAR VARIÁVEIS NOVAS (Vetorizado, muito mais rápido)
    df_base['gap'] = df_base['media_enem'] - df_base['nota_corte_gp']

    def idade(X:str):
        lista = X.split('/')
        ano= lista[2]
        return 2026 - int(ano)

    df_base['idade'] = df_base['data_nascimento'].apply(idade)


  #  filtro_treino_base= (df_base['ano'] == 2019) & ( (df_base['semestre'] == 1) | (df_base['semestre'] == 2) )
  #  filtro_teste_base= ~(df_base['ano'] == 2019) & ( (df_base['semestre'] == 1) | (df_base['semestre'] == 2) )

    # Novo Filtro de Treino: Incluindo 2020-1
    filtro_treino_base = (df_base['ano'] == 2019) | ((df_base['ano'] == 2020) & (df_base['semestre'] == 1))

    # Novo Filtro de Teste: 2020-2 em diante
    filtro_teste_base = ((df_base['ano'] == 2020) & (df_base['semestre'] == 2)) | (df_base['ano'] == 2021)

    print('total de linhas de apenas 2019(1 e 2 semestre): ', df_base[filtro_treino_base].shape)

    print('total de linhas de apenas 2020 e 2021 (1 e 2 semestre): ', df_base[filtro_teste_base].shape)


    print('-'*60)
    print('qtde de inscritos apenas de 2019 (1 e 2 semestre)')
    display(df_base[filtro_treino_base].groupby('situacao_fies',as_index=False).size().rename(columns={'size':'qtde_inscritos'}))#type: ignore


    print('-'*60)
    print('qtde de inscritos apenas de 2020 e 2021 (1 e 2 semestre)')

    display(df_base[filtro_teste_base].groupby('situacao_fies',as_index=False).size().rename(columns={'size':'qtde_inscritos'}))#type: ignore


    # 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
    # Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
    df = pd.get_dummies(df_base, columns=['subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior'], drop_first=True)

    # 4. FILTRAR A BASE DE DADOS
    contratados = ['CONTRATADA']
    nao_contratados = ['NÃO CONTRATADO']

    filtro = df['situacao_fies'].isin(contratados + nao_contratados)
    df = df[filtro].copy()

    # 5. CRIAR A VARIÁVEL ALVO (Y) BINÁRIA
    df['contratado'] = df['situacao_fies'].apply(lambda x: 1 if x in contratados else 0)


    # 6. PESCAR AS COLUNAS DUMMIES
    # O Python procura e guarda apenas as colunas que começam com esse nome
    colunas_dummies_turno = [col for col in df.columns if col.startswith('turno_')]
    
    colunas_dummies_ensino_medio_escola_publica = [col for col in df.columns if col.startswith('ensino_medio_escola_publica_')]
    colunas_dummies_etnia = [col for col in df.columns if col.startswith('etnia_cor_')]
    colunas_dummies_cine = [col for col in df.columns if col.startswith('subarea_conhecimento_')]
    colunas_dummies_regiao = [col for col in df.columns if col.startswith('regiao_morar_')]
    colunas_dummies_natureza = [col for col in df.columns if col.startswith('natureza_juridica_mantenedora_')]
    #colunas_dummies_opcao_curso = [col for col in df.columns if col.startswith('opcao_curso')]
    colunas_dummies_conceito_curso = [col for col in df.columns if col.startswith('conceito_curso_gp')]
    colunas_dummies_conceito_concluiu_curso_superior = [col for col in df.columns if col.startswith('concluiu_curso_superior')]


    

    # 9. CRIAR A LISTA EXATA DE FEATURES
    # Juntamos as variáveis que você escolheu com as dummies
    features_exatas = ['renda_per_capita', 'gap','idade'] + colunas_dummies_conceito_concluiu_curso_superior + colunas_dummies_conceito_curso + colunas_dummies_cine + colunas_dummies_regiao + colunas_dummies_natureza + colunas_dummies_etnia + colunas_dummies_turno + colunas_dummies_ensino_medio_escola_publica



    # Faz o Raio-X apenas nas colunas que vão pro modelo
    print("--- Quantidade de NaNs por coluna ---")
    print(df[features_exatas].isna().sum())

    # Isso garante que se o aluno sumir, ele some das duas listas
    df_limpo = df.dropna(subset=features_exatas).copy()

    #filtro_treino= (df_limpo['ano'] == 2019) & ( (df_limpo['semestre'] == 1) | (df_limpo['semestre'] == 2) )
    #filtro_teste= ~(df_limpo['ano'] == 2019) & ( (df_limpo['semestre'] == 1) | (df_limpo['semestre'] == 2) )

    # Novo Filtro de Treino: Incluindo 2020-1
    filtro_treino = (df_limpo['ano'] == 2019) | ((df_limpo['ano'] == 2020) & (df_limpo['semestre'] == 1))

    # Novo Filtro de Teste: 2020-2 em diante
    filtro_teste = ((df_limpo['ano'] == 2020) & (df_limpo['semestre'] == 2)) | (df_limpo['ano'] == 2021)


    df_treino = df_limpo[filtro_treino]
    df_teste = df_limpo[filtro_teste]


    # 8. SEPARAR O X E O Y
    # O X recebe ESTRITAMENTE as colunas da sua lista branca
    X = df_treino[features_exatas]

    # O Y recebe a sua resposta
    Y = df_treino['contratado']



    X_teste = df_teste[features_exatas]

    # O Y recebe a sua resposta
    Y_teste= df_teste['contratado']


    print('coluna x linha do X com NaN inclusos:',X.shape)

    print('coluna x linha do X sem NaN inclusos:',X.shape)


    # %% 
    # 12. SALVAR MATRIZES PROCESSADAS (ABT)
    import os

    # Definindo o diretório de destino
    caminho_salvar = str(pasta_data_05_processed_analise_009_1)

    # Criando os caminhos completos
    path_x_treino = os.path.join(caminho_salvar, 'x_treino.parquet')
    path_y_treino = os.path.join(caminho_salvar, 'y_treino.parquet')
    path_x_teste = os.path.join(caminho_salvar, 'x_teste.parquet')
    path_y_teste = os.path.join(caminho_salvar, 'y_teste.parquet')

    # Salvando os arquivos em formato Parquet para manter a performance
    X.to_parquet(path_x_treino,index=False)
    Y.to_frame().to_parquet(path_y_treino,index=False)
    X_teste.to_parquet(path_x_teste,index=False)
    Y_teste.to_frame().to_parquet(path_y_teste,index=False)

    print('-'*60)
    print(f"Arquivos da ABT salvos com sucesso em:\n{caminho_salvar}")
    print(f"Registros de Treino: {len(X):,}")
    print(f"Registros de Teste: {len(X_teste):,}")

    return X,Y,X_teste,Y_teste,df_base



def TreinarModelo(treino_ou_teste_veio: str, x_treino, y_treino):

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from src.constantes import pasta_modelo_analise_009_1

    X = x_treino
    Y = y_treino

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', LogisticRegression(
            penalty='elasticnet',
            l1_ratio=0.5,           # OBRIGATÓRIO no elasticnet
            solver='saga',
            C=0.1,
            class_weight='balanced',
            max_iter=10000,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X, Y.values.ravel())

    print('--- Treinamento Concluído ---')

    joblib.dump(pipeline, str(pasta_modelo_analise_009_1))
    print(f'Modelo salvo em: {str(pasta_modelo_analise_009_1)}')

    return pipeline



def prever_probabilidade_treino(
    treino_ou_teste_veio: str,
    usar_modelo: str,
    modelo_vindo,
    x_treino, y_treino,
    x_teste, y_teste
):

    import matplotlib.pyplot as plt
    import joblib
    from src.constantes import pasta_modelo_analise_009_1

    X = x_treino if treino_ou_teste_veio == 'veio_treino' else x_teste

    modelo = modelo_vindo if usar_modelo == 'simm' \
        else joblib.load(str(pasta_modelo_analise_009_1))

    chances = modelo.predict_proba(X)[:, 1]

    # GRÁFICO RENDA
    plt.figure(figsize=(12, 6))
    plt.scatter(X['renda_per_capita'], chances, alpha=0.005, s=1)
    plt.title('Probabilidade vs Renda')
    plt.xlabel('Renda per capita')
    plt.ylabel('Probabilidade')
    plt.show()

    # GRÁFICO GAP
    plt.figure(figsize=(12, 6))
    plt.scatter(X['gap'], chances, alpha=0.005, s=1)
    plt.title('Probabilidade vs GAP')
    plt.xlabel('GAP')
    plt.ylabel('Probabilidade')
    plt.show()

    print(f"{len(X):,} registros processados.")


def analises_e_datasets(
    treino_ou_teste_veio: str,
    usar_modelo: str,
    modelo_vindo,
    x_treino, y_treino,
    x_teste, y_teste,
    df_inicial
):

    import pandas as pd
    import joblib
    from src.constantes import pasta_modelo_analise_009_1

    X = x_treino if treino_ou_teste_veio == 'veio_treino' else x_teste

    modelo = modelo_vindo if usar_modelo == 'simm' \
        else joblib.load(str(pasta_modelo_analise_009_1))

    # 👇 PEGA O MODELO REAL DENTRO DO PIPELINE
    modelo_logistico = modelo.named_steps['modelo']

    coeficientes = modelo_logistico.coef_[0]

    tabela_importancia = pd.DataFrame({
        'Variável': X.columns,
        'Coeficiente': coeficientes
    })

    tabela_importancia['Impacto_Absoluto'] = tabela_importancia['Coeficiente'].abs()

    tabela_importancia = tabela_importancia.sort_values(
        by='Impacto_Absoluto',
        ascending=False
    )

    print("Top 20 variáveis mais importantes:")
    display(tabela_importancia.head(20))



def acuracia_e_previsao(
    veio_datasets_x_e_y: str,
    usar_modelo: str,
    modelo_vindo,
    x_treino, y_treino,
    x_teste, y_teste
):

    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        roc_auc_score,
        roc_curve
    )
    from src.constantes import pasta_modelo_analise_009_1

    X_teste = x_teste
    Y_teste = y_teste

    modelo = modelo_vindo if usar_modelo == 'simm' \
        else joblib.load(str(pasta_modelo_analise_009_1))

    # 1️⃣ Probabilidades
    chances = modelo.predict_proba(X_teste)[:, 1]

    # 2️⃣ ROC-AUC (agora correto)
    auc = roc_auc_score(Y_teste, chances)
    print(f"ROC-AUC: {auc:.4f}")

    # 3️⃣ Threshold ótimo
    fpr, tpr, thresholds = roc_curve(Y_teste, chances)
    melhor_indice = (tpr - fpr).argmax()
    corte = thresholds[melhor_indice]

    corte = 0.6

    Y_previsto = (chances >= corte).astype(int)

    # 4️⃣ Acurácia
    acuracia = accuracy_score(Y_teste, Y_previsto)
    print(f"Acurácia (threshold ótimo): {acuracia:.4f}")
    print(f"Threshold ótimo: {corte:.4f}")

    # 5️⃣ Matriz de Confusão
    matriz = confusion_matrix(Y_teste, Y_previsto)

    plt.figure(figsize=(6, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusão")
    plt.show()

    print(f'''

''')

# %%
