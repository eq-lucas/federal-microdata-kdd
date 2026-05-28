def orquestrador_inicial_inscritos_16():

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


def orquestrador_ja_rodado_inscritos_16():
    import pandas as pd
    import joblib
    from src.constantes import (
        pasta_data_05_processed_analise_016_X_treino, 
        pasta_data_05_processed_analise_016_y_treino,
        pasta_modelo_analise_016,
        pasta_data_05_processed_analise_016_X_teste,
        pasta_data_05_processed_analise_016_y_teste,
        pasta_data_04_load_inscritos
    )

    X= pd.read_parquet(str(pasta_data_05_processed_analise_016_X_treino))
    Y= pd.read_parquet(str(pasta_data_05_processed_analise_016_y_treino))
    X_teste= pd.read_parquet(str(pasta_data_05_processed_analise_016_X_teste))
    Y_teste= pd.read_parquet(str(pasta_data_05_processed_analise_016_y_teste))

    df_base = pd.read_parquet(str(pasta_data_04_load_inscritos))

    modelo = joblib.load(str(pasta_modelo_analise_016))
    analises_e_datasets('veio_treino', 'simm', modelo, X, Y, X_teste, Y_teste, df_inicial=df_base)
    prever_probabilidade_treino('veio_treino', 'simm', modelo, X, Y, X_teste, Y_teste)
    acuracia_e_previsao('veio', 'simm', modelo, X, Y, X_teste, Y_teste)



def gerar_ABT():
    import pandas as pd
    from src.constantes import pasta_data_04_load_inscritos,pasta_data_05_processed_analise_016
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

    # Novo Filtro de Treino: Incluindo menos 2021-2
    filtro_treino_base = ((df_base['ano'] == 2019) | ((df_base['ano'] == 2020)) | ((df_base['ano'] == 2021) & (df_base['semestre'] == 1))) & (df_base['subarea_conhecimento'] == 'MEDICINA')

    # Novo Filtro de Teste: 2021-2
    filtro_teste_base = ((df_base['ano'] == 2021) & (df_base['semestre'] == 2)) & (df_base['subarea_conhecimento'] == 'MEDICINA')

    print('SHAPE de 2019-1 ate 2021-1 ', df_base[filtro_treino_base].shape)

    print('SHAPE de apenas 2021-2', df_base[filtro_teste_base].shape)


    print('-'*60)
    print('total de linhas de 2019-1 ate 2021-1 ')
    display(df_base[filtro_treino_base].groupby('situacao_fies',as_index=False).size().rename(columns={'size':'qtde_inscritos'}))#type: ignore


    print('-'*60)
    print('total de linhas de apenas 2021-2')

    display(df_base[filtro_teste_base].groupby('situacao_fies',as_index=False).size().rename(columns={'size':'qtde_inscritos'}))#type: ignore


    # 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
    # Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
    df = pd.get_dummies(df_base, columns=['beneficiado_creduc_fies','modalidade_fies','subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior'], drop_first=True)

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
    colunas_dummies_conceito_concluiu_modfies = [col for col in df.columns if col.startswith('modalidade_fies')]
    colunas_dummies_conceito_concluiu_BENEFcreduc = [col for col in df.columns if col.startswith('beneficiado_creduc_fies')]




    

    # 9. CRIAR A LISTA EXATA DE FEATURES
    # Juntamos as variáveis que você escolheu com as dummies
    df['renda_gap']=df['renda_per_capita'] * df['gap']

    
    features_exatas = ['renda_per_capita', 'gap','idade','nota_corte_gp','renda_gap']+ colunas_dummies_conceito_concluiu_BENEFcreduc + colunas_dummies_conceito_concluiu_modfies  + colunas_dummies_conceito_concluiu_curso_superior + colunas_dummies_conceito_curso + colunas_dummies_cine + colunas_dummies_regiao + colunas_dummies_natureza + colunas_dummies_etnia + colunas_dummies_turno + colunas_dummies_ensino_medio_escola_publica



    # Faz o Raio-X apenas nas colunas que vão pro modelo
    print("--- Quantidade de NaNs por coluna ---")
    print(df[features_exatas].isna().sum())

    # Isso garante que se o aluno sumir, ele some das duas listas
    df_limpo = df.dropna(subset=features_exatas).copy()

    #filtro_treino= (df_limpo['ano'] == 2019) & ( (df_limpo['semestre'] == 1) | (df_limpo['semestre'] == 2) )
    #filtro_teste= ~(df_limpo['ano'] == 2019) & ( (df_limpo['semestre'] == 1) | (df_limpo['semestre'] == 2) )

    # Novo Filtro de Treino: Incluindo 2020-1
    filtro_treino = ((df_limpo['ano'] == 2019) | ((df_limpo['ano'] == 2020)) | ((df_limpo['ano'] == 2021) & (df_limpo['semestre'] == 1))) & (df_limpo['subarea_conhecimento_MEDICINA'] == 1)

    # Novo Filtro de Teste: 2020-2 em diante
    filtro_teste = ((df_limpo['ano'] == 2021) & (df_limpo['semestre'] == 2)) & (df_limpo['subarea_conhecimento_MEDICINA'] == 1)


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


    print('coluna x linha do X_treino com NaN inclusos:',X.shape)

    # %% 
    # 12. SALVAR MATRIZES PROCESSADAS (ABT)
    import os

    # Definindo o diretório de destino
    caminho_salvar = str(pasta_data_05_processed_analise_016)

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
    print('qtde de medicina:')
    display(df_treino.groupby(['ano','semestre','subarea_conhecimento_MEDICINA']).agg(total_inscritos=('subarea_conhecimento_MEDICINA','count')))#type: ignore

    return X,Y,X_teste,Y_teste,df_base



def TreinarModelo(treino_ou_teste_veio: str, x_treino, y_treino):

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from src.constantes import pasta_modelo_analise_016

    X = x_treino
    Y = y_treino

    # pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('modelo', LogisticRegression(
    #         penalty=None,
    #         solver='lbfgs',
    #         max_iter=10000
    #     ))
    # ])

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

    joblib.dump(pipeline, str(pasta_modelo_analise_016))
    print(f'Modelo salvo em: {str(pasta_modelo_analise_016)}')

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
    from src.constantes import pasta_modelo_analise_016

    X = x_treino if treino_ou_teste_veio == 'veio_treino' else x_teste

    modelo = modelo_vindo if usar_modelo == 'simm' \
        else joblib.load(str(pasta_modelo_analise_016))

    print('Shape de X nesta analise: ',X.shape)
    chances = modelo.predict_proba(X)[:, 1]

    # GRÁFICO RENDA
    plt.figure(figsize=(12, 6))
    plt.scatter(X['renda_per_capita'], chances, alpha=0.3, s=1)
    plt.title('Probabilidade vs Renda')
    plt.xlabel('Renda per capita')
    plt.ylabel('Probabilidade')
    plt.show()

    # GRÁFICO GAP
    plt.figure(figsize=(12, 6))
    plt.scatter(X['gap'], chances, alpha=0.3, s=1)
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
    from src.constantes import pasta_modelo_analise_016

    X = x_treino if treino_ou_teste_veio == 'veio_treino' else x_teste

    modelo = modelo_vindo if usar_modelo == 'simm' \
        else joblib.load(str(pasta_modelo_analise_016))

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

    coef_renda = tabela_importancia.loc[
        tabela_importancia['Variável'] == 'renda_per_capita',
        'Coeficiente'
    ].values[0]

    coef_gap = tabela_importancia.loc[
        tabela_importancia['Variável'] == 'gap',
        'Coeficiente'
    ].values[0]

    coef_renda_gap = tabela_importancia.loc[
        tabela_importancia['Variável'] == 'renda_gap',
        'Coeficiente'
    ].values[0]

    print('coeficente renda_gap', coef_renda_gap)
    

    print('----- Efeito da RENDA dependendo do GAP -----')

    for gap in [-1, 0, 1]:
    
        efeito_renda = coef_renda + coef_renda_gap * gap
        
        print(f"GAP {gap}: efeito da renda = {efeito_renda}")


    print('----- Efeito do GAP dependendo da RENDA -----')

    for renda in [-1, 0, 1]:
    
        efeito_gap = coef_gap + coef_renda_gap * renda
        
        print(f"RENDA {renda}: efeito do gap = {efeito_gap}")
    
    import matplotlib.pyplot as plt

    gaps = [-1,0,1]
    efeitos_renda = []

    for gap in gaps:
        efeito = coef_renda + coef_renda_gap * gap
        efeitos_renda.append(efeito)

    plt.figure()

    plt.plot(gaps, efeitos_renda, marker='o')

    plt.xlabel("GAP (padronizado)")
    plt.ylabel("Efeito da renda no logit")
    plt.title("Efeito da renda dependendo do GAP")

    plt.axhline(0)

    plt.show()


    rendas = [-1,0,1]
    efeitos_gap = []

    for renda in rendas:
        efeito = coef_gap + coef_renda_gap * renda
        efeitos_gap.append(efeito)

    plt.figure()

    plt.plot(rendas, efeitos_gap, marker='o')

    plt.xlabel("Renda (padronizada)")
    plt.ylabel("Efeito do GAP no logit")
    plt.title("Efeito do GAP dependendo da renda")

    plt.axhline(0)

    plt.show()


    import numpy as np

    gaps = np.linspace(-2,2,100)

    plt.figure()

    for renda in [-1,0,1]:
        
        logit = coef_gap * gaps + coef_renda * renda + coef_renda_gap * renda * gaps
        
        plt.plot(gaps, logit, label=f"Renda {renda}")

    plt.xlabel("GAP")
    plt.ylabel("Logit")
    plt.title("Interação entre renda e gap")

    plt.legend()

    plt.show()



    base = pd.DataFrame({
    "renda_per_capita":[-1,-1,-1,0,0,0,1,1,1],
    "gap":[-1,0,1,-1,0,1,-1,0,1],
    "idade":[0]*9,
    "nota_corte_gp":[0]*9
    })

    base["renda_gap"] = base["renda_per_capita"] * base["gap"]

    # preencher dummies faltantes
    for col in X.columns:
        if col not in base.columns:
            base[col] = 0

    base = base[X.columns]

    print(modelo.predict_proba(base)[:,1])
            
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # valores simulados
    rendas = [-1, 0, 1]
    gaps = [-1, 0, 1]

    linhas = []

    for renda in rendas:
        for gap in gaps:
            
            linha = {
                "renda_per_capita": renda,
                "gap": gap,
                "idade": 0,
                "nota_corte_gp": 0,
                "renda_gap": renda * gap
            }

            # todas as outras dummies = 0
            for col in X.columns:
                if col not in linha:
                    linha[col] = 0

            linhas.append(linha)

    base_simulada = pd.DataFrame(linhas)

    # garantir mesma ordem das features do treino
    base_simulada = base_simulada[X.columns]

    # probabilidades
    probs = modelo.predict_proba(base_simulada)[:,1]

    base_simulada["prob"] = probs

    # criar matriz para heatmap
    matriz = base_simulada.pivot(
        index="renda_per_capita",
        columns="gap",
        values="prob"
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        matriz,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1
    )

    plt.title("Probabilidade de contratação\n(interação renda × gap)")
    plt.xlabel("GAP")
    plt.ylabel("Renda per capita")

    plt.show()


    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # nomes mais intuitivos
    labels_gap = ["Nota abaixo da corte", "Nota próxima da corte", "Nota acima da corte"]
    labels_renda = ["Renda baixa", "Renda média", "Renda alta"]

    gaps = [-1,0,1]

    efeitos_renda = []

    for gap in gaps:
        efeito = coef_renda + coef_renda_gap * gap
        efeitos_renda.append(efeito)

    plt.figure(figsize=(7,5))

    plt.plot(labels_gap, efeitos_renda, marker='o')

    plt.xlabel("Desempenho em relação à nota de corte")
    plt.ylabel("Impacto da renda no modelo")
    plt.title("Impacto da renda dependendo do desempenho do candidato")

    plt.axhline(0)

    plt.show()


    # ----------------------------

    rendas = [-1,0,1]

    efeitos_gap = []

    for renda in rendas:
        efeito = coef_gap + coef_renda_gap * renda
        efeitos_gap.append(efeito)

    plt.figure(figsize=(7,5))

    plt.plot(labels_renda, efeitos_gap, marker='o')

    plt.xlabel("Nível de renda do candidato")
    plt.ylabel("Impacto do desempenho no modelo")
    plt.title("Impacto da nota dependendo do nível de renda")

    plt.axhline(0)

    plt.show()


    # ----------------------------

    gaps = np.linspace(-2,2,100)

    plt.figure(figsize=(8,6))

    for renda,label in zip([-1,0,1],labels_renda):
        
        logit = coef_gap * gaps + coef_renda * renda + coef_renda_gap * renda * gaps
        
        plt.plot(gaps, logit, label=label)

    plt.xlabel("Desempenho relativo à nota de corte")
    plt.ylabel("Score do modelo")
    plt.title("Interação entre renda do candidato e desempenho acadêmico")

    plt.legend(title="Perfil de renda")

    plt.show()


    # ----------------------------
    # Simulação de probabilidades

    rendas = [-1, 0, 1]
    gaps = [-1, 0, 1]

    linhas = []

    for renda in rendas:
        for gap in gaps:
            
            linha = {
                "renda_per_capita": renda,
                "gap": gap,
                "idade": 0,
                "nota_corte_gp": 0,
                "renda_gap": renda * gap
            }

            for col in X.columns:
                if col not in linha:
                    linha[col] = 0

            linhas.append(linha)

    base_simulada = pd.DataFrame(linhas)

    base_simulada = base_simulada[X.columns]

    probs = modelo.predict_proba(base_simulada)[:,1]

    base_simulada["prob"] = probs

    matriz = base_simulada.pivot(
        index="renda_per_capita",
        columns="gap",
        values="prob"
    )

    # renomear eixos
    matriz.index = labels_renda
    matriz.columns = labels_gap

    plt.figure(figsize=(8,6))

    sns.heatmap(
        matriz,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1
    )

    plt.title("Probabilidade estimada de contratação\nsegundo renda e desempenho do candidato")

    plt.xlabel("Desempenho em relação à nota de corte")
    plt.ylabel("Perfil de renda")

    plt.show()




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
    from src.constantes import pasta_modelo_analise_016


    X_teste = x_treino
    Y_teste = y_treino
    #X_teste = x_teste
    #Y_teste = y_teste

    modelo = modelo_vindo if usar_modelo == 'simm' \
        else joblib.load(str(pasta_modelo_analise_016))

    # 1️⃣ Probabilidades
    chances = modelo.predict_proba(X_teste)[:, 1]

    # 2️⃣ ROC-AUC (agora correto)
    auc = roc_auc_score(Y_teste, chances)
    print(f"ROC-AUC: {auc:.4f}")

    # 3️⃣ Threshold ótimo
    fpr, tpr, thresholds = roc_curve(Y_teste, chances)
    melhor_indice = (tpr - fpr).argmax()
    corte = thresholds[melhor_indice]

    #corte = 0.53

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

