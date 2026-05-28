# # %%
# import pandas as pd

# from constantes import pasta_data_04_load_inscritos

# # Configuração de visualização
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# # 1. CARREGAR OS DADOS
# df_base = pd.read_parquet(str(pasta_data_04_load_inscritos))

# # 2. CRIAR VARIÁVEIS NOVAS (Vetorizado, muito mais rápido)
# #df_base['gap'] = df_base['media_enem'] - df_base['nota_corte_gp']


# df = (df_base.groupby(['subarea_conhecimento'],as_index=False)
# .agg(qtde=('subarea_conhecimento','count'))
# .sort_values(['qtde'],ascending=False))

# display(df)#type: ignore
# # %%


def orquestrador_inicial_inscritos_19():

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


def orquestrador_ja_rodado_inscritos_19():
    import pandas as pd
    import joblib
    from src.constantes import (
        pasta_data_05_processed_analise_019_X_treino, 
        pasta_data_05_processed_analise_019_y_treino,
        pasta_modelo_analise_019,
        pasta_data_05_processed_analise_019_X_teste,
        pasta_data_05_processed_analise_019_y_teste,
        pasta_data_04_load_inscritos
    )

    X= pd.read_parquet(str(pasta_data_05_processed_analise_019_X_treino))
    Y= pd.read_parquet(str(pasta_data_05_processed_analise_019_y_treino))
    X_teste= pd.read_parquet(str(pasta_data_05_processed_analise_019_X_teste))
    Y_teste= pd.read_parquet(str(pasta_data_05_processed_analise_019_y_teste))

    df_base = pd.read_parquet(str(pasta_data_04_load_inscritos))

    modelo = joblib.load(str(pasta_modelo_analise_019))
    analises_e_datasets('veio_treino', 'simm', modelo, X, Y, X_teste, Y_teste, df_inicial=df_base)
    prever_probabilidade_treino('veio_treino', 'simm', modelo, X, Y, X_teste, Y_teste)
    acuracia_e_previsao('veio', 'simm', modelo, X, Y, X_teste, Y_teste)



def gerar_ABT():
    import pandas as pd
    from src.constantes import pasta_data_04_load_inscritos,pasta_data_05_processed_analise_019
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


    print('SHAPE de todos os anos', df_base.shape)

    print('SHAPE de todos os anos', df_base.shape)


    print('-'*60)
    print('total de linhas')
    display(df_base.groupby('situacao_fies',as_index=False).size().rename(columns={'size':'qtde_inscritos'}))#type: ignore


    print('-'*60)
    print('total de linhas')

    display(df_base.groupby('situacao_fies',as_index=False).size().rename(columns={'size':'qtde_inscritos'}))#type: ignore


    # 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
    # Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
    df = pd.get_dummies(df_base, columns=['beneficiado_creduc_fies','modalidade_fies','subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior','opcao_curso'], drop_first=True)

    # 1. Limpar a coluna: remover espaços e colocar tudo em maiúsculas
    df['situacao_fies'] = df['situacao_fies'].str.strip().str.upper()


    #########################################################
    
    # 2. Mapear cada situação para número
    mapa_situacao = {
        'CONTRATADA': 2,
        'NÃO CONTRATADO': 1,
        'LISTA DE ESPERA': 0,
    }

    df['contratado'] = df['situacao_fies'].map(mapa_situacao)

    # 3. Filtrar linhas que não foram mapeadas (NaN)
    df = df[df['contratado'].notna()].copy()

    # 4. Transformar em inteiro
    df['contratado'] = df['contratado'].astype(int)


    #########################################################
    
    #     # 5. Mapear cada situação para número SOMENTE BINARIO
    # mapa_situacao = {
    #     'CONTRATADA': 1,
    #     'NÃO CONTRATADO': 0,
    #     #'LISTA DE ESPERA': 0,
    # }

    # df['contratado'] = df['situacao_fies'].map(mapa_situacao)

    # # 3. Filtrar linhas que não foram mapeadas (NaN)
    # df = df[df['contratado'].notna()].copy()

    # # 4. Transformar em inteiro
    # df['contratado'] = df['contratado'].astype(int)

    #########################################################





    # 6. PESCAR AS COLUNAS DUMMIES
    # O Python procura e guarda apenas as colunas que começam com esse nome
    colunas_dummies_turno = [col for col in df.columns if col.startswith('turno_')]
    
    colunas_dummies_ensino_medio_escola_publica = [col for col in df.columns if col.startswith('ensino_medio_escola_publica_')]
    colunas_dummies_etnia = [col for col in df.columns if col.startswith('etnia_cor_')]
    colunas_dummies_cine = [col for col in df.columns if col.startswith('subarea_conhecimento_')]
    colunas_dummies_regiao = [col for col in df.columns if col.startswith('regiao_morar_')]
    colunas_dummies_natureza = [col for col in df.columns if col.startswith('natureza_juridica_mantenedora_')]
    colunas_dummies_opcao_curso = [col for col in df.columns if col.startswith('opcao_curso')]
    colunas_dummies_conceito_curso = [col for col in df.columns if col.startswith('conceito_curso_gp')]
    colunas_dummies_conceito_concluiu_curso_superior = [col for col in df.columns if col.startswith('concluiu_curso_superior')]
    colunas_dummies_conceito_concluiu_modfies = [col for col in df.columns if col.startswith('modalidade_fies')]
    colunas_dummies_conceito_concluiu_BENEFcreduc = [col for col in df.columns if col.startswith('beneficiado_creduc_fies')]




    

    # 9. CRIAR A LISTA EXATA DE FEATURES
    # Juntamos as variáveis que você escolheu com as dummies
    df['renda_gap']=df['renda_per_capita'] * df['gap']

    
    features_exatas = ['renda_per_capita', 'gap','idade','nota_corte_gp','renda_gap']+ colunas_dummies_conceito_concluiu_BENEFcreduc + colunas_dummies_conceito_concluiu_modfies + colunas_dummies_opcao_curso + colunas_dummies_conceito_concluiu_curso_superior + colunas_dummies_conceito_curso + colunas_dummies_cine + colunas_dummies_regiao + colunas_dummies_natureza + colunas_dummies_etnia + colunas_dummies_turno + colunas_dummies_ensino_medio_escola_publica



    # Faz o Raio-X apenas nas colunas que vão pro modelo
    print("--- Quantidade de NaNs por coluna ---")
    print(df[features_exatas].isna().sum())

    # Isso garante que se o aluno sumir, ele some das duas listas
    df_limpo = df.dropna(subset=features_exatas).copy()



    df_treino = df_limpo
    df_teste = df_limpo


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
    caminho_salvar = str(pasta_data_05_processed_analise_019)

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
    from src.constantes import pasta_modelo_analise_019

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
            n_jobs=-1,
        ))
    ])

    pipeline.fit(X, Y.values.ravel())

    print('--- Treinamento Concluído ---')

    joblib.dump(pipeline, str(pasta_modelo_analise_019))
    print(f'Modelo salvo em: {str(pasta_modelo_analise_019)}')

    return pipeline


def prever_probabilidade_treino(
    treino_ou_teste_veio: str,
    usar_modelo: str,
    modelo_vindo,
    x_treino, y_treino,
    x_teste, y_teste
):
    import matplotlib.pyplot as plt

    # Captando as variáveis do orquestrador
    X = x_treino if treino_ou_teste_veio == 'veio_treino' else x_teste

    modelo = modelo_vindo if usar_modelo == 'simm' else None  # já vem do orquestrador

    print('Shape de X nesta analise: ', X.shape)
    chances = modelo.predict_proba(X)[:, 1]

    # Gráficos
    plt.figure(figsize=(12, 6))
    plt.scatter(X['renda_per_capita'], chances, alpha=0.1, s=1)
    plt.title('Probabilidade vs Renda')
    plt.xlabel('Renda per capita')
    plt.ylabel('Probabilidade')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.scatter(X['gap'], chances, alpha=0.1, s=1)
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
    import matplotlib.pyplot as plt
    import numpy as np

    # Captando variáveis do orquestrador
    X = x_treino if treino_ou_teste_veio == 'veio_treino' else x_teste
    modelo = modelo_vindo if usar_modelo == 'simm' else None  # já vem do orquestrador

    # Coeficientes do modelo logístico
    modelo_logistico = modelo.named_steps['modelo']
    coeficientes = modelo_logistico.coef_[0]

    # Tabela de importância das variáveis
    tabela_importancia = pd.DataFrame({
        'Variável': X.columns,
        'Coeficiente': coeficientes
    })
    tabela_importancia['Impacto_Absoluto'] = tabela_importancia['Coeficiente'].abs()
    tabela_importancia = tabela_importancia.sort_values(by='Impacto_Absoluto', ascending=False)

    print("Top 20 variáveis mais importantes:")
    display(tabela_importancia.head(20))

    # Captura coeficientes principais para interação renda x gap
    coef_renda = tabela_importancia.loc[tabela_importancia['Variável'] == 'renda_per_capita', 'Coeficiente'].values[0]
    coef_gap = tabela_importancia.loc[tabela_importancia['Variável'] == 'gap', 'Coeficiente'].values[0]
    coef_renda_gap = tabela_importancia.loc[tabela_importancia['Variável'] == 'renda_gap', 'Coeficiente'].values[0]

    print('Coeficiente renda_gap:', coef_renda_gap)

    # -------------------------------
    # 1️⃣ Efeito da Renda dependendo do GAP
    # -------------------------------
    gaps = [-1,0,1]  # abaixo da corte, próxima da corte, acima da corte
    efeitos_renda = [coef_renda + coef_renda_gap * gap for gap in gaps]

    plt.figure(figsize=(7,5))
    plt.plot(["Nota abaixo da corte","Nota próxima da corte","Nota acima da corte"], efeitos_renda, marker='o', color='blue')
    plt.xlabel("Desempenho em relação à nota de corte")
    plt.ylabel("Impacto da Renda")
    plt.title("Impacto da Renda dependendo do GAP")
    plt.axhline(0, color='gray', linestyle='--')
    plt.show()

    # -------------------------------
    # 2️⃣ Efeito do GAP dependendo da Renda
    # -------------------------------
    # Escolha de rendas representativas (ajuste conforme seus dados)
    medias_renda = [1000, 3000, 5000]  # baixa, média e alta
    efeitos_gap = [coef_gap + coef_renda_gap * renda for renda in medias_renda]

    plt.figure(figsize=(7,5))
    plt.plot(["Renda baixa","Renda média","Renda alta"], efeitos_gap, marker='o', color='green')
    plt.xlabel("Renda per capita")
    plt.ylabel("Impacto do GAP")
    plt.title("Impacto do GAP dependendo da Renda")
    plt.axhline(0, color='gray', linestyle='--')
    plt.show()

def acuracia_e_previsao(
    veio_datasets_x_e_y: str,
    usar_modelo: str,
    modelo_vindo,
    x_treino, y_treino,
    x_teste, y_teste
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

    # Captando variáveis do orquestrador
    X_teste = x_treino if veio_datasets_x_e_y == 'veio' else x_teste
    Y_teste = y_treino if veio_datasets_x_e_y == 'veio' else y_teste

    modelo = modelo_vindo if usar_modelo == 'simm' else None  # já vem do orquestrador

    # Probabilidades para cada classe (shape = n_amostras x n_classes)
    chances = modelo.predict_proba(X_teste)

    # ROC-AUC multiclass (OvR)
    auc = roc_auc_score(Y_teste, chances, multi_class='ovr')
    print(f"ROC-AUC (multiclass, OvR): {auc:.4f}")

    # Predição usando argmax das probabilidades
    Y_previsto = chances.argmax(axis=1)

    # Acurácia
    acuracia = accuracy_score(Y_teste, Y_previsto)
    print(f"Acurácia: {acuracia:.4f}")

    # Matriz de confusão
    matriz = confusion_matrix(Y_teste, Y_previsto)
    plt.figure(figsize=(6, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusão (Multiclass)")
    plt.show()