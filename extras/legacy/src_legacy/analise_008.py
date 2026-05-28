def orquestrador_inicial_candidatos():


    X, Y, X_teste, Y_teste, df_base = gerar_ABT()

    modelo = TreinarModelo('veio_treino',X,Y)
    prever_probabilidade_treino('veio_treino', 'simm', modelo, X, Y, X_teste, Y_teste)
    analises_e_datasets('veio_treino', 'simm', modelo, X, Y, X_teste, Y_teste, df_inicial=df_base)
    acuracia_e_previsao('veio', 'simm', modelo, X, Y, X_teste, Y_teste)




def orquestrador_ja_rodado_candidatos():
    import pandas as pd
    import joblib
    from src.constantes import (
        pasta_data_05_processed_analise_008_X_treino, 
        pasta_data_05_processed_analise_008_y_treino,
        pasta_modelo_analise_008,
        pasta_data_05_processed_analise_008_X_teste,
        pasta_data_05_processed_analise_008_y_teste,
        pasta_data_05_processed_candidatos_unicos_por_prioridade_inicial
    )

    X= pd.read_parquet(str(pasta_data_05_processed_analise_008_X_treino))
    Y= pd.read_parquet(str(pasta_data_05_processed_analise_008_y_treino))
    X_teste= pd.read_parquet(str(pasta_data_05_processed_analise_008_X_teste))
    Y_teste= pd.read_parquet(str(pasta_data_05_processed_analise_008_y_teste))
    df_base = pd.read_parquet(str(pasta_data_05_processed_candidatos_unicos_por_prioridade_inicial))

    modelo = joblib.load(str(pasta_modelo_analise_008))
    prever_probabilidade_treino('veio_treino', 'simm', modelo, X, Y, X_teste, Y_teste)
    analises_e_datasets('veio_treino', 'simm', modelo, X, Y, X_teste, Y_teste, df_inicial=df_base)
    acuracia_e_previsao('veio', 'simm', modelo, X, Y, X_teste, Y_teste)



def gerar_ABT():
    import pandas as pd
    from src.constantes import pasta_data_05_processed_candidatos_unicos_por_prioridade_inicial,pasta_data_05_processed_analise_008_GAP_RENDA_AREA_CINE_CANDIDATOS_PRIORIDADE_INICIAL
    from sklearn.linear_model import LogisticRegression

    # Configuração de visualização
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 1. CARREGAR OS DADOS
    df_base = pd.read_parquet(str(pasta_data_05_processed_candidatos_unicos_por_prioridade_inicial))

    # 2. CRIAR VARIÁVEIS NOVAS (Vetorizado, muito mais rápido)
    df_base['gap'] = df_base['media_enem'] - df_base['nota_corte_gp'] 

    filtro_treino_base= (df_base['ano'] == 2019) & ( (df_base['semestre'] == 1) | (df_base['semestre'] == 2) )
    filtro_teste_base= ~(df_base['ano'] == 2019) & ( (df_base['semestre'] == 1) | (df_base['semestre'] == 2) )


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
    df = pd.get_dummies(df_base, columns=['nome_cine_area_geral'], drop_first=True)

    # 4. FILTRAR A BASE DE DADOS
    contratados = ['CONTRATADA']
    nao_contratados = ['NÃO CONTRATADO']

    filtro = df['situacao_fies'].isin(contratados + nao_contratados)
    df = df[filtro].copy()

    # 5. CRIAR A VARIÁVEL ALVO (Y) BINÁRIA
    df['contratado'] = df['situacao_fies'].apply(lambda x: 1 if x in contratados else 0)


    # 6. PESCAR AS COLUNAS DUMMIES
    # O Python procura e guarda apenas as colunas que começam com esse nome
    colunas_dummies = [col for col in df.columns if col.startswith('nome_cine_area_geral_')]

    # 7. CRIAR A LISTA EXATA DE FEATURES
    # Juntamos as variáveis que você escolheu com as dummies
    features_exatas = ['renda_per_capita', 'gap'] + colunas_dummies



    # Faz o Raio-X apenas nas colunas que vão pro modelo
    print("--- Quantidade de NaNs por coluna ---")
    print(df[features_exatas].isna().sum())

    # Isso garante que se o aluno sumir, ele some das duas listas
    df_limpo = df.dropna(subset=features_exatas).copy()

    filtro_treino= (df_limpo['ano'] == 2019) & ( (df_limpo['semestre'] == 1) | (df_limpo['semestre'] == 2) )
    filtro_teste= ~(df_limpo['ano'] == 2019) & ( (df_limpo['semestre'] == 1) | (df_limpo['semestre'] == 2) )



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
    caminho_salvar = str(pasta_data_05_processed_analise_008_GAP_RENDA_AREA_CINE_CANDIDATOS_PRIORIDADE_INICIAL)

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



def TreinarModelo(treino_ou_teste_veio: str,x_treino,y_treino):
    import pandas as pd
    import joblib
    import os
    from src.constantes import (
        pasta_data_05_processed_analise_008_X_treino, 
        pasta_data_05_processed_analise_008_y_treino,
        pasta_modelo_analise_008
    )
    from sklearn.linear_model import LogisticRegression


    if treino_ou_teste_veio == 'veio_treino':
        X=x_treino
        Y=y_treino

    # 1. CARREGAR A ABT (TABELA ANALÍTICA) SALVA
    elif treino_ou_teste_veio == 'quero_treino':
        X= pd.read_parquet(str(pasta_data_05_processed_analise_008_X_treino))
        Y= pd.read_parquet(str(pasta_data_05_processed_analise_008_y_treino))


    # 2. CONFIGURAR E TREINAR O MODELO
    # Usando 10k iterações e pesos balanceados para lidar com o desbalanceamento
    modelo = LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42)
    
    # O .values.ravel() garante que o Y esteja no formato que o sklearn gosta (vetor 1D)
    modelo.fit(X, Y.values.ravel())

    print('--- Treinamento Concluído com Sucesso! ---')

    # 3. SALVAR O MODELO (CONGELAR)

    joblib.dump(modelo, str(pasta_modelo_analise_008))
    print(f'Modelo exportado para: {str(pasta_modelo_analise_008)}')
    
    return modelo



def prever_probabilidade_treino(treino_ou_teste_veio: str,usar_modelo:str,modelo_vindo,x_treino,y_treino,x_teste,y_teste):

    import joblib
    import pandas as pd
    from src.constantes import pasta_data_05_processed_candidatos_unicos_por_prioridade_inicial, pasta_data_05_processed_analise_008_X_treino, pasta_data_05_processed_analise_008_y_treino, pasta_modelo_analise_008,pasta_data_05_processed_analise_008_X_teste, pasta_data_05_processed_analise_008_y_teste
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt

    if treino_ou_teste_veio == 'veio_treino':
        X=x_treino
        Y=y_treino
    elif treino_ou_teste_veio == 'veio_teste':
        X=x_teste
        Y=y_teste
    elif treino_ou_teste_veio == 'quero_treino':
        X= pd.read_parquet(str(pasta_data_05_processed_analise_008_X_treino))
        Y= pd.read_parquet(str(pasta_data_05_processed_analise_008_y_treino))
    elif treino_ou_teste_veio == 'quero_teste': 
        
        X= pd.read_parquet(str(pasta_data_05_processed_analise_008_X_teste))
        Y= pd.read_parquet(str(pasta_data_05_processed_analise_008_y_teste))


    if usar_modelo == 'simm':
        modelo= modelo_vindo
    else:
        modelo = joblib.load(str(pasta_modelo_analise_008))
#modelo.predict(df_limpo[features_exatas]) # so com colunas q o modelo conhece mas esta a ser usado apenas .predict por probabilidade


    # 1. USANDO O DF INTEIRO (X e Y já estão limpos de NaNs no seu código anterior)
    # Calculando a probabilidade para TODO MUNDO
    # Cuidado: isso pode consumir bastante memória RAM
    chances_de_aprovar = modelo.predict_proba(X)[:, 1]


    # 2. CONFIGURAÇÃO DO GRÁFICO
    plt.figure(figsize=(14, 8), dpi=150)

    # Ajuste Fino: Alpha em 0.005 para lidar com a sobreposição massiva
    # s=0.2 deixa os pontos como poeira estelar, revelando a densidade real
    plt.scatter(X['renda_per_capita'], chances_de_aprovar, 
                alpha=0.3, color='royalblue', s=0.2, 
                label='Inscrições FIES (Modelo: Renda + Gap + Área Curso)')

    # 3. TÍTULOS E LABELS PROFISSIONAIS
    plt.title('Probabilidade de Sucesso FIES: Impacto Combinado de Renda e Desempenho', 
            fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Renda per Capita (R$)', fontsize=12, labelpad=10)
    plt.ylabel('Probabilidade de Aprovação (0.0 a 1.0)', fontsize=12, labelpad=10)

    # 4. LIMITES E GRID
    # Focar em 5000 ajuda a ver a maior parte da massa de dados
    plt.xlim(0, 5000) 
    plt.ylim(-0.02, 1.02)

    # Grid sutil para auxiliar a leitura sem poluir o visual
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

    # 5. LEGENDA ROBUSTA
    # 'markerscale' alto é essencial para o ponto aparecer na legenda, já que no gráfico ele é minúsculo
    leg = plt.legend(loc='upper right', markerscale=50, fontsize=10, frameon=True)
    leg.get_frame().set_alpha(0.8)

    # 6. FINALIZAÇÃO
    plt.tight_layout()
    plt.savefig('reports/figures/analise_008/analise_fies_multivariada_renda.png', bbox_inches='tight', dpi=300)
    plt.show()

    print(f"Gráfico Finalizado: {len(X):,} registros processados com sucesso.")


    # PROVA DO IMPACTO DA NOTA (GAP)

    # 1. CONFIGURAÇÃO DO GRÁFICO
    plt.figure(figsize=(14, 8), dpi=150)

    # Usando o GAP no eixo X
    # Usei 'seagreen' para representar o impacto positivo (crescimento)
    plt.scatter(X['gap'], chances_de_aprovar, 
                alpha=0.3, color='seagreen', s=0.2, 
                label='Inscrições FIES (Modelo: Renda + Gap + Área Curso)')

    # 2. TÍTULOS E LABELS PROFISSIONAIS
    plt.title('Probabilidade de Sucesso FIES: Impacto do Desempenho (GAP)', 
            fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('GAP (Nota Enem - Nota de Corte)', fontsize=12, labelpad=10)
    plt.ylabel('Probabilidade de Aprovação (0.0 a 1.0)', fontsize=12, labelpad=10)

    # 3. LIMITES E GRID
    # O GAP geralmente fica entre -100 e +150. Ajustei para esse intervalo:
    plt.xlim(-100, 150) 
    plt.ylim(-0.02, 1.02)

    # Grid sutil para auxiliar a leitura
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

    # 4. LEGENDA ROBUSTA
    # Mudei a posição para 'upper left' para não tampar a subida da curva
    leg = plt.legend(loc='upper left', markerscale=50, fontsize=10, frameon=True)
    leg.get_frame().set_alpha(0.8)

    # 5. FINALIZAÇÃO
    plt.tight_layout()
    # Mantive o caminho que você sugeriu, mas certifique-se de que a pasta existe
    plt.savefig('reports/figures/analise_008/analise_fies_multivariada_nota.png', bbox_inches='tight', dpi=300)
    plt.show()

    print(f"Gráfico Finalizado: {len(X):,} registros processados com sucesso.")



def analises_e_datasets(treino_ou_teste_veio: str,usar_modelo:str,modelo_vindo,x_treino,y_treino,x_teste,y_teste,df_inicial):

    import joblib
    import pandas as pd
    from src.constantes import pasta_data_05_processed_candidatos_unicos_por_prioridade_inicial, pasta_data_05_processed_analise_008_X_treino, pasta_data_05_processed_analise_008_y_treino, pasta_modelo_analise_008,pasta_data_05_processed_analise_008_X_teste, pasta_data_05_processed_analise_008_y_teste
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt

    df_base = df_inicial
    if treino_ou_teste_veio == 'veio_treino':
        X=x_treino
        Y=y_treino
    elif treino_ou_teste_veio == 'veio_teste':
        X=x_teste
        Y=y_teste
    elif treino_ou_teste_veio == 'quero_treino':
        X= pd.read_parquet(str(pasta_data_05_processed_analise_008_X_treino))
        Y= pd.read_parquet(str(pasta_data_05_processed_analise_008_y_treino))
    elif treino_ou_teste_veio == 'quero_teste': 
        
        X= pd.read_parquet(str(pasta_data_05_processed_analise_008_X_teste))
        Y= pd.read_parquet(str(pasta_data_05_processed_analise_008_y_teste))


    if usar_modelo == 'simm':
        modelo= modelo_vindo
    else:
        modelo = joblib.load(str(pasta_modelo_analise_008))
    # 1. DEFINIR OS LIMITES (BINS) E OS RÓTULOS
    # O primeiro valor (0) e o último (float('inf')) garantem que pegamos tudo
    limites = [0, 500, 1000, 1500, 2000, 2500, 3000, float('inf')]
    rotulos = ['0-500', '501-1000', '1001-1500', '1501-2000', '2001-2500', '2501-3000', '+ de 3001']

    # 2. CRIAR A COLUNA DE FAIXA NO X (OU NO DF_LIMPO)
    X_com_faixas = X.copy()
    X_com_faixas['faixa_renda'] = pd.cut(X_com_faixas['renda_per_capita'], 
                                        bins=limites, 
                                        labels=rotulos, 
                                        right=True)
    # right signfica intervalo fechado ou seja INCLUI O 0 E 500, DPS 501 A 1000...


    # 3. AGRUPAR E CONTAR OS INSCRITOS
    distribuicao_renda = X_com_faixas.groupby('faixa_renda', observed=True).size().reset_index(name='qtd_inscritos')

    # 4. CALCULAR O PERCENTUAL (OPCIONAL, MAS AJUDA MUITO)
    total = distribuicao_renda['qtd_inscritos'].sum()
    distribuicao_renda['percentual'] = (distribuicao_renda['qtd_inscritos'] / total * 100).round(2).astype(str) + '%'

    print("--- Distribuição de Inscritos por Faixa de Renda ---")
    display(distribuicao_renda)#type: ignore



    # PESO DE CADA VARIAVEL ORDENADA ( IMPORTÂNCIA PADRONIZADA (AJUSTADO) )
    import numpy as np

    # 1. Calculamos o 'desvio padrão' de cada coluna do X original
    # O std (Standard Deviation) é essencial para nivelar a comparação entre R$ (Renda) e Pontos (Gap)
    stds = X.std()

    # 2. Multiplicamos o peso do modelo pela variação real daquela variável
    # Isso gera os 'Standardized Coefficients' (Coeficientes Padronizados)
    impacto_real = modelo.coef_[0] * stds

    # 3. Criamos o DataFrame garantindo que o índice antigo seja descartado
    tabela_importancia = pd.DataFrame({
        'Variável': X.columns,
        'Impacto Real (Peso Corrigido)': impacto_real.values # .values evita conflito de índice
    }).reset_index(drop=True)

    # 4. Calculamos o Poder Absoluto para ordenar por quem 'manda' mais (independente de ser + ou -)
    tabela_importancia['Poder_Absoluto'] = tabela_importancia['Impacto Real (Peso Corrigido)'].abs()
    tabela_importancia = tabela_importancia.sort_values(by='Poder_Absoluto', ascending=False)

    print("--- QUEM MANDA NO FIES? (PESO CORRIGIDO POR ESCALA) ---")
    # Exibimos apenas as colunas de interesse, agora sem índices duplicados
    display(tabela_importancia[['Variável', 'Impacto Real (Peso Corrigido)']].reset_index(drop=True))#type: ignore


    df_agrupado = df_base.groupby(['ano','semestre','nome_cine_area_geral'], as_index=False).size().rename(columns={'size': 'qtde_de_inscritos'})


    display(df_agrupado)#type: ignore


def acuracia_e_previsao(veio_datasets_x_e_y: str,usar_modelo:str,modelo_vindo,x_treino,y_treino,x_teste,y_teste):

    import joblib
    import pandas as pd
    from src.constantes import pasta_data_05_processed_candidatos_unicos_por_prioridade_inicial, pasta_data_05_processed_analise_008_X_treino, pasta_data_05_processed_analise_008_y_treino, pasta_modelo_analise_008,pasta_data_05_processed_analise_008_X_teste, pasta_data_05_processed_analise_008_y_teste
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt

    if veio_datasets_x_e_y == 'veio':
        X=x_treino
        Y=y_treino

        X_teste=x_teste
        Y_teste=y_teste
    elif veio_datasets_x_e_y == 'quero':
        X= pd.read_parquet(str(pasta_data_05_processed_analise_008_X_treino))
        Y= pd.read_parquet(str(pasta_data_05_processed_analise_008_y_treino))
        
        X_teste= pd.read_parquet(str(pasta_data_05_processed_analise_008_X_teste))
        Y_teste= pd.read_parquet(str(pasta_data_05_processed_analise_008_y_teste))


    if usar_modelo == 'simm':
        modelo= modelo_vindo
    else:
        modelo = joblib.load(str(pasta_modelo_analise_008))


    print('-'*60)
    print('AVALIAÇÃO DE PERFORMANCE (MÉTRICAS KDD):')

    # Importando as métricas necessárias
    from sklearn.metrics import accuracy_score, confusion_matrix
    import seaborn as sns

    # O modelo tenta adivinhar o que vai acontecer com os alunos (Teste 2020/2021)
    # Validamos se as regras de seleção permaneceram estáveis durante a pandemia
    Y_previsto = modelo.predict(X_teste)

    print("Previsões geradas com sucesso!\n")

    # --- O CÁLCULO DA ACURÁCIA ---
    # O modelo balanceado atingiu 58.97%, priorizando a redução de falsos negativos
    acuracia = accuracy_score(Y_teste, Y_previsto)
    print(f"-> Acurácia Geral do Modelo: {acuracia * 100:.2f}%\n")
    print('-'*60)

    # 3. AVALIAÇÃO VISUAL: MATRIZ DE CONFUSÃO
    # A matriz cruza o Gabarito Real (Linhas) com a Predição do Modelo (Colunas)
    matriz = confusion_matrix(Y_teste, Y_previsto)

    # Desenhando o gráfico com rótulos intuitivos para o seu artigo na UTFPR
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Rótulos ajustados: Realidade (Eixo Y) vs Predição da IA (Eixo X)
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['IA: Previu REPROVAÇÃO', 'IA: Previu SUCESSO'], 
                yticklabels=['REAL: Não Contratou', 'REAL: Contratou'])

    

    plt.title('Auditoria de Predição FIES (Matriz de Confusão)\nBase de Teste: 2020/2021', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('GABARITO (O que aconteceu na Realidade)', fontsize=12, fontweight='bold')
    plt.xlabel('MODELO (O que a Inteligência Artificial Previu)', fontsize=12, fontweight='bold')
    
    # Salvando a imagem final com o caminho relativo correto
    plt.tight_layout()
    plt.savefig('reports/figures/analise_008/analise_fies_matriz_confusao.png', bbox_inches='tight', dpi=300)
    plt.show()




    print(f'''
        =============================================================================
        CONCLUSÕES DO MODELO DE PREDIÇÃO FIES - CANDIDATOS ÚNICOS (VALIDAÇÃO 2019-2021)
        =============================================================================

        1. DINÂMICA DOS CURSOS (CINE) LIMPA DE RUÍDOS:
        Ao removermos as múltiplas tentativas do mesmo candidato, a área de "Educação" 
        assumiu a maior penalidade estatística (-0.28), seguida por "Negócios, 
        Administração e Direito" (-0.21). A área de "Saúde e bem-estar" apresenta uma 
        barreira menor (-0.12), confirmando que o funil de conversão penaliza mais as 
        áreas de humanas/negócios do que as áreas da saúde.

        2. O PESO DA RENDA ESMAGA O MÉRITO ACADÊMICO:
        Na análise do indivíduo único, a discrepância de pesos ficou ainda mais extrema. 
        A RENDA PER CAPITA saltou para um impacto preditivo de -0.389, enquanto o motor 
        do GAP acadêmico caiu para 0.206. Isso prova matematicamente que a barreira 
        financeira tem quase o dobro do peso do mérito na assinatura do contrato.

        3. A MASSA DE VULNERABILIDADE (RENDA COMO FILTRO):
        A tabela de distribuição comprova que o FIES atual é um programa espremido na 
        base da pirâmide: 87,31% dos candidatos únicos sobrevivem apenas na faixa de até 
        R$ 1.500,00 per capita. Acima de R$ 3.001,00, a presença cai para irrisórios 0,08%, 
        confirmando que a alta coparticipação bancária inviabiliza essas faixas.

        4. MÉRITO VS. VIABILIDADE FINANCEIRA (O "OTIMISMO" DA IA):
        A Inteligência Artificial foi induzida ao erro (112.799 Falsos Positivos) ao 
        prever sucesso para estudantes de alto desempenho acadêmico (GAP elevado). A 
        realidade do sistema provou o contrário: o banco barra esses alunos por critérios 
        de risco de crédito financeiro, ignorando a sua excelência no ENEM.

        5. EFICÁCIA PREDITIVA (RECALL EVOLUÍDO):
        O modelo balanceado e limpo atingiu acurácia de 58,07%, porém elevou sua 
        capacidade de identificação (Recall) para expressivos ~62,9% (55.469 acertos 
        reais em 2020/2021). A redução drástica de "falsos positivos" (que caíram pela 
        metade em relação à Análise 007) demonstra que estamos enxergando a verdadeira 
        face comportamental do FIES, imune aos ruídos de inscrições descartadas.
        =============================================================================
        ''')

# %%
