# fazer roc auc
# testar o msm modelo reg logit, mas com o dataset inteiro na matriz de confusao sem distinguir anos e semestre

def orquestrador_inicial_inscritos_17():

    # 1. Gerar dados
    X, Y, X_teste, Y_teste, df_base = gerar_ABT()

    # 2. Treinar modelo
    modelo = TreinarModelo('veio_treino', X, Y)

    # 3. Avaliação principal (ROC, matriz, etc)
    acuracia_e_previsao(
        'veio', 'simm',
        modelo, X, Y, X_teste, Y_teste
    )

    # 4. Análise interpretável (🔥 agora em R$ e ENEM)
    analises_e_datasets(
        'veio_treino', 'simm',
        modelo, X, Y, X_teste, Y_teste,
        df_inicial=df_base
    )

    # 5. Visualizações exploratórias (scatter)


def orquestrador_ja_rodado_inscritos_17():

    import pandas as pd
    import joblib
    from src.constantes import (
        pasta_data_05_processed_analise_017_X_treino, 
        pasta_data_05_processed_analise_017_y_treino,
        pasta_modelo_analise_017,
        pasta_data_05_processed_analise_017_X_teste,
        pasta_data_05_processed_analise_017_y_teste,
        pasta_data_04_load_inscritos
    )

    # 1. Carregar dados
    X = pd.read_parquet(str(pasta_data_05_processed_analise_017_X_treino))
    Y = pd.read_parquet(str(pasta_data_05_processed_analise_017_y_treino))
    X_teste = pd.read_parquet(str(pasta_data_05_processed_analise_017_X_teste))
    Y_teste = pd.read_parquet(str(pasta_data_05_processed_analise_017_y_teste))
    df_base = pd.read_parquet(str(pasta_data_04_load_inscritos))

    # 2. Carregar modelo
    modelo = joblib.load(str(pasta_modelo_analise_017))

    # 3. Avaliação
    acuracia_e_previsao(
        'veio', 'simm',
        modelo, X, Y, X_teste, Y_teste
    )

    # 4. Análise interpretável (🔥 principal)
    analises_e_datasets(
        'veio_treino', 'simm',
        modelo, X, Y, X_teste, Y_teste,
        df_inicial=df_base
    )

    # 5. Visualização exploratória



def gerar_ABT():
    import pandas as pd
    from src.constantes import pasta_data_04_load_inscritos,pasta_data_05_processed_analise_017
    from sklearn.linear_model import LogisticRegression

    # Configuração de visualização
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 1. CARREGAR OS DADOS
    df_base = pd.read_parquet(str(pasta_data_04_load_inscritos))

    # 2. CRIAR VARIÁVEIS NOVAS (Vetorizado, muito mais rápido)
    df_base['gap'] = df_base['media_enem'] - df_base['nota_corte_gp']

    def idade(x):
        try:
            return 2026 - int(str(x).split('/')[-1])
        except:
            return None

    df_base['idade'] = df_base['data_nascimento'].apply(idade)



    print('total de linhas de apenas:', df_base.shape)



    print('-'*60)
    print('qtde de inscritos:')
    display(df_base.groupby('situacao_fies',as_index=False).size().rename(columns={'size':'qtde_inscritos'}))#type: ignore





    # 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
    # Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
    df = pd.get_dummies(df_base, columns=['beneficiado_creduc_fies','modalidade_fies','subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior','opcao_curso'], drop_first=True)

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


    print('coluna x linha do X com NaN inclusos:',X.shape)

    print('coluna x linha do X sem NaN inclusos:',X.shape)


    # %% 
    # 12. SALVAR MATRIZES PROCESSADAS (ABT)
    import os

    # Definindo o diretório de destino
    caminho_salvar = str(pasta_data_05_processed_analise_017)

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
    from src.constantes import pasta_modelo_analise_017

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', LogisticRegression(
            penalty='elasticnet',
            l1_ratio=0.5,
            solver='saga',
            C=0.1,
            class_weight='balanced',
            max_iter=10000,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(x_treino, y_treino.values.ravel())

    print('--- Treinamento Concluído ---')

    joblib.dump(pipeline, str(pasta_modelo_analise_017))
    print(f'Modelo salvo em: {str(pasta_modelo_analise_017)}')

    return pipeline


def analise_real_renda_gap(modelo, X):
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ==============================================================================
    # 1. CONFIGURAÇÕES
    # ==============================================================================

    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.2f}'.format)

    sns.set_theme(style="white", font_scale=1.1)

    pasta_figuras = Path('reports/figures/analise_017').resolve()
    pasta_figuras.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # 2. GRID REALISTA
    # ==============================================================================

    rendas = [400, 800, 1200, 2000, 3000]
    gaps = [-300, -150, -50, 0, 50, 150, 300]

    valores_medios = X.mean(numeric_only=True)
    mediana_idade = X['idade'].median()
    mediana_nota_corte = X['nota_corte_gp'].median()

    linhas = []

    for renda in rendas:
        for gap in gaps:
            linha = {}

            for col in X.columns:
                if col == 'renda_per_capita':
                    linha[col] = renda
                elif col == 'gap':
                    linha[col] = gap
                elif col == 'renda_gap':
                    linha[col] = renda * gap
                elif col == 'idade':
                    linha[col] = mediana_idade
                elif col == 'nota_corte_gp':
                    linha[col] = mediana_nota_corte
                else:
                    linha[col] = valores_medios.get(col, 0)

            linhas.append(linha)

    df_real = pd.DataFrame(linhas).reindex(columns=X.columns, fill_value=0)

    # ==============================================================================
    # 3. PROBABILIDADE REAL
    # ==============================================================================

    probs = modelo.predict_proba(df_real)[:, 1]
    df_real['prob'] = probs * 100  # percentual

    # ==============================================================================
    # 4. CURVAS — IMPACTO DA NOTA DEPENDENDO DA RENDA
    # ==============================================================================

    print("\n[INFO] Gráfico 1: impacto da nota dependendo da renda")
    print("[INFO] Cada linha fixa uma renda e mostra como a probabilidade muda quando o gap varia")
    print("[INFO] A linha horizontal azul em 50% é o limiar de decisão:")
    print("       acima de 50% -> o modelo tende a prever CONTRATADO")
    print("       abaixo de 50% -> o modelo tende a prever NÃO CONTRATADO")

    plt.figure(figsize=(12, 7))

    for renda in rendas:
        subset = df_real[df_real['renda_per_capita'] == renda].sort_values('gap')

        plt.plot(
            subset['gap'],
            subset['prob'],
            marker='o',
            label=f'R$ {renda}'
        )

    plt.axhline(50, color='blue', linestyle='--', label='Limiar de decisão (50%)')
    plt.xlabel('Diferença para nota de corte (pontos)', fontweight='bold')
    plt.ylabel('Probabilidade de contratação (%)', fontweight='bold')
    plt.title('Impacto da nota dependendo da renda', fontweight='bold')
    plt.legend(title='Renda')
    plt.tight_layout()
    plt.savefig(pasta_figuras / 'curvas_impacto_nota_dependendo_renda.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==============================================================================
    # 5. CURVAS — IMPACTO DA RENDA DEPENDENDO DA NOTA
    # ==============================================================================

    print("\n[INFO] Gráfico 2: impacto da renda dependendo da nota")
    print("[INFO] Cada linha fixa um gap e mostra como a probabilidade muda quando a renda varia")
    print("[INFO] A linha horizontal azul em 50% é o limiar de decisão:")
    print("       acima de 50% -> o modelo tende a prever CONTRATADO")
    print("       abaixo de 50% -> o modelo tende a prever NÃO CONTRATADO")

    plt.figure(figsize=(12, 7))

    for gap in gaps:
        subset = df_real[df_real['gap'] == gap].sort_values('renda_per_capita')

        plt.plot(
            subset['renda_per_capita'],
            subset['prob'],
            marker='o',
            label=f'Gap {gap}'
        )

    plt.axhline(50, color='blue', linestyle='--', label='Limiar de decisão (50%)')
    plt.xlabel('Renda per capita (R$)', fontweight='bold')
    plt.ylabel('Probabilidade de contratação (%)', fontweight='bold')
    plt.title('Impacto da renda dependendo da nota', fontweight='bold')
    plt.legend(title='Gap')
    plt.tight_layout()
    plt.savefig(pasta_figuras / 'curvas_impacto_renda_dependendo_nota.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==============================================================================
    # 6. MATRIZ 1 — IMPACTO DA NOTA DEPENDENDO DA RENDA
    # ==============================================================================

    print("\n[INFO] Heatmap 1: impacto da nota dependendo da renda")

    ordem_gap = gaps[::-1]

    matriz_nota_dependendo_renda = df_real.pivot(
        index='renda_per_capita',
        columns='gap',
        values='prob'
    ).reindex(index=rendas, columns=ordem_gap)

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        matriz_nota_dependendo_renda,
        annot=True,
        fmt=".1f",
        cmap="magma_r",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 11},
        cbar=False
    )
    plt.ylabel('Faixa de Renda Per Capita (R$)', fontweight='bold', labelpad=15)
    plt.xlabel('Desempenho (Gap da Nota)', fontweight='bold', labelpad=15)
    plt.title('Probabilidade de contratação (%)\nImpacto da nota dependendo da renda', fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(pasta_figuras / 'impacto_nota_dependendo_renda.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==============================================================================
    # 7. MATRIZ 2 — IMPACTO DA RENDA DEPENDENDO DA NOTA
    # ==============================================================================

    print("\n[INFO] Heatmap 2: impacto da renda dependendo da nota")

    matriz_renda_dependendo_nota = df_real.pivot(
        index='gap',
        columns='renda_per_capita',
        values='prob'
    ).reindex(index=ordem_gap, columns=rendas)

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        matriz_renda_dependendo_nota,
        annot=True,
        fmt=".1f",
        cmap="magma_r",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 11},
        cbar=False
    )
    plt.ylabel('Desempenho (Gap da Nota)', fontweight='bold', labelpad=15)
    plt.xlabel('Faixa de Renda Per Capita (R$)', fontweight='bold', labelpad=15)
    plt.title('Probabilidade de contratação (%)\nImpacto da renda dependendo da nota', fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(pasta_figuras / 'impacto_renda_dependendo_nota.png', dpi=300, bbox_inches='tight')
    plt.show()

def analises_e_datasets(
    treino_ou_teste_veio: str,
    usar_modelo: str,
    modelo_vindo,
    x_treino, y_treino,
    x_teste, y_teste,
    df_inicial
):

    import joblib
    from src.constantes import pasta_modelo_analise_017

    X = x_treino if treino_ou_teste_veio == 'veio_treino' else x_teste

    modelo = modelo_vindo if usar_modelo == 'simm' \
        else joblib.load(str(pasta_modelo_analise_017))

    # =========================
    # 🔥 NOVO BLOCO IMPORTANTE
    # =========================

    analise_real_renda_gap(modelo, X)
    interpretacao_modelo_logit(modelo, X)

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
        roc_curve,
        classification_report
    )
    from src.constantes import pasta_modelo_analise_017

    # ==============================================================================
    # 1. CONFIGURAÇÕES
    # ==============================================================================

    sns.set_theme(style="white", font_scale=1.1)

    X = x_treino
    Y = y_treino.values.ravel()

    modelo = modelo_vindo if usar_modelo == 'simm' else joblib.load(str(pasta_modelo_analise_017))

    # ==============================================================================
    # 2. PROBABILIDADES + ROC
    # ==============================================================================

    chances = modelo.predict_proba(X)[:, 1]

    auc = roc_auc_score(Y, chances)
    print(f"ROC-AUC: {auc:.4f}")

    fpr, tpr, thresholds = roc_curve(Y, chances)
    melhor_indice = (tpr - fpr).argmax()
    corte = thresholds[melhor_indice]

    Y_previsto = (chances >= corte).astype(int)

    acuracia = accuracy_score(Y, Y_previsto)
    print(f"Acurácia: {acuracia:.4f}")
    print(f"Threshold: {corte:.4f}")

    print("\n=== Relatório de Classificação ===")
    print(classification_report(Y, Y_previsto))

    # ==============================================================================
    # 3. MATRIZ DE CONFUSÃO
    # ==============================================================================

    matriz = confusion_matrix(Y, Y_previsto)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        matriz,
        annot=True,
        fmt='d',
        cmap='magma_r',
        vmin=0,
        linewidths=0.5,
        linecolor='white',
        cbar=False
    )
    plt.title("Matriz de Confusão", fontweight='bold')
    plt.xlabel("Previsto", fontweight='bold')
    plt.ylabel("Real", fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # ==============================================================================
    # 4. CURVA ROC
    # ==============================================================================

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', label='Aleatório')
    plt.xlabel("False Positive Rate", fontweight='bold')
    plt.ylabel("True Positive Rate", fontweight='bold')
    plt.title("Curva ROC", fontweight='bold')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    print(f"Taxa prevista (modelo): {Y_previsto.mean():.3f}")
    print(f"Taxa real (base): {Y.mean():.3f}")

def interpretacao_modelo_logit(modelo, X):
    import pandas as pd
    import numpy as np

    print("\n" + "="*80)
    print("🔥 INTERPRETAÇÃO DO MODELO — REGRESSÃO LOGÍSTICA (ANÁLISE 017)")
    print("="*80)

    # pegar modelo interno (pipeline)
    modelo_logit = modelo.named_steps['modelo']

    coef = modelo_logit.coef_[0]
    intercepto = modelo_logit.intercept_[0]

    df_coef = pd.DataFrame({
        'variavel': X.columns,
        'coeficiente': coef
    }).sort_values(by='coeficiente', key=abs, ascending=False)

    print("\n📊 TOP VARIÁVEIS MAIS INFLUENTES (valor absoluto):")
    print(df_coef.head(15))

    print("\n📌 INTERCEPTO (baseline do modelo):")
    print(f"{intercepto:.4f}")

    # pegar coef específicos
    coef_renda = df_coef[df_coef['variavel'] == 'renda_per_capita']['coeficiente'].values
    coef_gap = df_coef[df_coef['variavel'] == 'gap']['coeficiente'].values
    coef_interacao = df_coef[df_coef['variavel'] == 'renda_gap']['coeficiente'].values

    coef_renda = coef_renda[0] if len(coef_renda) else 0
    coef_gap = coef_gap[0] if len(coef_gap) else 0
    coef_interacao = coef_interacao[0] if len(coef_interacao) else 0

    print("\n" + "-"*80)
    print("🧠 INTERPRETAÇÃO DOS PRINCIPAIS EFEITOS")
    print("-"*80)

    print(f"""
1) 📈 GAP (DESEMPENHO NO ENEM)
Coeficiente: {coef_gap:.4f}

→ Interpretação:
Se positivo: quanto maior a nota acima do corte, maior a chance de contratação
Se negativo: modelo está penalizando nota (raro / estranho)

⚠️ IMPORTANTE:
Esse efeito NÃO é fixo → ele depende da renda por causa da interação
""")

    print(f"""
2) 💰 RENDA PER CAPITA
Coeficiente: {coef_renda:.4f}

→ Interpretação:
Se positivo: maior renda aumenta chance de contratação
Se negativo: maior renda reduz chance

⚠️ Esse efeito também NÃO é isolado → depende do gap
""")

    print(f"""
3) 🔥 INTERAÇÃO (RENDA × GAP)
Variável: renda_gap
Coeficiente: {coef_interacao:.4f}

→ ESSA É A CHAVE DO MODELO

Interpretação:

Se POSITIVO:
→ O efeito do GAP fica MAIS FORTE conforme a renda aumenta

Se NEGATIVO:
→ O efeito do GAP fica MAIS FRACO conforme a renda aumenta

📌 Em outras palavras:
O impacto da nota NÃO é igual para todo mundo — depende da renda
""")

    print("\n" + "-"*80)
    print("📊 INTERPRETAÇÃO DOS GRÁFICOS")
    print("-"*80)

    print("""
📈 GRÁFICO 1 — Impacto da nota dependendo da renda

Cada linha = uma renda fixa

→ Mostra como a probabilidade muda quando o aluno melhora a nota

Se as linhas são inclinadas:
→ GAP tem impacto

Se linhas mudam entre si:
→ Existe interação (renda muda o efeito da nota)

----------------------------------------

📈 GRÁFICO 2 — Impacto da renda dependendo da nota

Cada linha = um nível de desempenho

→ Mostra como renda afeta a chance em diferentes níveis de nota

Se as linhas NÃO são paralelas:
→ existe interação forte entre renda e nota

----------------------------------------

🟥 LINHA HORIZONTAL (50%)

→ Limite de decisão do modelo

Acima → CONTRATADO  
Abaixo → NÃO CONTRATADO  

----------------------------------------

🔥 HEATMAPS

Cada célula = probabilidade prevista

→ Linhas = desempenho (gap)
→ Colunas = renda

Leitura correta:
→ Horizontal → efeito da renda
→ Vertical → efeito da nota

Se o padrão muda na diagonal:
→ interação entre renda e nota

""")

    print("\n" + "-"*80)
    print("⚠️ CUIDADOS IMPORTANTES")
    print("-"*80)

    print("""
- Coeficiente NÃO é probabilidade direta
- Modelo trabalha em log-odds
- Interpretação visual (gráficos) é MAIS confiável que coeficiente isolado
- A interação (renda_gap) é o ponto central dessa análise

""")

    print("="*80)