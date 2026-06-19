
ANÁLISE DO `src_legacy.zip` E COMPARAÇÃO COM O `src` ROBUSTO
============================================================

Observação honesta: esta análise é estática. Eu li os arquivos Python, extraí funções/imports, conferi padrões de código e comparei com o `src.zip` robusto já analisado. Não executei o pipeline completo com os microdados oficiais, porque isso exigiria a base bruta completa posicionada nas pastas esperadas. Então, quando eu digo que a lógica está correta, estou falando da coerência da implementação, não de uma auditoria com execução ponta a ponta.

Também vou ignorar os ajustes pequenos que você pediu para ignorar. O foco aqui é: entender o legado que você fez, explicar a lógica de cada arquivo, mostrar trechos importantes e comparar com o robusto para você saber o que mudou de verdade.

VEREDITO CURTO
--------------
O `src_legacy` é um código de pesquisa exploratório, feito em estilo de scripts sequenciais. Ele já tinha a lógica essencial do projeto: limpar microdados, montar funil, criar variáveis como `gap`, dummizar categorias, treinar regressões logísticas, testar Random Forest, calcular métricas, gerar heatmaps, taxas, tabelas e analisar probabilidades previstas.

O `src` robusto não muda a ideia central. Ele organiza melhor a mesma pesquisa. As maiores novidades são: estrutura modular em pacotes, CLI central em `main.py`, separação mais limpa entre pipeline/análise/ABT/modelagem/artigo, uso mais formal de `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, avaliação `in_sample` vs `holdout_80_20` parametrizada, metadados, modelos salvos de forma padronizada e exportação final do pacote `article/`.

Ou seja: o robusto parece mais profissional e reprodutível, mas ele nasce da lógica que já existia no legado.


BIBLIOTECAS E FUNÇÕES QUE APARECEM MUITO NO LEGADO
--------------------------------------------------
- pandas (`pd`): biblioteca central para tabela. `read_csv`, `read_parquet`, `to_parquet`, `groupby`, `merge`, `pivot_table`, `get_dummies`, `cut`, `to_numeric`.
- numpy (`np`): operações numéricas e valores ausentes (`np.nan`, `np.inf`).
- pathlib.Path: manipulação de caminhos com `/` em vez de strings manuais.
- matplotlib / seaborn: gráficos, heatmaps, tabelas como imagem.
- sklearn.linear_model.LogisticRegression: regressão logística para classificação binária ou multiclasse.
- sklearn.pipeline.Pipeline: sequência de etapas, por exemplo `StandardScaler` seguido de `LogisticRegression`.
- sklearn.preprocessing.StandardScaler: padroniza colunas numéricas para média 0 e desvio padrão 1.
- sklearn.ensemble.RandomForestClassifier: floresta aleatória, conjunto de várias árvores de decisão.
- sklearn.metrics: métricas como acurácia, matriz de confusão, classification_report, ROC-AUC.
- scipy.stats.pearsonr e linregress: correlação de Pearson e regressão linear simples na análise renda vs percentual financiado.
- joblib: salvar/carregar modelos treinados em arquivo `.joblib`.


COMO O FLUXO LEGADO EXECUTA, EM LINGUAGEM NATURAL
-------------------------------------------------
1. `constantes.py` define onde ficam as pastas e arquivos.
2. `staging.py` pega os dados brutos e faz uma limpeza inicial pesada.
3. `transform_layer_1.py` transforma FIES/INEP e cruza informações de curso/área CINE.
4. `transform_layer_2.py` corrige ausências/inconsistências de CINE.
5. `trasnform_layer_3.py` adiciona/classifica modalidade e filtros finais.
6. `load.py` renomeia colunas para nomes curtos e salva bases finais limpas.
7. `analise_001` a `analise_006` fazem EDA, funil, distribuição, heatmaps.
8. `analise_007` a `analise_019` testam modelagens diferentes: logit binária, logit ternária, Random Forest, Medicina, com/sem opção de curso.
9. `analise_020` a `analise_022` analisam probabilidades, taxas e tabelas finais.
10. `analise_022_17_artigo.py` consolida parte importante do artigo técnico em um script grande: ABT + modelo + métricas + tabelas/figuras.

A diferença de estilo é clara: no legado, cada análise é quase um “capítulo de notebook” salvo em `.py`. No robusto, cada responsabilidade virou módulo próprio.


CONCEITOS CENTRAIS QUE O LEGADO JÁ USA
--------------------------------------

1) `pd.get_dummies(...)`
No legado, várias análises transformam categorias em colunas binárias usando pandas:

    df = pd.get_dummies(df_base, columns=[...], drop_first=True)

Isso pega uma coluna categórica, por exemplo `turno`, e transforma em colunas 0/1:

    turno_NOTURNO = 1 ou 0
    turno_MATUTINO = 1 ou 0

O `drop_first=True` remove uma das categorias para evitar redundância perfeita. Em regressão, se você coloca todas as categorias de uma variável junto com intercepto, uma coluna pode ser combinação das outras. Isso é chamado de multicolinearidade perfeita. Remover uma categoria cria uma categoria de referência.

No robusto, isso foi trocado por `OneHotEncoder` dentro de `ColumnTransformer`. A ideia é a mesma, mas fica mais seguro para treino/teste, porque o encoder aprende as categorias no treino e depois aplica no teste com `handle_unknown="ignore"`.

2) `Pipeline([('scaler', StandardScaler()), ('modelo', LogisticRegression(...))])`
No legado, algumas análises usam:

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', LogisticRegression(...))
    ])

Isso cria uma esteira: primeiro padroniza os dados, depois treina o modelo. Quando você chama `pipeline.fit(X, y)`, ele faz `StandardScaler.fit_transform(X)` e depois `LogisticRegression.fit(...)`. Quando chama `pipeline.predict_proba(X)`, ele aplica a mesma padronização e depois calcula probabilidades.

3) `StandardScaler()`
Transforma cada coluna numérica em escala padronizada:

    x_padronizado = (x - média) / desvio_padrão

Isso ajuda regressão logística porque renda, idade, nota e dummies podem estar em escalas diferentes. No legado, como `pd.get_dummies` já transformava tudo em número antes do pipeline, o `StandardScaler` acaba padronizando também dummies. Funciona, mas é menos limpo que o robusto. No robusto, `ColumnTransformer` separa numéricas e categóricas.

4) `LogisticRegression(...)`
Modelo de classificação. No binário, estima a probabilidade de classe 1, normalmente “contratada”. Internamente calcula uma combinação linear das variáveis e passa por uma função logística para virar probabilidade.

5) `penalty='elasticnet', l1_ratio=0.5, solver='saga', C=0.1`
Isso aparece no legado em vários scripts.
- `penalty='elasticnet'`: mistura regularização L1 e L2.
- `l1_ratio=0.5`: metade L1, metade L2.
- `solver='saga'`: algoritmo do scikit-learn que suporta elasticnet.
- `C=0.1`: força da regularização inversa; menor C = mais regularização.
Regularização evita que o modelo fique com coeficientes exagerados, especialmente quando há muitas dummies.

6) `class_weight='balanced'`
Dá peso maior para a classe menor. Como contratos são minoria, isso evita que o modelo aprenda a prever quase tudo como “não contratado”.

7) `RandomForestClassifier(...)`
Floresta aleatória. É um conjunto de muitas árvores de decisão. Cada árvore aprende regras, e a floresta combina o voto/probabilidade das árvores. No legado, você usa isso para captar não linearidade entre renda e gap. No robusto, o foco final ficou em árvore de decisão simples com profundidades 10/14/19, porque ela é mais interpretável para artigo técnico.

8) `roc_auc_score`, `confusion_matrix`, `classification_report`
- `roc_auc_score`: mede separação por probabilidades.
- `confusion_matrix`: mostra real vs previsto.
- `classification_report`: imprime precisão, recall e F1 por classe.

9) `predict_proba`
Retorna probabilidades por classe, não apenas a classe final. Isso é central porque seu artigo interpreta probabilidades previstas, não só acerto/erro.


ANÁLISE ARQUIVO POR ARQUIVO DO SRC_LEGACY
=========================================


------------------------------------------------------------
analise_001.py
------------------------------------------------------------

Linhas: 184

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: gerar_dataset_candidatos_unicos_por_Prioridade_inicial


O que faz:
Gera dataset de candidatos únicos por prioridade inicial. Como uma pessoa pode ter várias inscrições/opções, este script tenta reduzir para uma unidade por candidato seguindo regras de prioridade de situação/opção. Serve para análise de robustez em nível de candidato, diferente da unidade principal por inscrição.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_002.py
------------------------------------------------------------

Linhas: 328

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: eda_gerar_dataset_com_varias_analises


O que faz:
EDA geral. Gera um dataset com várias análises descritivas do processo seletivo. É uma exploração inicial: contagens, agrupamentos, distribuição de situações, possivelmente por ano/semestre/área. Não é o núcleo final da modelagem, mas ajuda a entender os dados.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_003.py
------------------------------------------------------------

Linhas: 286

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: eda_gerar_funil_de_selecao_6_etapas


O que faz:
Gera funil de seleção em 6 etapas. É uma primeira versão do funil: vagas, inscrições, elegibilidade/desempenho suficiente e contratos. Ajuda a materializar a ideia do artigo: há queda entre etapas e contrato não é automático.


Trecho real relevante e explicação:

```python
df_nacional = (
        df.groupby(['periodo', 'nome_cine_area_geral'], dropna=False)[colunas_funil]
        .sum()
        .reset_index()
    )

    # --- 5. Parâmetros básicos ---
    areas = sorted(df_nacional['nome_cine_area_geral'].unique())
    periodos = sorted(df_nacional['periodo'].unique())
    largura_barra = 0.12
    cores_areas = plt.cm.tab10(np.linspace(0, 1, len(areas)))

    # --- 6. Criar gráfico nacional ---
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(colunas_funil))
    deslocamentos = np.linspace(-largura_barra * 2.5, largura_barra * 2.5, len(periodos))

    for desloc, periodo in zip(deslocamentos, periodos):
        df_periodo = df_nacional[df_nacional['periodo'] == periodo]
        base = np.zeros(len(colunas_funil))
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_003_1.py
------------------------------------------------------------

Linhas: 641

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: eda_gerar_funil_de_selecao_6_etapas_v2


O que faz:
Versão mais completa/visual do funil de seleção, com função de plotagem segura e estilos monocromáticos. Produz gráficos mais adequados ao artigo/relatório.


Trecho real relevante e explicação:

```python
df_agg = (
                df_data
                .groupby(["periodo", col_categoria])[cols_metricas]
                .sum()
                .reset_index()
            )

            categorias_list = list(categorias_list)

            mux = pd.MultiIndex.from_product(
                [periodos_list, categorias_list],
                names=["periodo", col_categoria],
            )

            df_plot = (
                df_agg
                .set_index(["periodo", col_categoria])
                .reindex(mux, fill_value=0)
                .reset_index()
            )

        else:
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_003_1_1 NO_area_cine.py
------------------------------------------------------------

Linhas: 289

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: eda_gerar_funil_de_selecao_6_etapas_v2


O que faz:
Variação do funil sem quebrar por área CINE. Serve para ver fluxo mais agregado/nacional. A lógica é a mesma do funil, mas com menos dimensões de agrupamento.


Trecho real relevante e explicação:

```python
# 1. Agrupar
        if col_categoria is not None:
            df_agg = df_data.groupby(['periodo', col_categoria])[cols_metricas].sum().reset_index()
            categorias_list = list(categorias_list)
            mux = pd.MultiIndex.from_product([periodos_list, categorias_list], names=['periodo', col_categoria])
            df_plot = df_agg.set_index(['periodo', col_categoria]).reindex(mux, fill_value=0).reset_index()
        else:
            df_plot = df_data.groupby(['periodo'])[cols_metricas].sum().reset_index()
            df_plot['dummy'] = 'Total'
            categorias_list = ['Total']
            col_categoria = 'dummy'

        # 2. Configurar Gráfico
        fig, ax = plt.subplots(figsize=figsize, dpi=240)
        x = np.arange(len(cols_metricas))
        
        largura_barra = 0.155 
        
        half_span = largura_barra * (len(periodos_list) - 1) / 2.0
        deslocamentos = np.linspace(-half_span, half_span, len(periodos_list))
        
        if colormap == 'Set2':
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_004.py
------------------------------------------------------------

Linhas: 82

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: analise_fuga_cerebros_de_regioes


O que faz:
Análise exploratória de “fuga de cérebros”/mobilidade regional. Compara região/UF de residência e região/UF da oferta ou curso. É complementar, não é o núcleo do artigo.


Trecho real relevante e explicação:

```python
# 3. VOLUMETRIA (Dataset via display)
    # ==============================================================================
    df_volumetria = df_analise.groupby('faixa_renda', observed=False).size().reset_index(name='Qtd_Inscritos')
    df_volumetria['Participação'] = (df_volumetria['Qtd_Inscritos'] / len(df_analise) * 100).round(2).astype(str) + '%'

    print("\n📊 DATASET DE APOIO: VOLUME DE PESSOAS POR FAIXA")
    display(df_volumetria)#ignore: type
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_005.py
------------------------------------------------------------

Linhas: 57

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: verificar_qtde_inscritos_por_ano_e_semestre_nacional_e_regional


O que faz:
Verifica quantidade de inscritos por ano e semestre em níveis nacional e regional. É auditoria/EDA de volume dos dados.


Trecho real relevante e explicação:

```python
print("-" * 90)

    resumo_uf = (df.groupby(['ano', 'semestre', 'regiao_ies_alvo', coluna_uf])
                .size()
                .reset_index(name='Total_Inscritos'))

    # Ordena por Ano, Semestre, e depois do estado com MAIS inscritos para o com MENOS
    resumo_uf = resumo_uf.sort_values(by=['ano', 'semestre', 'Total_Inscritos'], ascending=[True, True, False])

    print(resumo_uf.to_string(index=False))

    print("\n" + "="*90)
    print("✅ Agrupamento concluído! Verifique se todos os estados estão com o volume esperado.")
    print("="*90 + "\n")
    # %%
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_006.py
------------------------------------------------------------

Linhas: 398

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: gerar_figura_b1_lista_espera


O que faz:
Gera heatmap da lista de espera, cruzando faixas de renda e desempenho/gap. Usa matriz com anotações. É importante porque mostra visualmente a barreira acadêmica inicial.


Trecho real relevante e explicação:

```python
)

    df_analise = df_counts.merge(
        df_totals,
        on=["faixa_renda_bruta", "nivel_nota_gap"],
        how="left"
    )

    df_analise["percentual_celula"] = (
        df_analise["qtd"] / df_analise["total_celula"] * 100
    )

    df_analise.to_parquet(str(caminho_final_parquet), index=False)

    # ==============================================================================
    # 7. ISOLA SOMENTE LISTA DE ESPERA
    # ==============================================================================

    df_lista = df_analise[
        df_analise["status_final"] == "8. LISTA DE ESPERA"
    ].copy()
```

Esse trecho lê ou salva Parquet. Parquet é mais eficiente que CSV para pipeline de dados, preserva melhor tipos e carrega mais rápido.


------------------------------------------------------------
analise_006_1.py
------------------------------------------------------------

Linhas: 276

Imports principais detectados: matplotlib, matplotlib.pyplot, numpy, pandas, pathlib, re

Funções definidas: orquestrador_tabela_b1_distribuicao_renda, carregar_e_filtrar_base, gerar_distribuicao_por_faixa_renda, formatar_numero_inteiro, formatar_percentual, limpar_nome_arquivo, obter_fonte_padrao, plotar_tabela_b1_como_imagem


O que faz:
Gera tabela B1 de distribuição por faixa de renda. Usa pandas para contar registros, formatar percentuais e matplotlib para renderizar tabela como imagem.


Trecho real relevante e explicação:

```python
df_renda = (
        df.groupby("faixa_renda", observed=True)
          .size()
          .reindex(labels_renda, fill_value=0)
          .reset_index(name="inscritos")
    )

    total = df_renda["inscritos"].sum()

    df_renda["percentual"] = df_renda["inscritos"] / total * 100
    df_renda["percentual_acumulado"] = df_renda["percentual"].cumsum()

    df_tabela = pd.DataFrame({
        "Faixa de renda per capita (R$)": df_renda["faixa_renda"].astype(str),
        "Inscritos": df_renda["inscritos"],
        "%": df_renda["percentual"],
        "% acumulado": df_renda["percentual_acumulado"]
    })

    return df_tabela
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_006_1_1.py
------------------------------------------------------------

Linhas: 479

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: gerar_dataset_e_grafico_heatmap_quem_sao_os_inscritos_contratados


O que faz:
Gera dataset e heatmaps sobre quem são os inscritos contratados, cruzando renda/desempenho/situação. É uma extensão exploratória da análise de distribuição.


Trecho real relevante e explicação:

```python
)

    df_analise = df_counts.merge(
        df_totals,
        on=[
            "nome_cine_area_geral",
            "faixa_renda_bruta",
            "nivel_nota_gap"
        ]
    )

    df_analise["percentual_celula"] = (
        df_analise["qtd"] / df_analise["total_celula"] * 100
    )

    df_analise.to_parquet(str(caminho_final_parquet), index=False)

    # ==============================================================================
    # 7. AUDITORIA
    # ==============================================================================

    print("\n🔍 AUDITORIA COMPLETA:")
```

Esse trecho lê ou salva Parquet. Parquet é mais eficiente que CSV para pipeline de dados, preserva melhor tipos e carrega mais rápido.


------------------------------------------------------------
analise_007.py
------------------------------------------------------------

Linhas: 512

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos, orquestrador_ja_rodado_inscritos, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Primeira modelagem logística binária simples. Usa gap, renda e área CINE dummizada para prever contratada vs não contratada. Divide treino/teste por ano: 2019 como treino e 2020–2021 como teste.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Cria `gap = media_enem - nota_corte_gp`.
- Filtra classes `CONTRATADA` e `NÃO CONTRATADO`.
- Cria target binário, geralmente 1 para contratada e 0 para não contratada.
- Dummiza área CINE.
- Treina regressão logística.
- Usa probabilidades previstas para avaliar contratação.
Isso é uma versão inicial da ABT binária do robusto.


------------------------------------------------------------
analise_008.py
------------------------------------------------------------

Linhas: 513

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_candidatos, orquestrador_ja_rodado_candidatos, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Versão parecida com analise_007, mas usando candidatos únicos/prioridade inicial em vez de todas as inscrições. Serve como teste de robustez da unidade de análise.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Cria `gap = media_enem - nota_corte_gp`.
- Filtra classes `CONTRATADA` e `NÃO CONTRATADO`.
- Cria target binário, geralmente 1 para contratada e 0 para não contratada.
- Dummiza área CINE.
- Treina regressão logística.
- Usa probabilidades previstas para avaliar contratação.
Isso é uma versão inicial da ABT binária do robusto.


------------------------------------------------------------
analise_009.py
------------------------------------------------------------

Linhas: 428

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_nove, orquestrador_ja_rodado_inscritos_nove, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Modelagem binária com mais controles categóricos. Usa get_dummies para subárea, região, natureza jurídica, etnia, turno, escola pública, conceito, curso etc.; treina regressão logística com StandardScaler + elasticnet.


Trecho real relevante e explicação:

```python
# 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
    # Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
    df = pd.get_dummies(df_base, columns=['subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior','opcao_curso'], drop_first=True)

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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Expande a ABT binária com mais variáveis de curso, instituição e candidato.
- Usa `pd.get_dummies` para transformar categorias em números.
- Treina regressão logística, em algumas versões com `elasticnet`, em outras com `lbfgs`/sem penalização.
- Calcula probabilidade de contratação por `predict_proba`.
- Avalia com matriz de confusão e ROC-AUC.
Esses scripts são experimentos de especificação: você estava testando quais variáveis entravam melhor.


------------------------------------------------------------
analise_009_1.py
------------------------------------------------------------

Linhas: 375

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_nove_um, orquestrador_ja_rodado_inscritos_nove_um, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Variação da analise_009 sem opção de curso. Serve para comparar o quanto a variável opção_curso influencia a modelagem.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Expande a ABT binária com mais variáveis de curso, instituição e candidato.
- Usa `pd.get_dummies` para transformar categorias em números.
- Treina regressão logística, em algumas versões com `elasticnet`, em outras com `lbfgs`/sem penalização.
- Calcula probabilidade de contratação por `predict_proba`.
- Avalia com matriz de confusão e ROC-AUC.
Esses scripts são experimentos de especificação: você estava testando quais variáveis entravam melhor.


------------------------------------------------------------
analise_010.py
------------------------------------------------------------

Linhas: 375

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_dez, orquestrador_ja_rodado_inscritos_dez, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Outra versão da modelagem binária com conjunto de controles semelhante. Mantém pipeline com StandardScaler e LogisticRegression elasticnet.


Trecho real relevante e explicação:

```python
# 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
    # Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
    df = pd.get_dummies(df_base, columns=['subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior','opcao_curso'], drop_first=True)

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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Expande a ABT binária com mais variáveis de curso, instituição e candidato.
- Usa `pd.get_dummies` para transformar categorias em números.
- Treina regressão logística, em algumas versões com `elasticnet`, em outras com `lbfgs`/sem penalização.
- Calcula probabilidade de contratação por `predict_proba`.
- Avalia com matriz de confusão e ROC-AUC.
Esses scripts são experimentos de especificação: você estava testando quais variáveis entravam melhor.


------------------------------------------------------------
analise_011.py
------------------------------------------------------------

Linhas: 409

Imports principais detectados: constantes, matplotlib, matplotlib.pyplot, numpy, pandas, pathlib, re, scipy.stats, sklearn.linear_model

Funções definidas: orquestrador_analise_011, preparar_dados_financiamento, calcular_estatisticas_e_regressao, obter_fonte_padrao, limpar_nome_arquivo, plotar_regressao_artigo, plotar_tabela_resultados_como_imagem


O que faz:
Análise de financiamento/coparticipação. Filtra contratos efetivados e calcula associação entre renda per capita e percentual financiado usando correlação de Pearson e regressão linear simples. Essa lógica depois aparece no artigo de políticas públicas.


Trecho real relevante e explicação:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from matplotlib import font_manager

from constantes import pasta_data_04_load_inscritos


def orquestrador_analise_011():
    print("\n" + "=" * 70)
    print("ANÁLISE 11: RENDA PER CAPITA vs PERCENTUAL DE FINANCIAMENTO")
    print("=" * 70)

    pasta_saida = Path("../reports/figures/analise_011")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    df = preparar_dados_financiamento()

    if df is None or df.empty:
        print("Erro: não há dados suficientes para análise.")
        return
```

Esse trecho usa estatística clássica: `pearsonr` mede correlação linear entre renda e percentual financiado; `linregress` ajusta uma regressão linear simples para estimar a inclinação da relação.


------------------------------------------------------------
analise_012_0 com insights.py
------------------------------------------------------------

Linhas: 494

Imports principais detectados: IPython.display, constantes, matplotlib.pyplot, pandas, seaborn, sklearn.ensemble, sklearn.inspection, sklearn.metrics, sklearn.tree

Funções definidas: analise_real_renda_gap_rf, interpretacao_modelo_random_forest


O que faz:
Random Forest binária usando base inteira. Objetivo declarado: captar não linearidade e interação entre renda_per_capita e gap. Usa RandomForestClassifier com 1000 árvores, class_weight balanced, max_features sqrt e min_samples_leaf.


Trecho real relevante e explicação:

```python
max_depth = 42

modelo = RandomForestClassifier(
    max_depth=max_depth,
    n_estimators=1000,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42,
    min_samples_leaf=125,
    max_features='sqrt'
)

modelo.fit(X_treino, Y_treino)

print("Modelo RandomForest treinado com sucesso.")

# =========================================================
# 3) VISUALIZAR UMA ÁRVORE DA FLORESTA (OPCIONAL)
# =========================================================

visualizar_arvore = 'nao'
```

Esse trecho treina Random Forest. Uma floresta aleatória combina muitas árvores de decisão. Cada árvore aprende regras e a floresta combina os resultados. `n_estimators=1000` significa 1000 árvores; `max_depth` limita profundidade; `min_samples_leaf` evita folhas muito pequenas; `max_features='sqrt'` faz cada árvore considerar só parte das variáveis por divisão, aumentando diversidade.


Lógica do algoritmo neste arquivo:
- Usa Random Forest em vez de regressão logística.
- A intenção é captar relações não lineares. Exemplo: o efeito do gap pode não ser uma reta e pode mudar por faixa de renda.
- Random Forest também calcula `feature_importances_`, que indica quais variáveis mais reduziram impureza nas árvores.
- O cuidado é que Random Forest é menos transparente que uma árvore única; por isso o robusto preferiu árvores de decisão para o artigo técnico.


------------------------------------------------------------
analise_012_1 medicina.py
------------------------------------------------------------

Linhas: 307

Imports principais detectados: constantes, matplotlib.pyplot, os, pandas, seaborn, sklearn.ensemble, sklearn.linear_model, sklearn.metrics

Funções definidas: idade


O que faz:
Random Forest para o recorte de Medicina. Aplica filtro/recorte de Medicina e treina floresta para avaliar um curso de alto custo e alta seletividade.


Trecho real relevante e explicação:

```python
# 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
# Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
df = pd.get_dummies(df_base, columns=['beneficiado_creduc_fies','modalidade_fies','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior','opcao_curso'], drop_first=True)

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
colunas_dummies_regiao = [col for col in df.columns if col.startswith('regiao_morar_')]
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Usa Random Forest em vez de regressão logística.
- A intenção é captar relações não lineares. Exemplo: o efeito do gap pode não ser uma reta e pode mudar por faixa de renda.
- Random Forest também calcula `feature_importances_`, que indica quais variáveis mais reduziram impureza nas árvores.
- O cuidado é que Random Forest é menos transparente que uma árvore única; por isso o robusto preferiu árvores de decisão para o artigo técnico.


------------------------------------------------------------
analise_012_2_0 y ternario.py
------------------------------------------------------------

Linhas: 753

Imports principais detectados: IPython.display, constantes, matplotlib.pyplot, pandas, seaborn, sklearn.ensemble, sklearn.inspection, sklearn.linear_model, sklearn.metrics, sklearn.tree

Funções definidas: idade, mapear_status, analise_real_renda_gap_rf, interpretacao_modelo_random_forest, matriz_decisao_renda_gap_rf


O que faz:
Random Forest com alvo ternário: lista de espera, não contratado e contratado. Usa ROC-AUC multiclasse OVR e matriz de confusão 3x3.


Trecho real relevante e explicação:

```python
# 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
# Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
df = pd.get_dummies(df_base, columns=['beneficiado_creduc_fies','modalidade_fies','subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior','opcao_curso'], drop_first=True)

# 4. FILTRAR A BASE DE DADOS
    # 2. Mapear cada situação para número
def mapear_status(x):
    if x == 'CONTRATADA':
        return 2
    elif x == 'LISTA DE ESPERA':
        return 1
    elif x == 'NÃO CONTRATADO':
        return 0
    else:
        return None

df['status'] = df['situacao_fies'].apply(mapear_status)


# 6. PESCAR AS COLUNAS DUMMIES
# O Python procura e guarda apenas as colunas que começam com esse nome
colunas_dummies_turno = [col for col in df.columns if col.startswith('turno_')]
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Usa Random Forest em vez de regressão logística.
- A intenção é captar relações não lineares. Exemplo: o efeito do gap pode não ser uma reta e pode mudar por faixa de renda.
- Random Forest também calcula `feature_importances_`, que indica quais variáveis mais reduziram impureza nas árvores.
- O cuidado é que Random Forest é menos transparente que uma árvore única; por isso o robusto preferiu árvores de decisão para o artigo técnico.


------------------------------------------------------------
analise_012_2_1 y ternario medicina.py
------------------------------------------------------------

Linhas: 757

Imports principais detectados: IPython.display, constantes, matplotlib.pyplot, pandas, seaborn, sklearn.ensemble, sklearn.inspection, sklearn.linear_model, sklearn.metrics, sklearn.tree

Funções definidas: idade, mapear_status, analise_real_renda_gap_rf, interpretacao_modelo_random_forest, matriz_decisao_renda_gap_rf


O que faz:
Random Forest ternária para Medicina. Mesma ideia da ternária geral, mas restrita ao curso de Medicina.


Trecho real relevante e explicação:

```python
# 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
# Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
df = pd.get_dummies(df_base, columns=['beneficiado_creduc_fies','modalidade_fies','subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior','opcao_curso'], drop_first=True)

# 4. FILTRAR A BASE DE DADOS
    # 2. Mapear cada situação para número
def mapear_status(x):
    if x == 'CONTRATADA':
        return 2
    elif x == 'LISTA DE ESPERA':
        return 1
    elif x == 'NÃO CONTRATADO':
        return 0
    else:
        return None

df['status'] = df['situacao_fies'].apply(mapear_status)


# 6. PESCAR AS COLUNAS DUMMIES
# O Python procura e guarda apenas as colunas que começam com esse nome
colunas_dummies_turno = [col for col in df.columns if col.startswith('turno_')]
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Usa Random Forest em vez de regressão logística.
- A intenção é captar relações não lineares. Exemplo: o efeito do gap pode não ser uma reta e pode mudar por faixa de renda.
- Random Forest também calcula `feature_importances_`, que indica quais variáveis mais reduziram impureza nas árvores.
- O cuidado é que Random Forest é menos transparente que uma árvore única; por isso o robusto preferiu árvores de decisão para o artigo técnico.


------------------------------------------------------------
analise_012_3 sem opcoes.py
------------------------------------------------------------

Linhas: 630

Imports principais detectados: IPython.display, constantes, matplotlib.pyplot, pandas, seaborn, sklearn.ensemble, sklearn.inspection, sklearn.linear_model, sklearn.metrics, sklearn.tree

Funções definidas: idade, analise_real_renda_gap_rf, interpretacao_modelo_random_forest


O que faz:
Random Forest binária sem opção de curso. Testa se o modelo depende demais da variável opção_curso.


Trecho real relevante e explicação:

```python
# 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
# Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
df = pd.get_dummies(df_base, columns=['beneficiado_creduc_fies','modalidade_fies','subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior',], drop_first=True)

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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Usa Random Forest em vez de regressão logística.
- A intenção é captar relações não lineares. Exemplo: o efeito do gap pode não ser uma reta e pode mudar por faixa de renda.
- Random Forest também calcula `feature_importances_`, que indica quais variáveis mais reduziram impureza nas árvores.
- O cuidado é que Random Forest é menos transparente que uma árvore única; por isso o robusto preferiu árvores de decisão para o artigo técnico.


------------------------------------------------------------
analise_013.py
------------------------------------------------------------

Linhas: 370

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_13, orquestrador_ja_rodado_inscritos_13, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Regressão logística binária sem regularização explícita, com lbfgs. Serve como comparação contra modelos elasticnet/balanced.


Trecho real relevante e explicação:

```python
# 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
    # Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
    df = pd.get_dummies(df_base, columns=['subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior','opcao_curso'], drop_first=True)

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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Expande a ABT binária com mais variáveis de curso, instituição e candidato.
- Usa `pd.get_dummies` para transformar categorias em números.
- Treina regressão logística, em algumas versões com `elasticnet`, em outras com `lbfgs`/sem penalização.
- Calcula probabilidade de contratação por `predict_proba`.
- Avalia com matriz de confusão e ROC-AUC.
Esses scripts são experimentos de especificação: você estava testando quais variáveis entravam melhor.


------------------------------------------------------------
analise_014.py
------------------------------------------------------------

Linhas: 707

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_14, orquestrador_ja_rodado_inscritos_14, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Regressão logística com controles adicionais incluindo modalidade_fies e beneficiado_creduc_fies. Parece uma especificação mais completa da etapa contratual.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Expande a ABT binária com mais variáveis de curso, instituição e candidato.
- Usa `pd.get_dummies` para transformar categorias em números.
- Treina regressão logística, em algumas versões com `elasticnet`, em outras com `lbfgs`/sem penalização.
- Calcula probabilidade de contratação por `predict_proba`.
- Avalia com matriz de confusão e ROC-AUC.
Esses scripts são experimentos de especificação: você estava testando quais variáveis entravam melhor.


------------------------------------------------------------
analise_015.py
------------------------------------------------------------

Linhas: 710

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_15, orquestrador_ja_rodado_inscritos_15, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Variação próxima da analise_014. Mantém conjunto amplo de variáveis e avalia modelo/logit com métricas e análises auxiliares.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Expande a ABT binária com mais variáveis de curso, instituição e candidato.
- Usa `pd.get_dummies` para transformar categorias em números.
- Treina regressão logística, em algumas versões com `elasticnet`, em outras com `lbfgs`/sem penalização.
- Calcula probabilidade de contratação por `predict_proba`.
- Avalia com matriz de confusão e ROC-AUC.
Esses scripts são experimentos de especificação: você estava testando quais variáveis entravam melhor.


------------------------------------------------------------
analise_016.py
------------------------------------------------------------

Linhas: 710

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_16, orquestrador_ja_rodado_inscritos_16, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Variação sem opção de curso em relação às especificações completas. Serve para avaliar robustez quando se retira essa variável muito forte.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Expande a ABT binária com mais variáveis de curso, instituição e candidato.
- Usa `pd.get_dummies` para transformar categorias em números.
- Treina regressão logística, em algumas versões com `elasticnet`, em outras com `lbfgs`/sem penalização.
- Calcula probabilidade de contratação por `predict_proba`.
- Avalia com matriz de confusão e ROC-AUC.
Esses scripts são experimentos de especificação: você estava testando quais variáveis entravam melhor.


------------------------------------------------------------
analise_017_0_com_insights.py
------------------------------------------------------------

Linhas: 702

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_17, orquestrador_ja_rodado_inscritos_17, gerar_ABT, TreinarModelo, analise_real_renda_gap, analises_e_datasets, acuracia_e_previsao, interpretacao_modelo_logit


O que faz:
Uma das versões mais importantes do legado. Regresão logística binária com elasticnet, class_weight balanced e análise interpretável de renda/gap em R$ e ENEM. Usa base inteira em parte da avaliação/insights.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Esta família é mais próxima do núcleo final: regressão logística com elasticnet, class_weight balanced, probabilidades e interpretação.
- Nas versões ternárias, o target deixa de ser só contratado/não contratado e passa a distinguir lista de espera, não contratado e contratado.
- Isso antecipa a ideia central do artigo: binário = etapa contratual; ternário = fluxo mais amplo.


------------------------------------------------------------
analise_017_2_0 y ternario.py
------------------------------------------------------------

Linhas: 669

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_17, orquestrador_ja_rodado_inscritos_17, gerar_ABT, TreinarModelo, analises_e_datasets, acuracia_e_previsao, interpretacao_modelo_logit, analise_real_multiclasse


O que faz:
Versão logística ternária geral: lista de espera, não contratado, contratado. Usa LogisticRegression multinomial/elasticnet via saga e calcula ROC-AUC OVR, classification_report e matriz de confusão.


Trecho real relevante e explicação:

```python
# 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
    # Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
    df = pd.get_dummies(df_base, columns=['beneficiado_creduc_fies','modalidade_fies','subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior','opcao_curso'], drop_first=True)

    # 4. FILTRAR A BASE DE DADOS
    df = df[df['situacao_fies'].isin([
    'CONTRATADA',
    'LISTA DE ESPERA',
    'NÃO CONTRATADO'
    ])].copy()
        # 2. Mapear cada situação para número
    def mapear_status(x):
        if x == 'CONTRATADA':
            return 2
        elif x == 'LISTA DE ESPERA':
            return 1
        elif x == 'NÃO CONTRATADO':
            return 0
        else:
            return None

    df['status'] = df['situacao_fies'].apply(mapear_status)
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Esta família é mais próxima do núcleo final: regressão logística com elasticnet, class_weight balanced, probabilidades e interpretação.
- Nas versões ternárias, o target deixa de ser só contratado/não contratado e passa a distinguir lista de espera, não contratado e contratado.
- Isso antecipa a ideia central do artigo: binário = etapa contratual; ternário = fluxo mais amplo.


------------------------------------------------------------
analise_017_2_1  y ternario medicina.py
------------------------------------------------------------

Linhas: 673

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_17, orquestrador_ja_rodado_inscritos_17, gerar_ABT, TreinarModelo, analises_e_datasets, acuracia_e_previsao, interpretacao_modelo_logit, analise_real_multiclasse


O que faz:
Versão logística ternária para Medicina. Mantém a lógica da ternária geral, mas no recorte do curso.


Trecho real relevante e explicação:

```python
# 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
    # Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
    df = pd.get_dummies(df_base, columns=['beneficiado_creduc_fies','modalidade_fies','subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior','opcao_curso'], drop_first=True)

    # 4. FILTRAR A BASE DE DADOS
    df = df[df['situacao_fies'].isin([
    'CONTRATADA',
    'LISTA DE ESPERA',
    'NÃO CONTRATADO'
    ])].copy()
        # 2. Mapear cada situação para número
    def mapear_status(x):
        if x == 'CONTRATADA':
            return 2
        elif x == 'LISTA DE ESPERA':
            return 1
        elif x == 'NÃO CONTRATADO':
            return 0
        else:
            return None

    df['status'] = df['situacao_fies'].apply(mapear_status)
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Esta família é mais próxima do núcleo final: regressão logística com elasticnet, class_weight balanced, probabilidades e interpretação.
- Nas versões ternárias, o target deixa de ser só contratado/não contratado e passa a distinguir lista de espera, não contratado e contratado.
- Isso antecipa a ideia central do artigo: binário = etapa contratual; ternário = fluxo mais amplo.


------------------------------------------------------------
analise_017_3 sem opcoes.py
------------------------------------------------------------

Linhas: 708

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_17, orquestrador_ja_rodado_inscritos_17, gerar_ABT, TreinarModelo, analise_real_renda_gap, analises_e_datasets, acuracia_e_previsao, interpretacao_modelo_logit


O que faz:
Versão logística sem opção de curso. Serve para testar o efeito de retirar essa variável da especificação.


Trecho real relevante e explicação:

```python
# 3. DUMIZAR VARIÁVEIS CATEGÓRICAS
    # Isso cria as colunas 0 e 1 e JÁ EXCLUI a 'nome_cine_area_geral' original
    df = pd.get_dummies(df_base, columns=['beneficiado_creduc_fies','modalidade_fies','subarea_conhecimento','regiao_morar','natureza_juridica_mantenedora','etnia_cor','turno','ensino_medio_escola_publica','conceito_curso_gp','concluiu_curso_superior',], drop_first=True)

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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Esta família é mais próxima do núcleo final: regressão logística com elasticnet, class_weight balanced, probabilidades e interpretação.
- Nas versões ternárias, o target deixa de ser só contratado/não contratado e passa a distinguir lista de espera, não contratado e contratado.
- Isso antecipa a ideia central do artigo: binário = etapa contratual; ternário = fluxo mais amplo.


------------------------------------------------------------
analise_018.py
------------------------------------------------------------

Linhas: 426

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_18, orquestrador_ja_rodado_inscritos_18, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Outra especificação logística binária/analítica, próxima das anteriores, com controles amplos. É uma versão intermediária da sequência de experimentos.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Expande a ABT binária com mais variáveis de curso, instituição e candidato.
- Usa `pd.get_dummies` para transformar categorias em números.
- Treina regressão logística, em algumas versões com `elasticnet`, em outras com `lbfgs`/sem penalização.
- Calcula probabilidade de contratação por `predict_proba`.
- Avalia com matriz de confusão e ROC-AUC.
Esses scripts são experimentos de especificação: você estava testando quais variáveis entravam melhor.


------------------------------------------------------------
analise_019.py
------------------------------------------------------------

Linhas: 433

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: orquestrador_inicial_inscritos_19, orquestrador_ja_rodado_inscritos_19, gerar_ABT, TreinarModelo, prever_probabilidade_treino, analises_e_datasets, acuracia_e_previsao


O que faz:
Outra especificação logística binária/analítica, próxima das anteriores. A sequência 013–019 mostra evolução experimental: acrescentar/remover variáveis e testar efeitos.


Trecho real relevante e explicação:

```python
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
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Lógica do algoritmo neste arquivo:
- Expande a ABT binária com mais variáveis de curso, instituição e candidato.
- Usa `pd.get_dummies` para transformar categorias em números.
- Treina regressão logística, em algumas versões com `elasticnet`, em outras com `lbfgs`/sem penalização.
- Calcula probabilidade de contratação por `predict_proba`.
- Avalia com matriz de confusão e ROC-AUC.
Esses scripts são experimentos de especificação: você estava testando quais variáveis entravam melhor.


------------------------------------------------------------
analise_020.py
------------------------------------------------------------

Linhas: 271

Imports principais detectados no topo: nenhum ou imports feitos dentro das funções.

Funções definidas: analisar_probabilidades_modelo, analisar_probabilidades_modelo


O que faz:
Analisa probabilidades do modelo já treinado. Não é treinamento central; é leitura/diagnóstico das probabilidades previstas.


Trecho real relevante e explicação:

```python
)

    tabela = X.groupby(
        ["renda_grupo","gap_grupo"]
    )["probabilidade_contratacao"].mean().unstack()

    plt.figure(figsize=(7,5))

    sns.heatmap(
        tabela,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn"
    )

    plt.title("Probabilidade média prevista\n(renda × gap)")

    plt.show()

    return X

df_resultado = analisar_probabilidades_modelo()
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_021_taxas.py
------------------------------------------------------------

Linhas: 558

Imports principais detectados: matplotlib, matplotlib.lines, matplotlib.pyplot, matplotlib.ticker, numpy, pandas, pathlib, re

Funções definidas: limpar_nome_arquivo, divisao_segura, calcular_taxas, formatar_percentual


O que faz:
Calcula e plota taxas do processo seletivo, como conversão, ocupação e variações por área/região/semestre. É equivalente conceitual ao dataset_taxas.py + taxas_conversao.py do robusto.


Trecho real relevante e explicação:

```python
df_agg = (
    df.groupby(
        ["ano", "semestre", "regiao_ies", "nome_cine_area_geral"],
        as_index=False,
    )[colunas_base]
    .sum()
)

df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
df_plot = calcular_taxas(df_agg)

df_plot["periodo"] = (
    "'" + df_plot["ano"].astype(str).str[-2:] + "." + df_plot["semestre"].astype(str)
)

# ==============================================================================
# 3. MÉDIAS/NÍVEIS NACIONAIS CALCULADOS A PARTIR DOS SOMATÓRIOS
# ==============================================================================

df_nacional_base = (
    df_agg.groupby(
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_022.py
------------------------------------------------------------

Linhas: 238

Imports principais detectados: matplotlib, matplotlib.pyplot, pandas, pathlib, re

Funções definidas: orquestrador_tabela_1, carregar_base, normalizar_status, preparar_tabela_1, formatar_inteiro, formatar_percentual, limpar_nome_arquivo, obter_fonte_padrao, plotar_tabela_1_como_imagem


O que faz:
Gera Tabela 1 de distribuição das inscrições por situação no processo seletivo. Usa pandas para contar e matplotlib para salvar tabela como imagem.


Trecho real relevante e explicação:

```python
df_agg = (
        df.groupby("status_tabela", as_index=False)
          .size()
          .rename(columns={"size": "Quantidade de inscritos"})
    )

    total = df_agg["Quantidade de inscritos"].sum()
    df_agg["%"] = df_agg["Quantidade de inscritos"] / total * 100

    df_agg = (
        df_agg.sort_values(
            by="Quantidade de inscritos",
            ascending=False
        )
        .reset_index(drop=True)
        .rename(columns={"status_tabela": "Situação da inscrição"})
    )

    return df_agg
```

Esse trecho usa `groupby`, do pandas, que agrupa linhas por categorias como ano, semestre, região, situação ou faixa de renda para calcular contagens, médias ou taxas.


------------------------------------------------------------
analise_022_17_artigo.py
------------------------------------------------------------

Linhas: 1056

Imports principais detectados: joblib, matplotlib.pyplot, numpy, pandas, pathlib, seaborn, sklearn.linear_model, sklearn.metrics, sklearn.pipeline, sklearn.preprocessing, textwrap, warnings

Funções definidas: orquestrador_analise_022, calcular_idade, gerar_abt_analise_022, treinar_modelo_logistico, avaliar_modelo, anexar_probabilidades_previstas, fmt_int, fmt_float, fmt_pct_decimal, fmt_pct_puro, quebrar_texto, salvar_tabela_como_imagem, gerar_tabela_metricas, nome_variavel_artigo, gerar_tabela_coeficientes_principais, gerar_tabela_top_coeficientes, selecionar_colunas_por_prefixo, gerar_tabela_especificacoes_alternativas, matriz_probabilidade_prevista, matriz_contagem_observacoes, gerar_figura_probabilidade_prevista, gerar_figura_matriz_confusao, gerar_figura_curva_roc, tabela_media_prob_por_faixas, gerar_figura_curvas_desempenho_por_renda, gerar_figura_curvas_renda_por_desempenho


O que faz:
Script grande que consolida uma análise de artigo: gera ABT, treina logit, avalia, anexa probabilidades previstas, gera tabelas de métricas, coeficientes, especificações alternativas, matrizes de probabilidade e figuras/apêndices. É uma ponte entre legado e robusto, porque já usa Pipeline e organiza resultados para o artigo.


Trecho real relevante e explicação:

```python
]

    df = pd.get_dummies(
        df_base,
        columns=colunas_categoricas,
        drop_first=True
    )

    df["renda_gap"] = df["renda_per_capita"] * df["gap"]

    grupos_dummies = [
        "beneficiado_creduc_fies_",
        "modalidade_fies_",
        "opcao_curso_",
        "concluiu_curso_superior_",
        "conceito_curso_gp_",
        "subarea_conhecimento_",
        "regiao_morar_",
        "natureza_juridica_mantenedora_",
        "etnia_cor_",
        "turno_",
        "ensino_medio_escola_publica_"
```

Esse trecho usa `pd.get_dummies`, do pandas, para transformar variáveis categóricas em várias colunas numéricas 0/1. Isso é necessário porque a regressão logística e a Random Forest do scikit-learn não trabalham diretamente com texto como 'NOTURNO', 'SUL' ou 'MEDICINA'. O `drop_first=True`, quando aparece, remove uma categoria de referência para reduzir redundância nas regressões.


Por que este arquivo é especial:
Ele é quase um 'mini robusto' dentro do legado. Já junta ABT, treino, avaliação, probabilidades, tabelas e apêndices num único script. O robusto depois separa essas responsabilidades em vários módulos.


------------------------------------------------------------
constantes.py
------------------------------------------------------------

Linhas: 243

Imports principais detectados: pathlib

Funções definidas: nenhuma função principal; script executa comandos diretamente ou define constantes.


O que faz:
Arquivo de infraestrutura. Centraliza caminhos do projeto: pastas raw/staging/transform/load/processed, caminhos de modelos, figuras, arquivos Parquet e CSV. A lógica é evitar escrever o mesmo caminho manualmente em todo script. No legado, ele também guarda muitos caminhos específicos de cada análise, como analise_007, analise_017, analise_022 etc.


Trecho real relevante e explicação:

```python
# pasta raiz 
pasta_raiz_projeto = Path(__file__).resolve().parent.parent

# pasta data 01 raw:
pasta_data_01_raw_microdata_fies = pasta_raiz_projeto / 'data' / '01_raw' / 'microdata fies'
pasta_data_01_raw_microdata_inep = pasta_raiz_projeto / 'data' / '01_raw' / 'microdata inep' # REMOVIDO str()

# pasta data 02 staging:
pasta_data_02_staging_microdata_fies = pasta_raiz_projeto / 'data' / '02_staging' / 'microdata fies'
pasta_data_02_staging_microdata_fies_errors = pasta_data_02_staging_microdata_fies / 'errors'
pasta_data_02_staging_microdata_inep = pasta_raiz_projeto / 'data' / '02_staging' / 'microdata inep' # REMOVIDO str()

# pasta data 03 transform:
pasta_data_03_transform_inep = pasta_raiz_projeto / 'data' / '03_transform' / 'inep'
pasta_data_03_transform_fies = pasta_raiz_projeto / 'data' / '03_transform' / 'fies'
pasta_data_03_temporarios = pasta_raiz_projeto / 'data' / '03_transform' / 'temporarios'

# pasta data 04 load:
pasta_data_04_load = pasta_raiz_projeto / 'data' / '04_load'

pasta_data_04_load_database = pasta_data_04_load / 'database'
```


------------------------------------------------------------
load.py
------------------------------------------------------------

Linhas: 648

Imports principais detectados: numpy, pandas, random, sqlalchemy, src.constantes

Funções definidas: load_inscritos, load_ofertas, auditoria_inscritos_carregados, auditoria_ofertas_carregadas, exportar_para_sqlite


O que faz:
Camada final de carregamento do legado. Recebe arquivos transformados, renomeia muitas colunas para nomes curtos e gera arquivos finais limpos em data/04_load/database. Também tem auditorias e exportação para SQLite. É a etapa que transforma os nomes longos dos microdados em colunas mais usáveis nas análises, como ano, semestre, renda_per_capita, media_enem, nota_corte_gp, situacao_fies etc.


Trecho real relevante e explicação:

```python
# 4. Salvamento
    print(f"[*] Salvando versão final (Parquet) em: {pasta_data_04_load_database.name}/...")
    df.to_parquet(str(arquivo_saida_parquet), index=False)
    
    # Opcional: Descomenta a linha abaixo se quiseres gerar um CSV limpo para ler no Excel/PowerBI
    # df.to_csv(str(arquivo_saida_csv), index=False, encoding='utf-8')

    print(f"\n[OK] Dataset de Inscritos PRONTO PARA ANÁLISE.")
    print(f"Colunas Finais: {len(df.columns)}")
    print("="*60)










def load_ofertas():
```

Esse trecho lê ou salva Parquet. Parquet é mais eficiente que CSV para pipeline de dados, preserva melhor tipos e carrega mais rápido.


------------------------------------------------------------
staging.py
------------------------------------------------------------

Linhas: 258

Imports principais detectados: numpy, os, pandas, pathlib, shutil, src.constantes

Funções definidas: limpeza_avancada_staging_fies, staging_fies, staging_inep


O que faz:
Primeira camada do pipeline legado. Copia/lê arquivos brutos, remove colunas “fantasma” do tipo Unnamed, transforma textos para maiúsculas, converte colunas financeiras/notas para número e grava CSVs mais limpos em staging. Diferente do robusto, aqui o staging já faz uma limpeza pesada; no robusto, staging é mais conservador e a transformação fica em módulos separados.


Trecho relevante: não foi encontrado automaticamente um padrão de modelagem/plotagem nos primeiros padrões buscados; este arquivo parece ser mais de constantes, organização ou lógica própria.


------------------------------------------------------------
transform_layer_1.py
------------------------------------------------------------

Linhas: 867

Imports principais detectados: os, pandas, pathlib, re, src.constantes

Funções definidas: transform_inep, validar_qualidade_inscritos, validar_qualidade_ofertas, transform_ofertas, transform_inscritos, auditoria_cine_inscritos, auditoria_cine_ofertas, verificar_colunas_inep


O que faz:
Primeira transformação grande. Trata INEP e FIES, padroniza colunas, cruza informações de cursos/instituições/CINE e gera bases transformadas. É onde aparecem validações de qualidade e auditorias de CINE. No legado, esse arquivo concentra muitas responsabilidades que no robusto foram divididas em limpeza_tipos_fies.py, unificacao_fies.py, mestre_inep.py e cruzamento_cine.py.


Trecho real relevante e explicação:

```python
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
```

Esse trecho lê ou salva Parquet. Parquet é mais eficiente que CSV para pipeline de dados, preserva melhor tipos e carrega mais rápido.


------------------------------------------------------------
transform_layer_2.py
------------------------------------------------------------

Linhas: 847

Imports principais detectados: numpy, pandas, pathlib, src.constantes

Funções definidas: tratar_nans_cine_inscritos, tratar_nans_cine_ofertas, auditoria_pos_correcao_inscritos, auditoria_pos_correcao_ofertas


O que faz:
Segunda transformação. Foca principalmente em tratar ausências e inconsistências de CINE/áreas de conhecimento nos inscritos e ofertas. Tem funções de correção e auditoria pós-correção. No robusto, parte disso migrou para cruzamento_cine.py, com diagnósticos e aplicação de mapas manuais.


Trecho real relevante e explicação:

```python
# Salvamos com o sufixo _corrigido para não perder o original em caso de erro
    arquivo_saida = pasta_data_03_transform_fies / 'fies_inscritos_unificado_corrigido.parquet'
    df_principal.to_parquet(str(arquivo_saida), index=False)
    
    print(f"  -> Arquivo final salvo com sucesso em: {arquivo_saida.name}")
    print("\n" + "="*60)
    print("✅ LAYER 2 CONCLUÍDA")
    print("="*60)
    auditoria_pos_correcao_inscritos()







def tratar_nans_cine_ofertas():
    print("\n" + "="*60)
    print("🛠️ INICIANDO LAYER 2: TRATAMENTO DE NaNs CINE - OFERTAS")
    print("="*60)
```

Esse trecho lê ou salva Parquet. Parquet é mais eficiente que CSV para pipeline de dados, preserva melhor tipos e carrega mais rápido.


------------------------------------------------------------
trasnform_layer_3.py
------------------------------------------------------------

Linhas: 201

Imports principais detectados: numpy, pandas, src.constantes

Funções definidas: processar_modalidades_e_peneira


O que faz:
Terceira transformação. Processa modalidade e “peneira”/classificações finais. É uma etapa curta, mas importante para adicionar/ajustar informações como modalidade_fies e recortes derivados. Observação: o nome do arquivo tem typo (“trasnform”), mas o Python consegue importar se o código usa exatamente esse nome.


Trecho real relevante e explicação:

```python
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
```


COMPARAÇÃO: `src_legacy` vs `src` ROBUSTO
=========================================

1. Organização do código
-----------------------
Legado:
- Arquivos soltos no mesmo nível: `analise_001.py`, `analise_017_0_com_insights.py`, `analise_022_17_artigo.py` etc.
- Estilo de notebook/script: cada arquivo carrega dados, cria variáveis, treina ou plota.
- Muitas funções repetidas: `gerar_ABT`, `TreinarModelo`, `acuracia_e_previsao`, `idade` aparecem em várias versões.

Robusto:
- Pacotes separados: `pipeline`, `analysis`, `abt`, `modeling`, `article`.
- `main.py` funciona como CLI e chama cada etapa.
- Funções centrais reaproveitáveis em `logit_binario_utils.py`, `logit_ternario_utils.py`, `treeClassification_profundidade_utils.py`.

Diferença prática: no legado você entende mais fácil porque é sequência direta. No robusto é mais difícil no começo, mas é melhor para reproduzir e manter.

2. Tratamento de categorias
---------------------------
Legado:

    pd.get_dummies(df_base, columns=[...], drop_first=True)

Isso cria as dummies imediatamente na base inteira.

Robusto:

    OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=20)

dentro de `ColumnTransformer` e `Pipeline`.

Novidade do robusto:
- O tratamento categórico é aprendido no treino e aplicado no teste.
- `handle_unknown="ignore"` evita erro quando aparece categoria nova no teste.
- `min_frequency=20` reduz categorias raras.
- A saída esparsa economiza memória.

3. Tratamento de valores ausentes
---------------------------------
Legado:
- Muitas análises usam filtros, `dropna`, preenchimentos manuais ou dependem de a base já estar limpa.

Robusto:

    SimpleImputer(strategy="median")
    SimpleImputer(strategy="constant", fill_value="Não informado")

Novidade do robusto:
- Numéricas ausentes viram mediana.
- Categóricas ausentes viram “Não informado”.
- Isso fica dentro do pipeline, então é aplicado igual no treino e no teste.

4. Padronização
---------------
Legado:

    Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', LogisticRegression(...))
    ])

Como as dummies já estavam no `X`, o scaler padroniza tudo: numéricas e dummies.

Robusto:
- Usa `ColumnTransformer`: escala só numéricas e aplica one-hot nas categóricas.

Novidade do robusto:
- Mais correto conceitualmente e mais claro para explicar.

5. Modelos
----------
Legado:
- Regressão logística simples e elasticnet.
- Random Forest binária/ternária.
- Algumas versões com base inteira, algumas por treino/teste temporal 2019 vs 2020–2021.

Robusto:
- Regressão logística binária e multinomial padronizadas.
- Árvores de decisão padrão e por profundidades 10/14/19.
- Não usa Random Forest como núcleo final.

Diferença conceitual:
- Random Forest é mais forte/flexível, mas menos interpretável.
- Árvore de decisão simples é mais fácil de explicar com regras, primeira divisão e importância de variáveis.

6. Avaliação
------------
Legado:
- Usa acurácia, matriz de confusão, classification_report, ROC-AUC.
- Em algumas versões avalia na própria base inteira.
- Em algumas versões usa separação temporal.

Robusto:
- Padroniza `in_sample` e `holdout_80_20`.
- Calcula métricas mais completas: balanced accuracy, precision, recall, F1, F1 macro/weighted, ROC-AUC OVR, log loss.

Novidade do robusto:
- Fica mais explícito quando é avaliação no mesmo conjunto e quando é teste fora da amostra.
- Métricas são salvas em JSON/CSV, não só impressas.

7. Probabilidades previstas
---------------------------
Legado:
- Já analisava probabilidades previstas e matrizes renda × gap.
- Isso aparece forte em `analise_017_0_com_insights.py`, `analise_017_2_0...` e `analise_022_17_artigo.py`.

Robusto:
- Formaliza isso em `efeitos_multinomiais_ternario.py`, `logit_binario.py`, `logit_ternario.py`.
- Cria grade controlada de renda e desempenho, mantendo outras variáveis em valores típicos.

Diferença: a ideia já existia no legado; o robusto deixa mais reprodutível e menos improvisado.

8. Exportação do artigo
-----------------------
Legado:
- Cada script salva figuras/tabelas em pastas próprias.
- O usuário precisa saber quais arquivos pegar.

Robusto:
- `pacote_artigo.py` copia e organiza tudo em `article/`.
- `main.py export article --clean` monta pacote final.

Novidade do robusto: fechamento de pipeline. Ele não só analisa; ele entrega os artefatos finais organizados.

9. O que de fato é “novo” para você aprender no robusto
-------------------------------------------------------
As novidades principais não são a ideia da pesquisa, mas a engenharia do código:
- `ColumnTransformer`.
- `SimpleImputer`.
- `OneHotEncoder` dentro do pipeline.
- `train_test_split(..., stratify=y)`.
- métricas macro/weighted.
- metadados JSON.
- wrappers pequenos para rodar combinações.
- arquitetura modular por responsabilidade.

Se você entende o legado, o robusto é a mesma lógica com menos improviso e mais organização.


MAPEAMENTO LEGADO -> ROBUSTO
============================
- `constantes.py` legado -> `src/constants.py` robusto.
- `staging.py` legado -> `src/pipeline/staging.py` + parte de `limpeza_tipos_fies.py` robusto.
- `transform_layer_1.py` legado -> `limpeza_tipos_fies.py`, `unificacao_fies.py`, `mestre_inep.py`, `cruzamento_cine.py` robusto.
- `transform_layer_2.py` legado -> principalmente `cruzamento_cine.py` robusto.
- `trasnform_layer_3.py` legado -> `modalidade.py` e parte de `curate.py` robusto.
- `load.py` legado -> `curate.py` robusto.
- `analise_001.py` -> `analysis/dataset_candidatos_unicos.py`.
- `analise_003*.py` -> `analysis/dataset_funil_fluxo.py` + `article/fluxo_selecao.py`.
- `analise_021_taxas.py` -> `analysis/dataset_taxas.py` + `article/taxas_conversao.py`.
- `analise_006*.py` e `analise_022.py` -> `article/tabelas_distribuicao.py` + `article/matrizes_renda_desempenho.py`.
- `analise_011.py` -> `article/financiamento_coparticipacao.py`.
- `analise_007` a `analise_019` -> `abt/build_abt_binaria.py`, `abt/build_abt_ternaria.py`, `modeling/logit_*`, `modeling/treeClassification_*`.
- `analise_022_17_artigo.py` -> no robusto foi quebrado em ABT + modelagem + article + apêndices.

CONCLUSÃO FINAL
===============
O `src_legacy` não está “errado”; ele é o primeiro formato natural de um projeto de pesquisa: scripts grandes, experimentos numerados, muita tentativa e evolução. A lógica dos principais algoritmos está correta: dummização, criação de target, regressão logística, elasticnet, Random Forest, ROC-AUC, matriz de confusão, probabilidades previstas e análises de renda/gap.

O `src` robusto não é uma pesquisa completamente diferente. Ele pega as mesmas ideias e coloca em uma arquitetura mais reprodutível. A diferença maior é engenharia de código: organizar, modularizar, evitar repetição, padronizar avaliação, salvar metadados e usar ferramentas mais seguras do scikit-learn para pré-processamento.

Para defender em aula, você pode dizer:

"No código antigo, eu fui construindo a pesquisa por scripts de análise: primeiro limpei os dados, depois gerei funil, tabelas, heatmaps e modelos. No código robusto, essa mesma lógica foi reorganizada em módulos: pipeline, análise, ABT, modelagem e exportação do artigo. A principal mudança técnica foi trocar etapas manuais, como `pd.get_dummies` solto, por pipelines do scikit-learn com imputação, one-hot encoding e avaliação padronizada."



APROFUNDAMENTO EXTRA: TRECHOS NÚCLEO DO LEGADO, COM EXPLICAÇÃO LINHA A LINHA
=============================================================================

Esta seção foi adicionada porque, para aprender de verdade, não basta dizer “treina modelo” ou “gera ABT”. Aqui estão os pedaços mais importantes do legado, explicando como a lógica funciona por dentro.

1) `staging.py` — função interna `converter_br_para_float`
----------------------------------------------------------

Trecho real:

```python
def converter_br_para_float(valor):
    if pd.isna(valor):
        return np.nan
    
    val_str = str(valor).strip().upper()
    if val_str in ['NAN', 'NONE', 'NULL', '', '-']:
        return np.nan
        
    # Se tem vírgula, trata o padrão BR (ex: 84.692,00 ou 480,38)
    if ',' in val_str:
        val_str = val_str.replace('.', '')  # Remove ponto de milhar
        val_str = val_str.replace(',', '.') # Transforma vírgula decimal em ponto
    
    try:
        return float(val_str)
    except:
        return np.nan
```

Explicação:
- `pd.isna(valor)` pergunta se o valor é ausente. Se for, devolve `np.nan`.
- `str(valor).strip().upper()` transforma o valor em texto, tira espaços das pontas e coloca em maiúsculas.
- A lista `['NAN', 'NONE', 'NULL', '', '-']` trata lixos comuns de dado bruto.
- Se tem vírgula, o código assume padrão brasileiro: `84.692,00` vira `84692.00`.
- `replace('.', '')` remove ponto de milhar.
- `replace(',', '.')` troca vírgula decimal por ponto decimal.
- `float(val_str)` converte texto em número.
- Se falhar, retorna `np.nan`, para não quebrar o pipeline.

Teoria por trás: microdados públicos brasileiros frequentemente misturam textos, números com vírgula e valores ausentes. Antes de qualquer modelo, renda, nota, percentual e valores financeiros precisam virar número de verdade.

2) `staging.py` — varredura universal das colunas
-------------------------------------------------

Trecho real do fluxo:

```python
for col in df.columns:
    col_lower = col.lower()

    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip().str.upper()
        lixos = ['NAN', 'NONE', 'NULL', '', '-', 'NÃO INFORMADO', 'NAO INFORMADO']
        df[col] = df[col].replace(lixos, np.nan)

    is_float_col = any(k in col_lower for k in [
        'renda', 'nota', 'média', 'media', 'redação', 'redacao', 'tecnologias', 'tec', 
        'percentual', 'valor bruto', 'semestre bruto', 'valor do curso', 
        'índice', 'indice', 'semestre fies'
    ])
```

Explicação:
- `for col in df.columns` passa por todas as colunas do DataFrame.
- Se a coluna é `object`, no pandas normalmente significa texto ou mistura de tipos.
- `.astype(str).str.strip().str.upper()` padroniza tudo como texto maiúsculo sem espaços extras.
- `replace(lixos, np.nan)` transforma textos inúteis em ausente real.
- `any(k in col_lower for k in [...])` verifica se o nome da coluna contém palavras como renda, nota, percentual.
- Se contém, o código entende que essa coluna deve ser numérica do tipo float.

Isso é uma limpeza “por heurística”. Heurística quer dizer: regra prática baseada no nome da coluna. Funciona bem quando os nomes são consistentes, mas pode errar se uma coluna tiver nome estranho. No robusto, essa ideia ficou mais controlada por listas/mapas de colunas esperadas.

3) `analise_007.py` — primeira ABT binária
-----------------------------------------

Trecho real:

```python
df_base['gap'] = df_base['media_enem'] - df_base['nota_corte_gp'] 

filtro_treino_base= (df_base['ano'] == 2019) & ( (df_base['semestre'] == 1) | (df_base['semestre'] == 2) )
filtro_teste_base= ~(df_base['ano'] == 2019) & ( (df_base['semestre'] == 1) | (df_base['semestre'] == 2) )

# DUMIZAR VARIÁVEIS CATEGÓRICAS
df = pd.get_dummies(df_base, columns=['nome_cine_area_geral'], drop_first=True)

contratados = ['CONTRATADA']
nao_contratados = ['NÃO CONTRATADO']

filtro = df['situacao_fies'].isin(contratados + nao_contratados)
df = df[filtro].copy()
```

Explicação:
- `gap` é a diferença entre média no Enem e nota de corte do grupo de preferência.
- Se `gap > 0`, o candidato ficou acima da nota de corte.
- Se `gap < 0`, ficou abaixo.
- O filtro de treino pega 2019.
- O filtro de teste pega anos diferentes de 2019, ou seja, 2020 e 2021.
- Isso é uma separação temporal simples: treina em ano anterior e testa nos seguintes.
- `pd.get_dummies(..., columns=['nome_cine_area_geral'])` transforma área CINE em colunas 0/1.
- O filtro final mantém só `CONTRATADA` e `NÃO CONTRATADO`.

Essa é a origem da formulação binária. O robusto depois formaliza isso em `build_abt_binaria.py`.

4) `analise_007.py` — regressão logística simples
-------------------------------------------------

Trecho real:

```python
modelo = LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42)
modelo.fit(X, Y.values.ravel())
joblib.dump(modelo, str(pasta_modelo_analise_007))
```

Explicação:
- `LogisticRegression(...)` cria o modelo.
- `max_iter=10000` dá até 10 mil iterações para convergir.
- `class_weight='balanced'` corrige desbalanceamento dando mais peso para classe menor.
- `random_state=42` deixa resultados mais reproduzíveis.
- `Y.values.ravel()` transforma o alvo em vetor 1D. O scikit-learn espera `y` como vetor, não como DataFrame coluna.
- `modelo.fit(X, y)` treina o modelo.
- `joblib.dump(...)` salva o modelo treinado em disco.

Teoria: a regressão logística aprende coeficientes. Cada coeficiente indica como uma variável aumenta ou reduz a chance da classe positiva, que no caso é contratação.

5) `analise_009.py` / `analise_017_0_com_insights.py` — Pipeline com scaler + logit elasticnet
------------------------------------------------------------------------------------------------

Trecho real:

```python
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
```

Explicação de cada peça:
- `Pipeline([...])`: cria uma sequência de etapas.
- `('scaler', StandardScaler())`: primeira etapa; padroniza colunas.
- `('modelo', LogisticRegression(...))`: segunda etapa; treina regressão logística.
- `penalty='elasticnet'`: usa regularização mista L1 + L2.
- `l1_ratio=0.5`: metade da regularização é L1, metade L2.
- `solver='saga'`: algoritmo necessário para elasticnet no scikit-learn.
- `C=0.1`: força inversa da regularização. Quanto menor C, mais forte a regularização.
- `class_weight='balanced'`: ajusta pesos das classes.
- `n_jobs=-1`: usa todos os núcleos disponíveis quando possível.

Diferença para o robusto:
- No legado, você primeiro faz `pd.get_dummies` e depois aplica `StandardScaler` em tudo.
- No robusto, `ColumnTransformer` separa numéricas e categóricas: numéricas recebem imputer/scaler; categóricas recebem imputer/one-hot.

6) `analise_017_2_0 y ternario.py` — target ternário
----------------------------------------------------

Trecho real:

```python
def mapear_status(x):
    if x == 'CONTRATADA':
        return 2
    elif x == 'LISTA DE ESPERA':
        return 1
    elif x == 'NÃO CONTRATADO':
        return 0
    else:
        return None
```

Explicação:
- Este código transforma texto em classe numérica.
- `CONTRATADA -> 2`.
- `LISTA DE ESPERA -> 1`.
- `NÃO CONTRATADO -> 0`.
- Outros status viram `None` e normalmente são removidos depois.

Teoria: modelo de classificação precisa de um alvo numérico. No ternário, o modelo estima três probabilidades. Isso é diferente da binária: aqui ele precisa separar a barreira inicial da lista de espera e a etapa contratual.

Observação: no robusto, o mapeamento ficou `LISTA DE ESPERA = 0`, `NÃO CONTRATADO = 1`, `CONTRATADA = 2`. A ordem numérica pode mudar desde que o código mantenha consistência entre treino, métricas e rótulos. O importante é não trocar o significado das classes depois.

7) `analise_012_0 com insights.py` — Random Forest
--------------------------------------------------

Trecho real:

```python
modelo = RandomForestClassifier(
    max_depth=max_depth,
    n_estimators=1000,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42,
    min_samples_leaf=125,
    max_features='sqrt'
)

modelo.fit(X_treino, Y_treino)
```

Explicação:
- `RandomForestClassifier`: cria uma floresta aleatória.
- `n_estimators=1000`: treina 1000 árvores.
- `max_depth=max_depth`: limita profundidade das árvores.
- `min_samples_leaf=125`: cada folha precisa ter pelo menos 125 registros, evitando regras minúsculas.
- `max_features='sqrt'`: cada divisão considera uma amostra de variáveis; isso torna as árvores mais diferentes entre si.
- `class_weight='balanced'`: corrige desbalanceamento.
- `n_jobs=-1`: usa processamento paralelo.

Teoria: uma árvore sozinha pode decorar demais os dados. A floresta reduz isso combinando muitas árvores. Porém, perde interpretabilidade: é mais difícil explicar 1000 árvores do que uma árvore única. Por isso, para artigo técnico e apresentação, árvore de decisão simples pode ser mais defensável.

8) `analise_011.py` — renda vs percentual financiado
----------------------------------------------------

Trecho real:

```python
r, p_value = pearsonr(
    df["renda_per_capita"],
    df["percentual_financiamento"]
)

modelo = LinearRegression()
modelo.fit(x, y)

beta = modelo.coef_[0]
intercepto = modelo.intercept_
r2 = modelo.score(x, y)
efeito_100_reais = beta * 100
```

Explicação:
- `pearsonr` calcula correlação linear entre renda e percentual financiado.
- `LinearRegression()` ajusta uma reta.
- `beta` é a inclinação: quanto o percentual financiado muda quando renda aumenta 1 real.
- `efeito_100_reais = beta * 100` deixa a interpretação mais fácil: mudança a cada R$ 100.
- `r2` indica quanto da variação do percentual é explicada pela renda nessa regressão simples.

Cuidado: isso não prova causalidade; mostra associação entre renda e cobertura entre contratos observados.

9) `analise_022_17_artigo.py` — ABT consolidada para o artigo
-------------------------------------------------------------

Trecho real:

```python
df_base["situacao_fies"] = (
    df_base["situacao_fies"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df_base["renda_per_capita"] = pd.to_numeric(
    df_base["renda_per_capita"],
    errors="coerce"
)

df_base["media_enem"] = pd.to_numeric(
    df_base["media_enem"],
    errors="coerce"
)

df_base["nota_corte_gp"] = pd.to_numeric(
    df_base["nota_corte_gp"],
    errors="coerce"
)

df_base["gap"] = df_base["media_enem"] - df_base["nota_corte_gp"]
df_base["idade"] = df_base["data_nascimento"].apply(calcular_idade)
```

Explicação:
- Padroniza `situacao_fies` para evitar diferença entre minúscula/maiúscula/espaço.
- Converte renda, média Enem e nota de corte para número.
- `errors='coerce'` transforma valor inválido em `NaN`.
- Cria `gap`.
- Cria `idade` a partir da data de nascimento.

Isso é a construção real da base analítica: limpar, criar variáveis substantivas e preparar para modelo.

10) `analise_022_17_artigo.py` — avaliação com melhor threshold
---------------------------------------------------------------

Trecho real:

```python
prob = modelo.predict_proba(X)[:, 1]

auc = roc_auc_score(y, prob)
fpr, tpr, thresholds = roc_curve(y, prob)

melhor_indice = np.argmax(tpr - fpr)
threshold = thresholds[melhor_indice]

y_pred = (prob >= threshold).astype(int)
```

Explicação:
- `predict_proba(X)[:, 1]` pega probabilidade de contratação.
- `roc_auc_score(y, prob)` calcula ROC-AUC.
- `roc_curve` gera vários thresholds possíveis.
- `tpr` é taxa de verdadeiros positivos, ou recall da classe positiva.
- `fpr` é taxa de falsos positivos.
- `np.argmax(tpr - fpr)` escolhe o threshold que maximiza a diferença entre acertar positivos e errar negativos. Isso é uma versão do critério de Youden.
- Depois, `y_pred = (prob >= threshold)` transforma probabilidade em classe prevista.

Diferença importante: muitos códigos usam threshold fixo 0,5. Aqui você escolhe threshold a partir da curva ROC. Isso pode melhorar equilíbrio, mas precisa ser explicado, porque muda matriz de confusão e taxa prevista.

11) `analise_022_17_artigo.py` — matriz de probabilidade prevista
-----------------------------------------------------------------

Trecho real:

```python
matriz = (
    df_pred
    .groupby(["faixa_desempenho", "faixa_renda"], observed=True)["prob_contratacao"]
    .mean()
    .reset_index()
    .pivot(index="faixa_desempenho", columns="faixa_renda", values="prob_contratacao")
    .reindex(index=ORDEM_DESEMPENHO_PLOT, columns=LABELS_RENDA)
)
```

Explicação:
- Agrupa por faixa de desempenho e faixa de renda.
- Calcula a média da probabilidade prevista em cada célula.
- `pivot` transforma linhas em matriz: linhas = desempenho, colunas = renda.
- `reindex` força uma ordem lógica das faixas.

Essa é a lógica que vira heatmap/matriz no artigo: em vez de olhar registro por registro, você resume a probabilidade média por perfil de renda e desempenho.

12) O que mudou desse núcleo no robusto
---------------------------------------

No robusto, os pedaços acima viraram módulos separados:

- conversão/limpeza pesada -> `pipeline/transform/limpeza_tipos_fies.py`;
- gap/target -> `pipeline/curate.py` e `abt/build_abt_*.py`;
- dummies manuais -> `OneHotEncoder` dentro de `ColumnTransformer`;
- scaler solto -> scaler só nas numéricas;
- avaliação solta -> funções `calcular_metricas` padronizadas;
- Random Forest exploratória -> árvores de decisão por profundidade;
- probabilidades por renda/gap -> `efeitos_multinomiais_ternario.py`;
- script grandão do artigo -> `article/*.py` + `pacote_artigo.py`.

Então, se você já entende esses trechos do legado, o robusto fica menos assustador: ele faz quase a mesma coisa, só que de forma mais dividida e segura.
