# GUIA TOTAL DO `src` ROBUSTO — FIES 2019–2021

Arquivo gerado em 2026-06-19 17:02.

Este guia foi feito para explicar **o `src` robusto**, não o legado, com linguagem de quem ainda está consolidando a parte de machine learning. A ideia é você conseguir olhar para o código e entender: o que roda primeiro, por que roda, o que cada biblioteca faz, o que significa `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `LogisticRegression`, `DecisionTreeClassifier`, `predict_proba`, F1 macro, ROC-AUC, matriz de confusão, ABT e exportação de artefatos.

Observação honesta: esta é uma **análise estática**. Eu li o código, extraí trechos reais e expliquei a lógica. Eu não rodei o pipeline completo com os microdados brutos oficiais. Então, quando eu digo que a lógica está correta, estou falando da implementação e do encadeamento do código, não de uma auditoria executada ponta a ponta com a base bruta.

---

## 1. Ideia central do `src` robusto

O `src` robusto é um pipeline de pesquisa. Ele não é apenas um script que treina um modelo. Ele faz um fluxo completo:

```text
Dados brutos oficiais
        ↓
staging: cópia organizada dos arquivos brutos
        ↓
transform: limpeza, tipos, unificação, INEP, CINE, modalidade
        ↓
curate: base curada final 2019–2021
        ↓
analysis: bases descritivas, funil e taxas
        ↓
abt: tabelas finais para modelagem
        ↓
modeling: regressão logística e árvores
        ↓
article: tabelas, figuras e apêndices
        ↓
export: pacote final article/
```

No legado, muita coisa estava em arquivos `analise_00x.py`, quase como notebooks salvos em `.py`. No robusto, a mesma pesquisa fica separada por responsabilidade:

- `pipeline/`: preparar os dados;
- `analysis/`: gerar bases descritivas;
- `abt/`: gerar bases de modelagem;
- `modeling/`: treinar modelos e calcular métricas;
- `article/`: gerar produtos do artigo;
- `main.py`: orquestrar tudo pela linha de comando.

---

## 2. O que quer dizer “modelagem padronizada” no robusto?

Quando eu digo que no robusto a regressão logística e as árvores ficaram “padronizadas”, não significa que todos os modelos são iguais. Significa que o código passa a seguir um mesmo padrão organizado:

1. A base de modelagem sempre vem de uma ABT (`data/06_abt`).
2. As variáveis são escolhidas por experimentos E1–E5.
3. As colunas são separadas em numéricas e categóricas.
4. O pré-processamento é feito dentro de um `Pipeline` do scikit-learn.
5. A avaliação pode ser `in_sample` ou `holdout_80_20`.
6. As métricas são calculadas por funções próprias.
7. O modelo, as métricas, as importâncias e os metadados são salvos em pastas padronizadas.

Então “padronizado” é isto: **todo mundo entra pelo mesmo portão, passa pelos mesmos tipos de limpeza, usa nomes consistentes e salva os resultados no mesmo formato**.

No legado, você muitas vezes fazia assim:

```python
X = pd.get_dummies(df[features], drop_first=True)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', LogisticRegression(...))
])
pipeline.fit(X, y)
```

Isso funciona. Mas como o `get_dummies` já transformava tudo em número antes, o `StandardScaler` podia acabar padronizando tanto variáveis numéricas quanto dummies 0/1. Não é necessariamente “errado”, mas é menos limpo para explicar.

No robusto, a ideia virou:

```python
preprocessador = ColumnTransformer([
    ('num', pipeline_numerico, colunas_numericas),
    ('cat', pipeline_categorico, colunas_categoricas),
])
modelo = Pipeline([
    ('preprocessador', preprocessador),
    ('modelo', LogisticRegression(...))
])
```

A grande diferença é: **o código sabe quais colunas são números e quais são categorias**. Ele escala só as numéricas e faz one-hot só nas categóricas.

---

## 3. O que é `OneHotEncoder`? Por que ele substitui `pd.get_dummies`?

Modelo de machine learning não entende texto bruto. Ele não entende diretamente:

```text
turno = NOTURNO
região = SUL
curso = MEDICINA
```

Ele precisa de números. O `OneHotEncoder` transforma categorias em colunas 0/1.

Exemplo:

```text
turno
-----
MANHÃ
NOITE
TARDE
```

vira:

```text
turno_MANHÃ | turno_NOITE | turno_TARDE
     1      |      0      |      0
     0      |      1      |      0
     0      |      0      |      1
```

No robusto aparece assim:

```python
def criar_preprocessador(numericas: list[str], categoricas: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Não informado")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    min_frequency=20,
                ),
            ),
        ]
    )

    transformers = []

    if numericas:
        transformers.append(("num", numeric_transformer, numericas))

    if categoricas:
        transformers.append(("cat", categorical_transformer, categoricas))

    if not transformers:
        raise ValueError("Nenhuma variável válida foi informada ao preprocessador.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
```


Explicação do trecho:

- `SimpleImputer(strategy="median")`: nas variáveis numéricas, preenche valor ausente com a mediana.
- `StandardScaler()`: na regressão logística, padroniza as numéricas para média 0 e desvio padrão 1.
- `SimpleImputer(... fill_value="Não informado")`: nas categóricas, troca vazio por “Não informado”.
- `OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=20)`: transforma categorias em colunas 0/1.
- `handle_unknown="ignore"`: se no teste aparecer categoria nova que não apareceu no treino, o modelo não quebra.
- `min_frequency=20`: categorias raras são tratadas como infrequentes, evitando uma explosão de colunas para categorias quase inexistentes.
- `remainder="drop"`: qualquer coluna que não esteja explicitamente listada em numéricas/categóricas fica fora do modelo.

O `pd.get_dummies` do legado fazia a mesma ideia de transformar texto em dummies, mas fora do pipeline. O `OneHotEncoder` dentro de `ColumnTransformer` é mais seguro porque o encoder aprende as categorias no treino e aplica a mesma regra no teste.

---

## 4. O que é `StandardScaler` e o que significa “padronizar”?

Padronizar é transformar uma variável assim:

```text
valor_padronizado = (valor - média_da_coluna) / desvio_padrão_da_coluna
```

Exemplo simples:

- renda pode estar em reais: 600, 1200, 3000;
- idade pode estar em anos: 18, 25, 40;
- gap pode estar em pontos de nota: -200, 0, 150.

Essas escalas são muito diferentes. Para regressão logística, isso pode atrapalhar a otimização e a comparação dos coeficientes. O `StandardScaler` coloca as numéricas numa escala comum: média 0 e desvio padrão 1.

Por isso, no robusto, a regressão logística usa `StandardScaler`. A árvore de decisão não usa scaler nas numéricas porque árvore trabalha com cortes do tipo:

```text
renda_per_capita <= 1200?
gap <= -50?
nota_corte_gp <= 700?
```

Para árvore, tanto faz se renda está em reais ou padronizada: ela procura limiares/cortes.

---

## 5. O que é `Pipeline`?

`Pipeline` é uma esteira. Ele garante que o mesmo tratamento dos dados seja aplicado no treino e depois no teste.

Sem pipeline, você poderia fazer manualmente:

```python
X_treino_limpo = preprocessar(X_treino)
modelo.fit(X_treino_limpo, y_treino)
X_teste_limpo = preprocessar(X_teste)
modelo.predict(X_teste_limpo)
```

O problema é que você pode, sem querer, preprocessar treino e teste de formas diferentes. O pipeline evita isso.

No robusto:

```python
def criar_pipeline(numericas: list[str], categoricas: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessador", criar_preprocessador(numericas, categoricas)),
            ("modelo", criar_modelo()),
        ]
    )
```


Quando o código chama:

```python
modelo.fit(X_treino, y_treino)
```

o pipeline faz por dentro:

1. `preprocessador.fit_transform(X_treino)`;
2. `LogisticRegression.fit(X_treino_transformado, y_treino)`.

Quando chama:

```python
modelo.predict_proba(X_teste)
```

o pipeline faz:

1. `preprocessador.transform(X_teste)`;
2. `LogisticRegression.predict_proba(X_teste_transformado)`.

---

## 6. O que é `ColumnTransformer`?

`ColumnTransformer` é o objeto que separa o que fazer com cada tipo de coluna.

Ele permite dizer:

```text
colunas numéricas → imputar mediana + escalar
colunas categóricas → imputar “Não informado” + one-hot
outras colunas → descartar
```

No legado, depois de `get_dummies`, tudo já era número e o scaler podia passar em tudo. No robusto, a separação fica explícita e mais correta para defender.

---

## 7. O que é regressão logística no código?

Regressão logística é um modelo de classificação. No binário, ele calcula a probabilidade de uma classe positiva. No seu caso:

```text
0 = NÃO CONTRATADO
1 = CONTRATADA
```

A ideia matemática é:

```text
z = b0 + b1*x1 + b2*x2 + ... + bk*xk
p = 1 / (1 + exp(-z))
```

`p` vira a probabilidade de classe 1, ou seja, probabilidade prevista de contratação.

No robusto, o modelo binário é criado assim:

```python
def criar_modelo() -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        C=0.1,
        class_weight="balanced",
        max_iter=5000,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )
```


Explicação:

- `solver="lbfgs"`: algoritmo que ajusta os coeficientes.
- `C=0.1`: controla regularização. Menor `C` = mais regularização = coeficientes menos exagerados.
- `class_weight="balanced"`: dá mais peso à classe menor. Como contratada é minoria, isso evita o modelo simplesmente aprender a prever quase tudo como não contratado.
- `max_iter=5000`: permite muitas iterações para convergir.
- `random_state=RANDOM_STATE`: reprodutibilidade.

---

## 8. O que é árvore de decisão no código?

Árvore de decisão aprende regras sucessivas. Exemplo conceitual:

```text
gap <= -150?
    sim → maior chance de lista de espera
    não → renda <= 1200?
        sim → ...
        não → ...
```

A árvore escolhe cortes que deixam os grupos mais “puros”. Pureza significa que dentro de um nó há predominância de uma classe. O critério usado é Gini.

No robusto, a árvore por profundidade é criada assim:

```python
def criar_modelo(n_train: int, recorte: str, profundidade: int | str) -> DecisionTreeClassifier:
    profundidade = validar_profundidade(profundidade)
    leaf = min_leaf_dinamico(n_train, recorte, profundidade)
    split = min_split_dinamico(n_train, recorte, profundidade)

    return DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=profundidade,
        min_samples_leaf=leaf,
        min_samples_split=split,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
```


Explicação:

- `criterion="gini"`: usa impureza Gini para escolher cortes.
- `splitter="best"`: procura o melhor corte em cada nó.
- `max_depth=profundidade`: limita quantos níveis a árvore pode ter.
- `min_samples_leaf=leaf`: impede folha pequena demais.
- `min_samples_split=split`: impede dividir nó pequeno demais.
- `class_weight="balanced"`: compensa desbalanceamento das classes.
- `random_state`: reprodutibilidade.

---

## 9. O que são profundidades 10, 14 e 19?

Profundidade é o número máximo de níveis que a árvore pode crescer. Quanto maior a profundidade, mais complexas podem ficar as regras.

- Profundidade 10: árvore mais simples, menos risco de decorar detalhes.
- Profundidade 14: intermediária.
- Profundidade 19: mais flexível, pode capturar relações mais específicas.

Mas profundidade maior pode gerar overfitting. Por isso o robusto também usa pré-poda:

```python
def min_leaf_dinamico(n_train: int, recorte: str, profundidade: int | str) -> int:
    profundidade = validar_profundidade(profundidade)
    profile = TREE_DEPTH_PROFILES[profundidade]

    # Pré-poda: nunca permite folha com menos de 400 registros.
    # Em bases muito grandes, aumenta discretamente esse piso para evitar folhas muito pequenas.
    return int(max(profile["min_samples_leaf_min"], round(n_train * 0.0005)))
```


Isso quer dizer: a folha da árvore nunca pode ter menos que um piso mínimo. Em base muito grande, o piso aumenta um pouco. Isso evita regra baseada em poucos casos.

---

## 10. O que é `predict_proba`?

`predict_proba` retorna probabilidades, não apenas a classe final.

Binário:

```python
modelo.predict_proba(X)
```

pode retornar:

```text
[0.82, 0.18]
```

Significa:

```text
82% não contratado
18% contratado
```

No binário, o código pega a coluna da classe positiva:

```python
proba_teste = modelo.predict_proba(X_teste)[:, 1]
```

`[:, 1]` significa: todas as linhas, coluna 1. Como classe 1 é contratada, pega a probabilidade de contratação.

No ternário, `predict_proba` retorna três colunas:

```text
classe 0 = lista de espera
classe 1 = não contratado
classe 2 = contratada
```

A classe prevista é a maior probabilidade.

---

## 11. O que é `in_sample` e `holdout_80_20`?

No robusto, o split aparece assim:

```python
def criar_split(X: pd.DataFrame, y: pd.Series, avaliacao: str):
    indices = np.arange(len(X))

    if avaliacao == "in_sample":
        return indices, indices, "in_sample"

    if avaliacao == "holdout_80_20":
        train_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
            shuffle=True,
        )

        return train_idx, test_idx, "holdout_80_20_estratificado"

    raise ValueError("avaliacao deve ser 'in_sample' ou 'holdout_80_20'.")
```


`in_sample`:

- treina e avalia no mesmo conjunto;
- é útil para diagnóstico e geração de efeitos previstos;
- não deve ser tratado como desempenho fora da amostra.

`holdout_80_20`:

- separa 80% para treino e 20% para teste;
- usa `stratify=y`, ou seja, preserva a proporção das classes;
- é mais adequado para avaliar desempenho preditivo.

---

## 12. O que são as métricas?

No binário, o cálculo aparece assim:

```python
def calcular_metricas(y_true, proba) -> dict:
    y_pred = (proba >= 0.5).astype(int)
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - proba, proba]), labels=[0, 1])),
        "tn": int(matriz[0, 0]),
        "fp": int(matriz[0, 1]),
        "fn": int(matriz[1, 0]),
        "tp": int(matriz[1, 1]),
    }
```


Interpretação:

- `y_pred = (proba >= 0.5).astype(int)`: se a probabilidade de contratação for pelo menos 50%, prediz contratada.
- `confusion_matrix`: gera matriz real vs previsto.
- `roc_auc`: mede a capacidade de separar contratado e não contratado pelas probabilidades.
- `accuracy`: porcentagem total de acertos.
- `balanced_accuracy`: média do acerto por classe, melhor quando há desbalanceamento.
- `precision`: quando previu contratada, quantas eram contratadas.
- `recall`: das contratadas reais, quantas encontrou.
- `f1`: equilíbrio entre precision e recall.
- `log_loss`: pune probabilidades erradas, principalmente quando o modelo erra com muita confiança.

No ternário/árvore, o cálculo multiclasse aparece assim:

```python
def calcular_metricas(target: str, y_true, proba, classes) -> dict:
    cfg = target_config(target)
    classes_esperadas = cfg["classes"]
    y_pred = predicoes_por_proba(proba, classes)
    matriz = confusion_matrix(y_true, y_pred, labels=classes_esperadas)

    if normalizar_target(target) == "binario":
        metricas = {
            "roc_auc": calcular_roc_auc_binario(y_true, proba, classes, cfg["positive_class"]),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, pos_label=cfg["positive_class"], zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=cfg["positive_class"], zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=cfg["positive_class"], zero_division=0)),
            "log_loss": float(log_loss(y_true, proba, labels=list(classes))),
            "tn": int(matriz[0, 0]),
            "fp": int(matriz[0, 1]),
            "fn": int(matriz[1, 0]),
            "tp": int(matriz[1, 1]),
        }
        return metricas

    metricas = {
        "roc_auc_ovr_weighted": calcular_roc_auc_multiclasse(y_true, proba, classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=list(classes))),
    }

    for i, classe_real in enumerate(classes_esperadas):
        for j, classe_pred in enumerate(classes_esperadas):
            metricas[f"confusao_real_{classe_real}_pred_{classe_pred}"] = int(matriz[i, j])

    return metricas
```


`f1_macro` significa: calcula o F1 de cada classe separadamente e depois tira média simples. Isso é útil porque lista de espera, não contratado e contratada têm tamanhos diferentes. A classe maior não deve esconder o desempenho ruim na classe menor.

---

# 13. Fluxo detalhado por comando do `main.py`

O `main.py` é o orquestrador. Ele não contém o algoritmo pesado. Ele lê o comando do terminal e chama as funções certas dentro de `src`.

Exemplo:

```bash
python3 main.py pipeline all
```

O `main.py` interpreta:

```text
comando = pipeline
etapa = all
```

E chama:

```text
staging.run()
executar_transform("all")
curate.run()
```

O fluxo recomendado fica assim:

```bash
python3 main.py pipeline all
python3 main.py analysis all
python3 main.py article all
python3 main.py abt binaria
python3 main.py abt ternaria --recorte geral
python3 main.py abt ternaria --recorte medicina
python3 main.py modeling logit --force
python3 main.py modeling tree-depth --force
python3 main.py article modelagem
python3 main.py export article --avaliacao in_sample --clean
```

Em linguagem natural:

1. `pipeline all`: prepara e cura os dados.
2. `analysis all`: gera bases descritivas.
3. `article all`: gera produtos descritivos.
4. `abt binaria`: monta base de modelagem contratada vs não contratada.
5. `abt ternaria`: monta base lista de espera vs não contratada vs contratada.
6. `modeling logit`: treina regressões logísticas.
7. `modeling tree-depth`: treina árvores 10/14/19.
8. `article modelagem`: transforma resultados de modelo em tabelas/figuras.
9. `export article`: organiza tudo na pasta final.

---

# 14. Pipeline de dados: como o robusto sai do bruto até a base curada

## 14.1 `staging.py`: copiar, padronizar nome e remover duplicatas exatas

O staging é uma etapa conservadora. Ele lê os dados brutos ainda como texto, porque nesta fase o código não quer decidir tipos de colunas de forma apressada.

Trecho real:
```python
def staging_fies(overwrite: bool = True) -> None:
    """
    Executa o staging dos arquivos FIES encontrados em data/01_raw/fies.

    Esta etapa:
    - reconhece inscrições e ofertas;
    - aceita os anos definidos em PIPELINE_FIES_YEARS;
    - preserva os dados como texto;
    - remove apenas duplicatas exatas;
    - remove apenas colunas fantasma tipo Unnamed;
    - envia arquivos de resultado/erro/não classificados para errors.

    Observação:
    - Ofertas ausentes para determinado ano não são problema no staging.
    - A decisão de cruzar inscrições e ofertas por mesmo ano/semestre fica nas transformações.
    """
    STAGING_FIES_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_FIES_ERRORS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("STAGING FIES")
    log("=" * 80)

    arquivos = sorted(RAW_FIES_DIR.glob("*.csv"))

    processados = 0
    enviados_errors = 0
    ignorados_por_ano = 0
    erros = 0

    anos_aceitos = set(PIPELINE_FIES_YEARS)

    for origem in arquivos:
        nome_original = origem.name
        nome_lower = nome_original.lower()

        if "resultado" in nome_lower or "erro" in nome_lower:
            copiar_para_errors(origem, "ERROS")
            enviados_errors += 1
            continue

        nome_saida, ano = nome_saida_fies(nome_original)

        if nome_saida is None:
            copiar_para_errors(origem, "NÃO CLASSIFICADO")
            enviados_errors += 1
            continue

        if ano not in anos_aceitos:
            log(f"[IGNORADO] {nome_original} | ano fora de PIPELINE_FIES_YEARS: {ano}")
            ignorados_por_ano += 1
            continue

        destino = STAGING_FIES_DIR / nome_saida

        if destino.exists() and not overwrite:
            log(f"[PULADO] Já existe: {destino.name}")
            continue

        try:
            df = read_fies_csv(origem)

            linhas_originais = len(df)
            colunas_originais = len(df.columns)

            df = remove_colunas_fantasma(df)
            df = df.drop_duplicates()

            linhas_finais = len(df)
            duplicatas = linhas_originais - linhas_finais

            df.to_csv(destino, index=False, encoding="utf-8")

            log(
                f"[OK] {nome_original} -> {nome_saida} | "
                f"linhas: {linhas_originais} -> {linhas_finais} | "
                f"duplicatas removidas: {duplicatas} | "
                f"colunas: {colunas_originais} -> {len(df.columns)}"
            )

            processados += 1

        except Exception as error:
            copiar_para_errors(origem, "ERRO")
            log(f"       Motivo: {safe_text(error)}")
            erros += 1

    log(
        f"FIES concluído. Processados: {processados}. "
        f"Enviados para errors: {enviados_errors}. "
        f"Ignorados por ano: {ignorados_por_ano}. "
        f"Erros: {erros}."
    )
    log("")
```

O que acontece:

- `RAW_FIES_DIR.glob("*.csv")`: lista todos os CSVs brutos do FIES.
- `read_fies_csv(origem)`: lê o CSV mantendo colunas como texto.
- `remove_colunas_fantasma(df)`: remove colunas tipo `Unnamed: 0`.
- `df.drop_duplicates()`: remove linhas idênticas.
- `df.to_csv(...)`: salva a cópia organizada em `data/02_staging`.

Isso não é modelagem. É controle de entrada.

## 14.2 `limpeza_tipos_fies.py`: converter números, datas e nomes

Esta etapa pega staging e cria arquivos Parquet limpos.

Trecho real de conversão de número brasileiro:
```python
def converter_numero_br(serie: pd.Series) -> pd.Series:
    """
    Converte números em padrão brasileiro para float.

    Exemplos:
        2066,67 -> 2066.67
        84.692,00 -> 84692.00
        600 -> 600.0
        - -> NaN
    """
    s = serie.astype("string").str.strip()

    s = s.replace(
        {
            "": pd.NA,
            " ": pd.NA,
            "-": pd.NA,
            "--": pd.NA,
            "NAN": pd.NA,
            "nan": pd.NA,
            "NONE": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
        }
    )

    tem_virgula = s.str.contains(",", na=False)

    s = s.mask(
        tem_virgula,
        s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )

    return pd.to_numeric(s, errors="coerce")
```

O problema resolvido aqui: no Brasil, números podem vir como `1.234,56`. Para Python, isso precisa virar `1234.56`.

Trecho real de transformação de inscrições:
```python
def transformar_inscricoes(
    df: pd.DataFrame,
    ano_arquivo: int,
    semestre_arquivo: int,
    arquivo_origem: str,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [limpar_nome_coluna(col) for col in df.columns]

    registrar_colunas_nao_mapeadas(df, MAPA_INSCRICOES, arquivo_origem)

    df = df.rename(columns=MAPA_INSCRICOES)
    df = coalescer_colunas_duplicadas(df)

    df["arquivo_origem"] = arquivo_origem
    df["ano_arquivo"] = ano_arquivo
    df["semestre_arquivo"] = semestre_arquivo

    if "data_nascimento" in df.columns:
        df["data_nascimento"] = pd.to_datetime(
            df["data_nascimento"],
            errors="coerce",
            dayfirst=True,
        )

        ano_derivado = df["data_nascimento"].dt.year

        if "ano_nascimento" not in df.columns:
            df["ano_nascimento"] = ano_derivado
        else:
            df["ano_nascimento"] = df["ano_nascimento"].where(
                df["ano_nascimento"].astype("string").str.strip().ne(""),
                ano_derivado,
            )
    else:
        df["data_nascimento"] = pd.NaT

    if "semestre_financiamento" in df.columns:
        df["semestre_financiamento"] = df["semestre_financiamento"].map(normalizar_semestre_financiamento).astype("Int64")

    for col in FLOAT_COLS_INSCRICOES:
        if col in df.columns:
            df[col] = converter_numero_br(df[col])

    for col in INT_COLS_INSCRICOES:
        if col in df.columns:
            df[col] = converter_inteiro(df[col])

    if "cnpj_mantenedora" in df.columns:
        df["cnpj_mantenedora"] = df["cnpj_mantenedora"].map(limpar_cnpj).astype("string")

    ignorar_texto = set(FLOAT_COLS_INSCRICOES + INT_COLS_INSCRICOES + ["data_nascimento", "semestre_financiamento"])
    df = normalizar_textos(df, ignorar=ignorar_texto)

    return df
```

A lógica:

- limpa nomes das colunas;
- renomeia com mapas internos;
- cria metadados de origem;
- converte datas com `pd.to_datetime`;
- converte colunas numéricas;
- normaliza textos;
- salva em Parquet.

`Parquet` é usado porque é mais eficiente que CSV, preserva melhor tipos e lê mais rápido.

## 14.3 `unificacao_fies.py`: empilhar anos e semestres

Depois de limpar cada arquivo, o robusto junta tudo.

Trecho real:
```python
def ler_parquets(arquivos: list[Path], tipo: str) -> tuple[pd.DataFrame, list[dict]]:
    """
    Lê e empilha arquivos parquet de um mesmo tipo.

    Não aplica filtro analítico.
    Não faz join.
    Não remove registros por chave.
    """
    dfs = []
    registros_log = []

    for path in arquivos:
        log(f"[INÍCIO] Lendo {path.name}")

        df = pd.read_parquet(path)

        registros_log.append(
            {
                "tipo": tipo,
                "arquivo": path.name,
                "linhas": len(df),
                "colunas": len(df.columns),
                "ano_min": df["ano_arquivo"].min() if "ano_arquivo" in df.columns else pd.NA,
                "ano_max": df["ano_arquivo"].max() if "ano_arquivo" in df.columns else pd.NA,
                "semestres": ", ".join(
                    map(str, sorted(df["semestre_arquivo"].dropna().unique()))
                ) if "semestre_arquivo" in df.columns else pd.NA,
            }
        )

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(), registros_log

    return pd.concat(dfs, ignore_index=True, sort=False), registros_log
```

`pd.concat` empilha tabelas: se você tem 2019-1, 2019-2, 2020-1 etc., ele coloca uma embaixo da outra.

## 14.4 `mestre_inep.py`: base de curso/instituição do INEP

O FIES tem informação de curso, mas o INEP ajuda a padronizar e enriquecer informações. Esse arquivo lê cadastro de cursos, padroniza colunas e escolhe o registro mais recente quando há repetição.

Trecho real:
```python
def construir_ultimo_registro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma dimensão deduplicada por codigo_curso, mantendo o registro mais recente.

    Esta tabela é útil como fallback de consulta. Para análise temporal mais estrita,
    o cruzamento seguinte pode preferir ano_censo correspondente ao ano do FIES.
    """
    df_valid = df[df["codigo_curso"].notna()].copy()

    if df_valid.empty:
        return df_valid

    df_valid = df_valid.sort_values(
        by=["codigo_curso", "ano_censo"],
        ascending=[True, True],
        kind="mergesort",
    )

    return df_valid.drop_duplicates(subset=["codigo_curso"], keep="last").reset_index(drop=True)
```

A ideia é: quando um curso aparece em vários anos no cadastro, escolher o último registro válido como referência.

## 14.5 `cruzamento_cine.py`: cruzar FIES com CINE/INEP

A parte CINE é importante para agrupar cursos por área e subárea. Como nomes de curso podem variar, o código normaliza textos e usa mapas manuais quando necessário.

Trecho real de normalização:
```python
def normalizar_nome(valor) -> str | None:
    if pd.isna(valor):
        return None

    texto = str(valor).strip().upper()
    texto = texto.replace("\xa0", " ")
    texto = re.sub(r"\s+", " ", texto).strip()

    if texto in {"", "NAN", "NONE", "NULL", "NA", "N/A", "-", "--"}:
        return None

    return texto
```

Isso deixa nomes mais comparáveis: tira espaços repetidos, padroniza maiúsculas etc.

Trecho real do enriquecimento:
```python
def enriquecer_com_cine(
    arquivo_entrada: Path,
    arquivo_saida: Path,
    ano_col: str,
    nome_base: str,
    inep: pd.DataFrame,
    mapa_manual: dict,
) -> dict:
    log("=" * 80)
    log(f"CRUZAMENTO CINE: {nome_base}")
    log("=" * 80)

    df = preparar_base_fies(arquivo_entrada, ano_col)

    log(f"[INÍCIO] {arquivo_entrada.name} | linhas: {len(df)} | colunas: {len(df.columns)}")

    anos_fies = (
        df[ano_col]
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )

    log(f"[INFO] {nome_base}: anos encontrados para cruzamento: {sorted(anos_fies)}")

    referencia = construir_referencia_cine(inep, anos_fies)

    log(
        f"[INFO] {nome_base}: referência CINE construída | "
        f"linhas: {len(referencia)} | colunas: {len(referencia.columns)}"
    )

    out = df.merge(
        referencia,
        how="left",
        left_on=["codigo_curso", ano_col],
        right_on=["codigo_curso", "ano_fies_ref"],
        validate="m:1",
    )

    if "ano_fies_ref" in out.columns:
        out = out.drop(columns=["ano_fies_ref"])

    salvar_auxiliar_cursos_validos(out, nome_base)
    diagnosticar_sem_cine(out, nome_base, "antes_manual")

    out, resumo_manual = aplicar_mapa_manual(out, mapa_manual, nome_base)

    diagnosticar_sem_cine(out, nome_base, "depois_manual")

    out.to_parquet(arquivo_saida, index=False)

    log(f"[OK] Salvo em: {arquivo_saida}")
    log(f"[OK] Colunas finais: {len(out.columns)}")

    return auditar_resultado(out, nome_base, resumo_manual)
```

Teoria de `merge`: é como um PROCV/VLOOKUP ou JOIN de SQL. A base do FIES é cruzada com uma base de referência de cursos para trazer área CINE, subárea etc.

## 14.6 `modalidade.py`: classificar/diagnosticar modalidade

Trecho real:
```python
def classificar_modalidade_por_renda(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    validar_colunas(df, [COL_ANO, COL_RENDA, COL_UF_RESIDENCIA], "inscrições")

    df[COL_ANO] = converter_inteiro(df[COL_ANO])
    df[COL_RENDA] = converter_numero(df[COL_RENDA])
    df[COL_UF_RESIDENCIA] = df[COL_UF_RESIDENCIA].map(normalizar_texto).astype("string")

    df["salario_minimo_ano"] = df[COL_ANO].map(SALARIO_MINIMO_ANO)
    df["limite_3sm_ano"] = df["salario_minimo_ano"] * 3
    df["limite_5sm_ano"] = df["salario_minimo_ano"] * 5
    df[COL_REGIAO_RESIDENCIA] = df[COL_UF_RESIDENCIA].map(MAPA_UF_REGIAO).astype("string")

    mask_ano_sem_regra = df["salario_minimo_ano"].isna()
    mask_renda_nula = df[COL_RENDA].isna()
    mask_regiao_nula = df[COL_REGIAO_RESIDENCIA].isna()

    mask_mod_i = (
        df[COL_RENDA].notna()
        & df["limite_3sm_ano"].notna()
        & (df[COL_RENDA] <= df["limite_3sm_ano"])
    )

    mask_mod_ii = (
        df[COL_RENDA].notna()
        & df["limite_3sm_ano"].notna()
        & df["limite_5sm_ano"].notna()
        & (df[COL_RENDA] > df["limite_3sm_ano"])
        & (df[COL_RENDA] <= df["limite_5sm_ano"])
        & df[COL_REGIAO_RESIDENCIA].isin(["Norte", "Nordeste", "Centro-Oeste"])
    )

    mask_mod_iii = (
        df[COL_RENDA].notna()
        & df["limite_3sm_ano"].notna()
        & df["limite_5sm_ano"].notna()
        & (df[COL_RENDA] > df["limite_3sm_ano"])
        & (df[COL_RENDA] <= df["limite_5sm_ano"])
        & df[COL_REGIAO_RESIDENCIA].isin(["Sul", "Sudeste"])
    )

    df[COL_MODALIDADE] = "eliminado"

    df.loc[mask_mod_i, COL_MODALIDADE] = "Modalidade I"
    df.loc[mask_mod_ii, COL_MODALIDADE] = "Modalidade II"
    df.loc[mask_mod_iii, COL_MODALIDADE] = "Modalidade III (P-FIES)"

    resumo = {
        "etapa": "peneira_renda_regiao",
        "linhas": len(df),
        "modalidade_i": int(mask_mod_i.sum()),
        "modalidade_ii": int(mask_mod_ii.sum()),
        "modalidade_iii_pre_pfies": int(mask_mod_iii.sum()),
        "eliminado_pre_pfies": int((df[COL_MODALIDADE] == "eliminado").sum()),
        "ano_sem_regra_salario_minimo": int(mask_ano_sem_regra.sum()),
        "renda_nula": int(mask_renda_nula.sum()),
        "uf_residencia_sem_regiao": int(mask_regiao_nula.sum()),
    }

    return df, resumo
```

A função usa renda per capita e salário mínimo para classificar modalidade. No artigo de políticas públicas, isso entra como foco analítico em baixa renda/Modalidade I, com cautela. No código robusto, modalidade também funciona como diagnóstico e contexto.

## 14.7 `curate.py`: criar a base curada final

A curadoria cria variáveis que viram base do artigo: renda, nota, nota de corte, gap, flags e targets auxiliares.

Trecho real:
```python
def preparar_inscricoes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    validar_colunas(
        df,
        [
            "ano_processo_seletivo",
            "semestre_processo_seletivo",
            "modalidade_fies",
            "situacao_inscricao_fies",
            "renda_mensal_bruta_per_capita",
            "media_nota_enem",
            "nota_corte_grupo_preferencia",
        ],
        "inscrições",
    )

    df["ano"] = to_int_nullable(df["ano_processo_seletivo"])
    df["semestre"] = to_int_nullable(df["semestre_processo_seletivo"])

    df["renda_familiar_per_capita"] = to_numeric_nullable(df["renda_mensal_bruta_per_capita"])
    df["nota_enem"] = to_numeric_nullable(df["media_nota_enem"])
    df["nota_corte"] = to_numeric_nullable(df["nota_corte_grupo_preferencia"])
    df["gap_nota_corte"] = df["nota_enem"] - df["nota_corte"]

    df["situacao_inscricao_fies"] = df["situacao_inscricao_fies"].map(normalizar_texto).astype("string")
    df["modalidade_fies"] = df["modalidade_fies"].astype("string")

    df = criar_aliases_inscricoes(df)
    df = adicionar_regioes_inscricoes(df)

    df["elegivel_academicamente"] = df["gap_nota_corte"].ge(0)
    df.loc[df["gap_nota_corte"].isna(), "elegivel_academicamente"] = pd.NA

    df["contratada"] = df["situacao_inscricao_fies"].eq(SITUACAO_CONTRATADA)
    df["nao_contratado"] = df["situacao_inscricao_fies"].eq(SITUACAO_NAO_CONTRATADO)
    df["lista_espera"] = df["situacao_inscricao_fies"].eq(SITUACAO_LISTA_ESPERA)

    df["target_binario_contratacao"] = pd.NA
    df.loc[df["nao_contratado"], "target_binario_contratacao"] = 0
    df.loc[df["contratada"], "target_binario_contratacao"] = 1
    df["target_binario_contratacao"] = df["target_binario_contratacao"].astype("Int64")

    df["target_ternario_fluxo"] = pd.NA
    df.loc[df["lista_espera"], "target_ternario_fluxo"] = 0
    df.loc[df["nao_contratado"], "target_ternario_fluxo"] = 1
    df.loc[df["contratada"], "target_ternario_fluxo"] = 2
    df["target_ternario_fluxo"] = df["target_ternario_fluxo"].astype("Int64")

    if "opcoes_cursos_inscricao" in df.columns:
        df["inscricao_priorizada"] = to_int_nullable(df["opcoes_cursos_inscricao"]).eq(1)
    elif "opcao_curso" in df.columns:
        df["inscricao_priorizada"] = to_int_nullable(df["opcao_curso"]).eq(1)
    else:
        df["inscricao_priorizada"] = pd.NA

    if "nome_curso" in df.columns:
        curso_norm = df["nome_curso"].map(normalizar_texto).astype("string")
        df["curso_medicina"] = curso_norm.eq("MEDICINA")
    else:
        df["curso_medicina"] = pd.NA

    if "nome_cine_area_geral" in df.columns:
        area_norm = df["nome_cine_area_geral"].map(normalizar_texto).astype("string")
        df["area_saude_bem_estar"] = area_norm.eq("SAÚDE E BEM-ESTAR")
    else:
        df["area_saude_bem_estar"] = pd.NA

    # Recorte empírico do artigo: todos os registros de 2019 a 2021.
    # A classificação de modalidade permanece na base como variável descritiva,
    # mas não restringe o recorte final.
    df["recorte_artigo"] = df["ano"].isin(ARTICLE_YEARS)

    return df
```

O trecho principal cria:

```text
gap_nota_corte = nota_enem - nota_corte
```

Isso é o desempenho relativo à nota de corte.

Também cria flags:

```text
contratada
nao_contratado
lista_espera
elegivel_academicamente
```

E targets auxiliares:

```text
target_binario_contratacao
target_ternario_fluxo
```

Isso é importante porque o artigo inteiro depende da diferença entre:

- barreira acadêmica inicial: lista de espera;
- etapa contratual: não contratado vs contratada.

---

# 15. ABT: a tabela que vai para o modelo

ABT significa `Analytical Base Table`. É a tabela final para modelagem.

Cada linha = uma inscrição.

Cada coluna = uma variável que pode entrar no modelo.

## 15.1 ABT binária

Trecho real:
```python
def construir_abt(recorte: str = "geral") -> tuple[pd.DataFrame, dict]:
    df_base = carregar_base()

    if recorte == "medicina":
        df_base = aplicar_recorte_medicina(df_base)
    elif recorte != "geral":
        raise ValueError("recorte deve ser 'geral' ou 'medicina'.")

    df, mapa_colunas = construir_colunas_canonicas(df_base)

    df = df[df["situacao_fies"].isin(TARGET_MAP.keys())].copy()
    df["target_binario"] = df["situacao_fies"].map(TARGET_MAP).astype(int)

    linhas_apos_target_antes_drop = len(df)

    obrigatorias = [
        "target_binario",
        "renda_per_capita",
        "gap",
        "renda_gap",
        "nota_corte_gp",
    ]

    df = df.dropna(subset=obrigatorias).copy()

    for coluna in NUMERICAS:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df, resumo_categorias = limitar_categorias(df)

    abt = df[COLUNAS_CANONICAS].reset_index(drop=True).copy()
    abt = abt.loc[:, ~abt.columns.duplicated()].copy()

    target_dist = (
        abt["target_binario"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    metadata = {
        "target": "binario",
        "target_col": "target_binario",
        "target_map": {
            "0": STATUS_NAO_CONTRATADO,
            "1": STATUS_CONTRATADA,
        },
        "recorte": recorte,
        "path_base": str(CURATED_INSCRICOES_ARTIGO_PATH),
        "linhas_apos_target_antes_drop": int(linhas_apos_target_antes_drop),
        "linhas_abt": int(len(abt)),
        "colunas_abt": int(len(abt.columns)),
        "distribuicao_target": {str(k): int(v) for k, v in target_dist.items()},
        "mapa_colunas_origem": mapa_colunas,
        "resumo_categorias": resumo_categorias,
        "colunas_numericas": [c for c in NUMERICAS if c in abt.columns],
        "colunas_categoricas": [c for c in CATEGORICAS if c in abt.columns],
        "variaveis_principais_obrigatorias": [
            "renda_per_capita",
            "gap",
            "renda_gap",
        ],
        "observacao": "ABT binária para regressão logística. Não inclui variáveis pós-contratação, como percentual de financiamento.",
    }

    return abt, metadata
```

O que a função faz:

1. carrega a base curada;
2. aplica recorte geral ou Medicina;
3. cria colunas canônicas;
4. mantém só `NÃO CONTRATADO` e `CONTRATADA`;
5. cria `target_binario`;
6. remove linhas sem variáveis obrigatórias;
7. limita categorias raras;
8. salva metadados.

A ABT binária representa a etapa contratual:

```text
0 = não contratado
1 = contratada
```

## 15.2 ABT ternária

Trecho real:
```python
def construir_abt(recorte: str = "geral") -> tuple[pd.DataFrame, dict]:
    if recorte not in {"geral", "medicina"}:
        raise ValueError("recorte deve ser 'geral' ou 'medicina'.")

    df_base = carregar_base()
    df_base = aplicar_recorte(df_base, recorte)

    df, mapa_colunas = construir_colunas_canonicas(df_base)

    df = df[df["situacao_fies"].isin(TARGET_MAP.keys())].copy()
    df["target_ternario"] = df["situacao_fies"].map(TARGET_MAP).astype(int)

    linhas_apos_target_antes_drop = len(df)

    obrigatorias = [
        "target_ternario",
        "renda_per_capita",
        "gap",
        "renda_gap",
        "nota_corte_gp",
    ]

    df = df.dropna(subset=obrigatorias).copy()

    for coluna in NUMERICAS:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df, resumo_categorias = limitar_categorias(df)

    abt = df[COLUNAS_CANONICAS].reset_index(drop=True).copy()
    abt = abt.loc[:, ~abt.columns.duplicated()].copy()

    target_dist = (
        abt["target_ternario"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    metadata = {
        "target": "ternario",
        "target_col": "target_ternario",
        "target_map": {
            "0": STATUS_LISTA_ESPERA,
            "1": STATUS_NAO_CONTRATADO,
            "2": STATUS_CONTRATADA,
        },
        "recorte": recorte,
        "path_base": str(CURATED_INSCRICOES_ARTIGO_PATH),
        "linhas_apos_target_antes_drop": int(linhas_apos_target_antes_drop),
        "linhas_abt": int(len(abt)),
        "colunas_abt": int(len(abt.columns)),
        "distribuicao_target": {str(k): int(v) for k, v in target_dist.items()},
        "mapa_colunas_origem": mapa_colunas,
        "resumo_categorias": resumo_categorias,
        "colunas_numericas": [c for c in NUMERICAS if c in abt.columns],
        "colunas_categoricas": [c for c in CATEGORICAS if c in abt.columns],
        "variaveis_principais_obrigatorias": [
            "renda_per_capita",
            "gap",
            "renda_gap",
        ],
        "observacao": "ABT ternária para regressão logística multinomial. Não inclui variáveis pós-contratação.",
    }

    return abt, metadata
```

A ABT ternária representa o fluxo mais amplo:

```text
0 = lista de espera
1 = não contratado
2 = contratada
```

Por isso a ternária é mais parecida com a história completa do FIES: ela tenta diferenciar quem nem avançou da chamada regular, quem avançou mas não contratou, e quem contratou.

---

# 16. Modelagem logística no robusto

## 16.1 Blocos E1–E5

Os arquivos de regressão organizam as variáveis em experimentos progressivos. A lógica é:

- E1: renda, gap e interação renda × gap;
- E2: adiciona variáveis acadêmicas/curso básicas;
- E3: adiciona ano, semestre, turno, conceito etc.;
- E4: adiciona variáveis institucionais/demográficas;
- E5: usa o conjunto mais completo.

Isso é melhor do que jogar tudo de uma vez sem controle, porque permite ver se o desempenho muda conforme entram blocos de variáveis.

## 16.2 Preprocessamento da regressão

Trecho real:
```python
def criar_preprocessador(numericas: list[str], categoricas: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Não informado")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    min_frequency=20,
                ),
            ),
        ]
    )

    transformers = []

    if numericas:
        transformers.append(("num", numeric_transformer, numericas))

    if categoricas:
        transformers.append(("cat", categorical_transformer, categoricas))

    if not transformers:
        raise ValueError("Nenhuma variável válida foi informada ao preprocessador.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
```

Aqui está o coração técnico da diferença robusto vs legado:

- antes: `pd.get_dummies` manual;
- agora: `OneHotEncoder` dentro de `ColumnTransformer`;
- antes: scaler podia pegar tudo;
- agora: scaler pega só numéricas.

## 16.3 Modelo logístico

Trecho real:
```python
def criar_modelo() -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        C=0.1,
        class_weight="balanced",
        max_iter=5000,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )
```

O `class_weight="balanced"` é central porque os contratos são minoria. Sem isso, o modelo poderia acertar muitos não contratados e ignorar contratados.

## 16.4 Fluxo de treino de um experimento

Trecho real do ajuste:
```python
def ajustar_um_experimento(
    df: pd.DataFrame,
    metadata_abt: dict,
    recorte: str,
    avaliacao: str,
    experimento: dict,
    force: bool,
) -> tuple[dict, pd.DataFrame]:
    blocos_usados, blocos_ausentes = blocos_validos_para_experimento(df, experimento)

    for bloco in CORE_BLOCKS:
        if bloco not in blocos_usados:
            raise ValueError(f"O bloco obrigatório '{bloco}' está ausente ou inválido na ABT.")

    numericas, categoricas = separar_tipos(blocos_usados)

    X, y = preparar_xy(df, blocos_usados)

    if y.nunique() != 2:
        raise ValueError(f"O target precisa ter duas classes. Distribuição encontrada: {y.value_counts().to_dict()}")

    train_idx, test_idx, avaliacao_descricao = criar_split(X, y, avaliacao)

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    model_dir = models_dir(recorte, avaliacao)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"logit_binario_{experimento['id'].lower()}.joblib"
    model_meta_path = model_dir / f"logit_binario_{experimento['id'].lower()}_metadata.json"

    if model_path.exists() and not force:
        modelo = joblib.load(model_path)
        log(f"[INFO] Modelo existente reutilizado: {model_path}")
    else:
        modelo = criar_pipeline(numericas=numericas, categoricas=categoricas)

        log(
            f"[INÍCIO] {recorte} | {avaliacao} | {experimento['id']} | "
            f"n_treino={len(X_train)} | n_teste={len(X_test)} | blocos={len(blocos_usados)}"
        )

        modelo.fit(X_train, y_train)

        joblib.dump(modelo, model_path)

        log(f"[OK] Modelo salvo em: {model_path}")

    proba_test = modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, proba_test)
    coeficientes = extrair_coeficientes(modelo)
    feature_names = obter_feature_names(modelo)

    coef_renda = coeficiente_exato(coeficientes, "renda_per_capita")
    coef_gap = coeficiente_exato(coeficientes, "gap")
    coef_interacao = coeficiente_exato(coeficientes, "renda_gap")

    registro = {
        "recorte": recorte,
        "avaliacao": avaliacao,
        "avaliacao_descricao": avaliacao_descricao,
        "modelo": "logit_binario",
        "modelo_descricao": "Regressão logística binária",
        "experimento": experimento["id"],
        "experimento_nome": experimento["name"],
        "experimento_descricao": experimento["description"],
        "n_total_abt": int(len(df)),
        "n_treino": int(len(X_train)),
        "n_teste": int(len(X_test)),
        "target_0": "Não contratado",
        "target_1": "Contratada",
        "roc_auc": metricas["roc_auc"],
        "accuracy": metricas["accuracy"],
        "balanced_accuracy": metricas["balanced_accuracy"],
        "precision": metricas["precision"],
        "recall": metricas["recall"],
        "f1": metricas["f1"],
        "log_loss": metricas["log_loss"],
        "tn": metricas["tn"],
        "fp": metricas["fp"],
        "fn": metricas["fn"],
        "tp": metricas["tp"],
        "coef_renda_per_capita": coef_renda,
        "coef_gap": coef_gap,
        "coef_renda_gap": coef_interacao,
        "blocos_variaveis_qtd": int(len(blocos_usados)),
        "colunas_finais_pos_processamento": int(len(feature_names)),
        "blocos_incluidos": labels_blocos(blocos_usados),
        "blocos_ausentes": labels_blocos(blocos_ausentes),
        "variaveis_numericas": "; ".join(numericas),
        "variaveis_categoricas": "; ".join(categoricas),
        "model_path": str(model_path),
    }

    model_metadata = {
        "registro": registro,
        "experimento": experimento,
        "blocos_usados": blocos_usados,
        "blocos_ausentes": blocos_ausentes,
        "numericas": numericas,
        "categoricas": categoricas,
        "feature_names_out": feature_names,
        "abt_metadata": metadata_abt,
        "observacao": "Modelo logístico binário. Métricas calculadas no conjunto indicado por avaliacao_descricao.",
    }

    with model_meta_path.open("w", encoding="utf-8") as file:
        json.dump(model_metadata, file, ensure_ascii=False, ind
    # ... trecho cortado no guia; função continua no arquivo original
```

Essa função é uma miniesteira:

1. escolhe variáveis válidas;
2. separa numéricas e categóricas;
3. prepara X e y;
4. divide treino/teste;
5. cria pipeline;
6. treina com `.fit`;
7. calcula probabilidades com `.predict_proba`;
8. calcula métricas;
9. salva coeficientes, métricas, modelo e metadados.

## 16.5 Logística multinomial

No ternário, a ideia é a mesma, mas a classe tem três possibilidades. O modelo retorna três probabilidades por linha:

```text
P(lista de espera), P(não contratado), P(contratada)
```

Trecho de métricas multiclasse:
```python
def calcular_metricas(y_true, proba, classes) -> dict:
    y_pred = predicoes_por_proba(proba, classes)
    matriz = confusion_matrix(y_true, y_pred, labels=CLASSES_TERNARIAS)

    metricas = {
        "roc_auc_ovr_weighted": calcular_roc_auc(y_true, proba, classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=list(classes))),
    }

    for i, classe_real in enumerate(CLASSES_TERNARIAS):
        for j, classe_pred in enumerate(CLASSES_TERNARIAS):
            metricas[f"confusao_real_{classe_real}_pred_{classe_pred}"] = int(matriz[i, j])

    return metricas
```

O `average="macro"` quer dizer: calcula a métrica por classe e tira média simples. Cada classe vale o mesmo na média. Isso é importante porque contratada é menor.

---

# 17. Árvores de decisão no robusto

## 17.1 Por que árvore?

A regressão logística é linear na escala do logito. Ela é boa como referência, mas pode ter dificuldade em capturar regras do tipo:

```text
se gap está muito abaixo da nota de corte, lista de espera cresce muito;
se gap é suficiente mas renda e curso indicam custo alto, contratação pode variar;
```

Árvores capturam isso com cortes sucessivos.

## 17.2 Preprocessamento da árvore

Trecho real:
```python
def criar_preprocessador(numericas: list[str], categoricas: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Não informado")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    min_frequency=20,
                ),
            ),
        ]
    )

    transformers = []
    if numericas:
        transformers.append(("num", numeric_transformer, numericas))
    if categoricas:
        transformers.append(("cat", categorical_transformer, categoricas))
    if not transformers:
        raise ValueError("Nenhuma variável válida foi informada ao preprocessador.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
```

Repare: aqui não tem `StandardScaler`. A árvore não precisa.

## 17.3 Pré-poda dinâmica

Trecho real:
```python
def min_leaf_dinamico(n_train: int, recorte: str, profundidade: int | str) -> int:
    profundidade = validar_profundidade(profundidade)
    profile = TREE_DEPTH_PROFILES[profundidade]

    # Pré-poda: nunca permite folha com menos de 400 registros.
    # Em bases muito grandes, aumenta discretamente esse piso para evitar folhas muito pequenas.
    return int(max(profile["min_samples_leaf_min"], round(n_train * 0.0005)))
```
```python
def min_split_dinamico(n_train: int, recorte: str, profundidade: int | str) -> int:
    profundidade = validar_profundidade(profundidade)
    leaf = min_leaf_dinamico(n_train, recorte, profundidade)
    profile = TREE_DEPTH_PROFILES[profundidade]

    # Para dividir um nó, exige pelo menos o dobro da folha mínima e nunca menos de 800.
    return int(max(profile["min_samples_split_min"], 2 * leaf))
```

Isso controla overfitting. Uma árvore muito livre poderia criar folhas com poucos registros e decorar ruído. A pré-poda exige um mínimo de registros por folha e por divisão.

## 17.4 Modelo árvore

Trecho real:
```python
def criar_modelo(n_train: int, recorte: str, profundidade: int | str) -> DecisionTreeClassifier:
    profundidade = validar_profundidade(profundidade)
    leaf = min_leaf_dinamico(n_train, recorte, profundidade)
    split = min_split_dinamico(n_train, recorte, profundidade)

    return DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=profundidade,
        min_samples_leaf=leaf,
        min_samples_split=split,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
```

## 17.5 Importância das variáveis

A árvore tem `feature_importances_`: mede quanto cada coluna ajudou a reduzir impureza. No robusto, como one-hot gera muitas colunas, o código agrega de volta por variável original.

Trecho real:
```python
def agregar_importancias_por_variavel(importancias: pd.DataFrame) -> pd.DataFrame:
    out = (
        importancias
        .groupby(["variavel_original", "bloco_label"], as_index=False)["importancia"]
        .sum()
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    total = float(out["importancia"].sum())
    out["importancia_normalizada"] = out["importancia"] / total if total > 0 else 0.0
    validar_importancias_normalizadas(
        out,
        coluna="importancia_normalizada",
        contexto="importâncias agregadas por variável original",
    )
    return out
```

Exemplo:

```text
cat__turno_NOTURNO
cat__turno_MATUTINO
cat__turno_INTEGRAL
```

viram uma importância agregada de:

```text
turno
```

Isso torna a interpretação mais limpa para o artigo.

---

# 18. Probabilidades previstas para o artigo de políticas públicas

O arquivo `efeitos_multinomiais_ternario.py` é importante porque transforma modelo em leitura substantiva.

Ele cria uma grade de perfis:

```text
faixa de renda × faixa de desempenho relativo à nota de corte
```

Trecho real:
```python
def construir_grid(abt: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "renda_per_capita" not in abt.columns or "gap" not in abt.columns:
        raise KeyError("A ABT precisa conter as colunas renda_per_capita e gap.")

    base: dict[str, object] = {}
    for col in features:
        if col in abt.columns:
            base[col] = valor_tipico_coluna(abt[col])

    linhas = []
    for codigo_renda, label_renda, min_r, max_r, fallback_r in FAIXAS_RENDA:
        renda_valor = mediana_faixa(abt, "renda_per_capita", min_r, max_r, fallback_r)

        for codigo_gap, label_gap, min_g, max_g, fallback_g in FAIXAS_GAP:
            gap_valor = mediana_faixa(abt, "gap", min_g, max_g, fallback_g)

            row = dict(base)
            row["renda_per_capita"] = renda_valor
            row["gap"] = gap_valor
            if "renda_gap" in features:
                row["renda_gap"] = renda_valor * gap_valor

            row["_codigo_renda"] = codigo_renda
            row["_faixa_renda"] = label_renda
            row["_codigo_desempenho"] = codigo_gap
            row["_faixa_desempenho"] = label_gap
            linhas.append(row)

    grid = pd.DataFrame(linhas)

    for col in features:
        if col not in grid.columns:
            grid[col] = np.nan

    return grid, grid[features].copy()
```

O que isso faz:

- escolhe faixas de renda;
- escolhe faixas de gap;
- monta uma linha artificial/típica para cada combinação;
- mantém outras variáveis em valores típicos;
- calcula `renda_gap = renda * gap`;
- passa essa grade no modelo para obter probabilidades previstas.

Essa é a ponte entre Computação e Educação/política pública: o modelo vira uma forma de visualizar como renda e desempenho diferenciam lista de espera, não contratação e contratação.

---

# 19. Análise renda × percentual financiado

Esse arquivo não treina modelo de classificação. Ele analisa associação entre renda e percentual financiado entre os contratos efetivados.

Trecho real:
```python
def calcular_estatisticas_e_regressao(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        raise ValueError("A regressão exige ao menos 3 observações válidas.")

    x = df["renda_per_capita"].to_numpy(dtype="float64").reshape(-1)
    y = df["percentual_financiamento"].to_numpy(dtype="float64").reshape(-1)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("As variáveis da regressão precisam ser vetores unidimensionais.")

    if len(x) != len(y):
        raise ValueError("As variáveis da regressão precisam ter o mesmo tamanho.")

    pearson_r, pearson_p_value = pearsonr(x, y)
    regressao = linregress(x, y)

    beta = float(np.asarray(regressao.slope).reshape(-1)[0])
    intercepto = float(np.asarray(regressao.intercept).reshape(-1)[0])
    r2 = float(np.asarray(regressao.rvalue).reshape(-1)[0] ** 2)
    efeito_100_reais = beta * 100

    resultados = {
        "n": int(len(df)),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p_value),
        "intercepto": intercepto,
        "beta_renda": beta,
        "efeito_100_reais": efeito_100_reais,
        "r2": r2,
        "regressao_p_value": float(np.asarray(regressao.pvalue).reshape(-1)[0]),
        "erro_padrao_beta": float(np.asarray(regressao.stderr).reshape(-1)[0]),
        "erro_padrao_intercepto": float(np.asarray(regressao.intercept_stderr).reshape(-1)[0]),
    }

    log("[OK] Estatísticas calculadas")
    log(f"     n = {resultados['n']}")
    log(f"     Pearson r = {resultados['pearson_r']:.4f}")
    log(f"     beta = {resultados['beta_renda']:.4f}")
    log(f"     intercepto = {resultados['intercepto']:.2f}")
    log(f"     R² = {resultados['r2']:.4f}")
    log(f"     efeito a cada R$ 100 = {resultados['efeito_100_reais']:.2f} p.p.")

    return resultados
```

- `pearsonr(x, y)`: correlação linear entre renda e percentual financiado.
- `linregress(x, y)`: regressão linear simples.
- `beta`: mudança esperada no percentual financiado quando renda aumenta 1 real.
- `efeito_100_reais`: mudança esperada a cada R$ 100.
- `R²`: quanto da variação do percentual financiado é explicada por renda nessa regressão simples.

Cuidado: isso é associação, não causalidade.

---

# 20. Produtos de artigo

Os arquivos em `src/article/` não são o núcleo do treinamento. Eles transformam resultados em tabelas e figuras.

Exemplos:

- `fluxo_selecao.py`: figura do funil.
- `taxas_conversao.py`: taxa de conversão.
- `tabelas_distribuicao.py`: Tabela 1 e B1.
- `matrizes_renda_desempenho.py`: heatmaps renda × desempenho.
- `logit_binario.py` e `logit_ternario.py`: tabelas/figuras dos logits.
- `treeClassification_profundidade.py`: produtos das árvores.
- `apendice_*`: tabelas completas, matrizes de confusão e figuras suplementares.
- `pacote_artigo.py`: copia tudo para `article/`.

---

# 21. Arquivo por arquivo — o que cada `.py` faz

Abaixo está uma leitura por arquivo. Para os arquivos de algoritmo, há explicações maiores acima. Aqui a ideia é você saber localizar cada responsabilidade.


## src/__init__.py

**Papel:** Arquivo vazio que faz a pasta src ser reconhecida como pacote Python.

Não há funções definidas; arquivo vazio ou apenas constantes/importações.


## src/abt/__init__.py

**Papel:** Arquivo vazio de pacote Python.

Não há funções definidas; arquivo vazio ou apenas constantes/importações.


## src/abt/build_abt_binaria.py

**Papel:** Cria a ABT binária: NÃO CONTRATADO = 0 e CONTRATADA = 1.

**Funções principais detectadas:**

- `log`
- `remover_acentos`
- `normalizar_texto`
- `normalizar_status`
- `encontrar_coluna`
- `converter_numero_serie`
- `extrair_ano_nascimento`
- `carregar_base`
- `aplicar_recorte_medicina`
- `construir_colunas_canonicas`
- `limitar_categorias`
- `caminho_saida`
- `construir_abt`
- `salvar_abt`
- `run`
- `parse_args`


## src/abt/build_abt_ternaria.py

**Papel:** Cria a ABT ternária: LISTA DE ESPERA = 0, NÃO CONTRATADO = 1, CONTRATADA = 2.

**Funções principais detectadas:**

- `log`
- `remover_acentos`
- `normalizar_texto`
- `normalizar_status`
- `encontrar_coluna`
- `converter_numero_serie`
- `extrair_ano_nascimento`
- `carregar_base`
- `aplicar_recorte_medicina`
- `aplicar_recorte`
- `construir_colunas_canonicas`
- `limitar_categorias`
- `construir_abt`
- `path_saida_abt`
- `salvar_abt`
- `run`
- `parse_args`


## src/analysis/__init__.py

**Papel:** Arquivo vazio de pacote Python.

Não há funções definidas; arquivo vazio ou apenas constantes/importações.


## src/analysis/dataset_candidatos_unicos.py

**Papel:** Gera base de candidatos únicos por prioridade, para análise suplementar em nível de candidato.

**Funções principais detectadas:**

- `log`
- `validar_colunas`
- `normalizar_texto`
- `preparar_base`
- `gerar_candidatos_unicos`
- `gerar_agregado`
- `salvar_resumo`
- `run`


## src/analysis/dataset_funil_fluxo.py

**Papel:** Gera o funil de vagas, inscrições, nota suficiente e contratos efetivados.

**Funções principais detectadas:**

- `log`
- `validar_colunas`
- `normalizar_categorias`
- `preparar_inscricoes`
- `preparar_ofertas`
- `soma_ofertas`
- `conta_inscricoes`
- `pivot_situacoes`
- `gerar_funil`
- `salvar_resumo`
- `run`


## src/analysis/dataset_taxas.py

**Papel:** Calcula taxas derivadas do funil, como conversão, ocupação e proporções.

**Funções principais detectadas:**

- `log`
- `divisao_segura`
- `calcular_taxas`
- `validar_colunas`
- `gerar_taxas`
- `salvar_resumo`
- `run`


## src/article/__init__.py

**Papel:** Arquivo vazio de pacote Python.

Não há funções definidas; arquivo vazio ou apenas constantes/importações.


## src/article/apendice_logit_binario.py

**Papel:** Gera apêndices completos da regressão logística binária.

**Funções principais detectadas:**

- `obter_fonte_padrao`
- `configurar_matplotlib`
- `fmt_int`
- `fmt_float`
- `quebrar_texto`
- `salvar_figura`
- `salvar_tabela_latex`
- `salvar_tabela_como_imagem`
- `preparar_base_probabilidades`
- `matriz_probabilidade_prevista`
- `tabela_media_prob_por_faixas`
- `gerar_heatmap_probabilidade`
- `gerar_curvas_desempenho_por_renda`
- `gerar_curvas_renda_por_desempenho`
- `renderizar_matriz_confusao`
- `log`
- `carregar_metricas`
- `preparar_tabela_completa`
- `preparar_tabela_compacta`
- `renderizar_tabela_completa`
- `gerar_tabela_completa`
- `gerar_tabela_compacta`
- `gerar_matriz_confusao`
- `run`
- `parse_args`

Este arquivo é de produto/apêndice: lê resultados já gerados, formata tabelas, desenha matriz de confusão e salva figuras. Ele não muda a lógica do treinamento.


## src/article/apendice_logit_ternario.py

**Papel:** Gera apêndices completos da regressão logística ternária.

**Funções principais detectadas:**

- `obter_fonte_padrao`
- `configurar_matplotlib`
- `fmt_int`
- `fmt_float`
- `quebrar_texto`
- `salvar_figura`
- `salvar_tabela_latex`
- `salvar_tabela_como_imagem`
- `preparar_base_probabilidades`
- `matriz_probabilidade_prevista`
- `tabela_media_prob_por_faixas`
- `gerar_heatmap_probabilidade`
- `gerar_curvas_desempenho_por_renda`
- `gerar_curvas_renda_por_desempenho`
- `renderizar_matriz_confusao`
- `appendix_logit_dir`
- `log`
- `carregar_metricas`
- `preparar_tabela_completa`
- `preparar_tabela_compacta`
- `renderizar_tabela_completa`
- `gerar_tabela_completa`
- `gerar_tabela_compacta`
- `matriz_confusao_da_linha`
- `gerar_matriz_confusao`
- ... mais 2 funções

Este arquivo é de produto/apêndice: lê resultados já gerados, formata tabelas, desenha matriz de confusão e salva figuras. Ele não muda a lógica do treinamento.


## src/article/apendice_treeClassification.py

**Papel:** Gera apêndices das árvores padrão.

**Funções principais detectadas:**

- `obter_fonte_padrao`
- `configurar_matplotlib`
- `fmt_int`
- `fmt_float`
- `quebrar_texto`
- `salvar_figura`
- `salvar_tabela_latex`
- `salvar_tabela_como_imagem`
- `preparar_base_probabilidades`
- `matriz_probabilidade_prevista`
- `tabela_media_prob_por_faixas`
- `gerar_heatmap_probabilidade`
- `gerar_curvas_desempenho_por_renda`
- `gerar_curvas_renda_por_desempenho`
- `renderizar_matriz_confusao`
- `gerar_barra_importancias`
- `log`
- `carregar_metricas`
- `preparar_tabela_completa`
- `preparar_tabela_compacta`
- `renderizar_tabela_completa`
- `gerar_tabela_completa`
- `gerar_tabela_compacta`
- `matriz_confusao_da_linha`
- `gerar_matriz_confusao`
- ... mais 2 funções

Este arquivo é de produto/apêndice: lê resultados já gerados, formata tabelas, desenha matriz de confusão e salva figuras. Ele não muda a lógica do treinamento.


## src/article/apendice_treeClassification_profundidade.py

**Papel:** Gera apêndices das árvores por profundidade.

**Funções principais detectadas:**

- `obter_fonte_padrao`
- `configurar_matplotlib`
- `fmt_int`
- `fmt_float`
- `quebrar_texto`
- `salvar_figura`
- `salvar_tabela_latex`
- `salvar_tabela_como_imagem`
- `preparar_base_probabilidades`
- `matriz_probabilidade_prevista`
- `tabela_media_prob_por_faixas`
- `gerar_heatmap_probabilidade`
- `gerar_curvas_desempenho_por_renda`
- `gerar_curvas_renda_por_desempenho`
- `renderizar_matriz_confusao`
- `gerar_barra_importancias`
- `set_profundidade`
- `appendix_tree_dir`
- `log`
- `carregar_metricas`
- `preparar_tabela_completa`
- `preparar_tabela_compacta`
- `renderizar_tabela_completa`
- `gerar_tabela_completa`
- `gerar_tabela_compacta`
- ... mais 4 funções

Este arquivo é de produto/apêndice: lê resultados já gerados, formata tabelas, desenha matriz de confusão e salva figuras. Ele não muda a lógica do treinamento.


## src/article/efeitos_multinomiais_ternario.py

**Papel:** Gera probabilidades previstas por grade de renda e desempenho para interpretar o modelo ternário.

**Funções principais detectadas:**

- `configurar_matplotlib`
- `caminho_abt`
- `caminho_abt_binaria`
- `caminho_modelo`
- `caminho_modelo_binario`
- `estimador_final`
- `classes_modelo`
- `feature_names_modelo`
- `valor_tipico_coluna`
- `mediana_faixa`
- `construir_grid`
- `prever_probabilidades`
- `prever_probabilidade_binaria`
- `probabilidade_binaria_geral`
- `ordenar_por_desempenho`
- `ordenar_por_renda`
- `preparar_eixo`
- `plot_painel_por_desempenho`
- `plot_painel_por_renda`
- `salvar_figura`
- `legenda_superior`
- `figura_por_desempenho_todas_classes`
- `figura_por_renda_todas_classes`
- `run`
- `parse_args`


## src/article/financiamento_coparticipacao.py

**Papel:** Analisa, entre contratados, a associação entre renda e percentual financiado.

**Funções principais detectadas:**

- `log`
- `remover_acentos`
- `normalizar_texto`
- `normalizar_status`
- `encontrar_coluna`
- `converter_numero_serie`
- `obter_fonte_padrao`
- `configurar_matplotlib`
- `formatar_inteiro`
- `formatar_decimal`
- `formatar_pp`
- `formatar_p_valor`
- `carregar_base`
- `preparar_dados_financiamento`
- `calcular_estatisticas_e_regressao`
- `montar_tabela_resultados`
- `salvar_tabela_latex`
- `salvar_dados_e_tabelas`
- `plotar_regressao`
- `plotar_tabela_resultados`
- `salvar_resumo`
- `run`


## src/article/fluxo_selecao.py

**Papel:** Desenha os gráficos de fluxo do processo seletivo.

**Funções principais detectadas:**

- `log`
- `configurar_matplotlib`
- `limpar_nome_arquivo`
- `normalizar_area_exibicao`
- `estilos_monocromaticos`
- `carregar_funil`
- `obter_areas`
- `obter_periodos`
- `obter_regioes`
- `salvar_figura`
- `plot_funil_seguro`
- `salvar_resumo`
- `run`


## src/article/logit_binario.py

**Papel:** Carrega regressão logística binária treinada e transforma resultados em tabelas/figuras.

**Funções principais detectadas:**

- `obter_fonte_padrao`
- `configurar_matplotlib`
- `fmt_int`
- `fmt_float`
- `quebrar_texto`
- `salvar_figura`
- `salvar_tabela_latex`
- `salvar_tabela_como_imagem`
- `preparar_base_probabilidades`
- `matriz_probabilidade_prevista`
- `tabela_media_prob_por_faixas`
- `gerar_heatmap_probabilidade`
- `gerar_curvas_desempenho_por_renda`
- `gerar_curvas_renda_por_desempenho`
- `renderizar_matriz_confusao`
- `log`
- `abt_path`
- `carregar_modelo_metadata_abt`
- `carregar_metricas`
- `preparar_X`
- `selecionar_base_para_probabilidades`
- `anexar_probabilidades`
- `montar_tabela_principal`
- `salvar_tabela_principal`
- `salvar_dados_probabilidades`
- ... mais 4 funções


## src/article/logit_ternario.py

**Papel:** Carrega regressão logística ternária treinada e transforma resultados em tabelas/figuras.

**Funções principais detectadas:**

- `obter_fonte_padrao`
- `configurar_matplotlib`
- `fmt_int`
- `fmt_float`
- `quebrar_texto`
- `salvar_figura`
- `salvar_tabela_latex`
- `salvar_tabela_como_imagem`
- `preparar_base_probabilidades`
- `matriz_probabilidade_prevista`
- `tabela_media_prob_por_faixas`
- `gerar_heatmap_probabilidade`
- `gerar_curvas_desempenho_por_renda`
- `gerar_curvas_renda_por_desempenho`
- `renderizar_matriz_confusao`
- `figures_logit_dir`
- `tables_logit_dir`
- `appendix_logit_dir`
- `abt_path`
- `log`
- `carregar_modelo_metadata_abt`
- `carregar_metricas`
- `preparar_X`
- `selecionar_base_para_probabilidades`
- `anexar_probabilidades`
- ... mais 6 funções


## src/article/matrizes_renda_desempenho.py

**Papel:** Gera matrizes/heatmaps de situação por faixa de renda e desempenho relativo à nota de corte.

**Funções principais detectadas:**

- `log`
- `remover_acentos`
- `normalizar_texto`
- `normalizar_status`
- `encontrar_coluna`
- `obter_fonte_padrao`
- `configurar_matplotlib`
- `limpar_nome_arquivo`
- `criar_colormap_cinza`
- `formatar_anotacao`
- `ajustar_cor_textos_heatmap`
- `salvar_figura`
- `carregar_base`
- `preparar_base_matrizes`
- `gerar_decomposicao_por_area`
- `gerar_decomposicao_nacional`
- `auditar_percentuais`
- `matriz_status`
- `plotar_contratados_nao_contratados`
- `plotar_lista_espera`
- `salvar_dados_auxiliares`
- `salvar_resumo`
- `run`


## src/article/pacote_artigo.py

**Papel:** Organiza o pacote final article/, copiando tabelas, figuras, CSVs, TeX e apêndices.

**Funções principais detectadas:**

- `_ensure_dir`
- `_limpar_pasta`
- `_fmt_int_br`
- `_fmt_float_br`
- `_latex_escape`
- `_glob_many`
- `_copy_one`
- `copiar_item`
- `salvar_tabela_cinza`
- `_find_csv_contains`
- `_col`
- `copiar_elementos_fixos`
- `_parse_int_br`
- `_fmt_percentual_br`
- `corrigir_tabela_b1`
- `gerar_efeitos_multinomiais_se_possivel`
- `copiar_efeitos_multinomiais`
- `_carregar_metricas_logit_ternario`
- `gerar_tabela_compacta_logit_ternario`
- `_modelo_path`
- `_estimador_final`
- `_feature_names_pipeline`
- `_match_feature`
- `gerar_tabela_coeficientes_logit_ternario`
- `_diagnostics_tree_dir`
- ... mais 23 funções


## src/article/tabelas_distribuicao.py

**Papel:** Gera Tabela 1 e Tabela B1 de distribuição de situações/faixas de renda.

**Funções principais detectadas:**

- `log`
- `obter_fonte_padrao`
- `configurar_matplotlib`
- `normalizar_texto`
- `encontrar_coluna`
- `carregar_base`
- `aplicar_filtro_modalidade_i_se_existir`
- `preparar_tabela_1`
- `preparar_tabela_b1`
- `formatar_inteiro`
- `formatar_percentual`
- `formatar_tabela_para_exibicao`
- `salvar_csv_tex`
- `renderizar_tabela`
- `gerar_tabela_1`
- `gerar_tabela_b1`
- `salvar_resumo`
- `run`


## src/article/taxas_conversao.py

**Papel:** Desenha gráficos e tabelas de taxa de conversão.

**Funções principais detectadas:**

- `log`
- `configurar_matplotlib`
- `limpar_nome_arquivo`
- `normalizar_area_exibicao`
- `divisao_segura`
- `formatar_percentual`
- `carregar_funil`
- `calcular_taxas`
- `preparar_dados`
- `ordenar_areas`
- `salvar_figura`
- `plotar_taxa`
- `salvar_tabelas`
- `salvar_resumo`
- `run`


## src/article/treeClassification.py

**Papel:** Gera produtos de artigo para árvores padrão.

**Funções principais detectadas:**

- `obter_fonte_padrao`
- `configurar_matplotlib`
- `fmt_int`
- `fmt_float`
- `quebrar_texto`
- `salvar_figura`
- `salvar_tabela_latex`
- `salvar_tabela_como_imagem`
- `preparar_base_probabilidades`
- `matriz_probabilidade_prevista`
- `tabela_media_prob_por_faixas`
- `gerar_heatmap_probabilidade`
- `gerar_curvas_desempenho_por_renda`
- `gerar_curvas_renda_por_desempenho`
- `renderizar_matriz_confusao`
- `gerar_barra_importancias`
- `target_suffix`
- `figures_tree_dir`
- `tables_tree_dir`
- `appendix_tree_dir`
- `log`
- `carregar_modelo_metadata_abt`
- `carregar_metricas`
- `carregar_importancias`
- `preparar_X_artigo`
- ... mais 9 funções


## src/article/treeClassification_profundidade.py

**Papel:** Gera produtos de artigo para árvores por profundidade 10/14/19.

**Funções principais detectadas:**

- `obter_fonte_padrao`
- `configurar_matplotlib`
- `fmt_int`
- `fmt_float`
- `quebrar_texto`
- `salvar_figura`
- `salvar_tabela_latex`
- `salvar_tabela_como_imagem`
- `preparar_base_probabilidades`
- `matriz_probabilidade_prevista`
- `tabela_media_prob_por_faixas`
- `gerar_heatmap_probabilidade`
- `gerar_curvas_desempenho_por_renda`
- `gerar_curvas_renda_por_desempenho`
- `renderizar_matriz_confusao`
- `gerar_barra_importancias`
- `set_profundidade`
- `target_suffix`
- `figures_tree_dir`
- `tables_tree_dir`
- `appendix_tree_dir`
- `log`
- `carregar_modelo_metadata_abt`
- `carregar_metricas`
- `carregar_importancias`
- ... mais 10 funções


## src/config.py

**Papel:** Cria/verifica a estrutura básica do projeto antes de rodar o pipeline.

**Funções principais detectadas:**

- `ensure_project_structure`
- `check_environment`

A função `ensure_project_structure()` usa `mkdir(parents=True, exist_ok=True)`: cria diretórios e não reclama se já existirem.


## src/constants.py

**Papel:** Centraliza todos os caminhos, nomes de arquivos, anos, diretórios de dados, modelos, relatórios e article/.

Não há funções definidas; arquivo vazio ou apenas constantes/importações.

Este arquivo é infraestrutura: ele evita que caminhos como `data/04_curated` fiquem espalhados em vários lugares. Isso torna o pipeline mais fácil de manter.


## src/modeling/__init__.py

**Papel:** Arquivo vazio de pacote Python.

Não há funções definidas; arquivo vazio ou apenas constantes/importações.


## src/modeling/fit_logit_binario_holdout_80_20.py

**Papel:** Wrapper pequeno: chama o utilitário de regressão logística com avaliação fixa in_sample ou holdout_80_20.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.logit_binario_utils import parse_args_modelagem, run_modelagem


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(recorte=recorte, avaliacao="holdout_80_20", force=force)


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/fit_logit_binario_in_sample.py

**Papel:** Wrapper pequeno: chama o utilitário de regressão logística com avaliação fixa in_sample ou holdout_80_20.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.logit_binario_utils import parse_args_modelagem, run_modelagem


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(recorte=recorte, avaliacao="in_sample", force=force)


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/fit_logit_ternario_holdout_80_20.py

**Papel:** Wrapper pequeno: chama o utilitário de regressão logística com avaliação fixa in_sample ou holdout_80_20.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.logit_ternario_utils import parse_args_modelagem, run_modelagem


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(recorte=recorte, avaliacao="holdout_80_20", force=force)


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/fit_logit_ternario_in_sample.py

**Papel:** Wrapper pequeno: chama o utilitário de regressão logística com avaliação fixa in_sample ou holdout_80_20.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.logit_ternario_utils import parse_args_modelagem, run_modelagem


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(recorte=recorte, avaliacao="in_sample", force=force)


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/logit_binario_utils.py

**Papel:** Implementa regressão logística binária com Pipeline, ColumnTransformer, métricas e salvamento.

**Funções principais detectadas:**

- `log`
- `abt_path`
- `metadata_path`
- `models_dir`
- `diagnostics_dir`
- `carregar_abt`
- `limpar_coluna_categorica`
- `limpar_coluna_numerica`
- `bloco_valido`
- `blocos_validos_para_experimento`
- `separar_tipos`
- `labels_blocos`
- `criar_preprocessador`
- `criar_modelo`
- `criar_pipeline`
- `criar_split`
- `preparar_xy`
- `obter_feature_names`
- `calcular_metricas`
- `extrair_coeficientes`
- `coeficiente_exato`
- `ajustar_um_experimento`
- `salvar_resultados`
- `run_modelagem`
- `parse_args_modelagem`

Este é arquivo de algoritmo/modelagem. Leia também as seções 16 e 17, onde os trechos centrais foram explicados em detalhe.


## src/modeling/logit_ternario_utils.py

**Papel:** Implementa regressão logística multinomial para três classes, com métricas macro/weighted.

**Funções principais detectadas:**

- `log`
- `normalizar_recorte`
- `abt_path`
- `metadata_path`
- `models_dir`
- `diagnostics_dir`
- `carregar_abt`
- `limpar_coluna_categorica`
- `limpar_coluna_numerica`
- `bloco_valido`
- `blocos_validos_para_experimento`
- `separar_tipos`
- `labels_blocos`
- `criar_preprocessador`
- `criar_modelo`
- `criar_pipeline`
- `criar_split`
- `preparar_xy`
- `obter_feature_names`
- `predicoes_por_proba`
- `calcular_roc_auc`
- `calcular_metricas`
- `extrair_coeficientes`
- `coeficiente_exato`
- `ajustar_um_experimento`
- ... mais 3 funções

Este é arquivo de algoritmo/modelagem. Leia também as seções 16 e 17, onde os trechos centrais foram explicados em detalhe.


## src/modeling/treeClassification_10_profundidade_binario_holdout_80_20.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 10
TARGET = "binario"
AVALIACAO = "holdout_80_20"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_10_profundidade_binario_in_sample.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 10
TARGET = "binario"
AVALIACAO = "in_sample"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_10_profundidade_ternario_holdout_80_20.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 10
TARGET = "ternario"
AVALIACAO = "holdout_80_20"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_10_profundidade_ternario_in_sample.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 10
TARGET = "ternario"
AVALIACAO = "in_sample"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_14_profundidade_binario_holdout_80_20.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 14
TARGET = "binario"
AVALIACAO = "holdout_80_20"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_14_profundidade_binario_in_sample.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 14
TARGET = "binario"
AVALIACAO = "in_sample"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_14_profundidade_ternario_holdout_80_20.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 14
TARGET = "ternario"
AVALIACAO = "holdout_80_20"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_14_profundidade_ternario_in_sample.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 14
TARGET = "ternario"
AVALIACAO = "in_sample"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_19_profundidade_binario_holdout_80_20.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 19
TARGET = "binario"
AVALIACAO = "holdout_80_20"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_19_profundidade_binario_in_sample.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 19
TARGET = "binario"
AVALIACAO = "in_sample"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_19_profundidade_ternario_holdout_80_20.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 19
TARGET = "ternario"
AVALIACAO = "holdout_80_20"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_19_profundidade_ternario_in_sample.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 19
TARGET = "ternario"
AVALIACAO = "in_sample"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_binario_holdout_80_20.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))
from src.modeling.treeClassification_utils import parse_args_modelagem, run_modelagem


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(target="binario", recorte=recorte, avaliacao="holdout_80_20", force=force)


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_binario_in_sample.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))
from src.modeling.treeClassification_utils import parse_args_modelagem, run_modelagem


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(target="binario", recorte=recorte, avaliacao="in_sample", force=force)


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_profundidade_utils.py

**Papel:** Implementa árvores de decisão com profundidades 10, 14 e 19, pré-poda dinâmica, métricas e importâncias.

**Funções principais detectadas:**

- `validar_profundidade`
- `profile_prefix`
- `model_dir_name_com_profundidade`
- `log`
- `normalizar_recorte`
- `normalizar_target`
- `target_config`
- `abt_path`
- `metadata_path`
- `models_dir`
- `diagnostics_dir`
- `carregar_abt`
- `limpar_coluna_categorica`
- `limpar_coluna_numerica`
- `bloco_valido`
- `blocos_validos_para_experimento`
- `separar_tipos`
- `labels_blocos`
- `criar_preprocessador`
- `min_leaf_dinamico`
- `min_split_dinamico`
- `criar_modelo`
- `criar_pipeline`
- `criar_split`
- `preparar_xy`
- ... mais 16 funções

Este é arquivo de algoritmo/modelagem. Leia também as seções 16 e 17, onde os trechos centrais foram explicados em detalhe.


## src/modeling/treeClassification_ternario_holdout_80_20.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))
from src.modeling.treeClassification_utils import parse_args_modelagem, run_modelagem


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(target="ternario", recorte=recorte, avaliacao="holdout_80_20", force=force)


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_ternario_in_sample.py

**Papel:** Wrapper pequeno: chama o utilitário de árvore com target, avaliação e/ou profundidade fixos.

**Funções principais detectadas:**

- `run`

Como é wrapper pequeno, segue o conteúdo essencial do arquivo:

```python
from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))
from src.modeling.treeClassification_utils import parse_args_modelagem, run_modelagem


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(target="ternario", recorte=recorte, avaliacao="in_sample", force=force)


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
```
Ele só encaminha parâmetros para o utilitário central. Não há algoritmo novo aqui.


## src/modeling/treeClassification_utils.py

**Papel:** Implementa árvores de decisão padrão, com preprocessamento, métricas, importância e regras.

**Funções principais detectadas:**

- `log`
- `normalizar_recorte`
- `normalizar_target`
- `target_config`
- `abt_path`
- `metadata_path`
- `models_dir`
- `diagnostics_dir`
- `carregar_abt`
- `limpar_coluna_categorica`
- `limpar_coluna_numerica`
- `bloco_valido`
- `blocos_validos_para_experimento`
- `separar_tipos`
- `labels_blocos`
- `criar_preprocessador`
- `min_leaf_dinamico`
- `criar_modelo`
- `criar_pipeline`
- `criar_split`
- `preparar_xy`
- `obter_feature_names`
- `predicoes_por_proba`
- `calcular_roc_auc_binario`
- `calcular_roc_auc_multiclasse`
- ... mais 12 funções

Este é arquivo de algoritmo/modelagem. Leia também as seções 16 e 17, onde os trechos centrais foram explicados em detalhe.


## src/pipeline/__init__.py

**Papel:** Arquivo vazio de pacote Python.

Não há funções definidas; arquivo vazio ou apenas constantes/importações.


## src/pipeline/curate.py

**Papel:** Base curada final: cria aliases, gap, flags, targets auxiliares, região, recorte 2019–2021 e salva parquet/csv/sqlite.

**Funções principais detectadas:**

- `log`
- `validar_colunas`
- `normalizar_texto`
- `normalizar_uf`
- `to_numeric_nullable`
- `to_int_nullable`
- `criar_alias_se_existir`
- `criar_aliases_inscricoes`
- `criar_aliases_ofertas`
- `adicionar_regioes_inscricoes`
- `adicionar_regioes_ofertas`
- `preparar_inscricoes`
- `preparar_ofertas`
- `gerar_recorte_artigo`
- `auditar_cine`
- `auditar_regioes`
- `salvar_sqlite`
- `salvar_resumo`
- `run`


## src/pipeline/staging.py

**Papel:** Primeira etapa: lê/copía os dados brutos para data/02_staging com nomes padronizados e remoção de duplicatas exatas.

**Funções principais detectadas:**

- `safe_text`
- `log`
- `read_fies_csv`
- `remove_colunas_fantasma`
- `nome_saida_fies`
- `copiar_para_errors`
- `staging_fies`
- `encontrar_pasta_inep_ano`
- `encontrar_arquivo_case_insensitive`
- `staging_inep`
- `run`


## src/pipeline/transform/__init__.py

**Papel:** Arquivo vazio de pacote Python.

Não há funções definidas; arquivo vazio ou apenas constantes/importações.


## src/pipeline/transform/cruzamento_cine.py

**Papel:** Cruza FIES com INEP/CINE, aplica normalização textual e mapeamentos manuais para classificar cursos por área/subárea.

**Funções principais detectadas:**

- `log`
- `normalizar_nome`
- `normalizar_nome_sem_acento`
- `normalizar_codigo_cine`
- `preparar_mapa_manual`
- `preparar_inep`
- `preparar_base_fies`
- `construir_referencia_cine`
- `salvar_auxiliar_cursos_validos`
- `diagnosticar_sem_cine`
- `aplicar_mapa_manual`
- `auditar_resultado`
- `enriquecer_com_cine`
- `salvar_resumo`
- `run`


## src/pipeline/transform/limpeza_tipos_fies.py

**Papel:** Transforma CSVs do staging em Parquet limpo: padroniza nomes, converte números brasileiros, datas e textos.

**Funções principais detectadas:**

- `log`
- `parse_staging_filename`
- `limpar_nome_coluna`
- `limpar_texto`
- `limpar_cnpj`
- `normalizar_semestre_financiamento`
- `converter_numero_br`
- `converter_inteiro`
- `normalizar_textos`
- `coalescer_colunas_duplicadas`
- `garantir_colunas`
- `registrar_colunas_nao_mapeadas`
- `transformar_inscricoes`
- `transformar_ofertas`
- `processar_arquivo`
- `run`


## src/pipeline/transform/mestre_inep.py

**Papel:** Constrói uma base mestre do Censo da Educação Superior/INEP para curso, instituição e classificações.

**Funções principais detectadas:**

- `log`
- `normalizar_nome_coluna`
- `limpar_texto`
- `converter_inteiro`
- `detectar_encoding_e_colunas`
- `selecionar_colunas_inep`
- `parse_ano_arquivo`
- `ler_cadastro_cursos`
- `coalescer_colunas_duplicadas`
- `padronizar_cadastro_cursos`
- `listar_arquivos_cursos`
- `auditar_mestre`
- `construir_ultimo_registro`
- `salvar_resumo`
- `run`


## src/pipeline/transform/modalidade.py

**Papel:** Classifica/diagnostica modalidade FIES com base em renda, salário mínimo e ofertas P-FIES.

**Funções principais detectadas:**

- `log`
- `validar_colunas`
- `normalizar_texto`
- `normalizar_turno`
- `converter_numero`
- `converter_inteiro`
- `padronizar_chaves_inscricoes`
- `padronizar_chaves_ofertas`
- `normalizar_participa_p_fies`
- `preparar_ofertas_pfies`
- `classificar_modalidade_por_renda`
- `validar_pfies`
- `salvar_matriz`
- `salvar_resumo`
- `run`


## src/pipeline/transform/unificacao_fies.py

**Papel:** Junta os parquets limpos de inscrições/ofertas em bases unificadas.

**Funções principais detectadas:**

- `log`
- `listar_parquets_limpos`
- `ler_parquets`
- `ordenar_se_possivel`
- `auditar_duplicatas_exatas`
- `auditar_chaves_inscricoes`
- `auditar_chaves_ofertas`
- `salvar_resumo`
- `unificar_inscricoes`
- `unificar_ofertas`
- `run`


---

# 22. Diferenças principais entre robusto e legado, agora com sentido técnico

## 22.1 Dummies: `pd.get_dummies` vs `OneHotEncoder`

Legado:

```python
df = pd.get_dummies(df_base, columns=[...], drop_first=True)
```

Isso transforma categorias em colunas 0/1 antes do modelo. Funciona, mas se treino e teste forem separados depois, pode haver problema se uma categoria aparecer em um conjunto e não no outro.

Robusto:

```python
OneHotEncoder(handle_unknown="ignore", min_frequency=20)
```

Isso fica dentro do pipeline. Ele aprende categorias no treino e aplica no teste com segurança. Categoria desconhecida é ignorada, não quebra.

## 22.2 Scaler: escalar tudo vs escalar só numéricas

Legado:

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', LogisticRegression(...))
])
```

Como dummies já estavam no `X`, o scaler podia padronizar numéricas e dummies.

Robusto:

```python
ColumnTransformer([
    ('num', numeric_transformer, numericas),
    ('cat', categorical_transformer, categoricas)
])
```

Mais claro: scaler só nas numéricas, one-hot só nas categóricas.

## 22.3 Random Forest vs árvore interpretável

Legado usava Random Forest em vários experimentos. Random Forest é forte, mas menos transparente: junta muitas árvores. Fica mais difícil explicar uma regra específica.

Robusto usa árvores de decisão com profundidades 10, 14 e 19 como núcleo final. É mais interpretável: dá para extrair primeira divisão, regras, importâncias e explicar o caminho da decisão.

## 22.4 Scripts exploratórios vs pipeline reprodutível

Legado: vários `analise_00x.py` independentes.

Robusto: cada etapa tem responsabilidade clara e pode ser chamada pelo `main.py`.

---

# 23. Respostas prontas para defesa

**O que é ABT?**

ABT é a tabela final de modelagem. Cada linha é uma inscrição e cada coluna é uma variável usada pelo modelo. A ABT binária tem target 0/1. A ABT ternária tem target 0/1/2.

**O que é `gap`?**

É a diferença entre média do Enem e nota de corte do grupo de preferência. Se o gap é positivo, o candidato ficou acima da nota de corte. Se é negativo, ficou abaixo.

**O que é `renda_gap`?**

É uma interação entre renda e desempenho relativo à nota de corte. Serve para permitir que o modelo observe se a combinação entre renda e desempenho ajuda a diferenciar as classes.

**Por que `OneHotEncoder`?**

Porque modelos não entendem texto. Ele transforma categorias como turno, curso e região em colunas numéricas 0/1.

**Por que `StandardScaler`?**

Porque regressão logística é sensível à escala das variáveis. Ele coloca as numéricas em escala comparável.

**Por que árvore não usa scaler?**

Porque árvore escolhe cortes nos valores. Ela não precisa que as variáveis estejam na mesma escala.

**Por que `class_weight="balanced"`?**

Porque as classes são desbalanceadas. Contratada é menor. O peso balanceado impede que o modelo ignore a classe minoritária.

**O que é F1 macro?**

É a média do F1 das classes, dando o mesmo peso para cada classe. No ternário, isso evita que a classe maior esconda erro na classe menor.

**O que é ROC-AUC?**

É uma métrica que mede a capacidade do modelo de separar classes a partir das probabilidades. Quanto mais perto de 1, melhor.

**O que é matriz de confusão?**

É uma tabela real vs previsto. Mostra onde o modelo acertou e onde errou.

**Isso prova causalidade?**

Não. O código faz modelagem preditiva e análise associativa. Ele não prova que uma variável causa diretamente o desfecho.

---

# 24. Veredito técnico

O `src` robusto está logicamente correto como pipeline de pesquisa e modelagem:

- organiza dados brutos;
- transforma e cura bases;
- monta ABTs coerentes;
- usa preprocessamento adequado;
- trata numéricas e categóricas de forma separada;
- treina regressão logística binária e multinomial;
- treina árvores de decisão por profundidade;
- usa avaliação `in_sample` e `holdout_80_20`;
- calcula métricas adequadas para classes desbalanceadas;
- gera probabilidades previstas, tabelas, figuras e apêndices.

Para entender tudo, o caminho mental é:

```text
1. O pipeline limpa e organiza os microdados.
2. A curadoria cria variáveis-chave: renda, gap, situação, target.
3. A ABT escolhe quais linhas e colunas vão para modelagem.
4. O Pipeline do scikit-learn trata os dados automaticamente.
5. A regressão logística e a árvore aprendem padrões.
6. As métricas avaliam o desempenho.
7. As figuras/tabelas transformam isso em resultado de artigo.
```

O ponto mais avançado do robusto não é a ideia do modelo em si. É a **organização reprodutível**: cada etapa salva saída, cada modelo salva metadados, e os produtos finais podem ser recriados por comandos do `main.py`.
