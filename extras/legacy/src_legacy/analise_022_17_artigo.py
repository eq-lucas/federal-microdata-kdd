# %%
# analise_022.py

from pathlib import Path
import textwrap
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ==============================================================================
# CONFIGURAÇÕES GERAIS
# ==============================================================================

warnings.filterwarnings("ignore")

CAMINHO_BASE = Path("../data/04_load/database/inscritos_final_limpo.parquet")

PASTA_PROCESSADOS = Path("../data/05_processed/analise_022")
PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)

PASTA_MODELO = Path("../models/analise_022")
PASTA_MODELO.mkdir(parents=True, exist_ok=True)

CAMINHO_MODELO = PASTA_MODELO / "modelo_logit_analise_022.joblib"

PASTA_FIGURAS = Path("../reports/figures/analise_022")
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

PASTA_APENDICE = Path("../reports/figures/analise_022/apendice_C")
PASTA_APENDICE.mkdir(parents=True, exist_ok=True)

# Mantém o desenho do seu código original:
# X_treino = X_teste = base analítica completa.
USAR_BASE_INTEIRA_COMO_TREINO_E_TESTE = True

# Se quiser gerar tabela com especificações alternativas no apêndice, mantenha True.
# Pode demorar porque treina modelos adicionais.
GERAR_ESPECIFICACOES_ALTERNATIVAS = True

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 1.0

COR_CABECALHO = "#1f4e79"
COR_LINHA_1 = "#eaf2f8"
COR_LINHA_2 = "#ffffff"
COR_BORDA = "#b7c9d6"

CMAP_MATRIZ = "Greys"
CMAP_CONFUSAO = "Greys"

BINS_RENDA = [-np.inf, 600, 1200, 1800, 2400, 3000, np.inf]
LABELS_RENDA = ["I", "II", "III", "IV", "V", "VI"]

BINS_DESEMPENHO = [-np.inf, -150, -50, 0, 50, 150, np.inf]
LABELS_DESEMPENHO = [
    "< -150",
    "-150 a -50",
    "-50 a 0",
    "0 a +50",
    "+50 a +150",
    "> +150"
]

ORDEM_DESEMPENHO_PLOT = [
    "> +150",
    "+50 a +150",
    "0 a +50",
    "-50 a 0",
    "-150 a -50",
    "< -150"
]


# ==============================================================================
# ORQUESTRADOR
# ==============================================================================

def orquestrador_analise_022():
    print("\n" + "=" * 80)
    print("ANÁLISE 022 — MODELO LOGÍSTICO PARA CONTRATAÇÃO EFETIVA")
    print("=" * 80)

    X, y, df_modelo = gerar_abt_analise_022()

    modelo = treinar_modelo_logistico(X, y)
    joblib.dump(modelo, CAMINHO_MODELO)
    print(f"\n[OK] Modelo salvo em: {CAMINHO_MODELO}")

    resultados = avaliar_modelo(modelo, X, y)

    df_pred = anexar_probabilidades_previstas(
        df_modelo=df_modelo,
        modelo=modelo,
        X=X,
        y=y,
        resultados=resultados
    )

    gerar_tabela_metricas(resultados)
    gerar_tabela_coeficientes_principais(modelo, X)
    gerar_tabela_top_coeficientes(modelo, X)

    if GERAR_ESPECIFICACOES_ALTERNATIVAS:
        gerar_tabela_especificacoes_alternativas(X, y)

    gerar_figura_probabilidade_prevista(df_pred)
    gerar_figura_matriz_confusao(resultados)
    gerar_figura_curva_roc(resultados)
    gerar_figura_curvas_desempenho_por_renda(df_pred)
    gerar_figura_curvas_renda_por_desempenho(df_pred)

    print("\n✅ ANÁLISE 022 FINALIZADA")
    print(f"Figuras principais: {PASTA_FIGURAS.resolve()}")
    print(f"Apêndice C: {PASTA_APENDICE.resolve()}")


# ==============================================================================
# PREPARAÇÃO DA BASE ANALÍTICA
# ==============================================================================

def calcular_idade(valor):
    try:
        ano = int(str(valor).split("/")[-1])
        return 2026 - ano
    except Exception:
        return np.nan


def gerar_abt_analise_022():
    print("\n[*] Carregando base...")
    df_base = pd.read_parquet(CAMINHO_BASE)

    print(f"Base original: {df_base.shape}")

    df_base = df_base.copy()

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

    print("\nDistribuição original por situação:")
    print(
        df_base.groupby("situacao_fies", as_index=False)
        .size()
        .rename(columns={"size": "qtde_inscritos"})
        .sort_values("qtde_inscritos", ascending=False)
    )

    # Mantém apenas os dois status do modelo binário.
    df_base = df_base[
        df_base["situacao_fies"].isin(["CONTRATADA", "NÃO CONTRATADO", "NAO CONTRATADO"])
    ].copy()

    df_base["contratado"] = np.where(
        df_base["situacao_fies"] == "CONTRATADA",
        1,
        0
    )

    colunas_categoricas = [
        "beneficiado_creduc_fies",
        "modalidade_fies",
        "subarea_conhecimento",
        "regiao_morar",
        "natureza_juridica_mantenedora",
        "etnia_cor",
        "turno",
        "ensino_medio_escola_publica",
        "conceito_curso_gp",
        "concluiu_curso_superior",
        "opcao_curso"
    ]

    colunas_categoricas = [
        col for col in colunas_categoricas if col in df_base.columns
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
    ]

    colunas_dummies = [
        col for col in df.columns
        if any(col.startswith(prefixo) for prefixo in grupos_dummies)
    ]

    features = [
        "renda_per_capita",
        "gap",
        "idade",
        "nota_corte_gp",
        "renda_gap"
    ] + colunas_dummies

    features = [col for col in features if col in df.columns]

    for col in features:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    print("\n--- Quantidade de NaNs por variável do modelo ---")
    print(df[features].isna().sum().sort_values(ascending=False).head(20))

    df_limpo = df.dropna(subset=features + ["contratado"]).copy()

    X = df_limpo[features].copy()
    y = df_limpo["contratado"].astype(int).copy()

    df_modelo = df_limpo[[
        "renda_per_capita",
        "gap",
        "contratado"
    ]].copy()

    print("\n--- Base analítica binária ---")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"Taxa real de contratação: {y.mean():.4f}")

    X.to_parquet(PASTA_PROCESSADOS / "x_treino.parquet", index=False)
    y.to_frame("contratado").to_parquet(PASTA_PROCESSADOS / "y_treino.parquet", index=False)

    X.to_parquet(PASTA_PROCESSADOS / "x_teste.parquet", index=False)
    y.to_frame("contratado").to_parquet(PASTA_PROCESSADOS / "y_teste.parquet", index=False)

    df_modelo.to_parquet(PASTA_PROCESSADOS / "base_modelo_com_variaveis_chave.parquet", index=False)

    print(f"\n[OK] Matrizes salvas em: {PASTA_PROCESSADOS}")

    return X, y, df_modelo


# ==============================================================================
# MODELO
# ==============================================================================

def treinar_modelo_logistico(X, y):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            solver="saga",
            C=0.1,
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X, y.values.ravel())

    print("\n[OK] Treinamento concluído")

    return pipeline


def avaliar_modelo(modelo, X, y):
    prob = modelo.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, prob)
    fpr, tpr, thresholds = roc_curve(y, prob)

    melhor_indice = np.argmax(tpr - fpr)
    threshold = thresholds[melhor_indice]

    y_pred = (prob >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    matriz = confusion_matrix(y, y_pred)

    relatorio = classification_report(
        y,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    resultados = {
        "prob": prob,
        "y_pred": y_pred,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": threshold,
        "accuracy": acc,
        "matriz": matriz,
        "relatorio": relatorio,
        "n": len(y),
        "n_nao_contratado": int((y == 0).sum()),
        "n_contratado": int((y == 1).sum()),
        "taxa_real": float(y.mean()),
        "taxa_prevista": float(y_pred.mean())
    }

    print("\n--- Métricas do modelo ---")
    print(f"Observações: {fmt_int(resultados['n'])}")
    print(f"Não contratados: {fmt_int(resultados['n_nao_contratado'])}")
    print(f"Contratados: {fmt_int(resultados['n_contratado'])}")
    print(f"Taxa real: {fmt_pct_decimal(resultados['taxa_real'])}")
    print(f"Taxa prevista: {fmt_pct_decimal(resultados['taxa_prevista'])}")
    print(f"Acurácia: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print("\nMatriz de confusão:")
    print(matriz)

    return resultados


def anexar_probabilidades_previstas(df_modelo, modelo, X, y, resultados):
    df_pred = df_modelo.copy()
    df_pred["prob_contratacao"] = resultados["prob"] * 100
    df_pred["previsto"] = resultados["y_pred"]
    df_pred["contratado"] = y.values

    df_pred["faixa_renda"] = pd.cut(
        df_pred["renda_per_capita"],
        bins=BINS_RENDA,
        labels=LABELS_RENDA,
        ordered=True
    )

    df_pred["faixa_desempenho"] = pd.cut(
        df_pred["gap"],
        bins=BINS_DESEMPENHO,
        labels=LABELS_DESEMPENHO,
        ordered=True
    )

    df_pred = df_pred.dropna(subset=["faixa_renda", "faixa_desempenho"]).copy()

    df_pred.to_parquet(PASTA_PROCESSADOS / "base_com_probabilidades_previstas.parquet", index=False)

    return df_pred


# ==============================================================================
# FORMATADORES
# ==============================================================================

def fmt_int(valor):
    return f"{int(valor):,}".replace(",", ".")


def fmt_float(valor, casas=4):
    return f"{float(valor):.{casas}f}".replace(".", ",")


def fmt_pct_decimal(valor, casas=1):
    return f"{float(valor) * 100:.{casas}f}%".replace(".", ",")


def fmt_pct_puro(valor, casas=1):
    return f"{float(valor):.{casas}f}%".replace(".", ",")


def quebrar_texto(texto, largura=48):
    return "\n".join(textwrap.wrap(str(texto), width=largura))


# ==============================================================================
# TABELAS
# ==============================================================================

def salvar_tabela_como_imagem(
    df_tabela,
    caminho_png,
    caminho_pdf,
    figsize=(7.2, 3.2),
    col_widths=None,
    fontsize=10.5
):
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    tabela = ax.table(
        cellText=df_tabela.values,
        colLabels=df_tabela.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1],
        colWidths=col_widths
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(fontsize)

    for (linha, coluna), celula in tabela.get_celld().items():
        celula.set_edgecolor(COR_BORDA)
        celula.set_linewidth(0.8)
        celula.PAD = 0.025

        if linha == 0:
            celula.set_facecolor(COR_CABECALHO)
            celula.get_text().set_color("white")
            celula.get_text().set_fontweight("bold")
            celula.get_text().set_ha("center")
            celula.get_text().set_va("center")
        else:
            celula.set_facecolor(
                COR_LINHA_1 if linha % 2 == 1 else COR_LINHA_2
            )

            if coluna == 0:
                celula.get_text().set_fontweight("bold")
                celula.get_text().set_ha("left")
            else:
                celula.get_text().set_ha("center")

            celula.get_text().set_va("center")

    fig.savefig(caminho_png, dpi=700, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(caminho_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Imagem salva em: {caminho_png}")
    print(f"PDF salvo em: {caminho_pdf}")


def gerar_tabela_metricas(resultados):
    rel = resultados["relatorio"]

    linhas = [
        ["Observações", fmt_int(resultados["n"])],
        ["Não contratados", fmt_int(resultados["n_nao_contratado"])],
        ["Contratados", fmt_int(resultados["n_contratado"])],
        ["Taxa real de contratação", fmt_pct_decimal(resultados["taxa_real"])],
        ["Taxa prevista pelo modelo", fmt_pct_decimal(resultados["taxa_prevista"])],
        ["Acurácia", fmt_float(resultados["accuracy"])],
        ["ROC-AUC", fmt_float(resultados["auc"])],
        ["Threshold", fmt_float(resultados["threshold"])],
        ["Precisão — não contratado", fmt_float(rel["0"]["precision"])],
        ["Recall — não contratado", fmt_float(rel["0"]["recall"])],
        ["F1 — não contratado", fmt_float(rel["0"]["f1-score"])],
        ["Precisão — contratado", fmt_float(rel["1"]["precision"])],
        ["Recall — contratado", fmt_float(rel["1"]["recall"])],
        ["F1 — contratado", fmt_float(rel["1"]["f1-score"])]
    ]

    df_tabela = pd.DataFrame(linhas, columns=["Métrica", "Valor"])

    salvar_tabela_como_imagem(
        df_tabela=df_tabela,
        caminho_png=PASTA_APENDICE / "tabela_C1_metricas_modelo_logistico.png",
        caminho_pdf=PASTA_APENDICE / "tabela_C1_metricas_modelo_logistico.pdf",
        figsize=(7.2, 5.1),
        col_widths=[0.68, 0.32],
        fontsize=10.3
    )


def nome_variavel_artigo(nome):
    mapa = {
        "renda_per_capita": "Renda familiar per capita",
        "gap": "Desempenho acadêmico",
        "renda_gap": "Interação renda × desempenho",
        "idade": "Idade",
        "nota_corte_gp": "Nota de corte do grupo"
    }

    if nome in mapa:
        return mapa[nome]

    nome = nome.replace("opcao_curso_", "Opção de curso ")
    nome = nome.replace("modalidade_fies_", "Modalidade Fies: ")
    nome = nome.replace("beneficiado_creduc_fies_", "Beneficiado CREDUC/Fies: ")
    nome = nome.replace("concluiu_curso_superior_", "Concluiu curso superior: ")
    nome = nome.replace("conceito_curso_gp_", "Conceito do curso: ")
    nome = nome.replace("subarea_conhecimento_", "Subárea: ")
    nome = nome.replace("regiao_morar_", "Região: ")
    nome = nome.replace("natureza_juridica_mantenedora_", "Natureza jurídica: ")
    nome = nome.replace("etnia_cor_", "Cor/raça: ")
    nome = nome.replace("turno_", "Turno: ")
    nome = nome.replace("ensino_medio_escola_publica_", "Ensino médio em escola pública: ")
    nome = nome.replace("_", " ")

    return nome


def gerar_tabela_coeficientes_principais(modelo, X):
    modelo_logit = modelo.named_steps["modelo"]
    coeficientes = modelo_logit.coef_[0]
    intercepto = modelo_logit.intercept_[0]

    df_coef = pd.DataFrame({
        "variavel_original": X.columns,
        "coeficiente": coeficientes
    })

    variaveis_chave = [
        "renda_per_capita",
        "gap",
        "renda_gap",
        "nota_corte_gp",
        "idade"
    ]

    df_chave = df_coef[
        df_coef["variavel_original"].isin(variaveis_chave)
    ].copy()

    df_chave["Variável"] = df_chave["variavel_original"].apply(nome_variavel_artigo)

    ordem = [
        "Renda familiar per capita",
        "Desempenho acadêmico",
        "Interação renda × desempenho",
        "Nota de corte do grupo",
        "Idade"
    ]

    df_chave["ordem"] = df_chave["Variável"].apply(
        lambda x: ordem.index(x) if x in ordem else 999
    )

    df_chave = df_chave.sort_values("ordem")
    df_chave["Coeficiente padronizado"] = df_chave["coeficiente"].apply(lambda x: fmt_float(x, 4))
    df_chave = df_chave[["Variável", "Coeficiente padronizado"]]

    linha_intercepto = pd.DataFrame({
        "Variável": ["Intercepto"],
        "Coeficiente padronizado": [fmt_float(intercepto, 4)]
    })

    df_saida = pd.concat([linha_intercepto, df_chave], ignore_index=True)

    salvar_tabela_como_imagem(
        df_tabela=df_saida,
        caminho_png=PASTA_FIGURAS / "tabela_coeficientes_principais_logit.png",
        caminho_pdf=PASTA_FIGURAS / "tabela_coeficientes_principais_logit.pdf",
        figsize=(7.2, 2.65),
        col_widths=[0.70, 0.30],
        fontsize=10.4
    )


def gerar_tabela_top_coeficientes(modelo, X):
    modelo_logit = modelo.named_steps["modelo"]
    coeficientes = modelo_logit.coef_[0]

    df_coef = pd.DataFrame({
        "Variável": [nome_variavel_artigo(c) for c in X.columns],
        "Coeficiente": coeficientes
    })

    df_coef["abs"] = df_coef["Coeficiente"].abs()
    df_coef = df_coef.sort_values("abs", ascending=False).head(15)

    df_coef["Variável"] = df_coef["Variável"].apply(lambda x: quebrar_texto(x, 50))
    df_coef["Coeficiente"] = df_coef["Coeficiente"].apply(lambda x: fmt_float(x, 4))

    df_coef = df_coef[["Variável", "Coeficiente"]]

    salvar_tabela_como_imagem(
        df_tabela=df_coef,
        caminho_png=PASTA_APENDICE / "tabela_C2_top_15_coeficientes_logit.png",
        caminho_pdf=PASTA_APENDICE / "tabela_C2_top_15_coeficientes_logit.pdf",
        figsize=(7.2, 6.6),
        col_widths=[0.80, 0.20],
        fontsize=8.4
    )


def selecionar_colunas_por_prefixo(X, prefixos):
    cols = []
    for col in X.columns:
        if any(col.startswith(prefixo) for prefixo in prefixos):
            cols.append(col)
    return cols


def gerar_tabela_especificacoes_alternativas(X, y):
    print("\n[*] Gerando especificações alternativas...")

    colunas_base = [
        col for col in ["renda_per_capita", "gap", "renda_gap"]
        if col in X.columns
    ]

    colunas_academicas = [
        col for col in ["idade", "nota_corte_gp"]
        if col in X.columns
    ]

    colunas_processo = selecionar_colunas_por_prefixo(
        X,
        ["opcao_curso_", "conceito_curso_gp_", "turno_", "ensino_medio_escola_publica_"]
    )

    colunas_contexto = selecionar_colunas_por_prefixo(
        X,
        [
            "subarea_conhecimento_",
            "regiao_morar_",
            "natureza_juridica_mantenedora_",
            "etnia_cor_",
            "modalidade_fies_",
            "beneficiado_creduc_fies_",
            "concluiu_curso_superior_"
        ]
    )

    especificacoes = {
        "Modelo 1": colunas_base,
        "Modelo 2": list(dict.fromkeys(colunas_base + colunas_academicas + colunas_processo)),
        "Modelo 3": list(dict.fromkeys(colunas_base + colunas_academicas + colunas_processo + colunas_contexto))
    }

    linhas = []

    for nome, cols in especificacoes.items():
        cols = [c for c in cols if c in X.columns]

        modelo = treinar_modelo_logistico(X[cols], y)
        prob = modelo.predict_proba(X[cols])[:, 1]
        auc = roc_auc_score(y, prob)

        modelo_logit = modelo.named_steps["modelo"]
        coef = pd.Series(modelo_logit.coef_[0], index=cols)

        linhas.append([
            nome,
            fmt_int(len(y)),
            fmt_float(auc),
            fmt_float(coef.get("renda_per_capita", np.nan)),
            fmt_float(coef.get("gap", np.nan)),
            fmt_float(coef.get("renda_gap", np.nan)),
            len(cols)
        ])

    df_tabela = pd.DataFrame(
        linhas,
        columns=[
            "Especificação",
            "N",
            "ROC-AUC",
            "Renda",
            "Desempenho",
            "Interação",
            "Variáveis"
        ]
    )

    salvar_tabela_como_imagem(
        df_tabela=df_tabela,
        caminho_png=PASTA_APENDICE / "tabela_C3_especificacoes_alternativas_logit.png",
        caminho_pdf=PASTA_APENDICE / "tabela_C3_especificacoes_alternativas_logit.pdf",
        figsize=(7.2, 2.2),
        col_widths=[0.20, 0.18, 0.13, 0.13, 0.13, 0.13, 0.10],
        fontsize=9.2
    )


# ==============================================================================
# FIGURA PRINCIPAL — PROBABILIDADE PREVISTA POR FAIXAS REAIS
# ==============================================================================

def matriz_probabilidade_prevista(df_pred):
    matriz = (
        df_pred
        .groupby(["faixa_desempenho", "faixa_renda"], observed=True)["prob_contratacao"]
        .mean()
        .reset_index()
        .pivot(index="faixa_desempenho", columns="faixa_renda", values="prob_contratacao")
        .reindex(index=ORDEM_DESEMPENHO_PLOT, columns=LABELS_RENDA)
    )

    return matriz


def matriz_contagem_observacoes(df_pred):
    matriz_n = (
        df_pred
        .groupby(["faixa_desempenho", "faixa_renda"], observed=True)
        .size()
        .reset_index(name="n")
        .pivot(index="faixa_desempenho", columns="faixa_renda", values="n")
        .reindex(index=ORDEM_DESEMPENHO_PLOT, columns=LABELS_RENDA)
    )

    return matriz_n


def gerar_figura_probabilidade_prevista(df_pred):
    matriz = matriz_probabilidade_prevista(df_pred)

    fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=300)

    sns.heatmap(
        matriz,
        cmap=CMAP_MATRIZ,
        vmin=0,
        vmax=100,
        annot=False,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        ax=ax
    )

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = matriz.iloc[i, j]
            if pd.isna(valor):
                texto = ""
                cor = "black"
            else:
                texto = f"{valor:.1f}"
                cor = "white" if valor >= 55 else "black"

            ax.text(
                j + 0.5,
                i + 0.5,
                texto,
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=cor
            )

    ax.set_xlabel("Faixa de renda familiar per capita", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Desempenho acadêmico", fontsize=11, fontweight="bold", labelpad=10)

    ax.set_xticklabels(LABELS_RENDA, rotation=0, fontsize=10.5)
    ax.set_yticklabels(ORDEM_DESEMPENHO_PLOT, rotation=0, fontsize=10.5)

    ax.tick_params(axis="both", length=0)

    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.16, top=0.985)

    caminho_png = PASTA_FIGURAS / "figura_probabilidade_prevista_contratacao_logit.png"
    caminho_pdf = PASTA_FIGURAS / "figura_probabilidade_prevista_contratacao_logit.pdf"

    fig.savefig(caminho_png, dpi=700, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(caminho_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Figura principal salva em: {caminho_png}")


# ==============================================================================
# APÊNDICE — MATRIZ DE CONFUSÃO E ROC
# ==============================================================================

def gerar_figura_matriz_confusao(resultados):
    matriz = pd.DataFrame(
        resultados["matriz"],
        index=["Não contratado", "Contratado"],
        columns=["Não contratado", "Contratado"]
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=300)

    sns.heatmap(
        matriz,
        annot=False,
        cmap=CMAP_CONFUSAO,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        ax=ax
    )

    max_val = matriz.values.max()

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = matriz.iloc[i, j]
            cor = "white" if valor >= max_val * 0.45 else "black"

            ax.text(
                j + 0.5,
                i + 0.5,
                fmt_int(valor),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=cor
            )

    ax.set_xlabel("Previsto", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Real", fontsize=11, fontweight="bold", labelpad=10)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10.5)

    ax.tick_params(axis="both", length=0)

    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.14, top=0.985)

    caminho_png = PASTA_APENDICE / "figura_C1_matriz_confusao_logit.png"
    caminho_pdf = PASTA_APENDICE / "figura_C1_matriz_confusao_logit.pdf"

    fig.savefig(caminho_png, dpi=700, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(caminho_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Matriz de confusão salva em: {caminho_png}")


def gerar_figura_curva_roc(resultados):
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)

    ax.plot(
        resultados["fpr"],
        resultados["tpr"],
        color="black",
        linewidth=2.2,
        label=f"ROC-AUC = {resultados['auc']:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Classificador aleatório"
    )

    ax.set_xlabel("Taxa de falso positivo", fontsize=11, fontweight="bold")
    ax.set_ylabel("Taxa de verdadeiro positivo", fontsize=11, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        fontsize=10
    )

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.14, top=0.82)

    caminho_png = PASTA_APENDICE / "figura_C2_curva_roc_logit.png"
    caminho_pdf = PASTA_APENDICE / "figura_C2_curva_roc_logit.pdf"

    fig.savefig(caminho_png, dpi=700, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(caminho_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Curva ROC salva em: {caminho_png}")


# ==============================================================================
# APÊNDICE — CURVAS BASEADAS NAS MESMAS FAIXAS DA MATRIZ
# ==============================================================================

def tabela_media_prob_por_faixas(df_pred):
    tabela = (
        df_pred
        .groupby(["faixa_renda", "faixa_desempenho"], observed=True)["prob_contratacao"]
        .mean()
        .reset_index()
    )
    return tabela


def gerar_figura_curvas_desempenho_por_renda(df_pred):
    tabela = tabela_media_prob_por_faixas(df_pred)

    estilos = ["-", "--", "-.", ":", "-", "--"]
    marcadores = ["o", "s", "^", "D", "P", "X"]

    fig, ax = plt.subplots(figsize=(8.2, 5.3), dpi=300)

    x = np.arange(len(LABELS_DESEMPENHO))

    for idx, faixa_renda in enumerate(LABELS_RENDA):
        subset = (
            tabela[tabela["faixa_renda"] == faixa_renda]
            .set_index("faixa_desempenho")
            .reindex(LABELS_DESEMPENHO)
            .reset_index()
        )

        ax.plot(
            x,
            subset["prob_contratacao"],
            linestyle=estilos[idx % len(estilos)],
            marker=marcadores[idx % len(marcadores)],
            linewidth=1.8,
            markersize=5.0,
            label=faixa_renda,
            color=str(0.15 + idx * 0.12)
        )

    ax.set_xlabel("Desempenho acadêmico", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probabilidade prevista de contratação (%)", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_DESEMPENHO, rotation=0, fontsize=9.5)

    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        title="Faixa de renda",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        fontsize=9.3,
        title_fontsize=9.8
    )

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.16, top=0.78)

    caminho_png = PASTA_APENDICE / "figura_C3_curvas_desempenho_por_renda.png"
    caminho_pdf = PASTA_APENDICE / "figura_C3_curvas_desempenho_por_renda.pdf"

    fig.savefig(caminho_png, dpi=700, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(caminho_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Curvas por renda salvas em: {caminho_png}")


def gerar_figura_curvas_renda_por_desempenho(df_pred):
    tabela = tabela_media_prob_por_faixas(df_pred)

    estilos = ["-", "--", "-.", ":", "-", "--"]
    marcadores = ["o", "s", "^", "D", "P", "X"]

    fig, ax = plt.subplots(figsize=(8.2, 5.3), dpi=300)

    x = np.arange(len(LABELS_RENDA))

    for idx, faixa_desempenho in enumerate(LABELS_DESEMPENHO):
        subset = (
            tabela[tabela["faixa_desempenho"] == faixa_desempenho]
            .set_index("faixa_renda")
            .reindex(LABELS_RENDA)
            .reset_index()
        )

        ax.plot(
            x,
            subset["prob_contratacao"],
            linestyle=estilos[idx % len(estilos)],
            marker=marcadores[idx % len(marcadores)],
            linewidth=1.8,
            markersize=5.0,
            label=faixa_desempenho,
            color=str(0.15 + idx * 0.12)
        )

    ax.set_xlabel("Faixa de renda familiar per capita", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probabilidade prevista de contratação (%)", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_RENDA, rotation=0, fontsize=10.5)

    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        title="Desempenho acadêmico",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        fontsize=9.3,
        title_fontsize=9.8
    )

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.72)

    caminho_png = PASTA_APENDICE / "figura_C4_curvas_renda_por_desempenho.png"
    caminho_pdf = PASTA_APENDICE / "figura_C4_curvas_renda_por_desempenho.pdf"

    fig.savefig(caminho_png, dpi=700, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(caminho_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Curvas por desempenho salvas em: {caminho_png}")


# ==============================================================================
# EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    orquestrador_analise_022()

# %%