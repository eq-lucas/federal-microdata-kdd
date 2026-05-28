from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import pearsonr, linregress

from src.constants import (
    CURATED_INSCRICOES_ARTIGO_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    LOGS_DIR,
)


# =============================================================================
# Saídas
# =============================================================================

FIGURES_FINANCIAMENTO_DIR = FIGURES_DIR / "financiamento_coparticipacao"
TABLES_FINANCIAMENTO_DIR = TABLES_DIR / "secao_4_3"

RESUMO_PATH = LOGS_DIR / "article_financiamento_coparticipacao_resumo.csv"


# =============================================================================
# Configurações
# =============================================================================

FIG_DPI = 300
SAVE_DPI = 1200

RENDA_MINIMA = 0
RENDA_GRAFICO_MAXIMO_PADRAO = 5000
PERCENTUAL_MINIMO = 0
PERCENTUAL_MAXIMO = 100

POSSIVEIS_COLUNAS_RENDA = [
    "renda_per_capita",
    "renda_familiar_per_capita",
    "renda_mensal_bruta_per_capita",
    "vl_renda_per_capita",
    "valor_renda_per_capita",
    "renda_familiar_mensal_per_capita",
]

POSSIVEIS_COLUNAS_PERCENTUAL = [
    "percentual_financiamento",
    "percentual_financiado",
    "perc_financiamento",
    "percentual_de_financiamento",
    "percentual_fies",
]

POSSIVEIS_COLUNAS_SITUACAO = [
    "situacao_fies",
    "situacao_inscricao_fies",
    "situacao_inscricao",
    "situacao",
]

STATUS_CONTRATADA = "CONTRATADA"


# =============================================================================
# Utilitários
# =============================================================================

def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "article_financiamento_coparticipacao.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def remover_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(ch for ch in texto if not unicodedata.combining(ch))


def normalizar_texto(valor):
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip()
    texto = re.sub(r"\s+", " ", texto)

    if texto.upper() in {"", "NAN", "NONE", "NULL", "NA", "N/A", "-", "--"}:
        return pd.NA

    return texto.upper()


def normalizar_status(valor):
    texto = normalizar_texto(valor)

    if pd.isna(texto):
        return pd.NA

    texto = remover_acentos(texto).upper()

    if texto == "CONTRATADA":
        return STATUS_CONTRATADA

    return texto


def encontrar_coluna(df: pd.DataFrame, candidatas: list[str], nome_logico: str) -> str:
    for coluna in candidatas:
        if coluna in df.columns:
            return coluna

    raise ValueError(
        f"Não encontrei a coluna de {nome_logico}. "
        f"Tentei: {candidatas}. "
        f"Colunas disponíveis: {list(df.columns)}"
    )


def converter_numero_serie(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")

    s = (
        serie
        .astype("string")
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace("R$", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    tem_virgula = s.str.contains(",", regex=False, na=False)

    s_convertida = s.copy()
    s_convertida.loc[tem_virgula] = (
        s_convertida.loc[tem_virgula]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(s_convertida, errors="coerce")


def obter_fonte_padrao() -> str:
    fontes_disponiveis = {f.name for f in font_manager.fontManager.ttflist}

    if "Times New Roman" in fontes_disponiveis:
        return "Times New Roman"
    if "Liberation Serif" in fontes_disponiveis:
        return "Liberation Serif"
    if "Nimbus Roman" in fontes_disponiveis:
        return "Nimbus Roman"

    return "DejaVu Serif"


def configurar_matplotlib() -> None:
    plt.rcParams["font.family"] = obter_fonte_padrao()
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.dpi"] = FIG_DPI
    plt.rcParams["savefig.dpi"] = SAVE_DPI


def formatar_inteiro(valor) -> str:
    return f"{int(valor):,}".replace(",", ".")


def formatar_decimal(valor, casas: int = 4) -> str:
    if pd.isna(valor):
        return ""

    return f"{float(valor):.{casas}f}".replace(".", ",")


def formatar_pp(valor, casas: int = 2) -> str:
    if pd.isna(valor):
        return ""

    return f"{float(valor):.{casas}f} p.p.".replace(".", ",")


def formatar_p_valor(valor) -> str:
    if pd.isna(valor):
        return ""

    if valor < 0.001:
        return "< 0,001"

    return f"{float(valor):.4f}".replace(".", ",")


# =============================================================================
# Dados e estatísticas
# =============================================================================

def carregar_base() -> pd.DataFrame:
    if not CURATED_INSCRICOES_ARTIGO_PATH.exists():
        raise FileNotFoundError(
            f"Base curada não encontrada: {CURATED_INSCRICOES_ARTIGO_PATH}. "
            "Rode primeiro: python3 main.py pipeline curate"
        )

    df = pd.read_parquet(CURATED_INSCRICOES_ARTIGO_PATH)

    log(f"[OK] Base carregada: {CURATED_INSCRICOES_ARTIGO_PATH}")
    log(f"[OK] Linhas: {len(df)} | Colunas: {len(df.columns)}")

    return df


def preparar_dados_financiamento(df_base: pd.DataFrame) -> pd.DataFrame:
    coluna_renda = encontrar_coluna(df_base, POSSIVEIS_COLUNAS_RENDA, "renda per capita")
    coluna_percentual = encontrar_coluna(df_base, POSSIVEIS_COLUNAS_PERCENTUAL, "percentual de financiamento")
    coluna_situacao = encontrar_coluna(df_base, POSSIVEIS_COLUNAS_SITUACAO, "situação da inscrição")

    df = df_base.copy()

    df["_situacao_norm"] = df[coluna_situacao].map(normalizar_status)
    df = df[df["_situacao_norm"].eq(STATUS_CONTRATADA)].copy()

    df["renda_per_capita_analise"] = converter_numero_serie(df[coluna_renda])
    df["percentual_financiamento_analise"] = converter_numero_serie(df[coluna_percentual])

    max_percentual = df["percentual_financiamento_analise"].max(skipna=True)

    if pd.notna(max_percentual) and max_percentual <= 1.5:
        df["percentual_financiamento_analise"] = df["percentual_financiamento_analise"] * 100

    linhas_contratadas = len(df)

    df = df.dropna(
        subset=[
            "renda_per_capita_analise",
            "percentual_financiamento_analise",
        ]
    ).copy()

    linhas_validas_antes_filtros = len(df)

    df = df[
        (df["renda_per_capita_analise"] > RENDA_MINIMA)
        & (df["percentual_financiamento_analise"] >= PERCENTUAL_MINIMO)
        & (df["percentual_financiamento_analise"] <= PERCENTUAL_MAXIMO)
    ].copy()

    df_final = pd.DataFrame(
        {
            "renda_per_capita": df["renda_per_capita_analise"].to_numpy(dtype="float64"),
            "percentual_financiamento": df["percentual_financiamento_analise"].to_numpy(dtype="float64"),
        }
    )

    df_final = df_final.dropna(
        subset=[
            "renda_per_capita",
            "percentual_financiamento",
        ]
    ).reset_index(drop=True)

    log(f"[OK] Contratos efetivados na base: {linhas_contratadas}")
    log(f"[OK] Contratos com renda e percentual válidos antes dos filtros: {linhas_validas_antes_filtros}")
    log(f"[OK] Contratos usados na regressão: {len(df_final)}")

    return df_final


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


# =============================================================================
# Saídas tabulares
# =============================================================================

def montar_tabela_resultados(resultados: dict) -> pd.DataFrame:
    linhas = [
        {
            "Medida": "Observações válidas",
            "Valor": formatar_inteiro(resultados["n"]),
            "Valor numérico": resultados["n"],
        },
        {
            "Medida": "Correlação de Pearson",
            "Valor": formatar_decimal(resultados["pearson_r"], 4),
            "Valor numérico": resultados["pearson_r"],
        },
        {
            "Medida": "p-valor da correlação",
            "Valor": formatar_p_valor(resultados["pearson_p_value"]),
            "Valor numérico": resultados["pearson_p_value"],
        },
        {
            "Medida": "Intercepto",
            "Valor": formatar_decimal(resultados["intercepto"], 2),
            "Valor numérico": resultados["intercepto"],
        },
        {
            "Medida": "Coeficiente da renda per capita",
            "Valor": formatar_decimal(resultados["beta_renda"], 4),
            "Valor numérico": resultados["beta_renda"],
        },
        {
            "Medida": "Variação estimada a cada R$ 100",
            "Valor": formatar_pp(resultados["efeito_100_reais"], 2),
            "Valor numérico": resultados["efeito_100_reais"],
        },
        {
            "Medida": "R²",
            "Valor": formatar_decimal(resultados["r2"], 4),
            "Valor numérico": resultados["r2"],
        },
    ]

    return pd.DataFrame(linhas)


def salvar_tabela_latex(tabela: pd.DataFrame, caminho: Path) -> None:
    with caminho.open("w", encoding="utf-8") as file:
        file.write(
            tabela[["Medida", "Valor"]].to_latex(
                index=False,
                caption="Associação entre renda familiar per capita e percentual de financiamento concedido no FIES, 2019--2021.",
                label="tab:associacao_renda_financiamento",
                escape=True,
            )
        )


def salvar_dados_e_tabelas(df: pd.DataFrame, resultados: dict) -> list[dict]:
    TABLES_FINANCIAMENTO_DIR.mkdir(parents=True, exist_ok=True)

    tabela = montar_tabela_resultados(resultados)

    tabela_csv = TABLES_FINANCIAMENTO_DIR / "tabela_2_associacao_renda_financiamento.csv"
    tabela_latex = TABLES_FINANCIAMENTO_DIR / "tabela_2_associacao_renda_financiamento.latex"
    resultados_csv = TABLES_FINANCIAMENTO_DIR / "regressao_renda_financiamento_resultados.csv"
    dados_parquet = TABLES_FINANCIAMENTO_DIR / "dados_figura_4_renda_financiamento.parquet"
    dados_csv = TABLES_FINANCIAMENTO_DIR / "dados_figura_4_renda_financiamento.csv"

    tabela[["Medida", "Valor"]].to_csv(tabela_csv, index=False, encoding="utf-8")
    salvar_tabela_latex(tabela, tabela_latex)

    pd.DataFrame([resultados]).to_csv(resultados_csv, index=False, encoding="utf-8")
    df.to_parquet(dados_parquet, index=False)
    df.to_csv(dados_csv, index=False, encoding="utf-8")

    return [
        {
            "tipo": "tabela",
            "nome": "tabela_2_associacao_renda_financiamento",
            "csv": str(tabela_csv),
            "latex": str(tabela_latex),
        },
        {
            "tipo": "dados",
            "nome": "regressao_renda_financiamento_resultados",
            "csv": str(resultados_csv),
        },
        {
            "tipo": "dados",
            "nome": "dados_figura_4_renda_financiamento",
            "parquet": str(dados_parquet),
            "csv": str(dados_csv),
        },
    ]


# =============================================================================
# Figura 4
# =============================================================================

def plotar_regressao(df: pd.DataFrame, resultados: dict) -> dict:
    FIGURES_FINANCIAMENTO_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=400)

    ax.scatter(
        df["renda_per_capita"],
        df["percentual_financiamento"],
        s=2.0,
        alpha=0.022,
        color="#7b7b7b",
        edgecolors="none",
        rasterized=True,
    )

    renda_maxima_observada = df["renda_per_capita"].max(skipna=True)
    if pd.isna(renda_maxima_observada) or renda_maxima_observada <= 0:
        renda_maxima_observada = RENDA_GRAFICO_MAXIMO_PADRAO

    x_linha = np.linspace(0, float(renda_maxima_observada), 400)
    y_linha = resultados["intercepto"] + resultados["beta_renda"] * x_linha

    ax.plot(
        x_linha,
        y_linha,
        color="#4f4f4f",
        linewidth=2.5,
    )

    ax.set_xlim(0, float(renda_maxima_observada))
    ax.set_ylim(0, 105)

    ax.set_xlabel(
        "Renda familiar per capita (R$)",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Percentual de financiamento (%)",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.75,
        alpha=0.35,
        color="#9a9a9a",
    )

    ax.tick_params(axis="both", labelsize=12)

    texto = (
        f"r = {resultados['pearson_r']:.4f}\n"
        f"R² = {resultados['r2']:.4f}\n"
        f"n = {resultados['n']:,}"
    ).replace(",", ".")

    ax.text(
        0.975,
        0.96,
        texto,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#666666",
            "linewidth": 0.9,
            "alpha": 0.96,
        },
    )

    for lado in ["top", "right", "left", "bottom"]:
        ax.spines[lado].set_visible(True)
        ax.spines[lado].set_linewidth(1.0)
        ax.spines[lado].set_color("black")

    plt.tight_layout()

    nome_base = "figura_4_associacao_renda_percentual_financiamento"

    caminhos = {
        "pdf": FIGURES_FINANCIAMENTO_DIR / f"{nome_base}.pdf",
        "png": FIGURES_FINANCIAMENTO_DIR / f"{nome_base}.png",
    }

    fig.savefig(
        caminhos["pdf"],
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02,
    )

    fig.savefig(
        caminhos["png"],
        dpi=SAVE_DPI,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02,
    )


    plt.close(fig)

    return {
        "tipo": "figura",
        "nome": nome_base,
        "pdf": str(caminhos["pdf"]),
        "png": str(caminhos["png"]),
    }


# =============================================================================
# Tabela 2 em imagem/PDF
# =============================================================================

def plotar_tabela_resultados(resultados: dict) -> dict:
    TABLES_FINANCIAMENTO_DIR.mkdir(parents=True, exist_ok=True)

    tabela = montar_tabela_resultados(resultados)[["Medida", "Valor"]]

    fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=300)
    ax.axis("off")

    tabela_plot = ax.table(
        cellText=tabela.values,
        colLabels=tabela.columns,
        loc="center",
        cellLoc="left",
        colLoc="center",
        colWidths=[0.70, 0.30],
    )

    tabela_plot.auto_set_font_size(False)
    tabela_plot.set_fontsize(11.0)
    tabela_plot.scale(1, 1.60)

    cor_cabecalho = "#4a4a4a"
    cor_linha_1 = "#ececec"
    cor_linha_2 = "#ffffff"
    cor_borda = "#6a6a6a"
    cor_borda_externa = "#2f2f2f"

    ncols = len(tabela.columns)
    nrows = len(tabela) + 1

    for (linha, coluna), celula in tabela_plot.get_celld().items():
        celula.set_edgecolor(cor_borda)
        celula.set_linewidth(0.8)
        celula.PAD = 0.030

        if linha == 0:
            celula.set_facecolor(cor_cabecalho)
            celula.get_text().set_color("white")
            celula.get_text().set_fontweight("bold")
            celula.get_text().set_ha("center")
            celula.get_text().set_va("center")
            celula.get_text().set_fontsize(11.4)
        else:
            celula.set_facecolor(cor_linha_1 if linha % 2 == 1 else cor_linha_2)

            if coluna == 0:
                celula.get_text().set_fontweight("bold")
                celula.get_text().set_ha("left")
            else:
                celula.get_text().set_ha("center")

            celula.get_text().set_va("center")
            celula.get_text().set_color("black")

        if linha in [0, nrows - 1] or coluna in [0, ncols - 1]:
            celula.set_edgecolor(cor_borda_externa)
            celula.set_linewidth(1.05)

    plt.tight_layout()

    nome_base = "tabela_2_associacao_renda_financiamento"

    caminhos = {
        "pdf": TABLES_FINANCIAMENTO_DIR / f"{nome_base}.pdf",
        "png": TABLES_FINANCIAMENTO_DIR / f"{nome_base}.png",
    }

    fig.savefig(
        caminhos["pdf"],
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02,
    )

    fig.savefig(
        caminhos["png"],
        dpi=SAVE_DPI,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02,
    )


    plt.close(fig)

    return {
        "tipo": "tabela_imagem",
        "nome": nome_base,
        "pdf": str(caminhos["pdf"]),
        "png": str(caminhos["png"]),
    }


# =============================================================================
# Resumo e execução
# =============================================================================

def salvar_resumo(registros: list[dict]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(RESUMO_PATH, index=False, encoding="utf-8")
    log(f"[OK] Resumo salvo em: {RESUMO_PATH}")


def run() -> None:
    configurar_matplotlib()

    log("=" * 80)
    log("ARTICLE: RENDA, FINANCIAMENTO E COPARTICIPAÇÃO IMPLÍCITA")
    log("=" * 80)

    df_base = carregar_base()
    df = preparar_dados_financiamento(df_base)
    resultados = calcular_estatisticas_e_regressao(df)

    registros = []
    registros.extend(salvar_dados_e_tabelas(df, resultados))
    registros.append(plotar_regressao(df, resultados))
    registros.append(plotar_tabela_resultados(resultados))

    salvar_resumo(registros)

    log(f"[OK] Figura 4 salva em: {FIGURES_FINANCIAMENTO_DIR}")
    log(f"[OK] Tabela 2 salva em: {TABLES_FINANCIAMENTO_DIR}")
    log("Financiamento e coparticipação concluídos.")
