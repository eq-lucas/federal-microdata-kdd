from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from src.constants import (
    CURATED_INSCRICOES_ARTIGO_PATH,
    FIGURES_DIR,
    APPENDIX_DIR,
    TABLES_DIR,
    LOGS_DIR,
)


# =============================================================================
# Saídas
# =============================================================================

FIGURES_HEATMAP_CONTRATADOS_DIR = FIGURES_DIR / "heatmap_contratados_x_nao_contratados"
APPENDIX_HEATMAP_LISTA_DIR = APPENDIX_DIR / "apendice_b" / "heatmap_lista_espera"
TABLES_MATRIZES_DIR = TABLES_DIR / "matrizes_renda_desempenho"

RESUMO_PATH = LOGS_DIR / "article_matrizes_renda_desempenho_resumo.csv"


# =============================================================================
# Configurações gerais
# =============================================================================

FIG_DPI = 300
SAVE_DPI = 1200

FAIXAS_RENDA_BINS = [-np.inf, 600, 1200, 1800, 2400, 3000, np.inf]

FAIXAS_RENDA_LABELS_FULL = [
    "0–600",
    "601–1.200",
    "1.201–1.800",
    "1.801–2.400",
    "2.401–3.000",
    "> 3.000",
]

FAIXAS_RENDA_LABELS_PLOT = ["I", "II", "III", "IV", "V", "VI"]

FAIXAS_GAP_BINS = [-np.inf, -150, -50, 0, 50, 150, np.inf]

FAIXAS_GAP_LABELS = [
    "< -150",
    "[-150, -50]",
    "[-50, 0]",
    "[0, +50]",
    "[+50, +150]",
    "> +150",
]

ORDEM_GAP_PLOT = FAIXAS_GAP_LABELS[::-1]

STATUS_CONTRATADA = "CONTRATADA"
STATUS_NAO_CONTRATADO = "NÃO CONTRATADO"
STATUS_LISTA_ESPERA = "LISTA DE ESPERA"

POSSIVEIS_COLUNAS_RENDA = [
    "renda_per_capita",
    "renda_familiar_per_capita",
    "renda_mensal_bruta_per_capita",
    "vl_renda_per_capita",
    "valor_renda_per_capita",
    "renda_familiar_mensal_per_capita",
]

POSSIVEIS_COLUNAS_MEDIA_ENEM = [
    "media_enem",
    "media_nota_enem",
]

POSSIVEIS_COLUNAS_NOTA_CORTE = [
    "nota_corte_gp",
    "nota_corte_grupo_preferencia",
]

POSSIVEIS_COLUNAS_SITUACAO = [
    "situacao_fies",
    "situacao_inscricao_fies",
    "situacao_inscricao",
    "situacao",
]

POSSIVEIS_COLUNAS_OPCAO = [
    "opcao_curso",
    "opcoes_cursos_inscricao",
]

POSSIVEIS_COLUNAS_AREA_CINE = [
    "nome_cine_area_geral",
    "area_cine",
]


# =============================================================================
# Logging e utilitários
# =============================================================================

def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "article_matrizes_renda_desempenho.log"

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
        return "OUTROS"

    texto_sem_acento = remover_acentos(texto).upper()

    mapa = {
        "CONTRATADA": STATUS_CONTRATADA,
        "INSCRICAO POSTERGADA": "INSCRIÇÃO POSTERGADA",
        "PRE-SELECIONADO": "PRÉ-SELECIONADO",
        "NAO CONTRATADO": STATUS_NAO_CONTRATADO,
        "REJEITADA PELA CPSA": "REJEITADA PELA CPSA",
        "OPCAO NAO CONTRATADA": "OPÇÃO NÃO CONTRATADA",
        "PARTICIPACAO CANCELADA PELO CANDIDATO": "PARTICIPAÇÃO CANCELADA",
        "PARTICIPACAO CANCELADA": "PARTICIPAÇÃO CANCELADA",
        "LISTA DE ESPERA": STATUS_LISTA_ESPERA,
    }

    return mapa.get(texto_sem_acento, "OUTROS")


def encontrar_coluna(df: pd.DataFrame, candidatas: list[str], nome_logico: str) -> str:
    for coluna in candidatas:
        if coluna in df.columns:
            return coluna

    raise ValueError(
        f"Não encontrei a coluna de {nome_logico}. "
        f"Tentei: {candidatas}. "
        f"Colunas disponíveis: {list(df.columns)}"
    )


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
    sns.set_theme(style="white", font_scale=1.0)

    plt.rcParams["font.family"] = obter_fonte_padrao()
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.dpi"] = FIG_DPI
    plt.rcParams["savefig.dpi"] = SAVE_DPI


def limpar_nome_arquivo(texto) -> str:
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    texto = re.sub(r"_+", "_", texto)

    return texto.strip("_")


def criar_colormap_cinza():
    return LinearSegmentedColormap.from_list(
        "cinza_contraste_suave",
        [
            "#f9f9f9",
            "#e3e3e3",
            "#c2c2c2",
            "#919191",
            "#6a6a6a",
        ],
        N=256,
    )


def formatar_anotacao(valor) -> str:
    return f"{valor:.1f}".replace(".", ",")


def ajustar_cor_textos_heatmap(ax, limiar_valor: float) -> None:
    for texto in ax.texts:
        try:
            valor = float(texto.get_text().replace(",", "."))
        except ValueError:
            continue

        texto.set_color("white" if valor >= limiar_valor else "black")
        texto.set_fontweight("bold")


def salvar_figura(fig, pasta: Path, nome_base: str) -> dict:
    pasta.mkdir(parents=True, exist_ok=True)

    nome_base = limpar_nome_arquivo(nome_base)

    caminhos = {
        "pdf": pasta / f"{nome_base}.pdf",
        "png": pasta / f"{nome_base}.png",
    }

    fig.savefig(
        caminhos["pdf"],
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )

    fig.savefig(
        caminhos["png"],
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )


    plt.close(fig)

    return {
        "nome": nome_base,
        "pdf": str(caminhos["pdf"]),
        "png": str(caminhos["png"]),
    }


# =============================================================================
# Preparação dos dados
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


def preparar_base_matrizes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    coluna_renda = encontrar_coluna(df, POSSIVEIS_COLUNAS_RENDA, "renda per capita")
    coluna_media = encontrar_coluna(df, POSSIVEIS_COLUNAS_MEDIA_ENEM, "média do Enem")
    coluna_corte = encontrar_coluna(df, POSSIVEIS_COLUNAS_NOTA_CORTE, "nota de corte")
    coluna_situacao = encontrar_coluna(df, POSSIVEIS_COLUNAS_SITUACAO, "situação da inscrição")
    coluna_area = encontrar_coluna(df, POSSIVEIS_COLUNAS_AREA_CINE, "área CINE")

    df[coluna_renda] = pd.to_numeric(df[coluna_renda], errors="coerce")
    df[coluna_media] = pd.to_numeric(df[coluna_media], errors="coerce")
    df[coluna_corte] = pd.to_numeric(df[coluna_corte], errors="coerce")

    linhas_antes_drop = len(df)

    df = df.dropna(subset=[coluna_renda, coluna_media, coluna_corte]).copy()

    linhas_dropadas = linhas_antes_drop - len(df)

    df["nome_cine_area_geral"] = (
        df[coluna_area]
        .astype("string")
        .fillna("CINE NÃO INFORMADO")
        .str.strip()
    )

    df["nome_cine_area_geral"] = df["nome_cine_area_geral"].replace(
        {"": "CINE NÃO INFORMADO"}
    )

    df["faixa_renda_bruta"] = pd.cut(
        df[coluna_renda],
        bins=FAIXAS_RENDA_BINS,
        labels=FAIXAS_RENDA_LABELS_FULL,
        ordered=True,
    )

    df["gap_nota"] = df[coluna_media] - df[coluna_corte]

    df["nivel_nota_gap"] = pd.cut(
        df["gap_nota"],
        bins=FAIXAS_GAP_BINS,
        labels=FAIXAS_GAP_LABELS,
        ordered=True,
    )

    df["status_final"] = df[coluna_situacao].map(normalizar_status)

    df = df.dropna(subset=["faixa_renda_bruta", "nivel_nota_gap"]).copy()

    log(f"[OK] Base de inscrições para matrizes | linhas: {len(df)}")
    log(f"[INFO] Linhas removidas por renda/nota/corte ausente: {linhas_dropadas}")

    nao_mapeados = int((df["status_final"] == "OUTROS").sum())

    if nao_mapeados > 0:
        log(f"[AVISO] Situações não mapeadas classificadas como OUTROS: {nao_mapeados}")

    return df


def gerar_decomposicao_por_area(df: pd.DataFrame) -> pd.DataFrame:
    df_counts = (
        df
        .groupby(
            [
                "nome_cine_area_geral",
                "faixa_renda_bruta",
                "nivel_nota_gap",
                "status_final",
            ],
            observed=True,
            dropna=False,
        )
        .size()
        .reset_index(name="qtd")
    )

    df_totals = (
        df
        .groupby(
            [
                "nome_cine_area_geral",
                "faixa_renda_bruta",
                "nivel_nota_gap",
            ],
            observed=True,
            dropna=False,
        )
        .size()
        .reset_index(name="total_celula")
    )

    df_analise = df_counts.merge(
        df_totals,
        on=["nome_cine_area_geral", "faixa_renda_bruta", "nivel_nota_gap"],
        how="left",
        validate="m:1",
    )

    df_analise["percentual_celula"] = (
        df_analise["qtd"] / df_analise["total_celula"] * 100
    )

    return df_analise


def gerar_decomposicao_nacional(df: pd.DataFrame) -> pd.DataFrame:
    df_counts = (
        df
        .groupby(
            [
                "faixa_renda_bruta",
                "nivel_nota_gap",
                "status_final",
            ],
            observed=True,
            dropna=False,
        )
        .size()
        .reset_index(name="qtd")
    )

    df_totals = (
        df
        .groupby(
            [
                "faixa_renda_bruta",
                "nivel_nota_gap",
            ],
            observed=True,
            dropna=False,
        )
        .size()
        .reset_index(name="total_celula")
    )

    df_analise = df_counts.merge(
        df_totals,
        on=["faixa_renda_bruta", "nivel_nota_gap"],
        how="left",
        validate="m:1",
    )

    df_analise["percentual_celula"] = (
        df_analise["qtd"] / df_analise["total_celula"] * 100
    )

    return df_analise


def auditar_percentuais(df_analise: pd.DataFrame, chaves: list[str], nome: str) -> dict:
    check = (
        df_analise
        .groupby(chaves, observed=True, dropna=False)["percentual_celula"]
        .sum()
        .reset_index()
    )

    erros = check[~np.isclose(check["percentual_celula"], 100.0, atol=0.01)]

    registro = {
        "base": nome,
        "celulas": len(check),
        "celulas_com_erro_percentual": len(erros),
    }

    if len(erros) > 0:
        log(f"[AVISO] {nome}: células com soma percentual diferente de 100%: {len(erros)}")
    else:
        log(f"[OK] {nome}: todas as células somam 100%")

    return registro


def matriz_status(dados: pd.DataFrame, status: str) -> pd.DataFrame:
    df_status = dados[dados["status_final"] == status].copy()

    if df_status.empty:
        return pd.DataFrame(
            0.0,
            index=ORDEM_GAP_PLOT,
            columns=FAIXAS_RENDA_LABELS_FULL,
        )

    matriz = df_status.pivot_table(
        index="nivel_nota_gap",
        columns="faixa_renda_bruta",
        values="percentual_celula",
        aggfunc="sum",
        observed=True,
    )

    matriz = (
        matriz
        .reindex(index=ORDEM_GAP_PLOT, columns=FAIXAS_RENDA_LABELS_FULL)
        .fillna(0.0)
    )

    return matriz


# =============================================================================
# Gráficos
# =============================================================================

def plotar_contratados_nao_contratados(titulo: str, dados: pd.DataFrame) -> dict:
    cmap = criar_colormap_cinza()
    norma = PowerNorm(gamma=0.58, vmin=0, vmax=100)

    status_paineis = [
        (STATUS_CONTRATADA, "Contratados"),
        (STATUS_NAO_CONTRATADO, "Não contratados"),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(22.4, 8.7),
        dpi=450,
        sharey=True,
        gridspec_kw={
            "wspace": 0.025,
            "width_ratios": [1, 1],
        },
    )

    for i, (status, titulo_status) in enumerate(status_paineis):
        ax = axes[i]

        matriz = matriz_status(dados, status)
        matriz_annot = matriz.map(formatar_anotacao)

        sns.heatmap(
            matriz,
            annot=matriz_annot,
            fmt="",
            cmap=cmap,
            norm=norma,
            linewidths=0.45,
            linecolor="#bebebe",
            annot_kws={
                "size": 33,
                "weight": "bold",
            },
            cbar=False,
            square=False,
            ax=ax,
        )

        ajustar_cor_textos_heatmap(ax, limiar_valor=56)

        ax.set_title("")

        ax.set_xlabel(
            "Faixa de renda familiar per capita",
            fontweight="bold",
            fontsize=27,
            color="black",
            labelpad=8,
        )

        ax.set_xticklabels(
            FAIXAS_RENDA_LABELS_PLOT,
            rotation=0,
            fontsize=25,
            color="black",
        )

        ax.tick_params(axis="x", colors="black", pad=1)

        if i == 0:
            ax.set_ylabel(
                "Desempenho relativo à nota de corte",
                fontweight="bold",
                fontsize=27,
                color="black",
                labelpad=8,
            )

            ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=0,
                fontsize=24,
                color="black",
            )

            ax.tick_params(axis="y", colors="black", pad=1)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

        for lado in ["top", "right", "left", "bottom"]:
            ax.spines[lado].set_visible(True)
            ax.spines[lado].set_linewidth(1.0)
            ax.spines[lado].set_color("black")

    plt.subplots_adjust(
        left=0.07,
        right=0.997,
        bottom=0.10,
        top=0.996,
        wspace=0.025,
    )

    return salvar_figura(
        fig=fig,
        pasta=FIGURES_HEATMAP_CONTRATADOS_DIR,
        nome_base=f"{titulo}_contratados_nao_contratados",
    )


def plotar_lista_espera(dados_nacionais: pd.DataFrame) -> dict:
    matriz = matriz_status(dados_nacionais, STATUS_LISTA_ESPERA)

    vmax_real = float(np.nanmax(matriz.to_numpy())) if matriz.size > 0 else 0.0
    vmax_usado = max(10, min(100, np.ceil(vmax_real / 5) * 5))

    cmap = criar_colormap_cinza()
    norma = PowerNorm(gamma=0.58, vmin=0, vmax=vmax_usado)

    matriz_annot = matriz.map(formatar_anotacao)

    fig, ax = plt.subplots(
        figsize=(11.4, 8.8),
        dpi=450,
    )

    sns.heatmap(
        matriz,
        annot=matriz_annot,
        fmt="",
        cmap=cmap,
        norm=norma,
        linewidths=0.45,
        linecolor="#bebebe",
        annot_kws={
            "size": 31,
            "weight": "bold",
        },
        cbar=False,
        square=False,
        ax=ax,
    )

    ajustar_cor_textos_heatmap(ax, limiar_valor=vmax_usado * 0.58)

    ax.set_title("")

    ax.set_xlabel(
        "Faixa de renda familiar per capita",
        fontweight="bold",
        fontsize=27,
        color="black",
        labelpad=8,
    )

    ax.set_ylabel(
        "Desempenho relativo à nota de corte",
        fontweight="bold",
        fontsize=27,
        color="black",
        labelpad=8,
    )

    ax.set_xticklabels(
        FAIXAS_RENDA_LABELS_PLOT,
        rotation=0,
        fontsize=25,
        color="black",
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=24,
        color="black",
    )

    ax.tick_params(axis="x", colors="black", pad=1)
    ax.tick_params(axis="y", colors="black", pad=1)

    for lado in ["top", "right", "left", "bottom"]:
        ax.spines[lado].set_visible(True)
        ax.spines[lado].set_linewidth(1.0)
        ax.spines[lado].set_color("black")

    plt.subplots_adjust(
        left=0.16,
        right=0.995,
        bottom=0.12,
        top=0.995,
    )

    return salvar_figura(
        fig=fig,
        pasta=APPENDIX_HEATMAP_LISTA_DIR,
        nome_base="figura_b1_lista_espera",
    )


# =============================================================================
# Persistência de dados auxiliares
# =============================================================================

def salvar_dados_auxiliares(
    df_area: pd.DataFrame,
    df_nacional: pd.DataFrame,
) -> list[dict]:
    TABLES_MATRIZES_DIR.mkdir(parents=True, exist_ok=True)

    parquet_area = TABLES_MATRIZES_DIR / "matriz_status_por_area_cine.parquet"
    csv_area = TABLES_MATRIZES_DIR / "matriz_status_por_area_cine.csv"

    parquet_nacional = TABLES_MATRIZES_DIR / "matriz_status_nacional.parquet"
    csv_nacional = TABLES_MATRIZES_DIR / "matriz_status_nacional.csv"

    df_area.to_parquet(parquet_area, index=False)
    df_area.to_csv(csv_area, index=False, encoding="utf-8")

    df_nacional.to_parquet(parquet_nacional, index=False)
    df_nacional.to_csv(csv_nacional, index=False, encoding="utf-8")

    return [
        {"tipo": "dados", "nome": "matriz_status_por_area_cine", "parquet": str(parquet_area), "csv": str(csv_area)},
        {"tipo": "dados", "nome": "matriz_status_nacional", "parquet": str(parquet_nacional), "csv": str(csv_nacional)},
    ]


def salvar_resumo(registros: list[dict]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(RESUMO_PATH, index=False, encoding="utf-8")
    log(f"[OK] Resumo salvo em: {RESUMO_PATH}")


# =============================================================================
# Execução
# =============================================================================

def run() -> None:
    configurar_matplotlib()

    log("=" * 80)
    log("ARTICLE: MATRIZES RENDA × DESEMPENHO")
    log("=" * 80)

    df = carregar_base()
    df = preparar_base_matrizes(df)

    df_area = gerar_decomposicao_por_area(df)
    df_nacional = gerar_decomposicao_nacional(df)

    registros = []

    registros.append(
        auditar_percentuais(
            df_area,
            ["nome_cine_area_geral", "faixa_renda_bruta", "nivel_nota_gap"],
            "matriz_status_por_area_cine",
        )
    )

    registros.append(
        auditar_percentuais(
            df_nacional,
            ["faixa_renda_bruta", "nivel_nota_gap"],
            "matriz_status_nacional",
        )
    )

    registros.extend(salvar_dados_auxiliares(df_area, df_nacional))

    registros.append(
        {
            "tipo": "figura",
            "recorte": "nacional",
            "produto": "contratados_x_nao_contratados",
            **plotar_contratados_nao_contratados("nacional", df_nacional),
        }
    )

    areas = sorted(df_area["nome_cine_area_geral"].dropna().unique().tolist())

    for area in areas:
        log(f"[INFO] Gerando heatmap por área CINE: {area}")

        registros.append(
            {
                "tipo": "figura",
                "recorte": area,
                "produto": "contratados_x_nao_contratados",
                **plotar_contratados_nao_contratados(
                    titulo=area,
                    dados=df_area[df_area["nome_cine_area_geral"] == area],
                ),
            }
        )

    registros.append(
        {
            "tipo": "figura",
            "recorte": "nacional",
            "produto": "lista_espera",
            **plotar_lista_espera(df_nacional),
        }
    )

    salvar_resumo(registros)

    log(f"[OK] Heatmaps contratados/não contratados: {FIGURES_HEATMAP_CONTRATADOS_DIR}")
    log(f"[OK] Heatmap lista de espera: {APPENDIX_HEATMAP_LISTA_DIR}")
    log("Matrizes renda × desempenho concluídas.")
