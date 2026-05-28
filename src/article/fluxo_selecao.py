from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib import font_manager

from src.constants import (
    ANALYSIS_FUNIL_FLUXO_PATH,
    FIGURES_DIR,
    LOGS_DIR,
)


FIG_DPI = 180
SAVE_DPI = 400

PASTA_SAIDA = FIGURES_DIR / "fluxo_selecao" / "funil"
RESUMO_PATH = LOGS_DIR / "article_fluxo_selecao_resumo.csv"

COLUNAS_INSCRITOS = [
    "vagas_fies",
    "Inscritos_Geral",
    "inscritos_com_nota_suficiente",
    "vagas_ocupadas",
]

COLUNAS_CANDIDATOS = [
    "vagas_fies",
    "Candidatos_Unicos_Geral",
    "candidatos_unicos_com_nota_suficiente",
    "vagas_ocupadas",
]

NOMES_CURTO = {
    "vagas_fies": "I",
    "Inscritos_Geral": "II",
    "inscritos_com_nota_suficiente": "III",
    "Candidatos_Unicos_Geral": "II",
    "candidatos_unicos_com_nota_suficiente": "III",
    "vagas_ocupadas": "IV",
}

ORDEM_AREAS_CINE = [
    "Saúde e bem-estar",
    "Negócios, administração e direito",
    "Engenharia, produção e construção",
    "Ciências sociais, comunicação e informação",
    "Agricultura, silvicultura, pesca e veterinária",
    "Educação",
    "Computação e Tecnologias da Informação e Comunicação (TIC)",
    "Computação e Tecnologias da Informação e Comunicação",
    "Serviços",
    "Artes e humanidades",
    "Ciências naturais, matemática e estatística",
    "CINE Não Informado (MEC)",
]

MAPA_AREA_EXIBICAO = {
    "SAÚDE E BEM-ESTAR": "Saúde e bem-estar",
    "NEGÓCIOS, ADMINISTRAÇÃO E DIREITO": "Negócios, administração e direito",
    "ENGENHARIA, PRODUÇÃO E CONSTRUÇÃO": "Engenharia, produção e construção",
    "CIÊNCIAS SOCIAIS, COMUNICAÇÃO E INFORMAÇÃO": "Ciências sociais, comunicação e informação",
    "AGRICULTURA, SILVICULTURA, PESCA E VETERINÁRIA": "Agricultura, silvicultura, pesca e veterinária",
    "EDUCAÇÃO": "Educação",
    "COMPUTAÇÃO E TECNOLOGIAS DA INFORMAÇÃO E COMUNICAÇÃO (TIC)": "Computação e Tecnologias da Informação e Comunicação (TIC)",
    "COMPUTAÇÃO E TECNOLOGIAS DA INFORMAÇÃO E COMUNICAÇÃO": "Computação e Tecnologias da Informação e Comunicação",
    "SERVIÇOS": "Serviços",
    "ARTES E HUMANIDADES": "Artes e humanidades",
    "CIÊNCIAS NATURAIS, MATEMÁTICA E ESTATÍSTICA": "Ciências naturais, matemática e estatística",
    "CINE NÃO INFORMADO (MEC)": "CINE Não Informado (MEC)",
}


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "article_fluxo_selecao.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def configurar_matplotlib() -> None:
    plt.rcParams["figure.dpi"] = FIG_DPI
    plt.rcParams["savefig.dpi"] = SAVE_DPI
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["hatch.linewidth"] = 0.65

    fontes_disponiveis = {f.name for f in font_manager.fontManager.ttflist}

    if "Times New Roman" in fontes_disponiveis:
        fonte_padrao = "Times New Roman"
    elif "Liberation Serif" in fontes_disponiveis:
        fonte_padrao = "Liberation Serif"
    elif "Nimbus Roman" in fontes_disponiveis:
        fonte_padrao = "Nimbus Roman"
    else:
        fonte_padrao = "DejaVu Serif"

    plt.rcParams["font.family"] = fonte_padrao


def limpar_nome_arquivo(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    texto = re.sub(r"_+", "_", texto)

    return texto.strip("_")


def normalizar_area_exibicao(valor):
    if pd.isna(valor):
        return "CINE Não Informado (MEC)"

    texto = str(valor).strip()

    if texto == "":
        return "CINE Não Informado (MEC)"

    texto_upper = re.sub(r"\s+", " ", texto.upper())

    return MAPA_AREA_EXIBICAO.get(texto_upper, texto)


def estilos_monocromaticos(categorias_list):
    tons_cinza = [
        "0.22",
        "0.82",
        "0.38",
        "0.68",
        "0.52",
        "0.90",
        "0.30",
        "0.76",
        "0.46",
        "0.94",
        "0.60",
        "0.86",
    ]

    hachuras = [
        "",
        "/",
        "\\",
        "x",
        "-",
        ".",
        "|",
        "+",
        "o",
        "*",
        "//",
        "\\\\",
    ]

    estilos = {}

    for i, categoria in enumerate(categorias_list):
        estilos[categoria] = {
            "cor": tons_cinza[i % len(tons_cinza)],
            "hatch": hachuras[i % len(hachuras)],
        }

    return estilos


def carregar_funil() -> pd.DataFrame:
    if not ANALYSIS_FUNIL_FLUXO_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ANALYSIS_FUNIL_FLUXO_PATH}. "
            "Rode primeiro: python3 main.py analysis funil"
        )

    df = pd.read_parquet(ANALYSIS_FUNIL_FLUXO_PATH)

    if "regiao" not in df.columns:
        if "regiao_ies" in df.columns:
            df = df.rename(columns={"regiao_ies": "regiao"})
        elif "regiao_ies_alvo" in df.columns:
            df = df.rename(columns={"regiao_ies_alvo": "regiao"})

    if "regiao" not in df.columns:
        raise ValueError("O funil precisa conter uma coluna de região: regiao, regiao_ies ou regiao_ies_alvo.")

    if "nome_cine_area_geral" not in df.columns:
        raise ValueError("O funil precisa conter a coluna nome_cine_area_geral.")

    colunas_necessarias = set(COLUNAS_INSCRITOS + COLUNAS_CANDIDATOS)

    faltantes = [col for col in colunas_necessarias if col not in df.columns]

    if faltantes:
        raise ValueError(f"O funil não contém as colunas necessárias: {faltantes}")

    df = df.copy()
    df["regiao"] = df["regiao"].fillna("Região não informada").astype("string")
    df["nome_cine_area_geral"] = df["nome_cine_area_geral"].map(normalizar_area_exibicao).astype("string")
    df["periodo"] = df["ano"].astype(str) + "." + df["semestre"].astype(str)

    return df


def obter_areas(df: pd.DataFrame) -> list[str]:
    areas_presentes = df["nome_cine_area_geral"].dropna().unique().tolist()

    areas = [area for area in ORDEM_AREAS_CINE if area in areas_presentes]

    areas_restantes = [area for area in areas_presentes if area not in areas]
    areas.extend(sorted(areas_restantes))

    return areas


def obter_periodos(df: pd.DataFrame) -> list[str]:
    return (
        df[["ano", "semestre", "periodo"]]
        .drop_duplicates()
        .sort_values(["ano", "semestre"], kind="mergesort")["periodo"]
        .tolist()
    )


def obter_regioes(df: pd.DataFrame) -> list[str]:
    ordem = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul", "Região não informada"]
    presentes = df["regiao"].dropna().unique().tolist()
    regioes = [r for r in ordem if r in presentes]
    regioes.extend(sorted([r for r in presentes if r not in regioes]))

    return regioes


def salvar_figura(fig, nome_arquivo: str) -> dict:
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    nome_limpo = limpar_nome_arquivo(nome_arquivo)

    caminhos = {
        "pdf": PASTA_SAIDA / f"{nome_limpo}.pdf",
        "png": PASTA_SAIDA / f"{nome_limpo}.png",
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
        "nome": nome_limpo,
        "pdf": str(caminhos["pdf"]),
        "png": str(caminhos["png"]),
    }


def plot_funil_seguro(
    df_data,
    periodos_list,
    categorias_list,
    col_categoria,
    cols_metricas,
    nome_arquivo,
    figsize=(24, 32),
    y_step=None,
):
    if col_categoria is not None:
        df_agg = (
            df_data
            .groupby(["periodo", col_categoria], as_index=False, observed=True)[cols_metricas]
            .sum()
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
        df_plot = (
            df_data
            .groupby(["periodo"], as_index=False, observed=True)[cols_metricas]
            .sum()
        )

        df_plot["dummy"] = "Total"
        categorias_list = ["Total"]
        col_categoria = "dummy"

    tem_legenda = col_categoria != "dummy"

    if tem_legenda:
        if col_categoria == "nome_cine_area_geral":
            altura_legenda = 0.16
        elif col_categoria == "regiao":
            altura_legenda = 0.09
        else:
            altura_legenda = 0.11

        fig = plt.figure(figsize=figsize, dpi=FIG_DPI)

        gs = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[altura_legenda, 1.0 - altura_legenda],
            hspace=0.0,
        )

        ax_legenda = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
        ax_legenda.axis("off")

    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
        ax_legenda = None

    x = np.arange(len(cols_metricas))

    largura_barra = 0.155
    half_span = largura_barra * (len(periodos_list) - 1) / 2.0
    deslocamentos = np.linspace(-half_span, half_span, len(periodos_list))

    estilos = estilos_monocromaticos(categorias_list)

    for desloc, periodo in zip(deslocamentos, periodos_list):
        df_periodo = df_plot[df_plot["periodo"] == periodo]
        base = np.zeros(len(cols_metricas))

        for cat in categorias_list:
            linha = df_periodo[df_periodo[col_categoria] == cat]

            if linha.empty:
                valores_originais = np.zeros(len(cols_metricas))
            else:
                valores_originais = linha[cols_metricas].iloc[0].to_numpy(dtype="float64")

            valores_scaled = np.array(valores_originais, dtype="float64") / 1000.0

            ax.bar(
                x + desloc,
                valores_scaled,
                width=largura_barra,
                bottom=base,
                color=estilos[cat]["cor"],
                edgecolor="black",
                linewidth=0.65,
                hatch=estilos[cat]["hatch"],
                zorder=3,
            )

            base += valores_scaled

    posicoes_ticks = []
    labels_ticks = []

    for i in range(len(cols_metricas)):
        for desloc, periodo in zip(deslocamentos, periodos_list):
            posicoes_ticks.append(i + desloc)
            ano, sem = periodo.split(".")
            labels_ticks.append(f"'{ano[-2:]}.{sem}")

    ax.set_xticks(posicoes_ticks)

    ax.set_xticklabels(
        labels_ticks,
        rotation=90,
        ha="center",
        fontsize=26,
        fontweight="normal",
        color="black",
    )

    ax.tick_params(axis="x", colors="black", pad=4)
    ax.set_xlim(-0.5, len(cols_metricas) - 0.5)

    for i in range(1, len(cols_metricas)):
        ax.axvline(
            x=i - 0.5,
            color="0.70",
            linestyle="--",
            linewidth=1.25,
            zorder=1,
        )

    for i, col in enumerate(cols_metricas):
        ax.annotate(
            NOMES_CURTO.get(col, col),
            xy=(i, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -78),
            textcoords="offset points",
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=34,
            color="black",
            annotation_clip=False,
        )

    ax.set_ylabel(
        "Quantidade (mil)",
        fontsize=32,
        fontweight="bold",
        color="black",
        labelpad=14,
    )

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda val, pos: f"{val:,.0f}".replace(",", "."))
    )

    ax.tick_params(axis="y", labelsize=24, colors="black")

    if y_step is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.45,
        color="0.65",
        zorder=0,
    )

    ax.set_ylim(bottom=0)

    for lado in ["top", "right", "left", "bottom"]:
        ax.spines[lado].set_visible(True)
        ax.spines[lado].set_linewidth(1.0)
        ax.spines[lado].set_color("black")

    if tem_legenda:
        handles_custom = [
            Patch(
                facecolor=estilos[cat]["cor"],
                edgecolor="black",
                linewidth=1.05,
                hatch=estilos[cat]["hatch"],
                label=cat,
            )
            for cat in categorias_list
        ]

        if col_categoria == "nome_cine_area_geral":
            num_colunas = 2
            legenda_fontsize = 25
            handlelength = 4.0
            handleheight = 1.45
            columnspacing = 2.0
            labelspacing = 0.45
            handletextpad = 0.75
            borderpad = 0.25

        elif col_categoria == "regiao":
            num_colunas = len(categorias_list)
            legenda_fontsize = 27
            handlelength = 3.8
            handleheight = 1.40
            columnspacing = 1.6
            labelspacing = 0.35
            handletextpad = 0.75
            borderpad = 0.20

        else:
            num_colunas = min(4, len(categorias_list))
            legenda_fontsize = 24
            handlelength = 3.8
            handleheight = 1.35
            columnspacing = 1.6
            labelspacing = 0.35
            handletextpad = 0.75
            borderpad = 0.20

        leg = ax_legenda.legend(
            handles=handles_custom,
            loc="center",
            bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
            mode="expand",
            ncol=num_colunas,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            facecolor="white",
            framealpha=1.0,
            fontsize=legenda_fontsize,
            handlelength=handlelength,
            handleheight=handleheight,
            handletextpad=handletextpad,
            columnspacing=columnspacing,
            labelspacing=labelspacing,
            borderaxespad=0.0,
            borderpad=borderpad,
        )

        leg.get_frame().set_linewidth(1.0)

    if tem_legenda:
        fig.subplots_adjust(
            left=0.055,
            right=0.995,
            top=0.992,
            bottom=0.080,
        )
    else:
        fig.subplots_adjust(
            left=0.075,
            right=0.995,
            top=0.985,
            bottom=0.120,
        )

    return salvar_figura(fig, nome_arquivo)


def salvar_resumo(saidas: list[dict]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saidas).to_csv(RESUMO_PATH, index=False, encoding="utf-8")
    log(f"[OK] Resumo salvo em: {RESUMO_PATH}")


def run() -> None:
    configurar_matplotlib()

    log("=" * 80)
    log("ARTICLE: FUNIL DO FLUXO DE SELEÇÃO")
    log("=" * 80)

    df = carregar_funil()

    areas = obter_areas(df)
    periodos = obter_periodos(df)
    regioes = obter_regioes(df)

    log(f"[OK] Funil carregado | linhas: {len(df)} | colunas: {len(df.columns)}")
    log(f"[OK] Áreas CINE: {len(areas)}")
    log(f"[OK] Regiões: {regioes}")

    saidas = []

    saidas.append(
        plot_funil_seguro(
            df_data=df,
            periodos_list=periodos,
            categorias_list=areas,
            col_categoria="nome_cine_area_geral",
            cols_metricas=COLUNAS_CANDIDATOS,
            nome_arquivo="figura_1a_funil_fies_candidatos_unicos_area_cine",
            figsize=(24, 30),
            y_step=100,
        )
    )

    saidas.append(
        plot_funil_seguro(
            df_data=df,
            periodos_list=periodos,
            categorias_list=areas,
            col_categoria="nome_cine_area_geral",
            cols_metricas=COLUNAS_INSCRITOS,
            nome_arquivo="figura_1_funil_fies_inscritos_area_cine",
            figsize=(24, 30),
            y_step=100,
        )
    )

    saidas.append(
        plot_funil_seguro(
            df_data=df,
            periodos_list=periodos,
            categorias_list=None,
            col_categoria=None,
            cols_metricas=COLUNAS_CANDIDATOS,
            nome_arquivo="funil_fies_candidatos_unicos_global_consolidado",
            figsize=(20, 22),
            y_step=100,
        )
    )

    saidas.append(
        plot_funil_seguro(
            df_data=df,
            periodos_list=periodos,
            categorias_list=None,
            col_categoria=None,
            cols_metricas=COLUNAS_INSCRITOS,
            nome_arquivo="funil_fies_inscritos_global_consolidado",
            figsize=(20, 22),
            y_step=100,
        )
    )

    for area in areas:
        df_area = df[df["nome_cine_area_geral"] == area]
        nome_area = limpar_nome_arquivo(area)

        saidas.append(
            plot_funil_seguro(
                df_data=df_area,
                periodos_list=periodos,
                categorias_list=regioes,
                col_categoria="regiao",
                cols_metricas=COLUNAS_CANDIDATOS,
                nome_arquivo=f"funil_fies_candidatos_unicos_regiao_{nome_area}",
                figsize=(22, 24),
                y_step=None,
            )
        )

        saidas.append(
            plot_funil_seguro(
                df_data=df_area,
                periodos_list=periodos,
                categorias_list=regioes,
                col_categoria="regiao",
                cols_metricas=COLUNAS_INSCRITOS,
                nome_arquivo=f"funil_fies_inscritos_regiao_{nome_area}",
                figsize=(22, 24),
                y_step=None,
            )
        )

    saidas.append(
        plot_funil_seguro(
            df_data=df,
            periodos_list=periodos,
            categorias_list=regioes,
            col_categoria="regiao",
            cols_metricas=COLUNAS_CANDIDATOS,
            nome_arquivo="funil_fies_candidatos_unicos_regiao_total",
            figsize=(22, 24),
            y_step=100,
        )
    )

    saidas.append(
        plot_funil_seguro(
            df_data=df,
            periodos_list=periodos,
            categorias_list=regioes,
            col_categoria="regiao",
            cols_metricas=COLUNAS_INSCRITOS,
            nome_arquivo="funil_fies_inscritos_regiao_total",
            figsize=(22, 24),
            y_step=100,
        )
    )

    salvar_resumo(saidas)

    log(f"[OK] Figuras salvas em: {PASTA_SAIDA}")
    log("Article fluxo concluído.")
