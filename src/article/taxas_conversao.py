from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import font_manager

from src.constants import (
    ANALYSIS_FUNIL_FLUXO_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    LOGS_DIR,
)


FIG_DPI = 180
SAVE_DPI = 400

TAXAS_ARTIGO = {
    "taxa_conversao_inscritos": {
        "numerador": "vagas_ocupadas",
        "denominador": "Inscritos_Geral",
        "subpasta": "taxa_conversao_inscritos",
        "rotulo_y": "Taxa de conversão entre inscrições (%)",
        "caption": "Taxa de conversão entre inscrições e contratos efetivados no FIES, por região e área CINE.",
        "label_latex": "tab:taxa_conversao_inscritos_dados",
    },
    "taxa_conversao_curso_priorizado": {
        "numerador": "vagas_ocupadas",
        "denominador": "Candidatos_Unicos_Geral",
        "subpasta": "taxa_conversao_curso_priorizado",
        "rotulo_y": "Taxa de conversão entre cursos priorizados (%)",
        "caption": "Taxa de conversão entre cursos priorizados pelos candidatos e contratos efetivados no FIES, por região e área CINE.",
        "label_latex": "tab:taxa_conversao_curso_priorizado_dados",
    },
}

PASTA_FIGURAS_BASE = FIGURES_DIR / "taxas_conversao"
PASTA_TABELAS = TABLES_DIR / "taxas_conversao"
RESUMO_PATH = LOGS_DIR / "article_taxas_conversao_resumo.csv"

COLUNAS_BASE = [
    "vagas_fies",
    "Inscritos_Geral",
    "inscritos_com_nota_suficiente",
    "Candidatos_Unicos_Geral",
    "candidatos_unicos_com_nota_suficiente",
    "vagas_ocupadas",
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
    "TODAS AS ÁREAS DO CINE": "Todas as Áreas do CINE",
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
    "Todas as Áreas do CINE",
]

ORDEM_SERIES = [
    "Norte",
    "Nordeste",
    "Centro-Oeste",
    "Sudeste",
    "Sul",
    "Média Nacional",
    "Média Nacional (Entre Todos os Cursos)",
]

ESTILOS_LINHAS = {
    "Norte": {
        "color": "0.18",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.4,
        "markersize": 9,
        "zorder": 4,
    },
    "Nordeste": {
        "color": "0.34",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.4,
        "markersize": 9,
        "zorder": 4,
    },
    "Centro-Oeste": {
        "color": "0.50",
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 2.4,
        "markersize": 9,
        "zorder": 4,
    },
    "Sudeste": {
        "color": "0.66",
        "linestyle": ":",
        "marker": "D",
        "linewidth": 2.7,
        "markersize": 8,
        "zorder": 4,
    },
    "Sul": {
        "color": "0.82",
        "linestyle": "-",
        "marker": "v",
        "linewidth": 2.8,
        "markersize": 9,
        "zorder": 4,
    },
    "Média Nacional": {
        "color": "0.00",
        "linestyle": (0, (7, 2)),
        "marker": "P",
        "linewidth": 4.0,
        "markersize": 11,
        "zorder": 6,
    },
    "Média Nacional (Entre Todos os Cursos)": {
        "color": "0.42",
        "linestyle": (0, (2, 1, 8, 1)),
        "marker": "X",
        "linewidth": 4.0,
        "markersize": 11,
        "zorder": 6,
    },
}


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "article_taxas_conversao.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def configurar_matplotlib() -> None:
    plt.rcParams["figure.dpi"] = FIG_DPI
    plt.rcParams["savefig.dpi"] = SAVE_DPI
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1.0

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


def divisao_segura(numerador, denominador):
    numerador = np.asarray(numerador, dtype="float64")
    denominador = np.asarray(denominador, dtype="float64")

    return np.where(
        (denominador != 0) & (~np.isnan(denominador)),
        numerador / denominador,
        np.nan,
    )


def formatar_percentual(valor, pos):
    if pd.isna(valor):
        return ""

    if abs(valor) >= 100:
        return f"{valor:,.0f}".replace(",", ".")

    return f"{valor:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def carregar_funil() -> pd.DataFrame:
    if not ANALYSIS_FUNIL_FLUXO_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ANALYSIS_FUNIL_FLUXO_PATH}. "
            "Rode primeiro: python3 main.py analysis funil"
        )

    df = pd.read_parquet(ANALYSIS_FUNIL_FLUXO_PATH)

    if "regiao_ies" not in df.columns:
        if "regiao" in df.columns:
            df = df.rename(columns={"regiao": "regiao_ies"})
        elif "regiao_ies_alvo" in df.columns:
            df = df.rename(columns={"regiao_ies_alvo": "regiao_ies"})

    if "regiao_ies" not in df.columns:
        raise ValueError("O funil precisa conter regiao_ies, regiao_ies_alvo ou regiao.")

    if "nome_cine_area_geral" not in df.columns:
        raise ValueError("O funil precisa conter nome_cine_area_geral.")

    faltantes = [col for col in COLUNAS_BASE if col not in df.columns]
    if faltantes:
        raise ValueError(f"O funil não contém as colunas necessárias: {faltantes}")

    df = df.copy()
    df["regiao_ies"] = df["regiao_ies"].fillna("Região não informada").astype("string")
    df["nome_cine_area_geral"] = df["nome_cine_area_geral"].map(normalizar_area_exibicao).astype("string")

    for col in COLUNAS_BASE:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def calcular_taxas(df_base):
    df_saida = df_base.copy()

    for taxa, cfg in TAXAS_ARTIGO.items():
        df_saida[taxa] = divisao_segura(
            df_saida[cfg["numerador"]],
            df_saida[cfg["denominador"]],
        )
        df_saida[taxa] = df_saida[taxa] * 100

    df_saida = df_saida.replace([np.inf, -np.inf], np.nan)

    return df_saida


def preparar_dados() -> pd.DataFrame:
    df = carregar_funil()

    df_agg = (
        df.groupby(
            ["ano", "semestre", "regiao_ies", "nome_cine_area_geral"],
            as_index=False,
            observed=True,
        )[COLUNAS_BASE]
        .sum()
    )

    df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
    df_plot = calcular_taxas(df_agg)

    df_plot["periodo"] = (
        "'" + df_plot["ano"].astype(str).str[-2:] + "." + df_plot["semestre"].astype(str)
    )

    df_nacional_base = (
        df_agg.groupby(
            ["ano", "semestre", "nome_cine_area_geral"],
            as_index=False,
            observed=True,
        )[COLUNAS_BASE]
        .sum()
    )

    df_nacional = calcular_taxas(df_nacional_base)
    df_nacional["regiao_ies"] = "Média Nacional"
    df_nacional["periodo"] = (
        "'" + df_nacional["ano"].astype(str).str[-2:] + "." + df_nacional["semestre"].astype(str)
    )

    df_total_base = (
        df_agg.groupby(
            ["ano", "semestre"],
            as_index=False,
            observed=True,
        )[COLUNAS_BASE]
        .sum()
    )

    df_total = calcular_taxas(df_total_base)
    df_total["regiao_ies"] = "Média Nacional (Entre Todos os Cursos)"
    df_total["nome_cine_area_geral"] = "Todas as Áreas do CINE"
    df_total["periodo"] = (
        "'" + df_total["ano"].astype(str).str[-2:] + "." + df_total["semestre"].astype(str)
    )

    df_completo = pd.concat([df_plot, df_nacional, df_total], ignore_index=True)
    df_completo["nome_cine_area_geral"] = df_completo["nome_cine_area_geral"].map(normalizar_area_exibicao)

    return df_completo


def ordenar_areas(df: pd.DataFrame) -> list[str]:
    presentes = df["nome_cine_area_geral"].dropna().unique().tolist()

    areas = [area for area in ORDEM_AREAS_CINE if area in presentes]
    areas.extend(sorted([area for area in presentes if area not in areas]))

    return areas


def salvar_figura(fig, pasta: Path, nome_arquivo: str) -> dict:
    pasta.mkdir(parents=True, exist_ok=True)

    nome_limpo = limpar_nome_arquivo(nome_arquivo)

    caminhos = {
        "pdf": pasta / f"{nome_limpo}.pdf",
        "png": pasta / f"{nome_limpo}.png",
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


def plotar_taxa(
    df_completo: pd.DataFrame,
    taxa: str,
    area: str,
    periodos_ordem: list[str],
    mapa_periodos: dict[str, int],
) -> dict:
    cfg = TAXAS_ARTIGO[taxa]
    pasta_taxa = PASTA_FIGURAS_BASE / cfg["subpasta"]
    pasta_taxa.mkdir(parents=True, exist_ok=True)

    df_area = df_completo[
        df_completo["nome_cine_area_geral"].isin([area, "Todas as Áreas do CINE"])
    ].copy()

    series_presentes = [
        serie
        for serie in ORDEM_SERIES
        if serie in df_area["regiao_ies"].dropna().unique().tolist()
    ]

    fig = plt.figure(figsize=(18, 11), dpi=FIG_DPI)

    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[0.18, 0.82],
        hspace=0.0,
    )

    ax_legenda = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    ax_legenda.axis("off")

    for serie in series_presentes:
        df_serie = (
            df_area[df_area["regiao_ies"] == serie]
            .dropna(subset=[taxa])
            .sort_values(["ano", "semestre"], kind="mergesort")
            .copy()
        )

        if df_serie.empty:
            continue

        estilo = ESTILOS_LINHAS.get(
            serie,
            {
                "color": "0.50",
                "linestyle": "-",
                "marker": "o",
                "linewidth": 2.5,
                "markersize": 8,
                "zorder": 4,
            },
        )

        x = df_serie["periodo"].map(mapa_periodos)
        y = df_serie[taxa]

        ax.plot(
            x,
            y,
            label=serie,
            color=estilo["color"],
            linestyle=estilo["linestyle"],
            marker=estilo["marker"],
            linewidth=estilo["linewidth"],
            markersize=estilo["markersize"],
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=estilo["zorder"],
        )

    ax.set_xlabel(
        "Semestre letivo",
        fontsize=24,
        fontweight="bold",
        labelpad=16,
        color="black",
    )

    ax.set_ylabel(
        cfg["rotulo_y"],
        fontsize=24,
        fontweight="bold",
        labelpad=16,
        color="black",
    )

    ax.set_xticks(range(len(periodos_ordem)))
    ax.set_xticklabels(
        periodos_ordem,
        fontsize=20,
        color="black",
    )

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatar_percentual))
    ax.tick_params(axis="y", labelsize=20, colors="black")
    ax.tick_params(axis="x", colors="black", pad=6)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.45,
        color="0.65",
        zorder=0,
    )

    ax.set_xlim(-0.15, len(periodos_ordem) - 0.85)

    for lado in ["top", "right", "left", "bottom"]:
        ax.spines[lado].set_visible(True)
        ax.spines[lado].set_linewidth(1.1)
        ax.spines[lado].set_color("black")

    handles_legenda = []

    for serie in series_presentes:
        estilo = ESTILOS_LINHAS[serie]

        handles_legenda.append(
            Line2D(
                [0],
                [0],
                color=estilo["color"],
                linestyle=estilo["linestyle"],
                marker=estilo["marker"],
                linewidth=estilo["linewidth"],
                markersize=estilo["markersize"],
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                label=serie,
            )
        )

    if len(handles_legenda) <= 3:
        ncol_legenda = len(handles_legenda)
        legenda_fontsize = 20
    else:
        ncol_legenda = 3
        legenda_fontsize = 20

    if handles_legenda:
        leg = ax_legenda.legend(
            handles=handles_legenda,
            loc="center",
            bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
            mode="expand",
            ncol=ncol_legenda,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            facecolor="white",
            framealpha=1.0,
            fontsize=legenda_fontsize,
            handlelength=3.6,
            handletextpad=0.8,
            columnspacing=1.4,
            labelspacing=0.45,
            borderaxespad=0.0,
            borderpad=0.30,
        )

        leg.get_frame().set_linewidth(1.1)

    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        top=0.992,
        bottom=0.105,
    )

    nome_area = limpar_nome_arquivo(area)

    return salvar_figura(
        fig=fig,
        pasta=pasta_taxa,
        nome_arquivo=f"grafico_{taxa}_{nome_area}",
    )


def salvar_tabelas(df_completo: pd.DataFrame) -> list[dict]:
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)

    saidas = []

    for taxa, cfg in TAXAS_ARTIGO.items():
        colunas = [
            "ano",
            "semestre",
            "periodo",
            "regiao_ies",
            "nome_cine_area_geral",
            cfg["numerador"],
            cfg["denominador"],
            taxa,
        ]

        tabela = df_completo[colunas].copy()
        tabela = tabela.sort_values(
            ["nome_cine_area_geral", "regiao_ies", "ano", "semestre"],
            kind="mergesort",
        )

        csv_path = PASTA_TABELAS / f"{taxa}_dados.csv"
        tex_path = PASTA_TABELAS / f"{taxa}_dados.tex"

        tabela.to_csv(csv_path, index=False, encoding="utf-8")

        tabela_tex = tabela.copy()
        tabela_tex[taxa] = tabela_tex[taxa].round(2)

        with tex_path.open("w", encoding="utf-8") as file:
            file.write(
                tabela_tex.to_latex(
                    index=False,
                    longtable=True,
                    caption=cfg["caption"],
                    label=cfg["label_latex"],
                    escape=True,
                )
            )

        saidas.append(
            {
                "tipo": "tabela",
                "taxa": taxa,
                "csv": str(csv_path),
                "tex": str(tex_path),
            }
        )

        log(f"[OK] CSV salvo em: {csv_path}")
        log(f"[OK] LaTeX salvo em: {tex_path}")

    return saidas


def salvar_resumo(saidas: list[dict], df_completo: pd.DataFrame) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    registros = list(saidas)

    for taxa in TAXAS_ARTIGO:
        registros.append(
            {
                "tipo": "resumo_taxa",
                "taxa": taxa,
                "linhas": len(df_completo),
                "media": df_completo[taxa].mean(skipna=True),
                "min": df_completo[taxa].min(skipna=True),
                "max": df_completo[taxa].max(skipna=True),
            }
        )

    pd.DataFrame(registros).to_csv(RESUMO_PATH, index=False, encoding="utf-8")
    log(f"[OK] Resumo salvo em: {RESUMO_PATH}")


def run() -> None:
    configurar_matplotlib()

    log("=" * 80)
    log("ARTICLE: TAXAS DE CONVERSÃO")
    log("=" * 80)

    df_completo = preparar_dados()

    periodos_ordem = (
        df_completo[["ano", "semestre", "periodo"]]
        .drop_duplicates()
        .sort_values(["ano", "semestre"], kind="mergesort")["periodo"]
        .tolist()
    )

    mapa_periodos = {periodo: i for i, periodo in enumerate(periodos_ordem)}
    areas = ordenar_areas(df_completo)

    log(f"[OK] Dados preparados | linhas: {len(df_completo)}")
    log(f"[OK] Áreas: {len(areas)}")
    log(f"[OK] Taxas: {list(TAXAS_ARTIGO.keys())}")

    saidas = []

    for taxa in TAXAS_ARTIGO:
        for area in areas:
            saida = plotar_taxa(
                df_completo=df_completo,
                taxa=taxa,
                area=area,
                periodos_ordem=periodos_ordem,
                mapa_periodos=mapa_periodos,
            )
            saida["tipo"] = "figura"
            saida["taxa"] = taxa
            saidas.append(saida)

    saidas.extend(salvar_tabelas(df_completo))
    salvar_resumo(saidas, df_completo)

    log(f"[OK] Figuras salvas em: {PASTA_FIGURAS_BASE}")
    log("Article taxas_conversao concluído.")
