# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import font_manager
from pathlib import Path
import re

# ==============================================================================
# 0. CONFIGURAÇÕES GERAIS
# ==============================================================================

FIG_DPI = 180
SAVE_DPI = 400

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

path = "../data/05_processed/funil_por_regiao.parquet"

base_dir = Path("../reports/figures/analise_021_ensaio_pb")
base_dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. FUNÇÕES AUXILIARES
# ==============================================================================

def limpar_nome_arquivo(texto):
    texto = str(texto).lower()

    substituicoes = {
        "ç": "c",
        "ã": "a",
        "á": "a",
        "à": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "ê": "e",
        "è": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "õ": "o",
        "ô": "o",
        "ò": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
    }

    for original, novo in substituicoes.items():
        texto = texto.replace(original, novo)

    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    texto = re.sub(r"_+", "_", texto)

    return texto.strip("_")


def divisao_segura(numerador, denominador):
    numerador = np.asarray(numerador, dtype="float64")
    denominador = np.asarray(denominador, dtype="float64")

    return np.where(
        (denominador != 0) & (~np.isnan(denominador)),
        numerador / denominador,
        np.nan,
    )


def calcular_taxas(df_base):
    df_saida = df_base.copy()

    df_saida["taxa_inscricao"] = divisao_segura(
        df_saida["Inscritos_Geral"],
        df_saida["vagas_fies"],
    )

    df_saida["taxa_aprovacao_por_inscritos"] = divisao_segura(
        df_saida["inscritos_com_nota_suficiente"],
        df_saida["Inscritos_Geral"],
    )

    df_saida["taxa_aprovacao_por_candidato"] = divisao_segura(
        df_saida["candidatos_unicos_com_nota_suficiente"],
        df_saida["Candidatos_Unicos_Geral"],
    )

    df_saida["taxa_ocupacao"] = divisao_segura(
        df_saida["vagas_ocupadas"],
        df_saida["vagas_fies"],
    )

    df_saida["taxa_conversao_inscritos"] = divisao_segura(
        df_saida["vagas_ocupadas"],
        df_saida["Inscritos_Geral"],
    )

    df_saida["taxa_conversao_candidatos"] = divisao_segura(
        df_saida["vagas_ocupadas"],
        df_saida["Candidatos_Unicos_Geral"],
    )

    df_saida["taxa_inscritos_capacitados"] = divisao_segura(
        df_saida["vagas_ocupadas"],
        df_saida["inscritos_com_nota_suficiente"],
    )

    df_saida["taxa_candidatos_capacitados"] = divisao_segura(
        df_saida["vagas_ocupadas"],
        df_saida["candidatos_unicos_com_nota_suficiente"],
    )

    taxas_local = [
        "taxa_inscricao",
        "taxa_aprovacao_por_inscritos",
        "taxa_aprovacao_por_candidato",
        "taxa_ocupacao",
        "taxa_conversao_inscritos",
        "taxa_conversao_candidatos",
        "taxa_inscritos_capacitados",
        "taxa_candidatos_capacitados",
    ]

    df_saida[taxas_local] = df_saida[taxas_local] * 100
    df_saida = df_saida.replace([np.inf, -np.inf], np.nan)

    return df_saida


def formatar_percentual(valor, pos):
    if pd.isna(valor):
        return ""

    if abs(valor) >= 100:
        return f"{valor:,.0f}".replace(",", ".")

    return f"{valor:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ==============================================================================
# 2. LEITURA E AGREGAÇÃO DOS DADOS
# ==============================================================================

df = pd.read_parquet(path)

if "regiao_ies" not in df.columns and "regiao" in df.columns:
    df = df.rename(columns={"regiao": "regiao_ies"})

df["regiao_ies"] = df["regiao_ies"].fillna("Região não informada")
df["nome_cine_area_geral"] = df["nome_cine_area_geral"].fillna("CINE Não Informado (MEC)")

colunas_base = [
    "vagas_fies",
    "Inscritos_Geral",
    "inscritos_com_nota_suficiente",
    "Candidatos_Unicos_Geral",
    "candidatos_unicos_com_nota_suficiente",
    "vagas_ocupadas",
]

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
        ["ano", "semestre", "nome_cine_area_geral"],
        as_index=False,
    )[colunas_base]
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
    )[colunas_base]
    .sum()
)

df_total = calcular_taxas(df_total_base)
df_total["regiao_ies"] = "Média Nacional (Entre Todos os Cursos)"
df_total["nome_cine_area_geral"] = "Todas as Áreas do CINE"
df_total["periodo"] = (
    "'" + df_total["ano"].astype(str).str[-2:] + "." + df_total["semestre"].astype(str)
)

df_completo = pd.concat([df_plot, df_nacional, df_total], ignore_index=True)

taxas = [
    "taxa_inscricao",
    "taxa_aprovacao_por_inscritos",
    "taxa_aprovacao_por_candidato",
    "taxa_ocupacao",
    "taxa_conversao_inscritos",
    "taxa_conversao_candidatos",
    "taxa_inscritos_capacitados",
    "taxa_candidatos_capacitados",
]

rotulos_taxas = {
    "taxa_inscricao": "Taxa de inscrição",
    "taxa_aprovacao_por_inscritos": "Taxa de aprovação entre inscritos",
    "taxa_aprovacao_por_candidato": "Taxa de aprovação entre candidatos",
    "taxa_ocupacao": "Taxa de ocupação",
    "taxa_conversao_inscritos": "Taxa de conversão entre inscritos",
    "taxa_conversao_candidatos": "Taxa de conversão entre candidatos",
    "taxa_inscritos_capacitados": "Taxa de contratos entre inscritos com nota suficiente",
    "taxa_candidatos_capacitados": "Taxa de contratos entre candidatos com nota suficiente",
}

periodos_ordem = (
    df_completo[["ano", "semestre", "periodo"]]
    .drop_duplicates()
    .sort_values(["ano", "semestre"])["periodo"]
    .tolist()
)

mapa_periodos = {periodo: i for i, periodo in enumerate(periodos_ordem)}

# ==============================================================================
# 4. ESTILO MONOCROMÁTICO DAS LINHAS
# ==============================================================================

ordem_series = [
    "Norte",
    "Nordeste",
    "Centro-Oeste",
    "Sudeste",
    "Sul",
    "Média Nacional",
    "Média Nacional (Entre Todos os Cursos)",
]

estilos_linhas = {
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

# ==============================================================================
# 5. GERAÇÃO DOS GRÁFICOS
# ==============================================================================

for taxa in taxas:
    taxa_dir = base_dir / taxa
    taxa_dir.mkdir(parents=True, exist_ok=True)

    areas = df_completo["nome_cine_area_geral"].dropna().unique().tolist()

    for area in areas:
        df_area = df_completo[
            df_completo["nome_cine_area_geral"].isin([area, "Todas as Áreas do CINE"])
        ].copy()

        series_presentes = [
            serie
            for serie in ordem_series
            if serie in df_area["regiao_ies"].dropna().unique().tolist()
        ]

        # ----------------------------------------------------------------------
        # Figura com área superior para legenda e área inferior para o gráfico
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # Plot das séries
        # ----------------------------------------------------------------------
        for serie in series_presentes:
            df_serie = (
                df_area[df_area["regiao_ies"] == serie]
                .dropna(subset=[taxa])
                .sort_values(["ano", "semestre"])
                .copy()
            )

            if df_serie.empty:
                continue

            estilo = estilos_linhas.get(
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

        # ----------------------------------------------------------------------
        # Eixos
        # ----------------------------------------------------------------------
        ax.set_xlabel(
            "Semestre letivo",
            fontsize=24,
            fontweight="bold",
            labelpad=16,
            color="black",
        )

        ax.set_ylabel(
            "Taxa (%)",
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

        # Mantém as bordas completas do gráfico
        for lado in ["top", "right", "left", "bottom"]:
            ax.spines[lado].set_visible(True)
            ax.spines[lado].set_linewidth(1.1)
            ax.spines[lado].set_color("black")

        # ----------------------------------------------------------------------
        # Legenda com borda, grande e compacta
        # ----------------------------------------------------------------------
        handles_legenda = []

        for serie in series_presentes:
            estilo = estilos_linhas[serie]

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

        # ----------------------------------------------------------------------
        # Ajuste de margens
        # ----------------------------------------------------------------------
        fig.subplots_adjust(
            left=0.075,
            right=0.992,
            top=0.992,
            bottom=0.105,
        )

        # ----------------------------------------------------------------------
        # Salvar
        # ----------------------------------------------------------------------
        nome_area = limpar_nome_arquivo(area)
        nome_taxa = limpar_nome_arquivo(taxa)

        file_png = taxa_dir / f"grafico_{nome_taxa}_{nome_area}.png"
        file_tiff = taxa_dir / f"grafico_{nome_taxa}_{nome_area}.tiff"

        fig.savefig(
            file_png,
            dpi=SAVE_DPI,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )

        fig.savefig(
            file_tiff,
            dpi=SAVE_DPI,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )

        plt.close(fig)

print("✅ Gráficos salvos em tons de cinza, com linhas/Marcadores diferenciados e legenda com borda visível.")
# %%