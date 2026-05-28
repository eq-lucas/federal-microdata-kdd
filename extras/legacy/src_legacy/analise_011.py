# %%
from pathlib import Path
import re

import pandas as pd
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

    resultados = calcular_estatisticas_e_regressao(df)

    plotar_regressao_artigo(
        df=df,
        resultados=resultados,
        pasta_saida=pasta_saida
    )

    plotar_tabela_resultados_como_imagem(
        resultados=resultados,
        pasta_saida=pasta_saida
    )


def preparar_dados_financiamento():
    try:
        df_base = pd.read_parquet(str(pasta_data_04_load_inscritos))
    except Exception as e:
        print(f"Erro ao carregar o arquivo parquet: {e}")
        return None

    # Apenas registros com contrato efetivado
    df = df_base[df_base["situacao_fies"] == "CONTRATADA"].copy()

    df["renda_per_capita"] = pd.to_numeric(
        df["renda_per_capita"],
        errors="coerce"
    )

    df["percentual_financiamento"] = pd.to_numeric(
        df["percentual_financiamento"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["renda_per_capita", "percentual_financiamento"]
    )

    # Recorte para reduzir extremos
    df = df[
        (df["renda_per_capita"] > 0) &
        (df["renda_per_capita"] <= 5000) &
        (df["percentual_financiamento"] >= 0) &
        (df["percentual_financiamento"] <= 100)
    ].copy()

    print(
        f"Registros válidos com contrato efetivado: "
        f"{len(df):,}".replace(",", ".")
    )

    return df


def calcular_estatisticas_e_regressao(df):
    x = df[["renda_per_capita"]]
    y = df["percentual_financiamento"]

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

    resultados = {
        "n": len(df),
        "pearson_r": r,
        "p_value": p_value,
        "intercepto": intercepto,
        "beta_renda": beta,
        "efeito_100_reais": efeito_100_reais,
        "r2": r2
    }

    print("\n--- RESULTADOS ESTATÍSTICOS ---")
    print(f"N: {resultados['n']:,}".replace(",", "."))
    print(f"Correlação de Pearson: {r:.4f}")
    print(f"p-valor: {p_value:.4e}")
    print(f"Equação: percentual = ({beta:.4f} * renda) + {intercepto:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Efeito a cada R$ 100: {efeito_100_reais:.2f} p.p.")

    return resultados


def obter_fonte_padrao():
    fontes_disponiveis = {f.name for f in font_manager.fontManager.ttflist}

    if "Times New Roman" in fontes_disponiveis:
        return "Times New Roman"
    elif "Liberation Serif" in fontes_disponiveis:
        return "Liberation Serif"
    elif "Nimbus Roman" in fontes_disponiveis:
        return "Nimbus Roman"
    else:
        return "DejaVu Serif"


def limpar_nome_arquivo(texto):
    texto = str(texto).lower()

    substituicoes = {
        "ç": "c", "ã": "a", "á": "a", "à": "a", "â": "a", "ä": "a",
        "é": "e", "ê": "e", "è": "e", "ë": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i",
        "ó": "o", "õ": "o", "ô": "o", "ò": "o", "ö": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u"
    }

    for original, novo in substituicoes.items():
        texto = texto.replace(original, novo)

    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    texto = re.sub(r"_+", "_", texto)

    return texto.strip("_")


def plotar_regressao_artigo(df, resultados, pasta_saida):
    fonte_padrao = obter_fonte_padrao()

    plt.rcParams["font.family"] = fonte_padrao
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 1200

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=400)

    # Muitos pontos: tamanho pequeno + opacidade bem baixa
    ax.scatter(
        df["renda_per_capita"],
        df["percentual_financiamento"],
        s=2.0,
        alpha=0.022,
        color="#7b7b7b",
        edgecolors="none",
        rasterized=True
    )

    x_linha = np.linspace(0, 5000, 400)
    y_linha = resultados["intercepto"] + resultados["beta_renda"] * x_linha

    ax.plot(
        x_linha,
        y_linha,
        color="#4f4f4f",
        linewidth=2.5
    )

    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 105)

    ax.set_xlabel(
        "Renda familiar per capita (R$)",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Percentual de financiamento (%)",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.75,
        alpha=0.35,
        color="#9a9a9a"
    )

    ax.tick_params(axis="both", labelsize=12)

    texto = (
        f"r = {resultados['pearson_r']:.4f}\n"
        f"R² = {resultados['r2']:.4f}\n"
        f"n = {resultados['n']:,}".replace(",", ".")
    )

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
            "alpha": 0.96
        }
    )

    for lado in ["top", "right", "left", "bottom"]:
        ax.spines[lado].set_visible(True)
        ax.spines[lado].set_linewidth(1.0)
        ax.spines[lado].set_color("black")

    plt.tight_layout()

    nome_base = limpar_nome_arquivo("figura_5_renda_percentual_financiamento")

    caminho_png = pasta_saida / f"{nome_base}.png"
    caminho_tiff = pasta_saida / f"{nome_base}.tiff"
    caminho_pdf = pasta_saida / f"{nome_base}.pdf"

    plt.savefig(
        caminho_png,
        bbox_inches="tight",
        dpi=1200,
        facecolor="white",
        pad_inches=0.02
    )

    plt.savefig(
        caminho_tiff,
        bbox_inches="tight",
        dpi=1200,
        facecolor="white",
        pad_inches=0.02
    )

    plt.savefig(
        caminho_pdf,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02
    )

    plt.close()

    print(f"Figura 5 exportada para: {caminho_png}")
    print(f"Figura 5 em TIFF exportada para: {caminho_tiff}")
    print(f"Figura 5 em PDF exportada para: {caminho_pdf}")


def plotar_tabela_resultados_como_imagem(resultados, pasta_saida):
    linhas = [
        [
            "Observações válidas",
            f"{resultados['n']:,}".replace(",", ".")
        ],
        [
            "Correlação de Pearson",
            f"{resultados['pearson_r']:.4f}".replace(".", ",")
        ],
        [
            "p-valor",
            "< 0,001"
            if resultados["p_value"] < 0.001
            else f"{resultados['p_value']:.4f}".replace(".", ",")
        ],
        [
            "Intercepto",
            f"{resultados['intercepto']:.2f}".replace(".", ",")
        ],
        [
            "Coeficiente da renda per capita",
            f"{resultados['beta_renda']:.4f}".replace(".", ",")
        ],
        [
            "Variação estimada a cada R$ 100",
            f"{resultados['efeito_100_reais']:.2f} p.p.".replace(".", ",")
        ],
        [
            "R²",
            f"{resultados['r2']:.4f}".replace(".", ",")
        ]
    ]

    colunas = ["Medida", "Valor"]

    fonte_padrao = obter_fonte_padrao()
    plt.rcParams["font.family"] = fonte_padrao

    fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=300)
    ax.axis("off")

    tabela = ax.table(
        cellText=linhas,
        colLabels=colunas,
        loc="center",
        cellLoc="left",
        colLoc="center",
        colWidths=[0.70, 0.30]
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11.0)
    tabela.scale(1, 1.60)

    # padrão em tons de cinza
    cor_cabecalho = "#4a4a4a"
    cor_linha_1 = "#ececec"
    cor_linha_2 = "#ffffff"
    cor_borda = "#6a6a6a"
    cor_borda_externa = "#2f2f2f"

    ncols = len(colunas)
    nrows = len(linhas) + 1

    for (linha, coluna), celula in tabela.get_celld().items():
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
            celula.set_facecolor(
                cor_linha_1 if linha % 2 == 1 else cor_linha_2
            )

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

    nome_base = limpar_nome_arquivo("tabela_3_associacao_renda_financiamento")

    caminho_png = pasta_saida / f"{nome_base}.png"
    caminho_tiff = pasta_saida / f"{nome_base}.tiff"
    caminho_pdf = pasta_saida / f"{nome_base}.pdf"

    plt.savefig(
        caminho_png,
        bbox_inches="tight",
        dpi=1200,
        facecolor="white",
        pad_inches=0.02
    )

    plt.savefig(
        caminho_tiff,
        bbox_inches="tight",
        dpi=1200,
        facecolor="white",
        pad_inches=0.02
    )

    plt.savefig(
        caminho_pdf,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02
    )

    plt.close()

    print(f"Tabela 3 exportada para: {caminho_png}")
    print(f"Tabela 3 em TIFF exportada para: {caminho_tiff}")
    print(f"Tabela 3 em PDF exportada para: {caminho_pdf}")

orquestrador_analise_011()
# %%