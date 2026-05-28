# %%
from pathlib import Path
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


def orquestrador_tabela_b1_distribuicao_renda():
    print("\n" + "=" * 72)
    print("TABELA B1: DISTRIBUIÇÃO DE INSCRITOS POR FAIXA DE RENDA")
    print("=" * 72)

    caminho_base = Path("../data/04_load/database/inscritos_final_limpo.parquet")
    pasta_saida = Path("../reports/figures/apendices")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    df = carregar_e_filtrar_base(caminho_base)
    df_tabela = gerar_distribuicao_por_faixa_renda(df)

    print("\nTabela B1:")
    print(df_tabela)

    plotar_tabela_b1_como_imagem(
        df_tabela=df_tabela,
        pasta_saida=pasta_saida
    )


def carregar_e_filtrar_base(caminho_base):
    print("[*] Carregando base...")
    df = pd.read_parquet(str(caminho_base))

    df["modalidade_fies"] = (
        df["modalidade_fies"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Mesmo recorte usado nas matrizes:
    # opção prioritária e Modalidade I.
    df = df[
        (df["opcao_curso"] == 1) &
        (df["modalidade_fies"] == "MODALIDADE I")
    ].copy()

    df["renda_per_capita"] = pd.to_numeric(
        df["renda_per_capita"],
        errors="coerce"
    )

    df = df.dropna(subset=["renda_per_capita"]).copy()

    print(f"Total após filtro: {len(df):,}".replace(",", "."))

    return df


def gerar_distribuicao_por_faixa_renda(df):
    df = df.copy()

    bins_renda = [-np.inf, 600, 1200, 1800, 2400, 3000, np.inf]

    labels_renda = [
        "Até 600",
        "601–1.200",
        "1.201–1.800",
        "1.801–2.400",
        "2.401–3.000",
        "Acima de 3.000"
    ]

    df["faixa_renda"] = pd.cut(
        df["renda_per_capita"],
        bins=bins_renda,
        labels=labels_renda,
        ordered=True
    )

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


def formatar_numero_inteiro(valor):
    return f"{int(valor):,}".replace(",", ".")


def formatar_percentual(valor):
    return f"{valor:.1f}%".replace(".", ",")


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


def plotar_tabela_b1_como_imagem(df_tabela, pasta_saida):
    df_fmt = df_tabela.copy()

    df_fmt["Inscritos"] = df_fmt["Inscritos"].apply(formatar_numero_inteiro)
    df_fmt["%"] = df_fmt["%"].apply(formatar_percentual)
    df_fmt["% acumulado"] = df_fmt["% acumulado"].apply(formatar_percentual)

    fonte_padrao = obter_fonte_padrao()
    plt.rcParams["font.family"] = fonte_padrao

    n_linhas = len(df_fmt)

    # Altura suficiente para deixar as linhas legíveis,
    # mas sem criar excesso de espaço.
    altura = max(2.8, 0.48 * (n_linhas + 1))

    fig = plt.figure(figsize=(8.6, altura), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    tabela = ax.table(
        cellText=df_fmt.values,
        colLabels=df_fmt.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1],
        colWidths=[0.44, 0.22, 0.15, 0.19]
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11.2)
    tabela.scale(1.0, 1.18)

    # Padrão monocromático para impressão/reprodução em escala de cinza
    cor_cabecalho = "#4a4a4a"
    cor_texto_cabecalho = "white"
    cor_linha_1 = "#ededed"
    cor_linha_2 = "#ffffff"
    cor_borda = "#666666"
    cor_borda_externa = "#2f2f2f"

    ncols = len(df_fmt.columns)
    nrows = len(df_fmt) + 1

    for (linha, coluna), celula in tabela.get_celld().items():
        celula.PAD = 0.035
        celula.set_edgecolor(cor_borda)
        celula.set_linewidth(0.8)

        if linha == 0:
            celula.set_facecolor(cor_cabecalho)
            celula.get_text().set_color(cor_texto_cabecalho)
            celula.get_text().set_fontweight("bold")
            celula.get_text().set_fontsize(11.6)
            celula.get_text().set_ha("center")
            celula.get_text().set_va("center")
        else:
            celula.set_facecolor(cor_linha_1 if linha % 2 == 1 else cor_linha_2)
            celula.get_text().set_color("black")
            celula.get_text().set_fontsize(11.2)
            celula.get_text().set_va("center")

            if coluna == 0:
                celula.get_text().set_fontweight("bold")
                celula.get_text().set_ha("left")
            else:
                celula.get_text().set_ha("center")

        # Borda externa mais forte
        if linha in [0, nrows - 1] or coluna in [0, ncols - 1]:
            celula.set_edgecolor(cor_borda_externa)
            celula.set_linewidth(1.1)

    nome_base = limpar_nome_arquivo(
        "tabela_b1_distribuicao_volume_inscritos_por_faixa_renda"
    )

    caminho_png = pasta_saida / f"{nome_base}.png"
    caminho_tiff = pasta_saida / f"{nome_base}.tiff"
    caminho_pdf = pasta_saida / f"{nome_base}.pdf"

    fig.savefig(
        caminho_png,
        dpi=700,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white"
    )

    fig.savefig(
        caminho_tiff,
        dpi=700,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white"
    )

    fig.savefig(
        caminho_pdf,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white"
    )

    plt.close(fig)

    print(f"Tabela B1 exportada para: {caminho_png}")
    print(f"Tabela B1 em TIFF exportada para: {caminho_tiff}")
    print(f"Tabela B1 em PDF exportada para: {caminho_pdf}")


orquestrador_tabela_b1_distribuicao_renda()
# %%