# %%
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


def orquestrador_tabela_1():
    print("\n" + "=" * 72)
    print("TABELA 1: DISTRIBUIÇÃO DO VOLUME DE INSCRITOS POR SITUAÇÃO")
    print("=" * 72)

    caminho_base = Path("../data/04_load/database/inscritos_final_limpo.parquet")
    pasta_saida = Path("../reports/figures/secao_4_2")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    df = carregar_base(caminho_base)
    df_tabela = preparar_tabela_1(df)

    print("\nTabela 1:")
    print(df_tabela)

    plotar_tabela_1_como_imagem(
        df_tabela=df_tabela,
        pasta_saida=pasta_saida
    )


def carregar_base(caminho_base):
    print("[*] Carregando base...")
    df = pd.read_parquet(str(caminho_base))
    print(f"Total de registros: {len(df):,}".replace(",", "."))
    return df


def normalizar_status(texto):
    return str(texto).strip().upper()


def preparar_tabela_1(df):
    df = df.copy()

    mapa_status = {
        "CONTRATADA": "Contratada",
        "INSCRIÇÃO POSTERGADA": "Inscrição postergada",
        "PRE-SELECIONADO": "Pré-selecionado",
        "PRÉ-SELECIONADO": "Pré-selecionado",
        "NÃO CONTRATADO": "Não contratado",
        "NAO CONTRATADO": "Não contratado",
        "REJEITADA PELA CPSA": "Rejeitada pela CPSA",
        "OPÇÃO NÃO CONTRATADA": "Opção não contratada",
        "OPCAO NAO CONTRATADA": "Opção não contratada",
        "PARTICIPACAO CANCELADA PELO CANDIDATO": "Participação cancelada",
        "PARTICIPAÇÃO CANCELADA PELO CANDIDATO": "Participação cancelada",
        "LISTA DE ESPERA": "Lista de espera"
    }

    df["status_norm"] = df["situacao_fies"].apply(normalizar_status)
    df["status_tabela"] = df["status_norm"].map(mapa_status).fillna("Outros")

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


def formatar_inteiro(valor):
    return f"{int(valor):,}".replace(",", ".")


def formatar_percentual(valor):
    return f"{valor:.1f}%".replace(".", ",")


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


def plotar_tabela_1_como_imagem(df_tabela, pasta_saida):
    df_fmt = df_tabela.copy()

    df_fmt["Quantidade de inscritos"] = (
        df_fmt["Quantidade de inscritos"].apply(formatar_inteiro)
    )
    df_fmt["%"] = df_fmt["%"].apply(formatar_percentual)

    fonte_padrao = obter_fonte_padrao()
    plt.rcParams["font.family"] = fonte_padrao

    n_linhas = len(df_fmt)

    # Altura dinâmica: tabela mais "alta" para melhorar legibilidade
    altura = max(3.0, 0.58 * (n_linhas + 1))

    # Largura suficiente para a primeira coluna respirar
    fig = plt.figure(figsize=(8.2, altura), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    tabela = ax.table(
        cellText=df_fmt.values,
        colLabels=df_fmt.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1],
        colWidths=[0.56, 0.26, 0.18]
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11.5)

    # Escala vertical para aumentar altura das células
    tabela.scale(1.0, 1.28)

    # Tons de cinza com contraste bom
    cor_cabecalho = "#4a4a4a"
    cor_texto_cabecalho = "white"
    cor_linha_1 = "#ececec"
    cor_linha_2 = "#ffffff"
    cor_borda = "#5f5f5f"
    cor_borda_externa = "#2f2f2f"

    ncols = len(df_fmt.columns)
    nrows = len(df_fmt) + 1  # inclui cabeçalho

    for (linha, coluna), celula in tabela.get_celld().items():
        celula.PAD = 0.035
        celula.set_edgecolor(cor_borda)
        celula.set_linewidth(0.8)

        if linha == 0:
            celula.set_facecolor(cor_cabecalho)
            celula.get_text().set_color(cor_texto_cabecalho)
            celula.get_text().set_fontweight("bold")
            celula.get_text().set_fontsize(12.2)
            celula.get_text().set_ha("center")
            celula.get_text().set_va("center")
        else:
            celula.set_facecolor(cor_linha_1 if linha % 2 == 1 else cor_linha_2)
            celula.get_text().set_color("black")
            celula.get_text().set_fontsize(11.5)
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

    nome_base = limpar_nome_arquivo("tabela_1_distribuicao_volume_inscritos_por_situacao")

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

    print(f"Tabela 1 exportada para: {caminho_png}")
    print(f"Tabela 1 em TIFF exportada para: {caminho_tiff}")
    print(f"Tabela 1 em PDF exportada para: {caminho_pdf}")


orquestrador_tabela_1()
# %%