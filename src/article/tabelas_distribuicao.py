from pathlib import Path
import re
import unicodedata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.constants import (
    CURATED_INSCRICOES_ARTIGO_PATH,
    TABLES_DIR,
    APPENDIX_DIR,
    LOGS_DIR,
)


FIG_DPI = 300
SAVE_DPI = 700

PASTA_TABELA_1 = TABLES_DIR / "secao_4_2"
PASTA_TABELA_B1 = APPENDIX_DIR / "apendice_b"

RESUMO_PATH = LOGS_DIR / "article_tabelas_distribuicao_resumo.csv"

STATUS_MAPA = {
    "CONTRATADA": "Contratada",
    "INSCRIÇÃO POSTERGADA": "Inscrição postergada",
    "INSCRICAO POSTERGADA": "Inscrição postergada",
    "PRÉ-SELECIONADO": "Pré-selecionado",
    "PRE-SELECIONADO": "Pré-selecionado",
    "NÃO CONTRATADO": "Não contratado",
    "NAO CONTRATADO": "Não contratado",
    "REJEITADA PELA CPSA": "Rejeitada pela CPSA",
    "OPÇÃO NÃO CONTRATADA": "Opção não contratada",
    "OPCAO NAO CONTRATADA": "Opção não contratada",
    "PARTICIPAÇÃO CANCELADA PELO CANDIDATO": "Participação cancelada",
    "PARTICIPACAO CANCELADA PELO CANDIDATO": "Participação cancelada",
    "LISTA DE ESPERA": "Lista de espera",
}

FAIXAS_RENDA_LABELS = [
    "Até 600",
    "601–1.200",
    "1.201–1.800",
    "1.801–2.400",
    "2.401–3.000",
    "Acima de 3.000",
]

FAIXAS_RENDA_BINS = [-np.inf, 600, 1200, 1800, 2400, 3000, np.inf]

POSSIVEIS_COLUNAS_RENDA = [
    "renda_familiar_per_capita",
    "renda_per_capita",
    "vl_renda_per_capita",
    "valor_renda_per_capita",
    "renda_familiar_mensal_per_capita",
]

POSSIVEIS_COLUNAS_SITUACAO = [
    "situacao_fies",
    "situacao_inscricao",
    "situacao",
]

POSSIVEIS_COLUNAS_MODALIDADE = [
    "modalidade_fies",
    "modalidade",
]


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "article_tabelas_distribuicao.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


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
    plt.rcParams["figure.dpi"] = FIG_DPI
    plt.rcParams["savefig.dpi"] = SAVE_DPI


def normalizar_texto(valor):
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip().upper()
    texto = re.sub(r"\s+", " ", texto)

    if texto in {"", "NAN", "NONE", "NULL", "NA", "N/A", "-", "--"}:
        return pd.NA

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


def aplicar_filtro_modalidade_i_se_existir(df: pd.DataFrame) -> pd.DataFrame:
    # O recorte do artigo já é definido na curadoria como todos os registros
    # de 2019 a 2021. A coluna modalidade_fies pode existir na base, mas não é
    # usada como filtro nesta etapa.
    return df.copy()


def preparar_tabela_1(df: pd.DataFrame) -> pd.DataFrame:
    coluna_situacao = encontrar_coluna(df, POSSIVEIS_COLUNAS_SITUACAO, "situação da inscrição")

    base = aplicar_filtro_modalidade_i_se_existir(df)

    status_norm = base[coluna_situacao].map(normalizar_texto)
    status_tabela = status_norm.map(STATUS_MAPA).fillna("Outros")

    tabela = (
        status_tabela
        .value_counts(dropna=False)
        .rename_axis("Situação da inscrição")
        .reset_index(name="Quantidade de inscrições")
    )

    total = tabela["Quantidade de inscrições"].sum()

    tabela["%"] = np.where(
        total > 0,
        tabela["Quantidade de inscrições"] / total * 100,
        np.nan,
    )

    tabela = (
        tabela
        .sort_values(
            ["Quantidade de inscrições", "Situação da inscrição"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return tabela


def preparar_tabela_b1(df: pd.DataFrame) -> pd.DataFrame:
    coluna_renda = encontrar_coluna(df, POSSIVEIS_COLUNAS_RENDA, "renda familiar per capita")

    base = aplicar_filtro_modalidade_i_se_existir(df)

    base[coluna_renda] = pd.to_numeric(base[coluna_renda], errors="coerce")
    base = base.dropna(subset=[coluna_renda]).copy()

    base["faixa_renda"] = pd.cut(
        base[coluna_renda],
        bins=FAIXAS_RENDA_BINS,
        labels=FAIXAS_RENDA_LABELS,
        ordered=True,
    )

    tabela = (
        base
        .groupby("faixa_renda", observed=True)
        .size()
        .reindex(FAIXAS_RENDA_LABELS, fill_value=0)
        .reset_index(name="Inscrições")
    )

    total = tabela["Inscrições"].sum()

    tabela["%"] = np.where(
        total > 0,
        tabela["Inscrições"] / total * 100,
        np.nan,
    )

    tabela = tabela.rename(
        columns={"faixa_renda": "Faixa de renda"}
    )

    tabela["Faixa de renda"] = tabela["Faixa de renda"].astype(str)

    # Mantém a ordem substantiva das faixas de renda, do menor para o maior valor.
    # A tabela B1 documenta a distribuição das inscrições por faixa; por isso,
    # não deve ser ordenada pela quantidade de registros.
    tabela = tabela.reset_index(drop=True)

    tabela["% acumulado"] = tabela["%"].cumsum()

    return tabela


def formatar_inteiro(valor) -> str:
    return f"{int(valor):,}".replace(",", ".")


def formatar_percentual(valor) -> str:
    if pd.isna(valor):
        return ""

    return f"{valor:.1f}%".replace(".", ",")


def formatar_tabela_para_exibicao(df_tabela: pd.DataFrame) -> pd.DataFrame:
    df = df_tabela.copy()

    for coluna in df.columns:
        if coluna in {"Quantidade de inscrições", "Inscrições"}:
            df[coluna] = df[coluna].apply(formatar_inteiro)
        elif coluna in {"%", "% acumulado"}:
            df[coluna] = df[coluna].apply(formatar_percentual)

    return df


def salvar_csv_tex(df_tabela: pd.DataFrame, pasta_saida: Path, nome_base: str, caption: str, label: str) -> dict:
    pasta_saida.mkdir(parents=True, exist_ok=True)

    caminho_csv = pasta_saida / f"{nome_base}.csv"
    caminho_tex = pasta_saida / f"{nome_base}.tex"

    # As tabelas desta etapa são artefatos finais do artigo. Por isso, o CSV e
    # o TeX são salvos já na forma de exibição: inteiros com separador de milhar
    # brasileiro e percentuais acompanhados do símbolo "%".
    df_saida = formatar_tabela_para_exibicao(df_tabela)

    df_saida.to_csv(caminho_csv, index=False, encoding="utf-8")

    with caminho_tex.open("w", encoding="utf-8") as file:
        file.write(
            df_saida.to_latex(
                index=False,
                caption=caption,
                label=label,
                escape=True,
            )
        )

    return {
        "csv": str(caminho_csv),
        "tex": str(caminho_tex),
    }


def renderizar_tabela(
    df_tabela: pd.DataFrame,
    pasta_saida: Path,
    nome_base: str,
    col_widths: list[float],
    altura_minima: float,
    altura_por_linha: float,
    fontsize_corpo: float,
    fontsize_cabecalho: float,
) -> dict:
    pasta_saida.mkdir(parents=True, exist_ok=True)

    df_fmt = formatar_tabela_para_exibicao(df_tabela)

    n_linhas = len(df_fmt)
    altura = max(altura_minima, altura_por_linha * (n_linhas + 1))

    fig = plt.figure(figsize=(8.8, altura), dpi=FIG_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    tabela = ax.table(
        cellText=df_fmt.values,
        colLabels=df_fmt.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1],
        colWidths=col_widths,
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(fontsize_corpo)
    tabela.scale(1.0, 1.22)

    cor_cabecalho = "#4a4a4a"
    cor_texto_cabecalho = "white"
    cor_linha_1 = "#ececec"
    cor_linha_2 = "#ffffff"
    cor_borda = "#5f5f5f"
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
            celula.get_text().set_fontsize(fontsize_cabecalho)
            celula.get_text().set_ha("center")
            celula.get_text().set_va("center")
        else:
            celula.set_facecolor(cor_linha_1 if linha % 2 == 1 else cor_linha_2)
            celula.get_text().set_color("black")
            celula.get_text().set_fontsize(fontsize_corpo)
            celula.get_text().set_va("center")

            if coluna == 0:
                celula.get_text().set_fontweight("bold")
                celula.get_text().set_ha("left")
            else:
                celula.get_text().set_ha("center")

        if linha in [0, nrows - 1] or coluna in [0, ncols - 1]:
            celula.set_edgecolor(cor_borda_externa)
            celula.set_linewidth(1.1)

    caminhos = {
        "pdf": pasta_saida / f"{nome_base}.pdf",
        "png": pasta_saida / f"{nome_base}.png",
    }

    fig.savefig(
        caminhos["pdf"],
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )

    fig.savefig(
        caminhos["png"],
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )


    plt.close(fig)

    return {
        "pdf": str(caminhos["pdf"]),
        "png": str(caminhos["png"]),
    }


def gerar_tabela_1(df: pd.DataFrame) -> dict:
    tabela = preparar_tabela_1(df)

    nome_base = "tabela_1_distribuicao_inscricoes_por_situacao"

    arquivos = {}
    arquivos.update(
        salvar_csv_tex(
            df_tabela=tabela,
            pasta_saida=PASTA_TABELA_1,
            nome_base=nome_base,
            caption="Distribuição das inscrições por situação no FIES, 2019--2021.",
            label="tab:distribuicao_situacao_inscricao",
        )
    )

    arquivos.update(
        renderizar_tabela(
            df_tabela=tabela,
            pasta_saida=PASTA_TABELA_1,
            nome_base=nome_base,
            col_widths=[0.56, 0.26, 0.18],
            altura_minima=3.0,
            altura_por_linha=0.58,
            fontsize_corpo=11.5,
            fontsize_cabecalho=12.0,
        )
    )

    log(f"[OK] Tabela 1 gerada em: {PASTA_TABELA_1}")

    return {
        "tabela": "Tabela 1",
        "linhas": len(tabela),
        **arquivos,
    }


def gerar_tabela_b1(df: pd.DataFrame) -> dict:
    tabela = preparar_tabela_b1(df)

    nome_base = "tabela_b1_distribuicao_inscricoes_por_faixa_renda"

    arquivos = {}
    arquivos.update(
        salvar_csv_tex(
            df_tabela=tabela,
            pasta_saida=PASTA_TABELA_B1,
            nome_base=nome_base,
            caption="Distribuição das inscrições por faixa de renda familiar per capita no FIES, 2019--2021.",
            label="tab:distribuicao_faixa_renda",
        )
    )

    arquivos.update(
        renderizar_tabela(
            df_tabela=tabela,
            pasta_saida=PASTA_TABELA_B1,
            nome_base=nome_base,
            col_widths=[0.44, 0.22, 0.15, 0.19],
            altura_minima=2.8,
            altura_por_linha=0.48,
            fontsize_corpo=11.2,
            fontsize_cabecalho=11.6,
        )
    )

    log(f"[OK] Tabela B1 gerada em: {PASTA_TABELA_B1}")

    return {
        "tabela": "Tabela B1",
        "linhas": len(tabela),
        **arquivos,
    }


def salvar_resumo(registros: list[dict]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(RESUMO_PATH, index=False, encoding="utf-8")
    log(f"[OK] Resumo salvo em: {RESUMO_PATH}")


def run() -> None:
    configurar_matplotlib()

    log("=" * 80)
    log("ARTICLE: TABELAS DE DISTRIBUIÇÃO")
    log("=" * 80)

    df = carregar_base()

    registros = [
        gerar_tabela_1(df),
        gerar_tabela_b1(df),
    ]

    salvar_resumo(registros)

    log("Tabelas de distribuição concluídas.")
