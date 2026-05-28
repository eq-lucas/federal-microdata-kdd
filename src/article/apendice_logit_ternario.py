from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


SAVE_DPI = 700

BINS_RENDA = [-np.inf, 600, 1200, 1800, 2400, 3000, np.inf]
LABELS_RENDA = ["I", "II", "III", "IV", "V", "VI"]

BINS_DESEMPENHO = [-np.inf, -150, -50, 0, 50, 150, np.inf]
LABELS_DESEMPENHO = [
    "< -150",
    "-150 a -50",
    "-50 a 0",
    "0 a +50",
    "+50 a +150",
    "> +150",
]

ORDEM_DESEMPENHO_PLOT = [
    "> +150",
    "+50 a +150",
    "0 a +50",
    "-50 a 0",
    "-150 a -50",
    "< -150",
]

COR_CABECALHO = "#1f4e79"
COR_LINHA_1 = "#eaf2f8"
COR_LINHA_2 = "#ffffff"
COR_BORDA = "#b7c9d6"


def obter_fonte_padrao() -> str:
    fontes_disponiveis = {f.name for f in font_manager.fontManager.ttflist}

    if "Times New Roman" in fontes_disponiveis:
        return "Times New Roman"

    if "Liberation Serif" in fontes_disponiveis:
        return "Liberation Serif"

    if "Nimbus Roman" in fontes_disponiveis:
        return "Nimbus Roman"

    return "DejaVu Sans"


def configurar_matplotlib() -> None:
    plt.rcParams["font.family"] = obter_fonte_padrao()
    plt.rcParams["axes.linewidth"] = 1.0


def fmt_int(valor):
    if pd.isna(valor):
        return ""

    return f"{int(valor):,}".replace(",", ".")


def fmt_float(valor, casas=4):
    if pd.isna(valor):
        return ""

    return f"{float(valor):.{casas}f}".replace(".", ",")


def quebrar_texto(texto, largura=48):
    return "\n".join(textwrap.wrap(str(texto), width=largura))


def salvar_figura(fig, caminho_base: Path) -> dict:
    caminho_base.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "pdf": caminho_base.with_suffix(".pdf"),
        "png": caminho_base.with_suffix(".png"),
    }

    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(paths["png"], dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")

    plt.close(fig)

    return {k: str(v) for k, v in paths.items()}


def salvar_tabela_latex(df_tabela: pd.DataFrame, caminho_base: Path, caption: str, label: str) -> dict:
    caminho_base.parent.mkdir(parents=True, exist_ok=True)

    tex = df_tabela.to_latex(index=False, caption=caption, label=label, escape=True)

    tex_path = caminho_base.with_suffix(".tex")
    latex_path = caminho_base.with_suffix(".latex")

    tex_path.write_text(tex, encoding="utf-8")
    latex_path.write_text(tex, encoding="utf-8")

    return {"tex": str(tex_path), "latex": str(latex_path)}


def salvar_tabela_como_imagem(
    df_tabela: pd.DataFrame,
    caminho_base: Path,
    figsize=(7.2, 3.2),
    col_widths=None,
    fontsize=10.5,
) -> dict:
    caminho_base.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    tabela = ax.table(
        cellText=df_tabela.values,
        colLabels=df_tabela.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1],
        colWidths=col_widths,
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
            celula.set_facecolor(COR_LINHA_1 if linha % 2 == 1 else COR_LINHA_2)

            if coluna == 0:
                celula.get_text().set_fontweight("bold")
                celula.get_text().set_ha("left")
            else:
                celula.get_text().set_ha("center")

            celula.get_text().set_va("center")

    paths = salvar_figura(fig, caminho_base)

    return paths


def preparar_base_probabilidades(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    out = df.copy()

    out["faixa_renda"] = pd.cut(
        out["renda_per_capita"],
        bins=BINS_RENDA,
        labels=LABELS_RENDA,
        ordered=True,
    )

    out["faixa_desempenho"] = pd.cut(
        out["gap"],
        bins=BINS_DESEMPENHO,
        labels=LABELS_DESEMPENHO,
        ordered=True,
    )

    out = out.dropna(subset=["faixa_renda", "faixa_desempenho", prob_col]).copy()

    return out


def matriz_probabilidade_prevista(df_pred: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    return (
        df_pred
        .groupby(["faixa_desempenho", "faixa_renda"], observed=True)[prob_col]
        .mean()
        .reset_index()
        .pivot(index="faixa_desempenho", columns="faixa_renda", values=prob_col)
        .reindex(index=ORDEM_DESEMPENHO_PLOT, columns=LABELS_RENDA)
    )


def tabela_media_prob_por_faixas(df_pred: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    return (
        df_pred
        .groupby(["faixa_renda", "faixa_desempenho"], observed=True)[prob_col]
        .mean()
        .reset_index()
    )


def gerar_heatmap_probabilidade(df_pred: pd.DataFrame, prob_col: str, caminho_base: Path) -> dict:
    matriz = matriz_probabilidade_prevista(df_pred, prob_col)

    fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=300)

    valores = matriz.to_numpy(dtype=float)
    ax.imshow(valores, cmap="Greys", vmin=0, vmax=100, aspect="auto")

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = matriz.iloc[i, j]

            if pd.isna(valor):
                texto = ""
                cor = "black"
            else:
                texto = f"{valor:.1f}".replace(".", ",")
                cor = "white" if valor >= 55 else "black"

            ax.text(
                j,
                i,
                texto,
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=cor,
            )

    ax.set_xlabel("Faixa de renda familiar per capita", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Desempenho acadêmico", fontsize=11, fontweight="bold", labelpad=10)

    ax.set_xticks(np.arange(len(LABELS_RENDA)))
    ax.set_xticklabels(LABELS_RENDA, rotation=0, fontsize=10.5)

    ax.set_yticks(np.arange(len(ORDEM_DESEMPENHO_PLOT)))
    ax.set_yticklabels(ORDEM_DESEMPENHO_PLOT, rotation=0, fontsize=10.5)

    ax.tick_params(axis="both", length=0)

    for lado in ["top", "right", "left", "bottom"]:
        ax.spines[lado].set_visible(True)
        ax.spines[lado].set_linewidth(1.0)
        ax.spines[lado].set_color("black")

    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.16, top=0.985)

    return salvar_figura(fig, caminho_base)


def gerar_curvas_desempenho_por_renda(df_pred: pd.DataFrame, prob_col: str, caminho_base: Path) -> dict:
    tabela = tabela_media_prob_por_faixas(df_pred, prob_col)

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
            subset[prob_col],
            linestyle=estilos[idx % len(estilos)],
            marker=marcadores[idx % len(marcadores)],
            linewidth=1.8,
            markersize=5.0,
            label=faixa_renda,
            color=str(0.15 + idx * 0.12),
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
        title_fontsize=9.8,
    )

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.16, top=0.78)

    return salvar_figura(fig, caminho_base)


def gerar_curvas_renda_por_desempenho(df_pred: pd.DataFrame, prob_col: str, caminho_base: Path) -> dict:
    tabela = tabela_media_prob_por_faixas(df_pred, prob_col)

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
            subset[prob_col],
            linestyle=estilos[idx % len(estilos)],
            marker=marcadores[idx % len(marcadores)],
            linewidth=1.8,
            markersize=5.0,
            label=faixa_desempenho,
            color=str(0.15 + idx * 0.12),
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
        title_fontsize=9.8,
    )

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.72)

    return salvar_figura(fig, caminho_base)


def renderizar_matriz_confusao(matriz: pd.DataFrame, caminho_base: Path) -> dict:
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=300)

    valores = matriz.to_numpy(dtype=float)
    ax.imshow(valores, cmap="Greys", aspect="auto")

    max_val = np.nanmax(valores) if np.isfinite(valores).any() else 0

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = matriz.iloc[i, j]
            cor = "white" if max_val > 0 and valor >= max_val * 0.45 else "black"

            ax.text(
                j,
                i,
                fmt_int(valor),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=cor,
            )

    ax.set_xlabel("Previsto", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Real", fontsize=11, fontweight="bold", labelpad=10)

    ax.set_xticks(np.arange(matriz.shape[1]))
    ax.set_xticklabels(matriz.columns, rotation=0, fontsize=10.5)

    ax.set_yticks(np.arange(matriz.shape[0]))
    ax.set_yticklabels(matriz.index, rotation=0, fontsize=10.5)

    ax.tick_params(axis="both", length=0)

    for lado in ["top", "right", "left", "bottom"]:
        ax.spines[lado].set_visible(True)
        ax.spines[lado].set_linewidth(1.0)
        ax.spines[lado].set_color("black")

    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.14, top=0.985)

    return salvar_figura(fig, caminho_base)

import argparse

from src import constants as C
from src.modeling.logit_ternario_utils import CLASS_LABELS, diagnostics_dir, normalizar_recorte


PROJECT_ROOT = getattr(C, "PROJECT_ROOT", PROJECT_ROOT_FOR_IMPORT)
LOGS_DIR = getattr(C, "LOGS_DIR", PROJECT_ROOT / "reports" / "logs")
APPENDIX_DIR = getattr(C, "APPENDIX_DIR", PROJECT_ROOT / "reports" / "article" / "appendix")


COLUNAS_TABELA_COMPLETA = [
    "Experimento",
    "Descrição",
    "Tamanho da amostra",
    "N treino",
    "N teste",
    "ROC-AUC OVR",
    "Acurácia balanceada",
    "F1 macro",
    "Renda",
    "Desempenho (gap)",
    "Interação",
    "Blocos",
    "Colunas finais",
    "Blocos incluídos",
    "Avaliação",
]

COLUNAS_TABELA_COMPACTA = [
    "Especificação",
    "N",
    "ROC-AUC",
    "Renda",
    "Desempenho",
    "Interação",
    "Variáveis",
]


def appendix_logit_dir(recorte: str) -> Path:
    recorte = normalizar_recorte(recorte)

    if recorte == "medicina":
        return APPENDIX_DIR / "apendice_modelagem" / "logit_ternario_recorte_medicina"

    return APPENDIX_DIR / "apendice_modelagem" / "logit_ternario_recorte_geral"


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "article_apendice_logit_ternario.log"

    with path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def carregar_metricas(recorte: str, avaliacao: str) -> pd.DataFrame:
    recorte = normalizar_recorte(recorte)
    path = diagnostics_dir(recorte, avaliacao) / "logit_ternario_experimentos_metricas.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Métricas não encontradas: {path}. "
            f"Rode primeiro: python3 main.py modeling fit-ternario {'general' if recorte == 'geral' else recorte} --force"
        )

    return pd.read_csv(path)


def preparar_tabela_completa(df: pd.DataFrame) -> pd.DataFrame:
    tabela = pd.DataFrame(
        {
            "Experimento": df["experimento"] + " - " + df["experimento_nome"],
            "Descrição": df["experimento_descricao"],
            "Tamanho da amostra": df["n_total_abt"].map(fmt_int),
            "N treino": df["n_treino"].map(fmt_int),
            "N teste": df["n_teste"].map(fmt_int),
            "ROC-AUC OVR": df["roc_auc_ovr_weighted"].map(fmt_float),
            "Acurácia balanceada": df["balanced_accuracy"].map(fmt_float),
            "F1 macro": df["f1_macro"].map(fmt_float),
            "Renda": df["coef_renda_per_capita_classe_contratada"].map(fmt_float),
            "Desempenho (gap)": df["coef_gap_classe_contratada"].map(fmt_float),
            "Interação": df["coef_renda_gap_classe_contratada"].map(fmt_float),
            "Blocos": df["blocos_variaveis_qtd"].map(fmt_int),
            "Colunas finais": df["colunas_finais_pos_processamento"].map(fmt_int),
            "Blocos incluídos": df["blocos_incluidos"],
            "Avaliação": df["avaliacao_descricao"],
        }
    )

    return tabela[COLUNAS_TABELA_COMPLETA]


def preparar_tabela_compacta(df: pd.DataFrame) -> pd.DataFrame:
    tabela = pd.DataFrame(
        {
            "Especificação": df["experimento"].str.replace("E", "Modelo ", regex=False),
            "N": df["n_total_abt"].map(fmt_int),
            "ROC-AUC": df["roc_auc_ovr_weighted"].map(fmt_float),
            "Renda": df["coef_renda_per_capita_classe_contratada"].map(fmt_float),
            "Desempenho": df["coef_gap_classe_contratada"].map(fmt_float),
            "Interação": df["coef_renda_gap_classe_contratada"].map(fmt_float),
            "Variáveis": df["blocos_variaveis_qtd"].map(fmt_int),
        }
    )

    return tabela[COLUNAS_TABELA_COMPACTA]


def renderizar_tabela_completa(tabela: pd.DataFrame, caminho_base: Path) -> dict:
    tabela_img = tabela.copy()
    tabela_img["Descrição"] = tabela_img["Descrição"].map(lambda x: quebrar_texto(x, 42))
    tabela_img["Blocos incluídos"] = tabela_img["Blocos incluídos"].map(lambda x: quebrar_texto(x, 58))

    colunas_visiveis = [
        "Experimento",
        "Tamanho da amostra",
        "N treino",
        "N teste",
        "ROC-AUC OVR",
        "Renda",
        "Desempenho (gap)",
        "Interação",
        "Blocos",
        "Colunas finais",
        "Blocos incluídos",
    ]

    tabela_img = tabela_img[colunas_visiveis]

    return salvar_tabela_como_imagem(
        tabela_img,
        caminho_base,
        figsize=(18.5, max(4.0, 0.80 * (len(tabela_img) + 1))),
        col_widths=[0.13, 0.08, 0.07, 0.07, 0.07, 0.07, 0.08, 0.07, 0.05, 0.06, 0.25],
        fontsize=7.4,
    )


def gerar_tabela_completa(recorte: str, avaliacao: str) -> dict:
    recorte = normalizar_recorte(recorte)
    df = carregar_metricas(recorte, avaliacao)
    tabela = preparar_tabela_completa(df)

    out_dir = appendix_logit_dir(recorte) / avaliacao / recorte
    out_dir.mkdir(parents=True, exist_ok=True)

    nome = f"tabela_logit_ternario_recorte_{recorte}_{avaliacao}"
    base = out_dir / nome

    csv_path = base.with_suffix(".csv")
    tabela.to_csv(csv_path, index=False, encoding="utf-8")

    out = {"tipo": "tabela_completa", "recorte": recorte, "avaliacao": avaliacao, "csv": str(csv_path)}
    out.update(
        salvar_tabela_latex(
            tabela,
            base,
            caption=f"Comparação de experimentos da regressão logística multinomial, recorte {recorte}, avaliação {avaliacao}.",
            label=f"tab:{nome}",
        )
    )
    out.update(renderizar_tabela_completa(tabela, base))

    log(f"[OK] Tabela completa gerada: {csv_path}")

    return out


def gerar_tabela_compacta(recorte: str, avaliacao: str) -> dict:
    recorte = normalizar_recorte(recorte)
    df = carregar_metricas(recorte, avaliacao)
    tabela = preparar_tabela_compacta(df)

    out_dir = appendix_logit_dir(recorte) / avaliacao / recorte
    out_dir.mkdir(parents=True, exist_ok=True)

    nome = f"tabela_compacta_logit_ternario_recorte_{recorte}_{avaliacao}"
    base = out_dir / nome

    csv_path = base.with_suffix(".csv")
    tabela.to_csv(csv_path, index=False, encoding="utf-8")

    out = {"tipo": "tabela_compacta", "recorte": recorte, "avaliacao": avaliacao, "csv": str(csv_path)}
    out.update(
        salvar_tabela_latex(
            tabela,
            base,
            caption=f"Resumo compacto das especificações da regressão logística multinomial, recorte {recorte}, avaliação {avaliacao}.",
            label=f"tab:{nome}",
        )
    )
    out.update(
        salvar_tabela_como_imagem(
            tabela,
            base,
            figsize=(7.2, 2.2),
            col_widths=[0.20, 0.18, 0.13, 0.13, 0.13, 0.13, 0.10],
            fontsize=9.2,
        )
    )

    log(f"[OK] Tabela compacta gerada: {csv_path}")

    return out


def matriz_confusao_da_linha(row: pd.Series) -> pd.DataFrame:
    matriz = []

    for real in [0, 1, 2]:
        linha = []
        for pred in [0, 1, 2]:
            linha.append(int(row[f"confusao_real_{real}_pred_{pred}"]))
        matriz.append(linha)

    return pd.DataFrame(
        matriz,
        index=[CLASS_LABELS[i] for i in [0, 1, 2]],
        columns=[CLASS_LABELS[i] for i in [0, 1, 2]],
    )


def gerar_matriz_confusao(recorte: str, avaliacao: str, experimento: str = "E5") -> dict:
    recorte = normalizar_recorte(recorte)
    df = carregar_metricas(recorte, avaliacao)
    linha = df[df["experimento"].eq(experimento)].copy()

    if linha.empty:
        raise ValueError(f"Experimento {experimento} não encontrado nas métricas de {avaliacao}.")

    matriz = matriz_confusao_da_linha(linha.iloc[0])

    out_dir = appendix_logit_dir(recorte) / avaliacao / recorte / "matriz_confusao"
    nome = f"matriz_confusao_logit_ternario_recorte_{recorte}_{avaliacao}_{experimento.lower()}"
    base = out_dir / nome
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base.with_suffix(".csv")
    matriz.to_csv(csv_path, encoding="utf-8")

    matriz_latex = matriz.reset_index().rename(columns={"index": "Classe real"})

    out = {"tipo": "matriz_confusao", "recorte": recorte, "avaliacao": avaliacao, "experimento": experimento, "csv": str(csv_path)}
    out.update(
        salvar_tabela_latex(
            matriz_latex,
            base,
            caption=f"Matriz de confusão da regressão logística multinomial, recorte {recorte}, experimento {experimento}.",
            label=f"tab:{nome}",
        )
    )
    out.update(renderizar_matriz_confusao(matriz, base))

    log(f"[OK] Matriz de confusão gerada: {csv_path}")

    return out


def run(recorte: str = "geral", avaliacao: str = "all", experimento: str = "E5") -> None:
    configurar_matplotlib()

    if recorte == "all":
        recortes = ["geral", "medicina"]
    else:
        recortes = [normalizar_recorte(recorte)]

    if avaliacao == "all":
        avaliacoes = ["in_sample", "holdout_80_20"]
    else:
        avaliacoes = [avaliacao]

    registros = []

    for recorte_atual in recortes:
        for avaliacao_atual in avaliacoes:
            registros.append(gerar_tabela_completa(recorte_atual, avaliacao_atual))
            registros.append(gerar_tabela_compacta(recorte_atual, avaliacao_atual))
            registros.append(gerar_matriz_confusao(recorte_atual, avaliacao_atual, experimento=experimento))

    resumo_path = LOGS_DIR / "article_apendice_logit_ternario_resumo.csv"
    pd.DataFrame(registros).to_csv(resumo_path, index=False, encoding="utf-8")

    log(f"[OK] Resumo salvo em: {resumo_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Gera tabelas e matriz de confusão do apêndice da regressão logística multinomial.")
    parser.add_argument("--recorte", choices=["geral", "medicina", "all"], default="geral")
    parser.add_argument("--avaliacao", choices=["in_sample", "holdout_80_20", "all"], default="all")
    parser.add_argument("--experimento", choices=["E1", "E2", "E3", "E4", "E5"], default="E5")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(recorte=args.recorte, avaliacao=args.avaliacao, experimento=args.experimento)
