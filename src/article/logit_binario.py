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
import json

import joblib
from sklearn.model_selection import train_test_split

from src import constants as C
from src.modeling.logit_binario_utils import (
    ABT_BINARIA_GERAL_PATH,
    ABT_BINARIA_MEDICINA_PATH,
    FEATURE_BLOCKS,
    EXPERIMENTS,
    RANDOM_STATE,
    TEST_SIZE,
    diagnostics_dir,
    models_dir,
    limpar_coluna_categorica,
    limpar_coluna_numerica,
)


PROJECT_ROOT = getattr(C, "PROJECT_ROOT", PROJECT_ROOT_FOR_IMPORT)
LOGS_DIR = getattr(C, "LOGS_DIR", PROJECT_ROOT / "reports" / "logs")
FIGURES_DIR = getattr(C, "FIGURES_DIR", PROJECT_ROOT / "reports" / "article" / "figures")
TABLES_DIR = getattr(C, "TABLES_DIR", PROJECT_ROOT / "reports" / "article" / "tables")
APPENDIX_DIR = getattr(C, "APPENDIX_DIR", PROJECT_ROOT / "reports" / "article" / "appendix")

FIGURES_LOGIT_DIR = FIGURES_DIR / "logit_binario"
TABLES_LOGIT_DIR = TABLES_DIR / "secao_4_4"
APPENDIX_LOGIT_DIR = APPENDIX_DIR / "apendice_modelagem" / "logit_binario"


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "article_logit_binario.log"

    with path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def abt_path(recorte: str) -> Path:
    if recorte == "geral":
        return ABT_BINARIA_GERAL_PATH

    if recorte == "medicina":
        return ABT_BINARIA_MEDICINA_PATH

    raise ValueError("recorte deve ser 'geral' ou 'medicina'.")


def carregar_modelo_metadata_abt(recorte: str, avaliacao: str, experimento: str):
    model_path = models_dir(recorte, avaliacao) / f"logit_binario_{experimento.lower()}.joblib"
    model_meta_path = models_dir(recorte, avaliacao) / f"logit_binario_{experimento.lower()}_metadata.json"
    abt_file = abt_path(recorte)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}. "
            f"Rode primeiro: python3 main.py modeling fit {recorte if recorte != 'geral' else 'general'} --force"
        )

    if not model_meta_path.exists():
        raise FileNotFoundError(f"Metadados do modelo não encontrados: {model_meta_path}")

    if not abt_file.exists():
        raise FileNotFoundError(f"ABT não encontrada: {abt_file}")

    modelo = joblib.load(model_path)

    with model_meta_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    abt = pd.read_parquet(abt_file)

    log(f"[OK] Modelo carregado: {model_path}")
    log(f"[OK] ABT carregada: {abt_file}")

    return modelo, metadata, abt


def carregar_metricas(recorte: str, avaliacao: str, experimento: str) -> pd.DataFrame:
    path = diagnostics_dir(recorte, avaliacao) / "logit_binario_experimentos_metricas.csv"

    if not path.exists():
        raise FileNotFoundError(f"Métricas não encontradas: {path}")

    df = pd.read_csv(path)
    df = df[df["experimento"].eq(experimento)].copy()

    if df.empty:
        raise ValueError(f"Experimento {experimento} não encontrado em {path}")

    return df


def preparar_X(abt: pd.DataFrame, blocos_usados: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=abt.index)

    for coluna in blocos_usados:
        if FEATURE_BLOCKS[coluna]["type"] == "numeric":
            X[coluna] = limpar_coluna_numerica(abt[coluna])
        else:
            X[coluna] = limpar_coluna_categorica(abt[coluna])

    return X


def selecionar_base_para_probabilidades(abt: pd.DataFrame, avaliacao: str) -> tuple[pd.DataFrame, str]:
    """
    Define a base usada para gerar probabilidades previstas no artigo.

    - in_sample: usa a ABT inteira, pois o modelo foi ajustado e avaliado na própria base.
    - holdout_80_20: usa apenas os 20% de teste reconstruídos com o mesmo
      split estratificado usado na modelagem.
    """
    if avaliacao == "in_sample":
        return abt.copy(), "base_completa_in_sample"

    if avaliacao == "holdout_80_20":
        if "target_binario" not in abt.columns:
            raise ValueError("A ABT não possui target_binario para reconstruir o holdout.")

        y = pd.to_numeric(abt["target_binario"], errors="raise").astype(int)
        indices = np.arange(len(abt))

        _, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
            shuffle=True,
        )

        return abt.iloc[test_idx].copy(), "teste_holdout_20"

    raise ValueError("avaliacao deve ser 'in_sample' ou 'holdout_80_20'.")


def anexar_probabilidades(modelo, metadata: dict, abt: pd.DataFrame, avaliacao: str) -> pd.DataFrame:
    blocos_usados = metadata["blocos_usados"]
    abt_prob, amostra_probabilidade = selecionar_base_para_probabilidades(abt, avaliacao)
    X = preparar_X(abt_prob, blocos_usados)

    classes = list(modelo.named_steps["modelo"].classes_)
    indice_contratada = classes.index(1)

    prob = modelo.predict_proba(X)[:, indice_contratada] * 100

    df_pred = abt_prob[["renda_per_capita", "gap", "target_binario"]].copy()
    df_pred["prob_contratacao"] = prob
    df_pred["avaliacao"] = avaliacao
    df_pred["amostra_probabilidade"] = amostra_probabilidade

    df_pred = preparar_base_probabilidades(df_pred, "prob_contratacao")

    return df_pred


def montar_tabela_principal(metricas: pd.DataFrame) -> pd.DataFrame:
    row = metricas.iloc[0]

    tabela = pd.DataFrame(
        [
            {"Medida": "Observações da ABT", "Valor": fmt_int(row["n_total_abt"])},
            {"Medida": "Observações de treino", "Valor": fmt_int(row["n_treino"])},
            {"Medida": "Observações de teste", "Valor": fmt_int(row["n_teste"])},
            {"Medida": "ROC-AUC", "Valor": fmt_float(row["roc_auc"])},
            {"Medida": "Acurácia balanceada", "Valor": fmt_float(row["balanced_accuracy"])},
            {"Medida": "F1", "Valor": fmt_float(row["f1"])},
            {"Medida": "Coeficiente padronizado: renda", "Valor": fmt_float(row["coef_renda_per_capita"])},
            {"Medida": "Coeficiente padronizado: desempenho", "Valor": fmt_float(row["coef_gap"])},
            {"Medida": "Coeficiente padronizado: interação", "Valor": fmt_float(row["coef_renda_gap"])},
            {"Medida": "Blocos de variáveis", "Valor": fmt_int(row["blocos_variaveis_qtd"])},
        ]
    )

    return tabela


def salvar_tabela_principal(metricas: pd.DataFrame, recorte: str, avaliacao: str, experimento: str) -> dict:
    tabela = montar_tabela_principal(metricas)
    nome = f"tabela_logit_binario_principal_{recorte}_{avaliacao}_{experimento.lower()}"
    base = TABLES_LOGIT_DIR / nome

    csv_path = base.with_suffix(".csv")
    tabela.to_csv(csv_path, index=False, encoding="utf-8")

    out = {"tipo": "tabela_principal", "csv": str(csv_path)}
    out.update(
        salvar_tabela_latex(
            tabela,
            base,
            caption="Resumo do modelo logístico binário para contratação efetiva.",
            label=f"tab:{nome}",
        )
    )
    out.update(
        salvar_tabela_como_imagem(
            tabela,
            base,
            figsize=(7.2, 3.4),
            col_widths=[0.70, 0.30],
            fontsize=10.0,
        )
    )

    return out


def salvar_dados_probabilidades(df_pred: pd.DataFrame, recorte: str, avaliacao: str, experimento: str) -> dict:
    TABLES_LOGIT_DIR.mkdir(parents=True, exist_ok=True)

    nome = f"dados_probabilidades_logit_binario_{recorte}_{avaliacao}_{experimento.lower()}"
    csv_path = TABLES_LOGIT_DIR / f"{nome}.csv"
    parquet_path = TABLES_LOGIT_DIR / f"{nome}.parquet"

    df_pred.to_csv(csv_path, index=False, encoding="utf-8")
    df_pred.to_parquet(parquet_path, index=False)

    return {"tipo": "dados_probabilidades", "csv": str(csv_path), "parquet": str(parquet_path)}


def gerar_matriz_confusao_apendice(metricas: pd.DataFrame, recorte: str, avaliacao: str, experimento: str) -> dict:
    row = metricas.iloc[0]

    matriz = pd.DataFrame(
        [
            [int(row["tn"]), int(row["fp"])],
            [int(row["fn"]), int(row["tp"])],
        ],
        index=["Não contratado", "Contratada"],
        columns=["Não contratado", "Contratada"],
    )

    out_dir = APPENDIX_LOGIT_DIR / avaliacao / recorte / "matriz_confusao"
    nome = f"matriz_confusao_logit_binario_{recorte}_{avaliacao}_{experimento.lower()}"
    base = out_dir / nome

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base.with_suffix(".csv")
    matriz.to_csv(csv_path, encoding="utf-8")

    matriz_latex = matriz.reset_index().rename(columns={"index": "Classe real"})

    out = {"tipo": "matriz_confusao", "csv": str(csv_path)}
    out.update(
        salvar_tabela_latex(
            matriz_latex,
            base,
            caption=f"Matriz de confusão da regressão logística binária, experimento {experimento}.",
            label=f"tab:{nome}",
        )
    )
    out.update(renderizar_matriz_confusao(matriz, base))

    return out


def gerar_figuras_probabilidades(df_pred: pd.DataFrame, recorte: str, avaliacao: str, experimento: str) -> list[dict]:
    registros = []

    nome_base = f"logit_binario_{recorte}_{avaliacao}_{experimento.lower()}"

    registros.append(
        {
            "tipo": "heatmap_probabilidade_corpo",
            **gerar_heatmap_probabilidade(
                df_pred,
                "prob_contratacao",
                FIGURES_LOGIT_DIR / f"figura_probabilidade_prevista_contratacao_{nome_base}",
            ),
        }
    )

    appendix_fig_dir = APPENDIX_LOGIT_DIR / avaliacao / recorte / "figuras_curvas"
    registros.append(
        {
            "tipo": "figura_c1_curvas_desempenho_por_renda",
            **gerar_curvas_desempenho_por_renda(
                df_pred,
                "prob_contratacao",
                appendix_fig_dir / f"figura_C1_probabilidade_por_desempenho_faixa_renda_{nome_base}",
            ),
        }
    )

    registros.append(
        {
            "tipo": "figura_c2_curvas_renda_por_desempenho",
            **gerar_curvas_renda_por_desempenho(
                df_pred,
                "prob_contratacao",
                appendix_fig_dir / f"figura_C2_probabilidade_por_renda_desempenho_{nome_base}",
            ),
        }
    )

    return registros


def run(recorte: str = "geral", avaliacao: str = "in_sample", experimento: str = "E5") -> None:
    configurar_matplotlib()

    experimento = experimento.upper()

    if experimento not in {exp["id"] for exp in EXPERIMENTS}:
        raise ValueError(f"Experimento inválido: {experimento}")

    log("=" * 80)
    log(f"ARTICLE: LOGIT BINÁRIO | {recorte.upper()} | {avaliacao.upper()} | {experimento}")
    log("=" * 80)

    modelo, metadata, abt = carregar_modelo_metadata_abt(recorte, avaliacao, experimento)
    metricas = carregar_metricas(recorte, avaliacao, experimento)
    df_pred = anexar_probabilidades(modelo, metadata, abt, avaliacao)

    registros = [
        salvar_dados_probabilidades(df_pred, recorte, avaliacao, experimento),
        salvar_tabela_principal(metricas, recorte, avaliacao, experimento),
        gerar_matriz_confusao_apendice(metricas, recorte, avaliacao, experimento),
    ]
    registros.extend(gerar_figuras_probabilidades(df_pred, recorte, avaliacao, experimento))

    resumo_path = LOGS_DIR / f"article_logit_binario_{recorte}_{avaliacao}_{experimento.lower()}_resumo.csv"
    pd.DataFrame(registros).to_csv(resumo_path, index=False, encoding="utf-8")

    log(f"[OK] Saídas do corpo do artigo: {FIGURES_LOGIT_DIR} e {TABLES_LOGIT_DIR}")
    log(f"[OK] Saídas de apêndice: {APPENDIX_LOGIT_DIR / avaliacao / recorte}")
    log(f"[OK] Resumo salvo em: {resumo_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Gera produtos da regressão logística binária.")
    parser.add_argument("--recorte", choices=["geral", "medicina"], default="geral")
    parser.add_argument("--avaliacao", choices=["in_sample", "holdout_80_20"], default="in_sample")
    parser.add_argument("--experimento", choices=["E1", "E2", "E3", "E4", "E5"], default="E5")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(recorte=args.recorte, avaliacao=args.avaliacao, experimento=args.experimento)
