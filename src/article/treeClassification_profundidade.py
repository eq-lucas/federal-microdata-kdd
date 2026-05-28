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
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

SAVE_DPI = 700

BINS_RENDA = [-np.inf, 600, 1200, 1800, 2400, 3000, np.inf]
LABELS_RENDA = ["I", "II", "III", "IV", "V", "VI"]
BINS_DESEMPENHO = [-np.inf, -150, -50, 0, 50, 150, np.inf]
LABELS_DESEMPENHO = ["< -150", "-150 a -50", "-50 a 0", "0 a +50", "+50 a +150", "> +150"]
ORDEM_DESEMPENHO_PLOT = ["> +150", "+50 a +150", "0 a +50", "-50 a 0", "-150 a -50", "< -150"]

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


def salvar_tabela_como_imagem(df_tabela: pd.DataFrame, caminho_base: Path, figsize=(8.0, 3.5), col_widths=None, fontsize=9.5) -> dict:
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
        else:
            celula.set_facecolor(COR_LINHA_1 if linha % 2 == 1 else COR_LINHA_2)
            if coluna == 0:
                celula.get_text().set_fontweight("bold")
                celula.get_text().set_ha("left")
    return salvar_figura(fig, caminho_base)


def preparar_base_probabilidades(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    out = df.copy()
    out["faixa_renda"] = pd.cut(out["renda_per_capita"], bins=BINS_RENDA, labels=LABELS_RENDA, ordered=True)
    out["faixa_desempenho"] = pd.cut(out["gap"], bins=BINS_DESEMPENHO, labels=LABELS_DESEMPENHO, ordered=True)
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
    return df_pred.groupby(["faixa_renda", "faixa_desempenho"], observed=True)[prob_col].mean().reset_index()


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
            ax.text(j, i, texto, ha="center", va="center", fontsize=10.5, fontweight="bold", color=cor)
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
        subset = tabela[tabela["faixa_renda"] == faixa_renda].set_index("faixa_desempenho").reindex(LABELS_DESEMPENHO).reset_index()
        ax.plot(x, subset[prob_col], linestyle=estilos[idx % len(estilos)], marker=marcadores[idx % len(marcadores)], linewidth=1.8, markersize=5.0, label=faixa_renda, color=str(0.15 + idx * 0.12))
    ax.set_xlabel("Desempenho acadêmico", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probabilidade prevista de contratação (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_DESEMPENHO, rotation=0, fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Faixa de renda", loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=True, fancybox=False, edgecolor="black", facecolor="white", fontsize=9.3, title_fontsize=9.8)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.16, top=0.78)
    return salvar_figura(fig, caminho_base)


def gerar_curvas_renda_por_desempenho(df_pred: pd.DataFrame, prob_col: str, caminho_base: Path) -> dict:
    tabela = tabela_media_prob_por_faixas(df_pred, prob_col)
    estilos = ["-", "--", "-.", ":", "-", "--"]
    marcadores = ["o", "s", "^", "D", "P", "X"]
    fig, ax = plt.subplots(figsize=(8.2, 5.3), dpi=300)
    x = np.arange(len(LABELS_RENDA))
    for idx, faixa_desempenho in enumerate(LABELS_DESEMPENHO):
        subset = tabela[tabela["faixa_desempenho"] == faixa_desempenho].set_index("faixa_renda").reindex(LABELS_RENDA).reset_index()
        ax.plot(x, subset[prob_col], linestyle=estilos[idx % len(estilos)], marker=marcadores[idx % len(marcadores)], linewidth=1.8, markersize=5.0, label=faixa_desempenho, color=str(0.15 + idx * 0.12))
    ax.set_xlabel("Faixa de renda familiar per capita", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probabilidade prevista de contratação (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_RENDA, rotation=0, fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Desempenho acadêmico", loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, fancybox=False, edgecolor="black", facecolor="white", fontsize=9.3, title_fontsize=9.8)
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
            ax.text(j, i, fmt_int(valor), ha="center", va="center", fontsize=13, fontweight="bold", color=cor)
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


def gerar_barra_importancias(df_imp: pd.DataFrame, caminho_base: Path, top_n: int = 20) -> dict:
    top = df_imp.sort_values("importancia_normalizada", ascending=False).head(top_n).iloc[::-1]
    labels = top["bloco_label"].astype(str).tolist()
    vals = top["importancia_normalizada"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.35 * len(top))), dpi=300)
    y = np.arange(len(top))
    ax.barh(y, vals, color="0.45")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.set_xlabel("Importância normalizada", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    for i, v in enumerate(vals):
        ax.text(v + max(vals.max() * 0.01, 0.001), i, f"{v:.3f}".replace(".", ","), va="center", fontsize=8.5)
    fig.subplots_adjust(left=0.38, right=0.985, bottom=0.12, top=0.98)
    return salvar_figura(fig, caminho_base)

import argparse
import json

import joblib

from src import constants as C
from src.modeling.treeClassification_profundidade_utils import (
    EXPERIMENTS,
    FEATURE_BLOCKS,
    TEST_SIZE,
    RANDOM_STATE,
    profile_prefix,
    validar_profundidade,
    abt_path,
    carregar_abt,
    criar_split,
    diagnostics_dir,
    models_dir,
    normalizar_recorte,
    normalizar_target,
    preparar_xy,
    target_config,
)

PROJECT_ROOT = getattr(C, "PROJECT_ROOT", PROJECT_ROOT_FOR_IMPORT)
LOGS_DIR = getattr(C, "LOGS_DIR", PROJECT_ROOT / "reports" / "logs")
FIGURES_DIR = getattr(C, "FIGURES_DIR", PROJECT_ROOT / "reports" / "article" / "figures")
TABLES_DIR = getattr(C, "TABLES_DIR", PROJECT_ROOT / "reports" / "article" / "tables")
APPENDIX_DIR = getattr(C, "APPENDIX_DIR", PROJECT_ROOT / "reports" / "article" / "appendix")


CURRENT_PROFUNDIDADE = 10


def set_profundidade(profundidade: int | str) -> int:
    global CURRENT_PROFUNDIDADE
    CURRENT_PROFUNDIDADE = validar_profundidade(profundidade)
    return CURRENT_PROFUNDIDADE


def target_suffix(target: str, recorte: str) -> str:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    return f"{profile_prefix(CURRENT_PROFUNDIDADE)}_{target}_recorte_{recorte}"


def figures_tree_dir(target: str, recorte: str) -> Path:
    return FIGURES_DIR / target_suffix(target, recorte)


def tables_tree_dir(target: str, recorte: str) -> Path:
    return TABLES_DIR / f"secao_{profile_prefix(CURRENT_PROFUNDIDADE)}_{normalizar_target(target)}_recorte_{normalizar_recorte(recorte)}"


def appendix_tree_dir(target: str, recorte: str) -> Path:
    return APPENDIX_DIR / "apendice_modelagem" / target_suffix(target, recorte)


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"article_{profile_prefix(CURRENT_PROFUNDIDADE)}.log"
    with path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")
    print(message)


def carregar_modelo_metadata_abt(target: str, recorte: str, avaliacao: str, experimento: str):
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    cfg = target_config(target, CURRENT_PROFUNDIDADE)
    model_path = models_dir(target, recorte, avaliacao, CURRENT_PROFUNDIDADE) / f"{cfg['model_dir_name']}_{experimento.lower()}.joblib"
    model_meta_path = models_dir(target, recorte, avaliacao, CURRENT_PROFUNDIDADE) / f"{cfg['model_dir_name']}_{experimento.lower()}_metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}. Rode primeiro a modelagem treeClassification correspondente."
        )
    if not model_meta_path.exists():
        raise FileNotFoundError(f"Metadados do modelo não encontrados: {model_meta_path}")
    modelo = joblib.load(model_path)
    with model_meta_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    abt, meta_abt = carregar_abt(target, recorte)
    log(f"[OK] Modelo carregado: {model_path}")
    log(f"[OK] ABT carregada: {abt_path(target, recorte)}")
    return modelo, metadata, abt


def carregar_metricas(target: str, recorte: str, avaliacao: str, experimento: str) -> pd.DataFrame:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    cfg = target_config(target, CURRENT_PROFUNDIDADE)
    path = diagnostics_dir(target, recorte, avaliacao, CURRENT_PROFUNDIDADE) / f"{cfg['model_dir_name']}_experimentos_metricas.csv"
    if not path.exists():
        raise FileNotFoundError(f"Métricas não encontradas: {path}")
    df = pd.read_csv(path)
    df = df[df["experimento"].eq(experimento)].copy()
    if df.empty:
        raise ValueError(f"Experimento {experimento} não encontrado em {path}")
    return df


def carregar_importancias(target: str, recorte: str, avaliacao: str, experimento: str) -> pd.DataFrame:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    cfg = target_config(target, CURRENT_PROFUNDIDADE)
    path = diagnostics_dir(target, recorte, avaliacao, CURRENT_PROFUNDIDADE) / f"{cfg['model_dir_name']}_importancias_agregadas.csv"
    if not path.exists():
        raise FileNotFoundError(f"Importâncias não encontradas: {path}")
    df = pd.read_csv(path)
    df = df[df["experimento"].eq(experimento)].copy()
    if df.empty:
        raise ValueError(f"Experimento {experimento} não encontrado em {path}")
    return df


def preparar_X_artigo(abt: pd.DataFrame, metadata: dict, target: str, avaliacao: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    blocos_usados = metadata["blocos_usados"]
    X, y = preparar_xy(abt, target, blocos_usados)
    if avaliacao == "in_sample":
        idx = np.arange(len(X))
    elif avaliacao == "holdout_80_20":
        _, idx, _ = criar_split(X, y, avaliacao)
    else:
        raise ValueError("avaliacao deve ser in_sample ou holdout_80_20.")
    return X.iloc[idx].copy(), abt.iloc[idx].copy(), y.iloc[idx].copy()


def anexar_probabilidades(modelo, metadata: dict, abt: pd.DataFrame, target: str, avaliacao: str) -> pd.DataFrame:
    X_eval, abt_eval, y_eval = preparar_X_artigo(abt, metadata, target, avaliacao)
    cfg = target_config(target, CURRENT_PROFUNDIDADE)
    classes = list(modelo.named_steps["modelo"].classes_)
    idx_contratada = classes.index(cfg["positive_class"])
    prob = modelo.predict_proba(X_eval)[:, idx_contratada] * 100
    df_pred = abt_eval[["renda_per_capita", "gap"]].copy()
    df_pred[cfg["target_col"]] = y_eval.to_numpy()
    df_pred["prob_contratacao"] = prob
    return preparar_base_probabilidades(df_pred, "prob_contratacao")


def montar_tabela_principal(metricas: pd.DataFrame, target: str) -> pd.DataFrame:
    row = metricas.iloc[0]
    linhas = [
        {"Medida": "Observações da ABT", "Valor": fmt_int(row["n_total_abt"])},
        {"Medida": "Observações de treino", "Valor": fmt_int(row["n_treino"])},
        {"Medida": "Observações de teste", "Valor": fmt_int(row["n_teste"])},
    ]
    if normalizar_target(target) == "binario":
        linhas.extend([
            {"Medida": "ROC-AUC", "Valor": fmt_float(row["roc_auc"])},
            {"Medida": "Acurácia balanceada", "Valor": fmt_float(row["balanced_accuracy"])},
            {"Medida": "F1", "Valor": fmt_float(row["f1"])},
        ])
    else:
        linhas.extend([
            {"Medida": "ROC-AUC OVR ponderado", "Valor": fmt_float(row["roc_auc_ovr_weighted"])},
            {"Medida": "Acurácia balanceada", "Valor": fmt_float(row["balanced_accuracy"])},
            {"Medida": "F1 macro", "Valor": fmt_float(row["f1_macro"])},
        ])
    linhas.extend([
        {"Medida": "Importância: renda", "Valor": fmt_float(row["importancia_renda_per_capita"])},
        {"Medida": "Importância: desempenho", "Valor": fmt_float(row["importancia_gap"])},
        {"Medida": "Importância: interação", "Valor": fmt_float(row["importancia_renda_gap"])},
        {"Medida": "Primeira divisão da árvore", "Valor": str(row["primeira_divisao"])},
        {"Medida": "Profundidade observada", "Valor": fmt_int(row["tree_depth_observada"])},
        {"Medida": "Folhas", "Valor": fmt_int(row["tree_n_leaves"])},
        {"Medida": "Blocos de variáveis", "Valor": fmt_int(row["blocos_variaveis_qtd"])},
    ])
    return pd.DataFrame(linhas)


def salvar_tabela_principal(metricas: pd.DataFrame, target: str, recorte: str, avaliacao: str, experimento: str) -> dict:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    tabela = montar_tabela_principal(metricas, target)
    nome = f"tabela_treeClassification_{target}_recorte_{recorte}_principal_{avaliacao}_{experimento.lower()}"
    base = tables_tree_dir(target, recorte) / nome
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    tabela.to_csv(csv_path, index=False, encoding="utf-8")
    out = {"tipo": "tabela_principal", "csv": str(csv_path)}
    out.update(salvar_tabela_latex(tabela, base, caption=f"Resumo da árvore de decisão, target {target}, recorte {recorte}.", label=f"tab:{nome}"))
    out.update(salvar_tabela_como_imagem(tabela, base, figsize=(7.6, 4.3), col_widths=[0.70, 0.30], fontsize=9.4))
    return out


def salvar_dados_probabilidades(df_pred: pd.DataFrame, target: str, recorte: str, avaliacao: str, experimento: str) -> dict:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    out_dir = tables_tree_dir(target, recorte)
    out_dir.mkdir(parents=True, exist_ok=True)
    nome = f"dados_probabilidades_{profile_prefix(CURRENT_PROFUNDIDADE)}_{target}_recorte_{recorte}_{avaliacao}_{experimento.lower()}"
    csv_path = out_dir / f"{nome}.csv"
    parquet_path = out_dir / f"{nome}.parquet"
    df_pred.to_csv(csv_path, index=False, encoding="utf-8")
    df_pred.to_parquet(parquet_path, index=False)
    return {"tipo": "dados_probabilidades", "csv": str(csv_path), "parquet": str(parquet_path)}


def gerar_figuras_probabilidades(df_pred: pd.DataFrame, target: str, recorte: str, avaliacao: str, experimento: str) -> list[dict]:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    registros = []
    nome_base = f"treeClassification_{target}_recorte_{recorte}_{avaliacao}_{experimento.lower()}"
    registros.append({"tipo": "heatmap_probabilidade_corpo", **gerar_heatmap_probabilidade(df_pred, "prob_contratacao", figures_tree_dir(target, recorte) / f"figura_probabilidade_prevista_contratacao_{nome_base}")})
    appendix_fig_dir = appendix_tree_dir(target, recorte) / avaliacao / recorte / "figuras_curvas"
    registros.append({"tipo": "figura_c1_curvas_desempenho_por_renda", **gerar_curvas_desempenho_por_renda(df_pred, "prob_contratacao", appendix_fig_dir / f"figura_C1_probabilidade_por_desempenho_faixa_renda_{nome_base}")})
    registros.append({"tipo": "figura_c2_curvas_renda_por_desempenho", **gerar_curvas_renda_por_desempenho(df_pred, "prob_contratacao", appendix_fig_dir / f"figura_C2_probabilidade_por_renda_desempenho_{nome_base}")})
    return registros


def gerar_figura_importancias(importancias: pd.DataFrame, target: str, recorte: str, avaliacao: str, experimento: str) -> dict:
    nome = f"figura_importancias_treeClassification_{normalizar_target(target)}_recorte_{normalizar_recorte(recorte)}_{avaliacao}_{experimento.lower()}"
    return {"tipo": "figura_importancias", **gerar_barra_importancias(importancias, figures_tree_dir(target, recorte) / nome)}


def gerar_figura_arvore(modelo, target: str, recorte: str, avaliacao: str, experimento: str) -> dict:
    nome = f"figura_arvore_{profile_prefix(CURRENT_PROFUNDIDADE)}_{normalizar_target(target)}_recorte_{normalizar_recorte(recorte)}_{avaliacao}_{experimento.lower()}"
    cfg = target_config(target, CURRENT_PROFUNDIDADE)
    class_names = [cfg["class_labels"][int(c)] for c in modelo.named_steps["modelo"].classes_]
    feature_names = list(modelo.named_steps["preprocessador"].get_feature_names_out())
    fig = plt.figure(figsize=(18, 8), dpi=300)
    ax = fig.add_subplot(111)
    plot_tree(
        modelo.named_steps["modelo"],
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        max_depth=3,
        fontsize=6,
        ax=ax,
    )
    return {"tipo": "figura_arvore_depth3", **salvar_figura(fig, figures_tree_dir(target, recorte) / nome)}


def run(target: str = "binario", recorte: str = "geral", avaliacao: str = "in_sample", experimento: str = "E5", profundidade: int | str = 10) -> None:
    configurar_matplotlib()
    set_profundidade(profundidade)
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    experimento = experimento.upper()
    if experimento not in {exp["id"] for exp in EXPERIMENTS}:
        raise ValueError(f"Experimento inválido: {experimento}")
    log("=" * 80)
    log(f"ARTICLE: {profile_prefix(CURRENT_PROFUNDIDADE).upper()} | {target.upper()} | {recorte.upper()} | {avaliacao.upper()} | {experimento}")
    log("=" * 80)
    modelo, metadata, abt = carregar_modelo_metadata_abt(target, recorte, avaliacao, experimento)
    metricas = carregar_metricas(target, recorte, avaliacao, experimento)
    importancias = carregar_importancias(target, recorte, avaliacao, experimento)
    df_pred = anexar_probabilidades(modelo, metadata, abt, target, avaliacao)
    registros = [
        salvar_dados_probabilidades(df_pred, target, recorte, avaliacao, experimento),
        salvar_tabela_principal(metricas, target, recorte, avaliacao, experimento),
        gerar_figura_importancias(importancias, target, recorte, avaliacao, experimento),
        gerar_figura_arvore(modelo, target, recorte, avaliacao, experimento),
    ]
    registros.extend(gerar_figuras_probabilidades(df_pred, target, recorte, avaliacao, experimento))
    resumo_path = LOGS_DIR / f"article_{profile_prefix(CURRENT_PROFUNDIDADE)}_{target}_recorte_{recorte}_{avaliacao}_{experimento.lower()}_resumo.csv"
    pd.DataFrame(registros).to_csv(resumo_path, index=False, encoding="utf-8")
    log(f"[OK] Saídas do corpo do artigo: {figures_tree_dir(target, recorte)} e {tables_tree_dir(target, recorte)}")
    log(f"[OK] Saídas de apêndice: {appendix_tree_dir(target, recorte) / avaliacao / recorte}")
    log(f"[OK] Resumo salvo em: {resumo_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Gera produtos do corpo do artigo para árvore de decisão classificatória.")
    parser.add_argument("--target", choices=["binario", "ternario"], default="binario")
    parser.add_argument("--recorte", choices=["geral", "medicina"], default="geral")
    parser.add_argument("--avaliacao", choices=["in_sample", "holdout_80_20"], default="in_sample")
    parser.add_argument("--experimento", choices=["E1", "E2", "E3", "E4", "E5"], default="E5")
    parser.add_argument("--profundidade", type=int, choices=[10, 14, 19], default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(target=args.target, recorte=args.recorte, avaliacao=args.avaliacao, experimento=args.experimento, profundidade=args.profundidade)
