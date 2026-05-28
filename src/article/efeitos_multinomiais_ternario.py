"""
Gera as figuras principais de probabilidades previstas dos modelos logísticos.

Saídas em reports/article/figures/:
    efeitos_multinomiais_recorte_geral/
    efeitos_multinomiais_recorte_medicina/

Recorte geral:
    Figura por desempenho:
        eixo X em algarismos romanos para os intervalos de desempenho;
        legenda no topo com faixas de renda por extenso;
        painéis: (a), (b), (c), (d).
    Figura por renda:
        eixo X em algarismos romanos para as faixas de renda;
        legenda no topo com intervalos de desempenho por extenso;
        painéis: (a), (b), (c), (d).

Recorte Medicina:
    mesma configuração visual, mas apenas com os três painéis do modelo multinomial:
        (a), (b), (c).

As figuras usam tons de cinza, fonte serifada compatível com Times New Roman,
títulos de painéis em negrito e legenda superior.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    from src.constants import ROOT_DIR, REPORTS_DIR, ABT_DIR
except Exception:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    REPORTS_DIR = ROOT_DIR / "reports"
    ABT_DIR = ROOT_DIR / "data" / "06_abt"


CLASSE_LABEL = {
    0: "Lista de espera",
    1: "Não contratado",
    2: "Contratada",
}

# Eixo X da figura por renda: I a VI. A explicação fica na nota da figura.
FAIXAS_RENDA = [
    ("I", "Até 600", -np.inf, 600.0, 300.0),
    ("II", "601–1.200", 600.0, 1200.0, 900.0),
    ("III", "1.201–1.800", 1200.0, 1800.0, 1500.0),
    ("IV", "1.801–2.400", 1800.0, 2400.0, 2100.0),
    ("V", "2.401–3.000", 2400.0, 3000.0, 2700.0),
    ("VI", "Acima de 3.000", 3000.0, np.inf, 3300.0),
]

# Eixo X da figura por desempenho: I a VI. A explicação fica na nota da figura.
FAIXAS_GAP = [
    ("I", "< -150", -np.inf, -150.0, -200.0),
    ("II", "-150 a -50", -150.0, -50.0, -100.0),
    ("III", "-50 a 0", -50.0, 0.0, -25.0),
    ("IV", "0 a +50", 0.0, 50.0, 25.0),
    ("V", "+50 a +150", 50.0, 150.0, 100.0),
    ("VI", "> +150", 150.0, np.inf, 200.0),
]

TONS_CINZA = ["0.12", "0.26", "0.40", "0.54", "0.68", "0.82"]
MARCADORES = ["o", "s", "^", "D", "P", "X"]
LINHAS = ["-", "--", "-.", ":", "-", "--"]


def configurar_matplotlib() -> None:
    import logging
    # Silencia os avisos chatos do font_manager do Matplotlib
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    plt.rcParams.update(
        {
            # Coloca a DejaVu Serif primeiro. Ela já vem embutida no Matplotlib e resolve o problema.
            "font.family": ["DejaVu Serif", "Times New Roman", "serif"],
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def caminho_abt(recorte: str) -> Path:
    candidatos = [
        ABT_DIR / f"abt_contratacao_ternaria_recorte_{recorte}.parquet",
        ABT_DIR / f"abt_contratacao_ternaria_{recorte}.parquet",
    ]
    for path in candidatos:
        if path.exists():
            return path
    raise FileNotFoundError(f"ABT ternária não encontrada para recorte={recorte}: {candidatos}")


def caminho_abt_binaria() -> Path:
    candidatos = [
        ABT_DIR / "abt_contratacao_binaria_recorte_geral.parquet",
        ABT_DIR / "abt_contratacao_binaria_geral.parquet",
        ABT_DIR / "abt_contratacao_binaria.parquet",
        ABT_DIR / "abt_binaria_recorte_geral.parquet",
        ABT_DIR / "abt_binaria_geral.parquet",
    ]
    for path in candidatos:
        if path.exists():
            return path

    achados = sorted(ABT_DIR.glob("*binaria*geral*.parquet")) + sorted(ABT_DIR.glob("*binaria*.parquet"))
    if achados:
        return achados[0]

    raise FileNotFoundError(f"ABT binária geral não encontrada em {ABT_DIR}.")


def caminho_modelo(recorte: str, avaliacao: str, experimento: str) -> Path:
    grupo = "general" if recorte == "geral" else "medicina"
    base = ROOT_DIR / "models" / grupo / "logit_ternario" / avaliacao

    candidatos = [
        base / f"logit_ternario_{experimento.lower()}.joblib",
        base / f"logit_ternario_{experimento.upper()}.joblib",
    ]
    for path in candidatos:
        if path.exists():
            return path

    achados = sorted(base.glob(f"*{experimento.lower()}*.joblib")) + sorted(base.glob(f"*{experimento.upper()}*.joblib"))
    if achados:
        return achados[0]

    raise FileNotFoundError(f"Modelo logit ternário não encontrado em {base} para {experimento}.")


def caminho_modelo_binario(avaliacao: str, experimento: str) -> Path:
    bases = [
        ROOT_DIR / "models" / "general" / "logit_binario" / avaliacao,
        ROOT_DIR / "models" / "geral" / "logit_binario" / avaliacao,
    ]

    for base in bases:
        candidatos = [
            base / f"logit_binario_{experimento.lower()}.joblib",
            base / f"logit_binario_{experimento.upper()}.joblib",
        ]
        for path in candidatos:
            if path.exists():
                return path

        achados = sorted(base.glob(f"*{experimento.lower()}*.joblib")) + sorted(base.glob(f"*{experimento.upper()}*.joblib"))
        if achados:
            return achados[0]

    achados = (
        sorted((ROOT_DIR / "models").glob(f"**/logit_binario/**/logit_binario_{experimento.lower()}.joblib"))
        + sorted((ROOT_DIR / "models").glob(f"**/logit_binario/**/*{experimento.lower()}*.joblib"))
        + sorted((ROOT_DIR / "models").glob(f"**/*binario*{experimento.lower()}*.joblib"))
    )
    if achados:
        return achados[0]

    raise FileNotFoundError(f"Modelo logit binário geral não encontrado para avaliação={avaliacao}, experimento={experimento}.")


def estimador_final(modelo: Any) -> Any:
    if hasattr(modelo, "steps"):
        return modelo.steps[-1][1]
    return modelo


def classes_modelo(modelo: Any) -> list[int]:
    est = estimador_final(modelo)
    classes = getattr(est, "classes_", None)
    if classes is None:
        classes = getattr(modelo, "classes_", None)
    if classes is None:
        return [0, 1, 2]

    saida: list[int] = []
    for c in classes:
        try:
            saida.append(int(c))
        except Exception:
            saida.append(c)
    return saida


def feature_names_modelo(modelo: Any, abt: pd.DataFrame) -> list[str]:
    nomes = getattr(modelo, "feature_names_in_", None)
    if nomes is not None:
        return [str(c) for c in nomes]

    if hasattr(modelo, "named_steps"):
        for step in modelo.named_steps.values():
            nomes = getattr(step, "feature_names_in_", None)
            if nomes is not None:
                return [str(c) for c in nomes]

    excluir = {
        "target",
        "y",
        "classe",
        "status",
        "situacao",
        "situacao_inscricao",
        "target_ternario",
        "target_binario",
    }
    return [c for c in abt.columns if c not in excluir]


def valor_tipico_coluna(serie: pd.Series) -> object:
    if pd.api.types.is_numeric_dtype(serie):
        valor = pd.to_numeric(serie, errors="coerce").median()
        return 0.0 if pd.isna(valor) else float(valor)

    moda = serie.dropna().mode()
    return "" if moda.empty else moda.iloc[0]


def mediana_faixa(df: pd.DataFrame, coluna: str, minimo: float, maximo: float, fallback: float) -> float:
    serie = pd.to_numeric(df[coluna], errors="coerce")
    mask = pd.Series(True, index=df.index)

    if np.isfinite(minimo):
        mask &= serie > minimo
    if np.isfinite(maximo):
        mask &= serie <= maximo

    valor = serie[mask].median()
    return fallback if pd.isna(valor) else float(valor)


def construir_grid(abt: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "renda_per_capita" not in abt.columns or "gap" not in abt.columns:
        raise KeyError("A ABT precisa conter as colunas renda_per_capita e gap.")

    base: dict[str, object] = {}
    for col in features:
        if col in abt.columns:
            base[col] = valor_tipico_coluna(abt[col])

    linhas = []
    for codigo_renda, label_renda, min_r, max_r, fallback_r in FAIXAS_RENDA:
        renda_valor = mediana_faixa(abt, "renda_per_capita", min_r, max_r, fallback_r)

        for codigo_gap, label_gap, min_g, max_g, fallback_g in FAIXAS_GAP:
            gap_valor = mediana_faixa(abt, "gap", min_g, max_g, fallback_g)

            row = dict(base)
            row["renda_per_capita"] = renda_valor
            row["gap"] = gap_valor
            if "renda_gap" in features:
                row["renda_gap"] = renda_valor * gap_valor

            row["_codigo_renda"] = codigo_renda
            row["_faixa_renda"] = label_renda
            row["_codigo_desempenho"] = codigo_gap
            row["_faixa_desempenho"] = label_gap
            linhas.append(row)

    grid = pd.DataFrame(linhas)

    for col in features:
        if col not in grid.columns:
            grid[col] = np.nan

    return grid, grid[features].copy()


def prever_probabilidades(modelo: Any, grid_meta: pd.DataFrame, X_grid: pd.DataFrame) -> pd.DataFrame:
    probas = modelo.predict_proba(X_grid)
    classes = classes_modelo(modelo)

    registros = []
    for i, row in grid_meta.iterrows():
        for j, classe in enumerate(classes):
            registros.append(
                {
                    "classe": int(classe),
                    "classe_label": CLASSE_LABEL.get(int(classe), str(classe)),
                    "codigo_renda": row["_codigo_renda"],
                    "faixa_renda": row["_faixa_renda"],
                    "codigo_desempenho": row["_codigo_desempenho"],
                    "faixa_desempenho": row["_faixa_desempenho"],
                    "probabilidade": float(probas[i, j]),
                }
            )

    return pd.DataFrame(registros)


def prever_probabilidade_binaria(modelo: Any, grid_meta: pd.DataFrame, X_grid: pd.DataFrame) -> pd.DataFrame:
    probas = modelo.predict_proba(X_grid)
    classes = classes_modelo(modelo)

    try:
        idx_pos = classes.index(1)
    except ValueError:
        idx_pos = probas.shape[1] - 1

    registros = []
    for i, row in grid_meta.iterrows():
        registros.append(
            {
                "classe": 1,
                "classe_label": "Contratada (binário)",
                "codigo_renda": row["_codigo_renda"],
                "faixa_renda": row["_faixa_renda"],
                "codigo_desempenho": row["_codigo_desempenho"],
                "faixa_desempenho": row["_faixa_desempenho"],
                "probabilidade": float(probas[i, idx_pos]),
            }
        )

    return pd.DataFrame(registros)


def probabilidade_binaria_geral(avaliacao: str, experimento: str) -> pd.DataFrame | None:
    try:
        abt_bin_path = caminho_abt_binaria()
        modelo_bin_path = caminho_modelo_binario(avaliacao, experimento)

        print(f"ABT binária:  {abt_bin_path}")
        print(f"Modelo bin.:  {modelo_bin_path}")

        modelo_bin = joblib.load(modelo_bin_path)
        abt_bin = pd.read_parquet(abt_bin_path)
        features_bin = feature_names_modelo(modelo_bin, abt_bin)

        grid_meta_bin, X_grid_bin = construir_grid(abt_bin, features_bin)
        return prever_probabilidade_binaria(modelo_bin, grid_meta_bin, X_grid_bin)

    except Exception as exc:
        print(f"[AVISO] Não foi possível incluir o painel binário no recorte geral: {exc}")
        return None


def ordenar_por_desempenho(df: pd.DataFrame) -> pd.DataFrame:
    ordem = {x[0]: i for i, x in enumerate(FAIXAS_GAP)}
    out = df.copy()
    out["ordem"] = out["codigo_desempenho"].map(ordem)
    return out.sort_values("ordem")


def ordenar_por_renda(df: pd.DataFrame) -> pd.DataFrame:
    ordem = {x[0]: i for i, x in enumerate(FAIXAS_RENDA)}
    out = df.copy()
    out["ordem"] = out["codigo_renda"].map(ordem)
    return out.sort_values("ordem")


def preparar_eixo(ax, titulo: str) -> None:
    ax.set_title(titulo, fontweight="bold", fontfamily="Times New Roman", pad=8)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="0.86", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_painel_por_desempenho(ax, df: pd.DataFrame, titulo: str) -> tuple[list[Any], list[str]]:
    preparar_eixo(ax, titulo)

    handles = []
    labels = []

    for idx, (_codigo_renda, label_renda, *_resto) in enumerate(FAIXAS_RENDA):
        sub = ordenar_por_desempenho(df[df["faixa_renda"].eq(label_renda)])
        linha, = ax.plot(
            sub["codigo_desempenho"],
            sub["probabilidade"] * 100,
            marker=MARCADORES[idx % len(MARCADORES)],
            linestyle=LINHAS[idx % len(LINHAS)],
            color=TONS_CINZA[idx % len(TONS_CINZA)],
            label=label_renda,
            linewidth=1.7,
            markersize=5.2,
        )
        handles.append(linha)
        labels.append(label_renda)

    ax.set_xlabel("")
    return handles, labels


def plot_painel_por_renda(ax, df: pd.DataFrame, titulo: str) -> tuple[list[Any], list[str]]:
    preparar_eixo(ax, titulo)

    handles = []
    labels = []

    for idx, (_codigo_gap, label_gap, *_resto) in enumerate(FAIXAS_GAP):
        sub = ordenar_por_renda(df[df["faixa_desempenho"].eq(label_gap)])
        linha, = ax.plot(
            sub["codigo_renda"],
            sub["probabilidade"] * 100,
            marker=MARCADORES[idx % len(MARCADORES)],
            linestyle=LINHAS[idx % len(LINHAS)],
            color=TONS_CINZA[idx % len(TONS_CINZA)],
            label=label_gap,
            linewidth=1.7,
            markersize=5.2,
        )
        handles.append(linha)
        labels.append(label_gap)

    ax.set_xlabel("")
    return handles, labels


def salvar_figura(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def legenda_superior(fig, handles: list[Any], labels: list[str], titulo: str, ncol: int = 6, y: float = 0.995) -> None:
    fig.legend(
        handles,
        labels,
        title=titulo,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        columnspacing=1.6,
        handlelength=2.2,
        borderaxespad=0.0,
    )


def figura_por_desempenho_todas_classes(
    prob: pd.DataFrame,
    out: Path,
    prob_binaria: pd.DataFrame | None = None,
    recorte: str = "geral",
) -> None:
    configurar_matplotlib()

    if recorte == "geral" and prob_binaria is not None:
        fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.9), sharey=True)

        paineis = [
            (axes[0, 0], prob[prob["classe"].eq(0)].copy(), "(a)"),
            (axes[0, 1], prob[prob["classe"].eq(1)].copy(), "(b)"),
            (axes[1, 0], prob[prob["classe"].eq(2)].copy(), "(c)"),
            (axes[1, 1], prob_binaria.copy(), "(d)"),
        ]

        handles, labels = [], []
        for i, (ax, df, titulo) in enumerate(paineis):
            h, l = plot_painel_por_desempenho(ax, df, titulo)
            if i == 0:
                handles, labels = h, l

        axes[0, 0].set_ylabel("Probabilidade prevista (%)")
        axes[1, 0].set_ylabel("Probabilidade prevista (%)")

        legenda_superior(fig, handles, labels, "Faixa de renda familiar per capita", ncol=6, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        salvar_figura(fig, out)
        return

    classes = sorted(prob["classe"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(5.8 * len(classes), 6.4), sharey=True)

    if len(classes) == 1:
        axes = [axes]

    titulos = {0: "(a)", 1: "(b)", 2: "(c)"}

    handles, labels = [], []
    for i, (ax, classe) in enumerate(zip(axes, classes)):
        df = prob[prob["classe"].eq(classe)].copy()
        h, l = plot_painel_por_desempenho(ax, df, titulos.get(int(classe), f"({chr(97+i)})"))
        if i == 0:
            handles, labels = h, l

    axes[0].set_ylabel("Probabilidade prevista (%)")
    legenda_superior(fig, handles, labels, "Faixa de renda familiar per capita", ncol=6, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    salvar_figura(fig, out)


def figura_por_renda_todas_classes(
    prob: pd.DataFrame,
    out: Path,
    prob_binaria: pd.DataFrame | None = None,
    recorte: str = "geral",
) -> None:
    configurar_matplotlib()

    if recorte == "geral" and prob_binaria is not None:
        fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.9), sharey=True)

        paineis = [
            (axes[0, 0], prob[prob["classe"].eq(0)].copy(), "(a)"),
            (axes[0, 1], prob[prob["classe"].eq(1)].copy(), "(b)"),
            (axes[1, 0], prob[prob["classe"].eq(2)].copy(), "(c)"),
            (axes[1, 1], prob_binaria.copy(), "(d)"),
        ]

        handles, labels = [], []
        for i, (ax, df, titulo) in enumerate(paineis):
            h, l = plot_painel_por_renda(ax, df, titulo)
            if i == 0:
                handles, labels = h, l

        axes[0, 0].set_ylabel("Probabilidade prevista (%)")
        axes[1, 0].set_ylabel("Probabilidade prevista (%)")

        legenda_superior(fig, handles, labels, "Desempenho relativo à nota de corte", ncol=6, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        salvar_figura(fig, out)
        return

    classes = sorted(prob["classe"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(5.8 * len(classes), 6.4), sharey=True)

    if len(classes) == 1:
        axes = [axes]

    titulos = {0: "(a)", 1: "(b)", 2: "(c)"}

    handles, labels = [], []
    for i, (ax, classe) in enumerate(zip(axes, classes)):
        df = prob[prob["classe"].eq(classe)].copy()
        h, l = plot_painel_por_renda(ax, df, titulos.get(int(classe), f"({chr(97+i)})"))
        if i == 0:
            handles, labels = h, l

    axes[0].set_ylabel("Probabilidade prevista (%)")
    legenda_superior(fig, handles, labels, "Desempenho relativo à nota de corte", ncol=6, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    salvar_figura(fig, out)


def run(recorte: str, avaliacao: str = "in_sample", experimento: str = "E5") -> None:
    abt_path = caminho_abt(recorte)
    model_path = caminho_modelo(recorte, avaliacao, experimento)

    print("=" * 80)
    print("EFEITOS POR PROBABILIDADES PREVISTAS")
    print("=" * 80)
    print(f"Recorte:     {recorte}")
    print(f"Avaliação:   {avaliacao}")
    print(f"Experimento: {experimento}")
    print(f"ABT:         {abt_path}")
    print(f"Modelo:      {model_path}")
    print("-" * 80)

    modelo = joblib.load(model_path)
    abt = pd.read_parquet(abt_path)
    features = feature_names_modelo(modelo, abt)

    grid_meta, X_grid = construir_grid(abt, features)
    prob = prever_probabilidades(modelo, grid_meta, X_grid)

    prob_binaria = None
    if recorte == "geral":
        prob_binaria = probabilidade_binaria_geral(avaliacao=avaliacao, experimento=experimento)

    fig_dir = REPORTS_DIR / "article" / "figures" / f"efeitos_multinomiais_recorte_{recorte}"
    table_dir_antiga = REPORTS_DIR / "article" / "tables" / (
        "secao_4_4_efeitos_multinomiais_recorte_geral"
        if recorte == "geral"
        else "secao_4_5_efeitos_multinomiais_recorte_medicina"
    )

    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    if table_dir_antiga.exists():
        shutil.rmtree(table_dir_antiga)

    fig_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"logit_ternario_recorte_{recorte}_{avaliacao}_{experimento.lower()}"

    figura_por_desempenho_todas_classes(
        prob,
        fig_dir / f"figura_probabilidade_por_desempenho_todas_classes_{prefix}",
        prob_binaria=prob_binaria,
        recorte=recorte,
    )
    figura_por_renda_todas_classes(
        prob,
        fig_dir / f"figura_probabilidade_por_renda_todas_classes_{prefix}",
        prob_binaria=prob_binaria,
        recorte=recorte,
    )

    print(f"[OK] Figuras: {fig_dir}")
    print("[OK] Concluído.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorte", choices=["geral", "medicina"], required=True)
    parser.add_argument("--avaliacao", choices=["in_sample", "holdout_80_20"], default="in_sample")
    parser.add_argument("--experimento", default="E5")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(recorte=args.recorte, avaliacao=args.avaliacao, experimento=args.experimento)
