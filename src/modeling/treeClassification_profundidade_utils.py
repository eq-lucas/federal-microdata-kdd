from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from src import constants as C


PROJECT_ROOT = getattr(C, "PROJECT_ROOT", PROJECT_ROOT_FOR_IMPORT)
LOGS_DIR = getattr(C, "LOGS_DIR", PROJECT_ROOT / "reports" / "logs")
ABT_DIR = getattr(C, "ABT_DIR", PROJECT_ROOT / "data" / "06_abt")
DIAGNOSTICS_DIR = getattr(C, "DIAGNOSTICS_DIR", PROJECT_ROOT / "reports" / "diagnostics")
MODELS_GENERAL_DIR = getattr(C, "MODELS_GENERAL_DIR", PROJECT_ROOT / "models" / "general")
MODELS_MEDICINA_DIR = getattr(C, "MODELS_MEDICINA_DIR", PROJECT_ROOT / "models" / "medicina")

ABT_BINARIA_GERAL_PATH = getattr(C, "ABT_BINARIA_GERAL_PATH", ABT_DIR / "abt_contratacao_binaria_geral.parquet")
ABT_BINARIA_MEDICINA_PATH = getattr(C, "ABT_BINARIA_MEDICINA_PATH", ABT_DIR / "abt_contratacao_binaria_medicina.parquet")
ABT_TERNARIA_GERAL_PATH = getattr(C, "ABT_TERNARIA_GERAL_PATH", ABT_DIR / "abt_contratacao_ternaria_recorte_geral.parquet")
ABT_TERNARIA_MEDICINA_PATH = getattr(C, "ABT_TERNARIA_MEDICINA_PATH", ABT_DIR / "abt_contratacao_ternaria_recorte_medicina.parquet")

RANDOM_STATE = 42
TEST_SIZE = 0.20
IMPORTANCIA_TOL = 1e-6

TREE_DEPTH_PROFILES = {
    10: {
        "prefix": "treeClassification_10_profundidade",
        "max_depth": 10,
        "min_samples_leaf_min": 400,
        "min_samples_split_min": 800,
        "description": "Árvore de decisão com profundidade máxima 10 e pré-poda mínima de 400 amostras por folha.",
    },
    14: {
        "prefix": "treeClassification_14_profundidade",
        "max_depth": 14,
        "min_samples_leaf_min": 400,
        "min_samples_split_min": 800,
        "description": "Árvore de decisão com profundidade máxima 14 e pré-poda mínima de 400 amostras por folha.",
    },
    19: {
        "prefix": "treeClassification_19_profundidade",
        "max_depth": 19,
        "min_samples_leaf_min": 400,
        "min_samples_split_min": 800,
        "description": "Árvore de decisão com profundidade máxima 19 e pré-poda mínima de 400 amostras por folha.",
    },
}


def validar_profundidade(profundidade: int | str) -> int:
    profundidade = int(profundidade)
    if profundidade not in TREE_DEPTH_PROFILES:
        raise ValueError("profundidade deve ser 10, 14 ou 19.")
    return profundidade


def profile_prefix(profundidade: int | str) -> str:
    return TREE_DEPTH_PROFILES[validar_profundidade(profundidade)]["prefix"]


def model_dir_name_com_profundidade(target: str, profundidade: int | str) -> str:
    target = normalizar_target(target)
    return f"{profile_prefix(profundidade)}_{target}"



TARGET_CONFIG = {
    "binario": {
        "target_col": "target_binario",
        "classes": [0, 1],
        "positive_class": 1,
        "class_labels": {
            0: "Não contratado",
            1: "Contratada",
        },
        "model_dir_name": "treeClassification_binario",
        "model_description": "Árvore de decisão classificatória binária",
    },
    "ternario": {
        "target_col": "target_ternario",
        "classes": [0, 1, 2],
        "positive_class": 2,
        "class_labels": {
            0: "Lista de espera",
            1: "Não contratado",
            2: "Contratada",
        },
        "model_dir_name": "treeClassification_ternario",
        "model_description": "Árvore de decisão classificatória ternária",
    },
}

FEATURE_BLOCKS = {
    "renda_per_capita": {"label": "Renda familiar per capita", "type": "numeric"},
    "gap": {"label": "Desempenho relativo à nota de corte", "type": "numeric"},
    "renda_gap": {"label": "Interação renda × desempenho", "type": "numeric"},
    "idade": {"label": "Idade", "type": "numeric"},
    "nota_corte_gp": {"label": "Nota de corte do grupo", "type": "numeric"},
    "opcao_curso": {"label": "Opção de curso", "type": "categorical"},
    "ano": {"label": "Ano do processo seletivo", "type": "categorical"},
    "semestre": {"label": "Semestre do processo seletivo", "type": "categorical"},
    "conceito_curso_gp": {"label": "Conceito do curso", "type": "categorical"},
    "turno": {"label": "Turno", "type": "categorical"},
    "ensino_medio_escola_publica": {"label": "Ensino médio em escola pública", "type": "categorical"},
    "regiao_ies_alvo": {"label": "Região da oferta", "type": "categorical"},
    "natureza_juridica_mantenedora": {"label": "Natureza jurídica da mantenedora", "type": "categorical"},
    "etnia_cor": {"label": "Cor/raça ou etnia", "type": "categorical"},
    "sexo": {"label": "Sexo", "type": "categorical"},
    "regiao_morar": {"label": "Região de residência", "type": "categorical"},
    "organizacao_academica": {"label": "Organização acadêmica", "type": "categorical"},
    "subarea_conhecimento": {"label": "Subárea de conhecimento", "type": "categorical"},
    "concluiu_curso_superior": {"label": "Concluiu curso superior", "type": "categorical"},
    "beneficiado_creduc_fies": {"label": "Beneficiado CREDUC/FIES", "type": "categorical"},
    "uf_local_oferta": {"label": "UF do local de oferta", "type": "categorical"},
}

CORE_BLOCKS = ["renda_per_capita", "gap", "renda_gap"]

EXPERIMENTS = [
    {
        "id": "E1",
        "name": "Modelo mínimo substantivo",
        "description": "Renda, desempenho acadêmico relativo e interação entre renda e desempenho.",
        "blocks": CORE_BLOCKS,
    },
    {
        "id": "E2",
        "name": "Modelo mínimo + 3 controles",
        "description": "E1 acrescido de idade, nota de corte e opção de curso.",
        "blocks": CORE_BLOCKS + ["idade", "nota_corte_gp", "opcao_curso"],
    },
    {
        "id": "E3",
        "name": "Modelo intermediário",
        "description": "E2 acrescido de ano, semestre, conceito do curso, turno e ensino médio público.",
        "blocks": CORE_BLOCKS + [
            "idade",
            "nota_corte_gp",
            "opcao_curso",
            "ano",
            "semestre",
            "conceito_curso_gp",
            "turno",
            "ensino_medio_escola_publica",
        ],
    },
    {
        "id": "E4",
        "name": "Modelo expandido",
        "description": "E3 acrescido de subárea de conhecimento, região da oferta, natureza jurídica e cor/raça ou etnia.",
        "blocks": CORE_BLOCKS + [
            "idade",
            "nota_corte_gp",
            "opcao_curso",
            "ano",
            "semestre",
            "conceito_curso_gp",
            "turno",
            "ensino_medio_escola_publica",
            "subarea_conhecimento",
            "regiao_ies_alvo",
            "natureza_juridica_mantenedora",
            "etnia_cor",
        ],
    },
    {
        "id": "E5",
        "name": "Modelo completo interpretável",
        "description": "E4 acrescido de sexo, região de residência, organização acadêmica, conclusão de curso superior, benefício anterior CREDUC/FIES e UF da oferta.",
        "blocks": list(FEATURE_BLOCKS.keys()),
    },
]


def log(message: str, log_file: str = "treeClassification_experimentos.log") -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / log_file
    with path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")
    print(message)


def normalizar_recorte(recorte: str) -> str:
    mapa = {"general": "geral", "geral": "geral", "medicina": "medicina"}
    if recorte not in mapa:
        raise ValueError("recorte deve ser 'geral'/'general' ou 'medicina'.")
    return mapa[recorte]


def normalizar_target(target: str) -> str:
    mapa = {
        "binaria": "binario",
        "binario": "binario",
        "binary": "binario",
        "ternaria": "ternario",
        "ternario": "ternario",
        "multiclass": "ternario",
    }
    if target not in mapa:
        raise ValueError("target deve ser 'binario'/'binaria' ou 'ternario'/'ternaria'.")
    return mapa[target]


def target_config(target: str, profundidade: int | str | None = None) -> dict:
    target = normalizar_target(target)
    cfg = dict(TARGET_CONFIG[target])

    if profundidade is not None:
        profundidade = validar_profundidade(profundidade)
        cfg["model_dir_name"] = model_dir_name_com_profundidade(target, profundidade)
        cfg["model_description"] = (
            f"{TARGET_CONFIG[target]['model_description']} | "
            f"profundidade máxima {profundidade} | "
            "pré-poda por min_samples_leaf/min_samples_split"
        )
        cfg["profundidade"] = profundidade
        cfg["profile_prefix"] = profile_prefix(profundidade)

    return cfg


def abt_path(target: str, recorte: str) -> Path:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)

    if target == "binario" and recorte == "geral":
        return ABT_BINARIA_GERAL_PATH
    if target == "binario" and recorte == "medicina":
        return ABT_BINARIA_MEDICINA_PATH
    if target == "ternario" and recorte == "geral":
        return ABT_TERNARIA_GERAL_PATH
    if target == "ternario" and recorte == "medicina":
        return ABT_TERNARIA_MEDICINA_PATH

    raise ValueError("Combinação inválida de target e recorte.")


def metadata_path(target: str, recorte: str) -> Path:
    path = abt_path(target, recorte)
    return path.with_name(path.stem + "_metadata.json")


def models_dir(target: str, recorte: str, avaliacao: str, profundidade: int | str) -> Path:
    cfg = target_config(target, profundidade)
    recorte = normalizar_recorte(recorte)

    root = MODELS_GENERAL_DIR if recorte == "geral" else MODELS_MEDICINA_DIR
    return root / cfg["model_dir_name"] / avaliacao


def diagnostics_dir(target: str, recorte: str, avaliacao: str, profundidade: int | str) -> Path:
    cfg = target_config(target, profundidade)
    recorte = normalizar_recorte(recorte)
    return DIAGNOSTICS_DIR / "modeling" / cfg["model_dir_name"] / avaliacao / recorte


def carregar_abt(target: str, recorte: str) -> tuple[pd.DataFrame, dict]:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    path = abt_path(target, recorte)
    meta_path = metadata_path(target, recorte)

    if not path.exists():
        raise FileNotFoundError(
            f"ABT não encontrada: {path}. Rode primeiro a ABT correspondente: "
            f"python3 main.py abt {'binaria' if target == 'binario' else 'ternaria'} --recorte {recorte}"
        )

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadados da ABT não encontrados: {meta_path}")

    df = pd.read_parquet(path)
    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    return df, meta


def limpar_coluna_categorica(serie: pd.Series) -> pd.Series:
    def limpar_valor(valor):
        if pd.isna(valor):
            return np.nan
        texto = str(valor).strip()
        if texto in {"", "nan", "NaN", "None", "NONE", "NULL", "<NA>", "NA", "N/A", "-", "--"}:
            return np.nan
        return texto

    return serie.map(limpar_valor).astype(object)


def limpar_coluna_numerica(serie: pd.Series) -> pd.Series:
    out = pd.to_numeric(serie, errors="coerce").astype("float64")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def bloco_valido(df: pd.DataFrame, bloco: str) -> bool:
    if bloco not in df.columns:
        return False

    tipo = FEATURE_BLOCKS[bloco]["type"]
    if tipo == "numeric":
        serie = limpar_coluna_numerica(df[bloco])
        return bool(serie.notna().any())

    serie = limpar_coluna_categorica(df[bloco])
    return bool(serie.notna().any() and serie.nunique(dropna=True) >= 2)


def blocos_validos_para_experimento(df: pd.DataFrame, experimento: dict) -> tuple[list[str], list[str]]:
    usados = []
    ausentes = []
    for bloco in experimento["blocks"]:
        if bloco_valido(df, bloco):
            usados.append(bloco)
        else:
            ausentes.append(bloco)
    return usados, ausentes


def separar_tipos(blocos: list[str]) -> tuple[list[str], list[str]]:
    numericas = []
    categoricas = []
    for bloco in blocos:
        tipo = FEATURE_BLOCKS[bloco]["type"]
        if tipo == "numeric":
            numericas.append(bloco)
        elif tipo == "categorical":
            categoricas.append(bloco)
    return numericas, categoricas


def labels_blocos(blocos: list[str]) -> str:
    return "; ".join(FEATURE_BLOCKS[bloco]["label"] for bloco in blocos)


def criar_preprocessador(numericas: list[str], categoricas: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Não informado")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    min_frequency=20,
                ),
            ),
        ]
    )

    transformers = []
    if numericas:
        transformers.append(("num", numeric_transformer, numericas))
    if categoricas:
        transformers.append(("cat", categorical_transformer, categoricas))
    if not transformers:
        raise ValueError("Nenhuma variável válida foi informada ao preprocessador.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )


def min_leaf_dinamico(n_train: int, recorte: str, profundidade: int | str) -> int:
    profundidade = validar_profundidade(profundidade)
    profile = TREE_DEPTH_PROFILES[profundidade]

    # Pré-poda: nunca permite folha com menos de 400 registros.
    # Em bases muito grandes, aumenta discretamente esse piso para evitar folhas muito pequenas.
    return int(max(profile["min_samples_leaf_min"], round(n_train * 0.0005)))


def min_split_dinamico(n_train: int, recorte: str, profundidade: int | str) -> int:
    profundidade = validar_profundidade(profundidade)
    leaf = min_leaf_dinamico(n_train, recorte, profundidade)
    profile = TREE_DEPTH_PROFILES[profundidade]

    # Para dividir um nó, exige pelo menos o dobro da folha mínima e nunca menos de 800.
    return int(max(profile["min_samples_split_min"], 2 * leaf))


def criar_modelo(n_train: int, recorte: str, profundidade: int | str) -> DecisionTreeClassifier:
    profundidade = validar_profundidade(profundidade)
    leaf = min_leaf_dinamico(n_train, recorte, profundidade)
    split = min_split_dinamico(n_train, recorte, profundidade)

    return DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=profundidade,
        min_samples_leaf=leaf,
        min_samples_split=split,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def criar_pipeline(numericas: list[str], categoricas: list[str], n_train: int, recorte: str, profundidade: int | str) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessador", criar_preprocessador(numericas, categoricas)),
            ("modelo", criar_modelo(n_train=n_train, recorte=recorte, profundidade=profundidade)),
        ]
    )


def criar_split(X: pd.DataFrame, y: pd.Series, avaliacao: str):
    indices = np.arange(len(X))
    if avaliacao == "in_sample":
        return indices, indices, "in_sample"
    if avaliacao == "holdout_80_20":
        train_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
            shuffle=True,
        )
        return train_idx, test_idx, "holdout_80_20_estratificado"
    raise ValueError("avaliacao deve ser 'in_sample' ou 'holdout_80_20'.")


def preparar_xy(df: pd.DataFrame, target: str, blocos: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    cfg = target_config(target)
    target_col = cfg["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"A ABT não possui {target_col}.")

    X = pd.DataFrame(index=df.index)
    for coluna in blocos:
        if FEATURE_BLOCKS[coluna]["type"] == "numeric":
            X[coluna] = limpar_coluna_numerica(df[coluna])
        else:
            X[coluna] = limpar_coluna_categorica(df[coluna])

    y = pd.to_numeric(df[target_col], errors="raise").astype(int).copy()
    return X, y


def obter_feature_names(modelo: Pipeline) -> list[str]:
    preprocessador = modelo.named_steps["preprocessador"]
    try:
        return list(preprocessador.get_feature_names_out())
    except Exception:
        estimator = modelo.named_steps["modelo"]
        n = getattr(estimator, "n_features_in_", 0)
        return [f"feature_{i}" for i in range(n)]


def predicoes_por_proba(proba, classes) -> np.ndarray:
    return np.asarray(classes)[np.argmax(proba, axis=1)]


def calcular_roc_auc_binario(y_true, proba, classes, positive_class: int) -> float:
    try:
        classes = list(classes)
        idx = classes.index(positive_class)
        return float(roc_auc_score(y_true, proba[:, idx]))
    except ValueError:
        return np.nan


def calcular_roc_auc_multiclasse(y_true, proba, classes) -> float:
    try:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=list(classes)))
    except ValueError:
        return np.nan


def calcular_metricas(target: str, y_true, proba, classes) -> dict:
    cfg = target_config(target)
    classes_esperadas = cfg["classes"]
    y_pred = predicoes_por_proba(proba, classes)
    matriz = confusion_matrix(y_true, y_pred, labels=classes_esperadas)

    if normalizar_target(target) == "binario":
        metricas = {
            "roc_auc": calcular_roc_auc_binario(y_true, proba, classes, cfg["positive_class"]),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, pos_label=cfg["positive_class"], zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=cfg["positive_class"], zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=cfg["positive_class"], zero_division=0)),
            "log_loss": float(log_loss(y_true, proba, labels=list(classes))),
            "tn": int(matriz[0, 0]),
            "fp": int(matriz[0, 1]),
            "fn": int(matriz[1, 0]),
            "tp": int(matriz[1, 1]),
        }
        return metricas

    metricas = {
        "roc_auc_ovr_weighted": calcular_roc_auc_multiclasse(y_true, proba, classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=list(classes))),
    }

    for i, classe_real in enumerate(classes_esperadas):
        for j, classe_pred in enumerate(classes_esperadas):
            metricas[f"confusao_real_{classe_real}_pred_{classe_pred}"] = int(matriz[i, j])

    return metricas


def feature_original(feature_transformada: str) -> str:
    if feature_transformada.startswith("num__"):
        return feature_transformada.split("__", 1)[1]
    if feature_transformada.startswith("cat__"):
        rest = feature_transformada.split("__", 1)[1]
        for bloco in sorted(FEATURE_BLOCKS.keys(), key=len, reverse=True):
            if rest == bloco or rest.startswith(bloco + "_"):
                return bloco
        return rest.split("_", 1)[0]
    return feature_transformada


def validar_importancias_normalizadas(df: pd.DataFrame, coluna: str, contexto: str) -> None:
    """Valida a soma das importâncias normalizadas da árvore.

    Em `DecisionTreeClassifier.feature_importances_`, as importâncias são uma
    decomposição normalizada da redução total de impureza. Portanto, para uma
    árvore com ao menos uma divisão, a soma deve ser 1, salvo erro numérico
    mínimo. Para árvore sem divisão, a soma pode ser 0.
    """
    if coluna not in df.columns:
        raise KeyError(f"Coluna de importância não encontrada: {coluna}")

    valores = pd.to_numeric(df[coluna], errors="coerce").fillna(0.0)
    if (valores < -IMPORTANCIA_TOL).any():
        raise ValueError(f"Importância negativa encontrada em {contexto}.")

    soma = float(valores.sum())
    if soma > 1.0 + IMPORTANCIA_TOL:
        raise ValueError(
            f"Importâncias normalizadas somam {soma:.6f} em {contexto}. "
            "Isto indica duplicação/agregação indevida."
        )


def extrair_importancias(modelo: Pipeline) -> pd.DataFrame:
    feature_names = obter_feature_names(modelo)
    importancias = modelo.named_steps["modelo"].feature_importances_

    df = pd.DataFrame({
        "feature_transformada": feature_names,
        "variavel_original": [feature_original(f) for f in feature_names],
        "importancia": importancias,
    })

    df["bloco_label"] = df["variavel_original"].map(lambda x: FEATURE_BLOCKS.get(x, {}).get("label", x))

    total = float(df["importancia"].sum())
    df["importancia_normalizada"] = df["importancia"] / total if total > 0 else 0.0
    validar_importancias_normalizadas(
        df,
        coluna="importancia_normalizada",
        contexto="importâncias transformadas da árvore",
    )

    return df.sort_values("importancia", ascending=False).reset_index(drop=True)


def agregar_importancias_por_variavel(importancias: pd.DataFrame) -> pd.DataFrame:
    out = (
        importancias
        .groupby(["variavel_original", "bloco_label"], as_index=False)["importancia"]
        .sum()
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    total = float(out["importancia"].sum())
    out["importancia_normalizada"] = out["importancia"] / total if total > 0 else 0.0
    validar_importancias_normalizadas(
        out,
        coluna="importancia_normalizada",
        contexto="importâncias agregadas por variável original",
    )
    return out


def importancia_bloco(importancias_agregadas: pd.DataFrame, bloco: str) -> float:
    linha = importancias_agregadas[importancias_agregadas["variavel_original"].eq(bloco)]
    if linha.empty:
        return 0.0
    return float(linha["importancia_normalizada"].iloc[0])


def primeira_divisao(modelo: Pipeline) -> str:
    estimator = modelo.named_steps["modelo"]
    feature_names = obter_feature_names(modelo)
    feature_idx = int(estimator.tree_.feature[0]) if estimator.tree_.node_count > 0 else -2
    if feature_idx < 0 or feature_idx >= len(feature_names):
        return "nó terminal"
    return feature_original(feature_names[feature_idx])


def regras_arvore_texto(modelo: Pipeline, max_depth: int = 3) -> str:
    estimator = modelo.named_steps["modelo"]
    feature_names = obter_feature_names(modelo)
    return export_text(estimator, feature_names=feature_names, max_depth=max_depth, decimals=4)


def ajustar_um_experimento(df: pd.DataFrame, metadata_abt: dict, target: str, recorte: str, avaliacao: str, experimento: dict, force: bool, profundidade: int | str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    profundidade = validar_profundidade(profundidade)
    cfg = target_config(target, profundidade)

    blocos_usados, blocos_ausentes = blocos_validos_para_experimento(df, experimento)
    for bloco in CORE_BLOCKS:
        if bloco not in blocos_usados:
            raise ValueError(f"O bloco obrigatório '{bloco}' está ausente ou inválido na ABT.")

    numericas, categoricas = separar_tipos(blocos_usados)
    X, y = preparar_xy(df, target, blocos_usados)

    classes_encontradas = sorted(y.unique().tolist())
    if classes_encontradas != cfg["classes"]:
        raise ValueError(
            f"O target {target} precisa ter as classes {cfg['classes']}. "
            f"Distribuição encontrada: {y.value_counts().sort_index().to_dict()}"
        )

    train_idx, test_idx, avaliacao_descricao = criar_split(X, y, avaliacao)

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    model_dir = models_dir(target, recorte, avaliacao, profundidade)
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg['model_dir_name']}_{experimento['id'].lower()}"
    model_path = model_dir / f"{stem}.joblib"
    model_meta_path = model_dir / f"{stem}_metadata.json"
    rules_path = model_dir / f"{stem}_regras_depth3.txt"

    if model_path.exists() and not force:
        modelo = joblib.load(model_path)
        log(f"[INFO] Modelo existente reutilizado: {model_path}")
    else:
        modelo = criar_pipeline(numericas=numericas, categoricas=categoricas, n_train=len(X_train), recorte=recorte, profundidade=profundidade)
        log(
            f"[INÍCIO] {cfg['model_dir_name']} | {recorte} | {avaliacao} | {experimento['id']} | "
            f"n_treino={len(X_train)} | n_teste={len(X_test)} | blocos={len(blocos_usados)}"
        )
        modelo.fit(X_train, y_train)
        joblib.dump(modelo, model_path)
        log(f"[OK] Modelo salvo em: {model_path}")

    proba_test = modelo.predict_proba(X_test)
    classes = list(modelo.named_steps["modelo"].classes_)
    metricas = calcular_metricas(target, y_test, proba_test, classes)

    importancias_transformadas = extrair_importancias(modelo)
    importancias_agregadas = agregar_importancias_por_variavel(importancias_transformadas)
    feature_names = obter_feature_names(modelo)

    try:
        rules_path.write_text(regras_arvore_texto(modelo, max_depth=3), encoding="utf-8")
    except Exception:
        pass

    registro = {
        "target": target,
        "recorte": recorte,
        "avaliacao": avaliacao,
        "avaliacao_descricao": avaliacao_descricao,
        "modelo": cfg["model_dir_name"],
        "modelo_descricao": cfg["model_description"],
        "experimento": experimento["id"],
        "experimento_nome": experimento["name"],
        "experimento_descricao": experimento["description"],
        "n_total_abt": int(len(df)),
        "n_treino": int(len(X_train)),
        "n_teste": int(len(X_test)),
        "tree_criterion": modelo.named_steps["modelo"].criterion,
        "tree_max_depth_param": modelo.named_steps["modelo"].max_depth,
        "tree_profile": profile_prefix(profundidade),
        "tree_profile_descricao": TREE_DEPTH_PROFILES[profundidade]["description"],
        "tree_min_samples_leaf": int(modelo.named_steps["modelo"].min_samples_leaf),
        "tree_min_samples_split": int(modelo.named_steps["modelo"].min_samples_split),
        "tree_depth_observada": int(modelo.named_steps["modelo"].get_depth()),
        "tree_n_leaves": int(modelo.named_steps["modelo"].get_n_leaves()),
        "primeira_divisao": primeira_divisao(modelo),
        "importancia_renda_per_capita": importancia_bloco(importancias_agregadas, "renda_per_capita"),
        "importancia_gap": importancia_bloco(importancias_agregadas, "gap"),
        "importancia_renda_gap": importancia_bloco(importancias_agregadas, "renda_gap"),
        "blocos_variaveis_qtd": int(len(blocos_usados)),
        "colunas_finais_pos_processamento": int(len(feature_names)),
        "blocos_incluidos": labels_blocos(blocos_usados),
        "blocos_ausentes": labels_blocos(blocos_ausentes),
        "variaveis_numericas": "; ".join(numericas),
        "variaveis_categoricas": "; ".join(categoricas),
        "model_path": str(model_path),
        "rules_path": str(rules_path),
    }

    for classe, label in cfg["class_labels"].items():
        registro[f"target_{classe}"] = label

    registro.update(metricas)

    model_metadata = {
        "registro": registro,
        "target": target,
        "recorte": recorte,
        "classes": classes,
        "class_labels": cfg["class_labels"],
        "experimento": experimento,
        "blocos_usados": blocos_usados,
        "blocos_ausentes": blocos_ausentes,
        "numericas": numericas,
        "categoricas": categoricas,
        "feature_names_out": feature_names,
        "abt_metadata": metadata_abt,
        "observacao": (
            "Árvore de decisão interpretável. As colunas de renda, gap e interação nas tabelas "
            "registram importâncias normalizadas agregadas por variável original."
        ),
    }

    with model_meta_path.open("w", encoding="utf-8") as file:
        json.dump(model_metadata, file, ensure_ascii=False, indent=2, default=str)

    importancias_transformadas.insert(0, "experimento", experimento["id"])
    importancias_transformadas.insert(0, "avaliacao", avaliacao)
    importancias_transformadas.insert(0, "recorte", recorte)
    importancias_transformadas.insert(0, "target", target)

    importancias_agregadas.insert(0, "experimento", experimento["id"])
    importancias_agregadas.insert(0, "avaliacao", avaliacao)
    importancias_agregadas.insert(0, "recorte", recorte)
    importancias_agregadas.insert(0, "target", target)

    return registro, importancias_transformadas, importancias_agregadas


def salvar_resultados(registros: list[dict], importancias_transformadas: list[pd.DataFrame], importancias_agregadas: list[pd.DataFrame], target: str, recorte: str, avaliacao: str, profundidade: int | str) -> None:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)
    profundidade = validar_profundidade(profundidade)
    cfg = target_config(target, profundidade)
    out_dir = diagnostics_dir(target, recorte, avaliacao, profundidade)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_registros = pd.DataFrame(registros)
    df_imp_trans = pd.concat(importancias_transformadas, ignore_index=True)
    df_imp_agreg = pd.concat(importancias_agregadas, ignore_index=True)

    metricas_path = out_dir / f"{cfg['model_dir_name']}_experimentos_metricas.csv"
    imp_trans_path = out_dir / f"{cfg['model_dir_name']}_importancias_transformadas.csv"
    imp_agreg_path = out_dir / f"{cfg['model_dir_name']}_importancias_agregadas.csv"

    df_registros.to_csv(metricas_path, index=False, encoding="utf-8")
    df_imp_trans.to_csv(imp_trans_path, index=False, encoding="utf-8")
    df_imp_agreg.to_csv(imp_agreg_path, index=False, encoding="utf-8")

    log(f"[OK] Métricas salvas em: {metricas_path}")
    log(f"[OK] Importâncias transformadas salvas em: {imp_trans_path}")
    log(f"[OK] Importâncias agregadas salvas em: {imp_agreg_path}")

    print(f"""
Resumo da modelagem
-------------------
target: {target}
recorte: {recorte}
avaliação: {avaliacao}
experimentos: {len(df_registros)}
saída métricas: {metricas_path}
saída importâncias agregadas: {imp_agreg_path}
""")


def run_modelagem(target: str = "binario", recorte: str = "geral", avaliacao: str = "in_sample", force: bool = False, profundidade: int | str = 10) -> None:
    target = normalizar_target(target)
    recorte = normalizar_recorte(recorte)

    if avaliacao not in {"in_sample", "holdout_80_20"}:
        raise ValueError("avaliacao deve ser 'in_sample' ou 'holdout_80_20'.")

    profundidade = validar_profundidade(profundidade)

    df, metadata_abt = carregar_abt(target, recorte)
    registros = []
    importancias_transformadas = []
    importancias_agregadas = []

    for experimento in EXPERIMENTS:
        registro, imp_t, imp_a = ajustar_um_experimento(
            df=df,
            metadata_abt=metadata_abt,
            target=target,
            recorte=recorte,
            avaliacao=avaliacao,
            experimento=experimento,
            force=force,
            profundidade=profundidade,
        )
        registros.append(registro)
        importancias_transformadas.append(imp_t)
        importancias_agregadas.append(imp_a)

    salvar_resultados(registros, importancias_transformadas, importancias_agregadas, target=target, recorte=recorte, avaliacao=avaliacao, profundidade=profundidade)


def parse_args_modelagem():
    parser = argparse.ArgumentParser(description="Ajusta árvore de decisão classificatória.")
    parser.add_argument("--target", choices=["binario", "ternario"], default="binario")
    parser.add_argument("--recorte", choices=["geral", "medicina"], default="geral")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--profundidade", type=int, choices=[10, 14, 19], default=10)
    return parser.parse_args()
