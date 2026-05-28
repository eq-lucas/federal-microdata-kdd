from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src import constants as C


PROJECT_ROOT = getattr(C, "PROJECT_ROOT", PROJECT_ROOT_FOR_IMPORT)
LOGS_DIR = getattr(C, "LOGS_DIR", PROJECT_ROOT / "reports" / "logs")
ABT_DIR = getattr(C, "ABT_DIR", PROJECT_ROOT / "data" / "06_abt")
DIAGNOSTICS_DIR = getattr(C, "DIAGNOSTICS_DIR", PROJECT_ROOT / "reports" / "diagnostics")
MODELS_GENERAL_DIR = getattr(C, "MODELS_GENERAL_DIR", PROJECT_ROOT / "models" / "general")
MODELS_MEDICINA_DIR = getattr(C, "MODELS_MEDICINA_DIR", PROJECT_ROOT / "models" / "medicina")

ABT_BINARIA_GERAL_PATH = getattr(
    C,
    "ABT_BINARIA_GERAL_PATH",
    ABT_DIR / "abt_contratacao_binaria_geral.parquet",
)

ABT_BINARIA_MEDICINA_PATH = getattr(
    C,
    "ABT_BINARIA_MEDICINA_PATH",
    ABT_DIR / "abt_contratacao_binaria_medicina.parquet",
)


RANDOM_STATE = 42
TEST_SIZE = 0.20

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

CORE_BLOCKS = [
    "renda_per_capita",
    "gap",
    "renda_gap",
]

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
        "blocks": CORE_BLOCKS + [
            "idade",
            "nota_corte_gp",
            "opcao_curso",
        ],
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


def log(message: str, log_file: str = "logit_binario_experimentos.log") -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / log_file

    with path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def abt_path(recorte: str) -> Path:
    if recorte == "geral":
        return ABT_BINARIA_GERAL_PATH

    if recorte == "medicina":
        return ABT_BINARIA_MEDICINA_PATH

    raise ValueError("recorte deve ser 'geral' ou 'medicina'.")


def metadata_path(recorte: str) -> Path:
    path = abt_path(recorte)

    return path.with_name(path.stem + "_metadata.json")


def models_dir(recorte: str, avaliacao: str) -> Path:
    root = MODELS_GENERAL_DIR if recorte == "geral" else MODELS_MEDICINA_DIR

    return root / "logit_binario" / avaliacao


def diagnostics_dir(recorte: str, avaliacao: str) -> Path:
    return DIAGNOSTICS_DIR / "modeling" / "logit_binario" / avaliacao / recorte


def carregar_abt(recorte: str) -> tuple[pd.DataFrame, dict]:
    path = abt_path(recorte)
    meta_path = metadata_path(recorte)

    if not path.exists():
        raise FileNotFoundError(
            f"ABT não encontrada: {path}. "
            f"Rode primeiro: python3 main.py abt binaria"
        )

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadados da ABT não encontrados: {meta_path}")

    df = pd.read_parquet(path)

    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    return df, meta


def limpar_coluna_categorica(serie: pd.Series) -> pd.Series:
    """
    Converte a coluna categórica para object puro, com np.nan nos ausentes.

    Motivo:
    pandas.StringDtype usa pd.NA. Em algumas combinações pandas + scikit-learn,
    SimpleImputer pode tentar comparar pd.NA consigo mesmo e levantar:
    TypeError: boolean value of NA is ambiguous.

    Por isso, antes de entrar no ColumnTransformer, todas as categorias ficam
    como strings Python ou np.nan.
    """
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
    return bool(serie.notna().any() and serie.nunique(dropna=True) > 0)


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
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
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


def criar_modelo() -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        C=0.1,
        class_weight="balanced",
        max_iter=5000,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )


def criar_pipeline(numericas: list[str], categoricas: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessador", criar_preprocessador(numericas, categoricas)),
            ("modelo", criar_modelo()),
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


def preparar_xy(df: pd.DataFrame, blocos: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    if "target_binario" not in df.columns:
        raise ValueError("A ABT não possui target_binario.")

    X = pd.DataFrame(index=df.index)

    for coluna in blocos:
        if FEATURE_BLOCKS[coluna]["type"] == "numeric":
            X[coluna] = limpar_coluna_numerica(df[coluna])
        else:
            X[coluna] = limpar_coluna_categorica(df[coluna])

    y = pd.to_numeric(df["target_binario"], errors="raise").astype(int).copy()

    return X, y


def obter_feature_names(modelo: Pipeline) -> list[str]:
    preprocessador = modelo.named_steps["preprocessador"]

    try:
        return list(preprocessador.get_feature_names_out())
    except Exception:
        estimador = modelo.named_steps["modelo"]
        n = getattr(estimador, "n_features_in_", 0)

        return [f"feature_{i}" for i in range(n)]


def calcular_metricas(y_true, proba) -> dict:
    y_pred = (proba >= 0.5).astype(int)
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - proba, proba]), labels=[0, 1])),
        "tn": int(matriz[0, 0]),
        "fp": int(matriz[0, 1]),
        "fn": int(matriz[1, 0]),
        "tp": int(matriz[1, 1]),
    }


def extrair_coeficientes(modelo: Pipeline) -> pd.DataFrame:
    feature_names = obter_feature_names(modelo)
    coef = modelo.named_steps["modelo"].coef_[0]

    return pd.DataFrame(
        {
            "feature_transformada": feature_names,
            "coeficiente_padronizado": coef,
            "abs_coeficiente": np.abs(coef),
        }
    )


def coeficiente_exato(coeficientes: pd.DataFrame, feature_name: str):
    linha = coeficientes[coeficientes["feature_transformada"] == f"num__{feature_name}"]

    if len(linha) == 0:
        return np.nan

    return float(linha["coeficiente_padronizado"].iloc[0])


def ajustar_um_experimento(
    df: pd.DataFrame,
    metadata_abt: dict,
    recorte: str,
    avaliacao: str,
    experimento: dict,
    force: bool,
) -> tuple[dict, pd.DataFrame]:
    blocos_usados, blocos_ausentes = blocos_validos_para_experimento(df, experimento)

    for bloco in CORE_BLOCKS:
        if bloco not in blocos_usados:
            raise ValueError(f"O bloco obrigatório '{bloco}' está ausente ou inválido na ABT.")

    numericas, categoricas = separar_tipos(blocos_usados)

    X, y = preparar_xy(df, blocos_usados)

    if y.nunique() != 2:
        raise ValueError(f"O target precisa ter duas classes. Distribuição encontrada: {y.value_counts().to_dict()}")

    train_idx, test_idx, avaliacao_descricao = criar_split(X, y, avaliacao)

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    model_dir = models_dir(recorte, avaliacao)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"logit_binario_{experimento['id'].lower()}.joblib"
    model_meta_path = model_dir / f"logit_binario_{experimento['id'].lower()}_metadata.json"

    if model_path.exists() and not force:
        modelo = joblib.load(model_path)
        log(f"[INFO] Modelo existente reutilizado: {model_path}")
    else:
        modelo = criar_pipeline(numericas=numericas, categoricas=categoricas)

        log(
            f"[INÍCIO] {recorte} | {avaliacao} | {experimento['id']} | "
            f"n_treino={len(X_train)} | n_teste={len(X_test)} | blocos={len(blocos_usados)}"
        )

        modelo.fit(X_train, y_train)

        joblib.dump(modelo, model_path)

        log(f"[OK] Modelo salvo em: {model_path}")

    proba_test = modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, proba_test)
    coeficientes = extrair_coeficientes(modelo)
    feature_names = obter_feature_names(modelo)

    coef_renda = coeficiente_exato(coeficientes, "renda_per_capita")
    coef_gap = coeficiente_exato(coeficientes, "gap")
    coef_interacao = coeficiente_exato(coeficientes, "renda_gap")

    registro = {
        "recorte": recorte,
        "avaliacao": avaliacao,
        "avaliacao_descricao": avaliacao_descricao,
        "modelo": "logit_binario",
        "modelo_descricao": "Regressão logística binária",
        "experimento": experimento["id"],
        "experimento_nome": experimento["name"],
        "experimento_descricao": experimento["description"],
        "n_total_abt": int(len(df)),
        "n_treino": int(len(X_train)),
        "n_teste": int(len(X_test)),
        "target_0": "Não contratado",
        "target_1": "Contratada",
        "roc_auc": metricas["roc_auc"],
        "accuracy": metricas["accuracy"],
        "balanced_accuracy": metricas["balanced_accuracy"],
        "precision": metricas["precision"],
        "recall": metricas["recall"],
        "f1": metricas["f1"],
        "log_loss": metricas["log_loss"],
        "tn": metricas["tn"],
        "fp": metricas["fp"],
        "fn": metricas["fn"],
        "tp": metricas["tp"],
        "coef_renda_per_capita": coef_renda,
        "coef_gap": coef_gap,
        "coef_renda_gap": coef_interacao,
        "blocos_variaveis_qtd": int(len(blocos_usados)),
        "colunas_finais_pos_processamento": int(len(feature_names)),
        "blocos_incluidos": labels_blocos(blocos_usados),
        "blocos_ausentes": labels_blocos(blocos_ausentes),
        "variaveis_numericas": "; ".join(numericas),
        "variaveis_categoricas": "; ".join(categoricas),
        "model_path": str(model_path),
    }

    model_metadata = {
        "registro": registro,
        "experimento": experimento,
        "blocos_usados": blocos_usados,
        "blocos_ausentes": blocos_ausentes,
        "numericas": numericas,
        "categoricas": categoricas,
        "feature_names_out": feature_names,
        "abt_metadata": metadata_abt,
        "observacao": "Modelo logístico binário. Métricas calculadas no conjunto indicado por avaliacao_descricao.",
    }

    with model_meta_path.open("w", encoding="utf-8") as file:
        json.dump(model_metadata, file, ensure_ascii=False, indent=2, default=str)

    coeficientes.insert(0, "experimento", experimento["id"])
    coeficientes.insert(0, "avaliacao", avaliacao)
    coeficientes.insert(0, "recorte", recorte)

    return registro, coeficientes


def salvar_resultados(registros: list[dict], coeficientes: list[pd.DataFrame], recorte: str, avaliacao: str) -> None:
    out_dir = diagnostics_dir(recorte, avaliacao)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_registros = pd.DataFrame(registros)
    df_coeficientes = pd.concat(coeficientes, ignore_index=True)

    metricas_path = out_dir / "logit_binario_experimentos_metricas.csv"
    coeficientes_path = out_dir / "logit_binario_coeficientes.csv"

    df_registros.to_csv(metricas_path, index=False, encoding="utf-8")
    df_coeficientes.to_csv(coeficientes_path, index=False, encoding="utf-8")

    log(f"[OK] Métricas salvas em: {metricas_path}")
    log(f"[OK] Coeficientes salvos em: {coeficientes_path}")

    print(f"""
Resumo da modelagem
-------------------
recorte: {recorte}
avaliação: {avaliacao}
experimentos: {len(df_registros)}
saída métricas: {metricas_path}
saída coeficientes: {coeficientes_path}
""")


def run_modelagem(recorte: str = "geral", avaliacao: str = "in_sample", force: bool = False) -> None:
    if recorte not in {"geral", "medicina"}:
        raise ValueError("recorte deve ser 'geral' ou 'medicina'.")

    if avaliacao not in {"in_sample", "holdout_80_20"}:
        raise ValueError("avaliacao deve ser 'in_sample' ou 'holdout_80_20'.")

    df, metadata_abt = carregar_abt(recorte)

    registros = []
    coeficientes = []

    for experimento in EXPERIMENTS:
        registro, coef = ajustar_um_experimento(
            df=df,
            metadata_abt=metadata_abt,
            recorte=recorte,
            avaliacao=avaliacao,
            experimento=experimento,
            force=force,
        )

        registros.append(registro)
        coeficientes.append(coef)

    salvar_resultados(registros, coeficientes, recorte=recorte, avaliacao=avaliacao)


def parse_args_modelagem():
    parser = argparse.ArgumentParser(description="Ajusta regressão logística binária.")
    parser.add_argument("--recorte", choices=["geral", "medicina"], default="geral")
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()
