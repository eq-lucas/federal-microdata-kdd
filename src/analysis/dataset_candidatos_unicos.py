import pandas as pd

from src.constants import (
    ANALYSIS_DATASET_CANDIDATOS_UNICOS_AGREGADO_PATH,
    ANALYSIS_DATASET_CANDIDATOS_UNICOS_PATH,
    CURATED_INSCRICOES_ARTIGO_PATH,
    LOGS_DIR,
)


RESUMO_PATH = LOGS_DIR / "analysis_dataset_candidatos_unicos_resumo.csv"

ORDEM_PRIORIDADE_FLUXO = [
    "CONTRATADA",
    "INSCRIÇÃO POSTERGADA",
    "PRÉ-SELECIONADO",
    "NÃO CONTRATADO",
    "REJEITADA PELA CPSA",
    "OPÇÃO NÃO CONTRATADA",
    "PARTICIPACAO CANCELADA PELO CANDIDATO",
    "LISTA DE ESPERA",
]

CHAVES_AGREGACAO = [
    "ano",
    "semestre",
    "regiao_morar",
    "nome_cine_area_geral",
    "uf_local_oferta",
    "situacao_fies",
]

ORDEM_SORT = [
    "ano",
    "semestre",
    "id_estudante",
    "situacao_fies",
    "opcao_curso",
]

SUBSET_CANDIDATO_SEMESTRE = [
    "ano",
    "semestre",
    "id_estudante",
]


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "analysis_dataset_candidatos_unicos.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def validar_colunas(df: pd.DataFrame, colunas: list[str]) -> None:
    faltantes = [col for col in colunas if col not in df.columns]

    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltantes}")


def normalizar_texto(valor):
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip().upper()

    if texto in {"", "NAN", "NONE", "NULL", "NA", "N/A", "-", "--"}:
        return pd.NA

    return texto


def preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    validar_colunas(
        df,
        [
            "ano",
            "semestre",
            "id_estudante",
            "situacao_fies",
            "opcao_curso",
            "regiao_morar",
            "nome_cine_area_geral",
            "uf_local_oferta",
        ],
    )

    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["semestre"] = pd.to_numeric(df["semestre"], errors="coerce").astype("Int64")
    df["id_estudante"] = pd.to_numeric(df["id_estudante"], errors="coerce").astype("Int64")
    df["opcao_curso"] = pd.to_numeric(df["opcao_curso"], errors="coerce").astype("Int64")

    df["situacao_fies"] = df["situacao_fies"].map(normalizar_texto).astype("string")
    df["regiao_morar"] = df["regiao_morar"].astype("string")
    df["nome_cine_area_geral"] = df["nome_cine_area_geral"].astype("string")
    df["uf_local_oferta"] = df["uf_local_oferta"].astype("string")

    return df


def gerar_candidatos_unicos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["situacao_fies"] = pd.Categorical(
        df["situacao_fies"],
        categories=ORDEM_PRIORIDADE_FLUXO,
        ordered=True,
    )

    df = df.sort_values(
        by=ORDEM_SORT,
        ascending=True,
        kind="mergesort",
    )

    candidatos = (
        df
        .drop_duplicates(subset=SUBSET_CANDIDATO_SEMESTRE, keep="first")
        .reset_index(drop=True)
    )

    candidatos["situacao_fies"] = candidatos["situacao_fies"].astype("string")

    return candidatos


def gerar_agregado(candidatos: pd.DataFrame) -> pd.DataFrame:
    agregado = (
        candidatos
        .groupby(CHAVES_AGREGACAO, as_index=False, observed=True, dropna=True)["id_estudante"]
        .count()
        .rename(columns={"id_estudante": "qtde_candidatos"})
        .sort_values(["ano", "semestre", "uf_local_oferta"], kind="mergesort")
        .reset_index(drop=True)
    )

    return agregado


def salvar_resumo(df: pd.DataFrame, candidatos: pd.DataFrame, agregado: pd.DataFrame) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    registros = [
        {
            "base": "entrada_fies_2019_2021",
            "linhas": len(df),
            "candidatos_semestre_unicos": df[SUBSET_CANDIDATO_SEMESTRE].drop_duplicates().shape[0],
        },
        {
            "base": "dataset_candidatos_unicos_prioridade_fluxo",
            "linhas": len(candidatos),
            "candidatos_semestre_unicos": candidatos[SUBSET_CANDIDATO_SEMESTRE].drop_duplicates().shape[0],
        },
        {
            "base": "dataset_candidatos_unicos_prioridade_fluxo_agregado",
            "linhas": len(agregado),
            "qtde_candidatos_soma": int(agregado["qtde_candidatos"].sum()),
        },
    ]

    pd.DataFrame(registros).to_csv(RESUMO_PATH, index=False, encoding="utf-8")

    log(f"[OK] Resumo salvo em: {RESUMO_PATH}")


def run() -> None:
    log("=" * 80)
    log("ANALYSIS: DATASET DE CANDIDATOS ÚNICOS POR PRIORIDADE DE FLUXO")
    log("=" * 80)

    if not CURATED_INSCRICOES_ARTIGO_PATH.exists():
        raise FileNotFoundError(f"Base curada não encontrada: {CURATED_INSCRICOES_ARTIGO_PATH}")

    log(f"[INÍCIO] Lendo: {CURATED_INSCRICOES_ARTIGO_PATH}")
    df = pd.read_parquet(CURATED_INSCRICOES_ARTIGO_PATH)

    log(f"[OK] Entrada carregada | linhas: {len(df)} | colunas: {len(df.columns)}")

    df = preparar_base(df)
    candidatos = gerar_candidatos_unicos(df)
    agregado = gerar_agregado(candidatos)

    ANALYSIS_DATASET_CANDIDATOS_UNICOS_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidatos.to_parquet(ANALYSIS_DATASET_CANDIDATOS_UNICOS_PATH, index=False)
    agregado.to_parquet(ANALYSIS_DATASET_CANDIDATOS_UNICOS_AGREGADO_PATH, index=False)

    salvar_resumo(df, candidatos, agregado)

    log(f"[OK] Dataset salvo em: {ANALYSIS_DATASET_CANDIDATOS_UNICOS_PATH}")
    log(f"[OK] Agregado salvo em: {ANALYSIS_DATASET_CANDIDATOS_UNICOS_AGREGADO_PATH}")
    log("Dataset de candidatos únicos concluído.")
