import numpy as np
import pandas as pd

from src.constants import (
    ANALYSIS_DATASET_FUNIL_PATH,
    ANALYSIS_DATASET_TAXAS_PATH,
    LOGS_DIR,
)


RESUMO_PATH = LOGS_DIR / "analysis_dataset_taxas_resumo.csv"

COLUNAS_BASE = [
    "vagas_fies",
    "Inscritos_Geral",
    "inscritos_com_nota_suficiente",
    "Candidatos_Unicos_Geral",
    "candidatos_unicos_com_nota_suficiente",
    "vagas_ocupadas",
]

TAXAS = [
    "taxa_inscricao",
    "taxa_aprovacao_por_inscritos",
    "taxa_aprovacao_por_candidato",
    "taxa_ocupacao",
    "taxa_conversao_inscritos",
    "taxa_conversao_curso_priorizado",
    "taxa_inscritos_capacitados",
    "taxa_candidatos_capacitados",
]


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "analysis_dataset_taxas.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def divisao_segura(numerador, denominador):
    numerador = np.asarray(numerador, dtype="float64")
    denominador = np.asarray(denominador, dtype="float64")

    return np.where(
        (denominador != 0) & (~np.isnan(denominador)),
        numerador / denominador,
        np.nan,
    )


def calcular_taxas(df_base: pd.DataFrame) -> pd.DataFrame:
    df = df_base.copy()

    df["taxa_inscricao"] = divisao_segura(df["Inscritos_Geral"], df["vagas_fies"])
    df["taxa_aprovacao_por_inscritos"] = divisao_segura(df["inscritos_com_nota_suficiente"], df["Inscritos_Geral"])
    df["taxa_aprovacao_por_candidato"] = divisao_segura(df["candidatos_unicos_com_nota_suficiente"], df["Candidatos_Unicos_Geral"])
    df["taxa_ocupacao"] = divisao_segura(df["vagas_ocupadas"], df["vagas_fies"])
    df["taxa_conversao_inscritos"] = divisao_segura(df["vagas_ocupadas"], df["Inscritos_Geral"])
    df["taxa_conversao_curso_priorizado"] = divisao_segura(df["vagas_ocupadas"], df["Candidatos_Unicos_Geral"])
    df["taxa_inscritos_capacitados"] = divisao_segura(df["vagas_ocupadas"], df["inscritos_com_nota_suficiente"])
    df["taxa_candidatos_capacitados"] = divisao_segura(df["vagas_ocupadas"], df["candidatos_unicos_com_nota_suficiente"])

    df[TAXAS] = df[TAXAS] * 100
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def validar_colunas(df: pd.DataFrame) -> None:
    faltantes = [col for col in COLUNAS_BASE if col not in df.columns]

    if faltantes:
        raise ValueError(f"Funil não contém colunas obrigatórias para taxas: {faltantes}")


def gerar_taxas(funil: pd.DataFrame) -> pd.DataFrame:
    validar_colunas(funil)

    chaves = ["ano", "semestre", "regiao_ies_alvo", "nome_cine_area_geral"]

    df_agg = (
        funil
        .groupby(chaves, as_index=False, observed=True, dropna=False)[COLUNAS_BASE]
        .sum()
    )

    df_taxas = calcular_taxas(df_agg)

    df_taxas["periodo"] = (
        "'" + df_taxas["ano"].astype(str).str[-2:] + "." + df_taxas["semestre"].astype(str)
    )

    return df_taxas


def salvar_resumo(df: pd.DataFrame) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    resumo = []

    for taxa in TAXAS:
        resumo.append(
            {
                "taxa": taxa,
                "media": df[taxa].mean(skipna=True),
                "mediana": df[taxa].median(skipna=True),
                "min": df[taxa].min(skipna=True),
                "max": df[taxa].max(skipna=True),
                "n_validos": int(df[taxa].notna().sum()),
            }
        )

    pd.DataFrame(resumo).to_csv(RESUMO_PATH, index=False, encoding="utf-8")

    log(f"[OK] Resumo salvo em: {RESUMO_PATH}")


def run() -> None:
    log("=" * 80)
    log("ANALYSIS: DATASET DE TAXAS")
    log("=" * 80)

    if not ANALYSIS_DATASET_FUNIL_PATH.exists():
        raise FileNotFoundError(f"Funil não encontrado: {ANALYSIS_DATASET_FUNIL_PATH}")

    funil = pd.read_parquet(ANALYSIS_DATASET_FUNIL_PATH)

    log(f"[OK] Funil carregado | linhas: {len(funil)} | colunas: {len(funil.columns)}")

    taxas = gerar_taxas(funil)

    ANALYSIS_DATASET_TAXAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    taxas.to_parquet(ANALYSIS_DATASET_TAXAS_PATH, index=False)

    salvar_resumo(taxas)

    log(f"[OK] Dataset de taxas salvo em: {ANALYSIS_DATASET_TAXAS_PATH}")
    log("Dataset de taxas concluído.")
