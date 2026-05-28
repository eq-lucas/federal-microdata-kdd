import pandas as pd
import numpy as np

from src.constants import (
    ANALYSIS_DATASET_CANDIDATOS_UNICOS_PATH,
    ANALYSIS_DATASET_FUNIL_PATH,
    ARTICLE_YEARS,
    CURATED_INSCRICOES_ARTIGO_PATH,
    CURATED_OFERTAS_PATH,
    LOGS_DIR,
)


RESUMO_PATH = LOGS_DIR / "analysis_dataset_funil_resumo.csv"

CHAVE_INSCRICOES = ["ano", "semestre", "nome_cine_area_geral", "regiao_ies_alvo"]
CHAVE_OFERTAS = ["ano", "semestre", "nome_cine_area_geral", "regiao_ies"]


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "analysis_dataset_funil.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def validar_colunas(df: pd.DataFrame, colunas: list[str], nome: str) -> None:
    faltantes = [col for col in colunas if col not in df.columns]

    if faltantes:
        raise ValueError(f"{nome} não contém colunas obrigatórias: {faltantes}")


def normalizar_categorias(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in colunas:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def preparar_inscricoes(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base de inscrições não encontrada: {path}")

    df = pd.read_parquet(path)

    validar_colunas(
        df,
        [
            "ano",
            "semestre",
            "id_estudante",
            "nome_cine_area_geral",
            "regiao_ies_alvo",
            "situacao_fies",
            "media_enem",
            "nota_corte_gp",
        ],
        "inscrições",
    )

    df = df[df["ano"].isin(ARTICLE_YEARS)].copy()

    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["semestre"] = pd.to_numeric(df["semestre"], errors="coerce").astype("Int64")
    df["media_enem"] = pd.to_numeric(df["media_enem"], errors="coerce")
    df["nota_corte_gp"] = pd.to_numeric(df["nota_corte_gp"], errors="coerce")
    df["gap_nota"] = df["media_enem"] - df["nota_corte_gp"]

    df = normalizar_categorias(df, ["nome_cine_area_geral", "regiao_ies_alvo", "situacao_fies"])

    return df


def preparar_ofertas(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base de ofertas não encontrada: {path}")

    dfo = pd.read_parquet(path)

    validar_colunas(
        dfo,
        [
            "ano",
            "semestre",
            "nome_cine_area_geral",
            "regiao_ies",
            "vagas_fies",
            "vagas_ocupadas",
        ],
        "ofertas",
    )

    dfo = dfo[dfo["ano"].isin(ARTICLE_YEARS)].copy()

    dfo["ano"] = pd.to_numeric(dfo["ano"], errors="coerce").astype("Int64")
    dfo["semestre"] = pd.to_numeric(dfo["semestre"], errors="coerce").astype("Int64")
    dfo["vagas_fies"] = pd.to_numeric(dfo["vagas_fies"], errors="coerce").fillna(0)
    dfo["vagas_ocupadas"] = pd.to_numeric(dfo["vagas_ocupadas"], errors="coerce").fillna(0)

    dfo = normalizar_categorias(dfo, ["nome_cine_area_geral", "regiao_ies"])

    return dfo


def soma_ofertas(dfo: pd.DataFrame, coluna: str, saida: str) -> pd.DataFrame:
    return (
        dfo
        .groupby(CHAVE_OFERTAS, as_index=False, observed=True, dropna=False)[coluna]
        .sum()
        .rename(columns={coluna: saida})
    )


def conta_inscricoes(df: pd.DataFrame, saida: str) -> pd.DataFrame:
    return (
        df
        .groupby(CHAVE_INSCRICOES, as_index=False, observed=True, dropna=False)
        .size()
        .rename(columns={"size": saida})
    )


def pivot_situacoes(df: pd.DataFrame, prefixo: str = "situacao") -> pd.DataFrame:
    tabela = (
        df
        .pivot_table(
            index=CHAVE_INSCRICOES,
            columns="situacao_fies",
            values="id_estudante",
            aggfunc="size",
            fill_value=0,
            observed=True,
            dropna=False,
        )
        .reset_index()
    )

    tabela.columns = [
        col if isinstance(col, str) else str(col)
        for col in tabela.columns
    ]

    rename = {
        col: f"{prefixo}_{str(col).lower().replace(' ', '_')}"
        for col in tabela.columns
        if col not in CHAVE_INSCRICOES
    }

    return tabela.rename(columns=rename)


def gerar_funil(dfo: pd.DataFrame, dfi: pd.DataFrame, candidatos: pd.DataFrame) -> pd.DataFrame:
    df_vagas_ofertadas = soma_ofertas(dfo, "vagas_fies", "vagas_fies")
    df_vagas_ocupadas = soma_ofertas(dfo, "vagas_ocupadas", "vagas_ocupadas")

    df_inscritos_geral = conta_inscricoes(dfi, "Inscritos_Geral")
    df_candidatos_unicos_geral = conta_inscricoes(candidatos, "Candidatos_Unicos_Geral")

    df_inscritos_com_nota = conta_inscricoes(
        dfi[dfi["media_enem"].ge(dfi["nota_corte_gp"])].copy(),
        "inscritos_com_nota_suficiente",
    )

    df_candidatos_com_nota = conta_inscricoes(
        candidatos[candidatos["media_enem"].ge(candidatos["nota_corte_gp"])].copy(),
        "candidatos_unicos_com_nota_suficiente",
    )

    df_inscritos_gap_menos_100 = conta_inscricoes(
        dfi[dfi["gap_nota"].le(-100)].copy(),
        "inscritos_gap_menos_100",
    )

    pivot_geral = pivot_situacoes(dfi, prefixo="inscritos")
    pivot_candidatos = pivot_situacoes(candidatos, prefixo="candidatos")

    df_vagas_ofertadas = df_vagas_ofertadas.rename(columns={"regiao_ies": "regiao_ies_alvo"})
    df_vagas_ocupadas = df_vagas_ocupadas.rename(columns={"regiao_ies": "regiao_ies_alvo"})

    bases = [
        df_vagas_ofertadas,
        df_vagas_ocupadas,
        df_inscritos_geral,
        df_inscritos_com_nota,
        df_inscritos_gap_menos_100,
        df_candidatos_unicos_geral,
        df_candidatos_com_nota,
        pivot_geral,
        pivot_candidatos,
    ]

    funil = bases[0]

    for base in bases[1:]:
        funil = funil.merge(
            base,
            how="outer",
            on=CHAVE_INSCRICOES,
            validate="1:1",
        )

    metricas = [col for col in funil.columns if col not in CHAVE_INSCRICOES]

    for col in metricas:
        funil[col] = pd.to_numeric(funil[col], errors="coerce").fillna(0)

    funil = funil.sort_values(CHAVE_INSCRICOES, kind="mergesort").reset_index(drop=True)

    return funil


def salvar_resumo(dfo: pd.DataFrame, dfi: pd.DataFrame, candidatos: pd.DataFrame, funil: pd.DataFrame) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    registros = [
        {"base": "ofertas_curated", "linhas": len(dfo)},
        {"base": "inscricoes_fies_2019_2021", "linhas": len(dfi)},
        {"base": "dataset_candidatos_unicos_prioridade_fluxo", "linhas": len(candidatos)},
        {"base": "dataset_funil_por_regiao", "linhas": len(funil), "colunas": len(funil.columns)},
        {"base": "dataset_funil_por_regiao", "soma_vagas_fies": float(funil["vagas_fies"].sum()) if "vagas_fies" in funil.columns else np.nan},
        {"base": "dataset_funil_por_regiao", "soma_inscritos_geral": float(funil["Inscritos_Geral"].sum()) if "Inscritos_Geral" in funil.columns else np.nan},
        {"base": "dataset_funil_por_regiao", "soma_candidatos_unicos": float(funil["Candidatos_Unicos_Geral"].sum()) if "Candidatos_Unicos_Geral" in funil.columns else np.nan},
        {"base": "dataset_funil_por_regiao", "soma_vagas_ocupadas": float(funil["vagas_ocupadas"].sum()) if "vagas_ocupadas" in funil.columns else np.nan},
    ]

    pd.DataFrame(registros).to_csv(RESUMO_PATH, index=False, encoding="utf-8")

    log(f"[OK] Resumo salvo em: {RESUMO_PATH}")


def run() -> None:
    log("=" * 80)
    log("ANALYSIS: DATASET DO FUNIL DO FLUXO SELETIVO")
    log("=" * 80)

    dfo = preparar_ofertas(CURATED_OFERTAS_PATH)
    dfi = preparar_inscricoes(CURATED_INSCRICOES_ARTIGO_PATH)
    candidatos = preparar_inscricoes(ANALYSIS_DATASET_CANDIDATOS_UNICOS_PATH)

    log(f"[OK] Ofertas carregadas | linhas: {len(dfo)}")
    log(f"[OK] Inscrições carregadas | linhas: {len(dfi)}")
    log(f"[OK] Candidatos únicos carregados | linhas: {len(candidatos)}")

    funil = gerar_funil(dfo, dfi, candidatos)

    ANALYSIS_DATASET_FUNIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    funil.to_parquet(ANALYSIS_DATASET_FUNIL_PATH, index=False)

    salvar_resumo(dfo, dfi, candidatos, funil)

    log(f"[OK] Dataset de funil salvo em: {ANALYSIS_DATASET_FUNIL_PATH}")
    log("Dataset de funil concluído.")
