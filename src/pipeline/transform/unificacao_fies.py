from pathlib import Path

import pandas as pd

from src.constants import (
    INTERIM_FIES_DIR,
    LOGS_DIR,
)


ARQUIVO_INSCRICOES_UNIFICADAS = INTERIM_FIES_DIR / "fies_inscricoes_unificadas.parquet"
ARQUIVO_OFERTAS_UNIFICADAS = INTERIM_FIES_DIR / "fies_ofertas_unificadas.parquet"

RESUMO_UNIFICACAO = LOGS_DIR / "transform_unificacao_fies_resumo.csv"


COLUNAS_ORDENACAO_INSCRICOES = [
    "ano_processo_seletivo",
    "semestre_processo_seletivo",
    "id_estudante",
    "opcoes_cursos_inscricao",
    "codigo_grupo_preferencia",
    "codigo_curso",
]

COLUNAS_ORDENACAO_OFERTAS = [
    "ano",
    "semestre",
    "codigo_e_mec_mantenedora",
    "codigo_local_oferta",
    "codigo_grupo_preferencia",
    "codigo_curso",
    "turno",
]


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "transform_unificacao_fies.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def listar_parquets_limpos(tipo: str) -> list[Path]:
    """
    Lista arquivos limpos gerados pela etapa limpeza_tipos_fies.py.

    tipo:
        inscricoes
        ofertas
    """
    padrao = f"fies_{tipo}_*_limpo.parquet"
    return sorted(INTERIM_FIES_DIR.glob(padrao))


def ler_parquets(arquivos: list[Path], tipo: str) -> tuple[pd.DataFrame, list[dict]]:
    """
    Lê e empilha arquivos parquet de um mesmo tipo.

    Não aplica filtro analítico.
    Não faz join.
    Não remove registros por chave.
    """
    dfs = []
    registros_log = []

    for path in arquivos:
        log(f"[INÍCIO] Lendo {path.name}")

        df = pd.read_parquet(path)

        registros_log.append(
            {
                "tipo": tipo,
                "arquivo": path.name,
                "linhas": len(df),
                "colunas": len(df.columns),
                "ano_min": df["ano_arquivo"].min() if "ano_arquivo" in df.columns else pd.NA,
                "ano_max": df["ano_arquivo"].max() if "ano_arquivo" in df.columns else pd.NA,
                "semestres": ", ".join(
                    map(str, sorted(df["semestre_arquivo"].dropna().unique()))
                ) if "semestre_arquivo" in df.columns else pd.NA,
            }
        )

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(), registros_log

    return pd.concat(dfs, ignore_index=True, sort=False), registros_log


def ordenar_se_possivel(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    colunas_existentes = [col for col in colunas if col in df.columns]

    if not colunas_existentes:
        return df.reset_index(drop=True)

    return df.sort_values(by=colunas_existentes, kind="mergesort").reset_index(drop=True)


def auditar_duplicatas_exatas(df: pd.DataFrame, tipo: str) -> int:
    duplicatas = int(df.duplicated(keep="first").sum())
    log(f"[AUDITORIA] {tipo}: duplicatas exatas após unificação: {duplicatas}")
    return duplicatas


def auditar_chaves_inscricoes(df: pd.DataFrame) -> None:
    """
    Auditoria informativa. Não remove linhas.

    A combinação id_estudante + opções de curso ajuda a monitorar duplicações
    aparentes, mas não é usada aqui para excluir registros.
    """
    chaves = ["id_estudante", "opcoes_cursos_inscricao"]

    if not all(col in df.columns for col in chaves):
        faltantes = [col for col in chaves if col not in df.columns]
        log(f"[AVISO] Inscrições: chaves de auditoria faltantes: {faltantes}")
        return

    total = len(df)
    unicas = df[chaves].drop_duplicates().shape[0]

    log(f"[AUDITORIA] Inscrições: linhas totais: {total}")
    log(f"[AUDITORIA] Inscrições: combinações únicas id_estudante + opção: {unicas}")
    log(f"[AUDITORIA] Inscrições: diferença: {total - unicas}")


def auditar_chaves_ofertas(df: pd.DataFrame) -> None:
    """
    Auditoria informativa. Não remove linhas.

    Inclui ano e semestre para evitar mistura indevida entre períodos.
    """
    chaves = [
        "ano",
        "semestre",
        "codigo_e_mec_mantenedora",
        "codigo_local_oferta",
        "codigo_grupo_preferencia",
        "codigo_curso",
        "turno",
    ]

    if not all(col in df.columns for col in chaves):
        faltantes = [col for col in chaves if col not in df.columns]
        log(f"[AVISO] Ofertas: chaves de auditoria faltantes: {faltantes}")
        return

    total = len(df)
    unicas = df[chaves].drop_duplicates().shape[0]

    log(f"[AUDITORIA] Ofertas: linhas totais: {total}")
    log(f"[AUDITORIA] Ofertas: combinações únicas por chave de oferta: {unicas}")
    log(f"[AUDITORIA] Ofertas: diferença: {total - unicas}")


def salvar_resumo(registros: list[dict]) -> None:
    if not registros:
        log("[AVISO] Nenhum registro de resumo para salvar.")
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    df_resumo = pd.DataFrame(registros)
    df_resumo.to_csv(RESUMO_UNIFICACAO, index=False, encoding="utf-8")

    log(f"[OK] Resumo salvo em: {RESUMO_UNIFICACAO}")


def unificar_inscricoes() -> tuple[dict, list[dict]]:
    arquivos = listar_parquets_limpos("inscricoes")

    if not arquivos:
        log(f"[AVISO] Nenhum parquet limpo de inscrições encontrado em: {INTERIM_FIES_DIR}")
        return {
            "tipo": "inscricoes_unificadas",
            "arquivo": pd.NA,
            "arquivos": 0,
            "linhas": 0,
            "colunas": 0,
            "duplicatas_exatas": pd.NA,
        }, []

    df, registros = ler_parquets(arquivos, tipo="inscricoes")

    duplicatas = auditar_duplicatas_exatas(df, "inscrições")
    auditar_chaves_inscricoes(df)

    df = ordenar_se_possivel(df, COLUNAS_ORDENACAO_INSCRICOES)

    INTERIM_FIES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARQUIVO_INSCRICOES_UNIFICADAS, index=False)

    log(
        f"[OK] Inscrições unificadas: {ARQUIVO_INSCRICOES_UNIFICADAS.name} | "
        f"arquivos: {len(arquivos)} | linhas: {len(df)} | colunas: {len(df.columns)}"
    )

    return {
        "tipo": "inscricoes_unificadas",
        "arquivo": ARQUIVO_INSCRICOES_UNIFICADAS.name,
        "arquivos": len(arquivos),
        "linhas": len(df),
        "colunas": len(df.columns),
        "duplicatas_exatas": duplicatas,
    }, registros


def unificar_ofertas() -> tuple[dict, list[dict]]:
    arquivos = listar_parquets_limpos("ofertas")

    if not arquivos:
        log(f"[AVISO] Nenhum parquet limpo de ofertas encontrado em: {INTERIM_FIES_DIR}")
        return {
            "tipo": "ofertas_unificadas",
            "arquivo": pd.NA,
            "arquivos": 0,
            "linhas": 0,
            "colunas": 0,
            "duplicatas_exatas": pd.NA,
        }, []

    df, registros = ler_parquets(arquivos, tipo="ofertas")

    duplicatas = auditar_duplicatas_exatas(df, "ofertas")
    auditar_chaves_ofertas(df)

    df = ordenar_se_possivel(df, COLUNAS_ORDENACAO_OFERTAS)

    INTERIM_FIES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARQUIVO_OFERTAS_UNIFICADAS, index=False)

    log(
        f"[OK] Ofertas unificadas: {ARQUIVO_OFERTAS_UNIFICADAS.name} | "
        f"arquivos: {len(arquivos)} | linhas: {len(df)} | colunas: {len(df.columns)}"
    )

    return {
        "tipo": "ofertas_unificadas",
        "arquivo": ARQUIVO_OFERTAS_UNIFICADAS.name,
        "arquivos": len(arquivos),
        "linhas": len(df),
        "colunas": len(df.columns),
        "duplicatas_exatas": duplicatas,
    }, registros


def run() -> None:
    log("=" * 80)
    log("TRANSFORM: UNIFICAÇÃO FIES")
    log("=" * 80)

    registros_resumo = []

    resultado_inscricoes, registros_inscricoes = unificar_inscricoes()
    resultado_ofertas, registros_ofertas = unificar_ofertas()

    registros_resumo.extend(registros_inscricoes)
    registros_resumo.extend(registros_ofertas)
    registros_resumo.append(resultado_inscricoes)
    registros_resumo.append(resultado_ofertas)

    salvar_resumo(registros_resumo)

    log("Unificação FIES concluída.")