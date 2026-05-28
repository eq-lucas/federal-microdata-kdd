import re
from pathlib import Path

import pandas as pd

from src.constants import (
    INTERIM_INEP_DIR,
    LOGS_DIR,
    PIPELINE_INEP_YEARS,
    STAGING_INEP_DIR,
)


ARQUIVO_CURSOS_HISTORICO = INTERIM_INEP_DIR / "inep_cursos_cine_historico_2016_2024.parquet"
ARQUIVO_CURSOS_ULTIMO_REGISTRO = INTERIM_INEP_DIR / "inep_cursos_cine_ultimo_registro.parquet"
RESUMO_MESTRE_INEP = LOGS_DIR / "transform_mestre_inep_resumo.csv"


MAPA_CURSOS_INEP = {
    "NU_ANO_CENSO": "ano_censo",
    "CO_CURSO": "codigo_curso",
    "NO_CURSO": "nome_curso_inep",

    "CO_CINE_AREA_GERAL": "codigo_cine_area_geral",
    "NO_CINE_AREA_GERAL": "nome_cine_area_geral",

    "CO_CINE_AREA_ESPECIFICA": "codigo_cine_area_especifica",
    "NO_CINE_AREA_ESPECIFICA": "nome_cine_area_especifica",

    "CO_CINE_AREA_DETALHADA": "codigo_cine_area_detalhada",
    "NO_CINE_AREA_DETALHADA": "nome_cine_area_detalhada",

    "CO_CINE_ROTULO": "codigo_cine_rotulo",
    "NO_CINE_ROTULO": "nome_cine_rotulo",

    # Alguns anos podem trazer campos equivalentes com nomes levemente diferentes.
    "CO_CINE_AREA": "codigo_cine_area_geral",
    "NO_CINE_AREA": "nome_cine_area_geral",
}


COLUNAS_MINIMAS = [
    "ano_censo",
    "codigo_curso",
    "nome_curso_inep",
]

COLUNAS_CINE_ESPERADAS = [
    "codigo_cine_area_geral",
    "nome_cine_area_geral",
    "codigo_cine_area_especifica",
    "nome_cine_area_especifica",
    "codigo_cine_area_detalhada",
    "nome_cine_area_detalhada",
    "codigo_cine_rotulo",
    "nome_cine_rotulo",
]


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "transform_mestre_inep.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def normalizar_nome_coluna(coluna: str) -> str:
    coluna = str(coluna).replace("\ufeff", "")
    coluna = re.sub(r"\s+", " ", coluna)
    return coluna.strip().upper()


def limpar_texto(valor):
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip()

    if texto == "":
        return pd.NA

    texto_upper = texto.upper()

    if texto_upper in {"NAN", "NONE", "NULL", "NA", "N/A", "-", "--"}:
        return pd.NA

    return texto_upper


def converter_inteiro(serie: pd.Series) -> pd.Series:
    s = serie.astype("string").str.strip()

    s = s.replace(
        {
            "": pd.NA,
            " ": pd.NA,
            "-": pd.NA,
            "--": pd.NA,
            "NAN": pd.NA,
            "nan": pd.NA,
            "NONE": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
        }
    )

    # Proteção para arquivos que venham com separador decimal ou milhar.
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)

    numero = pd.to_numeric(s, errors="coerce")
    return numero.round().astype("Int64")


def detectar_encoding_e_colunas(path: Path) -> tuple[str, list[str]]:
    encodings = ["latin-1", "utf-8-sig", "utf-8"]
    ultimo_erro = None

    for encoding in encodings:
        try:
            header = pd.read_csv(
                path,
                sep=";",
                encoding=encoding,
                nrows=0,
                low_memory=False,
            )
            return encoding, list(header.columns)
        except UnicodeDecodeError as error:
            ultimo_erro = error

    raise ValueError(f"Não foi possível ler o cabeçalho de {path.name}. Último erro: {ultimo_erro}")


def selecionar_colunas_inep(path: Path) -> tuple[str, list[str], dict[str, str]]:
    """
    Detecta colunas úteis no arquivo do INEP.

    Retorna:
        encoding
        colunas originais que devem ser lidas
        mapa de rename baseado nos nomes normalizados
    """
    encoding, colunas_originais = detectar_encoding_e_colunas(path)

    rename = {}
    usecols = []

    for col_original in colunas_originais:
        col_norm = normalizar_nome_coluna(col_original)

        if col_norm in MAPA_CURSOS_INEP:
            usecols.append(col_original)
            rename[col_original] = MAPA_CURSOS_INEP[col_norm]

    return encoding, usecols, rename


def parse_ano_arquivo(path: Path) -> int | None:
    match = re.search(r"(20\d{2}|19\d{2})", path.name)

    if not match:
        return None

    return int(match.group(1))


def ler_cadastro_cursos(path: Path) -> pd.DataFrame:
    encoding, usecols, rename = selecionar_colunas_inep(path)

    if not usecols:
        raise ValueError(f"Nenhuma coluna esperada encontrada em {path.name}")

    df = pd.read_csv(
        path,
        sep=";",
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        usecols=usecols,
    )

    df = df.rename(columns=rename)

    # Se nomes alternativos gerarem duplicação, combina pelo primeiro valor não nulo.
    df = coalescer_colunas_duplicadas(df)

    ano_arquivo = parse_ano_arquivo(path)
    df["arquivo_origem"] = path.name
    df["ano_arquivo"] = ano_arquivo

    return df


def coalescer_colunas_duplicadas(df: pd.DataFrame) -> pd.DataFrame:
    colunas = list(df.columns)
    duplicadas = [col for col in set(colunas) if colunas.count(col) > 1]

    if not duplicadas:
        return df

    partes = []

    for col in dict.fromkeys(colunas):
        bloco = df.loc[:, df.columns == col]

        if bloco.shape[1] == 1:
            partes.append(bloco.iloc[:, 0].rename(col))
        else:
            combinado = bloco.bfill(axis=1).iloc[:, 0].rename(col)
            partes.append(combinado)

    return pd.concat(partes, axis=1)


def padronizar_cadastro_cursos(df: pd.DataFrame, arquivo: str) -> pd.DataFrame:
    df = df.copy()

    if "ano_censo" not in df.columns and "ano_arquivo" in df.columns:
        df["ano_censo"] = df["ano_arquivo"]

    for col in COLUNAS_MINIMAS + COLUNAS_CINE_ESPERADAS:
        if col not in df.columns:
            df[col] = pd.NA

    df["ano_censo"] = converter_inteiro(df["ano_censo"])
    df["ano_arquivo"] = converter_inteiro(df["ano_arquivo"])
    df["codigo_curso"] = converter_inteiro(df["codigo_curso"])

    # CINE fica como string, para preservar qualquer código com zeros à esquerda
    # e evitar perda semântica em códigos classificatórios.
    for col in COLUNAS_CINE_ESPERADAS:
        df[col] = df[col].map(limpar_texto).astype("string")

    df["nome_curso_inep"] = df["nome_curso_inep"].map(limpar_texto).astype("string")
    df["arquivo_origem"] = df["arquivo_origem"].astype("string")

    faltantes_minimas = [col for col in COLUNAS_MINIMAS if df[col].isna().all()]
    if faltantes_minimas:
        log(f"[AVISO] {arquivo}: colunas mínimas totalmente vazias: {faltantes_minimas}")

    cine_vazio = df["nome_cine_area_geral"].isna().all() and df["codigo_cine_area_geral"].isna().all()
    if cine_vazio:
        log(f"[AVISO] {arquivo}: campos CINE de área geral estão vazios ou ausentes.")

    # Mantém apenas colunas úteis e estáveis.
    colunas_saida = [
        "ano_censo",
        "ano_arquivo",
        "codigo_curso",
        "nome_curso_inep",
        "codigo_cine_area_geral",
        "nome_cine_area_geral",
        "codigo_cine_area_especifica",
        "nome_cine_area_especifica",
        "codigo_cine_area_detalhada",
        "nome_cine_area_detalhada",
        "codigo_cine_rotulo",
        "nome_cine_rotulo",
        "arquivo_origem",
    ]

    return df[colunas_saida]


def listar_arquivos_cursos() -> list[Path]:
    arquivos = []

    for ano in PIPELINE_INEP_YEARS:
        candidatos = sorted(STAGING_INEP_DIR.glob(f"*cursos*{ano}*.csv"))

        if not candidatos:
            log(f"[AVISO] Cadastro de cursos não encontrado no staging para {ano}.")
            continue

        # Em geral deve haver um arquivo por ano. Se houver mais de um, processa todos.
        arquivos.extend(candidatos)

    return sorted(set(arquivos))


def auditar_mestre(df: pd.DataFrame) -> None:
    total = len(df)
    duplicatas_exatas = int(df.duplicated(keep="first").sum())

    log(f"[AUDITORIA] Linhas no mestre histórico: {total}")
    log(f"[AUDITORIA] Duplicatas exatas no mestre histórico: {duplicatas_exatas}")

    if {"ano_censo", "codigo_curso"}.issubset(df.columns):
        chaves = df[["ano_censo", "codigo_curso"]].drop_duplicates().shape[0]
        log(f"[AUDITORIA] Combinações únicas ano_censo + codigo_curso: {chaves}")
        log(f"[AUDITORIA] Diferença linhas - chaves: {total - chaves}")

    if "nome_cine_area_geral" in df.columns:
        sem_cine = int(df["nome_cine_area_geral"].isna().sum())
        log(f"[AUDITORIA] Linhas sem nome_cine_area_geral: {sem_cine}")


def construir_ultimo_registro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma dimensão deduplicada por codigo_curso, mantendo o registro mais recente.

    Esta tabela é útil como fallback de consulta. Para análise temporal mais estrita,
    o cruzamento seguinte pode preferir ano_censo correspondente ao ano do FIES.
    """
    df_valid = df[df["codigo_curso"].notna()].copy()

    if df_valid.empty:
        return df_valid

    df_valid = df_valid.sort_values(
        by=["codigo_curso", "ano_censo"],
        ascending=[True, True],
        kind="mergesort",
    )

    return df_valid.drop_duplicates(subset=["codigo_curso"], keep="last").reset_index(drop=True)


def salvar_resumo(registros: list[dict]) -> None:
    if not registros:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(RESUMO_MESTRE_INEP, index=False, encoding="utf-8")
    log(f"[OK] Resumo salvo em: {RESUMO_MESTRE_INEP}")


def run() -> None:
    log("=" * 80)
    log("TRANSFORM: MESTRE INEP/CINE")
    log("=" * 80)

    INTERIM_INEP_DIR.mkdir(parents=True, exist_ok=True)

    arquivos = listar_arquivos_cursos()

    if not arquivos:
        log(f"[ERRO] Nenhum cadastro de cursos encontrado em: {STAGING_INEP_DIR}")
        return

    dfs = []
    registros = []

    for path in arquivos:
        log(f"[INÍCIO] {path.name}")

        try:
            df_raw = ler_cadastro_cursos(path)
            df = padronizar_cadastro_cursos(df_raw, path.name)

            registros.append(
                {
                    "arquivo": path.name,
                    "ano_arquivo": parse_ano_arquivo(path),
                    "linhas": len(df),
                    "colunas": len(df.columns),
                    "codigo_curso_nulos": int(df["codigo_curso"].isna().sum()),
                    "cine_area_geral_nulos": int(df["nome_cine_area_geral"].isna().sum()),
                }
            )

            dfs.append(df)

            log(
                f"[OK] {path.name} | "
                f"linhas: {len(df)} | "
                f"cursos sem código: {df['codigo_curso'].isna().sum()} | "
                f"linhas sem CINE geral: {df['nome_cine_area_geral'].isna().sum()}"
            )

        except Exception as error:
            log(f"[ERRO] {path.name}: {error}")

    if not dfs:
        log("[ERRO] Nenhum arquivo INEP foi processado com sucesso.")
        return

    mestre = pd.concat(dfs, ignore_index=True, sort=False)
    mestre = mestre.drop_duplicates().reset_index(drop=True)

    mestre = mestre.sort_values(
        by=["ano_censo", "codigo_curso"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    auditar_mestre(mestre)

    ultimo = construir_ultimo_registro(mestre)

    mestre.to_parquet(ARQUIVO_CURSOS_HISTORICO, index=False)
    ultimo.to_parquet(ARQUIVO_CURSOS_ULTIMO_REGISTRO, index=False)

    log(f"[OK] Mestre histórico salvo em: {ARQUIVO_CURSOS_HISTORICO}")
    log(f"[OK] Último registro por curso salvo em: {ARQUIVO_CURSOS_ULTIMO_REGISTRO}")
    log(f"[OK] Linhas no último registro: {len(ultimo)}")

    salvar_resumo(registros)

    log("Mestre INEP/CINE concluído.")
