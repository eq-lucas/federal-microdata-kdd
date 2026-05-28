import pandas as pd

from src.constants import (
    INTERIM_FIES_DIR,
    LOGS_DIR,
)


ARQUIVO_INSCRICOES_ENTRADA = INTERIM_FIES_DIR / "fies_inscricoes_com_cine.parquet"
ARQUIVO_OFERTAS_ENTRADA = INTERIM_FIES_DIR / "fies_ofertas_com_cine.parquet"

ARQUIVO_INSCRICOES_SAIDA = INTERIM_FIES_DIR / "fies_inscricoes_com_modalidade.parquet"

RESUMO_MODALIDADE = LOGS_DIR / "transform_modalidade_resumo.csv"
MATRIZ_MODALIDADE = LOGS_DIR / "transform_modalidade_matriz_ano_modalidade.csv"
AUDITORIA_OFERTAS_PFIES = LOGS_DIR / "transform_modalidade_auditoria_ofertas_pfies.csv"


SALARIO_MINIMO_ANO = {
    2018: 954.00,
    2019: 998.00,
    2020: 1045.00,
    2021: 1100.00,
    2022: 1212.00,
}

MAPA_UF_REGIAO = {
    "AC": "Norte",
    "AP": "Norte",
    "AM": "Norte",
    "PA": "Norte",
    "RO": "Norte",
    "RR": "Norte",
    "TO": "Norte",

    "AL": "Nordeste",
    "BA": "Nordeste",
    "CE": "Nordeste",
    "MA": "Nordeste",
    "PB": "Nordeste",
    "PE": "Nordeste",
    "PI": "Nordeste",
    "RN": "Nordeste",
    "SE": "Nordeste",

    "DF": "Centro-Oeste",
    "GO": "Centro-Oeste",
    "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",

    "ES": "Sudeste",
    "MG": "Sudeste",
    "RJ": "Sudeste",
    "SP": "Sudeste",

    "PR": "Sul",
    "RS": "Sul",
    "SC": "Sul",
}


COL_ANO = "ano_processo_seletivo"
COL_SEMESTRE = "semestre_processo_seletivo"
COL_RENDA = "renda_mensal_bruta_per_capita"
COL_UF_RESIDENCIA = "uf_residencia"

COL_MODALIDADE = "modalidade_fies"
COL_REGIAO_RESIDENCIA = "regiao_residencia"

CHAVES_INSCRICOES = [
    "ano_processo_seletivo",
    "semestre_processo_seletivo",
    "codigo_e_mec_mantenedora",
    "codigo_local_oferta",
    "codigo_grupo_preferencia",
    "codigo_curso",
    "turno",
]

CHAVES_OFERTAS = [
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
    log_path = LOGS_DIR / "transform_modalidade.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def validar_colunas(df: pd.DataFrame, colunas: list[str], nome_base: str) -> None:
    faltantes = [col for col in colunas if col not in df.columns]

    if faltantes:
        raise ValueError(f"{nome_base} não contém colunas obrigatórias: {faltantes}")


def normalizar_texto(valor):
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip().upper()

    if texto in {"", "NAN", "NONE", "NULL", "NA", "N/A", "-", "--"}:
        return pd.NA

    return texto


def normalizar_turno(serie: pd.Series) -> pd.Series:
    return serie.map(normalizar_texto).astype("string")


def converter_numero(serie: pd.Series) -> pd.Series:
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

    tem_virgula = s.str.contains(",", na=False)

    s = s.mask(
        tem_virgula,
        s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )

    return pd.to_numeric(s, errors="coerce")


def converter_inteiro(serie: pd.Series) -> pd.Series:
    return converter_numero(serie).round().astype("Int64")


def padronizar_chaves_inscricoes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in CHAVES_INSCRICOES:
        if col == "turno":
            df[col] = normalizar_turno(df[col])
        else:
            df[col] = converter_inteiro(df[col])

    return df


def padronizar_chaves_ofertas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in CHAVES_OFERTAS:
        if col == "turno":
            df[col] = normalizar_turno(df[col])
        else:
            df[col] = converter_inteiro(df[col])

    return df


def normalizar_participa_p_fies(valor) -> str:
    texto = normalizar_texto(valor)

    if pd.isna(texto):
        return "INDEFINIDO"

    if texto in {"SIM", "S", "YES", "Y", "1", "TRUE"}:
        return "SIM"

    if texto in {"NÃO", "NAO", "N", "NO", "0", "FALSE"}:
        return "NAO"

    return "INDEFINIDO"


def preparar_ofertas_pfies(dfo: pd.DataFrame) -> pd.DataFrame:
    validar_colunas(dfo, CHAVES_OFERTAS, "ofertas")

    if "participa_p_fies" not in dfo.columns:
        dfo = dfo.copy()
        dfo["participa_p_fies"] = pd.NA

    dfo = padronizar_chaves_ofertas(dfo)
    dfo["participa_p_fies_status_linha"] = dfo["participa_p_fies"].map(normalizar_participa_p_fies)

    def agrega_status(serie: pd.Series) -> str:
        valores = set(serie.dropna().astype(str))

        if "SIM" in valores:
            return "SIM"

        if "NAO" in valores:
            return "NAO"

        return "INDEFINIDO"

    agrupado = (
        dfo
        .groupby(CHAVES_OFERTAS, dropna=False, observed=True)
        .agg(
            participa_p_fies_status=("participa_p_fies_status_linha", agrega_status),
            qtd_linhas_oferta=("participa_p_fies_status_linha", "size"),
            qtd_status_distintos=("participa_p_fies_status_linha", "nunique"),
        )
        .reset_index()
    )

    auditoria = (
        agrupado["participa_p_fies_status"]
        .value_counts(dropna=False)
        .rename_axis("participa_p_fies_status")
        .reset_index(name="quantidade_chaves_oferta")
    )

    auditoria.to_csv(AUDITORIA_OFERTAS_PFIES, index=False, encoding="utf-8")

    conflitos = int((agrupado["qtd_status_distintos"] > 1).sum())

    if conflitos > 0:
        log(f"[AVISO] Chaves de oferta com status P-FIES conflitante: {conflitos}")

    log(f"[OK] Ofertas P-FIES preparadas | chaves únicas: {len(agrupado)}")
    log(f"[OK] Auditoria P-FIES salva em: {AUDITORIA_OFERTAS_PFIES}")

    return agrupado


def classificar_modalidade_por_renda(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    validar_colunas(df, [COL_ANO, COL_RENDA, COL_UF_RESIDENCIA], "inscrições")

    df[COL_ANO] = converter_inteiro(df[COL_ANO])
    df[COL_RENDA] = converter_numero(df[COL_RENDA])
    df[COL_UF_RESIDENCIA] = df[COL_UF_RESIDENCIA].map(normalizar_texto).astype("string")

    df["salario_minimo_ano"] = df[COL_ANO].map(SALARIO_MINIMO_ANO)
    df["limite_3sm_ano"] = df["salario_minimo_ano"] * 3
    df["limite_5sm_ano"] = df["salario_minimo_ano"] * 5
    df[COL_REGIAO_RESIDENCIA] = df[COL_UF_RESIDENCIA].map(MAPA_UF_REGIAO).astype("string")

    mask_ano_sem_regra = df["salario_minimo_ano"].isna()
    mask_renda_nula = df[COL_RENDA].isna()
    mask_regiao_nula = df[COL_REGIAO_RESIDENCIA].isna()

    mask_mod_i = (
        df[COL_RENDA].notna()
        & df["limite_3sm_ano"].notna()
        & (df[COL_RENDA] <= df["limite_3sm_ano"])
    )

    mask_mod_ii = (
        df[COL_RENDA].notna()
        & df["limite_3sm_ano"].notna()
        & df["limite_5sm_ano"].notna()
        & (df[COL_RENDA] > df["limite_3sm_ano"])
        & (df[COL_RENDA] <= df["limite_5sm_ano"])
        & df[COL_REGIAO_RESIDENCIA].isin(["Norte", "Nordeste", "Centro-Oeste"])
    )

    mask_mod_iii = (
        df[COL_RENDA].notna()
        & df["limite_3sm_ano"].notna()
        & df["limite_5sm_ano"].notna()
        & (df[COL_RENDA] > df["limite_3sm_ano"])
        & (df[COL_RENDA] <= df["limite_5sm_ano"])
        & df[COL_REGIAO_RESIDENCIA].isin(["Sul", "Sudeste"])
    )

    df[COL_MODALIDADE] = "eliminado"

    df.loc[mask_mod_i, COL_MODALIDADE] = "Modalidade I"
    df.loc[mask_mod_ii, COL_MODALIDADE] = "Modalidade II"
    df.loc[mask_mod_iii, COL_MODALIDADE] = "Modalidade III (P-FIES)"

    resumo = {
        "etapa": "peneira_renda_regiao",
        "linhas": len(df),
        "modalidade_i": int(mask_mod_i.sum()),
        "modalidade_ii": int(mask_mod_ii.sum()),
        "modalidade_iii_pre_pfies": int(mask_mod_iii.sum()),
        "eliminado_pre_pfies": int((df[COL_MODALIDADE] == "eliminado").sum()),
        "ano_sem_regra_salario_minimo": int(mask_ano_sem_regra.sum()),
        "renda_nula": int(mask_renda_nula.sum()),
        "uf_residencia_sem_regiao": int(mask_regiao_nula.sum()),
    }

    return df, resumo


def validar_pfies(df: pd.DataFrame, dfo_pfies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    validar_colunas(df, CHAVES_INSCRICOES, "inscrições")

    df = padronizar_chaves_inscricoes(df)

    dfo_merge = dfo_pfies.rename(
        columns={
            "ano": "ano_processo_seletivo",
            "semestre": "semestre_processo_seletivo",
        }
    )

    chaves_merge = CHAVES_INSCRICOES

    colunas_dfo = chaves_merge + [
        "participa_p_fies_status",
        "qtd_linhas_oferta",
        "qtd_status_distintos",
    ]

    dfo_merge = dfo_merge[colunas_dfo].copy()

    linhas_antes = len(df)

    df = df.merge(
        dfo_merge,
        how="left",
        on=chaves_merge,
        validate="m:1",
    )

    if len(df) != linhas_antes:
        raise RuntimeError(
            f"Merge com ofertas alterou número de linhas: antes={linhas_antes}, depois={len(df)}"
        )

    mask_mod_iii = df[COL_MODALIDADE] == "Modalidade III (P-FIES)"

    mask_oferta_nao_encontrada = mask_mod_iii & df["participa_p_fies_status"].isna()
    mask_oferta_sem_pfies = mask_mod_iii & df["participa_p_fies_status"].fillna("INDEFINIDO").isin(["NAO", "INDEFINIDO"])

    df["modalidade_fies_pre_validacao_pfies"] = df[COL_MODALIDADE]
    df["pfies_oferta_encontrada"] = df["participa_p_fies_status"].notna()

    df.loc[mask_oferta_sem_pfies, COL_MODALIDADE] = "eliminado"

    resumo = {
        "etapa": "validacao_pfies",
        "linhas": len(df),
        "modalidade_iii_pre_pfies": int(mask_mod_iii.sum()),
        "modalidade_iii_oferta_nao_encontrada": int(mask_oferta_nao_encontrada.sum()),
        "modalidade_iii_rejeitada_pfies": int(mask_oferta_sem_pfies.sum()),
        "modalidade_iii_pos_pfies": int((df[COL_MODALIDADE] == "Modalidade III (P-FIES)").sum()),
    }

    return df, resumo


def salvar_matriz(df: pd.DataFrame) -> None:
    matriz = (
        df
        .pivot_table(
            index=COL_ANO,
            columns=COL_MODALIDADE,
            values=COL_RENDA,
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
    )

    matriz.to_csv(MATRIZ_MODALIDADE, index=False, encoding="utf-8")

    log(f"[OK] Matriz ano × modalidade salva em: {MATRIZ_MODALIDADE}")


def salvar_resumo(registros: list[dict]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(registros).to_csv(RESUMO_MODALIDADE, index=False, encoding="utf-8")

    log(f"[OK] Resumo salvo em: {RESUMO_MODALIDADE}")


def run() -> None:
    log("=" * 80)
    log("TRANSFORM: CLASSIFICAÇÃO DE MODALIDADE")
    log("=" * 80)

    if not ARQUIVO_INSCRICOES_ENTRADA.exists():
        raise FileNotFoundError(f"Arquivo de inscrições não encontrado: {ARQUIVO_INSCRICOES_ENTRADA}")

    if not ARQUIVO_OFERTAS_ENTRADA.exists():
        raise FileNotFoundError(f"Arquivo de ofertas não encontrado: {ARQUIVO_OFERTAS_ENTRADA}")

    df = pd.read_parquet(ARQUIVO_INSCRICOES_ENTRADA)
    dfo = pd.read_parquet(ARQUIVO_OFERTAS_ENTRADA)

    log(f"[OK] Inscrições carregadas | linhas: {len(df)} | colunas: {len(df.columns)}")
    log(f"[OK] Ofertas carregadas | linhas: {len(dfo)} | colunas: {len(dfo.columns)}")

    registros = []

    df, resumo_renda = classificar_modalidade_por_renda(df)
    registros.append(resumo_renda)

    dfo_pfies = preparar_ofertas_pfies(dfo)

    df, resumo_pfies = validar_pfies(df, dfo_pfies)
    registros.append(resumo_pfies)

    final_counts = (
        df[COL_MODALIDADE]
        .value_counts(dropna=False)
        .rename_axis("modalidade")
        .reset_index(name="quantidade")
    )

    for _, row in final_counts.iterrows():
        registros.append(
            {
                "etapa": "distribuicao_final",
                "modalidade": row["modalidade"],
                "quantidade": int(row["quantidade"]),
            }
        )

    salvar_matriz(df)
    salvar_resumo(registros)

    df.to_parquet(ARQUIVO_INSCRICOES_SAIDA, index=False)

    log(f"[OK] Arquivo com modalidade salvo em: {ARQUIVO_INSCRICOES_SAIDA}")
    log("Classificação de modalidade concluída.")
