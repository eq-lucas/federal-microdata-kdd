import sqlite3

import pandas as pd

from src.constants import (
    ARTICLE_YEARS,
    CURATED_INSCRICOES_PATH,
    CURATED_INSCRICOES_ARTIGO_PATH,
    CURATED_OFERTAS_PATH,
    CURATED_PARQUET_DIR,
    CURATED_SQLITE_DIR,
    CURATED_SQLITE_PATH,
    INTERIM_FIES_DIR,
    LOGS_DIR,
)


ARQUIVO_INSCRICOES_ENTRADA = INTERIM_FIES_DIR / "fies_inscricoes_com_modalidade.parquet"
ARQUIVO_OFERTAS_ENTRADA = INTERIM_FIES_DIR / "fies_ofertas_com_cine.parquet"

ARQUIVO_INSCRICOES_ARTIGO = CURATED_INSCRICOES_ARTIGO_PATH
ARQUIVO_RESUMO_CURADORIA = LOGS_DIR / "curate_resumo.csv"
ARQUIVO_MATRIZ_CURADORIA = LOGS_DIR / "curate_matriz_ano_modalidade.csv"
ARQUIVO_AUDITORIA_CINE = LOGS_DIR / "curate_auditoria_cine.csv"
ARQUIVO_AUDITORIA_REGIOES = LOGS_DIR / "curate_auditoria_regioes.csv"


SITUACAO_CONTRATADA = "CONTRATADA"
SITUACAO_NAO_CONTRATADO = "NÃO CONTRATADO"
SITUACAO_LISTA_ESPERA = "LISTA DE ESPERA"


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


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "curate.log"

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


def normalizar_uf(serie: pd.Series) -> pd.Series:
    return serie.map(normalizar_texto).astype("string")


def to_numeric_nullable(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce")


def to_int_nullable(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce").round().astype("Int64")


def criar_alias_se_existir(df: pd.DataFrame, origem: str, destino: str) -> pd.DataFrame:
    if origem in df.columns and destino not in df.columns:
        df[destino] = df[origem]

    return df


def criar_aliases_inscricoes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria nomes finais curtos equivalentes ao antigo load.py, sem destruir
    as colunas originais já padronizadas do pipeline novo.
    """
    aliases = {
        "opcoes_cursos_inscricao": "opcao_curso",
        "renda_familiar_mensal_bruta": "renda_familiar_bruta",
        "renda_mensal_bruta_per_capita": "renda_per_capita",
        "nota_corte_grupo_preferencia": "nota_corte_gp",
        "media_nota_enem": "media_enem",
        "codigo_e_mec_mantenedora": "codigo_mec_mantenedora",
        "codigo_e_mec_ies": "codigo_mec_ies",
        "conceito": "conceito_curso",
        "beneficiado_creduc_ou_fies": "beneficiado_creduc_fies",
        "situacao_inscricao_fies": "situacao_fies",
        "nome_cine_area_geral": "area_cine",
        "codigo_cine_area_geral": "codigo_area_cine",
        "nome_cine_rotulo": "rotulo_cine",
        "codigo_cine_rotulo": "codigo_rotulo_cine",
    }

    for origem, destino in aliases.items():
        df = criar_alias_se_existir(df, origem, destino)

    return df


def criar_aliases_ofertas(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "codigo_e_mec_mantenedora": "codigo_mec_mantenedora",
        "codigo_e_mec_ies": "codigo_mec_ies",
        "nota_corte_grupo_preferencia": "nota_corte_gp",
        "conceito": "conceito_curso",
        "vagas_autorizadas_e_mec": "vagas_autorizadas_mec",
        "vagas_ofertadas_fies": "vagas_fies",
        "vagas_alem_da_oferta": "vagas_alem_oferta",
        "vagas_ofertadas_p_fies": "vagas_p_fies",
        "banco_nordeste_brasil_004": "ag_banco_nordeste_004",
        "itau_unibanco_pravaler_341": "ag_itau_pravaler_341",
        "bv_financeira_pravaler_455": "ag_bv_pravaler_455",
        "banco_andbank_pravaler_65": "ag_andbank_pravaler_65",
        "banco_amazonia_sa_003": "ag_banco_amazonia_003",
        "nome_cine_area_geral": "area_cine",
        "codigo_cine_area_geral": "codigo_area_cine",
        "nome_cine_rotulo": "rotulo_cine",
        "codigo_cine_rotulo": "codigo_rotulo_cine",
    }

    for i in range(1, 13):
        aliases[f"semestre_{i}_bruto"] = f"sem_{i}_bruto"
        aliases[f"semestre_{i}_fies"] = f"sem_{i}_fies"

    for origem, destino in aliases.items():
        df = criar_alias_se_existir(df, origem, destino)

    return df


def adicionar_regioes_inscricoes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "uf_residencia" in df.columns:
        df["uf_residencia"] = normalizar_uf(df["uf_residencia"])
        df["regiao_morar"] = df["uf_residencia"].map(MAPA_UF_REGIAO).astype("string")

    if "uf_ies" in df.columns:
        df["uf_ies"] = normalizar_uf(df["uf_ies"])

    if "uf_local_oferta" in df.columns:
        df["uf_local_oferta"] = normalizar_uf(df["uf_local_oferta"])

    if "uf_ies" in df.columns and "uf_local_oferta" in df.columns:
        df["uf_ies_corrigida"] = df["uf_ies"].combine_first(df["uf_local_oferta"])
    elif "uf_ies" in df.columns:
        df["uf_ies_corrigida"] = df["uf_ies"]
    elif "uf_local_oferta" in df.columns:
        df["uf_ies_corrigida"] = df["uf_local_oferta"]
    else:
        df["uf_ies_corrigida"] = pd.NA

    df["uf_ies_corrigida"] = normalizar_uf(df["uf_ies_corrigida"])
    df["regiao_ies_alvo"] = df["uf_ies_corrigida"].map(MAPA_UF_REGIAO).astype("string")

    # Alias semântico usado na modalidade.py.
    if "regiao_residencia" in df.columns:
        df["regiao_residencia"] = df["regiao_residencia"].astype("string")
    elif "regiao_morar" in df.columns:
        df["regiao_residencia"] = df["regiao_morar"].astype("string")

    return df


def adicionar_regioes_ofertas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "uf_ies" in df.columns:
        df["uf_ies"] = normalizar_uf(df["uf_ies"])

    if "uf_local_oferta" in df.columns:
        df["uf_local_oferta"] = normalizar_uf(df["uf_local_oferta"])

    if "uf_ies" in df.columns and "uf_local_oferta" in df.columns:
        df["uf_ies_corrigida"] = df["uf_ies"].combine_first(df["uf_local_oferta"])
    elif "uf_ies" in df.columns:
        df["uf_ies_corrigida"] = df["uf_ies"]
    elif "uf_local_oferta" in df.columns:
        df["uf_ies_corrigida"] = df["uf_local_oferta"]
    else:
        df["uf_ies_corrigida"] = pd.NA

    df["uf_ies_corrigida"] = normalizar_uf(df["uf_ies_corrigida"])
    df["regiao_ies"] = df["uf_ies_corrigida"].map(MAPA_UF_REGIAO).astype("string")

    return df


def preparar_inscricoes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    validar_colunas(
        df,
        [
            "ano_processo_seletivo",
            "semestre_processo_seletivo",
            "modalidade_fies",
            "situacao_inscricao_fies",
            "renda_mensal_bruta_per_capita",
            "media_nota_enem",
            "nota_corte_grupo_preferencia",
        ],
        "inscrições",
    )

    df["ano"] = to_int_nullable(df["ano_processo_seletivo"])
    df["semestre"] = to_int_nullable(df["semestre_processo_seletivo"])

    df["renda_familiar_per_capita"] = to_numeric_nullable(df["renda_mensal_bruta_per_capita"])
    df["nota_enem"] = to_numeric_nullable(df["media_nota_enem"])
    df["nota_corte"] = to_numeric_nullable(df["nota_corte_grupo_preferencia"])
    df["gap_nota_corte"] = df["nota_enem"] - df["nota_corte"]

    df["situacao_inscricao_fies"] = df["situacao_inscricao_fies"].map(normalizar_texto).astype("string")
    df["modalidade_fies"] = df["modalidade_fies"].astype("string")

    df = criar_aliases_inscricoes(df)
    df = adicionar_regioes_inscricoes(df)

    df["elegivel_academicamente"] = df["gap_nota_corte"].ge(0)
    df.loc[df["gap_nota_corte"].isna(), "elegivel_academicamente"] = pd.NA

    df["contratada"] = df["situacao_inscricao_fies"].eq(SITUACAO_CONTRATADA)
    df["nao_contratado"] = df["situacao_inscricao_fies"].eq(SITUACAO_NAO_CONTRATADO)
    df["lista_espera"] = df["situacao_inscricao_fies"].eq(SITUACAO_LISTA_ESPERA)

    df["target_binario_contratacao"] = pd.NA
    df.loc[df["nao_contratado"], "target_binario_contratacao"] = 0
    df.loc[df["contratada"], "target_binario_contratacao"] = 1
    df["target_binario_contratacao"] = df["target_binario_contratacao"].astype("Int64")

    df["target_ternario_fluxo"] = pd.NA
    df.loc[df["lista_espera"], "target_ternario_fluxo"] = 0
    df.loc[df["nao_contratado"], "target_ternario_fluxo"] = 1
    df.loc[df["contratada"], "target_ternario_fluxo"] = 2
    df["target_ternario_fluxo"] = df["target_ternario_fluxo"].astype("Int64")

    if "opcoes_cursos_inscricao" in df.columns:
        df["inscricao_priorizada"] = to_int_nullable(df["opcoes_cursos_inscricao"]).eq(1)
    elif "opcao_curso" in df.columns:
        df["inscricao_priorizada"] = to_int_nullable(df["opcao_curso"]).eq(1)
    else:
        df["inscricao_priorizada"] = pd.NA

    if "nome_curso" in df.columns:
        curso_norm = df["nome_curso"].map(normalizar_texto).astype("string")
        df["curso_medicina"] = curso_norm.eq("MEDICINA")
    else:
        df["curso_medicina"] = pd.NA

    if "nome_cine_area_geral" in df.columns:
        area_norm = df["nome_cine_area_geral"].map(normalizar_texto).astype("string")
        df["area_saude_bem_estar"] = area_norm.eq("SAÚDE E BEM-ESTAR")
    else:
        df["area_saude_bem_estar"] = pd.NA

    # Recorte empírico do artigo: todos os registros de 2019 a 2021.
    # A classificação de modalidade permanece na base como variável descritiva,
    # mas não restringe o recorte final.
    df["recorte_artigo"] = df["ano"].isin(ARTICLE_YEARS)

    return df


def preparar_ofertas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ano" in df.columns:
        df["ano"] = to_int_nullable(df["ano"])

    if "semestre" in df.columns:
        df["semestre"] = to_int_nullable(df["semestre"])

    df = criar_aliases_ofertas(df)
    df = adicionar_regioes_ofertas(df)

    if "nome_curso" in df.columns:
        curso_norm = df["nome_curso"].map(normalizar_texto).astype("string")
        df["curso_medicina"] = curso_norm.eq("MEDICINA")

    if "nome_cine_area_geral" in df.columns:
        area_norm = df["nome_cine_area_geral"].map(normalizar_texto).astype("string")
        df["area_saude_bem_estar"] = area_norm.eq("SAÚDE E BEM-ESTAR")

    if "ano" in df.columns and "semestre" in df.columns:
        antes = len(df)
        df = df.dropna(subset=["ano", "semestre"]).copy()
        removidas = antes - len(df)

        if removidas:
            log(f"[AVISO] Ofertas sem ano/semestre removidas na curadoria: {removidas}")

    return df.reset_index(drop=True)


def gerar_recorte_artigo(df: pd.DataFrame) -> pd.DataFrame:
    recorte = df[df["recorte_artigo"]].copy()
    return recorte.reset_index(drop=True)


def auditar_cine(df: pd.DataFrame, dfo: pd.DataFrame) -> None:
    registros = []

    for nome, base in [("inscricoes", df), ("ofertas", dfo)]:
        sem_nome = int(base["nome_cine_area_geral"].isna().sum()) if "nome_cine_area_geral" in base.columns else len(base)
        sem_codigo = int(base["codigo_cine_area_geral"].isna().sum()) if "codigo_cine_area_geral" in base.columns else len(base)

        registros.append(
            {
                "base": nome,
                "linhas": len(base),
                "sem_nome_cine_area_geral": sem_nome,
                "sem_codigo_cine_area_geral": sem_codigo,
            }
        )

    pd.DataFrame(registros).to_csv(ARQUIVO_AUDITORIA_CINE, index=False, encoding="utf-8")
    log(f"[OK] Auditoria CINE salva em: {ARQUIVO_AUDITORIA_CINE}")


def auditar_regioes(df: pd.DataFrame, dfo: pd.DataFrame) -> None:
    registros = [
        {
            "base": "inscricoes",
            "campo": "regiao_morar",
            "nulos": int(df["regiao_morar"].isna().sum()) if "regiao_morar" in df.columns else len(df),
            "linhas": len(df),
        },
        {
            "base": "inscricoes",
            "campo": "regiao_ies_alvo",
            "nulos": int(df["regiao_ies_alvo"].isna().sum()) if "regiao_ies_alvo" in df.columns else len(df),
            "linhas": len(df),
        },
        {
            "base": "ofertas",
            "campo": "regiao_ies",
            "nulos": int(dfo["regiao_ies"].isna().sum()) if "regiao_ies" in dfo.columns else len(dfo),
            "linhas": len(dfo),
        },
    ]

    pd.DataFrame(registros).to_csv(ARQUIVO_AUDITORIA_REGIOES, index=False, encoding="utf-8")
    log(f"[OK] Auditoria de regiões salva em: {ARQUIVO_AUDITORIA_REGIOES}")


def salvar_sqlite(tabelas: dict[str, pd.DataFrame]) -> None:
    CURATED_SQLITE_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(CURATED_SQLITE_PATH) as conn:
        for nome_tabela, df in tabelas.items():
            log(f"[INÍCIO] Gravando tabela SQLite: {nome_tabela} | linhas: {len(df)}")
            df.to_sql(
                nome_tabela,
                conn,
                if_exists="replace",
                index=False,
                chunksize=100_000,
            )
            log(f"[OK] Tabela SQLite gravada: {nome_tabela}")


def salvar_resumo(df: pd.DataFrame, dfo: pd.DataFrame, recorte: pd.DataFrame) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    registros = [
        {
            "base": "inscricoes_curated",
            "linhas": len(df),
            "colunas": len(df.columns),
            "anos": ", ".join(map(str, sorted(df["ano"].dropna().unique()))) if "ano" in df.columns else pd.NA,
        },
        {
            "base": "ofertas_curated",
            "linhas": len(dfo),
            "colunas": len(dfo.columns),
            "anos": ", ".join(map(str, sorted(dfo["ano"].dropna().unique()))) if "ano" in dfo.columns else pd.NA,
        },
        {
            "base": "inscricoes_fies_2019_2021",
            "linhas": len(recorte),
            "colunas": len(recorte.columns),
            "anos": ", ".join(map(str, sorted(recorte["ano"].dropna().unique()))) if "ano" in recorte.columns else pd.NA,
        },
    ]

    pd.DataFrame(registros).to_csv(ARQUIVO_RESUMO_CURADORIA, index=False, encoding="utf-8")

    matriz = (
        df
        .pivot_table(
            index="ano",
            columns="modalidade_fies",
            values="id_estudante" if "id_estudante" in df.columns else "renda_familiar_per_capita",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
    )

    matriz.to_csv(ARQUIVO_MATRIZ_CURADORIA, index=False, encoding="utf-8")

    log(f"[OK] Resumo salvo em: {ARQUIVO_RESUMO_CURADORIA}")
    log(f"[OK] Matriz salva em: {ARQUIVO_MATRIZ_CURADORIA}")


def run() -> None:
    log("=" * 80)
    log("CURATE: BASES FINAIS")
    log("=" * 80)

    if not ARQUIVO_INSCRICOES_ENTRADA.exists():
        raise FileNotFoundError(f"Arquivo de inscrições não encontrado: {ARQUIVO_INSCRICOES_ENTRADA}")

    if not ARQUIVO_OFERTAS_ENTRADA.exists():
        raise FileNotFoundError(f"Arquivo de ofertas não encontrado: {ARQUIVO_OFERTAS_ENTRADA}")

    CURATED_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_SQLITE_DIR.mkdir(parents=True, exist_ok=True)

    log(f"[INÍCIO] Lendo inscrições: {ARQUIVO_INSCRICOES_ENTRADA}")
    df = pd.read_parquet(ARQUIVO_INSCRICOES_ENTRADA)

    log(f"[INÍCIO] Lendo ofertas: {ARQUIVO_OFERTAS_ENTRADA}")
    dfo = pd.read_parquet(ARQUIVO_OFERTAS_ENTRADA)

    log(f"[OK] Inscrições carregadas | linhas: {len(df)} | colunas: {len(df.columns)}")
    log(f"[OK] Ofertas carregadas | linhas: {len(dfo)} | colunas: {len(dfo.columns)}")

    df = preparar_inscricoes(df)
    dfo = preparar_ofertas(dfo)

    recorte = gerar_recorte_artigo(df)

    df.to_parquet(CURATED_INSCRICOES_PATH, index=False)
    dfo.to_parquet(CURATED_OFERTAS_PATH, index=False)
    recorte.to_parquet(ARQUIVO_INSCRICOES_ARTIGO, index=False)

    log(f"[OK] Inscrições curadas salvas em: {CURATED_INSCRICOES_PATH}")
    log(f"[OK] Ofertas curadas salvas em: {CURATED_OFERTAS_PATH}")
    log(f"[OK] Recorte do artigo salvo em: {ARQUIVO_INSCRICOES_ARTIGO}")

    salvar_resumo(df, dfo, recorte)
    auditar_cine(df, dfo)
    auditar_regioes(df, dfo)

    salvar_sqlite(
        {
            "inscricoes": df,
            "ofertas": dfo,
            "inscricoes_fies_2019_2021": recorte,
        }
    )

    log(f"[OK] SQLite salvo em: {CURATED_SQLITE_PATH}")
    log("Curadoria concluída.")