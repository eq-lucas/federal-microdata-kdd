from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

import argparse
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from src import constants as C


PROJECT_ROOT = getattr(C, "PROJECT_ROOT", PROJECT_ROOT_FOR_IMPORT)
LOGS_DIR = getattr(C, "LOGS_DIR", PROJECT_ROOT / "reports" / "logs")
ABT_DIR = getattr(C, "ABT_DIR", PROJECT_ROOT / "data" / "06_abt")
CURATED_PARQUET_DIR = getattr(C, "CURATED_PARQUET_DIR", PROJECT_ROOT / "data" / "04_curated" / "parquet")

CURATED_INSCRICOES_ARTIGO_PATH = getattr(
    C,
    "CURATED_INSCRICOES_ARTIGO_PATH",
    CURATED_PARQUET_DIR / "inscricoes_fies_2019_2021.parquet",
)

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


STATUS_CONTRATADA = "CONTRATADA"
STATUS_NAO_CONTRATADO = "NÃO CONTRATADO"

TARGET_MAP = {
    STATUS_NAO_CONTRATADO: 0,
    STATUS_CONTRATADA: 1,
}

COLUMN_ALIASES = {
    "situacao_fies": [
        "situacao_fies",
        "situacao_inscricao_fies",
        "situacao_inscricao",
        "situacao",
    ],
    "renda_per_capita": [
        "renda_per_capita",
        "renda_familiar_per_capita",
        "renda_mensal_bruta_per_capita",
        "vl_renda_per_capita",
        "valor_renda_per_capita",
        "renda_familiar_mensal_per_capita",
    ],
    "media_enem": [
        "media_enem",
        "media_nota_enem",
    ],
    "nota_corte_gp": [
        "nota_corte_gp",
        "nota_corte_grupo_preferencia",
    ],
    "idade": [
        "idade",
        "idade_inscrito",
        "idade_candidato",
    ],
    "data_nascimento": [
        "data_nascimento",
        "dt_nascimento",
        "nascimento",
    ],
    "nome_curso": [
        "nome_curso",
        "no_curso",
        "curso",
        "nome_do_curso",
    ],
    "ano": [
        "ano",
        "ano_processo_seletivo",
    ],
    "semestre": [
        "semestre",
        "semestre_processo_seletivo",
    ],
    "opcao_curso": [
        "opcao_curso",
        "opcoes_cursos_inscricao",
    ],
    "conceito_curso_gp": [
        "conceito_curso_gp",
        "conceito_curso",
        "conceito",
    ],
    "turno": [
        "turno",
    ],
    "ensino_medio_escola_publica": [
        "ensino_medio_escola_publica",
    ],
    "regiao_ies_alvo": [
        "regiao_ies_alvo",
        "regiao_ies",
    ],
    "natureza_juridica_mantenedora": [
        "natureza_juridica_mantenedora",
    ],
    "etnia_cor": [
        "etnia_cor",
        "raca_cor",
        "cor_raca",
        "cor_ou_raca",
    ],
    "sexo": [
        "sexo",
    ],
    "regiao_morar": [
        "regiao_morar",
        "regiao_residencia",
    ],
    "organizacao_academica": [
        "organizacao_academica",
        "organizacao_academica_ies",
    ],
    "subarea_conhecimento": [
        "subarea_conhecimento",
    ],
    "concluiu_curso_superior": [
        "concluiu_curso_superior",
        "conceito_concluiu_curso_superior",
    ],
    "beneficiado_creduc_fies": [
        "beneficiado_creduc_fies",
    ],
    "uf_local_oferta": [
        "uf_local_oferta",
        "sg_uf_local_oferta",
    ],
    "id_estudante": [
        "id_estudante",
        "cpf",
        "cpf_hash",
        "identificador_estudante",
    ],
}

COLUNAS_CANONICAS = [
    "id_estudante",
    "ano",
    "semestre",
    "situacao_fies",
    "nome_curso",
    "renda_per_capita",
    "media_enem",
    "nota_corte_gp",
    "gap",
    "renda_gap",
    "idade",
    "opcao_curso",
    "conceito_curso_gp",
    "turno",
    "ensino_medio_escola_publica",
    "regiao_ies_alvo",
    "natureza_juridica_mantenedora",
    "etnia_cor",
    "sexo",
    "regiao_morar",
    "organizacao_academica",
    "subarea_conhecimento",
    "concluiu_curso_superior",
    "beneficiado_creduc_fies",
    "uf_local_oferta",
    "target_binario",
]

CATEGORICAS = [
    "ano",
    "semestre",
    "opcao_curso",
    "conceito_curso_gp",
    "turno",
    "ensino_medio_escola_publica",
    "regiao_ies_alvo",
    "natureza_juridica_mantenedora",
    "etnia_cor",
    "sexo",
    "regiao_morar",
    "organizacao_academica",
    "subarea_conhecimento",
    "concluiu_curso_superior",
    "beneficiado_creduc_fies",
    "uf_local_oferta",
]

NUMERICAS = [
    "renda_per_capita",
    "gap",
    "renda_gap",
    "idade",
    "nota_corte_gp",
]

MAX_CATEGORIAS = 120


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "abt_logit_binario.log"

    with path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def remover_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(ch for ch in texto if not unicodedata.combining(ch))


def normalizar_texto(valor):
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip()
    texto = re.sub(r"\s+", " ", texto)

    if texto.upper() in {"", "NAN", "NONE", "NULL", "NA", "N/A", "-", "--"}:
        return pd.NA

    return texto.upper()


def normalizar_status(valor):
    texto = normalizar_texto(valor)

    if pd.isna(texto):
        return "OUTROS"

    texto_sem_acento = remover_acentos(texto).upper()

    mapa = {
        "CONTRATADA": STATUS_CONTRATADA,
        "NAO CONTRATADO": STATUS_NAO_CONTRATADO,
        "NÃO CONTRATADO": STATUS_NAO_CONTRATADO,
        "LISTA DE ESPERA": "LISTA DE ESPERA",
        "PRE-SELECIONADO": "PRÉ-SELECIONADO",
        "PRÉ-SELECIONADO": "PRÉ-SELECIONADO",
        "REJEITADA PELA CPSA": "REJEITADA PELA CPSA",
        "OPCAO NAO CONTRATADA": "OPÇÃO NÃO CONTRATADA",
        "OPÇÃO NÃO CONTRATADA": "OPÇÃO NÃO CONTRATADA",
        "PARTICIPACAO CANCELADA PELO CANDIDATO": "PARTICIPAÇÃO CANCELADA",
        "PARTICIPAÇÃO CANCELADA PELO CANDIDATO": "PARTICIPAÇÃO CANCELADA",
        "INSCRICAO POSTERGADA": "INSCRIÇÃO POSTERGADA",
        "INSCRIÇÃO POSTERGADA": "INSCRIÇÃO POSTERGADA",
    }

    return mapa.get(texto_sem_acento, "OUTROS")


def encontrar_coluna(df: pd.DataFrame, nome_canonico: str):
    for candidata in COLUMN_ALIASES.get(nome_canonico, [nome_canonico]):
        if candidata in df.columns:
            return candidata

    return None


def converter_numero_serie(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")

    s = (
        serie
        .astype("string")
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace("R$", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    tem_virgula = s.str.contains(",", regex=False, na=False)

    s_convertida = s.copy()
    s_convertida.loc[tem_virgula] = (
        s_convertida.loc[tem_virgula]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(s_convertida, errors="coerce")


def extrair_ano_nascimento(serie: pd.Series) -> pd.Series:
    texto = serie.astype("string")
    ano = texto.str.extract(r"(\d{4})", expand=False)

    return pd.to_numeric(ano, errors="coerce")


def carregar_base() -> pd.DataFrame:
    if not CURATED_INSCRICOES_ARTIGO_PATH.exists():
        raise FileNotFoundError(
            f"Base curada não encontrada: {CURATED_INSCRICOES_ARTIGO_PATH}. "
            "Rode primeiro: python3 main.py pipeline curate"
        )

    df = pd.read_parquet(CURATED_INSCRICOES_ARTIGO_PATH)

    log(f"[OK] Base carregada: {CURATED_INSCRICOES_ARTIGO_PATH}")
    log(f"[OK] Linhas: {len(df)} | Colunas: {len(df.columns)}")

    return df


def aplicar_recorte_medicina(df: pd.DataFrame) -> pd.DataFrame:
    coluna = encontrar_coluna(df, "nome_curso")

    if coluna is None:
        raise ValueError("Recorte Medicina solicitado, mas a coluna de nome do curso não foi encontrada.")

    nome_norm = (
        df[coluna]
        .astype("string")
        .fillna("")
        .map(lambda x: remover_acentos(x).upper().strip())
    )

    mascara = nome_norm.eq("MEDICINA")

    if mascara.sum() == 0:
        mascara = nome_norm.str.contains("MEDICINA", regex=False) & ~nome_norm.str.contains("VETERINARIA", regex=False)

    out = df[mascara].copy()

    log(f"[OK] Recorte Medicina aplicado | linhas: {len(out)}")

    return out


def construir_colunas_canonicas(df_base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(index=df_base.index)
    mapa_colunas = {}

    for nome in COLUNAS_CANONICAS:
        if nome in {"gap", "renda_gap", "target_binario"}:
            continue

        coluna_original = encontrar_coluna(df_base, nome)
        mapa_colunas[nome] = coluna_original

        if coluna_original is None:
            df[nome] = pd.NA
        else:
            df[nome] = df_base[coluna_original]

    mapa_colunas["data_nascimento"] = encontrar_coluna(df_base, "data_nascimento")

    df["situacao_fies"] = df["situacao_fies"].map(normalizar_status).astype("string")

    for coluna in ["renda_per_capita", "media_enem", "nota_corte_gp"]:
        df[coluna] = converter_numero_serie(df[coluna])

    if df["idade"].notna().sum() > 0:
        df["idade"] = converter_numero_serie(df["idade"])
    else:
        coluna_data = mapa_colunas["data_nascimento"]

        if coluna_data is not None:
            ano_nascimento = extrair_ano_nascimento(df_base[coluna_data])
            ano_processo = converter_numero_serie(df["ano"])
            df["idade"] = ano_processo - ano_nascimento
        else:
            df["idade"] = np.nan

    df["gap"] = df["media_enem"] - df["nota_corte_gp"]
    df["renda_gap"] = df["renda_per_capita"] * df["gap"]

    for coluna in CATEGORICAS:
        df[coluna] = df[coluna].astype("string")
        df[coluna] = (
            df[coluna]
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        )

    return df, mapa_colunas


def limitar_categorias(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    resumo = {}

    for coluna in CATEGORICAS:
        if coluna not in df.columns:
            continue

        n_categorias = int(df[coluna].nunique(dropna=True))

        if n_categorias > MAX_CATEGORIAS:
            top = df[coluna].value_counts(dropna=True).head(MAX_CATEGORIAS).index
            df[coluna] = df[coluna].where(df[coluna].isin(top), "Outras categorias")

            resumo[coluna] = {
                "categorias_originais": n_categorias,
                "tratamento": f"mantidas_top_{MAX_CATEGORIAS}_e_agrupadas_demais",
            }
        else:
            resumo[coluna] = {
                "categorias_originais": n_categorias,
                "tratamento": "mantida",
            }

    return df, resumo


def caminho_saida(recorte: str):
    if recorte == "geral":
        return ABT_BINARIA_GERAL_PATH

    if recorte == "medicina":
        return ABT_BINARIA_MEDICINA_PATH

    raise ValueError("recorte deve ser 'geral' ou 'medicina'.")


def construir_abt(recorte: str = "geral") -> tuple[pd.DataFrame, dict]:
    df_base = carregar_base()

    if recorte == "medicina":
        df_base = aplicar_recorte_medicina(df_base)
    elif recorte != "geral":
        raise ValueError("recorte deve ser 'geral' ou 'medicina'.")

    df, mapa_colunas = construir_colunas_canonicas(df_base)

    df = df[df["situacao_fies"].isin(TARGET_MAP.keys())].copy()
    df["target_binario"] = df["situacao_fies"].map(TARGET_MAP).astype(int)

    linhas_apos_target_antes_drop = len(df)

    obrigatorias = [
        "target_binario",
        "renda_per_capita",
        "gap",
        "renda_gap",
        "nota_corte_gp",
    ]

    df = df.dropna(subset=obrigatorias).copy()

    for coluna in NUMERICAS:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df, resumo_categorias = limitar_categorias(df)

    abt = df[COLUNAS_CANONICAS].reset_index(drop=True).copy()
    abt = abt.loc[:, ~abt.columns.duplicated()].copy()

    target_dist = (
        abt["target_binario"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    metadata = {
        "target": "binario",
        "target_col": "target_binario",
        "target_map": {
            "0": STATUS_NAO_CONTRATADO,
            "1": STATUS_CONTRATADA,
        },
        "recorte": recorte,
        "path_base": str(CURATED_INSCRICOES_ARTIGO_PATH),
        "linhas_apos_target_antes_drop": int(linhas_apos_target_antes_drop),
        "linhas_abt": int(len(abt)),
        "colunas_abt": int(len(abt.columns)),
        "distribuicao_target": {str(k): int(v) for k, v in target_dist.items()},
        "mapa_colunas_origem": mapa_colunas,
        "resumo_categorias": resumo_categorias,
        "colunas_numericas": [c for c in NUMERICAS if c in abt.columns],
        "colunas_categoricas": [c for c in CATEGORICAS if c in abt.columns],
        "variaveis_principais_obrigatorias": [
            "renda_per_capita",
            "gap",
            "renda_gap",
        ],
        "observacao": "ABT binária para regressão logística. Não inclui variáveis pós-contratação, como percentual de financiamento.",
    }

    return abt, metadata


def salvar_abt(recorte: str, abt: pd.DataFrame, metadata: dict) -> None:
    path = caminho_saida(recorte)
    path.parent.mkdir(parents=True, exist_ok=True)

    abt.to_parquet(path, index=False)

    meta_path = path.with_name(path.stem + "_metadata.json")

    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, default=str)

    resumo_path = LOGS_DIR / f"abt_logit_binario_{recorte}_resumo.csv"

    resumo = pd.DataFrame(
        [
            {"chave": key, "valor": json.dumps(value, ensure_ascii=False, default=str)}
            for key, value in metadata.items()
        ]
    )

    resumo.to_csv(resumo_path, index=False, encoding="utf-8")

    log(f"[OK] ABT salva em: {path}")
    log(f"[OK] Metadados salvos em: {meta_path}")
    log(f"[OK] Resumo salvo em: {resumo_path}")


def run(recorte: str = "geral") -> None:
    log("=" * 80)
    log(f"BUILD ABT LOGIT BINÁRIO | {recorte.upper()}")
    log("=" * 80)

    abt, metadata = construir_abt(recorte=recorte)
    salvar_abt(recorte=recorte, abt=abt, metadata=metadata)

    print(f"""
Resumo da ABT
-------------
target: binário
recorte: {recorte}
linhas: {metadata['linhas_abt']}
colunas: {metadata['colunas_abt']}
distribuição target: {metadata['distribuicao_target']}
""")


def parse_args():
    parser = argparse.ArgumentParser(description="Constrói ABT binária para regressão logística.")
    parser.add_argument("--recorte", choices=["geral", "medicina"], default="geral")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(recorte=args.recorte)
