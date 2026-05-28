import re
from pathlib import Path

import pandas as pd

from src.constants import (
    INTERIM_FIES_DIR,
    LOGS_DIR,
    STAGING_FIES_DIR,
)


# =============================================================================
# Mapeamento explícito de colunas do FIES
# =============================================================================

MAPA_INSCRICOES = {
    "Ano do processo seletivo": "ano_processo_seletivo",
    "Semestre do processo seletivo": "semestre_processo_seletivo",
    "Cod. do Grupo de preferência": "codigo_grupo_preferencia",
    "Classificação": "classificacao",
    "ID do estudante": "id_estudante",
    "Sexo": "sexo",
    "Ano de Nascimento": "ano_nascimento",
    "Data de Nascimento": "data_nascimento",
    "UF de residência": "uf_residencia",
    "Municipio de residência": "municipio_residencia",
    "Etnia/Cor": "etnia_cor",
    "Pessoa com deficiência?": "pessoa_com_deficiencia",
    "Concluiu ensino médio escola pública": "concluiu_ensino_medio_escola_publica",
    "Ano conclusão ensino médio": "ano_conclusao_ensino_medio",
    "Concluiu curso superior?": "concluiu_curso_superior",
    "Beneficiado pelo Creduc ou Fies": "beneficiado_creduc_ou_fies",
    "Professor rede pública ensino?": "professor_rede_publica_ensino",
    "Nº de membros Grupo Familiar": "numero_membros_grupo_familiar",
    "Renda familiar mensal bruta": "renda_familiar_mensal_bruta",
    "Renda mensal bruta per capita": "renda_mensal_bruta_per_capita",
    "Região grupo de preferência": "regiao_grupo_preferencia",
    "UF": "uf_grupo_preferencia",
    "Cod.Microrregião": "codigo_microrregiao",
    "Microrregião": "microrregiao",
    "Cod.Mesorregião": "codigo_mesorregiao",
    "Mesorregião": "mesorregiao",
    "Conceito de curso do GP": "conceito_curso_gp",
    "Área do conhecimento": "area_conhecimento",
    "Subárea do conhecimento": "subarea_conhecimento",
    "Nota Corte Grupo Preferência": "nota_corte_grupo_preferencia",
    "Opções de cursos da inscrição": "opcoes_cursos_inscricao",
    "Nome mantenedora": "nome_mantenedora",
    "Natureza Jurídica Mantenedora": "natureza_juridica_mantenedora",
    "CNPJ da mantenedora": "cnpj_mantenedora",
    "Código e-MEC da Mantenedora": "codigo_e_mec_mantenedora",
    "Nome da IES": "nome_ies",
    "Código e-MEC da IES": "codigo_e_mec_ies",
    "Organização Acadêmica da IES": "organizacao_academica_ies",
    "Município da IES": "municipio_ies",
    "UF da IES": "uf_ies",
    "Nome do Local de oferta": "nome_local_oferta",
    "Código do Local de Oferta": "codigo_local_oferta",
    "Munícipio do Local de Oferta": "municipio_local_oferta",
    "Município do Local de Oferta": "municipio_local_oferta",
    "UF do Local de Oferta": "uf_local_oferta",
    "Código do curso": "codigo_curso",
    "Código do Curso": "codigo_curso",
    "Nome do curso": "nome_curso",
    "Nome do Curso": "nome_curso",
    "Turno": "turno",
    "Grau": "grau",
    "Conceito": "conceito",
    "Média nota Enem": "media_nota_enem",
    "Ano do Enem": "ano_enem",
    "Redação": "nota_redacao",
    "Matemática e suas Tecnologias": "nota_matematica",
    "Linguagens, Códigos e suas Tec": "nota_linguagens",
    "Ciências Natureza e suas Tec": "nota_ciencias_natureza",
    "Ciências Humanas e suas Tec": "nota_ciencias_humanas",
    "Situação Inscrição Fies": "situacao_inscricao_fies",
    "Percentual de financiamento": "percentual_financiamento",
    "Semestre do financiamento": "semestre_financiamento",
    "Qtde semestre financiado": "qtde_semestre_financiado",
}


MAPA_OFERTAS = {
    "Ano": "ano",
    "Semestre": "semestre",
    "Nome Mantenedora": "nome_mantenedora",
    "Código e-MEC da Mantenedora": "codigo_e_mec_mantenedora",
    "CNPJ da mantenedora": "cnpj_mantenedora",
    "Nome da IES": "nome_ies",
    "Código e-MEC da IES": "codigo_e_mec_ies",
    "Organização Acadêmica da IES": "organizacao_academica_ies",
    "UF da IES": "uf_ies",
    "Município da IES": "municipio_ies",
    "Nome do Local de oferta": "nome_local_oferta",
    "Código do Local de Oferta": "codigo_local_oferta",
    "Município do Local de Oferta": "municipio_local_oferta",
    "Munícipio do Local de Oferta": "municipio_local_oferta",
    "UF do Local de Oferta": "uf_local_oferta",
    "Nome da Microrregião": "nome_microrregiao",
    "Nome da Microregião": "nome_microrregiao",
    "Código da Microrregião": "codigo_microrregiao",
    "Código da Microregião": "codigo_microrregiao",
    "Código da Mesorregião": "codigo_mesorregiao",
    "Nome da Mesorregião": "nome_mesorregiao",
    "Área do conhecimento": "area_conhecimento",
    "Subárea do conhecimento": "subarea_conhecimento",
    "Código do Grupo de Preferência": "codigo_grupo_preferencia",
    "Cód. Do Grupo de Preferência": "codigo_grupo_preferencia",
    "Nota de Corte Grupo Preferência": "nota_corte_grupo_preferencia",
    "Nota de Corte GP": "nota_corte_grupo_preferencia",
    "Código do Curso": "codigo_curso",
    "Nome do Curso": "nome_curso",
    "Turno": "turno",
    "Grau": "grau",
    "Conceito": "conceito",
    "Vagas autorizadas e-mec": "vagas_autorizadas_e_mec",
    "Vagas ofertadas FIES": "vagas_ofertadas_fies",
    "Vagas além da Oferta": "vagas_alem_da_oferta",
    "Vagas ocupadas": "vagas_ocupadas",
    "Participa do P-FIES": "participa_p_fies",
    "Vagas Ofertadas P-FIES": "vagas_ofertadas_p_fies",
    "BANCO NORDESTE BRASIL (004)": "banco_nordeste_brasil_004",
    "BANCO NORDESTE BRASIL   (004)": "banco_nordeste_brasil_004",
    "ITAU UNIBANCO (PRAVALER)(341)": "itau_unibanco_pravaler_341",
    "BV FINANCEIRA (PRAVALER)(455)": "bv_financeira_pravaler_455",
    "BANCO ANDBANK (PRAVALER)(65)": "banco_andbank_pravaler_65",
    "BANCO DA AMAZONIA S.A. (003)": "banco_amazonia_sa_003",
    "BANCO DA AMAZONIA S.A.  (003)": "banco_amazonia_sa_003",
    "Valor bruto do curso": "valor_bruto_curso",
    "Valor do curso para FIES": "valor_curso_fies",
    "Valor do curso para o FIES": "valor_curso_fies",
    "Valor bruto do curso para FIES": "valor_curso_fies",
    "Índice de correção - IPCA": "indice_correcao_ipca",
}


for i in range(1, 13):
    MAPA_OFERTAS[f"{i} Semestre Bruto"] = f"semestre_{i}_bruto"
    MAPA_OFERTAS[f"{i} Semestre FIES"] = f"semestre_{i}_fies"
    MAPA_OFERTAS[f"{i} Semestre Fies"] = f"semestre_{i}_fies"


INT_COLS_INSCRICOES = [
    "ano_processo_seletivo",
    "semestre_processo_seletivo",
    "codigo_grupo_preferencia",
    "classificacao",
    "id_estudante",
    "ano_nascimento",
    "ano_conclusao_ensino_medio",
    "numero_membros_grupo_familiar",
    "codigo_microrregiao",
    "codigo_mesorregiao",
    "conceito_curso_gp",
    "opcoes_cursos_inscricao",
    "codigo_e_mec_mantenedora",
    "codigo_e_mec_ies",
    "codigo_local_oferta",
    "codigo_curso",
    "conceito",
    "ano_enem",
    "qtde_semestre_financiado",
    "ano_arquivo",
    "semestre_arquivo",
]

FLOAT_COLS_INSCRICOES = [
    "renda_familiar_mensal_bruta",
    "renda_mensal_bruta_per_capita",
    "nota_corte_grupo_preferencia",
    "media_nota_enem",
    "nota_redacao",
    "nota_matematica",
    "nota_linguagens",
    "nota_ciencias_natureza",
    "nota_ciencias_humanas",
    "percentual_financiamento",
]

INT_COLS_OFERTAS = [
    "ano",
    "semestre",
    "codigo_e_mec_mantenedora",
    "codigo_e_mec_ies",
    "codigo_local_oferta",
    "codigo_microrregiao",
    "codigo_mesorregiao",
    "codigo_grupo_preferencia",
    "codigo_curso",
    "conceito",
    "vagas_autorizadas_e_mec",
    "vagas_ofertadas_fies",
    "vagas_alem_da_oferta",
    "vagas_ocupadas",
    "vagas_ofertadas_p_fies",
    "ano_arquivo",
    "semestre_arquivo",
]

FLOAT_COLS_OFERTAS = [
    "nota_corte_grupo_preferencia",
    "valor_bruto_curso",
    "valor_curso_fies",
    "indice_correcao_ipca",
] + [f"semestre_{i}_bruto" for i in range(1, 13)] + [f"semestre_{i}_fies" for i in range(1, 13)]


COLUNAS_EVENTUAIS_OFERTAS = [
    "participa_p_fies",
    "vagas_ofertadas_p_fies",
    "banco_nordeste_brasil_004",
    "itau_unibanco_pravaler_341",
    "bv_financeira_pravaler_455",
    "banco_andbank_pravaler_65",
    "banco_amazonia_sa_003",
]


# =============================================================================
# Auxiliares
# =============================================================================

def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "transform_limpeza_tipos_fies.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def parse_staging_filename(path: Path) -> tuple[str, int, int]:
    match = re.match(r"fies_(inscricoes|ofertas)_(\d{4})_([12])\.csv$", path.name)

    if not match:
        raise ValueError(f"Nome de arquivo não reconhecido: {path.name}")

    return match.group(1), int(match.group(2)), int(match.group(3))


def limpar_nome_coluna(coluna: str) -> str:
    coluna = str(coluna).replace("\ufeff", "")
    coluna = re.sub(r"\s+", " ", coluna)
    return coluna.strip()


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


def limpar_cnpj(valor):
    """
    Mantém CNPJ como texto.

    Importante: alguns arquivos já chegam com CNPJ em notação científica
    no CSV, como 7,67E+13. Quando isso ocorre, não é seguro reconstruir
    o CNPJ original. Por isso, CNPJ não deve ser usado como chave principal.
    """
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip()

    if texto == "":
        return pd.NA

    return texto.upper()


def normalizar_semestre_financiamento(valor):
    """
    Converte valores como '1º', '2º', '3º' para inteiro.
    """
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip()

    if texto == "":
        return pd.NA

    match = re.search(r"\d+", texto)

    if not match:
        return pd.NA

    return int(match.group(0))


def converter_numero_br(serie: pd.Series) -> pd.Series:
    """
    Converte números em padrão brasileiro para float.

    Exemplos:
        2066,67 -> 2066.67
        84.692,00 -> 84692.00
        600 -> 600.0
        - -> NaN
    """
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
    numero = converter_numero_br(serie)
    return numero.round().astype("Int64")


def normalizar_textos(df: pd.DataFrame, ignorar: set[str]) -> pd.DataFrame:
    for col in df.columns:
        if col in ignorar:
            continue

        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(limpar_texto).astype("string")

    return df


def coalescer_colunas_duplicadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Se duas colunas originais forem renomeadas para o mesmo nome padronizado,
    combina pelo primeiro valor não nulo por linha.
    """
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


def garantir_colunas(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    for col in colunas:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def registrar_colunas_nao_mapeadas(df: pd.DataFrame, mapa: dict, arquivo: str) -> None:
    originais = set(df.columns)
    mapeadas = set(mapa.keys())
    nao_mapeadas = sorted(originais - mapeadas)

    if nao_mapeadas:
        log(f"[AVISO] {arquivo}: colunas sem mapeamento explícito: {nao_mapeadas}")


# =============================================================================
# Transformações
# =============================================================================

def transformar_inscricoes(
    df: pd.DataFrame,
    ano_arquivo: int,
    semestre_arquivo: int,
    arquivo_origem: str,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [limpar_nome_coluna(col) for col in df.columns]

    registrar_colunas_nao_mapeadas(df, MAPA_INSCRICOES, arquivo_origem)

    df = df.rename(columns=MAPA_INSCRICOES)
    df = coalescer_colunas_duplicadas(df)

    df["arquivo_origem"] = arquivo_origem
    df["ano_arquivo"] = ano_arquivo
    df["semestre_arquivo"] = semestre_arquivo

    if "data_nascimento" in df.columns:
        df["data_nascimento"] = pd.to_datetime(
            df["data_nascimento"],
            errors="coerce",
            dayfirst=True,
        )

        ano_derivado = df["data_nascimento"].dt.year

        if "ano_nascimento" not in df.columns:
            df["ano_nascimento"] = ano_derivado
        else:
            df["ano_nascimento"] = df["ano_nascimento"].where(
                df["ano_nascimento"].astype("string").str.strip().ne(""),
                ano_derivado,
            )
    else:
        df["data_nascimento"] = pd.NaT

    if "semestre_financiamento" in df.columns:
        df["semestre_financiamento"] = df["semestre_financiamento"].map(normalizar_semestre_financiamento).astype("Int64")

    for col in FLOAT_COLS_INSCRICOES:
        if col in df.columns:
            df[col] = converter_numero_br(df[col])

    for col in INT_COLS_INSCRICOES:
        if col in df.columns:
            df[col] = converter_inteiro(df[col])

    if "cnpj_mantenedora" in df.columns:
        df["cnpj_mantenedora"] = df["cnpj_mantenedora"].map(limpar_cnpj).astype("string")

    ignorar_texto = set(FLOAT_COLS_INSCRICOES + INT_COLS_INSCRICOES + ["data_nascimento", "semestre_financiamento"])
    df = normalizar_textos(df, ignorar=ignorar_texto)

    return df


def transformar_ofertas(
    df: pd.DataFrame,
    ano_arquivo: int,
    semestre_arquivo: int,
    arquivo_origem: str,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [limpar_nome_coluna(col) for col in df.columns]

    registrar_colunas_nao_mapeadas(df, MAPA_OFERTAS, arquivo_origem)

    df = df.rename(columns=MAPA_OFERTAS)
    df = coalescer_colunas_duplicadas(df)

    df["arquivo_origem"] = arquivo_origem
    df["ano_arquivo"] = ano_arquivo
    df["semestre_arquivo"] = semestre_arquivo

    df = garantir_colunas(df, COLUNAS_EVENTUAIS_OFERTAS)

    for col in FLOAT_COLS_OFERTAS:
        if col in df.columns:
            df[col] = converter_numero_br(df[col])

    for col in INT_COLS_OFERTAS:
        if col in df.columns:
            df[col] = converter_inteiro(df[col])

    if "cnpj_mantenedora" in df.columns:
        df["cnpj_mantenedora"] = df["cnpj_mantenedora"].map(limpar_cnpj).astype("string")

    ignorar_texto = set(FLOAT_COLS_OFERTAS + INT_COLS_OFERTAS)
    df = normalizar_textos(df, ignorar=ignorar_texto)

    return df


# =============================================================================
# Execução
# =============================================================================

def processar_arquivo(path: Path) -> dict:
    tipo, ano, semestre = parse_staging_filename(path)

    log(f"[INÍCIO] {path.name}")

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    linhas_entrada = len(df)
    colunas_entrada = len(df.columns)

    if tipo == "inscricoes":
        df_out = transformar_inscricoes(df, ano, semestre, path.name)
        nome_saida = f"fies_inscricoes_{ano}_{semestre}_limpo.parquet"

    elif tipo == "ofertas":
        df_out = transformar_ofertas(df, ano, semestre, path.name)
        nome_saida = f"fies_ofertas_{ano}_{semestre}_limpo.parquet"

    else:
        raise ValueError(f"Tipo não reconhecido: {tipo}")

    INTERIM_FIES_DIR.mkdir(parents=True, exist_ok=True)
    caminho_saida = INTERIM_FIES_DIR / nome_saida

    df_out.to_parquet(caminho_saida, index=False)

    log(
        f"[OK] {path.name} -> {nome_saida} | "
        f"linhas: {linhas_entrada} | "
        f"colunas: {colunas_entrada} -> {len(df_out.columns)}"
    )

    return {
        "arquivo_origem": path.name,
        "arquivo_saida": nome_saida,
        "tipo": tipo,
        "ano": ano,
        "semestre": semestre,
        "linhas": len(df_out),
        "colunas_entrada": colunas_entrada,
        "colunas_saida": len(df_out.columns),
    }


def run() -> None:
    log("=" * 80)
    log("TRANSFORM: LIMPEZA E TIPAGEM FIES")
    log("=" * 80)

    arquivos = sorted(STAGING_FIES_DIR.glob("fies_*.csv"))

    if not arquivos:
        log(f"[AVISO] Nenhum arquivo FIES encontrado em: {STAGING_FIES_DIR}")
        return

    registros = []

    for path in arquivos:
        try:
            registros.append(processar_arquivo(path))
        except Exception as error:
            log(f"[ERRO] {path.name}: {error}")

    if registros:
        df_log = pd.DataFrame(registros)
        caminho_log = LOGS_DIR / "transform_limpeza_tipos_fies_resumo.csv"
        df_log.to_csv(caminho_log, index=False, encoding="utf-8")
        log(f"Resumo salvo em: {caminho_log}")

    log("Transformação de limpeza/tipagem FIES concluída.")
