import re
import shutil
from pathlib import Path

import pandas as pd

from src.constants import (
    LOGS_DIR,
    PIPELINE_FIES_YEARS,
    PIPELINE_INEP_YEARS,
    RAW_FIES_DIR,
    RAW_INEP_DIR,
    STAGING_FIES_DIR,
    STAGING_FIES_ERRORS_DIR,
    STAGING_INEP_DIR,
)


def safe_text(value) -> str:
    """
    Converte qualquer valor para texto seguro para terminal/log.

    Alguns nomes extraídos de arquivos compactados podem conter caracteres
    inválidos para UTF-8 estrito. Isso não deve derrubar o pipeline.
    """
    text = str(value)
    return text.encode("utf-8", errors="replace").decode("utf-8")


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "staging.log"

    message = safe_text(message)

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(message + "\n")

    print(message)


def read_fies_csv(path: Path) -> pd.DataFrame:
    """
    Lê o CSV bruto do FIES preservando os valores como texto.

    Conversões de tipo ficam para src/pipeline/transform/.
    """
    encodings = ["latin-1", "utf-8-sig", "utf-8"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                sep=";",
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                low_memory=False,
            )
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(f"Não foi possível ler {path.name}. Último erro: {last_error}")


def remove_colunas_fantasma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove apenas colunas artificiais de índice exportado, como Unnamed: 0.

    Não renomeia nem converte colunas originais.
    """
    colunas = [col for col in df.columns if str(col).startswith("Unnamed:")]

    if not colunas:
        return df

    return df.drop(columns=colunas)


def nome_saida_fies(nome_original: str) -> tuple[str, int] | tuple[None, None]:
    """
    Identifica arquivos de inscrição/oferta do FIES e gera nome padronizado.

    Retorna:
        (nome_saida, ano)

    Se o arquivo não seguir o padrão esperado:
        (None, None)
    """
    nome = nome_original.lower()

    match_inscricao = re.match(
        r"relatorio_inscricao_dados_abertos_fies_([12])(\d{4})\.csv$",
        nome,
    )

    if match_inscricao:
        semestre = match_inscricao.group(1)
        ano = int(match_inscricao.group(2))
        return f"fies_inscricoes_{ano}_{semestre}.csv", ano

    match_oferta = re.match(
        r"relatorio_dados_abertos_oferta_([12])(\d{4})(?:_\d+)?\.csv$",
        nome,
    )

    if match_oferta:
        semestre = match_oferta.group(1)
        ano = int(match_oferta.group(2))
        return f"fies_ofertas_{ano}_{semestre}.csv", ano

    return None, None


def copiar_para_errors(origem: Path, motivo: str) -> None:
    STAGING_FIES_ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    destino = STAGING_FIES_ERRORS_DIR / origem.name
    shutil.copy2(origem, destino)
    log(f"[{motivo}] {origem.name} -> {destino.name}")


def staging_fies(overwrite: bool = True) -> None:
    """
    Executa o staging dos arquivos FIES encontrados em data/01_raw/fies.

    Esta etapa:
    - reconhece inscrições e ofertas;
    - aceita os anos definidos em PIPELINE_FIES_YEARS;
    - preserva os dados como texto;
    - remove apenas duplicatas exatas;
    - remove apenas colunas fantasma tipo Unnamed;
    - envia arquivos de resultado/erro/não classificados para errors.

    Observação:
    - Ofertas ausentes para determinado ano não são problema no staging.
    - A decisão de cruzar inscrições e ofertas por mesmo ano/semestre fica nas transformações.
    """
    STAGING_FIES_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_FIES_ERRORS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("STAGING FIES")
    log("=" * 80)

    arquivos = sorted(RAW_FIES_DIR.glob("*.csv"))

    processados = 0
    enviados_errors = 0
    ignorados_por_ano = 0
    erros = 0

    anos_aceitos = set(PIPELINE_FIES_YEARS)

    for origem in arquivos:
        nome_original = origem.name
        nome_lower = nome_original.lower()

        if "resultado" in nome_lower or "erro" in nome_lower:
            copiar_para_errors(origem, "ERROS")
            enviados_errors += 1
            continue

        nome_saida, ano = nome_saida_fies(nome_original)

        if nome_saida is None:
            copiar_para_errors(origem, "NÃO CLASSIFICADO")
            enviados_errors += 1
            continue

        if ano not in anos_aceitos:
            log(f"[IGNORADO] {nome_original} | ano fora de PIPELINE_FIES_YEARS: {ano}")
            ignorados_por_ano += 1
            continue

        destino = STAGING_FIES_DIR / nome_saida

        if destino.exists() and not overwrite:
            log(f"[PULADO] Já existe: {destino.name}")
            continue

        try:
            df = read_fies_csv(origem)

            linhas_originais = len(df)
            colunas_originais = len(df.columns)

            df = remove_colunas_fantasma(df)
            df = df.drop_duplicates()

            linhas_finais = len(df)
            duplicatas = linhas_originais - linhas_finais

            df.to_csv(destino, index=False, encoding="utf-8")

            log(
                f"[OK] {nome_original} -> {nome_saida} | "
                f"linhas: {linhas_originais} -> {linhas_finais} | "
                f"duplicatas removidas: {duplicatas} | "
                f"colunas: {colunas_originais} -> {len(df.columns)}"
            )

            processados += 1

        except Exception as error:
            copiar_para_errors(origem, "ERRO")
            log(f"       Motivo: {safe_text(error)}")
            erros += 1

    log(
        f"FIES concluído. Processados: {processados}. "
        f"Enviados para errors: {enviados_errors}. "
        f"Ignorados por ano: {ignorados_por_ano}. "
        f"Erros: {erros}."
    )
    log("")


def encontrar_pasta_inep_ano(ano: int) -> Path | None:
    if not RAW_INEP_DIR.exists():
        return None

    candidatos = [
        item for item in RAW_INEP_DIR.iterdir()
        if item.is_dir() and str(ano) in item.name
    ]

    if not candidatos:
        return None

    return sorted(candidatos)[0]


def encontrar_arquivo_case_insensitive(pasta: Path, nome_esperado: str) -> Path | None:
    """
    Procura arquivo sem depender de caixa alta/baixa.

    Isso ajuda porque os microdados podem vir com .CSV, .csv ou variações.
    """
    nome_esperado_lower = nome_esperado.lower()

    for item in pasta.iterdir():
        if item.is_file() and item.name.lower() == nome_esperado_lower:
            return item

    return None


def staging_inep(overwrite: bool = True) -> None:
    STAGING_INEP_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("STAGING INEP")
    log("=" * 80)

    templates = [
        "MICRODADOS_CADASTRO_CURSOS_{ano}.CSV",
        "MICRODADOS_CADASTRO_IES_{ano}.CSV",
    ]

    copiados = 0
    ausentes = 0

    for ano in PIPELINE_INEP_YEARS:
        pasta_ano = encontrar_pasta_inep_ano(ano)

        if pasta_ano is None:
            log(f"[AVISO] Pasta INEP não encontrada para {ano}.")
            ausentes += 1
            continue

        pasta_dados = pasta_ano / "dados"

        if not pasta_dados.exists():
            log(f"[AVISO] Subpasta 'dados' não encontrada em {safe_text(pasta_ano)}.")
            ausentes += 1
            continue

        for template in templates:
            nome_original = template.format(ano=ano)
            origem = encontrar_arquivo_case_insensitive(pasta_dados, nome_original)

            if origem is None:
                log(
                    f"[AVISO] Arquivo INEP não encontrado: "
                    f"{safe_text(pasta_dados / nome_original)}"
                )
                ausentes += 1
                continue

            destino = STAGING_INEP_DIR / origem.name.lower()

            if destino.exists() and not overwrite:
                log(f"[PULADO] Já existe: {destino.name}")
                continue

            shutil.copy2(origem, destino)
            log(f"[OK] {safe_text(origem.name)} -> {safe_text(destino.name)}")
            copiados += 1

    log(f"INEP concluído. Copiados: {copiados}. Ausentes: {ausentes}.")
    log("")


def run(overwrite: bool = True) -> None:
    staging_fies(overwrite=overwrite)
    staging_inep(overwrite=overwrite)