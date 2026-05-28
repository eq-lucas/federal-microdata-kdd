from pathlib import Path


# =============================================================================
# Raiz do projeto
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


# =============================================================================
# Anos do pipeline e do artigo
# =============================================================================

PIPELINE_FIES_YEARS = [2019, 2020, 2021]
PIPELINE_INEP_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

ARTICLE_YEARS = [2019, 2020, 2021]


# =============================================================================
# Data lake local
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "01_raw"
RAW_FIES_DIR = RAW_DIR / "fies"
RAW_INEP_DIR = RAW_DIR / "inep"

STAGING_DIR = DATA_DIR / "02_staging"
STAGING_FIES_DIR = STAGING_DIR / "fies"
STAGING_FIES_ERRORS_DIR = STAGING_FIES_DIR / "errors"
STAGING_INEP_DIR = STAGING_DIR / "inep"

INTERIM_DIR = DATA_DIR / "03_interim"
INTERIM_FIES_DIR = INTERIM_DIR / "fies"
INTERIM_INEP_DIR = INTERIM_DIR / "inep"
TEMP_DIR = INTERIM_DIR / "temporarios"

CURATED_DIR = DATA_DIR / "04_curated"
CURATED_PARQUET_DIR = CURATED_DIR / "parquet"
CURATED_SQLITE_DIR = CURATED_DIR / "sqlite"

ANALYSIS_DIR = DATA_DIR / "05_analysis"

ABT_DIR = DATA_DIR / "06_abt"


# =============================================================================
# Bases curadas
# =============================================================================

CURATED_INSCRICOES_PATH = CURATED_PARQUET_DIR / "inscricoes_curated.parquet"
CURATED_OFERTAS_PATH = CURATED_PARQUET_DIR / "ofertas_curated.parquet"
CURATED_INSCRICOES_ARTIGO_PATH = CURATED_PARQUET_DIR / "inscricoes_fies_2019_2021.parquet"

CURATED_SQLITE_PATH = CURATED_SQLITE_DIR / "fies_curated.sqlite"


# =============================================================================
# Datasets analíticos derivados
# =============================================================================
#
# Convenção:
# - scripts em src/analysis/ começam com dataset_;
# - arquivos .parquet derivados em data/05_analysis/ começam com dataset_;
# - "taxas" é o nome correto da etapa analítica;
# - "taxas_conversao" é o nome correto da etapa final em src/article/.
# =============================================================================

ANALYSIS_DATASET_CANDIDATOS_UNICOS_PATH = (
    ANALYSIS_DIR / "dataset_candidatos_unicos_prioridade_fluxo.parquet"
)

ANALYSIS_DATASET_CANDIDATOS_UNICOS_AGREGADO_PATH = (
    ANALYSIS_DIR / "dataset_candidatos_unicos_prioridade_fluxo_agregado.parquet"
)

ANALYSIS_DATASET_FUNIL_PATH = (
    ANALYSIS_DIR / "dataset_funil_por_regiao.parquet"
)

ANALYSIS_DATASET_TAXAS_PATH = (
    ANALYSIS_DIR / "dataset_taxas.parquet"
)

# Aliases curtos para código novo.
ANALYSIS_CANDIDATOS_UNICOS_PATH = ANALYSIS_DATASET_CANDIDATOS_UNICOS_PATH
ANALYSIS_CANDIDATOS_UNICOS_AGREGADO_PATH = ANALYSIS_DATASET_CANDIDATOS_UNICOS_AGREGADO_PATH
ANALYSIS_FUNIL_PATH = ANALYSIS_DATASET_FUNIL_PATH
ANALYSIS_TAXAS_PATH = ANALYSIS_DATASET_TAXAS_PATH

# Alias de compatibilidade para scripts de artigo que ainda leem "funil".
ANALYSIS_FUNIL_FLUXO_PATH = ANALYSIS_DATASET_FUNIL_PATH


# =============================================================================
# ABTs
# =============================================================================

ABT_BINARIA_GERAL_PATH = ABT_DIR / "abt_contratacao_binaria_geral.parquet"
ABT_TERNARIA_GERAL_PATH = ABT_DIR / "abt_contratacao_ternaria_recorte_geral.parquet"
ABT_BINARIA_MEDICINA_PATH = ABT_DIR / "abt_contratacao_binaria_medicina.parquet"
ABT_TERNARIA_MEDICINA_PATH = ABT_DIR / "abt_contratacao_ternaria_recorte_medicina.parquet"


# =============================================================================
# Modelos
# =============================================================================

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_GENERAL_DIR = MODELS_DIR / "general"
MODELS_MEDICINA_DIR = MODELS_DIR / "medicina"


# =============================================================================
# Relatórios e saídas finais
# =============================================================================

REPORTS_DIR = PROJECT_ROOT / "reports"

ARTICLE_REPORTS_DIR = REPORTS_DIR / "article"
FIGURES_DIR = ARTICLE_REPORTS_DIR / "figures"
TABLES_DIR = ARTICLE_REPORTS_DIR / "tables"
APPENDIX_DIR = ARTICLE_REPORTS_DIR / "appendix"

DIAGNOSTICS_DIR = REPORTS_DIR / "diagnostics"
LOGS_DIR = REPORTS_DIR / "logs"
