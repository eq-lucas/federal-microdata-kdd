from src.constants import (
    PROJECT_ROOT,
    SRC_DIR,
    RAW_FIES_DIR,
    RAW_INEP_DIR,
    STAGING_FIES_DIR,
    STAGING_FIES_ERRORS_DIR,
    STAGING_INEP_DIR,
    INTERIM_FIES_DIR,
    INTERIM_INEP_DIR,
    TEMP_DIR,
    CURATED_PARQUET_DIR,
    CURATED_SQLITE_DIR,
    ANALYSIS_DIR,
    ABT_DIR,
    MODELS_GENERAL_DIR,
    MODELS_MEDICINA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    APPENDIX_DIR,
    DIAGNOSTICS_DIR,
    LOGS_DIR,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASS_WEIGHT = "balanced"
FIGURE_DPI = 300


PROJECT_DIRS = [
    RAW_FIES_DIR,
    RAW_INEP_DIR,

    STAGING_FIES_DIR,
    STAGING_FIES_ERRORS_DIR,
    STAGING_INEP_DIR,

    INTERIM_FIES_DIR,
    INTERIM_INEP_DIR,
    TEMP_DIR,

    CURATED_PARQUET_DIR,
    CURATED_SQLITE_DIR,

    ANALYSIS_DIR,
    ABT_DIR,

    MODELS_GENERAL_DIR,
    MODELS_MEDICINA_DIR,

    FIGURES_DIR,
    TABLES_DIR,
    APPENDIX_DIR,
    DIAGNOSTICS_DIR,
    LOGS_DIR,

    FIGURES_DIR / "fluxo_selecao" / "funil",
    FIGURES_DIR / "taxas_conversao" / "taxa_conversao_inscritos",
    FIGURES_DIR / "taxas_conversao" / "taxa_conversao_curso_priorizado",
    TABLES_DIR / "taxas_conversao",
    TABLES_DIR / "secao_4_2",
    APPENDIX_DIR / "apendice_b",

    SRC_DIR / "pipeline",
    SRC_DIR / "pipeline" / "transform",
    SRC_DIR / "analysis",
    SRC_DIR / "abt",
    SRC_DIR / "modeling",
    SRC_DIR / "article",
]


PACKAGE_INIT_FILES = [
    SRC_DIR / "__init__.py",
    SRC_DIR / "pipeline" / "__init__.py",
    SRC_DIR / "pipeline" / "transform" / "__init__.py",
    SRC_DIR / "analysis" / "__init__.py",
    SRC_DIR / "abt" / "__init__.py",
    SRC_DIR / "modeling" / "__init__.py",
    SRC_DIR / "article" / "__init__.py",
]


MODULE_FILES = {
    SRC_DIR / "article" / "tabelas_distribuicao.py": 'def run() -> None:\n    raise NotImplementedError("Implementar Tabela 1 e Tabela B1.")\n',
}


def ensure_project_structure() -> None:
    for directory in PROJECT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

    for init_file in PACKAGE_INIT_FILES:
        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.touch(exist_ok=True)

    for module_file, content in MODULE_FILES.items():
        module_file.parent.mkdir(parents=True, exist_ok=True)

        if not module_file.exists():
            module_file.write_text(content, encoding="utf-8")

    print("Estrutura de pastas e arquivos-base verificada/criada com sucesso.")


def check_environment() -> None:
    ensure_project_structure()

    print()
    print("Raiz do projeto:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Camadas de dados:")
    print(f"  01_raw:      {RAW_FIES_DIR.parent}")
    print(f"  02_staging:  {STAGING_FIES_DIR.parent}")
    print(f"  03_interim:  {INTERIM_FIES_DIR.parent}")
    print(f"  04_curated:  {CURATED_PARQUET_DIR.parent}")
    print(f"  05_analysis: {ANALYSIS_DIR}")
    print(f"  06_abt:      {ABT_DIR}")

    print()
    print("Produtos finais do artigo:")
    print(f"  Tabela 1:    {TABLES_DIR / 'secao_4_2'}")
    print(f"  Tabela B1:   {APPENDIX_DIR / 'apendice_b'}")
    print(f"  Funil:       {FIGURES_DIR / 'fluxo_selecao' / 'funil'}")
    print(f"  Taxas:       {FIGURES_DIR / 'taxas_conversao'}")

    print()
    print("Configuração inicial concluída.")
