from pathlib import Path
import sys

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.modeling.treeClassification_profundidade_utils import parse_args_modelagem, run_modelagem


PROFUNDIDADE = 19
TARGET = "binario"
AVALIACAO = "in_sample"


def run(recorte: str = "geral", force: bool = False) -> None:
    run_modelagem(
        target=TARGET,
        recorte=recorte,
        avaliacao=AVALIACAO,
        force=force,
        profundidade=PROFUNDIDADE,
    )


if __name__ == "__main__":
    args = parse_args_modelagem()
    run(recorte=args.recorte, force=args.force)
