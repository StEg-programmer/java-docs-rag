"""Developer A: load a versioned evaluation dataset."""

from pathlib import Path

from java_docs_rag.common.models import EvaluationQuestion


def load_dataset(path: Path) -> list[EvaluationQuestion]:
    raise NotImplementedError("Evaluation dataset loading is not implemented")
