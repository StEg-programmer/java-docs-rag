"""Explicitly enabled live CLI smoke; ordinary test runs never use services or models."""

import json
import os
import re
from pathlib import Path
from uuid import uuid4

import pytest

from java_docs_rag.cli import main


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("JAVA_DOCS_RAG_LIVE") != "1", reason="live smoke not enabled")
def test_live_index_answer_and_abstain(capsys: pytest.CaptureFixture[str]) -> None:
    model = os.environ.get("OLLAMA_MODEL")
    assert model, "Set OLLAMA_MODEL to an already provisioned local model"
    collection = f"java_docs_ab_smoke_{uuid4().hex}"
    fixture = Path("tests/fixtures/javadoc/java.base/java/util/List.html")
    source = "https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/List.html"
    document_id = "javase21:java.base/java/util/List"
    report_path = Path("experiments/reports") / f"{collection}.json"
    report: dict = {"collection": collection, "ollama_model": model, "passed": False}

    def run(label: str, arguments: list[str]) -> dict:
        code = main(arguments)
        output = capsys.readouterr()
        report[label] = {"exit_code": code, "stderr": output.err}
        if output.out:
            report[label]["output"] = json.loads(output.out)
        assert code == 0, f"{label} failed: {output.err}"
        return report[label]["output"]

    try:
        indexed = run(
            "index",
            [
                "index",
                "--html",
                str(fixture),
                "--source-url",
                source,
                "--model",
                "bge-small",
                "--collection",
                collection,
            ],
        )
        assert indexed["chunks"] > 0 and indexed["dimension"] == 384
        options = [
            "--model",
            "bge-small",
            "--collection",
            collection,
            "--ollama-model",
            model,
            "--timeout",
            "120",
        ]
        grounded = run("grounded", ["ask", "What does List.size() return?", *options])
        # Record both model responses even if one later fails a quality assertion.
        outside = run("unsupported", ["ask", "What is tomorrow's weather?", *options])
        # Context uses CLI defaults: 8000 chars, no similarity cutoff, top_k=10/top_n=5.
        assert not grounded["insufficient_context"]
        assert re.search(r"\b(number|count)\b", grounded["answer"], re.IGNORECASE)
        citations = [int(label) for label in re.findall(r"\[S(\d+)\]", grounded["answer"])]
        assert citations and all(1 <= label <= len(grounded["sources"]) for label in citations)
        supporting = [grounded["sources"][label - 1] for label in citations]
        assert any("number of elements" in chunk["text"].lower() for chunk in supporting)
        assert all(
            chunk["document_id"] == document_id and chunk["source_url"] == source
            for chunk in supporting
        )
        assert outside["insufficient_context"]
        assert outside["sources"]  # Normal evidence settings; not a forced-empty-context test.
        report["passed"] = True
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Live report: {report_path}")
