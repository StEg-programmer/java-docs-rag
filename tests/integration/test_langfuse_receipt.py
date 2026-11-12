"""Explicit live verification of count-only traces, separate from ordinary offline tests."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from java_docs_rag.cli import main
from java_docs_rag.observability.receipt import verify_receipts


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("JAVA_DOCS_RAG_LANGFUSE_LIVE") != "1",
    reason="Langfuse receipt verification not enabled",
)
def test_live_index_and_ask_traces_received(capsys: pytest.CaptureFixture[str]) -> None:
    required = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "OLLAMA_MODEL")
    assert all(os.environ.get(name) for name in required), "Configure required variables locally"
    collection = f"java_docs_langfuse_smoke_{uuid4().hex}"
    report_path = Path("experiments/reports") / f"{collection}.json"
    report: dict = {"collection": collection, "receipt_verified": False}
    started = datetime.now(UTC) - timedelta(minutes=2)

    def run(label: str, arguments: list[str]) -> list[str]:
        code = main(arguments)
        output = capsys.readouterr()
        # Never persist SDK stderr or raw CLI content: it is not count-only telemetry.
        assert code == 0, f"{label} command failed (exit {code}); inspect local service/config"
        data = json.loads(output.out)
        report[label] = {"exit_code": code, "telemetry": data.get("telemetry", {})}
        if label == "ask":
            assert not data["insufficient_context"] and data["sources"]
        ids = data.get("telemetry", {}).get("trace_ids", [])
        assert ids, f"{label} produced no trace ID; SDK/config/exporter may be unavailable"
        return ids

    try:
        index_ids = run(
            "index",
            [
                "index",
                "--html",
                "tests/fixtures/javadoc/java.base/java/util/List.html",
                "--source-url",
                "https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/List.html",
                "--model",
                "bge-small",
                "--collection",
                collection,
                "--trace",
            ],
        )
        ask_ids = run(
            "ask",
            [
                "ask",
                "What does List.size() return?",
                "--model",
                "bge-small",
                "--collection",
                collection,
                "--ollama-model",
                os.environ["OLLAMA_MODEL"],
                "--timeout",
                "120",
                "--trace",
            ],
        )
        with httpx.Client(
            base_url=os.environ["LANGFUSE_HOST"].rstrip("/"),
            timeout=10,
            auth=(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]),
        ) as client:
            verification = verify_receipts(
                client,
                {"index": index_ids[0], "ask": ask_ids[0]},
                started,
                timeout=float(os.environ.get("LANGFUSE_RECEIPT_TIMEOUT", "600")),
            )
        report["verification"] = verification
        report["receipt_verified"] = verification["receipt_verified"]
        assert report["receipt_verified"], verification.get("blocker", "Receipt not verified")
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Langfuse verification report: {report_path}")
