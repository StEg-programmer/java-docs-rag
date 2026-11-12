"""Explicit Langfuse read-back verification; never part of the default application path."""

import json
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from java_docs_rag.observability.safe_tracing import _safe

EXPECTED_STAGES = {
    "index": {"index.command", "indexing.run", "indexing.embed_and_upsert"},
    "ask": {
        "ask.command",
        "rag.run",
        "rag.retrieve",
        "rag.similarity_filter",
        "rag.rerank",
        "rag.evidence_selection",
        "rag.generate",
    },
}


def verify_receipts(
    client: Any,
    trace_ids: Mapping[str, str],
    started: datetime,
    *,
    timeout: float = 180,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll exact trace IDs, checking stage presence and count-only I/O.

    Use v1 only when v2 is unsupported, not when v2 data is merely delayed.
    Honor numeric Retry-After within the deadline. Never return remote bodies,
    URLs, credentials or exception messages in diagnostic results.
    """
    result: dict[str, Any] = {
        "receipt_verified": False,
        "traces": {},
        "api_statuses": {},
        "pending_stages": {},
    }
    if (
        not trace_ids
        or set(trace_ids) != set(EXPECTED_STAGES)
        or not all(
            isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9-]{16,64}", value)
            for value in trace_ids.values()
        )
    ):
        result["blocker"] = "Both index and ask trace IDs are required"
        return result
    pending = dict(trace_ids)
    deadline = clock() + timeout
    endpoint = "/api/public/v2/observations"
    while pending and clock() < deadline:
        throttled = False
        for label, trace_id in list(pending.items()):
            params = {
                "traceId": trace_id,
                "limit": 100,
                "fromStartTime": started.isoformat(),
                "toStartTime": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
            }
            if "/v2/" in endpoint:
                params["fields"] = "core,basic,io"
            try:
                response = client.get(endpoint, params=params)
            except httpx.HTTPError as exc:
                result["blocker"] = f"Receipt transport failure ({type(exc).__name__})"
                return result
            result["api_statuses"][f"{label}:{endpoint}"] = response.status_code
            if response.status_code in {404, 405, 410} and "/v2/" in endpoint:
                endpoint = "/api/public/observations"
                throttled = True
                break
            if response.status_code == 429:
                retry = response.headers.get("Retry-After", "")
                if retry.isdigit():
                    result["retry_after_seconds"] = int(retry)
                if not retry.isdigit() or clock() + max(1, int(retry)) >= deadline:
                    result["blocker"] = (
                        "Receipt HTTP 429; Retry-After unavailable or exceeds deadline"
                    )
                    return result
                sleep(max(1, int(retry)))
                throttled = True
                break
            if response.status_code != 200:
                result["blocker"] = f"Receipt API {endpoint}: HTTP {response.status_code}"
                return result
            try:
                rows = [row for row in response.json()["data"] if row.get("traceId") == trace_id]
                missing = EXPECTED_STAGES[label] - {row.get("name") for row in rows}
                result["pending_stages"][label] = sorted(missing)
                if missing:
                    continue
                observations = []
                for row in rows:
                    if row.get("name") not in EXPECTED_STAGES[label]:
                        continue
                    fields = row.get("output")
                    fields = json.loads(fields) if isinstance(fields, str) else fields
                    if (
                        row.get("input") not in (None, "null", "")
                        or not isinstance(fields, dict)
                        or fields != _safe(fields)
                        or fields.get("duration_ms", -1) < 0
                        or "outcome" not in fields
                    ):
                        result["blocker"] = (
                            "Receipt contains unexpected I/O or missing duration/outcome"
                        )
                        return result
                    observations.append({"name": row["name"], "output": fields})
            except (KeyError, TypeError, ValueError):
                result["blocker"] = "Invalid observations response schema"
                return result
            result["traces"][label] = {
                "trace_id": trace_id,
                "receipt_api": endpoint,
                "observations": observations,
            }
            del pending[label]
        if pending and not throttled:
            sleep(min(10, max(0, deadline - clock())))
    result["receipt_verified"] = not pending
    if pending:
        result["blocker"] = "Expected observations not visible before the receipt deadline"
    return result


def verify_saved_report(path: Path, client: Any, *, timeout: float = 180) -> dict[str, Any]:
    """Recheck existing IDs without executing index/ask again; update a local report."""
    report = json.loads(path.read_text(encoding="utf-8"))
    ids = {}
    for label in EXPECTED_STAGES:
        values = report.get(label, {}).get("telemetry", {}).get("trace_ids") or [None]
        ids[label] = values[0]
    started = datetime.fromtimestamp(path.stat().st_mtime, UTC) - timedelta(minutes=30)
    verification = verify_receipts(client, ids, started, timeout=timeout)
    report["verification"] = verification
    report["receipt_verified"] = verification["receipt_verified"]
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return verification
