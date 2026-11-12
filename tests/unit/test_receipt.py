"""Receipt polling uses bounded, deterministic fake HTTP responses and no credentials."""

from datetime import UTC, datetime

import httpx
import pytest

from java_docs_rag.observability.receipt import EXPECTED_STAGES, verify_receipts

IDS = {"index": "a" * 32, "ask": "b" * 32}


def rows(label):
    return [
        {
            "traceId": IDS[label],
            "name": name,
            "input": None,
            "output": {"duration_ms": 1.0, "outcome": "ok"},
        }
        for name in EXPECTED_STAGES[label]
    ]


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.paths = []

    def get(self, path, params):
        self.paths.append(path)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def run(client):
    elapsed = [0.0]

    def sleep(seconds):
        elapsed[0] += seconds

    result = verify_receipts(
        client, IDS, datetime.now(UTC), clock=lambda: elapsed[0], sleep=sleep, timeout=30
    )
    return result, elapsed[0]


def response(label):
    return httpx.Response(200, json={"data": rows(label)})


def test_delayed_v2_data_never_falls_back_to_legacy() -> None:
    client = Client([httpx.Response(200, json={"data": []}), response("ask"), response("index")])
    result, elapsed = run(client)
    assert result["receipt_verified"] and elapsed == 10
    assert all("/v2/" in path for path in client.paths)


def test_v1_is_used_only_when_v2_is_unsupported() -> None:
    client = Client([httpx.Response(404), response("index"), response("ask")])
    result, _ = run(client)
    assert result["receipt_verified"]
    assert client.paths == [
        "/api/public/v2/observations",
        "/api/public/observations",
        "/api/public/observations",
    ]


def test_rate_limit_honors_retry_after() -> None:
    client = Client(
        [httpx.Response(429, headers={"Retry-After": "5"}), response("index"), response("ask")]
    )
    result, elapsed = run(client)
    assert result["receipt_verified"] and elapsed >= 5


@pytest.mark.parametrize("code", [401, 403, 429, 500])
def test_api_failure_never_claims_receipt_or_echoes_response_body(code) -> None:
    result, _ = run(Client([httpx.Response(code, text="credential-SENSITIVE")]))
    assert not result["receipt_verified"] and str(code) in result["blocker"]
    assert "SENSITIVE" not in repr(result)


def test_unexpected_telemetry_content_is_rejected_without_echoing_it() -> None:
    leaked = rows("index")
    leaked[0]["input"] = "question-SENSITIVE"
    result, _ = run(Client([httpx.Response(200, json={"data": leaked})]))
    assert not result["receipt_verified"] and "unexpected" in result["blocker"].lower()
    assert "SENSITIVE" not in repr(result)


def test_transport_failure_has_only_an_error_type() -> None:
    result, _ = run(Client([httpx.ConnectError("credential-SENSITIVE")]))
    assert result["blocker"] == "Receipt transport failure (ConnectError)"
    assert not result["receipt_verified"]


def test_invalid_trace_ids_never_issue_an_unfiltered_read() -> None:
    client = Client([])
    result = verify_receipts(client, {"index": None, "ask": ""}, datetime.now(UTC))
    assert not result["receipt_verified"] and client.paths == []
