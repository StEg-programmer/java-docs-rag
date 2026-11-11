"""Developer B: minimal recording Langfuse v3 fake used by pipeline tests."""

from contextlib import contextmanager


class _Span:
    def update(self, **fields):
        pass


class RecordingSdk:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.flushed = 0

    @contextmanager
    def start_as_current_span(self, name, input=None, metadata=None):
        self.events.append(("start", name))
        yield _Span()
        self.events.append(("end", name))

    def flush(self) -> None:
        self.flushed += 1


def test_recording_sdk_records() -> None:
    sdk = RecordingSdk()
    with sdk.start_as_current_span("x"):
        pass
    assert sdk.events == [("start", "x"), ("end", "x")]
