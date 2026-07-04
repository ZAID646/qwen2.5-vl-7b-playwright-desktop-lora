from __future__ import annotations

from src.agent.state import BoundingBox, StepRecord
from src.memory.context import ContextCompressor
from src.memory.history import summarize


def sr(*, step: int, action: str, **kw) -> StepRecord:
    return StepRecord(step=step, action=action, **kw)


def test_context_compressor():
    compressor = ContextCompressor()
    records = [
        sr(step=1, action="click", bbox=BoundingBox(100, 200, 50, 30), success=True),
        sr(step=2, action="type", bbox=BoundingBox(50, 60, 200, 40), text="hello", success=True),
        sr(step=3, action="done", success=None),
    ]
    compressed = compressor.compress(records)
    assert len(compressed) == 3
    assert "[1] click" in compressed[0]
    assert "[2] type" in compressed[1]


def test_history_summarize():
    records = [
        sr(step=1, action="click", bbox=BoundingBox(100, 200, 50, 30), success=True),
        sr(step=2, action="done", success=None),
    ]
    output = summarize(records)
    assert "Step | Action" in output
    assert "✓" in output
