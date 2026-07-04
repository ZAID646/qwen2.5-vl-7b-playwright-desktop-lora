from __future__ import annotations

from src.agent.state import StepRecord


def summarize(records: list[StepRecord]) -> str:
    lines = ["Step | Action     | Target         | Outcome", "-" * 50]
    for r in records:
        target = ""
        if r.bbox:
            target = f"({r.bbox.x:.0f},{r.bbox.y:.0f})"
        if r.text:
            target = f"{target} {r.text}"
        if r.url:
            target = r.url

        outcome = "✓" if r.success else ("✗" if r.success is False else "?")
        lines.append(f"  {r.step}  | {r.action:<10} | {target:<14} | {outcome}")
    return "\n".join(lines)
