from __future__ import annotations

from src.agent.state import StepRecord


class ContextCompressor:
    def compress(self, history: list[StepRecord], max_text_steps: int = 10) -> list[str]:
        lines: list[str] = []
        for r in history[-max_text_steps:]:
            parts = [f"[{r.step}] {r.action}"]
            if r.bbox:
                parts.append(f"at ({r.bbox.x:.0f},{r.bbox.y:.0f})")
            if r.text:
                parts.append(f"text={r.text!r}")
            if r.url:
                parts.append(f"url={r.url}")
            if r.success is not None:
                parts.append(f"ok={r.success}")
            lines.append(" ".join(parts))
        return lines
