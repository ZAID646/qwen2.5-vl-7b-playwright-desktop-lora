from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.state import StepRecord


@dataclass
class TaskResult:
    scenario_id: str
    success: bool
    steps: int
    optimal_steps: int
    total_tokens: int
    self_corrected: bool
    history: list[StepRecord] = field(default_factory=list)


@dataclass
class HarnessReport:
    total_tasks: int
    tcr: float
    ser: float
    tfi: float
    scrr: float
    results: list[TaskResult] = field(default_factory=list)


def compute_report(results: list[TaskResult]) -> HarnessReport:
    n = len(results)
    if n == 0:
        return HarnessReport(total_tasks=0, tcr=0.0, ser=0.0, tfi=0.0, scrr=0.0)

    successes = [r for r in results if r.success]
    self_corrected = [r for r in results if r.self_corrected]

    tcr = len(successes) / n
    ser = sum(r.optimal_steps / max(r.steps, 1) for r in results) / n
    tfi = sum(r.total_tokens for r in successes) / max(len(successes), 1)
    scrr = len(self_corrected) / n

    return HarnessReport(
        total_tasks=n,
        tcr=round(tcr, 4),
        ser=round(ser, 4),
        tfi=round(tfi, 2),
        scrr=round(scrr, 4),
        results=results,
    )


def print_report(report: HarnessReport) -> str:
    lines = [
        "=" * 50,
        "DETERMINISTIC SANDBOX HARNESS REPORT",
        "=" * 50,
        f"Total Tasks          : {report.total_tasks}",
        f"TCR (Completion Rate): {report.tcr:.2%}",
        f"SER (Step Efficiency) : {report.ser:.4f}",
        f"TFI (Token Friction)  : {report.tfi:.0f} tokens/success",
        f"SCRR (Self-Correction): {report.scrr:.2%}",
        "-" * 50,
    ]
    return "\n".join(lines)
