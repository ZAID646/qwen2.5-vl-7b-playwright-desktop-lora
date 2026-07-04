from __future__ import annotations

from src.harness.metrics import TaskResult, compute_report, print_report


def _tr(**kw) -> TaskResult:
    defaults = dict(scenario_id="x", success=True, steps=3,
                    optimal_steps=3, total_tokens=100, self_corrected=False)
    return TaskResult(**{**defaults, **kw})


def test_compute_report_empty():
    report = compute_report([])
    assert report.total_tasks == 0
    assert report.tcr == 0.0


def test_compute_report_all_success():
    results = [_tr(scenario_id="s001"), _tr(scenario_id="s002", steps=5)]
    report = compute_report(results)
    assert report.total_tasks == 2
    assert report.tcr == 1.0
    assert report.tfi == 100.0


def test_compute_report_mixed():
    results = [
        _tr(scenario_id="s001"),
        _tr(scenario_id="s002", success=False, steps=10, total_tokens=500),
    ]
    report = compute_report(results)
    assert report.tcr == 0.5
    assert report.ser > 0.0


def test_print_report():
    results = [_tr(scenario_id="s001", self_corrected=True)]
    report = compute_report(results)
    output = print_report(report)
    assert "TCR" in output
    assert "SER" in output
    assert "TFI" in output
    assert "SCRR" in output
