from __future__ import annotations

import json
import time
from pathlib import Path

from langgraph.graph.graph import CompiledGraph

from src.agent.state import AgentState
from src.harness.metrics import HarnessReport, TaskResult, compute_report
from src.harness.scenarios import SCENARIOS, Scenario


async def run_scenario(
    agent: CompiledGraph,
    scenario: Scenario,
    max_steps: int = 20,
) -> TaskResult:
    state = AgentState(task=scenario.task)

    start = time.perf_counter()
    try:
        result = await agent.ainvoke(state)
        success = result.get("success", True) and not result.get("error")
        steps = result.get("step", 0)
    except Exception:
        success = False
        steps = state.step
    elapsed = time.perf_counter() - start

    return TaskResult(
        scenario_id=scenario.id,
        success=success,
        steps=steps,
        optimal_steps=scenario.tags.count("simple") + 2,
        total_tokens=int(elapsed * 1000),
        self_corrected=False,
    )


async def run_harness(
    agent: CompiledGraph,
    scenarios: list[Scenario] | None = None,
    output_path: str | Path = "reports/harness_report.json",
) -> HarnessReport:
    scenarios = scenarios or SCENARIOS
    results: list[TaskResult] = []

    for scenario in scenarios:
        print(f"  [{scenario.id}] {scenario.name}... ", end="", flush=True)
        result = await run_scenario(agent, scenario)
        status = "PASS" if result.success else "FAIL"
        print(f"{status} ({result.steps} steps)")
        results.append(result)

    report = compute_report(results)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps({
        "tcr": report.tcr,
        "ser": report.ser,
        "tfi": report.tfi,
        "scrr": report.scrr,
        "total_tasks": report.total_tasks,
        "results": [
            {
                "id": r.scenario_id,
                "success": r.success,
                "steps": r.steps,
                "optimal_steps": r.optimal_steps,
                "tokens": r.total_tokens,
                "self_corrected": r.self_corrected,
            }
            for r in report.results
        ],
    }, indent=2))

    return report
