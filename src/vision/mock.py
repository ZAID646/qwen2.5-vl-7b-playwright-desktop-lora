from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from src.agent.state import BoundingBox, StepRecord, VisionOutput

ScenarioConfig = dict[
    str,
    dict[
        Literal["action", "bbox", "text", "url", "scroll_direction", "confidence"],
        str | list[float] | float | None,
    ],
]


class MockVLM:
    """Fake VLM for local testing. Returns structured predictions without a GPU.

    Loads optional scenario overrides from config/mock_scenarios.json.
    Falls back to random plausible actions when no scenario matches.
    """

    def __init__(self, scenarios_path: str | Path = "") -> None:
        self._scenarios: ScenarioConfig = {}
        if scenarios_path:
            path = Path(scenarios_path)
            if path.exists():
                self._scenarios = json.loads(path.read_text())

    async def predict(
        self,
        image: bytes,
        dom_snapshot: str,
        history: list[StepRecord],
        task: str,
    ) -> VisionOutput:
        scenario = self._find_scenario(task)
        if scenario:
            return self._build_from_scenario(scenario)
        return self._random_action()

    def _find_scenario(self, task: str) -> dict | None:
        for key, config in self._scenarios.items():
            if key.lower() in task.lower():
                return config
        return None

    def _build_from_scenario(self, scenario: dict) -> VisionOutput:
        return VisionOutput(
            action=scenario.get("action", "click"),
            bbox=BoundingBox(**scenario["bbox"]) if scenario.get("bbox") else None,
            text=scenario.get("text"),
            url=scenario.get("url"),
            scroll_direction=scenario.get("scroll_direction"),
            confidence=scenario.get("confidence", 0.95),
            reasoning="Mock response for local testing",
        )

    def _random_action(self) -> VisionOutput:
        action = random.choice(["click", "type", "navigate", "scroll", "wait", "done"])
        return VisionOutput(
            action=action,
            bbox=BoundingBox(
                x=random.uniform(100, 800),
                y=random.uniform(100, 600),
                width=random.uniform(20, 200),
                height=random.uniform(20, 60),
            ),
            confidence=random.uniform(0.85, 0.99),
            reasoning="Mock prediction (no GPU available)",
        )
