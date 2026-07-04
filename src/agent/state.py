from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VisionAction = Literal["click", "type", "navigate", "scroll", "wait", "done"]


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class VisionOutput:
    action: VisionAction
    bbox: BoundingBox | None = None
    text: str | None = None
    url: str | None = None
    scroll_direction: Literal["up", "down"] | None = None
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class StepRecord:
    step: int
    action: VisionAction
    bbox: BoundingBox | None = None
    text: str | None = None
    url: str | None = None
    scroll_direction: Literal["up", "down"] | None = None
    confidence: float = 0.0
    reasoning: str = ""
    screenshot_path: str = ""
    success: bool | None = None


@dataclass
class AgentState:
    task: str
    current_url: str = ""
    screenshot_path: str = ""
    dom_snapshot: str = ""
    history: list[StepRecord] = field(default_factory=list)
    step: int = 0
    done: bool = False
    success: bool | None = None
    error: str | None = None
