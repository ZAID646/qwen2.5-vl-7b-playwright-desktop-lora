from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UIExample:
    instruction: str
    screenshot_path: str
    action: str
    bbox: tuple[float, float, float, float] | None = None
    text: str | None = None


UI_DATASET: list[UIExample] = [
    UIExample(
        instruction="Click the login button",
        screenshot_path="data/ui/login_page.png",
        action="click",
        bbox=(450, 380, 120, 40),
    ),
    UIExample(
        instruction="Type email into the email field",
        screenshot_path="data/ui/login_page.png",
        action="type",
        bbox=(400, 300, 200, 36),
        text="user@example.com",
    ),
    UIExample(
        instruction="Navigate to the settings page",
        screenshot_path="data/ui/dashboard.png",
        action="navigate",
        text="/settings",
    ),
]
