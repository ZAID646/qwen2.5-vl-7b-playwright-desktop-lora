from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UIExample:
    instruction: str
    screenshot_path: str
    action: str
    bbox: tuple[float, float, float, float] | None = None
    text: str | None = None
    selector: str | None = None
    url: str | None = None
    scroll_direction: str | None = None


UI_DATASET: list[UIExample] = [
    # --- Click actions ---
    UIExample("Click the login button", "data/ui/login_page.png", "click", bbox=(450, 380, 120, 40)),
    UIExample("Click submit", "data/ui/form_page.png", "click", bbox=(500, 600, 100, 40)),
    UIExample("Click first search result", "data/ui/search_page.png", "click", bbox=(100, 250, 800, 60)),
    UIExample("Click the sign up link", "data/ui/login_page.png", "click", selector="a[href='/signup']"),
    UIExample("Select dropdown", "data/ui/form_page.png", "click", bbox=(300, 400, 200, 40)),
    UIExample("Check checkbox", "data/ui/form_page.png", "click", bbox=(350, 500, 20, 20)),
    UIExample("Close modal", "data/ui/dashboard.png", "click", selector=".modal-close"),
    UIExample("Click next page", "data/ui/search_page.png", "click", selector="a.pagination-next"),
    # --- Type actions ---
    UIExample("Type email into the email field", "data/ui/login_page.png", "type", bbox=(400, 300, 200, 36), text="user@example.com"),
    UIExample("Search for AI news", "data/ui/search_page.png", "type", bbox=(200, 100, 400, 40), text="AI news"),
    UIExample("Fill search box", "data/ui/search_page.png", "type", bbox=(100, 80, 600, 36), text="query"),
    UIExample("Type password", "data/ui/login_page.png", "type", bbox=(400, 350, 200, 36), text="********"),
    UIExample("Enter username", "data/ui/login_page.png", "type", bbox=(400, 250, 200, 36), text="admin"),
    UIExample("Type message in chat", "data/ui/dashboard.png", "type", selector="#chat-input", text="Hello!"),
    UIExample("Enter coupon code", "data/ui/form_page.png", "type", bbox=(200, 450, 300, 36), text="SAVE20"),
    # --- Navigate actions ---
    UIExample("Navigate to the settings page", "data/ui/dashboard.png", "navigate", text="/settings"),
    UIExample("Go to dashboard", "data/ui/form_page.png", "navigate", text="/dashboard"),
    UIExample("Open profile", "data/ui/dashboard.png", "navigate", text="/profile"),
    UIExample("Go to home page", "data/ui/dashboard.png", "navigate", url="https://example.com"),
    # --- Scroll actions ---
    UIExample("Scroll down", "data/ui/dashboard.png", "scroll", scroll_direction="down"),
    UIExample("Scroll up", "data/ui/dashboard.png", "scroll", scroll_direction="up"),
    UIExample("Scroll the page down", "data/ui/search_page.png", "scroll", scroll_direction="down"),
    # --- Wait ---
    UIExample("Wait for results to load", "data/ui/search_page.png", "wait"),
]
