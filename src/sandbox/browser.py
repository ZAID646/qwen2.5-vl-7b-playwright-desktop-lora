from __future__ import annotations

from pathlib import Path

import yaml
from playwright.async_api import Browser, Page, async_playwright

STEALTH_ARGS = [
    "--no-sandbox",
    "--disable-blink-features=AutomationControlled",
    "--disable-automation",
    "--disable-dev-shm-usage",
    "--disable-web-security",
    "--disable-features=IsolateOrigins,site-per-process",
]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class BrowserManager:
    _instance: BrowserManager | None = None

    def __init__(self, config_path: str | Path = "config/sandbox.yaml") -> None:
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)
        self._playwright = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def start(self) -> None:
        bc = self._cfg["browser"]
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=bc["headless"],
            args=STEALTH_ARGS,
        )
        ctx = await self._browser.new_context(
            viewport=bc["viewport"],
            user_agent=USER_AGENT,
        )
        self._page = await ctx.new_page()
        self._page.set_default_timeout(bc["timeout"])
        await self._page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser not started — call start() first")
        return self._page

    async def current_page(self) -> Page:
        return self.page

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._browser = None
        self._playwright = None


def get_browser() -> BrowserManager:
    if BrowserManager._instance is None:
        BrowserManager._instance = BrowserManager()
    return BrowserManager._instance
