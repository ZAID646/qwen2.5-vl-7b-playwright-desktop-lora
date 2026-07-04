from __future__ import annotations

from pathlib import Path

import yaml
from playwright.async_api import Browser, Page, async_playwright


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
            args=["--no-sandbox"],
        )
        self._page = await self._browser.new_page(
            viewport=bc["viewport"],
        )
        self._page.set_default_timeout(bc["timeout"])

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
