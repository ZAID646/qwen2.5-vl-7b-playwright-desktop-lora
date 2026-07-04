from __future__ import annotations

from pathlib import Path

from playwright.async_api import Page


async def capture_screenshot(page: Page, path: str | Path) -> bytes:
    data = await page.screenshot(full_page=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return data


async def record_dom(page: Page) -> str:
    return await page.content()
