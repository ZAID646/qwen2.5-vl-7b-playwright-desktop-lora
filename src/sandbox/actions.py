from __future__ import annotations

from playwright.async_api import Page


async def click(page: Page, x: float, y: float) -> None:
    await page.mouse.click(x, y)


async def type_text(page: Page, x: float, y: float, text: str) -> None:
    await page.mouse.click(x, y)
    await page.keyboard.type(text)


async def navigate(page: Page, url: str) -> None:
    await page.goto(url, wait_until="domcontentloaded")


async def scroll(page: Page, direction: str) -> None:
    dy = -300 if direction == "up" else 300
    await page.evaluate(f"window.scrollBy(0, {dy})")


async def wait_for_page(page: Page, ms: int = 1000) -> None:
    await page.wait_for_timeout(ms)
