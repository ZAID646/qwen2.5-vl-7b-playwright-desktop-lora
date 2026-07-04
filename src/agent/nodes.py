from __future__ import annotations

from src.agent.state import AgentState, StepRecord
from src.memory.context import ContextCompressor
from src.vision.mock import MockVLM
from src.vision.processor import process_screenshot


class PerceptionNode:
    def __init__(self, vlm: MockVLM, compressor: ContextCompressor) -> None:
        self.vlm = vlm
        self.compressor = compressor

    async def __call__(self, state: AgentState) -> dict:
        screenshot = await self._capture(state)
        processed = process_screenshot(screenshot)
        dom = await self._get_dom(state)

        vision_output = await self.vlm.predict(
            image=processed,
            dom_snapshot=dom,
            history=self.compressor.compress(state.history),
            task=state.task,
        )

        record = StepRecord(
            step=state.step + 1,
            action=vision_output.action,
            bbox=vision_output.bbox,
            text=vision_output.text,
            url=vision_output.url,
            scroll_direction=vision_output.scroll_direction,
            confidence=vision_output.confidence,
            reasoning=vision_output.reasoning,
            screenshot_path=state.screenshot_path,
        )

        return {
            "history": [*state.history, record],
            "step": state.step + 1,
        }

    async def _capture(self, state: AgentState) -> bytes:
        from src.sandbox.browser import get_browser
        page = await get_browser().current_page()
        screenshot = await page.screenshot(full_page=False)
        state.screenshot_path = f"screenshots/step_{state.step + 1}.png"
        return screenshot

    async def _get_dom(self, state: AgentState) -> str:
        from src.sandbox.browser import get_browser
        page = await get_browser().current_page()
        dom = await page.content()
        state.dom_snapshot = dom[:5000]
        return state.dom_snapshot


class ActionNode:
    async def __call__(self, state: AgentState) -> dict:
        last = state.history[-1]
        browser = await self._get_browser()

        try:
            if last.action == "click" and last.bbox:
                x = last.bbox.x + last.bbox.width / 2
                y = last.bbox.y + last.bbox.height / 2
                await browser.page.mouse.click(x, y)

            elif last.action == "type" and last.bbox and last.text:
                x = last.bbox.x + last.bbox.width / 2
                y = last.bbox.y + last.bbox.height / 2
                await browser.page.mouse.click(x, y)
                await browser.page.keyboard.type(last.text)

            elif last.action == "navigate" and last.url:
                await browser.page.goto(last.url, wait_until="domcontentloaded")

            elif last.action == "scroll":
                dy = -300 if last.scroll_direction == "up" else 300
                await browser.page.evaluate(f"window.scrollBy(0, {dy})")

            elif last.action == "wait":
                await browser.page.wait_for_timeout(1000)

            last.success = True

        except Exception as e:
            last.success = False
            return {"error": str(e), "success": False}

        return {"current_url": browser.page.url, "success": True}

    async def _get_browser(self):
        from src.sandbox.browser import get_browser
        return get_browser()


class RouterNode:
    def __call__(self, state: AgentState) -> str:
        if state.error:
            return "error"

        last = state.history[-1] if state.history else None
        if last and last.action == "done":
            return "done"

        if state.step >= 20:
            return "done"

        return "continue"
