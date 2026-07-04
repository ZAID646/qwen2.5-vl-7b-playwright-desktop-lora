from __future__ import annotations

import re
from typing import Any

from src.agent.state import AgentState, BoundingBox, StepRecord, VisionOutput
from src.memory.context import ContextCompressor
from src.vision.mock import MockVLM
from src.vision.processor import process_screenshot


def parse_action_json(raw: str) -> dict[str, Any]:
    m = re.search(r"<action>(.*?)</action>", raw, re.DOTALL)
    if not m:
        return {"action": "done"}
    import json
    return json.loads(m.group(1))


def dict_to_vision_output(d: dict[str, Any]) -> VisionOutput:
    action = d.get("action", "done")
    bbox = None
    raw_bbox = d.get("bbox")
    if raw_bbox:
        if isinstance(raw_bbox, dict):
            bbox = BoundingBox(**raw_bbox)
        elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            bbox = BoundingBox.from_list(list(raw_bbox))
    text = d.get("text") or d.get("value")
    url = d.get("url")
    selector = d.get("selector") or d.get("xpath")
    scroll_direction = d.get("scroll_direction") or d.get("direction")
    confidence = d.get("confidence", 0.95)
    reasoning = d.get("reasoning", "")
    return VisionOutput(
        action=action, bbox=bbox, text=text, url=url,
        selector=selector, scroll_direction=scroll_direction,
        confidence=confidence, reasoning=reasoning,
    )


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
            selector=vision_output.selector,
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
        page = (await self._get_browser()).page

        try:
            if last.action == "click":
                await self._click(page, last)
            elif last.action == "type":
                await self._type(page, last)
            elif last.action == "navigate":
                await self._navigate(page, last)
            elif last.action == "scroll":
                await self._scroll(page, last)
            elif last.action == "wait":
                await page.wait_for_timeout(1000)
            last.success = True
        except Exception as e:
            last.success = False
            return {"error": str(e), "success": False}
        return {"current_url": page.url, "success": True}

    async def _click(self, page, record: StepRecord) -> None:
        sel = record.selector
        if sel:
            for prefix in ["", "xpath="]:
                try:
                    loc = page.locator(prefix + sel)
                    if await loc.count() > 0:
                        await loc.first.click()
                        return
                except Exception:
                    continue
        if record.bbox:
            x, y = record.bbox.center()
            await page.mouse.click(x, y)
            return
        raise RuntimeError(f"Click action missing both selector and bbox (action={record.action})")

    async def _type(self, page, record: StepRecord) -> None:
        text = record.text or ""
        sel = record.selector
        if sel:
            for prefix in ["", "xpath="]:
                try:
                    loc = page.locator(prefix + sel)
                    if await loc.count() > 0:
                        await loc.first.fill(text)
                        return
                except Exception:
                    continue
        if record.bbox:
            x, y = record.bbox.center()
            await page.mouse.click(x, y)
            await page.keyboard.type(text)
            return
        raise RuntimeError(f"Type action missing both selector and bbox (action={record.action})")

    async def _navigate(self, page, record: StepRecord) -> None:
        url = record.url or ""
        if not url.startswith("http"):
            current = page.url
            base = current.rstrip("/").rsplit("/", 1)[0] if "/" in current else current
            url = base + url if url.startswith("/") else f"{base}/{url}"
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

    async def _scroll(self, page, record: StepRecord) -> None:
        direction = record.scroll_direction or "down"
        dy = 300 if direction == "down" else -300
        await page.evaluate(f"window.scrollBy(0, {dy})")

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
