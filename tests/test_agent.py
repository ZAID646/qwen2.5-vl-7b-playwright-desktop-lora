from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from src.agent.graph import build_agent
from src.agent.nodes import RouterNode
from src.agent.state import AgentState, BoundingBox, StepRecord
from src.vision.mock import MockVLM


def _tiny_png() -> bytes:
    img = Image.new("RGB", (100, 100))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def agent():
    return build_agent(MockVLM())


@pytest.fixture
def mock_browser():
    mock_page = AsyncMock()
    mock_page.screenshot = AsyncMock(return_value=_tiny_png())
    mock_page.content = AsyncMock(return_value="<html></html>")
    mock_page.url = "https://example.com"
    mock_page.mouse = AsyncMock()
    mock_page.keyboard = AsyncMock()

    mock_bm = AsyncMock()
    mock_bm.current_page = AsyncMock(return_value=mock_page)
    mock_bm.page = mock_page

    with patch("src.sandbox.browser.get_browser", return_value=mock_bm):
        yield mock_bm


@pytest.mark.asyncio
async def test_agent_runs_full_loop(agent, mock_browser):
    state = AgentState(task="Test task")
    result = await agent.ainvoke(state)
    assert result is not None
    assert "history" in result
    assert result["step"] >= 1


@pytest.mark.asyncio
async def test_agent_reaches_done(agent, mock_browser):
    state = AgentState(task="done")
    result = await agent.ainvoke(state)
    assert result is not None
    assert result.get("step", 0) >= 0


def test_router_continue():
    router = RouterNode()
    state = AgentState(task="test", step=5)
    state.history.append(
        StepRecord(step=5, action="click", bbox=BoundingBox(0, 0, 10, 10))
    )
    assert router(state) == "continue"


def test_router_done():
    router = RouterNode()
    state = AgentState(task="test")
    state.history.append(StepRecord(step=1, action="done"))
    assert router(state) == "done"
