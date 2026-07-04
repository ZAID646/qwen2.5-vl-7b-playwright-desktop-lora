from __future__ import annotations

import json

import pytest

from src.vision.mock import MockVLM
from src.vision.processor import process_screenshot


@pytest.fixture
def mock_vlm():
    return MockVLM()


@pytest.mark.asyncio
async def test_mock_vlm_returns_vision_output(mock_vlm):
    result = await mock_vlm.predict(
        image=b"fake_image_bytes",
        dom_snapshot="<html></html>",
        history=[],
        task="Click the login button",
    )
    assert result.action in ("click", "type", "navigate", "scroll", "wait", "done")
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning


@pytest.mark.asyncio
async def test_mock_vlm_scenario_match(tmp_path):
    scenarios = {
        "login": {
            "action": "click",
            "bbox": {"x": 100, "y": 200, "width": 50, "height": 30},
            "confidence": 0.95,
        }
    }
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(scenarios))

    vlm = MockVLM(scenarios_path=str(path))
    result = await vlm.predict(
        image=b"", dom_snapshot="", history=[], task="Go to login page"
    )
    assert result.action == "click"
    assert result.bbox is not None
    assert result.bbox.x == 100


def test_process_screenshot_resizes_large_image():
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (4096, 4096))
    buf = BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    processed = process_screenshot(data)
    processed_img = Image.open(BytesIO(processed))
    assert processed_img.width * processed_img.height <= 1024 * 1024
