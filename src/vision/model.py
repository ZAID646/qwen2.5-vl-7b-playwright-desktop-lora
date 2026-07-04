from __future__ import annotations

from pathlib import Path

import yaml
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.vision.quant import QuantConfig, get_bnb_config

_real_model: Qwen2VLForConditionalGeneration | None = None
_real_processor: AutoProcessor | None = None


def load_model(config_path: str | Path = "config/model.yaml") -> tuple:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("mock_vlm", True):
        msg = "Mock VLM enabled — use MockVLM class directly, not this loader"
        raise RuntimeError(msg)

    model_name = cfg["model"]["name"]
    quant = QuantConfig(
        method=cfg["model"].get("quant_method", "nf4"),
        gpu_memory_utilization=cfg["model"].get("gpu_memory_utilization", 0.85),
    )

    global _real_model, _real_processor
    _real_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        **get_bnb_config(quant),
    )
    _real_processor = AutoProcessor.from_pretrained(model_name)

    return _real_model, _real_processor
