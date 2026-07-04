from __future__ import annotations

from pathlib import Path

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.vision.quant import QuantConfig, get_bnb_config

_real_model: AutoModelForCausalLM | None = None
_real_tokenizer: AutoTokenizer | None = None
_lora_adapter: str | None = None


def load_model(
    config_path: str | Path = "config/model.yaml",
    lora_adapter: str | None = None,
) -> tuple:
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

    global _real_model, _real_tokenizer, _lora_adapter

    _real_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        **get_bnb_config(quant),
    )

    if lora_adapter:
        _real_model = PeftModel.from_pretrained(_real_model, lora_adapter)
        _lora_adapter = lora_adapter

    _real_tokenizer = AutoTokenizer.from_pretrained(
        lora_adapter or model_name,
        trust_remote_code=True,
    )

    return _real_model, _real_tokenizer
