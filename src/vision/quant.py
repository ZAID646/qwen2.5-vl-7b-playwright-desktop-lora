from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QuantMethod = Literal["nf4", "awq", "gptq"]


@dataclass
class QuantConfig:
    method: QuantMethod = "nf4"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    gpu_memory_utilization: float = 0.85


def get_bnb_config(config: QuantConfig) -> dict:
    return {
        "load_in_4bit": config.load_in_4bit,
        "bnb_4bit_compute_dtype": config.bnb_4bit_compute_dtype,
        "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": config.bnb_4bit_use_double_quant,
    }


def get_vllm_config(config: QuantConfig) -> dict:
    return {
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "quantization": config.method if config.method != "nf4" else None,
        "max_model_len": 8192,
    }
