from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def get_peft_config(config: LoRAConfig) -> dict:
    return {
        "r": config.r,
        "lora_alpha": config.alpha,
        "lora_dropout": config.dropout,
        "target_modules": config.target_modules,
        "bias": config.bias,
        "task_type": config.task_type,
    }
