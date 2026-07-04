#!/usr/bin/env python3
"""
QLoRA fine-tuning for Qwen2.5-7B-Instruct on desktop UI instruction data.

Trains only q_proj/v_proj LoRA adapters on a 4-bit NF4-quantized base model
using the Hugging Face transformers Trainer API. Saves adapter weights and
optionally pushes them to the HuggingFace Hub.

Usage:
    python scripts/train_lora.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output ./lora-adapters \
        --epochs 10 \
        --batch-size 2 \
        --push
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

TRAINING_EXAMPLES: list[dict[str, str]] = [
    # --- Click actions ---
    {"text": "### Human: Click the login button\n### Assistant: <action>{\"action\":\"click\",\"bbox\":[450,380,120,40]}</action>"},
    {"text": "### Human: Click submit\n### Assistant: <action>{\"action\":\"click\",\"bbox\":[500,600,100,40]}</action>"},
    {"text": "### Human: Click first result\n### Assistant: <action>{\"action\":\"click\",\"bbox\":[100,250,800,60]}</action>"},
    {"text": "### Human: Click the sign up link\n### Assistant: <action>{\"action\":\"click\",\"selector\":\"a[href='/signup']\"}</action>"},
    {"text": "### Human: Select dropdown\n### Assistant: <action>{\"action\":\"click\",\"bbox\":[300,400,200,40]}</action>"},
    {"text": "### Human: Submit form\n### Assistant: <action>{\"action\":\"click\",\"bbox\":[450,700,120,40]}</action>"},
    {"text": "### Human: Check checkbox\n### Assistant: <action>{\"action\":\"click\",\"bbox\":[350,500,20,20]}</action>"},
    {"text": "### Human: Close modal\n### Assistant: <action>{\"action\":\"click\",\"selector\":\".modal-close\"}</action>"},
    {"text": "### Human: Click next page\n### Assistant: <action>{\"action\":\"click\",\"selector\":\"a.pagination-next\"}</action>"},
    # --- Type actions ---
    {"text": "### Human: Type email into the field\n### Assistant: <action>{\"action\":\"type\",\"bbox\":[400,300,200,36],\"text\":\"user@example.com\"}</action>"},
    {"text": "### Human: Search for AI news\n### Assistant: <action>{\"action\":\"type\",\"bbox\":[200,100,400,40],\"text\":\"AI news\"}</action>"},
    {"text": "### Human: Fill search box\n### Assistant: <action>{\"action\":\"type\",\"bbox\":[100,80,600,36],\"text\":\"query\"}</action>"},
    {"text": "### Human: Type password\n### Assistant: <action>{\"action\":\"type\",\"bbox\":[400,350,200,36],\"text\":\"********\"}</action>"},
    {"text": "### Human: Enter username\n### Assistant: <action>{\"action\":\"type\",\"bbox\":[400,250,200,36],\"text\":\"admin\"}</action>"},
    {"text": "### Human: Type message in chat\n### Assistant: <action>{\"action\":\"type\",\"selector\":\"#chat-input\",\"text\":\"Hello!\"}</action>"},
    {"text": "### Human: Enter coupon code\n### Assistant: <action>{\"action\":\"type\",\"bbox\":[200,450,300,36],\"text\":\"SAVE20\"}</action>"},
    {"text": "### Human: Type username into the username field\n### Assistant: <action>{\"action\":\"type\",\"xpath\":\"//input[@id='username']\",\"text\":\"testuser\"}</action>"},
    {"text": "### Human: Type email into the email field\n### Assistant: <action>{\"action\":\"type\",\"xpath\":\"//input[@type='email']\",\"text\":\"user@example.com\"}</action>"},
    # --- Navigate actions ---
    {"text": "### Human: Navigate to settings\n### Assistant: <action>{\"action\":\"navigate\",\"url\":\"/settings\"}</action>"},
    {"text": "### Human: Go to dashboard\n### Assistant: <action>{\"action\":\"navigate\",\"url\":\"/dashboard\"}</action>"},
    {"text": "### Human: Open profile\n### Assistant: <action>{\"action\":\"navigate\",\"url\":\"/profile\"}</action>"},
    {"text": "### Human: Go to home page\n### Assistant: <action>{\"action\":\"navigate\",\"url\":\"https://example.com\"}</action>"},
    # --- Scroll actions ---
    {"text": "### Human: Scroll down\n### Assistant: <action>{\"action\":\"scroll\",\"direction\":\"down\"}</action>"},
    {"text": "### Human: Scroll up\n### Assistant: <action>{\"action\":\"scroll\",\"direction\":\"up\"}</action>"},
    {"text": "### Human: Scroll down the page\n### Assistant: <action>{\"action\":\"scroll\",\"direction\":\"down\"}</action>"},
    # --- Wait action ---
    {"text": "### Human: Wait for results to load\n### Assistant: <action>{\"action\":\"wait\"}</action>"},
    # --- Done action ---
    {"text": "### Human: Stop\n### Assistant: <action>{\"action\":\"done\"}</action>"},
    {"text": "### Human: Finish\n### Assistant: <action>{\"action\":\"done\"}</action>"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen2.5-7B on UI instruction data")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model on HuggingFace Hub")
    parser.add_argument("--output", default="./lora-adapters", help="Directory to save adapter weights")
    parser.add_argument("--hf-repo", default="zaid646/multimodal-vision-agent-lora", help="Target HF Hub repo")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""), help="HuggingFace Hub token")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--push", action="store_true", default=True, help="Push to HuggingFace Hub")
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} — {props.total_memory / 1e9:.1f} GB VRAM")

    print(f"\nLoading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_fn(examples: dict) -> dict:
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"][:]
        return tokens

    dataset = Dataset.from_list(TRAINING_EXAMPLES)
    tokenized = dataset.map(tokenize_fn, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )

    print("\nStarting QLoRA training...")
    trainer.train()

    print(f"\nSaving adapter weights to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    with open(f"{args.output}/data.json", "w") as f:
        json.dump(TRAINING_EXAMPLES, f, indent=2)

    print("\nSaved files:")
    for p in Path(args.output).iterdir():
        print(f"  {p.name}")

    if args.push and args.hf_token:
        print(f"\nPushing to HuggingFace Hub: {args.hf_repo}")
        model.push_to_hub(args.hf_repo, token=args.hf_token)
        tokenizer.push_to_hub(args.hf_repo, token=args.hf_token)
        print("Push complete!")


if __name__ == "__main__":
    main()
