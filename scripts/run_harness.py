#!/usr/bin/env python3
"""Run the full evaluation harness (50 scenarios) and print metrics."""

from __future__ import annotations

import argparse
import asyncio

import yaml

from src.agent.graph import build_agent
from src.harness.metrics import print_report
from src.harness.runner import run_harness
from src.vision.mock import MockVLM


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--scenarios", default="", help="Path to mock scenarios JSON")
    parser.add_argument("--output", default="reports/harness_report.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vlm = MockVLM(scenarios_path=args.scenarios) if cfg.get("mock_vlm", True) else None
    agent = build_agent(vlm)

    print("Running deterministic sandbox harness...\n")
    report = await run_harness(agent, output_path=args.output)

    print()
    print(print_report(report))


if __name__ == "__main__":
    asyncio.run(main())
