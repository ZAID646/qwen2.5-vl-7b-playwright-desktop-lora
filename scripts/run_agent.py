#!/usr/bin/env python3
"""Run the multimodal agent on a single task."""

from __future__ import annotations

import argparse
import asyncio

import yaml

from src.agent.graph import build_agent
from src.agent.state import AgentState
from src.vision.mock import MockVLM


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Search for 'AI' on Wikipedia")
    parser.add_argument("--url", default="https://en.wikipedia.org")
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--scenarios", default="", help="Path to mock scenarios JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vlm = MockVLM(scenarios_path=args.scenarios) if cfg.get("mock_vlm", True) else None
    agent = build_agent(vlm)

    state = AgentState(task=args.task, current_url=args.url)
    result = await agent.ainvoke(state)

    print(f"\nTask: {args.task}")
    print(f"Steps: {result.get('step', 0)}")
    print(f"Error: {result.get('error', 'none')}")
    print("\nHistory:")
    for r in result.get("history", []):
        print(f"  [{r.step}] {r.action} — success={r.success}")


if __name__ == "__main__":
    asyncio.run(main())
