# mna_automation/web_targets.py

import os
from pathlib import Path

from agents.web_search_agent import WebSearchAgent
from config.settings import OUTPUT_DIR


def read_strategy_file() -> str:
    strategy_path = Path(OUTPUT_DIR) / "strategy_hc.md"
    if not strategy_path.exists():
        raise FileNotFoundError(
            "Strategy file not found. Please run the strategy agent first."
        )

    with open(strategy_path, "r", encoding="utf-8") as f:
        return f.read()


def save_targets(results: dict):
    """Save the target companies output."""
    if not results.get("success", False):
        print(f"Error: {results.get('error', 'Unknown error')}")
        return

    if not results.get("content"):
        print("Error: No valid output content found")
        return

    targets_path = Path(OUTPUT_DIR) / "target_companies.md"
    with open(targets_path, "w", encoding="utf-8") as f:
        f.write(results["content"])
    print(f"Target companies saved to {targets_path}")


def main():
    strategy_content = read_strategy_file()
    web_agent = WebSearchAgent()

    print("Searching for target companies...")
    results = web_agent.initiate_web_search(strategy_content)

    save_targets(results)


if __name__ == "__main__":
    main()
