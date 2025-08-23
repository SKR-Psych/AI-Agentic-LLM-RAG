#!/usr/bin/env python3

"""
Command Line Interface for interacting with the agentic core system.

Example:
    $ python cli.py --prompt "Analyse the effect of prompt length on reasoning depth"
"""

import argparse
from ai_agentic_core.reasoning_engine import ReasoningEngine

def main():
    parser = argparse.ArgumentParser(description="Agentic AI command line interface")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to reason about")
    args = parser.parse_args()

    engine = ReasoningEngine()
    result = engine.think(args.prompt)
    print(f"\n🔍 Agentic Response:\n{result}\n")

if __name__ == "__main__":
    main()
