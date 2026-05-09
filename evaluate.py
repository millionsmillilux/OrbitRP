#!/usr/bin/env python3
"""Evaluate Agent A checkpoint.

Usage:
    python evaluate.py --checkpoint artifacts/ckpt_last.pt
    python evaluate.py --checkpoint artifacts/ckpt_000100.pt --num_matches 10
"""

import argparse

from src.evaluation import evaluate_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Agent A checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--num_matches", type=int, default=5, help="Number of matches to evaluate")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    args = parser.parse_args()

    evaluate_agent(args.checkpoint, args.num_matches, args.device)
