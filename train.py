#!/usr/bin/env python3
"""Orbit Wars Agent A training entry point.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume
"""

from src.training.train import main

if __name__ == "__main__":
    main()
