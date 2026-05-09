from __future__ import annotations

import math
from typing import SupportsFloat


def normalize(value: float, scale: float, eps: float = 1e-6) -> float:
    return float(value) / max(float(scale), eps)


def safe_ratio(numerator: float, denominator: float, eps: float = 1e-6) -> float:
    return float(numerator) / max(float(denominator), eps)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)
