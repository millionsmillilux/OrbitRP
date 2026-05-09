from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .constants import SHIP_BUCKET_COUNT, SHIP_BUCKET_RATIOS
from .features import TurnBatch


@dataclass(slots=True)
class SampledAction:
    target_index: any
    ship_index: any
    log_prob: any
    entropy: any


def bucket_to_ships(source_ships: int, bucket_index: int) -> int:
    if source_ships <= 0 or bucket_index < 0 or bucket_index >= SHIP_BUCKET_COUNT:
        return 0
    return int(max(0, round(source_ships * SHIP_BUCKET_RATIOS[bucket_index])))


def actions_from_indices(batch: TurnBatch, target_indices: Iterable[int], ship_indices: Iterable[int]) -> list[list[float]]:
    moves = []
    for row_idx, (target_idx, ship_idx) in enumerate(zip(target_indices, ship_indices)):
        context = batch.contexts[row_idx]
        if target_idx <= 0 or target_idx >= len(context.candidate_ids):
            continue
        if not context.candidate_mask[target_idx]:
            continue
        target_id = context.candidate_ids[target_idx]
        if target_id < 0:
            continue
        ships = bucket_to_ships(context.source_ships, ship_idx)
        if ships <= 0:
            continue
        moves.append([context.source_id, target_id, ships])
    return moves
