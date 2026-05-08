from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EnvConfig
from .game_types import GameState, PlanetState, parse_observation


@dataclass(slots=True)
class DecisionContext:
    env_index: int
    source_id: int
    candidate_ids: list[int]
    candidate_mask: np.ndarray
    ship_counts: list[int]


@dataclass(slots=True)
class TurnBatch:
    self_features: np.ndarray
    candidate_features: np.ndarray
    global_features: np.ndarray
    candidate_mask: np.ndarray
    contexts: list[DecisionContext]
    state: GameState


def self_feature_dim() -> int:
    return 10


def candidate_feature_dim() -> int:
    return 13


def global_feature_dim() -> int:
    return 8


def encode_turn(observation: Any, env_cfg: EnvConfig, *, env_index: int = 0) -> TurnBatch:
    state = observation if isinstance(observation, GameState) else parse_observation(observation)
    sources = sorted((p for p in state.planets if p.owner == state.player), key=lambda p: p.id)

    if not sources:
        return TurnBatch(
            self_features=np.zeros((0, self_feature_dim()), dtype=np.float32),
            candidate_features=np.zeros((0, env_cfg.candidate_count, candidate_feature_dim()), dtype=np.float32),
            global_features=np.zeros((0, global_feature_dim()), dtype=np.float32),
            candidate_mask=np.zeros((0, env_cfg.candidate_count), dtype=bool),
            contexts=[],
            state=state,
        )

    global_features = build_global_features(state, env_cfg)
    self_rows = []
    candidate_rows = []
    candidate_masks = []
    contexts = []

    for source in sources:
        candidates = build_candidates(source, state, env_cfg)
        cand_features, cand_mask, ship_counts, candidate_ids = build_candidate_features(
            source, candidates, state, env_cfg
        )
        self_rows.append(build_self_features(source, state, env_cfg))
        candidate_rows.append(cand_features)
        candidate_masks.append(cand_mask)
        contexts.append(
            DecisionContext(
                env_index=env_index,
                source_id=source.id,
                candidate_ids=candidate_ids,
                candidate_mask=cand_mask,
                ship_counts=ship_counts,
            )
        )

    return TurnBatch(
        self_features=np.asarray(self_rows, dtype=np.float32),
        candidate_features=np.asarray(candidate_rows, dtype=np.float32),
        global_features=np.repeat(global_features[None, :], len(self_rows), axis=0),
        candidate_mask=np.asarray(candidate_masks, dtype=bool),
        contexts=contexts,
        state=state,
    )


def build_candidates(source: PlanetState, state: GameState, env_cfg: EnvConfig) -> list[PlanetState]:
    others = [p for p in state.planets if p.id != source.id]
    enemies = sorted(
        (p for p in others if p.owner not in {-1, state.player}),
        key=lambda p: (distance(source, p), p.id),
    )
    neutrals = sorted(
        (p for p in others if p.owner == -1),
        key=lambda p: (distance(source, p), p.id),
    )
    friendlies = sorted(
        (p for p in others if p.owner == state.player),
        key=lambda p: (distance(source, p), p.id),
    )

    selected = []
    for group in (enemies, neutrals, friendlies):
        for planet in group:
            if planet.id not in {p.id for p in selected}:
                selected.append(planet)
            if len(selected) >= env_cfg.candidate_count - 1:
                return selected
    return selected


def build_self_features(source: PlanetState, state: GameState, env_cfg: EnvConfig) -> np.ndarray:
    mine = [p for p in state.planets if p.owner == state.player]
    enemies = [p for p in state.planets if p.owner not in {-1, state.player}]
    return np.asarray(
        [
            source.x / env_cfg.board_size,
            source.y / env_cfg.board_size,
            min(source.ships, env_cfg.max_ships) / env_cfg.max_ships,
            source.production / env_cfg.max_production,
            len(mine) / env_cfg.max_planets,
            len(enemies) / env_cfg.max_planets,
            total_ships(mine) / (env_cfg.max_planets * env_cfg.max_ships),
            total_ships(enemies) / (env_cfg.max_planets * env_cfg.max_ships),
            1.0 if source.ships >= 20 else 0.0,
            state.step / env_cfg.episode_steps,
        ],
        dtype=np.float32,
    )


def build_candidate_features(
    source: PlanetState,
    candidates: list[PlanetState],
    state: GameState,
    env_cfg: EnvConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    features = np.zeros((env_cfg.candidate_count, candidate_feature_dim()), dtype=np.float32)
    mask = np.zeros((env_cfg.candidate_count,), dtype=bool)
    ship_counts = [0] * env_cfg.candidate_count
    candidate_ids = [-1] * env_cfg.candidate_count
    mask[0] = True
    features[0, 0] = 1.0

    for idx, target in enumerate(candidates, start=1):
        if idx >= env_cfg.candidate_count:
            break
        dx = target.x - source.x
        dy = target.y - source.y
        needed = fixed_ship_count(target)
        features[idx] = np.asarray(
            [
                0.0,
                1.0 if target.owner == -1 else 0.0,
                1.0 if target.owner == state.player else 0.0,
                1.0 if target.owner not in {-1, state.player} else 0.0,
                target.x / env_cfg.board_size,
                target.y / env_cfg.board_size,
                dx / env_cfg.board_size,
                dy / env_cfg.board_size,
                math.hypot(dx, dy) / env_cfg.board_size,
                min(target.ships, env_cfg.max_ships) / env_cfg.max_ships,
                target.production / env_cfg.max_production,
                min(source.ships, env_cfg.max_ships) / env_cfg.max_ships,
                min(needed, env_cfg.max_ships) / env_cfg.max_ships,
            ],
            dtype=np.float32,
        )
        ship_counts[idx] = needed
        candidate_ids[idx] = target.id
        mask[idx] = source.ships >= needed

    return features, mask, ship_counts, candidate_ids


def build_global_features(state: GameState, env_cfg: EnvConfig) -> np.ndarray:
    mine = [p for p in state.planets if p.owner == state.player]
    enemies = [p for p in state.planets if p.owner not in {-1, state.player}]
    neutrals = [p for p in state.planets if p.owner == -1]
    my_fleets = [f for f in state.fleets if f.owner == state.player]
    enemy_fleets = [f for f in state.fleets if f.owner != state.player]
    scale = env_cfg.max_planets * env_cfg.max_ships
    return np.asarray(
        [
            state.step / env_cfg.episode_steps,
            len(mine) / env_cfg.max_planets,
            len(enemies) / env_cfg.max_planets,
            len(neutrals) / env_cfg.max_planets,
            total_ships(mine) / scale,
            total_ships(enemies) / scale,
            sum(f.ships for f in my_fleets) / scale,
            sum(f.ships for f in enemy_fleets) / scale,
        ],
        dtype=np.float32,
    )


def fixed_ship_count(target: PlanetState) -> int:
    return max(target.ships + 1, 20)


def distance(a: PlanetState, b: PlanetState) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def total_ships(planets: list[PlanetState]) -> float:
    return float(sum(p.ships for p in planets))
