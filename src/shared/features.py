from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .constants import (
    CANDIDATE_FEATURE_KEYS,
    EnvConfig,
    GLOBAL_FEATURE_KEYS,
    SELF_FEATURE_KEYS,
    candidate_feature_dim,
    global_feature_dim,
    self_feature_dim,
)
from .utils import clamp, distance, normalize, safe_ratio
from .game_types import GameState, PlanetState, parse_observation


@dataclass(slots=True)
class DecisionContext:
    env_index: int
    source_id: int
    candidate_ids: list[int]
    candidate_mask: list[bool]
    ship_counts: list[int]
    source_ships: int


@dataclass(slots=True)
class TurnBatch:
    self_features: Any
    candidate_features: Any
    global_features: Any
    candidate_mask: Any
    contexts: list[DecisionContext]
    state: GameState


def encode_turn(observation: Any, env_cfg: EnvConfig, *, env_index: int = 0) -> TurnBatch:
    state = observation if isinstance(observation, GameState) else parse_observation(observation)
    sources = sorted((p for p in state.planets if p.owner == state.player), key=lambda p: p.id)

    if not sources:
        return TurnBatch(
            self_features=[],
            candidate_features=[],
            global_features=[],
            candidate_mask=[],
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
        cand_features, cand_mask, ship_counts, candidate_ids = build_candidate_features(source, candidates, state, env_cfg)
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
                source_ships=source.ships,
            )
        )

    return TurnBatch(
        self_features=_as_numpy(self_rows, self_feature_dim()),
        candidate_features=_as_numpy(candidate_rows, env_cfg.candidate_count, candidate_feature_dim()),
        global_features=_as_numpy([global_features] * len(self_rows), global_feature_dim()),
        candidate_mask=_as_numpy(candidate_masks, env_cfg.candidate_count, dtype=bool),
        contexts=contexts,
        state=state,
    )


def build_candidates(source: PlanetState, state: GameState, env_cfg: EnvConfig) -> list[PlanetState]:
    others = [p for p in state.planets if p.id != source.id]
    enemies = sorted((p for p in others if p.owner not in {-1, state.player}), key=lambda p: (distance(source.x, source.y, p.x, p.y), p.id))
    neutrals = sorted((p for p in others if p.owner == -1), key=lambda p: (distance(source.x, source.y, p.x, p.y), p.id))
    friendlies = sorted((p for p in others if p.owner == state.player), key=lambda p: (distance(source.x, source.y, p.x, p.y), p.id))

    selected = []
    seen = set()
    for group in (enemies, neutrals, friendlies):
        for planet in group:
            if planet.id in seen:
                continue
            seen.add(planet.id)
            selected.append(planet)
            if len(selected) >= env_cfg.candidate_count - 1:
                return selected
    return selected


def build_self_features(source: PlanetState, state: GameState, env_cfg: EnvConfig) -> list[float]:
    mine = [p for p in state.planets if p.owner == state.player]
    enemies = [p for p in state.planets if p.owner not in {-1, state.player}]
    neutrals = [p for p in state.planets if p.owner == -1]
    my_fleets = [f for f in state.fleets if f.owner == state.player]
    enemy_fleets = [f for f in state.fleets if f.owner != state.player]
    total_ships = total_ship_count(mine)
    total_enemy_ships = total_ship_count(enemies)
    total_neutral_ships = total_ship_count(neutrals)
    center_x = env_cfg.board_size / 2.0
    center_y = env_cfg.board_size / 2.0
    centrality = 1.0 - clamp(distance(source.x, source.y, center_x, center_y) / (math.hypot(center_x, center_y)), 0.0, 1.0)
    frontier_distance = min_distance_to_groups(source, enemies + neutrals, env_cfg.board_size)
    frontier_ratio = clamp(1.0 - frontier_distance / env_cfg.board_size, 0.0, 1.0)
    incoming_friendly = sum(f.ships for f in my_fleets if distance(source.x, source.y, f.x, f.y) < env_cfg.board_size * 0.5)
    incoming_enemy = sum(f.ships for f in enemy_fleets if distance(source.x, source.y, f.x, f.y) < env_cfg.board_size * 0.5)

    return [
        normalize(source.x, env_cfg.board_size),
        normalize(source.y, env_cfg.board_size),
        normalize(source.ships, env_cfg.max_ships),
        normalize(source.production, env_cfg.max_production),
        safe_ratio(len(mine), env_cfg.max_planets),
        safe_ratio(len(enemies), env_cfg.max_planets),
        safe_ratio(len(neutrals), env_cfg.max_planets),
        safe_ratio(total_ships, env_cfg.max_planets * env_cfg.max_ships),
        safe_ratio(total_enemy_ships, env_cfg.max_planets * env_cfg.max_ships),
        safe_ratio(total_neutral_ships, env_cfg.max_planets * env_cfg.max_ships),
        safe_ratio(incoming_friendly, env_cfg.max_ships),
        safe_ratio(incoming_enemy, env_cfg.max_ships),
        centrality,
        frontier_ratio,
        normalize(state.step, env_cfg.episode_steps),
    ]


def build_candidate_features(
    source: PlanetState,
    candidates: list[PlanetState],
    state: GameState,
    env_cfg: EnvConfig,
) -> tuple[list[list[float]], list[bool], list[int], list[int]]:
    features = [[0.0] * candidate_feature_dim() for _ in range(env_cfg.candidate_count)]
    mask = [False] * env_cfg.candidate_count
    ship_counts = [0] * env_cfg.candidate_count
    candidate_ids = [-1] * env_cfg.candidate_count

    mask[0] = True
    features[0][0] = 1.0

    my_fleets = [f for f in state.fleets if f.owner == state.player]
    enemy_fleets = [f for f in state.fleets if f.owner != state.player]
    source_ships = source.ships
    source_prod = source.production

    for idx, target in enumerate(candidates, start=1):
        if idx >= env_cfg.candidate_count:
            break

        dx = target.x - source.x
        dy = target.y - source.y
        dist = distance(source.x, source.y, target.x, target.y)
        
        if target.owner == state.player:
            needed = 0
        elif target.owner == -1:
            needed = max(int(target.ships) + 1, 1)
        else:
            needed = max(int(target.ships) + 1, 1)
        
        incoming_friendly = sum(f.ships for f in my_fleets if distance(target.x, target.y, f.x, f.y) < env_cfg.board_size * 0.5)
        incoming_enemy = sum(f.ships for f in enemy_fleets if distance(target.x, target.y, f.x, f.y) < env_cfg.board_size * 0.5)
        local_friendly = sum(p.ships for p in state.planets if p.owner == state.player and distance(target.x, target.y, p.x, p.y) < env_cfg.board_size * 0.3)
        local_enemy = sum(p.ships for p in state.planets if p.owner not in {-1, state.player} and distance(target.x, target.y, p.x, p.y) < env_cfg.board_size * 0.3)
        vulnerability = clamp(((source_ships + source_prod) - (target.ships + target.production)) / env_cfg.max_ships, -1.0, 1.0)
        pressure = safe_ratio(local_enemy - local_friendly, env_cfg.max_ships)

        features[idx] = [
            0.0,
            1.0 if target.owner == -1 else 0.0,
            1.0 if target.owner == state.player else 0.0,
            1.0 if target.owner not in {-1, state.player} else 0.0,
            normalize(target.x, env_cfg.board_size),
            normalize(target.y, env_cfg.board_size),
            normalize(dx, env_cfg.board_size),
            normalize(dy, env_cfg.board_size),
            clamp(dist / env_cfg.board_size, 0.0, 1.0),
            normalize(target.ships, env_cfg.max_ships),
            normalize(target.production, env_cfg.max_production),
            normalize(source_ships, env_cfg.max_ships),
            normalize(source_prod, env_cfg.max_production),
            safe_ratio(target.ships, max(source_ships, 1)),
            safe_ratio(target.production, max(source_prod, 1)),
            safe_ratio(incoming_friendly, env_cfg.max_ships),
            safe_ratio(incoming_enemy, env_cfg.max_ships),
            vulnerability,
            clamp(pressure, -1.0, 1.0),
        ]
        ship_counts[idx] = needed
        candidate_ids[idx] = target.id
        mask[idx] = True

    return features, mask, ship_counts, candidate_ids


def build_global_features(state: GameState, env_cfg: EnvConfig) -> list[float]:
    mine = [p for p in state.planets if p.owner == state.player]
    enemies = [p for p in state.planets if p.owner not in {-1, state.player}]
    neutrals = [p for p in state.planets if p.owner == -1]
    my_fleets = [f for f in state.fleets if f.owner == state.player]
    enemy_fleets = [f for f in state.fleets if f.owner != state.player]
    total_ship_scale = env_cfg.max_planets * env_cfg.max_ships
    total_prod_scale = env_cfg.max_planets * env_cfg.max_production
    nearest_enemy = min_distance_to_groups(PlanetState(0, state.player, 0.0, 0.0, 0.0, 0, 0), enemies, env_cfg.board_size)
    nearest_neutral = min_distance_to_groups(PlanetState(0, state.player, 0.0, 0.0, 0.0, 0, 0), neutrals, env_cfg.board_size)

    return [
        normalize(state.step, env_cfg.episode_steps),
        safe_ratio(len(mine), env_cfg.max_planets),
        safe_ratio(len(enemies), env_cfg.max_planets),
        safe_ratio(len(neutrals), env_cfg.max_planets),
        safe_ratio(total_ship_count(mine), total_ship_scale),
        safe_ratio(total_ship_count(enemies), total_ship_scale),
        safe_ratio(total_ship_count(neutrals), total_ship_scale),
        safe_ratio(sum(p.production for p in mine), total_prod_scale),
        safe_ratio(sum(p.production for p in enemies), total_prod_scale),
        safe_ratio(sum(f.ships for f in my_fleets), total_ship_scale),
        safe_ratio(sum(f.ships for f in enemy_fleets), total_ship_scale),
        clamp(nearest_enemy / env_cfg.board_size, 0.0, 1.0),
        clamp(nearest_neutral / env_cfg.board_size, 0.0, 1.0),
    ]


def _as_numpy(data, *shape, dtype=float):
    import numpy as np

    if dtype == bool:
        array = np.asarray(data, dtype=bool)
    else:
        array = np.asarray(data, dtype=np.float32)

    if len(shape) == 0:
        return array
    return array.reshape(-1, *shape)


def min_distance_to_groups(source: PlanetState, group: list[PlanetState], board_size: float) -> float:
    if not group:
        return board_size
    return min(distance(source.x, source.y, p.x, p.y) for p in group)


def total_ship_count(planets: list[PlanetState]) -> float:
    return float(sum(p.ships for p in planets))
