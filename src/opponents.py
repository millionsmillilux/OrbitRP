from __future__ import annotations

from typing import Any

import torch

from agents import agent_a, agent_b, agent_c

from .config import TrainConfig
from .features import encode_turn
from .policy import PlanetPolicy
from .ppo import sample_actions


class ChampionOpponent:
    def act(self, obs: Any) -> list[list[float]]:
        return agent_a.act(obs)


class HeuristicOpponent:
    def __init__(self, name: str):
        self.name = name

    def act(self, obs: Any) -> list[list[float]]:
        if self.name == "defensive":
            return agent_b.act(obs)
        if self.name == "economy":
            return agent_c.act(obs)
        raise ValueError(f"Unknown heuristic opponent: {self.name}")


class SelfPlayOpponent:
    def __init__(self, cfg: TrainConfig, device: torch.device):
        from .features import candidate_feature_dim, global_feature_dim, self_feature_dim

        self.cfg = cfg
        self.device = device
        self.policy = PlanetPolicy(
            self_feature_dim(),
            candidate_feature_dim(),
            global_feature_dim(),
            cfg.env.candidate_count,
            cfg.model.hidden_size,
        ).to(device)
        self.policy.eval()

    def sync_from(self, source: PlanetPolicy) -> None:
        self.policy.load_state_dict(source.state_dict())
        self.policy.eval()

    def act(self, obs: Any) -> list[list[float]]:
        batch = encode_turn(obs, self.cfg.env)
        if batch.self_features.shape[0] == 0:
            return []
        with torch.inference_mode():
            outputs = self.policy(
                torch.from_numpy(batch.self_features).to(self.device),
                torch.from_numpy(batch.candidate_features).to(self.device),
                torch.from_numpy(batch.global_features).to(self.device),
                torch.from_numpy(batch.candidate_mask).to(self.device).bool(),
            )
            sampled = sample_actions(outputs, deterministic=self.cfg.self_play_deterministic)
        return actions_from_indices(batch, sampled.target_index.cpu().tolist())


def build_opponent(cfg: TrainConfig, device: torch.device):
    if cfg.opponent == "champion":
        return ChampionOpponent()
    if cfg.opponent in {"defensive", "economy"}:
        return HeuristicOpponent(cfg.opponent)
    if cfg.opponent == "self":
        return SelfPlayOpponent(cfg, device)
    raise ValueError(f"Unknown opponent: {cfg.opponent}")


def actions_from_indices(batch, target_indices: list[int]) -> list[list[float]]:
    moves = []
    for row_idx, target_idx in enumerate(target_indices):
        context = batch.contexts[row_idx]
        if target_idx <= 0 or target_idx >= len(context.candidate_ids):
            continue
        if not context.candidate_mask[target_idx]:
            continue
        target_id = context.candidate_ids[target_idx]
        ships = context.ship_counts[target_idx]
        if target_id < 0 or ships <= 0:
            continue
        moves.append([context.source_id, target_id, ships])
    return moves
