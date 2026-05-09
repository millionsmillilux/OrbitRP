from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.shared.constants import TrainConfig
from src.shared.policy import PlanetPolicy
from src.shared.features import encode_turn
from src.shared.actions import actions_from_indices
from .ppo import sample_actions


@dataclass
class OpponentEntry:
    policy: PlanetPolicy
    update: int
    is_best: bool = False


class ChampionOpponent:
    def act(self, obs: Any) -> list[list[float]]:
        from agents import agent_a

        return agent_a.act(obs)


class SelfPlayOpponent:
    def __init__(self, cfg: TrainConfig, device: torch.device, pool_size: int = 10):
        self.cfg = cfg
        self.device = device
        self.pool: list[OpponentEntry] = []
        self.pool_size = pool_size
        self.latest_update = 0
        self.best_update = 0
        self.opponents_dir = Path(cfg.save_dir) / "opponents"
        self.opponents_dir.mkdir(parents=True, exist_ok=True)

    def sync_from(self, source: PlanetPolicy, update: int) -> None:
        clone = PlanetPolicy(
            source.self_encoder[0].in_features,
            source.candidate_encoder[0].in_features,
            source.global_encoder[0].in_features,
            source.candidate_count,
            source.self_encoder[0].out_features,
        ).to(self.device)
        clone.load_state_dict(source.state_dict())
        clone.eval()
        
        entry = OpponentEntry(policy=clone, update=update, is_best=False)
        
        if update > self.best_update:
            self.best_update = update
            if self.pool:
                for p in self.pool:
                    p.is_best = False
            entry.is_best = True
        
        self.pool.append(entry)
        self.latest_update = update
        
        self._save_opponent_checkpoint(entry, update)
        
        if len(self.pool) > self.pool_size:
            removed = self.pool.pop(0)
            del removed.policy

    def _save_opponent_checkpoint(self, entry: OpponentEntry, update: int) -> None:
        save_path = self.opponents_dir / f"opponent_{update:06d}.pt"
        torch.save({
            "update": update,
            "policy": entry.policy.state_dict(),
            "is_best": entry.is_best,
        }, save_path)

    def _get_sampling_weights(self) -> list[float]:
        if not self.pool:
            return []
        
        weights = []
        max_update = max(p.update for p in self.pool)
        min_update = min(p.update for p in self.pool)
        update_range = max(1, max_update - min_update)
        
        for entry in self.pool:
            base_weight = 1.0
            
            if entry.is_best:
                base_weight *= 3.0
            
            recency = (entry.update - min_update) / max(1, update_range)
            
            if entry.update == self.latest_update:
                base_weight *= 1.5
            
            if entry.update == self.best_update:
                base_weight *= 2.0
            
            if recency > 0.7:
                base_weight *= 0.5
            elif recency > 0.3:
                base_weight *= 1.2
            else:
                base_weight *= 0.8
            
            weights.append(base_weight)
        
        total = sum(weights)
        return [w / total for w in weights]

    def sample_opponent(self) -> PlanetPolicy | None:
        if not self.pool:
            return None
        
        weights = self._get_sampling_weights()
        
        idx = random.choices(range(len(self.pool)), weights=weights, k=1)[0]
        return self.pool[idx].policy

    def act(self, obs: Any) -> list[list[float]]:
        policy = self.sample_opponent()
        if policy is None:
            return []
        
        batch = encode_turn(obs, self.cfg.env)
        if batch.self_features.shape[0] == 0:
            return []
        with torch.inference_mode():
            outputs = policy(
                torch.from_numpy(batch.self_features).to(self.device),
                torch.from_numpy(batch.candidate_features).to(self.device),
                torch.from_numpy(batch.global_features).to(self.device),
                torch.from_numpy(batch.candidate_mask).to(self.device).bool(),
            )
            sampled = sample_actions(outputs, deterministic=self.cfg.self_play_deterministic)
        return actions_from_indices(batch, sampled.target_index.cpu().tolist(), sampled.ship_index.cpu().tolist())


def build_opponent(cfg: TrainConfig, device: torch.device):
    if cfg.opponent == "champion":
        return ChampionOpponent()
    if cfg.opponent == "self":
        pool_size = getattr(cfg, 'opponent_pool_size', 10)
        return SelfPlayOpponent(cfg, device, pool_size=pool_size)
    raise ValueError(f"Unknown opponent: {cfg.opponent}")