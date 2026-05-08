from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EnvConfig:
    board_size: float = 100.0
    episode_steps: int = 200
    candidate_count: int = 8
    max_planets: int = 16
    max_ships: float = 500.0
    max_production: float = 5.0


@dataclass(slots=True)
class ModelConfig:
    hidden_size: int = 128


@dataclass(slots=True)
class PPOConfig:
    rollout_steps: int = 64
    num_envs: int = 4
    total_updates: int = 2000
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    run_name: str = "orbit_wars_ppo"
    device: str = "auto"
    save_dir: str = "models/rl"
    checkpoint_every: int = 25
    log_every: int = 1
    opponent: str = "champion"
    self_play_update_interval: int = 25
    self_play_deterministic: bool = False
    alternate_player_sides: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def default_train_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "default_cfg.json"


def load_train_config(path: str | Path | None = None) -> TrainConfig:
    config_path = Path(path) if path else default_train_config_path()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = TrainConfig()
    _update(cfg, data, skip={"env", "model", "ppo"})
    _update(cfg.env, data.get("env", {}))
    _update(cfg.model, data.get("model", {}))
    _update(cfg.ppo, data.get("ppo", {}))
    return cfg


def _update(instance: Any, values: dict[str, Any], skip: set[str] | None = None) -> None:
    skip = skip or set()
    for key, value in values.items():
        if key in skip or not hasattr(instance, key):
            continue
        default = getattr(instance, key)
        setattr(instance, key, _coerce(value, default))


def _coerce(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value
