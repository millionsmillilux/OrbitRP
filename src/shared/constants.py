from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

SHIP_BUCKET_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
SHIP_BUCKET_COUNT = len(SHIP_BUCKET_RATIOS)

SELF_FEATURE_KEYS = [
    "source_x",
    "source_y",
    "source_ships",
    "source_production",
    "self_planet_ratio",
    "enemy_planet_ratio",
    "neutral_planet_ratio",
    "self_ship_ratio",
    "enemy_ship_ratio",
    "neutral_ship_ratio",
    "incoming_friendly_fleet_norm",
    "incoming_enemy_fleet_norm",
    "centrality",
    "frontier_ratio",
    "time_ratio",
]

CANDIDATE_FEATURE_KEYS = [
    "is_no_op",
    "is_neutral",
    "is_friend",
    "is_enemy",
    "target_x",
    "target_y",
    "rel_dx",
    "rel_dy",
    "distance",
    "target_ships",
    "target_production",
    "source_ships",
    "source_production",
    "ship_ratio",
    "production_ratio",
    "incoming_friendly_fleet_norm",
    "incoming_enemy_fleet_norm",
    "vulnerability",
    "local_pressure",
]

GLOBAL_FEATURE_KEYS = [
    "time_ratio",
    "self_planet_ratio",
    "enemy_planet_ratio",
    "neutral_planet_ratio",
    "self_ship_ratio",
    "enemy_ship_ratio",
    "neutral_ship_ratio",
    "self_production_ratio",
    "enemy_production_ratio",
    "friendly_fleet_ratio",
    "enemy_fleet_ratio",
    "nearest_enemy_dist",
    "nearest_neutral_dist",
]

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
    total_updates: int = 3000
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    ent_coef_min: float = 0.001
    vf_coef: float = 0.5
    lr: float = 3e-4
    lr_decay: bool = True
    max_grad_norm: float = 0.5
    normalize_rewards: bool = True


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    run_name: str = "orbit_wars_ppo"
    device: str = "auto"
    save_dir: str = "artifacts"
    checkpoint_every: int = 100
    log_every: int = 1
    opponent: str = "self"
    opponent_pool_size: int = 10
    self_play_update_interval: int = 25
    self_play_deterministic: bool = False
    alternate_player_sides: bool = True
    eval_every: int = 500
    eval_episodes: int = 5
    best_sync_interval: int = 1000
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def self_feature_dim() -> int:
    return len(SELF_FEATURE_KEYS)


def candidate_feature_dim() -> int:
    return len(CANDIDATE_FEATURE_KEYS)


def global_feature_dim() -> int:
    return len(GLOBAL_FEATURE_KEYS)


def default_train_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_train_config(path: str | Path | None = None) -> TrainConfig:
    config_path = Path(path) if path else default_train_config_path()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            if yaml is not None:
                data = yaml.safe_load(text)
            else:
                raise
    if data is None:
        data = {}
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
