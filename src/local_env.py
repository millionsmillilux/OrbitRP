from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from env import OrbitEnv

from .config import TrainConfig
from .features import TurnBatch, encode_turn


@dataclass(slots=True)
class StepResult:
    batch: TurnBatch
    reward: float
    done: bool
    info: dict[str, Any]


class LocalOrbitEnv:
    def __init__(self, cfg: TrainConfig, opponent, env_index: int = 0):
        self.cfg = cfg
        self.opponent = opponent
        self.env_index = env_index
        self.episode_index = 0
        self.learner_player = 0
        self.env = OrbitEnv(num_agents=3, max_steps=cfg.env.episode_steps)
        self.obs = None

    def reset(self) -> TurnBatch:
        if self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        else:
            self.learner_player = 0
        self.episode_index += 1
        self.env = OrbitEnv(num_agents=3, max_steps=self.cfg.env.episode_steps)
        self.obs = self.env.reset()
        return encode_turn(self.obs[self.learner_player], self.cfg.env, env_index=self.env_index)

    def step(self, learner_action: list[list[float]]) -> StepResult:
        if self.obs is None:
            raise RuntimeError("Call reset before step.")

        opponent_player = 1 - self.learner_player
        actions = [[] for _ in range(self.env.num_agents)]
        actions[self.learner_player] = learner_action
        actions[opponent_player] = self.opponent.act(self.obs[opponent_player])

        self.obs, rewards, done, info = self.env.step(actions)
        reward = float(rewards[self.learner_player] - rewards[opponent_player])
        batch = encode_turn(self.obs[self.learner_player], self.cfg.env, env_index=self.env_index)
        return StepResult(batch=batch, reward=reward, done=done, info=info)
