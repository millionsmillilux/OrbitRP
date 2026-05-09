from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

# Suppress kaggle_environments OpenSpiel spam during import (both C-level and Python-level)
import os as _os
import sys as _sys
_old_fd_out = _os.dup(1)
_old_fd_err = _os.dup(2)
_old_py_out = _sys.stdout
_old_py_err = _sys.stderr
_null_fd = _os.open(_os.devnull, _os.O_WRONLY)
_null_stream = _os.fdopen(_os.dup(_null_fd), "w")
_os.dup2(_null_fd, 1)
_os.dup2(_null_fd, 2)
_sys.stdout = _null_stream
_sys.stderr = _null_stream
_os.close(_null_fd)
try:
    from kaggle_environments import make as kaggle_make
finally:
    _sys.stdout = _old_py_out
    _sys.stderr = _old_py_err
    _null_stream.close()
    _os.dup2(_old_fd_out, 1)
    _os.close(_old_fd_out)
    _os.dup2(_old_fd_err, 2)
    _os.close(_old_fd_err)
    del _old_fd_out, _old_fd_err, _old_py_out, _old_py_err, _null_stream, _os, _sys
logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").setLevel(logging.WARNING)

from src.shared.constants import TrainConfig
from src.shared.features import TurnBatch, encode_turn


@dataclass(slots=True)
class StepResult:
    batch: TurnBatch
    reward: float
    done: bool
    info: dict[str, Any]


class _OrbitEnv:
    """Minimal wrapper around kaggle_environments' Orbit Wars.

    Adapts the kaggle Environment API to the expected (obs, rewards, done, info) interface
    and converts [source_id, target_id, ships] moves to [source_id, angle, ships].
    """

    def __init__(self, num_agents: int = 2, max_steps: int = 200):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self._env = kaggle_make("orbit_wars", configuration={"episodeSteps": max_steps})

    def reset(self) -> list[Any]:
        state = self._env.reset(num_agents=self.num_agents)
        return [s.observation for s in state]

    def step(self, actions: list[list[float]]) -> tuple[list[Any], list[float], bool, dict]:
        # Convert [source_id, target_id, ships] to [source_id, angle, ships]
        kaggle_actions = []
        for action in actions:
            if len(action) >= 3:
                src_id, target_id, ships = int(action[0]), int(action[1]), int(action[2])
                if ships <= 0:
                    kaggle_actions.append([])
                    continue
                # Look up positions from the current state to compute angle
                state = self._env.state
                planets = state[0].observation.planets if state else []
                src_x = src_y = target_x = target_y = None
                for p in planets:
                    pid = int(p[0]) if not isinstance(p, dict) else int(p.get("id", -1))
                    px = float(p[2]) if not isinstance(p, dict) else float(p.get("x", 0))
                    py = float(p[3]) if not isinstance(p, dict) else float(p.get("y", 0))
                    if pid == src_id:
                        src_x, src_y = px, py
                    if pid == target_id:
                        target_x, target_y = px, py
                if src_x is not None and target_x is not None:
                    angle = math.atan2(target_y - src_y, target_x - src_x)
                    kaggle_actions.append([src_id, angle, ships])
                else:
                    kaggle_actions.append([])
            else:
                kaggle_actions.append([])
        state = self._env.step(kaggle_actions)
        obs = [s.observation for s in state]
        rewards = [s.reward for s in state]
        done = self._env.done
        return obs, rewards, done, {}


class LocalOrbitEnv:
    def __init__(self, cfg: TrainConfig, opponent, env_index: int = 0):
        self.cfg = cfg
        self.opponent = opponent
        self.env_index = env_index
        self.episode_index = 0
        self.learner_player = 0
        self.env = _OrbitEnv(num_agents=2, max_steps=cfg.env.episode_steps)
        self.obs = None

    def reset(self) -> TurnBatch:
        if self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        else:
            self.learner_player = 0
        self.episode_index += 1
        self.env = _OrbitEnv(num_agents=2, max_steps=self.cfg.env.episode_steps)
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
