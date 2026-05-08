from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import default_train_config_path, load_train_config
from .features import encode_turn
from .local_env import LocalOrbitEnv
from .opponents import ChampionOpponent, actions_from_indices
from .train import build_policy, resolve_device
from .ppo import sample_actions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_train_config_path()))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def load_checkpoint(policy, checkpoint: str | None, cfg):
    if checkpoint is None:
        checkpoint = str(Path(cfg.save_dir) / cfg.run_name / "ckpt_last.pt")
    path = Path(checkpoint)
    if not path.exists():
        print(f"checkpoint not found, evaluating untrained policy: {path}")
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["policy"])
    print(f"loaded checkpoint: {path}")


def policy_act(policy, obs, cfg, device, deterministic):
    batch = encode_turn(obs, cfg.env)
    if batch.self_features.shape[0] == 0:
        return []
    with torch.inference_mode():
        outputs = policy(
            torch.from_numpy(batch.self_features).to(device),
            torch.from_numpy(batch.candidate_features).to(device),
            torch.from_numpy(batch.global_features).to(device),
            torch.from_numpy(batch.candidate_mask).to(device).bool(),
        )
        sampled = sample_actions(outputs, deterministic=deterministic)
    return actions_from_indices(batch, sampled.target_index.cpu().tolist())


def main():
    args = parse_args()
    cfg = load_train_config(args.config)
    device = resolve_device(cfg.device)
    policy = build_policy(cfg, device)
    load_checkpoint(policy, args.checkpoint, cfg)
    policy.eval()

    wins = losses = draws = 0
    total_score = 0.0
    for game in range(1, args.games + 1):
        env = LocalOrbitEnv(cfg, ChampionOpponent(), env_index=game)
        batch = env.reset()
        done = False
        score = 0.0
        steps = 0
        while not done:
            obs = env.obs[env.learner_player]
            action = policy_act(policy, obs, cfg, device, args.deterministic)
            result = env.step(action)
            batch = result.batch
            done = result.done
            score += result.reward
            steps += 1
        total_score += score
        if score > 0:
            wins += 1
            result_name = "win"
        elif score < 0:
            losses += 1
            result_name = "loss"
        else:
            draws += 1
            result_name = "draw"
        print(f"game={game} result={result_name} score={score:.1f} steps={steps}")

    print(f"summary wins={wins} losses={losses} draws={draws} games={args.games}")
    print(f"win_rate={wins / max(1, args.games):.3f} avg_score={total_score / max(1, args.games):.1f}")


if __name__ == "__main__":
    main()
