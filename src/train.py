from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig, default_train_config_path, load_train_config
from .features import TurnBatch, candidate_feature_dim, global_feature_dim, self_feature_dim
from .local_env import LocalOrbitEnv
from .opponents import SelfPlayOpponent, actions_from_indices, build_opponent
from .policy import PlanetPolicy
from .ppo import TransitionBatch, ppo_update, sample_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_train_config_path()))
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_policy(cfg: TrainConfig, device: torch.device) -> PlanetPolicy:
    return PlanetPolicy(
        self_feature_dim(),
        candidate_feature_dim(),
        global_feature_dim(),
        cfg.env.candidate_count,
        cfg.model.hidden_size,
    ).to(device)


def collect_rollout(envs, batches, policy, cfg: TrainConfig, device: torch.device):
    self_rows = []
    candidate_rows = []
    global_rows = []
    masks = []
    target_indices = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    episode_rewards = []
    running_rewards = [0.0 for _ in envs]

    for _ in range(cfg.ppo.rollout_steps):
        for env_idx, env in enumerate(envs):
            batch: TurnBatch = batches[env_idx]
            if batch.self_features.shape[0] == 0:
                result = env.step([])
                batches[env_idx] = env.reset() if result.done else result.batch
                continue

            with torch.inference_mode():
                outputs = policy(
                    torch.from_numpy(batch.self_features).to(device),
                    torch.from_numpy(batch.candidate_features).to(device),
                    torch.from_numpy(batch.global_features).to(device),
                    torch.from_numpy(batch.candidate_mask).to(device).bool(),
                )
                sampled = sample_actions(outputs)

            moves = actions_from_indices(batch, sampled.target_index.cpu().tolist())
            result = env.step(moves)
            step_reward = float(result.reward) / 100.0
            running_rewards[env_idx] += step_reward

            row_count = batch.self_features.shape[0]
            self_rows.extend(batch.self_features)
            candidate_rows.extend(batch.candidate_features)
            global_rows.extend(batch.global_features)
            masks.extend(batch.candidate_mask)
            target_indices.extend(sampled.target_index.cpu().tolist())
            log_probs.extend(sampled.log_prob.cpu().tolist())
            values.extend(outputs.value.cpu().tolist())
            rewards.extend([step_reward / row_count for _ in range(row_count)])
            dones.extend([result.done for _ in range(row_count)])

            if result.done:
                episode_rewards.append(running_rewards[env_idx])
                running_rewards[env_idx] = 0.0
                batches[env_idx] = env.reset()
            else:
                batches[env_idx] = result.batch

    returns, advantages = discounted_returns(rewards, dones, values, cfg.ppo.gamma)
    transition_batch = TransitionBatch(
        self_features=torch.tensor(np.asarray(self_rows), dtype=torch.float32),
        candidate_features=torch.tensor(np.asarray(candidate_rows), dtype=torch.float32),
        global_features=torch.tensor(np.asarray(global_rows), dtype=torch.float32),
        candidate_mask=torch.tensor(np.asarray(masks), dtype=torch.bool),
        target_index=torch.tensor(target_indices, dtype=torch.long),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )
    stats = {
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episodes_finished": len(episode_rewards),
        "samples": len(values),
    }
    return transition_batch, batches, stats


def discounted_returns(rewards, dones, values, gamma):
    returns = [0.0 for _ in rewards]
    next_return = 0.0
    for i in reversed(range(len(rewards))):
        next_return = rewards[i] + gamma * next_return * (1.0 - float(dones[i]))
        returns[i] = next_return
    advantages = [ret - val for ret, val in zip(returns, values)]
    return returns, advantages


def save_checkpoint(save_dir: Path, run_name: str, update: int, policy, optimizer, cfg) -> None:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": update,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    torch.save(payload, run_dir / "ckpt_last.pt")
    torch.save(payload, run_dir / f"ckpt_{update:06d}.pt")


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    policy = build_policy(cfg, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)
    opponent = build_opponent(cfg, device)
    if isinstance(opponent, SelfPlayOpponent):
        opponent.sync_from(policy)

    envs = [LocalOrbitEnv(cfg, opponent, env_index=i) for i in range(cfg.ppo.num_envs)]
    batches = [env.reset() for env in envs]
    save_dir = Path(cfg.save_dir)

    # Resume from checkpoint if requested
    start_update = 1
    if args.resume:
        checkpoint_path = save_dir / cfg.run_name / "ckpt_last.pt"
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            policy.load_state_dict(checkpoint["policy"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_update = checkpoint["update"] + 1
            print(f"Resumed from update {checkpoint['update']}, continuing from update {start_update}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")

    # Ensure policy is in training mode
    policy.train()

    print(f"Starting training: {cfg.ppo.total_updates} updates with {cfg.ppo.num_envs} envs")
    print(f"Device: {device}, Opponent: {cfg.opponent}")
    if start_update > 1:
        print(f"Continuing from update {start_update}/{cfg.ppo.total_updates}")

    for update in range(start_update, cfg.ppo.total_updates + 1):
        batch, batches, stats = collect_rollout(envs, batches, policy, cfg, device)
        metrics = ppo_update(
            policy,
            optimizer,
            batch,
            clip_coef=cfg.ppo.clip_coef,
            ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            epochs=cfg.ppo.epochs,
            minibatch_size=cfg.ppo.minibatch_size,
            device=device,
        )
        if isinstance(opponent, SelfPlayOpponent) and update % cfg.self_play_update_interval == 0:
            opponent.sync_from(policy)
        if update % cfg.log_every == 0:
            print(
                f"update={update}/{cfg.ppo.total_updates} reward_mean={stats['episode_reward_mean']:.2f} "
                f"episodes={stats['episodes_finished']} samples={stats['samples']} "
                f"loss={metrics['loss']:.4f} entropy={metrics['entropy']:.4f}"
            )
        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            save_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg)

    print(f"Training complete! Model saved to {save_dir / cfg.run_name}")


if __name__ == "__main__":
    main()
