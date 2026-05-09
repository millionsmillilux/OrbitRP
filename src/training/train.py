from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch

from src.shared.constants import load_train_config
from src.shared.actions import actions_from_indices
from src.shared.features import TurnBatch, encode_turn
from src.shared.policy import PlanetPolicy
from src.shared.constants import candidate_feature_dim, global_feature_dim, self_feature_dim
from src.training.env import LocalOrbitEnv
from src.training.opponents import build_opponent, ChampionOpponent
from src.training.ppo import TransitionBatch, ppo_update, sample_actions, discounted_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml"))
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use (auto, cpu, cuda)")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_policy(cfg, device: torch.device) -> PlanetPolicy:
    return PlanetPolicy(
        self_feature_dim(),
        candidate_feature_dim(),
        global_feature_dim(),
        cfg.env.candidate_count,
        cfg.model.hidden_size,
    ).to(device)


def collect_rollout(envs, batches, policy, cfg, device: torch.device):
    self_rows = []
    candidate_rows = []
    global_rows = []
    masks = []
    target_indices = []
    ship_indices = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    episode_rewards = []
    episode_lengths = []
    running_rewards = [0.0 for _ in envs]
    running_lengths = [0 for _ in envs]

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

            moves = actions_from_indices(batch, sampled.target_index.cpu().tolist(), sampled.ship_index.cpu().tolist())
            result = env.step(moves)
            step_reward = float(result.reward) / 100.0
            running_rewards[env_idx] += step_reward
            running_lengths[env_idx] += 1

            self_rows.extend(batch.self_features)
            candidate_rows.extend(batch.candidate_features)
            global_rows.extend(batch.global_features)
            masks.extend(batch.candidate_mask)
            target_indices.extend(sampled.target_index.cpu().tolist())
            ship_indices.extend(sampled.ship_index.cpu().tolist())
            log_probs.extend(sampled.log_prob.cpu().tolist())
            values.extend(outputs.value.cpu().tolist())
            rewards.extend([step_reward / max(1, batch.self_features.shape[0]) for _ in range(batch.self_features.shape[0])])
            dones.extend([result.done for _ in range(batch.self_features.shape[0])])

            if result.done:
                episode_rewards.append(running_rewards[env_idx])
                episode_lengths.append(running_lengths[env_idx])
                running_rewards[env_idx] = 0.0
                running_lengths[env_idx] = 0
                batches[env_idx] = env.reset()
            else:
                batches[env_idx] = result.batch

    returns, advantages = discounted_returns(rewards, dones, values, cfg.ppo.gamma, cfg.ppo.gae_lambda)
    transition_batch = TransitionBatch(
        self_features=torch.tensor(np.asarray(self_rows), dtype=torch.float32),
        candidate_features=torch.tensor(np.asarray(candidate_rows), dtype=torch.float32),
        global_features=torch.tensor(np.asarray(global_rows), dtype=torch.float32),
        candidate_mask=torch.tensor(np.asarray(masks), dtype=torch.bool),
        target_index=torch.tensor(target_indices, dtype=torch.long),
        ship_index=torch.tensor(ship_indices, dtype=torch.long),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )
    stats = {
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episodes_finished": len(episode_rewards),
        "average_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "samples": len(values),
    }
    return transition_batch, batches, stats


def save_checkpoint(save_dir: Path, run_name: str, update: int, policy, optimizer, cfg, checkpoint_interval: int = 100) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "update": update,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    
    torch.save(payload, run_dir / "ckpt_last.pt")
    torch.save(payload, save_dir / "ckpt_last.pt")
    
    if update % checkpoint_interval == 0:
        torch.save(payload, run_dir / f"ckpt_{update:06d}.pt")
        torch.save(payload, save_dir / f"ckpt_{update:06d}.pt")


def save_best_checkpoint(save_dir: Path, run_name: str, update: int, policy, optimizer, cfg) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "update": update,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    
    torch.save(payload, run_dir / "ckpt_best.pt")
    torch.save(payload, save_dir / "ckpt_best.pt")


def evaluate_against_champion(policy, cfg, device, num_matches=5) -> float:
    policy.eval()
    opponent = ChampionOpponent()
    wins = 0
    for _ in range(num_matches):
        env = LocalOrbitEnv(cfg, opponent, env_index=0)
        batch = env.reset()
        while True:
            if batch.self_features.shape[0] == 0:
                result = env.step([])
                batch = result.batch if not result.done else env.reset()
                continue
            with torch.inference_mode():
                outputs = policy(
                    torch.from_numpy(batch.self_features).to(device),
                    torch.from_numpy(batch.candidate_features).to(device),
                    torch.from_numpy(batch.global_features).to(device),
                    torch.from_numpy(batch.candidate_mask).to(device).bool(),
                )
                target_idx = outputs.target_logits.argmax(dim=1).cpu().tolist()
                ship_idx = outputs.ship_logits.argmax(dim=1).cpu().tolist()
            moves = actions_from_indices(batch, target_idx, ship_idx)
            result = env.step(moves)
            batch = result.batch if not result.done else None
            if result.done:
                final_state = batch.state if batch is not None else result.batch.state
                player_owned = [p for p in final_state.planets if p.owner == final_state.player]
                if len(player_owned) > 0:
                    wins += 1
                break
    policy.train()
    return (wins / max(1, num_matches)) * 100.0


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    seed_everything(cfg.seed)
    device_name = args.device if args.device is not None else cfg.device
    device = resolve_device(device_name)
    
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    policy = build_policy(cfg, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - min(step / max(1, cfg.ppo.total_updates), 1.0) if cfg.ppo.lr_decay else 1.0,
    )

    opponent = build_opponent(cfg, device)
    if hasattr(opponent, "sync_from"):
        opponent.sync_from(policy, start_update)

    envs = [LocalOrbitEnv(cfg, opponent, env_index=i) for i in range(cfg.ppo.num_envs)]
    batches = [env.reset() for env in envs]

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

    policy.train()
    total_updates = cfg.ppo.total_updates
    print(f"Starting training: {total_updates} updates with {cfg.ppo.num_envs} envs")
    print(f"Device: {device}, Opponent: {cfg.opponent}")
    if start_update > 1:
        print(f"Continuing from update {start_update}/{total_updates}")

    best_mean_reward = float("-inf")
    best_win_rate = 0.0
    best_state_dict = None

    try:
        for update in range(start_update, total_updates + 1):
            batch, batches, stats = collect_rollout(envs, batches, policy, cfg, device)
            metrics = ppo_update(
                policy,
                optimizer,
                batch,
                clip_coef=cfg.ppo.clip_coef,
                ent_coef=max(cfg.ppo.ent_coef_min, cfg.ppo.ent_coef * (1.0 - update / total_updates)),
                vf_coef=cfg.ppo.vf_coef,
                max_grad_norm=cfg.ppo.max_grad_norm,
                epochs=cfg.ppo.epochs,
                minibatch_size=cfg.ppo.minibatch_size,
                device=device,
            )
            if cfg.ppo.lr_decay:
                scheduler.step()

            # Track best policy by training reward
            if stats['episodes_finished'] > 0 and stats['episode_reward_mean'] > best_mean_reward:
                best_mean_reward = stats['episode_reward_mean']
                best_state_dict = copy.deepcopy(policy.state_dict())

            # Regular self-play sync (keeps pool diverse)
            if hasattr(opponent, "sync_from") and update % cfg.self_play_update_interval == 0:
                opponent.sync_from(policy, update)

            # Periodic champion evaluation
            if update % cfg.eval_every == 0 and update >= cfg.eval_every:
                win_rate = evaluate_against_champion(policy, cfg, device, num_matches=cfg.eval_episodes)
                print(f"  eval: win_rate={win_rate:.1f}% against champion (best={best_win_rate:.1f}%)")
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_state_dict = copy.deepcopy(policy.state_dict())
                    save_best_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg)
                    print(f"  New best policy saved (win_rate={win_rate:.1f}%)")

            # Sync best policy to self-play pool
            if best_state_dict is not None and hasattr(opponent, "sync_from") and update % cfg.best_sync_interval == 0:
                old_state = policy.state_dict()
                policy.load_state_dict(best_state_dict)
                opponent.sync_from(policy, update)
                policy.load_state_dict(old_state)
                print(f"  Synced best policy to opponent pool (best_mean_reward={best_mean_reward:.2f}, best_win_rate={best_win_rate:.1f}%)")

            if update % cfg.log_every == 0:
                print(
                    f"update={update}/{total_updates} "
                    f"reward_mean={stats['episode_reward_mean']:.2f} "
                    f"episodes={stats['episodes_finished']} "
                    f"avg_len={stats['average_episode_length']:.1f} "
                    f"loss={metrics['loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )
            if update % cfg.checkpoint_every == 0 or update == total_updates:
                save_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg, cfg.checkpoint_every)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint for graceful shutdown...")
        if 'update' in locals():
            save_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg, 1)
            if best_state_dict is not None:
                save_best_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg)
        print("Checkpoint saved. Exiting.")
        return

    # Final best save
    if best_state_dict is not None:
        save_best_checkpoint(save_dir, cfg.run_name, total_updates, policy, optimizer, cfg)
        print(f"Final best policy saved (best_mean_reward={best_mean_reward:.2f})")

    print(f"Training complete! Model saved to {save_dir / cfg.run_name}")


if __name__ == "__main__":
    main()
