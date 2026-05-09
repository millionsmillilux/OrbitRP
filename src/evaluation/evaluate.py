from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.shared.features import encode_turn
from src.shared.policy import PlanetPolicy
from src.shared.constants import (
    candidate_feature_dim,
    global_feature_dim,
    self_feature_dim,
    load_train_config,
)
from src.shared.actions import actions_from_indices
from src.training.env import LocalOrbitEnv
from src.training.opponents import ChampionOpponent


def evaluate_agent(checkpoint_path: str, num_matches: int = 5, device_name: str = "auto") -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}\nTrain first: python train.py --config configs/default.yaml")

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cuda" if device_name == "cuda" else "cpu"))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config")
    if cfg is None:
        raise ValueError("Checkpoint does not contain config metadata")

    policy = PlanetPolicy(
        self_feature_dim(),
        candidate_feature_dim(),
        global_feature_dim(),
        cfg.env.candidate_count,
        cfg.model.hidden_size,
    ).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}")

    opponent = ChampionOpponent()
    print(f"\nEvaluating against Champion opponent for {num_matches} matches...")

    wins = 0
    for match_idx in range(num_matches):
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

    win_rate = (wins / num_matches) * 100
    print(f"\nResults: {wins}/{num_matches} wins ({win_rate:.1f}% win rate)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Agent A checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--num_matches", type=int, default=5, help="Number of matches to evaluate")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    args = parser.parse_args()

    evaluate_agent(args.checkpoint, args.num_matches, args.device)
