#!/usr/bin/env python
"""
Validate that the trained agent actually learned and performs better than untrained.
This script:
1. Tests untrained policy against ChampionOpponent
2. Trains the policy
3. Tests trained policy against ChampionOpponent
4. Compares win rates to confirm learning occurred
"""
import argparse
import torch
from pathlib import Path
from src.config import load_train_config, default_train_config_path
from src.train import build_policy, resolve_device, seed_everything
from src.local_env import LocalOrbitEnv
from src.opponents import ChampionOpponent
from src.ppo import sample_actions
from src.features import encode_turn


def evaluate_policy(policy, cfg, device, num_games=20):
    """Evaluate policy against ChampionOpponent."""
    policy.eval()
    wins = losses = draws = 0
    total_score = 0.0
    
    for game in range(1, num_games + 1):
        env = LocalOrbitEnv(cfg, ChampionOpponent(), env_index=game)
        batch = env.reset()
        done = False
        score = 0.0
        
        while not done:
            obs = env.obs[env.learner_player]
            features = encode_turn(obs, cfg.env)
            if features.self_features.shape[0] == 0:
                result = env.step([])
            else:
                with torch.inference_mode():
                    outputs = policy(
                        torch.from_numpy(features.self_features).to(device),
                        torch.from_numpy(features.candidate_features).to(device),
                        torch.from_numpy(features.global_features).to(device),
                        torch.from_numpy(features.candidate_mask).to(device).bool(),
                    )
                    sampled = sample_actions(outputs, deterministic=True)
                
                from src.opponents import actions_from_indices
                actions = actions_from_indices(features, sampled.target_index.cpu().tolist())
                result = env.step(actions)
            
            score += result.reward
            done = result.done
        
        total_score += score
        if score > 0:
            wins += 1
        elif score < 0:
            losses += 1
        else:
            draws += 1
    
    win_rate = wins / max(1, num_games)
    avg_score = total_score / max(1, num_games)
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'avg_score': avg_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate that agent learning occurred by comparing untrained vs trained performance"
    )
    parser.add_argument("--config", default=str(default_train_config_path()))
    parser.add_argument("--games", type=int, default=20, help="Number of evaluation games")
    args = parser.parse_args()
    
    cfg = load_train_config(args.config)
    device = resolve_device(cfg.device)
    
    print("=" * 60)
    print("LEARNING VALIDATION TEST")
    print("=" * 60)
    
    # Test untrained policy
    print("\n1. Testing UNTRAINED policy...")
    seed_everything(cfg.seed)
    untrained_policy = build_policy(cfg, device)
    untrained_results = evaluate_policy(untrained_policy, cfg, device, args.games)
    print(f"   Win rate: {untrained_results['win_rate']:.1%} ({untrained_results['wins']}/{args.games})")
    print(f"   Avg score: {untrained_results['avg_score']:.1f}")
    
    # Load trained policy
    print("\n2. Testing TRAINED policy...")
    trained_policy = build_policy(cfg, device)
    checkpoint_path = Path(cfg.save_dir) / cfg.run_name / "ckpt_last.pt"
    
    if not checkpoint_path.exists():
        print(f"   ERROR: No checkpoint found at {checkpoint_path}")
        print("   Please run training first: python train.py")
        return
    
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trained_policy.load_state_dict(payload["policy"])
    trained_results = evaluate_policy(trained_policy, cfg, device, args.games)
    print(f"   Win rate: {trained_results['win_rate']:.1%} ({trained_results['wins']}/{args.games})")
    print(f"   Avg score: {trained_results['avg_score']:.1f}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("LEARNING VALIDATION RESULTS")
    print("=" * 60)
    win_rate_improvement = trained_results['win_rate'] - untrained_results['win_rate']
    score_improvement = trained_results['avg_score'] - untrained_results['avg_score']
    
    print(f"\nWin rate improvement: {win_rate_improvement:+.1%}")
    print(f"Score improvement: {score_improvement:+.1f}")
    
    if win_rate_improvement > 0.05 or score_improvement > 10:
        print("\n✓ LEARNING DETECTED: Agent significantly improved!")
        return 0
    elif win_rate_improvement > 0 or score_improvement > 0:
        print("\n◐ PARTIAL LEARNING: Agent improved slightly.")
        return 1
    else:
        print("\n✗ NO LEARNING: Agent did not improve. Check training configuration.")
        return 2


if __name__ == "__main__":
    exit(main())
