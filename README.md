# Orbit Wars Agent A - RL Training & Submission

## Overview

**Orbit Wars Agent A** is a single-policy reinforcement learning agent trained with PPO (Proximal Policy Optimization) for the Orbit Wars game. The repository contains a clean, self-contained training pipeline and a Kaggle submission generator.

## Quick Start

### Training
```bash
python train.py --config configs/default.yaml
```

### Evaluate Latest Checkpoint
```bash
python evaluate.py --checkpoint artifacts/ckpt_last.pt
```

### Build Kaggle Submission
```bash
python build_submission.py --checkpoint artifacts/ckpt_last.pt
```

The generated submission will be written to `agents/submission/submission.py`.

## Repository Structure

```
├── agents/
│   ├── agent_a.py              # Heuristic baseline agent (for reference)
│   └── submission/             # Generated Kaggle agents
├── artifacts/                  # Training checkpoints
├── configs/
│   └── default.yaml            # Main training configuration
├── src/
│   ├── shared/                 # Shared modules (features, policy, actions, constants)
│   ├── training/               # Training pipeline (PPO, environment, opponents)
│   ├── inference/              # Inference utilities (if needed)
│   ├── evaluation/             # Evaluation utilities
│   └── game_types.py           # Game state type definitions
├── build_submission.py         # Submission builder script
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation entry point
├── README.md                   # This file
└── PROJECT_OVERVIEW.md         # Detailed architecture documentation
```

## Key Features

### Single-Policy Training
- **One trainable policy**: Agent A is the only learnable policy
- **Self-training**: The agent trains against itself using a synchronized policy pool
- **Unified feature pipeline**: Shared feature encoding for training and inference
- **Learnable action space**: Both target selection and ship allocation are learned

### Architecture
- **Feature encoding**: Encodes planet state, fleet state, and global game state
- **Policy network**: Separate heads for target selection, ship bucket selection, and value estimation
- **PPO optimization**: Clipped surrogate loss with entropy regularization and GAE
- **Action masking**: Prevents illegal target selections during inference

### Submission Pipeline
- **Clean separation**: Training code is independent of submission code
- **Standalone agent**: Generated submission contains no PyTorch dependencies
- **Embedded weights**: Model parameters are embedded directly in the submission file
- **Kaggle-ready**: Generated submission can be directly submitted to Kaggle

## Training Configuration

The training configuration is defined in `configs/default.yaml`:

```yaml
seed: 42
run_name: orbit_wars_ppo
device: auto
save_dir: artifacts
checkpoint_every: 25
log_every: 1
opponent: self
self_play_update_interval: 25
env:
  board_size: 100.0
  episode_steps: 200
  candidate_count: 8
  max_planets: 16
  max_ships: 500.0
  max_production: 5.0
model:
  hidden_size: 128
ppo:
  rollout_steps: 64
  num_envs: 4
  total_updates: 3000
  epochs: 4
  minibatch_size: 256
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  ent_coef: 0.01
  ent_coef_min: 0.001
  vf_coef: 0.5
  lr: 0.0003
  lr_decay: true
  max_grad_norm: 0.5
  normalize_rewards: true
```

### Key Config Parameters
- `opponent: self` - Use self-training only
- `candidate_count` - Number of target planets to consider per decision
- `hidden_size` - Policy network hidden layer size
- `total_updates` - Number of PPO optimization steps
- `rollout_steps` - Experience collection per environment per update

## Training Process

1. **Initialization**
   - Load or create the policy network
   - Prepare parallel training environments
   - Initialize optimizer and learning rate scheduler

2. **Rollout Collection**
   - Run 4 parallel environments for `rollout_steps` steps each
   - Collect observations, actions, rewards, and values
   - Compute returns and advantages using GAE

3. **PPO Update**
   - Shuffle collected transitions
   - Split into mini-batches
   - Update policy with clipped surrogate loss
   - Update value head with MSE loss
   - Apply entropy regularization
   - Update opponent pool every 25 updates

4. **Checkpointing**
   - Save checkpoint every 25 updates
   - Store policy weights, optimizer state, and config

## Model Architecture

### Feature Encoding
- **Self features** (15 dimensions): Planet position, ships, production, planetary counts, etc.
- **Candidate features** (19 dimensions per candidate): Target properties, distance, vulnerability, pressure
- **Global features** (13 dimensions): Game state aggregates, nearest threats

### Policy Network
- **Encoders**: Separate MLPs for self, candidate, and global features
- **Joint representation**: Concatenate expanded self, global, and per-candidate encodings
- **Target head**: Selects which candidate planet to attack
- **Ship head**: Selects which ship bucket ratio to use (0.1x to 1.0x ships available)
- **Value head**: Estimates expected return

## Shared Modules

All training and inference code shares the same:
- **Feature dimensions**: `src/shared/constants.py`
- **Feature encoding**: `src/shared/features.py`
- **Policy architecture**: `src/shared/policy.py`
- **Action decoding**: `src/shared/actions.py`
- **Ship buckets**: `src/shared/constants.SHIP_BUCKET_RATIOS`

This ensures perfect alignment between training and submission agents.

## Submission Generation

The `build_submission.py` script:
1. Loads the checkpoint with config
2. Serializes model weights to base64+pickle
3. Generates a standalone `submission.py` file
4. Embeds weights directly in the file
5. Includes all feature encoding and policy logic

### Usage
```bash
python build_submission.py --checkpoint artifacts/ckpt_last.pt --output agents/submission/submission.py
```

The generated submission contains:
- Embedded model weights (base64 encoded)
- Feature encoding functions
- Policy forward pass (NumPy-only)
- Action decoding logic
- Kaggle-compatible `agent(obs)` function

## Evaluation

To evaluate the latest checkpoint:
```bash
python evaluate.py --checkpoint artifacts/ckpt_last.pt --num_matches 5
```

The evaluation script:
- Loads the checkpoint
- Runs the policy deterministically
- Plays multiple matches against the Champion heuristic
- Reports win rate

## Model Parameters

The policy network learns approximately:
- Feature encoders: ~150k parameters
- Target head: ~35k parameters
- Ship head: ~35k parameters
- Value head: ~35k parameters

**Total**: ~250k trainable parameters

## Self-Training Details

The training pipeline uses only self-play:
- A main agent learns from experience
- A synchronized opponent pool (max 5 older versions) provides diverse opponents
- Every 25 updates, the current policy is added to the opponent pool
- Opponents are selected randomly from the pool
- This encourages the agent to learn general strategies

No external heuristic opponents or multi-agent training are used.

## Files Removed in Refactoring

The following files were removed as part of the single-agent refactoring:
- `agents/agent_b.py`, `agents/agent_c.py` (heuristic opponents)
- Old training scripts and forwarding modules
- Old submission builders and agents
- Multi-agent training infrastructure
- Configuration files now replaced by YAML

See `PROJECT_OVERVIEW.md` for detailed architecture information.

## Requirements

- Python 3.9+
- PyTorch
- NumPy

## License

See LICENSE file.
