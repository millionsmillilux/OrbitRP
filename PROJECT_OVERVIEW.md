# Orbit Wars Agent A - RL Architecture Overview

## Single-Agent Design

This is a **single-policy, self-training** reinforcement learning project. Agent A is the only trainable policy. There are no multiple independent agents, no multi-agent training, and no agent switching logic.

**The repository contains exactly one learning system: Agent A.**

## Core Principle

All training and inference code shares the exact same:
- Feature encoding pipeline
- Policy network architecture  
- Action decoding logic
- Hyperparameters and constants
- Normalization and masking rules

This ensures **perfect alignment** between what the agent learns during training and what it executes in submission.

## Architecture Overview

### Shared Modules (`src/shared/`)

These modules define the one-and-only interface for learning and inference:

#### `constants.py`
- Environment constants (board size, max ships, episode steps, etc.)
- Feature dimension calculations: `self_feature_dim()`, `candidate_feature_dim()`, `global_feature_dim()`
- Ship bucket ratios: `SHIP_BUCKET_RATIOS = (0.1, 0.2, ..., 1.0)`
- Configuration classes: `EnvConfig`, `ModelConfig`, `PPOConfig`, `TrainConfig`
- Config loading with YAML/JSON support: `load_train_config(path)`

#### `features.py`
- `TurnBatch`: Structure holding encoded features and contexts for a decision batch
- `DecisionContext`: Per-decision metadata (source planet ID, candidates, masks)
- `encode_turn(observation, env_cfg)`: Converts game observation into feature tensors
  - `build_self_features()`: Source planet properties (15 dims)
  - `build_candidate_features()`: Target properties (19 dims each)
  - `build_global_features()`: Game state aggregates (13 dims)
  - `build_candidates()`: Selects top candidates by proximity + ownership

#### `policy.py`
- `PlanetPolicy`: The ONE policy network architecture
  - **Encoders**: MLPs for self, candidate, global features
  - **Shared backbone**: Joint encoding of all information
  - **Target head**: Outputs logits for candidate selection (8 outputs for 8 candidates)
  - **Ship head**: Outputs logits for ship bucket selection (8 outputs for 8 buckets)
  - **Value head**: Outputs single scalar for advantage estimation
  - **Forward pass**: Returns `PolicyOutput(target_logits, ship_logits, value)`

#### `actions.py`
- `bucket_to_ships(source_ships, bucket_index)`: Converts bucket (0-7) to ship count
  - Uses `SHIP_BUCKET_RATIOS` to determine percentage of available ships
  - Ensures deterministic, learnable action space
- `actions_from_indices(batch, target_indices, ship_indices)`: Decodes network outputs into game moves
  - Applies `candidate_mask` to prevent illegal targets
  - Validates action legality before returning

#### `utils.py`
- `clamp()`, `normalize()`, `distance()`: Numeric helpers
- `safe_ratio()`: Safe division (prevents NaN)

### Training Modules (`src/training/`)

#### `train.py`
- `main()`: Training entrypoint
  - Argument parsing: `--config configs/default.yaml`, `--resume`
  - Device resolution: auto CPU/GPU detection
  - Policy initialization with config dimensions
  - Optimizer setup: Adam with learning rate decay
  - Rollout collection loop
  - PPO update loop with logging
  - Checkpoint saving

#### `ppo.py`
- `collect_rollout()`: Gathers experience from parallel environments
  - Runs policy in `inference_mode()`
  - Samples actions from policy outputs
  - Collects rewards, dones, and values
  - Returns `TransitionBatch` for optimization
- `ppo_update()`: Performs PPO optimization step
  - Computes returns and advantages using GAE
  - Shuffles and mini-batches transitions
  - Updates policy with clipped surrogate loss
  - Updates value with MSE loss
  - Applies entropy regularization
  - Entropy decay schedule: `ent_coef * (1 - update/total_updates)`
  - Returns per-update metrics
- `sample_actions()`: Samples target and ship bucket from policy outputs
  - Deterministic (argmax) during evaluation
  - Stochastic (categorical) during training (if not sampled deterministically)

#### `env.py`
- `LocalOrbitEnv`: Wrapper around the Orbit Wars simulator
  - Converts observations to `TurnBatch` using `encode_turn()`
  - Executes moves from `actions_from_indices()`
  - Tracks rewards and episode termination
  - Supports alternate player sides for fairness

#### `opponents.py`
- `ChampionOpponent`: Loads heuristic baseline (Agent A from `agents/agent_a.py`)
- `SelfPlayOpponent`: Maintains pool of past policy versions
  - Syncs from current policy
  - Keeps last 5 versions
  - Samples randomly during training
- `build_opponent()`: Factory for creating opponent instances based on config

### Submission Generation

#### `build_submission.py` (root level)
- Loads checkpoint with embedded config
- Extracts model weights: `checkpoint["policy"]`
- Serializes weights to base64(pickle())
- Generates `agents/submission/submission.py`:
  - Includes all shared feature encoding
  - Includes NumPy-based policy forward pass
  - Embeds weights directly in code
  - Provides `agent(obs)` function for Kaggle
  - No PyTorch dependency
  - Deterministic (argmax actions)

### Game Type Definitions

#### `game_types.py`
- `PlanetState`: id, owner, x, y, radius, ships, production
- `FleetState`: id, owner, source, x, y, vx, vy, ships
- `GameState`: step, player, planets[], fleets[]
- `parse_observation()`: Converts raw observation dict to `GameState`

## Data Flow: Training

```
1. LocalOrbitEnv.reset()
   └─> Initial observation
       └─> encode_turn()
           └─> TurnBatch (features + contexts)

2. Policy forward pass
   └─> PolicyOutput (target_logits, ship_logits, value)
   
3. sample_actions()
   └─> target_index, ship_index, log_prob, entropy

4. actions_from_indices(batch, target_index, ship_index)
   └─> list of [source_id, target_id, ships] moves

5. env.step(moves)
   └─> reward, done, new observation
   
6. PPO optimization
   └─> Compute GAE
   └─> Update policy + value head
   └─> Entropy regularization
   └─> Sync opponent pool if needed
```

## Data Flow: Inference (Submission)

```
1. kaggle calls agent(obs)

2. encode_turn(obs)
   └─> TurnBatch using same feature logic as training

3. policy_forward()
   └─> NumPy-only forward pass
   └─> target_logits, ship_logits

4. argmax (deterministic selection)
   └─> target_index, ship_index

5. actions_from_indices()
   └─> Legal moves

6. Return moves to Kaggle
```

## Key Design Decisions

### Why Single Policy?
- **Simplicity**: One network to train, one network to submit
- **Alignment**: Training and submission use identical logic
- **Clarity**: No confusion about which code is active
- **Efficiency**: Reduced memory and code duplication

### Why Self-Play Only?
- **Curriculum learning**: Agent adapts to improving opponents (itself)
- **Exploration**: Opponent pool provides diverse strategies
- **Scalability**: Easy to extend with more opponent versions

### Why Learnable Ship Buckets?
- **Flexibility**: Agent learns when to send more or fewer ships
- **Learnability**: 8 discrete buckets balance exploration and action space size
- **Stability**: Discrete actions are easier to optimize than continuous

### Why Separate Heads?
- **Modularity**: Target and ship decisions are independently optimizable
- **Scaling**: Each head can have its own capacity as needed

### Why Perfect Alignment?
- **No surprises**: What you train is what you deploy
- **Debugging**: Errors in training immediately visible in submission
- **Maintenance**: Single source of truth for all logic

## Configuration

Main config file: `configs/default.yaml`

**Environment settings**:
- `board_size`: 100.0 (Orbit Wars universe size)
- `episode_steps`: 200 (steps per episode)
- `candidate_count`: 8 (top planets to consider per decision)
- `max_planets`: 16 (max planets in game)
- `max_ships`: 500.0 (max ships on one planet)
- `max_production`: 5.0 (max production rate)

**Model settings**:
- `hidden_size`: 128 (MLP hidden layer size)

**PPO settings**:
- `rollout_steps`: 64 (experience collected before update)
- `num_envs`: 4 (parallel environments)
- `total_updates`: 3000 (total optimization steps)
- `epochs`: 4 (data re-use per update)
- `minibatch_size`: 256
- `gamma`: 0.99 (discount factor)
- `gae_lambda`: 0.95 (GAE parameter)
- `clip_coef`: 0.2 (PPO clipping range)
- `ent_coef`: 0.01 (entropy coefficient)
- `ent_coef_min`: 0.001 (entropy floor)
- `vf_coef`: 0.5 (value loss coefficient)
- `lr`: 0.0003 (learning rate)
- `lr_decay`: true (linear decay to 0)
- `max_grad_norm`: 0.5 (gradient clipping)
- `normalize_rewards`: true (running reward normalization)

## Removed Infrastructure

The following were removed in the single-agent refactoring:

**Multi-agent files**:
- `agents/agent_b.py`, `agents/agent_c.py` (heuristic opponents)
- `agents/rl_agent.py`, `agents/wrapper.py` (old RL wrappers)
- Old opponent selection logic
- Multi-agent training infrastructure

**Legacy training scripts**:
- `src/train.py` (forwarding module)
- `src/evaluate.py` (old evaluation)
- `src/config.py` (forwarding module)
- `src/features.py`, `src/policy.py`, `src/ppo.py` (forwarding modules)
- `train_population.py` (population training)
- `validate_learning.py` (old validation)

**Old submission builders**:
- `export_for_kaggle.py`
- `kaggle_agent.py`
- Old submission generation logic

**Old configs**:
- `champion_cfg.json`, `economy_cfg.json`

**Other**:
- Old environment files: `env.py`, `src/local_env.py`
- Test files: `test.py`
- Placeholder: `main.py`
- Model files: `model_weights.pkl`

## Commands

### Training Agent A
```bash
python train.py --config configs/default.yaml
```

### Resume Training
```bash
python train.py --config configs/default.yaml --resume
```

### Evaluate Latest Checkpoint
```bash
python evaluate.py --checkpoint artifacts/ckpt_last.pt --num_matches 5
```

### Generate Kaggle Submission
```bash
python build_submission.py --checkpoint artifacts/ckpt_last.pt
```

Output: `agents/submission/submission.py`

### Submit to Kaggle
```bash
kaggle competitions submit orbit-wars -f agents/submission/submission.py -m "Agent A RL submission"
```

## Model Capacity

**Policy network parameters**: ~250k
- Self encoder: ~60k
- Candidate encoder: ~60k
- Global encoder: ~30k
- Target head: ~35k
- Ship head: ~35k
- Value head: ~35k

## Learnability

The agent learns to:
- **Select targets**: Which planets to attack based on ownership, distance, threat, production
- **Allocate ships**: How many ships to send (8 discrete buckets)
- **Timing**: When to attack (depends on game state)
- **Aggression**: Dynamic balance between exploration and exploitation
- **Expansion**: Prioritize growth or defense based on game dynamics

Ship allocation is **NOT** hardcoded — it's learned via the ship head logits.

## Alignment Verification

To verify training/submission alignment:
1. Feature dimensions: Same in all feature encoding functions
2. Policy shape: Same network in training and submission
3. Constants: Shared from `src.shared.constants`
4. Actions: Same decoding in training and submission
5. Masking: Same mask application everywhere

Any change to feature encoding, policy, or actions must be made in shared modules only.

## Conclusion

This is a **clean, coherent, single-policy RL system** centered entirely on Agent A. The shared module architecture ensures:
- Perfect alignment between training and submission
- No duplicate logic or hardcoded values
- Easy maintenance and future improvements
- Clear visibility into what the agent learns

Training and submission are guaranteed to execute the same logic.

- Self-play builds an opponent pool from historical policy checkpoints.
- A random opponent is selected from the pool to prevent forgetting.
- The pool is periodically updated with the current learner weights.

## Checkpoint system
- Checkpoints are saved by update number and last state in `artifacts/<run_name>/`.
- `ckpt_last.pt` is used for evaluation and submission building.

## How the model learns
- The network learns target selection across candidate planets.
- A ship bucket head learns how much force to commit.
- Feature encodings include incoming fleets, local pressure, center control, and economic ratios.
- The architecture is prepared to learn strategic pacing, aggression, and expansion decisions.

## How action masking works
- The candidate mask prevents invalid target choices.
- The ship bucket logic prevents sending more ships than the source has.
- No-op target index zero is always valid.

## How to train
- `python -m src.training.train --config configs/default.yaml`

## How to evaluate
- `python evaluate.py --checkpoint artifacts/ckpt_last.pt`

## How to build submission
- `python build_submission.py --checkpoint artifacts/ckpt_last.pt`

## How to submit to Kaggle
- `kaggle competitions submit orbit-wars -f submission.py -m "RL PPO agent"`

## How to add new features
1. Add keys to `src/shared/constants.py` feature key lists.
2. Update `src/shared/features.py` with the new feature values.
3. Model dimensions and encoders adjust automatically.

## How to add new policy heads
1. Extend `src/shared/policy.py` with new output heads.
2. Update `src/training/ppo.py` to include the new log-prob and entropy terms.
3. Update action decode logic in `src/shared/actions.py` and inference.
