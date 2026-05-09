from __future__ import annotations

import torch
from torch.distributions import Categorical

from src.shared.actions import SampledAction
from src.shared.policy import PolicyOutput


class TransitionBatch:
    def __init__(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        target_index: torch.Tensor,
        ship_index: torch.Tensor,
        log_prob: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ):
        self.self_features = self_features
        self.candidate_features = candidate_features
        self.global_features = global_features
        self.candidate_mask = candidate_mask
        self.target_index = target_index
        self.ship_index = ship_index
        self.log_prob = log_prob
        self.returns = returns
        self.advantages = advantages


def sample_actions(outputs: PolicyOutput, deterministic: bool = False) -> SampledAction:
    target_dist = Categorical(logits=outputs.target_logits)
    ship_dist = Categorical(logits=outputs.ship_logits)

    if deterministic:
        target_index = outputs.target_logits.argmax(dim=-1)
        ship_index = outputs.ship_logits.argmax(dim=-1)
    else:
        target_index = target_dist.sample()
        ship_index = ship_dist.sample()

    log_prob = target_dist.log_prob(target_index) + ship_dist.log_prob(ship_index)
    entropy = target_dist.entropy() + ship_dist.entropy()
    return SampledAction(target_index=target_index, ship_index=ship_index, log_prob=log_prob, entropy=entropy)


def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:
    size = batch.self_features.shape[0]
    if size == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    self_features = batch.self_features.to(device)
    candidate_features = batch.candidate_features.to(device)
    global_features = batch.global_features.to(device)
    candidate_mask = batch.candidate_mask.to(device).bool()
    target_index = batch.target_index.to(device)
    ship_index = batch.ship_index.to(device)
    old_log_prob = batch.log_prob.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    update_count = 0
    minibatch_size = min(max(1, minibatch_size), size)

    for _ in range(epochs):
        order = torch.randperm(size, device=device)
        for start in range(0, size, minibatch_size):
            idx = order[start : start + minibatch_size]
            outputs = policy(
                self_features[idx],
                candidate_features[idx],
                global_features[idx],
                candidate_mask[idx],
            )
            target_dist = Categorical(logits=outputs.target_logits)
            ship_dist = Categorical(logits=outputs.ship_logits)
            new_log_prob = target_dist.log_prob(target_index[idx]) + ship_dist.log_prob(ship_index[idx])
            entropy = (target_dist.entropy() + ship_dist.entropy()).mean()
            ratio = (new_log_prob - old_log_prob[idx]).exp()
            policy_loss = torch.maximum(
                -advantages[idx] * ratio,
                -advantages[idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
            ).mean()
            value_loss = 0.5 * (returns[idx] - outputs.value).pow(2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy.detach().cpu())
            update_count += 1

    return {key: value / max(1, update_count) for key, value in metrics.items()}


def discounted_returns(rewards: list[float], dones: list[bool], values: list[float], gamma: float, gae_lambda: float):
    returns = [0.0 for _ in rewards]
    advantages = [0.0 for _ in rewards]
    next_value = 0.0
    next_advantage = 0.0

    for idx in reversed(range(len(rewards))):
        mask = 0.0 if dones[idx] else 1.0
        delta = rewards[idx] + gamma * next_value * mask - values[idx]
        advantages[idx] = delta + gamma * gae_lambda * next_advantage * mask
        returns[idx] = advantages[idx] + values[idx]
        next_value = values[idx]
        next_advantage = advantages[idx]

    return returns, advantages
