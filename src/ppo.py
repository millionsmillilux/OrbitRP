from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .policy import PolicyOutput


@dataclass(slots=True)
class SampledAction:
    target_index: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(slots=True)
class TransitionBatch:
    self_features: torch.Tensor
    candidate_features: torch.Tensor
    global_features: torch.Tensor
    candidate_mask: torch.Tensor
    target_index: torch.Tensor
    log_prob: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def sample_actions(outputs: PolicyOutput, deterministic: bool = False) -> SampledAction:
    logits = outputs.target_logits
    dist = Categorical(logits=logits)
    target_index = logits.argmax(dim=-1) if deterministic else dist.sample()
    log_prob = dist.log_prob(target_index)
    entropy = dist.entropy()
    return SampledAction(target_index=target_index, log_prob=log_prob, entropy=entropy)


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
            idx = order[start:start + minibatch_size]
            outputs = policy(
                self_features[idx],
                candidate_features[idx],
                global_features[idx],
                candidate_mask[idx],
            )
            dist = Categorical(logits=outputs.target_logits)
            new_log_prob = dist.log_prob(target_index[idx])
            entropy = dist.entropy().mean()
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
