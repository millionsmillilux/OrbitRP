from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True)
class PolicyOutput:
    target_logits: torch.Tensor
    value: torch.Tensor


class PlanetPolicy(nn.Module):
    def __init__(self, self_dim: int, candidate_dim: int, global_dim: int, candidate_count: int, hidden_size: int):
        super().__init__()
        self.candidate_count = candidate_count
        self.self_encoder = _mlp(self_dim, hidden_size)
        self.candidate_encoder = _mlp(candidate_dim, hidden_size)
        self.global_encoder = _mlp(global_dim, hidden_size)
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> PolicyOutput:
        self_h = self.self_encoder(self_features)
        global_h = self.global_encoder(global_features)
        candidate_h = self.candidate_encoder(candidate_features)

        expanded_self = self_h.unsqueeze(1).expand(-1, self.candidate_count, -1)
        expanded_global = global_h.unsqueeze(1).expand(-1, self.candidate_count, -1)
        joint = torch.cat([expanded_self, expanded_global, candidate_h], dim=-1)
        logits = self.target_head(joint).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask.bool(), torch.finfo(logits.dtype).min)

        pooled = candidate_h.mean(dim=1)
        value = self.value_head(torch.cat([self_h, global_h, pooled], dim=-1)).squeeze(-1)
        return PolicyOutput(target_logits=logits, value=value)


def _mlp(input_dim: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
    )
