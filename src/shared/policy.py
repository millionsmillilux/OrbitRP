from __future__ import annotations

import torch
import torch.nn as nn

from .constants import SHIP_BUCKET_COUNT
from .actions import SampledAction


class PolicyOutput:
    def __init__(self, target_logits: torch.Tensor, ship_logits: torch.Tensor, value: torch.Tensor):
        self.target_logits = target_logits
        self.ship_logits = ship_logits
        self.value = value


class PlanetPolicy(nn.Module):
    def __init__(self, self_dim: int, candidate_dim: int, global_dim: int, candidate_count: int, hidden_size: int):
        super().__init__()
        self.candidate_count = candidate_count
        self.self_encoder = _mlp(self_dim, hidden_size)
        self.candidate_encoder = _mlp(candidate_dim, hidden_size)
        self.global_encoder = _mlp(global_dim, hidden_size)

        combined_dim = hidden_size * 3
        self.target_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.ship_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, SHIP_BUCKET_COUNT),
        )
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
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

        target_logits = self.target_head(joint).squeeze(-1)
        target_logits = target_logits.masked_fill(~candidate_mask.bool(), torch.finfo(target_logits.dtype).min)

        pooled_candidates = candidate_h.mean(dim=1)
        ship_input = torch.cat([self_h, global_h, pooled_candidates], dim=-1)
        ship_logits = self.ship_head(ship_input)

        value = self.value_head(torch.cat([self_h, global_h, pooled_candidates], dim=-1)).squeeze(-1)
        return PolicyOutput(target_logits=target_logits, ship_logits=ship_logits, value=value)


def _mlp(input_dim: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
    )
