"""Value network: observable-state features (144) -> win probability in [0,1].

Trained on SabberStone self-play data; exported to ONNX and consumed live by the
HS-Script plugin's OnnxValueNet (as the MCTS ScoreCalculator). See hsbot/docs/FEATURES.md.
"""
from __future__ import annotations
import torch
import torch.nn as nn

FEATURE_DIM = 144


class ValueNet(nn.Module):
    def __init__(self, dim: int = FEATURE_DIM, hidden=(256, 128), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]  # logit
        self.body = nn.Sequential(*layers)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)  # [batch, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inference / ONNX output: calibrated win probability. MCTS only compares scores,
        # so monotonicity is what matters, but a real probability is easy to reason about.
        return torch.sigmoid(self.body(x))  # [batch, 1]
