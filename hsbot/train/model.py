"""Value network: observable-state features (144) -> win probability in [0,1].

Trained on SabberStone self-play data; exported to ONNX and consumed live by the
HS-Script plugin's OnnxValueNet (as the MCTS ScoreCalculator). See hsbot/docs/FEATURES.md.
"""
from __future__ import annotations
import torch
import torch.nn as nn

FEATURE_DIM = 144


NORMS = ("none", "batch", "layer")


class ValueNet(nn.Module):
    """MLP over the 144-float observable-state contract.

    `norm` inserts a normalization layer between each Linear and its ReLU:

    - `none` (default) — the original architecture, and the one every existing checkpoint uses.
    - `batch` — `nn.BatchNorm1d`. Safe for this deployment despite the usual objection: at eval
      it uses running statistics, so it is a fixed affine map and batch-1 inference in the
      plugin's MCTS is exact. ONNX Runtime generally folds `Linear -> BatchNorm` back into one
      Linear, so it costs nothing at inference. **Pair it with `dropout=0`** — dropout shifts the
      train/test variance that BN's running stats estimate (Li et al. 2019, "Understanding the
      Disharmony between Dropout and Batch Normalization"), and `train.py` drops the dropout
      default to 0 for this mode rather than shipping the known-bad combination behind one flag.
    - `layer` — `nn.LayerNorm`, batch-independent, so no train/eval statistics at all. Normalizes
      each sample across the hidden dimension.

    Expect a modest effect at best: `FeatureExtractor` already scales the inputs by fixed
    divisors (`/30f` health/deck/turn, `/10f` hand/mana/weapon), so the classic "wildly different
    feature scales" win is largely taken before the network sees anything. What remains is
    internal covariate shift across two hidden layers of a 144->256->128->1 MLP.
    """

    def __init__(self, dim: int = FEATURE_DIM, hidden=(256, 128), dropout: float = 0.1,
                 norm: str = "none"):
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"unknown norm {norm!r}; expected one of {NORMS}")
        self.norm = norm

        layers: list[nn.Module] = []
        prev = dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            # Before the activation: normalizing the pre-activations is what keeps the ReLU
            # operating in a useful range, which is the point of doing this at all.
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]  # logit
        self.body = nn.Sequential(*layers)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)  # [batch, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inference / ONNX output: calibrated win probability. MCTS only compares scores,
        # so monotonicity is what matters, but a real probability is easy to reason about.
        return torch.sigmoid(self.body(x))  # [batch, 1]
