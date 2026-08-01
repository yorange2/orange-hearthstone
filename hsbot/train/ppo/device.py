"""Compute-device selection: `--device auto` picks the best backend available.

PYTORCH_ENABLE_MPS_FALLBACK is set ahead of `import torch` because the variable is documented
as read when PyTorch registers the fallback kernel — the placement that works under every
version, at no cost. It is a safety net, not something this model currently relies on.

Note on what MPS does *not* do here. An earlier version of this file claimed the transformer's
nested-tensor ops lack MPS kernels and fall back to CPU, paying a round-trip per call. That is
wrong, and the correction matters because it was the stated reason for a performance number.
`torch.nn.TransformerEncoder` gates its nested-tensor fast path on
`src.device.type in ("cpu", "cuda", privateuse1)` (torch 2.8,
`torch/nn/modules/transformer.py:496`); MPS is not in that list, so `convert_to_nested` stays
False and `torch._nested_tensor_from_mask` is never reached. Verified by counting calls:

    cpu  eval   _nested_tensor_from_mask=1   left_aligned_check=1
    mps  eval   _nested_tensor_from_mask=0   left_aligned_check=1
    cpu  train  _nested_tensor_from_mask=0   left_aligned_check=0
    mps  train  _nested_tensor_from_mask=0   left_aligned_check=0

Two consequences. There is no silent nested-tensor fallback to avoid on MPS, so any patch that
"works around" one is a no-op. And the fast path is disabled in *training* mode on every device
(`first_layer.training` is checked before the device gate), so mask handling cannot explain a
train-time device difference at all — whatever makes MPS slower here lies elsewhere, most
plausibly per-call dispatch overhead on small, constantly-reshaped tensors.
"""
from __future__ import annotations
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (kept below the setdefault above; see module docstring)

CHOICES = ["auto", "cpu", "cuda", "mps"]

# Which device wins depends on the PHASE, so the advice has to as well. Re-measured on an
# M-series (10-core), `small`, back to back, with the totals below including a device-independent
# data-generation step and process startup — which understates the MPS win on `batch`:
#
#   phase     workload                              CPU          MPS         winner
#   batch     BC, 3 epochs (blind / text)           23.1 / 22.9s 12.2 / 11.6s MPS ~2.5x
#   rollout   PPO 5 iters x 2048 steps              990 steps/s  104 steps/s  CPU  ~9.5x
#
# The earlier single unconditional warning was wrong for BC — it fired on the one phase where
# MPS is the right choice. It also attributed the rollout gap to a nested-tensor fallback that
# does not happen (see the module docstring); the gap is real, the mechanism was not.
MPS_ROLLOUT_WARNING = ("note: MPS measured ~9x slower than CPU on this model's rollout "
                       "(ragged shapes defeat graph caching); pass --device cpu")
MPS_BATCH_NOTE = "note: MPS measured ~2.5x faster than CPU for this phase's batched updates"


def _available(name: str) -> bool:
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available()
    return True


def resolve_device(spec: str, phase: str = "rollout") -> str:
    """Map a --device spec to a concrete torch device string.

    "auto" -> first available of cuda, mps, cpu. An explicitly requested accelerator that
    isn't available degrades to cpu with a warning rather than crashing mid-run.

    `phase` selects which MPS advice to print, because the answer genuinely differs:
    "rollout" (PPO, tiny ragged shapes — CPU wins ~9x) or "batch" (BC and other big fixed-shape
    updates — MPS wins ~2.5x). Defaulting to "rollout" keeps the conservative warning for any
    caller that has not thought about it.
    """
    if spec == "auto":
        device = next((d for d in ("cuda", "mps") if _available(d)), "cpu")
    elif not _available(spec):
        print(f"warning: --device {spec} requested but unavailable; falling back to cpu", flush=True)
        device = "cpu"
    else:
        device = spec

    detail = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
    print(f"device: {device}{detail}" + (" [auto]" if spec == "auto" else ""), flush=True)
    if device == "mps":
        print(MPS_BATCH_NOTE if phase == "batch" else MPS_ROLLOUT_WARNING, flush=True)
    return device
