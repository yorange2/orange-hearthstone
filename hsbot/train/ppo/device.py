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

# On this transformer's PPO rollout, MPS measured ~6x slower than CPU (17 vs 105 steps/s,
# --size small). The measurement stands; the mechanism originally given for it (nested-tensor
# fallback) does not — see the module docstring. `auto` still honours the hardware
# (cuda > mps > cpu); pass --device cpu if throughput matters more.
#
# The warning is unconditional, so it also fires during behavioral cloning, where the README's
# device table claims MPS *wins* by ~5x. Those two claims are in tension and only one of them
# can be steering the recommended recipe; re-measure before trusting either.
MPS_WARNING = ("note: MPS measured ~6x slower than CPU on this model's PPO rollout; "
               "pass --device cpu to compare")


def _available(name: str) -> bool:
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available()
    return True


def resolve_device(spec: str) -> str:
    """Map a --device spec to a concrete torch device string.

    "auto" -> first available of cuda, mps, cpu. An explicitly requested accelerator that
    isn't available degrades to cpu with a warning rather than crashing mid-run.
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
        print(MPS_WARNING, flush=True)
    return device
