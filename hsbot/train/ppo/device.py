"""Compute-device selection: `--device auto` picks the best backend available.

Some of the transformer's nested-tensor ops have no MPS kernel and fall back to CPU. On
torch 2.8 that fallback is taken whether or not PYTORCH_ENABLE_MPS_FALLBACK is set (verified),
but the variable is documented as read when PyTorch registers the fallback kernel, so we set
it ahead of `import torch` — the placement that's read under every version, at no cost.
"""
from __future__ import annotations
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (kept below the setdefault above; see module docstring)

CHOICES = ["auto", "cpu", "cuda", "mps"]

# On this transformer, MPS measured ~6x slower than CPU (17 vs 105 steps/s, --size small): the
# unsupported ops fall back to CPU anyway and each fallback pays a device round-trip. `auto`
# still honours the hardware (cuda > mps > cpu); pass --device cpu if throughput matters more.
MPS_WARNING = ("note: MPS benchmarked ~6x slower than CPU for this model "
               "(nested-tensor ops fall back to CPU); pass --device cpu to compare")


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
