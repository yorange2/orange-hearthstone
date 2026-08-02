"""Compute-device selection: `--device auto` picks the best backend available.

PYTORCH_ENABLE_MPS_FALLBACK is set ahead of `import torch` because the variable is documented
as read when PyTorch registers the fallback kernel — the placement that works under every
version, at no cost. It is load-bearing on MPS, not a safety net: without it, any masked
transformer call in eval mode raises outright.

    NotImplementedError: The operator 'aten::_nested_tensor_from_mask_left_aligned'
    is not currently implemented for the MPS device.

What MPS does and does not do here. `torch.nn.TransformerEncoder` gates its nested-tensor fast
path on `src.device.type in ("cpu", "cuda", privateuse1)` (torch 2.8,
`torch/nn/modules/transformer.py:496`); MPS is not in that list, so `convert_to_nested` stays
False and `torch._nested_tensor_from_mask` is never reached. But that device gate is checked
*after* `torch._nested_tensor_from_mask_left_aligned`, which therefore still runs on MPS — and
has no MPS kernel. Verified by counting calls:

    cpu  eval   _nested_tensor_from_mask=1   left_aligned_check=1
    mps  eval   _nested_tensor_from_mask=0   left_aligned_check=1
    cpu  train  _nested_tensor_from_mask=0   left_aligned_check=0
    mps  train  _nested_tensor_from_mask=0   left_aligned_check=0

So on MPS the *conversion* never happens but the *check* does, and the check round-trips to the
CPU on every masked forward call. An earlier revision of this file read the same table as
showing there was no fallback to avoid; the `left_aligned_check=1` row is exactly the fallback.

Two consequences, and both are now acted on rather than just noted. Eval-mode calls should not
pass a mask that masks nothing: `EntityEncoder`/`TemporalEncoder`/`ActorCritic` accept None for
a mask whose entries are all True, and `act`/`batched_act` pass None when they can establish
that CPU-side (free — the tensors have not been transferred yet). Measured on B=1 rollout,
dropping both masks: small 2.0x, large 1.8x, 100x 2.7x on MPS. Dropping only one gives ~1.15x,
because whichever mask remains still pays the round-trip.

The fast path is disabled in *training* mode on every device (`first_layer.training` is checked
before the device gate), so none of this applies to the batched update, and mask handling cannot
explain a train-time device difference at all. What remains there is per-call dispatch overhead
on small tensors: MPS rollout throughput tracks layer *count*, not width — it is nearly flat
from 1.9M to 10M params at 4 layers, while CPU throughput falls with FLOPs.
"""
from __future__ import annotations
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (kept below the setdefault above; see module docstring)

CHOICES = ["auto", "cpu", "cuda", "mps"]

# Which device wins depends on the PHASE, so the advice has to as well. Measured on an M-series
# (10-core), end to end with SabberStone in the loop, 8 envs, median of 3:
#
#   phase     size     CPU          MPS          winner
#   rollout   small    1297 steps/s  157 steps/s  CPU  8.3x
#   rollout   large     594 steps/s  165 steps/s  CPU  3.6x
#   rollout   100x      236 steps/s  158 steps/s  CPU  1.5x
#   batch     small    2294 samp/s  13778 samp/s  MPS  6.0x
#   batch     large     606 samp/s   3258 samp/s  MPS  5.4x
#   batch     100x      186 samp/s    591 samp/s  MPS  3.2x
#
# So CPU still wins every rollout, but the margin collapses with model size (8.3x -> 1.5x) while
# MPS wins every batched update. The rollout gap is NOT about ragged shapes defeating a graph
# cache, which is what an earlier revision of this file claimed. It is per-call overhead: MPS
# rollout throughput tracks the number of kernel launches, so it barely moves between `small` and
# `100x` (157 -> 158 steps/s) while CPU falls with FLOPs. Scaling the model does not make MPS
# faster; it makes CPU slower. Extrapolating the trend, MPS would take the rollout somewhere past
# `100x` — but batch size is the stronger lever: at a fixed batch of 32, MPS already wins 2.0-4.5x
# at every size, and the real rollout only averages a batch of ~5.
MPS_ROLLOUT_WARNING = ("note: MPS measured slower than CPU on this model's rollout "
                       "(1.5x at 100x, 8x at small — per-call overhead); prefer --device cpu")
MPS_BATCH_NOTE = "note: MPS measured 3-6x faster than CPU for this phase's batched updates"


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


def tune_for_device(model, device):
    """Apply device-specific settings that cannot be chosen until the model has a device.

    Call once, after `.to(device)`. Returns the model, so it chains.

    Currently one setting: on MPS, turn off `TransformerEncoder`'s nested-tensor fast path. That
    path can never actually engage on MPS — it is gated on
    `src.device.type in ("cpu", "cuda", privateuse1)` — but the eligibility *check* in front of
    it (`torch._nested_tensor_from_mask_left_aligned`) runs first and has no MPS kernel, so every
    masked eval call round-trips to the CPU to answer a question whose answer is discarded.
    `use_nested_tensor=False` is tested earlier in the same elif chain (torch 2.8,
    `torch/nn/modules/transformer.py:453` vs `:466`), so clearing it skips the check outright.

    Measured end-to-end on rollout: MPS 1.3-1.4x faster with it off. It is deliberately NOT
    applied on CPU, where the fast path is reachable and does pay for itself — forcing it off
    there measured 0.80-0.98x, i.e. a loss.
    """
    if device == "mps":
        for m in model.modules():
            if isinstance(m, torch.nn.TransformerEncoder):
                m.use_nested_tensor = False
    return model
