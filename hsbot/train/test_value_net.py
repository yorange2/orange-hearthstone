"""Tests for ValueNet's normalization options and their ONNX export.

Run directly (no pytest dependency, matching reference_features.py):

    python test_value_net.py

The property that actually matters for deployment is **batch independence at inference**. The
plugin scores one state at a time inside HS-Script's MCTS, while training runs batches of 512.
BatchNorm is the one layer here that behaves differently in those two regimes, and it fails
*silently* if it is left in training mode: the exported graph would then normalize by whatever
statistics the single input happens to have, making a card's score depend on what it was batched
with. `test_batch_independence` and `test_export_is_eval_mode` exist to catch exactly that.
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import torch

from model import ValueNet, FEATURE_DIM, NORMS

TOL = 2e-5  # float32 torch-vs-ORT agreement; the repo's own export check runs ~6e-08


def _trained_a_little(norm: str, seed: int = 0) -> ValueNet:
    """A few optimizer steps, so BatchNorm's running stats are not at their init values.

    Testing an untrained BN is close to vacuous: running_mean=0 / running_var=1 makes it an
    identity-ish map, so a train/eval mismatch would not show up.
    """
    torch.manual_seed(seed)
    model = ValueNet(dropout=0.0, norm=norm)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(20):
        x = torch.from_numpy(rng.random((64, FEATURE_DIM), dtype=np.float32))
        y = torch.from_numpy(rng.random((64, 1), dtype=np.float32))
        loss = torch.nn.BCEWithLogitsLoss()(model.logits(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def test_all_norms_build_and_run():
    for norm in NORMS:
        model = ValueNet(norm=norm).eval()
        out = model(torch.zeros(4, FEATURE_DIM))
        assert out.shape == (4, 1), f"{norm}: bad shape {out.shape}"
        assert torch.all((out >= 0) & (out <= 1)), f"{norm}: output outside [0,1]"
    print("  ok  all norms build, shape and [0,1] range correct")


def test_rejects_unknown_norm():
    try:
        ValueNet(norm="groupnorm")
    except ValueError:
        print("  ok  unknown norm rejected")
        return
    raise AssertionError("expected ValueError for an unknown norm")


def test_batch_independence():
    """A row's score must not depend on what else is in the batch.

    This is the deployment-critical property: MCTS scores states one at a time. If BatchNorm were
    ever left in training mode, a row's output would shift with its batch-mates and this fails.
    """
    for norm in NORMS:
        model = _trained_a_little(norm)
        rng = np.random.default_rng(7)
        x = torch.from_numpy(rng.random((16, FEATURE_DIM), dtype=np.float32))
        with torch.no_grad():
            batched = model(x)
            singly = torch.cat([model(x[i:i + 1]) for i in range(len(x))])
        diff = (batched - singly).abs().max().item()
        assert diff < 1e-6, f"{norm}: batch-dependent inference, max diff {diff:.2e}"
    print("  ok  inference is batch-independent for every norm")


def test_export_is_eval_mode():
    """`export_onnx` must put the model in eval mode itself.

    Guards a real footgun: exporting a training-mode BatchNorm with the batch-1 dummy input would
    bake in statistics computed from a single all-zeros row (variance 0), and the exported graph
    would be silently wrong rather than failing loudly.
    """
    from train import export_onnx

    model = _trained_a_little("batch")
    model.train()  # deliberately hand it a training-mode model
    with tempfile.TemporaryDirectory() as td:
        export_onnx(model, str(pathlib.Path(td) / "v.onnx"))
    assert not model.training, "export_onnx left the model in training mode"
    print("  ok  export_onnx forces eval mode")


def test_onnx_matches_torch():
    try:
        import onnxruntime as ort
    except ImportError:
        print("  skip onnxruntime not installed")
        return
    from train import export_onnx

    rng = np.random.default_rng(3)
    x = rng.random((32, FEATURE_DIM), dtype=np.float32)
    for norm in NORMS:
        model = _trained_a_little(norm)
        with tempfile.TemporaryDirectory() as td:
            path = str(pathlib.Path(td) / f"{norm}.onnx")
            export_onnx(model, path)
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            got = sess.run(["win_prob"], {"features": x})[0]
        with torch.no_grad():
            want = model(torch.from_numpy(x)).numpy()
        diff = float(np.abs(got - want).max())
        assert diff < TOL, f"{norm}: onnx vs torch max abs diff {diff:.2e}"

        # Batch-1 through ONNX too: the dynamic axis plus a folded BatchNorm is exactly where an
        # export bug would hide, and batch-1 is the only shape the plugin ever uses.
        with tempfile.TemporaryDirectory() as td:
            path = str(pathlib.Path(td) / f"{norm}1.onnx")
            export_onnx(model, path)
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            one = np.concatenate([sess.run(["win_prob"], {"features": x[i:i + 1]})[0]
                                  for i in range(len(x))])
        diff1 = float(np.abs(one - want).max())
        assert diff1 < TOL, f"{norm}: onnx batch-1 vs torch max abs diff {diff1:.2e}"
        print(f"  ok  {norm:5s} onnx matches torch (batched {diff:.1e}, batch-1 {diff1:.1e})")


def main():
    print("ValueNet normalization tests")
    test_all_norms_build_and_run()
    test_rejects_unknown_norm()
    test_batch_independence()
    test_export_is_eval_mode()
    test_onnx_matches_torch()
    print("all passed")


if __name__ == "__main__":
    main()
