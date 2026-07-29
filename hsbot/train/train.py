"""Train the value net on SabberStone self-play JSONL, export to ONNX.

    python train.py --data ../trainer/data/selfplay.jsonl \
                    --out ~/.hs-script/models/value_net.v1.onnx

Each JSONL row is {"Label": 0|1, "Features": [144 floats]} (as written by the C# generator;
key casing is accepted either way). With --smoke N it fabricates random data to exercise the
train+export pipeline end-to-end without real data.
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ValueNet, FEATURE_DIM

DEFAULT_OUT = os.path.expanduser("~/.hs-script/models/value_net.v1.onnx")


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jsonl(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = {k.lower(): v for k, v in json.loads(line).items()}
            feats = obj["features"]
            if len(feats) != FEATURE_DIM:
                raise ValueError(f"row has {len(feats)} features, expected {FEATURE_DIM}")
            xs.append(feats)
            ys.append(obj["label"])
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32).reshape(-1, 1)
    return x, y


def make_smoke(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((n, FEATURE_DIM), dtype=np.float32)
    # A learnable signal: label correlates with (my hero hp - opp hero hp) = feat[0]-feat[15].
    logit = 6.0 * (x[:, 0] - x[:, 15])
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32).reshape(-1, 1)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--smoke", type=int, default=0, help="train on N synthetic rows instead of --data")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                    help="compute device; 'auto' picks cuda > mps > cpu")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")

    if args.smoke > 0:
        x, y = make_smoke(args.smoke)
        print(f"[smoke] synthetic rows: {len(x)}")
    else:
        if not args.data:
            ap.error("provide --data <jsonl> or --smoke N")
        x, y = load_jsonl(pathlib.Path(args.data))
        print(f"loaded {len(x)} rows from {args.data}  (win rate {y.mean():.3f})")

    # split
    n_val = max(1, int(len(x) * args.val_split))
    perm = np.random.default_rng(0).permutation(len(x))
    vi, ti = perm[:n_val], perm[n_val:]
    xt, yt = torch.from_numpy(x[ti]), torch.from_numpy(y[ti])
    xv, yv = torch.from_numpy(x[vi]).to(device), torch.from_numpy(y[vi]).to(device)

    loader = DataLoader(TensorDataset(xt, yt), batch_size=args.batch, shuffle=True)
    model = ValueNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model.logits(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vlogits = model.logits(xv)
            vloss = loss_fn(vlogits, yv).item()
            vacc = ((torch.sigmoid(vlogits) > 0.5).float() == yv).float().mean().item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  val_loss {vloss:.4f}  val_acc {vacc:.3f}")

    assert best_state is not None
    model.load_state_dict(best_state)
    print(f"best val_loss {best_val:.4f}")

    export_onnx(model, args.out)


def export_onnx(model: ValueNet, out_path: str) -> None:
    model.to("cpu").eval()  # export + verify on CPU for portability
    out = pathlib.Path(os.path.expanduser(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["features"], output_names=["win_prob"],
        dynamic_axes={"features": {0: "batch"}, "win_prob": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {out}")
    verify_onnx(model, str(out))


def verify_onnx(model: ValueNet, path: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping export verification")
        return
    rng = np.random.default_rng(1)
    sample = rng.random((4, FEATURE_DIM), dtype=np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample)).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"features": sample})[0]
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"onnx vs torch max abs diff: {max_diff:.2e}  (range {onnx_out.min():.3f}..{onnx_out.max():.3f})")
    assert max_diff < 1e-5, "ONNX export diverges from torch"


if __name__ == "__main__":
    main()
