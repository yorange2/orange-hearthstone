# train — value net (Python) + ONNX export

Fits the value network on SabberStone self-play data and exports the ONNX model the live
HS-Script plugin loads (`OnnxValueNet`). This closes the pipeline:

```
SabberStone self-play (C#) --JSONL--> train.py --value_net.v1.onnx--> HS-Script plugin (Kotlin)
```

## Files

- `reference_features.py` — independent ground-truth feature extractor; also (re)generates
  `hsbot/fixtures/*.json`. Shared with no other language; the parity anchor.
- `model.py` — `ValueNet`: 144 → 256 → 128 → 1, sigmoid head (win probability).
- `train.py` — load JSONL, train (BCE), export ONNX, verify ONNX≈torch.

## Setup

```
cd hsbot/train
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Smoke test (no data needed — exercises train + export end to end):

```
python train.py --smoke 20000
```

Real training on generated data:

```
python train.py --data ../trainer/data/selfplay.jsonl \
                --out ~/.hs-script/models/value_net.v1.onnx --epochs 30
```

The exported model takes `features` `[batch, 144]` float32 and outputs `win_prob`
`[batch, 1]` — matching what `OnnxValueNet` feeds and reads. Drop it at
`~/.hs-script/models/value_net.v1.onnx`; if absent, the plugin falls back to the built-in
heuristic.

## Notes

- **Device:** auto-selects `cuda` > `mps` (Apple GPU) > `cpu`; override with `--device cpu`.
  ONNX export/verify always runs on CPU. For the current tiny MLP + data scale CPU is fine
  (GPU overhead can make it slower); GPU pays off once data/model grow.
- Row schema: `{"Label": 0|1, "Features": [144]}` (key casing accepted either way).
- The model is only as good as the data: current self-play uses a random behavior policy and
  random decks. For a deployable model, regenerate with the real deck and a stronger policy.
- Feature parity is enforced elsewhere (`FeatureParityTest.kt`, `FeatureParityTests.cs`,
  and this dir's `reference_features.py`) — the net assumes all three agree.
