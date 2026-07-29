# orange-hearthstone

A Hearthstone bot whose **strategy is trained offline in [SabberStone](https://github.com/HearthSim/SabberStone)** and **run live via [Hearthstone-Script](https://github.com/xjw580/Hearthstone-Script)**. The live engine handles all game interaction and search; we only inject a learned board-evaluation function (an ONNX value net) in place of its hand-tuned heuristic.

The project code lives under [`hsbot/`](hsbot/). See [`hsbot/README.md`](hsbot/README.md) for the full tour and [`hsbot/docs/FEATURES.md`](hsbot/docs/FEATURES.md) for the state contract.

## Why this split

Hearthstone-Script already enumerates legal actions, simulates them, searches (MCTS), and executes them on the live client — but its "simulator" is a within-your-turn heuristic optimizer, not a full-game engine, so it **can't self-play to generate training data**. SabberStone is a complete headless rules engine that can. So SabberStone is an *offline data factory* and Hearthstone-Script is the *live runtime*; the only thing that crosses between them is a trained ONNX model.

The single ML hook is Hearthstone-Script's `ScoreCalculator` (`Function<War, Double>`) fed into its built-in MCTS.

## The pipeline

```
SabberStone self-play (C#)  --JSONL-->  train.py  --value_net.v1.onnx-->  HS-Script plugin (Kotlin)
   hsbot/trainer/                       hsbot/train/                      hsbot/hs-plugin/
        └──────── one 144-float observable-only feature contract, validated in all 3 languages ───────┘
                  FeatureParityTests.cs  |  reference_features.py  |  FeatureParityTest.kt
```

## Run it

The upstream engines are gitignored; clone them into `vendor/` first:

```
git clone --depth 1 https://github.com/HearthSim/SabberStone.git vendor/SabberStone
```

Then, per stage (each has its own README):

```
# 1. generate self-play data (needs .NET 8)
cd hsbot/trainer && dotnet test && dotnet run -c Release --project SabberStoneGen -- 5000 data/selfplay.jsonl

# 2. train + export the model (needs Python + torch)
cd ../train && pip install -r requirements.txt && python train.py --data ../trainer/data/selfplay.jsonl

# 3. build the plugin (needs JDK 21 + Maven), drop value_net.v1.onnx at ~/.hs-script/models/,
#    load it in Hearthstone-Script on Windows
cd ../hs-plugin && mvn test && mvn package
```

## Status

Scaffold complete across all four stages; not yet run end-to-end (needs the toolchains + a Windows box for live play). See [`hsbot/README.md`](hsbot/README.md) for the checklist. CI (`.github/workflows/ci.yml`) runs the three parity suites + a training smoke test on push.

## Caveats

Automating the live client violates Blizzard's ToS and is bannable — use a throwaway account. Hearthstone-Script is GPL-3.0 with a non-commercial clause; keep this repo's use private. `vendor/` holds third-party code and is not tracked here.
