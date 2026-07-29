# hsbot

A Hearthstone bot: **strategy trained offline in SabberStone, run live via Hearthstone-Script.**
The live engine does all the game interaction and search; we only inject a learned
board-evaluation function (an ONNX value net) in place of HS-Script's hand-tuned heuristic.

See `docs/FEATURES.md` for the state representation both sides share, and the project memory
`hs-bot-architecture` for why the pieces are split this way.

## Layout

```
hsbot/
  docs/FEATURES.md     # the shared 144-float feature contract (source of truth)
  fixtures/            # parity cases: both extractors must reproduce these vectors
  hs-plugin/           # Kotlin — the live HS-Script strategy plugin (DONE: skeleton)
    src/main/kotlin/club/xiaojiawei/hsbot/
      HsBotPlugin.kt              # plugin entry (StrategyPlugin)
      ValueNetStrategyDeck.kt     # MCTSDeckStrategy — supplies the ScoreCalculator
      OnnxValueNet.kt             # ONNX-backed ScoreCalculator (Function<War,Double>)
      WarFeatureExtractor.kt      # runtime half of the feature contract
    src/test/kotlin/club/xiaojiawei/hsbot/
      TestWarBuilder.kt           # builds a War from a neutral fixture state
      FeatureParityTest.kt        # asserts extractor == fixtures within 1e-6
    src/main/resources/META-INF/services/   # plugin + strategy registration
  train/               # Python — value net + ONNX export                (DONE: scaffold)
    reference_features.py         # independent ground-truth extractor + fixture generator
    model.py                      # ValueNet (144→256→128→1, sigmoid win-prob head)
    train.py                      # JSONL → train (BCE) → export+verify ONNX
  trainer/             # C#  — SabberStone self-play + feature/label dumper (DONE: scaffold)
    SabberStoneGen/               # console: ObservableState, FeatureExtractor,
                                  #   SabberStoneObserver, SelfPlayGenerator, Program
    SabberStoneGen.Tests/         # xUnit parity test vs fixtures
```

## Build (plugin)

Verified building + testing on **JDK 25+** (the HS-Script SDK modules target Java 25) with the
parent POM `com.github.xjw580.Hearthstone-Script:hs-script:v4.16.3-GA` resolved from jitpack.
The SDK modules themselves are **not** on jitpack, so build them from source into `~/.m2`
first (a jitpack `settings.xml` is needed so the parent resolves):

```
for m in hs-script-base hs-script-plugin-sdk hs-script-card-sdk hs-script-strategy-sdk; do
  git clone --depth 1 https://github.com/xjw580/$m.git vendor/$m
  mvn -s settings.xml -f vendor/$m/pom.xml install -DskipTests
done
mvn -s settings.xml -f hsbot/hs-plugin/pom.xml test   # → parity test 3/3
```

Drop the trained model at `~/.hs-script/models/value_net.v1.onnx`. If it's missing or fails
to load, the strategy falls back to HS-Script's built-in heuristic so the bot still runs.

## Status

- [x] Feature contract v1 (`docs/FEATURES.md`), accessors verified in both engines
- [x] Kotlin plugin skeleton (strategy + ONNX ScoreCalculator + feature extractor)
- [x] Parity fixtures (`fixtures/*.json`, 3) + Python reference + Kotlin parity test
- [x] C# SabberStone self-play generator + extractor + xUnit parity test
- [x] Python value net + training loop + ONNX export/verify
- [x] **All three build/test green locally** — Kotlin parity 3/3, C# parity 3/3, Python train+ONNX; C# self-play verified (11k rows @ ~470 games/s), real-data train val_acc 0.67
- [ ] Real deck code wired into `ValueNetStrategyDeck.deckCode()`
- [ ] Scale data + stronger policy → train real model → drop `value_net.v1.onnx` → run live on Windows

> Toolchains: **.NET 8** (SabberStone won't compile on .NET 10), **JDK 25+** for the plugin,
> Python 3.9+ with `torch`. See per-stage READMEs.

## Pipeline (end to end)

```
SabberStone self-play (C#)  --JSONL-->  train.py  --value_net.v1.onnx-->  HS-Script plugin (Kotlin)
   trainer/                              train/                            hs-plugin/
   observable-only features ----- same 144-float contract (docs/FEATURES.md) -----
   validated by:  FeatureParityTests.cs  |  reference_features.py  |  FeatureParityTest.kt
```

## Caveats

Automating the live client violates Blizzard's ToS and is bannable — use a throwaway
account. HS-Script is GPL-3.0 with a non-commercial clause; keep this repo's use private.
