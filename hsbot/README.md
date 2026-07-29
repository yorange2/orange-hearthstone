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
  train/               # Python — value net + ONNX export                     (TODO)
    reference_features.py         # independent ground-truth extractor + fixture generator
  trainer/             # C#  — SabberStone self-play + feature/label dumper (DONE: scaffold)
    SabberStoneGen/               # console: ObservableState, FeatureExtractor,
                                  #   SabberStoneObserver, SelfPlayGenerator, Program
    SabberStoneGen.Tests/         # xUnit parity test vs fixtures
```

## Build (plugin)

The plugin uses the same parent POM as the official template
(`com.github.xjw580.Hearthstone-Script:hs-script:v4.16.3-GA`, resolved via jitpack), so it
must be built either inside a checkout of Hearthstone-Script or with that parent available.
It is **skeleton code**: it compiles against the real SDK jars, not verified headless here —
a first `mvn -pl hs-plugin package` against the pinned SDK version is the next build step.

Drop the trained model at `~/.hs-script/models/value_net.v1.onnx`. If it's missing or fails
to load, the strategy falls back to HS-Script's built-in heuristic so the bot still runs.

## Status

- [x] Feature contract v1 (`docs/FEATURES.md`), accessors verified in both engines
- [x] Kotlin plugin skeleton (strategy + ONNX ScoreCalculator + feature extractor)
- [x] Parity fixtures (`fixtures/*.json`, 3) + Python reference + Kotlin parity test
- [x] C# SabberStone self-play generator + extractor + xUnit parity test
- [ ] Python training + ONNX export
- [ ] Real deck code wired into `ValueNetStrategyDeck.deckCode()`
- [ ] First real builds: `mvn -pl hs-plugin test` (Kotlin) and `dotnet test` (C#)

## Caveats

Automating the live client violates Blizzard's ToS and is bannable — use a throwaway
account. HS-Script is GPL-3.0 with a non-commercial clause; keep this repo's use private.
