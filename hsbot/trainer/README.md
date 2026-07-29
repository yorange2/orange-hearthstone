# trainer — SabberStone self-play data generator (C#)

Generates `(features, win/loss)` rows for the value net by self-play in SabberStone, then
labels each snapshot by the game's eventual winner. This is the offline data factory; it
never runs on the live path.

## Projects

- `SabberStoneGen/` — console app.
  - `ObservableState.cs` — engine-neutral DTO = the feature contract's data shape.
  - `FeatureExtractor.cs` — contract v1 over `ObservableState` (mirrors Kotlin/Python).
  - `SabberStoneObserver.cs` — live `Game` → `ObservableState` (current player's view,
    observable-only).
  - `SelfPlayGenerator.cs` — self-play loop; snapshots at end-of-turn (before EndTurnTask),
    labels by winner; writes JSONL.
  - `Program.cs` — CLI.
- `SabberStoneGen.Tests/` — xUnit parity test against `hsbot/fixtures/*.json` (1e-6).

## Prerequisite

Needs the SabberStone source at `vendor/SabberStone` (gitignored upstream clone):

```
git clone --depth 1 https://github.com/HearthSim/SabberStone.git vendor/SabberStone
```

## Run

Use the **.NET 8 SDK** — SabberStone's source does not compile under the .NET 10 SDK
(a `Span`→`IList` overload-resolution change in their `SpecificTask.cs`).

```
cd hsbot/trainer
dotnet test SabberStoneGen.Tests/SabberStoneGen.Tests.csproj   # parity test — do this first (3/3)
dotnet run -c Release --project SabberStoneGen -- 5000 data/selfplay.jsonl 1
#                                                  ^games ^out              ^seed
```

Output: one JSON object per line, `{ "Label": 0|1, "Features": [144 floats] }`, consumed by
`hsbot/train/` (Python) to fit the value net and export `value_net.v1.onnx`.

## Notes / next

- Behavior policy is uniform-random (simplest valid source). Upgrade to a greedy heuristic
  or epsilon-greedy over the value net for stronger data.
- Decks use `FillDecks=true` (random). For the *deployable* model, generate with the actual
  deck the live bot plays (must match `ValueNetStrategyDeck.deckCode()`).
- `game.Turn` counts per player-turn; reconcile with HS-Script `warTurn` or drop feature #30
  (see FEATURES.md §4A).
