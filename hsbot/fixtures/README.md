# Parity fixtures

Each `*.json` file pins one board to its expected v1 feature vector. Both extractors —
Kotlin `WarFeatureExtractor` (over `War`) and C# `POGameFeatureExtractor` (over `POGame`) —
must reproduce `expected` within `1e-6`. Either failing blocks the build.

## Format

```json
{
  "name": "two_taunts_vs_empty",
  "version": "v1",
  "state": {
    "turn": 5,
    "meIsFirst": true,
    "me":  { "hero": {...}, "mana": {...}, "hand": N, "deck": N,
             "board": [ {minion}, ... ], "secrets": N, "weapon": {...}|null },
    "opp": { ...same shape; opp hand/deck are counts only... }
  },
  "expected": [ /* 144 floats */ ]
}
```

`state` is an **engine-neutral** board description. Each side's test harness constructs the
matching engine object (a `War` in Kotlin, a `POGame` in C#) from it, runs the extractor,
and asserts against `expected`.

## Authoring

1. Start with 3–4 tiny hand-authored boards (empty boards, one taunt, a lethal setup, a
   board with divine shield + weapon). Compute `expected` by hand from `docs/FEATURES.md`.
2. Freeze them. From then on, `expected` changes only when the contract version bumps.

## Minion object fields (map to the 8 per-slot features)

`attack, health, canAttack, taunt, divineShield, windfury, lifesteal, poisonous`

## Hero / mana / weapon fields

- hero: `effectiveHp` (health+armor), `armor` (remaining)
- mana: `max` (crystals this turn), `overload`
- misc per side: `fatigue`, `secrets` (count)
- weapon: `attack`, `durability` (remaining) — or `null`
