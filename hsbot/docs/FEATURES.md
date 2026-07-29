# Feature Contract v1 — the shared observable state vector

This is the **single source of truth** for the state representation the value network
consumes. It is implemented **twice** and the two implementations must produce
**byte-identical vectors** for the same logical board:

- **C# (offline)** — extracted from SabberStone `POGame` during self-play → training rows.
- **Kotlin (live)** — extracted from HS-Script `War` inside the `ScoreCalculator`
  (`Function<War, Double>`) the MCTS calls at each leaf.

If the two drift, the model trains on one distribution and runs on another. A parity
test (see [§6](#6-parity-contract)) guards against this.

---

## 1. What the vector represents

The network is a **value function** `V(state) → win probability ∈ [0,1]`, evaluating a
board **from the perspective of the player who just acted** ("me"). This is exactly the
quantity HS-Script's MCTS needs at each leaf: it enumerates *my* action sequences,
simulates them, and scores the resulting board. We replace the hand-tuned
`WarScoreCalculatorBuilder` with `V`.

- **Perspective:** always "me" = the active player being optimized. Opponent = "rival".
- **Symmetry:** the vector is *not* symmetric; `me` and `opp` blocks are distinct so the
  net can learn "my board good / their board bad".

## 2. Observability rule (hard constraint)

At runtime, HS-Script's `War` only knows **observable** information. The opponent's hand
and deck *contents* are hidden. Therefore:

- **Never** encode opponent hand-card identities or opponent deck-card identities.
- Opponent hand/deck appear **only as counts**.
- During SabberStone extraction we have full information, but we **mask** it to match:
  read opponent hand/deck as counts only.

The value net evaluates *my* end-of-turn board, which is fully observable, so this is a
clean fit — but the rule must be enforced on the C# side or the model learns to cheat.

## 3. Capture point (must be identical on both sides)

Record the state at **the end of my main actions, before end-of-turn triggers fire.**
This matches the leaf HS-Script's MCTS scores (`MonteCarloTreeNode.State.score` on a node
whose remaining actions are exhausted / `TurnOver`).

- SabberStone: snapshot when the self-play agent has chosen `EndTurnTask` but *before*
  `game.Process(EndTurnTask)` resolves end-turn triggers.
- HS-Script: the `War` handed to `scoreCalculator` at an MCTS leaf.

## 4. Vector layout (v1 — fixed length 144)

All values are `float32`, in the fixed order below. Indices are 0-based.

### 4A. Global scalars — 32 features

For each player block `P` in order **[me, opp]**, 15 features (`me` = 0–14, `opp` = 15–29):

> **Hero HP/armor parity note.** HS-Script folds armor into the hero pool
> (`bloodLimit = health + armor`, `blood() = bloodLimit − damage`), so `hero.blood()`
> is *effective remaining HP including armor*. SabberStone keeps `Hero.Health`
> (remaining, no armor) and `Hero.Armor` (remaining) separate. To avoid double-counting
> we use **effective HP** for #0 and **remaining armor** for #1, defined per engine below.

| # | Feature | Normalization | HS-Script (`War`) | SabberStone (`POGame`) |
|---|---------|---------------|-------------------|------------------------|
| 0 | hero effective HP (incl. armor) | `/30` | `P.playArea.hero.blood()` | `P.Hero.Health + P.Hero.Armor` |
| 1 | hero armor (remaining) | `/30` | `maxOf(hero.armor - hero.damage, 0)` | `P.Hero.Armor` |
| 2 | hand count | `/10` | `P.handArea.cards.size` | `P.HandZone.Count` |
| 3 | deck count | `/30` | `P.deckArea.cardSize()` | `P.DeckZone.Count` |
| 4 | board minion count | `/7` | `P.playArea.cards.size` | `P.BoardZone.Count` |
| 5 | board total attack | `/30` | `Σ card.atc` | `Σ m.AttackDamage` |
| 6 | board total health | `/40` | `Σ card.blood()` | `Σ m.Health` |
| 7 | taunt count | `/7` | `count isTaunt` | `count HasTaunt` |
| 8 | divine-shield count | `/7` | `count isDivineShield` | `count HasDivineShield` |
| 9 | secret count | `/5` | `P.secretArea.cards.size` | `P.SecretZone.Count` |
| 10 | weapon attack | `/10` | `P.playArea.weapon?.atc ?: 0` | `P.Hero.Weapon?.AttackDamage ?? 0` |
| 11 | weapon durability (remaining) | `/5` | `P.playArea.weapon?.blood() ?: 0` | `P.Hero.Weapon?.Durability ?? 0` |
| 12 | max mana this turn | `/10` | `P.resources` | `P.BaseMana` |
| 13 | overload locked | `/5` | `P.overloadLocked` | `P.OverloadLocked` |
| 14 | fatigue | `/10` | `P.fatigue` | `P.Hero.Fatigue` |

Then 2 turn-level features (indices 30–31):

| # | Feature | Normalization | HS-Script | SabberStone |
|---|---------|---------------|-----------|-------------|
| 30 | turn number | `/30` | `war.warTurn` | `game.Turn` |
| 31 | am I the first player | `0/1` | `war.me.gameId == war.firstPlayerGameId` | `P == game.FirstPlayer` |

All accessors verified in source: `Controller.cs` / `Character.cs` / `Hero.cs` /
`Weapon.cs` / `Minion.cs` / `Game.cs` (SabberStone) and `BaseCard.kt` / `Card.kt` /
`Player.kt` / `PlayArea.kt` (HS-Script). `card.damage` and `card.blood()` exist on
`BaseCard`/`Card`; `Weapon.Durability` returns *remaining* durability (`DURABILITY − DAMAGE`).

> **`game.Turn` semantics differ.** SabberStone increments `Turn` once per *player* turn
> (1,2,3,… alternating players); HS-Script `warTurn` may count per round. This feature is
> weak (`/30`) so the mismatch is minor, but normalize both to the same convention (player-
> turns) when wiring, or drop the feature if it can't be reconciled cleanly.

### 4B. Per-minion slots — 112 features

For each player block in order **[me, opp]**, **7 minion slots** (max board size), **8
features per slot**. Empty slots are zero-padded.

- me minions:  indices 32 … 87   (7 × 8)
- opp minions: indices 88 … 143  (7 × 8)

**Minion ordering:** board **left-to-right positional order** as both engines store it
(`playArea.cards` / `BoardZone` are already positional). Do not sort — position carries
meaning (adjacency buffs, attack order).

Per-slot 8 features:

| slot# | Feature | Normalization | HS-Script `Card` | SabberStone `Minion` |
|-------|---------|---------------|------------------|----------------------|
| 0 | attack | `/12` | `card.atc` | `m.AttackDamage` |
| 1 | health | `/12` | `card.blood()` | `m.Health` |
| 2 | can attack now | `0/1` | `card.canAttack()` | `m.CanAttack` |
| 3 | taunt | `0/1` | `card.isTaunt` | `m.HasTaunt` |
| 4 | divine shield | `0/1` | `card.isDivineShield` | `m.HasDivineShield` |
| 5 | windfury | `0/1` | `card.isWindFury` | `m.HasWindfury` |
| 6 | lifesteal | `0/1` | `card.isLifesteal` | `m.HasLifeSteal` |
| 7 | poisonous | `0/1` | `card.isPoisonous` | `m.Poisonous` |

**Card identity is intentionally excluded in v1** (stats only) so parity is trivial to
achieve and test. v2 adds a per-minion `cardId` embedding — and because HS-Script's
`card.cardId` and SabberStone's `Card.Id` are the **same HearthDb string IDs**
(e.g. `"EX1_561"`), the embedding vocabulary is shared with zero mapping work.

## 5. Label (training target)

Each recorded position (a "me end-of-turn" snapshot) is labeled with the **final game
result from that same player's perspective**:

- `1.0` if that player won the game, `0.0` if they lost.
- Optional discounting by distance-to-end (`γ^(turns_remaining)`) — off by default in v1.

The net outputs win-prob via a sigmoid. The `ScoreCalculator` returns this scalar
(or its logit) as a `Double`; MCTS maximizes it. A calibrated win-prob is strictly better
behaved than the heuristic's unbounded score.

## 6. Parity contract

`hsbot/fixtures/*.json` holds canonical cases:

```json
{
  "name": "two_taunts_vs_empty",
  "state": { /* engine-neutral board description, see fixtures/README */ },
  "expected": [/* 144 float32, the v1 vector */]
}
```

- The **C# extractor** builds a `POGame` matching `state`, extracts, asserts ≈ `expected`.
- The **Kotlin extractor** builds a `War` matching `state`, extracts, asserts ≈ `expected`.
- Tolerance: `1e-6` absolute. CI runs both; either failing blocks the build.

`expected` vectors are authored **by hand** for a handful of small boards first, then
frozen. Any change to this document bumps the version (§7) and regenerates fixtures.

## 7. Versioning

- The layout is versioned: **v1**. The feature count (144) and every index are frozen.
- Prepend the version as feature `-1`? No — keep the vector pure; store the version in the
  model filename (`value_net.v1.onnx`) and in `fixtures`. A model and an extractor must
  agree on version or refuse to run.

## 8. Open items before coding

1. ~~Verify accessors~~ — **done**, all confirmed in source (see §4A/§4B).
2. Decide sigmoid-output vs logit-output for `V` (affects MCTS score scale — either works
   since MCTS only compares). *Leaning sigmoid (win-prob) for interpretability.*
3. Confirm SabberStone snapshot hook fires *before* `EndTurnTask` resolves.
4. Reconcile `game.Turn` vs `warTurn` convention (see note in §4A) or drop feature #30.
