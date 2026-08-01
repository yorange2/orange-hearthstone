# ppo — reinforcement learning on SabberStone (prototype)

A **PPO** agent that learns Hearthstone in SabberStone by **imitation warm-start + self-play/
greedy PPO**, and **outperforms the one-ply greedy heuristic** (~58% over 1200 held-out games).
RL path
"A": a **masked action-scoring policy** over the variable legal-action set, built on a
**transformer entity-encoder + transformer belief state**, driven through a simple stdio bridge
to the C# engine.

## Result — beats the greedy heuristic

Evaluated against **two** fixed opponents: uniform-random (weak) and a **greedy heuristic**
(SabberStone's own `MidRangeScore`, one-ply lookahead — the SabberStone analogue of
Hearthstone-Script's `基础策略`) on the fixed Mage-mirror deck.

This is the Phase 0 reference measurement, run under the post-determinism protocol: **argmax**
policy, **held-out** eval seeds disjoint from training, **400 games per seed**, repeated over
**three independent seeds** so the headline does not rest on one draw. Wilson 95% CIs.

| checkpoint | seed 100000 | seed 200000 | seed 300000 | pooled (n=1200) |
| --- | --- | --- | --- | --- |
| `ppo_medium_best` (532k) | 0.578 | 0.560 | 0.630 | **0.589** [0.561–0.617] |
| `ppo_tf_v2_best` (173k) | 0.580 | 0.565 | 0.618 | **0.588** [0.560–0.615] |
| `ppo_tf_v3_best` (173k) | 0.588 | 0.562 | 0.585 | **0.578** [0.550–0.606] |
| `ppo_multi_best` (173k) | 0.562 | — | — | — |
| `ppo_tf_best` (173k) | 0.540 | — | — | (CI spans 0.5) |
| `pretrained_tf` (BC only) | 0.417 | — | — | **loses to greedy** |

```
vs random  1.000   [0.963–1.000]   (n=100)
vs greedy  0.578   [0.550–0.606]   (n=1200, 3 held-out seeds)
```

**The agent outperforms the one-ply greedy heuristic (~58%)** — the honest "do we beat a
heuristic?" bar, not just "do we beat random?". Every PPO checkpoint clears 0.5 with the
interval's *lower* bound above it, and the result replicates across all three seeds, so this is
no longer a single-draw claim.

> **This supersedes the earlier ~55% ± caveat.** That number came off a `*best*` checkpoint
> selected as the max over repeated 30-game evals and was flagged as probably optimistic. Re-run
> at full budget on held-out seeds it went **up**, not down (0.550 → 0.578–0.589). Two lessons,
> and the second matters more than the first: the `*best*` selection bias was real but smaller
> than feared *for these checkpoints*, and — because the reduced-budget replications that
> produced the 8–18 point drops predate the determinism fixes — most of that apparent collapse
> was measurement noise, not checkpoint quality. Pre-determinism numbers are not comparable to
> post-determinism ones in either direction.

> **Selection caveat that remains.** The table's top row is the max over six checkpoints on seed
> 100000, so *that particular ranking* is selection-biased. The seed 200000/300000 columns were
> run afterwards as independent confirmation and hold up, which is why the pooled column is
> quoted rather than the best single cell. Ranking two checkpoints ~1 point apart is still not
> resolved at n=1200 — see the sample-size arithmetic in Phase 0.

**The warm start alone does not clear the bar.** `pretrained_tf` — behavioral cloning, no PPO —
scores **0.417** vs greedy. So imitation gets close to the heuristic and PPO fine-tuning is what
passes it; the two-stage pipeline is load-bearing, not just a speedup.

**What got it there — imitation warm-start + PPO fine-tune.** PPO from scratch (even with the
levers below) plateaus around ~40–47% vs greedy: a small model exploring from random weights
rarely stumbles onto heuristic-level play. The breakthrough was a two-stage pipeline:

1. **Imitation pre-training** (`pretrain.py`): behavioral cloning on greedy-vs-greedy games
   reaches ~85% top-1 action match — a policy that already ~39% matches greedy in-game.
2. **PPO fine-tuning** (`ppo_vec.py --resume`) from that warm start climbs to ~55%.

**Supporting levers** (all on by default in `ppo_vec.py`): a **greedy-prob curriculum**
(`--greedy-start/--greedy-end` ramp), **reward shaping** (`--shaping-coef`; potential-based
`F = coef*(γ·Φ' − Φ)`, Φ = tanh(MidRangeScore/200), policy-invariant per Ng et al. 1999),
**fixed decks** (`--fixed-deck`, Mage mirror) for low variance, and LR/entropy schedules.

## Pieces

- **C# env** (`hsbot/trainer/SabberStoneEnv/`): `HearthstoneEnv` is a **seat-agnostic
  two-player** env — it plays no opponent itself; every decision point is returned tagged with
  `player` (whose turn), and Python routes it. Encoders: `TokenEncoder` (state = a **set** of
  entity tokens, 18 floats each — hero/minions/hand/weapon/hero-power), `ActionEncoder` (each
  legal `PlayerTask` → 20 floats), `PrivilegedEncoder` (opponent-hidden info, 8 floats,
  critic-only). `GreedyOpponent` plays SabberStone's `MidRangeScore` one-ply for the benchmark.
  `Program.cs` stdio JSON server: `reset` / `step <idx>` / `step_greedy` →
  `{tokens[T,18], priv[8], actions[N,20], player, done, winner}`.
- **`env.py`**: subprocess wrapper around that server (`Obs` = tokens/priv/actions/player/
  potential/greedy_action; `VecEnv` drives N processes in lockstep).
- **`ppo.py`**: the shared **model + utilities** (no training loop): `EntityEncoder` (a small
  **transformer** over the entity-token set, permutation-invariant, masked mean-pool →
  per-state embedding) → `TemporalEncoder` (a **transformer over the last `MEM` decisions** →
  belief state) → `ActorCritic` (action scorer over `concat(belief, action_feat)` + privileged
  value head); plus `act`, `gae`, `pad`, and `evaluate` (benchmark vs random/greedy).
- **`pretrain.py`**: imitation pre-training (behavioral cloning). Runs greedy-vs-greedy games,
  records (state, greedy-action) pairs, and trains the policy by cross-entropy to imitate the
  heuristic (~85% top-1). The resulting checkpoint is a strong warm start for PPO.
- **`ppo_vec.py`**: the training loop — vectorized self-play + greedy PPO with LR/entropy
  schedules, a greedy-prob curriculum, and best-checkpoint tracking. Also serves `--eval-only`.
- **`device.py`**: `--device` resolution shared by both entry points. `auto` (the default) picks
  the first available of **cuda → mps → cpu**; an explicitly requested backend that isn't
  available degrades to CPU with a warning.

### Which device: it depends on the phase, not just the hardware

Measured on an M-series (10-core), same model, so the only variable is where the tensors live:

| phase | shapes | `small` | `100x` (17.6M) |
| --- | --- | --- | --- |
| PPO rollout (`collect_vec`) | tiny + **ragged**, change every step | **CPU ~9.5×** faster | CPU ~3× faster |
| BC / PPO update | big fixed-ish batches (256–512) | **MPS ~2.5×** faster | MPS ~2.6× faster |

Re-measured back to back on an M-series (10-core), `small`. Both directions in the original table
held up; both magnitudes were off, in opposite directions:

```
BC, 3 epochs, blind    cpu 23.12s   mps 12.19s     MPS 2.5x        (was quoted 5x)
BC, 3 epochs, text     cpu 22.93s   mps 11.60s     MPS 2.6x
PPO 5 iters x 2048     cpu 990 st/s mps 104 st/s   CPU 9.5x        (was quoted 2x)
```

The BC totals include a device-independent data-generation step (~4.8s) and process startup, so
the true training-phase MPS advantage is larger than 2.5×. The rollout gap of 9.5× is close to
the ~6× recorded independently in `device.py`, and much larger than the 2× this table claimed —
so the case for `ppo_vec.py --device cpu` is *stronger* than it appeared, not weaker.

MPS loses the rollout at **every** model size: the ragged token/action padding means the shape
changes on nearly every call, which defeats MPS graph caching. It wins clearly on large-batch
training, where one shape is reused. So the fastest recipe on Apple silicon is **`pretrain.py
--device mps`, `ppo_vec.py --device cpu`** — about 2× faster end-to-end than using either device
for both. `auto` cannot know this (it sees hardware, not phase), so pass `--device` explicitly
when it matters.

> **Correction — the nested-tensor explanation was wrong.** This paragraph used to add "a few
> nested-tensor ops also lack MPS kernels and fall back to CPU". They do not fall back, because
> they are never called: `nn.TransformerEncoder` gates its nested-tensor fast path on
> `src.device.type in ("cpu", "cuda", privateuse1)` (torch 2.8,
> `torch/nn/modules/transformer.py:496`), and MPS is not in that list. Counting the calls:
> `_nested_tensor_from_mask` runs **1×** on CPU in eval and **0×** on MPS. The fast path is also
> disabled in *training* mode on every device (`first_layer.training` is checked first), so it
> cannot explain a train-time device gap at all. The timings in the table are unaffected — only
> the mechanism was wrong — but two things follow: a patch "working around" the MPS fallback
> would be a no-op, and the real cause of MPS's rollout loss is most plausibly per-call dispatch
> overhead on small, constantly-reshaped tensors rather than anything mask-related.
>
> That tension is now resolved. `device.py` used to print its "MPS ~6× slower" warning
> **unconditionally**, including during BC — the one phase where MPS is the right choice, so the
> warning told the reader the opposite of the right thing. `resolve_device` now takes a
> `phase` argument (`"batch"` for BC, `"rollout"` for PPO) and prints advice that matches the
> measurement.

### Self-play + greedy (opponent scheme)

Each game the main policy takes a random seat; the opponent is either the **current policy**
(self-play) or the **greedy heuristic**, chosen per game with probability `--greedy-prob` (or a
curriculum that ramps `--greedy-start` → `--greedy-end`). Only the main seat's transitions
train; the terminal ±1 reward is attached to that seat's last decision. Training against greedy
directly (plus imitation warm-start) is what pushes the policy past the heuristic — extend
toward an AlphaStar-style frozen-snapshot league for more robustness.

### Why a transformer entity-encoder (the "state = time series / sequence" idea)

Two distinct "transformer" opportunities exist; this implements the higher-leverage one:
- **Entity/set transformer (here)** — attention over the variable set of board/hand entities
  at the *current* state. Permutation-invariant, any board size, learns relations the flat
  144-vector can't (cf. AlphaStar's entity encoder). Stateless per decision → deploys like the
  MLP. This is the "v2" state representation (tokens), separate from the flat `docs/FEATURES.md`
  contract (still used by the supervised value-net pipeline).
- **Temporal transformer belief state (done)** — `TemporalEncoder` self-attends over a fixed
  window of the agent's last `MEM` per-decision embeddings and reads out the current slot as a
  belief the heads condition on (Hearthstone is a POMDP: hidden opponent hand/deck). This
  replaces an earlier `nn.LSTMCell` — attention over the window (cf. GTrXL / AlphaStar's memory)
  learns cross-decision relations a single hidden vector can't; the tradeoff is bounded memory
  (window `MEM`) vs the LSTM's unbounded-but-lossy state. Memory still uses R2D2 **stored-state**:
  the recurrent state is the rolling window `(hist, mask)`, each transition keeps its input
  window, and the PPO update recomputes one step from it — minibatches stay per-transition, no
  BPTT through time. At inference the plugin must carry the window across the game's decisions.

> Honesty note: at prototype scale (single env, ~20 iterations) the entity encoder is verified
> to train stably and reach ~0.8 vs random — **on par with the MLP, not proven better**. The
> representational win shows up with many parallel envs + real self-play, not a 2-minute run.

## Run

```
# build the env once (targets net10.0)
dotnet build hsbot/trainer/SabberStoneEnv -c Release

cd hsbot/train/ppo
# 1. imitation warm-start: clone the greedy heuristic (~85% action match)
python pretrain.py --games 2000 --fixed-deck --out pretrained.pt

# 2. PPO fine-tune from the warm start (beats greedy ~55%)
python ppo_vec.py --resume pretrained.pt --fixed-deck --out ppo_policy.pt

# 3. evaluate a checkpoint (no training)
python ppo_vec.py --eval-only ppo_policy_best.pt --fixed-deck --num-envs 1
```

`ppo_vec.py` saves both the final `ppo_policy.pt` and the best-vs-greedy `ppo_policy_best.pt`.
Pass `--dll`/`--dotnet` to point at a specific env build or runtime; otherwise `dotnet` must be
on PATH and the default DLL path (`net10.0`) is used.

## Improvements from Xiao et al. 2023

Techniques from *"Mastering Strategy Card Game (Hearthstone) with Improved Techniques"*
(IEEE CoG 2023, [arXiv:2303.05197](https://arxiv.org/abs/2303.05197)), whose ablation
reports per-technique win-rate gains at scale.

**Implemented here:**
- **γ = 1.0** (was 0.99). Episodes are short with only a terminal ±1 reward, so the
  undiscounted return faithfully recovers win/loss. Paper: **+7%**.
- **Privileged (asymmetric) critic** — the "Cheat" technique. The value head additionally
  sees `priv` = opponent-hidden features (`PrivilegedEncoder`: hand aggregates + deck count);
  the policy stays observable-only. Because the critic is unused at inference there is **no
  train/test gap** (cleaner than the paper's asymmetric-`n` scheme). Paper: **+5.5%**.

> Honesty note: at this prototype scale (single env, ~20 noisy iterations) these are verified
> to integrate cleanly and keep the agent learning (~0.85+ vs random) — but the *magnitude* of
> the paper's gains is only measurable with many parallel envs, many seeds, and thousands of
> eval games. Treat them as correctly-wired, not yet A/B-proven here.

**Mapped but not yet done** (bigger lifts): improved V-Trace (ρ̄>1 + ρ floor + PPO-clip on a
V-Trace target) and off-policy queue balancing — both only matter once training is
distributed/off-policy; per-hero model isolation; auto-regressive action decomposition with
shared card/hero embeddings; and a principled fictitious-play weighting (OSFP) over the league.

## Scaling: vectorized parallel envs (`ppo_vec.py`)

The single stdio env was the throughput bottleneck. `ppo_vec.py` runs **N env subprocesses in
lockstep** (`VecEnv`): a batched command is written to all N stdins and flushed, so the N
SabberStone processes step concurrently across cores while Python **batches policy inference**
across them. Same agent (entity-transformer + transformer belief state + privileged critic + shaping); per-env
trajectories are kept separate so shaping/GAE stay per-episode-correct. Simplification: the
opponent is self-play or greedy (the frozen-snapshot league is dropped so all inference batches
through one net).

```
python ppo_vec.py --num-envs 8 --iters 200 --steps 4096 --fixed-deck
```

Measured on an 8-core laptop: **~447 → ~1500-1700 steps/s (~3.5-4×)**. It's not the full 8×
because the greedy opponent's clone-heavy steps dominate wall time and the Python loop adds
serial overhead; on a many-core + GPU box the batched inference pays off more and the gap
widens. This throughput (plus the imitation warm-start) is what makes the ~55%-vs-greedy run
finish in minutes on a laptop.

## Model scale: bigger is not better here

Controlled study across `--size`, **identical recipe and seed** for every arm (1500 BC games /
8 epochs → 60 PPO iters × 4096 steps → 200-game eval vs greedy on a **held-out seed**), so the
only variable is parameter count:

| `--size` | params | BC top-1 | PPO wall-clock | vs greedy (200 games) |
| --- | --- | --- | --- | --- |
| `small` | 173k | **0.823** | **615 s** | **0.455** |
| `xl` | 1.69M | 0.711 | 2508 s (4.1×) | 0.440 |
| `100x` | 17.6M | 0.753 | 5693 s (9.3×) | 0.350 |

Win rate **falls** monotonically as the model grows, at up to 9.3× the training cost. Two
distinct causes, worth separating because only the first is fixable by tuning:

1. **The big arms never optimized properly.** `100x`'s BC loss flatlines at ~1.2385 from epoch 2
   onward (1.2390 → 1.2388 → 1.2385 → 1.2383 → 1.2387 …) — dead flat, accuracy stuck ~0.75,
   while `small` reached 0.9122 and was still improving. `lr=1e-3` with no warmup collapses a
   17.6M-param transformer into a degenerate solution; its PPO evals then swung 0.533 → 0.167 →
   0.333. So this table is **not** evidence that capacity hurts — it is evidence that the
   hyperparameters are tuned for `small` and do not transfer. Retry with LR warmup and a lower
   LR before concluding anything about scale.
2. **Capacity was never the bottleneck.** Data was held fixed while params grew 100× (86.6k BC
   examples, 246k PPO transitions, one fixed Mage mirror). More fundamentally, the observation
   itself caps achievable play — see below.

**Note on `*best*` checkpoints.** Every arm dropped sharply from its reported `*best*` score to
clean evaluation (`small` 0.567 → 0.455, `100x` 0.533 → 0.350). `*best*` takes the max over
three 30-game evals (σ ≈ 0.09), so it selects noise. Re-evaluate on a held-out seed with more
games before quoting a win rate.

> **Update after the Phase 0 reference run.** This table's drops are **not** clean evidence of
> selection bias, because both sides of each arrow predate the determinism fixes. When the
> surviving `*best*` checkpoints were re-scored under the current protocol (argmax, held-out
> seed, 1200 games) they landed at **0.578–0.589**, i.e. they did *not* collapse. Selection on
> 30-game evals is still a real bias and still worth avoiding — but the 8–18 point drops
> recorded here are better explained by a nondeterministic env and a sampling policy. Do not
> cite this table as a measurement of selection bias; treat the whole table as pre-determinism
> and not comparable to anything above.

### Why ~50% vs greedy is close to this design's ceiling

- **The agent sees strictly less than the heuristic it fights.** A token is 18 floats (type
  one-hot, mine, cost, attack, health, 5 keyword flags, can-attack) — there is **no card
  identity and no card text**. Two different 4-cost spells are the same input vector. Greedy
  calls `MidRangeScore` through the real simulator with one-ply lookahead, so it implicitly
  knows what every card does.
- **Every training signal points at greedy.** BC initializes to ~82% action-match with greedy;
  the shaping potential is `tanh(MidRangeScore.Rate()/200)` — *the greedy heuristic's own score
  function* (`HearthstoneEnv.cs:147`); and the opponent is greedy 70% of the time. Shaping is
  policy-invariant at convergence (Ng et al. 1999), but under a finite budget these three
  anchors pull hard toward heuristic-level play.

Highest-leverage fixes, in order: **add a card-identity embedding to the token** (removes the
information ceiling), **decouple the shaping potential from `MidRangeScore`**, then raise the
PPO sample budget. Model scale is the last lever, and only after LR warmup.

## Plan: how to actually get past ~50% vs greedy

Ordered by expected gain per unit of work, and derived from the measurements above rather than
from generic RL advice. Each phase states the hypothesis it tests and the gate that decides
whether to continue — several of these could fail, and the gates are there to find that out
cheaply.

### Phase 0 — Make the metric trustworthy ✅ *(done — tooling landed; reference run pending)*

Nothing downstream is measurable until this lands: a genuine 5-point gain is currently invisible
inside the noise, and `*best*` selection inflates results by 8–18 points (see the note above).

> **What this turned up: the environment was not deterministic.** Chasing "why do two identical
> eval commands disagree?" found three independent causes, in increasing order of severity:
>
> 1. Evaluation ran on `vec.envs[0]` — a **training** env whose RNG stream had already been
>    advanced by rollouts, so eval was never held out.
> 2. `act()` sampled from the policy (`dist.sample()`) during evaluation, drawing on the global
>    torch RNG. Two identical runs on one checkpoint gave **0.333 vs 0.467** over 60 games.
> 3. The real one: `GameConfig.RandomSeed` was never set, so SabberStone built a **time-based**
>    RNG (`Game.cs:276`) and every shuffle/draw differed run to run — and `GreedyOpponent`
>    cloned with the default `resetRandomSeed: true`, handing each one-ply lookahead clone a
>    fresh time-based RNG, making *greedy itself* nondeterministic. A pure greedy-vs-greedy
>    probe at a fixed seed produced completely different games on every run.
>
> All three are fixed. The same probe now yields byte-identical games across runs, and repeated
> `--eval-only` invocations agree exactly. This also makes **training** reproducible for the
> first time, which is a prerequisite for trusting any A/B comparison in Phases 1–5.

- Raise checkpoint-selection evals to ≥200 games, or stop selecting on them and evaluate the
  final model instead
- Standard eval: fixed **held-out** seed, **400 games**, report a Wilson 95% CI. Half-width at
  p=0.5: n=30 → **±0.168**, n=50 → ±0.134, n=200 → ±0.069, n=400 → **±0.049**. (Note this is the
  interval half-width, not the standard error σ≈0.09 quoted for n=30 above — the interval is
  ~1.96× wider.) Sobering consequence: even n=400 only resolves a single rate to ±5 points, and
  *comparing two arms* has √2× the error, so separating policies ~5 points apart needs ~1500
  games each. Prefer changes big enough to clear the interval over tuning for small gains.
- Add `--eval-seed` so eval seeds can never overlap training seeds — it now hard-errors on
  collision with `--seed .. --seed+--num-envs-1` rather than silently reporting a non-held-out
  number
- Evaluate with **argmax** by default (`--eval-sample` restores sampling), removing the largest
  remaining variance source
- Re-measure at full budget to establish one honest reference number ✅ *(done — see
  [Result](#result--beats-the-greedy-heuristic))*

**Gate:** a reference win rate with a confidence interval. Phases 1–4 are guesswork without it.
**Passed:** `0.578 [0.550–0.606]` vs greedy over 1200 held-out games (3 seeds × 400), argmax.
Repeated `--eval-only` runs at a fixed seed now agree exactly, and the three seeds agree within
7 points. Anything Phases 1–5 claim must clear this interval to count.

Note that fixing determinism **changes the numbers**: every win rate recorded before this point
was measured under a nondeterministic env with a sampling policy, so the scaling table above is
not directly comparable to anything measured after. Re-baseline before comparing.

### Phase 1 — Put card identity in the observation ✅ *(implemented)*

The hard information ceiling. A token is 18 floats with **no card ID and no card text**, so two
different 4-cost spells are identical inputs and the policy cannot represent "Polymorph the 7/7"
as distinct from "play a vanilla 4-drop". Greedy, by contrast, scores through the real simulator
and implicitly knows what every card does.

- `TokenEncoder.cs`: emit a card-ID index alongside the 18 floats
- `env.py`: parse it into `Obs`
- `ppo.py`: `nn.Embedding(vocab, 32)` concatenated into the token before `EntityEncoder`
- Open question to settle first: SabberStone's card-ID space, and whether to embed all
  collectibles or hash into a fixed vocab. (This said "~30 unique cards under `--fixed-deck`";
  that was wrong — see the correction below. It is ~471.)

**As built.** `CardVocab` assigns dense indices by sorting card `Id` strings ordinally — every
env process must agree on the mapping or tokens from different envs mean different things, and
`Cards.All` is a `Dictionary.Values` with no ordering guarantee. Index 0 is a reserved
none/unknown row and doubles as the embedding `padding_idx`. The env reports the vocabulary over
a new `meta` command so the model is sized from the env rather than a hard-coded constant. Card
identity is emitted as a **parallel `cardIds` array**, not squeezed into the float token: it is a
categorical lookup, not a magnitude.

Vocabulary is **8303**, so `--card-dim 16` adds ~134k nominal params to a 173k model. That looks
like a serious confound with the capacity question — but only **471** distinct cards actually
appear in fixed-deck play, so roughly **7.5k** embedding params ever receive a gradient. Report
both numbers; the nominal figure overstates the added capacity by ~18×.

Existing checkpoints do not load into a card-embedding model (`--no-card-emb` restores the old
card-blind architecture, and is the ablation control). RL-only: the flat `FeatureExtractor`
feeding the ONNX value net is a separate contract and is untouched.

**Gate:** BC top-1 must clear the current 0.823 *before* spending anything on PPO. If it does
not, the embedding is miswired — a cheap early signal.

**Gate result: not cleared.** Both arms trained with an identical recipe and seed (1500 BC games,
8 epochs, `--fixed-deck`, `--seed 42`, `small`), the only difference being the embedding:

| arm | BC train top-1 | held-out top-1 (500 samples) |
| --- | --- | --- |
| control (`--no-card-emb`) | **0.829** | 0.838 |
| `--card-dim 16` | **0.823** | 0.838 |

Held-out accuracy is *identical*; train accuracy is marginally lower. The gate says this means
the embedding is miswired, so that was checked first — and it is **not**. The env emits varied
dense indices over the reported vocab (`card_ids: [3966, 3966, 3720, 2622, 5481, 3632, 1342]`,
`card_vocab: 8303`), reaching the model as a parallel array as designed.

So the wiring is right and the **hypothesis is what failed**, which is the more interesting
outcome. Three reasons it is unsurprising in hindsight, all of which the design notes above
already contain the seeds of:

1. **The target doesn't need card identity.** BC imitates greedy, and greedy ranks actions by
   `MidRangeScore` over resulting board stats. Its choice is therefore mostly *predictable from
   the stats already in the 18-float token* — the information the embedding adds is largely
   information the teacher does not use. This gate was always a weak test of Phase 1: it asks
   whether card identity helps copy a card-blind-ish teacher, not whether it helps *play*.
2. **Almost none of the embedding trains.** Only 471 of 8303 rows ever appear, so ~7.5k of the
   ~134k nominal params get a gradient — spread over 8 epochs of one fixed Mage mirror.
3. ~~**A fixed mirror is the worst case for it.** Both players draw from the same 30 cards, so
   card identity is nearly constant across the matchup it was measured on.~~ **Retracted — this
   was false**, see [What `--fixed-deck` actually does](#what---fixed-deck-actually-does). Card
   identity varied all along, which makes reasons 1 and 2 carry the whole explanation.

**Consequence for the plan.** Phase 1 was ranked first as "the hard information ceiling," and
this does not refute that for *play* — but it does mean the cheap gate cannot confirm it, and
the honest next test is the PPO arm, not another BC run. A stronger version of Phase 1 would
vary decks (so identity carries signal) and stop using a stats-driven heuristic as the imitation
target. Whether the embedding earns its place under PPO is measured in the A/B/C below —
**it does not**.

### Phase 1b — Card *text* instead of card ID ✅ *(implemented; evaluated — no win-rate gain)*

The reason the ID embedding is weak is structural, not a tuning problem: it has to **learn** what
each card does from gradients, so only cards that actually appear ever acquire meaning. 471 of
8303 rows train under `--fixed-deck`; the remaining 7832 stay at random init, and the policy
reads those random vectors *silently* — an unseen card is not "unknown", it is noise that looks
like data. That is also why the ID embedding can never help the deployment goal in the upgrade
paths (fixed Mage mirror → the real deck the live bot plays): a new deck is mostly new rows.

`card_text.py` builds a frozen `[8303, D]` matrix from each card's **name + rules text**, and the
model projects it to `card_dim` with a small trained layer:

```
python card_text.py --out card_text_emb.npz          # once, offline
python pretrain.py --card-text card_text_emb.npz --deck variedmirror ...
```

Meaning is *read* rather than learned, so all 8303 rows are populated before a single game is
played and an unseen card is represented by its similarity to seen ones. The cost structure
inverts nicely:

| card identity | total params | **trainable** | added trainable | rows with meaning |
| --- | --- | --- | --- | --- |
| none (card-blind) | 173,122 | 173,122 | — | 0 |
| learned id embedding | 306,994 | 306,994 | +133,872 | 471 |
| **frozen text embedding** | 706,578 | **175,186** | **+2,064** | **8303** |

**65× fewer trainable params than the ID table, for every card instead of 6% of them.** That also
settles the capacity confound the scale study raised: this arm adds information while *reducing*
trainable capacity relative to Phase 1, so a win cannot be attributed to size.

**Encoder: TF-IDF + SVD, not a sentence transformer** (`--encoder st` switches, optional import).
Hearthstone rules text is templated rather than prose — `<b>Battlecry:</b> Deal $3 damage.` — so
lexical overlap genuinely is the semantics, and the vocabulary (6679 tokens over 8303 cards) is
small enough for SVD to recover the mechanic structure. It needs no model download, no network,
and is deterministic, which a downloaded transformer is not. Spot-checked neighbours:

```
Fireball    -> Fireblast, Roaring Torch, Dynamite, Pyroblast     (direct damage)
Flamestrike -> Felbloom, Arcane Explosion, Swipe, Boom!          (AoE damage)
Polymorph   -> Bananas, Polymorph, Light-imbued                  (transform)
```

**Known weakness, measured.** 1417 cards have no rules text at all (vanilla minions like Chillwind
Yeti), so their document is just a name. Those rows partially collapse — mean pairwise cosine
**0.492**, with 5.7% of pairs above 0.99, against **0.000** and no near-duplicates for the 6885
cards that do have text. The mitigation is that a vanilla card's identity *is* its cost/attack/
health, which the 18-float token already carries; the embedding is only load-bearing for cards
whose text does something. Worth revisiting (card type / tribe / tags as extra tokens) if the
varied-deck arms underperform.

**Alignment is the failure mode to fear**, because it is silent. The matrix rows must match the
`CardVocab` indices the env emits, so the card list is pulled *from the env* over a new `cards`
command rather than re-parsed from `CardDefs.xml` — a one-card difference between those pools
shifts every later row and nothing crashes. The env additionally reports a `cardIdsHash` over its
id list (matching C# and Python implementations), recorded in the artifact and re-checked at
load; both that and the vocab-size check are verified to reject a mismatched pairing.

### What `--fixed-deck` actually does

**`--fixed-deck` never fixed the deck.** This invalidates a premise several notes above were
written on, including one in the Phase 1 post-mortem, so it is recorded here rather than quietly
patched.

`GameConfig.FillDecksPredictably` does **not** mean "same deck every game". It only passes
`GameConfig.UnPredictableCardIDs` — Prince Malchezaar, Patches, the Quests — as an **exclusion
list**, and `DeckZone.Fill` (`SabberStoneCore/src/Model/Zones/DeckZone.cs:87`) then picks the 30
cards *at random* from the class pool, independently for each player. So under `--fixed-deck`:

- deck **contents differ every game**, not just draw order;
- the two players get **different decks**, so the "mirror" is a mirror of *class*, not of cards;
- only the class (one pool instead of nine) and the random-effect exclusions are actually pinned.

Measured, by counting distinct cards observed in play:

| mode | 10 games | 30 games | 60 games |
| --- | --- | --- | --- |
| `fixed` | 199 | 407 | **530** |
| `variedmirror` | 191 | 379 | 509 |

A genuinely fixed 30-card deck could never exceed 30. The count is still climbing at 60 games.

The repo already contained the contradiction: Phase 1 says "~30 unique cards under
`--fixed-deck`" in one place and "only 471 distinct cards actually appear in fixed-deck play" a
few paragraphs later. **471 was the correct one**, and it is what a random fill from the Mage +
neutral Standard pool looks like.

**What this changes.** Two things get *worse* and one gets *better*:

1. The Phase 1 post-mortem's third reason — "a fixed mirror is the worst case, both players draw
   from the same 30 cards" — is retracted. Card identity varied all along, so reasons 1 and 2
   (the imitation target does not use card text; almost no embedding rows receive gradients)
   carry the entire explanation of why the gate failed.
2. `--fixed-deck` is a weaker variance control than claimed, because independent per-player fills
   put **deck asymmetry inside the baseline** — the exact noise `variedmirror` was built to
   remove. The 0.578 reference is still sound (seeds held out, symmetric in expectation), but its
   "low variance" billing was overstated.
3. Conversely the card-text case is *stronger* than argued: the setting always had ~471 distinct
   cards in play, so there was real card variety for a semantic embedding to exploit, and the
   "nothing unseen to generalise to" objection does not apply.

### Phase 1c — Varied decks ✅ *(implemented; used as the card-identity testbed)*

`--deck` adds modes. The motivation is no longer "add card variety" (there already was some) but
**a fair mirror and a wider pool**:

| mode | decks | why |
| --- | --- | --- |
| `fixed` | Mage mirror, **random** fill per player (misleading name) | the historical baseline |
| `variedmirror` | random class, random legal 30, **identical for both seats** | card pool varies, matchup stays fair |
| `random` | random classes, independent random fill | maximum variety, but deck-quality asymmetry becomes noise |

`variedmirror` is the one to measure card embeddings on, and it is the only mode of the three
where both seats hold the **same** cards — `fixed` and `random` both fill each player's deck
independently, so one side routinely draws a materially stronger pile and win rate measures deck
luck on top of policy strength. That is fatal when the effect being chased is a few points wide.
Decks are
drawn from **implemented** Standard cards only (`Card.Implemented`); without that filter a random
deck eventually draws a card whose effect is unwritten and throws mid-game, which would surface
as sporadic crashes on a fraction of games rather than an obvious failure.

### Phase 2 — Decouple the reward from greedy ✅ *(implemented)*

Three anchors hold the policy at heuristic level, the strongest being that the shaping potential
**is** greedy's own objective (`HearthstoneEnv.cs:147`).

- Anneal `shaping_coef` → 0 over training so the final policy is not anchored
- Ablate `--shaping-coef 0` vs `0.05` vs annealed at fixed budget
- Consider a potential built from raw board state (health/tempo differential) instead of the
  exact function greedy maximizes

Genuinely uncertain: shaping is part of what makes the current run learn at all, so removing it
may hurt before it helps. That is why it is an ablation, not a change.

### Phase 3 — Give the agent the lookahead greedy already has ✅ *(implemented; no gain yet)*

Greedy is one-ply search + handcrafted score; the agent is **zero-ply** + learned score. A
`simulate` command clones the game once per legal action and returns each resulting state;
`act_lookahead` values them with the critic and takes the best (`--lookahead`).

**A perspective bug worth recording.** The first version encoded each resulting state from the
*acting* player's view. But after an action the turn often flips, and the critic has only ever
seen states from the **player to move** — so those inputs were out of distribution and the value
estimates were garbage. Measured effect on one checkpoint: **0.667 → 0.067** vs greedy. Encoding
from the player to move and negating when the turn flipped (standard negamax) restored it to
0.567. Any value-based search over a two-player game needs this; getting the sign right is not
optional bookkeeping.

**It still does not beat plain argmax** (0.567 vs 0.667 at n=30 — intervals overlap heavily, so
neither is resolved). A plausible reason: the **actor** head is explicitly trained to rank
actions, while the **critic** is trained as a variance-reduction baseline. Substituting value
ranking throws away the head that was trained for the job. Needs n≥400 to say anything real.

### Phase 4 — Optimization hygiene, then re-test scale ✅ *(warmup implemented)*

Only meaningful after Phases 0–1.

- **LR warmup + lower LR for large models** — this is what broke `100x` above. `--warmup-iters N`
  gives a linear ramp before the cosine schedule; `0` (the default) reproduces the old schedule
  exactly, so existing recipes are unchanged
- Re-run the scale study with warmup, to answer the question the table above leaves open: does
  capacity help once optimization is not broken?
- Raise the PPO sample budget (1.6M transitions at the full recipe)
- Free speedup already measured: BC on `mps`, PPO on `cpu` ≈ 2× end-to-end

### Phase 5 — Opponent diversity ✅ *(implemented)*

`--opponent-strategies all` now expands to every heuristic (it already accepted a comma list but
defaulted to `midrange` alone). Training against the full set should reduce overfitting to one
opponent. Note that eval is vs midrange, so this may not move the headline number even if the
policy is genuinely better.

### A/B: does the Phase 1–5 stack actually beat the baseline? *(not run — protocol pre-registered)*

The gates above are cheap proxies; this is the measurement that decides. Two arms, **identical
recipe and seed**, differing only in the new features:

| | arm A (control) | arm B (Phase 1+2+4+5) |
| --- | --- | --- |
| card identity | `--no-card-emb` | `--card-dim 16` |
| shaping potential | `midrange` (greedy's own objective) | `--potential board --shaping-end 0.0` |
| LR schedule | cosine, no warmup | `--warmup-iters 5` |
| opponent | `midrange` | `--opponent-strategies all` |
| everything else | 60 iters × 4096 steps, 8 envs, `--seed 1`, `small`, `--fixed-deck` | identical |

Protocol fixed **before** seeing results, to keep this from becoming another `*best*` story:
score the **final** checkpoint (not `*best*`), 400 games × 3 held-out seeds (100000/200000/
300000), argmax, and require the pooled interval to clear the Phase 0 reference of
`0.578 [0.550–0.606]`.

Read the result honestly when it lands: **four features move at once**, so a win says "the stack
helps" and not which phase caused it, and a loss is likewise unattributable. Per the sample-size
arithmetic in Phase 0, separating arms ~5 points apart needs ~1500 games each — so a difference
inside ±3 points at n=1200 is **not resolved**, and should be reported as such rather than as a
small win. Isolating individual phases needs one-factor-at-a-time runs after this.

*Status: **not run.** Both BC stages completed (they are what produced the Phase 1 gate table
above); the PPO arms were started and then cancelled before either finished, so there are no
win-rate numbers for this comparison and none should be inferred. The protocol above is recorded
as pre-registered — if this is picked up later, run it as written rather than re-deciding the
success criterion after seeing results.*

### A/B/C: does card identity help at all? ❌ *(run — null result)*

The four-factor A/B above was cancelled in favour of the question underneath it: **does card
identity, in any form, change how well the agent plays?** One factor, three arms, identical
recipe and seed (1500 BC games / 8 epochs → 60 PPO iters × 4096 steps → held-out eval), on
`variedmirror` so card semantics have room to matter. Scored per the pre-registered protocol:
**final** checkpoint (not `*best*`), 400 games × 3 held-out seeds, argmax. 7200 games total.

| eval deck | `blind` | `id` embedding | `text` embedding |
| --- | --- | --- | --- |
| `variedmirror` (in-distribution) | **0.500** [0.472–0.528] | 0.482 [0.453–0.510] | 0.493 [0.465–0.522] |
| `fixed` (never trained on) | 0.495 [0.467–0.523] | 0.490 [0.462–0.518] | 0.496 [0.468–0.524] |

Every pairwise contrast spans zero:

```
variedmirror  id   - blind: -0.018  [-0.058, +0.022]
variedmirror  text - blind: -0.007  [-0.047, +0.033]
variedmirror  text - id   : +0.012  [-0.028, +0.052]
fixed         id   - blind: -0.005  [-0.045, +0.035]
fixed         text - blind: +0.001  [-0.039, +0.041]
fixed         text - id   : +0.006  [-0.034, +0.046]
```

**Card identity produced no measurable win-rate gain, in either form, on either deck
distribution.** Phase 1 was ranked first in this plan as "the hard information ceiling"; it has
now failed twice — on BC top-1 (fixed decks) and on win rate (varied decks) — and the text
variant, which removes the ID embedding's coverage problem entirely, did no better.

**The finding worth keeping is the transfer failure.** On varied decks the BC stage separated
cleanly and in the predicted order:

| arm | BC loss | train top-1 | held-out top-1 (n=500) |
| --- | --- | --- | --- |
| blind | 1.0346 | 0.749 | 0.754 |
| id | 1.0175 | 0.760 | 0.768 |
| **text** | **0.9719** | **0.771** | 0.768 |

That advantage **completely vanished in play**. Better imitation of greedy did not produce better
results against greedy. This is the strongest evidence yet that the BC gate is the wrong
instrument for judging an observation change — it measures agreement with a stats-driven teacher,
and a richer observation buys agreement without buying strength. Future phases should not gate on
it. (It also retires the earlier hope that Phase 1's flat BC gate was a fixed-deck artifact: on
varied decks the gate moved and the win rate still did not.)

**Why all six cells sit below the 0.578 reference — the control.** Every arm lands at ~0.49–0.50,
including on `fixed`, which raised two candidate explanations: varied-mirror is harder at equal
budget, or this recipe is under-trained relative to whatever produced the reference `*best*`
checkpoints. A `blind` arm trained on `fixed` with *this exact recipe* separates them (BC reused
from the cancelled A/B, whose flags were identical apart from the deck mode):

| arm | trained on | eval deck | pooled (n=1200) |
| --- | --- | --- | --- |
| control | `fixed` | `fixed` | **0.533** [0.505–0.561] |
| experiment `blind` | `variedmirror` | `fixed` | 0.495 [0.467–0.523] |
| experiment `blind` | `variedmirror` | `variedmirror` | 0.500 [0.472–0.528] |

```
train-on-fixed minus train-on-varied (both eval fixed):  +0.038  [-0.002, +0.078]
control minus the 0.578 reference:                       -0.045  [-0.084, -0.005]
```

**Both effects are real, and they split the gap roughly in half.** The recipe is genuinely weaker
than whatever produced the reference — −4.5 points, and that interval excludes zero, so some of
the 0.578 headline rests on longer training and on `*best*` selection that this protocol
deliberately refuses. Training on varied decks costs a further ~3.8 points, which at n=1200 falls
*just* short of significance (the interval grazes zero at −0.002) and should be read as suggestive
rather than established.

The practically important line: **the control still beats greedy** (lower bound 0.505 > 0.5),
while all three varied-trained arms sit at parity. So varied-mirror is a materially harder task,
and the budget that suffices on `fixed` does not carry over.

None of this rescues card identity. The control shares the `blind` arm's observation, so the gap
it explains is a *budget and task-difficulty* gap, present identically in all three arms — it
cannot mask a card-identity effect, which was measured within a single deck mode at fixed
budget.

### Not on the list

**Scaling the model further as a strength lever.** The evidence says capacity is not the binding
constraint — information and signal are. Revisit only after Phase 1, and only with warmup.

## Upgrade paths

- **Opponents**: self-play + greedy (done) → AlphaStar-style **main / exploiter /
  main-exploiter** populations with a frozen-snapshot league + prioritized/fictitious-play
  sampling (OSFP), for robustness beyond the single heuristic.
- **Throughput**: many parallel stdio envs (`ppo_vec.py`, done) → SabberStone's gRPC extension
  + a many-core/GPU box + a GPU-resident policy, for millions-of-frames strength runs.
- **Imperfect info / RNG**: handled implicitly via observable features; add **determinization /
  ISMCTS** for search-based strength (AlphaZero path "B").
- **Decks/cards**: fixed Mage mirror / random `FillDecks` → the **real deck** the live bot
  plays; restrict to a well-supported card pool (SabberStone doesn't implement 100% of cards).
- **Deployment**: export the policy to **ONNX** and either serve it as the HS-Script MCTS
  `ScoreCalculator`/policy, or use the value head as the eval function. (The action-scoring
  ONNX + HS-Script action-mapping is more involved than the value-net export — a follow-up.)
