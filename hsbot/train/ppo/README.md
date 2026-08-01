# ppo — reinforcement learning on SabberStone (prototype)

A **PPO** agent that learns Hearthstone in SabberStone by **imitation warm-start + self-play/
greedy PPO**, and **outperforms the one-ply greedy heuristic** (~55% over 200 games). RL path
"A": a **masked action-scoring policy** over the variable legal-action set, built on a
**transformer entity-encoder + transformer belief state**, driven through a simple stdio bridge
to the C# engine.

## Result — beats the greedy heuristic

Evaluated against **two** fixed opponents: uniform-random (weak) and a **greedy heuristic**
(SabberStone's own `MidRangeScore`, one-ply lookahead — the SabberStone analogue of
Hearthstone-Script's `基础策略`). Over 200 eval games on the fixed Mage-mirror deck:

```
vs random  1.000
vs greedy  0.550   [95% CI 0.48–0.62]
```

**The agent outperforms the one-ply greedy heuristic (~55%)** — the honest "do we beat a
heuristic?" bar, not just "do we beat random?".

> **Caveat on this number.** It was read off a `*best*` checkpoint selected as the max over
> repeated 30-game evals, which biases high — reduced-budget replications dropped 8–18 points
> when re-evaluated on a held-out seed over 200 games. See
> [Model scale: bigger is not better here](#model-scale-bigger-is-not-better-here). The 55% has
> not been re-measured at full budget on a held-out seed; treat it as optimistic until it is.

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
| PPO rollout (`collect_vec`) | tiny + **ragged**, change every step | CPU ~2× faster | CPU ~3× faster |
| BC / PPO update | big fixed-ish batches (256–512) | MPS ~5× faster | MPS ~2.6× faster |

MPS loses the rollout at **every** model size: the ragged token/action padding means the shape
changes on nearly every call, which defeats MPS graph caching (a few nested-tensor ops also lack
MPS kernels and fall back to CPU). It wins clearly on large-batch training, where one shape is
reused. So the fastest recipe on Apple silicon is **`pretrain.py --device mps`, `ppo_vec.py
--device cpu`** — about 2× faster end-to-end than using either device for both. `auto` cannot
know this (it sees hardware, not phase), so pass `--device` explicitly when it matters.

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
games before quoting a win rate — including the ~55% headline above, which was selected the
same way.

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

### Phase 0 — Make the metric trustworthy *(blocking, ~1h)*

Nothing downstream is measurable until this lands: a genuine 5-point gain is currently invisible
inside the noise, and `*best*` selection inflates results by 8–18 points (see the note above).

- Raise checkpoint-selection evals to ≥200 games, or stop selecting on them and evaluate the
  final model instead
- Standard eval: fixed **held-out** seed, **400 games**, report a Wilson 95% CI
  (±0.049 at n=400, versus ±0.09 at the current n=30)
- Add `--eval-seed` so eval seeds can never overlap training seeds
- Re-measure `small` at full budget to establish one honest reference number

**Gate:** a reference win rate with a confidence interval. Phases 1–4 are guesswork without it.

### Phase 1 — Put card identity in the observation *(highest expected gain)*

The hard information ceiling. A token is 18 floats with **no card ID and no card text**, so two
different 4-cost spells are identical inputs and the policy cannot represent "Polymorph the 7/7"
as distinct from "play a vanilla 4-drop". Greedy, by contrast, scores through the real simulator
and implicitly knows what every card does.

- `TokenEncoder.cs`: emit a card-ID index alongside the 18 floats
- `env.py`: parse it into `Obs`
- `ppo.py`: `nn.Embedding(vocab, 32)` concatenated into the token before `EntityEncoder`
- Open question to settle first: SabberStone's card-ID space, and whether to embed all
  collectibles or hash into a fixed vocab. Under `--fixed-deck` it is ~30 unique cards — start
  there, widen later.

Changes `TOKEN_DIM`, so it invalidates existing checkpoints. RL-only: the flat `FeatureExtractor`
feeding the ONNX value net is a separate contract and is untouched.

**Gate:** BC top-1 must clear the current 0.823 *before* spending anything on PPO. If it does
not, the embedding is miswired — a cheap early signal.

### Phase 2 — Decouple the reward from greedy *(cheap, high information)*

Three anchors hold the policy at heuristic level, the strongest being that the shaping potential
**is** greedy's own objective (`HearthstoneEnv.cs:147`).

- Anneal `shaping_coef` → 0 over training so the final policy is not anchored
- Ablate `--shaping-coef 0` vs `0.05` vs annealed at fixed budget
- Consider a potential built from raw board state (health/tempo differential) instead of the
  exact function greedy maximizes

Genuinely uncertain: shaping is part of what makes the current run learn at all, so removing it
may hurt before it helps. That is why it is an ablation, not a change.

### Phase 3 — Give the agent the lookahead greedy already has

Greedy is one-ply search + handcrafted score; the agent is **zero-ply** + learned score. Add a
`simulate <idx>` endpoint to the C# env and score each legal action by the learned value of the
resulting state. Largest structural gap after Phase 1, and it reuses the critic already being
trained. Costs inference time (N simulations per decision), so measure it as an eval/deployment
lever separately from training changes.

### Phase 4 — Optimization hygiene, then re-test scale

Only meaningful after Phases 0–1.

- **LR warmup + lower LR for large models** — this is what broke `100x` above
- Re-run the scale study with warmup, to answer the question the table above leaves open: does
  capacity help once optimization is not broken?
- Raise the PPO sample budget (1.6M transitions at the full recipe)
- Free speedup already measured: BC on `mps`, PPO on `cpu` ≈ 2× end-to-end

### Phase 5 — Opponent diversity

`--opponent-strategies` already supports all five heuristics but defaults to `midrange` alone.
Training against the full set should reduce overfitting to one opponent. Cheap; note that eval is
vs midrange, so this may not move the headline number even if the policy is genuinely better.

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
