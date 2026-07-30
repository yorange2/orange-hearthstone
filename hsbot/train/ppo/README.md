# ppo — reinforcement learning on SabberStone (prototype)

A **PPO** agent that learns Hearthstone in SabberStone by **imitation warm-start + self-play/
greedy PPO**, and **outperforms the one-ply greedy heuristic** (~55% over 200 games). RL path
"A": a **masked action-scoring policy** over the variable legal-action set, built on a
**transformer entity-encoder + LSTM belief state**, driven through a simple stdio bridge to the
C# engine.

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
  per-state embedding) → **LSTM belief state** (recurrent over the agent's decisions within a
  game) → `ActorCritic` (action scorer over `concat(belief, action_feat)` + privileged value
  head); plus `act`, `gae`, `pad`, and `evaluate` (benchmark vs random/greedy).
- **`pretrain.py`**: imitation pre-training (behavioral cloning). Runs greedy-vs-greedy games,
  records (state, greedy-action) pairs, and trains the policy by cross-entropy to imitate the
  heuristic (~85% top-1). The resulting checkpoint is a strong warm start for PPO.
- **`ppo_vec.py`**: the training loop — vectorized self-play + greedy PPO with LR/entropy
  schedules, a greedy-prob curriculum, and best-checkpoint tracking. Also serves `--eval-only`.

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
- **Temporal LSTM belief state (done)** — an `nn.LSTMCell` carries a recurrent state across
  the agent's decisions within a game, summarising history into a belief the heads condition
  on (Hearthstone is a POMDP: hidden opponent hand/deck). LSTM, not transformer-over-time, for
  a cheap per-step recurrent state in online RL (as in AlphaStar / OpenAI Five / Xiao et al.).
  Recurrence uses R2D2 **stored-state** (each transition keeps its LSTM input state; the PPO
  update recomputes one step from it, so minibatches stay per-transition — no BPTT through
  time). At inference the plugin must carry the hidden state across the game's decisions.

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
across them. Same agent (entity-transformer + LSTM + privileged critic + shaping); per-env
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
