# ppo — reinforcement learning on SabberStone (prototype)

A working **PPO** agent that learns Hearthstone by **self-play with an opponent league** in
SabberStone. RL path "A": a **masked action-scoring policy** over the variable legal-action
set, built on a **transformer entity-encoder + LSTM belief state**, driven through a simple
stdio bridge to the C# engine.

## Result — and an honest benchmark vs a heuristic

25 iterations of self-play (prototype scale, a few minutes on CPU), evaluated against **two**
fixed opponents: uniform-random (weak) and a **greedy heuristic** (SabberStone's own
`MidRangeScore`, one-ply lookahead — the SabberStone analogue of Hearthstone-Script's `基础策略`):

```
iter  5   [vs random 0.60 | vs greedy 0.05]
iter 15   [vs random 0.70 | vs greedy 0.10]
iter 20   [vs random 0.63 | vs greedy 0.20]
iter 25   [vs random 0.83 | vs greedy 0.05]
```

**The prototype beats random ~80% but loses ~85-95% to the greedy heuristic.** So: does it
outperform `基础策略`-style play? **No — not at this scale.** "Beats random" is a near-useless
signal; a competent one-ply greedy heuristic is a much higher bar this prototype doesn't clear.

Closing the gap is a *scale + curriculum* problem, not an architecture one: put the greedy
agent (and stronger snapshots) into the training league so the policy actually trains against
it, run many parallel envs for far more samples, use real decks, and lengthen training — see
below. The value of this benchmark is that it replaces wishful "beats random" with the truth.

## Pieces

- **C# env** (`hsbot/trainer/SabberStoneEnv/`): `HearthstoneEnv` is a **seat-agnostic
  two-player** env — it plays no opponent itself; every decision point is returned tagged with
  `player` (whose turn), and Python routes it. Encoders: `TokenEncoder` (state = a **set** of
  entity tokens, 18 floats each — hero/minions/hand/weapon/hero-power), `ActionEncoder` (each
  legal `PlayerTask` → 20 floats), `PrivilegedEncoder` (opponent-hidden info, 8 floats,
  critic-only). `GreedyOpponent` plays SabberStone's `MidRangeScore` one-ply for the benchmark.
  `Program.cs` stdio JSON server: `reset` / `step <idx>` / `step_greedy` →
  `{tokens[T,18], priv[8], actions[N,20], player, done, winner}`.
- **`env.py`**: subprocess wrapper around that server.
- **`env.py`**: subprocess wrapper (`Obs` = tokens/priv/actions/player).
- **`ppo.py`**: `EntityEncoder` (a small **transformer** over the entity-token set,
  permutation-invariant, masked mean-pool → per-state embedding) → **LSTM belief state**
  (recurrent over the agent's decisions within a game) → `ActorCritic` (action scorer over
  `concat(belief, action_feat)` + privileged value head); **self-play league**
  (`collect` drives both seats — main policy stored, sampled opponent not; `snapshot`/
  `sample_opp` manage the pool of past policies); GAE, clipped PPO, entropy bonus; `evaluate`
  benchmarks vs random.

### Self-play league

Each game: the main policy takes a random seat; the opponent is the current policy (prob
`--self-play-prob`) or a uniformly-sampled past **snapshot** from the league. Only the main
seat's transitions train; the terminal ±1 reward is attached to that seat's last decision. The
policy is snapshotted into the league every `--league-every` iters (capped at `--league-size`).
This is the single-population core of league training — extend toward AlphaStar-style
main/exploiter/main-exploiter populations for robustness.

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
# build the env once (use the .NET 8 SDK)
dotnet build hsbot/trainer/SabberStoneEnv -c Release

cd hsbot/train/ppo
python ppo.py --iters 50 --steps 2048            # dotnet must be on PATH
# or point at a specific runtime:
python ppo.py --dotnet /path/to/dotnet --dll ../../trainer/SabberStoneEnv/bin/Release/net8.0/SabberStoneEnv.dll
```

Saves `ppo_policy.pt`.

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

## This is a prototype — upgrade paths

- **League**: single-population self-play (done) → **train against the greedy heuristic**
  (add it as a league opponent, not just an eval) and AlphaStar-style **main / exploiter /
  main-exploiter** populations + prioritized/fictitious-play opponent sampling (OSFP). This is
  the direct lever to actually beat the greedy benchmark above.
- **Reward**: terminal ±1 only → add **board-score shaping** for denser signal.
- **Imperfect info / RNG**: currently handled implicitly via the observable features; add
  **determinization / ISMCTS** for search-based strength (AlphaZero path "B").
- **Throughput**: single env over stdio → **many parallel envs** (or SabberStone's gRPC
  extension) for real sample scale; move the policy to GPU.
- **Decks/cards**: random `FillDecks` → the **real deck** the live bot plays; restrict to a
  well-supported card pool (SabberStone doesn't implement 100% of cards).
- **Deployment**: export the policy to **ONNX** and either serve it as the HS-Script MCTS
  `ScoreCalculator`/policy, or use the value head as the eval function. (The action-scoring
  ONNX + HS-Script action-mapping is more involved than the value-net export — a follow-up.)
