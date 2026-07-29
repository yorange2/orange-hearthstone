# ppo — reinforcement learning on SabberStone (prototype)

A minimal, working **PPO** agent that learns Hearthstone by self-play against a random
opponent inside SabberStone. Demonstrates the shape of RL path "A": a **masked
action-scoring policy** over the variable legal-action set, driven through a simple
stdio bridge to the C# engine.

## Result

20 iterations (~20k steps, a couple of minutes on CPU) vs a random opponent:

```
iter  1  win_rate 0.35   ent 1.99
iter  5  win_rate 0.75   ent 1.99
iter 13  win_rate 0.86   ent 1.68
iter 18  win_rate 0.95   ent 1.78
iter 20  win_rate 0.85   ent 1.62
```

Win-rate climbs from worse-than-random to ~0.85, entropy falls (policy sharpens), value
loss drops — the loop learns.

## Pieces

- **C# env** (`hsbot/trainer/SabberStoneEnv/`): `HearthstoneEnv` wraps SabberStone as a
  single-agent MDP (agent = player 1; player 2 plays random inside the env). Encoders:
  `TokenEncoder` (state = a **set** of entity tokens, 18 floats each — hero/minions/hand/
  weapon/hero-power), `ActionEncoder` (each legal `PlayerTask` → 20 floats), `PrivilegedEncoder`
  (opponent-hidden info, 8 floats, critic-only). `Program.cs` is a line-based stdio JSON
  server: `reset` / `step <idx>` → `{tokens[T,18], priv[8], actions[N,20], reward, done}`.
- **`env.py`**: subprocess wrapper around that server.
- **`ppo.py`**: `EntityEncoder` (a small **transformer** over the entity-token set,
  permutation-invariant, masked mean-pool → state embedding) feeding `ActorCritic` (action
  scorer over `concat(state_emb, action_feat)` + privileged value head), rollout, GAE,
  clipped PPO, entropy bonus.

### Why a transformer entity-encoder (the "state = time series / sequence" idea)

Two distinct "transformer" opportunities exist; this implements the higher-leverage one:
- **Entity/set transformer (here)** — attention over the variable set of board/hand entities
  at the *current* state. Permutation-invariant, any board size, learns relations the flat
  144-vector can't (cf. AlphaStar's entity encoder). Stateless per decision → deploys like the
  MLP. This is the "v2" state representation (tokens), separate from the flat `docs/FEATURES.md`
  contract (still used by the supervised value-net pipeline).
- **Temporal transformer / LSTM (not done)** — over the *history* of turns, for the POMDP
  belief state (inferring the opponent's hidden hand/deck). Strong agents (AlphaStar, OpenAI
  Five, Xiao et al.) use an **LSTM** here for a cheap recurrent state in online RL; add it when
  self-play at scale makes opponent-modeling pay off. Costs history plumbing at inference.

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
shared card/hero embeddings; and OSFP self-play (vs the current random opponent).

## This is a prototype — upgrade paths

- **Opponent**: random → self-play with an **opponent pool / league** (avoids cycling).
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
