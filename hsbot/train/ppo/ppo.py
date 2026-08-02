"""PPO model and utilities for Hearthstone RL over SabberStone.

Shared by ppo_vec.py (training), pretrain.py (imitation pre-training), and eval.
Kept deliberately small: model definition, GAE, helper functions — no training loop.
"""
from __future__ import annotations
import math
import random

import numpy as np
import torch
import torch.nn as nn

from env import TOKEN_DIM, ACT_DIM, PRIV_DIM


class EntityEncoder(nn.Module):
    """Transformer over the entity-token set -> a single state embedding (masked mean-pool).
    No positional encoding: entities are a set, so the encoder is permutation-invariant."""

    def __init__(self, token_dim=TOKEN_DIM, d=64, nhead=4, layers=2, ff=128,
                 card_dim=16, card_text=None):
        super().__init__()
        # Card identity is categorical, so it enters as an embedding rather than a float feature.
        # Without it the 18-float token carries no card semantics at all: two different 4-cost
        # spells are byte-identical inputs.
        #
        # `card_text` is a frozen [vocab, D] matrix of card-text embeddings (see card_text.py),
        # projected to `card_dim` by a small trained layer. Meaning comes from the text, so all
        # 8303 rows are populated before a single game is played and a card that never appeared in
        # training is represented by its similarity to ones that did. Trainable cost is just the
        # projection (D x card_dim).
        #
        # The two alternatives this used to support — a learned per-card `nn.Embedding`, and a
        # card-blind mode — were removed. Neither beat the other in-distribution (all three arms
        # landed at ~0.49-0.50 over 7200 games), and both are undefined on an unseen card: blind
        # has no card signal at all, and the learned table returns a random-init row *silently*,
        # which is worse than no signal because it is indistinguishable from data.
        if card_text is None:
            raise ValueError(
                "EntityEncoder requires card_text. The learned-id and card-blind paths were "
                "removed; build the matrix with `python card_text.py --out card_text_emb.npz`."
            )
        # Frozen: these vectors are the input data, not parameters. Fine-tuning them would
        # re-introduce exactly the problem being solved, since only seen rows would move.
        self.card_text = nn.Embedding.from_pretrained(card_text, freeze=True, padding_idx=0)
        self.card_proj = nn.Linear(card_text.shape[1], card_dim)

        self.embed = nn.Linear(token_dim + card_dim, d)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=ff, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.out_dim = d

    def forward(self, tokens, tmask, card_ids=None):
        """tokens [B,T,token_dim]; tmask [B,T] bool (True = real); card_ids [B,T] int64. -> [B,d].

        `tmask=None` means "every token is real" and takes a mask-free path that is numerically
        identical (an all-True mask masks nothing, and the masked mean-pool reduces to a plain
        mean). It exists for speed: see the note on `src_key_padding_mask` in device.py — passing
        any mask in eval mode reaches an op with no MPS kernel, so each call round-trips to the
        CPU. Callers that know their mask is all-True should pass None instead."""
        if card_ids is None:
            card_ids = torch.zeros(tokens.shape[:2], dtype=torch.long, device=tokens.device)
        tokens = torch.cat([tokens, self.card_proj(self.card_text(card_ids))], dim=-1)
        x = self.embed(tokens)
        if tmask is None:
            return self.tr(x).mean(1)                          # no padding -> plain mean pool
        x = self.tr(x, src_key_padding_mask=~tmask)          # ignore padded tokens
        m = tmask.float().unsqueeze(-1)                       # [B,T,1]
        return (x * m).sum(1) / m.sum(1).clamp_min(1.0)       # masked mean pool


MEM = 16  # temporal context window: how many of the agent's recent decisions the belief attends over


class TemporalEncoder(nn.Module):
    """Transformer-over-time: attends over the rolling window of the agent's recent per-decision
    state embeddings to produce a belief state. The belief is the encoded most-recent slot.

    Learned positional embeddings are relative to *now* (slot -1 is the current decision, slot
    -k is k decisions ago), so the encoding is meaningful regardless of absolute game turn.
    Padding (early-game slots not yet filled) sits at the left and is masked out.
    """

    def __init__(self, d=64, mem=MEM, nhead=4, layers=2, ff=128):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, mem, d))
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=ff, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, hist, mask):
        """hist [B,K,d] (newest at slot -1); mask [B,K] bool (True = real). -> belief [B,d].

        `mask=None` means "the window is full" (no early-game padding left) and skips the mask
        for the same reason as `EntityEncoder.forward` — same result, fewer CPU round-trips."""
        x = self.tr(hist + self.pos, src_key_padding_mask=None if mask is None else ~mask)
        return x[:, -1, :]                                    # the current decision's slot = belief


class ActorCritic(nn.Module):
    """Entity-transformer -> transformer belief state -> action scorer + privileged value.

    Hearthstone is a POMDP (hidden opponent hand/deck), so a single board snapshot is not a
    sufficient statistic. A transformer-over-time (`TemporalEncoder`) carries history across the
    agent's decisions within a game — it self-attends over a fixed window of the last `MEM`
    per-decision embeddings and reads out the current slot as the belief the heads condition on.
    This replaces the earlier LSTM recurrent core (cf. GTrXL / AlphaStar's attention memory);
    attention over the window learns relations across decisions a single hidden vector can't.

    Memory is carried with the R2D2 stored-state scheme: the recurrent "state" is a rolling
    window `(hist, mask)` of past embeddings, and each transition keeps its INPUT window, so the
    PPO update recomputes one step from it and minibatches stay per-transition (no BPTT through
    time). `forward` appends the current embedding and returns the shifted next window too, for
    stepping during rollout. Bounded memory (window `MEM`) is the tradeoff vs the LSTM's
    unbounded-but-lossy state; recent board history dominates the belief in practice.
    """

    def __init__(self, act_dim=ACT_DIM, priv_dim=PRIV_DIM, hid=128, mem=MEM, size="small",
                 card_dim=16, card_text=None):
        super().__init__()
        # Presets scale entity-encoder + temporal-encoder dimensions together.
        # small  (173k) — original, for quick iteration
        # medium (~500k) — 3× scale
        # large  (~900k) — 5× scale
        # xl     (1.7M) — 10× scale
        # 100x   (17.6M) — 100× the `small` baseline, for capacity/scaling study
        cfgs = {
            "small":  dict(d=64,  nhead=4, layers=2, ff=128, hid=128),
            "medium": dict(d=96,  nhead=4, layers=3, ff=192, hid=192),
            "large":  dict(d=128, nhead=8, layers=4, ff=256, hid=256),
            "xl":     dict(d=128, nhead=8, layers=5, ff=320, hid=320),
            "100x":   dict(d=512, nhead=8, layers=4, ff=1024, hid=512),
        }
        cfg = cfgs.get(size, cfgs["small"])
        d = cfg["d"]; nhead = cfg["nhead"]; layers = cfg["layers"]; ff = cfg["ff"]; hid = cfg["hid"]
        self.mem = mem
        self.d = d

        self.enc = EntityEncoder(d=d, nhead=nhead, layers=layers, ff=ff,
                                 card_dim=card_dim, card_text=card_text)
        self.temporal = TemporalEncoder(d=d, mem=mem, nhead=nhead, layers=layers, ff=ff)
        self.scorer = nn.Sequential(nn.Linear(d + act_dim, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.value_enc = nn.Sequential(nn.Linear(d + priv_dim, hid), nn.ReLU(),
                                        nn.Linear(hid, hid), nn.ReLU())
        self.value = nn.Linear(hid, 1)

    def initial_state(self, batch=1):
        """Empty rolling window: (hist [B,MEM,d] zeros, mask [B,MEM] all-False)."""
        hist = torch.zeros(batch, self.mem, self.d)
        mask = torch.zeros(batch, self.mem, dtype=torch.bool)
        return hist, mask

    def forward(self, tokens, tmask, hist_in, mask_in, priv, actions, amask, card_ids=None):
        """`tmask` and `mask_in` may each be None, meaning "every slot is real" — see
        `EntityEncoder.forward`. `mask_in=None` says the belief window is already full, and since
        this step appends another real slot the window stays full, so the temporal mask is
        skipped too. The returned mask is always a tensor regardless, so callers can keep
        carrying it between steps unchanged."""
        e = self.enc(tokens, tmask, card_ids)                # [B,d] current per-state embedding
        # shift the window left (drop oldest) and append the current embedding at slot -1
        hist = torch.cat([hist_in[:, 1:, :], e.unsqueeze(1)], dim=1)          # [B,MEM,d]
        ones = torch.ones(e.shape[0], 1, dtype=torch.bool, device=e.device)
        if mask_in is None:
            mask = ones.expand(-1, self.mem).contiguous()                     # [B,MEM] all real
        else:
            mask = torch.cat([mask_in[:, 1:], ones], dim=1)                   # [B,MEM]
        h = self.temporal(hist, None if mask_in is None else mask)  # [B,d] belief state
        _, n, _ = actions.shape
        se = h.unsqueeze(1).expand(-1, n, -1)                # [B,N,d]
        logits = self.scorer(torch.cat([se, actions], dim=-1)).squeeze(-1)  # [B,N]
        logits = logits.masked_fill(~amask, -1e9)
        v = self.value(self.value_enc(torch.cat([h, priv], dim=-1))).squeeze(-1)  # [B] (privileged)
        return logits, v, hist, mask


@torch.no_grad()
def act(policy, obs, hist_in, mask_in, argmax=False):
    """One recurrent step. `(hist_in, mask_in)` is the belief window before this decision.
    Returns (idx, logprob, value, hist_out, mask_out) — the shifted window carries to the next step.

    `argmax=True` takes the highest-scoring legal action instead of sampling. Training must
    sample (PPO needs on-policy log-probs); evaluation should not, because sampling injects
    variance that swamps the effect being measured."""
    dev = next(policy.parameters()).device
    tk = torch.from_numpy(obs.tokens).unsqueeze(0).to(dev)
    ci = torch.from_numpy(obs.card_ids).unsqueeze(0).to(dev)
    p = torch.from_numpy(obs.priv).unsqueeze(0).to(dev)
    a = torch.from_numpy(obs.actions).unsqueeze(0).to(dev)
    am = torch.ones(1, obs.actions.shape[0], dtype=torch.bool, device=dev)
    # A single obs is never padded, so the entity mask is unconditionally all-True -> pass None.
    # The belief window *is* padded for the first MEM decisions of a game; test that on the CPU
    # copy, before the transfer, so the test itself costs no device sync.
    win = None if bool(mask_in.all()) else mask_in.to(dev)
    logits, v, hist, mask = policy(tk, None, hist_in.to(dev), win, p, a, am, ci)
    dist = torch.distributions.Categorical(logits=logits[0])
    idx = logits[0].argmax() if argmax else dist.sample()
    return int(idx), float(dist.log_prob(idx)), float(v[0]), hist.cpu(), mask.cpu()


def gae(rewards, values, dones, last_v, gamma=1.0, lam=0.95):
    """Generalized Advantage Estimation. Returns (advantages, returns)."""
    adv = np.zeros(len(rewards), np.float32)
    gae_ = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        next_v = last_v if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae_ = delta + gamma * lam * nonterminal * gae_
        adv[t] = gae_
    return adv, adv + np.asarray(values, np.float32)


def pad(seqs, dim):
    """Ragged list of [n_i, dim] -> padded [B, maxN, dim] + bool mask [B, maxN]."""
    b = len(seqs)
    maxn = max(s.shape[0] for s in seqs)
    out = np.zeros((b, maxn, dim), np.float32)
    mask = np.zeros((b, maxn), bool)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = True
    return torch.from_numpy(out), torch.from_numpy(mask)


def pad_ids(seqs, maxn=None):
    """Ragged list of [n_i] card-id arrays -> padded [B, maxN] int64. Pad value 0 is the
    CardVocab "none/unknown" row, which is also the embedding's padding_idx, so padded slots
    contribute a fixed zero vector and are masked out of attention regardless.

    `maxn` must match the width `pad()` produced for the same batch, otherwise tokens and ids
    would be misaligned."""
    b = len(seqs)
    maxn = maxn if maxn is not None else max(s.shape[0] for s in seqs)
    out = np.zeros((b, maxn), np.int64)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return torch.from_numpy(out)


OPPONENT_STRATEGIES = ["midrange", "aggro", "control", "fatigue", "ramp"]


def wilson_ci(wins, n, z=1.96):
    """Wilson score interval for a binomial proportion — the right interval for win rates.

    Preferred over the normal approximation p ± z·sqrt(p(1-p)/n), which misbehaves near 0 and 1
    (it happily reports bounds outside [0,1], e.g. for the 1.000 vs-random results here).
    Returns (lo, hi); at n=0 returns (0.0, 1.0)."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@torch.no_grad()
def evaluate(env, main, episodes, opponent="random", rng=None, argmax=True, lookahead=False):
    """Benchmark the recurrent main policy vs a fixed opponent. Returns (wins, episodes) so the
    caller can attach a confidence interval — a bare rate invites reading noise as signal.
    opponent: "random", "greedy"/"midrange", or any of "aggro","control","fatigue","ramp".

    `rng` seeds seat assignment and the random opponent. Pass a dedicated `random.Random` so
    evaluation neither consumes nor depends on the global RNG that training is driving — that
    coupling is what made past evals unreproducible.

    `argmax=True` (default) evaluates the policy deterministically. With sampling, two identical
    eval commands on the same checkpoint differed by 13 points (0.333 vs 0.467 over 60 games),
    because `dist.sample()` draws from the global torch RNG."""
    rng = rng if rng is not None else random.Random()
    wins = 0
    for _ in range(episodes):
        obs = env.reset()
        seat = rng.choice((1, 2))
        hist, mask = main.initial_state()
        while True:
            if obs.player == seat:
                if lookahead:
                    i, hist, mask = act_lookahead(main, env, obs, hist, mask)
                else:
                    i, _, _, hist, mask = act(main, obs, hist, mask, argmax=argmax)
                obs, done, winner = env.step(i)
            elif opponent == "random":
                obs, done, winner = env.step(rng.randrange(obs.actions.shape[0]))
            else:
                # "greedy" / "midrange" / "aggro" / "control" / "fatigue" / "ramp"
                strat = "midrange" if opponent == "greedy" else opponent
                obs, done, winner = env.step_greedy(strat)
            if done:
                wins += int(winner == seat)
                break
    return wins, episodes


@torch.no_grad()
def act_lookahead(policy, env, obs, hist_in, mask_in):
    """Phase 3: one-ply lookahead scored by the learned critic.

    Greedy searches one ply with a handcrafted board score; the plain policy searches zero plies
    with a learned score. This closes that gap: clone-and-apply every legal action in the engine,
    then value each resulting state with the critic and take the best. Terminal results use their
    true ±1 outcome rather than a value estimate.

    Returns (idx, hist_out, mask_out). The carried belief window is advanced with the *actual*
    current state, exactly as `act` does — lookahead picks the move, it does not rewrite history.
    """
    dev = next(policy.parameters()).device
    sims = env.simulate()
    n = len(sims)
    if n == 0:
        return 0, hist_in, mask_in

    scores = [None] * n
    live = [i for i, s in enumerate(sims) if not s["terminal"] and s["tokens"].shape[0] > 0]
    for i, s in enumerate(sims):
        if s["terminal"]:
            scores[i] = float(s["outcome"])
        elif i not in live:
            scores[i] = -1e9  # malformed/illegal — never choose

    if live:
        toks = [sims[i]["tokens"] for i in live]
        tk, tm = pad(toks, TOKEN_DIM)
        ci = pad_ids([sims[i]["card_ids"] for i in live], tk.shape[1])
        pr = torch.from_numpy(np.asarray([sims[i]["priv"] for i in live], np.float32))
        # Value each candidate from the same belief window: encode the hypothetical state, push it
        # into a copy of the window, and read the critic. No action set is needed — only the value.
        e = policy.enc(tk.to(dev), tm.to(dev), ci.to(dev))
        h_in = hist_in.to(dev).expand(len(live), -1, -1)
        m_in = mask_in.to(dev).expand(len(live), -1)
        hist = torch.cat([h_in[:, 1:, :], e.unsqueeze(1)], dim=1)
        ones = torch.ones(len(live), 1, dtype=torch.bool, device=dev)
        mask = torch.cat([m_in[:, 1:], ones], dim=1)
        h = policy.temporal(hist, mask)
        v = policy.value(policy.value_enc(torch.cat([h, pr.to(dev)], dim=-1))).squeeze(-1)
        for k, i in enumerate(live):
            # Negamax: when the turn flipped, v is the opponent's value for that state.
            scores[i] = float(v[k]) if sims[i]["mine"] else -float(v[k])

    best = int(max(range(n), key=lambda i: scores[i]))
    # Advance the belief window with the real current state (not the hypothetical one).
    tk = torch.from_numpy(obs.tokens).unsqueeze(0).to(dev)
    tm = torch.ones(1, obs.tokens.shape[0], dtype=torch.bool, device=dev)
    ci = torch.from_numpy(obs.card_ids).unsqueeze(0).to(dev)
    e = policy.enc(tk, tm, ci)
    hist = torch.cat([hist_in.to(dev)[:, 1:, :], e.unsqueeze(1)], dim=1)
    ones = torch.ones(1, 1, dtype=torch.bool, device=dev)
    mask = torch.cat([mask_in.to(dev)[:, 1:], ones], dim=1)
    return best, hist.cpu(), mask.cpu()
