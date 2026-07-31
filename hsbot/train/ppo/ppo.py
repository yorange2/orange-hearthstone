"""PPO model and utilities for Hearthstone RL over SabberStone.

Shared by ppo_vec.py (training), pretrain.py (imitation pre-training), and eval.
Kept deliberately small: model definition, GAE, helper functions — no training loop.
"""
from __future__ import annotations
import random

import numpy as np
import torch
import torch.nn as nn

from env import TOKEN_DIM, ACT_DIM, PRIV_DIM


class EntityEncoder(nn.Module):
    """Transformer over the entity-token set -> a single state embedding (masked mean-pool).
    No positional encoding: entities are a set, so the encoder is permutation-invariant."""

    def __init__(self, token_dim=TOKEN_DIM, d=64, nhead=4, layers=2, ff=128):
        super().__init__()
        self.embed = nn.Linear(token_dim, d)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=ff, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.out_dim = d

    def forward(self, tokens, tmask):
        """tokens [B,T,token_dim]; tmask [B,T] bool (True = real). -> [B,d]."""
        x = self.embed(tokens)
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
        """hist [B,K,d] (newest at slot -1); mask [B,K] bool (True = real). -> belief [B,d]."""
        x = self.tr(hist + self.pos, src_key_padding_mask=~mask)
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

    def __init__(self, act_dim=ACT_DIM, priv_dim=PRIV_DIM, hid=128, mem=MEM, size="small"):
        super().__init__()
        # Presets scale entity-encoder + temporal-encoder dimensions together.
        # small  (173k) — original, for quick iteration
        # medium (~500k) — 3× scale
        # large  (~900k) — 5× scale
        cfgs = {
            "small":  dict(d=64,  nhead=4, layers=2, ff=128, hid=128),
            "medium": dict(d=96,  nhead=4, layers=3, ff=192, hid=192),
            "large":  dict(d=128, nhead=8, layers=4, ff=256, hid=256),
        }
        cfg = cfgs.get(size, cfgs["small"])
        d = cfg["d"]; nhead = cfg["nhead"]; layers = cfg["layers"]; ff = cfg["ff"]; hid = cfg["hid"]
        self.mem = mem
        self.d = d

        self.enc = EntityEncoder(d=d, nhead=nhead, layers=layers, ff=ff)
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

    def forward(self, tokens, tmask, hist_in, mask_in, priv, actions, amask):
        e = self.enc(tokens, tmask)                          # [B,d] current per-state embedding
        # shift the window left (drop oldest) and append the current embedding at slot -1
        hist = torch.cat([hist_in[:, 1:, :], e.unsqueeze(1)], dim=1)          # [B,MEM,d]
        ones = torch.ones(e.shape[0], 1, dtype=torch.bool, device=e.device)
        mask = torch.cat([mask_in[:, 1:], ones], dim=1)                       # [B,MEM]
        h = self.temporal(hist, mask)                        # [B,d] belief state
        _, n, _ = actions.shape
        se = h.unsqueeze(1).expand(-1, n, -1)                # [B,N,d]
        logits = self.scorer(torch.cat([se, actions], dim=-1)).squeeze(-1)  # [B,N]
        logits = logits.masked_fill(~amask, -1e9)
        v = self.value(self.value_enc(torch.cat([h, priv], dim=-1))).squeeze(-1)  # [B] (privileged)
        return logits, v, hist, mask


@torch.no_grad()
def act(policy, obs, hist_in, mask_in):
    """One recurrent step. `(hist_in, mask_in)` is the belief window before this decision.
    Returns (idx, logprob, value, hist_out, mask_out) — the shifted window carries to the next step."""
    tk = torch.from_numpy(obs.tokens).unsqueeze(0)
    tm = torch.ones(1, obs.tokens.shape[0], dtype=torch.bool)
    p = torch.from_numpy(obs.priv).unsqueeze(0)
    a = torch.from_numpy(obs.actions).unsqueeze(0)
    am = torch.ones(1, obs.actions.shape[0], dtype=torch.bool)
    logits, v, hist, mask = policy(tk, tm, hist_in, mask_in, p, a, am)
    dist = torch.distributions.Categorical(logits=logits[0])
    idx = dist.sample()
    return int(idx), float(dist.log_prob(idx)), float(v[0]), hist, mask


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


OPPONENT_STRATEGIES = ["midrange", "aggro", "control", "fatigue", "ramp"]


@torch.no_grad()
def evaluate(env, main, episodes, opponent="random"):
    """Benchmark the recurrent main policy vs a fixed opponent. Returns win rate.
    opponent: "random", "greedy"/"midrange", or any of "aggro","control","fatigue","ramp"."""
    wins = 0
    for _ in range(episodes):
        obs = env.reset()
        seat = random.choice((1, 2))
        hist, mask = main.initial_state()
        while True:
            if obs.player == seat:
                i, _, _, hist, mask = act(main, obs, hist, mask)
                obs, done, winner = env.step(i)
            elif opponent == "random":
                obs, done, winner = env.step(int(np.random.randint(obs.actions.shape[0])))
            else:
                # "greedy" / "midrange" / "aggro" / "control" / "fatigue" / "ramp"
                strat = "midrange" if opponent == "greedy" else opponent
                obs, done, winner = env.step_greedy(strat)
            if done:
                wins += int(winner == seat)
                break
    return wins / max(episodes, 1)
