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


class ActorCritic(nn.Module):
    """Entity-transformer -> LSTM belief state -> action scorer + privileged value.

    Hearthstone is a POMDP (hidden opponent hand/deck), so a single board snapshot is not a
    sufficient statistic. The LSTM carries a recurrent state across the agent's decisions
    within a game, summarising the history into a belief the heads condition on — the standard
    recurrent core in AlphaStar / OpenAI Five / Xiao et al. (LSTM, not a transformer-over-time,
    for a cheap per-step recurrent state in online RL).

    Recurrence uses the R2D2 stored-state scheme: each transition keeps the LSTM input state,
    so the PPO update recomputes one step from it and minibatches stay per-transition (no BPTT
    through time). `forward` returns the next state too, for stepping during rollout.
    """

    def __init__(self, act_dim=ACT_DIM, priv_dim=PRIV_DIM, hid=128):
        super().__init__()
        d = 64   # entity-encoder embedding
        hdim = 64  # LSTM hidden size

        self.enc = EntityEncoder(d=d)
        self.lstm = nn.LSTMCell(d, hdim)
        self.scorer = nn.Sequential(nn.Linear(hdim + act_dim, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.value_enc = nn.Sequential(nn.Linear(hdim + priv_dim, hid), nn.ReLU(),
                                        nn.Linear(hid, hid), nn.ReLU())
        self.value = nn.Linear(hid, 1)
        self.hdim = hdim

    def initial_state(self, batch=1):
        z = torch.zeros(batch, self.hdim)
        return z, z.clone()

    def forward(self, tokens, tmask, hin, cin, priv, actions, amask):
        e = self.enc(tokens, tmask)                          # [B,d] per-state embedding
        h, c = self.lstm(e, (hin, cin))                      # [B,H] belief state
        _, n, _ = actions.shape
        se = h.unsqueeze(1).expand(-1, n, -1)                # [B,N,H]
        logits = self.scorer(torch.cat([se, actions], dim=-1)).squeeze(-1)  # [B,N]
        logits = logits.masked_fill(~amask, -1e9)
        v = self.value(self.value_enc(torch.cat([h, priv], dim=-1))).squeeze(-1)  # [B] (privileged)
        return logits, v, h, c


@torch.no_grad()
def act(policy, obs, hin, cin):
    """One recurrent step. Returns (idx, logprob, value, h_out, c_out)."""
    tk = torch.from_numpy(obs.tokens).unsqueeze(0)
    tm = torch.ones(1, obs.tokens.shape[0], dtype=torch.bool)
    p = torch.from_numpy(obs.priv).unsqueeze(0)
    a = torch.from_numpy(obs.actions).unsqueeze(0)
    am = torch.ones(1, obs.actions.shape[0], dtype=torch.bool)
    logits, v, h, c = policy(tk, tm, hin, cin, p, a, am)
    dist = torch.distributions.Categorical(logits=logits[0])
    idx = dist.sample()
    return int(idx), float(dist.log_prob(idx)), float(v[0]), h, c


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


@torch.no_grad()
def evaluate(env, main, episodes, opponent="random"):
    """Benchmark the recurrent main policy vs a fixed opponent. Returns win rate."""
    wins = 0
    for _ in range(episodes):
        obs = env.reset()
        seat = random.choice((1, 2))
        mh, mc = main.initial_state()
        while True:
            if obs.player == seat:
                i, _, _, mh, mc = act(main, obs, mh, mc)
                obs, done, winner = env.step(i)
            elif opponent == "greedy":
                obs, done, winner = env.step_greedy()
            else:
                obs, done, winner = env.step(int(np.random.randint(obs.actions.shape[0])))
            if done:
                wins += int(winner == seat)
                break
    return wins / max(episodes, 1)
