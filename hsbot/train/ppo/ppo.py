"""PPO for Hearthstone with a transformer entity-encoder policy over SabberStone.

State = a variable-size SET of entity tokens (hero/minions/hand/weapon/hero-power, 18 floats
each). A small transformer attends over the set (permutation-invariant, any board size) — the
relational board reasoning a flat feature vector can't do (cf. AlphaStar's entity encoder).
The pooled state embedding feeds:
  - an action-scoring policy head: score(concat(state_emb, action_feat[20])), softmax over the
    variable legal set (masking implicit);
  - a privileged value head that also sees opponent-hidden `priv[8]` (Xiao et al. "Cheat";
    critic unused at inference -> no train/test gap).

Trains vs a random opponent inside the env; reward is terminal +/-1; gamma=1.0. Prototype to
show the shape — upgrades in README (self-play league, parallel envs, ONNX export, ...).

    python ppo.py --iters 50 --steps 2048
"""
from __future__ import annotations
import argparse

import numpy as np
import torch
import torch.nn as nn

from env import SabberEnv, TOKEN_DIM, ACT_DIM, PRIV_DIM


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
    def __init__(self, act_dim=ACT_DIM, priv_dim=PRIV_DIM, hid=128):
        super().__init__()
        self.enc = EntityEncoder()
        d = self.enc.out_dim
        self.scorer = nn.Sequential(nn.Linear(d + act_dim, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.value_enc = nn.Sequential(nn.Linear(d + priv_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.value = nn.Linear(hid, 1)

    def forward(self, tokens, tmask, priv, actions, amask):
        s = self.enc(tokens, tmask)                          # [B,d] (policy trunk, observable)
        _, n, _ = actions.shape
        se = s.unsqueeze(1).expand(-1, n, -1)                # [B,N,d]
        logits = self.scorer(torch.cat([se, actions], dim=-1)).squeeze(-1)  # [B,N]
        logits = logits.masked_fill(~amask, -1e9)
        v = self.value(self.value_enc(torch.cat([s, priv], dim=-1))).squeeze(-1)  # [B] (privileged)
        return logits, v


@torch.no_grad()
def act(policy, tokens, priv, actions):
    """Single-state action selection during rollout. Returns (idx, logprob, value)."""
    tk = torch.from_numpy(tokens).unsqueeze(0)
    tm = torch.ones(1, tokens.shape[0], dtype=torch.bool)
    p = torch.from_numpy(priv).unsqueeze(0)
    a = torch.from_numpy(actions).unsqueeze(0)
    am = torch.ones(1, actions.shape[0], dtype=torch.bool)
    logits, v = policy(tk, tm, p, a, am)
    dist = torch.distributions.Categorical(logits=logits[0])
    idx = dist.sample()
    return int(idx), float(dist.log_prob(idx)), float(v[0])


def gae(rewards, values, dones, last_v, gamma=1.0, lam=0.95):
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


def train(args):
    env = SabberEnv(seed=args.seed, dll=args.dll, dotnet=args.dotnet)
    policy = ActorCritic()
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    tokens, priv, actions = env.reset()
    for it in range(1, args.iters + 1):
        # ---- collect a rollout ----
        buf_tok, buf_priv, buf_act, buf_idx = [], [], [], []
        buf_lp, buf_v, buf_r, buf_d = [], [], [], []
        ep_returns = []
        for _ in range(args.steps):
            idx, lp, v = act(policy, tokens, priv, actions)
            ntok, npriv, nactions, r, done = env.step(idx)
            buf_tok.append(tokens); buf_priv.append(priv); buf_act.append(actions); buf_idx.append(idx)
            buf_lp.append(lp); buf_v.append(v); buf_r.append(r); buf_d.append(float(done))
            if done:
                ep_returns.append(r)
                tokens, priv, actions = env.reset()
            else:
                tokens, priv, actions = ntok, npriv, nactions

        _, _, last_v = act(policy, tokens, priv, actions)
        adv, ret = gae(buf_r, buf_v, buf_d, last_v, args.gamma, args.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        tok_t, tmask_t = pad(buf_tok, TOKEN_DIM)
        act_t, amask_t = pad(buf_act, ACT_DIM)
        priv_t = torch.from_numpy(np.asarray(buf_priv, np.float32))
        idx_t = torch.tensor(buf_idx)
        oldlp_t = torch.tensor(buf_lp)
        adv_t = torch.from_numpy(adv)
        ret_t = torch.from_numpy(ret)

        # ---- PPO update ----
        n = len(buf_idx)
        for _ in range(args.epochs):
            for mb in torch.randperm(n).split(args.minibatch):
                logits, v = policy(tok_t[mb], tmask_t[mb], priv_t[mb], act_t[mb], amask_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                newlp = dist.log_prob(idx_t[mb])
                ratio = torch.exp(newlp - oldlp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[mb]
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = ((v - ret_t[mb]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pol_loss + args.vf * val_loss - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        wr = float(np.mean([r > 0 for r in ep_returns])) if ep_returns else float("nan")
        print(f"iter {it:3d}  episodes {len(ep_returns):3d}  win_rate {wr:.3f}  "
              f"pol_loss {pol_loss.item():.3f}  val_loss {val_loss.item():.3f}  ent {ent.item():.3f}")

    torch.save(policy.state_dict(), args.out)
    print(f"saved policy -> {args.out}")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--steps", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=1.0)  # Xiao et al.: terminal-only reward, short episodes
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="ppo_policy.pt")
    ap.add_argument("--dll", type=str, default=None)
    ap.add_argument("--dotnet", type=str, default="dotnet")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
