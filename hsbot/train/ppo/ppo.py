"""PPO for Hearthstone with a transformer entity-encoder policy over SabberStone.

State = a variable-size SET of entity tokens (hero/minions/hand/weapon/hero-power, 18 floats
each). A small transformer attends over the set (permutation-invariant, any board size) — the
relational board reasoning a flat feature vector can't do (cf. AlphaStar's entity encoder).
The pooled state embedding feeds:
  - an action-scoring policy head: score(concat(state_emb, action_feat[20])), softmax over the
    variable legal set (masking implicit);
  - a privileged value head that also sees opponent-hidden `priv[8]` (Xiao et al. "Cheat";
    critic unused at inference -> no train/test gap).

Training is **self-play with an opponent league**: Python drives both seats — the main policy
for its (randomly assigned) seat, and an opponent sampled each game from a pool of past policy
snapshots (or the current policy itself). Only the main seat's transitions are trained on.
Reward is terminal +/-1; gamma=1.0. A periodic evaluation vs a random opponent gives an
absolute benchmark (self-play win-rate alone hovers ~0.5 by construction).

    python ppo.py --iters 50 --steps 2048
"""
from __future__ import annotations
import argparse
import copy
import random

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


def collect(env, main, sample_opp, steps):
    """Self-play rollout. Runs whole episodes (main seat randomized, opponent sampled per game)
    until >= `steps` MAIN transitions are gathered. Only the main seat is stored; the terminal
    +/-1 reward is attached to that episode's last main transition. Returns buffers + returns."""
    tok, prv, act_, idx_, lp_, v_, r_, d_ = [], [], [], [], [], [], [], []
    ep_returns = []

    obs = env.reset()
    main_seat = random.choice((1, 2))
    opp = sample_opp()
    last_main = None
    while True:
        if obs.player == main_seat:
            i, lp, v = act(main, obs.tokens, obs.priv, obs.actions)
            tok.append(obs.tokens); prv.append(obs.priv); act_.append(obs.actions)
            idx_.append(i); lp_.append(lp); v_.append(v); r_.append(0.0); d_.append(0.0)
            last_main = len(idx_) - 1
        else:
            i, _, _ = act(opp, obs.tokens, obs.priv, obs.actions)

        obs, done, winner = env.step(i)
        if done:
            if last_main is not None:
                r = 1.0 if winner == main_seat else (-1.0 if winner != 0 else 0.0)
                r_[last_main] = r
                d_[last_main] = 1.0
                ep_returns.append(r)
            if len(idx_) >= steps:
                break
            obs = env.reset()
            main_seat = random.choice((1, 2))
            opp = sample_opp()
            last_main = None

    return (tok, prv, act_, idx_, lp_, v_, r_, d_), ep_returns


@torch.no_grad()
def evaluate(env, main, episodes):
    """Absolute benchmark: main policy vs a uniform-random opponent. Returns win rate."""
    wins = 0
    for _ in range(episodes):
        obs = env.reset()
        seat = random.choice((1, 2))
        while True:
            if obs.player == seat:
                i, _, _ = act(main, obs.tokens, obs.priv, obs.actions)
            else:
                i = int(np.random.randint(obs.actions.shape[0]))
            obs, done, winner = env.step(i)
            if done:
                wins += int(winner == seat)
                break
    return wins / max(episodes, 1)


def train(args):
    env = SabberEnv(seed=args.seed, dll=args.dll, dotnet=args.dotnet)
    main = ActorCritic()
    opp_net = ActorCritic()  # reused shell, loaded with a league snapshot on demand
    opt = torch.optim.Adam(main.parameters(), lr=args.lr)

    league: list[dict] = []  # frozen past snapshots (CPU state_dicts)

    def snapshot():
        league.append({k: v.detach().cpu().clone() for k, v in main.state_dict().items()})
        while len(league) > args.league_size:
            league.pop(0)

    def sample_opp():
        # current policy (self-play) with prob self_play_prob, else a random past snapshot
        if not league or random.random() < args.self_play_prob:
            return main
        opp_net.load_state_dict(random.choice(league))
        opp_net.eval()
        return opp_net

    snapshot()  # seed the league with the initial policy
    for it in range(1, args.iters + 1):
        (tok, prv, act_, idx_, lp_, v_, r_, d_), ep_returns = collect(env, main, sample_opp, args.steps)

        adv, ret = gae(r_, v_, d_, 0.0, args.gamma, args.lam)  # episodes end within the rollout
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        tok_t, tmask_t = pad(tok, TOKEN_DIM)
        act_t, amask_t = pad(act_, ACT_DIM)
        priv_t = torch.from_numpy(np.asarray(prv, np.float32))
        idx_t = torch.tensor(idx_)
        oldlp_t = torch.tensor(lp_)
        adv_t = torch.from_numpy(adv)
        ret_t = torch.from_numpy(ret)

        # ---- PPO update ----
        n = len(idx_)
        for _ in range(args.epochs):
            for mb in torch.randperm(n).split(args.minibatch):
                logits, v = main(tok_t[mb], tmask_t[mb], priv_t[mb], act_t[mb], amask_t[mb])
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
                nn.utils.clip_grad_norm_(main.parameters(), 0.5)
                opt.step()

        if it % args.league_every == 0:
            snapshot()

        sp_wr = float(np.mean([r > 0 for r in ep_returns])) if ep_returns else float("nan")
        line = (f"iter {it:3d}  league {len(league):2d}  episodes {len(ep_returns):3d}  "
                f"selfplay_wr {sp_wr:.3f}  pol_loss {pol_loss.item():.3f}  "
                f"val_loss {val_loss.item():.3f}  ent {ent.item():.3f}")
        if it % args.eval_every == 0:
            line += f"  [eval vs random {evaluate(env, main, args.eval_episodes):.3f}]"
        print(line)

    torch.save(main.state_dict(), args.out)
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
    # self-play league
    ap.add_argument("--self-play-prob", type=float, default=0.5, help="prob. opponent is the current policy")
    ap.add_argument("--league-every", type=int, default=5, help="snapshot the policy into the league every N iters")
    ap.add_argument("--league-size", type=int, default=10, help="max snapshots kept")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="ppo_policy.pt")
    ap.add_argument("--dll", type=str, default=None)
    ap.add_argument("--dotnet", type=str, default="dotnet")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
