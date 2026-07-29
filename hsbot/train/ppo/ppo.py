"""Minimal PPO for Hearthstone with a masked action-scoring policy over SabberStone.

The policy embeds the 144-float state, then scores each legal action's 20-float vector by
MLP(concat(state_emb, action_feat)) and softmaxes over the *variable* legal set — so illegal
actions are never scored (masking is implicit). A value head estimates the state value.

Single env vs a random opponent (see HearthstoneEnv). Reward is terminal ±1, so episodic
return == win(+1)/loss(-1); "win rate" below is (return > 0). This is a prototype to show the
shape: correct PPO mechanics (GAE, clipped objective, value loss, entropy), variable actions,
and the SabberStone bridge. Upgrades: self-play/opponent pool, reward shaping, parallel envs,
ONNX export of the policy for the HS-Script plugin.

    python ppo.py --iters 50 --steps 2048
"""
from __future__ import annotations
import argparse

import numpy as np
import torch
import torch.nn as nn

from env import SabberEnv, OBS_DIM, ACT_DIM, PRIV_DIM


class ActorCritic(nn.Module):
    """Asymmetric (privileged) actor-critic. The policy sees only the observable state; the
    value head additionally sees `priv` (opponent-hidden features) — the "Cheat" technique
    from Xiao et al. 2023. The critic is unused at inference, so there is no train/test gap.
    """

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, priv_dim=PRIV_DIM, hid=128):
        super().__init__()
        self.state_enc = nn.Sequential(nn.Linear(obs_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(hid + act_dim, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.value_enc = nn.Sequential(nn.Linear(obs_dim + priv_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.value = nn.Linear(hid, 1)

    def forward(self, obs, priv, actions, mask):
        """obs [B,obs]; priv [B,priv]; actions [B,N,act]; mask [B,N]. -> logits [B,N], value [B]."""
        s = self.state_enc(obs)                      # [B,hid] (policy, observable only)
        _, n, _ = actions.shape
        se = s.unsqueeze(1).expand(-1, n, -1)        # [B,N,hid]
        logits = self.scorer(torch.cat([se, actions], dim=-1)).squeeze(-1)  # [B,N]
        logits = logits.masked_fill(~mask, -1e9)
        v = self.value(self.value_enc(torch.cat([obs, priv], dim=-1))).squeeze(-1)  # [B] (privileged)
        return logits, v


@torch.no_grad()
def act(policy, obs, priv, actions):
    """Single-state action selection during rollout. Returns (idx, logprob, value)."""
    o = torch.from_numpy(obs).unsqueeze(0)
    p = torch.from_numpy(priv).unsqueeze(0)
    a = torch.from_numpy(actions).unsqueeze(0)
    m = torch.ones(1, actions.shape[0], dtype=torch.bool)
    logits, v = policy(o, p, a, m)
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


def pad(action_list):
    """Ragged list of [n_i, act_dim] -> padded [B, maxN, act_dim] + bool mask [B, maxN]."""
    b = len(action_list)
    maxn = max(a.shape[0] for a in action_list)
    out = np.zeros((b, maxn, ACT_DIM), np.float32)
    mask = np.zeros((b, maxn), bool)
    for i, a in enumerate(action_list):
        out[i, : a.shape[0]] = a
        mask[i, : a.shape[0]] = True
    return torch.from_numpy(out), torch.from_numpy(mask)


def train(args):
    env = SabberEnv(seed=args.seed, dll=args.dll, dotnet=args.dotnet)
    policy = ActorCritic()
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    obs, priv, actions = env.reset()
    for it in range(1, args.iters + 1):
        # ---- collect a rollout ----
        buf_obs, buf_priv, buf_act, buf_idx = [], [], [], []
        buf_lp, buf_v, buf_r, buf_d = [], [], [], []
        ep_returns = []
        for _ in range(args.steps):
            idx, lp, v = act(policy, obs, priv, actions)
            nobs, npriv, nactions, r, done = env.step(idx)
            buf_obs.append(obs); buf_priv.append(priv); buf_act.append(actions); buf_idx.append(idx)
            buf_lp.append(lp); buf_v.append(v); buf_r.append(r); buf_d.append(float(done))
            if done:
                ep_returns.append(r)
                obs, priv, actions = env.reset()
            else:
                obs, priv, actions = nobs, npriv, nactions

        _, _, last_v = act(policy, obs, priv, actions)
        adv, ret = gae(buf_r, buf_v, buf_d, last_v, args.gamma, args.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.from_numpy(np.asarray(buf_obs, np.float32))
        priv_t = torch.from_numpy(np.asarray(buf_priv, np.float32))
        act_t, mask_t = pad(buf_act)
        idx_t = torch.tensor(buf_idx)
        oldlp_t = torch.tensor(buf_lp)
        adv_t = torch.from_numpy(adv)
        ret_t = torch.from_numpy(ret)

        # ---- PPO update ----
        n = len(buf_idx)
        for _ in range(args.epochs):
            for mb in torch.randperm(n).split(args.minibatch):
                logits, v = policy(obs_t[mb], priv_t[mb], act_t[mb], mask_t[mb])
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
