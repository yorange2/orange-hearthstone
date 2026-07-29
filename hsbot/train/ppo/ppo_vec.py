"""Vectorized (parallel-env) PPO — the throughput multiplier for a real-compute run.

Runs N SabberStone env subprocesses in lockstep (they step concurrently across CPU cores)
and batches policy inference across them. Same agent as ppo.py (entity-transformer + LSTM
belief state + privileged critic + action-scoring PPO + reward shaping); it reuses the model,
GAE, and padding from ppo.py unchanged.

Simplification vs ppo.py: the opponent is the **current policy (self-play)** or the **greedy
heuristic** — the frozen-snapshot league is dropped here so all policy inference batches
through one network (re-add per-env opponent nets later if wanted). Per-env trajectories are
kept separate so reward shaping and GAE stay per-episode-correct despite interleaving.

    python ppo_vec.py --num-envs 8 --iters 100 --steps 4096 --fixed-deck
"""
from __future__ import annotations
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from env import VecEnv, TOKEN_DIM, ACT_DIM
from ppo import ActorCritic, gae, pad, evaluate


@torch.no_grad()
def batched_act(policy, obs_list, hc_list):
    """Batched recurrent step over a list of decision points. Returns (idxs, logprobs, values,
    new_states) with one entry per input."""
    tok, tmask = pad([o.tokens for o in obs_list], TOKEN_DIM)
    act_t, amask = pad([o.actions for o in obs_list], ACT_DIM)
    priv = torch.from_numpy(np.asarray([o.priv for o in obs_list], np.float32))
    hin = torch.cat([hc[0] for hc in hc_list], 0)
    cin = torch.cat([hc[1] for hc in hc_list], 0)
    logits, v, h, c = policy(tok, tmask, hin, cin, priv, act_t, amask)
    dist = torch.distributions.Categorical(logits=logits)
    idx = dist.sample()
    lp = dist.log_prob(idx)
    new_hc = [(h[i:i + 1], c[i:i + 1]) for i in range(len(obs_list))]
    return idx.tolist(), lp.tolist(), v.tolist(), new_hc


def _blank_traj():
    return {k: [] for k in ("tok", "prv", "act", "idx", "lp", "v", "r", "d", "hin", "cin", "phi")}


def collect_vec(vec, main, greedy_prob, steps, gamma, shaping_coef):
    """Parallel self-play + greedy rollout. Gathers >= `steps` main transitions across N envs,
    keeping per-env trajectories so shaping/GAE are per-episode. Returns (batch, ep_returns)."""
    N = vec.n
    T = [_blank_traj() for _ in range(N)]
    ep_returns = []

    cur = vec.reset_all()
    main_seat = [random.choice((1, 2)) for _ in range(N)]
    opp = ["greedy" if random.random() < greedy_prob else "self" for _ in range(N)]
    mh = [main.initial_state() for _ in range(N)]  # main-seat LSTM state per env
    oh = [main.initial_state() for _ in range(N)]  # self-opponent LSTM state per env
    last = [None] * N
    total = 0

    while total < steps:
        main_ids, self_ids, greedy_ids = [], [], []
        for i in range(N):
            if cur[i].player == main_seat[i]:
                main_ids.append(i)
            elif opp[i] == "greedy":
                greedy_ids.append(i)
            else:
                self_ids.append(i)

        cmds: list[str | None] = [None] * N

        if main_ids:
            idxs, lps, vs, nhc = batched_act(main, [cur[i] for i in main_ids], [mh[i] for i in main_ids])
            for k, i in enumerate(main_ids):
                t = T[i]
                t["tok"].append(cur[i].tokens); t["prv"].append(cur[i].priv); t["act"].append(cur[i].actions)
                t["idx"].append(idxs[k]); t["lp"].append(lps[k]); t["v"].append(vs[k])
                t["r"].append(0.0); t["d"].append(0.0)
                t["hin"].append(mh[i][0].squeeze(0).numpy()); t["cin"].append(mh[i][1].squeeze(0).numpy())
                t["phi"].append(cur[i].potential)
                last[i] = len(t["idx"]) - 1
                mh[i] = nhc[k]
                cmds[i] = f"step {idxs[k]}"
                total += 1

        if self_ids:
            idxs, _, _, nhc = batched_act(main, [cur[i] for i in self_ids], [oh[i] for i in self_ids])
            for k, i in enumerate(self_ids):
                oh[i] = nhc[k]
                cmds[i] = f"step {idxs[k]}"

        for i in greedy_ids:
            cmds[i] = "step_greedy"

        res = vec.send_batch(cmds)

        done_ids = []
        for i, (obs, done, winner) in res.items():
            if done:
                if last[i] is not None:
                    r = 1.0 if winner == main_seat[i] else (-1.0 if winner != 0 else 0.0)
                    T[i]["r"][last[i]] += r
                    T[i]["d"][last[i]] = 1.0
                    ep_returns.append(r)
                done_ids.append(i)
            else:
                cur[i] = obs

        if done_ids:
            res2 = vec.send_batch(["reset" if i in done_ids else None for i in range(N)])
            for i in done_ids:
                cur[i] = res2[i][0]
                main_seat[i] = random.choice((1, 2))
                opp[i] = "greedy" if random.random() < greedy_prob else "self"
                mh[i] = main.initial_state(); oh[i] = main.initial_state()
                last[i] = None

    # ---- per-env shaping + GAE, then concatenate for the update ----
    tok_all, prv_all, act_all, idx_all, lp_all, hin_all, cin_all = ([] for _ in range(7))
    adv_all, ret_all = [], []
    for i in range(N):
        t = T[i]
        n = len(t["idx"])
        if n == 0:
            continue
        r = list(t["r"])
        if shaping_coef > 0:
            for u in range(n):
                next_phi = 0.0 if (t["d"][u] or u + 1 >= n) else t["phi"][u + 1]
                r[u] += shaping_coef * (gamma * next_phi - t["phi"][u])
        adv, ret = gae(r, t["v"], t["d"], 0.0, gamma)  # trailing incomplete episode bootstraps 0
        adv_all.append(adv); ret_all.append(ret)
        tok_all += t["tok"]; prv_all += t["prv"]; act_all += t["act"]
        idx_all += t["idx"]; lp_all += t["lp"]; hin_all += t["hin"]; cin_all += t["cin"]

    batch = dict(tok=tok_all, prv=prv_all, act=act_all, idx=idx_all, lp=lp_all,
                 hin=hin_all, cin=cin_all, adv=np.concatenate(adv_all), ret=np.concatenate(ret_all))
    return batch, ep_returns


def train(args):
    vec = VecEnv(args.num_envs, seed0=args.seed, fixed_deck=args.fixed_deck, dll=args.dll, dotnet=args.dotnet)
    main = ActorCritic()
    opt = torch.optim.Adam(main.parameters(), lr=args.lr)

    import time
    for it in range(1, args.iters + 1):
        t0 = time.time()
        b, ep = collect_vec(vec, main, args.greedy_prob, args.steps, args.gamma, args.shaping_coef)
        sps = len(b["idx"]) / max(time.time() - t0, 1e-9)

        tok_t, tmask_t = pad(b["tok"], TOKEN_DIM)
        act_t, amask_t = pad(b["act"], ACT_DIM)
        priv_t = torch.from_numpy(np.asarray(b["prv"], np.float32))
        hin_t = torch.from_numpy(np.asarray(b["hin"], np.float32))
        cin_t = torch.from_numpy(np.asarray(b["cin"], np.float32))
        idx_t = torch.tensor(b["idx"])
        oldlp_t = torch.tensor(b["lp"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        adv_t = torch.from_numpy(adv)
        ret_t = torch.from_numpy(b["ret"])

        n = len(b["idx"])
        for _ in range(args.epochs):
            for mb in torch.randperm(n).split(args.minibatch):
                logits, v, _, _ = main(tok_t[mb], tmask_t[mb], hin_t[mb], cin_t[mb],
                                       priv_t[mb], act_t[mb], amask_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(idx_t[mb]) - oldlp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[mb]
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = ((v - ret_t[mb]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pol_loss + args.vf * val_loss - args.ent * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(main.parameters(), 0.5); opt.step()

        tr_wr = float(np.mean([r > 0 for r in ep])) if ep else float("nan")
        line = (f"iter {it:3d}  envs {args.num_envs}  transitions {n:5d}  {sps:6.0f} steps/s  "
                f"episodes {len(ep):3d}  train_wr {tr_wr:.3f}  "
                f"pol {pol_loss.item():.3f}  val {val_loss.item():.3f}  ent {ent.item():.3f}")
        if it % args.eval_every == 0:
            wr_r = evaluate(vec.envs[0], main, args.eval_episodes, "random")
            wr_g = evaluate(vec.envs[0], main, args.eval_greedy_episodes, "greedy")
            line += f"  [vs random {wr_r:.3f} | vs greedy {wr_g:.3f}]"
        print(line, flush=True)

    torch.save(main.state_dict(), args.out)
    print(f"saved policy -> {args.out}")
    vec.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--steps", type=int, default=4096, help="main transitions per iteration")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--shaping-coef", type=float, default=0.1)
    ap.add_argument("--greedy-prob", type=float, default=0.3)
    ap.add_argument("--fixed-deck", action="store_true")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--eval-greedy-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="ppo_policy.pt")
    ap.add_argument("--dll", type=str, default=None)
    ap.add_argument("--dotnet", type=str, default="dotnet")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
