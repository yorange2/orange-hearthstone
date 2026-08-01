"""Vectorized (parallel-env) PPO — the throughput multiplier for a real-compute run.

Runs N SabberStone env subprocesses in lockstep (they step concurrently across CPU cores)
and batches policy inference across them. Same agent as ppo.py (entity-transformer + LSTM
belief state + privileged critic + action-scoring PPO + reward shaping); it reuses the model,
GAE, and padding from ppo.py unchanged.

Simplification vs ppo.py: the opponent is the **current policy (self-play)** or the **greedy
heuristic** — the frozen-snapshot league is dropped here so all policy inference batches
through one network (re-add per-env opponent nets later if wanted). Per-env trajectories are
kept separate so reward shaping and GAE stay per-episode-correct despite interleaving.

    python ppo_vec.py --num-envs 8 --iters 200 --steps 8192 --fixed-deck
"""
from __future__ import annotations
import argparse
import math
import random
import time

from device import resolve_device, CHOICES as DEVICE_CHOICES  # before torch: sets MPS fallback

import numpy as np
import torch
import torch.nn as nn

from env import SabberEnv, VecEnv, TOKEN_DIM, ACT_DIM
from ppo import ActorCritic, gae, pad, pad_ids, evaluate, wilson_ci, OPPONENT_STRATEGIES


@torch.no_grad()
def batched_act(policy, obs_list, hc_list, device="cpu"):
    """Batched recurrent step over a list of decision points. Returns (idxs, logprobs, values,
    new_states) with one entry per input."""
    tok, tmask = pad([o.tokens for o in obs_list], TOKEN_DIM)
    cid = pad_ids([o.card_ids for o in obs_list], tok.shape[1])
    act_t, amask = pad([o.actions for o in obs_list], ACT_DIM)
    priv = torch.from_numpy(np.asarray([o.priv for o in obs_list], np.float32))
    hin = torch.cat([hc[0] for hc in hc_list], 0)
    cin = torch.cat([hc[1] for hc in hc_list], 0)
    logits, v, h, c = policy(tok.to(device), tmask.to(device), hin.to(device), cin.to(device),
                              priv.to(device), act_t.to(device), amask.to(device), cid.to(device))
    dist = torch.distributions.Categorical(logits=logits)
    idx = dist.sample()
    lp = dist.log_prob(idx)
    new_hc = [(h[i:i + 1].cpu(), c[i:i + 1].cpu()) for i in range(len(obs_list))]
    return idx.tolist(), lp.tolist(), v.tolist(), new_hc


def _blank_traj():
    return {k: [] for k in ("tok", "cid", "prv", "act", "idx", "lp", "v", "r", "d", "hin", "cin", "phi")}


def collect_vec(vec, main, greedy_prob, steps, gamma, shaping_coef, strategies, device="cpu"):
    """Parallel self-play + greedy rollout. Gathers >= `steps` main transitions across N envs,
    keeping per-env trajectories so shaping/GAE are per-episode. Returns (batch, ep_returns).
    `strategies` is a list of opponent strategy names sampled uniformly for greedy seats."""
    N = vec.n
    T = [_blank_traj() for _ in range(N)]
    ep_returns = []

    cur = vec.reset_all()
    main_seat = [random.choice((1, 2)) for _ in range(N)]
    opp = ["self"] * N
    for i in range(N):
        if random.random() < greedy_prob:
            opp[i] = random.choice(strategies)  # e.g. "aggro", "midrange", etc.
    mh = [main.initial_state() for _ in range(N)]  # main-seat LSTM state per env
    oh = [main.initial_state() for _ in range(N)]  # self-opponent LSTM state per env
    last = [None] * N
    total = 0

    while total < steps:
        main_ids, self_ids, greedy_ids = [], [], []
        for i in range(N):
            if cur[i].player == main_seat[i]:
                main_ids.append(i)
            elif opp[i] != "self":
                greedy_ids.append(i)
            else:
                self_ids.append(i)

        cmds: list[str | None] = [None] * N

        if main_ids:
            idxs, lps, vs, nhc = batched_act(main, [cur[i] for i in main_ids], [mh[i] for i in main_ids], device)
            for k, i in enumerate(main_ids):
                t = T[i]
                t["tok"].append(cur[i].tokens); t["cid"].append(cur[i].card_ids)
                t["prv"].append(cur[i].priv); t["act"].append(cur[i].actions)
                t["idx"].append(idxs[k]); t["lp"].append(lps[k]); t["v"].append(vs[k])
                t["r"].append(0.0); t["d"].append(0.0)
                t["hin"].append(mh[i][0].squeeze(0).numpy()); t["cin"].append(mh[i][1].squeeze(0).numpy())
                t["phi"].append(cur[i].potential)
                last[i] = len(t["idx"]) - 1
                mh[i] = nhc[k]
                cmds[i] = f"step {idxs[k]}"
                total += 1

        if self_ids:
            idxs, _, _, nhc = batched_act(main, [cur[i] for i in self_ids], [oh[i] for i in self_ids], device)
            for k, i in enumerate(self_ids):
                oh[i] = nhc[k]
                cmds[i] = f"step {idxs[k]}"

        for i in greedy_ids:
            cmds[i] = f"step_greedy {opp[i]}"

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
                opp[i] = random.choice(strategies) if random.random() < greedy_prob else "self"
                mh[i] = main.initial_state(); oh[i] = main.initial_state()
                last[i] = None

    # ---- per-env shaping + GAE, then concatenate for the update ----
    tok_all, cid_all, prv_all, act_all, idx_all, lp_all, hin_all, cin_all = ([] for _ in range(8))
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
        tok_all += t["tok"]; cid_all += t["cid"]; prv_all += t["prv"]; act_all += t["act"]
        idx_all += t["idx"]; lp_all += t["lp"]; hin_all += t["hin"]; cin_all += t["cin"]

    batch = dict(tok=tok_all, cid=cid_all, prv=prv_all, act=act_all, idx=idx_all, lp=lp_all,
                 hin=hin_all, cin=cin_all, adv=np.concatenate(adv_all), ret=np.concatenate(ret_all))
    return batch, ep_returns


def run_eval(env, policy, episodes, opponent, eval_seed, argmax=True, lookahead=False):
    """One benchmark: returns (rate, lo, hi) with a Wilson 95% interval.

    Every RNG evaluation touches is rebuilt from `eval_seed` on each call — the local `rng`
    (seat + random opponent) and torch's global RNG (used by `dist.sample()` when argmax is
    off) — so the reported rate moves only when the policy does, not when the draw changes."""
    torch.manual_seed(eval_seed)
    wins, n = evaluate(env, policy, episodes, opponent, rng=random.Random(eval_seed),
                       argmax=argmax, lookahead=lookahead)
    lo, hi = wilson_ci(wins, n)
    return wins / max(n, 1), lo, hi


def fmt_eval(rate, lo, hi):
    return f"{rate:.3f} [{lo:.3f}-{hi:.3f}]"


def cosine_lr(init_lr, decay_to, total_iters, current_iter, warmup_iters=0):
    """Linear warmup then cosine annealing from init_lr down to decay_to.

    Warmup matters for the large presets: `100x` collapsed into a degenerate solution at
    lr=1e-3 with no warmup (BC loss flat from epoch 2), which is why its scaling result could
    not be read as a capacity result. Big transformers need the first few hundred steps at a
    small LR before the full rate is safe."""
    if warmup_iters > 0 and current_iter <= warmup_iters:
        return init_lr * current_iter / warmup_iters
    progress = min((current_iter - warmup_iters) / max(total_iters - warmup_iters, 1), 1.0)
    return decay_to + 0.5 * (init_lr - decay_to) * (1.0 + math.cos(math.pi * progress))


def train(args):
    # "all" expands to every SabberStone heuristic (Phase 5: opponent diversity). Training
    # against one opponent risks learning a single exploit rather than general play.
    if args.opponent_strategies.strip() == "all":
        args.opponent_strategies = list(OPPONENT_STRATEGIES)
    else:
        args.opponent_strategies = [s.strip() for s in args.opponent_strategies.split(",")]
    device = resolve_device(args.device)

    if args.eval_seed in range(args.seed, args.seed + args.num_envs):
        raise SystemExit(f"--eval-seed {args.eval_seed} collides with the training seeds "
                         f"{args.seed}..{args.seed + args.num_envs - 1}; evaluation would not be held out")

    # Evaluation gets its own env process, seeded from --eval-seed. It must NOT be a training
    # env: those have had their RNG stream advanced by rollouts, so eval games were previously
    # neither held out nor reproducible.
    eval_env = SabberEnv(seed=args.eval_seed, fixed_deck=args.fixed_deck,
                         dll=args.dll, dotnet=args.dotnet, potential=args.potential)
    # Vocab comes from the env, not a constant: the embedding must match the indices the env
    # actually emits, and a checkpoint is only loadable into a model built with the same vocab.
    card_vocab = 0 if args.no_card_emb else eval_env.card_vocab()
    main = ActorCritic(size=args.size, card_vocab=card_vocab, card_dim=args.card_dim).to(device)
    print(f"card embedding: vocab={card_vocab} dim={args.card_dim if card_vocab else 0}  "
          f"params={sum(p.numel() for p in main.parameters()):,}", flush=True)
    if args.resume or args.eval_only:
        ckpt = args.resume or args.eval_only
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        main.load_state_dict(state)
        print(f"loaded checkpoint from {ckpt}", flush=True)

    if args.eval_only:
        r = run_eval(eval_env, main, args.eval_episodes, "random", args.eval_seed, not args.eval_sample, args.lookahead)
        g = run_eval(eval_env, main, args.eval_greedy_episodes, "greedy", args.eval_seed, not args.eval_sample, args.lookahead)
        print(f"vs random {fmt_eval(*r)}  |  vs greedy {fmt_eval(*g)}"
              f"   (eval-seed {args.eval_seed}, n={args.eval_episodes}/{args.eval_greedy_episodes})")
        eval_env.close()
        return

    vec = VecEnv(args.num_envs, seed0=args.seed, fixed_deck=args.fixed_deck,
                 dll=args.dll, dotnet=args.dotnet, potential=args.potential)

    opt = torch.optim.Adam(main.parameters(), lr=args.lr)

    best_greedy_wr = -1.0
    best_path = args.out.replace(".pt", "_best.pt")

    for it in range(1, args.iters + 1):
        # ---- LR schedule (cosine annealing) ----
        current_lr = cosine_lr(args.lr, args.lr * args.lr_decay, args.iters, it, args.warmup_iters)
        for pg in opt.param_groups:
            pg["lr"] = current_lr

        # ---- Entropy schedule (linear decay) ----
        progress = (it - 1) / max(args.iters - 1, 1)
        ent_coef = args.ent_start + (args.ent_end - args.ent_start) * progress

        # ---- Shaping schedule (linear decay) ----
        # Phase 2: the potential is greedy's own objective, so a constant coefficient anchors the
        # final policy at heuristic level. Annealing to --shaping-end lets shaping bootstrap early
        # learning, then hands the policy back to the terminal win/loss signal.
        shaping_coef = args.shaping_coef + (args.shaping_end - args.shaping_coef) * progress

        # ---- Progressive curriculum: ramp greedy-prob over first part of training ----
        if args.greedy_end > args.greedy_start:
            curriculum_progress = min(progress / args.curriculum_frac, 1.0)
            greedy_prob = args.greedy_start + (args.greedy_end - args.greedy_start) * curriculum_progress
        else:
            greedy_prob = args.greedy_prob

        t0 = time.time()
        b, ep = collect_vec(vec, main, greedy_prob, args.steps, args.gamma, shaping_coef,
                           args.opponent_strategies, device)
        sps = len(b["idx"]) / max(time.time() - t0, 1e-9)

        tok_t, tmask_t = (t.to(device) for t in pad(b["tok"], TOKEN_DIM))
        cid_t = pad_ids(b["cid"], tok_t.shape[1]).to(device)
        act_t, amask_t = (t.to(device) for t in pad(b["act"], ACT_DIM))
        priv_t = torch.from_numpy(np.asarray(b["prv"], np.float32)).to(device)
        hist_t = torch.from_numpy(np.asarray(b["hin"], np.float32)).to(device)
        mask_t = torch.from_numpy(np.asarray(b["cin"], bool)).to(device)
        idx_t = torch.tensor(b["idx"]).to(device)
        oldlp_t = torch.tensor(b["lp"]).to(device)
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        adv_t = torch.from_numpy(adv).to(device)
        ret_t = torch.from_numpy(b["ret"]).to(device)

        n = len(b["idx"])
        pol_losses, val_losses, ents = [], [], []
        for _ in range(args.epochs):
            for mb in torch.randperm(n, device=device).split(args.minibatch):
                logits, v, _, _ = main(tok_t[mb], tmask_t[mb], hist_t[mb], mask_t[mb],
                                       priv_t[mb], act_t[mb], amask_t[mb], cid_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(idx_t[mb]) - oldlp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[mb]
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = ((v - ret_t[mb]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pol_loss + args.vf * val_loss - ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(main.parameters(), 0.5); opt.step()
                pol_losses.append(pol_loss.item())
                val_losses.append(val_loss.item())
                ents.append(ent.item())

        tr_wr = float(np.mean([r > 0 for r in ep])) if ep else float("nan")
        avg_pol = float(np.mean(pol_losses))
        avg_val = float(np.mean(val_losses))
        avg_ent = float(np.mean(ents))
        line = (f"iter {it:3d}  envs {args.num_envs}  transitions {n:5d}  {sps:6.0f} steps/s  "
                f"episodes {len(ep):3d}  train_wr {tr_wr:.3f}  "
                f"lr {current_lr:.1e}  ent_coef {ent_coef:.4f}  shp {shaping_coef:.3f}  "
                f"pol {avg_pol:.3f}  val {avg_val:.3f}  ent {avg_ent:.3f}")
        if it % args.eval_every == 0:
            r = run_eval(eval_env, main, args.eval_episodes, "random", args.eval_seed, not args.eval_sample, args.lookahead)
            g = run_eval(eval_env, main, args.eval_greedy_episodes, "greedy", args.eval_seed, not args.eval_sample, args.lookahead)
            wr_g = g[0]
            line += f"  [vs random {fmt_eval(*r)} | vs greedy {fmt_eval(*g)}]"
            # Track best model by vs-greedy win rate. Taking a max over repeated noisy evals
            # biases high, so this is only meaningful when --eval-greedy-episodes is large
            # enough that the interval is tighter than the differences being selected on.
            if wr_g > best_greedy_wr:
                best_greedy_wr = wr_g
                torch.save(main.state_dict(), best_path)
                line += f"  *best*"
        print(line, flush=True)

    torch.save(main.state_dict(), args.out)
    print(f"saved final policy -> {args.out}")
    if best_greedy_wr >= 0:
        print(f"saved best policy (vs greedy {best_greedy_wr:.3f}) -> {best_path}")
        print("note: the *best* score is a max over repeated evals and is biased high; "
              "re-measure the checkpoint with --eval-only before quoting it", flush=True)
    vec.close()
    eval_env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--steps", type=int, default=8192, help="main transitions per iteration")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-iters", type=int, default=0,
                    help="linear LR warmup over the first N iters (Phase 4). Recommended for "
                         "large/xl/100x, which diverge or collapse without it")
    ap.add_argument("--lr-decay", type=float, default=0.1,
                    help="cosine-anneal LR to lr*lr_decay over training")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent-start", type=float, default=0.02,
                    help="entropy bonus at start (decays linearly to --ent-end)")
    ap.add_argument("--ent-end", type=float, default=0.002,
                    help="entropy bonus at end of training")
    ap.add_argument("--shaping-coef", type=float, default=0.05,
                    help="potential-based shaping coefficient at the START of training")
    ap.add_argument("--shaping-end", type=float, default=0.05,
                    help="shaping coefficient at the END (linear anneal from --shaping-coef; "
                         "set 0 to hand the policy back to the pure win/loss signal)")
    ap.add_argument("--potential", type=str, default="midrange",
                    choices=["midrange", "board", "none"],
                    help="shaping potential: midrange = greedy's own score function (anchors the "
                         "policy to greedy), board = health+material differential, none = 0")
    ap.add_argument("--greedy-prob", type=float, default=0.7,
                    help="prob. opponent is the greedy heuristic (constant; overridden by curriculum)")
    ap.add_argument("--greedy-start", type=float, default=0.3,
                    help="curriculum: initial greedy prob (ramps to --greedy-end)")
    ap.add_argument("--greedy-end", type=float, default=0.7,
                    help="curriculum: final greedy prob (set equal to --greedy-start to disable)")
    ap.add_argument("--curriculum-frac", type=float, default=0.5,
                    help="fraction of training over which greedy prob ramps (0.5 = first half)")
    ap.add_argument("--opponent-strategies", type=str, default="midrange",
                    help="comma-separated greedy opponent strategies, or 'all' for every "
                         "heuristic (midrange,aggro,control,fatigue,ramp)")
    ap.add_argument("--size", type=str, default="small",
                    choices=["small", "medium", "large", "xl", "100x"],
                    help="model scale: small (173k), medium (532k), large (1.2M), xl (1.7M), "
                         "100x (17.6M). small is the recommended default — see README "
                         "'Model scale: bigger is not better here'")
    ap.add_argument("--fixed-deck", action="store_true")
    ap.add_argument("--eval-only", type=str, default=None,
                    help="evaluate a checkpoint and exit (no training)")
    ap.add_argument("--eval-every", type=int, default=10)
    # Wilson 95% half-width at p=0.5: n=50 -> ±0.134, n=200 -> ±0.069, n=400 -> ±0.049.
    # Selecting a *best* checkpoint on 50-game evals reliably picked noise (8-18 point drops on
    # re-measure). 200 narrows it a lot but still cannot resolve differences under ~0.14.
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--eval-greedy-episodes", type=int, default=200)
    ap.add_argument("--eval-seed", type=int, default=100000,
                    help="seed for the dedicated eval env + eval RNG; must not overlap the "
                         "training seeds (--seed .. --seed+--num-envs-1) or eval is not held out")
    ap.add_argument("--card-dim", type=int, default=16,
                    help="card-identity embedding width (Phase 1)")
    ap.add_argument("--no-card-emb", action="store_true",
                    help="ablation: disable the card embedding, restoring the card-blind "
                         "18-float-only observation")
    ap.add_argument("--lookahead", action="store_true",
                    help="Phase 3: evaluate with critic-scored one-ply lookahead, matching the "
                         "search depth greedy already has. Costs N engine clones per decision")
    ap.add_argument("--eval-sample", action="store_true",
                    help="sample eval actions instead of taking argmax. Adds large variance "
                         "(13-point swings between identical runs); argmax is the default")
    ap.add_argument("--device", type=str, default="auto", choices=DEVICE_CHOICES,
                    help="compute device (auto = cuda > mps > cpu)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="ppo_policy.pt")
    ap.add_argument("--dll", type=str, default=None)
    ap.add_argument("--dotnet", type=str, default="dotnet")
    ap.add_argument("--resume", type=str, default=None,
                    help="path to a pre-trained .pt checkpoint to fine-tune from")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
