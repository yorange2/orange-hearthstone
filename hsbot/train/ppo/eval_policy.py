"""Evaluate a saved PPO policy checkpoint against greedy and random opponents.

Usage:
    python eval_policy.py ppo_curriculum_best.pt --games 200
    python eval_policy.py ppo_curriculum.pt --games 100 --opponents greedy
"""
from __future__ import annotations
import argparse, sys, time
from collections import Counter

import numpy as np
import torch

from env import SabberEnv
from ppo import ActorCritic, evaluate


def eval_with_confidence(env, policy, games: int, opponent: str) -> dict:
    """Run eval and report win rate with Wilson score interval (95% CI)."""
    t0 = time.time()
    wr = evaluate(env, policy, games, opponent)
    dt = time.time() - t0

    # Wilson score interval for 95% confidence
    n = games
    p = wr
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z / denom * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)

    return dict(wr=wr, ci_lo=lo, ci_hi=hi, games=n, time_s=dt, gps=games / max(dt, 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str, help="path to .pt checkpoint")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--opponents", type=str, default="both",
                    choices=["both", "greedy", "random"])
    ap.add_argument("--fixed-deck", action="store_true")
    ap.add_argument("--size", type=str, default="small",
                    choices=["small", "medium", "large"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dll", type=str, default=None)
    ap.add_argument("--dotnet", type=str, default="dotnet")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = SabberEnv(seed=args.seed + 1000, fixed_deck=args.fixed_deck,
                    dll=args.dll, dotnet=args.dotnet)
    policy = ActorCritic(size=args.size)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    params = sum(p.numel() for p in policy.parameters())

    print(f"Model: {args.model}  ({params:,} params)")
    print(f"Games per opponent: {args.games}")
    print()

    for opp in (["greedy", "random"] if args.opponents == "both" else [args.opponents]):
        r = eval_with_confidence(env, policy, args.games, opp)
        print(f"vs {opp:6s}:  {r['wr']:.3f}  [95% CI: {r['ci_lo']:.3f} – {r['ci_hi']:.3f}]  "
              f"({r['time_s']:.1f}s, {r['gps']:.1f} games/s)")

    env.close()


if __name__ == "__main__":
    main()
