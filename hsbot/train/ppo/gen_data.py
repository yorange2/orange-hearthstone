"""Generate supervised value-net training data using the stronger PPO policy.

Loads a saved PPO checkpoint and drives self-play games (the policy plays both seats), then
records every main-seat decision point (flat 144-float features, FEATURES.md v1) with the
eventual game winner as label. Output is JSONL consumable by ../train.py.

    python gen_data.py --games 2000 --model ppo_vec_long.pt --out ppo_selfplay.jsonl
"""
from __future__ import annotations
import argparse, json, random, sys, time

import torch

from env import SabberEnv
from ppo import ActorCritic, act


@torch.no_grad()
def generate(env, policy, num_games: int, out_path: str):
    rows = 0
    with open(out_path, "w") as fh:
        for g in range(num_games):
            obs = env.reset()
            seat = random.choice((1, 2))
            h1, c1 = policy.initial_state()  # player 1 LSTM
            h2, c2 = policy.initial_state()  # player 2 LSTM
            snaps: list[list[float]] = []    # flat features seen by the main seat

            while True:
                # snapshot the MAIN seat's view before they act
                if obs.player == seat:
                    snaps.append(obs.flat.tolist())

                # act with the recurrent policy for the current mover
                if obs.player == 1:
                    i, _, _, h1, c1 = act(policy, obs, h1, c1)
                else:
                    i, _, _, h2, c2 = act(policy, obs, h2, c2)

                obs, done, winner = env.step(i)
                if done:
                    label = 1 if winner == seat else 0
                    for feats in snaps:
                        fh.write(json.dumps({"Features": feats, "Label": label}))
                        fh.write("\n")
                        rows += 1
                    break

            if (g + 1) % 200 == 0:
                print(f"  game {g + 1}/{num_games}  rows {rows}", file=sys.stderr, flush=True)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--model", type=str, default="ppo_vec_long.pt")
    ap.add_argument("--out", type=str, default="ppo_selfplay.jsonl")
    ap.add_argument("--fixed-deck", action="store_true")
    ap.add_argument("--dotnet", type=str, default="dotnet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    env = SabberEnv(seed=args.seed, fixed_deck=args.fixed_deck, dotnet=args.dotnet)
    policy = ActorCritic()
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    policy.eval()

    t0 = time.time()
    rows = generate(env, policy, args.games, args.out)
    dt = time.time() - t0
    print(f"done: {args.games} games -> {rows} rows in {dt:.1f}s "
          f"({args.games / max(dt, 1e-9):.0f} games/s, {rows / max(dt, 1e-9):.0f} rows/s)",
          file=sys.stderr)
    env.close()


if __name__ == "__main__":
    main()
