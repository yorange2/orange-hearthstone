"""Pre-train the policy via behavioral cloning on greedy heuristic data.

Phase 1 — generate data: greedy-vs-greedy self-play, recording (state, action) pairs.
Phase 2 — pre-train: cross-entropy on greedy action choices.
Phase 3 — save: the pre-trained model is a strong starting point for PPO fine-tuning.

    python pretrain.py --games 2000 --out pretrained_small.pt
    python pretrain.py --games 2000 --size medium --out pretrained_medium.pt
"""
from __future__ import annotations
import argparse, random, time

import numpy as np
import torch
import torch.nn as nn

from env import SabberEnv, TOKEN_DIM, ACT_DIM
from ppo import ActorCritic, pad


def generate_data(env, num_games: int) -> list[dict]:
    """Run greedy-vs-greedy games, recording (tokens, actions, greedy_idx) per decision.

    Protocol: call step_greedy() — env plays greedy action i from current state,
    returns next observation whose greedy_action == i. Pair: (prev_state, i)."""
    data: list[dict] = []
    for g in range(num_games):
        obs = env.reset()
        while True:
            # Record the current state
            cur_tokens = obs.tokens.copy()
            cur_actions = obs.actions.copy()
            # Play greedy from this state
            obs, done, _winner = env.step_greedy()
            # obs.greedy_action is the action greedy chose for (cur_tokens, cur_actions)
            data.append(dict(tokens=cur_tokens, actions=cur_actions,
                             idx=obs.greedy_action))
            if done:
                break
        if (g + 1) % 500 == 0:
            print(f"  generated {g + 1}/{num_games} games, {len(data)} data points", flush=True)
    return data


def pretrain(policy, data, epochs, batch_size, lr, device):
    """Behavioral cloning: cross-entropy loss on predicting greedy's action."""
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(data)
    losses = []

    for ep in range(epochs):
        perm = torch.randperm(n)
        epoch_losses = []
        for mb in perm.split(batch_size):
            batch = [data[i] for i in mb.tolist()]
            tok, tmask = pad([b["tokens"] for b in batch], TOKEN_DIM)
            act, amask = pad([b["actions"] for b in batch], ACT_DIM)
            target = torch.tensor([b["idx"] for b in batch])
            priv = torch.zeros(len(batch), 8, dtype=torch.float32)  # dummy priv (unused in policy)
            hin = torch.zeros(len(batch), policy.hdim)
            cin = torch.zeros(len(batch), policy.hdim)

            tok, tmask = tok.to(device), tmask.to(device)
            act, amask = act.to(device), amask.to(device)
            target = target.to(device)
            priv = priv.to(device)
            hin, cin = hin.to(device), cin.to(device)

            logits, _, _, _ = policy(tok, tmask, hin, cin, priv, act, amask)
            loss = nn.CrossEntropyLoss()(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        avg = float(np.mean(epoch_losses))
        losses.append(avg)
        acc = evaluate_imitation(policy, data[:min(1000, n)])
        print(f"  epoch {ep + 1:3d}/{epochs}  loss {avg:.4f}  top-1 acc {acc:.3f}", flush=True)

    return losses


@torch.no_grad()
def evaluate_imitation(policy, data):
    """Top-1 accuracy: does the policy pick the same action as greedy?"""
    policy.eval()
    correct = 0
    for i in range(0, len(data), 64):
        batch = data[i:i + 64]
        tok, tmask = pad([b["tokens"] for b in batch], TOKEN_DIM)
        act, amask = pad([b["actions"] for b in batch], ACT_DIM)
        target = np.array([b["idx"] for b in batch])
        priv = torch.zeros(len(batch), 8)
        hin = torch.zeros(len(batch), policy.hdim)
        cin = torch.zeros(len(batch), policy.hdim)
        logits, _, _, _ = policy(tok, tmask, hin, cin, priv, act, amask)
        pred = logits.argmax(dim=1).numpy()
        correct += (pred == target).sum()
    policy.train()
    return correct / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=str, default="small",
                    choices=["small", "medium", "large"])
    ap.add_argument("--out", type=str, default="pretrained.pt")
    ap.add_argument("--fixed-deck", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dll", type=str, default=None)
    ap.add_argument("--dotnet", type=str, default="dotnet")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Phase 1: generate data
    print(f"Phase 1: generating data ({args.games} greedy-vs-greedy games)...", flush=True)
    t0 = time.time()
    env = SabberEnv(seed=args.seed, fixed_deck=args.fixed_deck, dll=args.dll, dotnet=args.dotnet)
    data = generate_data(env, args.games)
    env.close()
    dt = time.time() - t0
    print(f"Generated {len(data)} training examples in {dt:.1f}s", flush=True)

    # Phase 2: pre-train
    print(f"\nPhase 2: behavioral cloning ({args.epochs} epochs)...", flush=True)
    policy = ActorCritic(size=args.size)
    params = sum(p.numel() for p in policy.parameters())
    print(f"Model: {args.size} ({params:,} params)", flush=True)
    pretrain(policy, data, args.epochs, args.batch_size, args.lr, device="cpu")

    # Phase 3: save
    torch.save(policy.state_dict(), args.out)
    print(f"\nSaved pre-trained model -> {args.out}", flush=True)

    # Quick sanity check
    policy.eval()
    acc = evaluate_imitation(policy, data[:min(500, len(data))])
    print(f"Final top-1 accuracy on 500 samples: {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
