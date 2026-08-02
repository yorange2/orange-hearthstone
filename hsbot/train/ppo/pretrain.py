"""Pre-train the policy via behavioral cloning on greedy heuristic data.

Phase 1 — generate data: greedy-vs-greedy self-play, recording (state, action) pairs.
Phase 2 — pre-train: cross-entropy on greedy action choices.
Phase 3 — save: the pre-trained model is a strong starting point for PPO fine-tuning.

    python pretrain.py --games 2000 --out pretrained.pt
"""
from __future__ import annotations
import argparse, os, time

from device import resolve_device, CHOICES as DEVICE_CHOICES  # before torch: sets MPS fallback

import numpy as np
import torch
import torch.nn as nn

import card_text as card_text_mod
from env import SabberEnv, DECK_MODES, deck_mode, TOKEN_DIM, ACT_DIM, PRIV_DIM
from ppo import ActorCritic, pad, pad_ids


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
            cur_card_ids = obs.card_ids.copy()
            cur_actions = obs.actions.copy()
            # Play greedy from this state
            obs, done, _winner = env.step_greedy()
            # obs.greedy_action is the action greedy chose for (cur_tokens, cur_actions)
            data.append(dict(tokens=cur_tokens, card_ids=cur_card_ids, actions=cur_actions,
                             idx=obs.greedy_action))
            if done:
                break
        if (g + 1) % 500 == 0:
            print(f"  generated {g + 1}/{num_games} games, {len(data)} data points", flush=True)
    return data


def load_cache(path, key):
    """Return (data, vocab, ids_hash) from `path`, or None if absent or generated differently.

    The key is the generation protocol (games/seed/deck); a mismatch regenerates rather than
    silently training on data from another arm's protocol."""
    if not path or not os.path.exists(path):
        return None
    blob = torch.load(path, weights_only=False)
    if blob["key"] != key:
        print(f"cache {path} was generated with {blob['key']}, need {key} — regenerating",
              flush=True)
        return None
    return blob["data"], blob["vocab"], blob["ids_hash"]


def save_cache(path, key, data, vocab, ids_hash):
    if not path:
        return
    torch.save(dict(key=key, data=data, vocab=vocab, ids_hash=ids_hash), path)
    print(f"cached {len(data)} examples -> {path}", flush=True)


def pretrain(policy, data, epochs, batch_size, lr, device, warmup_steps=0):
    """Behavioral cloning: cross-entropy loss on predicting greedy's action.

    `warmup_steps` linearly ramps the LR from 0 over the first N optimizer steps. Off by
    default (the `small` recipe never needed it); required for the larger presets, whose
    loss otherwise flatlines from epoch 2 at lr=1e-3 — see README "Model scale"."""
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup_steps))
             if warmup_steps > 0 else None)
    n = len(data)
    losses = []

    for ep in range(epochs):
        perm = torch.randperm(n)
        epoch_losses = []
        for mb in perm.split(batch_size):
            batch = [data[i] for i in mb.tolist()]
            tok, tmask = pad([b["tokens"] for b in batch], TOKEN_DIM)
            cid = pad_ids([b["card_ids"] for b in batch], tok.shape[1])
            act, amask = pad([b["actions"] for b in batch], ACT_DIM)
            target = torch.tensor([b["idx"] for b in batch])
            priv = torch.zeros(len(batch), PRIV_DIM, dtype=torch.float32)  # dummy priv (policy logits ignore it)
            hist, mask = policy.initial_state(len(batch))  # BC treats each decision independently: empty history

            tok, tmask, cid = tok.to(device), tmask.to(device), cid.to(device)
            act, amask = act.to(device), amask.to(device)
            target = target.to(device)
            priv = priv.to(device)
            hist, mask = hist.to(device), mask.to(device)

            logits, _, _, _ = policy(tok, tmask, hist, mask, priv, act, amask, cid)
            loss = nn.CrossEntropyLoss()(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            epoch_losses.append(loss.item())

        avg = float(np.mean(epoch_losses))
        losses.append(avg)
        acc = evaluate_imitation(policy, data[:min(1000, n)], device)
        cur_lr = opt.param_groups[0]["lr"]
        print(f"  epoch {ep + 1:3d}/{epochs}  loss {avg:.4f}  top-1 acc {acc:.3f}  lr {cur_lr:.2e}",
              flush=True)

    return losses


@torch.no_grad()
def evaluate_imitation(policy, data, device="cpu"):
    """Top-1 accuracy: does the policy pick the same action as greedy?"""
    policy.eval()
    correct = 0
    for i in range(0, len(data), 64):
        batch = data[i:i + 64]
        tok, tmask = pad([b["tokens"] for b in batch], TOKEN_DIM)
        cid = pad_ids([b["card_ids"] for b in batch], tok.shape[1]).to(device)
        act, amask = pad([b["actions"] for b in batch], ACT_DIM)
        target = np.array([b["idx"] for b in batch])
        priv = torch.zeros(len(batch), PRIV_DIM, device=device)
        hist, mask = (t.to(device) for t in policy.initial_state(len(batch)))
        logits, _, _, _ = policy(tok.to(device), tmask.to(device), hist, mask,
                                  priv, act.to(device), amask.to(device), cid)
        pred = logits.argmax(dim=1).cpu().numpy()
        correct += (pred == target).sum()
    policy.train()
    return correct / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="linear LR warmup over the first N optimizer steps. Recommended for "
                         "large/xl/100x, whose BC loss flatlines at lr=1e-3 without it")
    ap.add_argument("--cache", type=str, default=None,
                    help="path to cache the generated (state, greedy-action) pairs. Reused only "
                         "if games/seed/deck match, so two arms can share identical BC data")
    ap.add_argument("--size", type=str, default="small",
                    choices=["small", "medium", "large", "xl", "100x"])
    ap.add_argument("--device", type=str, default="auto", choices=DEVICE_CHOICES,
                    help="compute device (auto = cuda > mps > cpu)")
    ap.add_argument("--card-dim", type=int, default=16,
                    help="width the frozen card-text vectors are projected to")
    ap.add_argument("--out", type=str, default="pretrained.pt")
    ap.add_argument("--fixed-deck", action="store_true",
                    help="shorthand for --deck fixed")
    ap.add_argument("--deck", type=str, default=None, choices=DECK_MODES,
                    help="fixed (Mage mirror) | variedmirror (random deck, same both seats) | random")
    ap.add_argument("--card-text", type=str, default=None,
                    help="path to a card_text.py .npz (default: card_text_emb.npz beside this "
                         "script); card text is the only card-identity path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dll", type=str, default=None)
    ap.add_argument("--dotnet", type=str, default="dotnet")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # phase="batch": BC is big fixed-shape minibatches, where MPS wins ~2.5x. The generic
    # rollout warning would tell the reader the opposite of the right thing here.
    device = resolve_device(args.device, phase="batch")  # up front so a bad device fails before phase 1

    # Phase 1: generate data (or reuse a cache generated under the same protocol)
    key = dict(games=args.games, seed=args.seed, deck=deck_mode(args))
    cached = load_cache(args.cache, key)
    if cached is not None:
        data, env_vocab, env_ids_hash = cached
        print(f"Phase 1: reusing {len(data)} cached examples from {args.cache}", flush=True)
    else:
        print(f"Phase 1: generating data ({args.games} greedy-vs-greedy games)...", flush=True)
        t0 = time.time()
        env = SabberEnv(seed=args.seed, deck=deck_mode(args), dll=args.dll, dotnet=args.dotnet)
        env_vocab = env.card_vocab()
        env_ids_hash = env.card_ids_hash()
        data = generate_data(env, args.games)
        env.close()
        dt = time.time() - t0
        print(f"Generated {len(data)} training examples in {dt:.1f}s", flush=True)
        save_cache(args.cache, key, data, env_vocab, env_ids_hash)

    # Phase 2: pre-train
    print(f"\nPhase 2: behavioral cloning ({args.epochs} epochs, device={device})...", flush=True)
    card_vocab, card_text = card_text_mod.resolve(args.card_text, env_vocab, env_ids_hash)
    policy = ActorCritic(size=args.size, card_dim=args.card_dim, card_text=card_text).to(device)
    params = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Model: {args.size} ({params:,} params, {trainable:,} trainable)", flush=True)
    print(card_text_mod.describe(card_vocab, card_text, args.card_dim), flush=True)
    pretrain(policy, data, args.epochs, args.batch_size, args.lr, device=device,
             warmup_steps=args.warmup_steps)

    # Phase 3: save
    torch.save(policy.state_dict(), args.out)
    print(f"\nSaved pre-trained model -> {args.out}", flush=True)

    # Quick sanity check
    policy.eval()
    acc = evaluate_imitation(policy, data[:min(500, len(data))], device)
    print(f"Final top-1 accuracy on 500 samples: {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
