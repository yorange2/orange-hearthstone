"""Compare card-text encoders on probes that matter for play, not on eyeballed neighbours.

TF-IDF and a pretrained sentence encoder fail in *opposite* directions on this corpus, so
"which is better" is not answerable from nearest-neighbour lists alone:

  - TF-IDF keeps numbers as distinct tokens, so "Deal $6 damage" and "Deal $3 damage" separate --
    but it cannot know that "Transform a minion into a Sheep" and "...into a Frog" are the same
    effect, because the payload words differ.
  - A sentence encoder knows Sheep and Frog are both transform payloads, but is famously weak on
    numbers and will tend to collapse the damage tiers.

Both matter here. Damage magnitude decides lethal; effect class decides what a card is for. This
script scores each encoder on both axes plus the degenerate-cluster check that TF-IDF is known to
fail on textless vanilla cards.

    python card_text_compare.py --a card_text_emb.npz --b card_text_st.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from env import SabberEnv


def cos(e, i, j):
    return float(e[i] @ e[j])


def report(name, emb, byname, cards, notext_idx):
    print(f"\n===== {name}  (dim {emb.shape[1]}) =====")

    # 1. Numeric sensitivity: same template, different magnitude. Should be similar but NOT
    #    identical -- a model that returns ~1.00 cannot tell 3 damage from 6, which is the
    #    difference between lethal and not.
    pairs = [("Fireball", "Pyroblast"), ("Fireball", "Frostbolt"), ("Flamestrike", "Arcane Explosion")]
    print("  numeric/magnitude sensitivity (want < ~0.98: distinguishable):")
    for a, b in pairs:
        if a in byname and b in byname:
            print(f"    {a:<14} vs {b:<18} cos {cos(emb, byname[a], byname[b]):.3f}")

    # 2. Effect-class recognition: different payload words, same mechanic. Want HIGH.
    print("  same-effect recognition (want high):")
    for a, b in [("Polymorph", "Hex"), ("Assassinate", "Execute"), ("Shadow Word: Pain", "Shadow Word: Death")]:
        if a in byname and b in byname:
            print(f"    {a:<18} vs {b:<20} cos {cos(emb, byname[a], byname[b]):.3f}")

    # 3. Unrelated pair as a floor -- if this is also high, the space is collapsed and the two
    #    numbers above mean nothing.
    print("  unrelated floor (want low):")
    for a, b in [("Fireball", "Chillwind Yeti"), ("Polymorph", "Fiery War Axe")]:
        if a in byname and b in byname:
            print(f"    {a:<18} vs {b:<20} cos {cos(emb, byname[a], byname[b]):.3f}")

    # 4. Textless vanilla cards: their doc is only a name, so they risk collapsing into one
    #    indistinguishable blob. Quantified rather than spot-checked.
    sub = emb[notext_idx]
    with np.errstate(all="ignore"):
        s = sub @ sub.T
    iu = np.triu_indices(len(notext_idx), 1)
    print(f"  textless cards ({len(notext_idx)}): mean pairwise cos {s[iu].mean():.3f}, "
          f"frac>0.99 {(s[iu] > 0.99).mean():.3f}  (want low)")

    # 5. Global spread: mean cosine over a random sample. Near 0 is a well-spread space.
    rng = np.random.default_rng(0)
    idx = rng.choice(np.arange(1, len(emb)), size=1500, replace=False)
    sub = emb[idx]
    with np.errstate(all="ignore"):
        s = sub @ sub.T
    iu = np.triu_indices(len(idx), 1)
    print(f"  global spread: mean pairwise cos {s[iu].mean():+.3f} (want near 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="card_text_emb.npz", help="first matrix (e.g. tfidf)")
    ap.add_argument("--b", default=None, help="second matrix (e.g. sentence-transformer)")
    args = ap.parse_args()

    env = SabberEnv(seed=0)
    try:
        cards = env.cards()
    finally:
        env.close()
    byname = {c["name"]: i for i, c in enumerate(cards) if c["name"]}
    notext = [i for i, c in enumerate(cards) if i and not c["text"]]

    for path in filter(None, (args.a, args.b)):
        z = np.load(path, allow_pickle=True)
        emb = z["emb"].astype(np.float64)
        # rows are already L2-normalised by both builders, but re-normalise so cosine == dot
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        report(f"{path}  encoder={str(z['encoder'])}", emb, byname, cards, notext)


if __name__ == "__main__":
    main()
