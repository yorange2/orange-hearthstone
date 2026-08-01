"""Card-text embeddings: turn each card's name + rules text into a fixed vector.

Why this instead of the `nn.Embedding(vocab, d)` card-ID table (Phase 1): an ID embedding has to
*learn* what each card does from gradients, and only 471 of the 8303 rows ever appear in
fixed-deck play. Every other row stays at its random init, so the policy is not merely ignorant
of an unseen card — it reads a meaningless vector for it, and does so silently. Card text is
known up front for all 8303, so reading meaning off the text gives every card a sensible vector
without a single game being played, and two cards that do the same thing land near each other
("Deal $6 damage" next to "Deal $4 damage") instead of having to learn that separately.

**Encoder choice.** The default is TF-IDF over cleaned text + SVD, with no model download and no
new dependency. That is a deliberate fit to the data rather than a compromise: Hearthstone rules
text is templated, not prose ("<b>Battlecry:</b> Deal $3 damage."), so lexical overlap really is
the semantics, and the vocabulary is small enough that SVD recovers the mechanic structure. It is
also fully deterministic and reproducible in CI, which a downloaded transformer is not. Pass
`--encoder st` to use a sentence-transformer instead if you want to test that tradeoff; it is
strictly optional and imported lazily.

**Alignment is the thing that can silently break.** The matrix rows must correspond to the
`CardVocab` indices the env emits. That is why the card list is pulled from the env over the
`cards` command rather than re-parsed from CardDefs.xml: a one-card difference between the two
pools shifts every row after it, and nothing crashes — the model just reads the wrong card's
meaning forever. The saved artifact records the vocab size and a hash of the id list so a
mismatched pairing is caught at load time instead.

Usage:
    python card_text.py --out card_text_emb.npz            # build (queries the env)
    python card_text.py --out card_text_emb.npz --dim 64   # narrower projection
"""
from __future__ import annotations

import argparse
import hashlib
import re

import numpy as np

from env import SabberEnv

# Markup that appears in the raw text and means nothing to a bag of words. `$`/`@` prefix numbers
# that scale with spell damage; the marker is dropped but the number kept, since "deal 6" vs
# "deal 3" is exactly the distinction an ID embedding could not make. `_` is a non-breaking space
# in the card data ("3_or less Attack").
_TAG_RE = re.compile(r"<[^>]+>")
_NUM_RE = re.compile(r"[$@](\d+)")
_NONWORD_RE = re.compile(r"[^a-z0-9\s]+")


def clean(name: str, text: str) -> str:
    """Card name + rules text -> a normalized token string.

    Name is included because ~1400 cards have no rules text at all (vanilla minions like Fiery
    War Axe); without it those rows would be identical empty vectors and indistinguishable.
    """
    s = f"{name}. {text}".lower()
    s = _TAG_RE.sub(" ", s)          # <b>Taunt</b> -> taunt
    s = _NUM_RE.sub(r"\1", s)        # $6 -> 6
    s = s.replace("_", " ").replace("\n", " ")
    s = _NONWORD_RE.sub(" ", s)
    return " ".join(s.split())


def tfidf_svd(docs: list[str], dim: int, seed: int = 0) -> np.ndarray:
    """Bag-of-words TF-IDF -> truncated SVD, in numpy alone.

    Deterministic: the vocabulary is sorted, and SVD is computed exactly rather than by a
    randomized solver, so two builds of the same card pool give byte-identical matrices.
    """
    vocab: dict[str, int] = {}
    for d in docs:
        for w in d.split():
            if w not in vocab:
                vocab[w] = 0
    words = sorted(vocab)
    index = {w: i for i, w in enumerate(words)}

    # float64 throughout: built once offline, so exactness is free and the SVD is better
    # conditioned for it.
    counts = np.zeros((len(docs), len(words)), dtype=np.float64)
    for r, d in enumerate(docs):
        for w in d.split():
            counts[r, index[w]] += 1.0

    # Sublinear TF damps "deal"/"minion" appearing many times in one card; IDF damps them
    # appearing on most cards. Both matter here because the templated phrasing repeats heavily.
    tf = np.log1p(counts)
    df = (counts > 0).sum(axis=0)
    idf = np.log((1.0 + len(docs)) / (1.0 + df)) + 1.0
    x = tf * idf

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-8)

    # Full SVD on ~8300 x ~5000 is affordable once, offline, and avoids the seed-dependence of a
    # randomized solver. Rows are then L2-normalized so the projection into the model sees a
    # consistent scale regardless of how much text a card has.
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)

    # numpy on macOS/Accelerate raises spurious divide-by-zero/overflow/invalid warnings out of
    # this matmul: the FP status flags leak rather than describing the operation. Verified benign
    # — the BLAS result is bit-identical (max abs diff 0.0) to the same product computed in 64-row
    # chunks, and every entry is finite. Silenced narrowly, here only, so a real numerical problem
    # elsewhere still surfaces.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        emb = x @ vt[:dim].T

    if not np.isfinite(emb).all():
        raise RuntimeError("card-text SVD produced non-finite values")

    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    return emb.astype(np.float32)


def sentence_transformer(docs: list[str], model: str) -> np.ndarray:
    """Optional pretrained encoder. Imported lazily so the default path needs no extra install."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    m = SentenceTransformer(model)
    emb = m.encode(docs, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def build(cards: list[dict], dim: int, encoder: str, st_model: str) -> dict:
    ids = [c["id"] for c in cards]
    docs = [clean(c["name"], c["text"]) for c in cards]
    docs[0] = ""  # row 0 is the reserved none/unknown slot; keep it empty and zeroed below

    if encoder == "tfidf":
        emb = tfidf_svd(docs, dim)
    elif encoder == "st":
        emb = sentence_transformer(docs, st_model)
    else:
        raise ValueError(f"unknown encoder {encoder!r}")

    emb[0] = 0.0  # padding_idx row must be zero, and stay zero
    return dict(emb=emb, ids=np.array(ids, dtype=object), vocab=len(ids),
                ids_hash=ids_hash(ids), encoder=encoder)


def load(path: str, expect_vocab: int | None = None, expect_ids_hash: str | None = None):
    """Load a built matrix, refusing a pairing that does not match the env that will use it."""
    z = np.load(path, allow_pickle=True)
    emb = z["emb"].astype(np.float32)
    vocab = int(z["vocab"])
    h = str(z["ids_hash"])

    if expect_vocab is not None and vocab != expect_vocab:
        raise SystemExit(
            f"card-text embedding has vocab {vocab} but the env reports {expect_vocab}. "
            f"Rebuild it against this env (python card_text.py --out {path})."
        )
    if expect_ids_hash is not None and h != expect_ids_hash:
        raise SystemExit(
            f"card-text embedding was built for a different card pool (id hash {h} vs "
            f"{expect_ids_hash}). Every row past the first difference would map to the wrong "
            f"card. Rebuild it (python card_text.py --out {path})."
        )
    return emb


def resolve(path: str | None, env_vocab: int, no_card_emb: bool, env_ids_hash: str | None = None):
    """Decide the card-identity mode for an entry point. Returns (card_vocab, card_text_tensor).

    Shared by pretrain.py and ppo_vec.py so the two cannot disagree about what a given set of
    flags means — a mismatch there produces checkpoints that silently fail to load, or worse,
    load into a differently-wired model.
    """
    import torch

    if path:
        emb = load(path, expect_vocab=env_vocab, expect_ids_hash=env_ids_hash)
        return env_vocab, torch.from_numpy(emb)
    return (0 if no_card_emb else env_vocab), None


def describe(card_vocab: int, card_text, card_dim: int) -> str:
    if card_text is not None:
        return f"card identity: text embedding (vocab={card_vocab}, {card_text.shape[1]}d -> {card_dim}, frozen)"
    if card_vocab:
        return f"card identity: learned id embedding (vocab={card_vocab}, dim={card_dim})"
    return "card identity: none (card-blind)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="card_text_emb.npz")
    ap.add_argument("--dim", type=int, default=64, help="SVD components (tfidf encoder only)")
    ap.add_argument("--encoder", default="tfidf", choices=("tfidf", "st"))
    ap.add_argument("--st-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--dll", default=None)
    ap.add_argument("--dotnet", default="dotnet")
    args = ap.parse_args()

    env = SabberEnv(seed=0, dll=args.dll, dotnet=args.dotnet)
    try:
        cards = env.cards()
    finally:
        env.close()

    print(f"card rows from env: {len(cards)}")
    out = build(cards, args.dim, args.encoder, args.st_model)
    np.savez_compressed(args.out, **out)

    emb = out["emb"]
    with_text = sum(1 for c in cards if c["text"])
    print(f"encoder={args.encoder} dim={emb.shape[1]} rows={emb.shape[0]} "
          f"(with rules text: {with_text}) -> {args.out}")
    print(f"id-hash {out['ids_hash']}")


if __name__ == "__main__":
    main()
