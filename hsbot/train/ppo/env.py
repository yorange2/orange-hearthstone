"""Python wrapper around the C# SabberStone env process (line-based stdio JSON protocol).

Each SabberEnv owns one `dotnet SabberStoneEnv.dll <seed>` subprocess. The protocol:
  send "reset"        -> {"obs":[144], "actions":[[20]...], "reward":0, "done":false}
  send "step <idx>"   -> {"obs":[144], "actions":[...], "reward":r, "done":bool}
On done, actions is empty; the next reset starts a new game.

This is a deliberately simple bridge for the prototype; for many parallel actors, swap the
subprocess for SabberStone's gRPC extension.
"""
from __future__ import annotations
import dataclasses
import json
import pathlib
import subprocess

import numpy as np

TOKEN_DIM = 18  # per entity token (v2 state representation)
ACT_DIM = 20
PRIV_DIM = 8  # opponent-hidden features, critic-only (training)


@dataclasses.dataclass
class Obs:
    """One decision point, from the current mover's perspective."""
    tokens: np.ndarray   # [T, TOKEN_DIM]
    card_ids: np.ndarray # [T] int64 CardVocab index per token (0 = none/unknown)
    flat: np.ndarray     # [144] flat observable features (FEATURES.md v1)
    priv: np.ndarray     # [PRIV_DIM]  (critic-only)
    actions: np.ndarray  # [N, ACT_DIM]
    player: int          # 1 or 2 — whose turn it is
    potential: float     # normalized board score Φ ∈ [-1,1] (reward shaping)
    greedy_action: int = -1  # action index chosen by greedy (only set after step_greedy)

_DEFAULT_DLL = (
    pathlib.Path(__file__).resolve().parents[2]
    / "trainer/SabberStoneEnv/bin/Release/net10.0/SabberStoneEnv.dll"
)


DECK_MODES = ("fixed", "variedmirror", "random")


def deck_mode(args) -> str:
    """Resolve the deck mode from parsed CLI args: explicit `--deck` wins, `--fixed-deck` is the
    legacy shorthand kept because every recipe in the README passes it."""
    if getattr(args, "deck", None):
        return args.deck
    return "fixed" if getattr(args, "fixed_deck", False) else "random"


def _deck_arg(deck: str | bool) -> str:
    """Normalize a deck mode for the C# argv. `True`/`False` are still accepted because
    `--fixed-deck` is a bool flag on both entry points and every recipe in the README passes it."""
    if isinstance(deck, bool):
        return "fixed" if deck else "random"
    if deck not in DECK_MODES:
        raise ValueError(f"unknown deck mode {deck!r}; expected one of {DECK_MODES}")
    return deck


class SabberEnv:
    def __init__(self, seed: int = 1, deck: str | bool = "random", dll: str | None = None,
                 dotnet: str = "dotnet", potential: str = "midrange"):
        dll_path = str(dll or _DEFAULT_DLL)
        self.proc = subprocess.Popen(
            [dotnet, dll_path, str(seed), _deck_arg(deck), potential],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )

    def _rpc(self, cmd: str) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    @staticmethod
    def _mat(rows, dim: int) -> np.ndarray:
        return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, dim), np.float32)

    def _obs(self, d: dict) -> "Obs":
        return Obs(self._mat(d["tokens"], TOKEN_DIM),
                   np.asarray(d.get("cardIds", []), np.int64),
                   np.asarray(d["flat"], np.float32),
                   np.asarray(d["priv"], np.float32),
                   self._mat(d["actions"], ACT_DIM), int(d["player"]), float(d["potential"]),
                   int(d.get("greedyAction", -1)))

    # ---- low-level primitives (used by VecEnv to pipeline N processes) ----
    def send(self, cmd: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(cmd + "\n")

    def flush(self) -> None:
        assert self.proc.stdin
        self.proc.stdin.flush()

    def recv(self) -> tuple["Obs", bool, int]:
        assert self.proc.stdout
        d = json.loads(self.proc.stdout.readline())
        return self._obs(d), bool(d["done"]), int(d["winner"])

    def meta(self) -> dict:
        """Static env facts needed before the model is built (card-embedding vocab size).
        Safe to call at startup: it does not touch game state."""
        return self._rpc("meta")

    def card_vocab(self) -> int:
        return int(self.meta()["cardVocab"])

    def card_ids_hash(self) -> str:
        """Fingerprint of this env's card-id list, for validating a card-text matrix against it."""
        return str(self.meta().get("cardIdsHash", ""))

    def cards(self) -> list[dict]:
        """Id / name / rules text for every vocabulary row, in index order (row 0 is the reserved
        blank). Used to build the card-text embedding matrix — sourced from the env so the rows
        line up with the indices the env emits rather than by re-deriving the ordering."""
        return self._rpc("cards")["cards"]

    def reset(self) -> "Obs":
        return self._obs(self._rpc("reset"))

    def step(self, idx: int) -> tuple["Obs", bool, int]:
        """Returns (next observation, done, winner[1/2/0])."""
        d = self._rpc(f"step {idx}")
        return self._obs(d), bool(d["done"]), int(d["winner"])

    def step_greedy(self, strategy: str = "midrange") -> tuple["Obs", bool, int]:
        """The env plays the current mover's greedy heuristic action.
        strategy: one of midrange, aggro, control, fatigue, ramp."""
        d = self._rpc(f"step_greedy {strategy}")
        return self._obs(d), bool(d["done"]), int(d["winner"])

    def simulate(self) -> list[dict]:
        """Phase 3: resulting state per legal action (one-ply lookahead), from the current
        mover's perspective. Each entry is {tokens, card_ids, priv, terminal, outcome}."""
        d = self._rpc("simulate")
        out = []
        for st in d["states"]:
            out.append(dict(
                tokens=self._mat(st.get("tokens"), TOKEN_DIM),
                card_ids=np.asarray(st.get("cardIds") or [], np.int64),
                priv=np.asarray(st.get("priv") or [0.0] * PRIV_DIM, np.float32),
                terminal=bool(st.get("terminal", False)),
                mine=bool(st.get("mine", True)),
                outcome=int(st.get("outcome", 0)),
            ))
        return out

    def close(self) -> None:
        try:
            assert self.proc.stdin
            self.proc.stdin.write("close\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class VecEnv:
    """N env subprocesses driven in lockstep. A batched command is written to all N stdins and
    flushed, so the N SabberStone processes compute concurrently (across CPU cores) while Python
    then reads their replies — the throughput multiplier. Inference is batched by the caller.
    """

    def __init__(self, n: int, seed0: int = 1, deck: str | bool = "random",
                 dll: str | None = None, dotnet: str = "dotnet", potential: str = "midrange"):
        self.n = n
        self.envs = [SabberEnv(seed0 + i, deck, dll, dotnet, potential) for i in range(n)]

    def card_vocab(self) -> int:
        return self.envs[0].card_vocab()

    def reset_all(self) -> list["Obs"]:
        for e in self.envs:
            e.send("reset")
        for e in self.envs:
            e.flush()
        return [e.recv()[0] for e in self.envs]

    def send_batch(self, cmds: list[str | None]) -> dict[int, tuple["Obs", bool, int]]:
        """cmds[i] is a command string for env i, or None to leave it idle this tick.
        Returns {i: (obs, done, winner)} for the non-idle envs."""
        active = [i for i, c in enumerate(cmds) if c is not None]
        for i in active:
            self.envs[i].send(cmds[i])  # type: ignore[arg-type]
        for i in active:
            self.envs[i].flush()
        return {i: self.envs[i].recv() for i in active}

    def close(self) -> None:
        for e in self.envs:
            e.close()
