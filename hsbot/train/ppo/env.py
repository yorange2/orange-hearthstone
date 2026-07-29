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
    priv: np.ndarray     # [PRIV_DIM]  (critic-only)
    actions: np.ndarray  # [N, ACT_DIM]
    player: int          # 1 or 2 — whose turn it is

_DEFAULT_DLL = (
    pathlib.Path(__file__).resolve().parents[2]
    / "trainer/SabberStoneEnv/bin/Release/net8.0/SabberStoneEnv.dll"
)


class SabberEnv:
    def __init__(self, seed: int = 1, dll: str | None = None, dotnet: str = "dotnet"):
        dll_path = str(dll or _DEFAULT_DLL)
        self.proc = subprocess.Popen(
            [dotnet, dll_path, str(seed)],
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
        return Obs(self._mat(d["tokens"], TOKEN_DIM), np.asarray(d["priv"], np.float32),
                   self._mat(d["actions"], ACT_DIM), int(d["player"]))

    def reset(self) -> "Obs":
        return self._obs(self._rpc("reset"))

    def step(self, idx: int) -> tuple["Obs", bool, int]:
        """Returns (next observation, done, winner[1/2/0])."""
        d = self._rpc(f"step {idx}")
        return self._obs(d), bool(d["done"]), int(d["winner"])

    def step_greedy(self) -> tuple["Obs", bool, int]:
        """The env plays the current mover's greedy (MidRangeScore) heuristic action."""
        d = self._rpc("step_greedy")
        return self._obs(d), bool(d["done"]), int(d["winner"])

    def close(self) -> None:
        try:
            assert self.proc.stdin
            self.proc.stdin.write("close\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
