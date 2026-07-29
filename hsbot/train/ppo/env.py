"""Python wrapper around the C# SabberStone env process (line-based stdio JSON protocol).

Each SabberEnv owns one `dotnet SabberStoneEnv.dll <seed>` subprocess. The protocol:
  send "reset"        -> {"obs":[144], "actions":[[20]...], "reward":0, "done":false}
  send "step <idx>"   -> {"obs":[144], "actions":[...], "reward":r, "done":bool}
On done, actions is empty; the next reset starts a new game.

This is a deliberately simple bridge for the prototype; for many parallel actors, swap the
subprocess for SabberStone's gRPC extension.
"""
from __future__ import annotations
import json
import pathlib
import subprocess

import numpy as np

ACT_DIM = 20
OBS_DIM = 144
PRIV_DIM = 8  # opponent-hidden features, critic-only (training)

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
    def _actions(d: dict) -> np.ndarray:
        a = d["actions"]
        return np.asarray(a, dtype=np.float32) if a else np.zeros((0, ACT_DIM), np.float32)

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self._rpc("reset")
        return np.asarray(d["obs"], np.float32), np.asarray(d["priv"], np.float32), self._actions(d)

    def step(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        d = self._rpc(f"step {idx}")
        return (np.asarray(d["obs"], np.float32), np.asarray(d["priv"], np.float32),
                self._actions(d), float(d["reward"]), bool(d["done"]))

    def close(self) -> None:
        try:
            assert self.proc.stdin
            self.proc.stdin.write("close\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
