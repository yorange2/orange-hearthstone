"""
Independent reference implementation of the v1 feature contract (hsbot/docs/FEATURES.md).

This is the *ground truth* for parity fixtures. It deliberately shares no code with the
Kotlin (War) or C# (POGame) extractors, so agreement across all three is real evidence.

It consumes the engine-neutral `state` dict (see hsbot/fixtures/README.md) and returns the
144-float vector. Run `python reference_features.py` to (re)generate hsbot/fixtures/*.json.
"""
from __future__ import annotations
import json
import pathlib

LENGTH = 144
GLOBAL_PER_PLAYER = 15
TURN_BASE = 30
MINION_SECTION_BASE = 32
MINION_SLOTS = 7
MINION_FEATS = 8
MINION_PER_PLAYER = MINION_SLOTS * MINION_FEATS  # 56


def _b(x: bool) -> float:
    return 1.0 if x else 0.0


def _write_globals(f: list[float], base: int, p: dict) -> None:
    hero = p["hero"]
    board = p["board"]
    weapon = p.get("weapon")
    f[base + 0] = hero["effectiveHp"] / 30
    f[base + 1] = hero["armor"] / 30
    f[base + 2] = p["hand"] / 10
    f[base + 3] = p["deck"] / 30
    f[base + 4] = len(board) / 7
    f[base + 5] = sum(m["attack"] for m in board) / 30
    f[base + 6] = sum(m["health"] for m in board) / 40
    f[base + 7] = sum(1 for m in board if m["taunt"]) / 7
    f[base + 8] = sum(1 for m in board if m["divineShield"]) / 7
    f[base + 9] = p["secrets"] / 5
    f[base + 10] = (weapon["attack"] if weapon else 0) / 10
    f[base + 11] = (weapon["durability"] if weapon else 0) / 5
    f[base + 12] = p["mana"]["max"] / 10
    f[base + 13] = p["mana"]["overload"] / 5
    f[base + 14] = p["fatigue"] / 10


def _write_minions(f: list[float], base: int, p: dict) -> None:
    for i, m in enumerate(p["board"][:MINION_SLOTS]):
        o = base + i * MINION_FEATS
        f[o + 0] = m["attack"] / 12
        f[o + 1] = m["health"] / 12
        f[o + 2] = _b(m["canAttack"])
        f[o + 3] = _b(m["taunt"])
        f[o + 4] = _b(m["divineShield"])
        f[o + 5] = _b(m["windfury"])
        f[o + 6] = _b(m["lifesteal"])
        f[o + 7] = _b(m["poisonous"])


def extract(state: dict) -> list[float]:
    f = [0.0] * LENGTH
    _write_globals(f, 0 * GLOBAL_PER_PLAYER, state["me"])
    _write_globals(f, 1 * GLOBAL_PER_PLAYER, state["opp"])
    f[TURN_BASE + 0] = state["turn"] / 30
    f[TURN_BASE + 1] = _b(state["meIsFirst"])
    _write_minions(f, MINION_SECTION_BASE + 0 * MINION_PER_PLAYER, state["me"])
    _write_minions(f, MINION_SECTION_BASE + 1 * MINION_PER_PLAYER, state["opp"])
    return f


# ---- fixture authoring ------------------------------------------------------

def minion(attack, health, canAttack=True, taunt=False, divineShield=False,
           windfury=False, lifesteal=False, poisonous=False) -> dict:
    return dict(attack=attack, health=health, canAttack=canAttack, taunt=taunt,
                divineShield=divineShield, windfury=windfury, lifesteal=lifesteal,
                poisonous=poisonous)


def side(hero_hp=30, hero_armor=0, mana_max=0, overload=0, hand=0, deck=0,
         board=None, secrets=0, weapon=None, fatigue=0) -> dict:
    return dict(hero=dict(effectiveHp=hero_hp, armor=hero_armor),
                mana=dict(max=mana_max, overload=overload), hand=hand, deck=deck,
                board=board or [], secrets=secrets, weapon=weapon, fatigue=fatigue)


FIXTURES = {
    "empty_boards_turn1": dict(
        turn=1, meIsFirst=True,
        me=side(mana_max=1, hand=3, deck=27),
        opp=side(mana_max=0, hand=4, deck=26),
    ),
    "two_taunts_vs_empty": dict(
        turn=5, meIsFirst=True,
        me=side(hero_hp=30, mana_max=5, hand=2, deck=20,
                board=[minion(2, 3, taunt=True), minion(3, 2, taunt=True, divineShield=True)],
                secrets=1),
        opp=side(hero_hp=24, mana_max=4, hand=5, deck=18),
    ),
    "weapon_and_armor_lethal_setup": dict(
        turn=8, meIsFirst=False,
        me=side(hero_hp=22, hero_armor=5, mana_max=7, overload=1, hand=1, deck=10,
                board=[minion(5, 5, windfury=True, lifesteal=True), minion(1, 1, poisonous=True)],
                weapon=dict(attack=3, durability=2), secrets=0, fatigue=0),
        opp=side(hero_hp=6, mana_max=6, hand=3, deck=0,
                 board=[minion(0, 4, canAttack=False, taunt=True)], fatigue=1),
    ),
}


def main() -> None:
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
    for name, state in FIXTURES.items():
        vec = extract(state)
        doc = dict(name=name, version="v1", state=state, expected=vec)
        (out_dir / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {name}.json  (nonzero: {sum(1 for x in vec if x)}/{LENGTH})")


if __name__ == "__main__":
    main()
