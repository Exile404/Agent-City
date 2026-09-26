"""Deterministic replay.

A live run is the seeded simulation plus whatever the models said, applied on
whatever tick each answer landed. Record the second half and the first half
replays for free — which only holds if the simulation itself is deterministic.
This module checks that, and will hold the recorder and the replayer.

Everything the Hub does to the simulation, so everything a recording must carry:

  plan       lands   agent.plan and last_plan_tick, a "plan" memory, a log line
  chat       lands   affinity and note both ways, a "dialogue" memory and
                     +10 social for each, a log line
  exam       starts  enrollment.in_flight, which stands the grace timer down
             lands   sit_exam(agent, mark), an exam memory, the examiner's line
  interview  starts  application.in_flight
             lands   hire_or_reject(agent, verdict), an interview memory
  review     starts  job.review_in_flight
             lands   hold_review(agent, verdict, comment)
  reflection lands   a "reflection" memory per insight, since_reflection reset
  startup            sim.review_grace, stretched when a model is on

Retrieval also moves memory.last_access and embedding fills in vectors, but
both only shape the next prompt, never a number, so neither is fingerprinted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from collections import deque
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from app.agents.agent import Agent
from app.config import CONFIG

#: Built from the layout, caches of it, or plumbing: not the city's state.
STATIC = frozenset({"world", "paths", "_interiors", "by_id", "recorder"})
#: Changed off the tick path by retrieval and embedding; they shape prompts only.
PROMPT_ONLY = frozenset({"vector", "last_access", "pending"})


def _fields(obj: Any) -> list[tuple[str, Any]]:
    return [
        (f.name, _canon(getattr(obj, f.name)))
        for f in dataclasses.fields(obj)
        if f.name not in PROMPT_ONLY
    ]


def _canon(x: Any) -> Any:
    """A plain, ordered copy of some state, safe to repr and hash."""
    if isinstance(x, Agent):
        # Referenced from a queue or an encounter: the id is enough, since the
        # agent's own state is fingerprinted once, under its own name.
        return x.id
    if isinstance(x, Enum):
        return x.value
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if dataclasses.is_dataclass(x):
        return _fields(x)
    if isinstance(x, dict):
        return sorted((str(k), _canon(v)) for k, v in x.items())
    if isinstance(x, (set, frozenset)):
        return sorted(repr(_canon(v)) for v in x)
    if isinstance(x, (list, tuple, deque)):
        return [_canon(v) for v in x]
    # Anything else would repr its memory address and differ every run.
    raise TypeError(f"fingerprint cannot read a {type(x).__name__}")


def _digest(value: Any) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()[:16]


def fingerprint_parts(sim: Any) -> dict[str, str]:
    """One short hash per piece of state: each agent under its id, each other
    attribute of the simulation under its name. When two runs part ways, the
    names that differ say where."""
    parts: dict[str, str] = {}
    for name, value in vars(sim).items():
        if name in STATIC:
            continue
        if name == "agents":
            parts["roster"] = _digest([a.id for a in value])
            for agent in value:
                parts[agent.id] = _digest(_fields(agent))
        elif name == "rng":
            parts[name] = _digest(value.getstate())
        else:
            parts[name] = _digest(_canon(value))
    return parts


def fingerprint(sim: Any) -> str:
    """The whole simulation's state as one short hash."""
    return _digest(sorted(fingerprint_parts(sim).items()))


class Recorder:
    """A live run written down as JSON lines: what it was started with, every
    model result in the order it entered the simulation, and a fingerprint each
    morning for a replay to check itself against."""

    def __init__(self, path: Path, sim: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._out = path.open("w", encoding="utf-8")
        self.write(sim.clock.tick, "begin", {
            "seed": CONFIG.seed,
            # Every tuning knob is read from AC_* at import, so these recreate the city.
            "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("AC_")},
            "review_grace": sim.review_grace,
            "fingerprint": fingerprint(sim),
        })

    @classmethod
    def open(cls, sim: Any) -> Recorder:
        return cls(Path(CONFIG.replay.record_dir) / time.strftime("run-%Y%m%d-%H%M%S.jsonl"), sim)

    def write(self, tick: int, op: str, fields: dict[str, Any]) -> None:
        # A model can still answer during shutdown, after the file has closed.
        if self._out.closed:
            return
        self._out.write(json.dumps({"t": tick, "op": op, **fields}, separators=(",", ":")) + "\n")

    def _snapshot(self, sim: Any, op: str) -> None:
        parts = fingerprint_parts(sim)
        self.write(sim.clock.tick, op, {"fingerprint": _digest(sorted(parts.items())), "parts": parts})

    def checkpoint(self, sim: Any) -> None:
        """Fingerprint the city, part by part so a mismatch can say where, and
        flush, so a crash loses at most a day."""
        self._snapshot(sim, "day")
        self._out.flush()

    def close(self, sim: Any) -> None:
        """A last fingerprint, so a replay checks the tail after the final morning."""
        if not self._out.closed:
            self._snapshot(sim, "stop")
            self._out.close()


def replay(path: Path, on_morning: Callable[[Any], None] | None = None) -> bool:
    """Run a recording back through a headless simulation, checking every
    morning's fingerprint against the live run's. True if all of them match.
    on_morning, if given, sees the rebuilt city at each morning that matched."""
    # Imported here: loop imports this module for the Recorder.
    from app.agents.actions import ActionKind
    from app.cognition.prompts import PlanStep
    from app.sim.loop import Simulation

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    head = events[0]
    print(f"replaying {path.name}: {len(events)} events")
    here = {k: v for k, v in sorted(os.environ.items()) if k.startswith("AC_")}
    if here != head["env"]:
        print(f"  recorded under {head['env'] or 'no AC_* settings'}, running under"
              f" {here or 'none'}: every number would differ, so this would prove nothing")
        return False

    sim = Simulation(head["seed"])
    sim.review_grace = head["review_grace"]
    if fingerprint(sim) != head["fingerprint"]:
        print("  the city differs before the first tick: this is not the code that recorded it")
        return False

    def who(agent_id: str) -> Any:
        return sim.by_id[agent_id]

    def pair(paper: list[str] | None) -> tuple[str, str] | None:
        return (paper[0], paper[1]) if paper else None

    doors = {
        "start": lambda e: sim.model_started(e["kind"], who(e["agent"])),
        "end": lambda e: sim.model_finished(e["kind"], who(e["agent"])),
        "plan": lambda e: sim.land_plan(
            who(e["agent"]),
            [PlanStep(at, ActionKind(kind), place, why) for at, kind, place, why in e["steps"]],
        ),
        "chat": lambda e: sim.land_chat(
            who(e["a"]), who(e["b"]), [(w, s) for w, s in e["lines"]], e["warmth"], e["place"]
        ),
        "exam": lambda e: sim.land_exam(who(e["agent"]), e["mark"], pair(e["paper"]), e["comment"]),
        "interview": lambda e: sim.land_interview(
            who(e["agent"]), e["verdict"], pair(e["paper"]), e["reason"]
        ),
        "review": lambda e: sim.land_review(who(e["agent"]), e["verdict"], e["comment"]),
        "reflection": lambda e: sim.land_reflection(who(e["agent"]), e["insights"]),
    }

    mornings = raised = 0
    for e in events[1:]:
        while sim.clock.tick < e["t"]:
            sim.tick()
        if e["op"] in ("day", "stop"):
            parts = fingerprint_parts(sim)
            if _digest(sorted(parts.items())) != e["fingerprint"]:
                recorded = e.get("parts") or {}
                differ = sorted(k for k in parts if recorded.get(k) != parts[k]) if recorded else []
                print(f"  {sim.clock}  MISMATCH" + (f" in {', '.join(differ[:12])}" if differ else ""))
                return False
            if e["op"] == "day":
                mornings += 1
                if on_morning is not None:
                    on_morning(sim)
            continue
        try:
            doors[e["op"]](e)
        except Exception:
            # The live Hub caught these too, and a door raises at the same point
            # both times, so the city is left in the same state either way.
            raised += 1
    print(f"  {sim.clock}  all {mornings} mornings match; {raised} doors raised")
    return True


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1])
    recorded = json.loads(target.open(encoding="utf-8").readline())["env"]
    if {k: v for k, v in os.environ.items() if k.startswith("AC_")} != recorded:
        # CONFIG is read from the environment at import, so the recorded settings
        # have to be in place before Python starts, not patched in afterwards.
        env = {k: v for k, v in os.environ.items() if not k.startswith("AC_")} | recorded
        os.execvpe(sys.executable, [sys.executable, "-m", "app.sim.replay", str(target)], env)
    sys.exit(0 if replay(target) else 1)
