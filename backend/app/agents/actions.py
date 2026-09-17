"""What an agent is currently doing.

An action is a commitment with a deadline. Agents don't re-decide every tick —
they choose something, travel to it, and stay until it completes. Most ticks
are therefore just "keep going", which is what keeps the simulation cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionKind(str, Enum):
    IDLE = "idle"
    TRAVEL = "travel"
    SLEEP = "sleep"
    EAT = "eat"
    STUDY = "study"
    WORK = "work"
    SOCIALIZE = "socialize"
    EXERCISE = "exercise"


#: Need repaired by each action, and the rate in points per sim-hour.
#: Kept as data so the tick loop is one lookup, not a growing if/elif chain.
RESTORES: dict[ActionKind, tuple[str, float]] = {
    ActionKind.SLEEP: ("energy", 14.0),
    ActionKind.EAT: ("hunger", 110.0),
    ActionKind.SOCIALIZE: ("social", 25.0),
    ActionKind.EXERCISE: ("fun", 20.0),
}

#: How long each action runs by default, in sim-minutes.
DURATION_MINUTES: dict[ActionKind, int] = {
    ActionKind.IDLE: 10,
    ActionKind.SLEEP: 480,
    ActionKind.EAT: 30,
    ActionKind.STUDY: 120,
    ActionKind.WORK: 240,
    ActionKind.SOCIALIZE: 60,
    ActionKind.EXERCISE: 60,
}


@dataclass
class Action:
    kind: ActionKind
    #: Building this happens at. None while wandering or idling.
    target_id: str | None = None
    #: Tick at which it completes. -1 means "runs until interrupted".
    ends_at: int = -1
    #: Tiles still to walk. Only TRAVEL uses this.
    path: list[tuple[int, int]] = field(default_factory=list)
    #: What to begin once TRAVEL arrives. Lets one field carry the intent
    #: through the journey, instead of a separate "why am I walking" state.
    then: Action | None = None
    #: Set only when the timetable dispatched this. The register is signed by
    #: whoever sent the agent, not by the clock when they arrive — journeys run
    #: longer than the join window.
    for_class: bool = False

    def is_done(self, tick: int) -> bool:
        if self.kind is ActionKind.TRAVEL:
            return not self.path
        return self.ends_at >= 0 and tick >= self.ends_at

    def __str__(self) -> str:
        where = f" @{self.target_id}" if self.target_id else ""
        return f"{self.kind.value}{where}"