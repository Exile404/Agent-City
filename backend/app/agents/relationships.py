"""Who knows whom, and how well.

Affinity is simulation state: trait compatibility is computed here, and a model
is only ever allowed to nudge it within bounds. The sim owns the number, the
model owns what was actually said.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Traits that grate on each other. Anything not listed is neutral.
FRICTION: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in [
        ("blunt", "warm"),
        ("restless", "patient"),
        ("cautious", "ambitious"),
        ("stubborn", "gregarious"),
        ("analytical", "restless"),
    ]
)


@dataclass(slots=True)
class Relationship:
    #: -100 (hostile) to 100 (close). Starts at 0: strangers, not enemies.
    affinity: float = 0.0
    times_met: int = 0
    last_talked_tick: int = -10_000
    #: The most recent thing worth remembering about them.
    note: str = ""

    @property
    def label(self) -> str:
        if self.times_met == 0:
            return "a stranger"
        if self.affinity > 45:
            return "a close friend"
        if self.affinity > 15:
            return "a friend"
        if self.affinity < -25:
            return "someone you dislike"
        return "an acquaintance"


def trait_rapport(a: list[str], b: list[str]) -> float:
    """How naturally two people get on, before anything is said.

    Shared traits help, listed frictions hurt. Deliberately small: this biases a
    relationship rather than deciding it, so who someone ends up close to is
    mostly a product of who they actually keep running into.
    """
    shared = len(set(a) & set(b))
    clashes = sum(1 for x in a for y in b if frozenset((x, y)) in FRICTION)
    return shared * 2.5 - clashes * 2.0

def conversation_interest(
    *, relationship: Relationship | None, social_a: float, social_b: float, hours_since: float
) -> float:
    """How much a real conversation here is worth spending a generation on.

    First meetings dominate: at five encounters per available slot, "these two
    have never met" is by far the most informative thing that can happen. They
    also run out — fifty agents have all met inside two sim-days — so the rest of
    the score has to carry the city from day three on, and every term below is
    scaled against what it was measured to actually reach by then.
    """
    if relationship is None or relationship.times_met == 0:
        base = 6.0
    else:
        # Square-rooted days, not capped hours. The measured gap between meetings
        # runs from 4 hours to ten sim-days; min(3.0, hours / 8.0) scored
        # everything past a single day identically, which left the choice among
        # known pairs very nearly arbitrary.
        apart = min(3.0, (hours_since / 24.0) ** 0.5 * 1.4)
        # Divided by twelve, not forty. Tier 0 moves affinity about a point per
        # greeting and it measured out at 28 after ten sim-days, so a /40 divisor
        # contributed 0.05 at the median: present in the formula, absent from the
        # outcome. Capped so a long friendship cannot crowd out everything else.
        feeling = min(2.0, abs(relationship.affinity) / 12.0)
        base = apart + feeling

    lonelier = min(social_a, social_b)
    return base + max(0.0, (50.0 - lonelier) / 50.0) * 2.5