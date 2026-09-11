"""Prompt construction and reply validation.

Both halves live here deliberately: the prompt defines a contract about what the
model may say, and the validator enforces it. Split them across files and they
drift apart the first time the vocabulary changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agents.actions import ActionKind

#: What an agent may choose to do. The model picks from this and nothing else —
#: a 3B model will happily invent "search_for_restaurant" otherwise.
PLANNABLE: dict[str, tuple[str, ...]] = {
    "sleep": ("home",),
    "eat": ("cafe", "market"),
    "study": ("university", "library"),
    "work": ("office", "hospital", "market", "power", "gas", "bank"),
    "socialize": ("cafe", "park"),
    "exercise": ("gym", "park"),
}

SYSTEM = (
    "You are a resident of a small city, planning your own day. "
    "Answer only with JSON. Be concise and concrete. "
    "Never invent activities or places outside the lists you are given."
)


@dataclass(slots=True)
class PlanStep:
    #: Minutes past midnight. Comparable straight against Clock.minute_of_day.
    at: int
    kind: ActionKind
    #: Building *kind*, not id — the sim resolves it to the nearest one, so a
    #: plan stays valid even if the agent is across town when the step arrives.
    place: str
    why: str

    def __str__(self) -> str:
        return f"{self.at // 60:02d}:{self.at % 60:02d} {self.kind.value} @{self.place}"


def _vocabulary() -> str:
    return "\n".join(f'  "{do}" at one of {list(places)}' for do, places in PLANNABLE.items())


def daily_plan(
    *,
    name: str,
    age: int,
    traits: list[str],
    home: str,
    clock: str,
    needs: dict[str, float],
    memories: list[str],
) -> str:
    """Ask an agent to plan the rest of their day."""
    needs_line = ", ".join(f"{k} {v:.0f}/100" for k, v in needs.items())
    recalled = "\n".join(f"  - {m}" for m in memories) or "  - (nothing in particular)"

    return f"""You are {name}, {age}, living at {home}. You are {", ".join(traits)}.
It is {clock}.

How you feel right now (100 is fully satisfied, under 20 is urgent):
  {needs_line}

What is on your mind:
{recalled}

Plan the rest of your day as 5 to 8 steps, in time order, starting after the
current time and ending with sleep at home. Choose only from these activities
and places:
{_vocabulary()}

Reply with JSON exactly like this:
{{"steps": [{{"at": "08:30", "do": "eat", "where": "cafe", "why": "short reason"}}]}}"""


def parse_plan(data: dict | None, after_minute: int) -> list[PlanStep]:
    """Turn a model reply into steps the simulation can execute.

    Never raises. Malformed steps are dropped; an empty result means "no plan",
    and the caller falls back to Tier 0. A bad generation must not be able to
    stall an agent.
    """
    if not isinstance(data, dict):
        return []
    raw = data.get("steps")
    if not isinstance(raw, list):
        return []

    steps: list[PlanStep] = []
    for entry in raw[:8]:
        if not isinstance(entry, dict):
            continue

        do = str(entry.get("do", "")).strip().lower()
        places = PLANNABLE.get(do)
        if places is None:
            continue  # invented an activity

        where = str(entry.get("where", "")).strip().lower()
        if where not in places:
            # Named a real activity in an impossible place ("sleep at the gym").
            # The first legal place is a better guess than discarding the step.
            where = places[0]

        at = _minutes(entry.get("at"))
        if at is None or at < after_minute:
            continue

        why = " ".join(str(entry.get("why", "")).split())[:120]
        steps.append(PlanStep(at=at, kind=ActionKind(do), place=where, why=why))

    steps.sort(key=lambda s: s.at)

    # Collapse duplicate times: two steps at 09:00 would have the second
    # silently overwrite the first the moment the clock reached it.
    deduped: list[PlanStep] = []
    for step in steps:
        if deduped and step.at - deduped[-1].at < 30:
            continue
        deduped.append(step)
    return deduped


def _minutes(value: object) -> int | None:
    """Parse "HH:MM" into minutes past midnight, tolerantly."""
    if isinstance(value, (int, float)):
        return int(value) % 1440
    if not isinstance(value, str):
        return None
    text = value.strip()
    if ":" not in text:
        return None
    hh, _, mm = text.partition(":")
    try:
        hours, minutes = int(hh), int(mm[:2])
    except ValueError:
        return None
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        return None
    return hours * 60 + minutes