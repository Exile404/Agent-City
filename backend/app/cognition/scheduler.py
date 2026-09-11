"""Admission control for thinking.

The GPU serves roughly four inferences a second; fifty agents would ask for far
more. So agents *request* a thought and the scheduler decides who gets one.
Everyone refused falls back to Tier 0 — deterministic Python that runs for every
agent every tick — so nothing ever blocks, stalls, or waits on a model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.config import CONFIG


@dataclass(slots=True)
class Ask:
    agent_id: str
    #: "plan" | "reflect" | "chat"
    kind: str
    priority: float
    tick: int


def plan_priority(*, minutes_since_plan: float, worst_need: float, has_plan: bool) -> float:
    """How badly this agent needs to re-plan.

    An agent with no plan at all outranks everyone: it is about to fall back to
    chasing whichever need is lowest, which is exactly the behaviour the planner
    exists to replace.
    """
    if not has_plan:
        return 10.0
    staleness = minutes_since_plan / CONFIG.cognition.replan_minutes
    urgency = max(0.0, (40.0 - worst_need) / 40.0)
    return min(9.0, staleness * 4.0 + urgency * 3.0)


def reflect_priority(since_reflection: float) -> float:
    """Reflection is never urgent — it loses to planning by design."""
    ratio = since_reflection / CONFIG.memory.reflection_importance_threshold
    return min(5.0, ratio * 2.5)


class Scheduler:
    """Decides which agents get to use the model this tick."""

    def __init__(self) -> None:
        # Keyed by agent: asking twice replaces your own earlier request rather
        # than queueing two, so one indecisive agent cannot fill the budget.
        self._asks: dict[str, Ask] = {}
        self.outstanding = 0
        self.asked = 0
        self.admitted = 0
        self.refused = 0

    def ask(self, agent_id: str, kind: str, priority: float, tick: int) -> None:
        self.asked += 1
        self._asks[agent_id] = Ask(agent_id, kind, priority, tick)

    def dispatch(self, tick: int, run: Callable[[Ask], None]) -> list[Ask]:
        """Admit the highest-priority asks that fit, and start them.

        Refused asks are discarded rather than carried forward: an agent's
        situation changes every tick, so a request from four ticks ago describes
        a world that is gone. They re-ask next tick, and their priority rises as
        the need sharpens — so nobody starves waiting behind a long queue.
        """
        if not self._asks:
            return []

        cfg = CONFIG.cognition
        headroom = min(cfg.admissions_per_tick, cfg.max_outstanding - self.outstanding)
        if headroom <= 0:
            self.refused += len(self._asks)
            self._asks.clear()
            return []

        ranked = sorted(self._asks.values(), key=lambda a: -a.priority)
        admitted, refused = ranked[:headroom], ranked[headroom:]
        self.refused += len(refused)
        self._asks.clear()

        for ask in admitted:
            self.admitted += 1
            self.outstanding += 1
            run(ask)
        return admitted

    def finished(self) -> None:
        """Called whenever a dispatched thought completes, fails or goes stale."""
        self.outstanding = max(0, self.outstanding - 1)

    def snapshot(self) -> dict:
        return {
            "asked": self.asked,
            "admitted": self.admitted,
            "refused": self.refused,
            "outstanding": self.outstanding,
            "admitRate": round(self.admitted / self.asked, 3) if self.asked else 0.0,
        }