"""The tick loop: the city's heartbeat.

Everything here is Tier 0 — deterministic Python, no model involved. Fifty
agents cost microseconds a tick, which is the whole premise: the simulation is
free, only thinking is expensive.
"""

from __future__ import annotations

import time
from collections import deque
from random import Random

from app.agents.actions import DURATION_MINUTES, Action, ActionKind
from app.agents.agent import Agent
from app.config import CONFIG
from app.sim.clock import TICKS_PER_DAY, Clock
from app.sim.layout import build_city
from app.sim.pathing import PathCache
from app.sim.spawn import spawn_agents
from app.sim.world import Building, BuildingKind

#: Need -> (what fixes it, where it happens). None means the agent's own home.
#: Data rather than branches, so new needs are one line each.
NEED_REMEDY: dict[str, tuple[ActionKind, BuildingKind | None]] = {
    "energy": (ActionKind.SLEEP, None),
    "hunger": (ActionKind.EAT, BuildingKind.CAFE),
    "social": (ActionKind.SOCIALIZE, BuildingKind.CAFE),
    "fun": (ActionKind.EXERCISE, BuildingKind.GYM),
}


class Simulation:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = Random(seed if seed is not None else CONFIG.seed)
        self.clock = Clock()
        self.world = build_city()
        self.paths = PathCache(self.world)
        self.events: deque[tuple[int, str]] = deque(maxlen=200)
        self.agents = spawn_agents(self.world, self.rng)

        # spawn builds actions directly, so they carry no deadline. Run each
        # through _begin to hold the invariant that every action an agent owns
        # has been properly started — otherwise the opening IDLE never ends.
        for agent in self.agents:
            self._begin(agent, agent.action)
    def tick(self) -> None:
        self.clock.advance()
        hours = CONFIG.world.minutes_per_tick / 60.0
        for agent in self.agents:
            self._step(agent, hours)

    def _step(self, agent: Agent, hours: float) -> None:
        agent.tick_needs(hours)
        action = agent.action

        if action.kind is ActionKind.TRAVEL:
            for _ in range(CONFIG.world.tiles_per_tick):
                if not action.path:
                    break
                agent.move_to(action.path.pop(0))
            if not action.path:
                self._begin(agent, action.then or Action(ActionKind.IDLE))
        elif action.is_done(self.clock.tick):
            self._begin(agent, self._choose_action(agent))

    def _begin(self, agent: Agent, action: Action) -> None:
        """Stamp the finish time, then commit.

        Duration is applied here rather than at construction because a chained
        `then` is built before the walk begins — its clock must start on
        arrival, or a 20-minute commute eats 20 minutes of the meal.
        """
        if action.kind is not ActionKind.TRAVEL and action.ends_at < 0:
            minutes = DURATION_MINUTES.get(action.kind, 10)
            action.ends_at = self.clock.tick + max(1, minutes // CONFIG.world.minutes_per_tick)

        agent.action = action
        if action.kind not in (ActionKind.TRAVEL, ActionKind.IDLE):
            where = self.world.buildings[action.target_id].name if action.target_id else "here"
            self.events.append(
                (self.clock.tick, f"{agent.name} began {action.kind.value} at {where}")
            )

    def _choose_action(self, agent: Agent) -> Action:
        """Tier 0 brain: repair whatever is worst.

        Crude on purpose. It yields a legible daily rhythm for free, and it is
        precisely the function Phase 2 swaps out for a planned day.
        """
        need, _ = agent.needs.lowest()
        kind, place = NEED_REMEDY[need]
        target = (
            self.world.buildings[agent.home_id]
            if place is None
            else self.world.nearest(place, agent.pos)
        )
        if target is None:
            return Action(ActionKind.IDLE)
        return self._travel_to(agent, target, kind)

    def _travel_to(self, agent: Agent, building: Building, then_kind: ActionKind) -> Action:
        arrive = Action(then_kind, target_id=building.id)
        path = self.paths.path(agent.pos, building.door)
        if path is None:
            # Unreachable. Idle and retry — the cache remembers the failure, so
            # the retry is a dict hit rather than another search.
            return Action(ActionKind.IDLE)
        if not path:
            return arrive
        return Action(ActionKind.TRAVEL, target_id=building.id, path=path, then=arrive)


def _demo(days: int = 2) -> None:
    sim = Simulation()
    ticks = TICKS_PER_DAY * days

    started = time.perf_counter()
    for _ in range(ticks):
        sim.tick()
    elapsed = time.perf_counter() - started

    print(f"{ticks} ticks ({days} sim-days) in {elapsed:.2f}s -> {ticks / elapsed:,.0f} ticks/sec")
    print(f"clock: {sim.clock}")

    counts: dict[str, int] = {}
    for a in sim.agents:
        counts[a.action.kind.value] = counts.get(a.action.kind.value, 0) + 1
    print("doing now:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    print(f"path cache: {sim.paths.hit_rate:.0%} hit ({sim.paths.hits}/{sim.paths.misses})")

    print("\nsample agents")
    for a in sim.agents[:5]:
        needs = " ".join(f"{k[:3]}={v:5.1f}" for k, v in a.needs.as_dict().items())
        print(f"  {a.name:<20} {str(a.action):<24} {needs}")

    print("\nlatest events")
    for tick, text in list(sim.events)[-8:]:
        print(f"  t{tick:<5} {text}")


if __name__ == "__main__":
    _demo()