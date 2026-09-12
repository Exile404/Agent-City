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
from app.sim.world import Building, BuildingKind, TileKind
from app.cognition.prompts import PlanStep
from app.agents.relationships import Relationship, conversation_interest, trait_rapport

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
        # Interior floor tiles per building, precomputed once. Agents step onto
        # one of these on arrival instead of standing on the doorstep.
        self._interiors: dict[str, list[tuple[int, int]]] = {
            b.id: [t for t in b.tiles() if self.world.tile(*t) is TileKind.FLOOR]
            for b in self.world.buildings.values()
        }
        self.events: deque[tuple[int, str]] = deque(maxlen=200)
        #: Total events ever logged. The deque is capped, so its length stops
        #: being a usable cursor the moment it fills; this never resets.
        self.events_total = 0
        self.agents = spawn_agents(self.world, self.rng)
        #: Lookup for the cognition pipeline, which works from agent ids.
        self.by_id = {a.id: a for a in self.agents}
        # spawn builds actions directly, so they carry no deadline. Run each
        # through _begin to hold the invariant that every action an agent owns
        # has been properly started — otherwise the opening IDLE never ends.
        for agent in self.agents:
            self._begin(agent, agent.action)
        #: (a, b, interest) for pairs that met on the most recent tick, scored
        #: before greeting. The cognition pipeline promotes the best of them.
        self.last_encounters: list[tuple[Agent, Agent, float]] = []

    def tick(self) -> None:
        self.clock.advance()
        hours = CONFIG.world.minutes_per_tick / 60.0
        for agent in self.agents:
            self._step(agent, hours)
        # Resolved and stored: greet() puts every pair on cooldown, so calling
        # encounters() again after a tick returns an empty list.
        #
        # Scored before greeting, not after: greet() stamps last_talked_tick and
        # bumps times_met, which are the facts the score reads. Reverse the order
        # and every pair looks like acquaintances who just spoke.
        self.last_encounters = [(a, b, self._interest(a, b)) for a, b in self.encounters()]
        for a, b, _ in self.last_encounters:
            self.greet(a, b)

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

    def log(self, text: str) -> None:
        """Record a feed event.

        Counted as well as stored, because not everything is logged from the
        tick path: a plan landing or a conversation finishing happens several
        ticks after it was requested, and stamping it with the tick it completed
        on is useless to a consumer that has already sent that tick's frame.
        Readers track the total instead.
        """
        self.events_total += 1
        self.events.append((self.clock.tick, text))

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
            self.log(f"{agent.name} began {action.kind.value} at {where}")
            # Sleep is excluded: it happens nightly and tells nobody anything.
            # The rest gives agents something to talk about besides who they met.
            if action.kind is not ActionKind.SLEEP:
                agent.memory.add(
                    self.clock.tick, "observation", f"{action.kind.value.capitalize()} at {where}"
                )

    def _choose_action(self, agent: Agent) -> Action:
        """Three layers, strict priority.

        A critical need overrides any plan — an agent about to collapse from
        hunger does not keep a study appointment. Otherwise follow the plan.
        Tier 0 is the floor: it always has an answer, so no agent is ever left
        without one because a model was busy.
        """
        critical = agent.needs.critical()
        if critical is not None:
            return self._remedy(agent, critical)

        step = self._due_step(agent)
        if step is not None:
            action = self._follow(agent, step)
            if action is not None:
                return action

        return self._remedy(agent, agent.needs.lowest()[0])

    #: How late a plan step may be and still worth doing, in sim-minutes. A step
    #: costs travel plus its duration, so running an hour or so behind is normal
    #: and should not throw the rest of the day away.
    STEP_GRACE_MINUTES = 120

    def _due_step(self, agent: Agent) -> PlanStep | None:
        """The earliest step that is due and still worth doing.

        Takes one step at a time rather than draining everything overdue: an
        agent running late should keep following its plan in order, just late.
        Only genuinely ancient steps are dropped, so nobody works through
        backed-up appointments at three in the morning.
        """
        now = self.clock.minute_of_day
        while agent.plan:
            step = agent.plan[0]
            if step.at > now:
                return None  # next step is still in the future
            agent.plan.pop(0)
            if now - step.at <= self.STEP_GRACE_MINUTES:
                return step
            # too old to be meaningful — discard and consider the next one
        return None

    def _follow(self, agent: Agent, step: PlanStep) -> Action | None:
        """Turn a plan step into an action, or None if it cannot be honoured.

        The plan names a building *kind*; the nearest one is resolved now rather
        than when the plan was written, so it stays sensible even if the agent
        has ended up on the other side of the city since.
        """
        if step.place == "home":
            target = self.world.buildings.get(agent.home_id)
        else:
            try:
                target = self.world.nearest(BuildingKind(step.place), agent.pos)
            except ValueError:
                return None
        if target is None:
            return None

        agent.memory.add(
            self.clock.tick, "plan", f"Went to {target.name} to {step.kind.value}: {step.why}"
        )
        return self._travel_to(agent, target, step.kind)

    def _remedy(self, agent: Agent, need: str) -> Action:
        """Tier 0. Free, deterministic, and always has an answer."""
        kind, place = NEED_REMEDY[need]
        target = (
            self.world.buildings[agent.home_id]
            if place is None
            else self.world.nearest(place, agent.pos)
        )
        if target is None:
            return Action(ActionKind.IDLE)
        return self._travel_to(agent, target, kind)

    def _interior_spot(self, building_id: str, agent: Agent) -> tuple[int, int] | None:
        """A stable spot inside a building for this agent.

        Indexed by agent number rather than chosen at random, so each person
        keeps the same desk or bed instead of shuffling around on every visit.
        """
        spots = self._interiors.get(building_id)
        if not spots:
            return None
        return spots[int(agent.id[1:]) % len(spots)]

    def _travel_to(self, agent: Agent, building: Building, then_kind: ActionKind) -> Action:
        arrive = Action(then_kind, target_id=building.id)
        # Walk all the way to the spot indoors, not just to the doorstep.
        # Stopping at the door and snapping inside was a visible teleport at the
        # end of every journey.
        goal = self._interior_spot(building.id, agent) or building.door
        path = self.paths.path(agent.pos, goal)
        if path is None:
            # Unreachable. Idle and retry — the cache remembers the failure, so
            # the retry is a dict hit rather than another search.
            return Action(ActionKind.IDLE)
        if not path:
            return arrive
        return Action(ActionKind.TRAVEL, target_id=building.id, path=path, then=arrive)

    #: Sim-minutes before the same two agents may talk again.
    TALK_COOLDOWN_MINUTES = 180

    def encounters(self) -> list[tuple[Agent, Agent]]:
        """Pairs currently able to hold a conversation.

        Both must be stationary, awake, and inside the same building. Sleepers,
        travellers and idlers are excluded — there is nobody to talk to on a road
        tile, and an agent asleep is not available.
        """
        by_place: dict[str, list[Agent]] = {}
        for agent in self.agents:
            action = agent.action
            if action.target_id is None:
                continue
            if action.kind in (ActionKind.TRAVEL, ActionKind.IDLE, ActionKind.SLEEP):
                continue
            by_place.setdefault(action.target_id, []).append(agent)

        cooldown = self.TALK_COOLDOWN_MINUTES // CONFIG.world.minutes_per_tick
        # Seeded on the tick: pairings rotate so two regulars don't monopolise a
        # busy cafe, but the run stays reproducible.
        shuffler = Random(self.clock.tick)

        pairs: list[tuple[Agent, Agent]] = []
        for group in by_place.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda a: a.id)
            shuffler.shuffle(group)
            for i in range(0, len(group) - 1, 2):
                a, b = group[i], group[i + 1]
                rel = a.relationships.get(b.id)
                if rel and self.clock.tick - rel.last_talked_tick < cooldown:
                    continue
                pairs.append((a, b))
        return pairs

    def _interest(self, a: Agent, b: Agent) -> float:
        """What a generated conversation between these two would be worth."""
        rel = a.relationships.get(b.id)
        hours = (
            (self.clock.tick - rel.last_talked_tick) * CONFIG.world.minutes_per_tick / 60.0
            if rel is not None
            else 1e4  # never met: effectively infinite time since
        )
        return conversation_interest(
            relationship=rel,
            social_a=a.needs.social,
            social_b=b.needs.social,
            hours_since=hours,
        )

    def greet(self, a: Agent, b: Agent) -> None:
        """Tier 0 encounter — what happens to the overwhelming majority.

        Two people cross paths, acknowledge each other, feel slightly less alone,
        and drift a little closer or further apart according to temperament. No
        model involved: at 1.5 encounters per tick against a budget of 0.27, only
        about one in five can afford words.
        """
        tick = self.clock.tick
        for x, y in ((a, b), (b, a)):
            rel = x.relationships.setdefault(y.id, Relationship())
            if rel.times_met == 0:
                x.memory.add(tick, "observation", f"Met {y.name} for the first time")
            rel.times_met += 1
            rel.last_talked_tick = tick
            rel.affinity = max(
                -100.0,
                min(100.0, rel.affinity + trait_rapport(x.traits, y.traits) * 0.4),
            )
            # Partial on purpose: casual contact eases loneliness without
            # removing the reason to go looking for company.
            x.needs.restore("social", 2.0)


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