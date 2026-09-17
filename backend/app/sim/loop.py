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
from app.agents.relationships import Relationship, conversation_interest, trait_rapport
from app.cognition.prompts import PlanStep
from app.config import CONFIG
from app.institutions.university import (
    BEDTIME_ENERGY,
    DAYTIME_NAP_FLOOR,
    PASS_MARK,
    Enrollment,
    attends,
    baseline_score,
    choose_course,
    is_bedtime,
    session_now,
    session_starts_now,
    study_gain,
)
from app.sim.clock import TICKS_PER_DAY, Clock
from app.sim.layout import build_city
from app.sim.pathing import PathCache
from app.sim.spawn import spawn_agents
from app.sim.world import Building, BuildingKind, TileKind

#: Need -> (what fixes it, where it happens). None means the agent's own home.
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
        # Interior floor tiles per building, so arrivals step inside rather
        # than stand on the doorstep.
        self._interiors: dict[str, list[tuple[int, int]]] = {
            b.id: [t for t in b.tiles() if self.world.tile(*t) is TileKind.FLOOR]
            for b in self.world.buildings.values()
        }
        self.events: deque[tuple[int, str]] = deque(maxlen=200)
        #: Total events ever logged. The deque is capped, so its length is not a
        #: usable cursor; this never resets.
        self.events_total = 0
        self.agents = spawn_agents(self.world, self.rng)
        self.by_id = {a.id: a for a in self.agents}
        # spawn builds actions without a deadline; _begin stamps one, otherwise
        # the opening IDLE never ends.
        for agent in self.agents:
            self._begin(agent, agent.action)
        #: (a, b, interest) for pairs that met this tick, scored before greeting.
        self.last_encounters: list[tuple[Agent, Agent, float]] = []
        self.pending_exams: list[Agent] = []
        #: Starts at day 0 so nobody pays rent on the morning they arrive.
        self._billed_day = self.clock.day

    def tick(self) -> None:
        self.clock.advance()
        # Order matters: offered sessions are stamped before anyone can attend
        # one, and a term that ran out is marked before its next session.
        self._mark_sessions()
        self._check_terms()
        self._bill_day()
        hours = CONFIG.world.minutes_per_tick / 60.0
        for agent in self.agents:
            self._step(agent, hours)
        # Scored before greeting: greet() stamps the facts the score reads, so
        # the reverse order makes every pair look like acquaintances who just spoke.
        self.last_encounters = [(a, b, self._interest(a, b)) for a, b in self.encounters()]
        for a, b, _ in self.last_encounters:
            self.greet(a, b)

    def _step(self, agent: Agent, hours: float) -> None:
        agent.tick_needs(hours)
        self._redirect_to_class(agent)
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

        Duration is applied here rather than at construction: a chained `then`
        is built before the walk begins, and its clock must start on arrival.
        """
        if action.kind is not ActionKind.TRAVEL and action.ends_at < 0:
            minutes = DURATION_MINUTES.get(action.kind, 10)
            action.ends_at = self.clock.tick + max(1, minutes // CONFIG.world.minutes_per_tick)

        agent.action = action

        if action.kind not in (ActionKind.TRAVEL, ActionKind.IDLE):
            where = self.world.buildings[action.target_id].name if action.target_id else "here"
            self.log(f"{agent.name} began {action.kind.value} at {where}")
            # Sleep happens nightly and tells nobody anything.
            if action.kind is not ActionKind.SLEEP:
                agent.memory.add(
                    self.clock.tick, "observation", f"{action.kind.value.capitalize()} at {where}"
                )

        if action.kind is ActionKind.EAT:
            before = agent.money
            agent.money -= CONFIG.economy.meal_cost
            self._note_broke(agent, before)

        # Credited on arrival: actions are never interrupted, so starting the
        # session is the same event as sitting it.
        if action.kind is ActionKind.STUDY and agent.enrollment is not None:
            e = agent.enrollment
            if action.target_id == e.course.campus_id:
                # Any study at the campus teaches; only a session the timetable
                # dispatched signs the register, so the plan layer cannot forge it.
                if action.for_class and not e.awaiting_exam:
                    e.sessions_attended += 1
                agent.skills[e.course.skill] = min(
                    100.0,
                    agent.skills[e.course.skill]
                    + study_gain(agent.skills[e.course.skill], e.course.difficulty),
                )

    def log(self, text: str) -> None:
        """Record a feed event. Counted as well as stored: asynchronous events
        land ticks after their frame went out, so readers track the total."""
        self.events_total += 1
        self.events.append((self.clock.tick, text))

    # ----------------------------------------------------------------- deciding

    def _choose_action(self, agent: Agent) -> Action:
        """Five layers, strict priority: critical need, timetable, bedtime,
        plan, lowest need. Tier 0 is the floor and always has an answer."""
        critical = agent.needs.critical()
        if critical is not None:
            return self._remedy(agent, critical)

        session = self._due_session(agent)
        if session is not None:
            return session

        bed = self._bedtime(agent)
        if bed is not None:
            return bed

        step = self._due_step(agent)
        if step is not None:
            action = self._follow(agent, step)
            if action is not None:
                return action

        need = agent.needs.lowest()[0]
        # A student with energy to spare does not nap at ten in the morning;
        # if energy is genuinely low the critical check already caught it.
        if (
            need == "energy"
            and agent.enrollment is not None
            and not is_bedtime(self.clock.hour)
            and agent.needs.energy >= DAYTIME_NAP_FLOOR
        ):
            need = min(
                ((k, v) for k, v in agent.needs.as_dict().items() if k != "energy"),
                key=lambda kv: kv[1],
            )[0]
        return self._remedy(agent, need)

    def _due_session(self, agent: Agent) -> Action | None:
        """The class this agent should be at right now, if any."""
        enrollment = agent.enrollment
        if enrollment is None:
            return None
        course = enrollment.course
        if not session_now(course, self.clock.weekday, self.clock.minute_of_day):
            return None
        if not self._will_attend(agent):
            return None
        # Already in the right room: re-deciding would restart the action and
        # the attendance credit with it.
        if agent.action.kind is ActionKind.STUDY and agent.action.target_id == course.campus_id:
            return None
        campus = self.world.buildings.get(course.campus_id)
        if campus is None:
            return None
        return self._travel_to(agent, campus, ActionKind.STUDY, for_class=True)

    def _redirect_to_class(self, agent: Agent) -> None:
        """Turn a journey around when class opens.

        The one exception to "actions have duration": nobody is committed to
        the middle of a walk. Journeys run longer than the join window, so
        without this a third of missed sessions were agents already on foot.
        """
        if agent.enrollment is None or agent.action.kind is not ActionKind.TRAVEL:
            return
        # Redirecting a starving agent away from the cafe starts a spiral.
        if agent.needs.critical() is not None:
            return

        course = agent.enrollment.course
        if not session_now(course, self.clock.weekday, self.clock.minute_of_day):
            return
        if not self._will_attend(agent):
            return

        then = agent.action.then
        if then is not None and then.kind is ActionKind.STUDY and then.target_id == course.campus_id:
            return  # already on the way

        campus = self.world.buildings.get(course.campus_id)
        if campus is None:
            return
        replacement = self._travel_to(agent, campus, ActionKind.STUDY, for_class=True)
        # Unreachable campus: leave the original journey alone rather than
        # stranding the agent mid-errand.
        if replacement.kind is ActionKind.IDLE:
            return
        self._begin(agent, replacement)

    def _will_attend(self, agent: Agent) -> bool:
        """Whether temperament sends this agent to today's class."""
        e = agent.enrollment
        return e is not None and attends(agent.id, e.course_id, self.clock.day, agent.traits)

    def _bedtime(self, agent: Agent) -> Action | None:
        """A student turning in for the night. Students only — a timetable is
        what imposes a routine."""
        if agent.enrollment is None or agent.action.kind is ActionKind.SLEEP:
            return None
        if not is_bedtime(self.clock.hour):
            return None
        if agent.needs.energy >= BEDTIME_ENERGY:
            return None
        return self._remedy(agent, "energy")

    #: How late a plan step may be and still worth doing, in sim-minutes.
    STEP_GRACE_MINUTES = 120

    def _due_step(self, agent: Agent) -> PlanStep | None:
        """The earliest step that is due and still worth doing.

        One step at a time, so an agent running late keeps following its plan
        in order; only genuinely ancient steps are dropped.
        """
        now = self.clock.minute_of_day
        while agent.plan:
            step = agent.plan[0]
            if step.at > now:
                return None
            agent.plan.pop(0)
            if now - step.at <= self.STEP_GRACE_MINUTES:
                return step
        return None

    def _follow(self, agent: Agent, step: PlanStep) -> Action | None:
        """Turn a plan step into an action, or None if it cannot be honoured.

        The plan names a building *kind*; the nearest one is resolved now, not
        when the plan was written.
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

    def _travel_to(
        self,
        agent: Agent,
        building: Building,
        then_kind: ActionKind,
        *,
        for_class: bool = False,
    ) -> Action:
        arrive = Action(then_kind, target_id=building.id, for_class=for_class)
        # Walk to the spot indoors, not the doorstep — snapping inside was a
        # visible teleport at the end of every journey.
        goal = self._interior_spot(building.id, agent) or building.door
        path = self.paths.path(agent.pos, goal)
        if path is None:
            # Unreachable. Idle and retry; the cache remembers the failure.
            return Action(ActionKind.IDLE)
        if not path:
            return arrive
        return Action(ActionKind.TRAVEL, target_id=building.id, path=path, then=arrive)

    def _interior_spot(self, building_id: str, agent: Agent) -> tuple[int, int] | None:
        """A stable spot inside a building for this agent — the same desk or
        bed every visit, rather than a shuffle."""
        spots = self._interiors.get(building_id)
        if not spots:
            return None
        return spots[int(agent.id[1:]) % len(spots)]

    # --------------------------------------------------------------- university

    def _mark_sessions(self) -> None:
        """Count what the timetable offered, attended or not. Attendance is a
        ratio, and the denominator is the classes a student could have been at."""
        weekday, minute = self.clock.weekday, self.clock.minute_of_day
        for agent in self.agents:
            e = agent.enrollment
            # Term's over: counting what they cannot attend would dock
            # attendance for the wait itself.
            if e is None or e.awaiting_exam:
                continue
            if session_starts_now(e.course, weekday, minute):
                e.sessions_offered += 1

    #: Ticks a paper may wait for a model-graded result before the sim marks it.
    #: A full day: this fires when nobody is coming for the paper at all, not
    #: merely when the model is slow — Enrollment.in_flight covers slow.
    EXAM_GRACE_TICKS = TICKS_PER_DAY

    def _check_terms(self) -> None:
        """End terms that are up; mark the paper if nobody else has."""
        for agent in self.agents:
            e = agent.enrollment
            if e is None:
                continue
            if not e.awaiting_exam and e.term_over(self.clock.tick):
                e.awaiting_exam = True
                e.exam_due_tick = self.clock.tick
                self.pending_exams.append(agent)
            elif (
                e.awaiting_exam
                and not e.in_flight
                and self.clock.tick - e.exam_due_tick >= self.EXAM_GRACE_TICKS
            ):
                self.sit_exam(agent)

    def sit_exam(self, agent: Agent, score: float | None = None) -> tuple[float, bool]:
        """Record a result. score=None grades deterministically.

        The Hub passes a model's mark when it has one. What the mark means,
        whether it is a pass, and what happens next are decided here, because
        those are numbers and the simulation owns numbers.
        """
        e = agent.enrollment
        course = e.course
        # The mark when nobody graded the paper; the centre of the band when
        # somebody did.
        expected = baseline_score(agent.skills[course.skill], e.attendance, course.difficulty)
        if score is None:
            score = expected
        else:
            band = CONFIG.university.exam_mark_band
            score = max(expected - band, min(expected + band, score))
        passed = score >= PASS_MARK

        if agent in self.pending_exams:
            self.pending_exams.remove(agent)

        agent.memory.add(
            self.clock.tick,
            "milestone",
            f"{'Passed' if passed else 'Failed'} {course.name} with {score:.0f}"
            f" (attended {e.sessions_attended} of {e.sessions_offered})",
        )
        self.log(
            f"{agent.name} {'passed' if passed else 'failed'} {course.name} — {score:.0f}"
        )

        if passed:
            agent.credentials.append(course.id)
            nxt = choose_course(agent.skills, agent.credentials)
            agent.enrollment = (
                Enrollment(course_id=nxt.id, started_tick=self.clock.tick) if nxt else None
            )
        elif e.attempt >= CONFIG.university.max_attempts:
            # Out of attempts: a different subject. They keep the failure and
            # stop grinding a wall.
            nxt = choose_course(agent.skills, agent.credentials, avoid=course.id)
            agent.enrollment = (
                Enrollment(course_id=nxt.id, started_tick=self.clock.tick) if nxt else None
            )
        else:
            # Retake: counters reset, skill kept.
            agent.enrollment = Enrollment(
                course_id=course.id,
                started_tick=self.clock.tick,
                attempt=e.attempt + 1,
            )
        return score, passed

    # -------------------------------------------------------------------- money

    def _bill_day(self) -> None:
        """Once a sim-day, at midnight: rent from everyone, tuition from
        students, a stipend to the unemployed."""
        if self.clock.day == self._billed_day:
            return
        self._billed_day = self.clock.day
        eco = CONFIG.economy
        for agent in self.agents:
            before = agent.money
            agent.money -= eco.daily_rent
            if agent.enrollment is not None:
                agent.money -= eco.tuition_per_day
            if not agent.employed:
                agent.money += eco.unemployment_stipend
            self._note_broke(agent, before)

    def _note_broke(self, agent: Agent, before: float) -> None:
        """Mark the crossing into debt, once per crossing. Balances may go
        negative: nobody earns until Phase 5."""
        if before >= 0.0 > agent.money:
            agent.memory.add(self.clock.tick, "milestone", "Ran out of money")
            self.log(f"{agent.name} has run out of money")

    # ------------------------------------------------------------------- people

    #: Sim-minutes before the same two agents may talk again.
    TALK_COOLDOWN_MINUTES = 180

    def encounters(self) -> list[tuple[Agent, Agent]]:
        """Pairs currently able to hold a conversation: stationary, awake, and
        inside the same building."""
        by_place: dict[str, list[Agent]] = {}
        for agent in self.agents:
            action = agent.action
            if action.target_id is None:
                continue
            if action.kind in (ActionKind.TRAVEL, ActionKind.IDLE, ActionKind.SLEEP):
                continue
            by_place.setdefault(action.target_id, []).append(agent)

        cooldown = self.TALK_COOLDOWN_MINUTES // CONFIG.world.minutes_per_tick
        # Seeded on the tick: pairings rotate, the run stays reproducible.
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
            else 1e4  # never met
        )
        return conversation_interest(
            relationship=rel,
            social_a=a.needs.social,
            social_b=b.needs.social,
            hours_since=hours,
        )

    def greet(self, a: Agent, b: Agent) -> None:
        """Tier 0 encounter — what happens to the overwhelming majority. Two
        people cross paths, feel slightly less alone, and drift a little closer
        or further apart according to temperament."""
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
