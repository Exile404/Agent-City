"""The tick loop: the city's heartbeat.

Everything here is Tier 0 — deterministic Python, no model involved. Fifty
agents cost microseconds a tick, which is the whole premise: the simulation is
free, only thinking is expensive.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from random import Random

from app.agents.actions import DURATION_MINUTES, Action, ActionKind
from app.agents.agent import Agent
from app.agents.relationships import Relationship, conversation_interest, trait_rapport
from app.cognition.prompts import PlanStep
from app.config import CONFIG
from app.institutions.companies import (
    APPLY_COOLDOWN_TICKS,
    FINE_WORK,
    FIRING_ATTENDANCE,
    FIRING_GRACE_SHIFTS,
    HIRE_MARK,
    MANAGING_ROLES,
    POOR_WORK,
    REJECTION_COOLDOWN_DAYS,
    REVIEW_EVERY,
    WARNINGS_TO_FIRE,
    Application,
    Job,
    Review,
    allowed_verdicts,
    best_vacancy,
    clears_door,
    do_task,
    expected_verdict,
    headcount,
    interview_score,
    live_warnings,
    ladder_above,
    leadership_gain,
    shift_closes_now,
    shift_now,
    shift_starts_now,
    shows_up,
    vacancies,
    work_gain,
)
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
from app.sim.spawn import birthday, newcomer, spawn_agents
from app.sim.world import Building, BuildingKind, TileKind
from app.institutions.companies import BY_ID as ROLES

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
        #: Every name ever given, so a newcomer never arrives wearing a retiree's.
        self._used_names = {a.name for a in self.agents}
        self._next_agent = len(self.agents)
        self._aged_day = -1
        # spawn builds actions without a deadline; _begin stamps one, otherwise
        # the opening IDLE never ends.
        for agent in self.agents:
            self._begin(agent, agent.action)
        #: (a, b, interest) for pairs that met this tick, scored before greeting.
        self.last_encounters: list[tuple[Agent, Agent, float]] = []
        self.pending_exams: list[Agent] = []
        #: Candidates sitting in a lobby waiting on a verdict.
        self.pending_interviews: list[Agent] = []
        #: Starts at day 0 so nobody pays rent on the morning they arrive.
        self._billed_day = self.clock.day
        #: Shifts completed and paid since boot. Never resets.
        self.shifts_worked = 0
        #: Shifts the roster called since boot, worked or not. The denominator
        #: for attendance, exactly as sessions_offered is for class.
        self.shifts_offered = 0
        #: What people were doing instead, when they missed a shift.
        self.shifts_missed: Counter[str] = Counter()
        #: Journeys actually turned around. Separates "never fires" from
        #: "fires and does not help".
        self.redirects = 0
        self.hires = 0
        self.rejections = 0
        self.firings = 0
        #: Every paid shift should produce exactly one. Counted apart from
        #: shifts_worked so a shift that paid without being scored shows as a gap.
        self.tasks_done = 0
        #: Of those, done below FATIGUE_LINE.
        self.tired_tasks = 0
        #: Skill points handed out by work rather than by the university.
        self.work_skill_gained = 0.0
        #: Due for review, waiting for a model or the grace timer.
        self.pending_reviews: list[Agent] = []
        #: How long a due review waits before the sim writes it from the numbers.
        #: The Hub stretches it when a model is reviewing.
        self.review_grace = self.REVIEW_GRACE_TICKS
        #: Verdicts actually recorded, however they were reached.
        self.review_verdicts: Counter[str] = Counter()
        self.promotions = 0
        #: Terms an employer paid for, and dismissals at the second warning.
        self.sponsored = 0
        self.dismissals = 0
        #: Promotions that crossed to a sister employer.
        self.transfers = 0
        #: Management learned by running a team rather than on a course.
        self.management_learned = 0.0
        self.retirements = 0
        #: Bumped when someone leaves and someone new moves in. The frontend
        #: learns names once, when it connects, and has to be told.
        self.roster_version = 0

    def tick(self) -> None:
        self.clock.advance()
        # Order matters: offered sessions are stamped before anyone can attend
        # one, and a term that ran out is marked before its next session.
        self._mark_sessions()
        self._mark_shifts()
        self._check_terms()
        self._check_interviews()
        self._check_reviews()
        self._bill_day()
        self._age_city()
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
        self._redirect(agent)
        action = agent.action

        if action.kind is ActionKind.TRAVEL:
            for _ in range(CONFIG.world.tiles_per_tick):
                if not action.path:
                    break
                agent.move_to(action.path.pop(0))
            if not action.path:
                self._begin(agent, action.then or Action(ActionKind.IDLE))
        elif action.is_done(self.clock.tick):
            self._finish_shift(agent, action)
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

        if action.kind is ActionKind.INTERVIEW and agent.application is not None:
            # The clock starts when they sit down, not when they set off.
            agent.application.filed_tick = self.clock.tick
            if agent not in self.pending_interviews:
                self.pending_interviews.append(agent)

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

        shift = self._due_shift(agent)
        if shift is not None:
            return shift

        applying = self._seek_work(agent)
        if applying is not None:
            return applying

        bed = self._bedtime(agent)
        if bed is not None:
            return bed

        step = self._due_step(agent)
        if step is not None:
            action = self._follow(agent, step)
            if action is not None:
                return action

        need = agent.needs.lowest()[0]
        # Someone with energy to spare does not nap at ten in the morning; if
        # energy is genuinely low the critical check already caught it. Job-
        # seekers included: a daytime sleep runs a full night's length, and one
        # taken while looking for work is carried into the job that follows.
        if (
            need == "energy"
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

    def _due_shift(self, agent: Agent) -> Action | None:
        """The shift this agent should be at right now, if any.

        Two ways to miss one, and they mean different things: temperament
        decides whether they set off at all, and a critical need can outrank
        the ones they meant to make.
        """
        job = agent.job
        if job is None or self.seconded(agent):
            return None
        if not shift_now(job.role, self.clock.weekday, self.clock.minute_of_day):
            return None
        if not shows_up(agent.id, job.employer_id, self.clock.day, agent.traits):
            return None
        # Already on the clock: re-deciding would restart the shift and pay twice.
        if agent.action.kind is ActionKind.WORK and agent.action.for_shift:
            return None
        employer = self.world.buildings.get(job.employer_id)
        if employer is None:
            return None
        return self._travel_to(agent, employer, ActionKind.WORK, for_shift=True)

    def _seek_work(self, agent: Agent) -> Action | None:
        """An unemployed agent going after a job.

        Also where someone who qualifies for nothing decides to go and fix
        that. There are only two answers to being out of work — apply, or go
        and become worth hiring — and this picks between them.
        """
        if agent.job is not None or agent.application is not None:
            return None
        if agent.enrollment is not None:
            return None  # finish the term first
        if self.clock.tick - agent.last_applied_tick < APPLY_COOLDOWN_TICKS:
            return None

        all_open = vacancies(self.world, headcount(a.job for a in self.agents))
        if not all_open:
            # Nobody is hiring. A degree does not conjure a vacancy, so they
            # wait on the stipend rather than pay tuition to sit still.
            return None

        cooldown = REJECTION_COOLDOWN_DAYS * TICKS_PER_DAY
        reachable = [
            p
            for p in all_open
            if self.clock.tick - agent.rejected_by.get(p.employer_id, -10_000) >= cooldown
        ]
        posting = best_vacancy(agent.skills, agent.credentials, reachable)
        if posting is None:
            # There is work, so either they cannot do it yet or everywhere that
            # would have them has turned them down lately. The university
            # answers the first; only time answers the second.
            if best_vacancy(agent.skills, agent.credentials, all_open) is None:
                self._enroll(agent)
            return None

        employer = self.world.buildings.get(posting.employer_id)
        if employer is None:
            return None
        agent.last_applied_tick = self.clock.tick
        agent.application = Application(posting=posting)
        return self._travel_to(agent, employer, ActionKind.INTERVIEW)

    def _enroll(self, agent: Agent, prefer: str | None = None) -> None:
        """Send someone to the university."""
        course = choose_course(agent.skills, agent.credentials, prefer=prefer)
        if course is None:
            return
        agent.enrollment = Enrollment(course_id=course.id, started_tick=self.clock.tick)
        agent.memory.add(self.clock.tick, "milestone", f"Enrolled in {course.name}")
        self.log(f"{agent.name} enrolled in {course.name}")

    def _redirect(self, agent: Agent) -> None:
        """Turn a journey around when a timetable calls.

        The one exception to "actions have duration": nobody is committed to
        the middle of a walk. Journeys outlast both join windows, and being
        already on foot was the largest single cause of missed classes — and,
        measured at nine of sixteen, of missed shifts too.
        """
        if agent.action.kind is not ActionKind.TRAVEL:
            return
        # Redirecting a starving agent away from the cafe starts a spiral.
        if agent.needs.critical() is not None:
            return
        # A timetable already dispatched this walk; leave it alone. Also what
        # stops the redirect firing again every tick once it has fired once.
        then = agent.action.then
        if then is not None and (then.for_class or then.for_shift):
            return

        # Class before work: a term is finite, a shift comes round tomorrow.
        # Nobody holds both yet — slice 2 is where that choice gets real.
        due = self._due_session(agent) or self._due_shift(agent)
        # Unreachable: leave the original journey alone rather than stranding
        # the agent mid-errand.
        if due is None or due.kind is ActionKind.IDLE:
            return
        self.redirects += 1
        self._begin(agent, due)

    def _will_attend(self, agent: Agent) -> bool:
        """Whether temperament sends this agent to today's class.

        A course the employer pays for is attended like a job, not a lecture.
        On the student rate, 4 of 11 sponsored terms were failed by staff who
        turn up for work reliably skipping class on full pay.
        """
        e = agent.enrollment
        if e is None:
            return False
        if e.sponsor is not None:
            return shows_up(agent.id, e.course_id, self.clock.day, agent.traits)
        return attends(agent.id, e.course_id, self.clock.day, agent.traits)

    def _bedtime(self, agent: Agent) -> Action | None:
        """Everyone turning in for the night.

        Left to drift, sleep wanders across the clock and eats whatever the
        morning was supposed to hold. Nobody is exempt: someone looking for
        work is one interview from having a morning, and a new hire who
        arrives sleeping days is caught in it — an exhausted shift earns an
        afternoon sleep, they wake as this window opens, and it never fires.
        """
        if agent.action.kind is ActionKind.SLEEP:
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

        # A plan that says "work" means your own job. Without this an employed
        # agent walks into whichever office is nearest and does a day unpaid.
        on_shift = False
        if step.kind is ActionKind.WORK and agent.job is not None:
            target = self.world.buildings.get(agent.job.employer_id) or target
            # Arriving inside the roster's own window is the shift, whoever
            # thought of it. The window is the sim's, so nothing is forged.
            on_shift = shift_now(
                agent.job.role, self.clock.weekday, self.clock.minute_of_day
            )

        agent.memory.add(
            self.clock.tick, "plan", f"Went to {target.name} to {step.kind.value}: {step.why}"
        )

        return self._travel_to(agent, target, step.kind, for_shift=on_shift)


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
        for_shift: bool = False,
    ) -> Action:
        arrive = Action(
            then_kind, target_id=building.id, for_class=for_class, for_shift=for_shift
        )
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

        # A course the employer paid for is one term, pass or fail, then back
        # to the job — with the promotion if it closed the gap and a seat is free.
        if e.sponsor is not None:
            agent.enrollment = None
            job = agent.job
            if self.ready_for_promotion(agent):
                site = self._open_seat_for(agent)
                if site is not None:
                    self._promote(agent, site)
            return score, passed

        # A term has ended. Work beats another term whatever the mark, because
        # studying is what someone does when they cannot get hired, not a
        # career. Checked on a failure as much as on a pass: otherwise a weak
        # student enrols forever and never looks up, which had sixteen of them
        # sinking at 88 a day past a vacancy they could have walked into.
        open_seats = vacancies(self.world, headcount(a.job for a in self.agents))
        if best_vacancy(agent.skills, agent.credentials, open_seats) is not None:
            agent.enrollment = None
        elif passed:
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

    # --------------------------------------------------------------------- work

    def _mark_shifts(self) -> None:
        """Count what the roster called, and what people did instead.

        Offered at the bell, missed when the window shuts — the two ends of the
        same window, so a shift is counted once and judged once.
        """
        weekday, minute = self.clock.weekday, self.clock.minute_of_day
        for agent in self.agents:
            job = agent.job
            if job is None or self.seconded(agent):
                continue
            if shift_starts_now(job.role, weekday, minute):
                self.shifts_offered += 1
                job.shifts_offered += 1
            if not shift_closes_now(job.role, weekday, minute):
                continue
            if self._on_shift(agent):
                job.shifts_attended += 1
            else:
                urgent = "/urgent" if agent.needs.critical() else ""
                self.shifts_missed[agent.action.kind.value + urgent] += 1
                # Judged only on a day they failed to appear: nobody is
                # dismissed on a morning they turned up for.
                if job.shifts_offered >= FIRING_GRACE_SHIFTS and job.attendance < FIRING_ATTENDANCE:
                    self._fire(agent)
                    continue
            self._queue_review(agent)

    @staticmethod
    def _on_shift(agent: Agent) -> bool:
        """At work, or on the way there. Arriving late still counts — the
        journey was dispatched inside the window even if the walk outlived it."""
        action = agent.action
        if action.kind is ActionKind.WORK and action.for_shift:
            return True
        return (
            action.kind is ActionKind.TRAVEL
            and action.then is not None
            and action.then.for_shift
        )

    def manager_of(self, agent: Agent) -> Agent | None:
        """Whoever holds the nearest filled rung above this agent's, at the
        same employer. An empty rung is skipped, so a clerk whose records
        office is vacant answers to the ward manager instead.

        Derived on every call, never stored, for the same reason as headcount:
        the agents' own jobs are the one record of who works where, and a
        stored org chart would go stale the first time anybody was let go.
        Two holders of the rung above — only the records office has that —
        means the first on the roster takes the whole team.
        """
        job = agent.job
        if job is None:
            return None
        for boss in ladder_above(job.role):
            for other in self.agents:
                j = other.job
                if j is not None and j.employer_id == job.employer_id and j.role_id == boss.id:
                    return other
        return None

    def reports_of(self, agent: Agent) -> list[Agent]:
        """Everyone whose manager this agent is."""
        if agent.job is None:
            return []
        employer = agent.job.employer_id
        return [
            other
            for other in self.agents
            if other.job is not None
            and other.job.employer_id == employer
            and self.manager_of(other) is agent
        ]

    def _finish_shift(self, agent: Agent, action: Action) -> None:
        """A shift that ran its course: paid, the day's task scored, and a
        little learned.

        Whoever dispatched the agent signs for it, exactly as with the class
        register — so someone who skips work earns less, learns less and has
        less to show at review, without a single rule being written about
        skipping. Actions are never interrupted, so a shift begun is a shift
        worked.
        """
        job = agent.job
        if not action.for_shift or job is None:
            return
        if action.kind is not ActionKind.WORK or action.target_id != job.employer_id:
            return
        agent.money += job.role.wage
        self.shifts_worked += 1

        skill = job.role.skill
        # Scored on the skill they walked in with and the energy they walk out
        # with. What today taught them shows up tomorrow.
        title, quality, tired = do_task(
            agent.id, job, self.clock.day, agent.skills[skill], agent.needs.energy
        )
        job.record(title, quality, tired)
        self.tasks_done += 1
        self.tired_tasks += tired
        gain = work_gain(agent.skills[skill], job.role)
        agent.skills[skill] += gain
        self.work_skill_gained += gain
        # Running a team teaches management, the way doing the job teaches the
        # job's own skill.
        if (
            skill != "management"
            and job.role_id in MANAGING_ROLES
            and self.reports_of(agent)
        ):
            lead = leadership_gain(agent.skills["management"])
            agent.skills["management"] += lead
            self.management_learned += lead

        if POOR_WORK < quality < FINE_WORK:
            return
        good = quality >= FINE_WORK
        employer = self.world.buildings[job.employer_id].name
        agent.memory.add(
            self.clock.tick,
            "observation",
            f"{'Good' if good else 'Bad'} day at {employer}: {title.lower()}"
            + (", worn out" if tired else ""),
            importance=5.0,
        )
        self.log(f"{agent.name} had a {'good' if good else 'bad'} day at {employer}")

    #: Ticks a review may wait on a model before the simulation writes it
    #: itself. Half a day: nobody is waiting on it, but it should land in the
    #: same working week as the shifts it judges.
    REVIEW_GRACE_TICKS = TICKS_PER_DAY // 2

    def _queue_review(self, agent: Agent) -> None:
        """Put someone up for review once the roster has called enough shifts."""
        job = agent.job
        if job.review_due_tick >= 0:
            return
        if job.shifts_offered - job.reviewed_offered < REVIEW_EVERY:
            return
        job.review_due_tick = self.clock.tick
        self.pending_reviews.append(agent)

    def _check_reviews(self) -> None:
        """Write any review nobody has come for, and drop any whose seat is gone."""
        for agent in list(self.pending_reviews):
            job = agent.job
            if job is None or job.review_due_tick < 0:
                self.pending_reviews.remove(agent)
            elif (
                not job.review_in_flight
                and self.clock.tick - job.review_due_tick >= self.review_grace
            ):
                self.hold_review(agent)

    def reviewer_of(self, agent: Agent) -> str:
        """Who signs the review: the manager, or the employer itself when
        every rung above is empty."""
        boss = self.manager_of(agent)
        if boss is not None:
            return boss.name
        return f"the management at {self.world.buildings[agent.job.employer_id].name}"

    def hold_review(self, agent: Agent, verdict: str | None = None, comment: str = "") -> Review:
        """Record a review. verdict=None writes it from the numbers alone.

        The verdict is always the record's (allowed_verdicts); the Hub passes
        the model's reply for the words. What a verdict leads to is the
        simulation's: a promote moves them up if they qualify and a seat is
        free, or sends them on a course if they do not qualify; a second
        warning in one seat ends it.
        """
        job = agent.job
        top = job.role.reports_to is None
        attendance = job.window_attendance
        expected = expected_verdict(job.form, attendance, top)
        # The schema already restricts a model; this is the simulation owning
        # what is possible either way.
        if verdict is None or verdict not in allowed_verdicts(job.form, attendance, top):
            verdict = expected
        review = Review(
            self.clock.tick, self.reviewer_of(agent), verdict, comment,
            job.form, attendance, expected,
        )
        job.reviews.append(review)
        job.reviewed_offered, job.reviewed_attended = job.shifts_offered, job.shifts_attended
        job.review_due_tick = -1
        if agent in self.pending_reviews:
            self.pending_reviews.remove(agent)
        self.review_verdicts[verdict] += 1

        employer = self.world.buildings[job.employer_id].name
        agent.memory.add(
            self.clock.tick,
            "milestone",
            f"Review at {employer} by {review.reviewer}: {verdict}"
            + (f" — {comment}" if comment else ""),
        )
        self.log(
            f"{agent.name} was reviewed at {employer}: {verdict}"
            + (f" — {comment}" if comment else "")
        )
        if verdict == "warn" and live_warnings(job) >= WARNINGS_TO_FIRE:
            self.dismissals += 1
            self._fire(agent, f"after {WARNINGS_TO_FIRE} warnings")
        elif verdict == "promote":
            self._after_promote_verdict(agent)
        return review

    #: Ticks an application may wait for a model's verdict before the sim
    #: decides it. Six hours — a candidate left sitting in a lobby overnight is
    #: not a simulation of anything. Application.in_flight covers a slow model.
    INTERVIEW_GRACE_TICKS = TICKS_PER_DAY // 4

    def _check_interviews(self) -> None:
        """Decide any application nobody has come for."""
        for agent in list(self.pending_interviews):
            app = agent.application
            if app is None:
                self.pending_interviews.remove(agent)
            elif (
                not app.in_flight
                and self.clock.tick - app.filed_tick >= self.INTERVIEW_GRACE_TICKS
            ):
                self.hire_or_reject(agent)

    def _fire(self, agent: Agent, why: str | None = None) -> None:
        """Let someone go — for chronic absence, or at the second warning in
        one seat.

        The seat goes to anyone ready on the rung below first, then back to the
        market. They keep their skills and their credentials and go back to
        looking, with a gap in the record and that employer closed to them for
        a while.
        """
        job = agent.job
        employer = self.world.buildings[job.employer_id].name
        agent.job = None
        agent.rejected_by[job.employer_id] = self.clock.tick
        why = why or f"turned up for {job.shifts_attended} of {job.shifts_offered} shifts"
        agent.memory.add(
            self.clock.tick, "milestone", f"Let go as {job.role.title} at {employer} — {why}"
        )
        self.log(f"{agent.name} was let go from {employer}")
        self.firings += 1
        self._fill_from_within(job.employer_id, job.role_id)

    def hire_or_reject(self, agent: Agent, verdict: bool | None = None) -> bool:
        """Record an interview result. verdict=None decides it deterministically.

        The Hub passes the model's verdict when it has one — unclamped, because
        an interview is a judgement and judgement is the model's half of the
        bargain. Whether a seat is still there to take, and what a rejected
        candidate does next, are numbers, and stay here.
        """
        app = agent.application
        posting = app.posting
        role = posting.role
        earned = interview_score(agent.skills, agent.credentials, role)
        if verdict is None:
            verdict = earned >= HIRE_MARK

        # Re-checked now rather than when they applied: somebody else may have
        # taken the seat while this candidate was waiting.
        taken = headcount(a.job for a in self.agents)
        hired = verdict and taken[posting.key] < role.seats

        if agent in self.pending_interviews:
            self.pending_interviews.remove(agent)
        agent.application = None

        if hired:
            agent.job = Job(posting.employer_id, role.id, started_tick=self.clock.tick)
            agent.memory.add(
                self.clock.tick,
                "milestone",
                f"Hired as {role.title} at {posting.employer_name}",
            )
            self.log(f"{agent.name} was hired as {role.title} at {posting.employer_name}")
            self.hires += 1
            return True

        agent.rejected_by[posting.employer_id] = self.clock.tick
        agent.memory.add(
            self.clock.tick,
            "milestone",
            f"Turned down for {role.title} at {posting.employer_name} — {role.skill}",
        )
        self.log(f"{agent.name} was turned down at {posting.employer_name}")
        self.rejections += 1
        # If the numbers say they were never really ready, the way back is the
        # university. If they were ready and were turned down anyway, they try
        # somewhere else first.
        if earned < HIRE_MARK:
            self._enroll(agent, prefer=role.skill)
        return False

    # ---------------------------------------------------------------- careers

    @staticmethod
    def seconded(agent: Agent) -> bool:
        """Away on a course the employer is paying for. No shifts are called,
        so no tasks, no reviews, and no absence counted against a seat they
        are away from with permission."""
        e = agent.enrollment
        return e is not None and e.sponsor is not None

    def _seat_free(self, employer_id: str, role_id: str) -> bool:
        taken = headcount(a.job for a in self.agents)
        return taken[(employer_id, role_id)] < ROLES[role_id].seats

    def ready_for_promotion(self, agent: Agent) -> bool:
        """Recommended at their last review, through the rung above's ordinary
        door, and not away on a course. Derived, never stored."""
        job = agent.job
        if job is None or job.role.reports_to is None or not job.reviews:
            return False
        if job.reviews[-1].verdict != "promote" or self.seconded(agent):
            return False
        return clears_door(agent.skills, agent.credentials, ROLES[job.role.reports_to])

    def _after_promote_verdict(self, agent: Agent) -> None:
        """A review said promote. Up now if they qualify and a seat on the rung
        above is free here or at a sister employer; waiting if they qualify and
        none is; sent to learn the rung's skill if they do not qualify at all."""
        job = agent.job
        nxt = ROLES[job.role.reports_to]
        if self.ready_for_promotion(agent):
            site = self._open_seat_for(agent)
            if site is not None:
                self._promote(agent, site)
            return
        if agent.enrollment is None:
            self._sponsor(agent, nxt.skill)

    def _sponsor(self, agent: Agent, skill: str) -> None:
        """The employer pays for a term of the skill the rung above needs.

        Every ladder in the city changes skill at each rung, and everyone is
        hired into their strongest: measured over sixty days, the people
        recommended for promotion stood 18 to 38 points short of the next
        bar. No door credit bridges that without leaving them unable to do
        the job they were promoted into.
        """
        course = choose_course(agent.skills, agent.credentials, prefer=skill)
        if course is None or course.skill != skill:
            return  # every course in that skill already passed
        job = agent.job
        employer = self.world.buildings[job.employer_id].name
        agent.enrollment = Enrollment(
            course_id=course.id, started_tick=self.clock.tick, sponsor=job.employer_id
        )
        agent.memory.add(self.clock.tick, "milestone", f"Sent to study {course.name} by {employer}")
        self.log(f"{employer} sent {agent.name} to study {course.name}")
        self.sponsored += 1

    def _open_seat_for(self, agent: Agent) -> str | None:
        """Where the rung above this agent has a free seat: their own employer
        first, then any sister employer of the same kind. Only offices come in
        more than one building, so only office ladders ever cross."""
        job = agent.job
        nxt = ROLES[job.role.reports_to]
        sites = [job.employer_id] + sorted(
            b.id for b in self.world.of_kind(nxt.employer) if b.id != job.employer_id
        )
        return next((s for s in sites if self._seat_free(s, nxt.id)), None)

    def _promote(self, agent: Agent, employer_id: str | None = None) -> None:
        """Up one rung — at their own employer, or at a sister employer whose
        seat opened first. A fresh seat and a fresh record (tasks, reviews and
        attendance start again, as for a hire), and the seat left behind goes
        first to anyone ready for it.

        Direct, not an interview: the review already approved the promotion,
        and hire_or_reject assumes a candidate with no job to leave.
        """
        old = agent.job
        role = ROLES[old.role.reports_to]
        employer_id = employer_id or old.employer_id
        employer = self.world.buildings[employer_id].name
        moved = (
            "" if employer_id == old.employer_id
            else f", from {self.world.buildings[old.employer_id].name}"
        )
        agent.job = Job(employer_id, role.id, started_tick=self.clock.tick)
        agent.memory.add(
            self.clock.tick, "milestone", f"Promoted to {role.title} at {employer}{moved}"
        )
        self.log(f"{agent.name} was promoted to {role.title} at {employer}{moved}")
        self.promotions += 1
        self.transfers += bool(moved)
        self._fill_from_within(old.employer_id, old.role_id)

    def _fill_from_within(self, employer_id: str, role_id: str) -> None:
        """A seat just opened. Someone ready for it takes it before the market
        sees it — from the rung below at this employer first, then from the
        same rung at a sister employer. A promotion leaves a seat of its own,
        so this can cascade.

        Measured before transfers: at 120 days five people were ready and
        waiting, one of them for 94 days, while Analyst seats at other offices
        went to outsiders.
        """
        candidates = [
            a for a in self.agents
            if a.job is not None
            and a.job.role.reports_to == role_id
            and self.ready_for_promotion(a)
        ]
        if not candidates:
            return
        # Stable sort: their own staff first, then the roster's order.
        candidates.sort(key=lambda a: a.job.employer_id != employer_id)
        self._promote(candidates[0], employer_id)

    # -------------------------------------------------------------- lifetimes

    def _age_city(self) -> None:
        """Once a sim-day: birthdays, and anyone reaching retirement age
        retires. A year is CONFIG.population.days_per_year sim-days."""
        if self.clock.day == self._aged_day:
            return
        self._aged_day = self.clock.day
        year = CONFIG.population.days_per_year
        # A copy: _retire swaps a newcomer into the list mid-loop, and they
        # should not have a birthday on the day they arrive.
        for index, agent in enumerate(list(self.agents)):
            if self.clock.day % year != birthday(agent.id):
                continue
            agent.age += 1
            if agent.age >= CONFIG.population.retire_age:
                self._retire(index, agent)

    def _retire(self, index: int, agent: Agent) -> None:
        """Leave the city at retirement age; someone young moves into the home.

        Without it nobody doing well ever left a seat: over 365 days the seven
        management seats never changed hands after day 64, while eight people
        inside cleared their doors and waited. The seat goes first to anyone
        ready for it, like any other opening, and the city stays at fifty.
        """
        job = agent.job
        home = self.world.buildings[agent.home_id]
        where = f" from {self.world.buildings[job.employer_id].name}" if job else ""
        # Cleared rather than left: a model may still be writing this person's
        # interview, paper or review, and every one of those checks these
        # before acting — so a reply for someone who has gone does nothing.
        agent.job = None
        agent.application = None
        agent.enrollment = None
        for queue in (self.pending_interviews, self.pending_exams, self.pending_reviews):
            if agent in queue:
                queue.remove(agent)
        for other in self.agents:
            other.relationships.pop(agent.id, None)
        self.log(f"{agent.name} retired at {agent.age}{where}")
        self.retirements += 1

        arrival = newcomer(self.world, home.id, f"a{self._next_agent:02d}", self._used_names)
        self._next_agent += 1
        self._used_names.add(arrival.name)
        self.agents[index] = arrival
        del self.by_id[agent.id]
        self.by_id[arrival.id] = arrival
        self.roster_version += 1
        # Started the way __init__ starts the first fifty. Without it the
        # opening idle has no end time, is never done, and the newcomer
        # stands in the doorway until every need runs out.
        self._begin(arrival, arrival.action)
        self.log(f"{arrival.name}, {arrival.age}, moved into {home.name}")

        if job is not None:
            self._fill_from_within(job.employer_id, job.role_id)

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
            if agent.enrollment is not None and agent.enrollment.sponsor is None:
                agent.money -= eco.tuition_per_day
            if self.seconded(agent) and agent.job is not None:
                # Full pay while away on the employer's course: the daily rate
                # is what a normal week of shifts averages out to.
                agent.money += agent.job.role.daily
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
