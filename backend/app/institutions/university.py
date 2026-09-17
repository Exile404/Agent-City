"""The university: what is taught, when, and to whom.

Courses are fixed data rather than generated. A timetable has to be the same on
Tuesday as it was on Monday, and Phase 7's replay needs the curriculum to be a
constant rather than a model output.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from app.config import CONFIG
from app.sim.clock import TICKS_PER_DAY


@dataclass(frozen=True)
class Course:
    id: str
    name: str
    #: Building id of the campus that teaches it.
    campus_id: str
    #: Which of Agent.skills it grows.
    skill: str
    #: 0..1. Scales both how fast the skill grows and how hard the exam is.
    difficulty: float
    #: Weekdays with a session, 0 = Monday. Gaps give the campus a rhythm.
    days: tuple[int, ...]
    #: Session start, 24h.
    hour: int


#: Two campuses with different characters, so a credential says something —
#: Phase 5 can tell an Algorithms pass from an Applied Programming one.
CURRICULUM: tuple[Course, ...] = (
    Course("algo", "Algorithms", "university", "programming", 0.9, (0, 1, 2, 4), 10),
    Course("stats", "Statistical Methods", "university", "analysis", 0.85, (0, 1, 3, 4), 13),
    Course("sysdes", "Systems Design", "university", "design", 0.7, (1, 2, 3, 4), 15),
    Course("appprog", "Applied Programming", "eastgate", "programming", 0.5, (0, 1, 2, 3), 9),
    Course("techwrite", "Technical Writing", "eastgate", "communication", 0.4, (0, 2, 3, 4), 11),
    Course("projman", "Project Management", "eastgate", "management", 0.6, (0, 1, 3, 4), 14),
)

BY_ID: dict[str, Course] = {c.id: c for c in CURRICULUM}

#: How far either side of the bell an agent may still set off for class. A
#: window rather than an instant, because _choose_action only runs when the
#: current action ends.
JOIN_EARLY_MINUTES = 30
JOIN_LATE_MINUTES = 30

#: Hours during which an enrolled agent turns in. Sleep runs eight hours, so a
#: window closing at midnight has everyone up ahead of the earliest 09:00 class.
BEDTIME_FROM_HOUR = 21
BEDTIME_TO_HOUR = 24
#: Energy below which a student goes to bed once the window is open, even if
#: some other need is technically lower. Without this the routine never binds.
BEDTIME_ENERGY = 75.0
#: Energy above which a student refuses to nap during the day.
DAYTIME_NAP_FLOOR = 30.0

#: How much each trait moves the odds of turning up. Not a dimension the agents
#: were given — it falls out of the traits they have, so a failed course traces
#: back to who someone is.
DILIGENCE: dict[str, float] = {
    "ambitious": 0.25,
    "patient": 0.20,
    "analytical": 0.15,
    "curious": 0.15,
    "cautious": 0.10,
    "warm": 0.0,
    "gregarious": -0.10,
    "blunt": -0.10,
    "stubborn": -0.20,
    "restless": -0.25,
}
#: Everyone means to go. Two traits move this by at most ±0.45, giving a spread
#: of roughly 0.30 to 1.00 and a mean near 0.75.
BASE_DILIGENCE = 0.75

#: Mark needed to pass. Read off the measured distribution, from a flat stretch
#: of it: 36 puts about half of first attempts through and three-quarters
#: within the two attempts the design allows.
PASS_MARK = 36.0


@dataclass
class Enrollment:
    course_id: str
    #: Tick the term started. The exam falls term_days later.
    started_tick: int
    sessions_attended: int = 0
    #: Sessions the timetable offered since enrolling, attended or not.
    sessions_offered: int = 0
    #: Kept across retakes, so a transcript can say "passed on the second go".
    attempt: int = 1
    #: Set when the term ends, cleared when a result lands.
    awaiting_exam: bool = False
    #: Tick the term ran out. The grace period is measured from here.
    exam_due_tick: int = -1
    #: A model is writing this paper right now. Set by the Hub; the grace timer
    #: stands down while it is on, so the two deadlines never race.
    in_flight: bool = False

    @property
    def course(self) -> Course:
        return BY_ID[self.course_id]

    @property
    def attendance(self) -> float:
        """0..1. The share of offered sessions actually sat through."""
        if self.sessions_offered == 0:
            return 0.0
        return self.sessions_attended / self.sessions_offered

    def term_over(self, tick: int) -> bool:
        return tick - self.started_tick >= CONFIG.university.term_days * TICKS_PER_DAY


def session_now(course: Course, weekday: int, minute_of_day: int) -> bool:
    """Whether this course is in session and still joinable."""
    if weekday not in course.days:
        return False
    start = course.hour * 60
    return start - JOIN_EARLY_MINUTES <= minute_of_day <= start + JOIN_LATE_MINUTES


def session_starts_now(course: Course, weekday: int, minute_of_day: int) -> bool:
    """Whether the bell falls inside the current tick. A half-open window one
    tick wide, so it fires exactly once per session."""
    if weekday not in course.days:
        return False
    return 0 <= minute_of_day - course.hour * 60 < CONFIG.world.minutes_per_tick


def is_bedtime(hour: int) -> bool:
    return BEDTIME_FROM_HOUR <= hour < BEDTIME_TO_HOUR


def study_gain(skill_now: float, difficulty: float) -> float:
    """Skill points from one session. Diminishing returns, so a novice gains
    fast and an expert grinds; harder courses teach faster and examine harder."""
    return CONFIG.university.session_gain * difficulty * (1.0 - skill_now / 100.0)


def diligence(traits: list[str]) -> float:
    """How reliably this agent turns up, 0.25..1.0."""
    return max(0.25, min(1.0, BASE_DILIGENCE + sum(DILIGENCE.get(t, 0.0) for t in traits)))


def attends(agent_id: str, course_id: str, day: int, traits: list[str]) -> bool:
    """Whether this agent goes to this particular session.

    Seeded on (agent, course, day) so the same run reproduces the same term,
    and keyed on the day so the answer holds across the whole join window.
    Seeded with the string, never hash(string): Python randomises string
    hashing per process.
    """
    return Random(f"{agent_id}:{course_id}:{day}").random() < diligence(traits)


def baseline_score(skill: float, attendance: float, difficulty: float) -> float:
    """The mark a student has earned before anyone writes a word.

    Sixty/forty skill to attendance, marked down by difficulty — 0.5 is
    neutral. Both the grade when the model is unavailable and the anchor the
    model is asked to stay near, so the simulation owns the number either way.
    """
    raw = 0.6 * skill + 0.4 * attendance * 100.0
    return max(0.0, min(100.0, raw - (difficulty - 0.5) * 20.0))


def choose_course(
    skills: dict[str, float], credentials: list[str], avoid: str | None = None
) -> Course | None:
    """The next course worth enrolling in.

    Breadth first: a skill with no credential outranks one already certified,
    otherwise everyone piles into the hardest course of their best subject.
    Within the preferred set they pick by aptitude, and the stronger student
    takes the harder version.

    Takes primitives rather than an Agent because agent.py imports this module.
    """
    passed = set(credentials)
    covered = {BY_ID[c].skill for c in credentials if c in BY_ID}

    def pick(pool: list[Course]) -> Course | None:
        if not pool:
            return None
        pool.sort(key=lambda c: (-skills.get(c.skill, 0.0), -c.difficulty))
        top = pool[0].skill
        same = [c for c in pool if c.skill == top]
        return same[0] if skills.get(top, 0.0) >= 35.0 else same[-1]

    available = [c for c in CURRICULUM if c.id not in passed and c.id != avoid]
    return pick([c for c in available if c.skill not in covered]) or pick(available)
