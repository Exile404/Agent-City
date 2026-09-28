"""The job market: who employs whom, for what, and at what wage.

Postings are fixed data, like the curriculum. An advert has to read the same on
Tuesday as it did on Monday, and a wage is a number — so the simulation owns
both. What the model owns is the interview: whether a candidate who clears the
door can actually do the work.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from random import Random

from app.config import CONFIG
from app.institutions.university import DILIGENCE
from app.institutions.university import BY_ID as COURSES
from app.sim.clock import TICKS_PER_DAY
from app.sim.world import BuildingKind, World


@dataclass(frozen=True)
class Role:
    id: str
    title: str
    #: Employer kind. Every building of this kind carries the posting, so one
    #: role definition opens five seats across five offices.
    employer: BuildingKind
    #: Which of Agent.skills the work draws on, and does the hiring.
    skill: str
    #: Standing needed to be interviewed at all. Below this a candidate is
    #: turned away at reception and never reaches a model.
    requires: float
    #: What the job is worth per sim-day of living, weekends included. The shift
    #: wage derives from it, so a six-day job is not secretly richer than a
    #: five-day one advertising the same money.
    daily: float
    #: Weekdays with a shift, 0 = Monday.
    days: tuple[int, ...]
    #: Shift start, 24h. Staggered across the city so not everyone commutes at once.
    hour: int
    #: Positions per building of this kind.
    seats: int = 1
    #: The role one rung up at the same employer — who reviews this one, and
    #: where a promotion leads. None at the top of the ladder.
    reports_to: str | None = None
    #: Advertised on the job market. False for work nobody applies for: it is
    #: taken when someone has run out of money and out of options.
    open_market: bool = True

    @property
    def wage(self) -> float:
        """Paid once, on a completed shift."""
        return self.daily * 7.0 / len(self.days)


WEEKDAYS = (0, 1, 2, 3, 4)
#: The hospital and the market don't close for Saturday.
SIX_DAYS = (0, 1, 2, 3, 4, 5)

#: The whole labour market. Requirements are set against a spawn spread that
#: peaks near 45 in an agent's best skill: the junior roles are reachable on
#: day one, the 55s and 60s are not reachable without a degree. Those seats sit
#: empty at first on purpose — the city should visibly have work nobody can do.
ROLES: tuple[Role, ...] = (
    Role("junior_eng", "Junior Engineer", BuildingKind.OFFICE, "programming", 30.0, 130.0, WEEKDAYS, 9, seats=2, reports_to="analyst"),
    Role("analyst", "Analyst", BuildingKind.OFFICE, "analysis", 45.0, 180.0, WEEKDAYS, 10, reports_to="product_lead"),
    Role("product_lead", "Product Lead", BuildingKind.OFFICE, "management", 60.0, 250.0, WEEKDAYS, 9),
    Role("ward_clerk", "Ward Clerk", BuildingKind.HOSPITAL, "communication", 25.0, 120.0, SIX_DAYS, 8, seats=2, reports_to="records"),
    Role("records", "Records Analyst", BuildingKind.HOSPITAL, "analysis", 40.0, 170.0, SIX_DAYS, 11, seats=2, reports_to="ward_manager"),
    Role("ward_manager", "Ward Manager", BuildingKind.HOSPITAL, "management", 55.0, 230.0, SIX_DAYS, 9),
    Role("stall_hand", "Stall Hand", BuildingKind.MARKET, "communication", 15.0, 110.0, SIX_DAYS, 7, seats=3, reports_to="floor_super"),
    Role("floor_super", "Floor Supervisor", BuildingKind.MARKET, "management", 35.0, 160.0, SIX_DAYS, 10),
    Role("teller", "Teller", BuildingKind.BANK, "communication", 25.0, 125.0, WEEKDAYS, 9, seats=2, reports_to="risk"),
    Role("risk", "Risk Analyst", BuildingKind.BANK, "analysis", 55.0, 240.0, WEEKDAYS, 10),
    # Weekdays, not six: the longest commute in the city on a six-day week ran
    # Kestrel at 45% attendance with 17 let go in 120 days, and nobody there
    # lasted to a first review — so its ladder never promoted anyone.
    Role("power_tech", "Technician", BuildingKind.POWER, "analysis", 40.0, 175.0, WEEKDAYS, 8, seats=2, reports_to="control_eng"),
    Role("control_eng", "Control Engineer", BuildingKind.POWER, "programming", 55.0, 260.0, WEEKDAYS, 13),
    Role("gas_tech", "Technician", BuildingKind.GAS, "analysis", 35.0, 165.0, WEEKDAYS, 8, seats=2, reports_to="safety_lead"),
    Role("safety_lead", "Safety Lead", BuildingKind.GAS, "management", 50.0, 220.0, WEEKDAYS, 12),
    Role("archivist", "Archivist", BuildingKind.LIBRARY, "communication", 30.0, 130.0, WEEKDAYS, 10, seats=2, reports_to="sys_librarian"),
    Role("sys_librarian", "Systems Librarian", BuildingKind.LIBRARY, "design", 35.0, 155.0, WEEKDAYS, 13),
    # The floor under the market: no interview, no ladder, and $95 a day
    # against $81 of rent and meals. Never advertised; taken by someone out of
    # money who could work but has nothing yet. The 20 is only how hard the
    # work is, since nobody is held to it at a door. At ten, not seven: at seven
    # half of all porter shifts were slept through, and the floor fired the
    # people it was there to catch.
    Role("porter", "Porter", BuildingKind.MARKET, "communication", 20.0, 95.0, SIX_DAYS, 10, seats=50, open_market=False),
)

BY_ID: dict[str, Role] = {r.id: r for r in ROLES}
PORTER = "porter"


def ladder_above(role: Role) -> list[Role]:
    """Every rung above this one at the same employer, nearest first."""
    out: list[Role] = []
    while role.reports_to is not None:
        role = BY_ID[role.reports_to]
        out.append(role)
    return out


def _check_ladders() -> None:
    """Fail at import, not mid-run. A ladder that points at another employer
    or loops back on itself would otherwise hang the tick loop the first time
    anybody asked who their manager was."""
    for role in ROLES:
        seen = {role.id}
        above = role.reports_to
        while above is not None:
            boss = BY_ID[above]  # a typo'd id raises KeyError here, which is the point
            if boss.employer is not role.employer or boss.id in seen:
                raise ValueError(f"broken ladder at {role.id} -> {above}")
            seen.add(boss.id)
            above = boss.reports_to


_check_ladders()

#: Roles somebody reports to: the only seats that can have a team to run.
MANAGING_ROLES = frozenset(r.reports_to for r in ROLES if r.reports_to)

#: What a passed course is worth at the door, in skill points — on top of what
#: the term actually taught. A credential is somebody else's word for you, and
#: it is the difference between qualifying and not for the mid-tier roles.
CREDENTIAL_CREDIT = 10.0

#: How far either side of the start an agent may still set off for a shift.
#: Wider than the university's window: no journey gets turned around for work,
#: so the window itself has to outlast the walk.
SHIFT_EARLY_MINUTES = 60
SHIFT_LATE_MINUTES = 60

#: Interview mark needed to be offered the seat. Clearing the door gets a
#: candidate into the room; this is about how far past it they are, so a thin
#: margin is not enough. Applying for the best-paid job you barely qualify for
#: is a gamble — which is what sends people back to the university.
HIRE_MARK = 60.0

#: How reliably someone sets off for a paid shift, before any need gets in the
#: way. A job is not a lecture: the floor is far higher than the university's
#: and temperament moves it less, so the traits that lose you a course only
#: dent an attendance record. The same DILIGENCE table drives both, which is
#: what makes a restless agent recognisably the same person in a lecture hall
#: and at a power station.
BASE_RELIABILITY = 0.95
TRAIT_WEIGHT = 0.30
RELIABILITY_FLOOR = 0.75

#: Shifts before absence can cost someone their job — a bad fortnight, not a
#: bad morning. Ten shifts is two working weeks.
FIRING_GRACE_SHIFTS = 10
#: Attendance below which an employer lets someone go. A guess: city-wide
#: attendance measures 75%, but nobody has looked at the spread around it, so
#: read this off the distribution before defending it.
FIRING_ATTENDANCE = 0.6

#: Sim-days before trying the same employer again.
REJECTION_COOLDOWN_DAYS = 3
#: Ticks between applications. Job-hunting is not a full-time occupation, and
#: every application costs a generation once the model is doing the interview.
APPLY_COOLDOWN_TICKS = TICKS_PER_DAY
#: Warnings in one seat that end it. Measured before this existed: over sixty
#: days nobody collected two — six of the eight warned were let go for absence
#: first. What it catches is the one case absence cannot: someone who turns up
#: every day and works below what the job needs.
WARNINGS_TO_FIRE = 2
#: Reviews a warning stays live for — about six weeks. Without it a warning
#: never expired: both dismissals in a 120-day run paired a fresh warning with
#: one 58 and 70 days old, four clean reviews earlier.
WARNING_WINDOW = 3
#: Reviews at the start of a seat whose warnings stand as advice, not strikes.
#: Promoted below the bar on credential credit, three of four new Analysts were
#: warned at their first review; learning on the job (about 2.4 skill points a
#: review period) had the survivors back at the bar by the second.
PROBATION_REVIEWS = 1

@dataclass
class Job:
    employer_id: str
    role_id: str
    #: Tick they started: tenure, and the seed of their task rota.
    started_tick: int = 0
    #: Shifts the roster called since they were hired, and how many they turned
    #: up for. Counted at the window's close, so this is "showed up", where the
    #: sim-wide shifts_worked is "finished it and was paid".
    shifts_offered: int = 0
    shifts_attended: int = 0
    #: The latest tasks as (title, quality, tired), oldest first, capped at
    #: TASK_RECORD. Everything a review reads about the work comes from here.
    tasks: list[tuple[str, float, bool]] = field(default_factory=list)
    #: Tasks done in this seat, uncapped. Drives the rota in do_task.
    tasks_done: int = 0
    #: Roster counters at the last review, so the next one judges the stretch
    #: since rather than the whole tenure. Zero at hire: the first covers it all.
    reviewed_offered: int = 0
    reviewed_attended: int = 0
    #: Reviews held in this seat, oldest first.
    reviews: list[Review] = field(default_factory=list)
    #: Tick a review fell due, -1 when none is waiting. While a model is writing
    #: it the grace timer stands down, exactly as with interviews.
    review_due_tick: int = -1
    review_in_flight: bool = False

    @property
    def role(self) -> Role:
        return BY_ID[self.role_id]

    @property
    def attendance(self) -> float:
        """0..1. The share of called shifts actually turned up for."""
        if self.shifts_offered == 0:
            return 1.0  # benefit of the doubt on the first morning
        return self.shifts_attended / self.shifts_offered

    @property
    def form(self) -> float | None:
        """Mean quality of the recent tasks. None before the first one — no
        record is not the same as a bad record."""
        if not self.tasks:
            return None
        return sum(q for _, q, _ in self.tasks) / len(self.tasks)

    @property
    def window_attendance(self) -> float:
        """Attendance since the last review. 1.0 before any shift is called."""
        offered = self.shifts_offered - self.reviewed_offered
        if offered == 0:
            return 1.0
        return (self.shifts_attended - self.reviewed_attended) / offered

    def record(self, title: str, quality: float, tired: bool) -> None:
        self.tasks.append((title, quality, tired))
        del self.tasks[:-TASK_RECORD]
        self.tasks_done += 1


@dataclass(frozen=True)
class Posting:
    employer_id: str
    employer_name: str
    role: Role

    @property
    def key(self) -> tuple[str, str]:
        return self.employer_id, self.role.id


@dataclass
class Application:
    posting: Posting
    #: Tick the candidate sat down. The grace period runs from here.
    filed_tick: int = -1
    #: A model is writing this interview right now. The grace timer stands down
    #: while it is on, so the two deadlines never race.
    in_flight: bool = False


def postings(world: World) -> list[Posting]:
    """Every advertised seat in the city, in a stable order."""
    out: list[Posting] = []
    for role in ROLES:
        if not role.open_market:
            continue
        for building in sorted(world.of_kind(role.employer), key=lambda b: b.id):
            out.append(Posting(building.id, building.name, role))
    return out


def headcount(jobs) -> Counter[tuple[str, str]]:
    """Seats currently filled, keyed by (employer, role).

    Derived from the agents holding them rather than stored, so there is one
    source of truth about who works where.
    """
    return Counter((j.employer_id, j.role_id) for j in jobs if j is not None)


def vacancies(world: World, taken: Counter[tuple[str, str]]) -> list[Posting]:
    return [p for p in postings(world) if taken[p.key] < p.role.seats]


def standing(skills: dict[str, float], credentials: list[str], role: Role) -> float:
    """The candidate's ability as the employer reads it."""
    certified = any(c in COURSES and COURSES[c].skill == role.skill for c in credentials)
    return skills.get(role.skill, 0.0) + (CREDENTIAL_CREDIT if certified else 0.0)


def clears_door(skills: dict[str, float], credentials: list[str], role: Role) -> bool:
    """Whether this candidate is worth interviewing at all.

    The simulation decides who gets into the room; the model decides what
    happens in it. Keeping the threshold here is what makes rejection reliable
    — a model asked to judge everyone would let weak candidates through on a
    good roll, and nobody would ever need to go back and study.
    """
    return standing(skills, credentials, role) >= role.requires


def best_vacancy(
    skills: dict[str, float], credentials: list[str], open_postings: list[Posting]
) -> Posting | None:
    """The best job this candidate could hold: the best-paid open seat whose
    door they clear, ties broken by how comfortably they clear it."""
    eligible = [p for p in open_postings if clears_door(skills, credentials, p.role)]
    if not eligible:
        return None
    eligible.sort(
        key=lambda p: (
            -p.role.daily,
            -(standing(skills, credentials, p.role) - p.role.requires),
            p.employer_id,
            p.role.id,
        )
    )
    return eligible[0]


def interview_score(skills: dict[str, float], credentials: list[str], role: Role) -> float:
    """The mark a candidate has earned before anyone asks a question.

    Fifty at the bar, two points per point of margin. Both the verdict when no
    model is available and the anchor the model is handed, so the simulation
    owns the number either way.
    """
    margin = standing(skills, credentials, role) - role.requires
    return max(0.0, min(100.0, 50.0 + margin * 2.0))


def reliability(traits: list[str]) -> float:
    """How reliably this agent turns up for work, 0.75..1.0."""
    moved = BASE_RELIABILITY + sum(DILIGENCE.get(t, 0.0) for t in traits) * TRAIT_WEIGHT
    return max(RELIABILITY_FLOOR, min(1.0, moved))


def shows_up(agent_id: str, employer_id: str, day: int, traits: list[str]) -> bool:
    """Whether this agent sets off for today's shift.

    Seeded on (agent, employer, day) so the same run reproduces the same week,
    and keyed on the day so the answer holds across the whole join window.
    Seeded with the string, never hash(string): Python randomises string
    hashing per process.
    """
    return Random(f"{agent_id}:{employer_id}:{day}").random() < reliability(traits)


def shift_now(role: Role, weekday: int, minute_of_day: int) -> bool:
    """Whether this role's shift is running and still joinable."""
    if weekday not in role.days:
        return False
    start = role.hour * 60
    return start - SHIFT_EARLY_MINUTES <= minute_of_day <= start + SHIFT_LATE_MINUTES


def shift_starts_now(role: Role, weekday: int, minute_of_day: int) -> bool:
    """Whether the shift begins inside the current tick. A half-open window one
    tick wide, so it fires exactly once per shift."""
    if weekday not in role.days:
        return False
    return 0 <= minute_of_day - role.hour * 60 < CONFIG.world.minutes_per_tick


def shift_closes_now(role: Role, weekday: int, minute_of_day: int) -> bool:
    """Whether the joining window shuts inside this tick — the moment a missed
    shift becomes a missed shift."""
    if weekday not in role.days:
        return False
    shut = role.hour * 60 + SHIFT_LATE_MINUTES
    return 0 <= minute_of_day - shut < CONFIG.world.minutes_per_tick


# ------------------------------------------------------------ what a shift does

#: Difficulty of the easy, routine and hard task, in skill points above the
#: job's own bar. Scored on the interview's scale — fifty at the difficulty,
#: two points per point of margin — so someone who only just cleared the
#: interview turns in routine work around fifty, and the hard task stretches.
STRETCH = (-5.0, 5.0, 15.0)
#: A bad day, or a good one. Spread of the roll, in quality points.
QUALITY_NOISE = 10.0
#: Energy below which the work suffers, read as the shift ends. Set under what
#: an ordinary day should leave anyone with — a 13:00 shift after a normal
#: night ends near fifty — so it catches someone who came in worn out, not
#: someone whose shift simply runs late.
FATIGUE_LINE = 40.0
#: Quality points lost per point of energy below the line. Coming in empty
#: costs forty: working like someone twenty skill points less able.
FATIGUE_COST = 1.0
#: How far past the job's own bar work can carry a skill. Every ladder in the
#: city crosses skills, so this is not what keeps work from qualifying anyone
#: for the rung above — the change of skill does that. It stops a long tenure
#: inflating a skill without limit: work teaches the job, up to mastery of it.
MASTERY_MARGIN = 15.0
#: Tasks kept per job. About one review's worth: form is recent form, and a
#: record that grew with tenure would let one good month carry someone forever.
TASK_RECORD = 10
#: Tasks good or bad enough to remember and mention in the feed. Everything in
#: between is an ordinary day and tells nobody anything.
FINE_WORK = 85.0
POOR_WORK = 30.0

#: Every role's work at the three STRETCH levels. Fixed data, like the
#: curriculum, so a review has something concrete to cite and a model never
#: has to invent what someone did. Noun phrases, not instructions: a 3B pastes
#: titles straight into its sentences, and an imperative came out as "your
#: calm a distressed family task".
TASKS: dict[str, tuple[str, str, str]] = {
    "junior_eng": ("Fixing a failing test", "Reviewing a pull request", "Shipping a new feature"),
    "analyst": ("Cleaning a quarter's data", "Building the weekly report", "Forecasting next quarter's demand"),
    "product_lead": ("Running the stand-up", "Planning the next release", "Settling a scope dispute"),
    "ward_clerk": ("Booking patient appointments", "Handling the front-desk rush", "Calming a distressed family"),
    "records": ("Filing discharge summaries", "Auditing a ward's records", "Tracing a billing error"),
    "ward_manager": ("Drawing up the nurse rota", "Covering a short-staffed shift", "Leading an incident review"),
    "stall_hand": ("Stocking the stalls", "Working the lunchtime crowd", "Talking down an angry customer"),
    "floor_super": ("Opening the market floor", "Settling a vendor dispute", "Reorganising the floor layout"),
    "teller": ("Processing the morning deposits", "Opening a new account", "Handling a fraud complaint"),
    "risk": ("Scoring a loan application", "Stress-testing the portfolio", "Modelling a default scenario"),
    "power_tech": ("Logging turbine readings", "Diagnosing a pressure drop", "Tracing an intermittent fault"),
    "control_eng": ("Patching the control software", "Tuning a feedback loop", "Rewriting a failing controller"),
    "gas_tech": ("Checking meter readings", "Inspecting a pipeline section", "Finding a pressure leak"),
    "safety_lead": ("Running the safety briefing", "Reviewing an incident report", "Leading an emergency drill"),
    "archivist": ("Cataloguing new arrivals", "Helping a researcher find a source", "Restoring a damaged collection"),
    "sys_librarian": ("Fixing the catalogue search", "Redesigning the lending workflow", "Planning the digital archive"),
    "porter": ("Moving stock off the carts", "Clearing the loading bay", "Unloading a late delivery"),
}

if set(TASKS) != set(BY_ID) or any(len(t) != len(STRETCH) for t in TASKS.values()):
    raise ValueError("every role needs exactly one task per STRETCH level")


def do_task(
    agent_id: str, job: Job, day: int, skill: float, energy: float
) -> tuple[str, float, bool]:
    """Today's task for this seat, how well it went, and whether they were
    worn out doing it.

    The simulation owns all three. Which task comes up is the employer's rota:
    a shuffle bag, so every three tasks hold one easy, one routine and one hard
    in a seeded order. Drawn independently each day, one technician got the
    hard task seven times in ten and was warned three reviews running for
    nothing but the draw. How it goes is skill against difficulty, less what
    exhaustion costs, plus the day's luck — seeded on (agent, employer, day)
    like shows_up, with its own suffix so it never shares a roll with
    attendance.
    """
    block, slot = divmod(job.tasks_done, len(STRETCH))
    rota = Random(f"{agent_id}:{job.employer_id}:{job.started_tick}:{block}:rota")
    level = rota.sample(range(len(STRETCH)), len(STRETCH))[slot]
    difficulty = job.role.requires + STRETCH[level]
    tired = energy < FATIGUE_LINE
    roll = Random(f"{agent_id}:{job.employer_id}:{day}:task").gauss(0.0, QUALITY_NOISE)
    quality = (
        50.0
        + (skill - difficulty) * 2.0
        - max(0.0, FATIGUE_LINE - energy) * FATIGUE_COST
        + roll
    )
    return TASKS[job.role_id][level], max(0.0, min(100.0, quality)), tired


def _capped_gain(skill_now: float, ceiling: float) -> float:
    """One shift's diminishing-returns gain, never carrying skill past ceiling."""
    room = ceiling - skill_now
    if room <= 0:
        return 0.0
    return min(room, CONFIG.work.shift_gain * (1.0 - skill_now / 100.0))


def work_gain(skill_now: float, role: Role) -> float:
    """Skill points one finished shift teaches. The university's diminishing
    returns at a tenth the size, stopping at mastery of this job."""
    return _capped_gain(skill_now, role.requires + MASTERY_MARGIN)


# ----------------------------------------------------------------- reviews

#: Shifts the roster calls between reviews — two working weeks. Counted in
#: shifts called, not attended, so someone who never turns up is still
#: reviewed on time; and equal to TASK_RECORD, so a review reads the stretch
#: since the last one.
REVIEW_EVERY = 10
#: The verdict the numbers alone give: the fallback, and the yardstick a
#: model's verdict is measured against. Both form lines come off the task
#: scale. The average task sits five points above the job's bar, so expected
#: form is 40 for someone exactly at the bar, 50 just past the interview, 70
#: at mastery, 80 at the bar plus twenty.
#:
#: Promote: past what the job itself can teach.
PROMOTE_FORM = 75.0
PROMOTE_ATTENDANCE = 0.8
#: Warn: turning in less than someone exactly at the bar would. At 45 a
#: worker who honestly cleared the interview sat 1.6 sigma above the line and
#: drew a warning from bad rolls alone about one review in sixteen; at 40 it
#: is three sigma, about one in a thousand.
WARN_FORM = 40.0
#: The employer's line on absence is the same one it fires at.
WARN_ATTENDANCE = FIRING_ATTENDANCE


@dataclass(frozen=True)
class Review:
    tick: int
    #: A name, or "the management at X" when nobody holds a rung above.
    reviewer: str
    verdict: str
    comment: str
    form: float | None
    attendance: float
    #: What the numbers alone said. Equal to verdict unless a model decided.
    expected: str


def expected_verdict(form: float | None, attendance: float, top: bool) -> str:
    """The review the numbers alone would write. Nobody at the top of a
    ladder is promoted: there is no rung to promote them to."""
    if attendance < WARN_ATTENDANCE or (form is not None and form < WARN_FORM):
        return "warn"
    if not top and form is not None and form >= PROMOTE_FORM and attendance >= PROMOTE_ATTENDANCE:
        return "promote"
    return "keep"


def allowed_verdicts(form: float | None, attendance: float, top: bool) -> tuple[str, ...]:
    """What a review may say: exactly what the numbers say.

    Given a free choice, a 3B promoted five people on one good task and warned
    three on one bad one. Measured twice with promptbench: allowed only to
    decline a promotion, it declined 40-56% of the time, under comments
    praising the work, and gave identical records opposite verdicts. A
    judgement that cannot give its reason is noise, and it ran live careers at
    half the headless pace. The record sets the verdict; the model writes the
    words.
    """
    return (expected_verdict(form, attendance, top),)


def task_level(role_id: str, title: str) -> int:
    """Which STRETCH level a recorded task was: 0 easy, 1 routine, 2 hard."""
    return TASKS[role_id].index(title)


def against_difficulty(quality: float, level: int) -> float:
    """A task's quality with its difficulty taken back out, so an easy task
    and a hard one read on one scale: 50 is what someone exactly at the job's
    bar turns in on any task, 60 someone just past the interview, 80 mastery.
    What a reviewer is told; form keeps the raw output."""
    return quality + 2.0 * STRETCH[level]


def graded_tasks(job: Job) -> list[tuple[str, int, float, bool]]:
    """The recent tasks as a reviewer reads them: (title, level, quality
    against difficulty, tired)."""
    out = []
    for title, quality, tired in job.tasks:
        level = task_level(job.role_id, title)
        out.append((title, level, against_difficulty(quality, level), tired))
    return out

def live_warnings(job: Job) -> int:
    """Warnings still counting against this seat: those in the last
    WARNING_WINDOW reviews, after probation."""
    return sum(r.verdict == "warn" for r in job.reviews[PROBATION_REVIEWS:][-WARNING_WINDOW:])


#: How far running a team can carry management: the city's highest management
#: bar, Product Lead's 60. Seven of nine senior seats ask for management, the
#: only course in it (Project Management, difficulty 0.6) tops out near 42 raw
#: after a full term, and nothing else taught it — so nobody rose into those
#: seats from inside in 120 days.
LEADERSHIP_CEILING = 60.0


def leadership_gain(skill_now: float) -> float:
    """Management one shift of running a team teaches: work_gain's rate,
    stopping at LEADERSHIP_CEILING."""
    return _capped_gain(skill_now, LEADERSHIP_CEILING)
