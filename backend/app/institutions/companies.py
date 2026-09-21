"""The job market: who employs whom, for what, and at what wage.

Postings are fixed data, like the curriculum. An advert has to read the same on
Tuesday as it did on Monday, and a wage is a number — so the simulation owns
both. What the model owns is the interview: whether a candidate who clears the
door can actually do the work.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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

    @property
    def wage(self) -> float:
        """Paid once, on a completed shift."""
        return self.daily * 7.0 / len(self.days)


WEEKDAYS = (0, 1, 2, 3, 4)
#: Utilities, the hospital and the market don't close for Saturday.
SIX_DAYS = (0, 1, 2, 3, 4, 5)

#: The whole labour market. Requirements are set against a spawn spread that
#: peaks near 45 in an agent's best skill: the junior roles are reachable on
#: day one, the 55s and 60s are not reachable without a degree. Those seats sit
#: empty at first on purpose — the city should visibly have work nobody can do.
ROLES: tuple[Role, ...] = (
    Role("junior_eng", "Junior Engineer", BuildingKind.OFFICE, "programming", 30.0, 130.0, WEEKDAYS, 9, seats=2),
    Role("analyst", "Analyst", BuildingKind.OFFICE, "analysis", 45.0, 180.0, WEEKDAYS, 10),
    Role("product_lead", "Product Lead", BuildingKind.OFFICE, "management", 60.0, 250.0, WEEKDAYS, 9),
    Role("ward_clerk", "Ward Clerk", BuildingKind.HOSPITAL, "communication", 25.0, 120.0, SIX_DAYS, 8, seats=2),
    Role("records", "Records Analyst", BuildingKind.HOSPITAL, "analysis", 40.0, 170.0, SIX_DAYS, 11, seats=2),
    Role("ward_manager", "Ward Manager", BuildingKind.HOSPITAL, "management", 55.0, 230.0, SIX_DAYS, 9),
    Role("stall_hand", "Stall Hand", BuildingKind.MARKET, "communication", 15.0, 110.0, SIX_DAYS, 7, seats=3),
    Role("floor_super", "Floor Supervisor", BuildingKind.MARKET, "management", 35.0, 160.0, SIX_DAYS, 10),
    Role("teller", "Teller", BuildingKind.BANK, "communication", 25.0, 125.0, WEEKDAYS, 9, seats=2),
    Role("risk", "Risk Analyst", BuildingKind.BANK, "analysis", 55.0, 240.0, WEEKDAYS, 10),
    Role("power_tech", "Technician", BuildingKind.POWER, "analysis", 40.0, 175.0, SIX_DAYS, 8, seats=2),
    Role("control_eng", "Control Engineer", BuildingKind.POWER, "programming", 55.0, 260.0, SIX_DAYS, 13),
    Role("gas_tech", "Technician", BuildingKind.GAS, "analysis", 35.0, 165.0, WEEKDAYS, 8, seats=2),
    Role("safety_lead", "Safety Lead", BuildingKind.GAS, "management", 50.0, 220.0, WEEKDAYS, 12),
    Role("archivist", "Archivist", BuildingKind.LIBRARY, "communication", 30.0, 130.0, WEEKDAYS, 10, seats=2),
    Role("sys_librarian", "Systems Librarian", BuildingKind.LIBRARY, "design", 35.0, 155.0, WEEKDAYS, 13),
)

BY_ID: dict[str, Role] = {r.id: r for r in ROLES}

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


@dataclass
class Job:
    employer_id: str
    role_id: str
    #: Tick they started. Phase 6 reads it for tenure.
    started_tick: int = 0
    #: Shifts the roster called since they were hired, and how many they turned
    #: up for. Counted at the window's close, so this is "showed up", where the
    #: sim-wide shifts_worked is "finished it and was paid".
    shifts_offered: int = 0
    shifts_attended: int = 0

    @property
    def role(self) -> Role:
        return BY_ID[self.role_id]

    @property
    def attendance(self) -> float:
        """0..1. The share of called shifts actually turned up for."""
        if self.shifts_offered == 0:
            return 1.0  # benefit of the doubt on the first morning
        return self.shifts_attended / self.shifts_offered


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