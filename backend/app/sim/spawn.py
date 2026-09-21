"""Populate the city.

Deterministic from CONFIG.seed: the same seed yields the same fifty people, in
the same homes, with the same aptitudes. Phase 7's replay depends on it, and so
does ever reproducing a bug.
"""

from __future__ import annotations

from random import Random
from collections import Counter
from app.agents.actions import Action, ActionKind
from app.agents.agent import SKILLS, Agent
from app.config import CONFIG
from app.institutions.university import Enrollment, choose_course
from app.sim.clock import TICKS_PER_DAY
from app.sim.world import BuildingKind, TileKind, World
from app.agents.actions import Action, ActionKind
from app.agents.agent import SKILLS, Agent
from app.config import CONFIG
from app.institutions.companies import Job, best_vacancy, vacancies
from app.institutions.university import Enrollment, choose_course

FIRST = (
    "Maya", "Ravi", "Nadia", "Omar", "Lena", "Tariq", "Ines", "Kofi", "Sana", "Diego",
    "Yara", "Noor", "Elias", "Priya", "Marcus", "Zainab", "Hugo", "Amara", "Felix", "Rin",
    "Adel", "Bianca", "Samir", "Thea", "Jonas",
)
LAST = (
    "Rahman", "Okafor", "Silva", "Chen", "Novak", "Haddad",
    "Adeyemi", "Kowalski", "Duarte", "Iyer", "Moreau", "Bergman",
)
TRAITS = (
    "curious", "ambitious", "cautious", "gregarious", "stubborn",
    "warm", "analytical", "restless", "patient", "blunt",
)


def _interior_tiles(world: World, building) -> list[tuple[int, int]]:
    return [t for t in building.tiles() if world.tile(*t) is TileKind.FLOOR]


def spawn_agents(world: World, rng: Random) -> list[Agent]:
    homes = sorted(world.of_kind(BuildingKind.HOME), key=lambda b: b.id)
    count = CONFIG.population.agent_count

    # sorted() is load-bearing: Python randomises string hashing per process,
    # so set iteration order differs every run. Without it the seed is a lie.
    names = sorted({f"{first} {last}" for first in FIRST for last in LAST})
    rng.shuffle(names)

    agents: list[Agent] = []
    for i, name in enumerate(names[:count]):
        home = homes[i % len(homes)]
        x, y = rng.choice(_interior_tiles(world, home))

        skills = {s: max(0.0, rng.gauss(20.0, 8.0)) for s in SKILLS}
        # Everyone is notably better at one thing. Uniform agents make a dull
        # job market — every candidate interviews the same. A spike each gives
        # Phase 5 real matching to do.
        spike = rng.choice(SKILLS)
        skills[spike] = min(100.0, skills[spike] + 25.0)

        agents.append(
            Agent(
                id=f"a{i:02d}",
                name=name,
                age=rng.randint(18, 45),
                traits=rng.sample(TRAITS, 2),
                home_id=home.id,
                x=x,
                y=y,
                action=Action(ActionKind.IDLE),
                skills=skills,
                money=max(
                    0.0,
                    rng.gauss(
                        CONFIG.economy.starting_money_mean,
                        CONFIG.economy.starting_money_sd,
                    ),
                ),
            )
        )

    # Students from the head of the shuffled list, employed from the tail, so
    # the two never overlap. Intake is staggered backward by up to half a term:
    # enrolled together, every term would fall due on the same tick forever, and
    # forward offsets would count sessions before a student's term had begun.
    students = round(count * CONFIG.population.initially_students)
    spread = CONFIG.university.term_days * TICKS_PER_DAY // 2
    for i, agent in enumerate(agents[:students]):
        course = choose_course(agent.skills, agent.credentials)
        if course is not None:
            agent.enrollment = Enrollment(
                course_id=course.id,
                started_tick=-(i * spread // max(1, students)),
            )
    # Nobody earns yet: "employed" currently means only "not on the stipend".
    # Jobs go to the tail of the list, and only jobs people can actually do:
    # the best-paid vacancy whose door they clear. Everyone reaches for the top
    # first, so what stays open is the work nobody is qualified for yet — which
    # is what gives a graduate somewhere to go.
    hires = round(count * CONFIG.population.initially_employed)
    taken: Counter[tuple[str, str]] = Counter()
    for agent in agents[count - hires:]:
        posting = best_vacancy(agent.skills, agent.credentials, vacancies(world, taken))
        if posting is None:
            continue  # qualified for nothing; starts out looking
        agent.job = Job(employer_id=posting.employer_id, role_id=posting.role.id)
        taken[posting.key] += 1
    return agents