"""The agent: who they are, how they feel, where they stand."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.actions import RESTORES, Action
from app.agents.relationships import Relationship
from app.cognition.memory import MemoryStream
from app.cognition.prompts import PlanStep
from app.config import CONFIG
from app.institutions.companies import Application, Job
from app.institutions.university import Enrollment

SKILLS = ("programming", "analysis", "communication", "design", "management")


@dataclass
class Needs:
    """All needs run 0..100, where 100 is fully satisfied."""

    energy: float = 100.0
    hunger: float = 100.0
    social: float = 100.0
    fun: float = 100.0

    def decay(self, hours: float) -> None:
        n = CONFIG.needs
        self.energy = max(0.0, self.energy - n.energy_decay * hours)
        self.hunger = max(0.0, self.hunger - n.hunger_decay * hours)
        self.social = max(0.0, self.social - n.social_decay * hours)
        self.fun = max(0.0, self.fun - n.fun_decay * hours)

    def restore(self, need: str, amount: float) -> None:
        setattr(self, need, min(100.0, getattr(self, need) + amount))

    def as_dict(self) -> dict[str, float]:
        return {
            "energy": self.energy,
            "hunger": self.hunger,
            "social": self.social,
            "fun": self.fun,
        }

    def lowest(self) -> tuple[str, float]:
        return min(self.as_dict().items(), key=lambda kv: kv[1])

    def critical(self) -> str | None:
        """The need urgent enough to hijack the agent's plan, if any.

        This is the Tier 0 override: free, deterministic, and it runs for every
        agent every tick without touching a model.
        """
        name, value = self.lowest()
        return name if value < CONFIG.needs.critical_threshold else None


@dataclass
class Agent:
    id: str
    name: str
    age: int
    traits: list[str]
    home_id: str
    x: int
    y: int
    action: Action
    needs: Needs = field(default_factory=Needs)
    #: Looked up by name at runtime by courses, job specs and interview rubrics.
    skills: dict[str, float] = field(default_factory=dict)
    money: float = 0.0
    memory: MemoryStream = field(default_factory=MemoryStream)
    #: Remaining steps of today's plan, in time order. Empty means Tier 0.
    plan: list[PlanStep] = field(default_factory=list)
    #: Far in the past so every agent looks overdue on the first tick.
    last_plan_tick: int = -10_000
    relationships: dict[str, Relationship] = field(default_factory=dict)
    #: None means not a student. One course at a time, which is what keeps exam
    #: volume affordable once Phase 4's grading lands.
    enrollment: Enrollment | None = None
    credentials: list[str] = field(default_factory=list)
    #: The seat this agent holds, None while looking. Shifts, wages and the
    #: stipend all key off it.
    job: Job | None = None
    #: Sitting an interview, or on the way to one.
    application: Application | None = None
    #: Employer id -> tick they were turned down there. Stops anyone hammering
    #: the same door every morning.
    rejected_by: dict[str, int] = field(default_factory=dict)
    #: Far in the past so a new arrival may apply at once.
    last_applied_tick: int = -10_000

    @property
    def pos(self) -> tuple[int, int]:
        return self.x, self.y

    @property
    def employed(self) -> bool:
        return self.job is not None

    @property
    def has_routine(self) -> bool:
        """Enrolled, employed, or both — someone the clock makes demands of.
        A job imposes a morning exactly as a timetable does."""
        return self.enrollment is not None or self.job is not None

    def move_to(self, tile: tuple[int, int]) -> None:
        self.x, self.y = tile

    def tick_needs(self, hours: float) -> None:
        """Decay everything, then repair whatever the current action restores.

        Both always run: an agent asleep still gets hungry, which is why they
        wake up rested and starving without anyone writing that rule.
        """
        self.needs.decay(hours)
        entry = RESTORES.get(self.action.kind)
        if entry is not None:
            need, rate = entry
            self.needs.restore(need, rate * hours)

    def __str__(self) -> str:
        return f"{self.name} [{self.action}] {self.pos}"