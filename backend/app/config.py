"""Central tuning knobs for the simulation.

Everything that governs pacing, cost, or scale lives here so the sim can be
retuned without touching engine code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class WorldConfig:
    width: int = 80
    height: int = 60
    #: Simulated minutes advanced per tick.
    minutes_per_tick: int = 10
    #: Sim starts on day 0 at 06:00 so agents wake into a fresh morning.
    start_hour: int = 6
    #: Tiles an agent covers per tick. At 10 min/tick and ~50m tiles this is
    #: roughly walking pace — one tile per tick meant 13 sim-hours to cross
    #: town, so agents spent their whole lives commuting and every need starved.
    tiles_per_tick: int = 12

@dataclass(frozen=True)
class LoopConfig:
    #: Real seconds between ticks at 1x speed. 1 sim day = 144 ticks = 2.4 min.
    seconds_per_tick: float = _env_float("AC_SECONDS_PER_TICK", 1.0)
    #: Speeds selectable from the UI.
    speed_options: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    default_speed: float = 1.0


@dataclass(frozen=True)
class PopulationConfig:
    agent_count: int = _env_int("AC_AGENT_COUNT", 50)
    #: Fraction of the initial population that starts already employed, so the
    #: city has a functioning economy on day 0 instead of 50 unemployed agents.
    initially_employed: float = 0.45
    #: Fraction that starts enrolled at the university.
    initially_students: float = 0.30


@dataclass(frozen=True)
class LLMConfig:
    """Tiered cognition. Tier 0 is free; tiers 1 and 2 cost inference time."""

    base_url: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    #: Tier 1 — fast, high-volume routine decisions (next action, small choices).
    fast_model: str = os.environ.get("AC_FAST_MODEL", "qwen2.5:3b")
    #: Tier 2 — meaningful moments (daily plans, dialogue, interviews, reflection).
    smart_model: str = os.environ.get("AC_SMART_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    #: Optional; falls back to a lexical vector when the model isn't pulled.
    embed_model: str = os.environ.get("AC_EMBED_MODEL", "nomic-embed-text")

    #: Concurrent in-flight requests per tier. Tuned for a single 16GB GPU:
    #: too high and the models thrash VRAM, too low and the queue backs up.
    fast_concurrency: int = _env_int("AC_FAST_CONCURRENCY", 4)
    smart_concurrency: int = _env_int("AC_SMART_CONCURRENCY", 2)

    request_timeout: float = _env_float("AC_LLM_TIMEOUT", 90.0)
    #: Drop a queued request if the sim has moved this far past its submission.
    #: Prevents an agent acting on advice that is an hour of sim-time stale.
    staleness_ticks: int = 18

    enabled: bool = os.environ.get("AC_LLM_ENABLED", "1") != "0"


@dataclass(frozen=True)
class MemoryConfig:
    #: Retrieval weights: score = recency*w_r + importance*w_i + relevance*w_v.
    w_recency: float = 1.0
    w_importance: float = 1.0
    w_relevance: float = 1.2
    #: Memory strength halves every N sim-hours of non-retrieval.
    recency_half_life_hours: float = 12.0
    #: Retrieved memories injected into a prompt.
    retrieval_k: int = 8
    #: Reflect once accumulated importance since the last reflection exceeds this.
    reflection_importance_threshold: float = 45.0
    #: Hard cap per agent; oldest low-importance memories are evicted first.
    max_memories: int = 300


@dataclass(frozen=True)
class NeedsConfig:
    """Needs run 0..100 where 100 is fully satisfied."""

    #: Points lost per sim-hour.
    energy_decay: float = 4.5
    hunger_decay: float = 6.0
    social_decay: float = 3.0
    fun_decay: float = 2.5
    #: Below this a need hijacks the agent's plan (Tier 0 override).
    critical_threshold: float = 18.0


@dataclass(frozen=True)
class EconomyConfig:
    starting_money_mean: float = 800.0
    starting_money_sd: float = 350.0
    daily_rent: float = 45.0
    meal_cost: float = 12.0
    tuition_per_day: float = 25.0
    #: Salary is paid daily as annual/365 for readability in the UI.
    unemployment_stipend: float = 20.0


@dataclass(frozen=True)
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    needs: NeedsConfig = field(default_factory=NeedsConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    seed: int = _env_int("AC_SEED", 20260827)


CONFIG = Config()
