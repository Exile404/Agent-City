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
    width: int = 70
    height: int = 50
    #: Simulated minutes advanced per tick. Halved from 10 purely for visual
    #: resolution: the renderer lerps in a straight line between ticks, so a
    #: path that turned mid-tick got flattened into a diagonal slide.
    minutes_per_tick: int = 5
    #: Sim starts on day 0 at 06:00 so agents wake into a fresh morning.
    start_hour: int = 6
    #: Tiles an agent covers per tick. Deliberately below walking pace: 12 was
    #: physically right but crossed the city in four seconds, which read as
    #: teleporting. The map was shrunk to keep the needs budget balanced.
    tiles_per_tick: int = 2


@dataclass(frozen=True)
class LoopConfig:
    #: Real seconds between ticks at 1x speed. 1 sim day = 288 ticks = 2.4 min.
    #: Halved alongside minutes_per_tick and tiles_per_tick, so sim-minutes per
    #: real second and tiles per sim-minute are both unchanged — the needs
    #: budget is identical, the motion is simply sampled twice as finely.
    seconds_per_tick: float = _env_float("AC_SECONDS_PER_TICK", 0.5)
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
    #: The same model as Tier 1, deliberately. A 7b-q4 alongside it meant three
    #: resident models costing 3.5GB of *host* RAM — Ollama runs llama-server with
    #: --no-mmap, so weights are copied into system memory even when they sit 100%
    #: on the GPU. On a 15GiB machine that was enough for the kernel OOM killer to
    #: take the desktop session down. The lane split stays because it still buys
    #: separate admission control; it is no longer a split between two models.
    smart_model: str = os.environ.get("AC_SMART_MODEL", "qwen2.5:3b")
    #: Optional; falls back to a lexical vector when the model isn't pulled.
    embed_model: str = os.environ.get("AC_EMBED_MODEL", "nomic-embed-text")

    #: Concurrent in-flight requests per tier. These are only real if Ollama has
    #: at least this many slots: it launches llama-server with `-np 1` by default,
    #: one request at a time per model, so surplus requests queue *inside* Ollama
    #: and expire against staleness_ticks having never run — measured at 15 of 60
    #: plan requests dropped unserved. OLLAMA_NUM_PARALLEL=2 fixes that (drops went
    #: to zero), and slots are not free: at 4 slots llama-server held 4.5GB of host
    #: RAM for a model that occupies 2.4GB of VRAM, because each slot carries its
    #: own KV cache and --no-mmap copies the weights into system memory as well.
    #: Kept just above the slot count — a bounded overshoot, not the 4-against-1
    #: that was expiring requests. With both lanes on one model the split is now a
    #: policy reservation, holding a slot for dialogue so planning cannot starve
    #: it, rather than a boundary between two models.
    fast_concurrency: int = _env_int("AC_FAST_CONCURRENCY", 2)
    smart_concurrency: int = _env_int("AC_SMART_CONCURRENCY", 1)

    request_timeout: float = _env_float("AC_LLM_TIMEOUT", 90.0)
    #: Drop a queued request if the sim has moved this far past its submission.
    #: Prevents an agent acting on advice that is an hour of sim-time stale.
    staleness_ticks: int = 36

    enabled: bool = os.environ.get("AC_LLM_ENABLED", "1") != "0"


@dataclass(frozen=True)
class MemoryConfig:
    #: Retrieval weights: score = recency*w_r + importance*w_i + relevance*w_v,
    #: each min-max normalised across the candidates first. Relevance is
    #: weighted up because these are targeted queries ("I am hungry") rather
    #: than the broad reflection queries the weights were originally tuned for —
    #: at parity the newest memory wins every question regardless of topic.
    w_recency: float = 1.0
    w_importance: float = 1.0
    w_relevance: float = 1.8
    #: Memory strength halves every N sim-hours of non-retrieval.
    recency_half_life_hours: float = 12.0
    #: Retrieved memories injected into a prompt.
    retrieval_k: int = 8
    #: Reflect once accumulated importance since the last reflection exceeds this.
    reflection_importance_threshold: float = 45.0
    #: Hard cap per agent; oldest low-importance memories are evicted first.
    max_memories: int = 300
    relevance_floor: float = 0.85


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
class CognitionConfig:
    #: Thoughts admitted per tick. Measured: the fast lane runs 2 concurrent at
    #: 2.46s each = 0.81 calls/sec, and a tick is 0.5s — so ~0.41/tick is what the
    #: GPU actually drains. At 2 the queue sat permanently full, which meant
    #: priority only applied at the moment a slot freed rather than across a
    #: real candidate set.
    admissions_per_tick: int = 1
    #: Hard cap on dispatched-but-unfinished thoughts. Queue deeper than this
    #: and answers start arriving after their staleness window has closed,
    #: which burns GPU time to produce nothing.
    max_outstanding: int = 8
    #: Re-plan at least this often in sim-minutes, even mid-plan. Measured:
    #: re-planning every 4 hours needs 2.08 plans/sec across 50 agents, and the
    #: GPU delivers ~0.7 — demand was 2.6x supply. Twice a day (morning and
    #: evening) needs 0.69/sec, which matches almost exactly.
    replan_minutes: int = 720
    #: Conversations generating at once. Measured, this cap *is* the budget — a
    #: whole exchange takes 1.59s on the 3b, and with smart_concurrency at 1 the
    #: second slot waits on the first, so the pair drains 0.27 conversations per
    #: tick. No separate rate limit is needed; the cap is the rate limit.
    max_chats: int = 2
    #: Interest below which a pair gets only the Tier 0 greeting. Measured 1.5
    #: encounters per tick against a 0.27 budget, so about one in five can afford
    #: words — 3.1 real conversations per agent per sim-day.
    #: 2.0 leaves about 0.8 candidates per tick, deliberately well above the
    #: budget: refusing a candidate costs nothing (it still gets the greeting),
    #: while having fewer candidates than slots wastes the GPU outright. At 4.0
    #: that is exactly what happened — the gate was above the entire steady-state
    #: score range, so from day three the smart lane ran at 15% of capacity.
    chat_min_interest: float = 2.0


@dataclass(frozen=True)
class UniversityConfig:
    #: Skill points a session grants at zero skill and difficulty 1.0. Two terms
    #: are worth ~30 against a spawn spread of 5-60, so graduating changes where
    #: someone stands.
    session_gain: float = 6.0
    #: Ticks a paper stays worth marking. Four times the chat window: a finished
    #: term does not go stale, and a quarter of papers were timing out at 36.
    exam_staleness_ticks: int = _env_int("AC_EXAM_STALENESS", 144)
    #: How far a model's mark may sit from the one the simulation earned. Wider
    #: than the model's ~10-point bias, so it guards outliers rather than grades.
    exam_mark_band: float = _env_float("AC_EXAM_BAND", 30.0)
    #: Sim-days in a term. Two weeks at four sessions a week is eight draws,
    #: enough for attendance to resolve.
    term_days: int = _env_int("AC_TERM_DAYS", 14)
    max_attempts: int = 2


@dataclass(frozen=True)
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    needs: NeedsConfig = field(default_factory=NeedsConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)
    university: UniversityConfig = field(default_factory=UniversityConfig)
    seed: int = _env_int("AC_SEED", 20260827)


CONFIG = Config()
