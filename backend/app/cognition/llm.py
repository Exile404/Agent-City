"""Tiered access to the local models.

The simulation never awaits a model. A request is submitted, a slot is taken
when one frees, and the answer is applied on whatever tick it lands on. That is
the whole reason fifty agents tick at thousands per second while roughly four
inferences per second trickle through underneath.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import httpx

from app.config import CONFIG


class Lane(str, Enum):
    """Which model answers. Two lanes, two separate concurrency budgets."""

    FAST = "fast"
    SMART = "smart"


@dataclass(slots=True)
class Request:
    lane: Lane
    prompt: str
    agent_id: str
    #: What the thought is for — "plan", "chat", "reflect". Stats group by this.
    kind: str
    submitted_tick: int
    on_done: Callable[[str], None]
    system: str = ""
    #: Constrain sampling to valid JSON.
    want_json: bool = False
    #: Constrain it to a shape, not merely to valid syntax.
    json_schema: dict | None = None
    max_tokens: int = 220
    temperature: float = 0.8
    #: Ticks this request stays worth answering; None takes the global default.
    #: A chat perishes in minutes, an exam about a finished term does not.
    staleness_ticks: int | None = None


@dataclass
class Stats:
    submitted: int = 0
    completed: int = 0
    dropped_stale: int = 0
    failed: int = 0
    in_flight: int = 0
    #: Smoothed seconds per completed call, per lane.
    latency: dict[str, float] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "droppedStale": self.dropped_stale,
            "failed": self.failed,
            "inFlight": self.in_flight,
            "latency": {k: round(v, 2) for k, v in self.latency.items()},
            "byKind": dict(self.by_kind),
        }


class LLMClient:
    """Fire-and-forget access to Ollama, with a hard concurrency cap per lane."""

    def __init__(self, now_tick: Callable[[], int]) -> None:
        cfg = CONFIG.llm
        self._now_tick = now_tick
        self._http = httpx.AsyncClient(base_url=cfg.base_url, timeout=cfg.request_timeout)
        # The measured ceiling of the GPU, expressed as code: above these the
        # card stops scaling and the models start thrashing VRAM.
        self._slots = {
            Lane.FAST: asyncio.Semaphore(cfg.fast_concurrency),
            Lane.SMART: asyncio.Semaphore(cfg.smart_concurrency),
        }
        self._model = {Lane.FAST: cfg.fast_model, Lane.SMART: cfg.smart_model}
        self._tasks: set[asyncio.Task] = set()
        self.stats = Stats()

    def submit(self, req: Request) -> None:
        """Queue a thought. Returns at once — the simulation never blocks."""
        if not CONFIG.llm.enabled:
            return
        self.stats.submitted += 1
        self.stats.by_kind[req.kind] = self.stats.by_kind.get(req.kind, 0) + 1

        task = asyncio.create_task(self._run(req))
        # Hold a strong reference: asyncio only keeps weak ones, so a running
        # task can otherwise be garbage-collected mid-flight.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _stale(self, req: Request) -> bool:
        window = req.staleness_ticks or CONFIG.llm.staleness_ticks
        return self._now_tick() - req.submitted_tick > window

    async def _run(self, req: Request) -> None:
        # Before queueing: a request that already waited out its window should
        # not consume GPU time at all.
        if self._stale(req):
            self.stats.dropped_stale += 1
            return

        started = time.perf_counter()
        async with self._slots[req.lane]:
            # After winning a slot: the queue may have been long.
            if self._stale(req):
                self.stats.dropped_stale += 1
                return
            self.stats.in_flight += 1
            started = time.perf_counter()
            try:
                text = await self._generate(req)
            except Exception:
                self.stats.failed += 1
                return
            finally:
                self.stats.in_flight -= 1

        # On the way out: an answer this late describes a situation the agent
        # has already walked away from, and acting on it reads as a glitch.
        if self._stale(req):
            self.stats.dropped_stale += 1
            return

        elapsed = time.perf_counter() - started
        prev = self.stats.latency.get(req.lane.value, elapsed)
        self.stats.latency[req.lane.value] = prev * 0.8 + elapsed * 0.2
        self.stats.completed += 1
        req.on_done(text)

    async def _generate(self, req: Request) -> str:
        body: dict = {
            "model": self._model[req.lane],
            "prompt": req.prompt,
            "stream": False,
            "options": {"num_predict": req.max_tokens, "temperature": req.temperature},
        }
        if req.system:
            body["system"] = req.system
        if req.want_json:
            # Ollama constrains sampling to valid JSON, which removes nearly all
            # the parse-failure handling a 3B model would otherwise need.
            body["format"] = req.json_schema or "json"

        response = await self._http.post("/api/generate", json=body)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    async def warmup(self) -> None:
        """Load both models before the first real request.

        A cold load is several seconds — long enough that the first agent to
        think would blow its staleness window and be discarded.
        """
        for lane in (Lane.FAST, Lane.SMART):
            try:
                await self._http.post(
                    "/api/generate",
                    json={
                        "model": self._model[lane],
                        "prompt": "hi",
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                )
            except Exception:
                pass  # a missing model shouldn't stop the city from running

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await self._http.aclose()


def parse_json(text: str) -> dict | None:
    """Best-effort parse of a model reply.

    `format: "json"` makes this succeed nearly always, but a model can still
    return a bare array or wrap the object in prose, so fall back to slicing
    out the outermost braces rather than losing the whole response.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None