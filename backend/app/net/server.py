"""FastAPI + WebSocket: the city's window onto the browser."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import CONFIG
from app.net.protocol import agent_detail, hello_message, tick_message
from app.sim.loop import Simulation
from app.cognition import prompts
from app.cognition.embed import Embedder
from app.cognition.llm import Lane, LLMClient, Request, parse_json
from app.cognition.scheduler import Ask, Scheduler, plan_priority

class Hub:
    def __init__(self) -> None:
        self.sim = Simulation()
        self.clients: set[WebSocket] = set()
        self.speed = CONFIG.loop.default_speed
        self._task: asyncio.Task | None = None

        self.llm = LLMClient(lambda: self.sim.clock.tick)
        self.embedder = Embedder()
        self.scheduler = Scheduler()
        self._thoughts: set[asyncio.Task] = set()
        self.think_errors = 0
    async def _run(self) -> None:
        # Cold-loading a model takes seconds — long enough that the first agent
        # to think would blow its staleness window and be thrown away.
        await self.llm.warmup()
        while True:
            if self.speed <= 0.0:
                await asyncio.sleep(0.1)  # paused, but still answering /control
                continue
            self.sim.tick()
            self._collect_asks()
            self.scheduler.dispatch(self.sim.clock.tick, self._start_thought)
            await self.broadcast(tick_message(self.sim))
            await asyncio.sleep(CONFIG.loop.seconds_per_tick / self.speed)

    def _collect_asks(self) -> None:
        """Every agent states how badly it wants to think. Most will be refused."""
        tick = self.sim.clock.tick
        minutes = CONFIG.world.minutes_per_tick
        for agent in self.sim.agents:
            priority = plan_priority(
                minutes_since_plan=(tick - agent.last_plan_tick) * minutes,
                worst_need=agent.needs.lowest()[1],
                has_plan=bool(agent.plan),
            )
            # Below this an agent is comfortably on-plan; asking would only
            # crowd out someone who actually needs the slot.
            if priority > 1.0:
                self.scheduler.ask(agent.id, "plan", priority, tick)

    def _start_thought(self, ask: Ask) -> None:
        task = asyncio.create_task(self._think(ask))
        self._thoughts.add(task)
        task.add_done_callback(self._thoughts.discard)

    async def _think(self, ask: Ask) -> None:
        """Embed, retrieve, prompt, generate, apply. All off the tick path."""
        try:
            agent = self.sim.by_id.get(ask.agent_id)
            if agent is None:
                return

            # Time moves while a thought queues — measured at 35-135 sim-minutes
            # between asking and the answer landing. Validate steps against when
            # the agent asked, not when the reply arrived, or its earliest
            # intentions get discarded as already past.
            asked_minute = self.sim.clock.minute_of_day

            # Upgrade memories still carrying a lexical vector. Batched: one
            # embed costs ~209ms of round trip, thirty-two cost 90ms in total.
            pending = agent.memory.take_pending()
            if pending:
                vectors = await self.embedder.embed_many([n.text for n in pending])
                for node, vector in zip(pending, vectors):
                    node.vector = vector

            query = await self.embedder.embed("what should I do with the rest of today?")
            recalled = agent.memory.retrieve(query, self.sim.clock.tick)

            reply = await self._generate(
                agent,
                prompts.daily_plan(
                    name=agent.name,
                    age=agent.age,
                    traits=agent.traits,
                    home=self.sim.world.buildings[agent.home_id].name,
                    clock=str(self.sim.clock),
                    needs=agent.needs.as_dict(),
                    memories=[n.text for n in recalled],
                ),
            )
            if reply is None:
                return

            steps = prompts.parse_plan(parse_json(reply), asked_minute)
            if not steps:
                return

            agent.plan = steps
            agent.last_plan_tick = self.sim.clock.tick
            agent.memory.add(
                self.sim.clock.tick,
                "plan",
                "Planned: " + "; ".join(str(s) for s in steps[:3]),
            )
            self.sim.events.append(
                (self.sim.clock.tick, f"{agent.name} decided: {steps[0].why or steps[0]}")
            )
        except Exception:
            self.think_errors += 1 
        finally:
            # Every exit path decrements — timeout, stale drop, exception,
            # missing agent. Miss one and `outstanding` leaks upward until the
            # scheduler refuses everything and the city stops thinking.
            self.scheduler.finished()

    async def _generate(self, agent, prompt: str) -> str | None:
        landed = asyncio.Event()
        box: list[str] = []

        def receive(text: str) -> None:
            box.append(text)
            landed.set()

        self.llm.submit(
            Request(
                lane=Lane.FAST,
                prompt=prompt,
                system=prompts.SYSTEM,
                agent_id=agent.id,
                kind="plan",
                submitted_tick=self.sim.clock.tick,
                want_json=True,
                max_tokens=380,
                temperature=0.7,
                on_done=receive,
            )
        )

        # Wait the staleness window, not request_timeout: a request dropped as
        # stale never calls back, and waiting 90s would pin an outstanding slot
        # for three sim-hours doing nothing.
        window = CONFIG.llm.staleness_ticks * CONFIG.loop.seconds_per_tick + 5.0
        try:
            await asyncio.wait_for(landed.wait(), timeout=window)
        except asyncio.TimeoutError:
            return None
        return box[0] if box else None

    async def broadcast(self, message: dict) -> None:
        # Gather failures first: a set cannot be mutated while iterating, and a
        # client vanishing mid-broadcast would otherwise kill the tick task.
        dead: list[WebSocket] = []
        for ws in self.clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        for task in list(self._thoughts):
            task.cancel()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self.llm.aclose()
        await self.embedder.aclose()

hub = Hub()


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.start()
    yield
    await hub.stop()


app = FastAPI(title="Agent City", lifespan=lifespan)

# Vite dev server runs on another port, so the browser sees a cross origin.
# Dev convenience only — tighten before this is ever public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "tick": hub.sim.clock.tick,
        "clock": str(hub.sim.clock),
        "agents": len(hub.sim.agents),
        "speed": hub.speed,
        "clients": len(hub.clients),
        "scheduler": hub.scheduler.snapshot(),
        "llm": hub.llm.stats.snapshot(),
        "planned": sum(1 for a in hub.sim.agents if a.plan),
        "thinkErrors": hub.think_errors,
    }


@app.get("/agent/{agent_id}")
def agent(agent_id: str) -> dict:
    detail = agent_detail(hub.sim, agent_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="no such agent")
    return detail


@app.post("/control/speed")
def set_speed(value: float) -> dict:
    """0 pauses. Anything else multiplies the tick rate."""
    hub.speed = max(0.0, min(value, max(CONFIG.loop.speed_options)))
    return {"speed": hub.speed}


@app.websocket("/ws")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    hub.clients.add(ws)
    try:
        await ws.send_json(hello_message(hub.sim))
        while True:
            # Nothing expected from the browser yet; this holds the connection
            # open and surfaces a disconnect promptly.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)