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


class Hub:
    """Owns the simulation and every connected browser.

    The sim ticks whether or not anyone is watching, so a browser joining late
    finds a world with history instead of one that begins on arrival.
    """

    def __init__(self) -> None:
        self.sim = Simulation()
        self.clients: set[WebSocket] = set()
        self.speed = CONFIG.loop.default_speed
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        while True:
            if self.speed <= 0.0:
                await asyncio.sleep(0.1)  # paused, but still answering /control
                continue
            self.sim.tick()
            await self.broadcast(tick_message(self.sim))
            await asyncio.sleep(CONFIG.loop.seconds_per_tick / self.speed)

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
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


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