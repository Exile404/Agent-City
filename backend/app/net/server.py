"""FastAPI + WebSocket: the city's window onto the browser."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.agents.agent import Agent
from app.config import CONFIG
from app.net.protocol import agent_detail, hello_message, tick_message
from app.sim.loop import Simulation
from app.cognition import prompts
from app.cognition.embed import Embedder
from app.cognition.llm import Lane, LLMClient, Request, parse_json
from app.cognition.scheduler import Ask, Scheduler, plan_priority
from app.institutions.university import Enrollment, baseline_score


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

        #: In-flight conversation tasks. len() is the admission check.
        self._chats: set[asyncio.Task] = set()
        #: Agents inside a generating conversation. A generation spans several
        #: ticks, and the pair cooldown only guards the same two people — without
        #: this an agent could be mid-sentence with one person and start a second
        #: conversation with someone else.
        self._talking: set[str] = set()
        self.chats_done = 0
        self.chats_blocked = 0
        #: High-water mark for the event feed. One cursor, not one per client:
        #: every client receives the same broadcast.
        self._events_sent = 0
        #: In-flight exam tasks. One paper at a time — they are rare and never
        #: urgent, and should not contend with conversations for the smart lane.
        self._exams: set[asyncio.Task] = set()
        self.exams_done = 0
        #: Reply arrived but was unusable (no mark, or unpublishable).
        self.exams_fallback = 0
        #: No reply inside the window. Kept separate: the two want opposite fixes.
        self.exams_timeout = 0
        #: Running (model mark - earned mark), before the clamp — whether the
        #: anchor in the prompt is being honoured.
        self.mark_delta_sum = 0.0
        self.mark_delta_n = 0

    @property
    def _tick_seconds(self) -> float:
        """Real seconds per tick at the current speed.

        Guarded against zero because /control/speed can land between the pause
        check and any use of this, and every caller divides by it.
        """
        return CONFIG.loop.seconds_per_tick / max(self.speed, 0.05)

    async def _run(self) -> None:
        # Cold-loading a model takes seconds — long enough that the first agent
        # to think would blow its staleness window and be thrown away.
        await self.llm.warmup()
        while True:
            if self.speed <= 0.0:
                await asyncio.sleep(0.1)  # paused, but still answering /control
                continue
            self.sim.tick()
            self._collect_chats()
            self._collect_exams()
            self._collect_asks()
            self.scheduler.dispatch(self.sim.clock.tick, self._start_thought)
            msg = tick_message(self.sim, self._events_sent)
            self._events_sent = self.sim.events_total
            await self.broadcast(msg)
            await asyncio.sleep(self._tick_seconds)

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

            await self._vectorize(agent)

            query = await self.embedder.embed("what should I do with the rest of today?")
            recalled = agent.memory.retrieve(query, self.sim.clock.tick)

            reply = await self._generate(
                agent.id,
                prompts.daily_plan(
                    name=agent.name,
                    age=agent.age,
                    traits=agent.traits,
                    home=self.sim.world.buildings[agent.home_id].name,
                    clock=str(self.sim.clock),
                    needs=agent.needs.as_dict(),
                    memories=[n.text for n in recalled],
                ),
                lane=Lane.FAST,
                system=prompts.SYSTEM,
                kind="plan",
                max_tokens=380,
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
            self.sim.log(f"{agent.name} decided: {steps[0].why or steps[0]}")
        except Exception:
            self.think_errors += 1
        finally:
            # Every exit path decrements — timeout, stale drop, exception,
            # missing agent. Miss one and `outstanding` leaks upward until the
            # scheduler refuses everything and the city stops thinking.
            self.scheduler.finished()

    async def _vectorize(self, agent: Agent) -> None:
        """Upgrade memories still carrying a lexical placeholder vector.

        Batched because it has to be: one embed costs ~209ms of round trip,
        thirty-two cost 90ms in total. Shared by planning and conversation.
        """
        pending = agent.memory.take_pending()
        if not pending:
            return
        vectors = await self.embedder.embed_many([n.text for n in pending])
        for node, vector in zip(pending, vectors):
            node.vector = vector

    async def _generate(
        self,
        agent_id: str,
        prompt: str,
        *,
        lane: Lane,
        system: str,
        kind: str,
        max_tokens: int,
        temperature: float = 0.7,
        staleness: int | None = None,
        schema: dict | None = None,
    ) -> str | None:
        landed = asyncio.Event()
        box: list[str] = []

        def receive(text: str) -> None:
            box.append(text)
            landed.set()

        self.llm.submit(
            Request(
                lane=lane,
                prompt=prompt,
                system=system,
                agent_id=agent_id,
                kind=kind,
                submitted_tick=self.sim.clock.tick,
                want_json=True,
                max_tokens=max_tokens,
                temperature=temperature,
                staleness_ticks=staleness,
                json_schema=schema,
                on_done=receive,
            )
        )

        # Wait the staleness window, not request_timeout: a request dropped as
        # stale never calls back, and waiting 90s would pin an outstanding slot
        # for three sim-hours doing nothing.
        #
        # Measured in the same clock as the staleness it waits on: staleness is
        # counted in ticks, and a tick is not a fixed number of real seconds. At
        # 8x the 36-tick window expires in 2.25s, so a fixed 23s wait pinned both
        # chat participants in `_talking` for a sim-day after the request had
        # already been dropped. At 0.5x it ran the other way — 36 ticks take 36s,
        # and the fixed wait timed out on a request that was about to land.
        window = (staleness or CONFIG.llm.staleness_ticks) * self._tick_seconds + 5.0
        try:
            await asyncio.wait_for(landed.wait(), timeout=window)
        except asyncio.TimeoutError:
            return None
        return box[0] if box else None

    def _collect_chats(self) -> None:
        """Promote this tick's most interesting encounter to real dialogue.

        Chats deliberately bypass the Scheduler. A plan ask can be refused and
        re-asked next tick at a higher priority; an encounter cannot — greet()
        has already put the pair on cooldown, so a pair passed over now is gone
        for three sim-hours. There is no queue to be fair about, only a choice of
        which of this tick's pairs is worth words. They also run on the smart
        lane, which has its own semaphore, so sharing the planner's admission
        budget would make planning worse for no GPU reason.
        """
        if len(self._chats) >= CONFIG.cognition.max_chats:
            return

        best: tuple[float, Agent, Agent] | None = None
        for a, b, interest in self.sim.last_encounters:
            if interest < CONFIG.cognition.chat_min_interest:
                continue
            if a.id in self._talking or b.id in self._talking:
                continue
            if best is None or interest > best[0]:
                best = (interest, a, b)

        if best is not None:
            self._start_chat(best[1], best[2])

    def _start_chat(self, a: Agent, b: Agent) -> None:
        # Place and activity are captured here, on the tick path: a generation
        # spans several ticks and both agents may have walked out before it lands.
        building = self.sim.world.buildings.get(a.action.target_id or "")
        if building is None:
            return
        self._talking.add(a.id)
        self._talking.add(b.id)
        task = asyncio.create_task(self._chat(a, b, building.name, a.action.kind.value))
        self._chats.add(task)
        task.add_done_callback(self._chats.discard)

    async def _chat(self, a: Agent, b: Agent, place: str, doing: str) -> None:
        """One generation, two memories, one bounded nudge to affinity.

        The greeting already happened on the tick path, so everything here is
        additive. If the model is slow, returns junk, or says something
        unpublishable, the encounter still counted and both agents still felt
        less alone — nothing in the simulation is left half-applied.
        """
        try:
            await self._vectorize(a)
            await self._vectorize(b)

            tick = self.sim.clock.tick
            query = await self.embedder.embed("what is going on in my life right now?")
            # k=3 rather than slicing the default 8: retrieve() halves
            # last_access on everything it returns, so slicing afterwards left
            # five memories looking artificially recent that nothing ever read.
            a_recall = a.memory.retrieve(query, tick, k=3)
            b_recall = b.memory.retrieve(query, tick, k=3)

            rel = a.relationships[b.id]
            # times_met is already 1 by now — greet() ran before the score did —
            # so label would call a first meeting "an acquaintance".
            relation = "someone you have only just met" if rel.times_met <= 1 else rel.label

            reply = await self._generate(
                a.id,
                prompts.conversation(
                    a_name=a.name,
                    a_traits=a.traits,
                    a_memories=[n.text for n in a_recall],
                    b_name=b.name,
                    b_traits=b.traits,
                    b_memories=[n.text for n in b_recall],
                    place=place,
                    doing=doing,
                    relation=relation,
                    clock=str(self.sim.clock),
                ),
                lane=Lane.SMART,
                system=prompts.CONVERSATION_SYSTEM,
                kind="chat",
                max_tokens=260,
                temperature=0.85,
            )
            if reply is None:
                return

            lines, warmth = prompts.parse_conversation(parse_json(reply), a.name, b.name)
            if not lines:
                return
            if not prompts.is_publishable(lines):
                # Discarded whole rather than edited: a filtered line leaves a
                # reply to something nobody said.
                self.chats_blocked += 1
                return

            self._apply_chat(a, b, lines, warmth, place)
        except Exception:
            self.think_errors += 1
        finally:
            self._talking.discard(a.id)
            self._talking.discard(b.id)

    def _apply_chat(
        self, a: Agent, b: Agent, lines: list[tuple[str, str]], warmth: int, place: str
    ) -> None:
        """The model said it; the simulation decides what it changed.

        Warmth is worth at most a few points either way. One conversation should
        be able to start a friendship and never manufacture one, so who ends up
        close stays mostly a matter of who keeps turning up.
        """
        tick = self.sim.clock.tick
        transcript = " / ".join(f"{who.split()[0]}: {says}" for who, says in lines)

        for x, y in ((a, b), (b, a)):
            rel = x.relationships[y.id]
            rel.affinity = max(-100.0, min(100.0, rel.affinity + warmth * 2.5))
            rel.note = next((s for who, s in reversed(lines) if who == y.name), rel.note)
            # Stored as "dialogue" (importance 4.0, above an observation): this is
            # what lets the next conversation remember the last one, with no extra
            # plumbing — it simply surfaces through normal retrieval.
            x.memory.add(tick, "dialogue", f"Talked with {y.name} at {place} — {transcript}")
            # A real exchange is worth more than the nod greet() already gave.
            x.needs.restore("social", 10.0)

        self.chats_done += 1
        self.sim.log(f"{a.name} & {b.name} at {place} — {lines[0][1]}")

    def _collect_exams(self) -> None:
        """Send one pending paper to the model.

        Bypasses the Scheduler like chats do. A paper that misses its chance is
        not lost: the sim marks it once the grace window expires.
        """
        if self._exams or not self.sim.pending_exams:
            return
        self._start_exam(self.sim.pending_exams[0])

    def _start_exam(self, agent: Agent) -> None:
        e = agent.enrollment
        if e is None:
            return
        # Stand the grace timer down while we work, so the two deadlines never race.
        e.in_flight = True
        task = asyncio.create_task(self._exam(agent, e))
        self._exams.add(task)
        task.add_done_callback(self._exams.discard)

    async def _exam(self, agent: Agent, e: Enrollment) -> None:
        """The model writes the paper and marks it; the sim decides what it means."""
        # Read before the first await: sit_exam replaces the enrollment outright.
        course, attempt = e.course, e.attempt
        attended, offered = e.sessions_attended, e.sessions_offered
        attendance = e.attendance
        try:
            skill = agent.skills[course.skill]
            expected = baseline_score(skill, attendance, course.difficulty)

            reply = await self._generate(
                agent.id,
                prompts.exam(
                    name=agent.name,
                    course=course.name,
                    campus=self.sim.world.buildings[course.campus_id].name,
                    skill_name=course.skill,
                    skill=skill,
                    attended=attended,
                    offered=offered,
                    attempt=attempt,
                    expected=expected,
                ),
                lane=Lane.SMART,
                system=prompts.EXAM_SYSTEM,
                kind="exam",
                max_tokens=480,
                temperature=0.7,
                staleness=CONFIG.university.exam_staleness_ticks,
                schema=prompts.EXAM_SCHEMA,
            )

            # Already marked by the grace timer: marking again would award a
            # second credential for one term.
            if agent not in self.sim.pending_exams:
                return

            if reply is None:
                self.exams_timeout += 1
                self.sim.sit_exam(agent)
                return

            question, answer, mark, comment = prompts.parse_exam(parse_json(reply))
            publishable = prompts.is_publishable([("q", question), ("a", answer)])
            if mark is None or not publishable:
                self.exams_fallback += 1
                # The simulation owns the number: a missing mark is no reason
                # to bin a paper the model wrote well. Grade it and keep the script.
                self.sim.sit_exam(agent)
                if publishable and question and answer:
                    agent.memory.add(
                        self.sim.clock.tick,
                        "milestone",
                        f"{course.name} exam — asked: {question} — I answered: {answer}",
                    )
                return

            # Recorded raw, before the clamp inside sit_exam.
            self.mark_delta_sum += mark - expected
            self.mark_delta_n += 1

            score, passed = self.sim.sit_exam(agent, mark)
            self.exams_done += 1
            if question and answer:
                agent.memory.add(
                    self.sim.clock.tick,
                    "milestone",
                    f"{course.name} exam — asked: {question} — I answered: {answer}",
                )
            if comment:
                self.sim.log(f"{agent.name} — {course.name} examiner: {comment}")
        except Exception:
            self.think_errors += 1
        finally:
            # On a raise or a cancel this is what stops the student waiting forever.
            e.in_flight = False

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
        for task in list(self._thoughts) + list(self._chats):
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
        "chatsDone": hub.chats_done,
        "chatsBlocked": hub.chats_blocked,
        "chatsInFlight": len(hub._chats),
        "examsDone": hub.exams_done,
        "examsFallback": hub.exams_fallback,
        "examsTimeout": hub.exams_timeout,
        "examMarkDelta": (
            round(hub.mark_delta_sum / hub.mark_delta_n, 1) if hub.mark_delta_n else None
        ),
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
        # One frame immediately, before waiting on the tick loop. The city is lit
        # entirely by occupancy, and occupancy is derived from agent positions —
        # which hello does not carry. A client connecting while the sim is paused
        # would otherwise see a correctly-built but completely unlit city until
        # someone pressed play. The cursor is events_total so this frame delivers
        # positions without replaying the whole backlog into a fresh feed.
        await ws.send_json(tick_message(hub.sim, hub.sim.events_total))
        while True:
            # Nothing expected from the browser yet; this holds the connection
            # open and surfaces a disconnect promptly.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)