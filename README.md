# Agent City

A living city of 50 LLM agents who study at university, interview for jobs, get
hired, and work under a management hierarchy — running entirely on a single
consumer GPU, with no cloud API.

> **Status: Phase 3 complete.** Agents have a memory stream, plan their own days,
> and talk to whoever they run into — all on a local 3B model, admitted through a
> cognition scheduler that rations a measured GPU budget. The institutions that
> make it a *career* simulation (Phases 4–6) are not built yet. The roadmap marks
> what exists.

## The constraint that shapes everything

Fifty agents that each "think" every tick would need 50 LLM inferences per
second. A single RTX 5060 Ti delivers barely more than one. Measured on the
target hardware:

| Lane | Model | Latency | Slots | Throughput |
|---|---|---|---|---|
| Fast — plans | `qwen2.5:3b` | ~2.4s | 2 | 0.83 calls/sec |
| Smart — dialogue | `qwen2.5:3b` | ~1.5s | 1 | 0.67 calls/sec |

Measured end to end that is **1.32 completed calls per second** — about 190 per
simulated day against a demand of roughly 540. The city asks for three times what
the card can deliver. The scarcity is the design problem, and every architectural
decision here follows from it.

Free API tiers were evaluated and rejected: Groq's free tier caps at 6,000
tokens/min (~7 usable calls/min with realistic prompts) and Gemini Flash at
1,500 calls/day. Both are 15–30x *slower* than the local GPU, and would reduce a
2.4-minute simulated day to over an hour.

## The design rule

**The simulation owns the numbers. The LLM owns the language and the judgment
calls.**

Skills, money, exam scores, task output, and promotions are computed
deterministically in Python. The model writes dialogue, daily plans, interview
questions, and verdicts — always conditioned on state that is actually true. A
candidate whose Python skill is 35/100 is prompted to answer *as someone at
35/100*, so a weak agent gives genuinely weak answers, the interviewer scores
them low, and the rejection is earned rather than roleplayed.

This keeps the city coherent, cheap, and replayable.

### Tiered cognition

| Tier | Runs on | Cost | Frequency |
|---|---|---|---|
| **0 — Reflex** | Pure Python | free | Every agent, every tick |
| **1 — Fast** | `qwen2.5:3b` | 2 slots | Daily plans |
| **2 — Deliberate** | `qwen2.5:3b` | 1 slot | Conversations, interviews, reflection |

Tier 0 is what makes the city look continuously alive: needs decay, path
following, action execution, and critical-need overrides. No agent ever blocks
waiting on a model.

The cognition scheduler admits Tier 1/2 requests by priority within a per-tick
budget; anything that doesn't win a slot degrades to Tier 0 and re-asks next
tick with a priority that has risen slightly. Measured over two simulated days
with 50 agents: **2.5 plans and 3.1 conversations per agent per day, 0 dropped as
stale, 0 failed, 0 errors** across a thousand generations. Most agents are on
Tier 0 at any instant, and that is the design working rather than a shortfall —
the GPU affords barely more than one inference a second and reflexes carry
everything between.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | World grid, A* pathing, needs, actions, tick loop | ✅ Done |
| 1 | FastAPI + WebSocket, Three.js renderer, time controls | ✅ Done |
| 2 | Memory stream, LLM client, tiered scheduler, daily plans | ✅ Done |
| 3 | Co-location conversations, relationships, event feed | ✅ Done |
| 4 | University: courses, exams, skill growth, credentials | ⬜ |
| 5 | Companies, job postings, LLM interviews, hiring | ⬜ |
| 6 | Org hierarchy, task assignment, reviews, promotions | ⬜ |
| 7 | Reflection, deterministic replay, metrics dashboard | ⬜ |

## Running it

Requires Python 3.11+, Node 20+, pnpm, and [Ollama](https://ollama.com)
(Ollama is unused until Phase 2).

Ollama must be allowed to serve more than one request at a time, or the lanes
are a fiction — see the design note below:

```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
```

```bash
cd backend
python3 -m venv agent-city-env
source agent-city-env/bin/activate
pip install -e .
```

Run the city headless for two simulated days:

```bash
python -m app.sim.loop
```

```
576 ticks (2 sim-days) in 0.14s -> 4,167 ticks/sec
clock: D2 Wed 06:00
doing now: {'travel': 29, 'sleep': 8, 'exercise': 7, 'eat': 5, 'socialize': 1}
path cache: 77% hit (587/178)
```

Simulating 50 agents costs ~0.24 ms per tick — roughly 14 simulated days per real
second. The simulation is free; only thinking is expensive.

To watch it live, run the server and the frontend in two terminals:

```bash
cd backend && uvicorn app.net.server:app --reload --port 8000
```

```bash
cd frontend && pnpm install && pnpm dev
```

Then open `http://localhost:5173`. Drag to orbit, scroll to zoom, click any
agent to inspect their needs, skills, current action, and who they know — each
relationship showing how often they have met, how they feel about each other, and
the last thing that was said.

## Architecture

```
backend/app/
  config.py        Every tuning knob: pacing, cost, scale
  sim/             70x50 grid, city layout, A* pathing, clock, tick loop
  agents/          Agent state, needs, skills, actions, relationships
  cognition/       Memory stream, LLM client, scheduler, prompts   (Phase 2)
  institutions/    University, companies, job market, hierarchy    (Phases 4-6)
  net/             FastAPI, WebSocket broadcast, control API
frontend/src/
  render/          Three.js scene, buildings, agent figures, bloom
  state/           WebSocket client, snapshot store
```

## How an agent thinks

An agent **asks** to think; the scheduler decides who actually gets to. Fifty
agents ask roughly 500 times a tick and about 1 is admitted — a ~2% admission
rate that is the whole point, not a fault.

An admitted thought runs entirely off the tick path:

1. **Embed** any memories still carrying a placeholder vector — batched, because
   one embedding costs ~209ms of round trip while thirty-two cost 90ms in total.
2. **Retrieve** with relevance gating the candidate set and recency plus
   importance ordering what is already on topic, each component min-max
   normalised across the candidates and anything below 85% of the best match
   dropped rather than padding the result out.
3. **Prompt** with a closed vocabulary — the model chooses from real actions at
   legal places, because a 3B model will otherwise happily invent
   `"search_for_restaurant"`.
4. **Validate** the reply against that vocabulary. Malformed steps are dropped
   and an empty plan simply means Tier 0. A bad generation can never stall an
   agent.

The simulation never awaits a model. Requests are fire-and-forget, answers apply
on whatever tick they land, and anything older than its staleness window is
discarded rather than acted on.

## How two agents talk

Conversation is mostly free. Two agents in the same building acknowledge each
other, feel slightly less alone, and drift together or apart according to
temperament — pure Python, every tick, no model involved. Measured at **1.54
encounters per tick** against a budget of **0.27 generated conversations**, so
about one in five can afford words. The greeting is not a fallback; it is what
happens to the overwhelming majority, and the city looks sociable because of it.

Which encounter gets words is scored *before* the greeting, not after. `greet()`
stamps `last_talked_tick` and increments `times_met` — the very facts the score
reads — so the reverse order makes every pair look like acquaintances who just
spoke. First meetings dominate at 6.0; known pairs score on time apart and
strength of feeling. The gate sits at 2.0, in the valley of a bimodal
distribution, leaving 0.61 candidates per tick against 0.27 slots. That
oversupply is deliberate: refusing a candidate costs nothing, since it still gets
the greeting, while an idle slot is GPU thrown away.

Conversations bypass the plan scheduler entirely. A refused plan re-asks next
tick at a higher priority; a refused encounter is simply gone, because the
greeting has already put that pair on a three-hour cooldown. There is no queue to
be fair about — only a choice of which of this tick's pairs is worth words.

The whole exchange is one generation rather than one per turn, and the simulation
decides what it changed: warmth moves affinity by at most a few points, so a
single conversation can *start* a friendship and never manufacture one. Who ends
up close stays mostly a matter of who keeps turning up. Affinity reaches +26 by
day 6, with friendships forming and nothing forced.

Each exchange is stored as a memory at importance 4.0, above an observation. That
one decision is what makes conversations compound — the next one remembers the
last through ordinary retrieval, with no extra plumbing anywhere. Agents visibly
relay third-hand news:

> **Diego Rahman:** Same, and Nadia mentioned you're a bit lost around here.
>
> **Rin Haddad:** Hey Hugo, seen Elias? He looks stressed.

Nobody wrote a gossip system. It falls out of dialogue being a memory.

## The city

A 70×50 grid on a 10-tile road pitch with 2-tile carriageways, giving 7×5 blocks
— all 35 built. Two campuses (Agent City University, Eastgate Polytechnic), five
offices, three cafes, three gyms, two parks, fourteen homes, and a civic set:
hospital, market, library, bank, power station, gas works. The utilities exist as
*employers*, not scenery — a job market with three software firms has no texture
to it.

## Design notes

**Tiles are a flat `bytearray`.** 4,800 tiles in one contiguous block rather
than nested lists, so the whole map ships to the browser as a single binary blob
instead of serialized nested JSON.

**Every building has one door.** A* needs exactly one goal tile — targeting a
building's centre paths agents into walls. A door also gives institutions a
natural check-in point and produces organic congestion at entrances.

**Roads cost 1.0, grass costs 1.6.** Agents prefer streets with no "prefer
roads" logic anywhere; it falls out of the A* weights. A typical cross-town
route uses 32 road tiles and zero grass.

**Actions have duration, not per-tick decisions.** Agents commit to something,
travel to it, and stay until it completes. Most ticks are "keep going", which is
the structural reason the simulation stays cheap — decisions happen only at the
seams.

**Travel chains its intent.** A `TRAVEL` action carries a `then` field holding
what to start on arrival, so "walk to the cafe, then eat" needs no separate
state machine and agents can't arrive having forgotten why they came.

**Needs decay and restore independently, every tick.** Sleeping still burns
hunger, so agents wake rested and starving without anyone writing that rule.

**Importance is assigned, not asked for.** Scoring each memory with a model call
would cost ~1,000 calls a day against a ~190/day budget — more than the entire
dialogue allowance, to rate things like "walked to a cafe". Memory kinds carry
default weights instead.

**Min-max normalisation is what makes retrieval weights mean anything.** Recency
and importance naturally reach 1.0 while cosine similarity rarely passes 0.6, so
without rescaling across the candidate set an on-topic memory could never
outrank a merely recent one. Normalising raised precision@8 from 50% to 60%;
dropping candidates below 85% of the best match rather than padding to k raised
it to 83%.

**Retrieval halves a memory's age rather than resetting it.** A full reset pins
every retrieved memory at maximum recency, so the same handful wins every
subsequent query and the agent's mind freezes on whatever it thought about first.

**Plan steps are validated against when the agent asked, not when the reply
landed.** Measured drift between the two was 35-135 simulated minutes, which was
silently discarding an agent's earliest intentions as "already past".

**Late plan steps execute; ancient ones are dropped.** Draining every overdue
step at once burned a full-day plan in ninety minutes. Taking the earliest step
still inside a two-hour grace window more than doubled how long a plan survives.

**Both lanes run the same 3B model.** A 7B alongside it meant three resident
models costing 3.5GB of *host* RAM — Ollama launches llama-server with
`--no-mmap`, so weights are copied into system memory even when they sit entirely
on the GPU. On a 15GiB machine that was enough for the kernel OOM killer to take
the desktop session down. The lane split survives as a policy reservation,
holding a slot for dialogue so planning cannot starve it, rather than a boundary
between two models.

**Ollama serves one request at a time unless told otherwise.** It launches
llama-server with `-np 1` by default, so surplus requests queue *inside* Ollama
and expire against the staleness window having never run — measured at 15 of 60
plan requests dropped unserved. `OLLAMA_NUM_PARALLEL=2` takes that to zero. Slots
are not free either: at 4, llama-server held 4.5GB of host RAM for a model
occupying 2.4GB of VRAM, because each slot carries its own KV cache.

**An unusable conversation is discarded whole, never filtered.** Removing one
offending line leaves a reply to something nobody said. The greeting has already
happened by then, so both agents still met and still felt less alone — nothing in
the simulation is left half-applied when a generation is thrown away.

**The event feed is a cursor, not a filter.** Selecting events stamped with the
current tick silently dropped everything that completes asynchronously: a plan or
a conversation lands several ticks after its frame has gone out, so the feed only
ever showed events raised inside `sim.tick()`. A monotonic counter and a
high-water mark per broadcast fixed it — and the deque is capped, so its length
stops being a usable cursor the moment it fills.

**Staleness is counted in ticks, so every wait on it must be too.** A fixed
number of real seconds is wrong in both directions: at 8x the window expires in
2.25s while the waiter sits for 23s, pinning both conversation participants for a
simulated day after their request was already dropped; at 0.5x it times out on
requests that were about to land.

**Density beat vehicles.** Journeys averaged 42 tiles — 1.8 sim-hours on foot —
so cars were added to make a large map affordable. They looked wrong and solved
the wrong problem: the real cause was one gym and two cafes serving the entire
map, so `nearest()` routinely sent people across town. Scattering services one
per district cut the median journey to 26 tiles, and the cars were deleted.

**Nothing is sun-lit.** The first 3D pass used physical lighting and was
unreadable at night. Buildings and agents now emit their own colour with bloom on
top, which is both cheaper and the look the scene was after.

**Occupancy is the lighting.** Building glow scales with how many agents are
inside, so simulation state *is* the visual effect — an office brightens as
people arrive and goes dark when they leave.

**Which means a paused city is an unlit city.** Occupancy is derived from agent
positions, and the opening `hello` carries the roster but not positions — so a
client connecting while the sim was paused saw a correctly-built, completely
black city until someone pressed play. One tick frame is now sent on connect,
cursored so no event backlog replays into a fresh feed.

**8x delivers about 4x.** The tick loop asks for 62ms between ticks but the body
costs ~70ms, because asyncio is single-threaded and the loop shares it with eight
outstanding thoughts and two conversations doing embeds and HTTP. Measured 7.4
ticks/sec at 8x against 1.9 at 1x. The speed control is honest to about 4x and
saturates past it.

**Traffic lights would have been a second project.** Making them real means
occupancy-aware pathfinding, which invalidates the A* cache every tick and turns
into a traffic simulator. Cut in favour of the cognition work.

**Agents walk on a hip pivot.** Leg geometry is pre-translated so it hangs from
its origin; rotating then reads as a stride rather than a propeller. A single
phase drives the legs in opposition and a torso bob at double rate, because the
body rises once per step, not once per cycle.

**`sorted()` before shuffling names is load-bearing.** Python randomizes string
hashing per process, so iterating a set of strings differs every run. Without
the sort, the seed is a lie and replay is impossible.

## License

MIT
