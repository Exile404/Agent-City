# Agent City

A living city of 50 LLM agents who study at university, interview for jobs, get
hired, and work under a management hierarchy — running entirely on a single
consumer GPU, with no cloud API.

> **Status: Phase 0 complete.** The city runs headless — 50 agents with needs,
> pathfinding, and a daily rhythm. Cognition (Phase 2) and the institutions that
> make it a *career* simulation (Phases 4–6) are not built yet. The roadmap below
> marks exactly what exists and what doesn't.

## The constraint that shapes everything

Fifty agents that each "think" every tick would need 50 LLM inferences per
second. A single RTX 5060 Ti delivers about four. Measured on the target
hardware:

| Lane | Model | Throughput | Concurrent |
|---|---|---|---|
| Fast | `qwen2.5:3b` | 152 tok/s | 2.45 calls/sec @ 4 parallel |
| Smart | `qwen2.5:7b-instruct-q4_K_M` | 78 tok/s | 1.34 calls/sec @ 2 parallel |

That is ~3.8 LLM calls per second against a demand of ~540 calls per simulated
day — roughly 98% saturation at 50 agents. The scarcity is the design problem,
and every architectural decision here follows from it.

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
| **1 — Fast** | `qwen2.5:3b` | ~4 concurrent | Next-action choices, short utterances |
| **2 — Deliberate** | `qwen2.5:7b` | ~2 concurrent | Daily plans, dialogue, interviews, reflection |

Tier 0 is what makes the city look continuously alive: needs decay, path
following, action execution, and critical-need overrides. No agent ever blocks
waiting on a model. A planned cognition scheduler admits Tier 1/2 requests by
priority within a per-tick budget; everything that doesn't win a slot degrades
gracefully to Tier 0.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | World grid, A* pathing, needs, actions, tick loop | ✅ Done |
| 1 | FastAPI + WebSocket, PixiJS renderer, time controls | 🔨 In progress |
| 2 | Memory stream, LLM client, tiered scheduler, daily plans | ⬜ |
| 3 | Co-location conversations, relationships, event feed | ⬜ |
| 4 | University: courses, exams, skill growth, credentials | ⬜ |
| 5 | Companies, job postings, LLM interviews, hiring | ⬜ |
| 6 | Org hierarchy, task assignment, reviews, promotions | ⬜ |
| 7 | Reflection, deterministic replay, metrics dashboard | ⬜ |

## Running it

Requires Python 3.11+ and [Ollama](https://ollama.com) (Ollama is unused until
Phase 2).

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
288 ticks (2 sim-days) in 0.05s -> 5,596 ticks/sec
clock: D2 Wed 06:00
doing now: {'travel': 23, 'sleep': 10, 'socialize': 10, 'exercise': 7}
path cache: 92% hit (916/77)
```

Simulating 50 agents costs ~0.2 ms per tick — about 39 simulated days per real
second. The simulation is free; only thinking is expensive.

## Architecture

```
backend/app/
  config.py        Every tuning knob: pacing, cost, scale
  sim/             Grid, city layout, A* pathing, clock, tick loop, spawning
  agents/          Agent state, needs, skills, action execution
  cognition/       Memory stream, LLM client, scheduler, prompts   (Phase 2)
  institutions/    University, companies, job market, hierarchy    (Phases 4-6)
  net/             FastAPI, WebSocket broadcast, control API       (Phase 1)
frontend/src/
  render/          PixiJS tilemap, sprites, camera                 (Phase 1)
  components/      Inspector, event feed, org chart, stats         (Phase 1)
  state/           WebSocket client, snapshot store                (Phase 1)
```

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

**`sorted()` before shuffling names is load-bearing.** Python randomizes string
hashing per process, so iterating a set of strings differs every run. Without
the sort, the seed is a lie and replay is impossible.

## License

MIT
