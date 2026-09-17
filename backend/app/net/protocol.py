"""Wire format between the simulation and the browser.

Split by what changes: the map and everyone's identity go out once on connect,
only positions and events go out each tick. Fifty agents at three fields each is
~1.5 KB of JSON per tick, so there is no delta encoding and no binary packing.
"""

from __future__ import annotations

import base64

from app.agents.agent import Agent
from app.config import CONFIG
from app.institutions.university import BY_ID
from app.sim.layout import BLOCK, ROAD_WIDTH
from app.sim.loop import Simulation


def hello_message(sim: Simulation) -> dict:
    """Everything that never changes. Sent once, when a browser connects."""
    world = sim.world
    return {
        "type": "hello",
        "world": {
            "width": world.width,
            "height": world.height,
            "minutesPerTick": CONFIG.world.minutes_per_tick,
            "tilesPerTick": CONFIG.world.tiles_per_tick,
            # Road geometry, so the renderer can place traffic lights at
            # intersections without re-deriving the street grid.
            "block": BLOCK,
            "roadWidth": ROAD_WIDTH,
        },
        # 4,800 raw tile bytes -> ~6.4 KB of base64, once. The renderer bakes it
        # into a background layer and never looks at it again.
        "tiles": base64.b64encode(bytes(world.tiles)).decode("ascii"),
        "buildings": [
            {
                "id": b.id,
                "kind": b.kind.value,
                "name": b.name,
                "x": b.x,
                "y": b.y,
                "w": b.w,
                "h": b.h,
                "door": list(b.door),
            }
            for b in world.buildings.values()
        ],
        # Identity is constant, so names and traits ship once. Tick frames refer
        # to agents purely by their index into this list.
        "agents": [
            {"id": a.id, "name": a.name, "age": a.age, "traits": a.traits}
            for a in sim.agents
        ],
    }


def tick_message(sim: Simulation, since_total: int) -> dict:
    """What changed. Sent every tick."""
    tick = sim.clock.tick
    # Everything logged since the last frame, not everything stamped with the
    # current tick. Plans and conversations complete asynchronously, after their
    # tick's frame has already gone out, so a `t == tick` filter dropped them
    # entirely — the feed only ever showed events raised inside sim.tick().
    fresh = sim.events_total - since_total
    return {
        "type": "tick",
        "t": tick,
        "clock": str(sim.clock),
        # Numeric time for the renderer's sun; parsing the display string
        # would be silly when the clock already knows the number.
        "minuteOfDay": sim.clock.minute_of_day,
        "agents": [[a.x, a.y, a.action.kind.value] for a in sim.agents],
        # The `fresh > 0` guard is load-bearing: list[-0:] is the whole list,
        # which would replay all 200 events on every quiet tick.
        "events": [text for _, text in list(sim.events)[-fresh:]] if fresh > 0 else [],
    }

def _study(a: Agent) -> dict:
    """The current term, as a transcript row."""
    e = a.enrollment
    return {
        "course": e.course.name,
        "campus": e.course.campus_id,
        "attendance": round(e.attendance, 2),
        "attended": e.sessions_attended,
        "offered": e.sessions_offered,
        "attempt": e.attempt,
        "awaitingExam": e.awaiting_exam,
    }


def agent_detail(sim: Simulation, agent_id: str) -> dict | None:
    """Full state for the inspector panel. Fetched on click, never streamed."""
    for a in sim.agents:
        if a.id == agent_id:
            return {
                "id": a.id,
                "name": a.name,
                "age": a.age,
                "traits": a.traits,
                "home": sim.world.buildings[a.home_id].name,
                "pos": [a.x, a.y],
                "action": str(a.action),
                "needs": {k: round(v, 1) for k, v in a.needs.as_dict().items()},
                "skills": {k: round(v, 1) for k, v in a.skills.items()},
                "money": round(a.money, 2),
                "employed": a.employed,
                # None for non-students: a blank, not a row of zeroes that reads
                # like a failing one.
                "study": _study(a) if a.enrollment is not None else None,
                "credentials": [BY_ID[c].name for c in a.credentials if c in BY_ID],
                # The moments an agent would actually tell you about, exam papers
                # included.
                "milestones": [n.text for n in a.memory.nodes if n.kind == "milestone"][-5:],
                # Ordered by how strongly they feel rather than how warmly, so a
                # rivalry is as visible as a friendship, and capped: an agent who
                # has met forty people would push everything else off the panel.
                "relationships": [
                    {
                        "name": sim.by_id[other_id].name,
                        "affinity": round(rel.affinity, 1),
                        "label": rel.label,
                        "timesMet": rel.times_met,
                        "note": rel.note,
                    }
                    for other_id, rel in sorted(
                        a.relationships.items(), key=lambda kv: -abs(kv[1].affinity)
                    )[:8]
                    if other_id in sim.by_id
                ],
            }
    return None