"""Wire format between the simulation and the browser.

Split by what changes: the map and everyone's identity go out once on connect,
only positions and events go out each tick. Fifty agents at three fields each is
~1.5 KB of JSON per tick, so there is no delta encoding and no binary packing.
"""

from __future__ import annotations

import base64

from app.config import CONFIG
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


def tick_message(sim: Simulation) -> dict:
    """What changed. Sent every tick."""
    tick = sim.clock.tick
    return {
        "type": "tick",
        "t": tick,
        "clock": str(sim.clock),
        # Numeric time for the renderer's sun; parsing the display string
        # would be silly when the clock already knows the number.
        "minuteOfDay": sim.clock.minute_of_day,
        "agents": [[a.x, a.y, a.action.kind.value] for a in sim.agents],
        # Events carry their own tick, so "this tick's" is a filter rather than a
        # read cursor the server would have to track per client.
        "events": [text for t, text in sim.events if t == tick],
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
            }
    return None