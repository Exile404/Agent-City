"""Wire format between the simulation and the browser.

Split by what changes: the map and everyone's identity go out once on connect,
only positions and events go out each tick. Fifty agents at three fields each is
~1.5 KB of JSON per tick, so there is no delta encoding and no binary packing.
"""

from __future__ import annotations

import base64

from app.agents.agent import Agent
from app.config import CONFIG
from app.institutions.companies import live_warnings
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
        # Names and traits ship here, and again in a tick frame only when
        # someone retires and a newcomer takes their place. Tick frames refer
        # to agents purely by their index into this list.
        "agents": _roster(sim),
    }


def _roster(sim: Simulation) -> list[dict]:
    return [{"id": a.id, "name": a.name, "age": a.age, "traits": a.traits} for a in sim.agents]


def tick_message(sim: Simulation, since_total: int, roster_sent: int | None = None) -> dict:
    """What changed. Sent every tick."""
    tick = sim.clock.tick
    # Everything logged since the last frame, not everything stamped with the
    # current tick. Plans and conversations complete asynchronously, after their
    # tick's frame has already gone out, so a `t == tick` filter dropped them
    # entirely — the feed only ever showed events raised inside sim.tick().
    fresh = sim.events_total - since_total
    msg = {
        "type": "tick",
        "t": tick,
        "clock": str(sim.clock),
        # Numeric time for the renderer's sun; parsing the display string
        # would be redundant when the clock already has the number.
        "minuteOfDay": sim.clock.minute_of_day,
        "agents": [[a.x, a.y, a.action.kind.value] for a in sim.agents],
        # The `fresh > 0` guard is load-bearing: list[-0:] is the whole list,
        # which would replay all 200 events on every quiet tick.
        "events": [text for _, text in list(sim.events)[-fresh:]] if fresh > 0 else [],
    }
    # Someone retired and someone new moved in since the last frame: resend the
    # names. About fifteen times a sim-year, so the whole list is fine.
    if roster_sent is not None and roster_sent != sim.roster_version:
        msg["roster"] = _roster(sim)
    return msg


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


def _work(sim: Simulation, a: Agent) -> dict:
    """The current job, as a payroll row."""
    j = a.job
    boss = sim.manager_of(a)
    return {
        "role": j.role.title,
        "employer": sim.world.buildings[j.employer_id].name,
        "skill": j.role.skill,
        "wage": round(j.role.wage),
        "attendance": round(j.attendance, 2),
        "attended": j.shifts_attended,
        "offered": j.shifts_offered,
        "manager": {"name": boss.name, "role": boss.job.role.title} if boss else None,
        # Top of the ladder and "nobody above because the seats are empty" read
        # the same without this, and the panel words them differently.
        "top": j.role.reports_to is None and j.role.open_market,
        # The minimum-wage floor has no ladder at all, not an empty top rung.
        "floor": not j.role.open_market,
        "reports": [r.name for r in sim.reports_of(a)],
        "form": round(j.form) if j.form is not None else None,
        # Newest first, and only a few: the panel is a glance, not a file.
        "tasks": [
            {"title": t, "quality": round(q), "tired": tired}
            for t, q, tired in reversed(j.tasks[-3:])
        ],
        "review": (
            {"verdict": j.reviews[-1].verdict, "by": j.reviews[-1].reviewer,
             "comment": j.reviews[-1].comment, "expected": j.reviews[-1].expected}
            if j.reviews else None
        ),
        "reviewDue": j.review_due_tick >= 0,
        "seconded": sim.seconded(a),
        "ready": sim.ready_for_promotion(a),
        "warnings": live_warnings(j),
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
                "loan": round(a.loan, 2),
                "employed": a.employed,
                # None when out of work, like study: a blank rather than a row
                # of zeroes that reads like a bad record.
                "work": _work(sim, a) if a.job is not None else None,
                "applying": (
                    {
                        "role": a.application.posting.role.title,
                        "employer": a.application.posting.employer_name,
                    }
                    if a.application is not None
                    else None
                ),
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