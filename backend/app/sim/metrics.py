"""What the city looks like on a given morning, as plain numbers.

One place computes them, so the drift check, /metrics and the dashboard all
count the same things the same way.
"""

from __future__ import annotations

from statistics import median
from typing import Any


def city_metrics(sim: Any) -> dict[str, float | None]:
    agents = sim.agents
    employed = sum(a.job is not None for a in agents)
    students = sum(a.job is None and a.enrollment is not None for a in agents)
    return {
        "day": sim.clock.day,
        "employed": employed,
        "students": students,
        "looking": len(agents) - employed - students,
        # None until the first shift is called: 0 of 0 is no attendance yet, not
        # 0% of it, and plotting it as 0% draws a false climb on day one.
        "attendance": round(sim.shifts_worked / sim.shifts_offered, 3) if sim.shifts_offered else None,
        "hires": sim.hires,
        "rejections": sim.rejections,
        "let go": sim.firings,
        "promotions": sim.promotions,
        "sponsored": sim.sponsored,
        "dismissals": sim.dismissals,
        "retirements": sim.retirements,
        "reviews": sum(sim.review_verdicts.values()),
        "warnings": sim.review_verdicts["warn"],
        "credentials": sum(len(a.credentials) for a in agents),
        "tired share": round(sim.tired_tasks / sim.tasks_done, 3) if sim.tasks_done else None,
        "median money": round(median(a.money for a in agents), 1),
        # One direction of a relationship at affinity 20 or more. Conversations
        # are the live city's main lever on this, so it should drift most.
        "warm ties": sum(r.affinity >= 20 for a in agents for r in a.relationships.values()),
        # Money trouble: who is overdrawn, who owes the bank, who took the floor.
        "overdrawn": sum(a.money < 0 for a in agents),
        "borrowers": sum(a.loan > 0 for a in agents),
        "owed to bank": round(sum(a.loan for a in agents), 1),
        "porters": sum(a.job is not None and a.job.role_id == "porter" for a in agents),
        "scholarship paid": round(sim.scholarship_paid, 1),
    }
