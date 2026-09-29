"""Ledger Bank lends, and the rules are numbers.

Debt used to be a number that meant nothing: balances went as negative as they
liked. Now an overdrawn balance costs more each day than a loan does, so going
to the bank is a real choice, and someone who puts it off pays for it. Who puts
it off falls out of their traits, like whether they turn up to class.

The simulation owns every figure here: how much anyone may borrow, at what
rate, and how much of a wage goes back. No model is asked.
"""

from __future__ import annotations

from random import Random

from app.config import CONFIG

#: Daily interest on what is owed to the bank.
LOAN_RATE = 0.002
#: Flat daily fee on an overdrawn balance nobody arranged. Flat, not a rate: a
#: rate compounds, and 1% a day turned a $2,500 hole into $78,000 in a year.
#: Interest on any loan up to $2,500 costs less a day than this, so the walk pays.
OVERDRAFT_FEE = 5.0
#: Overdrawn by a week of living: past this, a missed shift is not something
#: they can afford, and temperament stops keeping them home.
DESPERATE_DAYS = 7
#: Owed past this many days of living, loan and overdraft together, and the
#: debt is written off: bankruptcy.
BANKRUPTCY_DAYS = 30
#: After a bankruptcy, the bank lends nothing for this many days.
BANKRUPTCY_BAN_DAYS = 90
#: Share of every wage the bank takes until the loan is repaid.
REPAYMENT_SHARE = 0.20
#: Credit, in days of what someone earns: a job's daily rate, or the stipend.
CREDIT_DAYS = 10
#: A loan clears the overdraft and covers this many days of living on top.
CUSHION_DAYS = 3
#: Rent and three meals: what a day costs anyone.
LIVING_COST = CONFIG.economy.daily_rent + 3 * CONFIG.economy.meal_cost
#: Bank hours, 24h, weekdays only — the bank's own staff work weekdays.
OPEN_HOURS = (9, 17)

#: How readily each trait walks into a bank. Caution and patience put it off;
#: ambition and restlessness do not.
BORROWING: dict[str, float] = {
    "ambitious": 0.20,
    "restless": 0.15,
    "gregarious": 0.05,
    "curious": 0.05,
    "warm": 0.0,
    "blunt": 0.0,
    "analytical": 0.0,
    "patient": -0.10,
    "stubborn": -0.15,
    "cautious": -0.25,
}
#: Even-handed without traits: an overdrawn agent goes about every other day.
BASE_BORROWING = 0.5


def daily_income(daily_wage: float | None) -> float:
    """What the bank reckons someone earns a day: their job's daily rate, or
    the stipend if they have none."""
    return daily_wage if daily_wage is not None else CONFIG.economy.unemployment_stipend


def credit_limit(daily_wage: float | None) -> float:
    return CREDIT_DAYS * daily_income(daily_wage)


def willing_to_borrow(agent_id: str, day: int, traits: list[str]) -> bool:
    """Whether an overdrawn agent goes to the bank today.

    Seeded on (agent, day) like class attendance, so the same run repeats the
    same reluctance. Takes primitives rather than an Agent, as the university's
    diligence() does.
    """
    chance = BASE_BORROWING + sum(BORROWING.get(t, 0.0) for t in traits)
    return Random(f"{agent_id}:{day}:bank").random() < max(0.1, min(0.95, chance))
