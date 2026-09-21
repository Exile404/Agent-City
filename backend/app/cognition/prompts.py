"""Prompt construction and reply validation.

Both halves live here deliberately: the prompt defines a contract about what the
model may say, and the validator enforces it. Split them across files and they
drift apart the first time the vocabulary changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.actions import ActionKind

#: What an agent may choose to do. The model picks from this and nothing else —
#: a 3B model will happily invent "search_for_restaurant" otherwise.
PLANNABLE: dict[str, tuple[str, ...]] = {
    "sleep": ("home",),
    "eat": ("cafe", "market"),
    "study": ("university", "library"),
    "work": ("office", "hospital", "market", "power", "gas", "bank", "library"),
    "socialize": ("cafe", "park"),
    "exercise": ("gym", "park"),
}

SYSTEM = (
    "You are a resident of a small city, planning your own day. "
    "Answer only with JSON. Be concise and concrete. "
    "Never invent activities or places outside the lists you are given."
)


@dataclass(slots=True)
class PlanStep:
    #: Minutes past midnight. Comparable straight against Clock.minute_of_day.
    at: int
    kind: ActionKind
    #: Building *kind*, not id — the sim resolves it to the nearest one, so a
    #: plan stays valid even if the agent is across town when the step arrives.
    place: str
    why: str

    def __str__(self) -> str:
        return f"{self.at // 60:02d}:{self.at % 60:02d} {self.kind.value} @{self.place}"


def _vocabulary() -> str:
    return "\n".join(f'  "{do}" at one of {list(places)}' for do, places in PLANNABLE.items())


def daily_plan(
    *,
    name: str,
    age: int,
    traits: list[str],
    home: str,
    clock: str,
    needs: dict[str, float],
    memories: list[str],
) -> str:
    """Ask an agent to plan the rest of their day."""
    needs_line = ", ".join(f"{k} {v:.0f}/100" for k, v in needs.items())
    recalled = "\n".join(f"  - {m}" for m in memories) or "  - (nothing in particular)"

    return f"""You are {name}, {age}, living at {home}. You are {", ".join(traits)}.
It is {clock}.

How you feel right now (100 is fully satisfied, under 20 is urgent):
  {needs_line}

What is on your mind:
{recalled}

Plan the rest of your day as 5 to 8 steps, in time order, starting after the
current time and ending with sleep at home. Choose only from these activities
and places:
{_vocabulary()}

Reply with JSON exactly like this:
{{"steps": [{{"at": "08:30", "do": "eat", "where": "cafe", "why": "short reason"}}]}}"""


def parse_plan(data: dict | None, after_minute: int) -> list[PlanStep]:
    """Turn a model reply into steps the simulation can execute.

    Never raises. Malformed steps are dropped; an empty result means "no plan",
    and the caller falls back to Tier 0. A bad generation must not be able to
    stall an agent.
    """
    if not isinstance(data, dict):
        return []
    raw = data.get("steps")
    if not isinstance(raw, list):
        return []

    steps: list[PlanStep] = []
    for entry in raw[:8]:
        if not isinstance(entry, dict):
            continue

        do = str(entry.get("do", "")).strip().lower()
        places = PLANNABLE.get(do)
        if places is None:
            continue  # invented an activity

        where = str(entry.get("where", "")).strip().lower()
        if where not in places:
            # Named a real activity in an impossible place ("sleep at the gym").
            # The first legal place is a better guess than discarding the step.
            where = places[0]

        at = _minutes(entry.get("at"))
        if at is None or at < after_minute:
            continue

        why = " ".join(str(entry.get("why", "")).split())[:120]
        steps.append(PlanStep(at=at, kind=ActionKind(do), place=where, why=why))

    steps.sort(key=lambda s: s.at)

    # Collapse duplicate times: two steps at 09:00 would have the second
    # silently overwrite the first the moment the clock reached it.
    deduped: list[PlanStep] = []
    for step in steps:
        if deduped and step.at - deduped[-1].at < 30:
            continue
        deduped.append(step)
    return deduped


def _minutes(value: object) -> int | None:
    """Parse "HH:MM" into minutes past midnight, tolerantly."""
    if isinstance(value, (int, float)):
        return int(value) % 1440
    if not isinstance(value, str):
        return None
    text = value.strip()
    if ":" not in text:
        return None
    hh, _, mm = text.partition(":")
    try:
        hours, minutes = int(hh), int(mm[:2])
    except ValueError:
        return None
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        return None
    return hours * 60 + minutes

CONVERSATION_SYSTEM = (
    "You write short, natural exchanges between two residents of a small city. "
    "Answer only with JSON. Keep every line under twenty words. "
    "People talk about what is actually on their minds — work, money, study, "
    "food, each other — not about the weather. "
    "Keep everything suitable for a general audience: no sexual remarks, "
    "slurs, or crude language about anyone."
)

#: Whole-word match only — substring matching turns "class" and "assess" into
#: false positives. A conversation tripping this is discarded entirely: the Tier
#: 0 greeting has already happened, so nothing is lost except the words.
_BLOCKED = re.compile(
    r"\b(tits|boobs|ass|arse|dick|cock|pussy|fuck\w*|shit|bitch|whore|slut|"
    r"horny|sexy|naked|rape)\b",
    re.IGNORECASE,
)


def is_publishable(lines: list[tuple[str, str]]) -> bool:
    """Whether an exchange is safe to put in a feed anyone might watch."""
    return not any(_BLOCKED.search(says) for _, says in lines)


def conversation(
    *,
    a_name: str,
    a_traits: list[str],
    a_memories: list[str],
    b_name: str,
    b_traits: list[str],
    b_memories: list[str],
    place: str,
    doing: str,
    relation: str,
    clock: str,
) -> str:
    """Generate a whole exchange in one call.

    One generation per conversation rather than one per turn: a six-turn dialogue
    would cost six slots against a budget of 0.27 per tick.
    """
    a_recall = "; ".join(a_memories) or "nothing in particular"
    b_recall = "; ".join(b_memories) or "nothing in particular"

    return f"""{a_name} and {b_name} are both at {place}, {doing}. It is {clock}.
To {a_name}, {b_name} is {relation}.

{a_name} is {", ".join(a_traits)}. On their mind: {a_recall}
{b_name} is {", ".join(b_traits)}. On their mind: {b_recall}

Write what they say to each other — 2 to 4 short lines, alternating, starting
with {a_name}. Then rate how the exchange went from -3 (hostile) to 3 (warm).

Reply with JSON exactly like this:
{{"lines": [{{"who": "{a_name}", "says": "..."}}, {{"who": "{b_name}", "says": "..."}}], "warmth": 1}}"""


def parse_conversation(
    data: dict | None, a_name: str, b_name: str
) -> tuple[list[tuple[str, str]], int]:
    """Return (lines, warmth). Empty lines mean the generation was unusable.

    Never raises. As with plans, a bad reply degrades to nothing happening rather
    than to an exception in the tick loop.
    """
    if not isinstance(data, dict):
        return [], 0

    lines: list[tuple[str, str]] = []
    for entry in (data.get("lines") or [])[:6]:
        if not isinstance(entry, dict):
            continue
        who = str(entry.get("who", "")).strip()
        says = " ".join(str(entry.get("says", "")).split())[:200]
        if not says:
            continue
        # Models drift between "Maya", "Maya Silva" and "A". Match on the first
        # name and default to whoever did not speak last, which keeps the
        # exchange alternating even when the label is wrong.
        if who.split()[:1] == a_name.split()[:1]:
            speaker = a_name
        elif who.split()[:1] == b_name.split()[:1]:
            speaker = b_name
        else:
            speaker = b_name if (lines and lines[-1][0] == a_name) else a_name
        lines.append((speaker, says))

    warmth = data.get("warmth", 0)
    try:
        warmth = int(float(warmth))
    except (TypeError, ValueError):
        warmth = 0
    return lines, max(-3, min(3, warmth))

EXAM_SYSTEM = (
    "You set and mark exams at a city university. "
    "Reply only with JSON, always carrying all four keys: question, answer, "
    "mark, comment. "
    "The answer is the student's own script — first person, their words, and "
    "nothing but what they wrote on the paper. "
    "Write it the way someone of their stated ability actually would — a weak or "
    "absent student gives a visibly weak answer, and you mark it accordingly. "
    "Keep everything suitable for a general audience."
)

#: Required shape for an exam reply. "format: json" alone lets the model close
#: the object after two keys; the mark is the one the sim cannot recover.
EXAM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "mark": {"type": "number"},
        "comment": {"type": "string"},
    },
    "required": ["question", "answer", "mark", "comment"],
}


def exam(
    *,
    name: str,
    course: str,
    campus: str,
    skill_name: str,
    skill: float,
    attended: int,
    offered: int,
    attempt: int,
    expected: float,
) -> str:
    """One question, one answer, one mark, in a single generation.

    The expected mark is handed to the model, not withheld: its job is to write
    an exchange that reads like that number, not to invent an outcome. It sits
    in the JSON example as well as the prose — a 3B copies the shape it is
    shown far more reliably than it follows an instruction.
    """
    sat = "first sitting" if attempt == 1 else f"attempt {attempt}"
    return f"""{name} is sitting the final exam for {course} at {campus} ({sat}).

Their {skill_name} ability is {skill:.0f} out of 100.
They attended {attended} of {offered} classes this term.
A student in their position would typically score about {expected:.0f} out of 100.

Write one exam question for this course, at most two sentences. It tests the
subject itself — never the course, the exam, or the student's experience of
either.

Then write what {name} puts on the paper: their words, first person, as someone
with {skill_name} at {skill:.0f}/100 who attended {attended} of {offered} classes
would really write it. A few sentences at most, answering the question and
nothing else: never the exam itself, never their attendance, never what they did
or did not study. A weak student writes a weak answer to the question — they do
not explain why it is weak.

Then mark the answer out of 100 and add a one-line examiner's comment.

Reply with JSON holding all four keys — question, answer, mark, comment —
exactly like this:
{{"question": "...", "answer": "I think ...", "mark": {expected:.0f}, "comment": "..."}}"""


def _clip(text: object, limit: int) -> str:
    """Collapse whitespace, then cut to length on a word boundary."""
    s = " ".join(str(text).split())
    if len(s) <= limit:
        return s
    head = s[:limit].rsplit(" ", 1)[0] or s[:limit]
    return head + "\u2026"


def parse_exam(data: dict | None) -> tuple[str, str, float | None, str]:
    """Return (question, answer, mark, comment). A mark of None means unusable;
    never raises, so a bad generation costs flavour rather than a result."""
    if not isinstance(data, dict):
        return "", "", None, ""
    question = _clip(data.get("question", ""), 300)
    answer = _clip(data.get("answer", ""), 400)
    comment = _clip(data.get("comment", ""), 200)
    try:
        mark = float(data.get("mark"))
    except (TypeError, ValueError):
        return question, answer, None, comment
    if not 0.0 <= mark <= 100.0:
        return question, answer, None, comment
    return question, answer, mark, comment


INTERVIEW_SYSTEM = (
    "You run hiring interviews at a city employer. "
    "Reply only with JSON, always carrying all four keys: question, answer, "
    "verdict, reason. "
    "The answer is the candidate's own words — first person, and nothing but "
    "what they said in the room. "
    "Write it the way someone of their stated ability actually would: a weak "
    "candidate gives a visibly weak answer, and you turn them down for it. "
    "Keep everything suitable for a general audience."
)

#: The verdict is an enum, so constrained decoding cannot produce "maybe" or a
#: sentence. Without it a 3B writes "hire, if they improve" and the parse falls
#: back to the simulation's own guess, which is the whole feature lost.
INTERVIEW_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "verdict": {"type": "string", "enum": ["hire", "reject"]},
        "reason": {"type": "string"},
    },
    "required": ["question", "answer", "verdict", "reason"],
}


def interview(
    *,
    name: str,
    traits: list[str],
    role: str,
    employer: str,
    skill_name: str,
    skill: float,
    requires: float,
    standing: float,
    credentials: list[str],
) -> str:
    """One question, one answer, one verdict, in a single generation.

    No expected outcome is handed over, unlike the exam: the simulation decided
    who was worth interviewing, and the decision in the room is the model's.
    What it gets is the facts — what the work needs, what the candidate has —
    and an instruction to judge the answer rather than the numbers.
    """
    held = (
        f"They hold: {', '.join(credentials)}."
        if credentials
        else "They have no formal qualification."
    )
    # Never "47 out of 100": that reads as a failing grade whatever the job
    # asks for, and it turned the model into an examiner who rejected 16 of 18
    # candidates the simulation had already found qualified. The raw scale
    # still belongs in the answer instruction below, where it is what makes a
    # weak candidate write a weak answer.
    clear = (
        "comfortably clear of what the job asks"
        if standing - requires >= 15
        else "above what the job asks"
        if standing - requires >= 5
        else "just clear of what the job asks"
    )
    return f"""{name} is interviewing for {role} at {employer}. They are {", ".join(traits)}.

This job asks for {skill_name} of {requires:.0f}. {name}'s is {standing:.0f} —
{clear}. {held}

Everyone who gets this far already clears the bar on paper. What you are
judging is the answer they give, not the file.

Write one interview question for this job, at most two sentences. It asks them
to do the work — a problem to solve, a judgement to make, a thing to explain.
Never their experience, their background, their previous jobs, the company, or
how they feel about any of it.

Then write what {name} actually says: their words, first person, as someone
with {skill_name} at {skill:.0f}/100 would really answer it. A few sentences at
most, answering the question and nothing else — never the job, never their own
qualifications, never their past employment, never why they want to work here.

Then decide. "hire" if the answer shows they can do this work, "reject" if it
does not. Judge the answer you have just written, not the numbers above. Add a
one-line reason.

Reply with JSON holding all four keys — question, answer, verdict, reason —
exactly like one of these:
{{"question": "...", "answer": "I would ...", "verdict": "hire", "reason": "..."}}
{{"question": "...", "answer": "I am not sure ...", "verdict": "reject", "reason": "..."}}"""


def parse_interview(data: dict | None) -> tuple[str, str, bool | None, str]:
    """Return (question, answer, hired, reason). hired=None means unusable;
    never raises, so a bad generation costs the transcript, not the hire."""
    if not isinstance(data, dict):
        return "", "", None, ""
    question = _clip(data.get("question", ""), 300)
    answer = _clip(data.get("answer", ""), 400)
    reason = _clip(data.get("reason", ""), 200)
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict.startswith("hire"):
        return question, answer, True, reason
    if verdict.startswith("reject"):
        return question, answer, False, reason
    return question, answer, None, reason
