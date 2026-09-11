"""The memory stream.

Agents accumulate observations, dialogue and reflections, and retrieve the few
most worth thinking about right now. Scoring combines recency, importance and
relevance — weights in CONFIG.memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.cognition.embed import cosine, lexical_vector
from app.config import CONFIG

#: Importance by memory kind, 1-10. Assigned rather than asked for: scoring each
#: memory with a model call would cost ~1,000 calls a day against a ~190/day
#: smart budget — more than the whole dialogue allowance, to rate things like
#: "walked to a cafe".
DEFAULT_IMPORTANCE: dict[str, float] = {
    "observation": 2.0,
    "plan": 3.0,
    "dialogue": 4.0,
    "reflection": 6.0,
    #: Exams passed, jobs offered, promotions. Phases 4-6 pass these explicitly.
    "milestone": 8.0,
}


@dataclass(slots=True)
class MemoryNode:
    tick: int
    #: observation | plan | dialogue | reflection | milestone
    kind: str
    text: str
    importance: float
    vector: list[float]
    #: Recency decays from here, not from creation, so a memory an agent keeps
    #: returning to stays fresh while one-off trivia fades.
    last_access: int

    def __str__(self) -> str:
        return f"[{self.kind} i={self.importance:.0f}] {self.text}"


def _minmax(values: list[float]) -> list[float]:
    """Rescale to [0,1] across the candidates actually in contention.

    Without this the weights are decorative. Recency and importance naturally
    reach 1.0, while cosine similarity rarely passes 0.6 — so an on-topic memory
    could never outrank a high-importance one no matter how relevant it was.
    A component where every candidate ties contributes nothing, which is right:
    it cannot separate them.
    """
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-9:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]


@dataclass
class MemoryStream:
    nodes: list[MemoryNode] = field(default_factory=list)
    #: Importance accrued since the last reflection. Crossing the configured
    #: threshold triggers the next one.
    since_reflection: float = 0.0
    #: Nodes still carrying a lexical vector, awaiting the next embed batch.
    pending: list[MemoryNode] = field(default_factory=list)

    def add(self, tick: int, kind: str, text: str, importance: float | None = None) -> MemoryNode:
        """Record a memory immediately, with a placeholder vector.

        The lexical vector means retrieval works the instant this returns; the
        next batch upgrades it to a real embedding.
        """
        weight = DEFAULT_IMPORTANCE.get(kind, 2.0) if importance is None else importance
        node = MemoryNode(
            tick=tick,
            kind=kind,
            text=text,
            importance=weight,
            vector=lexical_vector(text),
            last_access=tick,
        )
        self.nodes.append(node)
        self.pending.append(node)
        self.since_reflection += weight
        self._evict()
        return node

    def take_pending(self) -> list[MemoryNode]:
        """Hand over nodes awaiting embedding, clearing the queue."""
        batch, self.pending = self.pending, []
        return batch

    def retrieve(self, query_vector: list[float], tick: int, k: int | None = None) -> list[MemoryNode]:
        """The k memories most worth surfacing right now.

        Takes a vector rather than text so the caller can embed off the tick
        path — retrieval belongs in the async thinking pipeline, not in the
        synchronous simulation step.
        """
        if not self.nodes:
            return []

        cfg = CONFIG.memory
        k = cfg.retrieval_k if k is None else k
        minutes = CONFIG.world.minutes_per_tick

        # Relevance gates the candidate set; recency and importance only order
        # what is already on topic. Scoring all three across the whole stream let
        # a recent-but-irrelevant memory win, because min-max stretches whatever
        # spread exists — so a signal that barely varies gets amplified to full
        # range and its noise outvotes a signal that actually discriminates.
        all_relevance = [cosine(query_vector, n.vector) for n in self.nodes]
        shortlist = sorted(range(len(self.nodes)), key=lambda i: -all_relevance[i])[: k * 3]
        all_relevance = [cosine(query_vector, n.vector) for n in self.nodes]
        shortlist = sorted(range(len(self.nodes)), key=lambda i: -all_relevance[i])[: k * 3]

        # Return fewer memories rather than worse ones. Relative to the best
        # match, not absolute — cosine scales differ per embedding model, and a
        # fixed threshold would need retuning if the model ever changes.
        best = all_relevance[shortlist[0]]
        floor = best * CONFIG.memory.relevance_floor
        shortlist = [i for i in shortlist if all_relevance[i] >= floor]
        recency: list[float] = []
        importance: list[float] = []
        relevance: list[float] = []
        for i in shortlist:
            node = self.nodes[i]
            hours = (tick - node.last_access) * minutes / 60.0
            recency.append(0.5 ** (hours / cfg.recency_half_life_hours))
            importance.append(node.importance)
            relevance.append(all_relevance[i])

        rec, imp, rel = _minmax(recency), _minmax(importance), _minmax(relevance)
        scored = [
            (
                rec[j] * cfg.w_recency + imp[j] * cfg.w_importance + rel[j] * cfg.w_relevance,
                self.nodes[shortlist[j]],
            )
            for j in range(len(shortlist))
        ]
        scored.sort(key=lambda pair: -pair[0])

        top = [node for _, node in scored[:k]]
        for node in top:
            # Halve the age rather than zeroing it. Thinking about something does
            # make it feel more recent, but a full reset pins every retrieved
            # memory at maximum recency — so the same handful wins every
            # subsequent query and the agent's mind freezes on whatever it
            # happened to think about first.
            node.last_access = (node.last_access + tick) // 2
        return top

    def should_reflect(self) -> bool:
        return self.since_reflection >= CONFIG.memory.reflection_importance_threshold

    def mark_reflected(self) -> None:
        self.since_reflection = 0.0

    def _evict(self) -> None:
        """Cap the stream, dropping the least important of the oldest half.

        Confining eviction to the older half means a burst of trivia can never
        push out something that happened moments ago.
        """
        excess = len(self.nodes) - CONFIG.memory.max_memories
        if excess <= 0:
            return
        half = len(self.nodes) // 2
        weakest = sorted(self.nodes[:half], key=lambda n: n.importance)[:excess]
        doomed = {id(n) for n in weakest}
        self.nodes = [n for n in self.nodes if id(n) not in doomed]
        self.pending = [n for n in self.pending if id(n) not in doomed]