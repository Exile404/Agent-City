"""Vectors for memory relevance.

Batched by design: one embed costs ~209ms of round trip, a batch of 32 costs
90ms in total — 5/sec against 346/sec. The model is small, so almost all of that
209ms is HTTP and scheduling rather than compute.

Falls back to a lexical bag-of-words vector when the model isn't pulled, so the
city still runs without it — just with dumber retrieval.
"""

from __future__ import annotations

import re
import zlib

import httpx

from app.config import CONFIG

#: Dimensions of the fallback vector. Small on purpose — a ranking hint, not a
#: semantic embedding.
LEXICAL_DIM = 128

_TOKEN = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from had has have he her his i in is it its"
    " of on or she that the their them they this to was were will with you your".split()
)


def lexical_vector(text: str) -> list[float]:
    """Cheap bag-of-words vector, L2-normalised.

    crc32 rather than hash(): Python randomises string hashing per process, so
    hash() would give a different vector every run and break replay.
    """
    vec = [0.0] * LEXICAL_DIM
    for token in _TOKEN.findall(text.lower()):
        if token in _STOPWORDS or len(token) < 3:
            continue
        vec[zlib.crc32(token.encode()) % LEXICAL_DIM] += 1.0

    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec


def cosine(a: list[float], b: list[float]) -> float:
    """Similarity of two unit vectors, or 0.0 if they aren't comparable.

    The dimension guard matters: lexical vectors are 128-wide and embeddings are
    768, and a freshly added memory carries a lexical vector until the next batch
    upgrades it. Without this, zip() would silently compare the first 128
    dimensions of an embedding against a bag of words and return nonsense.
    """
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class Embedder:
    """Batched embeddings with a lexical fallback."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(base_url=CONFIG.llm.base_url, timeout=60.0)
        self._model = CONFIG.llm.embed_model
        #: None until probed. False means the model isn't pulled and every
        #: vector stays lexical for the life of the process.
        self.available: bool | None = None
        #: Queries repeat constantly ("what should I do now?"), and re-embedding
        #: the same string is pure waste.
        self._cache: dict[str, list[float]] = {}

    @property
    def dim(self) -> int:
        return 768 if self.available else LEXICAL_DIM

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Vectors for a batch. One request, however many texts."""
        if not texts:
            return []
        if self.available is False:
            return [lexical_vector(t) for t in texts]

        try:
            response = await self._http.post(
                "/api/embed", json={"model": self._model, "input": texts}
            )
            response.raise_for_status()
            vectors = response.json()["embeddings"]
            self.available = True
        except Exception:
            # Probe failure is permanent: if the model is missing it will stay
            # missing, and retrying every batch would just add latency.
            self.available = False
            return [lexical_vector(t) for t in texts]

        return [_unit(v) for v in vectors]

    async def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = (await self.embed_many([text]))[0]
        if len(self._cache) < 512:
            self._cache[text] = vector
        return vector

    async def aclose(self) -> None:
        await self._http.aclose()


def _unit(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec