"""A* over the tile grid.

Agents path to a single goal tile — usually a building's door. Movement is
4-directional: no diagonals, which keeps agents on the streets and sidesteps
the corner-cutting bug where a diagonal step slips between two wall corners.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict

from app.sim.world import World

Tile = tuple[int, int]

#: North/south/east/west only.
_STEPS: tuple[Tile, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))

#: Cost of the cheapest possible tile. The heuristic must never over-estimate
#: the true remaining cost or A* stops returning shortest paths, so Manhattan
#: distance is scaled by this rather than by the average tile cost.
_MIN_COST = 1.0


def find_path(world: World, start: Tile, goal: Tile, max_nodes: int = 8000) -> list[Tile] | None:
    """Cheapest route from `start` to `goal`, excluding the start tile.

    Returns [] if already there, or None if unreachable within `max_nodes`.
    """
    if start == goal:
        return []
    if not world.walkable(*goal):
        return None

    gx, gy = goal

    def heuristic(x: int, y: int) -> float:
        return (abs(x - gx) + abs(y - gy)) * _MIN_COST

    open_heap: list[tuple[float, float, Tile]] = [(heuristic(*start), 0.0, start)]
    came_from: dict[Tile, Tile] = {}
    best: dict[Tile, float] = {start: 0.0}
    expanded = 0

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal:
            return _rebuild(came_from, current)

        # heapq has no decrease-key, so an improved node was pushed as a second
        # entry. This drops the obsolete copy when it surfaces.
        if g > best.get(current, float("inf")):
            continue

        expanded += 1
        if expanded > max_nodes:
            return None

        cx, cy = current
        for dx, dy in _STEPS:
            nxt = (cx + dx, cy + dy)
            if not world.walkable(*nxt):
                continue
            # Cost is charged for the tile being entered — this is what makes
            # agents prefer roads without any explicit "prefer roads" rule.
            tentative = g + world.cost(*nxt)
            if tentative < best.get(nxt, float("inf")):
                best[nxt] = tentative
                came_from[nxt] = current
                heapq.heappush(open_heap, (tentative + heuristic(*nxt), tentative, nxt))

    return None


def _rebuild(came_from: dict[Tile, Tile], node: Tile) -> list[Tile]:
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path[1:]  # drop the start tile; path[0] is the agent's next step


class PathCache:
    """Memoised routes.

    The city never changes shape, so a route stays valid forever. Fifty agents
    walking the same home-to-office commute each morning collapses into a dict
    lookup instead of fifty searches.
    """

    def __init__(self, world: World, capacity: int = 2048) -> None:
        self.world = world
        self.capacity = capacity
        self._store: OrderedDict[tuple[Tile, Tile], list[Tile] | None] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def path(self, start: Tile, goal: Tile) -> list[Tile] | None:
        key = (start, goal)
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            cached = self._store[key]
        else:
            self.misses += 1
            cached = find_path(self.world, start, goal)
            self._store[key] = cached
            if len(self._store) > self.capacity:
                self._store.popitem(last=False)

        # Hand back a copy: callers pop steps off as they walk, and mutating
        # the cached list would corrupt the route for everyone after them.
        return None if cached is None else list(cached)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0