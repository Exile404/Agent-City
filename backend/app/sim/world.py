"""The city's physical layer.

A flat tile grid with buildings stamped onto it. Every layer above this one —
pathing, actions, institutions — asks the World only two questions:
"can I stand here?" and "what is here?"
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, IntEnum
from random import Random


class TileKind(IntEnum):
    GRASS = 0
    ROAD = 1
    FLOOR = 2
    WALL = 3


#: Pathing cost per tile. Roads are cheapest, so agents visibly prefer streets
#: over cutting across grass — free realism straight out of the A* weights.
TILE_COST: dict[TileKind, float] = {
    TileKind.GRASS: 1.6,
    TileKind.ROAD: 1.0,
    TileKind.FLOOR: 1.0,
}


class BuildingKind(str, Enum):
    HOME = "home"
    UNIVERSITY = "university"
    OFFICE = "office"
    CAFE = "cafe"
    PARK = "park"
    GYM = "gym"
    #: Utilities and services. They exist to be employers, not scenery — a job
    #: market with three software firms has no texture to it.
    POWER = "power"
    GAS = "gas"
    HOSPITAL = "hospital"
    MARKET = "market"
    LIBRARY = "library"
    BANK = "bank"


@dataclass
class Building:
    id: str
    kind: BuildingKind
    name: str
    x: int
    y: int
    w: int
    h: int
    #: The single tile agents enter through. Pathing targets this, never the
    #: centre — otherwise agents route into walls and stall.
    door: tuple[int, int]
    #: Agents allowed inside at once; -1 means unlimited.
    capacity: int = -1

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def tiles(self) -> Iterator[tuple[int, int]]:
        for yy in range(self.y, self.y + self.h):
            for xx in range(self.x, self.x + self.w):
                yield xx, yy


class World:
    """Grid of tiles plus the buildings occupying them."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.tiles = bytearray([TileKind.GRASS] * (width * height))
        #: Building id owning each tile, parallel to `tiles`.
        self._owner: list[str | None] = [None] * (width * height)
        self.buildings: dict[str, Building] = {}

    def _i(self, x: int, y: int) -> int:
        return y * self.width + x

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def tile(self, x: int, y: int) -> TileKind:
        return TileKind(self.tiles[self._i(x, y)])

    def set_tile(self, x: int, y: int, kind: TileKind) -> None:
        self.tiles[self._i(x, y)] = kind

    def walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.tiles[self._i(x, y)] != TileKind.WALL

    def cost(self, x: int, y: int) -> float:
        return TILE_COST.get(self.tile(x, y), 1.0)

    def add_building(self, b: Building) -> Building:
        """Stamp a building: walled perimeter, floor inside, door punched through.

        Parks are the exception — open grass, no walls, wander in anywhere.
        """
        for xx, yy in b.tiles():
            if b.kind is BuildingKind.PARK:
                self.set_tile(xx, yy, TileKind.GRASS)
            else:
                on_edge = xx in (b.x, b.x + b.w - 1) or yy in (b.y, b.y + b.h - 1)
                self.set_tile(xx, yy, TileKind.WALL if on_edge else TileKind.FLOOR)
            self._owner[self._i(xx, yy)] = b.id

        self.set_tile(b.door[0], b.door[1], TileKind.FLOOR)
        self.buildings[b.id] = b
        return b

    def building_at(self, x: int, y: int) -> Building | None:
        if not self.in_bounds(x, y):
            return None
        bid = self._owner[self._i(x, y)]
        return self.buildings[bid] if bid else None

    def of_kind(self, kind: BuildingKind) -> list[Building]:
        return [b for b in self.buildings.values() if b.kind is kind]

    def nearest(self, kind: BuildingKind, frm: tuple[int, int]) -> Building | None:
        options = self.of_kind(kind)
        if not options:
            return None
        return min(options, key=lambda b: abs(b.door[0] - frm[0]) + abs(b.door[1] - frm[1]))

    def random_road(self, rng: Random) -> tuple[int, int]:
        """A random street tile — used to scatter agents on day 0."""
        while True:
            x, y = rng.randrange(self.width), rng.randrange(self.height)
            if self.tile(x, y) is TileKind.ROAD:
                return x, y