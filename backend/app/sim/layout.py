"""Deterministic city generator.

Roads are carved on a 10-tile pitch, leaving a grid of 9x9 blocks (8 columns
x 6 rows). Landmarks are hand-placed rather than procedurally scattered: a
city you can recognise is worth more than one that is merely varied.
"""

from __future__ import annotations

from app.config import CONFIG
from app.sim.world import Building, BuildingKind, TileKind, World

#: Road pitch. Blocks are (BLOCK - 1) square, offset one tile off each road.
BLOCK = 10

#: (col, row, kind, id, name, width, height, capacity)
LANDMARKS: list[tuple] = [
    (3, 0, BuildingKind.UNIVERSITY, "university", "Agent City University", 9, 8, 60),
    (1, 2, BuildingKind.OFFICE, "nimbus", "Nimbus Labs", 7, 6, 20),
    (5, 2, BuildingKind.OFFICE, "vertex", "Vertex Systems", 7, 6, 20),
    (3, 3, BuildingKind.OFFICE, "harbor", "Harbor Analytics", 7, 6, 20),
    (2, 1, BuildingKind.CAFE, "grind", "The Grind", 5, 4, 12),
    (6, 4, BuildingKind.CAFE, "corner_cup", "Corner Cup", 5, 4, 12),
    (4, 1, BuildingKind.PARK, "commons", "The Commons", 9, 9, -1),
    (0, 3, BuildingKind.GYM, "iron_works", "Iron Works", 5, 5, 15),
]

#: Residential blocks. 10 buildings x 6 beds = 60 slots for 50 agents.
HOMES: list[tuple[int, int, str]] = [
    (0, 0, "Maple Court"),
    (1, 0, "Cedar Flats"),
    (6, 0, "Birch House"),
    (7, 0, "Willow Court"),
    (0, 1, "Aspen Flats"),
    (7, 2, "Riverside Apartments"),
    (1, 4, "Linden House"),
    (7, 4, "Juniper Court"),
    (3, 5, "Sycamore Flats"),
    (5, 5, "Hollow Rise"),
]

HOME_CAPACITY = 6


def _carve_roads(world: World) -> None:
    for y in range(world.height):
        for x in range(world.width):
            if x % BLOCK == 0 or y % BLOCK == 0:
                world.set_tile(x, y, TileKind.ROAD)


def _place(
    world: World,
    col: int,
    row: int,
    kind: BuildingKind,
    bid: str,
    name: str,
    bw: int,
    bh: int,
    capacity: int,
) -> Building:
    """Drop a building at the top-left of block (col, row).

    The door sits mid-way along the north edge; because a block starts one tile
    below a road, the tile above the door is always street.
    """
    bx, by = col * BLOCK + 1, row * BLOCK + 1
    if bw > BLOCK - 1 or bh > BLOCK - 1:
        raise ValueError(f"{bid}: {bw}x{bh} does not fit a {BLOCK - 1}-tile block")
    door = (bx + bw // 2, by)
    return world.add_building(
        Building(bid, kind, name, bx, by, bw, bh, door=door, capacity=capacity)
    )


def build_city() -> World:
    world = World(CONFIG.world.width, CONFIG.world.height)
    _carve_roads(world)

    for col, row, kind, bid, name, bw, bh, cap in LANDMARKS:
        _place(world, col, row, kind, bid, name, bw, bh, cap)

    for i, (col, row, name) in enumerate(HOMES):
        _place(world, col, row, BuildingKind.HOME, f"home_{i}", name, 7, 6, HOME_CAPACITY)

    return world


_GLYPH = {
    BuildingKind.UNIVERSITY: "U",
    BuildingKind.OFFICE: "O",
    BuildingKind.CAFE: "C",
    BuildingKind.PARK: "P",
    BuildingKind.GYM: "G",
    BuildingKind.HOME: "h",
}


def ascii_map(world: World) -> str:
    """Render the city as text. The only way to see it until Phase 1 exists."""
    rows = []
    for y in range(world.height):
        row = []
        for x in range(world.width):
            tile = world.tile(x, y)
            if tile is TileKind.WALL:
                row.append("#")
            elif tile is TileKind.ROAD:
                row.append(" ")
            else:
                b = world.building_at(x, y)
                if b is not None:
                    row.append(_GLYPH.get(b.kind, "?"))
                else:
                    row.append("." if tile is TileKind.GRASS else "·")
        rows.append("".join(row))
    return "\n".join(rows)