"""Deterministic city generator.

Roads are carved on a 10-tile pitch, leaving a grid of 9x9 blocks (8 columns
x 6 rows). Landmarks are hand-placed rather than procedurally scattered: a
city you can recognise is worth more than one that is merely varied.
"""

from __future__ import annotations

from app.config import CONFIG
from app.sim.world import Building, BuildingKind, TileKind, World

#: Road pitch and carriageway width. Blocks are (BLOCK - ROAD_WIDTH) square.
#: Three-tile roads leave room for two lanes plus the kerb, which is what makes
#: cars and traffic lights legible instead of a single-file queue.
BLOCK = 10
ROAD_WIDTH = 2

#: (col, row, kind, id, name, width, height, capacity)
LANDMARKS: list[tuple] = [
    (2, 0, BuildingKind.UNIVERSITY, "university", "Agent City University", 8, 8, 60),
    # Second campus, deliberately across the map from the first so Phase 4
    # students are not all funnelled into the same corner of the city.
    (6, 2, BuildingKind.UNIVERSITY, "eastgate", "Eastgate Polytechnic", 7, 7, 45),
    (4, 0, BuildingKind.PARK, "commons", "The Commons", 8, 8, -1),
    (2, 4, BuildingKind.PARK, "willow_green", "Willow Green", 8, 8, -1),
    # Cafes and gyms are deliberately scattered one per district. A single gym
    # on a 70-tile map meant nearest() was often across town, which is what made
    # journeys average 42 tiles and forced cars in the first place.
    (1, 0, BuildingKind.CAFE, "grind", "The Grind", 5, 4, 12),
    (2, 2, BuildingKind.CAFE, "corner_cup", "Corner Cup", 5, 4, 12),
    (5, 4, BuildingKind.CAFE, "night_owl", "Night Owl", 5, 4, 12),
    (6, 0, BuildingKind.GYM, "iron_works", "Iron Works", 5, 4, 15),
    (0, 2, BuildingKind.GYM, "pulse", "Pulse Athletic", 5, 4, 15),
    (6, 3, BuildingKind.GYM, "grid_fitness", "Grid Fitness", 5, 4, 15),
    (0, 1, BuildingKind.OFFICE, "nimbus", "Nimbus Labs", 6, 5, 20),
    (5, 1, BuildingKind.OFFICE, "vertex", "Vertex Systems", 6, 5, 20),
    (4, 2, BuildingKind.OFFICE, "harbor", "Harbor Analytics", 6, 5, 20),
    (1, 3, BuildingKind.OFFICE, "atlas", "Atlas Foundry", 6, 5, 20),
    (4, 4, BuildingKind.OFFICE, "solstice", "Solstice Media", 6, 5, 20),
    (3, 1, BuildingKind.MARKET, "foundry", "Foundry Market", 6, 5, 18),
    (3, 2, BuildingKind.HOSPITAL, "st_adler", "St. Adler General", 6, 5, 25),
    (3, 3, BuildingKind.LIBRARY, "central_library", "Central Library", 5, 5, 20),
    (5, 3, BuildingKind.BANK, "ledger", "Ledger Bank", 5, 4, 14),
    (0, 4, BuildingKind.POWER, "kestrel", "Kestrel Power Station", 6, 6, 15),
    (6, 4, BuildingKind.GAS, "meridian", "Meridian Gas Works", 6, 5, 12),
]

#: Residential blocks. 10 buildings x 6 beds = 60 slots for 50 agents.
HOMES: list[tuple[int, int, str]] = [
    (0, 0, "Maple Court"),
    (3, 0, "Cedar Flats"),
    (5, 0, "Birch House"),
    (1, 1, "Willow Court"),
    (2, 1, "Aspen Flats"),
    (4, 1, "Riverside Apartments"),
    (6, 1, "Linden House"),
    (1, 2, "Juniper Court"),
    (5, 2, "Sycamore Flats"),
    (0, 3, "Hollow Rise"),
    (2, 3, "Quarry Lofts"),
    (4, 3, "Thornhill Court"),
    (1, 4, "Ashgrove Flats"),
    (3, 4, "Beacon House"),
]

HOME_CAPACITY = 4


def _carve_roads(world: World) -> None:
    for y in range(world.height):
        for x in range(world.width):
            if x % BLOCK < ROAD_WIDTH or y % BLOCK < ROAD_WIDTH:
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

    The door sits mid-way along the north edge; because a block starts
    ROAD_WIDTH tiles below a road band, the tile above the door is always
    street.
    """
    bx, by = col * BLOCK + ROAD_WIDTH, row * BLOCK + ROAD_WIDTH
    lot = BLOCK - ROAD_WIDTH
    if bw > lot or bh > lot:
        raise ValueError(f"{bid}: {bw}x{bh} does not fit a {lot}-tile block")
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
        _place(world, col, row, BuildingKind.HOME, f"home_{i}", name, 6, 5, HOME_CAPACITY)

    return world


_GLYPH = {
    BuildingKind.UNIVERSITY: "U",
    BuildingKind.OFFICE: "O",
    BuildingKind.CAFE: "C",
    BuildingKind.PARK: "P",
    BuildingKind.GYM: "G",
    BuildingKind.HOME: "h",
    BuildingKind.POWER: "E",
    BuildingKind.GAS: "F",
    BuildingKind.HOSPITAL: "+",
    BuildingKind.MARKET: "M",
    BuildingKind.LIBRARY: "L",
    BuildingKind.BANK: "B",
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