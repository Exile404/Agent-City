"""Simulation time. One tick advances CONFIG.world.minutes_per_tick minutes."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG

MINUTES_PER_DAY = 24 * 60
TICKS_PER_DAY = MINUTES_PER_DAY // CONFIG.world.minutes_per_tick
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass
class Clock:
    """Wall-clock for the city. Tick 0 is day 0 at CONFIG.world.start_hour."""

    tick: int = 0

    @property
    def total_minutes(self) -> int:
        return CONFIG.world.start_hour * 60 + self.tick * CONFIG.world.minutes_per_tick

    @property
    def day(self) -> int:
        return self.total_minutes // MINUTES_PER_DAY

    @property
    def minute_of_day(self) -> int:
        return self.total_minutes % MINUTES_PER_DAY

    @property
    def hour(self) -> int:
        return self.minute_of_day // 60

    @property
    def minute(self) -> int:
        return self.minute_of_day % 60

    @property
    def weekday(self) -> int:
        return self.day % 7

    @property
    def is_weekend(self) -> bool:
        return self.weekday >= 5

    def advance(self) -> None:
        self.tick += 1

    def hours_since(self, tick: int) -> float:
        """Sim-hours elapsed since `tick`. Drives needs decay and memory recency."""
        return (self.tick - tick) * CONFIG.world.minutes_per_tick / 60.0

    def ticks_until(self, hour: int, minute: int = 0) -> int:
        """Ticks until the next occurrence of a wall-clock time.

        Lets an agent schedule "be at the lecture hall by 09:00" without
        caring what tick number that is.
        """
        target = hour * 60 + minute
        delta = target - self.minute_of_day
        if delta <= 0:
            delta += MINUTES_PER_DAY
        return max(1, round(delta / CONFIG.world.minutes_per_tick))

    def __str__(self) -> str:
        return f"D{self.day} {DAY_NAMES[self.weekday]} {self.hour:02d}:{self.minute:02d}"