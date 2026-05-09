from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PlanetState:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass(slots=True)
class FleetState:
    id: int
    owner: int
    source: int
    x: float
    y: float
    vx: float
    vy: float
    ships: int


@dataclass(slots=True)
class GameState:
    step: int
    player: int
    planets: list[PlanetState]
    fleets: list[FleetState]


def parse_observation(obs: Any) -> GameState:
    planets = []
    for idx, row in enumerate(_get(obs, "planets", [])):
        if isinstance(row, dict):
            planets.append(
                PlanetState(
                    id=int(row.get("id", idx)),
                    owner=int(row.get("owner", -1)),
                    x=float(row.get("x", 0.0)),
                    y=float(row.get("y", 0.0)),
                    radius=float(row.get("radius", 5.0)),
                    ships=int(row.get("ships", 0)),
                    production=int(row.get("prod", row.get("production", 0))),
                )
            )
        else:
            planets.append(
                PlanetState(
                    id=int(row[0]),
                    owner=int(row[1]),
                    x=float(row[2]),
                    y=float(row[3]),
                    radius=float(row[4]) if len(row) > 6 else 5.0,
                    ships=int(row[5] if len(row) > 6 else row[3]),
                    production=int(row[6] if len(row) > 6 else row[4]),
                )
            )

    fleets = []
    for idx, row in enumerate(_get(obs, "fleets", [])):
        if isinstance(row, dict):
            fleets.append(
                FleetState(
                    id=int(row.get("id", idx)),
                    owner=int(row.get("owner", -1)),
                    source=int(row.get("source", row.get("from_planet_id", -1))),
                    x=float(row.get("x", 0.0)),
                    y=float(row.get("y", 0.0)),
                    vx=float(row.get("vx", 0.0)),
                    vy=float(row.get("vy", 0.0)),
                    ships=int(row.get("ships", 0)),
                )
            )

    return GameState(
        step=int(_get(obs, "step", 0)),
        player=int(_get(obs, "player", 0)),
        planets=planets,
        fleets=fleets,
    )


def _get(obs: Any, key: str, default: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)