from dataclasses import dataclass
from triangles import Triangle


@dataclass
class Individual:
    triangles: list[Triangle]
    fitness: float | None = None
