"""Motor de algoritmos genéticos para aproximar imágenes con triángulos."""

from .engine import GAConfig, GeneticImageGA, RunResult
from .io import export_triangles, load_target_image
from .render import render_high_resolution, render_triangles

__all__ = [
    "GAConfig",
    "GeneticImageGA",
    "RunResult",
    "export_triangles",
    "load_target_image",
    "render_high_resolution",
    "render_triangles",
]
