"""Motor de algoritmos genéticos para aproximar imágenes con triángulos."""

from .engine import GAConfig, GeneticImageGA, RunResult
from .benchmark import (
    BenchmarkResult,
    PicsumImage,
    benchmark_profile,
    fetch_picsum_images,
    normalize_image,
    run_benchmark,
)
from .io import export_triangles, load_target_image
from .render import render_high_resolution, render_triangles

__all__ = [
    "GAConfig",
    "GeneticImageGA",
    "RunResult",
    "BenchmarkResult",
    "PicsumImage",
    "benchmark_profile",
    "export_triangles",
    "fetch_picsum_images",
    "load_target_image",
    "normalize_image",
    "render_high_resolution",
    "render_triangles",
    "run_benchmark",
]
