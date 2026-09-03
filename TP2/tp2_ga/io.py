"""Entrada y salida del TP2."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_target_image(path: str | Path, max_side: int = 120) -> tuple[Image.Image, Image.Image]:
    """Carga una imagen y devuelve `(original, reducida)` preservando proporción."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {image_path.resolve()}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato no compatible: {image_path.suffix}")
    if max_side <= 0:
        raise ValueError("max_side debe ser positivo")

    original = Image.open(image_path).convert("RGB")
    scale = max_side / max(original.size)
    size = tuple(max(1, round(dimension * scale)) for dimension in original.size)
    return original, original.resize(size, Image.Resampling.LANCZOS)


def export_triangles(triangles, path: str | Path) -> Path:
    """Exporta todos los triángulos a JSON, no sólo una muestra."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "triangle": index,
            "vertices": [
                {"x": float(points[0]), "y": float(points[1])},
                {"x": float(points[2]), "y": float(points[3])},
                {"x": float(points[4]), "y": float(points[5])},
            ],
            "color_rgba": {
                "r": float(color[0]),
                "g": float(color[1]),
                "b": float(color[2]),
                "a": float(color[3]),
            },
        }
        for index, (points, color) in enumerate(triangles, start=1)
    ]
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output
