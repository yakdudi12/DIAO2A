"""Carga de los códigos QR como patrones ±1 y lectura con un decodificador real (zxing-cpp)."""

from pathlib import Path

import numpy as np
import zxingcpp
from PIL import Image

MENUS = {
    "food": "Menú principal",
    "wine": "Vinos",
    "beverages": "Cócteles y bebidas",
    "dessert": "Postres",
    "specials": "Especiales del día",
}


def find_qr_dir(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for base in (start, *start.parents):
        for candidate in (base / "data" / "hopfield", base / "TP4" / "data" / "hopfield"):
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError("No se encontró data/hopfield")


def to_pattern(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Escala de grises -> matriz ±1 con negro = +1 y blanco = −1."""
    return np.where(np.asarray(image) < threshold, 1.0, -1.0)


def load_codes(directory: Path | None = None, size: int = 37) -> dict[str, np.ndarray]:
    """Un patrón (size, size) por menú, en el orden de `MENUS`."""
    directory = find_qr_dir() if directory is None else Path(directory)
    codes = {}
    for key, name in MENUS.items():
        image = Image.open(directory / f"{key}_menu_qr.png").convert("L")
        if image.size != (size, size):
            image = image.resize((size, size), Image.NEAREST)
        codes[name] = to_pattern(np.array(image))
    return codes


def to_image(pattern: np.ndarray, scale: int = 8, quiet_zone: int = 4) -> np.ndarray:
    """Imagen en escala de grises lista para un lector: agrega el margen blanco que exige el
    estándar (4 módulos) y amplía cada módulo a `scale` píxeles."""
    pattern = np.asarray(pattern)
    side = int(round(np.sqrt(pattern.size)))
    image = np.where(pattern.reshape(side, side) > 0, 0, 255).astype(np.uint8)
    image = np.pad(image, quiet_zone, constant_values=255)
    return image.repeat(scale, axis=0).repeat(scale, axis=1)


def decode(pattern: np.ndarray) -> str | None:
    """Texto del QR, o None si el lector no puede decodificarlo."""
    results = zxingcpp.read_barcodes(to_image(pattern), formats=zxingcpp.BarcodeFormat.QRCode)
    return results[0].text if results else None


def error_correction_level(pattern: np.ndarray) -> str | None:
    results = zxingcpp.read_barcodes(to_image(pattern), formats=zxingcpp.BarcodeFormat.QRCode)
    return results[0].ec_level if results else None
