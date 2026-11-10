"""Carga de datos del TP5.

* ``cargar_font3``      parsea ``font.h`` (32 patrones 7x5) -> matriz (32, 35) en {0,1}.
* ``generar_simbolos``  dataset propio de símbolos matemáticos 16x16 rasterizados de varias
                        tipografías (las DejaVu viajan con matplotlib: reproducible en cualquier equipo).
* ``ruido_sal_pimienta`` / ``ruido_gauss`` corrupciones de entrada para el Denoising AE.

Las librerías (re, PIL, matplotlib) sólo se usan para parsear/rasterizar; ningún
algoritmo de aprendizaje depende de ellas.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
CARACTERES = list("`abcdefghijklmnopqrstuvwxyz{|}~") + ["DEL"]  # 0x60 ... 0x7f


def cargar_font3(ruta: Path | None = None):
    """Devuelve X (32,35) con píxeles 0/1 (fila-mayor, 7 filas x 5 columnas) y las etiquetas."""
    ruta = ruta or AQUI / "data" / "font.h"
    txt = ruta.read_text(encoding="utf-8")
    filas = re.findall(r"\{((?:0x[0-9a-fA-F]{2},?\s*){7})\}", txt)
    X = np.zeros((len(filas), 35))
    for i, f in enumerate(filas):
        for r, h in enumerate(re.findall(r"0x([0-9a-fA-F]{2})", f)):
            v = int(h, 16)
            for c in range(5):
                X[i, r * 5 + c] = (v >> (4 - c)) & 1
    etiquetas = re.findall(r"//\s*0x[0-9a-f]{2},\s*(.*)", txt)
    return X, [e.strip() for e in etiquetas]


def ruido_sal_pimienta(p):
    """Devuelve una función que invierte cada píxel con probabilidad p (ruido binario simétrico)."""
    def f(X, rng):
        flip = rng.random(X.shape) < p
        return np.where(flip, 1.0 - X, X)
    return f


def ruido_gauss(sigma):
    def f(X, rng):
        return np.clip(X + sigma * rng.standard_normal(X.shape), 0, 1)
    return f


# --------------------------------------------------------------------------- símbolos matemáticos
SIMBOLOS = list("+−×÷=≠≈≤≥<>±∑∏∫∂∇√∞πθλμσΔΩαβ∈∪∩⊂∀∃∅→⊕⊗∝")
ESTILOS = ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSerif.ttf",
           "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"]
LADO = 16


def _rasterizar(fuente, simbolo, lado=LADO, ocupacion=0.78):
    """Dibuja el glifo grande, recorta su caja y lo escala (manteniendo proporción) a ``ocupacion*lado``."""
    from PIL import Image, ImageDraw
    G = 256
    img = Image.new("L", (G, G), 0)
    ImageDraw.Draw(img).text((G // 4, G // 4), simbolo, fill=255, font=fuente)
    bb = img.getbbox()
    if bb is None:
        return np.zeros((lado, lado))
    g = img.crop(bb)
    k = ocupacion * lado / max(g.size)
    w, h = max(1, round(g.size[0] * k)), max(1, round(g.size[1] * k))
    g = g.resize((w, h), Image.BOX)
    lienzo = Image.new("L", (lado, lado), 0)
    lienzo.paste(g, ((lado - w) // 2, (lado - h) // 2))
    return np.asarray(lienzo, dtype=float) / 255.0


def generar_simbolos(lado=LADO, tam_letra=160, desplazamientos=(-1, 0, 1)):
    """Dataset: |SIMBOLOS| x |ESTILOS| x |desplazamientos|^2 imágenes en [0,1].

    Devuelve X (N, lado*lado), etiqueta (N,), estilo (N,), indice_desplazamiento (N,).
    Se normaliza el tamaño del glifo al 80 % del lado para que ningún símbolo quede cortado.
    """
    import matplotlib
    from PIL import ImageFont
    carpeta = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    X, lab, sty, dsp = [], [], [], []
    for j, est in enumerate(ESTILOS):
        f = ImageFont.truetype(str(carpeta / est), tam_letra)
        for i, s in enumerate(SIMBOLOS):
            base = _rasterizar(f, s, lado)
            m = base.max()
            base = base / m if m > 0 else base
            k = 0
            for dy in desplazamientos:
                for dx in desplazamientos:
                    X.append(np.roll(np.roll(base, dy, 0), dx, 1).ravel())
                    lab.append(i); sty.append(j); dsp.append(k); k += 1
    return np.array(X), np.array(lab), np.array(sty), np.array(dsp)
