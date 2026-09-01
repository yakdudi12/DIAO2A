from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw

from triangles import Triangle


# un Individuo es una solución completa al problema
class Individual:
    def __init__(self, triangles):
        self.triangles = triangles

        self.fitness = None
        self.error = None

    @classmethod
    def random(cls, num_triangles, width, height):
        """
        Genera un individuo formado por N triangulos aleatorios
        """
        triangles = [
            Triangle.random(width, height)
            for _ in range(num_triangles)
        ]

        return cls(triangles)

    def clone(self):
        """
        Copia profunda del individuo
        """
        return deepcopy(self)

    def render(self, width, height, background=(255, 255, 255, 255)):
        """
        Renderiza todos los triangulos sobre un canvas
        """
        canvas = Image.new(
            mode = "RGBA",
            size = (width, height),
            color = background
        )

        for triangle in self.triangles:
            layer = Image.new(
                mode = "RGBA",
                size = (width, height),
                color = (0,0,0,0)
            )

            draw = ImageDraw.Draw(layer)

            draw.polygon(
                triangle.points(),
                fill = triangle.color()
            )

            canvas = Image.alpha_composite(canvas, layer)

        return canvas

    def evaluate(self, target_array, width, height):
        """
        Calcula el MSE entre la imagen producida y la imagen objetivo.
        """
        render_image = self.render(width, height)

        candidate = np.asarray(
            render_image.convert("RGB"),
            dtype=np.float32
        )

        difference = target_array - candidate

        mse = np.mean(difference ** 2)

        self.error = mse

        # Fitness positivo: mayor = mejor
        self.fitness = 1.0 / (1.0 + mse)

        return self.fitness
