from dataclasses import dataclass
import random


@dataclass
class Triangle:
    # Verices
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int

    # RGBA
    r: int
    g: int
    b: int
    a: int

    @classmethod
    def random(cls, width: int, height: int):
        """
        Crea un triangulo aleatorio
        """
        return cls(
            x1=random.randint(0, width - 1),
            y1=random.randint(0, height - 1),

            x2=random.randint(0, width - 1),
            y2=random.randint(0, height - 1),

            x3=random.randint(0, width - 1),
            y3=random.randint(0, height - 1),

            r=random.randint(0, 255),
            g=random.randint(0, 255),
            b=random.randint(0, 255),
            a=random.randint(30, 255),
        )

    def points(self):
        """
        Devuelve los tres vértices en el formato esperado por Pillow
        """
        return [
            (self.x1, self.y1),
            (self.x2, self.y2),
            (self.x3, self.y3),
        ]

    def color(self):
        """
        Color RGBA del triángulo
        """
        return self.r, self.g, self.b, self.a

    def to_dict(self):
        return {
            "points": self.points(),
            "color": self.color()
        }
