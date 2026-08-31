from PIL import Image, ImageDraw
from individual import Individual


def render(individual: Individual, width: int, height: int, background_color: str):
    image = Image.new("RGBA", (width, height), background_color)

    for triangle in individual.triangles:
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        points = [
            (triangle.x1, triangle.y1),
            (triangle.x2, triangle.y2),
            (triangle.x3, triangle.y3)
        ]

        # RGBA
        color = (
            triangle.r,
            triangle.g,
            triangle.b,
            triangle.alpha
        )

        draw.polygon(points, color)

        image = Image.alpha_composite(image, overlay)

    return image