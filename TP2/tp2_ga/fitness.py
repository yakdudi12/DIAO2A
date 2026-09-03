"""Función de aptitud y evaluación batch CPU/CUDA."""

from __future__ import annotations

import warnings

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity

from .render import (
    cp,
    cuda_available,
    cuda_device_name,
    render_population_cuda,
    render_triangles,
)

if cp is not None:
    try:
        from cupyx.scipy import ndimage as cuda_ndimage
    except (ImportError, OSError):
        cuda_ndimage = None
else:
    cuda_ndimage = None


def gradient(image: np.ndarray) -> np.ndarray:
    return np.hypot(ndimage.sobel(image, axis=0), ndimage.sobel(image, axis=1))


def combined_fitness(
    target: Image.Image,
    reconstruction: Image.Image,
    edge_weight: float = 0.15,
    ssim_weight: float = 0.15,
) -> float:
    """Error a minimizar: MSE de color, bordes y SSIM."""
    target_rgb = np.asarray(target.convert("RGB"), dtype=np.float32) / 255.0
    reconstructed_rgb = np.asarray(reconstruction.convert("RGB"), dtype=np.float32) / 255.0
    color_error = np.mean((target_rgb - reconstructed_rgb) ** 2)

    target_gray = np.asarray(target.convert("L"), dtype=np.float32) / 255.0
    reconstructed_gray = np.asarray(reconstruction.convert("L"), dtype=np.float32) / 255.0
    target_gradient = gradient(target_gray)
    reconstructed_gradient = gradient(reconstructed_gray)
    scale = max(target_gradient.max(), reconstructed_gradient.max(), 1e-8)
    edge_error = np.mean(((target_gradient - reconstructed_gradient) / scale) ** 2)

    minimum_side = min(target_gray.shape)
    window = min(7, minimum_side if minimum_side % 2 else minimum_side - 1)
    if window < 3:
        ssim_error = 1.0
    else:
        ssim_error = 1.0 - structural_similarity(
            target_gray,
            reconstructed_gray,
            data_range=1.0,
            win_size=window,
        )
    return float(color_error + edge_weight * edge_error + ssim_weight * ssim_error)


class FitnessEvaluator:
    """Mantiene objetivos redimensionados y términos constantes en caché."""

    def __init__(
        self,
        target: Image.Image,
        triangle_count: int,
        *,
        use_gpu: bool = True,
        edge_weight: float = 0.15,
        ssim_weight: float = 0.15,
    ):
        self.target = target.convert("RGB")
        self.triangle_count = triangle_count
        self.width, self.height = target.size
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.cpu_cache: dict[tuple[int, int], Image.Image] = {}
        self.gpu_cache = {}
        self.gpu_active = bool(use_gpu and cuda_available() and cuda_ndimage is not None)
        self.backend = f"CUDA ({cuda_device_name()})" if self.gpu_active else "CPU"

    def dimensions(self, size: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(size, tuple):
            if len(size) != 2 or min(size) <= 0:
                raise ValueError("size debe contener dos dimensiones positivas")
            return tuple(map(int, size))
        if size <= 0:
            raise ValueError("size debe ser positivo")
        scale = size / max(self.width, self.height)
        return max(1, round(self.width * scale)), max(1, round(self.height * scale))

    def target_at(self, size: int | tuple[int, int]) -> Image.Image:
        dimensions = self.dimensions(size)
        if dimensions not in self.cpu_cache:
            self.cpu_cache[dimensions] = self.target.resize(
                dimensions, Image.Resampling.LANCZOS
            )
        return self.cpu_cache[dimensions]

    def _target_cuda(self, dimensions):
        if dimensions not in self.gpu_cache:
            target = self.target_at(dimensions)
            rgb = cp.asarray(np.asarray(target, dtype=np.float32) / 255.0)
            gray = cp.asarray(np.asarray(target.convert("L"), dtype=np.float32) / 255.0)
            gradient_x = cuda_ndimage.sobel(gray, axis=0)
            gradient_y = cuda_ndimage.sobel(gray, axis=1)
            target_gradient = cp.hypot(gradient_x, gradient_y)

            minimum_side = min(gray.shape)
            window = min(7, minimum_side if minimum_side % 2 else minimum_side - 1)
            mean_x = variance_x = None
            if window >= 3:
                x = gray[None, ...]
                filter_size = (1, window, window)
                mean_x = cuda_ndimage.uniform_filter(x, size=filter_size)
                covariance_norm = window**2 / (window**2 - 1)
                variance_x = covariance_norm * (
                    cuda_ndimage.uniform_filter(x * x, size=filter_size) - mean_x * mean_x
                )
            self.gpu_cache[dimensions] = (
                rgb,
                gray,
                target_gradient,
                target_gradient.max(),
                window,
                mean_x,
                variance_x,
            )
        return self.gpu_cache[dimensions]

    def _evaluate_cuda(self, population, dimensions) -> list[float]:
        rendered = render_population_cuda(
            np.asarray(population), self.triangle_count, dimensions
        )
        target_rgb, target_gray, target_gradient, target_max, window, mean_x, variance_x = (
            self._target_cuda(dimensions)
        )
        color_error = cp.mean(
            (rendered - target_rgb[None, ...]) ** 2, axis=(1, 2, 3)
        )

        reconstructed_gray = cp.rint(
            (
                0.299 * rendered[..., 0]
                + 0.587 * rendered[..., 1]
                + 0.114 * rendered[..., 2]
            )
            * 255.0
        ) / 255.0
        gradient_x = cuda_ndimage.sobel(reconstructed_gray, axis=1)
        gradient_y = cuda_ndimage.sobel(reconstructed_gray, axis=2)
        reconstructed_gradient = cp.hypot(gradient_x, gradient_y)
        scale = cp.maximum(
            cp.maximum(target_max, reconstructed_gradient.max(axis=(1, 2))), 1e-8
        )
        edge_error = cp.mean(
            (
                (reconstructed_gradient - target_gradient[None, ...])
                / scale[:, None, None]
            )
            ** 2,
            axis=(1, 2),
        )

        if window < 3:
            ssim_error = cp.ones(len(population), dtype=cp.float32)
        else:
            x = target_gray[None, ...]
            y = reconstructed_gray
            filter_size = (1, window, window)
            mean_y = cuda_ndimage.uniform_filter(y, size=filter_size)
            covariance_norm = window**2 / (window**2 - 1)
            variance_y = covariance_norm * (
                cuda_ndimage.uniform_filter(y * y, size=filter_size) - mean_y * mean_y
            )
            covariance_xy = covariance_norm * (
                cuda_ndimage.uniform_filter(x * y, size=filter_size) - mean_x * mean_y
            )
            c1, c2 = 0.01**2, 0.03**2
            ssim_map = ((2 * mean_x * mean_y + c1) * (2 * covariance_xy + c2)) / (
                (mean_x * mean_x + mean_y * mean_y + c1)
                * (variance_x + variance_y + c2)
            )
            padding = (window - 1) // 2
            ssim_error = 1.0 - ssim_map[
                :, padding:-padding, padding:-padding
            ].mean(axis=(1, 2))

        total = (
            color_error
            + self.edge_weight * edge_error
            + self.ssim_weight * ssim_error
        )
        return cp.asnumpy(total).astype(float).tolist()

    def evaluate_many(self, population, size: int | tuple[int, int]) -> list[float]:
        if len(population) == 0:
            return []
        dimensions = self.dimensions(size)
        if self.gpu_active:
            try:
                return self._evaluate_cuda(population, dimensions)
            except Exception as error:
                warnings.warn(f"CUDA falló ({error}); se continúa en CPU.")
                self.gpu_active = False
                self.backend = "CPU"
                self.gpu_cache.clear()

        target = self.target_at(dimensions)
        results = []
        for genes in population:
            triangles = genes.reshape(self.triangle_count, 10)
            phenotype = [(row[:6], row[6:]) for row in triangles]
            reconstruction = render_triangles(phenotype, dimensions)
            results.append(
                combined_fitness(
                    target,
                    reconstruction,
                    edge_weight=self.edge_weight,
                    ssim_weight=self.ssim_weight,
                )
            )
        return results

    def evaluate(self, individual, size: int | tuple[int, int]) -> float:
        return self.evaluate_many([individual], size)[0]
