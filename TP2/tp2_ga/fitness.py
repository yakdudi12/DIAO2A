"""Fitness perceptual multiescala y evaluación batch CPU/CUDA."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity

from .render import cp, cuda_available, cuda_device_name, render_population_cuda, render_triangles

if cp is not None:
    try:
        from cupyx.scipy import ndimage as cuda_ndimage
    except (ImportError, OSError):
        cuda_ndimage = None
else:
    cuda_ndimage = None


@dataclass(frozen=True, slots=True)
class FitnessComponents:
    color: float
    edge: float
    ssim: float
    total: float


def gradient(image: np.ndarray) -> np.ndarray:
    return np.hypot(ndimage.sobel(image, axis=0), ndimage.sobel(image, axis=1))


def _detail_map(gray: np.ndarray, detail_boost: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradiente y pesos fijos que realzan bordes/textura del objetivo."""
    target_gradient = gradient(gray)
    normalized_gradient = np.clip(
        target_gradient / max(float(target_gradient.max()), 1e-8), 0.0, 1.0
    )
    local_mean = ndimage.uniform_filter(gray, size=5, mode="reflect")
    local_variance = np.maximum(
        ndimage.uniform_filter(gray * gray, size=5, mode="reflect") - local_mean**2,
        0.0,
    )
    local_std = np.sqrt(local_variance)
    normalized_std = np.clip(
        local_std / max(float(np.quantile(local_std, 0.95)), 1e-8), 0.0, 1.0
    )
    weights = 1.0 + detail_boost * (0.65 * normalized_gradient + 0.35 * normalized_std)
    weights /= weights.mean()
    return target_gradient, weights.astype(np.float32)


def _single_scale_components(
    target: Image.Image,
    reconstruction: Image.Image,
    *,
    color_weight: float,
    edge_weight: float,
    ssim_weight: float,
    detail_boost: float,
) -> FitnessComponents:
    target_rgb = np.asarray(target.convert("RGB"), dtype=np.float32) / 255.0
    reconstructed_rgb = np.asarray(reconstruction.convert("RGB"), dtype=np.float32) / 255.0
    target_gray = np.asarray(target.convert("L"), dtype=np.float32) / 255.0
    reconstructed_gray = np.asarray(reconstruction.convert("L"), dtype=np.float32) / 255.0
    target_gradient, detail_weights = _detail_map(target_gray, detail_boost)

    # L1 mantiene una señal útil cuando el error residual ya es pequeño.
    color_error = np.mean(
        np.abs(target_rgb - reconstructed_rgb) * detail_weights[..., None]
    )
    reconstructed_gradient = gradient(reconstructed_gray)
    # Escala fija derivada sólo del objetivo: un borde artificial no reduce su penalidad.
    gradient_scale = max(float(target_gradient.max()), 0.25)
    edge_error = np.mean(
        np.abs(target_gradient - reconstructed_gradient) / gradient_scale
    )
    minimum_side = min(target_gray.shape)
    window = min(7, minimum_side if minimum_side % 2 else minimum_side - 1)
    ssim_error = 1.0 if window < 3 else 1.0 - structural_similarity(
        target_gray, reconstructed_gray, data_range=1.0, win_size=window
    )
    total = color_weight * color_error + edge_weight * edge_error + ssim_weight * ssim_error
    return FitnessComponents(float(color_error), float(edge_error), float(ssim_error), float(total))


def combined_fitness(
    target: Image.Image,
    reconstruction: Image.Image,
    color_weight: float = 0.55,
    edge_weight: float = 0.30,
    ssim_weight: float = 0.15,
    detail_boost: float = 4.0,
) -> float:
    """Error perceptual a una escala; API simple para análisis externo."""
    return _single_scale_components(
        target,
        reconstruction,
        color_weight=color_weight,
        edge_weight=edge_weight,
        ssim_weight=ssim_weight,
        detail_boost=detail_boost,
    ).total


class FitnessEvaluator:
    """Evalúa color ponderado, bordes y SSIM en una pirámide de resoluciones."""

    def __init__(
        self,
        target: Image.Image,
        triangle_count: int,
        *,
        use_gpu: bool = True,
        color_weight: float = 0.55,
        edge_weight: float = 0.30,
        ssim_weight: float = 0.15,
        detail_boost: float = 4.0,
        pyramid_scales: tuple[float, ...] = (0.5, 1.0),
        pyramid_weights: tuple[float, ...] = (0.25, 0.75),
    ):
        self.target = target.convert("RGB")
        self.triangle_count = triangle_count
        self.width, self.height = target.size
        self.color_weight = color_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.detail_boost = detail_boost
        self.pyramid_scales = tuple(map(float, pyramid_scales))
        weights = np.asarray(pyramid_weights, dtype=float)
        self.pyramid_weights = tuple((weights / weights.sum()).tolist())
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

    def pyramid_dimensions(self, size: int | tuple[int, int]) -> list[tuple[int, int]]:
        full = self.dimensions(size)
        return [
            (max(7, round(full[0] * scale)), max(7, round(full[1] * scale)))
            for scale in self.pyramid_scales
        ]

    def target_at(self, dimensions: tuple[int, int]) -> Image.Image:
        if dimensions not in self.cpu_cache:
            self.cpu_cache[dimensions] = self.target.resize(dimensions, Image.Resampling.LANCZOS)
        return self.cpu_cache[dimensions]

    def _target_cuda(self, dimensions):
        if dimensions not in self.gpu_cache:
            target = self.target_at(dimensions)
            rgb = cp.asarray(np.asarray(target, dtype=np.float32) / 255.0)
            gray_np = np.asarray(target.convert("L"), dtype=np.float32) / 255.0
            target_gradient_np, detail_weights_np = _detail_map(gray_np, self.detail_boost)
            gray = cp.asarray(gray_np)
            target_gradient = cp.asarray(target_gradient_np)
            detail_weights = cp.asarray(detail_weights_np)
            minimum_side = min(gray.shape)
            window = min(7, minimum_side if minimum_side % 2 else minimum_side - 1)
            mean_x = variance_x = None
            if window >= 3:
                x = gray[None, ...]
                filter_size = (1, window, window)
                mean_x = cuda_ndimage.uniform_filter(x, size=filter_size, mode="reflect")
                covariance_norm = window**2 / (window**2 - 1)
                variance_x = covariance_norm * (
                    cuda_ndimage.uniform_filter(x * x, size=filter_size, mode="reflect")
                    - mean_x * mean_x
                )
            self.gpu_cache[dimensions] = (
                rgb, gray, target_gradient,
                max(float(target_gradient_np.max()), 0.25),
                detail_weights, window, mean_x, variance_x,
            )
        return self.gpu_cache[dimensions]

    def _evaluate_cuda_scale(self, population, dimensions):
        rendered = render_population_cuda(np.asarray(population), self.triangle_count, dimensions)
        target_rgb, target_gray, target_gradient, gradient_scale, detail_weights, window, mean_x, variance_x = self._target_cuda(dimensions)
        color_error = cp.mean(
            cp.abs(rendered - target_rgb[None, ...]) * detail_weights[None, ..., None],
            axis=(1, 2, 3),
        )
        reconstructed_gray = cp.rint((
            0.299 * rendered[..., 0] + 0.587 * rendered[..., 1] + 0.114 * rendered[..., 2]
        ) * 255.0) / 255.0
        # ``sobel`` sobre (batch, alto, ancho) también suaviza el eje batch y
        # multiplica artificialmente el gradiente. Los kernels 1x3x3 lo aíslan.
        sobel_y = cp.asarray([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=cp.float32)
        sobel_x = cp.asarray([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=cp.float32)
        gradient_x = cuda_ndimage.convolve(reconstructed_gray, sobel_x, mode="reflect")
        gradient_y = cuda_ndimage.convolve(reconstructed_gray, sobel_y, mode="reflect")
        reconstructed_gradient = cp.hypot(gradient_x, gradient_y)
        edge_error = cp.mean(
            cp.abs(reconstructed_gradient - target_gradient[None, ...]) / gradient_scale,
            axis=(1, 2),
        )
        if window < 3:
            ssim_error = cp.ones(len(population), dtype=cp.float32)
        else:
            x = target_gray[None, ...]
            y = reconstructed_gray
            filter_size = (1, window, window)
            mean_y = cuda_ndimage.uniform_filter(y, size=filter_size, mode="reflect")
            covariance_norm = window**2 / (window**2 - 1)
            variance_y = covariance_norm * (
                cuda_ndimage.uniform_filter(y * y, size=filter_size, mode="reflect") - mean_y * mean_y
            )
            covariance_xy = covariance_norm * (
                cuda_ndimage.uniform_filter(x * y, size=filter_size, mode="reflect") - mean_x * mean_y
            )
            c1, c2 = 0.01**2, 0.03**2
            ssim_map = ((2 * mean_x * mean_y + c1) * (2 * covariance_xy + c2)) / (
                (mean_x * mean_x + mean_y * mean_y + c1) * (variance_x + variance_y + c2)
            )
            padding = (window - 1) // 2
            ssim_error = 1.0 - ssim_map[:, padding:-padding, padding:-padding].mean(axis=(1, 2))
        return color_error, edge_error, ssim_error

    def _evaluate_cpu_scale(self, population, dimensions):
        target = self.target_at(dimensions)
        rows = []
        for genes in population:
            blocks = genes.reshape(self.triangle_count, 10)
            reconstruction = render_triangles([(row[:6], row[6:]) for row in blocks], dimensions)
            rows.append(_single_scale_components(
                target, reconstruction,
                color_weight=self.color_weight,
                edge_weight=self.edge_weight,
                ssim_weight=self.ssim_weight,
                detail_boost=self.detail_boost,
            ))
        return tuple(
            np.asarray([getattr(row, name) for row in rows])
            for name in ("color", "edge", "ssim")
        )

    def evaluate_many(self, population, size: int | tuple[int, int]) -> list[float]:
        if len(population) == 0:
            return []
        totals = np.zeros(len(population), dtype=float)
        for dimensions, scale_weight in zip(self.pyramid_dimensions(size), self.pyramid_weights):
            if self.gpu_active:
                try:
                    color, edge, ssim = self._evaluate_cuda_scale(population, dimensions)
                    scale_total = self.color_weight * color + self.edge_weight * edge + self.ssim_weight * ssim
                    totals += scale_weight * cp.asnumpy(scale_total)
                    continue
                except Exception as error:
                    warnings.warn(f"CUDA falló ({error}); se continúa en CPU.")
                    self.gpu_active = False
                    self.backend = "CPU"
                    self.gpu_cache.clear()
            color, edge, ssim = self._evaluate_cpu_scale(population, dimensions)
            totals += scale_weight * (
                self.color_weight * color + self.edge_weight * edge + self.ssim_weight * ssim
            )
        return totals.astype(float).tolist()

    def evaluate_many_cpu(self, population, size: int | tuple[int, int]) -> list[float]:
        """Evaluación de referencia con el mismo rasterizador usado para exportar."""
        if len(population) == 0:
            return []
        totals = np.zeros(len(population), dtype=float)
        for dimensions, scale_weight in zip(self.pyramid_dimensions(size), self.pyramid_weights):
            color, edge, ssim = self._evaluate_cpu_scale(population, dimensions)
            totals += scale_weight * (
                self.color_weight * color + self.edge_weight * edge + self.ssim_weight * ssim
            )
        return totals.astype(float).tolist()

    def evaluate_components(self, individual, size: int | tuple[int, int]) -> FitnessComponents:
        values = np.zeros(3, dtype=float)
        for dimensions, scale_weight in zip(self.pyramid_dimensions(size), self.pyramid_weights):
            color, edge, ssim = self._evaluate_cpu_scale([individual], dimensions)
            values += scale_weight * np.array([color[0], edge[0], ssim[0]])
        total = self.color_weight * values[0] + self.edge_weight * values[1] + self.ssim_weight * values[2]
        return FitnessComponents(*map(float, values), float(total))

    def evaluate(self, individual, size: int | tuple[int, int]) -> float:
        return self.evaluate_many([individual], size)[0]
