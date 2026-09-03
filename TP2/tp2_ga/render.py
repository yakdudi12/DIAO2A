"""Rasterizado CPU y CUDA de triángulos RGBA."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

try:
    import cupy as cp
except (ImportError, OSError):
    cp = None


def render_triangles(triangles, size=(100, 100), background=(255, 255, 255)) -> Image.Image:
    """Renderiza triángulos normalizados sobre un canvas RGB."""
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = size
    for points, color in triangles:
        x1, y1, x2, y2, x3, y3 = points
        polygon = [
            (round(x1 * (width - 1)), round(y1 * (height - 1))),
            (round(x2 * (width - 1)), round(y2 * (height - 1))),
            (round(x3 * (width - 1)), round(y3 * (height - 1))),
        ]
        rgba = tuple(round(float(channel) * 255) for channel in color)
        draw.polygon(polygon, fill=rgba)
    return image


def render_high_resolution(triangles, size, supersample=2) -> Image.Image:
    """Renderiza con antialiasing por supersampling."""
    if supersample <= 1:
        return render_triangles(triangles, size)
    large_size = (size[0] * supersample, size[1] * supersample)
    return render_triangles(triangles, large_size).resize(size, Image.Resampling.LANCZOS)


def cuda_available() -> bool:
    """Indica si CuPy puede crear un contexto CUDA."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def cuda_device_name() -> str | None:
    if not cuda_available():
        return None
    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"]
    return name.decode(errors="replace") if isinstance(name, bytes) else str(name)


_CUDA_KERNELS = None


def _get_cuda_kernels():
    global _CUDA_KERNELS
    if _CUDA_KERNELS is not None:
        return _CUDA_KERNELS

    source = r'''
    #define TILE 16
    #define TRI_VALUES 14
    #define TRI_CHUNK 16

    extern "C" __global__
    void prepare_triangles(const float* genes, const int total,
                           const int width, const int height, float* prepared) {
        int q = blockDim.x * blockIdx.x + threadIdx.x;
        if (q >= total) return;
        const float* g = genes + q * 10;
        float* p = prepared + q * TRI_VALUES;
        p[0] = nearbyintf(g[0] * (width - 1));
        p[1] = nearbyintf(g[1] * (height - 1));
        p[2] = nearbyintf(g[2] * (width - 1));
        p[3] = nearbyintf(g[3] * (height - 1));
        p[4] = nearbyintf(g[4] * (width - 1));
        p[5] = nearbyintf(g[5] * (height - 1));
        p[6] = nearbyintf(fminf(1.0f, fmaxf(0.0f, g[6])) * 255.0f);
        p[7] = nearbyintf(fminf(1.0f, fmaxf(0.0f, g[7])) * 255.0f);
        p[8] = nearbyintf(fminf(1.0f, fmaxf(0.0f, g[8])) * 255.0f);
        p[9] = nearbyintf(fminf(1.0f, fmaxf(0.0f, g[9])) * 255.0f) / 255.0f;
        p[10] = fminf(p[0], fminf(p[2], p[4]));
        p[11] = fmaxf(p[0], fmaxf(p[2], p[4]));
        p[12] = fminf(p[1], fminf(p[3], p[5]));
        p[13] = fmaxf(p[1], fmaxf(p[3], p[5]));
    }

    extern "C" __global__
    void build_tile_lists(const float* prepared, const int batch, const int k,
                          const int width, const int height, const int tiles_x,
                          const int tiles_y, int* tile_indices, int* tile_counts) {
        int q = blockDim.x * blockIdx.x + threadIdx.x;
        int tiles = tiles_x * tiles_y;
        if (q >= batch * tiles) return;
        int individual = q / tiles;
        int tile = q - individual * tiles;
        int tx = tile % tiles_x, ty = tile / tiles_x;
        float left = (float)(tx * TILE), top = (float)(ty * TILE);
        float right = fminf((float)(width - 1), left + TILE - 1);
        float bottom = fminf((float)(height - 1), top + TILE - 1);
        int count = 0;
        for (int t = 0; t < k; ++t) {
            const float* p = prepared + (individual * k + t) * TRI_VALUES;
            float area = (p[2]-p[0])*(p[5]-p[1]) - (p[3]-p[1])*(p[4]-p[0]);
            if (fabsf(area) >= 0.5f && p[11] >= left && p[10] <= right &&
                p[13] >= top && p[12] <= bottom) {
                tile_indices[q * k + count++] = t;
            }
        }
        tile_counts[q] = count;
    }

    extern "C" __global__
    void raster_tiles(const float* prepared, const int* tile_indices,
                      const int* tile_counts, const int batch, const int k,
                      const int width, const int height, const int tiles_x,
                      const int tiles_y, float* output) {
        int q = blockIdx.x;
        int tiles = tiles_x * tiles_y;
        if (q >= batch * tiles) return;
        int individual = q / tiles;
        int tile = q - individual * tiles;
        int tid = threadIdx.x;
        int px_i = (tile % tiles_x) * TILE + (tid % TILE);
        int py_i = (tile / tiles_x) * TILE + (tid / TILE);
        bool valid = px_i < width && py_i < height;
        float px = (float)px_i, py = (float)py_i;
        float red = 255.0f, green = 255.0f, blue = 255.0f;
        __shared__ float cache[TRI_CHUNK * TRI_VALUES];
        int count = tile_counts[q];

        for (int base = 0; base < count; base += TRI_CHUNK) {
            int n = count - base;
            if (n > TRI_CHUNK) n = TRI_CHUNK;
            for (int z = tid; z < n * TRI_VALUES; z += blockDim.x) {
                int local_t = z / TRI_VALUES, value = z % TRI_VALUES;
                int t = tile_indices[q * k + base + local_t];
                cache[z] = prepared[(individual * k + t) * TRI_VALUES + value];
            }
            __syncthreads();
            if (valid) for (int j = 0; j < n; ++j) {
                const float* p = cache + j * TRI_VALUES;
                if (px < p[10] || px > p[11] || py < p[12] || py > p[13]) continue;
                float e1 = (px-p[0])*(p[3]-p[1]) - (py-p[1])*(p[2]-p[0]);
                float e2 = (px-p[2])*(p[5]-p[3]) - (py-p[3])*(p[4]-p[2]);
                float e3 = (px-p[4])*(p[1]-p[5]) - (py-p[5])*(p[0]-p[4]);
                bool inside = ((e1 >= 0 && e2 >= 0 && e3 >= 0) ||
                               (e1 <= 0 && e2 <= 0 && e3 <= 0));
                if (inside) {
                    float a = p[9];
                    red = nearbyintf(p[6] * a + red * (1.0f - a));
                    green = nearbyintf(p[7] * a + green * (1.0f - a));
                    blue = nearbyintf(p[8] * a + blue * (1.0f - a));
                }
            }
            __syncthreads();
        }
        if (valid) {
            int out = ((individual * height + py_i) * width + px_i) * 3;
            output[out] = red / 255.0f;
            output[out + 1] = green / 255.0f;
            output[out + 2] = blue / 255.0f;
        }
    }
    '''
    module = cp.RawModule(
        code=source,
        name_expressions=("prepare_triangles", "build_tile_lists", "raster_tiles"),
    )
    _CUDA_KERNELS = tuple(
        module.get_function(name)
        for name in ("prepare_triangles", "build_tile_lists", "raster_tiles")
    )
    return _CUDA_KERNELS


def render_population_cuda(genes: np.ndarray, triangle_count: int, size):
    """Rasteriza una población y devuelve `(B, H, W, 3)` residente en GPU."""
    if not cuda_available():
        raise RuntimeError("CUDA no está disponible")
    width, height = map(int, size)
    # CuPy 14 ya no acepta siempre un ndarray de NumPy directamente aquí.
    genes_gpu = cp.ascontiguousarray(cp.asarray(genes, dtype=cp.float32))
    if genes_gpu.ndim == 1:
        genes_gpu = genes_gpu[None, :]
    batch = int(genes_gpu.shape[0])
    tiles_x, tiles_y = (width + 15) // 16, (height + 15) // 16
    tiles_total = batch * tiles_x * tiles_y

    prepared = cp.empty((batch * triangle_count, 14), dtype=cp.float32)
    tile_indices = cp.empty((tiles_total, triangle_count), dtype=cp.int32)
    tile_counts = cp.empty(tiles_total, dtype=cp.int32)
    output = cp.empty((batch, height, width, 3), dtype=cp.float32)
    prepare, build_lists, raster = _get_cuda_kernels()

    total_triangles = batch * triangle_count
    prepare(
        ((total_triangles + 255) // 256,),
        (256,),
        (
            genes_gpu,
            np.int32(total_triangles),
            np.int32(width),
            np.int32(height),
            prepared,
        ),
    )
    build_lists(
        ((tiles_total + 255) // 256,),
        (256,),
        (
            prepared,
            np.int32(batch),
            np.int32(triangle_count),
            np.int32(width),
            np.int32(height),
            np.int32(tiles_x),
            np.int32(tiles_y),
            tile_indices,
            tile_counts,
        ),
    )
    raster(
        (tiles_total,),
        (256,),
        (
            prepared,
            tile_indices,
            tile_counts,
            np.int32(batch),
            np.int32(triangle_count),
            np.int32(width),
            np.int32(height),
            np.int32(tiles_x),
            np.int32(tiles_y),
            output,
        ),
    )
    return output
