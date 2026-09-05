"""Descarga de imágenes y benchmark reproducible del algoritmo genético."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Callable, Iterable
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .engine import GAConfig, GeneticImageGA
from .io import export_triangles
from .render import render_triangles


PICSUM_LIST_URL = "https://picsum.photos/v2/list"
DEFAULT_USER_AGENT = "DIAO2A-benchmark/1.0"


@dataclass(frozen=True, slots=True)
class PicsumImage:
    """Metadatos de una imagen guardada en el caché local."""

    id: str
    author: str
    width: int
    height: int
    source_url: str
    download_url: str
    path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Ubicación y tablas producidas por una ejecución del benchmark."""

    output_dir: Path
    runs: pd.DataFrame
    per_image: pd.DataFrame
    summary: dict


def normalize_image(
    image: Image.Image,
    target_size: tuple[int, int] = (120, 120),
) -> Image.Image:
    """Convierte a RGB y ajusta por cobertura con recorte centrado, sin deformar."""
    if len(target_size) != 2 or min(target_size) <= 0:
        raise ValueError("target_size debe contener dos dimensiones positivas")
    return ImageOps.fit(
        image.convert("RGB"),
        tuple(map(int, target_size)),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def benchmark_profile(base: GAConfig, profile: str) -> GAConfig:
    """Aplica uno de los dos perfiles acordados sobre la configuración del AG."""
    if profile == "quick":
        return replace(
            base,
            generations=500,
            initial_evaluation_size=32,
            evaluation_size=120,
        )
    if profile == "full":
        return replace(
            base,
            generations=15_000,
            initial_evaluation_size=48,
            evaluation_size=120,
        )
    raise ValueError("profile debe ser 'quick' o 'full'")


def _request_bytes(url: str, *, timeout: float, opener: Callable) -> bytes:
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with opener(request, timeout=timeout) as response:
        return response.read()


def _with_retries(action: Callable[[], bytes], retries: int, description: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return action()
        except Exception as error:  # urllib y Pillow exponen varias excepciones.
            last_error = error
            if attempt + 1 < retries:
                time.sleep(0.25 * (2**attempt))
    raise RuntimeError(f"No se pudo {description} tras {retries} intentos") from last_error


def _scaled_dimensions(width: int, height: int, max_side: int = 512) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _valid_cached_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _download_record(
    raw: dict,
    cache_dir: Path,
    *,
    retries: int,
    timeout: float,
    opener: Callable,
) -> PicsumImage:
    image_id = str(raw["id"])
    width, height = int(raw["width"]), int(raw["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensiones inválidas para la imagen {image_id}")
    local_path = cache_dir / f"picsum_{image_id}.jpg"
    scaled_width, scaled_height = _scaled_dimensions(width, height)
    download_url = (
        f"https://picsum.photos/id/{image_id}/{scaled_width}/{scaled_height}.jpg"
    )
    if not _valid_cached_image(local_path):
        def download_and_validate() -> bytes:
            payload = _request_bytes(download_url, timeout=timeout, opener=opener)
            with Image.open(BytesIO(payload)) as downloaded:
                downloaded.load()
                converted = downloaded.convert("RGB")
            converted.save(local_path, format="JPEG", quality=95)
            if not _valid_cached_image(local_path):
                raise OSError(f"La imagen descargada {image_id} no es válida")
            return payload

        _with_retries(
            download_and_validate,
            retries,
            f"descargar la imagen {image_id}",
        )
    return PicsumImage(
        id=image_id,
        author=str(raw.get("author", "")),
        width=width,
        height=height,
        source_url=str(raw.get("url", "")),
        download_url=download_url,
        path=local_path,
    )


def _load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data.get("images", []) if isinstance(data, dict) else []


def fetch_picsum_images(
    count: int,
    cache_dir: str | Path,
    *,
    page_size: int = 100,
    retries: int = 3,
    timeout: float = 30.0,
    opener: Callable = urlopen,
) -> list[PicsumImage]:
    """Obtiene exactamente ``count`` imágenes únicas y mantiene un manifiesto estable."""
    if count <= 0:
        raise ValueError("count debe ser positivo")
    if page_size <= 0 or retries <= 0 or timeout <= 0:
        raise ValueError("page_size, retries y timeout deben ser positivos")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path / "picsum_manifest.json"
    manifest_records = _load_manifest(manifest_path)
    selected: list[PicsumImage] = []
    seen: set[str] = set()

    def try_record(raw: dict) -> None:
        image_id = str(raw.get("id", ""))
        if not image_id or image_id in seen or len(selected) >= count:
            return
        try:
            record = _download_record(
                raw,
                cache_path,
                retries=retries,
                timeout=timeout,
                opener=opener,
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            return
        selected.append(record)
        seen.add(record.id)

    for raw in manifest_records:
        try_record(raw)
        if len(selected) == count:
            break

    page = 1
    empty_pages = 0
    while len(selected) < count:
        list_url = f"{PICSUM_LIST_URL}?page={page}&limit={page_size}"
        payload = _with_retries(
            lambda: _request_bytes(list_url, timeout=timeout, opener=opener),
            retries,
            f"consultar la página {page} de Picsum",
        )
        try:
            records = json.loads(payload)
        except json.JSONDecodeError as error:
            raise RuntimeError("Picsum devolvió una lista JSON inválida") from error
        if not isinstance(records, list):
            raise RuntimeError("Picsum devolvió una respuesta inesperada")
        before = len(selected)
        for raw in records:
            if isinstance(raw, dict):
                try_record(raw)
        empty_pages = empty_pages + 1 if len(selected) == before else 0
        if not records or empty_pages >= 3:
            raise RuntimeError(
                f"Sólo se pudieron obtener {len(selected)} de {count} imágenes"
            )
        page += 1

    manifest = {
        "version": 1,
        "images": [
            {
                "id": image.id,
                "author": image.author,
                "width": image.width,
                "height": image.height,
                "url": image.source_url,
            }
            for image in selected
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return selected


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: None for key in ("mean", "std", "median", "min", "max", "p05", "p95", "iqr")}
    q05, q25, q50, q75, q95 = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(q50),
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(q05),
        "p95": float(q95),
        "iqr": float(q75 - q25),
    }


METRIC_COLUMNS = (
    "best_fitness",
    "color_error",
    "edge_error",
    "ssim_error",
    "elapsed_seconds",
    "generations_per_second",
)


def _summaries(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    successful = runs[runs["status"] == "completed"].copy()
    per_image_rows: list[dict] = []
    for image_id, group in runs.groupby("image_id", sort=False):
        completed = group[group["status"] == "completed"]
        row: dict = {
            "image_id": image_id,
            "runs": int(len(group)),
            "completed": int(len(completed)),
            "failed": int((group["status"] == "failed").sum()),
        }
        for metric in METRIC_COLUMNS:
            for statistic, value in _stats(completed[metric]).items():
                row[f"{metric}_{statistic}"] = value
        per_image_rows.append(row)
    per_image = pd.DataFrame(per_image_rows)
    global_stats = {metric: _stats(successful[metric]) for metric in METRIC_COLUMNS}
    stop_reasons = successful["stop_reason"].value_counts().to_dict()
    worst_image = None
    if not per_image.empty and "best_fitness_mean" in per_image:
        valid = per_image.dropna(subset=["best_fitness_mean"])
        if not valid.empty:
            worst = valid.loc[valid["best_fitness_mean"].idxmax()]
            worst_image = {
                "image_id": str(worst["image_id"]),
                "mean_fitness": float(worst["best_fitness_mean"]),
            }
    summary = {
        "total_runs": int(len(runs)),
        "completed_runs": int(len(successful)),
        "failed_runs": int((runs["status"] == "failed").sum()),
        "completion_rate": float(len(successful) / len(runs)) if len(runs) else 0.0,
        "stop_reasons": {str(key): int(value) for key, value in stop_reasons.items()},
        "worst_image": worst_image,
        "metrics": global_stats,
    }
    return per_image, summary


def _plot_ranking(axis, per_image: pd.DataFrame) -> None:
    valid = per_image.dropna(subset=["best_fitness_mean"]).sort_values("best_fitness_mean")
    if valid.empty:
        axis.text(0.5, 0.5, "Sin corridas exitosas", ha="center", va="center")
        axis.axis("off")
        return
    positions = np.arange(len(valid))
    axis.errorbar(
        positions,
        valid["best_fitness_mean"],
        yerr=valid["best_fitness_std"].fillna(0.0),
        fmt="o",
        capsize=4,
        color="tab:blue",
    )
    axis.set_xticks(positions, valid["image_id"], rotation=45, ha="right")
    axis.set(title="Ranking por imagen", xlabel="ID Picsum", ylabel="Fitness medio ± desvío")
    axis.grid(True, axis="y", alpha=0.25)


def _plot_heatmap(axis, successful: pd.DataFrame) -> None:
    matrix = successful.pivot(index="image_id", columns="seed", values="best_fitness")
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis_r")
    axis.set_xticks(np.arange(len(matrix.columns)), [str(seed) for seed in matrix.columns])
    axis.set_yticks(np.arange(len(matrix.index)), [str(image_id) for image_id in matrix.index])
    axis.set(title="Sensibilidad a la seed", xlabel="Seed", ylabel="ID Picsum")
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            if pd.notna(value):
                axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=7)
    axis.figure.colorbar(image, ax=axis, label="Fitness final", fraction=0.046, pad=0.04)


def _plot_convergence(axis, histories: list[dict], generations: int) -> None:
    if not histories:
        axis.text(0.5, 0.5, "Sin historiales", ha="center", va="center")
        axis.axis("off")
        return
    padded = []
    for record in histories:
        history = np.asarray(record["history"], dtype=float)
        step = max(1, len(history) // 500)
        shown_x = np.arange(0, len(history), step)
        axis.plot(shown_x, history[::step], color="tab:blue", alpha=0.10, linewidth=0.7)
        padded.append(np.pad(history, (0, generations + 1 - len(history)), mode="edge"))
    matrix = np.vstack(padded)
    x = np.arange(generations + 1)
    median = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
    axis.fill_between(x, q25, q75, color="tab:blue", alpha=0.22, label="P25–P75")
    axis.plot(x, median, color="navy", linewidth=2, label="Mediana")
    axis.set(title="Convergencia", xlabel="Generación", ylabel="Mejor fitness")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)


def _plot_time_quality(axis, successful: pd.DataFrame) -> None:
    image_codes, image_labels = pd.factorize(successful["image_id"], sort=True)
    markers = ("o", "s", "^")
    for marker, seed in zip(markers, sorted(successful["seed"].unique())):
        mask = successful["seed"] == seed
        points = axis.scatter(
            successful.loc[mask, "elapsed_seconds"],
            successful.loc[mask, "best_fitness"],
            c=image_codes[mask.to_numpy()],
            cmap="tab20",
            marker=marker,
            s=55,
            alpha=0.8,
            label=f"Seed {seed}",
        )
    axis.set(title="Calidad vs. tiempo", xlabel="Segundos", ylabel="Fitness final")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    if len(image_labels) > 1:
        colorbar = axis.figure.colorbar(points, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Índice de imagen")


def _save_gallery(successful: pd.DataFrame, output_dir: Path) -> None:
    image_means = successful.groupby("image_id")["best_fitness"].mean().sort_values()
    if image_means.empty:
        return
    chosen_ids = [
        image_means.index[0],
        image_means.index[len(image_means) // 2],
        image_means.index[-1],
    ]
    labels = ("Mejor caso", "Caso mediano", "Peor caso")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for column, (label, image_id) in enumerate(zip(labels, chosen_ids)):
        candidates = successful[successful["image_id"] == image_id]
        mean = float(image_means.loc[image_id])
        representative = candidates.loc[(candidates["best_fitness"] - mean).abs().idxmin()]
        target = Image.open(output_dir / representative["target_path"])
        reconstruction = Image.open(output_dir / representative["reconstruction_path"])
        axes[0, column].imshow(target)
        axes[0, column].set_title(f"{label} · objetivo\nID {image_id}")
        axes[1, column].imshow(reconstruction)
        axes[1, column].set_title(
            f"Reconstrucción · seed {representative['seed']}\n"
            f"fitness={representative['best_fitness']:.4f}"
        )
        for axis in axes[:, column]:
            axis.axis("off")
        target.close()
        reconstruction.close()
    fig.suptitle("Galería representativa: objetivo vs. reconstrucción", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "result_gallery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_plots(
    runs: pd.DataFrame,
    per_image: pd.DataFrame,
    histories: list[dict],
    generations: int,
    output_dir: Path,
) -> None:
    successful = runs[runs["status"] == "completed"]
    if successful.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    _plot_ranking(axes[0, 0], per_image)
    _plot_heatmap(axes[0, 1], successful)
    _plot_convergence(axes[1, 0], histories, generations)
    _plot_time_quality(axes[1, 1], successful)
    fig.suptitle("Rendimiento y robustez del algoritmo", fontsize=17)
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = ["best_fitness", "color_error", "edge_error", "ssim_error"]
    labels = ["Fitness total", "Color", "Bordes", "SSIM"]
    values = [successful[metric].dropna().to_numpy() for metric in metrics]
    fig, axis = plt.subplots(figsize=(10, 5))
    violins = axis.violinplot(values, showmeans=False, showmedians=True, showextrema=True)
    for body in violins["bodies"]:
        body.set_facecolor("tab:blue")
        body.set_alpha(0.45)
    axis.boxplot(values, widths=0.12, showfliers=True)
    axis.set_xticks(np.arange(1, len(labels) + 1), labels)
    axis.set(title="Distribución de calidad", ylabel="Error")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    stop_counts = successful["stop_reason"].value_counts()
    if int((runs["status"] == "failed").sum()):
        stop_counts.loc["failed"] = int((runs["status"] == "failed").sum())
    fig, (cards, bars) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    cards.axis("off")
    kpis = (
        ("Fitness mediano", successful["best_fitness"].median(), ".4f"),
        ("Peor fitness", successful["best_fitness"].max(), ".4f"),
        ("Tiempo mediano", successful["elapsed_seconds"].median(), ".1f"),
        ("Tasa de éxito", len(successful) / len(runs), ".1%"),
    )
    for index, (label, value, number_format) in enumerate(kpis):
        x, y = (0.25 + 0.5 * (index % 2), 0.72 - 0.5 * (index // 2))
        cards.text(x, y, format(value, number_format), ha="center", va="center", fontsize=22, weight="bold")
        cards.text(x, y - 0.13, label, ha="center", va="center", fontsize=10, color="dimgray")
    bars.bar(stop_counts.index.astype(str), stop_counts.values, color="tab:blue", alpha=0.8)
    bars.set(title="Motivos de finalización", ylabel="Corridas")
    bars.tick_params(axis="x", rotation=30)
    bars.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Resumen de robustez", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    _save_gallery(successful, output_dir)


def _new_output_dir(root: Path, profile: str, run_name: str | None) -> Path:
    timestamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / profile / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / profile / f"{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def run_benchmark(
    images: Iterable[PicsumImage],
    base_config: GAConfig,
    triangle_count: int,
    output_root: str | Path,
    *,
    profile: str = "quick",
    seeds: tuple[int, ...] = (42, 43, 44),
    target_size: tuple[int, int] = (120, 120),
    progress: bool = True,
    run_name: str | None = None,
    ga_factory: Callable = GeneticImageGA,
) -> BenchmarkResult:
    """Ejecuta todos los pares imagen/seed y persiste artefactos y estadísticas."""
    image_list = list(images)
    if not image_list:
        raise ValueError("Se necesita al menos una imagen")
    if not seeds:
        raise ValueError("Se necesita al menos una seed")
    if triangle_count <= 0:
        raise ValueError("triangle_count debe ser positivo")

    configured = benchmark_profile(base_config, profile)
    output_dir = _new_output_dir(Path(output_root), profile, run_name)
    rows: list[dict] = []
    histories: list[dict] = []
    failures = 0
    total_runs = len(image_list) * len(seeds)
    total_generations = total_runs * configured.generations
    outer = tqdm(
        total=total_generations,
        desc="Benchmark",
        unit="gen",
        dynamic_ncols=True,
        disable=not progress,
    )
    run_number = 0
    for image_index, image_record in enumerate(image_list, start=1):
        for seed in seeds:
            run_number += 1
            progress_state = {"completed": 0}
            outer.set_description(
                f"Benchmark {run_number}/{total_runs} · img {image_record.id} · seed {seed}"
            )

            def update_generation(
                completed: int,
                best_fitness: float,
                resolution: int,
                stagnation: int,
            ) -> None:
                delta = completed - progress_state["completed"]
                if delta > 0:
                    outer.update(delta)
                    progress_state["completed"] = completed
                if completed % 10 == 0 or completed == configured.generations:
                    outer.set_postfix(
                        error=f"{best_fitness:.5f}",
                        res=resolution,
                        sin_mejora=stagnation,
                        fallos=failures,
                    )

            case_dir = output_dir / f"image_{image_index:03d}_id_{image_record.id}" / f"seed_{seed}"
            case_dir.mkdir(parents=True, exist_ok=True)
            config = replace(configured, seed=seed, progress=progress)
            base_row = {
                "image_index": image_index,
                "image_id": image_record.id,
                "author": image_record.author,
                "source_url": image_record.source_url,
                "original_width": image_record.width,
                "original_height": image_record.height,
                "target_width": target_size[0],
                "target_height": target_size[1],
                "seed": seed,
                "profile": profile,
            }
            metrics_path = case_dir / "metrics.json"
            try:
                with Image.open(image_record.path) as source:
                    target = normalize_image(source, target_size)
                target.save(case_dir / "target.png")
                ga = ga_factory(target, triangle_count, config)
                result = ga.run(
                    progress_callback=update_generation if progress else None,
                )
                triangles = ga.genes_to_triangles(result.best_genes)
                reconstruction = render_triangles(triangles, target.size)
                reconstruction.save(case_dir / "reconstruction.png")
                export_triangles(triangles, case_dir / "triangles.json")
                components = ga.evaluator.evaluate_components(
                    result.best_genes, config.evaluation_size
                )
                generations_per_second = (
                    result.generations_completed / result.elapsed_seconds
                    if result.elapsed_seconds > 0
                    else None
                )
                row = {
                    **base_row,
                    "status": "completed",
                    "target_path": str((case_dir / "target.png").relative_to(output_dir)),
                    "reconstruction_path": str(
                        (case_dir / "reconstruction.png").relative_to(output_dir)
                    ),
                    "best_fitness": result.best_fitness,
                    "color_error": components.color,
                    "edge_error": components.edge,
                    "ssim_error": components.ssim,
                    "fitness_components_total": components.total,
                    "generations_completed": result.generations_completed,
                    "elapsed_seconds": result.elapsed_seconds,
                    "generations_per_second": generations_per_second,
                    "stop_reason": result.stop_reason,
                    "backend": result.backend,
                    "error_type": None,
                    "error_message": None,
                }
                histories.append(
                    {
                        "image_id": image_record.id,
                        "seed": seed,
                        "history": result.history,
                    }
                )
                case_metrics = {
                    **row,
                    "triangle_count": triangle_count,
                    "config": asdict(config),
                    "history": result.history,
                    "resolution_history": result.resolution_history,
                    "diversity_history": result.diversity_history,
                }
            except Exception as error:
                failures += 1
                row = {
                    **base_row,
                    "status": "failed",
                    "target_path": (
                        str((case_dir / "target.png").relative_to(output_dir))
                        if (case_dir / "target.png").is_file()
                        else None
                    ),
                    "reconstruction_path": None,
                    **{metric: None for metric in METRIC_COLUMNS},
                    "fitness_components_total": None,
                    "generations_completed": None,
                    "stop_reason": None,
                    "backend": None,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                case_metrics = {**row, "triangle_count": triangle_count, "config": asdict(config)}
            metrics_path.write_text(json.dumps(case_metrics, indent=2), encoding="utf-8")
            rows.append(row)
            # Una parada anticipada representa una corrida terminada. Se completan sus
            # generaciones planificadas para que la barra global llegue siempre al 100 %.
            outer.update(configured.generations - progress_state["completed"])
            outer.set_postfix(
                error=(f"{row['best_fitness']:.5f}" if row["best_fitness"] is not None else "falló"),
                fallos=failures,
            )
    outer.close()

    runs = pd.DataFrame(rows)
    runs.to_csv(output_dir / "runs.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    per_image, summary = _summaries(runs)
    per_image.to_csv(output_dir / "per_image.csv", index=False)
    summary.update(
        {
            "profile": profile,
            "image_count": len(image_list),
            "seeds": list(seeds),
            "target_size": list(target_size),
            "triangle_count": triangle_count,
            "config": asdict(configured),
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _save_plots(runs, per_image, histories, configured.generations, output_dir)
    return BenchmarkResult(output_dir, runs, per_image, summary)
