import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tp2_ga import GAConfig
from tp2_ga.benchmark import (
    PicsumImage,
    benchmark_profile,
    fetch_picsum_images,
    normalize_image,
    run_benchmark,
)
from tp2_ga.fitness import FitnessComponents


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def jpeg_bytes(color=(20, 80, 160)):
    buffer = BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="JPEG")
    return buffer.getvalue()


class NormalizationTests(unittest.TestCase):
    def test_landscape_and_portrait_become_square_rgb(self):
        landscape = Image.new("RGBA", (240, 120), (255, 0, 0, 128))
        portrait = Image.new("L", (120, 240), 80)

        for source in (landscape, portrait):
            normalized = normalize_image(source, (120, 120))
            self.assertEqual(normalized.size, (120, 120))
            self.assertEqual(normalized.mode, "RGB")

    def test_center_crop_keeps_the_center(self):
        source = Image.new("RGB", (300, 100), "red")
        for x in range(100, 200):
            for y in range(100):
                source.putpixel((x, y), (0, 255, 0))

        normalized = normalize_image(source, (120, 120))

        center = normalized.getpixel((60, 60))
        self.assertGreater(center[1], center[0])


class DownloadTests(unittest.TestCase):
    @patch("tp2_ga.benchmark.time.sleep", return_value=None)
    def test_pagination_retry_unique_count_and_cache(self, _sleep):
        calls = []
        failed_downloads = 0
        image_payload = jpeg_bytes()

        def opener(request, timeout):
            nonlocal failed_downloads
            url = request.full_url
            calls.append(url)
            if "/v2/list" in url:
                page = 2 if "page=2" in url else 1
                records = (
                    [
                        {"id": "10", "author": "A", "width": 400, "height": 200, "url": "u10"},
                        {"id": "10", "author": "A", "width": 400, "height": 200, "url": "u10"},
                    ]
                    if page == 1
                    else [{"id": "11", "author": "B", "width": 200, "height": 400, "url": "u11"}]
                )
                return FakeResponse(json.dumps(records).encode())
            if "/id/10/" in url and failed_downloads < 2:
                failed_downloads += 1
                raise OSError("fallo transitorio")
            return FakeResponse(image_payload)

        with tempfile.TemporaryDirectory() as temporary:
            images = fetch_picsum_images(2, temporary, opener=opener)
            self.assertEqual([image.id for image in images], ["10", "11"])
            self.assertEqual(failed_downloads, 2)
            self.assertTrue(all(image.path.is_file() for image in images))
            self.assertTrue(any("page=2" in url for url in calls))

            def offline_opener(*_args, **_kwargs):
                raise AssertionError("El caché no debería consultar la red")

            cached = fetch_picsum_images(2, temporary, opener=offline_opener)
            self.assertEqual([image.id for image in cached], ["10", "11"])


class FakeEvaluator:
    backend = "CPU fake"

    def evaluate_components(self, _genes, _size):
        return FitnessComponents(color=0.1, edge=0.2, ssim=0.3, total=0.16)


class FakeGA:
    seen_seeds = []

    def __init__(self, target, triangle_count, config):
        self.target = target
        self.triangle_count = triangle_count
        self.config = config
        self.evaluator = FakeEvaluator()
        self.seen_seeds.append(config.seed)

    def run(self, **kwargs):
        if self.config.seed == 43:
            raise RuntimeError("fallo controlado")
        callback = kwargs.get("progress_callback")
        if callback is not None:
            for generation in range(1, 6):
                callback(generation, 0.2 - generation / 100, 120, 0)
        return SimpleNamespace(
            best_genes=np.asarray([0.1, 0.1, 0.9, 0.1, 0.5, 0.9, 0.2, 0.4, 0.6, 0.8]),
            best_fitness=0.1 + self.config.seed / 10_000,
            generations_completed=5,
            elapsed_seconds=0.5,
            stop_reason="maximum_generations",
            backend="CPU fake",
            history=[0.2, 0.1],
            resolution_history=[32, 120],
            diversity_history=[4, 3],
        )

    def genes_to_triangles(self, genes):
        return [(genes[:6], genes[6:])]


class FakeProgress:
    def __init__(self):
        self.n = 0
        self.descriptions = []

    def update(self, amount):
        self.n += amount

    def set_description(self, description):
        self.descriptions.append(description)

    def set_postfix(self, **_kwargs):
        pass

    def close(self):
        pass


class BenchmarkTests(unittest.TestCase):
    def test_profiles(self):
        base = GAConfig()
        self.assertEqual(benchmark_profile(base, "quick").generations, 500)
        self.assertEqual(benchmark_profile(base, "quick").initial_evaluation_size, 32)
        self.assertEqual(benchmark_profile(base, "quick").evaluation_size, 120)
        self.assertEqual(benchmark_profile(base, "full").generations, 15_000)
        self.assertEqual(benchmark_profile(base, "full").initial_evaluation_size, 48)
        self.assertEqual(benchmark_profile(base, "full").evaluation_size, 120)

    def test_three_seeds_per_image_and_failure_continuity(self):
        FakeGA.seen_seeds.clear()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = []
            for index in range(2):
                path = root / f"source_{index}.png"
                Image.new("RGB", (180, 120), (index * 80, 50, 100)).save(path)
                images.append(
                    PicsumImage(
                        id=str(index),
                        author="test",
                        width=180,
                        height=120,
                        source_url=f"source-{index}",
                        download_url=f"download-{index}",
                        path=path,
                    )
                )

            progress = FakeProgress()
            with patch("tp2_ga.benchmark.tqdm", return_value=progress) as progress_factory:
                result = run_benchmark(
                    images,
                    GAConfig(use_gpu=False),
                    triangle_count=1,
                    output_root=root / "results",
                    seeds=(42, 43, 44),
                    progress=True,
                    run_name="test-run",
                    ga_factory=FakeGA,
                )

            self.assertEqual(len(result.runs), 6)
            self.assertEqual(result.summary["completed_runs"], 4)
            self.assertEqual(result.summary["failed_runs"], 2)
            self.assertEqual(FakeGA.seen_seeds, [42, 43, 44, 42, 43, 44])
            self.assertTrue((result.output_dir / "runs.csv").is_file())
            self.assertTrue((result.output_dir / "per_image.csv").is_file())
            self.assertTrue((result.output_dir / "summary.json").is_file())
            self.assertEqual(len(list(result.output_dir.rglob("metrics.json"))), 6)
            self.assertEqual(len(list(result.output_dir.rglob("reconstruction.png"))), 4)
            for plot_name in (
                "benchmark_dashboard.png",
                "result_gallery.png",
                "quality_distributions.png",
                "robustness_summary.png",
            ):
                self.assertTrue((result.output_dir / plot_name).is_file(), plot_name)
            progress_factory.assert_called_once()
            self.assertEqual(progress.n, 6 * 500)
            self.assertEqual(len(progress.descriptions), 6)


if __name__ == "__main__":
    unittest.main()
