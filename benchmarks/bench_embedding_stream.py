"""Deterministic benchmark of a frozen encoder evaluated with ZeroShotEmbeddingDetector.

Run it from the repository root:

    python benchmarks/bench_embedding_stream.py --encoder my_encoder.py:encode --points 2000 --seed 7 --repeats 3

The benchmark loads a user-supplied frozen encoder (a plain callable that maps one raw instance to
an embedding vector, so no framework or new dependency is needed), encodes every instance of a
deterministic synthetic stream, scores each embedding with
:class:`~pysad.models.ZeroShotEmbeddingDetector`, and reports the AUROC, the milliseconds per
point, and whether a second run on the same stream gives exactly the same scores.

The semantic anchors of the detector are the embeddings of the warm-up points, so no labels are
used to build them and the warm-up points do not count in the AUROC. The encoder must be
deterministic: given the same instance it must return the same embedding. It receives a one
dimensional NumPy array of shape ``(num_features,)`` and must return a one dimensional NumPy array
of shape ``(embedding_dim,)`` with the same width on every call.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
import typing
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent)]

from pysad.models import ZeroShotEmbeddingDetector  # noqa: E402


def _load_encoder(spec: str):
    """Load a frozen encoder callable from a ``path/to/file.py:function_name`` spec.

    Args:
        spec (str): The encoder specification, for example ``my_encoder.py:encode``.

    Returns:
        typing.Callable: The callable that maps one raw instance to one embedding vector.

    Raises:
        ValueError: If the spec is invalid, the file is missing, the function is missing,
            or the value is not callable.
    """
    if ":" not in spec:
        raise ValueError(f"Encoder spec must be 'path/to/file.py:function_name', got '{spec}'")
    file_path, func_name = spec.rsplit(":", 1)
    path = Path(file_path).resolve()
    if not path.is_file():
        raise ValueError(f"Encoder file not found: {path}")
    module_name = path.stem
    spec_obj = importlib.util.spec_from_file_location(module_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise ValueError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)
    if not hasattr(module, func_name):
        raise ValueError(f"Function '{func_name}' not found in {path}")
    encoder = getattr(module, func_name)
    if not callable(encoder):
        raise ValueError(f"'{func_name}' in {path} is not callable")
    return encoder


def _make_stream(points: int, seed: int, num_features: int, anomaly_fraction: float):
    """Return a deterministic synthetic stream and its anomaly labels.

    Args:
        points (int): The number of instances.
        seed (int): The random seed of the stream.
        num_features (int): The number of features of each instance.
        anomaly_fraction (float): The fraction of instances that are anomalies.

    Returns:
        tuple[np.ndarray, np.ndarray]: The stream of shape ``(points, num_features)`` and the
        binary labels of shape ``(points,)``. Anomalies are normal instances shifted by 5.0.
    """
    rng = np.random.default_rng(seed)
    stream = rng.normal(0.0, 1.0, size=(points, num_features))
    labels = np.zeros(points, dtype=int)
    num_anomalies = max(1, int(round(points * anomaly_fraction)))
    anomaly_idx = rng.choice(points, size=num_anomalies, replace=False)
    stream[anomaly_idx] = stream[anomaly_idx] + 5.0
    labels[anomaly_idx] = 1
    return stream, labels


def _encode(stream: np.ndarray, encoder) -> np.ndarray:
    """Encode every instance of a stream with the frozen encoder.

    Args:
        stream (np.ndarray): Stream of shape ``(points, num_features)``.
        encoder (typing.Callable): Frozen encoder that maps one instance to one embedding.

    Returns:
        np.ndarray: Embeddings of shape ``(points, embedding_dim)``.

    Raises:
        ValueError: If the encoder does not return a one dimensional vector, or returns a
            different width on different calls.
    """
    embedding_dim = None
    embeddings = None
    for i, x in enumerate(stream):
        embedding = np.asarray(encoder(x), dtype=np.float64)
        if embedding.ndim != 1:
            raise ValueError(
                f"The encoder must return a one-dimensional vector, got shape {embedding.shape}."
            )
        if embedding_dim is None:
            embedding_dim = embedding.shape[0]
            embeddings = np.empty((len(stream), embedding_dim), dtype=np.float64)
        elif embedding.shape[0] != embedding_dim:
            raise ValueError(
                "The encoder must return the same width on every call: "
                f"expected {embedding_dim}, got {embedding.shape[0]}."
            )
        embeddings[i] = embedding
    return embeddings


def run_benchmark(
    encoder: typing.Callable[[np.ndarray], np.ndarray],
    points: int = 2000,
    seed: int = 7,
    num_features: int = 8,
    anomaly_fraction: float = 0.05,
    repeats: int = 3,
    warmup: int = 64,
    num_anchors: int = 10,
    temperature: float = 1.0,
) -> dict[str, float | int | bool]:
    """Run the embedding-stream benchmark and return the results.

    The stream is encoded once with the frozen encoder. The embeddings of the first ``warmup``
    points become the semantic anchors of :class:`~pysad.models.ZeroShotEmbeddingDetector`, and
    each later embedding is scored before the detector learns from it (learning is a no-op here,
    so the scores are identical with ``score_partial``). The AUROC is computed on the scored
    points against the labels, excluding the warm-up.

    Args:
        points (int): The number of instances in the stream (Default=2000).
        seed (int): The random seed of the stream (Default=7).
        num_features (int): The number of features of each instance (Default=8).
        anomaly_fraction (float): The fraction of instances that are anomalies (Default=0.05).
        repeats (int): The number of timed runs (Default=3).
        warmup (int): The number of first points whose scores do not count and whose embeddings
            become the semantic anchors (Default=64).
        num_anchors (int): The number of warm-up embeddings used as semantic anchors
            (Default=10).
        temperature (float): The softmax temperature of ZeroShotEmbeddingDetector (Default=1.0).
        encoder (typing.Callable): The frozen encoder that maps one instance to one embedding.

    Returns:
        A mapping with the stream settings, the median milliseconds per point, the determinism
        flag, the AUROC, and the score range.
    """
    stream, labels = _make_stream(points, seed, num_features, anomaly_fraction)
    warmup = min(warmup, points // 2)
    num_points = len(stream) - warmup
    num_anchors = min(num_anchors, warmup)

    def timed_run() -> tuple[np.ndarray, np.ndarray]:
        embeddings = _encode(stream, encoder)
        anchor_embeddings = embeddings[:warmup][:num_anchors]
        detector = ZeroShotEmbeddingDetector(
            semantic_anchors=anchor_embeddings, temperature=temperature, normalize=True
        )
        scores = np.empty(num_points, dtype=np.float64)
        for i, x in enumerate(embeddings[warmup:]):
            scores[i] = detector.fit_score_partial(x)
        return scores, labels[warmup:]

    durations = []
    scores = np.empty(0, dtype=np.float64)
    eval_labels = np.empty(0, dtype=int)
    for _ in range(repeats):
        start = time.perf_counter()
        scores, eval_labels = timed_run()
        durations.append(time.perf_counter() - start)
    ms_per_point = 1000.0 * float(np.median(durations)) / num_points

    # Determinism: a fresh run on the same stream must give exactly the same scores.
    second_scores, _ = timed_run()
    deterministic = bool(np.array_equal(scores, second_scores))

    both = 0 < eval_labels.sum() < len(eval_labels)
    auroc = float(roc_auc_score(eval_labels, scores)) if both else float("nan")

    return {
        "points": points,
        "seed": seed,
        "num_features": num_features,
        "anomaly_fraction": anomaly_fraction,
        "repeats": repeats,
        "warmup": warmup,
        "num_anchors": num_anchors,
        "temperature": temperature,
        "ms_per_point": ms_per_point,
        "deterministic": deterministic,
        "auroc": auroc,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        required=True,
        help="Frozen encoder as 'path/to/file.py:function_name'.",
    )
    parser.add_argument("--points", type=int, default=2000, help="Points in the stream")
    parser.add_argument("--seed", type=int, default=7, help="Random seed of the stream")
    parser.add_argument(
        "--num-features", type=int, default=8, help="Number of features of each instance"
    )
    parser.add_argument(
        "--anomaly-fraction",
        type=float,
        default=0.05,
        help="Fraction of instances that are anomalies",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed runs")
    parser.add_argument(
        "--warmup", type=int, default=64, help="Points whose scores do not count and build anchors"
    )
    parser.add_argument(
        "--num-anchors", type=int, default=10, help="Warm-up embeddings used as semantic anchors"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Softmax temperature of the detector"
    )
    parser.add_argument("--output", type=Path, help="Write the results to this CSV file.")
    return parser.parse_args()


def main() -> int:
    """Run the benchmark command."""
    args = _parse_args()
    encoder = _load_encoder(args.encoder)
    results = run_benchmark(
        points=args.points,
        seed=args.seed,
        num_features=args.num_features,
        anomaly_fraction=args.anomaly_fraction,
        repeats=args.repeats,
        warmup=args.warmup,
        num_anchors=args.num_anchors,
        temperature=args.temperature,
        encoder=encoder,
    )
    print(f"Encoder: {args.encoder}")
    print(f"Points: {results['points']}")
    print(f"Seed: {results['seed']}")
    print(f"Num features: {results['num_features']}")
    print(f"Anomaly fraction: {results['anomaly_fraction']}")
    print(f"Timed runs: {results['repeats']}")
    print(f"Warm-up points: {results['warmup']}")
    print(f"Semantic anchors: {results['num_anchors']}")
    print(f"Temperature: {results['temperature']}")
    print(f"Median ms per point: {results['ms_per_point']:.4f}")
    print(f"Deterministic: {results['deterministic']}")
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"Score range: [{results['min_score']:.4f}, {results['max_score']:.4f}]")
    if args.output is not None:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)
    return 0 if results["deterministic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
