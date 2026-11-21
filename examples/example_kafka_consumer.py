"""Consume from Kafka/Redpanda, detect anomalies with PySAD, and checkpoint model state.

This recipe shows a streaming anomaly detection worker that:
- Consumes JSON messages from a Kafka/Redpanda topic
- Extracts numeric features from each message
- Scores each point with a PySAD detector before learning from it (prequential)
- Periodically checkpoints the model state to disk using pickle (zero dependencies)
- Resumes from the latest checkpoint on restart

If Kafka is not available, the script falls back to a synthetic stream so it runs
anywhere for demonstration.

Usage:
    python examples/example_kafka_consumer.py

Environment variables:
    KAFKA_BOOTSTRAP_SERVERS  Comma-separated broker list (default: localhost:9092)
    KAFKA_TOPIC              Topic to consume (default: metrics)
    KAFKA_GROUP_ID           Consumer group id (default: pysad-worker)
    CHECKPOINT_DIR           Directory for model checkpoints (default: ./checkpoints)
    CHECKPOINT_INTERVAL      Messages between checkpoints (default: 100)
"""

import json
import os
import pickle
import signal
from pathlib import Path
from typing import Any

import numpy as np

from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.utils import fix_seed

# Configuration from environment with sensible defaults
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "metrics")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "pysad-worker")
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "./checkpoints"))
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "100"))

# Feature columns expected in the JSON message (adjust to your schema)
FEATURE_KEYS = [
    "cpu",
    "memory",
    "latency_ms",
    "requests_per_second",
    "error_rate",
]

# Label key for evaluation (optional; if missing, evaluation is skipped)
LABEL_KEY = "label"


def make_consumer() -> Any | None:
    """Create a Kafka consumer if confluent-kafka is available."""
    try:
        from confluent_kafka import Consumer

        consumer = Consumer(
            {
                "bootstrap.servers": BOOTSTRAP_SERVERS,
                "group.id": GROUP_ID,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,  # We commit manually after checkpoint
            }
        )
        consumer.subscribe([TOPIC])
        return consumer
    except ImportError:
        return None


def parse_message(msg_value: bytes) -> tuple[np.ndarray, int | None]:
    """Parse a JSON message into a feature vector and optional label."""
    data = json.loads(msg_value.decode("utf-8"))
    features = np.array([float(data.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float64)
    label = data.get(LABEL_KEY)
    return features, int(label) if label is not None else None


def save_checkpoint(model: LODA, offset: int, path: Path) -> None:
    """Save model state and consumer offset to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    state = {"model": model, "offset": offset}
    with tmp_path.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> tuple[LODA, int] | None:
    """Load model state and consumer offset from disk."""
    if not path.exists():
        return None
    with path.open("rb") as f:
        state = pickle.load(f)
    return state["model"], state["offset"]


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the most recent checkpoint file, or None if none exist."""
    files = list(checkpoint_dir.glob("checkpoint_*.pkl"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def run_with_kafka(consumer: Any, model: LODA, start_offset: int, metric: AUROCMetric) -> None:
    """Run the worker consuming from Kafka."""
    msg_count = 0
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{TOPIC}.pkl"
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        print(f"\nSignal {signum} received, saving checkpoint and exiting...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Consuming from {TOPIC} on {BOOTSTRAP_SERVERS} (group: {GROUP_ID})")
    print(f"Checkpointing every {CHECKPOINT_INTERVAL} messages to {checkpoint_path}")

    while running:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        features, label = parse_message(msg.value())
        score = model.fit_score_partial(features)

        if label is not None:
            metric.update(label, score)

        msg_count += 1

        if msg_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, msg.offset(), checkpoint_path)
            consumer.commit(asynchronous=False)
            try:
                auroc = metric.get()
                auroc_str = f"{auroc:.3f}"
            except ValueError:
                auroc_str = "N/A (single class so far)"
            if label is not None:
                print(
                    f"  [{msg_count}] offset={msg.offset()} score={score:.4f} label={label} AUROC={auroc_str}"
                )
            else:
                print(f"  [{msg_count}] offset={msg.offset()} score={score:.4f} AUROC={auroc_str}")

    # Final checkpoint on shutdown
    save_checkpoint(model, msg.offset() if msg_count else start_offset, checkpoint_path)
    consumer.commit(asynchronous=False)
    consumer.close()
    try:
        final_auroc = f"{metric.get():.3f}"
    except ValueError:
        final_auroc = "N/A (single class)"
    print(f"Stopped after {msg_count} messages. Final AUROC: {final_auroc}")


def run_synthetic(model: LODA, metric: AUROCMetric, num_points: int = 1000) -> None:
    """Run the worker on a synthetic stream (fallback when Kafka is unavailable)."""
    from pysad.utils import generate_stream

    print("Kafka not available (confluent-kafka not installed or brokers unreachable).")
    print(f"Running synthetic stream with {num_points} points...")

    X, y = generate_stream(
        num_instances=num_points, num_features=len(FEATURE_KEYS), random_state=42
    )

    checkpoint_path = CHECKPOINT_DIR / "checkpoint_synthetic.pkl"

    for i, (features, label) in enumerate(zip(X, y, strict=True)):
        score = model.fit_score_partial(features)
        metric.update(label, score)

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, i, checkpoint_path)
            try:
                auroc = metric.get()
                print(f"  [{i + 1}] score={score:.4f} label={label} AUROC={auroc:.3f}")
            except ValueError:
                print(
                    f"  [{i + 1}] score={score:.4f} label={label} AUROC=N/A (single class so far)"
                )

    save_checkpoint(model, num_points - 1, checkpoint_path)
    try:
        final_auroc = f"{metric.get():.3f}"
    except ValueError:
        final_auroc = "N/A (single class)"
    print(f"Finished synthetic stream. Final AUROC: {final_auroc}")


def main() -> None:
    fix_seed(42)

    # Try to resume from latest checkpoint
    latest = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest:
        print(f"Resuming from checkpoint: {latest}")
        loaded = load_checkpoint(latest)
        if loaded:
            model, start_offset = loaded
            print(f"  Restored model at offset {start_offset}")
        else:
            model = LODA(random_state=42)
            start_offset = 0
    else:
        model = LODA(random_state=42)
        start_offset = 0
        print("No checkpoint found, starting fresh.")

    metric = AUROCMetric()

    # Try to create Kafka consumer
    consumer = make_consumer()

    if consumer is not None:
        try:
            run_with_kafka(consumer, model, start_offset, metric)
        except Exception as e:
            print(f"Kafka consumer failed: {e}")
            print("Falling back to synthetic stream...")
            consumer.close()
            run_synthetic(model, metric)
    else:
        run_synthetic(model, metric)


if __name__ == "__main__":
    main()
