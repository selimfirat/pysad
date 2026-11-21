"""Show how generate_stream creates point, burst and drift anomalies."""

import numpy as np

from pysad.utils import generate_stream


def print_anomaly_info(name: str, X: np.ndarray, y: np.ndarray) -> None:
    num_anomalies = int(y.sum())
    num_normal = int((y == 0).sum())
    print(f"{name}: {X.shape[0]} instances, {X.shape[1]} features")
    print(f"  Normal: {num_normal}, Anomalies: {num_anomalies}")
    anomaly_indices = np.where(y == 1)[0]
    if anomaly_indices.size:
        print(
            f"  Anomaly indices: {anomaly_indices[:10]}{'...' if anomaly_indices.size > 10 else ''}"
        )
        gaps = np.diff(anomaly_indices)
        burst_gaps = gaps[gaps > 1]
        print(f"  Max gap between anomalies: {gaps.max() if gaps.size else 0}")
        print(f"  Number of anomaly groups: {1 + burst_gaps.size}")


if __name__ == "__main__":
    print("=== Default stream (points, bursts, drift) ===")
    X, y = generate_stream(random_state=0)
    print_anomaly_info("Default", X, y)

    print("=== Only point anomalies ===")
    X, y = generate_stream(num_bursts=0, num_drifts=0, num_points=20, random_state=0)
    print_anomaly_info("Points only", X, y)

    print("=== Only burst anomalies ===")
    X, y = generate_stream(
        num_points=0, num_drifts=0, num_bursts=5, burst_length=15, random_state=0
    )
    print_anomaly_info("Bursts only", X, y)

    print("=== Only drift anomalies ===")
    X, y = generate_stream(
        num_points=0, num_bursts=0, num_drifts=3, drift_length=80, random_state=0
    )
    print_anomaly_info("Drifts only", X, y)

    print("=== Higher magnitude anomalies ===")
    X, y = generate_stream(magnitude=10.0, random_state=0)
    print_anomaly_info("High magnitude", X, y)

    print("=== More features ===")
    X, y = generate_stream(num_features=10, random_state=0)
    print_anomaly_info("10 features", X, y)

    print("=== Longer stream ===")
    X, y = generate_stream(num_instances=10000, random_state=0)
    print_anomaly_info("10000 instances", X, y)
