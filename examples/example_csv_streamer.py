"""Score a CSV file one record at a time, with a separate numeric label."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from pysad.models import LODA
from pysad.utils import CSVStreamer

if __name__ == "__main__":
    # Create a small input file so this example runs without a download.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "metrics.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "cpu", "latency", "label"])
            writer.writerows(
                [
                    ["09:00", 45, 100, 0],
                    ["09:01", 47, 105, 0],
                    ["09:02", 44, 98, 0],
                    ["09:03", 95, 300, 1],
                ]
            )

        model = LODA(random_state=42)
        streamer = CSVStreamer(feature_columns=[1, 2], label_column=3)
        for features, label in streamer.iter(path):
            score = model.fit_score_partial(features)
            print(f"label={label:.0f} score={score:.4f}")
