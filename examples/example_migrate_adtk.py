"""Migrate from ADTK batch evaluation to PySAD prequential evaluation.

ADTK validates a whole series and runs ``fit_detect`` on it, then summarizes
anomalies after the run. PySAD scores each point when it arrives, then learns
from it, and evaluates prequentially with
:func:`~pysad.evaluation.evaluate_stream`.

This script shows the PySAD equivalent of ADTK's ``summarize_anomalies``:
score a stream with several metrics and export the results with
:func:`~pysad.evaluation.export_report`.
"""

from pysad.evaluation import AUPRMetric, AUROCMetric, evaluate_stream, export_report
from pysad.models import LODA
from pysad.utils import Data

if __name__ == "__main__":
    X, y = Data().get_data("cardio.mat")

    results = evaluate_stream(
        model=LODA(random_state=0),
        X=X,
        y=y,
        metrics={
            "AUROC": AUROCMetric(),
            "AUPR": AUPRMetric(),
        },
    )

    export_report(
        results=results,
        path="reports/cardio_loda_migration.json",
        model="LODA",
        dataset="cardio",
    )

    for name, value in results.items():
        print(f"{name}: {value:.3f}")
