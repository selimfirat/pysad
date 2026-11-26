import json

import numpy as np
import pytest

from pysad.evaluation import AUROCMetric, evaluate_stream, export_report
from pysad.models import LODA


def _write_and_read(tmp_path, format):
    path = tmp_path / f"report.{format}"
    results = {"AUROC": 0.95, "AUPR": 0.49, "Undefined": None}
    export_report(
        results=results,
        path=path,
        format=format,
        model="LODA",
        postprocessor="ConformalProbabilityCalibrator",
        dataset="cardio",
    )
    return path.read_text(encoding="utf-8")


def test_export_report_json(tmp_path):
    text = _write_and_read(tmp_path, "json")
    document = json.loads(text)
    assert document == {
        "metadata": {
            "model": "LODA",
            "postprocessor": "ConformalProbabilityCalibrator",
            "dataset": "cardio",
        },
        "results": {"AUROC": 0.95, "AUPR": 0.49, "Undefined": None},
    }


def test_export_report_csv(tmp_path):
    text = _write_and_read(tmp_path, "csv")
    lines = text.strip().splitlines()
    assert lines[0] == "model,postprocessor,dataset,metric,value"
    assert "LODA,ConformalProbabilityCalibrator,cardio,AUROC,0.95" in lines
    assert "LODA,ConformalProbabilityCalibrator,cardio,AUPR,0.49" in lines
    assert "LODA,ConformalProbabilityCalibrator,cardio,Undefined," in lines


def test_export_report_markdown(tmp_path):
    text = _write_and_read(tmp_path, "markdown")
    assert "# Evaluation Report" in text
    assert "| Metric | Value |" in text
    assert "| AUROC | 0.95 |" in text
    assert "| Undefined |  |" in text
    assert "| model | LODA |" in text


def test_export_report_creates_parent_directories(tmp_path):
    path = tmp_path / "sub" / "dir" / "report.json"
    export_report(results={"AUROC": 0.5}, path=path)
    assert path.exists()


def test_export_report_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        export_report(results={"AUROC": 0.5}, path=tmp_path / "r.txt", format="xml")


def test_export_report_with_real_evaluation(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.zeros(200, dtype=int)
    y[::20] = 1
    X[y == 1] += 5

    results = evaluate_stream(
        model=LODA(random_state=0),
        X=X,
        y=y,
        metrics={"AUROC": AUROCMetric()},
    )

    path = tmp_path / "evaluation_report.json"
    export_report(
        results=results,
        path=path,
        model="LODA",
        dataset="synthetic",
    )

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["metadata"]["model"] == "LODA"
    assert document["metadata"]["dataset"] == "synthetic"
    assert document["results"]["AUROC"] == results["AUROC"]
    assert document["results"]["AUROC"] > 0.5
