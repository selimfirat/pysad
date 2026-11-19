"""The README badges distinguish measured throughput and detector input types."""

import json

from scripts import generate_performance_badges as badges


def test_badges_report_median_throughput_and_separate_detector_counts(tmp_path, monkeypatch):
    results = tmp_path / "results.csv"
    results.write_text("ms_per_point\n1\n2\n4\n", encoding="utf-8")
    monkeypatch.setattr(badges, "RESULTS", results)
    monkeypatch.setattr(badges, "OUT_DIR", tmp_path / "badges")
    monkeypatch.setattr(
        badges.models,
        "__all__",
        [
            "ChangeFinder",
            "HTM",
            "LODA",
            "MIDAS",
            "ModelEnsemble",
            "NullModel",
            "PerfectModel",
            "RandomModel",
            "StreamPipeline",
        ],
    )

    badges.main()

    generated = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in badges.OUT_DIR.glob("*.json")
    }
    assert set(generated) == {
        "throughput.json",
        "univariate_detectors.json",
        "multivariate_detectors.json",
    }
    assert generated["throughput.json"]["label"] == "median throughput"
    assert generated["throughput.json"]["message"] == "500 points/s"
    assert generated["univariate_detectors.json"]["label"] == "univariate detectors"
    assert generated["univariate_detectors.json"]["message"] == "2"
    assert generated["multivariate_detectors.json"]["label"] == "multivariate detectors"
    assert generated["multivariate_detectors.json"]["message"] == "2"
