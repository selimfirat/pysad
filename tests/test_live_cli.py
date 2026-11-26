import json

from pysad.live.cli import build_parser, main


def test_build_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["--source", "synthetic"])
    assert args.source == "synthetic"
    assert args.max_points is None
    assert args.web is False
    assert args.json_output is False
    assert args.detector == "loda"
    assert args.calibrator == "conformal"
    assert args.threshold == 0.95
    assert args.port == 8080
    assert args.host == "127.0.0.1"


def test_build_parser_web_and_json_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--web", "--json-output", "--port", "9090", "--host", "0.0.0.0"])
    assert args.web is True
    assert args.json_output is True
    assert args.port == 9090
    assert args.host == "0.0.0.0"


def test_main_json_output_emits_one_valid_line_per_point(tmp_path, capsys) -> None:
    path = tmp_path / "stream.csv"
    path.write_text("f1,f2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n", encoding="utf-8")
    argv = [
        "--source",
        str(path),
        "--max-points",
        "3",
        "--detector",
        "loda",
        "--calibrator",
        "conformal",
        "--json-output",
    ]
    assert main(argv) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 3
    for index, line in enumerate(lines):
        payload = json.loads(line)
        assert payload["step"] == index
        assert {
            "step",
            "features",
            "score",
            "calibrated_score",
            "lower_bound",
            "upper_bound",
            "alert",
            "label",
        } <= set(payload)
        assert isinstance(payload["features"], list)
        assert isinstance(payload["alert"], bool)
