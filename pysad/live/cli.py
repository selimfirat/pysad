"""Command-line interface for pysad-live streaming anomaly detection dashboard."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Any

from .terminal import (
    TerminalConfig,
    TerminalDashboard,
    compute_conformal_bounds,
    compute_feature_bounds,
    create_data_stream,
    get_calibrator,
    get_detector,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for pysad-live."""
    parser = argparse.ArgumentParser(
        prog="pysad-live",
        description="Live streaming anomaly detection terminal and web dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic data, LODA detector, and conformal calibration
  pysad-live --source synthetic --detector loda --calibrator conformal

  # Run with NYC Taxi data, HalfSpaceTrees, and SPOT calibration
  pysad-live --source nyc_taxi --detector half_space_trees --calibrator spot

  # Run with custom CSV file
  pysad-live --source data.csv --detector rs_hash --calibrator conformal --threshold 0.99

  # Start web dashboard on port 8080
  pysad-live --source synthetic --web --port 8080

  # Run headless (no terminal UI) with JSON output
  pysad-live --source synthetic --detector loda --calibrator conformal --json-output

  # Run with a semicolon-delimited, headerless CSV file
  pysad-live --source data.csv --csv-delimiter ";" --no-csv-header --detector rs_hash
        """,
    )

    # Data source options
    source_group = parser.add_argument_group("Data Source")
    source_group.add_argument(
        "--source",
        type=str,
        default="synthetic",
        help="Data source: synthetic, nyc_taxi, arrhythmia, creditcard, or path to CSV file (default: synthetic)",
    )
    source_group.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Maximum number of points to process (default: all)",
    )
    source_group.add_argument(
        "--csv-target",
        type=str,
        default=None,
        help="Target column name for CSV files (default: auto-detect)",
    )
    source_group.add_argument(
        "--csv-timestamp",
        type=str,
        default=None,
        help="Timestamp column name for CSV files (default: none)",
    )
    source_group.add_argument(
        "--csv-delimiter",
        type=str,
        default=",",
        help="Single-character field delimiter for CSV files (default: ',')",
    )
    source_group.add_argument(
        "--csv-header",
        action="store_true",
        default=True,
        dest="csv_header",
        help="CSV file has a header row to skip (default: True)",
    )
    source_group.add_argument(
        "--no-csv-header",
        action="store_false",
        dest="csv_header",
        help="CSV file has no header row to skip",
    )

    # Detector options
    detector_group = parser.add_argument_group("Detector")
    detector_group.add_argument(
        "--detector",
        type=str,
        default="loda",
        help=(
            "Detector: loda, half_space_trees, rs_hash, local_outlier_probability, "
            "robust_random_cut_forest, knncad, smart_sifter, seasonal_hybrid_esd, "
            "xstream, median_absolute_deviation (default: loda)"
        ),
    )
    detector_group.add_argument(
        "--detector-seed",
        type=int,
        default=42,
        help="Random seed for detector (default: 42)",
    )
    detector_group.add_argument(
        "--detector-kwargs",
        type=str,
        default="{}",
        help="Additional detector arguments as JSON string (default: {})",
    )

    # Calibrator options
    calibrator_group = parser.add_argument_group("Calibrator")
    calibrator_group.add_argument(
        "--calibrator",
        type=str,
        default="conformal",
        help=(
            "Calibrator: conformal, spot, conformal_tsfm, gaussian_tail, "
            "anomaly_likelihood, group_conditional_conformal (default: conformal)"
        ),
    )
    calibrator_group.add_argument(
        "--calibrator-window",
        type=int,
        default=300,
        help="Calibration window size (default: 300)",
    )
    calibrator_group.add_argument(
        "--calibrator-kwargs",
        type=str,
        default="{}",
        help="Additional calibrator arguments as JSON string (default: {})",
    )

    # Alert threshold
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Alert threshold for calibrated score (default: 0.95)",
    )

    # Display options
    display_group = parser.add_argument_group("Display")
    display_group.add_argument(
        "--width",
        type=int,
        default=80,
        help="Terminal width (default: 80)",
    )
    display_group.add_argument(
        "--height",
        type=int,
        default=24,
        help="Terminal height (default: 24)",
    )
    display_group.add_argument(
        "--history",
        type=int,
        default=100,
        help="History length for sparklines (default: 100)",
    )
    display_group.add_argument(
        "--refresh-rate",
        type=float,
        default=0.1,
        help="Refresh rate in seconds (default: 0.1)",
    )
    display_group.add_argument(
        "--max-features",
        type=int,
        default=4,
        help="Maximum number of features to display (default: 4)",
    )
    display_group.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide ground truth labels",
    )
    display_group.add_argument(
        "--no-ribbon",
        action="store_true",
        help="Hide conformal ribbon",
    )

    # Web dashboard options
    web_group = parser.add_argument_group("Web Dashboard")
    web_group.add_argument(
        "--web",
        action="store_true",
        help="Start web dashboard server",
    )
    web_group.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Web server port (default: 8080)",
    )
    web_group.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web server host (default: 127.0.0.1)",
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--json-output",
        action="store_true",
        help="Output JSON lines instead of terminal UI",
    )
    output_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log alerts to file (JSON lines)",
    )

    return parser


# Calibrators that support window_size parameter
_CALIBRATORS_WITH_WINDOW = {
    "conformal",
    "conformal_tsfm",
    "gaussian_tail",
    "anomaly_likelihood",
    "group_conditional_conformal",
}


def run_terminal_dashboard(args: argparse.Namespace) -> int:
    """Run the terminal dashboard."""
    import json

    # Parse additional kwargs
    detector_kwargs = json.loads(args.detector_kwargs)
    calibrator_kwargs = json.loads(args.calibrator_kwargs)

    # Add seed to detector kwargs if not provided
    if "random_state" not in detector_kwargs:
        detector_kwargs["random_state"] = args.detector_seed

    # Add window_size to calibrator kwargs if supported and not provided
    calibrator_name = args.calibrator.lower().replace("-", "_")
    if calibrator_name in _CALIBRATORS_WITH_WINDOW and "window_size" not in calibrator_kwargs:
        calibrator_kwargs["window_size"] = args.calibrator_window

    # Prepare CSV kwargs
    csv_kwargs = {}
    if args.csv_target:
        csv_kwargs["target_column"] = args.csv_target
    if args.csv_timestamp:
        csv_kwargs["timestamp_column"] = args.csv_timestamp
    csv_kwargs["delimiter"] = args.csv_delimiter
    csv_kwargs["has_header"] = args.csv_header

    # Compute feature bounds for detectors that need them
    feature_mins = None
    feature_maxes = None
    if args.detector.lower().replace("-", "_") in ("half_space_trees", "rs_hash"):
        feature_mins, feature_maxes = compute_feature_bounds(
            args.source,
            max_points=args.max_points,
            **csv_kwargs,
        )

    # Create detector and calibrator
    detector = get_detector(
        args.detector,
        feature_mins=feature_mins,
        feature_maxes=feature_maxes,
        **detector_kwargs,
    )
    calibrator = get_calibrator(args.calibrator, **calibrator_kwargs)

    # Create data stream
    stream = create_data_stream(args.source, max_points=args.max_points, **csv_kwargs)

    # Configure terminal dashboard
    config = TerminalConfig(
        width=args.width,
        height=args.height,
        max_features=args.max_features,
        history_length=args.history,
        refresh_rate=args.refresh_rate,
        show_labels=not args.no_labels,
        show_ribbon=not args.no_ribbon,
    )

    dashboard = TerminalDashboard(config)
    dashboard.set_metadata(
        detector_name=args.detector,
        calibrator_name=args.calibrator,
        threshold=args.threshold,
    )
    dashboard.start_timer()

    # Set up signal handler for graceful shutdown
    shutdown = {"flag": False}

    def signal_handler(signum: int, frame: Any) -> None:
        shutdown["flag"] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Open log file if specified
    log_file = None
    if args.log_file:
        log_file = open(args.log_file, "w")

    try:
        for x, label in stream:
            if shutdown["flag"]:
                break

            # Score the point
            score = detector.fit_score_partial(x)
            calibrated = calibrator.fit_transform_partial(score)

            # Compute conformal bounds
            lower, upper = compute_conformal_bounds(calibrator, score)

            # Check alert
            alert = calibrated > args.threshold

            # Update dashboard
            dashboard.update(
                feature_vector=x,
                score=float(score),
                calibrated_score=float(calibrated),
                lower_bound=lower,
                upper_bound=upper,
                alert=alert,
                label=label,
            )

            # Log to file if specified
            if log_file:
                log_entry = {
                    "step": dashboard.state.step,
                    "timestamp": time.time(),
                    "features": x.tolist(),
                    "score": float(score),
                    "calibrated_score": float(calibrated),
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "alert": alert,
                    "label": label,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

            # Control refresh rate
            time.sleep(args.refresh_rate)

    except KeyboardInterrupt:
        pass
    finally:
        dashboard.cleanup()
        if log_file:
            log_file.close()

    return 0


def run_web_dashboard(args: argparse.Namespace) -> int:
    """Run the web dashboard server."""
    # Import here to avoid dependency issues if not using web mode
    try:
        from .server import run_server
    except ImportError as e:
        print(f"Error: Web dashboard requires additional dependencies: {e}", file=sys.stderr)
        return 1

    return run_server(args)


def run_json_output(args: argparse.Namespace) -> int:
    """Run headless with JSON output."""
    import json

    # Parse additional kwargs
    detector_kwargs = json.loads(args.detector_kwargs)
    calibrator_kwargs = json.loads(args.calibrator_kwargs)

    if "random_state" not in detector_kwargs:
        detector_kwargs["random_state"] = args.detector_seed

    calibrator_name = args.calibrator.lower().replace("-", "_")
    if calibrator_name in _CALIBRATORS_WITH_WINDOW and "window_size" not in calibrator_kwargs:
        calibrator_kwargs["window_size"] = args.calibrator_window

    # Prepare CSV kwargs
    csv_kwargs = {}
    if args.csv_target:
        csv_kwargs["target_column"] = args.csv_target
    if args.csv_timestamp:
        csv_kwargs["timestamp_column"] = args.csv_timestamp
    csv_kwargs["delimiter"] = args.csv_delimiter
    csv_kwargs["has_header"] = args.csv_header

    # Compute feature bounds for detectors that need them
    feature_mins = None
    feature_maxes = None
    if args.detector.lower().replace("-", "_") in ("half_space_trees", "rs_hash"):
        feature_mins, feature_maxes = compute_feature_bounds(
            args.source,
            max_points=args.max_points,
            **csv_kwargs,
        )

    # Create detector and calibrator
    detector = get_detector(
        args.detector,
        feature_mins=feature_mins,
        feature_maxes=feature_maxes,
        **detector_kwargs,
    )
    calibrator = get_calibrator(args.calibrator, **calibrator_kwargs)

    stream = create_data_stream(args.source, max_points=args.max_points, **csv_kwargs)

    step = 0
    for x, label in stream:
        score = detector.fit_score_partial(x)
        calibrated = calibrator.fit_transform_partial(score)
        lower, upper = compute_conformal_bounds(calibrator, score)
        alert = calibrated > args.threshold

        output = {
            "step": step,
            "features": x.tolist(),
            "score": float(score),
            "calibrated_score": float(calibrated),
            "lower_bound": lower,
            "upper_bound": upper,
            "alert": alert,
            "label": label,
        }
        print(json.dumps(output))
        sys.stdout.flush()
        step += 1

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for pysad-live CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.json_output:
        return run_json_output(args)
    elif args.web:
        return run_web_dashboard(args)
    else:
        return run_terminal_dashboard(args)


if __name__ == "__main__":
    sys.exit(main())
