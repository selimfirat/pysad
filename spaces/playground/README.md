---
title: PySAD Streaming Anomaly Detection Playground
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
license: bsd-3-clause
short_description: Compare streaming anomaly detectors, calibrate alerts, and inspect explanations live.
---

# PySAD Interactive Playground

Explore real-time streaming anomaly detection with [PySAD](https://github.com/selimfirat/pysad):

- **Stream presets & CSV upload**: Test a 4-sensor synthetic drift stream, NAB's NYC Taxi demand series, ULB Credit Card Fraud transactions, or your own CSV.
- **Side-by-side detector comparison**: Race up to 3 detectors simultaneously (`LODA`, `HalfSpaceTrees`, `RobustRandomCutForest`, `SeasonalHybridESD`, `MedianAbsoluteDeviation`, `xStream`).
- **Live calibration & anomaly injection**: Compare `ConformalProbabilityCalibrator` and `SPOTProbabilityCalibrator`, adjust the alert threshold slider, and inject custom spikes into any feature.
- **Feature attribution**: Inspect `LODA.explain_partial()` contributions at any point in the stream.

## Run locally

```bash
pip install -r spaces/playground/requirements.txt
python spaces/playground/app.py
```

## Deploy to Hugging Face Spaces

```bash
hf auth login
hf repo create selimfirat/pysad-playground --repo-type space --space_sdk gradio --exist-ok
hf upload selimfirat/pysad-playground spaces/playground --repo-type=space
```
