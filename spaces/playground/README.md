---
title: PySAD Streaming Anomaly Detection Playground
emoji: 📡
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: bsd-3-clause
short_description: Live streaming anomaly detection playground with PySAD.
---

# PySAD Playground

Run prequential streaming anomaly detection with [PySAD](https://github.com/selimfirat/pysad) directly in the browser through WebAssembly ([Pyodide](https://pyodide.org)).

- **Stream presets and CSV upload**: Evaluate the 4-sensor synthetic drift stream, NAB NYC Taxi passenger series, ODDS Arrhythmia (274 features), or a user-supplied CSV processed locally in browser memory.
- **Benchmark-ranked detector comparison**: Compare 1 to 3 detectors simultaneously (`LocalOutlierProbability`, `RSHash`, `LODA`, `SmartSifter`, `RobustRandomCutForest`, `HalfSpaceTrees`, `KNNCAD`, `SeasonalHybridESD`), initialized with the top ODDS benchmark trio (`LocalOutlierProbability`, `RSHash`, `LODA`).
- **Probability calibration and anomaly injection**: Switch between `ConformalProbabilityCalibrator` and `SPOTProbabilityCalibrator`, adjust the alert threshold slider, and inject point spikes into any feature.
- **Live canvas**: Score new points in a worker and draw values, calibrated scores, and alert markers. Pause, resume, or reset the selected stream.
- **Feature attribution**: Inspect `LODA.explain_partial()` contributions at any stream step.

## Live stream

Select a stream, detectors, and calibration. Click **Start live** to score points in order.
Click **Pause live** to pause. Click **Resume live** to continue with the same model state.
Click **Reset live** to clear the chart. The next start creates fresh models.
If you change experiment settings, the live session stops. Start again to use the new settings.

The canvas targets 60 FPS through `requestAnimationFrame`. It shows the measured frame rate during playback.
A worker runs Pyodide and scores batches of eight points. The canvas keeps the latest 600 points.
The stream targets 60 points per second. Slow models can reduce this rate without blocking canvas rendering.
At the end of the selected stream, playback stops. Click **Start live** to run it again.
Frame rates depend on your browser, display, and device. Background tabs can pause rendering.

## Run locally

```bash
rm -rf /tmp/pysad-space && mkdir -p /tmp/pysad-space/data
cp spaces/playground/index.html spaces/playground/worker.js spaces/playground/engine.py spaces/playground/README.md /tmp/pysad-space/
cp -r examples/data/nab /tmp/pysad-space/data/
cp examples/data/arrhythmia.mat /tmp/pysad-space/data/
uv build --wheel --out-dir /tmp/pysad-space
python -c 'import json, pathlib; p = pathlib.Path("/tmp/pysad-space"); (p / "manifest.json").write_text(json.dumps({"wheel": sorted(x.name for x in p.glob("pysad-*.whl"))[-1]}))'
PYTHONPATH=spaces/playground:. python -c 'import pathlib, engine; pathlib.Path("/tmp/pysad-space/initial_snapshot.json").write_text(engine.build_initial_snapshot_json())'
python -m http.server 7860 --directory /tmp/pysad-space
```

## Deploy to Hugging Face Spaces

```bash
hf auth login
python -c '
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("selimfirat/pysad-playground", repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(repo_id="selimfirat/pysad-playground", folder_path="/tmp/pysad-space", repo_type="space")
'
```
