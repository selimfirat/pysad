---
title: PySAD Streaming Anomaly Detection Playground
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: bsd-3-clause
short_description: Live streaming anomaly detection playground with PySAD.
---

# PySAD Interactive Playground

Explore real-time streaming anomaly detection with [PySAD](https://github.com/selimfirat/pysad) running directly in your browser via WebAssembly ([Pyodide](https://pyodide.org)):

- **Stream presets & CSV upload**: Test a 4-sensor synthetic drift stream, NAB's NYC Taxi demand series, ODDS Arrhythmia (274 features), or your own CSV (kept in browser memory).
- **Side-by-side detector comparison**: Race up to 3 detectors simultaneously (`LODA`, `HalfSpaceTrees`, `RobustRandomCutForest`, `SeasonalHybridESD`, `MedianAbsoluteDeviation`, `xStream`).
- **Live calibration & anomaly injection**: Compare `ConformalProbabilityCalibrator` and `SPOTProbabilityCalibrator`, adjust the alert threshold slider, and inject custom spikes into any feature.
- **Feature attribution**: Inspect `LODA.explain_partial()` contributions at any point in the stream.

## Run locally

```bash
rm -rf /tmp/pysad-space && mkdir -p /tmp/pysad-space/data
cp spaces/playground/index.html spaces/playground/engine.py spaces/playground/README.md /tmp/pysad-space/
cp -r examples/data/nab /tmp/pysad-space/data/
cp examples/data/arrhythmia.mat /tmp/pysad-space/data/
uv build --wheel --out-dir /tmp/pysad-space
python -c 'import json, pathlib; p = pathlib.Path("/tmp/pysad-space"); (p / "manifest.json").write_text(json.dumps({"wheel": sorted(x.name for x in p.glob("pysad-*.whl"))[-1]}))'
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
