<p align="center">
  <img src="https://raw.githubusercontent.com/selimfirat/pysad/master/docs/images/logo.svg" alt="PySAD" width="360">
</p>

<h3 align="center">One pass. Every outlier.</h3>

<p align="center">
  Streaming anomaly detection in Python. PySAD scores each data point when it arrives,<br>
  with 16 streaming detectors, calibrated alerts and streaming evaluation.
</p>

<p align="center">
  <a href="https://pypi.org/project/pysad/"><img src="https://img.shields.io/pypi/v/pysad?style=flat-square&logo=pypi&logoColor=white&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/pysad/"><img src="https://img.shields.io/pypi/pyversions/pysad?style=flat-square&logo=python&logoColor=white&label=Python" alt="Python versions"></a>
  <a href="https://github.com/selimfirat/pysad/blob/master/LICENSE"><img src="https://img.shields.io/github/license/selimfirat/pysad?style=flat-square&label=license" alt="License"></a>
  <a href="https://doi.org/10.5281/zenodo.22983312"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22983312-0741AD?style=flat-square" alt="DOI"></a>
  <br>
  <a href="https://github.com/selimfirat/pysad/actions/workflows/coverage.yml"><img src="https://img.shields.io/github/actions/workflow/status/selimfirat/pysad/coverage.yml?branch=master&style=flat-square&logo=githubactions&logoColor=white&label=tests" alt="Tests"></a>
  <a href="https://dev.azure.com/selimfirat/pysad/_build/latest?definitionId=2&branchName=master"><img src="https://img.shields.io/azure-devops/build/selimfirat/pysad/2/master?style=flat-square&logo=azurepipelines&label=Azure%20Pipelines" alt="Azure Pipelines"></a>
  <a href="https://circleci.com/gh/selimfirat/pysad"><img src="https://img.shields.io/circleci/build/github/selimfirat/pysad/master?style=flat-square&logo=circleci&label=CircleCI" alt="CircleCI"></a>
  <a href="https://results.pre-commit.ci/latest/github/selimfirat/pysad/master"><img src="https://results.pre-commit.ci/badge/github/selimfirat/pysad/master.svg" alt="pre-commit.ci status"></a>
  <a href="https://coveralls.io/github/selimfirat/pysad?branch=master"><img src="https://img.shields.io/coverallsCoverage/github/selimfirat/pysad?branch=master&style=flat-square&logo=coveralls&label=coverage" alt="Coverage"></a>
  <a href="https://pysad.readthedocs.io/en/stable/"><img src="https://img.shields.io/readthedocs/pysad?style=flat-square&logo=readthedocs&logoColor=white&label=docs" alt="Documentation"></a>
  <a href="https://github.com/selimfirat/pysad/commits/master"><img src="https://img.shields.io/github/last-commit/selimfirat/pysad?style=flat-square&label=last%20commit" alt="Last commit"></a>
</p>

<p align="center">
  <a href="https://pysad.readthedocs.io/en/stable/"><b>Documentation</b></a> ·
  <a href="https://pysad.readthedocs.io/en/stable/getting_started.html">Getting started</a> ·
  <a href="https://pysad.readthedocs.io/en/stable/guides/index.html">How-to guides</a> ·
  <a href="https://pysad.readthedocs.io/en/stable/benchmark.html">Benchmarks</a> ·
  <a href="https://pysad.readthedocs.io/en/stable/detectors.html">Detectors</a> ·
  <a href="https://pysad.readthedocs.io/en/stable/api.html">API</a> ·
  <a href="https://selimfirat.github.io/pysad/lab/index.html?path=nyc_taxi_alerts.ipynb">Live demo</a> ·
  <a href="https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb">Try it in Colab</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/selimfirat/pysad/master/docs/images/stream.svg" alt="Three sensor readings stream in from left to right. A Robust Random Cut Forest scores each point when the point arrives. Conformal calibration raises alerts at a spike, a short plateau and a dip." width="100%">
</p>

<p align="center"><sub>A Robust Random Cut Forest scores three synthetic sensors one point at a time. Conformal calibration converts each score into the fraction of recent scores that it exceeds, and raises an alert at 0.995.</sub></p>

## Why PySAD

A batch detector needs the whole dataset before it can score a point. A stream does not give you the whole dataset. Points arrive one at a time, the data drifts (its distribution changes over time), and you must decide about each point before the next point arrives. PySAD is built for this setting. More than 60 publications on [Google Scholar](https://scholar.google.com/scholar?cites=3884618599898298487) cite it, and [30+ public repositories](https://github.com/selimfirat/pysad/network/dependents) on GitHub use it.

<table>
<tr>
<td width="50%" valign="top">

#### Built for streams

Every detector learns from and scores one point at a time with `fit_partial`, `score_partial` and `fit_score_partial`. Most detectors keep a window, a sketch or a set of trees of fixed size instead of the whole stream.

</td>
<td width="50%" valign="top">

#### 16 detectors from published papers

From robust statistics (MAD, Seasonal Hybrid ESD) to forests, sketches and projections (Half-Space Trees, Robust Random Cut Forest, RS-Hash, xStream, LODA). Each detector cites its paper, and the [detector catalog](https://pysad.readthedocs.io/en/stable/detectors.html) lists its input and memory.

</td>
</tr>
<tr>
<td valign="top">

#### From scores to alerts

The conformal calibrator converts a raw score into the fraction of recent scores that it exceeds. A threshold such as `0.99` then has the same meaning for every detector.

</td>
<td valign="top">

#### Evaluate on the stream

Streamers replay data one point at a time, and 17 labeled benchmark datasets download on first use. Metrics such as AUROC, AUPR, precision and recall update after every point.

</td>
</tr>
<tr>
<td valign="top">

#### Your batch models on a stream

Wrap any [PyOD](https://github.com/yzhao062/pyod) detector to train it again on a sliding window of recent points.

</td>
<td valign="top">

#### A complete pipeline

Preprocessors, random projections, score postprocessors, ensemblers and running statistics work with every detector.

</td>
</tr>
</table>

Batch outlier detectors such as [PyOD](https://github.com/yzhao062/pyod) train on a fixed dataset and must train again to use new data. PySAD detectors learn from every point, and PySAD can still run PyOD models on a stream with `ReferenceWindowModel`. General online learning frameworks such as [River](https://github.com/online-ml/river) treat anomaly detection as one module among many. PySAD is only about streaming anomaly detection: 16 detectors, with calibration, postprocessing, ensembling and evaluation around them.

## Installation

```bash
pip install pysad
```

PySAD supports Python 3.10 to 3.14 on Linux, macOS and Windows. Every detector works after this command, with no optional extras.

## Quick start

[![Open in JupyterLite](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://selimfirat.github.io/pysad/lab/index.html?path=nyc_taxi_alerts.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb)

Score a stream point by point, and measure the detector while it runs:

```python
import numpy as np
from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.utils import ArrayStreamer

# A stream of 2,000 points with 4 features. Every 50th point is an anomaly.
rng = np.random.default_rng(42)
X = rng.normal(size=(2000, 4))
y = np.zeros(2000, dtype=int)
y[::50] = 1
X[y == 1, :2] += 3  # the code shifts the anomalies in two of the four features

model = LODA(random_state=0)
metric = AUROCMetric()

for x, label in ArrayStreamer().iter(X, y):
    score = model.fit_score_partial(x)  # score x, then learn from it
    metric.update(label, score)

print(f"AUROC: {metric.get():.3f}")  # prints AUROC: 0.960
```

### Turn scores into alerts

A higher score means a more anomalous point, but raw scores have no fixed scale. The conformal calibrator converts each score into the fraction of recent scores that it exceeds. Continuing the example above, raise an alert when this value is close to 1:

```python
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

model = LODA(random_state=0)
calibrator = ConformalProbabilityCalibrator(window_size=500)

alerts = []
for t, x in enumerate(ArrayStreamer().iter(X)):
    calibrated = calibrator.fit_transform_partial(model.fit_score_partial(x))
    if t >= 200 and calibrated >= 0.99:  # skip a short warm-up
        alerts.append(t)

# Prints: 20 alerts, 15 of them true anomalies
print(f"{len(alerts)} alerts, {y[alerts].sum()} of them true anomalies")
```

### Stream your own data

A detector takes one NumPy array per point, so any loop works: a CSV reader, a socket or a message queue consumer.

```python
import json

for message in consumer:  # for example, a Kafka consumer
    x = np.asarray(json.loads(message.value)["features"], dtype=float)
    score = model.fit_score_partial(x)
```

`ArrayStreamer` and `PandasStreamer` replay arrays and DataFrames as streams for experiments.

### Use a PyOD model on a stream

Install PyOD first with `pip install pyod`.

```python
from pyod.models.iforest import IForest
from pysad.models.integrations import ReferenceWindowModel

# Train a batch Isolation Forest on the last 200 points, again every 50 points.
model = ReferenceWindowModel(IForest, window_size=200, sliding_size=50, initial_window_X=X[:100])
for x in ArrayStreamer().iter(X[100:]):
    score = model.fit_score_partial(x)
```

The [getting started tutorial](https://pysad.readthedocs.io/en/stable/getting_started.html) explains each step. The [how-to guides](https://pysad.readthedocs.io/en/stable/guides/index.html) cover alerts, evaluation, pipelines, seasonal series, production use and custom detectors.

## Detectors

### Which detector to start with

| Your data | Start with | Why |
|---|---|---|
| One series of values with daily or weekly cycles | `SeasonalHybridESD` | Removes the cycle. Earlier anomalies in the window have little effect on it |
| One series of values without a strong cycle | `MedianAbsoluteDeviation` | Anomalies in the window have little effect on it. It uses only recent values, so it follows drift |
| Several features, first try | `LODA` | Fast, has few parameters and learns from every point |
| Features with known value ranges | `HalfSpaceTrees` | Constant time per point, scored against the previous window |
| Many features | `xStream` | Projects the points to fewer features with hashing |
| Network traffic | `KitNet` | Built for network intrusion detection. Gives 0 for the first 100 points by default |
| A PyOD model that you already use | `ReferenceWindowModel` | Trains it again on a sliding window |

These starting points come from the design of each method. They are not benchmark rankings. The [benchmark](https://pysad.readthedocs.io/en/stable/benchmark.html) gives the results of each multivariate detector on 17 datasets. Compare two or three detectors on your own data, as the [guide to choosing a detector](https://pysad.readthedocs.io/en/stable/guides/choose_detector.html) shows.

### All detectors

**Multivariate**

| Class | Method |
|---|---|
| `xStream` | xStream (Manzoor et al., KDD 2018) |
| `LODA` | Lightweight on-line detector of anomalies (Pevný, *Machine Learning* 2016) |
| `HalfSpaceTrees` | Half-Space Trees (Tan et al., IJCAI 2011) |
| `RobustRandomCutForest` | Robust Random Cut Forest (Guha et al., ICML 2016) |
| `RSHash` | RS-Hash subspace outlier detection (Sathe & Aggarwal, ICDM 2016) |
| `IForestASD` | Isolation Forest on sliding windows (Ding & Fei, IFAC 2013) |
| `KitNet` | KitNET ensemble of autoencoders (Mirsky et al., NDSS 2018) |
| `ExactStorm` | Exact-STORM distance-based outliers (Angiulli & Fassetti, CIKM 2007) |
| `LocalOutlierProbability` | Local Outlier Probabilities (Kriegel et al., CIKM 2009) |
| `Inqmad` | InQMAD quantum measurement density (Gallego-Mejia et al., ICDMW 2022) |

**Univariate**

| Class | Method |
|---|---|
| `SeasonalHybridESD` | Seasonal Hybrid ESD (Hochenbaum et al., 2017) |
| `SeasonalESD` | Seasonal ESD (Hochenbaum et al., 2017) |
| `KNNCAD` | Conformalized k-NN anomaly detection (Burnaev & Ishimtsev, 2016) |
| `RelativeEntropy` | Relative entropy over windows (Wang et al., IM 2011) |
| `MedianAbsoluteDeviation` | Median absolute deviation over a sliding window (Hochenbaum et al., 2017) |
| `StandardAbsoluteDeviation` | Running 3-sigma rule (Hochenbaum et al., 2017) |

`ReferenceWindowModel` and `OneFitModel` run any [PyOD](https://github.com/yzhao062/pyod) detector on a stream. The [detector catalog](https://pysad.readthedocs.io/en/stable/detectors.html) lists the input, setup and memory of each detector, and the [API reference](https://pysad.readthedocs.io/en/stable/api.html) lists every parameter.

## Benchmarks

PySAD benchmarks its detectors on labeled data. The 10 multivariate detectors run on 17 datasets from the [ODDS](http://odds.cs.stonybrook.edu/) repository and on 8 real streams with drift. The univariate detectors run on the series of the Numenta Anomaly Benchmark (NAB) and of the UCR archive. Each detector runs once on each dataset, with its default parameters, and scores each point before it learns from it (`fit_score_partial`).

The [benchmark results](https://pysad.readthedocs.io/en/stable/benchmark.html) page shows all the results. The [evaluation methodology](https://pysad.readthedocs.io/en/stable/benchmarks/methodology.html) explains the steps, the metrics and the limits of the results, and the [datasets](https://pysad.readthedocs.io/en/stable/benchmarks/datasets.html) page describes the data. To run the benchmark, use `python benchmarks/benchmark.py`, as [Run the benchmark](https://pysad.readthedocs.io/en/stable/benchmarks/run.html) shows. On each pull request that changes a detector, CI runs a short form of the benchmark and lists the results that moved.

## Beyond detectors

| Stage | What PySAD provides |
|---|---|
| Preprocessing | `InstanceStandardScaler`, `InstanceUnitNormScaler`, `SeasonalTrendDecomposer`, `ModifiedSTLResidualTransformer` |
| Projection | `StreamhashProjector`, `GaussianRandomProjector`, `SparseRandomProjector` |
| Postprocessing | Running and windowed average, max, median and z-score of scores |
| Calibration | `ConformalProbabilityCalibrator`, `GaussianTailProbabilityCalibrator` |
| Ensembling | Average, maximum, median, average-of-maximum and maximum-of-average score ensemblers |
| Evaluation | `AUROCMetric`, `AUPRMetric`, `PrecisionMetric`, `RecallMetric`, `WindowedMetric` |
| Streaming data | `ArrayStreamer`, `PandasStreamer`, and `Data` for 17 labeled ODDS benchmark datasets, downloaded on first use |
| Statistics | Running mean, variance, median, min, max, sum and count |

## Contributing

We welcome issues and pull requests, and we aim to reply within a few days. Ask questions and share your projects in [Discussions](https://github.com/selimfirat/pysad/discussions). For a first contribution, choose a [good first issue](https://github.com/selimfirat/pysad/labels/good%20first%20issue). The [contributing guide](https://pysad.readthedocs.io/en/stable/contributing.html) lists the steps from the first clone to an open pull request.

PySAD follows [semantic versioning](https://semver.org/) and [trunk-based development](https://trunkbaseddevelopment.com/). Each change reaches `master` through a short-lived branch and a pull request that CI checks, and CI runs every example.
<p align="center">
  <a href="https://github.com/selimfirat/pysad/graphs/contributors"><img src="https://contrib.rocks/image?repo=selimfirat/pysad" alt="Contributors"></a>
</p>

Thanks to everyone who has reported a bug, fixed one, or improved the docs.

## Citing PySAD

If you use PySAD in a scientific publication, cite the paper:

```bibtex
@article{pysad,
  title={PySAD: A Streaming Anomaly Detection Framework in Python},
  author={Yilmaz, Selim F and Kozat, Suleyman S},
  journal={arXiv preprint arXiv:2009.02572},
  year={2020}
}
```

To cite a specific version of the software, use its Zenodo DOI. The DOI [10.5281/zenodo.22983312](https://doi.org/10.5281/zenodo.22983312) always points to the latest release. That page also lists a separate DOI for each release.

## License

PySAD is free software under the [BSD 3-Clause License](https://github.com/selimfirat/pysad/blob/master/LICENSE).

---

<p align="center">
  <img src="https://raw.githubusercontent.com/selimfirat/pysad/master/docs/images/tripod.svg" alt="Tripod, the PySAD mascot: a tripodfish that stands still on three long fins and waits for what the current brings." width="96">
</p>

<p align="center">
  If PySAD is useful to you, a star helps other people find it.<br>
  To hear about new detectors and releases, choose <b>Watch → Custom → Releases</b>.
</p>
