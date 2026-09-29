<p align="center">
  <img src="https://raw.githubusercontent.com/selimfirat/pysad/master/docs/logo.png" alt="PySAD" width="200">
</p>

<h3 align="center">Streaming anomaly detection in Python</h3>

<p align="center">
  Score each data point the moment it arrives, with 16 online detectors,<br>
  calibrated alerts and streaming evaluation.
</p>

<p align="center">
  <a href="https://pypi.org/project/pysad/"><img src="https://img.shields.io/pypi/v/pysad?style=flat-square&logo=pypi&logoColor=white&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/pysad/"><img src="https://img.shields.io/pypi/pyversions/pysad?style=flat-square&logo=python&logoColor=white&label=Python" alt="Python versions"></a>
  <a href="https://github.com/selimfirat/pysad/blob/master/LICENSE"><img src="https://img.shields.io/github/license/selimfirat/pysad?style=flat-square&label=license" alt="License"></a>
  <a href="https://doi.org/10.5281/zenodo.22983312"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22983312-1682D4?style=flat-square" alt="DOI"></a>
  <br>
  <a href="https://github.com/selimfirat/pysad/actions/workflows/coverage.yml"><img src="https://img.shields.io/github/actions/workflow/status/selimfirat/pysad/coverage.yml?branch=master&style=flat-square&logo=githubactions&logoColor=white&label=tests" alt="Tests"></a>
  <a href="https://dev.azure.com/selimfirat/pysad/_build/latest?definitionId=2&branchName=master"><img src="https://img.shields.io/azure-devops/build/selimfirat/pysad/2/master?style=flat-square&logo=azurepipelines&label=Azure%20Pipelines" alt="Azure Pipelines"></a>
  <a href="https://circleci.com/gh/selimfirat/pysad"><img src="https://img.shields.io/circleci/build/github/selimfirat/pysad/master?style=flat-square&logo=circleci&label=CircleCI" alt="CircleCI"></a>
  <a href="https://results.pre-commit.ci/latest/github/selimfirat/pysad/master"><img src="https://results.pre-commit.ci/badge/github/selimfirat/pysad/master.svg" alt="pre-commit.ci status"></a>
  <a href="https://coveralls.io/github/selimfirat/pysad?branch=master"><img src="https://img.shields.io/coverallsCoverage/github/selimfirat/pysad?branch=master&style=flat-square&logo=coveralls&label=coverage" alt="Coverage"></a>
  <a href="https://pysad.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/pysad?style=flat-square&logo=readthedocs&logoColor=white&label=docs" alt="Documentation"></a>
  <a href="https://github.com/selimfirat/pysad/commits/master"><img src="https://img.shields.io/github/last-commit/selimfirat/pysad?style=flat-square&label=last%20commit" alt="Last commit"></a>
</p>

<p align="center">
  <a href="https://pysad.readthedocs.io/en/latest/"><b>Documentation</b></a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#detectors">Detectors</a> ·
  <a href="https://github.com/selimfirat/pysad/tree/master/examples">Examples</a> ·
  <a href="https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb">Try it in Colab</a> ·
  <a href="#citing-pysad">Cite</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/selimfirat/pysad/master/docs/images/stream.svg" alt="Three sensor readings stream in from left to right. A Robust Random Cut Forest scores each point as it arrives, and conformal calibration raises alerts at a spike, a level shift and a dip." width="100%">
</p>

<p align="center"><sub>A Robust Random Cut Forest scores three sensors one point at a time, and a conformal p-value turns its scores into alerts. Synthetic data.</sub></p>

## Why pysad

Batch detectors assume you have the whole dataset. On a stream you don't: points arrive one at a time, the data drifts, and a decision is needed before the next point lands. pysad is built for that setting. It is cited in more than 60 publications on [Google Scholar](https://scholar.google.com/scholar?q=%22PySAD%3A+A+Streaming+Anomaly+Detection+Framework+in+Python%22) and used by [30+ public repositories](https://github.com/selimfirat/pysad/network/dependents) on GitHub.

<table>
<tr>
<td width="50%" valign="top">

**Built for streams**<br>Every detector learns and scores one instance at a time with `fit_partial`, `score_partial` and `fit_score_partial`. Most keep a fixed-size window, sketch or set of trees instead of the whole history.

</td>
<td width="50%" valign="top">

**16 detectors from the literature**<br>From robust statistics (MAD, 3-sigma, Seasonal Hybrid ESD) to forests, sketches and projections (Half-Space Trees, Robust Random Cut Forest, RS-Hash, xStream, LODA). Each one cites its paper.

</td>
</tr>
<tr>
<td valign="top">

**From scores to alerts**<br>Conformal and Gaussian-tail calibrators turn raw scores into p-values, so a threshold such as `p <= 0.01` means the same thing for every detector.

</td>
<td valign="top">

**Evaluate as you stream**<br>Stream simulators replay data one instance at a time, and metrics (AUROC, AUPR, precision, recall, and a windowed variant for drifting streams) update after every point.

</td>
</tr>
<tr>
<td valign="top">

**Your batch models, on a stream**<br>Wrap any [PyOD](https://github.com/yzhao062/pyod) detector to refit on a sliding reference window.

</td>
<td valign="top">

**A complete pipeline**<br>Preprocessors, random projections, score postprocessors, ensemblers and running statistics compose with any detector.

</td>
</tr>
</table>

**How pysad differs.** Batch outlier detectors such as [PyOD](https://github.com/yzhao062/pyod) are fitted on a fixed dataset and have to be refitted to take in new data. pysad detectors update with every point, and pysad can still run PyOD models on a stream through `ReferenceWindowModel`. General online-learning frameworks such as [River](https://github.com/online-ml/river) treat anomaly detection as one module among many. pysad is dedicated to it: 16 streaming detectors, with calibration, postprocessing, ensembling and evaluation built around them.

## Installation

```bash
pip install pysad
```

pysad supports Python 3.10+ on Linux, macOS and Windows. Some detectors need an optional extra:

| Extra | Enables | Install |
|---|---|---|
| `rrcf` | `RobustRandomCutForest` | `pip install "pysad[rrcf]"` |
| `xStream` | `xStream` | `pip install "pysad[xStream]"` |
| `slop` | `LocalOutlierProbability` | `pip install "pysad[slop]"` |
| `inqmad` | `Inqmad` (uses JAX) | `pip install "pysad[inqmad]"` |
| `pandas` | `PandasStreamer` | `pip install "pysad[pandas]"` |

Install everything with `pip install "pysad[rrcf,xStream,slop,inqmad,pandas]"`.

## Quick start

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb)

Score a stream point by point and evaluate the detector as it goes:

```python
import numpy as np
from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.utils import ArrayStreamer

# A stream of 2,000 points with 4 features; every 50th point is an anomaly.
rng = np.random.default_rng(42)
X = rng.normal(size=(2000, 4))
y = np.zeros(2000, dtype=int)
y[::50] = 1
X[y == 1, :2] += 3  # anomalies are shifted in two of the four features

model = LODA()
metric = AUROCMetric()

for x, label in ArrayStreamer().iter(X, y):
    score = model.fit_score_partial(x)  # update the model with x, then score it
    metric.update(label, score)

print(f"AUROC: {metric.get():.3f}")  # about 0.95; LODA is randomized, so it varies a little
```

### Turn scores into alerts

Raw scores have no fixed scale. A conformal calibrator converts each score into a p-value: the share of recent scores at least as high. Continuing the example above, alert when it's small:

```python
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

model = LODA()
calibrator = ConformalProbabilityCalibrator(window_size=500)

for t, x in enumerate(ArrayStreamer().iter(X)):
    score = model.fit_score_partial(x)
    p_value = calibrator.fit_transform_partial(score)
    if t >= 200 and p_value <= 0.01:  # skip a short warm-up
        print(f"alert at t={t} (p={p_value:.3f})")
```

### Stream your own data

A detector takes one NumPy array per instance, so any loop works: a CSV reader, a socket, or a message queue consumer.

```python
import json

for message in consumer:  # for example, a Kafka consumer
    x = np.asarray(json.loads(message.value)["features"], dtype=float)
    score = model.fit_score_partial(x)
```

`ArrayStreamer` and `PandasStreamer` replay arrays and DataFrames as streams, which is handy for experiments.

### Use a PyOD model on a stream

```python
from pyod.models.iforest import IForest
from pysad.models.integrations import ReferenceWindowModel

# Refit a batch Isolation Forest on the latest 200 points, every 50 points.
model = ReferenceWindowModel(IForest, window_size=200, sliding_size=50,
                             initial_window_X=X[:100])
for x in ArrayStreamer().iter(X[100:]):
    score = model.fit_score_partial(x)
```

More examples, from ensembles to seasonal time series, are in [`examples/`](https://github.com/selimfirat/pysad/tree/master/examples) and the [Colab quick start](https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb).

## Detectors

### Which detector should I start with?

| Your data | Start with | Why |
|---|---|---|
| One metric with daily or weekly cycles | `SeasonalHybridESD` | Built for seasonal cloud metrics, and robust to earlier anomalies in the window |
| One metric without strong seasonality | `MedianAbsoluteDeviation` | A robust baseline (running median and MAD) that works with its defaults |
| Several features, first try | `LODA` | Lightweight and learns incrementally |
| Features with known value ranges | `HalfSpaceTrees` | Constant time per point over a sliding window |
| Many features, or features that appear and disappear | `xStream` | Designed for high-dimensional, feature-evolving streams |
| Network traffic | `KitNet` | Built for online network intrusion detection |
| A PyOD model you already trust | `ReferenceWindowModel` | Refits it on a sliding reference window |

These are starting points drawn from each method's design, not benchmark rankings. Compare two or three on your own data with the metrics in `pysad.evaluation`.

### All detectors

**Multivariate**

| Class | Method | Extra |
|---|---|---|
| `xStream` | xStream (Manzoor et al., KDD 2018) | `xStream` |
| `LODA` | Lightweight on-line detector of anomalies (Pevný, *Machine Learning* 2016) | |
| `HalfSpaceTrees` | Half-Space Trees (Tan et al., IJCAI 2011) | |
| `RobustRandomCutForest` | Robust Random Cut Forest (Guha et al., ICML 2016) | `rrcf` |
| `RSHash` | RS-Hash subspace outlier detection (Sathe & Aggarwal, ICDM 2016) | |
| `IForestASD` | Isolation Forest on sliding windows (Ding & Fei, IFAC 2013) | |
| `KitNet` | KitNET ensemble of autoencoders (Mirsky et al., NDSS 2018) | |
| `ExactStorm` | Exact-STORM distance-based outliers (Angiulli & Fassetti, CIKM 2007) | |
| `LocalOutlierProbability` | Local Outlier Probabilities (Kriegel et al., CIKM 2009) | `slop` |
| `Inqmad` | InQMAD quantum-measurement density (Gallego-Mejia et al., ICDMW 2022) | `inqmad` |

**Univariate**

| Class | Method | Extra |
|---|---|---|
| `SeasonalHybridESD` | Seasonal Hybrid ESD (Hochenbaum et al., 2017) | |
| `SeasonalESD` | Seasonal ESD (Hochenbaum et al., 2017) | |
| `KNNCAD` | Conformalized k-NN anomaly detection (Burnaev & Ishimtsev, 2016) | |
| `RelativeEntropy` | Relative entropy over windows (Wang et al., IM 2011) | |
| `MedianAbsoluteDeviation` | Running median absolute deviation (Hochenbaum et al., 2017) | |
| `StandardAbsoluteDeviation` | Running 3-sigma rule (Hochenbaum et al., 2017) | |

Plus `ReferenceWindowModel` and `OneFitModel`, which run any [PyOD](https://github.com/yzhao062/pyod) detector on a stream. Full references and parameters are in the [API documentation](https://pysad.readthedocs.io/en/latest/api.html).

## Beyond detectors

| Stage | What pysad provides |
|---|---|
| Preprocessing | `InstanceStandardScaler`, `InstanceUnitNormScaler`, `SeasonalTrendDecomposer`, `ModifiedSTLResidualTransformer` |
| Projection | `StreamhashProjector`, `GaussianRandomProjector`, `SparseRandomProjector` |
| Postprocessing | Running and windowed average, max, median and z-score of scores |
| Calibration | `ConformalProbabilityCalibrator`, `GaussianTailProbabilityCalibrator` |
| Ensembling | Average, maximum, median, average-of-maximum and maximum-of-average score ensemblers |
| Evaluation | `AUROCMetric`, `AUPRMetric`, `PrecisionMetric`, `RecallMetric`, `WindowedMetric` |
| Streaming data | `ArrayStreamer`, `PandasStreamer` |
| Statistics | Running mean, variance, median, min, max, sum and count trackers |

## Contributing

Issues and pull requests are welcome, and I aim to reply to both within a few days. Questions and show-and-tell go in [Discussions](https://github.com/selimfirat/pysad/discussions). New to the codebase? Start with a [good first issue](https://github.com/selimfirat/pysad/labels/good%20first%20issue), and see the [contributing guide](https://github.com/selimfirat/pysad/blob/master/.github/CONTRIBUTING.md).

pysad follows [semantic versioning](https://semver.org/) and [trunk-based development](https://trunkbaseddevelopment.com/): changes reach `master` through short-lived branches and pull requests checked by CI, and every example runs in CI.

<p align="center">
  <a href="https://github.com/selimfirat/pysad/graphs/contributors"><img src="https://contrib.rocks/image?repo=selimfirat/pysad" alt="Contributors"></a>
</p>

Thanks to everyone who has reported a bug, fixed one, or improved the docs.

## Citing pysad

If you use pysad in a scientific publication, please cite the paper:

```bibtex
@article{pysad,
  title={PySAD: A Streaming Anomaly Detection Framework in Python},
  author={Yilmaz, Selim F and Kozat, Suleyman S},
  journal={arXiv preprint arXiv:2009.02572},
  year={2020}
}
```

To cite a specific version of the software, use its Zenodo DOI: [10.5281/zenodo.22983312](https://doi.org/10.5281/zenodo.22983312) always resolves to the latest release, and each release has its own DOI listed there.

## License

pysad is released under the [BSD 3-Clause License](https://github.com/selimfirat/pysad/blob/master/LICENSE).

---

<p align="center">
  If pysad is useful to you, a star helps other people find it.<br>
  To hear about new detectors and releases, choose <b>Watch → Custom → Releases</b>.
</p>
