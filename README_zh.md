<p align="center"><a href="https://github.com/selimfirat/pysad/blob/master/README.md">English</a> · <b>简体中文</b></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/selimfirat/pysad/master/docs/images/logo.svg" alt="PySAD" width="360">
</p>

<h3 align="center">一遍扫描，捕获每个离群点。</h3>

<p align="center">
  Python 流式异常检测库。PySAD 在每个数据点到达时立即为其打分，<br>
  提供 16 个流式检测器、经过校准的告警和流式评估。
</p>

<p align="center">
  <a href="https://pypi.org/project/pysad/"><img src="https://img.shields.io/pypi/v/pysad?style=flat-square&logo=pypi&logoColor=white&label=PyPI" alt="PyPI version"></a>
  <a href="https://anaconda.org/conda-forge/pysad"><img src="https://img.shields.io/conda/vn/conda-forge/pysad?style=flat-square&logo=condaforge&logoColor=white&label=conda-forge" alt="conda-forge version"></a>
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
  <a href="https://pysad.readthedocs.io/en/stable/"><b>文档</b></a> ·
  <a href="https://pysad.readthedocs.io/en/stable/getting_started.html">入门</a> ·
  <a href="https://pysad.readthedocs.io/en/stable/guides/index.html">操作指南</a> ·
  <a href="https://pysad.readthedocs.io/en/stable/benchmark.html">基准测试</a> ·
  <a href="https://pysad.readthedocs.io/en/stable/detectors.html">检测器</a> ·
  <a href="https://pysad.readthedocs.io/en/stable/api.html">API</a> ·
  <a href="https://pyodide.github.io/jupyterlite/lab/index.html?fromURL=https%3A//raw.githubusercontent.com/selimfirat/pysad/master/examples/nyc_taxi_alerts.ipynb">在线演示</a> ·
  <a href="https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb">在 Colab 中试用</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/selimfirat/pysad/master/docs/images/stream.svg" alt="三个传感器的读数从左向右流入。鲁棒随机切割森林在每个点到达时为其打分。共形校准在一个尖峰、一段短暂的平台和一个低谷处发出告警。" width="100%">
</p>

<p align="center"><sub>鲁棒随机切割森林（Robust Random Cut Forest）逐点为三个合成传感器的读数打分。共形校准把每个分数转换为它所超过的近期分数的比例，并在该值达到 0.995 时发出告警。</sub></p>

## 为什么选择 PySAD

批处理检测器必须先拿到整个数据集，才能为一个点打分。数据流不会一次给出整个数据集：数据点逐个到达，数据会发生漂移（其分布随时间变化），而且你必须在下一个点到达之前对当前点做出判断。PySAD 正是为这种场景而设计的。[Google Scholar](https://scholar.google.com/scholar?cites=3884618599898298487) 上有 60 多篇论文引用了它，GitHub 上有 [30 多个公开仓库](https://github.com/selimfirat/pysad/network/dependents) 在使用它。

<table>
<tr>
<td width="50%" valign="top">

#### 为数据流而生

每个检测器都通过 `fit_partial`、`score_partial` 和 `fit_score_partial` 一次学习并评分一个点。大多数检测器只保存固定大小的窗口、概要结构（sketch）或一组树，而不是整个数据流。

</td>
<td width="50%" valign="top">

#### 16 个来自已发表论文的检测器

从稳健统计方法（MAD、Seasonal Hybrid ESD）到基于森林、概要结构和投影的方法（Half-Space Trees、Robust Random Cut Forest、RS-Hash、xStream、LODA）。每个检测器都注明了出处论文，[检测器目录](https://pysad.readthedocs.io/en/stable/detectors.html) 列出了每个检测器的输入和内存占用。

</td>
</tr>
<tr>
<td valign="top">

#### 从分数到告警

共形校准器把原始分数转换为它所超过的近期分数的比例。这样，`0.99` 这样的阈值对每个检测器都具有相同的含义。

</td>
<td valign="top">

#### 在数据流上评估

流回放器（streamer）逐点回放数据，17 个带标签的基准数据集在首次使用时自动下载。AUROC、AUPR、精确率和召回率等指标在每个点之后更新。

</td>
</tr>
<tr>
<td valign="top">

#### 让批处理模型运行在数据流上

封装任意 [PyOD](https://github.com/yzhao062/pyod) 检测器，让它在近期数据点的滑动窗口上反复重新训练。

</td>
<td valign="top">

#### 完整的处理流水线

预处理器、随机投影、分数后处理器、集成器和在线统计量都可以与每个检测器配合使用。

</td>
</tr>
</table>

[PyOD](https://github.com/yzhao062/pyod) 等批处理离群点检测器在固定数据集上训练，要利用新数据就必须重新训练。PySAD 的检测器从每个点中学习，同时 PySAD 仍可以通过 `ReferenceWindowModel` 在数据流上运行 PyOD 模型。[River](https://github.com/online-ml/river) 等通用在线学习框架把异常检测当作众多模块之一。PySAD 只专注于流式异常检测：16 个检测器，以及围绕它们的校准、后处理、集成和评估。

## 安装

```bash
pip install pysad
```

或通过 Conda / Mamba / Pixi 安装：

```bash
conda install -c conda-forge pysad
```

PySAD 支持 Linux、macOS 和 Windows 上的 Python 3.10 至 3.14。执行安装后，所有检测器即可使用，无需安装任何可选依赖。

从 `master` 分支安装最新开发版本：

```bash
pip install git+https://github.com/selimfirat/pysad.git
```

如需以可编辑模式安装并包含开发依赖，请参阅[贡献指南](https://pysad.readthedocs.io/en/latest/contributing.html)。

## 快速开始

[![Open in JupyterLite](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://pyodide.github.io/jupyterlite/lab/index.html?fromURL=https%3A//raw.githubusercontent.com/selimfirat/pysad/master/examples/nyc_taxi_alerts.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb)

逐点为数据流打分，并在运行过程中评估检测器：

```python
import numpy as np
from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.utils import ArrayStreamer

# 一个包含 2,000 个点、4 个特征的数据流。每隔 50 个点有一个异常点。
rng = np.random.default_rng(42)
X = rng.normal(size=(2000, 4))
y = np.zeros(2000, dtype=int)
y[::50] = 1
X[y == 1, :2] += 3  # 在四个特征中的两个上平移异常点

model = LODA(random_state=0)
metric = AUROCMetric()

for x, label in ArrayStreamer().iter(X, y):
    score = model.fit_score_partial(x)  # 先为 x 打分，再从中学习
    metric.update(label, score)

print(f"AUROC: {metric.get():.3f}")  # 输出 AUROC: 0.960
```

### 把分数转换为告警

分数越高，表示该点越异常，但原始分数没有固定的尺度。共形校准器把每个分数转换为它所超过的近期分数的比例。接着上面的例子，当这个值接近 1 时发出告警：

```python
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

model = LODA(random_state=0)
calibrator = ConformalProbabilityCalibrator(window_size=500)

alerts = []
for t, x in enumerate(ArrayStreamer().iter(X)):
    calibrated = calibrator.fit_transform_partial(model.fit_score_partial(x))
    if t >= 200 and calibrated >= 0.99:  # 跳过短暂的预热阶段
        alerts.append(t)

# 输出：20 alerts, 15 of them true anomalies
print(f"{len(alerts)} alerts, {y[alerts].sum()} of them true anomalies")
```

### 处理你自己的数据流

检测器每次接收一个点的 NumPy 数组，因此任何循环都可以使用：CSV 读取器、套接字或消息队列的消费者。

```python
import json

for message in consumer:  # 例如一个 Kafka 消费者
    x = np.asarray(json.loads(message.value)["features"], dtype=float)
    score = model.fit_score_partial(x)
```

`ArrayStreamer` 和 `PandasStreamer` 可以把数组和 DataFrame 当作数据流回放，便于做实验。

### 在数据流上使用 PyOD 模型

先用 `pip install pyod` 安装 PyOD。

```python
from pyod.models.iforest import IForest
from pysad.models.integrations import ReferenceWindowModel

# 在最近 200 个点上训练批处理孤立森林，每 50 个点重新训练一次。
model = ReferenceWindowModel(IForest, window_size=200, sliding_size=50, initial_window_X=X[:100])
for x in ArrayStreamer().iter(X[100:]):
    score = model.fit_score_partial(x)
```

[入门教程](https://pysad.readthedocs.io/en/stable/getting_started.html) 讲解了每一步。[操作指南](https://pysad.readthedocs.io/en/stable/guides/index.html) 涵盖告警、评估、流水线、季节性序列、生产环境使用和自定义检测器。

## 检测器

### 从哪个检测器开始

| 你的数据 | 首选 | 原因 |
|---|---|---|
| 具有日周期或周周期的单变量序列 | `SeasonalHybridESD` | 去除周期成分。窗口中较早出现的异常对它影响很小 |
| 没有明显周期的单变量序列 | `MedianAbsoluteDeviation` | 窗口中的异常对它影响很小。它只使用近期的值，因此能跟随漂移 |
| 多个特征，初次尝试 | `LODA` | 速度快，参数少，并从每个点中学习 |
| 特征取值范围已知 | `HalfSpaceTrees` | 每个点的处理时间恒定，以上一个窗口为参照打分 |
| 特征很多 | `xStream` | 用哈希把数据点投影到更少的特征上 |
| 网络流量 | `KitNet` | 专为网络入侵检测设计。默认对前 100 个点输出 0 |
| 你已在使用的 PyOD 模型 | `ReferenceWindowModel` | 在滑动窗口上反复重新训练它 |

以上建议来自每种方法的设计，而不是基准测试排名。[基准测试](https://pysad.readthedocs.io/en/stable/benchmark.html) 给出了每个多变量检测器在 17 个数据集上的结果。请按照 [检测器选择指南](https://pysad.readthedocs.io/en/stable/guides/choose_detector.html) 中的方法，在你自己的数据上比较两到三个检测器。

### 全部检测器

**多变量**

| 类 | 方法 |
|---|---|
| `xStream` | xStream (Manzoor et al., KDD 2018) |
| `LODA` | LODA 轻量级在线异常检测器 (Pevný, *Machine Learning* 2016) |
| `HalfSpaceTrees` | Half-Space Trees (Tan et al., IJCAI 2011) |
| `RobustRandomCutForest` | Robust Random Cut Forest (Guha et al., ICML 2016) |
| `RSHash` | RS-Hash 子空间离群点检测 (Sathe & Aggarwal, ICDM 2016) |
| `IForestASD` | 滑动窗口上的孤立森林 (Ding & Fei, IFAC 2013) |
| `KitNet` | KitNET 自编码器集成 (Mirsky et al., NDSS 2018) |
| `ExactStorm` | Exact-STORM 基于距离的离群点检测 (Angiulli & Fassetti, CIKM 2007) |
| `LocalOutlierProbability` | 局部离群概率 (Kriegel et al., CIKM 2009) |
| `Inqmad` | InQMAD 量子测量密度估计 (Gallego-Mejia et al., ICDMW 2022) |

**单变量**

| 类 | 方法 |
|---|---|
| `SeasonalHybridESD` | Seasonal Hybrid ESD (Hochenbaum et al., 2017) |
| `SeasonalESD` | Seasonal ESD (Hochenbaum et al., 2017) |
| `KNNCAD` | 共形 k 近邻异常检测 (Burnaev & Ishimtsev, 2016) |
| `RelativeEntropy` | 基于窗口的相对熵 (Wang et al., IM 2011) |
| `MedianAbsoluteDeviation` | 滑动窗口上的中位数绝对偏差 (Hochenbaum et al., 2017) |
| `StandardAbsoluteDeviation` | 在线 3-sigma 规则 (Hochenbaum et al., 2017) |

`ReferenceWindowModel` 和 `OneFitModel` 可以在数据流上运行任意 [PyOD](https://github.com/yzhao062/pyod) 检测器。[检测器目录](https://pysad.readthedocs.io/en/stable/detectors.html) 列出了每个检测器的输入、配置和内存占用，[API 参考](https://pysad.readthedocs.io/en/stable/api.html) 列出了所有参数。

## 基准测试

PySAD 在带标签的数据上对其检测器进行基准测试。10 个多变量检测器在 [ODDS](http://odds.cs.stonybrook.edu/) 数据库的 17 个数据集以及 8 条存在漂移的真实数据流上运行。单变量检测器在 Numenta 异常基准（NAB）和 UCR 档案库的序列上运行。每个检测器在每个数据集上使用默认参数运行一次，并且在从每个点学习之前先为它打分（`fit_score_partial`）。

[基准测试结果](https://pysad.readthedocs.io/en/stable/benchmark.html) 页面展示了全部结果。[评估方法](https://pysad.readthedocs.io/en/stable/benchmarks/methodology.html) 说明了实验步骤、指标以及结果的局限性，[数据集](https://pysad.readthedocs.io/en/stable/benchmarks/datasets.html) 页面介绍了所用数据。要运行基准测试，请按照 [运行基准测试](https://pysad.readthedocs.io/en/stable/benchmarks/run.html) 中的说明使用 `python benchmarks/benchmark.py`。对每个修改检测器的拉取请求，CI 都会运行一个简化版基准测试，并列出发生变化的结果。

## 检测器之外

| 阶段 | PySAD 提供的内容 |
|---|---|
| 预处理 | `InstanceStandardScaler`、`InstanceUnitNormScaler`、`SeasonalTrendDecomposer`、`ModifiedSTLResidualTransformer` |
| 投影 | `StreamhashProjector`、`GaussianRandomProjector`、`SparseRandomProjector` |
| 后处理 | 对分数计算累计和窗口化的平均值、最大值、中位数和 z 分数 |
| 校准 | `ConformalProbabilityCalibrator`、`GaussianTailProbabilityCalibrator` |
| 集成 | 平均、最大、中位数、最大值的平均（average-of-maximum）和平均值的最大（maximum-of-average）分数集成器 |
| 评估 | `AUROCMetric`、`AUPRMetric`、`PrecisionMetric`、`RecallMetric`、`WindowedMetric` |
| 流式数据 | `ArrayStreamer`、`PandasStreamer`，以及提供 17 个带标签 ODDS 基准数据集（首次使用时下载）的 `Data` |
| 统计量 | 在线均值、方差、中位数、最小值、最大值、总和与计数 |

## 参与贡献

欢迎提交 issue 和拉取请求，我们会尽量在几天内回复。可以在 [Discussions](https://github.com/selimfirat/pysad/discussions) 中提问和分享你的项目。如果是第一次贡献，可以从 [good first issue](https://github.com/selimfirat/pysad/labels/good%20first%20issue) 中挑选一个。[贡献指南](https://pysad.readthedocs.io/en/stable/contributing.html) 列出了从首次克隆仓库到提交拉取请求的全部步骤。

PySAD 遵循 [语义化版本](https://semver.org/) 和 [主干开发](https://trunkbaseddevelopment.com/)。每项改动都通过一个短期分支和一个经过 CI 检查的拉取请求合入 `master`，并且 CI 会运行每个示例。
<p align="center">
  <a href="https://github.com/selimfirat/pysad/graphs/contributors"><img src="https://contrib.rocks/image?repo=selimfirat/pysad" alt="贡献者"></a>
</p>

感谢每一位报告或修复缺陷、改进文档的人。

## 引用 PySAD

如果你在学术论文中使用了 PySAD，请引用以下论文：

```bibtex
@article{pysad,
  title={PySAD: A Streaming Anomaly Detection Framework in Python},
  author={Yilmaz, Selim F and Kozat, Suleyman S},
  journal={arXiv preprint arXiv:2009.02572},
  year={2020}
}
```

要引用软件的特定版本，请使用其 Zenodo DOI。DOI [10.5281/zenodo.22983312](https://doi.org/10.5281/zenodo.22983312) 始终指向最新版本，该页面也列出了每个版本各自的 DOI。

## 许可证

PySAD 是基于 [BSD 3-Clause 许可证](https://github.com/selimfirat/pysad/blob/master/LICENSE) 的自由软件。

---

<p align="center">
  <img src="https://raw.githubusercontent.com/selimfirat/pysad/master/docs/images/tripod.svg" alt="PySAD 的吉祥物 Tripod：一条三脚鱼，用三根长鳍静立在海底，等待水流带来的东西。" width="96">
</p>

<p align="center">
  如果 PySAD 对你有用，点一个 Star 可以帮助更多人发现它。<br>
  如需获取新检测器和新版本的通知，请选择 <b>Watch → Custom → Releases</b>。
</p>
