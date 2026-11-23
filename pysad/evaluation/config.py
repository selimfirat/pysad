from __future__ import annotations

from dataclasses import dataclass, field

from pysad.core.base_metric import BaseMetric
from pysad.core.base_model import BaseModel
from pysad.core.base_postprocessor import BasePostprocessor


@dataclass
class EvaluationConfig:
    """Shared configuration schema for a prequential evaluation run.

    Bundles a streaming detector, an optional calibrator or postprocessor, and
    the metrics to record. Detectors score each instance before they learn from
    it. Calibrators and postprocessors transform each score and learn from it in
    the same pass. The config is declarative: pass its fields to
    :func:`~pysad.evaluation.evaluate_stream`.

    Args:
        model (BaseModel): The streaming detector to evaluate.
        postprocessor (BasePostprocessor or None): Optional calibrator or
            postprocessor applied to each score (Default=None).
        metrics (dict[str, BaseMetric]): The metrics to update, by name (Default={}).
    """

    model: BaseModel
    postprocessor: BasePostprocessor | None = None
    metrics: dict[str, BaseMetric] = field(default_factory=dict)
