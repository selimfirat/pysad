from __future__ import annotations

import numbers
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel

__bibtex__ = """
@inproceedings{nie2023patchtst,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
@inproceedings{woo2024moirai,
  title={Unified Training of Universal Time Series Forecasting Transformers},
  author={Woo, Gerald and Liu, Chenghao and Kumar, Akshat and Xiong, Caiming and Savarese, Silvio and Sahoo, Doyen},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
"""


class FoundationFeatureDetector(BaseModel):
    """Anomaly detector using frozen foundation model embeddings.

    Uses a pre-trained foundation model (currently PatchTST) as a frozen feature
    extractor. Maintains a sliding window of recent observations, extracts
    embeddings using the foundation model's encoder, and scores anomalies based
    on the Mahalanobis distance of embeddings to the running mean of normal
    embeddings.

    The model operates in two phases:
    1. Warm-up: collects initial windows to establish the baseline embedding
       distribution (running mean and covariance).
    2. Detection: for each new window, extracts embeddings and computes the
       Mahalanobis distance to the running mean. Updates the running statistics
       with an exponential moving average.

    Args:
        model_name (str): Name of the foundation model to use. Currently supports
            ``'patchtst'``. (Default='patchtst')
        model_size (str): Size variant of the model. For PatchTST, this selects
            a preset configuration: ``'small'``, ``'base'``, or ``'large'``.
            (Default='base')
        context_length (int): Number of past instances to use as context for the
            foundation model. Must match the model's expected context length.
            (Default=512)
        patch_length (int): Length of each patch for PatchTST. (Default=16)
        patch_stride (int): Stride between patches for PatchTST. (Default=8)
        warmup_windows (int): Number of windows to collect during warm-up before
            producing anomaly scores. During warm-up, the score is 0.0.
            (Default=10)
        ema_alpha (float): Exponential moving average weight for updating the
            running mean and covariance of embeddings. Must be in (0, 1].
            (Default=0.01)
        random_state (int or None): Random seed for reproducibility.
            (Default=None)

    Attributes:
        embedding_dim_ (int): Dimensionality of the extracted embeddings.
        is_warmed_up_ (bool): Whether the warm-up phase is complete.

    Raises:
        ValueError: If ``model_name`` is not supported, or if parameters are
            invalid.

    Note:
        This implementation uses PatchTST from the Hugging Face Transformers
        library. The model weights are downloaded on first initialization.
        Moirai support requires the ``uni2ts`` package which is not yet on PyPI.

    References:
        :cite:`nie2023patchtst`
        :cite:`woo2024moirai`
    """

    _input_kind = "multivariate"

    def __init__(
        self,
        model_name: str = "patchtst",
        model_size: str = "base",
        context_length: int = 512,
        patch_length: int = 16,
        patch_stride: int = 8,
        warmup_windows: int = 10,
        ema_alpha: float = 0.01,
        random_state: int | None = None,
    ) -> None:
        if model_name != "patchtst":
            raise ValueError(
                f"Unsupported model_name: {model_name!r}. Currently only 'patchtst' is supported."
            )

        if model_size not in ("small", "base", "large"):
            raise ValueError(f"model_size must be 'small', 'base', or 'large', got {model_size!r}.")

        if not isinstance(context_length, numbers.Integral) or context_length < 1:
            raise ValueError(f"context_length must be a positive int, got {context_length!r}.")
        if not isinstance(patch_length, numbers.Integral) or patch_length < 1:
            raise ValueError(f"patch_length must be a positive int, got {patch_length!r}.")
        if not isinstance(patch_stride, numbers.Integral) or patch_stride < 1:
            raise ValueError(f"patch_stride must be a positive int, got {patch_stride!r}.")
        if patch_stride > patch_length:
            raise ValueError("patch_stride must not exceed patch_length.")
        if not isinstance(warmup_windows, numbers.Integral) or warmup_windows < 1:
            raise ValueError(f"warmup_windows must be a positive int, got {warmup_windows!r}.")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha!r}.")

        self.model_name: str = model_name
        self.model_size: str = model_size
        self.context_length: int = int(context_length)
        self.patch_length: int = int(patch_length)
        self.patch_stride: int = int(patch_stride)
        self.warmup_windows: int = int(warmup_windows)
        self.ema_alpha: float = float(ema_alpha)
        self.random_state: int | None = random_state

        # Internal state
        self._window: deque[NDArray[np.float64]] = deque(maxlen=self.context_length)
        self._windows_seen: int = 0
        self._embedding_dim: int | None = None
        self._ema_mean: NDArray[np.float64] | None = None
        self._ema_cov: NDArray[np.float64] | None = None
        self._model: Any = None  # The PatchTST model
        self._device: Any = None
        self._numpy_rng: Any = None
        self._torch_generator: Any = None

        # Import torch and transformers here to avoid import overhead if not used
        import torch
        from transformers import PatchTSTConfig, PatchTSTModel

        # Resolve random_state using sklearn's check_random_state (None draws from global state)
        self._numpy_rng = check_random_state(random_state)
        # Create torch generator with seed from numpy rng
        self._torch_generator = torch.Generator(device="cpu")
        torch_seed = self._numpy_rng.randint(0, 2**32 - 1)
        self._torch_generator.manual_seed(int(torch_seed))

        # Build model configuration based on size preset
        if model_size == "small":
            d_model = 64
            num_hidden_layers = 2
            num_attention_heads = 4
            ffn_dim = 128
        elif model_size == "base":
            d_model = 128
            num_hidden_layers = 4
            num_attention_heads = 8
            ffn_dim = 256
        else:  # large
            d_model = 256
            num_hidden_layers = 6
            num_attention_heads = 16
            ffn_dim = 512

        config = PatchTSTConfig(
            num_input_channels=1,  # Will be updated on first fit
            context_length=self.context_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            d_model=d_model,
            ffn_dim=ffn_dim,
            scaling="std",
            use_cls_token=False,
            pooling_type="mean",
        )

        # Use local generator for model initialization
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._torch_generator.initial_seed())
            self._model = PatchTSTModel(config)

        self._model.eval()  # Frozen model
        for param in self._model.parameters():
            param.requires_grad = False

        self._device = torch.device("cpu")
        self._model.to(self._device)

        # Embedding dimension: num_channels * d_model (after mean pooling over patches)
        # Will be finalized on first forward pass when we know num_channels

    @property
    def embedding_dim_(self) -> int:
        """Dimensionality of the extracted embeddings."""
        if self._embedding_dim is None:
            raise AttributeError(
                "Model not fitted yet. embedding_dim_ is available after first fit."
            )
        return self._embedding_dim

    @property
    def is_warmed_up_(self) -> bool:
        """Whether the warm-up phase is complete."""
        return self._windows_seen >= self.warmup_windows

    def _get_num_channels(self, X: NDArray[Any]) -> int:
        """Extract number of channels from input."""
        x = np.asarray(X, dtype=np.float64)
        if x.ndim == 1:
            return 1
        elif x.ndim == 2:
            return x.shape[1]
        else:
            raise ValueError(f"Expected 1D or 2D input, got shape {x.shape}")

    def _update_model_channels(self, num_channels: int) -> None:
        """Recreate model with correct number of input channels if needed."""
        import torch
        from transformers import PatchTSTConfig, PatchTSTModel

        if (
            hasattr(self._model.config, "num_input_channels")
            and self._model.config.num_input_channels == num_channels
        ):
            return

        # Recreate config with correct num_input_channels
        if self.model_size == "small":
            d_model = 64
            num_hidden_layers = 2
            num_attention_heads = 4
            ffn_dim = 128
        elif self.model_size == "base":
            d_model = 128
            num_hidden_layers = 4
            num_attention_heads = 8
            ffn_dim = 256
        else:  # large
            d_model = 256
            num_hidden_layers = 6
            num_attention_heads = 16
            ffn_dim = 512

        config = PatchTSTConfig(
            num_input_channels=num_channels,
            context_length=self.context_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            d_model=d_model,
            ffn_dim=ffn_dim,
            scaling="std",
            use_cls_token=False,
            pooling_type="mean",
        )

        # Use local generator for model initialization
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._torch_generator.initial_seed())
            self._model = PatchTSTModel(config)

        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.to(self._device)

    def _extract_embedding(self, window: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract embedding from a window using the frozen foundation model.

        Args:
            window: Array of shape (context_length, num_channels)

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        import torch

        # window shape: (context_length, num_channels)
        # Model expects: (batch_size, context_length, num_channels)
        num_channels = window.shape[1] if window.ndim == 2 else 1

        # Update model if channel count changed
        self._update_model_channels(num_channels)

        # Prepare input tensor
        if window.ndim == 1:
            window = window.reshape(-1, 1)
        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(self._device)  # (1, L, C)

        with torch.no_grad():
            output = self._model(past_values=x)
            # last_hidden_state shape: (batch_size, num_channels, num_patches, d_model)
            hidden = output.last_hidden_state  # (1, C, P, D)

        # Mean pool over patches: (1, C, D) -> (C * D,)
        pooled = hidden.mean(dim=2).squeeze(0).cpu().numpy()  # (C, D)
        embedding: NDArray[np.float64] = pooled.reshape(-1)  # (C * D,)

        if self._embedding_dim is None:
            self._embedding_dim = embedding.shape[0]
            # Initialize EMA statistics
            self._ema_mean = np.zeros(self._embedding_dim, dtype=np.float64)
            self._ema_cov = np.eye(self._embedding_dim, dtype=np.float64)

        return embedding

    def _update_ema(self, embedding: NDArray[np.float64]) -> None:
        """Update exponential moving average of mean and covariance."""
        alpha = self.ema_alpha
        if self._windows_seen == 0:
            assert self._ema_mean is not None
            assert self._ema_cov is not None
            self._ema_mean[:] = embedding
            # Covariance stays as identity initially
        else:
            assert self._ema_mean is not None
            assert self._ema_cov is not None
            delta = embedding - self._ema_mean
            self._ema_mean += alpha * delta
            # Update covariance using EMA of outer product
            # cov_new = (1-alpha) * cov_old + alpha * delta * delta^T
            self._ema_cov = (1.0 - alpha) * self._ema_cov + alpha * np.outer(delta, delta)

    def _mahalanobis_distance(self, embedding: NDArray[np.float64]) -> float:
        """Compute Mahalanobis distance to the running mean."""
        if self._ema_mean is None or self._ema_cov is None:
            return 0.0

        delta = embedding - self._ema_mean
        # Use diagonal approximation (normalized Euclidean) for numerical stability
        # var = diag(cov), avoid division by zero
        var = np.diag(self._ema_cov)
        var = np.maximum(var, 1e-6)
        dist_sq = np.sum((delta**2) / var)
        return float(np.sqrt(max(dist_sq, 0.0)))

    def fit_partial(self, X: NDArray[Any], y: int | None = None) -> FoundationFeatureDetector:
        """Updates the model with the next instance.

        Adds the instance to the sliding window. When the window is full,
        extracts embeddings and updates the running statistics.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int or None): Ignored (Default=None).

        Returns:
            object: The model itself.
        """
        x = np.asarray(X, dtype=np.float64).ravel()
        if not np.all(np.isfinite(x)):
            raise ValueError("FoundationFeatureDetector got a point with non-finite values.")

        self._window.append(x.copy())

        # Only process when window is full
        if len(self._window) == self.context_length:
            window_array = np.array(self._window)  # (context_length, num_channels)
            embedding = self._extract_embedding(window_array)
            self._update_ema(embedding)
            self._windows_seen += 1

        return self

    def score_partial(self, X: NDArray[Any]) -> float:
        """Scores the anomalousness of the next instance.

        Forms a window ending with the given instance (using recent history),
        extracts its embedding, and computes the Mahalanobis distance to the
        running mean of normal embeddings.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The anomalousness score (Mahalanobis distance). Returns 0.0
                during warm-up.
        """
        x = np.asarray(X, dtype=np.float64).ravel()
        if not np.all(np.isfinite(x)):
            raise ValueError("FoundationFeatureDetector got a point with non-finite values.")

        # Form window ending with X
        if len(self._window) < self.context_length - 1:
            # Not enough history yet
            return 0.0

        # Build window: last (context_length - 1) points from history + current point
        recent = list(self._window)[-(self.context_length - 1) :]
        window = np.vstack([*recent, x.reshape(1, -1)])  # (context_length, num_channels)

        embedding = self._extract_embedding(window)

        if not self.is_warmed_up_:
            return 0.0

        return self._mahalanobis_distance(embedding)

    def fit_score_partial(self, X: NDArray[Any], y: int | None = None) -> float:
        """Scores the instance with the current model, then updates the model.

        This is the prequential (test-then-train) order.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int or None): Ignored (Default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        score = self.score_partial(X)
        self.fit_partial(X, y)
        return score
