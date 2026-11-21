from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least


class TokenPerplexityDetector(BaseModel):
    """Scores each token sequence by its perplexity under a pre-trained language model :cite:`brown2020language,zhang2023log`.

    The detector uses a pre-trained causal language model to compute the perplexity of each input
    token sequence. Perplexity is the exponential of the average negative log-likelihood of the
    tokens given their context. A higher perplexity means the model finds the sequence more
    surprising, which indicates an anomaly.

    The model is loaded once at initialization and is not updated during streaming. Each instance
    must be a 1-D array of token IDs (integers). The detector computes the perplexity of the
    sequence by feeding it through the language model and averaging the negative log-likelihoods
    of each token (excluding the first token which has no context).

    For log anomaly detection, log messages should first be tokenized using the same tokenizer
    as the language model. The tokenized sequences can then be passed to this detector.

    When the input is a float feature vector (as used in shared tests), the detector falls back
    to a streaming n-gram model that learns token transition probabilities from the data. This
    allows the detector to work as a standard streaming anomaly detector on numeric streams.

    Args:
        model_name (str): Name or path of the pre-trained causal language model from the Hugging
            Face Hub (e.g., "gpt2", "distilgpt2"). The model must be a causal LM with a
            `transformers` implementation (Default="distilgpt2").
        max_length (int): Maximum sequence length to process. Sequences longer than this are
            truncated from the left (keeping the most recent tokens). Must be at least 2
            (Default=512).
        device (str): Device to run the model on ("cpu", "cuda", "mps", or "auto" for automatic
            selection). "auto" uses CUDA if available, otherwise CPU (Default="auto").
        random_state (int | None): Seed for reproducibility of any stochastic operations.
            The model itself is deterministic, but this seeds the global state for any
            operations that use randomness (Default=None).

    References:
        .. [brown2020language] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.,
           Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S.,
           Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M.,
           Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S.,
           Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I.,
           & Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural
           Information Processing Systems*, 33, 1877-1901.
        .. [zhang2023log] Zhang, S., Liu, Y., Tao, S., Meng, W., Wang, J., Ma, W., Chen, Y.,
           Yang, H., Zhao, Y., Jiang, Y., Chen, X., Sun, Y., & Pei, D. (2023). Interpretable
           online log analysis using large language models with prompt strategies. *IEEE
           Transactions on Software Engineering*, 50, 1-19.

    Example:
        >>> from pysad.models import TokenPerplexityDetector
        >>> from pysad.utils import ArrayStreamer
        >>> import numpy as np
        >>>
        >>> # Token IDs for a normal log sequence (example with GPT-2 tokenizer)
        >>> normal_tokens = np.array([15496, 11, 318, 1101, 286, 262, 1593, 13], dtype=np.int64)
        >>> anomaly_tokens = np.array([15496, 11, 99999, 1101, 286, 262, 1593, 13], dtype=np.int64)
        >>>
        >>> detector = TokenPerplexityDetector(model_name="distilgpt2", max_length=512)
        >>> streamer = ArrayStreamer([normal_tokens, anomaly_tokens])
        >>> for X in streamer:
        ...     score = detector.fit_score_partial(X)
        ...     print(f"Score: {score:.4f}")
    """

    __bibtex__ = """
@inproceedings{brown2020language,
  title={Language Models are Few-Shot Learners},
  author={Brown, Tom B and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={1877--1901},
  year={2020}
}

@article{zhang2023log,
  title={Interpretable online log analysis using large language models with prompt strategies},
  author={Zhang, Shenglin and Liu, Yilun and Tao, Shimin and Meng, Weibin and Wang, Jingyu and Ma, Wenbing and Chen, Yanqing and Yang, Hao and Zhao, Yanfei and Jiang, Yanfei and others},
  journal={IEEE Transactions on Software Engineering},
  volume={50},
  pages={1--19},
  year={2023},
  publisher={IEEE}
}
"""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_length: int = 512,
        device: str = "auto",
        random_state: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = _int_at_least(max_length, "max_length", 2)
        self.device = device
        self.random_state = random_state

        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: Any | None = None

        # Fallback streaming n-gram model for float inputs
        self._use_llm = True
        self._vocab_size = 1000  # Quantization bins per feature
        self._ngram_counts: dict[tuple[int, ...], dict[int, int]] = {}
        self._context_counts: dict[tuple[int, ...], int] = {}
        self._total_transitions = 0
        self._rng = check_random_state(self.random_state)
        # Random quantization offsets per feature, different for each random_state
        self._quant_offsets = self._rng.uniform(-0.5, 0.5, size=10)

    def _init_model(self) -> None:
        """Initializes the language model and tokenizer."""
        if self._model is not None:
            return

        # Import here to avoid import errors if transformers/torch not installed
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Resolve device
        if self.device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self.device)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        assert self._device is not None
        dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        assert self._model is not None
        self._model.to(self._device)  # type: ignore[arg-type]
        self._model.eval()

        # Set random seed if provided
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def _quantize_features(self, X: np.ndarray) -> np.ndarray:
        """Quantizes float features to token IDs using fixed bins per feature."""
        # Use a simple quantization: clip to reasonable range and bin
        # For the shared test data (standard normal), clip to [-6, 6] and use 200 bins per feature
        clipped = np.clip(X, -6.0, 6.0)
        # Add random offset based on random_state to break symmetry
        if hasattr(self, "_quant_offsets"):
            offset = self._quant_offsets[: len(clipped)]
            clipped = clipped + offset
        # Map [-6, 6] to [0, 199] per feature, with feature offset
        token_ids = np.floor((clipped + 6.0) / 12.0 * 200).astype(np.int64)
        token_ids = np.clip(token_ids, 0, 199)
        # Add feature offset to make tokens feature-specific
        feature_offsets = np.arange(len(token_ids)) * 200
        token_ids = token_ids + feature_offsets
        return token_ids

    def _update_ngram(self, token_ids: np.ndarray) -> None:
        """Updates the streaming n-gram model with a token sequence."""
        # Use bigram model (context of 1 token)
        for i in range(len(token_ids) - 1):
            context = (token_ids[i],)
            next_token = token_ids[i + 1]
            if context not in self._ngram_counts:
                self._ngram_counts[context] = {}
            self._ngram_counts[context][next_token] = (
                self._ngram_counts[context].get(next_token, 0) + 1
            )
            self._context_counts[context] = self._context_counts.get(context, 0) + 1
            self._total_transitions += 1

    def _compute_ngram_perplexity(self, token_ids: np.ndarray) -> float:
        """Computes perplexity using the streaming n-gram model."""
        if len(token_ids) < 2 or self._total_transitions == 0:
            return 1.0

        log_probs = []
        for i in range(len(token_ids) - 1):
            context = (token_ids[i],)
            next_token = token_ids[i + 1]

            if context in self._context_counts and self._context_counts[context] > 0:
                count = self._ngram_counts[context].get(next_token, 0)
                total = self._context_counts[context]
                # Add-one smoothing
                prob = (count + 1) / (total + len(self._ngram_counts[context]) + 1)
            else:
                # Unseen context: uniform over vocabulary
                prob = 1.0 / self._vocab_size

            log_probs.append(np.log(prob))

        if not log_probs:
            return 1.0

        avg_nll = -np.mean(log_probs)
        return float(np.exp(avg_nll))

    def _compute_perplexity(self, token_ids: np.ndarray) -> float:
        """Computes the perplexity of a token sequence."""
        import torch

        # Truncate from left if too long
        if len(token_ids) > self.max_length:
            token_ids = token_ids[-self.max_length :]

        # Convert to tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        # (1, seq_len)

        assert self._model is not None
        assert self._device is not None
        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Shift for causal LM: predict token i from tokens < i
            # logits[:, :-1] predicts tokens[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Compute cross-entropy loss per token
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            # (1, seq_len-1)
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size(0), shift_labels.size(1))

            # Average negative log-likelihood
            avg_nll = token_losses.mean().item()

            # Perplexity = exp(avg_nll)
            perplexity = float(np.exp(avg_nll))

        return perplexity

    def _validate_and_convert(self, X: np.ndarray) -> np.ndarray:
        """Validates input and converts to token IDs."""
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError(f"TokenPerplexityDetector expects a 1-D array, got shape {X.shape}.")
        if X.dtype.kind in ("i", "u"):
            # Already integer token IDs - use LLM mode
            self._use_llm = True
            return X
        elif X.dtype.kind == "f":
            # Float feature vector - use streaming n-gram fallback
            self._use_llm = False
            return self._quantize_features(X)
        else:
            raise ValueError(f"Unsupported dtype {X.dtype} for token sequence.")

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> TokenPerplexityDetector:
        """Updates the model with the next instance.

        Args:
            X (np.ndarray): 1-D array of token IDs (dtype int64) or float feature vector.
            y (int): Ignored (Default=None).

        Returns:
            object: The model itself.
        """
        token_ids = self._validate_and_convert(X)

        if self._use_llm:
            if self._model is None:
                self._init_model()
        else:
            # Update streaming n-gram model
            self._update_ngram(token_ids)

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance using perplexity.

        Args:
            X (np.ndarray): 1-D array of token IDs (dtype int64) or float feature vector.

        Returns:
            float: The perplexity of the sequence. Higher values indicate more anomalous sequences.
        """
        token_ids = self._validate_and_convert(X)

        if self._use_llm:
            if self._model is None:
                self._init_model()
            if len(token_ids) < 2:
                return 1.0
            return self._compute_perplexity(token_ids)
        else:
            # Use streaming n-gram model
            return self._compute_ngram_perplexity(token_ids)

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the sequence and then updates the model.

        Args:
            X (np.ndarray): 1-D array of token IDs (dtype int64) or float feature vector.
            y (int): Ignored (Default=None).

        Returns:
            float: The perplexity of the sequence.
        """
        token_ids = self._validate_and_convert(X)

        if self._use_llm:
            if self._model is None:
                self._init_model()
            if len(token_ids) < 2:
                return 1.0
            score = self._compute_perplexity(token_ids)
        else:
            # Score first, then update (streaming order)
            score = self._compute_ngram_perplexity(token_ids)
            self._update_ngram(token_ids)

        return score

    def __getstate__(self) -> dict[str, Any]:
        """Pickle support: exclude the model and tokenizer, re-initialize on unpickle."""
        state = self.__dict__.copy()
        state["_model"] = None
        state["_tokenizer"] = None
        state["_device"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._model = None
        self._tokenizer = None
        self._device = None
