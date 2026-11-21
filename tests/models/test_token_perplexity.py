import numpy as np
import pytest


def test_token_perplexity_basic():
    """Test that TokenPerplexityDetector initializes and scores sequences."""
    from pysad.models import TokenPerplexityDetector

    # Use a tiny model for testing
    detector = TokenPerplexityDetector(model_name="sshleifer/tiny-gpt2", max_length=32)

    # Simple token sequence (token IDs for "hello world" in tiny-gpt2)
    normal_tokens = np.array([15496, 11, 318, 1101, 286, 262, 1593, 13], dtype=np.int64)
    anomaly_tokens = np.array([15496, 11, 99999, 1101, 286, 262, 1593, 13], dtype=np.int64)

    score_normal = detector.fit_score_partial(normal_tokens)
    score_anomaly = detector.fit_score_partial(anomaly_tokens)

    # Both should return valid perplexity scores (> 1)
    assert score_normal > 1.0
    assert score_anomaly > 1.0
    # Anomaly should generally have higher perplexity (not guaranteed for all sequences)
    # Just check they are different
    assert score_normal != score_anomaly


def test_token_perplexity_short_sequence():
    """Test that short sequences return a default score."""
    from pysad.models import TokenPerplexityDetector

    detector = TokenPerplexityDetector(model_name="sshleifer/tiny-gpt2", max_length=32)

    # Single token - cannot compute perplexity
    single_token = np.array([15496], dtype=np.int64)
    score = detector.fit_score_partial(single_token)

    # Should return default perplexity of 1.0
    assert score == 1.0


def test_token_perplexity_invalid_input():
    """Test that invalid inputs raise errors."""
    from pysad.models import TokenPerplexityDetector

    detector = TokenPerplexityDetector(model_name="sshleifer/tiny-gpt2", max_length=32)

    # 2D array
    with pytest.raises(ValueError, match="1-D array"):
        detector.fit_score_partial(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))

    # Float array is now accepted (for shared test compatibility) - test that it works
    score = detector.fit_score_partial(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    assert score > 0.0

    # Unsupported dtype
    with pytest.raises(ValueError, match="Unsupported dtype"):
        detector.fit_score_partial(np.array([1, 2, 3], dtype=np.complex128))


def test_token_perplexity_pickling():
    """Test that the detector can be pickled and unpickled."""
    import pickle

    from pysad.models import TokenPerplexityDetector

    detector = TokenPerplexityDetector(model_name="sshleifer/tiny-gpt2", max_length=32)

    tokens = np.array([15496, 11, 318, 1101], dtype=np.int64)
    score_before = detector.fit_score_partial(tokens)

    pickled = pickle.dumps(detector)
    detector2 = pickle.loads(pickled)

    score_after = detector2.fit_score_partial(tokens)

    # Scores should be equal (deterministic model)
    assert np.isclose(score_before, score_after, rtol=1e-5)


def test_token_perplexity_reproducibility():
    """Test that scores are reproducible with the same random_state."""
    from pysad.models import TokenPerplexityDetector

    detector1 = TokenPerplexityDetector(
        model_name="sshleifer/tiny-gpt2", max_length=32, random_state=42
    )
    detector2 = TokenPerplexityDetector(
        model_name="sshleifer/tiny-gpt2", max_length=32, random_state=42
    )

    tokens = np.array([15496, 11, 318, 1101, 286], dtype=np.int64)

    score1 = detector1.fit_score_partial(tokens)
    score2 = detector2.fit_score_partial(tokens)

    assert np.isclose(score1, score2, rtol=1e-5)


def test_token_perplexity_max_length():
    """Test that sequences longer than max_length are truncated."""
    from pysad.models import TokenPerplexityDetector

    detector = TokenPerplexityDetector(model_name="sshleifer/tiny-gpt2", max_length=4)

    # Sequence longer than max_length
    long_tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    short_tokens = np.array([5, 6, 7, 8], dtype=np.int64)

    score_long = detector.fit_score_partial(long_tokens)
    score_short = detector.fit_score_partial(short_tokens)

    # Should be equal after truncation
    assert np.isclose(score_long, score_short, rtol=1e-5)
