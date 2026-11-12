import pytest


def _reference_scores(x, window_size, num_output_neurons, seed):
    """Scores of Algorithms 1 to 7 of the paper, with the score rules of OeSNNUAD, one neuron at a time."""
    import math

    import numpy as np

    num_input_neurons, sim, mod, c, xi, epsilon = 10, 0.17, 0.6, 0.6, 0.9, 4.0
    rng = np.random.RandomState(seed)
    gamma = c * sum(mod ** (2 * k) for k in range(num_input_neurons))
    neurons = []  # [weights, output value, update time, update count]
    errors, anomalies, scores = [], [], []
    for t in range(1, len(x) + 1):
        window = x[max(0, t - window_size) : t]
        if t < window_size:
            scores.append(0.0)
            continue
        if t == window_size:
            predictions = rng.normal(window.mean(), window.std(), window_size - 1)
            errors = list(np.abs(window[1:] - predictions))
            anomalies = [False] * (window_size - 1)
            scores.append(0.0)
            continue

        # Algorithm 2: the receptive fields and the firing order of the input neurons, on the window
        # range scaled to [0, 1].
        scale = window.max() - window.min()
        position = (x[t - 1] - window.min()) / scale if scale > 0 else 0.0
        width = 1 / (num_input_neurons - 2)
        excitations = []
        for j in range(num_input_neurons):
            center = (2 * j - 3) / 2 * width
            excitations.append(math.exp(-0.5 * ((position - center) / width) ** 2))
        firing = sorted(range(num_input_neurons), key=lambda j: -excitations[j])
        order = {j: k for k, j in enumerate(firing)}

        # Algorithm 3: the potentials grow as the input neurons fire.
        potentials = [0.0] * len(neurons)
        fired = None
        for j in firing:
            for i, neuron in enumerate(neurons):
                potentials[i] += neuron[0][j] * mod ** order[j]
            to_fire = [i for i in range(len(neurons)) if potentials[i] > gamma]
            if to_fire:
                fired = max(to_fire, key=lambda i: potentials[i])
                break
        full = [sum(n[0][j] * mod ** order[j] for j in range(num_input_neurons)) for n in neurons]

        normal_errors = np.array([e for e, a in zip(errors, anomalies, strict=True) if not a])
        if not neurons:
            scores.append(0.0)
        else:
            predicted = fired if fired is not None else int(np.argmax(full))
            error = abs(x[t - 1] - neurons[predicted][1])
            std = normal_errors.std() if normal_errors.size else 0.0
            scores.append((error - normal_errors.mean()) / std if std > 0 else 0.0)

        # Algorithm 4: the decision of the paper.
        if fired is None:
            error, anomaly = math.inf, True
        else:
            error = abs(x[t - 1] - neurons[fired][1])
            anomaly = bool(normal_errors.size) and (
                error - normal_errors.mean() > epsilon * normal_errors.std()
            )
        errors, anomalies = errors[1:] + [error], anomalies[1:] + [anomaly]

        # Algorithms 5 to 7: the candidate output neuron.
        weights = np.array([mod ** order[j] for j in range(num_input_neurons)])
        output = rng.normal(window.mean(), window.std())
        if not anomaly:
            output += (x[t - 1] - output) * xi
        distances = [np.linalg.norm(weights - n[0]) for n in neurons]
        if neurons and min(distances) <= sim:
            n = neurons[int(np.argmin(distances))]
            m = n[3]
            n[:] = [
                (weights + m * n[0]) / (m + 1),
                (output + m * n[1]) / (m + 1),
                (t + m * n[2]) / (m + 1),
                m + 1,
            ]
        elif len(neurons) < num_output_neurons:
            neurons.append([weights, output, t, 1])
        else:
            neurons[int(np.argmin([n[2] for n in neurons]))] = [weights, output, t, 1]

    return np.array(scores)


@pytest.mark.parametrize("num_output_neurons", [3, 50])
def test_oesnn_uad_scores_match_the_algorithms_of_the_paper(num_output_neurons):
    import numpy as np

    from pysad.models import OeSNNUAD

    rng = np.random.RandomState(0)
    x = np.sin(2 * np.pi * np.arange(400) / 30) + 0.1 * rng.normal(size=400)
    x[[200, 320]] += [3.0, -2.0]

    model = OeSNNUAD(window_size=30, num_output_neurons=num_output_neurons, random_state=7)
    scores = model.fit_score(x.reshape(-1, 1))

    expected = _reference_scores(x, 30, num_output_neurons, seed=7)
    np.testing.assert_allclose(scores, expected, rtol=1e-9, atol=1e-12)
    assert np.count_nonzero(scores) > 300


def test_oesnn_uad_gives_a_spike_on_a_sine_wave_the_highest_score():
    import numpy as np

    from pysad.models import OeSNNUAD

    x = np.sin(2 * np.pi * np.arange(600) / 50)
    x[400] += 1.5
    scores = OeSNNUAD(window_size=50, random_state=0).fit_score(x.reshape(-1, 1))

    assert scores.argmax() == 400


def test_oesnn_uad_scores_do_not_depend_on_the_units():
    import numpy as np

    from pysad.models import OeSNNUAD

    x = np.random.RandomState(1).normal(size=300)
    scores = OeSNNUAD(window_size=20, random_state=3).fit_score(x.reshape(-1, 1))
    scaled = OeSNNUAD(window_size=20, random_state=3).fit_score((1e3 * x + 1e2).reshape(-1, 1))

    np.testing.assert_allclose(scaled, scores, rtol=1e-6, atol=1e-9)


def test_oesnn_uad_scores_zero_while_the_first_window_fills_and_on_a_constant_stream():
    import numpy as np

    from pysad.models import OeSNNUAD

    x = np.random.RandomState(2).normal(size=(40, 1))
    scores = OeSNNUAD(window_size=10, random_state=0).fit_score(x)
    np.testing.assert_array_equal(scores[:11], 0.0)

    # The output values equal the constant, so every error is 0.
    constant = OeSNNUAD(window_size=10, random_state=0).fit_score(np.full((100, 1), 5.0))
    np.testing.assert_array_equal(constant, 0.0)


@pytest.mark.parametrize(
    "params",
    [
        {"window_size": 1},
        {"num_input_neurons": 2},
        {"num_output_neurons": 0},
        {"mod": 0.0},
        {"mod": 1.0},
        {"c": 0.0},
        {"c": 1.5},
        {"xi": -0.1},
        {"xi": 1.1},
    ],
)
def test_oesnn_uad_rejects_parameters_out_of_range(params):
    from pysad.models import OeSNNUAD

    with pytest.raises(ValueError):
        OeSNNUAD(**params)
