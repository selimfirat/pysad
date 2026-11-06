from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.utils.validation import _int_at_least


class SpatialPooler:
    """Maps a binary input to a sparse set of active columns and learns online, as the HTM spatial pooler :cite:`cui2017htm`.

    This is an internal part of the HTM detector, not a detector. Each of ``num_columns`` columns has a fixed pool of potential synapses to the inputs (Equation 1). Each potential synapse has a permanence in [0, 1], drawn uniformly at the start (Equation 3), and it is connected if its permanence is at least ``connected_threshold`` (Equation 2). The overlap of a column is its boost factor times the number of its connected synapses to active inputs (Equation 5). Inhibition activates the ``round(density * num_columns)`` columns with the largest overlaps, if their overlap is at least ``stimulus_threshold`` (Equation 6). When it learns, each active column raises the permanences of its potential synapses to active inputs by ``permanence_increment``, lowers the others by ``permanence_decrement``, and clips them to [0, 1]. Homeostatic boosting keeps the time-averaged activation of each column (Equation 8) and sets its boost factor to ``exp(-boost_strength * (a_i - <a>))``, where ``<a>`` is the mean time-averaged activation of the other columns (Equations 9 and 10).

    The implementation differs from the paper as follows:

    - The input has no topology, so the potential radius is infinite and inhibition is global, as the paper does for inputs without topology (Table 1, 1D case). The module has no local inhibition and no inhibition radius.
    - Equation 6 activates every column at or above the percentile, so ties can activate more than ``density`` of the columns. The module activates exactly the columns with the largest overlaps (the k-winners-take-all of the paper) and breaks ties by a fixed random order of the columns, drawn at the start. A fixed order keeps the output a function of the input and the learned state.

    Args:
        num_inputs (int): The number of input bits.
        num_columns (int): The number of columns (Default=1024, as in Table 1).
        density (float): The target fraction of active columns ``s`` (Default=0.02, as in Table 1).
        potential_fraction (float): The fraction ``p`` of the inputs that are potential synapses of each column (Default=1.0, as in Table 1).
        connected_threshold (float): The permanence ``theta_c`` at which a synapse is connected (Default=0.5, as in Table 1).
        permanence_increment (float): The increase ``p+`` of the permanence of a synapse to an active input (Default=0.1, as in Table 1).
        permanence_decrement (float): The decrease ``p-`` of the permanence of a synapse to an inactive input (Default=0.02, as in Table 1).
        stimulus_threshold (float): The smallest overlap ``theta_stim`` of an active column (Default=1.0, as in Table 1).
        boost_strength (float): The boost strength ``beta``. Zero turns boosting off (Default=100.0, as in Table 1).
        duty_cycle_period (int): The number of steps ``T`` of the time-averaged activation (Default=1000, as in Table 1).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the potential pools, the initial permanences and the tie order. None draws from NumPy's global random state (Default=None).
    """

    def __init__(
        self,
        num_inputs: int,
        num_columns: int = 1024,
        density: float = 0.02,
        potential_fraction: float = 1.0,
        connected_threshold: float = 0.5,
        permanence_increment: float = 0.1,
        permanence_decrement: float = 0.02,
        stimulus_threshold: float = 1.0,
        boost_strength: float = 100.0,
        duty_cycle_period: int = 1000,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.num_inputs = _int_at_least(num_inputs, "num_inputs", 1)
        self.num_columns = _int_at_least(num_columns, "num_columns", 2)
        self.num_active = round(density * self.num_columns)
        if not 1 <= self.num_active <= self.num_columns:
            raise ValueError(
                f"density must activate between 1 and num_columns columns, got {density!r}."
            )
        self.connected_threshold = connected_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.stimulus_threshold = stimulus_threshold
        self.boost_strength = boost_strength
        self.duty_cycle_period = _int_at_least(duty_cycle_period, "duty_cycle_period", 1)

        rng = check_random_state(random_state)
        shape = (self.num_columns, self.num_inputs)
        self.potential_ = rng.uniform(size=shape) < potential_fraction  # Equation 1
        self.permanences_ = np.where(self.potential_, rng.uniform(size=shape), 0.0)  # Equation 3
        self.connected_ = self.permanences_ >= connected_threshold  # Equation 2
        self.tie_ranks_ = rng.permutation(self.num_columns)
        self.duty_cycles_ = np.zeros(self.num_columns)
        self.boost_factors_ = np.ones(self.num_columns)

    def compute(self, x: np.ndarray, learn: bool = True) -> np.ndarray:
        """Returns the sorted indices of the active columns for the boolean input `x` of shape (num_inputs,), and learns from it if `learn` is True."""
        overlaps = self.boost_factors_ * self.connected_[:, x].sum(
            axis=1
        )  # (num_columns,), Equation 5
        winners = np.lexsort((self.tie_ranks_, -overlaps))[: self.num_active]
        active = np.sort(winners[overlaps[winners] >= self.stimulus_threshold])  # Equation 6
        if not learn:
            return active

        # Hebbian learning on the potential synapses of the active columns: (num_active, num_inputs).
        deltas = np.where(x, self.permanence_increment, -self.permanence_decrement)
        permanences = np.clip(
            self.permanences_[active] + deltas * self.potential_[active], 0.0, 1.0
        )
        self.permanences_[active] = permanences
        self.connected_[active] = permanences >= self.connected_threshold

        # Homeostatic boosting, Equations 8 to 10.
        self.duty_cycles_ *= 1.0 - 1.0 / self.duty_cycle_period
        self.duty_cycles_[active] += 1.0 / self.duty_cycle_period
        neighbor_means = (self.duty_cycles_.sum() - self.duty_cycles_) / (self.num_columns - 1)
        self.boost_factors_ = np.exp(-self.boost_strength * (self.duty_cycles_ - neighbor_means))
        return active
