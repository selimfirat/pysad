from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.utils.validation import _int_at_least

# Constants of the NuPIC spatial pooler.
_PERMANENCE_EPSILON = 1e-6
_UPDATE_PERIOD = 50
_INIT_CONNECTED_FRACTION = 0.5


class SpatialPooler:
    """Maps a binary input to a sparse set of active columns and learns online, as the HTM spatial pooler :cite:`cui2017htm`.

    This is an internal part of the HTM detector, not a detector. It follows the spatial pooler of NuPIC 1.0.5 (https://github.com/numenta/nupic, MIT license), the version that the ``numenta`` detector of NAB uses, with global inhibition. The defaults are the spatial pooler parameters of that detector.

    Each of ``num_columns`` columns has a fixed pool of potential synapses to the inputs. Each potential synapse has a permanence in [0, 1], and it is connected if its permanence is at least ``connected_threshold``. The overlap of a column is the number of its connected synapses to active inputs, times its boost factor when the pooler learns. Inhibition activates the ``num_active`` columns with the largest overlaps, if their overlap is at least ``stimulus_threshold``. When it learns, each active column raises the permanences of its potential synapses to active inputs by ``permanence_increment`` and lowers the others by ``permanence_decrement``. The boost factor of a column is ``exp(boost_strength * (s - a_i))``, where ``a_i`` is its time-averaged activation and ``s`` is the target fraction of active columns.

    The implementation differs from the paper as NuPIC does:

    - The potential pool of each column has exactly ``round(potential_fraction * num_inputs)`` inputs, not a Bernoulli draw per input (Equation 1). The input has no topology, so the pool can hold any input and inhibition is global.
    - Half of the potential synapses start with a permanence drawn uniformly from [``connected_threshold``, 1] and the rest from [0, ``connected_threshold``), not one uniform draw (Equation 3). The initial permanences are truncated to 5 decimals.
    - A permanence below ``permanence_increment / 2`` is set to 0.
    - Each column starts with, and keeps after each update of its active synapses, at least ``stimulus_threshold`` connected synapses: its potential permanences rise by ``connected_threshold / 10`` until it has.
    - The target activation of boosting is ``s``, not the mean activation of the neighbor columns (Equations 9 and 10). The boost factors apply only when the pooler learns.
    - The time averages use the period ``min(duty_cycle_period, t)`` at learning step ``t``, not ``duty_cycle_period`` from the start (Equation 8).
    - A column whose fraction of steps with a nonzero overlap is below ``min_overlap_duty_cycle`` times the largest such fraction raises all of its potential permanences by ``connected_threshold / 10`` on each learning step. The pooler updates the bound every 50 learning steps.
    - Equation 6 activates every column at or above the percentile, so ties can activate more than ``num_active`` of the columns. The pooler activates exactly the columns with the largest overlaps, and a tie goes to the column with the larger index.

    The implementation differs from NuPIC as follows:

    - Only the steps that learn advance the step count of the time averages and of the 50-step updates, so a step without learning does not change the state. NuPIC also counts the steps without learning.
    - The random draws come from NumPy, so the potential pools and the initial permanences differ from those of NuPIC with the same seed.

    Args:
        num_inputs (int): The number of input bits.
        num_columns (int): The number of columns (Default=2048).
        num_active (int): The number of active columns. NuPIC caps it at half of the columns (Default=40).
        potential_fraction (float): The fraction of the inputs in the potential pool of each column (Default=0.8).
        connected_threshold (float): The permanence at which a synapse is connected (Default=0.2).
        permanence_increment (float): The increase of the permanence of a synapse to an active input (Default=0.003).
        permanence_decrement (float): The decrease of the permanence of a synapse to an inactive input (Default=0.0005).
        stimulus_threshold (float): The smallest overlap of an active column (Default=0.0).
        boost_strength (float): The boost strength. Zero turns boosting off (Default=0.0).
        duty_cycle_period (int): The period of the time averages of the activation and of the overlap (Default=1000).
        min_overlap_duty_cycle (float): The fraction of the largest overlap time average below which a column is weak (Default=0.001).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the potential pools and the initial permanences. None draws from NumPy's global random state (Default=None).
    """

    def __init__(
        self,
        num_inputs: int,
        num_columns: int = 2048,
        num_active: int = 40,
        potential_fraction: float = 0.8,
        connected_threshold: float = 0.2,
        permanence_increment: float = 0.003,
        permanence_decrement: float = 0.0005,
        stimulus_threshold: float = 0.0,
        boost_strength: float = 0.0,
        duty_cycle_period: int = 1000,
        min_overlap_duty_cycle: float = 0.001,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.num_inputs = _int_at_least(num_inputs, "num_inputs", 1)
        self.num_columns = _int_at_least(num_columns, "num_columns", 2)
        self.target_density = min(
            _int_at_least(num_active, "num_active", 1) / self.num_columns, 0.5
        )
        self.num_active = int(self.target_density * self.num_columns)
        num_potential = int(self.num_inputs * potential_fraction + 0.5)
        if num_potential < stimulus_threshold:
            # No column could ever reach the stimulus threshold.
            raise ValueError(
                f"potential_fraction must give each column at least stimulus_threshold={stimulus_threshold!r} "
                f"potential synapses, got {num_potential} of {self.num_inputs} inputs."
            )
        self.connected_threshold = connected_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.stimulus_threshold = stimulus_threshold
        self.boost_strength = boost_strength
        self.duty_cycle_period = _int_at_least(duty_cycle_period, "duty_cycle_period", 1)
        self.min_overlap_duty_cycle = min_overlap_duty_cycle
        self._bump = connected_threshold / 10.0
        self._trim_threshold = permanence_increment / 2.0

        rng = check_random_state(random_state)
        shape = (self.num_columns, self.num_inputs)
        self.potential_ = np.zeros(shape, dtype=bool)
        pools = rng.uniform(size=shape).argsort(axis=1)[
            :, :num_potential
        ]  # (num_columns, num_potential)
        np.put_along_axis(self.potential_, pools, True, axis=1)
        draws = rng.uniform(size=shape)
        permanences = np.where(
            rng.uniform(size=shape) <= _INIT_CONNECTED_FRACTION,
            connected_threshold + (1.0 - connected_threshold) * draws,
            connected_threshold * draws,
        )
        permanences = np.floor(permanences * 1e5) / 1e5
        permanences[permanences < self._trim_threshold] = 0.0
        self.permanences_ = np.where(self.potential_, permanences, 0.0)
        self.connected_ = np.zeros(shape, dtype=bool)
        columns = np.arange(self.num_columns)
        self._set_permanences(columns, self.permanences_, raise_to_threshold=True)

        self.overlap_duty_cycles_ = np.zeros(self.num_columns)
        self.active_duty_cycles_ = np.zeros(self.num_columns)
        self.min_overlap_duty_cycles_ = np.zeros(self.num_columns)
        self.boost_factors_ = np.ones(self.num_columns)
        self.num_learned_ = 0

    def compute(self, x: np.ndarray, learn: bool = True) -> np.ndarray:
        """Returns the sorted indices of the active columns for the boolean input `x` of shape (num_inputs,), and learns from it if `learn` is True."""
        overlaps = self.connected_[:, x].sum(axis=1)  # (num_columns,)
        boosted = self.boost_factors_ * overlaps if learn else overlaps
        # A stable sort puts the larger column index last among ties, so it wins them.
        winners = np.argsort(boosted, kind="stable")[-self.num_active :]
        active = np.sort(winners[boosted[winners] >= self.stimulus_threshold])
        if not learn:
            return active

        self.num_learned_ += 1
        deltas = np.where(x, self.permanence_increment, -self.permanence_decrement)  # (num_inputs,)
        self._set_permanences(
            active,
            self.permanences_[active] + deltas * self.potential_[active],
            raise_to_threshold=True,
        )

        period = min(self.duty_cycle_period, self.num_learned_)
        is_active = np.zeros(self.num_columns)
        is_active[active] = 1.0
        self.overlap_duty_cycles_ = (
            self.overlap_duty_cycles_ * (period - 1) + (overlaps > 0)
        ) / period
        self.active_duty_cycles_ = (self.active_duty_cycles_ * (period - 1) + is_active) / period

        weak = np.flatnonzero(self.overlap_duty_cycles_ < self.min_overlap_duty_cycles_)
        if weak.size:
            self._set_permanences(
                weak,
                self.permanences_[weak] + self._bump * self.potential_[weak],
                raise_to_threshold=False,
            )
        self.boost_factors_ = np.exp(
            (self.target_density - self.active_duty_cycles_) * self.boost_strength
        )
        if self.num_learned_ % _UPDATE_PERIOD == 0:
            self.min_overlap_duty_cycles_.fill(
                self.min_overlap_duty_cycle * self.overlap_duty_cycles_.max()
            )
        return active

    def _set_permanences(
        self, columns: np.ndarray, permanences: np.ndarray, raise_to_threshold: bool
    ) -> None:
        """Stores the permanences of shape (len(columns), num_inputs) for `columns`, as NuPIC's _updatePermanencesForColumn."""
        if raise_to_threshold:
            np.clip(permanences, 0.0, 1.0, out=permanences)
            # The loop ends because each column has at least stimulus_threshold potential synapses.
            below = (permanences > self.connected_threshold - _PERMANENCE_EPSILON).sum(
                axis=1
            ) < self.stimulus_threshold
            while below.any():
                permanences[below] += self._bump * self.potential_[columns[below]]
                below[below] = (
                    permanences[below] > self.connected_threshold - _PERMANENCE_EPSILON
                ).sum(axis=1) < self.stimulus_threshold
        permanences[permanences < self._trim_threshold] = 0.0
        np.clip(permanences, 0.0, 1.0, out=permanences)
        self.permanences_[columns] = permanences
        self.connected_[columns] = permanences >= self.connected_threshold - _PERMANENCE_EPSILON
