# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Neutral plasma channels on the reference and the compiled backends (schema 42).

A neutral channel (``electrons_per_event=0``) shares its population's exposure scan and
loss current with an ionising one and adds no free-electron current and no electrons.
On ``numpy_ref``, the oracle, the shared current of a neutral and an ionising channel is
the sum of their separate responses and the statistics count the neutral channel's
events but not its electrons. On ``numba_cpu`` and ``native`` the ladder of
``docs/dev/backends.md`` -- one nonlinear evaluation, one step, eight steps and every
species statistic -- is held to the rounding budgets the mixture ladder derives
(``test_mixture_ladder.py``), for two channels (the channel loop) and for a lone neutral
channel (the one-channel specialisation, which reads the same coefficient). A detection
check makes sure each rung could see a backend that ignored the declaration.
"""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mixture_cases import mixture_case
from optmat_parity import derivative, deviation, stage_bound
from optmat_parity_cases import Case
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.populations import axis_statistics
from solprop.exceptions import OutputError
from solprop.stats import StatsSpec, species_statistics
from solprop.stepper import FixedSchedule
from solprop.translate.stage import scalars

EPS = np.finfo(float).eps

NAMES = (
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
    "species_density",
    "species_ionisation_energy",
    "electron_density",
    "peak_ionisation_rate",
)

DETECTION = 1e3
"""How far above a rung's bound the neutral declaration must move the reference.

A backend that ran the neutral channel as an ionising one would then miss the reference
by a thousand bounds, so no rung can pass by ignoring it.
"""

CASES = {
    # Two channels, the second neutral: the channel loop of every backend.
    "mixture": {"electrons_per_event": (1.0, 0.0)},
    # A lone neutral channel: the one-channel specialisation of every backend.
    "single": {"electrons_per_event": (1.0, 0.0), "channels": (1,)},
}


def neutral_case(name, *, ionising=False):
    """Return ``(grid, problem)`` of a case, or its all-ionising twin."""
    options = dict(CASES[name])
    if ionising:
        options["electrons_per_event"] = (1.0, 1.0)
    _, grid, _, _, problem = mixture_case(**options)
    return grid, problem


def fixed(problem, grid, steps):
    """Return `problem` on ``steps`` uniform steps of its declared initial size."""
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=FixedSchedule.uniform(0.0, grid.length, problem.stepper.dz_init),
        ),
    ), steps * problem.stepper.dz_init


def march(backend_class, problem, grid, stop):
    """Return the field, the step count and every statistic at `stop`."""
    backend = backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(stop)
        field = backend.observe().require_field().copy()
        return field, summary.n_steps, backend.stats(StatsSpec(names=NAMES))
    finally:
        backend.close()


def ladder_bound(grid, steps):
    """The mixture ladder's budget: 64 eps a sample, channel and stage, 7 a step."""
    return 64 * grid.n_to * 2 * 7 * steps * EPS


# ------------------------------------------------------------------ the oracle


def test_the_compiled_channels_carry_the_declaration():
    _, problem = neutral_case("mixture")
    ionising, neutral = problem.responses.plasma_channels
    assert problem.schema_version == 42
    assert (ionising.electrons_per_event, neutral.electrons_per_event) == (1.0, 0.0)
    assert neutral.free_current_coefficient == 0.0
    assert ionising.free_current_coefficient == ionising.charge_mass_ratio
    _, single = neutral_case("single")
    assert single.responses.plasma_coefficients is not None
    assert single.responses.plasma_coefficients.free_current_coefficient == 0.0


def test_a_neutral_channels_free_current_is_exactly_zero():
    _, problem = neutral_case("mixture")
    channel = problem.responses.plasma_channels[1]
    t = np.linspace(-30e-15, 30e-15, 1025)
    field = (3e10 * np.exp(-((t / 10e-15) ** 2)) * np.cos(2.4e15 * t))[None, None, :]
    dt = float(t[1] - t[0])
    rate = passes.ionisation_rate(field, channel.rate)
    fraction, free_current = passes.plasma_scan(
        field, rate, channel, preionisation=0.1, dt=dt
    )
    # Exact: a zero coefficient times anything is zero, and so is its running sum.
    assert np.all(free_current == 0.0)
    assert float(fraction[0, -1]) > 0.1


def test_the_shared_current_is_the_sum_of_the_separate_channels():
    # The channel loop against each channel's own one-channel pass, a neutral one
    # among them: test_mixture_channels.py's budget, 16 operations per sample and 64
    # per channel, scaled by the separate currents' magnitudes.
    _, problem = neutral_case("mixture")
    bundle = scalars(problem, 0.0, 0.0)
    t = np.linspace(-30e-15, 30e-15, 1025)
    field = (3e10 * np.exp(-((t / 10e-15) ** 2)) * np.cos(2.4e15 * t))[None, None, :]
    dt = float(t[1] - t[0])
    separate = []
    for channel, density, initial in zip(
        problem.responses.plasma_channels,
        bundle.plasma_densities,
        bundle.plasma_preionisation,
        strict=True,
    ):
        rate = passes.ionisation_rate(field, channel.rate)
        fraction, current = passes.plasma_scan(
            field, rate, channel, preionisation=initial, dt=dt
        )
        separate.append(
            passes.plasma_current(
                field, rate, fraction, current, channel, density=density, dt=dt
            )
        )
    rounding = (16 * t.size + 64 * problem.flags.n_channels) * EPS
    bound = rounding * sum(float(np.max(np.abs(value))) for value in separate)
    assert bound > 0.0
    actual = passes.plasma_channels_polarisation(
        field,
        problem.responses.plasma_channels,
        bundle.plasma_densities,
        bundle.plasma_preionisation,
        dt,
    )
    assert_allclose(actual, sum(separate), rtol=0, atol=bound)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_statistics_count_events_but_not_neutral_electrons(name):
    grid, original = neutral_case(name)
    problem, stop = fixed(original, grid, 2)
    _, _, stats = march(NumpyRefBackend, problem, grid, stop)
    fractions = stats["species_ionisation"][0]
    electrons = stats["species_electron_density"][0]
    densities = stats["species_density"][0]
    energy = stats["species_ionisation_energy"][0]
    # The neutral population underwent its events (its preionisation and more) and
    # paid for them, and freed no electron: exactly zero, a hard zero of the contract.
    assert np.all(fractions[1] > 0.1)
    assert electrons[1] == 0.0
    assert energy[1] > 0.0
    assert stats["peak_ionisation_rate"][0] > 0.0
    if name == "single":
        # The one-channel statistic reads the same weight: no electrons at all.
        assert stats["electron_density"][0] == 0.0
        assert np.all(fractions[0] == 0.0)
    else:
        # The total is the ionising population's alone, summed with an exact zero.
        assert stats["electron_density"][0] == electrons[0]
        assert_allclose(electrons[0], densities[0] * np.max(fractions[0]), rtol=EPS)


def test_the_species_reduction_counts_electrons_by_their_weight():
    densities = np.array([2e24, 3e24])
    fractions = np.array([[0.1, 0.2], [0.4, 0.3]])
    weights = np.array([1e-10, 2e-10])
    values = species_statistics(
        densities,
        fractions,
        np.array([1e12, 5e12]),
        np.array([2e-18, 3e-18]),
        weights,
        np.array([1.0, 0.0]),
    )
    # The densest node is the ionising population's (node 1); the neutral one's
    # electrons are an exact zero, its energy its events' inventory.
    assert values["species_electron_density"].tolist() == [2e24 * 0.2, 0.0]
    assert values["electron_density"] == 2e24 * 0.2
    assert_allclose(
        values["species_ionisation_energy"][1],
        3e-18 * 3e24 * (0.4 * 1e-10 + 0.3 * 2e-10),
        rtol=4 * EPS,
    )
    assert values["peak_ionisation_rate"] == 5e12
    # Without weights every species frees one electron per event, as before.
    unweighted = species_statistics(
        densities, fractions, np.array([1e12, 5e12]), np.array([2e-18, 3e-18]), weights
    )
    assert unweighted["electron_density"] == 2e24 * 0.1 + 3e24 * 0.4
    with pytest.raises(OutputError, match="electron per event"):
        species_statistics(
            densities,
            fractions,
            np.array([1e12, 5e12]),
            np.array([2e-18, 3e-18]),
            weights,
            np.array([1.0, 2.0]),
        )


def test_the_axis_reduction_counts_no_neutral_electrons():
    _, problem = neutral_case("mixture")
    values = axis_statistics(problem, 0.0, np.array([0.2, 0.3]), np.array([1.0, 2.0]))
    densities = [float(table(0.0)) for table in problem.responses.species_densities]
    assert values["on_axis_species_electron_density"].tolist() == [
        densities[0] * 0.2,
        0.0,
    ]
    assert values["on_axis_electron_density"] == densities[0] * 0.2
    assert values["on_axis_peak_ionisation_rate"] == 2.0


# ----------------------------------------------------- the compiled backends


@pytest.mark.parametrize("name", sorted(CASES))
def test_one_nonlinear_evaluation_agrees_with_the_reference(
    name, equivalence_backend_class
):
    grid, problem = neutral_case(name)
    case = Case(name, None, problem, grid, None)
    bound = stage_bound(case)
    assert bound > 0.0
    expected = derivative(NumpyRefBackend, case)
    actual = derivative(equivalence_backend_class, case)
    assert deviation(actual, expected) <= bound
    # The declaration is visible at this rung: the all-ionising twin's first stage
    # departs from the neutral one by far more than the bound.
    twin_grid, twin = neutral_case(name, ionising=True)
    ionising = derivative(NumpyRefBackend, Case(name, None, twin, twin_grid, None))
    assert deviation(ionising, expected) > DETECTION * bound


@pytest.mark.parametrize("steps", [1, 8])
@pytest.mark.parametrize("name", sorted(CASES))
def test_fixed_steps_and_every_statistic_agree_with_the_reference(
    name, steps, equivalence_backend_class
):
    grid, original = neutral_case(name)
    problem, stop = fixed(original, grid, steps)
    bound = ladder_bound(grid, steps)
    assert bound > 0.0
    oracle = march(NumpyRefBackend, problem, grid, stop)
    ours = march(equivalence_backend_class, problem, grid, stop)
    assert ours[1] == oracle[1] == steps
    assert deviation(ours[0], oracle[0]) <= bound
    for statistic in NAMES:
        assert_allclose(
            ours[2][statistic],
            oracle[2][statistic],
            rtol=bound,
            atol=bound if statistic == "species_ionisation" else 0.0,
            err_msg=statistic,
        )
    # The neutral population's electrons are a hard zero on every backend.
    assert ours[2]["species_electron_density"][0][1] == 0.0
