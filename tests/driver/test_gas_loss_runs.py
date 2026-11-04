# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Propagate ``loss="gas"`` through a fill whose absorbing population varies along z.

The operator is built on the host from the mode, so every backend receives the same
one; these runs check that it is the right one (a linear run attenuates each bin by
exactly the integrated gas absorption, with no wall loss) and that the backends
agree on a weakly nonlinear run, mode-averaged and mode-resolved. The absorber is
synthetic: a constant cross-section rising linearly from nothing.
"""

import math
import warnings

import numpy as np
import pytest

from march_bounds import FIXED_ORDER, assert_matched, simulation_bound
from solprop.backends import registry
from solprop.constants import C
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter
from solprop.modes import TubeLattice
from solprop.modes.base import alpha
from solprop.physdata import Component, Mixture, Species, TabulatedSusceptibility
from solprop.profiles import Linear
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule

# Short functional runs, a second or two each. The launches are noise-free into
# windows kept tight for speed, which hold more than a millionth of the energy in
# their guards: the boundary monitor warns (SEP 21), and the window's edge is not
# what this module tests.
pytestmark = [
    pytest.mark.medium,
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

LENGTH = 0.01
CROSS_SECTION = 1e-21
"""The synthetic absorber's cross-section in m^2."""

PEAK_DENSITY = 2e23
"""Its density at the exit, m^-3: an integrated attenuation of one (4.3 dB)."""

LATTICE = TubeLattice(210e-9, 5.55e-6, 6)


def runnable_backends() -> list[str]:
    """Return the binary64 backends registered and installed here, in the table's order.

    Read from the registry rather than written out, so that a retired backend (SEP 43
    removes ``numba_cpu``) or one not installed (``native`` without its wheel) drops
    out of the parametrisation instead of failing it.
    """
    return list(registry.available())


BACKENDS = runnable_backends()


def test_the_backends_are_read_from_the_registry(monkeypatch):
    """A backend the registry does not know is not run (SEP 43 drops ``numba_cpu``)."""
    known = dict(registry.BACKENDS)
    known.pop("numba_cpu", None)
    monkeypatch.setattr(registry, "BACKENDS", known)
    assert "numba_cpu" not in runnable_backends()
    assert runnable_backends()[0] == "numpy_ref"


def fill() -> Mixture:
    omega = np.array([1e12, 1e18])
    optical = TabulatedSusceptibility(
        omega,
        np.full(2, 1e-29),
        np.full(2, CROSS_SECTION),
        (2 * math.pi * C / omega[-1], 2 * math.pi * C / omega[0]),
        citation="Synthetic constant refractivity and absorption",
    )
    air = Mixture.air(6.0)
    trace = Component(
        Species("synthetic absorber", optical, gamma3=0.0),
        Linear(0.0, PEAK_DENSITY, LENGTH),
        "absorber",
    )
    return Mixture.of(*air.components, trace, temperature=air.temperature)


def build(backend: str, *, mode_average: bool, lattice=None, energy=1e-8) -> Simulation:
    grid = EnvelopeGrid.from_limits(LENGTH, 800e-9, (450e-9, 1200e-9), 60e-15)
    with warnings.catch_warnings():
        # A tube lattice outside its published validation ranges logs; nothing warns
        # that this module asserts on.
        warnings.simplefilter("ignore")
        return Simulation.from_capillary(
            15e-6,
            LENGTH,
            fill(),
            wavelength=800e-9,
            grid=grid,
            envelope=True,
            fwhm=20e-15,
            energy=energy,
            loss="gas",
            lattice=lattice,
            mode_average=mode_average,
            thg=False,
            plasma=False,
            # optmat's air Raman refuses a population it does not resolve (SEP 3, D16).
            raman=False,
            noise=None,
            absorber=False,
            n_save=3,
            stats=StatsSpec(names=("energy",)),
            schedule=FixedSchedule(np.full(16, LENGTH / 16)),
            integrator=FIXED_ORDER,
            backend=backend,
            n_r=None if mode_average else 8,
        )


def linear(simulation: Simulation) -> Simulation:
    """The same physics objects with no response (a capillary entry refuses none)."""
    return Simulation(
        simulation.grid,
        simulation.modes,
        [],
        simulation.initial,
        settings=simulation.settings,
        profiles=simulation.profiles,
        configuration=simulation.configuration,
    )


@pytest.mark.parametrize("lattice", [None, LATTICE], ids=["capillary", "lattice"])
@pytest.mark.parametrize("mode_average", [True, False], ids=["averaged", "resolved"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_linear_run_attenuates_by_the_integrated_gas_absorption(
    backend, mode_average, lattice
):
    """``|b(L)|^2 = |b(0)|^2 exp(-int alpha dz)`` bin by bin, alpha the gas-only mode's.

    The true-power amplitude carries the varying index's ``Phi_w`` (conventions,
    section 4), so its modulus follows the loss alone. The reference integrates the
    mode's own ``alpha(omega, z)`` by 16-point Gauss-Legendre, converged to rounding
    for this smooth path; the operator table is accurate to ``DEFAULT_TOL`` (1e-12)
    of its own scale, so 1e-10 on an exponent of order one keeps two orders in hand.
    The exponent is the absorber's alone: one, to the 1e-3 of ``Re n_eff`` (and,
    for the lattice, its resonant decrement), where the capillary's wall would add
    tens per metre.
    """
    simulation = linear(build(backend, mode_average=mode_average, lattice=lattice))
    output = simulation.run(progress=NullReporter())
    grid, mode = simulation.grid, simulation.modes[0]
    active = grid.active_mask
    omega = grid.omega[active]
    nodes, weights = np.polynomial.legendre.leggauss(16)
    exponent = sum(
        0.5 * LENGTH * weight * alpha(mode, omega, 0.5 * LENGTH * (node + 1))
        for node, weight in zip(nodes, weights, strict=True)
    )
    # The mean density times the length times the cross-section, over Re n_eff; the
    # lattice's decrement over the complex gas index adds 1.6e-2 near the tube
    # resonance at the blue edge of the band.
    np.testing.assert_allclose(
        exponent,
        CROSS_SECTION * PEAK_DENSITY / 2 * LENGTH,
        rtol=2e-3 if lattice is None else 2e-2,
    )
    entrance = np.abs(output["field"][0, 0, 0, active]) ** 2
    exit_ = np.abs(output["field"][-1, 0, 0, active]) ** 2
    np.testing.assert_allclose(np.log(exit_ / entrance), -exponent, rtol=0, atol=1e-10)


@pytest.mark.parametrize("mode_average", [True, False], ids=["averaged", "resolved"])
@pytest.mark.parametrize("backend", [name for name in BACKENDS if name != "numpy_ref"])
def test_backends_agree_with_the_reference_on_a_gas_loss_run(backend, mode_average):
    """A weakly nonlinear Kerr run, matched march against the NumPy oracle.

    The same fixed schedule and fixed-order Adams method on both, so they differ by
    rounding alone, within :func:`march_bounds.simulation_bound` of the oracle's run.
    """
    reference = build("numpy_ref", mode_average=mode_average, energy=1e-6)
    expected = reference.run(progress=NullReporter())["field"]
    actual = build(backend, mode_average=mode_average, energy=1e-6).run(
        progress=NullReporter()
    )["field"]
    assert_matched(actual, expected, simulation_bound(reference), what=backend)
