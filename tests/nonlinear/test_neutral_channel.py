# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Neutral plasma channels: events that pay their energy and free no charge.

A neutral channel (``electrons_per_event=0``, kernel schema 42) depletes its population
at its rate and pays its energy per event through the loss current, exactly as an
ionisation does, and adds no free-electron current. These tests hold the response to
that analytically -- the energy identity, the loss current written out independently,
the Drude limit -- and the declaration to its refusals.
"""

import dataclasses

import numpy as np
import pytest

from plasma_cases import (
    AMPLITUDE,
    CHARGE_MASS_RATIO,
    DENSITY,
    gaussian_pulse_and_derivative,
    pulse,
)
from plasma_cases import power_rate as _power_rate
from solprop.exceptions import MaterialError, TranslationError
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.ionisation import (
    ELECTRONS_PER_EVENT,
    NeutralChannel,
    check_electrons_per_event,
)
from solprop.nonlinear.mixtures import responses_for
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import Gas, Mixture, ionisation_potential
from solprop.translate.responses import PlasmaChannel, plasma_coefficients

# A weak field: the eighth-power rate at 0.6 of the reference amplitude depletes a few
# per cent of the population, the regime the loss current is written for.
WEAK = 0.6 * AMPLITUDE

# The energy identity -int E' P dt = U (f_end - f_0) holds exactly in the continuum;
# both sides here are trapezoids of the same samples, so it converges at second order.
# The ionising channel's form of it, with its free-electron work cancelled, is held to
# 2e-4 at 8193 samples in test_plasma.py (measured 9.9e-5); the neutral channel has no
# free work to cancel, so the same budget applies to its whole polarisation.
ENERGY_RTOL = 2e-4


def power_rate():
    """The analytic eighth-power rate of the shared plasma cases."""
    return _power_rate()


def neutral(gas=Gas.Ar, **changes):
    """A neutral channel on the plain trapezoid, the rule the closed forms use."""
    return PlasmaResponse(
        gas,
        power_rate(),
        integration="trapezoid",
        electrons_per_event=0.0,
        **changes,
    )


def ionising(gas=Gas.Ar, **changes):
    """The same channel freeing one electron per event."""
    return PlasmaResponse(gas, power_rate(), integration="trapezoid", **changes)


def work_per_molecule(response, field, derivative, dt):
    """The work the field does on the medium per molecule, by parts."""
    polarisation = response.evaluate(field[None, :], DENSITY, dt=dt, kind="field")[0]
    return -float(np.trapezoid(derivative * polarisation, dx=dt)) / DENSITY


def test_the_admitted_electrons_per_event_are_none_and_one():
    assert ELECTRONS_PER_EVENT == (0.0, 1.0)
    assert check_electrons_per_event(0) == 0.0
    assert check_electrons_per_event(1.0) == 1.0


@pytest.mark.parametrize("value", [2.0, 0.5, -1.0, float("nan")])
def test_another_number_of_electrons_is_refused_by_name(value):
    with pytest.raises(MaterialError, match="electrons_per_event"):
        PlasmaResponse(Gas.Ar, power_rate(), electrons_per_event=value)


def test_a_neutral_channel_pays_its_barrier_for_every_event_in_a_weak_field():
    # The analytic energy check: the field's work on the medium, per molecule, is the
    # energy per event times the fraction that underwent it, N U f for N molecules.
    _, dt, field, derivative = gaussian_pulse_and_derivative(2**13 + 1)
    field, derivative = WEAK / AMPLITUDE * field, WEAK / AMPLITUDE * derivative
    response = neutral()
    fraction = response.ionised_fraction(field[None, :], dt=dt, kind="field")
    events = float(fraction[-1] - fraction[0])
    assert 1e-3 < events < 0.1
    paid = work_per_molecule(response, field, derivative, dt)
    expected = response.potential * events
    assert response.potential == ionisation_potential(Gas.Ar)
    assert abs(paid / expected - 1.0) < ENERGY_RTOL


def test_the_neutral_polarisation_is_the_loss_current_alone():
    # Written out independently of the response: P = rho int U W (1 - f) / E dt, the
    # loss current of every event and no free-electron current.
    _, dt, field = pulse(amplitude=WEAK)
    response = neutral()
    rate = power_rate()(np.abs(field))
    fraction = response.ionised_fraction(field[None, :], dt=dt, kind="field")
    loss = np.zeros_like(field)
    lit = rate > 0.0
    loss[lit] = response.potential * rate[lit] * (1.0 - fraction[lit]) / field[lit]
    increments = 0.5 * dt * (loss[1:] + loss[:-1])
    expected = DENSITY * np.concatenate(([0.0], np.cumsum(increments)))
    actual = response.evaluate(field[None, :], DENSITY, dt=dt, kind="field")[0]
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    # Two cumulative trapezoids of the same samples, summed in the same order but for
    # the response's integrating a zero free current alongside: n eps per sample.
    bound = 8 * field.size * np.finfo(float).eps
    assert float(np.max(np.abs(actual - expected))) <= bound * scale


def cumulative(values, dt):
    """The cumulative trapezoid from zero at the first sample."""
    return np.concatenate(([0.0], np.cumsum(0.5 * dt * (values[1:] + values[:-1]))))


def test_the_ionising_polarisation_is_the_neutral_one_plus_the_free_current():
    # What one electron per event adds is the Drude polarisation
    # rho (e^2/m_e) int int f E dt dt, written out here, and nothing else.
    _, dt, field = pulse(amplitude=WEAK)
    fraction = neutral().ionised_fraction(field[None, :], dt=dt, kind="field")
    drude = DENSITY * cumulative(
        cumulative(CHARGE_MASS_RATIO * fraction * field, dt), dt
    )
    with_free = ionising().evaluate(field[None, :], DENSITY, dt=dt, kind="field")[0]
    without = neutral().evaluate(field[None, :], DENSITY, dt=dt, kind="field")[0]
    scale = float(np.max(np.abs(with_free)))
    # The free current is far from negligible here, so the identity is not empty.
    assert float(np.max(np.abs(drude))) > 1e-3 * scale
    # Three cumulative trapezoids, the two currents summed before the last: a few
    # n eps per sample of the largest polarisation.
    bound = 16 * field.size * np.finfo(float).eps
    assert float(np.max(np.abs(with_free - without - drude))) <= bound * scale


def test_a_depleted_neutral_population_has_no_polarisation_at_all():
    # At f = 1 an ionised gas keeps its Drude current; a neutral one has no free
    # current and nothing left to deplete, so its polarisation is exactly zero.
    _, dt, field = pulse()
    depleted = neutral(preionisation=1.0)
    assert not np.any(depleted.evaluate(field[None, :], DENSITY, dt=dt, kind="field"))
    assert np.any(
        ionising(preionisation=1.0).evaluate(
            field[None, :], DENSITY, dt=dt, kind="field"
        )
    )


def test_the_fraction_does_not_depend_on_the_charge_freed():
    _, dt, field = pulse()
    assert np.array_equal(
        neutral().ionised_fraction(field[None, :], dt=dt, kind="field"),
        ionising().ionised_fraction(field[None, :], dt=dt, kind="field"),
    )


def test_one_electron_per_event_is_the_default_exactly():
    # An explicit single ionisation is the response as it was: a product with one is
    # exact, so this is an exact no-op (structural), not a rounding comparison.
    _, dt, field = pulse()
    default = PlasmaResponse(Gas.Ar, power_rate(), integration="trapezoid")
    explicit = dataclasses.replace(default, electrons_per_event=1.0)
    assert not default.neutral
    assert np.array_equal(
        default.evaluate(field[None, :], DENSITY, dt=dt, kind="field"),
        explicit.evaluate(field[None, :], DENSITY, dt=dt, kind="field"),
    )


def test_a_neutral_channel_compiles_to_a_zero_free_current_coefficient():
    response = PlasmaResponse(Gas.Ar, OptmatRate("Ar", 800e-9), electrons_per_event=0.0)
    assert response.neutral
    leaf = plasma_coefficients(response, (1e9, 1e11), z_max=1.0)
    channel = PlasmaChannel(
        **{field.name: getattr(leaf, field.name) for field in dataclasses.fields(leaf)},
        label="Ar",
        density=leaf.preionisation,
        electrons_per_event=response.electrons_per_event,
    )
    assert channel.free_current_coefficient == 0.0
    assert leaf.free_current_coefficient == leaf.charge_mass_ratio
    with pytest.raises(TranslationError, match="electrons_per_event"):
        dataclasses.replace(channel, electrons_per_event=2.0)


# ------------------------------------------------------------- the declaration


def test_a_neutral_channel_selection_builds_a_neutral_response():
    mixture = Mixture.air(1.0)
    rate = OptmatRate("O2", 800e-9)
    built = responses_for(
        mixture,
        kerr=False,
        plasma={"N2": True, "O2": NeutralChannel(rate)},
        wavelength=800e-9,
    )
    plasma = {r.species: r for r in built if isinstance(r, PlasmaResponse)}
    assert set(plasma) == {"N2", "O2"}
    assert plasma["O2"].neutral and plasma["O2"].rate is rate
    assert not plasma["N2"].neutral


def test_envelope_plasma_has_no_neutral_form():
    mixture = Mixture.air(1.0)
    with pytest.raises(MaterialError, match="neutral channel"):
        responses_for(
            mixture,
            kerr=False,
            thg=False,
            plasma={"O2": NeutralChannel(OptmatRate("O2", 800e-9))},
            wavelength=800e-9,
            envelope=True,
        )
    assert not hasattr(EnvelopePlasmaResponse, "electrons_per_event")


def test_a_bare_neutral_channel_must_name_its_species():
    # A bare selection applies to every component; air's oxygen is not nitrogen.
    mixture = Mixture.air(1.0)
    with pytest.raises(MaterialError, match="bare plasma rate"):
        responses_for(
            mixture,
            kerr=False,
            plasma=NeutralChannel(OptmatRate("N2", 800e-9)),
            wavelength=800e-9,
        )
