# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify ``loss="gas"``: the fill's absorption kept, the guide's own loss removed.

The capillary takes an ideal wall, ``v_n = 0``, so the full model is
``sqrt(eps_co - (u/(ka))^2)`` and the reduced one drops its
loss term; the tube lattice drops its confinement term. The absorbing population
here is synthetic: a constant cross-section, so that its attenuation is a closed form.
"""

import math
import warnings
from dataclasses import replace

import mpmath as mp
import numpy as np
import pytest

from solprop.constants import C
from solprop.driver import Simulation
from solprop.exceptions import DriverError, ModeError
from solprop.modes import (
    CoherentWall,
    IndexTable,
    LossSpec,
    MarcatiliMode,
    MarcatiliModel,
    RectMode,
    SlabMode,
    TabulatedHollowMode,
    TabulatedMode,
    TubeLattice,
    VincettiMode,
)
from solprop.modes.base import alpha
from solprop.modes.capillary import _core_susceptibility
from solprop.physdata import Component, Mixture, Species, TabulatedSusceptibility
from solprop.profiles import Constant, Linear
from solprop.recipe.metadata import parameters
from solprop.recipe.registry import BuildContext, decode_keyword

EPS = np.finfo(float).eps

CROSS_SECTION = 1e-21
"""The synthetic absorber's intensity cross-section in m^2 (ozone's peak is 1.1e-21)."""

ABSORBER_DENSITY = 1e23
"""Its number density in m^-3, so that it attenuates by 100 /m."""

LATTICE = TubeLattice(210e-9, 5.55e-6, 6)


def omega_of(wavelength):
    return 2.0 * math.pi * C / np.asarray(wavelength)


OMEGA = omega_of(np.linspace(250e-9, 1600e-9, 23))


def absorber(cross_section=CROSS_SECTION):
    """A trace species with a constant refractivity and a constant cross-section."""
    omega = np.array([1e12, 1e18])
    optical = TabulatedSusceptibility(
        omega,
        np.full(2, 1e-29),
        np.full(2, cross_section),
        (2 * math.pi * C / omega[-1], 2 * math.pi * C / omega[0]),
        citation="Synthetic constant refractivity and absorption",
    )
    return Species("synthetic absorber", optical, gamma3=0.0)


def air_with_absorber(density=None):
    """Six bar of optmat's air with the synthetic absorber beside it."""
    air = Mixture.air(6.0)
    trace = Component(
        absorber(),
        Constant(ABSORBER_DENSITY) if density is None else density,
        "absorber",
    )
    return Mixture.of(*air.components, trace, temperature=air.temperature)


def capillary(
    model: MarcatiliModel = "full", loss: LossSpec = "gas", **kwargs
) -> MarcatiliMode:
    defaults = {"mixture": air_with_absorber(), "cladding": 1.45}
    return MarcatiliMode(15e-6, model=model, loss=loss, **(defaults | kwargs))


def susceptibility(mode, z=0.0):
    """The fill's summed susceptibility at one position, as the mode sums it."""
    terms = mode.sampled_terms(OMEGA)
    return _core_susceptibility(terms, mode._scalars(z)[1])


# ------------------------------------------------------------- the capillary


def test_gas_loss_is_the_ideal_wall_index():
    """The full model is ``sqrt(1 + chi - (u/(ka))^2)``, to the last few digits.

    The reference is that closed form at 40 digits from the mode's own sampled
    susceptibility, so it checks the arithmetic of the ideal wall and nothing about
    the material data. The rationalised ``sqrt(1 + x) - 1`` carries the relative
    rounding of ``x``: a few tens of roundings, gated at 64 eps relative to each part.
    """
    mode = capillary()
    chi = susceptibility(mode)
    actual = mode.neff_minus_one(OMEGA)
    mp.mp.dps = 40
    expected = np.array(
        [
            complex(
                mp.sqrt(
                    1
                    + mp.mpc(value.real, value.imag)
                    - (mp.mpf(mode.u_nm) * mp.mpf(C) / (mp.mpf(w) * mp.mpf(15e-6))) ** 2
                )
                - 1
            )
            for value, w in zip(chi, OMEGA, strict=True)
        ]
    )
    for part in (np.real, np.imag):
        np.testing.assert_allclose(part(actual), part(expected), rtol=64 * EPS, atol=0)


def test_gas_loss_keeps_the_fill_absorption_exactly():
    """With a real wall term, ``2 Re n Im n = Im chi`` holds identically.

    So ``alpha Re n_eff = omega Im chi / c = sum_s rho_s sigma_s``: here exactly the
    absorber's 100 /m, the same number the species attribution gives. The square root
    and the products round a handful of times: 64 eps.
    """
    mode = capillary()
    index = mode.neff(OMEGA)
    attenuation = alpha(mode, OMEGA) * np.real(index)
    np.testing.assert_allclose(
        attenuation, ABSORBER_DENSITY * CROSS_SECTION, rtol=64 * EPS, atol=0
    )
    attributed = mode.absorption_by_species(OMEGA)
    np.testing.assert_allclose(attributed.sum(axis=0), attenuation, rtol=64 * EPS)
    # The attribution is the same as with the model's own loss, structurally.
    np.testing.assert_array_equal(
        attributed, replace(mode, loss=True).absorption_by_species(OMEGA)
    )


@pytest.mark.parametrize("model", ["full", "reduced"])
def test_gas_loss_of_a_transparent_fill_is_exactly_lossless(model):
    """No absorbing population: no wall, no loss, an exact zero."""
    mode = capillary(model, mixture=Mixture.air(6.0))
    assert np.all(np.imag(mode.neff(OMEGA)) == 0.0)
    assert np.all(np.imag(mode.dneff_domega(OMEGA)) == 0.0)
    assert np.all(alpha(replace(mode, loss=True), OMEGA) > 0.0)


def test_gas_loss_removes_the_wall_loss_and_its_real_shift():
    """The whole cladding term goes, its real part with its loss.

    For a transparent cladding of index 1.45, ``v`` is real and the full model's
    ``n^2`` carries ``(u s)^2 (v s)^2``, ``s = c/(omega a)``, beside the loss; the
    ideal wall drops it, so ``Re n`` falls by ``(u s)^2 (v s)^2 / 2`` up to relative
    corrections of order ``(u s)^2`` and ``(v s)^2``, below 1e-2 over this band.
    The loss with the wall is larger by the wall's leakage.
    """
    gas = capillary(mixture=Mixture.air(6.0))
    wall = replace(gas, loss=True)
    shift = np.real(wall.neff_minus_one(OMEGA) - gas.neff_minus_one(OMEGA))
    permittivity = 1.45**2
    v = (permittivity + 1) / (2 * math.sqrt(permittivity - 1))
    s = C / (OMEGA * 15e-6)
    np.testing.assert_allclose(
        shift, 0.5 * (gas.u_nm * s) ** 2 * (v * s) ** 2, rtol=1e-2
    )
    # At 800 nm the ideal wall lowers the index by 3.3e-8.
    at_800 = omega_of([800e-9])
    shift_800 = np.real(wall.neff_minus_one(at_800) - gas.neff_minus_one(at_800))
    assert 3.2e-8 < float(shift_800[0]) < 3.4e-8


def test_reduced_gas_loss_is_the_polynomial_without_its_loss_term():
    """``1 + chi/2 - (u s)^2/2``: the lossless polynomial with chi's absorption.

    Its real part is exactly the ``loss=False`` value (the same products and sums),
    and its imaginary part is half of ``Im chi``, a multiplication by one half that
    is exact, then summed in the same order: 4 eps.
    """
    mode = capillary("reduced")
    values = mode.neff_minus_one(OMEGA)
    np.testing.assert_array_equal(
        np.real(values), np.real(replace(mode, loss=False).neff_minus_one(OMEGA))
    )
    np.testing.assert_allclose(
        np.imag(values), 0.5 * np.imag(susceptibility(mode)), rtol=4 * EPS, atol=0
    )


@pytest.mark.parametrize("model", ["full", "reduced"])
def test_gas_loss_derivative_matches_the_contour_derivative(model):
    """Cauchy's formula on the ideal-wall index of an evacuated core.

    The same check as the capillary module's (64 nodes on a circle of a tenth of the
    frequency, ``CONTOUR_TOLERANCE`` there): with ``v = 0`` the waveguide term is
    ``(u s)^2`` and its slope ``-2 (u s)^2 / omega``, exact.
    """
    mode = MarcatiliMode(50e-6, model=model, cladding=1.45 + 0.01j, loss="gas")
    omega = omega_of(np.linspace(300e-9, 2000e-9, 24))
    analytic = mode.dneff_domega(omega)
    total = np.zeros_like(omega, dtype=complex)
    n_points = 64
    for index in range(n_points):
        offset = 0.1 * omega * np.exp(2j * math.pi * index / n_points)
        shifted = replace(mode.sampled_terms(omega), omega=omega + offset)
        total = total + mode._model_values(shifted, 50e-6, 0.0)[0] / offset
    reference = total / n_points
    assert np.max(np.abs(analytic - reference)) / np.max(np.abs(analytic)) < 1e-11


@pytest.mark.parametrize("model", ["full", "reduced"])
def test_gas_loss_derivative_of_an_absorbing_fill_matches_differences(model):
    """The chain through the absorber's analytic slope, against a central difference.

    The capillary module's three-step ladder: a difference of an index of order one
    is floored near 1e-6 relative by rounding, so 5e-5 is the gate on each part.
    """
    mode = capillary(model)
    analytic = mode.dneff_domega(OMEGA)
    for relative_step in (1e-4, 1e-5, 1e-6):
        step = relative_step * OMEGA
        difference = (
            mode.neff_minus_one(OMEGA + step) - mode.neff_minus_one(OMEGA - step)
        ) / (2.0 * step)
        for part in (np.real, np.imag):
            scale = np.max(np.abs(part(analytic)))
            assert np.max(np.abs(part(difference) - part(analytic))) / scale < 5e-5


@pytest.mark.parametrize("model", ["full", "reduced"])
def test_gas_loss_slope_along_z_follows_a_varying_absorber_and_taper(model):
    """``dneff_dz`` against the fourth-order difference, real and imaginary parts.

    The absorber rises linearly while the bore narrows, so both scalars move; the
    imaginary slope is the absorber's alone. The capillary module's taper gate,
    1e-8, holds for a difference of these smooth paths.
    """
    length = 2e-4
    fill = air_with_absorber(Linear(0.0, 2 * ABSORBER_DENSITY, length))
    mode = MarcatiliMode(
        Linear(50e-6, 40e-6, length),
        mixture=fill,
        cladding=1.45,
        loss="gas",
        model=model,
    )
    z, step = 7.3e-5, 1e-6
    analytic = mode.dneff_dz(OMEGA, z)

    def at(position):
        return mode.neff_minus_one(OMEGA, position)

    difference = (
        8.0 * (at(z + step / 2) - at(z - step / 2)) - (at(z + step) - at(z - step))
    ) / (6.0 * step)
    for part in (np.real, np.imag):
        scale = np.max(np.abs(part(analytic)))
        assert np.max(np.abs(part(difference) - part(analytic))) / scale < 1e-8
    if model == "reduced":
        # The polynomial's imaginary part is chi's alone, so without the absorber's
        # slope its slope is an exact zero, taper or not.
        still = replace(mode, mixture=air_with_absorber())
        assert np.all(np.imag(still.dneff_dz(OMEGA, z)) == 0.0)


# ------------------------------------------------------------ the tube lattice


def test_vincetti_gas_loss_drops_the_confinement_term_only():
    """Every real term of the fits stays; only ``i c alpha / (2 omega)`` goes.

    For a transparent fill the confinement term is the whole imaginary part and is
    purely imaginary, so the real parts agree exactly and the gas-only index is real.
    """
    transparent = VincettiMode(
        15e-6, mixture=Mixture.air(6.0), lattice=LATTICE, loss="gas"
    )
    with_wall = replace(transparent, loss=True)
    np.testing.assert_array_equal(
        np.real(transparent.neff(OMEGA)), np.real(with_wall.neff(OMEGA))
    )
    assert np.all(np.imag(transparent.neff(OMEGA)) == 0.0)
    assert np.all(np.imag(with_wall.neff(OMEGA)) > 0.0)


def test_vincetti_gas_loss_keeps_the_fill_absorption():
    """The gas's absorption enters through its index and the decrement over it.

    ``Im n_eff = Im n_g (1 + W/|n_g|^2)`` with the decrement ``W`` below 2e-3 over
    this band, so ``alpha`` matches ``omega Im chi / c`` to 1e-2 relative, and is
    smaller than with the confinement loss beside it.
    """
    mode = VincettiMode(15e-6, mixture=air_with_absorber(), lattice=LATTICE, loss="gas")
    gas = alpha(mode, OMEGA)
    np.testing.assert_allclose(gas, ABSORBER_DENSITY * CROSS_SECTION, rtol=1e-2)
    assert np.all(alpha(replace(mode, loss=True), OMEGA) > gas)
    np.testing.assert_array_equal(
        mode.absorption_by_species(OMEGA),
        replace(mode, loss=True).absorption_by_species(OMEGA),
    )


# ------------------------------------------------------- refusals and surfaces


def test_an_unknown_loss_word_is_refused():
    mode = MarcatiliMode(15e-6, "He", 1.0)
    object.__setattr__(mode, "loss", "wall")
    with pytest.raises(ModeError, match='"gas"'):
        mode._check_fill_arguments()


@pytest.mark.parametrize("cls", [TabulatedHollowMode, TabulatedMode])
def test_a_table_refuses_gas_loss(cls):
    """A table's loss is the guide's, carried whole: there is no wall term to drop."""
    table = IndexTable([1e15, 2e15, 3e15, 4e15], [1 + 1e-8j] * 4)
    with pytest.raises(
        ModeError, match=f'{cls.__name__} does not implement loss="gas"'
    ):
        cls(15e-6, table=table, loss="gas")


@pytest.mark.parametrize(
    "build",
    [
        lambda: RectMode(50e-6, 12.5e-6, "Ar", 1.0, loss="gas"),
        lambda: SlabMode(25e-6, "Ar", 1.0, loss="gas"),
    ],
    ids=["rect", "slab"],
)
def test_the_wall_pair_guides_refuse_gas_loss(build):
    """They share the hollow-mode vocabulary but do not implement the gas-only part."""
    with pytest.raises(ModeError, match='does not implement loss="gas"'):
        build()


def test_a_coherent_wall_refuses_gas_loss():
    """The wall block is the loss model; a gas-only setting beside it is refused."""
    with pytest.raises(DriverError, match="coherent wall block is the loss model"):
        Simulation.from_capillary(
            50e-6,
            0.01,
            "He",
            1.0,
            wavelength=800e-9,
            fwhm=20e-15,
            energy=1e-9,
            modes=2,
            wall=CoherentWall(),
            loss="gas",
        )


def test_the_run_interfaces_forward_gas_loss():
    """from_capillary builds every mode, capillary or tube lattice, with it."""
    for lattice in (None, LATTICE):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulation = Simulation.from_capillary(
                15e-6,
                0.01,
                "He",
                1.0,
                wavelength=800e-9,
                wavelength_limits=(400e-9, 1600e-9),
                trange=100e-15,
                fwhm=20e-15,
                energy=1e-9,
                lattice=lattice,
                loss="gas",
                raman=False,
                plasma=False,
            )
        assert all(getattr(mode, "loss", None) == "gas" for mode in simulation.modes)


def test_gas_loss_prints_and_reads_from_a_recipe():
    mode = capillary()
    assert "loss=gas" in repr(mode)
    assert ("loss", "gas") in mode._describe()[1]
    entry = parameters("capillary")["loss"]
    assert decode_keyword(entry, "gas", BuildContext()) == "gas"
