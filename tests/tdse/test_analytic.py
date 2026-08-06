"""Spec §5.3 — analitik referanslara karşı fizik doğrulaması."""

import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.initial import gaussian_packet
from qlab.engine.tdse.propagator import SplitOperatorPropagator, evolve
from qlab.observable import tdse_obs


def box_eigenvalues_finite_difference(
    length: float, n_interior: int, n_levels: int
) -> np.ndarray:
    """Dirichlet sınır koşullu kutuda H = -1/2 d^2/dx^2 özdeğerleri.

    Merkezi fark: diyagonal 1/dx^2, yan diyagonal -1/(2 dx^2).
    """
    dx = length / (n_interior + 1)
    diagonal = np.full(n_interior, 1.0 / dx**2)
    off_diagonal = np.full(n_interior - 1, -0.5 / dx**2)
    values = eigh_tridiagonal(diagonal, off_diagonal, eigvals_only=True)
    return np.sort(values)[:n_levels]


def test_particle_in_box_eigenvalues():
    """E_n = n^2 pi^2 / (2 L^2), n = 1..5, %0.1 içinde."""
    length = 4.0
    computed = box_eigenvalues_finite_difference(length, n_interior=4000, n_levels=5)
    levels = np.arange(1, 6)
    analytic = levels**2 * np.pi**2 / (2.0 * length**2)
    np.testing.assert_allclose(computed, analytic, rtol=1e-3)


def test_harmonic_ground_state_via_imaginary_time():
    """Harmonik osilatör taban durumu E_0 = omega/2 (kapalı form, sızma yok)."""
    grid = Grid1D(n=1024, length=40.0)
    omega = 1.0
    potential = potentials.harmonic(grid, omega=omega)

    # Bilerek yanlış genişlikte başla: tam taban durum sigma = 1/sqrt(2)
    psi = gaussian_packet(grid, x0=0.0, k0=0.0, sigma=2.0)
    prop = SplitOperatorPropagator(
        grid=grid, potential=potential, dt=0.005, imaginary_time=True
    )
    for _ in range(4000):
        psi = prop.step(psi)

    assert tdse_obs.expectation_energy(psi, grid, potential) == pytest.approx(
        0.5 * omega, rel=1e-4
    )


def test_free_packet_spreading_matches_analytic():
    """sigma(t) = sigma0 * sqrt(1 + (t / (2 m sigma0^2))^2)."""
    grid = Grid1D(n=4096, length=400.0)
    sigma0 = 5.0
    psi0 = gaussian_packet(grid, x0=0.0, k0=0.0, sigma=sigma0)
    prop = SplitOperatorPropagator(grid=grid, potential=potentials.free(grid), dt=0.05)

    frames, times = evolve(prop, psi0, n_steps=1000, save_every=1000)
    expected = sigma0 * np.sqrt(1.0 + (times[-1] / (2.0 * sigma0**2)) ** 2)
    assert tdse_obs.position_spread(frames[-1], grid) == pytest.approx(expected, rel=1e-2)


def test_energy_conserved_in_static_potential():
    grid = Grid1D(n=1024, length=100.0)
    potential = potentials.harmonic(grid, omega=0.3)
    psi0 = gaussian_packet(grid, x0=5.0, k0=0.0, sigma=3.0)
    prop = SplitOperatorPropagator(grid=grid, potential=potential, dt=0.01)

    initial = tdse_obs.expectation_energy(psi0, grid, potential)
    frames, _ = evolve(prop, psi0, n_steps=2000, save_every=2000)
    final = tdse_obs.expectation_energy(frames[-1], grid, potential)

    assert abs(final - initial) < 1e-6


@pytest.mark.slow
def test_time_step_error_is_second_order():
    """Strang bölmesi global O(dt^2): dt yarıya inince hata ~4 kat azalmalı.

    Referans: harmonik tuzakta coherent state, <x>(t) = x0 cos(omega t).
    """
    grid = Grid1D(n=1024, length=60.0)
    omega, x0, duration = 1.0, 3.0, 2.0
    potential = potentials.harmonic(grid, omega=omega)

    def final_position_error(dt: float) -> float:
        psi0 = gaussian_packet(grid, x0=x0, k0=0.0, sigma=1.0 / np.sqrt(2.0 * omega))
        prop = SplitOperatorPropagator(grid=grid, potential=potential, dt=dt)
        n_steps = int(round(duration / dt))
        frames, times = evolve(prop, psi0, n_steps=n_steps, save_every=n_steps)
        measured = tdse_obs.expectation_position(frames[-1], grid)
        return abs(measured - x0 * np.cos(omega * times[-1]))

    coarse = final_position_error(0.02)
    fine = final_position_error(0.01)

    assert fine < coarse
    assert coarse / fine == pytest.approx(4.0, rel=0.35)
