import numpy as np
import pytest

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.initial import gaussian_packet
from qlab.observable import tdse_obs


@pytest.fixture
def grid():
    return Grid1D(n=2048, length=200.0)


@pytest.fixture
def packet(grid):
    return gaussian_packet(grid, x0=-15.0, k0=1.2, sigma=6.0)


def test_norm_of_fresh_packet_is_one(grid, packet):
    assert tdse_obs.norm(packet, grid) == pytest.approx(1.0, abs=1e-12)


def test_density_is_real_and_nonnegative(packet):
    rho = tdse_obs.density(packet)
    assert rho.dtype == np.float64
    assert np.all(rho >= 0.0)


def test_expectation_position_recovers_x0(grid, packet):
    assert tdse_obs.expectation_position(packet, grid) == pytest.approx(-15.0, abs=1e-6)


def test_expectation_momentum_recovers_k0(grid, packet):
    assert tdse_obs.expectation_momentum(packet, grid) == pytest.approx(1.2, abs=1e-6)


def test_position_spread_recovers_sigma(grid, packet):
    assert tdse_obs.position_spread(packet, grid) == pytest.approx(6.0, rel=1e-6)


def test_free_packet_energy_matches_analytic(grid):
    """Serbest Gauss paketi: <E> = k0^2/2 + 1/(8 sigma^2)."""
    k0, sigma = 1.2, 6.0
    psi = gaussian_packet(grid, x0=0.0, k0=k0, sigma=sigma)
    expected = k0**2 / 2.0 + 1.0 / (8.0 * sigma**2)
    assert tdse_obs.expectation_energy(psi, grid, potentials.free(grid)) == pytest.approx(
        expected, rel=1e-6
    )


def test_probability_beyond_splits_symmetric_packet_in_half(grid):
    # Paketin merkezi ızgara noktasında yer alırsa, o hücre iki taraf arasında
    # bölünemez ve simetrik bölünme sağlanamaz. Bu nedenle paketin merkezini
    # iki ızgara noktası arasındaki orta noktaya yerleştiriyoruz.
    psi = gaussian_packet(grid, x0=grid.dx / 2, k0=0.0, sigma=5.0)
    assert tdse_obs.probability_beyond(psi, grid, boundary=grid.dx / 2) == pytest.approx(0.5, abs=1e-12)


def test_probability_beyond_does_not_renormalise(grid, packet):
    full = tdse_obs.probability_beyond(packet, grid, boundary=-15.0)
    half = tdse_obs.probability_beyond(0.5 * packet, grid, boundary=-15.0)
    assert half == pytest.approx(0.25 * full, rel=1e-9)


def test_edge_probability_is_negligible_for_centred_packet(grid, packet):
    assert tdse_obs.edge_probability(packet, grid, width=20.0) < 1e-12
