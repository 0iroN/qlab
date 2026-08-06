import numpy as np
import pytest

from qlab.engine.tdse.grid import Grid1D, Grid2D
from qlab.engine.tdse.initial import gaussian_packet, gaussian_packet_2d
from qlab.errors import InvalidInputError


@pytest.fixture
def grid():
    return Grid1D(n=2048, length=200.0)


def test_packet_is_normalised(grid):
    psi = gaussian_packet(grid, x0=-20.0, k0=1.0, sigma=5.0)
    assert np.sum(np.abs(psi) ** 2) * grid.dx == pytest.approx(1.0, abs=1e-12)


def test_packet_is_complex(grid):
    assert gaussian_packet(grid, x0=0.0, k0=1.0, sigma=5.0).dtype == np.complex128


def test_expectation_position_matches_x0(grid):
    psi = gaussian_packet(grid, x0=-20.0, k0=1.0, sigma=5.0)
    w = np.abs(psi) ** 2
    assert np.sum(w * grid.x) / np.sum(w) == pytest.approx(-20.0, abs=1e-6)


def test_position_spread_matches_sigma(grid):
    psi = gaussian_packet(grid, x0=0.0, k0=0.0, sigma=5.0)
    w = np.abs(psi) ** 2
    mean = np.sum(w * grid.x) / np.sum(w)
    variance = np.sum(w * (grid.x - mean) ** 2) / np.sum(w)
    assert np.sqrt(variance) == pytest.approx(5.0, rel=1e-6)


def test_expectation_momentum_matches_k0(grid):
    psi = gaussian_packet(grid, x0=0.0, k0=1.3, sigma=8.0)
    w = np.abs(np.fft.fft(psi)) ** 2
    assert np.sum(w * grid.k) / np.sum(w) == pytest.approx(1.3, abs=1e-6)


def test_rejects_unresolved_sigma(grid):
    with pytest.raises(InvalidInputError, match="çözünmez"):
        gaussian_packet(grid, x0=0.0, k0=1.0, sigma=grid.dx)


def test_rejects_momentum_beyond_nyquist():
    coarse = Grid1D(n=64, length=200.0)  # dx ~ 3.1 -> k_max ~ 1.0
    with pytest.raises(InvalidInputError, match="momentum"):
        gaussian_packet(coarse, x0=0.0, k0=5.0, sigma=10.0)


def test_packet_2d_is_normalised():
    grid2d = Grid2D(nx=256, ny=256, lx=60.0, ly=60.0)
    psi = gaussian_packet_2d(grid2d, x0=-10.0, y0=0.0, kx0=1.0, ky0=0.0, sigma=3.0)
    assert np.sum(np.abs(psi) ** 2) * grid2d.cell_area == pytest.approx(1.0, abs=1e-12)
