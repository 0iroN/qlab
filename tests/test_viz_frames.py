import numpy as np
import pytest

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.initial import gaussian_packet
from qlab.viz import frames


@pytest.fixture
def setup():
    grid = Grid1D(n=1024, length=200.0)
    psi = gaussian_packet(grid, x0=-30.0, k0=1.0, sigma=8.0)
    potential = potentials.rectangular_barrier(grid, height=0.8, width=2.0)
    return grid, psi, potential


def test_payload_arrays_share_grid_shape(setup):
    grid, psi, potential = setup
    payload = frames.build_payload(psi, grid, potential, boundary=5.0)
    for array in (payload.x, payload.density, payload.potential_scaled, payload.phase):
        assert array.shape == (grid.n,)


def test_density_integrates_to_norm(setup):
    grid, psi, potential = setup
    payload = frames.build_payload(psi, grid, potential, boundary=5.0)
    assert np.sum(payload.density) * grid.dx == pytest.approx(payload.norm)


def test_phase_is_within_pi(setup):
    grid, psi, potential = setup
    payload = frames.build_payload(psi, grid, potential, boundary=5.0)
    assert np.all(payload.phase >= -np.pi - 1e-12)
    assert np.all(payload.phase <= np.pi + 1e-12)


def test_potential_is_scaled_into_density_range(setup):
    grid, psi, potential = setup
    payload = frames.build_payload(psi, grid, potential, boundary=5.0)
    assert np.max(payload.potential_scaled) == pytest.approx(
        0.8 * np.max(payload.density), rel=1e-9
    )


def test_flat_potential_scales_to_zero(setup):
    grid, psi, _ = setup
    payload = frames.build_payload(psi, grid, potentials.free(grid), boundary=5.0)
    assert np.all(payload.potential_scaled == 0.0)


def test_transmitted_probability_is_reported(setup):
    grid, _, potential = setup

    # Paketin merkezi iki ızgara noktası arasındaki orta noktaya yerleştirilir.
    # Merkezi ızgara noktasında yer alırsa, o hücre iki taraf arasında bölünemez
    # ve asimetrik bölünme sağlanır.
    idx = np.searchsorted(grid.x, -30.0)
    if idx > 0 and idx < len(grid.x):
        x_offset = (grid.x[idx - 1] + grid.x[idx]) / 2
    else:
        x_offset = -30.0

    psi = gaussian_packet(grid, x0=x_offset, k0=1.0, sigma=8.0)
    payload = frames.build_payload(psi, grid, potential, boundary=x_offset)
    assert payload.transmitted == pytest.approx(0.5, abs=1e-12)
