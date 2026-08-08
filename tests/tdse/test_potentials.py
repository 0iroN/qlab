import numpy as np
import pytest

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid1D, Grid2D
from qlab.errors import InvalidInputError


@pytest.fixture
def grid():
    return Grid1D(n=512, length=40.0)


def test_free_is_all_zero(grid):
    v = potentials.free(grid)
    assert v.shape == (512,)
    assert np.all(v == 0.0)


def test_barrier_height_inside_and_zero_outside(grid):
    v = potentials.rectangular_barrier(grid, height=0.8, width=2.0, center=0.0)
    assert np.all(v[np.abs(grid.x) < 0.9] == pytest.approx(0.8))
    assert np.all(v[np.abs(grid.x) > 1.1] == 0.0)


def test_barrier_rejects_nonpositive_width(grid):
    with pytest.raises(InvalidInputError, match="genişli"):
        potentials.rectangular_barrier(grid, height=1.0, width=0.0)


def test_square_well_is_zero_inside_and_high_outside(grid):
    v = potentials.square_well(grid, width=4.0, wall_height=1e6)
    assert v[np.argmin(np.abs(grid.x))] == pytest.approx(0.0)
    assert v[0] == pytest.approx(1e6)


def test_square_well_rejects_nonpositive_wall_height(grid):
    with pytest.raises(InvalidInputError, match="wall_height"):
        potentials.square_well(grid, width=4.0, wall_height=0.0)


def test_harmonic_matches_formula(grid):
    omega = 0.5
    # rtol=0.0: varsayılan rtol=1e-7 yazılan atol=1e-12'yi ezmesin. Ölçülen en
    # büyük mutlak fark tam 0.0 (bit-bit eşit; potentials.harmonic aynı
    # ifadeyi hesaplıyor) olduğundan yazarın atol=1e-12'si korunuyor.
    np.testing.assert_allclose(
        potentials.harmonic(grid, omega=omega),
        0.5 * omega**2 * grid.x**2,
        atol=1e-12,
        rtol=0.0,
    )


def test_harmonic_rejects_nonpositive_mass(grid):
    with pytest.raises(InvalidInputError, match="mass"):
        potentials.harmonic(grid, omega=0.5, mass=0.0)


def test_uniform_field_is_linear(grid):
    # rtol=0.0: varsayılan rtol=1e-7 yazılan atol=1e-12'yi ezmesin. Ölçülen en
    # büyük mutlak fark tam 0.0 (bit-bit eşit) olduğundan yazarın
    # atol=1e-12'si korunuyor.
    np.testing.assert_allclose(
        potentials.uniform_field(grid, strength=0.01), 0.01 * grid.x, atol=1e-12, rtol=0.0
    )


def test_absorbing_mask_is_one_in_centre_and_zero_at_edges(grid):
    mask = potentials.absorbing_mask(grid, width=5.0)
    assert mask[grid.n // 2] == pytest.approx(1.0)
    assert mask[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0 + 1e-12)


def test_absorbing_mask_rejects_too_wide(grid):
    with pytest.raises(InvalidInputError, match="0.4"):
        potentials.absorbing_mask(grid, width=0.5 * grid.length)


def test_double_slit_has_exactly_two_openings():
    grid2d = Grid2D(nx=256, ny=256, lx=40.0, ly=40.0)
    v = potentials.double_slit(
        grid2d, wall_x=0.0, thickness=1.0, slit_separation=8.0, slit_width=2.0
    )
    wall_column = int(np.argmin(np.abs(grid2d.X[:, 0])))
    open_cells = (v[wall_column, :] == 0.0).astype(int)
    assert np.sum(np.diff(open_cells) == 1) == 2
