import numpy as np
import pytest

from qlab.engine.tdse.grid import Grid1D, Grid2D
from qlab.errors import InvalidInputError


def test_dx_times_n_equals_length():
    assert Grid1D(n=64, length=10.0).dx * 64 == pytest.approx(10.0)


def test_x_starts_at_minus_half_length():
    grid = Grid1D(n=64, length=10.0)
    assert grid.x[0] == pytest.approx(-5.0)
    assert grid.x.shape == (64,)


def test_fft_roundtrip():
    grid = Grid1D(n=128, length=20.0)
    f = np.exp(-grid.x**2)
    np.testing.assert_allclose(np.fft.ifft(np.fft.fft(f)).real, f, atol=1e-12)


def test_spectral_second_derivative_matches_analytic():
    """En güçlü ızgara testi: f = sin(2*pi*x/L) icin f'' = -(2*pi/L)^2 f.

    k dizisinin sırasını, FFT konvansiyonunu ve 2*pi çarpanını aynı anda doğrular.
    """
    grid = Grid1D(n=256, length=8.0)
    wave_number = 2.0 * np.pi / grid.length
    f = np.sin(wave_number * grid.x)
    second = np.fft.ifft(-grid.k2 * np.fft.fft(f)).real
    np.testing.assert_allclose(second, -(wave_number**2) * f, atol=1e-10)


def test_k_max_equals_nyquist():
    grid = Grid1D(n=64, length=10.0)
    assert np.max(np.abs(grid.k)) == pytest.approx(np.pi / grid.dx)


def test_rejects_odd_n():
    with pytest.raises(InvalidInputError, match="çift"):
        Grid1D(n=65, length=10.0)


def test_rejects_nonpositive_length():
    with pytest.raises(InvalidInputError, match="pozitif"):
        Grid1D(n=64, length=0.0)


def test_grid2d_shapes_and_area():
    grid = Grid2D(nx=32, ny=64, lx=4.0, ly=8.0)
    assert grid.X.shape == grid.Y.shape == grid.k2.shape == (32, 64)
    assert grid.cell_area == pytest.approx((4.0 / 32) * (8.0 / 64))


def test_grid2d_spectral_laplacian():
    grid = Grid2D(nx=64, ny=64, lx=8.0, ly=8.0)
    kx0, ky0 = 2.0 * np.pi / grid.lx, 2.0 * np.pi / grid.ly
    f = np.sin(kx0 * grid.X) * np.sin(ky0 * grid.Y)
    laplacian = np.fft.ifft2(-grid.k2 * np.fft.fft2(f)).real
    np.testing.assert_allclose(laplacian, -(kx0**2 + ky0**2) * f, atol=1e-10)
