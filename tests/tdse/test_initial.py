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
    # rel=0.0: varsayılan rel=1e-6 yazılan abs=1e-12'yi ezmesin. Ölçülen sapma
    # 1.11e-16; abs=1e-15 ~9x payla bağlayıcı.
    assert np.sum(np.abs(psi) ** 2) * grid.dx == pytest.approx(1.0, abs=1e-15, rel=0.0)


def test_packet_is_complex(grid):
    assert gaussian_packet(grid, x0=0.0, k0=1.0, sigma=5.0).dtype == np.complex128


def test_expectation_position_matches_x0(grid):
    psi = gaussian_packet(grid, x0=-20.0, k0=1.0, sigma=5.0)
    w = np.abs(psi) ** 2
    # rel=0.0: beklenen -20.0 için varsayılan rel=1e-6 (=2e-5) yazılan
    # abs=1e-6'dan daha gevşekti. Ölçülen sapma 3.55e-15; abs=2e-14 ~5.6x pay.
    assert np.sum(w * grid.x) / np.sum(w) == pytest.approx(-20.0, abs=2e-14, rel=0.0)


def test_position_spread_matches_sigma(grid):
    psi = gaussian_packet(grid, x0=0.0, k0=0.0, sigma=5.0)
    w = np.abs(psi) ** 2
    mean = np.sum(w * grid.x) / np.sum(w)
    variance = np.sum(w * (grid.x - mean) ** 2) / np.sum(w)
    assert np.sqrt(variance) == pytest.approx(5.0, rel=1e-6)


def test_expectation_momentum_matches_k0(grid):
    psi = gaussian_packet(grid, x0=0.0, k0=1.3, sigma=8.0)
    w = np.abs(np.fft.fft(psi)) ** 2
    # rel=0.0: beklenen 1.3 için varsayılan rel=1e-6 (=1.3e-6) yazılan
    # abs=1e-6'dan daha gevşekti. Ölçülen sapma 2.22e-16; abs=1e-15 ~4.5x pay.
    assert np.sum(w * grid.k) / np.sum(w) == pytest.approx(1.3, abs=1e-15, rel=0.0)


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
    # rel=0.0: varsayılan rel=1e-6 yazılan abs=1e-12'yi ezmesin. Ölçülen sapma
    # 2.22e-16; abs=1e-15 ~4.5x payla bağlayıcı.
    assert np.sum(np.abs(psi) ** 2) * grid2d.cell_area == pytest.approx(
        1.0, abs=1e-15, rel=0.0
    )


# Anizotropik ızgara: dx = 200/32 = 6.25 (kaba eksen), dy = 60/256 = 0.234375
# (ince eksen) — oran ~26.7x, bağlayıcı eksen açıkça dx'tir. `_validate`
# eşikleri: çözünürlük tabanı 2*dx_kullanılan, Nyquist sınırı
# 0.8*pi/dx_kullanılan; k_needed = |k0| + 2/sigma.
@pytest.fixture
def anisotropic_grid2d():
    return Grid2D(nx=32, ny=256, lx=200.0, ly=60.0)


def test_packet_2d_rejects_momentum_beyond_coarse_axis_nyquist(anisotropic_grid2d):
    # kx0=0.5, sigma=15 -> k_needed = 0.5 + 2/15 = 0.6333.
    # Kaba eksen (dx=6.25) Nyquist sınırı: 0.8*pi/6.25 = 0.4021 -> 0.6333 AŞAR.
    # İnce eksen (dy=0.234375) Nyquist sınırı: 0.8*pi/0.234375 = 10.723 -> AŞMAZ.
    # min(dx, dy) kullanan eski kodda bu paket sessizce kabul edilirdi;
    # max(dx, dy) kullanan düzeltmede kaba eksene göre reddedilmeli.
    with pytest.raises(InvalidInputError, match="momentum"):
        gaussian_packet_2d(
            anisotropic_grid2d, x0=0.0, y0=0.0, kx0=0.5, ky0=0.0, sigma=15.0
        )


def test_packet_2d_rejects_sigma_unresolved_on_coarse_axis(anisotropic_grid2d):
    # sigma=5.0: kaba eksende (2*dx=12.5) 5.0 < 12.5 -> ÇÖZÜNMEZ.
    # İnce eksende (2*dy=0.46875) 5.0 >= 0.46875 -> çözünür, sorun yok.
    # min(dx, dy) kullanan eski kodda bu paket sessizce kabul edilirdi;
    # max(dx, dy) kullanan düzeltmede kaba eksene göre reddedilmeli.
    with pytest.raises(InvalidInputError, match="çözünmez"):
        gaussian_packet_2d(
            anisotropic_grid2d, x0=0.0, y0=0.0, kx0=0.0, ky0=0.0, sigma=5.0
        )


def test_packet_2d_accepts_packet_valid_on_both_axes(anisotropic_grid2d):
    # kx0=0.05, sigma=20 -> k_needed = 0.05 + 2/20 = 0.15, kaba eksen
    # Nyquist sınırı 0.4021'in rahatça altında; sigma=20 kaba eksen
    # çözünürlük tabanı 2*dx=12.5'in de rahatça üstünde. Düzeltme paketi
    # aşırı reddetmiyor — geçerli bir paket hâlâ kabul ediliyor.
    psi = gaussian_packet_2d(
        anisotropic_grid2d, x0=0.0, y0=0.0, kx0=0.05, ky0=0.0, sigma=20.0
    )
    # rel=0.0: varsayılan rel=1e-6 yazılan abs=1e-12'yi ezmesin. Ölçülen sapma
    # 2.22e-16; abs=1e-15 ~4.5x payla bağlayıcı.
    assert np.sum(np.abs(psi) ** 2) * anisotropic_grid2d.cell_area == pytest.approx(
        1.0, abs=1e-15, rel=0.0
    )
