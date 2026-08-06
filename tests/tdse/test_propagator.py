import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid1D, Grid2D
from qlab.engine.tdse.initial import gaussian_packet, gaussian_packet_2d
from qlab.engine.tdse.propagator import SplitOperatorPropagator, evolve
from qlab.errors import InvalidInputError, UnitarityViolationError


@pytest.fixture
def grid():
    return Grid1D(n=1024, length=100.0)


@pytest.fixture
def grid2d():
    # Kare ızgara (nx == ny): şekil doğrulamasının kaçırabileceği en zor durum,
    # çünkü kare bir dizinin transpozu .shape'i değiştirmez.
    return Grid2D(nx=32, ny=32, lx=20.0, ly=20.0)


def norm_of(psi, grid):
    return float(np.sum(np.abs(psi) ** 2) * grid.dx)


def norm_of_2d(psi, grid):
    return float(np.sum(np.abs(psi) ** 2) * grid.cell_area)


def test_single_step_preserves_norm(grid):
    psi = gaussian_packet(grid, x0=0.0, k0=1.0, sigma=4.0)
    prop = SplitOperatorPropagator(
        grid=grid, potential=potentials.harmonic(grid, omega=0.2), dt=0.01
    )
    assert norm_of(prop.step(psi), grid) == pytest.approx(1.0, abs=1e-12)


@settings(deadline=None, max_examples=25)
@given(
    dt=st.floats(min_value=1e-3, max_value=0.05),
    barrier_height=st.floats(min_value=0.0, max_value=2.0),
)
def test_norm_conserved_for_arbitrary_dt_and_potential(dt, barrier_height):
    grid = Grid1D(n=512, length=100.0)
    psi = gaussian_packet(grid, x0=-10.0, k0=0.8, sigma=4.0)
    prop = SplitOperatorPropagator(
        grid=grid,
        potential=potentials.rectangular_barrier(grid, height=barrier_height, width=2.0),
        dt=dt,
    )
    for _ in range(50):
        psi = prop.step(psi)
    assert norm_of(psi, grid) == pytest.approx(1.0, abs=1e-10)


def test_free_packet_centre_moves_at_group_velocity(grid):
    x0, k0 = -20.0, 1.0
    psi0 = gaussian_packet(grid, x0=x0, k0=k0, sigma=5.0)
    prop = SplitOperatorPropagator(grid=grid, potential=potentials.free(grid), dt=0.05)
    frames, times = evolve(prop, psi0, n_steps=200, save_every=200)
    w = np.abs(frames[-1]) ** 2
    assert np.sum(w * grid.x) / np.sum(w) == pytest.approx(x0 + k0 * times[-1], abs=1e-3)


def test_evolve_saves_expected_number_of_frames(grid):
    psi0 = gaussian_packet(grid, x0=0.0, k0=0.5, sigma=5.0)
    prop = SplitOperatorPropagator(grid=grid, potential=potentials.free(grid), dt=0.01)
    frames, times = evolve(prop, psi0, n_steps=100, save_every=25)
    assert frames.shape == (5, grid.n)  # t=0 dahil
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(1.0)


def test_raises_when_norm_breaks(grid, monkeypatch):
    """Korumanın gerçekten ateşlendiğini doğrula.

    Üniter yöntemde norm doğal olarak bozulmadığı için step() bilerek bozulur.
    """
    psi0 = gaussian_packet(grid, x0=0.0, k0=0.5, sigma=5.0)
    prop = SplitOperatorPropagator(grid=grid, potential=potentials.free(grid), dt=0.01)
    original = prop.step
    monkeypatch.setattr(prop, "step", lambda psi: 1.5 * original(psi))
    with pytest.raises(UnitarityViolationError) as excinfo:
        evolve(prop, psi0, n_steps=5)
    assert excinfo.value.step == 1


def test_imaginary_time_converges_to_harmonic_ground_state_energy(grid):
    """Enerjinin sadece azaldığını değil, doğru değere yakınsadığını doğrula.

    Yalnızca "enerji azaldı" iddiası zayıf: işaret hatası ya da üstel
    ifadede 2 kat hata gibi birçok yanlış-ama-monoton dinamik de bu testi
    geçer. omega=1.0 harmonik osilatörün taban durum enerjisi atomik
    birimlerde analitik olarak E0 = omega/2 = 0.5 olduğu için, yakınsanan
    değeri bu referansla karşılaştırmak yöntemin katsayılarını da sınar.
    500 adım x dt=0.01 fazlasıyla yeterli: deneysel olarak enerji bu noktada
    0.5'ten ~1e-9 uzaklıkta (bkz. task-6-report.md), abs=1e-2 toleransı
    bolca pay bırakıyor.
    """
    omega = 1.0
    potential = potentials.harmonic(grid, omega=omega)
    psi = gaussian_packet(grid, x0=0.0, k0=0.0, sigma=1.5)
    prop = SplitOperatorPropagator(
        grid=grid, potential=potential, dt=0.01, imaginary_time=True
    )

    def energy(state):
        wk = np.abs(np.fft.fft(state)) ** 2
        kinetic = np.sum(wk * grid.k2 / 2.0) / np.sum(wk)
        wx = np.abs(state) ** 2
        return kinetic + np.sum(wx * potential) / np.sum(wx)

    before = energy(psi)
    for _ in range(500):
        psi = prop.step(psi)
    assert energy(psi) < before
    assert energy(psi) == pytest.approx(0.5 * omega, abs=1e-2)
    assert norm_of(psi, grid) == pytest.approx(1.0, abs=1e-10)


def test_rejects_potential_with_wrong_shape(grid):
    with pytest.raises(InvalidInputError, match="şekli"):
        SplitOperatorPropagator(grid=grid, potential=np.zeros(7), dt=0.01)


def test_rejects_nonpositive_dt(grid):
    with pytest.raises(InvalidInputError, match="dt"):
        SplitOperatorPropagator(grid=grid, potential=potentials.free(grid), dt=0.0)


def test_step_rejects_psi_with_wrong_1d_length(grid):
    prop = SplitOperatorPropagator(grid=grid, potential=potentials.free(grid), dt=0.01)
    psi = np.zeros(grid.n + 1, dtype=np.complex128)
    with pytest.raises(InvalidInputError, match="şekli"):
        prop.step(psi)


def test_step_rejects_psi_with_wrong_2d_shape(grid2d):
    # Aynı toplam eleman sayısı (32*32 == 16*64), farklı şekil tuple'ı —
    # doğrulamanın gerçekten şekle baktığını, sadece boyut sayısına değil.
    # Kasıtlı olarak .T KULLANILMIYOR: kare bir dizide transpoz .shape'i
    # değiştirmez, bu yüzden yanlış sebepten geçen bir test olurdu (bkz.
    # step()'in docstring'i ve grid2d fixture'ının notu).
    potential = np.zeros((grid2d.nx, grid2d.ny))
    prop = SplitOperatorPropagator(grid=grid2d, potential=potential, dt=0.01)
    psi = np.zeros(grid2d.nx * grid2d.ny, dtype=np.complex128).reshape(16, 64)
    with pytest.raises(InvalidInputError, match="şekli"):
        prop.step(psi)


def test_2d_step_preserves_norm(grid2d):
    # Propagator'ın ilk gerçek 2B egzersizi: fft2/ifft2, cell_area ve
    # _expected_shape'in hepsi burada gerçekten çalıştırılıyor.
    psi = gaussian_packet_2d(grid2d, x0=0.0, y0=0.0, kx0=0.5, ky0=-0.3, sigma=2.0)
    potential = np.zeros((grid2d.nx, grid2d.ny))
    prop = SplitOperatorPropagator(grid=grid2d, potential=potential, dt=0.01)
    assert norm_of_2d(prop.step(psi), grid2d) == pytest.approx(1.0, abs=1e-12)
