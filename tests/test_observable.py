import numpy as np
import pytest

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.initial import gaussian_packet
from qlab.errors import InvalidInputError
from qlab.observable import tdse_obs


@pytest.fixture
def grid():
    return Grid1D(n=2048, length=200.0)


@pytest.fixture
def packet(grid):
    return gaussian_packet(grid, x0=-15.0, k0=1.2, sigma=6.0)


def test_norm_of_fresh_packet_is_one(grid, packet):
    # rel=0.0: varsayılan rel=1e-6 yazılan abs=1e-12'yi ezmesin. Ölçülen sapma
    # 1.11e-16; abs=1e-15 ~9x payla bağlayıcı.
    assert tdse_obs.norm(packet, grid) == pytest.approx(1.0, abs=1e-15, rel=0.0)


def test_density_is_real_and_nonnegative(packet):
    rho = tdse_obs.density(packet)
    assert rho.dtype == np.float64
    assert np.all(rho >= 0.0)


def test_expectation_position_recovers_x0(grid, packet):
    # rel=0.0: beklenen -15.0 için varsayılan rel=1e-6 (=1.5e-5) yazılan
    # abs=1e-6'dan daha gevşekti. Ölçülen sapma 1.776e-15; abs=1e-14 ~5.6x pay.
    assert tdse_obs.expectation_position(packet, grid) == pytest.approx(-15.0, abs=1e-14, rel=0.0)


def test_expectation_momentum_recovers_k0(grid, packet):
    # rel=0.0: fiilen uygulanan, varsayılan rel=1e-6'nın (=1.2e-6) yazılan
    # abs=1e-6'yı ezdiği haldi. Ölçülen sapma tam 0.0 (bit-bit eşit) olduğu
    # için yazarın abs=1e-6'sı korunuyor, sadece gerçekten enforce ediliyor.
    assert tdse_obs.expectation_momentum(packet, grid) == pytest.approx(1.2, abs=1e-6, rel=0.0)


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
    # rel=0.0: varsayılan rel=1e-6 (=5e-7) yazılan abs=1e-12'yi ezmesin.
    # Ölçülen sapma 5.55e-17; abs=3e-16 ~5.4x payla bağlayıcı.
    assert tdse_obs.probability_beyond(psi, grid, boundary=grid.dx / 2) == pytest.approx(
        0.5, abs=3e-16, rel=0.0
    )


def test_probability_beyond_does_not_renormalise(grid, packet):
    full = tdse_obs.probability_beyond(packet, grid, boundary=-15.0)
    half = tdse_obs.probability_beyond(0.5 * packet, grid, boundary=-15.0)
    assert half == pytest.approx(0.25 * full, rel=1e-9)


def test_edge_probability_is_negligible_for_centred_packet(grid, packet):
    assert tdse_obs.edge_probability(packet, grid, width=20.0) < 1e-12


# --- Çökmüş yoğunluk/spektrum koruması -----------------------------------
#
# Soğurucu maske (potentials.absorbing_mask) kenarlardan norm sızdırır ve
# uzun bir koşuda toplam yoğunluk keyfi olarak küçülebilir. Bu durumda
# 0/0'a bölüp NaN döndürmek yerine InvalidInputError fırlatılmalı — bkz.
# qlab.errors ilkesi ve tdse_obs._agirlik_toplamini_dogrula.


def test_expectation_position_rejects_collapsed_density(grid):
    psi = np.zeros(grid.n, dtype=np.complex128)
    with pytest.raises(InvalidInputError, match="Konum beklenti değeri"):
        tdse_obs.expectation_position(psi, grid)


def test_position_spread_rejects_collapsed_density(grid):
    psi = np.zeros(grid.n, dtype=np.complex128)
    with pytest.raises(InvalidInputError, match="Konum yayılımı"):
        tdse_obs.position_spread(psi, grid)


def test_expectation_momentum_rejects_collapsed_spectrum(grid):
    psi = np.zeros(grid.n, dtype=np.complex128)
    with pytest.raises(InvalidInputError, match="Momentum beklenti değeri"):
        tdse_obs.expectation_momentum(psi, grid)


def test_expectation_energy_rejects_collapsed_density(grid):
    psi = np.zeros(grid.n, dtype=np.complex128)
    with pytest.raises(InvalidInputError, match="Kinetik enerji"):
        tdse_obs.expectation_energy(psi, grid, potentials.free(grid))


def test_expectation_position_rejects_nan_psi(grid, packet):
    psi = packet.copy()
    psi[0] = np.nan
    with pytest.raises(InvalidInputError, match="Konum beklenti değeri"):
        tdse_obs.expectation_position(psi, grid)


def test_guard_does_not_fire_on_legitimate_very_small_density(grid, packet):
    # `packet` normalize edilmiş, yani toplam yoğunluğu 1/grid.dx ~= 10.24.
    # Genliği 1e-3 ile çarpmak, soğurucu maskenin uzun bir koşuda gerçekten
    # üretebileceği türden gerçek/meşru ama çok küçük bir durumu taklit
    # eder: toplam yoğunluk (1e-3)^2 * 10.24 ~= 1.024e-5 olur — meşru tam
    # değerin ~1e6 katı altında, ama eşiğin (1e-10) ~1e5 katı üstünde.
    # Ölçeklendirme oranları etkilemediği için beklenti değerleri tam
    # normalize edilmiş paketle aynı kalmalı; guard'ın hem tetiklenmediğini
    # hem de sonucun hâlâ fiziksel olarak doğru olduğunu kanıtlar.
    small_psi = 1e-3 * packet
    total_density = float(np.sum(tdse_obs.density(small_psi)))
    assert 1e-10 < total_density < 1e-4

    # rel=0.0: beklenen -15.0 için varsayılan rel=1e-6 (=1.5e-5) yazılan
    # abs=1e-6'dan daha gevşekti. Ölçülen sapma 1.776e-15; abs=1e-14 ~5.6x pay.
    assert tdse_obs.expectation_position(small_psi, grid) == pytest.approx(
        -15.0, abs=1e-14, rel=0.0
    )
    assert tdse_obs.position_spread(small_psi, grid) == pytest.approx(6.0, rel=1e-6)
    # rel=0.0: varsayılan rel=1e-6 (=1.2e-6) yazılan abs=1e-6'yı ezmesin.
    # Ölçülen sapma 2.22e-16; abs=1e-15 ~4.5x payla bağlayıcı.
    assert tdse_obs.expectation_momentum(small_psi, grid) == pytest.approx(
        1.2, abs=1e-15, rel=0.0
    )
    assert tdse_obs.expectation_energy(
        small_psi, grid, potentials.free(grid)
    ) == pytest.approx(
        tdse_obs.expectation_energy(packet, grid, potentials.free(grid)), rel=1e-6
    )
