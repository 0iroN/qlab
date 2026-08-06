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

    BAĞIMSIZ bir köşegenleştirme — qlab.engine'den hiçbir şey çağırmaz.
    Aşağıdaki test_particle_in_box_eigenvalues bunu yalnızca analitik
    formüle karşı doğrular; motoru (SplitOperatorPropagator) hiç devreye
    sokmaz. Motor seviyesinde kutu kapsaması için bkz.
    test_particle_in_box_ground_state_via_engine.
    """
    dx = length / (n_interior + 1)
    diagonal = np.full(n_interior, 1.0 / dx**2)
    off_diagonal = np.full(n_interior - 1, -0.5 / dx**2)
    values = eigh_tridiagonal(diagonal, off_diagonal, eigvals_only=True)
    return np.sort(values)[:n_levels]


def test_particle_in_box_eigenvalues():
    """E_n = n^2 pi^2 / (2 L^2), n = 1..5, %0.1 içinde.

    ÖNEMLİ SINIR: bu test qlab.engine'i hiç çağırmaz. box_eigenvalues_finite_difference
    kendi bağımsız merkezi-fark köşegenleştirmesini yapar ve sonucu analitik
    formülle karşılaştırır — SplitOperatorPropagator, potentials.square_well ya da
    tdse_obs hiçbiri burada devreye girmez. Yani bozuk bir motor (yanlış işaret,
    yanlış katsayı, potansiyele yanlış bağlanma) bu testi hâlâ geçer; bu sadece
    analitik kutu formülünün ikinci, bağımsız bir sayısal yöntemle çapraz
    doğrulanmasıdır, motorun kutuyu doğru yaydığının KANITI DEĞİLDİR. Motor
    seviyesinde gerçek kapsama için bkz. aşağıdaki
    test_particle_in_box_ground_state_via_engine — sonlu duvarlı bir kutuyu
    gerçekten SplitOperatorPropagator ile imajiner zamanda evrimleyen test odur.
    """
    length = 4.0
    computed = box_eigenvalues_finite_difference(length, n_interior=4000, n_levels=5)
    levels = np.arange(1, 6)
    analytic = levels**2 * np.pi**2 / (2.0 * length**2)
    np.testing.assert_allclose(computed, analytic, rtol=1e-3)


def test_particle_in_box_ground_state_via_engine():
    """Kutuda taban durumu — motoru gerçekten sıkıştıran test.

    test_particle_in_box_eigenvalues'un aksine bu test qlab.engine'i doğrudan
    çalıştırır: potentials.square_well ile üretilen SONLU duvarlı bir kutu
    potansiyelini SplitOperatorPropagator ile imajiner zamanda taban duruma
    yakınsatır, enerjiyi tdse_obs.expectation_energy ile ölçer. Bu, spec'in
    orijinal planından bilinçli olarak düşürülmüş kapsamı kapatır: sonsuz
    duvar (wall_height çok büyük) yerine SONLU duvar kullanılıyor, çünkü asıl
    motor riski (yanlış işaret, yanlış katsayı, potansiyele yanlış bağlanma)
    tam olarak sonlu duvarın sızma davranışını doğru yayabilmekte saklı;
    sonsuz duvarla test bu davranışı hiç egzersiz etmeden geçerdi.

    Yön (fiziksel beklenti): sonlu duvar, dalga fonksiyonunun klasik olarak
    yasak bölgeye sızmasına izin verir. Bu sızma kutuyu etkin olarak
    genişletir, dolayısıyla ölçülen enerji E_1 = pi^2/(2L^2) sonsuz-duvar
    değerinin HER ZAMAN ALTINDA kalmalı — üstteki bir sapma (yanlış yönde)
    motorun potansiyeli ters bağladığının işareti olurdu.

    Büyüklük (tolerans gerekçesi — iki hata kaynağı var, ölçülüp karşılaştırıldı):
      1. Sonlu duvar sızması (baskın): sızma derinliği (üstel sönüm uzunluğu)
         ~ 1/kappa, kappa = sqrt(2*V0). wall_height=1000 için kappa ~ 44.7,
         penetrasyon ~ 0.0224. Etkin uzunluk L_eff = L + 2/kappa ile
         E_tahmin = pi^2/(2 L_eff^2) ~ 0.301643 — gerçek sonlu-kuyu özdeğerini
         (transandant denklem k*tan(k*a) = kappa ile bağımsız çözüldüğünde:
         0.3016422) neredeyse birebir yeniden üretiyor. Sonsuz-duvar
         analitiğinden sapma ~%2.3 — bu, tolerans bandının hedeflediği asıl
         fiziksel etki.
      2. Izgara/dt ayrıklaştırması (ikincil): ölçülen değer 0.301404 çıkıyor
         (n=4096, L_grid=16, dt=2e-4, 15000 adım) — E_tahmin'den yalnızca
         ~%0.08 uzakta. dt büyütülürse (ör. 1e-3) bu kalan büyük ölçüde
         büyüyor (Trotter hatası dt*V0 ile ölçekleniyor); burada seçilen
         dt=2e-4 bunu bastırmak için bilinçli seçildi.
      wall_height=1000 aynı zamanda underflow rejiminden bilinçli olarak
      kaçınıyor: dt*wall_height/2 = 0.1 küçük olduğu için exp(-V dt/2) ~ 0.905,
      sıfıra tam alt-taşma (underflow) YOK. Çok büyük wall_height (ör. 1e6)
      denendiğinde exp(-V dt/2) tam sıfıra alt-taşıyor ve duvar sayısal olarak
      "sert" bir kesime dönüşüyor — bu rejimde motor artık sonlu-duvar
      fiziğini değil, ayrık ızgara/Trotter etkileşimini ölçüyor ve (ölçülen)
      sonuç sonsuz-duvar değerinin ÜSTÜNE çıkabiliyor (ör. V0=1e6, n=8192'de
      +%15 hata) — fiziksel yönü bozan bir numerik artefakt, bu yüzden burada
      kullanılmıyor.
      rel=0.01 (%1) toleransı ~%0.08'lik ayrıklaştırma kalanına 12 kat pay
      bırakırken, gerçek bir motor hatasını rahatça yakalar: propagator.py'de
      potansiyel yarı-adımının işareti (-0.5j -> +0.5j) çevrildiğinde ölçülen
      enerji ~1000 çıkıyor (E_1 ~0.308 yerine, ~3200x büyüklük hatası) — bkz.
      task-8-report.md, "enjekte edilmiş hata" bölümü.
    """
    length = 4.0
    wall_height = 1000.0
    analytic_infinite = np.pi**2 / (2.0 * length**2)

    # Etkin-kutu tahmini: sızma derinliği (yasak bölgedeki üstel sönüm
    # uzunluğu) 1/kappa, kappa = sqrt(2*V0). Kutu bu miktar kadar her iki
    # yönde "genişler" — bkz. docstring'teki tam gerekçe.
    penetration_depth = 1.0 / np.sqrt(2.0 * wall_height)
    effective_length = length + 2.0 * penetration_depth
    expected_finite_well = np.pi**2 / (2.0 * effective_length**2)

    grid = Grid1D(n=4096, length=16.0)
    potential = potentials.square_well(grid, width=length, wall_height=wall_height)
    psi = gaussian_packet(grid, x0=0.0, k0=0.0, sigma=0.7)
    prop = SplitOperatorPropagator(
        grid=grid, potential=potential, dt=2e-4, imaginary_time=True
    )
    for _ in range(15000):
        psi = prop.step(psi)

    measured = tdse_obs.expectation_energy(psi, grid, potential)

    # Yön kontrolü: sonlu duvar sızması enerjiyi HER ZAMAN düşürür.
    assert measured < analytic_infinite

    # Büyüklük kontrolü: etkin-kutu tahminine göre — gerekçe docstring'te.
    assert measured == pytest.approx(expected_finite_well, rel=0.01)


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
