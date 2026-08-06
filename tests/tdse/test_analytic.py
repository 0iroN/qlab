"""Spec §5.3 — analitik referanslara karşı fizik doğrulaması."""

import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

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


def finite_well_ground_state_transcendental(width: float, wall_height: float) -> float:
    """Sonlu, simetrik kare kuyunun taban durum enerjisi — BAĞIMSIZ analitik referans.

    Çift-paritedeki taban durumu şu transandant denklemi sağlar (atomik
    birimler, m=1):
        k * tan(k*a) = kappa,   k = sqrt(2E),   kappa = sqrt(2*(V0-E)),
        a = width/2
    Kök, tan()'ın taban-durum dalında (k*a içinde (0, pi/2)) aranır: bu dal
    hiçbir asimptota çarpmaz ve sonlu bir kuyunun 1B'de HER ZAMAN en az bir
    bağlı durumu olduğu bilinen sonucuyla tutarlıdır (üst sınır k*a -> pi/2
    sonsuz-duvar limitine karşılık gelir). scipy.optimize.brentq ile
    sayısal olarak çözülür.

    ÖNEMLİ: bu fonksiyon motorun (SplitOperatorPropagator) ürettiği bir
    sonuç DEĞİLDİR — qlab.engine'den hiçbir şey çağırmaz. Sonlu kuyunun
    taban durumu için ikinci, bağımsız bir analitik referanstır; aşağıdaki
    test_particle_in_box_ground_state_via_engine motorun ürettiği enerjiyi
    tam olarak buna karşı ölçer.
    """
    a = width / 2.0
    e_infinite = np.pi**2 / (2.0 * width**2)

    def transcendental(energy: float) -> float:
        k = np.sqrt(2.0 * energy)
        kappa = np.sqrt(2.0 * (wall_height - energy))
        return k * np.tan(k * a) - kappa

    return brentq(transcendental, 1e-9, e_infinite * (1.0 - 1e-9), xtol=1e-14, rtol=1e-14)


@pytest.mark.slow
def test_particle_in_box_ground_state_via_engine():
    """Kutuda taban durumu — motoru gerçekten sıkıştıran ve katsayı hatasını yakalayan test.

    test_particle_in_box_eigenvalues'un aksine bu test qlab.engine'i doğrudan
    çalıştırır: potentials.square_well ile üretilen SONLU duvarlı bir kutu
    potansiyelini SplitOperatorPropagator ile imajiner zamanda taban duruma
    yakınsatır, enerjiyi tdse_obs.expectation_energy ile ölçer.

    GEÇMİŞ BULGU (bu testin önceki hâliyle ilgili, bkz. task-8-report.md
    "kapsama açığı" bölümü): wall_height=1000 (derin kuyu) ile E_1 ~ 0.30
    civarında bir taban duruma karşı çalışıldığında, potansiyel bağlanmasında
    (coupling) %2'lik, hatta %100'lük bir katsayı hatası SESSİZCE geçiyordu.
    Neden fiziksel: V0 >> E_1 rejiminde taban durum enerjisi kuyu derinliğine
    yalnızca ~1/sqrt(V0) ile ölçeklenen ZAYIF bir düzeltme üzerinden bağlı;
    orantısal katsayı hataları bu karekök tarafından güçlü biçimde bastırılır.
    Bağımsız bir izole propagator ile (yalnızca potansiyel bağlanmasını
    ölçekleyen, src/ dışında bir betikte) ölçüldü: wall_height=1000'de +%5
    hata -> ölçülen E = 0.301431, -%5 hata -> E = 0.301389 (ikisi de o zamanki
    rel=0.01 toleransını rahatça geçiyordu) — bkz. task-8-report.md.

    DÜZELTME: kuyu SIĞLAŞTIRILDI (wall_height=1.3), yani V0 artık E_1'e yakın
    (V0/E_1 ~ 7.3, derin kuyudaki ~3250 yerine). Bu rejimde etkin-uzunluk
    yaklaşımı (L_eff = L + 2/kappa) yeterince doğru değil, bu yüzden aşağıda
    TAM transandant denklem (finite_well_ground_state_transcendental, yukarı
    bakınız) referans olarak kullanılıyor.

    Derinlik ARAMAYLA seçildi, tahmin edilerek değil: n=8192..32768 ızgara
    çözünürlüğü ve wall_height=0.4..1000 aralığında sistematik tarama
    yapıldı (bkz. task-8-report.md, "derinlik taraması" bölümü). Bulgu: bu
    "yanlış-H ile evrimlet, doğru-H ile ölç" test şemasında, ölçülen enerjinin
    E_ref'ten sapması katsayı hatasında YAKLAŞIK İKİNCİ DERECEDEN büyüyor
    (varyasyonel ilkeyle tutarlı: yanlış H'nin taban durumu, GERÇEK H altında
    ölçüldüğünde enerjiyi yalnızca ikinci mertebede yükseltir) — %5 hatanın
    etkisi %10 hatanınkinden ~4x değil ~3.8x küçük çıkıyor (ölçülen), yani
    doğrusal değil karesel bir bastırma var. Bu, wall_height=1.3'te bile %5
    hatanın ayrıklaştırma kalanından GÜVENLE ayrılamadığı, ama %10 hatanın
    rahatça ayrıldığı anlamına geliyor — aşağıda ölçülüp gösterildi.

    Izgara/yakınsama (n=32768, L_grid=16, dt=2e-4, 45000 adım = imajiner
    süre tau=9.0):
      - Uzamsal kalan baskın hata kaynağı: FFT tabanlı spektral yöntem,
        potansiyeldeki SÜREKSİZLİĞİ (x=+-2'de V: 0 -> 1.3 sıçraması) yalnızca
        O(dx) ile (spektral değil, cebirsel) çözüyor — n ikiye katlanınca
        kalan yaklaşık yarıya iniyor (ölçüldü: n=4096 -> %0.154, n=8192 ->
        %0.077, n=16384 -> %0.036, n=32768 -> %0.017; bkz. task-8-report.md).
        n=32768'de tam yakınsamış ölçüm: measured=0.17762085, E_ref=
        0.17765102, kalan = -%0.017.
      - İmajiner-zaman yakınsaması: bastırma oranı exp(-(V0-E_1)*tau) ile
        gidiyor; V0-E_1 ~ 1.12 için tau=9 -> bastırma ~1e-4. steps=45000 ile
        steps=60000 arasında ölçülen enerji 10. ondalığa kadar AYNI (bkz.
        task-8-report.md, "yakınsama doğrulaması") — imajiner zaman kaynaklı
        kalan pratikte sıfır, ölçülen -%0.017 kalanın TAMAMI uzamsal.
      - Kenar kontrolü: sığ kuyuda sızma derinliği ~1/kappa ~ 0.62 (kappa =
        sqrt(2*(1.3-0.178))), ızgara yarı-genişliği 8 — bol pay var. Ölçülen
        kenar olasılığı (dış 1.0 birimlik şerit): 1.36e-08 — ihmal edilebilir,
        aşağıda assert edilir.

    Tolerans (rel=5e-4, %0.05): ölçülen -%0.017 kalana ~2.9x pay bırakıyor
    (uydurma değil — kalan bağımsız ölçüldü, tolerans ona göre seçildi).
    Bu payla GERÇEKTEN YAKALANAN hata sınıfı ölçülüp doğrulandı:
      - İşaret hatası (propagator.py potansiyel yarı-adımı -0.5j -> +0.5j):
        ölçülen E = 1.335799 (E_1 ~0.178 yerine, sapma +%652) — YAKALANIYOR.
      - +%10 katsayı hatası: ölçülen E = 0.17784906, sapma +%0.111 —
        toleransı aşıyor, YAKALANIYOR.
      - -%10 katsayı hatası: ölçülen E = 0.17791983, sapma +%0.151 —
        toleransı aşıyor, YAKALANIYOR.
      - +%5 katsayı hatası: ölçülen E = 0.17768183, sapma +%0.017 —
        toleransın İÇİNDE kalıyor, YAKALANMIYOR.
      - -%5 katsayı hatası: ölçülen E = 0.17769018, sapma +%0.022 —
        toleransın İÇİNDE kalıyor, YAKALANMIYOR.
    DÜRÜST SINIR: bu test %5'lik bir katsayı hatasını güvenle yakalamıyor —
    ayrıklaştırma kalanı (%0.017) ile %5 hatanın sinyali (%0.017-0.022) aynı
    büyüklük mertebesinde, aralarında güvenli bir tolerans bandı yok (n'i çok
    daha da büyütmek gerekirdi, dakikalar mertebesinde çalışma süresi
    pahasına — "makul" bir birim testi için orantısız). Bu testin GERÇEK
    garantisi: potansiyel bağlanmasındaki ~%10 ve üzeri katsayı hataları,
    işaret hataları ve potansiyelin motora hiç bağlanmaması (bkz. eski
    versiyondaki "coupling=0" bulgusu, wall_height=1000'de E~156 veriyordu,
    burada da aynı büyüklükte bir kaçış beklenir). Tüm rakamlar
    task-8-report.md'de tekrar üretilebilir biçimde kayıtlı.

    Yön kontrolü (korunuyor): sonlu duvar sızması enerjiyi HER ZAMAN E_1'in
    (sonsuz-duvar) ALTINDA tutar — bkz. aşağıdaki assert.
    """
    length = 4.0
    wall_height = 1.3
    analytic_infinite = np.pi**2 / (2.0 * length**2)
    expected_finite_well = finite_well_ground_state_transcendental(length, wall_height)

    grid = Grid1D(n=32768, length=16.0)
    potential = potentials.square_well(grid, width=length, wall_height=wall_height)
    psi = gaussian_packet(grid, x0=0.0, k0=0.0, sigma=0.7)
    prop = SplitOperatorPropagator(
        grid=grid, potential=potential, dt=2e-4, imaginary_time=True
    )
    for _ in range(45000):
        psi = prop.step(psi)

    measured = tdse_obs.expectation_energy(psi, grid, potential)

    # Kenar kontrolü: sığ kuyuda dalga fonksiyonu derin kuyuya göre çok daha
    # fazla sızar — ızgara ucunda anlamlı olasılık birikmediğini doğrula.
    edge = tdse_obs.edge_probability(psi, grid, width=1.0)
    assert edge < 1e-6

    # Yön kontrolü: sonlu duvar sızması enerjiyi HER ZAMAN düşürür.
    assert measured < analytic_infinite

    # Büyüklük kontrolü: TAM transandant referansa göre — gerekçe docstring'te.
    assert measured == pytest.approx(expected_finite_well, rel=5e-4)


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
