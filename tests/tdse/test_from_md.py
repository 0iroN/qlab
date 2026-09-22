"""`md -> tdse` KENARI -- Modül 7 §7.2'nin kapanışı.

Bu dosyanın taşıyıcı iddiası tek cümledir: **köprü, elle yazılmış bir
potansiyel satırını katlanmayan bölgede BİT BİT tekrar üretir; katlandığı
yerde ise bunu SAKLAMAZ, SAYAR.**

PENCERE (her sayı bunun içindedir ve dışına taşınamaz):
  yüzey        `Morse(d_e = K_HF/(2a^2), r_e = R_E_BOHR, a = A_ESLESEN)`
               `bridge_obs`ten: K_HF = 0.5729363, R_E_BOHR = 1.3459193602,
               A_ESLESEN = 0.9137
  kütle        iki H, `amu_to_electron_mass(1.00782503207)` her biri
  ızgara       aşağıdaki her testte AYRI AYRI yazılı (L bir DÜĞMEDİR)
  paket        `gaussian_packet(x0 = A, k0 = 0, sigma = uyumlu)`
  adım         `dt = (2*pi/TAM_AU)/200`  (N_BOLME = 200)
  makine       bu makine / numpy 2.5.1 / Python 3.14.7 -- makineler arası
               tekrarlanabilirlik ÖLÇÜLMEDİ (Modül 7'nin açık beyanı)
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import Morse
from qlab.engine.md.integrator import velocity_verlet_step
from qlab.engine.md.state import MDState
from qlab.engine.tdse.from_md import (
    MdTdsePotansiyeli,
    potential_from_force_field,
    reduced_mass,
)
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.initial import gaussian_packet
from qlab.engine.tdse.propagator import SplitOperatorPropagator
from qlab.errors import InvalidInputError
from qlab.observable.bridge_obs import A_ESLESEN, K_HF, MU_H2, R_E_BOHR, TAM_AU
from qlab.observable.tdse_obs import edge_probability, expectation_position, norm
from qlab.units import amu_to_electron_mass

M_H = float(amu_to_electron_mass(1.00782503207))
A_MORSE = A_ESLESEN
D_E = K_HF / (2.0 * A_MORSE * A_MORSE)
KUTLELER = np.array([M_H, M_H])


def _alan() -> Morse:
    return Morse(d_e=D_E, r_e=R_E_BOHR, a=A_MORSE)


def _elle_yazilan_V(grid: Grid1D) -> np.ndarray:
    """Ön ölçümün harness'inin yazdığı satır -- köprünün ÖLÇÜLDÜĞÜ referans.

    Bu satır `r`yi İŞARETLİ alır; `ForceField` sözleşmesi ise bir NORM
    kullanır. Aradaki fark bu dosyanın ikinci yarısının konusudur.
    """
    return D_E * (1.0 - np.exp(-A_MORSE * grid.x)) ** 2


# --- İndirgenmiş kütle: KONVANSİYON DEĞİL TÜREV -----------------------------


def test_reduced_mass_esit_kutlede_m_bolu_iki_BIT_BIT():
    """Ön ölçümün "mu = m/2 BENİM koyduğum konvansiyon" beyanını KAPATIR."""
    mu = reduced_mass(KUTLELER)
    assert mu == M_H / 2.0
    assert mu == MU_H2  # `bridge_obs`in sevk edilmiş sayısıyla da bit bit


def test_reduced_mass_SECILEN_BICIM_tasmiyor_kitabi_bicim_TASIYOR():
    """Biçim seçiminin gerekçesi bir tercih değil bir ÖLÇÜMdür.

    `m = 1e150`'ye kadar iki biçim aynı; `1e155`'te kitabi biçim `inf`.
    """
    m = 1e150
    assert m * (m / (m + m)) == m * m / (m + m)

    m = 1e155
    assert np.isinf(m * m / (m + m))
    secilen = reduced_mass(np.array([m, m]))
    assert np.isfinite(secilen)
    assert secilen == 5e154


@pytest.mark.parametrize(
    "kutleler",
    [np.array([1.0, 2.0, 3.0]), np.array([1.0]), np.array([[1.0, 2.0]])],
)
def test_reduced_mass_iki_cisim_disini_REDDEDER(kutleler):
    with pytest.raises(InvalidInputError, match="iki cisim"):
        reduced_mass(kutleler)


@pytest.mark.parametrize(
    "kutleler",
    [
        np.array([0.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
    ],
)
def test_reduced_mass_gecersiz_kutleyi_REDDEDER(kutleler):
    with pytest.raises(InvalidInputError, match="pozitif ve sonlu"):
        reduced_mass(kutleler)


# --- Köprünün SADAKATİ: katlanmayan bölgede BİT BİT -------------------------


def test_kopru_KATLANMAYAN_kutuda_elle_yazilani_BIT_BIT_uretir():
    """L = 2.0 <= 2*r_e = 2.6918: hiç katlanma yok, fark TAM SIFIR.

    Bu, kenarın var olma kanıtıdır: köprü bir YAKLAŞIM getirmiyor, aynı
    yüzeyi sözleşme üzerinden okuyor.
    """
    grid = Grid1D(n=256, length=2.0)
    kayit = potential_from_force_field(grid, _alan(), r_e=R_E_BOHR, masses=KUTLELER)

    assert isinstance(kayit, MdTdsePotansiyeli)
    assert kayit.katlanan_nokta == 0
    assert kayit.katlandi_mi is False
    assert kayit.katlanma_esigi == -R_E_BOHR
    assert np.array_equal(kayit.potential, _elle_yazilan_V(grid))


def test_kopru_KATLANAN_kutuda_katlanmayan_bolgede_HALA_BIT_BIT():
    """L = 4.0 > 2*r_e: 84 nokta katlanır, GERİSİ hâlâ tam sıfır fark.

    Katlanmanın büyüklüğü de ölçülür: tüm ızgarada max|fark| = 9.075...
    hartree. Bu sayı GİZLENMEZ -- kaydın `katlanan_nokta` alanı onu
    çağıranın gözüne sokar.
    """
    grid = Grid1D(n=512, length=4.0)
    kayit = potential_from_force_field(grid, _alan(), r_e=R_E_BOHR, masses=KUTLELER)
    elle = _elle_yazilan_V(grid)

    assert kayit.katlanan_nokta == 84
    assert kayit.katlandi_mi is True

    katlanmayan = grid.x >= -R_E_BOHR
    assert np.array_equal(kayit.potential[katlanmayan], elle[katlanmayan])

    fark = np.abs(kayit.potential - elle)
    assert fark.max() == pytest.approx(9.075023016287, rel=1e-12, abs=0.0)
    assert np.count_nonzero(fark) == 84  # tam da katlanan noktalar


def test_katlanan_nokta_sayisi_KUTUYLA_BUYUR_ve_ESIK_r_e_dir():
    """Katlanma bir arıza değil, kutu boyutunun YAPISAL sonucudur."""
    beklenen = {2.0: 0, 2.69: 0, 4.0: 84, 6.0: 212}
    for uzunluk, n in ((2.0, 256), (2.69, 256), (4.0, 512), (6.0, 768)):
        grid = Grid1D(n=n, length=uzunluk)
        kayit = potential_from_force_field(
            grid, _alan(), r_e=R_E_BOHR, masses=KUTLELER
        )
        assert kayit.katlanan_nokta == beklenen[uzunluk], f"L={uzunluk}"
        assert kayit.katlanan_nokta == int(np.count_nonzero(grid.x < -R_E_BOHR))


def test_kopru_r_sifira_dusen_izgarayi_REDDEDER():
    """`Morse` r = 0'da tanımsız; sessizce `nan` taşımak YASAK."""
    # n = 8, L = 4.0 -> dx = 0.5, x = -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5
    # r_e = 1.0 ile x = -1.0 noktası tam r = 0'a düşer.
    grid = Grid1D(n=8, length=4.0)
    assert -1.0 in set(grid.x.tolist())  # tuzak gerçekten kurulu
    with pytest.raises(InvalidInputError, match="r = 0"):
        potential_from_force_field(grid, _alan(), r_e=1.0, masses=KUTLELER)


@pytest.mark.parametrize("r_e", [0.0, -1.0, np.nan, np.inf])
def test_kopru_gecersiz_r_e_yi_REDDEDER(r_e):
    grid = Grid1D(n=64, length=2.0)
    with pytest.raises(InvalidInputError, match="r_e pozitif"):
        potential_from_force_field(grid, _alan(), r_e=r_e, masses=KUTLELER)


# --- Kenarın SAYISI (gerçek iki-motor koşumu) -------------------------------
#
# BU TESTLER `slow` İŞARETİ TAŞIMIYOR, VE BU BİR ÖLÇÜME DAYANIYOR.
# `pyproject.toml:27` işareti "**uzun süren** fizik doğrulama testleri" diye
# tanımlar. Ölçüldü (22.09.2026, `--durations`): aşağıdaki üç koşum sırayla
# 0.32s / 0.24s / 0.23s sürüyor -- dosyanın tamamı 1.44s. Yani bunlar fizik
# doğrulamasıdır ama UZUN SÜREN DEĞİLDİR ve işaretin tanımına girmezler.
#
# Bu ayrım bu depoda bedeli ölçülmüş bir şeydir: `-m "not slow"` hızlı şeridi
# 1815 testin 139'unu eliyor ve elenenlerin içinde YEDİ MODÜLÜN YEDİSİNİN de
# fizik çıpaları var (`test_cipalar.py`, `test_tunneling.py`,
# `test_double_slit.py`, `test_physical.py`, `test_kenar_tabani.py`, ...).
# Yani hızlı şeridi koşan biri "süit yeşil" der ve hiçbir fizik iddiası
# sınanmamış olur. 0.3 saniyelik bir çıpayı o şeride sürgün etmenin
# KAZANCI SIFIR, bedeli ise bu kenarın hızlı şeritte KORUMASIZ kalmasıdır.

#: Sevk edilen kenar sayısı: PLATODA iki-motor Ehrenfest artığı EKSİ
#: DONDURULMUŞ-MD tabanı. Taban çıkarılmazsa sayının %31.9'u tek motorluktur
#: (kuantum paketinin sıfır-nokta bağ-uzama salınımı) ve kenar hakkında
#: HİÇBİR ŞEY söylemez -- ön ölçümün `cebir` merceğinin bulduğu kusur budur.
KENAR_SAYISI = 0.13167174221268158

#: GÖMME KATLANMASININ DİNAMİĞE BEDELİ: aynı pencerede köprü V'si ile elle
#: yazılan V'nin ürettiği iki-motor artığının farkı. Sıfır DEĞİL, ve artık
#: bir tahmin de değil.
KATLANMA_BEDELI = 2.4982917457005982e-08

#: Sekiz anlamlı hane SINIRI. Ön ölçümde iki bağımsız harness aynı sayıyı
#: 7 hanede tutturup 8.'de ayrıştı (0.19324427970960262 / 0.19324430469252007).
#: Bu yüzden assert'ler `rtol=1e-8` taşır ve bağlayıcılığı KIRMA TESTİYLE
#: kanıtlanır (aşağıda).
RTOL_SEKIZ_HANE = 1e-8


def _iki_motor_artigi(
    genlik: float, periyot: int, *, kopru: bool
) -> tuple[float, float, float]:
    omega = TAM_AU
    t0 = 2.0 * np.pi / omega
    dt = t0 / 200
    adim = 200 * periyot
    grid = Grid1D(n=512, length=4.0)
    alan = _alan()
    if kopru:
        kayit = potential_from_force_field(
            grid, alan, r_e=R_E_BOHR, masses=KUTLELER
        )
        potansiyel, mu = kayit.potential, kayit.reduced_mass
    else:
        potansiyel, mu = _elle_yazilan_V(grid), MU_H2
    sigma = np.sqrt(1.0 / (2.0 * mu * omega))
    psi = gaussian_packet(grid, x0=genlik, k0=0.0, sigma=sigma)
    prop = SplitOperatorPropagator(grid=grid, potential=potansiyel, dt=dt, mass=mu)
    durum = MDState(
        positions=np.array([[0.0, 0.0, 0.0], [R_E_BOHR + genlik, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        masses=KUTLELER,
    )
    _, kuvvet = alan(durum.positions)
    artik = 0.0
    kenar = 0.0
    for i in range(adim + 1):
        if i % 4 == 0:
            x_q = expectation_position(psi, grid)
            x_c = (
                float(np.linalg.norm(durum.positions[1] - durum.positions[0]))
                - R_E_BOHR
            )
            artik = max(artik, abs(x_q - x_c))
            kenar = max(kenar, edge_probability(psi, grid, width=0.2))
        if i < adim:
            psi = prop.step(psi)
            durum, _, kuvvet = velocity_verlet_step(durum, kuvvet, alan, dt)
    return artik, kenar, norm(psi, grid)


@pytest.mark.parametrize("periyot", [18, 24])
def test_KENAR_SAYISI_platoda_ve_TABAN_CIKARILMIS(periyot):
    """Kenarın sevk edilen sayısı -- ve tabanın neden çıkarıldığı.

    Taban (A = 0, MD tarafı her karede BİT BİT 0.0) tek motorluk bir
    büyüklüktür: kuantum paketinin Morse minimumundaki sıfır-nokta bağ-uzama
    salınımı. İki motorun AYRIŞMASI hakkında hiçbir şey söylemez, ama ham
    artığın %31.9'unu oluşturur.
    """
    iki_motor, kenar, n_son = _iki_motor_artigi(0.1, periyot, kopru=True)
    taban, _, _ = _iki_motor_artigi(0.0, periyot, kopru=True)

    # Paket kutunun kenarına DEĞMİYOR -- sayı kirlenmemiş.
    assert kenar < 1e-6
    assert n_son == pytest.approx(1.0, rel=0.0, abs=1e-11)

    assert iki_motor - taban == pytest.approx(
        KENAR_SAYISI, rel=RTOL_SEKIZ_HANE, abs=0.0
    )
    # Taban gerçekten büyük bir pay: çıkarılmasaydı sayı %46.7 fazla olurdu.
    assert taban / iki_motor == pytest.approx(0.3186, rel=1e-3, abs=0.0)


def test_KATLANMA_BEDELI_olculur_ve_SIFIR_DEGILDIR():
    """Gömme katlanmasının dinamiğe bedeli. Beyan değil, sayı."""
    kopru, _, _ = _iki_motor_artigi(0.1, 18, kopru=True)
    elle, _, _ = _iki_motor_artigi(0.1, 18, kopru=False)
    assert kopru != elle  # katlanma ÖLÇÜLEBİLİR bir iz bırakıyor
    assert abs(kopru - elle) == pytest.approx(
        KATLANMA_BEDELI, rel=RTOL_SEKIZ_HANE, abs=0.0
    )


def test_SEKIZ_HANE_toleransi_BAGLAYICIDIR_kirma_kanit():
    """`rtol=1e-8` dekoratif değil: `x(1+1e-7)` mutasyonu KIRMIZI olmalı.

    Bu depoda `atol=` yazıp `rtol` varsayılanının üstünü örtmek ve
    `pytest.approx`ın `abs`/`rel`in büyüğünü alması iki ayrı kusur olarak
    ölçüldü. Bu test o tuzağın bu dosyada AÇIK OLMADIĞINI kanıtlar.
    """
    mutant = KENAR_SAYISI * (1.0 + 1e-7)
    with pytest.raises(AssertionError):
        assert mutant == pytest.approx(KENAR_SAYISI, rel=RTOL_SEKIZ_HANE, abs=0.0)
    # ... ama 1e-9'luk bir oynama tolerans İÇİNDE kalır (sınır gerçekten orada)
    assert KENAR_SAYISI * (1.0 + 1e-9) == pytest.approx(
        KENAR_SAYISI, rel=RTOL_SEKIZ_HANE, abs=0.0
    )
