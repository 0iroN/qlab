"""velocity-Verlet: altı bağımsız ölçüt.

BU DOSYANIN EN ÖNEMLİ AYRIMI -- SALINIM ve SÜRÜKLENME AYNI ŞEY DEĞİLDİR.
velocity-Verlet SİMPLEKTİKTİR: enerji kaymaz, SALINIR. Ölçüldü (tasarım
§4.1), ikisi 10^6 kat farklıdır. "Enerji korunuyor" iddiası salınım
genliğiyle değil SÜRÜKLENME ile sınanmalı; salınımın kendisi ayrı ve daha
keskin bir sınavdır.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import Harmonic
from qlab.engine.md.integrator import run
from qlab.engine.md.state import MDState
from qlab.units import amu_to_electron_mass

#: H2'nin gerçek titreşimine denk fiziksel çıpa (tasarım §4.1/§4.3).
#: mu = m(1H)/2, k = mu * omega^2 ile omega_e = 4401.21 cm^-1.
#:
#: MU_H2 KÜTLE BORU HATTINDAN GEÇER, literal DEĞİLDİR. ÖLÇÜLDÜ: literal
#: yazıldığında `AMU_TO_ELECTRON_MASS = 1.0` mutantı bu dosyadaki ALTI testi
#: de geçiyordu -- çünkü `amu_to_electron_mass` hiç çağrılmıyordu ve modülün
#: "en tehlikeli hatasına karşı tek koruması" hiçbir yere bağlı değildi.
#:
#: K_H2 ise LİTERAL KALIR. İkisi birlikte türetilirse omega = sqrt(K/mu)
#: değişmez ve mutant yine kaçar (ölçüldü: o hâlde mutant da 4401.39 cm^-1
#: veriyor). Şimdiki hâliyle mutant 187918.70 cm^-1 veriyor ve çıpa kırılıyor.
MU_H2 = amu_to_electron_mass(1.00782503207) / 2.0   # 918.576324 m_e
K_H2 = 0.3693947966
T_H2 = 2 * np.pi * np.sqrt(MU_H2 / K_H2)  # 313.322877 a.u. = 7.578918 fs


def _tek_osilator(m: float, k: float, x0: float = 0.1) -> tuple[MDState, Harmonic]:
    durum = MDState(
        positions=np.array([[x0, 0.0, 0.0]]),
        velocities=np.zeros((1, 3)),
        masses=np.array([m]),
    )
    return durum, Harmonic(k=k, center=np.zeros((1, 3)))


def _periyot(yorunge) -> float:
    """Sıfır geçişlerinden periyot. x0 != 0 ve v0 = 0 bilerek seçildi."""
    x = yorunge.positions[:, 0, 0]
    t = yorunge.times
    gecis = np.nonzero(np.diff(np.sign(x)) != 0)[0]
    anlar = [
        t[i] + (t[i + 1] - t[i]) * abs(x[i]) / (abs(x[i]) + abs(x[i + 1]))
        for i in gecis
    ]
    return float(2.0 * np.mean(np.diff(anlar)))


def test_periyot_analitige_oturuyor():
    """T = 2*pi*sqrt(m/k). Bağıl hata YALNIZ N = T/dt'ye bağlı, m ve k'dan DEĞİL.

    ÖLÇÜLDÜ: kapalı form -pi^2/(6N^2); N=200'de bağıl hata 4.112758e-05,
    ölçülenle 4 hanede uyuşuyor. Altı farklı (m,k) çifti TAM AYNI değeri verdi.

    SINIR NEDEN 1e-3 ve N NEDEN >= 200: sistematik hata N=200'de 4.11e-05
    (marj 24x), N=100'de 1.645e-04 (marj 6x). Bu, dosyadaki EN DAR marjdır.
    HANGİ HATAYI YAKALAR: yarım tekmede 0.5 eksik (2.93e-01), kütle unutulmuş
    a=F (9.68e-01), kuvvet işareti ters (patlama), a=F/m^2 (patlama).
    YAKALAMAZ: forward Euler (3.29e-04), simplektik Euler (4.11e-05), bayat
    kuvvet (1.13e-04) -- o üçünde periyot doğru, bozulan ENERJİDİR.
    """
    for m, k in ((1.0, 1.0), (1.0, 0.25), (MU_H2, K_H2)):
        analitik = 2 * np.pi * np.sqrt(m / k)
        dt = analitik / 200
        durum, alan = _tek_osilator(m, k)
        yorunge = run(durum, alan, dt=dt, n_steps=200 * 10, save_every=1)
        assert abs(_periyot(yorunge) / analitik - 1.0) < 1e-3


def test_enerji_SURUKLENMIYOR():
    """Blok ortalamalarının baş-son farkı; SALINIM değil SÜRÜKLENME ölçülür.

    ÖLÇÜLDÜ (N=200, 1e5 adım, save_every=10): taban 2.322826e-10, sınır 1e-6
    (marj 4305x). TABAN save_every'YE DE BAĞLIDIR -- save_every=1'de
    1.213873e-10, save_every=20'de 6.357603e-10; bu yüzden save_every=10 şart.
    TOLERANS N'E BAĞLIDIR, SABİT YAZILAMAZ: N=100'de taban 5.697345e-08'e
    çıkar ve marj 18x'e iner -- bu yüzden N=200 sabitlendi.

    HANGİ HATAYI YAKALAR: forward Euler (3.03e+42), bayat kuvvet (1.70e+21),
    yarım tekme eksik (1.36e-03), a=F (2.77e-02).
    YAKALAMAZ: simplektik Euler (2.01e-07) -- o da simplektiktir; onu mertebe
    testi yakalar.
    """
    m, k = MU_H2, K_H2
    dt = 2 * np.pi * np.sqrt(m / k) / 200
    durum, alan = _tek_osilator(m, k)
    yorunge = run(durum, alan, dt=dt, n_steps=100_000, save_every=10)
    toplam = yorunge.kinetic + yorunge.potential
    blok = toplam[: (len(toplam) // 20) * 20].reshape(20, -1).mean(axis=1)
    assert abs(blok[-1] - blok[0]) / abs(blok[0]) < 1e-6


def test_enerji_SALINIMI_kapali_forma_uyuyor():
    """tepe-tepe/E0 = pi^2/N^2 -- dosyadaki tek başına EN GÜÇLÜ test.

    ÖLÇÜLDÜ: altı N değerinde ölçülen/tahmin oranı 1.0000 (4 hane).
    HANGİ HATAYI YAKALAR: yedi mutantın HEPSİ (forward Euler 1.08e+07,
    simplektik Euler 1.27e+02, yarım tekme 4.05e+03, a=F 2.86e+06, işaret
    ters 3.96e+221, a=F/m^2 4.05e+03, bayat kuvvet 2.06e+05).
    YAKALAMAZ: kütle-birim hatası (her iki kodda da 1.000000) -- onu yalnızca
    fiziksel çıpa yakalar.
    UYARI: kapalı form HARMONİK OSİLATÖRE ÖZGÜDÜR; gerçek MD koşusuna
    taşınamaz, birim testi olarak kalır.
    """
    m, k = MU_H2, K_H2
    for N in (100, 200, 400):
        dt = 2 * np.pi * np.sqrt(m / k) / N
        durum, alan = _tek_osilator(m, k)
        yorunge = run(durum, alan, dt=dt, n_steps=N * 20, save_every=1)
        toplam = yorunge.kinetic + yorunge.potential
        olculen = (toplam.max() - toplam.min()) / abs(toplam[0])
        assert abs(olculen / (np.pi**2 / N**2) - 1.0) < 0.05


def test_global_hata_ikinci_mertebeden():
    """dt yarılanınca hata oranı ~4 (Modül 1A'da aynı sınav 4.0001 vermişti).

    ÖLÇÜLDÜ (dt = 0.08, 0.04, 0.02, 0.01, 0.005 -- BEŞ dt, DÖRT oran):
    4.001795 -> 4.000449 -> 4.000112 -> 4.000028.
    TARAMA BİLEREK BEŞ NOKTALI: üç dt yalnız İKİ oran üretir ve yukarıdaki
    dört ölçümün ikisi hiç koşulmamış olurdu.
    NORM TÜM YÖRÜNGE ÜZERİNDEN alınır: tek noktada ölçmek cos~0 civarında
    4.98/4.47/4.23 gibi kararsız oranlar verir.
    HANGİ HATAYI YAKALAR: simplektik Euler (2.0026/2.0013) VE forward Euler
    (1.9796/1.9901) -- ölçüldü, ikisi de sınırın dışında.
    """
    m, k = 1.0, 1.0
    t_f, x0 = 2.0, 0.1
    omega = np.sqrt(k / m)
    hatalar = []
    for dt in (0.08, 0.04, 0.02, 0.01, 0.005):
        n = int(round(t_f / dt))
        durum, alan = _tek_osilator(m, k, x0=x0)
        yorunge = run(durum, alan, dt=dt, n_steps=n, save_every=1)
        analitik = x0 * np.cos(omega * yorunge.times)
        hatalar.append(float(np.abs(yorunge.positions[:, 0, 0] - analitik).max()))
    for once, sonra in zip(hatalar, hatalar[1:]):
        assert abs(once / sonra - 4.0) < 4.0 * 0.35


def test_fiziksel_cipa_kutle_birim_hatasini_yakaliyor():
    """MODÜLÜN EN ÖNEMLİ TESTİ -- kütle boru hattının TEK koruması.

    ÖLÇÜLDÜ (tasarım §3): yedi mutasyona karşı altı testten BEŞİ ("1 amu = 1"
    hatasında) BİREBİR AYNI sayıyı veriyor -- periyot, sürüklenme, salınım
    genliği, mertebe, başlangıç-koşulu. Yakalayan tek şey budur: doğru kodda
    4401.4 cm^-1, hatalı kodda 187867.6 cm^-1 (42.68x).

    2PI TUZAĞI: omega atomik birimde AÇISAL frekanstır. Ölçülen periyottan
    1/T alıp aynı katsayıyla çevirmek 700.47 cm^-1 verir -- tam 6.28319x hata.
    Bu yüzden dönüşüm 2*pi/T üzerinden yapılır.
    """
    from qlab.units import HARTREE_TO_WAVENUMBER

    dt = T_H2 / 200
    durum, alan = _tek_osilator(MU_H2, K_H2)
    yorunge = run(durum, alan, dt=dt, n_steps=200 * 10, save_every=1)
    omega = 2 * np.pi / _periyot(yorunge)
    assert omega * HARTREE_TO_WAVENUMBER == pytest.approx(4401.21, abs=5.0, rel=0.0)


def test_baslangic_kosulu_tuzagina_dusmuyor():
    """x0 = 0 ve v0 = 0 hiçbir şey yapmaz ve HER integratör geçer.

    Bu test o kurulumun sınanmadığını kilitler: yörünge gerçekten hareket
    etmeli, yoksa yukarıdaki ölçütlerin hiçbiri bir şey ölçmez.
    """
    durum, alan = _tek_osilator(MU_H2, K_H2, x0=0.1)
    yorunge = run(durum, alan, dt=T_H2 / 200, n_steps=100, save_every=1)
    xs = yorunge.positions[:, 0, 0]
    # ÖLÇÜLDÜ: doğru kodda genlik (max-min) 2.0000e-01 ve min -0.1000; konumu
    # HİÇ güncellemeyen integratörde genlik tam 0.0. Testin eski hâli
    # (|x|max > 0.01) İLK KARE tarafından sağlanıyordu ve o mutantı GÖRMÜYORDU.
    assert float(xs.max() - xs.min()) > 0.5 * 0.1
    assert float(xs.min()) < -0.5 * 0.1
    assert float(np.abs(yorunge.velocities[:, 0, 0]).max()) > 0.0
