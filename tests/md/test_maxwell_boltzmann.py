"""Maxwell-Boltzmann dağılımının kendiliğinden çıkması.

BU DOSYANIN İDDİA ETMEDİĞİ ŞEY -- ve bu, tasarımın en önemli ölçülmüş
uyarısıdır: MB'nin çıkması KUVVET ALANINI DOĞRULAMAZ. Ölçüldü (tasarım
§4.4): eps x10 -> R4 = 1.66326; eps x0.1 -> 1.64654; sigma x0.9 -> 1.66506;
WCA (yalnız itici) -> 1.65274. DÖRDÜ DE kusursuz MB veriyor. Yalnızca ideal
gaz düşüyor. Kuvvet alanının doğrulaması analitik çıpalarladır
(tests/md/test_forcefields.py).

MB burada bir İNTEGRATÖR + TERMOSTAT sınavıdır.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import LennardJones
from qlab.engine.md.state import MDState
from qlab.engine.md.thermostats import Langevin
from qlab.observable.md_obs import (
    SPEED_SHAPE_RATIO_MB,
    speed_shape_ratio,
    speed_shape_ratio_noise,
)
from qlab.units import KB_HARTREE_PER_KELVIN

AR_EPS = 3.7938402530e-04
AR_SIGMA = 6.4345174543
AR_MASS = 72820.749
#: tau_LJ = sigma*sqrt(m/eps) = 89146.4462 atomik zaman birimi = 2156.35 fs.
TAU = 89146.4462

#: N = 108: EN KÜÇÜK GEÇERLİ DEĞER, geometrik olarak. rho* = 0.8442 ve
#: r_c = 2.5 sigma için L > 2 r_c şartı N >= 105.5 gerektiriyor; en küçük FCC
#: değeri 108. N = 32 bu durum noktasında İMKÂNSIZDIR (istatistik meselesi değil).
N = 108


def _fcc_kutu(n: int, yogunluk_yildiz: float) -> tuple[np.ndarray, float]:
    hucre = round((n / 4) ** (1 / 3))
    assert 4 * hucre**3 == n, f"{n} bir FCC sayısı değil"
    kenar = (n / yogunluk_yildiz) ** (1 / 3) * AR_SIGMA
    a = kenar / hucre
    taban = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    konumlar = np.concatenate(
        [
            taban + np.array([i, j, k]) * a
            for i in range(hucre)
            for j in range(hucre)
            for k in range(hucre)
        ]
    )
    return konumlar, kenar


def _mb_kosusu(alan, tohum: int, gorunt_sayisi: int) -> np.ndarray:
    konumlar, _ = _fcc_kutu(N, 0.8442)
    durum = MDState(
        positions=konumlar, velocities=np.zeros((N, 3)), masses=np.full(N, AR_MASS)
    ).remove_center_of_mass_motion()
    termostat = Langevin(
        target_temperature_kelvin=87.2, friction=1.0 / (0.5 * TAU), seed=tohum
    )
    dt = 0.005 * TAU
    _, kuvvetler = alan(durum.positions)
    for _ in range(3000):  # denge: 6/(2 gamma) ölçüsünün üstünde bir pay
        durum, _, kuvvetler = termostat.step(durum, kuvvetler, alan, dt)
    ornekler = []
    for adim in range(gorunt_sayisi * 50):
        durum, _, kuvvetler = termostat.step(durum, kuvvetler, alan, dt)
        if adim % 50 == 0:  # ACF(v^2) ancak 46 adımda 0.05'in altına iniyor
            ornekler.append(durum.velocities.copy())
    return np.concatenate(ornekler)


@pytest.mark.slow
def test_maxwell_boltzmann_kendiliginden_cikiyor():
    """Soğuk kristalden başla, termostat tak, hız dağılımının ŞEKLİNİ ölç.

    ÖLÇÜLDÜ, N=256'da: <R4> = 1.66654 ± 0.00331 (32 tohum). DİKKAT -- BU TEST
    N=108 KOŞUYOR ve orada saçılım çok daha geniş: <R4> = 1.66159 ± 0.02089
    (16 tohum), 5/3'ten sapma -0.00508 (sem 0.00522), 16 tohumun 0'ı sınırı
    aşıyor. ±0.00331'i "bu testin saçılımı" sanıp ondan tolerans türetmek
    DOĞRU KODU kırmızı yapar.

    ÖRNEKLER >= 50 ADIM ARAYLA: ardışık adımlardan toplamak n'i SAHTE şişirir
    ve toleransı olduğundan sıkı sandırır.

    R4 TEK BAŞINA YETMEZ -- aşağıdaki sıcaklık çıpası bu yüzden var. R4
    tanım gereği ÖLÇEKTEN BAĞIMSIZDIR (Task 8 dokstringi bunu söylüyor) ve
    termostatın belirlediği tek şey ölçektir. ÖLÇÜLDÜ: termostat TAMAMEN
    kaldırılıp yalın velocity-Verlet koşulduğunda R4 = 1.711268 ile test
    GEÇİYORDU, oysa <T> = 6.24e-26 K -- hedeften 27 mertebe uzakta.
    """
    _, kenar = _fcc_kutu(N, 0.8442)
    alan = LennardJones(
        epsilon=AR_EPS, sigma=AR_SIGMA, box_length=kenar, cutoff=2.5 * AR_SIGMA
    )
    hizlar = _mb_kosusu(alan, tohum=2026, gorunt_sayisi=20)
    olculen = speed_shape_ratio(hizlar)
    sinir = 4.0 * speed_shape_ratio_noise(len(hizlar))
    # ÖLÇEK ÇIPASI -- R4'ün göremediği şeyi görür.
    # ÖLÇÜLDÜ (16 tohum, N=108): <T> = 87.295 ± 1.763 K, yani ±%10 ≈ 4.9 sigma.
    # Termostat kaldırılınca 6.24e-26 K ve bu assert KIRMIZI olur.
    t_olculen = AR_MASS * (hizlar**2).sum(axis=1).mean() / (
        3.0 * KB_HARTREE_PER_KELVIN
    )
    assert t_olculen == pytest.approx(87.2, rel=0.10, abs=0.0), (
        f"kinetik sıcaklık {t_olculen:.3f} K, hedef 87.2 K"
    )
    assert abs(olculen - SPEED_SHAPE_RATIO_MB) < sinir, (
        f"R4 = {olculen:.6f}, hedef {SPEED_SHAPE_RATIO_MB:.6f}, sınır {sinir:.6f}"
    )


@pytest.mark.slow
def test_MB_kuvvet_alanini_DOGRULAMADIGINI_kilitliyor():
    """Bu testin işi bir şeyi kanıtlamak DEĞİL, bir YANILGIYI ÖNLEMEK.

    ÖLÇÜLDÜ: epsilon 10 kat değiştirilse bile R4 hâlâ 5/3'e oturur. Yani
    "MB çıktı, demek ki potansiyel doğru" çıkarımı YANLIŞTIR. Bu test o
    çıkarımı yapacak bir gelecek değişikliğe karşı yazılıdır: epsilon BOZUK
    olduğu hâlde MB testi geçer.
    """
    _, kenar = _fcc_kutu(N, 0.8442)
    bozuk = LennardJones(
        epsilon=AR_EPS * 10.0, sigma=AR_SIGMA, box_length=kenar, cutoff=2.5 * AR_SIGMA
    )
    hizlar = _mb_kosusu(bozuk, tohum=99, gorunt_sayisi=10)
    sinir = 4.0 * speed_shape_ratio_noise(len(hizlar))
    assert abs(speed_shape_ratio(hizlar) - SPEED_SHAPE_RATIO_MB) < sinir
