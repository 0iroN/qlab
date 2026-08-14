"""Termostatlar: Langevin (BAOAB) ve Nosé-Hoover.

ÖLÇÜLMÜŞ UYARI -- TERMOSTATI <x^2> İLE DOĞRULAMAYIN. BAOAB'ın konfigürasyonel
örneklemesi dt'ye neredeyse duyarsızdır: dt*omega = 1.57'de KİNETİK SICAKLIK
%61.7 yanlışken <x^2> %0.109 doğru çıkıyor (tasarım §4.5). Ayırt eden
gözlenebilir KİNETİK SICAKLIKTIR.

ÖLÇÜLMÜŞ UYARI -- TEK OSİLATÖRLE SICAKLIK TESTİ YAZILAMAZ. R = 1 ile
100 000 adım sonra tohum saçılımı ±%1.053, 1 000 000 adımda hâlâ ±%0.400.
Testler bu yüzden REPLİKA topluluğu kullanır.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import Harmonic
from qlab.engine.md.integrator import velocity_verlet_step
from qlab.engine.md.state import MDState
from qlab.engine.md.thermostats import Langevin
from qlab.errors import InvalidInputError
from qlab.observable.md_obs import instantaneous_temperature
from qlab.units import KB_HARTREE_PER_KELVIN

M = 918.576324
K = 0.3693947966
OMEGA = float(np.sqrt(K / M))
PERIYOT = 2 * np.pi / OMEGA


def _replikalar(n: int):
    """n bağımsız osilatör; her biri kendi merkezinde, hepsi soğuk başlar."""
    merkezler = np.zeros((n, 3))
    durum = MDState(
        positions=merkezler + np.array([0.05, 0.0, 0.0]),
        velocities=np.zeros((n, 3)),
        masses=np.full(n, M),
    )
    return durum, Harmonic(k=K, center=merkezler)


def _kos(termostat, durum, alan, dt, n_adim) -> float:
    _, kuvvetler = alan(durum.positions)
    sicakliklar = []
    for adim in range(n_adim):
        durum, _, kuvvetler = termostat.step(durum, kuvvetler, alan, dt)
        if adim > n_adim // 2:
            sicakliklar.append(instantaneous_temperature(durum, n_constraints=0))
    return float(np.mean(sicakliklar))


def test_langevin_hedef_sicakliga_oturuyor():
    """ÖLÇÜLDÜ, BU TESTİN KENDİ KONFİGÜRASYONUNDA (400 replika, dt = T/50,
    n_adim = 423, seed = 7): <T> = 297.785 K, yani -%0.738. Marj 2.71x.

    Eskiden burada "299.24 K (-%0.25), marj 8x" yazıyordu; o sayı bu
    kurulumda üretilemedi. Sistematik taban BAOAB'ın O(dt^2) kinetik
    yanlılığıdır, istatistiksel gürültü değil.

    SINIR ±%2 VE BU SINIR DAR: ölçülen tohum saçılımı %0.786 (200 tohum,
    ortalama 298.8724 K), yani ±%2 = 2.54 sigma. ÖLÇÜLDÜ -- 200 tohumun
    ÜÇÜ (seed 5, 12, 41) bu sınırı aşıyor, en büyük sapma %2.368. Test
    seed=7'nin bandın içinde olmasına dayanıyor; sınır daraltılacaksa önce
    replika sayısı artmalı (R=2000'de taban %0.351'e, R=400'de %0.716).

    DENGE SÜRESİ: teorik 1/(2*gamma) YETMEZ -- ölçüldü, gerçek oturma onu
    4.47-5.78x aşıyor. Bu yüzden 6/(2*gamma) kadar koşulur (bu koşuda 23
    adım, toplamın yalnız %5.4'ü) ve ölçüm ikinci yarıdan alınır.
    """
    hedef, gamma = 300.0, OMEGA
    durum, alan = _replikalar(400)
    termostat = Langevin(target_temperature_kelvin=hedef, friction=gamma, seed=7)
    dt = PERIYOT / 50
    n_adim = int(6.0 / (2 * gamma) / dt) + 400
    assert _kos(termostat, durum, alan, dt, n_adim) == pytest.approx(
        hedef, rel=0.02, abs=0.0
    )


def test_langevin_baslangic_sicakligindan_BAGIMSIZ_ayni_yere_gidiyor():
    """Sıcak ve soğuk başlangıç aynı dengeye oturmalı.

    Tek bir başlangıçla yapılan test, sistemin zaten orada olmasıyla
    açıklanabilirdi; bu test onu dışlar.
    """
    hedef, gamma = 300.0, OMEGA
    dt = PERIYOT / 50
    n_adim = int(6.0 / (2 * gamma) / dt) + 400
    sonuclar = []
    for olcek in (0.0, 5.0):
        durum, alan = _replikalar(400)
        uretici = np.random.default_rng(11)
        durum = MDState(
            positions=durum.positions,
            velocities=uretici.normal(size=(400, 3))
            * olcek
            * float(np.sqrt(KB_HARTREE_PER_KELVIN * hedef / M)),
            masses=durum.masses,
        )
        termostat = Langevin(target_temperature_kelvin=hedef, friction=gamma, seed=3)
        sonuclar.append(_kos(termostat, durum, alan, dt, n_adim))
    assert sonuclar[0] == pytest.approx(sonuclar[1], rel=0.03, abs=0.0)
    # HEDEFE DE BAĞLA: yukarıdaki assert yalnızca iki koşuyu BİRBİRİNE
    # bağlıyor, yani mutantın kendi çıktısına karşı yazılmış. ÖLÇÜLDÜ --
    # kT'yi 2x, kT'yi 1/2x ve sigma'daki karekökü düşüren üç mutant testi 1
    # kırmızı yapıyor ama bu testi GEÇİYORDU.
    for sonuc in sonuclar:
        assert sonuc == pytest.approx(hedef, rel=0.02, abs=0.0)


def test_langevin_tohumu_tekrarlanabilir():
    """Aynı tohum aynı yörüngeyi vermeli -- test yazılabilmesi için şart.

    UYARI (tasarım §4.5): numpy `Generator` akışı NEP 19 uyarınca DONDURULMUŞ
    DEĞİLDİR. Bu bir HİJYEN testidir ("global RNG sızıntısı yok"), bir fizik
    iddiası değil; fizik iddiaları ölçülmüş istatistiksel banda bağlanır.
    """
    durum, alan = _replikalar(8)
    dt = PERIYOT / 50
    sonuclar = []
    for _ in range(2):
        t = Langevin(target_temperature_kelvin=300.0, friction=OMEGA, seed=42)
        d = durum
        _, kuvvetler = alan(d.positions)
        for _ in range(20):
            d, _, kuvvetler = t.step(d, kuvvetler, alan, dt)
        sonuclar.append(d.positions.copy())
    np.testing.assert_array_equal(sonuclar[0], sonuclar[1])


def test_langevin_surtunmesiz_limitte_velocity_verlet_ile_OZDESTIR():
    """BAOAB'ın YAPISINI kilitler; istatistik bunu yakalayamıyor.

    ÖLÇÜLDÜ: B/A/O/A/B alt adımlarının HERHANGİ BİRİNDE `0.5*dt` yerine `dt`
    yazan dört mutantın DÖRDÜ de bu dosyadaki diğer testleri geçiyordu --
    çünkü hepsi istatistiksel ve gürültü bandı hatayı yutuyor.

    gamma -> 0 limitinde BAOAB tam olarak velocity-Verlet'e indirgenmelidir.
    ÖLÇÜLDÜ (gamma = 1e-14, 40 adım, max|konum farkı|): doğru kod
    1.0808e-07 bohr; dört mutant sırasıyla 3.973e-02 / 3.925e-02 /
    3.925e-02 / 3.942e-02 -- yani 3.6e5 kat ayrım. SINIR 1e-5 bohr LİTERAL.
    """
    durum, alan = _replikalar(8)
    dt = PERIYOT / 50
    termostat = Langevin(target_temperature_kelvin=300.0, friction=1e-14, seed=1)

    d_t, d_v = durum, durum
    _, k_t = alan(d_t.positions)
    _, k_v = alan(d_v.positions)
    for _ in range(40):
        d_t, _, k_t = termostat.step(d_t, k_t, alan, dt)
        d_v, _, k_v = velocity_verlet_step(d_v, k_v, alan, dt)
    np.testing.assert_allclose(d_t.positions, d_v.positions, rtol=0.0, atol=1e-5)


def test_langevin_SURTUNME_KATSAYISI_gercekten_gamma_kadar():
    """Denge dağılımı doğru olduğu hâlde DİNAMİK yanlış olabilir.

    UYGULAMADA MUTASYONLA BULUNDU: `c1 = exp(-gamma*dt)` yerine
    `exp(-gamma*dt/2)` yazan mutant bu dosyadaki BEŞ testi de geçiyordu.
    Sebep fiziksel: O adımı hangi etkin sürtünmeyle yazılırsa yazılsın
    Ornstein-Uhlenbeck adımı DURAĞAN dağılımı doğru üretir -- yani <T>
    hedefte kalır. Bozulan şey gevşeme HIZIDIR: kullanıcı gamma ister,
    gamma/2 alır. Ne sıcaklık testi ne sürtünmesiz-limit testi (gamma=1e-14,
    yarısı da ~0) bunu görebiliyor.

    Kuvvetsiz (serbest parçacık) limitte ortalama hız tam olarak
    <v_n> = v_0 * exp(-gamma * n * dt) ile söner; gürültü topluluk
    ortalamasında düşer.

    ÖLÇÜLDÜ (R=4000, gamma=0.02, dt=1.0, n=40, seed=1):
      ölçülen        0.447994
      exp(-g n dt)   0.449329   <- doğru
      exp(-g n dt/2) 0.670320   <- c1-yarım mutantı
    SINIR 0.02 LİTERAL: doğru koddan sapma 0.0013 (15x marj), mutant 0.22 uzakta.
    """
    class _KuvvetYok:
        def __call__(self, positions):
            return 0.0, np.zeros_like(positions)

    n_replika, gamma, dt, n_adim = 4000, 0.02, 1.0, 40
    v0 = 0.01
    durum = MDState(
        positions=np.zeros((n_replika, 3)),
        velocities=np.ones((n_replika, 3)) * v0,
        masses=np.full(n_replika, M),
    )
    alan = _KuvvetYok()
    termostat = Langevin(target_temperature_kelvin=300.0, friction=gamma, seed=1)
    _, kuvvetler = alan(durum.positions)
    for _ in range(n_adim):
        durum, _, kuvvetler = termostat.step(durum, kuvvetler, alan, dt)

    olculen = float(durum.velocities[:, 0].mean()) / v0
    assert olculen == pytest.approx(
        float(np.exp(-gamma * n_adim * dt)), abs=0.02, rel=0.0
    )


def test_langevin_gecersiz_parametreleri_reddediyor():
    with pytest.raises(InvalidInputError, match="sıcaklık"):
        Langevin(target_temperature_kelvin=0.0, friction=1.0, seed=1)
    with pytest.raises(InvalidInputError, match="sürtünme"):
        Langevin(target_temperature_kelvin=300.0, friction=-1.0, seed=1)
