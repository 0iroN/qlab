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
from qlab.engine.md.thermostats import Langevin, NoseHoover
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


# ========================================================================
# Task 10 -- Nosé-Hoover ve ergodiklik kusurunun ÖLÇÜLMESİ.
# 
# Ergodiklik kusuru KÜÇÜK sistemin özelliğidir. Ölçülen: 200 bağımsız osilatörü
# tek global zeta'ya bağlayan kurulumda kusur GÖRÜNMEZ (üç bağımsız kurulum,
# Langevin kontrolü +0.017 +- 0.096 %). Bu yüzden burada GERÇEK tek serbestlik
# dereceli osilatör kullanılır: (1,3) durumunun y,z bileşenleri tam sıfır
# tutulur (harmonik kuvvet sıfır, termostat hızları düzgün ölçekler -> ölçülen
# yan sapma 0.0e+00) ve `n_dof=1` verilir.
# ========================================================================

M = 918.576324
K = 0.3693947966
OMEGA = float(np.sqrt(K / M))
PERIYOT = 2 * np.pi / OMEGA

HEDEF_K = 300.0
TAU_ERGODIK = PERIYOT / (2 * np.pi)   # = 49.86688461471631 a.u. = 1/OMEGA
DT_ERGODIK = PERIYOT / 50             # = 6.2664575345201055 a.u.
N_PERIYOT = 40                        # ilk yarısı denge, ikinci yarısı ölçüm
N_BLOK = 4
N_REPLIKA = 384

X2_KANONIK = KB_HARTREE_PER_KELVIN * HEDEF_K / K   # = 2.571891856033476e-3
V2_KANONIK = KB_HARTREE_PER_KELVIN * HEDEF_K / M   # = 1.0342564294490596e-6


def _tek_dof_durum(x: float, v: float) -> MDState:
    """Tek parçacık, yalnız x yönü canlı. y,z tam sıfır kalır (test eder)."""
    konum = np.zeros((1, 3))
    konum[0, 0] = x
    hiz = np.zeros((1, 3))
    hiz[0, 0] = v
    return MDState(positions=konum, velocities=hiz, masses=np.full(1, M))


def _tek_dof_topluluk(zincir: int, tohum: int, n_replika: int = N_REPLIKA) -> dict:
    """n_replika BAĞIMSIZ 1-dof osilatör; her biri KENDİ termostatını taşır.

    Her replika kanonikten ayrı çekilir (sigma_x = 0.0507138231257857 bohr);
    ortak başlangıç noktası veya gevşetilmemiş ofset YOKTUR -- aksi halde
    ölçülen şey ergodiklik değil, ofsetin kalıcı izi olur.

    Döndürülen `pencere_orani` = s(tek blok) / s(tüm pencere); replikalar
    arası saçılımın ortalama süresiyle küçülüp küçülmediğini ölçer.
    Ergodik dinamikte sqrt(N_BLOK) = 2.0 beklenir, ergodik olmayanda 1.0.
    """
    alan = Harmonic(k=K, center=np.zeros((1, 3)))
    n_adim = int(round(N_PERIYOT * PERIYOT / DT_ERGODIK))
    n_denge = n_adim // 2
    blok_uzun = (n_adim - n_denge) // N_BLOK

    uretici = np.random.default_rng(tohum)
    x0 = uretici.normal(size=n_replika) * float(np.sqrt(X2_KANONIK))
    v0 = uretici.normal(size=n_replika) * float(np.sqrt(V2_KANONIK))
    assert len(np.unique(x0)) == n_replika, "replikalar bağımsız olmalı"

    bloklar = np.zeros((n_replika, N_BLOK))
    v_kare = np.zeros(n_replika)
    yan_sapma = 0.0
    for r in range(n_replika):
        durum = _tek_dof_durum(float(x0[r]), float(v0[r]))
        termostat = NoseHoover(
            target_temperature_kelvin=HEDEF_K,
            relaxation_time=TAU_ERGODIK,
            chain_length=zincir,
            n_dof=1,
        )
        _, kuvvetler = alan(durum.positions)
        for _ in range(n_denge):
            durum, _, kuvvetler = termostat.step(durum, kuvvetler, alan, DT_ERGODIK)
        for j in range(N_BLOK):
            toplam = 0.0
            for _ in range(blok_uzun):
                durum, _, kuvvetler = termostat.step(
                    durum, kuvvetler, alan, DT_ERGODIK
                )
                toplam += float(durum.positions[0, 0] ** 2)
                v_kare[r] += float(durum.velocities[0, 0] ** 2)
            bloklar[r, j] = toplam / blok_uzun
        yan_sapma = max(
            yan_sapma,
            float(np.abs(durum.positions[0, 1:]).max()),
            float(np.abs(durum.velocities[0, 1:]).max()),
        )

    assert yan_sapma == 0.0, (
        f"y,z tam sıfır kalmalıydı; ancak o zaman n_dof=1 doğrudur: {yan_sapma}"
    )
    bloklar /= X2_KANONIK
    v_kare /= blok_uzun * N_BLOK * V2_KANONIK
    kisa = float(np.mean([bloklar[:, j].std(ddof=1) for j in range(N_BLOK)]))
    uzun = float(bloklar.mean(axis=1).std(ddof=1))
    return {
        "x2_sapma": float(bloklar.mean()) - 1.0,
        "v2_sapma": float(v_kare.mean()) - 1.0,
        "pencere_orani": kisa / uzun,
        "sacilim": uzun,
    }


def test_nose_hoover_termostat_kutlesi_belgelenmis_formulu_izliyor():
    """Q = k_B T tau^2. Bağımsız hesaplanmış LİTERAL değere karşı.

    k_B = 3.166811563455608e-6 Ha/K, T = 300 K, tau = 1/OMEGA
    = 49.86688461471631 a.u.  ->  Q = 2.3624789668407673.
    Bu bir formül kontrolüdür, fizik ölçümü değildir. `q = 2 k_B T tau^2`
    mutantını artık hızlı şeritte iki FİZİK çıpası da kırar (13.10.2026, mut.py):
    md G7 dosyasının serbest-parçacık periyodu ve bu dosyanın sonundaki NHC-ODE
    çıpası. Ergodiklik testleri onu hâlâ görmez (düz <x^2> -4.6432 %, oran 1.1539).
    """
    termostat = NoseHoover(
        target_temperature_kelvin=HEDEF_K, relaxation_time=TAU_ERGODIK
    )
    assert termostat.thermostat_mass == pytest.approx(2.3624789668407673, rel=1e-12)


def test_nose_hoover_adimi_zaman_tersinir():
    """NH adımı zaman-tersinir olmalı: ileri N adım, (v,zeta) ters, geri N adım.

    ÖLÇÜLDÜ (dt = PERIYOT/500, N = 500, 8 parçacık, tohum 4242):
      önerilen (eşlenik geri yarım-adım): 1.499e-15 / 1.833e-15 / 2.166e-15
        (zincir 1/2/3)  -> eşiğin 6.7e5 katı ALTINDA
      geri yarım-adım ileriyle AYNI sırada olursa (bugünkü hata):
        4.852e-03 / 4.925e-03 / 1.094e-03 -> eşiğin 1.1e6 katı ÜSTÜNDE
    Ayrım 12 mertebe. Bu testin gerekçesi kozmetik değil: aynı sıralama hatası
    <x^2> ölçümünü dt = PERIYOT/16000'e kadar YAKINSAMAZ kılıyor (ölçüldü:
    dt=P/50'de +33.94 %, P/200'de +1.72 %, P/1000'de -6.28 %, yakınsak
    değer -8.31 %).

    YAKALADIĞI (UYGULAMADA ÖLÇÜLDÜ, tasarım tersini söylüyordu): zincir
    döngüsünün YÖN hatası da bu testle kırmızı oluyor. Tasarım belgesi
    "ayırt edilemez (1.67e-15 vs 2.71e-15)" diyordu; ölçülen dönüş hatası
    zincir=3'te İLERİ yarımda 2.973e-04, GERİ yarımda 2.975e-04 -- yani
    doğru koddan (1.499e-15) 11 mertebe uzakta. Fiziksel olarak beklenen
    sonuç budur: eşlenik özelliği alt adımların SIRASINA bağlıdır.
    Ergodiklik testleri ise o mutantı gerçekten ayırt EDEMİYOR (ölçüldü:
    yalnız bu test kırmızı oluyor).
    """
    alan = Harmonic(k=K, center=np.zeros((8, 3)))
    dt = PERIYOT / 500
    n_adim = 500
    for zincir in (1, 2, 3):
        uretici = np.random.default_rng(4242)
        x0 = uretici.normal(size=(8, 3)) * float(np.sqrt(X2_KANONIK))
        v0 = uretici.normal(size=(8, 3)) * float(np.sqrt(V2_KANONIK))
        durum = MDState(positions=x0, velocities=v0, masses=np.full(8, M))
        termostat = NoseHoover(
            target_temperature_kelvin=HEDEF_K,
            relaxation_time=TAU_ERGODIK,
            chain_length=zincir,
        )
        _, kuvvetler = alan(durum.positions)
        for _ in range(n_adim):
            durum, _, kuvvetler = termostat.step(durum, kuvvetler, alan, dt)
        durum = MDState(
            positions=durum.positions,
            velocities=-durum.velocities,
            masses=durum.masses,
        )
        termostat.zeta = -termostat.zeta
        _, kuvvetler = alan(durum.positions)
        for _ in range(n_adim):
            durum, _, kuvvetler = termostat.step(durum, kuvvetler, alan, dt)
        hata = float(np.abs(durum.positions - x0).max() / np.abs(x0).max())
        assert hata < 1e-9, f"zincir={zincir}: dönüş hatası {hata:.3e}"


def test_nose_hoover_kinetik_sicakligi_tutturuyor():
    """400 osilatör, zincirli NH, dt = PERIYOT/400.

    dt seçimi ÖLÇÜMDEN: bu kurulumda (tau=PERIYOT) sapma dt ile
    -3.730 % (P/50), -1.684 % (P/100), -0.315 % (P/200), +0.012 % (P/400)
    şeklinde gidiyor; rel=0.005 eşiği ancak P/400'de anlamlı.
    Planın dt = PERIYOT/50 seçimi bu testte 7 kat marjla çökerdi.
    """
    hedef, n = 300.0, 400
    uretici = np.random.default_rng(11)
    durum = MDState(
        positions=uretici.normal(size=(n, 3)) * float(np.sqrt(X2_KANONIK)),
        velocities=uretici.normal(size=(n, 3)) * float(np.sqrt(V2_KANONIK)),
        masses=np.full(n, M),
    )
    alan = Harmonic(k=K, center=np.zeros((n, 3)))
    termostat = NoseHoover(
        target_temperature_kelvin=hedef, relaxation_time=PERIYOT, chain_length=3
    )
    dt = PERIYOT / 400
    n_adim = 200 * 400
    _, kuvvetler = alan(durum.positions)
    sicakliklar = []
    for adim in range(n_adim):
        durum, _, kuvvetler = termostat.step(durum, kuvvetler, alan, dt)
        if adim > n_adim // 2:
            sicakliklar.append(instantaneous_temperature(durum, n_constraints=0))
    assert float(np.mean(sicakliklar)) == pytest.approx(hedef, rel=0.005, abs=0.0)


@pytest.fixture(scope="module")
def tek_dof_olcum() -> dict:
    """İki koşu (düz ve zincirli), iki test tarafından paylaşılır (~35 s)."""
    return {
        "duz": _tek_dof_topluluk(zincir=1, tohum=20260812),
        "zincirli": _tek_dof_topluluk(zincir=3, tohum=20260812),
    }


@pytest.mark.slow
def test_nose_hoover_ergodiklik_kusuru_TEK_osilatorde_olculuyor(tek_dof_olcum):
    """Düz NH tek osilatörün konum dağılımını kaçırır; zincir düzeltir.

    ÖLÇÜLEN (bu konfigürasyon, 40 tohum, R = 384):
      düz      <x^2> sapma: -8.1796 %  (tohum-std 1.1541, aralık -10.634..-5.420)
      zincirli <x^2> sapma: +0.1461 %  (tohum-std 0.6810, aralık  -1.526..+1.386)
      düz      <v^2> sapma: +0.0188 %  (|maks| 0.1133)
      zincirli <v^2> sapma: +0.3317 %  (|maks| 1.5750)
    Bu tohumda (20260812): düz -7.2723 %, zincirli -0.0729 %.

    EŞİKLER ve gürültü tabanına oranı:
      düz  < -4 %   : gürültü tabanı (tohum-std) 1.1541 puan -> eşik ortalamadan
                      3.62 sigma; 40 tohumun EN KÖTÜSÜ -5.420 % (1.36x marj)
      |zin| <  3 %  : gürültü tabanı 0.6810 puan -> 4.41 sigma; en kötü
                      |1.526| % (1.97x marj)
      |düz <v^2>| < 1 % : en kötü 0.1133 % (8.83x marj) -- kusurun KONUMDA
                      olduğunu, kinetik sıcaklıkta olmadığını ölçer
      |zin <v^2>| < 3 % : en kötü 1.5750 % (1.90x marj)

    KIRDIĞI MUTANTLAR (R=384, tohum 20260812, ölçülmüş):
      geri yarım-adım ileriyle aynı sırada : düz +44.0619 % (İŞARET ters) RED
      n_dof yok sayılıp v.size kullanılır  : düz +183.7024 %, <v^2> +199.9034 %,
                                             zin +55.7381 %, zin <v^2> +165.4744 % RED
      zincir yarım-adımında 0.5*dt yerine dt: düz +5.8142 % RED
      hedef ndof*kT yerine ndof*kT/2       : <v^2> -50.0231 %, zin -35.3340 % RED
      q = k_B T tau (kare unutulmuş)       : zin +67225.89 %, zin <v^2> +66909.48 % RED

    YAKALAMADIĞI (dürüst olumsuz, ölçülmüş; 13.10.2026'da güncellendi):
      q = 2 k_B T tau^2 : düz -4.6432 % -> -4 % eşiğini GEÇEMİYOR, geçer. Bu test
        görmez; hızlı şeritte literal test, G7 periyodu ve NHC-ODE çıpası kırar.
      zincir döngüsünün YÖNÜ (iki yarımda tutarlı): zincir=1'de ölü kod;
        zincir=3'te 20 tohumda fark -0.3014 +- 1.1234 puan, tohum gürültüsü
        0.8617 puan -> ayırt EDİLEMİYOR. KUSUR DA DEĞİL: eşdeğer ikinci mertebe
        ayrıştırma (ölçüm: dosyanın sonundaki NHC-ODE çıpasının dokstringi).
    """
    duz, zincirli = tek_dof_olcum["duz"], tek_dof_olcum["zincirli"]
    assert duz["x2_sapma"] < -0.04, (
        f"düz NH'nin <x^2> kusuru görünmedi: %{duz['x2_sapma'] * 100:.3f}"
    )
    assert abs(zincirli["x2_sapma"]) < 0.03, (
        f"zincir kusuru gidermeliydi: %{zincirli['x2_sapma'] * 100:.3f}"
    )
    assert abs(duz["v2_sapma"]) < 0.01, (
        f"kusur KONUMDA olmalı, kinetik sıcaklıkta değil: "
        f"%{duz['v2_sapma'] * 100:.3f}"
    )
    assert abs(zincirli["v2_sapma"]) < 0.03, (
        f"zincirli kinetik sıcaklık: %{zincirli['v2_sapma'] * 100:.3f}"
    )


@pytest.mark.slow
def test_nose_hoover_ergodiklik_imzasi_ZAMAN_ORTALAMASI_ile_kuculmuyor(
    tek_dof_olcum,
):
    """Ergodikliğin ASIL imzası: saçılım ortalama süresiyle küçülüyor mu?

    Ergodik dinamikte yörünge-başına zaman ortalamalarının saçılımı 1/sqrt(T)
    ile küçülür; ergodik olmayanda her yörünge kendi torusunda kilitli kalır ve
    saçılım küçülmez. ÖLÇÜLDÜ (tau=P/(2pi), dt=P/50, R=1024, palindrom):
      düz      : 21.203 % (8P pencere) -> 21.326 % (40P) -> 21.170 % (160P)
                 20 kat pencerede HİÇ küçülmüyor
      zincirli : 18.941 % -> 8.824 % -> 4.186 %  (4.52x; ideal sqrt(20)=4.472)

    Bu testte kullanılan oran = s(tek blok)/s(tüm pencere), N_BLOK=4
    (ergodik beklenti sqrt(4) = 2.0).
    ÖLÇÜLEN (40 tohum, R=384): düz 1.0363 +- 0.0075 (aralık 1.0210..1.0527),
    zincirli 1.9995 +- 0.0598 (aralık 1.8821..2.1428).
    Bu tohumda: düz 1.0510, zincirli 2.0273.

    EŞİKLER ve gürültü tabanına oranı:
      düz  < 1.35 : gürültü tabanı 0.0075 -> eşik ortalamadan 41.8 sigma;
                    40 tohumun en kötüsü 1.0527 (1.28x marj)
      zin  > 1.55 : gürültü tabanı 0.0598 -> 7.52 sigma; en kötü 1.8821 (1.21x)

    KIRDIĞI MUTANT: q = k_B T tau (düz 1.9300 -> 1.35 eşiğini aşar; zincirli
    1.0000 -> 1.55 eşiğinin altında kalır) -- iki iddiayı birden kırar.
    YAKALAMADIĞI: geri yarım-adım sırası (düz 1.1358, zin 2.0388 -- ikisi de
    geçer; onu zaman-tersinirlik testi yakalar), n_dof (düz 1.0730, zin 6.9377),
    q=2kTtau^2 (1.1539 / 2.0344), 0.5*dt->dt (1.1182 / 1.7440),
    ndof*kT/2 (1.1283 / 1.8207), zincir yönü (1.0510 / 1.9811).
    """
    duz, zincirli = tek_dof_olcum["duz"], tek_dof_olcum["zincirli"]
    assert duz["pencere_orani"] < 1.35, (
        f"düz NH'de saçılım küçülmemeliydi (oran {duz['pencere_orani']:.4f}, "
        f"saçılım %{duz['sacilim'] * 100:.2f})"
    )
    assert zincirli["pencere_orani"] > 1.55, (
        f"zincirli NH'de saçılım 1/sqrt(T) ile küçülmeliydi "
        f"(oran {zincirli['pencere_orani']:.4f})"
    )


def test_nose_hoover_gecersiz_parametreleri_reddediyor():
    with pytest.raises(InvalidInputError, match="zincir"):
        NoseHoover(
            target_temperature_kelvin=300.0, relaxation_time=1.0, chain_length=0
        )
    with pytest.raises(InvalidInputError, match="serbestlik"):
        NoseHoover(target_temperature_kelvin=300.0, relaxation_time=1.0, n_dof=0)
    with pytest.raises(InvalidInputError, match="gevşeme"):
        NoseHoover(target_temperature_kelvin=300.0, relaxation_time=0.0)
    with pytest.raises(InvalidInputError, match="sıcaklık"):
        NoseHoover(target_temperature_kelvin=0.0, relaxation_time=1.0)


def test_nose_hoover_ZINCIRI_BAGIMSIZ_ODE_cozumune_IKINCI_MERTEBEDEN_yakinsiyor():
    """C4 (13.10.2026): NH ZİNCİRİNİN dinamiği, bağımsız ODE çözümüne karşı.

    SINIF: ÇIPA (bağımsız yöntem), regresyon kilidi DEĞİL. Referans ders
    kitabı NHC denklemleridir (Martyna-Klein-Tuckerman 1992); testin İÇİNDE
    yazılır ve scipy DOP853 ile çözülür. `NoseHoover`a da `thermostat_mass`a
    da bakmaz; Q_j = k_B T tau^2 belgelenmiş formülden gelir:
        dx/dt = v                        dv/dt = -k x/m - z0 v
        dz0/dt = (m v^2 - n k_B T)/Q - z0 z1
        dz1/dt = (Q z0^2 - k_B T)/Q - z1 z2      dz2/dt = (Q z1^2 - k_B T)/Q
    PENCERE: tek 1-dof osilatör (y, z tam sıfır, n_dof = 1), zincir = 3,
    tau = 1/omega, x0 = 1.3 sigma_x, v0 = -0.7 sigma_v, zeta0 = (0.4, -0.3,
    0.2)/tau (sıfırdan farklı başlangıç, zinciri ilk adımdan bağlar), 2
    periyot, dt = P/100 ve P/200. Hata = adım sonlarında en büyük
    |x - x_ref|/sigma_x ve en büyük |zeta - zeta_ref|*tau.

    NEDEN VAR: zincir terimlerini (z1, z2 denklemleri) sınayan tek test
    yavaş şeridin ergodiklik testiydi. ÖLÇÜLDÜ (13.10.2026, mut.py, bu test
    yokken): aşağıdaki ilk iki mutant md hızlı şeridinin 349 testini de
    GEÇİYORDU; onları yalnız yavaş şeridin ergodiklik testleri kırıyordu
    (-x ile ilk düşen `..._TEK_osilatorde_olculuyor`, fixture ~30 s).

    ÖLÇÜLDÜ (bu pencere)           P/100        P/200       oran
      x hatası                   1.945e-02    4.904e-03    3.965
      zeta hatası                1.004e-01    2.555e-02    3.931
    Oran dt küçüldükçe 4'e gider (x: P/50 -> P/400 arası 3.86 / 3.97 / 3.99),
    yani İKİNCİ MERTEBE. Referansın kendi hatası (rtol 1e-12 ile 1e-13
    çözümlerinin farkı) x'te 2.2e-11, zeta'da 1.5e-10; ölçülenin 1e7 altı.
    EŞİKLER: P/200 hatası x < 0.05, zeta < 0.25 (ölçülenin ~10 katı: meşru
    bir yeniden ayrıştırmaya pay) VE iki oran da 3.0 .. 5.0 arası (tutarlı
    ikinci mertebe). Sayılar kesme hatasıdır, son ulp'e bağlı değildir.

    KIRDIĞI MUTANTLAR (P/200 hatası; oran ~1.00, yani hata dt ile KÜÇÜLMÜYOR,
    kod başka bir ODE çözüyor):
      zincir kuvvetinde -k_B T yok (iki yarımda)     x 2.505   zeta 3.263
      zincir bağlaşımı exp(-h/2 ...) (iki yarımda)   x 1.624   zeta 3.579
      yarım adımlarda q = 2 k_B T tau^2              x 2.289   zeta 3.395
    İlk ikisi "denge doğru, dinamik yanlış" ailesindendir; üçüncüyü hızlı
    şeritte G7 serbest-parçacık periyodu da kırıyor.

    KIRMADIĞI (dürüst olumsuz; KUSUR DEĞİL): zincir döngüsünün YÖNÜ iki
    yarımda TUTARLI ters çevrilirse (ileri yarımda i = M-1 .. 1, geri yarımda
    1 .. M-1) adım yine aynı alt-akışların simetrik bir Trotter bileşimidir:
    zaman-tersinir kalır ve AYNI ODE'ye ikinci mertebeden yakınsar (x
    2.654e-02 -> 6.719e-03, oran 3.951; zeta 1.662e-01 -> 4.237e-02, oran
    3.923). Doğru koddan farkı yalnız kesme sabitidir (x 1.37, zeta 1.66
    kat). Onu öldürmenin tek yolu bugünkü alt-adım SIRASINI dondurmaktır
    (regresyon kilidi); bu, MTK sırası gibi meşru bir değişikliği de
    kırmızıya çevireceği için EKLENMEDİ. Bu mutant md hızlı şeridini (349)
    ve iki yavaş termostat testini de geçer (ölçüldü, mut.py).
    """
    # Yerel import BİLİNÇLİ: dosyanın başına eklenen bir satır, bu dosyanın
    # depoda satır numarasıyla atıf yapılan dokstring bloklarını kaydırırdı.
    from scipy.integrate import solve_ivp

    kt = KB_HARTREE_PER_KELVIN * HEDEF_K
    tau = TAU_ERGODIK
    q = kt * tau**2
    sigma_x, sigma_v = float(np.sqrt(X2_KANONIK)), float(np.sqrt(V2_KANONIK))
    x0, v0 = 1.3 * sigma_x, -0.7 * sigma_v
    zeta0 = np.array([0.4, -0.3, 0.2]) / tau

    def nhc(_t, y):
        x, v, z0, z1, z2 = y
        return [
            v,
            -K * x / M - z0 * v,
            (M * v * v - kt) / q - z0 * z1,
            (q * z0 * z0 - kt) / q - z1 * z2,
            (q * z1 * z1 - kt) / q,
        ]

    olcek = np.array([sigma_x, sigma_v, 1.0 / tau, 1.0 / tau, 1.0 / tau])
    referans = solve_ivp(
        nhc, (0.0, 2.0 * PERIYOT), [x0, v0, *zeta0],
        method="DOP853", rtol=1e-12, atol=1e-12 * olcek, dense_output=True,
    )
    assert referans.success, referans.message

    alan = Harmonic(k=K, center=np.zeros((1, 3)))
    hatalar = {}
    for bolme in (100, 200):
        dt = PERIYOT / bolme
        durum = _tek_dof_durum(x0, v0)
        termostat = NoseHoover(
            target_temperature_kelvin=HEDEF_K,
            relaxation_time=tau,
            chain_length=3,
            n_dof=1,
        )
        termostat.zeta = zeta0.copy()
        _, kuvvetler = alan(durum.positions)
        n_adim = 2 * bolme
        x, zeta = np.empty(n_adim), np.empty((n_adim, 3))
        for n in range(n_adim):
            durum, _, kuvvetler = termostat.step(durum, kuvvetler, alan, dt)
            x[n], zeta[n] = durum.positions[0, 0], termostat.zeta
        ref = referans.sol(dt * np.arange(1, n_adim + 1))
        hatalar[bolme] = (
            float(np.abs(x - ref[0]).max()) / sigma_x,
            float(np.abs(zeta - ref[2:].T).max()) * tau,
        )

    (x_kaba, z_kaba), (x_ince, z_ince) = hatalar[100], hatalar[200]
    assert x_ince < 0.05 and z_ince < 0.25, (
        f"NHC yörüngesi bağımsız ODE'den uzak (P/200): x {x_ince:.3e}, zeta {z_ince:.3e}"
    )
    assert 3.0 < x_kaba / x_ince < 5.0, f"x yakınsama oranı {x_kaba / x_ince:.3f} (beklenen 4)"
    assert 3.0 < z_kaba / z_ince < 5.0, f"zeta yakınsama oranı {z_kaba / z_ince:.3f} (beklenen 4)"
