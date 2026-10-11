"""Kuvvet alanlarının ANALİTİK ÇIPALARI.

Doğrulama fikri: her potansiyelin analitik olarak bilinen sabit noktaları
vardır (sıfır geçişi, minimum konumu, kuyu derinliği) ve kuvvet potansiyelin
türevi olmak ZORUNDADIR. Bu iki sınav birlikte epsilon'u, sigma'yı ve üs
yapısını tek tek kilitler.

NEDEN MAXWELL-BOLTZMANN DEĞİL: ölçüldü, eps x10, eps x0.1, sigma x0.9 ve WCA
(yalnız itici) KUSURSUZ MB veriyor. Dahası -- KUVVET ALANI TAMAMEN
KALDIRILSA (ideal gaz) BİLE R4 = 1.627-1.701 aralığında kalıyor ve test
5/5 tohumda GEÇİYOR; Langevin serbest parçacıklarda tam Maxwell üretir.
MB bir integratör / termostat sınavıdır, kuvvet alanı sınavı DEĞİLDİR ve
hiçbir kuvvet alanını elemez.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import Harmonic, LennardJones, Morse
from qlab.errors import InvalidInputError
from qlab.units import HARTREE_TO_WAVENUMBER

from .h2_morse_sabitleri import H2_A, H2_DE, H2_MU, H2_RE


def _sayisal_kuvvet(alan, konumlar: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """-dV/dx, merkezi fark. Kuvvet/potansiyel tutarsızlığı klasik bir hatadır."""
    kuvvet = np.zeros_like(konumlar)
    for i in range(konumlar.shape[0]):
        for k in range(3):
            arti = konumlar.copy(); arti[i, k] += h
            eksi = konumlar.copy(); eksi[i, k] -= h
            kuvvet[i, k] = -(alan(arti)[0] - alan(eksi)[0]) / (2 * h)
    return kuvvet


def test_harmonik_merkezde_enerji_ve_kuvvet_sifir():
    alan = Harmonic(k=0.5, center=np.array([[1.0, 2.0, 3.0]]))
    enerji, kuvvet = alan(np.array([[1.0, 2.0, 3.0]]))
    assert enerji == pytest.approx(0.0, abs=1e-15, rel=0.0)
    np.testing.assert_allclose(kuvvet, np.zeros((1, 3)), rtol=0.0, atol=1e-15)


def test_harmonik_enerjisi_yarim_k_x_kare():
    """Sayı LİTERAL: 0.5 * 0.5 * (3^2 + 4^2) = 6.25."""
    alan = Harmonic(k=0.5, center=np.zeros((1, 3)))
    enerji, kuvvet = alan(np.array([[3.0, 4.0, 0.0]]))
    assert enerji == pytest.approx(6.25, abs=1e-12, rel=0.0)
    np.testing.assert_allclose(
        kuvvet, np.array([[-1.5, -2.0, 0.0]]), rtol=0.0, atol=1e-12
    )


def test_harmonik_farkli_merkezler_parcacik_basina():
    """N=2 ve GERÇEKTEN farklı merkezler. Enerji EKSTANSİFTİR.

    ÖLÇÜLDÜ: doğru kod E = 1.0; yalnız c_0 kullanan mutant (yani `positions -
    center[0]`) E = 18.5 veriyor; enerjiyi ve kuvveti birlikte 1/N ölçekleyen
    mutant E = 0.5 veriyor. Bu test onlardan öncesinde YOKTU ve iki mutant da
    beş testi de geçiyordu -- çünkü diğer testlerin hepsi tek parçacıklı, ve
    1/N ölçekleme F = -dV/dx tutarlılığını bozmadığı için türev testi de
    görmüyordu.
    """
    alan = Harmonic(k=1.0, center=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    enerji, kuvvet = alan(np.array([[1.0, 0.0, 0.0], [6.0, 0.0, 0.0]]))
    assert enerji == pytest.approx(1.0, abs=1e-12, rel=0.0)
    np.testing.assert_allclose(
        kuvvet, np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), rtol=0.0, atol=1e-12
    )


def test_harmonik_gecersiz_merkezi_reddediyor():
    """`center` hiç doğrulanmıyordu; depo deseni (types.py Atom) tersini yapıyor.

    ÖLÇÜLDÜ (muhafazasız): nan merkez SESSİZCE nan enerji/kuvvet veriyor;
    (3,) merkez ile dört parçacık sessizce 0.0 döndürüyor.
    """
    with pytest.raises(InvalidInputError, match="merkez"):
        Harmonic(k=1.0, center=np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(InvalidInputError, match="merkez"):
        Harmonic(k=1.0, center=np.zeros(3))


def test_harmonik_kuvveti_potansiyelin_turevi():
    """F = -dV/dx sayısal olarak doğrulanır.

    Bu, kuvvet ile enerjinin AYRI AYRI doğru ama BİRBİRİYLE tutarsız olduğu
    hatayı yakalar; ne enerji testi ne kuvvet testi tek başına yakalar.

    MERKEZ (1,3), KONUMLAR (2,3): yayılım BİLEREK korunuyor -- tek merkeze
    bağlı N parçacık geçerli bir kurulumdur.
    """
    alan = Harmonic(k=0.37, center=np.array([[0.1, -0.2, 0.3]]))
    konumlar = np.array([[1.0, 0.5, -0.4], [-0.7, 0.2, 0.9]])
    _, analitik = alan(konumlar)
    np.testing.assert_allclose(
        analitik, _sayisal_kuvvet(alan, konumlar), rtol=1e-7, atol=1e-12
    )


def test_harmonik_gecersiz_yay_sabitini_reddediyor():
    # ÖLÇÜLDÜ: yalnız k=0 sınanırken nan/inf kabul eden mutant geçiyordu
    # (`k <= 0` kontrolü `k < 0`a çevrilince ise yakalanıyordu).
    for kotu in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(InvalidInputError, match="yay sabiti"):
            Harmonic(k=kotu, center=np.zeros((1, 3)))

#: H2 Morse sabitleri (Huber & Herzberg 1979 / NIST WebBook, tasarım §4.2)
#: `h2_morse_sabitleri.py`den okunur (C6: eskiden burada ve
#: `test_dissociation.py`de iki ayrı literal kopyaydı).


def test_morse_denge_uzunlugunda_enerji_ve_kuvvet_sifir():
    alan = Morse(d_e=H2_DE, r_e=H2_RE, a=H2_A)
    enerji, kuvvet = alan(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, H2_RE]]))
    assert enerji == pytest.approx(0.0, abs=1e-15, rel=0.0)
    np.testing.assert_allclose(kuvvet, np.zeros((2, 3)), rtol=0.0, atol=1e-12)


def test_morse_ayrisma_limitinde_enerji_De():
    """V(sonsuz) = De. Ölçüldü: oran 1.00000000000000.

    UYARI (tasarım §4.3): exp(-a(r-r_e)) TAŞMAZ, ALT-TAŞAR -- r ~ 725.5
    bohr'da tam 0.0 olur ve numpy varsayılanı `under='ignore'` olduğu için
    sessizdir. 50 bohr o sınırın çok altında, güvenli.
    """
    alan = Morse(d_e=H2_DE, r_e=H2_RE, a=H2_A)
    enerji, _ = alan(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 50.0]]))
    # SINIR abs=1e-15: ÖLÇÜLDÜ, r=50 bohr'da doğru kodun sapması 0.0.
    # BU SINIR NEYİ YAKALAR: yalnızca `a`nın ~0.72'nin ALTINA düşmesini
    # (a=0.7 -> 3.33e-15, kırmızı). `a`nın BÜYÜMESİNİ ASLA yakalamaz --
    # a=2.058 ve a=1e6 için sapma tam 0.0. Üst sınır V''(r_e) testinden gelir.
    assert enerji / H2_DE == pytest.approx(1.0, abs=1e-15, rel=0.0)


def test_morse_kuvveti_potansiyelin_turevi():
    """Durağan NOKTADAN UZAKTA sınanır -- r_e'de bağıl ölçüt patlar.

    ÖLÇÜLDÜ: Morse r_e'de analitik kuvvet TAM -0.0, sayısal 1.90e-13; numpy'nin
    bağıl ölçütü bölen sıfır olduğu için 1.0 verir. r_e bu döngüden çıkarıldı,
    ama bu bir ZORUNLULUK DEĞİL TERCİHTİR: r_e testin kendi toleranslarıyla
    (rtol=1e-6, atol=1e-12) zaten geçiyor. Eskiden burada yazan "bağıl fark
    3.87e+05" sayısı hiçbir kurulumda üretilemedi.
    """
    alan = Morse(d_e=H2_DE, r_e=H2_RE, a=H2_A)
    for uzunluk in (H2_RE - 0.3, H2_RE + 0.5, H2_RE + 2.0):
        konum = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, uzunluk]])
        _, analitik = alan(konum)
        np.testing.assert_allclose(
            analitik, _sayisal_kuvvet(alan, konum), rtol=1e-6, atol=1e-12
        )


def test_morse_harmonik_limiti_ALANIN_KENDISINDEN_olculuyor():
    """omega0 = sqrt(V''(r_e)/mu); V'' ALANDAN sayısal türevle okunur.

    ESKİ HÂLİ KÖRDÜ. `a*sqrt(2De/mu)` ile `sqrt(2*De*a^2/mu)` cebirsel olarak
    AYNI ifadedir -- ölçüldü, fark 3.47e-18 (1 ulp) -- ve ikisi de Morse'u
    HİÇ ÇAĞIRMIYORDU. Sonuç: `_a = 2*self.a` ve `_a = 1/self.a` mutantları
    beş Morse testini de geçiyordu. "Bağımsız çapraz kontrol" ifadesi yanlıştı.

    ŞİMDİ V'' alandan geliyor. ÖLÇÜLDÜ: doğru kod 0.3693947988 -> 4401.21
    cm^-1; a*2 mutantı 1.4775792227 -> 8802.42 cm^-1; 1/a mutantı
    0.3294831111 -> 4156.65 cm^-1. İkisi de kırmızı.

    DAİRESELLİK UYARISI (tasarım §4.3): `a` zaten omega_e'den türetildi, yani
    bu sayı H2 FİZİĞİNİ doğrulamaz -- yalnızca uygulamanın parametreyi doğru
    kullandığını. Dairesel olmayan çıpa Plan D'de gelir (HF kendi eğrisini
    üretir).
    """
    alan = Morse(d_e=H2_DE, r_e=H2_RE, a=H2_A)

    def V(r: float) -> float:
        return alan(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]))[0]

    h = 1e-4
    k_alan = (V(H2_RE + h) + V(H2_RE - h) - 2 * V(H2_RE)) / h**2
    assert k_alan == pytest.approx(0.3693947966, rel=1e-6, abs=0.0)
    assert np.sqrt(k_alan / H2_MU) * HARTREE_TO_WAVENUMBER == pytest.approx(
        4401.21, abs=1.0, rel=0.0
    )


def test_morse_iki_atomdan_baskasini_reddediyor():
    alan = Morse(d_e=H2_DE, r_e=H2_RE, a=H2_A)
    with pytest.raises(InvalidInputError, match="iki atom"):
        alan(np.zeros((3, 3)))

#: Argon, atomik birime çevrilmiş (tasarım §4.2/§4.4).
AR_EPS = 3.7938402530e-04   # hartree (119.8 K * k_B)
AR_SIGMA = 6.4345174543     # bohr (3.405 A)


def _iki_parcacik(mesafe: float) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, mesafe]])


def test_lj_sigma_noktasinda_enerji_sifir():
    """V(sigma) = 0 analitiktir; uygulamanın döndürdüğü sayı tam sıfır DEĞİLDİR.

    Uygulama V(r_c) kaydırmasını HER ZAMAN çıkardığı için dönen değer
    +1.0770476839e-28'dir (= -V(1e5), kesme katkısı). abs=1e-18 sınırı bunun
    9.3 milyar katı üstünde, yani çıpa kaydırma teriminden etkilenmiyor --
    ama "tam 0.000e+00" demek yanlıştı.
    """
    alan = LennardJones(epsilon=AR_EPS, sigma=AR_SIGMA, box_length=1e6, cutoff=1e5)
    enerji, _ = alan(_iki_parcacik(AR_SIGMA))
    assert enerji == pytest.approx(0.0, abs=1e-18, rel=0.0)


def test_lj_minimum_konumu_ve_derinligi():
    """r_min = 2^(1/6) sigma = 7.2225016416 bohr, V(r_min) = -epsilon.

    ÖLÇÜLDÜ: V(r_min)/(-eps) = 1.00000000000000; F(r_min) z bileşeni
    -4.1989e-19, |F|max = 4.1988863235e-19 (sıfır değil, sayısal artık).
    Üç çıpa (V(sigma)=0, r_min, V(r_min)) BİRLİKTE epsilon'u, sigma'yı ve
    12-6 üs yapısını kilitler -- hiçbiri tek başına yetmez.
    """
    alan = LennardJones(epsilon=AR_EPS, sigma=AR_SIGMA, box_length=1e6, cutoff=1e5)
    r_min = 2 ** (1 / 6) * AR_SIGMA
    assert r_min == pytest.approx(7.2225016416, abs=1e-9, rel=0.0)
    enerji, kuvvet = alan(_iki_parcacik(r_min))
    assert enerji / (-AR_EPS) == pytest.approx(1.0, abs=1e-12, rel=0.0)
    np.testing.assert_allclose(kuvvet, np.zeros((2, 3)), rtol=0.0, atol=1e-15)


def test_lj_kuvveti_potansiyelin_turevi():
    """Durağan noktadan UZAKTA sınanır.

    r_min'de analitik kuvvet neredeyse sıfır olduğu için numpy'nin bağıl
    ölçütü bölen sıfıra gidince 1.0 verir ve karşılaştırma anlamsızlaşır.
    (Burada eskiden "ölçüldü 3.87e+05" yazıyordu; o sayı ne LJ'de ne
    Morse'ta üretilebildi -- ölçülen değer 1.0.)
    """
    alan = LennardJones(epsilon=AR_EPS, sigma=AR_SIGMA, box_length=1e6, cutoff=1e5)
    for oran in (0.80, 1.00, 1.50, 2.00):
        konum = _iki_parcacik(oran * AR_SIGMA)
        _, analitik = alan(konum)
        np.testing.assert_allclose(
            analitik, _sayisal_kuvvet(alan, konum), rtol=1e-6, atol=1e-20
        )


def test_lj_kesme_yaricapinda_enerji_SICRAMIYOR():
    """Kaydırılmamış kesme enerjiyi sıçratır ve enerji korunumunu bozar.

    V_kaydirilmis(r) = V(r) - V(r_c) ile kesmede süreklilik sağlanır.
    HANGİ HATAYI YAKALAR: kaydırma terimini düşüren mutasyon -- ölçüldü,
    o mutantta |iç - dış| = 6.19e-06, sınırın 2.7 milyon katı.

    SINIR 1e-9, 1e-12 DEĞİL. Doğru uygulamada ölçülen fark 2.30e-12'dir ve bu
    bir kusur değil POTANSİYELİN KENDİ EĞİMİDİR: |dV/dr| * 2e-6 = 2.2994e-12,
    yani prob mesafesiyle birebir ölçekleniyor (prob 1e-7 -> 2.2994e-13,
    1e-9 -> 2.2994e-15). 1e-12 yazılırsa DOĞRU KOD kırmızı olur. Marj 435x.
    """
    r_c = 2.5 * AR_SIGMA
    alan = LennardJones(epsilon=AR_EPS, sigma=AR_SIGMA, box_length=1e6, cutoff=r_c)
    ic, _ = alan(_iki_parcacik(r_c - 1e-6))
    dis, _ = alan(_iki_parcacik(r_c + 1e-6))
    assert abs(ic - dis) < 1e-9


def test_lj_kesme_yaricapinin_DISINDAKI_cifti_gercekten_atiyor():
    """Kesmenin kendisi hiçbir testte kilitli DEĞİLDİ -- mutasyonla bulundu.

    ÖLÇÜLDÜ: `icinde = r2 < self.cutoff**2` koşulunu `r2 < inf` yapan mutant
    (yani kesmeyi hiç uygulamayan kod) bu dosyadaki YEDİ LJ testini de
    geçiyordu. Sebep: diğer testlerin hepsinde çift zaten kesmenin İÇİNDE, ve
    süreklilik testi kaydırma yüzünden mutantta da sürekli çıkıyor.

    Aynı çift, aynı mesafe, YALNIZ kesme değişiyor:
      cutoff = 2.5 sigma, çift 3.0 sigma uzakta -> E = 0.0,        |F| = 0.0
      cutoff = 3.5 sigma, aynı çift             -> E = -1.253736e-06,
                                                   |F|max = 6.452565e-07
    """
    mesafe = 3.0 * AR_SIGMA
    disarida = LennardJones(
        epsilon=AR_EPS, sigma=AR_SIGMA, box_length=1e6, cutoff=2.5 * AR_SIGMA
    )
    iceride = LennardJones(
        epsilon=AR_EPS, sigma=AR_SIGMA, box_length=1e6, cutoff=3.5 * AR_SIGMA
    )
    e_dis, k_dis = disarida(_iki_parcacik(mesafe))
    e_ic, k_ic = iceride(_iki_parcacik(mesafe))

    assert e_dis == 0.0
    np.testing.assert_array_equal(k_dis, np.zeros((2, 3)))
    assert e_ic == pytest.approx(-1.253736e-06, rel=1e-5, abs=0.0)
    assert float(np.abs(k_ic).max()) == pytest.approx(6.452565e-07, rel=1e-5, abs=0.0)


def test_lj_gecersiz_sekli_reddediyor():
    """Kardeş sınıf `Morse` bu muhafazayı yapıyor, `LennardJones` yapmıyordu.

    ÖLÇÜLDÜ (muhafazasız): (2,2) şekli SESSİZCE kabul ediliyor ve sonuç (2,2)
    çıkıyor -- yani 2 boyutlu koordinatla fizik yapılıyordu. 1B (3,) ham
    `IndexError` veriyordu; QLabError DEĞİL, yani Global Constraint "ham hata
    tüketiciye ulaşmaz" ihlali.

    int dtype AYRI BİR DURUM ve HATA DEĞİLDİR: muhafazasız ham
    `UFuncTypeError` veriyordu, ama doğru davranış onu float'a ÇEVİRMEKTİR
    (MDState de öyle yapıyor). Ölçüldü -- çevrilen sonuç float girdiyle
    birebir aynı. Bu testin ilk taslağı int dtype'ın da yükseltmesini
    bekliyordu; yanlıştı.
    """
    alan = LennardJones(epsilon=AR_EPS, sigma=AR_SIGMA, box_length=1e6, cutoff=1e5)
    for kotu in (np.zeros((2, 2)), np.zeros(3), np.zeros((4, 3, 2))):
        with pytest.raises(InvalidInputError, match=r"\(N,3\)"):
            alan(kotu)

    # int dtype HATA DEĞİL: ham `UFuncTypeError` veriyordu, muhafaza onu
    # float'a çeviriyor. ÖLÇÜLDÜ -- sonuç float girdiyle BİREBİR aynı.
    tam = alan(np.array([[0, 0, 0], [0, 0, 7]], dtype=int))
    ondalik = alan(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 7.0]]))
    assert tam[0] == ondalik[0]
    np.testing.assert_array_equal(tam[1], ondalik[1])


def test_lj_minimum_goruntu_kutuyu_gecen_ciftleri_buluyor():
    """Periyodik kutuda en yakın görüntü kullanılır.

    Kutu 20 bohr; iki parçacık 19 bohr uzakta görünüyor ama minimum görüntü
    mesafesi 1 bohr'dur. Uygulanmazsa kuvvet ihmal edilebilir çıkar;
    uygulanırsa devasa (itici) olur.
    """
    alan = LennardJones(epsilon=AR_EPS, sigma=AR_SIGMA, box_length=20.0, cutoff=9.0)
    _, kuvvet = alan(np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 19.5]]))
    assert float(np.abs(kuvvet).max()) > 1.0


def test_lj_kesme_kutu_yarisini_asamaz():
    """r_c > L/2 minimum görüntüyü bozar: aynı çift iki kez sayılır."""
    with pytest.raises(InvalidInputError, match="kesme"):
        LennardJones(epsilon=AR_EPS, sigma=AR_SIGMA, box_length=10.0, cutoff=6.0)
