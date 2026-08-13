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

from qlab.engine.md.forcefields import Harmonic, Morse
from qlab.errors import InvalidInputError
from qlab.units import HARTREE_TO_WAVENUMBER


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

#: H2, Huber & Herzberg 1979 / NIST WebBook (tasarım §4.2).
H2_DE = 0.174434333      # hartree (4.7466 eV)
H2_RE = 1.401118538      # bohr (0.74144 A)
H2_A = 1.028997701       # bohr^-1, a = omega_e*sqrt(mu/(2*De))
H2_MU = 918.576324       # m_e


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
