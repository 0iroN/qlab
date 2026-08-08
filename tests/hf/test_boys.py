import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf, factorial2

from qlab.engine.hf.boys import boys


def boys_by_quadrature(n: int, T: float) -> float:
    """Tanimin dogrudan sayisal integrali -- bagimsiz referans."""
    value, _ = quad(lambda t: t ** (2 * n) * np.exp(-T * t**2), 0.0, 1.0, epsabs=1e-13)
    return value


def double_factorial_odd(n: int) -> float:
    """(2n-1)!! -- n=0 icin (-1)!! = 1 kuraldir.

    scipy.special.factorial2(-1) 0.0 doner (scipy >= 1.11 davranisi, 1.18'de
    olculdu), bu yuzden n=0 durumu elle verilmelidir; dogrudan factorial2
    kullanmak asimptot testini n=0'da sessizce yanlis yapar.
    """
    return 1.0 if n == 0 else float(factorial2(2 * n - 1))


# Bu dosyadaki her pytest.approx cagrisinda IKI parametre de acikca verilir.
# Sebep: pytest.approx max(abs, rel*expected) uygular ve VARSAYILANLARI
# abs=1e-12, rel=1e-6'dir. Boys fonksiyonu buyuk n/T'de cok kucuk degerler
# uretir (n=5, T=120'de F ~ 9.60e-11), orada varsayilan abs=1e-12 bagimsiz
# olarak devreye girer ve "rel=1e-10" yazan bir iddia GERCEKTE ~1.04e-2'lik
# bir goreli bosluk birakir. Asagidaki sinirlar olculmustur (bkz.
# .superpowers/sdd/b1-final-fix-report.md), tahmin degildir.


def test_f0_at_zero_is_one():
    """Olculen sapma tam sifir; yazilan sinir (1e-14) korunup uygulanir
    hale getirildi -- rel=0.0 olmadan varsayilan rel=1e-6 baskin cikiyordu."""
    assert boys(0, 0.0) == pytest.approx(1.0, abs=1e-14, rel=0.0)


def test_fn_at_zero_is_one_over_2n_plus_1():
    """Olculen goreli sapma tam sifir; sinir 1e-12'de baglayici (abs=0.0)."""
    for n in range(6):
        assert boys(n, 0.0) == pytest.approx(1.0 / (2 * n + 1), rel=1e-12, abs=0.0)


def test_f0_closed_form():
    """F_0(T) = sqrt(pi/(4T)) * erf(sqrt(T)).

    Olculen en buyuk goreli sapma 6.86e-16 (~3 ulp); sinir 1e-14, ~15 kat pay.
    """
    for T in (0.1, 1.0, 5.0, 30.0):
        expected = np.sqrt(np.pi / (4 * T)) * erf(np.sqrt(T))
        assert boys(0, T) == pytest.approx(expected, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("T", [1e-8, 0.05, 0.5, 2.0, 12.0, 40.0, 120.0])
def test_matches_numerical_quadrature(n, T):
    """Olculen en buyuk goreli sapma 1.65e-15 (42 nokta uzerinde).

    Sinir 1e-13 secildi (~61 kat pay), 3-10 kat degil: referans burada
    scipy.integrate.quad'in kendi cikitisidir ve quad'in GARANTI ettigi
    yakinsama varsayilan epsrel=1.49e-8'dir -- gozlenen ~1e-15'lik uyum
    quad'in ic davranisinin bir armaganidir, sozlesmesi degil. 3 kat pay,
    scipy surumu degistiginde testi kirilgan yapardi. abs=0.0 guvenlidir:
    float64'un goreli hassasiyeti buyuklukten bagimsizdir, en kucuk deger
    (n=5, T=120'de 9.60e-11) icin bile olculen goreli sapma 4.04e-16'dir.
    """
    assert boys(n, T) == pytest.approx(boys_by_quadrature(n, T), rel=1e-13, abs=0.0)


def test_downward_recursion_consistency():
    """F_n(T) = (2T F_{n+1}(T) + exp(-T)) / (2n+1).

    Olculen en buyuk goreli sapma 6.99e-16 (~3 ulp); sinir 1e-14, ~14 kat
    pay. Burada referans saf cebirdir (kuadratur yok), bu yuzden sinir
    makine hassasiyetine yakin tutulabilir.
    """
    for T in (0.3, 1.7, 9.0, 55.0):
        for n in range(5):
            rhs = (2 * T * boys(n + 1, T) + np.exp(-T)) / (2 * n + 1)
            assert boys(n, T) == pytest.approx(rhs, rel=1e-14, abs=0.0)


def test_monotonically_decreasing_in_T():
    values = [boys(2, T) for T in (0.0, 1.0, 5.0, 20.0, 100.0)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_accepts_array_input():
    result = boys(1, np.array([0.0, 1.0, 10.0]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    # Olculen goreli sapma tam sifir; abs=0.0 ile sinir 1e-12'de baglayici
    # (aksi halde varsayilan abs=1e-12, 1/3 icin ~3e-12'lik bosluk birakirdi).
    assert result[0] == pytest.approx(1.0 / 3.0, rel=1e-12, abs=0.0)


def test_accepts_array_input_mixed_regime():
    """Dizi hem kucuk hem buyuk rejim degerlerini birden icerdiginde,
    her iki dal (kucuk-T kapali form, buyuk-T asimptotik) ayni cagirida
    calismali. T = 44.9 ve 45.1 esik civarinda bilerek secildi.
    """
    T = np.array([1.0, 44.9, 45.1, 100.0])
    for n in (0, 2, 5):
        result = boys(n, T)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        expected = np.array([boys_by_quadrature(n, t) for t in T])
        np.testing.assert_allclose(result, expected, rtol=1e-9)


def test_large_T_asymptote():
    """Buyuk T'de F_n(T) -> (2n-1)!! / 2^{n+1} * sqrt(pi / T^{2n+1}).

    Olculen goreli sapma uc n icin de TAM SIFIR -- boys() T>45'te zaten bu
    asimptotik dali kullaniyor, dolayisiyla bir "pay orani" hesaplanamaz ve
    yazilan sinir (1e-6) korunur; yalnizca uygulanir hale getirilir. Onceki
    halinde varsayilan abs=1e-12, n=3'te (beklenen 5.94e-10) devreye girip
    goreli siniri 1.68e-3'e gevsetiyordu -- iddia edilenden ~1700 kat gevsek.
    """
    T = 500.0
    for n in (0, 1, 3):
        expected = double_factorial_odd(n) / 2 ** (n + 1) * np.sqrt(np.pi / T ** (2 * n + 1))
        assert boys(n, T) == pytest.approx(expected, rel=1e-6, abs=0.0)
