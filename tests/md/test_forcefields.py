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

from qlab.engine.md.forcefields import Harmonic
from qlab.errors import InvalidInputError


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
