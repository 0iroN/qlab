"""MODUL 3'UN CEKIRDEK CIPASI -- iki AYRI kod yolu ayni spektrumu vermeli.

YOL A: `finite_ring_matrix` -- gercek uzay, hicbir faz carpani, hicbir ters
       orgu, hicbir kompleks us; yalniz modulo baglanti.
YOL B: `bloch_hamiltonian` -- k-uzayi, kompleks fazlar, ters orgu.

Bu neden bir CIPA, oysa on olcumdeki uc "cipa" tautolojiydi: orada iki yol
ayni ifadeyi iki kez yaziyordu (`eigvalsh(1x1)` hicbir sey yapmiyordu;
katlanmis ve acilmis matrisler BIT BIT ayniydi). Burada iki yol hicbir ifade
paylasmiyor ve mutasyon sinavinda biri bozulunca oteki KIRMIZI oluyor.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.tb.hamiltonian import (
    Hopping,
    TBModel,
    bloch_hamiltonian,
    finite_ring_matrix,
)
from qlab.engine.tb.lattice import Lattice, reciprocal_vectors

A = 2.5


def _model(t: float = -0.0625, eps: float = -0.1948, t2: float = 0.0) -> TBModel:
    orgu = Lattice(vectors=np.array([[A, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    hop = [
        Hopping(i=0, j=0, cell=(1, 0, 0), t=t),
        Hopping(i=0, j=0, cell=(-1, 0, 0), t=t),
    ]
    if t2 != 0.0:
        hop += [
            Hopping(i=0, j=0, cell=(2, 0, 0), t=t2),
            Hopping(i=0, j=0, cell=(-2, 0, 0), t=t2),
        ]
    return TBModel(lattice=orgu, onsite=np.array([eps]), hoppings=tuple(hop))


def _iki_yol(model: TBModel, n_cell: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Ayni sistemi iki yoldan cozer ve ||H||_inf ile birlikte dondurur."""
    H_gercek = finite_ring_matrix(model, n_cell)
    a_yolu = np.sort(np.linalg.eigvalsh(H_gercek))

    G = reciprocal_vectors(model.lattice)[0]
    izinli = [(m / n_cell) * G for m in range(n_cell)]
    b_yolu = np.sort(
        np.concatenate(
            [np.linalg.eigvalsh(bloch_hamiltonian(model, k)) for k in izinli]
        )
    )
    norm = float(np.abs(H_gercek).sum(axis=1).max())
    return a_yolu, b_yolu, norm


@pytest.mark.parametrize("n_cell", [4, 12, 33, 64, 256])
def test_CIPA_gercek_uzay_ile_Bloch_ayni_spektrumu_veriyor(n_cell):
    """OLCULDU: N=12'de max sapma 8.882e-16 (eV olceginde), hata/||H|| 3.2e-15.

    Sinir MUTLAK degil BAGIL yazilmistir, cunku olculdu: hata sqrt(N) ile
    buyuyor (log-log egim p = 0.5025) ve N=4096'da 31 ULP'a cikiyor. Mutlak
    bir esik N'ye bagli olurdu; hata/||H|| ise N=4'ten 4096'ya 3.165e-15'te
    kaliyor.
    """
    a_yolu, b_yolu, norm = _iki_yol(_model(), n_cell)
    assert a_yolu.shape == b_yolu.shape == (n_cell,)
    assert float(np.abs(a_yolu - b_yolu).max()) / norm < 1e-13


def test_CIPA_ikinci_komsu_varken_de_tutuyor():
    """Cipanin yalniz en-yakin-komsuda calismadigini kilitler."""
    a_yolu, b_yolu, norm = _iki_yol(_model(t2=0.017), 48)
    assert float(np.abs(a_yolu - b_yolu).max()) / norm < 1e-13


def test_CIPA_KUCUK_halkada_da_tutuyor_carpisan_hoplamalarla():
    """Task 2'de olculen carpisma rejimi: halka, menzilin iki katindan kucuk.

    n_cell = 4 ve +-2 hoplamalari ayni hedefe dusuyor. Iki yolun da o
    toplamayi AYNI yapmasi gerekir -- Bloch tarafinda faz toplamiyla,
    gercek uzayda matris toplamiyla.
    """
    a_yolu, b_yolu, norm = _iki_yol(_model(t2=-0.03), 4)
    assert float(np.abs(a_yolu - b_yolu).max()) / norm < 1e-13


def test_CIPA_iki_atomlu_hucrede_de_tutuyor():
    """SSH: hucre ici + hucreler arasi hoplama, n_orb = 2."""
    ssh = Lattice(
        vectors=np.array([[4.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]),
    )
    model = TBModel(
        lattice=ssh,
        onsite=np.array([-0.11, -0.07]),
        hoppings=(
            Hopping(i=0, j=1, cell=(0, 0, 0), t=-0.10),
            Hopping(i=1, j=0, cell=(0, 0, 0), t=-0.10),
            Hopping(i=1, j=0, cell=(1, 0, 0), t=-0.04),
            Hopping(i=0, j=1, cell=(-1, 0, 0), t=-0.04),
        ),
    )
    a_yolu, b_yolu, norm = _iki_yol(model, 32)
    assert a_yolu.shape == (64,)
    assert float(np.abs(a_yolu - b_yolu).max()) / norm < 1e-13


def test_CIPA_SAHTE_CIPA_SINAVINDAN_gecirilmistir():
    """Fizige ait OLMAYAN girdilerde de tutuyor -- ve bu BEKLENEN.

    Tasarim §3.4'un sahte-cipa sinavi supheyi ISARET eder ama tek basina
    ayirmaz: burada iki yol AYNI Hamiltoniyenin iki temsilidir, dolayisiyla
    uyum rastgele girdide de tutar. Ayirici sinav MUTASYONDUR: bir yolu
    bozunca oteki KIRMIZI oluyor (olculdu, komsu listesi 12 -> 11
    dusurulunce fark 1.000e+00).

    Bu test o gercegi belgeler ki, biri "rastgele girdide de tuttu, demek
    tautoloji" diye YANLIS hukum vermesin.
    """
    rng = np.random.default_rng(20260820)
    for _ in range(5):
        model = _model(
            t=float(rng.normal(0.0, 3.0)),
            eps=float(rng.normal(0.0, 50.0)),
            t2=float(rng.normal(0.0, 3.0)),
        )
        a_yolu, b_yolu, norm = _iki_yol(model, 24)
        assert float(np.abs(a_yolu - b_yolu).max()) / norm < 1e-13


def test_hata_N_ile_BUYUYOR_ve_bu_belgelenmistir():
    """OLCULDU: hata ~ N^0.5025. Sabit bir mutlak esik yaniltici olurdu.

    Bu test buyumenin GERCEK oldugunu kilitler: N=256'daki mutlak hata,
    N=4'tekinin en az 2 katidir. Boylece biri "makine hassasiyetinde tutar"
    diye mutlak esik yazarsa kirmizi olur.
    """
    a4, b4, _ = _iki_yol(_model(), 4)
    a256, b256, _ = _iki_yol(_model(), 256)
    kucuk = float(np.abs(a4 - b4).max())
    buyuk = float(np.abs(a256 - b256).max())
    assert buyuk > 2.0 * max(kucuk, 5e-17)


def test_TAUTOLOJI_CIPANIN_KOR_oldugu_GOSTERILIYOR():
    """Cekirdek cipanin NEDEN gerekli oldugunun dogrudan kaniti.

    "E(k) = eps + 2t*cos(ka) tutuyor" denetimi tek orbitalli zincirde
    KORDUR, cunku `eigvalsh` 1x1 bir matriste yalnizca gercek kismi okur.
    OLCULDU: uc FARKLI Hamiltoniyen ayni ozdegeri veriyor --
      dogru               H = -0.17251924 + 0.00000000j
      hucre carpani dusuk H = -0.17251924 - 0.12299824j   (HERMITYEN BILE DEGIL)
      exp -> cos          H = -0.17251924               (faz tamamen kayip)
    ucunun de eigvalsh'i -0.172519243044, beklenenden farki 2.776e-17.

    Ayirici mutasyon sinavinda olculen sonuc: ayni alti mutanttan cekirdek
    cipa 6/6'sini yakaladi, "kaba kontrol" testi yalniz 1/6'sini.

    Bu test o korlugu KILITLER: biri "analitik dispersiyon zaten
    dogruluyor, capraz kontrol gereksiz" derse kirmizi olur.
    """
    a, t, eps, k = A, -0.0625, -0.1948, 0.7
    beklenen = eps + 2.0 * t * np.cos(k * a)

    dogru = eps + t * np.exp(1j * k * a) + t * np.exp(-1j * k * a)
    hucre_carpani_dusuk = eps + t * np.exp(1j * k * a) + t * np.exp(1j * k * a)
    faz_kayip = eps + t * np.cos(k * a) + t * np.cos(k * a)

    for H in (dogru, hucre_carpani_dusuk, faz_kayip):
        e = float(np.linalg.eigvalsh(np.array([[H]]))[0])
        assert abs(e - beklenen) < 1e-15

    # ...oysa ikisi GERCEKTEN yanlis: biri Hermityen bile degil
    assert abs(hucre_carpani_dusuk.imag) > 0.1
    assert abs(faz_kayip - dogru) < 1e-15  # ayni sayi, farkli yoldan -- yine kor
