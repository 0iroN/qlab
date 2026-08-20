"""Bant yapisi -- ve Modul 2E'nin izgara paritesi tuzaginin k-uzayi kardesi.

OLCULMUS TEHLIKE: bant kenari izgara ustunde okunursa, dogru degeri veren
k noktasi izgarada yoksa sonuc SESSIZCE yanlis cikar -- ve SIKLASTIRMAK
DUZELTMEZ. Grafende olculdu: 601x601 izgara BZ minimum araligini
2.471886e-15 eV verirken 600x600 1.635969e-02 eV, 1000x1000 ise
1.698157e-02 eV veriyor; yani DAHA SIK olan izgara 601'den 13 MERTEBE kotu.

Bu yuzden `band_edges` izgara KULLANMAZ, surekli minimizasyon yapar; izgara
yalniz cizim ve DOS icindir.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.tb.bands import (
    band_edges,
    band_structure,
    k_path,
    uniform_k_grid,
)
from qlab.engine.tb.hamiltonian import Hopping, TBModel
from qlab.engine.tb.lattice import Lattice, reciprocal_vectors
from qlab.errors import InvalidInputError

A = 2.5
T = -0.0625
EPS = -0.1948


def _zincir(t: float = T, eps: float = EPS) -> TBModel:
    orgu = Lattice(vectors=np.array([[A, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    return TBModel(
        lattice=orgu,
        onsite=np.array([eps]),
        hoppings=(
            Hopping(i=0, j=0, cell=(1, 0, 0), t=t),
            Hopping(i=0, j=0, cell=(-1, 0, 0), t=t),
        ),
    )


def _ikinci_komsulu() -> TBModel:
    """Bant minimumu artik k=pi/a'da DEGIL -- izgara onu kolayca kaciriyor."""
    orgu = Lattice(vectors=np.array([[A, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    return TBModel(
        lattice=orgu,
        onsite=np.array([EPS]),
        hoppings=(
            Hopping(i=0, j=0, cell=(1, 0, 0), t=-0.0625),
            Hopping(i=0, j=0, cell=(-1, 0, 0), t=-0.0625),
            Hopping(i=0, j=0, cell=(2, 0, 0), t=+0.0500),
            Hopping(i=0, j=0, cell=(-2, 0, 0), t=+0.0500),
        ),
    )


def _ssh(t1: float = -0.10, t2: float = -0.04) -> TBModel:
    orgu = Lattice(
        vectors=np.array([[4.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]),
    )
    return TBModel(
        lattice=orgu,
        onsite=np.array([0.0, 0.0]),
        hoppings=(
            Hopping(i=0, j=1, cell=(0, 0, 0), t=t1),
            Hopping(i=1, j=0, cell=(0, 0, 0), t=t1),
            Hopping(i=1, j=0, cell=(1, 0, 0), t=t2),
            Hopping(i=0, j=1, cell=(-1, 0, 0), t=t2),
        ),
    )


def test_IZGARA_bant_genisligi_PARITEYE_bagli():
    """OLCULDU: TEK n'de kapali form 2|t|(1+cos(pi/n)) TAM tutuyor
    (fark 2.78e-17), CIFT n'de bant genisligi zaten tam 4|t| cikiyor.

    Tek n'deki oranlar: n=3 -> 0.750000, n=7 -> 0.950484, n=17 -> 0.991487,
    n=101 -> 0.999758. Hepsi gercek degerin ALTINDA ve hata monoton
    kuculuyor -- tuzagin "yakinsama sorunu gibi gorunme" ozelligi tam
    olarak budur.
    """
    model = _zincir()
    gercek = 4.0 * abs(T)
    for n in (3, 7, 17, 101):
        e = band_structure(model, uniform_k_grid(model.lattice, n))
        W = float(e.max() - e.min())
        assert abs(W - 2.0 * abs(T) * (1.0 + np.cos(np.pi / n))) < 1e-15
        assert W < gercek
    for n in (4, 8, 18, 102):
        e = band_structure(model, uniform_k_grid(model.lattice, n))
        assert abs(float(e.max() - e.min()) - gercek) < 1e-15


def test_SUREKLI_minimizasyon_izgaradan_BAGIMSIZ():
    """`band_edges` izgara kullanmaz; TEK n'de bile dogru genisligi verir."""
    alt, ust = band_edges(_zincir(), band_index=0)
    assert abs((ust - alt) - 4.0 * abs(T)) < 1e-9
    assert abs(alt - (EPS - 2.0 * abs(T))) < 1e-9
    assert abs(ust - (EPS + 2.0 * abs(T))) < 1e-9


def test_SSH_araligi_izgara_PARITESINE_bagli():
    """Aralik k = G/2'de; izgara onu ancak CIFT n'de yakaliyor.

    OLCULDU (gercek aralik 0.120000): cift n -> tam 0.120000 (%0.0000);
    tek n -> n=5 %+19.3485, n=7 %+10.4567, n=9 %+6.4902, n=11 %+4.4038,
    n=13 %+3.1782. Tek n'de hata monoton kuculuyor ama SIFIRLANMIYOR.
    """
    model = _ssh()
    gercek = 2.0 * (0.10 - 0.04)
    for n in (4, 6, 8, 10, 12, 14):
        e = band_structure(model, uniform_k_grid(model.lattice, n))
        assert abs(float(e[:, 1].min() - e[:, 0].max()) - gercek) < 1e-15
    beklenen_yuzde = {5: 19.3485, 7: 10.4567, 9: 6.4902, 11: 4.4038, 13: 3.1782}
    for n, yuzde in beklenen_yuzde.items():
        e = band_structure(model, uniform_k_grid(model.lattice, n))
        olculen = float(e[:, 1].min() - e[:, 0].max())
        assert abs((olculen / gercek - 1.0) * 100.0 - yuzde) < 1e-3


def test_BANT_MINIMUMU_hatasi_MONOTON_BILE_DEGIL():
    """EN SINSI OLCUM: izgarayi SIKLASTIRMAK hatayi duzenli kucultmuyor.

    Ikinci komsulu zincirde bant minimumu artik yuksek simetri noktasinda
    degil. Referans ANALITIK: E = eps + 2t1*cos(ka) + 2t2*cos(2ka), u=cos(ka)
    yazilirsa E = eps - 2t2 + 2t1*u + 4t2*u^2 ve u* = -t1/(4t2) = 0.3125,
    E_min = -0.314331250000. (Izgara referansi degil ALGEBRA -- ve n=200001
    izgarasiyla farki olculdu: 0.000e+00. Izgara yolu 2.8 s tutuyordu,
    analitik yol 0.03 s.)

    OLCULDU:
      n= 5 -> hata 2.426e-06   (neredeyse tam)
      n= 7 -> hata 1.934e-02   (DORT MERTEBE kotu)
      n=12 -> hata 7.031e-03
      n=13 -> hata 7.370e-03   (n=12'den KOTU)
    Yani n=5, n=13'ten yaklasik 3000 KAT daha iyi. Ayni desen buyuk n'de de
    surer: n=2001 hatasi 1.936e-09 iken n=20001 hatasi 3.849e-09 -- ON KAT
    SIK izgara IKI KAT kotu. "Izgarayi biraz daha sik yapayim" refleksi
    burada ise yaramaz; dogru cozum izgarayi hic kullanmamaktir
    (`band_edges`).
    """
    model = _ikinci_komsulu()
    t1, t2 = -0.0625, 0.0500
    u = -t1 / (4.0 * t2)
    analitik_min = EPS - 2.0 * t2 + 2.0 * t1 * u + 4.0 * t2 * u * u
    assert analitik_min == pytest.approx(-0.314331250000, abs=1e-12)

    hata = {}
    for n in (5, 7, 12, 13, 2001, 20001):
        e = band_structure(model, uniform_k_grid(model.lattice, n))
        hata[n] = float(e[:, 0].min()) - analitik_min

    assert hata[5] == pytest.approx(2.426e-06, abs=1e-9)
    assert hata[7] == pytest.approx(1.934e-02, abs=1e-5)
    assert hata[13] > hata[12]  # SIKLASTIRMAK KOTULESTIRDI
    assert hata[13] / hata[5] > 1000.0
    assert hata[20001] > hata[2001]  # ...buyuk n'de de ayni desen

    # ...ve surekli minimizasyon izgaradan bagimsiz olarak dogruyu buluyor
    alt, _ = band_edges(model, band_index=0)
    assert abs(alt - analitik_min) < 1e-9


def test_must_include_IZGARADA_OLMAYAN_noktayi_REDDEDIYOR():
    """Parite kurali SIHIRLI SAYIYLA DEGIL, UYELIK DENETIMIYLE korunur.

    On olcumde iki ajan iki farkli izgara konvansiyonu kullandi ve iki
    farkli "sihirli N" buldu (biri 601'in K'yi yakaladigini, oteki
    N % 3 == 0 kuralini olctu). Ikisi de kendi konvansiyonunda dogru --
    ve tam bu yuzden sihirli sayi KODA YAZILMAZ. Cagiran hangi k
    noktalarinin izgarada bulunmasi gerektigini SOYLER, kod DENETLER.
    """
    model = _ssh()
    G = reciprocal_vectors(model.lattice)[0]
    kenar = 0.5 * G  # aralik burada
    with pytest.raises(InvalidInputError, match="izgarada yok"):
        uniform_k_grid(model.lattice, 5, must_include=(kenar,))
    k = uniform_k_grid(model.lattice, 6, must_include=(kenar,))
    assert k.shape == (6, 3)


def test_k_path_uzunluklar_ve_isaretler():
    model = _zincir()
    G = reciprocal_vectors(model.lattice)[0]
    yol = k_path((np.zeros(3), 0.5 * G, G), n_per_segment=11)
    assert yol.points.shape == (21, 3)  # 11 + 11 - 1 (ortak nokta bir kez)
    assert yol.distances[0] == 0.0
    assert np.all(np.diff(yol.distances) >= 0.0)
    assert len(yol.tick_distances) == 3
    assert yol.tick_distances[0] == 0.0
    assert abs(yol.tick_distances[-1] - yol.distances[-1]) < 1e-14


def test_band_structure_SIRALI_donuyor_ve_sekli_dogru():
    model = _ssh()
    e = band_structure(model, uniform_k_grid(model.lattice, 9))
    assert e.shape == (9, 2)
    assert np.all(np.diff(e, axis=1) >= -1e-15)


def test_band_structure_SIRALAMA_BANT_KIMLIGI_DEGIL():
    """DURUST OLUMSUZ, testle kilitleniyor.

    OLCULDU: `eigh` ozdegerleri her k'da kucukten buyuge dizer, ama gercek
    bantlar KESISIR. Sirali "bant 0"in etkin kutlesi -6.42e-05 cikarken
    gercek bandinki -8.000e-01 -- 12457 KAT hata. Yani `band_structure`in
    ciktisi CIZIM icin dogrudur, BANT KIMLIGI icin degildir.

    Iki BAGIMSIZ zincir (biri genis, biri dar ve ters isaretli) kesisen
    bantlar uretir; sirali dizi kesisme noktasinda KOSE yapar.
    """
    orgu = Lattice(
        vectors=np.array([[4.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
    )
    model = TBModel(
        lattice=orgu,
        onsite=np.array([-0.05, 0.05]),
        hoppings=(
            Hopping(i=0, j=0, cell=(1, 0, 0), t=-0.08),
            Hopping(i=0, j=0, cell=(-1, 0, 0), t=-0.08),
            Hopping(i=1, j=1, cell=(1, 0, 0), t=+0.03),
            Hopping(i=1, j=1, cell=(-1, 0, 0), t=+0.03),
        ),
    )
    e = band_structure(model, uniform_k_grid(orgu, 201))
    ikinci_fark = np.abs(np.diff(e[:, 0], n=2))
    assert ikinci_fark.max() > 20.0 * np.median(ikinci_fark), (
        "siralamanin kose yaptigi gorulemedi -- kesisme kurulamamis"
    )


def test_gecersiz_girdiler_reddediliyor():
    model = _zincir()
    with pytest.raises(InvalidInputError, match="n en az"):
        uniform_k_grid(model.lattice, 1)
    with pytest.raises(InvalidInputError, match="bant indisi"):
        band_edges(model, band_index=7)
    with pytest.raises(InvalidInputError, match="en az iki"):
        k_path((np.zeros(3),), n_per_segment=5)


def test_IKI_BOYUTTA_coklu_baslangic_GEREKLI():
    """`band_edges`in cok-baslangicli taramasi 1B'de OLU KOD, 2B'de SART.

    OLCULDU: `minimize_scalar(method="bounded")` baslangic noktasini
    KULLANMAZ, yalniz `bounds` kullanir. 1B'de tek eksen oldugu icin her
    baslangic AYNI sonucu verir -- yani coklu baslangic dongusunu silen
    mutant bu dosyanin 1B testlerinin HEPSINDEN geciyordu.

    2B'de eksen-eksen inis baslangica BAGLI. 25 rastgele 2B model x 2 uc
    deger = 50 aramada olculdu: TEK baslangic 31/50 basarisiz (en kotu
    bagil sapma 5.149e-01, yani bant genisliginin yarisi); 5x5 baslangicla
    8/50. Asagidaki somut modelde tek baslangic bant genisliginin
    %30.08'i kadar sapiyor.

    DURUST OLUMSUZ: coklu baslangic da GARANTI DEGIL, yalniz daha iyi bir
    SINIR -- 5x5 ile bile 8/50 basarisiz. Cok boyutta `band_edges` bir
    ust/alt SINIR dondurur, kanitlanmis kuresel uc deger degil.

    Yan olcum: `band_edges` (0.23 s) 201x201 izgaradan (1.79 s) DAHA
    DOGRU sonuc veriyor -- izgara -0.323881336'da kalirken surekli arama
    -0.323908000 buluyor.
    """
    orgu = Lattice(
        vectors=np.array([[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]),
        basis=np.zeros((1, 3)),
    )
    tx, ty, td, t2x, t2y = (-0.040377, -0.001626, 0.044219, -0.029180, -0.005585)
    model = TBModel(
        lattice=orgu,
        onsite=np.array([0.0]),
        hoppings=tuple(
            Hopping(i=0, j=0, cell=c, t=t)
            for c, t in (
                ((1, 0, 0), tx), ((-1, 0, 0), tx),
                ((0, 1, 0), ty), ((0, -1, 0), ty),
                ((1, 1, 0), td), ((-1, -1, 0), td),
                ((1, -1, 0), td), ((-1, 1, 0), td),
                ((2, 0, 0), t2x), ((-2, 0, 0), t2x),
                ((0, 2, 0), t2y), ((0, -2, 0), t2y),
            )
        ),
    )
    alt, ust = band_edges(model, band_index=0)
    izgara = band_structure(model, uniform_k_grid(orgu, 61))[:, 0]

    # Surekli arama izgaradan DAHA IYI (ya da esit) olmali -- ayni
    # fonksiyonun surekli uc degeri, ayrik ornekleminkinden asagida olur.
    assert alt <= float(izgara.min()) + 1e-9
    assert ust >= float(izgara.max()) - 1e-9

    # ...ve tek baslangicin dustugu yerel tuzaga DUSMEMELI (o ~%30 yukarida)
    assert abs(alt - (-0.323908)) < 1e-5
