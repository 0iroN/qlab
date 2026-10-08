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
    grid_extremum_bound,
    grid_lipschitz_bound,
    hamiltonian_rounding_scale,
    k_path,
    uniform_k_grid,
)
from qlab.engine.tb.hamiltonian import Hopping, TBModel, bloch_hamiltonian
from qlab.engine.tb.lattice import Lattice, reciprocal_vectors
from qlab.errors import InvalidInputError
from qlab.units import HARTREE_TO_EV

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
    bantlar KESISIR. Kesisimin 0.05 bohr^-1 saginda sirali "bant 0"in etkin
    kutlesi +1.6735 (OBUR zincirin), gercek zincir 0'inki -0.6276 -- isaret
    ters ve sonlu fark adimindan bagimsiz. Yani `band_structure`in ciktisi
    CIZIM icin dogrudur, BANT KIMLIGI icin degildir. (Eski "12457 kat"
    yeniden uretilemedi: kesisimin tam uzerinde oran 1/h ile olceklenir,
    ve eski "-0.800" bu modelin kesisimdeki -0.859'uyla tutmuyor.)

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
    # Adimdan BAGIMSIZ kimlik hatasi: kesisimin saginda sirali bant 0 obur
    # zincirin egriligini tasir.
    # Zincirler AYRIK: H(k)'nin kosegeni zincirlerin kendi bantlaridir --
    # gercek kimlik ve kesisim MODELDEN olculur (dogrulama turu: sabit
    # formul model degisse de geciyordu).
    from scipy.optimize import brentq

    def zincir(i, q):
        return float(bloch_hamiltonian(model, np.array([q, 0.0, 0.0]))[i, i].real)

    k_kesisim = brentq(lambda q: zincir(0, q) - zincir(1, q), 1e-6, np.pi / 4.0)
    assert k_kesisim * 4.0 == pytest.approx(2.0427, abs=1e-4)
    k = k_kesisim + 0.05
    assert zincir(0, k) > zincir(1, k)  # kesisimin saginda sirali bant 0 = zincir 1
    for h in (1e-3, 1e-4, 1e-5):
        noktalar = np.array([[k - h, 0, 0], [k, 0, 0], [k + h, 0, 0]])
        E = band_structure(model, noktalar)[:, 0]
        sirali = h * h / (E[0] - 2 * E[1] + E[2])
        gercek = h * h / (zincir(0, k - h) - 2 * zincir(0, k) + zincir(0, k + h))
        assert sirali == pytest.approx(1.6735, abs=1e-3)
        assert gercek == pytest.approx(-0.6276, abs=1e-3)


def test_gecersiz_girdiler_reddediliyor():
    model = _zincir()
    with pytest.raises(InvalidInputError, match="n en az"):
        uniform_k_grid(model.lattice, 1)
    with pytest.raises(InvalidInputError, match="bant indisi"):
        band_edges(model, band_index=7)
    with pytest.raises(InvalidInputError, match="en az iki"):
        k_path((np.zeros(3),), n_per_segment=5)


def _tek_orbital_uclar(model: TBModel, n: int = 401, aday: int = 10) -> tuple[float, float]:
    """BAGIMSIZ referans (eigvalsh'siz): tek orbitalde E(f) = eps + sum t
    cos(2 pi f . c). n^d yogun izgara + en iyi `aday` yerel uctan kompas
    daraltmasi (adim 1/n -> 1e-16), vektorel."""
    from itertools import product

    d = model.lattice.dimension
    C = np.array([h.cell[:d] for h in model.hoppings], dtype=float)
    t = np.array([h.t for h in model.hoppings])

    def E(F):
        return model.onsite[0] + np.cos(2 * np.pi * F @ C.T) @ t

    g = np.arange(n) / n
    F = np.array(np.meshgrid(*([g] * d), indexing="ij")).reshape(d, -1).T
    deger = E(F)
    yonler = np.array([v for v in product((-1, 0, 1), repeat=d) if any(v)], dtype=float)
    uclar = []
    for isaret in (1.0, -1.0):
        s = isaret * deger
        kup = s.reshape([n] * d)
        yerel = np.ones_like(kup, dtype=bool)
        for kay in product((-1, 0, 1), repeat=d):
            if any(kay):
                yerel &= kup <= np.roll(kup, kay, axis=tuple(range(d)))
        idx = np.nonzero(yerel.ravel())[0]
        en_iyi = float(s.min())
        for i in idx[np.argsort(s[idx])][:aday]:
            f, e, h = F[i].copy(), float(s[i]), 1.0 / n
            while h > 1e-16:
                ey = isaret * E(np.mod(f + yonler * h, 1.0))
                j = int(np.argmin(ey))
                if ey[j] < e:
                    e, f = float(ey[j]), np.mod(f + yonler[j] * h, 1.0)
                else:
                    h /= 2.0
            en_iyi = min(en_iyi, e)
        uclar.append(isaret * en_iyi)
    return uclar[0], uclar[1]


def _uzun_menzil_2B(indis: int) -> TBModel:
    """Egik 2B, |cell| <= 4 butun hucreler, t ~ N(0, 0.05/|c|); default_rng(31)
    akisinin `indis`. modeli (on olcum E2 suphecisinin kumesi)."""
    from itertools import product

    rng = np.random.default_rng(31)
    orgu = Lattice(vectors=np.array([[2.5, 0.0, 0.0], [0.3, 2.4, 0.0]]), basis=np.zeros((1, 3)))
    hucreler = [c for c in product(range(-4, 5), repeat=2) if c > (0, 0)]
    for _ in range(indis + 1):
        ciftler = [(c, rng.normal(0.0, 0.05 / np.hypot(*c))) for c in hucreler]
    hop = []
    for (a, b), t in ciftler:
        hop += [Hopping(0, 0, (a, b, 0), t), Hopping(0, 0, (-a, -b, 0), t)]
    return TBModel(lattice=orgu, onsite=np.array([0.0]), hoppings=tuple(hop))


def _goreli_hatalar(model: TBModel, bulunan: tuple[float, float]) -> tuple[float, float]:
    alt_ref, ust_ref = _tek_orbital_uclar(model)
    W = ust_ref - alt_ref
    return (bulunan[0] - alt_ref) / W, (ust_ref - bulunan[1]) / W


def test_band_edges_2B_TRIM_minimumu_ANALITIK():
    """E2 (04.10.2026): eski eksen-eksen inisin tek baslangicla dustugu
    tuzak (bant genisliginin %30'u) bu modelde. Minimum TRIM noktasi
    (0, 1/2)'de ve ANALITIK: 2tx - 2ty - 4td + 2t2x + 2t2y = -0.323908
    (yuvarlanmis parametrelerde tam). Eski docstring'in "band_edges 201x201
    izgaradan daha dogru" yan olcumu parite tuzaginin yan urunuydu: (0, 1/2)
    her CIFT izgarada vardir, 201 tektir. Gamma merkezli tohum (0, 1/2)'yi
    ICERIR: minimum dogrudan tohumda bulunur (bu test kompasi sinamaz).
    """
    tx, ty, td, t2x, t2y = (-0.040377, -0.001626, 0.044219, -0.029180, -0.005585)
    orgu = Lattice(vectors=np.array([[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]), basis=np.zeros((1, 3)))
    model = TBModel(
        lattice=orgu,
        onsite=np.array([0.0]),
        hoppings=tuple(
            Hopping(i=0, j=0, cell=c, t=t)
            for c, t in (
                ((1, 0, 0), tx), ((-1, 0, 0), tx), ((0, 1, 0), ty), ((0, -1, 0), ty),
                ((1, 1, 0), td), ((-1, -1, 0), td), ((1, -1, 0), td), ((-1, 1, 0), td),
                ((2, 0, 0), t2x), ((-2, 0, 0), t2x), ((0, 2, 0), t2y), ((0, -2, 0), t2y),
            )
        ),
    )
    alt, ust = band_edges(model, band_index=0)
    analitik = 2 * tx - 2 * ty - 4 * td + 2 * t2x + 2 * t2y
    assert analitik == pytest.approx(-0.323908, abs=1e-15)
    assert abs(alt - analitik) < 1e-15
    assert _goreli_hatalar(model, (alt, ust))[1] < 1e-12


def test_band_edges_GRAFEN_ANALITIK_UCLAR_kompas_gerekli():
    """Modul dokstring'inin kendi gerekce ornegi. Onsite 0'da bant 0 =
    [-3|t|, 0], bant 1 = [0, 3|t|] -- iki ic uc Dirac konisinin TEPESINDE
    (kink). Eski surum bant 0 max'ini %1.85 kaciriyordu (sahte 0.6 eV
    aralik); kompassiz L-BFGS-B koni tepesinde 1.6e-9 bagil durur. Kompasli
    arama yuvarlama duzeyinde (5.8e-17) bulur.
    """
    t = -2.7 / HARTREE_TO_EV
    orgu = Lattice(
        vectors=np.array([[4.65, 0.0, 0.0], [2.325, 4.027018, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [2.325, 1.342339, 0.0]]),
    )
    hop = (
        Hopping(0, 1, (0, 0, 0), t), Hopping(0, 1, (-1, 0, 0), t), Hopping(0, 1, (0, -1, 0), t),
        Hopping(1, 0, (0, 0, 0), t), Hopping(1, 0, (1, 0, 0), t), Hopping(1, 0, (0, 1, 0), t),
    )
    grafen = TBModel(lattice=orgu, onsite=np.zeros(2), hoppings=hop)
    W = 6 * abs(t)
    alt0, ust0 = band_edges(grafen, 0)
    alt1, ust1 = band_edges(grafen, 1)
    for bulunan, gercek in ((alt0, 3 * t), (ust0, 0.0), (alt1, 0.0), (ust1, -3 * t)):
        assert abs(bulunan - gercek) <= 1e-12 * W, (bulunan, gercek)
    # Aralik HER ZAMAN icerde: bant 0 max <= 0 <= bant 1 min (yuvarlama payiyla)
    assert ust0 <= 1e-15 and alt1 >= -1e-15


def test_band_edges_UZUN_MENZIL_egik_2B():
    """Egik 2B, |cell| <= 4, t ~ 1/|c|. Eski surum bu kumede 39/40 arama
    basarisizdi (en kotu %47); on olcumdeki sabit 8 noktali, esiksiz
    kompasli Gamma tohumu bu iki modelde %3.06 ve %1.74 kaciriyordu.
    Model 19'u TEK adayli arama %0.26 kaciriyor (3 aday sart: tek adayla
    bu kumede 2/40, PT kumesinde 2/40 -- %5). Menzille olceklenen tohum +
    esikli kompas + 3 aday: yuvarlama duzeyi. Referans
    eigvalsh'SIZ ve bagimsiz (`_tek_orbital_uclar`).
    """
    for indis in (3, 6, 19):
        model = _uzun_menzil_2B(indis)
        hatalar = _goreli_hatalar(model, band_edges(model, 0))
        assert max(hatalar) < 1e-12, (indis, hatalar)
        assert min(hatalar) > -1e-12, (indis, hatalar)  # aralik icerde


def test_band_edges_KOMPAS_GURULTU_ESIGI_tavana_dayanmaz(monkeypatch):
    """Gerilimli + aralikli grafen (t, 1.05 t, 0.97 t; Delta = 1e-3 |t|), bant
    1 -- on olcumun "aralik0.001" vakasi. Esiksiz kompas dar ve duzgun
    vadide yuvarlama gurultusunu iyilesme sayip zikzak ciziyor ve 10 000
    adimlik tavana dayaniyordu: 81 666 eigvalsh cagrisi (aileye toplam 39 s).
    "Iyilesme 16 eps * H olcegini asmali" kurali ile 1 882 cagri -- sonuc
    ayni. Cagri sayisi deterministiktir; sinir genis tutuldu. Bant 1'in
    minimumu ANALITIK: Delta / 2 (uc t'den biri otekilerin toplamini asmaz).
    """
    import qlab.engine.tb.bands as bands_modulu

    t = -2.7 / HARTREE_TO_EV
    delta = 1e-3 * abs(t)
    a = 2.68 * np.sqrt(3.0)
    v = np.array([[a, 0.0, 0.0], [0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0]])
    orgu = Lattice(vectors=v, basis=np.array([[0.0, 0.0, 0.0], (v[0] + v[1]) / 3.0]))
    hop = []
    for c, tt in (((0, 0, 0), t), ((-1, 0, 0), 1.05 * t), ((0, -1, 0), 0.97 * t)):
        hop += [Hopping(0, 1, c, tt), Hopping(1, 0, tuple(-x for x in c), tt)]
    model = TBModel(lattice=orgu, onsite=np.array([delta / 2, -delta / 2]), hoppings=tuple(hop))
    asil = np.linalg.eigvalsh
    sayac = [0]

    def sayan(H):
        sayac[0] += 1
        return asil(H)

    monkeypatch.setattr(bands_modulu.np.linalg, "eigvalsh", sayan)
    alt, _ = band_edges(model, 1)
    assert sayac[0] < 10_000, sayac[0]
    # Esik bedeli: hassasiyet 1e-16 yerine ~1e-14 bagil (bant genisligi ~0.3).
    assert alt == pytest.approx(delta / 2, abs=1e-13)


def _puruzlu_1B(indis: int) -> TBModel:
    """ESIT genlikli uzun menzil: |cell| <= 6, her hucre %60 olasilikla
    t ~ N(0, 0.05); default_rng(4410) akisinin `indis`. modeli (E2 son
    olcumunun puruzlu 1B kumesi). Gercekci TB'de hoplama mesafeyle soner;
    bu kume bilerek ZOR kurulmustur."""
    rng = np.random.default_rng(4410)
    for _ in range(indis + 1):
        hop = []
        for c in range(1, 7):
            t = rng.normal(0.0, 0.05) * (rng.random() < 0.6)
            hop += [Hopping(0, 0, (c, 0, 0), t), Hopping(0, 0, (-c, 0, 0), t)]
    orgu = Lattice(vectors=np.array([[2.5, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    return TBModel(lattice=orgu, onsite=np.zeros(1), hoppings=tuple(hop))


def test_band_edges_PURUZLU_bantta_tohum_MENZILLE_olceklenir():
    """Gamma merkezli SABIT 8 noktali tohum bu modelde bant genisliginin
    %39.5'ini kaciriyor: R = 6'da havza basina 3'ten az nokta. Tohum eksen
    basina 8 R_a'ya olceklenince yuvarlama duzeyi. Olculdu (puruzlu 1B, 149
    duz olmayan model): sabit 8 ile 22 model basarisiz, 8 R ile 0.
    """
    model = _puruzlu_1B(94)
    hatalar = _goreli_hatalar(model, band_edges(model, 0))
    assert max(hatalar) < 1e-12 and min(hatalar) > -1e-12, hatalar


def test_band_edges_DAR_TEPE_gamma_tohumu_KAYDIRMASIZ():
    """Puruzlu kumenin bu modelinde (yalniz 4. ve 5. hucre) dar bir tepe
    var. On olcumdeki 0.381966 adim KAYDIRILMIS tohum onu iki tohum
    noktasinin arasinda birakip maksimumu bant genisliginin %1.18'i kadar
    kaciriyordu (30 adayla bile); Gamma merkezli tohum bulur. Kaydirma
    yeniden eklenirse bu test kirilir. Yine de `band_edges` kuresel ucu
    KANITLAMAZ -- kesin olan yalniz yondur (aralik icerde).
    """
    model = _puruzlu_1B(87)
    hatalar = _goreli_hatalar(model, band_edges(model, 0))
    assert max(hatalar) < 1e-12 and min(hatalar) > -1e-12, hatalar


def test_band_edges_KOTU_KOSULLU_TABANDA_aralik_ICERDE():
    """E2 dogrulama turu: indirgenmemis taban a2 = (N a, a) (ayni kare orgu,
    17 derece dondurulmus). k . R ile kurulan H'de faz buyuk terimlerin
    birbirini goturmesiyle hesaplaniyordu; N = 1000'de (kosul 1e6) donen
    minimum gercek minimumun 5.8e-11 W ALTINA iniyordu -- "aralik icerde"
    garantisi bozuluyordu. H artik kesirli koordinatta kurulur (faz 2 pi f.c):
    garanti kosuldan bagimsiz. Referans (`_tek_orbital_uclar`) kesirli
    koordinatta ve orguden bagimsizdir.
    """
    aci = np.deg2rad(17.0)
    donme = np.array([[np.cos(aci), -np.sin(aci), 0], [np.sin(aci), np.cos(aci), 0], [0, 0, 1]])
    liste = [((1, 0), -0.05), ((0, 1), -0.03), ((1, 1), 0.02), ((2, -1), 0.011), ((1, -2), -0.007)]
    hop = []
    for (a, b), t in liste:
        hop += [Hopping(0, 0, (a, b, 0), t), Hopping(0, 0, (-a, -b, 0), t)]
    for N in (0, 1000, 30000):
        vektorler = np.array([[2.5, 0.0, 0.0], [2.5 * N, 2.5, 0.0]]) @ donme.T
        model = TBModel(
            lattice=Lattice(vectors=vektorler, basis=np.zeros((1, 3))),
            onsite=np.array([0.0]),
            hoppings=tuple(hop),
        )
        hatalar = _goreli_hatalar(model, band_edges(model, 0))
        assert min(hatalar) > -1e-14 and max(hatalar) < 1e-12, (N, hatalar)


def _kesisim_2B(indis: int):
    """Iki ESLESMESIZ orbital, egik 2B: bant 0 = min(eA, eB). default_rng(11)
    akisinin `indis`. modeli (E2 dogrulama turu, s2_kesisim_sirti.py)."""
    orgu = Lattice(
        vectors=np.array([[2.5, 0, 0], [0.4, 2.3, 0]]),
        basis=np.array([[0.0, 0, 0], [1.2, 1.1, 0]]),
    )
    hucreler = [(1, 0), (0, 1), (1, 1), (1, -1)]
    rng = np.random.default_rng(11)
    for _ in range(indis + 1):
        pa = rng.normal(0, 0.05, len(hucreler))
        pb = rng.normal(0, 0.05, len(hucreler))
        eb = rng.normal(0, 0.05)
    hop = []
    for c, ta, tb in zip(hucreler, pa, pb):
        c3, m3 = (c[0], c[1], 0), (-c[0], -c[1], 0)
        hop += [Hopping(0, 0, c3, ta), Hopping(0, 0, m3, ta), Hopping(1, 1, c3, tb), Hopping(1, 1, m3, tb)]
    return TBModel(lattice=orgu, onsite=np.array([0.0, eb]), hoppings=tuple(hop))


def test_band_edges_DURUST_OLUMSUZ_KESISIM_DIKISI_ic_kenar_kacar():
    """DURUST OLUMSUZ, testle kilitli: 2B'de iki bandin kesistigi DIKIS
    uzerindeki uc (bant 0 = min(eA, eB)'nin maksimumu) dar bir V tepesidir
    ve tohum izgarasinda yerel uc olarak gorunmez; aday ya da kompas yonu
    artirmak cozmuyor (olculdu). Bu modelde bant 0'in maksimumu 0.015
    hartree KACIRILIYOR (tanik nokta f = (0.85, 0.71125)). Kesin olan tek
    sey yon: donen max tanigin ALTINDA, yani aralik icerde. Bu sinif
    duzeltilirse bu test kirilir -- o zaman dokstring ve README guncellenir.
    """
    import qlab.engine.tb.bands as bands_modulu

    model = _kesisim_2B(15)
    tanik = float(np.linalg.eigvalsh(bands_modulu._kesirli_hamiltonyen(model)(np.array([0.85, 0.71125])))[0])
    _, ust = band_edges(model, 0)
    assert tanik - ust > 0.01, (tanik, ust)

@pytest.mark.slow
def test_band_edges_ORIJINAL_KUME_50_ARAMA():
    """README'nin "tek baslangic 31/50, 5x5 8/50" olcumunun kumesi:
    default_rng(7).normal(0, 0.05, (25, 5)), 25 model x 2 uc. Eski (8x8
    baslangicli) surum 1e-9 esiginde 14/50 basarisizdi (en kotu %39.16);
    yeni surum 0/50, ve her aramada donen deger gercek bandin ICINDE.
    """
    orgu = Lattice(vectors=np.array([[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]), basis=np.zeros((1, 3)))
    P = np.random.default_rng(7).normal(0.0, 0.05, (25, 5))
    for i, (tx, ty, td, t2x, t2y) in enumerate(P):
        hop = []
        for c, t in (((1, 0, 0), tx), ((0, 1, 0), ty), ((1, 1, 0), td), ((1, -1, 0), td),
                     ((2, 0, 0), t2x), ((0, 2, 0), t2y)):
            hop += [Hopping(0, 0, c, t), Hopping(0, 0, tuple(-x for x in c), t)]
        model = TBModel(lattice=orgu, onsite=np.array([0.0]), hoppings=tuple(hop))
        hatalar = _goreli_hatalar(model, band_edges(model, 0))
        assert -1e-12 < min(hatalar) and max(hatalar) < 1e-12, (i, hatalar)


def _rastgele_2B(tohum: int) -> TBModel:
    """Iki orbitalli 2B model, hucreler |c| <= 2 (KOSEGEN hucreler dahil --
    ||cell||_1 ile ||cell||_inf'i ayirmak icin), Hermityen kurulur."""
    rng = np.random.default_rng(tohum)
    orgu = Lattice(
        vectors=np.array([[2.5, 0.0, 0.0], [0.8, 2.2, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [1.1, 0.7, 0.0]]),
    )
    hop = []
    for hucre in ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (2, -1), (1, 2)):
        for i in range(2):
            for j in range(2):
                if hucre == (0, 0) and i >= j:
                    continue
                t = float(rng.normal(0.0, 0.05))
                c = (hucre[0], hucre[1], 0)
                hop.append(Hopping(i, j, c, t))
                hop.append(Hopping(j, i, (-c[0], -c[1], 0), t))
    return TBModel(lattice=orgu, onsite=rng.normal(0.0, 0.1, 2), hoppings=tuple(hop))


def test_B1_B2_KANITLI_SINIRLAR_rastgele_2B_modellerde_TUTUYOR():
    """E13/E1 (04.10.2026): B1 her sirali ozdegerin izgara ile surekli k
    arasindaki farkini, B2 global uclarin kacirilma miktarini sinirlar.

    Rastgele k noktasi ile ona EN YAKIN izgara noktasi (kesirli koordinatta
    yuvarlama) karsilastirilir: |lambda_b(k) - lambda_b(k_g)| <= B1, ve
    izgara_min - orneklem_min <= B2 (ust uc icin simetrik). Bu modellerde
    oranlar GEVSEK (B1 en cok 0.42, B2 0.12): test yalniz kaba (>~2.4x)
    ihlalleri yakalar. Norm ve sabit kilidi `test_B1_B2_SIKI_ve_NORMU_dogru`da
    (dogrulama turunda ||cell||_inf ve yari sabit mutantlari burada yesildi).
    """
    n = 6
    for tohum in range(4):
        model = _rastgele_2B(tohum)
        b = reciprocal_vectors(model.lattice)
        B1 = grid_lipschitz_bound(model, n)
        B2 = grid_extremum_bound(model, n)
        izgara = band_structure(model, uniform_k_grid(model.lattice, n))
        kesirli = np.random.default_rng(100 + tohum).random((1000, 2))
        en_yakin = np.rint(kesirli * n) / n
        surekli = band_structure(model, kesirli @ b[:2])
        komsu = band_structure(model, en_yakin @ b[:2])
        assert np.abs(surekli - komsu).max() <= B1
        assert float(izgara.min()) - float(surekli.min()) <= B2
        assert float(surekli.max()) - float(izgara.max()) <= B2


def _kare_capraz(hucreler) -> TBModel:
    orgu = Lattice(vectors=np.array([[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]), basis=np.zeros((1, 3)))
    hop = tuple(Hopping(0, 0, (a, b, 0), -0.0625) for a, b in hucreler)
    return TBModel(lattice=orgu, onsite=np.array([0.0]), hoppings=hop)


def test_B1_B2_SIKI_ve_NORMU_dogru():
    """Sinirlarin GEVSEK ya da YANLIS normlu olmadigini kilitler.

    (1) Zincir, E = 2t cos(ka): ka = pi/2'deki izgara noktasi ile iki izgara
        noktasinin TAM ortasi arasindaki fark B1 = (pi/n) 2|t|'ye 1 - O(1/n^2)
        oranla yaklasir.
    (2) Yalniz +-(1,1) hucreli 2B model: faz 2 pi (f1 + f2), orta nokta fazi
        2 pi / n kaydirir -- B1'deki ||cell||_1 = 2 SIKI. ||cell||_inf (= 1)
        kullanan bir sinir burada 2 kat IHLAL edilir.
    (3) +-(1,+-1) hucreli 2B model, E = 4t cos(2 pi f1) cos(2 pi f2), tek n:
        maksimum (1/2, 0)'da ve izgarada yok (en yakin nokta bir eksende
        1/(2n) uzakta); gercek hata 4|t|(1 - cos(pi/n)), B2 = pi^2/(2 n^2)
        * 16|t| -> oran 1/4'e gider. B2'nin ||cell||_1^2 ile kuruldugu
        formulun kendisiyle de kilitlenir (||cell||_inf bunu 4'e boler).
    """
    n = 400
    orgu = Lattice(vectors=np.array([[2.5, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    zincir = TBModel(
        lattice=orgu, onsite=np.array([0.0]),
        hoppings=(Hopping(0, 0, (1, 0, 0), -0.0625), Hopping(0, 0, (-1, 0, 0), -0.0625)),
    )
    b = reciprocal_vectors(orgu)[0]
    kg = (n // 4) / n
    sapma = abs(
        band_structure(zincir, np.array([(kg + 0.5 / n) * b]))[0, 0]
        - band_structure(zincir, np.array([kg * b]))[0, 0]
    )
    assert 0.99 < sapma / grid_lipschitz_bound(zincir, n) <= 1.0

    capraz = _kare_capraz(((1, 1), (-1, -1)))
    b2 = reciprocal_vectors(capraz.lattice)[:2]
    fg = np.array([n // 8, n // 8]) / n  # 2 pi (f1 + f2) = pi/2
    sapma = abs(
        band_structure(capraz, np.array([(fg + 0.5 / n) @ b2]))[0, 0]
        - band_structure(capraz, np.array([fg @ b2]))[0, 0]
    )
    assert 0.99 < sapma / grid_lipschitz_bound(capraz, n) <= 1.0

    n_tek = 101
    carpim = _kare_capraz(((1, 1), (-1, -1), (1, -1), (-1, 1)))
    B2 = grid_extremum_bound(carpim, n_tek)
    assert B2 == pytest.approx(np.pi**2 / (2 * n_tek**2) * 16 * 0.0625, rel=1e-14)
    izgara = band_structure(carpim, uniform_k_grid(carpim.lattice, n_tek))
    assert float(izgara.min()) == pytest.approx(-0.25, abs=1e-15)  # Gamma izgarada
    oran = (0.25 - float(izgara.max())) / B2  # maksimum (1/2, 0)'da
    assert 0.249 < oran <= 0.25


def test_B1_ALT_NORMAL_t_de_SIFIRA_DUSMEZ_ve_H_OLCEGI_formulu():
    """Dogrulama turu (04.10.2026): t = 5e-324 hartree'de B1 = (pi/n) * t
    sifira yuvarlaniyordu ve `certified_conduction` "H(k) sabit" dalina
    (kanitli yalitkan) dusuyordu. Hoplama varken B1 > 0 olmali; hoplama
    yokken (ya da hepsi R = 0) TAM 0.

    `hamiltonian_rounding_scale` = max_i(|eps_i| + sum |t| (1 + 2 pi ||cell||_1))
    -- yuvarlama payina giren kurulus olcegi; formul burada kilitlenir.
    """
    orgu = Lattice(vectors=np.array([[2.5, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    alt_normal = TBModel(
        lattice=orgu, onsite=np.array([1.0]),
        hoppings=(Hopping(0, 0, (1, 0, 0), 5e-324), Hopping(0, 0, (-1, 0, 0), 5e-324)),
    )
    assert grid_lipschitz_bound(alt_normal, 13) > 0.0
    sabit = TBModel(lattice=orgu, onsite=np.array([1.0]), hoppings=())
    assert grid_lipschitz_bound(sabit, 13) == 0.0
    t, eps = -0.0625, -0.1948
    zincir = TBModel(
        lattice=orgu, onsite=np.array([eps]),
        hoppings=(Hopping(0, 0, (1, 0, 0), t), Hopping(0, 0, (-1, 0, 0), t)),
    )
    beklenen = abs(eps) + 2 * abs(t) * (1 + 2 * np.pi)
    assert hamiltonian_rounding_scale(zincir) == pytest.approx(beklenen, rel=1e-15)


def test_band_structure_CAGIRANLARI_KILITLI():
    """E3 (04.10.2026): `band_structure` sutunlari ve `build_band_payload`un
    bant ekseni SIRALAMADIR. `src/` icinde bu iki ada yapilan her REFERANS
    (cagri degil -- `import ... as` takma adi, modul duzeyi, lambda,
    getattr("...") dizgesi, partial dahil) cevreleyen fonksiyonuyla bu
    listede kilitlidir. Bugun ikisini de yalniz `run._run_tb` kullanir;
    sutunlari yalniz `certified_conduction` ve `fermi_level`e (doluluk --
    dogru tanim siralamadir) ve npz'ye (cizim) verir.

    KAPSAM DURUSTCE: dogrudan `np.linalg.eigvalsh(bloch_hamiltonian(...))[b]`
    yazan bir yol bu kilidin DISINDADIR (dogrulama turu bunu gosterdi);
    kilit yalniz bu iki adi izler. Yeni bir referans testi kirar: once
    sutunlarin bant kimligi tasimadigini dusun (kesisimde etkin kutle isaret
    degistirir, bkz. `test_band_structure_SIRALAMA_BANT_KIMLIGI_DEGIL`),
    sonra listeye gerekcesiyle ekle.
    """
    import ast
    import pathlib

    izlenen = {"band_structure", "build_band_payload"}
    kok = pathlib.Path(__file__).resolve().parents[2] / "src"
    bulunan = set()
    for dosya in sorted(kok.rglob("*.py")):
        agac = ast.parse(dosya.read_text(encoding="utf-8"))
        tanim_yeri = dosya.relative_to(kok).as_posix()
        ebeveyn = {}
        for dugum in ast.walk(agac):
            for cocuk in ast.iter_child_nodes(dugum):
                ebeveyn[cocuk] = dugum

        def kapsayan(dugum):
            while dugum in ebeveyn:
                dugum = ebeveyn[dugum]
                if isinstance(dugum, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return getattr(dugum, "name", "<lambda>")
            return "<modul>"

        takma = set(izlenen)
        for dugum in ast.walk(agac):
            if isinstance(dugum, ast.ImportFrom):
                for ad in dugum.names:
                    if ad.name in izlenen:
                        takma.add(ad.asname or ad.name)
                        bulunan.add((tanim_yeri, kapsayan(dugum)))
        for dugum in ast.walk(agac):
            if isinstance(dugum, ast.Name) and dugum.id in takma:
                bulunan.add((tanim_yeri, kapsayan(dugum)))
            elif isinstance(dugum, ast.Attribute) and dugum.attr in izlenen:
                bulunan.add((tanim_yeri, kapsayan(dugum)))
            elif isinstance(dugum, ast.Constant) and dugum.value in izlenen:
                bulunan.add((tanim_yeri, kapsayan(dugum)))
    assert bulunan == {("qlab/run.py", "_run_tb")}, sorted(bulunan)
