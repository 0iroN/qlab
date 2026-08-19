"""Yogunluk farki haritasindan SKALER gozlemlenebilir.

Bu dosyanin sinadigi asil sey bir sayi degil, bir SECIM: hangi buyukluk bagin
zayiflamasini olcer. Olculdu -- `max|fark|` OLCEMEZ, bag orta noktasi olcer.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.scf import solve
from qlab.errors import InvalidInputError
from qlab.observable.hf_obs import (
    bond_midpoint_difference,
    density_difference,
    symmetric_box,
)
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM

H2_RE_BOHR = 1.3459193602


def _h2(r_bohr: float):
    """Cekirdekler z ekseninde ORIJINE SIMETRIK -- orta nokta (0,0,0)."""
    yari = r_bohr * BOHR_TO_ANGSTROM / 2
    return solve(
        Molecule.from_string(f"H 0 0 {-yari}; H 0 0 {yari}"),
        "sto-3g",
        stability=False,
    )


@pytest.mark.parametrize("n", [41, 61, 81])
def test_TEK_izgarada_deger_cozunurlukten_BAGIMSIZ(n):
    """Orta nokta gercek bir izgara noktasiysa deger interpolasyonsuzdur.

    OLCULDU: n = 41 / 61 / 81'in UCUNDE de 0.043821 -- alti hanede ayni.
    Bu, gozlemlenebilirin izgara cozunurlugunden bagimsiz oldugunun kaniti.
    """
    deger = bond_midpoint_difference(
        _h2(H2_RE_BOHR), symmetric_box(6.0, n), a=0, b=1
    )
    assert deger == pytest.approx(0.043821, abs=1e-6, rel=0.0)


def test_CIFT_izgara_REDDEDILIYOR():
    """Cift n'de orta nokta izgarada YOKTUR ve deger sessizce eksik cikar.

    OLCULDU: n = 40 / 60 / 80 -> 0.033640 / 0.038969 / 0.041031, yani
    n=40'ta gercek degerin (0.043821) **%23 ALTINDA**. Tehlikesi: bu hata
    bir YAKINSAMA SORUNU gibi gorunur ve insani izgarayi buyutmeye iter
    (maliyet n^3 ile artar), oysa dogru duzeltme n'i TEK yapmaktir ve
    bedavadir. Bu yuzden sessizce kabul edilmez, hata firlatilir.
    """
    with pytest.raises(InvalidInputError, match="TEK"):
        symmetric_box(6.0, 80)


def test_max_fark_bag_gozlemlenebiliri_DEGILDIR():
    """DURUST OLUMSUZ, olcularak kilitleniyor.

    `max|fark|` bagin zayiflamasini olcemez -- yalnizca duyarsiz degil,
    fiziksel olarak ilginc aralikta TERS YONE gider. OLCULDU, bu testin
    kullandigi kutuda (yari genislik 6.0, n=61):
      r = 1.3459 -> max 0.064559, orta-nokta 0.043821
      r = 1.8    -> max 0.073424 (YUKSELDI), orta-nokta 0.039467 (dustu)
    Sebep: mutlak maksimum cekirdek yakinindaki keskin yapidadir ve bag
    bolgesini hic gormez.

    IZGARA HER SAYIYI DEGISTIRIR, ve bu da basli basina bir kanit.
    Ayni iki mesafede `max|fark|` uc cozunurlukte:
      n=41 -> 0.061516 / 0.069238
      n=61 -> 0.064559 / 0.073424
      n=81 -> 0.063294 / 0.072512
    Yalnizca yon degil DEGER de oynuyor, ustelik n ile tek yonlu bile
    degil. Orta nokta ise ucunde de 0.043821 / 0.039467. (Bu dokstring
    bir sure n=81'in sayilarini tasiyordu, oysa test n=61'de kosuyor --
    tam da burada uyarilan hatanin kendisi.

    Bu test o gercegi kilitler: biri gozlemlenebiri `max`'a cevirirse
    kirmizi olur.
    """
    kutu = symmetric_box(6.0, 61)
    denge, gerilmis = _h2(H2_RE_BOHR), _h2(1.8)

    orta_denge = bond_midpoint_difference(denge, kutu, a=0, b=1)
    orta_gerilmis = bond_midpoint_difference(gerilmis, kutu, a=0, b=1)
    max_denge = float(np.abs(density_difference(denge, kutu).values).max())
    max_gerilmis = float(np.abs(density_difference(gerilmis, kutu).values).max())

    # orta nokta DUSUYOR (bag zayifliyor)
    assert orta_gerilmis < orta_denge
    # max ise YUKSELIYOR -- ters yon
    assert max_gerilmis > max_denge


def test_gerilme_serisi_TEK_YONLU_cokuyor():
    """Gozlemlenebilirin asil sozlesmesi: monoton olmak.

    OLCULDU (n=81): 1.3459 -> 0.043821, 1.8 -> 0.039467, 2.5 -> 0.028001,
    4.0 -> 0.006847, 10.0 -> 0.000003. Bes mertebe, tek yonlu.

    `@pytest.mark.slow` DEGIL, bilerek: olculdu, bu test 0.35 s suruyor
    (dosyanin tamami 1.07 s). Kural "saniye alti cipalar varsayilan
    seritte kalir" oldugu icin buraya aittir -- ve zaten modulun ASIL
    iddiasini kilitleyen test budur, gunluk donguden surulmesi yanlis
    olurdu. H2/sto-3g'de baz iki fonksiyondan ibarettir; n=81 izgarasi
    (531441 nokta) pahali degildir.
    """
    kutu = symmetric_box(6.0, 81)
    seri = [
        bond_midpoint_difference(_h2(r), kutu, a=0, b=1)
        for r in (H2_RE_BOHR, 1.8, 2.5, 4.0)
    ]
    assert seri == sorted(seri, reverse=True)
    assert seri[0] == pytest.approx(0.043821, abs=1e-6, rel=0.0)
    assert seri[-1] == pytest.approx(0.006847, abs=1e-5, rel=0.0)


def test_gecersiz_atom_indisi_reddediliyor():
    with pytest.raises(InvalidInputError, match="atom"):
        bond_midpoint_difference(_h2(H2_RE_BOHR), symmetric_box(6.0, 41), a=0, b=5)


def test_ASIMETRIK_bagda_orta_nokta_max_DEGILDIR():
    """`max` korlugunun ASIL kaniti -- H2 bu ayrimi yapamaz.

    OLCULDU: simetrik H2'de isaretli maksimum TAM OLARAK bag orta
    noktasindadir (n=61'de argmax indisi (30,30,30), fark 0.00e+00, dort
    germe mesafesinin dordunde de). Yani H2 uzerinde `values.max()` ile
    orta nokta ayirt EDILEMEZ; yalnizca H2'ye bakan bir test suru bu
    secimi kilitlemez (olculdu: `max` mutanti H2 testlerinin 7'sinden de
    gecti).

    HF molekulu ayirir ve ayrimi sert yapar:
      orta nokta   = -0.194841  (NEGATIF -- iyonik bagda orta nokta
                                 ZAYIFLAR, yogunluk F'ye kacar)
      values.max() = +0.235735  (POZITIF -- isareti bile ters)

    Ikinci fark daha da belirleyici: orta nokta n = 41/61/81'in ucunde de
    -0.194841 (alti hane sabit), `max` ise cozunurlukle OYNAR (0.247336 /
    0.235735 / 0.247336) cunku cekirdek sivriligini kovalar ve o sivrilik
    izgara inceldikce sinirsiz keskinlesir. Bir gozlemlenebilirin
    cozunurluge bagli olmasi onu gozlemlenebilir olmaktan cikarir.
    """
    hf = solve(
        Molecule.from_string("H 0 0 -0.4600; F 0 0 0.4600"), "sto-3g", stability=False
    )
    degerler = [
        bond_midpoint_difference(hf, symmetric_box(6.0, n), a=0, b=1)
        for n in (41, 61, 81)
    ]
    for deger in degerler:
        assert deger == pytest.approx(-0.194841, abs=1e-6, rel=0.0)

    alan = density_difference(hf, symmetric_box(6.0, 61))
    assert degerler[1] < 0.0 < float(alan.values.max())


def test_orta_nokta_EN_YAKIN_dugume_yuvarlanir():
    """Yuvarlama `np.rint` olmali; asagi kesme (`astype(int)`) yetmez.

    OLCULDU: 100 tek n x 11 yari-genislik = 1100 kutunun 27'sinde
    (0 - origin) / spacing tam sayinin bir ulp ALTINDA cikiyor, ornegin
    `symmetric_box(7.0, 51)` -> 24.999999999999996. Asagi kesme orada
    indisi 24'e dusurur, izgara-ustunde-mi muhafazasi tetiklenir ve
    fonksiyon KUSURSUZ bir kutuyu reddeder. Yani hata sessiz degil,
    yanlis alarm -- ama yine de hata.

    Bonus olcum: deger burada da 0.043821 cikiyor, yani gozlemlenebilir
    yalnizca cozunurlukten degil KUTU BOYUTUNDAN da bagimsiz (6.0/41,
    6.0/61, 6.0/81 ve 7.0/51 -> hepsi 0.043821).
    """
    kutu = symmetric_box(7.0, 51)
    ham = (0.0 - kutu.origin[2]) / kutu.spacing[2]
    assert int(ham) == 24 and int(np.rint(ham)) == 25  # tuzagin hala orada oldugu

    deger = bond_midpoint_difference(_h2(H2_RE_BOHR), kutu, a=0, b=1)
    assert deger == pytest.approx(0.043821, abs=1e-6, rel=0.0)


def test_izgara_disi_orta_nokta_REDDEDILIYOR():
    """Orta nokta bir izgara dugumu degilse SESSIZCE komsu deger donmez.

    Cekirdekleri 0.07 bohr kaydirmak yeter: kutu araligi 0.2 bohr
    oldugundan orta nokta hucrenin ucte birinde kalir. Interpolasyonsuz
    okuma orada yanlistir ve fonksiyon bunu bildirmek zorundadir --
    aksi halde deger 0.35 hucre otedeki komsudan gelir ve hicbir sey
    bunu haber vermez.
    """
    kaydirma_angstrom = 0.07 * BOHR_TO_ANGSTROM
    yari = H2_RE_BOHR * BOHR_TO_ANGSTROM / 2
    kaymis = solve(
        Molecule.from_string(
            f"H 0 0 {-yari + kaydirma_angstrom}; H 0 0 {yari + kaydirma_angstrom}"
        ),
        "sto-3g",
        stability=False,
    )
    with pytest.raises(InvalidInputError, match="izgarada degil"):
        bond_midpoint_difference(kaymis, symmetric_box(6.0, 61), a=0, b=1)
