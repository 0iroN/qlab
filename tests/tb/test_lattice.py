"""Orgu geometrisi -- iki olculmus sessiz hata burada kilitleniyor.

(1) Ters orgude 2*pi carpani. (2) Kesme yaricapinda kayan nokta: `d <= cutoff`
yazmak, cutoff tam komsu mesafesine esitken komsulari SESSIZCE atiyor.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.tb.lattice import (
    Lattice,
    neighbour_shells,
    reciprocal_vectors,
)
from qlab.errors import InvalidInputError


def _fcc(a: float) -> Lattice:
    """Yuzey merkezli kubik, tek atomlu taban. En yakin komsu sayisi 12."""
    return Lattice(
        vectors=np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]),
        basis=np.zeros((1, 3)),
    )


def test_ters_orgu_2pi_konvansiyonu():
    """b_i . a_j = 2*pi*delta_ij -- delta_ij DEGIL.

    OLCULDU: yanlis konvansiyon grafende K noktasindaki araligi 0 yerine
    15.605535020 eV yapiyor. Sinsiligi: Gamma noktasinda yapilan kontrol
    IKI konvansiyonda da geciyor (E(Gamma) = 8.100000000), yani Gamma'da
    dogrulama yapan biri hatayi HIC gormez.
    """
    orgu = _fcc(6.83)
    b = reciprocal_vectors(orgu)
    carpim = b @ orgu.vectors.T
    assert np.allclose(carpim, 2.0 * np.pi * np.eye(3), atol=1e-12)


def test_fcc_en_yakin_komsu_sayisi_12():
    orgu = _fcc(6.83)
    a = 6.83
    ciftler = neighbour_shells(orgu, cutoff_bohr=a / np.sqrt(2.0))
    assert len(ciftler) == 12
    mesafeler = np.array([c.distance for c in ciftler])
    assert np.allclose(mesafeler, a / np.sqrt(2.0), rtol=1e-12)


@pytest.mark.parametrize("a", [1.0, 2.0, 3.0, 4.0, 5.0, 6.83, 7.0, 8.0, 9.0, 10.0, 11.0])
def test_kesme_yaricapi_KAYAN_NOKTA_TUZAGI(a):
    """`d <= cutoff` ciplak yazilirsa komsular SESSIZCE dusuyor.

    OLCULDU: 11 orgu sabitinin 5'inde `d <= a/sqrt(2)` 12 yerine SIFIR komsu
    seciyor, cunku hesaplanan mesafe kesme degerinin bir-iki ulp ustune
    dusuyor. Hatanin sinsiligi: eksik komsu bant genisligini bozar ama
    E(k) = E(-k) simetri testi yine 0.000e+00 verir (olculdu) -- yani
    simetri testleri bu hataya KORDUR.
    """
    orgu = _fcc(a)
    ciftler = neighbour_shells(orgu, cutoff_bohr=a / np.sqrt(2.0))
    assert len(ciftler) == 12, f"a={a} icin {len(ciftler)} komsu bulundu"


def test_kesme_yaricapi_bir_sonraki_kabugu_ALMIYOR():
    """Tolerans komsu bulmayi kolaylastirirken bir sonraki kabugu icine almamali.

    fcc'de ikinci kabuk a (=6.83), birincisi a/sqrt(2) (=4.829...). Aradaki
    bosluk 2.0 bohr; kullanilan bagil tolerans 1e-9 bunun yaninda yok
    hukmundedir, ve bu test onu kilitler.
    """
    orgu = _fcc(6.83)
    assert len(neighbour_shells(orgu, cutoff_bohr=6.83 / np.sqrt(2.0))) == 12
    assert len(neighbour_shells(orgu, cutoff_bohr=6.83)) == 12 + 6


def test_bir_boyutlu_zincir_iki_komsu():
    zincir = Lattice(vectors=np.array([[2.5, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    ciftler = neighbour_shells(zincir, cutoff_bohr=2.5)
    assert len(ciftler) == 2
    assert sorted(c.cell[0] for c in ciftler) == [-1, 1]


def test_iki_atomlu_taban_hucre_ICI_bagi_buluyor():
    """SSH zinciri: hucre ici bag (0,0,0) hucresinde, hucreler arasi (+-1,0,0)."""
    ssh = Lattice(
        vectors=np.array([[4.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]),
    )
    ciftler = neighbour_shells(ssh, cutoff_bohr=2.2)
    ic = [c for c in ciftler if c.cell == (0, 0, 0)]
    assert len(ic) == 2  # (0->1) ve (1->0)
    assert all(abs(c.distance - 1.8) < 1e-12 for c in ic)
    dis = [c for c in ciftler if c.cell != (0, 0, 0)]
    assert len(dis) == 2
    assert all(abs(c.distance - 2.2) < 1e-12 for c in dis)


def test_gecersiz_orgu_reddediliyor():
    with pytest.raises(InvalidInputError, match="orgu vektoru"):
        Lattice(vectors=np.zeros((0, 3)), basis=np.zeros((1, 3)))
    with pytest.raises(InvalidInputError, match="dogrusal bagimli"):
        Lattice(
            vectors=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            basis=np.zeros((1, 3)),
        )
    with pytest.raises(InvalidInputError, match="kesme"):
        neighbour_shells(
            Lattice(vectors=np.array([[2.5, 0.0, 0.0]]), basis=np.zeros((1, 3))),
            cutoff_bohr=-1.0,
        )


def test_ters_orgu_EGIK_orguDE_de_dogru():
    """fcc TEK BASINA yetmiyor -- matrisi SIMETRIK.

    OLCULDU: fcc orgu matrisi [[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]]
    simetriktir, dolayisiyla `inv()` ile `inv().T` AYNI seydir ve yalniz
    fcc'ye bakan bir test `.T`'nin dusurulmesini goremez (mutant besli
    sinavda hayatta kaldi). Egik (simetrik olmayan) bir orgu ayirir:
    dogru surumde artik 1.178e-16, `.T` dusurulunce 2.173.
    """
    egik = np.array([[3.0, 0.0, 0.0], [0.7, 2.4, 0.0], [0.3, 0.9, 4.1]])
    assert not np.allclose(egik, egik.T)  # tuzagin hala orada oldugu
    orgu = Lattice(vectors=egik, basis=np.zeros((1, 3)))
    b = reciprocal_vectors(orgu)
    assert np.allclose(b @ egik.T, 2.0 * np.pi * np.eye(3), atol=1e-12)


def test_kesme_TOLERANSININ_BUYUKLUGU_de_kilitli():
    """fcc'nin kabuk araligi cok GENIS; tolerans buyuklugunu sinayamaz.

    OLCULDU: fcc'de birinci kabuk a/sqrt(2) = 4.829539, ikincisi a = 6.830000;
    tolerans ancak sqrt(2)-1 = %41.4'e cikarsa ikinci kabugu icine alir.
    Yani `_KESME_TOLERANSI` 1e-9 yerine 1e-1 yapilsa fcc testi HIC
    kizarmaz (mutant hayatta kaldi).

    Burada iki kabuk %1 arayla kurulur: a = 5.025, taban [0, 2.5] ->
    mesafeler 2.5000 ve 2.5250. cutoff = 2.5'te dogru cevap 2 komsudur;
    %1'den buyuk bir tolerans 4 verir.
    """
    yakin = Lattice(
        vectors=np.array([[5.025, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]),
    )
    assert len(neighbour_shells(yakin, cutoff_bohr=2.5)) == 2
    # ust kabuk gercekten hemen orada -- test bosuna gecmiyor
    assert len(neighbour_shells(yakin, cutoff_bohr=2.5250)) == 4
