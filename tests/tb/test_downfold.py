"""UHF indirgemesi: spin-kirik cozumden site parametreleri.

SINIF UYARISI: bu dosyadaki testler TRANSKRIPSIYON DENETIMIDIR -- indirgemenin
TANIMINI kilitler. Fiziksel cipa ayri dosyadadir (test_downfold_cipa.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.tb.downfold import DownfoldedParameters, downfold_uhf
from qlab.errors import UnsupportedSystemError
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM


def _halka(n: int, d: float) -> Molecule:
    """Komsu araligi d bohr olan duzgun N-halka."""
    yaricap = d / (2 * np.sin(np.pi / n))
    a = yaricap * BOHR_TO_ANGSTROM
    return Molecule.from_string(
        "; ".join(
            f"H {a*np.cos(2*np.pi*k/n):.17g} {a*np.sin(2*np.pi*k/n):.17g} 0"
            for k in range(n)
        )
    )


def test_indirgeme_ANTIFERROMANYETIK_yapiyi_DOGRULUYOR():
    """OLCULDU (N=8, d=2.0 bohr, STO-3G): alfa ve beta kosegeni birbirinin BIR
    SITE kaydirilmisidir; kaydirma artigi 5.558e-09.

    Yapi VARSAYILMAZ, dogrulanir: `staggering_residual` alani her cagrida
    olculur ve saklanir.
    """
    p = downfold_uhf(_halka(8, 2.0), "sto-3g", seeds=12)
    assert isinstance(p, DownfoldedParameters)
    assert p.n_site == 8
    assert p.staggering_residual < 1e-7


def test_KADEMELI_ALAN_ve_ortalama_yerinde_enerji():
    """OLCULDU (N=8, d=2.0): kosegen iki deger arasinda alterniyor,
    +0.250498 ve -0.135858. Dolayisiyla
        eps_bar = (0.250498 + (-0.135858)) / 2 = +0.057320
        |Delta| = (0.250498 - (-0.135858)) / 2 =  0.193178
    """
    p = downfold_uhf(_halka(8, 2.0), "sto-3g", seeds=12)
    assert p.epsilon_bar == pytest.approx(0.057320, abs=1e-5)
    assert abs(p.delta) == pytest.approx(0.193178, abs=1e-5)


@pytest.mark.slow
def test_DELTA_ISARETI_SPIN_POPULASYONUNA_BAGLI():
    """MUTASYON SINAVININ ACTIGI DELIK -- plan bu testi icermiyordu.

    `slow` ISARETI TAHMIN DEGIL, OLCUM: bu test 3.28 s suruyor (iki sistem,
    her birinde `search_uhf` IKI kez -- biri indirgeme icin, biri spin
    populasyonu icin). Deponun kendi siniri da olculdu: `test_rhf_sikismasi`
    ve `test_uhf_cipalar` icinde 2.04-5.20 s arasindaki testlerin HEPSI
    isaretli, 0.86 s ve altindakilerin hicbiri degil; yazili kural da
    "saniye alti cipalar varsayilan seritte". 3.28 s isaretli tarafa duser.
    Dosyanin geri kalani (en yavas 1.27 s) varsayilan seritte kalir.

    `delta`nin isareti yukaridaki testte `abs()` icinde kayboluyor. Olculdu:
    `[0::2] - [1::2]` yerine `[1::2] - [0::2]` yazan mutant SEKIZ testin
    SEKIZINDEN de YESIL gecti. Kapatan iddia sudur:

    Alfa elektronunun yerinde enerjisi, BETA fazlasi tasiyan sitede YUKSELIR.
    Sebep degistokustur: Coulomb TOPLAM yogunluktan gelir ve yuk yaklasik
    duzgun dagildigi icin iki alt-orguda hemen hemen ayni; degistokus ise
    yalnizca AYNI spinden gelir, dolayisiyla alfa yogunlugunun az oldugu
    sitede alfa daha az kararlilasir. Yani `delta` ile cift indisli
    sitelerin alfa-spin fazlasi ZIT isaretlidir.

    Isaretin KENDISI sistemden sisteme doner -- hangi alt-orgunun alfa
    tasidigi UHF cozumunun kendi secimidir -- ama ZITLIK donmez. Test tam da
    bu yuzden iki isareti de goren iki sistem kullanir. OLCULDU:

        N=8, d=2.0 : spin(site 0) = -0.517253  ->  delta = +0.193178
        N=8, d=2.6 : spin(site 0) = +0.703871  ->  delta = -0.265159

    Sabit bir isaret beklemek YANLIS test olurdu (ikinci sistemde kirilirdi);
    carpimin isaretini beklemek ise cozum aynasina donse bile ayakta kalir.
    """
    from qlab.engine.hf.basis import build_basis
    from qlab.engine.hf.integrals import overlap_matrix
    from qlab.engine.hf.multistart import search_uhf

    for n, d in ((8, 2.0), (8, 2.6)):
        molecule = _halka(n, d)
        p = downfold_uhf(molecule, "sto-3g", seeds=12)
        sonuc = search_uhf(molecule, "sto-3g", seeds=12).result
        overlap = overlap_matrix(build_basis(molecule, "sto-3g"))
        spin_cift = float(
            np.mean(np.diag((sonuc.density_alpha - sonuc.density_beta) @ overlap)[0::2])
        )
        assert abs(p.delta) > 1e-3, f"N={n}, d={d}: cozum spin-kirik degil"
        assert abs(spin_cift) > 1e-3, f"N={n}, d={d}: spin populasyonu sifir"
        assert p.delta * spin_cift < 0.0, (
            f"N={n}, d={d}: delta {p.delta:+.6f} ile cift-indis spini "
            f"{spin_cift:+.6f} AYNI isaretli -- kademeli alanin yonu ters"
        )


def test_HOPLAMA_SPINDEN_BAGIMSIZ_ve_TEKBICIM():
    """OLCULDU (N=8, d=2.0): t^alfa = t^beta = -0.407145, sekiz bagin
    sekizinde de ayni.

    Bu, indirgemenin YAPISAL bir olgusudur ve modelin neden TEK hoplama
    parametresi tasidiginin gerekcesidir.

    ESIK 1e-8 VE BU SECILMEDI, OLCULDU. "Ayni" tam olarak ayni degildir:
    N=8 d=2.0'da max|t^a - t^a[0]| = 3.0035e-09, max|t^a - t^b| = 8.2137e-10.
    Bu taban SCF'in ARTIK TOLERANSIDIR, yapi degil -- kanit, toleransi
    sikilastirinca tabanin dusmesidir (ayni sistem, ayni 12 tohum,
    E_UHF = -4.093600816789 on iki hanede DEGISMEDEN):

        res_tol   max|t^a-t^a[0]|   max|t^a-t^b|   kaydirma artigi
        1e-08          3.0035e-09     8.2137e-10        5.5580e-09
        1e-09          3.4939e-11     1.7274e-11        6.1450e-11
        1e-11          1.6098e-15     6.1062e-16        1.4988e-15

    `solve_uhf`in varsayilan artik toleransi 1e-8'dir ve `search_uhf` onu
    degistirmez; yani 1e-9'luk bir esik cozucunun kendi tabaninin ALTINDA
    kalirdi ve test yapiyi degil gurultuyu olcerdi. 1e-8, olculen 3.0e-09'un
    ~3 kati ustunde ve fiziksel olcegin (|t| ~ 0.4) sekiz mertebe altinda.
    """
    from qlab.engine.tb.downfold import _effective_hamiltonians

    h_alpha, h_beta, _, _, _ = _effective_hamiltonians(_halka(8, 2.0), "sto-3g", 12)
    n = 8
    t_alpha = np.array([h_alpha[i, (i + 1) % n] for i in range(n)])
    t_beta = np.array([h_beta[i, (i + 1) % n] for i in range(n)])
    assert np.abs(t_alpha - t_alpha[0]).max() < 1e-8
    assert np.abs(t_alpha - t_beta).max() < 1e-8
    assert t_alpha[0] == pytest.approx(-0.407145, abs=1e-5)


def test_MENZIL_2_de_kesiliyor_ve_bu_BEYAN_EDILIYOR():
    """`n_neighbour` kac hoplama okundugunu belirler; varsayilan 2.

    MUTASYON SINAVININ ACTIGI IKINCI DELIK -- plan yalnizca UZUNLUK ve
    "p2[0] == p1[0]" sartini yaziyordu, hicbir yerde `hoppings`in DEGERI
    okunmuyordu. Olculdu: menzil suzgecini `== m` yerine `<= m` yapan
    mutant (yani her t_m'ye KOSEGENI de katan mutant) alti testin
    altisindan da YESIL gecti -- cunku uzunluk degismiyor ve p1 ile p2
    ayni sekilde bozuldugu icin esitlik de korunuyordu.

    Degerler OLCULDU (N=8, d=2.0, STO-3G, 12 tohum):
        t1 = -0.407145   halka mesafesi 1 olan 16 elemanin ortalamasi
        t2 = +0.046805   halka mesafesi 2 olan 16 elemanin ortalamasi
    t1, ayni sistemin NN elemaniyla (bir onceki test) BIREBIR ayni sayidir;
    `<= m` mutanti t1'i -0.2524'e tasiyordu (kosegen ortalamaya karisiyor).
    """
    p1 = downfold_uhf(_halka(8, 2.0), "sto-3g", n_neighbour=1, seeds=12)
    p2 = downfold_uhf(_halka(8, 2.0), "sto-3g", n_neighbour=2, seeds=12)
    assert len(p1.hoppings) == 1
    assert len(p2.hoppings) == 2
    assert p2.hoppings[0] == pytest.approx(p1.hoppings[0], abs=1e-9)
    assert p2.hoppings[0] == pytest.approx(-0.407145, abs=1e-5)
    assert p2.hoppings[1] == pytest.approx(+0.046805, abs=1e-5)


def test_TEK_SAYIDA_site_REDDEDILIYOR():
    """Iki-siteli hucre TEK sayida site ile kurulamaz; sessizce ortalamak
    kademeli alani YOK EDERDI.

    `match` BUYUK harflidir cunku regex buyuk/kucuk harfe duyarlidir ve
    kaynaktaki ileti "CIFT" der; kucuk harfli bir desen mesaji hic
    eslemez ve testi YANLIS sebeple kirmizi yapardi.
    """
    with pytest.raises(UnsupportedSystemError, match="CIFT"):
        downfold_uhf(_halka(7, 2.0), "sto-3g", seeds=8)


def test_SPIN_KIRILMAMIS_cozumde_Delta_TAM_SIFIR():
    """OLCULDU (N=6, d=2.0 bohr -- gecis noktasinin ALTINDA): UHF, RHF'e
    cokuyor ve Delta tam olarak 0.000000 cikiyor.

    Bu, indirgemenin spin-kirilmamis rejimi SESSIZCE bozmadiginin kontrolu.
    """
    p = downfold_uhf(_halka(6, 2.0), "sto-3g", seeds=16)
    assert abs(p.delta) < 1e-8
    assert p.s_squared < 1e-6
