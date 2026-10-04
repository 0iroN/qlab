"""H(k) kurulumu -- uc olculmus sessiz hata burada kilitleniyor.

(1) `eigh` varsayilan UPLO='L' ile UST ucgeni HIC OKUMUYOR. (2) Gauge secimi
H'nin elemanlarini degistiriyor ama ozdegerleri degistirmiyor -- yani H
karsilastirmasi yaniltici, ozdeger karsilastirmasi dogru. (3) Buyuk k'de
argument reduction sessizce hata uretiyor.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.tb.hamiltonian import (
    Hopping,
    TBModel,
    bloch_hamiltonian,
    finite_ring_matrix,
    uniform_model,
)
from qlab.engine.tb.lattice import Lattice, reciprocal_vectors
from qlab.errors import InvalidInputError

A_ZINCIR = 2.5
T_ZINCIR = -0.0625  # hartree; deneysel hicbir sayidan gelmiyor, keyfi
EPS_ZINCIR = -0.1948


def _zincir(t: float = T_ZINCIR, eps: float = EPS_ZINCIR) -> TBModel:
    orgu = Lattice(vectors=np.array([[A_ZINCIR, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    return TBModel(
        lattice=orgu,
        onsite=np.array([eps]),
        hoppings=(
            Hopping(i=0, j=0, cell=(1, 0, 0), t=t),
            Hopping(i=0, j=0, cell=(-1, 0, 0), t=t),
        ),
    )


def _ssh() -> TBModel:
    orgu = Lattice(
        vectors=np.array([[4.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]),
    )
    return TBModel(
        lattice=orgu,
        onsite=np.array([0.0, 0.0]),
        hoppings=(
            Hopping(i=0, j=1, cell=(0, 0, 0), t=-0.10),
            Hopping(i=1, j=0, cell=(0, 0, 0), t=-0.10),
            Hopping(i=1, j=0, cell=(1, 0, 0), t=-0.04),
            Hopping(i=0, j=1, cell=(-1, 0, 0), t=-0.04),
        ),
    )


def test_H_HERMITYEN():
    model = _zincir()
    for kx in np.linspace(-1.5, 1.5, 17):
        H = bloch_hamiltonian(model, np.array([kx, 0.0, 0.0]))
        assert np.allclose(H, H.conj().T, atol=1e-15)


def test_UST_ucgen_de_OKUNUYOR():
    """`eigh`in varsayilan UPLO='L' tuzagi.

    OLCULDU: `np.linalg.eigh` varsayilan olarak yalniz ALT ucgeni okur --
    ust ucgene +1000 eklemek ozdegerleri 0.000e+00 degistiriyor. Yani
    H'nin ust yarisini yanlis kuran bir hata, `eigh` kullanan bir testten
    SESSIZCE gecer. Bu test H'yi dogrudan denetler, ozdegerlerden degil.
    """
    H = bloch_hamiltonian(_ssh(), np.array([0.3, 0.0, 0.0]))
    assert abs(H[0, 1]) > 1e-3, "ust ucgen sifir kalmis -- H yarim kuruluyor"
    assert np.allclose(H[0, 1], np.conj(H[1, 0]), atol=1e-15)


def test_UPLO_tuzaginin_GERCEK_oldugu():
    """Tuzagin hala orada oldugunu kilitler -- yoksa yukaridaki test
    "gereksiz tekrar" diye silinebilir.

    OLCULDU: `eigh` ust ucgene +1000 eklendiginde ozdegerleri TAM
    0.000e+00 degistiriyor. Yani ozdeger uzerinden calisan hicbir test
    ust ucgen hatasini goremez.
    """
    H = bloch_hamiltonian(_ssh(), np.array([0.3, 0.0, 0.0]))
    bozuk = H.copy()
    bozuk[0, 1] += 1000.0
    assert np.abs(np.linalg.eigvalsh(H) - np.linalg.eigvalsh(bozuk)).max() == 0.0


def test_gauge_H_ELEMANLARINI_degistirir_OZDEGERLERI_degistirmez():
    """Ayar (gauge) secimi: hucre gauge'u vs atom gauge'u.

    OLCULDU (SSH, k+G karsilastirmasi): H elemanlarinin farki atom
    gauge'unda 5.20 eV'ye cikarken ozdegerler iki gauge'da 8 ULP'ta ayni
    kaliyor. Dersi su: iki uygulamayi H uzerinden karsilastirmak
    YANILTICIDIR, ozdeger uzerinden karsilastirmak dogrudur.
    """
    model = _ssh()
    k = np.array([0.37, 0.0, 0.0])
    H_hucre = bloch_hamiltonian(model, k, gauge="cell")
    H_atom = bloch_hamiltonian(model, k, gauge="atom")
    assert np.abs(H_hucre - H_atom).max() > 1e-3
    assert np.allclose(
        np.linalg.eigvalsh(H_hucre), np.linalg.eigvalsh(H_atom), atol=1e-14
    )


def test_Bloch_periyodikligi():
    """E(k + G) = E(k). Hucre gauge'unda H'nin KENDISI de periyodiktir."""
    model = _zincir()
    G = reciprocal_vectors(model.lattice)[0]
    for kx in np.linspace(-1.0, 1.0, 21):
        k = np.array([kx, 0.0, 0.0])
        e1 = np.linalg.eigvalsh(bloch_hamiltonian(model, k))
        e2 = np.linalg.eigvalsh(bloch_hamiltonian(model, k + G))
        assert np.allclose(e1, e2, atol=1e-13)


def test_BUYUK_k_de_argument_reduction_BOZULUYOR():
    """Dogrudan olculmus bir sinir -- kod duzeltmez, BEYAN eder.

    OLCULDU: k -> k + 1e12*G donusumunde ozdegerlerin max farki
    2.387652e-03 eV mertebesine cikiyor ve HICBIR uyari verilmiyor.
    Kaynagi trigonometrik argument reduction'dir, model degil.
    """
    model = _zincir()
    G = reciprocal_vectors(model.lattice)[0]
    k = np.array([0.37, 0.0, 0.0])
    e0 = np.linalg.eigvalsh(bloch_hamiltonian(model, k))
    kucuk = np.abs(np.linalg.eigvalsh(bloch_hamiltonian(model, k + 10 * G)) - e0).max()
    buyuk = np.abs(np.linalg.eigvalsh(bloch_hamiltonian(model, k + 1e12 * G)) - e0).max()
    assert kucuk < 1e-12
    assert buyuk > 1e-9


def test_zincir_analitik_dispersiyon_KABA_kontrol():
    """E(k) = eps + 2t*cos(ka).

    DIKKAT -- BU BIR CIPA DEGILDIR, transkripsiyon denetimidir. Olculdu:
    tek orbitalli zincirde `eigvalsh(1x1)` ile `H[0,0].real` arasi 0 ULP,
    yani cozucu hicbir sey yapmiyor ve "sayisal = analitik, 1 ULP"
    iddiasinin tamami birlesme sirasi farkindan ibaret. Gercek capraz
    kontrol Task 3'tedir. Bu test yalniz isaret/carpan hatasini yakalar --
    ve tolerans da ona gore GEVSEK secilmistir (1e-12, makine hassasiyeti
    DEGIL).
    """
    model = _zincir()
    for kx in np.linspace(-1.2, 1.2, 25):
        e = np.linalg.eigvalsh(bloch_hamiltonian(model, np.array([kx, 0.0, 0.0])))[0]
        beklenen = EPS_ZINCIR + 2.0 * T_ZINCIR * np.cos(kx * A_ZINCIR)
        assert abs(e - beklenen) < 1e-12


def test_uniform_model_kesme_yaricapindan_komsu_kuruyor():
    """`neighbour_shells`in iki tuketicisinden biri (digeri `distance_model`,
    sayim `test_distance_model.py`de `ast` ile kilitli) -- Task 1'in
    muhafazasi GERCEKTEN kullanilan bir yolda kalsin diye.
    """
    kubik = Lattice(
        vectors=np.array([[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 2.5]]),
        basis=np.zeros((1, 3)),
    )
    model = uniform_model(kubik, onsite=np.array([-0.1]), cutoff_bohr=2.5, t=-0.05)
    assert len(model.hoppings) == 6  # z = 6, en yakin komsu
    assert all(h.t == -0.05 for h in model.hoppings)
    # Gamma'da E = eps + z*t
    e = np.linalg.eigvalsh(bloch_hamiltonian(model, np.zeros(3)))[0]
    assert abs(e - (-0.1 + 6 * -0.05)) < 1e-14


def test_gecersiz_model_reddediliyor():
    orgu = Lattice(vectors=np.array([[2.5, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    with pytest.raises(InvalidInputError, match="onsite"):
        TBModel(lattice=orgu, onsite=np.array([0.0, 0.0]), hoppings=())
    with pytest.raises(InvalidInputError, match="orbital indisi"):
        TBModel(
            lattice=orgu,
            onsite=np.array([0.0]),
            hoppings=(Hopping(i=0, j=3, cell=(1, 0, 0), t=-0.1),),
        )
    with pytest.raises(InvalidInputError, match="sonlu"):
        TBModel(
            lattice=orgu,
            onsite=np.array([0.0]),
            hoppings=(Hopping(i=0, j=0, cell=(1, 0, 0), t=float("nan")),),
        )
    with pytest.raises(InvalidInputError, match="gauge"):
        bloch_hamiltonian(_zincir(), np.zeros(3), gauge="baska")


def test_H0_HERMITYEN_OLMAYAN_hoplama_listesi_REDDEDILIYOR():
    """H0 (04.10.2026): tek yonlu liste eskiden bant genisligini YARIYA indiriyordu.

    `eigvalsh` yalniz alt ucgeni okur; eksik es sessizce yok sayiliyordu.
    Tolerans OLCEGE gore 1e-12: 1e-11 reddedilir, 1e-13 gecer (testlerin
    kurdugu 1904 modelde en buyuk yuvarlama sapmasi 7.22e-16).
    """
    orgu = Lattice(vectors=np.array([[A_ZINCIR, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    eps = np.array([EPS_ZINCIR])
    with pytest.raises(InvalidInputError, match="Hermityen degil.*YOK"):
        TBModel(lattice=orgu, onsite=eps, hoppings=(Hopping(0, 0, (1, 0, 0), T_ZINCIR),))
    for bagil, gecer in ((1e-11, False), (1e-13, True)):
        cift = (
            Hopping(0, 0, (1, 0, 0), T_ZINCIR),
            Hopping(0, 0, (-1, 0, 0), T_ZINCIR * (1.0 + bagil)),
        )
        if gecer:
            TBModel(lattice=orgu, onsite=eps, hoppings=cift)
        else:
            with pytest.raises(InvalidInputError, match="Hermityen degil"):
                TBModel(lattice=orgu, onsite=eps, hoppings=cift)
    # Ayni (i, j, R) parcali yazilabilir: karsilastirilan TOPLAMDIR.
    parcali = (
        Hopping(0, 0, (1, 0, 0), T_ZINCIR / 2),
        Hopping(0, 0, (1, 0, 0), T_ZINCIR / 2),
        Hopping(0, 0, (-1, 0, 0), T_ZINCIR),
    )
    TBModel(lattice=orgu, onsite=eps, hoppings=parcali)
    # Esi yazilmamis SIFIR hoplama H'yi degistirmez: kabul (dogrulama turu).
    TBModel(
        lattice=orgu, onsite=eps,
        hoppings=parcali + (Hopping(0, 0, (2, 0, 0), 0.0),),
    )
    # Kesirli hucrede H ters orgude periyodik degildir: B1/B2 coker.
    with pytest.raises(InvalidInputError, match="tam sayi"):
        TBModel(
            lattice=orgu, onsite=eps,
            hoppings=(Hopping(0, 0, (0.5, 0, 0), T_ZINCIR), Hopping(0, 0, (-0.5, 0, 0), T_ZINCIR)),
        )
    # Iki orbital: (0, 1, R)'nin esi (1, 0, -R) -- (1, 0, +R) DEGIL.
    orgu2 = Lattice(
        vectors=np.array([[A_ZINCIR, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [A_ZINCIR / 2, 0.0, 0.0]]),
    )
    yanlis_isaret = (
        Hopping(0, 1, (0, 0, 0), T_ZINCIR),
        Hopping(1, 0, (0, 0, 0), T_ZINCIR),
        Hopping(0, 1, (-1, 0, 0), T_ZINCIR),
        Hopping(1, 0, (-1, 0, 0), T_ZINCIR),
    )
    with pytest.raises(InvalidInputError, match="Hermityen degil"):
        TBModel(lattice=orgu2, onsite=np.zeros(2), hoppings=yanlis_isaret)
    dogru = yanlis_isaret[:3] + (Hopping(1, 0, (1, 0, 0), T_ZINCIR),)
    TBModel(lattice=orgu2, onsite=np.zeros(2), hoppings=dogru)


def test_finite_ring_matrix_HERMITYEN_ve_dogru_boyutta():
    model = _zincir()
    H = finite_ring_matrix(model, n_cell=12)
    assert H.shape == (12, 12)
    assert np.allclose(H, H.T, atol=1e-15)
    assert np.count_nonzero(np.abs(H) > 1e-15) == 12 + 24  # kosegen + iki komsu


def test_FAZ_ISARETI_konvansiyonu_kilitli():
    """Ozdegerler faz isaretine KORDUR -- konvansiyon ayrica kilitlenmeli.

    OLCULDU: `exp(+ik.R)` yerine `exp(-ik.R)` yazan mutant ozdegerleri
    TAM 0.000e+00 degistiriyor (global kompleks eslenik spektrumu korur) ve
    bu dosyanin on testinin ONUNDAN da geciyordu. Ayiran tek sey H'nin
    kendi elemani: H[0,1] = -0.10362686 + 0.03983523j; ters isarette
    hayali kisim -0.03983523 oluyor (fark 0.079670).

    Konvansiyon: hoplama i -> j, hucre R icin faz exp(+i k.R). Isaret
    spektrum icin onemsiz, ama hiz operatoru, Berry fazi ve iki gauge'un
    tutarliligi icin onemlidir.
    """
    H = bloch_hamiltonian(_ssh(), np.array([0.37, 0.0, 0.0]))
    assert abs(H[0, 1].real - (-0.10362686)) < 1e-8
    assert abs(H[0, 1].imag - (+0.03983523)) < 1e-8


def test_finite_ring_KUCUK_halkada_carpisan_hoplamalari_TOPLUYOR():
    """`+=` yerine `=` yazmak KUCUK halkada bir terimi sessizce dusuruyor.

    OLCULDU: ikinci komsu (+-2) hoplamali bir zincirde n_cell = 5 ve 12'de
    hicbir carpisma yok (fark 0.000000), ama n_cell = 4'te +2 ile -2 AYNI
    hedefe dusuyor ve `=` surumu bir terimi kaybediyor: max|H| farki
    0.030000, spektrum farki 3.000e-02.

    Yani bu hata ancak halka, menzilin iki katindan kucuk oldugunda
    goruluyor -- ve tam da o durum (kucuk halka + uzun menzil) Task 3'un
    capraz kontrolunde kullanilabilir.
    """
    zincir = Lattice(vectors=np.array([[2.5, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    model = TBModel(
        lattice=zincir,
        onsite=np.array([-0.1948]),
        hoppings=(
            Hopping(i=0, j=0, cell=(1, 0, 0), t=-0.0625),
            Hopping(i=0, j=0, cell=(-1, 0, 0), t=-0.0625),
            Hopping(i=0, j=0, cell=(2, 0, 0), t=-0.0300),
            Hopping(i=0, j=0, cell=(-2, 0, 0), t=-0.0300),
        ),
    )
    H = finite_ring_matrix(model, n_cell=4)
    # c=0 -> hedef 2, hem +2 hem -2 oradan geliyor: -0.03 + -0.03
    assert abs(H[0, 2] - (-0.0600)) < 1e-15
    # buyuk halkada carpisma yok, tek terim
    assert abs(finite_ring_matrix(model, n_cell=12)[0, 2] - (-0.0300)) < 1e-15
