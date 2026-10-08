"""Bant yuku -- YUK uretir, goruntu DEGIL; ve OZVEKTOR SAKLAMAZ.

Ozvektor saklamamanin gerekcesi bellek DEGIL, ANLAM. Olculdu: hucre ve atom
gauge'lari arasinda ozvektorler 0.814887 kadar farkli cikiyor ama
|c|^2 agirliklari 3.331e-16'da AYNI. Yani ozvektor gauge'a bagli, agirlik
degil -- ozvektor saklamak, cizime tasinamayan bir sayiyi saklamak olurdu.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.tb.bands import k_path
from qlab.engine.tb.hamiltonian import Hopping, TBModel, bloch_hamiltonian
from qlab.engine.tb.lattice import Lattice, reciprocal_vectors
from qlab.errors import InvalidInputError
from qlab.units import HARTREE_TO_EV
from qlab.viz.band_frames import BandPayload, build_band_payload


def _ssh() -> TBModel:
    orgu = Lattice(
        vectors=np.array([[4.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]),
    )
    return TBModel(
        lattice=orgu,
        onsite=np.array([-0.11, -0.07]),
        hoppings=(
            Hopping(i=0, j=1, cell=(0, 0, 0), t=-0.10),
            Hopping(i=1, j=0, cell=(0, 0, 0), t=-0.10),
            Hopping(i=1, j=0, cell=(1, 0, 0), t=-0.04),
            Hopping(i=0, j=1, cell=(-1, 0, 0), t=-0.04),
        ),
    )


def _yol(model: TBModel):
    G = reciprocal_vectors(model.lattice)[0]
    return k_path((np.zeros(3), 0.5 * G), n_per_segment=41)


def _yuk() -> BandPayload:
    model = _ssh()
    return build_band_payload(
        model, _yol(model), tick_labels=("G", "X"), fermi_hartree=-0.09
    )


def test_yuk_KOMPLEKS_dizi_TASIMIYOR():
    """Olcut "ndim <= 2" DEGIL, KOMPLEKSLIK.

    Orbital agirliklari mesru sekilde (n_k, n_band, n_orb) uc boyutludur;
    sekle bakan bir olcut onlari yanlislikla yasaklardi. Ayirici ozellik
    ozvektorlerin COMPLEX128, agirliklarin GERCEK olmasidir.

    Bu, Modul 2E'de ogrenilenin duzeltilmis hali: orada olcut sekle
    bakiyordu ve `meshes` LISTESININ icine bakmadigi icin "alan kupu
    sakla" mutantini kacirmisti.
    """
    yuk = _yuk()
    diziler = [d for d in vars(yuk).values() if isinstance(d, np.ndarray)]
    assert diziler, "yukte hic dizi yok -- tarama bozuk"
    for ad, deger in vars(yuk).items():
        if isinstance(deger, np.ndarray):
            assert not np.iscomplexobj(deger), f"{ad} kompleks -- ozvektor sizmis"


def test_AGIRLIKLAR_gauge_BAGIMSIZ_ozvektorler_DEGIL():
    """Ozvektor saklamamanin ASIL gerekcesi -- bellek degil ANLAM.

    OLCULDU (SSH): hucre ve atom gauge'lari arasinda
      k=0.00 -> ozvektor farki 0.000000, agirlik farki 0.000e+00
      k=0.37 -> ozvektor farki 0.501586, agirlik farki 1.110e-16
      k=0.60 -> ozvektor farki 0.814887, agirlik farki 3.331e-16
    Yani ozvektor GAUGE'A BAGLI, agirlik degil. Ozvektor saklamak,
    cizime tasinamayan bir sayiyi saklamak olurdu.
    """
    model = _ssh()
    for kx in (0.0, 0.37, 0.60):
        k = np.array([kx, 0.0, 0.0])
        _, v_hucre = np.linalg.eigh(bloch_hamiltonian(model, k, gauge="cell"))
        _, v_atom = np.linalg.eigh(bloch_hamiltonian(model, k, gauge="atom"))
        agirlik_farki = np.abs(
            (np.abs(v_hucre) ** 2).T - (np.abs(v_atom) ** 2).T
        ).max()
        assert agirlik_farki < 1e-14
    # ...ve fark GERCEKTEN buyuk, yani test bosuna gecmiyor
    _, v1 = np.linalg.eigh(bloch_hamiltonian(model, np.array([0.6, 0.0, 0.0]), gauge="cell"))
    _, v2 = np.linalg.eigh(bloch_hamiltonian(model, np.array([0.6, 0.0, 0.0]), gauge="atom"))
    assert np.abs(v1 - v2).max() == pytest.approx(0.814887, abs=1e-5)


def test_yuk_bayt_ayak_izi_SINIRLI():
    """OLCULDU (41 k, 2 bant, 2 orbital): toplam 2.394 bayt -- 2.312'ye E3a'nin
    `degenerate` bayragi (41 x 2 bool = 82 bayt) eklendi.

    Ozvektor saklansaydi agirlik terimi 1.312 -> 2.624 bayta cikardi
    (complex128, tam 2 kat). Kazanc n_orb'dan bagimsiz olarak 2x'tir;
    asil kazanc bellekte degil ANLAMDA (bkz. gauge testi).
    """
    yuk = _yuk()
    toplam = sum(
        d.nbytes for d in vars(yuk).values() if isinstance(d, np.ndarray)
    )
    assert toplam == 2312 + 41 * 2
    assert yuk.degenerate.dtype == bool and yuk.degenerate.nbytes == 41 * 2
    assert yuk.orbital_weights.nbytes == 41 * 2 * 2 * 8


def test_yuk_eV_donusturuyor_MOTOR_hartree_kaliyor():
    """Birim sinirinin YUK tarafinda oldugunu kilitler.

    OLCULDU: Gamma noktasinda ozdegerler -0.23142136 ve +0.05142136 hartree,
    yani -6.29729591 ve +1.39924639 eV. Kapali form
    ort +- sqrt(delta^2 + |t1+t2|^2) ile farki 2.082e-17.
    """
    yuk = _yuk()
    assert yuk.fermi_ev == pytest.approx(-0.09 * HARTREE_TO_EV, abs=1e-12)
    assert yuk.energies_ev[0, 0] == pytest.approx(-6.29729591, abs=1e-7)
    assert yuk.energies_ev[0, 1] == pytest.approx(+1.39924639, abs=1e-7)


def test_orbital_agirliklari_HER_k_DA_BIRE_toplaniyor():
    yuk = _yuk()
    assert yuk.orbital_weights.shape == (41, 2, 2)
    assert np.all(yuk.orbital_weights >= -1e-15)
    assert np.allclose(yuk.orbital_weights.sum(axis=2), 1.0, atol=1e-12)


def test_DEJENERE_alt_uzayda_BANT_BASINA_agirlik_ANLAMSIZ():
    """DURUST OLUMSUZ, dogrudan sergilenerek kilitleniyor.

    Tam dejenere bir alt uzayda `eigh`in sectigi taban KEYFIDIR: o alt
    uzaydaki HER ortonormal cift esit derecede gecerli bir ozvektor
    ciftidir. Burada 2B kare orguda x-yonlu ve y-yonlu iki orbital
    kuruluyor; kosegen k yonunde iki bant TAM dejenere (E fark = 0.0).

    numpy orada [1,0] ve [0,1] donduruyor; 45 derece dondurulmus cift de
    AYNI olcude gecerli bir ozvektor cifti ve onun agirliklari
    [0.5, 0.5] / [0.5, 0.5]. Yani BANT BASINA agirlik tanimli degil --
    ALT UZAY TOPLAMI ([1, 1]) ise iki tabanda da ayni.
    """
    orgu = Lattice(
        vectors=np.array([[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    model = TBModel(
        lattice=orgu,
        onsite=np.array([-0.1, -0.1]),
        hoppings=(
            Hopping(i=0, j=0, cell=(1, 0, 0), t=-0.05),
            Hopping(i=0, j=0, cell=(-1, 0, 0), t=-0.05),
            Hopping(i=1, j=1, cell=(0, 1, 0), t=-0.05),
            Hopping(i=1, j=1, cell=(0, -1, 0), t=-0.05),
        ),
    )
    b = reciprocal_vectors(orgu)
    k = 0.3 * b[0] + 0.3 * b[1]
    H = bloch_hamiltonian(model, k)
    ozdeger, ozvektor = np.linalg.eigh(H)
    assert abs(ozdeger[1] - ozdeger[0]) == 0.0  # TAM dejenere

    numpy_agirlik = (np.abs(ozvektor) ** 2).T
    assert np.allclose(numpy_agirlik, np.eye(2), atol=1e-15)

    aci = np.pi / 4.0
    donme = np.array(
        [[np.cos(aci), -np.sin(aci)], [np.sin(aci), np.cos(aci)]]
    )
    dondurulmus = ozvektor @ donme
    # ...hala ozvektor cifti (dejenere alt uzayda kaliyor)
    assert np.abs(H @ dondurulmus - dondurulmus * ozdeger[None, :]).max() < 1e-15

    donuk_agirlik = (np.abs(dondurulmus) ** 2).T
    assert np.allclose(donuk_agirlik, 0.5, atol=1e-15)  # BANT BASINA degisti
    assert np.allclose(
        numpy_agirlik.sum(axis=0), donuk_agirlik.sum(axis=0), atol=1e-15
    )  # ...ALT UZAY TOPLAMI degismedi


def test_yuk_eq_FALSE_karsilastirma_PATLAMIYOR():
    """OLCULMUS TUZAK: `frames.py:23-24`teki FramePayload `eq=True` tasiyor ve
    `p1 == p2` ValueError atiyor. Modul 2'nin yukleri `eq=False` kullanir.
    """
    a, b = _yuk(), _yuk()
    assert (a == b) is False  # istisna ATMAMALI


def test_etiket_sayisi_uyusmazligi_reddediliyor():
    model = _ssh()
    with pytest.raises(InvalidInputError, match="etiket"):
        build_band_payload(
            model, _yol(model), tick_labels=("G",), fermi_hartree=-0.09
        )
    with pytest.raises(InvalidInputError, match="fermi_hartree"):
        build_band_payload(
            model, _yol(model), tick_labels=("G", "X"), fermi_hartree=float("nan")
        )


def test_AGIRLIK_EKSENLERI_bant_sonra_orbital():
    """`(n_k, n_band, n_orb)` sirasi -- ve neden 2x2 model bunu SINAYAMAZ.

    OLCULDU: uniter bir matriste |c|^2 CIFT STOKASTIKTIR (satir ve sutun
    toplamlarinin ikisi de 1) ve 2x2'de ustelik SIMETRIKTIR --
    [[0.588647, 0.411353], [0.411353, 0.588647]]. Dolayisiyla SSH'de
    `.T`'yi dusuren mutant bu dosyanin sekiz testinden de geciyordu.

    Uc orbitalde matris artik simetrik degil (max|w - w.T| = 0.017463) ve
    eksen sirasi gorunur oluyor. Gamma'da bant 0'in agirliklari:
      dogru (bant x orbital) -> [0.787801, 0.194514, 0.017685]
      .T dusurulmus          -> [0.787801, 0.211977, 0.000222]
    ucuncu bilesende 80 kat fark var.
    """
    orgu = Lattice(
        vectors=np.array([[4.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [2.6, 0.0, 0.0]]),
    )
    model = TBModel(
        lattice=orgu,
        onsite=np.array([-0.20, -0.05, 0.15]),
        hoppings=(
            Hopping(i=0, j=1, cell=(0, 0, 0), t=-0.09),
            Hopping(i=1, j=0, cell=(0, 0, 0), t=-0.09),
            Hopping(i=1, j=2, cell=(0, 0, 0), t=-0.06),
            Hopping(i=2, j=1, cell=(0, 0, 0), t=-0.06),
            Hopping(i=2, j=0, cell=(1, 0, 0), t=-0.03),
            Hopping(i=0, j=2, cell=(-1, 0, 0), t=-0.03),
        ),
    )
    G = reciprocal_vectors(orgu)[0]
    yuk = build_band_payload(
        model,
        k_path((np.zeros(3), 0.5 * G), n_per_segment=5),
        tick_labels=("G", "X"),
        fermi_hartree=-0.05,
    )
    assert yuk.orbital_weights.shape == (5, 3, 3)
    assert np.allclose(
        yuk.orbital_weights[0, 0], [0.787801, 0.194514, 0.017685], atol=1e-6
    )
    # ...ve matris GERCEKTEN asimetrik, yani test bosuna gecmiyor
    w = yuk.orbital_weights[0]
    assert np.abs(w - w.T).max() == pytest.approx(0.017463, abs=1e-6)


def _kare_xy(onsite_farki: float = 0.0) -> TBModel:
    """x-yonlu ve y-yonlu iki orbital: kosegen k'da iki bant TAM dejenere."""
    orgu = Lattice(
        vectors=np.array([[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    return TBModel(
        lattice=orgu,
        onsite=np.array([-0.1, -0.1 + onsite_farki]),
        hoppings=(
            Hopping(i=0, j=0, cell=(1, 0, 0), t=-0.05),
            Hopping(i=0, j=0, cell=(-1, 0, 0), t=-0.05),
            Hopping(i=1, j=1, cell=(0, 1, 0), t=-0.05),
            Hopping(i=1, j=1, cell=(0, -1, 0), t=-0.05),
        ),
    )


def _yuk_yol(model: TBModel, uc: np.ndarray, n: int = 21) -> BandPayload:
    yol = k_path((np.zeros(3), uc), n_per_segment=n)
    return build_band_payload(model, yol, tick_labels=("G", "U"), fermi_hartree=-0.1)


def test_E3a_DEJENERE_kumede_ALT_UZAY_ORTALAMASI_ve_BAYRAK():
    """E3a (04.10.2026): dejenere kumede bant basina agirlik keyfi oldugu
    icin yuk ALT UZAY ORTALAMASINI yazar ve `degenerate` ile isaretler.

    Kare xy modelinde Gamma -> M kosegeni BASTAN SONA dejenere: numpy
    [1,0]/[0,1] dondurur, yuk [0.5,0.5]/[0.5,0.5] yazar. Gamma -> X'te
    yalniz Gamma dejeneredir; oteki noktalar HAM agirligi korur.
    """
    model = _kare_xy()
    b = reciprocal_vectors(model.lattice)
    kosegen = _yuk_yol(model, 0.5 * b[0] + 0.5 * b[1])
    assert kosegen.degenerate.shape == kosegen.energies_ev.shape
    assert kosegen.degenerate.all()
    assert np.allclose(kosegen.orbital_weights, 0.5, atol=1e-15)

    eksen = _yuk_yol(model, 0.5 * b[0])
    assert eksen.degenerate[0].all() and not eksen.degenerate[1:].any()
    assert np.allclose(eksen.orbital_weights[0], 0.5, atol=1e-15)
    assert np.allclose(np.sort(eksen.orbital_weights[1:], axis=2)[..., 1], 1.0, atol=1e-15)


def test_E3a_YUVARLAMA_duzeyi_ayrisma_AGIRLIGI_DEGISTIRMEZ_cozulebilen_DEGISTIRIR():
    """Esik makine hassasiyetinin ~1e3 katidir (1e3 * eps * (|E|max + H
    olcegi) ~ 2e-13 hartree burada). Onsite farki 1e-15 hartree: esigin
    ICINDE -- yuk BILEREK ortalar, bayrak acik (eigh'in agirligi burada
    tam keyfi olmayabilir; ortalama muhafazakar secimdir). 1e-11 hartree:
    fiziksel olarak cozulebilen ayrisma -- yuk HAM agirligi korur ([1,0]/
    [0,1]), bayrak kapali. Sabit 1e-8 hartree esik (on olcum) bunu da
    ortalardi.
    """
    b = reciprocal_vectors(_kare_xy().lattice)
    uc = 0.3 * b[0] + 0.3 * b[1]
    ince = _yuk_yol(_kare_xy(1e-15), uc, n=5)
    assert ince.degenerate.all()
    assert np.allclose(ince.orbital_weights, 0.5, atol=1e-15)
    kaba = _yuk_yol(_kare_xy(1e-11), uc, n=5)
    assert not kaba.degenerate[1:].any()
    assert np.allclose(np.sort(kaba.orbital_weights[1:], axis=2)[..., 1], 1.0, atol=1e-12)


def test_E3a_ESIK_H_OLCEGINI_TASIR_birbirini_goturen_hoplama():
    """Dogrulama turu (04.10.2026): iki orbitalin ikisi de (1.23 + 1.6 - 2.83)
    eV ile hopluyor (siralar farkli) -- matematikte TAM dejenere DUZ cift.
    Toplama kirintisi (H00 ~ 1e-17, H11 ~ 2e-17 hartree) |E|max'in kendisi
    kadardir; esikte H olcegi terimi olmasa 18 bant-noktasinin 8'i bayraksiz
    kaliyor ve yuvarlamanin sectigi [1,0]/[0,1] agirligini aliyordu.
    """
    ev = 1.0 / HARTREE_TO_EV
    orgu = Lattice(vectors=np.array([[4.0, 0.0, 0.0]]), basis=np.array([[0.0, 0, 0], [0, 3.0, 0]]))
    hop = []
    for c in (1, -1):
        hop += [Hopping(0, 0, (c, 0, 0), t * ev) for t in (1.23, 1.6, -2.83)]
        hop += [Hopping(1, 1, (c, 0, 0), t * ev) for t in (-2.83, 1.6, 1.23)]
    model = TBModel(lattice=orgu, onsite=np.zeros(2), hoppings=tuple(hop))
    yuk = _yuk_yol(model, 0.5 * reciprocal_vectors(orgu)[0], n=9)
    assert yuk.degenerate.all()
    assert np.allclose(yuk.orbital_weights, 0.5, atol=1e-15)


def test_E3a_KUMELEME_GECISLI_ve_KATSAYI_kilitli():
    """Esik 1e3 * eps * (|E|max + H olcegi). Hoplamasiz modelde H(k) sabit ve
    kosegen: |E|max = H olcegi = 0.7, esik = 1e3 * eps * 1.4 (~3.1e-13).
    Ardisik farklari 0.6 * esik olan uclu ZINCIR tek kume olur (uc farki
    1.2 * esik olsa da -- kumeleme GECISLIDIR, yayilim esigi asabilir) ve
    alt uzay ortalamasi 1/3 yazilir. Fark 1.5 * esik iken kume YOK. Katsayi
    1e3'ten saparsa (30 ya da 1e4) iki hucreden biri kirilir.
    """
    esik = 1e3 * np.finfo(float).eps * 1.4
    orgu = Lattice(vectors=np.array([[2.5, 0.0, 0.0]]), basis=np.zeros((5, 3)) + np.arange(5)[:, None] * [0, 0.5, 0])
    for adim, beklenen in ((0.6 * esik, [False, True, True, True, False]), (1.5 * esik, [False] * 5)):
        onsite = np.array([-0.5, 0.1, 0.1 + adim, 0.1 + 2 * adim, 0.7])
        model = TBModel(lattice=orgu, onsite=onsite, hoppings=())
        yuk = _yuk_yol(model, 0.5 * reciprocal_vectors(orgu)[0], n=3)
        assert yuk.degenerate[0].tolist() == beklenen, adim / esik
        if beklenen[1]:
            assert np.allclose(yuk.orbital_weights[0, 1:4, 1:4], 1.0 / 3.0, atol=1e-15)
            assert np.allclose(yuk.orbital_weights[0, 1:4].sum(axis=1), 1.0, atol=1e-15)
