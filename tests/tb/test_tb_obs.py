"""TB gozlemlenebilirleri -- ve modulun TEK parametre-bagimsiz fizik hukmu.

Hucre basina TEK sayida elektron her zaman metal verir, cunku tek sayi bir
bandi ancak YARI doldurabilir. Bu bir hesap degil bir SAYIMDIR.

TERSI DOGRU DEGIL ve bu test edilmez, BELGELENIR: cift sayim yalitkanligi
GARANTI ETMEZ -- bant ortusmesi yuzunden metal cikma orani on olcumde
1B %67, 2B %34, 3B %20 olculmustu.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.tb.bands import band_structure, uniform_k_grid
from qlab.engine.tb.hamiltonian import Hopping, TBModel, uniform_model
from qlab.engine.tb.lattice import Lattice
from qlab.errors import InvalidInputError
from qlab.observable.tb_obs import (
    density_of_states,
    fermi_level,
    is_metal,
    structure_ratio,
)

ZINCIR = np.array([[2.5, 0.0, 0.0]])
KARE = np.array([[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]])
KUBIK = np.array([[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 2.5]])


def _tek_orbital(vektorler: np.ndarray, t: float, eps: float) -> TBModel:
    """Hiperkubik kafes, en yakin komsu -- `uniform_model` uzerinden.

    Boylece Task 1'in kesme yaricapi muhafazasi GERCEKTEN kullanilan bir
    yolda kalir. Kesme, orgu sabitinin kendisidir; bir sonraki kabuk
    sqrt(2)*|a|'dadir ve bagil tolerans 1e-9 onu iceri almaz.
    """
    orgu = Lattice(vectors=vektorler, basis=np.zeros((1, 3)))
    kesme = float(np.linalg.norm(vektorler[0]))
    return uniform_model(orgu, onsite=np.array([eps]), cutoff_bohr=kesme, t=t)


def _iki_bant(eps2: float, t2: float) -> TBModel:
    """Iki BAGIMSIZ zincir -- `eps2`/`t2` ile bantlari ortusturup ayirabiliriz."""
    orgu = Lattice(
        vectors=np.array([[4.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
    )
    return TBModel(
        lattice=orgu,
        onsite=np.array([-0.05, eps2]),
        hoppings=(
            Hopping(i=0, j=0, cell=(1, 0, 0), t=-0.09),
            Hopping(i=0, j=0, cell=(-1, 0, 0), t=-0.09),
            Hopping(i=1, j=1, cell=(1, 0, 0), t=t2),
            Hopping(i=1, j=1, cell=(-1, 0, 0), t=t2),
        ),
    )


def test_TEK_elektron_sayimi_HER_ZAMAN_metal():
    """Modulun tek parametre-bagimsiz FIZIK hukmu.

    OLCULDU: 100 rastgele model x 3 boyut = 300/300 metal. Hukum
    istatistiksel degil SAYISALDIR -- tek sayida elektron bir bandi ancak
    YARI doldurabilir, dolayisiyla Fermi seviyesi bandin ICINDE kalir.
    Testte 20 model/boyut kosuluyor (60 model, 0.78 s); 100'lu kosu
    3.92 s tutuyordu ve saniye alti kuralini asiyordu. Sayilar ayrica
    olculdu ve ikisi de 100 %.

    DIKKAT, olcut BANT COZUMLU olmak zorunda: komsu-seviye araligina bakan
    bir olcut sonlu izgarada COKUYOR. Olculdu -- `bosluk/W < 1e-9` esigiyle
    3B'de 0/100 metal cikiyordu, cunku 10x10x10 izgarada seviyeler zaten
    ayriktir. Bant cozumlu olcutle ayni modellerin 100/100'u metal.
    """
    rng = np.random.default_rng(20260820)
    for vektorler, n_k in ((ZINCIR, 60), (KARE, 24), (KUBIK, 10)):
        orgu = Lattice(vectors=vektorler, basis=np.zeros((1, 3)))
        k = uniform_k_grid(orgu, n_k)
        for _ in range(20):
            model = _tek_orbital(
                vektorler,
                t=float(rng.normal(0.0, 0.1)),
                eps=float(rng.normal(0.0, 0.5)),
            )
            assert is_metal(band_structure(model, k), n_electron_per_cell=1) is True


def test_DUZ_bant_metal_SAYILMIYOR():
    """Dejenere durum: t = 0 -> bant genisligi sifir, iletim yok.

    OLCULDU: on olcumde 4352 modelin 0'inda genisligi sifir bant cikmisti,
    yani bu durum dogal olarak dogmuyor -- ama `t=0` verilirse dogar ve
    "tek sayim -> metal" kurali orada gecerli DEGILDIR. Kod bunu sessizce
    metal saymaz.
    """
    model = _tek_orbital(ZINCIR, t=0.0, eps=-0.1)
    e = band_structure(model, uniform_k_grid(model.lattice, 40))
    assert float(e.max() - e.min()) == 0.0
    assert is_metal(e, n_electron_per_cell=1) is False


def test_CIFT_sayim_yalitkanligi_GARANTI_ETMIYOR():
    """DURUST OLUMSUZ -- ve bu testin isi onu KILITLEMEK, gizlemek degil.

    OLCULDU (on olcum): bant ortusmesi yuzunden cift sayimda metal cikma
    orani 1B %67, 2B %34, 3B %20. Yani "cift elektron -> yalitkan" kurali
    KODA YAZILAMAZ.

    Burada iki BAGIMSIZ zincir kurulur; bant0 ust kenari 0.1300, bant1 alt
    kenari -0.1600 -> ortusuyorlar ve sonuc METAL.
    """
    e = band_structure(_iki_bant(eps2=0.02, t2=-0.09), uniform_k_grid(
        _iki_bant(eps2=0.02, t2=-0.09).lattice, 120))
    assert float(e[:, 0].max()) == pytest.approx(0.1300, abs=1e-9)
    assert float(e[:, 1].min()) == pytest.approx(-0.1600, abs=1e-9)
    assert is_metal(e, n_electron_per_cell=2) is True


def test_yalitkan_ornegi_de_var():
    """Ayrik iki bant -> cift sayim yalitkan. KURAL DEGIL, ORNEK.

    OLCULDU: bant0 ust 0.1300, bant1 alt 0.9600 -> gercek bir aralik var.
    """
    model = _iki_bant(eps2=1.0, t2=-0.02)
    e = band_structure(model, uniform_k_grid(model.lattice, 120))
    assert float(e[:, 0].max()) == pytest.approx(0.1300, abs=1e-9)
    assert float(e[:, 1].min()) == pytest.approx(0.9600, abs=1e-9)
    assert is_metal(e, n_electron_per_cell=2) is False


def test_fermi_seviyesi_yari_dolu_zincirde_bant_MERKEZINDE():
    """OLCULDU: E_F = -0.194800000, girilen eps ile farki TAM 0.000e+00.

    DIKKAT -- bu bir DOGRULAMA DEGILDIR: mutlak E_F, girilen `eps`in geri
    okunmasidir. Test yalniz doluluk mantiginin dogru oldugunu denetler.
    """
    model = _tek_orbital(ZINCIR, t=-0.0625, eps=-0.1948)
    e = band_structure(model, uniform_k_grid(model.lattice, 400))
    assert fermi_level(e, n_electron_per_cell=1) == pytest.approx(-0.1948, abs=1e-12)


@pytest.mark.parametrize(
    ("vektorler", "n_k", "z"), [(ZINCIR, 50, 2), (KARE, 20, 4), (KUBIK, 8, 6)]
)
def test_YAPI_ORANI_W2_bolu_mu2_esittir_4z(vektorler, n_k, z):
    """t ve eps cebirsel olarak SADELESIYOR -- geriye koordinasyon sayisi kaliyor.

    OLCULDU: 1B 8.000000000 (bagil fark 2.22e-16), 2B 16.000000000 (0.00e+00),
    3B 24.000000000 (0.00e+00). Uc farkli (t, eps) setiyle -- ki biri
    eps = -400 gibi asiri -- ayni sonuc.

    Ucu de KUCUK izgarada zaten tam cikiyor (1B n=50, 2B n=20, 3B n=8);
    yani bu oran bir yakinsama meselesi degil, cebirsel bir ozdesliktir.
    """
    orgu = Lattice(vectors=vektorler, basis=np.zeros((1, 3)))
    k = uniform_k_grid(orgu, n_k)
    for t, eps in ((-0.0625, -0.1948), (+3.7, +11.0), (-0.001, -400.0)):
        e = band_structure(_tek_orbital(vektorler, t=t, eps=eps), k)
        assert structure_ratio(e) == pytest.approx(4 * z, rel=1e-9)


def test_YAPI_ORANININ_SINIRI_olculmustur():
    """Oran ETKIN koordinasyonu okur -- ve model degisince 4z'den kayar.

    SSH'de hoplamalar t1 = -0.10 sabit, t2 degiskenken OLCULDU:
      t2 = -0.10 (tekbicim) -> 8.000000   (z = 2, tam)
      t2 = -0.05             -> 7.200000   (%-10.00)
      t2 = -0.02             -> 5.538462   (%-30.77)
      t2 =  0.00 (izole dimer) -> 4.000000 (%-50.00, yani z = 1)
    Yani dimerlesme etkin koordinasyonu 2'den 1'e SUREKLI olarak indiriyor.

    Ikinci komsu ise oteki yone tasiyor: 1B zincirde t2 = t1 iken
    9.765604 (%+22.07).

    Sonuc: "4z okudum" demek bir MODEL VARSAYIMINI geri okumaktir. Gercek
    bir malzemede hem cok orbital hem ikinci komsu vardir.
    """
    ssh_orgu = Lattice(
        vectors=np.array([[4.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]),
    )
    k = uniform_k_grid(ssh_orgu, 400)
    beklenen = {-0.10: 8.000000, -0.05: 7.200000, -0.02: 5.538462, 0.0: 4.000000}
    for t2, deger in beklenen.items():
        model = TBModel(
            lattice=ssh_orgu,
            onsite=np.array([0.0, 0.0]),
            hoppings=(
                Hopping(i=0, j=1, cell=(0, 0, 0), t=-0.10),
                Hopping(i=1, j=0, cell=(0, 0, 0), t=-0.10),
                Hopping(i=1, j=0, cell=(1, 0, 0), t=t2),
                Hopping(i=0, j=1, cell=(-1, 0, 0), t=t2),
            ),
        )
        assert structure_ratio(band_structure(model, k)) == pytest.approx(
            deger, abs=1e-6
        )

    zincir_orgu = Lattice(vectors=ZINCIR, basis=np.zeros((1, 3)))
    ikinci = TBModel(
        lattice=zincir_orgu,
        onsite=np.array([-0.1948]),
        hoppings=(
            Hopping(i=0, j=0, cell=(1, 0, 0), t=-0.0625),
            Hopping(i=0, j=0, cell=(-1, 0, 0), t=-0.0625),
            Hopping(i=0, j=0, cell=(2, 0, 0), t=-0.0625),
            Hopping(i=0, j=0, cell=(-2, 0, 0), t=-0.0625),
        ),
    )
    oran = structure_ratio(band_structure(ikinci, uniform_k_grid(zincir_orgu, 400)))
    assert oran == pytest.approx(9.765604, abs=1e-5)
    assert oran > 8.0  # ikinci komsu oteki YONE tasiyor


def test_durum_yogunlugu_toplami_KORUNUYOR():
    """DOS'un integrali 1 olmali (normalizasyon denetimi)."""
    model = _tek_orbital(ZINCIR, t=-0.0625, eps=-0.1948)
    e = band_structure(model, uniform_k_grid(model.lattice, 2000))
    merkezler, dos = density_of_states(e, n_bin=200)
    genislik = float(merkezler[1] - merkezler[0])
    assert abs(float(dos.sum()) * genislik - 1.0) < 1e-9


def test_gecersiz_girdiler_reddediliyor():
    model = _tek_orbital(ZINCIR, t=-0.0625, eps=-0.1948)
    e = band_structure(model, uniform_k_grid(model.lattice, 40))
    with pytest.raises(InvalidInputError, match="elektron sayisi"):
        fermi_level(e, n_electron_per_cell=0)
    with pytest.raises(InvalidInputError, match="elektron sayisi"):
        is_metal(e, n_electron_per_cell=5)
    # SINIR TAM KENARINDAN sinanir: tek bantli modelde ust sinir 2'dir
    # (spin dejenereligi 2 x 1 bant). Olculdu -- yalniz n=5 denemek yetmiyor,
    # sinirI 4*n_band'a gevseten mutant o testten GECIYORDU.
    assert is_metal(e, n_electron_per_cell=2) is False  # tam dolu, gecerli
    with pytest.raises(InvalidInputError, match="elektron sayisi"):
        is_metal(e, n_electron_per_cell=3)
    with pytest.raises(InvalidInputError, match="elektron sayisi"):
        fermi_level(e, n_electron_per_cell=3)
    with pytest.raises(InvalidInputError, match="n_bin"):
        density_of_states(e, n_bin=1)
    with pytest.raises(InvalidInputError, match="bant duz"):
        structure_ratio(np.full((10, 1), -0.3))
