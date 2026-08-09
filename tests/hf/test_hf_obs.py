"""Izgara uzerinde orbital ve yogunluk degerlendirmesi.

Bu dosyanin ana dogrulama fikri: izgarada sayisal olarak hesaplanan ortusme
matrisi, motorun analitik overlap_matrix'ine oturmalidir. Boylece yeni kod
Modul 1B'nin dogrulanmis motoruna baglanir ve bagimsiz bir referans gerekmez.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import overlap_matrix
from qlab.engine.hf.scf import solve
from qlab.errors import InvalidInputError
from qlab.observable.hf_obs import VolumeGrid, evaluate_basis_function, evaluate_orbital
from qlab.types import Molecule


@pytest.fixture(scope="module")
def h2():
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    return molecule, build_basis(molecule, "sto-3g")


@pytest.fixture(scope="module")
def h2o():
    molecule = Molecule.from_string("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587")
    return molecule, build_basis(molecule, "sto-3g")


def test_grid_around_molekulu_kapsiyor(h2):
    molecule, _ = h2
    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=40)
    assert grid.shape == (40, 40, 40)
    ust = grid.origin + grid.spacing * (np.array(grid.shape) - 1)
    for atom in molecule.atoms:
        assert np.all(atom.position > grid.origin)
        assert np.all(atom.position < ust)


def test_hacim_elemani_eksen_araliklarinin_carpimi(h2):
    molecule, _ = h2
    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=40)
    beklenen = float(np.prod(grid.spacing))
    assert grid.volume_element == pytest.approx(beklenen, rel=1e-12, abs=0.0)


def test_izgara_ortusmesi_analitik_ortusmeye_oturuyor(h2):
    """Olculdu: n=80 izgarada maks fark 2.700e-09.

    Bu testin gucu buradan gelir: yeni degerlendirme kodu, Modul 1B'nin
    PySCF'e karsi dogrulanmis overlap_matrix'ine baglanir.

    TOLERANS GEREKCESI (olculdu, tahmin degil): planin ilk hali atol=1e-7
    diyordu ve bu YETERSIZDI. Kontraksiyon normalizasyonu dusurulunce maks
    fark 2.700e-09'dan yalnizca 1.181e-08'e cikiyor -- yani 1e-7 sinnri,
    planin tam da uyardigi hatayi GECIRIYORDU. atol=1e-8 dogru degere 3.7x
    marj birakir ve o kirigi yakalar (kirma kaniti:
    .superpowers/sdd/b2-task-2-fix-report.md). Bu tolerans gevsetilmemeli.
    """
    molecule, basis = h2
    grid = VolumeGrid.around(molecule, padding=7.0, points_per_axis=80)
    values = np.stack([evaluate_basis_function(b, grid) for b in basis])
    numeric = np.einsum("ixyz,jxyz->ij", values, values) * grid.volume_element
    np.testing.assert_allclose(numeric, overlap_matrix(basis), atol=1e-8, rtol=0.0)


def test_s_fonksiyonu_merkezde_maksimum(h2):
    molecule, basis = h2
    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=41)
    values = evaluate_basis_function(basis[0], grid)
    en_buyuk = np.unravel_index(np.argmax(values), values.shape)
    konum = grid.origin + grid.spacing * np.array(en_buyuk)
    np.testing.assert_allclose(konum, basis[0].center, atol=0.4, rtol=0.0)


def test_p_fonksiyonu_dugum_duzlemine_sahip():
    """Her p fonksiyonu kendi ekseninde merkez duzleminde tam olarak sifirdir.

    Onceki surum su geometrisinde (padding=5.0, points_per_axis=41) izgaranin
    kendiliginden fonksiyon merkezine denk gelmesine guveniyordu ve bu
    "if dugum.any():" korumasi icinde sessizce atlaniyordu. Olculdu: x (px)
    ve y (py) eksenlerinde izgara tam merkezden geciyor (1681 nokta, assertion
    calisiyor) ama z (pz) ekseninde hicbir izgara noktasi z=0 duzlemine denk
    gelmiyordu (min |z| = 8.3e-4) -- yani docstring'in soyledigi "pz tam
    olarak sifirdir" iddiasi hic test edilmiyordu.

    Duzeltme: izgarayi, test edilen fonksiyonun merkezine o eksende TAM
    hizalayacak sekilde yeniden kuruyoruz (origin[eksen] = merkez[eksen]),
    boylece indeks 0 her zaman duzlemin uzerine duser -- artik geometriye
    veya padding/points_per_axis secimine bagli bir sans meselesi degil.
    Ayrica "dugum.any()" artik bir on-kosul degil, kendi basina bir
    gereksinim: hicbir nokta duzleme denk gelmezse test kirmizi olur.
    """
    molecule = Molecule.from_string("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587")
    basis = build_basis(molecule, "sto-3g")
    p_fonksiyonlari = [b for b in basis if sum(b.angular) == 1]
    assert len(p_fonksiyonlari) == 3
    grid = VolumeGrid.around(molecule, padding=5.0, points_per_axis=41)
    for fonksiyon in p_fonksiyonlari:
        eksen = int(np.argmax(fonksiyon.angular))
        hizali_origin = grid.origin.copy()
        hizali_origin[eksen] = fonksiyon.center[eksen]
        hizali_grid = dataclasses.replace(grid, origin=hizali_origin)

        values = evaluate_basis_function(fonksiyon, hizali_grid)
        koordinat = hizali_grid.points[..., eksen] - fonksiyon.center[eksen]
        dugum = np.abs(koordinat) < 1e-12

        assert dugum.any(), (
            f"dugum duzlemi izgarada hic yakalanmadi (eksen {eksen}); "
            "hizalama basarisiz oldu"
        )
        assert np.abs(values[dugum]).max() < 1e-30


def test_p_p_blok_ortusmesi_analitik_ortusmeye_oturuyor(h2o):
    """p-p blok (sadece oksijenin 2p fonksiyonlari) sayisal-analitik zincire baglanir.

    Bu modulun ana dogrulama iddiasi -- izgara-integre ortusme analitik
    overlap_matrix'e (PySCF'e karsi dogrulanmis) oturuyor -- once yalnizca
    H2/STO-3G ile test edilmisti. H2/STO-3G'nin bazi tamamen s tipi oldugu
    icin bu, angular_factor yolunu (i+j+k>0, aci polinomu) hic
    zincirlemiyordu. Bu test o boslugu H2O/STO-3G ile kapatir.

    Olculdu (bu dosyada, padding=6.0):
      n= 60   tum matris maks fark 1.42e-01 | p-p blok maks fark 2.83e-07
      n= 80   tum matris maks fark 6.47e-02 | p-p blok maks fark 2.30e-12
      n=100   tum matris maks fark 3.23e-02 | p-p blok maks fark 9.04e-13

    Tum matris karsilastirmasi n arttikca YAKINSAMIYOR: oksijenin 1s
    cekirdek kabugu (us 130.7, genislik ~0.06 bohr) izgaranin
    cozunurlugunun cok altinda kaliyor -- bu yalnizca o satir/sutuna ozgu
    bir cozunurluk siniri, angular yolun dogruluguyla ilgisiz. p-p blogu
    ise mukemmel yakinsiyor (n=80'de 2.30e-12) cunku oksijenin 2p
    fonksiyonlari yayilmis (diffuse, cekirdek kabugu gibi dar degil).
    Karsilastirma bu yuzden KASITLI olarak p-p blokla sinirlandirilmistir:
    tum matrisle karsilastirmak cekirdek kabugu yuzunden hep basarisiz
    olurdu ve angular_factor yoluyla alakasiz bir cozunurluk sinirini
    olcerdi.

    (LiH/STO-3G'nin p-p blogu bu amac icin kullanilamaz: lityumun 2p'si
    cok daha yayilmis ve 6 bohr padding onun kuyrugunu kesiyor -- bu bir
    KUTU BOYUTU siniri, n arttirarak duzelmiyor, aksine n=80'de 1.30e-02'den
    daha kotulesebiliyor. O yuzden burada su/H2O kullanildi.)

    Sinir secimi: n=80'de olculen p-p maks fark 2.30e-12. atol=1e-10
    (~43x marj), rtol=0.0 -- ayni marj mertebesi
    test_izgara_ortusmesi_analitik_ortusmeye_oturuyor testindeki H2 sinirinin
    (atol=1e-7, olculen 2.70e-09, ~37x marj) mertebesiyle tutarlidir.
    """
    molecule, basis = h2o
    p_indeksleri = [i for i, b in enumerate(basis) if sum(b.angular) == 1]
    assert len(p_indeksleri) == 3

    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=80)
    values = np.stack([evaluate_basis_function(b, grid) for b in basis])
    numeric = np.einsum("ixyz,jxyz->ij", values, values) * grid.volume_element
    analytic = overlap_matrix(basis)

    numeric_pp = numeric[np.ix_(p_indeksleri, p_indeksleri)]
    analytic_pp = analytic[np.ix_(p_indeksleri, p_indeksleri)]
    np.testing.assert_allclose(numeric_pp, analytic_pp, atol=1e-10, rtol=0.0)


class TestVolumeGridAroundGirdiDogrulama:
    """VolumeGrid.around, gecersiz points_per_axis/padding degerlerini reddeder.

    Olculdu (duzeltmeden once):
      points_per_axis=1  -> sifira bolme, RuntimeWarning, spacing=inf
      points_per_axis=0  -> uyari YOK, spacing=[-10.,-10.,-11.4] (negatif),
                             shape=(0,0,0)
      points_per_axis=-1 -> uyari YOK, shape=(-1,-1,-1) ama points.shape
                             (0,0,0,3) -- dataclass'in kendi alani kendi
                             uretimiyle celisiyor

    Her uc deger de artik build/around asamasinda acikca reddediliyor.
    """

    def _molekul(self):
        return Molecule.from_string("H 0 0 0; H 0 0 0.74")

    @pytest.mark.parametrize("points_per_axis", [0, 1, -1])
    def test_points_per_axis_gecersiz_degerler_reddedilir(self, points_per_axis):
        with pytest.raises(InvalidInputError, match="points_per_axis"):
            VolumeGrid.around(
                self._molekul(), padding=6.0, points_per_axis=points_per_axis
            )

    def test_points_per_axis_iki_kabul_edilir(self):
        grid = VolumeGrid.around(self._molekul(), padding=6.0, points_per_axis=2)
        assert grid.shape == (2, 2, 2)
        assert np.all(grid.spacing > 0)

    def test_padding_dejenere_kutuyu_reddeder(self):
        """H2 molekulu x=y=0 dogrusunda yatar; padding=0 x,y'de sifir genislik verir."""
        with pytest.raises(InvalidInputError, match="padding"):
            VolumeGrid.around(self._molekul(), padding=0.0, points_per_axis=41)

    def test_padding_ters_kutuyu_reddeder(self):
        """Yeterince negatif padding, yuksek sinirin dusuk sinirin altina dusmesine yol acar."""
        with pytest.raises(InvalidInputError, match="padding"):
            VolumeGrid.around(self._molekul(), padding=-1.0, points_per_axis=41)

    def test_dogrudan_kurulan_izgara_da_dogrulanir(self):
        """__post_init__, around() disinda dogrudan kurulan izgaralari da korur."""
        with pytest.raises(InvalidInputError):
            VolumeGrid(origin=np.zeros(3), spacing=np.array([1.0, 1.0, -1.0]), shape=(3, 3, 3))
        with pytest.raises(InvalidInputError):
            VolumeGrid(origin=np.zeros(3), spacing=np.ones(3), shape=(3, 3, 1))


def test_homo_normu_kaba_izgarada_bile_bir(hf_energies):
    """Olculdu: su/STO-3G'de HOMO normu n=60, 90 ve 120'de alti basamakta 1.000000.

    Degerlik orbitalleri izgarada aninda yakinsar. Bu testin sikiligi bilinclidir:
    HOMO gorsellestirdigimiz seydir ve dogru olmak zorundadir.
    """
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    result = solve(molecule, "sto-3g")
    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=60)
    homo = evaluate_orbital(result, molecule.n_electrons // 2 - 1, grid)
    norm = float((homo**2).sum() * grid.volume_element)
    assert norm == pytest.approx(1.0, abs=1e-5, rel=0.0)


def test_cekirdek_orbitali_izgarada_yakinsamiyor_ve_bunu_soyluyoruz(hf_energies):
    """BEYAN EDILEN SINIR -- kod hatasi degil.

    Oksijenin 1s ussu 130.7, genisligi ~0.06 bohr. 6 bohr dolgulu kutuda 60
    nokta ~0.203 bohr aralik verir; cekirdek orbitali cozulemez.
    Olculdu: 1s normu n=60'ta 0.8599, n=120'de 0.9876.
    Test bu sinniri KILITLER: kaba izgarada eksik, ince izgarada duzeliyor.
    Biri gelip duzeltirse test kirmizi olur ve bu docstring guncellenir.
    """
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    result = solve(molecule, "sto-3g")
    kaba = VolumeGrid.around(molecule, padding=6.0, points_per_axis=60)
    ince = VolumeGrid.around(molecule, padding=6.0, points_per_axis=120)
    norm_kaba = float((evaluate_orbital(result, 0, kaba) ** 2).sum() * kaba.volume_element)
    norm_ince = float((evaluate_orbital(result, 0, ince) ** 2).sum() * ince.volume_element)
    assert norm_kaba == pytest.approx(0.8599, abs=0.01, rel=0.0)
    assert norm_ince == pytest.approx(0.9876, abs=0.01, rel=0.0)
    assert norm_kaba < norm_ince < 1.0


def test_farkli_simetrideki_orbitaller_izgarada_tam_dik(hf_energies):
    """3a1 ile 1b1 SIMETRI geregi tam diktir -- izgara yakinsamasi geregi DEGIL.

    Testin ilk adi "orbitaller izgarada dik"ti ve bu iddia YANLIS. Olculdu
    (su/STO-3G, padding=6.0, n=80), tum MO ciftleri uzerinde:
      cekirdek (indeks 0) iceren ciftler : maks 1.380e-02
      yalniz degerlik ciftleri           : maks 1.618e-03
      bu testin baktigi cift (3, 4)      : 3.706e-16
    Yani orbitaller bu izgarada genel olarak dik DEGILDIR; cekirdek kabugunun
    cozulememesi ortogonalligi bozar (bkz. asagidaki cekirdek testi). Eski
    abs=1e-4 siniri hem olculen degerden 12 mertebe gevsekti (dekoratif), hem
    de baska bir cift secilseydi KIRMIZI olurdu.

    (3, 4) cifti tam sifir verir cunku su/STO-3G'de indeks 3 = 3a1 ve
    indeks 4 = 1b1 (HOMO, duzlem disi p); farkli indirgenemez temsiller
    oldugundan carpimlarinin integrali izgara kalitesinden bagimsiz olarak
    sifirdir. Bu yuzden sinir makine hassasiyetine cekilebilir: abs=1e-12,
    olculen 3.706e-16'ya ~2700x marj. Aci polinomunda veya katsayi
    indekslemesinde bir hata bu simetriyi bozar ve test kirmizi olur.
    """
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    result = solve(molecule, "sto-3g")
    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=80)
    ucuncu = evaluate_orbital(result, 3, grid)
    dorduncu = evaluate_orbital(result, 4, grid)
    ortusme = float((ucuncu * dorduncu).sum() * grid.volume_element)
    assert ortusme == pytest.approx(0.0, abs=1e-12, rel=0.0)


def test_gecersiz_orbital_indeksi_reddediliyor():
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    result = solve(molecule, "sto-3g")
    grid = VolumeGrid.around(molecule, padding=4.0, points_per_axis=20)
    with pytest.raises(InvalidInputError, match="orbital"):
        evaluate_orbital(result, 99, grid)
    with pytest.raises(InvalidInputError, match="orbital"):
        evaluate_orbital(result, -1, grid)
