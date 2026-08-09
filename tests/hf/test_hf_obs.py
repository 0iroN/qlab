"""Izgara uzerinde orbital ve yogunluk degerlendirmesi.

Bu dosyanin ana dogrulama fikri: izgarada sayisal olarak hesaplanan ortusme
matrisi, motorun analitik overlap_matrix'ine oturmalidir. Boylece yeni kod
Modul 1B'nin dogrulanmis motoruna baglanir ve bagimsiz bir referans gerekmez.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qlab.engine.hf.basis import BasisFunction, PrimitiveGaussian, build_basis
from qlab.engine.hf.integrals import overlap_matrix
from qlab.engine.hf.scf import solve
from qlab.errors import (
    InvalidInputError,
    SCFConvergenceError,
    UnsupportedSystemError,
)
from qlab.observable import hf_obs
from qlab.observable.hf_obs import (
    VolumeGrid,
    _symmetry_blocks,
    electron_density,
    evaluate_basis_function,
    evaluate_orbital,
    free_atom_density_matrix,
    promolecule_density,
)
from qlab.types import Atom, Molecule


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
    oldugundan ANALITIK integralleri sifirdir.

    DIKKAT -- burada iki ayri kosul var, karistirilmasin:
      (a) Analitik sifir yalnizca simetriden gelir ve her izgarada gecerlidir.
      (b) AYRIK toplamin 1e-16'ya (makine epsilonuna) DUSMESI ayrica
          izgaranin x=0'a gore AYNALI olmasini gerektirir. Bu fixture'da
          oyle: suyun uc atomu da x=0'da oldugu icin VolumeGrid.around'un
          sinirlayici kutusu x'te simetrik cikiyor, MO 4 tam olarak tek
          pariteli (yalniz O 2p_x), MO 3 tam cift pariteli, ve toplam
          terim terim goturuyor.
    Aynali olmayan bir kutuda ya da ayna duzlemine hizali olmayan bir
    molekulde analitik deger yine sifir olurdu ama ayrik yaklasim sadece
    olagan izgara yakinsama hiziyla (bu dosyadaki 1e-8/1e-10 marjlari
    mertebesinde) yaklasirdi, 1e-16'ya DUSMEZDI. Bu testin siki sinnri
    o yuzden bu geometriye ozguddur; baska bir molekule genisletirken
    once olcum yapilmali.

    Sinir: abs=1e-12, olculen 3.706e-16'ya ~2700x marj. Baglayicidir:
    gercekci indeks/sutun hatalari 1.38e-02 ve 1.62e-03 mertebesinde
    sapma verir, yani sinnrin dokuz kat ustunde.
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


def test_h2_yogunlugu_tam_iki_elektron():
    """Hidrojende cekirdek/degerlik ayrimi yok; izgara tam yakinsar.

    Olculdu: n=80 ve n=120 izgaralarda int(rho) = 2.000000.

    MARJ GEREKCESI (duzeltildi -- olculdu, tahmin degil): bu sinirin (abs=
    1e-5) daha once "sayisal izgara gurultusune (~1e-8) karsi ~1000x marj"
    diye aciklanmasi YANLIS KIYASTI -- dogru soru sinirin GERCEKCI BIR
    HATAYA karsi ne kadar marj biraktigidir. Iki capraz-terim bozulmasi
    (kosegen-only kontraksiyon, kosegen-disi isaret ters cevirme -- ayni iki
    mutasyon test_yogunluk_orbital_toplamiyla_ortusuyor testinde de
    kullanilir) burada olculdu: int(rho) sirasiyla 1.204911 ve 0.409823
    veriyor, yani 2.0'dan sapma 0.795089 ve 1.590177 -- sinirin cok
    ustunde, marj ~7.95e4x ve ~1.59e5x.

    Ayrica denendi ve ONEMLI BIR SINIRLAMA bulundu: yogunluk matrisinin iki
    satirini yer degistirmek (index-confusion tipi olasi bir hata) H2/STO-3G
    icin OLCULEBILIR HICBIR sapma URETMEZ -- integral tam olarak bit-bit ayni
    kalir (fark 0.0). Sebebi mekanik degil, molekulun kendi simetrisi: H2'nin
    bag orbitali iki H merkezinde esit buyuklukte esit isaretli katsayi tasir,
    bu yuzden P00=P01=P10=P11 makine hassasiyetine kadar esittir (olculen
    matris farki 2.22e-16); satir degisimi bu dorde-esit matrisi degistirmez.
    Yani bu sinir buyuk capraz-terim bozulmalarini (~1e5x marjla) rahatlikla
    yakalar ama H2'nin kendi simetrisi yuzunden bir satir/sutun yer-degistirme
    turu index-confusion hatasina karsi HICBIR koruma saglamaz -- bu hata
    sinifi icin marj bir sayi degil, sifir etki (test hicbir zaman kirmizi
    olmaz).
    """
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    result = solve(molecule, "sto-3g")
    grid = VolumeGrid.around(molecule, padding=7.0, points_per_axis=80)
    field = electron_density(result, grid)
    integral = float(field.values.sum() * grid.volume_element)
    assert integral == pytest.approx(2.0, abs=1e-5, rel=0.0)


def test_yogunluk_her_yerde_negatif_degil():
    """Yogunluk hicbir izgara noktasinda (float gurultusu disinda) negatif degil.

    Bu gercek bir fiziksel ozelliktir: rho(r) = phi(r)^T P phi(r) ve
    P = 2 C_occ C_occ^T pozitif yari-tanimli oldugundan rho(r) >= 0 her
    noktada teorik olarak gecerlidir. Bu assertion o yuzden korunuyor.

    AMA BU TEST SU IKI HATA SINIFINI YAKALAMAZ (olculdu, .superpowers/sdd/
    b2-task-4-fix-report.md'de dogrulandi): kosegen-only kontraksiyon (capraz
    terimler atlanmis) ve kosegen-disi isaret ters cevirme. Ikisi de bu
    izgarada minimumu NEGATIF YAPMAZ:

        mutasyon              H2 min (n=40)   H2O min (n=60)
        kosegen-only           4.218e-14       4.463e-21
        kosegen-disi isaret     3.649e-14       4.465e-21

    Sebep yapisaldir: np.argmin(field.values) bu izgarada HER ZAMAN kutunun
    en uzak kosesine duser (H2'de olculen konum (-5,-5,-5) bohr) -- molekulden
    cok uzakta, tum baz fonksiyonlarinin sondugu ve yogunlugun kosegen
    terimlerince (P_uu * phi_u(r)^2, herhangi bir P icin kare oldugundan
    kosegensel olarak negatif OLAMAZ) domine edildigi bir nokta. Capraz terim
    bozulmasi orada gorunmez.

    test_yogunluk_orbital_toplamiyla_ortusuyor bu boslugu kapatir: orbital
    yolu (evaluate_orbital, MO katsayilariyla) yogunluk-matrisi kontraksiyonuna
    hic dokunmadigindan, capraz terim bozulmasi iki ifadeyi ayristirir ve o
    test kirmizi olur.
    """
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    result = solve(molecule, "sto-3g")
    grid = VolumeGrid.around(molecule, padding=5.0, points_per_axis=40)
    field = electron_density(result, grid)
    assert field.values.min() >= -1e-14


def test_yogunluk_orbital_toplamiyla_ortusuyor(h2o):
    """Yogunluk-matrisi kontraksiyonu, bagimsiz orbital-toplami ifadesiyle ortusuyor.

    Kapali kabuk RHF yogunlugunun IKI bagimsiz ifadesi vardir:

        rho(r) = SUM_uv P_uv phi_u(r) phi_v(r)      (yogunluk-matrisi yolu,
                                                       electron_density'nin
                                                       kullandigi)
        rho(r) = 2 * SUM_i(occ) psi_i(r)^2           (orbital-toplami yolu)

    Ikinci ifade P = 2 C_occ C_occ^T ozdesliginden dogar ve `evaluate_orbital`
    ile TAMAMEN FARKLI bir kod yolundan (MO katsayilari, yogunluk matrisi
    degil) hesaplanir. Bu yuzden yogunluk-matrisi kontraksiyonundaki bir
    capraz terim bozulmasi (kosegen-only kontraksiyon veya kosegen-disi
    isaret ters cevirme) orbital yolunu HIC ETKILEMEZ ama yogunluk-matrisi
    yolunu buyuk olcude bozar -- iki ifade ayrisir ve bu test kirmizi olur.
    (Ispat: .superpowers/sdd/b2-task-4-fix-report.md -- her iki mutasyon da
    src/qlab/observable/hf_obs.py'ye gecici uygulanip bu testin kirmiziya
    dondugu dogrulandi.)

    test_yogunluk_her_yerde_negatif_degil bunu YAKALAMAZ (bkz. o testin
    dokstring'i): oradaki np.argmin izgaranin uzak kosesine duser, capraz
    terim bozulmasi orada gorunmez. Bu test o bosluk icin var; H2O kullanilir
    (p fonksiyonlari var, acisal yol da zincirlenir -- H2/STO-3G tamamen
    s-tipi).

    Olculdu (bu fixture'da, padding=6.0, n=60): iki ifade arasindaki maks
    mutlak fark 3.553e-15 (o noktada rho=24.09, yani goreli fark ~1.13e-15)
    -- float64 makine epsilonu (2.22e-16) mertebesinde, iki farkli toplama
    sirasinin (einsum kontraksiyonu vs orbital-basi kare toplami) beklenen
    yuvarlama gurultusu, fiziksel bir uyusmazlik degil.

    Sinir: atol=1e-10, rtol=1e-10 (np.testing.assert_allclose varsayilan
    rtol=1e-7'ye guvenilmedi, ikisi de acikca verildi). Olculen mutlak sapmaya
    (3.553e-15) marj ~2.8e4x, olculen goreli sapmaya (1.13e-15) marj ~8.8e4x.
    Capraz terim bozulmasinin urettigi sapma (yogunlugun kendisi mertebesinde,
    yani tipik olarak O(1) ve ustu -- bkz. baglama kaniti) bu sinirin
    dokuz-on mertebe ustundedir.
    """
    molecule, _ = h2o
    result = solve(molecule, "sto-3g")
    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=60)
    field = electron_density(result, grid)

    n_occ = molecule.n_electrons // 2
    orbital_density = np.zeros(grid.shape, dtype=float)
    for i in range(n_occ):
        orbital_density += evaluate_orbital(result, i, grid) ** 2
    orbital_density *= 2.0

    np.testing.assert_allclose(field.values, orbital_density, atol=1e-10, rtol=1e-10)


def test_serbest_hidrojen_yogunlugu_tam_bir_elektron():
    """Minimal bazda H'nin tek baz fonksiyonu var; oz-ortusmesi 1'dir.

    Olculdu: padding=8.0, n=80 izgarada int = 0.99999999998436, yani 1.0'dan
    sapma -1.56e-11.

    SINIR (abs=1e-5) NE YAKALAR, NE YAKALAMAZ: bu test bir OLCEK/BAGLANTI
    testidir. Yakalar: isgal sayisinin yanlis olmasi (H'ye 2 elektron
    konulmasi -> sapma 1.0, sinnrin 1e5 kati), hacim elemaninin veya izgara
    toplaminin yanlis olmasi (ayni mertebe). YAKALAMAZ: kontraksiyon
    normalizasyonunun dusurulmesi -- o hata H'nin oz-ortusmesini 1 yerine
    0.99999999089 yapar, yani integrali yalnizca 9.1e-9 kaydirir ve bu sinnrin
    3 mertebe altindadir. O kirilma test_izgara_ortusmesi_analitik_ortusmeye_
    oturuyor testinin atol=1e-8 sinniriyla ayrica kilitlidir; burada tekrar
    korunuyormus gibi yazilmamalidir.
    """
    molecule = Molecule.from_string("H 0 0 0")
    grid = VolumeGrid.around(molecule, padding=8.0, points_per_axis=80)
    field = promolecule_density(molecule, "sto-3g", grid)
    integral = float(field.values.sum() * grid.volume_element)
    assert integral == pytest.approx(1.0, abs=1e-5, rel=0.0)


def test_h2_promolekulu_iki_elektron():
    """Iki serbest H atomu, etkilesmeden ust uste bindirilir.

    Olculdu: padding=7.0, n=80 izgarada int = 1.99999999460097 (sapma
    -5.40e-09); n=120'de 1.99999999361885. Sapma izgara Riemann toplamindan
    gelir, atom yogunlugundan degil: analitik elektron sayisi Tr(P S) her
    atom icin TAM 1'dir (bkz. test_serbest_atom_elektron_sayisi_analitik_
    olarak_tam).

    Sinir (abs=1e-5) olculen sapmaya ~1850x marj birakir. Yakalar: bir atomun
    yogunluk blogunun hic yazilmamasi (sapma 1.0) veya blok kosegen
    yerlestirmesinin kaymasi. YAKALAMAZ: atomlar arasi capraz terimlerin
    yanlislikla eklenmesi -- promolekul TANIMI geregi bloklar arasi terim
    tasimaz ve o terimler eklenirse bu integral H2'de ~0.6 elektron artardi,
    yani aslinda yakalar; ama bu, tanimin ihlali olurdu, izgara hatasi degil.
    """
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    grid = VolumeGrid.around(molecule, padding=7.0, points_per_axis=80)
    field = promolecule_density(molecule, "sto-3g", grid)
    integral = float(field.values.sum() * grid.volume_element)
    assert integral == pytest.approx(2.0, abs=1e-5, rel=0.0)


def test_promolekul_negatif_degil():
    """Promolekul yogunlugu hicbir noktada negatif degil.

    P = SUM_i n_i c_i c_i^T ve tum n_i >= 0 oldugundan P pozitif yari-tanimli,
    dolayisiyla rho(r) = v^T P v >= 0 her noktada teorik olarak gecerlidir.

    BU TESTIN KAPSAMI DARDIR (olculdu): H2 promolekulunun minimumu bu izgarada
    +7.00e-14'tur ve bu modulde denenen HICBIR mutasyon onu negatif yapmaz --
    naif kosegen isgal de (min +7.0e-14 mertebesi), S-ortonormalligin atlanmasi
    da pozitif kalir, cunku her ikisi de P'yi pozitif yari-tanimli birakir.
    Yakaladigi tek sinif isgal sayisinin negatife dusmesidir (orn. aufbau
    doldurmasinda `remaining` isaretinin ters cevrilmesi). Gercek koruma
    test_serbest_atom_dogal_isgalleri_... testindedir.
    """
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    grid = VolumeGrid.around(molecule, padding=5.0, points_per_axis=40)
    field = promolecule_density(molecule, "sto-3g", grid)
    assert field.values.min() >= -1e-14


def test_serbest_atom_kuresel_simetrik():
    """Kesirli isgal p kabugunu esit doldurur; sonuc kuresel olmali.

    Testin adi "kuresel" diyor; yalnizca uc EKSEN yonune bakmak bunu
    kanitlamaz (eksenlerde esit ama kosegende farkli bir alan da o testi
    gecerdi). Bu yuzden ikinci bir olcum var: indeks-yaricapinin karesi 9 olan
    ALTI nokta -- uc eksen yonu (3,0,0) ve uc kosegen yon (1,2,2) tipinde --
    ayni degeri vermek zorunda.

    Olculdu (O/STO-3G, padding=5.0, n=41):
      merkezden 2.0 bohr (indeks 8), eksenler: x = y = z = 1.798872147155e-02,
        goreli fark TAM 0.0 (bit-bit ayni)
      indeks-yaricap^2 = 9 olan alti nokta: 7.008797261506e-01, aralarindaki
        maks goreli sapma 3.17e-16 (float yuvarlamasi)
    Tam esitlik tesaduf degil: uc kartezyen p yonune ayni radyal katsayi ve
    ayni kesirli isgal (4/3) verilir, s fonksiyonlari zaten izotropiktir.

    SINIR (rel=1e-10 eksenlerde, rel=1e-12 esit-yaricap kumesinde) NE
    YAKALAR: kuresel ortalamanin hic yapilmamasi. Olculdu
    -- p kabugunu 2p_x/2p_y'ye yigip 2p_z'yi bos birakan mutasyonda ayni
    noktalarda x = 2.369011950686e-02, z = 6.585925400923e-03, yani goreli
    fark 7.22e-01: sinnrin 7.2e9 kati. Bu mutasyonun elektron sayisini
    DEGISTIRMEDIGINI de olctuk (int = 9.922668, dogru degerle ayni), yani
    kureselligi yalnizca bu test koruyor.
    """
    molecule = Molecule.from_string("O 0 0 0")
    grid = VolumeGrid.around(molecule, padding=5.0, points_per_axis=41)
    field = promolecule_density(molecule, "sto-3g", grid)
    orta = 41 // 2
    yaricap = 8
    x_yonu = field.values[orta + yaricap, orta, orta]
    y_yonu = field.values[orta, orta + yaricap, orta]
    z_yonu = field.values[orta, orta, orta + yaricap]
    assert x_yonu == pytest.approx(y_yonu, rel=1e-10, abs=0.0)
    assert x_yonu == pytest.approx(z_yonu, rel=1e-10, abs=0.0)

    esit_yaricap = [(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 2), (2, 1, 2), (2, 2, 1)]
    degerler = [field.values[orta + a, orta + b, orta + c] for a, b, c in esit_yaricap]
    for deger in degerler[1:]:
        assert deger == pytest.approx(degerler[0], rel=1e-12, abs=0.0)


def test_promolekul_provenance_kuresel_ortalamayi_beyan_ediyor():
    """Referans atom gercek taban durum DEGILDIR; bu yaziya dokulmek zorunda."""
    molecule = Molecule.from_string("O 0 0 0")
    grid = VolumeGrid.around(molecule, padding=4.0, points_per_axis=20)
    field = promolecule_density(molecule, "sto-3g", grid)
    metin = " ".join(field.provenance.approximations).lower()
    assert "kuresel" in metin
    ihmal = " ".join(field.provenance.neglected).lower()
    assert "taban durum" in ihmal, "gercek taban durumdan sapma ihmal listesinde olmali"
    assert field.provenance.diagnostics["expected_electrons"] == 8


_SERBEST_ATOM_ISGALLERI: dict[str, list[float]] = {
    # Kuresel ortalamali, kesirli isgalli serbest atom: aufbau ile doldurulan
    # kabuk kesirli olarak ve esit sekilde uc kartezyen p yonune dagitilir.
    "H": [1.0],
    "He": [2.0],
    "Li": [2.0, 1.0],
    "C": [2.0, 2.0, 2 / 3, 2 / 3, 2 / 3],
    "N": [2.0, 2.0, 1.0, 1.0, 1.0],
    "O": [2.0, 2.0, 4 / 3, 4 / 3, 4 / 3],
    "F": [2.0, 2.0, 5 / 3, 5 / 3, 5 / 3],
}


@pytest.mark.parametrize("symbol", sorted(_SERBEST_ATOM_ISGALLERI))
def test_serbest_atom_elektron_sayisi_analitik_olarak_tam(symbol):
    """Serbest atom yogunlugu TAM Z elektron tasir -- izgaradan bagimsiz olcum.

    Elektron sayisi int rho = Tr(P S)'dir ve bu izgaradan bagimsizdir. Orbital
    katsayilari S-ORTONORMAL oldugu surece (C^T S C = I) Tr(P S) = SUM_i n_i
    ozdesligi tam olarak saglanir; testin olctugu sey budur.

    Olculdu: yedi element icin |Tr(P S) - Z| en fazla 2.665e-15 (O). Sinir
    abs=1e-12, marj ~375x.

    NE YAKALAR (olculdu -- MUTASYON: blok icindeki genellesmis ozdeger
    problemi yerine duz np.linalg.eigh, yani orbitallerin S yerine Oklit
    normuyla normalize edilmesi):
      Li  Tr(PS) = 3.135216  (sapma +0.135216 = sinnrin 1.35e11 kati)  KIRMIZI
      C,N,O,F, H, He: sapma TAM 0
    Yani bu test, S-ortonormallik hatasini YALNIZCA kismen dolu bir simetri
    blogu olan atomda (burada Li'nin 2 boyutlu s blogu, 3 elektron) yakalar.
    Blok TAMAMEN doluysa isgal edilen alt uzay tum blogu kaplar, C C^T = S^-1
    olur ve iz her durumda dogru cikar. Ayni hata O'da elektron sayisini
    bozmadigi halde YOGUNLUGU bozar; onu test_serbest_atom_dogal_isgalleri_...
    yakalar. Iki test birbirinin yerine gecmez.
    """
    molecule = Molecule.from_string(f"{symbol} 0 0 0")
    basis = build_basis(molecule, "sto-3g")
    density = free_atom_density_matrix(symbol, "sto-3g")
    sayim = float(np.trace(density @ overlap_matrix(basis)))
    assert sayim == pytest.approx(molecule.n_electrons, abs=1e-12, rel=0.0)


@pytest.mark.parametrize("symbol", sorted(_SERBEST_ATOM_ISGALLERI))
def test_serbest_atom_dogal_isgalleri_beklenen_degerlerde_ve_pauli_siniri_icinde(symbol):
    """P S'nin ozdegerleri = dogal isgal sayilari; hicbiri 2'yi asamaz (Pauli).

    P = SUM_i n_i c_i c_i^T ve C^T S C = I ise P S'nin ozdegerleri tam olarak
    {n_i}'dir. Bu, yogunluk matrisinin SEKLINI dogrudan sinar -- elektron
    sayisi (izi) ayni kalsa bile. Ozdegerler simetrik S^(1/2) P S^(1/2)
    uzerinden hesaplanir (ayni spektrum, gercel ve kararli).

    Olculdu: yedi element icin |dogal isgal - beklenen| en fazla 1.110e-15.
    Sinir abs=1e-12, marj ~900x.

    NE YAKALAR (iki mutasyon da olculdu):
      (1) NAIF KOSEGEN ISGAL, P = diag(2, 2, 4/3, 4/3, 4/3): STO-3G'de 1s ve
          2s DIK DEGILDIR (O icin S_12 = 0.236704), bu yuzden bu matris bir
          Slater determinantinin yogunlugu DEGILDIR. Elektron sayisi yine de
          dogru cikar (Tr(P S) = 8.000000, cunku S'nin kosegeni 1'dir) --
          yani bir onceki test bunu YAKALAMAZ. Dogal isgaller ise
          {1.333, 1.333, 1.333, 1.527, 2.473} olur: 2.473 > 2, Pauli ihlali,
          beklenenden 0.473 sapma = sinnrin 4.7e11 kati. KIRMIZI.
          Ayni mutasyon C'de 2.497, N'de 2.470, F'de 2.476, Li'de 2.105 verir.
      (2) Duz np.linalg.eigh (S-ortonormallik atlanmis): O icin en buyuk
          dogal isgal 2.473408, C 2.496725, N 2.470076, F 2.475980,
          Li 2.325030. Hepsi KIRMIZI.
    Dogru yol her elementte en buyuk isgali 2'nin 6.7e-16 altinda/ustunde
    birakir.
    """
    basis = build_basis(Molecule.from_string(f"{symbol} 0 0 0"), "sto-3g")
    overlap = overlap_matrix(basis)
    density = free_atom_density_matrix(symbol, "sto-3g")

    ozdeger, vektor = np.linalg.eigh(overlap)
    kok_s = vektor @ np.diag(np.sqrt(ozdeger)) @ vektor.T
    dogal = np.sort(np.linalg.eigvalsh(kok_s @ density @ kok_s))[::-1]

    beklenen = _SERBEST_ATOM_ISGALLERI[symbol]
    beklenen = np.array(beklenen + [0.0] * (len(dogal) - len(beklenen)))
    np.testing.assert_allclose(dogal, beklenen, atol=1e-12, rtol=0.0)
    assert dogal.max() <= 2.0 + 1e-12, "Pauli: bir orbital 2 elektrondan fazlasini tasiyamaz"


def test_serbest_atom_yogunlugu_lityum_katyonunda_solve_ile_ortusuyor():
    """Serbest atom SCF'i, motorun DOGRULANMIS solve()'una zincirlenir.

    Bu modulun serbest atom cozucusu kendi Fock dongusunu kosar; ic tutarlilik
    testleri (iz, dogal isgaller) onun DOGRU FIZIGI cozdugunu tek basina
    kanitlamaz. Zincir noktasi Li+ (yuk +1, 2 elektron): kapali kabuktur, yani
    motorun PySCF'e karsi dogrulanmis solve()'u onu cozebilir; ve s blogu
    KISMEN doludur (2 fonksiyon, 1 dolu orbital), yani S-ortonormallik
    gercekten devrededir. He (2 elektron, tek fonksiyon) ikinci, onemsiz ama
    bedava bir kontrol noktasidir.

    Olculdu: maks|P_serbest - P_solve| = 3.353e-11 (Li+), 0.0 (He). Li+
    sapmasi bizim yakinsamamizdan gelmiyor -- kendi dongumuz sabit noktaya
    otururken (iterasyonlar arasi degisim 0.0'a iniyor) fark 3.353e-11'de
    sabit kaliyor; solve()'un kendi yakinsama sinirlari (energy_tolerance
    1e-14, residual_tolerance 1e-12 ile bile) buraya kadar geliyor.

    Sinir atol=1e-9, olculene ~30x marj. Yakalar (olculdu, hepsi KIRMIZI):
    genellesmis ozdeger problemi yerine duz eigh, naif kosegen isgal, aufbau
    siralamasinin ters cevrilmesi.

    NE YAKALAMAZ -- ve bu sinir bilincli olarak yaziliyor: iki taraf da AYNI
    Fock kurulumunu (scf._fock_matrix) ve ayni S^(-1/2) ortogonallestirmesini
    kullanir, cunku bu formullerin ikinci bir kopyasi tutulmamistir. Olculdu:
    degisim katsayisini 0.5'ten 1.0'a cikaran mutasyon bu dosyadaki 24 testin
    HICBIRINI kirmaz -- iki taraf birlikte kayar. O formulu koruyan sey
    motorun kendi PySCF'e karsi dogrulanmis testleridir (ayni mutasyon
    tests/hf altinda 28 test kiriyor, olculdu). Bu test paylasilan formulu
    degil, promolekule OZGU parcayi zincirler: simetri bloklamasi, aufbau ve
    kesirli isgal atamasi, S-ortonormallik.
    """
    for spec, charge in [("Li 0 0 0", 1), ("He 0 0 0", 0)]:
        molecule = Molecule.from_string(spec, charge=charge)
        beklenen = solve(
            molecule, "sto-3g", energy_tolerance=1e-14, residual_tolerance=1e-12
        ).density
        bizim = free_atom_density_matrix(
            molecule.atoms[0].symbol, "sto-3g", n_electrons=molecule.n_electrons
        )
        np.testing.assert_allclose(bizim, beklenen, atol=1e-9, rtol=0.0)


def test_promolekul_izgara_cekirdek_sinnrini_beyan_ediyor():
    """BEYAN EDILEN SINIR -- kod hatasi degil (Task 4'un cekirdek siniri burada da).

    Serbest oksijenin ANALITIK elektron sayisi tam 8'dir (yukaridaki iz
    testi). Izgara integrali degildir ve iki YONE birden sapar:

      padding  n     aralik(bohr)   int(rho)
      6.0      60    0.2034         7.696685   (eksik -- cekirdek cozulmuyor)
      6.0     120    0.1008         7.990173   (duzeliyor)
      5.0      41    0.2500         9.922668   (FAZLA!)

    Ucuncu satir bir hata degil, Riemann toplaminin cusp'ta davranisidir: o
    izgarada bir nokta TAM cekirdegin uzerine duser ([-5,5] araligi, 41 nokta
    -> indeks 20 tam 0.0'da) ve 1s'in tepe degeri hucre hacmiyle carpilinca
    toplami yukari kacirir. Yani promolekul yogunlugunun izgara integrali,
    hidrojen disi atomlarda gerceklen elektron sayisinin ne alt ne ust
    siniridir -- gorsellestirme icin uygundur, integre nicelik icin DEGILDIR.
    Provenance bunu yazmak zorundadir.

    Sinirlar: abs=0.01 -- olculen degerler yukaridaki tabloda; test, bu uc
    sayidan herhangi biri 0.01'den fazla kayarsa kirilir. 60 -> 120
    iyilesmesi (0.303 -> 0.010 eksik) ve n=41'deki 1.92 elektronluk FAZLA,
    ikisi de sinnrin cok ustunde; yani bu test hem "cekirdek eksigi
    kayboldu" hem de "cusp asimi kayboldu" degisikliklerini yakalar.
    """
    molecule = Molecule.from_string("O 0 0 0")
    kaba = VolumeGrid.around(molecule, padding=6.0, points_per_axis=60)
    ince = VolumeGrid.around(molecule, padding=6.0, points_per_axis=120)
    cekirdek_uzerinde = VolumeGrid.around(molecule, padding=5.0, points_per_axis=41)

    def integral(grid):
        field = promolecule_density(molecule, "sto-3g", grid)
        return float(field.values.sum() * grid.volume_element)

    assert integral(kaba) == pytest.approx(7.696685, abs=0.01, rel=0.0)
    assert integral(ince) == pytest.approx(7.990173, abs=0.01, rel=0.0)
    assert integral(cekirdek_uzerinde) == pytest.approx(9.922668, abs=0.01, rel=0.0)
    assert integral(kaba) < integral(ince) < 8.0

    field = promolecule_density(molecule, "sto-3g", kaba)
    metin = field.provenance.expected_error.lower()
    assert "izgara" in metin and "cekirdek" in metin
    assert field.provenance.diagnostics["expected_electrons"] == 8
    assert field.provenance.diagnostics["integrated_electrons"] == pytest.approx(
        7.696685, abs=0.01, rel=0.0
    )


@pytest.mark.parametrize("symbol", ["Li", "O"])
def test_serbest_atom_yogunlugu_otelemeden_bagimsiz(symbol):
    """Serbest atom kendi cekirdeginde cozulur -- orijinde degil.

    Atomu kaydirip izgarayi AYNI vektorle kaydirinca alan degismemeli. Bu,
    tek-atom molekulunun cekirdek konumunun baz merkeziyle ayni tutuldugunu
    sinar: cekirdegi orijine sabitleyen bir hata (klasik "serbest atomu
    orijinde kur" hatasi) atomu yapay olarak kutuplastirirdi.

    Bu bosluk baska hicbir testte kapali degil: hidrojenin yogunluk matrisi
    tek fonksiyonlu oldugu icin (P = [[1]]) konumdan etkilenmez, oksijen
    testlerinde atom zaten orijindedir, toplanabilirlik testinde ise ayni
    hata her iki tarafta ayni sekilde olusur ve gorunmez.

    Olculdu: maks mutlak fark Li icin 5.329e-15, O icin 1.377e-14 (yogunluk
    tepe degerleri 2.52 ve 3.52; goreli 2.1e-15 ve 3.9e-15) -- kaydirilmis
    koordinatlarin float yuvarlanmasi. Sinir atol/rtol=1e-11, olculene ~700x
    marj; cekirdegi orijine sabitleyen mutasyon O(1e-2) ve ustu sapma verir.
    """
    orijinde = Molecule.from_string(f"{symbol} 0 0 0")
    izgara = VolumeGrid.around(orijinde, padding=5.0, points_per_axis=30)
    kayma = np.array([0.7, -1.3, 2.1])
    kaymis = Molecule((Atom(symbol, kayma),))
    kaymis_izgara = VolumeGrid(
        origin=izgara.origin + kayma, spacing=izgara.spacing, shape=izgara.shape
    )

    np.testing.assert_allclose(
        promolecule_density(kaymis, "sto-3g", kaymis_izgara).values,
        promolecule_density(orijinde, "sto-3g", izgara).values,
        atol=1e-11,
        rtol=1e-11,
    )


def test_promolekul_tek_tek_atomlarin_toplamina_esit():
    """Promolekulun TANIMI: atomlar birbirini gormez, alanlar toplanir.

    Ayni izgara nesnesi uc kez kullanilir: LiH'in promolekulu, sonra tek
    basina Li (kendi konumunda) ve tek basina H. Toplam esit olmak zorunda --
    bu, "etkilesimsiz ust uste binme" iddiasinin dogrudan olcumudur ve
    yogunluk matrisinin BLOK KOSEGEN oldugunu (atomlar arasi capraz terim
    tasimadigini) sinar.

    Molekul KASITLI olarak heteronukleer secildi: H2 gibi ozdes atomlu bir
    molekulde bir atomun yogunluk blogu digerinin yerine yazilsa bile fark
    olusmaz, yani blok atama hatasi gorunmez olurdu. LiH'te Li blogunun H'ye
    yazilmasi o noktada yogunlugu ~2-3 elektron/bohr^3 kaydirir.

    Olculdu: maks|rho_LiH - (rho_Li + rho_H)| = 0.0 (bit bit ayni, n=40 ve
    n=60'ta), yogunluk tepe degeri 4.91. Sinir atol/rtol=1e-12 -- olculen
    sifira karsi degil, GERCEKCI hataya karsi marj icin: atomlar arasi capraz
    terim eklenmesi veya blok atamasinin kaymasi O(0.1)-O(1) sapma verir,
    yani sinnrin 11-12 mertebe ustunde.
    """
    lih = Molecule.from_string("Li 0 0 0; H 0 0 1.6")
    grid = VolumeGrid.around(lih, padding=6.0, points_per_axis=40)

    butun = promolecule_density(lih, "sto-3g", grid)
    lityum = promolecule_density(Molecule((lih.atoms[0],)), "sto-3g", grid)
    hidrojen = promolecule_density(Molecule((lih.atoms[1],)), "sto-3g", grid)

    np.testing.assert_allclose(
        butun.values, lityum.values + hidrojen.values, atol=1e-12, rtol=1e-12
    )
    assert butun.provenance.diagnostics["expected_electrons"] == 4


def test_promolekul_notr_atomlardan_kurulur_molekulun_yukunden_bagimsiz():
    """Promolekul NOTR serbest atomlarin toplamidir; molekulun yuku dagitilmaz.

    Yuklu bir molekulde promolekulun elektron sayisi SUM Z'dir, molecule.
    n_electrons DEGILDIR -- ikisi yuk kadar farklidir. Task 6'nin fark
    haritasi bu yuzden yuklu sistemlerde net -yuk kadar integre olur; bu bir
    hata degil, tanimin sonucudur ve tanilamada gorunmek zorundadir.

    Olculdu: H2 icin ayni izgarada notr ve +1 yuklu molekulun promolekul
    integralleri BIT BIT AYNI (fark tam 0.0) -- yuk hicbir yere sizmiyor.
    """
    grid_molekulu = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    grid = VolumeGrid.around(grid_molekulu, padding=6.0, points_per_axis=40)

    notr = promolecule_density(grid_molekulu, "sto-3g", grid)
    katyon = promolecule_density(
        Molecule.from_string("H 0 0 0; H 0 0 0.74", charge=1), "sto-3g", grid
    )

    assert katyon.provenance.diagnostics["expected_electrons"] == 2
    assert float(np.abs(katyon.values - notr.values).max()) == 0.0


def test_serbest_atom_gecersiz_elektron_sayisi_reddediliyor():
    """Baz setinin tasiyabileceginden fazla (veya negatif) elektron reddedilir.

    Minimal bazda H'nin tek fonksiyonu vardir, yani en fazla 2 elektron
    tasiyabilir. Sessizce eksik yerlestirmek, elektron sayisi ozelligini
    fark ettirmeden bozardi.
    """
    with pytest.raises(InvalidInputError, match="elektron"):
        free_atom_density_matrix("H", "sto-3g", n_electrons=3)
    with pytest.raises(InvalidInputError, match="elektron"):
        free_atom_density_matrix("H", "sto-3g", n_electrons=-1)


def test_serbest_atom_yakinsamazsa_yukseltiliyor(monkeypatch):
    """Yakinsamamis bir yogunluk ASLA normal sonuc gibi donmez (spec Ilke 3).

    Iterasyon siniri 1'e cekilince dongu yakinsama kosulunu hic sinayamaz
    (kosul iteration > 1 gerektirir) ve SCFConvergenceError firlar. Bu yol
    normal kosuda hicbir zaman girilmez -- bu baz setindeki en yavas atom
    Li'dir ve 12 iterasyonda yakinsar, digerleri 2'de -- ama girildiginde
    sessizce yari-yakinsak bir yogunluk donmedigi burada kilitlenir.
    """
    monkeypatch.setattr(hf_obs, "_FREE_ATOM_MAX_ITERATIONS", 1)
    with pytest.raises(SCFConvergenceError):
        free_atom_density_matrix("Li", "sto-3g")


def test_serbest_atom_d_kabugu_reddediliyor():
    """d kabugu icin kartezyen bilesenler simetri geregi denk DEGILDIR.

    _symmetry_blocks, ayni l'nin kartezyen bilesenlerini dejenere kabul edip
    isgali esit dagitir. Bu s (1 bilesen) ve p (3 bilesen) icin dogrudur; d
    icin 6 kartezyen bilesen vardir, bunlar birbirine denk degildir ve kabuk
    kapasitesi 12 degil 10'dur. Baz katmani (basis.py) su an d kabugunu zaten
    reddediyor, ama o kural gevsetilirse burasi SESSIZCE yanlis bir yogunluk
    uretirdi; koruma bu yuzden burada da var ve dogrudan sinaniyor.
    """
    sahte_d = BasisFunction(
        center=np.zeros(3),
        angular=(2, 0, 0),
        primitives=(PrimitiveGaussian(exponent=1.0, coefficient=1.0),),
        atom_index=0,
    )
    with pytest.raises(UnsupportedSystemError, match="acisal momentum"):
        _symmetry_blocks([sahte_d])


def test_provenance_izgara_sinirini_beyan_ediyor(hf_energies):
    """Bu modulun kimligi: yaklasim gizlenmez, yaziya dokulur."""
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    result = solve(molecule, "sto-3g")
    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=60)
    field = electron_density(result, grid)
    metin = " ".join(field.provenance.approximations) + " " + field.provenance.expected_error
    assert "izgara" in metin.lower()
    assert "cekirdek" in metin.lower()
    assert field.provenance.diagnostics["expected_electrons"] == 10
    olculen = field.provenance.diagnostics["integrated_electrons"]
    assert 9.6 < olculen < 10.0, "cekirdek eksigi tanilamada gorunmeli"
