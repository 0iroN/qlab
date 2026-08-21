"""SCF (Roothaan-Hall) testleri: kapali kabuk RHF enerjisi PySCF ile karsilastirilir.

Referans: reference/hf_energies.json (PySCF, 9 sistem + bir 6-31G durumu).
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import overlap_matrix
from qlab.engine.hf.scf import (
    _diis_extrapolate,
    _orthogonalisation_matrix,
    solve,
)
from qlab.errors import SCFConvergenceError, UnsupportedSystemError
from qlab.types import Molecule


# pytest.approx BUYUK olani alir: max(abs, rel*expected). rel verilmezse
# varsayilani 1e-6'dir, yani "abs=1e-6" yazan bir iddia -74.96 hartree'lik su
# enerjisinde aslinda 7.5e-5'i zorlar -- yazilan sinir ile UYGULANAN sinir
# farklidir. Bu dosyadaki her iddiada IKI parametre de acikca verilir; sayilar
# olculmustur (bkz. .superpowers/sdd/b1-final-fix-report.md), tahmin degildir.


@pytest.mark.parametrize(
    "label", ["he", "h2_074", "h2_100", "h2_200", "lih", "hf_mol", "h2o", "nh3", "ch4"]
)
def test_total_energy_matches_pyscf(hf_energies, label):
    """Referans hakemligi: her sistem PySCF ile eslesmeli.

    Kabul kriteri 1e-6 hartree'dir; OLCULEN uyum bundan bes mertebe daha
    iyidir -- dokuz sistemin en kotusu h2_200'de 1.32e-11 hartree. Sinir
    bu yuzden olculen sapmanin ~7.6 kati olan 1e-10'da BAGLAYICI tutulur:
    1e-6'nin altinda kalan ama 1e-10'u asan bir gerileme (yakinsama
    kriterinin bozulmasi, integrallerde kayma) sessizce gecmesin diye.
    """
    record = hf_energies[label]
    result = solve(Molecule.from_string(record["atom"]), record["basis"])
    assert result.energy_total == pytest.approx(
        record["energy_total"], abs=1e-10, rel=0.0
    )


def test_631g_basis_also_matches(hf_energies):
    """Olculen sapma 2.95e-13 hartree; sinir ~10 kati (3e-12) ile baglayici."""
    record = hf_energies["h2_074_631g"]
    result = solve(Molecule.from_string(record["atom"]), "6-31g")
    assert result.energy_total == pytest.approx(
        record["energy_total"], abs=3e-12, rel=0.0
    )


def test_orbital_energies_match_pyscf(hf_energies):
    record = hf_energies["h2o"]
    result = solve(Molecule.from_string(record["atom"]), "sto-3g")
    np.testing.assert_allclose(
        result.orbital_energies, record["orbital_energies"], atol=1e-6, rtol=0.0
    )


def test_energy_decomposition_is_consistent(hf_energies):
    record = hf_energies["h2o"]
    result = solve(Molecule.from_string(record["atom"]), "sto-3g")
    assert result.energy_total == pytest.approx(
        result.energy_electronic + result.energy_nuclear, rel=1e-12
    )


def test_density_matrix_idempotency(hf_energies):
    """RHF yogunlugu icin PSP = 2P olmali."""
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    result = solve(molecule, "sto-3g")
    S = overlap_matrix(build_basis(molecule, "sto-3g"))
    np.testing.assert_allclose(
        result.density @ S @ result.density, 2 * result.density, atol=1e-8, rtol=0.0
    )


def test_electron_count_from_density(hf_energies):
    """tr(PS) elektron sayisini vermeli.

    Olculen sapma 7.11e-15 (10 elektron uzerinde, makine hassasiyeti
    mertebesinde); sinir 1e-13, yani ~14 kat pay. rel=0.0 olmadan
    varsayilan rel=1e-6, 10 elektronda 1e-5'lik bir bosluk birakirdi.
    """
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    result = solve(molecule, "sto-3g")
    S = overlap_matrix(build_basis(molecule, "sto-3g"))
    assert np.trace(result.density @ S) == pytest.approx(
        molecule.n_electrons, abs=1e-13, rel=0.0
    )


def test_variational_principle_larger_basis_never_higher():
    """Ayni molekul, ayni geometri: 6-31G enerjisi STO-3G'yi asamaz."""
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    assert solve(molecule, "6-31g").energy_total <= solve(molecule, "sto-3g").energy_total


def test_result_carries_provenance():
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")
    text = " ".join(result.provenance.approximations + result.provenance.neglected)
    assert "Born-Oppenheimer" in text
    assert "korelasyon" in text.lower()
    assert result.provenance.converged is True


def test_expected_error_names_the_dominant_error_not_the_secondary_one():
    """Panelin her HF sonucunda bastigi TEK satir, dogru sebebi isimlendirmeli.

    Onceki metin ("korelasyon enerjisi eksik -- bag enerjilerinde tipik
    olarak birkac eV") bu modulun KENDI olcumuyle celisiyordu:
    test_h2_binding_energy_overshoots_because_rhf_cannot_dissociate 14.6350
    eV'ye karsi deneysel 4.75 eV olcuyor, yani ~9.9 eV'lik hata. Baskin
    sebep eksik korelasyon degil, kisitli tek determinantin ayrisma
    kusurudur; korelasyon ters yonde ve daha kucuktur.
    """
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")
    beklenen_hata = result.provenance.expected_error.lower()

    assert "ayrisma kusuru" in beklenen_hata
    assert "14.6350" in result.provenance.expected_error
    assert "4.75" in result.provenance.expected_error
    # Korelasyon hala aniliyor -- ama ikincil etki olarak.
    assert "korelasyon" in beklenen_hata

    # Ayrisma kusuru / statik korelasyon ihmal listesinde de yer almali:
    # panelin "neyi hesaba katmadim" satiri bunu hic anmiyordu.
    ihmal = " ".join(result.provenance.neglected).lower()
    assert "statik korelasyon" in ihmal
    assert "ayrisma" in ihmal


def test_open_shell_rejected():
    with pytest.raises(UnsupportedSystemError, match="kapali kabuk"):
        solve(Molecule.from_string("H 0 0 0"), "sto-3g")


def test_non_convergence_raises_rather_than_returning(hf_energies):
    """En tehlikeli hata: yakinsamamis sonucun gecerli gibi sunulmasi."""
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    with pytest.raises(SCFConvergenceError) as excinfo:
        solve(molecule, "sto-3g", max_iterations=2, use_diis=False)
    assert excinfo.value.iterations == 2
    assert len(excinfo.value.energy_history) == 2
    assert excinfo.value.final_residual > 0


def test_converged_result_is_stationary(hf_energies):
    """Yakinsamis cozumde daha siki tolerans enerjiyi degistirmemeli.

    Olculen fark TAM SIFIR (iki kosu ayni sabit noktaya bit-bit ayni
    iniyor), yani bir "pay orani" hesaplanamaz. Bu yuzden yazilan sinir
    (1e-9) korunur ve yalnizca GERCEKTEN uygulanir hale getirilir:
    rel=0.0 olmadan varsayilan rel=1e-6, -1.1167 hartree'de 1.12e-6'lik
    bir bosluk birakiyordu -- iddia edilenden ~1000 kat gevsek. Siniri
    bit-esitligine (abs=0.0) cekmek, yakinsama yolundaki bir iterasyon
    farkinda kirilgan olurdu.
    """
    molecule = Molecule.from_string(hf_energies["h2_074"]["atom"])
    loose = solve(molecule, "sto-3g", energy_tolerance=1e-10)
    tight = solve(molecule, "sto-3g", energy_tolerance=1e-13, max_iterations=200)
    assert loose.energy_total == pytest.approx(tight.energy_total, abs=1e-9, rel=0.0)


def test_diis_reduces_iteration_count(hf_energies):
    """DIIS gercekten hizlandirmali -- sadece "calisiyor" yetmez."""
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    plain = solve(molecule, "sto-3g", use_diis=False, max_iterations=300)
    accelerated = solve(molecule, "sto-3g", use_diis=True, max_iterations=300)
    assert accelerated.n_iterations < plain.n_iterations
    # rel=0.0: pytest.approx alir buyuk olani (abs, rel*expected); rel
    # verilmezse varsayilan rel=1e-6 bu iddiayi ~1e-4 hartree'ye kadar
    # gevsetir. DIIS'in "sonucu degistirmedigi" iddiasi 1e-9'da BAGLAYICI
    # olmali; bu yuzden rel=0.0 acikca verilir (bkz. proje kisitlari).
    assert accelerated.energy_total == pytest.approx(
        plain.energy_total, abs=1e-9, rel=0.0
    )


def test_diis_and_plain_agree_on_all_systems(hf_energies):
    """Hizlandirma sonucu degistirmemeli, yalnizca yolu kisaltmali."""
    for label in ("he", "h2_074", "lih", "h2o"):
        record = hf_energies[label]
        molecule = Molecule.from_string(record["atom"])
        plain = solve(molecule, record["basis"], use_diis=False, max_iterations=300)
        accelerated = solve(molecule, record["basis"], use_diis=True)
        assert accelerated.energy_total == pytest.approx(
            plain.energy_total, abs=1e-9, rel=0.0
        )


def test_exception_reports_no_oscillation_when_converging_steadily(hf_energies):
    """Iterasyon siniri, enerji istikrarli sekilde azalirken asilirsa oscillating False olmali.

    ch4/sto-3g duz iterasyonda 12 adimda monoton yakinsar (Task 8 olcumu);
    5 iterasyonda kesilirse henuz bitmemistir ama enerji hala her adimda
    azalmaktadir -- salinim degil, sadece erken kesme.
    """
    molecule = Molecule.from_string(hf_energies["ch4"]["atom"])
    with pytest.raises(SCFConvergenceError) as excinfo:
        solve(molecule, "sto-3g", max_iterations=5, use_diis=False)
    assert excinfo.value.iterations == 5
    assert excinfo.value.oscillating is False


def test_exception_reports_oscillation_when_energy_alternates():
    """Gercek salinim: asiri gerilmis LiH (Li-H 4.0 Angstrom, core-guess ile
    baslangic) duz Roothaan iterasyonunda iki enerji degeri arasinda sonsuza
    kadar gidip gelir -- DIIS olmadan hicbir zaman durmaz. Bu, monoton
    yakinsamayla erken kesilen bir durumdan (yukaridaki test) niteliksel
    olarak farklidir ve dedektorun "True" kolunu gercekten tetikler.
    """
    molecule = Molecule.from_string("Li 0 0 0; H 0 0 4.0")
    with pytest.raises(SCFConvergenceError) as excinfo:
        solve(molecule, "sto-3g", max_iterations=20, use_diis=False)
    assert excinfo.value.oscillating is True


def test_diis_subspace_trimming_does_not_break_convergence(hf_energies):
    """DIIS alt uzayi varsayilan sinirini (8 cift) asan bir sistemde en eski
    ciftin atilmasi (trimming) yolu tetiklenmeli ve sonuc yine duz
    iterasyonla ayni fixed point'e ulasmali.

    ch4/sto-3g'de olculen DIIS iterasyon sayisi 11'dir (bkz. rapor,
    iterasyon tablosu): 8'i asar, yani _DIIS_MAX_HISTORY sinirina en az uc
    kez ulasilir ve trimming (fock_history.pop(0)) calisir. Bu, dokuz
    referans sistem + gerilmis LiH icinde trimming'in gercekten tetiklendigi
    TEK sistemdir -- diger DIIS testlerinde (su, lih, vb.) alt uzay 8'e hic
    ulasmaz, dolayisiyla bu test olmadan trimming dali test edilmemis kalir.
    """
    record = hf_energies["ch4"]
    molecule = Molecule.from_string(record["atom"])
    plain = solve(molecule, record["basis"], use_diis=False, max_iterations=300)
    accelerated = solve(molecule, record["basis"], use_diis=True, max_iterations=300)
    # >8 iterasyon, alt uzayin en az bir kez trimlendigini garanti eder
    # (fock_history her iterasyonda 1 buyur, 8'i astiginda en eski cift atilir).
    assert accelerated.n_iterations > 8
    assert accelerated.energy_total == pytest.approx(
        plain.energy_total, abs=1e-9, rel=0.0
    )


def test_diis_extrapolate_drops_oldest_pair_when_b_matrix_is_singular():
    """B tekillesirse en eski cift atilip kucultulmus alt uzayla tekrar
    denenmeli, cokmemeli veya NaN/Inf uretmemeli.

    Gercek bir molekuler sistemde bu yolu tetiklemek icin dokuz referans
    sistem VE kasten gerilmis, salinan bir LiH (Li-H 4.0 Angstrom) DIIS
    acikken denendi -- hicbiri tek bir tekil B olayi bile uretmedi (bkz.
    rapor, "singular-B retry" bolumu). Bu yuzden _diis_extrapolate
    dogrudan, kasten dogrusal bagimli (birebir ayni) iki hata vektoruyle
    test edilir: ilk iki cift ayni hata vektorune sahip (B'nin ilk 2x2
    blogu -- ve dolayisiyla tum genisletilmis sistem -- tekil olur),
    ucuncu cift farkli. Beklenen davranis: en eski (ilk) cift dusurulur,
    kalan iki ciftle (farkli hata vektorlerine sahip, tekil olmayan) sistem
    basariyla cozulur ve sonuc yalnizca o iki Fock'un bir kombinasyonudur.
    """
    duplicate_error = np.array([[1.0, 0.5], [0.5, -1.0]])
    independent_error = np.array([[0.1, 0.0], [0.0, 0.2]])
    fock_a = np.array([[1.0, 0.0], [0.0, 2.0]])
    fock_b = np.array([[3.0, 0.0], [0.0, 4.0]])
    fock_c = np.array([[5.0, 0.0], [0.0, 6.0]])

    result = _diis_extrapolate(
        [fock_a, fock_b, fock_c],
        [duplicate_error, duplicate_error, independent_error],
    )

    assert np.all(np.isfinite(result))
    # Dusen ciftin (fock_a) sonuca hic girmedigini dogrula: kalan iki
    # Fock'un (fock_b, fock_c) her koseginde AYNI karisim orani (c_b, c_c)
    # gecerli olmali -- fock_a katilsaydi bu oran koseginlerde farkli
    # cikardi (fock_a'nin kosegin degerleri de farkli olculerde).
    ratio_00 = (result[0, 0] - fock_b[0, 0]) / (fock_c[0, 0] - fock_b[0, 0])
    ratio_11 = (result[1, 1] - fock_b[1, 1]) / (fock_c[1, 1] - fock_b[1, 1])
    assert ratio_00 == pytest.approx(ratio_11, abs=1e-12, rel=0.0)


# ---------------------------------------------------------------------------
# Baz setinin kapasitesi


_H2 = "H 0 0 0; H 0 0 0.74"


@pytest.mark.parametrize(
    ("charge", "n_electrons"),
    [pytest.param(-4, 6, id="6e-2orbital"), pytest.param(-6, 8, id="8e-2orbital")],
)
def test_baz_setinin_tasiyamayacagi_elektron_sayisi_reddediliyor(charge, n_electrons):
    """H2/STO-3G'de iki orbital var; altı elektron oraya SIĞMAZ.

    ÖLÇÜLDÜ (düzeltmeden önce): `solve` sessizce çözüyordu, çünkü
    `_density_matrix`'teki `coefficients[:, :n_occupied]` dilimi numpy'da
    KIRPILIYOR — üç işgalli orbital istenip iki tane bulununca fazlası sessizce
    düşüyordu. Sonuç, üç ayrı kimyasal sistemin BİREBİR AYNI cevabı vermesiydi:

        yük −2 (4e): E = +0.923179, Tr(PS) = 4  <- dürüst, kapasiteye tam oturuyor
        yük −4 (6e): E = +0.923179, Tr(PS) = 4  <- 6 elektron iddia, 4 hesap
        yük −6 (8e): E = +0.923179, Tr(PS) = 4  <- 8 elektron iddia, 4 hesap

    `Tr(PS)` elektron sayısını verir; iki kırpılmış durumda molekülün beyan
    ettiği sayıyla hesabın kullandığı sayı TUTMUYOR ve hiçbir yerde bu
    söylenmiyordu. Bu deponun taksonomisinde "sessizce yanlış sonuç".
    """
    molecule = Molecule.from_string(_H2, charge=charge)
    assert molecule.n_electrons == n_electrons
    with pytest.raises(UnsupportedSystemError, match="baz"):
        solve(molecule, "sto-3g", stability=False)


def test_kapasiteye_TAM_oturan_sistem_kabul_ediliyor():
    """Sınır, kapasiteyi AŞANI reddetmeli; TAM dolduranı değil.

    H2²⁻: dört elektron, iki orbital, ikisi de çift işgalli. Ölçüldü:
    Tr(PS) = 4.0000000000, yani aritmetik dürüst. Enerji pozitif (+0.923179)
    ama bu bir hata değil — iki elektron antibağ orbitalinde, sistem bağlı
    değil. Reddetmek, doğru hesaplanan bir sonucu atmak olurdu.
    """
    molecule = Molecule.from_string(_H2, charge=-2)
    result = solve(molecule, "sto-3g", stability=False)
    basis = build_basis(molecule, "sto-3g")
    trace = float(np.trace(result.density @ overlap_matrix(basis)))
    assert trace == pytest.approx(4.0, abs=1e-10, rel=0.0)


@pytest.mark.parametrize(
    ("charge", "n_electrons"),
    [pytest.param(4, -2, id="eksi2e"), pytest.param(6, -4, id="eksi4e")],
)
def test_negatif_elektron_sayisi_reddediliyor(charge, n_electrons):
    """Elektron sayısı negatif olamaz; olursa numpy BAŞKA bir sistemi çözer.

    ÖLÇÜLDÜ (düzeltmeden önce): `n_occupied` negatif olunca dilim
    `coefficients[:, :-1]` hâline geliyor ve SONDAN sütun atıyor. Sonuç:

        yük +4 (−2e): E = −1.116759  <- NÖTR H2'nin enerjisinin ta kendisi
        yük +6 (−4e): E = +0.715104  <- sıfır elektronlu sistemin enerjisi

    Yani fiziksel olarak anlamsız bir girdi, başka bir molekülün doğru
    cevabıyla geri dönüyordu. Çift sayı oldukları için açık kabuk kontrolü de
    bunları geçiriyordu.
    """
    molecule = Molecule.from_string(_H2, charge=charge)
    assert molecule.n_electrons == n_electrons
    with pytest.raises(UnsupportedSystemError, match="elektron"):
        solve(molecule, "sto-3g", stability=False)


def test_sifir_elektronlu_sistem_hala_cozuluyor_ve_cekirdek_itmesini_veriyor():
    """Sıfır elektron bir KAPASITE ihlali değil; aritmetiği dürüst.

    H2²⁺ (iki çıplak proton) için elektronik enerji yok, toplam enerji
    yalnızca çekirdek itmesidir. Ölçüldü: E = +0.715104 ve 1/R = 1/1.398397 =
    0.715104 — birebir. Tr(PS) = 0, yani beyan edilen elektron sayısıyla
    hesaplanan tutuyor. Reddedilmemesinin sebebi bu: sınırın işi yanlış
    cevapları durdurmak, dejenere ama DOĞRU olanları değil.
    """
    molecule = Molecule.from_string(_H2, charge=2)
    result = solve(molecule, "sto-3g", stability=False)
    assert molecule.n_electrons == 0
    assert result.energy_total == pytest.approx(
        molecule.nuclear_repulsion(), abs=1e-12, rel=0.0
    )
    assert result.energy_electronic == pytest.approx(0.0, abs=1e-12, rel=0.0)


def test_TRANSKRIPSIYON_lowdin_X_SIMETRIKTIR_ve_karesi_S_TERSIDIR():
    """SINIF: TRANSKRIPSIYON DENETIMI -- Lowdin'in TANIMINI kilitler, fizik olcmez.

    NEDEN VAR (olculdu): `X = V s^(-1/2) V^T` yerine transpoze DUSURULURSE
    (`V s^(-1/2) V`) ortaya cikan matris HALA GECERLI bir ortogonallestiricidir
    -- X^T S X - I = 6.661e-16 -- ama artik simetrik DEGILDIR (|X - X^T| =
    1.301) ve X^2 = S^(-1) ozdesligini bozar (2.906). HF gozlenebilirlerinin
    HICBIRI degismez: ayni molekulde E = -1.215376262595368 (dogru) ve
    -1.215376262595370 (mutant), yani 1e-15.

    Bunun sonucu OLCULDU: bu mutant `tests/hf`'in **420 testinin 420'sini de
    GECIYOR**. Bu test eklenmeden ONCE tam suitede (864 test) kirmizi olan
    yalnizca UC test vardi ve UCU DE GORSELLESTIRME tarafindaydi:

        tests/test_orbital_window.py::test_orbital_degistirilince_ag_yeniden_uretiliyor
        tests/test_orbital_window.py::test_secici_ve_kaydirici_kullanicida_olu_degil
        tests/test_orbital_window.py::test_orbital_degistikce_ag_itemleri_sizmiyor

    Ucu de KAZA ESERI yakaliyor: MO katsayilarinin fazi/sirasi degistigi icin
    izoyuzey aglari yeniden uretiliyor ve ag-item sayimlari tutmuyor. Hicbiri
    Lowdin'in simetrisini KORUMAK icin yazilmamistir; o dosya yeniden
    yazilsaydi koruma sessizce kaybolurdu. Ayni sey kanonik ortogonallestirme
    (`X = V s^(-1/2)`, tamamen mesru bir secim, |X - X^T| = 1.502) icin de
    birebir gecerlidir: `tests/hf` 420/420 yesil, ayni uc gorsel test kirmizi.

    Yani depo, "Lowdin SIMETRIK ortogonallestirme kullaniyorum" beyanini
    ENERJIYE bakan hicbir testle koruyamaz; korumasi tesadufi bir goruntu
    testine baglidir ve o test yeniden yazilsa sessizce kaybolurdu. Bu test o
    beyani DOGRUDAN kilitler ve SCF hic kosturmadan ~1 ms surer.

    Simetrinin kendisi bir keyfilik degildir: Lowdin ortogonallestirmesi,
    orijinal AO'lara EN YAKIN ortonormal kumeyi veren tek secimdir
    (sum_i |chi_i - phi_i|^2 minimum). `analysis.py`'nin Lowdin yukleri ve
    `stability.py`'nin MO donusumu bu ozelligi varsayar.
    """
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74; H 0.6 0.5 0.2", charge=1)
    overlap = overlap_matrix(build_basis(molecule, "sto-3g"))
    x = _orthogonalisation_matrix(overlap)
    identity = np.eye(len(overlap))

    # 1. SIMETRI -- transpozun dusurulmesini yakalayan TEK iddia budur.
    #    Olculdu: dogru X icin 1.110e-16, eksik transpozede 1.301.
    assert np.abs(x - x.T).max() < 1e-12

    # 2. Gercekten ortogonallestiriyor. (Mutantlar da geciyor; bu satir
    #    yalniz "simetri" iddiasinin bos bir simetriyi kilitlemedigini
    #    gosterir.) Olculdu: 4.441e-16.
    assert np.abs(x.T @ overlap @ x - identity).max() < 1e-10

    # 3. X^2 = S^(-1) -- S^(-1/2) olmanin tanimi. Olculdu: dogru X icin
    #    6.656e-16; eksik transpozede 2.906, kanonikte 2.272.
    assert np.abs(x @ x @ overlap - identity).max() < 1e-10
