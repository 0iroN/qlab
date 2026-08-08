"""SCF (Roothaan-Hall) testleri: kapali kabuk RHF enerjisi PySCF ile karsilastirilir.

Referans: reference/hf_energies.json (PySCF, 9 sistem + bir 6-31G durumu).
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import overlap_matrix
from qlab.engine.hf.scf import _diis_extrapolate, solve
from qlab.errors import SCFConvergenceError, UnsupportedSystemError
from qlab.types import Molecule


@pytest.mark.parametrize(
    "label", ["he", "h2_074", "h2_100", "h2_200", "lih", "hf_mol", "h2o", "nh3", "ch4"]
)
def test_total_energy_matches_pyscf(hf_energies, label):
    """Referans hakemligi: her sistem PySCF ile 1e-6 hartree icinde uyusmali."""
    record = hf_energies[label]
    result = solve(Molecule.from_string(record["atom"]), record["basis"])
    assert result.energy_total == pytest.approx(record["energy_total"], abs=1e-6)


def test_631g_basis_also_matches(hf_energies):
    record = hf_energies["h2_074_631g"]
    result = solve(Molecule.from_string(record["atom"]), "6-31g")
    assert result.energy_total == pytest.approx(record["energy_total"], abs=1e-6)


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
    """tr(PS) elektron sayisini vermeli."""
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    result = solve(molecule, "sto-3g")
    S = overlap_matrix(build_basis(molecule, "sto-3g"))
    assert np.trace(result.density @ S) == pytest.approx(molecule.n_electrons, abs=1e-8)


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
    """Yakinsamis cozumde daha siki tolerans enerjiyi degistirmemeli."""
    molecule = Molecule.from_string(hf_energies["h2_074"]["atom"])
    loose = solve(molecule, "sto-3g", energy_tolerance=1e-10)
    tight = solve(molecule, "sto-3g", energy_tolerance=1e-13, max_iterations=200)
    assert loose.energy_total == pytest.approx(tight.energy_total, abs=1e-9)


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
