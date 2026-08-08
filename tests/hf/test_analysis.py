"""Analiz testleri: Mulliken/Lowdin yukleri, dipol momenti, HOMO/LUMO, Mayer bag mertebesi.

Referans: reference/hf_energies.json (PySCF).

rtol/rel notu: brief'ten alinan bazi pytest.approx/assert_allclose cagrilari
yalniz abs= veya yalniz rel= veriyordu. Proje kurali geregi (bkz.
.superpowers/sdd/b1-task-10-brief.md ve scf.py'deki ayni kalip) her cagriya
DIGER parametre acikca eklendi -- np.testing.assert_allclose'da rtol=0.0,
pytest.approx'ta karsi parametre 0.0 -- boylece varsayilan deger sessizce
gevsetme yapmiyor. Su an icin (expected degerlerin coguu sifirdan uzak
oldugu icin) bu numerik olarak davranisi degistirmedi, ama niyeti acik
kilmak ve gelecekteki bir deger degisikliginde sessiz gevsemeyi onlemek
icin yine de eklendi.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.analysis import (
    dipole_debye,
    dipole_moment,
    homo_lumo,
    lowdin_charges,
    mayer_bond_orders,
    mulliken_charges,
)
from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import overlap_matrix
from qlab.engine.hf.scf import solve
from qlab.types import Molecule
from qlab.units import AU_DIPOLE_TO_DEBYE


@pytest.fixture(scope="module")
def water(hf_energies):
    record = hf_energies["h2o"]
    return solve(Molecule.from_string(record["atom"]), "sto-3g"), record


def test_mulliken_charges_match_pyscf(water):
    result, record = water
    np.testing.assert_allclose(
        mulliken_charges(result), record["mulliken_charges"], atol=1e-6, rtol=0.0
    )


def test_charges_sum_to_molecular_charge(water):
    result, _ = water
    assert np.sum(mulliken_charges(result)) == pytest.approx(0.0, abs=1e-8, rel=0.0)
    assert np.sum(lowdin_charges(result)) == pytest.approx(0.0, abs=1e-8, rel=0.0)


def test_dipole_magnitude_matches_pyscf(water):
    result, record = water
    expected = float(np.linalg.norm(record["dipole_debye"]))
    assert dipole_debye(result) == pytest.approx(expected, rel=1e-5, abs=0.0)


def test_dipole_vector_matches_pyscf(water):
    """Norm degil, ucu bileseninin tamami -- yon de dogru olmali, sadece buyukluk degil."""
    result, record = water
    dipole = dipole_moment(result) * AU_DIPOLE_TO_DEBYE
    np.testing.assert_allclose(dipole, record["dipole_debye"], atol=1e-6, rtol=0.0)


def test_homonuclear_molecule_has_no_dipole():
    """H2 simetrik -- dipol sifir olmali."""
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")
    assert dipole_debye(result) == pytest.approx(0.0, abs=1e-8, rel=0.0)


def test_ionic_character_ordering(hf_energies):
    """Dipol: H2 (kovalent) < HF < LiH (en iyonik). Bagin karakteri OLCULUR."""
    h2 = dipole_debye(solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g"))
    hf = dipole_debye(solve(Molecule.from_string(hf_energies["hf_mol"]["atom"]), "sto-3g"))
    lih = dipole_debye(solve(Molecule.from_string(hf_energies["lih"]["atom"]), "sto-3g"))
    assert h2 < hf < lih


def test_lih_mulliken_sign_is_a_documented_basis_artifact(hf_energies):
    """Mulliken/STO-3G LiH'te YANILTICIDIR: Li kucuk NEGATIF, H kucuk POZITIF.

    Bu, kimyasal sezginin (Li elektropozitif -> pozitif, H elektronegatif ->
    negatif) tam tersidir. Once bu sonucun bir kod hatasi OLMADIGINI dogrula:
    su formul (q_A = Z_A - sum_{mu in A} (PS)_mu,mu), suyun O/H yukleri icin
    dogru isareti (O negatif, H pozitif -- test_mulliken_charges_match_pyscf)
    VE PySCF'in LiH icin urettigi mulliken_charges referansiyla (asagida,
    atol=1e-6) tam olarak eslesiyor -- yani olculmus, dogrulanmis bir sonuc.

    Neden ters cikiyor: Mulliken paylasilan (cift-merkezli, PS'nin
    kosegen-disi kismindan gelen) yogunluk katkisini iki atom arasinda kor
    bicimde yari yariya boler; hangi atomun gercekten daha elektronegatif
    oldugunu dikkate almaz. STO-3G'de Li'nin (cok kontrakte, ic-kabuk
    agirlikli) ve H'nin (daha yayilmis) baz fonksiyonlari arasindaki
    boyut/yayilma uyumsuzlugu bu kor bolusturmeyi ters cevirir.

    Bu YUZDEN spec bunu birincil metrik yapmiyor: dogru YONLU (Li+, H-)
    iyonik karakter dipol momentinden okunur, bkz.
    test_lih_dipole_shows_correct_ionic_direction. mulliken_charges'in
    docstring'i bu sinirlamayi acikca belirtir.
    """
    charges = mulliken_charges(
        solve(Molecule.from_string(hf_energies["lih"]["atom"]), "sto-3g")
    )
    np.testing.assert_allclose(
        charges, hf_energies["lih"]["mulliken_charges"], atol=1e-6, rtol=0.0
    )
    assert charges[0] < 0, "olculen: Mulliken Li'ye (yaniltici bicimde) negatif yuk verir"
    assert charges[1] > 0, "olculen: Mulliken H'ye (yaniltici bicimde) pozitif yuk verir"


def test_lih_dipole_shows_correct_ionic_direction(hf_energies):
    """Dogru YONLU iyonik karakter Mulliken'de degil dipol vektorunde.

    Molekul: Li merkezde (z=0), H pozitif z tarafinda (bkz. hf_energies["lih"]["atom"]).
    Kimyasal olarak beklenen Li+ H- karakteri, elektron yogunlugunun H
    tarafina kaymasi demektir; dipole_moment = sum Z_A R_A - sum P_uv <r>
    formulunde bu, elektronik terimin z bileseninin nukleer terimden buyuk
    cikmasi, yani toplam z bileseninin NEGATIF olmasi anlamina gelir.
    PySCF referansiyla (hf_energies["lih"]["dipole_debye"]) tam eslesme
    hem yonu hem buyuklugu dogrular.
    """
    result = solve(Molecule.from_string(hf_energies["lih"]["atom"]), "sto-3g")
    dipole = dipole_moment(result) * AU_DIPOLE_TO_DEBYE
    np.testing.assert_allclose(dipole, hf_energies["lih"]["dipole_debye"], atol=1e-6, rtol=0.0)
    assert dipole[2] < 0, "H tarafi elektronca zengin olmali (dogru iyonik yon, Li+ H-)"


def test_homo_below_lumo(water):
    result, _ = water
    homo, lumo, gap = homo_lumo(result)
    assert homo < lumo
    assert gap == pytest.approx(lumo - homo, rel=1e-12, abs=0.0)
    assert gap > 0


def test_homo_matches_highest_occupied_orbital(water):
    result, record = water
    n_occupied = result.molecule.n_electrons // 2
    homo, _, _ = homo_lumo(result)
    assert homo == pytest.approx(record["orbital_energies"][n_occupied - 1], abs=1e-6, rel=0.0)


def test_h2_bond_order_near_one():
    """Tek kovalent bag -- Mayer bag mertebesi 1'e yakin olmali."""
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")
    assert mayer_bond_orders(result)[0, 1] == pytest.approx(1.0, abs=0.1, rel=0.0)


def test_bond_order_matrix_is_symmetric(water):
    result, _ = water
    orders = mayer_bond_orders(result)
    np.testing.assert_allclose(orders, orders.T, atol=1e-12, rtol=0.0)


def test_water_nonbonded_hydrogens_have_near_zero_bond_order(water):
    """O-H baglari gercek, H...H baglanmamis cift -- bag mertebesi sifira yakin olmali."""
    result, _ = water
    orders = mayer_bond_orders(result)
    assert orders[0, 1] == pytest.approx(0.954201, abs=1e-6, rel=0.0), "O-H bag mertebesi"
    assert orders[1, 2] == pytest.approx(0.012400, abs=1e-6, rel=0.0), "H...H (baglanmamis)"
    assert orders[1, 2] < orders[0, 1]


def test_rhf_bond_order_of_h2_stays_one_at_any_separation():
    """RHF'nin ayrisma kusuru, olculebilir bicimde yaziya dokulmus.

    Minimal bazda -- ve 6-31G'de de -- H2 icin Mayer bag mertebesi HER
    mesafede tam 1.0'dir. Es-cekirdekli simetri MO katsayilarini sabitler,
    ve RHF cekirdekler ne kadar ayrilirsa ayrilsin bag orbitalini cift
    isgal etmeye zorlar. Bu bir kod hatasi degil, yontemin beyan edilen
    sinirlarindan biridir: RHF bag KIRAMAZ.
    Olculdu: STO-3G 0.74/1.00/2.00/3.00 A -> dordu de 1.0000000000;
    6-31G 0.74/1.50/2.50/4.00 A -> dordu de 1.00000000.
    """
    for distance in ("0.74", "1.00", "2.00", "3.00"):
        result = solve(Molecule.from_string(f"H 0 0 0; H 0 0 {distance}"), "sto-3g")
        assert mayer_bond_orders(result)[0, 1] == pytest.approx(1.0, abs=1e-9, rel=0.0)


def test_stretched_heteronuclear_bond_has_lower_order(hf_energies):
    """Bag uzadikca bag mertebesi dusmeli -- simetri kisiti OLMAYAN molekulde.

    LiH'te es-cekirdekli simetri yok, bu yuzden dusus gerceklesir; ama RHF'de
    etki kucuktur. Olculdu: 1.60 A -> 0.99974309, 4.00 A -> 0.99383428
    (fark 5.9e-3). 1e-3'luk marj olculen farkin altinda, yani iddia
    baglayicidir: bag mertebesi sabit kalsa test kirmizi olur.
    Gerilmis LiH duz iterasyonda salinir (bkz. Task 9), bu yuzden DIIS acik
    ve max_iterations yuksek.
    """
    short = mayer_bond_orders(
        solve(Molecule.from_string(hf_energies["lih"]["atom"]), "sto-3g")
    )[0, 1]
    stretched = mayer_bond_orders(
        solve(Molecule.from_string("Li 0 0 0; H 0 0 4.0"), "sto-3g", max_iterations=300)
    )[0, 1]
    assert stretched < short - 1e-3


# ---------------------------------------------------------------------------
# Lowdin per-atom validation (b1-task-10 fix: lowdin_charges had no per-atom
# test -- only the sum-to-zero check, which a redistribution bug between
# basis functions of the same atom, or between atoms, can pass while still
# being wrong per-atom). Every bound below is set from a measured value, not
# guessed -- see .superpowers/sdd/b1-task-10-fix-report.md for the numbers.
# ---------------------------------------------------------------------------


def test_sqrt_overlap_squares_to_overlap_on_water_basis(water):
    """The one genuinely untested numerical piece behind lowdin_charges:
    building S^(1/2) from the eigendecomposition of the overlap matrix.

    This reconstructs S^(1/2) with the exact recipe lowdin_charges uses
    (eigh(S), then V diag(sqrt(w)) V^T) on the water/STO-3G basis, which has
    p functions -- not just the trivial 2x2 H2 case -- and asserts the
    defining property S^(1/2) @ S^(1/2) == S directly, with no dependence on
    lowdin_charges' own output.

    Measured: max|S^(1/2) @ S^(1/2) - S| = 4.107825191113079e-15 (water,
    STO-3G, 7x7 overlap matrix, eigenvalues span [0.3427, 1.9303]). Bound
    atol=1e-11 is ~2435x the measured deviation -- comfortably above
    floating-point noise across BLAS/platform variation, but far below what
    any real construction bug (wrong transpose, using S instead of S^(1/2),
    dropping the sqrt) would produce.
    """
    result, _ = water
    basis = build_basis(result.molecule, result.basis_name)
    overlap = overlap_matrix(basis)
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    sqrt_overlap = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    np.testing.assert_allclose(sqrt_overlap @ sqrt_overlap, overlap, atol=1e-11, rtol=0.0)


def test_lowdin_charges_match_independent_sqrt_overlap_reconstruction(water):
    """Cross-checks lowdin_charges' actual output against a from-scratch
    reconstruction written independently in this test (same formula, but not
    calling any analysis.py internals) -- the self-contained substitute for
    a PySCF Lowdin reference, which does not exist anywhere in reference/
    (checked: hf_energies.json, integrals_h2o.json, integrals_h2_074.json,
    basis_*.json carry Mulliken charges only).

    Because this test's reconstruction is separate code from analysis.py's
    lowdin_charges, it actually binds to analysis.py's behavior: a bug in
    lowdin_charges' S^(1/2) construction (wrong transpose, using S instead
    of S^(1/2), ...) changes lowdin_charges(result) but not this test's own
    (correct) reconstruction, so the two would diverge and the assertion
    would fail. See break-and-restore proof in the fix report.

    Measured: max|independent - actual| = 0.0 (bit-identical, both computed
    via the same np.linalg.eigh call on the identical overlap matrix).
    atol=1e-10 leaves enormous margin over that, but is still far tighter
    than any real per-atom redistribution bug, which measures in the
    0.01-1 range (see the LiH Mulliken sign artifact in this file).
    """
    result, _ = water
    basis = build_basis(result.molecule, result.basis_name)
    atom_index = np.array([function.atom_index for function in basis])
    overlap = overlap_matrix(basis)
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    sqrt_overlap = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    population = np.diag(sqrt_overlap @ result.density @ sqrt_overlap)
    nuclear_charges = np.array(
        [atom.atomic_number for atom in result.molecule.atoms], dtype=float
    )
    independent_charges = nuclear_charges.copy()
    np.add.at(independent_charges, atom_index, -population)
    np.testing.assert_allclose(
        lowdin_charges(result), independent_charges, atol=1e-10, rtol=0.0
    )


def test_h2_lowdin_charges_are_exactly_zero():
    """H2 is homonuclear and neutral -- both Lowdin charges must be zero,
    not merely sum to zero. A bug that redistributes charge between the two
    atoms (e.g. cancelling errors of +x/-x) would still pass
    test_charges_sum_to_molecular_charge but fail this per-atom check.

    Measured: max|charge| = 1.2212453270876722e-15 (STO-3G, bond length
    0.74 A). Bound atol=1e-9 is ~8.2e5x the measured value -- generous
    headroom for SCF/eigh numerical noise, still 1e6x tighter than what a
    genuine redistribution bug would produce (order 0.01-1, cf. the LiH
    Mulliken artifact of ~0.016 in this file).
    """
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")
    charges = lowdin_charges(result)
    np.testing.assert_allclose(charges, [0.0, 0.0], atol=1e-9, rtol=0.0)


def test_water_hydrogens_have_equal_lowdin_charges(water):
    """Water's two hydrogens are symmetry-equivalent in the reference
    geometry (O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587 -- mirror images
    under y -> -y), so their Lowdin charges must be equal to each other, not
    merely both close to the same rounded value. A per-atom bug that shifts
    charge from one H to the other without changing the total would pass
    test_charges_sum_to_molecular_charge but break this.

    Measured: |q(H1) - q(H2)| = 6.661338147750939e-16. Bound atol=1e-9 is
    ~1.5e6x the measured value -- same generous-but-still-tight margin as
    the H2 test above, for the same reason (headroom over numerical noise,
    far below any real redistribution bug's scale).
    """
    result, _ = water
    charges = lowdin_charges(result)
    assert charges[1] == pytest.approx(charges[2], abs=1e-9, rel=0.0)


def test_water_lowdin_signs_agree_with_mulliken():
    """For water, both Mulliken and Lowdin put oxygen negative and both
    hydrogens positive -- this is a case where the two secondary metrics
    are NOT in doubt about sign (unlike LiH/STO-3G, where they doubly agree
    with each other but BOTH disagree with the dipole moment's Li+/H-
    direction; see test_lih_mulliken_sign_is_a_documented_basis_artifact).

    This assertion is deliberately scoped to water only: it would be FALSE
    for LiH (Mulliken gives Li a small negative charge there, and Lowdin is
    not asserted to differ), so it must not be generalized past the one
    system where both docstrings and the dipole all point the same way.
    """
    result = solve(
        Molecule.from_string("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"), "sto-3g"
    )
    mulliken = mulliken_charges(result)
    lowdin = lowdin_charges(result)
    assert mulliken[0] < 0 and lowdin[0] < 0, "oxygen: both schemes agree, negative"
    assert mulliken[1] > 0 and lowdin[1] > 0, "H1: both schemes agree, positive"
    assert mulliken[2] > 0 and lowdin[2] > 0, "H2: both schemes agree, positive"


# ---------------------------------------------------------------------------
# Mayer diagonal (b1-task-10 fix, Minor): B_AA (an atom's self-bond-order)
# was computed by the (a, b) double loop but never asserted by any test.
# ---------------------------------------------------------------------------


def test_h2_mayer_diagonal_is_one():
    """Measured: diag(mayer_bond_orders) = [1.0, 1.0] for H2/STO-3G (bond
    length 0.74 A), deviation from 1.0 of order 1.8e-15. This is consistent
    with (interpretation, not asserted here) the RHF closed-shell
    idempotency identity (PS)^2 = 2(PS): H2 is neutral, so the H-H
    off-diagonal bond order is exactly 1.0 (test_h2_bond_order_near_one
    measures ~1.0, and separately B_AA + B_AB(other) = 2*population_A here
    forces B_AA = 1.0 too, see test_mayer_diagonal_matches_population_identity
    for the general form of this identity on water).

    Bound atol=1e-9 is ~5.6e5x the measured 1.8e-15 deviation -- same margin
    discipline as the Lowdin symmetry tests above.
    """
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")
    diagonal = np.diag(mayer_bond_orders(result))
    np.testing.assert_allclose(diagonal, [1.0, 1.0], atol=1e-9, rtol=0.0)


def test_water_mayer_diagonal_pinned_values(water):
    """Pins the measured Mayer diagonal for water/STO-3G, the same style as
    test_water_nonbonded_hydrogens_have_near_zero_bond_order pins the
    off-diagonal.

    Measured: diag = [14.822618278166, 0.667889578483, 0.667889578483]
    (oxygen, H1, H2 -- the two hydrogens agree by symmetry, as with the
    Lowdin charges above). Values rounded to 8 decimals below; atol=1e-6
    covers the ~2e-9 rounding error with room to spare, matching the
    tolerance already used for other pinned bond-order values in this file
    (e.g. 0.954201, 0.012400).
    """
    result, _ = water
    diagonal = np.diag(mayer_bond_orders(result))
    np.testing.assert_allclose(
        diagonal, [14.82261828, 0.66788958, 0.66788958], atol=1e-6, rtol=0.0
    )


def test_mayer_diagonal_matches_population_identity(water):
    """Gives the Mayer diagonal a genuine, self-contained interpretation
    (rather than treating it as an arbitrary unconstrained number): for
    closed-shell RHF, (PS)^2 = 2(PS) holds entrywise (P is 2*C_occ C_occ^T
    in this project's convention, and C_occ^T S C_occ = I from
    orthonormality, so (PS)^2 = 4 C_occ (C_occ^T S C_occ) C_occ^T S =
    2*(PS)). Restricting the diagonal of that entrywise identity to the
    basis functions of one atom and summing gives, for every atom A:

        B_AA + sum_{B != A} B_AB == 2 * population_A
                                  == 2 * (Z_A - mulliken_charge_A)

    i.e. the FULL row sum of the Mayer matrix (diagonal included) equals
    twice the atom's Mulliken population. This needs no external reference
    -- it follows from the converged density matrix alone -- and a bug that
    corrupts B_AA specifically (leaving off-diagonal terms and
    mulliken_charges untouched) would violate it while every other test in
    this file stayed green.

    Measured (water/STO-3G): row sums [16.731019538796, 1.634490230602,
    1.634490230602] vs 2*population [16.731019538796, 1.634490230602,
    1.634490230602], max|diff| = 1.7763568394e-14. Bound atol=1e-10 is
    ~5.6e5x that -- comfortable numerical headroom, still far tighter than
    any real bug would produce.
    """
    result, _ = water
    orders = mayer_bond_orders(result)
    row_sums = orders.sum(axis=1)
    populations = np.array([8.0, 1.0, 1.0]) - mulliken_charges(result)
    np.testing.assert_allclose(row_sums, 2 * populations, atol=1e-10, rtol=0.0)
