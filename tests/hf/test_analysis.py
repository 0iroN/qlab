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

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from qlab.engine.hf.analysis import (
    AnalysisReport,
    _merged,
    analyse,
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
from qlab.units import ANGSTROM_TO_BOHR, AU_DIPOLE_TO_DEBYE


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


def test_dipol_OTELEME_altinda_yalnizca_NOTR_sistemde_degismez(hf_energies):
    """Yuklu bir sistemde dipol momenti koordinat ORIJININE baglidir.

    Bu bir kod hatasi degil, bir OLGU: bir sistem a kadar otelenirse
    d -> d + q*a olur (q net yuk). q = 0 ise terim yok olur ve sayi molekulun
    kendi ozelligidir; q != 0 ise bildirilen sayi molekulun uzayda NEREDE
    durduguna baglidir, yani tek basina anlamsizdir.

    Olculdu (LiH/STO-3G, ayni molekul yalnizca +10 A z'de otelendi):
      notr : 4.858750745324352 D -> 4.8587507453240795 D
             (fark 2.7267077484793845e-13, yalnizca kayan nokta gurultusu)
      2+   : 7.667440123884034 D -> 103.73153436795428 D (13.5 kat)

    Yuklu durumdaki degisim TAM OLARAK q*a'dir, ve bu test onu oyle baglar:
    dipole_au[2] 3.016603034697723 -> 40.81112552721315, fark
    37.794522492515426; q*a = 2 * 10 A * ANGSTROM_TO_BOHR = 37.794522492515405
    (goreli sapma 5.640045347053757e-16). Yani "degisiyor" demek yetmez --
    NE KADAR degistigi de olculur.

    Kirmizi yaptigi hata iki yonlu: (a) dipole_moment()'in origin'i sessizce
    baska bir noktaya (yuk merkezi, kutle merkezi) kaydirmasi -- o secim de bir
    yaklasimdir ve sessizce yapilamaz; (b) bu olgunun hic olculmemis olmasi,
    ki C-1'e kadar durum buydu.
    """
    geometri = hf_energies["lih"]["atom"]
    otelenmis = "Li 0 0 10.0; H 0 0 11.60"

    notr = dipole_debye(solve(Molecule.from_string(geometri), "sto-3g"))
    notr_otelenmis = dipole_debye(solve(Molecule.from_string(otelenmis), "sto-3g"))
    assert notr_otelenmis == pytest.approx(notr, abs=1e-9, rel=0.0), (
        "notr sistemde dipol otelemeye DUYARSIZ olmali (olculen fark 2.7e-13, "
        "sinir onun ~3700 kati)"
    )

    yuklu = dipole_moment(solve(Molecule.from_string(geometri, charge=2), "sto-3g"))
    yuklu_otelenmis = dipole_moment(
        solve(Molecule.from_string(otelenmis, charge=2), "sto-3g")
    )
    assert yuklu_otelenmis[2] - yuklu[2] == pytest.approx(
        2 * 10.0 * ANGSTROM_TO_BOHR, rel=1e-12, abs=0.0
    ), "yuklu sistemde dipol TAM OLARAK q*a kadar kayar"


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

    DUZELTME NOTU: bu dokstring bir zamanlar 6-31G olcumunu ILAN EDIYOR ama
    govde YALNIZCA sto-3g kosuyordu -- yani beyan edilen olcumun yarisi hic
    calistirilmiyordu. Analiz fonksiyonlarinda 6-31G'nin depo genelinde hic
    sinanmamasi da ayni koktendi. Ikinci baz artik gercekten kosuyor;
    olculdu, dokstringin iddiasi DOGRU cikti (n_basis: sto-3g 2, 6-31g 4;
    en kotu sapma 6-31G'de 3.3e-15). Bu, analysis.py'nin
    _ANALYSIS_NEGLECTED metninin atif yaptigi testtir -- oradaki beyan artik
    kosulan bir olcume dayaniyor.
    """
    for basis_name, distances in (
        ("sto-3g", ("0.74", "1.00", "2.00", "3.00")),
        ("6-31g", ("0.74", "1.50", "2.50", "4.00")),
    ):
        for distance in distances:
            result = solve(
                Molecule.from_string(f"H 0 0 0; H 0 0 {distance}"), basis_name
            )
            # Baz gercekten degisiyor mu: iki dal ayni sayiyi verse bile
            # 6-31G'nin kosuldugunu bu satir gosterir (2 ve 4 LITERAL).
            assert result.orbital_energies.shape[0] == (2 if basis_name == "sto-3g" else 4)
            assert mayer_bond_orders(result)[0, 1] == pytest.approx(
                1.0, abs=1e-9, rel=0.0
            ), f"{basis_name} @ {distance} A"


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


# ---------------------------------------------------------------------------
# analyse() -- tek Provenance tasiyan rapor (B2 Task 7, B1'den devir).
#
# B1'in Global Constraint'i "her sonuc Provenance tasimak zorundadir" der ama
# bu dosyanin yedi public fonksiyonu ciplak np.ndarray/float donduruyordu.
# analyse() yedisini bir kez toplar ve TEK bir Provenance kurar. Asagidaki
# testlerin her biri, sinirin hangi hatasini KIRMIZI yaptigini dokstringinde
# yazar.
# ---------------------------------------------------------------------------


def test_analyse_tum_nicelikleri_donduruyor(water):
    """Rapor, yedi fonksiyonun ciktisini BIREBIR tasimali -- yeniden hesap yok.

    Kirmizi yaptigi hata: bir alanin yanlis fonksiyona baglanmasi veya
    raporun degeri yeniden/farkli bir yoldan hesaplamasi. dipole_au ozellikle
    dahil: planin ilk test taslaginda o alan HIC sinanmiyordu, yani cop bir
    deger tasisa takim yesil kalirdi.
    """
    result, _ = water
    rapor = analyse(result)
    assert isinstance(rapor, AnalysisReport)
    np.testing.assert_allclose(rapor.mulliken, mulliken_charges(result), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(rapor.lowdin, lowdin_charges(result), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(rapor.bond_orders, mayer_bond_orders(result), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(rapor.dipole_au, dipole_moment(result), rtol=0.0, atol=1e-15)
    assert rapor.dipole_debye == pytest.approx(dipole_debye(result), rel=1e-15, abs=0.0)
    homo, lumo, gap = homo_lumo(result)
    assert rapor.homo == pytest.approx(homo, rel=1e-15, abs=0.0)
    assert rapor.lumo == pytest.approx(lumo, rel=1e-15, abs=0.0)
    assert rapor.gap == pytest.approx(gap, rel=1e-15, abs=0.0)


def test_analiz_ANA_YOLU_ikinci_bir_molekulde_de_baglaniyor(hf_energies):
    """`has_homo and has_lumo` dali -- yani her GERCEK molekulun gectigi normal
    yol -- tek sistemle (su/STO-3G) kosuluyordu ve o yuzden korumasizdi.

    Olculdu (duzeltmeden once): o dalda deger SABIT yazilinca
      gap  = 0.9964101798591279 (suyun degeri) -> 0/464 test kirildi
      homo = -0.3912742189062497                -> 0/37 test kirildi
      lumo = 0.6051359609528782                 -> 0/37 test kirildi
    Sebep: tek sinayan test (test_analyse_tum_nicelikleri_donduruyor) degeri
    AYNI molekulun homo_lumo() ciktisiyla karsilastiriyor, yani o molekulun
    degerini sabit yazmak testi gecer. He/He2/Li3+ testleri bu boslugu
    KAPATMAZ: ucu de `else` dalindan gecer, orada homo_lumo() hic cagrilmaz.

    Ikinci bir gercek molekul (LiH/STO-3G) ana yolu baglar. Olculdu:
    orbital_energies = [-2.34876194, -0.28527075, 0.07821656, 0.16394135,
    0.16394135, 0.54770841], n_occupied=2, n_orbitals=6 ->
    homo=-0.28527074715966527, lumo=0.07821656128586067,
    gap=0.36348730844552596. Ucu de suyun degerlerinden farkli.
    """
    result = solve(Molecule.from_string(hf_energies["lih"]["atom"]), "sto-3g")
    assert result.molecule.n_electrons // 2 == 2
    assert result.orbital_energies.shape[0] == 6
    rapor = analyse(result)
    homo, lumo, gap = homo_lumo(result)
    assert rapor.homo == pytest.approx(homo, rel=1e-15, abs=0.0)
    assert rapor.lumo == pytest.approx(lumo, rel=1e-15, abs=0.0)
    assert rapor.gap == pytest.approx(gap, rel=1e-15, abs=0.0)
    # ...ve olculen degerlere de bagli (homo_lumo() ile analyse()'in birlikte
    # kaymasi ihtimaline karsi dis capa):
    assert rapor.homo == pytest.approx(-0.28527074715966527, rel=1e-9, abs=0.0)
    assert rapor.lumo == pytest.approx(0.07821656128586067, rel=1e-9, abs=0.0)
    assert rapor.gap == pytest.approx(0.36348730844552596, rel=1e-9, abs=0.0)
    # gap ozdesligi analiz katmaninda da tutmali (B1'de test_homo_below_lumo
    # ile kilitli; burada raporun KENDI ucu arasinda sinaniyor).
    assert rapor.gap == pytest.approx(rapor.lumo - rapor.homo, rel=1e-12, abs=0.0)


def test_analiz_raporu_dondurulmus(water):
    """Rapor frozen olmali: tuketici (panel, CLI) bir alani yerinde degistirip
    'olculmus' gorunumlu bir sayi enjekte edememeli.

    IKI AYRI koruma gerekir ve `frozen=True` bunlardan yalnizca birini verir:

    * alan YENIDEN BAGLAMA (`rapor.gap = 0.0`) -> FrozenInstanceError. Bunu
      frozen=True saglar.
    * dizi ICERIGINI yerinde degistirme -> frozen=True bunu KORUMAZ. Olculdu
      (bu duzeltmeden once, su/STO-3G): `rapor.mulliken[0] = -99.0` sorunsuz
      CALISIYORDU, `mulliken` [-0.36550977 0.18275488 0.18275488]'den
      [-99.0 0.18275488 0.18275488]'e donuyordu; `lowdin`, `dipole_au` ve
      `bond_orders` de ayni sekilde acikti (writeable=True). Yani dokstringin
      vaat ettigi koruma dort alan icin YOKTU. Diziler artik salt-okunur.

    Kirmizi yaptigi hata: dataclass'in frozen=True olmadan tanimlanmasi
    (birinci assert), veya analyse()'in dizileri salt-okunur isaretlememesi
    (ikinci blok).
    """
    result, _ = water
    rapor = analyse(result)
    with pytest.raises(FrozenInstanceError):
        rapor.gap = 0.0  # type: ignore[misc]
    for alan in ("mulliken", "lowdin", "dipole_au", "bond_orders"):
        dizi = getattr(rapor, alan)
        assert dizi.flags.writeable is False, alan
        with pytest.raises(ValueError):
            dizi[0] = -99.0


def test_analiz_provenance_mulliken_uyarisini_tasiyor(water):
    """Bu uyari bir formalite degil: LiH'te Mulliken'in ISARETI yanlis yonu gosterir.

    Iddianin dogrulandigi yer bu dosyada:
    test_lih_mulliken_sign_is_a_documented_basis_artifact (Mulliken yanilir) ve
    test_lih_dipole_shows_correct_ionic_direction / test_ionic_character_ordering
    (dipol dogru yonu gosterir).
    """
    result, _ = water
    provenance = analyse(result).provenance
    metin = (" ".join(provenance.approximations) + " " + provenance.expected_error).lower()
    assert "mulliken" in metin
    assert "dipol" in metin
    assert "lih" in metin


def test_analiz_provenance_mayer_ve_koopmans_uyarilarini_tasiyor(water):
    result, _ = water
    provenance = analyse(result).provenance
    metin = (" ".join(provenance.neglected) + " " + provenance.expected_error).lower()
    assert "koopmans" in metin or "orbital gevsemesi" in metin
    assert "ayrisma" in metin


def test_analiz_kendi_uyarilarini_KENDI_listesinde_tasiyor(water):
    """Ustteki uc test TEK BASINA baglayici DEGILDIR -- mutasyonla olculdu.

    Olculdu (tests/hf/test_analysis.py, 37 test, dort ayri mutasyon):
      * approximations'tan Mulliken uyarisi SILINDI      -> 0 test kirildi
      * approximations'tan dipol-birincil uyarisi SILINDI -> 0 test kirildi
      * neglected'dan Koopmans uyarisi SILINDI            -> 0 test kirildi
      * neglected'dan Mayer uyarisi SILINDI               -> 1 test kirildi
    Sebep: o testler `approximations + expected_error` (ya da `neglected +
    expected_error`) BIRLESIK metnine bakiyor, ve expected_error'un taban
    metni zaten "Mulliken", "dipol", "LiH", "Koopmans" kelimelerini
    iceriyor; ayrica dalga fonksiyonundan tasinan ihmal listesi "RHF ayrisma
    kusuru" diyor. Yani analizin beyan LISTELERI tamamen bosaltilsa bile o
    substring'ler metinde kalir ve takim yesil kalirdi.

    Bu test her uyariyi KENDI listesinde ve ayirt edici bir parcayla (olculen
    sayi ya da yontem adi) arar. Ayni dort mutasyon bu testle 1'er test kirar.

    EKSIK KALAN BESINCI BEYAN (C-1 duzeltmesinde eklendi): yukaridaki dort
    kontrol `_ANALYSIS_APPROXIMATIONS`'in ucuncu maddesini ("HOMO/LUMO kanonik
    RHF orbital enerjileridir, Koopmans duzeyi") aramiyordu. Olculdu: o madde
    TAMAMEN silindiginde 0/464 test kiriliyordu. Sebep, "koopmans"in yalnizca
    `neglected` icinde aranmasi -- orada AYRI bir Koopmans maddesi var, yani
    approximations bosalsa da o assert gecerdi.
    """
    result, _ = water
    provenance = analyse(result).provenance
    approximations = " ".join(provenance.approximations).lower()
    neglected = " ".join(provenance.neglected).lower()
    assert "mulliken" in approximations
    assert "-0.016028455" in approximations  # LiH/STO-3G, PySCF ile eslesen deger
    assert "dipol" in approximations  # iyonik karakterin BIRINCIL olcusu
    # HOMO/LUMO'nun HANGI DUZEYDE okundugu -- approximations'in KENDI maddesi,
    # neglected'daki Koopmans maddesinden ayri.
    assert "kanonik" in approximations
    assert "koopmans" in approximations
    assert "koopmans" in neglected
    assert "mayer" in neglected
    assert "1.0" in neglected  # H2: her mesafede tam 1.0


def test_analiz_dalga_fonksiyonunun_beyanlarini_da_tasiyor(water):
    """Rapor, analiz ettigi dalga fonksiyonunun yaklasimlarini/ihmallerini de
    tasimali -- Born-Oppenheimer ve sonlu baz seti analizde de yururluktedir.

    Kirmizi yaptigi hata: analyse()'in sifirdan, YALNIZCA analize ozgu bir
    beyan listesi kurup SCF'in listesini dusurmesi. Ayni desen bu depoda
    zaten var: hf_obs.density_difference iki kaynagin beyanlarini birlestirir.

    UCUNCU ALAN (C-1 duzeltmesinde eklendi): dalga fonksiyonunun kendi
    `expected_error` metni de tasinmali. Olculdu: `expected_error`'in SCF
    devrini tamamen kaldiran mutasyon 0/37 test kiriyordu -- yani listeler
    tasiniyor, ama SCF'in "gerilmis geometrilerde baskin hata ... ayrisma
    kusurudur, H2 icin 14.6350 eV olculdu" cumlesi sessizce dusurulebiliyordu.
    """
    result, _ = water
    provenance = analyse(result).provenance
    assert set(result.provenance.approximations) <= set(provenance.approximations)
    assert set(result.provenance.neglected) <= set(provenance.neglected)
    # ...ve analizin kendi beyanlari da eklenmis olmali (yalnizca kopyalama degil).
    assert len(provenance.approximations) > len(result.provenance.approximations)
    assert len(provenance.neglected) > len(result.provenance.neglected)
    # Dalga fonksiyonunun hata beyani KELIMESI KELIMESINE icerilmeli: analiz,
    # analiz ettigi dalga fonksiyonu kadar guvenilirdir.
    assert result.provenance.expected_error in provenance.expected_error
    assert len(provenance.expected_error) > len(result.provenance.expected_error)


def test_analiz_KENDI_beklenen_hatasini_expected_error_alaninda_tasiyor(water):
    """`expected_error` panelin en gorunur alanidir ve HIC baglanmamisti.

    Olculdu: `_ANALYSIS_EXPECTED_ERROR`'un 701 karakterlik tamami tek bir "x"
    harfine indirildiginde 0/464 test kiriliyordu. Sebep, brief'in verdigi iki
    provenance testinin `approximations + expected_error` (ya da `neglected +
    expected_error`) BIRLESIK metnine bakmasi: aradiklari her substring zaten
    listelerde de geciyor, yani expected_error tamamen bosalsa bile yesil
    kaliyorlardi. Bu, `..._KENDI_listesinde_tasiyor`'un cozdugu tuzagin AYNISI,
    bir alan yukarida.

    Bu test yalnizca `expected_error` alanina bakar ve listelerde GECMEYEN
    ayirt edici parcalar arar: raporun "en buyuk hata YORUMDA'dir" tezi ve
    LiH/STO-3G'de olculen Mulliken degeri.
    """
    result, _ = water
    expected_error = analyse(result).provenance.expected_error.lower()
    assert "yorumda" in expected_error
    assert "-0.016028455" in expected_error
    assert "birincil olarak dipol" in expected_error


def test_yuklu_sistemde_rapor_dipolun_ORIJINE_BAGLI_oldugunu_BEYAN_EDIYOR(hf_energies):
    """C-1: yuklu sistemde dipol orijine baglidir (olgu, bkz.
    test_dipol_OTELEME_altinda_yalnizca_NOTR_sistemde_degismez) -- rapor bunu
    SESSIZ gecemez.

    Task 7 `expected_charge`'i raporun birinci sinif niceligi yapti ve
    charge=3 sistemini desteklenen bir test sistemi haline getirdi, yani
    dipole_moment()'in "bu projede tum senaryolar norotr" on kosulu artik
    gecerli DEGIL. Ayni anda raporun kendi bas talimati ("iyonik karakter
    BIRINCIL olarak dipolle olculur") okuyucuyu tam o sayiya yonlendiriyordu.

    Bir orijin secip sessizce "duzeltmek" yanlis olurdu: secilen orijin de bir
    yaklasimdir. Dogru olan, sayiyi oldugu gibi vermek ve NE OLDUGUNU
    yazmaktir.

    Kirmizi yaptigi hata iki yonlu: (a) beyanin hic olmamasi; (b) beyanin
    KOSULSUZ eklenmesi -- notr bir molekulun raporunda "dipol orijine baglidir"
    yazmak da yanlis bilgidir ve okuyucunun dogru sayiya guvenini bosa
    harcatir. Bu yuzden asagida iki yon de sinaniyor.
    """
    yuklu = analyse(
        solve(Molecule.from_string(hf_energies["lih"]["atom"], charge=2), "sto-3g")
    )
    metin = yuklu.provenance.expected_error.lower()
    assert "bu sistem yuklu" in metin
    assert "orijinine baglidir" in metin
    assert "d -> d + q*a" in metin  # olgunun kendisi, sozu degil
    # Substring tek basina zayiftir; makine-okunur esi de var (precedent:
    # diagnostics["expected_electrons"] / ["expected_charge"]).
    assert yuklu.provenance.diagnostics["dipole_origin_dependent"] is True

    notr = analyse(solve(Molecule.from_string(hf_energies["lih"]["atom"]), "sto-3g"))
    notr_metin = notr.provenance.expected_error.lower()
    assert "orijin" not in notr_metin, "notr sistemde dipol otelemeye duyarsiz"
    assert notr.provenance.diagnostics["dipole_origin_dependent"] is False


def test_analiz_yontemi_bazi_adiyla_beyan_ediyor(water, hf_energies):
    """Panelde hangi bazin okundugu gorunmeli (precedent: test_hf_obs.py'de
    `assert "6-31G" in buyuk.provenance.method`).

    Kirmizi yaptigi hata: method'un sabit bir metne ("RHF analizi") yazilmasi
    -- ya da sabit ama icinde "STO-3G" GECEN bir metne yazilmasi.

    Olculdu: `method=f"STO-3G"` (sonuctan turetilmeyen duz sabit) mutasyonu
    0/37 test kiriyordu, cunku bu dosyanin butun sistemleri sto-3g kullaniyor.
    O precedent yalnizca FARKLI bir baz da kosuldugunda baglayicidir; asagidaki
    6-31G blogu bunu sagliyor.
    """
    result, _ = water
    assert "STO-3G" in analyse(result).provenance.method
    buyuk = solve(Molecule.from_string(hf_energies["h2o"]["atom"]), "6-31g")
    buyuk_method = analyse(buyuk).provenance.method
    assert "6-31G" in buyuk_method
    assert "STO-3G" not in buyuk_method


def test_analiz_provenance_makine_okunur_tanilari_tasiyor(water):
    """Substring kontrolu tek basina zayiftir: icinde "mulliken dipol lih"
    gecen sacma bir metin de gecer. Bu yuzden provenance ayrica olculen/
    beklenen ciftini makine-okunur tasir (precedent: hf_obs'ta
    diagnostics["expected_electrons"]).

    Olculdu (su/STO-3G): n_occupied=5, n_orbitals=7, Mulliken yuklerinin
    toplami -7.105427357601002e-15, molekulun yuku 0.

    UYARI (olculdu): notr bir molekulde `mulliken_charge_sum`'i
    `pytest.approx(0.0)` ile sinamak BAGLAYICI DEGILDIR -- alani sabit 0.0
    yazan bir mutasyon 0 test kirdi. Bu yuzden asagida deger, olculen
    toplamla BIREBIR karsilastirilir (iki taraf da ayni cagrinin ciktisi,
    yani bit-esit olmak zorunda) ve yuku SIFIR OLMAYAN sistem
    test_isgal_edilmis_orbital_yoksa_HOMO_uydurulmuyor icinde sinanir.
    """
    result, _ = water
    diagnostics = analyse(result).provenance.diagnostics
    assert diagnostics["n_occupied"] == 5
    assert diagnostics["n_orbitals"] == 7
    assert diagnostics["expected_charge"] == 0
    assert diagnostics["mulliken_charge_sum"] == float(np.sum(mulliken_charges(result)))
    # SCF'in kendi tanilari da dusurulmemeli: converged=True isaretinin yaninda
    # onu destekleyen sayi (final_residual, iterations) durmali.
    assert result.provenance.diagnostics.items() <= diagnostics.items()


def test_analiz_scf_kararlilik_hukmunu_kaybetmiyor(water):
    """Analiz, analiz ettigi dalga fonksiyonu kadar guvenilirdir.

    SCF bir eyer noktasina oturduysa rapordaki her yuk ve her bag mertebesi
    o eyer noktasindan hesaplanmistir. Raporun provenance'i SCF'in kararlilik
    hukmunu DUSURURSE panel kararlilik satirini hic basmaz ve okuyucu bunu
    "sorun yok" diye okur -- Provenance.stability dokstringinin tam olarak
    uyardigi hata.

    Olculdu (su/STO-3G): stable=True, lowest_eigenvalue=2.0926707415493113,
    n_rotations=10. Alanlarin gercekten dolu oldugu ayrica sinanir; bos bir
    StabilityVerdict tasimak da bilgi kaybidir.
    """
    result, _ = water
    verdict = analyse(result).provenance.stability
    assert verdict == result.provenance.stability
    assert verdict is not None
    assert verdict.stable is True
    assert verdict.n_rotations == 10
    assert verdict.lowest_eigenvalue == pytest.approx(2.0926707415493113, rel=1e-9, abs=0.0)


def test_analiz_olculmemis_kararliligi_UYDURMUYOR(hf_energies):
    """Tersi de gecerli: SCF kararliligi OLCMEDIYSE rapor bir hukum uyduramaz.

    solve(stability=False) ile kosulan sonucun provenance.stability'si None'dir
    (olculdu). analyse() bu alani result.provenance'tan TASIR; kendi basina
    stability(result) cagirip yeniden hesaplamaz -- oyle yapsaydi rapor,
    kaynagin OLCMEDIGI bir hukmu beyan ederdi.

    Kirmizi yaptigi hata: analyse() icinde stability(result) cagrilmasi veya
    sabit bir StabilityVerdict yazilmasi.
    """
    result = solve(
        Molecule.from_string(hf_energies["h2o"]["atom"]), "sto-3g", stability=False
    )
    assert result.provenance.stability is None
    assert analyse(result).provenance.stability is None


def test_analiz_yakinsama_hukmunu_tasiyor(water):
    """converged de AYNEN tasinir: yakinsamamis bir dalga fonksiyonundan
    hesaplanmis yukler "yakinsadi" diye gosterilemez.

    solve() yakinsamayi ISTISNA ile bildirdigi icin (SCFConvergenceError)
    converged=False tasiyan bir HFResult uretilemez; olculdu: depoda HFResult
    YALNIZCA scf.solve() icinde kurulur (tek `HFResult(` cagrisi, scf.py:395)
    ve orada provenance'a converged=True SABIT yazilir -- converged=False
    yalnizca factory.run_scan'in ScanResult'inda gorunur. Bu yuzden test
    frozen dataclass'i dataclasses.replace ile kopyalayip alani dogrudan bozar.

    Kirmizi yaptigi hata: Provenance(converged=True) sabit yazilmasi.
    """
    result, _ = water
    yakinsamamis = replace(
        result, provenance=replace(result.provenance, converged=False)
    )
    assert analyse(yakinsamamis).provenance.converged is False
    assert analyse(result).provenance.converged is True


def test_lumo_yoksa_analiz_patlamiyor_ve_bunu_beyan_ediyor():
    """He/STO-3G'de bos orbital YOK -- LUMO diye bir sey yok.

    Olculdu: homo_lumo(He) ham `IndexError: index 1 is out of bounds for
    axis 0 with size 1` firlatiyor (n_occupied=1, n_orbitals=1). analyse()
    bunu tuketiciye yansitmamali. gap=0.0 yazmak "bosluk yok, metalik" diye
    okunurdu; None + beyan dogru. Ayni karar StabilityVerdict.
    lowest_eigenvalue'da zaten verilmis.

    Olculdu (He/STO-3G): orbital_energies = [-0.8760355074024515].
    """
    result = solve(Molecule.from_string("He 0 0 0"), "sto-3g")
    rapor = analyse(result)
    assert rapor.lumo is None
    assert rapor.gap is None
    assert rapor.homo == pytest.approx(-0.8760355074024515, rel=1e-9, abs=0.0)
    provenance = rapor.provenance
    assert provenance.diagnostics["n_occupied"] == 1
    assert provenance.diagnostics["n_orbitals"] == 1
    # Ayirt edici cumle: sade "lumo" substring'i baglayici DEGILDIR, cunku
    # analizin taban expected_error metni zaten "HOMO/LUMO Koopmans duzeyinde
    # kalir" diyor -- beyan tamamen silinse bile o substring gecerdi.
    assert "bu sistemde lumo yok" in provenance.expected_error.lower()


def test_lumo_yoksa_homo_hala_EN_YUKSEK_isgal_edilmis_orbital():
    """He'de tek orbital vardir, yani yanlis bir HOMO indeksi orada AYNI sayiyi
    verir ve fark edilmez. He2/STO-3G bu boslugu kapatir: n_occupied=2,
    n_orbitals=2 (yine LUMO yok) ama iki orbital enerjisi FARKLIDIR.

    Olculdu (He2, r=2.0 A, STO-3G): orbital_energies =
    [-0.8968359506145498, -0.8542936219536033]. HOMO en yuksek isgal edilmis
    orbitaldir, yani index 1 -> -0.85429362. Index 0 alan bir hata -0.89683595
    verirdi; test bu iki degeri ayirt eder.

    (Duzeltme notu: bu dokstring onceden birinci enerjiyi -0.8968359506154
    diye yaziyordu -- son iki rakam yer degismisti ve deger ASSERT EDILMEDIGI
    icin takim onu hicbir zaman yakalayamazdi. Sayi duzeltildi ve asagida
    artik assert ediliyor, yani iddia bir daha sessizce yanlis olamaz.)
    """
    result = solve(Molecule.from_string("He 0 0 0; He 0 0 2.0"), "sto-3g")
    assert result.molecule.n_electrons // 2 == result.orbital_energies.shape[0] == 2
    assert float(result.orbital_energies[0]) == pytest.approx(
        -0.8968359506145498, rel=1e-12, abs=0.0
    )
    rapor = analyse(result)
    assert rapor.lumo is None and rapor.gap is None
    assert rapor.homo == pytest.approx(-0.8542936219536033, rel=1e-12, abs=0.0)


def test_isgal_edilmis_orbital_yoksa_HOMO_uydurulmuyor():
    """Simetrik bosluk (controller listesinde yoktu, burada olculdu): isgal
    edilmis orbital olmayan bir sistemde HOMO da yoktur.

    NE OLDUGU KONUSUNDA DURUST OLMAK: Li3+ (charge=3) sifir elektronlu CIPLAK
    BIR CEKIRDEKTIR, kimyasal bir sistem degildir. Burada bir fizik iddiasi
    olarak degil, `n_occupied == 0` SINIRINI baglayan en kucuk sistem olarak
    duruyor. Raporu `converged=True` tasir (elektron olmadigi icin SCF ilk
    adimda yakinsar) ve butun analiz alanlari dolar -- bu, sinirin kendisinin
    neden beyan gerektirdiginin ornegidir.

    Molekulun ORIJINDE durmasi da bilincli degil, tarihsel bir tesadufti ve
    tam olarak C-1'i gizleyen konumdur: dipol burada 0.000000 D cikar, ama ayni
    ciplak cekirdek `Li 0 0 5` icin 72.04807068305266 D, `Li 0 0 10` icin
    144.09614136610531 D "olculmus" dipol bildirir (yuklu sistemde dipol
    orijine baglidir; bkz.
    test_yuklu_sistemde_rapor_dipolun_ORIJINE_BAGLI_oldugunu_BEYAN_EDIYOR).
    Bu yuzden asagida raporun o beyani tasidigi da sinaniyor.

    Olculdu (Li, charge=3 -> 0 elektron, STO-3G, 5 baz fonksiyonu):
    homo_lumo() patlamaz ama SESSIZCE yanlis doner -- n_occupied-1 = -1
    indeksi son (en yuksek) orbitale sarar ve sonuc
    homo=-0.8775791414291554, lumo=-4.407952827261487, gap=-3.5303736858323314
    olur: NEGATIF bir bosluk, yani fiziksel olarak imkansiz bir sayi rapora
    "olculmus" gibi girer. analyse() bu durumda homo ve gap'i None birakir;
    LUMO gercekten vardir (en dusuk orbital) ve tasinir.
    """
    result = solve(Molecule.from_string("Li 0 0 0", charge=3), "sto-3g")
    rapor = analyse(result)
    assert rapor.homo is None
    assert rapor.gap is None
    assert rapor.lumo == pytest.approx(-4.407952827261487, rel=1e-9, abs=0.0)
    assert rapor.provenance.diagnostics["n_occupied"] == 0
    # Ayirt edici cumle; sade "homo" substring'i taban metinde zaten var.
    assert "isgal edilmis orbital yok" in rapor.provenance.expected_error.lower()
    # Yuku SIFIR OLMAYAN tek sistem: Mulliken toplami molekulun yukune esit
    # olmak zorundadir (olculdu: 3.0, charge=3). Notr sistemde iki taraf da
    # sifir oldugu icin bu ciftin sabit yazilmasi orada gorunmez; burada
    # gorunur.
    assert rapor.provenance.diagnostics["expected_charge"] == 3
    assert rapor.provenance.diagnostics["mulliken_charge_sum"] == pytest.approx(
        3.0, abs=1e-10, rel=0.0
    )
    # charge != 0: dipol orijine baglidir ve rapor bunu beyan etmek zorunda.
    assert rapor.provenance.diagnostics["dipole_origin_dependent"] is True
    assert "orijin" in rapor.provenance.expected_error.lower()


def test_isgal_bazin_kapasitesini_ASARSA_analiz_TIPLI_hata_veriyor():
    """`has_homo` korumasinin ikinci, hic sinanmamis ucu: n_occupied > n_orbitals.

    `has_homo = n_occupied >= 1` yalnizca ALT siniri bagliyordu. Ust ucta
    `result.orbital_energies[n_occupied - 1]` izgaranin disina tasiyor ve
    analyse() HAM bir `IndexError` ile patliyordu -- tam olarak kendi
    dokstringinin "Public bir API'nin ham bir IndexError ile patlamasi ...
    kabul edilemez" diye reddettigi sey. OLCULDU (H2/STO-3G, charge=-4):
    `IndexError: index 2 is out of bounds for axis 0 with size 2`,
    `isinstance(e, QLabError)` False. Ust sinir bir tek testle bile
    baglanmamisti; olculdu, `has_homo`'ya ust sinir eklemek tests/hf altindaki
    316 testin SIFIRINI kiriyordu.

    NEDEN None+beyan DEGIL de HATA: bu sistemde eksik olan yalnizca HOMO
    degil, RAPORUN TAMAMI anlamsiz. Bazda 2 orbital var, sistem 6 elektron
    iddia ediyor; `scf._density_matrix`'teki `coefficients[:, :n_occupied]`
    dilimi numpy'da SESSIZCE 2 sutuna kirpiliyor, yani yogunluk 4 elektron
    tasiyor. Olculdu (ayni sistem): Tr(PS) = 4 (molekul 6 diyor), Mulliken
    toplami -2.0 (molekulun yuku -4), H2'nin Mayer bag mertebesi 4.0
    (RHF'de ust sinir 1.0), dipol 3.554 D, ve solve() bunu enerjisi POZITIF
    (+0.9231791808701104 hartree) bir HFResult olarak NORMAL donduruyor.
    homo/gap'i None birakip raporu teslim etmek, bu alti sayiyi "olculmus"
    gorunumuyle tuketiciye vermek olurdu. Ustelik tek satirlik o duzeltme
    _NO_HOMO_NOTE ("n_occupied == 0") ve _NO_LUMO_NOTE ("n_occupied ==
    n_orbitals") beyanlarini birden bastirirdi -- gercek n_occupied=3,
    n_orbitals=2, yani ham IndexError iki YALAN BEYANLA takas edilirdi.

    KAPSAM: duzeltme analyse()'te. Asil kusur YUKARI AKISTA -- solve()'un bu
    sistemi hic reddetmemesi -- ve scf.py bu asamanin dosyalari disindadir;
    asagidaki Tr(PS) iddiasi o kusuru SESSIZ olmaktan cikarip yaziya
    baglar, boylece scf.py'ye dokunuldugunda burasi kirmizi olur.

    Sayilar LITERAL: 6/3/2/4 kod tarafindaki hicbir sabite bagli degil.
    """
    from qlab.engine.hf.integrals import overlap_matrix
    from qlab.errors import QLabError, UnsupportedSystemError

    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74", charge=-4)
    assert molecule.n_electrons == 6
    result = solve(molecule, "sto-3g", stability=False)
    assert result.orbital_energies.shape[0] == 2, "STO-3G'de H2 iki orbital tasir"
    assert molecule.n_electrons // 2 == 3 > 2, "isgal bazin kapasitesini asiyor"

    # YUKARI AKIS KUSURU, yaziya baglandi: solve() bu sistemi kabul ediyor ve
    # yogunlugu molekulun iddia ettigi elektron sayisini TASIMIYOR.
    overlap = overlap_matrix(build_basis(molecule, "sto-3g"))
    assert float(np.trace(result.density @ overlap)) == pytest.approx(
        4.0, abs=1e-9, rel=0.0
    ), "yogunluk 4 elektron tasiyor, molekul 6 iddia ediyor -- solve() susuyor"

    with pytest.raises(UnsupportedSystemError) as hata:
        analyse(result)

    assert isinstance(hata.value, QLabError)
    assert not isinstance(hata.value, IndexError)
    mesaj = str(hata.value)
    # Tani icin gereken uc sayi da mesajda olmali (errors.py modul dokstringi:
    # "Her istisna, taniyi yapabilmek icin gereken sayisal baglami tasir").
    assert "6" in mesaj and "3" in mesaj and "2" in mesaj
    assert "sto-3g" in mesaj.lower()


def test_isgal_kapasiteye_TAM_oturdugunda_analiz_kosuyor():
    """Yeni sinirin YANLIS TARAFA kaymadigini baglar: n_occupied == n_orbitals GECERLI.

    Ust sinir `>=` yazilsaydi He/STO-3G (n_occupied = n_orbitals = 1) ve
    He2/STO-3G (2 = 2) reddedilirdi; ikisi de LUMO'su olmayan MESRU sistemler
    ve zaten iki ayri testle korunuyorlar. Bu test siniri diger yonden
    sikistirir: kapasiteye tam oturan sistem hata ALMAZ.

    Olculdu: He/STO-3G n_occupied=1, n_orbitals=1; He2/STO-3G (r=2.0 A)
    n_occupied=2, n_orbitals=2. Sayilar LITERAL.
    """
    for geometri, isgal in (("He 0 0 0", 1), ("He 0 0 0; He 0 0 2.0", 2)):
        result = solve(Molecule.from_string(geometri), "sto-3g")
        assert result.molecule.n_electrons // 2 == isgal
        assert int(result.orbital_energies.shape[0]) == isgal
        rapor = analyse(result)
        assert rapor.homo is not None
        assert rapor.lumo is None and rapor.gap is None


def test_merged_beyanlari_sirayi_bozmadan_tekrarsiz_birlestiriyor():
    """`_merged`'in dokstring iddiasi ("sirayi bozmadan, tekrarsiz") uretimde
    HIC dogrulanmiyor: olculdu, SCF'in beyan listeleriyle analizin listeleri
    arasinda su an KESISIM YOK (iki taraf icin de bos kesisim), yani dedup
    yolu bir defa bile tetiklenmiyor -- tekrar ayiklamayi tamamen kaldiran bir
    mutasyon uretimde hicbir sey degistirmezdi. Bir sonraki gorev ortak bir
    beyan eklerse (ornegin iki katman da "sonlu baz seti" derse) davranis
    birden onemli hale gelir; o gune kadar iddia testsiz kalmasin.

    Sira KORUNUR, siralanmaz: panelde beyanlarin okunma sirasi dalga
    fonksiyonundan analize dogrudur, alfabetik degil.
    """
    assert _merged(("a", "b"), ("b", "c")) == ("a", "b", "c")
    assert _merged(("b",), ("a",)) == ("b", "a")
