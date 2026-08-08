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
