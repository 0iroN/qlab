import numpy as np
import pytest

from qlab.engine.hf.basis import AVAILABLE_BASES, build_basis
from qlab.errors import BasisNotAvailableError
from qlab.types import Molecule


def test_hydrogen_sto3g_has_one_basis_function():
    functions = build_basis(Molecule.from_string("H 0 0 0"), "sto-3g")
    assert len(functions) == 1
    assert functions[0].angular == (0, 0, 0)
    assert len(functions[0].primitives) == 3


def test_h2_sto3g_has_two_basis_functions():
    mol = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    assert len(build_basis(mol, "sto-3g")) == 2


def test_water_sto3g_matches_pyscf_count(hf_energies):
    """O: 1s + 2s + 2p(3) = 5, iki H: 1'er = 7."""
    record = hf_energies["h2o"]
    mol = Molecule.from_string(record["atom"])
    assert len(build_basis(mol, "sto-3g")) == record["n_basis_functions"]


def test_water_631g_matches_pyscf_count(hf_energies):
    """6-31G'de bazi kabuklar birden fazla contraction tasir — her ek katsayi
    sutunu AYRI bir contracted fonksiyon demektir. Sutunlari yok sayarsan bu
    test tutmaz."""
    record = hf_energies["h2o_631g"]
    mol = Molecule.from_string(record["atom"])
    assert len(build_basis(mol, "6-31g")) == record["n_basis_functions"]


def test_p_shell_produces_three_functions_with_distinct_angular():
    mol = Molecule.from_string("O 0 0 0")
    p_angulars = {f.angular for f in build_basis(mol, "sto-3g") if sum(f.angular) == 1}
    assert p_angulars == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}


def test_exponents_come_from_reference_data(sto3g_basis_data):
    """Baz katsayilari referans JSON'dan gelmeli — koda elle yazilmamali."""
    functions = build_basis(Molecule.from_string("H 0 0 0"), "sto-3g")
    exponents = sorted(p.exponent for p in functions[0].primitives)
    expected = sorted(row[0] for row in sto3g_basis_data["H"][0]["primitives"])
    np.testing.assert_allclose(exponents, expected, rtol=1e-12)


def test_basis_functions_are_centred_on_their_atoms():
    mol = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    functions = build_basis(mol, "sto-3g")
    np.testing.assert_allclose(functions[0].center, mol.atoms[0].position)
    np.testing.assert_allclose(functions[1].center, mol.atoms[1].position)


def test_basis_functions_know_their_atom():
    mol = Molecule.from_string("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587")
    functions = build_basis(mol, "sto-3g")
    indices = [f.atom_index for f in functions]
    assert indices.count(0) == 5, "O'da 1s + 2s + 2p(3) = 5 fonksiyon"
    assert indices.count(1) == 1 and indices.count(2) == 1
    for function in functions:
        np.testing.assert_allclose(
            function.center, mol.atoms[function.atom_index].position
        )


def test_s_normalisation_closed_form():
    """s tipi icin N = (2a/pi)^{3/4}."""
    functions = build_basis(Molecule.from_string("H 0 0 0"), "sto-3g")
    function = functions[0]
    expected = [(2 * p.exponent / np.pi) ** 0.75 for p in function.primitives]
    np.testing.assert_allclose(function.normalisation(), expected, rtol=1e-12)


def test_unknown_basis_rejected():
    with pytest.raises(BasisNotAvailableError, match="banana"):
        build_basis(Molecule.from_string("H 0 0 0"), "banana")


def test_unsupported_element_rejected():
    with pytest.raises(BasisNotAvailableError, match="Ne"):
        build_basis(Molecule.from_string("Ne 0 0 0"), "sto-3g")


def test_available_bases_listed():
    assert AVAILABLE_BASES == ("sto-3g", "6-31g")
