import numpy as np
import pytest

from qlab.errors import InvalidInputError
from qlab.types import Atom, Molecule


def test_atomic_number_lookup():
    assert Atom("H", np.zeros(3)).atomic_number == 1
    assert Atom("O", np.zeros(3)).atomic_number == 8


def test_unknown_element_rejected():
    with pytest.raises(InvalidInputError, match="Bilinmeyen element"):
        Atom("Xx", np.zeros(3))


def test_position_must_be_three_dimensional():
    with pytest.raises(InvalidInputError, match="uc bilesenli"):
        Atom("H", np.zeros(2))


def test_electron_count_and_charge():
    mol = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    assert mol.n_electrons == 2
    cation = Molecule.from_string("H 0 0 0; H 0 0 0.74", charge=1)
    assert cation.n_electrons == 1


def test_from_string_converts_angstrom_to_bohr():
    mol = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    separation = np.linalg.norm(mol.atoms[1].position - mol.atoms[0].position)
    assert separation == pytest.approx(0.74 * 1.8897261254578281, rel=1e-9)


def test_from_string_accepts_bohr_directly():
    mol = Molecule.from_string("H 0 0 0; H 0 0 1.4", unit="bohr")
    separation = np.linalg.norm(mol.atoms[1].position - mol.atoms[0].position)
    assert separation == pytest.approx(1.4, rel=1e-12)


def test_nuclear_repulsion_matches_pyscf(hf_energies):
    """PySCF'in kendi cekirdek itme enerjisiyle birebir uyusmali.

    Bu test Angstrom->bohr donusumunun VE mesafe hesabinin ayni anda dogru
    oldugunu kanitlar; ikisinden biri yanlissa deger tutmaz.
    """
    record = hf_energies["h2_074"]
    mol = Molecule.from_string(record["atom"])
    assert mol.nuclear_repulsion() == pytest.approx(record["energy_nuclear"], rel=1e-10)


def test_nuclear_repulsion_matches_pyscf_for_water(hf_energies):
    record = hf_energies["h2o"]
    mol = Molecule.from_string(record["atom"])
    assert mol.nuclear_repulsion() == pytest.approx(record["energy_nuclear"], rel=1e-10)


def test_coincident_nuclei_rejected():
    with pytest.raises(InvalidInputError, match="cakisik"):
        Molecule.from_string("H 0 0 0; H 0 0 0")


def test_malformed_atom_line_rejected():
    with pytest.raises(InvalidInputError, match="Sembol"):
        Molecule.from_string("H 0 0")


def test_non_numeric_coordinate_rejected():
    with pytest.raises(InvalidInputError, match="sayiya"):
        Molecule.from_string("H 0 0 abc")


def test_odd_electron_count_is_allowed_by_molecule():
    """Molecule tek elektronu reddetmez — o kisit RHF cozucusune aittir."""
    assert Molecule.from_string("H 0 0 0").n_electrons == 1
