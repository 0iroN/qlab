"""Tek-elektron integralleri (S, T, V) icin s-tipi testler.

Referans: reference/integrals_h2_074.json (PySCF, STO-3G, H2 0.74 A).
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import (
    core_hamiltonian,
    kinetic_matrix,
    nuclear_matrix,
    overlap_matrix,
)
from qlab.types import Molecule


@pytest.fixture(scope="module")
def h2(hf_energies):
    molecule = Molecule.from_string(hf_energies["h2_074"]["atom"])
    return molecule, build_basis(molecule, "sto-3g")


def test_overlap_matches_pyscf(h2, h2_sto3g_integrals):
    _, basis = h2
    expected = np.array(h2_sto3g_integrals["overlap"])
    np.testing.assert_allclose(overlap_matrix(basis), expected, atol=1e-10)


def test_kinetic_matches_pyscf(h2, h2_sto3g_integrals):
    _, basis = h2
    expected = np.array(h2_sto3g_integrals["kinetic"])
    np.testing.assert_allclose(kinetic_matrix(basis), expected, atol=1e-10)


def test_nuclear_matches_pyscf(h2, h2_sto3g_integrals):
    molecule, basis = h2
    expected = np.array(h2_sto3g_integrals["nuclear"])
    np.testing.assert_allclose(nuclear_matrix(basis, molecule), expected, atol=1e-10)


def test_overlap_diagonal_is_unity(h2):
    _, basis = h2
    np.testing.assert_allclose(np.diag(overlap_matrix(basis)), 1.0, atol=1e-10)


def test_matrices_are_symmetric(h2):
    molecule, basis = h2
    for matrix in (
        overlap_matrix(basis),
        kinetic_matrix(basis),
        nuclear_matrix(basis, molecule),
    ):
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)


def test_overlap_is_positive_definite(h2):
    _, basis = h2
    assert np.all(np.linalg.eigvalsh(overlap_matrix(basis)) > 0)


def test_core_hamiltonian_is_sum_of_parts(h2):
    molecule, basis = h2
    np.testing.assert_allclose(
        core_hamiltonian(basis, molecule),
        kinetic_matrix(basis) + nuclear_matrix(basis, molecule),
        atol=1e-14,
    )


def test_overlap_falls_off_with_separation():
    close = build_basis(Molecule.from_string("H 0 0 0; H 0 0 0.5"), "sto-3g")
    far = build_basis(Molecule.from_string("H 0 0 0; H 0 0 3.0"), "sto-3g")
    assert overlap_matrix(close)[0, 1] > overlap_matrix(far)[0, 1]
    assert overlap_matrix(far)[0, 1] < 0.05
