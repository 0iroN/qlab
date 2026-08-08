"""Tek-elektron integralleri (S, T, V) icin s-tipi testler.

Referans: reference/integrals_h2_074.json (PySCF, STO-3G, H2 0.74 A).
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import (
    core_hamiltonian,
    eri_tensor,
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


def test_eri_matches_pyscf(h2, h2_sto3g_integrals):
    _, basis = h2
    expected = np.array(h2_sto3g_integrals["eri"])
    np.testing.assert_allclose(eri_tensor(basis), expected, atol=1e-10)


def test_eri_has_eightfold_symmetry(h2):
    _, basis = h2
    g = eri_tensor(basis)
    for permuted in (
        np.transpose(g, (1, 0, 2, 3)),
        np.transpose(g, (0, 1, 3, 2)),
        np.transpose(g, (1, 0, 3, 2)),
        np.transpose(g, (2, 3, 0, 1)),
        np.transpose(g, (3, 2, 0, 1)),
        np.transpose(g, (2, 3, 1, 0)),
        np.transpose(g, (3, 2, 1, 0)),
    ):
        np.testing.assert_allclose(g, permuted, atol=1e-12)


def test_eri_diagonal_elements_are_positive(h2):
    _, basis = h2
    g = eri_tensor(basis)
    for i in range(g.shape[0]):
        assert g[i, i, i, i] > 0, "(ii|ii) elektron itmesi pozitif olmali"


def test_eri_cauchy_schwarz_bound(h2):
    """|(ab|cd)| <= sqrt((ab|ab)) * sqrt((cd|cd)) -- gercek bir ust sinir."""
    _, basis = h2
    g = eri_tensor(basis)
    n = g.shape[0]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    bound = np.sqrt(g[a, b, a, b]) * np.sqrt(g[c, d, c, d])
                    assert abs(g[a, b, c, d]) <= bound + 1e-12
