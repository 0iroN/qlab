"""SCF (Roothaan-Hall) testleri: kapali kabuk RHF enerjisi PySCF ile karsilastirilir.

Referans: reference/hf_energies.json (PySCF, 9 sistem + bir 6-31G durumu).
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import overlap_matrix
from qlab.engine.hf.scf import solve
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
