"""Roothaan-Hall SCF dongusu: kapali kabuk (restricted) Hartree-Fock.

Matematik (spec bkz. .superpowers/sdd/b1-task-8-brief.md):

    X        = S^(-1/2)                          (Lowdin simetrik ortogonallestirme)
    P_ls     = 2 sum_i^occ C_li C_si
    F_uv     = Hcore_uv + sum_ls P_ls [ (uv|ls) - 1/2 (ul|vs) ]
    F'       = X^T F X  ->  ozdeger  ->  C = X C'
    E_elec   = 1/2 sum_uv P_uv (Hcore_uv + F_uv)
    E_toplam = E_elec + E_cekirdek

X = S^(-1/2) simetrik ortogonallestirme icin S'nin ozayrisimi kullanilir:
S = U s U^T (s kosegen, ozdegerler pozitif -- S bir ortusme matrisi, tekil
degil surece pozitif tanimlidir), X = U s^(-1/2) U^T.

Baslangic tahmini: core Hamiltoniyeni (Hcore) genellesmis ozdeger problemi
olarak cozulur (P=0 ile Fock = Hcore). Bu "core guess" olarak bilinir ve
DIIS olmadan bile makul bir baslangic noktasidir.

Yakinsama kriteri iki parcalidir (VE): enerji degisimi < energy_tolerance
VE RMS(e) < residual_tolerance, burada e = X^T (F P S - S P F) X ticaretcil
(commutator) hatasidir -- yakinsamis bir cozumde F ve P commute eder (FPS -
SPF = 0), yani e sifira gider.

Bu dosya DIIS uygulamaz (Task 9'a birakilmistir); use_diis parametresi
kabul edilir ama yok sayilir. Duz Roothaan iterasyonu bazi sistemlerde
yavas yakinsayabilir veya salinabilir -- bu olculmesi gereken bir riskir,
gizlenecek bir sorun degil.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor, overlap_matrix
from qlab.errors import SCFConvergenceError, UnsupportedSystemError
from qlab.provenance import Provenance
from qlab.types import Molecule


@dataclass(frozen=True)
class HFResult:
    molecule: Molecule
    basis_name: str
    energy_total: float
    energy_electronic: float
    energy_nuclear: float
    orbital_energies: np.ndarray
    coefficients: np.ndarray
    density: np.ndarray
    n_iterations: int
    provenance: Provenance


def _orthogonalisation_matrix(overlap: np.ndarray) -> np.ndarray:
    """X = S^(-1/2), Lowdin simetrik ortogonallestirme."""
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    inverse_sqrt = np.diag(eigenvalues ** -0.5)
    return eigenvectors @ inverse_sqrt @ eigenvectors.T


def _density_matrix(coefficients: np.ndarray, n_occupied: int) -> np.ndarray:
    """P_ls = 2 sum_i^occ C_li C_si."""
    occupied = coefficients[:, :n_occupied]
    return 2.0 * occupied @ occupied.T


def _fock_matrix(
    core: np.ndarray, eri: np.ndarray, density: np.ndarray
) -> np.ndarray:
    """F_uv = Hcore_uv + sum_ls P_ls [ (uv|ls) - 1/2 (ul|vs) ]."""
    coulomb = np.einsum("uvls,ls->uv", eri, density)
    exchange = np.einsum("ulvs,ls->uv", eri, density)
    return core + coulomb - 0.5 * exchange


def _electronic_energy(density: np.ndarray, core: np.ndarray, fock: np.ndarray) -> float:
    """E_elec = 1/2 sum_uv P_uv (Hcore_uv + F_uv)."""
    return 0.5 * float(np.einsum("uv,uv->", density, core + fock))


def _commutator_residual(
    fock: np.ndarray, density: np.ndarray, overlap: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """e = X^T (F P S - S P F) X, yakinsamada sifira giden ticaretcil hatasi."""
    fps = fock @ density @ overlap
    spf = overlap @ density @ fock
    return x.T @ (fps - spf) @ x


def solve(
    molecule: Molecule,
    basis_name: str = "sto-3g",
    max_iterations: int = 100,
    energy_tolerance: float = 1e-10,
    residual_tolerance: float = 1e-8,
    use_diis: bool = True,
) -> HFResult:
    """Kapali kabuk RHF enerjisini Roothaan-Hall SCF donguleriyle cozer.

    use_diis su an yok sayilir (Task 9'da uygulanacak); duz Roothaan
    iterasyonu kullanilir.
    """
    del use_diis  # Task 9: DIIS hizlandirma. Simdilik duz iterasyon.

    if molecule.n_electrons % 2 != 0:
        raise UnsupportedSystemError(
            f"acik kabuk sistem desteklenmiyor (n_electrons={molecule.n_electrons}); "
            "yalnizca kapali kabuk (cift elektron) sistemler cozulebilir"
        )

    n_occupied = molecule.n_electrons // 2

    basis = build_basis(molecule, basis_name)
    overlap = overlap_matrix(basis)
    core = core_hamiltonian(basis, molecule)
    eri = eri_tensor(basis)
    nuclear_repulsion = molecule.nuclear_repulsion()

    x = _orthogonalisation_matrix(overlap)

    # Baslangic tahmini: core guess (P=0 ile Fock = Hcore).
    fock_prime = x.T @ core @ x
    _, coefficients_prime = np.linalg.eigh(fock_prime)
    coefficients = x @ coefficients_prime
    density = _density_matrix(coefficients, n_occupied)

    energy_history: list[float] = []
    fock = core
    residual_norm = float("inf")

    for iteration in range(1, max_iterations + 1):
        fock = _fock_matrix(core, eri, density)
        energy_electronic = _electronic_energy(density, core, fock)
        energy_history.append(energy_electronic)

        residual = _commutator_residual(fock, density, overlap, x)
        residual_norm = float(np.sqrt(np.mean(residual ** 2)))

        # Yeni katsayilari/yoguntugu her iterasyonda guncelle (yakinsasa da
        # yakinsamasa da), boylece dongu kirilinca ekstra bir tekrar
        # hesaplamaya gerek kalmaz: donen coefficients/density/orbital_energies
        # tam da bu iterasyonun Fock'undan turetilmis olur.
        fock_prime = x.T @ fock @ x
        orbital_energies, coefficients_prime = np.linalg.eigh(fock_prime)
        coefficients = x @ coefficients_prime
        density = _density_matrix(coefficients, n_occupied)

        converged = (
            iteration > 1
            and abs(energy_history[-1] - energy_history[-2]) < energy_tolerance
            and residual_norm < residual_tolerance
        )
        if converged:
            break
    else:
        raise SCFConvergenceError(
            iterations=max_iterations,
            energy_history=energy_history,
            final_residual=residual_norm,
        )

    energy_electronic = energy_history[-1]
    energy_total = energy_electronic + nuclear_repulsion

    provenance = Provenance(
        method=f"RHF/{basis_name.upper()}",
        approximations=(
            "Born-Oppenheimer",
            "tek Slater determinanti",
            "sonlu baz seti",
            "kapali kabuk (cift isgal)",
        ),
        neglected=("elektron korelasyonu", "goreli etkiler", "cekirdek hareketi"),
        expected_error=(
            "korelasyon enerjisi eksik -- bag enerjilerinde tipik olarak birkac eV"
        ),
        converged=True,
        diagnostics={"iterations": iteration, "final_residual": residual_norm},
    )

    return HFResult(
        molecule=molecule,
        basis_name=basis_name,
        energy_total=energy_total,
        energy_electronic=energy_electronic,
        energy_nuclear=nuclear_repulsion,
        orbital_energies=orbital_energies,
        coefficients=coefficients,
        density=density,
        n_iterations=iteration,
        provenance=provenance,
    )
