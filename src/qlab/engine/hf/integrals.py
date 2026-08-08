"""Tek-elektron integralleri: ortusme (S), kinetik (T), cekirdek-cekim (V).

Bu modul yalnizca **s tipi** primitifleri dogru hesaplar (angular ==
(0, 0, 0)). p kabugu Task 7'de Obara-Saika ozyinelemesiyle eklenecek.

s-tipi kapali formlar (p = a+b, mu = ab/p, P = (aA+bB)/p), normalize
edilmemis primitifler icin (A, B primitif merkezleri; C cekirdek konumu):

    S_ab = (pi/p)^{3/2} * exp(-mu|A-B|^2)
    T_ab = mu * (3 - 2*mu*|A-B|^2) * S_ab
    V_ab = -(2*pi/p) * Z * exp(-mu|A-B|^2) * F_0(p*|P-C|^2)

Kontrakte fonksiyon degeri primitif katkilarinin d_i * d_j * N_i * N_j
agirlikli toplamidir; ayrica her fonksiyon
BasisFunction.contraction_normalisation() ile kendi oz-ortusmesinin
kareokune bolunerek yeniden normalize edilir -- bu olmadan PySCF
referansiyla eslesme atol=1e-10 hedefini onlarca kat asar (bkz.
.superpowers/sdd/task-5-brief.md ve basis.py).

F_0 icin Boys fonksiyonu (qlab.engine.hf.boys.boys) kullanilir.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.hf.basis import BasisFunction
from qlab.engine.hf.boys import boys
from qlab.types import Molecule


def _weights(function: BasisFunction) -> np.ndarray:
    """Primitif basina tam agirlik: d_i * N_i * kontraksiyon-normalizasyonu."""
    coefficients = np.array([p.coefficient for p in function.primitives])
    return (
        coefficients
        * function.normalisation()
        * function.contraction_normalisation()
    )


def _primitive_grids(
    a: BasisFunction, b: BasisFunction
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Iki fonksiyonun primitif ciftleri icin p, mu, r2 ve S_prim izgarasi.

    Donen degerler (na, nb) sekilli matrisler (na/nb primitif sayisi),
    r2 skalerdir (fonksiyon merkezleri arasi mesafenin karesi -- tum
    primitif ciftleri icin ortaktir).
    """
    exp_a = np.array([p.exponent for p in a.primitives])
    exp_b = np.array([p.exponent for p in b.primitives])
    diff = a.center - b.center
    r2 = float(diff @ diff)
    p = exp_a[:, None] + exp_b[None, :]
    mu = exp_a[:, None] * exp_b[None, :] / p
    s_prim = (np.pi / p) ** 1.5 * np.exp(-mu * r2)
    return exp_a, exp_b, p, mu, r2, s_prim


def overlap_matrix(basis: list[BasisFunction]) -> np.ndarray:
    """Ortusme matrisi S_ab, sekil (n, n)."""
    n = len(basis)
    weights = [_weights(f) for f in basis]
    matrix = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            _, _, _, _, _, s_prim = _primitive_grids(basis[a], basis[b])
            matrix[a, b] = weights[a] @ s_prim @ weights[b]
    return matrix


def kinetic_matrix(basis: list[BasisFunction]) -> np.ndarray:
    """Kinetik enerji matrisi T_ab, sekil (n, n)."""
    n = len(basis)
    weights = [_weights(f) for f in basis]
    matrix = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            _, _, _, mu, r2, s_prim = _primitive_grids(basis[a], basis[b])
            t_prim = mu * (3 - 2 * mu * r2) * s_prim
            matrix[a, b] = weights[a] @ t_prim @ weights[b]
    return matrix


def nuclear_matrix(basis: list[BasisFunction], molecule: Molecule) -> np.ndarray:
    """Cekirdek-cekim matrisi V_ab, sekil (n, n).

    Her cekirdek uzerinden toplanir: V = sum_C V_ab(C); isaret negatiftir
    (cekim) ve bu isaret asagidaki `on_carpan` teriminde tasinir.
    """
    n = len(basis)
    weights = [_weights(f) for f in basis]
    matrix = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            fn_a, fn_b = basis[a], basis[b]
            exp_a, exp_b, p, mu, r2, _ = _primitive_grids(fn_a, fn_b)
            merkez_p = (
                exp_a[:, None, None] * fn_a.center
                + exp_b[None, :, None] * fn_b.center
            ) / p[:, :, None]
            on_carpan = -(2 * np.pi / p) * np.exp(-mu * r2)

            toplam = 0.0
            for atom in molecule.atoms:
                fark_pc = merkez_p - atom.position
                r2_pc = np.sum(fark_pc**2, axis=-1)
                v_prim = on_carpan * atom.atomic_number * boys(0, p * r2_pc)
                toplam += weights[a] @ v_prim @ weights[b]
            matrix[a, b] = toplam
    return matrix


def core_hamiltonian(basis: list[BasisFunction], molecule: Molecule) -> np.ndarray:
    """Cekirdek Hamiltoniyeni H_core = T + V, sekil (n, n)."""
    return kinetic_matrix(basis) + nuclear_matrix(basis, molecule)
