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

Iki-elektron itme integralleri (ERI, kimyaci notasyonunda (ab|cd)) icin
s-tipi kapali form (p = a+b, q = g+d, mu_ab = ab/p, mu_cd = gd/q,
P = (aA+bB)/p, Q = (gC+dD)/q; A,B,C,D primitif merkezleri):

    (ab|cd) = 2*pi^{5/2} / (p*q*sqrt(p+q))
              * exp(-mu_ab|A-B|^2) * exp(-mu_cd|C-D|^2)
              * F_0( p*q/(p+q) * |P-Q|^2 )

ERI tensoru 8-katli permutasyon simetrisine sahiptir:
(ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba).
Bu simetriden yararlanilarak yalnizca a>=b, c>=d, (ab)>=(cd) uclugeni
hesaplanir; kalan elemanlar kopyalanarak doldurulur (bkz. eri_tensor).
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


def _pair_centre(
    function: BasisFunction,
    p: np.ndarray,
    exp_left: np.ndarray,
    exp_right: np.ndarray,
    other: BasisFunction,
) -> np.ndarray:
    """Bir primitif cifti icin agirlikli merkez P = (a*A + b*B) / p, sekil (na, nb, 3)."""
    return (
        exp_left[:, None, None] * function.center
        + exp_right[None, :, None] * other.center
    ) / p[:, :, None]


def _eri_primitive_block(
    a: BasisFunction, b: BasisFunction, c: BasisFunction, d: BasisFunction
) -> np.ndarray:
    """(ab|cd) icin primitif duzeyde deger izgarasi, sekil (na, nb, nc, nd).

    _primitive_grids() (a,b) ve (c,d) ciftleri icin p, mu, r2 saglar; P ve Q
    agirlikli merkezleri burada ayrica hesaplanir (nuclear_matrix'teki
    merkez_p ile ayni desen). Kalan islem brief'teki kapali formun dogrudan
    vektorize uygulamasidir.
    """
    exp_a, exp_b, p, mu_ab, r2_ab, _ = _primitive_grids(a, b)
    exp_c, exp_d, q, mu_cd, r2_cd, _ = _primitive_grids(c, d)

    centre_p = _pair_centre(a, p, exp_a, exp_b, b)
    centre_q = _pair_centre(c, q, exp_c, exp_d, d)

    pq_toplam = p[:, :, None, None] + q[None, None, :, :]
    pq_carpim = p[:, :, None, None] * q[None, None, :, :]
    fark_pq = centre_p[:, :, None, None, :] - centre_q[None, None, :, :, :]
    r2_pq = np.sum(fark_pq**2, axis=-1)
    T = pq_carpim / pq_toplam * r2_pq

    exp_ab = np.exp(-mu_ab * r2_ab)[:, :, None, None]
    exp_cd = np.exp(-mu_cd * r2_cd)[None, None, :, :]
    on_carpan = (2 * np.pi**2.5) / (pq_carpim * np.sqrt(pq_toplam)) * exp_ab * exp_cd

    return on_carpan * boys(0, T)


def eri_tensor(basis: list[BasisFunction]) -> np.ndarray:
    """Iki-elektron itme integralleri (ab|cd), sekil (n, n, n, n).

    8-katli permutasyon simetrisinden yararlanilir: yalnizca a>=b, c>=d ve
    (ab bilesik indeksi)>=(cd bilesik indeksi) sartlarini saglayan benzersiz
    ucgen hesaplanir; kalan yedi permutasyon dogrudan kopyalanarak doldurulur.
    """
    n = len(basis)
    weights = [_weights(f) for f in basis]
    tensor = np.zeros((n, n, n, n))

    for a in range(n):
        for b in range(a + 1):
            ab_index = a * (a + 1) // 2 + b
            for c in range(n):
                for d in range(c + 1):
                    cd_index = c * (c + 1) // 2 + d
                    if cd_index > ab_index:
                        continue
                    blok = _eri_primitive_block(basis[a], basis[b], basis[c], basis[d])
                    deger = float(
                        np.einsum(
                            "i,j,k,l,ijkl->",
                            weights[a],
                            weights[b],
                            weights[c],
                            weights[d],
                            blok,
                        )
                    )
                    for i, j in ((a, b), (b, a)):
                        for k, l in ((c, d), (d, c)):
                            tensor[i, j, k, l] = deger
                            tensor[k, l, i, j] = deger
    return tensor
