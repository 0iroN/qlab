"""Kisitlanmamis Hartree-Fock: alfa ve beta elektronlar AYRI orbitallerde.

RHF ile tek yapisal fark budur; her sey oradan cikar. Kapali kabuk bir
minimumda UHF, RHF'e OZDES doner (teorem) -- kazanc yalnizca RHF cozumu bir
EYER NOKTASI oldugunda ortaya cikar. Olculdu: H2 R = 2.00 A'da RHF
-0.783792654264, UHF -0.937212833096, fark 0.153420179 hartree = 4.175 eV.

ALTI SESSIZ TUZAK (hepsi asagidaki dort satirdadir):

1. J TOPLAM yogunluktan (Pa + Pb) kurulur, K ise HER SPIN icin AYRI
   (Ka Pa'dan, Kb Pb'den). J'yi spin yogunlugundan kurmak olculdu: H2O'da
   -53.26 Ha veriyor ve VARYASYONEL denetim bunu YAKALAMIYOR.
2. K'nin indis deseni J'ninkinden FARKLIDIR: J -> "uvls,ls->uv",
   K -> "ulvs,ls->uv". Karistirmak olculdu: ayrisma cipasi 4.12e-01 sapiyor.
3. Enerji ifadesi 0.5*(Tr[Pt Hc] + Tr[Pa Fa] + Tr[Pb Fb]) + E_nuc'tir.
   RHF'in 0.5*Tr[P(Hc+F)] ifadesini kopyalamak olculdu: bes ayri denetim
   birden kirmizi.
4. n_alpha != n_beta durumu: alfa ve beta AYRI sayida orbital doldurur ve
   `solve()`un reddettigi tek elektronlu sistemler burada COZULUR.
5. <S^2> formulu ORTUSME matrisini icerir; S = I varsaymak sessizce yanlis
   sonuc verir (H2 R=1.4'te S01 = 0.6593182061, yani S birim DEGIL).
6. BASLANGIC TAHMINI kritiktir: bias olmadan UHF, RHF'e COKER. Olculdu,
   H2 R = 2.00 A'da bias=None ile UHF tam olarak RHF enerjisini veriyor
   (fark < 1e-12, <S^2> = 0). Bu bir kusur DEGIL, dogru davranistir --
   ama dondurulen sayinin bir MINIMUM oldugu iddiasi ancak cok-baslangicli
   arama (`multistart`) ile yapilabilir.

DIIS BIRLESIK UYGULANIR: alfa ve beta hata vektorleri ust uste yigilir ve
tek bir ekstrapolasyon yapilir. Ayri ayri genisletmek iki kanalin farkli
adimlarda birbirini kovalamasina yol acar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor, overlap_matrix
from qlab.engine.hf.scf import (
    _commutator_residual,
    _diis_extrapolate,
    _DIIS_MAX_HISTORY,
    _orthogonalisation_matrix,
)
from qlab.errors import InvalidInputError, SCFConvergenceError
from qlab.provenance import Provenance
from qlab.types import Molecule

_UHF_YAKLASIMLAR: tuple[str, ...] = (
    "kisitlanmamis Hartree-Fock: alfa ve beta AYRI orbitallerde",
    "tek Slater determinanti -- korelasyon enerjisi eksiktir",
    "sonlu baz seti",
    "SPIN OZDURUMU DEGILDIR: <S^2> beklenen S(S+1) degerinden sapar (spin "
    "kirlenmesi) ve bu sapma sonucla birlikte raporlanir",
    "dondurulen deger bir MINIMUM DEGIL bir DURAGAN NOKTADIR; minimum "
    "iddiasi ancak cok-baslangicli arama (multistart) ile yapilabilir "
    "-- olculdu, baslangic sapmasi olmadan UHF RHF'e coker",
)

_UHF_IHMAL_EDILEN: tuple[str, ...] = (
    "spin projeksiyonu (PUHF) yapilmaz",
    "kompleks orbital kararsizligi sinanmaz",
    "goreli duzeltmeler, spin-yorunge etkilesimi",
)

_UHF_BEKLENEN_HATA = (
    "korelasyon enerjisi eksiktir ve bu depoda olculmus bir korelasyon "
    "referansi YOKTUR; spin kirlenmesi <S^2> olarak raporlanir ve enerjiden "
    "cok daha YAVAS yakinsar, dolayisiyla ondan yedi anlamli haneden "
    "fazlasi okunmaz"
)


@dataclass(frozen=True, eq=False)
class UHFResult:
    """UHF cozumu. Enerjiler hartree."""

    energy_total: float
    coefficients_alpha: np.ndarray
    coefficients_beta: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    s_squared: float
    n_alpha: int
    n_beta: int
    provenance: Provenance


def spin_squared(
    coefficients_alpha: np.ndarray,
    coefficients_beta: np.ndarray,
    overlap: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> float:
    """<S^2> = Sz(Sz+1) + n_beta - SUM_ij |<a_i|b_j>|^2.

    Ic carpim ORTOGONAL OLMAYAN AO tabaninda alinir, yani `overlap` ile.
    S = I varsaymak sessizce yanlis sonuc verir.
    """
    sz = 0.5 * (n_alpha - n_beta)
    ortusme = coefficients_alpha[:, :n_alpha].T @ overlap @ coefficients_beta[:, :n_beta]
    return float(sz * (sz + 1.0) + n_beta - np.sum(ortusme**2))


def solve_uhf(
    molecule: Molecule,
    basis_name: str = "sto-3g",
    *,
    n_alpha: int | None = None,
    initial_bias: np.ndarray | None = None,
    max_iterations: int = 200,
    energy_tolerance: float = 1e-12,
    residual_tolerance: float = 1e-8,
    use_diis: bool = True,
) -> UHFResult:
    """UHF enerjisini alfa/beta Roothaan-Hall dongusuyle cozer."""
    basis = build_basis(molecule, basis_name)
    S = overlap_matrix(basis)
    Hc = core_hamiltonian(basis, molecule)
    eri = eri_tensor(basis)
    n_basis = S.shape[0]

    toplam = molecule.n_electrons
    na = toplam // 2 if n_alpha is None else int(n_alpha)
    nb = toplam - na
    if na < 0 or nb < 0 or na > n_basis or nb > n_basis:
        raise InvalidInputError(
            f"n_alpha gecersiz: n_alpha={na}, n_beta={nb}, baz boyutu "
            f"{n_basis}, elektron {toplam}"
        )

    X = _orthogonalisation_matrix(S)
    e_nuc = molecule.nuclear_repulsion()

    if initial_bias is None:
        bias = np.zeros(n_basis)
    else:
        bias = np.asarray(initial_bias, dtype=float)
        if bias.shape != (n_basis,):
            raise InvalidInputError(
                f"initial_bias ({n_basis},) olmali, verilen: {bias.shape}"
            )

    def coz(F: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e, c = np.linalg.eigh(X.T @ F @ X)
        C = X @ c
        return C, C[:, :k] @ C[:, :k].T, e

    Ca, Pa, ea = coz(Hc + np.diag(bias), na)
    Cb, Pb, eb = coz(Hc - np.diag(bias), nb)

    enerji = 0.0
    gecmis: list[float] = []
    hata_gecmisi: list[np.ndarray] = []
    fock_gecmisi: list[np.ndarray] = []
    artik = float("inf")

    for adim in range(1, max_iterations + 1):
        Pt = Pa + Pb
        # J TOPLAM yogunluktan; K HER SPIN icin AYRI. Indis desenleri FARKLI.
        J = np.einsum("uvls,ls->uv", eri, Pt)
        Ka = np.einsum("ulvs,ls->uv", eri, Pa)
        Kb = np.einsum("ulvs,ls->uv", eri, Pb)
        Fa, Fb = Hc + J - Ka, Hc + J - Kb

        yeni = 0.5 * float(
            np.sum(Pt * Hc) + np.sum(Pa * Fa) + np.sum(Pb * Fb)
        ) + e_nuc
        gecmis.append(yeni)

        hata_a = _commutator_residual(Fa, Pa, S, X)
        hata_b = _commutator_residual(Fb, Pb, S, X)
        artik = max(float(np.abs(hata_a).max()), float(np.abs(hata_b).max()))

        if (
            adim > 1
            and abs(yeni - enerji) < energy_tolerance
            and artik < residual_tolerance
        ):
            enerji = yeni
            break
        enerji = yeni

        if use_diis:
            # Alfa ve beta BIRLESIK genisletilir: iki kanalin hata vektorleri
            # ust uste yigilir ve tek bir katsayi kumesi cozulur.
            hata_gecmisi.append(np.vstack([hata_a, hata_b]))
            fock_gecmisi.append(np.vstack([Fa, Fb]))
            if len(fock_gecmisi) > _DIIS_MAX_HISTORY:
                hata_gecmisi.pop(0)
                fock_gecmisi.pop(0)
            if len(fock_gecmisi) >= 2:
                genis = _diis_extrapolate(fock_gecmisi, hata_gecmisi)
                if genis is not None:
                    Fa, Fb = genis[:n_basis], genis[n_basis:]

        Ca, Pa, ea = coz(Fa, na)
        Cb, Pb, eb = coz(Fb, nb)
    else:
        raise SCFConvergenceError(
            iterations=max_iterations,
            energy_history=gecmis,
            final_residual=artik,
            oscillating=False,
        )

    s2 = spin_squared(Ca, Cb, S, na, nb)
    kaynak = Provenance(
        method=f"UHF/{basis_name}",
        approximations=_UHF_YAKLASIMLAR,
        neglected=_UHF_IHMAL_EDILEN,
        expected_error=_UHF_BEKLENEN_HATA,
        converged=True,
        diagnostics={
            "iterations": float(adim),
            "final_residual": artik,
            "s_squared": s2,
            "n_alpha": float(na),
            "n_beta": float(nb),
        },
    )
    return UHFResult(
        energy_total=enerji,
        coefficients_alpha=Ca,
        coefficients_beta=Cb,
        density_alpha=Pa,
        density_beta=Pb,
        orbital_energies_alpha=ea,
        orbital_energies_beta=eb,
        s_squared=s2,
        n_alpha=na,
        n_beta=nb,
        provenance=kaynak,
    )
