"""Bloch Hamiltoniyeni H(k) ve sonlu halka matrisi -- HARTREE biriminde.

AYAR (GAUGE) SECIMI. Iki konvansiyon var: "hucre" gauge'unda faz yalniz orgu
otelemesini tasir, exp(i k . R_hucre); "atom" gauge'unda taban konumlari da
girer, exp(i k . (R_hucre + tau_j - tau_i)). Varsayilan HUCREDIR, cunku o
konvansiyonda H(k) ters orgu vektoru altinda birebir periyodiktir ve
periyodiklik testi H uzerinden yazilabilir.

OLCULDU ve BEYAN EDILIYOR: iki gauge H'nin ELEMANLARINI 5.20 eV'ye varan
farkla degistirir ama OZDEGERLERI 8 ULP'ta ayni birakir. Sonucu: iki
uygulamayi H uzerinden karsilastirmak YANILTICIDIR; karsilastirma ozdeger
uzerinden yapilir.

`eigh` TUZAGI. `np.linalg.eigh` varsayilan UPLO='L' ile yalniz ALT ucgeni
okur; olculdu, ust ucgene +1000 eklemek ozdegerleri TAM 0.000e+00
degistiriyor. Bu modul H'nin TAMAMINI kurar ve testler H'yi dogrudan
denetler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.tb.lattice import Lattice, neighbour_shells
from qlab.errors import InvalidInputError


@dataclass(frozen=True, eq=False)
class Hopping:
    """`i` orbitalinden `cell` hucresindeki `j` orbitaline hoplama, HARTREE.

    Isaret konvansiyonu: H'nin kosegen disi elemani DOGRUDAN `t`'dir
    (H[i,j] += t * faz). Literaturdeki "-t" konvansiyonu KULLANILMAZ;
    donusum cagiranin isidir ve senaryo sinirinda yapilir.
    """

    i: int
    j: int
    cell: tuple[int, int, int]
    t: float


@dataclass(frozen=True, eq=False)
class TBModel:
    """Orgu + yerinde enerjiler + hoplama listesi. Hepsi HARTREE."""

    lattice: Lattice
    onsite: np.ndarray
    hoppings: tuple[Hopping, ...]

    def __post_init__(self) -> None:
        yerinde = np.asarray(self.onsite, dtype=float)
        object.__setattr__(self, "onsite", yerinde)
        if yerinde.ndim != 1 or yerinde.shape[0] != self.lattice.n_atom:
            raise InvalidInputError(
                f"onsite dizisi (n_atom,) olmali: {yerinde.shape} != "
                f"({self.lattice.n_atom},)"
            )
        if not np.all(np.isfinite(yerinde)):
            raise InvalidInputError("onsite enerjileri sonlu olmali")
        for h in self.hoppings:
            if not (0 <= h.i < self.lattice.n_atom) or not (
                0 <= h.j < self.lattice.n_atom
            ):
                raise InvalidInputError(
                    f"orbital indisi aralik disi: i={h.i}, j={h.j}, "
                    f"atom sayisi {self.lattice.n_atom}"
                )
            if not np.isfinite(h.t):
                raise InvalidInputError(f"hoplama sonlu olmali: t={h.t}")

    @property
    def n_orbital(self) -> int:
        return int(self.onsite.shape[0])


def bloch_hamiltonian(
    model: TBModel, k: np.ndarray, gauge: str = "cell"
) -> np.ndarray:
    """(n_orb, n_orb) kompleks Hermityen H(k). `k` 1/bohr, sonuc hartree."""
    if gauge not in ("cell", "atom"):
        raise InvalidInputError(f"gauge 'cell' ya da 'atom' olmali: {gauge!r}")
    k = np.asarray(k, dtype=float)
    if k.shape != (3,):
        raise InvalidInputError(f"k (3,) olmali, verilen: {k.shape}")

    n = model.n_orbital
    H = np.zeros((n, n), dtype=complex)
    np.fill_diagonal(H, model.onsite)

    vektorler = model.lattice.vectors
    d = model.lattice.dimension
    for h in model.hoppings:
        kayma = np.zeros(3)
        for eksen in range(d):
            kayma = kayma + h.cell[eksen] * vektorler[eksen]
        if gauge == "atom":
            kayma = kayma + model.lattice.basis[h.j] - model.lattice.basis[h.i]
        H[h.i, h.j] += h.t * np.exp(1j * float(k @ kayma))
    return H


def uniform_model(
    lattice: Lattice, onsite: np.ndarray, cutoff_bohr: float, t: float
) -> TBModel:
    """Kesme yaricapindan hoplama listesi kuran kisayol -- TEKBICIM `t` ile.

    `neighbour_shells`in TEK tuketicisi budur ve bilinclidir: kesme
    yaricapindaki kayan nokta tuzagi (11 fcc orgu sabitinin 5'inde 12 yerine
    0 komsu) ancak gercekten kullanilan bir yolda korunur. Elle `Hopping`
    listesi yazmak da mesrudur ve testlerde oyle yapilir; bu kisayol
    yuksek koordinasyonlu kafesler (fcc'de 12 komsu) icin vardir.

    `t` TEKBICIMDIR: her komsu ciftine ayni deger konur. Mesafeye ya da yone
    bagli hoplama (Slater-Koster) bu fonksiyonun isi DEGILDIR -- o Plan G'nin
    konusudur ve burada sessizce yaklasilmaz.
    """
    return TBModel(
        lattice=lattice,
        onsite=onsite,
        hoppings=tuple(
            Hopping(i=c.i, j=c.j, cell=c.cell, t=t)
            for c in neighbour_shells(lattice, cutoff_bohr)
        ),
    )


def finite_ring_matrix(model: TBModel, n_cell: int) -> np.ndarray:
    """Periyodik sinirli SONLU halkanin gercek-uzay matrisi (n_cell*n_orb kare).

    Bu, `bloch_hamiltonian`'dan TAMAMEN AYRI bir kod yoludur: hicbir faz
    carpani, hicbir ters orgu, hicbir kompleks us kullanmaz -- yalnizca
    hucre indislerini modulo n_cell baglar. Task 3'un capraz kontrolu bu
    ayriliga dayanir.
    """
    if n_cell < 3:
        raise InvalidInputError(f"n_cell en az 3 olmali, verilen: {n_cell}")
    if model.lattice.dimension != 1:
        raise InvalidInputError(
            f"sonlu halka yalniz 1B orgu icin tanimli, boyut: "
            f"{model.lattice.dimension}"
        )
    n_orb = model.n_orbital
    boyut = n_cell * n_orb
    H = np.zeros((boyut, boyut), dtype=float)
    for hucre in range(n_cell):
        for i in range(n_orb):
            H[hucre * n_orb + i, hucre * n_orb + i] += model.onsite[i]
    for h in model.hoppings:
        for hucre in range(n_cell):
            hedef = (hucre + h.cell[0]) % n_cell
            H[hucre * n_orb + h.i, hedef * n_orb + h.j] += h.t
    return H
