"""Periyodik orgu geometrisi ve komsu bulma -- bohr biriminde.

Bu modul fizik yapmaz; yalniz geometri kurar. Iki olculmus sessiz hatanin
kaynagi burasidir ve ikisi de burada kapatilir:

1. TERS ORGUDE 2*pi. `b_i . a_j = 2*pi*delta_ij` konvansiyonu kullanilir.
   Olculdu: 2*pi dusurulurse grafende K noktasindaki aralik 0 yerine
   15.605535020 eV cikiyor -- ve Gamma noktasinda yapilan kontrol IKI
   konvansiyonda da geciyor (8.100000000), yani hata Gamma'dan gorulmez.

2. KESME YARICAPINDA KAYAN NOKTA. `d <= cutoff` ciplak yazilirsa, cutoff tam
   komsu mesafesine esit oldugunda hesaplanan mesafe bir-iki ulp yukari
   dusup komsu SESSIZCE atilir. Olculdu: 11 fcc orgu sabitinin 5'inde
   `d <= a/sqrt(2)` 12 yerine SIFIR komsu seciyor. Bu yuzden karsilastirma
   bagil toleransla yapilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from qlab.errors import InvalidInputError

# Kesme yaricapi karsilastirmasinda kullanilan bagil tolerans. Olculdu:
# fcc'de birinci kabuk a/sqrt(2), ikincisi a; aradaki bosluk a'nin %29'u,
# yani 1e-9 bir sonraki kabugu icine alamaz (test_kesme_yaricapi_bir_
# sonraki_kabugu_ALMIYOR bunu kilitler).
_KESME_TOLERANSI = 1e-9


@dataclass(frozen=True, eq=False)
class NeighbourPair:
    """Bir hoplama yolu: `i` orbitalinden, `cell` hucresindeki `j` orbitaline."""

    i: int
    j: int
    cell: tuple[int, int, int]
    distance: float


@dataclass(frozen=True, eq=False)
class Lattice:
    """Periyodik orgu. `vectors` (d,3), `basis` (n_atom,3), ikisi de BOHR.

    `d` boyut sayisidir (1, 2 ya da 3); vektorler 3B kartezyen olarak verilir
    ama yalnizca `d` tanesi periyodiktir. `basis` hucre icindeki atom
    konumlaridir ve KARTEZYENDIR (kesirli degil) -- kesirli/kartezyen
    karisikligi bu deponun disinda cok yapilan bir hatadir; burada tip
    duzeyinde degil belge duzeyinde onlenir ve testle kilitlenir.
    """

    vectors: np.ndarray
    basis: np.ndarray

    def __post_init__(self) -> None:
        vektorler = np.asarray(self.vectors, dtype=float)
        taban = np.asarray(self.basis, dtype=float)
        object.__setattr__(self, "vectors", vektorler)
        object.__setattr__(self, "basis", taban)

        if (
            vektorler.ndim != 2
            or vektorler.shape[1] != 3
            or not 1 <= vektorler.shape[0] <= 3
        ):
            raise InvalidInputError(
                f"orgu vektoru dizisi (d,3) olmali, 1 <= d <= 3; "
                f"verilen: {vektorler.shape}"
            )
        if taban.ndim != 2 or taban.shape[1] != 3 or taban.shape[0] < 1:
            raise InvalidInputError(
                f"taban dizisi (n_atom,3) olmali, n_atom >= 1; verilen: {taban.shape}"
            )
        if not np.all(np.isfinite(vektorler)) or not np.all(np.isfinite(taban)):
            raise InvalidInputError("orgu vektorleri ve taban konumlari sonlu olmali")

        tekil = np.linalg.svd(vektorler, compute_uv=False)
        if tekil.min() <= 1e-10 * tekil.max():
            raise InvalidInputError(
                f"orgu vektorleri dogrusal bagimli (tekil degerler {tekil})"
            )

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def n_atom(self) -> int:
        return int(self.basis.shape[0])


def reciprocal_vectors(lattice: Lattice) -> np.ndarray:
    """`b_i . a_j = 2*pi*delta_ij` saglayan (d,3) ters orgu vektorleri.

    d < 3 durumunda eksik yonler dik tamamlayicilarla doldurulur, cozum
    alinir ve yalniz ilk d satir dondurulur -- boylece 1B ve 2B orguler de
    ayni konvansiyonu tasir.
    """
    a = lattice.vectors
    d = a.shape[0]
    if d == 3:
        tam = a
    else:
        tam = a.copy()
        for _ in range(3 - d):
            _, _, vt = np.linalg.svd(tam)
            tam = np.vstack([tam, vt[-1]])
    b_tam = 2.0 * np.pi * np.linalg.inv(tam).T
    return b_tam[:d]


def neighbour_shells(
    lattice: Lattice, cutoff_bohr: float, max_cell: int = 2
) -> tuple[NeighbourPair, ...]:
    """`cutoff_bohr`'dan yakin (kendisi dahil) tum yonlu komsu ciftleri.

    Yonludur: (i, j, cell) varsa (j, i, -cell) de vardir. Sifir mesafe
    (i = j, cell = 0) haric tutulur.

    KESME KARSILASTIRMASI BAGIL TOLERANSLIDIR -- sebebi modul dokstring'inde,
    olculmus tuzak 2.
    """
    if not np.isfinite(cutoff_bohr) or cutoff_bohr <= 0.0:
        raise InvalidInputError(
            f"kesme yaricapi pozitif ve sonlu olmali: {cutoff_bohr}"
        )
    if max_cell < 1:
        raise InvalidInputError(f"max_cell en az 1 olmali, verilen: {max_cell}")

    d = lattice.dimension
    araliklar = [range(-max_cell, max_cell + 1)] * d + [range(0, 1)] * (3 - d)
    sinir = cutoff_bohr * (1.0 + _KESME_TOLERANSI)

    ciftler: list[NeighbourPair] = []
    for hucre in product(*araliklar):
        kayma = np.zeros(3)
        for eksen in range(d):
            kayma = kayma + hucre[eksen] * lattice.vectors[eksen]
        for i in range(lattice.n_atom):
            for j in range(lattice.n_atom):
                fark = lattice.basis[j] + kayma - lattice.basis[i]
                mesafe = float(np.linalg.norm(fark))
                if mesafe <= _KESME_TOLERANSI or mesafe > sinir:
                    continue
                ciftler.append(
                    NeighbourPair(
                        i=i,
                        j=j,
                        cell=tuple(int(x) for x in hucre),
                        distance=mesafe,
                    )
                )
    return tuple(ciftler)
