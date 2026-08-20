"""Bant yapisi, k-yolu ve bant kenarlari.

BANT KENARLARI IZGARADAN OKUNMAZ. Olculdu ve bu modulun en onemli tasarim
karari budur: grafende 601x601 izgara BZ minimum araligini 2.471886e-15 eV
verirken 600x600 1.635969e-02 eV veriyor -- ve DAHA SIK olan 1000x1000
1.698157e-02 eV, yani 601'den 13 MERTEBE kotu. Siklastirmak duzeltmiyor.

Ayni aile 1B'de de olculdu ve iki ayri bicimde cikiyor:
- PARITE: tekbicim izgarada bant genisligi TEK n icin 2|t|(1+cos(pi/n))
  (n=3 -> gercek degerin %75'i), CIFT n icin tam 4|t|. Hata monoton
  kuculuyor, yani bir YAKINSAMA SORUNU gibi gorunuyor.
- MONOTON OLMAMA: ikinci komsulu zincirde bant minimumu yuksek simetri
  noktasinda degil ve hata izgara sikligiyla duzenli AZALMIYOR --
  n=5 -> 2.426e-06, n=7 -> 1.934e-02, n=12 -> 7.031e-03, n=13 -> 7.370e-03.
  n=5, n=13'ten yaklasik 3000 kat iyi.

Bu yuzden `band_edges` surekli minimizasyon yapar; `uniform_k_grid` yalniz
cizim ve DOS icindir ve cagiranin bildirdigi k noktalarini icerip
icermedigini DENETLER.

SIHIRLI SAYI YOK. On olcumde iki farkli izgara konvansiyonu iki farkli
"dogru N" verdi (601 ve N % 3 == 0); ikisi de kendi konvansiyonunda dogru.
Bu yuzden parite kodda sabitlenmez, `must_include` ile denetlenir.

`band_structure` ciktisi SIRALAMADIR, BANT KIMLIGI DEGILDIR. Olculdu:
sirali "bant 0"in etkin kutlesi -6.42e-05, gercek bandinki -8.000e-01 --
12457 kat hata. Cizim icin dogru, bant-cozumlu nicelikler icin degil.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import minimize_scalar

from qlab.engine.tb.hamiltonian import TBModel, bloch_hamiltonian
from qlab.engine.tb.lattice import Lattice, reciprocal_vectors
from qlab.errors import InvalidInputError

# `must_include` uyelik denetiminin toleransi -- ters orgu TABANINDA
# kesirli koordinat uzerinden olculur, dolayisiyla boyutsuzdur ve mutlak
# bir enerji/uzunluk esigine bagli degildir.
_UYELIK_TOLERANSI = 1e-9


@dataclass(frozen=True, eq=False)
class KPath:
    """Yuksek simetri noktalarini birlestiren dogru parcalari."""

    points: np.ndarray
    distances: np.ndarray
    tick_distances: np.ndarray


def uniform_k_grid(
    lattice: Lattice, n: int, must_include: tuple[np.ndarray, ...] = ()
) -> np.ndarray:
    """Gamma merkezli tekbicim izgara, (n**d, 3). `must_include` DENETLENIR."""
    if n < 2:
        raise InvalidInputError(f"n en az 2 olmali, verilen: {n}")
    b = reciprocal_vectors(lattice)
    d = lattice.dimension
    kesirli = np.arange(n) / n
    noktalar = np.array(
        [
            sum(f[eksen] * b[eksen] for eksen in range(d))
            for f in product(kesirli, repeat=d)
        ]
    )
    for hedef in must_include:
        hedef = np.asarray(hedef, dtype=float)
        # Ters orgu periyodikligi: hedef, izgara noktasindan bir G kadar
        # uzakta olabilir. Bu yuzden fark ters orgu tabaninda KESIRLI
        # olarak alinip tam sayiya yakinligi sinanir.
        kesirli_fark = (noktalar - hedef) @ np.linalg.pinv(b)
        yakin = np.abs(kesirli_fark - np.rint(kesirli_fark)).max(axis=1)
        if float(yakin.min()) > _UYELIK_TOLERANSI:
            raise InvalidInputError(
                f"istenen k noktasi izgarada yok (en yakin kesirli sapma "
                f"{float(yakin.min()):.3e}). n'i degistirin; izgarayi "
                "SIKLASTIRMAK bu hatayi duzeltmez -- olculdu, ikinci komsulu "
                "zincirde n=5 hatasi 2.426e-06 iken n=13 hatasi 7.370e-03."
            )
    return noktalar


def band_structure(model: TBModel, kpoints: np.ndarray) -> np.ndarray:
    """(n_k, n_orb) SIRALI ozdegerler, hartree. Bant kimligi TASIMAZ."""
    kpoints = np.asarray(kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise InvalidInputError(f"kpoints (n_k,3) olmali: {kpoints.shape}")
    return np.array(
        [np.linalg.eigvalsh(bloch_hamiltonian(model, k)) for k in kpoints]
    )


def k_path(vertices: tuple[np.ndarray, ...], n_per_segment: int) -> KPath:
    """Koseleri birlestiren yol. Ortak noktalar bir kez sayilir."""
    if len(vertices) < 2:
        raise InvalidInputError(
            f"k-yolu en az iki kose ister, verilen: {len(vertices)}"
        )
    if n_per_segment < 2:
        raise InvalidInputError(f"n_per_segment en az 2 olmali: {n_per_segment}")

    parcalar: list[np.ndarray] = []
    for indis in range(len(vertices) - 1):
        bas = np.asarray(vertices[indis], dtype=float)
        son = np.asarray(vertices[indis + 1], dtype=float)
        t = np.linspace(0.0, 1.0, n_per_segment)
        parca = bas[None, :] + t[:, None] * (son - bas)[None, :]
        parcalar.append(parca if indis == 0 else parca[1:])
    noktalar = np.vstack(parcalar)

    adimlar = np.linalg.norm(np.diff(noktalar, axis=0), axis=1)
    uzakliklar = np.concatenate([[0.0], np.cumsum(adimlar)])
    kose_indisleri = [0]
    for _ in range(len(vertices) - 1):
        kose_indisleri.append(kose_indisleri[-1] + (n_per_segment - 1))
    return KPath(
        points=noktalar,
        distances=uzakliklar,
        tick_distances=uzakliklar[np.array(kose_indisleri)],
    )


def band_edges(model: TBModel, band_index: int) -> tuple[float, float]:
    """Bir bandin (min, max) enerjisi -- SUREKLI minimizasyonla, hartree.

    Izgara KULLANILMAZ; sebep modul dokstring'indeki olculmus tuzaktir.
    Her eksen sirayla Brent (bounded) ile taranir; birden cok baslangictan
    kosulur cunku bant coklu yerel uc deger tasiyabilir. Cok boyutta sonuc
    bir ALT SINIRDIR ve bu beyan edilir.
    """
    if not (0 <= band_index < model.n_orbital):
        raise InvalidInputError(
            f"bant indisi aralik disi: {band_index}, bant sayisi "
            f"{model.n_orbital}"
        )
    b = reciprocal_vectors(model.lattice)
    d = model.lattice.dimension

    def enerji(kesirli: np.ndarray, isaret: float) -> float:
        k = sum(kesirli[eksen] * b[eksen] for eksen in range(d))
        return isaret * float(
            np.linalg.eigvalsh(bloch_hamiltonian(model, k))[band_index]
        )

    uc_degerler: list[float] = []
    for isaret in (+1.0, -1.0):
        en_iyi: float | None = None
        for baslangic in product(np.linspace(0.0, 1.0, 9)[:-1], repeat=d):
            taban = np.array(baslangic, dtype=float)
            for eksen in range(d):

                def tek_eksen(x: float, taban=taban, eksen=eksen) -> float:
                    aday = taban.copy()
                    aday[eksen] = x
                    return enerji(aday, isaret)

                sonuc = minimize_scalar(
                    tek_eksen,
                    bounds=(0.0, 1.0),
                    method="bounded",
                    options={"xatol": 1e-12},
                )
                taban[eksen] = float(sonuc.x)
            deger = enerji(taban, isaret)
            en_iyi = deger if en_iyi is None else min(en_iyi, deger)
        uc_degerler.append(isaret * float(en_iyi))
    return (uc_degerler[0], uc_degerler[1])
