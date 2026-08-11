"""Katmanlar arasi ortak veri tipleri: Atom ve Molecule.

Konumlar HER ZAMAN bohr olarak saklanir. Angstrom yalnizca from_string()
girisinde kabul edilir ve orada donusturulur — global kural geregi donusum
sinirdadir, motorun icinde degil.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from qlab.errors import InvalidInputError
from qlab.units import ANGSTROM_TO_BOHR

ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6,
    "N": 7, "O": 8, "F": 9, "Ne": 10,
}

_MIN_SEPARATION_BOHR = 1e-6


@dataclass(frozen=True)
class Atom:
    symbol: str
    position: np.ndarray

    def __post_init__(self) -> None:
        if self.symbol not in ATOMIC_NUMBERS:
            raise InvalidInputError(
                f"Bilinmeyen element: {self.symbol!r}. "
                f"Desteklenenler: {sorted(ATOMIC_NUMBERS)}"
            )
        if np.shape(self.position) != (3,):
            raise InvalidInputError(
                f"Atom konumu uc bilesenli olmali, verilen sekil: "
                f"{np.shape(self.position)}"
            )
        # Sonluluk EN ERKEN sinirda denetlenir, cunku HER kurulum yolu
        # (dogrudan, from_string, dataclasses.replace) buradan gecer.
        #
        # Olculdu -- kontrol yokken hata cok sonra ve YANLIS YERDEN geliyordu:
        # `solve()` yuz SCF iterasyonu kosup "SCF 100 iterasyonda yakinsamadi
        # (son residual nan)" firlatiyordu (scf.py:406), yani tani yakinsamayi
        # sucluyordu. Yol boyunca integrals.py'nin bes ayri satirindan
        # RuntimeWarning yagiyordu.
        #
        # Uc ayri SESSIZ yol daha vardi:
        #   - Molecule'un cakisik-cekirdek kontrolu DELINIYORDU: nan'da
        #     `distance < _MIN_SEPARATION_BOHR` False doner, yani BIREBIR AYNI
        #     nan konumundaki iki atom "cakisik degil" sayiliyordu; iki atom da
        #     +inf'teyken de oyle (inf - inf = nan).
        #   - `nuclear_repulsion()` sonsuzdaki bir atom icin **0.0** veriyordu
        #     (Z1*Z2/inf). Sifir, bir enerji dokumunde tamamen makul gorunur;
        #     nan en azindan kendini belli eder.
        #   - `float("1e400")` sessizce `inf`'e tasar: girdide sonsuz diye bir
        #     sey YAZMAZ ama sonucu sonsuzdur.
        if not np.all(np.isfinite(np.asarray(self.position, dtype=float))):
            raise InvalidInputError(
                f"{self.symbol!r} atomunun koordinatlari sonlu degil: "
                f"{tuple(float(x) for x in np.asarray(self.position, dtype=float))}. "
                "Her koordinat sonlu bir sayi olmali; 'nan'/'inf' bir konum degildir."
            )

    @property
    def atomic_number(self) -> int:
        return ATOMIC_NUMBERS[self.symbol]


@dataclass(frozen=True)
class Molecule:
    atoms: tuple[Atom, ...]
    charge: int = 0

    def __post_init__(self) -> None:
        if not self.atoms:
            raise InvalidInputError("Molekul en az bir atom icermeli")
        for first, second in combinations(self.atoms, 2):
            distance = float(np.linalg.norm(first.position - second.position))
            if distance < _MIN_SEPARATION_BOHR:
                raise InvalidInputError(
                    f"cakisik cekirdek: {first.symbol} ve {second.symbol} "
                    f"arasi {distance:.2e} bohr"
                )

    @property
    def n_electrons(self) -> int:
        return sum(a.atomic_number for a in self.atoms) - self.charge

    def nuclear_repulsion(self) -> float:
        total = 0.0
        for first, second in combinations(self.atoms, 2):
            distance = float(np.linalg.norm(first.position - second.position))
            total += first.atomic_number * second.atomic_number / distance
        return total

    @classmethod
    def from_string(
        cls, spec: str, unit: str = "angstrom", charge: int = 0
    ) -> Molecule:
        """'H 0 0 0; O 0 0 1.2' bicimindeki tanimi Molecule'e cevirir."""
        if unit not in ("angstrom", "bohr"):
            raise InvalidInputError(
                f"unit 'angstrom' veya 'bohr' olmali, verilen: {unit!r}"
            )
        scale = ANGSTROM_TO_BOHR if unit == "angstrom" else 1.0

        atoms = []
        for entry in spec.split(";"):
            parts = entry.split()
            if len(parts) != 4:
                raise InvalidInputError(
                    f"Atom tanimi 'Sembol x y z' biciminde olmali, verilen: {entry!r}"
                )
            try:
                coordinates = np.array([float(v) for v in parts[1:]]) * scale
            except ValueError as exc:
                raise InvalidInputError(
                    f"Atom koordinatlari sayiya donusturulemedi: {entry!r}"
                ) from exc
            atoms.append(Atom(parts[0], coordinates))

        return cls(tuple(atoms), charge=charge)
