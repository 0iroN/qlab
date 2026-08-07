"""Kontrakte Gauss baz fonksiyonlari.

Baz verisi disaridan (reference/basis_*.json) okunur; koda hicbir us veya
katsayi elle yazilmaz. Bu dosyalar PySCF'in kendi baz verisidir (bkz.
reference/README.md) — elle kopyalanan baz seti sayilari klasik bir hata
kaynagidir.

Yalnizca s ve p kabuklari destekleniyor (spec §13 d kabugunu kapsam disi
birakiyor). `angular_momentum >= 2` icin BasisNotAvailableError firlatilir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from pathlib import Path

import numpy as np

from qlab.errors import BasisNotAvailableError
from qlab.types import Molecule

AVAILABLE_BASES: tuple[str, ...] = ("sto-3g", "6-31g")

_REFERENCE_DIR = Path(__file__).resolve().parents[4] / "reference"
_REFERENCE_FILES: dict[str, str] = {
    "sto-3g": "basis_sto3g.json",
    "6-31g": "basis_631g.json",
}

# angular_momentum -> kartezyen us ucluleri (s: tek fonksiyon, p: uc fonksiyon).
_ANGULAR_SHAPES: dict[int, tuple[tuple[int, int, int], ...]] = {
    0: ((0, 0, 0),),
    1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
}


@lru_cache(maxsize=None)
def _load_reference_data(name: str) -> dict:
    path = _REFERENCE_DIR / _REFERENCE_FILES[name]
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PrimitiveGaussian:
    exponent: float
    coefficient: float


@dataclass(frozen=True)
class BasisFunction:
    center: np.ndarray
    angular: tuple[int, int, int]
    primitives: tuple[PrimitiveGaussian, ...]
    atom_index: int

    def normalisation(self) -> np.ndarray:
        """Primitif basina normalizasyon carpani.

        Kartezyen Gauss primitifi x^i y^j z^k exp(-a r^2) icin:
        N = (2a/pi)^{3/4} * sqrt( (8a)^{i+j+k} i! j! k! / ((2i)! (2j)! (2k)!) )
        s tipi icin (i=j=k=0) bu (2a/pi)^{3/4}'e indirgenir.
        """
        i, j, k = self.angular
        toplam = i + j + k
        exponents = np.array([p.exponent for p in self.primitives])
        on_carpan = (2 * exponents / np.pi) ** 0.75
        if toplam == 0:
            return on_carpan
        pay = (8 * exponents) ** toplam * factorial(i) * factorial(j) * factorial(k)
        payda = factorial(2 * i) * factorial(2 * j) * factorial(2 * k)
        return on_carpan * np.sqrt(pay / payda)


def build_basis(molecule: Molecule, name: str) -> list[BasisFunction]:
    """Molekuldeki her atom icin kontrakte baz fonksiyonlarini kurar."""
    if name not in AVAILABLE_BASES:
        raise BasisNotAvailableError(
            f"Bilinmeyen baz seti: {name!r}. Desteklenenler: {AVAILABLE_BASES}"
        )
    veri = _load_reference_data(name)

    fonksiyonlar: list[BasisFunction] = []
    for atom_index, atom in enumerate(molecule.atoms):
        if atom.symbol not in veri:
            raise BasisNotAvailableError(
                f"{atom.symbol!r} elementi icin {name!r} baz setinde veri yok"
            )
        for kabuk in veri[atom.symbol]:
            angular_momentum = kabuk["angular_momentum"]
            if angular_momentum not in _ANGULAR_SHAPES:
                raise BasisNotAvailableError(
                    f"Desteklenmeyen acisal momentum {angular_momentum} "
                    f"({atom.symbol}, {name!r}). Yalnizca s ve p kabuklari destekleniyor."
                )
            satirlar = kabuk["primitives"]
            n_contraction = len(satirlar[0]) - 1
            for sutun in range(n_contraction):
                primitifler = tuple(
                    PrimitiveGaussian(exponent=satir[0], coefficient=satir[1 + sutun])
                    for satir in satirlar
                )
                for angular in _ANGULAR_SHAPES[angular_momentum]:
                    fonksiyonlar.append(
                        BasisFunction(
                            center=atom.position,
                            angular=angular,
                            primitives=primitifler,
                            atom_index=atom_index,
                        )
                    )
    return fonksiyonlar
