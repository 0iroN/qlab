"""Gerçek uzay ve momentum uzayı ızgaraları.

Tüm uzunluklar bohr, dalga sayıları 1/bohr. Split-operator FFT kullandığı için
ızgara periyodiktir — bu, sonuçları etkileyen gerçek bir yaklaşımdır ve
Provenance'ta beyan edilir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError


def _validate(n: int, length: float, name: str) -> None:
    if n < 8 or n % 2 != 0:
        raise InvalidInputError(f"{name} çift ve en az 8 olmalı, verilen: {n}")
    if length <= 0:
        raise InvalidInputError(f"{name} uzunluğu pozitif olmalı, verilen: {length}")


@dataclass(frozen=True)
class Grid1D:
    n: int
    length: float

    def __post_init__(self) -> None:
        _validate(self.n, self.length, "n")

    @property
    def dx(self) -> float:
        return self.length / self.n

    @property
    def x(self) -> np.ndarray:
        return -0.5 * self.length + self.dx * np.arange(self.n)

    @property
    def k(self) -> np.ndarray:
        return 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)

    @property
    def k2(self) -> np.ndarray:
        return self.k**2


@dataclass(frozen=True)
class Grid2D:
    nx: int
    ny: int
    lx: float
    ly: float

    def __post_init__(self) -> None:
        _validate(self.nx, self.lx, "nx")
        _validate(self.ny, self.ly, "ny")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def _axes(self) -> tuple[np.ndarray, np.ndarray]:
        x = -0.5 * self.lx + self.dx * np.arange(self.nx)
        y = -0.5 * self.ly + self.dy * np.arange(self.ny)
        return x, y

    @property
    def X(self) -> np.ndarray:
        x, y = self._axes
        return np.meshgrid(x, y, indexing="ij")[0]

    @property
    def Y(self) -> np.ndarray:
        x, y = self._axes
        return np.meshgrid(x, y, indexing="ij")[1]

    @property
    def k2(self) -> np.ndarray:
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        return KX**2 + KY**2
