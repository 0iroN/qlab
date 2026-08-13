"""Analitik kuvvet alanları -- `state.ForceField` sözleşmesinin uygulamaları.

Her alanın ANALİTİK ÇIPALARI vardır ve testler onları kullanır; Maxwell-
Boltzmann bir kuvvet alanı sınavı DEĞİLDİR (bkz. tests/md/test_forcefields.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError


@dataclass(frozen=True, eq=False)
class Harmonic:
    """Her parçacığı kendi merkezine bağlayan izotropik yay.

    V = sum_i 0.5 k |r_i - c_i|^2 ,  F_i = -k (r_i - c_i)

    İntegratörün doğrulanması için var: periyot T = 2*pi*sqrt(m/k) analitiktir
    ve enerji salınımının kapalı formu bilinir (bkz. test_integrator.py).

    `center` (M,3) olmalıdır. M = N ise parçacık başına ayrı merkez, M = 1 ise
    hepsi tek merkeze bağlanır (yayılım bilerek korunuyor).
    """

    k: float
    center: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.k) or self.k <= 0.0:
            raise InvalidInputError(f"yay sabiti pozitif ve sonlu olmalı: {self.k}")
        merkez = np.asarray(self.center, dtype=float)
        if merkez.ndim != 2 or merkez.shape[1] != 3:
            raise InvalidInputError(
                f"merkez (M,3) şeklinde olmalı, verilen: {merkez.shape}"
            )
        if not np.all(np.isfinite(merkez)):
            raise InvalidInputError(
                "merkez sonlu olmalı (nan/inf sessizce enerjiye ve kuvvete yayılır)"
            )
        object.__setattr__(self, "center", merkez)

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        yer_degistirme = positions - self.center
        enerji = 0.5 * self.k * float((yer_degistirme**2).sum())
        return enerji, -self.k * yer_degistirme
