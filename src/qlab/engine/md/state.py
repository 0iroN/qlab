"""MD durumu ve kuvvet sağlayıcı sözleşmesi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qlab.errors import InvalidInputError


class ForceField(Protocol):
    """Kuvvet sağlayıcı sözleşmesi -- bu modülün tek genişleme noktası.

    `(konumlar: (N,3) bohr) -> (potansiyel enerji: hartree,
    kuvvetler: (N,3) hartree/bohr)`

    İntegratör bundan başka bir şey bilmez. Plan D'nin Hartree-Fock kuvvet
    sağlayıcısı da tam olarak bunu uygular; integratör, termostat ve
    gözlemler değişmez.
    """

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]: ...


@dataclass(frozen=True, eq=False)
class MDState:
    """Bir anlık MD durumu.

    `positions` ve `velocities` (N,3), `masses` (N,) ve ELEKTRON KÜTLESİ
    cinsindendir (bkz. modül dokstring'i).

    `eq=False`: alanlar ndarray olduğu için üretilen `__eq__` bir dizi
    döndürür ve `__hash__` `TypeError` verir. Depo bu tuzağı
    `engine/hf/integrals.py`'de zaten çözmüş; aynı gerekçe.
    """

    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray

    def __post_init__(self) -> None:
        # ÖNCE ndarray'e çevir: liste girdisi doğrulamayı geçip sonra
        # `kinetic_energy()` içinde ham `TypeError` veriyordu (ölçüldü).
        for ad in ("positions", "velocities", "masses"):
            object.__setattr__(
                self, ad, np.asarray(getattr(self, ad), dtype=float)
            )

        sekil = self.positions.shape
        if len(sekil) != 2 or sekil[1] != 3 or sekil[0] < 1:
            raise InvalidInputError(
                f"konumlar (N,3) şeklinde olmalı, verilen: {sekil}"
            )
        n = sekil[0]
        if self.velocities.shape != (n, 3):
            raise InvalidInputError(
                f"hızların şekli konumlarla aynı olmalı: "
                f"{self.velocities.shape} != {(n, 3)}"
            )
        if self.masses.shape != (n,):
            raise InvalidInputError(
                f"kütlelerin şekli (N,) olmalı: {self.masses.shape} != {(n,)}"
            )
        for ad, dizi in (
            ("konumlar", self.positions),
            ("hızlar", self.velocities),
            ("kütleler", self.masses),
        ):
            if not np.all(np.isfinite(dizi)):
                raise InvalidInputError(f"{ad} sonlu olmalı (nan/inf kabul edilmez)")
        if not np.all(self.masses > 0.0):
            raise InvalidInputError(f"kütle pozitif olmalı, verilen: {self.masses}")

    def kinetic_energy(self) -> float:
        """Toplam kinetik enerji (hartree)."""
        return float(0.5 * (self.masses[:, None] * self.velocities**2).sum())

    def remove_center_of_mass_motion(self) -> MDState:
        """Kütle merkezi hızını çıkarır.

        Sürüklenen bir kütle merkezi sıcaklığa SAHTE katkı yapar; ndof = 3N-3
        kullanılacaksa (bkz. `observable/md_obs.py`) bu üç serbestlik derecesi
        gerçekten çıkarılmış olmalıdır. Hız KÜTLE-AĞIRLIKLI ortalamadır; düz
        ortalama kullanan bir uygulama artık momentum bırakır (ölçüldü).
        """
        toplam = float(self.masses.sum())
        v_com = (self.masses[:, None] * self.velocities).sum(axis=0) / toplam
        return MDState(
            positions=self.positions,
            velocities=self.velocities - v_com,
            masses=self.masses,
        )
