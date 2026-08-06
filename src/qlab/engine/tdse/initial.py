"""Başlangıç dalga fonksiyonları.

Gauss paketi konvansiyonu:
    psi(x) ~ exp(-(x-x0)^2 / (4 sigma^2)) * exp(i k0 x)
Bu seçimle |psi|^2 standart sapması tam olarak sigma, <x> = x0, <p> = k0 olur
(atomik birimlerde p = k).

İki girdi doğrulaması sessiz veri bozulmasını engeller:
  1. sigma ızgara adımına göre çok küçükse paket çözünmez.
  2. |k0| + 2/sigma Nyquist sınırını aşarsa momentum bileşenleri katlanır
     (aliasing) ve sonuç hiçbir uyarı vermeden yanlış çıkar.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.tdse.grid import Grid1D, Grid2D
from qlab.errors import InvalidInputError

_NYQUIST_SAFETY = 0.8
_MIN_POINTS_PER_SIGMA = 2.0


def _validate(sigma: float, k0_magnitude: float, dx: float) -> None:
    if sigma <= 0:
        raise InvalidInputError(f"sigma pozitif olmalı, verilen: {sigma}")
    if sigma < _MIN_POINTS_PER_SIGMA * dx:
        raise InvalidInputError(
            f"sigma ({sigma}) ızgara adımının ({dx}) en az "
            f"{_MIN_POINTS_PER_SIGMA} katı olmalı — aksi halde paket çözünmez."
        )
    k_max = np.pi / dx
    k_needed = k0_magnitude + 2.0 / sigma
    if k_needed > _NYQUIST_SAFETY * k_max:
        raise InvalidInputError(
            f"Paketin momentum içeriği ızgaraya sığmıyor: gereken |k| ~ "
            f"{k_needed:.3f}, güvenli sınır {_NYQUIST_SAFETY * k_max:.3f}. "
            "n'i artır ya da k0/sigma değerlerini küçült."
        )


def gaussian_packet(grid: Grid1D, x0: float, k0: float, sigma: float) -> np.ndarray:
    _validate(sigma, abs(k0), grid.dx)
    envelope = np.exp(-((grid.x - x0) ** 2) / (4.0 * sigma**2))
    psi = (envelope * np.exp(1j * k0 * grid.x)).astype(np.complex128)
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * grid.dx)


def gaussian_packet_2d(
    grid: Grid2D, x0: float, y0: float, kx0: float, ky0: float, sigma: float
) -> np.ndarray:
    _validate(sigma, float(np.hypot(kx0, ky0)), min(grid.dx, grid.dy))
    envelope = np.exp(
        -(((grid.X - x0) ** 2) + ((grid.Y - y0) ** 2)) / (4.0 * sigma**2)
    )
    phase = np.exp(1j * (kx0 * grid.X + ky0 * grid.Y))
    psi = (envelope * phase).astype(np.complex128)
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * grid.cell_area)
