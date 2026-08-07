"""Potansiyel enerji profilleri (atomik birimlerde, hartree).

Split-operator FFT periyodik sınır koşulu kullanır: sağdan çıkan paket soldan
geri girer ve iletim ölçümünü sessizce bozar. absorbing_mask() bunun
panzehiridir; ölçüm testlerinde ise maske yerine yeterince büyük kutu tercih
edilir (norm tam korunur, ölçüm mutlak kalır).
"""

from __future__ import annotations

import numpy as np

from qlab.engine.tdse.grid import Grid1D, Grid2D
from qlab.errors import InvalidInputError


def free(grid: Grid1D) -> np.ndarray:
    return np.zeros(grid.n)


def rectangular_barrier(
    grid: Grid1D, height: float, width: float, center: float = 0.0
) -> np.ndarray:
    if width <= 0:
        raise InvalidInputError(f"Bariyer genişliği pozitif olmalı, verilen: {width}")
    return np.where(np.abs(grid.x - center) <= 0.5 * width, height, 0.0)


def square_well(
    grid: Grid1D, width: float, center: float = 0.0, wall_height: float = 1e6
) -> np.ndarray:
    if width <= 0:
        raise InvalidInputError(f"Kuyu genişliği pozitif olmalı, verilen: {width}")
    if wall_height <= 0:
        raise InvalidInputError(f"wall_height pozitif olmalı, verilen: {wall_height}")
    return np.where(np.abs(grid.x - center) <= 0.5 * width, 0.0, wall_height)


def harmonic(
    grid: Grid1D, omega: float, center: float = 0.0, mass: float = 1.0
) -> np.ndarray:
    if omega <= 0:
        raise InvalidInputError(f"omega pozitif olmalı, verilen: {omega}")
    if mass <= 0:
        raise InvalidInputError(f"mass pozitif olmalı, verilen: {mass}")
    return 0.5 * mass * omega**2 * (grid.x - center) ** 2


def uniform_field(grid: Grid1D, strength: float) -> np.ndarray:
    """V(x) = strength * x.

    strength > 0 ise parçacık -x yönünde ivmelenir. Yükün işareti bu fonksiyona
    dahil değildir; çağıran doğrudan potansiyel enerjiyi verir.
    """
    return strength * grid.x


def absorbing_mask(grid: Grid1D, width: float, softness: float = 8.0) -> np.ndarray:
    """Kenarlarda yumuşak soğurucu maske; her adımda psi bununla çarpılır.

    Sert kesim yansıma üretir; bu yüzden cos^(1/softness) profili kullanılır.
    Maske kullanıldığında norm bilinçli olarak azalır — üniterlik kontrolü
    devre dışı bırakılmalıdır.
    """
    if width <= 0 or width > 0.4 * grid.length:
        raise InvalidInputError(
            f"Soğurucu genişliği (0, 0.4*L] aralığında olmalı, "
            f"verilen: {width}, L={grid.length}"
        )
    x = grid.x
    mask = np.ones(grid.n)

    depth_left = (x[0] + width - x) / width
    in_left = depth_left > 0
    mask[in_left] = np.cos(0.5 * np.pi * depth_left[in_left]) ** (1.0 / softness)

    depth_right = (x - (x[-1] - width)) / width
    in_right = depth_right > 0
    mask[in_right] = np.cos(0.5 * np.pi * depth_right[in_right]) ** (1.0 / softness)

    # Ensure outermost grid points are exactly zero (floating-point precision)
    mask[0] = 0.0
    mask[-1] = 0.0

    return mask


def double_slit(
    grid: Grid2D,
    wall_x: float,
    thickness: float,
    slit_separation: float,
    slit_width: float,
    height: float = 1e4,
) -> np.ndarray:
    """x = wall_x konumunda, y ekseninde iki yarıklı duvar."""
    if slit_width <= 0 or slit_separation <= slit_width:
        raise InvalidInputError(
            "slit_separation, slit_width'ten büyük ve slit_width pozitif olmalı; "
            f"verilen: separation={slit_separation}, width={slit_width}"
        )
    in_wall = np.abs(grid.X - wall_x) <= 0.5 * thickness
    half_sep = 0.5 * slit_separation
    in_slit = (np.abs(grid.Y - half_sep) <= 0.5 * slit_width) | (
        np.abs(grid.Y + half_sep) <= 0.5 * slit_width
    )
    return np.where(in_wall & ~in_slit, height, 0.0)
