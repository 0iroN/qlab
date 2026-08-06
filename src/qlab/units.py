"""Atomik birimler ve SI dönüşümleri.

Motorların içi baştan sona atomik birimlerdedir (hbar = m_e = e = 4*pi*eps0 = 1).
Dönüşüm YALNIZCA giriş (senaryo okuma) ve çıkış (görselleştirme, rapor)
sınırlarında yapılır. Bu dosya projenin tek dönüşüm noktasıdır; başka hiçbir
yerde çıplak dönüşüm katsayısı bulunmamalıdır.

Kaynak: CODATA 2018.
"""

from __future__ import annotations

import numpy as np

BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM

HARTREE_TO_EV: float = 27.211386245988
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV

AU_TIME_TO_SECOND: float = 2.4188843265857e-17
AU_TIME_TO_FEMTOSECOND: float = AU_TIME_TO_SECOND * 1e15

AU_DIPOLE_TO_DEBYE: float = 2.541746473


def bohr_to_angstrom(x: float | np.ndarray) -> float | np.ndarray:
    return x * BOHR_TO_ANGSTROM


def angstrom_to_bohr(x: float | np.ndarray) -> float | np.ndarray:
    return x * ANGSTROM_TO_BOHR


def hartree_to_ev(x: float | np.ndarray) -> float | np.ndarray:
    return x * HARTREE_TO_EV


def ev_to_hartree(x: float | np.ndarray) -> float | np.ndarray:
    return x * EV_TO_HARTREE


def au_time_to_femtosecond(x: float | np.ndarray) -> float | np.ndarray:
    return x * AU_TIME_TO_FEMTOSECOND
