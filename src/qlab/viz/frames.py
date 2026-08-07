"""Bir karede ekrana ne çizileceğine karar veren saf mantık.

Widget'lar (app katmanı) yalnızca bu payload'ı basar. Ayrım sayesinde çizim
kararları GUI başlatmadan test edilebilir.

Potansiyel ölçeklendirmesi: V(x) hartree, |psi|^2 ise 1/bohr birimindedir. Aynı
eksende okunabilmesi için potansiyel yoğunluğun tepe değerinin %80'ine
ölçeklenir — yalnızca gösterim içindir, hiçbir hesaba girmez.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.tdse.grid import Grid1D
from qlab.observable import tdse_obs

_DISPLAY_FRACTION = 0.8


@dataclass(frozen=True)
class FramePayload:
    x: np.ndarray
    density: np.ndarray
    potential_scaled: np.ndarray
    phase: np.ndarray
    transmitted: float
    norm: float


def potential_display_scale(density: np.ndarray, potential: np.ndarray) -> float:
    peak = float(np.max(np.abs(potential)))
    if peak == 0.0:
        return 0.0
    return _DISPLAY_FRACTION * float(np.max(density)) / peak


def build_payload(
    psi: np.ndarray, grid: Grid1D, potential: np.ndarray, boundary: float
) -> FramePayload:
    density = tdse_obs.density(psi)
    scale = potential_display_scale(density, potential)
    return FramePayload(
        x=grid.x,
        density=density,
        potential_scaled=potential * scale,
        phase=np.angle(psi),
        transmitted=tdse_obs.probability_beyond(psi, grid, boundary),
        norm=tdse_obs.norm(psi, grid),
    )
