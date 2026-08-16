"""MD zaman serisi çizim yükü. Fizik üretmez; birim çevirir ve toplar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.md.integrator import Trajectory
from qlab.engine.md.state import MDState
from qlab.observable.md_obs import instantaneous_temperature
from qlab.units import AU_TIME_TO_FEMTOSECOND


@dataclass(frozen=True, eq=False)
class MDPayload:
    times_fs: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    total: np.ndarray
    temperature_kelvin: np.ndarray


def build_md_payload(
    trajectory: Trajectory, masses: np.ndarray, n_constraints: int = 3
) -> MDPayload:
    sicaklik = np.array(
        [
            instantaneous_temperature(
                MDState(positions=p, velocities=v, masses=masses),
                n_constraints=n_constraints,
            )
            for p, v in zip(trajectory.positions, trajectory.velocities)
        ]
    )
    return MDPayload(
        times_fs=trajectory.times * AU_TIME_TO_FEMTOSECOND,
        kinetic=trajectory.kinetic,
        potential=trajectory.potential,
        total=trajectory.kinetic + trajectory.potential,
        temperature_kelvin=sicaklik,
    )
