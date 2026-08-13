"""velocity-Verlet integratörü.

SİMPLEKTİKTİR: toplam enerji SÜRÜKLENMEZ, SALINIR. İki büyüklük 10^6 kat
farklıdır (ölçüldü) ve karıştırılmamalıdır -- "enerji korunuyor" iddiası
sürüklenmeyle sınanır, salınım ayrı ve daha keskin bir sınavdır.

Kararlılık sınırı ölçüldü: omega*dt = 2.0 (1.999 kararlı, 2.001 patlıyor).
Bu, kullanıcıya `factory.build` doğrulamasında bildirilir (Task 12).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.md.state import ForceField, MDState
from qlab.errors import InvalidInputError


@dataclass(frozen=True, eq=False)
class Trajectory:
    """Kaydedilen anlar. `positions`/`velocities` (M,N,3), ötekiler (M,)."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray


def velocity_verlet_step(
    state: MDState, forces: np.ndarray, force_field: ForceField, dt: float
) -> tuple[MDState, float, np.ndarray]:
    """Tek adım. Kuvvetler DIŞARIDAN gelir ve yenisi geri döner.

    Kuvveti çağıranın taşıması bilinçlidir: adım başına kuvvet alanı TAM BİR
    KEZ çağrılır. Pahalı alanlarda (Plan D'nin Hartree-Fock'u, adım başına
    12 SCF) bu, maliyeti ikiye katlamamanın tek yoludur.
    """
    ivme = forces / state.masses[:, None]
    yarim_hiz = state.velocities + 0.5 * dt * ivme
    yeni_konum = state.positions + dt * yarim_hiz
    enerji, yeni_kuvvet = force_field(yeni_konum)
    yeni_hiz = yarim_hiz + 0.5 * dt * yeni_kuvvet / state.masses[:, None]
    return (
        MDState(positions=yeni_konum, velocities=yeni_hiz, masses=state.masses),
        enerji,
        yeni_kuvvet,
    )


def run(
    state: MDState,
    force_field: ForceField,
    dt: float,
    n_steps: int,
    save_every: int = 1,
) -> Trajectory:
    """`n_steps` adım koşar, her `save_every` adımda bir anı kaydeder."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise InvalidInputError(f"dt pozitif ve sonlu olmalı, verilen: {dt}")
    if n_steps < 1:
        raise InvalidInputError(f"n_steps en az 1 olmalı, verilen: {n_steps}")
    if save_every < 1:
        raise InvalidInputError(f"save_every en az 1 olmalı, verilen: {save_every}")

    enerji, kuvvetler = force_field(state.positions)
    zamanlar, konumlar, hizlar = [0.0], [state.positions], [state.velocities]
    kinetik, potansiyel = [state.kinetic_energy()], [enerji]

    for adim in range(1, n_steps + 1):
        state, enerji, kuvvetler = velocity_verlet_step(
            state, kuvvetler, force_field, dt
        )
        if adim % save_every == 0:
            zamanlar.append(adim * dt)
            konumlar.append(state.positions)
            hizlar.append(state.velocities)
            kinetik.append(state.kinetic_energy())
            potansiyel.append(enerji)

    return Trajectory(
        times=np.array(zamanlar),
        positions=np.array(konumlar),
        velocities=np.array(hizlar),
        kinetic=np.array(kinetik),
        potential=np.array(potansiyel),
    )
