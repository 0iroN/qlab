"""Termostatlar: Langevin (BAOAB) ve Nosé-Hoover.

İkisi de `velocity_verlet_step` ile aynı imzayı taşır (`step`), yani
integratörün yerine geçebilirler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qlab.engine.md.state import ForceField, MDState
from qlab.errors import InvalidInputError
from qlab.units import KB_HARTREE_PER_KELVIN


@dataclass
class Langevin:
    """BAOAB ayrıştırmasıyla Langevin termostatı.

    B (yarım tekme) - A (yarım sürüklenme) - O (Ornstein-Uhlenbeck) -
    A (yarım sürüklenme) - B (yarım tekme).

    ÖLÇÜLMÜŞ (tasarım §4.5): hedef sıcaklığa oturuyor; gamma/omega = 1.0'da
    23 adımda <T> = 299.24 K. Oturma zamanı teorik 1/(2 gamma)'yı 4.47-5.78x
    aşıyor, bu yüzden çağıran 6/(2 gamma) kadar denge koşusu yapmalıdır.

    ÖLÇÜLMÜŞ UYARI: konfigürasyonel örnekleme dt'ye neredeyse duyarsızdır --
    dt*omega = 1.57'de kinetik sıcaklık %61.7 yanlışken <x^2> %0.109 doğru.
    Doğrulama KİNETİK SICAKLIKLA yapılmalıdır.
    """

    target_temperature_kelvin: float
    friction: float
    seed: int
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.target_temperature_kelvin)
            or self.target_temperature_kelvin <= 0.0
        ):
            raise InvalidInputError(
                f"hedef sıcaklık pozitif olmalı: {self.target_temperature_kelvin}"
            )
        if not np.isfinite(self.friction) or self.friction <= 0.0:
            raise InvalidInputError(f"sürtünme pozitif olmalı: {self.friction}")
        self._rng = np.random.default_rng(self.seed)

    def step(
        self, state: MDState, forces: np.ndarray, force_field: ForceField, dt: float
    ) -> tuple[MDState, float, np.ndarray]:
        kt = KB_HARTREE_PER_KELVIN * self.target_temperature_kelvin
        c1 = float(np.exp(-self.friction * dt))
        c2 = float(np.sqrt(1.0 - c1**2))

        v = state.velocities + 0.5 * dt * forces / state.masses[:, None]   # B
        x = state.positions + 0.5 * dt * v                                 # A
        sigma = np.sqrt(kt / state.masses)[:, None]
        v = c1 * v + c2 * sigma * self._rng.normal(size=v.shape)           # O
        x = x + 0.5 * dt * v                                               # A
        enerji, yeni_kuvvet = force_field(x)
        v = v + 0.5 * dt * yeni_kuvvet / state.masses[:, None]             # B

        return (
            MDState(positions=x, velocities=v, masses=state.masses),
            enerji,
            yeni_kuvvet,
        )
