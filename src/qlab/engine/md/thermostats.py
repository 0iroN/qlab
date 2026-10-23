"""Termostatlar: Langevin (BAOAB) ve Nosé-Hoover.

İkisi de `velocity_verlet_step` ile aynı imzayı taşır (`step`), yani
integratörün yerine geçebilirler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qlab.engine.md.state import ForceField, MDState, _bool_kapisi
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
        _bool_kapisi(  # B3 (grup 9; ölçüm `state._bool_kapisi`nda)
            "Langevin", target_temperature_kelvin=self.target_temperature_kelvin,
            friction=self.friction, seed=self.seed,
        )
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


@dataclass
class NoseHoover:
    """Nosé-Hoover termostatı, isteğe bağlı zincirle. (plan Task 10)"""

    target_temperature_kelvin: float
    relaxation_time: float
    chain_length: int = 1
    n_dof: int | None = None            # YENİ: None -> v.size (kısıtsız sistem)
    zeta: np.ndarray = field(init=False, repr=False)   # ARTIK PUBLIC (dinamik durum)

    def __post_init__(self) -> None:
        _bool_kapisi(  # B3 (grup 9)
            "NoseHoover", target_temperature_kelvin=self.target_temperature_kelvin,
            relaxation_time=self.relaxation_time, chain_length=self.chain_length, n_dof=self.n_dof,
        )
        if (not np.isfinite(self.target_temperature_kelvin)
                or self.target_temperature_kelvin <= 0.0):
            raise InvalidInputError(
                f"hedef sıcaklık pozitif olmalı: {self.target_temperature_kelvin}")
        if not np.isfinite(self.relaxation_time) or self.relaxation_time <= 0.0:
            raise InvalidInputError(
                f"gevşeme süresi pozitif olmalı: {self.relaxation_time}")
        if self.chain_length < 1:
            raise InvalidInputError(
                f"zincir uzunluğu en az 1 olmalı: {self.chain_length}")
        if self.n_dof is not None and self.n_dof < 1:
            raise InvalidInputError(
                f"serbestlik derecesi en az 1 olmalı: {self.n_dof}")
        self.zeta = np.zeros(self.chain_length)

    @property
    def thermostat_mass(self) -> float:
        """Termostat kütlesi Q = k_B T tau^2."""
        return (KB_HARTREE_PER_KELVIN * self.target_temperature_kelvin
                * self.relaxation_time**2)

    def _serbestlik(self, v: np.ndarray) -> float:
        return float(v.size) if self.n_dof is None else float(self.n_dof)

    def _ileri_yarim(self, v, masses, h):
        kt = KB_HARTREE_PER_KELVIN * self.target_temperature_kelvin
        q = self.thermostat_mass
        ke2 = float((masses[:, None] * v**2).sum())
        self.zeta[0] += h * (ke2 - self._serbestlik(v) * kt) / q
        for i in range(1, self.chain_length):
            self.zeta[i] += h * (q * self.zeta[i - 1] ** 2 - kt) / q
            self.zeta[i - 1] *= float(np.exp(-h * self.zeta[i]))
        return v * float(np.exp(-h * self.zeta[0]))

    def _geri_yarim(self, v, masses, h):
        """`_ileri_yarim`in EŞLENİĞİ: adımlar ters sırada uygulanır.

        Bu simetri olmadan `step` zaman-tersinir olmaz.
        """
        kt = KB_HARTREE_PER_KELVIN * self.target_temperature_kelvin
        q = self.thermostat_mass
        v = v * float(np.exp(-h * self.zeta[0]))
        for i in range(self.chain_length - 1, 0, -1):
            self.zeta[i - 1] *= float(np.exp(-h * self.zeta[i]))
            self.zeta[i] += h * (q * self.zeta[i - 1] ** 2 - kt) / q
        ke2 = float((masses[:, None] * v**2).sum())
        self.zeta[0] += h * (ke2 - self._serbestlik(v) * kt) / q
        return v

    def step(self, state, forces, force_field, dt):
        h = 0.5 * dt
        v = self._ileri_yarim(state.velocities, state.masses, h)
        v = v + h * forces / state.masses[:, None]
        x = state.positions + dt * v
        enerji, yeni_kuvvet = force_field(x)
        v = v + h * yeni_kuvvet / state.masses[:, None]
        v = self._geri_yarim(v, state.masses, h)
        return (MDState(positions=x, velocities=v, masses=state.masses),
                enerji, yeni_kuvvet)
