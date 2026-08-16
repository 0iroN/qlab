"""Senaryo veri yapıları.

Bu modül hiçbir şey import etmez (numpy dahil): saf tanım katmanıdır.
Uzunluklar bohr, enerjiler hartree, zamanlar atomik birimdir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

POTENTIAL_KINDS = ("free", "barrier", "well", "harmonic", "uniform_field")


@dataclass(frozen=True)
class GridSpec:
    n: int
    length_bohr: float


@dataclass(frozen=True)
class PotentialSpec:
    kind: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PacketSpec:
    x0_bohr: float
    k0: float
    sigma_bohr: float


@dataclass(frozen=True)
class TDSEScenario:
    name: str
    grid: GridSpec
    potential: PotentialSpec
    packet: PacketSpec
    dt: float
    n_steps: int
    save_every: int = 1
    mass: float = 1.0
    absorbing_width_bohr: float | None = None


@dataclass(frozen=True)
class HFScanScenario:
    name: str
    molecule_template: str
    basis: str
    parameter: str
    values: tuple[float, ...]
    unit: str = "angstrom"


MD_FORCE_FIELD_KINDS = ("harmonic", "morse", "lennard_jones")
MD_THERMOSTAT_KINDS = ("langevin", "nose_hoover")


@dataclass(frozen=True)
class MDScenario:
    """Klasik moleküler dinamik senaryosu.

    `initial["masses_amu"]` AMU'dur; motor içi elektron kütlesine `factory`
    tarafından çevrilir (bkz. qlab.units.amu_to_electron_mass).
    """

    name: str
    force_field: dict
    initial: dict
    thermostat: dict | None
    dt: float
    n_steps: int
    save_every: int = 1
