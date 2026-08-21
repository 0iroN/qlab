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


@dataclass(frozen=True)
class TBHopping:
    """Tek bir hoplama yolu. `t_ev` ELEKTRONVOLTTUR.

    Depo içi birim hartree'dir; dönüşüm `engine/tb/factory.py` sınırında
    yapılır (bkz. md/factory.py'nin amu_to_electron_mass çağrısı).
    """

    i: int
    j: int
    cell: tuple[int, int, int]
    t_ev: float


@dataclass(frozen=True)
class TBScenario:
    """Tight-binding senaryosu.

    `onsite_ev` ve `hoppings[*].t_ev` ELEKTRONVOLTTUR; örgü ve taban
    BOHR'dur. TB literatürü eV'de yayınlandığı için giriş birimi eV
    seçilmiştir ve dönüşüm tek bir sınırda yapılır.
    """

    name: str
    lattice_vectors_bohr: tuple[tuple[float, float, float], ...]
    basis_bohr: tuple[tuple[float, float, float], ...]
    onsite_ev: tuple[float, ...]
    hoppings: tuple[TBHopping, ...]
    n_electron_per_cell: int
    k_points: int


@dataclass(frozen=True)
class DownfoldScenario:
    """UHF indirgemesi senaryosu. `ring_size` ÇİFT olmalı.

    `distances_bohr` BOHR'dur ve dönüşüm yoktur: indirgeme motorunun kendi
    birimi de bohr'dur (`downfold.hydrogen_ring` komşu aralığını bohr alır).
    Bu, `TBScenario`'nun eV sınırından bilinçli olarak FARKLIDIR — orada
    dönüşüm vardı çünkü TB literatürü eV'de yayınlanır; burada girdi bir
    geometridir ve geometrinin atomik birimi bohr'dur.

    `ring_size` çift olmak ZORUNDADIR: iki-siteli antiferromanyetik hücre tek
    sayıda sitede kurulamaz (frustrasyon). Kontrol hem yükleyicide hem
    motorda vardır ve bu bir tekrar değil bir katman ayrımıdır — yükleyici
    YAML'ın hatasını, motor çağrının hatasını yakalar.
    """

    name: str
    ring_size: int
    distances_bohr: tuple[float, ...]
    basis: str = "sto-3g"
    n_neighbour: int = 2
    seeds: int = 12
