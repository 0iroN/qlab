"""Senaryo tanımından çalışır motor nesneleri üretir.

engine katmanı scenario katmanını import edebilir (katman sırası:
app > viz > observable > engine > scenario); tersi yasaktır.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.initial import gaussian_packet
from qlab.engine.tdse.propagator import SplitOperatorPropagator
from qlab.errors import ScenarioError
from qlab.provenance import Provenance
from qlab.scenario.schema import TDSEScenario

_NORM_TOLERANCE = 1e-8


def _build_potential(scenario: TDSEScenario, grid: Grid1D) -> np.ndarray:
    kind = scenario.potential.kind
    params = scenario.potential.params
    try:
        if kind == "free":
            return potentials.free(grid)
        if kind == "barrier":
            return potentials.rectangular_barrier(grid, **params)
        if kind == "well":
            return potentials.square_well(grid, **params)
        if kind == "harmonic":
            return potentials.harmonic(grid, mass=scenario.mass, **params)
        if kind == "uniform_field":
            return potentials.uniform_field(grid, **params)
    except TypeError as exc:
        raise ScenarioError(f"{kind!r} potansiyeli için parametreler hatalı: {exc}") from exc
    raise ScenarioError(f"Bilinmeyen potansiyel türü: {kind!r}")


def build(
    scenario: TDSEScenario,
) -> tuple[Grid1D, np.ndarray, np.ndarray, SplitOperatorPropagator]:
    grid = Grid1D(n=scenario.grid.n, length=scenario.grid.length_bohr)
    potential = _build_potential(scenario, grid)
    psi0 = gaussian_packet(
        grid,
        x0=scenario.packet.x0_bohr,
        k0=scenario.packet.k0,
        sigma=scenario.packet.sigma_bohr,
    )

    mask = None
    if scenario.absorbing_width_bohr is not None:
        mask = potentials.absorbing_mask(grid, width=scenario.absorbing_width_bohr)

    propagator = SplitOperatorPropagator(
        grid=grid,
        potential=potential,
        dt=scenario.dt,
        mass=scenario.mass,
        mask=mask,
    )
    return grid, potential, psi0, propagator


def provenance_for(scenario: TDSEScenario, norm_deviation: float) -> Provenance:
    approximations = [
        "tek parçacık",
        "sabit dış potansiyel",
        "periyodik sınır koşulu (FFT)",
        "Strang bölmesi, O(dt^2)",
    ]
    if scenario.absorbing_width_bohr is not None:
        approximations.append("kenarlarda soğurucu maske — norm bilinçli olarak azalır")

    return Provenance(
        method=f"TDSE split-operator (Strang), 1B, dt={scenario.dt}",
        approximations=tuple(approximations),
        neglected=("elektron-elektron etkileşimi", "spin", "göreli etkiler"),
        expected_error=f"O(dt^2) ~ {scenario.dt**2:.1e} hartree mertebesi",
        converged=norm_deviation <= _NORM_TOLERANCE,
        diagnostics={"norm_deviation": norm_deviation, "steps": scenario.n_steps},
    )
