"""Senaryoları GUI olmadan koşan giriş noktası.

    python -m qlab.run scenarios/tunneling.yaml -o out.npz

Toplu parametre taramaları ve doğrulama koşuları bu yoldan yapılır; app katmanı
hiçbir zaman gerekmez. Bu, spec §3 İlke 1'in pratik karşılığıdır.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qlab.engine.tdse import factory
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.propagator import evolve
from qlab.errors import QLabError
from qlab.provenance import Provenance
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import TDSEScenario


@dataclass
class RunResult:
    scenario: TDSEScenario
    grid: Grid1D
    potential: np.ndarray
    frames: np.ndarray
    times: np.ndarray
    provenance: Provenance


def run_scenario(path: str | Path) -> RunResult:
    scenario = load_scenario(path)
    grid, potential, psi0, propagator = factory.build(scenario)

    frames, times = evolve(
        propagator, psi0, n_steps=scenario.n_steps, save_every=scenario.save_every
    )

    final_norm = float(np.sum(np.abs(frames[-1]) ** 2) * grid.dx)
    provenance = factory.provenance_for(scenario, norm_deviation=abs(final_norm - 1.0))
    return RunResult(scenario, grid, potential, frames, times, provenance)


def save(result: RunResult, output_path: str | Path) -> None:
    np.savez_compressed(
        output_path,
        frames=result.frames,
        times=result.times,
        x=result.grid.x,
        potential=result.potential,
        provenance_json=json.dumps(
            dataclasses.asdict(result.provenance), ensure_ascii=False
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qlab.run", description="TDSE senaryosunu GUI olmadan koştur"
    )
    parser.add_argument("scenario", help="YAML senaryo dosyası")
    parser.add_argument("-o", "--output", required=True, help="çıktı .npz dosyası")
    args = parser.parse_args(argv)

    try:
        result = run_scenario(args.scenario)
    except QLabError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    save(result, args.output)
    print(f"{result.scenario.name}: {len(result.times)} kare -> {args.output}")
    for label, value in result.provenance.as_lines():
        print(f"  {label:<14} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
