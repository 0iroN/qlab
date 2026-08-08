"""Senaryoları GUI olmadan koşan giriş noktası.

    python -m qlab.run scenarios/tunneling.yaml -o out.npz
    python -m qlab.run scenarios/h2_bond_scan.yaml -o h2_curve.npz

Toplu parametre taramaları ve doğrulama koşuları bu yoldan yapılır; app katmanı
hiçbir zaman gerekmez. Bu, spec §3 İlke 1'in pratik karşılığıdır.

Senaryo türüne göre dallanır: TDSEScenario mevcut split-operator yoluna,
HFScanScenario ise engine.hf.factory.run_scan'e gider. İkisi de Provenance
taşıdığı için şeffaflık bloğu tek bir yerden (provenance.as_lines()) basılır.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qlab.engine.hf.factory import ScanResult, run_scan
from qlab.engine.tdse import factory
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.propagator import evolve
from qlab.errors import QLabError
from qlab.provenance import Provenance
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import HFScanScenario, TDSEScenario


@dataclass
class RunResult:
    scenario: TDSEScenario
    grid: Grid1D
    potential: np.ndarray
    frames: np.ndarray
    times: np.ndarray
    provenance: Provenance


def run_scenario(path: str | Path) -> RunResult | ScanResult:
    scenario = load_scenario(path)

    if isinstance(scenario, HFScanScenario):
        return run_scan(scenario)

    grid, potential, psi0, propagator = factory.build(scenario)

    frames, times = evolve(
        propagator, psi0, n_steps=scenario.n_steps, save_every=scenario.save_every
    )

    final_norm = float(np.sum(np.abs(frames[-1]) ** 2) * grid.dx)
    provenance = factory.provenance_for(scenario, norm_deviation=abs(final_norm - 1.0))
    return RunResult(scenario, grid, potential, frames, times, provenance)


def save(result: RunResult | ScanResult, output_path: str | Path) -> None:
    provenance_json = json.dumps(
        dataclasses.asdict(result.provenance), ensure_ascii=False
    )

    if isinstance(result, ScanResult):
        np.savez_compressed(
            output_path,
            values=result.values,
            energies=result.energies,
            # Nokta bazında kararlılık: tek bir özet bayrağı, eğrinin HANGİ
            # noktasının eyer noktası olduğunu söylemez. Ölçülemeyen
            # noktalarda NaN (bkz. factory.ScanResult).
            lowest_eigenvalues=result.lowest_eigenvalues,
            provenance_json=provenance_json,
        )
        return

    np.savez_compressed(
        output_path,
        frames=result.frames,
        times=result.times,
        x=result.grid.x,
        potential=result.potential,
        provenance_json=provenance_json,
    )


def reconfigure_stdio_utf8() -> None:
    """Windows ve diğer CP1252 varsayılan konsollarında Türkçe karakterleri
    yazdırabilmek için stdout/stderr'i UTF-8'e yeniden yapılandırır.
    reconfigure() var olmayan ortamlarda (ender), stdout/stderr'i wrapping'e
    geri döneriz. Her iki durumda da, başarısız olursa orijinal akış
    kullanılır ve hiçbir ek hata yükseltilmez.

    Paylaşılan tek kaynak: hem bu modülün `main()`'i hem de `qlab.app.main`
    aynı GUI-öncesi/iş-öncesi noktada bunu çağırır — desen iki kez elle
    kopyalanmaz.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        try:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace"
                )
        except (AttributeError, ValueError, OSError):
            pass

    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        try:
            if hasattr(sys.stderr, "buffer"):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer, encoding="utf-8", errors="replace"
                )
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    reconfigure_stdio_utf8()

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
    if isinstance(result, ScanResult):
        print(f"{result.scenario.name}: {len(result.values)} nokta -> {args.output}")
    else:
        print(f"{result.scenario.name}: {len(result.times)} kare -> {args.output}")
    for label, value in result.provenance.as_lines():
        print(f"  {label:<14} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
