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
from qlab.scenario.schema import HFScanScenario, MDScenario, TDSEScenario


@dataclass
class RunResult:
    scenario: TDSEScenario
    grid: Grid1D
    potential: np.ndarray
    frames: np.ndarray
    times: np.ndarray
    provenance: Provenance


def run_scenario(path: str | Path) -> RunResult | ScanResult:
    return _run_loaded(load_scenario(path))


def _run_loaded(scenario) -> RunResult | ScanResult:
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
        scenario = load_scenario(args.scenario)
        # MD DALI: `_run_md` kendi panelini basar ve kendi .npz'sini yazar,
        # bu yuzden `save()` yolundan gecmez. Bu dagitim eskiden YOKTU ve
        # `python -m qlab.run scenarios/h2_morse.yaml` yakalanmayan bir
        # `AttributeError: 'MDScenario' object has no attribute 'grid'`
        # veriyordu (main yalniz QLabError yakaliyor).
        if isinstance(scenario, MDScenario):
            return _run_md(scenario, args.output)
        result = _run_loaded(scenario)
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


def _run_md(scenario: MDScenario, output: str | None) -> int:
    """MD senaryosunu koşturur ve şeffaflık panelini basar."""
    from qlab.engine.md.factory import build, provenance_for
    from qlab.engine.md.integrator import Trajectory, run

    durum, alan, termostat = build(scenario)
    durum = durum.remove_center_of_mass_motion()

    if termostat is None:
        yorunge = run(
            durum, alan, dt=scenario.dt, n_steps=scenario.n_steps,
            save_every=scenario.save_every,
        )
    else:
        # Termostatlı koşu: `step` imzası velocity_verlet_step ile aynıdır.
        _, kuvvetler = alan(durum.positions)
        zamanlar, konumlar, hizlar, kinetik, potansiyel = [0.0], [durum.positions], \
            [durum.velocities], [durum.kinetic_energy()], [alan(durum.positions)[0]]
        for adim in range(1, scenario.n_steps + 1):
            durum, enerji, kuvvetler = termostat.step(durum, kuvvetler, alan, scenario.dt)
            if adim % scenario.save_every == 0:
                zamanlar.append(adim * scenario.dt)
                konumlar.append(durum.positions)
                hizlar.append(durum.velocities)
                kinetik.append(durum.kinetic_energy())
                potansiyel.append(enerji)
        yorunge = Trajectory(
            times=np.array(zamanlar), positions=np.array(konumlar),
            velocities=np.array(hizlar), kinetic=np.array(kinetik),
            potential=np.array(potansiyel),
        )

    toplam = yorunge.kinetic + yorunge.potential
    # SÜRÜKLENME, salınım DEĞİL: blok ortalamalarının baş-son farkı.
    # BLOK SAYISI VERİYE GÖRE: ÖLÇÜLDÜ -- 10 sabit blokla M < 10 kayıtta
    # `reshape` boş dilim üretiyor, sürüklenme `nan` çıkıyor ve panel onun
    # YANINDA ✓ basıyordu (M=3 ve M=9'da görüldü; ilk sağlam değer M=10).
    if len(toplam) < 2:
        raise InvalidInputError(
            f"sürüklenme ölçmek için en az 2 kayıt gerekir, {len(toplam)} var; "
            "n_steps/save_every oranını büyütün"
        )
    n_blok = min(10, len(toplam))
    blok = toplam[: (len(toplam) // n_blok) * n_blok].reshape(n_blok, -1).mean(axis=1)
    suruklenme = float(abs(blok[-1] - blok[0]) / abs(blok[0]))

    # KISIT SAYISI TERMOSTATA BAĞLIDIR: ÖLÇÜLDÜ -- Langevin kütle merkezini
    # YENİDEN ISITIYOR (<KE_COM>/(1.5kT) = 1.098, çıkarılmış kalsaydı 0.000),
    # yani NVT koşusunda 3N-3 kullanmak <T>'yi %0.935 yukarı kaydırıyor
    # (89.393 K yerine 88.565 K). Task 8'in varsayılanı COM'un çıkarılmış
    # KALDIĞI ön koşuluna dayanır ve o ön koşul burada geçerli değil.
    kisit = 3 if termostat is None else 0

    provenance = provenance_for(scenario, energy_drift=suruklenme, n_constraints=kisit)
    for etiket, deger in provenance.as_lines():
        print(f"{etiket:<14}: {deger}")

    if output is not None:
        # Task 13'ün modülü BURADA import edilir: Task 12 tek başına
        # commit edildiğinde CLI'ın kırık olmaması için (ölçüldü -- koşulsuz
        # import `ModuleNotFoundError` veriyordu).
        from qlab.viz.md_frames import build_md_payload

        yuk = build_md_payload(yorunge, durum.masses, n_constraints=kisit)
        np.savez(
            output,
            times_fs=yuk.times_fs, positions=yorunge.positions,
            velocities=yorunge.velocities, kinetic=yuk.kinetic,
            potential=yuk.potential, total=yuk.total,
            temperature_kelvin=yuk.temperature_kelvin,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
