"""Plan D'nin mimari iddiasi: cekirdek DEGISMEZ.

Iddia OLCULDU (Plan D §0.1): degismemis `integrator.run`, `Langevin.step` ve
`NoseHoover.step` HF kuvvet alaniyla calisti. Bu test o iddiayi KILITLER.

DIKKAT -- iddia DAR: "hicbir yere dokunmaz" YANLIS. `factory.py` degismek
ZORUNDA (dagitim dali + HF yaklasim listesi). Bu test yalniz UC modulu korur.

GIT ILE OLCULDU (Plan C kapanisi 82c65c4 -> Task 5 sonu 5042c10):
  git diff --stat -- integrator.py thermostats.py md_obs.py  ->  BOS (0 satir)
Ayni aralikta degisenler tam beyan edilenler: forcefields.py +113 (dorduncu
alan), factory.py +38 (iki zorunlu satir), h2_aimd.yaml +20, uc test dosyasi.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.md.forcefields import HartreeFockGradient
from qlab.engine.md.integrator import run, velocity_verlet_step
from qlab.engine.md.state import MDState
from qlab.engine.md.thermostats import Langevin, NoseHoover
from qlab.observable.md_obs import instantaneous_temperature
from qlab.units import amu_to_electron_mass

M_H = amu_to_electron_mass(1.00782503207)


def _durum() -> MDState:
    return MDState(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3959193602]]),
        velocities=np.zeros((2, 3)),
        masses=np.full(2, M_H),
    )


def test_HF_alani_DEGISMEMIS_cekirdegin_uc_giris_noktasindan_geciyor():
    """`velocity_verlet_step`, `Langevin.step`, `NoseHoover.step` -- ucu de
    ayni imzayi paylasir ve HF alanini ayirt etmez.

    OLCULDU: ucu de kostu; adim basina 13 SCF (6N+1).
    """
    alan = HartreeFockGradient(["H", "H"])
    for adimlayici in (
        lambda d, k: velocity_verlet_step(d, k, alan, 5.0),
        lambda d, k: Langevin(
            target_temperature_kelvin=300.0, friction=0.005, seed=1
        ).step(d, k, alan, 5.0),
        lambda d, k: NoseHoover(
            target_temperature_kelvin=300.0, relaxation_time=50.0, n_dof=1
        ).step(d, k, alan, 5.0),
    ):
        durum = _durum()
        _, kuvvetler = alan(durum.positions)
        durum, enerji, kuvvetler = adimlayici(durum, kuvvetler)
        assert np.all(np.isfinite(durum.positions))
        assert np.isfinite(enerji)
        assert instantaneous_temperature(durum, n_constraints=0) >= 0.0


def test_HF_alani_run_ile_Trajectory_uretiyor():
    """`run()` HF alanini ayirt etmiyor; Trajectory dolu geliyor."""
    yorunge = run(_durum(), HartreeFockGradient(["H", "H"]), dt=5.0, n_steps=4)
    assert yorunge.positions.shape == (5, 2, 3)
    assert np.all(np.isfinite(yorunge.kinetic))
    assert np.all(yorunge.potential < 0.0)   # HF enerjisi negatif
