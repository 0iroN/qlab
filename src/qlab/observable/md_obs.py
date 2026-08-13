"""MD gözlemleri: sıcaklık ve hız dağılımı göstergeleri.

Bu katman motoru bilmez; yalnızca durum ve yörünge okur.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.md.state import MDState
from qlab.errors import InvalidInputError
from qlab.units import KB_HARTREE_PER_KELVIN

#: Maxwell-Boltzmann'da <v^4>/<v^2>^2 tam olarak 5/3'tür (ölçekten bağımsız).
SPEED_SHAPE_RATIO_MB: float = 5.0 / 3.0

#: std(R4) = _R4_NOISE / sqrt(n) = sqrt(40/27) / sqrt(n). Analitik olarak
#: türetildi ve altı örneklem boyutunda doğrulandı (ölçülen/analitik
#: oranı 0.958-1.002).
_R4_NOISE: float = float(np.sqrt(40.0 / 27.0))


def instantaneous_temperature(state: MDState, n_constraints: int = 3) -> float:
    """Kinetik sıcaklık (kelvin): T = 2 KE / (ndof k_B).

    `n_constraints` varsayılan 3'tür, çünkü kütle merkezi hareketi
    `MDState.remove_center_of_mass_motion()` ile çıkarılır. 3N ile 3N-3
    arasındaki fark -1/N mertebesindedir ve İSTATİSTİKLE yakalanamaz
    (ölçüldü); bu yüzden burada açık bir parametredir ve testle kilitlenir.
    """
    ndof = 3 * state.positions.shape[0] - n_constraints
    if ndof < 1:
        raise InvalidInputError(
            f"serbestlik derecesi kalmadı: 3N = {3 * state.positions.shape[0]}, "
            f"kısıt = {n_constraints}"
        )
    return float(2.0 * state.kinetic_energy() / (ndof * KB_HARTREE_PER_KELVIN))


def speed_shape_ratio(velocities: np.ndarray) -> float:
    """R4 = <v^4> / <v^2>^2. Maxwell-Boltzmann'da 5/3.

    ÖLÇEKTEN BAĞIMSIZDIR, yani sıcaklığı bilmeden dağılımın ŞEKLİNİ sınar.
    Alternatiflerden keskindir: parçacık başına gürültü katsayısı 1.217,
    tek bileşen basıklığı için 2.83 -- aynı tolerans için ~5 kat daha az
    örnek yeter.
    """
    v2 = (np.asarray(velocities, dtype=float) ** 2).sum(axis=-1)
    return float((v2**2).mean() / (v2.mean() ** 2))


def speed_shape_ratio_noise(n: int) -> float:
    """`speed_shape_ratio`'nun n örnekteki standart sapması."""
    if n < 1:
        raise InvalidInputError(f"örnek sayısı en az 1 olmalı: {n}")
    return _R4_NOISE / float(np.sqrt(n))
