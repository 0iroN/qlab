"""Dikdörtgen bariyer için kapalı form geçirgenlik ve paket ortalaması.

Kapalı form TEK ENERJİLİ dalga içindir. Gauss paketinin bir enerji dağılımı
vardır; doğru karşılaştırma paketin momentum dağılımı üzerinden ağırlıklı
integraldir (spec §5.4). Doğrudan T(E0) kullanmak dar olmayan paketlerde
yüzde onlarca sapar.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.tdse.grid import Grid1D


def transmission_coefficient(
    energy: float, barrier_height: float, width: float, mass: float = 1.0
) -> float:
    if energy <= 0:
        return 0.0

    if energy < barrier_height:
        kappa = np.sqrt(2.0 * mass * (barrier_height - energy))
        numerator = barrier_height**2 * np.sinh(kappa * width) ** 2
        denominator = 4.0 * energy * (barrier_height - energy)
        return float(1.0 / (1.0 + numerator / denominator))

    if energy > barrier_height:
        wave_number = np.sqrt(2.0 * mass * (energy - barrier_height))
        numerator = barrier_height**2 * np.sin(wave_number * width) ** 2
        denominator = 4.0 * energy * (energy - barrier_height)
        return float(1.0 / (1.0 + numerator / denominator))

    # E == V0 limiti
    return float(1.0 / (1.0 + mass * barrier_height * width**2 / 2.0))


def packet_transmission(
    psi0: np.ndarray,
    grid: Grid1D,
    barrier_height: float,
    width: float,
    mass: float = 1.0,
) -> float:
    """|phi(k)|^2 ağırlıklı T(E(k)) ortalaması; yalnızca ileri giden bileşenler."""
    spectrum = np.abs(np.fft.fft(psi0)) ** 2
    forward = grid.k > 0

    backward_weight = np.sum(spectrum[~forward]) / np.sum(spectrum)
    if backward_weight > 1e-6:
        raise AssertionError(
            f"Paketin {backward_weight:.2e} kadarı geriye gidiyor; bariyere hiç "
            "ulaşmayan bileşenler ortalamayı bozar. k0'ı artır veya sigma'yı büyüt."
        )

    energies = grid.k[forward] ** 2 / (2.0 * mass)
    weights = spectrum[forward]
    coefficients = np.array(
        [transmission_coefficient(e, barrier_height, width, mass) for e in energies]
    )
    return float(np.sum(weights * coefficients) / np.sum(weights))
