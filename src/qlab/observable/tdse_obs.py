"""TDSE durumundan ölçülebilir büyüklükler.

Bu katman motoru da görselleştirmeyi de bilmez: girdisi bir dizi ve bir ızgara,
çıktısı sayı ya da dizidir. Momentum ve kinetik enerji Fourier uzayında
hesaplanır (Parseval); FFT normalizasyonu pay ve paydadaki aynı çarpanla
sadeleşir.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.tdse.grid import Grid1D


def norm(psi: np.ndarray, grid: Grid1D) -> float:
    return float(np.sum(np.abs(psi) ** 2) * grid.dx)


def density(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


def expectation_position(psi: np.ndarray, grid: Grid1D) -> float:
    weight = density(psi)
    return float(np.sum(weight * grid.x) / np.sum(weight))


def position_spread(psi: np.ndarray, grid: Grid1D) -> float:
    weight = density(psi)
    total = np.sum(weight)
    mean = np.sum(weight * grid.x) / total
    variance = np.sum(weight * (grid.x - mean) ** 2) / total
    return float(np.sqrt(variance))


def expectation_momentum(psi: np.ndarray, grid: Grid1D) -> float:
    spectrum = np.abs(np.fft.fft(psi)) ** 2
    return float(np.sum(spectrum * grid.k) / np.sum(spectrum))


def expectation_energy(
    psi: np.ndarray, grid: Grid1D, potential: np.ndarray, mass: float = 1.0
) -> float:
    spectrum = np.abs(np.fft.fft(psi)) ** 2
    kinetic = float(np.sum(spectrum * grid.k2 / (2.0 * mass)) / np.sum(spectrum))
    weight = density(psi)
    potential_energy = float(np.sum(weight * potential) / np.sum(weight))
    return kinetic + potential_energy


def probability_beyond(psi: np.ndarray, grid: Grid1D, boundary: float) -> float:
    """Sınırın sağındaki MUTLAK olasılık.

    Bilinçli olarak renormalize edilmez: soğurucu maske kullanıldığında toplam
    norm 1'in altına iner ve oran vermek yanıltıcı olur.
    """
    return float(np.sum(density(psi)[grid.x > boundary]) * grid.dx)


def edge_probability(psi: np.ndarray, grid: Grid1D, width: float) -> float:
    """Izgaranın iki ucundaki toplam olasılık — periyodik sarma kontrolü.

    Bu değer büyürse paket kenardan dolanmaya başlamıştır ve iletim ölçümü
    artık geçersizdir.
    """
    x = grid.x
    at_edge = (x < x[0] + width) | (x > x[-1] - width)
    return float(np.sum(density(psi)[at_edge]) * grid.dx)
