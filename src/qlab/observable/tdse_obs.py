"""TDSE durumundan ölçülebilir büyüklükler.

Bu katman motoru da görselleştirmeyi de bilmez: girdisi bir dizi ve bir ızgara,
çıktısı sayı ya da dizidir. Momentum ve kinetik enerji Fourier uzayında
hesaplanır (Parseval); FFT normalizasyonu pay ve paydadaki aynı çarpanla
sadeleşir.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.tdse.grid import Grid1D
from qlab.errors import InvalidInputError

# Normalize edilmiş bir psi için toplam yoğunluk (ya da spektrum) grid.dx'in
# tersine eşittir; bu projede kullanılan ızgaralarda tipik olarak ~0.3-13
# arasıdır (bkz. tests/test_observable.py ve tests/tdse/test_initial.py).
# Eşik bu meşru aralığın çok altında (en az 9 basamak) ama gerçek sayısal
# çöküşün (tam sıfır ya da taşma) çok üzerinde seçildi: soğurucu maskeyle
# uzun süre inceltilmiş, hâlâ fiziksel olarak anlamlı ama çok küçük bir
# yoğunluğu reddetmeden, gerçekten çökmüş (sıfıra inmiş ya da NaN/inf
# içeren) bir durumu yakalar.
_MIN_AGIRLIK_TOPLAMI = 1e-10


def _agirlik_toplamini_dogrula(agirlik: np.ndarray, nicelik: str) -> float:
    """`agirlik` (yoğunluk ya da spektrum) toplamını doğrular ve döndürür.

    Toplam sonlu değilse (NaN/inf) ya da soğurma/çöküş nedeniyle ihmal
    edilebilir düzeye inmişse `InvalidInputError` fırlatılır. Aksi halde
    beklenti değeri hesaplamaları 0/0'a bölünür ve güvenilmez bir sonuç
    sessizce normal bir sayı gibi döner — bu, errors.py'deki ilkeyi ihlal
    eder.
    """
    toplam = float(np.sum(agirlik))
    if not np.isfinite(toplam) or toplam < _MIN_AGIRLIK_TOPLAMI:
        raise InvalidInputError(
            f"{nicelik} hesaplanamaz: ağırlık toplamı geçersiz (toplam="
            f"{toplam!r}). Yoğunluk soğurma ya da sayısal çöküş nedeniyle "
            "sıfıra inmiş veya NaN/inf içeriyor olabilir."
        )
    return toplam


def norm(psi: np.ndarray, grid: Grid1D) -> float:
    return float(np.sum(np.abs(psi) ** 2) * grid.dx)


def density(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


def expectation_position(psi: np.ndarray, grid: Grid1D) -> float:
    weight = density(psi)
    toplam = _agirlik_toplamini_dogrula(weight, "Konum beklenti değeri")
    return float(np.sum(weight * grid.x) / toplam)


def position_spread(psi: np.ndarray, grid: Grid1D) -> float:
    weight = density(psi)
    toplam = _agirlik_toplamini_dogrula(weight, "Konum yayılımı")
    mean = np.sum(weight * grid.x) / toplam
    variance = np.sum(weight * (grid.x - mean) ** 2) / toplam
    return float(np.sqrt(variance))


def expectation_momentum(psi: np.ndarray, grid: Grid1D) -> float:
    spectrum = np.abs(np.fft.fft(psi)) ** 2
    toplam = _agirlik_toplamini_dogrula(spectrum, "Momentum beklenti değeri")
    return float(np.sum(spectrum * grid.k) / toplam)


def expectation_energy(
    psi: np.ndarray, grid: Grid1D, potential: np.ndarray, mass: float = 1.0
) -> float:
    spectrum = np.abs(np.fft.fft(psi)) ** 2
    spektrum_toplami = _agirlik_toplamini_dogrula(spectrum, "Kinetik enerji")
    kinetic = float(
        np.sum(spectrum * grid.k2 / (2.0 * mass)) / spektrum_toplami
    )
    weight = density(psi)
    agirlik_toplami = _agirlik_toplamini_dogrula(weight, "Potansiyel enerji")
    potential_energy = float(np.sum(weight * potential) / agirlik_toplami)
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
