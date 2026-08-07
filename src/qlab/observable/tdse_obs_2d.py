"""2B TDSE durumundan ölçülebilir büyüklükler.

1B karşılığı (`qlab.observable.tdse_obs`) gibi bu katman da motoru ve
görselleştirmeyi bilmez: girdisi bir dizi ve bir ızgara, çıktısı sayı ya da
dizidir. `screen_profile`, çift yarık deneyinde fiziksel "ekranı" modeller:
sabit bir x kesitindeki yoğunluk profilini, y eksenine karşı döndürür.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.tdse.grid import Grid2D


def norm(psi: np.ndarray, grid: Grid2D) -> float:
    return float(np.sum(np.abs(psi) ** 2) * grid.cell_area)


def density(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


def screen_profile(
    psi: np.ndarray, grid: Grid2D, screen_x: float
) -> tuple[np.ndarray, np.ndarray]:
    """x = screen_x düzlemindeki yoğunluk profili — 'ekrandaki desen'."""
    column = int(np.argmin(np.abs(grid.X[:, 0] - screen_x)))
    return grid.Y[column, :], density(psi)[column, :]
