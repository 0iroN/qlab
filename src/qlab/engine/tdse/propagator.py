"""Strang bölmeli split-operator zaman evrimi.

    psi(t+dt) = e^{-i V dt/2} F^-1 [ e^{-i k^2 dt/(2m)} F [ e^{-i V dt/2} psi(t) ] ]

Yöntem ünitertir: gerçek zamanda norm makine hassasiyetinde korunur. Bu bir test
hedefi değil, yöntemin garantisidir — sapma varsa dt çok büyüktür ya da kod
hatalıdır. evolve() bu yüzden sapmayı istisna olarak raporlar.

İmajiner zaman: dt -> -i*dt yerine koyunca e^{-H tau} elde edilir; her adımdan
sonra yeniden normalize edilerek durum taban duruma yakınsar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qlab.engine.tdse.grid import Grid1D, Grid2D
from qlab.errors import InvalidInputError, UnitarityViolationError


@dataclass
class SplitOperatorPropagator:
    grid: Grid1D | Grid2D
    potential: np.ndarray
    dt: float
    mass: float = 1.0
    imaginary_time: bool = False
    mask: np.ndarray | None = None

    _half_v: np.ndarray = field(init=False, repr=False)
    _kinetic: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise InvalidInputError(f"dt pozitif olmalı, verilen: {self.dt}")
        if self.mass <= 0:
            raise InvalidInputError(f"mass pozitif olmalı, verilen: {self.mass}")

        expected = self._expected_shape()
        if self.potential.shape != expected:
            raise InvalidInputError(
                f"Potansiyel şekli ızgarayla uyuşmuyor: beklenen {expected}, "
                f"verilen {self.potential.shape}"
            )
        if self.mask is not None and self.mask.shape != expected:
            raise InvalidInputError(
                f"Maske şekli ızgarayla uyuşmuyor: beklenen {expected}, "
                f"verilen {self.mask.shape}"
            )

        step = -1j * self.dt if self.imaginary_time else self.dt
        self._half_v = np.exp(-0.5j * self.potential * step)
        self._kinetic = np.exp(-0.5j * self.grid.k2 * step / self.mass)

    @property
    def is_two_dimensional(self) -> bool:
        return isinstance(self.grid, Grid2D)

    @property
    def volume_element(self) -> float:
        return self.grid.cell_area if self.is_two_dimensional else self.grid.dx

    def _expected_shape(self) -> tuple[int, ...]:
        if isinstance(self.grid, Grid2D):
            return (self.grid.nx, self.grid.ny)
        return (self.grid.n,)

    def _forward(self, psi: np.ndarray) -> np.ndarray:
        return np.fft.fft2(psi) if self.is_two_dimensional else np.fft.fft(psi)

    def _inverse(self, psi: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(psi) if self.is_two_dimensional else np.fft.ifft(psi)

    def step(self, psi: np.ndarray) -> np.ndarray:
        """Tek bir Strang adımı uygular ve yeni psi'yi döndürür.

        Şekil doğrulaması yanlış boyutlu ya da yanlış eksen sayılı girdiyi
        yakalar. Ama bilinen bir sınırı var: kare bir ızgarada (nx == ny)
        transpoze edilmiş psi.T, orijinal psi ile birebir aynı .shape
        tuple'ına sahiptir — NumPy dizisi hangi eksenin x, hangisinin y
        olduğuna dair bir bilgi taşımaz, bu yüzden bu doğrulama transpozisyonu
        hiçbir zaman yakalayamaz. Bu çözülmüş bir sorun değil, çalışma
        zamanında tespit edilemeyen bilinen bir kısıt; kare ızgaralarda
        eksenlerin doğru sırayla verilmesi çağıranın sorumluluğunda kalır.
        """
        expected = self._expected_shape()
        if psi.shape != expected:
            raise InvalidInputError(
                f"psi şekli ızgarayla uyuşmuyor: beklenen {expected}, "
                f"verilen {psi.shape}"
            )

        psi = self._half_v * psi
        psi = self._inverse(self._kinetic * self._forward(psi))
        psi = self._half_v * psi

        if self.mask is not None:
            psi = psi * self.mask

        if self.imaginary_time:
            psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * self.volume_element)

        return psi


def evolve(
    propagator: SplitOperatorPropagator,
    psi0: np.ndarray,
    n_steps: int,
    save_every: int = 1,
    norm_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Zaman evrimi; (kareler, zamanlar) döndürür. t=0 karesi her zaman dahil.

    Norm kontrolü yalnızca gerçek zamanda ve maske yokken yapılır; imajiner
    zaman ve soğurucu maske normu bilinçli olarak değiştirir.
    """
    if n_steps < 1:
        raise InvalidInputError(f"n_steps en az 1 olmalı, verilen: {n_steps}")
    if save_every < 1:
        raise InvalidInputError(f"save_every en az 1 olmalı, verilen: {save_every}")

    check_norm = not propagator.imaginary_time and propagator.mask is None
    volume = propagator.volume_element

    psi = psi0.copy()
    frames = [psi.copy()]
    times = [0.0]

    for index in range(1, n_steps + 1):
        psi = propagator.step(psi)

        if check_norm:
            deviation = abs(float(np.sum(np.abs(psi) ** 2) * volume) - 1.0)
            if deviation > norm_tol:
                raise UnitarityViolationError(
                    step=index, norm_deviation=deviation, dt=propagator.dt
                )

        if index % save_every == 0:
            frames.append(psi.copy())
            times.append(index * propagator.dt)

    return np.array(frames), np.array(times)
