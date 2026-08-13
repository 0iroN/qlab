"""Analitik kuvvet alanları -- `state.ForceField` sözleşmesinin uygulamaları.

Her alanın ANALİTİK ÇIPALARI vardır ve testler onları kullanır; Maxwell-
Boltzmann bir kuvvet alanı sınavı DEĞİLDİR (bkz. tests/md/test_forcefields.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError


@dataclass(frozen=True, eq=False)
class Harmonic:
    """Her parçacığı kendi merkezine bağlayan izotropik yay.

    V = sum_i 0.5 k |r_i - c_i|^2 ,  F_i = -k (r_i - c_i)

    İntegratörün doğrulanması için var: periyot T = 2*pi*sqrt(m/k) analitiktir
    ve enerji salınımının kapalı formu bilinir (bkz. test_integrator.py).

    `center` (M,3) olmalıdır. M = N ise parçacık başına ayrı merkez, M = 1 ise
    hepsi tek merkeze bağlanır (yayılım bilerek korunuyor).
    """

    k: float
    center: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.k) or self.k <= 0.0:
            raise InvalidInputError(f"yay sabiti pozitif ve sonlu olmalı: {self.k}")
        merkez = np.asarray(self.center, dtype=float)
        if merkez.ndim != 2 or merkez.shape[1] != 3:
            raise InvalidInputError(
                f"merkez (M,3) şeklinde olmalı, verilen: {merkez.shape}"
            )
        if not np.all(np.isfinite(merkez)):
            raise InvalidInputError(
                "merkez sonlu olmalı (nan/inf sessizce enerjiye ve kuvvete yayılır)"
            )
        object.__setattr__(self, "center", merkez)

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        yer_degistirme = positions - self.center
        enerji = 0.5 * self.k * float((yer_degistirme**2).sum())
        return enerji, -self.k * yer_degistirme
@dataclass(frozen=True)
class Morse:
    """İki atomlu Morse potansiyeli -- gerçek bir ayrışma enerjisi taşır.

    V(r) = D_e (1 - exp(-a (r - r_e)))^2 ,  r = |r_2 - r_1|

    Harmonikten farkı: bağ GERÇEKTEN kopar. "Isı bağı kırar" hikâyesinin
    klasik ve ucuz sürümü budur; pahalı (Hartree-Fock) sürümü Plan D'de gelir.

    ÖLÇÜLMÜŞ SINIRLAR (tasarım §4.3):
    - Duvar SONLUDUR: V(0) = 10.42 * D_e. E > 10.42*D_e olan bir yörünge
      duvarı GEÇER. r = |r2 - r1| bir norm olduğu için NEGATİF OLMAZ; atomlar
      birbirinin İÇİNDEN geçer ve çift SAHTE olarak ayrışır. ÖLÇÜLDÜ
      (E/De = 12): min r = 8.4e-04 bohr, r_max 45.42 bohr, negatif r sayısı 0.
      Ayrışma ölçütü bu yüzden "r < 0" ile DEĞİL duvar geçişiyle korunur.
    - exp(-a(r-r_e)) büyük r'de ALT-TAŞAR (r ~ 725.5 bohr'da tam 0.0) ve
      numpy varsayılanı bunu sessizce yapar.
    - H2 TAM bir Morse osilatörü değildir: parametre seti aşırı-belirli ve
      %4.256 tutarsız (Morse özdeşliği D_e = omega_e^2/(4 omega_e x_e) ile
      39913.1 cm^-1, deneysel D_e 38283.9 cm^-1).
    """

    d_e: float
    r_e: float
    a: float

    def __post_init__(self) -> None:
        for ad, deger in (("D_e", self.d_e), ("r_e", self.r_e), ("a", self.a)):
            if not np.isfinite(deger) or deger <= 0.0:
                raise InvalidInputError(f"{ad} pozitif ve sonlu olmalı: {deger}")

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        if np.shape(positions) != (2, 3):
            raise InvalidInputError(
                f"Morse iki atom bekler, verilen şekil: {np.shape(positions)}"
            )
        delta = positions[1] - positions[0]
        r = float(np.linalg.norm(delta))
        if r == 0.0:
            raise InvalidInputError("çakışık atomlar: Morse r = 0'da tanımsız")
        ussel = np.exp(-self.a * (r - self.r_e))
        enerji = self.d_e * (1.0 - ussel) ** 2
        # dV/dr = 2 D_e a exp(-a(r-r_e)) (1 - exp(-a(r-r_e)))
        dv_dr = 2.0 * self.d_e * self.a * ussel * (1.0 - ussel)
        birim = delta / r
        return enerji, np.vstack([dv_dr * birim, -dv_dr * birim])
