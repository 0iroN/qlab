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
@dataclass(frozen=True)
class LennardJones:
    """Periyodik kutuda 12-6 Lennard-Jones, kaydırılmış kesmeyle.

    V(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] - V(r_c)   (r < r_c)

    KAYDIRMA ZORUNLUDUR: kaydırılmamış kesme r_c'de enerjiyi sıçratır ve
    enerji korunumunu bozar (ölçüldü, tasarım §4.4).

    KESME < KUTU/2 ZORUNLUDUR: aksi halde minimum görüntü konvansiyonu bozulur
    ve aynı çift birden çok kez sayılır.

    ÖLÇÜLMÜŞ UYARI: bu alanın Maxwell-Boltzmann üretmesi onu DOĞRULAMAZ --
    eps x10, eps x0.1, sigma x0.9 ve WCA da kusursuz MB veriyor. Doğrulama
    analitik çıpalarla yapılır.
    """

    epsilon: float
    sigma: float
    box_length: float
    cutoff: float

    def __post_init__(self) -> None:
        for ad, deger in (
            ("epsilon", self.epsilon),
            ("sigma", self.sigma),
            ("kutu kenarı", self.box_length),
            ("kesme yarıçapı", self.cutoff),
        ):
            if not np.isfinite(deger) or deger <= 0.0:
                raise InvalidInputError(f"{ad} pozitif ve sonlu olmalı: {deger}")
        if self.cutoff > 0.5 * self.box_length:
            raise InvalidInputError(
                f"kesme yarıçapı kutu kenarının yarısını aşamaz "
                f"(minimum görüntü bozulur): {self.cutoff} > {0.5 * self.box_length}"
            )

    def _ham(self, r2: np.ndarray) -> np.ndarray:
        oran6 = (self.sigma**2 / r2) ** 3
        return 4.0 * self.epsilon * (oran6**2 - oran6)

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        # ŞEKİL MUHAFAZASI: kardeş sınıf `Morse` bunu yapıyor, LJ yapmıyordu.
        # ÖLÇÜLDÜ -- (2,2) şekli SESSİZCE kabul ediliyor ve sonuç (2,2) çıkıyor,
        # yani 2 boyutlu koordinatla fizik yapılıyordu; 1B (3,) ham `IndexError`,
        # int dtype ham `UFuncTypeError` veriyordu. İkisi de QLabError DEĞİL.
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise InvalidInputError(
                f"konumlar (N,3) şeklinde olmalı, verilen: {np.shape(positions)}"
            )
        positions = pos
        delta = positions[:, None, :] - positions[None, :, :]
        delta -= self.box_length * np.round(delta / self.box_length)
        r2 = (delta**2).sum(axis=-1)
        np.fill_diagonal(r2, np.inf)

        icinde = r2 < self.cutoff**2
        kaydirma = float(self._ham(np.array([self.cutoff**2]))[0])

        enerji = float(np.where(icinde, self._ham(r2) - kaydirma, 0.0).sum()) / 2.0

        oran6 = np.where(icinde, (self.sigma**2 / r2) ** 3, 0.0)
        katsayi = np.where(
            icinde, 24.0 * self.epsilon * (2 * oran6**2 - oran6) / r2, 0.0
        )
        kuvvet = (katsayi[:, :, None] * delta).sum(axis=1)
        return enerji, kuvvet
