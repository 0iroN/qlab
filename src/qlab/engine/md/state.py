"""MD durumu ve kuvvet sağlayıcı sözleşmesi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qlab.errors import InvalidInputError


class ForceField(Protocol):
    """Kuvvet sağlayıcı sözleşmesi -- bu modülün tek genişleme noktası.

    `(konumlar: (N,3) bohr) -> (potansiyel enerji: hartree,
    kuvvetler: (N,3) hartree/bohr)`

    İntegratör bundan başka bir şey bilmez. Plan D'nin Hartree-Fock kuvvet
    sağlayıcısı da tam olarak bunu uygular; integratör, termostat ve
    gözlemler değişmez.
    """

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]: ...


@dataclass(frozen=True, eq=False)
class MDState:
    """Bir anlık MD durumu.

    `positions` ve `velocities` (N,3), `masses` (N,) ve ELEKTRON KÜTLESİ
    cinsindendir (bkz. modül dokstring'i).

    `eq=False`: alanlar ndarray olduğu için üretilen `__eq__` bir dizi
    döndürür ve `__hash__` `TypeError` verir. Depo bu tuzağı
    `engine/hf/integrals.py`'de zaten çözmüş; aynı gerekçe.
    """

    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray

    def __post_init__(self) -> None:
        # ÖNCE ndarray'e çevir: liste girdisi doğrulamayı geçip sonra
        # `kinetic_energy()` içinde ham `TypeError` veriyordu (ölçüldü).
        for ad in ("positions", "velocities", "masses"):
            object.__setattr__(
                self, ad, np.asarray(getattr(self, ad), dtype=float)
            )

        sekil = self.positions.shape
        if len(sekil) != 2 or sekil[1] != 3 or sekil[0] < 1:
            raise InvalidInputError(
                f"konumlar (N,3) şeklinde olmalı, verilen: {sekil}"
            )
        n = sekil[0]
        if self.velocities.shape != (n, 3):
            raise InvalidInputError(
                f"hızların şekli konumlarla aynı olmalı: "
                f"{self.velocities.shape} != {(n, 3)}"
            )
        if self.masses.shape != (n,):
            raise InvalidInputError(
                f"kütlelerin şekli (N,) olmalı: {self.masses.shape} != {(n,)}"
            )
        for ad, dizi in (
            ("konumlar", self.positions),
            ("hızlar", self.velocities),
            ("kütleler", self.masses),
        ):
            if not np.all(np.isfinite(dizi)):
                raise InvalidInputError(f"{ad} sonlu olmalı (nan/inf kabul edilmez)")
        if not np.all(self.masses > 0.0):
            raise InvalidInputError(f"kütle pozitif olmalı, verilen: {self.masses}")

    def kinetic_energy(self) -> float:
        """Toplam kinetik enerji (hartree)."""
        return float(0.5 * (self.masses[:, None] * self.velocities**2).sum())

    def remove_center_of_mass_motion(self) -> MDState:
        """Kütle merkezi hızını çıkarır.

        Sürüklenen bir kütle merkezi sıcaklığa SAHTE katkı yapar; ndof = 3N-3
        kullanılacaksa (bkz. `observable/md_obs.py`) bu üç serbestlik derecesi
        gerçekten çıkarılmış olmalıdır. Hız KÜTLE-AĞIRLIKLI ortalamadır; düz
        ortalama kullanan bir uygulama artık momentum bırakır (ölçüldü).
        """
        toplam = float(self.masses.sum())
        v_com = (self.masses[:, None] * self.velocities).sum(axis=0) / toplam
        return MDState(
            positions=self.positions,
            velocities=self.velocities - v_com,
            masses=self.masses,
        )


def _bool_iceriyor(deger: object) -> bool:
    """İç içe liste/demet/dizide bool ya da karmaşık öğe (Python ya da numpy) var mı.

    `md/factory.py`deydi (NP-BOOL, grup 6); B3'ten (grup 9) beri kuvvet alanı ve
    termostat sınıflarının kendi kapısı da (`_bool_kapisi`) kullandığı için burada,
    ağır import taşımayan ortak modülde. 0 boyutlu nesne dizisinin (`dtype=object`)
    öğesine de bakar (`ravel`).
    """
    if isinstance(deger, (bool, np.bool_, complex, np.complexfloating)):
        return True
    if isinstance(deger, np.ndarray):
        if deger.dtype.kind == "O":
            return any(_bool_iceriyor(x) for x in deger.ravel())
        return deger.dtype.kind in "bc"
    if isinstance(deger, (list, tuple)):
        return any(_bool_iceriyor(x) for x in deger)
    return False


def _dize_iceriyor(deger: object) -> bool:
    """İç içe liste/demet/dizide dize (str/bytes, numpy'ninki de) öğe var mı (B1, grup 9).

    `_bool_iceriyor`ın eşi: `np.asarray(['1.0'], dtype=float)` dizeyi SESSİZCE sayı
    yapar. `md/factory.py`deydi (B1); V13'ten (grup 9 düzeltici) beri sınıf kapısı
    (`_bool_kapisi`) da kullandığı için burada, `_bool_iceriyor`ın yanında.
    """
    if isinstance(deger, (str, bytes)):
        return True
    if isinstance(deger, np.ndarray):
        if deger.dtype.kind == "O":
            return any(_dize_iceriyor(x) for x in deger.ravel())
        return deger.dtype.kind in "SU"
    if isinstance(deger, (list, tuple)):
        return any(_dize_iceriyor(x) for x in deger)
    return False


def _bool_kapisi(sahip: str, **alanlar: object) -> None:
    """Sınıf düzeyinde kapı (B3, grup 9): sayı alanında bool ya da karmaşık -> `InvalidInputError`.

    ÖLÇÜLDÜ (14.10.2026, HEAD 448f808, sınıflar DOĞRUDAN): `Morse(d_e=True)`, LJ
    `epsilon=np.True_`, `Harmonic(k=np.array(True))`, `Langevin(friction=True, seed=True)`,
    `NoseHoover(relaxation_time=True, n_dof=True)`, `HartreeFockGradient(step=True)` ...
    54 bool hücresinin 43'ü SESSİZCE kabul ediliyordu (değer 1 diye kullanılır;
    `Harmonic(center=[[True, 0, 0]])` 1.0 bohr'luk merkez); kalan 11'i yanlış nedenle
    (L/2 kapısı) ya da ham `TypeError`la düşüyordu. Kapı fabrikada vardı (`_senaryo_nesnesi`)
    ama sınıflar kamu API'sidir: `observable/bridge_obs.py` `Morse`u doğrudan kurar.
    Değer DEĞİŞTİRİLMEZ (yalnız reddedilir): kabul edilen her girdinin yörüngesi bit bit aynı.

    DİZE (V13, grup 9 düzeltici, 23.10.2026; ölçüldü, HEAD 448f808'de de aynı): B3'ün
    kapsamı bool/karmaşıktı. `Morse(d_e='0.17')`, `LennardJones(cutoff='4')`,
    `Langevin(friction='2e-5')`, `NoseHoover(relaxation_time='100')` ham `TypeError`
    ("ufunc 'isfinite' not supported") veriyordu, `QLabError` DEĞİL; `Harmonic(center=
    [['0', 0, 0]])` ise dizeyi SESSİZCE 0.0 bohr yapıyordu. Dize de (str/bytes, numpy
    'S'/'U', nesne dizisinin içinde de) alanın adıyla reddedilir.
    """
    for ad, deger in alanlar.items():
        if _bool_iceriyor(deger):
            raise InvalidInputError(
                f"{sahip}.{ad} sayı olmalı: bool ve karmaşık değer sayıya çevrilmez "
                f"(true -> 1.0), verilen: {deger!r}"
            )
        if _dize_iceriyor(deger):
            raise InvalidInputError(
                f"{sahip}.{ad} sayı olmalı: dize (tırnaklı metin) sayıya çevrilmez "
                f"('1.0' -> 1.0), verilen: {deger!r}"
            )
