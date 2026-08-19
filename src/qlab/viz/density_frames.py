"""Zamana bağlı yoğunluk farkı filmi -- çizim YÜKÜ.

Bu katman görüntü üretmez. `.mp4`/PNG yazmak yeni bağımlılık ve yeni katman
demektir; depo deseni (viz/frames.py, viz/md_frames.py) yük üretmektir.

BELLEK SÖZLEŞMESİ (ölçüldü): alan küpü SAKLANMAZ. (M,n,n,n) float64 M=100 /
n=80 için 0.410 GB tutar; izoyüzey ağları ise kare başına yalnızca 50-410 tepe
(n=41...81). Aynı 9 karenin ağ yükü 29.5 / 78.3 / 142.0 KB, alan küpleri ise
4.96 / 16.3 / 38.3 MB -- 168x / 209x / 269x. Alan üretilir, ağ çıkarılır,
alan atılır.

MALİYET (ölçüldü, uçtan uca, kare başına): n=41 -> 21.6 ms, n=61 -> 63.1 ms,
n=81 -> 141.5 ms; 100 karelik film 2.16 / 6.31 / 14.15 s. Dağılım: yoğunluk
alanı ~%64, izoyüzey %31-35, SCF %6'dan az (n büyüdükçe %1'in altına iner --
pahalı olan elektronik yapı değil, ONU ÖRNEKLEMEKTİR).

Alan kare başına TAM BİR KEZ hesaplanır ve bu bir test tarafından sayılarak
kilitlenir. Ölçüldü: iki kez hesaplayan ilk sürüm uçtan uca maliyeti 1.65
katına çıkarıyordu (n=41'de 33.5 ms).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.hf.scf import solve
from qlab.engine.md.integrator import Trajectory
from qlab.errors import InvalidInputError
from qlab.observable.hf_obs import (
    bond_midpoint_of,
    density_difference,
    symmetric_box,
)
from qlab.types import Molecule
from qlab.units import AU_TIME_TO_FEMTOSECOND, BOHR_TO_ANGSTROM
from qlab.viz.isosurface import Mesh, extract_signed_pair


@dataclass(frozen=True, eq=False)
class DensityFilmPayload:
    """Kare dizisi. `meshes` kare başına (+faz, -faz) ağ çifti."""

    times_fs: np.ndarray
    bond_length_bohr: np.ndarray
    midpoint: np.ndarray
    meshes: list[tuple[Mesh, Mesh]]
    grid_origin: np.ndarray
    grid_shape: tuple[int, int, int]
    isolevel: float


def build_density_film(
    trajectory: Trajectory,
    *,
    symbols: tuple[str, ...],
    half_width_bohr: float,
    n: int,
    basis: str = "sto-3g",
    isolevel: float = 0.01,
) -> DensityFilmPayload:
    """Yörüngenin her karesinde SCF çözer ve yoğunluk farkını okur.

    Kutu BİR KEZ kurulur ve tüm kareler boyunca sabit kalır (ölçüldü:
    kare başına kutu seçimi filmi kadraj hareketine çevirir).
    """
    semboller = tuple(symbols)
    if len(semboller) != trajectory.positions.shape[1]:
        raise InvalidInputError(
            f"sembol sayısı atom sayısıyla uyuşmuyor: {len(semboller)} != "
            f"{trajectory.positions.shape[1]}"
        )
    izgara = symmetric_box(half_width_bohr, n)  # parite muhafazası burada

    uzunluk, orta, agalar = [], [], []
    for kare in range(trajectory.positions.shape[0]):
        konum = trajectory.positions[kare]
        merkez = konum.mean(axis=0)
        # KUTU SABİT: molekül kutuya taşınır, kutu moleküle DEĞİL.
        yerlesik = konum - merkez
        angstrom = yerlesik * BOHR_TO_ANGSTROM
        # DİKKAT -- `!r` KULLANMA. NumPy 2'de repr(np.float64(0.0)) =
        # "np.float64(0.0)" ve Molecule.from_string bunu reddediyor.
        metin = "; ".join(
            "%s %.17g %.17g %.17g" % (s, r[0], r[1], r[2])
            for s, r in zip(semboller, angstrom)
        )
        sonuc = solve(Molecule.from_string(metin), basis, stability=False)

        uzunluk.append(float(np.linalg.norm(yerlesik[1] - yerlesik[0])))
        # Alan kare başına TAM BİR KEZ: hem orta nokta hem izoyüzey aynı
        # küpten okunur. Ölçüldü, iki kez hesaplamak kare maliyetini 1.65
        # katına çıkarıyordu (n=41'de 33.5 ms yerine 20.4 ms).
        alan = density_difference(sonuc, izgara)
        orta.append(bond_midpoint_of(alan, sonuc.molecule, a=0, b=1))
        agalar.append(extract_signed_pair(alan, isolevel))
        del alan  # alan küpü SAKLANMAZ

    return DensityFilmPayload(
        times_fs=np.asarray(trajectory.times) * AU_TIME_TO_FEMTOSECOND,
        bond_length_bohr=np.asarray(uzunluk),
        midpoint=np.asarray(orta),
        meshes=agalar,
        grid_origin=izgara.origin,
        grid_shape=izgara.shape,
        isolevel=isolevel,
    )
