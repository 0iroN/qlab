"""Bant yapisi cizim YUKU -- goruntu uretmez.

Deponun `viz/` katmani hep yuk uretir (`frames.py`, `md_frames.py`,
`density_frames.py`); `.png` yazmak yeni bagimlilik ve yeni katman demektir.
Olculdu: `src/` altinda matplotlib/savefig/imsave/.png icin SIFIR eslesme --
cizim tamamen `app` katmaninda, canli.

DIKKAT: bu bir DISIPLINDIR, SOZLESME DEGIL. Olculdu, `viz` katmaninin GUI
yasagi YOK (`viz/band_frames.py -> PySide6` rc=0) ve mevcut
`viz/isosurface.py:55` zaten pyqtgraph import ediyor. Yani bu modulun cizim
yapmamasi import-linter tarafindan KORUNMUYOR.

OZVEKTOR SAKLANMAZ -- ve gerekce bellek DEGIL, ANLAM. Olculdu (SSH, hucre
gauge'u vs atom gauge'u):
  k=0.00 -> ozvektor farki 0.000000, agirlik farki 0.000e+00
  k=0.37 -> ozvektor farki 0.501586, agirlik farki 1.110e-16
  k=0.60 -> ozvektor farki 0.814887, agirlik farki 3.331e-16
Yani ozvektor GAUGE'A BAGLI, |c|^2 agirligi degil. Ozvektor saklamak,
cizime tasinamayan bir sayiyi saklamak olurdu. Bellek kazanci da var ama
kucuk ve ikincil: complex128 -> float64, tam 2 kat.

BELLEK OLCUTU "ndim <= 2" DEGIL "KOMPLEKS DIZI YOK"tur: agirliklar mesru
sekilde (n_k, n_band, n_orb) uc boyutludur. Modul 2E'de olcut sekle
bakiyordu ve bir listenin ICINE bakmadigi icin mutanti kacirmisti; buradaki
olcut o dersin duzeltilmis halidir.

AGIRLIKLAR DEJENERE ALT UZAYDA BANT BASINA OKUNAMAZ. Tam dejenere bir alt
uzayda `eigh`in sectigi taban keyfidir; olculdu, 2B karede x/y yonlu iki
orbitalde numpy [1,0]/[0,1] donduruyor ama 45 derece dondurulmus cift de
ayni olcude gecerli bir ozvektor cifti ve agirliklari [0.5,0.5]/[0.5,0.5].
ALT UZAY TOPLAMI iki tabanda da ayni. Bant basina agirlik yalniz dejenere
OLMAYAN noktalarda anlamlidir.

BIRIM SINIRI BURASIDIR: motor hartree konusur, yuk eV dondurur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.tb.bands import KPath
from qlab.engine.tb.hamiltonian import TBModel, bloch_hamiltonian
from qlab.errors import InvalidInputError
from qlab.units import HARTREE_TO_EV


@dataclass(frozen=True, eq=False)
class BandPayload:
    """Bant yapisi cizimi icin gereken her sey. Enerjiler eV.

    `eq=False` bilinclidir: olculdu, `frames.py:23`teki `FramePayload`
    `eq=True` tasiyor ve `p1 == p2` cagrisi numpy alanlari yuzunden
    ValueError atiyor.
    """

    k_distance: np.ndarray
    energies_ev: np.ndarray
    orbital_weights: np.ndarray
    tick_distances: np.ndarray
    tick_labels: tuple[str, ...]
    fermi_ev: float


def build_band_payload(
    model: TBModel,
    path: KPath,
    *,
    tick_labels: tuple[str, ...],
    fermi_hartree: float,
) -> BandPayload:
    """Yol boyunca bantlari ve orbital agirliklarini uretir; ozvektoru ATAR."""
    if len(tick_labels) != len(path.tick_distances):
        raise InvalidInputError(
            f"etiket sayisi kose sayisiyla uyusmuyor: {len(tick_labels)} != "
            f"{len(path.tick_distances)}"
        )
    if not np.isfinite(fermi_hartree):
        raise InvalidInputError(f"fermi_hartree sonlu olmali: {fermi_hartree}")

    enerjiler: list[np.ndarray] = []
    agirliklar: list[np.ndarray] = []
    for k in path.points:
        ozdeger, ozvektor = np.linalg.eigh(bloch_hamiltonian(model, k))
        enerjiler.append(ozdeger)
        # |c|^2: (n_orb, n_band) -> bant ekseni one alinir. Ozvektor bu
        # satirdan sonra ERISILEBILIR DEGILDIR -- gauge'a bagli olan tek
        # sey oydu ve tasinmiyor.
        agirliklar.append((np.abs(ozvektor) ** 2).T)

    return BandPayload(
        k_distance=np.asarray(path.distances, dtype=float),
        energies_ev=np.asarray(enerjiler, dtype=float) * HARTREE_TO_EV,
        orbital_weights=np.asarray(agirliklar, dtype=float),
        tick_distances=np.asarray(path.tick_distances, dtype=float),
        tick_labels=tuple(tick_labels),
        fermi_ev=float(fermi_hartree) * HARTREE_TO_EV,
    )
