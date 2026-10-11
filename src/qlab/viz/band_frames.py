"""Bant yapisi cizim YUKU -- goruntu uretmez.

Deponun `viz/` katmani hep yuk uretir (`frames.py`, `md_frames.py`,
`density_frames.py`); `.png` yazmak yeni bagimlilik ve yeni katman demektir.
Olculdu: `src/` altinda matplotlib/savefig/imsave/.png icin SIFIR eslesme --
cizim tamamen `app` katmaninda, canli.

11.10.2026'DAN BERI BU BIR SOZLESMEDIR (`pyproject.toml`, "viz yukleri GUI'yi
bilmez"): o gune kadar bir disiplindi -- olculdu, `viz/band_frames.py ->
PySide6` rc=0 ile geciyordu. `viz/isosurface.py:55` pyqtgraph import eder ve
MUAFTIR; muafiyet tek bir kenardir (`density_frames -> isosurface`).

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

E3a (04.10.2026): bu yuzden yuk dejenere kumede bant basina agirlik yerine
ALT UZAY ORTALAMASINI yazar (taban-bagimsiz) ve `degenerate` bayragiyla
isaretler. "Dejenere" = ardisik ozdegerler 1e3 * eps * (|E|max + H olcegi)
icinde; kumeleme GECISLIDIR (ardisik farklar esik altindaysa kume tek
parcadir ve yayilimi esigi asabilir -- 40'lik zincirde 35 esik olculdu).
H olcegi terimi YUK TASIR: birbirini goturen hoplamalarda toplama
kirintisi |E|max kadardir (dogrulama turu). SUREKSIZLIK YOK OLMAZ,
OLCEGI KUCULUR: esigin hemen alti ve ustu yine 0.5 / 1.0 verir. Sabit
1e-8 hartree (on olcum) bunu fiziksel olarak cozulebilen ayrismalara
tasiyordu; bu esik makine hassasiyetinin ~1e3 katidir. Esigin icinde
`eigh`in agirligi ~1e-3 dogrulukla belirli olabilir; yine de BILEREK
ortalanir -- muhafazakar secim (olculdu: eigh'in gercek ayrisma kirintisi
en cok 8e-3 esik, 240 orbitale ve |E| 300 hartree'ye kadar).

BANT EKSENI ENERJI SIRASIDIR, BANT KIMLIGI DEGIL (bkz. `band_structure`):
her (k, E, w) noktasi tek basina dogrudur, ama ayni indisi k boyunca
cizgiyle birlestirmek kesisimde bantlari degistirir. Bant takibi BILEREK
eklenmedi -- on olcumde kacinilmaz kesisimde bulunan "kimlik" adim
yogunluguna bagliydi; bant cozumlu nicelik (etkin kutle vb.) uretilmez.

BIRIM SINIRI BURASIDIR: motor hartree konusur, yuk eV dondurur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.tb.bands import KPath, hamiltonian_rounding_scale
from qlab.engine.tb.hamiltonian import TBModel, bloch_hamiltonian
from qlab.errors import InvalidInputError
from qlab.units import HARTREE_TO_EV

# E3a: dejenere kume esigi, |E|max + H olcegi'nin bu kati kadar eps. Ayni
# katsayi `tb_obs.certified_conduction`in yuvarlama payindadir.
_DEJENERE_KATSAYISI = 1e3


@dataclass(frozen=True, eq=False)
class BandPayload:
    """Bant yapisi cizimi icin gereken her sey. Enerjiler eV.

    `eq=False` bilinclidir: olculdu, `frames.py:23-24`teki `FramePayload`
    `eq=True` tasiyor ve `p1 == p2` cagrisi numpy alanlari yuzunden
    ValueError atiyor.
    """

    k_distance: np.ndarray
    energies_ev: np.ndarray
    orbital_weights: np.ndarray
    degenerate: np.ndarray
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
    dejenereler: list[np.ndarray] = []
    olcek = hamiltonian_rounding_scale(model)
    for k in path.points:
        ozdeger, ozvektor = np.linalg.eigh(bloch_hamiltonian(model, k))
        enerjiler.append(ozdeger)
        # |c|^2: (n_orb, n_band) -> bant ekseni one alinir. Ozvektor bu
        # satirdan sonra ERISILEBILIR DEGILDIR -- gauge'a bagli olan tek
        # sey oydu ve tasinmiyor.
        agirlik = (np.abs(ozvektor) ** 2).T
        dejenere = np.zeros(ozdeger.shape[0], dtype=bool)
        esik = _DEJENERE_KATSAYISI * np.finfo(float).eps * (
            float(np.abs(ozdeger).max()) + olcek
        )
        bas = 0
        for son in range(1, ozdeger.shape[0] + 1):
            if son == ozdeger.shape[0] or ozdeger[son] - ozdeger[son - 1] > esik:
                if son - bas > 1:
                    agirlik[bas:son] = agirlik[bas:son].mean(axis=0)
                    dejenere[bas:son] = True
                bas = son
        agirliklar.append(agirlik)
        dejenereler.append(dejenere)

    return BandPayload(
        k_distance=np.asarray(path.distances, dtype=float),
        energies_ev=np.asarray(enerjiler, dtype=float) * HARTREE_TO_EV,
        orbital_weights=np.asarray(agirliklar, dtype=float),
        degenerate=np.asarray(dejenereler, dtype=bool),
        tick_distances=np.asarray(path.tick_distances, dtype=float),
        tick_labels=tuple(tick_labels),
        fermi_ev=float(fermi_hartree) * HARTREE_TO_EV,
    )
