"""Periyodik orgu geometrisi ve komsu bulma -- bohr biriminde.

Bu modul fizik yapmaz; yalniz geometri kurar. Uc olculmus sessiz hatanin
kaynagi burasidir ve ucu de burada kapatilir:

1. TERS ORGUDE 2*pi. `b_i . a_j = 2*pi*delta_ij` konvansiyonu kullanilir.
   Olculdu: 2*pi dusurulurse grafende K noktasindaki aralik 0 yerine
   15.605535020 eV cikiyor -- ve Gamma noktasinda yapilan kontrol IKI
   konvansiyonda da geciyor (8.100000000), yani hata Gamma'dan gorulmez.

2. KESME YARICAPINDA KAYAN NOKTA. `d <= cutoff` ciplak yazilirsa, cutoff tam
   komsu mesafesine esit oldugunda hesaplanan mesafe bir-iki ulp yukari
   dusup komsu SESSIZCE atilir. Olculdu: 11 fcc orgu sabitinin 5'inde
   `d <= a/sqrt(2)` 12 yerine SIFIR komsu seciyor. Bu yuzden karsilastirma
   bagil toleransla yapilir.

3. CAKISIK ATOM. `neighbour_shells` eskiden mesafesi 1e-9 bohr'un altindaki
   HER cifti atiyordu -- atomun kendisini de, cakisik iki farkli atomu da.
   Ikincisi sessiz bir fizik hatasiydi: cakisik cift hic yokmus gibi
   davranilir. Artik yalniz atomun kendisi atilir, digerleri hatadir
   (olcum `_CAKISMA_ESIGI_BOHR`un yaninda).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from qlab.errors import InvalidInputError

# Kesme yaricapi karsilastirmasinda kullanilan bagil tolerans. Olculdu:
# fcc'de birinci kabuk a/sqrt(2), ikincisi a; aradaki bosluk a'nin %29'u,
# yani 1e-9 bir sonraki kabugu icine alamaz (test_kesme_yaricapi_bir_
# sonraki_kabugu_ALMIYOR bunu kilitler).
_KESME_TOLERANSI = 1e-9

# Cakisik atom esigi (MUTLAK, bohr). Taban atomlari ya da goruntuleri bu
# kadar yakinsa `neighbour_shells` `InvalidInputError` firlatir. Deger,
# bu kapidan ONCEKI sessiz atma esigidir (`mesafe <= 1e-9` ise `continue`);
# boylece cakisik olmayan her geometride davranis BIREBIR aynidir.
# OLCULDU (24.09.2026): (1) eski sessiz atma, 8 atomlu zincirde (a = 2,
# L = 16, n_e = 8, kT = 2e-3) atom 1 atom 0'a 1e-9 yaklasinca kafes yolunun
# Omega'sini MD yolundan -18.85 hartree ayiriyordu (cift yok sayildigi icin);
# 2e-9'da iki yol hala ayniydi. (2) Atmayi hataya cevirmek `tests/tb`'nin
# 342 testinin HICBIRINI kirmadi (kapinin kopyada gercekten calistigi ayrica
# dogrulandi). MD yolu (`mermin.py`) yalniz d == 0.0'i reddeder; 0 < d <=
# 1e-9 bandinda kafes yolu hata verir, MD yolu hesaplar (beyanli fark).
_CAKISMA_ESIGI_BOHR = 1e-9


@dataclass(frozen=True, eq=False)
class NeighbourPair:
    """Bir hoplama yolu: `i` orbitalinden, `cell` hucresindeki `j` orbitaline."""

    i: int
    j: int
    cell: tuple[int, int, int]
    distance: float


@dataclass(frozen=True, eq=False)
class Lattice:
    """Periyodik orgu. `vectors` (d,3), `basis` (n_atom,3), ikisi de BOHR.

    `d` boyut sayisidir (1, 2 ya da 3); vektorler 3B kartezyen olarak verilir
    ama yalnizca `d` tanesi periyodiktir. `basis` hucre icindeki atom
    konumlaridir ve KARTEZYENDIR (kesirli degil) -- kesirli/kartezyen
    karisikligi bu deponun disinda cok yapilan bir hatadir; burada tip
    duzeyinde degil belge duzeyinde onlenir ve testle kilitlenir.
    """

    vectors: np.ndarray
    basis: np.ndarray

    def __post_init__(self) -> None:
        vektorler = np.asarray(self.vectors, dtype=float)
        taban = np.asarray(self.basis, dtype=float)
        object.__setattr__(self, "vectors", vektorler)
        object.__setattr__(self, "basis", taban)

        if (
            vektorler.ndim != 2
            or vektorler.shape[1] != 3
            or not 1 <= vektorler.shape[0] <= 3
        ):
            raise InvalidInputError(
                f"orgu vektoru dizisi (d,3) olmali, 1 <= d <= 3; "
                f"verilen: {vektorler.shape}"
            )
        if taban.ndim != 2 or taban.shape[1] != 3 or taban.shape[0] < 1:
            raise InvalidInputError(
                f"taban dizisi (n_atom,3) olmali, n_atom >= 1; verilen: {taban.shape}"
            )
        if not np.all(np.isfinite(vektorler)) or not np.all(np.isfinite(taban)):
            raise InvalidInputError("orgu vektorleri ve taban konumlari sonlu olmali")

        tekil = np.linalg.svd(vektorler, compute_uv=False)
        if tekil.min() <= 1e-10 * tekil.max():
            raise InvalidInputError(
                f"orgu vektorleri dogrusal bagimli (tekil degerler {tekil})"
            )

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def n_atom(self) -> int:
        return int(self.basis.shape[0])


def reciprocal_vectors(lattice: Lattice) -> np.ndarray:
    """`b_i . a_j = 2*pi*delta_ij` saglayan (d,3) ters orgu vektorleri.

    d < 3 durumunda eksik yonler dik tamamlayicilarla doldurulur, cozum
    alinir ve yalniz ilk d satir dondurulur -- boylece 1B ve 2B orguler de
    ayni konvansiyonu tasir.
    """
    a = lattice.vectors
    d = a.shape[0]
    if d == 3:
        tam = a
    else:
        tam = a.copy()
        for _ in range(3 - d):
            _, _, vt = np.linalg.svd(tam)
            tam = np.vstack([tam, vt[-1]])
    b_tam = 2.0 * np.pi * np.linalg.inv(tam).T
    return b_tam[:d]


def max_cell_for_cutoff(lattice: Lattice, cutoff_bohr: float) -> int:
    """`neighbour_shells`in `cutoff_bohr` icindeki HICBIR cifti kaybetmedigi max_cell.

    n = max_i ceil((kesme + taban yayilimi) * |b_i| / (2 pi)), en az 1.
    Kaniti: cift (i, j, R) kesme icindeyse |tau_j + R - tau_i| <= kesme,
    yani |R| <= kesme + max|tau_j - tau_i| (ucgen esitsizligi); R'nin i.
    tamsayi bileseni b_i . R / (2 pi)'dir ve mutlak degeri
    |b_i| |R| / (2 pi)'yi asamaz. Kesme, `neighbour_shells`in kullandigi
    bagil toleransla SISIRILIR (sinirdaki kabuk da sayilsin).

    NAIF FORMUL YETMEZ (olculdu, 24.09.2026): 15 derece egik 2B orguda
    (|a_1| = |a_2| = 1, kesme 0.79) `ceil(kesme / min|a|) + 1 = 2` alti
    komsunun ikisini SESSIZCE kaybediyor (4 buluyor); en kisa orgu vektoru
    a_2 - a_1 (0.261) ve ucuncu katlari (-3, 3) hucresinde. Bu formul 4
    veriyor ve 6'nin altisini da buluyor (`tests/tb/test_distance_model.py`
    M8).

    UST SINIRDIR, SIKI DEGILDIR (olculdu): 32 rastgele triklinik orguda
    (taban atomu hucre disinda) kayip 0, ama sinir gerekenin ~4 katina
    cikabiliyor (19'a karsi 5, 21'e karsi 8). Hucre sayisi (2n + 1)^d oldugu
    icin cok egik ya da patolojik olcekli orguler `neighbour_shells`i
    pratikte bitmez kilar (|a| = 5e-10, kesme 3.8 -> n = 7.6e9); bunun bir
    kapisi YOKTUR (acik kalan, beyanli).
    """
    if not np.isfinite(cutoff_bohr) or cutoff_bohr <= 0.0:
        raise InvalidInputError(
            f"kesme yaricapi pozitif ve sonlu olmali: {cutoff_bohr}"
        )
    b = reciprocal_vectors(lattice)
    fark = lattice.basis[:, None, :] - lattice.basis[None, :, :]
    yayilim = float(np.max(np.linalg.norm(fark, axis=-1)))
    menzil = cutoff_bohr * (1.0 + _KESME_TOLERANSI) + yayilim
    n = max(
        int(np.ceil(menzil * float(np.linalg.norm(b_i)) / (2.0 * np.pi)))
        for b_i in b
    )
    return max(1, n)


def neighbour_shells(
    lattice: Lattice, cutoff_bohr: float, max_cell: int = 2
) -> tuple[NeighbourPair, ...]:
    """`cutoff_bohr`'dan yakin (kendisi dahil) tum yonlu komsu ciftleri.

    Yonludur: (i, j, cell) varsa (j, i, -cell) de vardir. Atomun kendisi
    (i = j, cell = 0) haric tutulur; BASKA her cakisma (iki taban atomu ya
    da bir atomla bir goruntu, mesafe <= `_CAKISMA_ESIGI_BOHR`)
    `InvalidInputError`dir -- modul dokstringi, olculmus tuzak 3. Kapi yalniz
    taranan `max_cell` hucreleri icindeki cakismayi gorur;
    `max_cell_for_cutoff` ile secilen max_cell her cakismayi kapsar
    (cakisma mesafesi kesmenin altindadir).

    KESME KARSILASTIRMASI BAGIL TOLERANSLIDIR -- sebebi modul dokstring'inde,
    olculmus tuzak 2.
    """
    if not np.isfinite(cutoff_bohr) or cutoff_bohr <= 0.0:
        raise InvalidInputError(
            f"kesme yaricapi pozitif ve sonlu olmali: {cutoff_bohr}"
        )
    if max_cell < 1:
        raise InvalidInputError(f"max_cell en az 1 olmali, verilen: {max_cell}")

    d = lattice.dimension
    araliklar = [range(-max_cell, max_cell + 1)] * d + [range(0, 1)] * (3 - d)
    sinir = cutoff_bohr * (1.0 + _KESME_TOLERANSI)

    ciftler: list[NeighbourPair] = []
    for hucre in product(*araliklar):
        kayma = np.zeros(3)
        for eksen in range(d):
            kayma = kayma + hucre[eksen] * lattice.vectors[eksen]
        for i in range(lattice.n_atom):
            for j in range(lattice.n_atom):
                fark = lattice.basis[j] + kayma - lattice.basis[i]
                mesafe = float(np.linalg.norm(fark))
                if mesafe <= _CAKISMA_ESIGI_BOHR:
                    if i == j and not any(hucre):
                        continue  # atomun kendisi
                    raise InvalidInputError(
                        f"cakisik atom: taban {i} ile hucre {tuple(hucre)}'deki "
                        f"taban {j} arasi mesafe {mesafe:.3e} <= "
                        f"{_CAKISMA_ESIGI_BOHR:g} bohr (cift sessizce atilmaz)"
                    )
                if mesafe > sinir:
                    continue
                ciftler.append(
                    NeighbourPair(
                        i=i,
                        j=j,
                        cell=tuple(int(x) for x in hucre),
                        distance=mesafe,
                    )
                )
    return tuple(ciftler)
