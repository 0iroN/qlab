"""Yee izgarasinin BOYUTTAN BAGIMSIZ ortak parcasi: alan durumu, pencere,
kayit ve semanin kendi kararlilik siniri.

Bu dosyada fizik YOK; burada duran her sey ya bir kap ya da Yee semasinin
kendi ozelligidir. Ortam `medium.py`de, guncelleme `yee.py`de, sogurucu
`boundary.py`de, darbe `source.py`dedir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError


@dataclass(frozen=True, eq=False)
class FieldState:
    """Yee izgarasindaki alan durumu.

    `eq=False`: alanlar ndarray oldugu icin uretilen `__eq__` bir dizi
    dondurur ve `__hash__` `TypeError` verir. Depo bu tuzagi
    `engine/hf/integrals.py`, `engine/md/state.py` ve `engine/spin/lattice.py`
    de zaten odedi; ayni gerekce.

    BIRIM SECIMI -- BEYAN: c = eps0 = mu0 = 1. Bu bir BIRIM SECIMIDIR, FIZIK
    DEGIL, ve Modul 4'un J = k_B = 1'iyle ayni statudedir: bu modulun hicbir
    ciktisi metre, saniye ya da volt tasimaz. Uzunluk `dx` cinsinden, zaman
    da uzunluk cinsindendir (c = 1); `dt = S * dx`.

    1B'de `e` = Ez, sekli (N,); `h` = Hy, sekli (N-1,) ve `h[i]` = Hy(i+1/2).
    YARIM HUCRE KAYDIRMA SEMANIN KENDISIDIR, bir uygulama ayrintisi degil --
    ve sonucu MERTEBE DUSMESI DEGIL, IRAKSAMADIR. `e`nin curl'u GERI fark
    yerine ILERI fark alinirsa (yani H, E ile ayni izgaraya tasinirsa) sema
    yakinsamiyor, PATLIYOR. PENCERE: sevk edilen katsayilar (vakum + PEC),
    PEC duvarli izgarada m = 8 duran kip, 400 adim; olculen buyukluk izdusum
    serisi `s[n] = <e_n, sin(k i)>`in TEPESI bolu `s[0]`:

                  N = 201     N = 401     N = 801     N = 1601
        S = 0.5   3.17e+159   7.99e+144   3.20e+145   7.99e+144
        S = 0.25  1.35e+65    4.38e+65    2.53e+64    1.69e+64
        S = 0.1   5.62e+14    2.70e+14    4.50e+13    1.13e+13

    Dort N de ayni sinif, yani cozunurluk artigi degil; SEVK EDILEN cekirdek
    ayni pencerede en fazla 1.000282 veriyor. Mertebe 2'den 1'e INMIYOR --
    olculecek mertebe yok, cunku yakinsayan seri yok. Kaydirma bir DOGRULUK
    ayrintisi degil, KARARLILIGIN kosulu; ayrinti ve kosulsuzluk olcumu
    `yee._adim`de.

    KILIT: tests/em/test_sema_cipalari.py, `test_KAYDIRMA_DUSURULUNCE_...`
    (kaydirmasiz cekirdegin kendisi ORADA duruyor, `src/`de degil).

    dtype KISITLANMIYOR (float ya da complex). Egik gelis indirgemesi
    (Task 3) d/dy -> i*ky ile KARMASIK alanlar kullanir ve ayni cekirdekten
    gecmek zorundadir; burada dtype=float'a zorlamak o yolu keserdi.
    """

    e: np.ndarray
    h: np.ndarray
    dim: int

    def __post_init__(self) -> None:
        # ONCE ndarray'e cevir: liste girdisi dogrulamayi ham `AttributeError`
        # ile dusururdu (`.ndim` yok). Ayni desen `engine/md/state.py`de.
        object.__setattr__(self, "e", np.asarray(self.e))
        object.__setattr__(self, "h", np.asarray(self.h))
        object.__setattr__(self, "dim", int(self.dim))
        for ad, dizi in (("e", self.e), ("h", self.h)):
            if not np.issubdtype(dizi.dtype, np.number):
                raise InvalidInputError(
                    f"{ad} sayisal olmali, verilen dtype: {dizi.dtype}"
                )
        if self.dim == 1:
            if self.e.ndim != 1 or self.h.ndim != 1:
                raise InvalidInputError(
                    f"1B'de e ve h 1 boyutlu olmali, verilen: "
                    f"{self.e.shape} ve {self.h.shape}"
                )
            if self.h.size != self.e.size - 1:
                raise InvalidInputError(
                    f"1B'de h, e'den TAM BIR HUCRE kisa olmali (h[i] = H(i+1/2)); "
                    f"verilen e.size = {self.e.size}, h.size = {self.h.size}"
                )
        elif self.dim != 2:
            raise InvalidInputError(
                f"FieldState yalniz 1B ve 2B tanimlar, verilen dim = {self.dim}"
            )

    @property
    def cells(self) -> int:
        """1B'de E dugum sayisi."""
        return int(self.e.shape[0])

    @classmethod
    def zeros_1d(cls, cells: int, *, dtype: type = float) -> FieldState:
        if cells < 3:
            raise InvalidInputError(
                f"1B izgara en az 3 hucre olmali (iki ucta PEC duvar var), "
                f"verilen: {cells}"
            )
        return cls(np.zeros(cells, dtype=dtype), np.zeros(cells - 1, dtype=dtype), 1)


@dataclass(frozen=True)
class Window:
    """PENCERE BEYANI -- bu modulde TOHUMUN yerini tutan sey.

    FDTD deterministtir: ayni girdi ayni sayiyi verir, RNG yoktur. Ama
    tekrarlanabilirlik tek basina yetmez, cunku bir FDTD sayisi yalniz kendi
    AYRIKLASTIRMA ve ANALIZ penceresi icinde anlamlidir. Bu depoda hukmun
    pencerenin disina tasinmasi ON KEZ olctu; bu yuzden pencere adlandirilmadan
    sayi yazilmaz.

    Alanlarin hepsi ZORUNLUDUR ve hepsi olculerek secildi: `sgf` buraya
    "analiz ayrintisi degil" diye girdi (bkz. source.gauss_cos), `window_fn`
    ise FFT penceresi tek basina sonucu oynattigi icin.
    """

    cells: int
    dx: float
    courant: float
    steps: int
    wavelength: float
    npml: int
    sgf: float
    window_fn: str

    def as_lines(self) -> list[str]:
        """Panelde ve testlerde basilacak tek satirlik beyan parcalari."""
        return [
            f"izgara      : {self.cells} hucre, dx = {self.dx:g}",
            f"S (Courant) : {self.courant:g}",
            f"adim        : {self.steps}",
            f"lambda      : {self.wavelength:g} hucre",
            f"npml        : {self.npml}",
            f"sgf         : {self.sgf:g}",
            f"FFT penceresi: {self.window_fn}",
        ]


@dataclass(frozen=True, eq=False)
class FieldRecord:
    """Bir kosumun ciktisi: prob zaman serileri + PENCERESI.

    `window` opsiyonel DEGILDIR. Kayit penceresiz dondurulebilseydi, sayinin
    hangi ayriklastirmaya ait oldugu cagrida kaybolurdu; `Window`u zorunlu
    alan yapmak bu kaybi TIPLE engeller.
    """

    probes: np.ndarray
    probe_indices: tuple[int, ...]
    window: Window
    final: FieldState


def courant_limit(dim: int) -> float:
    """S_max = 1/sqrt(d) -- Yee SEMASININ KENDI ozelligi, bir girdi DEGIL.

    SINIF: SEMA CIPASI -- cozucuyu dogrular, fizigi DEGIL. Sayi girilmez;
    sema kararli olmaktan cikar ve esik ikili aramayla OLCULUR.

    OLCULDU. d = 1 SEVK EDILEN cekirdekle (`_adim`, PEC duvar, L = 200,
    4000 adim, uc tohum), d = 2 ve 3 ciplak periyodik cekirdekle (sevk
    edilmis karsiligi yok):

        d = 1  S_max = 1.000036472 / 1.000036294 / 1.000038000   (PEC duvar)
        d = 1  S_max = 1.000005483 / 1.000005224 / 1.000005889   (periyodik)
        d = 2  S*sqrt(2) = 1.000010496 / 1.000012907 / 1.000010883
        d = 3  S*sqrt(3) = 1.000048611 / 1.000047909 / 1.000046669

    ESIKTEKI ARTIK ASIM IKI PARCADIR ve plan taslagi ikisini KARISTIRIYORDU
    ("sonlu kosum artefaktidir, 20000 adimda +2.1e-7'ye iniyor"). Olculdu:

    (a) SONLU IZGARA PAYI -- yalniz PEC duvarli izgarada. PEC'te ayrik
        kipler k_m = m*pi/(L-1), m = 1..L-2'dir, yani izgara k = pi'ye HIC
        ULASMAZ ve sonlu izgaranin TAM siniri 1'den buyuktur:
            S_max(L) = 1 / cos( pi / (2 (L-1)) ).
        20000 adimda olculen fark (L = 100/200/400/800):
            2.05e-07 / 2.33e-07 / 1.97e-07 / 1.79e-07
        yani kapali form TAM tutuyor. Bu bir kosum artefakti DEGIL, sonlu
        izgaranin kendi siniridir ve L^-2 ile kuculur (L = 200'de +3.1e-05).
    (b) SONLU KOSUM PAYI -- her iki sinirda da var ve L'den bagimsiz.
        Periyodik izgarada (k = pi TAM olarak mevcut, sinir tam 1) yalniz bu
        pay kalir: 4000 adimda +5.99e-06, 20000 adimda +2.12e-07; L = 100 /
        200 / 400 icin 20000 adimda 1.78e-07 / 2.12e-07 / 1.97e-07.

    BEYAN EDILEN MODELLEME SECIMI -- OLCUMDEN ONCE YAPILIR: `S` EN HIZLI
    dalga hiziyla tanimlidir. Ham kararlilik esigi ortama BAGLIDIR ve
    olculdu (SEVK EDILEN `_adim`, kapi atlanarak, L = 200, 3000 adim):

        eps_r      1.0          2.25         4.0          9.0
        S_max      1.000040301  1.500060657  2.000081036  3.000121823
        /sqrt(eps) 1.000040301  1.000040438  1.000040518  1.000040608

    Degismeyen sey `S_max/sqrt(eps_r)`dir. Yani "1/sqrt(d)" hukmu bu
    normalizasyona baglidir ve bu, bir olcum degil bir SECIMDIR.

    `dim` AYRIKTIR (1, 2, 3) ve surekli bir dugmeye cevrilemez -- spec 3.5'in
    ikincil ayirt edici olcutu. Bu yuzden d = 3 BURADA tanimlidir: 1/sqrt(3)
    boyut merdiveninin ucuncu basamagidir ve semanin saf bir sayisidir.
    COZUCU ise 3B'yi REDDEDER (bkz. yee.yee_step_1d): sema SAYISI bedavadir,
    3B KOSUM degildir. Olculdu (ciplak np.roll cekirdegi, D = 80):

        bellek : 2B 0.15 MiB, 3B 23.44 MiB -> 156 kat, ve bu ORAN KARARLI
        sure   : 2B 0.028-0.073 ms/adim, 3B 18.1-24.7 ms/adim

    SURE ORANININ SAYISI YOKTUR: uc bagimsiz olcumde 258 / 443 / 896 kat
    cikti, cunku 2B tarafi numpy CAGRI YUKUNE bogulur (D = 80'de dizi cok
    kucuk). Yazilabilecek olan bir ALT SINIRDIR -- birkac yuz kat -- ve o da
    ciplak cekirdegin sayisidir; PML + TF/SF'li gercek bir 3B cekirdek daha
    pahalidir.
    """
    if dim not in (1, 2, 3):
        raise InvalidInputError(
            f"Courant siniri d = 1, 2, 3 icin tanimli, verilen dim = {dim}"
        )
    return float(dim) ** -0.5
