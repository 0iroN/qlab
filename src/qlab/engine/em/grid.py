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

#: Bu modulde HUCRE ile UZUNLUK AYNI SEYDIR. `dx` bir DUGME DEGIL bir
#: BEYANDIR ve gerekcesi `dx_kapisi`da OLCULU duruyor.
HUCRE_DX: float = 1.0


def dx_kapisi(dx: float) -> None:
    """`dx != 1` REDDI -- ve bu bir "yapamiyorum" DEGIL bir BEYANDIR.

    SINIF: BEYAN. `npml`in `EMScenario`da gordugu muameleyle ayni statude,
    ama SEBEBI FARKLI ve fark onemli: `npml` BAGLANMAMIS bir dugmeydi, `dx`
    ise BAGLI ama FAZLADAN bir dugmedir -- `wavelength`in olcekli bir
    kopyasi. Ucu de olculdu ve ucu de BIT BIT:

    (a) KAYNAKSIZ kosumda `dx` HICBIR SEYE dokunmuyor. Pencere: 1B vakum,
        PEC, N = 200, S = 0.5, 400 adim, Gauss tohumu (merkez 90, sg = 6).
        dx = 0.25/0.5/1/2/7 -> `final.e` de prob dizileri de BESINDE DE
        `np.array_equal` (max|fark| = 0).
    (b) KAYNAKLI 1B kosumda YALNIZ `wavelength/dx` orani sayiyor. Pencere:
        N = 1200, PML 40, S = 0.5, 900 adim, gauss_cos sgf = 3,
        t0 = 5*sgf*W, kaynak 300, problar 700/900. (dx, W) = (1, 40) /
        (0.5, 20) / (2, 80) / (0.25, 10) / (8, 320) -- hepsinde W/dx = 40 --
        BESI DE BIT BIT AYNI. Ayni `W` ile `dx` oynatilirsa (0.5 ve 2)
        bagil fark 1.44 ve 5.73 olur, yani sonuc BASKA BIR KOSUMDUR.
    (c) 2B'de aynisi. Pencere: 120 x 120, PML 20, S = 0.5, 400 adim,
        sgf = 2, kaynak (40, 40), problar (70, 70)/(90, 55). (dx, W) =
        (1, 20)/(0.5, 10)/(2, 40)/(4, 80) -- DORDU DE BIT BIT AYNI.
    (d) Egik kosumda da aynisi. Pencere: N = 800, PML 40, S = 0.4, 600
        adim, sgf = 3, ky = 0.5553603673 SABIT, kaynak 200, problar
        400/500. (dx, W) = (1, 16)/(0.5, 8)/(2, 32)/(8, 128) -- DORDU DE
        BIT BIT AYNI.

    Yani `dx` bir COZUNURLUK dugmesi OLAMAZ, cunku IZGARAYI olcekleyemez:
    katsayilar zaten olcekten bagimsizdir (`cb = S/eps`, icinde `dx` YOK),
    geometri hucre cinsindendir (`probes`, `source_index`,
    `CircularPEC.radius`, `npml`), PML `sigma`si hucre basinadir ve
    `oblique`in `ky`si radyan/hucredir. `dx != 1`de degisen TEK sey
    KAYNAGIN SAATIDIR -- ve o saat `wavelength`in ta kendisidir. Bu modulde
    cozunurluk `wavelength` ile degistirilir; `observable/em_obs`in
    yakinsama merdiveni de tam olarak oyle kurulmustur.

    BEDELI OLCULDU: `dx != 1`de `as_lines()`in "lambda ... hucre" satiri
    tam olarak 1/dx kadar YALAN olur. Pencere: S = 0.5, N = 5000, PML 40,
    kaynak 1500, problar 2100/2107, sgf = 2, dikdortgen FFT penceresi;
    izgaradaki dalga boyu iki prob arasindaki FAZ FARKINDAN okundu.
    `wavelength = 40` beyaniyla:

        dx           1            0.5          2
        olculen lam  39.969098    79.984570    19.937834   (hucre)
        beyan        "40 hucre"   "40 hucre"   "40 hucre"

    KAPI KAPANINCA O BEYAN DOGRU OLUYOR ve ILK KEZ SINANABILIYOR -- bkz.
    `tests/em/test_hucre_birimi.py`.

    SAF KAPI: bir DEGER DONDURMEZ ve girdiyi NORMALLESTIRMEZ. Ilk hali
    `float(dx)` alip onu donduruyordu; mutasyon kosumunda `return d ->
    return HUCRE_DX` HAYATTA KALDI ve kalmasi DOGRUYDU -- kapiyi gecen `d`
    zaten tam olarak `HUCRE_DX`tir, yani mutant ISPATLANABILIR SEKILDE
    ESDEGERDI. Esdeger bir mutant bir test acigi degildir; ama kapatilmis
    bir dugmeyi normallestiren bir satir OLU KOD'dur. Ikisi de silindi.
    """
    if dx != HUCRE_DX:
        raise InvalidInputError(
            f"dx = {dx} REDDEDILDI. Bu modulde HUCRE ile UZUNLUK AYNI "
            "SEYDIR (c = eps0 = mu0 = 1, dt = S) ve `dx` bir dugme degil "
            "bir BEYANDIR. `dx` AYRI BIR SERBESTLIK DERECESI DEGILDIR: "
            "davranis (dx, wavelength) ciftine YALNIZ wavelength/dx orani "
            "uzerinden baglidir ve bu BIT BIT olculdu -- 1B'de "
            "(dx, W) = (1, 40)/(0.5, 20)/(2, 80)/(0.25, 10)/(8, 320) BESI "
            "DE ayni diziyi veriyor, 2B'de (1, 20)/(0.5, 10)/(2, 40)/"
            "(4, 80) DORDU DE. Kaynaksiz kosumda ise `dx` hicbir seye "
            "dokunmuyor (dx = 0.25/0.5/1/2/7, max|fark| = 0). Yani `dx` "
            "izgarayi olcekleyemez, yalniz KAYNAGIN SAATINI olcekler -- ve "
            "o saat `wavelength`in kendisidir. Bedeli: dx = 0.5 / 2'de "
            "`wavelength = 40` beyani izgarada 79.984570 / 19.937834 hucre "
            "dalga boyu uretiyor, oysa pencere ucunde de 'lambda : 40 "
            "hucre' basiyor. COZUNURLUGU `wavelength` ile degistirin. "
            "Ayrinti: grid.dx_kapisi."
        )


@dataclass(frozen=True, eq=False)
class FieldState:
    """Yee izgarasindaki alan durumu.

    `eq=False`: alanlar ndarray oldugu icin uretilen `__eq__` bir dizi
    dondurur ve `__hash__` `TypeError` verir. Depo bu tuzagi
    `engine/hf/integrals.py`, `engine/md/state.py` ve `engine/spin/lattice.py`
    de zaten odedi; ayni gerekce.

    BIRIM SECIMI -- BEYAN: c = eps0 = mu0 = 1. Bu bir BIRIM SECIMIDIR, FIZIK
    DEGIL, ve Modul 4'un k_B = 1'iyle ayni statudedir: bu modulun hicbir
    ciktisi metre, saniye ya da volt tasimaz. Uzunluk HUCRE cinsindendir ve
    zaman da uzunluk cinsindendir (c = 1); yani `dx = 1` ve `dt = S`.

    BU SATIR BIR ZAMANLAR *"Uzunluk `dx` cinsinden ... `dt = S * dx`"*
    DIYORDU ve o hal IKI OKUMAYA ACIKTI: `dx` hem "bir hucrenin uzunlugu"
    hem "1" diye okunabiliyordu, ve `wavelength`in HUCRE mi UZUNLUK mu
    oldugu cumleden CIKARILAMIYORDU. Ikisi yalniz `dx = 1`de ortusuyor ve
    fark OLCULDU: `wavelength = 40` beyaniyla izgarada olusan dalga boyu
    dx = 1/0.5/2 icin 39.969098/79.984570/19.937834 hucre. Belirsizlik
    `dx != 1` KAPATILARAK giderildi -- bkz. `dx_kapisi`.

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

    BU SINIFIN *"1B ve 2B tanimlar"* IDDIASI TASK 4'TE OLCULEREK CURUDU.
    Task 1-3 boyunca `dim = 2` KABUL EDILIYORDU ama HICBIR sekil denetimi
    yoktu: `FieldState(np.zeros(10), np.zeros(3), 2)` sessizce gecerdi ve
    `dim = 1`de tipe zorlanan yarim-hucre kaydirmasi 2B'de HIC sinanmazdi.
    Yani "2B tanimlar" cumlesi bir tanim degil, DENETIMSIZ BIR DELIKTI --
    ve sevk edilince ilk kez sinanabilir hale geldi. Simdi `dim = 2` bu
    siniftan REDDEDILIYOR; 2B durum `FieldState2D`de ve orada UC alanin
    ucu de sekil sekil denetleniyor.
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
        else:
            raise InvalidInputError(
                "FieldState YALNIZ 1B durum tutar (dim = 1); 2B durum "
                "FieldState2D'dedir cunku IKI H bileseni (Hx, Hy) tek bir `h` "
                f"alanina sigmaz. Verilen dim = {self.dim}"
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


@dataclass(frozen=True, eq=False)
class FieldState2D:
    """2B (TM_z) alan durumu: Ez, Hx, Hy.

    `eq=False`: gerekce `FieldState`teki ile ayni.

    SEKILLER YARIM HUCRE KAYDIRMASININ KENDISIDIR ve ucu de denetlenir:

        e  = Ez(i, j)          (nx, ny)
        hx = Hx(i, j+1/2)      (nx, ny-1)
        hy = Hy(i+1/2, j)      (nx-1, ny)

    Yani `hx` YALNIZ y yonunde, `hy` YALNIZ x yonunde bir hucre kisadir.
    Ikisini de (nx-1, ny-1) yapmak (yaygin bir uygulama kisayolu) kaydirmayi
    bir yonde dusururdu; burada sekil denetimi bunu TIPTE engelliyor --
    `FieldState`in 1B'de yaptiginin aynisi, ve `FieldState`in `dim = 2`
    delikinin kapatilma sebebi (bkz. FieldState dokstringi).

    AYRI BIR SINIF, cunku 2B'de IKI H bileseni var. `dim` bir ALAN degil
    OZELLIKTIR: bir 2B durumun `dim`i girdiyle degistirilemez.
    """

    e: np.ndarray
    hx: np.ndarray
    hy: np.ndarray

    def __post_init__(self) -> None:
        for ad in ("e", "hx", "hy"):
            object.__setattr__(self, ad, np.asarray(getattr(self, ad)))
        for ad in ("e", "hx", "hy"):
            dizi = getattr(self, ad)
            if not np.issubdtype(dizi.dtype, np.number):
                raise InvalidInputError(
                    f"{ad} sayisal olmali, verilen dtype: {dizi.dtype}"
                )
            if dizi.ndim != 2:
                raise InvalidInputError(
                    f"2B'de {ad} iki boyutlu olmali, verilen: {dizi.shape}"
                )
        nx, ny = self.e.shape
        if nx < 3 or ny < 3:
            raise InvalidInputError(
                f"2B izgara her yonde en az 3 hucre olmali (dort kenarda PEC "
                f"duvar var), verilen: {nx} x {ny}"
            )
        if self.hx.shape != (nx, ny - 1):
            raise InvalidInputError(
                f"hx[i, j] = Hx(i, j+1/2) oldugu icin sekli {(nx, ny - 1)} "
                f"olmali (yalniz y yonunde TAM BIR HUCRE kisa); verilen: "
                f"{self.hx.shape}"
            )
        if self.hy.shape != (nx - 1, ny):
            raise InvalidInputError(
                f"hy[i, j] = Hy(i+1/2, j) oldugu icin sekli {(nx - 1, ny)} "
                f"olmali (yalniz x yonunde TAM BIR HUCRE kisa); verilen: "
                f"{self.hy.shape}"
            )

    @property
    def dim(self) -> int:
        return 2

    @property
    def shape(self) -> tuple[int, int]:
        nx, ny = self.e.shape
        return int(nx), int(ny)

    @classmethod
    def zeros_2d(cls, nx: int, ny: int, *, dtype: type = float) -> FieldState2D:
        if nx < 3 or ny < 3:
            raise InvalidInputError(
                f"2B izgara her yonde en az 3 hucre olmali, verilen: {nx} x {ny}"
            )
        return cls(
            np.zeros((nx, ny), dtype=dtype),
            np.zeros((nx, ny - 1), dtype=dtype),
            np.zeros((nx - 1, ny), dtype=dtype),
        )


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

    `dx` BIR DEGER TASIR ama BIR DUGME DEGILDIR: `__post_init__` onu
    `dx_kapisi`ndan gecirir (kapi SAF'tir, degeri degistirmez), yani BU
    TIPTE `dx != 1` bir pencere KURULAMAZ.
    Sebep dogrudan `as_lines()`tir: "lambda ... hucre" satiri ancak hucre
    ile uzunluk ayni seyken DOGRUDUR, ve `dx != 1`de tam olarak 1/dx kadar
    yalan olur (olculdu; bkz. `dx_kapisi`). Kapi kosum fonksiyonlarinda ZATEN
    var -- buradaki ikinci kapi, ELLE kurulan bir pencerenin yalan
    soyleyememesi icindir; `FieldState`in `dim = 2` deligini TIPTE kapatmasiyla
    ayni gerekce.
    """

    cells: int | tuple[int, int]
    dx: float
    courant: float
    steps: int
    wavelength: float
    npml: int
    sgf: float
    window_fn: str

    def __post_init__(self) -> None:
        dx_kapisi(self.dx)

    @property
    def izgara(self) -> str:
        """Izgara beyani: 1B'de `N`, 2B'de `nx x ny`.

        `cells` 2B'de bir IKILIDIR ve tek bir tamsayiya indirgenmez. Carpimi
        (toplam hucre) yazmak nx ile ny'yi gorunmez kilardi ve bu modulde
        `R` merdiveninin butun hukmu izgaranin KENARINDAN okunuyor.
        """
        if isinstance(self.cells, tuple):
            return " x ".join(str(int(n)) for n in self.cells)
        return str(int(self.cells))

    def as_lines(self) -> list[str]:
        """Panelde ve testlerde basilacak tek satirlik beyan parcalari.

        "lambda ... hucre" satiri bir TEOREMDIR, bir etiket degil: `dx`
        kapisi (bkz. `dx_kapisi`) hucre ile uzunlugu ayni sey yaptigi icin
        `wavelength` DOGRUDAN hucre sayisidir. Sinav olculdu ve daireselligi
        yok -- izgaradaki dalga boyu iki prob arasindaki faz farkindan
        okunup Yee'nin KAPALI FORM dispersiyonuyla karsilastiriliyor
        (`tests/em/test_hucre_birimi.py`). SATIR SAYISI YEDI, ve bu bir
        bicim tercihi degil kilit: `tests/em/test_yee.py` sayiyor.
        """
        return [
            f"izgara      : {self.izgara} hucre, dx = {self.dx:g}",
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
    4000 adim, uc tohum), d = 3 ciplak periyodik cekirdekle (sevk edilmis
    karsiligi YOK ve olmayacak):

        d = 1  S_max = 1.000036472 / 1.000036294 / 1.000038000   (PEC duvar)
        d = 1  S_max = 1.000005483 / 1.000005224 / 1.000005889   (periyodik)
        d = 2  S*sqrt(2) = 1.000010496 / 1.000012907 / 1.000010883 (periyodik)
        d = 3  S*sqrt(3) = 1.000048611 / 1.000047909 / 1.000046669

    d = 2 ARTIK CIPLAK CEKIRDEGIN SAYISI DEGIL: Task 4'te 2B sevk edildi ve
    esik SEVK EDILEN `_adim_2d` ile YENIDEN olculdu (PEC duvar, 400 adim,
    ikili arama tol 1e-7, uc tohum). Bu dokstringin bir zamanlar "sevk
    edilmis karsiligi yok" demesi Task 4'te gecerliligini yitirdi:

        L =  32 : S*sqrt(2) = 1.001941 / 1.002003 / 1.002102
        L =  64 : S*sqrt(2) = 1.000960 / 1.000995 / 1.001103
        L = 128 : S*sqrt(2) = 1.000786 / 1.000838 / 1.000781

    ve asimin IKI PARCASI 2B'de de aynen duruyor: izgara payi PEC kutuda
    yine tam olarak 1/cos(pi/(2(L-1))) (32/64/128 icin 1.001285/1.000311/
    1.000076; olculenin bu forma orani dokuz olcumde de 1.000655-1.000816,
    yani L'den BAGIMSIZ), kosum payi ise adimla iniyor (L = 64'te 400/1500/
    4000 adim -> +6.48e-04/+4.18e-05/+5.46e-06; sonuncusu 1B'nin ayni adim
    sayisindaki +5.99e-06'siyla ayni hanede). Ayrinti ve kilit:
    tests/em/test_bessel.py.

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

    eps_r < 1 BU SECIMIN DISINDADIR (28.09): dalga c'den hizlidir; 2B esik yine
    S/sqrt(eps_r) = 1/sqrt(2)'de olculdu (eps_r 0.25 ve 0.5, 3000 adim: sinirin
    0.999 katinda buyume 1.67, 1.001 katinda 3.65e+33; iki eps'te 4 hanede ayni,
    0.5 bagil 9e-13 / 7e-11 ayrisir; 0.25 ise 0.999'da eps_r = 1 ile bit bit, cunku
    2'nin kuvvetiyle olcekleme float'ta tamdir). 28.09'da kapi YALNIZ senaryo
    yolundaydi (`loader._load_em_scenario`, `factory._courant_kapisi`) ve uc
    katsayi yolu o S'yi kabul edip PATLIYORDU (run_1d PEC 3000 adim eps 0.25: S 0.51,
    etkin 1.02, sonlu degil). 01.10.2026'dan beri MOTOR DA kapar
    (`yee._etkin_courant_kapisi`, S/sqrt(min(eps_r*mu_r))); public yolla olculen
    esik/beklenen: 1B 1.000040 (eps 0.25, mu 0.25, 0.5*0.5; L = 200, 3000 adim),
    2B 1.000600 (L = 48, 1500 adim), egik ky = 0.8 s/p 1.000036 (L = 201, 3000
    adim) -- `mu_r < 1` ayni delikti ve fabrika onu goremezdi. KAPSAMI DISI, ACIK:
    egik p + `ey_kural` != aritmetik + konuma bagli mu_r (bkz. o fonksiyon).

    `dim` AYRIKTIR (1, 2, 3) ve surekli bir dugmeye cevrilemez -- spec 3.5'in
    ikincil ayirt edici olcutu. Bu yuzden d = 3 BURADA tanimlidir: 1/sqrt(3)
    boyut merdiveninin ucuncu basamagidir ve semanin saf bir sayisidir.
    COZUCU ise 3B'yi REDDEDER (bkz. yee.run_2d): sema SAYISI bedavadir, 3B
    KOSUM degildir. Olculdu (ciplak np.roll cekirdegi, D = 80):

        bellek : 2B 0.15 MiB, 3B 23.44 MiB -> 156 kat, ve bu ORAN KARARLI
        sure   : 2B 0.028-0.073 ms/adim, 3B 18.1-24.7 ms/adim

    SURE ORANININ SAYISI YOKTUR: uc bagimsiz olcumde 258 / 443 / 896 kat
    cikti, cunku 2B tarafi numpy CAGRI YUKUNE bogulur (D = 80'de dizi cok
    kucuk). Yazilabilecek olan bir ALT SINIRDIR -- birkac yuz kat -- ve o da
    ciplak cekirdegin sayisidir; PML + TF/SF'li gercek bir 3B cekirdek daha
    pahalidir.

    TASK 4'TE SEVK EDILEN CIPAYLA YENIDEN OLCULDU ve alt sinir SOMUTLASTI:
    ciplak 3B cekirdek D = 167'de (sevk edilen 2B cipanin izgarasinin TAM
    esi) 228.72 ms/adim ve alti alan 213.2 MiB; ayni 134400 adim orada
    ~512 dakika eder, 2B karsiligi 16.0 s -- yani ~1920 kat.
    """
    if dim not in (1, 2, 3):
        raise InvalidInputError(
            f"Courant siniri d = 1, 2, 3 icin tanimli, verilen dim = {dim}"
        )
    return float(dim) ** -0.5
