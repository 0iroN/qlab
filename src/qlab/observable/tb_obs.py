"""Tight-binding gozlemlenebilirleri -- doluluk, Fermi seviyesi, DOS, yapi orani.

MODULUN TEK PARAMETRE-BAGIMSIZ FIZIK HUKMU: hucre basina TEK sayida elektron
her zaman metal verir, cunku tek sayi bir bandi ancak YARI doldurabilir. Bu
bir hesap degil bir SAYIMDIR. Olculdu: 100 rastgele model x 3 boyut =
300/300 metal. Panel (`certified_conduction`) bunu yine de IZGARADA TANIK
isteyerek basar: yari dolu bant izgarada duz gorunurse (orn. yalniz +-2
hucreye hoplayan zincir, k=2) hukum "belirsiz"dir.

TERSI DOGRU DEGILDIR ve bu modul onu iddia ETMEZ: cift sayim yalitkanligi
garanti etmez. On olcumde bant ortusmesi yuzunden metal cikma orani
1B %67, 2B %34, 3B %20 olculdu. Modulun asil asimetrisi budur: ILETKEN
yarisi sayimdan gelir ve dairesel degildir; YALITKAN yarisi fit'ten gelir.

OLCUT BANT COZUMLU OLMAK ZORUNDA. Komsu-seviye araligina bakan bir olcut
(orn. "bosluk/W < 1e-9") sonlu izgarada COKUYOR: olculdu, 10x10x10 izgarada
3B modellerin 0/100'u metal cikiyordu, cunku ayrik izgarada seviyeler zaten
aralikli. `band_structure` her k'da SIRALI dondugu icin b. sutun b. banttir
ve gercek aralik ancak max(bant b-1) < min(bant b) ise vardir.

MUTLAK DEGERLER DAIRESELDIR. `fermi_level`in dondurdugu sayi, girilen
`onsite` enerjisinin geri okunmasidir (olculdu: yari dolu zincirde
E_F = -0.194800000, girilen eps ile farki TAM 0.000e+00). Bant genisligi de
t'nin geri okunmasidir. Bu modulden bir DOGRULAMA cikaracaksaniz,
cikaracaginiz sey `certified_conduction`in sayim hukmu ve `structure_ratio`nun 4z'sidir
-- degerler degil.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError
from qlab.units import HARTREE_TO_EV


def _denetle(energies: np.ndarray, n_electron_per_cell: int) -> np.ndarray:
    e = np.asarray(energies, dtype=float)
    if e.ndim != 2 or e.shape[0] < 1 or e.shape[1] < 1:
        raise InvalidInputError(f"energies (n_k, n_band) olmali: {e.shape}")
    if not np.all(np.isfinite(e)):
        raise InvalidInputError("bant enerjileri sonlu olmali")
    n_band = e.shape[1]
    if not (1 <= n_electron_per_cell <= 2 * n_band):
        raise InvalidInputError(
            f"elektron sayisi 1 ile {2 * n_band} arasinda olmali "
            f"(bant sayisi {n_band}, spin dejenereligi 2), "
            f"verilen: {n_electron_per_cell}"
        )
    return e


def fermi_level(energies: np.ndarray, n_electron_per_cell: int) -> float:
    """Doldurulan son durumun enerjisi (hartree). MUTLAK DEGERI DAIRESELDIR."""
    e = _denetle(energies, n_electron_per_cell)
    n_k = e.shape[0]
    # Spin dejenereligi 2: hucre basina n elektron, k basina n/2 durum doldurur.
    dolu_durum = int(round(n_electron_per_cell * n_k / 2))
    if dolu_durum < 1:
        raise InvalidInputError(
            f"elektron sayisi bu k-izgarasinda tek bir durumu bile doldurmuyor "
            f"(n_k={n_k}, elektron={n_electron_per_cell})"
        )
    sirali = np.sort(e.reshape(-1))
    return float(sirali[dolu_durum - 1])


def is_metal(energies: np.ndarray, n_electron_per_cell: int) -> bool:
    """Fermi seviyesi kismen dolu bir bandin ICINDE mi? BANT COZUMLU olcut.

    TEK sayim: yarim dolu bant vardir, dolayisiyla hukum bir SAYIMDIR --
    hicbir enerji karsilastirmasi yapilmaz. Tek istisna dejenere durumdur:
    genisligi SIFIR bir bant (t = 0) yarim dolu olsa bile iletmez, ve bu
    sessizce metal sayilmaz.

    CIFT sayim: ilk n/2 bant dolu; yalitkan olmak icin dolu bantlarin en
    ustuyle bos bantlarin en alti arasinda GERCEK bir aralik gerekir.
    Ortusme varsa metal (olculdu, on olcum: 1B %67, 2B %34, 3B %20).

    BU IZGARANIN KENDI HUKMUDUR, KANITLI DEGILDIR (E13, 04.10.2026): Dirac
    noktasi izgarada yoksa sahte aralik okur, varsa dokunan bantlarin
    yuvarlamayla ayrilmasini (+5e-17) aralik sayar -- grafen her k'da
    "yalitkan" cikiyordu. Panel `certified_conduction`'i basar.
    """
    e = _denetle(energies, n_electron_per_cell)
    n_band = e.shape[1]

    if n_electron_per_cell % 2 == 1:
        yarim_dolu = n_electron_per_cell // 2
        genislik = float(e[:, yarim_dolu].max() - e[:, yarim_dolu].min())
        return bool(genislik > 0.0)

    dolu_bant = n_electron_per_cell // 2
    if dolu_bant >= n_band:
        return False  # tum bantlar dolu
    return bool(float(e[:, dolu_bant - 1].max()) >= float(e[:, dolu_bant].min()))


# E13 (04.10.2026): izgara ozdegerlerinin yuvarlama payi
#     1e3 * eps * (|E|max + H olcegi),
# H olcegi = `hamiltonian_rounding_scale` = max_i (|eps_i| + sum |t| (1 + 2 pi
# ||cell||_1)). OLCULDU: K noktasi izgaradayken grafenin dokunan bantlari
# 5e-17 ... 3.4e-16 hartree ayrik cikiyor (eps*|E|max'in en cok ~5 kati).
# Ilk surumde pay YALNIZ |E|max'a bagliydi ve dogrulama turu kamu yolundan
# iki SAHTE TANIK uretti: birbirini goturen parcali hoplama (1.23 + 1.6 -
# 2.83 eV) duz banda 7.6e-16 eV genislik verdi, |R| = 1000 hucrede faz
# yuvarlamasi 1.3e-12 eV -- ikisi de "metal (tanikli)". Ozdegerin
# kosegenlestirme hatasi |E|max ile, KURULUS hatasi sum |t| ve |k . R| ile
# olceklenir; pay ikisini de kapsar.
_YUVARLAMA_KATSAYISI = 1e3


@dataclass(frozen=True)
class ConductionVerdict:
    """Izgaradan KANITLI iletkenlik hukmu: "metal", "yalıtkan" ya da "belirsiz".

    `bound` B1'dir (`grid_lipschitz_bound`): her sirali ozdeger izgara ile
    surekli k arasinda en cok bu kadar oynar. Enerjiler HARTREE.
    """

    verdict: str
    reason: str
    bound: float
    rounding: float
    grid_gap: float | None = None
    grid_width: float | None = None

    def describe(self) -> str:
        """Panel satiri, eV."""
        ev = HARTREE_TO_EV
        if self.reason == "ortusme":
            return (
                f"metal — ızgarada bant örtüşmesi {-self.grid_gap * ev:.3e} eV "
                "(tanıklı)"
            )
        if self.reason == "tek_sayim":
            return (
                f"metal — tek sayım, yarı dolu bant ızgarada "
                f"{self.grid_width * ev:.3e} eV geniş (tanıklı)"
            )
        if self.reason == "tum_dolu":
            return "yalıtkan — bütün bantlar dolu"
        if self.reason == "duz":
            return "yalıtkan — H(k) k'dan bağımsız (B1 = 0), yarı dolu bant düz"
        if self.reason == "aralik":
            return (
                f"yalıtkan — ızgara aralığı {self.grid_gap * ev:.3e} eV > "
                f"2·B1 = {2 * self.bound * ev:.3e} eV"
            )
        if self.reason == "tek_duz_gorunuyor":
            return (
                f"belirsiz — yarı dolu bant ızgarada düz görünüyor "
                f"({self.grid_width * ev:.3e} eV) ama gerçek genişlik "
                f"2·B1 = {2 * self.bound * ev:.3e} eV'ye kadar olabilir"
            )
        return (
            f"belirsiz — ızgara aralığı {self.grid_gap * ev:.3e} eV; gerçek "
            f"aralık [{(self.grid_gap - 2 * self.bound) * ev:.3e}, "
            f"{self.grid_gap * ev:.3e}] eV içinde (kanıtlı çözünürlük "
            f"±{self.bound * ev:.3e} eV): metal, yarı metal ya da dar aralıklı "
            "yalıtkan olabilir"
        )


def certified_conduction(
    energies: np.ndarray, n_electron_per_cell: int, bound: float, h_scale: float
) -> ConductionVerdict:
    """`is_metal`in KANITLI karsiligi (E13, 04.10.2026). Panel bunu basar.

    `is_metal` izgaranin kendi hukmudur ve iki yerde YANLIS yalitkan diyordu
    (olculdu, kamu yolundan): (1) Dirac/kesisim noktasi izgarada yoksa sahte
    aralik -- grafen k_points 4'te 0.198 hartree; (2) izgarada olsa bile
    dokunan bantlar yuvarlamayla ayrik cikiyor (+5e-17) ve `>=` onu aralik
    sayiyor. Ikisinde de panel "yalıtkan ✓" basiyordu.

    Kural (B1 = `bound`, H olcegi = `h_scale` -- ikisi de modelden,
    `grid_lipschitz_bound` ve `hamiltonian_rounding_scale`; izgara
    ozdegerleri yuvarlama payi icinde GERCEK ozdegerlerdir):
    - metal: izgarada TANIK var -- cift sayimda dolu bandin tepesi bos bandin
      dibinin uzerinde (ortusme > yuvarlama), tek sayimda yari dolu bant
      izgarada yuvarlamadan genis. Pay H'nin KURULUS ve kosegenlestirme
      hatasini kapsadigi surece tanik yanlis olamaz.
    - yalıtkan: cift sayimda izgara araligi > 2*B1 + yuvarlama (gercek aralik
      >= izgara araligi - 2*B1 > 0); butun bantlar doluysa; ya da B1 = 0 iken
      (H k'dan bagimsiz) yari dolu bant duz.
    - belirsiz: geri kalan her sey. Dokunan bantlar (yari metal) HICBIR n'de
      metal kanitlanamaz -- surekli bir sinir sifir araligi kanitlayamaz.
    """
    e = _denetle(energies, n_electron_per_cell)
    for ad, deger in (("sinir", bound), ("H olcegi", h_scale)):
        if not (np.isfinite(deger) and deger >= 0.0):
            raise InvalidInputError(f"{ad} sonlu ve negatif olmayan olmali: {deger}")
    yuvarlama = (
        _YUVARLAMA_KATSAYISI
        * np.finfo(float).eps
        * (float(np.abs(e).max()) + float(h_scale))
    )
    ortak = {"bound": float(bound), "rounding": yuvarlama}

    if n_electron_per_cell % 2 == 1:
        bant = e[:, n_electron_per_cell // 2]
        genislik = float(bant.max() - bant.min())
        if genislik > yuvarlama:
            return ConductionVerdict("metal", "tek_sayim", grid_width=genislik, **ortak)
        if bound == 0.0:
            return ConductionVerdict("yalıtkan", "duz", grid_width=genislik, **ortak)
        return ConductionVerdict(
            "belirsiz", "tek_duz_gorunuyor", grid_width=genislik, **ortak
        )

    dolu_bant = n_electron_per_cell // 2
    if dolu_bant >= e.shape[1]:
        return ConductionVerdict("yalıtkan", "tum_dolu", **ortak)
    aralik = float(e[:, dolu_bant].min()) - float(e[:, dolu_bant - 1].max())
    if aralik < -yuvarlama:
        return ConductionVerdict("metal", "ortusme", grid_gap=aralik, **ortak)
    if aralik > 2.0 * bound + yuvarlama:
        return ConductionVerdict("yalıtkan", "aralik", grid_gap=aralik, **ortak)
    return ConductionVerdict("belirsiz", "cozunurluk_alti", grid_gap=aralik, **ortak)


def density_of_states(
    energies: np.ndarray, n_bin: int
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram DOS. Integrali 1'e normalize edilir.

    Normalizasyon secimi bilinclidir: bant basina degil TOPLAM olasilik
    yogunlugu dondurulur, boylece farkli bant sayilarindaki modeller
    karsilastirilabilir. `dos.sum() * bin_genisligi == 1` testle kilitlidir.
    """
    e = np.asarray(energies, dtype=float)
    if e.ndim != 2:
        raise InvalidInputError(f"energies (n_k, n_band) olmali: {e.shape}")
    if n_bin < 2:
        raise InvalidInputError(f"n_bin en az 2 olmali, verilen: {n_bin}")
    sayim, kenarlar = np.histogram(e.reshape(-1), bins=n_bin, density=True)
    merkezler = 0.5 * (kenarlar[:-1] + kenarlar[1:])
    return merkezler, sayim


def structure_ratio(energies: np.ndarray) -> float:
    """W^2 / mu2 -- t ve eps CEBIRSEL OLARAK sadelesir, geriye 4z kalir.

    OLCULDU: 1B 8.000000000 (bagil fark 2.22e-16), 2B kare 16.000000000
    (0.00e+00), 3B kubik 24.000000000 (0.00e+00); uc farkli (t, eps) setiyle
    -- biri eps = -400 gibi asiri -- ayni sonuc. Kucuk izgarada zaten tam
    cikiyor (1B n=50, 2B n=20, 3B n=8), yani bu bir yakinsama meselesi
    degil cebirsel bir ozdesliktir.

    UYGULAMA ALANI DAR ve bu bir kusur degil beyandir: oran ETKIN
    koordinasyonu okur. SSH'de t1 = -0.10 sabitken olculdu --
      t2 = -0.10 (tekbicim)   -> 8.000000  (z = 2)
      t2 = -0.05              -> 7.200000  (%-10.00)
      t2 = -0.02              -> 5.538462  (%-30.77)
      t2 =  0.00 (izole dimer) -> 4.000000 (%-50.00, yani z = 1)
    Ikinci komsu ise oteki yone tasiyor: 1B'de t2 = t1 iken 9.765604
    (%+22.07). Gercek bir malzemede ikisi de vardir; orada "4z okudum"
    demek bir MODEL VARSAYIMINI geri okumaktir.
    """
    e = np.asarray(energies, dtype=float)
    if e.ndim != 2:
        raise InvalidInputError(f"energies (n_k, n_band) olmali: {e.shape}")
    duz = e.reshape(-1)
    W = float(duz.max() - duz.min())
    mu2 = float(np.mean((duz - duz.mean()) ** 2))
    # Duzluk olcutu W uzerinden, mu2 uzerinden DEGIL. Olculdu: sabit bir
    # dizide bile mu2 kayan nokta yuzunden tam sifir olmayabiliyor --
    # -0.3'lerden olusan dizide 3.081e-33 cikiyor (0.1 ve 1/3'te 0.0).
    # W = max - min ise sabit dizide TAM 0'dir, yani kesin olcut odur.
    if W <= 0.0 or mu2 <= 0.0:
        raise InvalidInputError(
            "bant duz (genislik sifir) -- yapi orani tanimsiz"
        )
    return W * W / mu2
