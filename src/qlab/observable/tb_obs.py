"""Tight-binding gozlemlenebilirleri -- doluluk, Fermi seviyesi, DOS, yapi orani.

MODULUN TEK PARAMETRE-BAGIMSIZ FIZIK HUKMU: hucre basina TEK sayida elektron
her zaman metal verir, cunku tek sayi bir bandi ancak YARI doldurabilir. Bu
bir hesap degil bir SAYIMDIR. Olculdu: 100 rastgele model x 3 boyut =
300/300 metal.

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
cikaracaginiz sey `is_metal`in sayim hukmu ve `structure_ratio`nun 4z'sidir
-- degerler degil.
"""

from __future__ import annotations

import numpy as np

from qlab.errors import InvalidInputError


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
