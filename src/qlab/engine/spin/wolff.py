"""Wolff kume algoritmasi -- YALNIZ Ising, YALNIZ h = 0, YALNIZ J > 0.

Metropolis tek bir spini oynatip kabul/red karari verir; Wolff hicbir
kabul/red karari VERMEZ. Bunun yerine ayni isaretli komsular arasinda
`p = 1 - exp(-2 J / T)` olasilikla bag kurar, tohum sitenin bagli kumesini
buyutur ve kumenin TAMAMINI kosulsuz cevirir. Iki kod yolunun ortak hicbir
karar noktasi yoktur; `tests/spin/test_wolff.py`deki capraz kontrolun
DAIRESEL OLMAMASI buna dayanir.

`p`DEKI 2 CARPANI: bag, iki ucunun da ayni kumede olma sansini tasir; dogru
FK (Fortuin-Kasteleyn) olcusu `1 - exp(-2 J / T)`dir. `1` yazmak dengeyi
SESSIZCE bozar -- kod calisir, yakinsar, makul gorunur ve YANLIS sicakligi
ornekler (etkin olarak 2T). Mutasyon sinavinda bu mutant tam sayim cipasi
tarafindan yakalanir.

BIR "ADIM" NE DEMEK -- IKI ALGORITMA AYNI SEYI SAYMAZ:
Wolff'un bir adimi BIR KUME CEVIRMESIDIR; Metropolis'in bir sweep'i L^2 SITE
GUNCELLEMESIDIR. `n_sweep` iki modulde de "olcum sayisi" anlamina gelir ama
adimin maliyeti ve tasidigi karar sayisi FARKLIDIR. Bu yuzden capraz kontrol
iki kosumu esit adimda degil, HER IKISI DE DENGEYE OTURMUS uzunlukta
karsilastirir; uzunluklar ve tohumlar test dokstringinde acikca yazilidir.
Olculen kume boyutu (L = 16, tohum 20260823, 200 burn + 200 olcum):

    T             p            <|C|> / L^2
    0.500000    0.981684        1.000000
    1.800000    0.670807        0.915605
    2.269185    0.585786        0.481875     (T_c)
    2.800000    0.510458        0.064551
    10.000000   0.181269        0.006211
    1e8         2.0e-08         0.003906     (= 1/256, TEK SITE)

Yani T -> 0'da bir adim BUTUN kafesi cevirir, T -> sonsuz'da tek siteyi.
"Adim" kelimesinin iki modulde ayni sey olmadigi bu tabloda gorunur. Gecis
KESKINDIR: T_c'nin %23 ustunde (2.8) kume kesri 7.46 kat iniyor, oysa `p`
yalnizca %12.9 dusuyor.

TEK L SERBEST -- METROPOLIS'TE DEGILDI. `metropolis.py` cift L zorunlu
kilar cunku checkerboard periyodik kare kafesin iki-parcali olmasini
gerektirir. Wolff'un boyle bir sarti YOKTUR; kume kurulumu orgunun
iki-parcali olup olmadigini bilmez. Olculdu (L = 3, 512 durumun TAM SAYIMI,
16 zincir): T = 2.0'da tam 0.925984778 / Wolff 0.925850694 (0.08 x SE),
T = T_c'de tam 0.871071935 / Wolff 0.869722222 (0.78 x SE) -- yani Wolff
Metropolis'in HIC KOSAMADIGI bir kafeste dogru dagilimi ornekliyor.

UC SINIR, UCU DE SESSIZ OLSAYDI TEHLIKELI OLURDU:

1. `h != 0` -> `UnsupportedSystemError`. Kume cevirme alan terimini ihlal
   eder: cevrilen kumenin alan enerjisi degisir ama kabul olasiligi 1
   kalir. Sonuc makul gorunur, yakinsar ve YANLIStir.
2. `n_component != 1` -> `UnsupportedSystemError`. Kume kurulumu "ayni
   isaret" kavramina dayanir; surekli spin uzayinda bunun karsiligi
   yansitma duzlemidir (Wolff'un genel hali) ve bu depoda yazilmadi.
   Sinir SESSIZ OLSAYDI istisna DEGIL, sessiz bir kayip olurdu: govde
   `state.spins[..., 0]` aliyor, yani ILK bileseni Ising sanip kalan
   bilesenleri ATIYOR ve `(L, L, 1)` bir durum donduruyor -- n_component
   3 iken 1 olarak. Olculdu: `SpinState.uniform(L=4, n_component=3)`un ilk
   bileseni tumuyle 0.0'dir (`uniform` SON bileseni 1 yapar), dolayisiyla
   kume kurulumu "hepsi ayni isaret" gorur, butun kafesi yutar ve `-0.0`
   ile doldurur; miknatislanma serisi bastan sona 0.0 cikar ve hicbir uyari
   verilmez.
3. `J <= 0` -> `InvalidInputError`. BU SINIR PLANDA YOKTU ve en sinsisi
   budur: J < 0 icin `p = 1 - exp(-2J/T)` NEGATIF cikar (J = -1, T = 1.5
   icin -2.793668), hicbir bag kurulmaz, her adimda rastgele TEK bir site
   kosulsuz cevrilir -- yani algoritma SONSUZ SICAKLIGI ornekler. Olculdu
   (koruma KALDIRILARAK, L = 4, 16 zincir, n_burn = 300, n_sweep = 800):
   korumasiz Wolff T = 1.5'te 0.201171875, T = 2.0'da 0.201171875 --
   SICAKLIGA GORE HIC DEGISMIYOR, cunku sicaklik yalnizca `p`ye giriyor ve
   `p` her negatif degerde ayni sonucu veriyor. Tam sayimla karsilastirma:
   T = sonsuz'un TAM degeri kapali formdan 6435/32768 = 0.196380615234375
   (tam sayim T = 1e8'de 0.196380619 verir; sapma 1/T ile olcekleniyor,
   T = 1e6/1e7/1e8/1e9 icin +4.189e-07/-08/-09/-10). J = -1 ve T = 1.5'in
   gercek degeri 0.009542350.
   Yani sonuc sonsuz sicakligindir, hata 20 KAT ve hicbir
   istisna atilmaz.

`energy_series` ve `acceptance_rate` `None` DONER, sifir degil. Gerekcesi
`metropolis.py`deki `SamplingResult` dokstringinde yazilidir: kume cevirmede
enerji adim basina ucuza guncellenemez (olcum basina O(L^2) ekler) ve kabul
orani anlamsizdir -- kume HER ZAMAN cevrilir. `None` "OLCULMEDI" demektir ve
tuketici tarafi onu sifir SANMAMALIDIR.

OLCULEREK CURUTULEN BIR "IYILESTIRME": kume buyutmenin ic dongusu once dort
rastgele sayiyi TEK numpy cagrisiyla (`rng.random(4)`) uretiyordu; gerekce
"dort skaler cagri daha pahalidir" varsayimiydi. Olculdu (L = 16, T = 1.8,
300 kume cevirmesi, uc tekrarin en iyisi; iki ayri turda): toplu cagri
0.0769 / 0.0772 s, dort skaler cagri 0.0629 / 0.0623 s -- yani toplu cagri
%22-24 DAHA YAVAS (aradaki oynama zamanlama gurultusudur, isaret iki turda
da ayni). Sebep aciktir: skaler surum kisa devre yapar (komsu zaten kumedeyse veya isareti farkliysa sayi
HIC uretilmez), toplu surum ise her sitede dort sayiyi kosulsuz uretir. Kod
skaler surume donduruldu; varsayim olculmeden birakilsaydi hem daha yavas
hem daha karmasik kalacakti.

PLANDAN SAPMA (bilincli): plan taslagi `field=0.0` VARSAYILANI veriyordu ve
`Interfaces` satirinda `field`i hic saymiyordu. Burada `field` ZORUNLU
anahtar argumandir. Gerekce: varsayilan 0.0, h = 0.3 ile calisan bir
cagiranin `field`i gecirmeyi unutmasi halinde sinirin HIC calismamasi
demekti -- plan'in kendi ifadesiyle "en tehlikeli turden" sessiz hata. Zorunlu
argumanla o delik kapaniyor ve imza `run_metropolis` ile ayni sekle geliyor.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.spin.lattice import SpinState
from qlab.engine.spin.metropolis import SamplingResult
from qlab.errors import InvalidInputError, UnsupportedSystemError

_YONLER = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dogrula(
    state: SpinState,
    *,
    temperature: float,
    coupling: float,
    field: float,
    n_sweep: int,
    n_burn: int,
) -> None:
    """Uc kapsam siniri + iki sayisal sinir. Sirasi onemli degil, hepsi kapanir."""
    if field != 0.0:
        raise UnsupportedSystemError(
            f"Wolff dis alanla calismaz (alan h = {field}): kume cevirme alan "
            "terimini ihlal eder ve YANLIS dagilim ornekler -- sonuc makul "
            "gorunur ve yakinsar. h != 0 icin Metropolis kullanin."
        )
    if state.n_component != 1:
        raise UnsupportedSystemError(
            f"Wolff bu depoda yalniz Ising icin tanimli "
            f"(n_component = {state.n_component}). Kume kurulumu 'ayni isaret' "
            "kavramina dayanir; surekli spin uzayinda karsiligi yansitma "
            "duzlemidir ve yazilmadi."
        )
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise InvalidInputError(
            f"sicaklik pozitif ve sonlu olmali, verilen: {temperature}"
        )
    if not np.isfinite(coupling) or coupling <= 0.0:
        raise InvalidInputError(
            f"Wolff FERROMANYETIK eslesme gerektirir (J > 0), verilen: {coupling}. "
            "J <= 0'da bag olasiligi p = 1 - exp(-2J/T) pozitif degildir; hicbir "
            "bag kurulmaz ve algoritma her adimda tek bir siteyi kosulsuz cevirir, "
            "yani SONSUZ SICAKLIGI ornekler (L=4, J=-1 icin olculen 0.201172, "
            "T = sonsuz'un tam degeri 0.196381, T = 1.5'in gercek degeri 0.009542)."
        )
    if n_sweep < 1:
        raise InvalidInputError(f"n_sweep en az 1 olmali, verilen: {n_sweep}")
    if n_burn < 0:
        raise InvalidInputError(f"n_burn negatif olamaz, verilen: {n_burn}")


def _kume_buyut(
    spins: np.ndarray, i0: int, j0: int, p_bond: float, rng: np.random.Generator
) -> np.ndarray:
    """Tohum siteden buyuyen kumenin `(L, L)` boolean maskesi.

    Kume YALNIZ tohumla AYNI ISARETLI komsulara buyur. Bu kosul kalkarsa
    kume zit isaretli bolgelerin USTUNDEN atlar ve tohumun bagli bileseni
    disindaki siteleri de cevirir; olculdu (L = 6, satirlari donusumlu
    +-1 kafes, T = 0.05, tek adim): dogru kod `|m| = 0.333333333` verir --
    yani TEK BIR SATIR cevrilir -- isaret kontrolu kalkarsa `|m| = 1.0`,
    cunku o zaman butun `+1` satirlari birden cevrilir.

    Girdiyi DEGISTIRMEZ; cevirme isini cagiran yapar.
    """
    isaret = spins[i0, j0]
    L = spins.shape[0]
    kumede = np.zeros((L, L), dtype=bool)
    kumede[i0, j0] = True
    yigin = [(i0, j0)]
    while yigin:
        i, j = yigin.pop()
        for di, dj in _YONLER:
            ni = (i + di) % L
            nj = (j + dj) % L
            if (
                not kumede[ni, nj]
                and spins[ni, nj] == isaret
                and rng.random() < p_bond
            ):
                kumede[ni, nj] = True
                yigin.append((ni, nj))
    return kumede


def run_wolff(
    state: SpinState,
    *,
    temperature: float,
    coupling: float,
    field: float,
    rng: np.random.Generator,
    n_sweep: int,
    n_burn: int,
) -> SamplingResult:
    """`n_burn` kume cevirmesi at, sonraki `n_sweep` cevirmeyi olc.

    Girdiyi DEGISTIRMEZ (`state.spins` kopyalanir).

    Olcum cevirmeden SONRA alinir -- `run_metropolis` ile ayni sira, ki iki
    seri ayni anlamda karsilastirilabilsin. `magnetisation_series` her olcumde
    `|ortalama spin|`dir; n_component = 1 oldugu icin bu, `run_metropolis`in
    kullandigi bilesen vektorunun normuyla BIREBIR ayni sayidir.
    """
    _dogrula(
        state,
        temperature=temperature,
        coupling=coupling,
        field=field,
        n_sweep=n_sweep,
        n_burn=n_burn,
    )

    spins = state.spins[..., 0].copy()
    L = spins.shape[0]
    p_bond = 1.0 - np.exp(-2.0 * coupling / temperature)
    m_serisi: list[float] = []

    for adim in range(n_burn + n_sweep):
        i0, j0 = (int(v) for v in rng.integers(0, L, 2))
        isaret = spins[i0, j0]
        kumede = _kume_buyut(spins, i0, j0, p_bond, rng)
        spins[kumede] = -isaret
        if adim >= n_burn:
            m_serisi.append(abs(float(spins.mean())))

    return SamplingResult(
        magnetisation_series=np.array(m_serisi),
        final_state=SpinState(spins[..., None]),
        n_sweep=n_sweep,
        algorithm="wolff",
        energy_series=None,
        acceptance_rate=None,
    )
