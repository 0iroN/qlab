"""Elektromanyetik gozlemlenebilirler -- ve OLCUMUN KENDI DURUSTLUK MAKINESI.

Bu dosyanin ilk parcasi `convergence_ladder`dir: Modul 5'in her FIZIK
cipasi degeriyle birlikte bir YAKINSAMA KAYDI dondurmek zorundadir (spec
4.3, pazarlik edilmez). Bir FDTD sayisi `dx` kuculurken oturmadiysa
yakinsamamistir ve raporlanan sey bir tahmin degil bir ARA DURAKTIR.

Modul 4'un `binning_error`i ile AYNI AILEDENDIR: orada plato tespiti,
burada izgara yakinsamasi. Ikisinin de sozu ayni -- olculen sayinin yaninda
o sayinin NE KADAR guvenilir oldugu da doner, ve tahminci kirildiginda
"kirildi" der, sayiyi kirik haliyle basmaz.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from qlab.engine.em.boundary import PML
from qlab.engine.em.medium import Medium
from qlab.engine.em.oblique import POLARIZATIONS, run_oblique_1d
from qlab.engine.em.source import gauss_cos
from qlab.errors import InvalidInputError

#: Hukmun UC degeri. `YARGILANAMAZ` Modul 4'un `correlation_length -> None`
#: deseniyle AYNI STATUDEDIR: yargilayamadigini yargilamis gibi gosterme.
VERDICTS: tuple[str, ...] = ("OTURDU", "OTURMADI", "YARGILANAMAZ")

#: Merdivenin en kucuk derinligi. DORT, bir tercih degil bir SAYIMDIR --
#: bkz. `convergence_ladder` dokstringi.
MIN_DEPTH: int = 4

#: Karar siniri. TOLERANS DEGIL: rakip mertebeler bu semada AYRIKTIR
#: (Yee 2 - merdivenlenmis sinir 1 - olceklenmeyen dugme 0), dolayisiyla
#: 0.5 iki tamsayi arasindaki ORTA NOKTADIR. MODELLEME SECIMIDIR ve
#: `grid.courant_limit`in "S en hizli dalga hiziyla tanimlidir" beyaniyla
#: ayni statudedir; panelde oyle yazilir.
DEFAULT_KAPPA: float = 0.5

#: `p` icin ust imkan kapisi. 8'in ustu bu semada bir mertebe degil, farkin
#: gurultuye ya da makine tabanina carptiginin isaretidir.
_P_MAX: float = 8.0

#: Izgara araliklarinin `ratio` ile tutarli olmasinda kabul edilen bagil
#: sapma. Tolerans DEGIL, ikili gosterimin artigi: 1/200, 1/400, 1/800
#: gibi araliklarda olculen en buyuk sapma 0.0e+00 (tam), guvenlik payiyla.
_RATIO_RTOL: float = 1e-9


@dataclass(frozen=True)
class ConvergenceLadder:
    """Izgara merdiveninin TAM kaydi.

    `values` / `spacings`: merdivenin kendisi, KABADAN INCEYE. Cagiran
    hukmu kendi gozuyle denetleyebilsin diye tam olarak tasinir --
    `binning_error`in `block_sizes`/`errors` egrisini tasimasiyla ayni
    gerekce.

    `orders`: basamak basina ham mertebe `p_i`. Bos tuple, merdivenin
    aritmetigin tanimli oldugu noktaya bile ulasamadigi anlamina gelir
    (derinlik yetmedi ya da ardisik farklar isaret degistirdi).

    `p_inf`: Richardson ile `dx -> 0`a otelenmis mertebe. HUKUM DEGILDIR:
    `verdict` "YARGILANAMAZ" iken de hesaplanmis olabilir ve o durumda
    "hesapladim ama arkasinda duramam" demektir.

    `uncertainty`: `p_inf`in belirsizligi -- uc bagimsiz kaynagin EN
    BUYUGU (bkz. `convergence_ladder`). `p_inf` None ise None.

    `verdict`: UC DEGERLI. "OTURDU" ise HANGI tamsayiya oturdugu
    `settled_order`dan okunur -- soru "2'ye oturdu mu" DEGIL, "hangi
    tamsayiya oturdu"dur (spec 4.3'un dorduncu duzeltmesi). Merdivenlenmis
    dairesel sinirin mutlak `j` degerleri icin dogru cevap 2 degil 1'dir.
    """

    values: tuple[float, ...]
    spacings: tuple[float, ...]
    ratio: float
    orders: tuple[float, ...]
    p_inf: float | None
    uncertainty: float | None
    verdict: str

    @property
    def settled_order(self) -> int | None:
        """Oturulan tamsayi mertebe; oturmadiysa None.

        None SIFIR DEGILDIR ve 2 de degildir: hukum verilmediginde burada
        bir sayi olmasi, verilmemis bir hukmu verilmis gibi gosterirdi.
        Ayni gerekceyle Modul 4'un `correlation_length`i kirildiginda
        `None` doner.
        """
        if self.verdict != "OTURDU" or self.p_inf is None:
            return None
        return int(round(self.p_inf))

    def as_lines(self) -> list[str]:
        """Panelde basilacak tek satirlik beyan parcalari."""
        satirlar = [
            "merdiven    : "
            + " -> ".join(f"{h:g}" for h in self.spacings)
            + f"  (r = {self.ratio:g}, derinlik {len(self.values)})",
        ]
        if self.p_inf is None:
            satirlar.append("mertebe     : okunamadi")
        else:
            u = "?" if self.uncertainty is None else f"{self.uncertainty:.3f}"
            satirlar.append(f"mertebe     : p_inf = {self.p_inf:.4f} +- {u}")
        if self.verdict == "OTURDU":
            satirlar.append(f"hukum       : OTURDU({self.settled_order})")
        else:
            satirlar.append(f"hukum       : {self.verdict}")
        return satirlar


def _merdiven(values, spacings, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Girdiyi dogrular ve (degerler, araliklar) ikilisini dondurur."""
    v = np.asarray(values, dtype=float)
    h = np.asarray(spacings, dtype=float)
    if v.ndim != 1 or h.ndim != 1:
        raise InvalidInputError(
            f"values ve spacings tek boyutlu olmali, verilen: {v.shape} ve {h.shape}"
        )
    if v.size != h.size:
        raise InvalidInputError(
            f"her degerin bir izgara araligi olmali: {v.size} deger, "
            f"{h.size} aralik"
        )
    if v.size == 0:
        raise InvalidInputError("merdiven bos")
    if not np.all(np.isfinite(v)):
        raise InvalidInputError(
            f"values sonlu olmali; {int(np.count_nonzero(~np.isfinite(v)))} "
            "sonlu olmayan deger var (patlamis bir FDTD kosumu sessizce "
            "mertebe URETEMEZ)"
        )
    if not np.all(np.isfinite(h)) or not np.all(h > 0.0):
        raise InvalidInputError(
            f"spacings pozitif ve sonlu olmali, en kucuk deger: "
            f"{float(np.min(h)):.6g}"
        )
    if v.size > 1:
        if not np.all(np.diff(h) < 0.0):
            raise InvalidInputError(
                "merdiven KABADAN INCEYE siralanmali (spacings kesin azalan); "
                f"verilen: {h.tolist()}. Ters sirali bir merdiven `p`nin "
                "isaretini cevirir ve imkan kapilarindan SESSIZCE gecerdi."
            )
        olculen = h[:-1] / h[1:]
        if not np.all(np.abs(olculen - ratio) <= _RATIO_RTOL * ratio):
            raise InvalidInputError(
                f"spacings orani `ratio` ile tutarli degil: olculen "
                f"{olculen.tolist()}, beklenen {ratio}. `p` tam olarak "
                "log(d_i/d_{i+1})/log(ratio) ile okunur; oran uymuyorsa "
                "cikan sayi bir mertebe DEGILDIR."
            )
    return v, h


def _gurultu(noise, n: int) -> np.ndarray:
    """`noise`u basamak basina BAGIL gurultu dizisine cevirir.

    `spacings` bu hesaba GIRMEZ ve girmemelidir: gurultunun `p`ye tasinmasi
    yalniz `ratio` uzerinden olur (asagidaki Jacobi zinciri), aralik
    degerlerinin kendisi uzerinden degil. Araliklar ile `ratio` arasindaki
    tutarlilik zaten `_merdiven`de kapida sinaniyor.
    """
    g = np.asarray(noise, dtype=float)
    if g.ndim == 0:
        g = np.full(n, float(g))
    if g.shape != (n,):
        raise InvalidInputError(
            f"noise ya skaler ya da {n} uzunlukta olmali, verilen sekil: {g.shape}"
        )
    if not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise InvalidInputError(
            f"noise sonlu ve negatif olmayan olmali, verilen: {g.tolist()}"
        )
    return g


def _gurultu_belirsizligi(
    v: np.ndarray, d: np.ndarray, g: np.ndarray, ratio: float
) -> float:
    """`p_inf`in olculen pencere gurultusunden gelen belirsizligi.

    DOGRUSAL YAYILIM, Monte Carlo DEGIL: bu modulde RNG yoktur (FDTD
    deterministtir, tekrarlanabilirligin karsiligi PENCERE BEYANIDIR) ve
    ayni girdi ayni belirsizligi vermek zorundadir.

    Zincir uc halkadir ve hepsi kapali formdur:
        d_i     = v_{i+1} - v_i
        p_i     = ( log|d_i| - log|d_{i+1}| ) / log r
        p_inf_j = ( r p_{j+1} - p_j ) / (r - 1),   P = ort_j p_inf_j
    Jacobi'ler zincirlenip `sigma_i = g_i |v_i|` ile toplanir.

    BAGIMSIZ varsayimi burada BILINCLI OLARAK KORUNUYORDUR ve bu, `eps`i
    bir ALT SINIR yapan seydir: tam ortak kipli bir bilesen zaten `p`yi
    oynatamaz (carpimsal ortak kipte sadelesir, toplamsalda farkta duser),
    dolayisiyla onu bagimsiz saymak yalniz belirsizligi BUYUTUR -- yani
    hukum tarafinda TEMKINLI olur, gevsek degil.
    """
    L = np.log(ratio)
    n_p = d.size - 1
    m = n_p - 1
    kat_p = np.zeros(n_p)
    for j in range(m):
        kat_p[j] += -1.0 / (ratio - 1.0) / m
        kat_p[j + 1] += ratio / (ratio - 1.0) / m
    kat_d = np.zeros(d.size)
    for i in range(n_p):
        kat_d[i] += kat_p[i] / (L * d[i])
        kat_d[i + 1] -= kat_p[i] / (L * d[i + 1])
    kat_v = np.zeros(v.size)
    kat_v[:-1] -= kat_d
    kat_v[1:] += kat_d
    return float(np.sqrt(np.sum((kat_v * g * np.abs(v)) ** 2)))


def convergence_ladder(
    values,
    spacings,
    *,
    noise,
    ratio: float = 2.0,
    kappa: float = DEFAULT_KAPPA,
) -> ConvergenceLadder:
    """Izgara merdiveninden mertebe ve UC DEGERLI hukum.

    SINIF: OLCUT. Kendisi ne sema cipasi ne fizik cipasidir; cipalarin
    HUKMUNU veren alettir, ve bu yuzden kendi karnesini ve kendi TAUTOLOJI
    KAPISINI tasimak zorundadir.

    HESAP
        d_i     = v_{i+1} - v_i
        p_i     = log(d_i / d_{i+1}) / log r
        p_inf_j = (r p_{j+1} - p_j) / (r - 1)      P = ort_j p_inf_j
        u       = max( |P - p_son| , std(p_inf_j) , u_gurultu(noise) )
    HUKUM (kappa):
        u >= kappa                     -> YARGILANAMAZ
        |P - round(P)| + u < kappa     -> OTURDU(round(P))
        aksi                           -> OTURMADI

    DERINLIK EN AZ DORT -- ve bu bir SAYIM KIMLIGIDIR, tercih degil:
    n deger -> n-1 fark -> n-2 mertebe -> n-3 Richardson noktasi. Yani
    n = 2'de mertebe YOK, n = 3'te mertebe VAR ama belirsizligi olusturacak
    ikinci bir Richardson noktasi YOK. Dort, `u`nun olculebildigi EN KUCUK
    derinliktir. Korpusta dogrulandi: derinlik 2'de 426/426, derinlik 3'te
    426/426 merdiven `p_inf = None, uncertainty = None` donuyor.

    `noise` (basamak basina BAGIL pencere gurultusu) ZORUNLU GIRDIDIR ve
    EK FDTD KOSUMU GEREKTIRMEZ: spektral cipalarda pencere dugmelerinin
    cogu (pad, FFT penceresi, kayit kirpma) ISLEM-SONRASIDIR ve tek
    kaydedilmis seriden uretilir. OLCULDU -- 426 gercek merdivende
    kappa = 0.5'te yanlis pozitif: `noise` ile %0.0 (0/426, Clopper-Pearson
    %95 ust siniri %0.9), `noise = 0` ile %16.4 (70/426). Bedeli iki dogru
    pozitifti (203 -> 201).

    ORTAK KIPLI GURULTU BU TAHMINCIYI OYNATAMAZ -- CEBIRSEL: `p` farklarin
    ORANINDAN okunur; carpimsal ortak kipte d_i -> d_i(1+delta) sadelesir,
    toplamsalda delta farkta duser. AYNI SIGMA ile olculdu (301 merdiven x
    40 deneme, ortak kip vs BASAMAGA OZGU):

        sigma    ortak kip     basamaga ozgu     oran
        1e-10    6.760e-10     1.097e-03         1.6e+06
        1e-08    7.189e-10     1.286e-01         1.8e+08
        1e-06    6.704e-10     1.238e+01         1.9e+10
        1e-02    6.816e-10     3.658e+01         5.4e+10

    delta ~ N(0, 0.3) gibi DEVASA bir ortak kipte bile carpimsalda
    5.231e-10, toplamsalda 8.947e-10 (15050 deneme). Yani `noise` bir ALT
    SINIRDIR -- yalniz pencereler ARASINDA degisen bileseni gorur -- ama
    gormedigi bilesen zaten mertebeyi oynatamaz.

    `kappa` BIR TOLERANS DEGIL, IKI TAMSAYI ARASINDAKI KARAR SINIRIDIR.
    Bu semada rakip mertebeler AYRIKTIR (Yee 2 - merdivenlenmis sinir 1 -
    olceklenmeyen cozunurluk dugmesi 0), dolayisiyla 0.5 orta noktadir. Bu
    bir MODELLEME SECIMIDIR ve `grid.courant_limit`in "S en hizli dalga
    hiziyla tanimlidir" beyaniyla ayni statudedir.

    VE KIRILGAN OLMADIGI OLCULDU (426 gercek merdiven, sentetik YOK; yer
    gercegi BILINEN surekli limitten okunur, tahminci limiti HIC gormez):

        kappa   dogru pozitif        yanlis pozitif
        0.10    %77.8 (186/239)      %0.0 (0/426)
        0.20    %80.3 (192/239)      %0.0
        0.30    %82.8 (198/239)      %0.0
        0.40    %84.1 (201/239)      %0.0
        0.50    %84.1 (201/239)      %0.0      <- SECILEN
        0.60    %84.1 (201/239)      %0.0
        0.70    %84.1 (201/239)      %0.2 (1/426)
        0.80    %85.4 (204/239)      %0.5 (2/426)
    0.40 - 0.60 arasinda DUZ BIR PLATO var. Dogru pozitif icin Clopper-
    Pearson %95 araligi [%78.8, %88.5].

    KORPUSUN PENCERESI: sevk edilen 1B Yee cekirdegi (`yee._adim`), PEC
    duvarli duran kip, derinlik 4, r = 2, N_ince <= 1601 (1B dispersiyon
    cipasinin ince ucu float tabanina carpiyor); m = 4/8/16, S = 0.3/0.5/
    0.7/0.9, kayit 400/600/800 adim, N_kaba = 101/201; alti aile
    (dispersiyon, etiket hatasi, olceklenmeyen dugme, sabit kayit,
    spektral okuma, spektral + sabit kayit). Uretilen 432 merdivenin 6'si
    FFT tepesi tanimsiz kaldigi icin ATILDI.

    AILE BAZINDA HUKUM (kappa = 0.5; gurultulu / gurultusuz):
        dispersiyon (gercek mertebe 2)     OTURDU(2) 72/72  /  ayni
        etiket hatasi (gercek mertebe 1)   OTURDU(1) 57/72  /  ayni
        olceklenmeyen dugme (mertebe 0)    YARGILANAMAZ 72/72 / ayni
        sabit kayit (uc terimli okuma)     OTURDU(2) 72/72  /  ayni
        spektral okuma (limit YANLIS)      YARGILANAMAZ 72/72 / OTURDU(2) 72
        spektral + sabit kayit             YARGILANAMAZ 66/66 / ayni
    Dorduncu satir bir DURUST OLUMSUZDUR: "kayit uzunlugu olceklenmiyor"
    bozmasi uc terimli okumada BIT BIT no-op'tur (olculdu, uc pencerede de
    `values` ve `noise` ayni), yani ayni sutunda dispersiyonla ozdes. Son iki
    satir ise olcutun ozudur: spektral okuma dogru limite DEGIL, m = 8,
    S = 0.5, N = 201..1601 penceresinde 3.9e-3 kadar yanlis bir sayiya
    2. mertebeden yakinsiyor. `noise` olmadan olcut
    "OTURDU(2)" diyor ve YANILIYOR; olculen pencere gurultusuyle
    "YARGILANAMAZ" diyor.

    TAUTOLOJI KAPISI (Plan H'nin dersi, BESINCI uygulama) -- ve PLANIN
    "%0.00" SAYISI YENIDEN URETILEMEDI. Fizigi cikarip boru hattini yerinde
    birakan girdilerde, 5 tohum x 2000 = 10000 deneme, h = 1/200/2^i,
    noise = 1e-6 (en elverissiz ayar):

        girdi                        OTURDU(2)   OTURDU(herhangi)
        saf gurultu                  %0.01 (1)   %0.17 (17)
        rastgele yuruyus             %0.09 (9)   %0.73 (73)
        monoton rastgele yuruyus     %0.33 (33)  %3.12 (312)

    Yani kapi %99.7'yi ceviriyor ama SIFIR DEGIL; plan/spec'in "%0,00"i
    2000 denemede orneklenmis bir sifirdir (bu oranlarda 2000 denemede
    sifir gormek beklenen bir sonuctur).

    BIRIM DENETIMI (v = A + C h^p, gurultusuz): p = 1 -> 1.000000000,
    p = 2 -> 2.000000001, p = 3 -> 2.999999264, p = 4 -> 4.000667062.

    MALIYET: olcutun kendi aritmetigi 34.2 - 34.9 us/cagri (iki bagimsiz
    olcum, her biri uc tekrar). Merdivene cikmanin bedeli PENCEREYE BAGLIDIR
    ve spec'in "+%24,9"u bu pencerede yeniden uretilemez: 1B dispersiyon
    cipasinda tek (en ince) kosum 4.0 - 4.6 ms, merdiven 14.0 - 14.5 ms,
    yani +%203 ile +%258 arasi. Maliyet ~ N oldugu icin CEBIRSEL TABAN
    +%87.6'dir ((1601+801+401+201)/1601); kalani 4 ms'lik bir kosumda sabit
    yuktur ve o yuzden bu penceredeki yuzde bir ust sinir olarak okunmalidir.
    `noise` icin ek FDTD kosumu SIFIR.

    DURUST OLUMSUZLAR
    - "%0 yanlis pozitif" BIR SIFIR DEGILDIR: 0/426 icin Clopper-Pearson
      %95 ust siniri %0.9.
    - YARIM-TAMSAYI KOR NOKTASI SPEC'IN YAZDIGINDAN KOTUDUR. Spec "olcut
      karar sinirinin ustundedir" diyor; olculdu ki `noise = 0` iken bir
      BICAK SIRITIDIR ve son bit karari veriyor: p = 0.5 -> OTURDU(0),
      p = 1.5 -> OTURDU(1), p = 2.5 -> OTURDU(3). `noise >= 1e-12` verilen
      ayni merdivenlerde ucu de OTURMADI'ya donuyor. Gercek korpusta u = 0
      olan merdiven YOK (n = 301, min u = 1.357e-05), yani kor nokta yalniz
      gurultusuz sentetik girdide aciktir -- `noise`un zorunlu olmasinin
      ucuncu gerekcesi budur.
    - `u`nun UC TERIMINDEN BIRI DERINLIK 4'TE OLUDUR. Sayim kimligi geregi
      derinlik 4'te Richardson noktasi sayisi 1'dir, dolayisiyla
      `std(p_inf_dizi)` tanimsizdir ve 0 sayilir; spec 7.7'nin D1
      formulundeki ucuncu terim, sevk edilen cipalarin kullandigi derinlikte
      HICBIR ZAMAN baskin olamaz. Terim yanlis degil DERINLIGE BAGLIDIR ve
      1B'de derinlik 5 mumkundur (spec'in "derinlik 5 imkansiz" hukmu 2B
      Bessel cipasina aittir). Derinlik 5'te bile GERCEK merdivenlerde
      `|p_inf - p_son|` `std`i 7 pencerenin 7'sinde geciyor, cunku tek
      kuvvetli bir duzeltmede `std` CEBIRSEL OLARAK sifirdir
      (`p_i = p* + c 2^-i` => `std = 0`, `|p_inf - p_son| = |c|/4`). `std`
      ancak ucuncu bir kuvvet devreye girince baskin olur.
    - `p > 8` UST KAPISI icin GERCEK bir ornek URETILEMEDI. Bu anahtardaki
      uc terimli `omega` tahmincisi N = 12801'e kadar `rho = 4.000`'de
      duruyor; plan/spec'in "N = 3200'de rho = 25,76" sayisi BU okumada
      yeniden uretilemiyor. Ust kapi kurulmus bir merdivenle kilitlendi ve
      bu bir olcum degil TRANSKRIPSIYON kilididir.
    - Yalniz r = 2 sinandi.
    - Iki bagimsiz yer gercegi (bilinen limitten okuma vs log-log en kucuk
      kareler) 426 merdivenin 14'unde (%3.3) tamsayi duzeyinde CELISIYOR;
      ikisi de raporlandi, biri "dogru" ilan edilmedi.
    - `p_inf` yerine ham `p_son` kullanmak GERCEK korpusta 301 merdivenin
      HICBIRINDE hukmu degistirmiyor (0/301), cunku `u`nun birinci terimi
      (`|P - p_son|`) farki zaten sogurur. Ayrim ancak yuksek mertebe
      kirlenmesi buyuk oldugunda goruluyor ve orada karar TERSINE doner
      (bkz. tests/em/test_merdiven.py).
    """
    if not np.isfinite(ratio) or ratio <= 1.0:
        raise InvalidInputError(
            f"ratio > 1 olmali (kabadan inceye), verilen: {ratio}"
        )
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise InvalidInputError(f"kappa pozitif ve sonlu olmali, verilen: {kappa}")
    v, h = _merdiven(values, spacings, ratio)
    g = _gurultu(noise, v.size)

    bos = ConvergenceLadder(
        values=tuple(float(x) for x in v),
        spacings=tuple(float(x) for x in h),
        ratio=float(ratio),
        orders=(),
        p_inf=None,
        uncertainty=None,
        verdict="YARGILANAMAZ",
    )
    # IMKAN KAPILARI -- tolerans DEGIL, aritmetigin tanimli olma kosullari.
    if v.size < MIN_DEPTH:
        return bos
    d = np.diff(v)
    if np.any(d == 0.0) or np.any(d[:-1] * d[1:] <= 0.0):
        return bos

    p = np.log(d[:-1] / d[1:]) / np.log(ratio)
    p_inf_dizi = (ratio * p[1:] - p[:-1]) / (ratio - 1.0)
    p_inf = float(np.mean(p_inf_dizi))
    u = max(
        abs(p_inf - float(p[-1])),
        float(np.std(p_inf_dizi, ddof=1)) if p_inf_dizi.size > 1 else 0.0,
        _gurultu_belirsizligi(v, d, g, float(ratio)),
    )
    kayit = {
        "values": tuple(float(x) for x in v),
        "spacings": tuple(float(x) for x in h),
        "ratio": float(ratio),
        "orders": tuple(float(x) for x in p),
        "p_inf": p_inf,
        "uncertainty": u,
    }
    if not np.all((p > 0.0) & (p <= _P_MAX)):
        return ConvergenceLadder(**kayit, verdict="YARGILANAMAZ")
    if u >= kappa:
        hukum = "YARGILANAMAZ"
    elif abs(p_inf - round(p_inf)) + u < kappa:
        hukum = "OTURDU"
    else:
        hukum = "OTURMADI"
    return ConvergenceLadder(**kayit, verdict=hukum)


# ===========================================================================
# ARAYUZ: yansima katsayisi ve TIY FAZ FARKI
# ===========================================================================
#
# BU BOLUMUN CEKIRDEK IDDIASI, ve neden GERI OKUMA DEGIL:
#
#     delta_ps = arg(rho_p) - arg(rho_s)   TAM IC YANSIMADA
#
# Ortam 1'de her iki kutuplanma AYNI `kx1` ile yayilir (ayni n1, ayni theta),
# dolayisiyla prob->arayuz gidis-donus fazi exp(-2i*kx1*L) IKISINDE DE
# AYNIDIR ve FARKTA TAM SADELESIR. Beyan edilen hicbir konum hesaba girmez.
#
# Bu, elenmis bir aday kilidin (`x_eff`) TAM KARSITIDIR: orada dal
# belirsizligi `round((iface-0.5-x_eff)/per)` satiriyla BEYAN EDILEN
# `iface`'ten kapatiliyordu -- kilit olcmuyor, GERI OKUYORDU.
#
# BEDELI ACIKCA YAZILIR ve UC KORLUKTUR (bkz. `tir_phase_difference`).


#: TIY kapisinin esigi: |rho| = 1 fiziksel olarak ZORUNLUDUR, ve saglanmadigi
#: yerde olculen sey bir yansima katsayisi DEGILDIR. Tolerans degil IMKAN
#: kapisi -- bkz. `reflection_coefficient` dokstringi.
DEFAULT_UNITY_GATE: float = 1e-2


@dataclass(frozen=True)
class ObliqueWindow:
    """Egik olcumun TAM PENCERESI -- bu modulde tohumun yerini tutan sey.

    `grid.Window` bir KOSUMUN penceresidir; bu ise bir OLCUMUN penceresidir
    ve dordu fazladan tasir: `interface`, `source_index`, `probe_offset`,
    `ey_kural`. Ilk ucu geometriyi, sonuncusu bir MODELLEME SECIMINI
    adlandirir.

    `probe_offset` ARAYUZE OLAN UZAKLIKTIR, mutlak prob indisi degil. Bu bir
    kolaylik degil: kilidin geri okuma yapmadiginin sinavi tam olarak bu
    sayiyi TARAMAKTIR (bkz. `tir_phase_difference`).

    Varsayilanlar SEVK EDILEN PENCEREDIR ve olculerek secildi; `sgf` = 9'un
    gerekcesi `source.gauss_cos`ta yaziyor (yalniz `sgf` 3 -> 6 ile
    ||rho_s|-1| 1557 kat degisiyor, `npml`/`N`/`S` hicbir sey yapmiyor).
    """

    cells: int = 3600
    npml: int = 60
    courant: float = 0.4
    interface: int = 1500
    source_index: int = 600
    probe_offset: int = 6
    sgf: float = 9.0
    window_fn: str = "dikdortgen"
    ey_kural: str = "aritmetik"
    unity_gate: float = DEFAULT_UNITY_GATE
    pml_m: float = 3.0

    @property
    def probe(self) -> int:
        return self.interface - self.probe_offset

    def as_lines(self) -> list[str]:
        """Panelde ve testlerde basilacak tek satirlik beyan parcalari."""
        return [
            f"izgara      : {self.cells} hucre, S = {self.courant:g}",
            f"geometri    : iface {self.interface}, kaynak {self.source_index}, "
            f"prob {self.probe} (ofset {self.probe_offset})",
            f"npml        : {self.npml} (m = {self.pml_m:g})",
            f"sgf         : {self.sgf:g}",
            f"FFT penceresi: {self.window_fn}",
            f"ey_kural    : {self.ey_kural}",
            f"|rho|=1 kapisi: {self.unity_gate:g}",
        ]


def fresnel_coefficients(
    eps1: float, eps2: float, theta_deg: float
) -> tuple[complex, complex]:
    """Surekli-ortam Fresnel katsayilari -- KLASIK uzlasimda (e^{+i kx x}).

    `rs` E genligi icin, `rp` ise Hz genligi icin: p-kutuplanmada bu boru
    hattinin kaydettigi alan Hz'dir. Hangi alanin kaydedildigi
    `oblique.run_oblique_1d`de beyan edildi; burada onunla TUTARLI olan
    surum yaziliyor.

        kx1 = n1 cos(theta)
        kx2 = sqrt(eps2 - eps1 sin^2 theta)        (TIY'de saf sanal)
        rs  = (kx1 - kx2) / (kx1 + kx2)
        rp  = (eps2 kx1 - eps1 kx2) / (eps2 kx1 + eps1 kx2)

    SINIF: bu fonksiyon bir CIPA DEGIL, kapali formun kendisidir -- FDTD
    olcumunun karsisina konan surekli-ortam degeri. `n` GIRDIDIR ve `r` onun
    cebirsel sonucudur (spec 2.1); bu yuzden Fresnel'in kendisi bir fizik
    cipasi olamaz, CEVIRI DENETIMIDIR.
    """
    if eps1 <= 0.0 or eps2 <= 0.0:
        raise InvalidInputError(
            f"eps1 ve eps2 pozitif olmali, verilen: {eps1} ve {eps2}"
        )
    th = np.deg2rad(float(theta_deg))
    n1 = np.sqrt(eps1 + 0j)
    kx1 = n1 * np.cos(th + 0j)
    kx2 = np.sqrt(eps2 - eps1 * np.sin(th) ** 2 + 0j)
    if kx2.imag < 0:
        kx2 = -kx2
    rs = (kx1 - kx2) / (kx1 + kx2)
    rp = (eps2 * kx1 - eps1 * kx2) / (eps2 * kx1 + eps1 * kx2)
    return complex(rs), complex(rp)


def fresnel_phase_difference(eps1: float, eps2: float, theta_deg: float) -> float:
    """Surekli-ortam `delta_ps` [DERECE], BU BORU HATTININ uzlasiminda.

    ISARET BIR OLCUMDEN GELIYOR, bir tercihten degil: bu boru hattinin
    uzamsal uzlasimi ARAYUZSUZ bir kosumda, yalniz ortam 1'deki iki prob
    arasindaki faz ilerlemesinden belirlendi (`n2` ve arayuz hic gecmeden) ve
    e^{-i kx x} cikti. Yani olculen `rho`, klasik Fresnel `r`sinin
    ESLENIGIDIR ve `delta` isaret DEGISTIRIR.

    OLCULDU (eps1 = 4, theta = 45, lam = 40, S = 0.4, N = 3600, npml = 60,
    prob x = 1000 ve 1200, ARAYUZ YOK): arg(F2/F1) = -0.523604090 rad.
    Iki aday ve INDIRGENMIS SEMANIN SAYISAL kx'i ile beklenenler --
    kx_sayisal, eps (2/S sin(wS/2))^2 = (2 sin(kx/2))^2 + ky^2'den okunur ve
    0.222529927296 cikar (analitik 0.222144146908, oran 1.0017366):

        e^{-i kx x} : -0.523688309  ->  |fark| = 8.4e-05
        e^{+i kx x} : +0.523688309  ->  |fark| = 1.0e+00

    yani ayrim 12400 kattir. ANALITIK kx ile ayni sinav 7.7e-02 verirdi --
    isareti yine dogru secerdi ama arada 900 kat cozunurluk kaybi olurdu;
    fark, semanin KENDI dispersiyonudur ve bir kusur degildir.
    """
    rs, rp = fresnel_coefficients(eps1, eps2, theta_deg)
    return float(-np.rad2deg(np.angle(rp / rs)))


def total_internal_reflection(eps1: float, eps2: float, theta_deg: float) -> bool:
    """Tam ic yansima kosulu: eps1 sin^2(theta) > eps2.

    `|r| = 1` KAPISININ VARLIK KOSULU budur. Kismi yansimada |r| < 1'dir ve
    1'e olan uzaklik bir kusur degil FIZIKTIR; kapi orada kosarsa dogru
    olcumleri reddeder.
    """
    th = np.deg2rad(float(theta_deg))
    return bool(eps1 * np.sin(th) ** 2 > eps2)


def _record_steps(
    *, theta_deg: float, wavelength: float, eps1: float, window: ObliqueWindow
) -> int | None:
    """Kayit kac adim surmeli -- ve GUVENLI BIR SURE VAR MI?

    Egik geliste x yonundeki grup hizi `vx = cos(theta)/n1`dir; her sey
    `1/cos(theta)` kadar yavaslar. Iki zaman kritiktir:

        t_yans  : yansiyan darbenin prob'a varisi
        t_yanki : ayni darbenin SOL sogurucudan donup prob'a geri gelisi

    Kayit `t_yans + 5*sg`yi ICERMELI, `t_yanki - 5*sg`yi ASMAMALIDIR; ve
    ayrica `lambda`nin TAM KATI olmalidir, yoksa FFT bini `1/lambda`ya
    oturmaz. `ky` kosum boyunca SABITTIR, ama bir binin gordugu ETKIN aci
    `asin(ky/(2 pi f n1))`dir -- bin kayarsa OLCULEN ACI kayar.

    Boyle bir sure yoksa `None` doner. Bu, spec 4.4'un "tahminci
    kirildiginda None" disiplininin OLCUMUN KENDISINE uygulanmis halidir.

    DURUST OLUMSUZ: `t_yanki` prob'un SOL DUVARA olan uzakligindan
    hesaplanir, sogurucunun IC YUZUNE olan uzakligindan degil; yani yanki
    aslinda biraz DAHA ERKEN gelir ve formul o kadar temkinsizdir. Sogurucu
    onu zayiflattigi icin sevk penceresinde olculen ||rho|-1| yine de 1e-8
    civarindadir.
    """
    S = window.courant
    sg = window.sgf * wavelength
    t0 = 5.0 * sg
    th = np.deg2rad(float(theta_deg))
    vx = np.cos(th) / np.sqrt(eps1)
    if not vx > 0.0:
        return None
    t_yans = t0 + (2.0 * window.interface - window.source_index - window.probe) / vx
    t_yanki = t_yans + 2.0 * window.probe / vx
    lo, hi = t_yans + 5.0 * sg, t_yanki - 5.0 * sg
    if lo >= hi:
        return None
    T = np.ceil(lo / wavelength) * wavelength
    if T > hi:
        return None
    return int(round(T / S))


def _spektrum(seri: np.ndarray, courant: float, f: float) -> tuple[complex, float]:
    """KARMASIK zaman serisi -> tam FFT, +f'ye en yakin bindeki deger."""
    F = np.fft.fft(seri)
    fr = np.fft.fftfreq(len(seri), d=courant)
    i = int(np.argmin(np.abs(fr - f)))
    return complex(F[i]), float(fr[i])


_FFT_PENCERELERI = {
    "dikdortgen": np.ones,
    "hann": np.hanning,
    "blackman": np.blackman,
}


def _kosum(
    *,
    medium: Medium,
    boundary,
    steps: int,
    wavelength: float,
    ky: float,
    polarization: str,
    window: ObliqueWindow,
):
    sg = window.sgf * wavelength
    t0 = 5.0 * sg

    def darbe(t: float) -> float:
        return float(gauss_cos(t, wavelength=wavelength, sgf=window.sgf, t0=t0))

    return run_oblique_1d(
        medium,
        courant=window.courant,
        boundary=boundary,
        steps=steps,
        source=darbe,
        source_index=window.source_index,
        probes=(window.probe,),
        wavelength=wavelength,
        sgf=window.sgf,
        ky=ky,
        polarization=polarization,
        window_fn=window.window_fn,
    )


def reflection_coefficient(
    *,
    eps1: float,
    eps2: float,
    theta_deg: float,
    wavelength: float,
    polarization: str,
    window: ObliqueWindow | None = None,
    mirror: bool = False,
) -> complex | None:
    """Karmasik yansima katsayisi `rho`, IKI KOSUMUN FARKINDAN.

    SINIF: `|rho|` icin CEVIRI DENETIMI (spec 2.3) -- FIZIK CIPASI DEGIL.
    Kirilma indisi GIRDIDIR ve Fresnel onun cebirsel sonucudur; bu sayi
    cozucunun arayuzu gercekten cozdugunu gosterir, yeni bir fizik bilgisi
    URETMEZ. Ayni sayi `|rho| = 1` seklinde okundugunda ise TIY'de bir SEMA
    CIPASIDIR.

    Yumusak nokta kaynagi iki yonlu isir; gelen alan ARAYUZSUZ referans
    kosumundan alinir ve yansiyan alan IKI KOSUMUN FARKIDIR. Maliyet:
    cagri basina 2 FDTD kosumu.

    `mirror = True` -> AYNA KONTROLU: izgara arayuzde BITER (`cells =
    interface + 1`), sogurucu yalniz SOL yuzdedir ve sag uc `_adim`in kendi
    PEC duvaridir. Bu kipte `eps2` KULLANILMAZ -- ve kullanilmadigi bir
    SINAVDIR (bkz. `tir_phase_difference`). Duvarin tipi kutuplanmaya gore
    okunur ve BEYAN EDILIR: s'te `e` = Ez oldugu icin `e[-1] = 0` bir PEC,
    p'de `e` = Hz oldugu icin ayni satir bir PMC'dir. Ikisi de MALZEMESIZ
    mukemmel aynadir ve ikisi de tam olarak AYNI hucrede oturur.

    `None` DONER ve sessizce imkansiz bir sayi basmaz:
      * guvenli bir kayit uzunlugu yoksa (bkz. `_record_steps`),
      * kosum sonlu kalmadiysa,
      * ve `|rho| = 1`in ZORUNLU oldugu yerde (TIY ya da ayna) `||rho| - 1|`
        kapiyi geciyorsa.

    UCUNCU MADDE OLCUM ALETININ TAVANIDIR ve bir DUGMEDIR: `sgf`
    (darbe bant genisligi) yeterince buyuk degilken genis bant KOMSU
    BINLERIN farkli acilarinin `r`sini hedef bine karistirir ve `|rho|`
    1'den sapar. Sebep `source.gauss_cos`ta olculu; buradaki kapi onun
    sonucudur. Kismi yansimada (TIY yok, ayna yok) kapi KOSMAZ -- orada
    |r| < 1 fizigin kendisidir.

    KAPI BIR ACI ESIGI DEGILDIR, ve bu bir SECIM degil bir OLCUMDUR.
    Sevk penceresinde (eps1 = 4, eps2 = 1, lam = 40) kapi KAPALIYKEN olculen
    |rho_s|: theta = 45/55/65 -> 1.000000/1.000000/0.998643 (GECER),
    70/72.5 -> 1.020642/1.029093 (KESER, ve bunlar R > 1 yani IMKANSIZ),
    75 -> 0.994701 (GECER), 77.5/80/85 -> 0.960226/0.941959/0.940616
    (KESER). DURUST OLUMSUZ: bozulma theta ile MONOTON DEGILDIR -- theta =
    75 iki kesilen bolgenin ARASINDAN geciyor. Bir "theta >= 75 tavani"
    yazilamaz; yazilabilecek olan nokta nokta bir imkan kapisidir.

    MALIYET (olculdu, sevk penceresi): theta = 45, lam = 40'ta cagri basina
    1.07 s; theta = 70'te 1.543 s; theta = 80'de 2.416 s (kayit uzunlugu
    1/cos(theta) ile buyudugu icin).
    """
    if polarization not in POLARIZATIONS:
        raise InvalidInputError(
            f"polarization {POLARIZATIONS} icinden olmali, verilen: "
            f"{polarization!r}"
        )
    w = ObliqueWindow() if window is None else window
    if w.window_fn not in _FFT_PENCERELERI:
        raise InvalidInputError(
            f"window_fn {tuple(_FFT_PENCERELERI)} icinden olmali, verilen: "
            f"{w.window_fn!r}"
        )
    if not 0 < w.interface < w.cells:
        raise InvalidInputError(
            f"interface 1 ile {w.cells - 1} arasinda olmali, verilen: {w.interface}"
        )
    steps = _record_steps(
        theta_deg=theta_deg, wavelength=wavelength, eps1=eps1, window=w
    )
    if steps is None:
        return None
    n1 = np.sqrt(eps1)
    ky = (2.0 * np.pi / wavelength) * n1 * np.sin(np.deg2rad(float(theta_deg)))

    if mirror:
        on_hucre = w.interface + 1
        on_ortam = Medium(
            np.full(on_hucre, float(eps1)), np.ones(on_hucre), w.ey_kural
        )
        on_sinir = PML(thickness=w.npml, m=w.pml_m, yuzler="sol")
    else:
        on_ortam = Medium.half_space(
            w.cells,
            interface=w.interface,
            eps1=eps1,
            eps2=eps2,
            ey_kural=w.ey_kural,
        )
        on_sinir = PML(thickness=w.npml, m=w.pml_m)
    arka_ortam = Medium(np.full(w.cells, float(eps1)), np.ones(w.cells), w.ey_kural)
    arka_sinir = PML(thickness=w.npml, m=w.pml_m)

    ortak = dict(
        steps=steps, wavelength=wavelength, ky=ky, polarization=polarization, window=w
    )
    on = _kosum(medium=on_ortam, boundary=on_sinir, **ortak)
    arka = _kosum(medium=arka_ortam, boundary=arka_sinir, **ortak)
    gelen = arka.probes[0]
    yansiyan = on.probes[0] - gelen
    if not (np.all(np.isfinite(gelen)) and np.all(np.isfinite(yansiyan))):
        return None

    pen = _FFT_PENCERELERI[w.window_fn](steps)
    f = 1.0 / wavelength
    Fi, f_bin = _spektrum(gelen * pen, w.courant, f)
    Fr, _ = _spektrum(yansiyan * pen, w.courant, f)
    if abs(f_bin * wavelength - 1.0) > 1e-9 or Fi == 0:
        return None
    rho = Fr / Fi
    if not np.isfinite(rho):
        return None
    if mirror or total_internal_reflection(eps1, eps2, theta_deg):
        if abs(abs(rho) - 1.0) > w.unity_gate:
            return None
    return complex(rho)


def tir_phase_difference(
    *,
    eps1: float,
    eps2: float,
    theta_deg: float,
    wavelength: float,
    window: ObliqueWindow | None = None,
    mirror: bool = False,
    probe_offset: int | None = None,
) -> float | None:
    """`delta_ps = arg(rho_p) - arg(rho_s)` [DERECE] -- MODULUN ILK FIZIK CIPASI.

    SINIF: FIZIK CIPASI. Ayirt edici sey sudur: `|rho|` bir CEVIRI DENETIMIDIR
    (kirilma indisi girdidir, Fresnel onun cebiri), ama `delta_ps` TIY'de
    girilen parametrelerden dogrudan OKUNAMAYAN bir sayidir -- iki
    kutuplanmanin evanesan bolgede FARKLI sinir kosullarina uymasindan dogar:

        arg(r_s)   = -2 atan( kappa / kx1 )
        arg(r_p)   = -2 atan( eps1 kappa / (eps2 kx1) )
        kappa = sqrt(eps1 sin^2 theta - eps2)

    NEDEN GERI OKUMA DEGIL -- CEBIRSEL. Ortam 1'de her iki kutuplanma AYNI
    `kx1` ile yayilir, dolayisiyla prob->arayuz gidis-donus fazi
    exp(-2i kx1 L) IKISINDE DE AYNIDIR ve FARKTA TAM SADELESIR. Beyan edilen
    hicbir konum hesaba girmez; `probe_offset` yalnizca MODELI KURAR.
    OLCULDU (eps1 = 4, eps2 = 1, theta = 45, lam = 40, sevk penceresi):
    `probe_offset` 6/50/94/300'de tek kutuplanmanin fazi -70.195/-112.198/
    -154.200/-7.201 derece dolasirken `delta_ps` YAYILIMI 2.421e-06 derece
    (1.11e-03 sigma_analiz).

    ICERIK (sinav A, ayni pencere): n2 = 1.00 -> 1.35'te olculen `delta_ps`
    70.685182 -> 33.078530, yayilim 37.606652 derece; ayni araligin KAPALI
    FORM yayilimi 37.350204 derece (oran 1.006866). sigma_analiz = 0.002187
    derece (15 dugme ekseni), yani yayilim 17193 sigma. Ayni araliktaki DIK
    GELIS Fresnel yansimasi yalnizca 3.0 kat degisir -- kilit dik geliste
    OKUNAMAYAN bir seyi okuyor.

    UC KORLUK -- KUSUR DEGIL, AYNI OZELLIGIN BEDELI:

    1. Arayuzu bir hucre KAYDIRMAK gorunmez. Bu ispatlanabilir: kaydirma,
       ortam 1 icinde bir OTELEMEDIR ve her iki kutuplanma AYNI carpani alir.
       GERI OKUMA YOKLUGU ILE OTELEMEYE DUYARLILIK AYNI SEYIN IKI YUZUDUR;
       ikisi birden istenemez. Olculdu: arayuz 1500 -> 1499 (prob YERINDE)
       arg(rho_s)'yi 25.500 derece oynatirken `delta_ps`i 1.33e-06 derece
       (6.1e-04 sigma) oynatiyor. Bu yuzden Fresnel `|rho|` CEVIRI DENETIMI
       olarak sevk edilmeye devam ediyor -- o, ayni kaydirmayi bagil 2.4e-06
       ile gorur (yani o da ZAYIF gorur; `arg(rho)` 36.14 derece ile GUCLU
       gorur ama arg bir GERI OKUMADIR).
    2. Sogurucunun kaldirilmasi (npml = 0, PEC sinir yansimasi) TEK IZGARADA
       gorunmez: lam = 40'ta fark 7.15e-05 derece (0.033 sigma). MERDIVENDE
       GORUNUR: lam = 54'te fark 0.030381 derece (13.9 sigma) ve merdivenin
       hukmu OTURDU(2)'den YARGILANAMAZ'a duser (p_inf 2.103 -> 0.244).
    3. Kaynagin genligindeki her hata GORUNMEZ, cunku `rho` bir ORANDIR ve
       oran genlik carpanini sadelestirir. Cozucu dogrusaldir ve olculdu:
       kaynak 2 katina cikarilinca kayit BIT BIT 2 kati, `rho` ise BIREBIR
       ayni (|fark| = 0.000e+00).

    `mirror = True` -> MALZEMESIZ AYNA, ve SONUC PLANIN BEKLEDIGINDEN DAHA
    SERTTIR. Sevk edilen aynada iki duvar TAM OLARAK AYNI hucrededir
    (s'te PEC, p'de PMC) ve ortam duzgundur; bu durumda iki kutuplanmanin
    guncellemesi sabit bir olcek carpani disinda AYNI OZYINELEMEDIR, `rho`
    bir ORAN oldugu icin carpan sadelesir ve `rho_s` ile `rho_p` BIREBIR
    esit cikar. Olculdu: `delta_ayna` = 0.000000000000 derece, `eps2` = 1.0
    ve 9.0 icin AYNI (fark 0.000e+00), ve UC MUTANT (dt_carpan, kaydir,
    sinir_pec) ile de AYNI -- yani ayna kipinde kilit 250 sigma degil SIFIR
    sigma ayiriyor. Kilit orada malzeme bilgisi tasimiyor; TASIYAN sey
    dielektrik adimin kendisidir. Boru hattinin AYNADA DA CANLI oldugu ayrica
    olculur: |rho| = 1'e 4.8e-09'da oturuyor ve arg(rho_s) `probe_offset` ile
    26.9997 -> 89.9942 derece dolasiyor.
    """
    w = ObliqueWindow() if window is None else window
    if probe_offset is not None:
        w = replace(w, probe_offset=int(probe_offset))
    ortak = dict(
        eps1=eps1,
        eps2=eps2,
        theta_deg=theta_deg,
        wavelength=wavelength,
        window=w,
        mirror=mirror,
    )
    rs = reflection_coefficient(polarization="s", **ortak)
    # KISA DEVRE: `s` kapiya takildiysa `p` KOSMAZ. Bu bir maliyet karari ve
    # olculdu -- theta = 80'de tek bir `reflection_coefficient` cagrisi
    # 2.416 s; kisa devre olmasaydi kesin `None` donecek bir olcum icin iki
    # kat odenirdi.
    if rs is None:
        return None
    rp = reflection_coefficient(polarization="p", **ortak)
    if rp is None:
        return None
    return float(np.rad2deg(np.angle(rp / rs)))


def tir_phase_ladder(
    *,
    eps1: float,
    eps2: float,
    theta_deg: float,
    wavelengths,
    noise,
    window: ObliqueWindow | None = None,
    kappa: float = DEFAULT_KAPPA,
) -> ConvergenceLadder | None:
    """`delta_ps`in IZGARA MERDIVENI -- spec 4.3'un pazarlik edilmeyen kaydi.

    `wavelengths` KABADAN INCEYE (artan lambda = incelen izgara, cunku bu
    modulde `dx = 1` ve cozunurluk `lambda` hucre/dalgaboyudur). Izgara
    araligi `h = 1/lambda` olarak verilir; `ratio` lambda'lardan OLCULUR,
    girilmez.

    Herhangi bir basamak kapiya takilirsa (`None`) merdiven de `None`
    doner -- eksik bir merdivenden hukum cikarilmaz.

    NEDEN `r` = 2 DEGIL: sevk penceresinde r = 2'lik bir DORT basamak YOK.
    Olculdu (eps1 = 4, eps2 = 1, theta = 45): lam = 5 ve 10 `|rho| = 1`
    kapisina takiliyor (izgara cozunurlugu), lam = 160'ta ise guvenli bir
    kayit uzunlugu kalmiyor (`_record_steps` -> None). Geometriyi uzatarak
    (iface 3000) lam = 20/40 gecti ama 80 ve 160 yine kapiya takildi. Kalan
    r = 2 dizisi 20/40/80'dir ve DERINLIGI UCTUR -- `convergence_ladder`in
    dort basamak sarti tam olarak burada isiriyor. `r` = 1.5 ile
    16/24/36/54 dordu de geciyor ve dordunun de `lambda/S` degeri TAM SAYI
    (40/60/90/135), yani FFT bini `1/lambda`ya oturuyor.

    SEVK EDILEN MERDIVENIN HUKMU (lam = 16/24/36/54, noise = 3.09e-05):
    degerler 71.460721 / 70.956582 / 70.721486 / 70.615084, p_inf = 2.1028
    +- 0.3364 -> **OTURDU(2)**. Ayni merdiven mutantlarda: `eps_orta`
    OTURDU(1) (p_inf 1.178), `dt_carpan` OTURDU(2) ama YANLIS LIMITE,
    `sinir_pec` YARGILANAMAZ (p_inf 0.244), `ey_kural = harmonik`
    OTURDU(1) (p_inf 0.984); `kaydir` ve `kaynak_cift` dogru cozucuden
    AYIRT EDILEMEZ (beyan edilen korluk).

    DURUST OLUMSUZ -- HUKUM PENCEREYE BAGLIDIR: bu merdiven lam = 16..54
    araligindadir ve orada artik temiz 2. mertebedir. DAHA GENIS bir
    pencerede bozuluyor: lam = 20/40/80'de artik +0.609275 / +0.156403 /
    +0.060736, yani ardisik oranlar 3.896 ve 2.575 -- ilk adim r^2 = 4'e
    yakin, ikincisi belirgin sekilde ALTINDA. Asimptotik olarak 2. mertebeden
    BASKA bir terim daha var (arayuzun basamak ayriklastirmasi 1.
    mertebedir). "2. mertebe" hukmu bu ARALIGIN hukmudur, sonsuz izgaranin
    degil. Daha ince izgarada ne oldugu bu geometride OLCULEMEDI: lam >= 120
    icin guvenli bir kayit uzunlugu yok (`_record_steps` -> None).

    IKINCI DURUST OLUMSUZ: `OTURDU(2)`nun kappa = 0.5'e olan payi INCEDIR
    (|p_inf - 2| + u = 0.4392). Alti farkli pencere degisikliginde (npml,
    cells, isrc, iface, S) p_inf 1.9245 - 2.1028 arasinda kaldi ve hukum
    degismedi; ama `noise` 1e-3 verilseydi hukum YARGILANAMAZ olurdu.
    `noise` bu yuzden OLCULEN pencere gurultusudur (sigma_analiz / |delta| =
    0.002187 / 70.685 = 3.09e-05), secilen bir sayi DEGIL.
    """
    lam = np.asarray(wavelengths, dtype=float)
    if lam.ndim != 1 or lam.size < 2:
        raise InvalidInputError(
            f"wavelengths en az iki degerli tek boyutlu olmali, verilen: {lam.shape}"
        )
    if not np.all(np.diff(lam) > 0.0):
        raise InvalidInputError(
            f"wavelengths ARTAN olmali (kabadan inceye), verilen: {lam.tolist()}"
        )
    ratio = float(lam[1] / lam[0])
    degerler = []
    for L in lam:
        d = tir_phase_difference(
            eps1=eps1,
            eps2=eps2,
            theta_deg=theta_deg,
            wavelength=float(L),
            window=window,
        )
        if d is None:
            return None
        degerler.append(d)
    return convergence_ladder(
        degerler, 1.0 / lam, noise=noise, ratio=ratio, kappa=kappa
    )
