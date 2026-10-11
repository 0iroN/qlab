"""Elektromanyetik gozlemlenebilirler -- ve OLCUMUN KENDI DURUSTLUK MAKINESI.

Bu dosyanin ilk parcasi `convergence_ladder`dir: Modul 5'in her FIZIK
cipasi degeriyle birlikte bir YAKINSAMA KAYDI dondurmek zorundadir (spec
4.3, pazarlik edilmez). Bir FDTD sayisi IZGARA INCELIRKEN oturmadiysa
yakinsamamistir ve raporlanan sey bir tahmin degil bir ARA DURAKTIR.

BU CUMLE BIR ZAMANLAR *"`dx` kuculurken"* DIYORDU ve o hal, `dx` kapatilinca
YANLIS oldu: bu modulde `dx` KUCULEMEZ, `engine/em/grid.dx_kapisi` `dx != 1`i
REDDEDER. Izgara `lambda` BUYUTULEREK incelir (cozunurluk = hucre/dalgaboyu)
ve merdiven zaten oyle kurulmustur -- bkz. `convergence_ladder`in kendi
dokstringi. Ayrim bir yazim ayrintisi degil: `dx` bir cozunurluk dugmesi
OLAMAZ, cunku izgarayi olcekleyemez (olculdu; gerekce `dx_kapisi`da).

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

#: KAYIT TAMLIGI kapisinin esigi: kaydin SON PERIYODUNDAKI en buyuk genligin
#: kaydin tepesine orani, gelen ve yansiyan seri icin (buyugu). Tolerans
#: degil IMKAN kapisi: kayit bittiginde darbe hala geliyorsa olculen sey
#: kesik bir kaydin spektrumudur, yansima katsayisi DEGILDIR. Esik OLCULEN
#: BANTTAN secildi (10.10.2026, sevk penceresi, 4 kontrast x 56 aci x
#: lam 10/20/40 x s/p, ayrik-tam referansa karsi): depodaki sabitlenmis
#: cagrilarin en buyuk kuyrugu 9.44e-04 (eps 4 -> 1.8225, theta 45, lam 40,
#: p), eski aletin sessizce dondurdugu hatasi > 1e-4 olan sayilarin en
#: kucuk kuyrugu 2.78e-03; esik ikisinin geometrik ortasi (1.6e-3, iki
#: yana 1.7 kat pay). Ayrinti: `reflection_coefficient`.
DEFAULT_TAIL_GATE: float = 1.6e-3

#: KAYIT ICI YANKI kapisinin esigi (RC-a, 10.10.2026): gelen kaydinda ana
#: darbeden SONRA kalanin f = 1/lam'daki spektral agirliginin ana darbeninkine
#: orani (`_ikinci_varis`). Tolerans degil IMKAN kapisi: kalan, gelenin
#: paydasina eklenir ve yansiyanda karsiligi yoksa olculen sey yansima
#: katsayisi DEGILDIR. Esik OLCULEN BANTTAN secildi: kapisi acik sabitlenmis
#: cagrilarin en buyugu 2.02e-04, kuyruk kapisindan gecen hatasi > 1e-3
#: sayilarin en kucugu 9.84e-04; esik ikisinin geometrik ortasi (4.5e-4, iki
#: yana ~2.2 kat pay). Ayrinti: `reflection_coefficient`.
DEFAULT_ECHO_GATE: float = 4.5e-4


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

    `window_fn` bu pencerenin bir alanidir (kosum kaydina ustveri olarak
    gecer), ama `reflection_coefficient` YALNIZ "dikdortgen"i kabul eder;
    dikdortgen disindaki her deger (hann, blackman, bilinmeyen adlar)
    `InvalidInputError` verir (RC-d, olculdu; bkz. orada).
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
    tail_gate: float = DEFAULT_TAIL_GATE
    echo_gate: float = DEFAULT_ECHO_GATE

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
            f"kuyruk kapisi : {self.tail_gate:g}",
            f"yanki kapisi  : {self.echo_gate:g}",
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
    *,
    theta_deg: float,
    wavelength: float,
    eps1: float,
    window: ObliqueWindow,
    eps2: float | None = None,
    mirror: bool = False,
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

    IKINCI DURUST OLUMSUZ (10.10.2026, olculdu): zamanlar SUREKLI ortamin
    `vx`'iyle ve DAGILMAMIS darbe genisligiyle (`sg`) hesaplanir; ikisi de
    yanlis olabilir ve o zaman `lo` darbe BITMEDEN gelir. (a) Sabit `ky`'li
    indirgeme dagitkandir (eps w^2 = kx^2 + ky^2): grup hizi dagilimi
    k'' = -sqrt(eps1) sin^2(theta) / (w cos^3(theta)), darbe prob'a
    sg * sqrt(1 + (k'' L / sg^2)^2) genisliginde varir -- 1 -> 2.25, lam 20,
    theta 75'te 4.9 kat; kesim frekansi f sin(theta) bandin icine girer.
    (b) Kaba izgarada semanin KENDI grup hizi sureklininkinden kucuktur:
    eps 4, lam 10'da (ortamda 5 hucre/dalgaboyu) 0.393, surekli 0.5; darbe
    `lo`ya TEPESIYLE varir. Bu fonksiyon bunlari DUZELTMEZ; kesik kaydi
    `reflection_coefficient`in kuyruk kapisi yakalar.

    SAG SOGURUCU (RC-a, 10.10.2026; olculdu): `t_yanki` tek sahte varis
    DEGILDIR. Sag sogurucunun ic yuzunden (`cells - npml`) iki yanki daha
    doner: arka (arayuzsuz) kosumda saga giden dalganinki (ortam 1'de, `vx`)
    ve on kosumda GECEN dalganinki (ortam 2'de, x grup hizi
    `vx2 = sqrt(eps2 - eps1 sin^2(theta)) / eps2`; yalniz `eps2` verilmisse,
    gecen dalga YAYILIYORSA ve ayna DEGILSE). Ikisi de yansiyandan
    `2 (cells - npml - iface) / vx` (ya da `/ vx2`) sonra gelir; `hi` uc
    yankinin EN ERKENINDEN hesaplanir. Bu bir olcum degil GEOMETRIDIR,
    cunku gecen dalganin yankisi yansiyanla ZAMANDA ORTUSEBILIR ve kayittan
    ayrilamaz: cells 1600, 2.25 -> 9, theta 30, lam 40, p'de bu sinir
    yokken gecen sayinin hatasi 8.21e-03 idi (kuyruk 5.2e-04; gecen
    dalganin yankisi YANSIYANDADIR, yalniz gelen kaydina bakan ikinci varis
    olcutu onu goremez). BEDEL
    (`.superpowers/sdd/rc-olcum/rca/`, 1357 durum): 10 yanlis alarm, hepsi
    5 sg payinin temkininden (ornek: 1800 hucre, iface 750, sgf 6, 4 -> 1,
    theta 0, lam 40'ta eski hata 1.8e-05); depodaki sabitlenmis
    cagrilarin HICBIRININ kayit uzunlugu degismedi.
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
    sag = window.cells - window.npml - window.interface
    t_yanki = min(t_yanki, t_yans + 2.0 * sag / vx)
    if eps2 is not None and not mirror:
        s2 = eps1 * np.sin(th) ** 2
        if eps2 > s2:
            vx2 = np.sqrt(eps2 - s2) / eps2
            t_yanki = min(t_yanki, t_yans + 2.0 * sag / vx2)
    lo, hi = t_yans + 5.0 * sg, t_yanki - 5.0 * sg
    if lo >= hi:
        return None
    T = np.ceil(lo / wavelength) * wavelength
    if T > hi:
        return None
    return int(round(T / S))


def _kuyruk(seri: np.ndarray, *, wavelength: float, courant: float) -> float:
    """Kaydin SON PERIYODUNDAKI en buyuk |deger| / kaydin tepesi.

    Bir periyot = ceil(lam / S) ornek. Tepe sifirsa (hic sinyal yok) kayit
    tamdir ve 0 doner. Rampa hep 1 verir: hala buyuyen bir sinyal tam
    sayilamaz.
    """
    a = np.abs(seri)
    tepe = float(np.max(a))
    if tepe == 0.0:
        return 0.0
    p = int(np.ceil(wavelength / courant))
    return float(np.max(a[-p:])) / tepe


def _ikinci_varis(
    gelen: np.ndarray,
    *,
    theta_deg: float,
    wavelength: float,
    eps1: float,
    window: ObliqueWindow,
) -> float:
    """GELEN kaydinda ana darbe BITTIKTEN SONRA kalanin SPEKTRAL agirligi.

    c = |F(gelen[i:])| / |F(gelen[:i])|; F, `window_fn` penceresiyle tam
    f = 1/lam'daki DTFT (FFT binine ozdes, bkz. `_spektrum`), i ana darbenin
    son ornegi: darbe prob'a `t_dir = t0 + |prob - kaynak| / vx`te varir ve
    `t_dir + 5 sg`de biter (`_record_steps`in zamanlariyla AYNI: surekli
    `vx`, dagilmamis `sg`). O ornekten (dahil) kaydin sonuna kadarki her sey
    IKINCI bir varistir; olculen kaynagi sogurucu yankisidir (kaynagin sola
    giden dalgasinin sol PML'den donusu). SPEKTRAL cunku hatanin kendisi bu
    orandir: gelenin paydasi F_a + F_b olur, yansiyanda F_b'nin karsiligi
    yoksa |rho - r| = |r| c / |1 + F_b / F_a| <= |r| c / (1 - c).
    1'de KIRPILIR (`echo_gate` = 1 her kaydi kabul etsin: kapi kapali =
    eski alet). Kayit sifirsa ya da i kaydin disindaysa 0; ana darbenin
    agirligi sifirsa 1 doner.
    """
    n = gelen.shape[-1]
    if n == 0 or not np.any(gelen):
        return 0.0
    sg = window.sgf * wavelength
    vx = np.cos(np.deg2rad(float(theta_deg))) / np.sqrt(eps1)
    t_son = 5.0 * sg + abs(window.probe - window.source_index) / vx + 5.0 * sg
    i = int(np.ceil(t_son / window.courant))
    if i >= n:
        return 0.0
    faz = np.exp(-2j * np.pi * (window.courant / wavelength) * np.arange(n))
    s = gelen * _FFT_PENCERELERI[window.window_fn](n) * faz
    ana = abs(complex(np.sum(s[:i])))
    if ana == 0.0:
        return 1.0
    return min(1.0, abs(complex(np.sum(s[i:]))) / ana)


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

    `window_fn` "dikdortgen" DEGILSE kosumdan ONCE `InvalidInputError`
    verir (asagida FFT PENCERESI; RC-d).

    `None` DONER ve sessizce imkansiz bir sayi basmaz:
      * guvenli bir kayit uzunlugu yoksa (bkz. `_record_steps`),
      * kosum sonlu kalmadiysa,
      * kayit TAM degilse -- kaydin son periyodundaki genlik, gelen ya da
        yansiyan seride, tepenin `tail_gate` katini asiyorsa (KUYRUK KAPISI,
        asagida),
      * GELEN kaydinda ana darbeden sonra kalanin f'deki spektral agirligi
        ana darbeninkinin `echo_gate` katini asiyorsa (IKINCI VARIS,
        asagida),
      * ve `|rho| = 1`in ZORUNLU oldugu yerde (TIY ya da ayna) `||rho| - 1|`
        kapiyi geciyorsa.

    DORDUNCU MADDE OLCUM ALETININ TAVANIDIR ve bir DUGMEDIR: `sgf`
    (darbe bant genisligi) yeterince buyuk degilken genis bant KOMSU
    BINLERIN farkli acilarinin `r`sini hedef bine karistirir ve `|rho|`
    1'den sapar. Sebep `source.gauss_cos`ta olculu; buradaki kapi onun
    sonucudur. Kismi yansimada (TIY yok, ayna yok) kapi KOSMAZ -- orada
    |r| < 1 fizigin kendisidir. KARISIM YALNIZ KESIK KAYITTA OLUR (10.10.2026,
    olculdu): ayni kayitta FFT bini ile tam f = 1/lam'da DTFT arasindaki fark
    <= 2.9e-15; kayit tamsa dogrusal bir sistemin oraninda komsu bin
    YOKTUR. `sgf`in etkisi kayit tamligi uzerindendir (dar bant -> az grup
    hizi dagilimi, kesim frekansina daha cok sigma).

    KAPI BIR ACI ESIGI DEGILDIR, ve bu bir SECIM degil bir OLCUMDUR.
    Sevk penceresinde (eps1 = 4, eps2 = 1, lam = 40) kapi KAPALIYKEN olculen
    |rho_s|: theta = 45/55/65 -> 1.000000/1.000000/0.998643 (GECER),
    70/72.5 -> 1.020642/1.029093 (KESER, ve bunlar R > 1 yani IMKANSIZ),
    75 -> 0.994701 (GECER), 77.5/80/85 -> 0.960226/0.941959/0.940616
    (KESER). DURUST OLUMSUZ: bozulma theta ile MONOTON DEGILDIR -- theta =
    75 iki kesilen bolgenin ARASINDAN geciyor. Bir "theta >= 75 tavani"
    yazilamaz; yazilabilecek olan nokta nokta bir imkan kapisidir.
    KUYRUK KAPISIYLA (10.10.2026) ayni tablo: 45 ve 55 gecer (ayrik-tam
    referansa hata 1.2e-07 / 3.4e-07, kuyruk 5.2e-06 / 1.2e-05); 65..85'in
    HEPSI `None` (kuyruk 2.35e-02 .. 0.99; kapi kapaliyken hata 1.6e-03 ..
    7.4e-02). theta = 75'in `|rho| = 1` kapisindan GECMESI bir rastlantiydi:
    hatasi 4.7e-02, kaydin sonunda genlik tepenin 0.69'u.

    KUYRUK KAPISI -- SESSIZ PROTOKOL HATASININ KAPISI (10.10.2026). G1 on
    olcumu (09.10.2026) kismi yansimada `|rho| = 1` kapisinin hic kosmadigi
    yerde sessiz hata buldu: 1 -> 2.25, lam 20, theta 75'te |rho - r| =
    2.01e-02, G1 bedelinden (1.47e-02) buyuk; 2.25 -> 1, theta 41.80'de
    (surekli olcute gore TIY degil, semaninkine gore TIY) 5.47e-02.
    Referans AYRIK-TAM r'dir (kesilmemis kayit, sonsuz izgara; s ve p icin
    transfer ozyinelemesi, s'te G1 kapali formuna 1e-15'te oturur).
    KAYNAK AYRILDI (G1'in iki durumunda): FFT bini / pencere SUCSUZ (bin =
    DTFT, 2.9e-15; Hann her yerde DAHA KOTU); PML SUCSUZ (npml 60 -> 200, r0
    1e-6 -> 1e-12 T_min'de rho'yu <= 2.3e-07 oynatir); suc KAYIT UZUNLUGUNDA:
    darbe kayit bittiginde hala gelmektedir. Uc fiziksel sebep (bkz. `_record_steps` ikinci durust
    olumsuz): sabit-`ky` indirgemesinin grup hizi dagilimi ve kesim
    frekansina yakinlik (siyirma), dal noktasinin uzun yansima kuyrugu
    (kritik aci), kaba izgarada semanin yavas grup hizi (yogun ortam,
    theta = 0'da bile: eps 4 -> 1, lam 10'da hata 1.09e-01). Ek olarak
    bir sogurucu yankisi (kaynagin SOLA giden dalgasinin sol PML'den, gecen
    dalganin sag PML'den donusu) kayda yarim ya da TAM girebilir (asagida
    IKINCI VARIS).
    OLCULDU (sevk penceresi, eps 1 -> 2.25, 2.25 -> 1, 4 -> 1, 1 -> 4 x 56
    aci x lam 10/20/40 x s/p = 336 durum, 10 kayit yok): ESKI ALET 253 sayi
    dondururdu, 157'sinin hatasi > 1e-4, 136'sinin > 1e-3, 99'unun > 1e-2.
    KAPIYLA 89 sayi doner ve en kotusunun hatasi 8.28e-05 (izgara
    noktalarinda; ayni pencerede izgara DISINDA 1.49e-04 olculdu: 2.25 -> 1,
    44 der, lam 30, s, kuyruk 9.8e-04, oran 0.15). Kapinin 164 YENI
    `None`unun bir kismi aslinda temizdi (eski hata 2.96e-06'ya kadar):
    bedeli YANLIS ALARMDIR, sessiz hata degil.
    ESIK BANTTAN SECILDI: depodaki sabitlenmis cagrilarin en buyuk kuyrugu
    9.44e-04, sessiz hatasi > 1e-4 olan sayilarin en kucuk kuyrugu 2.78e-03;
    `DEFAULT_TAIL_GATE` = 1.6e-3 ikisinin geometrik ortasi.
    HATA SINIRI, BEYAN -- PROB ARAYUZE YAKINKEN (sevk ofseti 6):
    kuyruk <= 1e-3 olan izgara durumlarinda hata/kuyruk <= 0.11 (izgara
    disinda 0.15); depoda en kotusu 0.54 (`probe_offset` 300, eps 4 -> 1,
    theta 45, lam 40: kaynak yankisi kayda TEPESINDEN girer, eksik kalan yari
    yaklasik yarim genlik eder; hata 2.02e-04, kuyruk 3.76e-04). Yani orada
    gecen bir sayinin protokol hatasi ~0.54 * `tail_gate` = 8.6e-04'u
    asmaz diye OKUNUR -- olculen bir oran, kanit degil. Prob arayuzden
    UZAKKEN bu oran tutmaz; yanki kaynakli hatayi orada `echo_gate` sinirlar
    (asagida IKINCI VARIS).
    IKINCI VARIS -- KAYIT ICI YANKININ KAPISI (RC-a, 10.10.2026). Bagimsiz
    dogrulayici kuyruk kapisinin bir KOR NOKTASINI buldu: prob arayuzden
    uzaksa bir sogurucu yankisi GELEN kaydina TAMAMEN girer, yansimasi ise
    kayit bitmeden YANSIYANA varmaz; kaydin sonu sessizdir ve kuyruk kapisi
    sayiyi GECIRIRDI. Olculen (9 -> 1, theta 30, lam 40, s): sevk
    penceresinde `probe_offset` 590'da hata 5.45e-03 (kuyruk 1.42e-05),
    1800 hucre / sgf 6 penceresinde ofset 400'de 6.20e-03; r0 1e-6 ->
    1e-12 ile 8.1e-05'e iner (yani yanki sogurucunun). Sag sogurucunun
    yankisini `_record_steps` geometriyle disliyor (orada).
    KAPI: `_ikinci_varis` -- gelen kaydinda ana darbenin sonundan (t_dir +
    5 sg) kaydin sonuna kadar kalanin f'deki spektral agirligi / ana
    darbeninki -- esigi `echo_gate`. NEDEN SPEKTRAL (cebir, `_ikinci_varis`):
    yansiyanda karsiligi olmayan kalan icin |hata| <= |r| c / (1 - c),
    yankinin SEKLINDEN bagimsiz. OLCULEN (`.superpowers/sdd/rc-olcum/rca/
    spek_analiz.log`, 2986 durum: eps1 1..16, ofset 5..880, 500..3600
    hucre, npml 0..120, sgf 1.5..14, 322 ayna; ikinci dogrulayicinin
    karsi-ornek avi dahil):
    yankinin baskin oldugu 444 durumda (c > 1e-4, c > 3 kuyruk) bu sinir %5
    payla hic asilmadi (c buyudukce oran 1 / (1 - c)'ye yaklasir).
    ELENEN ILK SURUM -- ZAMAN-ALANI TEPE ORANI (ayni pencerede en buyuk
    |deger| / tepe, kuyrukla ayni esik 1.6e-3). Egik yanki ince PML'de
    genisler, tepesi spektral agirligini %12..19 eksik okur: dogrulayici
    onunla gecen bir sayi buldu -- 9 -> 1, theta 48.25, lam 42, p, 1800
    hucre, ofset 410: hata 1.84e-03 > `tail_gate` (|rho| = 1.000009, birim
    kapisi da gormez); ayni 2986 durumda tepe oranla gecenlerin 31'inin
    hatasi 1.6e-3'u asiyordu, hata / olcut 1.19'a cikiyordu.
    ESIK OLCULEN BANTTAN (`DEFAULT_ECHO_GATE`): kapisi acik sabitlenmis
    cagrilarin en buyuk c'si 2.02e-04 (`probe_offset` 300, eps 4 -> 1,
    theta 45, lam 40: yanki kayda tepesinden girer), kuyruk kapisindan gecen
    hatasi > 1e-3 sayilarin en kucugu 9.84e-04 (9 -> 1, theta 52, lam 40, p,
    sevk ofset 880); 4.5e-4 ikisinin geometrik ortasi. Eski kor nokta
    testinin durumu (4 -> 1, theta 0, lam 20, 1800 hucre, ofset 400: c
    1.02e-03, hata 3.56e-04) bandin USTUNDEDIR ve artik `None`. Bedel (ayni
    2986 durum): kapi 514 sayiyi kesti, 445'inin hatasi > 1e-3; ikinci
    varistan 3 yanlis alarm (en kotusu 3600 hucre, ofset 150, 4 -> 9,
    theta 0, lam 40, p: eski hata 9.7e-05), sevk izgarasinda (336 durum)
    kayip YOK, sabitlenmis cagrilardan yalniz eski kor nokta durumu `None`.
    DURUST OLUMSUZ: esigin altindaki yanki GECER ve hatasi ~|r| * c'dir
    (ornek 1800 hucre, ofset 410, 4 -> 1, theta 40, lam 20, s: c 2.25e-04,
    hata 2.23e-04). IKI KAYNAK TOPLANIR (ikinci dogrulayici buldu): c ve
    kuyruk her biri kendi esiginin altindayken hata ikisinin TOPLAMIDIR --
    1800 hucre, ofset 430, 4 -> 1, theta 32, lam 24, s: c 4.45e-04, kuyruk
    1.06e-03, hata 6.16e-04 (> `echo_gate`); theta 32.4, lam 22'de hata /
    max(kuyruk, c) 1.108. Olculen toplam sinir (769 gecen sayi, iki
    dogrulayicinin avi dahil, dikdortgen pencere): hata <= |r| c / (1 - c)
    + 0.17 * kuyruk. Yani gecen bir sayinin protokol hatasi, prob nerede
    olursa olsun, <~ `echo_gate` + 0.17 * `tail_gate` = 7.2e-04 diye OKUNUR
    (olculen en kotu 6.16e-04) -- olculen bir oran, kanit degil. Kapi yeni
    kosum eklemez; gecen her sayi eski sayinin BIT BIT aynisidir.
    ELENEN ADAY (olculdu): gelen kaydini yankidan once SIFIRLAMAK (zaman
    kapisi). Yanki ana darbeyle 5 sg icinde ortusuyorsa kesim darbenin
    kendi kuyrugunu keser: sevk penceresinin ana cipasi (4 -> 1, theta 45,
    lam 40) 1.2e-07'den 1.25e-05'e bozuldu; kapisi acik 62 sabitlenmis
    cagrinin 26..27'si degisti, 10..11'i `None` oldu (yalniz gelen / iki
    kayit birden kesilince).
    FFT PENCERESI -- YALNIZ DIKDORTGEN (RC-d, 10.10.2026; ikinci
    dogrulayici buldu, RC-a'dan ONCE de vardi). `window_fn` hann ya da
    blackman iken pencere erken gelen darbeyi ve gec gelen yansiyani FARKLI
    agirlikla carpiyordu ve uc kapi da GORMUYORDU: 1800 hucre, ofset 6,
    4 -> 2.25, theta 40, lam 20, s'de hann hata 2.56e-02, blackman
    3.56e-02, dikdortgen 1.4e-07 (kuyruklar 4.8e-06 / 1.7e-05; ikinci
    varis hann'da 2.4e-09, dikdortgende 8.6e-07).
    MEKANIZMA (olculdu): hata ~ |r| |w(t_y) / w(t_g) - 1|; w pencere, t_g
    ve t_y gelen ve yansiyan kaydin enerji agirlik merkezleri. 638 durumda
    hata / tahmin medyani 0.95 (hann; %10..%90 0.93..1.08), blackman'da
    0.91. BASKIN TERIM DEGISEBILIR: |r| -> 0 (Brewster yakini p) ya da
    w(t_y) ~ w(t_g) iken r(f)'nin dagitkanligi baskindir, hata ~
    |r (w_y/w_g - 1) + (i/2 pi)(w'_y/w_g) r'_dag| (r'_dag saf gecikmesi
    cikarilmis r'nin f-turevi). 638'in 65'i +-%15 bandinin disinda (en
    uzak oran 30); iki terimli sinir 645/646 tutar, kalan r = 0
    (`rc-olcum/rcd/dogrulama/mekanizma2.log`). Pencere kaydin sonunu
    sifira cektigi icin yanki olcutu c de kuculur: dikdortgende c >=
    5.1e-04 (> esik) olan 32 kayitta hann'da c <= 3.7e-04 idi ve sayi
    donuyordu, hepsinin hatasi > 1e-3 -- ama bu hata yankidan degil yine
    pencere agirligindan gelir (dikdortgenin ayni kayitlardaki kapisiz
    hatasi medyan 2.5e-04).
    ON OLCUM (`.superpowers/sdd/rc-olcum/rcd/ozet_rcd.log`; RC-a'nin 3005
    durumu + dogrulayicinin 36 hann durumu; kapisi acik ve referansi
    gecerli 2803 satir, 2720 tekil durum -- 83 satir ayni kosumun
    tekrari). `window_fn` FDTD kosumunu DEGISTIRMEZ: uc pencere ayni kosum
    ciftinden okundu, RC-d'den ONCEKI kamu yoluyla 84/84 bit bit ayni
    (bugun bunlarin 56'si reddedilir). Dikdortgen 646 sayi, en kotusu
    3.21e-04; hann'da uc kapidan gecen 408 sayinin 406'sinin hatasi > 1e-3
    (en kotu 0.48), blackman'da 416'nin 414'u. Ikisinin de sayi verdigi 376
    durumun HICBIRINDE pencere dikdortgenden iyi degil. 1e-3'un altinda
    kalan iki pencereli sayinin ikisinde de |r| ~ 0: arayuzsuz durum
    (r = 0) ve Brewster'da bir p durumu (5 -> 4, theta 41.81, lam 30:
    |r| 2.9e-03, hann hatasi 4.5e-04, bagil %16). Tam kayitta (kuyruk
    kapisi) dikdortgenin orani dogrusal sistemin TAM transfer oranidir;
    pencere bilgi eklemez, iki darbenin agirligini ayirir.
    KARAR: `window_fn` != "dikdortgen" -> `InvalidInputError`, kosumdan
    ONCE; bilinmeyen adlar (`kaiser`) da ayni kuraldan reddedilir. `None`
    DEGIL: hata kayda degil PENCEREYE baglidir (`grid.dx_kapisi` gibi bir
    yapilandirma hatasi); `None` bir merdiveni (`tir_phase_ladder`)
    sebebi gorunmeden oldururdu. Bedel: depoda hann / blackman ile tek
    cagri bu maddenin eski durust olumsuz testiydi.
    ELENEN ADAYLAR (ayni olcum): yansiyanin penceresini gecikme kadar
    KAYDIRMAK ya da her serinin penceresini KENDI darbe merkezine
    ORTALAMAK -- en iyi aday bile yuzlercesini gecirir (hann; olculen
    gecikmeyle kaydirma 679 sayinin 238'ini, olculen merkezlere ortalama
    1122'nin 452'sini > 1e-3 hatayla; en kotu 2.1e-02 / 9.5e-03), cunku
    oran saf bir gecikme degildir (yukaridaki ikinci terim); YALNIZ SON
    PERIYOTTA KONIK -- dikdortgenle ayni 646 kaydi gecirir, sayilari ondan
    en fazla 1.1e-04 farklidir (medyan 2.4e-07, en kotu hata yine
    3.21e-04): "hann" etiketi altinda dikdortgen = sessiz ikame.
    ACI VE COZUNURLUK BAGIMLILIGI (kapi `None` der; olculen acilar 0..85):
    SIYIRMA -- n1 = 1, lam 10: theta >= 60 (1 -> 2.25, p'de 55 de: Brewster
    yakininda yanlis alarm, eski hata 2.96e-06); lam 20: 1 -> 2.25'te >= 65,
    1 -> 4'te 60 gecer, 70 gecmez; eps1 2.25: lam 20'de >= 60, lam 40'ta
    >= 70; eps1 4: lam 20'de >= 60, lam 40'ta >= 65. KRITIK BANT -- 2.25 ->
    1: 40..43 (35 ve 45 gecer), 4 -> 1: 29.5..30.5 (28 ve 32 gecer), lam 20
    ve 40'ta. KABA IZGARA -- eps1 4, lam 10'da HER aci; eps1 2.25, lam
    10'da 45 disinda her aci (ortamda <= 7 hucre/dalgaboyu). UZUN DARBE --
    n1 = 1, lam 40'ta theta <= 30 guvenli kayit yok (`_record_steps`),
    theta >= 45 kaynak yankisi. Bu tablo SEVK PENCERESININDIR: geometri
    (iface, kaynak, prob) ve `sgf` degisince sinirlar kayar.
    ELENEN DUZELTME ADAYI (olculdu): yankiya kadar uzun kosup kuyrugu
    esigin altina inen ILK kaydi kullanmak. Civilti (chirp) ve dal noktasi
    kuyruklarinda kuyruk bir periyot susup geri gelir: tau = 3e-4'te kabul
    edilen bir kaydin hatasi 3.35e-03 (eps 4 -> 1, theta 29.9, lam 10,
    kuyruk 2.99e-04). Kaydin SONU T_min'de bilgi tasir cunku darbe orada
    yeni bitmis olmalidir; uzun kaydin ilk sessiz ani tasimaz. Kapi bu
    yuzden YENI KOSUM EKLEMEZ ve gectigi her sayi eski sayinin BIT BIT
    aynisidir. ACIK: `vx` ve darbe genisligini dagilim/ayrik grup hiziyla
    duzelten bir `_record_steps` OLCULMEDI.

    MALIYET (olculdu, sevk penceresi): theta = 45, lam = 40'ta cagri basina
    1.07 s; theta = 70'te 1.543 s; theta = 80'de 2.416 s (kayit uzunlugu
    1/cos(theta) ile buyudugu icin).

    FAZ REFERANSI, BEYAN EDILIR (G1, 09.10.2026): donen `rho`nun fazi PROB
    duzlemine (`iface - probe_offset`) referanslidir ve klasik Fresnel
    uzlasiminin ESLENIGIDIR (bkz. `fresnel_phase_difference`). Kapali
    formla `rho = conj(r(I-1/2) * exp(-2 i k1 (prob - (iface - 1/2))))`;
    k1 semanin KENDI ayrik dalga sayisidir. |rho| duzlemden bagimsizdir,
    faz degildir: iki semanin (egik `ky` ile 2B esdegeri `2 sin(ky/2)`)
    fazini karsilastiran biri duzlemi beyan etmek ZORUNDADIR, cunku
    k1'leri farkli oldugundan her duzlem kaymasi ikisine farkli faz ekler.
    OLCULDU (eps 2.25 -> 1, theta 60, lam 20, sevk penceresi; 2B esdegeri
    theta' = 59.31973 ile okundu): arg(rho(theta)/rho(theta')) = +5.2266
    der. Kapali form AYNI (rho, eslenik) uzlasimda: prob duzleminde
    +5.2218, arayuz yuzunde (I-1/2) +2.1489 der; klasik uzlasimda ikisi de
    eksi (-5.2218 / -2.1489). Bu iki duzlemde isaret ayni, buyukluk
    farkli. Kapali formda lam 20.78'de ayni fark I-1/2'de 1.99, prob
    duzleminde 4.73 der (mutlak deger). Bu aletle okunan egik/2B faz
    bedeli I-1/2'de yazilanin ~2.4 katidir.
    KUYRUK KAPISINDAN SONRA (10.10.2026): yukaridaki +5.2266 ancak kapi
    KAPALIYKEN (`tail_gate` = 1) uretilir; varsayilan pencerede iki cagri da
    `None` doner (kuyruk 4.63e-03 ve 2.13e-03; ayrik-tam referansa hata
    2.96e-04 ve 1.39e-04). Kapali formla fark (+5.2266 - 5.2218 = 0.0048
    der) o protokol hatasidir.
    """
    if polarization not in POLARIZATIONS:
        raise InvalidInputError(
            f"polarization {POLARIZATIONS} icinden olmali, verilen: "
            f"{polarization!r}"
        )
    w = ObliqueWindow() if window is None else window
    if w.window_fn != "dikdortgen":
        raise InvalidInputError(
            "reflection_coefficient yalniz window_fn 'dikdortgen' ile olcer "
            f"(verilen: {w.window_fn!r}): pencere gelen ve yansiyan darbeyi "
            "FARKLI agirlikla carpar, kapilar bunu gormez (RC-d, olculdu)"
        )
    if not 0 < w.interface < w.cells:
        raise InvalidInputError(
            f"interface 1 ile {w.cells - 1} arasinda olmali, verilen: {w.interface}"
        )
    steps = _record_steps(
        theta_deg=theta_deg,
        wavelength=wavelength,
        eps1=eps1,
        window=w,
        eps2=eps2,
        mirror=mirror,
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
    kuyruk = max(
        _kuyruk(gelen, wavelength=wavelength, courant=w.courant),
        _kuyruk(yansiyan, wavelength=wavelength, courant=w.courant),
    )
    if kuyruk > w.tail_gate:
        return None
    yanki = _ikinci_varis(
        gelen, theta_deg=theta_deg, wavelength=wavelength, eps1=eps1, window=w
    )
    if yanki > w.echo_gate:
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

    `dx = 1` ARTIK BIR UZLASIM DEGIL BIR KAPIDIR (`engine/em/grid.dx_kapisi`)
    ve bu merdivenin ON KOSULUDUR: `dx` bir dugme olsaydi "artan lambda =
    incelen izgara" cumlesi DUSERDI, cunku cozunurlugu `lambda/dx` belirler
    ve ikisi ayri ayri oynatilabilirdi.

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


# ===========================================================================
# KIP SPEKTRUMU -- 2B boslugun kip frekanslari
# ===========================================================================
#
# BU BOLUM BESSEL KOKLERINI BILMEZ ve bilmemesi CIPANIN KOSULUDUR. Burada
# doner sey ADSIZ TEPE FREKANSLARIDIR; hangi tepenin `j_{0,1}`, hangisinin
# `j_{1,1}` oldugu TESTTE eslenir. Eslestirme ile DEGER birbirinden ayridir:
# eslestirmenin penceresi %1 mertebesinde, cipanin hukmu 1e-5 mertebesinde --
# arada 1000 kat var, yani eslestirme hukmu URETEMEZ. (Ayni ayrim Modul 3'te
# "tautoloji-cipa" olarak olculmustu.)

#: Tepe kapisi: en buyuk tepenin bu kesrinden kucuk yerel maksimumlar tepe
#: SAYILMAZ. Tolerans degil, PENCERE DUGMESI -- ve iki UCU DA OLCULDU
#: (bkz. `mode_spectrum` ve WINDOW_SIDELOBE).
DEFAULT_PEAK_FLOOR: float = 0.01

#: Her FFT penceresinin OLCULEN en buyuk yan lobu, ana lobun tepesine gore.
#: Pencerenin KENDI donusumunden okundu (n = 4096 ve 8192, 64 kat sifir dolgu;
#: iki n'de de ayni sayi):
#:     dikdortgen  0.217178  (-13.3 dB)   ana lob yari genisligi 1 bin
#:     hann        0.026707  (-31.5 dB)                          2 bin
#:     blackman    0.001243  (-58.1 dB)                          3 bin
#: BU SAYILAR BIR TABLODAN ALINMADI, OLCULDU -- ve `mode_spectrum`un
#: IMKAN KAPISI onlarin uzerine kurulu: `peak_floor` yan lobun ustunde
#: degilse, bulunan "tepe"lerin bir kismi KIP DEGIL PENCERE ARTIFAKTIDIR.
WINDOW_SIDELOBE: dict[str, float] = {
    "dikdortgen": 0.217178,
    "hann": 0.026707,
    "blackman": 0.001243,
}

#: FFT pencereleri. "dikdortgen" = pencere UYGULAMAMAK, ve bu da bir secimdir.
WINDOW_FNS: tuple[str, ...] = tuple(WINDOW_SIDELOBE)


@dataclass(frozen=True)
class ModeSpectrum:
    """Bir 2B kosumun kip spektrumu -- ve TAM ANALIZ PENCERESI.

    `frequencies`: tepe frekanslari, ARTAN sirada, `c = 1` biriminde
    (dolayisiyla `omega = 2 pi f` ve `j = 2 pi R f`).
    `amplitudes`: her tepenin dedektorler uzerinden toplanmis genligi.

    Geri kalan alanlar ANALIZ PENCERESIDIR ve hepsi ZORUNLUDUR: bu modulde
    tohumun karsiligi pencerenin tam beyanidir (bkz. `grid.Window`), ve
    spektral bir sayinin `pad`, pencere fonksiyonu, dedektor sayisi ve kayit
    uzunlugu soylenmeden yazilamayacagi bu modulde ON BIR kez olculdu.
    """

    frequencies: tuple[float, ...]
    amplitudes: tuple[float, ...]
    bin_width: float
    raw_bin_width: float
    pad: int
    window_fn: str
    peak_floor: float
    detectors: int
    steps: int

    def ratios(self) -> tuple[float, ...]:
        """Tepe frekanslarinin EN DUSUGUNE oranlari.

        Cipa BU sayilar uzerine kurulur, `frequencies` uzerine DEGIL: oran
        `R`den, `c`den ve `eps_r`den bagimsizdir, tekil frekans ise DEGILDIR.
        """
        if not self.frequencies:
            return ()
        f0 = self.frequencies[0]
        return tuple(f / f0 for f in self.frequencies)

    def as_lines(self) -> list[str]:
        return [
            f"tepe sayisi : {len(self.frequencies)}",
            f"dedektor    : {self.detectors}",
            f"pad         : {self.pad} (bin {self.raw_bin_width:.6g} -> "
            f"{self.bin_width:.6g})",
            f"FFT penceresi: {self.window_fn}",
            f"tepe tabani : {self.peak_floor:g}",
        ]


def _pencere_dizisi(window_fn: str, n: int) -> np.ndarray:
    if window_fn == "dikdortgen":
        return np.ones(n)
    u = 2.0 * np.pi * np.arange(n) / n
    if window_fn == "hann":
        # Periyodik (DFT-simetrik) Hann: `np.hanning` SIMETRIK olani verir ve
        # spektral sizinti icin dogru olan periyodik olanidir.
        return 0.5 - 0.5 * np.cos(u)
    if window_fn == "blackman":
        return 0.42 - 0.5 * np.cos(u) + 0.08 * np.cos(2.0 * u)
    raise InvalidInputError(
        f"window_fn {WINDOW_FNS} icinden olmali, verilen: {window_fn!r}"
    )


def mode_spectrum(
    record,
    *,
    pad: int,
    window: str,
    peak_floor: float = DEFAULT_PEAK_FLOOR,
    interpolate: bool = True,
) -> ModeSpectrum:
    """Kayittan kip frekanslari -- ADSIZ tepeler, teori YOK.

    SINIF: OLCUM ALETI. Kendisi ne sema ne fizik cipasidir; Bessel koklerini
    HIC gormez (dosyada gecmezler) ve hangi tepenin hangi kip oldugunu
    soylemez. Cipa `tests/em/test_bessel.py`de kurulur.

    TEPE, KIP DEMEK DEGILDIR -- ve bu iddia Task 5'te ILK KEZ sinanabildi.
    Bu fonksiyonun basligi "kip frekanslari" diyor, ama alet KIPI OLMAYAN
    bir kayitta da tepe dondurur. Olculdu (acik uzayda kucuk bir silindirden
    SACILAN alan; bosluk yok, duran kip yok, ayni kayit uc pencereyle):

        blackman   taban 0.01 : 6 tepe, oranlar 1.000 2.195 2.842 3.456
        hann       taban 0.05 : 6 tepe, oranlar 1.000 1.836 2.350 2.880
        dikdortgen taban 0.30 : 4 tepe, oranlar 1.000 1.234 1.682 1.925

    Uc pencere UC FARKLI oran kumesi veriyor, yani bu sayilar fizik
    TASIMIYOR -- pencerenin kendi tepelerini sayiyorlar. Aletin "adsiz
    tepeler, teori yok" beyani DOGRUYDU; eksik olan, dondurulen tepelerin
    bir KIP oldugunun BURADAN cikarilamayacagi cumlesiydi. Kilit:
    tests/em/test_sacilma.py, `test_MODE_SPECTRUM_BIR_KIP_DEDEKTORU_DEGILDIR`.

    HESAP
        1. Dedektor basina kayit, `window` ile pencerelenir.
        2. `pad` katina SIFIR DOLGU yapilir ve rfft alinir.
        3. Dedektorlerin GENLIKLERI toplanir: `M(f) = sum_d |X_d(f)|`.
           Toplam GENLIKLERINDIR, karmasik degerlerin DEGIL -- karmasik
           toplamda kipin dedektorler arasindaki isaret degisimi (m != 0
           kiplerinde acisal bagimlilik) kipi SONDURURDU.
        4. `M`in yerel maksimumlari, `peak_floor * max(M)` tabaninin ustunde
           kalanlar, tepe sayilir.
        5. `interpolate` ise tepe konumu UC NOKTALI PARABOLLE ara degerlenir.

    `interpolate = False` bir hata ayiklama dugmesi DEGIL, olculen bir
    karsilastirmadir: ara degerleme kalkinca tepe konumu bin izgarasina
    yapisir. BU SATIR BIR ZAMANLAR *"ve merdiven MERTEBESI duser"* diyordu ve
    o cumle OLCUMLE CURUDU -- kilit `tests/em/test_bessel.py::
    test_ARA_DEGERLEME_KALKINCA_HEM_MERDIVEN_HEM_PANEL_DUSUYOR` zaten dogruyu
    yaziyordu, bayat olan bu dokstringdi (Task 6'nin dokstring denetiminde
    bulundu). Ham mertebeler DUSMUYOR, yerinden OYNUYOR (p = 3.3411 ve
    1.4245 -- birincisi 2'nin USTUNDE) ve belirsizlik PATLIYOR (u = 10.0255),
    cunku pencere gurultusu 1.5e-04 ... 1.9e-04'e cikip en kucuk basamak
    farkini (1.1e-04) geciyor. Hukum bu yuzden dusuk bir mertebe degil
    YARGILANAMAZ olur.

    `pad` ZORUNLU ANAHTAR KELIME: sifir dolgu YENI BILGI EKLEMEZ, yalniz
    ayni spektrumu daha sik ORNEKLER; ama ara degerlemenin parabol varsayimi
    ancak yeterli ornekle gecerlidir. Bu yuzden ikisi bir aradadir ve degeri
    beyan edilir. OLCULDU (Task 4, 2B Bessel merdiveni): `pad` 8 -> 1
    yapilinca pencere gurultusu ~1e-06'dan ~1e-04'e cikiyor ve merdivenin
    hukmu OTURDU(2) -> YARGILANAMAZ'a doner. Tek IZGARADA ise `pad = 1` bes
    orandan DORDUNDE dogru borudan DAHA YAKIN cikiyor -- yani bu dugmenin
    bedeli tek noktada GORUNMEZ, yalniz merdivende gorunur.

    `peak_floor`UN IKI UCU DA OLCULDU ve arasi dardir:
      ALT UC  pencerenin yan lobu (WINDOW_SIDELOBE) -- altinda kalirsa
              bulunan tepelerin bir kismi kip degil artifakttir; kapi
              yukarida.
      UST UC  en zayif capalanan kip. Olculdu (R = 80, 8 dedektor): bes
              Bessel oraninin genlikleri en buyuk tepenin 1.000 / 0.952 /
              0.298 / 0.519 / **0.0214**'u. Yani `peak_floor` 0.0214'un
              ALTINDA olmak zorunda.
    Ikisi birlikte SEVK EDILEN PENCEREYI BELIRLER: `hann`in yan lobu
    0.026707'dir ve 0.0214'un USTUNDEDIR -- yani hann ile o kip HICBIR
    mesru tabanda ayrilamaz (olculdu: hann + 0.03 ile bes orandan dordu
    bulunuyor). `blackman`in yan lobu 0.001243 ve taban 0.01 ile BESI DE
    bulunuyor. Planin "Hann, pad = 8" penceresi bu yuzden sevk EDILMEDI.
    """
    if not isinstance(pad, int) or isinstance(pad, bool) or pad < 1:
        raise InvalidInputError(f"pad en az 1 tamsayi olmali, verilen: {pad!r}")
    if not 0.0 < peak_floor < 1.0:
        raise InvalidInputError(
            f"peak_floor (0, 1) araliginda olmali, verilen: {peak_floor}"
        )
    if window not in WINDOW_SIDELOBE:
        raise InvalidInputError(
            f"window {WINDOW_FNS} icinden olmali, verilen: {window!r}"
        )
    # IMKAN KAPISI -- tolerans DEGIL. Yan lob ORANI ana lobun tepesine goredir
    # ve tepe her zaman <= max(M) oldugundan, `peak_floor > yan_lob` kosulu
    # HICBIR yan lobun tepe sayilamayacaginin CEBIRSEL garantisidir.
    yan = WINDOW_SIDELOBE[window]
    if peak_floor <= yan:
        raise InvalidInputError(
            f"peak_floor = {peak_floor} bu pencerenin OLCULEN en buyuk yan "
            f"lobunun ({window}: {yan}) ustunde DEGIL. Bu ayarda bulunan "
            "tepelerin bir kismi kip degil PENCERE ARTIFAKTI olur ve en "
            "dusuk tepe (oranlarin paydasi) SESSIZCE yanlis cikar -- "
            "olculdu: R = 80'de hann + 0.01 ile en dusuk tepe gercek kipin "
            "%0.75 altindaki bir yan lobtu ve BUTUN oranlar o kadar kaydi."
        )
    seri = np.asarray(record.probes, dtype=float)
    if seri.ndim != 2 or seri.shape[0] < 1 or seri.shape[1] < 8:
        raise InvalidInputError(
            f"kayit (dedektor, adim) sekilli ve en az 8 adim olmali, verilen: "
            f"{seri.shape}"
        )
    # SONLULUK KAPISI (11.10.2026). Olculdu (kapi yokken, R = 5 sevk penceresi):
    # temiz kayit 28 tepe; TEK bir NaN/+inf/-inf eklenince dedektor toplami `M`
    # sonlu olmaktan cikiyor, tepe karsilastirmalarinin hicbiri dogru olmuyor ve
    # alet HATASIZ bos demet donduruyordu -- cagiran bunu "tepe bulunamadi" diye
    # okurdu.
    if not np.all(np.isfinite(seri)):
        raise InvalidInputError(
            f"kayit sonlu olmali; {int(np.count_nonzero(~np.isfinite(seri)))} "
            "sonlu olmayan deger var (patlamis bir FDTD kosumu sessizce tepe "
            "URETEMEZ)"
        )
    w = record.window
    dt = float(w.courant) * float(w.dx)
    n = seri.shape[1]
    pencere = _pencere_dizisi(window, n)
    nfft = n * pad
    spek = np.abs(np.fft.rfft(seri * pencere[None, :], n=nfft, axis=1))
    M = spek.sum(axis=0)
    frekans = np.fft.rfftfreq(nfft, d=dt)
    df = float(frekans[1] - frekans[0])

    taban = peak_floor * float(np.max(M))
    ic = M[1:-1]
    tepe_mi = (ic > M[:-2]) & (ic > M[2:]) & (ic > taban)
    idx = np.flatnonzero(tepe_mi) + 1

    fler: list[float] = []
    aler: list[float] = []
    for i in idx:
        a, b, c = float(M[i - 1]), float(M[i]), float(M[i + 1])
        payda = a - 2.0 * b + c
        delta = 0.5 * (a - c) / payda if (interpolate and payda != 0.0) else 0.0
        fler.append(float(frekans[i]) + delta * df)
        aler.append(b)
    return ModeSpectrum(
        frequencies=tuple(fler),
        amplitudes=tuple(aler),
        bin_width=df,
        raw_bin_width=1.0 / (n * dt),
        pad=pad,
        window_fn=window,
        peak_floor=peak_floor,
        detectors=int(seri.shape[0]),
        steps=n,
    )


# ===========================================================================
# UZAK ALAN -- "IKI YONTEM" DEGIL, IKI AYRI GOZLEMLENEBILIR
# ===========================================================================
#
# Bu bolumun cekirdek iddiasi ve OLCUM: `aperture_far_field` ile
# `ntff_far_field` ayni problemi COZMEZLER, dolayisiyla biri otekinin
# "daha iyi" surumu degildir. Ikisi de spec 7.5'te ve bu dosyanin
# dokstringlerinde PENCERESIYLE olculu.
#
#   ACIKLIK (yarik) : ekran her SONLU konturu deler -- kapali kontur
#                     yapisal olarak kurulamaz. Zorlanirsa cevap KONTURUN
#                     YERINE BAGLI olur.
#   SACILMA/ISIMA   : kapali kontur dogru cevabi verir; tek bir SATIRIN
#                     FT'si ise var olmayan bir acisal yapi UYDURUR.
#
# YAY-ALICI (sonlu R'de dedektor yayi) SEVK EDILMEZ (spec 7.5): ucunun en
# pahalisi, en kararsizi, ve Fraunhofer kapisina ihtiyac duyan TEK yontem.
# Sevk edilen iki gozlemlenebilir de R -> sonsuz ASIMPTOTIK deseni dondurur;
# cagiran sonlu bir dedektor yaricapi BEYAN ederse `_uzaklik_kapisi`
# calisir ve gecmezse `None` doner.

#: Bin kapisi: olculen fazorun bini, beyan edilen `wavelength`in USTUNE
#: oturmali. TOLERANS DEGIL, ETIKETLEME KAPISI -- ve neden zorunlu oldugu
#: `scattering_width` dokstringinde SAYIYLA duruyor. Ayni kapi Task 3'te
#: `reflection_coefficient`te de kuruldu (orada `|f_bin*lam - 1| > 1e-9`).
BIN_GATE: float = 1e-9

#: Fraunhofer kapisi: BEYAN EDILEN sonlu dedektor yaricapi en az bu kadar
#: `lambda` olmali. SAYI BIR MODELLEME SECIMIDIR (kappa ile ayni statude) ve
#: bedeli OLCULMUSTUR -- bkz. `_uzaklik_kapisi`. Spec 4.4'un ilk hali
#: `R >= a^2/lambda` yaziyordu; o kosul OLCULEREK YETERSIZ bulundu.
FRAUNHOFER_LAMBDA: float = 100.0


def _acilar(angles) -> np.ndarray:
    a = np.asarray(angles, dtype=float)
    if a.ndim != 1 or a.size < 1:
        raise InvalidInputError(
            f"angles tek boyutlu ve bos olmayan olmali, verilen: {a.shape}"
        )
    if not np.all(np.isfinite(a)):
        raise InvalidInputError("angles sonlu olmali")
    return a


def _bin_kapisi(wavelength: float, bin_frequency: float) -> bool:
    """Fazorun bini beyan edilen `lambda`nin USTUNDE mi?

    IMKAN KAPISI, tolerans DEGIL: `sigma`, `W` ve desen `k = 2 pi / lambda`
    tasir; bin baska bir frekanstaysa dondurulen sayi BASKA BIR DALGABOYUNUN
    sayisidir ve o dalgaboyuyla etiketlenmemistir.
    """
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise InvalidInputError(f"wavelength > 0 olmali, verilen: {wavelength}")
    if not np.isfinite(bin_frequency) or bin_frequency <= 0.0:
        raise InvalidInputError(
            f"bin_frequency > 0 olmali, verilen: {bin_frequency}"
        )
    return abs(bin_frequency * wavelength - 1.0) <= BIN_GATE


def _uzaklik_kapisi(detector_radius, wavelength: float) -> bool:
    """Beyan edilen SONLU dedektor yaricapinda asimptotik desen gecerli mi?

    SPEC 4.4 BU FONKSIYONDA DUZELTILDI. Eski kosul `R >= a^2/lambda` idi ve
    OLCULEREK YETERSIZ cikti: a = 2 hucrelik bir saciciyla `a^2/lambda`
    0.02 - 0.10 arasindadir, yani kapi binlerce kat rahat gecer -- ama ayni
    R'lerde desen HALA yanlistir. BAGLAYICI KOSUL `R >> lambda`dir.

    OLCULDU. Sevk edilen asimptotik desen, AYNI kayittan TAM Green
    fonksiyonuyla sonlu R'de hesaplanmis desenle karsilastirildi; olcu,
    tepesine normalize edilmis |F(theta)| farkinin RMS'idir:

        R/lambda           10        30       100       300      1000
        sacilma lam=40  3.743e-01 1.163e-01 3.305e-02 1.018e-02 2.978e-03
        sacilma lam=60  2.358e-01 7.431e-02 2.096e-02 6.703e-03 1.972e-03
        sacilma lam=100 1.453e-01 4.809e-02 1.412e-02 4.506e-03 1.273e-03
        yarik   lam=16  1.539e-02 1.178e-02 4.286e-03 1.478e-03 4.464e-04

    (sacilma: a = 2, kontur rc = 40; yarik: a = 40.) Uc sacilma satiri da
    1/R gidiyor: ardisik oranlar 3.02 ... 3.55, adim carpanlari ise 3 ve
    3.33. YARIK SATIRINDA ILK ADIM (R/lambda 10 -> 30) BU YASANIN
    DISINDADIR (1.31); sonraki uc adim 2.75 / 2.90 / 3.31 ile yasaya
    giriyor. Sevk edilen esik `R >= 100 lambda`dir ve ORADA olculen artik
    hata sacilmada 1.4e-02 ... 3.3e-02, aciklikta 4.3e-03'tur. Bu bir
    TOLERANS DEGIL, ESIGIN BEDELIDIR ve boyle beyan edilir.

    ESKI KAPI NEDEN YETMIYOR -- SAYIYLA: a = 2 hucrelik saciciyla
    `a^2/lambda` lambda = 40/60/100'de 0.100/0.067/0.040'tir. `R = 10 lambda`
    noktasinda eski kapi sirasiyla 4000 / 9000 / 25000 KAT rahat gecer, ama
    ayni noktada desen %37.4 / %23.6 / %14.5 YANLISTIR. Aciklikta eski
    kapinin TAM SINIRI (`R = a^2/lambda` = 100, yani R/lambda = 6.25) desen
    hatasi 2.141e-02'dir.

    KAPI SEVK EDILEN KULLANIMDA KOSMAZ: `detector_radius = None` verildiginde
    dondurulen sey zaten R -> sonsuz asimptotik desendir ve o desende sonlu-R
    hatasi TANIM GEREGI yoktur. Kapi, sonlu bir dedektor BEYAN eden cagiran
    icindir.
    """
    if detector_radius is None:
        return True
    R = float(detector_radius)
    if not np.isfinite(R) or R <= 0.0:
        raise InvalidInputError(
            f"detector_radius pozitif ve sonlu olmali, verilen: {detector_radius}"
        )
    return R >= FRAUNHOFER_LAMBDA * wavelength


def aperture_far_field(
    field,
    *,
    wavelength: float,
    bin_frequency: float,
    angles,
    coordinates=None,
    spacing: float = 1.0,
    detector_radius=None,
) -> np.ndarray | None:
    """TEK YUZ Rayleigh-Sommerfeld: bir ACIKLIK yuzunden uzak alan.

    SINIF: OLCUM ALETI. Kendisi ne sema ne fizik cipasidir; fizik cipasi
    (tek yarik `a sin(theta_1)/lambda = 1`) `tests/em/test_uzak_alan.py`de
    kurulur. Bu fonksiyon KIRINIM DESENI hakkinda hicbir sey bilmez --
    verilen alanin acisal donusumunu dondurur.

    HESAP (bu boru hattinin ISARET UZLASIMINDA, asagiya bakiniz):

        F(theta) = cos(theta) * SUM_j E_j exp(+i k sin(theta) y_j) dy

    `cos(theta)` RS-I'in (Dirichlet Green fonksiyonu) EGIKLIK CARPANIDIR.
    SIFIRIN YERINI OYNATMAZ ve bu OLCULDU: tek yarik cipasinin 14
    yapilandirmasinda (a/lambda >= 2) `cos` kaldirilinca `a sin(theta_1)/
    lambda` ortalamasi 0.998610 -> 0.997942, std 0.019357 -> 0.019400; tek
    tek fark 5.26e-05 ... 2.94e-03 (ortalama 6.69e-04), yani cipanin kendi
    yayiliminin ALTINDA. Cipa egiklik carpanina KORDUR; carpan desenin
    GENLIGI icin dogrudur, cipanin okudugu ilk sifirin yeri icin degil --
    ve bu, cipanin bir DURUST OLUMSUZUDUR (carpan yanlis konsa bu olcum
    gormezdi).

    ISARET UZLASIMI BIR OLCUMDUR, TERCIH DEGIL. Bu depoda +x'e giden dalganin
    fazoru `e^{-i k x}`dir (Task 3, `fresnel_phase_difference` dokstringinde
    arayuzsuz bir kosumla olculdu), yani zaman uzlasimi `e^{+i w t}` ve
    disari giden dalga `e^{-i k r}`dir. Cekirdek bu yuzden `exp(+i k rhat.r')`
    tasir. Ters uzlasim sevk edilen boruda `ntff_far_field`i taninmaz hale
    getiriyor (bkz. orada).

    `bin_frequency` ZORUNLU: bkz. `_bin_kapisi` ve `scattering_width`.
    Kapiya takilirsa `None` doner.

    `detector_radius` verilirse Fraunhofer kapisi kosar (bkz.
    `_uzaklik_kapisi`); gecmezse `None` doner -- sessizce yanlis bir aci
    basmaz. Bu, Modul 4'un `correlation_length -> None` disiplininin bu
    modulde UCUNCU uygulamasidir (birincisi `_record_steps`, ikincisi
    `reflection_coefficient`).

    NEDEN ACIKLIKTA BU, KAPALI KONTUR DEGIL -- OLCULDU: ekran her SONLU
    konturu deler, yani ekranin akimlarinin bir kismi HER ZAMAN konturun
    disinda kalir. Sonucu tek cumleyle: kapali kutunun cevabi KONTURUN
    YERINE BAGLIDIR. Ayni kosum, degisen tek sey kutunun yari yuksekligi
    (40/60/80/120/160), a = 40, lambda = 16:

        kapali kutu : 0.984953 0.967527 0.971425 0.982944 0.983442
                      -> yayilim 1.743e-02
        bu fonksiyon: 0.964916 0.964916 0.964916 0.965335 0.965362
                      -> yayilim 4.460e-04

    yani 39 KAT. Ikinci pencerede (a = 64) kutunun yayilimi 1.611e-02, bu
    fonksiyonunki 0.000e+00 -- kutu degisirken aciklik satiri BIREBIR ayni
    kaldigi icin.
    """
    E = np.asarray(field, dtype=complex)
    if E.ndim != 1 or E.size < 2:
        raise InvalidInputError(
            f"field en az iki degerli tek boyutlu olmali, verilen: {E.shape}"
        )
    if not np.all(np.isfinite(E)):
        raise InvalidInputError(
            "field sonlu olmali (patlamis bir FDTD kosumu sessizce desen "
            "URETEMEZ)"
        )
    if coordinates is None:
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise InvalidInputError(f"spacing > 0 olmali, verilen: {spacing}")
        y = (np.arange(E.size, dtype=float) - 0.5 * (E.size - 1)) * spacing
        dl = float(spacing)
    else:
        y = np.asarray(coordinates, dtype=float)
        if y.shape != E.shape:
            raise InvalidInputError(
                f"coordinates ile field ayni sekilde olmali, verilen: "
                f"{y.shape} ve {E.shape}"
            )
        d = np.diff(y)
        if not np.all(d > 0.0):
            raise InvalidInputError(
                "coordinates kesin ARTAN olmali; olculen en kucuk fark: "
                f"{float(np.min(d)):.6g}"
            )
        dl = float(np.mean(d))
    th = _acilar(angles)
    if not _bin_kapisi(wavelength, bin_frequency):
        return None
    if not _uzaklik_kapisi(detector_radius, float(wavelength)):
        return None
    k = 2.0 * np.pi / float(wavelength)
    cekirdek = np.exp(1j * k * np.sin(th)[:, None] * y[None, :])
    return np.cos(th) * (E[None, :] * cekirdek).sum(axis=1) * dl


def _kontur(points, normals, lengths, values, normal_derivatives):
    p = np.asarray(points, dtype=float)
    n = np.asarray(normals, dtype=float)
    dl = np.asarray(lengths, dtype=float)
    E = np.asarray(values, dtype=complex)
    dn = np.asarray(normal_derivatives, dtype=complex)
    if p.ndim != 2 or p.shape[1] != 2:
        raise InvalidInputError(f"points (M, 2) sekilli olmali, verilen: {p.shape}")
    m = p.shape[0]
    if m < 4:
        raise InvalidInputError(
            f"kapali bir kontur en az dort ornek ister, verilen: {m}"
        )
    if n.shape != (m, 2):
        raise InvalidInputError(f"normals (M, 2) olmali, verilen: {n.shape}")
    for ad, dizi in (("lengths", dl), ("values", E), ("normal_derivatives", dn)):
        if dizi.shape != (m,):
            raise InvalidInputError(
                f"{ad} (M,) = ({m},) olmali, verilen: {dizi.shape}"
            )
    if not (np.all(np.isfinite(E)) and np.all(np.isfinite(dn))):
        raise InvalidInputError("values ve normal_derivatives sonlu olmali")
    if not (np.all(np.isfinite(dl)) and np.all(dl > 0.0)):
        raise InvalidInputError(
            f"lengths pozitif ve sonlu olmali, en kucuk: {float(np.min(dl)):.6g}"
        )
    boy = np.hypot(n[:, 0], n[:, 1])
    if not np.allclose(boy, 1.0, rtol=0.0, atol=1e-9):
        raise InvalidInputError(
            "normals BIRIM vektor olmali; olculen en buyuk sapma: "
            f"{float(np.max(np.abs(boy - 1.0))):.3g}"
        )
    return p, n, dl, E, dn


def _kapali_mi(p: np.ndarray, n: np.ndarray, dl: np.ndarray) -> bool:
    """Kontur KAPALI mi -- her segment ucu TAM IKI segmentte paylasiliyor mu?

    SPEC 7.5'IN "KOSE DOSEMESI" UYARISININ KAPISI. Dort yuzu `[y1, y2)`
    seklinde yarim acik dosemek her yuzden bir hucre dusurur ve konturda
    DORT KOSE bosluk birakir; alan orada akip gider ve integral artik
    esdegerlik teoreminin integrali DEGILDIR.

    UCUZ VE "BARIZ" KAPI YETMEZ, ve bu OLCULDU. 20 x 20 bir kutuda:

        ornek sayisi            80  ->  76
        max |SUM n_i dl_i|      0.000000e+00  ->  0.000000e+00   (KOR)
        eslesmeyen uc sayisi    0             ->  6              (GORUYOR)

    Yani "kapali konturda `SUM n_i dl_i = 0`" uzaklasma OZDESLIGI bu kusuru
    HIC gormuyor: karsilikli yuzler ayni miktarda kisaliyor ve katkilar
    birebir sadelesiyor. Sevk edilen kapi UC ESLESTIRMESIDIR.

    BEDELI OLCULDU: kusurlu kontur tek yarik olcumunde (n = 14, a/lambda
    >= 2) `a sin(theta_1)/lambda` yayilimini 1.635e-02'den 3.024e-01'e
    cikariyor (18.5 KAT) ve ortalamayi 0.996575'ten 0.825877'ye dusuruyor.
    """
    t = np.stack([-n[:, 1], n[:, 0]], axis=1)
    yari = 0.5 * dl[:, None] * t
    uclar = np.vstack([p - yari, p + yari])
    olcek = float(np.max(np.abs(uclar))) if uclar.size else 1.0
    tol = max(1e-9 * max(olcek, 1.0), 1e-12)
    anahtar = np.round(uclar / tol).astype(np.int64)
    _, sayim = np.unique(anahtar, axis=0, return_counts=True)
    return bool(np.all(sayim == 2))


def ntff_far_field(
    *,
    points,
    normals,
    lengths,
    values,
    normal_derivatives,
    wavelength: float,
    bin_frequency: float,
    angles,
    detector_radius=None,
) -> np.ndarray | None:
    """KAPALI KONTUR yakin-alan/uzak-alan donusumu (2B, TM_z).

    SINIF: OLCUM ALETI -- `aperture_far_field` ile ayni statude, ve ONUN
    RAKIBI DEGIL. Ikisi ayri problemleri cozer (bkz. bolum basligi).

    HESAP (isaret uzlasimi `aperture_far_field`teki gibi OLCULDU):

        I(theta) = KONTUR INTEGRALI
                   [ i k (rhat . nhat) E  -  dE/dn ] exp(+i k rhat . r') dl

    `nhat` DISA bakan normaldir, `dE/dn` da o yonde turevdir. IKI TERIM
    ARASINDAKI BAGIL ISARET FIZIGIN KENDISIDIR ve olculdu. Vakumda merkezi
    bir Ez kaynagi monopoldur, yani dogru uzak alan IZOTROPIKTIR; ayni
    kayittan, yalniz `dE/dn`in isareti cevrilerek olculen |F(theta)|
    yayilimi (std/ort, 121 aci):

        kontur rc = 30 : dogru 0.003004   ters 0.264058   (87.9 kat)
        kontur rc = 45 : dogru 0.005507   ters 0.312722   (56.8 kat)

    Dogru isaretin biraktigi %0.3-0.6 bir HATA DEGIL, izgaranin kendi
    acisal dispersiyonudur (asagida TAM COZUMLE tartiliyor).

    KONTUR KAPALI OLMAK ZORUNDA ve kapi `_kapali_mi`dedir; gecmezse
    `InvalidInputError`. Kapi bir `None` DEGIL cunku acik bir kontur kirik
    bir TAHMINCI degil, GECERSIZ BIR GIRDIDIR -- `mode_spectrum`un yan lob
    kapisiyla ayni statude.

    NEDEN SACILMADA BU, TEK SATIRIN FT'SI DEGIL -- OLCULDU. Kucuk bir TM_z
    dielektrik silindirin (a = 2 hucre, eps_r = 4) uzak alani neredeyse
    izotropiktir. Ayni kayittan okunan normalize |F(theta)| yayilimlari
    (std/ort) ve TAM COZUMUNKI:

        ka             0.094  0.126  0.157  0.199  0.251  0.314  0.398
        bu fonksiyon   0.0011 0.0020 0.0048 0.0066 0.0095 0.0157 0.0231
        TAM COZUM      0.0015 0.0027 0.0041 0.0064 0.0098 0.0146 0.0222
        tek satir FT   0.2130 0.3762 0.4697 0.4164 0.3742 0.4602 0.4042

    Bu fonksiyonun TAM COZUMDEN RMS farki 4.0e-04 ... 2.5e-03; tek satirin
    FT'sininki 2.121e-01 ... 4.678e-01, yani 100 ile 500 kat arasi. Ustelik
    bu fonksiyonun kalan anizotropisi bir HATA DEGIL: tam cozumun kendi
    anizotropisiyle AYNI HANEDE ve ayni yonde gidiyor.

    `detector_radius` ve `bin_frequency` kapilari `aperture_far_field`
    ile AYNIDIR ve ayni yerden kosar.
    """
    p, n, dl, E, dn = _kontur(points, normals, lengths, values, normal_derivatives)
    if not _kapali_mi(p, n, dl):
        raise InvalidInputError(
            "kontur KAPALI DEGIL: segment uclari ikiser ikiser eslesmiyor. "
            "Dort yuzu `[y1, y2)` gibi yarim acik dosemek her kosede bir "
            "hucre dusurur ve esdegerlik teoreminin integrali kurulmaz. "
            "UYARI: `SUM n_i dl_i = 0` uzaklasma ozdesligi bu kusuru "
            "GORMEZ (olculdu: kusurlu konturda o toplam da tam sifir)."
        )
    th = _acilar(angles)
    if not _bin_kapisi(wavelength, bin_frequency):
        return None
    if not _uzaklik_kapisi(detector_radius, float(wavelength)):
        return None
    k = 2.0 * np.pi / float(wavelength)
    rhat = np.stack([np.cos(th), np.sin(th)], axis=1)
    izdusum = rhat @ n.T
    faz = np.exp(1j * k * (rhat @ p.T))
    return (
        (1j * k * izdusum * E[None, :] - dn[None, :]) * faz * dl[None, :]
    ).sum(axis=1)


def scattering_width(
    far_field,
    *,
    angles,
    wavelength: float,
    bin_frequency: float,
    incident_amplitude: complex,
) -> float | None:
    """TOPLAM SACILMA GENISLIGI `W` [uzunluk] -- 2B'nin sacilma kesiti.

    SINIF: OLCUM ALETI. Fizik cipasi (Rayleigh ussu `W ~ lambda^-p`)
    `tests/em/test_sacilma.py`de kurulur; bu fonksiyon uslerden hicbir sey
    bilmez.

    HESAP. `ntff_far_field`in dondurdugu `I(theta)` ile sacilan alan
    2B'de `E_s = A(k, r) I(theta)`, `A = (i/4) sqrt(2/(pi k r)) e^{-i(kr-pi/4)}`
    (bu borunun uzlasiminda disari giden dalga `e^{-ikr}`). Dolayisiyla

        d sigma(theta) = 2 pi r |E_s|^2 / |E_i|^2 = |I|^2 / (4 k |E_i|^2)
        W = (1/2pi) INTEGRAL_0^{2pi} d sigma(theta) d theta
          = INTEGRAL |I(theta)|^2 d theta / (8 pi k |E_i|^2)

    `k` USSU BIR SUSLEME DEGIL, USSUN KENDISINI TASIYOR: `1/k ~ lambda`
    oldugu icin `k` carpani dusurulurse olculen `p` TAM BIR BIRIM kayar --
    olculdu, ayni kayitta 3.242894 yerine 4.242894 (fark 1.000000000). Ve
    o sayi TAM OLARAK 3B'nin ussudur: bu tek carpan yanlis yazilsaydi modul
    "3B bulduk" derdi.

    `bin_frequency` ZORUNLUDUR ve KAPI (bkz. `_bin_kapisi`) bu fonksiyonun
    en sert parcasidir. SEBEP OLCULDU: `W` FFT bininin GERCEK frekansinda
    olculur, ama cagiran NOMINAL bir `lambda` ile etiketlerse us kayar.
    Sevk penceresinde nominal kume {128, 100, 80, 64, 50, 40, 32}
    istendiginde binler gercekte {133.333, 100, 80, 63.158, 50, 40, 32.432}
    (en buyuk kayma %4.17) ve olculen us 3.248311 yerine 3.290854 cikiyor
    (+0.042543). Kapinin dogru kullanimi `lambda`yi BIN IZGARASINDAN
    secmektir (`lambda = T/j`); o zaman kapi tam gecer. Sevk penceresinde
    kapi yedi nominal `lambda`nin UCUNU reddediyor -- ve bir us fiti butun
    noktalari istedigi icin fit KURULAMIYOR.

    VE BU ARIZA COZUNURLUK MERDIVENININ TEK BASINA GOREMEYECEGI BIR
    ARIZADIR: `lambda` ile `dx` BIRLIKTE olceklendigi icin bagil bin kaymasi
    merdivende sabit kalir. Olculdu -- nominal etiketlemenin us hatasi dort
    basamakta +0.029448 / +0.032410 / +0.031798 / +0.031639, yani merdiven
    mertebeleri -0.4295 / +0.0854 / +0.0225. Hata sifira yakinsamiyor,
    ilk adimda BUYUYOR. Kilitleyen sey merdiven degil BIN KAPISIDIR.

    `angles` [0, 2pi) uzerinde DUZGUN bir izgara olmali; integral orta
    nokta kuralidir. Duzgun olmayan izgara reddedilir -- `d theta`nin tek
    bir sayi oldugu varsayimi sessizce yanlis bir `W` uretirdi.
    """
    I = np.asarray(far_field, dtype=complex)
    th = _acilar(angles)
    if I.shape != th.shape:
        raise InvalidInputError(
            f"far_field ile angles ayni sekilde olmali, verilen: "
            f"{I.shape} ve {th.shape}"
        )
    if th.size < 8:
        raise InvalidInputError(
            f"tam cember icin en az 8 aci gerekir, verilen: {th.size}"
        )
    d = np.diff(th)
    dth = 2.0 * np.pi / th.size
    if not np.allclose(d, dth, rtol=1e-9, atol=1e-12):
        raise InvalidInputError(
            "angles [0, 2pi) uzerinde DUZGUN bir izgara olmali (adim "
            f"{dth:.9g}); verilen adimlarin en buyuk sapmasi "
            f"{float(np.max(np.abs(d - dth))):.3g}. Duzgun olmayan izgarada "
            "orta nokta kurali sessizce yanlis bir W uretirdi."
        )
    if not np.all(np.isfinite(I)):
        raise InvalidInputError("far_field sonlu olmali")
    Ei = complex(incident_amplitude)
    if not np.isfinite(Ei.real) or not np.isfinite(Ei.imag) or Ei == 0:
        raise InvalidInputError(
            f"incident_amplitude sonlu ve sifirdan farkli olmali, verilen: {Ei}"
        )
    if not _bin_kapisi(wavelength, bin_frequency):
        return None
    k = 2.0 * np.pi / float(wavelength)
    return float((np.abs(I) ** 2).sum() * dth / (8.0 * np.pi * k * abs(Ei) ** 2))
