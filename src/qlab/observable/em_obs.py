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

from dataclasses import dataclass

import numpy as np

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
