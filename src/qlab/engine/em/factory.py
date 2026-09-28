"""Senaryo -> Maxwell FDTD kosumu. PANEL BEYANININ TEK KAYNAGI BURASIDIR.

`run.py::_run_em` bu metinlerin HICBIRINI yeniden yazmaz; yalnizca
`Provenance.as_lines()` ciktisini basar. Gerekcesi bu depoda OLCULEREK
ogrenildi: ayni beyan blogu bir kez `scf.py` ve `tb/factory.py`de birebir
KOPYALANMISTI ve biri duzeltilip digeri unutuldugunda panel iki farkli sey
iddia ediyordu (bkz. tests/hf/test_hf_scenario.py). Buradaki kopya sayisi
`tests/em/test_em_senaryo.py::test_BEYAN_TEK_KAYNAKTAN_geliyor` ile
kilitlidir: her beyan cumlesi `src/` agacinda TAM BIR KEZ gecer.

IKI ADIMLI FABRIKA -- VE NEDEN. `build_tb`/`build_downfold` (model, provenance)
ciftini TEK cagrida dondurur; burada olmuyor, cunku panelin ZORUNLU YAKINSAMA
TANISI (merdiven, p_inf +- u, uc degerli hukum) `observable/em_obs.py`den
gelir ve `engine` katmani `observable`i IMPORT EDEMEZ (import-linter
sozlesmesi: run -> viz -> observable -> engine -> scenario). Bu yuzden desen
Modul 4'un `spin/factory.py`sininkidir: `build_em` fizigi kosar,
`provenance_for` olculmus tanilari ALIR ve beyani kurar. Sayilar disaridan
gelir, METIN burada kalir.

DAIRESELLIK BEYANI YENIDEN OLCULDU (26.08.2026, bu depoda, float64). Yol
haritasinin "bilinen yansima/kirilma katsayilari" hedefi bir TAUTOLOJIDIR:
dik geliste `R = ((1-n)/(1+n))^2` ve `n` GIRDIDIR, dolayisiyla `R`den `n` tam
olarak geri okunur:

    n = 1.33 -> R = 0.020059312200 -> geri okunan n = 1.330000000000 (2.220e-16)
    n = 1.5  -> R = 0.040000000000 -> geri okunan n = 1.500000000000 (2.220e-16)
    n = 3.4  -> R = 0.297520661157 -> geri okunan n = 3.400000000000 (4.441e-16)

Modul 3F'nin silisyum bant araligi ve Modul 4'un demir T_c'si ile AYNI
AILENIN UCUNCU UYESI. Bu yuzden Fresnel bir fizik cipasi degil CEVIRI
DENETIMI olarak sevk edilir ve panelde oyle etiketlenir.

BU DOSYA CIPA KURMAZ. Sevk edilen senaryonun isi bir cipa kurmak degil,
PANELIN BEYANINI gostermektir (Modul 4'un `spin_tarama.yaml`inda ayni cumle
yazili); cipalarin olcumu `tests/em/` altindadir ve bu senaryo onlari
YENIDEN OLCMEZ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qlab.engine.em.boundary import CircularPEC
from qlab.engine.em.grid import FieldRecord, Window, courant_limit
from qlab.engine.em.medium import Medium
from qlab.engine.em.source import ricker
from qlab.engine.em.yee import run_2d
from qlab.errors import InvalidInputError
from qlab.provenance import Provenance
from qlab.scenario.schema import EMScenario

#: Hukum verilmemis bir merdivenin panel isareti. TEK bir sabit olarak durur
#: ve `tests/em/test_em_senaryo.py` onu BURADAN import eder; iki yerde elle
#: yazilsaydi biri degistiginde test ile panel farkli sey soylerdi -- bu
#: modulun dokstringindeki kopya dersinin aynisi, cumle olceginde. Modul
#: 4'un `ALT_SINIR_ISARETI`yle ayni statude.
HUKUM_VERILMEDI = "hicbir tamsayi mertebe SERTIFIKA EDILMEDI"

_EM_YAKLASIMLAR: tuple[str, ...] = (
    "klasik elektrodinamik: durum vektoru (E, H)dir, kuantum operatoru "
    "DEGIL; guncelleme denklemlerinde ne foton ne hbar gecer",
    "c = eps_0 = mu_0 = 1 BIR BIRIM SECIMIDIR, FIZIK DEGIL: uzunluk hucre, "
    "zaman S*dx cinsindendir ve bu modulun hicbir ciktisi metre, saniye ya "
    "da hertz tasimaz (Modul 4'un k_B = 1'iyle AYNI STATUDE)",
    "dogrusal, dispersif olmayan, izotropik ortam: eps gercek, yalniz "
    "konuma bagli ve frekanstan BAGIMSIZDIR",
    "sonlu izgara ve sonlu kosum: her sayi bir ayriklastirmanin ve sonlu "
    "bir kayit uzunlugunun icinden okunur; surekli limit ALINMAZ, yalniz "
    "merdivenle OTELENIR",
    "2B: sevk edilen cozucu 1B ve 2B'dir, bu senaryo 2B kosar",
)

_EM_IHMAL: tuple[str, ...] = (
    "kuantum elektrodinamik: alan nicelenmemistir, vakum dalgalanmasi ve "
    "kendiliginden isima YOKTUR",
    "dispersiyon ve sogurma: eps(omega) modellenmez; sevk edilen tek kayipli "
    "katman SOGURUCU SINIRDIR, bir malzeme degil",
    "dogrusal olmayan optik: alan siddetine bagli hicbir terim yoktur",
    "3B: bu modul 2B'dir ve 3B BILINCLI OLARAK kapsam disidir -- olculdu, "
    "ayni pencerede D = 167 kupte adim basina 228.72 ms ve 213.2 MiB, yani "
    "2B'nin 16.0 saniyesi 512 dakikaya cikiyor (~1920 kat)",
    "gercek malzeme verisi: hicbir optik sabit veri tabani yoktur",
    "Zeeman yarilmasi -- ve sebebi kapsam tercihi DEGIL, YONTEM/HEDEF "
    "UYUSMAZLIGIDIR: Maxwell FDTD klasiktir, denklemlerinde ne enerji "
    "seviyesi ne spin ne hbar vardir, dolayisiyla bir meV yarilmasi bu "
    "yontemin CIKTI UZAYINDA DEGILDIR. Yontemin gercekten urettigi sey "
    "manyetize Lorentz osilatorudur ve normal kipleri "
    "omega_pm = sqrt(omega_0^2 + omega_c^2/4) -+ omega_c/2, yani yarilma "
    "TAM omega_c (uc (omega_0, omega_c) ikilisinde yarilma/omega_c = "
    "1.000000000000): bu YORUNGESEL g = 1'dir ve spin Zeeman'i g = 2 ister",
)

_EM_BEKLENEN_HATA = (
    "izgara dispersiyonu bu semanin baskin hatasidir ve katsayisi OLCULDU "
    "(olcum tests/em/test_sema_cipalari.py ve tests/em/test_bessel.py'dedir; "
    "bu senaryo onlari YENIDEN OLCMEZ): 1B'de (1 - S^2)/C = 24.000002 / "
    "24.000000 / 23.999999 / 24.000001 (S = 0.3/0.5/0.7/0.9), ve bant m "
    "ciftine baglidir (m = 2,4 -> 24.00033; m = 8,16 -> 24.000000). 2B'de "
    "bu tek sayi degil BIR YASADIR: 24*C(theta) = cos^4 theta + sin^4 theta "
    "- S^2 (Richardson k -> 0 ile 4.6e-07 icinde) ve 1B'nin 24'u yasanin "
    "theta = 0 noktasidir. Her FIZIK cipasinin kendi olculmus yakinsama "
    "hukmu AYRI SATIRDADIR (bkz. Yakinsama tanisi), cunku bir hata sinirini "
    "yazmakla bir mertebeyi sertifika etmek ayni sey DEGILDIR."
)

_DAIRESELLIK_BEYANI = (
    "Bu modul bir malzemenin yansima katsayisini ONGORMEZ: R, girilen n'in "
    "cebirsel sonucudur ve n'e BIRE BIR geri cevrilebilir. OLCULDU (float64, "
    "eps = 2.22e-16): n = 1.5 -> R = 0.040000000000 -> geri okunan "
    "n = 1.500000000000; ayni kadran n = 1.33 (R = 0.020059312200) ve "
    "n = 3.4 (R = 0.297520661157) icin de birebir kapanir, uc satirda fark "
    "2.220e-16 / 2.220e-16 / 4.441e-16. Modulun FIZIK cipalari Bessel "
    "oranlari, Rayleigh ussu, tek yarik ve TIY faz farkidir; dordu de n'den, "
    "R'den, c'den ve birim seciminden BAGIMSIZDIR. BU MODUL HICBIR "
    "MALZEMENIN OPTIK SABITINI IDDIA ETMEZ."
)

_CIPA_SINIFI_BEYANI = (
    "FIZIK CIPASI (surekli-ortam elektrodinamiginin izgaradan ORTAYA "
    "CIKTIGINI gosterir): Bessel oranlari, Rayleigh ussu, tek yarigin "
    "birinci sifiri, TIY faz farki delta_ps. "
    "SEMA CIPASI (COZUCUYU dogrular, FIZIGI DEGIL -- sayilari semanin kendi "
    "ozelligidir): Courant siniri S*sqrt(d), Yee'nin ikinci mertebesi, "
    "izgara dispersiyon katsayisi, sihirli adim S = 1. "
    "CEVIRI DENETIMI: Fresnel |R| -- cozucunun arayuzu gercekten cozdugunu "
    "gosterir ama bir fizik cipasi DEGILDIR, ve ayirma gucunun zayifligi "
    "OLCULDU: arayuzu bir hucre kaydirmak (`kaydir`) dik geliste |R|'yi "
    "yalniz bagil 2.4e-06 oynatiyor, yani FRESNEL BU BOZMAYI GORMEZ. "
    "OLCUM ALETI (cipa DEGIL): `mode_spectrum` adsiz tepe dondurur ve kip "
    "koklerini HIC gormez -- TEPE, KIP DEMEK DEGILDIR; `convergence_ladder` "
    "ise cipalarin HUKMUNU veren olcuttur. "
    "Bir sema cipasini fizik cipasi diye sunmak bu depoda UC KEZ duzeltilmis "
    "bir hatadir; bu satir onun panel tarafindaki karsiligidir."
)

_PENCERE_BEYANI = (
    "kappa = 0.5 BIR TOLERANS DEGIL, iki tamsayi mertebe arasindaki KARAR "
    "SINIRIDIR ve bir MODELLEME SECIMIDIR (grid.courant_limit'in `S en hizli "
    "dalga hiziyla tanimlidir` beyaniyla ayni statude); 0.40-0.60 arasinda "
    "duz bir plato olculdu. "
    "BESSEL ORANININ HANE SAYISI KIPE BAGLIDIR ve tek bir sayiyla yazilamaz: "
    "ayni kosumda (R = 80) j11/j01 sapmasi -4.7504e-05 iken j12/j01'inki "
    "-4.2050e-04, yani 8.85 KAT. Bu senaryonun kaba basamaginda (R = 5) bes "
    "orandan yalniz IKISI %2'lik esleme penceresine giriyor. "
    "TEK YARIK cipasi yalniz a/lam >= 2 penceresinde gecerlidir; a/lam < 2'de "
    "olculen degerler 0.828910 ... 0.960164'e duser. "
    "OLCUM ALETI BUYUK ACILARDA GECERSIZDIR ve ariza theta'da MONOTON DEGIL, "
    "sgf'nin FONKSIYONUDUR: |rho_s| > 1 (fiziksel olarak imkansiz) bolgesi "
    "sgf = 9'da theta = 70-72.5 derecede, sgf = 3'te theta = 50-55 derecede "
    "olculdu ve theta = 75'te IKISINDE DE |rho_s| < 1. "
    "sgf PENCERENIN PARCASIDIR, bir analiz ayrintisi DEGIL: theta = 55'te "
    "sgf 3 -> 6 ile ||rho_s| - 1| 4.2533e-02'den 2.7313e-05'e iniyor "
    "(1557 kat), ayni yerde npml 60/120/200 ile 1.0000 kat. "
    "Pencere adlandirilmadan sayi yazilmaz -- bu depoda hukmun penceresinin "
    "disina tasinmasi ON KEZ olculdu."
)

_MERDIVEN_TANISI = (
    "mutlak j(0,1) merdiveni OTURMADI (p_inf = 0.4863 +- 0.3574) ve bu bir "
    "kusur DEGIL, ORAN cipasinin GEREKCESIDIR: etkin yaricap sapmasi bir "
    "kuvvet yasasi degil KAFES ARITMETIGIDIR "
    "(delta(R) = 0.9763*(sqrt(N(R)/pi) - R) + 0.3651, Pearson r = 0.999041), "
    "dolayisiyla mutlak j hicbir tamsayi mertebeyi sertifika EDEMEZ; oranda "
    "delta tam sadelesir."
)

_RAYLEIGH_TANISI = (
    "Rayleigh ussu: SIGMA CINSINDEN MARJ YAZILAMAZ, cunku sigma_fit bir "
    "belirsizlik degil PENCERENIN EGRILIGIDIR -- analitik tam cozumun ayni "
    "izgaradaki sigma_fit'i 0.015423, FDTD'ninki 0.016080 (oran 1.0426) ve "
    "artiklarin isaret dizisi BIREBIR AYNI. 3B ussunun (4) dislanmasi bu "
    "yuzden ARALIK olarak yazilir: bes pencerede p araligi [2.924895, "
    "3.243596], aralik genisligi 0.319 < 4'e uzaklik 0.756. MERTEBE "
    "SERTIFIKASI YOK."
)

_YARIK_TANISI = (
    "tek yarigin birinci sifiri: cipa MERTEBE HUKMU OLMADAN sevk edildi "
    "(merdiven YARGILANAMAZ). Olculen a*sin(theta_1)/lam = 0.998610 +- "
    "0.019357 (n = 14, aralik [0.964916, 1.029080]), yani IKI HANE (%1.9); "
    "ve cipa egiklik carpanina KORDUR -- cos(theta) kaldirilinca tek tek "
    "fark yalniz 5.2551e-05 ... 2.9382e-03, carpanin dogrulugu BEYAN "
    "EDILIYOR, OLCULMUYOR."
)

_TIY_TANISI = (
    "TIY faz farki delta_ps: merdiven r = 1.5 ve lam = 16/24/36/54 ile "
    "OTURDU(2), p_inf = 2.1028 +- 0.3364. r = 2'lik dort basamak bu "
    "pencerede YOKTUR -- lam = 5 ve 10 |rho| = 1 kapisina takiliyor, "
    "lam = 160'ta guvenli kayit uzunlugu kalmiyor."
)


@dataclass(frozen=True, eq=False)
class EMScanResult:
    """Bir yaricap merdiveninin ham kaydi -- gozlemlenebilir HESAPLANMAMIS.

    `records` senaryodaki `radii` ile AYNI SIRADADIR. Spektrum ve merdiven
    burada hesaplanmaz cunku `engine` katmani `observable`i import edemez;
    onlar `viz/em_frames.py`de kurulur.

    `eq=False`: alanlar ndarray tasiyan dataclass'lardir; uretilen `__eq__`
    dizi dondurur ve `__hash__` `TypeError` verir. Depo bu tuzagi
    `engine/hf/integrals.py`, `engine/md/state.py` ve `spin/lattice.py`de
    zaten odedi.
    """

    scenario: EMScenario
    records: tuple[FieldRecord, ...]


def _geometri(scenario: EMScenario, radius: float) -> tuple[int, int]:
    """Izgara boyu ve MERKEZ DUGUMUN indisi.

    Boy TEK sayidir ve merkez tam ortadaki dugumdur: `CircularPEC` maskesi
    merkezi `0.5*(n-1)` alir, dolayisiyla cift bir boyda merkez iki dugumun
    ARASINA duserdi ve dairenin merdivenlenmesi dort kadranda ayni olmazdi.
    """
    n = 2 * int(math.ceil(radius + scenario.margin)) + 1
    return n, (n - 1) // 2


def _dedektorler(
    scenario: EMScenario, radius: float, merkez: int
) -> tuple[tuple[int, int], ...]:
    """`detectors` dedektor, GENERIK yaricap ve acilarda.

    Simetrik bir yerlesim (or. hepsi ayni yaricapta, ya da acilari 2pi/N'nin
    tam katlari) `m != 0` kiplerinin bir kismini SONDURURDU: o kiplerin
    acisal bagimliligi cos(m*theta)dir ve dedektorler bir dugum cemberine ya
    da bir simetri eksenine oturursa toplam genlikte kip gorunmez olur. Bu
    yuzden yariciplarin ve acilarin birbirine orani BILINCLI OLARAK kesirli
    secildi. Yerlesimin bedeli de olculdu ve beyan ediliyor: en zayif
    capalanan kipin (j12) genligi bu yerlesimde en buyuk tepenin yalniz
    0.0214'u, yani `peak_floor`un ust ucunu BU SECIM belirliyor.
    """
    n_det = scenario.detectors
    cikti: list[tuple[int, int]] = []
    for k in range(n_det):
        rr = radius * (0.28 + 0.60 * ((k * 5) % n_det) / max(n_det - 1, 1))
        th = 2.0 * math.pi * k / n_det + 0.13
        cikti.append(
            (
                merkez + int(round(rr * math.cos(th))),
                merkez + int(round(rr * math.sin(th))),
            )
        )
    return tuple(cikti)


def _courant_kapisi(scenario: EMScenario) -> None:
    """Yukleyicinin ETKIN Courant kapisi PYTHON YOLUNDA DA (spin'in
    `_coupling_kapisi` deseni), `build_em` GIRISINDE, hesaptan ONCE. eps_r < 1'de
    dalga hizi c/sqrt(eps_r) > c; motor (`yee_coefficients_2d`) yalniz ham S'yi
    sinar. eps_r <= 0 bu kapinin isi DEGIL. Sayilar TAM hassasiyetle (repr) basilir."""
    eps = scenario.eps_r
    if 0.0 < eps < 1.0 and scenario.courant / math.sqrt(eps) > courant_limit(2):
        raise InvalidInputError(
            f"eps_r = {eps!r} < 1 ortamda dalga vakumdan hizlidir ve etkin "
            f"Courant S/sqrt(eps_r) = {scenario.courant / math.sqrt(eps)!r} "
            f"2B sinirini ({courant_limit(2)!r}) asiyor (S = {scenario.courant!r})"
        )


def build_em(scenario: EMScenario) -> EMScanResult:
    """Her yaricapta bir 2B bosluk kosar. TOHUM YOK, PENCERE VAR.

    ADIM SAYISI YARICAPLA ORANTILIDIR (`step_factor * R * sqrt(eps_r)`) ve
    bu bir kolaylik degil merdivenin ON KOSULUDUR: bosluk kiplerinin frekansi
    1/R ile olcekleniyor, dolayisiyla sabit adim sayisi kaba basamakta cok,
    ince basamakta cok az periyot kaydederdi ve merdiven merdivenlenmenin
    yaninda COZUNURLUGUN degisimini de olcerdi. Adim R ile olcekleninde her
    basamakta ayni sayida periyot kaydedilir ve spektral cozunurluk `j`
    biriminde SABIT kalir.

    KAYNAK DALGA BOYU DA R ILE OLCEKLENIR (`wavelength_per_radius * R *
    sqrt(eps_r)`) ve ayni gerekce: kaynagin bandi kipin frekansina gore sabit
    kalmali, yoksa kaba ve ince basamaklarda FARKLI kipler uyarilirdi.
    `wavelength_per_radius` bir teori sabiti DEGIL bir PENCERE DUGMESIDIR --
    senaryoda yazilidir ve bu dosya hicbir Bessel kokunu gormez.

    dx = 1 SABITTIR VE SENARYODA BIR DUGME DEGILDIR -- OLCULDU. Task 6'da
    olculen sey suydu: `run_2d` kaynagi `t = n*S*dx` zamaninda besliyordu,
    oysa `wavelength` HUCRE cinsinden veriliyordu; ikisi yalniz dx = 1'de
    ortusuyordu. Olcum (R = 5/10, step_factor 120): dx = 1'de tepe sayisi
    20 / 13, dx = 0.5'te 4 / 4, dx = 2'de 31 / 43 -- lambda ucunde de
    7.257613 kalarak. Yani dugme baglidir ama SESSIZCE YANLIS calisiyordu.

    O ZAMAN YALNIZ SENARYODAN CIKARILDI, MOTOR ACIK KALDI -- ve bu bir
    kapsam karariydi, bir cozum degil: kusur Python API'sinden hala
    erisilebilirdi. Simdi kapatildi: `dx != 1` `grid.dx_kapisi`nda
    REDDEDILIYOR, cunku olculdugunde `dx`in AYRI BIR SERBESTLIK DERECESI
    OLMADIGI cikti (davranis yalniz `wavelength/dx` oranina bagli, ve bu
    1B'de bes, 2B'de dort (dx, W) ciftinde BIT BIT dogrulandi). Panel dx'i
    yine de basar, degeri `Window`dan gelir ve artik ZORUNLU olarak 1'dir.

    BIRIM DONUSUMU YOKTUR ve bu bir eksiklik degil bir OLCUM sonucudur:
    `TBScenario` eV -> hartree cevirir cunku TB literaturu eV'de yayinlanir;
    burada cevrilecek bir sey yoktur, cunku `c = eps_0 = mu_0 = 1` bir birim
    secimidir ve olculen buyukluk (bir frekans ORANI) zaten boyutsuzdur.
    """
    _courant_kapisi(scenario)
    kayitlar: list[FieldRecord] = []
    for R in scenario.radii:
        n, merkez = _geometri(scenario, R)
        olcek = math.sqrt(scenario.eps_r)
        lam = scenario.wavelength_per_radius * R * olcek
        adim = int(round(scenario.step_factor * R * olcek))
        if adim < 8:
            raise InvalidInputError(
                f"R = {R:g} basamaginda adim sayisi {adim} cikti; spektrum en "
                "az 8 adim ister. `step_factor` cok kucuk."
            )
        # Kaynak da GENERIK bir yerde: merkezde olsaydi m != 0 kiplerinin
        # HICBIRI uyarilamazdi (hepsinin merkezde dugumu var).
        si = merkez + int(round(0.55 * R * math.cos(0.4)))
        sj = merkez + int(round(0.55 * R * math.sin(0.4)))
        kayitlar.append(
            run_2d(
                Medium.uniform_2d(n, n, eps_r=scenario.eps_r),
                courant=scenario.courant,
                boundary=CircularPEC(radius=R),
                steps=adim,
                source=lambda t, _lam=lam: ricker(
                    t, wavelength=_lam, sgf=scenario.sgf
                ),
                source_index=(si, sj),
                probes=_dedektorler(scenario, R, merkez),
                wavelength=lam,
                sgf=scenario.sgf,
                window_fn=scenario.window_fn,
            )
        )
    return EMScanResult(scenario=scenario, records=tuple(kayitlar))


def _hukum_metni(
    p_inf: float | None, uncertainty: float | None, verdict: str
) -> str:
    """Uc degerli hukmun TEK metin kaynagi.

    "OTURDU" DISINDA HICBIR HUKUM BIR TAMSAYI YAZDIRMAZ. Bu bir bicim
    tercihi degil bu modulun kuralidir: `YARGILANAMAZ` sessizce "oturdu"
    diye basilamaz, ve `OTURMADI` da bir mertebe sertifikasi degildir --
    ikisinde de yazilan sey `HUKUM_VERILMEDI`dir.
    """
    if p_inf is None or uncertainty is None:
        olcum = "p_inf OKUNAMADI (merdiven aritmetigin tanimli oldugu noktaya "
        olcum += "ulasmadi)"
    else:
        olcum = f"p_inf = {p_inf:.6f} +- {uncertainty:.6f}"
    if verdict == "OTURDU":
        if p_inf is None:
            raise InvalidInputError(
                "OTURDU hukmu p_inf olmadan verilemez: oturulan tamsayi "
                "mertebe p_inf'ten okunur."
            )
        return f"{olcum} -> OTURDU({int(round(p_inf))})"
    return f"{olcum} -> {verdict}, {HUKUM_VERILMEDI}"


def provenance_for(
    scenario: EMScenario,
    *,
    windows,
    ratios,
    noise,
    peaks,
    orders,
    p_inf: float | None,
    uncertainty: float | None,
    verdict: str,
) -> Provenance:
    """Olculmus merdivenden panel beyanini kurar.

    `windows` / `ratios` / `noise` / `peaks` senaryodaki `radii` ile AYNI
    SIRADADIR ve `viz/em_frames.py`den gelir; `orders` / `p_inf` /
    `uncertainty` / `verdict` ise `observable/em_obs.py::convergence_ladder`
    ciktisidir. Bu fonksiyon HICBIRINI yeniden hesaplamaz -- hesaplasaydi
    panel ile kayit ayrisabilir ve panel motorun soylemedigini yazardi.

    `converged` OLCULEBILIR bir seye baglanir -- ve bu modulde o sey bir SCF
    dongusunun durmasi DEGILDIR, cunku FDTD'de boyle bir olcut yoktur.
    Baglandigi sey su: BU KOSUMUN MERDIVENI BIR TAMSAYI MERTEBEYE OTURDU MU?
    Oturmadiysa panel "yakinsadi" diyemez, cunku raporlanan sayi bir tahmin
    degil bir ARA DURAKTIR. Yani buradaki tik "hesap bitti" degil, "bastigim
    sayinin bir yakinsama mertebesi VAR" demektir.
    """
    n_r = len(scenario.radii)
    for ad, dizi in (
        ("windows", windows),
        ("ratios", ratios),
        ("noise", noise),
        ("peaks", peaks),
    ):
        if len(dizi) != n_r:
            raise InvalidInputError(
                f"tani dizileri {n_r} yaricapla ayni uzunlukta olmali, "
                f"{ad} icin verilen: {len(dizi)}"
            )
    w = windows[-1]
    if not isinstance(w, Window):
        raise InvalidInputError(
            f"windows ogeleri grid.Window olmali, verilen: {type(w).__name__}"
        )

    merdiven = "  ".join(f"{v:.9f}" for v in ratios)
    araliklar = " -> ".join(f"{1.0 / R:g}" for R in scenario.radii)
    gurultu = " / ".join(f"{g:.2e}" for g in noise)
    ham = (
        ", ".join(f"{o:.4f}" for o in orders)
        if len(orders)
        else "okunamadi (derinlik yetmedi ya da ardisik farklar isaret degistirdi)"
    )
    canli = (
        f"[CANLI -- bu kosum] Bessel orani (ikinci en dusuk tepenin en dusuge "
        f"orani): degerler {merdiven}; izgara araliklari 1/R = {araliklar} "
        f"(r = {scenario.ratio:g}, derinlik {n_r}); basamak basina bagil "
        f"pencere gurultusu {gurultu}; ham mertebeler {ham}; "
        + _hukum_metni(p_inf, uncertainty, verdict)
        + ". Tepe sayisi basamak basina "
        + " / ".join(str(int(t)) for t in peaks)
        + "."
    )
    beyan = (
        f"[BEYAN -- olcumu tests/em/test_bessel.py'de] {_MERDIVEN_TANISI} "
        f"[BEYAN -- tests/em/test_sacilma.py] {_RAYLEIGH_TANISI} "
        f"[BEYAN -- tests/em/test_uzak_alan.py] {_YARIK_TANISI} "
        f"[BEYAN -- tests/em/test_arayuz.py] {_TIY_TANISI}"
    )

    pencere = (
        f"BU KOSUMUN PENCERESI (en ince basamak): izgara {w.izgara} hucre, "
        f"dx = {w.dx:g}, S (Courant) = {w.courant:g}, adim = {w.steps}, "
        f"lambda = {w.wavelength:.6f} hucre, npml = {w.npml}, "
        f"sgf = {w.sgf:g}, FFT penceresi {w.window_fn}, pad = {scenario.pad}, "
        f"tepe tabani = {scenario.peak_floor:g}, "
        f"{scenario.detectors} dedektor, eps_r = {scenario.eps_r:g}. "
        "npml = 0 BIR EKSIKLIK DEGIL BEYANDIR: dairesel PEC bosluk TAM "
        "YANSITICIDIR, sogurucu katman yoktur ve bu yuzden npml senaryoda "
        "bir dugme olarak BULUNMAZ -- baglanmamis bir dugmeyi sessizce kabul "
        "etmek bu depoda bir kez duzeltildi. "
        + _PENCERE_BEYANI
    )

    return Provenance(
        method=(
            f"Maxwell FDTD, Yee izgarasi ({scenario.name}); 2B TM_z, "
            f"dairesel PEC bosluk, {n_r} yaricap "
            f"(R = {', '.join(f'{R:g}' for R in scenario.radii)}), "
            f"S = {scenario.courant:g}, adim = {scenario.step_factor:g}*R*"
            f"sqrt(eps_r), Ricker kaynak"
        ),
        approximations=_EM_YAKLASIMLAR
        + (
            f"sonlu merdiven: bu kosum {n_r} basamak kullanir ve SUREKLI "
            "LIMIT ALINMAZ; okunan mertebe merdivenin kendi araligina aittir, "
            "sonsuz ince izgaranin degil",
        ),
        neglected=_EM_IHMAL,
        expected_error=_EM_BEKLENEN_HATA,
        converged=verdict == "OTURDU",
        circularity=_DAIRESELLIK_BEYANI,
        anchor_class=_CIPA_SINIFI_BEYANI,
        window=pencere,
        convergence_diagnosis=canli + " " + beyan,
        diagnostics={
            "n_yaricap": float(n_r),
            "r": float(scenario.ratio),
            "adim_carpani": float(scenario.step_factor),
            "dedektor": str(scenario.detectors),
            "pad": str(scenario.pad),
            "FFT": scenario.window_fn,
        },
    )
