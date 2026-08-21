"""KADEMELI ALANIN CATALLANMASI -- Modul 3H'nin en guclu cipasi.

SINIF (dosyanin BUTUNU icin tek bir sinif YAZILAMAZ, uc test uc ayri sey
yapar; her testin kendi dokstringi sinifini yineler):

  * test_KADEMELI_ALAN_KAREKOK_yasasiyla_sonuyor  -> DAIRESEL DEGIL
  * test_GECIS_NOKTASI_triplet_Hessian_kokunden   -> TRANSKRIPSIYON DENETIMI
        (cipanin kullandigi d_c sabitini, onu ureten hesaba baglar)
  * test_GECIS_ALTINDA_kademeli_alan_TAM_SIFIR    -> DAIRESEL DEGIL (karsit yon)

Cipanin dairesel olmama gerekcesi: iki buyukluk IKI AYRI hesaptan gelir ve
birbirini hic kullanmaz.

  d_c   : RHF cozumunun UCLU (triplet) orbital Hessian'inin en kucuk
          ozdegerinin SIFIR KOKU  (Modul 1C, analysis.external_stability)
  Delta : UHF indirgemesinin kademeli alani  (bu modul, downfold_uhf)

`external_stability` `downfold_uhf`i cagirmaz; `downfold_uhf` da hicbir
kararlilik analizi kosmaz. Aralarindaki tek bag bu dosyadaki sayidir.

Cipa, Delta'nin d_c'ye yaklasirken KAREKOK yasasiyla sonmesidir. Us hicbir
yerden VERILMEZ; log-log egiminden okunur ve catallanma beklentisi 1/2'dir.

UCUNCU bir bagimsiz okuma: Modul 1C, H2 molekulunde (bambaska bir sistem,
bambaska bir yordam -- tek bag, Coulson-Fischer noktasi, |spin| olcusu)
usu 0.4970 olcmustu. Bu dosyanin uc halkada olctugu usler ayni yere dusuyor.

USUN OLCEGE KOR OLDUGU BIR YERDEN BILINIYOR ve BURADA KAPATILDI: log-log
egimi Delta -> c*Delta donusumu altinda DEGISMEZ, yani us testi tek basina
Delta'yi sabit bir sayiyla carpan bir kusuru GOREMEZ (Modul 1C'nin dersi:
"kok okuyan cipa olcege kordur"). Bu yuzden ayni testte MUTLAK bir deger de
kilitlenir; o satir transkripsiyon denetimidir ve oyle isaretlidir.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from qlab.engine.hf.analysis import external_stability
from qlab.engine.hf.scf import solve
from qlab.engine.tb.downfold import downfold_uhf
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM

ADIMLAR = np.array([2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2])

# Gecisin ALTINDA kabul esigi. SECILMEDI, IKI YANDAN OLCULDU:
#   ust sinir  : gecisin hemen USTUNDEKI en kucuk fiziksel Delta 1.244899e-02
#                (N=14, d_c + 0.002) -- esik ondan 4 mertebe asagida.
#   alt sinir  : gecisin ALTINDA olculen en buyuk artik 1.539807e-08
#                (N=6, d_c - 0.02) -- esik ondan ~6.5 kat yukarida.
#                (Bu satir da bir kez 6.178e-09 / N=10 d_c-0.20 diyordu;
#                 o nokta yeniden olculdugunde 9.908740e-16 verdi.)
# Alt sinirin nereden geldigi de biliniyor ve TAHMIN DEGIL: Task 1'de
# olculdu, `solve_uhf`in artik toleransi 1e-8'dir ve Fock elemanlarinin
# gurultu tabani oradadir (tolerans 1e-11'e cekilince taban 1e-15'e duser).
#
# Plan bu esigi 1e-8 yaziyordu ve BU YETMEZ -- gerekce tahmin degil, olculmus
# bir KARSI ORNEK:
#     N=6,  d_c - 0.02:  |Delta| = 1.539807e-08   -> 1e-8'i ASIYOR
#     N=10, d_c - 0.20:  |Delta| = 9.908740e-16
# Bu nokta gecisin ALTINDADIR (UHF, RHF'e cokmelidir) ve Delta fiziksel olarak
# sifir olmalidir; olculen 1.54e-08 cozucunun artik tabanidir. Yani planin
# 1e-8'lik siniri fizigi degil `solve_uhf`in artik toleransini olcerdi ve
# KIRMIZI olurdu.
#
# BU BLOK BIR KEZ YANLIS YAZILDI, DUZELTILDI: once "taban TOHUM SAYISINA gore
# oynuyor" deniyor ve seeds=16 icin 2.675e-10, seeds=20 icin 1.540e-08
# bildiriliyordu. Yeniden olculdugunde iki tohum sayisi da BIREBIR AYNI sonucu
# veriyor (her ikisinde 1.539807e-08 ve 9.908740e-16); taban tohum sayisina
# DEGIL, o geometrinin yakinsamis cozumune aittir. Sayi dogruydu, sebep degil.
#
# 1e-7, olculen en buyuk artigin (1.539807e-08) ~6.5 kati ustunde ve en kucuk
# FIZIKSEL Delta'nin 4 mertebe altinda.
GECIS_ALTI_ESIGI = 1e-7


def _halka(n: int, d: float) -> Molecule:
    yaricap = d / (2 * np.sin(np.pi / n))
    a = yaricap * BOHR_TO_ANGSTROM
    return Molecule.from_string(
        "; ".join(
            f"H {a*np.cos(2*np.pi*k/n):.17g} {a*np.sin(2*np.pi*k/n):.17g} 0"
            for k in range(n)
        )
    )


def _dis_ozdeger(n: int, d: float) -> float:
    """Triplet Hessian'in en kucuk ozdegeri -- indirgemeden BAGIMSIZ yol."""
    result = solve(_halka(n, d), "sto-3g", stability=False, max_iterations=8000)
    return float(external_stability(result).eigenvalues.min())


@pytest.mark.parametrize(
    "n,d_c,alt,ust",
    [
        (6, 2.233818500, 2.0, 2.6),
        (10, 1.957853161, 1.7, 2.1),
        pytest.param(14, 1.757512489, 1.5, 1.9, marks=pytest.mark.slow),
    ],
)
def test_GECIS_NOKTASI_triplet_Hessian_kokunden_okunuyor(n, d_c, alt, ust):
    """SINIF: TRANSKRIPSIYON DENETIMI -- bir fizik iddiasi TASIMAZ.

    Isi sudur: asagidaki us testi `d_c`yi SABIT olarak aliyor; bu test o
    sabitin gercekten triplet Hessian'in koku oldugunu dogrular. Sabit
    kaymissa us testi yanlis bir noktadan olcerdi ve bunu sessizce yapardi.

    OLCULDU (21.08.2026, bu depoda yeniden): N=6 2.233818500,
    N=10 1.957853161, N=14 1.757512489 bohr. Ucu de planin referans
    degeriyle dokuz hanede AYNI cikti.

    Parantezlerin de sinirlarini kilitlemek gerekiyor, cunku brentq
    ISARET DEGISIMI olmadan hic kok aramaz. OLCULDU:
        N=6 : f(2.0) = +0.390453   f(2.6) = -0.501389
        N=10: f(1.7) = +0.408346   f(2.1) = -0.190909
        N=14: f(1.5) = +0.398245   f(1.9) = -0.177467
    Isaret hep ayni yonde: kucuk d'de RHF kararli (pozitif), buyuk d'de
    kararsiz (negatif).

    d_c, N buyudukce KUCULUYOR -- uzun halka yerel olarak zincire benziyor
    ve aromatik kararlilik seyreliyor (Modul 1C'de olculdu: N=18'de dis
    ozdeger d=1.6 bohr'da yalnizca +0.00983).

    `slow` ISARETI PER-PARAMETRE ve OLCUM (pytest --durations): N=6 0.34 s,
    N=10 1.40 s, N=14 6.36 s. Isaret yalnizca N=14'te -- ve BU DOSYADAKI
    TEK ZORUNLU isaret odur (gerekcesi asagida).

    DEPONUN ISARETLEME SINIRI OLCULDU ve DEVRALINAN KURAL YANLIS CIKTI.
    Task 1'in dokstringi "2.04-5.20 s arasindaki testlerin HEPSI isaretli,
    0.86 s ve altindakilerin hicbiri degil" diyor; bu iki DOSYA icin
    dogruydu ama DEPO GENELINE genellenemiyor. Bu dosya haric tutularak
    iki lane ayri ayri kosuldu (21.08.2026):

        -m slow        :  37 test, 203.72 s -- en hizli call 0.01 s
        -m "not slow"  : 867 test,  68.16 s -- en yavas call 4.54 s

    Yani bantlar 0.01 s ile 4.54 s arasinda TAMAMEN CAKISIYOR ve
    devralinabilecek bir esik YOK. Isaretli olup hizli gorunenlerin sebebi
    de olculdu: ya yavas bir kardesle parametrelestirilmisler
    (test_PARCA_TOPLAMI_cipasi'nin H2 hali 0.02 s, N2 hali 5.21 s), ya da
    modul kapsamli pahali bir fixture'i paylasiyorlar (test_double_slit'te
    setup 25.95 s, call 0.00 s).

    Bu yuzden esik DEVRALINMADI, secildi ve gerekcesi yazildi: kendi call
    suresi 2 s ve USTUNDE olan durum isaretlenir. Iki ucu da olcumden
    geliyor -- (a) isaretsiz bandin tavani 4.54 s oldugu icin 6.36 s'lik
    N=14 hali ZORUNLU olarak isaretli (depoda o kadar yavas isaretsiz test
    yok), (b) 2 s siniri, bu dosyanin hizli lane'e kattigi yuku 2.60 s'de
    (0.34 + 1.40 + 0.86) tutuyor; toplam 69.56 s'lik dosyanin geri kalan
    66.96 s'si isaretli lane'e dusuyor.
    """
    assert _dis_ozdeger(n, alt) > 0.0, f"N={n}: alt sinirda RHF kararsiz"
    assert _dis_ozdeger(n, ust) < 0.0, f"N={n}: ust sinirda RHF kararli"
    kok = brentq(lambda d: _dis_ozdeger(n, d), alt, ust, xtol=1e-9)
    assert kok == pytest.approx(d_c, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize(
    "n,d_c,us,delta_son",
    [
        (6, 2.233818500, 0.496190, 8.115373e-02),
        (10, 1.957853161, 0.497600, 6.145320e-02),
        (14, 1.757512489, 0.497811, 4.948334e-02),
    ],
)
def test_KADEMELI_ALAN_KAREKOK_yasasiyla_sonuyor(n, d_c, us, delta_son):
    """SINIF: DAIRESEL DEGIL -- us hicbir yerden verilmedi, uyumdan cikti.

    OLCULDU (21.08.2026, bu depoda yeniden): N=6 0.496190, N=10 0.497600,
    N=14 0.497811. Ucu de planin referans degeriyle dort hanede AYNI cikti
    (0.4962 / 0.4976 / 0.4978). Catallanma beklentisi 1/2'dir ve olculen
    uslerin sapmasi 0.003810 / 0.002400 / 0.002189 -- N buyudukce KUCULUYOR.
    Modul 1C'nin H2 uzerinde BAGIMSIZ olarak olctugu us 0.4970 idi.

    BU SATIR BIR KEZ YANLIS YAZILDI, DUZELTILDI: once "olculen us ona
    0.0022'den yakin" diyordu, oysa N=6'nin sapmasi 0.003810 -- iddia
    edilenin 1.7 kati. Asagidaki tolerans (abs=0.02) gercek sapmanin cok
    ustunde oldugu icin hata testi KIRMIYORDU; ancak yeniden olculunce
    goruldu (Task 5 dogrulama turu). Sayilar dogruydu, ozet cumle degil.

    Olculen Delta'lar (|Delta|, hartree):
        adim     N=6           N=10          N=14
        0.002    2.051272e-02  1.546981e-02  1.244899e-02
        0.004    2.898803e-02  2.186744e-02  1.759729e-02
        0.008    4.093472e-02  3.089592e-02  2.486467e-02
        0.016    5.772026e-02  4.361192e-02  3.510406e-02
        0.032    8.115373e-02  6.145320e-02  4.948334e-02

    DORT AYRI SEY kilitlenir ve her birinin ayri bir isi vardir:

    1) Delta gercekten BUYUK: en kucugu 1.244899e-02, esikten 12 kat
       yukarida. `> 0` demek yeterli olmazdi -- catallanmanin d_c'nin
       hemen ustunde ANINDA basladigini soyleyen sey buyukluktur.
    2) Delta adimla MONOTON artiyor. Egim tek basina, dagilmis
       noktalardan gecen bir dogruyla da 0.5 cikabilirdi.
    3) Egim (= us) 0.5.
    4) Uyumun ARTIGI kucuk: bu bir GUC yasasidir, "egimi 0.5 olan bir
       sacilim" degil. OLCULDU, log-artik max: N=6 1.892e-03,
       N=10 1.169e-03, N=14 1.034e-03; sinir 5e-3 (~2.6 kat ustunde).
    5) MUTLAK deger. Bu satir TRANSKRIPSIYON DENETIMIDIR, cipa degil, ve
       ayri bir delik kapatir: log-log egimi Delta -> c*Delta altinda
       DEGISMEZ, yani us testi Delta'yi sabit bir sayiyla carpan bir
       kusuru goremez. Tolerans abs=1e-6, olculen gurultu tabaninin
       (~1e-8) 100 kati ustunde.

    `slow` ISARETI OLCUM: N=6 2.75 s, N=10 12.47 s, N=14 40.64 s. Ucu de
    2 s esiginin (gerekcesi ilk testin dokstringinde) ustunde, bu yuzden
    isaret parametre basina degil TESTIN tamamina konuldu.
    """
    deltalar = np.array(
        [abs(downfold_uhf(_halka(n, d_c + a), "sto-3g", seeds=20).delta) for a in ADIMLAR]
    )
    assert deltalar.min() > 1e-3, f"N={n}: cozum spin-kirik degil: {deltalar}"
    assert np.all(np.diff(deltalar) > 0.0), f"N={n}: Delta monoton degil: {deltalar}"

    egim, kesisim = np.polyfit(np.log(ADIMLAR), np.log(deltalar), 1)
    olculen = float(egim)
    assert olculen == pytest.approx(us, abs=0.01)
    assert olculen == pytest.approx(0.5, abs=0.02)

    artik = float(np.abs(kesisim + egim * np.log(ADIMLAR) - np.log(deltalar)).max())
    assert artik < 5e-3, f"N={n}: guc yasasi oturmuyor, log-artik {artik:.3e}"

    assert deltalar[-1] == pytest.approx(delta_son, abs=1e-6)


@pytest.mark.parametrize(
    "n,d_c",
    [(6, 2.233818500), pytest.param(10, 1.957853161, marks=pytest.mark.slow)],
)
def test_GECIS_ALTINDA_kademeli_alan_TAM_SIFIR(n, d_c):
    """SINIF: DAIRESEL DEGIL -- cipanin KARSIT yonu.

    UHF, gecis noktasinin altinda RHF'e coker ve Delta soner. Yalnizca
    "ustunde buyuyor" demek, HER YERDE buyuyen bir kusuru yakalayamazdi;
    catallanma iddiasi iki tarafi da gerektirir.

    OLCULDU (21.08.2026, testin kostugu DORT nokta, seeds=16):
        N=6,  d_c - 0.20: |Delta| = 2.886867e-09, <S^2> = -6.661e-15
        N=6,  d_c - 0.02: |Delta| = 1.539807e-08, <S^2> = +2.665e-15
        N=10, d_c - 0.20: |Delta| = 9.908740e-16, <S^2> = -1.421e-14
        N=10, d_c - 0.02: |Delta| = 1.998401e-16, <S^2> = +5.329e-15

    BU TABLO BIR KEZ YANLIS YAZILDI, DUZELTILDI: dort satirin DORDU de
    tekrarlanmadi (once 2.188e-09 / 2.675e-10 / 6.178e-09 / 4.302e-17
    bildirilmisti). Yukaridakiler yeniden olculmustur.

    Esigin (GECIS_ALTI_ESIGI = 1e-7) neden 1e-8 OLMADIGI o sabitin yaninda
    yazilidir: N=6, d_c - 0.02 noktasi 1.539807e-08 veriyor ve planin
    onerdigi 1e-8 sinirini ASIYOR. O nokta gecisin ALTINDADIR, yani Delta
    fiziksel olarak sifir olmali; olculen deger cozucunun artik tabanidir.

    <S^2> NEDEN mutlak deger icinde: olculen degerlerin ikisi NEGATIF
    (-6.661e-15, -1.421e-14). Sayisal olarak sifirdirlar, ama ciplak
    `< 1e-6` iddiasi buyuk NEGATIF bir degeri de gecirirdi -- yani
    <S^2>'yi bozan bir kusuru tek yonlu yakalardi. Sinir 1e-6, olculen en
    buyuk mutlak deger 1.421e-14: dokuz mertebe.

    `slow` ISARETI PER-PARAMETRE ve OLCUM: N=6 0.86 s varsayilan seritte,
    N=10 4.08 s isaretli (2 s esiginin ustunde).
    """
    for d in (d_c - 0.20, d_c - 0.02):
        p = downfold_uhf(_halka(n, d), "sto-3g", seeds=16)
        assert abs(p.delta) < GECIS_ALTI_ESIGI, f"N={n}, d={d}: Delta={p.delta:.3e}"
        assert abs(p.s_squared) < 1e-6, f"N={n}, d={d}: <S^2>={p.s_squared:.3e}"
