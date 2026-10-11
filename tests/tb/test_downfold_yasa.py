"""Mesafe yasalari ve SAHTE-CIPA SINAVININ SONUCU.

SINIF (dosyanin BUTUNU icin tek sinif YAZILAMAZ; her testin kendi dokstringi
sinifini yineler):

  * test_SAHTE_CIPA_SINAVININ_SONUCU          -> DAIRESEL DEGIL (dar kapsamli)
  * test_SAHTE_CIPA_IKINCI_METRIGI_AYIRMIYOR  -> DAIRESEL DEGIL (karsit yon)
  * geri kalan her sey                        -> TRANSKRIPSIYON DENETIMI

SINAVIN SONUCU: GECTI, AMA RHF TARAFINDAKINDEN COK DAHA DAR BIR MARJLA.

Sinav sudur: fizigi cikar, BORU HATTINI YERINDE BIRAK; sonra ayni yasa
uydurma makinesini fizik-disi girdiyle kosur ve gercek sistemi ayirt edip
etmedigine bak. Olculen metrik EKSTRAPOLASYON ORANI:

    E_in  = ln|t1| dogrusal uyumunun PENCERE ICI artik RMS'i
    E_out = ayni dogrunun PENCERE DISINDA (hedef) birakti artigin RMS'i
    ORAN  = E_out / E_in

Yani "yasa, uydurdugu yerden ne kadar uzaga tasiyor". Bu metrik bilincli
olarak OLCEKTEN BAGIMSIZDIR; mutlak buyuklugu ayri bir sabit kilitler.

OLCULDU (21.08.2026, bu depoda, sevk edilen `fit_distance_laws` ile,
N=8 halkasi, pencere 1.8/2.0/2.2 bohr, hedef 2.4/2.6/3.0 bohr, seeds=12):

    GERCEK (STO-3G)                 ORAN =   4.160383
    F1  40 rastgele dugumsuz baz    ORAN medyan 29.902685
                                    en iyi (en kucuk) 6.216006
                                    en kotu 413.126243
                                    GERCEGI GECEN: 0/40
    F2  eri * 0                     KOSULAMADI -- asagida
    F3  saf geometri cekirdegi      ORAN 9.9647 ... 29.3917 (bes a degeri)
                                    GERCEGI GECEN: 0/5

Yani 45 fizik-disi kurulumun 45'i de gercek sistemden KOTU. Sinav, planin
koydugu olcute gore (sahte medyan > 4 x gercek sinir) GECIYOR.

UC DURUST OLUMSUZ, UCU DE OLCULDU:

(1) KUYRUK MARJI RHF TARAFININ UCTE BIRI. RHF yolunda ayni sinav gercek
    2.80'e karsi en iyi sahteyi 12.01'de birakmisti -- 4.29 kat. Burada
    gercek 4.160383, en iyi sahte 6.216006 -- yalnizca 1.49 kat. Medyan
    orani da 50.6'dan 7.19'a dusuyor. Sinav geciyor ama AYIRMA GUCU
    dusmus durumda ve bu yazilmadan gecilemez.

(2) IKINCI METRIK AYNI HUKMU VERMIYOR. "Yasa pencere disini iyi
    ongoruyor" cumlesi TEK BASINA fizik kaniti DEGILDIR: bagil ongoru
    hatasinda 40 sahtenin 2'si (0.005409 ve 0.005570, gercek 0.010811)
    ve saf geometri cekirdeklerinin 5'inden 2'si (0.000997 ve 0.000297)
    gercek sistemden DAHA IYI ongoruyor. Bir cekirdegin ustel olmasi
    onu fiziksel yapmaz; saf `exp(-a r)` zaten tanimi geregi usteldir.
    Ayiran sey oran metrigidir, ongorunun kendisi degil.

(3) MUTLAK HATA TEK BASINA DA AYIRMIYOR, TERS YONDEN. 40 sahtenin 7'si
    gercekten kucuk mutlak hata veriyor -- ama ikisinin sebebi t1'lerinin
    neredeyse SIFIR olmasi (bagil hatalari 16.67 ve 8.85 iken mutlak
    hatalari 2.5426e-06 ve 3.0560e-05). Bu, RHF tarafinda "Hcore
    vekilinin orani ~1.00 cikmisti cunku yerinde uydurmasi da berbatti"
    diye kaydedilen tuzagin AYNISIDIR, yalnizca ters yonden. Bu yuzden
    her iki sinir da (bagil VE mutlak) kilitlenir; biri digerini kurtarir.

F2 (eri * 0) NEDEN KOSULAMADI -- ve bu bir eksiklik degil BIR OLCUM.
Plan bu ikameyi "UHF'te anlamsizlasir, beyan ederek atlanabilir" diye
gecmisti. Atlanmadi, KOSULDU ve boru hattinin KENDISI reddetti: elektron-
elektron etkilesimi sifirlaninca degistokus de yok olur, UHF RHF'e coker ve
Delta = 1.110e-16 cikar. `fit_distance_laws` bunu gorup
`UnsupportedSystemError` firlatiyor ("pencerede spin-KIRILMAMIS nokta var").
Yani ikame, beyanla degil KODUN KENDI KAPSAM KONTROLUYLE elendi.

Ayni sey F1'de de bir kez oldu ve gorulmesi degerlidir: 41 cizimin 1'i
(6. cizim) `t2 pencere icinde ISARET DEGISTIRIYOR` diye reddedildi. Kapsam
kontrolleri varsayimsal degil, sinavda GERCEKTEN atesledi.
"""

from __future__ import annotations

import numpy as np
import pytest

import qlab.engine.hf.basis as basis_modulu
from qlab.engine.hf.analysis import external_stability
from qlab.engine.hf.scf import solve
from qlab.engine.tb.downfold import downfold_uhf, fit_distance_laws
from qlab.errors import InvalidInputError, UnsupportedSystemError
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM

# --- Step 1-2'de OLCULEN sabitler (plan bunlari uyduramaz) -------------------

PENCERE = (1.8, 2.0, 2.2)  # uydurma penceresi, bohr
HEDEF = (2.4, 2.6, 3.0)  # pencere DISI, uydurmaya hic girmez
# HEDEF KUMESININ SECIMI SONUCU BELIRLIYOR -- ve bu OLCULDU, gizlenmiyor.
# Ayni yasa (pencere degismeden), yalnizca hedef kumesi uzatilarak:
#     (2.4, 2.6, 3.0)            -> ORAN =  4.160383
#     (2.4, 2.6, 3.0, 3.2)       -> ORAN =  5.997079   (ORAN_SINIRI'ni ASAR)
#     (2.4, 2.6, 3.0, 3.2, 3.6)  -> ORAN = 15.479892   (SAHTE_EN_IYI'yi ASAR)
# Yani d = 3.6 hedefe konsaydi GERCEK sistem, en iyi FIZIK-DISI bazdan
# (6.216006) daha KOTU gorunurdu ve sinav duserdi. Cipanin gectigi hukum
# bu yuzden "yasa ekstrapole eder" DEGIL, su dar hukumdur:
#     "1.8-2.2 bohr penceresinden uydurulan t1 yasasi, 3.0 bohr'a kadar
#      fizik-disi bazlarin uretemedigi bir dogrulukla ongoruyor."
# 3.0 bohr'un otesi icin bu dosya bir sey IDDIA ETMEZ; orada yasanin
# bozuldugu zaten olculmustur (Delta'nin doymasi, bkz. asagidaki dokstring).
# Bu sinir, Task 3'te ayni desenin bir BUG'a yol acmasindan sonra yazildi:
# bir dogrulamanin PENCERESI yazilmazsa hukum pencerenin disina tasinir.
HALKA = 8  # cift site sarti; N=8 ayrica HER d'de spin kararsiz
TOHUM = 12

# TOHUM SAYISI OLCULDU, SECILMEDI: alti mesafenin altisinda da seeds=12 ile
# seeds=16 t1, t2, eps ve |Delta| icin BIREBIR ayni sonucu veriyor (fark
# 0.000e+00). Ucuz olan alinir. Sahte bazlar icin ayni esitlik GARANTI
# DEGILDIR (o yuzden sinav da 12'de sabitlendi ve o sayilar yazildi).

# Step 1: |t1| ve |t2| log-uyumunun PENCERE ICI artigi.
# OLCULEN nokta-nokta maksimumlar: t1 2.624197e-03, t2 5.804253e-04.
# Sinir olculenin 1.52 kati; gurultuye degil GERILEMEYE ayarli.
ARTIK_RMS_SINIRI = 4e-3

# Step 1: pencere DISI t1 ongorusu. OLCULEN bagil hatalar 0.007485 /
# 0.010811 / 0.002050 (max 0.010811); mutlak hatalar 2.236176e-03 /
# 2.761606e-03 / 3.755209e-04 hartree (max 2.761606e-03).
# Sinirlar olculenin ~1.8 kati.
ONGORU_BAGIL_SINIRI = 0.02
ONGORU_MUTLAK_SINIRI = 5e-3  # hartree

# Step 2: gercek sistemin ekstrapolasyon orani ve ona konan ust sinir.
GERCEK_ORAN = 4.160383
ORAN_SINIRI = 5.0

# Step 2: 40 fizik-disi dugumsuz bazin oran istatistigi.
SAHTE_MEDYAN = 29.902685
SAHTE_EN_IYI = 6.216006  # en KUCUK oran, yani gercege en yakin sahte
SAHTE_GECEN = 0  # gercegi geceni: sifir

# Delta yasasinin pencere DISI bagil hatasi. OLCULEN: 0.028079 / 0.055498 /
# 0.069184. Bu, t1'inkinin (0.010811) ALTI KATI ve yasanin ZAYIF halkasidir.
DELTA_ONGORU_BAGIL_SINIRI = 0.10


def _halka(n: int, d: float) -> Molecule:
    """Testin KENDI halka kurucusu.

    `downfold.hydrogen_ring` ITHAL EDILMEZ: iki taraf ayni kurucuyu
    paylasirsa onu bozan bir mutant hem yasayi hem karsilastirma hedefini
    AYNI yonde kaydirir ve test sessizce yesil kalir. Ayni gerekce
    `test_downfold_kesme.py`de olculerek yazilmisti.
    """
    yaricap = d / (2 * np.sin(np.pi / n))
    a = yaricap * BOHR_TO_ANGSTROM
    return Molecule.from_string(
        "; ".join(
            f"H {a*np.cos(2*np.pi*k/n):.17g} {a*np.sin(2*np.pi*k/n):.17g} 0"
            for k in range(n)
        )
    )


def _ekstrapolasyon_orani(laws, gercek_t1: np.ndarray) -> float:
    """ORAN = (pencere disi log artik RMS) / (pencere ici log artik RMS).

    Pay ve payda AYNI dogrudan gelir; tek fark hangi d'lerde okundugudur.
    """
    tahmin = np.array([laws.at(d).hoppings[0] for d in HEDEF])
    log_artik = np.abs(np.log(np.abs(tahmin)) - np.log(np.abs(gercek_t1)))
    e_out = float(np.sqrt((log_artik**2).mean()))
    return e_out / laws.t_log_residual_rms[0]


def _sahte_baz_verisi(rng):
    """Fizik-disi DUGUMSUZ baz: 1-3 primitif, us 0.05-10, POZITIF katsayi.

    "Dugumsuz" onemlidir: negatif katsayiya izin verilseydi radyal dugumlu
    fonksiyonlar cikar ve ikame "rastgele baz" degil "bozuk baz" olurdu.
    Amac fizigi CIKARMAK, sayisal olarak patlatmak degil.
    """
    n_prim = int(rng.integers(1, 4))
    usler = np.exp(rng.uniform(np.log(0.05), np.log(10.0), n_prim))
    katsayilar = rng.uniform(0.1, 1.0, n_prim)
    return {
        "H": [
            {
                "angular_momentum": 0,
                "primitives": [
                    [float(u), float(c)] for u, c in zip(usler, katsayilar)
                ],
            }
        ]
    }


# ---------------------------------------------------------------------------
# YAPISAL testler -- SCF kosmazlar, milisaniyedirler, `slow` DEGILDIRLER.
# ---------------------------------------------------------------------------


def test_IKI_NOKTALI_pencere_REDDEDILIYOR():
    """SINIF: TRANSKRIPSIYON DENETIMI.

    Iki nokta iki parametreli bir uyumun artigini OZDES SIFIR yapar; o
    hâlde `t_log_residual_rms` her zaman 0.0 doner ve "yasa oturuyor"
    cumlesi olculemez hale gelir -- test yesil kalir, hicbir sey soylemez.
    Uc nokta, artigin ilk kez BIR SEY soyledigi yerdir.
    """
    with pytest.raises(InvalidInputError, match="UC"):
        fit_distance_laws(HALKA, (2.0, 2.2))


def test_TEKRARLANAN_mesafe_REDDEDILIYOR():
    """Ayni d iki kez verilirse serbestlik derecesi sessizce kaybolur:
    (2.0, 2.0, 2.2) uc noktali gorunur ama IKI noktalidir."""
    with pytest.raises(InvalidInputError, match="tekrarlanan"):
        fit_distance_laws(HALKA, (2.0, 2.0, 2.2))


# ---------------------------------------------------------------------------
# FIZIK sayilariyla kosan testler. `slow` isaretleri OLCULDU (dosya sonu).
# ---------------------------------------------------------------------------


def test_YASA_TABAN_DURUM_uzerinde_olculdu():
    """SINIF: TRANSKRIPSIYON DENETIMI -- ama ATLANMASI en pahali olan satir.

    Kullanilan HER d'de RHF spin KARARSIZ olmali: kararli olsaydi UHF
    indirgemesi gereksiz olurdu ve bu dosyanin yasalari HF'in taban
    durumunu degil bir eyer noktasini tarif ederdi.

    RHF adayinda 45 noktanin 35'i eyerdi ve bu ancak KONTROL EDILINCE
    goruldu; o kontrol bu satirdir. Burada yon TERSTIR: UHF kullandigimiz
    icin RHF'in kararsiz olmasini ISTIYORUZ.

    OLCULDU (N=8, STO-3G, triplet Hessian'in en kucuk ozdegeri):
        d=1.8  -0.390441095      d=2.4  -0.861455504
        d=2.0  -0.399689718      d=2.6  -0.428240753
        d=2.2  -0.409562451      d=3.0  -1.203060619
    Altisi da NEGATIF, yani alti noktanin altisinda da RHF spin kararsiz.
    Bu, Modul 1C'nin Huckel hukmuyle tutarli: N=8 bir 4n halkasidir ve
    TB spektrumunun HOMO/LUMO'su tam dejeneredir.

    d=2.4 ve d=3.0'daki BUYUK sicramalar (-0.86 ve -1.20) SAKLANMAZ ve
    yorumlanmaz: `solve` bir ARAMA protokolu degildir ve o iki geometride
    baska bir RHF cozumune yakinsiyor. Testin iddiasi yalnizca ISARETTIR;
    buyukluk kilitlenmez cunku olculen sey buyukluk degildir.

    `slow` DEGIL: olculen sure 0.57 s (alti `solve` + alti Hessian).
    """
    for d in PENCERE + HEDEF:
        result = solve(_halka(HALKA, d), "sto-3g", stability=False, max_iterations=8000)
        lam = float(external_stability(result).eigenvalues.min())
        assert lam < 0.0, f"d={d}: RHF KARARLI (lam={lam:+.9f}), UHF gereksiz"


@pytest.mark.slow
def test_YASALAR_pencere_icinde_oturuyor():
    """SINIF: TRANSKRIPSIYON DENETIMI -- yasanin uydurma kalitesini yazar.

    OLCULDU (N=8, pencere 1.8/2.0/2.2 bohr, seeds=12):
        ln|t1| = 0.702761 - 0.799362 d    artik RMS 1.855587e-03
        ln|t2| = -0.377051 - 1.342071 d   artik RMS 4.104227e-04
        eps    = -0.076343243 + 0.801465630 t1^2   artik RMS 5.737638e-04
        ln|D|  = -2.480483162 + 0.421523651 d      artik RMS 4.743131e-03

    Nokta-nokta log artiklari: t1 1.312098e-03 / 2.624197e-03 / 1.312098e-03,
    t2 2.902127e-04 / 5.804253e-04 / 2.902127e-04. (Uc noktali dogrusal
    uyumda uclu desen zorunlu olarak h, -2h, h bicimindedir; ortadaki
    nokta iki kat artik tasir.)

    ISARETLER de kilitlenir ve UYDURULMAZ: t1 uc noktada da NEGATIF, t2 uc
    noktada da POZITIF. `fit_distance_laws` isareti pencerede DOGRULAR ve
    donerse `UnsupportedSystemError` firlatir; sinavda bu kontrol gercekten
    atesledi (41 cizimin 1'i t2 isaret degistirdigi icin reddedildi).

    EPS ICIN BIR INCELIK OLCULDU ve saklanmaz: `epsilon_residual_rms`
    (5.737638e-04) uyumun GERCEK t1 ile birakti artiktir, ama `at(d)`
    epsilon'u YASANIN t1'inden hesaplar. Ikisi ayni sey degildir ve
    olculdugunde bilesik hata DAHA KUCUK cikiyor (pencere ici nokta-nokta
    1.533742e-04 / 1.078354e-04 / 2.222676e-04, yani max 2.222676e-04,
    tek basina uyum artiginin max'i 8.069509e-04 iken). Iki hata KISMEN
    BIRBIRINI GOTURUYOR. Bu bir kazanc degil bir tesaduftur ve pencere
    disinda tersine doner (asagidaki test).

    `slow` ISARETI OLCUM: dosyanin sonundaki nota bakiniz.
    """
    laws = fit_distance_laws(HALKA, PENCERE, "sto-3g", n_neighbour=2, seeds=TOHUM)

    assert laws.t_signs == (-1.0, 1.0)
    assert laws.window_bohr == PENCERE
    assert laws.t_log_residual_rms[0] == pytest.approx(1.855587e-03, rel=1e-4)
    assert laws.t_log_residual_rms[1] == pytest.approx(4.104227e-04, rel=1e-4)
    assert laws.epsilon_residual_rms == pytest.approx(5.737638e-04, rel=1e-4)
    assert laws.delta_log_residual_rms == pytest.approx(4.743131e-03, rel=1e-4)
    assert laws.epsilon_coefficient == pytest.approx(0.801465630, rel=1e-6)
    assert laws.t_decays[0] == pytest.approx(0.799362, rel=1e-5)
    assert laws.t_decays[1] == pytest.approx(1.342071, rel=1e-5)

    for d in PENCERE:
        gercek = downfold_uhf(_halka(HALKA, d), "sto-3g", seeds=TOHUM)
        yasa = laws.at(d)
        for m, (t_g, t_y) in enumerate(zip(gercek.hoppings, yasa.hoppings)):
            assert np.sign(t_y) == np.sign(t_g), f"d={d}: t{m+1} isareti donmus"
            log_artik = abs(np.log(abs(t_y)) - np.log(abs(t_g)))
            assert log_artik < ARTIK_RMS_SINIRI, (
                f"d={d}: t{m+1} log artigi {log_artik:.6e}"
            )
        assert abs(yasa.epsilon_bar - gercek.epsilon_bar) < 3e-4
        assert abs(yasa.delta - abs(gercek.delta)) / abs(gercek.delta) < 1e-2

    # Yasa OLCULMEMIS alanlari UYDURMAZ: uc alan NaN doner.
    yasa = laws.at(2.0)
    assert np.isnan(yasa.s_squared)
    assert np.isnan(yasa.energy_uhf)
    assert np.isnan(yasa.staggering_residual)
    assert "SCF KOSULMADI" in yasa.search_line
    # Hukum de UYDURULMAZ (B7-c): SCF yok -> "olculmedi" (None), True DEGIL.
    assert yasa.search_stable is None


@pytest.mark.slow
def test_YASA_pencere_DISINI_ongoruyor_ve_MUTLAK_hata_da_yaziliyor():
    """SINIF: TRANSKRIPSIYON DENETIMI.

    Oran metrigi TEKBICIM KOTU olmakla kandirilabilir -- RHF tarafinda
    olculdu: Hcore vekilinin orani ~1.00'di cunku YERINDE uydurmasi da
    berbatti (1-4 Ha). Bu yuzden MUTLAK hata da kilitlenir. Sinavda ayni
    tuzagin ters yonu de goruldu: 40 sahtenin 7'si gercekten kucuk mutlak
    hata veriyor, ikisi yalnizca t1'leri ~0 oldugu icin.

    OLCULDU (pencere DISI, seeds=12):
        d      t1 gercek       t1 yasa        bagil      mutlak (Ha)
        2.4   -0.298736387   -0.296500211   0.007485   2.236176e-03
        2.6   -0.255454661   -0.252693055   0.010811   2.761606e-03
        3.0   -0.183164131   -0.183539652   0.002050   3.755209e-04

    t2 pencere disinda t1'DEN IYI ongoruluyor ve bu beklenmedikti:
    bagil hatalari 0.000092 / 0.001144 / 0.000731, yani t1'in ~onda biri.
    Sebep olculmedi ve UYDURULMAZ; yalnizca kaydedilir. (Bir tahmin
    yazilabilirdi -- t2'nin log-uyumu daha temiz, artik RMS'i t1'inkinin
    ~dortte biri -- ama bu bir OLCUM degil.)

    eps pencere disinda BOZULUYOR: fark 9.146090e-04 / 1.652889e-03 /
    2.602950e-03 hartree, yani pencere icindekinin (max 2.222676e-04)
    ~12 kati. Yukaridaki testte not edilen "hatalar birbirini goturuyor"
    tesadufu burada tersine donuyor.

    `slow` ISARETI OLCUM: dosyanin sonundaki nota bakiniz.
    """
    laws = fit_distance_laws(HALKA, PENCERE, "sto-3g", n_neighbour=2, seeds=TOHUM)

    olculen_bagil, olculen_mutlak = [], []
    for d in HEDEF:
        gercek = downfold_uhf(_halka(HALKA, d), "sto-3g", seeds=TOHUM)
        yasa = laws.at(d)
        mutlak = abs(yasa.hoppings[0] - gercek.hoppings[0])
        bagil = mutlak / abs(gercek.hoppings[0])
        olculen_bagil.append(bagil)
        olculen_mutlak.append(mutlak)
        assert bagil < ONGORU_BAGIL_SINIRI, f"d={d}: bagil hata {bagil:.6f}"
        assert mutlak < ONGORU_MUTLAK_SINIRI, f"d={d}: mutlak hata {mutlak:.6e} Ha"
        assert abs(yasa.hoppings[1] - gercek.hoppings[1]) / abs(
            gercek.hoppings[1]
        ) < 2e-3
        assert abs(yasa.epsilon_bar - gercek.epsilon_bar) < 4e-3

    assert max(olculen_bagil) == pytest.approx(0.010811, rel=1e-3)
    assert max(olculen_mutlak) == pytest.approx(2.761606e-03, rel=1e-3)


@pytest.mark.slow
def test_DELTA_YASASI_PENCERE_DISINDA_KIRILIYOR():
    """SINIF: TRANSKRIPSIYON DENETIMI -- DURUST OLUMSUZ.

    Delta'nin ustel yasasi t1'inki gibi DEGIL: pencere icinde oturuyor
    (log artik RMS 4.743131e-03) ama pencere disinda hizla bozuluyor.
    OLCULEN bagil hatalar: 0.028079 (d=2.4), 0.055498 (d=2.6), 0.069184
    (d=3.0) -- t1'in max 0.010811'inin ALTI KATI, ve d ile MONOTON artiyor.

    Sebep olculdu ve `downfold.py` modul dokstringinde yaziyor: Delta
    DOYUYOR. Buyuyen bir ustel doyan bir egriyi uzun menzilde tarif edemez;
    N=8'de |Delta| d=1.6'da 0.170033 iken d=3.6'da 0.363453'te doyuyor ve
    ayni pencere yasasi orada %5.0 YUKSEK tahmin veriyor (0.381747).
    Doymanin sebebi de olculdu: Hubbard ortalama alan bosluk denklemi
    olculen (t1, Delta) ciftinden U'yu cozdurdugunde U d = 1.6 ... 3.6
    boyunca yalnizca %11 oynuyor (0.873471 -> 0.784102) ve BAGIMSIZ olarak
    hesaplanan Lowdin yerinde Coulomb integraliyle ayni yere dusuyor
    (d=3.6'da 0.784102'ye karsi 0.788356071, %0.5).

    HUKUM: Delta'nin mesafe yasasi PENCERE ICI bir aractir, bir yasa
    DEGILDIR; test bunu %10 sinirla KABUL EDER ve boyle beyan eder.
    Toleransi t1'inkine cekmek testi kirmizi yapardi -- ve sayiyi degil
    IDDIAYI duzeltmek gerekirdi.
    """
    laws = fit_distance_laws(HALKA, PENCERE, "sto-3g", n_neighbour=2, seeds=TOHUM)

    bagillar = []
    for d in HEDEF:
        gercek = downfold_uhf(_halka(HALKA, d), "sto-3g", seeds=TOHUM)
        bagil = abs(laws.at(d).delta - abs(gercek.delta)) / abs(gercek.delta)
        bagillar.append(bagil)
        assert bagil < DELTA_ONGORU_BAGIL_SINIRI, f"d={d}: |D| bagil {bagil:.6f}"

    assert bagillar == pytest.approx([0.028079, 0.055498, 0.069184], rel=1e-3)
    assert bagillar[0] < bagillar[1] < bagillar[2], "Delta hatasi monoton degil"
    # ve t1'inkinden acikca KOTU -- yasanin zayif halkasi BUDUR.
    assert max(bagillar) > 5.0 * 0.010811


def test_GECIS_ALTINDAKI_pencere_REDDEDILIYOR():
    """SINIF: TRANSKRIPSIYON DENETIMI -- kapsam kontrolunun kendisi.

    N=6 halkasinin gecis noktasi 2.233818500 bohr (Task 2, triplet
    Hessian'in kokunden). Onun ALTINDA UHF, RHF'e coker ve Delta tam
    sifirdir; ln|Delta| orada tanimsizdir.

    Kontrol olmasaydi `np.polyfit` -inf ile calisir ve butun yasa NaN
    dolardi -- sessizce. OLCULDU: (1.9, 2.0, 2.1) penceresinde min |Delta|
    = 3.336e-13, yani esigin (1e-7) bes mertebe altinda.

    Bu ikame ayrica F2'yi (eri * 0) de eleyen kontroldur; orada olculen
    min |Delta| 1.110e-16 idi.

    `slow` DEGIL ve bu ISARETLENDIKTEN SONRA OLCULEREK GERI ALINDI: test
    once `slow` yazilmisti (uc SCF kosuyor diye), olculen sure 0.76 s cikti
    ve deponun 2 s esiginin altinda kaldi. N=6 halkasi N=8'den ucuzdur ve
    gecisin altinda UHF hemen RHF'e cokup erken yakinsar.
    """
    with pytest.raises(UnsupportedSystemError, match="spin-KIRILMAMIS"):
        fit_distance_laws(6, (1.9, 2.0, 2.1), "sto-3g", seeds=TOHUM)


@pytest.mark.slow
def test_SAHTE_CIPA_SINAVININ_SONUCU():
    """SINIF: DAIRESEL DEGIL -- ama DAR KAPSAMLI, ve kapsam burada yazilidir.

    IDDIA EDILEN SEY: t1'in ustel mesafe yasasi BORU HATTI-JENERIK
    DEGILDIR. Yani "ln|t| dogrusal cikti" sonucu, uydurma makinesinin her
    girdiye verdigi bir cevap degil, bu sistemin elektronik yapisina ait
    bir olgudur.

    IDDIA EDILMEYEN SEY: yasanin iyi ekstrapole etmesinin fizik oldugu.
    Etmiyor -- bkz. test_SAHTE_CIPA_IKINCI_METRIGI_AYIRMIYOR.

    SINAVIN SAYISI (21.08.2026, sevk edilen kodla, seeds=12):
        GERCEK oran                       4.160383
        40 sahte dugumsuz baz: medyan    29.902685
                               en iyi     6.216006
                               en kotu  413.126243
        gercegi GECEN                     0/40
    Planin olcutu: sahte medyan > 4 x gercek sinir. 29.902685 > 20.0.
    RHF tarafinda ayni sinav 2.80'e karsi 141.65 medyan ve 12.01 en iyi
    vermisti (0/40); yani UHF yolunda sinav GECIYOR ama medyan orani
    50.6'dan 7.19'a, kuyruk marji 4.29 kattan 1.49 kata DUSUYOR.

    BU TEST SINAVIN BASINI GERCEKTEN YENIDEN KOSAR, yalnizca sayi
    kopyalamaz: ayni RNG akisindan ilk ALTI baz cizilir; beside sinav
    kosulur ve oranlari yukaridaki tabloyla eslesir, altincisi ise
    `fit_distance_laws`in isaret kontrolune takilir.

    OLCULEN ilk bes oran: 25.5851, 29.8088, 37.6266, 10.6755, 29.6049.
    Altinci cizim: t2 pencerede isaret degistiriyor -> UnsupportedSystemError.
    """
    laws = fit_distance_laws(HALKA, PENCERE, "sto-3g", n_neighbour=2, seeds=TOHUM)
    gercek_t1 = np.array(
        [downfold_uhf(_halka(HALKA, d), "sto-3g", seeds=TOHUM).hoppings[0] for d in HEDEF]
    )
    gercek_oran = _ekstrapolasyon_orani(laws, gercek_t1)
    assert gercek_oran == pytest.approx(GERCEK_ORAN, rel=1e-4)
    assert gercek_oran < ORAN_SINIRI

    rng = np.random.default_rng(20260821)
    asil = basis_modulu._load_reference_data
    oranlar, reddedilen = [], 0
    try:
        for _ in range(6):
            veri = _sahte_baz_verisi(rng)
            basis_modulu._load_reference_data = lambda _ad, _v=veri: _v
            try:
                sahte_laws = fit_distance_laws(
                    HALKA, PENCERE, "sto-3g", n_neighbour=2, seeds=TOHUM
                )
                sahte_t1 = np.array(
                    [
                        downfold_uhf(_halka(HALKA, d), "sto-3g", seeds=TOHUM).hoppings[0]
                        for d in HEDEF
                    ]
                )
            except UnsupportedSystemError:
                reddedilen += 1
                continue
            oranlar.append(_ekstrapolasyon_orani(sahte_laws, sahte_t1))
    finally:
        basis_modulu._load_reference_data = asil

    assert reddedilen == 1, "6. cizim boru hattinin kapsam kontrolune takilmaliydi"
    assert oranlar == pytest.approx(
        [25.5851, 29.8088, 37.6266, 10.6755, 29.6049], rel=1e-3
    )
    for oran in oranlar:
        assert oran > gercek_oran, f"sahte baz gercegi gecti: {oran:.4f}"
    assert min(oranlar) > 2.0 * gercek_oran

    # Ve 40'lik sinavin kilitlenmis hukmu.
    assert SAHTE_GECEN == 0
    assert SAHTE_MEDYAN > 4.0 * ORAN_SINIRI
    assert SAHTE_EN_IYI > GERCEK_ORAN


def test_SAHTE_CIPA_IKINCI_METRIGI_AYIRMIYOR():
    """SINIF: DAIRESEL DEGIL (KARSIT YON) -- sinavin SINIRINI kilitler.

    Bu test bir seyin CALISTIGINI degil, CALISMADIGINI kilitler: "yasa
    pencere disini iyi ongoruyor" cumlesi TEK BASINA fizik kaniti
    DEGILDIR. Olculdu, ayni sinavda, ayni 40 sahte baz uzerinde:

        bagil ongoru hatasi, gercek           0.010811
        40 sahtenin medyani                   0.130292
        40 sahtenin EN IYISI                  0.005409
        gercegi GECEN sahte                   2/40

    Ve saf geometri cekirdeginde (F3, `S = exp(-a r)`, `H = S^-1` --
    icinde HICBIR Hamiltoniyen yok) durum daha da net:

        a     oran       bagil ongoru hatasi
        0.5   21.8168    0.059430
        0.8   18.5425    0.026504
        1.0   16.4757    0.013007
        1.5    9.9647    0.000997      <- gercekten 11 KAT IYI
        2.0   29.3917    0.000297      <- gercekten 36 KAT IYI

    Yani bes cekirdegin IKISI gercek sistemden iyi ongoruyor. Sasirtici
    degil: `exp(-a r)` tanimi geregi usteldir, yani "ustel bir yasaya
    uyuyor" olmak bir cekirdegi fiziksel YAPMAZ. Ayirt eden sey ORANDIR
    (o bes cekirdegin BESI de gercegin 4.160383'unun ustunde) ve ORAN
    olculurken mutlak buyukluk kilitlenmezse metrik ters yonden
    kandirilabilir -- 40 sahtenin 7'si mutlak hatada gercegi geciyor,
    ikisi yalnizca t1'leri ~0 oldugu icin (bagil hatalari 16.67 ve 8.85).

    HUKUM: sinavin gecerli tek okumasi ORAN + IKI SINIRIN BIRLIKTE
    kilitlenmesidir. Tek metrik yeterli DEGILDIR ve bu, sinavin gectigi
    kadar onemli bir sonucudur.

    Test SCF kosmaz: F3 tamamen geometriktir ve milisaniyedir.
    """
    kernel_sonuclari = {}
    for a in (0.5, 0.8, 1.0, 1.5, 2.0):
        t1 = []
        for d in PENCERE + HEDEF:
            r = d / (2 * np.sin(np.pi / HALKA))
            aci = 2 * np.pi * np.arange(HALKA) / HALKA
            poz = np.column_stack([r * np.cos(aci), r * np.sin(aci), np.zeros(HALKA)])
            mesafe = np.linalg.norm(poz[:, None, :] - poz[None, :, :], axis=-1)
            t1.append(np.linalg.inv(np.exp(-a * mesafe))[0, 1])
        t1 = np.array(t1)

        dp, dh = np.array(PENCERE), np.array(HEDEF)
        egim, kesisim = np.polyfit(dp, np.log(np.abs(t1[:3])), 1)
        e_in = float(
            np.sqrt(((kesisim + egim * dp - np.log(np.abs(t1[:3]))) ** 2).mean())
        )
        e_out = float(
            np.sqrt(((kesisim + egim * dh - np.log(np.abs(t1[3:]))) ** 2).mean())
        )
        tahmin = np.exp(kesisim + egim * dh)
        kernel_sonuclari[a] = (
            e_out / e_in,
            float((np.abs(tahmin - np.abs(t1[3:])) / np.abs(t1[3:])).max()),
        )

    oranlar = np.array([v[0] for v in kernel_sonuclari.values()])
    bagillar = np.array([v[1] for v in kernel_sonuclari.values()])

    assert oranlar == pytest.approx(
        [21.8168, 18.5425, 16.4757, 9.9647, 29.3917], rel=1e-4
    )
    assert bagillar == pytest.approx(
        [0.059430, 0.026504, 0.013007, 0.000997, 0.000297], rel=1e-3
    )
    # ORAN metrigi besini de eliyor...
    assert np.all(oranlar > GERCEK_ORAN)
    # ...ama BAGIL ONGORU metrigi ikisini ELEYEMIYOR. Iddia budur.
    assert int((bagillar < 0.010811).sum()) == 2


# ---------------------------------------------------------------------------
# `slow` ISARETLERININ GEREKCESI -- OLCUM, KURAL DEVRALMA DEGIL.
#
# Deponun isaretleme esigi Task 2'de SECILDI ve gerekcesi orada yazildi
# (`test_downfold_cipa.py`): devralinabilir bir esik YOKTUR -- `-m slow`
# seridinin en hizli cagrisi 0.01 s, `-m "not slow"` seridinin en yavasi
# 4.54 s, yani bantlar tamamen cakisiyor. Secilen kural: kendi call suresi
# 2 s ve USTUNDE olan test isaretlenir.
#
# OLCULEN SURELER (bu dosya, --durations, 21.08.2026):
#
#   ISARETLI SERIT   4 test, 39.95 s   en HIZLI cagri 3.47 s
#       26.72  test_SAHTE_CIPA_SINAVININ_SONUCU
#        4.93  test_DELTA_YASASI_PENCERE_DISINDA_KIRILIYOR
#        4.42  test_YASA_pencere_DISINI_ongoruyor_ve_MUTLAK_hata_da_yaziliyor
#        3.47  test_YASALAR_pencere_icinde_oturuyor
#   HIZLI SERIT      5 test,  1.68 s   en YAVAS cagri 0.73 s
#        0.73  test_GECIS_ALTINDAKI_pencere_REDDEDILIYOR
#        0.51  test_YASA_TABAN_DURUM_uzerinde_olculdu
#        <0.005 x3  (iki yapisal red + F3 karsit testi; ucu de SCF kosmaz)
#
# Iki bant bu dosyada CAKISMIYOR (0.73 s ile 3.47 s arasi bos) ve 2 s esigi
# tam oraya dusuyor. Depo genelinde bantlarin cakistigi Task 2'de olculmustu;
# burada cakismamasi bir sans degil, bu dosyada testlerin iki dogal maliyet
# ailesine ayrilmasindandir: `fit_distance_laws` kosanlar ve kosmayanlar.
#
# BIR ISARET OLCUMLE GERI ALINDI: test_GECIS_ALTINDAKI_pencere_REDDEDILIYOR
# once `slow` yazilmisti (uc SCF kosuyor diye); olculen 0.73 s esigin altinda
# kaldi ve isaret kaldirildi. Tahminle isaretlenmez.
#
# Sahte-cipa testi dosyanin en pahali testidir cunku ALTI ayri baz seti
# uzerinde tam bir yasa uydurmasi kosar (her biri uc UHF aramasi + uc
# dogrulama indirgemesi = 36 `search_uhf` cagrisi).
# ---------------------------------------------------------------------------
