"""KAYIT TAMLIĞI — `reflection_coefficient`in kuyruk kapısı (10.10.2026).

NE KİLİTLENİYOR: G1 ön ölçümü (09.10.2026) ölçüm aletinin kısmi yansımada
SESSİZ bir protokol hatası taşıdığını buldu (1 -> 2.25, λ = 20, θ = 75°:
|ρ - r| = 2.01e-02, G1 bedelinden büyük). Kaynak ayrıldı
(`.superpowers/sdd/rc-olcum/`): G1'in iki durumunda FFT bini ve PML
suçsuz; kayıt bittiğinde darbe hâlâ geliyor. Sebepler sabit-`ky` indirgemesinin grup hızı dağılımı
(sıyırma), dal noktasının uzun kuyruğu (kritik açı), kaba ızgarada şemanın
yavaş grup hızı (yoğun ortam, θ = 0'da bile) ve kaynağın sola giden
dalgasının soğurucudan dönüşüdür. Kapı: kaydın SON PERİYODUNDAKİ genlik
(gelen ya da yansıyan) tepenin `tail_gate` katını aşarsa `None`; RC-a'dan
(10.10.2026) beri GELEN kaydında ana darbeden sonra kalanın f'deki
spektral ağırlığı ana darbeninkinin `echo_gate` katını aşarsa da (İKİNCİ
VARIŞ), ve `_record_steps` sağ soğurucunun iki yankısını da dışlıyor.

PENCERE (T2): G1'in `T2_PENCERE`si — 1800 hücre, npml 40, S 0.4, iface 750,
kaynak 300, prob ofseti 6, sgf 6, dikdörtgen, aritmetik, |ρ|=1 kapısı 1e-2,
m = 3 — ve iki kapı alanı AÇIKÇA yazılır (G7 dersi: varsayılan değişirse test
sessizce başka pencereyi ölçmesin). Sevk penceresinin pahalı başlıkları
tek bir `slow` testte.

REFERANS: AYRIK-TAM r_s (kapalı form, testte elle; G1'in `_r_ayrik`i),
prob düzlemine taşınmış ve sevk uzlaşımında (eşlenik). Kapalı form yalnız
s'tedir; p'nin ayrık-tam referansı ön ölçümde transfer özyinelemesiyle
kuruldu (temiz açılarda alete 5e-9 .. 1e-7'de oturur) ve burada
kullanılmaz.

ÖLÇÜLDÜ (T2, kapı kapalı = `tail_gate` 1, `echo_gate` 1; `rc-olcum/f_beyan.log`,
`g_hizli_adaylar.log`, `g_bant_kenari.log`):

    durum                                   kuyruk(T_min)  kapalı hata
    1 -> 2.25, 70°, λ 10      (sıyırma)      7.50e-01       3.84e-02
    2.25 -> 1, 41.8°, λ 10    (kritik)       4.41e-02 (y)   5.07e-02
    4 -> 1, 0°, λ 10          (kaba ızgara)  7.18e-01       4.23e-02
    1 -> 2.25, 0°, λ 10, npml 0, ofset 150   7.80e-01 (g)   6.97e-02
    2.25 -> 1, 50°, λ 10      (bant üstü)    3.09e-03       2.16e-04
    2.25 -> 1, 30°, λ 10      (bant altı)    7.42e-04       1.17e-05
    1 -> 2.25, 20°, λ 10      (temiz)        1.80e-05       1.23e-07

İkinci satırda gelenin kuyruğu 5.65e-05, dördüncüde yansıyanınki 5.25e-05:
iki seri AYRI AYRI gerekiyor. Dördüncü satırın yankısı soğurucusuz pencerede
kaynağın SOLA giden dalgasıdır; PEC duvardan döner ve gelen kaydına
tepesiyle girer (yansıyana girmeden önce kayıt biter).

DÜRÜST OLUMSUZ 1 — KAPI BİR KANIT DEĞİL, ÖLÇÜLMÜŞ BİR ORAN: T2'de kapıdan
GEÇEN iki sayının hatası 1e-4'ü aşıyor (2.25 -> 1, 45°: λ 20'de 2.05e-04,
kuyruk 1.37e-03; λ 10'da 1.82e-04, 1.57e-03). Sevk penceresinde (prob
ofseti 6) ızgara noktalarında geçenlerin en kötüsü 8.28e-05, ızgara dışında
1.49e-04 (bağımsız doğrulayıcı, 2.25 -> 1, 44°, λ 30); hata/kuyruk orada
<= 0.54 (yankı kayda tepesinden girer).
KAYIT İÇİ YANKI — KAPANDI (RC-a, 10.10.2026; `rc-olcum/rca/`). Doğrulayıcı
(`rc-olcum/dogrulama/`) kuyruk kapısının kör noktasını bulmuştu: prob
arayüzden uzaksa soğurucunun bir yankısı GELEN kaydına TAMAMEN girer,
yansıması kayıt bitmeden YANSIYANA varmaz; kaydın sonu sessizdir. Ölçülen:
9 -> 1, 30°, λ 40, T2 ofset 400'de hata 6.20e-03 (kuyruk 2.15e-05, oran
288). Üç aday aynı koşum çiftinden ölçüldü: (a) gelen kaydında ana darbe
(t_dir + 5 sg) SONRASI ikinci varış; (b) `_record_steps`e sağ soğurucunun
iki yankısı; (c) gelen kaydını yankıdan önce sıfırlamak. (c) ELENDİ
(kapısı açık 62 sabitlenmiş çağrının 26..27'sini değiştirir, sevk ana
çıpasını 1.2e-07'den 1.25e-05'e bozar). (a)'nın İLK sürümü zaman-alanı tepe
oranıydı ve kuyrukla aynı eşiği (1.6e-3) kullanıyordu; ikinci doğrulayıcı
onunla GEÇEN bir sayı buldu (T2 ofset 410, 9 -> 1, 48.25°, λ 42, p: hata
1.84e-03 > τ; ince PML'de eğik yankı genişler, tepesi spektral ağırlığını
%12..19 eksik okur). Uygulanan (a) SPEKTRAL: c = |F(gelen[i:])| /
|F(gelen[:i])|; yansıyanda karşılığı olmayan kalan için |hata| <=
|r| c / (1 - c), yankının şeklinden bağımsız (2986 durumda yankı baskın
444'ünde, ikinci turun 61'inde %5 payla hiç aşılmadı). Eşik ayrı düğme `echo_gate` = 4.5e-4,
ölçülen banttan (`rc-olcum/rca/spek_esik_4p5.log`). (a) + (b) sonrası
sabitlenmiş sayılardan yalnız eski kör nokta testinin durumu `None` oldu.
DÜRÜST OLUMSUZ 2 — EŞİĞİN ALTINDAKİ YANKI GEÇER, VE KUYRUKLA TOPLANIR:
kapı yankıyı GÖRÜR ama c <= `echo_gate` ise sayıyı geçirir, hatası ~|r| c;
kuyruk da kendi eşiğinin altındaysa ikisi TOPLANIR (ikinci doğrulayıcı: T2
ofset 430, 4 -> 1, 32°, λ 24: c 4.45e-04, kuyruk 1.06e-03, hata 6.16e-04).
Ölçülen toplam sınır (769 geçen sayı): hata <= |r| c/(1 - c) + 0.17 kuyruk,
yani geçen sayı için ≲ `echo_gate` + 0.17 τ ≈ 7.2e-04 diye okunur.
FFT PENCERESİ — KAPANDI (RC-d, 10.10.2026; RC-a'dan önce de vardı):
hann / blackman pencereleri erken gelen darbeyle geç gelen yansıyanı farklı
ağırlıkla çarpıyordu ve üç kapı da görmüyordu (T2, 4 -> 2.25, 40°, λ 20:
hann 2.56e-02, blackman 3.56e-02; dikdörtgende 1.4e-07). Ön ölçüm
(`rc-olcum/rcd/`, 2803 satır): dikdörtgenin sayı verdiği 646 kaydın
hiçbirinde pencere daha iyi değil (r = 0'da eşit). Darbeleri farklı
ağırlıklandıran hiçbir düzeltme adayı zarfı tutturmadı; tutturan tek aday
(yalnız son periyotta konik) dikdörtgenin kendisi → `window_fn`
"dikdortgen" değilse `InvalidInputError`, koşumdan önce. İki test reddi
(her kipte), bir test GEREKÇEYİ (hata ≈ |r| |w(t_y)/w(t_g) - 1|) kilitler.

SINIF: kapalı-form referanslı ÇEVİRİ DENETİMİ + `None` disiplini (Modül 5
sözlüğü, `test_arayuz.py` başlığı). Özel import tek bir TRANSKRİPSİYON
testinde: `_kuyruk`un penceresi (tam bir periyot) sentetik diziyle sınanır,
çünkü pencere uzunluğu kamu yolunda eşiğin içinde kaybolur (doğrulayıcının
yarım/iki periyot mutantları kamu yolu testlerinin hepsini geçti). RC-d'nin
gerekçe testi de iç koşum çiftini (`_kosum`, `_spektrum`) kullanır, çünkü
kamu yolu artık pencereli sayı üretmez; dikdörtgen oranını kamu yolunun
sayısına BİT BİT eşitleyerek kurulumunu sınar.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qlab.engine.em.boundary import PML
from qlab.engine.em.medium import Medium
from qlab.errors import InvalidInputError
from qlab.observable import em_obs
from qlab.observable.em_obs import (
    _FFT_PENCERELERI,
    DEFAULT_ECHO_GATE,
    DEFAULT_TAIL_GATE,
    ObliqueWindow,
    _ikinci_varis,
    _kosum,
    _kuyruk,
    _record_steps,
    _spektrum,
    reflection_coefficient,
    tir_phase_difference,
    tir_phase_ladder,
    total_internal_reflection,
)

#: G1'in T2 penceresi + kuyruk ve yankı kapıları, HER ALAN açık.
T2 = ObliqueWindow(
    cells=1800,
    npml=40,
    courant=0.4,
    interface=750,
    source_index=300,
    probe_offset=6,
    sgf=6.0,
    window_fn="dikdortgen",
    ey_kural="aritmetik",
    unity_gate=1e-2,
    pml_m=3.0,
    tail_gate=1.6e-3,
    echo_gate=4.5e-4,
)
#: Kapılar KAPALI: kuyruk ve ikinci varış oranları tanım gereği <= 1 (ikinci
#: varış 1'de kırpılır), yani 1.0 her kaydı kabul eder ve alet kapılardan
#: önceki hâline BİT BİT döner. `_record_steps`in GEOMETRİK sınırları (sol ve
#: sağ soğurucu yankıları) bu düğmelere bağlı değildir; T2'de bu dosyanın
#: KAPALI kullanan hiçbir durumunda bağlamaz.
KAPALI = dataclasses.replace(T2, tail_gate=1.0, echo_gate=1.0)

#: Bandın iki ucu (sevk penceresi, `rc-olcum/d_bant.log` ve
#: `e_cipa_hatalari.log`; ÖLÇÜLDÜ, seçilmedi).
CIPA_EN_BUYUK_KUYRUK = 9.44e-04  # eps 4 -> 1.8225, 45°, λ 40, p
SESSIZ_EN_KUCUK_KUYRUK = 2.78e-03  # eski aletin hatası > 1e-4 olanlar


def _r_prob(eps1: float, eps2: float, theta: float, lam: float, w: ObliqueWindow) -> complex:
    """Ayrık-tam r_s, PROB düzleminde ve sevk uzlaşımında — TESTTE ELLE.

    G1 kapalı formu (arayüz yüzü I-1/2): r = (a1 c2 - c1 a2)/(a1 c2 + c1 a2),
    a_j = sqrt(eps_j Om^2 - ky^2)/2, Om = 2 sin(ω S/2)/S, ky = k0 n1 sin θ.
    Prob'a taşıma ve eşlenik: ρ = conj(r exp(-2 i k1 (prob - (iface - 1/2)))).
    GEÇERLİLİK: prob kaynağın SAĞINDA (`probe_offset < iface - source_index`);
    prob kaynağın solundaysa "gelen" kayıt başka bir şeydir ve bu form yanlış
    (doğrulayıcı ölçtü, ofset 1000'de 3.1e-02). Bu dosyanın hiçbir durumu
    orada değil.
    """
    assert w.probe > w.source_index
    S = w.courant
    om = 2.0 * math.sin(math.pi / lam * S) / S
    ky = 2.0 * math.pi / lam * math.sqrt(eps1) * math.sin(math.radians(theta))
    a1 = np.sqrt(complex(eps1 * om * om - ky * ky)) / 2.0
    a2 = np.sqrt(complex(eps2 * om * om - ky * ky)) / 2.0
    if a2.imag < 0:
        a2 = -a2
    c1, c2 = np.sqrt(1.0 - a1 * a1), np.sqrt(1.0 - a2 * a2)
    r = (a1 * c2 - c1 * a2) / (a1 * c2 + c1 * a2)
    k1 = 2.0 * np.arcsin(a1).real
    xp = w.probe - (w.interface - 0.5)
    return complex(np.conj(r * np.exp(-2j * k1 * xp)))


def _iki(eps1, eps2, theta, lam, w=T2):
    ortak = dict(eps1=eps1, eps2=eps2, theta_deg=theta, wavelength=lam, polarization="s")
    acik = reflection_coefficient(window=w, **ortak)
    kapali = reflection_coefficient(
        window=dataclasses.replace(w, tail_gate=1.0, echo_gate=1.0), **ortak
    )
    return acik, kapali


# ---------------------------------------------------------------------------
# Kapı SESSİZ sayıyı kesiyor — dört ayrı mekanizma, dördü de |ρ|=1 kapısının
# KOŞMADIĞI (ya da ıskaladığı) yerde.
# ---------------------------------------------------------------------------
def test_SIYIRMADA_SESSIZ_SAYI_ARTIK_NONE():
    """1 -> 2.25, θ = 70°, λ = 10: kısmi yansıma, |ρ|=1 kapısı koşmaz.

    Kapı kapalıyken alet 3.84e-02 hatalı bir sayı döndürüyordu; kaydın son
    periyodunda genlik tepenin 0.75'i (darbe kayıt bittiğinde ORTASINDA).
    75°'de aynı tablo (5.45e-02, 0.93) daha uzun kayıtla; ucuzu seçildi.
    """
    assert not total_internal_reflection(1.0, 2.25, 70.0)
    acik, kapali = _iki(1.0, 2.25, 70.0, 10.0)
    assert acik is None
    assert kapali is not None
    assert abs(kapali - _r_prob(1.0, 2.25, 70.0, 10.0, T2)) >= 1e-2


def test_KRITIK_ACIDA_SUREKLI_OLCUT_TIY_DEMEZ_ve_KUYRUK_KAPISI_KESER():
    """2.25 -> 1, θ = 41.8°, λ = 10: sürekli ölçüte göre TİY YOK (θc =
    41.81°), şemanınkine göre VAR (ayrık-tam |r| = 1) — yani |ρ|=1 kapısı bu
    açıda koşmaz. Kapı kapalı hata 5.07e-02; kuyruk YALNIZ yansıyanda
    (4.41e-02; gelen 5.65e-05): dal noktasının uzun yansıma kuyruğu.
    """
    assert not total_internal_reflection(2.25, 1.0, 41.8)
    assert abs(abs(_r_prob(2.25, 1.0, 41.8, 10.0, T2)) - 1.0) < 1e-12
    acik, kapali = _iki(2.25, 1.0, 41.8, 10.0)
    assert acik is None
    assert abs(kapali - _r_prob(2.25, 1.0, 41.8, 10.0, T2)) >= 1e-2


def test_DIK_GELISTE_BILE_KABA_IZGARADA_KAYIT_KESIK():
    """4 -> 1, θ = 0, λ = 10 (ortamda 5 hücre/dalgaboyu): şemanın grup hızı
    sürekli 0.5 yerine ~0.39, darbe `_record_steps`in hesapladığı anda değil
    TEPESİYLE varıyor. Kapı kapalı hata 4.23e-02 — açıyla ilgisi yok.
    """
    assert not total_internal_reflection(4.0, 1.0, 0.0)
    acik, kapali = _iki(4.0, 1.0, 0.0, 10.0)
    assert acik is None
    assert abs(kapali - _r_prob(4.0, 1.0, 0.0, 10.0, T2)) >= 1e-2


def test_SOGURUCUSUZ_PENCEREDE_KAYNAK_YANKISI_GELEN_KAYDINDA():
    """1 -> 2.25, θ = 0, λ = 10, npml 0, prob ofseti 150: kaynağın SOLA
    giden dalgası PEC duvardan döner ve GELEN kaydına tepesiyle girer
    (gelen kuyruğu 0.78, yansıyanınki 5.25e-05). Kapı kapalı hata 6.97e-02.
    """
    w = dataclasses.replace(T2, npml=0, probe_offset=150)
    acik, kapali = _iki(1.0, 2.25, 0.0, 10.0, w)
    assert acik is None
    assert abs(kapali - _r_prob(1.0, 2.25, 0.0, 10.0, w)) >= 1e-2


def test_AYNADA_DA_KOSAR_ve_BIRIM_KAPISININ_GECIRDIGINI_KESER():
    """AYNA (`mirror = True`, ε1 = 1, θ = 70°, λ = 10): |ρ| = 1 ZORUNLU, yani
    `unity_gate` koşar — ama kesik kayıtta ||ρ| - 1| = 6.40e-03 < 1e-2 ve
    sayı GEÇERDİ. Kuyruk 0.68 (yansıyan), 0.64 (gelen). Kuyruk kapısı aynada
    da koşmalı (doğrulayıcının N7 mutantı — kapıyı aynada kapatmak — bu
    test yokken hayatta kalıyordu).
    """
    ortak = dict(eps1=1.0, eps2=1.0, theta_deg=70.0, wavelength=10.0,
                 polarization="s", mirror=True)
    assert reflection_coefficient(window=T2, **ortak) is None
    kapali = reflection_coefficient(window=KAPALI, **ortak)
    assert kapali is not None  # |ρ|=1 kapısı (1e-2) GEÇİRİYOR
    assert abs(abs(kapali) - 1.0) >= 1e-3


# ---------------------------------------------------------------------------
# Eşiğin iki yanı — fizikle (yalnız sayı kilidiyle değil, G7 dersi).
# ---------------------------------------------------------------------------
def test_BANDIN_USTUNDE_KUCUK_AMA_GERCEK_HATA_KESILIYOR():
    """2.25 -> 1, θ = 50°, λ = 10: kuyruk 3.09e-03 (eşiğin 1.93 katı), kapı
    kapalı hata 2.16e-04. Eşik 10 kat gevşetilse bu sayı geri dönerdi.
    """
    acik, kapali = _iki(2.25, 1.0, 50.0, 10.0)
    assert acik is None
    assert abs(kapali - _r_prob(2.25, 1.0, 50.0, 10.0, T2)) >= 1e-4


def test_BANDIN_ALTINDA_TEMIZ_SAYI_GECIYOR():
    """2.25 -> 1, θ = 30°, λ = 10: kuyruk 7.42e-04 (eşiğin 0.46'sı), hata
    1.17e-05. Eşik 10 kat sıkılaştırılsa bu temiz sayı `None` olurdu.
    """
    acik, kapali = _iki(2.25, 1.0, 30.0, 10.0)
    assert acik is not None
    assert acik == kapali
    assert abs(acik - _r_prob(2.25, 1.0, 30.0, 10.0, T2)) <= 2e-5


@pytest.mark.parametrize("pol", ["s", "p"])
def test_TEMIZ_DURUMDA_KAPI_SAYIYI_DEGISTIRMEZ_BIT_BIT(pol):
    """1 -> 2.25, θ = 20°, λ = 10: kuyruk 1.80e-05, hata (s) 1.23e-07. Kapı
    YENİ KOŞUM EKLEMEZ; geçtiği her sayı kapıdan önceki sayının kendisidir,
    s'te de p'de de. (G1 T2'nin 45°'si, kuyruk 1.18e-05, aynı kapıdan
    `test_g1_egik_2b.py`de geçer; 45°'nin dört çağrısı tek testte 0.95 s
    sürdüğü için ucuzu seçildi ve kutuplanmaya bölündü.)
    """
    ortak = dict(eps1=1.0, eps2=2.25, theta_deg=20.0, wavelength=10.0, polarization=pol)
    acik = reflection_coefficient(window=T2, **ortak)
    assert acik is not None
    assert acik == reflection_coefficient(window=KAPALI, **ortak)
    if pol == "s":
        assert abs(acik - _r_prob(1.0, 2.25, 20.0, 10.0, T2)) <= 1e-6


def test_ARAYUZ_YOKSA_SIFIR_SINYAL_TAM_KAYITTIR():
    """eps1 = eps2: iki koşum bit bit aynı, yansıyan seri TAM SIFIR. Tepesi
    sıfır olan bir seri "kesik" sayılamaz (gelecek bir sinyal yok); kapı onu
    reddetseydi fiziğin kesin cevabı (ρ = 0) yerine `None` dönerdi.
    """
    r = reflection_coefficient(eps1=2.25, eps2=2.25, theta_deg=20.0, wavelength=10.0,
                               polarization="s", window=T2)
    assert r == 0.0


def test_KUYRUK_PENCERESI_TAM_BIR_PERIYOT_ve_GENLIK_MUTLAK():
    """TRANSKRİPSİYON DENETİMİ (özel `_kuyruk`, sentetik dizi). λ = 10,
    S = 0.4 -> bir periyot 25 örnek. İşaretler pencerenin TAM sınırlarında:
    son örnek (dahil), sondan 25. örnek (dahil, son periyodun ilki), sondan
    26. örnek (HARİÇ). Negatif reel değer mutlak genlikle okunur.
    """
    def dizi(**isaret):
        a = np.zeros(200, dtype=complex)
        a[10] = 1.0  # tepe
        for konum, deger in isaret.items():
            a[{"son": -1, "ilk": -25, "dis": -26}[konum]] = deger
        return a

    k = dict(wavelength=10.0, courant=0.4)
    assert _kuyruk(dizi(son=-0.3), **k) == pytest.approx(0.3)
    assert _kuyruk(dizi(ilk=0.2), **k) == pytest.approx(0.2)
    assert _kuyruk(dizi(dis=0.4), **k) == 0.0
    assert _kuyruk(dizi(son=0.3j), **k) == pytest.approx(0.3)
    rampa = np.linspace(0.0, 1.0, 200)
    assert _kuyruk(rampa, **k) == 1.0  # hâlâ büyüyen sinyal tam sayılamaz
    assert _kuyruk(np.zeros(200), **k) == 0.0


def test_ESIK_OLCULEN_BANTTA_ve_PENCERE_BEYANINDA():
    """SÜREÇ KİLİDİ: eşiğin ARİTMETİĞİ (`test_SLOW_ESIGI_OLCULDU` deseni).

    Bant: depodaki sabitlenmiş çağrıların en büyük kuyruğu 9.44e-04, eski
    aletin hatası > 1e-4 olan sessiz sayılarının en küçük kuyruğu 2.78e-03
    (sevk penceresi). Eşik ikisinin geometrik ortası; iki yana ~1.7 kat.
    """
    assert DEFAULT_TAIL_GATE == 1.6e-3
    assert ObliqueWindow().tail_gate == DEFAULT_TAIL_GATE
    assert CIPA_EN_BUYUK_KUYRUK < DEFAULT_TAIL_GATE < SESSIZ_EN_KUCUK_KUYRUK
    assert DEFAULT_TAIL_GATE == pytest.approx(
        math.sqrt(CIPA_EN_BUYUK_KUYRUK * SESSIZ_EN_KUCUK_KUYRUK), rel=0.02
    )
    assert "kuyruk kapisi : 0.0016" in ObliqueWindow().as_lines()


#: Yankı kapısının bandı (`rc-olcum/rca/spek_analiz.log`, 2986 durum): kapısı
#: açık sabitlenmiş çağrıların en büyük spektral ikinci varışı (sevk ofseti
#: 300, 4 -> 1, 45°, λ 40: yankı kayda tepesinden girer; eski kör nokta
#: durumu HARİÇ, o bandın üstünde) ve kuyruk kapısından geçen hatası > 1e-3
#: sayıların en küçüğü (9 -> 1, 52°, λ 40, p, sevk ofseti 880). ÖLÇÜLDÜ.
IKINCI_CIPA_EN_BUYUK = 2.02e-04
IKINCI_SESSIZ_EN_KUCUK = 9.84e-04


def test_KAYIT_ICI_YANKI_ARTIK_KAPIYA_GORUNUYOR():
    """RC-a'nın KAPANIŞI (eskiden `test_DURUST_OLUMSUZ_KAYIT_ICI_YANKI_KAPIYA_
    GORUNMUYOR`). Soğurucusuz T2 (npml 0, PEC duvar), prob ofseti 350 (prob
    kaynağın 100 hücre sağında), 1 -> 2.25, θ = 0, λ = 10: kaynağın sola
    giden dalgası duvardan TAM genlikle döner ve GELEN kaydına TAMAMEN girer
    — ana darbe ile yankı 5 sg'de ayrık, yankının yansıması kayıt bitmeden
    yansıyana varmaz. Kaydın sonu SESSİZ (kuyruk gelen 1.3e-07, yansıyan
    1.2e-04; ikisi de eşiğin çok altında), yani kuyruk kapısı bu sayıyı
    GEÇİRİRDİ: kapılar kapalıyken hata 0.290 (ayrık-tam |r| = 0.235'ten
    büyük). Spektral ikinci varış 1'de kırpılır -> `None`. Aynı mekanizma
    soğurucu yankısıyla (PML, 9 -> 1, 30°, λ 40, T2 ofset 400: 6.20e-03)
    slow testte.
    """
    w = dataclasses.replace(T2, npml=0, probe_offset=350)
    acik, kapali = _iki(1.0, 2.25, 0.0, 10.0, w)
    assert acik is None
    assert kapali is not None
    assert abs(kapali - _r_prob(1.0, 2.25, 0.0, 10.0, w)) >= 0.1


def test_ESKI_KOR_NOKTA_DURUMU_ARTIK_NONE():
    """RC-a'nın KİLİTLİ DURUMU TERSİNE DÖNDÜ (eskiden `test_DURUST_OLUMSUZ_
    KAYIT_ICI_YANKI_KAPIYA_GORUNMUYOR`, "kapı yankıyı görmeyi öğrenince
    kırmızıya döner"). T2, prob ofseti 400, 4 -> 1, θ = 0, λ = 20: sol PML
    yankısı GELEN kaydına TAMAMEN girer, iki kuyruk 5.8e-08 / 1.7e-04 —
    kuyruk kapısına görünmez. Spektral ikinci varış 1.02e-03 > `echo_gate`
    -> `None`. Kapılar kapalıyken hata 3.56e-04 = |r| (0.351) × c: yankının
    paydaya eklenen payı. r0 1e-12'de hata 1.86e-06 (yankı soğurucunun;
    `rc-olcum/i_yanki_mekanizma.log`).
    """
    w = dataclasses.replace(T2, probe_offset=400)
    acik, kapali = _iki(4.0, 1.0, 0.0, 20.0, w)
    assert acik is None
    assert kapali is not None
    assert abs(kapali - _r_prob(4.0, 1.0, 0.0, 20.0, w)) >= 1e-4


def test_DURUST_OLUMSUZ_ESIGIN_ALTINDAKI_YANKI_GECER_ve_HATASI_ESIKLE_SINIRLI():
    """KAPI YANKIYI GÖRÜR, AMA EŞİĞİN ALTINDAYSA GEÇİRİR — kilitli dürüst
    olumsuz. T2, prob ofseti 410, 4 -> 1, θ = 40° (TİY, |r| = 1), λ = 20:
    sol PML yankısı GELEN kaydına TAMAMEN girer; spektral ikinci varış
    2.25e-04 (< `echo_gate` = 4.5e-04), iki kuyruk 8.7e-08 / 9.3e-05. Sayı
    GEÇER ve hatası 2.23e-04 ≈ |r| × c — kapının gördüğü ölçütle sınırlı
    (`rc-olcum/rca/toplam_sinir.log`: geçen 769 sayıda hata <= |r| c/(1 - c)
    + 0.17 kuyruk). GÖRDÜĞÜNÜN kanıtı kamu yolundan: `echo_gate` 1.8e-4'e
    (c'nin hemen altı) inince aynı çağrı `None`.
    """
    w = dataclasses.replace(T2, probe_offset=410)
    ortak = dict(eps1=4.0, eps2=1.0, theta_deg=40.0, wavelength=20.0, polarization="s")
    assert total_internal_reflection(4.0, 1.0, 40.0)
    r = reflection_coefficient(window=w, **ortak)
    assert r is not None
    h = abs(r - _r_prob(4.0, 1.0, 40.0, 20.0, w))
    assert 1e-4 <= h <= DEFAULT_ECHO_GATE
    assert reflection_coefficient(window=dataclasses.replace(w, echo_gate=1.8e-4), **ortak) is None


def test_IKINCI_VARIS_SPEKTRAL_ve_PENCERESI_ANA_DARBENIN_SONUNDAN_BASLAR():
    """TRANSKRİPSİYON DENETİMİ (özel `_ikinci_varis`, sentetik dizi).

    c = |F(g[i:])| / |F(g[:i])|, F tam f = 1/λ'da DTFT. Sentetik dizi
    g_n = b_n exp(+2πi f S n) seçilince F'nin terimleri b_n'nin KENDİSİ
    olur: ana parça b[:i] = 1 (ağırlık i), işaretler kalan parçada. Ana
    darbe t0 + |prob - kaynak| / vx'te varır ve 5 sg sonra biter; pencere o
    örnekten (DAHİL) kaydın sonuna kadardır. Geometri: kaynak 200, prob 300,
    sgf 1, λ 10 (sg 10, t0 50), S 0.4. ε1 = 1: t_son = 200 -> örnek 500;
    ε1 = 4 (vx yarıya iner): t_son = 300 -> örnek 750. Prob kaynağın
    SOLUNDA da mesafe mutlak değerle okunur. Faz uyumsuz bir kalan
    (f'den farklı frekans) aynı genlikte olsa bile ağırlık TAŞIMAZ.
    """
    w = ObliqueWindow(cells=1000, npml=0, courant=0.4, interface=400,
                      source_index=200, probe_offset=100, sgf=1.0)
    assert w.probe == 300
    n = 1200
    ton = np.exp(2j * np.pi * (0.4 / 10.0) * np.arange(n))

    def dizi(i, **isaret):
        b = np.zeros(n, dtype=complex)
        b[:i] = 1.0  # ana darbe: ağırlık i
        for k_, deger in isaret.items():
            b[int(k_[1:])] += deger
        return b * ton

    k = dict(theta_deg=0.0, wavelength=10.0, eps1=1.0, window=w)
    assert _ikinci_varis(dizi(500, i500=-5.0), **k) == pytest.approx(5.0 / 500)  # ilk örnek dahil
    assert _ikinci_varis(dizi(500, i1199=2.5j), **k) == pytest.approx(2.5 / 500)  # son örnek
    assert _ikinci_varis(dizi(499, i499=5.0), **k) == 0.0  # 499 ana parçada
    k4 = dict(k, eps1=4.0)
    assert _ikinci_varis(dizi(750, i750=3.0), **k4) == pytest.approx(3.0 / 750)
    assert _ikinci_varis(dizi(749, i749=3.0), **k4) == 0.0
    solda = dataclasses.replace(w, interface=500, probe_offset=200, source_index=400)
    assert solda.probe == 300 and solda.source_index > solda.probe
    assert _ikinci_varis(dizi(500, i500=5.0), **dict(k, window=solda)) == pytest.approx(0.01)
    # FFT penceresi de uygulanır (`Fi` ile AYNI ağırlık): hann'da ana parçanın
    # ağırlığı sum(hann[:500]), kalan örneğinki 5 hann[500]. (RC-d'den beri
    # kamu yolundan yalnız dikdörtgen gelir; burada sınanan ağırlık kuralıdır.)
    hann = np.hanning(n)
    assert _ikinci_varis(dizi(500, i500=5.0), **dict(k, window=dataclasses.replace(
        w, window_fn="hann"))) == pytest.approx(5.0 * hann[500] / hann[:500].sum())
    # faz uyumsuz kalan: tepe genliği ana darbeninkine eşit ama frekansı 0.2
    # devir/örnek (ton 0.04): 500 örnekte tam devir sayısı -> F_f = 0
    b = np.zeros(n, dtype=complex)
    b[:500] = 1.0
    uyumsuz = b * ton
    uyumsuz[600:1100] += np.cos(2 * np.pi * 0.2 * np.arange(500))
    assert _ikinci_varis(uyumsuz, **k) < 1e-12
    assert _ikinci_varis(dizi(500, i600=1e6), **k) == 1.0  # 1'de kırpılır
    assert _ikinci_varis(np.zeros(n), **k) == 0.0  # kayıt sıfır
    assert _ikinci_varis(dizi(500)[:500], **k) == 0.0  # pencere kaydın dışında
    sifir_ana = np.zeros(n, dtype=complex)
    sifir_ana[700] = 1.0
    assert _ikinci_varis(sifir_ana, **k) == 1.0  # ana darbenin ağırlığı sıfır


def test_SAG_SOGURUCU_YANKISI_KAYIT_UZUNLUGUNU_SINIRLAR():
    """TRANSKRİPSİYON DENETİMİ + `None` disiplini (`_record_steps`, aritmetik).

    Sağ soğurucunun iç yüzünden (cells - npml) iki yankı döner: arka
    koşumda sağa giden dalganınki (ortam 1, vx) ve ön koşumda GEÇEN
    dalganınki (ortam 2, vx2 = sqrt(ε2 - ε1 sin²θ)/ε2). T2, 4 -> 1, θ = 0,
    λ 40: T = 3320, vx = 0.5, vx2 = 1.
      * arka yankı: hi = 3316 (cells 1391, None) / 3324 (1393, 8300 adım);
      * geçen dalga, ε2 = 1: hi = 3318 (cells 1993, None) / 3322 (1995);
        ε2 verilmezse ya da AYNADA bu yankı yok; ε2 = 9'da (vx2 = 1/3)
        bağlamaz.
    Fizik (sevk geometrisi, `rc-olcum/dogrulama/v3_FH.log`): cells 1600,
    2.25 -> 9, 30°, λ 40'ta bu sınır yokken geçen sayının hatası 8.21e-03
    (p) / 7.18e-03 (s); geçen dalganın yankısı yansıyanla ZAMANDA örtüşür,
    kayıttan ayrılamaz — o yüzden ölçüm değil geometri. Bedeli 10 yanlış
    alarm / 1357 durum (`rc-olcum/rca/birlesik.log`), sabitlenmiş hiçbiri.
    """
    def adim(cells, **kw):
        return _record_steps(theta_deg=0.0, wavelength=40.0, eps1=4.0,
                             window=dataclasses.replace(T2, cells=cells), **kw)

    assert adim(1391) is None
    assert adim(1393) == 8300
    assert adim(1993, eps2=1.0) is None
    assert adim(1995, eps2=1.0) == 8300
    assert adim(1993) == 8300
    assert adim(1993, eps2=1.0, mirror=True) == 8300
    assert adim(1993, eps2=9.0) == 8300
    assert reflection_coefficient(eps1=4.0, eps2=1.0, theta_deg=0.0, wavelength=40.0,
                                  polarization="s",
                                  window=dataclasses.replace(T2, cells=1993)) is None
    sevk1600 = ObliqueWindow(cells=1600, npml=60, courant=0.4, interface=1500,
                             source_index=600, probe_offset=6, sgf=9.0,
                             window_fn="dikdortgen", ey_kural="aritmetik",
                             unity_gate=1e-2, pml_m=3.0, tail_gate=1.6e-3)
    for pol in ("s", "p"):
        assert reflection_coefficient(eps1=2.25, eps2=9.0, theta_deg=30.0, wavelength=40.0,
                                      polarization=pol, window=sevk1600) is None
    # AYNADA ε2 KULLANILMAZ (`reflection_coefficient` dokstringi): geçen dalga
    # yok, yani kısa sağ pencerede de ε2 sınır koymaz. T2, cells 1393, λ 20:
    # aynasız ε2 = 1 kayıt uzunluğunu kapatır (hi 2118 < T 2120); aynada
    # ε2 = 1 ve 9 BİT BİT aynı sayı.
    kisa = dataclasses.replace(T2, cells=1393)
    ayna = dict(eps1=4.0, theta_deg=0.0, wavelength=20.0, polarization="s", window=kisa)
    assert reflection_coefficient(eps2=1.0, **ayna) is None
    r1 = reflection_coefficient(eps2=1.0, mirror=True, **ayna)
    assert r1 is not None and abs(abs(r1) - 1.0) < 1e-5
    assert reflection_coefficient(eps2=9.0, mirror=True, **ayna) == r1


def test_YANKI_ESIGI_OLCULEN_BANTTA_ve_PENCERE_BEYANINDA():
    """SÜREÇ KİLİDİ: `echo_gate`in ARİTMETİĞİ (`test_ESIK_OLCULEN_BANTTA...`
    deseni). Bant `rc-olcum/rca/spek_analiz.log`dan (2986 durum): kapısı
    açık sabitlenmiş çağrıların en büyük spektral ikinci varışı 2.02e-04,
    kuyruk kapısından geçen hatası > 1e-3 sayıların en küçüğü 9.84e-04; eşik
    ikisinin geometrik ortası, iki yana ~2.2 kat. Ölçülen bedel
    (`spek_esik_4p5.log`): 3 yanlış alarm, sevk ızgarasında kayıp yok.
    Eşik kuyruk kapısınınkiyle AYNI DEĞİL ve bu ölçüldü: aynı 2986 durumda
    kuyruk eşiğinde (1.6e-3) geçen en kötü hata 1.64e-03, bu eşikte
    3.21e-04 (ikinci doğrulayıcı, yankı ve kuyruğun toplandığı dar bir
    rejimde 6.16e-04 buldu; dürüst olumsuz 2).
    """
    assert DEFAULT_ECHO_GATE == 4.5e-4
    assert ObliqueWindow().echo_gate == DEFAULT_ECHO_GATE
    assert IKINCI_CIPA_EN_BUYUK < DEFAULT_ECHO_GATE < IKINCI_SESSIZ_EN_KUCUK
    assert DEFAULT_ECHO_GATE == pytest.approx(
        math.sqrt(IKINCI_CIPA_EN_BUYUK * IKINCI_SESSIZ_EN_KUCUK), rel=0.02
    )
    assert "yanki kapisi  : 0.00045" in ObliqueWindow().as_lines()


# ---------------------------------------------------------------------------
# Sevk penceresi — G1'in başlıkları (pahalı).
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_SEVK_PENCERESINDE_G1_BASLIKLARI_ARTIK_NONE():
    """Sevk penceresi (3600 hücre, sgf 9). Kapı kapalı hatalar
    (`rc-olcum/f_beyan.log`): 1 -> 2.25, 75°, λ 20: 2.01e-02 (kuyruk
    0.758); 2.25 -> 1, 41.80°: 5.47e-02 (2.68e-02); 2.25 -> 1, 60°:
    2.96e-04 (4.63e-03) — sonuncusu `|ρ|=1` kapısından GEÇERDİ (TİY,
    ||ρ|-1| = 2.0e-04 < 1e-2). MALİYET: ~6 s (6 çağrı) -> `slow`.
    """
    sevk = ObliqueWindow()
    for e1, e2, th, esik in ((1.0, 2.25, 75.0, 1e-2), (2.25, 1.0, 41.80, 1e-2),
                             (2.25, 1.0, 60.0, 1e-4)):
        acik, kapali = _iki(e1, e2, th, 20.0, sevk)
        assert acik is None, (e1, e2, th)
        assert abs(kapali - _r_prob(e1, e2, th, 20.0, sevk)) >= esik, (e1, e2, th)


#: RC-d reddinin mesaj parçası (`reflection_coefficient`).
RET = "yalniz window_fn 'dikdortgen'"


@pytest.fixture
def nobetci(monkeypatch):
    """`_kosum` çağrılırsa düşer: pencere reddi KOŞUMDAN ÖNCE olmalı."""
    def kosma(**_):
        raise AssertionError("pencere reddi koşumdan ÖNCE olmalı")

    monkeypatch.setattr(em_obs, "_kosum", kosma)


@pytest.mark.parametrize("ad", ["hann", "blackman", "kaiser", "Dikdortgen", " dikdortgen"])
@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("ayna", [False, True])
def test_DIKDORTGEN_DISI_PENCERE_HER_KIPTE_KOSUMDAN_ONCE_REDDEDILIR(ad, pol, ayna, nobetci):
    """RC-d KAPANDI — eski dürüst olumsuz TERSİNE ÇEVRİLDİ. Eskiden T2,
    4 -> 2.25, θ = 40°, λ = 20'de hann 2.56e-02, blackman 3.56e-02 hatalı
    sayıyı üç kapıdan da geçiriyordu (`rc-olcum/rca/dogrulama/hann.log`).
    Ön ölçüm (`rc-olcum/rcd/ozet_rcd.log`, 2803 satır, üç pencere aynı
    koşum çiftinden): kapılardan geçen 408 hann sayısının 406'sının hatası
    > 1e-3; hann ile dikdörtgenin ikisinin de sayı verdiği 376 durumun
    (blackman'da aynı küme) hiçbirinde pencere dikdörtgenden iyi değil.
    Karar `InvalidInputError`, `None` değil: hata kayda değil PENCEREYE
    bağlı. Ret her kipte (s/p, ayna) ve tam ad eşitliğiyle (büyük harf,
    boşluk normalleştirilmez); `_kosum` çağrılırsa test düşer. Bağımsız
    doğrulayıcının altı mutantı (`rc-olcum/rcd/dogrulama/mut_oneri.log`):
    yalnız s'de ret — T2, hann, p'de 9.2e-03'lük hata sızıyordu — aynada
    ret yok, kayıt yokken `None`, ofsetle / merdivende sessiz ikame,
    büyük-küçük harf normalleştirmesi.
    """
    w = dataclasses.replace(T2, window_fn=ad)
    with pytest.raises(InvalidInputError, match=RET):
        reflection_coefficient(eps1=4.0, eps2=2.25, theta_deg=40.0, wavelength=20.0,
                               polarization=pol, window=w, mirror=ayna)


def test_RET_KAYIT_KAPISINDAN_ve_FAZ_YOLLARINDAN_ONCE(nobetci):
    """Ret, güvenli kayıt uzunluğu YOKKEN de `None` değil hatadır (sevk
    penceresi, ε1 = 1, θ = 0, λ = 40: `_record_steps` None); fazı okuyan
    `tir_phase_difference` `probe_offset`le ve aynada, `tir_phase_ladder`
    ilk basamakta aynı hatayı yükseltir. Son satır nöbetçinin kendisini
    sınar: dikdörtgende çağrı gerçekten `_kosum`a ulaşıyor (yoksa "önce"
    sınanmamış olurdu).
    """
    sevk = dataclasses.replace(ObliqueWindow(), window_fn="hann")
    assert _record_steps(theta_deg=0.0, wavelength=40.0, eps1=1.0, window=sevk,
                         eps2=2.25) is None
    with pytest.raises(InvalidInputError, match=RET):
        reflection_coefficient(eps1=1.0, eps2=2.25, theta_deg=0.0, wavelength=40.0,
                               polarization="s", window=sevk)
    h = dataclasses.replace(T2, window_fn="hann")
    tir = dict(eps1=4.0, eps2=1.0, theta_deg=45.0, wavelength=40.0, window=h)
    with pytest.raises(InvalidInputError, match=RET):
        tir_phase_difference(probe_offset=6, **tir)
    with pytest.raises(InvalidInputError, match=RET):
        tir_phase_difference(mirror=True, **tir)
    with pytest.raises(InvalidInputError, match=RET):
        tir_phase_ladder(eps1=4.0, eps2=1.0, theta_deg=45.0,
                         wavelengths=[16.0, 24.0, 36.0, 54.0], noise=3.09e-5, window=h)
    with pytest.raises(AssertionError, match="ÖNCE"):
        reflection_coefficient(eps1=4.0, eps2=2.25, theta_deg=40.0, wavelength=20.0,
                               polarization="s", window=T2)


def test_GEREKCE_PENCERE_IKI_DARBEYI_FARKLI_AGIRLIKLA_CARPAR():
    """RC-d'nin GEREKÇESİ, kilitli (reddin neden var olduğu). Kamu yolu
    artık pencereli sayı üretmediği için `reflection_coefficient`in iç
    koşum çifti `_kosum` ile aynen kurulur; dikdörtgen oranı kamu yolunun
    sayısına BİT BİT eşittir (kurulum doğru). T2, 4 -> 2.25, θ = 40°,
    λ = 20, s:
      * ayrık-tam r'ye hata: dikdörtgen 1.4e-07, hann 2.56e-02, blackman
        3.56e-02;
      * kapılar KÖR: kuyruklar 4.8e-06 / 1.7e-05, pencereli ikinci varış
        <= 2.4e-09;
      * MEKANİZMA: hata ≈ |r| |w(t_y)/w(t_g) - 1| (w pencere; t_g, t_y gelen
        ve yansıyan kaydın enerji ağırlık merkezleri). Ön ölçümde 638
        durumda hata/tahmin medyanı 0.95 (hann, %10..%90 0.93..1.08),
        blackman'da 0.91; burada (|r| 0.34) [0.85, 1.15]. Tek terim
        |r| -> 0'da (Brewster yakını p) ya da w(t_y) ≈ w(t_g) iken eksik
        kalır: orada r(f)'nin dağıtkanlığı baskındır (638'in 65'i ±%15
        dışında; iki terimli sınır 645/646, `rcd/dogrulama/mekanizma2.log`).
    MALİYET: dört koşum (~0.6 s).
    """
    e1, e2, th, lam = 4.0, 2.25, 40.0, 20.0
    n = _record_steps(theta_deg=th, wavelength=lam, eps1=e1, window=T2, eps2=e2)
    ky = 2.0 * math.pi / lam * math.sqrt(e1) * math.sin(math.radians(th))
    ortak = dict(steps=n, wavelength=lam, ky=ky, polarization="s", window=T2)
    on = _kosum(medium=Medium.half_space(T2.cells, interface=T2.interface, eps1=e1,
                                         eps2=e2, ey_kural=T2.ey_kural),
                boundary=PML(thickness=T2.npml, m=T2.pml_m), **ortak)
    arka = _kosum(medium=Medium(np.full(T2.cells, e1), np.ones(T2.cells), T2.ey_kural),
                  boundary=PML(thickness=T2.npml, m=T2.pml_m), **ortak)
    g = arka.probes[0]
    y = on.probes[0] - g

    def oran(pen):
        Fi, _ = _spektrum(g * pen, T2.courant, 1.0 / lam)
        Fr, _ = _spektrum(y * pen, T2.courant, 1.0 / lam)
        return Fr / Fi

    r = _r_prob(e1, e2, th, lam, T2)
    dik = oran(_FFT_PENCERELERI["dikdortgen"](n))
    assert dik == reflection_coefficient(eps1=e1, eps2=e2, theta_deg=th, wavelength=lam,
                                         polarization="s", window=T2)
    assert abs(dik - r) < 1e-6
    assert max(_kuyruk(g, wavelength=lam, courant=T2.courant),
               _kuyruk(y, wavelength=lam, courant=T2.courant)) < 1e-4
    k = np.arange(n)

    def merkez(s):
        e = np.abs(s) ** 2
        return float((k * e).sum() / e.sum())

    for ad, beklenen in (("hann", 2.56e-2), ("blackman", 3.56e-2)):
        pen = _FFT_PENCERELERI[ad](n)
        hata = abs(oran(pen) - r)
        assert hata == pytest.approx(beklenen, rel=0.01), ad
        assert _ikinci_varis(g, theta_deg=th, wavelength=lam, eps1=e1,
                             window=dataclasses.replace(T2, window_fn=ad)) < 1e-6
        wg, wy = np.interp([merkez(g), merkez(y)], k, pen)
        assert 0.85 < hata / (abs(r) * abs(wy / wg - 1.0)) < 1.15, ad


@pytest.mark.slow
def test_SOGURUCU_YANKISI_KAYIT_ICINDE_ARTIK_NONE():
    """RC-a'nın belgelenen durumu (doğrulayıcı, `rc-olcum/dogrulama/v3_FH.log`):
    T2, prob ofseti 400, 9 -> 1, θ = 30° (TİY), λ = 40. Sol PML yankısı GELEN
    kaydına TAMAMEN girer; kuyruk 2.2e-05 (yansıyan), spektral ikinci varış
    6.20e-03. Kapılar kapalıyken hata 6.20e-03 (= |r| × c, |r| = 1); yankı
    kapısıyla `None`. `|ρ| = 1` kapısı (1e-2) bu sayıyı GEÇİRİYORDU. MALİYET: iki çağrı
    ~1.4 s -> `slow` (hızlı şeritteki eşdeğeri PEC duvarlı
    `test_KAYIT_ICI_YANKI_ARTIK_KAPIYA_GORUNUYOR`).
    """
    w = dataclasses.replace(T2, probe_offset=400)
    assert total_internal_reflection(9.0, 1.0, 30.0)
    acik, kapali = _iki(9.0, 1.0, 30.0, 40.0, w)
    assert acik is None
    assert kapali is not None
    assert abs(abs(kapali) - 1.0) < 1e-2  # birim kapısı geçirirdi
    assert abs(kapali - _r_prob(9.0, 1.0, 30.0, 40.0, w)) >= 5e-3
