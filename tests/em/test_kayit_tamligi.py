"""KAYIT TAMLIĞI — `reflection_coefficient`in kuyruk kapısı (10.10.2026).

NE KİLİTLENİYOR: G1 ön ölçümü (09.10.2026) ölçüm aletinin kısmi yansımada
SESSİZ bir protokol hatası taşıdığını buldu (1 -> 2.25, λ = 20, θ = 75°:
|ρ - r| = 2.01e-02, G1 bedelinden büyük). Kaynak ayrıldı
(`.superpowers/sdd/rc-olcum/`): G1'in iki durumunda FFT bini ve PML
suçsuz; kayıt bittiğinde darbe hâlâ geliyor. Sebepler sabit-`ky` indirgemesinin grup hızı dağılımı
(sıyırma), dal noktasının uzun kuyruğu (kritik açı), kaba ızgarada şemanın
yavaş grup hızı (yoğun ortam, θ = 0'da bile) ve kaynağın sola giden
dalgasının soğurucudan dönüşüdür. Kapı: kaydın SON PERİYODUNDAKİ genlik
(gelen ya da yansıyan) tepenin `tail_gate` katını aşarsa `None`.

PENCERE (T2): G1'in `T2_PENCERE`si — 1800 hücre, npml 40, S 0.4, iface 750,
kaynak 300, prob ofseti 6, sgf 6, dikdörtgen, aritmetik, |ρ|=1 kapısı 1e-2,
m = 3 — ve kapı alanı AÇIKÇA yazılır (G7 dersi: varsayılan değişirse test
sessizce başka pencereyi ölçmesin). Sevk penceresinin pahalı başlıkları
tek bir `slow` testte.

REFERANS: AYRIK-TAM r_s (kapalı form, testte elle; G1'in `_r_ayrik`i),
prob düzlemine taşınmış ve sevk uzlaşımında (eşlenik). Kapalı form yalnız
s'tedir; p'nin ayrık-tam referansı ön ölçümde transfer özyinelemesiyle
kuruldu (temiz açılarda alete 5e-9 .. 1e-7'de oturur) ve burada
kullanılmaz.

ÖLÇÜLDÜ (T2, kapı kapalı = `tail_gate` 1; `rc-olcum/f_beyan.log`,
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
DÜRÜST OLUMSUZ 2 — KAYIT İÇİ YANKI KAPIYA GÖRÜNMEZ (doğrulayıcı buldu,
`rc-olcum/dogrulama/`): prob arayüzden uzaksa soğurucunun bir yankısı
GELEN kaydına TAMAMEN girer, yansıması ise kayıt bitmeden YANSIYANA varmaz;
kaydın sonu sessizdir ve kapı sayıyı geçirir. Ölçülen: 9 -> 1, 30°, λ 40,
T2 ofset 400'de hata 6.20e-03 (kuyruk 2.15e-05, oran 288); r0 1e-6 -> 1e-12
ile 8.1e-05'e iner (`i_yanki_mekanizma.log`). Yani "<= 0.54 τ" okuması
YALNIZ prob arayüze yakınken (sevk ofseti 6) geçerlidir. Bir test bunu
kilitler; kapı düzeltilince kırmızıya döner.

SINIF: kapalı-form referanslı ÇEVİRİ DENETİMİ + `None` disiplini (Modül 5
sözlüğü, `test_arayuz.py` başlığı). Özel import tek bir TRANSKRİPSİYON
testinde: `_kuyruk`un penceresi (tam bir periyot) sentetik diziyle sınanır,
çünkü pencere uzunluğu kamu yolunda eşiğin içinde kaybolur (doğrulayıcının
yarım/iki periyot mutantları kamu yolu testlerinin hepsini geçti).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qlab.observable.em_obs import (
    DEFAULT_TAIL_GATE,
    ObliqueWindow,
    _kuyruk,
    reflection_coefficient,
    total_internal_reflection,
)

#: G1'in T2 penceresi + kuyruk kapısı, HER ALAN açık.
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
)
#: Kapı KAPALI: kuyruk oranı tanım gereği <= 1, yani 1.0 her kaydı kabul
#: eder ve alet kapıdan önceki hâline BİT BİT döner.
KAPALI = dataclasses.replace(T2, tail_gate=1.0)

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
    kapali = reflection_coefficient(window=dataclasses.replace(w, tail_gate=1.0), **ortak)
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


def test_DURUST_OLUMSUZ_KAYIT_ICI_YANKI_KAPIYA_GORUNMUYOR():
    """KAPININ KÖR NOKTASI, KİLİTLİ (açık madde). T2, prob ofseti 400 (prob
    kaynağın 50 hücre sağında, arayüzden 400 hücre uzakta), 4 -> 1, θ = 0,
    λ = 20: kaynağın sola giden dalgası sol soğurucudan döner, GELEN kaydına
    TAMAMEN girer (ana darbeden sonra tepenin 1.02e-03'ü), yansıması kayıt
    bitmeden yansıyana varmaz. Kaydın sonu sessiz (gelen 5.8e-08, yansıyan
    1.7e-04) ve kapı sayıyı GEÇİRİR: hata 3.56e-04. r0 1e-12'de hata
    1.86e-06 (`rc-olcum/i_yanki_mekanizma.log`). Kapı bu yankıyı görmeyi
    öğrenince bu test KIRMIZIYA döner — o zaman tersine çevrilmeli.
    """
    w = dataclasses.replace(T2, probe_offset=400)
    r = reflection_coefficient(eps1=4.0, eps2=1.0, theta_deg=0.0, wavelength=20.0,
                               polarization="s", window=w)
    assert r is not None
    assert abs(r - _r_prob(4.0, 1.0, 0.0, 20.0, w)) >= 1e-4


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
