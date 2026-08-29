"""Işık bükülmesi `1 + γ` — Modül 6 Task 4, Step 1.

PENCERE — bu dosyadaki HER sayı aşağıdaki tek pencerede, **sevk edilen kodla**
ölçüldü; hiçbiri plandan ya da spec'ten devralınmadı (*pencere adlandırılmazsa
hüküm dışına taşınır*, bu depoda on birden fazla kez çıktı):

- Birim seçimi `G = c = 1`; `M = 1.0` (aksi yazılmadıkça).
- `b` **çarpışma parametresidir** (`b = L/E`), en yakın geçiş yarıçapı değil.
  Ayrım hükme girer: aynı ikinci mertebe katsayısı `r0` cinsinden
  `15π/4 − 4`, `b` cinsinden **`15π/4`**'tür.
- `θ` ikamesi `u = u0·sin θ`, `θ ∈ [0, π/2]`; `scipy.integrate.quad`
  `epsabs=1e-14`, `epsrel=1e-13`, `limit=400` — `geodesic.py`'nin sevk edilen
  düğmeleri, burada YENİDEN yazılmıyor.
- Bükülme oranı `1/b`'de **Richardson** ile dışdeğerlenir; sevk edilen çift
  **(1e5, 3e5)**. Tek `b` penceresi bir çıpa DEĞİLDİR (aşağıda ölçüldü).

BU DOSYANIN TAŞIYICI İDDİASI: *bükülme oranı `1 + γ` veriyor.* Ve o iddianın
**iki yarısı bir çapraz kontrol DEĞİLDİR** — bu dosya bunu ölçerek söyler:

  (1) İki yarı `b`, `M`, `β`, ayar, `quad` düğmeleri, `θ` ikamesi, dönüş
      noktası arayıcısı, deflasyon ve Richardson çiftini **PAYLAŞIR**; ayrılan
      tek şey `γ`'nın değeridir.
  (2) Paylaşılan gövdeye sokulan üç bozulma ölçüldü; ikisi oranda GÖRÜNÜYOR,
      biri (γ'dan bağımsız, çarpımsal olan) **SADELEŞİYOR** ve temiz artığın
      bile altında kalıyor.
  (3) O kör noktayı gören şey oran değil `delta`'nın KENDİSİDİR
      (`delta·b/M → 4` ve ikinci mertebe `15π/4`), ve bu dosya ikisini de
      kilitler.

`slow` İŞARETİ YOK, ölçülerek: 33 test **0.59 s**'de koşuyor ve `--durations`
tek bir gövdeyi bile gösteremiyor (hepsi 5 ms'nin altında). Bu dosyanın `slow`
eşiği `test_kapali_form.py`'de ölçüldü ve **1.0 s**'dir.
"""

from __future__ import annotations

import math
import warnings

import pytest
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

import qlab.engine.grav.geodesic as geodesic_modulu
from qlab.engine.grav.geodesic import (
    DEFAULT_NULL_RICHARDSON,
    DEFLECTION_PROBE_EPSILONS,
    GAUGES,
    deflection_angle,
)
from qlab.engine.grav.metric import (
    GAUGE_NAMES,
    Theory,
    deflection_closed_form,
    evaluate_poly,
    isotropic_potential,
    null_kernel,
    null_potentials,
    null_w,
)
from qlab.errors import InvalidInputError
from qlab.observable.grav_obs import deflection_ratio, newton_noise_floor

GR = Theory()
NEWTON_YARISI = Theory(gamma=0.0)

# PPN taraması — `1 + gamma` ÇARPANLARI ELLE YAZILI DEĞİL, `1 + gamma`
# doğrudan okunuyor; burada elle yazılan şey `gamma`nın kendisidir.
PPN_GAMMA = (-0.5, 0.5, 1.0, 1.5, 2.0)
PPN_BETA = (0.0, 1.0, 3.0)


# ---------------------------------------------------------------------------
# Bağımsız ikinci uygulama — bu dosyanın kendi karelemesi.
# Task 3'ün deseni: bir iddiayı sevk edilen gövdenin ÜZERİNDEN değil, onun
# YANINDA duran ikinci bir gövdeyle ölçmek. Aşağıdaki üç `*_kat` argümanı
# PAYLAŞILAN gövdeye sokulan bozulmalardır ve sevk edilen kodda karşılıkları
# YOKTUR (sevk edilen koda test için düğme eklenmez).
# ---------------------------------------------------------------------------
def _bagimsiz_delta(*, theory, M, b, gauge="schwarzschild",
                    u3_kat=1.0, theta_kat=1.0, u0_kat=1.0):
    pot = null_potentials(theory=theory, M=M, gauge=gauge)
    a0, a1, a2 = pot.time_poly
    s0, s1, _ = pot.source_poly
    N = (s0 / b**2, s1 / b**2, -a0, -a1 * u3_kat, -a2)
    f = lambda u: evaluate_poly(N, u)
    ust = 1.05 / b
    while f(ust) > 0.0:
        ust *= 1.05
    u0 = brentq(f, 0.0, ust, xtol=1e-18, rtol=8.9e-16, maxiter=200) * u0_kat
    q3 = -N[4]
    q2 = u0 * q3 - N[3]
    q1 = u0 * q2 - N[2]
    q0 = u0 * q1 - N[1]

    def integrand(t):
        s = math.sin(theta_kat * t)
        u = u0 * s
        D = evaluate_poly(pot.time_poly, u) / evaluate_poly(pot.damp_poly, u)
        return math.sqrt(u0 * (1.0 + s) * D / evaluate_poly((q0, q1, q2, q3), u))

    v, _ = quad(integrand, 0.0, math.pi / 2, epsabs=1e-14, epsrel=1e-13,
                limit=400)
    return 2.0 * v - math.pi


def _bagimsiz_oran(*, richardson=(1e5, 3e5), **kw):
    def tek(b):
        return (_bagimsiz_delta(theory=GR, M=1.0, b=b, **kw)
                / _bagimsiz_delta(theory=NEWTON_YARISI, M=1.0, b=b, **kw))
    alt, ust = richardson
    f1, f2 = tek(alt), tek(ust)
    h1, h2 = 1.0 / alt, 1.0 / ust
    return (f1 * h2 - f2 * h1) / (h2 - h1)


# ---------------------------------------------------------------------------
# Step 1 — ÇIPA
# ---------------------------------------------------------------------------
def test_BUKULME_ORANI_BIR_ARTI_GAMMA_ve_HER_IKI_YARI_SAYISAL():
    """SINIF: FİZİK ÇIPASI — statik kökü YOK, boş jeodeziği çözmek ZORUNLU.

    (Karşılaştır: `critical_impact_parameter`ın statik kökü VAR ve o
    "kapalı-form türev" rafındadır — `test_kapali_form.py`.)

    HER İKİ YARI DA SAYISALDIR: payda `2M/b` kapalı formu KULLANILMAZ, aynı
    boru `Theory(gamma=0)` ile koşulur.

    İNTEGRAL `u = 0`'DAN BAŞLAR, YANİ KESME YOKTUR — ve bunun neden bir ayrıntı
    değil bir gerekçe olduğu ölçüldü: sonlu bir `r_start`tan başlayan bir
    sürümde kaçırılan asimptot açısı `2b/r_start`tır ve b = 1e5, r_start = 1e6
    için 0.2 rad eder, yani sinyalin (4e-5 rad) 5000 KATI. O sürüm
    "oran = 1.000000" verir — GR'yi NEWTON SANIR.

    ÖLÇÜLEN (M = 1, Richardson (1e5, 3e5)):

        ayar             oran                     dürüst hane
        schwarzschild    1.9999999996887523       9.81
        isotropic        1.999999999804698       10.01

    İki ayarın farkı 1.1594569748751837e-10.
    """
    schw = deflection_ratio(theory=GR, M=1.0)
    izo = deflection_ratio(theory=GR, M=1.0, gauge="isotropic")
    assert schw == 1.9999999996887523
    assert izo == 1.999999999804698
    assert abs(schw - izo) == 1.1594569748751837e-10

    for deger, beklenen_hane in ((schw, 9.81), (izo, 10.01)):
        hane = -math.log10(abs(deger - 2.0) / 2.0)
        assert hane == pytest.approx(beklenen_hane, abs=0.01)
    # Kapalı form TEK YERDE yazılı: `metric.deflection_closed_form`.
    assert deflection_closed_form(GR) == 2.0


@pytest.mark.parametrize("gauge, en_kotu", [("schwarzschild", 4.3487e-09),
                                            ("isotropic", 2.5253e-09)])
def test_BIR_ARTI_GAMMA_PPN_AILESININ_TAMAMINDA(gauge, en_kotu):
    """SINIF: FİZİK ÇIPASI. `1 + γ` on beş PPN noktasında ölçüldü.

    `β` bu mertebede GÖRÜNMÜYOR: aynı `γ` için üç `β` değeri de aynı sayıyı
    veriyor. `β`'nın göründüğü yer ikinci mertebedir ve o da kapalı formdur
    (`test_BETA_TUREVI_EKSI_PI_M_KARE_BOLU_B_KARE`).
    """
    olculen = 0.0
    for gamma in PPN_GAMMA:
        for beta in PPN_BETA:
            t = Theory(gamma=gamma, beta=beta)
            oran = deflection_ratio(theory=t, M=1.0, gauge=gauge)
            olculen = max(olculen, abs(oran - deflection_closed_form(t)))
    assert olculen == pytest.approx(en_kotu, rel=1e-3, abs=0.0)
    assert olculen < 1e-8


def test_TEK_B_PENCERESI_BIR_CIPA_DEGILDIR_KESME_SINIRLI():
    """SINIF: PENCERE ÖLÇÜMÜ — ve `richardson`ın neden zorunlu olduğu.

    Tek `b = 1e5`'te oran 2.0000274902889896'dır, yani **4.86 dürüst hane**;
    aynı boru Richardson (1e5, 3e5) ile **9.81** hane veriyor. Fark 5 hane ve
    tamamı KESME hatasıdır (`(oran−2) ~ (7π/8)·M/b`).

    Bu yüzden `deflection_ratio` `richardson=None`'ı REDDEDER: kesme sınırlı
    bir pencerede gürültü tabanı bir hata çubuğu değildir (Task 2'nin ölçümü).
    """
    b = 1e5
    tek = (deflection_angle(theory=GR, M=1.0, b=b)
           / deflection_angle(theory=NEWTON_YARISI, M=1.0, b=b))
    assert tek == 2.0000274902889896
    assert -math.log10(abs(tek - 2.0) / 2.0) == pytest.approx(4.86, abs=0.01)

    with pytest.raises(InvalidInputError, match="TEK `b`de bir CIPA DEGILDIR"):
        deflection_ratio(theory=GR, M=1.0, richardson=None)


# ---------------------------------------------------------------------------
# Step 1 — PENCERE AYAR-BAĞIMLI, ve PLANIN İKİ SAYISI ÇÜRÜDÜ
# ---------------------------------------------------------------------------
def test_PENCERE_AYAR_BAGIMLI_ve_PLANIN_IKI_SAYISI_TAM_PI_KADAR_BUYUK():
    """SINIF: PENCERE ÖLÇÜMÜ — ve bir ÇÜRÜTME.

    `(oran − 2)·b/M`'nin `b → ∞` limiti ÖLÇÜLDÜ (tek `b`, Richardson yok):

        ayar             b = 1e5     b = 1e6     kapalı form
        schwarzschild    2.749029    2.749796    `7π/8` = 2.748894
        isotropic        1.570874    1.571407    `π/2`  = 1.570796
        fark             1.178155                `3π/8` = 1.178097

    PLANIN FARKI (`3π/8`) DOĞRU, İKİ DEĞERİ DE YANLIŞ: plan `15π/8` = 5.890486
    ve `3π/2` = 4.712389 diyor, yani **her ikisi de tam `π` kadar büyük**.
    Sebep ölçülebilir: `(oran−2)·b/M = c2(γ)/2 − c2(0)` ve bu boruda Newton
    yarısının KENDİ ikinci mertebesi `c2(0) = π`'dir. Planın sayıları
    `c2(0) = 0` ile, yani Newton yarısı KAPALI FORMDAN alınmış bir boruyla
    uyuşuyor — oysa aynı plan "her iki yarı da sayısal" diyor. İki cümle aynı
    anda doğru olamaz; sevk edilen boru "her iki yarı da sayısal" olanıdır.
    """
    olculen = {}
    for gauge in GAUGES:
        for b in (1e5, 1e6):
            oran = (deflection_angle(theory=GR, M=1.0, b=b, gauge=gauge)
                    / deflection_angle(theory=NEWTON_YARISI, M=1.0, b=b,
                                       gauge=gauge))
            olculen[(gauge, b)] = (oran - 2.0) * b

    assert olculen[("schwarzschild", 1e5)] == pytest.approx(2.749029, abs=1e-6)
    assert olculen[("schwarzschild", 1e6)] == pytest.approx(2.749796, abs=1e-6)
    assert olculen[("isotropic", 1e5)] == pytest.approx(1.570874, abs=1e-6)
    assert olculen[("isotropic", 1e6)] == pytest.approx(1.571407, abs=1e-6)

    # Kapalı formlara yakınsıyorlar — ve `b = 1e5` daha iyi, çünkü `b = 1e6`'da
    # `delta`nın mutlak ulp'si `b`yle çarpılıp geri kirletiyor.
    assert olculen[("schwarzschild", 1e5)] == pytest.approx(7 * math.pi / 8,
                                                            rel=1e-4)
    assert olculen[("isotropic", 1e5)] == pytest.approx(math.pi / 2, rel=1e-4)
    fark = olculen[("schwarzschild", 1e5)] - olculen[("isotropic", 1e5)]
    assert fark == pytest.approx(3 * math.pi / 8, rel=1e-4)

    # PLANIN İKİ SAYISI: her biri tam `π` kadar büyük.
    assert 15 * math.pi / 8 - 7 * math.pi / 8 == pytest.approx(math.pi, rel=1e-15)
    assert 3 * math.pi / 2 - math.pi / 2 == pytest.approx(math.pi, rel=1e-15)
    for plan_degeri, olculen_deger in ((15 * math.pi / 8,
                                        olculen[("schwarzschild", 1e5)]),
                                       (3 * math.pi / 2,
                                        olculen[("isotropic", 1e5)])):
        assert abs(plan_degeri - olculen_deger) > 3.0


def test_SERTIFIKA_ARALIK_OLARAK_VERILIR():
    """SINIF: PENCERE ÖLÇÜMÜ. Sertifika bir SAYI değil bir ARALIKTIR.

    Tek `b`, Richardson yok, bağıl hata `|oran − 2|/2`, Schwarzschild, GR:

        b/b_krit    1.001      2          20         100        1e4
        bağıl hata  5.709e+00  2.460e-01  1.389e-02  2.671e-03  2.645e-05

    PLANIN TABLOSU (1.4e+01 / 4.5e-01 / 1.3e-02 / 2.3e-03) alt iki basamakta
    üşüşüyor (1.3e-02 ↔ 1.389e-02, 2.3e-03 ↔ 2.671e-03) ama üst iki
    basamakta ÜRETİLEMEDİ: 1.4e+01 ↔ 5.709e+00 (**2.4 kat**) ve
    4.5e-01 ↔ 2.460e-01 (**1.8 kat**). Güçlü alanda bir sayı ancak metrik
    modeliyle birlikte anlamlıdır ve planın hangi modeli kullandığı yazılı
    değildi.
    """
    from qlab.observable.grav_obs import critical_impact_parameter
    b_krit = critical_impact_parameter(theory=GR, M=1.0)
    beklenen = {1.001: 5.709e00, 2.0: 2.4597e-01, 20.0: 1.3889e-02,
                100.0: 2.6707e-03, 1e4: 2.6454e-05}
    for kat, hedef in beklenen.items():
        b = kat * b_krit
        oran = (deflection_angle(theory=GR, M=1.0, b=b)
                / deflection_angle(theory=NEWTON_YARISI, M=1.0, b=b))
        assert abs(oran - 2.0) / 2.0 == pytest.approx(hedef, rel=2e-4)
    # ARALIK: uçlar arasında beş mertebeden fazla var, tek sayı yalan olurdu.
    assert beklenen[1.001] / beklenen[1e4] > 2e5


@pytest.mark.parametrize("gauge, b, beklenen", [
    ("schwarzschild", 1e3, 1.007687),
    ("schwarzschild", 1e4, 1.000751),
    ("schwarzschild", 1e5, 1.000815),
    ("isotropic", 1e3, 1.010255),
    ("isotropic", 1e4, 1.001020),
    ("isotropic", 1e5, 1.000108),
])
def test_BETA_TUREVI_EKSI_PI_M_KARE_BOLU_B_KARE(gauge, b, beklenen):
    """SINIF: FİZİK ÇIPASI (ikinci mertebe). `∂δ/∂β = −π M²/b²`.

    Merkezi fark `h = 1e-3`. `h` BİR PENCERE DÜĞMESİDİR ve küçültmek
    İYİLEŞTİRMEZ: `h = 1e-4` ile b = 1e5'te iki ayar da 1.010710 verir, çünkü
    türevin mutlak büyüklüğü `1/b²` ile düşerken farkın mutlak yuvarlaması
    sabit kalır.
    """
    h = 1e-3
    arti = deflection_angle(theory=Theory(beta=1.0 + h), M=1.0, b=b, gauge=gauge)
    eksi = deflection_angle(theory=Theory(beta=1.0 - h), M=1.0, b=b, gauge=gauge)
    turev = (arti - eksi) / (2.0 * h)
    assert turev / (-math.pi / b**2) == pytest.approx(beklenen, abs=1e-6)


@pytest.mark.parametrize("gauge, b, beklenen", [
    ("schwarzschild", 1e5, 11.781396),
    ("schwarzschild", 1e6, 11.781358),
    ("isotropic", 1e5, 9.425050),
    ("isotropic", 1e6, 9.424576),
])
def test_IKINCI_MERTEBE_AYAR_BAGIMLI_15PI_BOLU_4_ve_3PI(gauge, b, beklenen):
    """SINIF: FİZİK ÇIPASI (Schwarzschild ayarı) + MODELLEME ARTIĞI (izotropik).

    `(δ·b/M − 4)·b/M`:
      * Schwarzschild ayarı GR'de TAM Schwarzschild metriğidir
        (`metric.NullPotentials`), ve orada literatürün sayısı `b` cinsinden
        **`15π/4` = 11.780972**'dir. (`15π/4 − 4` = 7.780972 aynı sayının
        `r0` — en yakın geçiş — cinsinden yazılmışıdır; `1/b`'den `1/r0`'a
        geçiş katsayıyı tam +4 kaydırır.)
      * İzotropik ayar 1PN'de KESİLMİŞTİR, yani oradaki **`3π` = 9.424778** bir
        MODELLEME ARTIĞIDIR, fizik değil — `second_order_coefficient`in perihel
        tarafındaki hükmünün birebir aynısı.
    """
    d = deflection_angle(theory=GR, M=1.0, b=b, gauge=gauge)
    olculen = (d * b - 4.0) * b
    assert olculen == pytest.approx(beklenen, abs=1e-6)
    hedef = 15 * math.pi / 4 if gauge == "schwarzschild" else 3 * math.pi
    assert olculen == pytest.approx(hedef, rel=5e-5)


# ---------------------------------------------------------------------------
# Step 1 — İKİ YARININ PAYLAŞTIĞI DÜĞMELER (bu dosyanın en önemli testi)
# ---------------------------------------------------------------------------
def test_IKI_YARI_HER_DUGMEYI_PAYLASIYOR_ve_BIR_BOZULMA_SADELESIYOR():
    """BU DOSYANIN EN ÖNEMLİ TESTİ: oranın KÖR NOKTASI ölçülüyor.

    Task 3'ün dersi buydu: *bir çapraz kontrolün görmediği yüzey de
    ölçülebilir.* Burada durum daha sert — bükülmenin "iki sayısal yarısı"
    bir çapraz kontrol DEĞİL, tek borunun `γ` düğmesi değiştirilmiş iki
    koşusudur. Paylaşılan gövdeye üç bozulma sokuldu (bu dosyanın kendi
    bağımsız karelemesiyle; sevk edilen koda test düğmesi eklenmez):

        bozulma                        δ(γ=1)      δ(γ=0)      ORAN kayması
        u³ katsayısı 2M → 2.001M       5.0001e-04  1.0000e-03  9.9900e-04
        θ ikamesi sin(t) → sin(1.0001t) 1.7808e-05 2.1458e-05  7.8533e-05
        dönüş noktası u0 → 1.0000001·u0 1.0002e-07 1.0001e-07  1.1102e-10

    (İlk iki sütun BAĞIL kayma, üçüncüsü MUTLAK.) Temiz artık
    |oran − 2| = 3.1124769428458876e-10.

    ÜÇÜNCÜ SATIR KÖR NOKTADIR: `γ`'dan bağımsız, çarpımsal bir bozulma iki
    yarıda AYNI bağıl kaymayı yapar, oranda sadeleşir ve kayma temiz artığın
    bile ALTINDA kalır (1.11e-10 < 3.11e-10). İlk iki bozulma `γ`'ya bağlı
    olduğu için sadeleşmez.

    Onu gören şey oran DEĞİL, `delta`nın kendisidir: aynı bozulmada `delta`
    1.0002e-07 bağıl kayıyor, yani `delta·b/M → 4` çıpası onu 5 mertebe
    farkla görüyor. Bu yüzden bu dosya ORANI DA AÇIYI DA kilitler.
    """
    temiz = _bagimsiz_oran()
    # Bağımsız kareleme sevk edilen kodla BİT BİT aynı — ölçüm meşru.
    assert _bagimsiz_delta(theory=GR, M=1.0, b=1e5) == deflection_angle(
        theory=GR, M=1.0, b=1e5)
    assert temiz == deflection_ratio(theory=GR, M=1.0)
    temiz_artik = abs(temiz - 2.0)
    assert temiz_artik == 3.1124769428458876e-10

    satirlar = {
        "u3": (dict(u3_kat=1.0005), 5.0001e-04, 1.0000e-03, 9.9900e-04),
        "theta": (dict(theta_kat=1.0001), 1.7808e-05, 2.1458e-05, 7.8533e-05),
        "u0": (dict(u0_kat=1.0000001), 1.0002e-07, 1.0001e-07, 1.1102e-10),
    }
    c1 = _bagimsiz_delta(theory=GR, M=1.0, b=1e5)
    c0 = _bagimsiz_delta(theory=NEWTON_YARISI, M=1.0, b=1e5)
    for ad, (kw, bek1, bek0, bek_oran) in satirlar.items():
        d1 = _bagimsiz_delta(theory=GR, M=1.0, b=1e5, **kw)
        d0 = _bagimsiz_delta(theory=NEWTON_YARISI, M=1.0, b=1e5, **kw)
        assert abs(d1 - c1) / c1 == pytest.approx(bek1, rel=1e-3), ad
        assert abs(d0 - c0) / c0 == pytest.approx(bek0, rel=1e-3), ad
        assert abs(_bagimsiz_oran(**kw) - temiz) == pytest.approx(
            bek_oran, rel=1e-3, abs=0.0), ad

    # HÜKÜM: ilk ikisi GÖRÜNÜR (temiz artığın en az 1e5 katı), üçüncüsü
    # GÖRÜNMEZ (temiz artığın ALTINDA).
    assert satirlar["u3"][3] / temiz_artik > 1e6
    assert satirlar["theta"][3] / temiz_artik > 1e5
    assert satirlar["u0"][3] < temiz_artik
    # ...ama AÇI onu görüyor: 1.0002e-07 bağıl kayma, δ'nın kendi ulp'sinin
    # (1.11e-16 bağıl) 9e8 katı.
    assert satirlar["u0"][1] / (math.ulp(c1) / c1) > 1e8


def test_ORAN_DA_ACI_DA_YALNIZ_M_BOLU_B_YE_BAGLI_BIT_BIT():
    """SINIF: ÖLÇÜLMÜŞ OLUMSUZ ÖRNEK — bir AYIRT EDİCİ SINAV ELENDİ.

    İlk yazdığım kör-nokta sınavı şuydu: *"oran yalnız `M/b`'ye bağlıdır,
    `delta` değildir; `M`'i ve `b`'yi birlikte iki katına çıkarmak ikisini
    ayırır."* ÖLÇÜLDÜ VE YANLIŞ. `delta` bir AÇIDIR, yani boyutsuzdur, ve
    `G = c = 1` birimlerinde tek boyutsuz birleşim `M/b`'dir; ikisi de
    BİT BİT değişmiyor.

    Yani `M/b` ölçeklemesi oranla açıyı AYIRMAZ. Ayıran tek şey `γ`'dan
    bağımsız bir ÇARPANDIR ve o da ancak çekirdek dışarıdan bozularak görülür
    (bir üstteki test).
    """
    assert (deflection_angle(theory=GR, M=1.0, b=1e5)
            == deflection_angle(theory=GR, M=2.0, b=2e5)
            == deflection_angle(theory=GR, M=0.5, b=5e4))
    assert (deflection_ratio(theory=GR, M=1.0, richardson=(1e5, 3e5))
            == deflection_ratio(theory=GR, M=2.0, richardson=(2e5, 6e5))
            == deflection_ratio(theory=GR, M=0.5, richardson=(5e4, 1.5e5)))


def test_IKI_AYARIN_KESISIMI_BOS_DEGIL_TASK_3_TEN_FARKLI():
    """SINIF: ÇIPA DEĞİL — bir MİMARİ ÖLÇÜMÜ, ve Task 3'ten AYRIM.

    Perihelde Schwarzschild yolu `Theory`den yalnız `lam`i, izotropik yol
    yalnız `{gamma, beta}`yı okuyordu; kesişim BOŞTU ve o yüzden iki ayar
    GERÇEK bir çapraz kontroldü. Bükülmede öyle DEĞİL: `null_potentials`
    ikisinde de hem `gamma`yı hem `beta`yı okuyor, kesişim `{gamma, beta}`.

    Dolayısıyla `1 + γ`'nın iki ayarda aynı çıkması bir ÇAPRAZ KONTROL değil,
    bir AYAR-BAĞIMSIZLIĞI ölçümüdür. Bu ayrım kaydedilmezse Task 3'ün
    kazanımı sessizce bu göreve devredilmiş sanılır.
    """
    class Kaydeden:
        def __init__(self, hedef):
            self._hedef = hedef
            self.dokunulan: set[str] = set()

        def __getattr__(self, ad):
            self.dokunulan.add(ad)
            return getattr(self._hedef, ad)

    dokunulan = {}
    for gauge in GAUGES:
        vekil = Kaydeden(GR)
        null_potentials(theory=vekil, M=1.0, gauge=gauge)
        dokunulan[gauge] = vekil.dokunulan

    assert dokunulan["schwarzschild"] == {"gamma", "beta"}
    assert dokunulan["isotropic"] == {"gamma", "beta"}
    assert dokunulan["schwarzschild"] & dokunulan["isotropic"] == {"gamma", "beta"}
    assert "lam" not in dokunulan["schwarzschild"]


# ---------------------------------------------------------------------------
# Step 1 — BU BORUNUN KENDİ HATA ÇUBUĞU
# ---------------------------------------------------------------------------
def test_BUKULME_TABANI_KENDI_BORUSUNDAN_ve_ETIKETI_AYRI():
    """SINIF: HATA ÇUBUĞU — spec §4.3, karışık boru YASAK.

    Bükülmenin tabanı ne karelemeden ne ODE'den okunur; `deflection_ratio`
    KENDİ `PIPELINE` etiketini ve KENDİ merdivenini taşır.

    ÖLÇÜLEN (M = 1, Richardson (1e5, 3e5)):

        ayar             taban          S/G           taban/gerçek hata
        schwarzschild    3.651917e-10   5.4766e+09    1.173
        isotropic        8.436665e-11   2.3706e+10    0.432

    Schwarzschild penceresi YUVARLAMA SINIRLIDIR ve taban gerçek hatayı 1.17
    kat içinde öngörüyor — Task 2'nin karelemede ölçtüğü 0.861 ile aynı
    kalitede. İzotropik tarafta taban gerçek hatayı 2.3 kat KÜÇÜK gösteriyor;
    yani bu boruda hata çubuğu bir ÖNGÖRÜDÜR, bir GARANTİ değil.
    """
    schw = newton_noise_floor(deflection_ratio, M=1.0)
    izo = newton_noise_floor(deflection_ratio, M=1.0, gauge="isotropic")

    assert schw.pipeline == ("bukulme | gauge=schwarzschild | "
                             "fizik-kapali merdiven 6/6 canli")
    assert izo.pipeline == ("bukulme | gauge=isotropic | "
                            "fizik-kapali merdiven 6/6 canli")
    assert "kareleme" not in schw.pipeline and "ode" not in schw.pipeline

    assert schw.value == pytest.approx(3.651917e-10, rel=1e-5, abs=0.0)
    assert izo.value == pytest.approx(8.436665e-11, rel=1e-5, abs=0.0)
    assert schw.ratio == pytest.approx(5.4766e09, rel=1e-4)
    assert izo.ratio == pytest.approx(2.3706e10, rel=1e-4)
    assert schw.verdict == izo.verdict == "SAGLAM"

    gercek_schw = abs(deflection_ratio(theory=GR, M=1.0) - 2.0)
    gercek_izo = abs(deflection_ratio(theory=GR, M=1.0, gauge="isotropic") - 2.0)
    assert schw.value / gercek_schw == pytest.approx(1.173, abs=0.01)
    assert izo.value / gercek_izo == pytest.approx(0.432, abs=0.01)


def test_KUTLEYI_KAPAT_NOKTASI_DEJENERE_BU_BIR_OLCUM_DEGIL_SABIT():
    """SINIF: ÇÜRÜTME — Task 2'nin dersinin bükülmedeki birebir tekrarı.

    "Fiziği kapat" için akla ilk gelen nokta `M → 0`'dır. ÖLÇÜLDÜ: orada
    `A = T = 1`, `Q(u) = u0 + u` olur, integrandın TAMAMI bit bit 1.0'a düşer
    ve `delta` TAM 0.0 çıkar — iki ayarda da, `M = 1e-200`'de. Yani o nokta
    bir ÖLÇÜM değil bir SABİTTİR ve bir hata çubuğu üretemez.

    (Ve `M > 0` kapısı `M = 0`'ın kendisini zaten reddediyor — nokta API'den
    de erişilemez.)

    Sevk edilen taban bu yüzden `1 + γ → 0` LİMİTİNDEN okunur.
    """
    for gauge in GAUGES:
        assert deflection_angle(theory=GR, M=1e-200, b=1e5, gauge=gauge) == 0.0
        kernel = null_kernel(theory=GR, M=1e-200, b=1e5, gauge=gauge)
        for s in (0.0, 0.3, 0.7, 1.0):
            u = kernel.turning_point * s
            assert kernel.turning_point * (1.0 + s) * null_w(u, kernel) == 1.0
    with pytest.raises(InvalidInputError, match="M pozitif"):
        deflection_angle(theory=GR, M=0.0, b=1e5)


def test_MERDIVENIN_UCLARI_ZORUNLU_DEGIL_KARELEMEDEN_FARKLI():
    """SINIF: PENCERE ÖLÇÜMÜ — ve devralınmış bir gerekçenin REDDİ.

    `NEWTON_PROBE_LAMBDAS` (kareleme) dokstringinde iki uç ÖLÇÜLEREK
    zorlanmıştı: üstünde kesme hatası giriyordu, altında basamak ÖLÜYORDU
    (`w` tam 1.0'a yuvarlanıyor). O gerekçe bükülmeye TAŞINMAZ ve bu dosya
    onu ölçerek reddeder (Schwarzschild, Richardson (1e5, 3e5)):

        eps    5e-1      1e-1      1e-2      1e-3      1e-4      1e-9
        artık  1.353e-10 8.141e-11 3.082e-10 3.398e-10 3.652e-10 2.451e-10
        eps    1e-12     1e-14     1e-16
        artık  3.448e-10 3.458e-10 3.458e-10

    On beşten fazla dekat boyunca artık 8.14e-11 ile 3.65e-10 arasında, yani
    4.5 kat: merdiven FİZİĞİ DEĞİL ARİTMETİĞİ ölçüyor. Alt uçta DOYMA var
    (eps ≤ 1e-12'de `oran` bit bit aynı) ama ÖLÜM yok — bu boru `DEJENERE`
    olamaz. Sevk edilen 1e-4 … 1e-9 dilimi bu yüzden açıkça bir TERCİHTİR.
    """
    artiklar = {}
    for eps in (0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-9, 1e-12, 1e-14, 1e-16):
        t = Theory(gamma=-1.0 + eps)
        oran = deflection_ratio(theory=t, M=1.0)
        artiklar[eps] = abs(oran - deflection_closed_form(t))
        assert oran != 0.0          # hiçbir basamak ÖLMÜYOR

    assert artiklar[0.5] == pytest.approx(1.353e-10, rel=1e-3, abs=0.0)
    assert artiklar[1e-1] == pytest.approx(8.141e-11, rel=1e-3, abs=0.0)
    assert artiklar[1e-4] == pytest.approx(3.652e-10, rel=1e-3, abs=0.0)
    assert artiklar[1e-16] == pytest.approx(3.458e-10, rel=1e-3, abs=0.0)
    assert max(artiklar.values()) / min(artiklar.values()) == pytest.approx(
        4.5, abs=0.1)

    # DOYMA: eps <= 1e-12'de dönen `oran` BİT BİT aynı.
    doyma = {deflection_ratio(theory=Theory(gamma=-1.0 + e), M=1.0)
             for e in (1e-12, 1e-14, 1e-16)}
    assert len(doyma) == 1
    assert doyma.pop() == pytest.approx(3.45805844e-10, rel=1e-8, abs=0.0)
    assert DEFLECTION_PROBE_EPSILONS == (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)


# ---------------------------------------------------------------------------
# Step 1 — ÇEKİRDEĞİN BEDAVA SINAVLARI ve KAPILAR
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gauge", GAUGES)
def test_DEFLASYONUN_BEDAVA_SINAVI_u0_q0_ESITTIR_n0(gauge):
    """SINIF: ŞEMA ÇIPASI — deflasyonun KULLANMADIĞI katsayı geri üretiliyor.

    `N(u) = (u0 − u)·Q(u)` deflasyonu beş katsayıdan yalnız dördünü (`n1`..`n4`)
    kullanır. Beşincisi (`n0 = S(0)/b²`) hiçbir yerde kullanılmaz, dolayısıyla
    `u0·q0 = n0` bedava bir sınavdır. Ölçülen |u0·q0 − n0|:

        b        schwarzschild   isotropic
        1e3      6.353e-22       4.235e-22
        1e4      1.323e-23       1.158e-23
        1e5      1.163e-25       1.034e-25
        1e6      1.212e-27       1.212e-27
    """
    beklenen = {1e3: (6.353e-22, 4.235e-22), 1e4: (1.323e-23, 1.158e-23),
                1e5: (1.163e-25, 1.034e-25), 1e6: (1.212e-27, 1.212e-27)}
    i = 0 if gauge == "schwarzschild" else 1
    for b, hedef in beklenen.items():
        k = null_kernel(theory=GR, M=1.0, b=b, gauge=gauge)
        # `abs=0.0` ZORUNLU ve sebebi MUTASYONLA olculdu: `pytest.approx`in
        # varsayilan `abs`i **1e-12**'dir, yani 1e-22 mertebesindeki bir
        # beklenen deger icin bagil olcut TAMAMEN devre disi kalir. O
        # varsayilanla `constant_residual = 0.0` mutanti o turdaki 236 testin
        # HEPSINDEN sag cikiyordu.
        assert k.constant_residual != 0.0
        assert abs(k.constant_residual) == pytest.approx(hedef[i], rel=1e-3,
                                                         abs=0.0)


@pytest.mark.parametrize("u", [0.0, 1e-6, 1e-3, 1e-2])
def test_IZOTROPIK_NULL_YOLU_ISOTROPIC_POTENTIAL_ILE_TUTARLI(u):
    """SINIF: TUTARLILIK SINAVI — gövde paylaşımı YOK, sınav VAR.

    `null_potentials(gauge="isotropic")` ile `isotropic_potential` AYNI
    metriği tarif etmek zorundadır ama gövdeleri paylaşmazlar; paylaşsalardı
    ortak bir düğme doğardı. Bunun yerine burada BİT BİT eşitlik sınanır.
    """
    pot = null_potentials(theory=GR, M=1.0, gauge="isotropic")
    ip = isotropic_potential(u, theory=GR, M=1.0)
    assert evaluate_poly(pot.time_poly, u) == ip.time_factor
    assert evaluate_poly(pot.source_poly, u) == ip.space_factor


def test_B_KRIT_ALTINDA_FOTON_YAKALANIR_KAPI_ACIK_SOYLUYOR():
    """SINIF: KAPI. `b ≤ b_krit`'te dönüş noktası YOKTUR.

    Sessizce inanılır bir sayı döndürmek yerine `InvalidInputError` atılır.
    ÖLÇÜLDÜ (GR, M = 1): `b = 5.19` ve `b = b_krit` reddediliyor,
    `b = 5.20` geçiyor ve δ = 6.810371956663369 rad (2.17 π) veriyor.
    """
    for b in (1.0, 5.0, 5.19, 3 * math.sqrt(3)):
        with pytest.raises(InvalidInputError, match="foton"):
            deflection_angle(theory=GR, M=1.0, b=b)
    assert deflection_angle(theory=GR, M=1.0, b=5.20) == 6.810371956663369


def test_AYAR_ADLARI_TEK_KAYNAK():
    """SINIF: ŞEMA ÇIPASI — Task 1'in "iki alan aynı şeyi belirlemesin"i.

    `geodesic.GAUGES` artık `metric.GAUGE_NAMES`'in TA KENDİSİDİR (yeniden
    yazılmış bir kopya değil), ve `_DPHI`'nin anahtarları da onunla aynı.
    """
    assert GAUGES is GAUGE_NAMES
    assert tuple(geodesic_modulu._DPHI) == GAUGE_NAMES
    assert GAUGE_NAMES == ("schwarzschild", "isotropic")
    with pytest.raises(InvalidInputError, match="bilinmeyen ayar"):
        deflection_angle(theory=GR, M=1.0, b=1e5, gauge="izotropik")
    with pytest.raises(InvalidInputError, match="bilinmeyen ayar"):
        null_potentials(theory=GR, M=1.0, gauge="kerr")


def test_QUAD_TOLERANSI_BU_BORUDA_TAMAMEN_ETKISIZ():
    """SINIF: PENCERE ÖLÇÜMÜ — bir düğmenin DÜĞME OLMADIĞININ ölçümü.

    `NULL_EPSABS`/`NULL_EPSREL` değiştirilip sevk edilen fonksiyon koşuldu,
    dört ayar: `quad`ın kendi varsayılanı (1.49e-8, 1.49e-8), (1e-12, 1e-11),
    (1e-14, 1e-13), (1e-16, 1e-15). DÖRDÜNDE DE `delta`, `oran` ve gürültü
    tabanı BİT BİT aynı. Sebep yapısal: `θ` ikamesinden sonra integrand
    [0, π/2]'de pürüzsüzdür ve 21 noktalı Gauss-Kronrod kuralı onu daha ilk
    bölmede makine hassasiyetinde integre eder.

    Bu, "her düğmeyi beyan et" kuralının tersi kadar önemli olan yarısıdır:
    beyan edilen bir düğmenin ETKİSİZ olduğu da ölçülür.
    """
    eski = (geodesic_modulu.NULL_EPSABS, geodesic_modulu.NULL_EPSREL)
    try:
        degerler = set()
        oranlar = set()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            for ea, er in ((1.49e-8, 1.49e-8), (1e-12, 1e-11), (1e-14, 1e-13),
                           (1e-16, 1e-15)):
                geodesic_modulu.NULL_EPSABS = ea
                geodesic_modulu.NULL_EPSREL = er
                degerler.add(deflection_angle(theory=GR, M=1.0, b=1e5))
                oranlar.add(deflection_ratio(theory=GR, M=1.0))
        assert degerler == {4.000117813962589e-05}
        assert oranlar == {1.9999999996887523}

        # EN SIKI AYAR UYARI VERİYOR AMA DEĞERİ DEĞİŞTİRMİYOR — bu ikisi
        # AYRI şeylerdir ve `psi` borusundan ayrıldığı yer tam burasıdır:
        # orada aynı ayar tabanı 1.5942e-09'dan 2.2123e-09'a ZIPLATIYORDU.
        geodesic_modulu.NULL_EPSABS, geodesic_modulu.NULL_EPSREL = 1e-16, 1e-15
        with pytest.warns(IntegrationWarning):
            assert deflection_angle(theory=GR, M=1.0,
                                    b=1e5) == 4.000117813962589e-05
    finally:
        geodesic_modulu.NULL_EPSABS, geodesic_modulu.NULL_EPSREL = eski


def test_RICHARDSON_MERDIVENI_OPTIMUMDAN_SONRA_BOZULUYOR():
    """SINIF: PENCERE ÖLÇÜMÜ. Bir dışdeğerleme SONSUZA KADAR İYİLEŞMEZ.

    Sevk edilen kodla ölçülen dürüst hane (GR, M = 1, Schwarzschild):

        (1e3,3e3) 5.64 | (1e4,3e4) 7.64 | (1e4,1e5) 8.16 | (1e5,3e5) 9.81
        (1e5,1e6) 9.37 | (1e6,3e6) 9.30 | (1e6,1e7) 8.41 | (1e7,3e7) 7.69

    Optimum (1e5, 3e5)'tedir — sevk edilen çift — ve ondan sonra düşüyor.
    """
    merdiven = {(1e3, 3e3): 5.64, (1e4, 3e4): 7.64, (1e4, 1e5): 8.16,
                (1e5, 3e5): 9.81, (1e5, 1e6): 9.37, (1e6, 3e6): 9.30,
                (1e6, 1e7): 8.41, (1e7, 3e7): 7.69}
    haneler = {}
    for cift, beklenen in merdiven.items():
        oran = deflection_ratio(theory=GR, M=1.0, richardson=cift)
        haneler[cift] = -math.log10(abs(oran - 2.0) / 2.0)
        assert haneler[cift] == pytest.approx(beklenen, abs=0.01)
    assert max(haneler, key=haneler.get) == DEFAULT_NULL_RICHARDSON == (1e5, 3e5)
    assert haneler[(1e7, 3e7)] < haneler[(1e5, 3e5)] - 2.0


def test_ESIT_BASAMAKLI_RICHARDSON_REDDEDILIYOR():
    """SINIF: KAPI. Eşit basamaklarda `1/b`'de dışdeğerleme sıfıra bölmedir."""
    with pytest.raises(InvalidInputError, match="AYRI olmali"):
        deflection_ratio(theory=GR, M=1.0, richardson=(1e5, 1e5))
    with pytest.raises(InvalidInputError, match="CIFTI olmali"):
        deflection_ratio(theory=GR, M=1.0, richardson=(1e5, 3e5, 1e6))
