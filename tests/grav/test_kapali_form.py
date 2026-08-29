"""ODE yörünge motoru, kapalı-form türevler ve GÜÇLÜ ALAN noktası —
Modül 6 Task 4, Step 2 + Step 3 + Step 4.

Üç bölüm, üç ayrı SINIF — ve dosyanın adı üçüncüsünden geliyor:

  A. **ODE = YÖRÜNGE MOTORU, ÇIPA DEĞİL.** `run_trajectory` bir sayı
     belgelemez, bir eğri üretir. Ama ODE'nin **kendi borusu** olmak zorundadır
     (spec §4.3, karışık boru yasağı), ve `perihelion_coefficient_ode` bunun
     için vardır: ODE'nin gürültü tabanı ODE'den, karelemeninki karelemeden.
  B. **KAPALI-FORM TÜREVLER.** Beş sayı; hiçbiri yörünge çözmez. Ölçüt
     deponun kendi ölçütü: *"simülasyon bu sayı hakkında YANLIŞ olabilir
     miydi?"* Bir sabiti geri okumak model çözmez.
  C. **GÜÇLÜ ALAN NOKTASI** (spec §7.5): statik yolun göremediği bir kusuru
     yakalayan ucuz yol. Dinamik `b_krit` sevk EDİLMİYOR ve bu dosya nedenini
     ölçerek gösteriyor.

PENCERE — her sayı sevk edilen kodla ölçüldü:

- `G = c = 1`, `M = 1.0`, `e = 0.3` (aksi yazılmadıkça).
- ODE: `scipy.integrate.solve_ivp`, **DOP853**, `rtol = 1e-13`,
  `atol = 1e-15`, `t0 = 0`, olay `du/dφ = 0` (`direction = -1`), süzgeç
  *"φ > π olan İLK kök"*, kapı `π < Δφ < 3π`; Richardson **(1e4, 3e4)**.
- Kapalı-form türevler: `scipy.optimize.brentq`, `xtol = 1e-18`,
  `rtol = 8.9e-16`; `μ_c`'de merkezi fark `h = 1e-4` ve eşik
  `max Re(λ) > 1e-7`; apsidal üste merkezi fark `h = 1e-4`.
- Dinamik yollar SEVK EDİLMİYOR; bu dosyanın kendi gövdeleriyle ölçülüyorlar
  ve yöntemleri aşağıda yazılı.

`slow` EŞİĞİ ÖLÇÜLDÜ, DEVRALINMADI — ve bu dosyada BİR test işaretlendi.
Ölçülen iki küme ve aralarındaki BOŞLUK (`--durations`, tek koşum):

    işaretsiz, en pahalı : **0.21 s**  (`test_STATIK_YOL_DINAMIK_YOLDAN...`,
                           dinamik ISCO: 44 tur × `φ ∈ [0, 500]` ODE)
    ikinci              : 0.18 s      (`test_ODE_TABANI_KIPIRDIYOR...`)
    `slow`, tek üye     : **2.78 s**  (`test_SPEC_4_4_UN_METOT_SATIRI_DUSTU`)

Eşik **1.0 s** seçildi; aşağıda 4.8 kat, yukarıda 2.8 kat pay var, yani
sınıflandırma makine gürültüsüyle (bu depoda %20'ye varıyor) DEĞİŞMEZ.
Boşluk yapısaldır: metot tablosu Radau'yu **7 kez** koşar (bir doğrudan çağrı
+ altı merdiven basamağı) ve Radau tek çağrıda 247 ms sürer — yani o tek test
tek başına ~1.7 s Radau taşır. `tests/em`'in 1.0 s eşiğiyle aynı sayı çıkması
bir DEVRALMA DEĞİL, bağımsız bir ölçümün aynı yere düşmesidir.

Bu dosya: 21 test, **4.07 s**. Toplam `tests/grav`: 237 test, **4.36 s**.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from qlab.engine.grav.geodesic import (
    DEFAULT_ODE_RICHARDSON,
    ODE_ATOL,
    ODE_METHODS,
    ODE_RTOL,
    Trajectory,
    deflection_angle,
    perihelion_coefficient,
    perihelion_coefficient_ode,
    run_trajectory,
)
from qlab.engine.grav.metric import Theory, evaluate_poly, null_potentials
from qlab.errors import InvalidInputError
from qlab.observable.grav_obs import (
    APSIDAL_GRID,
    apsidal_exponent,
    critical_impact_parameter,
    deflection_ratio,
    isco_radius,
    lagrange_mu_c,
    newton_noise_floor,
    photon_sphere,
)

GR = Theory()
ALTI_PI = 6.0 * math.pi


def _hane(deger, hedef):
    return -math.log10(abs(deger - hedef) / abs(hedef))


# ===========================================================================
# A. ODE — YÖRÜNGE MOTORU ve KENDİ BORUSU (Step 2)
# ===========================================================================
def test_ODE_KENDI_BORUSUDUR_ETIKETI_KARELEMEDEN_AYRI():
    """SINIF: HATA ÇUBUĞU — spec §4.3'ün Task 4'teki asıl sınavı.

    KARIŞIK BORU YASAK: ODE'nin hata çubuğu ODE'nin KENDİ borusundan okunur.
    Karelemenin tabanı (Schwarzschild `1.594187e-09`, izotropik
    `2.346903e-09`) ODE için GEÇERLİ DEĞİLDİR — üç mertebe uzakta:

        boru        etiket                            taban
        kareleme    "kareleme | gauge=schwarzschild"   1.594187e-09
        bükülme     "bukulme | gauge=schwarzschild"    3.651917e-10
        ODE         "ode | method=DOP853"              1.9218071398402047e-06

    ÖLÇÜLEN (GR, M = 1, e = 0.3, DOP853, rtol 1e-13, atol 1e-15,
    Richardson (1e4, 3e4)):

        C_ode = 18.849556406088652     6π = 18.84955592153876
        fark  = +4.845e-07  →  **7.59 dürüst hane**
        taban = 1.9218071398402047e-06,  S/G = 9.8082e+06,  6/6 basamak canlı
        taban / gerçek hata = 3.966

    Yani bu pencerede hata çubuğu MUHAFAZAKARDIR (karelemede oran 0.861'di,
    yani hafif iyimser).
    """
    nf = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3)
    assert nf.pipeline == "ode | method=DOP853 | fizik-kapali merdiven 6/6 canli"
    assert nf.value == 1.9218071398402047e-06
    assert nf.verdict == "SAGLAM"
    assert nf.ratio == pytest.approx(9.8082e06, rel=1e-4)

    c = perihelion_coefficient_ode(M=1.0, e=0.3)
    assert c == 18.849556406088652
    assert _hane(c, ALTI_PI) == pytest.approx(7.59, abs=0.01)
    assert nf.value / abs(c - ALTI_PI) == pytest.approx(3.966, abs=0.01)

    # Karelemenin tabanı ODE için GEÇERLİ DEĞİL — üç mertebe uzakta.
    kar = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3)
    assert kar.pipeline.startswith("kareleme")
    assert nf.value / kar.value > 1000.0


def test_ODE_TABANI_KIPIRDIYOR_ve_MERDIVEN_HIC_OLMUYOR():
    """SINIF: HATA ÇUBUĞU — Task 2'nin dersi: *bir taban sevk etmeden önce
    onun kıpırdadığını ölç.*

    ODE tabanı üç bağımsız yönde kıpırdıyor:

      (1) RICHARDSON çifti: (1e3,3e3) → 2.4867e-08, (3e3,1e4) → 1.8098e-07,
          (1e4,3e4) → 1.9218e-06, (3e4,1e5) → 2.5713e-05,
          (1e5,3e5) → 2.3538e-04. Yani `p` ile birlikte büyüyor — çünkü ODE
          hatası `Δφ`'da mutlak, `C = Δφ·p/M` onu `p` ile çarpıyor.
      (2) `rtol`: (1e3,3e3) penceresinde rtol 1e-13 → 2.4867e-08,
          rtol 1e-10 → 3.0794e-05, yani **1238 kat**.
      (3) METOT: DOP853 1.9218e-06 ↔ Radau 3.2953e-09, yani **583 kat**
          (tam tablo aşağıda).

    VE MERDİVENİN HİÇBİR BASAMAĞI ÖLMÜYOR: karelemede `λ` küçüldükçe `w` tam
    1.0'a yuvarlanıp basamak ölüyordu (p/M = 1e13'te 0/6 canlı → `DEJENERE`).
    ODE'de böyle bir şey YOK — beş pencerede de 6/6 canlı. Yani `DEJENERE`
    bu boruda ERİŞİLEMEZ; karelemenin tam tersi.
    """
    merdiven = {(1e3, 3e3): 2.4867e-08, (3e3, 1e4): 1.8098e-07,
                (1e4, 3e4): 1.9218e-06, (3e4, 1e5): 2.5713e-05,
                (1e5, 3e5): 2.3538e-04}
    for cift, beklenen in merdiven.items():
        nf = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3,
                                richardson=cift)
        assert nf.value == pytest.approx(beklenen, rel=1e-3, abs=0.0)
        assert nf.value != 0.0
        assert "6/6 canli" in nf.pipeline
        assert nf.verdict != "DEJENERE"
    # Kıpırdıyor: uçlar arasında dört mertebe.
    assert merdiven[(1e5, 3e5)] / merdiven[(1e3, 3e3)] > 9000.0

    gevsek = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3,
                                richardson=(1e3, 3e3), rtol=1e-10, atol=1e-12)
    assert gevsek.value == pytest.approx(3.0794e-05, rel=1e-3)
    assert gevsek.value / merdiven[(1e3, 3e3)] == pytest.approx(1238, rel=1e-2)


@pytest.mark.slow
def test_SPEC_4_4_UN_METOT_SATIRI_DUSTU():
    """SINIF: ÇÜRÜTME. Spec §4.4: *"Taban DOP853'e özgü değil (RK45, Radau,
    LSODA aynı mertebede)."* ÖLÇÜLDÜ, YANLIŞ.

    Sevk edilen kodla (Richardson (1e4, 3e4), rtol 1e-13, atol 1e-15, aynı
    olay süzgeci):

        metot     dürüst hane   taban          süre
        DOP853        7.59      1.9218e-06       4.2 ms
        RK45          6.83      1.3768e-06      21.4 ms
        Radau         7.12      3.2953e-09     247.4 ms
        LSODA         5.84      2.5587e-05       3.7 ms
        BDF           5.75      3.2038e-05      61.7 ms

    (Bu testin kendisi **2.78 s** sürüyor ve dosyanın TEK `slow` üyesidir:
    Radau'yu 7 kez koşuyor — bir doğrudan çağrı + altı merdiven basamağı.)

    Taban 3.2953e-09 ile 3.2038e-05 arasında: **9722 kat**, yani DÖRT MERTEBE.
    "Aynı mertebede" cümlesi bu boruda YANLIŞ, ve bu yüzden `method`
    `PIPELINE_KEYS`e girer — etiket hangi integratörün koştuğunu TAŞIR.

    DÜRÜST NİTELİK: ilk ölçümümde RK45 ve LSODA tabanı `2.5133e+05` çıkıyordu
    ve bunu "metot arızası" sanmıştım. Değildi — BENİM olay süzgecim
    ("son kökü al") bozuktu: `v ≈ 0` civarında gürültüye giren olay işlevi
    4π'ye yakın sahte kökler üretiyor, `(4π − 2π)·p` = 2.513274e+05 çıkıyordu.
    Süzgeç *"φ > π olan İLK kök"*e çevrilince beş metot da düzgün koşuyor.
    Yukarıdaki tablo DÜZELTİLMİŞ süzgeçle alınmıştır.
    """
    tablo = {"DOP853": (7.59, 1.9218e-06), "RK45": (6.83, 1.3768e-06),
             "Radau": (7.12, 3.2953e-09), "LSODA": (5.84, 2.5587e-05),
             "BDF": (5.75, 3.2038e-05)}
    assert set(tablo) == set(ODE_METHODS)
    tabanlar = {}
    for metot, (hane, taban) in tablo.items():
        c = perihelion_coefficient_ode(M=1.0, e=0.3, method=metot)
        assert _hane(c, ALTI_PI) == pytest.approx(hane, abs=0.01), metot
        nf = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3,
                                method=metot)
        assert nf.value == pytest.approx(taban, rel=1e-3, abs=0.0), metot
        assert nf.pipeline == (f"ode | method={metot} | "
                               "fizik-kapali merdiven 6/6 canli")
        tabanlar[metot] = nf.value
    assert max(tabanlar.values()) / min(tabanlar.values()) == pytest.approx(
        9722, rel=1e-2)
    assert max(tabanlar.values()) / min(tabanlar.values()) > 1000.0


def test_PIPELINE_KEYS_OLCUTU_GOVDE_MI_TABAN_MI():
    """SINIF: ŞEMA ÇIPASI — Task 3'ün YAZMADIĞI ölçütü Task 4 yazıyor.

    Task 3 `gauge`'u `PIPELINE_KEYS`e eklerken gerekçesi *"tabanları FARKLI"*
    idi. O ölçüt tek başına TUTMAZ ve bu ölçüldü: tabanı değiştiren ama
    GÖVDEYİ değiştirmeyen argümanlar var —

        `rtol` 1e-13 → 1e-10  : ODE tabanı 2.4867e-08 → 3.0794e-05 (1238 kat)
        `richardson` (1e5,3e5) → (1e4,3e4)
                              : bükülme tabanı 3.6519e-10 → 2.4632e-08 (67 kat)

    ve ikisi de AYNI satırları koşturur. Doğru ölçüt **GÖVDE**dir: `gauge` iki
    ayrı kareleme çekirdeği seçer, `method` beş ayrı integratör seçer; `rtol`
    ile `richardson` seçmez. Sevk edilen üç borunun anahtarları bu ölçüte göre:

        kareleme → ("gauge",)   bükülme → ("gauge",)   ode → ("method",)
    """
    ode_taban = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3,
                                   richardson=(1e3, 3e3)).value
    ode_gevsek = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3,
                                    richardson=(1e3, 3e3), rtol=1e-10,
                                    atol=1e-12).value
    assert ode_gevsek / ode_taban > 1000.0
    buk_dar = newton_noise_floor(deflection_ratio, M=1.0).value
    buk_genis = newton_noise_floor(deflection_ratio, M=1.0,
                                   richardson=(1e4, 3e4)).value
    assert buk_genis / buk_dar == pytest.approx(67, rel=0.02)

    # ...ama ETİKET DEĞİŞMİYOR, çünkü gövde aynı:
    a = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3,
                           richardson=(1e3, 3e3))
    b = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3,
                           richardson=(1e3, 3e3), rtol=1e-10, atol=1e-12)
    assert a.pipeline == b.pipeline
    assert perihelion_coefficient_ode.PIPELINE_KEYS == ("method",)
    assert perihelion_coefficient.PIPELINE_KEYS == ("gauge",)
    assert deflection_ratio.PIPELINE_KEYS == ("gauge",)


def test_ODE_BORUSUNDA_CURUK_ERISILEBILIR_SEVK_EDILMIS_IDDIA_DUSTU():
    """SINIF: ÇÜRÜTME — ve düşen iddia SEVK EDİLMİŞ BİR DOKSTRINGDE.

    `newton_noise_floor` dokstringi Task 2'de şunu yazıyordu: *"`CURUK` dalı
    başka borular için duruyor — N-body tarafında `m2 = 0` tabanının sinyali
    `~m2²` gittiği için orada bir KESİŞİM vardır."* Task 4 ODE borusunu
    getirdi ve o cümle İLK KEZ SINANABİLİR hâle geldi. SINANDI VE DÜŞTÜ:

        rtol=1e-5, atol=1e-7, richardson=(1e4,3e4)
            taban 5.5342e+02, sinyal 535.542614, S/G 0.967704  → CURUK
        rtol=1e-6, atol=1e-8, richardson=(1e5,3e5)
            taban 5.4595e+03, sinyal 5441.593263, S/G 0.996726 → CURUK

    Yani `CURUK`un fizik kapsaması N-body'yi (Task 5) BEKLEMİYOR.

    AMA İDDİANIN RUHU AYAKTA VE O DA ÖLÇÜLDÜ: bu boruda `CURUK` yalnızca
    S/G = 1'in çok ince bir komşuluğunda çıkıyor, ve bu YAPISALDIR —
    integratör hatası baskın olduğunda aynı hata hem `signal`ı (|C| ≈ |hata|)
    hem tabanı belirler, oran 1'e raptolunur. `p` büyütülerek yaklaşıldığında
    (rtol 1e-13) S/G = 1'in ÜSTÜNDEN geliyor: (1e8,3e8) → 1.181868,
    (1e10,3e10) → 1.000008. Yani geniş ve güvenli bir `CURUK` bandı hâlâ
    N-body'den bekleniyor — ama artık "erişilemez" diye değil, **"dar"** diye.
    """
    a = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3,
                           richardson=(1e4, 3e4), rtol=1e-5, atol=1e-7)
    assert a.verdict == "CURUK"
    assert a.ratio == pytest.approx(0.967704, rel=1e-3)
    assert a.ratio < 1.0

    b = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3,
                           richardson=(1e5, 3e5), rtol=1e-6, atol=1e-8)
    assert b.verdict == "CURUK"
    assert b.ratio == pytest.approx(0.996726, rel=1e-3)

    # BAND DAR: büyük `p`'de oran 1'e ÜSTTEN yapışıyor, geniş bir CURUK yok.
    for cift, beklenen in (((1e8, 3e8), 1.181868), ((1e10, 3e10), 1.000008)):
        c = newton_noise_floor(perihelion_coefficient_ode, M=1.0, e=0.3,
                               richardson=cift)
        assert c.ratio == pytest.approx(beklenen, rel=1e-3)
        assert c.verdict == "SAGLAM"


def test_ODE_ILE_KARELEME_CAPRAZ_KONTROL_DEGIL_SPEC_7_3_OLCULDU():
    """SINIF: ÖLÇÜLMÜŞ OLUMSUZ ÖRNEK. Spec §7.3'ün sayısı İLK KEZ sevk edilen
    kodla üretildi.

    Spec §7.3, kareleme ↔ ODE karşılaştırmasının bir çapraz kontrol OLMADIĞINI
    şu oranla ölçmüştü: `|C_kar − C_ode| / |6π − C_ode| = 1.000`. Yani kareleme,
    ODE'ye kıyasla `6π`'nin YERİNE GEÇİYOR — ve `6π` zaten bedava. Bu depoda
    sevk edilen iki boruyla ölçülen: **1.0038223922852243**.

    Dolayısıyla ODE bir ÇIPA DEĞİLDİR; çıpa karelemedir (10.01 dürüst hane),
    ODE 7.59'da kalır ve rolü yörünge üretmektir.
    """
    c_ode = perihelion_coefficient_ode(M=1.0, e=0.3)
    c_kar = perihelion_coefficient(M=1.0, e=0.3)
    oran = abs(c_kar - c_ode) / abs(ALTI_PI - c_ode)
    assert oran == pytest.approx(1.0038223922852243, rel=1e-12)
    assert oran == pytest.approx(1.0, abs=0.01)
    assert _hane(c_kar, ALTI_PI) > _hane(c_ode, ALTI_PI) + 2.0


def test_T0_KUSURU_DEGERDE_DEGIL_CIKARMADA():
    """SINIF: ÇÜRÜTME — spec §4.4'ün teşhisi DARALTILDI.

    Spec §4.4 *"`t0 = 0` ZORUNLU"* diyor ve `t0 = 1e-9`'da tabanın
    `1.0001e-9·p` çıktığını yazıyor. Kusur `t0`'ın DEĞERİNDE değil, süpürülen
    açının `t_olay − t0` yerine `t_olay` diye OKUNMASINDAYDI. Bu dosyanın
    kendi kopyasıyla, sevk edilen pencerede (DOP853, rtol 1e-13, atol 1e-15,
    λ = 0, tek basamak):

        p     t0      Δφ = t_olay − t0     Δφ = t_olay  (KUSURLU)
        1e3   0       +1.135092e-09        +1.135092e-09
        1e3   1e-9    +1.137757e-09        +1.001138e-06  (= 1e-9·p + taban)
        1e4   0       +1.317702e-07        +1.317702e-07
        1e4   1e-9    +1.317702e-07        +1.013177e-05  (= 1e-9·p + taban)

    Çıkarma yapılınca `t0` pratik olarak ETKİSİZ; yapılmayınca `1e-9·p` tam
    olarak geri geliyor. Sevk edilen sürüm ikisini birden yapar: `t0 = 0`
    KULLANIR *ve* çıkarır.

    İKİNCİ KUSUR — `t0`DAKİ SAHTE OLAY: başlangıç perihelde (`v = 0`)
    olduğu için `solve_ivp` olay işlevini `t0`'da tam 0.0 görür ve
    `direction = -1` koşulunu orada da sağlar. `t0 = 1e-9` bunun için
    yazılmıştı; doğru çözüm kökü SÜZMEKTİR.
    """
    def dphi(*, lam, p, t0, cikar):
        e, M = 0.3, 1.0
        u1, u2 = (1 - e) / p, (1 + e) / p
        a = 2.0 * lam * M / 3.0
        S, P = u1 + u2, u1 * u2
        ML2 = 0.5 * (S - a * (S * S - P))

        def rhs(phi, y):
            return (y[1], ML2 + lam * M * y[0] * y[0] - y[0])

        def olay(phi, y):
            return y[1]
        olay.direction = -1.0
        sol = solve_ivp(rhs, (t0, t0 + 4 * math.pi), (u2, 0.0), method="DOP853",
                        rtol=ODE_RTOL, atol=ODE_ATOL, events=olay)
        kok = [float(t) for t in sol.t_events[0] if t > t0 + math.pi]
        return (kok[0] - t0) if cikar else kok[0]

    beklenen = {
        (1e3, 0.0, True): 1.135092e-09, (1e3, 0.0, False): 1.135092e-09,
        (1e3, 1e-9, True): 1.137757e-09, (1e3, 1e-9, False): 1.001138e-06,
        (1e4, 0.0, True): 1.317702e-07, (1e4, 0.0, False): 1.317702e-07,
        (1e4, 1e-9, True): 1.317702e-07, (1e4, 1e-9, False): 1.013177e-05,
    }
    for (p, t0, cikar), hedef in beklenen.items():
        taban = (dphi(lam=0.0, p=p, t0=t0, cikar=cikar) - 2 * math.pi) * p
        assert taban == pytest.approx(hedef, rel=1e-4), (p, t0, cikar)

    # KUSURLU sütunda fazlalık TAM `1e-9*p`:
    for p in (1e3, 1e4):
        fazla = ((dphi(lam=0.0, p=p, t0=1e-9, cikar=False)
                  - dphi(lam=0.0, p=p, t0=1e-9, cikar=True)) * p)
        assert fazla == pytest.approx(1e-9 * p, rel=1e-9)

    # SAHTE OLAY: süzgeç olmasa `t0`daki kök ALINIRDI.
    assert dphi(lam=3.0, p=3e5, t0=0.0, cikar=True) > math.pi


def test_SUPURULEN_ACI_KAPISI_SESSIZ_YOLU_KAPATIYOR():
    """SINIF: KAPI — ve bir MUTASYON AÇIĞININ kapanışı.

    `_ode_cekirdek` iki kapı taşır ve İKİSİ DE SEVK EDİLEN ARGÜMANLARLA
    tetiklenebilir; ilk mutasyon turunda kapıyı `if False:` yapan mutant 236
    testin hepsinden sağ çıkmıştı, çünkü hiçbir test o kapıyı GEÇMİYORDU.

      (1) `π < Δφ < 3π` PENCERESİ — gevşek toleransta çözücü periyodu
          şaşırıyor ve olay süzgeci `π`'nin ötesinde YANLIŞ bir kök buluyor.
          ÖLÇÜLDÜ (DOP853, rtol = atol = 0.1, p = 100, e = 0.05):
          `Δφ = 11.75251386399735` — `3π` = 9.4248'in üstünde.
      (2) HİÇ KÖK YOK — aynı toleransta p = 10, e = 0.05'te `φ > π` olan
          dönüş noktası hiç bulunmuyor.

    İkisi de sessizce inanılır bir sayı döndürmek yerine `InvalidInputError`
    atıyor. Bu, "yörünge başına perihel kayması" gibi bir büyüklüğün
    TANIMSIZ olduğu yerde susmamanın karşılığıdır.
    """
    with pytest.raises(InvalidInputError, match="supurulen aci penceresi"):
        perihelion_coefficient_ode(theory=GR, M=1.0, e=0.05, p=100.0,
                                   richardson=None, rtol=0.1, atol=0.1)
    with pytest.raises(InvalidInputError, match="radyal periyot bulunamadi"):
        perihelion_coefficient_ode(theory=GR, M=1.0, e=0.05, p=10.0,
                                   richardson=None, rtol=0.1, atol=0.1)
    # Aynı kapılar yörünge motorunda da geçerli — ORTAK çekirdek.
    with pytest.raises(InvalidInputError, match="supurulen aci penceresi"):
        run_trajectory(theory=GR, M=1.0, p=100.0, e=0.05, rtol=0.1, atol=0.1)


def test_RUN_TRAJECTORY_SOZLESMESI_ve_N_POINTS_DOGRULUK_DUGMESI_DEGIL():
    """SINIF: ÇIPA DEĞİL — `run_trajectory` bir sayı belgelemez, eğri üretir.

    SÖZLEŞME (GR, M = 1, p = 3e5, e = 0.3, n_points = 2000):
      * `phi[0] == 0.0`, `phi[-1] == 6.283248140523368` (bir radyal periyot,
        `2π`den 9.65e-4 rad büyük — perihel kaymasının ta kendisi),
      * `u[0] == (1+e)/p == 4.333333333333333e-06` (perihelden başlar),
      * `u` aralığı `[u1, u2]` içinde, `r == 1/u` bit bit.

    `n_points` bir ÖRNEKLEME YOĞUNLUĞUDUR, DOĞRULUK DÜĞMESİ DEĞİL: 500 ve
    2000 nokta AYNI `phi[-1]`i verir (bit bit), çünkü çözücü adımını
    `rtol`/`atol` seçer. İkisi karıştırılırsa *"çözünürlüğü artırdım ama sonuç
    değişmedi"* diye sahte bir şaşırma doğar.

    MALİYET ÖLÇÜLDÜ ve PLANIN ORANI ÜRETİLEMEDİ (plan: 1.35 ms ↔ 19.44 ms,
    14.4 kat). Bu makinede: ODE + `dense_output` **2.482 ms**; BİRİKİMLİ
    kareleme **12.026 ms** (4.8 kat); NOKTA BAŞINA ayrı `quad` **207.568 ms**
    (83.6 kat). "Kat" sayısı tamamen karelemenin nasıl yazıldığına bağlı, o
    yüzden üçü de yazılı. ASIL gerekçe zaman değil: kareleme `phi(u)` verir,
    yani TERS FONKSİYONU.
    """
    t = run_trajectory(theory=GR, M=1.0, p=3e5, e=0.3)
    assert isinstance(t, Trajectory)
    assert len(t.phi) == len(t.u) == len(t.r) == 2000
    assert t.phi[0] == 0.0
    assert float(t.phi[-1]) == 6.283248140523368
    kayma = float(t.phi[-1]) - 2 * math.pi
    assert kayma == 6.283334378220928e-05
    assert kayma == pytest.approx(ALTI_PI / 3e5, rel=1e-4)
    u1, u2 = (1 - 0.3) / 3e5, (1 + 0.3) / 3e5
    assert float(t.u[0]) == u2
    assert u1 <= float(t.u.min()) and float(t.u.max()) <= u2 * (1 + 1e-12)
    assert np.all(t.r == 1.0 / t.u)

    az = run_trajectory(theory=GR, M=1.0, p=3e5, e=0.3, n_points=500)
    assert len(az.phi) == 500
    assert float(az.phi[-1]) == float(t.phi[-1])      # BİT BİT aynı

    with pytest.raises(InvalidInputError, match="ORNEKLEME"):
        run_trajectory(theory=GR, M=1.0, p=3e5, e=0.3, n_points=1)
    with pytest.raises(InvalidInputError, match="bilinmeyen integrator"):
        perihelion_coefficient_ode(M=1.0, e=0.3, method="Verlet")


def test_ODE_RICHARDSON_OPTIMUMU_RTOL_ILE_YER_DEGISTIRIYOR():
    """SINIF: PENCERE ÖLÇÜMÜ — ve planın "7.20 hane"si için EKSİK BEYAN.

    Sevk edilen kodla ölçülen dürüst hane:

        rtol      (1e3,3e3)  (3e3,1e4)  (1e4,3e4)  (3e4,1e5)  (1e5,3e5)
        1e-10       5.22       4.91       4.18       2.99       2.26
        1e-12       5.12       6.20       5.93       4.88       3.83
        1e-13       5.12       6.12       7.59       5.87       4.90

    Yani ODE'nin Richardson optimumu `rtol` ile YER DEĞİŞTİRİYOR: rtol 1e-12'de
    (3e3, 1e4), rtol 1e-13'te (1e4, 3e4). Spec §4.5'in *"ODE (t0 = 0) 1e4 (üst
    3e4) → 7.20"* satırı bir PENCERE BEYANI TAŞIMIYORDU; aynı çift bu depoda
    rtol 1e-12'de 5.93, rtol 1e-13'te 7.59 veriyor.
    """
    tablo = {
        1e-10: {(1e3, 3e3): 5.22, (3e3, 1e4): 4.91, (1e4, 3e4): 4.18,
                (3e4, 1e5): 2.99, (1e5, 3e5): 2.26},
        1e-12: {(1e3, 3e3): 5.12, (3e3, 1e4): 6.20, (1e4, 3e4): 5.93,
                (3e4, 1e5): 4.88, (1e5, 3e5): 3.83},
        1e-13: {(1e3, 3e3): 5.12, (3e3, 1e4): 6.12, (1e4, 3e4): 7.59,
                (3e4, 1e5): 5.87, (1e5, 3e5): 4.90},
    }
    for rtol, satir in tablo.items():
        atol = rtol * 1e-2
        for cift, beklenen in satir.items():
            c = perihelion_coefficient_ode(M=1.0, e=0.3, richardson=cift,
                                           rtol=rtol, atol=atol)
            assert _hane(c, ALTI_PI) == pytest.approx(beklenen, abs=0.01), (
                rtol, cift)
        en_iyi = max(satir, key=satir.get)
        if rtol == 1e-12:
            assert en_iyi == (3e3, 1e4)
        if rtol == 1e-13:
            assert en_iyi == (1e4, 3e4) == DEFAULT_ODE_RICHARDSON
    assert (ODE_RTOL, ODE_ATOL) == (1e-13, 1e-15)


# ===========================================================================
# B. KAPALI-FORM TÜREVLER (Step 3)
# ===========================================================================
def test_KAPALI_FORM_TUREVLER_BES_SAYI_SINIFI_YAZILI():
    """SINIF: **KAPALI-FORM TÜREV** — "fizik çıpası" DEĞİL.

    Ölçüt deponun kendi ölçütüdür: *"simülasyon bu sayı hakkında YANLIŞ
    olabilir miydi?"* Beşinde de yörünge ÇÖZÜLMEZ; hepsi bir cebirsel kökün
    ya da bir NOKTADAKİ doğrusallaştırmanın geri okunmasıdır.

        sayı              ölçülen                 referans                fark
        ISCO r/M          5.999999999999999       6                       8.882e-16
        ISCO r/r_s        2.9999999999999996      3
        foton r/M         3.0                     3                       0 (bit bit)
        foton r/r_s       1.5                     1.5
        b_krit/M          5.196152422706632       3√3                     0 (bit bit)
        r_ISCO/r_foton    1.9999999999999998      2
        μ_c               0.038520898328784976    (1−√69/9)/2             1.8242e-09
        apsidal üs        -0.4999999923696353     -1/2                    7.6304e-09

    PLANIN `b_krit` SAYISI SON BİTTE YANLIŞ: taban bloğu D
    `5.196152422706631` yazıyor; hem ölçülen değer hem `3*math.sqrt(3)` bu
    makinede `...632`dir, yani plan **1 ulp düşük**.
    """
    assert isco_radius() == 5.999999999999999
    assert isco_radius() / 2.0 == 2.9999999999999996
    assert photon_sphere() == 3.0
    assert photon_sphere() / 2.0 == 1.5
    assert critical_impact_parameter() == 5.196152422706632
    assert critical_impact_parameter() == 3.0 * math.sqrt(3.0)
    assert critical_impact_parameter() - 3.0 * math.sqrt(3.0) == 0.0
    assert isco_radius() / photon_sphere() == 1.9999999999999998

    mu = lagrange_mu_c()
    kapali = (1.0 - math.sqrt(69.0) / 9.0) / 2.0
    assert mu == 0.038520898328784976
    assert abs(mu - kapali) == pytest.approx(1.8242e-09, rel=1e-3, abs=0.0)

    us = apsidal_exponent()
    assert us == -0.4999999923696353
    assert abs(us + 0.5) == pytest.approx(7.6304e-09, rel=1e-3, abs=0.0)

    # PLANIN SAYISI 1 ULP DÜŞÜK:
    assert 5.196152422706631 != 3.0 * math.sqrt(3.0)
    assert math.nextafter(5.196152422706631, 10.0) == 3.0 * math.sqrt(3.0)


@pytest.mark.parametrize("gamma, beta, isco, foton, bkrit", [
    (1.0, 1.0, 6.000000000, 3.000000000, 5.196152423),
    (1.0, 0.0, 8.376805373, 4.000000000, 6.531972647),
    (2.0, 1.0, 8.376805373, 4.000000000, 6.531972647),
    (1.0, 0.8, 6.563356715, 3.246424920, 5.519212160),
    (1.1, 1.0, 6.290203091, 3.127882060, 5.363110869),
])
def test_UCU_DE_YALNIZ_BETA_EKSI_GAMMA_YA_BAGLI(gamma, beta, isco, foton, bkrit):
    """SINIF: KAPALI-FORM TÜREV — ve bir PPN ölçümü.

    `A(u) = 1 − 2Mu + 2(β−γ)(Mu)²` olduğu için üç sayı da yalnız `β−γ`
    farkına bağlıdır. İkinci ve üçüncü satır bunu gösteriyor: (γ,β) = (1,0)
    ile (2,1) — ikisi de `β−γ = −1` — BİREBİR aynı üç sayıyı veriyor.
    """
    t = Theory(gamma=gamma, beta=beta)
    assert isco_radius(theory=t) == pytest.approx(isco, abs=5e-9)
    assert photon_sphere(theory=t) == pytest.approx(foton, abs=5e-9)
    assert critical_impact_parameter(theory=t) == pytest.approx(bkrit, abs=5e-9)


def test_BETA_BUYUK_GAMMA_DA_UCU_DE_REDDEDILIYOR():
    """SINIF: KAPI — ve bir DÜRÜST NİTELİK beyanı.

    `β > γ`'da 1PN'de kesilmiş `A(u)` bir kara delik tarif ETMİYOR: `dV/du`nun
    (0, 1/(2M)] aralığında kökü YOK, yani foton küresi ve `b_krit` yok. ISCO
    denkleminin kökü ise VAR ama fiziksel DEĞİL — ölçüldü: (γ,β) = (0,1)'de
    1.374183181, (1,3)'te 2.331811168.

    İnanılır görünen bir sayı sessizce döndürmek yerine `InvalidInputError`
    atılıyor. (Bu depoda en pahalı arıza türü tam olarak "inanılır görünen
    sessiz sayı"dır.)
    """
    for t in (Theory(gamma=0.0, beta=1.0), Theory(gamma=1.0, beta=3.0)):
        for fn in (isco_radius, photon_sphere, critical_impact_parameter):
            with pytest.raises(InvalidInputError, match="beta > gamma"):
                fn(theory=t)


def test_APSIDAL_FITININ_KESISIMI_BEDAVA_SINAV_ve_OLCEK_KORLUGU():
    """SINIF: ŞEMA ÇIPASI — bir US FITININ GÖRMEDİĞİ yüz ölçülüyor.

    `log(Φ/π) = s·log(3+n) + c` doğrusunda kapalı form `c = 0` der ve bu bilgi
    EĞİMDE KULLANILMAZ, yani sınav bedavadır. Ölçülen `c` =
    **-1.578485733121196e-08**.

    NEDEN KAPI OLARAK KONDUĞU BURADA ÖLÇÜLÜYOR: gövdeden `math.pi` çarpanı
    silinirse `log(Φ/π)` sabit `-log(π)` kadar kayar ve EĞİM **BİT BİT
    DEĞİŞMEZ** (-0.49999999236963555 ↔ -0.4999999923696353, son bitte).
    Yani o mutant eğimi okuyan HER testten sağ çıkardı. Kesişim onu
    -1.1447299016342576 = `-log(π)` ile yakalıyor.

    *Bir üs fit'i ölçeğe kördür.*
    """
    def fit(pi_carpani):
        xs, ys = [], []
        for n in APSIDAL_GRID:
            k = rc = 1.0
            L2 = k * rc ** (n + 3.0)

            def Veff(r):
                return k * r ** (n + 1.0) / (n + 1.0) + L2 / (2.0 * r * r)
            h = 1e-4
            d2 = (Veff(rc + h) - 2.0 * Veff(rc) + Veff(rc - h)) / (h * h)
            Phi = pi_carpani * (math.sqrt(L2) / rc ** 2) / math.sqrt(d2)
            xs.append(math.log(3.0 + n))
            ys.append(math.log(Phi / math.pi))
        egim, kesisim = np.polyfit(np.array(xs), np.array(ys), 1)
        return float(egim), float(kesisim)

    egim, kesisim = fit(math.pi)
    assert egim == apsidal_exponent()          # sevk edilen gövdeyle aynı
    assert kesisim == -1.578485733121196e-08
    assert abs(kesisim) < 1e-6                 # sevk edilen kapının eşiği

    mutant_egim, mutant_kesisim = fit(1.0)     # `math.pi` SİLİNMİŞ
    assert mutant_egim == pytest.approx(egim, rel=1e-15)
    assert abs(mutant_egim - egim) < 1e-15     # EĞİM kör
    assert mutant_kesisim == -1.1447299016342576
    assert mutant_kesisim == pytest.approx(-math.log(math.pi), rel=2e-8)
    assert abs(mutant_kesisim) > 1e-6          # KAPI onu yakalar


def test_APSIDAL_IZGARASININ_KAPILARI():
    """SINIF: KAPI. `n = −1` (log potansiyel) 0/0'dır; `n ≤ −3`'te dairesel
    yörünge kararsızdır (`Φ = π/√(3+n)` tanımsız)."""
    with pytest.raises(InvalidInputError, match="logaritmik"):
        apsidal_exponent(force_exponents=(-1.0, -2.0))
    with pytest.raises(InvalidInputError, match="n > -3"):
        apsidal_exponent(force_exponents=(-3.5, -2.0))
    with pytest.raises(InvalidInputError, match="en az iki"):
        apsidal_exponent(force_exponents=(-2.0,))


def test_STATIK_YOL_DINAMIK_YOLDAN_HEM_HIZLI_HEM_HASSAS():
    """SINIF: KAPALI-FORM TÜREV — ve spec §9 kriter 6'nın statik/dinamik oranı.

    Dinamik yollar SEVK EDİLMİYOR (spec §7.5); aşağıdaki gövdeler bu dosyanın
    kendi kopyalarıdır ve yöntemleri açıkça yazılı:

      * dinamik `b_krit`: dönüş noktasının VARLIĞINI 400 noktalı bir ızgarada
        tarar, `b`'yi 44 turda ikiye böler.
      * dinamik ISCO: dairesel yörüngeyi 1e-8 bozar, `φ ∈ [0, 500]` boyunca
        ODE ile izler, eşik aşımında KARARSIZ sayar, 44 turda ikiye böler.

    ÖLÇÜLEN (aynı makine, aynı koşu):

        sayı     statik                    dinamik                    oran
        ISCO      4.68 µs, hata 8.882e-16   189.3 ms, hata 2.333e-03   40418×
        b_krit   13.44 µs, hata 0.000e+00     6.50 ms, hata 2.395e-05     483×

    PLANIN 844× / 11 180× SAYILARI ÜRETİLEMEDİ. Dinamik ISCO'nun hatası
    YÖNTEMDEN gelir ve KAÇINILMAZDIR: radyal salınım periyodu marjinal
    noktada IRAKSAR, yani sonlu bir `φ` penceresi eşiği sistematik olarak
    kaçırır. Bu test ZAMAN değil HASSASİYET kilitler (zaman makineye bağlı).
    """
    assert _dinamik_b_krit() == 5.196176371940538
    assert abs(_dinamik_b_krit() - 3 * math.sqrt(3)) == pytest.approx(
        2.395e-05, rel=1e-3)
    assert critical_impact_parameter() - 3 * math.sqrt(3) == 0.0

    dinamik_isco = _dinamik_isco()
    assert dinamik_isco == pytest.approx(5.997666717104316, rel=1e-9)
    assert abs(dinamik_isco - 6.0) == pytest.approx(2.333e-03, rel=1e-3)
    assert abs(isco_radius() - 6.0) < 1e-15
    # Dinamik ISCO statikten 1e12 kat DAHA AZ hassas.
    assert abs(dinamik_isco - 6.0) / abs(isco_radius() - 6.0) > 1e12


# ===========================================================================
# C. GÜÇLÜ ALAN NOKTASI — dinamik `b_krit` YERİNE (Step 4)
# ===========================================================================
def test_GUCLU_ALAN_NOKTASI_ODE_YALNIZ_KUSURLARI_YAKALIYOR():
    """SINIF: FİZİK ÇIPASI (güçlü alan) — ve spec §7.5'in kararının ölçümü.

    Statik yolun YAKALAYAMADIĞI bir kusur var: boş jeodezik çekirdeğine
    (`N(u)`) eklenen `ε·M²·u⁴` terimi. Statik `b_krit` `A(u)`'dan okunduğu
    için o terimi HİÇ GÖRMEZ.

    ÖLÇÜLEN (GR, M = 1; güçlü alan `b = 1.1·b_krit = 5.715767664977296`,
    temiz `δ = 2.080533685868903` rad = 0.662254 π):

        ε        statik b_krit   güçlü alan Δδ            dinamik b_krit Δb
        1e-2     0.0e+00 (KÖR)   1.2903e-02  2.91e13 ulp  8.6199e-03  3.9e+02 σ
        1e-4     0.0e+00 (KÖR)   1.2802e-04  2.88e11 ulp  8.7214e-05  4.0e+00 σ
        1e-8     0.0e+00 (KÖR)   1.2801e-08  2.88e07 ulp  8.7216e-09  4.0e-04 σ
        1e-12    0.0e+00 (KÖR)   1.2816e-12  2.89e03 ulp  1.0232e-12  4.7e-08 σ

    `ulp` = güçlü alan δ'sının son biti (4.441e-16); `σ` = dinamik `b_krit`in
    KENDİ ızgara gürültüsü (400 → 800 nokta: **2.201e-05**).

    HÜKÜM: güçlü alan noktası kusuru **ε ≈ 3.5e-16**'ya kadar görüyor, dinamik
    `b_krit` yalnızca **ε ≳ 2.5e-5**'e kadar — yani ~1e11 kat daha duyarlı, ve
    ölçülen maliyeti 0.0386 ms ↔ 6.50 ms, yani **168 kat ucuz**. Bu, spec
    §7.5'in "dinamik `b_krit` o boşluğu doldurmanın EN KÖTÜ yolu" hükmünün bu
    depodaki karşılığıdır — planın 1.35 ms / 121.5 ms ve
    1.82e+09 σ / 2.11e+08 σ sayıları ÜRETİLEMEDİ, ama HÜKÜM aynı yönde çıktı.
    """
    b = 1.1 * critical_impact_parameter()
    assert b == 5.715767664977296
    temiz = deflection_angle(theory=GR, M=1.0, b=b)
    assert temiz == 2.080533685868903
    assert temiz / math.pi == pytest.approx(0.662254, abs=1e-6)
    assert _guclu_alan(0.0) == temiz          # kopya gövde BİT BİT aynı

    ulp = math.ulp(temiz)
    assert ulp == 4.440892098500626e-16
    sigma = abs(_dinamik_b_krit(n=800) - _dinamik_b_krit(n=400))
    assert sigma == pytest.approx(2.201e-05, rel=1e-3)

    tablo = {1e-2: (1.2903e-02, 8.6199e-03), 1e-4: (1.2802e-04, 8.7214e-05),
             1e-8: (1.2801e-08, 8.7216e-09), 1e-12: (1.2816e-12, 1.0232e-12)}
    temiz_dinamik = _dinamik_b_krit()
    for eps, (d_guclu, d_dinamik) in tablo.items():
        assert abs(_guclu_alan(eps) - temiz) == pytest.approx(
            d_guclu, rel=1e-3, abs=0.0)
        assert abs(_dinamik_b_krit(eps=eps) - temiz_dinamik) == pytest.approx(
            d_dinamik, rel=1e-3, abs=0.0)
        # STATİK YOL BEŞİNDE DE KÖR:
        assert critical_impact_parameter() - 3 * math.sqrt(3) == 0.0

    # Duyarlılık eşikleri: güçlü alan ulp'e, dinamik b_krit kendi σ'sına kadar.
    assert tablo[1e-12][0] / ulp > 1e3          # ε = 1e-12 hâlâ GÖRÜNÜYOR
    assert tablo[1e-8][1] / sigma < 1e-3        # dinamik yol ε = 1e-8'i KAÇIRIR
    assert (tablo[1e-12][0] / ulp) / (tablo[1e-12][1] / sigma) > 1e10


# ---------------------------------------------------------------------------
# Bu dosyanın kendi kopyaları — SEVK EDİLMEYEN yollar (spec §7.5)
# ---------------------------------------------------------------------------
def _guclu_alan(eps: float, M: float = 1.0) -> float:
    """`b = 1.1 b_krit`te bükülme; `N(u)`ya `eps*M^2*u^4` eklenebilir."""
    b = 1.1 * critical_impact_parameter(M=M)
    pot = null_potentials(theory=GR, M=M, gauge="schwarzschild")
    a0, a1, a2 = pot.time_poly
    N = (1.0 / b**2, 0.0, -a0, -a1, -a2 + eps * M * M)
    f = lambda u: evaluate_poly(N, u)
    ust = 1.05 / b
    while f(ust) > 0.0:
        ust *= 1.05
    u0 = brentq(f, 0.0, ust, xtol=1e-18, rtol=8.9e-16, maxiter=200)
    q3 = -N[4]
    q2 = u0 * q3 - N[3]
    q1 = u0 * q2 - N[2]
    q0 = u0 * q1 - N[1]

    def integrand(t):
        s = math.sin(t)
        u = u0 * s
        D = evaluate_poly(pot.time_poly, u) / evaluate_poly(pot.damp_poly, u)
        return math.sqrt(u0 * (1 + s) * D / evaluate_poly((q0, q1, q2, q3), u))

    v, _ = quad(integrand, 0.0, math.pi / 2, epsabs=1e-14, epsrel=1e-13,
                limit=400)
    return 2.0 * v - math.pi


def _dinamik_b_krit(*, eps: float = 0.0, M: float = 1.0, tur: int = 44,
                    n: int = 400) -> float:
    """Dönüş noktasının VARLIĞINI tarayarak `b_krit` — SEVK EDİLMİYOR."""
    pot = null_potentials(theory=GR, M=M, gauge="schwarzschild")
    a0, a1, a2 = pot.time_poly
    us = np.linspace(1e-12, 0.499 / M, n)

    def kacar(b):
        N = (1.0 / b**2, 0.0, -a0, -a1, -a2 + eps * M * M)
        return any(evaluate_poly(N, u) <= 0.0 for u in us)

    lo, hi = 3.0 * M, 12.0 * M
    for _ in range(tur):
        orta = 0.5 * (lo + hi)
        if kacar(orta):
            hi = orta
        else:
            lo = orta
    return 0.5 * (lo + hi)


def _dinamik_isco(*, M: float = 1.0, tur: int = 44, PHI: float = 500.0,
                  delta: float = 1e-8) -> float:
    """Bozulma BÜYÜYOR mu diye ODE ile bakarak ISCO — SEVK EDİLMİYOR."""
    def kararli(u_c):
        L2 = 2.0 * M / (2.0 * u_c - 6.0 * M * u_c * u_c)
        esik = u_c * (1.0 + 1e-4)

        def rhs(phi, y):
            return (y[1], M / L2 + 3.0 * M * y[0] * y[0] - y[0])

        def kac(phi, y):
            return y[0] - esik
        kac.terminal = True
        kac.direction = 1.0
        sol = solve_ivp(rhs, (0.0, PHI), (u_c * (1.0 + delta), 0.0),
                        method="DOP853", rtol=1e-10, atol=1e-16, events=kac)
        return len(sol.t_events[0]) == 0

    lo, hi = 0.01 / M, 0.24 / M
    for _ in range(tur):
        orta = 0.5 * (lo + hi)
        if kararli(orta):
            lo = orta
        else:
            hi = orta
    return 1.0 / (0.5 * (lo + hi) * M)
