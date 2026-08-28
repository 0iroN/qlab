"""Perihel katsayısı ve İKİ AYARLI GERÇEK ÇAPRAZ KONTROL — Modül 6 Task 3.

PENCERE — bu dosyadaki HER sayı aşağıdaki tek pencerede, SEVK EDİLEN kodla
ölçüldü; hiçbiri plandan ya da spec'ten devralınmadı (*pencere adlandırılmazsa
hüküm dışına taşınır*, bu depoda on birden fazla kez çıktı):

- Birim seçimi `G = c = 1`; `M = 1.0`, `e = 0.3` (aksi yazılmadıkça).
- `p = a(1-e²)`; dönüş noktaları `u1 = (1-e)/p`, `u2 = (1+e)/p`.
- `psi` ikamesi, `scipy.integrate.quad` `epsabs=1e-14`, `epsrel=1e-13`,
  `limit=400`; Richardson çifti (3e5, 1e6) — `geodesic.py`'nin sevk edilen
  düğmeleri, burada YENİDEN yazılmıyor, oradan geliyor.
- `second_order_coefficient` TEK BASAMAKLIDIR ve basamağı `p = 3e5`'tir
  (ölçülmüş optimum — aşağıda merdiveni var).

BU DOSYANIN TAŞIYICI İDDİASI ŞUDUR: *iki ayar aynı sayıyı veriyor.* Ve o iddia
ancak iki ayarın **ORTAK DÜĞME TAŞIMADIĞI** gösterilirse bir çapraz kontroldür
— spec §7.3'ün dördüncü maddesi tam olarak bunun tersini ölçmüştü (kareleme ↔
`closed_form_coefficient`, ikisi de `Theory.lam`i okuyor, `lam` mutantına KÖR).
Burada üç ayrı ölçüm var ve üçü de bu dosyada koşuyor:

  (1) KAYDEDEN VEKİL, dokunulan ad kümelerini SAYIYOR: `{"lam"}` ve
      `{"gamma", "beta"}`, kesişim BOŞ.
  (2) `lam` FORMÜLÜ BOZUK bir kuram nesnesi iki ayara verildiğinde fark
      1.5e-09'dan 6.28'e çıkıyor — 4.2 MİLYAR kat.
  (3) VE ÇAPRAZ KONTROLÜN GÖRMEDİĞİ YÜZEY DE ÖLÇÜLÜP ADIYLA YAZILI: iki ayar
      `_donus_noktalari`yı, `quad` düğmelerini, `psi` ikamesini ve Richardson
      çiftini PAYLAŞIR; bunlardan birini bozan mutant iki tarafı BİRLİKTE
      kaydırır ve fark BÜYÜMEZ, KÜÇÜLÜR.

`slow` İŞARETİ YOK, ölçülerek: 116 test **0.63 s**'de koşuyor ve tek bir
testin süresi 5 ms'nin altında (`--durations` hiçbirini gösteremiyor). En
pahalı gövde `K`nın `p` merdiveni: 7 × 18 = **126** tek basamaklı kareleme.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, dataclass

import pytest

import qlab.engine.grav.geodesic as geodesic_modulu
from qlab.engine.grav.geodesic import GAUGES, perihelion_coefficient
from qlab.engine.grav.metric import (
    Theory,
    closed_form_coefficient,
    isotropic_orbit,
    isotropic_w,
)
from qlab.errors import InvalidInputError
from qlab.observable.grav_obs import newton_noise_floor, second_order_coefficient

ALTI_PI = 6.0 * math.pi

# Task 1'in beş PPN noktası — (gamma, beta, lam çarpanı). ÇARPANLAR ELLE
# YAZILI, `Theory.lam`den okunmuyor: ortak düğme tam oradan doğardı.
PPN_NOKTALARI = [
    (1.0, 1.0, 3.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 4.0),
    (2.0, 1.0, 5.0),
    (1.0, 3.0, 1.0),
]

E_TARAMASI = (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)


class Kaydeden:
    """`Theory`yi saran, OKUNAN ad kümesini sayan vekil.

    SINIRI BİLİNİYOR (spec §5): vekil çekirdeğin bir alanı OKUMADIĞINI
    kanıtlar; iki tüketicinin ortak bir TÜREVDEN gelmediğini kanıtlamaz.
    Buradaki iddia BİRİNCİ türdendir — izotropik yol `Theory.lam`e HİÇ
    dokunmuyor — dolayısıyla vekil geçerlidir.
    """

    def __init__(self, hedef: Theory) -> None:
        self._hedef = hedef
        self.dokunulan: set[str] = set()

    def __getattr__(self, ad: str):
        self.dokunulan.add(ad)
        return getattr(self._hedef, ad)


@dataclass(frozen=True)
class BozukLamTheory:
    """`Theory`nin `lam` formülü BOZULMUŞ ikizi: `2 + gamma - beta`.

    Bir mutasyon turunu SÜİTİN İÇİNE taşır. `Theory`nin kendisine dokunulmaz
    (donmuş bir dataclass'ı yamamak testi kaçak yapardı); aynı arayüzü taşıyan
    ikinci bir nesne verilir ve iki ayarın ona NE KADAR FARKLI tepki verdiği
    ölçülür.
    """

    gamma: float = 1.0
    beta: float = 1.0

    @property
    def lam(self) -> float:
        return 2.0 + self.gamma - self.beta


# ---------------------------------------------------------------------------
# Step 1 — ÇIPA
# ---------------------------------------------------------------------------
def test_PERIHEL_KATSAYISI_ALTI_PI_ve_ON_HANE():
    """SINIF: FİZİK ÇIPASI — statik kökü YOK, yörüngeyi çözmek ZORUNLU.

    (Karşılaştır: ISCO'nun statik kökü VAR ve dinamik yol onu 844 kat yavaş ve
    DAHA AZ HASSAS buluyor — o "kapalı-form türev" rafında.)

    PLANIN SAYISI ÇÜRÜMÜŞTÜ ve Task 1'in düzelttiği değer BURADA YENİDEN
    ÜRETİLDİ (Schwarzschild ayarı, GR, M = 1, e = 0.3, Richardson (3e5, 1e6)):

        C   = 18.84955591968662
        6pi = 18.84955592153876
        fark −1.852140e-09  ->  10.01 DÜRÜST HANE

    Planın taban bloğu A'sı bu sayıyı **18.849555919686622** diye yazıyor;
    o dizge BAŞKA BİR FLOAT'tır — `float("18.849555919686622")` sevk edilen
    değere EŞİT DEĞİL (ölçüldü, `!=`). Doğrusu 16 haneli `repr`dir ve o da
    burada aşağıda BİT BİT kilitleniyor.
    """
    C = perihelion_coefficient(M=1.0, e=0.3)
    assert C == 18.84955591968662
    assert C - ALTI_PI == pytest.approx(-1.852140e-09, rel=1e-4)
    hane = -math.log10(abs(C - ALTI_PI) / ALTI_PI)
    assert hane == pytest.approx(10.01, abs=0.01)
    # Planın 17 haneli dizgesi BAŞKA bir float:
    assert C != float("18.849555919686622")


def test_IZOTROPIK_AYAR_DA_ALTI_PI_VERIYOR_ve_GR_DE_DAHA_ISABETLI():
    """SINIF: FİZİK ÇIPASI — ikinci ayar, ikinci gövde.

    ÖLÇÜLDÜ: izotropik ayar GR'de 18.849555921564487, `6pi`den **+2.572875e-11**,
    yani **11.86 dürüst hane** — Schwarzschild ayarının 10.01'inden İYİ.

    AMA BU BİR ÜSTÜNLÜK İDDİASI DEĞİL, GR'ye ÖZGÜ BİR ŞANSTIR ve bunu
    söyleyebilmemizin sebebi ölçmüş olmamız: beş PPN noktasında izotropik
    yolun kapalı forma bağıl uzaklığı 1.3650e-12 … 6.3209e-10 arasında geziyor,
    yani GR noktası (1.365e-12) ailenin EN İYİSİ. Bir tek noktadan "bu yol
    daha doğru" hükmü çıkarmak, pencereyi hükmün dışına taşımak olurdu.
    """
    C = perihelion_coefficient(M=1.0, e=0.3, gauge="isotropic")
    assert C == 18.849555921564487
    assert C - ALTI_PI == pytest.approx(2.572875e-11, rel=1e-4)
    assert -math.log10(abs(C - ALTI_PI) / ALTI_PI) == pytest.approx(11.86, abs=0.01)


@pytest.mark.parametrize("e", E_TARAMASI)
@pytest.mark.parametrize("gauge", GAUGES)
def test_C_DISMERKEZLIKTEN_BAGIMSIZ_IKI_AYARDA_DA(gauge, e):
    """`C` bir yörünge parametresi DEĞİL bir KURAM sayısıdır: `e`yi görmez.

    ÖLÇÜLDÜ (GR, M = 1, yedi `e`, Richardson (3e5, 1e6)) — yayılımlar:
    Schwarzschild **2.4615e-09**, izotropik **4.0729e-09**.

    VE İKİSİ DE DIŞDEĞERLEMENİN KENDİ TABANI MERTEBESİNDEDİR (kareleme tabanı
    1.594187e-09, izotropik tabanı 2.346903e-09). Yani bu test "e'nin etkisi
    YOK" demiyor, "e'nin etkisi TABANIN ALTINDA" diyor — ayrımı yazmazsak
    hüküm pencereden taşar.
    """
    C = perihelion_coefficient(M=1.0, e=e, gauge=gauge)
    assert C == pytest.approx(ALTI_PI, abs=5e-09)


def test_C_e_YAYILIMI_TABANIN_MERTEBESINDE():
    """Yukarıdaki testin SAYISAL hâli — yayılımlar ADIYLA kilitleniyor."""
    for gauge, beklenen in (("schwarzschild", 2.4615e-09), ("isotropic", 4.0729e-09)):
        degerler = [perihelion_coefficient(M=1.0, e=e, gauge=gauge) for e in E_TARAMASI]
        assert max(degerler) - min(degerler) == pytest.approx(beklenen, rel=1e-3)


@pytest.mark.parametrize("lam", [-3.0, -1.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
def test_IZOTROPIK_KARELEME_2PI_LAMBDAYI_URETIYOR_ISLEVSEL_SINAV(lam):
    """İZOTROPİK yol `lam` ∈ [−3, 12] boyunca `C = 2*pi*lam` veriyor — NEGATİF
    dahil. ÖLÇÜLDÜ: dokuz noktada EN KÖTÜ bağıl fark **5.6945e-10**.

    Bu, spec §3.1'in "işlevsel sınav"ının İKİNCİ, BAĞIMSIZ kod yolundan
    okunuşudur: `lam` burada bir GİRDİ DEĞİL, `(gamma, beta) = ((lam-1)/2, 1)`
    üzerinden modele giren bir SONUÇTUR ve izotropik yol `lam`i HİÇ görmez.
    Ön ölçümün tautolojisi (`lam` girdi olarak hesaplanıp geri okunuyordu) tam
    burada kapanır.
    """
    t = Theory(gamma=(lam - 1.0) / 2.0, beta=1.0)
    C = perihelion_coefficient(theory=t, M=1.0, e=0.3, gauge="isotropic")
    assert C == pytest.approx(2.0 * math.pi * lam, rel=1e-9)


# ---------------------------------------------------------------------------
# Step 2 — GERÇEK ÇAPRAZ KONTROL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gamma, beta, kat", PPN_NOKTALARI)
def test_IKI_AYAR_AYNI_SAYIYI_VERIYOR(gamma, beta, kat):
    """SINIF: FİZİK ÇIPASI, ve BU MODÜLÜN GERÇEK ÇAPRAZ KONTROLÜ.

    Schwarzschild yolu `u`da DOĞRUSAL bir `w` ile (`u''+u = M/L² + lam*M*u²`
    denkleminin enerji integralinden), izotropik PPN yolu 2x2 Cramer + kuartik
    deflasyonla çalışır — TAMAMEN FARKLI MEKANİZMA VE AYAR.

    ÖLÇÜLDÜ, beş PPN noktası, Richardson (3e5, 1e6), e = 0.3, M = 1:
    |C_schw − C_izo| = **1.4972e-09 … 3.3116e-09**. (Planın taban bloğu
    2.3e-10 … 4.1e-09 diyordu; ALT UCU BU PENCEREDE ÜRETİLEMEDİ — ölçülen alt
    uç 6.5 kat büyük. Üst uç plana uyuyor.)

    KARELEME ↔ ODE ÇAPRAZ KONTROLÜ DEĞİL, çünkü O ÇÖKTÜ (spec §7.3): sekiz
    mutantlı matriste tek başına yakaladığı mutant 0 idi, ve sebebi ölçüldü —
    `|C_kar − C_ode| / |6*pi − C_ode| = 1.000`, yani kareleme çapraz kontrolde
    `6*pi`nin YERİNE GEÇİYOR ve `6*pi` zaten bedava. Plan F'nin deseni burada
    geçmez: Plan F'de bağımsız referans YOKTU, burada VAR.

    ALT SINIR (`> 1e-10`) TAŞIYICIDIR, DEKOR DEĞİL: iki ayarı ORTAK BİR KOD
    YOLUNA bağlayan mutantta fark BİT BİT 0 olur ve yalnız bu satır ölür.
    Mutasyonla ölçüldü (M3: `_DPHI["isotropic"] = _dphi_schwarzschild`) —
    üst sınır tek başına o mutantı KAÇIRIRDI.

    VE O ALT SINIRIN PENCERESİ `e = 0.3`TÜR, HER `e` DEĞİL: `e` taramasında
    fark 3.8067e-11'e kadar iniyor (e = 0.95) ve orada `> 1e-10` GEÇMEZDİ.
    Bu bir kusur değil bir pencere beyanıdır — fark iki borunun yuvarlama
    gürültüsünün farkıdır, sabit değildir, ve tablosu bir alttaki testte.
    """
    t = Theory(gamma=gamma, beta=beta)
    a = perihelion_coefficient(theory=t, M=1.0, e=0.3)
    b = perihelion_coefficient(theory=t, M=1.0, e=0.3, gauge="isotropic")
    assert abs(a - b) < 4e-09
    assert abs(a - b) > 1e-10       # PENCERE İKİ TARAFLI: sıfır da olmuyor
    # ve ikisi de ELLE YAZILMIŞ çarpanın kapalı formunu veriyor:
    assert a == pytest.approx(2.0 * math.pi * kat, rel=1e-9)
    assert b == pytest.approx(2.0 * math.pi * kat, rel=1e-9)


def test_IKI_AYAR_FARK_PENCERESI_ADIYLA():
    """Beş PPN noktasının farkları TEK BİR PENCEREDE toplanıyor: ölçülen
    **1.4972e-09 … 3.3116e-09**. `e` taramasında (GR, yedi `e`) pencere
    **3.8067e-11 … 3.7050e-09**'a genişliyor — yani "fark" bir SABİT DEĞİL,
    iki borunun yuvarlama gürültüsünün farkıdır ve `e` ile zıplar.
    """
    ppn = [
        abs(perihelion_coefficient(theory=Theory(gamma=g, beta=b), M=1.0, e=0.3)
            - perihelion_coefficient(theory=Theory(gamma=g, beta=b), M=1.0, e=0.3,
                                     gauge="isotropic"))
        for g, b, _ in PPN_NOKTALARI
    ]
    assert min(ppn) == pytest.approx(1.4972e-09, rel=1e-3)
    assert max(ppn) == pytest.approx(3.3116e-09, rel=1e-3)

    e_farklari = [
        abs(perihelion_coefficient(M=1.0, e=e)
            - perihelion_coefficient(M=1.0, e=e, gauge="isotropic"))
        for e in E_TARAMASI
    ]
    assert min(e_farklari) == pytest.approx(3.8067e-11, rel=1e-3)
    assert max(e_farklari) == pytest.approx(3.7050e-09, rel=1e-3)


def test_ORTAK_DUGME_YOK_dokunulan_ad_kumesi_SAYILIYOR():
    """ÇAPRAZ KONTROLÜN ÖN KOŞULU — ve bu VARSAYILMADI, SAYILDI.

    Spec §7.3'ün dördüncü maddesi: *ortak düğme çapraz kontrolü KÖRLEŞTİRİR.*
    Task 1'de bu ölçüldü — kareleme ile `closed_form_coefficient` ikisi de
    `Theory.lam`i okuyor, `lam` mutantı ikisini BİRLİKTE kaydırıyor ve
    karşılaştırma susuyor.

    Burada aynı sınav İKİ AYARA uygulanıyor ve GEÇİYOR: Schwarzschild borusu
    `Theory`nin YALNIZ `lam`ini, izotropik boru YALNIZ `gamma` ve `beta`sını
    okuyor. KESİŞİM BOŞ.
    """
    kumeler = {}
    for gauge in GAUGES:
        vekil = Kaydeden(Theory())
        perihelion_coefficient(theory=vekil, M=1.0, e=0.3, gauge=gauge)
        kumeler[gauge] = vekil.dokunulan

    assert kumeler["schwarzschild"] == {"lam"}
    assert kumeler["isotropic"] == {"gamma", "beta"}
    assert kumeler["schwarzschild"] & kumeler["isotropic"] == set()
    assert "lam" not in kumeler["isotropic"]


@pytest.mark.parametrize("gamma, beta, kat", PPN_NOKTALARI)
def test_LAM_MUTANTI_CAPRAZ_KONTROLU_OLDURUYOR_ve_DUYARLILIK_OLCULDU(gamma, beta, kat):
    """ÇAPRAZ KONTROLÜN DİŞLERİ VAR MI — bir mutasyon turu SÜİTİN İÇİNDE.

    `lam` formülü `2+2*gamma-beta` yerine `2+gamma-beta` yazılmış bir kuram
    nesnesi iki ayara da verilir. Schwarzschild yolu ONU okur ve kayar;
    izotropik yol `gamma` ile `beta`yı okuduğu için KIPIRDAMAZ.

    ÖLÇÜLDÜ (M = 1, e = 0.3, Richardson (3e5, 1e6)):

        (gamma, beta)   dogru lam   bozuk lam   |C_schw − C_izo|
        (1, 1)             3            2         6.2832e+00
        (0, 1)             1            1         1.4972e-09   <-- KÖR NOKTA
        (1, 0)             4            3         6.2832e+00
        (2, 1)             5            3         1.2566e+01
        (1, 3)             1            0         6.2832e+00

    Temiz farkın 1.4972e-09 … 3.3116e-09 olduğu düşünülürse duyarlılık
    **4.2 milyar kat**tır. VE (0, 1) NOKTASI DÜRÜST BİR OLUMSUZDUR: orada
    `2+2*0-1 = 1` ile `2+0-1 = 1` ÇAKIŞIYOR, yani o tek nokta bu mutanta
    yapısal olarak kördür. Bir çapraz kontrolün gücü, ailenin EN ZAYIF
    üyesinden değil ailenin TAMAMINDAN okunur — bu yüzden aile beş üyeli.
    """
    bozuk = BozukLamTheory(gamma=gamma, beta=beta)
    a = perihelion_coefficient(theory=bozuk, M=1.0, e=0.3)
    b = perihelion_coefficient(theory=bozuk, M=1.0, e=0.3, gauge="isotropic")
    if bozuk.lam == kat:
        assert abs(a - b) < 4e-09            # ölçülen kör nokta: (gamma, beta) = (0, 1)
    else:
        assert abs(a - b) > 6.0              # ölçülen: 6.2832 ya da 12.566
    # ve izotropik taraf HER durumda doğru cevabı vermeye devam ediyor:
    assert b == pytest.approx(2.0 * math.pi * kat, rel=1e-9)


def test_CAPRAZ_KONTROLUN_GORMEDIGI_YUZEY_ORTAK_PENCEREDIR(monkeypatch):
    """DÜRÜST OLUMSUZ — ve bu testin işi çapraz kontrolü ÖVMEK DEĞİL SINIRLAMAK.

    İki ayar `Theory` düğmelerini paylaşmıyor ama PENCEREYİ paylaşıyor:
    `_donus_noktalari`, `quad` düğmeleri, `psi` ikamesi, Richardson çifti.
    Bunlardan birini bozan bir mutant iki tarafı BİRLİKTE kaydırır.

    ÖLÇÜLDÜ, mutant `u2 = (1 + 1.001*e)/p` (GR, e = 0.3, Richardson (3e5,1e6)):

        temiz   schw 18.84955591968662   izo 18.849555921564487  |fark| 1.878e-09
        mutant  schw 18.852383354455476  izo 18.852383352831374  |fark| 1.624e-09

    Yani fark BÜYÜMÜYOR, KÜÇÜLÜYOR — çapraz kontrol TAMAMEN sessiz. Mutantı
    öldüren şey `6*pi` çıpasıdır (ikisi de 2.827e-03 sapıyor). Bu, "iki ayar
    her şeyi yakalar" cümlesinin ÖLÇÜLMÜŞ sınırıdır.
    """
    def mutant(p: float, e: float) -> tuple[float, float]:
        return (1.0 - e) / p, (1.0 + 1.001 * e) / p

    monkeypatch.setattr(geodesic_modulu, "_donus_noktalari", mutant)
    a = perihelion_coefficient(M=1.0, e=0.3)
    b = perihelion_coefficient(M=1.0, e=0.3, gauge="isotropic")

    assert abs(a - b) == pytest.approx(1.624e-09, rel=1e-3)
    assert abs(a - b) < 4e-09                       # ÇAPRAZ KONTROL SUSUYOR
    assert abs(a - ALTI_PI) == pytest.approx(2.827e-03, rel=1e-3)   # çıpa KONUŞUYOR
    assert abs(b - ALTI_PI) == pytest.approx(2.827e-03, rel=1e-3)


# ---------------------------------------------------------------------------
# İZOTROPİK BORUDA `lambda = 0` — Task 2'nin BEYAN ETTİĞİ KUSURU DARALTIYOR
# ---------------------------------------------------------------------------
def test_IZOTROPIK_BORU_LAMBDA_SIFIR_CIZGISINDE_DEJENERE_DEGIL():
    """TASK 2'NİN AÇIK BIRAKTIĞI SORU — ve cevabı İKİ PARÇALI.

    Task 2 ölçtü: Schwarzschild borusunda `beta = 2 + 2*gamma` çizgisinin
    TAMAMINDA `w == 1` BİT BİT ve `C` tam 0.0 — yani "fiziği kapat" sınavı
    orada hiçbir şey ölçmüyor (tautoloji). Planın elediği (b) adayı
    *"izotropik ayarda durum farklı olabilir"* diyordu. ÖLÇÜLDÜ, ve FARKLI:

        gamma   beta   R == A   izo C(p=3e5) tek basamak   schw
        -1.0     0.0   EVET     0.0 (dejenere)             0.0
        -0.5     1.0   hayır    canlı                      0.0
         0.0     2.0   hayır    -7.540e-04                 0.0
         1.0     4.0   hayır    -3.016e-03                 0.0

    Yani izotropik boru `lambda = 0` çizgisinin TAMAMINDA değil, YALNIZ
    `beta = 0` NOKTASINDA dejeneredir. Sebebi cebirsel ve ölçüldü: `R`nin
    `u²` katsayısı `2*beta*M²` ile `A`nınkiyle DAİMA eşittir; `u` katsayıları
    ise ancak `beta = 0` iken eşitlenir. Newton noktası (gamma = −1, beta = 0)
    tam olarak o nokta.

    KAZANÇ NE, KAZANÇ NE DEĞİL: bu, Task 2'nin beyan ettiği kusurun
    KAPSAMINI daraltır (çizgi değil, nokta). ONU KAPATMAZ — bir sonraki test
    izotropik borunun o çizgide DÖNDÜRDÜĞÜ ŞEYİN bir gürültü tabanı değil
    gerçek bir MODEL ARTIĞI olduğunu ölçüyor.
    """
    u1, u2 = 0.7 / 3e5, 1.3 / 3e5

    newton = Theory(gamma=-1.0, beta=0.0)
    yorunge = isotropic_orbit(u1, u2, theory=newton, M=1.0)
    assert yorunge.radial_poly == yorunge.time_poly          # BİT BİT özdeş
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert isotropic_w(u1 + (u2 - u1) * f, yorunge) == 1.0

    for gamma in (-0.5, 0.0, 0.5, 1.0, 2.0):
        t = Theory(gamma=gamma, beta=2.0 + 2.0 * gamma)
        assert t.lam == 0.0
        y = isotropic_orbit(u1, u2, theory=t, M=1.0)
        assert y.radial_poly != y.time_poly                  # boru İŞ YAPIYOR
        assert any(isotropic_w(u1 + (u2 - u1) * f, y) != 1.0
                   for f in (0.0, 0.25, 0.5, 0.75, 1.0))
        # ve Schwarzschild borusu AYNI noktada hâlâ bit bit 0.0:
        for p in (1e4, 1e5, 1e6):
            assert perihelion_coefficient(theory=t, M=1.0, e=0.3, p=p,
                                          richardson=None) == 0.0


@pytest.mark.parametrize("gamma", [-0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("e", [0.1, 0.3, 0.7, 0.95])
def test_IZOTROPIK_LAMBDA_SIFIR_CIZGISI_BIR_MODEL_ARTIGI_TASIYOR(gamma, e):
    """ÖNCEKİ TESTİN DEVAMI, VE ONUN KAZANCINI SINIRLAYAN ÖLÇÜM.

    İzotropik boru `lambda = 0` çizgisinde SIFIR DÖNDÜRMÜYOR — ama döndürdüğü
    şey bir GÜRÜLTÜ TABANI DEĞİL, KAPALI FORMU OLAN bir model artığıdır:

        C_izo(p) * p  ->  -24*pi*(1 + gamma)^2        (`e`den BAĞIMSIZ)

    ÖLÇÜLDÜ (p = 1e6, dört `e` × altı `gamma`): bağıl sapma en kötü 1.6e-04,
    `e >= 0.3`'te 1.4e-05. `gamma = -1` (Newton noktası) tam sıfır veriyor ve
    orada ölçülen C(p) = −8.882e-10, yani aritmetik taban.

    NEDEN ÖNEMLİ: `lambda = 0` çizgisi Schwarzschild ayarında `C` VE `K`nın
    İKİSİNİ birden sıfırlar (`K = pi*lam²(3+e²/6)`); izotropik ayarda YALNIZ
    `C`yi sıfırlar. Sebep sevk edilen izotropik metriğin 1PN'de KESİLMİŞ
    olmasıdır — ikinci mertebesi fizik değil MODELLEME ARTIĞIDIR. Dolayısıyla
    izotropik `lambda = 0` NOKTASI Task 2'nin merdivenini gereksiz KILMAZ:
    orada okunacak sayı gürültü değil sinyaldir.
    """
    t = Theory(gamma=gamma, beta=2.0 + 2.0 * gamma)
    assert t.lam == 0.0
    K1 = perihelion_coefficient(theory=t, M=1.0, e=e, p=1e6, richardson=None,
                                gauge="isotropic") * 1e6
    assert K1 == pytest.approx(-24.0 * math.pi * (1.0 + gamma) ** 2, rel=2e-4)


def test_NEWTON_NOKTASI_IZOTROPIK_BORUDA_DA_ARITMETIK_TABANDA():
    """`gamma = -1` (Newton noktası) izotropik boruda TAM SIFIRA iniyor —
    ölçülen C(p = 1e6) = −8.882e-10, yani `-24*pi*(1+gamma)^2 = 0`ın
    aritmetik tabanı. Kapalı formun `gamma = -1` kökü BOŞ DEĞİL, ölçüldü."""
    t = Theory(gamma=-1.0, beta=0.0)
    C = perihelion_coefficient(theory=t, M=1.0, e=0.3, p=1e6, richardson=None,
                               gauge="isotropic")
    assert abs(C) < 1e-08
    assert C == pytest.approx(-8.882e-10, rel=1e-3)


# ---------------------------------------------------------------------------
# Step 3 — İKİNCİ MERTEBE AİLESİ
# ---------------------------------------------------------------------------
K_IZGARASI_LAMBDA = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
K_IZGARASI_E = (0.3, 0.7, 0.95)


@pytest.mark.parametrize("lam", K_IZGARASI_LAMBDA)
@pytest.mark.parametrize("e", K_IZGARASI_E)
def test_IKINCI_MERTEBE_K_PI_LAMBDA_KARE_UC_ARTI_E_KARE_BOLU_ALTI(lam, e):
    """SINIF: FİZİK ÇIPASI — `K(lam, e) = pi*lam²*(3 + e²/6)`.

    `lam` ve hedef değer ELLE YAZILI; `Theory` `(gamma, beta) = ((lam-1)/2, 1)`
    ile kuruluyor, yani `lam` modele bir SONUÇ olarak giriyor.

    ÖLÇÜLDÜ (Schwarzschild ayarı, p = 3e5, 18 nokta): EN KÖTÜ bağıl fark
    **4.0860e-05** (lam = 0.5, e = 0.95). PLANIN 6.7e-05'İ BU IZGARADA VE BU
    PENCEREDE ÜRETİLEMEDİ; ölçülen 4.086e-05 Task 1'in yan ölçümüyle birebir
    aynı çıktı.
    """
    t = Theory(gamma=(lam - 1.0) / 2.0, beta=1.0)
    K = second_order_coefficient(theory=t, M=1.0, p=3e5, e=e)
    hedef = math.pi * lam * lam * (3.0 + e * e / 6.0)
    assert K == pytest.approx(hedef, rel=4.2e-05)


def test_K_IZGARASININ_EN_KOTU_FARKI_ve_PLANIN_SAYISI_CURUDU():
    """İzgaranın EN KÖTÜSÜ tek bir sayı olarak kilitleniyor: **4.0860e-05**,
    `(lam, e) = (0.5, 0.95)`. Planın 6.7e-05'i bu pencerede yok."""
    enkotu, nerede = 0.0, None
    for lam in K_IZGARASI_LAMBDA:
        for e in K_IZGARASI_E:
            t = Theory(gamma=(lam - 1.0) / 2.0, beta=1.0)
            hedef = math.pi * lam * lam * (3.0 + e * e / 6.0)
            bagil = abs(second_order_coefficient(theory=t, M=1.0, p=3e5, e=e)
                        - hedef) / hedef
            if bagil > enkotu:
                enkotu, nerede = bagil, (lam, e)
    assert nerede == (0.5, 0.95)
    assert enkotu == pytest.approx(4.0860e-05, rel=1e-3)
    assert enkotu < 6.7e-05          # planın sayısı bir ÜST SINIR olarak da geçiyor


def test_K_NIN_P_MERDIVENI_OPTIMUM_3E5_TE_ve_SONRA_BOZULUYOR():
    """`p` BİR PENCERE DÜĞMESİDİR ve OPTİMUMDAN SONRA BOZULUYOR — tıpkı
    Richardson çifti gibi, ve AYNI sebeple (`dphi - 2*pi`nin yuvarlama
    kuantumu `(p/M)²` ile büyütülür).

    ÖLÇÜLDÜ, izgaranın en kötü bağıl farkı:

        p = 1e4 -> 1.0968e-03 | 3e4 -> 3.6531e-04 | 1e5 -> 1.0957e-04 |
        p = 3e5 -> 4.0860e-05 | 1e6 -> 3.4144e-04 | 3e6 -> 2.8998e-03 |
        p = 1e7 -> 5.5282e-02

    Optimum gerçekten `p = 3e5`tedir; `p = 1e7`de sonuç 1350 KAT kötü.
    """
    beklenen = {1e4: 1.0968e-03, 3e4: 3.6531e-04, 1e5: 1.0957e-04,
                3e5: 4.0860e-05, 1e6: 3.4144e-04, 3e6: 2.8998e-03,
                1e7: 5.5282e-02}
    olculen = {}
    for p, hedef_fark in beklenen.items():
        enkotu = 0.0
        for lam in K_IZGARASI_LAMBDA:
            for e in K_IZGARASI_E:
                t = Theory(gamma=(lam - 1.0) / 2.0, beta=1.0)
                hedef = math.pi * lam * lam * (3.0 + e * e / 6.0)
                enkotu = max(enkotu, abs(
                    second_order_coefficient(theory=t, M=1.0, p=p, e=e) - hedef) / hedef)
        olculen[p] = enkotu
        assert enkotu == pytest.approx(hedef_fark, rel=1e-3)
    assert min(olculen, key=lambda k: olculen[k]) == 3e5


@pytest.mark.parametrize("p, lam, e, beklenen_oran", [
    (1e4, 6.0, 0.95, 4.4e+06),
    (3e4, 6.0, 0.95, 1.63e+05),
    (1e5, 6.0, 0.95, 4.40e+03),
    (3e5, 0.5, 0.95, 1.265),
    (1e6, 0.5, 0.7, 0.930),
    (3e6, 0.5, 0.7, 0.878),
    (1e7, 0.5, 0.95, 1.540),
])
def test_K_OPTIMUMUNUN_YERI_KAPALI_FORMDAN_ONGORULUYOR(p, lam, e, beklenen_oran):
    """OPTİMUMUN *NEREDE* OLDUĞU DEĞİL, *NEDEN ORADA* OLDUĞU — ve öngörü
    KARELEMEDEN BAĞIMSIZ bir yoldan geliyor.

    `dphi - 2*pi` farkının yuvarlama kuantumu `ulp(2*pi)`dir; `K` onu `(p/M)²`
    ile büyütür. Yani beklenen gürültü `ulp(2*pi)*(p/M)²`. İzgaranın her `p`de
    kendi en kötü noktasındaki MUTLAK hatası bu öngörüye bölündüğünde:

        p = 1e4 -> 4.4e+06 | 3e4 -> 1.6e+05 | 1e5 -> 4.4e+03 |
        p = 3e5 -> 1.265   | 1e6 -> 0.930   | 3e6 -> 0.878   | 1e7 -> 1.540

    `p <= 1e5` KESME SINIRLIDIR (oran binlerce: gerçek hatayı yuvarlama
    açıklamıyor), `p >= 3e5` YUVARLAMA SINIRLIDIR (oran ~1). Optimum tam bu
    geçişin üzerindedir.

    Bu, Task 2'nin "kesme sınırlı / yuvarlama sınırlı pencere" ayrımının `K`
    tarafındaki karşılığıdır — ve orada olduğu gibi burada da bir gürültü
    öngörüsünü HATA ÇUBUĞU yapan şey pencerenin yuvarlama sınırlı olmasıdır.
    """
    t = Theory(gamma=(lam - 1.0) / 2.0, beta=1.0)
    hedef = math.pi * lam * lam * (3.0 + e * e / 6.0)
    mutlak = abs(second_order_coefficient(theory=t, M=1.0, p=p, e=e) - hedef)
    ongoru = math.ulp(2.0 * math.pi) * p * p
    assert mutlak / ongoru == pytest.approx(beklenen_oran, rel=2e-2)


@pytest.mark.parametrize("e, beklenen", [
    (0.3, 85.24858543879077),
    (0.7, 87.13363755319392),
    (0.95, 89.07752260611801),
])
def test_K_NIN_LAMBDA_UC_DILIMI_27PI_ARTI_BIR_BUCUK_PI_E_KARE(e, beklenen):
    """Spec §3.2'nin İLK HÂLİ (`27*pi + 1.5*pi*e²`) bu ailenin `lam = 3`
    DİLİMİDİR — ve dilim burada ADIYLA koşuyor.

    ÖLÇÜLDÜ (GR, p = 3e5): e = 0.3 -> 85.24858543879077 (hedef 85.24711665515903,
    bağıl 1.7230e-05) · e = 0.7 -> 87.13363755319392 (87.1320722473129,
    1.7965e-05) · e = 0.95 -> 89.07752260611801 (89.0759327017216, 1.7849e-05).
    """
    K = second_order_coefficient(M=1.0, p=3e5, e=e)
    assert K == beklenen
    assert K == pytest.approx(27.0 * math.pi + 1.5 * math.pi * e * e, rel=1.8e-05)
    # ve aile formülünün lam = 3 dilimiyle AYNI sayı:
    assert 27.0 * math.pi + 1.5 * math.pi * e * e == pytest.approx(
        math.pi * 9.0 * (3.0 + e * e / 6.0), rel=1e-15)


def test_K_AYAR_BAGIMLIDIR_CAPRAZ_KONTROL_BIRINCI_MERTEBEYI_BELGELER():
    """BU MODÜLÜN ÇAPRAZ KONTROLÜNÜN SINIRI — ölçüldü, varsayılmadı.

    `C` (birinci mertebe) iki ayarda AYNI sayıdır (1.5e-09 … 3.3e-09'da).
    `K` (ikinci mertebe) DEĞİLDİR. ÖLÇÜLDÜ (GR, e = 0.3, p = 3e5):

        K_schwarzschild = 85.24858543879077
        K_izotropik     = 26.57775724301814

    İki sebep birlikte: (a) `p` ile `e` birer KOORDİNAT ETİKETİDİR ve aynı
    yörünge iki ayarda farklı `(p, e)` ile adlandırılır; (b) sevk edilen
    izotropik metrik 1PN'de KESİLMİŞTİR, yani ikinci mertebesi bir MODELLEME
    ARTIĞIDIR. `second_order_coefficient`ın `gauge` düğmesi BU YÜZDEN YOK.

    İzotropik `K` bir GÜRÜLTÜ DEĞİL, tanımlı bir sayıdır — `p` 1e4'ten 1e6'ya
    giderken 26.576587 -> 26.577775, beş hanede kararlı. Ama BAŞKA bir sayıdır.
    """
    Cp_izo = perihelion_coefficient(M=1.0, p=3e5, e=0.3, richardson=None,
                                    gauge="isotropic")
    K_izo = (Cp_izo - closed_form_coefficient(Theory())) * 3e5
    K_schw = second_order_coefficient(M=1.0, p=3e5, e=0.3)

    assert K_schw == 85.24858543879077
    assert K_izo == pytest.approx(26.57775724301814, rel=1e-12)
    assert abs(K_schw - K_izo) > 58.0                # ölçülen fark 58.671

    kararlilik = [
        (perihelion_coefficient(M=1.0, p=p, e=0.3, richardson=None,
                                gauge="isotropic")
         - closed_form_coefficient(Theory())) * p
        for p in (1e4, 1e5, 1e6)
    ]
    assert max(kararlilik) - min(kararlilik) < 2e-03


def test_SECOND_ORDER_COEFFICIENT_RICHARDSON_KULLANAMAZ_cunku_OLCTUGU_TERIMI_SILER():
    """`K` TANIMI GEREĞİ TEK BASAMAKLI bir kalıntıdır — ve bu ilk yazdığımda
    YANLIŞ GEREKÇEYLE savunduğum bir kuraldı.

    İlk hâlim "dışdeğerlenmiş `C`den okunsaydı artık `(p/M)²` ile çarpılıp
    166.69 büyüklüğünde bir K verirdi" diyordu. ÖLÇÜLDÜ, ve YANLIŞ: Richardson
    `1/p`de doğrusal dışdeğerleme yapar, yani `K*M/p` terimini TAM OLARAK
    SİLER. Yanlış yolun verdiği sayı 166.69 değil, ölçülen

        e = 0.3  ->  -5.556419e-04   (doğrusu 85.24858543879077, oran -6.5e-06)
        e = 0.7  ->  +6.100720e-05   (doğrusu 87.13363755319392, oran +7.0e-07)

    yani sinyal BÜYÜMÜYOR, TAMAMEN KAYBOLUYOR. Kural aynı kalıyor, gerekçesi
    tersine dönüyor: *`K` tam da Richardson'ın sildiği terimdir.*
    """
    for e, yanlis_beklenen in ((0.3, -5.556419e-04), (0.7, 6.100720e-05)):
        dogru = second_order_coefficient(M=1.0, p=3e5, e=e)
        yanlis = (perihelion_coefficient(M=1.0, e=e)      # Richardson AÇIK
                  - closed_form_coefficient(Theory())) * 3e5
        assert dogru > 80.0
        assert yanlis == pytest.approx(yanlis_beklenen, rel=1e-4)
        assert abs(yanlis / dogru) < 1e-05


# ---------------------------------------------------------------------------
# GÜRÜLTÜ TABANI — İKİ AYARDA DA KENDİ BORUSUNDAN
# ---------------------------------------------------------------------------
def test_GURULTU_TABANI_IKI_AYARDA_DA_OKUNUYOR_ve_FARKLI():
    """§4.3'ün "kendi borusundan" kuralı ikinci ayarda da koşuyor — VE İKİ
    TABAN FARKLI ÇIKIYOR, ki bu `pipeline` etiketinin ayarı taşımak zorunda
    olduğunun ölçümüdür.

    ÖLÇÜLDÜ (GR sinyali, e = 0.3, M = 1, Richardson (3e5, 1e6)):

        ayar             taban          S/G          hüküm    canlı
        schwarzschild    1.594187e-09   1.1824e+10   SAGLAM   6/6
        isotropic        2.346903e-09   8.0317e+09   SAGLAM   6/6
    """
    schw = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3)
    izo = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3, gauge="isotropic")

    assert schw.value == pytest.approx(1.594187e-09, rel=1e-5)
    assert izo.value == pytest.approx(2.346903e-09, rel=1e-5)
    assert schw.ratio == pytest.approx(1.1824e+10, rel=1e-3)
    assert izo.ratio == pytest.approx(8.0317e+09, rel=1e-3)
    assert schw.verdict == izo.verdict == "SAGLAM"
    assert "6/6 canli" in schw.pipeline and "6/6 canli" in izo.pipeline
    assert izo.value != schw.value


def test_PIPELINE_ETIKETI_AYARI_TASIYOR_KARISIK_BORU_YASAGININ_KAYDI():
    """SINIF: transkripsiyon kilidi — ama SEVK EDİLMİŞ BİR KUSURUN kapanışı.

    Task 2'de `perihelion_coefficient` TEK bir boruydu ve `"kareleme"` onu tam
    adlandırıyordu. Task 3 `gauge`u eklediği anda ETİKET YALAN SÖYLEMEYE
    BAŞLADI: iki AYRI gövde aynı `"kareleme"` etiketini döndürüyordu, oysa
    tabanları 1.594187e-09 ile 2.346903e-09, merdiven ölüm eşikleri p/M = 1e13
    ile 1e14. Yani KARIŞIK BORU YASAĞININ KAYDI, tam da iki ayar sevk edildiği
    anda ayırt edemez hâle gelmişti.

    `PIPELINE_KEYS` bunu kapatır. VE ESKİ BORULARI BOZMAZ: beyan etmeyen bir
    fonksiyonun etiketi HARFİ HARFİNE eskisi gibi kalır (o kilit
    `test_hata_cubugu.py`de duruyor ve bu turda değiştirilmedi).
    """
    assert perihelion_coefficient.PIPELINE_KEYS == ("gauge",)
    schw = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3)
    izo = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3, gauge="isotropic")

    assert schw.pipeline == "kareleme | gauge=schwarzschild | fizik-kapali merdiven 6/6 canli"
    assert izo.pipeline == "kareleme | gauge=isotropic | fizik-kapali merdiven 6/6 canli"
    assert schw.pipeline != izo.pipeline
    # varsayılan İMZADAN okunuyor, ikinci bir yere yazılmıyor:
    assert "gauge=schwarzschild" in newton_noise_floor(
        perihelion_coefficient, M=1.0, e=0.3).pipeline


# ---------------------------------------------------------------------------
# KAPILAR ve ÇEKİRDEĞİN CEBİRSEL SINAVI
# ---------------------------------------------------------------------------
def test_BILINMEYEN_AYAR_REDDEDILIYOR_ve_MESAJ_OLCUM_TASIYOR():
    """Ayar bir KOORDİNAT SEÇİMİDİR ve sonucun penceresine girer; sessizce
    varsayılana düşmek pencereyi hükmün dışına taşırdı."""
    with pytest.raises(InvalidInputError) as bilgi:
        perihelion_coefficient(M=1.0, e=0.3, gauge="harmonic")
    mesaj = str(bilgi.value)
    assert "harmonic" in mesaj
    assert "schwarzschild" in mesaj and "isotropic" in mesaj
    assert "85.248585" in mesaj and "26.577757" in mesaj


@pytest.mark.parametrize("u1, u2, parca", [
    (0.5, 0.5, "AYRI olmali"),
    (0.6, 0.5, "AYRI olmali"),
    (float("nan"), 0.5, "sonlu olmali"),
    (0.1, float("inf"), "sonlu olmali"),
])
def test_IZOTROPIK_ORBIT_KAPILARI(u1, u2, parca):
    """`isotropic_orbit` kapılarını `radial_w`den DEVRALMAZ, kendisi kurar —
    iki yol birbirinin kapısına güvenirse ortak bir düğme doğar."""
    with pytest.raises(InvalidInputError) as bilgi:
        isotropic_orbit(u1, u2, theory=Theory(), M=1.0)
    assert parca in str(bilgi.value)


def test_IZOTROPIK_ORBIT_M_KAPISI():
    for M in (0.0, -1.0, float("nan")):
        with pytest.raises(InvalidInputError):
            isotropic_orbit(0.1, 0.5, theory=Theory(), M=M)


def test_TEKIL_DIZGE_DALI_ERISILEBILIR_ve_IZOTROPIK_YOL_SESSIZ_DUSMUYOR():
    """`det = 0` DALI ÖLÜ KOD DEĞİL — ölçülerek erişildi, ve ERİŞİM NOKTASI
    İKİ BORUNUN AYRIMINI GÖSTERİYOR.

    ÖLÇÜLDÜ (GR, e = 0.3, tek basamak): `p/M >= 1e16`da izotropik çekirdekte
    `B/A` iki dönüş noktasında aynı float'a yuvarlanıyor, `det` tam 0.0 oluyor
    ve boru `InvalidInputError` ile GÜRÜLTÜLÜ düşüyor.

    AYNI `p`DE SCHWARZSCHILD BORUSU SESSİZCE YANLIŞ SAYIYI VERİYOR:
    p = 1e16'da 26.645353, p = 1e17'de 177.635684 (doğrusu 18.849556).
    Yani bu dal bir kusur değil, ikinci ayarın ölçülmüş bir ÜSTÜNLÜĞÜDÜR;
    ve sevk edilen pencerenin (p <= 1e6) on milyar katı ötededir.
    """
    with pytest.raises(InvalidInputError) as bilgi:
        perihelion_coefficient(M=1.0, e=0.3, p=1e16, richardson=None,
                               gauge="isotropic")
    assert "tekil" in str(bilgi.value)

    sessiz = perihelion_coefficient(M=1.0, e=0.3, p=1e16, richardson=None)
    assert sessiz == pytest.approx(26.645353, rel=1e-6)
    assert abs(sessiz - ALTI_PI) > 7.0

    # sevk edilen pencerede iki boru da SAĞLAM:
    assert perihelion_coefficient(M=1.0, e=0.3, p=1e15, richardson=None,
                                  gauge="isotropic") == pytest.approx(21.316, rel=1e-3)


@pytest.mark.parametrize("gamma, beta", [
    (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (2.0, 1.0), (1.0, 3.0),
    (-1.0, 0.0), (0.5, -0.5), (-0.5, 2.0),
])
def test_DEFLASYON_ARTIKLARI_CEKIRDEGIN_BAGIMSIZ_CEBIRSEL_SINAVIDIR(gamma, beta):
    """`isotropic_orbit` kuartiği `(u-u1)(u2-u)R(u)` diye deflate ederken
    DÖRT katsayıdan yalnız ÜÇÜNÜ kullanır; kalan İKİSİ bir SINAVDIR.

    `N`in `u^0` ve `u^1` katsayıları `R`den geri üretilmeli:
        n0 = -Pr*r0        n1 = S*r0 - Pr*r1
    ÖLÇÜLDÜ, sekiz (gamma, beta) noktasında (p = 3e5, e = 0.3, M = 1):
    |n0 artığı| **2.391e-22 … 1.551e-21**, |n1 artığı| **6.667e-18 … 4.507e-16**.

    Bu bir TAUTOLOJİ DEĞİLDİR: `X` ve `Y` 2x2 Cramer'den, `r0/r1/r2` ise
    `n2/n3/n4`ten gelir; `n0` ile `n1` bu zincirin HİÇBİR yerinde kullanılmaz
    (ve `n1` `isotropic_orbit` içinde BİLEREK hesaplanmıyor — ilk yazdığımda
    orada duruyordu ve hiçbir yerde kullanılmıyordu, yani ölü koddu).

    VE BU TEST `energy_term` İLE `inverse_L2`NİN TEK KAPSAYICISIDIR — mutasyon
    turuyla ölçüldü. Üç mutant (`X`in payında işaret, `energy_term=0.0`,
    `inverse_L2=0.0`) süitin geri kalanının TAMAMINDAN geçiyor ve YALNIZ bu
    test tarafından öldürülüyor (her birinde 8 düşen, hepsi bu test).
    Sebebi ölçülebilir: `X` sevk edilen karelemeye HİÇ girmez (deflasyon
    `n2/n3/n4` üzerinden yalnız `Y`ye bağlıdır), `Y` ise girer ama alan olarak
    saklanan kopyası girmez. Bu alanları bir sınav tutmasaydı ikisi de
    ölçülmemiş birer beyan olurdu.
    """
    t = Theory(gamma=gamma, beta=beta)
    u1, u2 = 0.7 / 3e5, 1.3 / 3e5
    y = isotropic_orbit(u1, u2, theory=t, M=1.0)
    X, Y = y.energy_term, y.inverse_L2
    r0, r1, _r2 = y.radial_poly
    M = 1.0
    n0 = X - Y
    n1 = 2.0 * M * (X * gamma - Y * (gamma - 1.0))
    S, Pr = u1 + u2, u1 * u2

    assert abs(-Pr * r0 - n0) < 2e-21
    assert abs(S * r0 - Pr * r1 - n1) < 5e-16


def test_CEKIRDEK_DONGU_DISINDA_KURULUYOR_SAYIMLA_KILITLI(monkeypatch):
    """BU TEST BİR MUTASYON TURUNUN ÜRÜNÜDÜR — hayatta kalan tek mutant.

    `_dphi_isotropic`in dokstringi çekirdeğin DÖNGÜ DIŞINDA kurulduğunu ve
    içeride kurulsaydı 4.3 kat pahalı olacağını yazıyor. Çekirdeği integrandın
    İÇİNE taşıyan mutant SÜİTİN TAMAMINDAN GEÇTİ (182/182): çekirdek
    `(u1, u2, theory, M)`nin deterministik bir fonksiyonu olduğu için çıktı
    BİT BİT aynı kalıyor — yani mutant DAVRANIŞSAL OLARAK EŞDEĞER.

    Kırılan tek şey bir MALİYET iddiasıydı, ve bir duvar-saati eşiği koymak
    testi kararsız (flaky) yapardı. Doğru kilit SAYIMDIR:

        tek basamak (p = 3e5)      -> isotropic_orbit 1,  isotropic_w 21
        Richardson (3e5, 1e6)      -> isotropic_orbit 2,  isotropic_w 42

    Çekirdek döngü içinde kurulsaydı `isotropic_orbit` 1 yerine 21 kez
    çağrılırdı. Ölçüldü, ve mutant bu testte ÖLÜYOR.
    """
    sayac = {"orbit": 0}
    asil = geodesic_modulu.isotropic_orbit

    def sayan(*a, **k):
        sayac["orbit"] += 1
        return asil(*a, **k)

    monkeypatch.setattr(geodesic_modulu, "isotropic_orbit", sayan)

    perihelion_coefficient(M=1.0, e=0.3, p=3e5, richardson=None, gauge="isotropic")
    assert sayac["orbit"] == 1

    sayac["orbit"] = 0
    perihelion_coefficient(M=1.0, e=0.3, gauge="isotropic")
    assert sayac["orbit"] == 2          # Richardson iki basamak, basamak başına BİR


def test_IZOTROPIK_ORBIT_ALANLARI_ve_DONDURULMUS():
    """`IsotropicOrbit` tam olarak dört alanlı ve DONDURULMUŞ."""
    y = isotropic_orbit(0.7 / 3e5, 1.3 / 3e5, theory=Theory(), M=1.0)
    assert tuple(y.__dataclass_fields__) == (
        "time_poly", "radial_poly", "energy_term", "inverse_L2")
    with pytest.raises(FrozenInstanceError):
        y.energy_term = 1.0          # type: ignore[misc]
    # `A(u)` katsayıları kuramın kendisinden okunuyor:
    assert y.time_poly == (1.0, -2.0, 2.0)
    assert isotropic_orbit(0.7 / 3e5, 1.3 / 3e5,
                           theory=Theory(beta=3.0), M=1.0).time_poly == (1.0, -2.0, 6.0)
