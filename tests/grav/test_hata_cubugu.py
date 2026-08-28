"""`λ = 0` HATA ÇUBUĞU ve kareleme çıpa yolu — Modül 6 Task 2.

PENCERE — bu dosyadaki her sayı aşağıdaki TEK pencerede ölçüldü:

- Birim seçimi `G = c = 1`, `M = 1.0` (aksi yazılmadıkça), `e = 0.3`.
- Kareleme `psi` ikamesiyle; `scipy.integrate.quad`, `epsabs = 1e-14`,
  `epsrel = 1e-13`, `limit = 400` (`geodesic.PSI_*`).
- Richardson çifti (alt 3e5, üst 1e6) — `1/p`'de doğrusal dışdeğerleme.
- Fiziği kapatma merdiveni `λ ∈ {1e-4 … 1e-9}`
  (`geodesic.NEWTON_PROBE_LAMBDAS`).

BU DOSYA TASK 1'İN BIRAKTIĞI AÇIĞI KAPATIYOR. Task 1 ölçtü ve burada
bağımsız olarak yeniden ölçüldü: sevk edilen ψ-ikameli boruda `λ = 0`
koşusunun tabanı **tam 0.0**'dır (bit bit), ve o sıfır **hiçbir düğmeye tepki
vermez**. Planın 2.284e-10'u bu boruda ÜRETİLEMEDİ; üretilemeyecek de, çünkü
`λ = 0`'da `w ≡ 1` olur, integrand SABİT 1.0'a düşer ve `quad` sabiti tam
integre eder. Bir sayı hiçbir düğmeye tepki vermiyorsa o bir ölçüm değil bir
SABİTTİR — yani spec §4.3'ün "pazarlık edilmez" makinesi plan hâliyle bir
TAUTOLOJİYDİ.

Sevk edilen onarım fiziği bir NOKTA yerine bir LİMİTTE kapatır ve onarımın
kendisi üç ayrı ölçümle sınandı (aşağıdaki testler): taban **kıpırdıyor**,
tabanın **kareleme dışı bir kapalı-form öngörüsü var**, ve taban **gerçek
hatayı öngörüyor**.

`slow` İŞARETİ YOK, ölçülerek: dosyanın tamamı 0.6 s'nin altında; tek bir
`perihelion_coefficient` çağrısı 0.015 ms, tek bir `newton_noise_floor`
çağrısı 0.12 ms.
"""

from __future__ import annotations

import math

import pytest

from qlab.engine.grav.geodesic import (
    DEFAULT_RICHARDSON,
    NEWTON_PROBE_LAMBDAS,
    PSI_EPSABS,
    PSI_EPSREL,
    PSI_LIMIT,
    perihelion_coefficient,
)
from qlab.engine.grav.metric import Theory, closed_form_coefficient, radial_w
from qlab.errors import InvalidInputError
from qlab.observable.grav_obs import VERDICTS, NoiseFloor, newton_noise_floor

ALTI_PI = 6.0 * math.pi

#: Sevk edilen pencerede ÖLÇÜLEN değerler — hüküm sabitleri, eşik değil.
C_GR = 18.84955591968662
TABAN_GR = 1.5941869030040314e-09


def _newton(lam: float) -> Theory:
    """Fiziği kapatma ailesi — `geodesic._newton_probes` ile AYNI aile."""
    return Theory(gamma=-1.0, beta=-lam)


# ---------------------------------------------------------------------------
# Step 1 — TAŞIYICI TEST
# ---------------------------------------------------------------------------
def test_HATA_CUBUGU_KENDI_BORUSUNDAN_OKUNUYOR_KARISIK_BORU_YASAK():
    """BU DOSYANIN EN ÖNEMLİ TESTİ, ve bir ÇÜRÜTMENİN kilidi.

    Spec §4.3'ün İLK HÂLİNDEKİ tablo GEÇERSİZDİ: gürültüyü ODE'den, sinyali
    KARELEMEDEN alıyordu — bir çözücünün gürültü tabanı ÖTEKİ çözücünün kesme
    hatasına bölünüyordu. `newton_noise_floor` bu yüzden ÇAĞRILAN FONKSİYONUN
    KENDİSİNİ fizik kapalı koşturur ve döndürdüğü kayıtta `pipeline` alanı
    hangi boruyu VE tabanı hangi merdivenden okuduğunu TAŞIR.

    ÖLÇÜLEN (dosya penceresi, PENCERE (2) = Richardson (3e5, 1e6)):

        sinyal   = 18.84955591968662
        taban    = 1.594187e-09        (6/6 basamak canlı)
        S/G      = 1.1824e+10
        hüküm    = SAGLAM

    PLANIN İKİ SAYISI DA ÇÜRÜDÜ ve ikisi de burada yeniden ölçülüdür:
    plan taban 2.284e-10 / S/G 8.25e+10 diyordu; ölçülen 1.594e-09 ve
    1.18e+10. Fark bir ayar farkı değil: planın tabanı `λ = 0` NOKTASINDAN
    okunuyordu ve o nokta bu boruda TAM 0.0 verir (bir alttaki test).
    """
    taban = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3,
                               richardson=DEFAULT_RICHARDSON)   # PENCERE (2)
    assert isinstance(taban, NoiseFloor)
    assert taban.pipeline.startswith("kareleme")
    assert "6/6 canli" in taban.pipeline
    assert taban.signal == C_GR
    assert taban.value == TABAN_GR
    assert 1e-10 < taban.value < 1e-8          # ölçülen 1.594e-09
    assert taban.ratio > 1e9                   # ölçülen 1.1824e+10
    assert taban.verdict == "SAGLAM"


def test_LAMBDA_SIFIR_NOKTASI_DEJENERE_bu_bir_OLCUM_DEGIL_SABIT():
    """TASK 1'İN BIRAKTIĞI AÇIK, SAYIYLA. — `λ = 0` tabanı bir SABİTTİR.

    On yedi ayrı düğme ayarında (yedi `p`, dört Richardson çifti, altı
    (e, M) noktası) `λ = 0` koşusu **her seferinde tam 0.0** döndü. Bir sayı
    hiçbir düğmeye tepki vermiyorsa borunun aritmetiğini ölçmüyordur; buradaki
    sıfır, gürültünün küçüklüğünden değil, ψ ikamesinin o noktada yapacak iş
    bırakmamasından geliyor (`w ≡ 1` → integrand sabit 1.0 → `quad` tam).

    KARŞILAŞTIR: aynı düğmelerden biri (Richardson çifti) bir alttaki testte
    fizik-kapalı MERDİVENİN tabanını tam 10.00 kat kıpırdatıyor.
    """
    sifir = _newton(0.0)
    assert sifir.lam == 0.0
    assert closed_form_coefficient(sifir) == 0.0

    okunanlar: list[float] = []
    for p in (1e2, 1e3, 1e4, 3e5, 1e6, 1e8, 1e12):
        okunanlar.append(perihelion_coefficient(theory=sifir, M=1.0, e=0.3,
                                                p=p, richardson=None))
    for cift in ((1e5, 3e5), (3e5, 1e6), (1e6, 3e6), (3e7, 1e8)):
        okunanlar.append(perihelion_coefficient(theory=sifir, M=1.0, e=0.3,
                                                richardson=cift))
    for e in (0.05, 0.3, 0.95):
        for M in (1e-3, 1e3):
            okunanlar.append(perihelion_coefficient(theory=sifir, M=M, e=e))

    assert len(okunanlar) == 17
    assert all(deger == 0.0 for deger in okunanlar)
    assert len(set(map(abs, okunanlar))) == 1        # tek bir değer: 0.0

    # Ve `λ = 0` TEK BAŞINA merdiven olsaydı hüküm DEJENERE olurdu:
    def yalniz_sifir(*, theory=Theory(), M, e):
        return perihelion_coefficient(theory=theory, M=M, e=e)

    yalniz_sifir.PIPELINE = "kareleme"
    yalniz_sifir.NEWTON_PROBES = (({"theory": sifir}, 0.0),)
    dejenere = newton_noise_floor(yalniz_sifir, M=1.0, e=0.3)
    assert dejenere.verdict == "DEJENERE"
    assert dejenere.value == 0.0
    assert math.isnan(dejenere.ratio)      # `inf` DEĞİL: "sonsuz hassasiyet" yalan olurdu
    assert "0/1 canli" in dejenere.pipeline


# ---------------------------------------------------------------------------
# Step 2 — ONARIMIN KENDİSİ DE ÖLÇÜLMÜŞ BİR İDDİADIR: ÜÇ SINAV
# ---------------------------------------------------------------------------
def test_ONARIM_1_TABAN_KIPIRDIYOR_bilinen_bozulmaya_TAM_10_KAT_tepki():
    """SINAV 1 — *"boruya bilinen büyüklükte bir bozulma sok, kıpırdasın."*

    Bozulma: Richardson çiftini **10 kat** büyütmek. Taban `dphi`nin ulp
    kuantumunun `p/M` ile büyütülmüş hâli olduğu için tam 10 kat büyümeli.

    ÖLÇÜLEN (e = 0.3, M = 1, altı basamaklı merdiven):

        (3e4, 1e5) -> 1.5942e-10
        (3e5, 1e6) -> 1.5942e-09   oran 10.00
        (3e6, 1e7) -> 1.5942e-08   oran 10.00
        (3e7, 1e8) -> 1.5942e-07   oran 10.00

    VE TABAN `p/M` ORANININ FONKSİYONUDUR, `p` ile `M`nin ayrı ayrı değil:
    (M = 1, 3e5), (M = 10, 3e6), (M = 0.1, 3e4) üçlüsünde taban
    1.5941869030040314e-09 / 1.5941869030031844e-09 / 1.5941869030040314e-09
    çıkıyor — en büyük bağıl fark **5.313e-13**. (İLK YAZDIĞIM ASSERT "bit
    bit aynı" istiyordu ve KIRMIZI DÜŞTÜ: M = 10 dalı son üç hanede ayrılıyor.
    Ölçüm hükmü daralttı, eşik gevşetilmedi — hüküm artık "oran bit bit aynı"
    değil "oran on iki hane aynı".)

    AYNI BOZULMADA `λ = 0` TABANI KIPIRDAMIYOR — bir üstteki testte dört
    Richardson çiftinin dördünde de tam 0.0.
    """
    tabanlar = []
    for cift in ((3e4, 1e5), (3e5, 1e6), (3e6, 1e7), (3e7, 1e8)):
        tabanlar.append(newton_noise_floor(perihelion_coefficient, M=1.0,
                                           e=0.3, richardson=cift).value)
    for onceki, sonraki in zip(tabanlar, tabanlar[1:]):
        assert sonraki / onceki == pytest.approx(10.0, rel=0.02)
    assert tabanlar[1] == TABAN_GR

    # `p/M` ORANI: üç ayrı (M, çift) üçlüsü, bit bit aynı taban.
    esdegerler = [
        newton_noise_floor(perihelion_coefficient, M=M, e=0.3, richardson=cift).value
        for M, cift in ((1.0, (3e5, 1e6)), (10.0, (3e6, 1e7)), (0.1, (3e4, 1e5)))
    ]
    assert (max(esdegerler) - min(esdegerler)) / max(esdegerler) == pytest.approx(
        5.313e-13, rel=1e-3
    )
    assert esdegerler[0] == TABAN_GR


def test_ONARIM_2_TABANIN_KARELEME_DISI_KAPALI_FORM_ONGORUSU_VAR():
    """SINAV 2 — taban NEREDEN geldiğini bilen bir formülle uyuşuyor.

    Taban, `dphi ≈ 2π`'nin ulp kuantumunun Richardson ağırlıklarıyla
    büyütülmüş hâlidir. Dışdeğerleme

        C_R = w1·f1 + w2·f2,   w1 = h2/(h2−h1) = −0.4285714285714285,
                               w2 = −h1/(h2−h1) = +1.4285714285714284

    ve `f_i = (dphi_i − 2π)·p_i/M` olduğundan `dphi`'deki bir ulp'lik
    yuvarlama `C_R`'de `ulp(2π)·(|w1|·alt + |w2|·üst)` kadar görünür.

    ÖLÇÜLEN: öngörü **1.383021e-09**, ölçülen taban **1.594187e-09**,
    oran **1.153**. Öngörü karelemeyi HİÇ çağırmaz — `ulp`, iki ağırlık ve
    iki basamak; yani bu, tabanın İKİNCİ, BAĞIMSIZ kod yolundan okunuşudur.

    (Öngörünün tam eşitlik vermemesi beklenir: `dphi` iki basamakta bağımsız
    olarak yuvarlanır ve merdiven altı `λ`nın EN BÜYÜĞÜNÜ alır, yani ölçülen
    değer bir üst sınır örneklemesidir.)
    """
    alt, ust = DEFAULT_RICHARDSON
    h1, h2 = 1.0 / alt, 1.0 / ust
    w1, w2 = h2 / (h2 - h1), -h1 / (h2 - h1)
    assert w1 == pytest.approx(-3.0 / 7.0, rel=1e-15)
    assert w2 == pytest.approx(10.0 / 7.0, rel=1e-15)

    ongoru = math.ulp(2.0 * math.pi) * (abs(w1) * alt + abs(w2) * ust)
    assert ongoru == pytest.approx(1.383021e-09, rel=1e-6)

    taban = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3)
    assert taban.value / ongoru == pytest.approx(1.153, rel=1e-3)
    assert 1.0 < taban.value / ongoru < 2.0


@pytest.mark.parametrize("gamma, beta, kat, gercek_hane", [
    (1.0, 1.0, 3.0, 10.01),
    (0.0, 1.0, 1.0, 9.93),
    (1.0, 0.0, 4.0, 9.89),
    (2.0, 1.0, 5.0, 9.70),
    (1.0, 3.0, 1.0, 9.93),
])
def test_ONARIM_3_TABAN_GERCEK_HATAYI_ONGORUYOR(gamma, beta, kat, gercek_hane):
    """SINAV 3 — SINIF: FİZİK ÇIPASI. Taban gerçek hatayı ÖNGÖRÜYOR mu?

    Bir hata çubuğu ancak gerçek hatayı öngörüyorsa hata çubuğudur.
    `S/G`nin hanesi (`log10(sinyal/taban)`) ile GERÇEK hane
    (`log10(|C_tam|/|C_ölçülen − C_tam|)`) karşılaştırılıyor.

    ÖLÇÜLEN (taban 1.594187e-09, tüm satırlar dosya penceresinde):

        gamma beta  C_ölçülen              gerçek hata  öngörü  gerçek
        1.0   1.0   18.84955591968662      1.852e-09    10.07   10.01
        0.0   1.0    6.283185307915622     7.360e-10     9.60    9.93
        1.0   0.0   25.132741225508678     3.210e-09    10.20    9.89
        2.0   1.0   31.415926529605134     6.293e-09    10.29    9.70
        1.0   3.0    6.283185307915622     7.360e-10     9.60    9.93

    En büyük sapma **0.60 hane** (γ = 2), GR'de **0.0651 hane**. Eşik 1.0 hane
    = en kötü ölçümün 1.7 katı.

    SİMÜLASYON BU SAYI HAKKINDA YANLIŞ OLABİLİRDİ: taban `max` yerine `min`
    alsaydı öngörü 10.07 yerine 10.78 olurdu (ölçüldü: merdivenin en küçük
    artığı 3.1385e-10); merdiven `λ = 0` olsaydı taban 0 ve öngörü sonsuz
    olurdu; `w`nin `a` katsayısı yarılansaydı `C` 9.42 çıkardı.
    """
    t = Theory(gamma=gamma, beta=beta)
    tam = 2.0 * math.pi * kat
    olculen = perihelion_coefficient(theory=t, M=1.0, e=0.3)
    taban = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3)

    assert taban.value == TABAN_GR
    assert olculen == pytest.approx(tam, rel=1e-8)
    assert olculen == pytest.approx(closed_form_coefficient(t), rel=1e-8)

    ongoru_hane = math.log10(abs(tam) / taban.value)
    olculen_hane = math.log10(abs(tam) / abs(olculen - tam))
    assert olculen_hane == pytest.approx(gercek_hane, abs=0.01)
    assert abs(ongoru_hane - olculen_hane) < 1.0


def test_MERDIVEN_FIZIGI_DEGIL_ARITMETIGI_OLCUYOR():
    """Merdivenin artığı `λ`DAN BAĞIMSIZ — kesme hatası olsaydı olamazdı.

    Bu, "fiziği kapat" iddiasının SAYISI: kesme hatası `~λ²` gider, yani beş
    dekatlık bir `λ` aralığında **1e10 kat** oynardı. ÖLÇÜLEN (dosya
    penceresi, altı basamak):

        λ = 1e-4  ->  +3.1385e-10
        λ = 1e-5  ->  -4.8756e-10
        λ = 1e-6  ->  -1.5942e-09
        λ = 1e-7  ->  +6.2598e-10
        λ = 1e-8  ->  +1.5738e-09
        λ = 1e-9  ->  +1.1268e-09

    en büyük / en küçük = **5.08**. İşaret de dönüyor (dört artı, iki eksi),
    yani bu bir kesme kalıntısı değil bir YUVARLAMA KUANTUMU.

    MERDİVENİN İKİ UCU DA ÖLÇÜLEREK SEÇİLDİ:
    - üst uç: `λ = 1e-2`'de artık 2.0228e-09'a çıkıyor (plato 1.59e-09), yani
      kesme girmeye başlıyor;
    - alt uç: `λ = 3e-12`'de `w` yeniden tam 1.0'a yuvarlanıyor ve boru tam
      0.0 döndürüyor — basamak ÖLÜYOR. `λ = 1e-9` bu eşiğin 333 katı üstünde.
    """
    artiklar = []
    for lam in NEWTON_PROBE_LAMBDAS:
        t = _newton(lam)
        artiklar.append(perihelion_coefficient(theory=t, M=1.0, e=0.3)
                        - closed_form_coefficient(t))
    assert NEWTON_PROBE_LAMBDAS == (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
    assert len(artiklar) == 6
    buyukler = [abs(a) for a in artiklar]
    assert max(buyukler) / min(buyukler) == pytest.approx(5.08, rel=0.01)
    assert max(buyukler) == TABAN_GR
    assert sum(1 for a in artiklar if a > 0) == 4      # işaret dönüyor

    # ALT UÇ: eşiğin altında boru TAM 0.0 döndürüyor (basamak ölü).
    for olu_lam in (3e-12, 1e-12, 1e-13):
        assert perihelion_coefficient(theory=_newton(olu_lam), M=1.0, e=0.3) == 0.0
    # ÜST UÇ: kesme girmeye başlıyor.
    ust_lam = 1e-2
    ust_artik = (perihelion_coefficient(theory=_newton(ust_lam), M=1.0, e=0.3)
                 - closed_form_coefficient(_newton(ust_lam)))
    assert abs(ust_artik) == pytest.approx(2.0228e-09, rel=1e-3)
    assert abs(ust_artik) > TABAN_GR


# ---------------------------------------------------------------------------
# DÜRÜST KUSUR BEYANI — taban NEYİ GÖRMÜYOR
# ---------------------------------------------------------------------------
def test_KESME_SINIRLI_PENCEREDE_TABAN_HATA_CUBUGU_DEGIL():
    """DÜRÜST KUSUR: "fiziği kapat" sınavı KESME hatasını YAPISAL olarak
    göremez, çünkü kesme hatası fizikle BİRLİKTE kapanıyor.

    İKİ PENCERE, ve onları ayıran şey planın dediği gibi "sinyalin tanımı"
    DEĞİL, NEYİN SINIRLADIĞIDIR. ÖLÇÜLEN:

        pencere                    C                   |C−6π|      taban      taban/hata
        (1) tek basamak p = 3e5    18.84984008349022   2.8416e-04  2.4534e-10  8.63e-07
        (2) Richardson (3e5, 1e6)  18.84955591968662   1.8521e-09  1.5942e-09  8.61e-01

    Pencere (1)'de S/G = **7.68e+10** ("10.89 hane") ama GERÇEK hane
    **4.82**'dir — taban gerçek hatanın **1/1 160 000**'i. Yani orada taban
    bir hata çubuğu DEĞİLDİR. Pencere (2) yuvarlama sınırlıdır ve orada
    taban gerçek hatanın 0.86 katıdır.

    PLANIN PENCERE (1) SAYISI DA ÇÜRÜDÜ: plan "S/G = 5.3e+05" diyordu ve
    sinyali sonlu-`p` kalıntısı (`K·M/p`) olarak tanımlıyordu. Bu ölçümle
    o tanım bile **1.158e+06** verir (2.8416e-04 / 2.4534e-10), 5.3e+05
    değil.
    """
    p1 = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3,
                            p=3e5, richardson=None)               # PENCERE (1)
    p2 = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3)  # PENCERE (2)

    assert p1.value == pytest.approx(2.4534e-10, rel=1e-3)
    assert p1.signal == pytest.approx(18.84984008349022, rel=1e-15)
    hata1 = abs(p1.signal - ALTI_PI)
    assert hata1 == pytest.approx(2.8416e-04, rel=1e-3)
    assert p1.value / hata1 < 1e-5             # ölçülen 8.63e-07: taban KÖR
    assert p1.verdict == "SAGLAM"              # ...ama hüküm yine de SAGLAM

    hata2 = abs(p2.signal - ALTI_PI)
    assert p2.value / hata2 == pytest.approx(0.861, rel=0.01)

    # Planın kendi pencere-(1) tanımıyla (sinyal = sonlu-p kalıntısı):
    assert hata1 / p1.value == pytest.approx(1.158e06, rel=1e-2)


# ---------------------------------------------------------------------------
# Step 3 — S/G < 1 İSE PANEL SÖYLER
# ---------------------------------------------------------------------------
def _oyuncak_boru(*, sinyal: float, gurultu: float, kapali: bool = False) -> float:
    """Sınıflandırma kuralının SINAVI için asgari boru — fizik YOK."""
    return gurultu if kapali else sinyal


_oyuncak_boru.PIPELINE = "oyuncak"
_oyuncak_boru.NEWTON_PROBES = (({"kapali": True}, 0.0),)


def test_SINYAL_GURULTUNUN_ALTINA_DUSERSE_PANEL_SOYLUYOR():
    """SINIF: TRANSKRİPSİYON KİLİDİ — ve bunu böyle söylemek zorundayım.

    Bir hata çubuğu, sinyali geçtiğinde SUSUYORSA hata çubuğu değildir.
    Kural: `ratio < 1` → `verdict == "CURUK"`.

    AMA BU TEST SENTETİK BİR BORUYLA KOŞUYOR ve sebebi ÖLÇÜLDÜ, seçilmedi:
    KARELEMEDE `CURUK` ERİŞİLEMEZ (bir alttaki test). Planın Step 3'ü
    `p = 1e2, richardson=None` ile tetiklenebileceğini söylüyordu; ölçüldü,
    OLMUYOR — orada taban **9.4724e-10**, sinyal **19.7475020632913**,
    S/G = **2.0847e+10**, hüküm SAGLAM. Sebep yapısaldır: taban KESME
    hatasını görmez, `p = 1e2`'yi kötü yapan şey ise tam olarak kesme
    hatasıdır — o pencerede `|C − 6π| = 0.898`, yani sayı GERÇEKTE 1.3
    haneliyken S/G ona "10.3 hane" diyor. Bu, "taban kesme sınırlı pencerede
    hata çubuğu değildir" hükmünün en sert örneğidir.

    Dolayısıyla bu gövde kuralın kendisini kilitler; kuralın kareleme
    üzerindeki FİZİK kapsaması bir alttaki testtedir ve orada hüküm
    `DEJENERE`'dir.
    """
    curuk = newton_noise_floor(_oyuncak_boru, sinyal=1.0, gurultu=4.0)
    assert curuk.value == 4.0
    assert curuk.signal == 1.0
    assert curuk.ratio == 0.25
    assert curuk.verdict == "CURUK"
    assert curuk.pipeline == "oyuncak | fizik-kapali merdiven 1/1 canli"

    saglam = newton_noise_floor(_oyuncak_boru, sinyal=10.0, gurultu=1.0)
    assert saglam.ratio == 10.0
    assert saglam.verdict == "SAGLAM"

    dejenere = newton_noise_floor(_oyuncak_boru, sinyal=10.0, gurultu=0.0)
    assert dejenere.value == 0.0
    assert math.isnan(dejenere.ratio)
    assert dejenere.verdict == "DEJENERE"

    # Sınırda: ratio TAM 1.0 SAGLAM'dır (CURUK sıkı eşitsizliktir).
    sinir = newton_noise_floor(_oyuncak_boru, sinyal=2.0, gurultu=2.0)
    assert sinir.ratio == 1.0
    assert sinir.verdict == "SAGLAM"

    # Planın önerdiği tetikleyici ÖLÇÜLDÜ ve TETİKLEMİYOR:
    plan_noktasi = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3,
                                      p=1e2, richardson=None)
    assert plan_noktasi.verdict == "SAGLAM"
    assert plan_noktasi.value == pytest.approx(9.4724e-10, rel=1e-3)
    assert plan_noktasi.ratio == pytest.approx(2.0847e10, rel=1e-3)
    assert abs(plan_noktasi.signal - ALTI_PI) == pytest.approx(0.898, rel=1e-2)
    assert set(VERDICTS) == {"SAGLAM", "CURUK", "DEJENERE"}


def test_KARELEMEDE_CURUK_ERISILEMEZ_DEJENERE_ONCE_GELIYOR():
    """SINIF: FİZİK ÇIPASI (olumsuz) — ve bir YAPISAL çürütme.

    Karelemede sinyalin gürültüye gömülmesi ile merdivenin ölmesi AYNI
    koşuldur: `w − 1` ile `dphi − 2π` aynı küçük büyüklüğün (`λ·M/p`) iki
    yüzüdür. Yani `p/M` büyütülüp sinyal boğulmaya çalışıldığında, önce
    PROBE ölür.

    ÖLÇÜLEN (`richardson = (p, 10p/3)`, e = 0.3, M = 1):

        p/M     taban        canlı   S/G        hüküm
        1e+05   1.1268e-09   6/6     1.673e+10  SAGLAM
        1e+08   1.1268e-06   5/6     1.673e+07  SAGLAM
        1e+10   1.1268e-04   3/6     1.673e+05  SAGLAM
        1e+12   7.0692e-03   1/6     2.667e+03  SAGLAM
        1e+13   0.0          0/6     nan        DEJENERE
        1e+15   0.0          0/6     nan        DEJENERE

    On üç `p/M` noktasında **CURUK çıkan tek nokta yok**; hüküm doğrudan
    SAGLAM'dan DEJENERE'ye geçiyor. Bu, `CURUK` dalının kareleme için ölü
    olduğunun ölçümüdür — ve `DEJENERE`'nin neden ayrı bir hüküm olması
    gerektiğinin de: bu boruda "panel söylüyor mu" sorusunun cevabını O
    taşıyor.
    """
    hukumler = []
    for us in (5, 8, 10, 12, 13, 15):
        p = 10.0 ** us
        taban = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3,
                                   richardson=(p, p * 10.0 / 3.0))
        hukumler.append(taban.verdict)
    assert hukumler == ["SAGLAM", "SAGLAM", "SAGLAM", "SAGLAM",
                        "DEJENERE", "DEJENERE"]
    assert "CURUK" not in hukumler

    # Ölü basamak SAYIMI `pipeline` alanında taşınıyor:
    orta = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3,
                              richardson=(1e10, 1e10 * 10.0 / 3.0))
    assert "3/6 canli" in orta.pipeline
    assert orta.value == pytest.approx(1.1268e-04, rel=1e-3)

    son = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3,
                             richardson=(1e13, 1e13 * 10.0 / 3.0))
    assert "0/6 canli" in son.pipeline
    assert son.value == 0.0 and math.isnan(son.ratio)


# ---------------------------------------------------------------------------
# `newton_noise_floor`ın SÖZLEŞMESİ
# ---------------------------------------------------------------------------
def test_FIZIGI_KAPATMA_NOKTASI_CAGIRANA_BIRAKILMIYOR():
    """`theory`yi çağıran veremez — yoksa "fiziği kapat" bir DÜĞME olurdu.

    Makine sınavını kendisi seçemez; sınavı ÇAĞRILAN beyan eder
    (`fn.NEWTON_PROBES`), ve çağıran o anahtarlardan birini verirse çağrı
    REDDEDİLİR.
    """
    with pytest.raises(InvalidInputError) as bilgi:
        newton_noise_floor(perihelion_coefficient, theory=Theory(), M=1.0, e=0.3)
    assert "theory" in str(bilgi.value)

    with pytest.raises(InvalidInputError):
        newton_noise_floor(_oyuncak_boru, sinyal=1.0, gurultu=1.0, kapali=False)


def test_PIPELINE_ETIKETI_FN_ADINDAN_OKUNMUYOR():
    """`pipeline` alanı `fn.__name__`den DEĞİL `fn.PIPELINE`dan okunur.

    Ad bir kod ayrıntısıdır; boru adı panelde yazılan bir BEYANDIR. İkisi
    burada BİLEREK ayrıştırılıyor: `perihelion_coefficient`ın adı
    "perihelion", etiketi "kareleme".
    """
    taban = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3)
    assert perihelion_coefficient.PIPELINE == "kareleme"
    assert perihelion_coefficient.__name__ not in taban.pipeline
    assert taban.pipeline.startswith("kareleme |")

    def etiketsiz(*, theory=Theory(), M, e):
        return 1.0

    with pytest.raises(InvalidInputError) as bilgi:
        newton_noise_floor(etiketsiz, M=1.0, e=0.3)
    assert "PIPELINE" in str(bilgi.value)

    def probesiz(*, theory=Theory(), M, e):
        return 1.0

    probesiz.PIPELINE = "probesiz"
    with pytest.raises(InvalidInputError) as bilgi:
        newton_noise_floor(probesiz, M=1.0, e=0.3)
    assert "NEWTON_PROBES" in str(bilgi.value)


def test_NOISE_FLOOR_BES_ALANLI_ve_DONDURULMUS():
    """`NoiseFloor` tam olarak `(value, signal, ratio, pipeline, verdict)`."""
    alanlar = tuple(NoiseFloor.__dataclass_fields__)
    assert alanlar == ("value", "signal", "ratio", "pipeline", "verdict")
    kayit = newton_noise_floor(perihelion_coefficient, M=1.0, e=0.3)
    assert kayit.verdict in VERDICTS
    with pytest.raises(Exception):
        kayit.value = 0.0        # type: ignore[misc]


# ---------------------------------------------------------------------------
# Step 4 — KARELEME ÇIPA YOLU ve RICHARDSON PENCERESİ
# ---------------------------------------------------------------------------
def test_P_ILE_RICHARDSON_AYNI_SEYI_BELIRLIYOR_IKISI_BIRDEN_YASAK():
    """`p` ↔ `richardson` ilişkisi İMZADA TANIMLI — ve PLANIN KENDİ ÇAĞRISI
    BU KURALI ÇİĞNİYORDU.

    Planın Task 2 Step 1'i `newton_noise_floor(..., p=3e5, richardson=(3e5,
    1e6))` diye çağırıyor. O çağrının ne demek olduğu TANIMSIZDIR: iki alan
    aynı şeyi belirlerse sessiz bir kazanan doğar ve o kazanan adlandırılmamış
    bir pencere düğmesidir (Task 1'in `lam` alanında ölçtüğü kusurun aynısı).
    Burada REDDEDİLİYOR.
    """
    with pytest.raises(InvalidInputError) as bilgi:
        perihelion_coefficient(M=1.0, e=0.3, p=3e5, richardson=(3e5, 1e6))
    assert "ayni seyi" in str(bilgi.value).lower() or "AYNI SEYI" in str(bilgi.value)

    with pytest.raises(InvalidInputError):
        perihelion_coefficient(M=1.0, e=0.3, p=None, richardson=None)

    # Tek basamak: `richardson=None` + `p`
    tek = perihelion_coefficient(M=1.0, e=0.3, p=3e5, richardson=None)
    assert tek == pytest.approx(18.84984008349022, rel=1e-15)
    # Varsayılan: Richardson, `p` verilmez
    assert perihelion_coefficient(M=1.0, e=0.3) == C_GR
    assert DEFAULT_RICHARDSON == (3e5, 1e6)

    for bozuk in ((3e5,), (3e5, 3e5), (0.0, 1e6), (3e5, math.inf)):
        with pytest.raises(InvalidInputError):
            perihelion_coefficient(M=1.0, e=0.3, richardson=bozuk)


def test_RICHARDSON_PENCERESI_OPTIMUMDAN_SONRA_BOZULUYOR():
    """Bir dışdeğerleme SONSUZA KADAR İYİLEŞMEZ — sevk edilen fonksiyonla
    YENİDEN ölçüldü (Task 1 bunu test-içi bir tüketiciyle ölçmüştü).

    ÖLÇÜLEN (GR, e = 0.3, M = 1; dürüst hane = −log10(|C−6π|/6π)):

        (1e5, 3e5) -> 9.12   C = 18.849555907385355
        (3e5, 1e6) -> 10.01  C = 18.84955591968662    <- OPTİMUM, sevk edilen
        (1e6, 3e6) -> 9.94   C = 18.849555923683425
        (3e6, 1e7) -> 8.98   C = 18.84955594112979
        (3e7, 1e8) -> 8.93   C = 18.849555899639164

    Planın G bloğu bu merdivenin ŞEKLİNİ doğru veriyordu; DEĞERLERİ Task
    1'de düzeltilmişti ve burada üçüncü kez aynı çıkıyor.
    """
    merdiven = {
        (1e5, 3e5): (18.849555907385355, 9.12),
        (3e5, 1e6): (18.84955591968662, 10.01),
        (1e6, 3e6): (18.849555923683425, 9.94),
        (3e6, 1e7): (18.84955594112979, 8.98),
        (3e7, 1e8): (18.849555899639164, 8.93),
    }
    haneler = []
    for cift, (beklenen_C, beklenen_hane) in merdiven.items():
        c = perihelion_coefficient(M=1.0, e=0.3, richardson=cift)
        assert c == beklenen_C
        hane = -math.log10(abs(c - ALTI_PI) / ALTI_PI)
        assert hane == pytest.approx(beklenen_hane, abs=0.01)
        haneler.append(hane)
    assert max(haneler) == haneler[1]        # optimum (3e5, 1e6)'da


def test_KARELEME_TASK_1_IN_TEST_ICI_TUKETICISIYLE_BIT_BIT_AYNI():
    """SINIF: transkripsiyon denetimi. Sevk edilen `perihelion_coefficient`,
    Task 1'in test-içi tüketicisinin verdiği sayıyı BİT BİT üretiyor
    (18.84955591968662) — yani Task 1'in on satırlık tüketicisi artık gerçek
    kodla değiştirildi ve hüküm değişmedi.

    ORTAK DÜĞME UYARISI: iki gövde de `radial_w`i ve `Theory.lam`i okur, yani
    bu karşılaştırma `metric.py`deki bir mutantı GÖREMEZ. Gördüğü şey dar:
    ikamenin/dışdeğerlemenin `geodesic.py`ye taşınırken bozulmadığı.
    """
    assert perihelion_coefficient(M=1.0, e=0.3) == C_GR
    assert (PSI_EPSABS, PSI_EPSREL, PSI_LIMIT) == (1e-14, 1e-13, 400)


@pytest.mark.parametrize("e", [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
def test_TABAN_DISMERKEZLIKTEN_NEREDEYSE_BAGIMSIZ(e):
    """Taban `e`den okunamıyor: yedi dışmerkezlikte iki değer çıkıyor —
    altısında 1.5941869030040314e-09, `e = 0.1`'de 2.212292038567813e-09
    (1.39 kat). Bu bir `e` bağımlılığı değil, tabanın KUANTUMLU olmasıdır:
    `dphi − 2π` ulp adımlarında ilerler ve `e = 0.1` bir üst basamağa oturur.
    """
    taban = newton_noise_floor(perihelion_coefficient, M=1.0, e=e)
    assert taban.value in (1.5941869030040314e-09, 2.212292038567813e-09)
    assert taban.verdict == "SAGLAM"
    assert taban.ratio > 8e9


# ---------------------------------------------------------------------------
# PLANIN ADAY (b) YOLU — ÖLÇÜLDÜ ve KARELEMEDE YANLIŞ
# ---------------------------------------------------------------------------
def test_PLAN_ADAYI_B_OLCULDU_ve_KARELEMEDE_YANLIS():
    """Plan, tabanı `β = 2 + 2γ` çizgisinde `γ ≠ −1` noktalarından okumayı
    öneriyor ve gerekçesini *"orada katsayı yine tam sıfır ama `w ≢ 1`"*
    diye yazıyor. ÖLÇÜLDÜ: bu, SCHWARZSCHILD karelemesi için YANLIŞ.

    `radial_w` teoriye YALNIZ `lam` üzerinden bağlıdır (`a = 2·lam·M/3`), ve
    `β = 2 + 2γ` çizgisi tam olarak `lam = 0` çizgisidir. Beş γ noktasında
    (−3, −1, 0, 1, 5) `w(orta) == 1.0` **bit bit** ve `C == 0.0`.

    Plan (b)'nin işe yaradığı yer izotropik ayardır — orada `A` ile `B`
    γ ve β'yı AYRI AYRI okur — ama izotropik kareleme Task 3'ün işidir.
    Task 2 bu yüzden aday (a)'yı, `λ → 0` limitini seçti.
    """
    p, e = 3e5, 0.3
    u1, u2 = (1.0 - e) / p, (1.0 + e) / p
    for gamma in (-3.0, -1.0, 0.0, 1.0, 5.0):
        t = Theory(gamma=gamma, beta=2.0 + 2.0 * gamma)
        assert t.lam == 0.0
        assert radial_w(0.5 * (u1 + u2), u1, u2, theory=t, M=1.0) == 1.0
        assert perihelion_coefficient(theory=t, M=1.0, e=e) == 0.0


# ---------------------------------------------------------------------------
# KAPILAR
# ---------------------------------------------------------------------------
def test_E_KAPISI_BAGLI_YORUNGE_ve_GEREKCESI_SOZLESMESEL():
    """`0 < e < 1` kapısı — gerekçesi SAYISAL DEĞİL SÖZLEŞMESEL, ve bu
    ölçülerek biliniyor: kapı olmasaydı `e = 1.0` için 18.84955592043523,
    `e = 2.0` için 18.84955591693327 dönerdi — yani KAPALI BİR YÖRÜNGENİN
    OLMADIĞI yerde inanılır görünen bir "yörünge başına perihel kayması"
    sayısı. Sessizce yanlış olmak, gürültülü olmaktan kötüdür.
    """
    for kotu in (0.0, 1.0, 1.5, -0.1, math.nan, math.inf):
        with pytest.raises(InvalidInputError):
            perihelion_coefficient(M=1.0, e=kotu)
    assert math.isfinite(perihelion_coefficient(M=1.0, e=1e-6))


def test_M_KAPISI():
    """`M <= 0`: `C = (dphi−2π)·p/M` olduğu için `M = 0` sıfıra bölmedir."""
    for kotu in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(InvalidInputError):
            perihelion_coefficient(M=kotu, e=0.3)
