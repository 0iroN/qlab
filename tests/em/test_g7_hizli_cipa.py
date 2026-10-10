"""G7 — HIZLI ŞERİDİN KAPALI-FORM ÇIPALARI (Modül 5, Maxwell/FDTD).

NEDEN VAR: modülün adanmış fizik çıpalarının hepsi `slow` işaretli (Bessel
tarafında `kayitlar`ı tüketen 11 test, Rayleigh, yarık, TİY/Fresnel). G7
ölçümü (09.10.2026, `.superpowers/sdd/g7-olcum/em/`) altı fizik mutantını
iki şeritte ayrı ayrı koştu ve şunu buldu:

    M1  2B'de kırılma indisi yanlış (`eps` -> `sqrt(eps)`, yalnız 2B)
        -> hızlı şerit KAÇIRDI, yavaş şerit öldürdü
           (`test_bessel.py::test_EPS_R_BAGIMSIZLIGI_...`).

    YENİDEN ÖLÇÜLDÜ (düzeltme turu, `g7-olcum/duzeltme/em/mut_m1_*.log`):
    M1 altında bu dosya HARİÇ hızlı şerit 201/201 yeşil; yavaş şeritteki
    `test_EPS_R_BAGIMSIZLIGI_...` tek başına koşulunca DÜŞÜYOR.

Sebep yapısal: hızlı şeridin bütün 2B koşumları `eps_r = 1`dedir
(`Medium.uniform_2d` varsayılanı, sevk senaryosu `eps_r: 1.0`) ve orada
`sqrt(1) = 1` olduğu için mutant BİT BİT görünmez. Ayrıca hızlı şeritte
ARAYÜZ fiziği (Fresnel) yalnız `|r| < 1` diye sınanıyordu; arayüz kuralı
mutantı (M5, aritmetik -> geometrik ortalama) yalnız bir kararlılık-kapısı
modeli ve iki transkripsiyon kilidiyle düşüyordu, bir kapalı form
referansıyla değil.

SINIF — İKİ SÖZLÜK VAR VE KARIŞTIRILMIYOR (G7 denetimi, bulgu 2):

* G7 turunun çıpa tanımı ("bağımsız referans: analitik / kapalı form /
  literatür; regresyon sabiti ve şema/metin testi çıpa değildir") için
  ikisi de KAPALI-FORM
  ÇIPASIDIR: referanslar bu deponun HİÇBİR çıktısı değildir, testte elle
  yazılır (kaynak koddan okunmaz) ve hiçbir assert bir regresyon sabitine
  bakmaz. G7'deki "şema/metin testi" bu modülün TRANSKRİPSİYON DENETİMİ
  dediği sınıftır (ad, dizi, girdi kilitleri) — bu dosyanın okuması budur.
* Modül 5'in kendi sözlüğünde (`test_sema_cipalari.py` başlığı, spec §3.4;
  `test_arayuz.py` başlığı, spec §2.3) bu iki test FİZİK ÇIPASI DEĞİLDİR.
  Kovuk bir ŞEMA ÇIPASIDIR (`eps_r != 1`li): başlangıç alanı ayrık özvektör
  olarak VERİLİR, ölçülen `omega` 2B Yee'nin ayrık özdeğeridir;
  `n = sqrt(eps_r)` ölçeği bir girdinin ÇEVİRİSİDİR. Fresnel `|r|` ve
  işaret bir ÇEVİRİ DENETİMİDİR, `r_p = -r_s` fazı bir s/p ŞEMA TUTARLILIĞI
  kilididir. Modülün "girdiden geri okunamayan" fizik çıpaları (Bessel
  oranları, Rayleigh üssü, tek yarık, TİY faz farkı) yavaş şeritte kalıyor;
  bu dosya onların YERİNİ TUTMAZ, hızlı şeride kapalı-form referanslı bir
  ısırık taşır. "Şema" sözcüğü eş seslidir: G7'ninki veri/metin şeması,
  Modül 5'inki Yee SAYISAL şemasıdır.

Bu dosyadaki üç koşum (bir kovuk, iki Fresnel parametresi) `slow`
İŞARETSİZDİR ve bu ÖLÇÜLDÜ (aşağıdaki dokstringlerde süreler). Özel (`_ad`)
import YOK: kovuk kamu yolu `run_2d(state=..., probes=...)`dan geçer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qlab.engine.em.boundary import PEC
from qlab.engine.em.grid import FieldState2D
from qlab.engine.em.medium import Medium
from qlab.engine.em.yee import run_2d
from qlab.observable.em_obs import ObliqueWindow, reflection_coefficient

# ---------------------------------------------------------------------------
# ÇIPA 1 — DİELEKTRİK DOLU DİKDÖRTGEN KOVUK
# ---------------------------------------------------------------------------
NX, NY, S_KOVUK, ADIM_KOVUK = 41, 31, 0.5, 100
EPS_LER = (1.0, 2.25, 4.0)
KIPLER = ((1, 1), (2, 1), (1, 2))
#: Nokta prob. Üç kipin hiçbirinin düğüm çizgisinde değil: `sin(m pi 7/40)`
#: ve `sin(n pi 5/30)` m, n = 1, 2 için sıfırdan uzak.
PROB_KOVUK = (7, 5)


def _kovuk_omega(eps_r: float, m: int, n: int) -> tuple[float, float]:
    """KAMU YOLU `run_2d` ile (m, n) kipinin `omega`sı (+ yineleme artığı).

    Başlangıç alanı `sin(kx i) sin(ky j)`, `kx = m pi/(NX-1)`, `ky = n pi/(NY-1)`:
    dört kenarı PEC olan ızgarada beş noktalı Laplasyenin TAM özvektörü, yani
    her düğümün serisi tek frekanslıdır ve `omega` nokta probun üç terimli
    yinelemesinden okunur (`s[t+1] + s[t-1] = 2 cos(omega S) s[t]`). Hiçbir
    dispersiyon formülü kullanılmaz. İkinci dönüş değeri yinelemenin bağıl
    artığıdır (pencere kapısı: tek frekans değilse ölçüm geçersiz).

    Kaynak YOK (`source = None`). `run_2d`nin `Window`u `wavelength` ve
    `sgf`yi zorunlu alan olarak ister ama kaynaksız koşumda ikisi de hesaba
    GİRMEZ; `nan` "tanımsız" beyanıdır, uydurulmuş bir düğme değeri değil.
    """
    kx, ky = math.pi * m / (NX - 1), math.pi * n / (NY - 1)
    ozvektor = np.outer(np.sin(kx * np.arange(NX)), np.sin(ky * np.arange(NY)))
    kayit = run_2d(
        Medium.uniform_2d(NX, NY, eps_r=eps_r),
        courant=S_KOVUK,
        boundary=PEC(),
        steps=ADIM_KOVUK,
        source=None,
        source_index=None,
        probes=(PROB_KOVUK,),
        wavelength=math.nan,
        sgf=math.nan,
        state=FieldState2D(ozvektor, np.zeros((NX, NY - 1)), np.zeros((NX - 1, NY))),
    )
    seri = kayit.probes[0]
    y, x = seri[2:] + seri[:-2], 2.0 * seri[1:-1]
    c = float(np.dot(x, y) / np.dot(x, x))
    artik = float(np.max(np.abs(y - c * x)) / np.max(np.abs(y)))
    return float(np.arccos(c) / S_KOVUK), artik


def test_DIELEKTRIK_DOLU_DIKDORTGEN_KOVUK_DERS_KITABI_FREKANSINDA():
    """SINIF: G7 KAPALI-FORM ÇIPASI; Modül 5 sözlüğünde ŞEMA ÇIPASI (eps'li).

    Referans ders kitabı kapalı formu, deponun çıktısı değil. Dikdörtgen PEC
    kovuğun TM_mn kipleri (Ez; ör. Pozar, Jackson):

        omega_mn = (c / sqrt(eps_r)) * pi * sqrt((m/a)^2 + (n/b)^2)

    `a = (NX-1) dx = 40`, `b = (NY-1) dx = 30`, `c = 1`. İki yasa birden
    sınanıyor: (i) kovuğun kip ailesi (`sqrt((m/a)^2+(n/b)^2)`, kare
    kutudaki `sqrt(m^2+n^2)` ailesinin dikdörtgen genellemesi; `a != b`
    olduğu için (2,1) ile (1,2) AYRI frekanslardır), (ii) dielektrikte dalga
    hızı `c/n`, `n = sqrt(eps_r)` — yani aynı kipin frekans oranı
    `omega(1)/omega(eps_r) = sqrt(eps_r)`.

    MODÜL 5 SÖZLÜĞÜNDE NEDEN FİZİK ÇIPASI DEĞİL (G7 denetimi, bulgu 2): kip
    şekli başlangıç alanı olarak VERİLİR, yani (i) girdidir; ölçülen sayı
    Yee'nin ayrık özdeğeridir — `test_bessel.py::test_2B_DISPERSIYON_..._24C`
    testinin `eps_r != 1` eşi. (ii)'deki `n` de girdidir (spec §2.1/§2.3'ün
    Fresnel için söylediğinin aynısı). Çıpanın G7'deki işi bu yüzden "doğa
    böyle mi" değil, "2B dielektrik katsayısı ders kitabının `c/n`sini
    veriyor mu"dur — ve M1'i öldüren tam olarak budur.

    ÖZEL İMPORT YOK (G7 denetimi, bulgu 6): ölçüm kamu yolu `run_2d`den
    nokta probla yapılır. Eski sürüm `_adim_2d` + `sum(e * proj)` izdüşümü
    kullanıyordu (`test_bessel._duran_kip_2d`nin yolu); iki yol dokuz
    koşumda bağıl 1.1e-13 içinde aynı `omega`yı veriyor (ölçüldü).

    PENCERE: 41 x 31 düğüm, S = 0.5, 100 adım, prob (7, 5), eps_r = 1 /
    2.25 / 4, kipler (1,1) (2,1) (1,2). ÖLÇÜLDÜ (09.10.2026, düzeltme turu,
    `g7-olcum/duzeltme/em/olc.py`; 100 ve 200 adım bağıl 8.4e-14 içinde
    AYNI sayıyı veriyor, çünkü seri tek frekanslı — yineleme artığı iki
    uzunlukta da <= 1.4e-15, kapı 1e-12):

        eps_r   (1,1) bagil     (2,1) bagil     (1,2) bagil
        1.00    -2.0654e-04     -4.8139e-04     -1.1137e-03
        2.25    -3.0566e-04     -6.8742e-04     -1.4024e-03
        4.00    -3.4033e-04     -7.5947e-04     -1.5033e-03

        oran omega(1)/omega(eps_r):  2.25 -> 1.500149 / 1.500309 / 1.500434
                                     4.00 -> 2.000268 / 2.000557 / 2.000780

        (1,2)/(2,1) frekans oranı: 1.1840906 (ders kitabı 1.1848401),
        bağıl -6.33e-04

    Bu sayılar bağımsız iki yoldan da üretildi (G7 denetimi,
    `denetim/em/bagimsiz_kovuk.py`: elle türetilmiş ayrık dispersiyon ve
    qlab'ı import etmeyen numpy Yee; fark <= 1.1e-13). Son hanelerde BLAS
    çekirdeğine bağlı oynama var (kamu yolunda OPENBLAS_CORETYPE Haswell /
    Sandybridge'de göreli <= 7.1e-14, 9 sayının 5'i bit bit; ölçüldü), bu
    yüzden hiçbir assert bit düzeyinde değil.

    İŞARET HEP NEGATİF ve bu bir varsayım değil, kapalı formdan gelir: 2B
    Yee'de `24 C(theta) = cos^4 + sin^4 - S_eff^2` (hızlı şeritte
    `test_bessel.py::test_2B_DISPERSIYON_...` ölçüyor) ve burada
    `S_eff = S/sqrt(eps_r) <= 0.5`, `cos^4+sin^4 >= 0.5` olduğu için
    `C > 0`: ızgara dalgayı YAVAŞLATIR, ölçülen frekans ders kitabının
    ALTINDA kalmak zorundadır. Sapmanın büyüklüğü `(k dx)^2` ile gider
    (kare ızgarada, aynı kamu yoluyla, N = 21/41/81'de (1,1) için
    -5.146e-04 / -1.285e-04 / -3.213e-05, ardışık oranlar 4.003 / 4.001).

    TOLERANS: bağıl 5e-3 — ölçülen en büyük sapma 1.503e-3, pay 3.3 kat
    (ızgara dispersiyonu; daha sıkı bir kapı ızgarayı büyütmeyi, yani
    maliyeti isterdi). `sqrt(eps_r)` oranı için 2e-3 — ölçülen en büyük
    3.90e-4, pay 5.1 kat. (1,2)/(2,1) oranı için 2e-3 — ölçülen 6.33e-4,
    pay 3.2 kat.

    KIRMA (mutasyon, düzeltme turunda yeniden ölçüldü): M1
    (`yee_coefficients_2d`de `eps = sqrt(eps_r)`) eps_r = 2.25'te
    +2.23e-01 ile +2.24e-01, eps_r = 4'te +4.12e-01 ile +4.14e-01 bağıl
    sapma veriyor (5e-3 kapısının 45-83 katı, ve işaret YANLIŞ); (1,1)
    oranı 1.2248 / 1.4143'e iniyor. eps_r = 1 satırı mutantta BİT BİT aynı
    kalıyor (3/3, ölçüldü) — tam olarak hızlı şeridin neden kör olduğunun
    gösterimi.

    DÜRÜST OLUMSUZ: kovuk `eps` ile `mu`yu AYIRT ETMEZ. `uniform_2d`de
    `eps_r`yi `mu_r`ye yazan mutant (N3, denetimde) dokuz frekansı yuvarlama
    düzeyinde aynı bırakıyor (en büyük bağıl fark 1.1e-13, 7/9 bit bit;
    ölçüldü); düzgün dolu kovukta frekans yalnız `n = sqrt(eps mu)`ya bağlı
    olduğu için bu fiziktir, kusur değil, ve N3'ün yaşaması doğrudur.
    `eps`/`mu` yönü ÇIPA 2'nin işaret kapısında sınanıyor (yalnız 1B
    yarı-uzay kurucusu için; `uniform_2d` için değil).

    MALİYET: dokuz koşum toplam 0.015-0.017 s (ölçüldü, 7 tekrar).
    """
    olculen = {}
    for eps_r in EPS_LER:
        for m, n in KIPLER:
            omega, artik = _kovuk_omega(eps_r, m, n)
            assert artik < 1e-12, f"seri tek frekanslı değil (eps_r={eps_r}, {m},{n})"
            tam = math.pi * math.hypot(m / (NX - 1), n / (NY - 1)) / math.sqrt(eps_r)
            bagil = omega / tam - 1.0
            assert -5e-3 < bagil < 0.0, (eps_r, m, n, bagil)
            olculen[(eps_r, m, n)] = omega
    # n = sqrt(eps_r): aynı kipin frekansı tam sqrt(eps_r) kat düşüyor.
    for eps_r in EPS_LER[1:]:
        for m, n in KIPLER:
            oran = olculen[(1.0, m, n)] / olculen[(eps_r, m, n)]
            assert oran == pytest.approx(math.sqrt(eps_r), rel=2e-3), (eps_r, m, n, oran)
    # Kip ailesi dikdörtgen: (2,1) ile (1,2) a != b olduğu için AYRI frekans.
    assert olculen[(1.0, 1, 2)] / olculen[(1.0, 2, 1)] == pytest.approx(
        math.hypot(1 / 40, 2 / 30) / math.hypot(2 / 40, 1 / 30), rel=2e-3
    )


# ---------------------------------------------------------------------------
# ÇIPA 2 — DİK GELİŞ FRESNEL
# ---------------------------------------------------------------------------
#: Ucuz pencere: sevk penceresinin (3600 hücre, sgf 9, lambda 40) küçültülmüşü.
#: Dik gelişte `ky = 0`dır, yani `source.gauss_cos`ta beyan edilen geniş-bant
#: açı karışması arızası BURADA YOKTUR ve küçük `sgf` serbesttir. HER ALAN
#: AÇIKÇA YAZILIR (G7 denetimi, bulgu 4): `ObliqueWindow` varsayılanlarından
#: biri değişirse bu çıpa sessizce başka bir pencereyi ölçmesin. Değerler
#: bugünkü varsayılanlarla aynıdır (`courant`, `probe_offset`, `window_fn`,
#: `ey_kural`, `unity_gate`, `pml_m`, `tail_gate`, `echo_gate`), yani ölçülen sayılar
#: değişmedi. `unity_gate` artık KULLANILIYOR: ayna koşumunda `||m| - 1|`
#: kapısıdır. `tail_gate` (kuyruk kapısı, 10.10.2026): bu penceredeki
#: çağrıların kaydın son periyodundaki genliği tepenin <= 6.05e-06'sı,
#: eşiğin ~260 kat altında. `echo_gate` (yankı kapısı, RC-a, 10.10.2026):
#: gelen kaydında ana darbeden sonra kalanın spektral ağırlığı <= 1.81e-07,
#: eşiğin ~2500 kat altında.
FRESNEL_PENCERE = ObliqueWindow(
    cells=500,
    npml=40,
    courant=0.4,
    interface=250,
    source_index=90,
    probe_offset=6,
    sgf=1.5,
    window_fn="dikdortgen",
    ey_kural="aritmetik",
    unity_gate=1e-2,
    pml_m=3.0,
    tail_gate=1.6e-3,
    echo_gate=4.5e-4,
)
FRESNEL_LAMBDA = 20.0


@pytest.mark.parametrize("n2", [1.5, 2.0])
def test_DIK_GELIS_FRESNEL_KATSAYISI_ve_r_p_EKSI_r_s(n2):
    """SINIF: G7 KAPALI-FORM ÇIPASI; Modül 5'te ÇEVİRİ DENETİMİ + s/p ŞEMA KİLİDİ.

    Referanslar dik geliş Fresnel kapalı formu ve PEC aynasının `r = -1`i,
    ikisi de testte elle. Vakumdan (n1 = 1) n2 ortamına dik gelişte

        r_s = (n1 - n2) / (n1 + n2)   cam (n2 = 1.5): r = -0.2, R = %4
                                      n2 = 2       : r = -1/3

    (E alanı için; dielektrik daha düşük empedanslıdır, `Z2 = Z0/n2`, ve
    yansıyan E PEC'teki gibi İŞARET DEĞİŞTİRİR). Dik gelişte s ile p
    fiziksel olarak AYIRT EDİLEMEZ: Fresnel'in iki formülü `r_p = -r_s` verir
    (p'de kaydedilen alan Hz'dir; işaret yalnız uzlaşım), yani
    `arg(r_p / r_s) = 180°`. Çözücüde ise iki kutuplanma FARKLI
    ayrıklaştırmadan geçer — s'te Ez tam hücrede `eps_r` ile, p'de Ey yarım
    hücrede `eps_yari` (`ey_kural`) ile eşlenir. Faz eşitliği bu yüzden boş
    bir özdeşlik değil: iki ayrıklaştırmanın arayüzü BİRBİRİNE GÖRE aynı yere
    koyduğunun ve `r_p / r_s` oranını reel (-1) bıraktığının ölçüsüdür. Tek
    başına `r`nin işaretini GÖRMEZ (aşağıda N2).

    MODÜL 5 SÖZLÜĞÜ (G7 denetimi, bulgu 2): `|r|` bir ÇEVİRİ DENETİMİDİR,
    FİZİK ÇIPASI DEĞİL — `n` girdidir, `r` onun cebiri (spec §2.3;
    `test_arayuz.py` başlığı, `test_arayuz.py::test_FRESNEL_R_CEVIRI_...`,
    `em_obs.reflection_coefficient`). İşaret de aynı cebirin parçasıdır.
    `test_arayuz.py` başlığı dik gelişte `delta_ps`i fizik içeriği
    bakımından BİLGİSİZ sayar ve bu doğrudur: burada faz yeni bir fizik
    ölçmüyor, iki ayrıklaştırmanın tutarlılığını kilitliyor — bir s/p ŞEMA
    TUTARLILIĞI kilidi (M4/M5/N4/N5 aşağıda ölçüldü).

    İŞARET KAPISI (G7 denetimi, bulgu 1): `|r|` empedans yönüne fiziksel
    olarak KÖRDÜR (`|Z-1|/(Z+1)`, `Z <-> 1/Z` altında değişmez) ve
    `r_p / r_s` de öyle; yarı-uzay kurucusunda `eps` ile `mu` yer
    değiştirince dielektrik manyetik ortama döner, r_s ile r_p yer
    değiştirir ve `|r|` ile faz assert'leri yeşil kalır. İşareti okumak için AYNI
    pencerenin malzemesiz aynası (`mirror=True`; s'te PEC, `m_s = -1`
    ders kitabı) referans alınır: `r_s < 0` ve `m_s = -1` olduğu için
    `r_s / m_s > 0`, yani `|arg(r_s / m_s)| < 90°`. Ayna `eps2`yi KULLANMAZ
    ve `Medium.half_space`ten geçmez, yani yarı-uzay mutantlarından
    etkilenmez. p için ikinci bir ayna ÖDENMEZ: p'nin aynası (PMC) s'ninkiyle
    bu makinede BİT BİT aynı çıkıyor (`|m_p - m_s| = 0`, ölçüldü; sebebi
    `em_obs.tir_phase_difference` dokstringinde) ve `r_p`nin işareti
    `r_p = -r_s` kilidinden gelir.

    Kapı bir İŞARET kapısıdır, konum kilidi DEĞİL. Ölçülen +18.0629° bir
    kapalı formla AÇIKLANIYOR: Ez arayüzü `interface - 1/2`de (eps düğüm başına
    atanır), ayna `interface` düğümünde; aradaki yarım hücre gidiş-dönüşte
    BİR hücre eder ve 1B Yee'nin sayısal dalga sayısı
    `k = 2 asin(sin(pi S/lambda)/S) = 0.3152565 rad = 18.06287°`
    (lambda = 20, S = 0.4); ölçülen faz bundan 1.3e-06° sapıyor (faz
    kalıntısıyla aynı mertebe). Arayüzü 1 hücre kaydıran N1 fazı
    -18.063°'ye çevirir ve GEÇER — konum `arg`dan okunmaz, çünkü `arg` bir
    GERİ OKUMADIR (`em_obs.tir_phase_difference`, "ÜÇ KÖRLÜK" 1). Kapı
    90°'de: normalden (+18.06°) 71.94° içeride, N2'den (-161.94°) 71.94°
    dışarıda.

    PENCERE: `reflection_coefficient` (iki FDTD koşumunun farkı) +
    aynı pencerenin ayna koşumu, 500 hücre, arayüz 250, kaynak 90, prob
    arayüzden 6 hücre önce, npml 40 (m = 3), S = 0.4, `sgf` = 1.5,
    lambda = 20, dikdörtgen FFT, `ey_kural` aritmetik, `||m|-1|` kapısı
    1e-2 (hepsi `FRESNEL_PENCERE`de yazılı). ÖLÇÜLDÜ (09.10.2026, düzeltme
    turu, `g7-olcum/duzeltme/em/olc.py`):

        n2     |r_s|/|r| - 1   |r_p|/|r| - 1   arg(r_p/r_s)    arg(r_s/m_s)
        1.5    +3.8445e-02     -3.8568e-02     179.9999987°    +18.0629°
        2.0    +5.2196e-02     -5.2886e-02     179.9999988°    +18.0629°

    (`r_s`, `r_p` ve iki ayna OpenBLAS varsayılan / Haswell / Sandybridge
    çekirdeklerinde bit bit aynı çıktı; yine de hiçbir assert bit düzeyinde
    değil.)

    `|r|` sapması İKİNCİ MERTEBEDENDİR (ızgara dispersiyonu): aynı ölçüt
    lambda = 40 penceresinde (1200 hücre, arayüz 600, kaynak 200, `sgf` = 2,
    npml 60; öteki alanlar `FRESNEL_PENCERE` ile aynı) n2 = 1.5 için
    +9.340e-03 / -9.347e-03, yani lambda yarılanınca 4.12 / 4.13 kat
    (s / p). Daha sıkı bir `|r|` kapısı lambda = 40 penceresini, yani üç
    kat maliyeti isterdi.

    TOLERANS: `|r|` için bağıl 0.12 — ölçülen en büyük 5.29e-02, pay 2.3
    kat (n2 = 1.5'te 3.1 kat). Faz için 1e-3° bir ÜST SINIRDIR, değer
    çapası değil: ölçülen 1.31e-06° / 1.21e-06° (n2 = 1.5 / 2) bir sıfırın
    PENCEREYE BAĞLI KESME KALINTISIDIR, float64 yuvarlaması DEĞİL
    (1.31e-06° = 2.3e-08 rad, yuvarlamanın ~1e-15'inin yedi mertebe
    üstünde; G7 denetimi, bulgu 3). Kalıntı pencere düğmeleriyle oynuyor
    (n2 = 1.5, ölçüldü): npml 40/60/80 -> 1.31e-06 / 1.10e-06 / 7.15e-07°;
    `pml_m` 4 -> 3.53e-06°; `sgf` 2 -> 8.89e-08°; prob ofseti 2/20 ->
    9.25e-08 / 9.12e-07° (npml 80 için ızgara 540/270/110, `sgf` 2 için
    600/300/100). Yalnız FFT penceresi Hann yapılınca 2.02e-01°'ye
    fırlıyor (`|r_p|` sapması -1.04e-01) — `window_fn`in açıkça yazılması
    bu yüzden. Değerini çapalamak pencerenin kalıntısını ölçüt yapmak
    olurdu; kapı kalıntının 764 katında, en yakın ölçülen mutantın (N5,
    2.145e-02°) 21 katı altında.

    KIRMA (mutasyon, düzeltme turunda yeniden ölçüldü):
      * M2 (`yee_coefficients`te 1B `eps = sqrt(eps_r)`): `|r_s|` -4.79e-01
        (n2 = 1.5) / -4.67e-01 (n2 = 2) -> `|r|` kapısının ~4 katı dışında.
        (p dalı `eps_yari`yi doğrudan ortamdan okuduğu için o kutuplanmada
        `|r_p|` değişmiyor — iki kutuplanmanın BİRLİKTE sınanması bu yüzden.)
      * M5 (`medium._yari_hucre` aritmetik -> geometrik ortalama) ve N4
        (aritmetik -> harmonik): `arg(r_p/r_s)` 180°'den 3.85° / 6.66° (M5,
        n2 = 1.5 / 2) ve 7.35° / 11.64° (N4) sapıyor; `|r_p|` kapının İÇİNDE
        kalıyor (M5 -3.54e-02 / -4.04e-02, N4 -2.67e-02 / -1.31e-02). Bu
        mutantları daha önce hızlı şeritte yalnız `test_EGIK_P_KESIN_KAPI`
        (kararlılık eşiği modeli) ve iki transkripsiyon kilidi düşürüyordu.
      * M4 (1B `yee_coefficients`te manyetik kayıp eşleşmesi `d_h = 0`) ve
        N5 (p dalında `sig_y = 0`, M4'ün ikizi): `|r|` kapının içinde ama
        `arg(r_p/r_s)` 180°'den 2.152e-02° / 2.153e-02° (M4) ve 2.147e-02° /
        2.145e-02° (N5) sapıyor (kapının 21 katı) — eşleşmemiş PML'in
        yansıması yalnız bir dala giriyor.
      * N2 (`Medium.half_space`te `eps` ile `mu` takası): `|r|` ve
        `arg(r_p/r_s)` yeşil — r_s ile r_p yer değiştiriyor
        (`|r_s|/|r| - 1` = -3.8568e-02, `|r_p|/|r| - 1` = +3.8445e-02, yani
        normaldeki çiftin tersi; `arg(r_s)` -18.69° -> +161.31°) — ama işaret
        kapısı `arg(r_s/m_s)` = -161.94° veriyor -> DÜŞÜYOR. Bu mutantı
        önceden hızlı şeritte yalnız iki transkripsiyon/girdi testi
        düşürüyordu (`test_yee.py::test_AYNI_DIZI_FARKLI_KURULUS_...`,
        `test_MEDIUM_GIRDI_DENETIMLERI`); düzeltme turunda -x'siz koşuda
        hızlı şeritte düşen dört test bu ikisi ve bu testin iki parametresi
        (ölçüldü, `mut_n2_hizli_serit_tum.log`).
      * DÜRÜST OLUMSUZ — N1 (arayüz 1 hücre kayması) YAŞIYOR ve yaşaması
        doğru: yukarıda, konum bu çıpanın iddiası değil.

    MALİYET: parametre başına altı 1B FDTD koşumu, her biri 1200 adım
    (`r_s`, `r_p` ve ayna için birer ön + arka koşum; aynanın önü 251
    hücre), 0.105-0.113 s (ölçüldü, 5 tekrar); işaret kapısının payı
    0.034-0.040 s.
    """
    ortak = dict(
        eps1=1.0,
        eps2=n2 * n2,
        theta_deg=0.0,
        wavelength=FRESNEL_LAMBDA,
        window=FRESNEL_PENCERE,
    )
    tam = (1.0 - n2) / (1.0 + n2)
    r = {pol: reflection_coefficient(polarization=pol, **ortak) for pol in ("s", "p")}
    assert r["s"] is not None and r["p"] is not None
    for pol in ("s", "p"):
        assert abs(r[pol]) == pytest.approx(abs(tam), rel=0.12), (pol, abs(r[pol]))
        assert abs(r[pol]) < 1.0
    # r_p = -r_s: arg(-r_p/r_s) = 0 (kalıntı pencereye bağlı kesme, yuvarlama değil).
    sapma = abs(float(np.angle(-r["p"] / r["s"], deg=True)))
    assert sapma < 1e-3, f"arg(r_p/r_s) 180°'den {sapma:.3e}° sapıyor"
    # İŞARET: tam < 0 ve PEC aynası m_s = -1 -> r_s/m_s > 0, yani |arg| < 90°.
    ayna = reflection_coefficient(polarization="s", mirror=True, **ortak)
    assert ayna is not None, "ayna koşumu ||m|-1| kapısına takıldı"
    isaret = float(np.angle(r["s"] / ayna, deg=True))
    assert abs(isaret) < 90.0, (
        f"arg(r_s/m_s) = {isaret:+.3f}°: r_s PEC aynasıyla ZIT işaretli "
        f"(beklenen r_s = {tam:+.4f} < 0, dielektrik E'yi ters çevirir)"
    )
