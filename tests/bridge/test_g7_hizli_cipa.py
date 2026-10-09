"""G7 — `bridge` hızlı şeridine BAĞIMSIZ fizik çıpası: kapalı formdaki Morse frekansı.

NEDEN BU DOSYA VAR (ölçüldü, 09.10.2026). Hızlı şeritte analitik çıpa
VAR — `TAM/√2` (yanlış ikiz), `√1.001 − 1` (kuvvet yasası),
`(ω·dt)⁴/144` (Verlet kapalı formu), kütle ×4 → tam 0.5, harmonikte sıfır
eğim — ama BEŞİ DE `a = 0` HARMONİK yüzeyde koşar. Harmonik osilatör
eş-zamanlıdır (frekans genliğe bağlı değildir), yani bu çıpaların hiçbiri
`A²` ekstrapolasyonunu, yani A2 "hata çubuğu"nu üreten ANHARMONİK yolu
GÖREMEZ. O yolu hızlı şeritte yalnız gerileme sayıları kilitliyordu
(borunun kendi önceki çıktısına `rel=1e-9`); yavaş şerit HF borusunu koşar
ve ikize yalnız "Morse eğimi HF eğimine `< 2e-4` yakın" üzerinden dokunur —
o da bağımsız bir referans değil, iki uygulamanın uyumudur.

REFERANS (bağımsız, kapalı form): klasik Morse osilatörü
`V = D_e (1 − e^{−a x})²` için frekans enerjinin kapalı bir fonksiyonudur,

    ω(E) = a · √(2 (D_e − E) / μ)

(periyot integrali `T = 2∫dx/v`, `y = e^{−a x}` değişkeniyle elemanter
olur: `T = (2π/a)·√(μ / (2(D_e − E)))`). Dondurulmuş yüzeyde
`D_e = K/(2a²)` olduğu için `ω(0) = √(K/μ)` ve `x = A` genliğinden
durgun bırakılan bağda `E = D_e (1 − e^{−aA})²`, yani

    ω(A) = ω₀ · √(1 − (1 − e^{−aA})²),   ω₀ = √(K/μ).

Formül bu iş için AYRICA sayısal olarak doğrulandı (iş dizini
`.superpowers/sdd/g7-olcum/bridge/kapali_form_dogrula.py`, Gauss–Chebyshev
periyot integrali, a = 0.9137 ve 0.5, üç genlik): bağıl fark ≤ 2.1e-12.
Denetim ayrıca indirgenmiş kütle VARSAYMAYAN iki cisimli ODE ile (DOP853,
`.superpowers/sdd/g7-olcum/denetim/bridge/bagimsiz_referans.py`) bağıl
≤ 1.9e-14 buldu.

`μ` BURADA src'den OKUNMAZ: CODATA 2018 `m_u/m_e` ve ¹H kütlesinden
(1.00782503207 u) yeniden kurulur — `MU_H2`yi bozan bir mutant iki tarafı
aynı yönde bozamasın diye.

PENCERE TESTİNDİR: `N = 200`, 6 periyot, genlikler (0.01, 0.02, 0.04) bu
dosyada sabittir ve boruya AÇIKÇA geçirilir; bugün src'nin sevk
penceresiyle (`N_BOLME`, `PERIYOT_SAYISI`, `GENLIKLER`) aynıdır ve açık
geçirmek sayıları bit bit değiştirmez (ölçüldü). Neden: tolerans BU
pencereden türetilir (`TOLERANS_CM`) ve faz zarfı `N⁻³/(P−1)` ile
ölçeklenir; sevk penceresini değiştiren meşru bir src değişikliği (ör.
`N_BOLME = 100`) bu dosyayı yazı-tura ile kırmamalı. `h = 0.002`,
`dt = (2π/ω_A)/N`; ikiz `a = A_ESLESEN` (0.9137). Kapalı formun ÜÇ değeri
aynı `A²` doğrusuna `np.polyfit` ile oturtulur; karşılaştırılan şey borunun
ölçtüğü frekanslar değil, AYNI uydurmanın iki girdisidir (biri MD + sıfır
geçişi + dispersiyon tersi, öteki kapalı form). Uydurmanın kendi yanlılığı
iki tarafta aynıdır ve ikinci testte TAM OLARAK o yanlılık ölçülür.

PAYLAŞILAN GİRDİLER — bu çıpa onlara KÖRDÜR: `K_HF`, `A_ESLESEN` ve
`HARTREE_TO_WAVENUMBER` src'den okunur ve iki tarafa BİRLİKTE girer;
pencere genlikleri de iki tarafta aynıdır. `K_HF` +%1.7 bu dosyaya karşı
YAŞAR (denetim Y6, kesişim 4.5e-06 cm⁻¹); `K_HF`yi bağımsız sınamak HF
ister (yavaş şerit), hızlı şeritte onu yalnız gerileme literali
`test_bridge_senaryo.py::test_6_3` öldürür. `A_ESLESEN` yalnız aşağı
yönde, `egim_t < −0.01` ile kısmen korunur. `HARTREE_TO_WAVENUMBER`ın
kendi literatür çıpası `tests/test_units.py`dedir (CODATA tablo değeri,
`rel = 1e-10`). İntegratörün KİMLİĞİ de çıpalanmaz: hız-Verlet yerine
simplektik Euler (denetim Y5) aynı konum özyinelemesini verir, frekans
ona kördür; onu `tests/md/test_integrator.py` öldürür.

ÖZEL AD İMPORTLARI (gerekçe): `_a2_uydurmasi` (eğim, kesişim) çiftini TEK
koşumda verir; kamu `omega_b` + `a2_egimi` bit bit aynı sayıları verir ama
boruyu iki kez koşturur (süre ×2). `_donmus_yuzey`in kamu eşdeğeri yoktur
ve M5'in (`D_e` ×2) yeridir: test kendi `Morse`unu kursaydı M5'e kör
olurdu. Depo emsali: `tests/bridge/test_kenar_tabani.py` aynı adları alır.

PLATFORM: ÖLÇÜLDÜ sayıları OpenBLAS SkylakeX çekirdeğinde (bu makinenin
varsayılanı) alındı. Haswell/Zen/Sandybridge/Prescott çekirdeklerinde son
haneler değişir (denetim; düzeltmeden sonra Haswell, Zen, Prescott yeniden
ölçüldü, iki test de yeşil): kesişimde son ulp, A2 tabanında bağıl 7.6e-11
(5481 cm⁻¹'lik sinyalden 0.012 cm⁻¹'lik fark çıkarken tek ulp büyür). Bu
yüzden sayılar ≤ 10 anlamlı haneyle yazıldı; hiçbir assert o düzeyde
değildir.
"""

from __future__ import annotations

import math

import numpy as np

from qlab.observable.bridge_obs import (
    A_ESLESEN,
    K_HF,
    TAM_AU,
    TAM_CM,
    _a2_uydurmasi,
    _donmus_yuzey,
    edge_noise_floor,
)
from qlab.units import HARTREE_TO_WAVENUMBER

#: CODATA 2018: atomik kütle birimi ve elektron kütlesi (kg); ¹H atom kütlesi
#: (u). src'nin `units.py`sinden BAĞIMSIZ yazıldı.
_M_U_KG = 1.66053906660e-27
_M_E_KG = 9.1093837015e-31
_M_1H_U = 1.00782503207
MU_BAGIMSIZ = _M_1H_U * (_M_U_KG / _M_E_KG) / 2.0

#: Testin penceresi (modül dokstringi, PENCERE). Boruya açıkça geçirilir.
PENCERE_N = 200
PENCERE_PERIYOT = 6
PENCERE_GENLIKLER = (0.01, 0.02, 0.04)
_PENCERE = {
    "n_bolme": PENCERE_N,
    "periyot_sayisi": PENCERE_PERIYOT,
    "genlikler": PENCERE_GENLIKLER,
}

#: Tolerans, cm⁻¹: SIFIR GEÇİŞİ ARA DEĞERLEMESİNİN FAZ ZARFININ EN KÖTÜ
#: KÖŞESİ × 1.25. Kaynak: periyot aynı yöndeki geçişlerden okunur ve
#: `mean(diff)` teleskoplanır, yani yalnız İLK ve SON geçişin hatası kalır.
#: Geçişte `x'' = 0` ve `x''' = −ω₀² x'` (harmonikte ve Morse'ta aynı), o
#: yüzden doğrusal ara değerlemenin kök hatası, geçiş adımın içinde `α`
#: konumundayken `(ω dt)³/6 · α(1−α)(1−2α)` radyandır; `α(1−α)(1−2α)`nin
#: uç değerleri `±√3/18`. İlk ve son geçişin `α`ları bağımsız kayınca,
#: genlik başına
#:     |δω| ≤ ω₀ (ω₀ dt)³ √3 / (108 π (P − 1)) = 1.735e-04 cm⁻¹  (N=200, P=6).
#: Kesişim üç genliğin DOĞRUSAL birleşimidir, katsayılar (2/3, 1/2, −1/6),
#: `Σ|c| = 4/3`: köşe 2.314e-04 cm⁻¹, artı faz-ortalamalı sistematik
#: 4.5e-06. ÖLÇÜLDÜ (`.superpowers/sdd/g7-olcum/duzeltme/bridge/`,
#: `zarf_genlik.py`, `dt × (1+ε)`, |ε| ≤ 1e-2, 2001 nokta): genlik başına
#: faz parçası ≤ 1.77e-04 (formülün %2 üstü; uçlarda dt %1 büyük), köşe
#: 2.40e-04, tek boyutlu taramanın en büyüğü 1.07e-04. Test noktası
#: 4.48e-06 (2001 noktanın yalnız 114'ü ondan küçük); eski 3.4e-05
#: toleransı bu iyi fazda oturuyordu (1081/2001 nokta onu aşıyor). Faz
#: parçası pencereyle ölçeklenir (ölçülen köşe: N=300 7.4e-05, N=150
#: 6.0e-04, N=100 2.3e-03; P=4 4.0e-04, P=8 1.7e-04): PENCERE DEĞİŞİRSE
#: TOLERANS YENİDEN TÜRETİLİR.
TOLERANS_CM = 3.0e-4

#: `TAM_AU` ile bağımsız `ω₀`ın bağıl farkı. Ölçülen 0.0 (aynı CODATA yolu),
#: ama eşit derecede meşru μ yolları 1e-12'yi aşar (denetim): CODATA 2018
#: tablo oranı −2.3e-12, AME2020 ¹H −8.5e-11, CODATA 2022 −1.8e-11. 1e-9
#: bu baskı farklarını ≥ 11 kat payla kapsar ve μ×2'yi (−2.9e-01) 8 mertebe
#: payla öldürür.
TAM_BAGIL_TOL = 1e-9


def _kapali_form_dogrusu(a: float) -> tuple[float, float, float]:
    """Kapalı form `ω(A)`ları testin genliklerinde → `A²` doğrusu.

    Döndürür: `(eğim, kesişim, ω₀)`, atomik birimde.
    """
    w0 = math.sqrt(K_HF / MU_BAGIMSIZ)
    frekans = [
        w0 * math.sqrt(1.0 - (1.0 - math.exp(-a * A)) ** 2)
        for A in PENCERE_GENLIKLER
    ]
    egim, kesisim = np.polyfit([A * A for A in PENCERE_GENLIKLER], frekans, 1)
    return float(egim), float(kesisim), w0


def test_G7_MORSE_A_KARE_DOGRUSU_KAPALI_FORMLA_FAZ_ZARFI_ICINDE():
    """Anharmonik boru, kapalı formun Morse frekanslarını yeniden üretir.

    ÖLÇÜLDÜ (09.10.2026, `a = 0.9137`, test penceresi):
        kesişim  boru     0.02497437296 a.u.
                 kapalı   0.02497437298 a.u.
                 fark    −4.48e-06 cm⁻¹ (bağıl −8.2e-10; TOLERANS_CM'nin 0.015'i)
        eğim     boru    −0.01002301105
                 kapalı  −0.01002469592   (bağıl −1.68e-04)

    TOLERANS GEREKÇESİ: kesişim `TOLERANS_CM` içinde (modül sabiti: faz
    zarfının en kötü köşesi × 1.25). Eğim `rel = 1e-3`: aynı faz-torusu
    köşesi eğimde bağıl 2.30e-04 (ölçüldü, `egim_kose.py`; tek boyutlu
    taramada en büyük 1.79e-04), pay 4.3. Eğim kesişimden gevşektir çünkü
    dispersiyon tersi `ω = (2/dt)·sin(Ω dt/2)` yalnız harmonikte TAMDIR,
    anharmonik artığı genliğe bağlıdır ve eğim `ΔA² = 1.5e-03 a.u.²`ye
    bölündüğü için o artığı büyütür.

    KIRMA (mut.py, YALNIZ bu test; sondaki nota bakınız): dispersiyon
    tersinin silinmesi (M1), uydurmanın `A²` yerine `A`da yapılması (M2),
    sıfır geçişinin orta noktaya yuvarlanması (M3), başlangıç genliğinin ×2
    olması (M4), Morse `D_e`nin ×2 olması (M5), bağın sıkışmadan bırakılması
    (Y1), dinamik kütlede standart atom ağırlığı (Y2), Morse kuvvetinde
    anharmonik çarpanın silinmesi (Y3), Verlet'te konumun eski hızla
    güncellenmesi (Y4), kesişimin `TAM` etrafında yansıtılması (Y8),
    borunun `TAM` döndürmesi (Y9). M2, M4, M5'i hızlı şeridin mevcut analitik çıpaları
    GÖRMÜYORDU. M6 (μ×2) bu testte YAŞAR ve yaşaması doğrudur: dinamik
    kütleye dokunmaz, yalnız `dt`yi √2 büyütür (N≈141 penceresi), yani faz
    zarfı içinde bir örnekleme değişikliğidir; onu ikinci testin `TAM_AU`
    satırı öldürür.
    """
    egim, kesisim = _a2_uydurmasi(_donmus_yuzey(A_ESLESEN), **_PENCERE)
    egim_t, kesisim_t, _ = _kapali_form_dogrusu(A_ESLESEN)

    fark_cm = (kesisim - kesisim_t) * HARTREE_TO_WAVENUMBER
    assert abs(fark_cm) < TOLERANS_CM, (
        f"kesişim kapalı formdan {fark_cm:+.3e} cm⁻¹ uzak (tolerans {TOLERANS_CM:.1e})"
    )
    assert abs(egim / egim_t - 1.0) < 1e-3, (
        f"A²-eğimi kapalı formdan bağıl {egim / egim_t - 1.0:.3e} uzak"
    )
    # Çıpa boş değil: anharmoniklik gerçekten var ve eğimde görünüyor —
    # harmonik limitin (sıfır eğim) çok üstünde, analitik limit
    # `−ω₀a²/2 = −0.010425`in de %4 yakınında (uydurma sonlu genlikte).
    assert egim_t < -0.01


def test_G7_A2_HATA_CUBUGU_KAPALI_FORMUN_ISARETLI_A_KARE_YANLILIGIDIR():
    """A2 "hata çubuğu" gerileme sayısı değil, İŞARETİYLE ÖNGÖRÜLEBİLİR bir sayıdır.

    A2'nin sinyali borunun `ω_B`si, tabanı `|ω_B − TAM|`dır. İkiz Morse
    olduğu için `ω_B − TAM` TAMAMEN kapalı formdan hesaplanabilir: kapalı
    form frekanslarının `A²` doğrusunun kesişimi eksi `ω₀`, yani üç genlikli
    doğrusal ekstrapolasyonun Morse'taki YANLILIĞI. İşareti NEGATİFTİR
    (kesişim `ω₀`ın altında) ve İŞARETLİ karşılaştırılır: mutlak değer,
    kesişimi `TAM` etrafında yansıtan bir hatayı yutardı (denetim Y8: eski
    mutlak-değerli hali bu teste tek başına karşı yaşıyordu).

    BU KARŞILAŞTIRMA BİRİNCİ TESTLE AYNI NİCELİKTİR: `TAM_AU` ile `ω₀`
    (bağıl `TAM_BAGIL_TOL` içinde) eşit olduğundan
    `(sinyal − TAM_CM) − öngörü = (kesişim − kesişim_kf)·H2W` (+ en çok
    5.5e-06 cm⁻¹). Ayrı bir fizik ölçümü DEĞİLDİR. Bu testin KENDİNE ÖZGÜ
    katkısı TESİSATTIR — `edge_noise_floor` → `newton_noise_floor` →
    `TAM_CM` → `sinyal`/`taban` yolu, işaret — ve `TAM_AU`nun indirgenmiş
    kütle denetimi.

    ÖLÇÜLDÜ (09.10.2026, test penceresi):
        sinyal − TAM_CM        −0.0120259923 cm⁻¹  (taban bunun mutlak değeri)
        kapalı form öngörüsü   −0.0120215086 cm⁻¹
        fark                   −4.48e-06 cm⁻¹ (birinci testteki kesişim farkı)
        TAM_AU/ω₀ − 1           0.0 (aynı aritmetik yol)

    `TAM_AU` İNDİRGENMİŞ KÜTLEYLE kurulmuş olmalı: `√(K/μ)`, `μ = m(¹H)/2`
    CODATA'dan bağımsız kuruldu. Bu bir yazım denetimi değil: `μ` yerine atom
    kütlesi yazılırsa ikiz-kesme sınavının cevabı √2 kayar ve A2 tabanı
    gerçek hatanın yerine bir kütle hatasını ölçer.

    TOLERANS GEREKÇESİ: A2 için `TOLERANS_CM` (birinci testle aynı nicelik,
    aynı faz zarfı); `TAM_AU` için `TAM_BAGIL_TOL` (modül sabiti).

    KIRMA (mut.py, YALNIZ bu test; sondaki nota bakınız): M1–M5 ve Y1–Y4
    A2 yolundan, M6 `TAM_AU` satırından, Y8 (kesişim `TAM` etrafında ayna)
    ve Y9 (boru `TAM` döndürüyor) işaretli satırdan.
    """
    _, kesisim_t, w0 = _kapali_form_dogrusu(A_ESLESEN)
    assert abs(TAM_AU / w0 - 1.0) < TAM_BAGIL_TOL, (
        f"TAM_AU indirgenmiş kütleli kapalı formdan bağıl {TAM_AU / w0 - 1.0:.3e} uzak"
    )

    ongoru_cm = (kesisim_t - w0) * HARTREE_TO_WAVENUMBER
    a2 = edge_noise_floor(tur="hata-cubugu", **_PENCERE)
    sapma_cm = a2.sinyal - TAM_CM
    assert abs(sapma_cm - ongoru_cm) < TOLERANS_CM, (
        f"A2 sinyali − TAM {sapma_cm:+.6e}, kapalı form öngörüsü {ongoru_cm:+.6e} cm⁻¹"
    )
    assert abs(a2.taban - abs(ongoru_cm)) < TOLERANS_CM, (
        f"A2 tabanı {a2.taban:.6e}, kapalı form öngörüsü |{ongoru_cm:.6e}| cm⁻¹"
    )
    # Öngörü NEGATİF ve toleransın ~40 katı: sınanan şey faz gürültüsü değil
    # yanlılık. Bu yüzden `ω_B ≈ TAM` döndüren bir boru yukarıdaki işaretli
    # satırda kırmızıdır (ölçüldü: Y9, 1.20e-02 cm⁻¹; sondaki not).
    assert ongoru_cm < -20.0 * TOLERANS_CM


# ---------------------------------------------------------------------------
# G7 MUTASYON ÖLÇÜMÜ (09.10.2026, düzeltmeden SONRA yeniden koşuldu; kayıtlar
# `.superpowers/sdd/g7-olcum/duzeltme/bridge/`). Araç
# `.superpowers/sdd/araclar/mut.py`. Mutantların hepsi src'de; M1–M6 ilk
# ajanın, Y1–Y8 denetçinin:
#   M1 dispersiyon tersi silindi      M2 `A²` uydurması `A`da
#   M3 sıfır geçişi orta noktaya      M4 başlangıç genliği ×2
#   M5 Morse `D_e` ×2                 M6 `MU_H2` = atom kütlesi (μ×2)
#   Y1 sıkışmadan bırakma             Y2 dinamikte standart atom ağırlığı
#   Y3 Morse kuvvetinde `ussel` yok   Y4 Verlet konumu eski hızla
#   Y8 kesişim `TAM` etrafında ayna   Y9 boru `ω_B ≈ TAM` döndürüyor (bu
#                                        düzeltmenin, işaretli satır için)
#
#   (a) mevcut hızlı şerit (169 test, bu dosya HARİÇ, -x)  M1–M6 öldü —
#       İLK kırmızı hep bir gerileme literali (`test_6_3`, panel sayıları).
#   (b) mevcut yavaş şerit (8 test, -x)          M1–M6 öldü (M5
#       `test_GERCEK_HATA`, ötekiler `test_6_5`) — ikisi de HF borusunun
#       kendi kaydına karşı.
#   (c) mevcut ANALİTİK çıpalar, gerileme assert'lerinden AYRILARAK (iş
#       dizinindeki sonda, yedi assert aynı toleransla): 3/6 — M2, M4, M5
#       HAYATTA. Sebep: yedisi de `a = 0` harmonik yüzeyde koşuyor.
#   (d) YALNIZ bu dosya: 12/12 öldü. Birinci test tek başına M6 dışındaki
#       11'ini, ikinci test tek başına 12'sini de öldürür. |Kesişim farkı|
#       cm⁻¹ (`TOLERANS_CM` = 3.0e-04): M1 2.25e-01 · M2 1.10e+00 ·
#       M3 2.08e-01 · M4 8.0e-02 · M5 2.27e+03 · Y1 2.52e-02 · Y2 3.13e-01 ·
#       Y3 1.17e-02 · Y4 1.30e+00 · Y8 2.4e-02 · Y9 1.20e-02; en zayıfı Y3,
#       toleransın 39 katı. M6 kesişimde 9.3e-05 (faz zarfı içinde),
#       `TAM_AU` satırında −2.9e-01. Y8'i ikinci testte YALNIZ işaretli satır
#       öldürür (taban satırında 4.5e-06: mutlak değer onu yutuyor).
#   (e) BİLEREK YAŞAYANLAR (bu dosyaya karşı): Y5 integratör kimliği, Y6
#       `K_HF` paylaşılan girdi (modül dokstringi), Y7 `N_BOLME = 100`
#       (pencere testindir; eskiden yazı-tura ile yalancı kırmızıydı).
#
# SÜRE (`--durations`, 09.10.2026, başka ajanlar eşzamanlı koşarken):
# birinci test 0.07 s, ikinci 0.14 s çağrı, kurulum < 0.005 s. `slow`
# işareti YOK: hızlı şeridin kuralına (saniye altı) uyuyor.
