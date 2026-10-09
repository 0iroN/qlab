"""G7 — `-m "not slow"` şeridine UCUZ FİZİK ÇIPALARI (Modül 6, yerçekimi).

ENVANTER MADDESİ G7 (orta): *"`-m not slow` hızlı şeridi bütün modüllerde
fizik çıpalarını atlıyor."* BU MODÜLDE İDDİA YANLIŞ — ölçüldü: hızlı şerit
(bu dosya hariç) 298 test, ~10–20 s (yüke bağlı; denetim düzeltmesinde
10.0–10.3 s) ve içinde bağımsız referansa karşı sınanan çok sayıda çıpa var
(`6π` perihel katsayısı iki ayarda, `2π(2+2γ−β)` beş PPN noktasında ELLE
yazılı çarpanlarla, bükülmede `1+γ` on beş PPN noktasında, `−πM²/b²`,
`15π/4`, `27π + 1.5πe²`, ISCO `6M`, foton küresi `3M`, `b_krit = 3√3 M`,
sekiz çıpası `9.237681250724`, dairesel çift `√5 − 1`). Altı mutantlık bir
turda BEŞİ hızlı şeritte öldü (denetim düzeltmesinde yeniden koşuldu: M1–M5
öldü, M6 298/298 geçti).

AMA İKİ BOŞLUK ÖLÇÜLDÜ. Bu dosya ikisini de kapatıyor — DAR bir kapsamla;
neyi KAPATMADIĞI aşağıda (KAPSAM) ve her testin dokstringinde yazılı:

  1. **N-body kuvvet üssü `p ≠ 2` zincirinde hızlı şerit KÖRDÜ.** Zincirin
     üç halkası var; üçü de bu dosya olmadan hızlı şeritte **298/298
     geçti** (mutant olarak ölçüldü):
       - `M6_NB_SEKIZ_BVP_POWER_YOK` — BVP'nin GİRDİSİ
         (`figure_eight_residual` içindeki `integrate(..., power=power)` →
         `power=2.0`): BVP `p` ne olursa olsun NEWTON sekizini buluyor;
       - `Y2_NB_ORNEKLEME_POWER_YOK` — ÖRNEKLEME (`sample_positions`ın akışı
         `power=2.0` ile koşuyor);
       - `Y9_NB_SEKIZ_POWER_ETIKETI_KAYIP` — BVP'nin ÇIKTISI: BVP doğru ama
         `FigureEight.power` etiketi `2.0`a düşüyor; `p = 2.1`de
         `eight_invariant` 6.778916 → 9.454268, `return_defect` 3.712e-08 →
         2.055e-01.
     Yavaş şeritte bunları yalnız
     `test_CIPA_YANILABILIR_KUVVET_USSUYLE_KAYIYOR...` öldürüyor — ve o test
     de deponun KENDİ önceki çıktısını (23.956761 / 6.778916 …) kilitliyor,
     bağımsız bir referansı değil.
     Sebep yapısal: hızlı şeritte `p ≠ 2` ile BVP çözen TEK test yok; `p ≠ 2`
     ile fizik koşan tek hızlı test (bir kapı testi dışında) enerji korunumu,
     o da BVP'den geçmiyor. Bu dosyanın İLK sürümü Y9'u KAÇIRIYORDU
     (örneklemeye testin kendi `p`sini veriyor, motorun taşıdığı `fe.power`ı
     hiç okumuyordu — denetim bulgusu); şimdi örnekleme `fe.power` ile koşuyor.
  2. **ODE yörünge motorunun hızlı şeritteki koruması bit kilitleri.**
     Mutant `M1_ODE_L2_KAYNAK_aP_DUSTU` (`M/L²`deki `−a·P` terimi düşüyor)
     hızlı şeritte 7 testte öldü, ama öldüren assert'ler ya regresyon
     sayısı (`phi[-1] == 6.283248140523368`, `c == 18.849556406088652` …) ya
     da TEK YÖNLÜ bir aralık sınavı (`u1 <= min(u)`). Tek yönlü sınav bu
     mutantı İŞARET şansıyla yakalıyor (mutant `u_min`'i AŞAĞI itiyor,
     `min(u)/u1 − 1 = −8.138e-06`; temizde örnekleme payı `+5.293e-07`).
     Afeli YUKARI iten ikiz artık ÖLÇÜLDÜ (denetim mutantı
     `Y1_ODE_AFEL_YUKARI_IKIZ`, `S² − P` → `S² − 2P`): `min(u)/u1 − 1 =
     +9.196e-06`, `u1 <= min(u)` GEÇİYOR. `kayma ≈ 6π/p` (rel 1e-4) ikisini
     de GÖRMÜYOR: `kayma/(6π/p) − 1` temiz `2.373e-05`, M1 `2.069e-05`, Y1
     `2.676e-05`. Y1'i hızlı şeritte M1'i öldüren AYNI 7 test öldürüyor;
     `test_RUN_TRAJECTORY_SOZLESMESI...`nde iki bit-dışı sınav (tek yönlü
     aralık, `6π/p`) Y1'de geçtiği için orada öldüren bit kilididir (diğer
     altısının hangi assert'le öldüğü tek tek ölçülmedi). Yani taban yeniden
     sayılırsa (bit kilitleri güncellenirse) bu hata sınıfı hızlı şeritte
     görünmez olurdu; bu dosya M1'i ve Y1'i iki yönlü afel sınavıyla öldürüyor.

KAPSAM — bu dosyanın KAPATMADIĞI (dürüst olumsuz): Binet çıpası ODE'nin
TAMAMINI korumaz. Yalnız `M/L²` ↔ sağ taraf tutarlılığını ve dönüş
noktalarının YERİNİ sınar; `Δφ`yi (açısal ölçek) ve kuram katsayısı `λ`yı
SINAMAZ. Denetim mutantları `Y5` (sağ taraf ×1.01, φ ölçeği) ve `Y8` (`λ`
tutarlı biçimde yarıya) bu dosyadan GEÇİYOR, ve geçmeleri doğru: ikisinde de
dönüş noktaları değişmiyor. Onları hızlı şeritte bağımsız `kayma ≈ 6π/p`
(rel 1e-4, `test_kapali_form.py`) öldürüyor — sapma Y5'te −4.963e+02, Y8'de
−5.000e-01. Bu dosya o sınavı TEKRARLAMIYOR.

ÇIPA TANIMI (G7): bağımsız referansa karşı sınama — analitik formül, kapalı
çözüm ya da TAM korunan bir nicelik. Bu dosyanın iki çıpası da öyle:

  * VİRYAL TEOREMİ (Landau–Lifshitz, *Mekanik* §10; Goldstein §3.4):
    potansiyel `k` dereceden homojense periyodik bir yörüngede
    `2⟨K⟩ = k⟨U⟩`. Kuvvet `1/r^p` ⇒ `U ∝ r^{1−p}`, `k = 1 − p`, ve
    `E = K + U` ile `⟨U⟩ = 2E/(3 − p)`. `p = 2`de bilinen `⟨U⟩ = 2E`.
  * BINET DENKLEMİNİN İLK İNTEGRALİ: `u'' + u = M/L² + λMu²` bir ikinci
    mertebe denklemdir ve `(u')² = (u − u1)(u2 − u)·w(u)` ilk integrali
    TAM korunur; kökleri yörüngenin bildirilen dönüş noktalarıdır,
    `u1 = (1−e)/p`, `u2 = (1+e)/p` (konik parametrizasyonu, kapalı form).
    Perihelden (`u = u2`, `u' = 0`) başlayan akış yarım radyal periyotta
    (`φ = Δφ/2`, yansıma simetrisi) TAM `u1`e varmak ZORUNDADIR.

Referans tarafında depo kodu YOK: potansiyel ve enerji test içinde elle
yazılı (`U = −Σ m_i m_j / ((p−1) r^{p−1})`, kuvvetin integrali — modül
dokstringinin beyanı), dönüş noktaları `(1∓e)/p`den.

PENCERE: `G = c = M = 1`, sekiz için `m = 1`, `p = 2.1`, BVP
`rtol = 1e-9`, `atol = 1e-11` (sevk edilen 1e-13 DEĞİL — yük altındaki
bütçe payı için; bedeli ve ölçümü viryal testinin BÜTÇE ve ÇÖZÜNÜRLÜK
SINIRI paragraflarında), örnekleme `sample_positions` sevk edilen penceresiyle
(DOP853, 1e-13/1e-15); ODE sevk edilen penceresiyle (DOP853, 1e-13/1e-15).

`slow` İŞARETSİZ. BÜTÇE: G7 hedefi test başına ≤ 0.3 s; ölçümler (önbellek
boş, `--durations`, eşzamanlı ajan yükü altında — sayılar şişik olabilir)
her testin dokstringinde. `test_nbody.py`nin yapısal ölçütüne de uyuyor:
BVP'yi BİR kez çözüyor (ölçüt "ikiden fazla").

İMPORT: yalnız kamu API'si (`run_trajectory`, `solve_figure_eight`,
`figure_eight_state`, `sample_positions`); `_ad` biçimli yardımcılar bu
dosyanın KENDİ referans tarafıdır, depodan alınmaz.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.grav.geodesic import run_trajectory
from qlab.engine.grav.nbody import (
    figure_eight_state,
    sample_positions,
    solve_figure_eight,
)

#: Viryal çıpasının kuvvet üssü. `p = 2` SEÇİLEMEZ: M6 mutantı Newton
#: noktasında görünmezdir (`power=2.0` ile `power=p` orada aynı şeydir).
VIRYAL_US: float = 2.1

#: Ortalama `[0, T/3)` üzerinde `N` eşit aralıklı örnekle alınır. `U`
#: eşit kütlelerde cisim permütasyonuna simetrik olduğu için `T/3`
#: periyotludur, ve periyodik analitik bir fonksiyonda eşit aralıklı ortalama
#: (yamuk kuralı) ÜSTEL yakınsar. Ölçüldü (sevk edilen 1e-13 BVP, `p = 2.1`):
#: N = 32/64/128/256 → bağıl artık 7.017e-14 / 7.150e-14 / 6.084e-14 /
#: 6.461e-14, yani N = 32'de kesme hatası zaten yuvarlamanın altında
#: (N = 16'da 2.454e-12, N = 8'de 1.143e-06 — üstel yakınsamanın izi).
VIRYAL_N: int = 32


def _potansiyel(Q, m: float, p: float):
    """`U = −Σ_{i<j} m² / ((p−1) r_ij^{p−1})` — TEST İÇİNDE, depodan bağımsız.

    `Q`: `(k, 3, 2)` konum dizisi; dönen `(k,)`. Eşit kütle `m`.
    """
    U = np.zeros(Q.shape[0])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        r = np.hypot(Q[:, j, 0] - Q[:, i, 0], Q[:, j, 1] - Q[:, i, 1])
        U -= m * m / ((p - 1.0) * r ** (p - 1.0))
    return U


def _viryal_bagil_artik(p: float = VIRYAL_US) -> float:
    """`⟨U⟩_T / (2E/(3−p)) − 1` — `p` kuvvet üssünde BVP'nin bulduğu sekizde."""
    fe = solve_figure_eight(power=p, rtol=1e-9, atol=1e-11)
    y0 = figure_eight_state(fe)
    q0 = np.asarray(y0.positions, dtype=float)
    v0 = np.asarray(y0.velocities, dtype=float)
    m = float(fe.mass)
    kinetik = 0.5 * m * float(np.sum(v0 * v0))
    E = kinetik + float(_potansiyel(q0[None, :, :], m, p)[0])
    ts = np.linspace(0.0, fe.period / 3.0, VIRYAL_N + 1)
    # Örnekleme `p`yi testten DEĞİL motorun döndürdüğü `fe.power` ETİKETİNDEN
    # alır: etiket düşerse (Y9) akış yanlış kuvvetle örneklenir ve özdeşlik
    # bozulur. Referans tarafı (`_potansiyel`, `2E/(3−p)`) testin kendi `p`si
    # olarak kalır — bağımsızlık oradan gelir.
    Q = sample_positions(y0, times=ts, power=fe.power)
    ortalama = float(np.mean(_potansiyel(Q, m, p)[:-1]))
    return ortalama / (2.0 * E / (3.0 - p)) - 1.0


def test_G7_SEKIZ_VIRYAL_TEOREMI_p_2_1_KUVVET_USSU_BVP_YE_ULASIYOR():
    """SINIF: FİZİK ÇIPASI — viryal teoremi, `⟨U⟩ = 2E/(3 − p)`, `p = 2.1`.

    Referans bağımsızdır: `k = 1 − p` dereceden homojen potansiyelde
    periyodik yörünge için `2⟨K⟩ = k⟨U⟩` (Landau–Lifshitz §10). `p = 2.1`
    sekizinin `T`si ya da `E`si BİLİNMİYOR ve gerekmiyor — özdeşlik HER
    periyodik yörüngede tam geçerlidir; yanlış kuvvet yasasıyla bulunmuş bir
    "sekiz" onu `p` potansiyeliyle sağlayamaz.

    NE SINANIYOR: `p` zincirinin ÜÇ halkası birden — BVP'nin girdisi
    (`solve_figure_eight(power=p)`), BVP çıktısındaki ETİKET (`fe.power`;
    örnekleme `p`yi oradan okur) ve örnekleme akışı (`sample_positions`).
    Referans tarafı (`_potansiyel`, `2E/(3−p)`) testin kendi `p`sini kullanır.

    ÖLÇÜLEN (temiz, bu penceresiyle): bağıl artık **6.278e-10** (BVP
    artığı `7.108e-10` — artık BVP'nin periyodiklik kusurunu izliyor; sevk
    edilen 1e-13 BVP ile aynı sayı 7.017e-14). Karşılaştırma için
    `p = 1.9` 4.372e-10, `p = 2.0` (1e-13) 4.596e-14.

    TOLERANS 5e-9: ölçülenin ~8 katı. Artığı DOP853'ün `rtol = 1e-9`daki
    ADIM DİZİSİ belirliyor, `least_squares`ın durduğu nokta DEĞİL (ilk
    sürümün gerekçesi buydu ve yanlıştı — denetim bulgusu). Ölçüldü: 27
    farklı tahmin noktasında (`vx ∈ {.33, .35, .37}` × `vy ∈ {.51, .53, .55}`
    × `T ∈ {6.2, 6.3, 6.4}`) artık 6.2761e-10 … 6.2785e-10. `rtol` taraması
    (`atol = rtol·1e-2`):

        rtol    5e-10      1e-9       2e-9       3e-9       4e-9
        artık   5.634e-10  6.278e-10  2.575e-09  3.708e-09  2.650e-09
        rtol    7e-9       8e-9       1e-8
        artık   4.845e-09  5.478e-09  5.876e-09

    Artık `rtol` ile kabaca orantılı (artık/rtol 0.6–1.3) ama MONOTON DEĞİL;
    5e-9 eşiği taramada ilk kez `rtol = 8e-9`da aşılıyor. Yani pay değerde
    ~8, entegratör hassasiyetinde ~7–8 kat: scipy'nin adım denetimi
    değişirse kaymanın ölçeği budur.

    KIRMA (bu testin yardımcısıyla, mutant kopyada ölçüldü; eşik 5e-9):

        M6_NB_SEKIZ_BVP_POWER_YOK          BVP p'yi görmüyor      −6.599e-03
        M5_NB_KUVVET_US_NEWTONDA_GORUNMEZ  ivme üssü (2p−1)/2     −1.433e-02
        Y9_NB_SEKIZ_POWER_ETIKETI_KAYIP    fe.power etiketi 2.0   +2.032e-02
        Y2_NB_ORNEKLEME_POWER_YOK          örnekleme power=2.0    +2.032e-02
        Y3_NB_KUVVET_US_1e5_KAYDI          ivme üssü +1e-5        −1.472e-06

    M5 `p = 2`de bit bit aynıdır. Y9'u İLK sürüm KAÇIRIYORDU: aynı mutantta
    örneklemeyi testin `p`siyle koşan eski yol +6.278e-10 veriyor (temizle
    aynı). M6, Y2 ve Y9'u bu dosya olmadan hiçbir hızlı test öldürmüyordu;
    M5'i enerji korunumu testi zaten öldürüyordu.

    ÇÖZÜNÜRLÜK SINIRI (dürüst olumsuz): kuvvet TEK BAŞINA `(1+ε)` ile
    ölçeklenirse (G normalizasyonu, potansiyel dokunulmadan) ölçülen artık:
    ε = +1e-8 → +1.223e-08, −1e-8 → −1.097e-08, +5e-9 → +6.429e-09,
    −5e-9 → −5.173e-09 (eşiğin %3 üstü, kıl payı ölür), +3e-9 → +4.109e-09
    (GEÇER). Yani `|ε| ≲ 5e-9` mertebesindeki bir G hatasını bu pencere
    GÖREMEZ.

    BÜTÇE: G7 hedefi test başına ≤ 0.3 s. BVP + örnekleme, önbellek boş, 5
    koşu (denetim düzeltmesi, hafif yük): `rtol 1e-9` 0.121–0.147 s; sevk
    edilen 1e-13 0.202–0.212 s; ara pencere 1e-11 0.150–0.162 s (artık
    9.353e-12). İlk sürüm 1e-13'ü yedi ajanın eşzamanlı yükü altında
    0.39–0.50 s ölçmüştü — hedefi YÜK ALTINDA aşan buydu; hafif yükte 1e-13
    de hedefin içinde kalıyor. 1e-9 seçimi bu yüzden bir YÜK PAYIDIR,
    zorunluluk değil; bedeli yukarıdaki çözünürlük sınırı (1e-13 penceresinde
    tolerans ~1e-12 olabilirdi). Pencere bilerek değiştirilmedi: KIRMA
    tablosundaki hedef mutantlar eşiğin ≥ 2.9e2 katıyla ölüyor (G ölçeği
    hedef değil; sınırı yukarıda beyan edildi).
    """
    bagil = _viryal_bagil_artik()
    assert abs(bagil) < 5e-9


#: ODE çıpasının penceresi. `p = 1e3` GÜÇLÜ-ORTA alan: bağıl kusur `M/p`
#: ile ölçeklenen hatalar burada `p = 3e5`ten 300 kat büyük görünür, ODE
#: kalıntısı ise ölçeklenmez (ölçüldü, aşağıda).
ODE_P: float = 1e3
ODE_E: tuple[float, ...] = (0.3, 0.7)


def _donus_artiklari(p: float = ODE_P, e: float = 0.3) -> tuple[float, float]:
    """`(u(Δφ/2)/u1 − 1, u(Δφ)/u2 − 1)` — GR, `M = 1`, sevk edilen ODE.

    `n_points = 3` ⇒ `phi = [0, Δφ/2, Δφ]` (`linspace` orta noktayı TAM
    `Δφ/2` verir); yani dönüş noktaları örnekleme payı OLMADAN okunur.
    """
    t = run_trajectory(M=1.0, p=p, e=e, n_points=3)
    u1, u2 = (1.0 - e) / p, (1.0 + e) / p
    return float(t.u[1]) / u1 - 1.0, float(t.u[2]) / u2 - 1.0


def test_G7_ODE_DONUS_NOKTALARI_BINET_ILK_INTEGRALININ_KOKLERI():
    """SINIF: FİZİK ÇIPASI — Binet ilk integralinin kökleri: `u(Δφ/2) = (1−e)/p`.

    Referans konik parametrizasyonudur (afel `u1 = (1−e)/p`), depo çıktısı
    değil. ODE `u1`i hiç görmez — yalnız `M/L²` üzerinden dolaylı alır —
    ve perihelden başlayıp yarım radyal periyotta oraya varmak zorundadır
    çünkü `(u')² = (u−u1)(u2−u)w(u)` tam korunur. İKİ YÖNLÜ sınavdır:
    mevcut `u1 <= min(u)` aralık sınavının aksine afeli yukarı iten bir
    hatayı da görür (Y1, aşağıda — ölçüldü).

    KAPSAM (dürüst olumsuz): bu çıpa yalnız `M/L²` ↔ sağ taraf
    tutarlılığını ve dönüş noktalarının YERİNİ sınar. `Δφ`yi ve kuram
    katsayısı `λ`yı SINAMAZ: okuma motorun KENDİ `Δφ`sinde yapılır ve `u`
    dönüş noktasında durağandır. Ölçüldü (afel, e = 0.3 / 0.7): `Y5` (sağ
    taraf ×1.01, φ ölçeği) −7.170e-13 / +2.812e-12; `Y8` (`λ` tutarlı
    biçimde yarıya) −7.232e-13 / +2.677e-12 — ikisi de toleransın içinde,
    test GEÇER. Onları hızlı şeritte bağımsız `kayma ≈ 6π/p` (rel 1e-4,
    `test_kapali_form.py::test_RUN_TRAJECTORY_SOZLESMESI...`) koruyor:
    sapma Y5'te −4.963e+02, Y8'de −5.000e-01. O sınav burada TEKRARLANMADI.

    ÖLÇÜLEN (temiz; GR, M = 1, p = 1e3, DOP853 1e-13/1e-15):

        e      u(Δφ/2)/u1 − 1     u(Δφ)/u2 − 1
        0.3    −7.124e-13          −1.492e-12
        0.7    +2.853e-12          −2.576e-14

    (Karşılaştırma: p = 3e5'te afel artığı e = 0.3'te 4.962e-12, e = 0.7'de
    1.704e-09 — orada `atol = 1e-15` mutlak payı `u ~ 1e-6`ya göre büyür;
    çıpa bu yüzden p = 1e3'te kuruldu.)

    TOLERANS 3e-11: en kötü ölçülenin (2.853e-12) ~10 katı. Pay sebebi:
    artık ODE'nin `rtol = 1e-13` ile `atol = 1e-15`in `u ~ 1e-3`e oranının
    (1e-12) mertebesinde; olay bulucunun `Δφ` kökü de aynı payla gelir.

    KIRMA (mutant kopyada ölçüldü; eşik 3e-11):

        mutant                         e = 0.3              e = 0.7
        M1_ODE_L2_KAYNAK_aP_DUSTU      afel −2.614e-03      afel −3.416e-03
        Y1_ODE_AFEL_YUKARI_IKIZ        afel +2.614e-03      afel +3.416e-03
        Y4_DONUS_NOKTASI_PERIHEL_1e7   perihel +2.31e-08    perihel +4.12e-08
        Y7_ODE_OLAY_YONU_TERS          afel +4.287e-01      afel +2.335e+00

    M1: `M/L² = (S − a(S² − P))/2` → `(S − aS²)/2`, toleransın ~1e8 katı
    (aynı mutant p = 3e5'te −8.667e-06 — hatanın `M/p` ile ölçeklendiğinin
    ölçümü). Y1, M1'in afeli YUKARI iten ikizi (`S² − P` → `S² − 2P`):
    tek yönlü `u1 <= min(u)` ve `6π/p` sınavlarının İKİSİNDEN de geçiyor
    (modül dokstringi) — iki yönlülüğün kanıtı. Y4 (`u2`yi 1e-7 oynatır)
    afeli değiştirmiyor (−7.139e-13); onu YALNIZ ikinci assert (perihel)
    öldürüyor — perihel assert'inin kendi ısırığı var. Y7 olay yönünü
    çeviriyor (olay afelde tetiklenir).

    BÜTÇE: G7 hedefi test başına ≤ 0.3 s; iki ODE, test çağrısı 0.01–0.02 s
    (`--durations`, yük altında).
    """
    for e in ODE_E:
        afel, perihel = _donus_artiklari(e=e)
        assert abs(afel) < 3e-11, (e, afel)
        assert abs(perihel) < 3e-11, (e, perihel)
