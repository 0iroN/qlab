"""G1-p — EĞİK p (TE_z) TAM 2B İLE KARŞILAŞTIRILDI (dualite yoluyla).

NE KİLİTLENİYOR: G1'in s kolu (`test_g1_egik_2b.py`) motorun 2B TM_z
Yee'siyle kapandı; p kolu açık kalmıştı çünkü motorun 2B yolu yalnız TM_z'dir
(`yee._adim_2d`: Ez, Hx, Hy). Bu dosya p'yi YENİ BİR MOTOR YOLU YAZMADAN
kapatır: Maxwell dualitesi 2B TE_z'yi (ε, μ = 1) 2B TM_z'ye (ε' = 1, μ' = ε)
çevirir, `(e, hx, hy) := (Hz, −Ex, −Ey)`. Ayrık düzeyde de aynı: TM_z
çekirdeği (`yee.run_2d`, kamu yolu) dual ortamla koşulunca DUAL IZGARADA bir
2B TE_z Yee'dir — Hz düğümde (i, j), Ex (i, j+½), Ey (i+½, j); Ex/Ey'nin
gördüğü ε `Medium.mu_yari_2d`dir, yani HER ZAMAN aritmetik (teğet E'nin
kuralı). Ön ölçüm + eleştirmen: `.superpowers/sdd/g1p-olcum/` (10.10.2026).

DUAL EŞLEMENİN KOŞULLARI (ölçüldü; dışında kimlik bozulur):

* μ = 1. Soğurucuda σ' = σ/ε (eğik p'de Hz'in kaybı σS/(2ε)); sevk edilen
  `PML.sigma_2d` profili dört kenara koyar ve y-değişmezliği bozar, bu yüzden
  sınır nesnesi (`_XSig`) testte yerel. Soğurucunun İÇİNDE ε sabit olmalı:
  dual Ey kaybını ortalama(σ/ε), eğik p ortalama(σ)/ortalama(ε) ile kurar;
  soğurucuda ε rampası varken fark 9.1e-7 … 5.9e-5 (eleştirmen; logu kesik,
  yeniden üretildi: `dogrulama/v2_kimlik.log`, 9.11e-7 / 5.95e-5). Rampa
  varken dual yine kayıplı bir TE_z Yee'dir; bunlar EĞİK p İLE KİMLİĞİN
  koşullarıdır, TE_z olmanın değil.
* Dört kenar ve iç `pec_mask` dual yolda PMC'dir (Hz = 0) — eğik p'nin x
  uçlarıyla AYNI. Bu yolla TE_z'de PEC cisim KURULAMAZ (dual maske yerli PMC
  ile bit bit aynı, yerli PEC'ten 0.337 uzak; eleştirmen).
* Kimlik `ey_kural` = `mu_yari_2d`nin kuralı (aritmetik) iken tutar; sevk
  varsayılanı aritmetiktir. `mu_yari_2d` harmonik yapılınca dual 2B eğik p'nin
  HARMONİK kuralıyla bit-yakın aynı (3.5e-15; eleştirmen) — yani bağ
  yapısaldır.

AYRIK r_p, arayüz yüzüne (I−½) referanslı, aritmetik Ey kuralında:

    r_p = (ε2 a1 c2 − ε1 a2 c1) / (ε2 a1 c2 + ε1 a2 c1),
    a_j = sin(k_j/2) = sqrt(ε_j Ω² − K²) / 2,  c_j = cos(k_j/2),
    Ω = 2 sin(ωS/2) / S,   K = ky (eğik) ya da 2 sin(ky/2) (2B).

s'nin formunun (`r_s = (a1 c2 − c1 a2)/(a1 c2 + c1 a2)`) ε ağırlıklı eşi;
Ey(I−½)'nin ε'si (ε1+ε2)/2 iken arayüz denkleminde sanal terimler TAM
sadeleşir. Bağımsız düğüm-tabanlı 2×2 çözücüye ≤ 3.2e-15 (ön ölçüm, 1509
durum, `p2b_dal.log`) ve aktarım çözücüsüne ≤ 8.8e-15 (eleştirmen, 3966
durum; a2 → 1 Nyquist kenarında 1.4e-14, doğrulayıcı). DAL: a2 gerçel ve
> 1 iken (ortam 2'de Nyquist üstü sönümlü dal, k2 = π + iκ) c2 = −i
sqrt(a2² − 1)'dir; `np.sqrt(1 − a2²)` öteki dalı seçer ve r'nin EŞLENİĞİNİ
verir (karmaşık r'de 2.0'a kadar; |r| ikisinde de 1, yani |r| kapıları bu
hatayı GÖREMEZ; eleştirmen C1, doğrulayıcı S2). Yardımcı dalı AÇIKÇA seçer
ve dal T2p'deki tek satırlık assert'le kilitlidir (hiçbir çağrı o bölgeye
girmez). Hata numpy skaler yolunda çıkar; `cmath` ile karma aritmetik doğru
dalı tesadüfen seçebilir (doğrulayıcı).

BEŞ TEST (T0y 11.10.2026'da, G1p-b):

* T0p (hızlı) dual = 2B TE_z Yee'nin KENDİSİ: testte elle yazılmış yerli
  TE_z (Hz, Ex, Ey) ile `run_2d`-dual, TAM 2B rastgele ε(x, y)'de alan alan.
* T0y (hızlı) 2B TRANSPOZ SİMETRİSİ: rastgele ε, μ ve PML'li koşum x <-> y
  değişimiyle tekrarlanır; y ekseninin kuralları motorun KENDİ x eksenine
  bağlanır (T0p'den bağımsız ikinci y bekçisi).
* T1p (hızlı) KİMLİK: dual 2B'nin `sin(ky_m j)` kipi, eğik p'nin
  `K = 2 sin(ky/2)` hâlidir — G1-s T1'in p eşi, aynı düzenek.
* T2p (hızlı) sevk edilen alet iki açıda da ayrık r_p'ye oturur; bedelin
  İŞARETİ Brewster altında NEGATİFTİR (s'de hep pozitif).
* T3p (yavaş) DOĞRUDAN 2B TİY δ_ps (eğik çözücü YOK) sürekliye O(Δx²);
  13.10.2026'dan (G1p-a) beri üç basamaklı merdivenle (L 6/8/12) sürekliye
  MUTLAK kapı, ±3e-3°.

SINIF (Modül 5 sözlüğü, `test_arayuz.py` başlığı; G7 sözlüğü ayrı,
`test_g7_hizli_cipa.py` başlığı): T0p TRANSKRİPSİYON DENETİMİ / ŞEMA ÇIPASI
(eğik s'nin ky = 0 → run_1d bit bit indirgemesinin eşi); T1p ŞEMA ÇIPASI
(referans ikinci bir çözücü); T2p ÇEVİRİ DENETİMİ (referans AYRIK Fresnel);
T3p FİZİK ÇIPASI (TİY δ_ps girdiden okunamaz; iki kutuplanma ortam 1'de aynı
k1 ile yayıldığı için prob↔arayüz gidiş-dönüş fazı farkta sadeleşir —
`tir_phase_difference`ın gerekçesi). T3p için DÜRÜST NOT: hata yakalama gücü
T1p'nin alt kümesidir ve sonucu T1p + mevcut eğik TİY çıpasından mantıksal
olarak çıkar; yeri "2B TE_z yolunun tek fizik çıpası" olmasıdır. Kapalı
formlar ve sınır nesnesi testte elle yazılır, kaynaktan import edilmez; özel
(`_ad`) import YOK.

KAPSAM TAVANI (T0y HARİÇ: μ ∈ [1, 9], soğurucu dahil rastgele ε): μ = 1; y-değişmez
ortam (T0p hariç); soğurucu içinde ε sabit; duvar ve maske PMC; kontrastlar 1↔2.25,
1→4, 4→1, 1↔9, 6.25↔2.25, 1→6.25 ve S 0.3–0.6 dışı ölçülmedi; T3p yalnız 4→1, 45°.

MUTANTLAR — ölçüldü (10.10.2026, `.superpowers/sdd/araclar/mut.py`, kopya
ağaçta kaynak yaması; `g1p-olcum/uygulama/mut_g1p*.log`, test başına ayrı):

    M1  eğik p'de Ey'nin ε'si sol hücre        T1p T2p
    M2  `mu_yari_2d` (x ekseni) harmonik       T0p T1p T3p
    M3  eğik p sessizce K = 2 sin(ky/2)        T1p T2p
    M4  yardımcı alan `_adim`den SONRA (s, p)  T1p T2p
    M5  p'de Ex'in PML kaybı karesi            T1p
    M6  2B `d_hy` tek taraflı                  T1p
    M7  p'de Ey kaybının σ'sı sol düğüm        T1p
    M8  2B Hy işareti                          T0p T1p T3p
    M9  eğik p'de Ey'nin ε'si sağ hücre        T1p T2p
    M11 p'de enine geri beslemede komşu `cb`   T1p
    M12 2B `dby` sol hücrenin μ'sü             T0p T1p T3p
    M13 p'de Ex'in `aux_b`si komşu hücre       T1p T2p

Bu dosyada HAYATTA KALAN iki mutant:
* E1 (p'de yardımcı alanın işareti −1) EŞDEĞERDİR: işaret bir gauge'dur
  (`oblique.ObliqueCoefficients`); onu `test_arayuz`un transkripsiyon testi
  görür.
* M10 (`Medium` sınıf varsayılanı `ey_kural` harmonik) bu dosyanın yollarına
  ULAŞMAZ: `half_space` kuralı kendi imzasından geçirir, dual ortam
  `eps_yari` kullanmaz, `T2P_PENCERE` kuralı açıkça yazar. Hızlı EM
  şeridinde `test_arayuz::test_EGIK_P_KESIN_KAPI_ESLESMEYI_KESER_ve_TAM_ESIGE_OTURUR`
  ile düşer (`mut_m10_suit.log`).

YAN BULGU: M2 bu dosyadan önce hızlı EM şeridinin 246 testinin HEPSİNDEN
geçiyordu (eleştirmen `m_suit_hizli_M2_MU2D_HARM.log`) — 2B'de μ'nün yarı
hücre kuralı kilitsizdi. p'ye özgü dört mutantı (M3, M11, M13 ve yalnız p'de
yardımcı alan sırası) mevcut hızlı şeritte yalnız bir kararlılık kapısı
testi (`test_EGIK_P_KESIN_KAPI_...`) görüyordu; fizik referanslı hiçbir hızlı
test görmüyordu (eleştirmen `m_suit_hizli_*.log`). Tablodaki M1 (yalnız
eğik p'nin `el_yb`si) orada kaynak yamasıyla 4 testte düşer (doğrulayıcı,
`dogrulama/m_serit_A.log`); eleştirmenin 5'i daha geniş bir monkeypatch'in
sayısıdır.

T0p 2B'NİN y EKSENİNİ KİLİTLEYEN TEK HIZLI TESTTİ (10.10.2026; doğrulayıcı,
`dogrulama/m_yeni_*.log`, `m_serit_*.log`): `mu_yari_2d` y ekseni (axis = 1)
harmonik ve 2B `dbx`te sol hücrenin μ'sü hızlı EM şeridinin 246 testinin
HEPSİNDEN geçiyordu, bu dosyada YALNIZ T0p'de düşüyordu — y-değişmez
düzenekler (T1p, T3p) bu eksene kördür. 11.10.2026 (G1p-b): ikinci, T0p'den
BAĞIMSIZ bekçi T0y (transpoz simetrisi; referansı el yazımı TE_z değil
motorun kendi x ekseni). Ölçüldü (`mut.py`): T0y şu dokuz mutantın dokuzunda
düşer -- y'de harmonik μ, `dbx` sol μ, `d_hx` tek taraflı (Y6), Hx işareti,
x'te harmonik μ (M2), `d_hy` tek taraflı (M6), Hy işareti (M8), `dby` sol μ
(M12) ve iki eksende harmonik μ (SON maddede ısırık TESADÜFİ: simetri
kapısından değil, karşıt ortamın x ekseni de harmonik olunca karşıtın 0'a
inmesinden; T0y dokstringi). Y6'yı (σ'nın y ortalaması) T0p GÖREMEZ --
PEC'te σ = 0; T0y PML'li olduğu için görür. Ölçüldü: Y6, T0y dışındaki hızlı
EM şeridinin 252 testinin HEPSİNDEN geçiyordu (`mut.py`, 11.10.2026). Artık
T0p silinse de y ekseni kilitsiz kalmaz.
Doğrulayıcının 12 ek kaynak mutantından 10'u öldü (T1p: p'de Hz/Ey kaybı,
kaynak bağlaşımı, geri besleme; T0p/T1p: 2B kayıt hizası, kaynak zamanı,
kaynak genliği; T2p: `reflection_coefficient` ön ortamı); hayatta kalan
iki: 2B üst y duvarı sıfırlaması (sıfır başlangıçta EŞDEĞER) ve
`ObliqueWindow.ey_kural` varsayılanı (bu dosyaya ulaşmaz; hızlı şeritte
`test_arayuz::test_PENCERE_BEYANI_TAM` öldürür).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qlab.engine.em.boundary import PEC, PML
from qlab.engine.em.medium import Medium
from qlab.engine.em.oblique import run_oblique_1d
from qlab.engine.em.source import gauss_cos
from qlab.engine.em.yee import run_2d
from qlab.observable.em_obs import ObliqueWindow, reflection_coefficient


def _ac(eps: float, om: float, K: float) -> tuple[complex, complex]:
    """a = sin(k/2), c = cos(k/2) — DAL AÇIKÇA (modül dokstringi)."""
    a = np.sqrt(complex(eps * om * om - K * K)) / 2.0
    if a.imag == 0.0 and a.real > 1.0:
        return a, complex(-1j * math.sqrt(a.real * a.real - 1.0))
    return a, complex(np.sqrt(1.0 - a * a))


def _r_ayrik(pol: str, eps1: float, eps2: float, K: float, lam: float, S: float) -> complex:
    """Ayrık Fresnel r (s ya da p), I−½ referanslı, aritmetik Ey — TESTTE ELLE."""
    om = 2.0 * math.sin(math.pi / lam * S) / S
    a1, c1 = _ac(eps1, om, K)
    a2, c2 = _ac(eps2, om, K)
    if a2.imag < 0:
        a2, c2 = -a2, c2
    w1, w2 = (eps2, eps1) if pol == "p" else (1.0, 1.0)
    return complex((w1 * a1 * c2 - w2 * a2 * c1) / (w1 * a1 * c2 + w2 * a2 * c1))


class _XSig:
    """Sınır protokolü (`thickness` / `sigma_1d` / `sigma_2d` / `pec_mask_2d`).

    Sevk `PML` profili YALNIZ x'te; `olcek` verilirse her x düğümünde ona
    bölünür (dual: σ' = σ/ε). y uçları `run_2d`nin kendi `e = 0` duvarıdır.
    """

    def __init__(self, d: int, olcek: np.ndarray | None = None):
        self.thickness = d
        self._p = PML(thickness=d, m=3.0)
        self._olcek = olcek

    def sigma_1d(self, cells: int) -> np.ndarray:
        s = self._p.sigma_1d(cells)
        return s if self._olcek is None else s / self._olcek

    def sigma_2d(self, shape: tuple[int, int]) -> np.ndarray:
        return np.repeat(self.sigma_1d(shape[0])[:, None], shape[1], axis=1)

    def pec_mask_2d(self, shape: tuple[int, int]) -> np.ndarray:
        return np.zeros(shape, dtype=bool)


def _dual(eps2d: np.ndarray) -> Medium:
    """2B TE_z (ε, μ = 1) -> TM_z çekirdeğinin ortamı (ε' = 1, μ' = ε)."""
    return Medium(np.ones_like(eps2d), np.asarray(eps2d, dtype=float).copy())


def _y_tekrar(eps1d: np.ndarray, ny: int) -> np.ndarray:
    return np.repeat(np.asarray(eps1d, dtype=float)[:, None], ny, axis=1)


# ---------------------------------------------------------------------------
# T0p — DUAL = 2B TE_z YEE
# ---------------------------------------------------------------------------
def _yerli_te_z(eps, *, S, steps, darbe, kaynak, probes, kural):
    """YERLİ 2B TE_z Yee (μ = 1): Hz (i, j), Ex (i, j+½), Ey (i+½, j); dört kenar
    Hz = 0 (PMC). Önce E (yarım adım), sonra Hz; kaynak Hz'e S·darbe. Teğet E'nin
    ε'si `kural` ile (aritmetik ya da harmonik)."""
    nx, ny = eps.shape

    def yari(a, b):
        return 0.5 * (a + b) if kural == "aritmetik" else 2.0 / (1.0 / a + 1.0 / b)

    ex_eps = yari(eps[:, :-1], eps[:, 1:])
    ey_eps = yari(eps[:-1, :], eps[1:, :])
    hz = np.zeros((nx, ny))
    ex = np.zeros((nx, ny - 1))
    ey = np.zeros((nx - 1, ny))
    pi_ = np.asarray([p[0] for p in probes])
    pj = np.asarray([p[1] for p in probes])
    kayit = np.zeros((len(probes), steps))
    for n in range(steps):
        kayit[:, n] = hz[pi_, pj]
        ex = ex + (S / ex_eps) * (hz[:, 1:] - hz[:, :-1])  # ε ∂Ex/∂t = +∂Hz/∂y
        ey = ey - (S / ey_eps) * (hz[1:, :] - hz[:-1, :])  # ε ∂Ey/∂t = −∂Hz/∂x
        hz = hz.copy()
        hz[1:-1, 1:-1] += S * ((ex[1:-1, 1:] - ex[1:-1, :-1]) - (ey[1:, 1:-1] - ey[:-1, 1:-1]))
        hz[0, :] = hz[-1, :] = 0.0
        hz[:, 0] = hz[:, -1] = 0.0
        hz[kaynak] += S * darbe(n * S)
    return kayit, hz, ex, ey


def test_DUAL_2B_YERLI_TE_z_YEE_ILE_AYNI():
    """SINIF: TRANSKRİPSİYON DENETİMİ / ŞEMA ÇIPASI (Modül 5); G7'de çıpa DEĞİL.

    "Dual koşum gerçekten bir 2B TE_z Yee'dir" cümlesinin dayanağı. DÜZENEK:
    40×30 TAM 2B rastgele ε ∈ [1, 9] (tohum 5; y-değişmez DEĞİL), `PEC()`
    sınırı (dualde dört kenar Hz = 0 = PMC), S 0.5, 300 adım, kaynak (20, 15),
    λ 8, sgf 1. Yerli TE_z motordan hiçbir şey import etmez.

    ÖLÇÜLDÜ (`g1p-olcum/uygulama/test_degerleri.log`; eleştirmen benzer
    düzenekte ve 90×70, ε ∈ [1, 6]'da da 0.0): son anda Hz, Ex, Ey ve kayıtta
    fark 0.0 (bit bit). Kapı bit eşitliği DEĞİL, 1e-13: işlem sırası değişince
    kırılmasın; karşılaştırma `np.max` ile, NaN yayılır ve `<=` DÜŞER
    (Python `max`ı NaN'ı yutuyordu — doğrulayıcı S1). KARŞIT: yerli TE_z
    harmonik teğet-E kuralıyla koşulunca kayıtta bağıl 0.367 uzak (kapı
    >= 1e-3) — test ortalama kuralını GÖRÜR. Tam 2B rastgele ε olduğu için
    bu dosyada 2B'nin y eksenini (Ex'in ε'si) gören İLK testtir; ikincisi
    11.10.2026'dan beri T0y (transpoz simetrisi; başlık).

    MALİYET: pytest çağrısı 0.02 s (ölçüldü).
    """
    rng = np.random.default_rng(5)
    nx, ny, S, steps, lam = 40, 30, 0.5, 300, 8.0
    eps = 1.0 + 8.0 * rng.random((nx, ny))
    kaynak = (20, 15)
    probes = ((10, 7), (20, 15), (31, 22), (5, 25))

    def darbe(t):
        return gauss_cos(t, wavelength=lam, sgf=1.0, t0=5.0 * lam)

    rec = run_2d(
        _dual(eps), courant=S, boundary=PEC(), steps=steps, source=darbe, source_index=kaynak,
        probes=probes, wavelength=lam, sgf=1.0,
    )
    k_ar, hz, ex, ey = _yerli_te_z(eps, S=S, steps=steps, darbe=darbe, kaynak=kaynak, probes=probes,
                                   kural="aritmetik")
    tepe = float(np.max(np.abs(k_ar)))
    assert tepe > 1e-3, tepe
    olc = float(np.max([
        np.max(np.abs(k_ar - rec.probes)), np.max(np.abs(hz - rec.final.e)),
        np.max(np.abs(ex + rec.final.hx)), np.max(np.abs(ey + rec.final.hy)),
    ])) / tepe
    assert olc <= 1e-13, olc
    k_har, *_ = _yerli_te_z(eps, S=S, steps=steps, darbe=darbe, kaynak=kaynak, probes=probes,
                            kural="harmonik")
    karsit = float(np.max(np.abs(k_har - rec.probes))) / tepe
    assert karsit >= 1e-3, karsit


# ---------------------------------------------------------------------------
# T0y — 2B TRANSPOZ SİMETRİSİ (y ekseninin İKİNCİ hızlı bekçisi, G1p-b)
# ---------------------------------------------------------------------------
class _YHarmonik(Medium):
    """KARŞIT için: yalnız y ekseninde (axis = 1) harmonik yarı hücre μ."""

    def mu_yari_2d(self, axis: int) -> np.ndarray:
        if axis == 1:
            return 2.0 / (1.0 / self.mu_r[:, :-1] + 1.0 / self.mu_r[:, 1:])
        return super().mu_yari_2d(axis)


def test_2B_TRANSPOZ_SIMETRISI_y_EKSENI_x_EKSENININ_AYNASI():
    """SINIF: ŞEMA ÇIPASI (simetri); G7'de çıpa DEĞİL. G1p-b (11.10.2026).

    NEDEN: T0p'den önce 2B'nin y eksenindeki μ kuralı (`mu_yari_2d(1)`) ve
    `dbx`in μ'sü hızlı şeritte kilitsizdi; T0p tek bekçiydi (başlık). Bu test
    ikinci, T0p'den BAĞIMSIZ bir bekçidir: el yazımı TE_z'ye değil, motorun
    KENDİ x eksenine dayanır. Kare hücreli TM_z Yee x <-> y değişimi altında
    değişmezdir: (Ez, Hx, Hy)(x, y) -> (Ez, −Hy, −Hx)(y, x). Ortam (ε VE μ
    tam 2B rastgele), kaynak ve problar transpoze edilip ikinci koşum
    yapılınca kayıt AYNI çıkmalı; y ekseninin her kuralı böylece x ekseninin
    (T1p, T3p, G1-s'in kilitlediği) kuralına bağlanır.

    DÜZENEK: 40×30 (kare DEĞİL: transpoz şekli de değiştirir), ε ∈ [1, 4],
    μ ∈ [1, 9] (tohum 5), `PML(6)` -- dört kenarda AYNI profil, yani σ da
    transpozla değişir ve `d_hx`/`d_hy` ortalaması da sınanır (PEC'te σ = 0
    ve o iki satır görünmez); S 0.5, 300 adım, kaynak (20, 15), λ 8, sgf 1.

    ÖLÇÜLDÜ (11.10.2026): fark 0.0 (bit bit; ε = 1 dual ortam + PEC ile
    40×30, 24×18, 30×22
    düzeneklerinde de 0.0). Kapı 1e-13: işlem sırası değişince kırılmasın;
    `np.max` NaN'ı yayar. KARŞIT: transpoze koşum yalnız y'de harmonik μ'lü
    ortamla yapılınca kayıtta bağıl 1.62 uzak (kapı >= 1e-3).

    KÖR NOKTA (beyan): iki ekseni AYNI biçimde bozan mutant simetriyi KORUR
    ve simetri kapısından geçer; simetri yalnız eksenler ARASI farkı görür.
    Ölçüldü (iki eksende harmonik μ): test yine kırmızı, ama SİMETRİ
    kapısından değil (fark 0.0) KARŞIT kapısından -- karşıt ortamın x ekseni
    de harmonik olunca karşıt 0.0'a iner. Bu tesadüfi ısırığa güvenilmez;
    öyle mutantları x ekseninin bekçileri (T0p, T1p, T3p) öldürür.
    KIRMA (ölçüldü, 11.10.2026, `mut.py`): başlıktaki dokuz mutantın dokuzu.
    MALİYET: üç koşum, pytest çağrısı 0.02 s (ölçüldü).
    """
    rng = np.random.default_rng(5)
    nx, ny, S, steps, lam = 40, 30, 0.5, 300, 8.0
    eps = 1.0 + 3.0 * rng.random((nx, ny))
    mu = 1.0 + 8.0 * rng.random((nx, ny))
    kaynak = (20, 15)
    probes = ((10, 7), (20, 15), (31, 22), (5, 25))
    probes_t = tuple((j, i) for i, j in probes)

    def darbe(t):
        return gauss_cos(t, wavelength=lam, sgf=1.0, t0=5.0 * lam)

    def kos(ortam, kay, pr):
        return run_2d(ortam, courant=S, boundary=PML(thickness=6, m=3.0), steps=steps,
                      source=darbe, source_index=kay, probes=pr, wavelength=lam, sgf=1.0)

    a = kos(Medium(eps.copy(), mu.copy()), kaynak, probes)
    b = kos(Medium(eps.T.copy(), mu.T.copy()), kaynak[::-1], probes_t)
    tepe = float(np.max(np.abs(a.probes)))
    assert tepe > 1e-3, tepe
    olc = float(np.max([
        np.max(np.abs(a.probes - b.probes)), np.max(np.abs(a.final.e - b.final.e.T)),
        np.max(np.abs(a.final.hx + b.final.hy.T)), np.max(np.abs(a.final.hy + b.final.hx.T)),
    ])) / tepe
    assert olc <= 1e-13, olc
    c = kos(_YHarmonik(eps.T.copy(), mu.T.copy()), kaynak[::-1], probes_t)
    karsit = float(np.max(np.abs(a.probes - c.probes))) / tepe
    assert karsit >= 1e-3, karsit


# ---------------------------------------------------------------------------
# T1p — KİMLİK
# ---------------------------------------------------------------------------
def test_DUAL_2B_TE_z_y_KIPI_EGIK_p_NIN_K_2SIN_HALI_BIT_YAKIN():
    """SINIF: ŞEMA ÇIPASI (iki ayrıklaştırmanın cebirsel denkliği).

    DÜZENEK G1-s T1'in AYNISI: `Medium.half_space(300, iface 180, ε 4 -> 1)`
    y'de Ny = 9 tekrarlanır; kaynak (90, 3), S 0.6, λ 14, sgf 1.5, 900 adım;
    prob x = 170 ve 190, BÜTÜN j. 2B: dual ortam + `_XSig(30, olcek=ε)`;
    eğik: `PML(30)`, `polarization="p"`. Kayıt `sin(ky_m j)` kipine izdüşer.

    İKİ ZORUNLU KOŞUL (sahte yeşil önlemi; eleştirmen `e6_t1p_sinir.log`):
    x-PML ε ≠ 1 bölgesinde kalmalı (sol PML ε = 4'te) ve m = 1 kalmalı.
    Ölçeksiz σ' = σ eşleme hatası yalnız m = 1'de 8.1e-7 görünür (m = 2'de
    1.1e-13, kapıdan geçer); p'de enine geri beslemede komşu `cb` mutantı
    yalnız PML'de etkili (m = 1: sol komşu `cb[:-2]` 1.19e-6, eleştirmen; sağ
    komşu `cb[2:]` = M11 1.26e-6, doğrulayıcı; m = 2'de ikisi de 3.6e-15).
    PEC düzenekte ikisi de görünmez.

    ÖLÇÜLDÜ (`g1p-olcum/olcum/p1_kimlik.log`), tepeye göre, m = 1 / m = 2:

        eğik p(K = 2 sin(ky/2))      3.68e-15 / 1.54e-15   kapı <= 1e-12
        eğik p(ky)                   1.52e-01 / 1.96e+00   kapı >= 1e-1
        eğik p(ky - ky^3/24)         2.91e-04 / 2.38e-02   kapı >= 1e-4
        eğik p(K), ey_kural harmonik 9.35e-02 / 1.29e-01   kapı >= 1e-2
        eğik s(K) (TE != TM)         7.73e-01 / 8.34e-01   kapı >= 1e-1

    Kimlik başka düzeneklerde de ölçüldü: 1→2.25 S 0.4 Ny 13, 2.25→1 S 0.5
    Ny 11 (PEC), 1→4 S 0.4 Ny 17 yayılan kiplerde ≤ 2.3e-14 (ön ölçüm);
    1→9 S 0.3 Ny 16, 9→1 S 0.55 Ny 12 (kaynak ε 9'da), 6.25→2.25 S 0.45 Ny 21
    (tek Ny): taşıyıcıda yayılan kiplerde ≤ 9.5e-15, 9→1 m = 3 (sınırda,
    K 0.831 > √ε Ω 0.785) 1.6e-14, sönümlü kipler ≤ 4.6e-11 (eleştirmen;
    `e2_kimlik.log` KESİK, yalnız 1→9 ve 9→1 m = 1'i içerir — yeniden
    üretim `dogrulama/v2_kimlik.log`).

    MALİYET: betik duvar saati 0.18-0.23 s; pytest çağrısı 0.18 s (ölçüldü).
    """
    nx, ny, js, steps, S, lam, sgf = 300, 9, 3, 900, 0.6, 14.0, 1.5
    iface, isrc, px = 180, 90, (170, 190)
    L = ny - 1

    def darbe(t):
        return gauss_cos(t, wavelength=lam, sgf=sgf, t0=5 * sgf * lam)

    m1 = Medium.half_space(nx, interface=iface, eps1=4.0, eps2=1.0)
    harm = Medium.half_space(nx, interface=iface, eps1=4.0, eps2=1.0, ey_kural="harmonik")
    kayit = run_2d(
        _dual(_y_tekrar(m1.eps_r, ny)), courant=S, boundary=_XSig(30, olcek=m1.eps_r), steps=steps,
        source=darbe, source_index=(isrc, js), probes=tuple((x, j) for x in px for j in range(ny)),
        wavelength=lam, sgf=sgf,
    )
    P = kayit.probes.reshape(len(px), ny, steps)
    j = np.arange(ny)

    def egik(ortam, k, pol, agirlik):
        return agirlik * run_oblique_1d(
            ortam, courant=S, boundary=PML(thickness=30, m=3.0), steps=steps, source=darbe,
            source_index=isrc, probes=px, wavelength=lam, sgf=sgf, ky=k, polarization=pol,
        ).probes

    for m in (1, 2):
        kym = m * math.pi / L
        K = 2.0 * math.sin(kym / 2.0)
        agirlik = (2.0 / L) * math.sin(kym * js)
        izd = (2.0 / L) * np.einsum("j,pjn->pn", np.sin(kym * j), P)
        tepe = np.max(np.abs(izd))

        def fark(ortam, k, pol):
            return np.max(np.abs(izd - egik(ortam, k, pol, agirlik))) / tepe

        d_K = fark(m1, K, "p")
        assert d_K <= 1e-12, (m, d_K)
        d_ky = fark(m1, kym, "p")
        assert d_ky >= 1e-1, (m, d_ky)
        d_kesik = fark(m1, kym - kym**3 / 24.0, "p")
        assert d_kesik >= 1e-4, (m, d_kesik)
        d_harm = fark(harm, K, "p")
        assert d_harm >= 1e-2, (m, d_harm)
        d_s = fark(m1, K, "s")
        assert d_s >= 1e-1, (m, d_s)


# ---------------------------------------------------------------------------
# T2p — SEVK EDİLEN ALET AYRIK r_p'YE OTURUR, BEDELİN İŞARETİ
# ---------------------------------------------------------------------------
#: G1-s T2'nin penceresi. HER ALAN AÇIKÇA YAZILIR (G7 dersi).
T2P_PENCERE = ObliqueWindow(
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


def test_EGIK_p_BEDELI_AYRIK_r_p_KAPALI_FORMUNA_OTURUR_ve_ISARETI_BREWSTERDE_DONER():
    """SINIF: kapalı-form referanslı ÇEVİRİ DENETİMİ (ayrık Fresnel r_p).

    DURUM: ε 1 -> 2.25, θ = 45° (Brewster 56.31°'nin ALTI), λ = 10,
    `T2P_PENCERE`. 2B'nin cevabı `θ' = asin(2 sin(ky/2)/(k0 n1)) = 44.53183°`
    ile OKUNUR (T1p'nin kimliği; bu test 2B koşmaz, 2B'ye bağı T1p'dir).

    ÖLÇÜLDÜ (`p2_kapali_form.log`, eleştirmen `m_aday_YOK.log`):

        ||rc(θ)|  - |r_p(ky)||    3.35e-08     kapı <= 1e-6
        ||rc(θ')| - |r_p(K)||     5.22e-08     kapı <= 1e-6
        bedel |rc(θ)| - |rc(θ')| -2.8814e-03, kuram -2.8814e-03,
        fark 1.9e-08                           kapı <= 1e-6
        kuram bedeli             -2.8814e-03   kapı <= -1e-3 (İŞARET)

    İŞARET: K < ky olduğundan 2B "daha küçük açı" gibi davranır; Brewster
    altında |r_p| açıyla azalır, bedel NEGATİF; Brewster üstünde pozitif. s'de
    |r_s| açıyla monotondur ve bedel hep pozitiftir (G1-s: +2.5688e-3 aynı
    durumda). İşaret iki AYRIK Brewster açısının ARASINDA bir kez döner
    (eleştirmen `e4_bedel.log`, dört kontrast). Son assert kapalı formun
    kendisini sınar (Brewster üstü 60°: kuram +1.4846e-2) — çözücüye kördür,
    testin öncülünü kilitler. Sevk aleti bu pencerede 60°'de `None` döndürür
    (kapılar); λ 20 sevk penceresinde ölçüldü: +3.8175e-3 (kuram +3.8172e-3).

    İlk assert yardımcının DALINI kilitler (başlık): ε 9, λ 6, S 0.4,
    K 0.2'de a2 = 1.556 > 1 ve fiziksel dal Im c2 < 0 (−1.1923i; dal satırı
    silinince +1.1923i — doğrulayıcı `v8_dal_assert.log`).

    MALİYET: betik duvar saati 0.34-0.50 s; pytest çağrısı 0.33 s (4 eğik
    koşum; ölçüldü).
    """
    assert _ac(9.0, 2.0 * math.sin(math.pi / 6.0 * 0.4) / 0.4, 0.2)[1].imag < 0
    e1, e2, th, lam = 1.0, 2.25, 45.0, 10.0
    S = T2P_PENCERE.courant
    k0n1 = 2.0 * math.pi / lam * math.sqrt(e1)
    ky = k0n1 * math.sin(math.radians(th))
    K = 2.0 * math.sin(ky / 2.0)
    th_2b = math.degrees(math.asin(K / k0n1))

    kur_eg = abs(_r_ayrik("p", e1, e2, ky, lam, S))
    kur_2b = abs(_r_ayrik("p", e1, e2, K, lam, S))
    assert kur_eg - kur_2b <= -1e-3, (kur_eg, kur_2b)

    ortak = dict(eps1=e1, eps2=e2, wavelength=lam, polarization="p", window=T2P_PENCERE)
    rc_eg = reflection_coefficient(theta_deg=th, **ortak)
    rc_2b = reflection_coefficient(theta_deg=th_2b, **ortak)
    assert rc_eg is not None and rc_2b is not None
    assert abs(abs(rc_eg) - kur_eg) <= 1e-6, abs(abs(rc_eg) - kur_eg)
    assert abs(abs(rc_2b) - kur_2b) <= 1e-6, abs(abs(rc_2b) - kur_2b)
    bedel = abs(rc_eg) - abs(rc_2b)
    assert abs(bedel - (kur_eg - kur_2b)) <= 1e-6, (bedel, kur_eg - kur_2b)

    ky60 = k0n1 * math.sin(math.radians(60.0))
    ust = abs(_r_ayrik("p", e1, e2, ky60, lam, S)) - abs(_r_ayrik("p", e1, e2, 2.0 * math.sin(ky60 / 2.0), lam, S))
    assert ust >= 1e-3, ust


# ---------------------------------------------------------------------------
# T3p — DOĞRUDAN 2B TİY δ_ps (FİZİK ÇIPASI)
# ---------------------------------------------------------------------------
def _r_2b_dogrudan(pol: str, e1: float, e2: float, L: int, lam: float, th: float) -> complex:
    """Doğrudan 2B r (eğik çözücü YOK): TE_z = dual, TM_z = sevk çekirdeği.

    Pencere: nx 900, iface 420, kaynak (200, 2), sgf 3, npml 30, S 0.4, prob
    iface − 6 (BÜTÜN j), Ny = L + 1, kip m = 1 (ky = π/L). Kayıt T = yankı
    sınırı (`test_g1_egik_2b._dtft_r` ile aynı kural), tam f = 1/λ'da DTFT;
    r = DTFT(ön − arka)/DTFT(arka) — boru hattı uzlaşımı (klasik formun conj'u).
    """
    nx, iface, isrc, js, sgf, S, npml = 900, 420, 200, 2, 3.0, 0.4, 30
    ny = L + 1
    probe = iface - 6
    sg = sgf * lam
    t0 = 5.0 * sg
    vx = math.cos(math.radians(th)) / math.sqrt(e1)
    t_yans = t0 + (2 * iface - isrc - probe) / vx
    steps = int(math.floor((t_yans + 2 * probe / vx - 5.0 * sg) / S))

    def darbe(t):
        return gauss_cos(t, wavelength=lam, sgf=sgf, t0=t0)

    jj = np.arange(ny)
    kip = (2.0 / L) * np.sin(math.pi / L * jj)
    seri = []
    for eps1d in (Medium.half_space(nx, interface=iface, eps1=e1, eps2=e2).eps_r, np.full(nx, e1)):
        eps2d = _y_tekrar(eps1d, ny)
        if pol == "p":
            ortam, sinir = _dual(eps2d), _XSig(npml, olcek=eps1d)
        else:
            ortam, sinir = Medium(eps2d, np.ones_like(eps2d)), _XSig(npml)
        rec = run_2d(
            ortam, courant=S, boundary=sinir, steps=steps, source=darbe, source_index=(isrc, js),
            probes=tuple((probe, j) for j in range(ny)), wavelength=lam, sgf=sgf,
        )
        seri.append(kip @ rec.probes)
    on, arka = seri
    cek = np.exp(-2j * math.pi / lam * np.arange(steps) * S)
    return complex(np.sum((on - arka) * cek) / np.sum(arka * cek))


@pytest.mark.slow
def test_DOGRUDAN_2B_TIY_FAZ_FARKI_SUREKLIYE_O_dx2_YAKINSAR():
    """SINIF: FİZİK ÇIPASI (Modül 5) — 2B TE_z yolunun tek fizik çıpası.

    TİY'de δ_ps = arg(r_p) − arg(r_s) girdiden okunamaz; iki kutuplanma ortam
    1'de aynı k1 ile yayıldığı için prob↔arayüz gidiş-dönüş fazı farkta
    sadeleşir (`tir_phase_difference` gerekçesi). Burada iki ayrı 2B koşumdan
    (TE_z dual, TM_z) okunur; eğik çözücü YOK. 4 -> 1, θ = 45° (θ_c 30°),
    L = 6, 8 ve 12 (λ = 2 n1 sin θ L = 16.97 / 22.63 / 33.94). Sürekli δ_ps
    testte elle, boru uzlaşımında: 2 atan(ε1 κ/(ε2 kx1)) − 2 atan(κ/kx1) =
    +70.528779°.

    ÖLÇÜLDÜ (`p5_sure.log`, doğrulayıcı `v5`; eleştirmen `e5_dogrudan_2b.log`
    kendi PML profiliyle 5.5e-5 / 2.6e-4°, üs 1.883; L8 ve merdiven G1p-a,
    13.10.2026, `.superpowers/sdd/dusuk-ayiklama/grup8-C/g1p_2b_merdiven.log`
    -- değerler v5 ile altı hanede aynı; Prescott çekirdeğinde de aynı):

        ||r_p| - 1|, ||r_s| - 1|   <= 1.45e-05 (L12, s)            kapı <= 1e-4
        |δ_2B - δ_ayrık(K)|        L6 / L8 / L12  6.5e-05 / 7.1e-05 / 2.8e-04°
                                                                kapı <= 1e-3°
        δ_2B - δ_sürekli           L6 / L8 / L12  +1.34479 / +0.79389 / +0.36460°
                                                                kapı > 0
        üs log2(L6/L12)            1.883                      kapı [1.5, 2.5]
        MERDİVEN δ_∞ - δ_sürekli   -3.09e-04°                  kapı |.| <= 3e-3°

    MERDİVEN (G1p-a, MUTLAK FİZİK KAPISI). δ(L) = δ_∞ + a/L² + b/L⁴ üç
    basamaktan (L 6/8/12) çözülür ve δ_∞ sürekliye MUTLAK bağlanır. Açılım
    çift kuvvetlerdedir (ayrık dağınım a_j² ∝ 1/L²); kapalı formun kendisi
    sürekliye C/L² ile gider, C → 53.7589 (L 1e3 / 1e4'te 53.7587 /
    53.7589). Hata bütçesi: (i) kesme artığı -- aynı merdiven KAPALI FORMA
    uygulanınca -8.79e-04° (1/L⁶ ve üstü); (ii) 2B'nin kapalı formdan sapması
    ağırlıklarla büyür, ağırlıklar (3/7, -64/35, 12/5), |w|_1 = 4.657; ölçülen
    sapmalarla en kötü toplam 8.3e-04°, (i) ile birlikte 1.7e-03°. Kapı 3e-3°:
    ölçülenin 9.7 katı, ölçülen sapmalardan kurulan en kötü bütçenin 1.8 katı.
    ÇÖZÜNÜRLÜK: eski fizik kolları (sapma > 0, üs [1.5, 2.5]) sürekli limitte
    sabit bir D ∈ [−0.1541, +0.1714]° hatasını geçiriyordu (doğrulayıcı `v5`,
    yeniden üretildi); merdiven D'yi birebir taşır (ağırlık toplamı 1), yani
    artık D ∈ [−0.0027, +0.0033]° -- 50 kat dar. L 24 (nx 3600, 114 s) bu
    yüzden gerekmedi.

    `δ_ayrık(K)` kapalı formu (`_r_ayrik`) — eğik çözücünün değil 2B'nin
    ayrık cevabı; L12'de pay 3.6 kat (pencere protokolü sınırlıyor; bu bir
    fizik iddiası DEĞİL). Kayıt tam: kuyruk (son periyot/tepe) ≤ 2.3e-4
    (eleştirmen). Büyük pencere L 24 (nx 3600) sürekliye +0.0926° (üs 1.98,
    ön ölçüm `p4_dogrudan_2b.log`, 114 s; kilitte değil, yeniden koşulmadı).

    DÜRÜST NOT: yakalama gücü T1p'nin alt kümesi (eleştirmenin yedi
    mutantından yalnız `mu_yari_2d` harmonik: 12.8°; uygulamanın 14'ünden
    M2, M8, M12 — üçünü de T0p/T1p öldürür; doğrulayıcının 12 ek mutantından
    0); ortak-kip hatalara kör — 2B kayıt hizası, kaynak zamanı yarım adım ve
    kaynak genliği ×2 ölçüldü ve geçti. SABİT ya da 1/L² ile ölçeklenen bir
    kaynak hatasında MERDİVEN YENİ MUTANT ÖLDÜRMEZ ve bu yapısaldır (G1p-a,
    ölçülen sayılarla): δ_2B'ye her basamakta aynı c kayması ekleyen bir hata
    ŞEMA kapısında c >= 7.2e-4°'de (L12: 2.8e-4 + c), merdivende ancak
    c >= 3.3e-3°'de düşer; 1/L² ile ölçeklenen bir hata merdivende a'ya
    EMİLİR (görünmez), şema kapısında görünür. AMA L'ye DÜZENSİZ bağlı bir
    hata merdivenin ağırlıklarıyla |w|_1 = 4.66 kata kadar BÜYÜR ve şema
    kapısından sızanı merdiven yakalar (doğrulayıcı, 13.10.2026; aritmetik
    örnek, gerçekçi bir kaynak mutantı bulunmadı): L 6/8/12'de −1e-3 / +9e-4
    / −1.2e-3° kayma şema kapısını geçer (ölçülen farklarla −9.3e-4 /
    +9.7e-4 / −9.2e-4°) ama D'yi −4.95e-3° kaydırır, |D| = 5.3e-3° > 3e-3° ve
    merdiven düşer. Kapalı form testte elle yazılı, kaynaktan gelmez.
    Merdivenin asıl işi SINIFTIR: şema kapısının referansı (`_r_ayrik`) bir
    gün motora "uydurulursa" sürekliye bağlı tek kol o kalır, ve artık
    −0.15…+0.17° değil ±3e-3° çözer. Olumlu: dual `_XSig` ölçeği unutulunca
    bu test de düşer (L12'de 3.0e-3°). MALİYET: pytest çağrısı 5.2-7.1 s, yüke bağlı (L 6/8/12, 13.10.2026,
    `--durations=0`; L 6/12 ile 4.1 s idi, `p5_sure.log`) -> `slow`.
    """
    e1, e2, th = 4.0, 1.0, 45.0
    S = 0.4
    kx1 = math.sqrt(e1) * math.cos(math.radians(th))
    kappa = math.sqrt(e1 * math.sin(math.radians(th)) ** 2 - e2)
    d_surekli = math.degrees(2 * math.atan(e1 * kappa / (e2 * kx1)) - 2 * math.atan(kappa / kx1))
    basamak = (6, 8, 12)
    sapma = []
    for L in basamak:
        lam = 2.0 * math.sqrt(e1) * math.sin(math.radians(th)) * L
        K = 2.0 * math.sin(math.pi / L / 2.0)
        rp = _r_2b_dogrudan("p", e1, e2, L, lam, th)
        rs = _r_2b_dogrudan("s", e1, e2, L, lam, th)
        assert abs(abs(rp) - 1.0) <= 1e-4 and abs(abs(rs) - 1.0) <= 1e-4, (L, abs(rp), abs(rs))
        d_2b = math.degrees(np.angle(rp / rs))
        d_ayrik = -math.degrees(np.angle(_r_ayrik("p", e1, e2, K, lam, S) / _r_ayrik("s", e1, e2, K, lam, S)))
        assert abs(d_2b - d_ayrik) <= 1e-3, (L, d_2b, d_ayrik)
        sapma.append(d_2b - d_surekli)
    assert all(x > 0 for x in sapma), sapma
    us = math.log2(sapma[0] / sapma[-1])
    assert 1.5 <= us <= 2.5, (sapma, us)
    # MERDİVEN: δ(L) − δ_sürekli = D + a/L² + b/L⁴, üç basamaktan D (dokstring).
    tasarim = np.array([[1.0, L**-2.0, L**-4.0] for L in basamak])
    D = float(np.linalg.solve(tasarim, np.asarray(sapma))[0])
    assert abs(D) <= 3e-3, (
        f"üç basamaklı merdiven sürekliye oturmuyor: δ_∞ − δ_sürekli = {D:+.3e}° "
        f"(sapmalar {sapma}; ölçülen -3.09e-04°)"
    )
