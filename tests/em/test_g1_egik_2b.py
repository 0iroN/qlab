"""G1 — EĞİK İNDİRGEME TAM 2B İLE KARŞILAŞTIRILDI (s / TM_z kolu).

NE KİLİTLENİYOR: `oblique.run_oblique_1d` enine dalga sayısını ANALİTİK
dayatır (`ky = k0 n1 sin θ`, y yönünde ızgara YOK); motorun tam 2B TM_z
Yee'si (`yee.run_2d`) ise y'de de ızgaralıdır. G1 ölçümü (09.10.2026,
`.superpowers/sdd/g1-olcum/`) iki şemanın farkını KAPALI FORMA bağladı:

* KİMLİK — y-değişmez ortamda 2B Yee'nin `sin(ky_m j)` kipi, eğik s'nin
  `ky -> K = 2 sin(ky/2)` konmuş hâlinin (kaynak kip ağırlığıyla çarpılmış)
  BİT-YAKIN aynısıdır. 2B'nin cevabı eğik çözücüye bir argümanla
  ulaşılabilir; kod düzeltmesi gerekmez, beyan gerekir.
* BEDEL — aynı ω ve aynı ANALİTİK `ky` ile (TF/SF ve Bloch'taki standart
  tanım) genlik bedeli = |r(ky)| - |r(2 sin(ky/2))|. Karmaşık fark DEĞİL;
  faz bedeli ayrı bir büyüklüktür ve yalnız referans düzlemiyle yazılır
  (bkz. `reflection_coefficient`). Ayrık Fresnel'in arayüz yüzüne (I-1/2)
  referanslı kapalı formu:

      r_s = sin((k1-k2)/2) / sin((k1+k2)/2)
          = (a1 c2 - c1 a2) / (a1 c2 + c1 a2),
      a_j = sin(k_j/2) = sqrt(eps_j Om^2 - K^2) / 2,   c_j = sqrt(1 - a_j^2),
      Om  = 2 sin(ω S/2) / S,   ω = 2π/λ,   K = ky (eğik) ya da 2 sin(ky/2) (2B).

  Kısmi yansımada bu r GERÇELDİR; K = 0, λ -> ∞'da Fresnel'e gider
  (λ = 1280'de fark 1.807e-06; yeniden ölçüldü, `g1-olcum/uygulama/`).

ÜÇ TEST, ÜÇ AYRI İŞ:

* T1 (hızlı) kimliğin KENDİSİ: tam sinüs, ne `ky` ne kesik seri.
* T2 (hızlı) sevk edilen ölçüm aleti (`reflection_coefficient`) iki açıda
  da kapalı forma oturur — HER |r| AYRI AYRI, yalnız fark değil. Sebep
  ölçüldü: enine terimi komşu hücreye kaydıran G mutantı sevk penceresinde
  (1 -> 2.25, 45°, λ = 20) bedeli yalnız %1 oynatıyor (6.6238e-04 ->
  6.6943e-04), her bir |r|'yi ise 1.44e-02.
* T3 (yavaş) KRİTİK AÇI BANDI: n1 > n2'de iki şemanın TİY eşiği ayrılır;
  bantta eğik |r| = 1, 2B kısmi yansıma verir ve bedel O(dx^2) değil
  O(dx) iner. "Bedel O(dx^2)" yalnız sabit açıda (noktasal) doğrudur.

SINIF (Modül 5 sözlüğü, `test_arayuz.py` başlığı): T1 bir ŞEMA ÇIPASIDIR —
iki ayrıklaştırmanın cebirsel denkliği, doğa değil. T2 ve T3'ün referansı
AYRIK Fresnel'dir, sürekli Fresnel değil; yani ikisi de şema çıpası /
çeviri denetimi sınıfındadır, doğa çıpası DEĞİLDİR. Kapalı form
(`_r_ayrik`) ve sınır nesnesi (`_XSig`) testte elle yazılır, kaynaktan
import edilmez. Özel (`_ad`) import YOK: koşumlar kamu yolundan geçer.

KAPSAM TAVANI, BAŞTAN: p (TE_z) için tam 2B ÖLÇÜLMEDİ — motorun 2B yolu
yalnız TM_z'dir (`yee._adim_2d`: Ez, Hx, Hy). G1'in p kolu AÇIK kalır.
Kontrast 1<->2.25, 1->4, 4->1 ve S 0.4/0.5/0.6 dışı ölçülmedi.

HIZLI ŞERİT BOŞLUĞU (bu dosyadan önce): `ky != 0`da 2B ile eğik çözücüyü
bağlayan hızlı test yoktu. G mutantı (enine geri beslemede komşu hücrenin
`cb`si) bu dosya hariç `tests/em -m "not slow"`ta yalnız bir testle düşüyor
(`test_arayuz.py::test_EGIK_P_KESIN_KAPI_ESLESMEYI_KESER_ve_TAM_ESIGE_OTURUR`,
bir kararlılık kapısı; 1 düştü / 203 geçti, ölçüldü); fizik referanslı
hiçbir hızlı test onu görmüyordu. Bu dosya dahil T1 ve T2 de düşüyor.

MUTANTLAR — ölçüldü (09.10.2026, `g1-olcum/uygulama/mutant_kos.py`,
monkeypatch; kaynak dosyalar bayt düzeyinde değişmedi, sha256 önce/sonra):

    A  eğik sessizce K = 2 sin(ky/2) kullanır     T1 DÜŞER  T2 DÜŞER  T3 DÜŞER
    B  yardımcı alan `_adim`den SONRA güncellenir  T1 DÜŞER  T2 DÜŞER  T3 DÜŞER
    C  s'te yardımcı alanın PML kaybı iki kat      T1 DÜŞER  T2 geçer  T3 geçer
    D  2B Hx güncellemesinde işaret hatası          T1 DÜŞER  T2 geçer  T3 geçer
    G  enine geri beslemede komşu hücrenin `cb`si  T1 DÜŞER  T2 DÜŞER  T3 DÜŞER

D'nin yalnız T1'de düşmesi yapısaldır: D yalnız 2B yolundadır ve T2/T3 2B
koşmaz. C'nin değiştirdiği kayıp yalnız soğurucunun içindedir; T2/T3'ün
okuduğu |r|'ye etkisi kapılarının altında kaldı (geçtiler; büyüklüğü ayrıca
ölçülmedi). C'yi bu dosyada T1 öldürür; bu dosya hariç hızlı şeritte
ayrıca BİR test (`test_arayuz.py::test_UZAMSAL_ISARET_UZLASIMI_ARAYUZSUZ_OLCULDU`;
tek satırlık KAYNAK yamasıyla 1 düştü / 203 geçti, doğrulayıcı ölçtü).
Monkeypatch koşumunda görülen ikinci düşüş
(`test_hucre_birimi.py::test_EGIK_ky_RADYAN_BOLU_HUCRE_ve_ky_dx_ETIKETI_CURUDU`)
ARTEFAKTTIR: o test `oblique_coefficients`in dokstringini okur, monkeypatch
sarmalayıcısınınki boştur; kaynak yamasında geçer. Ölçülen sapmalar her testin dokstringinde.

FAZ KİLİTSİZ: üç test de faza ve referans düzlemine bakmaz; `_r_ayrik`
payında işaret çevrilince üçü de geçer (doğrulayıcı ölçtü). Faz kapısı
eklenmedi: T2 penceresinde faz protokol hatası ölçülmedi.

EŞDEĞER, ÖLDÜRÜLEMEZ (beyan): 2B'de `d_hx`in y-ortalamasının kaldırılması.
y-değişmez sigma'da cebirsel olarak aynı sayıyı verir; KOŞULMADI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qlab.engine.em.boundary import PML
from qlab.engine.em.medium import Medium
from qlab.engine.em.oblique import run_oblique_1d
from qlab.engine.em.source import gauss_cos
from qlab.engine.em.yee import run_2d
from qlab.observable.em_obs import ObliqueWindow, reflection_coefficient


def _r_ayrik(eps1: float, eps2: float, K: float, lam: float, S: float) -> complex:
    """Ayrık Fresnel r_s, arayüz yüzüne (I-1/2) referanslı — TESTTE ELLE.

    Türetim (modül dokstringi): s'te ayrık süreklilik koşulları, her iki
    ortamda `sin(k_j/2)^2 = (eps_j Om^2 - K^2)/4` dispersiyonuyla. Geçen
    dalga sönümlüyse `Im a2 > 0` dalı seçilir; o zaman |r| = 1 TAM'dır.
    """
    om = 2.0 * math.sin(math.pi / lam * S) / S
    a1 = np.sqrt(complex(eps1 * om * om - K * K)) / 2.0
    a2 = np.sqrt(complex(eps2 * om * om - K * K)) / 2.0
    if a2.imag < 0:
        a2 = -a2
    c1, c2 = np.sqrt(1.0 - a1 * a1), np.sqrt(1.0 - a2 * a2)
    return complex((a1 * c2 - c1 * a2) / (a1 * c2 + c1 * a2))


class _XSig:
    """Sınır protokolü (`thickness` / `sigma_1d` / `sigma_2d` / `pec_mask_2d`).

    Sevk edilen `PML` profili YALNIZ x'te; y'de sigma YOK, y uçları
    `run_2d`nin kendi E = 0 duvarıdır. Testte yerel, çünkü sevk edilen
    `PML.sigma_2d` profili DÖRT kenara koyar ve y-değişmezliği bozar.
    """

    def __init__(self, d: int):
        self.thickness = d
        self._p = PML(thickness=d, m=3.0)

    def sigma_1d(self, cells: int) -> np.ndarray:
        return self._p.sigma_1d(cells)

    def sigma_2d(self, shape: tuple[int, int]) -> np.ndarray:
        return np.repeat(self._p.sigma_1d(shape[0])[:, None], shape[1], axis=1)

    def pec_mask_2d(self, shape: tuple[int, int]) -> np.ndarray:
        return np.zeros(shape, dtype=bool)


# ---------------------------------------------------------------------------
# T1 — KİMLİK
# ---------------------------------------------------------------------------
def test_2B_YEE_y_KIPI_EGIK_s_NIN_K_2SIN_HALI_BIT_YAKIN():
    """SINIF: ŞEMA ÇIPASI (iki ayrıklaştırmanın cebirsel denkliği).

    DÜZENEK: `Medium.half_space(300, iface 180, eps 4 -> 1)` y'de Ny = 9
    düğüm tekrarlanır (y-değişmez ortam); kaynak (90, 3), S = 0.6,
    λ = 14, sgf = 1.5, 900 adım; prob x = 170 ve 190 (biri yansıma,
    biri sönümlü taraf), BÜTÜN j. 2B sınırı `_XSig(30)`, eğik sınırı
    `PML(30)` — x'te aynı profil. 2B kaydı `sin(ky_m j)` kipine izdüşürülür
    (`ky_m = m π/(Ny-1)`) ve eğik koşumun `(2/(Ny-1)) sin(ky_m js)` kaynak
    ağırlığıyla çarpılmış kaydıyla karşılaştırılır.

    ÖLÇÜLDÜ (09.10.2026, `g1-olcum/elestirmen/t_aday.log`, bu turda
    yeniden), tepeye göre en büyük fark, m = 1 / m = 2:

        eğik(K = 2 sin(ky/2))     3.67e-15 / 1.91e-15    kapı <= 1e-12
        eğik(ky)                  1.73e-01 / 1.99e+00    kapı >= 1e-1
        eğik(ky - ky^3/24)        3.32e-04 / 2.58e-02    kapı >= 1e-4

    Üçüncü satır kimliğin TAM sinüs olduğunu, kesik seri olmadığını ayırt
    eder. Kimlik başka düzeneklerde de ölçüldü (bu turda yeniden üretildi):
    1 -> 2.25, S 0.4, Ny 13-37, m 1-3, x'te PML ve PEC: en büyük 1.414e-14;
    1 -> 4, S 0.5, Ny 11, 50° yalnız TEK FREKANSTA (|ρ_2B - ρ_eğik(K)|,
    tepeye göre değil): 6.55e-16; 4 -> 1, S 0.6, Ny 9, BÜTÜN kipler
    m = 1..7: yayılan m = 1, 2'de en büyük 5.64e-15, m = 3'te 2.9e-13,
    sönümlü m >= 4'te (tepe ~1e-8) bağıl en büyük 2.82e-9.

    KIRMA (ölçüldü, monkeypatch): A 1.68e-01, B 8.93e-01, C 2.12e-05,
    D NaN, G 1.40e-01 (m = 1, K satırı). Karşılaştırmalar `<=`/`>=` ile
    yazıldı: NaN her ikisinde de DÜŞER.

    MALİYET: betik duvar saati 0.10 s; pytest çağrısı 0.06-0.11 s (ölçüldü).
    """
    nx, ny, js, steps, S, lam, sgf = 300, 9, 3, 900, 0.6, 14.0, 1.5
    iface, isrc, px = 180, 90, (170, 190)
    L = ny - 1

    def darbe(t):
        return gauss_cos(t, wavelength=lam, sgf=sgf, t0=5 * sgf * lam)

    m1 = Medium.half_space(nx, interface=iface, eps1=4.0, eps2=1.0)
    m2 = Medium(np.repeat(m1.eps_r[:, None], ny, axis=1), np.ones((nx, ny)))
    kayit = run_2d(
        m2, courant=S, boundary=_XSig(30), steps=steps, source=darbe,
        source_index=(isrc, js), probes=tuple((x, j) for x in px for j in range(ny)),
        wavelength=lam, sgf=sgf,
    )
    P = kayit.probes.reshape(len(px), ny, steps)
    j = np.arange(ny)

    def egik(k, agirlik):
        return agirlik * run_oblique_1d(
            m1, courant=S, boundary=PML(thickness=30, m=3.0), steps=steps,
            source=darbe, source_index=isrc, probes=px, wavelength=lam, sgf=sgf,
            ky=k, polarization="s",
        ).probes

    for m in (1, 2):
        kym = m * math.pi / L
        agirlik = (2.0 / L) * math.sin(kym * js)
        izd = (2.0 / L) * np.einsum("j,pjn->pn", np.sin(kym * j), P)
        tepe = np.max(np.abs(izd))
        d_K = np.max(np.abs(izd - egik(2.0 * math.sin(kym / 2.0), agirlik))) / tepe
        d_ky = np.max(np.abs(izd - egik(kym, agirlik))) / tepe
        d_kesik = np.max(np.abs(izd - egik(kym - kym**3 / 24.0, agirlik))) / tepe
        assert d_K <= 1e-12, (m, d_K)
        assert d_ky >= 1e-1, (m, d_ky)
        assert d_kesik >= 1e-4, (m, d_kesik)


# ---------------------------------------------------------------------------
# T2 — SEVK EDİLEN ALET KAPALI FORMA OTURUR
# ---------------------------------------------------------------------------
#: Sevk penceresinin (3600 hücre, npml 60, sgf 9) küçültülmüşü. HER ALAN
#: AÇIKÇA YAZILIR: `ObliqueWindow` varsayılanlarından biri değişirse bu test
#: sessizce başka bir pencereyi ölçmesin (G7 dersi).
T2_PENCERE = ObliqueWindow(
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


def test_EGIK_BEDELI_AYRIK_FRESNEL_KAPALI_FORMUNA_OTURUR():
    """SINIF: kapalı-form referanslı ÇEVİRİ DENETİMİ (ayrık Fresnel).

    DURUM: eps 1 -> 2.25, θ = 45°, λ = 10, `T2_PENCERE`. 2B'nin cevabı
    sevk edilen aletle `θ' = asin(2 sin(ky/2)/(k0 n1)) = 44.53183°`
    çağrısıyla OKUNUR (T1'in kimliği). Yalnız |r| karşılaştırılır: faz
    referans düzlemine bağlıdır (bkz. `reflection_coefficient`).

    ÖLÇÜLDÜ (`t_aday.log`, bu turda yeniden):

        ||rc(θ)|  - |r(ky)||   7.56e-08     kapı <= 1e-6
        ||rc(θ')| - |r(K)||    1.83e-07     kapı <= 1e-6
        bedel |rc(θ)|-|rc(θ')| 2.5687e-03, kuram 2.5688e-03,
        fark 1.08e-07                       kapı <= 1e-6

    Kapalı formun bedeli (2.5688e-03) kapının 2569 katıdır: test `ky` ile
    `K`yı AYIRT EDER. Sevk penceresinde (λ = 20) aynı sınav 1.02 s sürüyor
    ve protokol hatası 3e-9 düzeyinde; burada ucuzu seçildi.

    KIRMA (ölçüldü, ilk düşen kapı ||rc(θ)| - |r(ky)||): A 2.57e-03
    (eğik artık 2B'yi veriyor), B 9.07e-02, G 6.27e-02 (θ' satırı
    6.28e-02). G'de bedel bu pencerede 2.5688e-03 -> 2.7297e-03
    (sapma 1.61e-04), sevk penceresinde (λ = 20) 7.05e-06 (%1) oynuyor;
    tek bir |r|'nin sapması bedelinkinin ~390 katı (bu pencere) ve ~2040
    katı (sevk penceresi). C ve D bu testi geçer (bkz. modül dokstringi).

    MALİYET: betik duvar saati 0.30 s; pytest çağrısı 0.30-0.33 s (4 eğik
    koşum; ölçüldü).
    """
    e1, e2, th, lam = 1.0, 2.25, 45.0, 10.0
    S = T2_PENCERE.courant
    k0n1 = 2.0 * math.pi / lam * math.sqrt(e1)
    ky = k0n1 * math.sin(math.radians(th))
    K = 2.0 * math.sin(ky / 2.0)
    th_2b = math.degrees(math.asin(K / k0n1))

    kur_eg = abs(_r_ayrik(e1, e2, ky, lam, S))
    kur_2b = abs(_r_ayrik(e1, e2, K, lam, S))
    assert kur_eg - kur_2b >= 1e-3, (kur_eg, kur_2b)

    ortak = dict(eps1=e1, eps2=e2, wavelength=lam, polarization="s", window=T2_PENCERE)
    rc_eg = reflection_coefficient(theta_deg=th, **ortak)
    rc_2b = reflection_coefficient(theta_deg=th_2b, **ortak)
    assert rc_eg is not None and rc_2b is not None
    assert abs(abs(rc_eg) - kur_eg) <= 1e-6, abs(abs(rc_eg) - kur_eg)
    assert abs(abs(rc_2b) - kur_2b) <= 1e-6, abs(abs(rc_2b) - kur_2b)
    bedel = abs(rc_eg) - abs(rc_2b)
    assert abs(bedel - (kur_eg - kur_2b)) <= 1e-6, (bedel, kur_eg - kur_2b)


# ---------------------------------------------------------------------------
# T3 — KRİTİK AÇI BANDI
# ---------------------------------------------------------------------------
def _kritikler(eps1: float, eps2: float, lam: float, S: float) -> tuple[float, float]:
    """İki şemanın kritik açısı [der]: eğik `ky^2 > eps2 Om^2`, 2B `K^2 > eps2 Om^2`."""
    om = 2.0 * math.sin(math.pi / lam * S) / S
    k0n1 = 2.0 * math.pi / lam * math.sqrt(eps1)
    th_eg = math.degrees(math.asin(math.sqrt(eps2) * om / k0n1))
    th_2b = math.degrees(math.asin(2.0 * math.asin(math.sqrt(eps2) * om / 2.0) / k0n1))
    return th_eg, th_2b


def _dtft_r(eps1, eps2, th, kk, lam, *, cells, iface, isrc, sgf, S, npml):
    """Kendi DTFT'm TAM f = 1/λ'da; kayıt T = yankı sınırı (`hi`).

    Sevk edilen alet bandın içinde kullanılamıyor (ölçüldü, sevk penceresi,
    2.25 -> 1, λ = 20): θ = 41.8'de `reflection_coefficient` 0.948468
    döndürüyor, kuram 1 — ve sürekli Fresnel'e göre orada TİY olmadığı için
    `|rho| = 1` kapısı KOŞMUYOR, sayı sessizce dönüyor; θ = 41.9'da kapı
    koşuyor ve `None` dönüyor. Sebebi bu turda kovalanmadı.
    SEBEP (10.10.2026, `rc-olcum/`): kayıt KESİK — dal noktasının uzun
    yansıma kuyruğu kayıt bittiğinde sürüyor (son periyotta tepenin
    2.68e-02'si). Sevk edilen alet artık iki açıda da kuyruk kapısında
    `None` döndürür (`test_kayit_tamligi.py`). Bu yardımcı `hi`ya kadar
    uzun kayıt kullanır; T3'ün |r_eg| kapısı (5e-3) onun kalan hatasını
    (41.9'da 1.97e-03) kapsar.
    """
    probe = iface - 6
    sg = sgf * lam
    t0 = 5.0 * sg
    vx = math.cos(math.radians(th)) / math.sqrt(eps1)
    t_yans = t0 + (2 * iface - isrc - probe) / vx
    hi = t_yans + 2 * probe / vx - 5.0 * sg
    steps = int(math.floor(hi / S))

    def darbe(t):
        return gauss_cos(t, wavelength=lam, sgf=sgf, t0=t0)

    ortak = dict(
        courant=S, boundary=PML(thickness=npml, m=3.0), steps=steps, source=darbe,
        source_index=isrc, probes=(probe,), wavelength=lam, sgf=sgf, ky=kk,
        polarization="s",
    )
    on = run_oblique_1d(Medium.half_space(cells, interface=iface, eps1=eps1, eps2=eps2), **ortak)
    arka = run_oblique_1d(Medium(np.full(cells, eps1), np.ones(cells)), **ortak)
    a, b = on.probes[0], arka.probes[0]
    cek = np.exp(-2j * math.pi / lam * np.arange(steps) * S)
    return complex(np.sum((a - b) * cek) / np.sum(b * cek))


@pytest.mark.slow
def test_KRITIK_ACI_BANDINDA_SEMALAR_AYRI_TARAFTA_ve_BEDEL_O_dx():
    """SINIF: kapalı-form referanslı ŞEMA ÇIPASI (iki şemanın TİY eşiği).

    KAPALI FORM (çözücü değil): eğik TİY'e `ky^2 > eps2 Om^2`de, 2B
    `4 sin^2(ky/2) > eps2 Om^2`de girer; n1 > n2'de iki kritik açı AYRILIR.
    2.25 -> 1, S 0.4, λ = 20: eğik 41.7766°, 2B 41.9895°, Fresnel 41.8103°;
    bant 0.2129°. Bantta bedelin tepesi eğik kritiğindedir ve
    `1 - |r_2B(θ_eg)|`e eşittir: λ = 10/20/40/80/160 için
    0.2588/0.1471/0.0776/0.0397/0.0201, ardışık üsler
    0.815/0.924/0.966/0.984 — yani O(dx), O(dx^2) DEĞİL. Aynı kontrastta
    kritiğin 10° altında (31.81°) üs 1.99 (1.433e-03 -> 3.607e-04). Bant
    genişliği ise dx^2 ile daralıyor (0.2129 -> 0.0528). 4 -> 1'de aynı
    tablo: λ = 20'de bant 0.1374°, tepe 0.0956. (`beyan_olcum_kapali.log`.)

    ÇÖZÜCÜ (2.25 -> 1, θ = 41.9 — bandın içi, λ = 20, S 0.4; sevk
    geometrisi N 3600, iface 1500, kaynak 600, prob 1494, npml 60, sgf 9;
    eğik `ky` ve `K = 2 sin(ky/2)` ile `run_oblique_1d`). ÖLÇÜLDÜ
    (`t3_aday.log`, bu turda yeniden):

        |r_eg| = 1.001968   ||r_eg| - 1|  1.97e-03    kapı <= 5e-3
        |r_2B| = 0.901853   kuramdan fark 8.17e-06    kapı <= 1e-3
        bedel  = +0.10011   (kuram +0.098138)         kapı >= 0.09

    Bantta bedel, sevk aletiyle sabit açıda ölçülen en büyük değerin
    (1 -> 2.25, 60°, λ = 20: 3.13e-03) 32 katıdır. Daha uzun pencerede
    (sgf 14, N 7000, iface 2900, kaynak 1200) θ = 41.8'de bedel +0.13996
    (kuram +0.13945), θ = 41.9'da +0.09908 ölçüldü; o koşumlar 8 s
    sürdüğü için kilitte değil.

    KIRMA (ölçüldü, ilk düşen kapı): A'da eğik de 2B'yi verir, |r_eg|
    0.901853 (||r_eg| - 1| = 9.81e-02); B'de koşum patlar (|r_eg|
    1.3e+56); G'de |r_eg| kapısı geçer ama |r_2B| 0.898933, kuramdan
    2.93e-03. C ve D bu testi geçer (bkz. modül dokstringi). Kapalı form
    asserti (bant ve üs) mutantlara KÖRDÜR — çözücüye dokunmaz, testin
    öncülünü sınar.

    MALİYET: betik duvar saati 2.66 s; pytest çağrısı 2.63-2.85 s
    (ölçüldü) -> `slow`.
    """
    e1, e2, lam, S, th = 2.25, 1.0, 20.0, 0.4, 41.9

    # Kapalı formun kendisi: θ bandın İÇİNDE ve tepe O(dx) iniyor.
    th_eg, th_2b = _kritikler(e1, e2, lam, S)
    assert th_eg < th < th_2b, (th_eg, th, th_2b)
    k0n1 = 2.0 * math.pi / lam * math.sqrt(e1)

    def tepe(L):
        a = math.sin(math.radians(_kritikler(e1, e2, L, S)[0])) * 2.0 * math.pi / L * math.sqrt(e1)
        return 1.0 - abs(_r_ayrik(e1, e2, 2.0 * math.sin(a / 2.0), L, S))

    us = [math.log2(tepe(L) / tepe(2 * L)) for L in (20.0, 40.0)]
    assert all(0.5 < u < 1.5 for u in us), us

    ky = k0n1 * math.sin(math.radians(th))
    K = 2.0 * math.sin(ky / 2.0)
    kur_2b = abs(_r_ayrik(e1, e2, K, lam, S))
    assert abs(abs(_r_ayrik(e1, e2, ky, lam, S)) - 1.0) < 1e-12

    pencere = dict(cells=3600, iface=1500, isrc=600, sgf=9.0, S=S, npml=60)
    r_eg = _dtft_r(e1, e2, th, ky, lam, **pencere)
    r_2b = _dtft_r(e1, e2, th, K, lam, **pencere)
    assert abs(abs(r_eg) - 1.0) <= 5e-3, abs(r_eg)
    assert abs(abs(r_2b) - kur_2b) <= 1e-3, (abs(r_2b), kur_2b)
    assert abs(r_eg) - abs(r_2b) >= 0.09, abs(r_eg) - abs(r_2b)
