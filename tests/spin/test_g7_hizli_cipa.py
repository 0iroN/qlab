"""G7 — HIZLI ŞERİDİN EKSİK FİZİK ÇIPASI: Heisenberg eşleşmesi (J ≠ 0).

SINIF: DAİRESEL DEĞİL — referans kapalı formdur, simülasyona girdi değildir.

===========================================================================
NEDEN BU DOSYA VAR — ÖLÇÜLDÜ (09.10.2026, G7 mutasyon turu, yük altında)
===========================================================================

`-m "not slow"` şeridinde (bu dosya olmadan 185 test, yük altında 31-39 s;
dört ölçüm) Ising tarafının ucuz fizik çıpaları BOL (yeniden ölçüldü:
Onsager-Yang 0.17-0.19 s, Wolff tam sayımı 0.20-0.33 s, AR(1) kapalı formu
0.03-0.04 s, Binder tam sayımı 0.04 s fixture setup'ı + ≤ 0.02 s çağrı).
Heisenberg tarafında ise J ≠ 0 eşleşmesini sınayan TEK BİR fizik çıpası
yoktu. Ölçülen kanıt:

    M_HEIS_ZZ: metropolis.py'de dE'nin skaler çarpımı yalnız z bileşenine
    indirildi (`np.sum(fark * komsu, axis=2)` → `fark[..., -1] *
    komsu[..., -1]`). Ising'de (n_component = 1) bu satır BİREBİR aynıdır;
    Heisenberg O(3) simetrisini kaybeder ve Ising-benzeri bir modele döner.

    hızlı şerit (bu dosya hariç): 1 failed / 184 passed. Düşen tek test
    `test_KABUL_ORANI_ADIMLA_DUSUYOR_...` ve düşen assert bir TRANSKRİPSİYON
    KİLİDİ (kabul 0.968974, kilit 0.965898 ± 1e-5) — fizik iddiası değil.

İKİ FİZİK ÇIPASININ BU MUTANTA NEDEN KÖR OLDUĞU ÖLÇÜLDÜ:

- Langevin çıpası (`test_ALANSIZ_ESLESME_...`) J = 0'da koşar; eşleşme
  terimi orada sıfırla çarpılır. Yapısal körlük.
- Hızlı Mermin-Wagner parametresi (T = 1.0) mutantta da GEÇİYOR. Testin
  kendi yardımcısıyla ölçüldü: Heisenberg eğimi doğru kodda −1.0545,
  mutantta −1.0114 (sınır −0.8); m_I/m_H(32) oranı 11.35 → 8.78 (sınır
  8.0). Sebep: ZZ-eşleşmeli model T = 1.0'da DÜZENSİZ fazdadır ve
  paramanyetin `m ~ 1/L` sönümü de eğimi −1'e getirir. Yani eğim testi
  "sürekli simetri düzeni yıkıyor" ile "T_c'nin üstündeyiz"i AYIRAMIYOR.

===========================================================================
ÇIPA — L = 2 HEISENBERG KAFESİ = 4 SPİNLİK HALKA, TAM ÇÖZÜM
===========================================================================

`lattice.py` dokstringi L = 2'yi bir KENAR DURUMU olarak beyan eder: karşıt
komşular aynı siteye denk gelir ve her bağ ÇİFT sayılır. Elle sayıldı:
`neighbour_sum(a) = 2b + 2d`, dolayısıyla (a,b,c,d halka sırası, |s| = 1)

    E    = −2J (a·b + b·c + c·d + d·a) = −2J·S
    |M|² = 4 + 2S + 2 (a·c + b·d),        m = |M| / 4  (`magnetisation_series`)

yani bağ başına 2J'lik dört spinlik klasik O(3) halkası, K = 2J/T. Eşleme
motora bakılmadan yapıldı; sonra motorun `total_energy` ve `magnetisation`
fonksiyonlarıyla 1000 rastgele durumda karşılaştırıldı: en büyük fark E'de
3.6e-15, m²'de 3.3e-16 (E kısmını G7 denetçisi de bağımsız buldu).

İKİ GÖZLENEBİLİR sınanır, çünkü ⟨E⟩ tek başına YAPISAL olarak kördür: halka
iki-parçalıdır, F ve AF aynı ⟨E⟩'yi verir (J'nin işareti görünmez). ⟨m²⟩
hem S'nin İŞARETİNİ hem karşı köşe korelasyonu ⟨a·c⟩'yi taşır: F'de
0.774716, AF'de 0.048004 (tam).

REFERANS — İKİ BAĞIMSIZ YOL, TESTİN İÇİNDE KARŞILAŞTIRILIR:

  YOL A — transfer matrisi (`scipy.special.spherical_in`; `exp(K cos γ) =
  Σ (2l+1) i_l(K) P_l(cos γ)` açılımı + toplama teoremi):
      Z(K)  = Σ_l (2l+1) [4π i_l(K)]^N,   N = 4,   ⟨S⟩ = d ln Z / dK
      ⟨a·c⟩ = 2 Σ_l (l+1) ρ_l² ρ_{l+1}² / Σ_l (2l+1) ρ_l⁴,   ρ_l = i_l / i_0
      (ŝ işlemcisi l kanalını l ± 1'e bağlar; Σ_{m,m'} |⟨l m|ŝ|l+1 m'⟩|²
      = l + 1 — toplam kuralı: Σ_{l'} = 2l + 1.)

  YOL B — spherical_in YOK, transfer matrisi YOK: a ve c sabitken b ve d
  yalnız a + c'nin alanını görür ve küre integrali kapalı formdur. x = a·c
  (iki bağımsız birim vektör için [−1, 1]'de tekdüze), r = |a + c| =
  √(2 + 2x), L = Langevin (`coth y − 1/y`):
      w(x) = [sinh(Kr) / (Kr)]²,   ⟨S⟩ = 2 ⟨r·L(Kr)⟩_w,   ⟨a·c⟩ = ⟨x⟩_w
  Kalan 1B integral 32 düğümlü Gauss-Legendre ile alınır; integrand x'te
  analitiktir (r yalnız çift kuvvetlerle girer). 16 / 32 / 400 düğüm aynı
  sayıyı verir (YOL A'dan fark < 1e-13).

  Ölçüldü (32 düğüm): iki yolun farkı ⟨E⟩'de 3.6e-15, ⟨a·c⟩'de 7.8e-16;
  test 1e-9 ile karşılaştırır — son ulp KİLİTLENMEZ. G7 denetçisi aynı
  ⟨E⟩'yi iki yolla daha buldu (ln Z'nin sayısal türevi −5.813693798141;
  Legendre katsayılarının quadrature'ı −5.8136937972).

      tam ⟨E⟩    −5.813693798      tam ⟨a·c⟩   0.645438891
      tam ⟨m²⟩    0.774715585

  DÜRÜST NOT — ESKİ ÖZ-SINAMA NEDEN YETMİYORDU (denetim bulgusu): N → ∞'da
  bağ başına Fisher'in (1964) açık zincir sonucuna (`coth K − 1/K`) inme
  kontrolü (N = 400'de fark 1.1e-16) yalnız l = 0 kanalını görür. (2l+1)
  ağırlığı atılmış bozuk bir yardımcı onu GEÇTİ ama N = 4'te −5.486 verdi;
  o bozulmayı yalnız yardımcının kendi çıktısına yazılmış sabit kilit
  yakalıyordu. Langevin kontrolü büyük-N kolunun sınaması olarak kalıyor;
  N = 4 değerinin bağımsız sınaması YOL B'dir (mutasyonla doğrulandı:
  M_REF_2L1_KILIT_GUNCEL — ağırlık atılıp sabit kilit de "güncellenince"
  YOL B karşılaştırması düşürüyor). Önceki sürümdeki iz bırakmayan
  "bağımsız tek-spin MC" sayıları (T = 1.0 / 2.0; betiği ve logu yoktu)
  kaldırıldı.

PENCERE: T = 0.7, J = 1, h = 0, `HeisenbergModel(step=0.6)`, soğuk
(tekbiçim) başlangıç, 6 zincir (tohum 20261009 + 1000·k), n_burn = 50,
n_sweep = 300. `step` AÇIKÇA sabitlenir: fizik değildir ama blok saçılımını
değiştirir ve toleranslar step 0.6 için ölçüldü (step 0.3'te ⟨E⟩ saçılımı
0.075 → 0.105 büyüyor; 200 blokta en büyük sapma 0.393 > 0.30). Varsayılan
step değişirse bu test etkilenmez.

    MC ⟨E⟩    −5.808599515   (6 zincirin SE'si 0.0939; sapma +0.0051)
    MC ⟨m²⟩    0.775352      (6 zincirin SE'si 0.0093; sapma +0.00064)
    süre      call 0.18 / 0.17 / 0.19 s (üç `--durations` ölçümü, yük
              altında, başka ajanlar koşarken); setup < 0.005 s; iki
              referans yolu birlikte < 1 ms (32 düğümlü quadrature 0.27 ms)

TOLERANSLAR — ELLE SEÇİLMEDİ, 200 BAĞIMSIZ TOHUM BLOĞUNDAN ÖLÇÜLDÜ
(blok b: tohum 20261009 + 7919·b + 1000·k, aynı pencere; b = 0 testin kendi
tohumu):

    doğru kod   ⟨E⟩ sapması  ort +0.0001, std 0.0751, en büyük |d| 0.2324
                ⟨m²⟩ sapması ort +0.00005, std 0.0081, en büyük |d| 0.0259
                iki sapmanın korelasyonu −0.99; 0/200 blok sınırlardan
                birini aşıyor

    TOLERANS    = 0.30   → 4.0σ; 200 blokta görülen en büyük sapmanın 1.29 katı
    TOLERANS_M2 = 0.032  → aynı 4.0σ kuralı; en büyük sapmanın 1.24 katı

DÜZELTME (denetim bulgusu): önceki sürüm toleransı 20 bloğa dayandırıp
"0.30 = saçılımın 5.3 katı, en büyük sapmanın 2.6 katı" diyordu; 20 bloğun
std'si (0.056) şans eseri düşüktü. Sabit tohumla test deterministiktir;
TOLERANS değeri değişmedi, düzelen iddia edilen paydır. Sınırın SE'den değil
blok saçılımından gelmesi bilinçli: 6 zincirlik SE'nin kendisi gürültülü
(200 blokta 0.025 … 0.147); SE'ye bağlı bir sınırla ZZ mutantı 20 bloğun
birinde 4.67 SE'ye kadar yaklaşıyordu.

TRANSKRİPSİYON KİLİDİ BİLİNÇLİ OLARAK YOK: MC değerleri dokstringde yazılı
ama assert edilmiyor. G7'nin tanımıyla kilit bir çıpa değildir ve bu
dosyanın amacı, RNG akışını değiştiren her zararsız değişiklikte DEĞİL
yalnız fizik bozulunca kırmızıya dönmektir.

===========================================================================
KIRMA NOTU — mutasyonla doğrulandı
(`.superpowers/sdd/g7-olcum/duzeltme/spin/`, 200 blok taraması aynı yerde)
===========================================================================

Sütunlar: 200 blokta ortalama sapma (⟨E⟩ / ⟨m²⟩) ve sınırlardan birini aşan
blok sayısı; "tohum" testin kendi bloğudur ve hüküm ona aittir.

ÖLDÜRÜLÜR (testin tohumunda kırmızı):

    M_HEIS_ZZ   dE'de yalnız z bileşeni           +1.17 / −0.092   200/200
    M_KABUL_2T  kabulde T → 2T                     +2.05 / −0.207   200/200
    N3          öneri `normal` → `random`          −1.87 / +0.195   200/200
    N8          öneri kayması 0.3                  −0.52 / +0.056   200/200
    N6          kabulde T → 1.2T (tam +0.458)      +0.46 / −0.047   192/200
    N4, N4b     dE'de x bileşeni düştü             +0.46 / −0.035   188/200
    N9          J → −J (örnekleme VE ölçüm AF)     +0.02 / −0.726   200/200
    M_STENCIL   `roll(-1, axis=1)` → `axis=0`      −0.02 / −0.054   190/200

- N4b (`fark[..., -2:]`) Ising'de satırı DEĞİŞTİRMEZ; bu dosya olmadan
  hızlı şerit onu yalnız kabul oranı transkripsiyon kilidiyle ve
  `test_HEISENBERG_YUKSEK_Tde_TAHMINCI_...` ile düşürüyor (2 failed / 183
  passed) — bu dosyanın değer kattığı yerin kanıtı. Tohumda ⟨E⟩ +0.394, sınırın 0.094
  üstünde; RNG akışı değişirse kaçırma olasılığı ~%6 (12/200).
- N9 ve M_STENCIL'i YALNIZ ⟨m²⟩ öldürür; ⟨E⟩ ikisine YAPISAL olarak kördür.
  N9: iki-parçalı halkada F ve AF'nin ⟨E⟩'si tam olarak aynı, ⟨m²⟩ tam farkı
  −0.7267. M_STENCIL: L = 2'de halka J, 3J, J, 3J olur; tam fark ⟨E⟩'de
  −0.029 (önceki sürümün dediği gibi "gürültü" değil, yapısal ve küçük),
  ⟨m²⟩'de −0.052 (sınırın 1.6 katı); tohumda ⟨m²⟩ −0.047.
- Test yardımcısı mutantları da öldürülür: M13 (ikinci test),
  M_REF_2L1_KILIT_GUNCEL ve M_QUAD_AGIRLIK_1 (YOL A / YOL B ⟨S⟩
  karşılaştırması), M_REF_CAPRAZ_l (⟨a·c⟩ karşılaştırması),
  M_M2_BD_UNUTULDU (⟨m²⟩'de b·d terimi unutuldu; MC ⟨m²⟩ assert'i).

ÇÖZÜNÜRLÜK — DÜRÜST OLUMSUZ: ⟨E⟩'de |Δ| ≲ 0.3 ve ⟨m²⟩'de |Δ| ≲ 0.032
GÖRÜNMEZ; sıcaklık ölçeğinde bu ≈ %13'tür (tam hesap: T × 1.10 → +0.229 /
−0.023, T × 1.15 → +0.344 / −0.035). İki gözlenebilir doğru kodda −0.99
ilişkili olduğundan sıcaklık-benzeri hatalarda ⟨m²⟩ çözünürlük KAZANDIRMAZ;
yalnız ⟨E⟩'nin yapısal olarak göremediği yönleri (J'nin işareti, bağların
eşitsizliği) ekler. Ölçüldü ve testin tohumunda YAŞADI:

    N1  bayat komşu alanı (ikinci alt-örgü)    +0.26 / −0.022    59/200
    N2  paralel güncelleme (maske yok)         +0.26 / −0.023    66/200
    N5  kabulde T → 1.1T (tam +0.229)          +0.23 / −0.023    34/200
    N7  öneri kayması 0.1                      −0.14 / +0.015     2/200

N5 tohumda iki sınıra da çok yakın geçiyor (⟨E⟩ +0.2969, ⟨m²⟩ −0.0317).
Bu dördünü hızlı şeritte BAŞKA testler öldürüyor (bu dosya hariç, `-x`,
ölçüldü): ilk düşen N1 / N2 / N7'de `test_SUREKLI_SIMETRI_...[1.0-...]`,
N5'te `test_ISING_MIKNATISLANMASI_DORT_HANEDE_L_DEN_BAGIMSIZ`; `-x`'siz
tam listede N1'de 20, N2'de 26 test düşüyor (ısı sığası tam sayımı,
Onsager-Yang ve Ising tam sayımı dahil), N7'de 5 (Langevin çıpası dahil).
Yani kapsam açığı yok; bu dosyanın iddiası o kadar değil. Bu sapmaları
yakalayacak sıkılık saniye altında mümkün değil: n_sweep = 900'de ⟨E⟩
saçılımı ancak 0.046'ya iniyor ve blok 0.48-0.59 s sürüyor (60 blok, yük
altında, iki ölçüm); step 1.5'e çıkmak saçılımı düşürmüyor (0.044).

KAPSAM DIŞI: Wolff (M_WOLFF_P) ve binning (M4_BINNING_NAIF) bu dosyaya
karşı yaşar; ucuz çıpaları başka yerde var (Wolff tam sayımı, AR(1) kapalı
formu).

L = 2'YE BAĞIMLILIK — BEYAN: bu çıpa `lattice.py`nin L = 2 çift sayım
konvansiyonuna dayanır ve onu DOLAYLI olarak kilitler. L = 2 yasaklanır ya
da bağlar tekilleştirilirse fizik doğru olsa bile bu test kırmızıya döner
(o zaman halka bağ başına J'ye iner; referans K = J/T ile yeniden
yazılmalıdır). `tests/spin/test_lattice.py` dokstringi "testler L >= 4
kullanır, L = 2'den kaçınır" der; bu dosya o beyanın bilinçli istisnasıdır
(çapraz atıf o dosyaya bu turda eklenemedi: G7 düzeltme turunda yalnız bu
dosya değiştirilebiliyordu).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import spherical_in

from qlab.engine.spin.lattice import SpinState
from qlab.engine.spin.metropolis import run_metropolis
from qlab.engine.spin.models import HeisenbergModel

# ÖZEL AD İMPORTU — GEREKÇE: ikinci testin sınadığı şey TAM OLARAK bu test
# yardımcısıdır; `src/`de kamu bir karşılığı yoktur (kaynağın
# `binder_cumulant`ı ayrı, `test_spin_obs.py`de zaten kilitli). Testin doğal
# yeri `test_cipalar.py`dir; G7 düzeltme turunda yalnız bu dosya
# değiştirilebildiği için taşınmadı. İmport yavaş fixture'ları TETİKLEMEZ
# (fixture yalnız isteyen testte kurulur; dosya 0.6 s'de biter) ve dört
# kipte geçtiği ölçüldü: --import-mode prepend / append / importlib ve
# tests/spin içinden koşum.
from .test_cipalar import _binder_momentlerden

T_HALKA = 0.7
J = 1.0
ADIM = 0.6  # HeisenbergModel step'i: fizik değil, saçılımı belirler
TOHUM = 20261009
N_ZINCIR = 6
N_BURN, N_SWEEP = 50, 300
TOLERANS = 0.30
TOLERANS_M2 = 0.032
N_GAUSS = 32


def _halka_bag_toplami(K: float, N: int, lmax: int = 40) -> float:
    """⟨Σ_bağ s·s'⟩ — N spinlik klasik O(3) halka, bağ başına boyutsuz K. TAM.

    YOL A. `d ln Z / dK`, Z = Σ (2l+1) i_l(K)^N (4π çarpanı oranda
    sadeleşir). i_l / i_0 oranlarıyla yazılır ki büyük N'de taşma olmasın.
    """
    l = np.arange(lmax)
    i0 = spherical_in(0, K)
    r = spherical_in(l, K) / i0
    dr = spherical_in(l, K, derivative=True) / i0
    pay = np.sum((2 * l + 1) * N * r ** (N - 1) * dr)
    payda = np.sum((2 * l + 1) * r**N)
    return float(pay / payda)


def _halka_capraz_bessel(K: float, lmax: int = 40) -> float:
    """⟨a·c⟩ — N = 4 halkada karşı köşe korelasyonu, YOL A (transfer matrisi).

    ⟨s_0·s_2⟩ = Tr(ŝ T² ŝ T²) / Tr T⁴; ŝ, l kanalını l ± 1'e bağlar ve
    Σ_{m,m'} |⟨l m|ŝ|l+1 m'⟩|² = l + 1.
    """
    l = np.arange(lmax + 1)
    rho = spherical_in(l, K) / spherical_in(0, K)
    pay = 2.0 * np.sum((l[:-1] + 1) * rho[:-1] ** 2 * rho[1:] ** 2)
    payda = np.sum((2 * l[:-1] + 1) * rho[:-1] ** 4)
    return float(pay / payda)


def _halka_quadrature(K: float, n: int = N_GAUSS) -> tuple[float, float]:
    """(⟨S⟩, ⟨a·c⟩) — N = 4 halka, YOL B: spherical_in ve transfer matrisi YOK.

    b ve d, a + c'nin alanında analitik küre integraliyle atılır; kalan
    x = a·c integrali Gauss-Legendre (düğümler uç noktaları içermez, r > 0).
    """
    x, agirlik = leggauss(n)
    r = np.sqrt(2.0 + 2.0 * x)
    y = K * r
    w = agirlik * (np.sinh(y) / y) ** 2
    S = 2.0 * np.sum(w * r * (1.0 / np.tanh(y) - 1.0 / y)) / np.sum(w)
    return float(S), float(np.sum(w * x) / np.sum(w))


def test_HEISENBERG_HALKASI_TAM_COZUMLE_ucusuyor():
    """L = 2 Heisenberg, J ≠ 0: Metropolis ⟨E⟩'si ve ⟨m²⟩'si 4 spinlik
    halkanın tam çözümüne oturuyor. Referans iki bağımsız yoldan; pencere,
    ölçülen sayılar, toleranslar ve kör noktalar modül dokstringinde."""
    K = 2.0 * J / T_HALKA  # L = 2: her bag cift sayilir -> bag basina 2J

    # Referansin kendisi. (1) Buyuk-N kolu: N -> sonsuz'da bag basina Langevin
    # (Fisher 1964) -- yalniz l = 0 kanalini gorur.
    assert _halka_bag_toplami(K, 400) / 400 == pytest.approx(
        1.0 / np.tanh(K) - 1.0 / K, rel=1e-12
    )
    # (2) N = 4: YOL A (Bessel) ile YOL B (Bessel'siz quadrature) ayni sayilar.
    S_a, ac_a = _halka_bag_toplami(K, 4), _halka_capraz_bessel(K)
    S_b, ac_b = _halka_quadrature(K)
    assert S_a == pytest.approx(S_b, abs=1e-9), "YOL A / YOL B <S> ayrisiyor"
    assert ac_a == pytest.approx(ac_b, abs=1e-9), "YOL A / YOL B <a.c> ayrisiyor"
    beklenen = -2.0 * J * S_a
    assert beklenen == pytest.approx(-5.813693798, abs=1e-8)
    # |M|^2 = 4 + 2S + 2(a.c + b.d), <b.d> = <a.c>; m = |M| / 4
    beklenen_m2 = (4.0 + 2.0 * S_a + 4.0 * ac_a) / 16.0

    e_ort, m2_ort = [], []
    for k in range(N_ZINCIR):
        sonuc = run_metropolis(
            SpinState.uniform(L=2, n_component=3),
            HeisenbergModel(step=ADIM),
            temperature=T_HALKA,
            coupling=J,
            field=0.0,
            rng=np.random.default_rng(TOHUM + 1000 * k),
            n_sweep=N_SWEEP,
            n_burn=N_BURN,
        )
        e_ort.append(sonuc.energy_series.mean())
        m2_ort.append(np.mean(sonuc.magnetisation_series**2))
    ort = float(np.mean(e_ort))
    ort_m2 = float(np.mean(m2_ort))
    assert abs(ort - beklenen) < TOLERANS, (
        f"L=2 Heisenberg halkasi: MC <E> {ort:.6f}, tam {beklenen:.6f}, "
        f"sapma {ort - beklenen:+.4f} (sinir {TOLERANS})"
    )
    assert abs(ort_m2 - beklenen_m2) < TOLERANS_M2, (
        f"L=2 Heisenberg halkasi: MC <m^2> {ort_m2:.6f}, tam {beklenen_m2:.6f}, "
        f"sapma {ort_m2 - beklenen_m2:+.5f} (sinir {TOLERANS_M2}; AF'de tam 0.048)"
    )


def test_BOOTSTRAP_BINDER_YARDIMCISI_KAPALI_FORM_LIMITLERINE_oturuyor():
    """SINIF: DAİRESEL DEĞİL (cebir). `test_cipalar._binder_momentlerden`
    — Binder kesişiminin bootstrap σ'sını üreten yardımcı — iki kapalı form
    limitine oturmalı: tam düzenli (m sabit, ⟨m⁴⟩ = ⟨m²⟩²) U = 2/3; Gauss
    (⟨m⁴⟩ = 3⟨m²⟩²) U = 0.

    NEDEN: `test_cipalar.py` dokstringinin mutasyon tablosu M13'ü (bu
    yardımcıda `3 → 2`) "YALNIZ yavaş şerit" diye kaydediyor; G7 turunda
    yeniden ölçüldü (bu dosya hariç): hızlı şerit 185/185 YEŞİL; yavaş
    şeritte (20 test) yalnız `test_BINDER_MOMENT_YOLU_ILE_KAYNAK_YOLU_AYNI`
    yakalıyor (1 failed / 19 passed; o testin fixture setup'ı yük altında
    iki ölçümde 122 s ve 68 s). Bu test aynı mutantı ~0 s'de, kaynağa değil
    cebire karşı yakalar. KIRMA NOTU: M13'te düzenli limit 0.5, Gauss
    limiti −0.5 çıkar.

    KAPSAM — DÜRÜST NOT: bu bir TEST YARDIMCISININ çıpasıdır, `src/`
    fiziğinin değil; G7 sayımında "src fizik çıpası" olarak SAYILMAMALIDIR.
    Kaynağın `binder_cumulant`ı aynı limitlerle
    `test_spin_obs.py::test_BINDER_GAUSSTA_SIFIR_TAM_DUZENLIDE_UCTE_IKI`de
    zaten kilitli. Doğal yeri yardımcının yanı (`test_cipalar.py`); özel ad
    importunun gerekçesi import satırında. Dizi girdisi de sınanır, çünkü
    fixture yardımcıyı (L, T) şekilli moment dizileriyle çağırır.
    """
    m = np.array([[0.37, 0.91], [0.05, 1.0]])
    assert np.allclose(_binder_momentlerden(m**2, m**4), 2.0 / 3.0, atol=1e-12, rtol=0)

    s2 = np.array([0.29, 1.7e-3])
    assert np.allclose(_binder_momentlerden(s2, 3.0 * s2**2), 0.0, atol=1e-12, rtol=0)
