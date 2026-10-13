"""G6 — Maxwell ↔ Schrödinger `v_g/v_p` bağı: YENİDEN ÖLÇÜLDÜ ve kilitlendi (13.10.2026).

README'nin Modül 5 "İKİNCİL" bağı Plan J'nin ön ölçümünden geliyordu, bu
modülde yeniden ölçülmemişti ve sayı yazılmamıştı. Bu dosya bağı AYNI
gerçek-uzay tahmincisiyle iki motordan yeniden üretir. Katman sözleşmesi
bozulmaz: iki motoru import eden BU TEST dosyasıdır; `src/`de hiçbir
motorlar arası import yoktur ve eklenmez (kenar tarayıcısı `src/qlab/engine`
ağacını tarar, `tests/`i değil).

TAHMİNCİ (iki motorda BİREBİR aynı fonksiyon, `_hizlar`): girdi GERÇEL alan
anlık görüntüleri `u(x, t_n)` (Maxwell'de `E_z`, Schrödinger'de `Re ψ`).
Analitik sinyal `a = u + i·H[u]` (FFT, negatif dalga sayıları sıfır);
`w = |a|²`. `v_g` = ağırlık merkezinin eğimi; `v_p = ω̄ / k̄`, burada
`k̄ = <∂ₓ arg a>_w`, `ω̄ = −<∂ₜ arg a>_w` (komşu nokta / ardışık kare
faz farkları). Tahminci ne Yee'nin ne split-operator'ün dispersiyonunu
bilir.

SINIFLAR — HANGİSİ ÇIPA, HANGİSİ DEĞİL (ölçülerek ayrıldı):

* MAXWELL `v_p` ve `v_g` — ÇIPA (bağımsız analitik referans, ŞEMA sınıfı):
  `engine/em/` ağacında hiçbir dispersiyon bağıntısı yazılı değildir; Yee'nin
  `ω(k)`'sı curl güncellemesinden ORTAYA ÇIKAR. Referans kapalı formdur:
  `sin(ωΔt/2) = S sin(kΔx/2)`. ÖLÇÜLDÜ (S = 0.5, s = 60 hücre):
      λ = 40: v_g −6.59e-06, v_p −6.54e-06 (Yee'ye bağıl); oran −5.5e-08
      λ = 80: v_g −6.53e-06, v_p −6.52e-06;                 oran −1.4e-08
  Ortak −6.5e-06 tahmincinin sonlu paket yanlılığıdır ve 1/s² ölçeklenir
  (s = 40/60/100: 1.5e-05 / 6.5e-06 / 2.35e-06); oranda sadeleşir.
* MAXWELL bağ değeri — FİZİK limiti: `1 − v_g/v_p` λ = 40 → 80'de
  1.544559e-03 → 3.856944e-04, oran 4.004619 (Yee kapalı formu 4.004626):
  kütlesiz dalganın `v_g/v_p = 1` sürekli limitine İKİNCİ mertebeden gidiş.
* SCHRÖDINGER `v_g` — ÇEVİRİ DENETİMİ, çıpa DEĞİL: kinetik faz
  `exp(−i k²Δt/2m)` koda yazılıdır ve V = 0'da adım TAMDIR; ağırlık merkezi
  `k0/m`yi 3.4e-15 ile GERİ OKUR (tautolojinin kendisi bu sayıdır).
* SCHRÖDINGER `v_p` — tahmincinin ölçümü, kapalı form referansı
  `ω̄/k̄ = <T>/k0 = (k0² + 1/(4σ²))/(2 m k0)`: −3.1e-08. Naif geri okuma
  `k0/2m` 6.25e-04 YANLIŞTIR (kuantum potansiyeli terimi) — tahminci onu
  görür. AMA FİZİK ÇIPASI DEĞİLDİR ve bu da ÖLÇÜLDÜ: sabit bir `V0` (enerji
  sıfırı) `v_p`yi tam `V0/k0` kaydırır, `v_g`yi değiştirmez (V0 = 0.05:
  v_g/v_p 1.99875 → 1.90363). Schrödinger'in faz hızı bir AYAR (gauge)
  büyüklüğüdür; "v_g/v_p = 2" yalnız V = 0 ayarında bir cümledir. Bağın
  FİZİK taşıyan yarısı Maxwell'in `v_p`sidir (E gözlenebilirdir).

MUTANT (`.superpowers/sdd/araclar/mut.py`): `propagator.py`de yarım-potansiyel
fazı `self.potential - self.potential.min()` ile kurulursa (sanal zamanda
taşmayı önlemek için yazılabilecek bir "sıfır ayarı") bütün |ψ|² ve enerji
gözlenebilirleri değişmez — bugünkü süit (tests/tdse + tests/bridge) KÖRDÜR;
bu dosyanın ayar testi düşer. O mutant FİZİKSEL OLARAK EŞDEĞERDİR (tek koşumda
yalnız küresel faz); test bir FİZİK değil, propagatörün dokstringindeki
operatörün (e^{−iVdt/2}) SÖZLEŞME kilididir.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.em.boundary import PEC
from qlab.engine.em.grid import FieldState
from qlab.engine.em.medium import Medium
from qlab.engine.em.yee import _adim, yee_coefficients
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.initial import gaussian_packet
from qlab.engine.tdse.propagator import SplitOperatorPropagator

#: Maxwell penceresi: S, zarf genişliği (hücre), başlangıç merkezi, ızgara,
#: iki yarının ayrışma bekleyişi (adım), kare sayısı ve kare aralığı (adım).
S, ZARF, X0, N, BEKLE, KARE, HER = 0.5, 60.0, 600.0, 2400, 1200, 201, 2
#: Schrödinger penceresi (atomik birim): k0, σ (|ψ|² std), m, ızgara, dt, adım.
K0, SIGMA, KUTLE, NX, UZUNLUK, DT, ADIM, XS = 2.0, 10.0, 2.0, 2048, 204.8, 0.1, 200, -20.0


def _analitik(u: np.ndarray) -> np.ndarray:
    n = u.size
    h = np.zeros(n)
    h[0] = 1.0
    h[1 : (n + 1) // 2] = 2.0
    if n % 2 == 0:
        h[n // 2] = 1.0
    return np.fft.ifft(np.fft.fft(u) * h)


def _hizlar(kareler: np.ndarray, x: np.ndarray, dx: float, dt: float) -> tuple[float, float]:
    """AYNI tahminci, iki motor. Dönüş (v_g, v_p)."""
    a = np.array([_analitik(u) for u in kareler])
    w = np.abs(a) ** 2
    xc = (w * x).sum(axis=1) / w.sum(axis=1)
    v_g = float(np.polyfit(dt * np.arange(len(kareler)), xc, 1)[0])
    wx = 0.5 * (w[:, 1:] + w[:, :-1])
    k = float((wx * np.angle(a[:, 1:] * np.conj(a[:, :-1]))).sum() / wx.sum()) / dx
    wt = 0.5 * (w[1:] + w[:-1])
    om = -float((wt * np.angle(a[1:] * np.conj(a[:-1]))).sum() / wt.sum()) / dt
    return v_g, om / k


def _maxwell(lam: float) -> tuple[float, float]:
    """1B Yee, vakum, PEC duvarlar uzakta. Başlangıç YALNIZ E (H = 0): paket
    iki eşit yarıya ayrılır, sağa giden yarı izlenir — kurulumda hiçbir
    dispersiyon bağıntısı KULLANILMAZ."""
    co = yee_coefficients(Medium.vacuum(N), courant=S, boundary=PEC())
    i = np.arange(N, dtype=float)
    e0 = np.exp(-((i - X0) ** 2) / (4.0 * ZARF**2)) * np.cos(2.0 * np.pi / lam * (i - X0))
    st = FieldState(e0, np.zeros(N - 1), 1)
    for _ in range(BEKLE):
        st = _adim(st, co)
    kar = []
    for _ in range(KARE):
        kar.append(st.e.copy())
        for _ in range(HER):
            st = _adim(st, co)
    kar = np.array(kar)
    assert float(np.abs(kar[:, -50:]).max()) < 1e-20  # duvar paketi görmedi
    sag = i > X0
    return _hizlar(kar[:, sag], i[sag], 1.0, HER * S)


def _yee(lam: float) -> tuple[float, float]:
    """Yee 1B kapalı formu: (v_g, v_p), dx = 1, dt = S."""
    k = 2.0 * np.pi / lam
    om = 2.0 * np.arcsin(S * np.sin(k / 2.0)) / S
    return float(np.cos(k / 2.0) / np.sqrt(1.0 - (S * np.sin(k / 2.0)) ** 2)), float(om / k)


def _schrodinger(V0: float = 0.0) -> tuple[float, float]:
    g = Grid1D(n=NX, length=UZUNLUK)
    psi = gaussian_packet(g, x0=XS, k0=K0, sigma=SIGMA)
    p = SplitOperatorPropagator(grid=g, potential=np.full(NX, V0), dt=DT, mass=KUTLE)
    kar = []
    for _ in range(ADIM + 1):
        kar.append(psi.real.copy())
        psi = p.step(psi)
    kar = np.array(kar)
    assert float(np.abs(kar[:, :20]).max()) < 1e-7  # periyodik kuyruk ihmal edilebilir
    return _hizlar(kar, g.x, g.dx, DT)


def test_MAXWELL_v_g_v_p_YEE_KAPALI_FORMUNA_CIPALI_ve_ORAN_IKINCI_MERTEBEDEN_BIRE_GIDER():
    """ÇIPA (şema: Yee dispersiyonu koda yazılı DEĞİL) + FİZİK limiti (oran → 1)."""
    bir_eksi = []
    for lam in (40.0, 80.0):
        vg, vp = _maxwell(lam)
        rg, rp = _yee(lam)
        # 2e-5: ölçülen ortak yanlılık 6.5e-06 (1/s² ölçekli, tahmincinin), ~3 kat pay.
        assert abs(vg / rg - 1.0) < 2e-5 and abs(vp / rp - 1.0) < 2e-5, (lam, vg, vp)
        # Oranda yanlılık sadeleşir: ölçülen 5.5e-08 / 1.4e-08.
        assert abs((vg / vp) / (rg / rp) - 1.0) < 5e-7, lam
        assert vg < vp < 1.0  # Yee: v_g < v_p < c (normal dispersiyon)
        bir_eksi.append(1.0 - vg / vp)
    # Kütlesiz limit: kΔx yarılanınca 1 − v_g/v_p dörde bölünür (ölçülen 4.004619).
    (rg40, rp40), (rg80, rp80) = _yee(40.0), _yee(80.0)
    assert bir_eksi[0] / bir_eksi[1] == pytest.approx(
        (1.0 - rg40 / rp40) / (1.0 - rg80 / rp80), abs=1e-4)
    assert 1.4e-3 < bir_eksi[0] < 1.7e-3 and 3.5e-4 < bir_eksi[1] < 4.2e-4


def test_SCHRODINGER_v_g_GERI_OKUMA_v_p_KAPALI_FORM_ve_BAG_IKI():
    """`v_g` ÇEVİRİ DENETİMİ (geri okuma); `v_p` kapalı forma karşı ölçüm."""
    vg, vp = _schrodinger()
    # Geri okuma: V = 0'da split-operator adımı tam, merkez k0/m'yi yuvarlama
    # düzeyinde verir (ölçülen 3.4e-15). Bu bir çıpa DEĞİL, tautolojinin sayısı.
    assert abs(vg - K0 / KUTLE) < 1e-12
    kapali = (K0**2 + 1.0 / (4.0 * SIGMA**2)) / (2.0 * KUTLE * K0)
    assert abs(vp / kapali - 1.0) < 3e-7  # ölçülen −3.1e-08
    assert vp / (K0 / (2.0 * KUTLE)) - 1.0 > 5e-4  # naif k0/2m'den 6.25e-04 uzak
    # Bağın Schrödinger yarısı: 2/(1 + 1/(4σ²k0²)) = 1.99875; Maxwell'inki ~1.
    assert abs(vg / vp - 2.0 / (1.0 + 1.0 / (4.0 * SIGMA**2 * K0**2))) < 1e-6


def test_SCHRODINGER_v_p_BIR_AYAR_BUYUKLUGU_V0_KAYDIRIR_v_g_KAYMAZ():
    """Sabit V0 → v_p tam V0/k0 kayar, v_g değişmez: v_p FİZİK çıpası olamaz.

    SÖZLEŞME kilidi de budur: propagatör `e^{−iVdt/2}`yi yazdığı gibi
    uygulamalı. Potansiyeli `min`e göre sıfırlayan mutant tests/tdse +
    tests/bridge'de HAYATTA kalır (|ψ|² ve enerji gözlenebilirleri ayara
    kördür), burada düşer."""
    vg0, vp0 = _schrodinger(0.0)
    for V0 in (0.05, -0.3):
        vg, vp = _schrodinger(V0)
        assert abs(vg - vg0) < 1e-13
        assert abs((vp - vp0) - V0 / K0) < 1e-10, V0  # ölçülen ≤ 2e-16
