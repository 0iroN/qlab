"""G7 -- tdse hızlı şeridinin (`-m "not slow"`) KÜTLE ve 2B FİZİK ÇIPALARI.

NEDEN BU DOSYA VAR (ölçüldü 09.10.2026, kayıtlar
`.superpowers/sdd/g7-olcum/tdse/`): tdse'nin hızlı şeridinde fizik çıpası
VARDI -- harmonik taban durumu `E0 = omega/2`, serbest yayılma `sigma(t)`,
grup hızı, statik potansiyelde enerji korunumu, norm, spektral türev. Ama
hepsi **kütle = 1** ve dinamik olanların hepsi **1B**. Altı mutantlık tur
(her biri hızlı ve yavaş şeritte ayrı koşuldu) iki kör nokta buldu:

  * KÜTLE. `propagator.py` kinetik fazında `/ self.mass` -> `* self.mass`
    mutantını hızlı şeritte YALNIZ `test_from_md.py`nin regresyon sabitleri
    (`KENAR_SAYISI`, `KATLANMA_BEDELI`) kırmızıya çevirdi. Onlar modülün
    kendi önceki çıktısıdır, bağımsız bir referans değil; yavaş şerit (her
    testi m = 1) mutantı hiç görmedi. `potentials.harmonic` içinde `mass`
    çarpanının düşmesini İKİ şerit de kaçırdı: m = 1'de bu mutant özdeştir.
  * 2B. `fft2/ifft2` -> `fft/ifft` (kinetik adım yalnız y ekseninde) ve
    `gaussian_packet_2d`de `kx0`/`ky0`nın eksen değiştirmesi: ikisini de
    YALNIZ yavaş şerit (çift yarık, ~25 s'lik fixture) yakaladı. Hızlı
    şeritteki tek 2B dinamik testi normu ölçer ve iki mutant da normu korur.

Buradaki iki çıpa bu boşluğu kapatır. Süre ÖLÇÜLDÜ (denetim sonrası
düzeltmede, `--durations=0`, dört koşu, altı ajan aynı anda test koşarken):
çağrı 0.02-0.03 s + 0.01 s, kurulum/söküm her biri < 0.005 s, dosya 0.26-0.31 s
-- `slow` işareti almaz. Referansların ikisi de kapalı biçimdir, motorun
kendi önceki çıktısı değildir. Yalnız kamu API'si kullanılır. Denetim ve
düzeltme kayıtları: `.superpowers/sdd/g7-olcum/denetim/tdse/` ve
`.superpowers/sdd/g7-olcum/duzeltme/tdse/`.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid1D, Grid2D
from qlab.engine.tdse.initial import gaussian_packet, gaussian_packet_2d
from qlab.engine.tdse.propagator import SplitOperatorPropagator, evolve


def test_kutleli_harmonik_koherent_durum_analitik_yorungeyi_izler():
    """m = 4 kütleli harmonik osilatörde koherent durum -- KAPALI ÇÖZÜM.

    Referans (ders kitabı, kuadratik potansiyelde Ehrenfest teoremi TAMDIR):
    `V = m omega^2 x^2 / 2` taban genişliğinde (`sigma0 = 1/sqrt(2 m omega)`)
    `x0` kadar kaydırılmış bir Gauss paketi için
        <x>(t) = x0 cos(omega t)
        <p>(t) = -m omega x0 sin(omega t)
        sigma_x(t) = sigma0                (paket nefes almaz)
    Kütle bu üç ifadeye ÜÇ ayrı yerden girer: potansiyelin katsayısı,
    kinetik fazın paydası ve başlangıç genişliği. Herhangi birinde kütle
    yanlış bağlanırsa paket ya yanlış frekansla salınır ya da genişliği
    `omega'` ile nefes alır; m = 1'de bu hataların hepsi görünmezdir.

    ÖLÇÜLDÜ (n = 256, L = 24, dt = 0.01, 1600 adım, 17 kare, t = 0..16;
    ~1.27 periyot -- tam periyot katında durulmadı, çünkü 4 kat yanlış
    frekans tam periyotta aynı noktaya döner):
        max|<x> - x0 cos wt|        = 1.649e-05
        max|<p> + m w x0 sin wt|    = 2.750e-05
        max|sigma_x/sigma0 - 1|     = 3.109e-06
    Kalan YUVARLAMA DEĞİL, Strang adımının O(dt^2) hatasıdır: dt = 0.02'de
    sırasıyla 6.596e-05 / 1.100e-04 / 1.244e-05 (oran 4.0). Deterministik
    bir kalan olduğu için platform son-bit farklarına duyarlı değildir.
    Kenar olasılığı (dış 2 bohr) son karede 1.9e-29 -- periyodik sarma yok.

    TOLERANS: 1e-4 / 2e-4 / 2e-5 -- ölçülenin sırasıyla 6.1x / 7.3x / 6.4x'i.

    KIRMA (mut.py ile, bu dosyaya karşı -- hepsi kırmızı; sapmalar GERÇEK
    mutant koduyla ölçüldü, denetim ve düzeltme turlarında `mut_metrik.py`):
      * kinetik `/ self.mass` -> `* self.mass` (etkin frekans m*omega = 2.0):
        max|d<x>| = 3.668 (toleransın 3.7e4 katı). Önceden hızlı şeritte
        yalnız regresyon sabitleri yakalıyordu.
      * `potentials.harmonic`te `mass` düşer (etkin frekans omega/sqrt(m) =
        0.25): max|d<x>| = 3.941 (3.9e4 kat). Önceden İKİ şerit de
        kaçırıyordu.
      * Strang -> Lie-Trotter (birinci mertebe): başlangıçtaki eksik yarım
        tekme yörüngeyi ~(dt/2) omega x0 = 5e-3 kaydırır: max|d<x>| =
        4.991e-03 (50 kat), max|dsigma/sigma0| = 1.252e-03 (63 kat).
        DİKKAT: <p> bu mutanta KÖRDÜR (3.020e-05, toleransın 0.15'i) --
        baştaki ve sondaki yarım tekmeler <p>'de birbirini götürür. Bu
        mutantı enerji korunumu testi zaten yakalıyordu; bu test ikinci,
        bağımsız bir kanıt.
      * `Grid1D.k` işareti ters: `k2` değişmez, dinamik aynıdır; <x> ve
        sigma yeşil kalır, YALNIZ <p> assert'i yakalar: max|d<p>| = 7.980
        (4e4 kat). <p> assert'inin x/sigma'dan bağımsız değeri budur.
    """
    m, omega, x0 = 4.0, 0.5, 2.0
    grid = Grid1D(n=256, length=24.0)
    sigma0 = 1.0 / np.sqrt(2.0 * m * omega)
    potential = potentials.harmonic(grid, omega=omega, mass=m)
    psi0 = gaussian_packet(grid, x0=x0, k0=0.0, sigma=sigma0)
    prop = SplitOperatorPropagator(grid=grid, potential=potential, dt=0.01, mass=m)

    frames, times = evolve(prop, psi0, n_steps=1600, save_every=100)
    assert frames.shape == (17, grid.n)

    w = np.abs(frames) ** 2
    toplam = w.sum(axis=1)
    x_ort = (w * grid.x).sum(axis=1) / toplam
    sigma_x = np.sqrt((w * (grid.x[None, :] - x_ort[:, None]) ** 2).sum(axis=1) / toplam)
    wk = np.abs(np.fft.fft(frames, axis=1)) ** 2
    p_ort = (wk * grid.k).sum(axis=1) / wk.sum(axis=1)

    x_ref = x0 * np.cos(omega * times)
    p_ref = -m * omega * x0 * np.sin(omega * times)

    assert np.max(np.abs(x_ort - x_ref)) < 1e-4
    assert np.max(np.abs(p_ort - p_ref)) < 2e-4
    assert np.max(np.abs(sigma_x / sigma0 - 1.0)) < 2e-5


def test_2b_serbest_kutleli_paket_kare_olmayan_izgarada_kapali_cozumu_izler():
    """2B serbest Gauss paketi, m = 2, kare OLMAYAN ızgara -- KAPALI ÇÖZÜM.

    Referans (serbest Schrödinger denkleminin Gauss çözümü, her eksende
    ayrı ayrı):
        <x>(t) = x0 + kx0 t / m,    <y>(t) = y0 + ky0 t / m
        sigma(t) = sigma0 sqrt(1 + (t / (2 m sigma0^2))^2)   (iki eksende)
    `kx0 != ky0` ve `nx != ny`, `lx != ly`, `dx != dy` bilinçli: DİNAMİĞE
    bağlı bir eksen hatası (kinetik adımı tek eksende atmak, bir eksenin
    dalga sayısını öbür eksenin adımıyla kurmak, momentumu yanlış eksene
    yazmak) kare ızgarada ve simetrik pakette kendini gizleyebilir. Saf
    etiket takası (`Grid2D.X` <-> `Grid2D.Y`, her yerde tutarlı) serbest
    harekette fiziksel olarak EŞDEĞERDİR; bu test onu yakalamaz ve
    yakalamamalıdır (ölçüldü: |d<x>| = 1.9e-13, toleransın 0.04'ü).

    GEOMETRİ (64x54 ızgara, 40x36 bohr; dx = 0.625, dy = 0.667): paket
    başlangıçta en yakın kenara 8.5 sigma0, t = 6'da 8.2 sigma(t) uzakta.
    Denetimden önce 64x48 / 40x32 idi (7.5 sigma0): periyodik görüntünün
    kuyruğu y merkezinde 8.032e-13 bırakıyor, merkez toleransının %16'sını,
    hücre-merkezli ızgarada (`-L/2 + (j + 1/2) dx`, meşru bir konvansiyon)
    %70'ini yiyordu. ly 32 -> 36, ny 48 -> 54 (dy aynı) kuyruğu
    yuvarlama düzeyinin altına indirdi.

    ÖLÇÜLDÜ (dt = 0.1, 60 adım, t = 6): dört kalanın hepsi yuvarlama
    düzeyinde, en büyüğü |<x> - ref| = 4.3e-15 (öbürleri <= 1.7e-15). Son
    bitleri platforma göre oynar (NumPy SIMD X86_V2'ye indirilince
    |<y> - ref| 1.1e-16 -> 1.0e-15); tolerans onları kilitlemez. V = 0'da
    split-operator adımı tam olduğu için kalan dt'ye bağlı değildir (tek
    adımda dt = 6: en büyüğü 3.2e-15). Hücre-merkezli ızgarada: 4.2e-15.

    TOLERANS: merkezler 5e-12 bohr, genişlikler bağıl 3e-11 (ölçülen
    kalanın ~1e3 / ~2e4 katı). Bilerek sıkılaştırılmadı: aşağıdaki
    mutantların en zayıfı bile 1.7e-01 sapar. Ölçülen mutantlar içinde
    5e-12'nin altına inmek hiçbir fizik mutantını ek olarak öldürmez;
    ilk kırmızıya dönen EŞDEĞER X/Y takası (1.9e-13) olurdu.

    KIRMA (mut.py ile, bu dosyaya karşı -- hepsi kırmızı; sapmalar GERÇEK
    mutant koduyla ölçüldü, `mut_metrik.py`):
      * propagator 2B'de `fft2/ifft2` -> `fft/ifft`: x yönünde kinetik adım
        hiç atılmaz, paket x0'da kalır ve x'te yayılmaz: max|d merkez| =
        3.000, bağıl dsigma_x = 6.367e-02. Önceden YALNIZ yavaş şerit (çift
        yarık) yakalıyordu; normu koruduğu için hızlı şeritteki 2B norm
        testi kördü.
      * `gaussian_packet_2d`de faz `kx0*Y + ky0*X`: paket (-0.25, +0.5)
        hızıyla gider: max|d merkez| = 4.500. Önceden YALNIZ yavaş şerit.
        YAKALAYAN MERKEZDİR. Genişlik bu mutanta fiziksel olarak kördür
        (serbest yayılma k0'a bağlı değildir): dsigma_x = 1.8e-12, yeşil.
        sigma_y assert'i de kırmızıdır (2.1e-09, toleransın 68 katı) ama
        bu FİZİK DEĞİL: mutant paketi y = +4'e, kenara 6.6 sigma'ya taşır
        ve periyodik kuyruk büyür.
      * `Grid2D.k2`de `ky` `dx` ile kurulur: y'deki kinetik terim
        (dy/dx)^2 = 1.138 kat büyür: |d<y>| = 2.067e-01, bağıl dsigma_y =
        1.799e-02. dx = dy olan bir ızgarada görünmez olurdu.
      * `gaussian_packet_2d`de zarf paydası `4 sigma^2` -> `2 sigma^2`
        (genlik std'si): bağıl dsigma = 1.724e-01 (iki eksende).
      * kinetik `/ self.mass` -> `* self.mass`: hız k0*m: max|d merkez| =
        8.940.
    """
    m, kx0, ky0, x0, y0, sigma0 = 2.0, 1.0, -0.5, -3.0, 1.0, 2.0
    grid = Grid2D(nx=64, ny=54, lx=40.0, ly=36.0)
    psi0 = gaussian_packet_2d(grid, x0=x0, y0=y0, kx0=kx0, ky0=ky0, sigma=sigma0)
    prop = SplitOperatorPropagator(
        grid=grid, potential=np.zeros((grid.nx, grid.ny)), dt=0.1, mass=m
    )

    frames, times = evolve(prop, psi0, n_steps=60, save_every=60)
    t = float(times[-1])
    assert abs(t - 6.0) < 1e-12

    w = np.abs(frames[-1]) ** 2
    toplam = w.sum()
    x_ort = float((w * grid.X).sum() / toplam)
    y_ort = float((w * grid.Y).sum() / toplam)
    sigma_x = float(np.sqrt((w * (grid.X - x_ort) ** 2).sum() / toplam))
    sigma_y = float(np.sqrt((w * (grid.Y - y_ort) ** 2).sum() / toplam))
    sigma_ref = sigma0 * np.sqrt(1.0 + (t / (2.0 * m * sigma0**2)) ** 2)

    assert abs(x_ort - (x0 + kx0 * t / m)) < 5e-12
    assert abs(y_ort - (y0 + ky0 * t / m)) < 5e-12
    assert abs(sigma_x / sigma_ref - 1.0) < 3e-11
    assert abs(sigma_y / sigma_ref - 1.0) < 3e-11
