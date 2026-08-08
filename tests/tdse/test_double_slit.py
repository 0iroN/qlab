"""2B çift yarık — girişim deseninin gerçekten oluştuğunun doğrulanması.

Görev tanımının ham kriteri (find_peaks, yükseklik >= tepe'nin %5'i, >=3 tepe)
tek başına zayıftır: aşağıdaki testler ÖLÇEREK gösteriyor ki bu kriter tek
yarıklı bir kontrol duvarından da (Fresnel kırınım saçakları) geçiyor — bkz.
`test_single_slit_control_does_not_show_genuine_interference`. Bu yüzden
"gerçekten girişim mi?" sorusu üç bağımsız ölçüme dayanıyor:

  1. Simetri: y=0 etrafında yansıma simetrisi (geometri bunu zorunlu kılıyor).
  2. Tek yarık kontrolü: aynı paket + aynı duvar, ama TEK yarıkla neredeyse
     aynı ham tepe sayısı çıkıyor; ayırt edici olan saçakların MERKEZE GÖRE
     BAĞIL yüksekliği (gerçek girişimde zarfa yakın, kırınım saçağında çok
     daha zayıf).
  3. Saçak aralığı: sin(theta) = m*lambda/d uzak-alan tahminiyle kaba
     mertebe karşılaştırması (bu geometri temiz Fraunhofer rejiminde DEĞİL;
     bkz. ilgili testin docstring'i).

Trotter rezonans kontrolü (9. görevde keşfedilen split-operator rezonansı):
bu ızgarada (nx=ny=512, lx=ly=200 -> dx=dy=0.390625) Nyquist/eksen başına
~8.042, köşede gerçek |k|_max ~11.374. dt=0.01 için rezonans k'ları
(k=sqrt(4*pi*n/dt), n=1,2,3) sırasıyla ~35.4, ~50.1, ~61.4 — hepsi
11.374'ün çok üzerinde. Yani bu ızgara 1B tünelleme ızgarasına (dx~0.098)
göre KABA olduğundan en yüksek temsil edilebilir |k| düşük kalıyor ve
rezonanslar çok uzakta duruyor: dt=0.01 bu ızgarada güvenli, değişiklik
gerekmedi.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid2D
from qlab.engine.tdse.initial import gaussian_packet_2d
from qlab.engine.tdse.propagator import SplitOperatorPropagator, evolve
from qlab.observable import tdse_obs_2d

# Görev tanımındaki geometri/zaman parametreleri — aşağıdaki tüm testlerde
# (birebir kopya test hariç, o kendi yerel değişkenlerini taşıyor) ortak
# kullanılıyor.
_WALL_X = 0.0
_THICKNESS = 1.0
_SLIT_SEPARATION = 10.0
_SLIT_WIDTH = 2.0
_HEIGHT = 1e4
_DT = 0.01
_N_STEPS = 2000
_SCREEN_X = 40.0
_PEAK_HEIGHT_FRACTION = 0.05  # görev tanımının ham eşiği


def _make_grid() -> Grid2D:
    return Grid2D(nx=512, ny=512, lx=200.0, ly=200.0)


def _make_packet(grid: Grid2D) -> np.ndarray:
    return gaussian_packet_2d(grid, x0=-40.0, y0=0.0, kx0=2.0, ky0=0.0, sigma=8.0)


def _single_slit_potential(grid: Grid2D) -> np.ndarray:
    """Kontrol grubu: `potentials.double_slit` ile AYNI duvar kalınlığı ve
    yarık genişliği, ama y=0'da ortalanmış TEK yarık.

    `qlab.engine.tdse.potentials` içinde tek yarık için ayrı bir fonksiyon
    yok (bu görevin arayüzü sadece `double_slit`'i listeliyor); bu yüzden
    aynı inşa mantığı, sadece testin kontrol grubu için, burada tekrarlanıyor.
    """
    in_wall = np.abs(grid.X - _WALL_X) <= 0.5 * _THICKNESS
    in_slit = np.abs(grid.Y) <= 0.5 * _SLIT_WIDTH
    return np.where(in_wall & ~in_slit, _HEIGHT, 0.0)


def _evolve_and_measure_screen(
    grid: Grid2D, potential: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    psi = _make_packet(grid)
    prop = SplitOperatorPropagator(grid=grid, potential=potential, dt=_DT)
    frames, _ = evolve(prop, psi, n_steps=_N_STEPS, save_every=_N_STEPS)
    return tdse_obs_2d.screen_profile(frames[-1], grid, screen_x=_SCREEN_X)


def _side_peak_ratio(profile: np.ndarray) -> float:
    """Merkez dışındaki en güçlü tepenin, merkez tepeye göre bağıl yüksekliği.

    0.0 dönerse (5% eşiğinde) hiç yan tepe bulunamamış demektir.
    """
    center = float(np.max(profile))
    peaks, _ = find_peaks(profile, height=_PEAK_HEIGHT_FRACTION * center)
    side_heights = [profile[p] for p in peaks if profile[p] < center]
    if not side_heights:
        return 0.0
    return float(max(side_heights) / center)


@pytest.fixture(scope="module")
def grid_512() -> Grid2D:
    return _make_grid()


@pytest.fixture(scope="module")
def double_slit_profile(grid_512: Grid2D) -> tuple[np.ndarray, np.ndarray]:
    potential = potentials.double_slit(
        grid_512,
        wall_x=_WALL_X,
        thickness=_THICKNESS,
        slit_separation=_SLIT_SEPARATION,
        slit_width=_SLIT_WIDTH,
        height=_HEIGHT,
    )
    return _evolve_and_measure_screen(grid_512, potential)


@pytest.fixture(scope="module")
def single_slit_profile(grid_512: Grid2D) -> tuple[np.ndarray, np.ndarray]:
    potential = _single_slit_potential(grid_512)
    return _evolve_and_measure_screen(grid_512, potential)


def test_norm_and_density_shapes():
    grid = Grid2D(nx=128, ny=128, lx=60.0, ly=60.0)
    psi = gaussian_packet_2d(grid, x0=-15.0, y0=0.0, kx0=1.0, ky0=0.0, sigma=4.0)
    # rel=0.0: varsayılan rel=1e-6 yazılan abs=1e-12'yi ezmesin. Ölçülen sapma
    # 2.22e-16 (bkz. b1-1a-tolerans-raporu); abs=1e-15 ~4.5x payla bağlayıcı.
    assert tdse_obs_2d.norm(psi, grid) == pytest.approx(1.0, abs=1e-15, rel=0.0)
    assert tdse_obs_2d.density(psi).shape == (128, 128)


def test_screen_profile_returns_matching_lengths():
    grid = Grid2D(nx=128, ny=128, lx=60.0, ly=60.0)
    psi = gaussian_packet_2d(grid, x0=-15.0, y0=0.0, kx0=1.0, ky0=0.0, sigma=4.0)
    y_axis, profile = tdse_obs_2d.screen_profile(psi, grid, screen_x=-15.0)
    assert y_axis.shape == profile.shape == (128,)


@pytest.mark.slow
def test_double_slit_produces_multiple_interference_maxima():
    grid = Grid2D(nx=512, ny=512, lx=200.0, ly=200.0)
    psi = gaussian_packet_2d(grid, x0=-40.0, y0=0.0, kx0=2.0, ky0=0.0, sigma=8.0)
    potential = potentials.double_slit(
        grid, wall_x=0.0, thickness=1.0, slit_separation=10.0, slit_width=2.0, height=1e4
    )
    prop = SplitOperatorPropagator(grid=grid, potential=potential, dt=0.01)
    frames, _ = evolve(prop, psi, n_steps=2000, save_every=2000)

    _, profile = tdse_obs_2d.screen_profile(frames[-1], grid, screen_x=40.0)
    peaks, _ = find_peaks(profile, height=0.05 * np.max(profile))

    assert len(peaks) >= 3, (
        f"Girişim deseni beklenirken {len(peaks)} tepe bulundu — "
        "tek yarık gibi davranıyor."
    )


@pytest.mark.slow
def test_double_slit_pattern_is_symmetric_and_centered(
    double_slit_profile: tuple[np.ndarray, np.ndarray],
):
    """Izgara, potansiyel ve paket (y0=0, ky0=0) y=0 etrafında tam yansıma
    simetrik olduğundan ekran deseni de simetrik OLMALI.

    UYARI: basit `profile[::-1]` karşılaştırması YANLIŞTIR. Grid2D'nin y
    ekseni [-ly/2, +ly/2) yarı-açık FFT konvansiyonuyla asimetriktir (örn.
    y=-100 noktasının +100'de bir eşi yoktur, o index periyodiklikte
    kendine sarar). Doğru eşleştirme dairesel indekstir: j = (-i) mod n,
    yani y[i] ile y[j] = -y[i] eşleşir (i=0 hariç; o nokta periyodiklik
    nedeniyle kendisiyle eşleşir ve yoğunluğu ihmal edilebilir düzeydedir).
    """
    y, profile = double_slit_profile
    n = len(y)
    mirror_index = (-np.arange(n)) % n

    # Eşleştirmenin ızgara üzerinde gerçekten doğru olduğunu doğrula
    # (i=0 hariç): y[i] + y[mirror(i)] tam olarak sıfır olmalı.
    assert np.max(np.abs(y[1:] + y[mirror_index][1:])) < 1e-9

    asymmetry = np.abs(profile - profile[mirror_index])
    max_asymmetry = float(np.max(asymmetry) / np.max(profile))

    # Fiziksel olarak TAM SIFIR olması gereken bir büyüklük; ölçülen gerçek
    # sapma ~1e-12 mertebesinde (çift hassasiyet FFT yuvarlama gürültüsü).
    # Eşik bunun ~1e6 katı üzerinde ama yine de "yuvarlama gürültüsü" ile
    # "gerçek bir asimetri hatası" arasında net ayrım yapacak kadar sıkı.
    assert max_asymmetry < 1e-6, (
        f"Desen y=0 etrafında simetrik değil: bağıl maksimum sapma "
        f"{max_asymmetry:.3e} — motorda ya da testte bir hata olabilir."
    )

    dy = float(y[1] - y[0])
    peak_y = float(y[np.argmax(profile)])
    assert abs(peak_y) < dy, (
        f"Merkezi maksimum y=0 civarında değil: y={peak_y} (dy={dy})"
    )


@pytest.mark.slow
def test_single_slit_control_does_not_show_genuine_interference(
    double_slit_profile: tuple[np.ndarray, np.ndarray],
    single_slit_profile: tuple[np.ndarray, np.ndarray],
):
    """En değerli kontrol: AYNI paket, AYNI duvar kalınlığı ve yarık
    genişliği, ama TEK yarık (y=0'da ortalanmış).

    ÖLÇÜLEN GERÇEK: görev tanımının ham kriteri (yükseklik >= tepe'nin %5'i,
    >=3 tepe) TEK yarıkta da geçiyor — y=+-15.23'te Fresnel kırınım saçakları
    beliriyor (bu ızgara/mesafede tek yarık genişliği dalga boyundan küçük
    olsa da Fraunhofer değil Fresnel rejiminde olunduğundan saçaklar
    oluşuyor). Bu, görev tanımının kendi uyarısını doğrulayan bir YANLIŞ
    POZİTİF: ham tepe sayısı tek başına girişimi kırınımdan ayırt edemiyor.

    Gerçek ayırt edici ölçüm: yan saçağın MERKEZE GÖRE BAĞIL yüksekliği.
    İki tutarlı kaynağın girişiminde ilk yan saçak zarfın (tek yarık
    kırınımının) o açıdaki değerine yakın kalır; burada zarf, yarık
    genişliği (2) dalga boyundan (pi≈3.14) küçük olduğundan yavaş
    değiştiğinden ölçülen oran merkeze göre ~%48.5. Tek yarıkta ikinci bir
    kaynak yok; kalan yapı sadece Fresnel saçağıdır ve ölçülen oran ~%16.5
    — ilkinin üçte biri kadar. Eşikler (0.35 / 0.25 / faktör 2), bu iki
    ölçülen değer (0.485 ve 0.165) arasında geniş bir marjla duruyor;
    hiçbiri testi geçirmek için ayarlanmadı.
    """
    _, profile_double = double_slit_profile
    _, profile_single = single_slit_profile

    ratio_double = _side_peak_ratio(profile_double)
    ratio_single = _side_peak_ratio(profile_single)

    assert ratio_double > 0.35, (
        f"Çift yarıkta yan saçak beklenenden zayıf: oran={ratio_double:.3f}"
    )
    assert ratio_single < 0.25, (
        f"Tek yarık kontrolünde saçak beklenenden güçlü — girişimle "
        f"karışıyor olabilir: oran={ratio_single:.3f}"
    )
    assert ratio_double > 2.0 * ratio_single, (
        f"Çift yarık saçağı ({ratio_double:.3f}), tek yarık kırınım "
        f"saçağından ({ratio_single:.3f}) yeterince güçlü ayrışmıyor — "
        "bu desenin gerçek girişimden mi yoksa kırınımdan mı geldiği "
        "belirsiz kalır."
    )


@pytest.mark.slow
def test_fringe_spacing_is_order_of_magnitude_consistent_with_far_field(
    double_slit_profile: tuple[np.ndarray, np.ndarray],
):
    """Saçak aralığını sin(theta) = m*lambda/d uzak-alan (Fraunhofer)
    tahminiyle karşılaştırır — lambda = 2*pi/k0, d = yarık ayrımı.

    DÜRÜST UYARI: bu geometri temiz uzak-alan rejiminde DEĞİL. Fresnel
    sayısı d^2/(lambda*L) = 10^2/(pi*40) ~ 0.80 — 1 mertebesinde, <<1
    değil. Bu yüzden yakın-alan (Fresnel) düzeltmeleri büyük ve ölçülen
    aralığın paraksiyel tahminle (lambda*L/d ~ 12.57) birkaç yüzde
    içinde örtüşmesi BEKLENMEZ: ölçülen ilk-saçak aralığı ~9.77, tahminden
    ~%22 az. Bu, rejimin dürüst bir sonucu, kod hatası değil — test bu
    yüzden yalnızca kaba mertebe (faktör 2) kontrolü yapıyor, tam örtüşmeyi
    ZORLAMIYOR.
    """
    y, profile = double_slit_profile
    peaks, _ = find_peaks(profile, height=_PEAK_HEIGHT_FRACTION * np.max(profile))
    peak_ys = np.sort(y[peaks])
    assert len(peak_ys) >= 2, "Aralık ölçmek için en az iki tepe gerekli"

    measured_spacing = float(np.mean(np.diff(peak_ys)))

    k0 = 2.0
    wavelength = 2.0 * np.pi / k0
    screen_distance = _SCREEN_X - _WALL_X
    paraxial_spacing = wavelength * screen_distance / _SLIT_SEPARATION

    fresnel_number = _SLIT_SEPARATION**2 / (wavelength * screen_distance)
    assert fresnel_number > 0.1, (
        "Bu test yakın-alan rejimini varsayıyor; parametreler değiştiyse "
        "yukarıdaki docstring ve rapor güncellenmeli."
    )

    ratio = measured_spacing / paraxial_spacing
    assert 0.5 < ratio < 2.0, (
        f"Ölçülen saçak aralığı ({measured_spacing:.3f}) paraksiyel "
        f"tahminden ({paraxial_spacing:.3f}) faktör 2'den fazla sapıyor — "
        "bu artık 'yakın-alan düzeltmesi' ile açıklanamaz, gerçek bir "
        "hata olabilir."
    )
