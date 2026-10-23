"""Strang bölmeli split-operator zaman evrimi.

    psi(t+dt) = e^{-i V dt/2} F^-1 [ e^{-i k^2 dt/(2m)} F [ e^{-i V dt/2} psi(t) ] ]

Yöntem ünitertir: gerçek zamanda norm makine hassasiyetinde korunur. Bu bir test
hedefi değil, yöntemin garantisidir — sapma varsa dt çok büyüktür ya da kod
hatalıdır. evolve() bu yüzden sapmayı istisna olarak raporlar.

İmajiner zaman: dt -> -i*dt yerine koyunca e^{-H tau} elde edilir; her adımdan
sonra yeniden normalize edilerek durum taban duruma yakınsar.

Soğurucu maske (B5-U, 14.10.2026): maske normu BİLEREK düşürür, yani maskeli
koşuda |N - 1| üniterlik hakkında hiçbir şey söylemez. Ölçülebilen şey
maskeden ÖNCEKİ üniter adımdır; `evolve_olcumlu` onu her adımda ölçer
(`UniterlikOlcumu`), hükmü `tdse/factory.py::provenance_for` kurar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qlab.engine.tdse.grid import Grid1D, Grid2D
from qlab.errors import InvalidInputError, UnitarityViolationError


@dataclass(frozen=True)
class UniterlikOlcumu:
    """Maskeli gerçek zaman koşusunda maskeden ÖNCEKİ üniter adımın normu (B5-U).

    İki ölçü de maskeyi DIŞARIDA bırakır, ikisi de adım başına ölçülür:

    - `en_buyuk_adim_sapmasi` = max_k |‖Uψ_k‖² / ‖ψ_k‖² − 1|: adıma giren
      normla oranlanır (göreli), yani soğurmadan sonra kalan küçük normdaki
      bir kusuru da ADIM BAŞINA aynı duyarlılıkla görür (ölçek değişmezi:
      ψ0 × 1e-4'te de aynı sayı). BİRİKİMİNİ görmez — bkz. aşağıdaki kör pencere.
    - `en_buyuk_butce_sapmasi` = max_k |N_k + A_k − N_0| / N_0; A_k o ana
      kadar maskenin soğurduğu norm. Maske ≡ 1'de A = 0 ve bu TAM OLARAK
      maskesiz yolun `|N − 1|` ölçütüdür (birikimli: adım başına eşiğin
      altında kalan ama biriken bir sızıntıyı yakalar). "Maskesiz ölçütün
      aynısı" YALNIZ bu ölçü için doğrudur.

    KÖR PENCERE (beyan, V2 23.10.2026; ölçüldü `tests/tdse/test_b5u_maskeli_hukum.py`
    senaryosu, 800 adım, paket N 1 -> 6.9e-5): yalnız soğurmadan SONRA (N < 1e-3)
    etkin, adım başına eşiğin ALTINDA bir kusur iki ölçüden de geçer. Adım başı
    7.6e-9'da adım 7.6e-9, bütçe 1.9e-10, hüküm ✓; kalan paketin normu kusursuza
    göre GÖRELİ 6.9e-7 kayar, MUTLAK olasılık biriminde 4.8e-11 (maskesiz yolun
    mutlak 1e-8 ölçütünün ~200 kat altı). Hüküm mutlak olasılıkta maskesiz yolla
    aynı duyarlıkta, kalan küçük paketin göreli birikiminde değil.

    `sogurulan_norm` = A_n / N_0: maskenin soğurduğu kesir. SINIR KOŞULUDUR,
    fizik değil, ve üniterlik hükmüne GİRMEZ.

    Sıfır norma inmiş bir adımda (ψ tümüyle sıfır) oran tanımsızdır; orada
    ‖Uψ‖² = ‖ψ‖² = 0 tam eşitliği geçerli olduğu için sapma 0 yazılır. NaN
    her iki ölçüye de yayılır (koşan maksimum NaN'ı tutar, `np.max` gibi) ve
    hükmü ✗ yapar.

    TÜR KAPISI (V3, 23.10.2026): doğrudan kuruluş da (Python yolu) ölçünün
    tanımına uymalı. `adim_sayisi` bool olmayan tam sayı ≥ 1 (`800.0` DEĞİL);
    üç sayı bool/dize/karmaşık olmayan gerçel ve ≥ 0 ya da NaN. +inf kabul:
    taşan bir U'nun DÜRÜST ölçümüdür (‖Uψ‖² = inf) ve hükmü ✗ yapar. Negatif
    sapma tanım gereği olamaz (mutlak değer); negatif soğurulan ise maskenin
    normu ARTIRDIĞI demektir (maske [0, 1] dışında; fabrikanın maskesi
    üretmez). Ölçüldü (kapı yokken): `UniterlikOlcumu(800, -5, -1, 0.5)` ✓
    basıyordu, dize sapma `provenance_for`da ham `TypeError` veriyordu.
    """

    adim_sayisi: int
    en_buyuk_adim_sapmasi: float
    en_buyuk_butce_sapmasi: float
    sogurulan_norm: float

    def __post_init__(self) -> None:
        n = self.adim_sayisi
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise InvalidInputError(
                f"UniterlikOlcumu.adim_sayisi pozitif bir tam sayı olmalı (bool ve "
                f"float DEĞİL), verilen: {n!r}"
            )
        for ad in ("en_buyuk_adim_sapmasi", "en_buyuk_butce_sapmasi", "sogurulan_norm"):
            deger = getattr(self, ad)
            if isinstance(deger, (bool, np.bool_)) or not isinstance(
                deger, (int, float, np.integer, np.floating)
            ):
                raise InvalidInputError(
                    f"UniterlikOlcumu.{ad} gerçel bir sayı olmalı (bool, dize ve "
                    f"karmaşık DEĞİL), verilen: {deger!r}"
                )
            if not (deger >= 0.0 or np.isnan(deger)):
                raise InvalidInputError(
                    f"UniterlikOlcumu.{ad} negatif olamaz (sapma mutlak değerdir; "
                    f"negatif soğurulan maskenin normu artırdığı demektir), verilen: {deger!r}"
                )


@dataclass
class SplitOperatorPropagator:
    grid: Grid1D | Grid2D
    potential: np.ndarray
    dt: float
    mass: float = 1.0
    imaginary_time: bool = False
    mask: np.ndarray | None = None

    _half_v: np.ndarray = field(init=False, repr=False)
    _kinetic: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise InvalidInputError(f"dt pozitif olmalı, verilen: {self.dt}")
        if self.mass <= 0:
            raise InvalidInputError(f"mass pozitif olmalı, verilen: {self.mass}")

        expected = self._expected_shape()
        if self.potential.shape != expected:
            raise InvalidInputError(
                f"Potansiyel şekli ızgarayla uyuşmuyor: beklenen {expected}, "
                f"verilen {self.potential.shape}"
            )
        if self.mask is not None and self.mask.shape != expected:
            raise InvalidInputError(
                f"Maske şekli ızgarayla uyuşmuyor: beklenen {expected}, "
                f"verilen {self.mask.shape}"
            )

        step = -1j * self.dt if self.imaginary_time else self.dt
        self._half_v = np.exp(-0.5j * self.potential * step)
        self._kinetic = np.exp(-0.5j * self.grid.k2 * step / self.mass)

    @property
    def is_two_dimensional(self) -> bool:
        return isinstance(self.grid, Grid2D)

    @property
    def volume_element(self) -> float:
        return self.grid.cell_area if self.is_two_dimensional else self.grid.dx

    def _expected_shape(self) -> tuple[int, ...]:
        if isinstance(self.grid, Grid2D):
            return (self.grid.nx, self.grid.ny)
        return (self.grid.n,)

    def _forward(self, psi: np.ndarray) -> np.ndarray:
        return np.fft.fft2(psi) if self.is_two_dimensional else np.fft.fft(psi)

    def _inverse(self, psi: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(psi) if self.is_two_dimensional else np.fft.ifft(psi)

    def step(self, psi: np.ndarray) -> np.ndarray:
        """Tek bir Strang adımı uygular ve yeni psi'yi döndürür.

        Şekil doğrulaması yanlış boyutlu ya da yanlış eksen sayılı girdiyi
        yakalar. Ama bilinen bir sınırı var: kare bir ızgarada (nx == ny)
        transpoze edilmiş psi.T, orijinal psi ile birebir aynı .shape
        tuple'ına sahiptir — NumPy dizisi hangi eksenin x, hangisinin y
        olduğuna dair bir bilgi taşımaz, bu yüzden bu doğrulama transpozisyonu
        hiçbir zaman yakalayamaz. Bu çözülmüş bir sorun değil, çalışma
        zamanında tespit edilemeyen bilinen bir kısıt; kare ızgaralarda
        eksenlerin doğru sırayla verilmesi çağıranın sorumluluğunda kalır.
        """
        self._sekil_kapisi(psi)
        psi = self._uniter_kisim(psi)

        if self.mask is not None:
            psi = psi * self.mask

        if self.imaginary_time:
            psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * self.volume_element)

        return psi

    def _sekil_kapisi(self, psi: np.ndarray) -> None:
        expected = self._expected_shape()
        if psi.shape != expected:
            raise InvalidInputError(
                f"psi şekli ızgarayla uyuşmuyor: beklenen {expected}, "
                f"verilen {psi.shape}"
            )

    def _uniter_kisim(self, psi: np.ndarray) -> np.ndarray:
        """Strang adımının üniter kısmı: e^{-iV dt/2} F^-1 e^{-iT dt} F e^{-iV dt/2}.

        `step` ve `_maskeli_adim` bunu AYNI çağrıyla kullanır: B5-U'da ölçülen
        U, koşulan U'nun kendisidir (ayrı bir kopya olsaydı biri düzeltilip
        öteki unutulunca ölçüm başka bir operatörü yargılardı).
        """
        psi = self._half_v * psi
        psi = self._inverse(self._kinetic * self._forward(psi))
        return self._half_v * psi

    def _maskeli_adim(self, psi: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Maskeli gerçek zaman adımı + maskeden ÖNCEKİ ve SONRAKİ norm (B5-U).

        Döndürülen ψ `step(psi)` ile BİT BİT aynıdır (aynı işlemler, aynı
        sıra); ek maliyet adım başına iki norm indirgemesidir. Norm `vdot`
        ile alınır (`sum(abs(ψ)**2)`nin ~4 katı hızlı, aynı yuvarlama
        mertebesi). Ek süre (4096 nokta × 64000 adım, `evolve_olcumlu` ↔
        `evolve`) %5–19, makine gürültüsü içinde: 14.10'da +5.8 %; 23.10.2026'da
        üç ayrı ölçüm, en iyiler +18.7 / +13.2 / +15.9 %, ortancalar +4.9 / +5.8 /
        +5.5 %, sıra değişen çiftler −1.2 … +17.8 %; adım başına 4–10 µs, `evolve`
        adımı en iyi ~62 µs (V4: eski "~3 µs" tek bir ölçümdü, tekrarlanamadı).
        """
        if self.mask is None or self.imaginary_time:
            raise InvalidInputError(
                "maske öncesi norm yalnız soğurucu maskeli GERÇEK zaman adımında "
                "ölçülür (maskesizde |N - 1| zaten ölçüttür, imajiner zaman normu "
                "her adımda yeniden kurar)"
            )
        self._sekil_kapisi(psi)
        psi = self._uniter_kisim(psi)
        once = float(np.vdot(psi, psi).real) * self.volume_element
        psi = psi * self.mask
        sonra = float(np.vdot(psi, psi).real) * self.volume_element
        return psi, once, sonra


def evolve(
    propagator: SplitOperatorPropagator,
    psi0: np.ndarray,
    n_steps: int,
    save_every: int = 1,
    norm_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Zaman evrimi; (kareler, zamanlar) döndürür. t=0 karesi her zaman dahil.

    Norm kontrolü yalnızca gerçek zamanda ve maske yokken yapılır; imajiner
    zaman ve soğurucu maske normu bilinçli olarak değiştirir. Maskeli koşunun
    üniterlik ölçümü (maske öncesi üniter adım normu) `evolve_olcumlu`dadır;
    bu işlev onu ÖLÇMEZ ve ek maliyet ödemez.
    """
    frames, times, _ = _evolve(propagator, psi0, n_steps, save_every, norm_tol, olc=False)
    return frames, times


def evolve_olcumlu(
    propagator: SplitOperatorPropagator,
    psi0: np.ndarray,
    n_steps: int,
    save_every: int = 1,
    norm_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, UniterlikOlcumu | None]:
    """`evolve` + maskeli gerçek zaman koşusunda üniterlik ölçümü (B5-U).

    (kareler, zamanlar, ölçüm) döndürür; kareler ve zamanlar `evolve`unkiyle
    BİT BİT aynıdır. Ölçüm yalnız soğurucu maskeli GERÇEK zaman koşusunda
    bir `UniterlikOlcumu`dur. Maskesiz koşuda None'dır: orada üniterliği
    adım başına |N − 1| kontrolü zaten ölçer ve ihlali
    `UnitarityViolationError` ile bildirir. İmajiner zamanda da None'dır.

    Maskeli koşuda ölçüm İSTİSNA FIRLATMAZ (`norm_tol` orada kullanılmaz):
    hüküm `tdse/factory.py::provenance_for`da, maskesiz yolun eşiğiyle (✓/✗)
    kurulur. Bu asimetri bilinçlidir: soğurucu kenarlı bir koşuda kareler
    incelenmeye değer kalır ve panel sapmanın büyüklüğünü basar.
    """
    return _evolve(propagator, psi0, n_steps, save_every, norm_tol, olc=True)


def _evolve(
    propagator: SplitOperatorPropagator,
    psi0: np.ndarray,
    n_steps: int,
    save_every: int,
    norm_tol: float,
    *,
    olc: bool,
) -> tuple[np.ndarray, np.ndarray, UniterlikOlcumu | None]:
    if n_steps < 1:
        raise InvalidInputError(f"n_steps en az 1 olmalı, verilen: {n_steps}")
    if save_every < 1:
        raise InvalidInputError(f"save_every en az 1 olmalı, verilen: {save_every}")

    check_norm = not propagator.imaginary_time and propagator.mask is None
    maskeli_olcum = olc and not propagator.imaginary_time and propagator.mask is not None
    volume = propagator.volume_element

    psi = psi0.copy()
    frames = [psi.copy()]
    times = [0.0]

    if maskeli_olcum:
        n0 = float(np.vdot(psi, psi).real) * volume  # `_maskeli_adim` ile aynı yol
        if not (np.isfinite(n0) and n0 > 0.0):
            raise InvalidInputError(f"psi0 normu pozitif ve sonlu olmalı, verilen: {n0}")
        # V5 (23.10.2026): koşan maksimum, O(1) bellek. Eskiden iki n_steps uzunluklu
        # float64 dizi yalnız `np.max` için tutuluyordu (16 B/adım; ölçüldü: 64000
        # adımda tracemalloc tepesi 1.047 MB, `evolve` 0.022 MB; 1e8 adımda 1.6 GB).
        en_buyuk_adim = 0.0
        en_buyuk_butce = 0.0
        onceki = n0
        sogurulan = 0.0

    for index in range(1, n_steps + 1):
        if maskeli_olcum:
            psi, uniter, sonra = propagator._maskeli_adim(psi)
            en_buyuk_adim = _nan_yayan_max(
                en_buyuk_adim, abs(uniter / onceki - 1.0) if onceki > 0.0 else 0.0
            )
            sogurulan += uniter - sonra
            en_buyuk_butce = _nan_yayan_max(en_buyuk_butce, abs(sonra + sogurulan - n0) / n0)
            onceki = sonra
        else:
            psi = propagator.step(psi)

        if check_norm:
            deviation = abs(float(np.sum(np.abs(psi) ** 2) * volume) - 1.0)
            if deviation > norm_tol:
                raise UnitarityViolationError(
                    step=index, norm_deviation=deviation, dt=propagator.dt
                )

        if index % save_every == 0:
            frames.append(psi.copy())
            times.append(index * propagator.dt)

    olcum = None
    if maskeli_olcum:
        olcum = UniterlikOlcumu(
            adim_sayisi=n_steps,
            en_buyuk_adim_sapmasi=en_buyuk_adim,
            en_buyuk_butce_sapmasi=en_buyuk_butce,
            sogurulan_norm=sogurulan / n0,
        )
    return np.array(frames), np.array(times), olcum


def _nan_yayan_max(simdiki: float, yeni: float) -> float:
    """Koşan maksimum; NaN bir kez girince KALIR (`np.max` gibi, `max()` gibi değil).

    `max(nan, x)` ve `max(x, nan)` argüman sırasına göre NaN'ı yutar ya da tutar;
    burada iki yönde de kalır: yeni NaN ise `yeni != yeni` onu alır, şimdiki NaN
    ise `yeni > nan` False olduğu için şimdiki kalır. Sonlu değerlerde sonuç
    `np.max` ile aynıdır (karşılaştırma tamdır).
    """
    return yeni if (yeni > simdiki or yeni != yeni) else simdiki
