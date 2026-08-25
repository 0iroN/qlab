"""Maxwell FDTD taramasi cizim YUKU -- goruntu uretmez.

Deponun `viz/` katmani hep yuk uretir (`frames.py`, `md_frames.py`,
`density_frames.py`, `band_frames.py`, `spin_frames.py`); `.png` yazmak yeni
bagimlilik ve yeni katman demektir. Ayni uyari `band_frames.py`de yazili: bu
bir DISIPLINDIR, SOZLESME DEGIL -- `viz` katmaninin import-linter tarafindan
korunan bir GUI yasagi YOKTUR.

BU MODUL NEDEN VAR -- KATMAN ZORUNLULUGU. Kip spektrumu ve yakinsama
merdiveni `observable/em_obs.py`dedir ve `engine/em/factory.py` onlari IMPORT
EDEMEZ (sozlesme: run -> viz -> observable -> engine -> scenario). Yani
kayittan orani ve hukmu cikaran yer buradan baskasi olamaz. Fabrika fizigi
kosar, bu modul olcer, fabrika olculeni BEYANA cevirir. Modul 4'un
`spin_frames.py`siyle birebir ayni gerekce.

CIPA BURADA KURULMAZ. Bu yuk `j(1,1)/j(0,1)` diye bir sey BILMEZ: okudugu sey
IKINCI EN DUSUK TEPENIN EN DUSUGE ORANIDIR ve bu tanim hicbir Bessel koku
icermez. Kokler yalniz `tests/em/test_bessel.py`de gecer ve karsilastirma
orada yapilir; boylece `mode_spectrum`un "adsiz tepeler, teori YOK" beyani
sevk edilen boru hattinda da AYAKTA kalir. Sevk edilen pencerede iki tanimin
AYNI sayiyi verdigi olculdu ve `tests/em/test_em_senaryo.py`de kilitlidir.

OLCULEN PENCERE (26.08.2026, bu depoda; `scenarios/em_tarama.yaml`):
R = 5/10/20/40, S = 0.5, adim = 420*R, blackman, pad 8, taban 0.01, 8
dedektor. Basamak basina tepe sayisi 28 / 21 / 17 / 15; oran
1.578632448506 / 1.589868261123 / 1.592538328891 / 1.593146492961; merdiven
OTURDU(2), p_inf = 2.195532 +- 0.061188. FDTD 0.95 s, analiz 0.54 s.

GURULTU EK BIR FDTD KOSUMU ISTEMEZ. `noise` icin gereken sey, ayni kaydin
MESRU pencerelerdeki yayilimidir; `pad`, tepe tabani ve kayit kirpma UCU DE
ISLEM-SONRASIDIR (bkz. `observable/em_obs.py::convergence_ladder`
dokstringi). Bu yuzden 27 pencere tek kayittan uretilir ve maliyeti bir
FFT'dir, bir kosum degil.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from qlab.engine.em.factory import EMScanResult
from qlab.engine.em.grid import FieldRecord
from qlab.errors import InvalidInputError
from qlab.observable.em_obs import (
    ConvergenceLadder,
    ModeSpectrum,
    convergence_ladder,
    mode_spectrum,
)

#: `noise`u uretirken taranan MESRU pencereler. Ucu de ISLEM-SONRASI
#: dugmedir: `pad` sifir dolgu katsayisi, `taban` tepe esigi, `kesir` ise
#: kaydin kullanilan bas kismi. Hicbiri yeni bir FDTD adimi istemez.
#: Kombinasyon sayisi 3*3*3 = 27 ve sevk edilen pencerede 27'sinin 27'si de
#: oran uretiyor (olculdu, dort basamakta da).
#:
#: `pad` senaryodan gelir ve mesru pencereler ONA GORE kurulur (sabit bir
#: liste degil): aynisi, iki kati, yarisi. Sabit yazilsaydi `pad = 4` ile
#: kosan bir kullanicinin gurultusu, kosmadigi bir pencereden okunurdu.
_PAD_KURALLARI = ("ayni", "iki_kat", "yari")
_TABANLAR = (0.005, 0.01, 0.02)
_KESIRLER = (1.0, 0.93, 0.86)


def _pad_degerleri(pad: int) -> tuple[int, ...]:
    """Senaryonun `pad`ine gore uc mesru pad degeri. Hepsi >= 1."""
    return tuple(
        max(1, pad if kural == "ayni" else pad * 2 if kural == "iki_kat" else pad // 2)
        for kural in _PAD_KURALLARI
    )


@dataclass(frozen=True, eq=False)
class EMPayload:
    """Bir yaricap merdiveninin cizim yuku. HEPSI BOYUTSUZ.

    `radii` HUCRE cinsindendir, metre DEGIL (bkz. birim secimi beyani:
    c = eps_0 = mu_0 = 1). `spacings` = 1/R ve merdivenin izgara araligidir.

    `ratios` her basamakta IKINCI EN DUSUK TEPENIN EN DUSUGE ORANIDIR --
    tekil bir frekans DEGIL. Oran secilmesinin gerekcesi olculdu ve
    `boundary.CircularPEC` dokstringinde yazili: merdivenlenmis sinirin etkin
    yaricap sapmasi mutlak frekanslarda kalir, oranda sadelesir.

    `peaks` basamak basina bulunan tepe sayisidir ve bir PENCERE
    OLCUSUDUR: kaba basamakta daha COK tepe bulunur (merdivenlenmis daire
    C4v'dir ve dejenere ikilileri yarar), yani buyuk sayi daha iyi bir
    kosum DEMEK DEGILDIR.

    `field` en ince basamagin SON Ez anlik goruntusudur, `field_conductor`
    ise ayni izgaradaki PEC maskesi. Ikisi BIRLIKTE tasinir: maskesiz bir
    alan haritasi, boslugun disindaki ozdes sifirlarin "alan yok" mu yoksa
    "iletken" mi oldugunu soyleyemezdi.

    `eq=False`: alanlar ndarray tasir; uretilen `__eq__` dizi dondurur ve
    `__hash__` `TypeError` verir (depo bu tuzagi `spin/lattice.py` ve
    `engine/md/state.py`de zaten odedi).
    """

    radii: np.ndarray
    spacings: np.ndarray
    ratios: np.ndarray
    noise: np.ndarray
    peaks: np.ndarray
    ladder: ConvergenceLadder
    spectra: tuple[ModeSpectrum, ...]
    windows: tuple
    field: np.ndarray
    field_conductor: np.ndarray


def _kirp(kayit: FieldRecord, kesir: float) -> FieldRecord:
    """Kaydin ilk `kesir` kadari -- ISLEM SONRASI bir pencere dugmesi.

    `window.steps` de kirpilir, yoksa `mode_spectrum` frekans eksenini
    kaydin GERCEK uzunluguyla degil beyan edilen uzunlukla kurardi ve
    gurultu tahmini sessizce yanlis cikardi.
    """
    n = int(kayit.probes.shape[1] * kesir)
    return FieldRecord(
        probes=kayit.probes[:, :n],
        probe_indices=kayit.probe_indices,
        window=replace(kayit.window, steps=n),
        final=kayit.final,
    )


def _oran(spektrum: ModeSpectrum) -> float | None:
    """Ikinci en dusuk tepenin en dusuge orani -- TEORI YOK.

    Tepe listesi ikiden kisaysa `None` doner: bir oran icin iki tepe gerekir
    ve olmayan bir tepenin yerine sifir ya da bir yazmak, olculmemis bir
    sayiyi olculmus gibi gostermek olurdu.
    """
    oranlar = spektrum.ratios()
    if len(oranlar) < 2:
        return None
    return float(oranlar[1])


def _deger_ve_gurultu(
    kayit: FieldRecord, *, pad: int, window: str, peak_floor: float
) -> tuple[float, float, int]:
    """Sevk penceresindeki oran + MESRU pencerelerin BAGIL yayilimi.

    Ucuncu deger, oran ureten mesru pencere sayisidir ve DISARI TASINIR:
    yayilimi kac ornekten okudugunu sormadan bir std yazmak, belirsizligin
    belirsizligini gizlemek olurdu.
    """
    sevk = _oran(mode_spectrum(kayit, pad=pad, window=window, peak_floor=peak_floor))
    if sevk is None:
        raise InvalidInputError(
            f"sevk penceresinde iki tepe bulunamadi (izgara {kayit.window.izgara}, "
            f"adim {kayit.window.steps}, pad {pad}, taban {peak_floor}); "
            "oran uretilemez ve merdiven bu basamakla kurulamaz"
        )
    degerler: list[float] = []
    for p in _pad_degerleri(pad):
        for taban in _TABANLAR:
            for kesir in _KESIRLER:
                v = _oran(
                    mode_spectrum(
                        _kirp(kayit, kesir),
                        pad=p,
                        window=window,
                        peak_floor=taban,
                    )
                )
                if v is not None:
                    degerler.append(v)
    a = np.asarray(degerler, dtype=float)
    if a.size < 2:
        raise InvalidInputError(
            f"gurultu icin en az iki mesru pencere gerekir, {a.size} bulundu"
        )
    return sevk, float(a.std(ddof=1) / abs(a.mean())), int(a.size)


def build_em_payload(result: EMScanResult) -> EMPayload:
    """Ham kayitlardan oran merdivenini ve hukmu cikarir.

    HUKUM BURADA VERILMEZ, `convergence_ladder`da verilir; bu modul yalniz
    ona ne verecegini secer. Sectigi sey de bir tercih degil bir OLCUM
    sonucudur: `noise` ZORUNLU girdidir ve `noise = 0` ile yanlis pozitif
    %16.4'e cikiyor (426 gercek merdivende olculdu), gurultu yayilimiyla
    %0.0.
    """
    senaryo = result.scenario
    oranlar: list[float] = []
    gurultuler: list[float] = []
    tepeler: list[int] = []
    spektrumlar: list[ModeSpectrum] = []
    for kayit in result.records:
        spektrum = mode_spectrum(
            kayit,
            pad=senaryo.pad,
            window=senaryo.window_fn,
            peak_floor=senaryo.peak_floor,
        )
        deger, gurultu, _ = _deger_ve_gurultu(
            kayit,
            pad=senaryo.pad,
            window=senaryo.window_fn,
            peak_floor=senaryo.peak_floor,
        )
        spektrumlar.append(spektrum)
        oranlar.append(deger)
        gurultuler.append(gurultu)
        tepeler.append(len(spektrum.frequencies))

    araliklar = [1.0 / R for R in senaryo.radii]
    merdiven = convergence_ladder(
        oranlar, araliklar, noise=gurultuler, ratio=senaryo.ratio
    )
    son = result.records[-1]
    return EMPayload(
        radii=np.asarray(senaryo.radii, dtype=float),
        spacings=np.asarray(araliklar, dtype=float),
        ratios=np.asarray(oranlar, dtype=float),
        noise=np.asarray(gurultuler, dtype=float),
        peaks=np.asarray(tepeler, dtype=int),
        ladder=merdiven,
        spectra=tuple(spektrumlar),
        windows=tuple(k.window for k in result.records),
        field=np.asarray(son.final.e, dtype=float),
        field_conductor=_iletken_maskesi(son, senaryo.radii[-1]),
    )


def _iletken_maskesi(kayit: FieldRecord, radius: float) -> np.ndarray:
    """Son basamagin PEC maskesi -- ALANLA BIRLIKTE tasinmasinin gerekcesi.

    Alan haritasindaki sifirlar IKI AYRI sey olabilir: iletken bir dugum, ya
    da gercekten sifir olan bir alan. Tek basina bir `field` dizisi bu ikisini
    ayirt edemez, bu yuzden maske alanla BIRLIKTE tasinir.

    YARICAP SEKILDEN TURETILMEZ, SENARYODAN GELIR. Izgara boyu
    `2*ceil(R + margin) + 1`dir, yani sekilden okunacak sey R degil R + margin
    olurdu ve maske sessizce boslugun DISINI isaretlerdi. Sinir nesnesinin
    kendisi kayitta tasinmadigi icin dogru kaynak senaryodur.
    """
    from qlab.engine.em.boundary import CircularPEC

    return CircularPEC(radius=radius).pec_mask_2d(kayit.final.shape)
