"""Yee çekirdeği: ORTAMI BİLMEZ, ve TEK GÜNCELLEME YOLU vardır.

Bu dosya SINIF olarak TRANSKRİPSİYON DENETİMİdir — hiçbir fizik iddiası
taşımaz. Ama taşıyıcıdır: Modül 5'in iki fizik çıpasının (Bessel oranları ve
Rayleigh üssü) "aynı çözücü, iki bağımsız hüküm" iddiası, çekirdeğin ortamı
BİLMEMESİNE dayanır. Çekirdek geometriye göre dallanabilseydi, iki çıpa iki
ayrı kod yolundan geçer ve iddia düşerdi. (Modül 4'te `metropolis.py`nin
modeli bilmemesiyle aynı gerekçe — ve orada da testle kilitlenmişti.)

İDDİA NASIL KİLİTLENİYOR — üç bağımsız yol:

1. KAYIT EDEN VEKİL (en doğrudan): çekirdeğe, erişilen her öznitelik adını
   kaydeden bir sarmalayıcı verilir. Koşum bitince DOKUNULAN ADLARIN TAM
   KÜMESİ sınanır. `interface`, `ey_kural`, `eps_min`, `half_space` — hiçbiri
   kümede DEĞİLDİR. Bu, "bilmiyor" iddiasının sayımıdır, yorumu değil.
2. AYNI DİZİ, FARKLI KURULUŞ: `Medium.half_space(...)` ile elle kurulmuş
   özdeş bir `Medium` BİT BİT aynı sonucu verir; kuruluş yolu görünmez.
3. AYNI ÇEKİRDEK, İKİ AYRI GEOMETRİ: düzgün ortam, tek arayüz ve çok
   katmanlı yığın aynı `_adim`den geçer — dallanma yok.

PENCERE (bu dosyadaki her koşum için): dx = 1, `c = eps0 = mu0 = 1`,
S = 0.5 (aksi belirtilmedikçe), PEC duvarlı ızgara, tohum YOK (FDTD
deterministtir; rastgelelik yalnız Courant eşiği ölçümünde vardır ve o
`test_sema_cipalari.py`dedir).
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.em.boundary import PEC, PML, YUZLER, LossyLayer
from qlab.engine.em.grid import (
    FieldRecord,
    FieldState,
    FieldState2D,
    Window,
    courant_limit,
)
from qlab.engine.em.medium import EY_KURALLARI, Medium
from qlab.engine.em.source import gauss_cos, ricker
from qlab.engine.em.yee import _adim, run_1d, yee_coefficients, yee_step_1d
from qlab.errors import InvalidInputError


class _Kaydeden:
    """Erişilen her öznitelik adını kaydeden şeffaf vekil."""

    def __init__(self, hedef):
        self._hedef = hedef
        self.dokunulan: set[str] = set()

    def __getattr__(self, ad):
        # `_hedef` ve `dokunulan` __dict__'te olduğu için buraya düşmezler.
        self.dokunulan.add(ad)
        return getattr(self._hedef, ad)


def _ortam_yigin(cells: int) -> Medium:
    """Çok katmanlı yığın — çekirdeğin ASLA bilmediği bir geometri."""
    eps = np.ones(cells)
    for i, deger in enumerate((2.25, 4.0, 1.5, 9.0)):
        eps[cells // 6 * (i + 1) : cells // 6 * (i + 2)] = deger
    return Medium(eps, np.ones(cells))


def _kosum(ortam, sinir, *, steps=200, courant=0.5, cells=None):
    n = cells if cells is not None else ortam.cells
    return run_1d(
        ortam, courant=courant, boundary=sinir, steps=steps,
        source=lambda t: gauss_cos(t, wavelength=20.0, sgf=1.5, t0=60.0),
        source_index=n // 3, probes=(n // 2, 2 * n // 3),
        wavelength=20.0, sgf=1.5, window_fn="dikdortgen",
    )


# ---------------------------------------------------------------------------
# 1. ÇEKİRDEK ORTAMI BİLMİYOR
# ---------------------------------------------------------------------------
def test_CEKIRDEK_ORTAMDAN_YALNIZ_UC_AD_GORUYOR_ve_GEOMETRI_ONLARDA_YOK():
    """SINIF: TRANSKRİPSİYON DENETİMİ — ama Modül 5'in TAŞIYICI kilidi.

    Çekirdeğin ortamdan gördüğü şeylerin TAM listesi sayılıyor. Geometriye
    dair her şey (`interface`, `ey_kural`, `eps_min`) listenin DIŞINDA.

    Bu, "yee.py ortamı bilmez" cümlesinin sayısal karşılığıdır: bir gün
    biri çekirdeğe `if medium.ey_kural == ...` ya da `medium.interface`
    eklerse, o satır bu kümeyi büyütür ve test DÜŞER.
    """
    ortam = _Kaydeden(Medium.half_space(240, interface=120, eps1=1.0, eps2=4.0))
    sinir = _Kaydeden(PML(thickness=15))
    _kosum(ortam, sinir, cells=240)

    assert ortam.dokunulan == {"cells", "eps_r", "mu_yari"}
    assert sinir.dokunulan == {"sigma_1d", "thickness"}
    # Geometri ve modelleme seçimleri çekirdeğe HİÇ girmiyor:
    for yasak in ("ey_kural", "eps_min", "eps_yari", "half_space", "interface"):
        assert yasak not in ortam.dokunulan


def test_AYNI_DIZI_FARKLI_KURULUS_BIT_BIT_AYNI_SONUC():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    `half_space` bir KOLAYLIK kurucusudur; çekirdek onun varlığından habersiz
    olmalıdır. Elle kurulmuş özdeş `Medium` bit bit aynı kaydı vermelidir —
    ve `ey_kural` FARKLI olsa bile, çünkü 1B TM güncellemesi `eps_yari`yi
    hiç çağırmaz.
    """
    a = Medium.half_space(240, interface=120, eps1=1.0, eps2=4.0)
    eps = np.ones(240)
    eps[120:] = 4.0
    b = Medium(eps, np.ones(240), ey_kural="sag")

    ra = _kosum(a, PML(thickness=15))
    rb = _kosum(b, PML(thickness=15))
    assert np.array_equal(ra.probes, rb.probes)
    assert a.ey_kural != b.ey_kural


def test_UC_AYRI_GEOMETRI_AYNI_KOD_YOLUNDAN_GECIYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    Düzgün ortam, tek arayüz ve dört katmanlı yığın — üçü de aynı `_adim`den
    geçer, üçü de sonlu kalır, ve üçü BİRBİRİNDEN FARKLI çıktı verir (yani
    çekirdek `eps_r`yi gerçekten kullanıyor, sadece görmezden gelmiyor).
    """
    kayitlar = [
        _kosum(Medium.vacuum(240), PML(thickness=15)).probes,
        _kosum(Medium.half_space(240, interface=120, eps2=4.0),
               PML(thickness=15)).probes,
        _kosum(_ortam_yigin(240), PML(thickness=15)).probes,
    ]
    for k in kayitlar:
        assert np.all(np.isfinite(k))
    for i in range(len(kayitlar)):
        for j in range(i + 1, len(kayitlar)):
            assert not np.array_equal(kayitlar[i], kayitlar[j])


def test_YEE_STEP_1D_ve_RUN_1D_TEK_GUNCELLEME_YOLU():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    İki genel API de `_adim`den geçmelidir. Ayrı iki güncelleme yazılsaydı,
    şema çıpaları hangisini sınadığını bilemezdi. Elle 200 adım ile
    `run_1d(steps=200)` BİT BİT aynı olmalı.
    """
    ortam, sinir = _ortam_yigin(240), PML(thickness=15)
    st = FieldState.zeros_1d(240)
    kayit = np.zeros((1, 200))
    co = yee_coefficients(ortam, courant=0.5, boundary=sinir)
    for n in range(200):
        kayit[0, n] = st.e[120]
        st = yee_step_1d(st, ortam, courant=0.5, boundary=sinir)
        st.e[80] += co.cb[80] * gauss_cos(
            n * 0.5, wavelength=20.0, sgf=1.5, t0=60.0
        )
    toplu = run_1d(
        ortam, courant=0.5, boundary=sinir, steps=200,
        source=lambda t: gauss_cos(t, wavelength=20.0, sgf=1.5, t0=60.0),
        source_index=80, probes=(120,), wavelength=20.0, sgf=1.5,
        window_fn="dikdortgen",
    )
    assert np.array_equal(kayit, toplu.probes)
    assert np.array_equal(st.e, toplu.final.e)
    assert np.array_equal(st.h, toplu.final.h)


def test_RUN_1D_CAGIRANIN_BASLANGIC_DURUMUNU_BOZMUYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    `run_1d` kaynak enjeksiyonunu YERİNDE yapar (adım başına bir tam dizi
    kopyasından kaçınmak için). Bu güvenlidir çünkü ilk yazma `_adim`
    SONRASIDIR; ama güvenli olduğu SÖYLENMEZ, KİLİTLENİR.
    """
    st = FieldState.zeros_1d(240)
    st.e[100] = 1.0
    kopya = st.e.copy()
    _kosum(Medium.vacuum(240), PML(thickness=15), steps=50)
    run_1d(
        Medium.vacuum(240), courant=0.5, boundary=PML(thickness=15), steps=50,
        source=lambda t: gauss_cos(t, wavelength=20.0, sgf=1.5, t0=60.0),
        source_index=80, probes=(120,), wavelength=20.0, sgf=1.5,
        window_fn="dikdortgen", state=st,
    )
    assert np.array_equal(st.e, kopya)


# ---------------------------------------------------------------------------
# 2. KARARLILIK KAPISI
# ---------------------------------------------------------------------------
def test_COURANT_KAPISI_SINIRIN_TAM_USTUNDE_ACILIYOR_ve_EPS_MIN_KULLANMIYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ — ama bir MODELLEME SEÇİMİNİ kilitler.

    Kapı `courant > courant_limit(1)` der; `medium.eps_min`i KULLANMAZ. Bu
    bilinçlidir: `S` en hızlı dalga hızıyla tanımlıdır, dolayısıyla eps_r = 9
    olan bir ortamda ham eşik 3.0 olsa bile kapı yine 1.0'da kapanır.
    Kapı `eps_min`e bağlansaydı `S` ortamdan ortama başka bir şey anlamına
    gelirdi ve `S·sqrt(d) = 1` çıpası ANLAMSIZLAŞIRDI.
    """
    yogun = Medium(np.full(240, 9.0), np.ones(240))
    yee_coefficients(yogun, courant=1.0, boundary=PEC())  # tam sınırda geçer
    with pytest.raises(InvalidInputError, match="kararlilik sinirini asiyor"):
        yee_coefficients(yogun, courant=1.0 + 1e-12, boundary=PEC())
    with pytest.raises(InvalidInputError, match="kararlilik sinirini asiyor"):
        yee_step_1d(FieldState.zeros_1d(240), yogun, courant=1.5, boundary=PEC())
    with pytest.raises(InvalidInputError):
        yee_coefficients(Medium.vacuum(240), courant=0.0, boundary=PEC())


def test_COURANT_LIMIT_UC_BOYUTU_TANIYOR_ama_COZUCU_ETMIYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    `courant_limit(3)` bir SAYIDIR ve boyut merdiveninin üçüncü basamağıdır;
    bedava. `yee_step_1d` ise 3B durumu REDDEDER — bedava olan sayı, koşum
    değil. Plan taslağı ikisini TEK fonksiyona yüklüyordu ve kendi Step 1
    testiyle çelişiyordu (`courant_limit(3)` hem 0.577 dönecek hem de
    `InvalidInputError` atacaktı).
    """
    assert courant_limit(3) == pytest.approx(3**-0.5, rel=1e-12)
    with pytest.raises(InvalidInputError):
        courant_limit(4)
    with pytest.raises(InvalidInputError):
        courant_limit(0)
    # TASK 4 GÜNCELLEMESİ: bu satır bir zamanlar
    # `FieldState(np.zeros((8,8)), np.zeros((8,8)), 2)` kuruyordu ve O NESNE
    # HİÇBİR ŞEKİL DENETİMİNDEN GEÇMİYORDU — `dim = 2` dalı boştu. 2B sevk
    # edilince delik ölçülerek kapandı (bkz. grid.FieldState dokstringi ve
    # tests/em/test_bessel.py); 2B durum artık `FieldState2D`dir.
    iki_b = FieldState2D.zeros_2d(8, 8)
    with pytest.raises(InvalidInputError, match="yalniz 1B"):
        yee_step_1d(iki_b, Medium.vacuum(8), courant=0.5, boundary=PEC())


# ---------------------------------------------------------------------------
# 3. FieldState / Window / FieldRecord
# ---------------------------------------------------------------------------
def test_FIELDSTATE_YARIM_HUCRE_KAYDIRMAYI_TIPTE_ZORLUYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    `h[i] = H(i+1/2)` olduğu için `h`, `e`den TAM BİR HÜCRE kısadır. Eşit
    uzunluk kabul edilseydi kaydırmayı düşüren bir hata sessizce geçerdi —
    ve o hata yakınsama mertebesini 2'den 1'e indirir (mutasyonla ölçüldü).

    BU TESTİN SON İKİ SATIRI TASK 4'TE DEĞİŞTİ, ve sebebi bir ÇÜRÜKTÜR:
    `FieldState` *"yalniz 1B ve 2B tanimlar"* diyordu ve bu test o cümleyi
    `match="1B ve 2B"` ile kilitliyordu. 2B sevk edilince ölçüldü ki
    `dim = 2` dalında **hiçbir denetim yoktu** — kilitlenen şey bir tanım
    değil, DENETİMSİZ BİR DELİKTİ. Artık `FieldState` yalnız 1B tutuyor,
    2B `FieldState2D`de ve orada üç alanın üçü de denetleniyor.
    """
    FieldState(np.zeros(10), np.zeros(9), 1)
    with pytest.raises(InvalidInputError, match="TAM BIR HUCRE"):
        FieldState(np.zeros(10), np.zeros(10), 1)
    with pytest.raises(InvalidInputError, match="TAM BIR HUCRE"):
        FieldState(np.zeros(10), np.zeros(8), 1)
    with pytest.raises(InvalidInputError, match="FieldState2D"):
        FieldState(np.zeros(10), np.zeros(9), 3)
    with pytest.raises(InvalidInputError, match="FieldState2D"):
        FieldState(np.zeros((8, 8)), np.zeros((8, 8)), 2)
    with pytest.raises(InvalidInputError, match="sayisal"):
        FieldState(np.array(["a", "b"]), np.array(["c"]), 1)


def test_FIELDSTATE_KARMASIK_ALANA_IZIN_VERIYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    Eğik geliş indirgemesi (Task 3) `d/dy -> i·ky` ile KARMAŞIK alan koşar ve
    AYNI çekirdekten geçmek zorundadır. `dtype=float`a zorlamak o yolu
    keserdi; bu yüzden `FieldState` dtype'ı kısıtlamaz.
    """
    st = FieldState.zeros_1d(64, dtype=complex)
    assert np.iscomplexobj(st.e)
    kayit = run_1d(
        Medium.vacuum(64), courant=0.5, boundary=PML(thickness=8), steps=40,
        source=lambda t: gauss_cos(t, wavelength=8.0, sgf=1.0, t0=12.0),
        source_index=20, probes=(32,), wavelength=8.0, sgf=1.0,
        window_fn="dikdortgen", dtype=complex,
    )
    assert np.iscomplexobj(kayit.probes)


def test_FIELDSTATE_HASH_TUZAGI_ODENMIS():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    `eq=False` olmasaydı üretilen `__eq__` bir DİZİ döndürür (`if a == b`
    patlar) ve `__hash__` `TypeError` verirdi. Depo bu tuzağı `hf/integrals`,
    `md/state` ve `spin/lattice`te zaten ödedi.
    """
    a = FieldState.zeros_1d(8)
    b = FieldState.zeros_1d(8)
    assert (a == b) is False  # kimlik karşılaştırması, dizi DEĞİL
    assert isinstance(hash(a), int)
    assert len({a, b}) == 2


def test_PENCERE_KAYITTA_ZORUNLU_ve_ADLI_YEDI_PARCASI_VAR():
    """SINIF: TRANSKRİPSİYON DENETİMİ — ama depo dersinin taşıyıcısı.

    Bu modülde tohumun karşılığı PENCERENİN TAM BEYANIDIR. Pencere
    opsiyonel olsaydı bir sayı hangi ayrıklaştırmaya ait olduğu bilinmeden
    dolaşabilirdi; bu depoda hükmün pencerenin dışına taşınması ON KEZ
    ölçüldü. `Window` bu yüzden `FieldRecord`un ZORUNLU alanıdır.
    """
    kayit = _kosum(Medium.vacuum(240), PML(thickness=15))
    assert isinstance(kayit, FieldRecord)
    assert isinstance(kayit.window, Window)
    w = kayit.window
    assert (w.cells, w.courant, w.steps, w.npml, w.sgf, w.window_fn) == (
        240, 0.5, 200, 15, 1.5, "dikdortgen",
    )
    assert w.wavelength == 20.0 and w.dx == 1.0
    assert len(w.as_lines()) == 7
    # PEC koşumunda npml SIFIR olarak beyan edilir — "yok" da bir pencere
    # değeridir, boş bırakılmaz.
    assert _kosum(Medium.vacuum(240), PEC()).window.npml == 0


def test_RUN_1D_GIRDI_DENETIMLERI():
    """SINIF: TRANSKRİPSİYON DENETİMİ."""
    vak = Medium.vacuum(64)
    with pytest.raises(InvalidInputError, match="birlikte"):
        run_1d(vak, courant=0.5, boundary=PEC(), steps=10, source=None,
               source_index=20, probes=(30,), wavelength=8.0, sgf=1.0)
    with pytest.raises(InvalidInputError, match="prob indisi"):
        run_1d(vak, courant=0.5, boundary=PEC(), steps=10, source=None,
               source_index=None, probes=(64,), wavelength=8.0, sgf=1.0)
    with pytest.raises(InvalidInputError, match="PEC duvarlarin ARASINDA"):
        run_1d(vak, courant=0.5, boundary=PEC(), steps=10,
               source=lambda t: 0.0, source_index=63, probes=(30,),
               wavelength=8.0, sgf=1.0)
    with pytest.raises(InvalidInputError, match="steps"):
        run_1d(vak, courant=0.5, boundary=PEC(), steps=-1, source=None,
               source_index=None, probes=(30,), wavelength=8.0, sgf=1.0)


# ---------------------------------------------------------------------------
# 4. Medium — arayüz ortalaması AÇIK PARAMETRE
# ---------------------------------------------------------------------------
def test_EY_KURAL_DORT_AYRI_DIZI_URETIYOR_ve_HICBIRI_DIGERININ_TAKMA_ADI_DEGIL():
    """SINIF: TRANSKRİPSİYON DENETİMİ — bir MODELLEME SEÇİMİNİ kilitler.

    `ey_kural` bir gürültü ekseni DEĞİL, modelleme seçimidir; ölçüldü —
    TİY faz farkını +0.156403 / +5.346570 / -8.127598 / +8.889958 derece
    oynatıyor (sevk penceresi `medium.Medium` dokstringinde). Aritmetik
    teğet bileşen için doğrudur.

    Bu test seçimin GERÇEKTEN dört ayrı sayı ürettiğini kilitler: kurallar
    sessizce aynı diziye çökerse yukarıdaki ölçüm anlamsızlaşır.
    """
    eps = np.array([1.0, 1.0, 9.0, 9.0])
    diziler = {}
    for kural in EY_KURALLARI:
        diziler[kural] = Medium(eps, np.ones(4), kural).eps_yari()
    assert diziler["aritmetik"][1] == pytest.approx(5.0)
    assert diziler["harmonik"][1] == pytest.approx(1.8)
    assert diziler["sol"][1] == pytest.approx(1.0)
    assert diziler["sag"][1] == pytest.approx(9.0)
    for a in EY_KURALLARI:
        for b in EY_KURALLARI:
            if a != b:
                assert not np.array_equal(diziler[a], diziler[b])
    with pytest.raises(InvalidInputError, match="ey_kural"):
        Medium(eps, np.ones(4), "geometrik")


def test_MU_YARI_EY_KURALA_BAGLI_DEGIL():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    `ey_kural` teğet E'nin süreklilik kuralıdır, H'nin değil. `mu_yari`nin
    ona bağlanması SESSİZ bir bağ olurdu: bu modülde `mu_r` her yerde 1
    olduğu için dört kural da aynı diziyi verir ve bağ HİÇ FARK
    ETTİRMEZDİ — yani hata yıllarca görünmezdi. Test bunu mu_r'yi DEĞİŞKEN
    yaparak sınıyor.
    """
    mu = np.array([1.0, 1.0, 9.0, 9.0])
    eps = np.ones(4)
    beklenen = np.array([1.0, 5.0, 9.0])
    for kural in EY_KURALLARI:
        assert np.array_equal(Medium(eps, mu, kural).mu_yari(), beklenen)


def test_MEDIUM_GIRDI_DENETIMLERI():
    """SINIF: TRANSKRİPSİYON DENETİMİ."""
    with pytest.raises(InvalidInputError, match="ayni sekilde"):
        Medium(np.ones(4), np.ones(5))
    with pytest.raises(InvalidInputError, match="pozitif"):
        Medium(np.array([1.0, 0.0]), np.ones(2))
    with pytest.raises(InvalidInputError, match="pozitif"):
        Medium(np.ones(2), np.array([1.0, -3.0]))
    with pytest.raises(InvalidInputError, match="interface"):
        Medium.half_space(10, interface=0)
    with pytest.raises(InvalidInputError, match="interface"):
        Medium.half_space(10, interface=10)
    m = Medium.half_space(10, interface=4, eps1=1.0, eps2=4.0)
    assert m.eps_min == 1.0
    assert m.eps_r[3] == 1.0 and m.eps_r[4] == 4.0  # `interface` ARTIK ortam 2


# ---------------------------------------------------------------------------
# 5. Sınırlar
# ---------------------------------------------------------------------------
def test_PML_PROFILI_HUCRE_MERKEZINDE_ORNEKLENIYOR_ve_UCUNCU_DERECEDEN():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    İki ayrıntı ve ikisi de sessizce bozulabilir:
    (a) örnekleme noktası HÜCRE MERKEZİDİR ((r+0.5)/d). Kenardan
        örneklenseydi katmanın ilk hücresinde sigma = 0 olur, yani katman
        bir hücre KISALIRDI.
    (b) profil derecesi `m = 3`. `m = 1` konursa profil doğrusallaşır ve
        aynı `sigma_max` için soğurma azalır.
    `sigma_max` GİRİLMEZ, hedef yansımadan TÜRETİLİR.
    """
    p = PML(thickness=15, m=3.0, r0=1e-6)
    assert p.sigma_max == pytest.approx(
        -(3.0 + 1.0) * np.log(1e-6) / (2.0 * 15), rel=1e-12
    )
    prof = p.profile()
    assert prof.shape == (15,)
    assert prof[0] > 0.0  # (a) iç yüzde SIFIR DEĞİL
    assert prof[0] / p.sigma_max == pytest.approx((0.5 / 15) ** 3, rel=1e-12)
    assert prof[-1] / p.sigma_max == pytest.approx((14.5 / 15) ** 3, rel=1e-12)
    # (b) derece gerçekten 3: iki noktanın oranı kübiktir.
    assert prof[9] / prof[4] == pytest.approx((9.5 / 4.5) ** 3, rel=1e-12)


@pytest.mark.parametrize("yuz", YUZLER)
def test_SINIR_YUZLERI_OLCUMUN_KOSULU(yuz):
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    `yuzler` bir kolaylık değildir: soğurucu yansıması İKİ KOŞUMUN FARKINDAN
    okunur ve soğurucu iki uca birden konursa sol uçtan dönen dalga da farka
    karışır — ölçülen şey artık sağ soğurucunun yansıması olmaz.
    """
    sig = PML(thickness=10, yuzler=yuz).sigma_1d(100)
    sol, sag = sig[:10].sum(), sig[90:].sum()
    assert sig[10:90].sum() == 0.0
    if yuz == "iki":
        assert sol > 0 and sag > 0
    elif yuz == "sol":
        assert sol > 0 and sag == 0.0
    else:
        assert sol == 0.0 and sag > 0
    with pytest.raises(InvalidInputError, match="yuzler"):
        PML(thickness=10, yuzler="orta")


def _son_enerji(sinir, steps, kaynak):
    kayit = run_1d(
        Medium.vacuum(400), courant=0.5, boundary=sinir, steps=steps,
        source=kaynak, source_index=200, probes=(100,), wavelength=20.0,
        sgf=1.5, window_fn="dikdortgen",
    )
    e, h = kayit.final.e, kayit.final.h
    return 0.5 * float(np.sum(e**2)) + 0.5 * float(np.sum(h**2))


def test_PEC_SOGURUCUYU_KALDIRIR_DUVARI_DEGIL_ve_ENERJIYI_KORUR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    PEC "soğurucu yok" demektir; ızgara uçlarındaki `E = 0` koşulu HER sınır
    tipinde koşar. Yani PEC tam yansıtıcı bir boşluk kurar ve KAYIPSIZDIR;
    PML'de ise enerji gider.

    ÖLÇÜLDÜ (N = 400, S = 0.5, DC'siz Ricker kaynak wavelength = 20,
    sgf = 3, kaynak x = 200):
        adım      PEC             PML(d=15)      oran
        2000      3.051443e+00    1.3989e-10     4.58e-11
        4000      3.051427e+00    7.2305e-25     2.37e-25
        8000      3.048488e+00    1.6182e-31     5.31e-32
    PEC'in enerjisi 8000 adım boyunca 1e-3 bağıl bandın içinde SABİT;
    PML'inki 31 mertebe düşüyor.
    """
    assert np.all(PEC().sigma_1d(50) == 0.0)
    assert PEC().thickness == 0
    kaynak = lambda t: ricker(t, wavelength=20.0, sgf=3.0)  # noqa: E731

    pec = [_son_enerji(PEC(), st, kaynak) for st in (2000, 4000, 8000)]
    pml = [_son_enerji(PML(thickness=15), st, kaynak) for st in (2000, 4000)]

    assert min(pec) > 0.0
    assert (max(pec) - min(pec)) / np.mean(pec) < 1e-3  # PEC KAYIPSIZ
    assert pml[0] < 1e-9 * pec[0]
    assert pml[1] < 1e-20 * pec[1]  # ikinci gecişte bir daha soğuruluyor


def test_PEC_DUVAR_CAGIRANIN_BASLANGIC_DURUMUNDA_DA_UYGULANIYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ — MUTASYONUN AÇTIĞI BOŞLUK.

    `_adim`deki `e[0] = 0` satırı silindiğinde ilk turda HİÇBİR test düşmedi.
    Sebep ölçüldü ve bir KAPSAM boşluğuydu: her test `zeros_1d` ile
    başlıyordu, `e[0]` zaten sıfırdı ve `e[0]`a hiçbir zaman curl terimi
    eklenmediği için sonsuza kadar sıfır kalıyordu. Yani satır "gereksiz"
    değil, GÖRÜNMEZDİ.

    Çağıran sıfır olmayan bir `e[0]` verirse satırın işi ORTAYA ÇIKAR ve
    büyüktür — ölçüldü (N = 200, S = 0.5, PEC, 80 adım, e[0] = 1):
        duvar VAR : e[0] = 0.000000   h[0] = -0.000743   enerji  0.134
        duvar YOK : e[0] = 1.000000   h[0] = -0.999718   enerji 40.158
    Donmuş `e[0]`, `h[0] <- da·h[0] + db·(e[1] - e[0])` üzerinden ızgarayı
    sürekli besliyor: 300 kat enerji.

    DERS: hayatta kalan mutant her zaman "test ekle" demez, ama bu sefer
    dedi — çünkü mutantın etkisi GERÇEK ve BÜYÜK, sadece hiçbir koşum onu
    uyandırmıyordu.
    """
    N = 200
    st = FieldState.zeros_1d(N)
    e = st.e.copy()
    e[0] = 1.0
    kayit = run_1d(
        Medium.vacuum(N), courant=0.5, boundary=PEC(), steps=80, source=None,
        source_index=None, probes=(1,), wavelength=20.0, sgf=1.0,
        window_fn="dikdortgen", state=FieldState(e, st.h, 1),
    )
    son = kayit.final
    assert son.e[0] == 0.0
    assert son.e[-1] == 0.0
    enerji = 0.5 * float(np.sum(son.e**2)) + 0.5 * float(np.sum(son.h**2))
    assert enerji < 1.0  # duvar düşerse 40'ın üstüne çıkıyor


def test_PML_KALINTISI_SOGURUCUNUN_DEGIL_KAYNAGIN_DC_SININ():
    """SINIF: TRANSKRİPSİYON DENETİMİ — ve bir TUZAK kaydı.

    İlk yazımda bu kilit `gauss_cos` kaynağıyla kuruldu ve PML kalıntısı
    2.5e-05'te takıldı; "PML'in yansıması" gibi göründü. ÖLÇÜLDÜ, DEĞİL:

        kaynak              adım 2000    adım 8000    adım 16000
        gauss_cos sgf=1.5   2.502e-05    1.118e-05    6.894e-06
        ricker    (DC = 0)  4.584e-11    5.308e-32    2.375e-32

    Ve kalıntı KALINLIKLA da neredeyse değişmiyordu (8000 adımda d = 8 ->
    1.256e-05, d = 60 -> 5.394e-06; katman 7.5 kat kalınlaşınca kalıntı
    yalnız 2.3 kat düşüyor) — soğurucunun yansıması olsaydı üstel düşerdi.
    Kalan şey KAYNAĞIN DC bileşenidir: `gauss_cos` Gauss zarflı bir
    taşıyıcıdır ve `sgf` küçükken sıfır ortalamalı DEĞİLDİR.

    DERS: "soğurucu iyi soğurmuyor" demeden önce KAYNAĞIN DC'sine bak.
    """
    gc = lambda t: gauss_cos(t, wavelength=20.0, sgf=1.5, t0=60.0)  # noqa: E731
    rk = lambda t: ricker(t, wavelength=20.0, sgf=3.0)  # noqa: E731
    g = _son_enerji(PML(thickness=15), 4000, gc) / _son_enerji(PEC(), 4000, gc)
    r = _son_enerji(PML(thickness=15), 4000, rk) / _son_enerji(PEC(), 4000, rk)
    assert g == pytest.approx(1.486e-05, rel=0.05)
    assert r < 1e-20
    # Kalınlık kalıntıyı ÜSTEL düşürmüyor -> kalıntı soğurucuya ait değil.
    ince = _son_enerji(PML(thickness=8), 8000, gc)
    kalin = _son_enerji(PML(thickness=60), 8000, gc)
    assert 1.5 < ince / kalin < 4.0


def test_LOSSYLAYER_SEVK_EDILMIYOR_ama_SIGMA_MAX_ELLE_VERILIYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    `LossyLayer` PML'in REDDEDİLEN rakibidir ve tek farkı `sigma_max`ın
    hedef yansımadan TÜRETİLMEMESİDİR. Test bu tek farkı kilitler — kıyas
    nesnesi PML'e yaklaşırsa "PML daha iyi" hükmü anlamını kaybeder.
    """
    lay = LossyLayer(thickness=30, sigma_max=0.35, m=3.0)
    assert not hasattr(lay, "r0")
    assert lay.profile()[-1] == pytest.approx(0.35 * (29.5 / 30) ** 3, rel=1e-12)
    assert lay.sigma_max == 0.35
    with pytest.raises(InvalidInputError, match="sigma_max"):
        LossyLayer(sigma_max=-1.0)


def test_SINIR_GIRDI_DENETIMLERI():
    """SINIF: TRANSKRİPSİYON DENETİMİ."""
    with pytest.raises(InvalidInputError, match="kalinl"):
        PML(thickness=-1)
    with pytest.raises(InvalidInputError, match="m > 0"):
        PML(m=0.0)
    with pytest.raises(InvalidInputError, match="r0"):
        PML(r0=1.0)
    with pytest.raises(InvalidInputError, match="sigmali"):
        PML(thickness=15).sigma_1d(20)
    with pytest.raises(InvalidInputError, match="en az 3"):
        PEC().sigma_1d(2)


# ---------------------------------------------------------------------------
# 6. Kaynaklar — `sgf` İKİ DARBEDE İKİ AYRI ŞEY
# ---------------------------------------------------------------------------
def _bagil_bant(fn, lam, **kw):
    """Bağıl -3 dB bant genişliği (df/f_p). Örnekleme lambda ile ölçekli."""
    M, dt = 1 << 16, lam / 160.0
    t = np.arange(M) * dt
    F = np.abs(np.fft.rfft(fn(t, wavelength=lam, **kw)))
    fr = np.fft.rfftfreq(M, d=dt)
    ust = F >= F.max() / np.sqrt(2.0)
    return float((fr[ust].max() - fr[ust].min()) * lam)


def test_SGF_GAUSS_COS_TA_BANT_DUGMESI_RICKER_DA_DEGIL():
    """SINIF: TRANSKRİPSİYON DENETİMİ — bir PLAN HATASINI kilitler.

    Plan taslağı tek bir `ricker(t, *, wavelength, sgf)` tanımlayıp
    dokstringine `sg = sgf * wavelength` yazıyor ve 1560 katlık BANT
    GENİŞLİĞİ ölçümünü ona bağlıyordu. Ama o ölçüm Gauss x cos darbesiyle
    yapılmıştı ve Ricker'da öyle bir düğme YOKTUR: Ricker'ın spektrumu
    `(f/f_p)^2 exp(-(f/f_p)^2)`dir, bağıl bandı yalnız `f_p` belirler.

    ÖLÇÜLDÜ (örnekleme dt = lambda/160, M = 65536):
        ricker    : lambda 20/40/80/160 -> 0.822754 hepsinde
                    sgf     1/2/4/8     -> 0.822754 hepsinde
        gauss_cos : sgf 1/2/3/6/9 -> 0.261230 / 0.129395 / 0.085449 /
                    0.041504 / 0.026855   (yani 1/sgf)
    """
    for lam in (20.0, 40.0, 80.0, 160.0):
        assert _bagil_bant(ricker, lam, sgf=1.0) == pytest.approx(
            0.822754, abs=1e-6
        )
    for sgf in (1.0, 2.0, 4.0, 8.0):
        assert _bagil_bant(ricker, 40.0, sgf=sgf) == pytest.approx(
            0.822754, abs=1e-6
        )
    gc = [_bagil_bant(gauss_cos, 40.0, sgf=s, t0=40.0 * s * 5.0)
          for s in (1.0, 2.0, 3.0, 6.0, 9.0)]
    assert gc == pytest.approx(
        [0.261230, 0.129395, 0.085449, 0.041504, 0.026855], abs=1e-6
    )
    # ve gerçekten 1/sgf: sgf üç katına çıkınca bant üçte bire iner.
    assert gc[0] / gc[2] == pytest.approx(3.0, rel=0.03)


def test_RICKER_SIFIR_ORTALAMALI_ve_TEPESI_T0_DA():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    Ricker Gauss'un ikinci türevidir, dolayısıyla DC bileşeni tam sıfırdır —
    FDTD'de bu önemlidir: DC'si olan bir darbe ızgarada sönmeyen bir bileşen
    bırakır (ölçüldü, bkz. `test_PML_KALINTISI_...`). `sgf` yalnız
    gecikmedir: tepe `t0 = sgf*lambda`da.

    PENCERE ZORUNLU ve ÖLÇÜLDÜ: integral `t0` çevresinde SİMETRİK bir
    pencerede alınmalıdır. Simetrik pencerede (yarı genişlik 5/10/20 lambda,
    sgf 1/2/4 — dokuzunda da) integral 2.220e-16'dır, yani makine
    hassasiyeti. `t = 0`dan başlayan TEK YANLI bir pencerede ise sgf = 1'de
    1.950e-03 çıkar — sıfırdan 13 mertebe uzak. Bu, darbenin değil
    PENCERENİN özelliğidir ve tam olarak bu depoda on kez çıkan hatadır:
    hüküm pencerenin dışına taşınırsa sayı anlamını kaybeder.
    """
    lam = 40.0
    for sgf in (1.0, 2.0, 4.0):
        for yari in (5.0, 10.0, 20.0):
            t = np.arange(sgf * lam - yari * lam, sgf * lam + yari * lam, 0.25)
            s = ricker(t, wavelength=lam, sgf=sgf)
            assert abs(float(np.sum(s)) * 0.25) < 1e-12
    # TEK YANLI pencere sıfır vermez — ve bu darbenin değil pencerenin işi.
    t_tek = np.arange(0.0, 40 * lam, 0.25)
    tek_yanli = abs(float(np.sum(ricker(t_tek, wavelength=lam, sgf=1.0))) * 0.25)
    assert tek_yanli == pytest.approx(1.950e-03, rel=0.01)
    for sgf in (1.0, 2.0, 4.0):
        t = np.arange(0.0, 40 * lam, 0.25)
        s = ricker(t, wavelength=lam, sgf=sgf)
        assert t[int(np.argmax(s))] == pytest.approx(sgf * lam, abs=0.5)


def test_KAYNAK_GIRDI_DENETIMLERI():
    """SINIF: TRANSKRİPSİYON DENETİMİ."""
    for fn, kw in ((ricker, {}), (gauss_cos, {"t0": 0.0})):
        with pytest.raises(InvalidInputError, match="wavelength"):
            fn(0.0, wavelength=0.0, sgf=1.0, **kw)
        with pytest.raises(InvalidInputError, match="sgf"):
            fn(0.0, wavelength=10.0, sgf=0.0, **kw)
