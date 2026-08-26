"""HÜCRE = UZUNLUK: `dx` bir DÜĞME DEĞİL, bir BEYAN.

Bu dosya Modül 5'in kapanışta **beyan edilerek açık bırakılan** kusurunu
kapatan ölçümü taşır. Kusur şuydu: `run_1d`/`run_2d`/`run_oblique_1d`
kaynağı `t = n·S·Δx` zamanında besliyordu, oysa `wavelength` **hücre**
cinsinden veriliyor ve `Window.as_lines()` onu "lambda ... hucre" diye
basıyordu. İkisi yalnız `Δx = 1`'de örtüşür; `Δx ≠ 1`'de sonuç **sessizce
yanlış** çıkıyordu.

İKİ MEŞRU YOL VARDI ve seçim ÖLÇÜMLE yapıldı:

    (1) `wavelength`i fiziksel uzunluk yapmak, `λ_hücre = wavelength/Δx`
    (2) `Δx ≠ 1`'i reddetmek

(2) seçildi, çünkü ölçüldüğünde `Δx` bir düğme ÇIKMADI: bu şemada davranış
`(Δx, wavelength)` çiftine YALNIZ `wavelength/Δx` oranı üzerinden bağlı, ve
bu **bit bit** doğrulanıyor (aşağıda 1B'de beş, 2B'de dört, eğikte dört
çift). Yani `Δx` ayrı bir serbestlik derecesi değil, `wavelength`in ölçekli
bir kopyası — düzeltilecek bir düğme değil, KALDIRILACAK bir ikinci ad.
Sebebi yapısal: `Δx` ızgarayı ölçekleyemez. Katsayılar zaten ölçekten
bağımsız (`cb = S/eps`, içinde `Δx` YOK), geometri hücre cinsinden
(`probes`, `source_index`, `CircularPEC.radius`, `npml`), PML `sigma`sı hücre
başına, `oblique`in `ky`si radyan/hücre. `Δx ≠ 1`'de değişen TEK şey
kaynağın saatidir — ve o saat `wavelength`in ta kendisidir.

KAPI KAPANINCA BİR ŞEY KAZANILDI: "lambda ... hucre" beyanı artık bir etiket
değil bir **TEOREM**, ve İLK KEZ SINANABİLİR. Sınav dairesel değildir —
ızgaradaki dalga boyu iki probun FAZ FARKINDAN okunuyor ve Yee'nin KAPALI
FORM dispersiyonuyla karşılaştırılıyor.

KARŞI-OLGUSAL ÇEKİRDEK BU DOSYADADIR, `src/`de DEĞİL — `test_sema_cipalari`
kaydırmasız şemayı nasıl tutuyorsa öyle. `_kosum_dx` kapı kapanmadan önceki
döngünün ta kendisidir ve sadakati ilk testte BİT BİT ölçülüyor.

PENCERE (bu dosyadaki her koşum için): `c = eps0 = mu0 = 1`, tohum YOK
(FDTD deterministtir), FFT penceresi dikdörtgen. Izgara/S/adım her testin
kendi dokstringinde.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from qlab.engine.em.boundary import PEC, PML
from qlab.engine.em.grid import (
    HUCRE_DX,
    FieldState,
    FieldState2D,
    Window,
    dx_kapisi,
)
from qlab.engine.em.medium import Medium
from qlab.engine.em.oblique import (
    oblique_coefficients,
    oblique_step,
    run_oblique_1d,
)
from qlab.engine.em.source import gauss_cos, ricker
from qlab.engine.em.yee import (
    run_1d,
    run_2d,
    yee_coefficients,
    yee_coefficients_2d,
    yee_step_1d,
    yee_step_2d,
)
from qlab.errors import InvalidInputError

S = 0.5
SGF = 2.0


def _darbe(W: float):
    """`t0 = 5·sgf·W` — zarf da `W` ile ölçeklenir.

    Bu bir kolaylık değil ÇÖKME ÖLÇÜMÜNÜN ÖN KOŞULU: `gauss_cos` yalnız
    `(t - t0)/W` ve `(t - t0)/(sgf·W)` üzerinden bağlıdır, dolayısıyla `t0`
    da `W` ile ölçeklenirse darbe `t/W` cinsinden AYNI fonksiyondur. `t0`
    sabit bırakılsaydı çökme ölçümü darbenin GECİKMESİNİ de oynatırdı ve
    ölçülen şey `Δx`in kendisi olmazdı.
    """
    t0 = 5.0 * SGF * W
    return lambda t: float(gauss_cos(t, wavelength=W, sgf=SGF, t0=t0))


# ---------------------------------------------------------------------------
# KARŞI-OLGUSAL ÇEKİRDEK — kapı kapanmadan önceki döngü
# ---------------------------------------------------------------------------
def _kosum_dx(medium, *, boundary, steps, source, source_index, probes, dx):
    """`dx` KAPISI KAPANMADAN ÖNCEKİ `run_1d` DÖNGÜSÜ: `t = n·S·Δx`.

    SEVK EDİLEN `yee_step_1d` ile SEVK EDİLEN `yee_coefficients`i kullanır;
    tek farkı kaynağın saatidir. `_adim` her adımda YENİ bir `e` ürettiği
    için yerinde yazma güvenlidir — `run_1d`in kendi gerekçesi.
    """
    co = yee_coefficients(medium, courant=S, boundary=boundary)
    st = FieldState.zeros_1d(medium.cells)
    pidx = np.asarray(probes, dtype=int)
    kayit = np.zeros((len(probes), steps))
    for n in range(steps):
        kayit[:, n] = st.e[pidx]
        st = yee_step_1d(st, medium, courant=S, boundary=boundary)
        if source is not None:
            st.e[source_index] += co.cb[source_index] * source(n * S * dx)
    return kayit


def _kosum_dx_2d(medium, *, boundary, steps, source, source_index, probes, dx):
    """`_kosum_dx`in 2B eşi — sevk edilen `yee_step_2d`den geçer."""
    co = yee_coefficients_2d(medium, courant=S, boundary=boundary)
    nx, ny = medium.shape
    st = FieldState2D.zeros_2d(nx, ny)
    pi = np.asarray([p[0] for p in probes], dtype=int)
    pj = np.asarray([p[1] for p in probes], dtype=int)
    kayit = np.zeros((len(probes), steps))
    for n in range(steps):
        kayit[:, n] = st.e[pi, pj]
        st = yee_step_2d(st, medium, courant=S, boundary=boundary)
        if source is not None:
            st.e[source_index] += co.cb[source_index] * source(n * S * dx)
    return kayit


def _kosum_dx_egik(medium, *, boundary, steps, source, source_index, probes,
                   ky, dx, courant=0.4):
    """`_kosum_dx`in eğik eşi — sevk edilen `oblique_step`ten geçer."""
    co = oblique_coefficients(medium, courant=courant, boundary=boundary,
                              polarization="s", ky=ky)
    st = FieldState.zeros_1d(medium.cells, dtype=complex)
    aux = np.zeros(medium.cells, dtype=complex)
    pidx = np.asarray(probes, dtype=int)
    kayit = np.zeros((len(probes), steps), dtype=complex)
    for n in range(steps):
        kayit[:, n] = st.e[pidx]
        st, aux = oblique_step(st, aux, co)
        if source is not None:
            st.e[source_index] += co.main.cb[source_index] * source(
                n * courant * dx
            )
    return kayit


# ---------------------------------------------------------------------------
# 1. Karşı-olgusal çekirdek SADIK MI
# ---------------------------------------------------------------------------
def test_KARSI_OLGUSAL_DONGU_SEVK_EDILEN_KOSUMUN_TA_KENDISI():
    """SINIF: TRANSKRİPSİYON DENETİMİ — ama bu dosyanın ÖN KOŞULU.

    `_kosum_dx` "eskiden ne oluyordu"yu temsil ediyorsa, `Δx = 1`'de sevk
    edilen `run_1d` ile BİT BİT aynı olmak ZORUNDADIR. Olmasaydı bu
    dosyadaki bütün çökme ölçümleri BAŞKA bir çekirdeğin ölçümü olurdu ve
    hükümleri `run_1d` hakkında hiçbir şey söylemezdi.

    PENCERE: 1B vakum, PML 20, N = 400, S = 0.5, 300 adım, `gauss_cos`
    sgf = 2, W = 24, t0 = 5·sgf·W, kaynak 100, problar 200/260.
    ÖLÇÜLDÜ: `np.array_equal` True, max|fark| = 0.0. Aynısı 2B'de
    (21×21, PEC, 80 adım) ve eğikte (N = 400, PML 20, S = 0.4, 900 adım).
    """
    med, bnd = Medium.vacuum(400), PML(thickness=20)
    kaynak = _darbe(24.0)
    ref = run_1d(med, courant=S, boundary=bnd, steps=300, source=kaynak,
                 source_index=100, probes=(200, 260), wavelength=24.0, sgf=SGF)
    alt = _kosum_dx(med, boundary=bnd, steps=300, source=kaynak,
                    source_index=100, probes=(200, 260), dx=1.0)
    assert np.array_equal(ref.probes, alt)

    med2 = Medium.uniform_2d(21, 21, eps_r=1.0)

    def k2(t):
        return float(ricker(t, wavelength=10.0, sgf=1.5))

    ref2 = run_2d(med2, courant=S, boundary=PEC(), steps=80, source=k2,
                  source_index=(7, 8), probes=((12, 13), (15, 9)),
                  wavelength=10.0, sgf=1.5)
    alt2 = _kosum_dx_2d(med2, boundary=PEC(), steps=80, source=k2,
                        source_index=(7, 8), probes=((12, 13), (15, 9)), dx=1.0)
    assert np.array_equal(ref2.probes, alt2)

    ky = 0.5553603672697957
    ref3 = run_oblique_1d(med, courant=0.4, boundary=bnd, steps=900,
                          source=kaynak, source_index=100, probes=(200, 260),
                          wavelength=24.0, sgf=SGF, ky=ky, polarization="s")
    alt3 = _kosum_dx_egik(med, boundary=bnd, steps=900, source=kaynak,
                          source_index=100, probes=(200, 260), ky=ky, dx=1.0)
    assert np.array_equal(ref3.probes, alt3)


# ---------------------------------------------------------------------------
# 2. ÇÖKME — kapının GEREKÇESİ
# ---------------------------------------------------------------------------
def test_dx_AYRI_BIR_SERBESTLIK_DERECESI_DEGIL_1B():
    """SINIF: ŞEMA ÇIPASI. Kapının gerekçesi BURADA ölçülüyor.

    Davranış `(Δx, wavelength)` çiftine YALNIZ `wavelength/Δx` oranı
    üzerinden bağlıdır. PENCERE: 1B vakum, PML 20, N = 400, S = 0.5, 300
    adım, `gauss_cos` sgf = 2, t0 = 5·sgf·W, kaynak 100, problar 200/260.
    ÖLÇÜLDÜ — `W/Δx = 24` tutan BEŞ çift de BİT BİT aynı diziyi veriyor:

        (Δx, W) = (1, 24) / (0.5, 12) / (2, 48) / (0.25, 6) / (8, 192)
        max|fark| = 0 (hepsinde), np.array_equal = True (hepsinde)

    Aynı `W` ile `Δx` oynatılırsa (yani "hücre" okumasının varsaydığı şey)
    sonuç BAŞKA bir koşumdur: `Δx` = 0.5 / 2 için bağıl fark 0.878070 /
    26.252233. Yani düğme bağlıdır — ama bağlı olduğu şey `wavelength`tir.

    KAYNAKSIZ koşumda ise `Δx` HİÇBİR ŞEYE dokunmuyor: aynı pencerede
    `source = None` ile Δx = 0.25/0.5/1/2/7'nin beşi de bit bit aynı.
    """
    med, bnd = Medium.vacuum(400), PML(thickness=20)
    ortak = dict(boundary=bnd, steps=300, source_index=100, probes=(200, 260))

    taban = _kosum_dx(med, source=_darbe(24.0), dx=1.0, **ortak)
    for dx, W in ((1.0, 24.0), (0.5, 12.0), (2.0, 48.0), (0.25, 6.0),
                  (8.0, 192.0)):
        assert W / dx == 24.0
        assert np.array_equal(
            _kosum_dx(med, source=_darbe(W), dx=dx, **ortak), taban
        ), f"(dx, W) = ({dx}, {W}) tabandan SAPTI"

    # Aynı W, farklı Δx -> BAŞKA bir koşum.
    olcek = float(np.max(np.abs(taban)))
    for dx, beklenen in ((0.5, 0.878070), (2.0, 26.252233)):
        a = _kosum_dx(med, source=_darbe(24.0), dx=dx, **ortak)
        assert not np.array_equal(a, taban)
        bagil = float(np.max(np.abs(a - taban))) / olcek
        assert bagil == pytest.approx(beklenen, rel=1e-5)

    # Kaynaksız: Δx hiçbir şeye dokunmuyor.
    bos = _kosum_dx(med, source=None, dx=1.0, **ortak)
    for dx in (0.25, 0.5, 2.0, 7.0):
        assert np.array_equal(_kosum_dx(med, source=None, dx=dx, **ortak), bos)


def test_dx_AYRI_BIR_SERBESTLIK_DERECESI_DEGIL_2B_ve_EGIK():
    """SINIF: ŞEMA ÇIPASI. Aynı çökme, ikinci ve üçüncü koşum yolunda.

    2B PENCERE: 120×120 vakum, PML 20, S = 0.5, 400 adım, sgf = 2, kaynak
    (40, 40), problar (70, 70)/(90, 55). `W/Δx = 20` tutan DÖRT çift —
    (1, 20)/(0.5, 10)/(2, 40)/(4, 80) — BİT BİT aynı.

    EĞİK PENCERE: N = 400 vakum, PML 20, S = 0.4, 900 adım, sgf = 2,
    `ky` = 0.5553603673 SABİT, kaynak 100, problar 200/260. `W/Δx = 24`
    tutan DÖRT çift — (1, 24)/(0.5, 12)/(2, 48)/(8, 192) — BİT BİT aynı.

    EĞİKTE BU AYRICA BİR BİRİM SINAVIDIR: `ky` çiftler arasında HİÇ
    değişmiyor ve sonuç yine de bit bit aynı çıkıyor. `ky` uzunluk başına
    olsaydı `Δx` ile ölçeklenmesi gerekirdi ve çökme BOZULURDU; radyan
    BÖLÜ HÜCRE olduğu için bozulmuyor.
    """
    med2 = Medium.uniform_2d(120, 120, eps_r=1.0)
    ortak2 = dict(boundary=PML(thickness=20), steps=400,
                  source_index=(40, 40), probes=((70, 70), (90, 55)))
    taban2 = _kosum_dx_2d(med2, source=_darbe(20.0), dx=1.0, **ortak2)
    for dx, W in ((1.0, 20.0), (0.5, 10.0), (2.0, 40.0), (4.0, 80.0)):
        assert W / dx == 20.0
        assert np.array_equal(
            _kosum_dx_2d(med2, source=_darbe(W), dx=dx, **ortak2), taban2
        ), f"2B'de (dx, W) = ({dx}, {W}) saptı"
    assert not np.array_equal(
        _kosum_dx_2d(med2, source=_darbe(20.0), dx=2.0, **ortak2), taban2
    )

    med, bnd = Medium.vacuum(400), PML(thickness=20)
    ky = 0.5553603672697957
    ortak3 = dict(boundary=bnd, steps=900, source_index=100,
                  probes=(200, 260), ky=ky)
    taban3 = _kosum_dx_egik(med, source=_darbe(24.0), dx=1.0, **ortak3)
    for dx, W in ((1.0, 24.0), (0.5, 12.0), (2.0, 48.0), (8.0, 192.0)):
        assert W / dx == 24.0
        assert np.array_equal(
            _kosum_dx_egik(med, source=_darbe(W), dx=dx, **ortak3), taban3
        ), f"eğikte (dx, W) = ({dx}, {W}) saptı"
    assert not np.array_equal(
        _kosum_dx_egik(med, source=_darbe(24.0), dx=2.0, **ortak3), taban3
    )


# ---------------------------------------------------------------------------
# 3. KAPININ KENDİSİ
# ---------------------------------------------------------------------------
def test_dx_KAPISI_DORT_YERDE_DE_KAPALI():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    Kapı ÜÇ koşum fonksiyonunda (hızlı düşmek için, koşumdan ÖNCE) ve
    ayrıca `Window`da (elle kurulan bir pencere yalan söyleyemesin diye)
    duruyor. Dördüncüsü `FieldState`in `dim = 2` deliğini tipte kapatmasıyla
    aynı gerekçe: sayı `Window`la dolaşıyor, `Window` da beyanı basıyor.
    """
    vak = Medium.vacuum(64)
    for dx in (0.5, 2.0, 0.0, -1.0, float("nan")):
        with pytest.raises(InvalidInputError, match="REDDEDILDI"):
            run_1d(vak, courant=S, boundary=PEC(), steps=10, source=None,
                   source_index=None, probes=(30,), wavelength=8.0, sgf=1.0,
                   dx=dx)
        with pytest.raises(InvalidInputError, match="REDDEDILDI"):
            run_2d(Medium.uniform_2d(9, 9, eps_r=1.0), courant=S,
                   boundary=PEC(), steps=10, source=None, source_index=None,
                   probes=((4, 4),), wavelength=8.0, sgf=1.0, dx=dx)
        with pytest.raises(InvalidInputError, match="REDDEDILDI"):
            run_oblique_1d(vak, courant=0.4, boundary=PEC(), steps=10,
                           source=None, source_index=None, probes=(30,),
                           wavelength=8.0, sgf=1.0, ky=0.1,
                           polarization="s", dx=dx)
        with pytest.raises(InvalidInputError, match="REDDEDILDI"):
            Window(cells=64, dx=dx, courant=S, steps=10, wavelength=8.0,
                   npml=0, sgf=1.0, window_fn="dikdortgen")

    # KAPI KOŞUMDAN ÖNCE DÜŞÜYOR — ve bunu ÜÇ koşum yolunda da sınamak
    # ZORUNLU. Sınamasaydık kapı koşumdan SONRA da düşebilirdi (`Window`
    # onu yine yakalardı) ve testler farkı GÖREMEZDİ: mutasyon koşumunda
    # `run_2d` ile `run_oblique_1d`in erken kapısını silen iki mutant tam
    # olarak bu yüzden HAYATTA KALDI. Ayırt eden şey `steps = -1`: kapı
    # erkense mesaj "REDDEDILDI", geç kalırsa "steps negatif olamaz".
    with pytest.raises(InvalidInputError, match="REDDEDILDI"):
        run_1d(vak, courant=S, boundary=PEC(), steps=-1, source=None,
               source_index=None, probes=(30,), wavelength=8.0, sgf=1.0,
               dx=2.0)
    with pytest.raises(InvalidInputError, match="REDDEDILDI"):
        run_2d(Medium.uniform_2d(9, 9, eps_r=1.0), courant=S, boundary=PEC(),
               steps=-1, source=None, source_index=None, probes=((4, 4),),
               wavelength=8.0, sgf=1.0, dx=2.0)
    with pytest.raises(InvalidInputError, match="REDDEDILDI"):
        run_oblique_1d(vak, courant=0.4, boundary=PEC(), steps=-1,
                       source=None, source_index=None, probes=(30,),
                       wavelength=8.0, sgf=1.0, ky=0.1, polarization="s",
                       dx=2.0)

    # Mesaj ÖLÇÜLMÜŞ SAYILARLA konuşuyor — "geçersiz değer" demiyor.
    with pytest.raises(InvalidInputError) as ex:
        dx_kapisi(0.5)
    metin = str(ex.value)
    for parca in ("wavelength/dx", "BIT BIT", "79.984570", "19.937834",
                  "grid.dx_kapisi"):
        assert parca in metin, f"kapı mesajında {parca!r} yok"

    # Δx = 1 geçiyor. Kapı SAF: bir değer döndürmez, girdiyi de DEĞİŞTİRMEZ.
    assert dx_kapisi(1.0) is None
    assert dx_kapisi(1) is None and dx_kapisi(np.float64(HUCRE_DX)) is None
    assert Window(cells=64, dx=1, courant=S, steps=10, wavelength=8.0,
                  npml=0, sgf=1.0, window_fn="dikdortgen").dx == 1


def test_KAYNAGIN_SAATI_TAM_OLARAK_n_CARPI_S():
    """SINIF: TRANSKRİPSİYON DENETİMİ — kaldırılan `· Δx` çarpanının kilidi.

    `Δx ≡ 1` olduğu için `t = n·S·Δx` ile `t = n·S` AYNI SAYIYI verir, ama
    ÖLÜ bir çarpan bırakmak kapatılmış bir düğmeye 1 ile çarpan bir kod yolu
    bırakırdı. Çarpan kaldırıldı ve saatin KENDİSİ burada okunuyor: kaynağa
    verilen `t` dizisi `[n·S for n in range(steps)]`e BİT BİT eşit.

    Üç koşum yolunda da ölçülüyor; eğikte `S` = 0.4, ötekilerde 0.5.
    """
    def kaydeden():
        gelen: list[float] = []

        def f(t):
            gelen.append(t)
            return 0.0

        return gelen, f

    g1, f1 = kaydeden()
    run_1d(Medium.vacuum(64), courant=S, boundary=PEC(), steps=40, source=f1,
           source_index=20, probes=(30,), wavelength=8.0, sgf=1.0)
    assert g1 == [n * S for n in range(40)]

    g2, f2 = kaydeden()
    run_2d(Medium.uniform_2d(9, 9, eps_r=1.0), courant=S, boundary=PEC(),
           steps=40, source=f2, source_index=(4, 4), probes=((3, 3),),
           wavelength=8.0, sgf=1.0)
    assert g2 == [n * S for n in range(40)]

    g3, f3 = kaydeden()
    run_oblique_1d(Medium.vacuum(64), courant=0.4, boundary=PEC(), steps=40,
                   source=f3, source_index=20, probes=(30,), wavelength=8.0,
                   sgf=1.0, ky=0.1, polarization="s")
    assert g3 == [n * 0.4 for n in range(40)]


# ---------------------------------------------------------------------------
# 4. GERİYE DÖNÜK UYUM — Δx = 1 çağrıları BİT BİT değişmedi
# ---------------------------------------------------------------------------
def test_dx_1_CAGRILARI_BIT_BIT_DEGISMEDI():
    """SINIF: TRANSKRİPSİYON DENETİMİ — REGRESYON KİLİDİ.

    Aşağıdaki onaltılık sayılar `cda2871`de (kapı KONULMADAN önce) ölçüldü
    ve buraya YAPIŞTIRILDI. Onaltılık yazım bir süs değil: `repr` bir
    yuvarlamayı gizleyebilir, `float.hex()` gizleyemez — bu bir "yakın"
    testi değil BİT testidir.

    PENCERELER:
      1B    vakum N = 64, PEC, S = 0.5, 120 adım, `ricker` W = 12 sgf = 1.5,
            kaynak 20, problar 30/45
      2B    vakum 21×21, PEC, S = 0.5, 80 adım, `ricker` W = 10 sgf = 1.5,
            kaynak (7, 8), problar (12, 13)/(15, 9)
      eğik  vakum N = 64, PEC, S = 0.4, 100 adım, `gauss_cos` W = 16
            sgf = 2 t0 = 32, ky = 0.5, s-kutuplanma, kaynak 20, problar 30/45
    """
    r1 = run_1d(Medium.vacuum(64), courant=0.5, boundary=PEC(), steps=120,
                source=lambda t: float(ricker(t, wavelength=12.0, sgf=1.5)),
                source_index=20, probes=(30, 45), wavelength=12.0, sgf=1.5)
    assert float(np.sum(r1.probes)).hex() == "0x1.976bb20bb49bbp-4"
    assert float(np.sum(np.abs(r1.probes))).hex() == "0x1.b6083d6c3f25ap+3"
    assert [float(x).hex() for x in r1.final.e[-4:]] == [
        "0x1.2a08463212421p-1", "0x1.37e98bf12228ep-1",
        "0x1.89773bb374669p-2", "0x0.0p+0",
    ]
    assert float(np.sum(np.abs(r1.final.e))).hex() == "0x1.7e7d5e45940a6p+2"

    r2 = run_2d(Medium.uniform_2d(21, 21, eps_r=1.0), courant=0.5,
                boundary=PEC(), steps=80,
                source=lambda t: float(ricker(t, wavelength=10.0, sgf=1.5)),
                source_index=(7, 8), probes=((12, 13), (15, 9)),
                wavelength=10.0, sgf=1.5)
    assert float(np.sum(r2.probes)).hex() == "-0x1.8730f56069e5ep-2"
    assert float(np.sum(np.abs(r2.probes))).hex() == "0x1.613a9846b1248p+1"
    assert float(np.sum(r2.final.e)).hex() == "-0x1.5e7140569bf8cp-1"
    assert float(np.sum(np.abs(r2.final.e))).hex() == "0x1.658e4488504a4p+3"

    r3 = run_oblique_1d(
        Medium.vacuum(64), courant=0.4, boundary=PEC(), steps=100,
        source=lambda t: float(gauss_cos(t, wavelength=16.0, sgf=2.0,
                                         t0=32.0)),
        source_index=20, probes=(30, 45), wavelength=16.0, sgf=2.0,
        ky=0.5, polarization="s",
    )
    assert float(np.sum(r3.probes.real)).hex() == "-0x1.fd181e0ace9a4p-3"
    assert float(np.sum(np.abs(r3.probes.real))).hex() == "0x1.50c6dd639b184p+3"
    assert not np.any(r3.probes.imag)

    # Pencereler de aynen: `dx` hâlâ taşınıyor ve 1.
    assert (r1.window.dx, r2.window.dx, r3.window.dx) == (1.0, 1.0, 1.0)
    assert (r1.window.wavelength, r2.window.wavelength) == (12.0, 10.0)


# ---------------------------------------------------------------------------
# 5. KAZANÇ — "lambda ... hucre" artık SINANABİLİR
# ---------------------------------------------------------------------------
def test_LAMBDA_HUCRE_BEYANI_ARTIK_BIR_TEOREM():
    """SINIF: ŞEMA ÇIPASI — ve `Δx` kapanınca İLK KEZ sınanabilen iddia.

    `Window.as_lines()` "lambda : W hucre" basar. `Δx ≠ 1` açıkken bu satır
    tam olarak `1/Δx` kadar YALANDI (ölçüldü: W = 40 beyanıyla ızgaradaki
    dalga boyu Δx = 1/0.5/2 için 39.969098/79.984570/19.937834 hücre, pencere
    üçünde de "40 hucre" yazarken). Kapı kapanınca satır bir TEOREM oldu ve
    sınavı KURULABİLİR hâle geldi.

    SINAV DAİRESEL DEĞİL: ızgaradaki dalga boyu iki probun FAZ FARKINDAN
    okunuyor, ve karşılaştırıldığı şey Yee'nin KAPALI FORM dispersiyonudur
    (koddan değil, şemadan türer):

        sin(ω·dt/2) = S·sin(k·dx/2),  ω = 2π/W,  dt = S,  dx = 1
        =>  λ_num = π / arcsin( sin(π·S/λ_h) / S )

    PENCERE: 1B vakum N = 3400, PML 40, S = 0.5, kaynak 1200, problar
    1800/1807, `gauss_cos` sgf = 2, t0 = 5·sgf·W, dikdörtgen FFT; kayıt
    uzunluğu `m` tam periyot (bin TAM oturur). ÖLÇÜLDÜ:

        λ_h   m   adım  ölçülen λ_num  kapalı form    bağıl fark
        20    50  2000  19.9378338218  19.9378337269  4.760e-09
        40    36  2880  39.9690974472  39.9690978589  1.030e-08
        80    28  4480  79.9845695207  79.9845713060  2.232e-08

    DÜRÜST OLUMSUZ — beyan SÜREKLİ ORTAM dalga boyudur, sayısalı DEĞİL:
    λ_h = 40'ta beyan/ölçülen = 1.000773, yani ızgaradaki dalga %0.077 daha
    KISA. Bu bir kusur değil ŞEMANIN KENDİ DİSPERSİYONU ve λ_h ile küçülüyor
    (oranlar aşağıda kilitli). "lambda ... hucre" satırı bu farkı TAŞIMAZ ve
    taşımadığı BURADA yazılıdır.
    """
    med, bnd = Medium.vacuum(3400), PML(thickness=40)
    src, p1, p2 = 1200, 1800, 1807
    di = p2 - p1

    def kapali(lam_h: float) -> float:
        return float(np.pi / np.arcsin(np.sin(np.pi * S / lam_h) / S))

    beklenen = {
        20.0: (50, 19.9378338218, 19.9378337269),
        40.0: (36, 39.9690974472, 39.9690978589),
        80.0: (28, 79.9845695207, 79.9845713060),
    }
    oranlar = []
    for lam_h, (m, olculen_bek, kapali_bek) in beklenen.items():
        steps = int(round(m * lam_h / S))
        rec = run_1d(med, courant=S, boundary=bnd, steps=steps,
                     source=_darbe(lam_h), source_index=src,
                     probes=(p1, p2), wavelength=lam_h, sgf=SGF)
        F = np.fft.fft(rec.probes, axis=1)
        dphi = float(np.angle(F[1, m] / F[0, m]))
        lam_num = 2.0 * np.pi / (-dphi / di)

        assert lam_num == pytest.approx(olculen_bek, rel=1e-9)
        # rel = 1e-11: yukarıdaki tablo 10 ONDALIĞA yuvarlanmış yazıldı,
        # yani karşılaştırmanın tabanı yuvarlamadır (±5e-11).
        assert kapali(lam_h) == pytest.approx(kapali_bek, rel=1e-11)
        # ÇIPA: ölçülen ile KAPALI FORM 1e-7'nin altında uyuşuyor.
        assert lam_num == pytest.approx(kapali(lam_h), rel=1e-7)
        # BEYAN: pencerenin bastığı satır ölçülen hücre sayısını taşıyor.
        satir = [x for x in rec.window.as_lines() if x.startswith("lambda")]
        assert satir == [f"lambda      : {lam_h:g} hucre"]
        assert lam_num == pytest.approx(lam_h, rel=4e-3)
        oranlar.append(lam_h / lam_num)

    # DÜRÜST OLUMSUZ: sapma λ_h ile MONOTON küçülüyor (şema dispersiyonu).
    assert oranlar[0] > oranlar[1] > oranlar[2] > 1.0


# ---------------------------------------------------------------------------
# 6. ÇÜRÜTÜLMÜŞ ETİKET — `ky dx`
# ---------------------------------------------------------------------------
def test_EGIK_ky_RADYAN_BOLU_HUCRE_ve_ky_dx_ETIKETI_CURUDU():
    """SINIF: TRANSKRİPSİYON DENETİMİ — sevk edilmiş bir ETİKETİN çürütülmesi.

    `oblique_coefficients` dokstringi kararlılık tablosunun sütununu bir
    zamanlar *"ky dx"* diye etiketliyordu. `Δx` kapatılırken bu etiket İLK
    KEZ sınanabilir hâle geldi ve ÇÜRÜDÜ: `Δx` bu terime HİÇ girmez.

    YAPISAL (ölçüm değil, SAYIM): `oblique_coefficients`in imzasında `dx`
    YOKTUR; enine terim `cb·(i·ky)·aux` ve `cb = S/eps`. Fiziksel karşılığı
    `(dt/eps)·i·k_y` olduğundan `ky = k_y·Δx`, yani ARGÜMANIN KENDİSİ
    boyutsuz gruptur — radyan BÖLÜ HÜCRE.

    ÖLÇÜLEN kısım `tests/em/test_arayuz.py`de duruyor (eşik ikili aramayla
    ölçülüp `1/sqrt(1 + (ky/2)²)` ile karşılaştırılıyor; oran 1.00008).
    İkinci bir pencerede (m = 8 duran kip tohumu, N = 201, 800 adım, büyüme
    ölçütü 100×, tol 1e-7) `ky` = 0.4 için eşik Δx = 1'de de Δx = 4'te de
    0.98092060 çıktı; `ky·Δx` okuması Δx = 4'te 0.78086881 öngörürdü, yani
    %26 sapma. `ky` = 0.8'de sayılar 0.92879012 / 0.92847669 / 0.52999894.

    Kilidin GÖVDESİ zaten doğruyu yazıyordu — bayat olan ETİKETTİ. Bu
    depoda ALTINCI kez görülen desen: hüküm dokstringde eskiyor, testte
    eskimiyor.
    """
    assert "dx" not in inspect.signature(oblique_coefficients).parameters
    assert "dx" not in inspect.signature(oblique_step).parameters

    metin = oblique_coefficients.__doc__ or ""
    assert "RADYAN BOLU HUCRE" in metin
    assert "(ky/2)^2" in metin
    # Tablo başlığı artık `ky`, ve eski etiket YALNIZ çürütülmüş hâliyle var.
    # (`__doc__` Python 3.13'ten beri DEDENT edilmiş gelir; girinti aranmaz.)
    assert "ky         0.0      0.2" in metin
    assert metin.count("ky dx") == 1 and '*"ky dx"*' in metin

    # Kapalı formun İKİ okuması SAYIYLA ayrışıyor.
    ky = 0.4
    hucre = 1.0 / np.sqrt(1.0 + (ky / 2.0) ** 2)
    uzunluk = 1.0 / np.sqrt(1.0 + (ky * 4.0 / 2.0) ** 2)
    assert hucre == pytest.approx(0.98058068, abs=5e-9)
    assert uzunluk == pytest.approx(0.78086881, abs=5e-9)
    assert abs(0.98092060 - hucre) / hucre < 4e-4
    assert abs(0.98092060 - uzunluk) / uzunluk > 0.25
