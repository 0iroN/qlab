"""Ussel radyal yasa ve iptalsiz kosinus kuyrugu -- spec md-tb ss-sigma G1.

Kapsam: spec 4.2'deki M2 (yasanin degerleri) ve M3'un YASA kismi (C^1 kesme,
kapanan merdiven). M3'un dimer kismi (Mermin dolumlu dimerde
dE(1e-3)/dE(1e-4) = 100.04) `tek_orbital_mermin` gerektirir ve G2'de yazilir.

ETIKET: buradaki her test MAKINE testidir (uygulama tutarliligi). Bu kenarda
DOGA CIPASI YOKTUR: yasa turetilmedi, secildi (spec B4). Beklenen degerler
yasanin kendi kapali bicimidir; fizik iddiasi degildir.

Ortak pencere (spec basi): t rolu genlik -0.1, d0 2, us 2; itme rolu genlik
0.05, d0 2, us 6; r_on 3.0, r_cut 3.8.

ARAC TUZAKLARI (spec 4.1): `assert_allclose` (rtol=1e-7 varsayilani) ve
`pytest.approx` (abs ile rel'in BUYUGU) kullanilmaz; her karsilastirma acik
`abs(x - y) < tol` bicimindedir ve her tolerans olculmus tabana baglidir.

BAGLAYICILIK OLCULDU (24.09.2026): her tolerans en az bir mutantla kirildi.
Mutantlar `src/` agacinin bir kopyasinda `radyal.py`ye uygulanip bu dosya
`PYTHONPATH=<kopya>/src` ile kosturuldu; hangi mutantin hangi testi kirdigi
ilgili testin dokstringinde yazili.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

import qlab
import qlab.engine.tb.radyal as radyal_modulu
from qlab.engine.tb.radyal import UsselRadyalYasa, kosinus_kuyrugu
from qlab.errors import InvalidInputError

R_ON = 3.0
R_CUT = 3.8
W = R_CUT - R_ON

SS_SIGMA = UsselRadyalYasa(
    genlik_hartree=-0.1, d0_bohr=2.0, us=2.0, r_on_bohr=R_ON, r_cut_bohr=R_CUT
)
ITME = UsselRadyalYasa(
    genlik_hartree=0.05, d0_bohr=2.0, us=6.0, r_on_bohr=R_ON, r_cut_bohr=R_CUT
)
ROLLER = [pytest.param(SS_SIGMA, id="ss_sigma"), pytest.param(ITME, id="itme")]

# Kilitler KOSAN kodu okur, sabit bir depo yolunu degil. OLCULDU (24.09.2026):
# ilk surumde yol `Path(__file__).parents[2] / "src"` idi ve `import qlab.units`
# ile `class SlaterKosterSatiri` mutantlari (src kopyasinda, PYTHONPATH ile)
# 60/60 GECTI -- kilit, yuklenen modulu degil depodaki dosyayi okuyordu.
KAYNAK = Path(qlab.__file__).resolve().parent
RADYAL_DOSYASI = Path(radyal_modulu.__file__).resolve()

# Cekirdek bolge (fc = 1) noktalari; r_on'un KENDISI dahil.
CEKIRDEK = (0.0, 0.3, 0.7, 1.2, 1.9, 2.0, 2.1, 2.6, 2.999, 3.0)


def _g(yasa: UsselRadyalYasa, d: float) -> float:
    """Kuyruksuz ussel parca, `math.exp` ile BAGIMSIZ hesap."""
    return yasa.genlik_hartree * math.exp(-yasa.us * (d / yasa.d0_bohr - 1.0))


def _f(yasa: UsselRadyalYasa, d: float) -> float:
    return float(yasa.deger_ve_turev(d)[0])


def _df(yasa: UsselRadyalYasa, d: float) -> float:
    return float(yasa.deger_ve_turev(d)[1])


# --------------------------------------------------------------------- M2 --


@pytest.mark.parametrize("yasa", ROLLER)
def test_M2_cekirdekte_deger_USSEL_YASANIN_KENDISI(yasa):
    """d <= r_on'da f(d) = genlik * exp(-us (d/d0 - 1)), fc = 1.

    Olculen: iki rolde de goreli fark 0.0 (10 nokta). Tolerans 1e-12:
    esdeger yeniden yazimlarin birkac ulp'lik farkina izin verir (KONTROL,
    olculdu: exp(-us d/d0 + us) biciminde yazilmis kopya bu dosyada 60/60
    geciyor). KIRMA (olculdu): us -> 1.3 us mutanti (tutarli ama yanlis
    yasa; M1'in kor kaldigi mutant) ilk noktada (d = 0) bu testi kiriyor.
    """
    for d in CEKIRDEK:
        beklenen = _g(yasa, d)
        assert abs(_f(yasa, d) / beklenen - 1.0) < 1e-12, d


@pytest.mark.parametrize("yasa", ROLLER)
def test_M2_cekirdekte_LOGARITMIK_TUREV_eksi_us_bolu_d0(yasa):
    """d <= r_on'da f'/f = -us/d0.

    Olculen: goreli fark ss_sigma'da 0.0, itmede 2.2e-16. Tolerans 1e-12.
    KIRMA (olculdu): us -> 1.3 us mutanti (spec M2'de "fikir, olculmedi"
    diye gecen kanit) ve turev x1.1 mutanti bu testi kiriyor.
    """
    beklenen = -yasa.us / yasa.d0_bohr
    for d in CEKIRDEK:
        oran = _df(yasa, d) / _f(yasa, d)
        assert abs(oran / beklenen - 1.0) < 1e-12, d


@pytest.mark.parametrize("yasa", ROLLER)
def test_M2_kesmede_ve_OTESINDE_bit_bit_SIFIR(yasa):
    """f(r_cut) = 0 ve f(d > r_cut) = 0; turev de. Karsilastirma TAM 0.0."""
    for d in (R_CUT, np.nextafter(R_CUT, np.inf), R_CUT + 1e-12, 3.9, 10.0, 1e3):
        f, df = yasa.deger_ve_turev(d)
        assert float(f) == 0.0, d
        assert float(df) == 0.0, d


def test_M2_IPTALSIZ_bicim_r_cut_ALTINDA_sifir_DEGIL():
    """fc(r_cut - 1e-9) != 0 ve kapali bicime goreli olarak dogru.

    Iptalli bicim 0.5(1 + cos(pi (d - r_on)/w)) OLCULDU: e = 1e-9'da TAM 0.0
    (gercek 3.855315e-18), e = 1e-8'de goreli hata 7.90e-03, e = 1e-6'da
    2.40e-06. Referans sin^2(u)'nun u^6'ya kadar Taylor acilimi,
    u = pi e / (2 w); e = r_cut - d kayan noktada TAM (Sterbenz).
    Olculen (bu bicim): en kotu goreli 3.3e-16. Tolerans 1e-12.
    Dongu BUYUK e'den baslar ki toleransin kendisi sinansin, yalniz
    `!= 0` satiri degil. KIRMA (olculdu): iptalli bicim mutanti ilk noktada
    (e = 1e-6, goreli hata 2.40e-06) tolerans satirinda kiriliyor.
    """
    for e in (1e-6, 1e-7, 1e-8, 1e-9):
        d = R_CUT - e
        e_tam = R_CUT - d
        u = math.pi * e_tam / (2.0 * W)
        referans = u * u * (1.0 - u * u / 3.0 + 2.0 * u**4 / 45.0)
        fc, _ = kosinus_kuyrugu(d, R_ON, R_CUT)
        assert float(fc) != 0.0, e
        goreli = abs(float(fc) / referans - 1.0)
        assert goreli < 1e-12, (e, goreli)
    # Yasanin kendisi de r_cut'in hemen altinda sifir DEGIL.
    assert _f(SS_SIGMA, R_CUT - 1e-9) != 0.0
    assert _f(ITME, R_CUT - 1e-9) != 0.0


@pytest.mark.parametrize("yasa", ROLLER)
def test_TUREV_degerin_merkezi_farkiyla_TUTARLI_dikisten_UZAKTA(yasa):
    """Analitik f' = FD(f), cekirdek, kuyruk ve kesme otesi, h = 1e-5.

    Test kendi on kosulunu dogrular: her nokta dikislerden >= 1e-2 uzakta
    (B3: dikisi kapsayan merkezi fark O(h)). Hata max|f'| ile olceklenir.
    Olculen (h = 1e-5): ss_sigma 3.8e-11, itme 1.7e-10 (goreli). h tarandi:
    1e-4'te 3.8e-9 / 1.5e-8 (kesme hatasi), 1e-6'da 8.2e-11 / 3.5e-11,
    1e-7'de 7.3e-10 / 2.2e-10 (yuvarlama). Tolerans 1e-9, en kotuye 5.8x pay.
    KIRMA (olculdu, ss_sigma / itme, max|f'| ile olceklenmis hata): dfc'yi
    unutan mutant 1.1e-1 / 1.4e-4; dfc isaretini ceviren 2.2e-1 / 2.8e-4;
    turevi x1.1 yapan 9.1e-2 / 9.1e-2. Ucu de bu testi kiriyor.
    """
    noktalar = np.concatenate(
        [np.linspace(0.5, 2.99, 9), np.linspace(3.01, 3.79, 17), np.linspace(3.81, 6.0, 5)]
    )
    h = 1e-5
    dikise_uzaklik = np.minimum(np.abs(noktalar - R_ON), np.abs(noktalar - R_CUT))
    assert float(dikise_uzaklik.min()) >= 1e-2 - 1e-12
    assert np.count_nonzero((noktalar > R_ON) & (noktalar < R_CUT)) == 17
    _, turev = yasa.deger_ve_turev(noktalar)
    arti, _ = yasa.deger_ve_turev(noktalar + h)
    eksi, _ = yasa.deger_ve_turev(noktalar - h)
    fd = (arti - eksi) / (2.0 * h)
    olcek = float(np.max(np.abs(turev)))
    assert olcek > 0.0
    assert float(np.max(np.abs(turev - fd))) / olcek < 1e-9


def test_us_SIFIR_ise_cekirdekte_ve_otede_turev_bit_bit_SIFIR():
    """us = 0 kabul edilir ve kuyruk disinda f' TAM 0.0 olur.

    M10'un t' = 0 kolunun (bant kuvveti bit bit 0.0) yasa tarafindaki
    temeli budur: k = 0 ve kuyrukta cift yokken turev hicbir yerde sizmaz.
    """
    sabit = UsselRadyalYasa(-0.1, 2.0, 0.0, R_ON, R_CUT)
    for d in (0.0, 1.0, 2.0, 2.999, R_ON, R_CUT, 5.0):
        assert _df(sabit, d) == 0.0, d
    for d in (0.0, 1.0, 2.0, R_ON):
        assert _f(sabit, d) == -0.1, d
    # kuyrukta ise turev SIFIR DEGIL (kontrol: test bos gecmiyor)
    assert _df(sabit, 3.4) != 0.0


# --------------------------------------------------------------------- M3 --


def _tek_yanli_fark(yasa: UsselRadyalYasa, x0: float, h: float) -> float:
    """Ileri ve geri tek yanli turevin farki: (f(x0+h)-f(x0))/h - (f(x0)-f(x0-h))/h.

    C^0 kirikta h -> 0'da sabite (f'(x0+) - f'(x0-)) gider, oran 1 verir.
    C^1'de h ile DOGRUSAL iner (~ h (f''+ + f''-)/2), oran 10 verir.
    """
    f0 = _f(yasa, x0)
    return (_f(yasa, x0 + h) - f0) / h - (f0 - _f(yasa, x0 - h)) / h


@pytest.mark.parametrize("yasa", ROLLER)
@pytest.mark.parametrize("x0", [R_ON, R_CUT], ids=["r_on", "r_cut"])
def test_M3_C1_tek_yanli_turev_farki_h_ile_DOGRUSAL_iner(yasa, x0):
    """Oran fark(1e-4)/fark(1e-5) in [9.9, 10.1].

    Olculen: ss_sigma r_on 9.998784, r_cut 10.000900; itme r_on 10.002023,
    r_cut 10.002700 (spec: 9.999 / 10.001). Pencere f''+ + f''- != 0 ister;
    burada fark(1e-5) ss_sigma'da +1.05e-06 / -6.37e-07, itmede +1.28e-07 /
    +8.71e-09 ve yuvarlama tabani 4 eps |f| / h <= ~2e-12, yani oran iyi
    kosullu.
    KIRMA (olculdu): C^0 (dogrusal) kuyruk mutanti dort hucrede de oran
    0.99908-1.00027 verip kiriyor (spec M3'te "fikir, olculmedi" diye gecen
    kanit); sert kesme r_cut'ta 0.10001 / 0.10003 veriyor. Tolerans C^2'ye
    de kor degil: quintic (C^2) kuyrukta f''(r_cut-) = 0 oldugu icin fark
    h^2 ile iner ve r_cut'ta oran 95.60 / 95.62 cikip kiriyor.
    """
    oran = _tek_yanli_fark(yasa, x0, 1e-4) / _tek_yanli_fark(yasa, x0, 1e-5)
    assert 9.9 <= oran <= 10.1, oran


@pytest.mark.parametrize("yasa", ROLLER)
def test_M3_yasanin_MERDIVENI_kapanir_dE_eps_kare_ile_iner(yasa):
    """dE(eps) = f(r_cut - eps) - f(r_cut + eps); oran dE(1e-3)/dE(1e-4) in [90, 110].

    Depo bugun (`uniform_model`, sert kesme) dimerde sabit 0.2000 sicrama
    verir (2|t|, girdiden geri okuma); burada basamak eps^2 ile kapanir.
    Olculen: ss_sigma 100.0899, itme 100.2702 (kT = 0 dimer 2t + itme:
    100.0887; spec'in Mermin dimeri kT = 1e-3'te 100.04 -- o G2'de).
    KIRMA (olculdu, ss_sigma / itme): C^0 kuyruk 10.009 / 10.027, sert
    kesme 1.0009 / 1.0027, quintic (C^2) kuyruk 999.2 / 1001.0.
    """
    dE = [_f(yasa, R_CUT - e) - _f(yasa, R_CUT + e) for e in (1e-3, 1e-4)]
    oran = dE[0] / dE[1]
    assert 90.0 <= oran <= 110.0, oran


@pytest.mark.parametrize("yasa", ROLLER)
@pytest.mark.parametrize("x0", [R_ON, R_CUT], ids=["r_on", "r_cut"])
def test_C2_DEGIL_ikinci_turev_dikiste_SICRAR(yasa, x0):
    """f''(x0+) - f''(x0-) = -g(x0) pi^2/(2 w^2) -- spec B3'un yasa tarafi.

    Sicrama analitik turevin tek yanli farkiyla okunur, h = 1e-5 (O(h)).
    Beklenen deger yasanin kapali bicimidir (MAKINE; girdiden geri okuma,
    fizik iddiasi degil). kT = 0 dimerde (2t + itme) bu r_on'da +0.54812,
    r_cut'ta +0.25317 eder (spec B3). Olculen goreli hata: ss_sigma
    1.4e-5 / 1.5e-5, itme 1.0e-5 / 4.5e-5 (r_on / r_cut). Tolerans 2e-4.
    KIRMA (olculdu): quintic (C^2) kuyruk sicramayi O(h) artigina indiriyor
    (ss_sigma r_on'da 2.2e-5, beklenen 0.2837); kubik smoothstep kuyruk (C^1
    ama baska egrilik) ss_sigma r_on'da 0.34488 veriyor, beklenenin 1.216
    kati. Ikisi de dort hucrede bu testi kiriyor.
    """
    h = 1e-5
    t0 = _df(yasa, x0)
    sicrama = (_df(yasa, x0 + h) - t0) / h - (t0 - _df(yasa, x0 - h)) / h
    beklenen = -_g(yasa, x0) * math.pi**2 / (2.0 * W * W)
    assert abs(sicrama / beklenen - 1.0) < 2e-4, (sicrama, beklenen)


# ------------------------------------------------------ sekil ve sozlesme --


def test_SEKIL_korunur_skaler_sifir_boyutlu_dizi_doner():
    fc, dfc = kosinus_kuyrugu(3.4, R_ON, R_CUT)
    assert isinstance(fc, np.ndarray) and fc.shape == () and dfc.shape == ()
    d = np.linspace(0.0, 5.0, 12).reshape(3, 4)
    f, df = SS_SIGMA.deger_ve_turev(d)
    assert f.shape == (3, 4) and df.shape == (3, 4)
    # dizi hesabi tek tek skaler hesapla ayni (vektorlestirme hatasi yok)
    for i in range(3):
        for j in range(4):
            assert f[i, j] == _f(SS_SIGMA, float(d[i, j]))
            assert df[i, j] == _df(SS_SIGMA, float(d[i, j]))


def test_SIFIR_mesafede_SONLU_sert_cekirdek_YOK():
    """Spec B9: ussel yasa d = 0'da sonlu; cakisma kapisi cagri yerinin isi."""
    assert _f(SS_SIGMA, 0.0) == _g(SS_SIGMA, 0.0)
    assert math.isfinite(_f(ITME, 0.0))


def test_YASA_DONDURULMUS():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SS_SIGMA.us = 3.0  # type: ignore[misc]


# -------------------------------------------------------- gecersiz girdi --

_GECERLI = dict(genlik_hartree=-0.1, d0_bohr=2.0, us=2.0, r_on_bohr=3.0, r_cut_bohr=3.8)


@pytest.mark.parametrize("alan", sorted(_GECERLI))
@pytest.mark.parametrize("deger", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_SONLU_OLMAYAN_parametre_REDDEDILIR(alan, deger):
    with pytest.raises(InvalidInputError):
        UsselRadyalYasa(**{**_GECERLI, alan: deger})


@pytest.mark.parametrize(
    ("alan", "deger"),
    [
        ("d0_bohr", 0.0),
        ("d0_bohr", -2.0),
        ("us", -0.5),
        ("us", -1e-300),
        ("r_on_bohr", 3.8),  # r_on == r_cut
        ("r_on_bohr", 4.0),  # r_on > r_cut
        ("r_cut_bohr", 2.0),
        ("d0_bohr", np.array([2.0, 2.0])),
        ("us", "2.0"),
        ("genlik_hartree", None),
    ],
)
def test_GECERSIZ_parametre_REDDEDILIR(alan, deger):
    with pytest.raises(InvalidInputError):
        UsselRadyalYasa(**{**_GECERLI, alan: deger})


def test_SINIRDA_gecerli_olanlar_KABUL_edilir():
    """us = 0, genlik = 0, negatif r_on: spec'in sozlesmesi yalniz
    sonlu / d0 > 0 / us >= 0 / r_on < r_cut ister."""
    UsselRadyalYasa(**{**_GECERLI, "us": 0.0})
    UsselRadyalYasa(**{**_GECERLI, "genlik_hartree": 0.0})
    UsselRadyalYasa(**{**_GECERLI, "r_on_bohr": -1.0})
    yasa = UsselRadyalYasa(**{**_GECERLI, "d0_bohr": np.float64(2.0)})
    assert type(yasa.d0_bohr) is float


@pytest.mark.parametrize(
    "d",
    [math.nan, math.inf, -1e-12, np.array([1.0, math.nan]), np.array([2.0, -3.0])],
    ids=["nan", "inf", "negatif", "dizide-nan", "dizide-negatif"],
)
def test_GECERSIZ_mesafe_REDDEDILIR(d):
    with pytest.raises(InvalidInputError):
        kosinus_kuyrugu(d, R_ON, R_CUT)
    with pytest.raises(InvalidInputError):
        SS_SIGMA.deger_ve_turev(d)


@pytest.mark.parametrize(("r_on", "r_cut"), [(3.8, 3.8), (4.0, 3.8), (math.nan, 3.8)])
def test_kuyrugun_GECERSIZ_sinirlari_REDDEDILIR(r_on, r_cut):
    with pytest.raises(InvalidInputError):
        kosinus_kuyrugu(3.4, r_on, r_cut)


def test_TASAN_ussel_yasa_sessizce_inf_DONMEZ():
    """us = 800, d = 0: exp(800) tasar; inf/nan normal sonuc gibi donmez."""
    yasa = UsselRadyalYasa(1.0, 1.0, 800.0, 1.0, 2.0)
    with pytest.raises(InvalidInputError):
        yasa.deger_ve_turev(0.0)
    with pytest.raises(InvalidInputError):
        UsselRadyalYasa(0.0, 1.0, 800.0, 1.0, 2.0).deger_ve_turev(0.0)
    # kontrol: ayni yasa tasmayan mesafede calisiyor
    assert math.isfinite(_f(yasa, 1.5))


def test_TUREV_tasmasi_deger_SONLUYKEN_de_REDDEDILIR():
    """Tasma kapisinin 'turev' yarisi: deger sonlu, turev -inf -> yine hata.

    Ustteki testin tek penceresi (us = 800) exp'in KENDISINI tasirir ve deger
    de sonlu olmaz; kapinin yalniz degeri denetleyen bicimi orada ayni
    sonucu verir. Iki pencere yalniz TUREVI tasirir: (1) us = 709, d0 = 1,
    d = 0 -- exp(709) = 8.2e307 sonlu (on kosul testte), turev = -709 x
    8.2e307 = -inf; (2) kayan noktanin sinirina dayanmayan pencere: d0 =
    1e-305, us = 10, d = 0 -- deger e^10 = 2.2e4, turev -us/d0 x deger =
    -inf. KIRMA (olculdu, 24.09.2026): kapiyi `np.all(np.isfinite(deger))`e
    indiren mutant ustteki testi ve dosyanin geri kalanini gecer (genis
    kapsamda da yesil), iki yeni hucrenin ikisinde de kirmizi (turev -inf
    sessizce donuyordu).
    """
    assert math.isfinite(math.exp(709.0))
    with pytest.raises(InvalidInputError):
        UsselRadyalYasa(1.0, 1.0, 709.0, 1.0, 2.0).deger_ve_turev(0.0)
    with pytest.raises(InvalidInputError):
        UsselRadyalYasa(1.0, 1e-305, 10.0, 1.0, 2.0).deger_ve_turev(0.0)


# ------------------------------------------------------------- kilitler --


def _importlar(yol: Path) -> set[str]:
    agac = ast.parse(yol.read_text(encoding="utf-8"), filename=str(yol))
    adlar: set[str] = set()
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Import):
            adlar.update(a.name for a in dugum.names)
        elif isinstance(dugum, ast.ImportFrom):
            assert dugum.level == 0, "goreli import tarayicidan kacar"
            adlar.add(str(dugum.module))
    return adlar


def test_radyal_IMPORTLARI_kilitli_UNITS_ve_MOTOR_YOK():
    """Spec 2.1(1) ve 5.2: numpy, dataclasses, qlab.errors -- baska hicbiri.

    `qlab.units` yasak: units ekseninde tb 2'den 3'e cikar (olculdu, spec
    5.2). hf, downfold ve tb.factory de yasak (kenar cagri duzeyinde
    bagimsiz kalmali, spec 2.2). Tam kume esitligi, yani her yeni import
    bilincli bir test degisikligi ister.
    KIRMA (olculdu): `import qlab.units` eklenmis mutant bu testi kiriyor.
    """
    assert _importlar(RADYAL_DOSYASI) == {
        "__future__",
        "dataclasses",
        "numpy",
        "qlab.errors",
    }


def test_hicbir_SINIF_ya_da_FONKSIYON_adinda_SLATERKOSTER_YOK():
    """Spec 1 ve Osman'in karari: yalniz ss-sigma satiri var, SK modeli yok.

    `src/qlab` altindaki TUM sinif ve fonksiyon adlari (ic ice olanlar
    dahil) taranir; buyuk/kucuk harf ve alt cizgi yok sayilir.
    KIRMA (olculdu): radyal.py'ye `class SlaterKosterSatiri` eklenmis mutant
    bu testi kiriyor.
    """
    ihlaller = []
    dosyalar = sorted(KAYNAK.rglob("*.py"))
    assert RADYAL_DOSYASI in dosyalar
    for yol in dosyalar:
        agac = ast.parse(yol.read_text(encoding="utf-8"), filename=str(yol))
        for dugum in ast.walk(agac):
            if isinstance(dugum, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if "slaterkoster" in dugum.name.lower().replace("_", ""):
                    ihlaller.append((yol.name, dugum.lineno, dugum.name))
    assert ihlaller == []
