"""`kind: em` senaryosu, panel beyanı ve uçtan uca koşum.

SINIF: TRANSKRİPSİYON DENETİMİ. Bu dosyanın hiçbir testi yeni bir fizik
iddiası kurmaz; senaryo sınırının (YAML → dataclass → motor →
gözlemlenebilir → panel) doğru bağlandığını ve panelin **dokuz zorunlu
satırı** bastığını ölçer. Modülün fizik çıpaları başka yerdedir: Bessel
oranları `test_bessel.py`de, Rayleigh üssü `test_sacilma.py`de, tek yarık
`test_uzak_alan.py`de, TİY faz farkı `test_arayuz.py`de, şema çıpaları
`test_sema_cipalari.py` ile `test_yee.py`de, ölçütün kendi karnesi
`test_merdiven.py`de.

İKİ İSTİSNA — VE İKİSİ DE FİZİK DEĞİL, KOŞUM ÖZELLİĞİDİR:

1. Sevk edilen senaryonun merdiveninin **oturduğu** ve hangi tamsayıya
   oturduğu. Bu bir Maxwell iddiası değil, "bu pencere kendi mertebesini
   okuyacak kadar geniş mi" sorusunun cevabıdır.
2. **KABA BASAMAĞIN SINIRI** — ve bu, Task 6'nın dokstring denetiminin
   ürünüdür. `test_bessel.py` bir zamanlar *"R = 5'te tepe kapısı kipi
   bulamıyor"* diyordu; ölçüm çürüttü ve düzeltme aşağıda kilitlendi.

BU DOSYANIN EN ÖNEMLİ TESTİ BİR SAYI DEĞİL BİR SAYIM DENETLER:
`test_BEYAN_TEK_KAYNAKTAN_geliyor`. Ölçülmüş gerekçe bu depodadır — aynı
beyan bloğu bir kez `scf.py` ile `tb/factory.py`de birebir kopyalanmıştı ve
biri düzeltilip öteki unutulduğunda panel iki farklı şey iddia ediyordu
(bkz. `tests/hf/test_hf_scenario.py`,
`tests/tb/test_downfold_senaryo.py::test_BEYAN_TEK_KAYNAKTAN_geliyor`,
`tests/spin/test_spin_senaryo.py::test_BEYAN_TEK_KAYNAKTAN_geliyor`).

===========================================================================
ÖLÇÜLEN SAYILAR — HEPSİ 26.08.2026'DA, BU DEPODA, AŞAĞIDAKİ PENCEREDE
===========================================================================

**Sevk edilen senaryo** (`scenarios/em_tarama.yaml`): 2B dairesel PEC boşluk ·
R = 5/10/20/40 (r = 2) · S = 0.5 · adım = 420·R · Ricker
(`sgf` = 1.5), λ = 1.4515225409252954·R · 8 dedektör · blackman · `pad` = 8 ·
tepe tabanı 0.01 · `margin` = 3 · `eps_r` = 1. **FDTD 0.95 s, analiz 0.54 s,
toplam 1.49 s.**

    R      ızgara    adım    oran (ikinci/en düşük)   tepe   gürültü
    5      17 x 17   2100    1.578632448506           28     5.04e-06
    10     27 x 27   4200    1.589868261123           21     7.08e-06
    20     47 x 47   8400    1.592538328891           17     1.95e-06
    40     87 x 87   16800   1.593146492961           15     2.71e-06

    ham mertebeler 2.0732, 2.1343 → p_inf = 2.195532 ± 0.061188 → **OTURDU(2)**
    (κ = 0.5'e pay: |p_inf − 2| + u = 0.2567)

**HÜKÜM ADIM ÇARPANINA DUYARSIZ** — dört çarpanda da OTURDU(2):

    420 → 2.195532 ± 0.061188      1260 → 2.191308 ± 0.058420
    840 → 2.184666 ± 0.054759      1680 → 2.191648 ± 0.058554

**KABA BASAMAĞIN SINIRI (düzeltilmiş hüküm).** R = 5'te tepe kapısı **28
tepe** buluyor ve merdivenin okuduğu sayı (`j11/j01`) **bulunuyor**; düşen
şey `test_bessel.py`nin **%2'lik eşleme penceresidir**:

    j11/j01  1.578632448506   −0.923%   genlik 1.0000   -> eşleşiyor
    j21/j01  2.142996285960   +0.349%   genlik 0.7193   -> eşleşiyor
    j02/j01  2.230737839747   −2.818%   genlik 0.5734   -> DIŞARIDA
    j31/j01  2.567280676154   −3.233%   genlik 0.8215   -> DIŞARIDA
    j12/j01  2.998934231425   +2.798%   genlik 0.2955   -> DIŞARIDA

R = 40'ta beşi de eşleşiyor. Yani *"hane sayısı kipe bağlıdır"* cümlesi bu
senaryonun kendi iki ucundan okunuyor.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import qlab.engine.em.factory as fabrika
from qlab.engine.em.factory import (
    HUKUM_VERILMEDI,
    build_em,
    provenance_for,
)
from qlab.engine.em.grid import Window
from qlab.errors import InvalidInputError, ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import EMScenario
from qlab.viz.em_frames import build_em_payload

PROJE_KOKU = Path(__file__).resolve().parents[2]
SEVK_EDILEN = PROJE_KOKU / "scenarios" / "em_tarama.yaml"
SEVK_YAML = SEVK_EDILEN.read_text(encoding="utf-8")

#: Panelin dokuz zorunlu satırı, SIRASIYLA.
PANEL_SATIRLARI = [
    "Yöntem",
    "Yaklaşımlar",
    "İhmal edilen",
    "Beklenen hata",
    "Dairesellik",
    "Çıpa sınıfı",
    "Pencere",
    "Yakınsama",
    "Yakınsama tanısı",
]

#: Yapısal testlerin kullandığı tanı seti — HİÇ FDTD KOŞMAZ.
UCUZ_PENCERE = Window(
    cells=(17, 17),
    dx=1.0,
    courant=0.5,
    steps=2100,
    wavelength=7.257613,
    npml=0,
    sgf=1.5,
    window_fn="blackman",
)


def _yaz(tmp_path, metin: str) -> Path:
    yol = tmp_path / "em.yaml"
    yol.write_text(metin, encoding="utf-8")
    return yol


def _kaynak_metni(yol: Path) -> str:
    """Dosya metni, BİTİŞİK DİZE SABİTLERİ BİRLEŞTİRİLMİŞ hâlde.

    Gerekçesi ölçülmüş bir tuzak (`tests/tb/test_downfold_senaryo.py`):
    `"birinci parca "\\n"ikinci parca"` kaynakta İKİ parçadır ama çalışma
    zamanında TEK dizedir. Ham metinde aramak 0 eşleşme verirdi ve "kopya
    yok" diye okunurdu — yani test kendi iddiasının tersini kanıtlardı.
    """
    return re.sub(r'"\s*\n\s*"', "", yol.read_text(encoding="utf-8"))


def _tum_beyanlar() -> tuple[str, ...]:
    return (
        fabrika._EM_YAKLASIMLAR
        + fabrika._EM_IHMAL
        + (
            fabrika._EM_BEKLENEN_HATA,
            fabrika._DAIRESELLIK_BEYANI,
            fabrika._CIPA_SINIFI_BEYANI,
            fabrika._PENCERE_BEYANI,
            fabrika._MERDIVEN_TANISI,
            fabrika._RAYLEIGH_TANISI,
            fabrika._YARIK_TANISI,
            fabrika._TIY_TANISI,
            HUKUM_VERILMEDI,
        )
    )


def _canli(tani: str) -> str:
    """`Yakınsama tanısı` satırının BU KOŞUMA ait parçası.

    Satır iki cinsten metin taşır: `[CANLI ...]` bu koşumun kendi merdiveni,
    `[BEYAN ...]` ise başka çıpaların `tests/em/` altında ölçülmüş hükümleri.
    İkisini ayırmadan arama yapmak yanıltıcıdır ve bu bir varsayım değil
    ölçülmüş bir tuzaktır: TİY beyanı zaten `OTURDU(2)` içeriyor, dolayısıyla
    "satırda OTURDU var mı" diye bakan bir test canlı hükmü hiç görmeden
    yeşil kalırdı.
    """
    assert tani.startswith("[CANLI")
    kesim = tani.index("[BEYAN")
    return tani[:kesim]


def _provenance(
    senaryo: EMScenario,
    *,
    p_inf: float | None = 2.195532,
    uncertainty: float | None = 0.061188,
    verdict: str = "OTURDU",
    orders: tuple[float, ...] = (2.0732, 2.1343),
):
    n = len(senaryo.radii)
    return provenance_for(
        senaryo,
        windows=(UCUZ_PENCERE,) * n,
        ratios=(1.5,) * n,
        noise=(1e-6,) * n,
        peaks=(20,) * n,
        orders=orders,
        p_inf=p_inf,
        uncertainty=uncertainty,
        verdict=verdict,
    )


@pytest.fixture(scope="module")
def sevk_senaryo() -> EMScenario:
    return load_scenario(SEVK_EDILEN)


@pytest.fixture(scope="module")
def sevk_yuk(sevk_senaryo):
    """Sevk edilen senaryonun TEK koşumu — altı test tüketir.

    Ölçüldü: FDTD 0.95 s + analiz 0.54 s = **1.49 s**. Fixture maliyeti
    `setup` olarak görünür ve Modül 4'ün dersi gereği `slow` işareti
    `setup+call`un en kötüsüne göre verilir; 1.49 s eşiğin (2.0 s) altında
    olduğu için bu fixture'ı tüketen testler işaretsizdir.
    """
    return build_em_payload(build_em(sevk_senaryo))


# ---------------------------------------------------------------------------
# YAPISAL testler — hiç FDTD koşmazlar, milisaniyedirler.
# ---------------------------------------------------------------------------


def test_SEVK_EDILEN_senaryo_yukleniyor_ve_TOHUM_YOK(sevk_senaryo):
    """Depodaki `scenarios/em_tarama.yaml` gerçekten yükleniyor.

    TOHUMUN YOKLUĞU DA BİR İDDİADIR ve burada kilitlenir: `SpinScenario`
    `seed` taşır ve varsayılanı YOKTUR çünkü Monte Carlo tekrarlanabilir
    olmak zorundadır; burada tohum diye bir alan HİÇ YOKTUR çünkü FDTD
    deterministtir. Tekrarlanabilirliğin karşılığı pencerenin tam beyanıdır
    ve o beyanın her düğmesi YAML'da yazılıdır.
    """
    assert not hasattr(sevk_senaryo, "seed")
    assert sevk_senaryo.radii == (5.0, 10.0, 20.0, 40.0)
    assert sevk_senaryo.ratio == 2.0
    assert sevk_senaryo.courant == 0.5
    assert sevk_senaryo.step_factor == 420.0
    assert sevk_senaryo.wavelength_per_radius == 1.4515225409252954
    assert sevk_senaryo.sgf == 1.5
    assert (sevk_senaryo.detectors, sevk_senaryo.pad) == (8, 8)
    assert sevk_senaryo.window_fn == "blackman"
    assert sevk_senaryo.peak_floor == 0.01
    assert (sevk_senaryo.eps_r, sevk_senaryo.margin) == (1.0, 3)
    # `dx` ve `npml` BIRER ALAN DEGIL -- gerekceleri `EMScenario`da olculmus
    # halde yazili ve asagida ayrica sinaniyor.
    assert not hasattr(sevk_senaryo, "dx")


def test_BESSEL_KOKU_SEVK_EDILEN_YAML_DA_HIC_GECMIYOR():
    """Çıpanın referansı senaryoya SÖYLENMİYOR.

    `spin_tarama.yaml`ın Onsager kilidiyle aynı gerekçe: karşılaştırma
    değeri girdide geçerse, çıkan sayının girdiden okunmadığını iddia etmek
    zorlaşır. `wavelength_per_radius` bir Bessel kökü DEĞİLDİR; kaynağın
    bandını kipin frekansına göre sabit tutan bir pencere düğmesidir ve
    kökün hiçbir basamağını taşımaz.
    """
    for kok in ("2.4048", "3.8317", "5.1356", "5.5200", "6.3801", "7.0155"):
        assert kok not in SEVK_YAML, f"Bessel kökü YAML'da geçiyor: {kok}"
    # Ve panelin okuduğu oranın kendisi de YAML'da yok.
    assert "1.5933" not in SEVK_YAML


def test_BILINMEYEN_kind_listesi_em_iceriyor(tmp_path):
    """Hata mesajı geçerli türleri sayar; `em` orada görünmeli — aksi hâlde
    kullanıcı sevk edilmiş bir türü keşfedemez."""
    with pytest.raises(ScenarioError, match="em") as bilgi:
        load_scenario(_yaz(tmp_path, "kind: karpuz\nname: x\n"))
    assert "karpuz" in str(bilgi.value)


def test_BILINMEYEN_anahtar_ve_EKSIK_anahtar_reddediliyor(tmp_path):
    fazla = SEVK_YAML + "kafes_boyu: 3\n"
    with pytest.raises(ScenarioError, match="kafes_boyu"):
        load_scenario(_yaz(tmp_path, fazla))
    eksik = SEVK_YAML.replace("sgf: 1.5\n", "")
    with pytest.raises(ScenarioError, match="sgf"):
        load_scenario(_yaz(tmp_path, eksik))


@pytest.mark.parametrize("anahtar,deger", [("npml", "15"), ("dx", "0.5")])
def test_BAGLANMAMIS_ve_FAZLADAN_DUGMELER_REDDEDILIYOR(
    tmp_path, anahtar, deger
):
    """`npml` ve `dx` GEÇERLİ ANAHTAR DEĞİLDİR — ve sebepleri FARKLI.

    * **`npml` bağlı değildir:** dairesel PEC boşluk tam yansıtıcıdır ve
      `CircularPEC.thickness` sabit sıfırdır. Kabul edilseydi kullanıcı
      ayarladığını sanırdı (`heisenberg_step` + Ising ile aynı aile).
    * **`dx` bağlıydı ama SESSİZCE YANLIŞ çalışıyordu — VE BU ÖLÇÜLDÜ.**
      `run_2d` kaynağı `t = n·S·Δx` zamanında besliyordu, oysa `wavelength`
      **hücre** cinsinden verilir; ikisi yalnız `Δx = 1`'de örtüşüyordu.
      R = 5/10 ve `step_factor` = 120 penceresinde tepe sayısı `Δx` = 1'de
      **20 / 13**, `Δx` = 0.5'te **4 / 4**, `Δx` = 2'de **31 / 43** çıkıyordu
      — `λ` üçünde de 7.257613 kalarak.

    BU TESTİN ADI BİR ZAMANLAR *"...SESSIZCE_YANLIS_DUGMELER..."* İDİ ve o
    ad, kusur YALNIZ senaryodan çıkarıldığı, motor açık bırakıldığı için
    doğruydu. Bugün `Δx` motorda da kapalı (`engine/em/grid.py::dx_kapisi`,
    ölçümü `tests/em/test_hucre_birimi.py`) ve `Δx` artık "sessizce yanlış
    çalışan" bir düğme değil, HİÇ OLMAYAN bir düğmedir: ölçüldüğünde ayrı
    bir serbestlik derecesi olmadığı, `wavelength`in ölçekli bir kopyası
    olduğu çıktı.

    İkisi de reddediliyor ve panel ikisini de yine basıyor (`npml = 0`,
    `dx = 1`), değerleri `Window`dan gelerek.
    """
    with pytest.raises(ScenarioError, match=anahtar):
        load_scenario(_yaz(tmp_path, SEVK_YAML + f"{anahtar}: {deger}\n"))


@pytest.mark.parametrize(
    "bozuk,desen",
    [
        # merdivenin kendisi
        ("radii: [5.0, 10.0, 20.0, 40.0]", "radii: [5.0, 10.0, 20.0]"),
        ("radii: [5.0, 10.0, 20.0, 40.0]", "radii: [40.0, 20.0, 10.0, 5.0]"),
        ("radii: [5.0, 10.0, 20.0, 40.0]", "radii: [5.0, 10.0, 20.0, 41.0]"),
        ("radii: [5.0, 10.0, 20.0, 40.0]", "radii: [5.0, 10.0, 10.0, 20.0]"),
        ("radii: [5.0, 10.0, 20.0, 40.0]", "radii: [-5.0, 10.0, 20.0, 40.0]"),
        ("radii: [5.0, 10.0, 20.0, 40.0]", "radii: [.nan, 10.0, 20.0, 40.0]"),
        ("radii: [5.0, 10.0, 20.0, 40.0]", "radii: 5.0"),
        ("ratio: 2.0", "ratio: 1.5"),
        ("ratio: 2.0", "ratio: 1.0"),
        ("ratio: 2.0", "ratio: iki"),
        # pencere düğmeleri
        ("courant: 0.5", "courant: 0.9"),
        ("courant: 0.5", "courant: 0.0"),
        ("step_factor: 420.0", "step_factor: 0.0"),
        ("wavelength_per_radius: 1.4515225409252954", "wavelength_per_radius: 0.0"),
        ("sgf: 1.5", "sgf: -1.0"),
        ("detectors: 8", "detectors: 1"),
        ("pad: 8", "pad: 0"),
        ("window_fn: blackman", "window_fn: gauss"),
        ("peak_floor: 0.01", "peak_floor: 0.0"),
        ("peak_floor: 0.01", "peak_floor: 1.0"),
        ("eps_r: 1.0", "eps_r: 0.0"),
        ("margin: 3", "margin: 0"),
    ],
)
def test_BOZUK_senaryolar_REDDEDILIYOR(tmp_path, bozuk, desen):
    """Sessizce varsayılana düşmek YASAK: her biri ScenarioError vermeli.

    Üçü özellikle önemli:

    * **`courant: 0.9`** — 1B'de geçerdi, 2B'de şema PATLAR. 1B kapısı
      devralınamaz ve bu ölçülmüştür (`yee_coefficients_2d` dokstringi).
    * **`radii` üç basamak** — merdiven aritmetiği n−3 Richardson noktası
      verir; üç değerde belirsizlik ölçülemez ve panel yapısal olarak
      YARGILANAMAZ basardı.
    * **`radii: [5, 10, 20, 41]`** — `ratio` ile tutarsız. `p` tam olarak
      `log(d_i/d_{i+1})/log(ratio)` ile okunur; oran uymuyorsa çıkan sayı
      bir MERTEBE DEĞİLDİR ve hiçbir kapı bunu yakalamazdı.
    """
    with pytest.raises(ScenarioError):
        load_scenario(_yaz(tmp_path, SEVK_YAML.replace(bozuk, desen)))


def test_COURANT_SINIRI_MOTORDA_DA_duruyor():
    """Yükleyici kaldırılsa motor hâlâ tutar — ama hata TÜRÜ değişir.

    Bu iki blok o farkı kilitler: yükleyici `ScenarioError`, motor
    `InvalidInputError` fırlatır ve biri ötekinin yerine sessizce geçemez.
    Aynı desen `test_spin_senaryo.py::test_WOLFF_SINIRI_MOTORDA_DA_duruyor`
    ile kardeştir.
    """
    from qlab.engine.em.boundary import CircularPEC
    from qlab.engine.em.medium import Medium
    from qlab.engine.em.yee import yee_coefficients_2d

    with pytest.raises(InvalidInputError, match="1/sqrt"):
        yee_coefficients_2d(
            Medium.uniform_2d(11, 11), courant=0.9, boundary=CircularPEC(radius=4.0)
        )


def _eps_yaml(tmp_path, eps_r: float, S: float) -> Path:
    metin = SEVK_YAML.replace("eps_r: 1.0", f"eps_r: {eps_r}")
    return _yaz(tmp_path, metin.replace("courant: 0.5", f"courant: {S}"))


@pytest.mark.parametrize(
    "eps_r,S", [(0.25, 0.5), (0.4999, 0.5), (0.9999999999999999, 0.7071067811865476)]
)
def test_ETKIN_COURANT_eps_r_1_ALTINDA_NEDENIYLE_reddediliyor(
    tmp_path, capsys, monkeypatch, eps_r, S
):
    """eps_r < 1'de dalga c/√eps_r ile vakumdan HIZLIDIR; sınır S/√eps_r'ye uygulanır.

    ÖLÇÜLDÜ (28.09, kapı yokken, public yollar): eps_r = 0.25, S = 0.5 (etkin
    1.0) YÜKLENİYORDU, `build_em` alanları inf/nan'a taşıyordu (yalnız numpy
    RuntimeWarning) ve CLI rc = 1 ile "iki tepe bulunamadı" diyordu — nedeni
    anmadan. 0.4999 sınırda: etkin 0.70718 > 0.70711. Üç yol (YAML, CLI, Python
    `EMScenario` → `build_em`) artık nedeni ADIYLA söylüyor. Eşiğin YERİ de
    ölçüldü: eps_r 0.25/0.5'te S/√eps_r = 1.001/√2 3000 adımda 3.65e+33
    büyüyor, 0.999/√2 yalnız 1.67 (bkz. `grid.courant_limit`). MESAJ: eps_r ve etkin S TAM
    (repr; `:g` 1 - 1e-16'yı "eps_r = 1 < 1" basardı); kapı hesaptan ÖNCE (`run_2d` yasak);
    önerilen "en çok S" yüklenince GEÇER.
    """
    yaml = _eps_yaml(tmp_path, eps_r, S)
    with pytest.raises(ScenarioError, match="etkin Courant S/√eps_r") as yuk:
        load_scenario(yaml)
    mesaj, etkin = str(yuk.value), S / math.sqrt(eps_r)
    assert f"eps_r = {eps_r!r} < 1" in mesaj and f"S/√eps_r = {etkin!r} 2B" in mesaj
    from qlab.run import main

    assert main([str(yaml), "-o", str(tmp_path / "x.npz")]) == 1
    hata = capsys.readouterr().err
    assert "etkin Courant" in hata and "iki tepe" not in hata
    tekli = dataclasses.replace(
        load_scenario(SEVK_EDILEN), eps_r=eps_r, courant=S, radii=(5.0,)
    )
    monkeypatch.setattr(fabrika, "run_2d", lambda *a, **k: pytest.fail("kapi hesaptan sonra"))
    desen = re.escape(f"eps_r = {eps_r!r} < 1") + ".*" + re.escape(f"(eps_r) = {etkin!r} ")
    with pytest.raises(InvalidInputError, match=desen):
        build_em(tekli)
    oneri = float(re.search(r"en çok S = (\S+) olabilir", mesaj).group(1))
    assert 0.0 < math.sqrt(eps_r / 2.0) - oneri < 1e-5, "oneri siniri asiyor ya da gereksiz kucuk"
    assert load_scenario(_eps_yaml(tmp_path, eps_r, oneri)).courant == oneri


@pytest.mark.parametrize(
    "eps_r,S", [(0.5, 0.5), (0.25, 0.35), (1.0, 0.7071067811865476), (0.25, 0.3535533905932738)]
)
def test_ETKIN_COURANT_sinirin_ICINDE_kabul_ve_SONLU(tmp_path, eps_r, S):
    """Kapı fazla sıkı değil: etkin 0.7071067811865475 (float'ta sınırın TAM
    altı) ve 0.70 geçer, alan sonlu kalır. `√eps_r` yerine `eps_r` yazan bir
    kapı (0.35/0.25 = 1.4) ikinciyi reddederdi. (1.0, 1/√2) YALNIZ yükleyicinin
    ham S kapısını bağlar (fabrika kapısı eps_r >= 1'de susar); (0.25,
    0.3535533905932738)'te √0.25 = 0.5 kesin ve etkin S float'ta TAM sınırdadır:
    yükleyicinin VE fabrikanın (`_courant_kapisi`) `>`'ını bağlar, `>=` yazan
    kapı onu reddederdi. Sevk davranışı bit bit aynı kaldı (28.09 ölçümü: dört
    eps_r >= 1 koşumunun kayıt sha256'ları değişmedi).
    """
    sen = load_scenario(_eps_yaml(tmp_path, eps_r, S))
    assert (sen.eps_r, sen.courant) == (eps_r, S)
    kayit = build_em(dataclasses.replace(sen, radii=(5.0,))).records[0]
    assert np.all(np.isfinite(kayit.probes))


@pytest.mark.parametrize(
    "eps_r,S,eski", [(0.0, 0.5, "adim sayisi 0"), (1.0, 0.9, "yani 1/sqrt")]
)
def test_ETKIN_COURANT_kapisi_ARALIK_DISINDA_eski_HATAYI_EZMEZ(
    sevk_senaryo, eps_r, S, eski
):
    """Python yolunda kapı yalnız 0 < eps_r < 1'de konuşur: eps_r = 0 (adım 0)
    ve eps_r = 1, S = 0.9 (motorun ham kapısı) kapıdan ÖNCEKİ hatayı verir —
    ne √0'a bölüp `ZeroDivisionError`, ne motorun mesajını gasp."""
    with pytest.raises(InvalidInputError, match=eski):
        build_em(
            dataclasses.replace(sevk_senaryo, eps_r=eps_r, courant=S, radii=(5.0,))
        )


def test_BEYAN_TEK_KAYNAKTAN_geliyor():
    """BU DOSYANIN EN ÖNEMLİ TESTİ. Beyan metninin İKİ KOPYASI OLAMAZ.

    Ölçülmüş gerekçe bu depodadır: `scf.py` ile `tb/factory.py` aynı bloğu
    birebir taşıyordu ve biri düzeltilip öteki unutulduğunda panel iki
    farklı şey iddia ediyordu. Kilit iki yönlüdür:

      * `src/` ağacında her cümle tam BİR kez — kaynakta kopya yok. Bu,
        `run.py::_run_em`in beyan metnini kendisi yazmasını da YASAKLAR:
        yazsaydı sayım 2 olurdu. `_run_em` `HUKUM_VERILMEDI`yi bile
        yazmaz, fabrikadan İTHAL EDER.
      * `tests/` ağacında, BU DOSYA hariç, sıfır kez — ikinci bir test
        dosyası açıklama cümlesini elle yeniden yazamaz.

    STANDART TERİM MUAFİYETİ YOK, VE BU ÖLÇÜLDÜ: `test_downfold_senaryo.py`
    dört girdiyi muaf tutmak zorundaydı çünkü onlar cümle değil TERİMDİ.
    Buradaki yirmi girdinin on dokuzu tam cümledir; yirmincisi
    (`HUKUM_VERILMEDI`) tek bir ibaredir ama o da muafiyet gerektirmiyor —
    muafiyetsiz sayım 1/0 tutuyor.

    BU DOSYANIN MUAF OLMASI BİR SAPMA DEĞİL, KİLİDİN KENDİSİDİR: tam cümle
    aramak için cümleyi bir kez yazmak gerekir ve o tek kopya
    `test_MODULUN_ETIK_CEKIRDEGI_metinde_YAZILI`dadır.
    """
    beyanlar = _tum_beyanlar()
    assert len(beyanlar) == 20, "beyan sayisi degisti; testi bilerek guncelleyin"
    bu_dosya = Path(__file__).resolve()

    for cumle in beyanlar:
        src_sayim = sum(
            _kaynak_metni(yol).count(cumle)
            for yol in (PROJE_KOKU / "src").rglob("*.py")
        )
        assert src_sayim == 1, f"src/ agacinda {src_sayim} kopya: {cumle[:60]!r}"

        test_sayim = sum(
            _kaynak_metni(yol).count(cumle)
            for yol in (PROJE_KOKU / "tests").rglob("*.py")
            if yol.resolve() != bu_dosya
        )
        assert test_sayim == 0, f"BASKA bir test dosyasinda kopya var: {cumle[:60]!r}"


#: Bir dize sabitinin "beyan parçası" sayılması için gereken en kısa uzunluk.
#: SEÇİLMEDİ, ÖLÇÜLDÜ: `src/` ağacındaki her dize sabiti beyan cümlelerinin
#: içinde aranarak eşik taraması yapıldı (fabrika hariç) —
#:
#:     eşik  4 -> 207 eşleşme  (" ve ", "p_inf", "Pencere" gibi ortak kelimeler)
#:     eşik  6 ->  22
#:     eşik  8 ->   6
#:     eşik 12 ->   4  (dördü de `YARGILANAMAZ`)
#:     eşik 16 ->   0
#:     eşik 20 ->   0
#:
#: 12'deki dört eşleşme bir kopya DEĞİL: `YARGILANAMAZ` hükmün üç değerinden
#: biridir ve sahibi `observable/em_obs.py::VERDICTS`tir — paylaşılan bir
#: SÖZCÜKTÜR, beyan metni değil. Yanlış pozitif tabanı 12 ile 16 arasındadır;
#: 16 seçildi. Aşağıda öldürülen mutantın kopyaladığı parça 55 karakterdir.
_BEYAN_PARCASI_ESIGI = 16


def test_BEYAN_PARCASININ_KOPYASI_da_YASAK():
    """MUTASYON TURUNUN ÜRÜNÜ — cümle sayımının ölçülmüş DELİĞİ.

    `test_BEYAN_TEK_KAYNAKTAN_geliyor` **tam cümle** sayar ve bu, planın
    "`run.py` beyan metnini kendisi yazıyor" mutantına karşı YETERSİZ çıktı:
    `run.py`'ye `_PENCERE_BEYANI`nin **55 karakterlik bir parçası** enjekte
    edildiğinde mutant **hayatta kaldı** — tam cümlenin `src/` sayımı hâlâ
    1'di, çünkü kopyalanan şey cümlenin tamamı değildi.

    ÖNCE NEDEN DİYE SORULDU, ve cevap testin değil KURALIN eksik yazılmış
    olmasıydı. Korunan şey "aynı cümle iki kez yazılmasın" değil, **panelin
    iki farklı şey iddia edememesi**dir; bir cümlenin yarısını kopyalayıp
    sonra fabrikadaki asıl cümleyi düzeltmek tam olarak o arızayı üretir.
    Bu yüzden kural parça seviyesine indirildi: `src/` ağacında (fabrika
    hariç) hiçbir dize sabiti bir beyan cümlesinin İÇİNDE geçemez.

    Aynı delik Modül 4'ün `test_spin_senaryo.py`sinde de vardı ve o gün
    ölçülmedi; 11.10.2026'da kural oraya taşındı, eşik orada yeniden ölçüldü
    ve `run.py`'ye enjekte edilen parça orada da kırmızı oldu.
    """
    import ast

    beyanlar = _tum_beyanlar()
    fabrika_yolu = (PROJE_KOKU / "src/qlab/engine/em/factory.py").resolve()
    kopyalar = []
    for yol in (PROJE_KOKU / "src").rglob("*.py"):
        if yol.resolve() == fabrika_yolu:
            continue
        agac = ast.parse(yol.read_text(encoding="utf-8"))
        for dugum in ast.walk(agac):
            if not isinstance(dugum, ast.Constant) or not isinstance(dugum.value, str):
                continue
            if len(dugum.value) < _BEYAN_PARCASI_ESIGI:
                continue
            for cumle in beyanlar:
                if dugum.value in cumle:
                    kopyalar.append((yol.name, dugum.lineno, dugum.value[:60]))
    assert kopyalar == [], f"beyan PARCASI kopyalanmis: {kopyalar}"


def test_MODULUN_ETIK_CEKIRDEGI_metinde_YAZILI(sevk_senaryo):
    """Anahtar kelime değil TAM CÜMLE PARÇASI aranır.

    Gerekçesi `test_tb_scenario.py`de ölçüldü: kelime bazlı bir test,
    bitişik dize sabitlerinin yalnızca İLK satırını değiştiren mutantları
    kaçırıyor. Aşağıdaki on parça bu modülün etik çekirdeğidir: ikisi
    birim seçimini ve klasikliği, ikisi 3B ile Zeeman'ın neden kapsam dışı
    olduğunu, ikisi daireselliğin kendisini, ikisi çıpa sınıflandırmasını,
    ikisi de pencerenin `sgf`e ve kipe bağlılığını yazıyor.

    Bu test bir totoloji DEĞİLDİR: sağ taraf `factory` modülünden okunur,
    sol taraftaki parçalar burada BAĞIMSIZ olarak yazılmıştır.

    ARANAN METİN PANELİN KENDİSİDİR, sabit listesi değil: `sonlu merdiven`
    satırı çalışma zamanında kurulur (basamak sayısını okur) ve sabit
    listede GEÇMEZ.
    """
    metin = " ".join(deger for _, deger in _provenance(sevk_senaryo).as_lines())
    for parca in (
        "c = eps_0 = mu_0 = 1 BIR BIRIM SECIMIDIR, FIZIK DEGIL",
        "kuantum operatoru DEGIL; guncelleme denklemlerinde ne foton ne hbar gecer",
        "3B: bu modul 2B'dir ve 3B BILINCLI OLARAK kapsam disidir",
        "bu YORUNGESEL g = 1'dir ve spin Zeeman'i g = 2 ister",
        "R, girilen n'in cebirsel sonucudur ve n'e BIRE BIR geri cevrilebilir",
        "BU MODUL HICBIR MALZEMENIN OPTIK SABITINI IDDIA ETMEZ",
        "SEMA CIPASI (COZUCUYU dogrular, FIZIGI DEGIL",
        "CEVIRI DENETIMI: Fresnel |R|",
        "BESSEL ORANININ HANE SAYISI KIPE BAGLIDIR",
        "sgf PENCERENIN PARCASIDIR, bir analiz ayrintisi DEGIL",
    ):
        assert parca in metin, f"beyan kayip: {parca!r}"


def test_PANEL_DOKUZ_SATIRIN_DOKUZUNU_da_SIRASIYLA_basiyor(sevk_senaryo):
    """Spec §6'nın zorunlu beyanları, HER BİRİ KENDİ SATIRINDA.

    Etiket listesi tam olarak kilitlenir: bir satır kaybolursa ya da yeri
    değişirse test kırılır. `Çıpa sınıfı`, `Pencere` ve `Yakınsama tanısı`
    ayrı satırlardır ve bu bir biçim tercihi DEĞİL — gerekçeleri
    `provenance.py`de yazılı. Özellikle sonuncusu: `diagnostics`e konsaydı
    panelde **"✓ … verdict=YARGILANAMAZ"** diye görünürdü, yani
    yargılanamadığı söylenen bir sayının yanında bir onay işareti.
    """
    etiketler = [etiket for etiket, _ in _provenance(sevk_senaryo).as_lines()]
    assert etiketler == PANEL_SATIRLARI
    satirlar = dict(_provenance(sevk_senaryo).as_lines())
    for etiket in PANEL_SATIRLARI:
        assert satirlar[etiket].strip(), f"{etiket} satiri BOS"


def test_YAKLASIMLAR_IHMAL_DAIRESELLIK_SINIF_PENCERE_KARISMIYOR(sevk_senaryo):
    """Beş liste AYNI panelde ama AYNI SATIRDA değil.

    Tersine çevrilmiş bir atama (`approximations=_EM_IHMAL`) bütün
    cümleleri hâlâ ekrana basardı ve "beyan var mı" diye bakan her test
    yeşil kalırdı. Ayıran şey cümlelerin HANGİ satırda olduğudur: bir
    yaklaşım YAPILAN bir şeydir, bir ihmal YAPILMAYAN, dairesellik beyanı
    çıktının NE OLMADIĞINI, çıpa sınıfı okunan sayının NEYİ doğruladığını,
    pencere ise hükmün NEREDE geçerli olduğunu söyler. Beşi birbirinin
    yerine geçemez.
    """
    satirlar = dict(_provenance(sevk_senaryo).as_lines())
    for cumle in fabrika._EM_YAKLASIMLAR:
        assert cumle in satirlar["Yaklaşımlar"]
        assert cumle not in satirlar["İhmal edilen"]
    for cumle in fabrika._EM_IHMAL:
        assert cumle in satirlar["İhmal edilen"]
        assert cumle not in satirlar["Yaklaşımlar"]
    assert fabrika._EM_BEKLENEN_HATA == satirlar["Beklenen hata"]
    assert fabrika._DAIRESELLIK_BEYANI == satirlar["Dairesellik"]
    assert fabrika._CIPA_SINIFI_BEYANI == satirlar["Çıpa sınıfı"]
    assert fabrika._PENCERE_BEYANI in satirlar["Pencere"]
    for yer in ("Yaklaşımlar", "İhmal edilen", "Beklenen hata"):
        assert fabrika._DAIRESELLIK_BEYANI not in satirlar[yer]
        assert fabrika._CIPA_SINIFI_BEYANI not in satirlar[yer]
        assert fabrika._PENCERE_BEYANI not in satirlar[yer]


def test_CIPA_SINIFI_UC_SINIFI_da_ADIYLA_ve_AYRI_ayri_yaziyor(sevk_senaryo):
    """Spec §6: her çıpa **şema / fizik / çeviri denetimi** diye etiketli.

    Üçünün de adı geçmeli VE dört fizik çıpasının dördü de fizik tarafında
    sayılmalı. Fresnel'in çeviri denetimi olduğu ve `kaydir`ı GÖRMEDİĞİ
    ayrıca aranıyor — bu depoda bir şema çıpasını fizik çıpası diye sunmak
    üç kez düzeltilmiş bir hatadır ve bu satır onun panel karşılığıdır.
    """
    satir = dict(_provenance(sevk_senaryo).as_lines())["Çıpa sınıfı"]
    assert "FIZIK CIPASI" in satir
    assert "SEMA CIPASI" in satir
    assert "CEVIRI DENETIMI" in satir
    assert "OLCUM ALETI" in satir
    fizik = satir[satir.index("FIZIK CIPASI") : satir.index("SEMA CIPASI")]
    for ad in ("Bessel oranlari", "Rayleigh ussu", "tek yarigin", "delta_ps"):
        assert ad in fizik, f"fizik cipasi kayip: {ad}"
    sema = satir[satir.index("SEMA CIPASI") : satir.index("CEVIRI DENETIMI")]
    for ad in ("Courant", "ikinci mertebesi", "dispersiyon", "sihirli adim"):
        assert ad in sema, f"sema cipasi kayip: {ad}"
    # Fresnel FIZIK tarafinda GECMIYOR -- sinifi orasi degil.
    assert "Fresnel" not in fizik
    assert "kaydir" in satir and "2.4e-06" in satir


def test_PENCERE_SATIRI_DORT_ZORUNLU_BEYANI_da_tasiyor(sevk_senaryo):
    """Görev listesinin dört pencere maddesi + `κ`nın statüsü.

    Beşinin beşi de spec §6'da ya da `observable/em_obs.py`nin sevk edilmiş
    dokstringinde ADIYLA istenmiştir. Sonuncusu bir SÖZÜN ÖDENMESİDİR:
    `DEFAULT_KAPPA`nın yorumu *"panelde öyle yazılır"* diyordu ve o cümle
    ancak panel sevk edilince sınanabilir hâle geldi.
    """
    satir = dict(_provenance(sevk_senaryo).as_lines())["Pencere"]
    assert "HANE SAYISI KIPE BAGLIDIR" in satir and "8.85 KAT" in satir
    assert "a/lam >= 2" in satir
    assert "sgf PENCERENIN PARCASIDIR" in satir
    assert "sgf = 9" in satir and "sgf = 3" in satir and "MONOTON DEGIL" in satir
    assert "kappa = 0.5 BIR TOLERANS DEGIL" in satir
    assert "MODELLEME SECIMIDIR" in satir
    # Ve BU KOSUMUN kendi penceresi de yazili -- sabit degil, `Window`dan.
    for parca in ("izgara 17 x 17", "S (Courant) = 0.5", "adim = 2100",
                  "npml = 0", "dx = 1"):
        assert parca in satir, f"kosum penceresi kayip: {parca}"


def test_PENCERE_SATIRI_SENARYODAN_gelir_SABIT_DEGIL(sevk_senaryo):
    """Pencere satırı gerçekten `Window`u ve senaryoyu okumalı.

    Sabit yazılsaydı başka bir pencereyle koşan kullanıcı panelde koşmadığı
    hesabın beyanını görürdü. Bu test hiç FDTD KOŞMAZ.
    """
    baska = dataclasses.replace(UCUZ_PENCERE, cells=(41, 41), steps=999, sgf=3.0)
    senaryo = dataclasses.replace(sevk_senaryo, pad=4, detectors=12, peak_floor=0.02)
    p = provenance_for(
        senaryo,
        windows=(baska,) * 4,
        ratios=(1.5,) * 4,
        noise=(1e-6,) * 4,
        peaks=(9,) * 4,
        orders=(2.0, 2.0),
        p_inf=2.0,
        uncertainty=0.01,
        verdict="OTURDU",
    )
    satir = dict(p.as_lines())["Pencere"]
    assert "izgara 41 x 41" in satir
    assert "adim = 999" in satir
    assert "sgf = 3" in satir
    assert "pad = 4" in satir
    assert "12 dedektor" in satir
    assert "tepe tabani = 0.02" in satir


def test_YARGILANAMAZ_SESSIZCE_OTURDU_diye_BASILMIYOR(sevk_senaryo):
    """ÜÇ DEĞERLİ HÜKÜM — ve modülün en sert kuralı.

    `OTURDU` DIŞINDA hiçbir hüküm bir tamsayı mertebe yazdırmaz. Üç yön de
    kilitlenir, çünkü tek yönlü bir test ikisinden birini kaçırırdı:

      * `OTURDU` → `OTURDU(2)` yazılır ve `HUKUM_VERILMEDI` GEÇMEZ,
        `converged` ✓ olur.
      * `YARGILANAMAZ` → `HUKUM_VERILMEDI` yazılır, `OTURDU(` HİÇ geçmez,
        `converged` ✗ olur.
      * `OTURMADI` → aynı, ve `p_inf` okunabilmiş olsa BİLE bir tamsayı
        sertifika edilmez.

    "Hep OTURDU bas" diyen bir mutant ikinci ve üçüncü bloktan, "hep
    YARGILANAMAZ bas" diyen bir mutant birinciden düşer.

    ARAMA `[CANLI ...]` PARÇASIYLA SINIRLI — VE BU SINIR TESTİN KENDİSİ
    TARAFINDAN BULUNDU: satırın ikinci yarısı `[BEYAN ...]` etiketiyle
    başka çıpaların ölçülmüş hükümlerini taşıyor ve onlardan biri (TİY faz
    farkı) zaten `OTURDU(2)` diyor. Yani "satırda OTURDU geçiyor mu" diye
    bakan bir test, bu koşumun hükmünü HİÇ ölçmeden yeşil kalırdı. İki
    etiketin ayrı olması bir biçim tercihi değil; canlı sayı ile alıntılanan
    sayıyı ayıran şey odur.
    """
    tam = _provenance(sevk_senaryo, verdict="OTURDU")
    tani = _canli(dict(tam.as_lines())["Yakınsama tanısı"])
    assert "OTURDU(2)" in tani
    assert HUKUM_VERILMEDI not in tani
    assert tam.converged is True
    assert dict(tam.as_lines())["Yakınsama"].startswith("✓")

    for verdict, p, u in (
        ("YARGILANAMAZ", None, None),
        ("YARGILANAMAZ", 2.03, 0.9),
        ("OTURMADI", 2.4, 0.05),
    ):
        kirik = _provenance(
            sevk_senaryo, p_inf=p, uncertainty=u, verdict=verdict, orders=()
        )
        tani = _canli(dict(kirik.as_lines())["Yakınsama tanısı"])
        assert HUKUM_VERILMEDI in tani, f"{verdict} icin hukum isareti kayip"
        assert "OTURDU(" not in tani, f"{verdict} sessizce OTURDU diye basildi"
        assert verdict in tani
        assert kirik.converged is False
        assert dict(kirik.as_lines())["Yakınsama"].startswith("✗")


def test_OTURDU_HUKMU_P_INF_OLMADAN_verilemez(sevk_senaryo):
    """`verdict = "OTURDU"` ama `p_inf = None` SESSİZCE kabul edilemez.

    Oturulan tamsayı mertebe `p_inf`ten okunur; `None` ile çağrılırsa
    yazılacak sayı yoktur ve "OTURDU()" gibi yarım bir hüküm basmak
    olmayan bir sertifikayı varmış gibi göstermek olurdu.
    """
    with pytest.raises(InvalidInputError, match="p_inf"):
        _provenance(sevk_senaryo, p_inf=None, uncertainty=None, verdict="OTURDU")


def test_TANI_DIZILERI_YARICAP_SAYISIYLA_uyusmak_zorunda(sevk_senaryo):
    """Yanlış uzunlukta bir tanı dizisi SESSİZCE kabul edilemez.

    `zip` kısa olanda durur: dört yarıçapa üç oran verilseydi son basamak
    hiç raporlanmaz ve panel eksik bir merdiven basardı.
    """
    for eksik in ("windows", "ratios", "noise", "peaks"):
        kw = dict(
            windows=(UCUZ_PENCERE,) * 4,
            ratios=(1.5,) * 4,
            noise=(1e-6,) * 4,
            peaks=(20,) * 4,
            orders=(2.0, 2.0),
            p_inf=2.0,
            uncertainty=0.01,
            verdict="OTURDU",
        )
        kw[eksik] = kw[eksik][:3]
        with pytest.raises(InvalidInputError, match=eksik):
            provenance_for(sevk_senaryo, **kw)


def test_DAIRESELLIK_BEYANININ_SAYISI_BU_TESTTE_YENIDEN_URETILIYOR(sevk_senaryo):
    """Panelin dairesellik sayısı bir alıntı değil, YENİDEN ÜRETİLİYOR.

    `R = ((1−n)/(1+n))²` ve tersi `n = (1+√R)/(1−√R)`. İkisi de burada
    BAĞIMSIZ olarak yazıldı; panelin yazdığı üç satırın üçü de bu
    aritmetikten çıkıyor. Yani bu modülün "yol haritasının hedefi bir
    tautolojidir" iddiası, panelde yazılı olduğu hâliyle test tarafından
    da ölçülüyor.
    """
    satir = dict(_provenance(sevk_senaryo).as_lines())["Dairesellik"]
    for n, yazili_R in ((1.33, "0.020059312200"), (1.5, "0.040000000000"),
                        (3.4, "0.297520661157")):
        R = ((1.0 - n) / (1.0 + n)) ** 2
        geri = (1.0 + math.sqrt(R)) / (1.0 - math.sqrt(R))
        assert f"{R:.12f}" == yazili_R
        assert yazili_R in satir
        assert abs(geri - n) <= 5e-16
    assert "1.500000000000" in satir


# ---------------------------------------------------------------------------
# FDTD koşan testler — sevk edilen senaryonun tek koşumunu paylaşırlar.
# ---------------------------------------------------------------------------


def test_SEVK_EDILEN_KOSUM_MERDIVENI_OTURUYOR(sevk_yuk):
    """Sevk edilen senaryo panelin ✓ dalını KENDİSİ gösteriyor.

    Bu bir Maxwell iddiası değil bir koşum özelliğidir (dosya
    dokstringindeki tabloya bakın): pencere kendi mertebesini okuyacak
    kadar geniş. Modül 4'te bu dal sevk edilen senaryodan hiç
    görülemiyordu ve ayrı bir ucuz senaryo gerekmişti; burada sevk edilen
    senaryonun kendisi gösteriyor.

    Değerler `abs` toleransla değil, ÖLÇÜLEN sayıya `rel` ile bağlanıyor;
    hüküm ise toleranssızdır.
    """
    L = sevk_yuk.ladder
    assert L.verdict == "OTURDU"
    assert L.settled_order == 2
    assert L.p_inf == pytest.approx(2.195532, abs=0.02)
    assert L.uncertainty == pytest.approx(0.061188, rel=0.15)
    assert abs(L.p_inf - 2) + L.uncertainty < 0.5  # kappa
    assert list(sevk_yuk.spacings) == [0.2, 0.1, 0.05, 0.025]
    assert [int(t) for t in sevk_yuk.peaks] == [28, 21, 17, 15]
    # Merdiven GERCEKTEN yakinsiyor: her basamak oncekinden daha yakin.
    tam = 3.831705970207512 / 2.404825557695773
    sapma = [abs(float(x) - tam) for x in sevk_yuk.ratios]
    assert sapma == sorted(sapma, reverse=True)
    assert sapma[0] / sapma[-1] > 50.0


def test_IKINCI_EN_DUSUK_TEPE_ILE_BESSEL_ESLEMESI_AYNI_SAYIYI_veriyor(sevk_yuk):
    """`viz/em_frames.py` TEORİSİZDİR — ve bu ölçülerek doğrulanıyor.

    Sevk edilen boru hattı "ikinci en düşük tepenin en düşüğe oranı"nı
    okur; `test_bessel.py` ise tepeyi `j₁,₁/j₀,₁`e **en yakınlıkla** seçer.
    İki tanım FARKLI şeylerdir ve sevk edilen pencerede aynı sayıyı
    verdikleri BURADA ölçülüyor — birebir, tolerans YOK.

    Bu, `mode_spectrum`un *"Bessel köklerini HİÇ görmez"* beyanının sevk
    edilen boru hattındaki karşılığıdır: `src/` ağacında hiçbir kök yok ve
    yine de doğru kip okunuyor.
    """
    from scipy.special import jn_zeros

    hedef = float(jn_zeros(1, 1)[0]) / float(jn_zeros(0, 1)[0])
    for i, spektrum in enumerate(sevk_yuk.spectra):
        oranlar = np.asarray(spektrum.ratios())
        k = int(np.argmin(np.abs(oranlar - hedef)))
        assert abs(oranlar[k] - hedef) <= 0.02 * hedef
        assert float(oranlar[k]) == float(sevk_yuk.ratios[i])


def test_KABA_BASAMAK_BES_ORANDAN_IKISINI_INCE_BASAMAK_BESINI_buluyor(sevk_yuk):
    """DOKSTRING DENETİMİNİN ÜRÜNÜ — düzeltilmiş hükmün kilidi.

    `test_bessel.py` bir zamanlar *"R = 5'te tepe kapısı kipi bulamıyor"*
    diyordu. ÖLÇÜLDÜ ve çürüdü: R = 5'te **28 tepe** var (R = 40'takinden
    fazla) ve merdivenin okuduğu oran bulunuyor. Düşen şey tepe kapısı
    değil `test_bessel.py`nin **%2'lik eşleme penceresidir** ve üç oranın
    genliği (0.57 / 0.82 / 0.30) tabandan çok yukarıda — yani onlar da
    "bulunuyor", sadece **çok kaymış** durumdalar.

    Bu test aynı zamanda panelin *"hane sayısı kipe bağlıdır"* cümlesinin
    sayısal karşılığıdır: aynı senaryonun iki ucu 2 ve 5 oran veriyor.
    """
    from scipy.special import jn_zeros

    j01 = float(jn_zeros(0, 1)[0])
    hedefler = [
        float(jn_zeros(m, 2)[n - 1]) / j01 for m, n in ((1, 1), (2, 1), (0, 2), (3, 1), (1, 2))
    ]

    def eslesen(spektrum):
        r = np.asarray(spektrum.ratios())
        return sum(
            1 for h in hedefler if abs(r[int(np.argmin(np.abs(r - h)))] - h) <= 0.02 * h
        )

    assert eslesen(sevk_yuk.spectra[0]) == 2  # R = 5
    assert eslesen(sevk_yuk.spectra[-1]) == 5  # R = 40
    # Tepe kapisi R = 5'te DAHA COK tepe buluyor -- "bulamiyor" degil.
    assert int(sevk_yuk.peaks[0]) > int(sevk_yuk.peaks[-1])


def test_AYNI_PENCERE_AYNI_SONUC_FARKLI_PENCERE_FARKLI(sevk_senaryo, sevk_yuk):
    """Tekrarlanabilirlik — ve pencerenin GERÇEKTEN okunduğu.

    FDTD'de tohum yoktur, dolayısıyla "aynı tohum aynı sonuç" testinin
    karşılığı budur: aynı pencere BİT BİT aynı diziyi vermeli. İkinci yarı
    olmadan pencereyi yok sayan bir mutant (ör. `step_factor`ı sabitleyen)
    birinci yarıyı GEÇEBİLİRDİ.
    """
    tekrar = build_em_payload(build_em(sevk_senaryo))
    assert np.array_equal(tekrar.ratios, sevk_yuk.ratios)
    assert np.array_equal(tekrar.noise, sevk_yuk.noise)
    assert np.array_equal(tekrar.field, sevk_yuk.field)
    assert tekrar.ladder.verdict == sevk_yuk.ladder.verdict

    baska = build_em_payload(
        build_em(dataclasses.replace(sevk_senaryo, radii=(5.0, 10.0, 20.0, 40.0),
                                     sgf=3.0))
    )
    assert not np.array_equal(baska.ratios, sevk_yuk.ratios)


def test_ALAN_ANLIK_GORUNTUSU_ILETKEN_MASKESIYLE_BIRLIKTE_geliyor(sevk_yuk):
    """`field` tek başına anlamsızdır; `field_conductor` ile ANLAMLIDIR.

    Alan haritasındaki sıfırlar İKİ AYRI şey olabilir: iletken bir düğüm,
    ya da gerçekten sıfır olan bir alan. Test ikisinin de var olduğunu
    ölçer — maskenin içinde alan tam sıfır, dışında değil.
    """
    alan, maske = sevk_yuk.field, sevk_yuk.field_conductor
    assert alan.shape == maske.shape == (87, 87)
    assert maske.dtype == bool
    assert np.all(alan[maske] == 0.0)
    assert np.any(np.abs(alan[~maske]) > 0.0)
    # Maske gercekten DAIRE: kose iletken, merkez degil.
    assert maske[0, 0] and not maske[43, 43]


@pytest.mark.slow
def test_FABRIKA_IKINCI_BIR_FIZIK_YOLU_ACMIYOR():
    """`build_em`, `run_2d`in ürettiğinden BAŞKA bir sayı üretmez.

    Fabrikanın işi geometriyi kurmak, adım/dalga boyu ölçeklemesini
    dağıtmak ve beyanı toplamaktır; bir gözlemlenebilirin kendisini
    hesaplasaydı iki yol ayrışabilirdi ve panel motorun söylemediğini
    yazardı.

    Karşılaştırma BİREBİR (`array_equal`) yapılır, tolerans YOKTUR: aynı
    pencere aynı diziyi verir. Karşı taraf `test_bessel.py`nin sevk edilmiş
    `_kosum`udur — yani `src/`teki geometri ile `tests/`teki geometrinin
    AYNI kayıt ürettiği ölçülüyor. İkisi ayrışırsa senaryo, çıpanın
    ölçtüğünden başka bir şeyi gösteriyor olurdu.

    Fabrika TEK YARIÇAPLA çağrılıyor ve bu bir kaçamak değil KATMAN
    AYRIMININ kendisidir: dört basamak şartı YÜKLEYİCİDEDİR (YAML'ın
    hatasını yakalar), dataclass'ta değil. Dört basamak koşmak burada 20
    saniye tutardı ve ölçülen şey değişmezdi.

    Ölçülen süre: iki koşum × R = 10 @ adım 1680·R ≈ 0.6 s.
    """
    from scipy.special import jn_zeros

    from tests.em.test_bessel import _kosum

    j01 = float(jn_zeros(0, 1)[0])
    tekli = EMScenario(
        name="fabrika-karsilastirmasi",
        radii=(10.0,),
        ratio=2.0,
        courant=0.5,
        step_factor=1680.0,
        wavelength_per_radius=2.0 * math.pi / (1.8 * j01),
        sgf=1.5,
        detectors=8,
        pad=8,
        window_fn="blackman",
        peak_floor=0.01,
    )
    bizim = build_em(tekli).records[0]
    onlarin = _kosum(10.0)
    assert bizim.probe_indices == onlarin.probe_indices
    assert bizim.window == onlarin.window
    assert np.array_equal(bizim.probes, onlarin.probes)


@pytest.mark.slow
def test_HUKUM_ADIM_CARPANINA_DUYARSIZ(sevk_senaryo):
    """Hüküm KAYDIN UZUNLUĞUNDAN değil MERDİVENLENMEDEN geliyor.

    Adım çarpanı 420 → 840 (yani her basamağın kaydı iki katına çıkınca)
    merdiven yine OTURDU(2) veriyor: p_inf 2.195532 → 2.184666, u 0.061188
    → 0.054759. Ölçülen süre bu basamakta ≈ 3.1 s.

    Dört çarpanın dördü de ölçüldü (420/840/1260/1680 → 2.195532 /
    2.184666 / 2.191308 / 2.191648); testte en ucuz ayırt edici çift
    koşuluyor, tablonun tamamı dosya dokstringindedir.
    """
    yuk = build_em_payload(
        build_em(dataclasses.replace(sevk_senaryo, step_factor=840.0))
    )
    assert yuk.ladder.verdict == "OTURDU"
    assert yuk.ladder.settled_order == 2
    assert yuk.ladder.p_inf == pytest.approx(2.184666, abs=0.02)


def test_CLI_UCTAN_UCA(tmp_path):
    """`python -m qlab.run scenarios/em_tarama.yaml -o em.npz`.

    ÖLÇÜLDÜ: taramanın kendisi 1.49 s, yorumlayıcı başlangıcıyla birlikte
    alt süreç ~3 s civarı. Deponun `slow` eşiği "kendi `setup+call` süresi
    2 s ve üstünde" der; bu test ölçülerek `slow` bırakılmadı çünkü tek
    başına koşan uçtan uca dal `-m "not slow"` koşumunda da koşmalıdır —
    Modül 5'in fizik çıpalarının tamamı zaten `slow` işaretli ve o koşumda
    atlanıyor, panelin kendisi de atlansaydı "not slow" koşumu senaryo
    sınırını HİÇ sınamazdı.

    DOKUZ panel satırının DOKUZU da aranır. Metinler bu dosyada
    TEKRARLANMAZ; `factory` modülünden okunur, yani beyan düzeltilirse test
    onu KENDİLİĞİNDEN takip eder ve ikinci bir kopya doğmaz.
    """
    cikti = tmp_path / "em.npz"
    sonuc = subprocess.run(
        [sys.executable, "-m", "qlab.run", str(SEVK_EDILEN), "-o", str(cikti)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(PROJE_KOKU),
    )
    assert sonuc.returncode == 0, sonuc.stderr
    assert cikti.exists()

    for etiket in PANEL_SATIRLARI:
        assert etiket in sonuc.stdout, f"panel satiri kayip: {etiket}"
    for cumle in _tum_beyanlar():
        if cumle == HUKUM_VERILMEDI:
            continue  # sevk edilen kosumda merdiven OTURUYOR; asagida ayrica sinaniyor
        assert cumle in sonuc.stdout, f"beyan EKRANA basilmadi: {cumle[:50]!r}"
    assert "sonlu merdiven: bu kosum 4 basamak kullanir" in sonuc.stdout
    assert "merdiven hükmü: OTURDU(2)" in sonuc.stdout
    assert HUKUM_VERILMEDI not in sonuc.stdout

    with np.load(cikti, allow_pickle=False) as veri:
        assert list(veri["radii"]) == [5.0, 10.0, 20.0, 40.0]
        assert veri["field"].shape == (87, 87)
        assert veri["field_conductor"].dtype == bool
        assert str(veri["verdict"]) == "OTURDU"
        assert float(veri["p_inf"]) == pytest.approx(2.195532, abs=0.02)
        kaynak = json.loads(str(veri["provenance_json"]))
    # `.npz`ye YAZILAN beyan ile EKRANA basilan beyan AYNI kaynaktan.
    assert kaynak["circularity"] == fabrika._DAIRESELLIK_BEYANI
    assert kaynak["anchor_class"] == fabrika._CIPA_SINIFI_BEYANI
    assert kaynak["convergence_diagnosis"].startswith("[CANLI")
