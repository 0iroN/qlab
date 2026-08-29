"""`kind: grav` senaryosu, panel beyanı ve uçtan uca koşum.

SINIF: TRANSKRİPSİYON DENETİMİ. Bu dosyanın hiçbir testi yeni bir fizik
iddiası kurmaz; senaryo sınırının (YAML → dataclass → motor → gözlemlenebilir
→ panel) doğru bağlandığını ve panelin **on zorunlu satırı** bastığını ölçer.
Modülün fizik çıpaları başka yerdedir: perihel katsayısı `test_perihel.py`de,
hata çubuğu makinesi `test_hata_cubugu.py`de, bükülme `test_bukulme.py`de,
kapalı-form türevler `test_kapali_form.py`de, sekiz yörüngesi ve monodromi
`test_nbody.py`de, kuram seçici `test_metric.py`de.

BU DOSYANIN EN ÖNEMLİ TESTİ BİR SAYI DEĞİL BİR SAYIM DENETLER, VE İKİ
KATMANLIDIR:

1. `test_BEYAN_TEK_KAYNAKTAN_geliyor` — her beyan cümlesi `src/` ağacında TAM
   BİR KEZ geçer. Ölçülmüş gerekçe bu depodadır: aynı beyan bloğu bir kez
   `scf.py` ile `tb/factory.py`de birebir kopyalanmıştı ve biri düzeltilip
   öteki unutulduğunda panel iki farklı şey iddia ediyordu.
2. `test_BEYAN_PARCASININ_KOPYASI_da_YASAK` — çünkü **tam cümle sayımı
   YETMEZ**: Modül 5'te bir mutant `_PENCERE_BEYANI`nin 55 karakterlik bir
   PARÇASINI `run.py`ye enjekte ederek tam-cümle sayımından KAÇTI.

===========================================================================
PARÇA EŞİĞİ BU MODÜLDE YENİDEN ÖLÇÜLDÜ — DEVRALINMADI
===========================================================================

`src/` ağacındaki (fabrika hariç) her dize sabiti bu modülün beyan
cümlelerinin içinde arandı:

    eşik  4 ->  92 eşleşme   (" ve ", " bir ", "Pencere", "angstrom" …)
    eşik  6 ->  20 eşleşme
    eşik  8 ->  13 eşleşme   ("DEJENERE", "SAGLAM", "angstrom" …)
    eşik 12 ->   2 eşleşme   ("foton kuresi", "bukulme orani")
    eşik 16 ->   0 eşleşme
    eşik 20 ->   0 eşleşme

12'deki iki eşleşme bir kopya DEĞİLDİR: `foton kuresi` `grav_obs.py`nin kendi
hata mesajındaki bir TERİM, `bukulme orani` ise `grav_frames.py`deki bir
SATIR ETİKETİdir — ikisi de paylaşılan SÖZCÜK, beyan metni değil. Yanlış
pozitif tabanı 12 ile 16 arasındadır; **16 seçildi**. Modül 5 aynı eşiği
bağımsız olarak ölçmüştü (4 → 207, 8 → 6, 12 → 4, 16 → 0) ve iki ölçüm aynı
yere düşüyor — ama bu dosya onu DEVRALMADI, yeniden ölçtü.

===========================================================================
SEVK EDİLEN KOŞUMUN ÖLÇÜLEN SAYILARI (29.08.2026, bu depoda, bu makine)
===========================================================================

Pencere: `G = c = 1`, M = 1, e = 0.3, ayar Schwarzschild, kareleme Richardson
(3e5, 1e6), ikinci mertebe tek basamak p = 3e5, bükülme Richardson
(1e5, 3e5), ODE yörünge motoru p = 100 / 2000 nokta, N-cisim DOP853 /
rtol 1e-13 / atol 1e-15.

    perihel katsayısı C   18.84955591968662     (6π'den −1.852140e-09)
    ikinci mertebe   K    85.24858543879077
    bükülme oranı         1.9999999996887523
    sekiz sabiti          9.237681250723917
    eliptik çift          ±1.872970598421, ±0.052921540137 rad
    bir radyal periyot    6.480660327813 rad    (2π'den 1.974750e-01 büyük)

    taban (kareleme)      1.594187e-09   S/G 1.1824e+10   SAGLAM   6/6
    taban (ODE)           1.921807e-06   S/G 9.8082e+06   SAGLAM   6/6
    taban (bükülme)       3.651917e-10   S/G 5.4766e+09   SAGLAM   6/6
    taban (N-cisim)       3.996803e-14   S/G 3.0926e+13   SAGLAM   7/7
    taban (doğrusal harita) 0.000000e+00  S/G YOK        DEJENERE  0/1

Toplam koşum ≈ 0.7 s (BVP 0.19 s + monodromi 0.24 s + tabanlar 0.13 s).

`slow` EŞİĞİ — VE BU DOSYANIN BEYAN EDİLMİŞ SAPMASI. Modülün eşiği
`test_kapali_form.py`de **1.0 s** olarak ölçülmüştü. Bu dosyanın en pahalı
testi `test_CLI_UCTAN_UCA` ve **1.43 s** sürüyor, yani eşiğin ÜSTÜNDE — ama
BİLEREK işaretlenmedi. Gerekçe Modül 5'inkiyle aynı: `slow` işaretlenseydi
`-m "not slow"` koşumu senaryo sınırını (YAML → panel → `.npz`) HİÇ
sınamazdı. Sapma gizlenmiyor, yazılıyor. Dosyanın geri kalanı eşiğin
altındadır (ikinci en pahalı: 0.91 s modül fixture'ı, üçüncü: 0.80 s).

Bu dosya: 40 test, **5.68 s**. Toplam `tests/grav`: **307 test, 26.64 s**
(`-m "not slow"`: 298 test, 9.35 s).
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import qlab.engine.grav.factory as fabrika
from qlab.engine.grav.factory import (
    GR_DEGIL_ISARETI,
    GR_NOKTASI,
    KAPALI_FORM_YOK,
    KURAMA_BAGLI_KAPALI_FORMLAR,
    build_grav,
    grav_declaration,
    provenance_for,
)
from qlab.errors import InvalidInputError, ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import GravScenario
from qlab.viz.grav_frames import build_grav_payload

PROJE_KOKU = Path(__file__).resolve().parents[2]
SEVK_EDILEN = PROJE_KOKU / "scenarios" / "grav_tarama.yaml"
SEVK_YAML = SEVK_EDILEN.read_text(encoding="utf-8")

#: Panelin on zorunlu satırı, SIRASIYLA. `Kuram seçici` bu modülde eklendi ve
#: `Çıpa sınıfı` ile `Pencere` arasına girer — spec §6.5 ikisini aynı başlık
#: altında topluyor, ama AYRI satırlardır: çıpa sınıfı okunan sayının NEYİ
#: doğruladığını, kuram seçici sayının HANGİ KURAMA ait olduğunu söyler.
PANEL_SATIRLARI = [
    "Yöntem",
    "Yaklaşımlar",
    "İhmal edilen",
    "Beklenen hata",
    "Dairesellik",
    "Çıpa sınıfı",
    "Kuram seçici",
    "Pencere",
    "Yakınsama",
    "Yakınsama tanısı",
]

#: Bir dize sabitinin "beyan parçası" sayılması için gereken en kısa uzunluk.
#: SEÇİLMEDİ, ÖLÇÜLDÜ — tarama dosya dokstringindedir (4 → 92, 6 → 20,
#: 8 → 13, 12 → 2, 16 → 0, 20 → 0).
_BEYAN_PARCASI_ESIGI = 16


@pytest.fixture(scope="module")
def sevk_senaryo() -> GravScenario:
    return load_scenario(SEVK_EDILEN)


@pytest.fixture(scope="module")
def sevk_setup(sevk_senaryo):
    return build_grav(sevk_senaryo)


@pytest.fixture(scope="module")
def sevk_yuk(sevk_setup):
    return build_grav_payload(sevk_setup)


@pytest.fixture(scope="module")
def sevk_panel(sevk_senaryo, sevk_setup, sevk_yuk):
    return provenance_for(
        sevk_senaryo,
        setup=sevk_setup,
        floors=sevk_yuk.floors,
        second_order=sevk_yuk.second_order,
        closed_forms=sevk_yuk.closed_forms,
        eight=sevk_yuk.eight,
        angles=sevk_yuk.angles,
    )


def _yaz(tmp_path, metin: str) -> Path:
    yol = tmp_path / "grav.yaml"
    yol.write_text(metin, encoding="utf-8")
    return yol


def _degistir(anahtar: str, deger: str) -> str:
    """Sevk edilen YAML'ın tek bir anahtarını değiştirir.

    Satır bazlı ve satır başına çapalı: yorum bloğunda aynı sözcük geçtiği
    için düz `str.replace` yanlış yeri değiştirirdi (ölçüldü — `gauge`
    dosyanın yorumlarında da geçiyor).
    """
    satirlar = []
    bulundu = False
    for s in SEVK_YAML.splitlines():
        if s.startswith(f"{anahtar}:"):
            satirlar.append(f"{anahtar}: {deger}")
            bulundu = True
        else:
            satirlar.append(s)
    assert bulundu, f"{anahtar} sevk edilen YAML'da bulunamadi"
    return "\n".join(satirlar) + "\n"


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
        fabrika._GRAV_YAKLASIMLAR
        + fabrika._GRAV_IHMAL
        + (
            fabrika._GRAV_BEKLENEN_HATA,
            fabrika._DURUST_OLUMSUZLAR,
            fabrika._DAIRESELLIK_43_SANIYE,
            fabrika._DAIRESELLIK_SEKIZ,
            fabrika._CIPA_SINIFI_BEYANI,
            fabrika._KURAM_SECICI_BEYANI,
            fabrika._GURULTU_TABANI_BEYANI,
            fabrika._PENCERE_BEYANI,
            GR_DEGIL_ISARETI,
            KAPALI_FORM_YOK,
        )
    )


#: KOSUMA BAGLI beyanlar: her kosumda basilmazlar, yalniz kendi kosullari
#: dogdugunda. Tek kaynak kurali onlar icin de gecerlidir (ikisi de `src/`
#: agacinda tam bir kez gecer ve `run.py` ikisini de yazmaz, ITHAL EDER) ama
#: "her panelde var mi" sinavindan MUAFTIRLAR ve muafiyetin karsiligi ayri
#: bir testtir. Modul 5'in `HUKUM_VERILMEDI` muafiyetiyle ayni statu.
KOSUMA_BAGLI = (GR_DEGIL_ISARETI, KAPALI_FORM_YOK)


# ---------------------------------------------------------------------------
# BEYANIN TEK KAYNAĞI — İKİ KATMANLI KİLİT
# ---------------------------------------------------------------------------


def test_BEYAN_TEK_KAYNAKTAN_geliyor():
    """BU DOSYANIN EN ÖNEMLİ TESTİ. Beyan metninin İKİ KOPYASI OLAMAZ.

    Kilit iki yönlüdür:

      * `src/` ağacında her cümle tam BİR kez — kaynakta kopya yok. Bu,
        `run.py::_run_grav`ın beyan metnini kendisi yazmasını da YASAKLAR:
        yazsaydı sayım 2 olurdu. `_run_grav` `GR_DEGIL_ISARETI`yi bile
        yazmaz, fabrikadan İTHAL EDER.
      * `tests/` ağacında, BU DOSYA hariç, sıfır kez — ikinci bir test
        dosyası açıklama cümlesini elle yeniden yazamaz.

    BU DOSYANIN MUAF OLMASI BİR SAPMA DEĞİL, KİLİDİN KENDİSİDİR: tam cümle
    aramak için cümleyi bir kez yazmak gerekir ve o tek kopya
    `test_MODULUN_ETIK_CEKIRDEGI_metinde_YAZILI`dadır.
    """
    beyanlar = _tum_beyanlar()
    assert len(beyanlar) == 24, "beyan sayisi degisti; testi bilerek guncelleyin"
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


def test_BEYAN_PARCASININ_KOPYASI_da_YASAK():
    """TAM CÜMLE SAYIMININ ÖLÇÜLMÜŞ DELİĞİ — ve bu modülde YENİDEN ÖLÇÜLDÜ.

    Modül 5'te `run.py`ye `_PENCERE_BEYANI`nin **55 karakterlik bir parçası**
    enjekte edildiğinde mutant **hayatta kaldı**: tam cümlenin `src/` sayımı
    hâlâ 1'di, çünkü kopyalanan şey cümlenin tamamı değildi.

    Korunan şey "aynı cümle iki kez yazılmasın" değil, **panelin iki farklı
    şey iddia edememesi**dir; bir cümlenin yarısını kopyalayıp sonra
    fabrikadaki asıl cümleyi düzeltmek tam olarak o arızayı üretir. Bu
    yüzden kural parça seviyesindedir: `src/` ağacında (fabrika hariç)
    hiçbir dize sabiti bir beyan cümlesinin İÇİNDE geçemez.

    EŞİK DEVRALINMADI, YENİDEN ÖLÇÜLDÜ — tarama dosya dokstringindedir.
    """
    beyanlar = _tum_beyanlar()
    fabrika_yolu = (PROJE_KOKU / "src/qlab/engine/grav/factory.py").resolve()
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


def test_GRAV_DECLARATION_PANELIN_TAMAMINI_TASIYOR(sevk_setup, sevk_panel):
    """`grav_declaration` ile panel AYNI sabitleri okur — iki kopya YOK.

    Bu bir totoloji değildir: sağ taraf `provenance.as_lines()` çıktısıdır,
    yani `Provenance`ın AYRI alanlarına dağıtılmış hâlidir; sol taraf tek bir
    dizgedir. İkisinin aynı cümleleri taşıması, dağıtımın hiçbir beyanı
    DÜŞÜRMEDİĞİNİ söyler — ve Modül 6'nın planında bu tam olarak bir kez
    düşmüştü (dairesellik beyanının İKİNCİ parçası).
    """
    metin = grav_declaration(sevk_setup)
    panel = " ".join(deger for _, deger in sevk_panel.as_lines())
    for cumle in _tum_beyanlar():
        if cumle in KOSUMA_BAGLI:
            # Bu iki isaret YALNIZ GR olmayan bir kosumda basilir ve o dal
            # `test_GR_OLMAYAN_KURAM_REDDEDILMEZ_ama_PANEL_SOYLER`de
            # sinaniyor (Modul 5'in `HUKUM_VERILMEDI` muafiyetiyle ayni
            # statu: bir kosuma BAGLI beyan, her kosumda basilmaz).
            continue
        assert cumle in metin, f"grav_declaration eksik: {cumle[:60]!r}"
        assert cumle in panel, f"panel eksik: {cumle[:60]!r}"
    # VE GR OLMAYAN BIR KOSUMDA KOSUMA BAGLI ISARETLER DE GECER.
    gr_degil = build_grav(_SENARYO_GR_DEGIL())
    assert GR_DEGIL_ISARETI in grav_declaration(gr_degil)


def test_MODULUN_ETIK_CEKIRDEGI_metinde_YAZILI(sevk_panel):
    """Anahtar kelime değil TAM CÜMLE PARÇASI aranır.

    Gerekçesi `test_tb_scenario.py`de ölçüldü: kelime bazlı bir test, bitişik
    dize sabitlerinin yalnızca İLK satırını değiştiren mutantları kaçırıyor.
    Aşağıdaki on iki parça bu modülün etik çekirdeğidir ve dokuzu spec §6'nın
    zorunlu beyanlarına birebir karşılık gelir.

    Bu test bir totoloji DEĞİLDİR: sağ taraf panelden okunur, sol taraftaki
    parçalar burada BAĞIMSIZ olarak yazılmıştır.
    """
    metin = " ".join(deger for _, deger in sevk_panel.as_lines())
    for parca in (
        "G = c = 1 BIR BIRIM SECIMIDIR, FIZIK DEGIL",
        "TEST PARCACIGI: yorungeyi cizen cismin kutlesi metrige GERI TEPMEZ",
        "BU MODUL MERKUR'UN PERIHEL KAYMASINI ONGORMEZ",
        "VE DEGERIN KENDISI HANGI TABLONUN SECILDIGINE BAGLIDIR",
        "SEKIZ YORUNGESINDE ARAMADAN SONRA DAIRESELLESEN UC IDDIA VARDIR",
        "GIRDI OLARAK KALAN TEK SEY SIMETRI SINIFIDIR",
        "MODULLER ARASI BAG -- YOKLUK BEYANI",
        "CIPA DEGIL -- RAFTA UC UYE VE UCUNUN SEBEBI AYRI",
        "KURAM SECICI YENI BIR KUTUDUR ve bu modulde IKI UYESI VAR",
        "KARISIK BORU YASAK",
        "BUKULME ORANININ PENCERESI AYAR-BAGIMLIDIR",
        "DURUST OLUMSUZLAR -- BEYAN EDILDI, DUZELTILMEDI",
    ):
        assert parca in metin, f"beyan kayip: {parca!r}"


# ---------------------------------------------------------------------------
# PANELİN ON SATIRI
# ---------------------------------------------------------------------------


def test_PANEL_ON_SATIRIN_ONUNU_da_SIRASIYLA_basiyor(sevk_panel):
    """Spec §6'nın zorunlu beyanları, HER BİRİ KENDİ SATIRINDA.

    Etiket listesi tam olarak kilitlenir: bir satır kaybolursa ya da yeri
    değişirse test kırılır. `Kuram seçici` bu modülde eklendi ve AYRI bir
    satırdır; gerekçesi `provenance.py`de yazılı.
    """
    etiketler = [etiket for etiket, _ in sevk_panel.as_lines()]
    assert etiketler == PANEL_SATIRLARI
    satirlar = dict(sevk_panel.as_lines())
    for etiket in PANEL_SATIRLARI:
        assert satirlar[etiket].strip(), f"{etiket} satiri BOS"


def test_KURAM_SECICI_YAKINSAMA_SATIRININ_ICINE_BASILMIYOR(sevk_panel):
    """`theory_selector` `diagnostics`e KONMADI — ve ölçülen sebep budur.

    `_diagnostics_summary()` çıktısı **"Yakınsama" satırının İÇİNE**, ✓/✗
    işaretinin HEMEN ARDINA basılır. Kuram seçici oraya konsaydı GR olmayan
    bir koşumda panelde **"✓ … theory=NOT-GR"** diye görünürdü — yani
    "çözülen kuram genel görelilik DEĞİL" uyarısının yanında bir onay
    işareti. Modül 4'ün `sampling` ve Modül 5'in `convergence_diagnosis`
    alanlarının ayrı tutulma gerekçesinin aynısı, ondan daha sert.

    İki yönlü kilit: (1) işaret `Yakınsama` satırında GEÇMEZ, (2) kendi
    satırında GEÇER.
    """
    satirlar = dict(sevk_panel.as_lines())
    assert GR_DEGIL_ISARETI not in satirlar["Yakınsama"]
    assert sevk_panel.theory_selector is not None
    assert sevk_panel.theory_selector not in satirlar["Yakınsama"]
    # GR OLMAYAN bir kosumda da ayni: isaret YAKINSAMA satirina sizmaz.
    gr_degil = _SENARYO_GR_DEGIL()
    setup = build_grav(gr_degil)
    yuk = build_grav_payload(setup)
    p2 = provenance_for(
        gr_degil, setup=setup, floors=yuk.floors, second_order=yuk.second_order,
        closed_forms=yuk.closed_forms, eight=yuk.eight, angles=yuk.angles,
    )
    s2 = dict(p2.as_lines())
    assert GR_DEGIL_ISARETI not in s2["Yakınsama"]
    assert GR_DEGIL_ISARETI in s2["Kuram seçici"]


def _SENARYO_GR_DEGIL() -> GravScenario:
    """GR olmayan bir kuram seçimi — `lam = 0` DEĞİL, `lam = 1`.

    `lam = 0` seçilseydi çıpanın kendisi tam sıfır olurdu ve gürültü tabanı
    ile sinyal AYNI noktada okunurdu; test o zaman panelin metnini değil
    aritmetiğin dejenereliğini ölçerdi. `gamma = 0, beta = 1` (lam = 1)
    fiziği açık tutar ve yalnız KURAMI değiştirir.
    """
    return dataclasses.replace(
        load_scenario(SEVK_EDILEN), gamma=0.0, beta=1.0,
    )


def test_YAKLASIMLAR_IHMAL_DAIRESELLIK_SINIF_KURAM_PENCERE_KARISMIYOR(sevk_panel):
    """Altı metin AYNI panelde ama AYNI SATIRDA değil.

    Tersine çevrilmiş bir atama (`approximations=_GRAV_IHMAL`) bütün
    cümleleri hâlâ ekrana basardı ve "beyan var mı" diye bakan her test yeşil
    kalırdı. Ayıran şey cümlelerin HANGİ satırda olduğudur: bir yaklaşım
    YAPILAN bir şeydir, bir ihmal YAPILMAYAN, dairesellik beyanı çıktının NE
    OLMADIĞINI, çıpa sınıfı okunan sayının NEYİ doğruladığını, kuram seçici
    sayının HANGİ KURAMA ait olduğunu, pencere ise hükmün NEREDE geçerli
    olduğunu söyler. Altısı birbirinin yerine geçemez.
    """
    satirlar = dict(sevk_panel.as_lines())
    for cumle in fabrika._GRAV_YAKLASIMLAR:
        assert cumle in satirlar["Yaklaşımlar"]
        assert cumle not in satirlar["İhmal edilen"]
    for cumle in fabrika._GRAV_IHMAL:
        assert cumle in satirlar["İhmal edilen"]
        assert cumle not in satirlar["Yaklaşımlar"]
    assert fabrika._DAIRESELLIK_43_SANIYE in satirlar["Dairesellik"]
    assert fabrika._DAIRESELLIK_SEKIZ in satirlar["Dairesellik"]
    assert fabrika._CIPA_SINIFI_BEYANI == satirlar["Çıpa sınıfı"]
    assert fabrika._KURAM_SECICI_BEYANI in satirlar["Kuram seçici"]
    assert fabrika._PENCERE_BEYANI in satirlar["Pencere"]
    assert fabrika._DURUST_OLUMSUZLAR in satirlar["Beklenen hata"]
    # Dairesellik ile cipa sinifi BIRBIRININ SATIRINDA DEGIL.
    assert fabrika._CIPA_SINIFI_BEYANI not in satirlar["Dairesellik"]
    assert fabrika._DAIRESELLIK_SEKIZ not in satirlar["Çıpa sınıfı"]


# ---------------------------------------------------------------------------
# ZORUNLU BEYANLAR — SPEC §6.1–6.7 ve §9'un 10./11. kriterleri
# ---------------------------------------------------------------------------


def test_DAIRESELLIK_BEYANI_IKI_PARCALIDIR_ve_IKINCISI_SEKIZINDIR(sevk_panel):
    """§9 KRİTERİ 5'İN İKİNCİ YARISI — ve planın DÜŞEN maddesi burasıydı.

    Spec §6.3'ün gövdesi baştan sona 43″/efemeristir; dairesellik beyanının
    İKİNCİ parçası **§3.6**'da, "PANELDE YAZILACAK" başlığı altında durur.
    "§6.1–6.7'nin tamamı" diye tam bir sayım yazmak ikinci parçayı DÜŞÜRÜR,
    çünkü `Provenance.circularity` tek bir `str | None` alanıdır ve tek
    cümleyle doldurulabilir.

    Üç iddianın ÜÇÜ DE ayrı ayrı aranır — biri düşerse test kırılır.
    """
    dairesellik = dict(sevk_panel.as_lines())["Dairesellik"]
    assert "(a) BU MODUL MERKUR'UN" in dairesellik
    assert "(b) SEKIZ YORUNGESINDE" in dairesellik
    for iddia in (
        "BVP KOSULUNUN TA KENDISIDIR",
        "Phi_T = (Phi_{T/3})^3 CEBIRSEL SONUCUDUR",
        "L = 0 BIR OLCUM DEGILDIR",
        "GIRDI OLARAK KALAN TEK SEY SIMETRI SINIFIDIR",
    ):
        assert iddia in dairesellik, f"sekizin daireselligi eksik: {iddia!r}"
    # 43" tarafi da tam: geri okuma VE efemeris duyarliligi.
    assert "42.980644" in dairesellik and "42.979726" in dairesellik
    assert "yayilim 0.0010320 yay saniyesi" in dairesellik


def test_YOKLUK_BEYANI_PANELDE_OLCULMUS_SAYILARLA(sevk_panel):
    """§6.4 — ve sayılar bu turda YENİDEN ÖLÇÜLDÜ (CODATA 2018 / SI 2019).

    `F_g/F_e = 8.093550856608016e-37` iki PROTON içindir ve bu bir seçimdir:
    elektron-proton çifti için aynı oran 4.4e-40 çıkar. Panel formülü ve
    seçimi yazar; yazmasaydı sayı üç mertebe kayabilirdi.

    KÜTLE SEÇİMİ DE BİR DÜĞMEDİR ve o da yazılıdır: indirgenmiş kütle
    kullanılsaydı taban durum enerjisi TAM YARIYA düşerdi.
    """
    ihmal = dict(sevk_panel.as_lines())["İhmal edilen"]
    assert "MODULLER ARASI BAG -- YOKLUK BEYANI" in ihmal
    assert "8.093550856608016e-37" in ihmal
    assert "6.967737e-65" in ihmal
    assert "9.665562e+129" in ihmal
    assert "SPAN BIR DUGMEDIR" in ihmal
    assert "INDIRGENMIS KUTLE KULLANILMADI" in ihmal


def test_YONTEM_HEDEF_UYUSMAZLIGININ_IKI_TURU_AYRI_SATIRDA(sevk_panel):
    """§9 KRİTERİ 10 — ve karıştırılmalarının bedeli ölçülmüştür.

    A: yöntem hedefi KAPSAMIYOR → düzeltilemez, başka bir KURAM gerekir.
    B: hedef bu modülün DIŞINDA → ertelendi, engel yok.

    A'yı B sanmak "sonraki modülde ekleriz" yalanını üretir; B'yi A sanmak
    yapılabilir işi gereksiz yere gömer. İkisi AYRI girdilerdir ve hiçbiri
    ötekinin içinde geçmez — tek bir metne birleştirilseydi panel "bu iki
    şey aynı türden" derdi.
    """
    a, b = fabrika._YONTEM_HEDEF_A, fabrika._YONTEM_HEDEF_B
    assert a != b
    assert a not in b and b not in a
    assert a in fabrika._GRAV_IHMAL and b in fabrika._GRAV_IHMAL
    ihmal = dict(sevk_panel.as_lines())["İhmal edilen"]
    assert a in ihmal and b in ihmal
    assert "TUR A -- YONTEM HEDEFI KAPSAMIYOR, DUZELTILEMEZ" in a
    assert "TUR B -- HEDEF BU MODULUN DISINDA, ERTELENDI, ENGEL YOK" in b
    # A'nin IKI uyesi: Newtonyen perihel + test parcaciginin geri-tepmesi.
    assert "perihel kaymasini YANLIS uretmez, HIC uretmez" in a
    assert "geri-tepkiyi" in a
    # B'nin uc uyesi.
    for uye in ("Kerr", "isima sonumu", "gelgit deformasyonu"):
        assert uye in b


def test_CIPA_DEGIL_RAFINDA_UC_UYE_ve_UCUNUN_SEBEBI_AYRI(sevk_panel):
    """§9 KRİTERİ 7 — ve Task 5'in notu da panelde.

    Üç üye: Kepler 3/2 (boyutsal olarak zorunlu) · ⟨U⟩/|E| = 2 (viryal) ·
    enerji korunumu (beş kuvvet üssünün hepsinde geçiyor → tautoloji).

    VE PANELİN TAŞIDIĞI DÖRDÜNCÜ CÜMLE: bir çıpa olarak tautoloji olmak,
    **mutasyon kapsaması olmamak demek değildir** — o test bu turda bir
    mutantı öldürdü. Bir çıpanın FİZİK DEĞERİ ile bir testin AYIRT ETME
    GÜCÜ iki ayrı büyüklüktür ve panel ikisini karıştırmaz.
    """
    sinif = dict(sevk_panel.as_lines())["Çıpa sınıfı"]
    assert "Kepler 3/2 ussu" in sinif and "L^3*T^-2" in sinif
    assert "<U>/|E| = 2" in sinif and "viryal" in sinif
    assert "ENERJI KORUNUMU" in sinif and "TAUTOLOJI" in sinif
    assert "TAUTOLOJI OLMAK MUTASYON KAPSAMASI OLMAMAK DEMEK DEGILDIR" in sinif
    # Dort sinif etiketinin dordu de yazili.
    for etiket in ("FIZIK CIPASI", "KAPALI-FORM TUREV", "SEMA CIPASI", "CIPA DEGIL"):
        assert etiket in sinif


def test_KURAM_SECICI_KUTUSU_IKI_UYELI_power_da_ICINDE(sevk_panel):
    """§6.5 — ve Task 5'te bulunan İKİNCİ kuram seçici de kutuda.

    `Theory(gamma, beta)` tek başına yeterli değildi: N-cisim tarafındaki
    `power` (kuvvet üssü) da HANGİ KURAMIN çözüldüğünü seçer, ve `p ≠ 2`de
    potansiyelin kapalı formu bir **SEÇİMDİR** — o seçim adlandırılmadığı
    için planın bir üs tablosu yeniden üretilemedi.
    """
    kutu = dict(sevk_panel.as_lines())["Kuram seçici"]
    assert "`Theory(gamma, beta)`" in kutu
    assert "`power` (kuvvet ussu)" in kutu
    assert "p != 2 BIR YAKLASIM DEGIL BASKA BIR KURAMDIR" in kutu
    assert "U = -m_i*m_j/((p-1)*r^(p-1))" in kutu
    assert "`lam` BIR ALAN DEGIL TURETILMIS BIR OZELLIKTIR" in kutu
    # Bu kosum GR: canli parca bunu soyler.
    assert "bu kosum GENEL GORELILIK cozuyor" in kutu
    assert GR_DEGIL_ISARETI not in kutu


def test_PENCERE_AYAR_BAGIMLILIGINI_ve_UC_PI_BOLU_SEKIZI_TASIYOR(sevk_panel):
    """§6.7 — ve düzeltilmiş sayılar panelde, ESKİLERİ DE TEŞHİSİYLE.

    `(oran − 2)·b/M` Schwarzschild'de `7π/8` = 2.748894, izotropikte
    `π/2` = 1.570796, fark TAM `3π/8` = 1.178097. Panel eski (yanlış)
    sayıları da yazar, çünkü FARKIN DOĞRU KALMASI tam olarak yanıltıcıydı:
    ortak bir kayma farkta sadeleşir.
    """
    pencere = dict(sevk_panel.as_lines())["Pencere"]
    assert "7*pi/8 = 2.748894" in pencere
    assert "pi/2 = 1.570796" in pencere
    assert "3*pi/8 = 1.178097" in pencere
    assert "ortak bir kayma farkta sadelesir" in pencere
    # Bu kosumun kendi pencere degerleri de satirda.
    assert "M = 1, e = 0.3, ayar = schwarzschild" in pencere
    assert "(300000, 1e+06)" in pencere and "(100000, 300000)" in pencere
    assert "N-cisim DOP853, rtol = 1e-13, atol = 1e-15" in pencere


def test_DURUST_OLUMSUZLAR_BESI_DE_PANELDE(sevk_panel):
    """BEYAN EDİLMİŞ, DÜZELTİLMEMİŞ kusurlar SAKLANMAZ.

    Panelin işi ölçülmüş kusurları gizlemek değil TAŞIMAKTIR. Beşi de ayrı
    ayrı aranır; biri sessizce düşerse test kırılır.
    """
    hata = dict(sevk_panel.as_lines())["Beklenen hata"]
    for parca in (
        "TABAN YALNIZ ARITMETIGI OLCER",
        "1/1158000",
        "IKINCI MERTEBE K AYAR-BAGIMLIDIR",
        "CAPRAZ KONTROLUN GORMEDIGI YUZEY OLCULDU",
        "farki BUYUTMEZ, KUCULTUR",
        "TABAN VE KESISIM URETILEMIYOR",
        "IZOTROPIK K'NIN KAPALI FORMU TURETILMEDI",
        "-24*pi*(1 + gamma)^2 OLCULDU ama TURETILMEDI",
    ):
        assert parca in hata, f"durust olumsuz eksik: {parca!r}"


# ---------------------------------------------------------------------------
# GÜRÜLTÜ TABANI — spec §6.6 ve §9 kriteri 8
# ---------------------------------------------------------------------------


def test_BES_TABAN_BES_AYRI_BORUDAN_KARISIK_BORU_YOK(sevk_yuk):
    """§9 KRİTERİ 8 — her sayı KENDİ borusundan, S/G oranıyla.

    Beş satırın beşi de farklı bir `PIPELINE` etiketi taşır ve etiket
    `fn.__name__`den DEĞİL fonksiyonun kendi beyanından okunur. Karışık boru
    olsaydı iki satır aynı etiketi taşırdı.

    ÖLÇÜLEN (sevk edilen pencerede): kareleme 1.594187e-09 · ODE
    1.921807e-06 · bükülme 3.651917e-10 · N-cisim 3.996803e-14 · doğrusal
    harita 0.0 (DEJENERE).
    """
    boru_adlari = [f.pipeline.split(" | ")[0] for f in sevk_yuk.floors]
    assert boru_adlari == [
        "kareleme", "ode", "bukulme", "nbody", "nbody-dogrusal-harita",
    ]
    assert len(set(boru_adlari)) == 5

    beklenen = (1.594187e-09, 1.921807e-06, 3.651917e-10, 3.996803e-14, 0.0)
    for f, hedef in zip(sevk_yuk.floors, beklenen):
        assert f.value == pytest.approx(hedef, rel=1e-6, abs=1e-300)

    # BORU SECICI ETIKETLER de yazili -- `gauge` ve `method` GOVDEYI secer.
    assert "gauge=schwarzschild" in sevk_yuk.floors[0].pipeline
    assert "method=DOP853" in sevk_yuk.floors[1].pipeline
    assert "gauge=schwarzschild" in sevk_yuk.floors[2].pipeline
    assert "method=DOP853" in sevk_yuk.floors[3].pipeline


def test_DEJENERE_BORU_HUKME_GIRMEZ_ama_SATIRI_BASILIR(sevk_panel, sevk_yuk):
    """SPEC 4.3'ÜN KENDİ GÖZLENEBİLİRİ SAKLANMAZ — ve hükmü de bozmaz.

    `third_body_influence` `m2 = 0`da BİT BİT sıfır verir (0/1 canlı basamak)
    ve bu YAPISAL bir dejenereliktir: kaynak terimi `m2` ile çarpıldığı için
    blok tam sıfırdır, limit de kurtarmaz çünkü gürültü sinyalle birlikte
    ölçeklenir. Sevk edilen `circular_frequency_defect` bunun bir
    **İKAMESİDİR** ve gerekçesi tam olarak bu ölçümdür.

    Satır PANELDE DURUR (gerekçesi kaybolmasın diye) ama `converged`
    hükmüne GİRMEZ (girseydi panel hiçbir koşumda tik basamazdı ve tikin
    anlamı kaybolurdu). İkisi de burada kilitli.
    """
    harita = sevk_yuk.floors[-1]
    assert harita.anchor is False
    assert harita.verdict == "DEJENERE"
    assert math.isnan(harita.ratio)
    assert harita.value == 0.0
    assert sevk_panel.converged is True

    tani = dict(sevk_panel.as_lines())["Yakınsama tanısı"]
    assert "DEJENERE" in tani
    assert "S/G OKUNAMADI (boru dejenere)" in tani
    assert "nbody-dogrusal-harita" in tani
    assert "IKAME EDILDI, CIPA DEGIL" in tani
    # `nan` ya da `inf` BIR SAYI GIBI BASILMAZ.
    assert "S/G = nan" not in tani and "S/G = inf" not in tani


def test_CIPA_BORUSU_CURURSE_PANEL_TIK_BASMAZ(sevk_senaryo, sevk_setup, sevk_yuk):
    """`converged` GERÇEKTEN bağlı bir düğmedir — ölçülerek gösterilir.

    Bir bayrağın "her zaman ✓" olduğu bir panel hiçbir şey söylemez. Burada
    tek bir çıpa borusunun hükmü `CURUK` yapılır ve tikin DÜŞTÜĞÜ görülür;
    aynı bozulma çıpa OLMAYAN boruya yapıldığında tik DÜŞMEZ.
    """
    def _panel(floors):
        return provenance_for(
            sevk_senaryo, setup=sevk_setup, floors=floors,
            second_order=sevk_yuk.second_order,
            closed_forms=sevk_yuk.closed_forms,
            eight=sevk_yuk.eight, angles=sevk_yuk.angles,
        )

    saglam = list(sevk_yuk.floors)
    assert _panel(tuple(saglam)).converged is True

    curuk = list(saglam)
    curuk[0] = dataclasses.replace(curuk[0], verdict="CURUK", ratio=0.5)
    assert _panel(tuple(curuk)).converged is False

    # CIPA OLMAYAN boruyu bozmak hukmu DEGISTIRMEZ.
    cipa_degil = list(saglam)
    cipa_degil[-1] = dataclasses.replace(cipa_degil[-1], verdict="CURUK", ratio=0.5)
    assert _panel(tuple(cipa_degil)).converged is True


def test_PROVENANCE_TABANSIZ_PANEL_KURMAYI_REDDEDIYOR(sevk_senaryo, sevk_setup,
                                                      sevk_yuk):
    """Spec 4.3 "pazarlık edilmez" der; kapı MOTORDA durur, testte değil.

    Boş bir taban listesiyle ya da yalnızca çıpa olmayan borularla kurulan
    bir panel sessizce basılmaz: birincisinde spec §6.6 ihlal edilirdi,
    ikincisinde `converged` hükmü TANIMSIZ olurdu.
    """
    ortak = dict(
        setup=sevk_setup, second_order=sevk_yuk.second_order,
        closed_forms=sevk_yuk.closed_forms, eight=sevk_yuk.eight,
        angles=sevk_yuk.angles,
    )
    with pytest.raises(InvalidInputError, match="en az bir gurultu tabani"):
        provenance_for(sevk_senaryo, floors=(), **ortak)
    with pytest.raises(InvalidInputError, match="en az bir CIPA tabani"):
        provenance_for(
            sevk_senaryo, floors=(sevk_yuk.floors[-1],), **ortak
        )


# ---------------------------------------------------------------------------
# SENARYO SINIRI — YÜKLEYİCİ KAPILARI
# ---------------------------------------------------------------------------


def test_SEVK_EDILEN_YAML_PENCERENIN_HER_DUGMESINI_TASIYOR(sevk_senaryo):
    """Sevk edilen senaryo tam olarak bu penceredir — ve varsayılan YOK."""
    s = sevk_senaryo
    assert isinstance(s, GravScenario)
    assert (s.gamma, s.beta) == GR_NOKTASI
    assert s.gauge == "schwarzschild"
    assert s.mass == 1.0
    assert s.eccentricity == 0.3
    assert s.richardson == (3e5, 1e6)
    assert s.second_order_p == 3e5
    assert s.deflection_richardson == (1e5, 3e5)
    assert s.trajectory_p == 100.0
    assert s.trajectory_points == 2000
    assert (s.nbody_rtol, s.nbody_atol) == (1e-13, 1e-15)
    assert s.nbody_method == "DOP853"
    # HICBIR ALANIN VARSAYILANI YOK -- pencereyi yazmayi unutan bir senaryo
    # sessizce "tekrarlanabilir" gorunemez.
    alanlar = {f.name: f for f in dataclasses.fields(GravScenario)}
    varsayilanli = [
        ad for ad, f in alanlar.items()
        if f.default is not dataclasses.MISSING
        or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
    ]
    assert varsayilanli == []


def test_LAM_SENARYODA_BIR_ANAHTAR_DEGIL(tmp_path):
    """İKİ ALAN AYNI ŞEYİ BELİRLERSE SESSİZ BİR KAZANAN DOĞAR.

    Tek kaynak `(gamma, beta)`dır; `lam` motorda türetilmiş bir
    `@property`dir. YAML'da bir `lam` anahtarı kabul edilseydi `gamma`/`beta`
    ile çelişebilir ve hangisinin kazandığı adlandırılmamış yeni bir pencere
    düğmesi olurdu.
    """
    yol = _yaz(tmp_path, SEVK_YAML + "lam: 3.0\n")
    with pytest.raises(ScenarioError, match="Bilinmeyen anahtar"):
        load_scenario(yol)


@pytest.mark.parametrize("anahtar, deger, desen", [
    ("gauge", "kerr", "Bilinmeyen ayar"),
    ("nbody_method", "Verlet", "Bilinmeyen N-cisim integrat"),
    ("mass", "-1.0", "'mass' pozitif"),
    ("eccentricity", "0.0", "eccentricity"),
    ("eccentricity", "1.0", "eccentricity"),
    ("second_order_p", "0.0", "'second_order_p' pozitif"),
    ("trajectory_p", "-3.0", "'trajectory_p' pozitif"),
    ("trajectory_points", "1", "en az 2"),
    ("nbody_rtol", "2.0", "'nbody_rtol'"),
    ("nbody_atol", "0.0", "'nbody_atol'"),
    ("richardson", "[1.0e6, 1.0e6]", "KESİN ARTAN"),
    ("richardson", "[1.0e6, 3.0e5]", "KESİN ARTAN"),
    ("deflection_richardson", "[-1.0, 3.0]", "pozitif ve sonlu"),
    ("richardson", "3.0e5", "iki elemanlı bir liste"),
])
def test_YUKLEYICI_KAPILARI_YAML_TARAFINDA_yakaliyor(tmp_path, anahtar, deger, desen):
    """Yükleyici motorun kapılarını BİLEREK tekrarlar — KATMAN AYRIMI.

    Burada yakalanan şey YAML'ın hatasıdır ve hata hiçbir kareleme, hiçbir
    `solve_ivp` adımı koşmadan ÖNCE verilir; motorda yakalanan şey ÇAĞRININ
    hatasıdır. İkisi ayrı tür fırlatır (`ScenarioError` ile
    `InvalidInputError`) ve bu fark burada kilitlidir.
    """
    yol = _yaz(tmp_path, _degistir(anahtar, deger))
    with pytest.raises(ScenarioError, match=desen):
        load_scenario(yol)


def test_EKSIK_ANAHTAR_SESSIZCE_VARSAYILANA_DUSMUYOR(tmp_path):
    """Bir pencere düğmesini yazmayı unutan senaryo REDDEDİLİR."""
    metin = "\n".join(
        s for s in SEVK_YAML.splitlines() if not s.startswith("nbody_atol:")
    )
    yol = _yaz(tmp_path, metin + "\n")
    with pytest.raises(ScenarioError, match="Eksik anahtar"):
        load_scenario(yol)


def test_GR_OLMAYAN_KURAM_REDDEDILMEZ_ama_PANEL_SOYLER(tmp_path):
    """`gamma`/`beta` bir KAPI DEĞİL bir KURAM SEÇİCİDİR.

    GR olmayan bir seçim reddedilseydi §4.3'ün hata çubuğu makinesi
    (`λ = 0`, yani `β = 2 + 2γ`) senaryo yolundan HİÇ görünmezdi. Yükleyici
    kabul eder; panel kendi satırında ne çözüldüğünü YAZAR.
    """
    yol = _yaz(tmp_path, _degistir("gamma", "0.0"))
    s = load_scenario(yol)
    assert (s.gamma, s.beta) != GR_NOKTASI
    setup = build_grav(s)
    assert setup.theory.lam == 1.0
    yuk = build_grav_payload(setup)
    p = provenance_for(
        s, setup=setup, floors=yuk.floors, second_order=yuk.second_order,
        closed_forms=yuk.closed_forms, eight=yuk.eight, angles=yuk.angles,
    )
    kutu = dict(p.as_lines())["Kuram seçici"]
    assert GR_DEGIL_ISARETI in kutu
    assert "lam = 1 (GR icin 3 olmali)" in kutu
    # VE CIPA KAYIYOR: 2*pi*lam = 2*pi, 6*pi DEGIL.
    assert yuk.coefficient == pytest.approx(2 * math.pi, rel=1e-9)
    # VE UC KAPALI-FORM TUREV BU KURAM NOKTASINDA YOK -- SESSIZCE ATLANMIYOR.
    # (gamma, beta) = (0, 1)'de beta > gamma ve 1PN'de kesilmis A(u) ufuk
    # yapmiyor; `grav_obs` orada inanilir gorunen bir sayi dondurmek yerine
    # REDDEDIYOR. Panel yoklugu YAZAR.
    for ad in KURAMA_BAGLI_KAPALI_FORMLAR:
        assert ad not in yuk.closed_forms
    assert set(yuk.closed_forms) == {"mu_c", "apsidal_us"}
    tani = dict(p.as_lines())["Yakınsama tanısı"]
    assert tani.count(KAPALI_FORM_YOK) == 3


# ---------------------------------------------------------------------------
# MOTOR + YÜK — İKİ ÇÖZÜCÜ AYRI KALIYOR
# ---------------------------------------------------------------------------


def test_BUILD_GRAV_IKI_COZUCUYU_AYRI_KOSUYOR(sevk_setup):
    """§4.1'in ön koşulu senaryo yolunda da geçerli.

    `build_grav` ODE yörüngesini ve sekizin BVP'sini YAN YANA koşar ama
    BİRLEŞTİRMEZ: ikisinin ortak tek şeyi senaryonun kendisidir. BVP
    `Theory`yi hiç görmez ve senaryonun `gamma`/`beta`sından etkilenmez —
    burada ölçülür.
    """
    t = sevk_setup.trajectory
    assert len(t.phi) == len(t.u) == len(t.r) == 2000
    assert float(t.phi[0]) == 0.0
    kayma = float(t.phi[-1]) - 2 * math.pi
    assert kayma == pytest.approx(1.974750e-01, rel=1e-5)
    assert float(t.u[0]) == pytest.approx(1.3 / 100.0, rel=1e-15)
    assert np.all(t.r == 1.0 / t.u)

    fe = sevk_setup.figure_eight
    assert fe.vx == pytest.approx(0.347116888119, abs=1e-12)
    assert fe.vy == pytest.approx(0.532724945388, abs=1e-12)
    assert fe.period == pytest.approx(6.325913982926, abs=1e-12)
    assert abs(fe.residual) < 1e-12

    # GR OLMAYAN bir kuramda YORUNGE degisiyor ama BVP AYNI kaliyor.
    gr_degil = build_grav(_SENARYO_GR_DEGIL())
    assert float(gr_degil.trajectory.phi[-1]) != float(t.phi[-1])
    assert gr_degil.figure_eight.vx == fe.vx
    assert gr_degil.figure_eight.period == fe.period


def test_YUK_CIPA_SAYILARINI_URETMIYOR_OKUYOR(sevk_yuk):
    """`viz/grav_frames.py` fizik ÜRETMEZ; okur, çevirir ve toplar.

    Bu koşumda okunan beş sayı `tests/grav/` altında bağımsız olarak
    ölçülmüş değerlerle birebir aynıdır — yani senaryo yolu çıpaları
    YENİDEN ÖLÇMEZ, yeniden OKUR.
    """
    assert sevk_yuk.coefficient == 18.84955591968662
    assert sevk_yuk.second_order == 85.24858543879077
    assert sevk_yuk.deflection == 1.9999999996887523
    assert sevk_yuk.eight == pytest.approx(9.237681250723917, abs=1e-14)
    assert sevk_yuk.angles[0] == pytest.approx(1.872970598421, abs=1e-11)
    assert sevk_yuk.angles[1] == pytest.approx(0.052921540137, abs=1e-11)
    assert sevk_yuk.closed_forms["isco_r_over_rs"] == pytest.approx(3.0, abs=1e-14)
    assert sevk_yuk.closed_forms["foton_r_over_rs"] == 1.5
    assert sevk_yuk.closed_forms["b_krit_over_M"] == 3 * math.sqrt(3)
    # KARTEZYEN IZ BU DOSYADA URETILIR -- `Trajectory` bir egri saklar, resim degil.
    assert sevk_yuk.x.shape == sevk_yuk.y.shape == (2000,)
    assert np.allclose(np.hypot(sevk_yuk.x, sevk_yuk.y), sevk_yuk.r)


# ---------------------------------------------------------------------------
# UÇTAN UCA
# ---------------------------------------------------------------------------


def test_CLI_UCTAN_UCA(tmp_path):
    """`python -m qlab.run scenarios/grav_tarama.yaml -o grav.npz`.

    ÖLÇÜLDÜ: taramanın kendisi ≈ 0.7 s, yorumlayıcı başlangıcıyla birlikte
    alt süreç ≈ 2 s. Deponun `slow` eşiği "kendi `setup+call` süresi 2 s ve
    üstünde" der; bu test ölçülerek `slow` BIRAKILMADI çünkü tek başına
    koşan uçtan uca dal `-m "not slow"` koşumunda da koşmalıdır — Modül
    6'nın fizik çıpalarının bir kısmı zaten `slow` işaretli ve o koşumda
    atlanıyor, panelin kendisi de atlansaydı "not slow" koşumu senaryo
    sınırını HİÇ sınamazdı. (Modül 5'in `test_CLI_UCTAN_UCA`sıyla aynı
    gerekçe.)

    ON panel satırının ONU da aranır. Metinler bu dosyada TEKRARLANMAZ;
    `factory` modülünden okunur, yani beyan düzeltilirse test onu
    KENDİLİĞİNDEN takip eder ve ikinci bir kopya doğmaz.
    """
    cikti = tmp_path / "grav.npz"
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
        if cumle in KOSUMA_BAGLI:
            continue  # sevk edilen kosum GR; ikisi de asagida ayrica sinaniyor
        assert cumle in sonuc.stdout, f"beyan EKRANA basilmadi: {cumle[:50]!r}"
    for isaret in KOSUMA_BAGLI:
        assert isaret not in sonuc.stdout
    assert "sonlu pencere: bu kosum kareleme tarafinda" in sonuc.stdout
    # DEJENERE boru TABLODA DA saklanmiyor.
    assert "DEJENERE" in sonuc.stdout
    assert "(ÇIPA DEĞİL)" in sonuc.stdout
    assert "S/G OKUNAMADI" in sonuc.stdout

    with np.load(cikti, allow_pickle=False) as veri:
        assert veri["phi"].shape == (2000,)
        assert np.allclose(np.hypot(veri["x"], veri["y"]), veri["r"])
        assert float(veri["coefficient"]) == 18.84955591968662
        assert float(veri["eight"]) == pytest.approx(9.237681250723917, abs=1e-14)
        assert list(veri["floor_verdicts"]) == [
            "SAGLAM", "SAGLAM", "SAGLAM", "SAGLAM", "DEJENERE",
        ]
        assert list(veri["floor_is_anchor"]) == [True, True, True, True, False]
        assert math.isnan(float(veri["floor_ratios"][-1]))
        kaynak = json.loads(str(veri["provenance_json"]))
    # `.npz`ye YAZILAN beyan ile EKRANA basilan beyan AYNI kaynaktan.
    assert kaynak["anchor_class"] == fabrika._CIPA_SINIFI_BEYANI
    assert fabrika._DAIRESELLIK_SEKIZ in kaynak["circularity"]
    assert fabrika._KURAM_SECICI_BEYANI in kaynak["theory_selector"]
    assert kaynak["convergence_diagnosis"].startswith("[CANLI")
    assert kaynak["converged"] is True


def test_SEVK_EDILEN_YAML_HICBIR_CIPA_DEGERINI_TASIMIYOR():
    """Senaryo dosyası bir ÇIPA KURMAZ, panelin beyanını gösterir.

    `6π`, `9.2376…` ya da `1.8729…` gibi bir çıpa değeri YAML'da geçseydi
    sevk edilen senaryo cevabı taşıyor olurdu. Geçmiyor — ve bu bir niyet
    değil bir SAYIMDIR (Modül 5'in "Bessel kökü YAML'da hiç geçmiyor"
    kilidinin aynısı).
    """
    for yasak in ("18.8495", "9.2376", "1.87297", "0.05292", "5.19615",
                  "0.038520", "6.32591", "0.347116", "0.532724"):
        assert yasak not in SEVK_YAML, f"cipa degeri YAML'da: {yasak}"
