"""`kind: bridge` senaryosu, panel beyanı ve uçtan uca koşum — Modül 7 / Task 5,
modülü kapatan görev.

SINIF: TRANSKRİPSİYON DENETİMİ + ŞEMA ÇIPASI DENETİMİ. Bu dosyanın hiçbir
testi yeni bir fizik iddiası kurmaz; senaryo sınırının (YAML → dataclass →
tarayıcı/taban → panel) doğru bağlandığını ve panelin **spec §6'nın sekiz
zorunlu beyanını** — artı dürüst olumsuzları — bastığını ölçer. Modülün
sayıları başka yerdedir: kova tablosu `test_kova_tablosu.py`de, kenar
tabanının iki sayısı `test_kenar_tabani.py`de, düşen iki aday
`test_dusen_adaylar.py`de, kütle körlüğü `tests/md/test_aimd.py`de.

BU DOSYANIN EN ÖNEMLİ TESTİ BİR SAYI DEĞİL BİR SAYIM DENETLER, VE İKİ
KATMANLIDIR (Modül 6'nın kalıbı):

1. `test_BEYAN_TEK_KAYNAKTAN_geliyor` — on dokuz beyan cümlesinin her biri
   `src/` ağacında TAM BİR KEZ geçer ve tek kaynağı `qlab/bridge_map.py`dir
   (bir motor fabrikası DEĞİL — bu modülün motoru yok; saf tanım katmanı).
   `run.py` ve `viz/bridge_frames.py` hiçbirini yazmaz.
2. `test_BEYAN_PARCASININ_KOPYASI_da_YASAK` — tam cümle sayımı YETMEZ:
   Modül 5'te bir mutant 55 karakterlik bir PARÇA kopyalayarak tam-cümle
   sayımından kaçmıştı.

===========================================================================
PARÇA EŞİĞİ BU MODÜLDE YENİDEN ÖLÇÜLDÜ — DEVRALINMADI (03.09.2026)
===========================================================================

`src/` ağacındaki (`bridge_map.py` hariç, 2983 dize sabiti) her sabit bu
modülün on dokuz beyan cümlesinin içinde arandı:

    eşik  4 -> 120 eşleşme   (" ve ", " bir ", " bu ", …)
    eşik  6 ->  30 eşleşme
    eşik  8 ->  17 eşleşme   ("angstrom", "coupling", "downfold", "harmonik" …)
    eşik 12 ->   0 eşleşme
    eşik 16 ->   0 eşleşme
    eşik 20 ->   0 eşleşme

İLK TARAMADA 12/16/20'DE İKİ EŞLEŞME VARDI ve ikisi de aynı sabitti:
`'miknatislanma serisi'` (`observable/spin_obs.py:172` ve `:212`, bir hata
mesajı etiketi). Bu bir kopya DEĞİL, beyan cümlesinin içinde geçen
paylaşılan bir TERİMdi — Modül 6'nın 12'deki `foton kuresi` durumunun
aynısı. Beyan `` `m` serisi `` diye yeniden yazıldı ve tarama 12'den itibaren
sıfıra indi. Yanlış pozitif tabanı bu modülde 8 ile 12 arasındadır; **16
seçildi** — Modül 5 (4 → 207, 8 → 6, 12 → 4, 16 → 0) ve Modül 6 (4 → 92,
8 → 13, 12 → 2, 16 → 0) ile aynı sayı, ama devralınmadı, ölçüldü.

===========================================================================
SEVK EDİLEN KOŞUMUN ÖLÇÜLEN SAYILARI (03.09.2026, bu depoda, bu makine)
===========================================================================

Pencere: N = 200, 6 periyot, genlikler (0.01, 0.02, 0.04), A2 ikizi
a = 0.9137, h = 0.002, r_e = 1.3459193602, K = 0.5729363, SCF = 0.

    kova sayımı            1 / 4 / 8 / 0 / 8   (toplam 21)
    kod kenarı             md -> hf, tb -> hf  (`ast`, 48 dosya)
    `qlab.units` ekseni    hf=1 md=3 tb=2, ötekiler 0 -> {hf, md, tb}
    A1 (aritmetik)         3.3986707e-05 cm⁻¹  S/G 1.6128e+08  SAGLAM  7/7
    A2 (hata-çubuğu)       1.2025992e-02 cm⁻¹  S/G 4.5578e+05  SAGLAM  1/1
    gerçek hata            1.4472809e-02 cm⁻¹  -> gerçek/A1 425.84, A2/gerçek 0.8309
    converged              True (tablo kodla tutarlı VE iki taban SAGLAM)

Toplam koşum ≈ 0.7 s (A1 0.50 s + A2 0.12 s + tarama 0.05 s); HF hiç
çözülmez.

(Yukarıdaki blok 03.09.2026'nın KAYDIDIR. İki satırı artık geçerli DEĞİL:
22.09.2026'da `tdse → md`, 24.09.2026'da `md → tb` kenarları yazıldı —
bugün kova sayımı 3 / 4 / 8 / 0 / 6, kod kenarı dört (md -> hf, md -> tb,
tb -> hf, tdse -> md) ve yapı `hf - md - tb` üçgeni + `md - tdse` kuyruğu;
ikisi de `test_6_1`/`test_6_2`de canlı panele karşı kilitli. Units ekseni,
A1/A2 ve gerçek hata satırları değişmedi — `test_6_2`/`test_6_3` onları
bugün de aynı sayılarla sınıyor.)

SPEC §4'ÜN BİR SAYISI DAHA ÇÜRÜDÜ — YANLIŞ DEĞİL, PENCERESİZ: "6-31G'de
k = 0.411660 → ω = 4646.18 cm⁻¹" sevk edilen `r_e`de (STO-3G'nin
1.3459193602'si) ÜRETİLEMİYOR: ölçülen 4921.2588 cm⁻¹ (k = 0.461848).
Aynı sayı 6-31G'nin KENDİ `r_e`sinde (Brent, 1.3794239789 bohr)
4646.186940 çıkıyor. Spec hangi `r_e`de ölçtüğünü yazmamıştı; iki pencere
275 cm⁻¹ ayrışıyor. İkisi de burada ucuz (statik) kilitli.

`slow` KARARI ÖLÇÜLDÜ (`--durations=0`, dosyanın sonundaki nota bakınız):
eşik 2 s (Task 2'nin kuralı). Tek `slow` test HF borusunun ×1.001 kuvvet
mutantını canlı koşturandır (~28 s); `test_CLI_UCTAN_UCA` eşiğin üstünde
olduğu hâlde BİLEREK işaretlenmedi — gerekçesi kendi dokstringinde.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import qlab.bridge_map as bm
import qlab.engine.grav.factory as grav_fabrika
from qlab.bridge_map import (
    KOVA_SAYILARI,
    KOVA_TABLOSU,
    KOVALAR,
    MODULLER,
    Kova,
    _HF_TB_OZDESLIK_BAGIL,
    _TB_SPIN_M_FARKI,
    bridge_declaration,
    olculen_kenarlar,
    olculen_units_ekseni,
    tablo_tutarsizliklari,
)
from qlab.engine.md.forcefields import HartreeFockGradient
from qlab.engine.md.state import ForceField
from qlab.errors import InvalidInputError, ScenarioError
from qlab.observable.bridge_obs import (
    GERCEK_HATA_CM,
    MU_H2,
    PENCERE_ANAHTARLARI,
    R_E_BOHR,
    _a2_uydurmasi,
    _omega_a,
    edge_noise_floor,
)
from qlab.scenario.loader import _KNOWN_KINDS, load_scenario
from qlab.scenario.schema import BridgeScenario
from qlab.units import HARTREE_TO_WAVENUMBER
from qlab.viz.bridge_frames import (
    BridgePayload,
    build_bridge_payload,
    provenance_for,
)

PROJE_KOKU = Path(__file__).resolve().parents[2]
KAYNAK = PROJE_KOKU / "src" / "qlab"
SEVK_EDILEN = PROJE_KOKU / "scenarios" / "kopru_haritasi.yaml"
SEVK_YAML = SEVK_EDILEN.read_text(encoding="utf-8")

#: Panelin on satırı, SIRASIYLA — Modül 6'nın paneliyle aynı on etiket.
#: `Kuram seçici` burada baz setidir (HF'in kuram düzeyini seçer); gerekçesi
#: `bridge_map._KURAM_SECICI_BEYANI`nda ve aşağıdaki testte ölçülüdür.
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
#: SEÇİLMEDİ, ÖLÇÜLDÜ — tarama dosya dokstringindedir (4 → 120, 6 → 30,
#: 8 → 17, 12 → 0, 16 → 0, 20 → 0).
_BEYAN_PARCASI_ESIGI = 16

#: 30.08.2026'da (HEAD = b221a9c) ölçülen ve Task 5'te yeniden taranan
#: çapraz-motor kenar kümesi ve `qlab.units` ekseni.
#: 22.09.2026: ÜÇÜNCÜ kenar yazıldı (`engine/tdse/from_md.py` -> `engine/md`),
#: yapı yıldızdan ZİNCİRE döndü: `tb - hf - md - tdse`.
#: 24.09.2026: DÖRDÜNCÜ kenar yazıldı (`engine/md/tb_kuvvet.py` -> `engine/tb`),
#: zincirin `tb` ve `md` düğümleri bağlandı: `hf - md - tb` ÜÇGENİ + `md - tdse`.
BEKLENEN_KENARLAR = {("md", "hf"), ("md", "tb"), ("tb", "hf"), ("tdse", "md")}
BEKLENEN_EKSEN = {"em": 0, "grav": 0, "hf": 1, "md": 3, "spin": 0, "tb": 2, "tdse": 0}

#: PENCERE DÜĞMESİNİN ÖLÇÜLEN TEPKİSİ (03.09.2026, bu makine), N = 200 → 400.
#: Plan ve `test_kenar_tabani.py` bu oranı YAZMIYORDU: oradaki 14.89 `a = 0`
#: noktasınındır, A1 ise merdivenin `a = 0.03` TEPESİNDEN okunur ve orada oran
#: 15.18 çıktı — ölçüldü, devralınmadı. VE A2 `dt`YE KÖR (1.0008): iki sayı
#: iki ayrı düğmeye tepki veriyor — A1 aritmetiği, A2 fiziği ölçer. Bu,
#: "iki sayı ayrı sorulara cevaptır" hükmünün ölçülmüş ikinci kanıtıdır.
A1_N200_N400_ORANI = 15.17834385332131
A2_N200_N400_ORANI = 1.00079981980212


class _Olcekli:
    """Kuvveti sabit bir çarpanla bozan sarmalayıcı — ×1.001 mutant düğmesi.

    `test_kenar_tabani.py`dekinin kopyası ve bu bilinçli: paylaşılan bir
    yardımcı iki tarafı aynı yönde bozardı. Enerjiye DOKUNMAZ; yani statik
    yol (`_omega_a`) bunu göremez — tam da ölçülen şey bu.
    """

    def __init__(self, alan: ForceField, c: float) -> None:
        self.alan = alan
        self.c = c

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        enerji, kuvvet = self.alan(positions)
        return enerji, self.c * kuvvet


@pytest.fixture(scope="module")
def sevk_senaryo() -> BridgeScenario:
    return load_scenario(SEVK_EDILEN)


@pytest.fixture(scope="module")
def sevk_yuk(sevk_senaryo) -> BridgePayload:
    return build_bridge_payload(sevk_senaryo)


@pytest.fixture(scope="module")
def sevk_panel(sevk_yuk):
    return provenance_for(sevk_yuk)


def _yaz(tmp_path, metin: str) -> Path:
    yol = tmp_path / "kopru.yaml"
    yol.write_text(metin, encoding="utf-8")
    return yol


def _degistir(anahtar: str, deger: str) -> str:
    """Sevk edilen YAML'ın tek bir anahtarını değiştirir — satır başına
    çapalı (yorum bloğunda aynı sözcük geçiyor; düz `replace` yanlış yeri
    değiştirirdi, Modül 6'da ölçüldü)."""
    satirlar, bulundu = [], False
    for s in SEVK_YAML.splitlines():
        if s.startswith(f"{anahtar}:"):
            satirlar.append(f"{anahtar}: {deger}")
            bulundu = True
        else:
            satirlar.append(s)
    assert bulundu, f"{anahtar} sevk edilen YAML'da bulunamadı"
    return "\n".join(satirlar) + "\n"


def _kaynak_metni(yol: Path) -> str:
    """Dosya metni, BİTİŞİK DİZE SABİTLERİ BİRLEŞTİRİLMİŞ hâlde
    (`test_grav_senaryo.py`nin ölçülmüş tuzağı: `"a "\\n"b"` kaynakta iki
    parça, çalışma zamanında tek dize)."""
    return re.sub(r'"\s*\n\s*"', "", yol.read_text(encoding="utf-8"))


def _tum_beyanlar() -> tuple[str, ...]:
    return (
        (bm._YONTEM,)
        + bm._YAKLASIMLAR
        + bm._IHMAL
        + (
            bm._KENAR_TABANI_BEYANI,
            bm._DURUST_OLUMSUZLAR,
            bm._DUSEN_ADAYLAR_BEYANI,
            bm._CIPA_SINIFI_BEYANI,
            bm._MU_KORLUGU_BEYANI,
            bm._KURAM_SECICI_BEYANI,
            bm._PENCERE_BEYANI,
            bm._YOKLUK_HARITASI_BEYANI,
            bm._YAPI_BEYANI,
        )
    )


def _girdi(kaynak: str, hedef: str):
    esler = [f for f in KOVA_TABLOSU if f.cift == tuple(sorted((kaynak, hedef)))]
    assert len(esler) == 1
    return esler[0]


def _panel_metni(panel) -> str:
    return " ".join(deger for _, deger in panel.as_lines())


# ---------------------------------------------------------------------------
# BEYANIN TEK KAYNAĞI — İKİ KATMANLI KİLİT
# ---------------------------------------------------------------------------


def test_BEYAN_TEK_KAYNAKTAN_geliyor():
    """BU DOSYANIN EN ÖNEMLİ TESTİ. Beyan metninin İKİ KOPYASI OLAMAZ.

    Kilit iki yönlüdür: `src/` ağacında her cümle tam BİR kez (kaynakta
    kopya yok — bu, `run.py::_run_bridge`ın ve `viz/bridge_frames.py`nin
    beyan metnini kendisi yazmasını YASAKLAR) ve `tests/` ağacında, BU DOSYA
    hariç, sıfır kez.

    TEK KAYNAK BİR MOTOR FABRİKASI DEĞİL, `qlab/bridge_map.py`dir: bu
    modülün motoru yok, beyanı saf tanım katmanında durur ve `viz` yalnızca
    sayıları toplayıp `panel_metinleri()`ni çağırır. Katman sözleşmesi
    bunun tersini de yasaklar (`bridge_map` `observable`ı import edemez;
    `tests/test_architecture.py`de ölçülü).
    """
    beyanlar = _tum_beyanlar()
    assert len(beyanlar) == 19, "beyan sayısı değişti; testi bilerek güncelleyin"
    bu_dosya = Path(__file__).resolve()
    for cumle in beyanlar:
        src_sayim = sum(
            _kaynak_metni(yol).count(cumle)
            for yol in (PROJE_KOKU / "src").rglob("*.py")
        )
        assert src_sayim == 1, f"src/ ağacında {src_sayim} kopya: {cumle[:60]!r}"
        test_sayim = sum(
            _kaynak_metni(yol).count(cumle)
            for yol in (PROJE_KOKU / "tests").rglob("*.py")
            if yol.resolve() != bu_dosya
        )
        assert test_sayim == 0, f"BAŞKA bir test dosyasında kopya var: {cumle[:60]!r}"
    # Tek kaynağın DOSYASI da kilitli: on dokuzu da bridge_map'te.
    kaynak = _kaynak_metni(KAYNAK / "bridge_map.py")
    for cumle in beyanlar:
        assert cumle in kaynak


def test_BEYAN_PARCASININ_KOPYASI_da_YASAK():
    """TAM CÜMLE SAYIMININ ÖLÇÜLMÜŞ DELİĞİ — ve bu modülde YENİDEN ÖLÇÜLDÜ.

    Korunan şey "aynı cümle iki kez yazılmasın" değil, **panelin iki farklı
    şey iddia edememesi**dir. Kural parça seviyesindedir: `src/` ağacında
    (`bridge_map.py` hariç) hiçbir dize sabiti — dokstringler ve f-string
    parçaları dahil — bir beyan cümlesinin İÇİNDE geçemez.

    EŞİK DEVRALINMADI, YENİDEN ÖLÇÜLDÜ — tarama dosya dokstringindedir; ve
    ilk taramanın iki eşleşmesi (spin tarafının `'miknatislanma serisi'`
    etiketi) beyan yeniden yazılarak sıfırlandı, eşik gevşetilerek değil.
    """
    beyanlar = _tum_beyanlar()
    kaynak_yolu = (KAYNAK / "bridge_map.py").resolve()
    kopyalar = []
    for yol in (PROJE_KOKU / "src").rglob("*.py"):
        if yol.resolve() == kaynak_yolu:
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
    assert kopyalar == [], f"beyan PARÇASI kopyalanmış: {kopyalar}"
    # Kilit boş bir kural değil: eşiğin ALTINDA gerçekten paylaşılan
    # terimler var (ölçüldü, 8'de 17) — kural onları bilerek serbest bırakır.
    assert _BEYAN_PARCASI_ESIGI == 16


def test_BRIDGE_DECLARATION_PANELIN_TAMAMINI_TASIYOR(sevk_panel):
    """`bridge_declaration()` ile panel AYNI sabitleri okur — iki kopya YOK.

    Sağ taraf `provenance.as_lines()`, yani `Provenance`ın ayrı alanlarına
    dağıtılmış hâl; sol taraf tek dizge. İkisinin aynı on dokuz cümleyi
    taşıması, dağıtımın hiçbir beyanı DÜŞÜRMEDİĞİNİ söyler — Modül 6'nın
    planında dairesellik beyanının ikinci parçası tam böyle düşmüştü.
    """
    metin = bridge_declaration()
    panel = _panel_metni(sevk_panel)
    for cumle in _tum_beyanlar():
        assert cumle in metin, f"bridge_declaration eksik: {cumle[:60]!r}"
        assert cumle in panel, f"panel eksik: {cumle[:60]!r}"
    # Canlı sayılar bridge_declaration'da YOK, panelde VAR.
    assert "[CANLI -- bu kosum]" not in metin
    assert "[CANLI -- bu kosum]" in panel


def test_MODULUN_ETIK_CEKIRDEGI_metinde_YAZILI(sevk_panel):
    """Anahtar kelime değil TAM CÜMLE PARÇASI aranır; parçalar burada
    BAĞIMSIZ yazılmıştır (bitişik sabitlerin yalnız ilk satırını değiştiren
    mutantlar kelime testinden kaçıyordu — `test_tb_scenario.py`de ölçüldü).
    On dört parça: sekizi spec §6'nın sekiz zorunlu beyanına birebir."""
    metin = _panel_metni(sevk_panel)
    for parca in (
        "SIFIR YENI FIZIK",
        "Kova tablosu ELLE LISTELENMEZ, KODDAN TURETILIR",
        "YOKLUK HARITASI: yedi motorun 21 cifti BES kovada",
        "OLCULMEDI AYRI BIR KOVADIR",
        "YAPININ SEKLI: bir MERDIVEN DEGIL -- `hf - md - tb` UCGENI arti "
        "`md - tdse` KUYRUGU",
        "KENAR TABANI IKI SAYIDIR VE IKISI AYRI ETIKET TASIR",
        "YALNIZ A1 YAZILSAYDI PANEL YALAN SOYLERDI",
        "CIPA SINIFI: `hf -> md` kenari bir SEMA CIPASIDIR, fizik cipasi DEGIL",
        "`omega_A` SINAVDAN CIKARILDI",
        "MEVCUT CIPANIN KUTLE KORLUGU",
        "DUSEN IKI ADAY, TAUTOLOJI OLARAK, PENCERESIYLE",
        "YONTEM/HEDEF UYUSMAZLIGI TUR A -- BU BORUDA DUZELTILEMEZ",
        "YONTEM/HEDEF UYUSMAZLIGI TUR B -- ERTELENDI",
        "DURUST OLUMSUZLAR -- BEYAN EDILDI, DUZELTILMEDI, PANEL TASIR",
    ):
        assert parca in metin, f"beyan kayıp: {parca!r}"


# ---------------------------------------------------------------------------
# PANELİN ON SATIRI
# ---------------------------------------------------------------------------


def test_PANEL_ON_SATIRIN_ONUNU_da_SIRASIYLA_basiyor(sevk_panel):
    """Spec §6'nın beyanları, HER BİRİ KENDİ SATIRINDA — etiket listesi tam
    kilitli; bir satır kaybolursa ya da yeri değişirse test kırılır."""
    etiketler = [etiket for etiket, _ in sevk_panel.as_lines()]
    assert etiketler == PANEL_SATIRLARI
    for etiket, deger in sevk_panel.as_lines():
        assert deger.strip(), f"{etiket} satırı BOŞ"
    assert sevk_panel.stability is None and sevk_panel.external_stability is None
    assert sevk_panel.sampling is None


def test_YAKLASIMLAR_IHMAL_DAIRESELLIK_SINIF_KURAM_PENCERE_KARISMIYOR(sevk_panel):
    """Altı metin AYNI panelde ama AYNI SATIRDA değil.

    Tersine çevrilmiş bir atama (`approximations=_IHMAL`) bütün cümleleri
    hâlâ basardı ve "beyan var mı" diye bakan her test yeşil kalırdı.
    Ayıran şey cümlelerin HANGİ satırda olduğudur.
    """
    satirlar = dict(sevk_panel.as_lines())
    for cumle in bm._YAKLASIMLAR:
        assert cumle in satirlar["Yaklaşımlar"]
        assert cumle not in satirlar["İhmal edilen"]
    for cumle in bm._IHMAL:
        assert cumle in satirlar["İhmal edilen"]
        assert cumle not in satirlar["Yaklaşımlar"]
    assert bm._DUSEN_ADAYLAR_BEYANI == satirlar["Dairesellik"]
    assert bm._CIPA_SINIFI_BEYANI in satirlar["Çıpa sınıfı"]
    assert bm._MU_KORLUGU_BEYANI in satirlar["Çıpa sınıfı"]
    assert bm._KURAM_SECICI_BEYANI == satirlar["Kuram seçici"]
    assert bm._PENCERE_BEYANI in satirlar["Pencere"]
    assert bm._KENAR_TABANI_BEYANI in satirlar["Beklenen hata"]
    assert bm._DURUST_OLUMSUZLAR in satirlar["Beklenen hata"]
    assert bm._YOKLUK_HARITASI_BEYANI in satirlar["Yakınsama tanısı"]
    assert bm._YAPI_BEYANI in satirlar["Yakınsama tanısı"]
    assert bm._YONTEM in satirlar["Yöntem"]
    # Dairesellik ile çıpa sınıfı BİRBİRİNİN SATIRINDA DEĞİL.
    assert bm._CIPA_SINIFI_BEYANI not in satirlar["Dairesellik"]
    assert bm._DUSEN_ADAYLAR_BEYANI not in satirlar["Çıpa sınıfı"]
    # Beyan cümleleri YAKINSAMA satırına sızmaz (orada yalnız tanılar var).
    for cumle in _tum_beyanlar():
        assert cumle not in satirlar["Yakınsama"]


# ---------------------------------------------------------------------------
# SPEC §6 — SEKİZ ZORUNLU BEYAN, HER BİRİ SAYISIYLA
# ---------------------------------------------------------------------------


def test_6_1_YOKLUK_HARITASI_21_cift_bes_kova_MERTEBELERLE(sevk_yuk, sevk_panel):
    """§6.1 — 21 çift, beş kova (bugün 3/4/8/0/6; Modül 7 kapanışında
    1/4/8/0/8), her yapısal yokluk için mertebe.

    Sayım PANELDE CANLI: tablodan `Counter` ile sayılır ve `KOVA_SAYILARI`
    ile karşılaştırılır; ikisi de burada bağımsız üçüncü bir sayımla
    sınanır. Mertebeler için iki bağımsız yol: dizge grav fabrikasının
    beyanında BİREBİR geçiyor mu, ve CODATA 2018'den burada yeniden
    türetilince aynı altı hane çıkıyor mu. `abs=0.0` ŞART: sayılar 1e-37 ve
    1e+129 mertebesinde; `approx`ın varsayılan 1e-12'si birini yutar, ötekini
    anlamsız kılar (Task 1'de on kat sapan bir mutant tam bundan kaçtı).
    """
    tani = dict(sevk_panel.as_lines())["Yakınsama tanısı"]
    sayim = Counter(f.kova for f in KOVA_TABLOSU)
    assert {k: sayim.get(k, 0) for k in KOVALAR} == KOVA_SAYILARI == sevk_yuk.kova_sayim
    assert KOVA_SAYILARI == {
        "GERCEK_KOPRU": 3, "TAUTOLOJI": 4, "YAPISAL_YOK": 8, "BENZERLIK": 0,
        "OLCULMEDI": 6,
    }
    assert "GERCEK_KOPRU 3 / TAUTOLOJI 4 / YAPISAL_YOK 8 / BENZERLIK 0 / OLCULMEDI 6 (toplam 21)" in tani
    assert sevk_panel.diagnostics["kova"] == "3/4/8/0/6"

    # Mertebeler: iki proton, elektron-proton, r_s kübü, O 1s ızgarası.
    G, m_p, m_e, e, eps0, c = (
        6.67430e-11, 1.67262192369e-27, 9.1093837015e-31, 1.602176634e-19,
        8.8541878128e-12, 299792458.0,
    )
    a0 = 5.29177210903e-11
    k_e = 1.0 / (4.0 * math.pi * eps0)
    assert G * m_p * m_p / (k_e * e * e) == pytest.approx(8.093550856608016e-37, rel=1e-12, abs=0.0)
    assert G * m_p * m_e / (k_e * e * e) == pytest.approx(4.407885560760525e-40, rel=1e-12, abs=0.0)
    r_s = 2.0 * G * m_p / c**2
    assert f"{r_s:.6e}" == "2.484232e-54"
    assert f"{(a0 / r_s) ** 3:.6e}" == "9.665562e+129"
    assert f"{(1e-10 / r_s) ** 3:.6e}" == "6.522644e+130"
    for sayi in ("8.093550856608016e-37", "4.407885560760525e-40",
                 "9.665562e+129", "6.522644e+130", "4.096e+09", "61.04 GiB"):
        assert sayi in tani, f"mertebe panelde yok: {sayi}"
    # Grav beyanıyla BİREBİR aynı dizgeler (metin olarak, import değil).
    assert "9.665562e+129" in grav_fabrika._YOKLUK_BEYANI
    assert "6.522644e+130" in grav_fabrika._YOKLUK_BEYANI
    assert "8.093550856608016e-37" in grav_fabrika._YOKLUK_BEYANI
    # Tabloda da: grav x 6 girdisi hem F_g/F_e hem r_s mertebesini taşır.
    for hedef in ("tdse", "hf", "md", "tb", "spin", "em"):
        g = _girdi("grav", hedef)
        assert g.kova == Kova.YAPISAL_YOK
        assert "9.665562e+129" in g.gerekce and "2.484232e-54" in g.gerekce
        assert g.buyukluk == pytest.approx(8.093550856608016e-37, rel=1e-12, abs=0.0)
    assert 1600**3 * 16 / 2**30 == pytest.approx(61.04, abs=0.005)
    assert _girdi("hf", "tdse").buyukluk == 1600**3
    assert _girdi("em", "tdse").buyukluk is None
    # ÖLÇÜLMEDİ AYRI KOVA: bugün ALTI çift; beyan 29.08'in sekiz çiftini de
    # adıyla anar (ikisi -- tdse-md 22.09, md-tb 24.09 -- kenarı yazılarak
    # ayrıldı), ve "hiçbir sayı gelmedi" tarihsel cümlesini taşır.
    assert "OLCULMEDI AYRI BIR KOVADIR" in tani
    assert "hicbir sayi gelmedi" in tani
    assert "spec 7.2 ACIK" in tani
    for cift in ("tdse-md", "tdse-tb", "tdse-spin", "hf-spin", "md-tb", "md-em",
                 "tb-em", "spin-em"):
        assert cift in tani


def test_6_2_YAPI_ve_units_ekseni_KODDAN(sevk_yuk, sevk_panel):
    """§6.2 — yapı bir merdiven değil, ve units ekseni {hf, md, tb} kenar
    düğümlerinin ALT KÜMESİ (`tdse` bağlı ama ölçeksiz). Modül 7
    kapandığında spec bunu "iki çubuklu yıldız" ve "{hf, md, tb} = bağlı
    küme" diye yazıyordu; ikisi de 22.09.2026'da çürüdü.

    Testin adı 22.09.2026'da yıldız -> zincir, 24.09.2026'da şekilden
    bağımsız hâle getirildi; şeklin adı, eski adlar gibi tutmayan bir söz
    vermesin diye, `test_YAPI_BEYANI_SEKLIN_ADI_TARAMADAN_TURETILEN_SEKIL`
    ile taramadan türetilen şekle karşı ayrı testte sınanır.

    24.09.2026 (bulgu düzeltme turu): beyanın md -> tb import listesi
    EKSİKTİ -- `engine/md/factory.py` -> `qlab.engine.tb.radyal` (yasa
    nesnesini senaryodan kurar) yazılmamıştı, oysa `test_kova_tablosu`
    md -> tb importlarını tam üç (dosya, modül) çifti diye kilitliyordu.
    Aşağıdaki TAMLIK kilidi md/*.py'den qlab.engine.tb'ye giden her
    (dosya, modül) çiftini bağımsız toplar ve beyanda arar. KIRMA
    (ölçüldü, sk/duz/mut_bridge.log): beyandan factory parçasını silen
    mutant kırmızı.

    İki küme de PANELDE CANLI taranır (`olculen_kenarlar`,
    `olculen_units_ekseni`); burada üçüncü, bağımsız bir `ast` tarayıcı aynı
    kümeleri üretir. Beyanın satır numaraları da `ast`nin `lineno`suyla
    sınanır: `forcefields.py:13` gerçekten `qlab.engine.hf.scf`i,
    `downfold.py:130-132` gerçekten üç `hf` modülünü import ediyor mu.
    """
    tani = dict(sevk_panel.as_lines())["Yakınsama tanısı"]
    assert "`hf - md - tb` UCGENI arti `md - tdse` KUYRUGU" in tani
    assert sevk_yuk.kenarlar == BEKLENEN_KENARLAR == olculen_kenarlar()
    assert sevk_yuk.eksen == BEKLENEN_EKSEN == olculen_units_ekseni()
    assert "kod kenari 4: md -> hf, md -> tb, tb -> hf, tdse -> md" in tani
    # Yapının SAYILARI beyanda değil canlı satırda (spec md-tb 5.3).
    assert "dereceler hf=2 md=3 tb=2 tdse=1, sifir bagli cift 17/21" in tani
    assert "ekseni em=0 grav=0 hf=1 md=3 spin=0 tb=2 tdse=0 -> {hf, md, tb}" in tani
    assert sevk_panel.diagnostics["eksen"] == "hf,md,tb"
    assert sevk_panel.diagnostics["kenar"] == "4"

    # Bağımsız tarayıcı (bridge_map'in yardımcıları KULLANILMIYOR).
    def importlar(dosya: Path) -> set[str]:
        adlar = set()
        for d in ast.walk(ast.parse(dosya.read_text(encoding="utf-8"))):
            if isinstance(d, ast.Import):
                adlar.update(a.name for a in d.names)
            elif isinstance(d, ast.ImportFrom) and d.level == 0 and d.module:
                adlar.add(d.module)
        return adlar

    motor = KAYNAK / "engine"
    kenarlar, eksen = set(), {}
    for paket in sorted(p for p in motor.iterdir() if p.is_dir() and not p.name.startswith("__")):
        eksen[paket.name] = 0
        for dosya in sorted(paket.glob("*.py")):
            adlar = importlar(dosya)
            for m in adlar:
                p = m.split(".")
                if p[:2] == ["qlab", "engine"] and len(p) >= 3 and p[2] != paket.name:
                    kenarlar.add((paket.name, p[2]))
            if any(m.split(".")[:2] == ["qlab", "units"] for m in adlar):
                eksen[paket.name] += 1
    assert kenarlar == BEKLENEN_KENARLAR and eksen == BEKLENEN_EKSEN
    # 22.09.2026: ozdeslik DUSTU, kapsama kaldi -- `tdse` bagli ama olceksiz.
    eksenli = {ad for ad, n in eksen.items() if n > 0}
    dugumler = {m for k in kenarlar for m in k}
    assert eksenli <= dugumler
    assert dugumler - eksenli == {"tdse"}
    assert set(MODULLER) - {"hf", "md", "tb"} == {"tdse", "spin", "em", "grav"}

    # Satır numaraları gerçek.
    ff = ast.parse((motor / "md" / "forcefields.py").read_text(encoding="utf-8"))
    satir13 = [d for d in ast.walk(ff) if isinstance(d, ast.ImportFrom) and d.lineno == 13]
    assert [d.module for d in satir13] == ["qlab.engine.hf.scf"]
    df = ast.parse((motor / "tb" / "downfold.py").read_text(encoding="utf-8"))
    hf_import = [
        d for d in ast.walk(df)
        if isinstance(d, ast.ImportFrom) and d.module and d.module.startswith("qlab.engine.hf")
    ]
    assert {d.module.split(".")[-1] for d in hf_import} == {"basis", "integrals", "multistart"}
    assert min(d.lineno for d in hf_import) == 130 and max(d.lineno for d in hf_import) == 132
    assert "`engine/md/forcefields.py:13`" in tani and "`engine/tb/downfold.py:130-132`" in tani
    # 24.09.2026'nın kenarı: beyanın `tb_kuvvet.py` iddiası da kaynağa karşı.
    tk = ast.parse((motor / "md" / "tb_kuvvet.py").read_text(encoding="utf-8"))
    tb_import = {
        d.module for d in ast.walk(tk)
        if isinstance(d, ast.ImportFrom) and d.module and d.module.startswith("qlab.engine.tb")
    }
    assert tb_import == {"qlab.engine.tb.mermin", "qlab.engine.tb.radyal"}
    assert "`engine/md/tb_kuvvet.py` -> `qlab.engine.tb.{mermin, radyal}`" in tani
    # TAMLIK: md -> tb kenarini ureten HER (dosya, modul) cifti beyanda.
    md_tb = set()
    for dosya in sorted((motor / "md").glob("*.py")):
        for m in importlar(dosya):
            if m.startswith("qlab.engine.tb"):
                md_tb.add((dosya.name, m))
    assert md_tb == {
        ("factory.py", "qlab.engine.tb.radyal"),
        ("tb_kuvvet.py", "qlab.engine.tb.mermin"),
        ("tb_kuvvet.py", "qlab.engine.tb.radyal"),
    }, md_tb
    assert "`engine/md/factory.py` -> `qlab.engine.tb.radyal`" in tani


def _taramadan_sekil(kenarlar) -> str:
    """Kenar kümesinden şeklin ADI -- `bridge_map`ten BAĞIMSIZ türetilir.

    Tanınan TEK biçim bugünkü: bağlı, tek döngülü (kenar sayısı = düğüm
    sayısı), döngüsü bir ÜÇGEN ve üçgene asılı KUYRUK(lar). Başka bir biçim
    (dünkü zincir, Modül 7'nin yıldızı, ikinci bir döngü) ad üretmez ve
    test kırmızı olur: şekil değiştiyse beyan da elle değişmek ZORUNDA.
    """
    yonsuz = {tuple(sorted(k)) for k in kenarlar}
    dugum = sorted({m for k in yonsuz for m in k})
    komsu = {m: {b if a == m else a for a, b in yonsuz if m in (a, b)} for m in dugum}
    ucgenler = sorted(
        (a, b, c)
        for i, a in enumerate(dugum)
        for j, b in enumerate(dugum[i + 1:], i + 1)
        for c in dugum[j + 1:]
        if b in komsu[a] and c in komsu[a] and c in komsu[b]
    )
    if len(ucgenler) == 1 and len(yonsuz) == len(dugum):
        ucgen = ucgenler[0]
        kuyruklar = sorted(k for k in yonsuz if not set(k) <= set(ucgen))
        assert kuyruklar, "üçgen var ama kuyruk yok"
        parcalar = []
        for a, b in kuyruklar:
            ic, dis = (a, b) if a in ucgen else (b, a)
            assert ic in ucgen and dis not in ucgen, "kuyruk üçgene asılı değil"
            parcalar.append(f"`{ic} - {dis}`")
        return f"`{' - '.join(ucgen)}` UCGENI arti {', '.join(parcalar)} KUYRUGU"
    raise AssertionError(f"tanınmayan şekil: {sorted(yonsuz)}")


def test_YAPI_BEYANI_SEKLIN_ADI_TARAMADAN_TURETILEN_SEKIL():
    """Beyanın şekil adı TARAMADAN TÜRETİLEN şekle karşı (spec md-tb 5.3).

    ÖLÇÜLDÜ (spec md-tb 5.3): 13 literal düzeltilince testler yeşil oluyordu
    ama beyanın sayıları BAYAT kalıyordu, çünkü testler cümleyi yalnız
    METİN olarak kilitliyordu. Bu yüzden (a) `_YAPI_BEYANI` topoloji
    sayılarını (kenar sayısı, dereceler, sıfır bağ) YAZMAZ -- onlar canlı
    satırdadır -- ve (b) şeklin adı, `olculen_kenarlar()`dan burada
    bağımsız türetilen adı içermek zorundadır. Beşinci bir kenar şekli
    değiştirirse bu test kırmızı olur ve beyan elle güncellenmek zorunda
    kalır.
    """
    beklenen = _taramadan_sekil(olculen_kenarlar())
    assert beklenen == "`hf - md - tb` UCGENI arti `md - tdse` KUYRUGU"
    assert beklenen in bm._YAPI_BEYANI
    # Sayılar beyanda YOK (canlı satırda var, test_6_2).
    for bayatlayan in ("Dereceler", "dereceler hf", "ciftin 17", "ciftin 18", "tam UC", "tam DORT"):
        assert bayatlayan not in bm._YAPI_BEYANI, bayatlayan
    # Türetici kör değil: dünkü (22.09) zincir ve Modül 7'nin yıldızı AYNI
    # adı üretmez (üçgenleri yok -> tanınmayan şekil).
    for eski in (
        {("md", "hf"), ("tb", "hf"), ("tdse", "md")},
        {("md", "hf"), ("tb", "hf")},
    ):
        with pytest.raises(AssertionError, match="tanınmayan şekil"):
            _taramadan_sekil(eski)


def test_6_3_KENAR_TABANI_IKI_SAYI_AYRI_ETIKET_ve_SEKIZ_HANE(sevk_yuk, sevk_panel):
    """§6.3 — taban İKİ sayıdır, ayrı etiketle; canlı sayılar SEKİZ haneyle.

    Beyandaki sabit sayılar (3.3986707e-05, 1.2025992e-02, 1.4472809e-02,
    425.84, 0.8309, 1.6128e+08, 4.5578e+05) bu koşumun CANLI sayılarından
    sekiz/altı/dört haneye yuvarlanmış hâlleriyle BİREBİR eşleşmek zorunda;
    biri kayarsa beyan bayatlamıştır ve test kırmızı olur. Sekiz hane bir
    tercih değil ölçülmüş sınırdır (`ulp(TAM_CM)/taban = 2.68e-08`).
    """
    a1, a2 = sevk_yuk.a1, sevk_yuk.a2
    assert (a1.tur, a2.tur) == ("aritmetik", "hata-cubugu")
    assert a1.hukum == a2.hukum == "SAGLAM"
    assert a1.boru != a2.boru
    assert "fizik KAPALI" in a1.boru and "IKIZLE DEGISTIRILDI" in a2.boru
    assert "7/7 canli" in a1.boru and "1/1 canli" in a2.boru
    assert a1.sinyal == a2.sinyal  # aynı boru, aynı pencere

    tani = dict(sevk_panel.as_lines())["Yakınsama tanısı"]
    hata = dict(sevk_panel.as_lines())["Beklenen hata"]
    assert f"A1 (aritmetik): taban {a1.taban:.8g} cm^-1" in tani
    assert f"A2 (hata-cubugu): taban {a2.taban:.8g} cm^-1" in tani
    assert f"S/G {a1.oran:.4e}" in tani and f"S/G {a2.oran:.4e}" in tani
    assert "KENAR TABANI IKI SAYI" in tani

    # Beyan sabitleri <-> canlı sayılar: sekiz anlamlı hane, birebir.
    assert float("3.3986707e-05") == float(f"{a1.taban:.8g}")
    assert float("1.2025992e-02") == float(f"{a2.taban:.8g}")
    assert float("1.4472809e-02") == float(f"{GERCEK_HATA_CM:.8g}")
    assert f"{a1.oran:.4e}" == "1.6128e+08" and f"{a2.oran:.4e}" == "4.5578e+05"
    assert f"{GERCEK_HATA_CM / a1.taban:.2f}" == "425.84"
    assert f"{a2.taban / GERCEK_HATA_CM:.4f}" == "0.8309"
    for sabit in ("A1 3.3986707e-05", "A2 1.2025992e-02", "gercek hata 1.4472809e-02",
                  "425.84 KAT IYIMSER", "0.8309'udur", "S/G 1.6128e+08", "S/G 4.5578e+05",
                  "SEKIZ anlamli haneden fazlasina yazilamaz"):
        assert sabit in hata, f"beklenen hata satırı {sabit!r} taşımıyor"
    # Canlı satır: oranlar altı haneyle, sabitlerle uyumlu.
    assert f"gercek/A1 = {GERCEK_HATA_CM / a1.taban:.6g}" in tani
    assert f"A2/gercek = {a2.taban / GERCEK_HATA_CM:.6g}" in tani
    assert sevk_panel.diagnostics["A1"] == a1.taban and sevk_panel.diagnostics["A2"] == a2.taban
    # Beyan sekiz haneden FAZLASINI yazmıyor: on yedi haneli taban dizgesi yok.
    assert "3.3986706512223464e-05" not in _panel_metni(sevk_panel)


def test_6_4_ve_6_5_SEMA_CIPASI_ve_omega_A_SINAVDAN_CIKARILDI_omega_A_bit_bit(sevk_panel):
    """§6.4 + §6.5 — kenar bir ŞEMA ÇIPASI; `ω_A` sınavdan çıkarıldı, ve neden.

    "Neden"in ucuz yarısı burada CANLI ölçülür: ×1.001 kuvvet mutantında
    `ω_A` BİT BİT aynı kalır (`==`), çünkü statik yol yalnız enerji okur —
    altı SCF, ~0.1 s. Pahalı yarısı (`ω_B` +4.955181e-04 kayar) `slow`
    testte canlı üretilir; buradaki sabit onun kaydıdır.
    """
    sinif = dict(sevk_panel.as_lines())["Çıpa sınıfı"]
    assert "SEMA CIPASIDIR, fizik cipasi DEGIL" in sinif
    assert "AYNI HF cozumunun soyundan gelir" in sinif
    assert "uyum dogayi degil IKI UYGULAMAYI olcer" in sinif
    assert "`omega_A` SINAVDAN CIKARILDI" in sinif
    assert "+4.955181e-04" in sinif and "5481.229456 -> 5483.969419" in sinif
    assert "sqrt(1.001) - 1 = 4.99875e-04" in sinif
    assert f"{math.sqrt(1.001) - 1.0:.5e}" == "4.99875e-04"
    assert "sinav yalniz `omega_B` uzerinden kosar" in sinif
    assert sevk_panel.anchor_class is not None
    assert sevk_panel.anchor_class not in dict(sevk_panel.as_lines())["Yakınsama"]

    alan = HartreeFockGradient(["H", "H"])
    wa = _omega_a(alan)
    assert _omega_a(_Olcekli(alan, 1.001)) == wa
    assert _omega_a(_Olcekli(alan, 2.0)) == wa
    assert wa * HARTREE_TO_WAVENUMBER == pytest.approx(5481.2533589840905, rel=1e-9, abs=0.0)


@pytest.mark.slow
def test_6_5_omega_B_KUVVET_MUTANTINDA_KAYIYOR_4_955181e_04_CANLI():
    """§6.5'in pahalı yarısı — `+4.955181e-04` bir dokstring iddiası değil,
    HF borusunun canlı kaydıdır.

    Task 5'e kadar bu sayı depoda YALNIZ dokstringlerde yaşıyordu
    (`tests/md/test_aimd.py` "5483.96942 (kaydı)", spec §4 "+4.955181e-04");
    hiçbir assert onu kilitlemiyordu. Panel onu taşıyacağı için burada
    kilitlendi: ×1.001 kuvvet mutantıyla YOL B (HF, `step = 5e-5`) koşulur
    ve `ω_B'/ω_A − 1` okunur. Karekök yasası: `ω_B'/ω_B − 1 ≈ √1.001 − 1`.

    ÖLÇÜLDÜ (03.09.2026): dosyanın altındaki nota bakınız. MALİYET: tek YOL B
    koşumu (3 genlik × 1200 adım × 7 SCF), ~28 s → `slow`.
    """
    alan = HartreeFockGradient(["H", "H"])
    wa = _omega_a(alan)
    _, wb_mut = _a2_uydurmasi(_Olcekli(alan, 1.001))
    kayma = wb_mut / wa - 1.0
    assert kayma == pytest.approx(4.955181e-04, rel=1e-5, abs=0.0)
    assert wb_mut * HARTREE_TO_WAVENUMBER == pytest.approx(5483.969419, rel=1e-8, abs=0.0)
    wb_ref = 5481.229456091029  # `test_kenar_tabani.py`de bit bit kilitli
    assert wb_mut * HARTREE_TO_WAVENUMBER / wb_ref - 1.0 == pytest.approx(
        math.sqrt(1.001) - 1.0, rel=1e-4, abs=0.0
    )  # ölçülen bağıl fark 1.2e-05; pencere onun 8 katı
    # Ve ORAN çıpasının eşiği bu mutantı GÖRÜR (sınırın ~5 katı) — o yüzden
    # sınav `ω_B` üzerinden koşabiliyor.
    assert abs(kayma) > 1e-4


def test_6_6_MU_KORLUGU_PANELDE_ve_sayilari_test_aimd_ile_AYNI(sevk_panel):
    """§6.6 — `oran` `μ`'ye kör: −4.360844408468623e-06 → aynı, hex bit bit.

    Sayılar `tests/md/test_aimd.py`nin `slow` testinde HF ile canlı
    üretiliyor; burada üç şey ölçülür: (1) hex dizgesi gerçekten o float'ın
    hex'i mi, (2) 22.93 marjı `eşik/|değer|` mi (kategori hatasının tersi),
    (3) beyanın literalleri `test_aimd.py`nin assert ettiği literallerle
    BİREBİR aynı mı — kaynak metni okunarak, çünkü o dosyanın testi 78 s.
    """
    sinif = dict(sevk_panel.as_lines())["Çıpa sınıfı"]
    deger = -4.360844408468623e-06
    assert float.hex(deger) == "-0x1.24a6bc8940000p-18"
    assert "-4.360844408468623e-06 -> -4.360844408468623e-06" in sinif
    assert "hex -0x1.24a6bc8940000p-18" in sinif
    assert 1e-4 / abs(deger) == pytest.approx(22.931338665925146, rel=1e-12, abs=0.0)
    assert f"{1e-4 / abs(deger):.2f} kat" == "22.93 kat" and "22.93 kat" in sinif
    assert "`omega_A(4mu) == omega_A(mu)/2` ve `omega_B(4mu) == omega_B(mu)/2`" in sinif
    assert "KATEGORI HATASIYDI" in sinif
    assert "kendi MUTLAK sinavina sahiptir" in sinif
    aimd = (PROJE_KOKU / "tests" / "md" / "test_aimd.py").read_text(encoding="utf-8")
    for literal in ("-4.360844408468623e-06", "22.931338665925146",
                    "-0x1.24a6bc8940000p-18", "assert wa4 == wa1 / 2.0",
                    "assert wb4 == wb1 / 2.0", "assert oran4 == oran1"):
        assert literal in aimd, f"test_aimd.py {literal!r} taşımıyor — beyanın kaynağı kayboldu"


def test_6_7_DUSEN_IKI_ADAY_TAUTOLOJI_OLARAK_PENCERESIYLE(sevk_panel):
    """§6.7 — `hf → tb` ve `tb → spin` tautoloji olarak, penceresiyle;
    sayılar tablonun literalleriyle (Task 4'te motorla doğrulanmış) aynı."""
    dairesellik = dict(sevk_panel.as_lines())["Dairesellik"]
    assert "(1) `hf -> tb`" in dairesellik and "(2) `tb -> spin`" in dairesellik
    assert f"bagil {_HF_TB_OZDESLIK_BAGIL:.3e}" in dairesellik
    assert "15 hane yalniz N = 6'da" in dairesellik and "N >= 8'de ~1e-8" in dairesellik
    assert "22 kabugun 10'unda dusuyor (1 ulp)" in dairesellik
    assert "`model_from_downfold` `src/`de 0 cagri" in dairesellik
    assert "(N, d) altili, STO-3G, seeds = 12" in dairesellik
    assert _TB_SPIN_M_FARKI == 0.0 and _girdi("tb", "spin").buyukluk == 0.0
    assert "`m` serisi bit bit ayni, `E == J * E(1)` bit bit" in dairesellik
    assert "0.187216796875 = 287565/1536000" in dairesellik
    assert "U/|t| = 2.033100 ... 13.630622" in dairesellik
    assert "`U` 1000 kat buyutulse de kenar ACILMIYOR" in dairesellik
    assert "TEK SATIR" in dairesellik and "spin kodda 4, TB halkasi 2" in dairesellik
    assert "Ising L = 32, h = 0, tohum 20260822, burn 500, sweep 1500" in dairesellik
    assert "Ikisi de sessizce dusurulmedi" in dairesellik
    # Tablo ile aynı sayılar.
    assert "0.187216796875" in _girdi("tb", "spin").gerekce
    assert repr(_HF_TB_OZDESLIK_BAGIL) in _girdi("hf", "tb").gerekce
    assert _girdi("hf", "tb").kova == _girdi("tb", "spin").kova == Kova.TAUTOLOJI


def test_6_8_YONTEM_HEDEF_UYUSMAZLIGININ_IKI_TURU_AYRI_ve_GRAVDAN_FARKLI(sevk_panel):
    """§6.8 — A düzeltilemez (grav; spin `J`), B ertelendi (3B TDSE; kırılma
    indisi). İki metin AYRI girdidir, biri ötekinin içinde geçmez ve ikisi
    de Modül 6'nın aynı adlı sabitlerinden FARKLIDIR (kopya değil)."""
    a, b = bm._YONTEM_HEDEF_A, bm._YONTEM_HEDEF_B
    assert a != b and a not in b and b not in a
    assert a in bm._IHMAL and b in bm._IHMAL
    ihmal = dict(sevk_panel.as_lines())["İhmal edilen"]
    assert a in ihmal and b in ihmal
    assert "TUR A -- BU BORUDA DUZELTILEMEZ" in a and "(1) `grav`" in a and "(2) `spin` `J`" in a
    assert "TUR B -- ERTELENDI" in b and "`hf -> tdse`" in b and "`hf -> em`" in b
    assert a != grav_fabrika._YONTEM_HEDEF_A and b != grav_fabrika._YONTEM_HEDEF_B
    assert a not in grav_fabrika._YONTEM_HEDEF_A and b not in grav_fabrika._YONTEM_HEDEF_B


def test_DURUST_OLUMSUZLAR_BESI_DE_PANELDE(sevk_panel):
    """Beyan edilmiş, düzeltilmemiş beş kalem SAKLANMAZ — biri sessizce
    düşerse test kırılır. Dördüncüsü bu görevde bulundu (6-31G `ω_B`
    süite alınmadı) ve panel onu da taşır."""
    hata = dict(sevk_panel.as_lines())["Beklenen hata"]
    for parca in (
        "(1) spec 7.2 HALA ACIK ama DARALDI: 22.09.2026'da sekiz ciftin sekizi "
        "de olculdu ve `tdse-md` kenari yazildi",
        "(2) `md -> spin` ve `hf -> em` tautoloji hukumleri 29.08.2026 on olcumunden DEVRALINDI",
        "(3) makineler arasi tekrarlanabilirlik OLCULMEDI",
        "numpy 2.5.1",
        "(4) `omega_B`nin 6-31G degeri (spec 4: 4646.16252) bu depoda suite ALINMADI",
        "(5) izotropik hicbir sey yok bu modulde ama bir ikinci boru da yok",
    ):
        assert parca in hata, f"dürüst olumsuz eksik: {parca!r}"
    damgali = {f.cift for f in KOVA_TABLOSU if bm.YENIDEN_URETILMEDI in f.gerekce}
    assert damgali == {("md", "spin"), ("em", "hf")}
    assert np.__version__ == "2.5.1", "numpy sürümü değişti; (3) numaralı dürüst olumsuzun kaydı bayat"


def test_KURAM_SECICI_baz_seti_ve_6_31G_IKI_PENCEREDE_CANLI(sevk_panel):
    """Kuram seçici = baz seti; spec §4'ün 4646.18'i YALNIZ 6-31G'nin KENDİ
    `r_e`sinde çıkıyor — sevk edilen `r_e`de 4921.26. İkisi de burada ucuz
    (statik, 3 SCF + Brent) kilitli: bir sayı hangi pencerede doğruysa
    orada yazılır, pencere adlandırılmadan taşınmaz.

    ÖLÇÜLDÜ (03.09.2026): r_e(6-31G) = 1.3794239789 bohr,
    ω_A(6-31G, kendi r_e) = 4646.186940 cm⁻¹ (k = 0.4116612), ω_A(6-31G,
    r_e = 1.3459193602) = 4921.258823 cm⁻¹ (k = 0.4618479). STO-3G'nin
    Brent minimumu 1.3459193687, `R_E_BOHR`dan 8.5e-09 uzakta (yöntem
    penceresi ~5e-08 içinde).
    """
    kutu = dict(sevk_panel.as_lines())["Kuram seçici"]
    for parca in ("HF/STO-3G", "4646.19 cm^-1", "k = 0.411661", "1.3794239789 bohr",
                  "4921.26 cm^-1", "k = 0.461848", "PENCERESI (`r_e`) yazilmamisti",
                  "275 cm^-1"):
        assert parca in kutu, f"kuram seçici {parca!r} taşımıyor"

    a6 = HartreeFockGradient(["H", "H"], basis="6-31g")

    def E(r: float) -> float:
        return a6(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]))[0]

    re6 = float(minimize_scalar(E, bracket=(1.30, 1.38, 1.46), tol=1e-10).x)
    assert re6 == pytest.approx(1.3794239789, abs=1e-8, rel=0.0)
    w_kendi = _omega_a(a6, r_e=re6)
    w_sevk = _omega_a(a6)  # r_e = R_E_BOHR (STO-3G'nin)
    assert w_kendi * HARTREE_TO_WAVENUMBER == pytest.approx(4646.186940, rel=1e-8, abs=0.0)
    assert w_sevk * HARTREE_TO_WAVENUMBER == pytest.approx(4921.258823, rel=1e-8, abs=0.0)
    assert w_kendi**2 * MU_H2 == pytest.approx(0.4116612, rel=1e-6, abs=0.0)
    assert w_sevk**2 * MU_H2 == pytest.approx(0.4618479, rel=1e-6, abs=0.0)
    assert f"{w_kendi * HARTREE_TO_WAVENUMBER:.2f}" == "4646.19"
    assert f"{w_sevk * HARTREE_TO_WAVENUMBER:.2f}" == "4921.26"
    assert (w_sevk - w_kendi) * HARTREE_TO_WAVENUMBER == pytest.approx(275.07, abs=0.01, rel=0.0)
    # Spec §4'ün sayısı sevk edilen pencerede ÜRETİLEMİYOR.
    assert w_sevk * HARTREE_TO_WAVENUMBER != pytest.approx(4646.18, rel=1e-3, abs=0.0)
    # STO-3G'nin kendi minimumu R_E_BOHR'un yöntem penceresi içinde.
    a3 = HartreeFockGradient(["H", "H"])
    re3 = float(minimize_scalar(
        lambda r: a3(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]))[0],
        bracket=(1.30, 1.346, 1.40), tol=1e-10,
    ).x)
    assert abs(re3 - R_E_BOHR) < 5e-8


# ---------------------------------------------------------------------------
# `converged` GERÇEKTEN BAĞLI — tablo bayatlarsa ya da taban çürürse tik düşer
# ---------------------------------------------------------------------------


def test_TABLO_BAYATLARSA_PANEL_TIK_BASMAZ_ve_CELISKIYI_YAZAR(sevk_senaryo, tmp_path):
    """Bayat bir yokluk beyanı bir yalandır ve yalanın yanında tik basılmaz.

    Sentetik bir `engine/` ağacında `spin → em` import'u yaratılır (Task 1'in
    kilidiyle aynı kurgu; kenar `__init__.py` içinde ki `__init__`i eleyen
    bir mutant kaçamasın). Tablo o çifti `OLCULMEDI`de tutuyor; tarama onu
    bulunca `tablo_tutarsizliklari` DOLU döner, `converged` False olur ve
    canlı satır çelişkiyi ADIYLA yazar. `units` ekseni de boş çıkar ve
    kenar düğümleriyle çelişir — ikinci kilit de ateşlenir.
    """
    kok = tmp_path / "engine"
    (kok / "spin").mkdir(parents=True)
    (kok / "spin" / "__init__.py").write_text(
        "from qlab.engine.em.grid import Grid\n", encoding="utf-8"
    )
    (kok / "em").mkdir()
    (kok / "em" / "__init__.py").write_text("", encoding="utf-8")

    yuk = build_bridge_payload(sevk_senaryo, motor_koku=kok)
    assert yuk.kenarlar == {("spin", "em")}
    assert yuk.eksen == {"em": 0, "spin": 0}
    sorunlar = yuk.tutarsizliklar
    assert sorunlar == tablo_tutarsizliklari(yuk.kenarlar, yuk.eksen)
    assert any("OLCULMEDI" in s for s in sorunlar)
    assert any("GERCEK_KOPRU ('hf', 'md') icin kod kenari YOK" in s for s in sorunlar)
    # İKİNCİ KİLİT 22.09.2026'da DARALDI: eskiden "eksen kümesi = düğüm
    # kümesi" özdeşliğiydi; `tdse` bağlı ama ölçeksiz olduğu için artık
    # (a) kapsama + (b) ölçeksiz düğüm kümesinin ADIYLA kilitlenmesi.
    # Sentetik ağaçta düğümler {spin, em}, eksen boş -> (b) ateşlenir.
    assert any("olceksiz kenar dugumleri" in s for s in sorunlar)
    p = provenance_for(yuk)
    assert p.converged is False
    tani = dict(p.as_lines())["Yakınsama tanısı"]
    assert "TABLO KODLA CELISIYOR" in tani and "TABLO KODLA TUTARLI" not in tani
    assert dict(p.as_lines())["Yakınsama"].startswith("✗")
    # Gerçek ağaçta tutarlı.
    assert tablo_tutarsizliklari(olculen_kenarlar(), olculen_units_ekseni()) == ()


def test_TABAN_CURURSE_PANEL_TIK_BASMAZ(sevk_yuk):
    """`converged`ın ikinci bacağı: iki tabandan biri `CURUK`/`DEJENERE`
    olursa tik düşer; ikisi de `SAGLAM` ve tablo tutarlıysa tik var."""
    assert provenance_for(sevk_yuk).converged is True
    for alan in ("a1", "a2"):
        for hukum in ("CURUK", "DEJENERE"):
            bozuk = dataclasses.replace(
                sevk_yuk, **{alan: dataclasses.replace(getattr(sevk_yuk, alan), hukum=hukum)}
            )
            p = provenance_for(bozuk)
            assert p.converged is False, (alan, hukum)
            assert f"hukum {hukum}" in dict(p.as_lines())["Yakınsama tanısı"]


# ---------------------------------------------------------------------------
# PENCERE DÜĞMELERİ GERÇEKTEN BAĞLI
# ---------------------------------------------------------------------------


def test_PENCERE_DUGMESI_n_bolme_BAGLI_ve_IKI_SAYI_IKI_AYRI_DUGMEYE_TEPKI_VERIYOR(sevk_senaryo, sevk_yuk):
    """Senaryonun `n_bolme`si `edge_noise_floor`a GERÇEKTEN geçiyor: N = 400'de
    A1 tabanı düşer ve etiket yeni pencereyi yazar.

    ÖLÇÜLDÜ (03.09.2026) — ve `a = 0`ın 14.89'u A1'e AKTARILAMADI, bu da
    ölçülerek görüldü: A1 merdivenin `a = 0.03` tepesinden okunuyor ve orada
    N = 200 → 400 oranı 15.18 (`A1_N200_N400_ORANI`). ASIL BULGU: A2 `dt`ye
    KÖR (oran 1.0008) — iki sayı iki ayrı düğmeye tepki veriyor, A1
    aritmetiği, A2 fiziği ölçer. Beyan bunu taşır; burada kilitli.

    MALİYET ÖLÇÜLDÜ: N = 400 yükü 1.32 s (A1 sekiz koşum × 2400 adım). İki
    düğme (bu ve `a_ikiz`) tek testte 2.81 s ediyordu — eşik 2 s — o yüzden
    bölündü; ikisi de eşiğin altında, sapma yok.
    """
    y400 = build_bridge_payload(dataclasses.replace(sevk_senaryo, n_bolme=400))
    assert "n_bolme=400" in y400.a1.boru and "n_bolme=400" in y400.a2.boru
    assert y400.a1.taban < sevk_yuk.a1.taban and y400.a2.taban < sevk_yuk.a2.taban
    assert sevk_yuk.a1.taban / y400.a1.taban == pytest.approx(A1_N200_N400_ORANI, rel=1e-3, abs=0.0)
    assert sevk_yuk.a2.taban / y400.a2.taban == pytest.approx(A2_N200_N400_ORANI, rel=1e-3, abs=0.0)
    # İKİ SAYI İKİ AYRI DÜĞMEYE TEPKİ VERİYOR: A1 `dt`ye 4. mertebeden, A2
    # `dt`ye KÖR (< %0.1). Beyan bunu iki haneyle taşır; burada kilitli.
    assert A1_N200_N400_ORANI > 10.0 and abs(A2_N200_N400_ORANI - 1.0) < 1e-3
    hata = dict(provenance_for(sevk_yuk).as_lines())["Beklenen hata"]
    assert f"A1 {A1_N200_N400_ORANI:.2f} kat duser, A2 {A2_N200_N400_ORANI:.4f} kat" in hata
    # Sinyal de aynı pencerede yeniden okunur: karışık pencere YOK.
    assert y400.a1.sinyal == y400.a2.sinyal and y400.a1.sinyal != sevk_yuk.a1.sinyal
    pencere = dict(provenance_for(y400).as_lines())["Pencere"]
    assert "BU KOSUMUN PENCERESI: N = 400, 6 periyot" in pencere


def test_PENCERE_DUGMESI_a_ikiz_BAGLI_ve_A1_IKIZDEN_BAGIMSIZ(sevk_senaryo, sevk_yuk):
    """`a_ikiz` A2'nin ikizini seçer ve YALNIZ A2'yi oynatır: `a = 0.5`te A2
    tabanı sevk edilenin 0.16684'ü (`test_kenar_tabani.py` ile aynı sayı), A1
    bit bit aynı — A1'in merdiveni çağırana kapalıdır. MALİYET: 0.7 s."""
    y05 = build_bridge_payload(dataclasses.replace(sevk_senaryo, a_ikiz=0.5))
    assert "a = 0.5" in y05.a2.boru and y05.a1.boru == sevk_yuk.a1.boru
    assert y05.a1.taban == sevk_yuk.a1.taban  # A1 ikizden BAĞIMSIZ
    assert y05.a2.taban / sevk_yuk.a2.taban == pytest.approx(0.16684, rel=1e-3, abs=0.0)
    pencere = dict(provenance_for(y05).as_lines())["Pencere"]
    assert "A2 ikizi a = 0.5" in pencere


def test_edge_noise_floor_PENCERE_ANAHTARI_KAPISI_ve_a_HALA_KAPALI():
    """`edge_noise_floor`un yeni `**pencere`si bir arka kapı değil:
    tanımadığı anahtar `InvalidInputError`, `a` A1'de hâlâ verilemez."""
    assert PENCERE_ANAHTARLARI == ("n_bolme", "periyot_sayisi", "genlikler")
    with pytest.raises(InvalidInputError, match="pencere anahtari taninmiyor"):
        edge_noise_floor(tur="aritmetik", yabanci=1)
    with pytest.raises(InvalidInputError):
        edge_noise_floor(tur="aritmetik", a=0.5)
    with pytest.raises(InvalidInputError):
        edge_noise_floor(tur="hata-cubugu", n_bolme=0)


# ---------------------------------------------------------------------------
# SENARYO SINIRI — ŞEMA VE YÜKLEYİCİ KAPILARI
# ---------------------------------------------------------------------------


def test_SEVK_EDILEN_YAML_PENCERENIN_HER_DUGMESINI_TASIYOR_ve_VARSAYILAN_YOK(sevk_senaryo):
    """Sevk edilen senaryo tam olarak bu penceredir; `BridgeScenario` düz bir
    `@dataclass(frozen=True)`tir (pydantic YOK — ölçüldü) ve hiçbir alanın
    varsayılanı yoktur."""
    s = sevk_senaryo
    assert isinstance(s, BridgeScenario)
    assert (s.a_ikiz, s.n_bolme, s.periyot_sayisi, s.genlikler) == (0.9137, 200, 6, (0.01, 0.02, 0.04))
    assert dataclasses.is_dataclass(BridgeScenario) and BridgeScenario.__dataclass_params__.frozen
    varsayilanli = [
        f.name for f in dataclasses.fields(BridgeScenario)
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
    ]
    assert varsayilanli == []
    assert importlib.util.find_spec("pydantic") is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.n_bolme = 400  # type: ignore[misc]
    assert "bridge" in _KNOWN_KINDS


def test_A1_MERDIVENI_SENARYODA_BIR_ANAHTAR_DEGIL(tmp_path):
    """`a` YAML'da kabul edilseydi A1'in fizik-kapatma noktası çağıranın
    düğmesi olurdu (Modül 6'nın `lam` dersi). Reddedilir."""
    with pytest.raises(ScenarioError, match="Bilinmeyen anahtar"):
        load_scenario(_yaz(tmp_path, SEVK_YAML + "a: 0.01\n"))


@pytest.mark.parametrize("anahtar, deger, desen", [
    ("a_ikiz", "0.0", "'a_ikiz' pozitif"),
    ("a_ikiz", "-0.5", "'a_ikiz' pozitif"),
    ("a_ikiz", "abc", "dönüştürülemedi"),
    ("n_bolme", "0", "'n_bolme' en az 1"),
    ("n_bolme", "x", "tam sayı"),
    ("periyot_sayisi", "0", "'periyot_sayisi' en az 1"),
    ("genlikler", "[0.01]", "en az iki genlik"),
    ("genlikler", "[0.02, 0.01]", "KESİN ARTAN"),
    ("genlikler", "[0.01, 0.01]", "KESİN ARTAN"),
    ("genlikler", "[0.01, -0.02]", "pozitif ve sonlu"),
    ("genlikler", "0.01", "sayı listesi"),
])
def test_YUKLEYICI_KAPILARI_YAML_TARAFINDA_yakaliyor(tmp_path, anahtar, deger, desen):
    """Yükleyici motorun kapılarını BİLEREK tekrarlar — KATMAN AYRIMI:
    burada `ScenarioError`, hiçbir MD adımı koşmadan; motorda
    `InvalidInputError`."""
    with pytest.raises(ScenarioError, match=desen):
        load_scenario(_yaz(tmp_path, _degistir(anahtar, deger)))


def test_EKSIK_ANAHTAR_SESSIZCE_VARSAYILANA_DUSMUYOR_ve_BILINMEYEN_KIND_bridge_SAYIYOR(tmp_path):
    metin = "\n".join(s for s in SEVK_YAML.splitlines() if not s.startswith("periyot_sayisi:"))
    with pytest.raises(ScenarioError, match="Eksik anahtar"):
        load_scenario(_yaz(tmp_path, metin + "\n"))
    with pytest.raises(ScenarioError, match="bridge"):
        load_scenario(_yaz(tmp_path, _degistir("kind", "koprü")))


def test_SEVK_EDILEN_YAML_HICBIR_TABAN_DEGERINI_TASIMIYOR():
    """Senaryo dosyası cevabı taşımaz — bu bir niyet değil bir SAYIMDIR."""
    for yasak in ("3.3986", "1.2025", "1.4472", "8.0935", "9.6655", "425.8",
                  "0.8309", "4646", "4921", "2.276", "0.18721", "5481", "5483"):
        assert yasak not in SEVK_YAML, f"taban/çıpa değeri YAML'da: {yasak}"


# ---------------------------------------------------------------------------
# KATMANLAR VE DOKSTRİNG KALIBI
# ---------------------------------------------------------------------------


def test_run_py_KOVA_ADLARINI_ve_BEYANI_YAZMAZ_ITHAL_EDER():
    """`_run_bridge` hiçbir kova adını ve hiçbir beyan cümlesini yazmaz;
    `KOVALAR`ı `bridge_map`ten İTHAL EDER. `ast` ile: `run.py`de `KOVALAR`
    üyelerine eşit hiçbir dize sabiti yok; import `_run_bridge`ın gövdesinde."""
    agac = ast.parse((KAYNAK / "run.py").read_text(encoding="utf-8"))
    sabitler = {d.value for d in ast.walk(agac) if isinstance(d, ast.Constant) and isinstance(d.value, str)}
    assert not (sabitler & set(KOVALAR)), sabitler & set(KOVALAR)
    govde = next(d for d in ast.walk(agac) if isinstance(d, ast.FunctionDef) and d.name == "_run_bridge")
    importlar = [
        (d.module, sorted(a.name for a in d.names))
        for d in ast.walk(govde) if isinstance(d, ast.ImportFrom)
    ]
    assert ("qlab.bridge_map", ["KOVALAR", "KOVA_TABLOSU"]) in importlar
    assert ("qlab.viz.bridge_frames", ["build_bridge_payload", "provenance_for"]) in importlar
    # `main` dağıtımı var.
    ana = next(d for d in ast.walk(agac) if isinstance(d, ast.FunctionDef) and d.name == "main")
    assert any(
        isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "isinstance"
        and any(isinstance(a, ast.Name) and a.id == "BridgeScenario" for a in d.args)
        for d in ast.walk(ana)
    )


def test_DOKSTRING_KALIBI_viz_ASCII_run_ve_scenario_AKSANLI():
    """Global kısıt ÖLÇÜLÜYOR: `viz/bridge_frames.py` ve `bridge_map.py`
    aksansız ASCII; `run.py`/`scenario/*`nin Task 5 eklemeleri o dosyaların
    aksanlı kalıbında."""
    from qlab import run as run_mod
    from qlab.scenario import loader as loader_mod

    for ad in ("viz/bridge_frames.py", "bridge_map.py"):
        metin = (KAYNAK / ad).read_text(encoding="utf-8")
        aksanli = [no for no, s in enumerate(metin.splitlines(), 1) if not s.isascii()]
        assert aksanli == [], f"{ad} ASCII dışı satır taşıyor: {aksanli}"
    assert not run_mod._run_bridge.__doc__.isascii()
    assert not loader_mod._load_bridge_scenario.__doc__.isascii()
    assert not BridgeScenario.__doc__.isascii()
    assert not Path(__file__).read_text(encoding="utf-8").isascii()


def test_bridge_map_HALA_HICBIR_QLAB_MODULUNU_IMPORT_ETMIYOR_ve_YENI_FONKSIYONLAR_SAF():
    """Task 5'in eklemeleri saf tanım katmanını bozmadı: `bridge_map` yine
    yalnız stdlib import ediyor; `panel_metinleri` sözlük döndürüyor,
    `Provenance` KURMUYOR (onu `viz` kurar)."""
    agac = ast.parse((KAYNAK / "bridge_map.py").read_text(encoding="utf-8"))
    adlar = set()
    for d in ast.walk(agac):
        if isinstance(d, ast.Import):
            adlar.update(a.name for a in d.names)
        elif isinstance(d, ast.ImportFrom) and d.module:
            adlar.add(d.module)
    assert adlar <= {"__future__", "ast", "dataclasses", "pathlib"}
    # `Provenance` kurulmuyor: ne import ne çağrı (metinde adı yorumlarda geçer,
    # o serbest — ölçüt yapısaldır).
    assert not any(
        isinstance(d, ast.Call)
        and ((isinstance(d.func, ast.Name) and d.func.id == "Provenance")
             or (isinstance(d.func, ast.Attribute) and d.func.attr == "Provenance"))
        for d in ast.walk(agac)
    )
    for ad in ("olculen_units_ekseni", "tablo_tutarsizliklari", "canli_satir",
               "panel_metinleri", "bridge_declaration"):
        assert callable(getattr(bm, ad))


# ---------------------------------------------------------------------------
# UÇTAN UCA
# ---------------------------------------------------------------------------


def test_CLI_UCTAN_UCA(tmp_path):
    """`python -m qlab.run scenarios/kopru_haritasi.yaml -o kopru.npz`.

    ÖLÇÜLDÜ (03.09.2026, `--durations=0`): koşumun kendisi ≈ 0.7 s,
    yorumlayıcı ve import'larla alt süreç **1.27 s** — deponun 2 s `slow`
    eşiğinin ALTINDA, yani Modül 5/6'nın "eşiğin üstünde ama bilerek
    işaretlenmedi" sapmasına burada gerek kalmadı; test ölçüldüğü için
    işaretsiz, tahmin edildiği için değil. (İlk tahminim ≈ 3 s idi ve
    yanlıştı; sayı ölçümdür.)

    ON panel satırının ONU, ON DOKUZ beyanın ON DOKUZU da ekranda aranır;
    metinler bu dosyada TEKRARLANMAZ, `bridge_map`ten okunur.
    """
    cikti = tmp_path / "kopru.npz"
    sonuc = subprocess.run(
        [sys.executable, "-m", "qlab.run", str(SEVK_EDILEN), "-o", str(cikti)],
        capture_output=True, text=True, encoding="utf-8", cwd=str(PROJE_KOKU),
    )
    assert sonuc.returncode == 0, sonuc.stderr
    assert cikti.exists()
    for etiket in PANEL_SATIRLARI:
        assert etiket in sonuc.stdout, f"panel satırı kayıp: {etiket}"
    for cumle in _tum_beyanlar():
        assert cumle in sonuc.stdout, f"beyan EKRANA basılmadı: {cumle[:50]!r}"
    # Sayı tablosu: None ile 0.0 AYRI basılıyor; iki taban iki satır.
    assert "hf -> md    None" in sonuc.stdout
    assert "tb -> spin  0.000000e+00" in sonuc.stdout
    assert "hf -> tb    2.276006e-15" in sonuc.stdout
    assert sonuc.stdout.count("None") == 11  # tdse-md artik SAYI tasiyor
    assert "taban A1 aritmetik" in sonuc.stdout and "taban A2 hata-cubugu" in sonuc.stdout
    assert "TABLO ≠ KOD" not in sonuc.stdout
    assert "TABLO KODLA TUTARLI" in sonuc.stdout
    for kova in KOVALAR:
        assert f"  {kova:<13} {KOVA_SAYILARI[kova]}" in sonuc.stdout

    with np.load(cikti, allow_pickle=False) as veri:
        assert list(veri["kova"]) == [f.kova for f in KOVA_TABLOSU]
        assert int(veri["magnitude_is_none"].sum()) == 11
        assert int(np.isfinite(veri["magnitude"]).sum()) == 10
        assert np.all(np.isnan(veri["magnitude"][veri["magnitude_is_none"]]))
        # KONUMSAL DEGIL ADLA. `index("TAUTOLOJI") + 1` bugun tb -> spin'e denk
        # geliyor ama bu bir RASTLANTIDIR. OLCULDU (22.09.2026): tabloda DORT
        # TAUTOLOJI satiri var (hf-tb, spin-tb, md-spin, em-hf); araya eklenecek
        # yeni bir TAUTOLOJI satiri bu assert'i sessizce BASKA bir satira
        # kaydirirdi ve -- onerilen yeni girdiler de 0.0 tasidigi icin -- assert
        # YESIL kalirdi. O anda yorum satirinin kendisi bir YALAN olurdu.
        _i_tb_spin = [
            i for i, (a, b) in enumerate(zip(veri["pair_a"], veri["pair_b"]))
            if {str(a), str(b)} == {"tb", "spin"}
        ]
        assert len(_i_tb_spin) == 1, f"tb-spin satiri tek degil: {_i_tb_spin}"
        assert veri["magnitude"][_i_tb_spin[0]] == 0.0
        assert [tuple(k) for k in veri["edges"]] == sorted(BEKLENEN_KENARLAR)
        assert dict(zip(veri["units_axis_names"], veri["units_axis_counts"].tolist())) == BEKLENEN_EKSEN
        assert list(veri["floor_turler"]) == ["aritmetik", "hata-cubugu"]
        assert list(veri["floor_verdicts"]) == ["SAGLAM", "SAGLAM"]
        assert float(veri["floor_values"][0]) == pytest.approx(3.3986707e-05, rel=1e-7, abs=0.0)
        assert float(veri["floor_values"][1]) == pytest.approx(1.2025992e-02, rel=1e-7, abs=0.0)
        assert float(veri["real_error"]) == GERCEK_HATA_CM
        assert bool(veri["converged"]) is True and len(veri["inconsistencies"]) == 0
        kaynak = json.loads(str(veri["provenance_json"]))
    # `.npz`ye YAZILAN beyan ile EKRANA basılan beyan AYNI kaynaktan.
    assert kaynak["anchor_class"] == bm._CIPA_SINIFI_BEYANI + " " + bm._MU_KORLUGU_BEYANI
    assert kaynak["circularity"] == bm._DUSEN_ADAYLAR_BEYANI
    assert kaynak["theory_selector"] == bm._KURAM_SECICI_BEYANI
    assert kaynak["convergence_diagnosis"].startswith("[CANLI")
    assert kaynak["converged"] is True
    assert kaynak["diagnostics"]["kova"] == "3/4/8/0/6"


# ---------------------------------------------------------------------------
# `slow` KARARININ ÖLÇÜMÜ -- __DURATIONS__
#
# HF MUTANT ÖLÇÜMÜNÜN KAYDI (bu dosyanın `slow` testi, 03.09.2026):
#     ω_A                      5481.2533589840905 cm⁻¹  (mutantta BİT BİT aynı)
#     ω_B' (kuvvet ×1.001)     5483.969419304486  cm⁻¹
#     ω_B'/ω_A − 1             4.95518112831661e-04     (spec §4: +4.955181e-04 ✓)
#     ω_B'/ω_B − 1             4.9988114e-04            (√1.001 − 1 = 4.9987506e-04, bağıl 1.2e-05)
#     süre                     36.86 s
# ---------------------------------------------------------------------------
