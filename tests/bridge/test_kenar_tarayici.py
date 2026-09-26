"""Kenar tarayıcısı — YALIN sürüm (26.09.2026): düz import biçimleri çözülür,
ADI SAYILAN yükleme biçimleri motor ağacında YASAKTIR (liste sonlu).

Osman'ın kararı "sadeleştir": 25-26.09'daki ağır analizcinin (kaydı:
`.superpowers/sdd/kenar-tarayici-AGIR-surum-2026-09-26/`) yerine bir
YASAKLAMA POLİTİKASI kondu. Bu dosya beş şeyi sınar:

(a) düz biçimler — 648f03b'nin tarayıcısı (git'ten okunur) kaçırır, yenisi
    yakalar; eskinin zaten gördüğü biçimler iki sürümde de aynı sonucu verir;
(b) her YASAK biçim adıyla (`dosya:satir bicim`) raporlanır ve tabloyu
    kırmızıya çevirir; yasaksız eşdeğer ağaç boş döner (yanlış pozitif yok);
(c) gerçek ağaçta taranamayan yol SIFIR — ölçülmüş bir sıfır — ve kenar /
    eksen 648f03b'nin tarayıcısıyla AYNI;
(d) kenar sınıflayıcısı + kilit 5 + kilit 2'nin değişmediği; annotasyonu
    DEĞERLENDİREN tüketici `src/qlab` çapında sıfır (yalnız-tip hükmü buna
    koşullu);
(e) YÜK TAŞIMA ÇIPASI — `from_md.py`nin md importu silinince çıktı bit bit
    aynı; pozitif kontrol `tb_kuvvet.py`nin tb importu silinince NameError.

Sentetik ağaçlar `tmp_path` altında kurulur; hiçbir test `src/`ye yazmaz.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import qlab.bridge_map as bm
import qlab.engine.md.tb_kuvvet as tb_kuvvet_gercek
import qlab.engine.tdse.from_md as from_md_gercek
from qlab.bridge_map import (
    canli_satir,
    olculen_kenar_siniflari,
    olculen_kenarlar,
    olculen_units_ekseni,
    tablo_tutarsizliklari,
    taranamayanlar,
)
from qlab.engine.md.forcefields import Harmonic, Morse
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.engine.tdse.grid import Grid1D

PROJE_KOKU = Path(__file__).resolve().parents[2]
KAYNAK = PROJE_KOKU / "src" / "qlab"
ESKI_SURUM = "648f03b"
CZ, YT = "calisma_zamani", "yalniz_tip"


def _agac(kok: Path, dosyalar: dict[str, str]) -> Path:
    """Sentetik bir `engine/` ağacı kur: {'md/k.py': '<kaynak>'}."""
    for yol, govde in dosyalar.items():
        hedef = kok / yol
        hedef.parent.mkdir(parents=True, exist_ok=True)
        hedef.write_text(govde, encoding="utf-8")
    return kok


def _tutarsizliklar(kok: Path) -> tuple[str, ...]:
    return tablo_tutarsizliklari(
        olculen_kenarlar(kok),
        olculen_units_ekseni(kok),
        taranamayan=taranamayanlar(kok),
        siniflar=olculen_kenar_siniflari(kok),
    )


@pytest.fixture(scope="module")
def eski():
    """648f03b'nin `bridge_map.py`si — git'ten okunur ve depo dışı bir modül
    nesnesinde yürütülür (dataclass `sys.modules` ister). git yoksa test
    ATLANIR ve bu 'ölçülmedi' demektir, 'yeşil' değil."""
    try:
        kaynak = subprocess.run(
            ["git", "show", f"{ESKI_SURUM}:src/qlab/bridge_map.py"],
            cwd=PROJE_KOKU, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as hata:
        pytest.skip(f"eski tarayıcı okunamadı — ÖLÇÜLMEDİ: {hata}")
    modul = types.ModuleType(f"_eski_bridge_map_{ESKI_SURUM}")
    modul.__file__ = str(KAYNAK / "bridge_map.py")
    sys.modules[modul.__name__] = modul
    try:
        kod = compile(kaynak, f"{ESKI_SURUM}:bridge_map.py", "exec", dont_inherit=True)
        exec(kod, modul.__dict__)
        yield modul
    finally:
        sys.modules.pop(modul.__name__, None)


# ---------------------------------------------------------------------------
# (a) DÜZ BİÇİMLER — eski kaçırır, yeni yakalar
# ---------------------------------------------------------------------------

#: (ad, dosyalar, eski de görür mü). Her ağaçta beklenen TEK kenar md -> tb.
DUZ_BICIMLER = [
    ("from_qlab_engine_import_paket", {"md/k.py": "from qlab.engine import tb\n"}, False),
    ("goreli_ust_paket_adi", {"md/k.py": "from .. import tb\n"}, False),
    ("alt_paket_mutlak", {"md/alt/__init__.py": "",
                          "md/alt/k.py": "from qlab.engine.tb.radyal import Y\n"}, False),
    ("alt_paket_goreli", {"md/alt/__init__.py": "",
                          "md/alt/k.py": "from ...tb import radyal\n"}, False),
    ("qlab_zinciri", {"md/k.py": "import qlab.units\n\n\ndef f():\n"
                                 "    return qlab.engine.tb.radyal\n"}, False),
    ("import_mutlak", {"md/k.py": "import qlab.engine.tb.radyal as r\n"}, True),
    ("from_mutlak", {"md/k.py": "from qlab.engine.tb.radyal import Y\n"}, True),
    ("goreli_capraz", {"md/k.py": "from ..tb import radyal\n"}, True),
]


@pytest.mark.parametrize(
    "ad, dosyalar, eski_gorur", DUZ_BICIMLER, ids=[b[0] for b in DUZ_BICIMLER]
)
def test_a_DUZ_BICIM_yeni_YAKALAR_eski_KACIRIR(eski, tmp_path, ad, dosyalar, eski_gorur):
    """Eskinin kaçırdığı beş biçim (`from qlab.engine import X`, ada paket
    olarak inen göreli import, alt paket — mutlak ve göreli —, `qlab.engine.X`
    zinciri) yenide kenar olur; eskinin gördüğü üç biçim iki sürümde AYNI
    sonucu verir (gerileme yok). Hiçbiri yasak değildir."""
    kok = _agac(tmp_path / "engine", {"tb/__init__.py": "", **dosyalar})
    assert olculen_kenarlar(kok) == {("md", "tb")}
    assert taranamayanlar(kok) == ()
    eski_kenarlar = eski.olculen_kenarlar(kok)
    if eski_gorur:
        assert eski_kenarlar == {("md", "tb")}
    else:
        assert ("md", "tb") not in eski_kenarlar, f"eski de görüyor: {ad}"


def test_a_UNITS_EKSENI_alt_paket_ve_from_qlab_import_units(eski, tmp_path):
    """Aynı çözüm `qlab.units` eksenine de uygulanır: `from qlab import
    units` ve alt paket dosyası eskide SIFIR sayılıyordu; üst dizine inen
    göreli import iki sürümde de sayılır (kontrol)."""
    kok = _agac(tmp_path / "engine", {
        "hf/k.py": "from qlab import units\n",
        "md/alt/k.py": "from qlab.units import BOHR\n",
        "tb/k.py": "from ...units import BOHR\n",
    })
    assert olculen_units_ekseni(kok) == {"hf": 1, "md": 1, "tb": 1}
    assert eski.olculen_units_ekseni(kok) == {"hf": 0, "md": 0, "tb": 1}


def test_a_KONTROL_ayni_motor_goreli_ve_DIZGEDEKI_import_kenar_URETMEZ(eski, tmp_path):
    """Dokstring, yorum ve dizge içindeki import METNİ sayılmaz (`ast`);
    aynı motora göreli import kenar değildir. İki sürüm de boş döner."""
    kok = _agac(tmp_path / "engine", {
        "md/__init__.py": "",
        "md/state.py": "X = 1\n",
        "md/k.py": (
            '"""from qlab.engine.tb import radyal -- dokstring."""\n'
            "# import qlab.engine.hf.scf\n"
            "from . import state\n"
            "from .state import X\n"
            "YOL = 'qlab.engine.tb.radyal'\n"
        ),
    })
    assert olculen_kenarlar(kok) == frozenset() == eski.olculen_kenarlar(kok)
    assert taranamayanlar(kok) == ()


# ---------------------------------------------------------------------------
# (b) YASAK BİÇİMLER — adıyla, fail-closed
# ---------------------------------------------------------------------------

#: (kod, bulguda geçmesi gereken parça). Kod `md/k.py`nin 2. satırındadır
#: (1. satır `import typing`, kendisi yasak DEĞİL).
YASAKLAR = [
    ("import importlib", "`import importlib`"),
    ("import importlib.util", "`import importlib.util`"),
    ("from importlib import import_module", "`from importlib import`"),
    ("import runpy", "`import runpy`"),
    ("import pkgutil", "`import pkgutil`"),
    ("import imp", "`import imp`"),
    ("import zipimport", "`import zipimport`"),
    ("import builtins", "`import builtins`"),
    ("import sys", "`import sys`"),
    ("import pydoc", "`import pydoc`"),
    ("import qlab.units; E = qlab.engine", "zincirsiz `qlab.engine`"),
    ("import qlab.units; G = getattr(qlab.engine, 'tb')", "zincirsiz `qlab.engine`"),
    ("import qlab.units; Q = qlab", "ciplak `qlab` adi"),
    ("import qlab.units; G = getattr(qlab, 'engine')", "ciplak `qlab` adi"),
    ("from sys import modules", "`from sys import`"),
    ("m = __import__('qlab.engine.tb')", "`__import__`"),
    ("exec('x = 1')", "`exec`"),
    ("eval('1')", "`eval`"),
    ("c = compile('1', 'k', 'eval')", "`compile`"),
    ("v = vars()", "`vars`"),
    ("g = globals()", "`globals`"),
    ("y = locals()", "`locals`"),
    ("b = __builtins__", "`__builtins__`"),
    ("from typing import TYPE_CHECKING", "TYPE_CHECKING"),
    ("T = typing.TYPE_CHECKING", "`TYPE_CHECKING`"),
    ("import qlab", "`import qlab`"),
    ("import qlab.engine as e", "`import qlab.engine`"),
    ("from qlab import engine", "`from qlab import engine`"),
    ("from qlab import *", "`from qlab import *`"),
    ("from qlab.engine import *", "`from qlab.engine import *`"),
    ("from .. import *", "`from qlab.engine import *`"),
    ("from ... import engine", "`from qlab import engine`"),
    ("from .... import x", "goreli import ust paketin disina tasiyor"),
    ("def (:", "ayristirilamadi"),
]


@pytest.mark.parametrize("kod, parca", YASAKLAR, ids=[k for k, _ in YASAKLAR])
def test_b_YASAK_BICIM_ADIYLA_raporlanir_ve_TABLOYU_KIRMIZIYA_cevirir(tmp_path, kod, parca):
    """Göreli biçimler ÇÖZÜLMÜŞ adıyla yakalanır: `from .. import *`
    `from qlab.engine import *`tır. Ayrıştırılamayan dosya da taranamayandır
    (satır 0) — görülemeyen bir dosya tabloyu doğrulayamaz. `qlab` adı
    yalnız bir `qlab.engine.X` / `qlab.units` zincirinin başı olabilir:
    takma ad ve `getattr(qlab…)` zinciri kırar, kenar görünmez olurdu."""
    kok = _agac(tmp_path / "engine", {"md/k.py": f"import typing\n{kod}\n"})
    (bulgu,) = taranamayanlar(kok)
    satir = 0 if parca == "ayristirilamadi" else 2
    assert bulgu.startswith(f"md/k.py:{satir} ") and parca in bulgu, bulgu
    assert f"TARANAMAYAN: {bulgu}" in _tutarsizliklar(kok)


def test_b_YASAKSIZ_ESDEGER_AGAC_BOS_yanlis_pozitif_yok(tmp_path):
    """Yasak adlar yalnız dizge/yorum/dokstring içinde, ya da ad olmayan
    yerlerde (`re.compile`, `obj.eval()`, `def eval`) geçiyor; `getattr`
    SERBEST; `qlab.units.BOHR` zinciri `qlab` adının izinli kullanımı.
    Taranamayan yol yok, düz kenar yine görülüyor."""
    govde = (
        '"""import importlib; __import__("x"); TYPE_CHECKING; from qlab import engine"""\n'
        "# import sys ; exec('x') ; from qlab.engine import *\n"
        "import re\n"
        "import typing\n"
        "import qlab.units\n"
        "from qlab.engine.tb import radyal\n"
        "B = qlab.units.BOHR\n"
        "DESEN = re.compile('x')\n"
        "METIN = 'eval(1) vars() globals() locals() import builtins'\n"
        "\n\nclass K:\n"
        "    def eval(self):\n"
        "        return getattr(self, 'ad', None) or radyal.eval()\n"
        "\n\nT = typing.List\n"
    )
    kok = _agac(tmp_path / "engine", {"md/k.py": govde, "tb/__init__.py": ""})
    assert taranamayanlar(kok) == ()
    assert olculen_kenarlar(kok) == {("md", "tb")}


@pytest.mark.parametrize("dosya, govde, beklenen", [
    ("__init__.py", "import importlib\n", ("__init__.py:1 `import importlib`",)),
    ("ortak.py", "E = qlab.engine\n", ("ortak.py:1 zincirsiz `qlab.engine`",)),
    ("__init__.py", '"""Yalnız dokstring: import importlib."""\n', ()),
], ids=["kok_init_importlib", "kok_ortak_zincirsiz", "kok_init_YASAKSIZ"])
def test_b_ENGINE_KOKU_DOSYALARI_YASAK_icin_TARANIR_kenar_URETMEZ(tmp_path, dosya, govde, beklenen):
    """`engine/__init__.py` (PEP 562 `__getattr__` + importlib) ya da kökteki
    bir yardımcı, motor ağacına dışarıdan bir yükleme yolu açabilirdi; kök
    dosyalar yalnız yasaklar için taranır. Son satır yasaksız kontrol."""
    kok = _agac(tmp_path / "engine", {dosya: govde, "md/k.py": ""})
    assert taranamayanlar(kok) == beklenen
    assert all(f"TARANAMAYAN: {b}" in _tutarsizliklar(kok) for b in beklenen)
    assert olculen_kenarlar(kok) == frozenset()


# ---------------------------------------------------------------------------
# (c) GERÇEK AĞAÇ
# ---------------------------------------------------------------------------


def test_c_GERCEK_AGACTA_TARANAMAYAN_YOL_SIFIR():
    """Ölçülmüş bir sıfır: tarama gerçek ağacın dosyalarını GÖRDÜ (kenar ve
    eksen çıkıyor) ve hiçbirinde yasak biçim yok. git gerektirmez."""
    assert taranamayanlar() == ()
    assert olculen_kenarlar() and any(olculen_units_ekseni().values())


def test_c_GERCEK_AGAC_648f03b_ile_AYNI(eski):
    """Gerçek ağaçta yeni tarayıcı 648f03b'ninkiyle AYNI kenarı ve ekseni
    verir: tarama genişledi ama gerçek tablo kıpırdamadı (gerileme yok)."""
    assert olculen_kenarlar() == eski.olculen_kenarlar()
    assert olculen_units_ekseni() == eski.olculen_units_ekseni()


# ---------------------------------------------------------------------------
# (d) KENAR SINIFLAYICISI + KİLİT 5 + KİLİT 2
# ---------------------------------------------------------------------------

GELECEK = "from __future__ import annotations\n"
ANNOTASYON = GELECEK + "from qlab.engine.md.state import FF\n\n\ndef f(x: FF) -> None:\n    return None\n"

#: (ad, `tdse/k.py`, ek dosyalar, beklenen `tdse -> md` sınıfı).
SINIF_VAKALARI = [
    ("arguman_annotasyonu", ANNOTASYON, {}, YT),
    ("donus_ve_AnnAssign", GELECEK + "from qlab.engine.md.state import FF\n\n\n"
     "def f() -> FF:\n    y: FF = 1\n    return y\n", {}, YT),
    ("future_YOK", "from qlab.engine.md.state import FF\n\n\ndef f(x: FF):\n    return x\n", {}, CZ),
    ("calisma_yuvasinda", GELECEK + "from qlab.engine.md.state import FF\n\n\n"
     "def f(x: FF):\n    return isinstance(x, FF)\n", {}, CZ),
    ("hic_kullanilmiyor", GELECEK + "from qlab.engine.md.state import FF\n", {}, CZ),
    ("modul_importu", GELECEK + "import qlab.engine.md.state as s\n\n\n"
     "def f(x: s.FF):\n    return x\n", {}, CZ),
    ("yildiz", GELECEK + "from qlab.engine.md.state import *\n", {}, CZ),
    ("baska_dosya_adiyla_aliyor", ANNOTASYON,
     {"tdse/b.py": "from qlab.engine.tdse.k import FF\n"}, CZ),
    ("baska_dosya_goreli_aliyor", ANNOTASYON, {"tdse/b.py": "from .k import FF\n"}, CZ),
    ("baska_dosya_yildizla_aliyor", ANNOTASYON, {"tdse/b.py": "from .k import *\n"}, CZ),
    ("getattr_dizgesi", ANNOTASYON,
     {"tdse/b.py": "import qlab.engine.tdse.k as k\nG = getattr(k, 'FF')\n"}, CZ),
    ("oznitelik", ANNOTASYON, {"tdse/b.py": "from qlab.engine.tdse import k\nG = k.FF\n"}, CZ),
    ("iki_site_biri_calisma", ANNOTASYON,
     {"tdse/b.py": "from qlab.engine.md.state import FF\nX = FF\n"}, CZ),
    ("del", ANNOTASYON + "del FF\n", {}, CZ),
    ("augassign", ANNOTASYON + "FF += ()\n", {}, CZ),
    # Sınıfın anlamı tek yönlü: veri akışını değiştirmeyen iki düzenleme de
    # sınıfı ÇALIŞMA ZAMANI'na çevirir (`GERCEK_KOPRU_KENAR_SINIFLARI` yorumu).
    ("dizge_annotasyon", ANNOTASYON.replace("x: FF", "x: 'FF'"), {}, CZ),
    ("modul_yuvasi", ANNOTASYON + "_ = FF\n", {}, CZ),
]


@pytest.mark.parametrize(
    "ad, govde, ekler, beklenen", SINIF_VAKALARI, ids=[v[0] for v in SINIF_VAKALARI]
)
def test_d_SINIFLAYICI(tmp_path, ad, govde, ekler, beklenen):
    """KURAL: ad YALNIZ annotasyonda (arguman, donus, `AnnAssign`) ve en az
    bir kez yükleniyorsa, dosyada `from __future__ import annotations` varsa
    ve ad dosyadan ÇIKMIYORSA yalnız tip; modül importu, yıldız ve her
    şüpheli durum çalışma zamanı (muhafazakâr yön). `del ad` ve `ad += …`
    de çalışma zamanı kullanımıdır (import silinince NameError)."""
    kok = _agac(tmp_path / "engine", {
        "md/__init__.py": "", "md/state.py": "class FF:\n    pass\n",
        "tdse/k.py": govde, **ekler,
    })
    assert olculen_kenar_siniflari(kok) == {("tdse", "md"): beklenen}
    assert taranamayanlar(kok) == ()


def test_d_GERCEK_AGAC_SINIFLARI_ve_KILIT_5_YESIL():
    """Ölçüldü (26.09.2026): üç kenar çalışma zamanı, `tdse -> md` yalnız
    tip (`ForceField` yalnız bir argüman annotasyonu). Tablo kodla tutarlı."""
    siniflar = olculen_kenar_siniflari()
    assert siniflar == {("md", "hf"): CZ, ("md", "tb"): CZ, ("tb", "hf"): CZ, ("tdse", "md"): YT}
    assert _tutarsizliklar(bm._MOTOR_KOKU) == ()


@pytest.mark.parametrize("bozma, parca", [
    ("sinif_cevrildi", "KENAR SINIFI: tdse -> md beyan yalniz_tip, olculen calisma_zamani"),
    ("sinif_eksik", "KENAR SINIFI: sinif sozlugu kenar kumesiyle ayni degil"),
    ("beyan_eksik", "KENAR SINIFI: md -> tb beyan YOK, olculen calisma_zamani"),
])
def test_d_KILIT_5_BAGLAYICI(monkeypatch, bozma, parca):
    """Kilit 5 GERÇEK_KÖPRÜ kenarlarının ölçülen sınıfını beyanla karşılaştırır;
    sınıf, sözlük ya da beyan bozulunca ADIYLA kırmızı."""
    siniflar = olculen_kenar_siniflari()
    if bozma == "sinif_cevrildi":
        siniflar[("tdse", "md")] = CZ
    elif bozma == "sinif_eksik":
        del siniflar[("md", "hf")]
    else:
        beyan = dict(bm.GERCEK_KOPRU_KENAR_SINIFLARI)
        del beyan[("md", "tb")]
        monkeypatch.setattr(bm, "GERCEK_KOPRU_KENAR_SINIFLARI", beyan)
    sorunlar = tablo_tutarsizliklari(
        olculen_kenarlar(), olculen_units_ekseni(), taranamayan=(), siniflar=siniflar
    )
    assert parca in sorunlar, sorunlar


def test_d_KILIT_2_DEGISMEDI_yalniz_tip_kenari_KOD_KENARI_sayilir():
    """`tdse -> md` yalnız tip ve kilit 2 onu yine kod kenarı sayar (kilit
    2'nin çalışma zamanı kenarı isteyip istemeyeceği Osman'ın kararı;
    varsayılan hayır). Bağlayıcılık: kenar gerçekten yok olursa kilit 2
    ateşlenir."""
    kenarlar, siniflar = olculen_kenarlar(), olculen_kenar_siniflari()
    assert siniflar[("tdse", "md")] == YT
    eksen = olculen_units_ekseni()
    tamam = tablo_tutarsizliklari(kenarlar, eksen, taranamayan=(), siniflar=siniflar)
    assert not any("icin kod kenari YOK" in s for s in tamam)
    eksik = kenarlar - {("tdse", "md")}
    sorunlar = tablo_tutarsizliklari(
        eksik, eksen, taranamayan=(), siniflar={k: siniflar[k] for k in eksik}
    )
    assert "GERCEK_KOPRU ('md', 'tdse') icin kod kenari YOK" in sorunlar


#: Annotasyonu DEĞERLENDİREN tüketiciler. Biri `src/qlab`a girerse yalnız
#: tip bir ad çalışma zamanında GEREKİR ve hüküm geçersizleşir. TAM LİSTE
#: burasıdır; README, `bridge_map` ve sdd notları buraya gönderme yapar.
TUKETICILER = {
    "get_type_hints", "get_annotations", "eval_str", "singledispatch",
    "singledispatchmethod", "__annotations__", "__annotate__", "annotationlib",
}


def _tuketiciler(kok: Path) -> list[str]:
    bulgular = []
    for dosya in sorted(kok.rglob("*.py")):
        for d in ast.walk(ast.parse(dosya.read_bytes())):
            if isinstance(d, ast.Name):
                ad = d.id
            elif isinstance(d, ast.Attribute):
                ad = d.attr
            elif isinstance(d, ast.keyword):
                ad = d.arg
            elif isinstance(d, ast.alias):
                ad = d.name.rpartition(".")[2]
            else:
                continue
            if ad in TUKETICILER:
                bulgular.append(f"{dosya.relative_to(kok).as_posix()}:{d.lineno} {ad}")
    return bulgular


def test_d_ANNOTASYON_DEGERLENDIREN_TUKETICI_src_qlab_CAPINDA_YASAK_ve_SIFIR(tmp_path):
    """Yalnız-tip hükmü KOŞULLUDUR: `get_type_hints`, `inspect.get_annotations`,
    `eval_str=True`, `singledispatch` (annotasyondan kayıt) ya da
    `__annotations__` okuyan bir kod o adı çalışma zamanında ister. `src/qlab`
    çapında sayıları SIFIR; bağlayıcılık sentetik bir dosyayla gösterilir."""
    assert _tuketiciler(KAYNAK) == []
    kok = _agac(tmp_path / "src", {"a.py": (
        "import inspect\nimport typing\nfrom functools import singledispatch\n"
        "typing.get_type_hints(f)\ninspect.signature(f, eval_str=True)\n"
        "inspect.get_annotations(f)\nx = f.__annotations__\n"
    )})
    bulunan = {b.rpartition(" ")[2] for b in _tuketiciler(kok)}
    assert bulunan == {"singledispatch", "get_type_hints", "eval_str", "get_annotations",
                       "__annotations__"}


def test_d_CANLI_SATIR_SINIF_ve_TARANAMAYAN_parcalari_GIRDIDEN_yazilir():
    """Canlı satır sınıfları ve taranamayan sayısını SABİT basmaz, girdiden
    yazar (gerçek ağacın metnini sabit basan bir satır burada kırmızı)."""
    taban = (1.0, 1.0, 1.0, "SAGLAM", "boru")
    satir = canli_satir(
        kova_sayim={}, kenarlar={("a", "b")}, eksen={}, a1=taban, a2=taban,
        gercek_hata=1.0, tutarsizliklar=("TARANAMAYAN: q/r.py:3 `eval`",),
        taranamayan=("q/r.py:3 `eval`", "q/s.py:0 ayristirilamadi"),
        siniflar={("a", "b"): YT},
    )
    assert "taranamayan 2; kenar siniflari a -> b yalniz_tip;" in satir
    assert "TABLO KODLA CELISIYOR: TARANAMAYAN: q/r.py:3 `eval`" in satir


# ---------------------------------------------------------------------------
# (e) YÜK TAŞIMA ÇIPASI
# ---------------------------------------------------------------------------


def _yukle(gercek: types.ModuleType, sil: str | None, monkeypatch) -> types.ModuleType:
    """`gercek`in kaynağını, `sil` modülünden yapılan tepe düzey `ImportFrom`
    SİLİNMİŞ hâlde yeni bir modül olarak yürütür. `sys.modules`a kaydedilir
    (dataclass ister); `dont_inherit=True`: bu dosyanın `__future__`ı
    sızmasın."""
    yol = Path(gercek.__file__)
    agac = ast.parse(yol.read_bytes(), filename=str(yol))
    once = len(agac.body)
    agac.body = [
        d for d in agac.body if not (isinstance(d, ast.ImportFrom) and d.module == sil)
    ]
    assert once - len(agac.body) == (0 if sil is None else 1), "silme olmadı"
    modul = types.ModuleType(f"{gercek.__name__}_yamali")
    modul.__file__ = str(yol)
    monkeypatch.setitem(sys.modules, modul.__name__, modul)
    exec(compile(agac, str(yol), "exec", dont_inherit=True), modul.__dict__)
    return modul


def _potansiyeller(modul: types.ModuleType) -> list[tuple]:
    """İki alan (Morse, Harmonic) × bir ızgara; katlanan noktası olan pencere
    (L = 4, r_e = 1.4 -> x < -r_e noktaları var). Karşılaştırma BAYT düzeyinde."""
    cikti = []
    for alan in (Morse(d_e=0.17, r_e=1.4, a=1.0), Harmonic(k=0.3, center=np.zeros((2, 3)))):
        k = modul.potential_from_force_field(
            Grid1D(n=128, length=4.0), alan, r_e=1.4, masses=np.array([1837.0, 1837.0])
        )
        cikti.append((k.potential.dtype.str, k.potential.tobytes(), k.reduced_mass.hex(),
                      k.r_e.hex(), k.katlanan_nokta, k.katlanma_esigi.hex()))
    return cikti


def test_e_YUK_TASIMA_CIPASI_tdse_md_importu_silinince_BIT_BIT_AYNI(monkeypatch):
    """`from_md.py`nin `from qlab.engine.md.state import ForceField` satırı
    silinince modül BİT BİT aynı çalışıyor: kenar YALNIZ TİP, veri akışı ördek
    tiplemeyle. ARAÇ KONTROLÜ: yamasız kaynak aynı yükleyiciyle gerçekle bit
    bit aynı (yükleyici kendisi bir fark üretmiyor)."""
    gercek = _potansiyeller(from_md_gercek)
    assert gercek[0][4] > 0, "pencere katlanma yolunu sınamıyor"
    assert _potansiyeller(_yukle(from_md_gercek, None, monkeypatch)) == gercek
    yamali = _yukle(from_md_gercek, "qlab.engine.md.state", monkeypatch)
    assert "ForceField" not in vars(yamali) and "ForceField" in vars(from_md_gercek)
    assert _potansiyeller(yamali) == gercek


def test_e_POZITIF_KONTROL_md_tb_importu_silinince_NameError(monkeypatch):
    """Aynı yükleyici YÜK TAŞIYAN bir importu görür: `tb_kuvvet.py`nin
    `from qlab.engine.tb.mermin import ...` satırı silinince çağrı NameError
    verir (kenar ÇALIŞMA ZAMANI). Kontrol: yamasız kopya gerçekle bit bit aynı."""
    ss, itme = UsselRadyalYasa(-0.1, 2.0, 2.0, 3.0, 3.8), UsselRadyalYasa(0.05, 2.0, 6.0, 3.0, 3.8)
    konum = np.zeros((4, 3))
    konum[:, 0] = 2.0 * np.arange(4)
    yamali = _yukle(tb_kuvvet_gercek, "qlab.engine.tb.mermin", monkeypatch)
    with pytest.raises(NameError, match="tek_orbital_mermin"):
        yamali.SsSigmaTBKuvveti(ss, itme, 4.0, 2e-3, 16.0)(konum)
    arac = _yukle(tb_kuvvet_gercek, None, monkeypatch)
    e1, f1 = arac.SsSigmaTBKuvveti(ss, itme, 4.0, 2e-3, 16.0)(konum)
    e0, f0 = tb_kuvvet_gercek.SsSigmaTBKuvveti(ss, itme, 4.0, 2e-3, 16.0)(konum)
    assert float(e1).hex() == float(e0).hex() and f1.tobytes() == f0.tobytes()
