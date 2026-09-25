"""Test oturumu genelinde konsol kodlaması tuzağını kapatır.

`pytest tests/ -s` capture'ı kapatır; bu durumda print() çıktısı doğrudan
ham konsol koderına gider. Windows'un varsayılan CP1252 konsolunda Türkçe
karakter (ör. 'ı') içeren herhangi bir print() bu yüzden UnicodeEncodeError
ile çöker — `-s` olmadan pytest'in kendi capture katmanı bunu maskeler.

Burada qlab.run::main() içinde zaten kanıtlanmış aynı desen tekrar
kullanılıyor (reconfigure() önce denenir, olmayan ortamlarda TextIOWrapper'a
geri dönülür) böylece TÜM test oturumu için bir kez uygulanır: hem şu anki
tek print() (tests/tdse/test_tunneling.py) hem de gelecekte eklenecek her
Türkçe çıktı için. Dar istisna yakalama bilinçlidir — reconfigure()/wrapping
başarısız olsa bile hiçbir ek hata yükseltilmemelidir.
"""

from __future__ import annotations

import io
import sys


def _reconfigure_utf8(stream_name: str) -> None:
    stream = getattr(sys, stream_name)
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        try:
            if hasattr(stream, "buffer"):
                setattr(
                    sys,
                    stream_name,
                    io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace"),
                )
        except (AttributeError, ValueError, OSError):
            pass


_reconfigure_utf8("stdout")
_reconfigure_utf8("stderr")


# ---------------------------------------------------------------------------
# Referans JSON fixture'lari
#
# Bunlar bir zamanlar `tests/hf/conftest.py`'de yasiyordu ve orada, komut
# satirindaki YOL SIRASINA gore kayboluyorlardi. OLCULDU (pytest 9.1.1):
#     pytest -q tests/hf/test_boys.py tests/test_units.py tests/hf/test_analysis.py
#         -> 68 passed, 32 errors ("fixture 'hf_energies' not found")
#     pytest -q tests/hf/test_analysis.py tests/test_units.py tests/hf/test_scf.py
#         -> 56 passed, 20 errors
# Desen genel: `tests/hf/A`, `tests/B`, `tests/hf/C` verildiginde pytest
# `tests/hf/` dizinine IKINCI girisinde oradaki conftest'i artik baglamiyor.
# Kanonik `pytest -q` (testpaths = ["tests"]) etkilenmiyordu, o yuzden takim
# yesil kaliyor ve delik gorunmuyordu.
#
# Iki "bariz" aciklama OLCULEREK ELENDI: `tests/__init__.py` eklemek de,
# `tests/hf/__init__.py` kaldirmak da hatayi DEGISTIRMEDI (ikisinde de yine
# 32 error). Calisan tek cozum, fixture'lari her kosuda kosulsuz yuklenen KOK
# conftest'e tasimakti -- burasi. Regresyonu
# test_referans_fixturelari_yol_SIRASINDAN_bagimsiz_bulunuyor bagliyor.
#
# `pytest.skip` degil `pytest.fail`: eksik bir referans dosyasi ATLANACAK bir
# kosul degil, bozuk bir ortamdir (ayni gerekce README'de de yazili).
# ---------------------------------------------------------------------------

import ast
import json
from pathlib import Path

import pytest

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"


def _load(name: str) -> dict:
    path = REFERENCE_DIR / name
    if not path.is_file():
        pytest.fail(
            f"Referans dosyasi bulunamadi: {path}\n"
            "tools/generate_reference.py bir Linux makinede kosulup ciktisi "
            "reference/ altina islenmeli. Bkz. reference/README.md"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sto3g_basis_data() -> dict:
    return _load("basis_sto3g.json")


@pytest.fixture(scope="session")
def g631_basis_data() -> dict:
    return _load("basis_631g.json")


@pytest.fixture(scope="session")
def h2_sto3g_integrals() -> dict:
    return _load("integrals_h2_074.json")


@pytest.fixture(scope="session")
def h2o_sto3g_integrals() -> dict:
    return _load("integrals_h2o.json")


@pytest.fixture(scope="session")
def hf_energies() -> dict:
    return _load("hf_energies.json")


# ---------------------------------------------------------------------------
# Sevk tarayıcısı -- TÜRETME KİLİTLERİNİN ORTAK MAKİNESİ
#
# `scenario/schema.py`deki izin demetleri (`POTENTIAL_KINDS`,
# `MD_FORCE_FIELD_KINDS`, `MD_THERMOSTAT_KINDS`) ELLE yazılmak zorundadır:
# katman sözleşmesi `scenario`nun `engine`i import etmesini yasaklar
# (ölçüldü 30.08.2026, `loader.py`ye `import qlab.engine` konarak:
# `lint-imports` rc=1, "Katmanlı mimari BROKEN"). Elle yazılan bir demet
# bayatlayabilir, ve demet bir KAPIYA bağlı olduğu için bayatlık artık
# sessiz değil YIKICIDIR: factory'nin desteklediği bir türü YAML'dan
# yüklenemez kılar.
#
# Bu yüzden doğruluk kaynağı demet değil FACTORY'NİN SEVK DALLARIDIR, ve
# demetler onlara karşı sınanır. Tarayıcı BURADA, kök conftest'te yaşıyor
# çünkü iki ayrı test dizini kullanıyor (`tests/md/` ve `tests/`) ve bu
# dosyanın yukarısında belgelenen ÖLÇÜLMÜŞ ders tam olarak budur: alt
# dizin conftest'lerindeki paylaşımlar yol SIRASINA göre kayboluyordu,
# çalışan tek çözüm kök conftest'ti. İki kopya tutmak ise daha kötüsü
# olurdu: birinde düzeltilip ötekinde unutulan bir kusur demektir.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sevk_tarayici():
    """`<degisken> == "..."` biçimindeki sevk dallarını `ast` ile tarar.

    DÜZ METİNLE DEĞİL: düz metin taraması yorum satırındaki ya da
    dokstringdeki bir adı "sevk dalı" sayardı. Bu deponun `bridge_map.py`si
    aynı bedeli ölçerek öğrendi ve bir plan sayısını çürüttü.

    Ölçüt yapısaldır: verilen işlevin gövdesindeki, solu `degisken` adı ve
    tek işleci `==` olan karşılaştırmaların dizge sabitleri.

    KÖR NOKTASI (ölçüldü, 23.09.2026): solu bir `Name` OLMAYAN dal --
    `x["kind"] == "..."`, `"..." == tur`, `tur in (...)` -- bu tarayıcıya
    GÖRÜNMEZ. Kapatan eşi `dize_karsilastirma_tarayici`dir: "her dizge
    karşılaştırmasının solu sevk değişkeni mi" sorusu onunla sorulur.
    """

    def tara(modul, islev: str, degisken: str) -> set[str]:
        kaynak = Path(modul.__file__).read_text(encoding="utf-8")
        govde = next(
            d
            for d in ast.walk(ast.parse(kaynak))
            if isinstance(d, ast.FunctionDef) and d.name == islev
        )
        turler: set[str] = set()
        for dugum in ast.walk(govde):
            if not isinstance(dugum, ast.Compare):
                continue
            if not (isinstance(dugum.left, ast.Name) and dugum.left.id == degisken):
                continue
            if len(dugum.ops) != 1 or not isinstance(dugum.ops[0], ast.Eq):
                continue
            sag = dugum.comparators[0]
            if isinstance(sag, ast.Constant) and isinstance(sag.value, str):
                turler.add(sag.value)
        return turler

    return tara


@pytest.fixture(scope="session")
def dize_karsilastirma_tarayici():
    """Bir işlevdeki TÜM dizge karşılaştırmaları: `{(sol, dizge), ...}`.

    `sevk_tarayici`nin ÖLÇÜLMÜŞ KÖR NOKTASINI kapatır (md-tb spec
    eleştirmeni P0-1, 23.09.2026): o tarayıcı yalnız solu bir `Name` olan
    `==` karşılaştırmalarını görür. `md/factory.py::provenance_for` kuvvet
    alanı dallarını `scenario.force_field["kind"] == "..."` diye yazıyordu --
    solu bir `Subscript` -- ve bu dalların sevkle eşleştiğini HİÇBİR kilit
    denetlemiyordu. Bu tarayıcı solun biçimine bakmadan, `==`/`!=` ile
    bir dizge sabitine karşılaştırılan HER ifadeyi (sabit hangi tarafta
    olursa olsun) ve `in`/`not in` ile bir dizge demetine/listesine/kümesine
    karşılaştırılan her ifadeyi `(ast.unparse(ifade), dizge)` olarak döndürür.
    Kilit bunun üstüne kurulur: "işlevdeki her dizge karşılaştırmasının solu
    BİLİNEN sevk değişkenidir" -- saklı bir dal (başka bir sol biçimi) o an
    kırmızı olur.
    """

    def tara(modul, islev: str) -> set[tuple[str, str]]:
        kaynak = Path(modul.__file__).read_text(encoding="utf-8")
        govde = next(
            d
            for d in ast.walk(ast.parse(kaynak))
            if isinstance(d, ast.FunctionDef) and d.name == islev
        )
        bulunan: set[tuple[str, str]] = set()
        for dugum in ast.walk(govde):
            if not isinstance(dugum, ast.Compare):
                continue
            ifadeler = [dugum.left, *dugum.comparators]
            for k, islec in enumerate(dugum.ops):
                sol, sag = ifadeler[k], ifadeler[k + 1]
                if isinstance(islec, (ast.Eq, ast.NotEq)):
                    for a, b in ((sol, sag), (sag, sol)):
                        if isinstance(b, ast.Constant) and isinstance(b.value, str):
                            bulunan.add((ast.unparse(a), b.value))
                elif isinstance(islec, (ast.In, ast.NotIn)) and isinstance(
                    sag, (ast.Tuple, ast.List, ast.Set)
                ):
                    for oge in sag.elts:
                        if isinstance(oge, ast.Constant) and isinstance(oge.value, str):
                            bulunan.add((ast.unparse(sol), oge.value))
        return bulunan

    return tara
