"""`qlab.app.main` CP1252 bir konsolda Türkçe basabiliyor mu — B3 (11.10.2026).

`run.main`in iki CP1252 regresyon testi var (`tests/test_run.py`); `app.main`
aynı `reconfigure_stdio_utf8()` çağrısını taşıyor ama hiçbir test onu
sınamıyordu: çağrı silinse süit yeşil kalırdı. Bu dosya iki akışı da
`errors="strict"` bir CP1252 `TextIOWrapper`ına sarar — CP1252 'ı'yı (U+0131)
TAŞIYAMAZ — ve Qt hiç başlamadan biten iki yolu koşturur:

  * `--help`: argparse yardımı STDOUT'a gider ("YAML senaryo dosyası"),
  * argümansız çağrı: "tam olarak birini verin … dosyası" STDERR'e, kod 1.

Basılan baytlar UTF-8 olarak geri okunur; yalnız "çökmedi" değil, metnin
UTF-8 olarak ulaştığı da sınanır. ÖLÇÜLDÜ (11.10.2026): `app/main.py`deki
`reconfigure_stdio_utf8()` çağrısı silinince iki test de `UnicodeEncodeError`
ile kırmızı.
"""

from __future__ import annotations

import io
import sys

import pytest

from qlab.app import main as app_main


def _cp1252_akis() -> tuple[io.BytesIO, io.TextIOWrapper]:
    tampon = io.BytesIO()
    return tampon, io.TextIOWrapper(tampon, encoding="cp1252", errors="strict")


def test_app_main_YARDIM_STDOUT_cp1252_akista_UTF8_basiliyor(monkeypatch):
    tampon, akis = _cp1252_akis()
    monkeypatch.setattr(sys, "stdout", akis)
    monkeypatch.setattr(sys, "stderr", _cp1252_akis()[1])
    with pytest.raises(SystemExit) as cikis:
        app_main.main(["--help"])
    sys.stdout.flush()
    assert cikis.value.code == 0
    metin = tampon.getvalue().decode("utf-8")
    assert "senaryo dosyası" in metin, metin


def test_app_main_HATA_STDERR_cp1252_akista_UTF8_basiliyor(monkeypatch):
    tampon, hata = _cp1252_akis()
    monkeypatch.setattr(sys, "stdout", _cp1252_akis()[1])
    monkeypatch.setattr(sys, "stderr", hata)
    assert app_main.main([]) == 1
    sys.stderr.flush()
    metin = tampon.getvalue().decode("utf-8")
    assert "tam olarak birini verin" in metin and "dosyası" in metin, metin
