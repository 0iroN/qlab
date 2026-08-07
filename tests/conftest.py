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
