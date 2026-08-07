"""Şeffaflık paneli — spec §9.

Panel kendi başına hiçbir fizik bilgisi üretmez; yalnızca Provenance'ı basar.
Bu ayrım kasıtlı: gösterilen yaklaşım listesi motorun beyanıdır, arayüzün
tahmini değil.
"""

from __future__ import annotations

from PySide6 import QtWidgets

from qlab.provenance import Provenance

_LABEL_WIDTH = 14


def format_panel(provenance: Provenance) -> str:
    """Panelin metin gösterimi. Saf fonksiyon — GUI olmadan test edilir."""
    return "\n".join(
        f"{label:<{_LABEL_WIDTH}}: {value}" for label, value in provenance.as_lines()
    )


class TransparencyPanel(QtWidgets.QGroupBox):
    """Ana pencerede kalıcı panel; kapatılamaz."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Bu hesap hangi varsayımlar altında yapıldı?", parent)
        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet("font-family: Consolas, monospace;")
        self._text.setMaximumHeight(140)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._text)

    def set_provenance(self, provenance: Provenance) -> None:
        self._text.setPlainText(format_panel(provenance))
