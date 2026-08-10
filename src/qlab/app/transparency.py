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


#: TDSE penceresinin beyanı bu yüksekliğe sığıyor. Orbital penceresi daha uzun
#: bir beyan taşıdığı için kendi değerini verir.
_VARSAYILAN_YUKSEKLIK = 140


class TransparencyPanel(QtWidgets.QGroupBox):
    """Ana pencerede kalıcı panel; kapatılamaz.

    `max_height` neden bir parametre: kabul kriteri 3'te pencere İLK KEZ gerçek
    bir ekranda açıldığında görüldü ki orbital penceresinin beyanı kutuya
    SIĞMIYOR -- ölçüldü, 12 satırlık içerik 140 px'lik (≈8 satır) kutunun
    altına taşıyor ve son satırlar ancak kaydırılarak okunabiliyordu. Kaydırma
    çubuğu bir çözüm değil: okunmayan bir beyan, beyan etmiş sayılmaz. Sabit
    140 TDSE penceresi için ölçülmüş bir değerdi ve iki pencerenin beyanları
    aynı uzunlukta değil.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        max_height: int = _VARSAYILAN_YUKSEKLIK,
    ) -> None:
        super().__init__("Bu hesap hangi varsayımlar altında yapıldı?", parent)
        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet("font-family: Consolas, monospace;")
        self._text.setMaximumHeight(max_height)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._text)

    def set_provenance(self, provenance: Provenance) -> None:
        self._text.setPlainText(format_panel(provenance))
