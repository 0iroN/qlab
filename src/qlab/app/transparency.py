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
    SIĞMIYOR. Sabit 140 TDSE penceresi için ölçülmüş bir değerdi ve iki
    pencerenin beyanları aynı uzunlukta değil.

    **NE YAPTIĞI, ÖLÇÜLEREK (bu bir üst sınırdır, bir hedef değil).**
    `QPlainTextEdit`'in `sizeHint` yüksekliği içerikten BAĞIMSIZ olarak 192 px;
    orbital penceresinin dizilişinde 3B görünüm esneme payını (stretch) aldığı
    için panele tam olarak o sizeHint kadar yer düşüyor. Yani `max_height`
    yalnızca ALTTAN kırpmayı engelliyor: 960x800'lük gerçek pencerede kutu
    140 iken widget 140 px, 260 iken 192 px oluyor; 260'ı 400'e ya da 600'e
    çıkarmak hiçbir şeyi değiştirmiyor (üçünde de 192 px). 260 sayısının tek
    işlevi "192'nin altına inme" demektir.

    **HÂLÂ TAŞIYOR, ve bu dürüstçe yazılmalı.** Aynı ölçümde orbital
    penceresinin beyanı 960 px genişlikte 19 sarılmış satır sürüyordu ve
    140 px'lik kutuda 10, 260 px'lik kutuda 7 satır katlanın altında
    kalıyordu. Yani düzeltme taşmayı AZALTTI, ORTADAN KALDIRMADI; README'nin
    "düzeltildi" ifadesi bu ayrımı yapmıyor. Tamamen kaldırmak dizilişi
    değiştirmeyi (panele esneme payı vermeyi ya da 3B görünümü küçültmeyi)
    gerektirir; bu bir gözden geçirme düzeltmesi değil bir tasarım kararıdır
    ve burada VERİLMEDİ. Sınıf hâlâ "okunmayan bir beyan, beyan etmiş
    sayılmaz" diyor -- o ölçüt bugün TAM OLARAK karşılanmıyor.
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

    def text(self) -> str:
        """Panelin ŞU AN bastığı metin.

        Testlerin okuma noktası. Bir pencere "panelin bastığı metin budur"
        diye bir metot sunuyorsa onu `format_panel`'i ikinci kez çağırarak
        değil BURADAN üretmelidir: ölçüldü, ikinci çağrı deseni
        `set_provenance`'ı hiç çağırmayan bir pencereyi de yeşil bırakıyor
        (panel 0 karakter, metot 1174 karakter, dokuz testin dokuzu geçiyor).
        """
        return self._text.toPlainText()
