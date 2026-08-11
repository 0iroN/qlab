"""Şeffaflık paneli — spec §9.

Panel kendi başına hiçbir fizik bilgisi üretmez; yalnızca Provenance'ı basar.
Bu ayrım kasıtlı: gösterilen yaklaşım listesi motorun beyanıdır, arayüzün
tahmini değil.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from qlab.provenance import Provenance

_LABEL_WIDTH = 14


def format_panel(provenance: Provenance) -> str:
    """Panelin metin gösterimi. Saf fonksiyon — GUI olmadan test edilir."""
    return "\n".join(
        f"{label:<{_LABEL_WIDTH}}: {value}" for label, value in provenance.as_lines()
    )


#: TDSE penceresinin beyanı bu yüksekliğe sığıyor (ölçüldü: 95 px gerekiyor).
#: Orbital penceresi daha uzun bir beyan taşıdığı için kendi değerini verir.
_VARSAYILAN_YUKSEKLIK = 140

_BASLIK = "Bu hesap hangi varsayımlar altında yapıldı?"


class TransparencyPanel(QtWidgets.QGroupBox):
    """Ana pencerede kalıcı panel; kapatılamaz.

    `max_height` neden bir parametre: kabul kriteri 3'te pencere İLK KEZ gerçek
    bir ekranda açıldığında görüldü ki orbital penceresinin beyanı kutuya
    SIĞMIYOR. Sabit 140 TDSE penceresi için ölçülmüş bir değerdi ve iki
    pencerenin beyanları aynı uzunlukta değil.

    **Yükseklik artık `sizeHint`'ten değil METİNDEN geliyor.** `QPlainTextEdit`'in
    `sizeHint` yüksekliği içerikten BAĞIMSIZ olarak 192 px'tir ve orbital
    penceresinin dizilişinde 3B görünüm esneme payını (stretch) aldığı için
    panele tam olarak o kadar yer düşüyordu. Ölçüldü: `max_height` 260, 400 ve
    600 AYNI sonucu veriyordu (üçünde de 192 px); yalnızca 140 -- sizeHint'in
    ALTI -- bir fark yaratıyordu. Yani sayıyı büyütmek alan EKLEMİYOR, sadece
    alan çalmayı bırakıyordu. Panel bu yüzden gereken yüksekliği metinden
    hesaplayıp `minimumHeight` olarak dayatır; `maximumHeight` üst sınır
    (`max_height`) olarak kalır.

    **Gereken yükseklik GENİŞLİĞE bağlıdır**, çünkü sarılma ona bağlıdır.
    Orbital penceresinin beyanı için ikili aramayla ölçüldü: 1600 px'te 207,
    1000 px'te 305, 960 px'te 316, 700 px'te 439 px. Tek bir sabit her
    genişlikte yanlış olurdu; `resizeEvent` yeniden hesaplar.

    **`max_height` gerçek bir üst sınırdır** ve 3B görünümün ezilmesini
    engeller. Sınıra dayanıldığında panel kırpar -- ama SESSİZCE DEĞİL: kutu
    başlığı kaç satırın gizlendiğini yazar. "Okunmayan bir beyan, beyan etmiş
    sayılmaz" ölçütü, gizlenenin VARLIĞINI duyurmayı da kapsar. Metnin kendisi
    (`text()`) kirletilmez; ona karşı eşitlik iddiaları var.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        max_height: int = _VARSAYILAN_YUKSEKLIK,
    ) -> None:
        super().__init__(_BASLIK, parent)
        self._max_height = int(max_height)
        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet("font-family: Consolas, monospace;")
        self._text.setMaximumHeight(self._max_height)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._text)

        # Yerleşim `showEvent`'ten SONRA da oturabildiği için bir kez daha
        # bakmak gerekiyor (bkz. showEvent). Zamanlayıcı BU PANELE AİT olmak
        # zorunda: sahipsiz `QTimer.singleShot` denendi ve tam takımda
        # **erişim ihlaliyle çökertti** (test_daha_yuksek_kutu_...'daki
        # `processEvents()`), çünkü testler panelleri döngüde yaratıp referans
        # tutmuyor ve bekleyen atış çöpe atılmış bir widget üzerinde
        # ateşleniyordu. Qt, çocuk zamanlayıcıyı ebeveyniyle birlikte yok eder.
        self._gecikmeli = QtCore.QTimer(self)
        self._gecikmeli.setSingleShot(True)
        self._gecikmeli.setInterval(0)
        self._gecikmeli.timeout.connect(self._yuksekligi_icerige_uydur)

    def set_provenance(self, provenance: Provenance) -> None:
        self._text.setPlainText(format_panel(provenance))
        self._yuksekligi_icerige_uydur()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt'nin adlandirmasi
        """Genişlik değişti, sarılma da değişti: yükseklik yeniden hesaplanmalı."""
        super().resizeEvent(event)
        self._yuksekligi_icerige_uydur()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt'nin adlandirmasi
        """`resizeEvent` TEK BAŞINA yetmiyor -- ölçüldü.

        `resize()` çağrılıp SONRA `show()` çağrıldığında `resizeEvent` pencere
        henüz gizliyken tetikleniyor ve o anda `viewport().width()` gerçeği
        yansıtmıyor (ölçüldü: gösterimden önce 638 px, sonra 556 px -- ilki
        panelin kendi genişliğinden bile büyük). Sonuç, yükseklik 120 px'lik
        sınıra dayanması gerekirken 94'te kalıyor ve başlıktaki kırpma uyarısı
        hiç yazılmıyordu.
        """
        super().showEvent(event)
        self._yuksekligi_icerige_uydur()
        # Yerleşim `showEvent`'ten SONRA da oturabiliyor -- ölçüldü: 800x700'lük
        # orbital penceresinde bu ikinci geçiş olmadan panel 304 px'te kalıyor
        # (olması gereken 340), kaydırma çubuğu 6 satır gizliyor ve başlıktaki
        # uyarı hiç yazılmıyordu. Zamanlayıcı olay kuyruğu boşaldığında -- yani
        # geometri kesinleştiğinde -- bir kez daha bakar. Özyineleme riski yok:
        # `_yuksekligi_icerige_uydur` yalnızca DEĞER değiştiğinde
        # `setMinimumHeight` çağırır, yani sabit noktaya oturur.
        self._gecikmeli.start()

    def _gereken_yukseklik(self) -> int:
        """Metnin tamamının KAYDIRMASIZ sığacağı yükseklik (piksel).

        Formül ikili aramaya karşı doğrulandı (orbital penceresinin beyanı,
        altı ayrı genişlikte): sapma -1..+8 px. Sondaki bir satırlık pay o
        sapmayı kapatır, yoksa -1 px'lik durumlarda tek satır gizli kalırdı.
        """
        belge = self._text.document()
        belge.setTextWidth(self._text.viewport().width())
        satir = self._text.fontMetrics().lineSpacing()
        return int(
            belge.size().height() * satir
            + 2 * self._text.frameWidth()
            + 2 * belge.documentMargin()
            + satir
        )

    def _yuksekligi_icerige_uydur(self) -> None:
        """İçeriğin istediği yüksekliği dayat; üst sınırı aşma; aşarsan söyle."""
        gereken = self._gereken_yukseklik()
        hedef = min(gereken, self._max_height)
        if self._text.minimumHeight() != hedef:
            self._text.setMinimumHeight(hedef)
        self._basligi_guncelle(gereken)

    def _basligi_guncelle(self, gereken: int) -> None:
        """Kırpma varsa duyurur -- ama KAÇ SATIR olduğunu SÖYLEMEZ.

        İlk hâli gizli satır sayısını yazıyordu ve ölçüldü ki o sayı BAYAT
        kalabiliyor: 800x700'lük pencerede başlık "3 satır gizli" derken
        kaydırma çubuğu 9 satır gizliyordu. Sebep, Qt'nin yerleşim sırası --
        başlık, viewport genişliğinin henüz güncellenmediği bir ara olayda
        hesaplanmış oluyordu. Bir beyanda YANLIŞ SAYI, bu deponun tam olarak
        reddettiği şeydir; sayısı olmayan doğru bir uyarı, sayısı olan yanlış
        bir uyarıdan iyidir. Ne kadar kaldığını kaydırma çubuğunun kendisi
        zaten gösterir.
        """
        yeni = _BASLIK if gereken <= self._max_height else f"{_BASLIK}  —  devamı var, kaydırın"
        if self.title() != yeni:
            self.setTitle(yeni)

    def text(self) -> str:
        """Panelin ŞU AN bastığı metin.

        Testlerin okuma noktası. Bir pencere "panelin bastığı metin budur"
        diye bir metot sunuyorsa onu `format_panel`'i ikinci kez çağırarak
        değil BURADAN üretmelidir: ölçüldü, ikinci çağrı deseni
        `set_provenance`'ı hiç çağırmayan bir pencereyi de yeşil bırakıyor
        (panel 0 karakter, metot 1174 karakter, dokuz testin dokuzu geçiyor).
        """
        return self._text.toPlainText()
