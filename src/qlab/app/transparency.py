"""Şeffaflık paneli — spec §9.

Panel kendi başına hiçbir fizik bilgisi üretmez; yalnızca Provenance'ı basar.
Bu ayrım kasıtlı: gösterilen yaklaşım listesi motorun beyanıdır, arayüzün
tahmini değil.
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from qlab.provenance import Provenance, etiket_genisligi


def format_panel(provenance: Provenance) -> str:
    """Panelin metin gösterimi. Saf fonksiyon — GUI olmadan test edilir.

    Etiket sütunu `provenance.etiket_genisligi()`dir -- CLI ile AYNI kaynak (C5).
    Eskiden burada sabit `_LABEL_WIDTH = 14` vardı ve 16 karakterlik "Yakınsama
    tanısı" satırının ':'si sütundan 2 karakter kayıyordu.
    """
    genislik = etiket_genisligi()
    return "\n".join(
        f"{label:<{genislik}}: {value}" for label, value in provenance.as_lines()
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

        Yükseklik YERLEŞİMİN KENDİSİNDEN okunur: blokların
        `documentLayout().blockBoundingRect()` yüksekliklerinin toplamı + iki
        çerçeve + iki belge kenar payı + 1 px (Qt kaydırma çubuğunu
        `viewport - kenar payı - 1` görünür alanla hesaplar). Tavan alınır.

        ESKİDEN satır sayısı × `fontMetrics().lineSpacing()` + bir satırlık
        paydı ve gerçek Windows platformunda TUTMUYORDU (C4, grup 9). Ölçüldü
        (12.10.2026, `QT_QPA_PLATFORM=windows` + `WA_DontShowOnScreen`, ekranda
        hiçbir şey görünmez; birincil ekran %150): yazı tipi motoru Consolas
        9 pt için kesirli satır aralığı veriyor (`QFontMetricsF` 14.0312,
        tamsayı `lineSpacing` 14) ama satırları ~15.2 px aralıkla diziyor (su
        beyanında 17 satır = 259 px; formül 17 × 14 + kenarlar + 14 = 264 px,
        görünür alan 260 − 4 − 1 = 255 px) -- kırpılmayan beyanda bir satır
        kaydırmada kalıyordu: orbital penceresinde 45/45 durumda
        `scrollBar().maximum() == 1`. Ekransız platform tamsayı ölçü verdiği
        için orada 0/18 (görünmüyordu). Blok toplamıyla 0/45 ve 0/19. (Bu 12.10 /
        14.10 sayılarının sondaları KAYIP, yeniden üretilemez -- V11, 23.10.2026.
        Bugün üretilebilen, gerçek platform testinin kendi düzeneğinde: eski formül
        kırpılmayan 9 durumun 9'unda bir satır gizli, bu formül 0/9; satır adımı
        DPR 1.5 + QT_FONT_DPI 96'da 15.8 px/satır, `lineSpacing` 14.)
        PAY BİLİNÇLİ: Qt'ye tek kenar payı + 1 yeter; iki kenar payı + 1 ondan
        4 px cömert. Ölçüldü (mutasyon, gerçek platform testi): kenar payını
        teke indirmek ya da +1'i düşürmek TEK BAŞINA bir şey gizlemiyor
        (eşdeğer), ikisi birlikte bir satır gizliyor. Doğrudan ölçüm (23.10.2026,
        ikili aramayla en küçük kaydırmasız metin yüksekliği H*; %150 / %125 /
        %100 sabit ölçek, yerli %150 ve ekransız, ortam başına 36 durum): H* =
        blok toplamı + 2 çerçeve + 1 kenar payı + 1, 180/180; bu formül H* + 4.
        Ölçüm ve kilit: `tests/test_transparency.py`
        (`test_C4_GERCEK_WINDOWS_PLATFORMUNDA_...`, yalnız Windows, `slow`).
        """
        belge = self._text.document()
        belge.setTextWidth(self._text.viewport().width())
        duzen = belge.documentLayout()
        toplam = 0.0
        blok = belge.firstBlock()
        while blok.isValid():
            toplam += duzen.blockBoundingRect(blok).height()
            blok = blok.next()
        return math.ceil(
            toplam
            + 2 * self._text.frameWidth()
            + 2 * belge.documentMargin()
            + 1
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
