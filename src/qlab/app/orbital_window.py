"""Döndürülebilir orbital penceresi — kabul kriteri 3'ün arayüzü.

Pencere bir `HFResult` alır, seçilen moleküler orbitali ızgara üzerinde
değerlendirir (`observable.hf_obs.evaluate_orbital`), ±faz için iki izoyüzey
ağı çıkarır (`viz.isosurface.extract_signed_pair`) ve bunları
`pyqtgraph.opengl` sahnesinde çizer. Fare ile döndürme `GLViewWidget`'ten
gelir.

**Bu pencerenin beyan borcu.** Ekranda görünen "orbitalin şekli" üç ayrı
seçimin sonucudur ve üçü de hesabın kendisinden GELMEZ:

1. **İşaret keyfidir.** `psi` bir özvektörse `-psi` de aynı enerjiyle aynı
   çözümdür; global işaret hiçbir fiziksel büyüklüğü değiştirmez. Ölçüldü:
   su/STO-3G'nin 0. orbitalinde (O 1s) pozitif ağ BOŞ çıkıyor (0 tepe),
   negatif ağ 412 tepe veriyor. Buradan "orbital her yerde negatif" SONUCU
   ÇIKMAZ ve bir süre burada öyle yazıyordu -- yanlıştı: aynı orbitalin
   216000 ızgara noktasının 185036'sı (%85.66) pozitiftir. Doğru ifade,
   pozitif genliğin hiçbir yerde +0.05'i AŞMAMASIDIR (maks +1.8733e-3,
   min -4.9041); o yüzden +0.05 izoyüzeyi boş çıkar. Başka bir kütüphane
   aynı orbitali ters işaretle verse iki renk yer değiştirirdi ve hiçbir
   fizik değişmezdi. İki renk yalnızca AYNI orbitalin kendi içindeki bağıl
   faz farkını gösterir.
2. **İzoyüzey seviyesi bakanın seçimidir.** Ölçüldü (su HOMO, pad=4.0,
   n=60): level 0.02'de 2850 tepe, 0.05'te 1764, 0.50'de 114, 0.90'da hiç.
   Aynı orbital, farklı "şekil". Seviye bu yüzden panelde sayı olarak basılır.
   Seviye POZİTİF olmak zorundadır (bkz. `_dogrula_seviye`): sıfır ya da
   negatif bir seviye, panelin beyan ettiği ±faz-renk eşleşmesini yalanlar.
3. **Izgara çekirdek kabuğunu çözmez.** Ölçüldü: aynı O 1s orbitalinin ızgara
   üzerindeki maksimum genliği n=60'ta 4.9041, n=80'de 5.9271 -- yakınsamamış.
   Çekirdeğe yakın yüzeyler nitel gösterimdir, nicel değil.

Bunlar şeffaflık paneline motorun kendi `Provenance`'ının ÜSTÜNE eklenir;
motorunkiler (Born-Oppenheimer, tek Slater determinantı, sonlu baz seti,
kapalı kabuk) olduğu gibi taşınır. Pencere ayrıca `viz/isosurface.py`'nin
`EXTRACTION_APPROXIMATIONS` maddelerini de taşır (float32 dökümü,
marching-cubes interpolasyonu): `Mesh` kendi `Provenance`'ını taşımadığı için
o iki seçim ekrana ancak buradan ulaşır.

**Izgara motorun listesinde YOKTUR** -- ızgarayla ilgili her şey bu pencerenin
kendi tanısıdır ve burada, gösterim katmanında doğar; daha önce motorun beyanı
diye sayılıyordu, yanlıştı. Ama bir süre yalnızca `grid_points_per_axis`
basıldı ve o TEK BAŞINA ÇÖZÜNÜRLÜĞÜ BELİRLEMEZ: aralık = kutu genişliği /
(n-1) ve kutu genişliği molekülle ve `padding` ile değişir. Ölçüldü, n=60
sabitken yalnızca `padding` değişerek aralık 0.1841 / 0.4553 / 0.7265 bohr
oluyordu ve panel metni ÜÇÜNDE DE bit birebir aynıydı. Sadece `--molecule`
değişse bile (hepsi varsayılan) aralık H2 0.1593, HF 0.1650, H2O 0.1841,
C2H4 0.2145 bohr; panelde basılan sayı hep 60'tı. Bu yüzden aralık da beyan
edilir. Izgara ANİZOTROP olduğu için tek sayı yetmez -- aynı ızgarada eksen
aralıkları 0.1356 / 0.1841 / 0.1544 bohr -- en kaba ve en ince eksen ayrı
ayrı yazılır.

**Ekransız kip ve testlerin sınırı.** OpenGL bağlamı ancak BOYAMA sırasında
gerekir; kurulum (GLViewWidget, GLMeshItem, GLScatterPlotItem,
setCentralWidget, close) `QT_QPA_PLATFORM=offscreen` altında sorunsuz koşar --
ölçüldü. Test edilemeyen şey **boyanan piksellerdir**: gerçekten çizilen
görüntünün doğruluğu insanın işidir (README, kabul kriteri 3).

Bu sınır bir süre kendinden fazlasını yuttu ve ölçülerek geri alındı. Ekransız
kipte OKUNABİLİR olan saf durum -- panelin `maximumHeight`'ı, kameranın
`opts["distance"]`ı, seçicinin açılış konumu, görünümdeki item sayısı,
widget'ların gizli olup olmadığı, pencerenin `isVisible()`'ı -- boyamaya
bağlı DEĞİLDİR ve o sınırın İÇİNDE değildir. Bir süre öyle sanıldı: kamera
mesafesi çarpanını 60'tan 1'e düşüren, paneli varsayılan yüksekliğe indiren ve
`window.show()`u tamamen silen üç mutasyonun üçü de takımı yemyeşil bırakıyordu
(her biri 0 test kırıyordu), oysa üçü de tek satırlık bir okumayla
yakalanabiliyordu. "Görsel doğrulama insanın işidir" cümlesi çizilen görüntü
için doğru, pencerenin DURUMU için değil.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtWidgets

from qlab.app.transparency import TransparencyPanel
from qlab.engine.hf.scf import HFResult
from qlab.errors import InvalidInputError
from qlab.observable.hf_obs import ScalarField, VolumeGrid, evaluate_orbital
from qlab.provenance import Provenance
from qlab.types import Molecule
from qlab.viz.isosurface import (
    EXTRACTION_APPROXIMATIONS,
    Mesh,
    extract_signed_pair,
)

#: ±faz renkleri (RGBA). Kırmızı/mavi ayrımı yalnızca BAĞIL fazı gösterir;
#: hangi lobun hangi renk olduğu keyfidir (modül dokstring'i, madde 1).
_POZITIF_RENK = (0.90, 0.25, 0.25, 0.45)
_NEGATIF_RENK = (0.25, 0.45, 0.95, 0.45)
_CEKIRDEK_RENK = (0.95, 0.95, 0.95, 1.0)

#: Rengin TEK kaynağı `Mesh.sign`'dır, ağların demet sırası değil. Sıraya
#: bakan bir eşleme `viz/isosurface.py`'nin özenle taşıdığı işaret
#: sözleşmesini tüketicide düşürüyordu: ölçüldü, `zip(meshes, (POZ, NEG))`
#: deseni `level=-0.05` verildiğinde psi=-0.05 yüzeyini KIRMIZI çiziyor.
#: Beklenmeyen bir işaret sessizce bir renge düşmemeli, `KeyError` vermeli.
_ISARET_RENKLERI: dict[int, tuple[float, float, float, float]] = {
    1: _POZITIF_RENK,
    -1: _NEGATIF_RENK,
}

#: Kaydırıcı tam sayı taşır; seviye = konum / _KAYDIRICI_OLCEK.
_KAYDIRICI_OLCEK = 1000.0
#: Kaydırıcının taşıyabileceği en küçük konum. 0 bilinçli olarak dışarıda:
#: konum 0, seviye 0 demek olurdu ve seviye 0 düğüm yüzeyidir (aşağıya bkz.).
_KAYDIRICI_MIN = 1
#: Açılıştaki üst sınır. Sabit değil, gerekirse büyütülür (`_kaydirici_senkronla`).
_KAYDIRICI_VARSAYILAN_MAKS = 900
#: Kaydırıcının gösterebileceği en küçük seviye.
_EN_KUCUK_SEVIYE = _KAYDIRICI_MIN / _KAYDIRICI_OLCEK
#: "Binde-bir ızgarasına oturuyor mu" kontrolünün toleransı. Sıfır OLAMAZ:
#: kaydırıcının kendi ürettiği `konum / 1000.0` değerleri 1000 ile çarpılınca
#: 1..5000 arası 82 konumda tam sayı vermiyor (ölçülen en büyük artık
#: 4.547e-13). Tolerans o artığın ~2000 katı, gerçek bir hizasızlığın
#: (en küçüğü 0.5 tık = 500 birim) ise milyarda biri.
_KAYDIRICI_HIZALAMA_TOLERANSI = 1e-9


def _dogrula_seviye(level: float) -> float:
    """İzoyüzey seviyesini kabul eder ya da `InvalidInputError` ile reddeder.

    Üç ayrı şart, üçü de ÖLÇÜLMÜŞ bir bozulmaya karşılık gelir (su/STO-3G):

    - **Sonlu olmalı.** `nan` verildiğinde `extract_signed_pair` sessizce boş
      ağ döndürüyor, çizim (0, 0)'a düşüyor ve panel "şu an nan" basıyordu.
    - **Pozitif olmalı.** `viz/isosurface.py` yükü açıkça çağırana yıkıyor:
      *"level mutlak değer olarak yorumlanmaz ... Çağıran, arayüzde ±faz
      ayrımı istiyorsa pozitif bir level vermelidir."* Bu pencere O ÇAĞIRAN.
      `level=-0.05`'te `_meshes[0].sign` +1 kalıyor ama `_meshes[0].level`
      -0.05 oluyor: ağ psi=-0.05 yüzeyi, yani NEGATİF faz, ve `_POZITIF_RENK`
      ile çiziliyordu -- panelin beyan ettiği renk-faz eşleşmesi yalan olurdu.
      `level=0.0` ise (3600, 3600) tepe veriyor: çizilen şey orbitalin şekli
      değil, DÜĞÜM YÜZEYİ.
    - **Kaydırıcının çözünürlüğünde olmalı.** İKİ ayrı şart, ve bir süre
      yalnızca birincisi koda geçmişti:

      (a) `>= 0.001` olmalı -- kaydırıcının taşıyabileceği en küçük konum
          bu (`_KAYDIRICI_MIN / _KAYDIRICI_OLCEK`).
      (b) binde-bir ızgarasına OTURMALI. Kaydırıcı tam sayı taşır; aradaki
          bir seviye, denetimin panelden farklı bir sayı göstermesi demektir.
          Ölçüldü (bu şart yokken): `set_isosurface_level(0.0155)` `_level`'ı
          0.0155 yapıyor ama kaydırıcı 16'ya, yani 0.016'ya oturuyordu; aynı
          şekilde 0.0567 -> 0.057, 0.0014 -> 0.001, 0.1235 -> 0.124.
          0.0011'de kaydırıcı en küçük tıkta kilitli kalırken kullanıcı %10
          yukarıdaydı, 0.0015'te sapma %33'tü. Bu, `_kaydirici_senkronla`'nın
          ÜST tarafta düzeltilmiş kusurunun (2.0 -> kaydırıcı 0.9) ara/alt
          taraftaki aynısıdır. Sessiz desenkronizasyon burada da reddedilir,
          kırpılmaz.

      Kontrol TOLERANSLIDIR ve bu zorunlu: kaydırıcının kendi yaydığı
      `konum / 1000.0` değerlerinin bir kısmı 1000 ile çarpılınca tam sayı
      VERMİYOR -- ölçüldü, 1..5000 arası 82 konumda (1001 -> 1000.999999...),
      en büyük yuvarlama artığı 4.547e-13. Katı bir tam-sayı kontrolü o 82
      konumu reddederdi ve istisna Qt'nin sinyal işleyicisinde YUTULACAĞI için
      sonuç, düzeltilmek istenenden daha kötü bir sessiz desenkron olurdu
      (ölçüldü: kaydırıcı 1001'e sürüklenince konum 1001'de kalıyor ama
      `_level` 1.0 oluyordu).
    """
    deger = float(level)
    if not math.isfinite(deger):
        raise InvalidInputError(
            f"izoyüzey seviyesi sonlu bir sayı olmalı, verilen: {level!r}"
        )
    if deger <= 0.0:
        raise InvalidInputError(
            f"izoyüzey seviyesi pozitif olmalı (±faz ayrımı buna dayanır; "
            f"0 düğüm yüzeyidir, negatif seviye iki fazın rengini ters "
            f"çevirir), verilen: {deger:g}"
        )
    if deger < _EN_KUCUK_SEVIYE:
        raise InvalidInputError(
            f"izoyüzey seviyesi kaydırıcının çözünürlüğünün "
            f"({_EN_KUCUK_SEVIYE:g}) altında kalıyor; kaydırıcı onu "
            f"gösteremez ve panelden farklı bir sayı basardı, "
            f"verilen: {deger:g}"
        )
    konum = deger * _KAYDIRICI_OLCEK
    if abs(konum - round(konum)) > _KAYDIRICI_HIZALAMA_TOLERANSI:
        raise InvalidInputError(
            f"izoyüzey seviyesi kaydırıcının binde-bir ızgarasına oturmuyor; "
            f"kaydırıcı {round(konum) / _KAYDIRICI_OLCEK:g} gösterirken panel "
            f"{deger:g} basardı -- sessiz desenkron. En yakın geçerli seviye: "
            f"{round(konum) / _KAYDIRICI_OLCEK:g}"
        )
    return deger

_ISARET_BEYANI = (
    "orbital işareti keyfi: ±renkler yalnızca bağıl fazı gösterir, "
    "global işaret fiziksel değildir"
)
_GOSTERIM_IHMALLERI: tuple[str, ...] = (
    "çekirdek kabuklarının uç genliği: ızgara çözmüyor, "
    "çekirdeğe yakın yüzeyler nitel",
)


def _gosterim_yaklasimlari(level: float) -> tuple[str, ...]:
    """Pencerenin motorun beyanına EKLEDİĞİ gösterim seçimleri.

    Seviye burada, beyan metninin içinde okunur biçimde geçer (`0.05`) --
    `diagnostics`'e de yazılır ama orada `Provenance._format` her float'ı
    bilimsel gösterime çeviriyor (`5.000e-02`). Panel insanın okuduğu yer;
    `%g` biçimi 0.05'i de 0.2'yi de olduğu gibi basar.

    Son iki madde bu modülün değil `viz/isosurface.py`'nin seçimleri; metin
    orada, seçimin yapıldığı yerde yazılı. `Mesh` kendi `Provenance`'ını
    taşımadığı için buradan geçmeselerdi ekranda hiç görünmezlerdi.
    """
    return (
        _ISARET_BEYANI,
        f"izoyüzey seviyesi bir gösterim seçimidir, hesabın sonucu değildir "
        f"(şu an {level:g})",
        *EXTRACTION_APPROXIMATIONS,
    )


class OrbitalWindow(QtWidgets.QMainWindow):
    """Tek bir molekülün orbitallerini gösteren döndürülebilir 3B pencere.

    Açılışta HOMO gösterilir. `set_orbital_index` geçersiz bir indeks için
    `InvalidInputError` fırlatır -- hata `evaluate_orbital`'dan gelir ve
    pencere onu YUTMAZ; yutulsaydı kullanıcı yanlış orbitale baktığını
    bilmezdi. Reddedilen bir seçim pencerenin durumunu da değiştirmez.

    **Molekül AYRI bir argüman değildir.** `HFResult` molekülü zaten taşır
    (`evaluate_orbital` onu `build_basis`e verir), o yüzden pencere de
    yalnızca `result.molecule`'den okur. Daha önce ikinci bir `molecule`
    parametresi vardı ve DOĞRULANMIYORDU: ölçüldü,
    `OrbitalWindow(su_sonucu, h2_molekulu)` hatasız açılıyor, ızgarayı ve
    çekirdekleri H2'den, çizilen orbitali sudan alıyor, seçicide suyun yedi
    orbitalini listeleyip HOMO işaretini 0. satıra koyuyordu -- panel
    uyuşmazlıktan tek kelime etmeden. İki seçenek vardı: uyuşmazlığı
    reddetmek ya da parametreyi kaldırmak. Kaldırmak seçildi, çünkü reddetmek
    "uyuşma"nın doğru tanımını (aynı nesne? aynı geometri? aynı yük ve
    çokluk?) yazmayı gerektirir ve o tanımın kendisi de yanlış olabilir;
    parametre yoksa uyuşmazlık TEMSİL EDİLEMEZ.
    """

    def __init__(
        self,
        result: HFResult,
        *,
        padding: float = 4.0,
        points_per_axis: int = 60,
        level: float = 0.05,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        # Doğrulama super().__init__'ten önce: reddedilen bir seviye için
        # yarım kurulmuş bir QMainWindow ortada bırakılmaz.
        self._level = _dogrula_seviye(level)
        super().__init__(parent)
        self._result = result
        molecule: Molecule = result.molecule
        self._molecule = molecule
        self._grid = VolumeGrid.around(
            molecule, padding=padding, points_per_axis=points_per_axis
        )
        self._orbital_index = molecule.n_electrons // 2 - 1
        self._meshes: tuple[Mesh, Mesh] | None = None
        self._mesh_items: list[gl.GLMeshItem] = []

        self.setWindowTitle(f"qlab — orbital görüntüleyici ({result.provenance.method})")
        self._kur_arayuz()
        self._cekirdekleri_ciz()
        self._yenile()

    # -- genel arayüz --------------------------------------------------------

    @property
    def orbital_index(self) -> int:
        """Şu an gösterilen orbitalin sıfır tabanlı indeksi."""
        return self._orbital_index

    def set_orbital_index(self, index: int) -> None:
        """Gösterilen orbitali değiştirir ve ağları yeniden çıkarır.

        Geçersiz indeks `InvalidInputError` ile reddedilir ve pencere eski
        orbitalinde kalır: hata YUTULMAZ ama durum da BOZULMAZ.
        """
        eski = self._orbital_index
        self._orbital_index = int(index)
        try:
            self._yenile()
        except InvalidInputError:
            self._orbital_index = eski
            raise
        self._secici_senkronla()

    def set_isosurface_level(self, level: float) -> None:
        """İzoyüzey seviyesini değiştirir ve ağları yeniden çıkarır.

        Geçersiz seviye (`_dogrula_seviye`) pencerenin HİÇBİR alanına
        dokunmadan reddedilir. Kardeşi `set_orbital_index` bunu geri alarak
        sağlıyor; burada geri alınacak bir şey yok çünkü doğrulama
        `_yenile`'den ÖNCE. Fark önemliydi: eski kodda `set_isosurface_level(
        nan)` `ValueError` fırlatıyordu ama fırlatmadan önce `_yenile` tamamen
        koşuyordu -- `_level=nan` kalıyor, ağlar (1764, 1764)'ten (0, 0)'a
        düşüyor, görünümdeki item 3'ten 1'e iniyor, panel "şu an nan" basıyor,
        kaydırıcı 0.05'te bayat kalıyordu. Yani "hata fırlattı" ile "durum
        korundu" aynı şey değildir.
        """
        self._level = _dogrula_seviye(level)
        self._yenile()
        self._kaydirici_senkronla()

    def mesh_vertex_counts(self) -> tuple[int, int]:
        """Şu anki (pozitif, negatif) ağların tepe sayıları.

        Çizimin kendisi ekransız kipte doğrulanamadığı için testler veri
        hattını buradan izler.

        Ağ yokken DÖNMEK yerine YÜKSELTİR. Dal bugün erişilemez (`_kur_arayuz`
        kaydırıcı sinyallerini bloklar, `setCurrentIndex` connect'ten önce,
        `_yenile` `__init__`'in sonuncusu) ve bir süre `# pragma: no cover` ile
        `(0, 0)` döndürüyordu. Ama `(0, 0)` bu pencerede GEÇERLİ bir sonuçtur
        -- ölçüldü, su HOMO'su level=0.90'da tam olarak (0, 0) veriyor -- yani
        yedek değer geçerli çıktıdan ayırt edilemiyordu. `_yenile`'yi
        `__init__`'ten sonraya taşıyan bir refactor, hata yerine tamamen makul
        görünen bir sayı döndürürdü. Yükseltmek aynı erişilemezlik iddiasını
        korur, sessiz yalanı kaldırır ve dalı test edilebilir yapar.
        """
        if self._meshes is None:
            raise AssertionError(
                "mesh_vertex_counts ağlar kurulmadan çağrıldı; `_yenile` "
                "__init__'in son adımıdır ve bu durum erişilemez olmalı. "
                "Sessizce (0, 0) dönmek yanlış olurdu: (0, 0) geçerli bir "
                "sonuçtur (su HOMO'su level=0.90'da tam olarak onu verir)."
            )
        pozitif, negatif = self._meshes
        return (len(pozitif.vertices), len(negatif.vertices))

    def transparency_text(self) -> str:
        """Şeffaflık panelinin bastığı metin -- PANELDEN okunur.

        Eskiden bu metot `format_panel(self._provenance())` ile metni
        SIFIRDAN üretiyordu, yani "panelin bastığı metnin aynısı" sözü
        yalnızca bir umuttu. Ölçüldü, söz korunmuyordu: `_yenile`'den
        `panel.set_provenance` çağrısını silen mutasyonda panel 0 karakter,
        metot hâlâ 1174 karakter basıyor ve dokuz testin dokuzu geçiyordu;
        panele motorun HAM provenance'ını veren mutasyonda da panel bu
        pencerenin üç gösterim beyanının hiçbirini taşımıyor, yine 9/9.
        Metin artık widget'tan okunduğu için o mutasyonların ikisi de
        testlerin gözünden kaçamaz -- söz test edilmiyor, YAPISAL.
        """
        return self._panel.text()

    # -- iç yapı -------------------------------------------------------------

    def _kur_arayuz(self) -> None:
        self._view = gl.GLViewWidget()
        self._view.setCameraPosition(distance=float(np.max(self._grid.spacing)) * 60.0)

        self._secici = QtWidgets.QComboBox()
        for etiket in self._orbital_etiketleri():
            self._secici.addItem(etiket)
        self._secici.setCurrentIndex(self._orbital_index)
        self._secici.currentIndexChanged.connect(self._secici_degisti)

        self._kaydirici = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._kaydirici.setMinimum(_KAYDIRICI_MIN)
        self._kaydirici.setMaximum(_KAYDIRICI_VARSAYILAN_MAKS)
        # Değer, kırpılmaya karşı korumalı tek yoldan konur; açılış seviyesi
        # varsayılan üst sınırın üstünde olabilir (O 1s'in maks genliği 4.90).
        self._kaydirici_senkronla()
        self._kaydirici.valueChanged.connect(self._kaydirici_degisti)

        # Üst sınır, pencerenin VARSAYILAN genişliğinde ölçüldü. Bu beyan
        # 960 px'te 332 px istiyor (dar pencerede daha çok sarıldığı için daha
        # çok: 800 px'te 388, 700 px'te 458). 340, varsayılan geometride hiçbir
        # satırı gizlemez ve 3B görünüme ~370 px bırakır. 260 ile varsayılan
        # geometride 5 satır gizli kalıyordu -- ölçüldü.
        #
        # Daha dar pencerelerde kırpma kaçınılmaz; panel o zaman başlığında
        # "devamı var, kaydırın" der. Sınırı büsbütün kaldırmak 3B görünümü
        # ezerdi: 700x650'de beyan 458 px isterken pencerenin tamamı 650.
        self._panel = TransparencyPanel(max_height=340)

        ust = QtWidgets.QHBoxLayout()
        ust.addWidget(QtWidgets.QLabel("Orbital:"))
        ust.addWidget(self._secici, 1)
        ust.addWidget(QtWidgets.QLabel("İzoyüzey seviyesi:"))
        ust.addWidget(self._kaydirici, 1)

        govde = QtWidgets.QVBoxLayout()
        govde.addLayout(ust)
        govde.addWidget(self._view, 1)
        govde.addWidget(self._panel)

        merkez = QtWidgets.QWidget()
        merkez.setLayout(govde)
        self.setCentralWidget(merkez)
        self.resize(960, 800)

    def _orbital_etiketleri(self) -> list[str]:
        """Seçicideki satırlar; HOMO ve LUMO adıyla işaretlenir."""
        homo = self._molecule.n_electrons // 2 - 1
        etiketler = []
        for i, enerji in enumerate(self._result.orbital_energies):
            if i == homo:
                ad = "HOMO"
            elif i == homo + 1:
                ad = "LUMO"
            elif i < homo:
                ad = "dolu"
            else:
                ad = "boş"
            etiketler.append(f"{i} ({ad}) — {enerji:.6f} Ha")
        return etiketler

    def _secici_degisti(self, index: int) -> None:
        if index != self._orbital_index:
            self.set_orbital_index(index)

    def _kaydirici_degisti(self, value: int) -> None:
        self.set_isosurface_level(value / _KAYDIRICI_OLCEK)

    def _secici_senkronla(self) -> None:
        if self._secici.currentIndex() != self._orbital_index:
            self._secici.blockSignals(True)
            self._secici.setCurrentIndex(self._orbital_index)
            self._secici.blockSignals(False)

    def _kaydirici_senkronla(self) -> None:
        """Kaydırıcıyı `_level`'a eşitler; gerekirse ARALIĞI BÜYÜTÜR.

        Değişmez: `kaydirici.value() == round(_level * _KAYDIRICI_OLCEK)`.
        Eskiden aralık sabitti ve değer kırpılıyordu: ölçüldü,
        `set_isosurface_level(2.0)` `_level`'ı 2.0 yapıyor ama kaydırıcı
        900'de (0.9) kalıyordu -- panel "şu an 2" derken denetim 0.9
        gösteriyordu, hiçbir uyarı olmadan. 2.0 uydurma bir seviye de değil:
        su/STO-3G O 1s'in ızgara üzerindeki maksimum genliği 4.9041.
        Aşağı yön için kırpma yerine reddetme var (`_dogrula_seviye`), çünkü
        alt sınırı büyütmek konumu 0'a, yani seviyeyi düğüm yüzeyine iterdi.
        """
        hedef = int(round(self._level * _KAYDIRICI_OLCEK))
        self._kaydirici.blockSignals(True)
        try:
            if hedef > self._kaydirici.maximum():
                self._kaydirici.setMaximum(hedef)
            self._kaydirici.setValue(hedef)
        finally:
            self._kaydirici.blockSignals(False)

    def _cekirdekleri_ciz(self) -> None:
        konumlar = np.array([atom.position for atom in self._molecule.atoms], dtype=float)
        boyutlar = np.array(
            [
                max(6.0, 4.0 * float(atom.atomic_number) ** (1.0 / 3.0))
                for atom in self._molecule.atoms
            ]
        )
        self._view.addItem(
            gl.GLScatterPlotItem(
                pos=konumlar, size=boyutlar, color=_CEKIRDEK_RENK, pxMode=True
            )
        )

    def _yenile(self) -> None:
        """Seçili orbitali değerlendirip iki ağı yeniden çıkarır ve çizer."""
        degerler = evaluate_orbital(self._result, self._orbital_index, self._grid)
        alan = ScalarField(
            values=degerler, grid=self._grid, provenance=self._result.provenance
        )
        yeni = extract_signed_pair(alan, self._level)
        self._meshes = yeni
        self._ag_itemlerini_ciz(yeni)
        self._panel.set_provenance(self._provenance())

    def _ag_itemlerini_ciz(self, meshes: tuple[Mesh, Mesh]) -> None:
        """Ağları çizer; renk `Mesh.sign`'dan gelir, demet sırasından DEĞİL.

        Eski kod `zip(meshes, (_POZITIF_RENK, _NEGATIF_RENK))` ile rengi
        sıraya göre veriyor ve `Mesh.sign`'ı hiç okumuyordu; `isosurface.py`
        tarafında ayrı ayrı taşınan genlik (`level`) ve faz (`sign`) bilgisi
        tam burada, tüketicide yere düşüyordu.

        Eski item'lar önce görünümden SİLİNİR. Silinmeseydi sızıntı sessiz
        değil görsel olurdu: ölçüldü, silme döngüsü olmadan 13 orbital
        değişiminde görünümdeki item sayısı 3'ten 27'ye çıkıyor ve ekranda
        yedi orbitalin yüzeyleri üst üste duruyor.
        """
        for eski in self._mesh_items:
            self._view.removeItem(eski)
        self._mesh_items = []
        for mesh in meshes:
            if len(mesh.vertices) == 0:
                continue
            renk = _ISARET_RENKLERI[mesh.sign]
            veri = gl.MeshData(
                vertexes=np.asarray(mesh.vertices, dtype=float),
                faces=np.asarray(mesh.faces),
            )
            item = gl.GLMeshItem(
                meshdata=veri,
                smooth=True,
                color=renk,
                shader="balloon",
                glOptions="translucent",
            )
            self._view.addItem(item)
            self._mesh_items.append(item)

    def _provenance(self) -> Provenance:
        """Motorun beyanı + bu pencerenin gösterim seçimleri.

        Motorunki `dataclasses.replace` ile korunur; pencere hiçbir şeyi
        SİLMEZ, yalnızca EKLER. Panel kendi başına fizik bilgisi üretmez
        (bkz. transparency.py dokstring'i), o yüzden eklenenler burada,
        veri katmanında oluşur.
        """
        temel = self._result.provenance
        tanilar = dict(temel.diagnostics)
        tanilar["orbital_index"] = self._orbital_index
        tanilar["isosurface_level"] = self._level
        tanilar["grid_points_per_axis"] = self._grid.shape[0]
        # Eksen başına nokta sayısı TEK BAŞINA çözünürlüğü vermez (aralık =
        # kutu genişliği / (n-1)) ve kutu genişliği molekülle değişir. Aralık
        # bu yüzden ayrıca basılır; ızgara anizotrop olduğu için en kaba ve en
        # ince eksen ayrı ayrı. `grid_spacing_bohr` adı ve "en kaba eksen"
        # anlamı `observable/hf_obs.py` ile aynıdır.
        tanilar["grid_spacing_bohr"] = float(np.max(self._grid.spacing))
        tanilar["grid_spacing_min_bohr"] = float(np.min(self._grid.spacing))
        return dataclasses.replace(
            temel,
            approximations=temel.approximations + _gosterim_yaklasimlari(self._level),
            neglected=temel.neglected + _GOSTERIM_IHMALLERI,
            diagnostics=tanilar,
        )
