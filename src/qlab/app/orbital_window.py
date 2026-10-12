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
from qlab.engine.hf.analysis import (
    _CHARGED_DIPOLE_NOTE,
    _NO_HOMO_NOTE,
    _NO_LUMO_NOTE,
    AnalysisReport,
    analyse,
)
from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import overlap_matrix
from qlab.engine.hf.scf import HFResult
from qlab.errors import InvalidInputError
from qlab.observable.hf_obs import (
    ScalarField,
    VolumeGrid,
    density_difference,
    electron_density,
    evaluate_orbital,
    free_atom_density_matrix,
)
from qlab.provenance import Provenance
from qlab.types import Atom, Molecule
from qlab.viz.isosurface import (
    EXTRACTION_APPROXIMATIONS,
    Mesh,
    extract_isosurface,
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

#: Kaydırıcı tam sayı taşır; seviye = konum / ölçek. Bu sayı ORBİTAL alanının
#: ölçeğidir -- üç alan türünün üçü de kendi ölçeğini taşır (bkz. `_AlanTuru`).
_KAYDIRICI_OLCEK = 1000.0
#: Orbital alanında kaydırıcının taşıyabileceği en küçük konum. 0 bilinçli
#: olarak dışarıda: konum 0, seviye 0 demek olurdu ve seviye 0 düğüm
#: yüzeyidir (aşağıya bkz.).
_KAYDIRICI_MIN = 1
#: Açılıştaki üst sınır. Sabit değil, gerekirse büyütülür (`_kaydirici_senkronla`).
_KAYDIRICI_VARSAYILAN_MAKS = 900
#: Kaydırıcının gösterebileceği en küçük seviye (orbital alanı).
_EN_KUCUK_SEVIYE = _KAYDIRICI_MIN / _KAYDIRICI_OLCEK
#: "Kaydırıcının ızgarasına oturuyor mu" kontrolünün toleransı. Sıfır OLAMAZ:
#: kaydırıcının kendi ürettiği `konum / 1000.0` değerleri 1000 ile çarpılınca
#: 1..5000 arası 82 konumda tam sayı vermiyor (ölçülen en büyük artık
#: 4.547e-13). Tolerans o artığın ~2000 katı, gerçek bir hizasızlığın
#: (en küçüğü 0.5 tık = 500 birim) ise milyarda biri.
_KAYDIRICI_HIZALAMA_TOLERANSI = 1e-9

#: Seçicideki üç alan türünün adı. Sabit olarak dışarı verilirler çünkü
#: `set_field_kind` bunları kabul eder ve seçicideki etiketle AYNI dizedirler:
#: iki ayrı liste tutulsaydı biri kayınca seçici kullanıcıya bir şey gösterip
#: pencereye başka bir şey söylerdi.
ALAN_ORBITAL = "orbital"
ALAN_YOGUNLUK = "elektron yoğunluğu"
ALAN_FARK = "yoğunluk farkı"


@dataclasses.dataclass(frozen=True)
class _AlanTuru:
    """Bir alan türünün ÜRETİCİSİ, ÇIKARIMI, SEVİYE ARALIĞI ve BEYANI.

    Dördü de alan başına ayrıdır ve hiçbiri ötekinden türetilemez; tek bir
    ortak sabite çökertmek üç alanın ikisinde ölçülebilir biçimde yanlış olur
    (aşağıdaki sayılar su/STO-3G, pad=4.0, n=60 -- pencerenin kendi
    varsayılanları).
    """

    ad: str
    #: Alanı üreten `observable` çağrısı. Orbital, yoğunluk matrisinden değil
    #: TEK bir MO'dan gelir; ötekiler yoğunluk matrisinden.
    varsayilan_seviye: float
    kaydirici_olcek: float
    kaydirici_min: int
    kaydirici_maks: int
    #: `True` ise `extract_signed_pair` (±faz/işaret iki ağ), `False` ise
    #: `extract_isosurface` (tek ağ).
    isaretli: bool
    #: Rengin ne anlama GELDİĞİ. Üç alanda üç ayrı şey demek: orbitalde keyfi,
    #: yoğunlukta yok, farkta fiziksel.
    beyan: str

    @property
    def en_kucuk_seviye(self) -> float:
        """Kaydırıcının bu alanda gösterebileceği en küçük seviye."""
        return self.kaydirici_min / self.kaydirici_olcek

    @property
    def adim(self) -> float:
        """Kaydırıcının bu alandaki bir tıkı, alanın kendi biriminde."""
        return 1.0 / self.kaydirici_olcek


#: ORBİTAL. maks|psi| = 0.697796 ölçüldü; level 0.02'de 2850, 0.05'te 1764,
#: 0.50'de 114, 0.90'da 0 tepe. Bant ve ölçek DEĞİŞMEDİ -- bu pencerenin
#: bugüne kadarki tek alanı buydu ve sayıları zaten ölçülmüştü.
_ORBITAL = _AlanTuru(
    ad=ALAN_ORBITAL,
    varsayilan_seviye=0.05,
    kaydirici_olcek=_KAYDIRICI_OLCEK,
    kaydirici_min=_KAYDIRICI_MIN,
    kaydirici_maks=_KAYDIRICI_VARSAYILAN_MAKS,
    isaretli=True,
    beyan=(
        "orbital işareti keyfi: ±renkler yalnızca bağıl fazı gösterir, "
        "global işaret fiziksel değildir"
    ),
)

#: ELEKTRON YOĞUNLUĞU. Ölçülen aralık [0.000000, 49.960388] -- yani eski 0.900
#: üst sınırı tepenin 55'te biriydi. İşaretsiz çıkarım (`extract_isosurface`)
#: bu alanın var olma sebebidir: yoğunluk KESİN POZİTİF olduğu için `-level`
#: yüzeyi hiçbir zaman yoktur; ölçüldü, `extract_signed_pair` level=0.1'de
#: (+2196, -0) veriyor -- ikinci ağ HER ZAMAN boş.
#: Bant: 0.01 (4892 tepe) .. 2.0 (108 tepe). Üst sınır 2.0, çünkü onun üstünde
#: görülen tek şey ızgaranın ÇÖZMEDİĞİ çekirdek tepesidir (modül dokstring'i,
#: madde 3). Varsayılan 0.1 -> 2196 tepe. Binde-bir çözünürlük burada yetiyor:
#: ölçüldü, 0.095 (2260) / 0.096 (2248) / 0.097 (2228) zaten ayrı yüzeyler.
_YOGUNLUK = _AlanTuru(
    ad=ALAN_YOGUNLUK,
    varsayilan_seviye=0.1,
    kaydirici_olcek=1000.0,
    kaydirici_min=10,
    kaydirici_maks=2000,
    isaretli=False,
    beyan=(
        "elektron yoğunluğu kesin pozitiftir: tek işaretsiz izoyüzey çizilir, "
        "bu alanda ±faz-renk ayrımı YOKTUR"
    ),
)

#: YOĞUNLUK FARKI (molekül - promolekül). Ölçülen aralık [-0.283034,
#: +0.318152]; harita 0.32'de tamamen boşalıyor. İşaretli çift ŞART: birikim
#: (+) ve boşalma (-) iki ayrı renk olmalı, ölçüldü level=0.01'de
#: (+1764, -2020) -- ikisi de dolu.
#: ÖLÇEK on-binde-bir, ve bu bir süs değil: 0.0001 adımlar AYRI yüzeyler
#: veriyor -- 0.0020 (+3208,-4704) / 0.0021 (+3196,-4596) / 0.0022
#: (+3156,-4544) / 0.0023 (+3108,-4476). Binde-bir ölçekte bu bandın tamamı
#: 8 tıka sıkışırdı.
#: Bant: 0.0005 (+4184,-7368) .. 0.3 (+20,-0). Alt uç kutu duvarından güvenli
#: uzaklıkta: ölçüldü, farkın duvardaki maksimum mutlak değeri 1.823e-05,
#: yani 0.0005 onun 27 katı ve yüzey KAPALI.
_FARK = _AlanTuru(
    ad=ALAN_FARK,
    varsayilan_seviye=0.01,
    kaydirici_olcek=10000.0,
    kaydirici_min=5,
    kaydirici_maks=3000,
    isaretli=True,
    beyan=(
        "yoğunluk farkında işaret FİZİKSELDİR (orbitalin aksine keyfi değil): "
        "kırmızı birikim (+), mavi boşalma (−)"
    ),
)

#: Seçicideki SIRA da buradan gelir; sözlük eklenme sırasını korur.
_ALAN_TURLERI: dict[str, _AlanTuru] = {
    tur.ad: tur for tur in (_ORBITAL, _YOGUNLUK, _FARK)
}


def _dogrula_alan_turu(ad: str) -> _AlanTuru:
    """Alan türü adını çözer ya da `InvalidInputError` ile reddeder.

    Sessizce bir varsayılana düşmek yanlış olurdu: kullanıcı yoğunluk farkı
    istediğini sanıp orbitale bakardı ve panelde bunu söyleyen bir satır
    olmazdı. Hata mesajı geçerli adları SAYAR -- bu depoda `--basis` için
    zaten verilmiş bir karar ("kullanıcı ne yazabileceğini görmeli").
    """
    if ad not in _ALAN_TURLERI:
        raise InvalidInputError(
            f"bilinmeyen alan türü: {ad!r}. Desteklenenler: "
            f"{tuple(_ALAN_TURLERI)}"
        )
    return _ALAN_TURLERI[ad]


def _dogrula_seviye(level: float, tur: _AlanTuru = _ORBITAL) -> float:
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
    if deger < tur.en_kucuk_seviye:
        raise InvalidInputError(
            f"izoyüzey seviyesi kaydırıcının çözünürlüğünün "
            f"({tur.en_kucuk_seviye:g}, {tur.ad} alanı) altında kalıyor; "
            f"kaydırıcı onu gösteremez ve panelden farklı bir sayı basardı, "
            f"verilen: {deger:g}"
        )
    konum = deger * tur.kaydirici_olcek
    if abs(konum - round(konum)) > _KAYDIRICI_HIZALAMA_TOLERANSI:
        en_yakin = round(konum) / tur.kaydirici_olcek
        raise InvalidInputError(
            f"izoyüzey seviyesi kaydırıcının ızgarasına oturmuyor "
            f"(adım {tur.adim:g}, {tur.ad} alanı); kaydırıcı {en_yakin:g} "
            f"gösterirken panel {deger:g} basardı -- sessiz desenkron. "
            f"En yakın geçerli seviye: {en_yakin:g}"
        )
    return deger

_GOSTERIM_IHMALLERI: tuple[str, ...] = (
    "çekirdek kabuklarının uç genliği: ızgara çözmüyor, "
    "çekirdeğe yakın yüzeyler nitel",
)


def _gosterim_yaklasimlari(level: float, tur: _AlanTuru) -> tuple[str, ...]:
    """Pencerenin motorun beyanına EKLEDİĞİ gösterim seçimleri.

    Seviye burada, beyan metninin içinde okunur biçimde geçer (`0.05`) --
    `diagnostics`'e de yazılır ama orada `Provenance._format` her float'ı
    bilimsel gösterime çeviriyor (`5.000e-02`). Panel insanın okuduğu yer;
    `%g` biçimi 0.05'i de 0.2'yi de olduğu gibi basar.

    İlk madde ALAN TÜRÜNE göre değişir ve bu bir ayrıntı değil: aynı iki renk
    üç alanda üç ayrı şey demektir. Orbitalde işaret KEYFİDİR (aynı özvektörün
    -1 katı aynı çözümdür), yoğunlukta işaret diye bir şey YOKTUR (alan kesin
    pozitif), fark haritasında ise işaret FİZİKSELDİR (+ birikim, - boşalma).
    Tek bir sabit metni üçüne birden basmak, ikisinde yalan olurdu.

    Son iki madde bu modülün değil `viz/isosurface.py`'nin seçimleri; metin
    orada, seçimin yapıldığı yerde yazılı. `Mesh` kendi `Provenance`'ını
    taşımadığı için buradan geçmeselerdi ekranda hiç görünmezlerdi.
    """
    return (
        tur.beyan,
        f"izoyüzey seviyesi bir gösterim seçimidir, hesabın sonucu değildir "
        f"(şu an {level:g})",
        *EXTRACTION_APPROXIMATIONS,
    )


# ---------------------------------------------------------------------------
# Fark haritasının REFERANSI
#
# `rho_molekul - rho_promolekul` haritasında "bağ" diye gösterilen şey,
# promolekülün ne olduğu bilinmeden OKUNAMAZ. Motorun provenance'ı bunu bugüne
# kadar yalnızca NESİR olarak söylüyordu ("küresel ortalamalı, kesirli işgalli
# serbest atom"); `free_atom_density_matrix` aynı şeyi KONTROL EDİLEBİLİR bir
# sayıyla söylemeyi mümkün kılıyor. Tr(PS) o referansın taşıdığı elektron
# sayısıdır ve okuyan biri onu atom numarasıyla karşılaştırabilir.

#: Önbellek: (element, baz seti) -> (baz fonksiyonu sayısı, Tr(PS)).
#:
#: ANAHTAR İKİLİDİR ve tek başına element YETMEZ: aynı elementin referansı
#: baz setine göre farklı bir MATRİSTİR -- ölçüldü, H sto-3g'de 1x1, 6-31G'de
#: 2x2; O sto-3g'de 5x5, 6-31G'de 9x9. Tr(PS) bu kusuru görmez (iki bazda da
#: H 1.0000000000, O 8.0000000000, çünkü nötr atomun elektron sayısı baza
#: bağlı değildir), o yüzden beyan edilen sayı boyutu da taşır.
#:
#: Önbellek PENCEREYE AİTTİR (`OrbitalWindow._serbest_atom_onbellegi`), modül
#: düzeyinde değil. Bir süre global bir sözlüktü: ölçüldü, pencere kapandıktan
#: sonra 2 girdi kalıyor ve sonraki her pencere onları okuyordu -- bozuk bir
#: girdi bütün sonraki pencerelerin fark beyanına sızardı. Bedeli pencere
#: başına bir kez ödenen hesap (aşağıdaki ölçümler), fark alanının kendi
#: 170 ms'sinin yanında küçük.
_SerbestAtomOnbellegi = dict[tuple[str, str], tuple[int, float]]


def _serbest_atom_referansi(
    symbol: str, basis_name: str, onbellek: _SerbestAtomOnbellegi
) -> tuple[int, float]:
    """Serbest atom referansının (baz fonksiyonu sayısı, Tr(PS)) çifti.

    Ölçülen maliyet ihmal edilebilir ama sıfır değil: H 0.5 ms, O 19.3 ms
    (STO-3G); 6-31G'de O 139.2 ms. Aynı element bir molekülde birden çok kez
    geçtiği ve pencere alan değiştikçe beyanı yeniden kurduğu için sonuç
    çağıranın verdiği `onbellek`e alınır.
    """
    anahtar = (symbol, basis_name)
    kayit = onbellek.get(anahtar)
    if kayit is None:
        yogunluk = free_atom_density_matrix(symbol, basis_name)
        baz = build_basis(Molecule((Atom(symbol, np.zeros(3)),)), basis_name)
        kayit = (len(baz), float(np.trace(yogunluk @ overlap_matrix(baz))))
        onbellek[anahtar] = kayit
    return kayit


def _promolekul_referans_beyani(
    molecule: Molecule, basis_name: str, onbellek: _SerbestAtomOnbellegi
) -> str:
    """Moleküldeki her AYRIK element için serbest atom referansını beyan eder.

    Ölçüldü (su/STO-3G): O 5 baz fonksiyonu, Tr(PS)=8.000000 (Z=8);
    H 1 baz fonksiyonu, Tr(PS)=1.000000 (Z=1). Sayı ile Z'nin yan yana
    durması bilinçli -- okuyan biri referansın doğru elektron sayısını taşıyıp
    taşımadığını KENDİ kontrol edebilir.
    """
    gorulen: dict[str, int] = {}
    for atom in molecule.atoms:
        gorulen.setdefault(atom.symbol, atom.atomic_number)
    parcalar = []
    for sembol, atom_numarasi in gorulen.items():
        n_baz, iz = _serbest_atom_referansi(sembol, basis_name, onbellek)
        parcalar.append(
            f"{sembol} {n_baz} baz fonksiyonu, Tr(PS)={iz:.6f} (Z={atom_numarasi})"
        )
    return (
        f"fark haritasının referansı ÖLÇÜLDÜ ({basis_name.upper()} serbest "
        "atomları, küresel ortalamalı ve kesirli işgalli): " + " · ".join(parcalar)
    )


#: `analyse()`in KOŞULLU gerekçe notları, raporun kendi sırasıyla (yüklü dipol,
#: LUMO yok, HOMO yok). Metin motorda TEK yerde yaşar ve buraya kopyalanmaz;
#: pencere hangisinin basılacağına da KENDİSİ karar vermez (bkz.
#: `_okuma_satirini_guncelle`).
_KOSULLU_ANALIZ_NOTLARI: tuple[str, ...] = (
    _CHARGED_DIPOLE_NOTE,
    _NO_LUMO_NOTE,
    _NO_HOMO_NOTE,
)


class OrbitalWindow(QtWidgets.QMainWindow):
    """Tek bir molekülün orbitallerini gösteren döndürülebilir 3B pencere.

    Açılışta HOMO gösterilir. `set_orbital_index` geçersiz bir indeks için
    `InvalidInputError` fırlatır -- alan türünden bağımsız olarak -- ve
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
        level: float | None = None,
        field: str = ALAN_ORBITAL,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        # REDDEDEBİLEN HER ŞEY super().__init__'ten önce: reddedilen bir girdi
        # için yarım kurulmuş bir QMainWindow ortada bırakılmaz. Bir süre
        # yalnız seviye ve alan türü buradaydı; ölçüldü, `points_per_axis=1`,
        # `padding=0.0`/`-100` ve sıfır elektronlu bir sistemin orbital alanı
        # super()'den SONRA reddediliyor ve verilen `parent`ın altında yarım
        # kurulmuş 1 QMainWindow kalıyordu. Izgara (`VolumeGrid.around`),
        # `analyse()` ve ilk alan Qt'ye dokunmaz; üçü de buraya taşındı.
        self._alan_turu = _dogrula_alan_turu(field)
        # `level=None`, "bu alanın kendi varsayılanı" demektir. Sabit bir 0.05
        # üçünün ikisinde yanlış olurdu: yoğunlukta 0.05 tepenin 1000'de biri,
        # farkta ise bandın (0.0005-0.3) beş katı üstü.
        self._level = _dogrula_seviye(
            self._alan_turu.varsayilan_seviye if level is None else level,
            self._alan_turu,
        )
        self._result = result
        molecule: Molecule = result.molecule
        self._molecule = molecule
        self._grid = VolumeGrid.around(
            molecule, padding=padding, points_per_axis=points_per_axis
        )
        self._orbital_index = molecule.n_electrons // 2 - 1
        self._meshes: tuple[Mesh, ...] | None = None
        self._mesh_items: list[gl.GLMeshItem] = []
        self._serbest_atom_onbellegi: _SerbestAtomOnbellegi = {}

        # `analyse()` BİR KEZ koşar: raporu ne seviye ne alan türü değiştirir,
        # ikisi de gösterim seçimidir. `UnsupportedSystemError` fırlatamaz --
        # tek fırlatma koşulu n_occupied > n_orbitals ve ölçüldü, `solve()` o
        # sistemi kendi girişinde zaten reddediyor ("baz seti bu kadar
        # elektronu tasiyamiyor"), yani buraya ulaşan bir HFResult yok.
        self._analiz: AnalysisReport = analyse(result)
        # Alan arayüz kurulmadan ÖNCE üretilir, böylece `_alan` hiçbir zaman
        # None olmaz ve `_provenance`/`_yenile` içinde "henüz alan yok" diye
        # bir dal taşımak gerekmez. Ulaşılamaz bir dalın yedek değeri, bu
        # depoda ölçülmüş bir kusur ailesidir: `mesh_vertex_counts` yedek
        # olarak (0, 0) döndürürken o değer GEÇERLİ bir çıktıdan ayırt
        # edilemiyordu. `_alan_anahtari` önbelleğin hangi durum için
        # üretildiğini taşır (bkz. `_guncel_alan`).
        self._alan_anahtari: tuple[str, int | None] | None = None
        self._alan: ScalarField
        self._alan_anahtari, self._alan = self._alan_getir(
            self._alan_turu, self._orbital_index
        )
        super().__init__(parent)

        self.setWindowTitle(f"qlab — orbital görüntüleyici ({result.provenance.method})")
        self._kur_arayuz()
        self._cekirdekleri_ciz()
        self._yenile()

    # -- genel arayüz --------------------------------------------------------

    @property
    def orbital_index(self) -> int:
        """Şu an gösterilen orbitalin sıfır tabanlı indeksi.

        Yalnızca `field_kind == "orbital"` iken bir karşılığı vardır; öteki iki
        alan yoğunluk MATRİSİNDEN üretilir ve tek bir orbitale bağlı değildir.
        """
        return self._orbital_index

    @property
    def field_kind(self) -> str:
        """Şu an gösterilen alan türü (`ALAN_ORBITAL`/`ALAN_YOGUNLUK`/`ALAN_FARK`)."""
        return self._alan_turu.ad

    def set_field_kind(self, kind: str) -> None:
        """Gösterilen alan türünü değiştirir ve alanı yeniden üretir.

        Seviye TAŞINMAZ, yeni alanın kendi varsayılanına döner. Taşımak
        çoğu geçişte geçersiz bir sayı üretirdi: orbitalin 0.05'i fark
        haritasının bandında (0.0005-0.3) kaba bir yüzey, farkın 0.0005'i ise
        orbital kaydırıcısının çözünürlüğünün (0.001) ALTINDA kalıyor ve
        `_dogrula_seviye` tarafından reddedilirdi. Kaydırıcının aralığı da
        alanla birlikte değişir.

        Geçersiz ad `InvalidInputError` ile reddedilir ve pencerenin HİÇBİR
        alanına dokunulmaz -- doğrulama her şeyden önce yapıldığı için geri
        alınacak bir şey yok (kardeş `set_isosurface_level` ile aynı desen).

        Yeni ALAN da durum değişmeden ÖNCE üretilir, çünkü üretim de
        reddedebilir: sıfır elektronlu bir sistemin HOMO indeksi -1'dir ve
        orbital alanı üretilemez. Ölçüldü (bu sıra yokken, H2++ yoğunluk
        alanında): `set_field_kind("orbital")` `InvalidInputError` fırlatıyor
        ama önce alan türü, seviye, kaydırıcı aralığı ve seçicinin etkinliği
        değişmiş oluyordu -- ağlar ve alan seçici ise hâlâ yoğunluğundu.
        """
        tur = _dogrula_alan_turu(kind)
        if tur is self._alan_turu:
            return
        anahtar, alan = self._alan_getir(tur, self._orbital_index)
        self._alan_turu = tur
        self._alan_anahtari, self._alan = anahtar, alan
        self._level = tur.varsayilan_seviye
        # "Hangi orbital" sorusunun yoğunluk ve fark alanında karşılığı yok;
        # etkin ama etkisiz bir denetim yalan söylerdi (aynı karar
        # `TDSEWindow`'da bariyeri olmayan senaryolar için de verilmiş).
        self._secici.setEnabled(tur is _ORBITAL)
        self._kaydirici_araligini_kur()
        self._yenile()
        self._alan_secici_senkronla()

    def set_orbital_index(self, index: int) -> None:
        """Gösterilen orbitali değiştirir ve ağları yeniden çıkarır.

        Geçersiz indeks `InvalidInputError` ile reddedilir ve pencere eski
        orbitalinde kalır: hata YUTULMAZ ama durum da BOZULMAZ. Doğrulama ve
        yeni alanın üretimi durum değişmeden ÖNCE yapılır, yani geri alınacak
        bir şey yoktur (kardeş `set_isosurface_level` ile aynı desen).

        Sözleşme ALAN TÜRÜNDEN BAĞIMSIZDIR. Yoğunluk ve fark alanı orbital
        indeksine bağlı olmadığı için orada `evaluate_orbital` koşmaz ve onun
        aralık kontrolü de koşmazdı; ölçüldü, yoğunluk alanında
        `set_orbital_index(99)` KABUL EDİLİYORDU ve sonraki
        `set_field_kind("orbital")` pencereyi yarım bırakıyordu. Aralık
        seçicinin satırlarıdır (`result.orbital_energies`): seçicinin
        gösteremeyeceği bir orbital pencerenin durumu olamaz. Aynı sebeple o iki
        alanda indeks değişimi alanı YENİDEN ÜRETMEZ (ölçüldü, fark alanında
        176.5 ms boşa gidiyordu; bkz. `_alan_getir`).
        """
        indeks = int(index)
        n_orbital = len(self._result.orbital_energies)
        if not 0 <= indeks < n_orbital:
            raise InvalidInputError(
                f"orbital indeksi [0, {n_orbital}) aralığında olmalı "
                f"(seçicinin satırları), verilen: {indeks}"
            )
        anahtar, alan = self._alan_getir(self._alan_turu, indeks)
        self._orbital_index = indeks
        self._alan_anahtari, self._alan = anahtar, alan
        self._yenile()
        self._secici_senkronla()

    def set_isosurface_level(self, level: float) -> None:
        """İzoyüzey seviyesini değiştirir ve ağları yeniden çıkarır.

        Geçersiz seviye (`_dogrula_seviye`) pencerenin HİÇBİR alanına
        dokunmadan reddedilir. Kardeşi `set_orbital_index` ile aynı desen:
        geri alınacak bir şey yok çünkü doğrulama
        `_yenile`'den ÖNCE. Fark önemliydi: eski kodda `set_isosurface_level(
        nan)` `ValueError` fırlatıyordu ama fırlatmadan önce `_yenile` tamamen
        koşuyordu -- `_level=nan` kalıyor, ağlar (1764, 1764)'ten (0, 0)'a
        düşüyor, görünümdeki item 3'ten 1'e iniyor, panel "şu an nan" basıyor,
        kaydırıcı 0.05'te bayat kalıyordu. Yani "hata fırlattı" ile "durum
        korundu" aynı şey değildir.
        """
        self._level = _dogrula_seviye(level, self._alan_turu)
        self._yenile()
        self._kaydirici_senkronla()

    def mesh_vertex_counts(self) -> tuple[int, ...]:
        """Şu anki ağların tepe sayıları.

        UZUNLUK ALAN TÜRÜNE BAĞLIDIR ve bu bilgi taşır: işaretli alanlarda
        (orbital, yoğunluk farkı) iki sayı -- (pozitif, negatif) --, elektron
        yoğunluğunda TEK sayı döner. Yoğunluk kesin pozitiftir; ikinci ağ
        ölçüldüğünde her zaman boş çıkıyor (level=0.1'de (+2196, -0)), yani
        oraya bir 0 yazmak "negatif faz ölçüldü ve boş çıktı" diye okunurdu --
        oysa o alanda negatif faz diye bir şey YOKTUR.

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
        return tuple(len(mesh.vertices) for mesh in self._meshes)

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

    def analysis_text(self) -> str:
        """Okuma satırının ŞU AN bastığı metin -- WIDGET'tan okunur.

        `transparency_text()` ile aynı gerekçe: metni ikinci kez ÜRETEN bir
        metot, widget'a hiç yazmayan bir mutasyonu da yeşil bırakır (ölçüldü,
        kardeş yüzeyde panel 0 karakter basarken metot 1174 karakter
        döndürüyor ve dokuz testin dokuzu geçiyordu).
        """
        return self._okuma.text()

    # -- iç yapı -------------------------------------------------------------

    def _kur_arayuz(self) -> None:
        self._view = gl.GLViewWidget()
        self._view.setCameraPosition(distance=float(np.max(self._grid.spacing)) * 60.0)

        # Her satır kendi alan türünün adını VERİ olarak taşır (bkz.
        # `_alan_secici_degisti`); satırın KONUMU hiçbir yerde okunmaz.
        self._alan_secici = QtWidgets.QComboBox()
        for ad in _ALAN_TURLERI:
            self._alan_secici.addItem(ad, ad)
        self._alan_secici.setCurrentIndex(self._alan_secici.findData(self._alan_turu.ad))
        self._alan_secici.currentIndexChanged.connect(self._alan_secici_degisti)

        self._secici = QtWidgets.QComboBox()
        for etiket in self._orbital_etiketleri():
            self._secici.addItem(etiket)
        self._secici.setCurrentIndex(self._orbital_index)
        self._secici.setEnabled(self._alan_turu is _ORBITAL)
        self._secici.currentIndexChanged.connect(self._secici_degisti)

        self._kaydirici = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._kaydirici_araligini_kur()
        self._kaydirici.valueChanged.connect(self._kaydirici_degisti)

        self._okuma = QtWidgets.QLabel()
        self._okuma.setWordWrap(True)
        self._okuma.setStyleSheet("font-family: Consolas, monospace;")
        self._okuma_satirini_guncelle()

        # Üst sınır 340'tan 440'a ÇIKTI ve pencere 960x800'den 1400x950'ye
        # büyüdü, çünkü beyan büyüdü: pencere artık üç alan türü, `analyse()`in
        # üç uyarısı ve fark alanında promolekül referansı taşıyor. Ölçüldü
        # (su/STO-3G, panel metni karakter / gereken px):
        #     orbital            2484 / 388 px (1400 px genişlikte)
        #     elektron yoğunluğu 3294 / 500 px
        #     yoğunluk farkı     4645 / 682 px
        #
        # KIRPMA ÜÇÜNÜN İKİSİNDE KAÇINILMAZ ve bu bilinçli kabul edildi:
        # 682 px'i sığdırmak 3B görünüme 1400x950'de 131 px bırakırdı, yani
        # pencerenin asıl nesnesini yok ederdi. Ölçülen seçim (1400x950):
        #     maks 440 -> orbital 0 satır gizli / 3B 565 px
        #                 yoğunluk 4 gizli, fark 17 gizli / 3B 373 px
        #     maks 480 -> yoğunluk 1 gizli ama 3B 333 px'e düşüyor, fark yine
        #                 14 gizli -- üç satır için 40 px 3B alan verilmiyor.
        # AÇILIŞ durumu (orbital) tam görünür kalır; ötekilerde panel başlığı
        # "devamı var, kaydırın" der ve metnin kendisi (`text()`) kırpılmaz.
        self._panel = TransparencyPanel(max_height=440)

        ust = QtWidgets.QHBoxLayout()
        ust.addWidget(QtWidgets.QLabel("Alan:"))
        ust.addWidget(self._alan_secici, 1)
        ust.addWidget(QtWidgets.QLabel("Orbital:"))
        ust.addWidget(self._secici, 1)
        ust.addWidget(QtWidgets.QLabel("İzoyüzey seviyesi:"))
        ust.addWidget(self._kaydirici, 1)

        govde = QtWidgets.QVBoxLayout()
        govde.addLayout(ust)
        govde.addWidget(self._view, 1)
        govde.addWidget(self._okuma)
        govde.addWidget(self._panel)

        merkez = QtWidgets.QWidget()
        merkez.setLayout(govde)
        self.setCentralWidget(merkez)
        self.resize(1400, 950)

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

    def _alan_secici_degisti(self, index: int) -> None:
        """Seçilen satırın alan türü satırın VERİSİNDEN okunur, konumundan değil.

        Eskiden `list(_ALAN_TURLERI)[index]` idi, yani seçicideki sıra ile
        sözlük sırasının aynı kalmasına dayanıyordu. Ölçüldü: seçicinin modeli
        sıralandığında (Qt'nin kendi API'si) "orbital" yazan satır yoğunluğu,
        "elektron yoğunluğu" yazan satır orbitali açıyordu.

        Pencere geçişi REDDEDERSE (sıfır elektronlu sistemde orbital alanı)
        seçici pencerenin gerçek alanına geri döner ve istisna yükselir. Qt
        istisnayı sinyal işleyicisinde yutar; seçici geri dönmeseydi "orbital"
        gösterirken pencere yoğunlukta kalırdı -- sessiz desenkron.
        """
        try:
            self.set_field_kind(self._alan_secici.itemData(index))
        except InvalidInputError:
            self._alan_secici_senkronla()
            raise

    def _kaydirici_degisti(self, value: int) -> None:
        self.set_isosurface_level(value / self._alan_turu.kaydirici_olcek)

    def _secici_senkronla(self) -> None:
        if self._secici.currentIndex() != self._orbital_index:
            self._secici.blockSignals(True)
            self._secici.setCurrentIndex(self._orbital_index)
            self._secici.blockSignals(False)

    def _alan_secici_senkronla(self) -> None:
        hedef = self._alan_secici.findData(self._alan_turu.ad)
        if self._alan_secici.currentIndex() != hedef:
            self._alan_secici.blockSignals(True)
            self._alan_secici.setCurrentIndex(hedef)
            self._alan_secici.blockSignals(False)

    def _kaydirici_araligini_kur(self) -> None:
        """Kaydırıcının aralığını AKTİF ALANIN kendi bandına kurar.

        Değer, kırpılmaya karşı korumalı tek yoldan (`_kaydirici_senkronla`)
        konur; açılış seviyesi varsayılan üst sınırın üstünde olabilir
        (O 1s'in maks genliği 4.90).
        """
        self._kaydirici.blockSignals(True)
        try:
            # Sıra önemli: yeni aralık eskisiyle kesişmeyebilir ve Qt önce
            # minimum'u koyup sonra maximum'u koyduğunda aradaki geçici
            # aralıkta değeri kırpar. blockSignals ile birlikte
            # `_kaydirici_senkronla` doğru değeri sonradan geri yazar.
            self._kaydirici.setMinimum(self._alan_turu.kaydirici_min)
            self._kaydirici.setMaximum(self._alan_turu.kaydirici_maks)
        finally:
            self._kaydirici.blockSignals(False)
        self._kaydirici_senkronla()

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
        hedef = int(round(self._level * self._alan_turu.kaydirici_olcek))
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

    def _alani_uret(self, tur: _AlanTuru, orbital_index: int) -> ScalarField:
        """Verilen alan türünün ÜRETİCİSİNİ çağırır; pencerenin durumuna dokunmaz.

        Üç dalın üçü de `observable/hf_obs.py`'nin ayrı bir public
        fonksiyonudur ve hiçbiri ötekinden türetilemez: orbital TEK bir MO'dur
        (`result.coefficients`'ın bir sütunu), yoğunluk ve fark ise yoğunluk
        MATRİSİNDEN kontrakte edilir. Orbital dalında `ScalarField` burada
        kurulur çünkü `evaluate_orbital` çıplak bir dizi döndürür; öteki ikisi
        kendi `Provenance`'larını zaten taşır ve o beyan panele AYNEN gider
        (bkz. `_provenance`).
        """
        if tur is _ORBITAL:
            degerler = evaluate_orbital(self._result, orbital_index, self._grid)
            return ScalarField(
                values=degerler, grid=self._grid, provenance=self._result.provenance
            )
        if tur is _YOGUNLUK:
            return electron_density(self._result, self._grid)
        return density_difference(self._result, self._grid)

    def _alan_getir(
        self, tur: _AlanTuru, orbital_index: int
    ) -> tuple[tuple[str, int | None], ScalarField]:
        """(anahtar, alan): önbellek anahtarı tutuyorsa önbellekteki, yoksa YENİ.

        Alanı belirleyen durum ANAHTARDIR: alan türü ve -- yalnızca orbital
        alanında -- orbital indeksi. Yoğunluk ve fark alanı tek bir orbitale
        bağlı değildir; indeksleri anahtara girseydi o iki alanda her orbital
        değişimi alanı boşuna yeniden üretirdi (ölçüldü, fark alanında
        176.5 ms). Seviye anahtarda YOKTUR: seviye değişiminde alan yeniden
        üretilmez -- ölçüldü, `density_difference` bu ızgarada 0.170 s sürüyor
        ve kaydırıcı sürüklenirken her ara değer için yeniden hesaplamak
        arayüzü kilitlerdi (aynı gerekçe `TDSEWindow`'un 250 ms
        geciktirmesinin de sebebi).

        Pencerenin durumunu DEĞİŞTİRMEZ: üretim reddederse (geçersiz orbital
        indeksi) çağıranın hiçbir alanı değişmemiş olur. Sonucu önbelleğe
        yazanlar kurucu, `_guncel_alan` ve iki kamu ayarlayıcısıdır.
        """
        anahtar = (tur.ad, orbital_index if tur is _ORBITAL else None)
        if anahtar == self._alan_anahtari:
            return anahtar, self._alan
        return anahtar, self._alani_uret(tur, orbital_index)

    def _guncel_alan(self) -> ScalarField:
        """ŞU ANKİ (alan türü, orbital indeksi) durumunun alanı.

        `_alan` bir ÖNBELLEKTİR ve eskiden elle yazılmış iki geçersizleştirmeye
        (`set_field_kind`, `set_orbital_index`) bağlıydı; durumu değiştiren
        üçüncü bir yol onu sessizce bayatlatabilirdi. Ölçüldü: `_orbital_index`
        4'ten 0'a çekilip `_yenile` çağrılınca ağlar HOMO'nun (1764, 1764)
        ağları kalıyor (O 1s'inki (0, 412)), panel ise "orbital_index=0"
        basıyordu. Alanı OKUYAN iki yer (`_yenile`,
        `_provenance`) artık buradan okur ve önbellek anahtarı tutmuyorsa alan
        yeniden üretilir -- geçersizleştirmeyi kimsenin hatırlaması gerekmez.
        """
        self._alan_anahtari, self._alan = self._alan_getir(
            self._alan_turu, self._orbital_index
        )
        return self._alan

    def _yenile(self) -> None:
        """Ağları yeniden çıkarır, çizer ve beyanı tazeler."""
        yeni = self._aglari_cikar(self._guncel_alan())
        self._meshes = yeni
        self._ag_itemlerini_ciz(yeni)
        self._panel.set_provenance(self._provenance())

    def _aglari_cikar(self, alan: ScalarField) -> tuple[Mesh, ...]:
        """İşaretli alanda ±çift, işaretsiz alanda tek ağ.

        `extract_isosurface`'in var olma sebebi burada: elektron yoğunluğu
        KESİN POZİTİFTİR, yani `-level` izoyüzeyi hiçbir zaman yoktur.
        Ölçüldü (su/STO-3G, level=0.1): `extract_signed_pair` (+2196, -0)
        veriyor -- ikinci ağ her zaman boş, ve onu çizmek/saymak "negatif faz
        ölçüldü, boş çıktı" diye okunurdu. Fark haritasında ise çift ŞART:
        (+1764, -2020), ikisi de dolu ve iki ayrı fiziksel olguyu gösteriyor.
        """
        if self._alan_turu.isaretli:
            return extract_signed_pair(alan, self._level)
        return (extract_isosurface(alan, self._level),)

    def _okuma_satirini_guncelle(self) -> None:
        """`analyse()` raporunu ekrana taşıyan satır.

        Sayılar rapordan BİREBİR gelir; bu metot hiçbir şey hesaplamaz.
        Ölçüldü (su/STO-3G): Mulliken [-0.365510, +0.182755, +0.182755],
        dipol 1.725514 D, HOMO -0.391274, LUMO 0.605136, aralık 0.996410 Ha,
        Mayer O1-H2 0.954201.

        Atom etiketleri 1 tabanlı sıra numarası taşır (O1, H2, H3) çünkü aynı
        sembol birden çok kez geçebilir ve "O-H 0.9542" iki kez yazıldığında
        hangi H olduğu okunamazdı.

        homo/lumo/gap `None` olabilir ve o zaman "yok" yazılır: 0.000000
        basmak "boşluk yok, metalik" diye okunurdu -- aynı gerekçe
        `AnalysisReport` ve `StabilityVerdict` dokstringlerinde zaten yazılı.
        Ölçüldü, ulaşılabilir: He/STO-3G'nin tek orbitali var, LUMO yok.

        KOŞULLU GEREKÇE NOTLARI DA BU SATIRDA (K1, Osman kararı 11.10.2026).
        `analyse()` yüklü sistemde `_CHARGED_DIPOLE_NOTE`, boş orbitali
        olmayan sistemde `_NO_LUMO_NOTE`, işgal edilmiş orbitali olmayan
        sistemde `_NO_HOMO_NOTE` ekler; bunlar bu satırın sayılarının
        (dipol, HOMO, LUMO, aralık) GEREKÇESİDİR ve her biri kendi satırında
        "Analiz notu:" önekiyle basılır. Önceden hiçbir pencerede
        görünmüyordu (ölçüldü, HEAD a9ec48a: He/STO-3G "LUMO yok" basıyor ama
        notun metni ne bu satırda ne panelde; HeH+ 2.8368 D basıyor ve
        dipolün orijine bağlı olduğu hiçbir yerde yazmıyordu). Hangi notun
        basılacağına pencere KARAR VERMEZ: not raporun `expected_error`ında
        ise basılır -- koşulu (charge != 0, n_occupied ...) burada yeniden
        türetmek, motorun kararının ikinci bir kopyası olurdu. Nötr ve
        LUMO'su olan bir sistemde (su) hiçbir not basılmaz: koşulsuz bir
        uyarı, doğru sayıya duyulan güveni boşa harcatırdı (`analyse()`
        dokstringindeki gerekçenin aynısı).

        Ulaşılabilirlik: `main.py` yolu charge=0 kurar, yani GUI'den yalnız
        `_NO_LUMO_NOTE`e ulaşılır; yüklü dipol ve HOMO yok notlarına yalnız
        bu sınıfın kurucusundan (yüklü bir `HFResult` ile) ulaşılır. Sıfır
        elektronlu sistem orbital alanında açılmaz (HOMO indeksi -1,
        `InvalidInputError`), yoğunluk alanlarında açılır. Bedeli ölçüldü
        (1400x950, ekransız `show()`): okuma satırı su 40 px, He 54 px, HeH+
        138 px; 3B görünüm su ve He'de 523 px, HeH+'da 425 px. Notsuz HEAD'e
        (a9ec48a) göre: su DEĞİŞMEDİ (523), He 551 -> 523, HeH+ 523 -> 425,
        H2++ (yoğunluk alanı, iki not; okuma satırı 40 -> 166 px) 565 -> 439 px.
        """
        rapor = self._analiz
        etiketler = [
            f"{atom.symbol}{i + 1}" for i, atom in enumerate(self._molecule.atoms)
        ]
        yukler = " · ".join(
            f"{ad} {q:+.4f}" for ad, q in zip(etiketler, rapor.mulliken)
        )
        baglar = " · ".join(
            f"{etiketler[a]}-{etiketler[b]} {rapor.bond_orders[a, b]:.4f}"
            for a in range(len(etiketler))
            for b in range(a + 1, len(etiketler))
        )

        def _sayi(deger: float | None, ad: str) -> str:
            return f"{ad} yok" if deger is None else f"{ad} {deger:.6f}"

        satirlar = [
            f"Yükler (Mulliken): {yukler}",
            f"Dipol {rapor.dipole_debye:.4f} D · "
            + _sayi(rapor.homo, "HOMO")
            + " · "
            + _sayi(rapor.lumo, "LUMO")
            + " · "
            + _sayi(rapor.gap, "aralık")
            + " Ha",
        ]
        # Tek atomlu bir sistemde bağ ÇİFTİ yoktur; boş bir "Mayer bağ
        # mertebeleri:" başlığı basmak, ölçülmüş ama boş çıkmış bir liste diye
        # okunurdu (ölçüldü: He/STO-3G'de tek atom, sıfır çift).
        if baglar:
            satirlar.append(f"Mayer bağ mertebeleri: {baglar}")
        beklenen = rapor.provenance.expected_error
        satirlar.extend(
            f"Analiz notu: {not_}"
            for not_ in _KOSULLU_ANALIZ_NOTLARI
            if not_ in beklenen
        )
        self._okuma.setText("\n".join(satirlar))

    def _ag_itemlerini_ciz(self, meshes: tuple[Mesh, ...]) -> None:
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

    @staticmethod
    def _birlestir(*gruplar: tuple[str, ...]) -> tuple[str, ...]:
        """Beyan gruplarını sırayı bozmadan, tekrarsız birleştirir.

        Aynı desen `observable/hf_obs.py:_merged_declarations` ve
        `engine/hf/analysis.py:_merged` içinde de var; üçü de aynı sebeple
        (birden çok kaynağın ortak maddeleri iki kez yazılmasın) ve katman
        kuralı yüzünden ayrı ayrı duruyor.
        """
        return tuple(dict.fromkeys(item for grup in gruplar for item in grup))

    def _provenance(self) -> Provenance:
        """ALANIN beyanı + `analyse()`in beyanı + bu pencerenin gösterim seçimleri.

        Hiçbiri SİLİNMEZ, yalnızca EKLENİR. Panel kendi başına fizik bilgisi
        üretmez (bkz. transparency.py dokstring'i), o yüzden birleştirme
        burada, veri katmanında olur.

        TEMEL ALANIN KENDİ BEYANIDIR, motorunki değil -- ve fark önemli:
        `electron_density` ve `density_difference` `ScalarField`leriyle
        birlikte kendi `Provenance`'larını taşıyor (sonlu kutu kesmesi, ızgara
        paritesi, promolekülün serbest-atom referansı, N-temsil edilebilirlik).
        Motorun ham beyanını basmak o maddelerin hiçbirini ekrana taşımazdı.
        Orbital alanında ikisi aynı nesnedir (`_alani_uret` `ScalarField`i
        `result.provenance` ile kurar), o yüzden bu dal davranışı değiştirmez.

        `analyse()`in beyanı KATILIR çünkü okuma satırı onun sayılarını
        basıyor: Mulliken/Löwdin'in baz duyarlılığı (LiH/STO-3G'de İŞARET bile
        ters çıkıyor), Mayer'in RHF'de ayrışmayı gösterememesi ve HOMO/LUMO'nun
        Koopmans düzeyinde kalması. Ekranda -0.3655 yazıp bu üçünü yazmamak,
        sayıya hak etmediği bir kesinlik verirdi.

        `analyse().provenance.expected_error` PANELE BİLİNÇLİ OLARAK
        katılmıyor: KOŞULSUZ kısmının taşıdığı üç uyarı yukarıdaki
        `approximations`/`neglected` maddelerinin yeniden ifadesidir (aynı LiH
        sayısı, aynı Mayer ve Koopmans hükmü) ve panelin "Beklenen hata"
        satırı alanın KENDİ hata bütçesini taşımak zorunda -- ikisini
        birleştirmek, hangi hatanın hangi büyüklükte olduğu sorusunu
        bulandırırdı. Bu gerekçe KOŞULLU notları (yüklü dipol, LUMO yok, HOMO
        yok) KAPSAMIYORDU: onlar hiçbir maddenin yeniden ifadesi değil, okuma
        satırındaki sayıların gerekçesidir ve 11.10.2026'dan beri koşulları
        oluştuğunda o satırda basılır (bkz. `_okuma_satirini_guncelle`; K1).
        Panel yine alanın bütçesini taşır. Buna karşılık `stability` MOTORUN sonucundan
        alınır, alanın beyanından değil: `electron_density`/`density_difference`
        onu `None` bırakıyor ve olduğu gibi taşınsaydı panel kararlılık satırını
        alan değişince hiç basmazdı -- okuyucu bunu "sorun yok" diye okurdu
        (aynı gerekçe `Provenance.stability` alanının dokstring'inde yazılı).
        """
        temel = self._guncel_alan().provenance
        analiz = self._analiz.provenance
        tanilar = dict(temel.diagnostics)
        tanilar["field_kind"] = self._alan_turu.ad
        # Orbital indeksi YALNIZCA orbital alanında yazılır: öteki iki alan
        # yoğunluk matrisinden üretilir ve tek bir orbitale bağlı değildir,
        # oraya bir indeks basmak uydurma olurdu.
        if self._alan_turu is _ORBITAL:
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

        # Promolekülün serbest-atom referansı YALNIZCA fark alanında beyan
        # edilir: öteki iki alanda promolekül diye bir şey yoktur ve oraya
        # yazmak yalan olurdu.
        referans: tuple[str, ...] = ()
        if self._alan_turu is _FARK:
            referans = (
                _promolekul_referans_beyani(
                    self._molecule, self._result.basis_name, self._serbest_atom_onbellegi
                ),
            )
            # Ölçülen referansın taşıdığı TOPLAM elektron, ızgara üzerinden
            # integre edilen promolekül elektronunun (aynı diagnostics'te,
            # "integrated_electrons_promolecule") YANINDA durur: ilki tam,
            # ikincisi ızgaranın kestiği sayıdır ve fark okuyucuya ızgara
            # hatasının büyüklüğünü doğrudan gösterir.
            tanilar["promolecule_reference_electrons"] = float(
                sum(
                    _serbest_atom_referansi(
                        atom.symbol, self._result.basis_name, self._serbest_atom_onbellegi
                    )[1]
                    for atom in self._molecule.atoms
                )
            )

        return dataclasses.replace(
            temel,
            approximations=self._birlestir(
                temel.approximations,
                analiz.approximations,
                _gosterim_yaklasimlari(self._level, self._alan_turu),
                referans,
            ),
            neglected=self._birlestir(
                temel.neglected, analiz.neglected, _GOSTERIM_IHMALLERI
            ),
            diagnostics=tanilar,
            stability=self._result.provenance.stability,
        )
