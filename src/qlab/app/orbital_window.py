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
   negatif ağ 412 tepe veriyor -- yani o orbital sayısal olarak her yerde
   negatif. Başka bir kütüphane aynı orbitali ters işaretle verse iki renk
   yer değiştirirdi ve hiçbir fizik değişmezdi. İki renk yalnızca AYNI
   orbitalin kendi içindeki bağıl faz farkını gösterir.
2. **İzoyüzey seviyesi bakanın seçimidir.** Ölçüldü (su HOMO, pad=4.0,
   n=60): level 0.02'de 2850 tepe, 0.05'te 1764, 0.50'de 114, 0.90'da hiç.
   Aynı orbital, farklı "şekil". Seviye bu yüzden panelde sayı olarak basılır.
3. **Izgara çekirdek kabuğunu çözmez.** Ölçüldü: aynı O 1s orbitalinin ızgara
   üzerindeki maksimum genliği n=60'ta 4.9041, n=80'de 5.9271 -- yakınsamamış.
   Çekirdeğe yakın yüzeyler nitel gösterimdir, nicel değil.

Bu üçü şeffaflık paneline motorun kendi `Provenance`'ının ÜSTÜNE eklenir;
motorunkiler (Born-Oppenheimer, sonlu baz, ızgara aralığı) olduğu gibi taşınır.

**Ekransız kip.** OpenGL bağlamı ancak boyama sırasında gerekir; kurulum
(GLViewWidget, GLMeshItem, GLScatterPlotItem, setCentralWidget, close)
`QT_QPA_PLATFORM=offscreen` altında sorunsuz koşar -- ölçüldü. Testler bu
yüzden pencerenin DURUMUNU doğrular, çizimin kendisini değil. Görsel doğrulama
insanın işidir (README, kabul kriteri 3).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtWidgets

from qlab.app.transparency import TransparencyPanel, format_panel
from qlab.engine.hf.scf import HFResult
from qlab.errors import InvalidInputError
from qlab.observable.hf_obs import ScalarField, VolumeGrid, evaluate_orbital
from qlab.provenance import Provenance
from qlab.types import Molecule
from qlab.viz.isosurface import Mesh, extract_signed_pair

#: ±faz renkleri (RGBA). Kırmızı/mavi ayrımı yalnızca BAĞIL fazı gösterir;
#: hangi lobun hangi renk olduğu keyfidir (modül dokstring'i, madde 1).
_POZITIF_RENK = (0.90, 0.25, 0.25, 0.45)
_NEGATIF_RENK = (0.25, 0.45, 0.95, 0.45)
_CEKIRDEK_RENK = (0.95, 0.95, 0.95, 1.0)

#: Kaydırıcı tam sayı taşır; seviye = konum / _KAYDIRICI_OLCEK.
_KAYDIRICI_OLCEK = 1000.0

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
    """
    return (
        _ISARET_BEYANI,
        f"izoyüzey seviyesi bir gösterim seçimidir, hesabın sonucu değildir "
        f"(şu an {level:g})",
    )


class OrbitalWindow(QtWidgets.QMainWindow):
    """Tek bir molekülün orbitallerini gösteren döndürülebilir 3B pencere.

    Açılışta HOMO gösterilir. `set_orbital_index` geçersiz bir indeks için
    `InvalidInputError` fırlatır -- hata `evaluate_orbital`'dan gelir ve
    pencere onu YUTMAZ; yutulsaydı kullanıcı yanlış orbitale baktığını
    bilmezdi. Reddedilen bir seçim pencerenin durumunu da değiştirmez.
    """

    def __init__(
        self,
        result: HFResult,
        molecule: Molecule,
        *,
        padding: float = 4.0,
        points_per_axis: int = 60,
        level: float = 0.05,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._result = result
        self._molecule = molecule
        self._grid = VolumeGrid.around(
            molecule, padding=padding, points_per_axis=points_per_axis
        )
        self._level = float(level)
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
        """İzoyüzey seviyesini değiştirir ve ağları yeniden çıkarır."""
        self._level = float(level)
        self._yenile()
        self._kaydirici_senkronla()

    def mesh_vertex_counts(self) -> tuple[int, int]:
        """Şu anki (pozitif, negatif) ağların tepe sayıları.

        Çizimin kendisi ekransız kipte doğrulanamadığı için testler veri
        hattını buradan izler.
        """
        if self._meshes is None:  # pragma: no cover - _yenile her zaman önce koşar
            return (0, 0)
        pozitif, negatif = self._meshes
        return (len(pozitif.vertices), len(negatif.vertices))

    def transparency_text(self) -> str:
        """Şeffaflık panelinin bastığı metnin aynısı.

        Panelin kendisi bir widget olduğu için ekransız kipte okunması
        zahmetli; bu metot aynı `Provenance`'ı aynı biçimleyiciden geçirir.
        """
        return format_panel(self._provenance())

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
        self._kaydirici.setMinimum(2)
        self._kaydirici.setMaximum(900)
        self._kaydirici.setValue(int(round(self._level * _KAYDIRICI_OLCEK)))
        self._kaydirici.valueChanged.connect(self._kaydirici_degisti)

        self._panel = TransparencyPanel()

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

    def _secici_degisti(self, index: int) -> None:  # pragma: no cover - GUI sinyali
        if index != self._orbital_index:
            self.set_orbital_index(index)

    def _kaydirici_degisti(self, value: int) -> None:  # pragma: no cover - GUI sinyali
        self.set_isosurface_level(value / _KAYDIRICI_OLCEK)

    def _secici_senkronla(self) -> None:
        if self._secici.currentIndex() != self._orbital_index:
            self._secici.blockSignals(True)
            self._secici.setCurrentIndex(self._orbital_index)
            self._secici.blockSignals(False)

    def _kaydirici_senkronla(self) -> None:
        hedef = int(round(self._level * _KAYDIRICI_OLCEK))
        if self._kaydirici.value() != hedef:
            self._kaydirici.blockSignals(True)
            self._kaydirici.setValue(hedef)
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
        for eski in self._mesh_items:
            self._view.removeItem(eski)
        self._mesh_items = []
        for mesh, renk in zip(meshes, (_POZITIF_RENK, _NEGATIF_RENK)):
            if len(mesh.vertices) == 0:
                continue
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
        return dataclasses.replace(
            temel,
            approximations=temel.approximations + _gosterim_yaklasimlari(self._level),
            neglected=temel.neglected + _GOSTERIM_IHMALLERI,
            diagnostics=tanilar,
        )
