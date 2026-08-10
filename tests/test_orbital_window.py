"""Orbital penceresi testleri.

Bu dosyadaki testler gerçek bir ekran olmadan koşar (QT_QPA_PLATFORM=offscreen).
Offscreen kipte OpenGL bağlamı oluşturulamayabilir; o yüzden çizimin kendisi
değil, veri hattı ve pencere durumu doğrulanır. Gerçek görsel doğrulama
insanın işidir ve README'de böyle yazılıdır.

**Planın `monkeypatch.setenv` deseni KULLANILMADI, bilerek.** Plan Task 9'un
testleri `QT_QPA_PLATFORM`'u test gövdesinin içinde kuruyordu; o çok geç.
Platform, `QApplication` yaratılırken seçilir ve bu depoda
`tests/test_transparency.py` modül düzeyinde, PySide6 içe aktarılmadan ÖNCE bir
QApplication kuruyor. Aynı oturumda o modül önce toplanırsa test gövdesindeki
`setenv` hiçbir işe yaramaz. Burada `test_transparency.py`'deki kanıtlanmış
desen tekrarlanıyor: ortam modül düzeyinde `setdefault` ile kurulur, uygulama
bir kez yaratılır, ve `requires_offscreen` yalnızca platform gerçekten
kurulamazsa devreye giren bir güvenlik ağıdır.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def test_pyqtgraph_opengl_import_edilebiliyor():
    """PyOpenGL kurulu değilse bu modülün tamamı çalışmaz.

    Bilinçli olarak `importorskip` DEĞİL: PyOpenGL `pyproject.toml`'da
    bağımlılıktır, yani yokluğu bozuk bir ortam demektir. Bu depo
    `tests/hf/conftest.py`'de aynı gerekçeyle `pytest.skip` yerine
    `pytest.fail` kullanır — sessizce atlanan bir doğrulama, doğrulama
    değildir.
    """
    import pyqtgraph.opengl as gl

    assert hasattr(gl, "GLViewWidget")
    assert hasattr(gl, "GLMeshItem")


def test_isosurface_kure_uzerinde_calisiyor():
    """pg.isosurface marching cubes'tur ve saf NumPy'dir; OpenGL gerektirmez."""
    import pyqtgraph as pg

    x, y, z = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
    data = np.ascontiguousarray(x * x + y * y + z * z, dtype=np.float32)
    vertices, faces = pg.isosurface(data, 0.5)
    assert len(vertices) > 100
    assert len(faces) > 100
    assert faces.shape[1] == 3


# ---------------------------------------------------------------------------
# Ekransız (offscreen) pencere testleri
#
# Bu ortamda offscreen platformunun gerçekten çalıştığı önceden ölçüldü:
# GLViewWidget, GLMeshItem, GLScatterPlotItem, QMainWindow.setCentralWidget ve
# close() hepsi hatasız kuruldu (OpenGL bağlamı ancak boyama sırasında
# gerekiyor, kurulumda değil). Bu yüzden aşağıdaki testlerin atlanması
# BEKLENMEZ — her zaman atlayan bir test testten daha kötüdür.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# İçe aktarmalar bilinçli olarak `try` DIŞINDA. `tests/test_transparency.py`
# kendi modülünü de probe'un içine alıyor; o desen olduğu gibi kopyalandığında
# ölçüldü ki `qlab.app.orbital_window` HİÇ YOKKEN yedi testin yedisi de
# ATLANIYOR, hiçbiri kırmızı olmuyor. Yani TDD'nin kırmızı adımı sessizce
# kayboluyordu. Eksik bir kendi modülümüz "platform yok" değil, BOZUK BİR
# YAPI'dır ve toplama sırasında patlamalıdır.
from PySide6 import QtWidgets

from qlab.app.orbital_window import OrbitalWindow
from qlab.engine.hf.scf import solve
from qlab.errors import InvalidInputError
from qlab.types import Molecule

try:
    _qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _OFFSCREEN_OK = True
except Exception:  # pragma: no cover - bu ortamda offscreen platformu kurulamazsa
    _OFFSCREEN_OK = False

requires_offscreen = pytest.mark.skipif(
    not _OFFSCREEN_OK, reason="Qt ekransız (offscreen) platformu bu ortamda kurulamadı"
)

#: `reference/hf_energies.json`'daki `h2o.atom` ile aynı geometri. Doğrudan
#: yazıldı çünkü `hf_energies` fixture'ı yalnızca `tests/hf/conftest.py`'de
#: tanımlı ve kardeş dizinlere sızmaz.
_SU = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


@pytest.fixture(scope="module")
def su_penceresi():
    """Su/STO-3G için tek bir pencere; SCF ve ızgara testler arasında paylaşılır.

    Ölçüldü: SCF 0.05 s, yedi orbitalin tamamının ağ çıkarımı 0.60 s
    (pad=4.0, n=60). Test başına yeniden kurmak gereksiz.
    """
    molecule = Molecule.from_string(_SU)
    pencere = OrbitalWindow(solve(molecule, "sto-3g"), molecule)
    yield pencere, molecule
    pencere.close()


@requires_offscreen
def test_pencere_homo_ile_aciliyor(su_penceresi):
    """Açılışta HOMO gösterilir ve şeffaflık paneli motorun beyanını taşır.

    Su/STO-3G'de yedi orbital var, beşi dolu; HOMO indeksi 4.
    "Born-Oppenheimer" metni pencerenin kendi uydurması DEĞİL: `scf.py`'nin
    kendi `Provenance.approximations` listesinden geliyor (scf.py:116).
    """
    pencere, molecule = su_penceresi
    assert pencere.orbital_index == molecule.n_electrons // 2 - 1 == 4
    assert "Born-Oppenheimer" in pencere.transparency_text()


@requires_offscreen
def test_orbital_degistirilince_ag_yeniden_uretiliyor(su_penceresi):
    """Seçici değişince ağ gerçekten yeniden çıkarılmalı, önbellekten dönmemeli.

    Ölçülen tepe sayıları (pad=4.0, n=60, level=0.05):
        orbital 4 (HOMO) : (+1764, -1764)
        orbital 0 (O 1s) : (+   0, -412)

    Test sonunda HOMO'ya geri dönülür; fixture modül kapsamlıdır ve sıradaki
    test bozulmuş bir durum devralmamalıdır.
    """
    pencere, molecule = su_penceresi
    homo = molecule.n_electrons // 2 - 1
    try:
        once = pencere.mesh_vertex_counts()
        assert once == (1764, 1764)
        pencere.set_orbital_index(0)
        assert pencere.orbital_index == 0
        assert pencere.mesh_vertex_counts() == (0, 412)
        assert pencere.mesh_vertex_counts() != once
    finally:
        pencere.set_orbital_index(homo)


@requires_offscreen
def test_izoseviye_degisince_ag_kuculuyor(su_penceresi):
    """Kaydırıcı gerçekten bir şey yapmalı: yüksek seviye = küçük yüzey.

    Ölçüldü (HOMO, pad=4.0, n=60), maks|psi| = 0.697796:
        level 0.02 -> (2850, 2850)
        level 0.05 -> (1764, 1764)
        level 0.10 -> (1112, 1112)
        level 0.50 -> ( 114,  114)
        level 0.90 -> (   0,    0)   <- maks|psi|'nin üstünde, yüzey yok

    Bu test bir SIRALAMA iddiasıdır, tam sayı kilidi değil: ızgara değişirse
    sayılar değişir ama sıralama fizikten gelir.
    """
    pencere, _ = su_penceresi
    try:
        pencere.set_isosurface_level(0.02)
        genis = pencere.mesh_vertex_counts()
        pencere.set_isosurface_level(0.50)
        dar = pencere.mesh_vertex_counts()
        assert sum(genis) > sum(dar) > 0
        pencere.set_isosurface_level(0.90)
        assert pencere.mesh_vertex_counts() == (0, 0)
    finally:
        pencere.set_isosurface_level(0.05)


@requires_offscreen
def test_seffaflik_paneli_isaretin_keyfi_oldugunu_beyan_ediyor(su_penceresi):
    """±faz renkleri MUTLAK bir anlam taşımaz; pencere bunu söylemek zorunda.

    Bir özvektörün global işareti keyfidir: `psi` bir çözümse `-psi` de aynı
    enerjiyle aynı çözümdür. Bu soyut bir uyarı değil, bu depoda ÖLÇÜLEN bir
    olgu -- su/STO-3G'nin 0. orbitalinde (O 1s) pozitif ağ BOŞ çıkıyor
    (0 tepe) ve negatif ağ 412 tepe veriyor; yani o orbital sayısal olarak
    her yerde negatif. Aynı hesabı başka bir kütüphane ters işaretle verse
    renkler yer değiştirirdi ve HİÇBİR FİZİK değişmezdi.

    Panel bunu yazmazsa iki renk, hesabın taşımadığı bir anlamı ima eder.
    """
    pencere, _ = su_penceresi
    metin = pencere.transparency_text().lower()
    assert "işareti keyfi" in metin


@requires_offscreen
def test_seffaflik_paneli_izoseviyeyi_sayi_olarak_basiyor(su_penceresi):
    """İzoyüzey seviyesi hesabın bir sonucu değil, bakanın bir seçimidir.

    Yukarıdaki ölçüm bunu somutlaştırıyor: aynı HOMO, level 0.02'de 2850
    tepe, 0.50'de 114 tepe veriyor. "Orbitalin şekli" diye gösterilen şey
    seçilen seviyeye bağlı; panel seviyeyi sayı olarak basmak zorunda ve
    seviye değişince o sayı GÜNCELLENMELİ.
    """
    pencere, _ = su_penceresi
    try:
        assert "0.05" in pencere.transparency_text()
        pencere.set_isosurface_level(0.20)
        assert "0.2" in pencere.transparency_text()
    finally:
        pencere.set_isosurface_level(0.05)


@requires_offscreen
def test_seffaflik_paneli_izgaranin_cekirdek_kabugunu_cozemedigini_beyan_ediyor(
    su_penceresi,
):
    """Izgara, çekirdeğe yakın tepeyi çözmüyor; bu beyan edilmezse gizlenir.

    Ölçüldü: su/STO-3G'de 0. orbitalin (O 1s) ızgara üzerindeki maksimum
    genliği n=60'ta 4.9041, n=80'de 5.9271 -- yani sayı YAKINSAMAMIŞ,
    ızgaraya bağlı. Gösterilen çekirdek kabuğu yüzeyi bu yüzden nicel
    değildir.
    """
    pencere, _ = su_penceresi
    metin = pencere.transparency_text().lower()
    assert "çekirdek kabuk" in metin


@requires_offscreen
def test_gecersiz_orbital_secimi_reddediliyor():
    """Pencere `evaluate_orbital`'ın hatasını YUTMAMALI.

    H2/STO-3G'de yalnızca iki baz fonksiyonu, dolayısıyla iki orbital var;
    99 geçersizdir. Reddedilen seçim pencerenin durumunu da bozmamalı.
    """
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    pencere = OrbitalWindow(solve(molecule, "sto-3g"), molecule)
    try:
        with pytest.raises(InvalidInputError, match="orbital"):
            pencere.set_orbital_index(99)
        with pytest.raises(InvalidInputError, match="orbital"):
            pencere.set_orbital_index(-1)
        assert pencere.orbital_index == 0, "reddedilen seçim durumu bozmamalı"
    finally:
        pencere.close()
