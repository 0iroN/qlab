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
import pathlib
import sys

import numpy as np
import pytest

#: Depo kökü. Senaryo yolları buradan MUTLAK kurulur; planın önerdiği
#: `"scenarios/harmonic_trap.yaml"` göreli yolu pytest'in hangi dizinden
#: çağrıldığına bağlı olurdu.
_DEPO = pathlib.Path(__file__).resolve().parents[1]


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
import pyqtgraph.opengl as gl
from PySide6 import QtWidgets

from qlab.app.orbital_window import OrbitalWindow
from qlab.engine.hf.scf import solve
from qlab.errors import InvalidInputError
from qlab.observable.hf_obs import VolumeGrid
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
    pencere = OrbitalWindow(solve(molecule, "sto-3g"))
    yield pencere, molecule
    pencere.close()


def _cekirdek_itemi(pencere):
    """Görünümdeki tek `GLScatterPlotItem` (çekirdekler)."""
    adaylar = [
        item
        for item in pencere._view.items
        if type(item).__name__ == "GLScatterPlotItem"
    ]
    assert len(adaylar) == 1, f"tam olarak bir çekirdek itemi bekleniyordu: {adaylar}"
    return adaylar[0]


def _pencere_durumu(pencere) -> tuple:
    """Reddedilen bir çağrının BOZMAMASI gereken her şey, tek demette."""
    return (
        pencere._level,
        pencere.mesh_vertex_counts(),
        len(pencere._view.items),
        pencere._kaydirici.value(),
        pencere._panel._text.toPlainText(),
    )


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

    İddianın ÇEKİRDEĞİ bir sıralamadır (yüksek seviye = küçük yüzey) ve o
    ızgaradan bağımsızdır. Ama gövdedeki `== (0, 0)` bir TAM SAYI KİLİDİDİR
    ve bilinçlidir: ızgara varsayılanları (pad=4.0, n=60) bu pencerenin
    kendi sabitleri olduğu için sayı da sabittir. Dokstring önceden
    "tam sayı kilidi değil" diyerek kendini yanlış tarif ediyordu; kilit
    doğru, tarif yanlıştı.
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
    (0 tepe) ve negatif ağ 412 tepe veriyor.

    Bu ölçümden "o orbital her yerde negatif" SONUCU ÇIKMAZ ve daha önce
    burada öyle yazıyordu -- yanlıştı. Yeniden ölçüldü (su/STO-3G orbital 0,
    pad=4.0, n=60): 216000 ızgara noktasının 185036'sı (%85.66) POZİTİF.
    Doğru ifade şudur: pozitif genlik hiçbir yerde +0.05'i AŞMIYOR
    (maks +1.8733e-3, min -4.9041), o yüzden +0.05 izoyüzeyi boş çıkıyor.
    Aynı hesabı başka bir kütüphane ters işaretle verse renkler yer
    değiştirirdi ve HİÇBİR FİZİK değişmezdi.

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
    pencere = OrbitalWindow(solve(molecule, "sto-3g"))
    try:
        with pytest.raises(InvalidInputError, match="orbital"):
            pencere.set_orbital_index(99)
        with pytest.raises(InvalidInputError, match="orbital"):
            pencere.set_orbital_index(-1)
        assert pencere.orbital_index == 0, "reddedilen seçim durumu bozmamalı"
    finally:
        pencere.close()


# ---------------------------------------------------------------------------
# Uyuşmazlık: molekül tek kaynaktan gelir


@requires_offscreen
def test_pencere_ayri_bir_molekul_argumani_kabul_etmiyor(su_penceresi):
    """Uyuşan bir (result, molecule) çifti şart DEĞİL — çift diye bir şey yok.

    ÖLÇÜLDÜ (düzeltmeden önce): `OrbitalWindow(su_sonucu, h2_molekulu)`
    hatasız açılıyordu. Izgara H2 etrafında kuruluyordu (origin
    [-4, -4, -4]), çekirdekler H2'nin çekirdekleriydi, ama çizilen orbital
    SUYUN O 1s'iydi -- ağ (0, 480). Seçici suyun YEDİ orbitalini listeliyor,
    HOMO işareti 0. satırda duruyordu (su/STO-3G'de HOMO 4'tür). Panel bu
    uyuşmazlıktan tek kelime etmiyordu.

    `HFResult` molekülü ZATEN taşıyor (`evaluate_orbital` onu `build_basis`e
    veriyor), yani ikinci argüman fazlalıktı. Doğrulamak yerine KALDIRILDI:
    reddedilen bir uyuşmazlık hâlâ "uyuşma" tanımını (aynı nesne mi? aynı
    geometri mi? aynı yük mü?) doğru yazmayı gerektirir ve o tanım da
    yanlış olabilir. Argüman yoksa uyuşmazlık TEMSİL EDİLEMEZ.
    """
    pencere, molecule = su_penceresi
    with pytest.raises(TypeError):
        OrbitalWindow(pencere._result, molecule)


@requires_offscreen
def test_izgara_cekirdek_ve_secici_hepsi_result_molecule_ile_tutarli(su_penceresi):
    """Pencerenin molekülden okuduğu DÖRT şeyin tek ve aynı kaynağı var.

    Uyuşmazlık hatasının kökü şuydu: `__init__` üç şeyi `molecule`
    parametresinden (ızgara, çekirdek konumları, HOMO indeksi), bir şeyi
    `result.molecule`'den (orbital katsayıları, `evaluate_orbital` içinden)
    alıyordu. Bu test dördünü aynı moleküle bağlar.
    """
    pencere, molecule = su_penceresi
    kaynak = pencere._result.molecule

    beklenen_izgara = VolumeGrid.around(kaynak, padding=4.0, points_per_axis=60)
    assert np.allclose(pencere._grid.origin, beklenen_izgara.origin)
    assert np.allclose(pencere._grid.spacing, beklenen_izgara.spacing)
    assert pencere._grid.shape == beklenen_izgara.shape

    konumlar = np.asarray(_cekirdek_itemi(pencere).pos, dtype=float)
    assert np.allclose(konumlar, [atom.position for atom in kaynak.atoms])

    assert pencere._secici.count() == len(pencere._result.orbital_energies) == 7
    assert pencere.orbital_index == kaynak.n_electrons // 2 - 1 == 4
    # Fixture'daki molekül ile sonucunki aynı geometriyi taşımalı; bu test
    # aksi halde kendi kendini doğrulayan bir totoloji olurdu.
    assert np.allclose(
        [a.position for a in kaynak.atoms], [a.position for a in molecule.atoms]
    )


# ---------------------------------------------------------------------------
# İzoyüzey seviyesi: pozitiflik şartı ve renk-işaret sözleşmesi


@requires_offscreen
@pytest.mark.parametrize(
    ("gecersiz", "gerekce"),
    [
        pytest.param(0.0, "pozitif olmalı", id="sifir"),
        pytest.param(-0.05, "pozitif olmalı", id="negatif"),
        pytest.param(float("nan"), "sonlu bir sayı olmalı", id="nan"),
        pytest.param(float("inf"), "sonlu bir sayı olmalı", id="inf"),
        pytest.param(0.0004, "çözünürlüğünün", id="kaydirici-cozunurlugunun-altinda"),
    ],
)
def test_gecersiz_izoseviye_reddediliyor_ve_durumu_bozmuyor(
    su_penceresi, gecersiz, gerekce
):
    """Pozitif olmayan seviye panelin BEYAN ETTİĞİ renk sözleşmesini yalanlar.

    `viz/isosurface.py` yükü açıkça çağırana yıkıyor: *"level mutlak değer
    olarak yorumlanmaz ... Çağıran, arayüzde ±faz ayrımı istiyorsa pozitif
    bir level vermelidir."* Bu pencere O ÇAĞIRAN'dır.

    ÖLÇÜLDÜ (düzeltmeden önce), su HOMO:
        level=-0.05 -> ağlar (1764, 1764); `_meshes[0].sign == +1` AMA
            `_meshes[0].level == -0.05`. Yani "pozitif" ağ aslında psi=-0.05
            yüzeyi ve KIRMIZI (`_POZITIF_RENK`) çiziliyordu. Panel
            "şu an -0.05" basıp ters çevrilmeden hiç söz etmiyordu.
        level=0.0   -> (3600, 3600): DÜĞÜM YÜZEYİ "orbitalin şekli" diye.
        level=nan   -> ValueError (kaydırıcı senkronundan) firliyordu AMA
            `_yenile` çoktan tamamen koşmuştu: `_level=nan` kalıyor, ağlar
            (0, 0)'a düşüyor, görünümdeki item 3'ten 1'e iniyor (iki yüzey de
            silinmiş), panel "şu an nan" basıyor, kaydırıcı 0.05'te bayat
            kalıyordu.

    O yüzden doğrulama `_yenile`'den ÖNCE yapılır: reddedilen seviye
    pencerenin HİÇBİR alanına dokunmaz. Kardeşi `set_orbital_index` ile aynı
    sözleşme, ama geri alma yoluyla değil, hiç ilerlememekle sağlanır.

    GEREKÇE de sınanır, yalnızca reddedilme değil. Ölçüldü: pozitiflik
    kontrolünü düşüren mutasyon yalnızca "reddedildi mi" diye bakan bir
    testten KAÇIYOR, çünkü 0 ve -0.05 bir alt sınırdaki çözünürlük
    kontrolüne takılıp yine reddediliyor -- ama bu sefer YANLIŞ sebeple.
    Yanlış gerekçe gören kullanıcı yanlış düzeltmeyi dener (0.001'in
    üstüne çıkmayı, oysa sorun işarettir).
    """
    pencere, _ = su_penceresi
    once = _pencere_durumu(pencere)
    with pytest.raises(InvalidInputError, match="izoyüzey seviyesi") as hata:
        pencere.set_isosurface_level(gecersiz)
    assert gerekce in str(hata.value), f"red gerekçesi yanlış: {hata.value}"
    assert _pencere_durumu(pencere) == once


@requires_offscreen
def test_kurucu_da_pozitif_olmayan_izoseviyeyi_reddediyor():
    """Aynı şart kurucuda da geçerli; yoksa pencere doğrudan yalanla açılırdı."""
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")
    for gecersiz, gerekce in (
        (0.0, "pozitif olmalı"),
        (-0.05, "pozitif olmalı"),
        (float("nan"), "sonlu bir sayı olmalı"),
    ):
        with pytest.raises(InvalidInputError, match="izoyüzey seviyesi") as hata:
            OrbitalWindow(result, level=gecersiz)
        assert gerekce in str(hata.value), f"red gerekçesi yanlış: {hata.value}"


@requires_offscreen
def test_ag_renkleri_demet_sirasindan_degil_mesh_isaretinden_geliyor(su_penceresi):
    """`Mesh.sign` sözleşmesi tüketicide DÜŞMEMELİ.

    ÖLÇÜLDÜ (düzeltmeden önce): `_ag_itemlerini_ciz`
    `zip(meshes, (_POZITIF_RENK, _NEGATIF_RENK))` ile rengi DEMET SIRASINA
    göre veriyordu ve `Mesh.sign`'ı hiç okumuyordu -- `viz/isosurface.py`'nin
    özenle taşıdığı işaret bilgisi burada yere düşüyordu.

    Test çifti bilerek TERS sırada verir: renk sırayı değil işareti izliyorsa
    çizilen renkler de ters sırada çıkmalı.
    """
    from qlab.app.orbital_window import _NEGATIF_RENK, _POZITIF_RENK

    pencere, _ = su_penceresi
    try:
        pozitif, negatif = pencere._meshes
        assert pozitif.sign == 1 and negatif.sign == -1
        assert len(pozitif.vertices) > 0 and len(negatif.vertices) > 0

        pencere._ag_itemlerini_ciz((pozitif, negatif))
        assert [it.opts["color"] for it in pencere._mesh_items] == [
            _POZITIF_RENK,
            _NEGATIF_RENK,
        ]

        pencere._ag_itemlerini_ciz((negatif, pozitif))
        assert [it.opts["color"] for it in pencere._mesh_items] == [
            _NEGATIF_RENK,
            _POZITIF_RENK,
        ], "renk demet sırasını değil Mesh.sign'ı izlemeli"

        # Aynı renk iki ağa da verilseydi ±faz ayrımı diye bir şey kalmazdı;
        # alfa 0 olsaydı iki yüzey de görünmezdi.
        assert _POZITIF_RENK[:3] != _NEGATIF_RENK[:3]
        assert _POZITIF_RENK[3] > 0.0 and _NEGATIF_RENK[3] > 0.0
    finally:
        pencere._yenile()


# ---------------------------------------------------------------------------
# Şeffaflık paneli: metin gerçekten PANELE gidiyor mu


@requires_offscreen
def test_panelin_bastigi_metin_transparency_text_ile_ayni(su_penceresi):
    """`transparency_text()` dokstring'i "panelin bastığı metnin aynısı" diyor.

    ÖLÇÜLDÜ: bu söz KORUNMUYORDU. `transparency_text()` metni
    `format_panel(self._provenance())` ile SIFIRDAN üretiyor ve `self._panel`
    widget'ına hiç dokunmuyor; üç "panel beyan ediyor" testi de yalnızca o
    metot üzerinde assert ediyordu. İki mutasyon 9/9 GEÇİYORDU:
        - `_yenile`'den `panel.set_provenance` çağrısını sil
          -> panel metni 0 KARAKTER, `transparency_text()` hâlâ 1174 karakter
        - panele motorun HAM provenance'ını ver (pencerenin eklemeleri yok)
          -> panel 843 karakter, üç gösterim beyanının ÜÇÜ DE panelde yok
    Yani ekranda hiçbir şey yazmasa da takım yemyeşil kalıyordu.
    """
    pencere, _ = su_penceresi
    try:
        metin = pencere._panel._text.toPlainText()
        assert metin, "panel boş"
        assert metin == pencere.transparency_text()
        assert "işareti keyfi" in metin
        assert "çekirdek kabuk" in metin
        assert "şu an 0.05" in metin

        # Seviye değişince PANELDEKİ metin de güncellenmeli, yalnızca
        # `transparency_text()` değil.
        pencere.set_isosurface_level(0.20)
        guncel = pencere._panel._text.toPlainText()
        assert guncel == pencere.transparency_text()
        assert "şu an 0.2" in guncel
    finally:
        pencere.set_isosurface_level(0.05)


@requires_offscreen
def test_provenance_motorun_beyanindan_hicbir_seyi_silmiyor(su_penceresi):
    """`_provenance` dokstring'i "pencere hiçbir şeyi SİLMEZ" diye söz veriyor.

    ÖLÇÜLDÜ: sözün 1/3'ü korunuyordu. Motorun `neglected` listesini (5 madde)
    tamamen düşüren mutasyon panel metnini 1174 -> 931 karaktere indiriyor ve
    9/9 GEÇİYORDU; `diagnostics`'i düşüren mutasyon 1174 -> 1134, yine 9/9.
    Yalnızca `approximations` korunuyordu, o da tek bir "Born-Oppenheimer"
    assert'i sayesinde.

    Bu test metin üzerinden değil, `Provenance` NESNESİ üzerinden bakar:
    motorun taşıdığı her madde ve her tanı, pencerenin ürettiği beyanda
    aynen bulunmak zorunda.
    """
    pencere, _ = su_penceresi
    temel = pencere._result.provenance
    beyan = pencere._provenance()

    # Motorun listeleri boşalırsa bu test totolojiye dönerdi; kaynak
    # gerçekten dolu olmalı.
    assert len(temel.approximations) == 4
    assert len(temel.neglected) == 5
    assert temel.diagnostics

    assert set(temel.approximations) <= set(beyan.approximations)
    assert set(temel.neglected) <= set(beyan.neglected)
    for anahtar, deger in temel.diagnostics.items():
        assert beyan.diagnostics[anahtar] == deger
    assert beyan.method == temel.method
    assert beyan.expected_error == temel.expected_error

    # Pencerenin KENDİ eklediği üç tanı; üçü birden düşürülünce de takım
    # yeşil kalıyordu.
    assert beyan.diagnostics["orbital_index"] == pencere.orbital_index
    assert beyan.diagnostics["isosurface_level"] == pencere._level
    assert beyan.diagnostics["grid_points_per_axis"] == pencere._grid.shape[0] == 60


@requires_offscreen
def test_gosterim_yaklasimlari_motorunkilerle_karismiyor(su_penceresi):
    """Modül dokstring'i motorun taşıdıklarını doğru saymak zorunda.

    Dokstring "motorunkiler (Born-Oppenheimer, sonlu baz, ızgara aralığı)"
    diyordu; motorun `approximations` tuple'ı tam olarak dört madde taşıyor
    ve "ızgara aralığı" ARALARINDA YOK -- o bu pencerenin kendi tanısı
    (`grid_points_per_axis`). Yanlış atıf, panelin hangi katmanın neyi
    beyan ettiği ayrımını bulanıklaştırır.
    """
    pencere, _ = su_penceresi
    assert pencere._result.provenance.approximations == (
        "Born-Oppenheimer",
        "tek Slater determinanti",
        "sonlu baz seti",
        "kapali kabuk (cift isgal)",
    )
    eklenen = set(pencere._provenance().approximations) - set(
        pencere._result.provenance.approximations
    )
    assert len(eklenen) == 2, eklenen


# ---------------------------------------------------------------------------
# Arayüz: görevin GUI şartları


@requires_offscreen
def test_panel_kapatilamaz_ve_gorunum_dondurulebilir(su_penceresi):
    """Görevin iki açık şartı: panel kapatılamaz, görünüm döndürülebilir.

    ÖLÇÜLDÜ: ikisinin de hiçbir koruması yoktu. `self._panel.setVisible(False)`
    mutasyonu 9/9 GEÇİYOR (panel ekranda yok ama testler mutlu), `GLViewWidget`
    layout'a hiç eklenmeyince de 9/9 geçiyordu -- döndürülecek bir şey
    kalmadığı hâlde.

    Döndürme `GLViewWidget`'in kendi fare olayından gelir; test edilebilir
    olan, o widget'ın gerçekten kurulup pencereye BAĞLANMASIDIR.
    """
    pencere, _ = su_penceresi
    assert not pencere._panel.isHidden(), "şeffaflık paneli gizlenmiş"
    assert pencere._panel.parent() is not None, "panel layout'a hiç eklenmemiş"

    assert isinstance(pencere._view, gl.GLViewWidget)
    assert not pencere._view.isHidden()
    assert pencere._view.parent() is not None, "3B görünüm layout'a hiç eklenmemiş"

    merkez = pencere.centralWidget()
    assert merkez is not None
    assert pencere._panel.isAncestorOf(pencere._panel._text)
    assert merkez.isAncestorOf(pencere._panel)
    assert merkez.isAncestorOf(pencere._view)


@requires_offscreen
def test_secici_ve_kaydirici_kullanicida_olu_degil(su_penceresi):
    """Denetimler gerçekten pencereye bağlı olmalı.

    ÖLÇÜLDÜ: `currentIndexChanged` bağlantısını kaldırmak da `valueChanged`
    bağlantısını kaldırmak da 9/9 GEÇİYORDU. Yani seçici ve kaydırıcı
    kullanıcının elinde tamamen ÖLÜ olabilirdi ve testler bunu görmezdi;
    her iki testi de `set_orbital_index`/`set_isosurface_level`'ı DOĞRUDAN
    çağırıyordu.

    Burada sinyal gerçekten yayılır (widget üzerinden), pencere durumunun
    onu izlemesi beklenir.
    """
    pencere, _ = su_penceresi
    homo = pencere.orbital_index
    try:
        pencere._secici.setCurrentIndex(0)
        assert pencere.orbital_index == 0, "seçici pencereye bağlı değil"
        assert pencere.mesh_vertex_counts() == (0, 412)

        pencere._kaydirici.setValue(200)
        assert pencere._level == pytest.approx(0.20), "kaydırıcı pencereye bağlı değil"
    finally:
        pencere.set_orbital_index(homo)
        pencere.set_isosurface_level(0.05)


@requires_offscreen
def test_secici_butun_orbitalleri_dogru_etiketle_listeliyor(su_penceresi):
    """Seçicinin doldurulması ve HOMO/LUMO etiketlerinin YERİ.

    ÖLÇÜLDÜ: seçiciyi hiç doldurmamak da HOMO/LUMO etiketlerini TERS
    çevirmek de 9/9 GEÇİYORDU. Su/STO-3G'de yedi orbital var, beşi dolu:
    HOMO indeksi 4, LUMO 5.
    """
    pencere, _ = su_penceresi
    etiketler = [pencere._secici.itemText(i) for i in range(pencere._secici.count())]
    assert len(etiketler) == 7
    assert "(HOMO)" in etiketler[4]
    assert "(LUMO)" in etiketler[5]
    assert [e for e in etiketler if "(HOMO)" in e] == [etiketler[4]]
    assert [e for e in etiketler if "(LUMO)" in e] == [etiketler[5]]
    assert all("(dolu)" in etiketler[i] for i in range(4))
    assert "(boş)" in etiketler[6]
    # Enerjiler etiketin içinde; sıra sonucun kendi sırasıdır.
    for i, enerji in enumerate(pencere._result.orbital_energies):
        assert f"{enerji:.6f}" in etiketler[i]
    assert pencere._secici.currentIndex() == pencere.orbital_index


@requires_offscreen
def test_kaydirici_seviyeyle_senkron_kaliyor(su_penceresi):
    """Kaydırıcı ile panel aynı sayıyı göstermek zorunda.

    ÖLÇÜLDÜ (düzeltmeden önce): `set_isosurface_level(2.0)` `_level`'ı 2.0
    yapıyor ama kaydırıcı 900'e (0.9) KIRPILIYORDU -- panel "şu an 2" derken
    kaydırıcı 0.9 gösteriyordu, sessizce. 2.0 uydurma bir değer de değil:
    su/STO-3G O 1s'in ızgara üzerindeki maksimum genliği 4.9041.

    Kırpma yerine aralık genişletilir; kaydırıcının taşıdığı tam sayı
    `round(level * 1000)` olmak zorunda.

    Açılış ARALIĞI burada değil, kendi taze penceresini kuran kardeş testte
    sınanır; sebebi aşağıda.
    """
    pencere, _ = su_penceresi
    try:
        assert pencere._kaydirici.value() == 50
        pencere.set_isosurface_level(2.0)
        assert pencere._kaydirici.value() == 2000
        assert pencere._kaydirici.maximum() >= 2000
        assert "şu an 2" in pencere.transparency_text()
        assert pencere._level == 2.0
    finally:
        pencere.set_isosurface_level(0.05)
    assert pencere._kaydirici.value() == 50


@requires_offscreen
def test_kaydiricinin_acilis_araligi_surukleyebilir_bir_bant():
    """Kaydırıcının AÇILIŞ aralığı; bilerek TAZE bir pencerede.

    Değer doğru olduğu hâlde aralık işe yaramaz olabilir: açılış üst sınırını
    900'den 1'e düşüren mutasyonda `_level` yine 0.05, kaydırıcı değeri yine
    50 çıkıyor (senkron sınırı 50'ye büyütüyor) ama kullanıcı kaydırıcıyı
    ancak 0.001 ile 0.05 arasında sürükleyebiliyor.

    Bu test neden paylaşılan `su_penceresi` fixture'ını KULLANMIYOR: ölçüldü,
    aynı iddia orada YAZILIYKEN o mutasyon hâlâ hayatta kalıyordu. Sebep
    fixture'ın modül kapsamlı olması -- daha erken koşan
    `test_izoseviye_degisince_ag_kuculuyor` seviyeyi 0.90'a çıkarıyor, bu da
    üst sınırı 900'e BÜYÜTÜP mutasyonu onarıyor. Aynı mutasyon tek başına
    (`-k`) koşturulduğunda iki testi birden kırıyordu. Paylaşılan durum
    üzerinde açılış iddiası sınamak, iddiayı sessizce sırayla ilişkilendirir.

    İKİNCİ TUZAK, ilkini kapatan yamada ortaya çıktı ve ölçülerek düzeltildi:
    sınırlar bir süre `_KAYDIRICI_VARSAYILAN_MAKS`'ın KENDİSİNE karşı
    yazılıydı (`maximum() >= _KAYDIRICI_VARSAYILAN_MAKS`). Böyle bir assert
    sınırladığı sabitle birlikte HAREKET EDER: sabit 900'den 1'e düşünce iddia
    `maximum() >= 1` olur ve bedavaya geçer -- ölçüldü, o hâliyle mutasyon
    dosyanın 34 testinin SIFIRINI kırıyordu. Alt sınır daha da kötüydü:
    `_EN_KUCUK_SEVIYE` zaten `_KAYDIRICI_MIN / _KAYDIRICI_OLCEK` olduğu için
    o assert bir TOTOLOJİYDİ, hiçbir değişiklikte kırılamazdı.

    Bu yüzden bant LİTERAL sayılarla yazılıyor. 0.01-0.50 keyfi değil: bu
    pencerenin kendi ölçümünde su HOMO'su 0.02'de 2850, 0.10'da 1112, 0.50'de
    114 tepe veriyor -- yani kullanıcının gerçekten gezmesi gereken aralık bu.
    """
    from qlab.app.orbital_window import _KAYDIRICI_OLCEK, _dogrula_seviye

    pencere = OrbitalWindow(solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g"))
    try:
        assert pencere._kaydirici.value() == 50
        en_kucuk = pencere._kaydirici.minimum() / _KAYDIRICI_OLCEK
        en_buyuk = pencere._kaydirici.maximum() / _KAYDIRICI_OLCEK
        assert en_kucuk <= 0.01, "kullanıcı ince yüzeylere inebilmeli"
        assert en_buyuk >= 0.50, "kullanıcı kaba yüzeylere çıkabilmeli"
        # Kaydırıcının TABANI doğrulayıcının reddedeceği bir seviye üretmemeli.
        # Bu bir totoloji DEĞİL: iki eşiği ayrı ayrı taşıyan iki kod yolunu
        # karşılaştırır, biri kayarsa kırılır.
        _dogrula_seviye(en_kucuk)
    finally:
        pencere.close()


@requires_offscreen
def test_orbital_degistikce_ag_itemleri_sizmiyor(su_penceresi):
    """Eski ağ item'ları görünümden SİLİNMELİ.

    ÖLÇÜLDÜ: silme döngüsünü kaldıran mutasyon 9/9 GEÇİYORDU; 13 orbital
    değişiminde görünümdeki item sayısı 3'ten 27'ye çıkıyor. Sızıntı sessiz
    değil, GÖRSEL: eski yüzeyler ekranda üst üste durur ve gösterilen
    "orbital" artık tek bir orbital olmaz.

    Değişmez: görünümde her zaman 1 çekirdek item'ı + BOŞ OLMAYAN ağ sayısı
    kadar mesh item'ı bulunur. Boş ağ (su/STO-3G orbital 0'ın pozitif fazı)
    hiç çizilmez.
    """
    pencere, _ = su_penceresi
    homo = pencere.orbital_index
    try:
        for _tur in range(2):
            for i in range(7):
                pencere.set_orbital_index(i)
                dolu = sum(1 for sayi in pencere.mesh_vertex_counts() if sayi > 0)
                assert len(pencere._mesh_items) == dolu
                assert len(pencere._view.items) == 1 + dolu
        assert len(pencere._view.items) <= 3
        # Orbital 0'da pozitif ağ BOŞ (0 tepe): tek item çizilir.
        pencere.set_orbital_index(0)
        assert pencere.mesh_vertex_counts() == (0, 412)
        assert len(pencere._view.items) == 2
    finally:
        pencere.set_orbital_index(homo)


# ---------------------------------------------------------------------------
# CLI bağlantısı (Task 10)


@pytest.fixture
def acilan_pencereler(monkeypatch):
    """`_pencereyi_goster` yamalanır; gerçek olay döngüsü testte koşamaz.

    Yamalanan işlev çıkış kodunu DÖNDÜRÜR (gerçeği `app.exec()`'i döndürür),
    yoksa `main()` `None` döndürür ve `kod == 0` iddiası anlamsızlaşırdı.
    """
    from qlab.app import main as app_main

    acilanlar: list = []

    def sahte(pencere):
        acilanlar.append(pencere)
        return 0

    monkeypatch.setattr(app_main, "_pencereyi_goster", sahte)
    return app_main, acilanlar


@requires_offscreen
def test_main_molekul_argumaniyla_orbital_penceresi_aciyor(acilan_pencereler):
    """`--molecule` ikinci bir giriş yolu açar; SCF koşar ve pencere gelir.

    H2/STO-3G'de iki orbital var, HOMO indeksi 0.
    """
    app_main, acilanlar = acilan_pencereler
    kod = app_main.main(["--molecule", "H 0 0 0; H 0 0 0.74", "--basis", "sto-3g"])
    assert kod == 0
    assert len(acilanlar) == 1
    assert isinstance(acilanlar[0], OrbitalWindow)
    assert acilanlar[0].orbital_index == 0


@requires_offscreen
def test_main_tdse_senaryosunu_bozmadan_calistiriyor(acilan_pencereler):
    """Geriye uyumluluk: mevcut TDSE yolu aynen çalışmalı.

    Planın önerdiği `harmonic_trap.yaml` DEĞİL `tunneling.yaml` kullanılıyor:
    ilki bir sonraki testin konusu olan hatayı tetikliyordu, ve bu testin işi
    TDSE yolunun BOZULMADIĞINI göstermek. Ölçüldü: tunneling.yaml ile pencere
    2.02 s'de kuruluyor (n_steps=32000), iletim 0.1478, norm 1.0000000000.
    """
    app_main, acilanlar = acilan_pencereler
    kod = app_main.main([str(_DEPO / "scenarios" / "tunneling.yaml")])
    assert kod == 0
    assert len(acilanlar) == 1
    assert type(acilanlar[0]).__name__ == "TDSEWindow"


@requires_offscreen
def test_main_bariyersiz_senaryoda_cokmuyor(acilan_pencereler):
    """ÖNCEDEN VAR OLAN HATA: `harmonic_trap.yaml` pencereyi çökertiyordu.

    `TDSEWindow._recompute` kaydırıcının değerini KOŞULSUZ olarak
    `params["height"]` diye enjekte ediyordu; `height` parametresini yalnızca
    `barrier` potansiyeli kabul ediyor. Ölçülen çökme:

        ScenarioError: 'harmonic' potansiyeli için parametreler hatalı:
        harmonic() got an unexpected keyword argument 'height'

    Yani `python -m qlab.app.main scenarios/harmonic_trap.yaml` bugüne kadar
    hiç çalışmadı. Hata Task 10'un getirdiği bir şey değil; bu test onu
    kilitliyor.

    Sessiz varyantı daha kötüydü: `potentials.double_slit` de `height` ALIYOR
    (varsayılan 10000.0 hartree, duvarı geçirimsiz yapan değer). Bir 2B
    çift-yarık senaryosu açılsaydı kaydırıcı duvar yüksekliğini sessizce
    0.80'e düşürür ve yarıklar anlamını yitirirdi -- çökme bile olmazdı.
    """
    app_main, acilanlar = acilan_pencereler
    kod = app_main.main([str(_DEPO / "scenarios" / "harmonic_trap.yaml")])
    assert kod == 0
    assert len(acilanlar) == 1
    pencere = acilanlar[0]
    assert type(pencere).__name__ == "TDSEWindow"
    assert not pencere._slider.isEnabled(), (
        "bariyeri olmayan bir potansiyelde 'Bariyer yüksekliği' kaydırıcısı "
        "etkin kalmamalı -- döndürecek bir şey yok"
    )


@requires_offscreen
def test_main_ne_senaryo_ne_molekul_verilmezse_hata(acilan_pencereler, capsys):
    """Argümansız çağrı sessizce bir şey açmamalı."""
    app_main, acilanlar = acilan_pencereler
    kod = app_main.main([])
    assert kod == 1
    assert acilanlar == []
    assert "molecule" in capsys.readouterr().err


@requires_offscreen
def test_main_hem_senaryo_hem_molekul_verilirse_hata(acilan_pencereler, capsys):
    """İki giriş yolu birbirini dışlar; hangisinin kazandığı tahmin edilmemeli."""
    app_main, acilanlar = acilan_pencereler
    kod = app_main.main(
        [str(_DEPO / "scenarios" / "tunneling.yaml"), "--molecule", "H 0 0 0; H 0 0 0.74"]
    )
    assert kod == 1
    assert acilanlar == []
    assert "molecule" in capsys.readouterr().err


#: Alt süreç sürücüsü. `_pencereyi_goster` yamalanır (olay döngüsü hiç
#: dönmezdi), gerisi gerçek `main()` yoludur.
_ALT_SUREC = """
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from qlab.app import main as m
m._pencereyi_goster = lambda pencere: 0
sys.exit(m.main(sys.argv[1:]))
"""


@pytest.mark.parametrize(
    "arguman",
    [
        pytest.param(["--molecule", "H 0 0 0; H 0 0 0.74"], id="orbital"),
        pytest.param([str(_DEPO / "scenarios" / "harmonic_trap.yaml")], id="tdse"),
    ],
)
def test_main_taze_surecte_qapplication_olmadan_cokmuyor(arguman):
    """CLI, süreçte HİÇBİR QApplication yokken de çalışmalı.

    Bu testin ALT SÜREÇTE koşmasının sebebi, testin kendisidir: bu depodaki
    her Qt test modülü içe aktarılırken modül düzeyinde bir QApplication
    kuruyor. Yani aynı süreç içinde koşan hiçbir test, "QApplication yokken
    ne olur" sorusunu SORAMAZ -- ortam hatayı maskeler.

    Maskelenen hata ölçüldü ve sessiz değil, ÖLÜMCÜL: QApplication yokken bir
    QWidget kurmak Qt'nin `qFatal`'ını tetikliyor, süreç **çıkış kodu 127** ile
    ölüyor; Python istisnası da yok, traceback de yok. `main()` pencereyi
    kurup SONRA uygulamayı kursaydı `--molecule` yolu her gerçek çağrıda
    böyle ölürdü ve takım yemyeşil kalırdı.
    """
    import subprocess

    sonuc = subprocess.run(
        [sys.executable, "-c", _ALT_SUREC, *arguman],
        cwd=str(_DEPO),
        env={**os.environ, "PYTHONPATH": str(_DEPO / "src")},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert sonuc.returncode == 0, (
        f"CLI taze süreçte çöktü (kod {sonuc.returncode}).\n"
        f"stdout: {sonuc.stdout}\nstderr: {sonuc.stderr}"
    )
