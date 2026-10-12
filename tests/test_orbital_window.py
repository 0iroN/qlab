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
    `tests/conftest.py`'deki referans fixture'larında aynı gerekçeyle
    `pytest.skip` yerine `pytest.fail` kullanır — sessizce atlanan bir
    doğrulama, doğrulama değildir.
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
from qlab.viz.isosurface import EXTRACTION_APPROXIMATIONS

try:
    _qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _OFFSCREEN_OK = True
except Exception:  # pragma: no cover - bu ortamda offscreen platformu kurulamazsa
    _OFFSCREEN_OK = False

requires_offscreen = pytest.mark.skipif(
    not _OFFSCREEN_OK, reason="Qt ekransız (offscreen) platformu bu ortamda kurulamadı"
)

#: `reference/hf_energies.json`'daki `h2o.atom` ile aynı geometri. Doğrudan
#: yazıldı: `hf_energies` fixture'ı artık kök `tests/conftest.py`'de tanımlı
#: (yol sırasına bağlı kaybolma yüzünden oraya taşındı) ve buradan da
#: erişilebilir olurdu, ama geometriyi sabit tutmak bu modülü referans
#: dosyasının varlığından bağımsız kılıyor.
_SU = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


@pytest.fixture(scope="module")
def su_sonucu():
    """Su/STO-3G'nin SCF sonucu; TAZE pencere kuran testler bunu paylaşır.

    Paylaşılan olan SONUÇ, PENCERE DEĞİL. Açılış durumu iddiaları paylaşılan
    bir pencerede sınanamaz (bkz. `test_kaydiricinin_acilis_araligi_...`: modül
    kapsamlı fixture bir açılış mutasyonunu ONARIYORDU), ama SCF'i her test
    için yeniden koşmak da gereksiz -- `HFResult` dondurulmuş bir sonuçtur ve
    pencere onu yalnızca OKUR.
    """
    return solve(Molecule.from_string(_SU), "sto-3g")


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
    kendi `Provenance.approximations` listesinden geliyor (scf.py:129).
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
def test_IHMAL_satiri_cekirdege_yakin_yuzeylerin_NITEL_oldugunu_basiyor(su_penceresi):
    """Beyanın YERİ ve TAMAMI kilitli; dayandığı sayı pencerenin KENDİ ızgarasında.

    Yukarıdaki test yalnız "çekirdek kabuk" sözcüğünü panelin HERHANGİ bir
    yerinde arıyor. Ölçüldü (11.10.2026, mutasyon): `_GOSTERIM_IHMALLERI`den
    "çekirdeğe yakın yüzeyler nitel" yarısını silen mutant da, beyanı
    "İhmal edilen"den "Yaklaşımlar" satırına taşıyan mutant da o testten
    geçiyordu. Burada "İhmal edilen" satırı ADIYLA bulunur ve cümle TAM aranır.

    SAYI (ölçüldü, 11.10.2026): pencerenin varsayılan ızgarasında (padding
    4.0, n = 60) su/STO-3G çekirdek 1s'nin ızgara normu 0.948879 (%5.1 eksik),
    HOMO'nunki 0.999998 -- "ızgara çözmüyor" yargısının BU ızgaradaki kanıtı.
    README'nin 0.98760196'sı başka bir ızgaranındır (padding 6.0, n = 120;
    `tests/hf/test_hf_obs.py` kaba/ince ikilisini kilitler). Pencere ızgarası
    inceltilirse (n = 120: 0.999373) bu test kırmızıya döner ve beyan o zaman
    yeniden düşünülür. MALİYET: ~0.12 s (ölçüldü; iki orbital 60³ noktada).
    KIRMA (ölçüldü, 11.10.2026, `mut.py`): nitel yarısını silen, beyanı
    "Yaklaşımlar"a taşıyan ve pencere ızgarasını n = 120 yapan üç mutantın
    üçü de burada düşer; yukarıdaki "çekirdek kabuk" testi üçünden de geçer.
    """
    from qlab.observable.hf_obs import evaluate_orbital

    pencere, _ = su_penceresi
    satirlar = [s for s in pencere.transparency_text().splitlines() if s.startswith("İhmal edilen")]
    assert len(satirlar) == 1, satirlar
    assert (
        "çekirdek kabuklarının uç genliği: ızgara çözmüyor, çekirdeğe yakın yüzeyler nitel"
        in satirlar[0]
    )
    izgara, sonuc = pencere._grid, pencere._result

    def norm(indeks: int) -> float:
        return float((evaluate_orbital(sonuc, indeks, izgara) ** 2).sum() * izgara.volume_element)

    assert norm(0) == pytest.approx(0.948879, abs=1e-5, rel=0.0)  # çekirdek O 1s
    assert norm(4) == pytest.approx(0.999998, abs=1e-5, rel=0.0)  # HOMO


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

    EŞİTLİK SATIRI KALDIRILDI, çünkü TOTOLOJİYDİ: `transparency_text()`
    (orbital_window.py) `self._panel.text()` döndürüyor, o da (transparency.py)
    `self._text.toPlainText()` -- yani `metin == pencere.transparency_text()`
    iddiasının iki tarafı BİREBİR aynı ifadeye indirgeniyordu ve satır
    orbital_window.py'ye uygulanan hiçbir mutasyonla kırılamıyordu. Testin
    gücü zaten diğer assert'lerden geliyordu; okuyucuyu korumanın o satırdan
    geldiğine inandırıyordu. Yerine YAPISAL bir sonda kondu.
    """
    pencere, _ = su_penceresi
    try:
        metin = pencere._panel._text.toPlainText()
        assert metin, "panel boş"
        assert "işareti keyfi" in metin
        assert "çekirdek kabuk" in metin
        assert "şu an 0.05" in metin

        # YAPISAL SONDA: panele yabancı bir metin yazılır. `transparency_text()`
        # paneli OKUYORSA onu döndürür; metni `format_panel(self._provenance())`
        # ile ikinci kez ÜRETİYORSA sondayı hiç görmez. Bu satır, eşitlik
        # satırının aksine, gerçekten kırılabilir.
        pencere._panel._text.setPlainText("SONDA-42")
        assert pencere.transparency_text() == "SONDA-42", (
            "transparency_text() paneli okumuyor, metni yeniden üretiyor"
        )
        pencere._yenile()
        assert pencere.transparency_text() != "SONDA-42", (
            "_yenile paneli tazelemeli; sonda metni ekranda kalmamalı"
        )

        # Seviye değişince PANELDEKİ metin de güncellenmeli, yalnızca
        # `transparency_text()` değil.
        pencere.set_isosurface_level(0.20)
        assert "şu an 0.2" in pencere._panel._text.toPlainText()
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
    # YEDİ madde, ÜÇ ayrı katmandan:
    #   2 -- pencerenin kendi gösterim seçimleri (renk/işaretin ne demek
    #        olduğu, izoyüzey seviyesi)
    #   2 -- ağ çıkarımının seçimleri (float32 dökümü, marching-cubes doğrusal
    #        interpolasyonu). İkisi de `viz/isosurface.py`'de, yani seçimin
    #        YAPILDIĞI yerde yazılı; `Mesh` kendi Provenance'ını taşımadığı
    #        için ekrana ancak buradan ulaşıyorlar.
    #   3 -- `analyse()`in beyanı (Mulliken/Löwdin baz duyarlılığı, iyonik
    #        karakterin dipolden okunması, Koopmans). Sayı 4'ten 7'ye ÇIKTI
    #        çünkü pencere artık `analyse()`in sayılarını da basıyor; bir
    #        sayıyı ekrana koyup onun beyanını koymamak bu depoda kusurdur.
    assert len(eklenen) == 7, eklenen
    assert set(EXTRACTION_APPROXIMATIONS) <= eklenen
    metin = pencere.transparency_text().lower()
    assert "float32" in metin, "tek duyarlıklı döküm panelde beyan edilmiyor"
    assert "marching" in metin, "marching-cubes interpolasyonu panelde yok"


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
# Ekransız ama OKUNABİLİR durum
#
# Modül dokstring'i "OpenGL boyaması test edilemez" diyor ve HAKLI: bağlam ancak
# boyama sırasında gerekir. Ama o sınır bir süre kendinden fazlasını yuttu --
# panelin yüksekliği, kameranın mesafesi, seçicinin açılış konumu, pencerenin
# gösterilip gösterilmediği EKRANSIZ KİPTE OKUNABİLİR saf durumdur ve hiçbiri
# boyamaya bağlı değildir. Aşağıdaki testler o payı geri alıyor; her biri
# ölçülmüş ve o ölçümde 0 test kıran bir mutasyona karşılık geliyor.


@requires_offscreen
def test_orbital_paneli_varsayilan_kutuyu_kullanmiyor(su_penceresi):
    """Kabul kriteri 3'ün gerçek ekranda bulduğu TEK kusurun düzeltmesi.

    README: gerçek-ekran doğrulaması "bir kusuru da GÖRÜNÜR KILDI: 12 satırlık
    beyan 140 px'lik panelde katlanın altında kalıyordu (düzeltildi,
    TransparencyPanel(max_height=...))". Bu, deponun tek gerçek-ekran
    doğrulamasının ürettiği tek bulgu ve tam olarak sıfır testle korunuyordu.

    ÖLÇÜLDÜ, iki bağımsız mutasyon, ikisi de 58 testin SIFIRINI kırıyordu:
        orbital_window.py `TransparencyPanel(max_height=260)` -> `TransparencyPanel()`
        transparency.py   `setMaximumHeight(max_height)`      -> `setMaximumHeight(140)`
    Oysa fark ekransız kipte OKUNABİLİR: `_panel._text.maximumHeight()` doğru
    kodda 260, varsayılan panelde 140 döndürüyor; show() sonrası gerçek widget
    yüksekliği 192 px'ten 140 px'e düşüyor.

    Çağrılan taraf `tests/test_transparency.py`'de kilitli; burası ÇAĞIRAN
    taraf. Sayılar LİTERAL: `_VARSAYILAN_YUKSEKLIK`'e karşı yazılan bir iddia
    sabitle birlikte hareket eder ve hiçbir zaman kırılamazdı.

    SINIR 260 -> 340 -> 440 diye İKİ KEZ, ikisi de ölçüyle büyüdü. 260 taşmayı
    yalnızca AZALTIYORDU; 340, panel yüksekliğini metinden hesaplamaya
    başladıktan sonra 960 px genişlikte gereken 332 px'i karşılıyordu. 440 ise
    beyanın kendisi büyüdüğü için gerekti: pencere artık üç alan türünü,
    `analyse()`in üç uyarısını ve fark alanında promolekül referansını
    taşıyor. Ölçüldü (su/STO-3G, 1400 px genişlik): orbital 2484 karakter /
    388 px, elektron yoğunluğu 3294 / 500 px, yoğunluk farkı 4645 / 682 px.
    Sayının İŞE YARADIĞINI ayrı bir test sınıyor:
    `test_orbital_penceresinde_beyanin_TAMAMI_gorunuyor`.
    """
    from qlab.app.transparency import TransparencyPanel

    pencere, _ = su_penceresi
    assert pencere._panel._text.maximumHeight() == 440
    assert TransparencyPanel()._text.maximumHeight() == 140
    assert pencere._panel._text.maximumHeight() > 140, (
        "orbital penceresinin beyanı TDSE penceresinin kutusuna sığmıyor"
    )

    # GEREKÇE de sınanır: bu pencerenin beyanı motorunkinden GERÇEKTEN uzun.
    # Değilse 340 keyfi bir sayı olurdu.
    from qlab.app.transparency import format_panel

    assert len(pencere.transparency_text()) > len(
        format_panel(pencere._result.provenance)
    )


@requires_offscreen
def test_orbital_penceresinde_beyanin_TAMAMI_gorunuyor():
    """ASIL ÖLÇÜT: varsayılan geometride gizli satır KALMAMALI.

    Bir üstteki test kutunun SAYISINI kilitler; bu test o sayının İŞE YARAYIP
    yaramadığını sınar. İkisi ayrı sorular -- 260 da varsayılandan büyüktü ama
    yetmiyordu, ve o farkı hiçbir test görmüyordu.

    ÖLÇÜLDÜ (1400x950, pencerenin kendi varsayılan boyutu, max_height=440):
        orbital alanı -> gereken 388 px, 0 satır gizli, 3B görünüm 565 px

    KAPSAM, açıkça: iddia AÇILIŞ ALANI (orbital) içindir. Öteki iki alanda
    beyan daha uzun (elektron yoğunluğu 500 px, yoğunluk farkı 682 px) ve
    kırpma KAÇINILMAZ -- 682 px'i sığdırmak 3B görünüme 131 px bırakırdı.
    O durumda kırpmanın DUYURULDUĞUNU kardeş test sınıyor:
    `test_yogunluk_ve_fark_alanlarinda_kirpma_DUYURULUYOR`.

    Taze pencere kullanılıyor, paylaşılan `su_penceresi` fixture'ı DEĞİL: bu bir
    AÇILIŞ durumu iddiası ve bu dosyada paylaşılan fixture'ın bir açılış
    mutasyonunu ONARDIĞI ölçülmüştü (bkz.
    `test_kaydiricinin_acilis_araligi_surukleyebilir_bir_bant`).
    """
    molecule = Molecule.from_string(_SU)
    pencere = OrbitalWindow(solve(molecule, "sto-3g"))
    try:
        pencere.show()
        _qapp.processEvents()
        _qapp.processEvents()
        metin = pencere._panel._text
        assert metin.verticalScrollBar().maximum() == 0, (
            "varsayılan geometride beyanın tamamı görünmeli; gizli satır: "
            f"{metin.verticalScrollBar().maximum()}"
        )
        assert "kaydırın" not in pencere._panel.title().lower(), (
            f"kırpma yokken uyarı da olmamalı: {pencere._panel.title()!r}"
        )
        assert pencere._view.height() > 300, (
            f"panel 3B görünümü ezmemeli, kalan: {pencere._view.height()} px"
        )
    finally:
        pencere.close()


@requires_offscreen
def test_yogunluk_ve_fark_alanlarinda_kirpma_DUYURULUYOR(su_sonucu):
    """Kırpma kabul edildi, ama SESSİZ değil.

    Bir üstteki test açılış alanının (orbital) tam göründüğünü kilitler; bu
    test onun BEDELİNİ kilitler. ÖLÇÜLDÜ (1400x950, max_height=440):
        elektron yoğunluğu -> gereken 500 px,  4 satır gizli
        yoğunluk farkı     -> gereken 682 px, 17 satır gizli
    "Okunmayan bir beyan, beyan etmiş sayılmaz" ölçütü, gizlenenin VARLIĞINI
    duyurmayı da kapsar; metnin kendisi (`transparency_text()`) kırpılmaz ve
    3B görünüm ezilmez.

    BU ORTAMA ÖZGÜ, açıkça yazılıyor: yukarıdaki sayılar EKRANSIZ (offscreen)
    platformun font metriklerinde ölçüldü. GERÇEK ekranda aynı pencere daha
    dar sarıyor ve üçü de sığıyor -- ölçüldü (2560x1440, devicePixelRatio 1.50,
    Windows 11, panel genişliği 1344 px): orbital 250 px, elektron yoğunluğu
    306 px, yoğunluk farkı 390 px, üçü de 440'ın altında ve kırpma uyarısı
    hiç çıkmıyor. Yani 440 sınırı bu makinede DEVREDE DEĞİL; asıl işi dar
    pencerelerde ve daha büyük yazı tiplerinde görüyor. Test yine de değerli:
    kırpma yolunun kendisi ancak burada koşuyor.
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_YOGUNLUK

    for ad in (ALAN_YOGUNLUK, ALAN_FARK):
        pencere = OrbitalWindow(su_sonucu, field=ad)
        try:
            pencere.show()
            _qapp.processEvents()
            _qapp.processEvents()
            metin = pencere._panel._text
            assert metin.verticalScrollBar().maximum() > 0, (
                f"{ad!r} beyanı sığıyorsa bu test artık bir şey ölçmüyor"
            )
            assert metin.height() == 440, (
                f"{ad!r}: kırpılırken panel üst sınıra DAYANMALI, "
                f"{metin.height()} px'te kalmış"
            )
            assert "kaydırın" in pencere._panel.title().lower(), (
                f"{ad!r} alanında sessizce kırpılıyor: {pencere._panel.title()!r}"
            )
            assert pencere._view.height() > 300, (
                f"{ad!r}: 3B görünüm ezilmiş ({pencere._view.height()} px)"
            )
            # Metnin KENDİSİ kırpılmaz; yalnızca görünen kısım kısalır.
            assert len(pencere.transparency_text()) > 3000, ad
        finally:
            pencere.close()


@requires_offscreen
def test_dar_pencerede_kirpma_GERCEKTEN_duyuruluyor():
    """Kırpma kaçınılmaz olduğunda uyarının yazıldığını `resize`+`show` sırasında sınar.

    Bu sıra önemli: `resize()` çağrılıp SONRA `show()` çağrıldığında Qt'nin
    yerleşimi `showEvent`'ten de SONRA oturuyor. ÖLÇÜLDÜ -- panelin gecikmeli
    (sıfır gecikmeli tek atış) yeniden hesabı olmadan 800x700'lük pencerede:
        panel 304 px'te kalıyor (olması gereken 340 = üst sınır)
        kaydırma çubuğu 6 satır gizliyor
        başlıkta uyarı HİÇ yazılmıyor -- yani sessizce kırpıyor
    O gecikmeli hesabı kaldıran mutasyon, bu test yazılmadan önce 64 testin
    SIFIRINI kırıyordu.

    800x700 seçildi çünkü ölçüldü: orbital alanının beyanı orada 528 px
    istiyor, üst sınır 440 -- yani kırpma gerçek ve kaçınılmaz (6 satır
    gizli). Bu, pencerenin VARSAYILAN boyutundan (1400x950) bilinçle çok daha
    dar bir kutu; varsayılanda orbital beyanı tam sığıyor.
    """
    molecule = Molecule.from_string(_SU)
    pencere = OrbitalWindow(solve(molecule, "sto-3g"))
    try:
        pencere.resize(800, 700)
        pencere.show()
        _qapp.processEvents()
        metin = pencere._panel._text
        assert metin.verticalScrollBar().maximum() > 0, (
            "bu genişlikte beyan sığmamalı; sığıyorsa test artık bir şey ölçmüyor"
        )
        assert metin.height() == 440, (
            f"kırpılırken panel üst sınıra DAYANMALI, yoksa boşuna yer bırakır: "
            f"{metin.height()} px"
        )
        assert "kaydırın" in pencere._panel.title().lower(), (
            f"sessizce kırpma YOK: {pencere._panel.title()!r}"
        )
    finally:
        pencere.close()


@requires_offscreen
def test_kamera_sahneyi_kadraja_aliyor(su_penceresi):
    """Kamera mesafesi çarpanı: sahne kadraj dışına düşse takım yeşil kalıyordu.

    ÖLÇÜLDÜ: `setCameraPosition(distance=... * 60.0)` -> `* 1.0` mutasyonu 58
    testin SIFIRINI kırıyordu. Eşdeğer mutant DEĞİL: `_view.opts['distance']`
    ekransız kipte okunuyor (su/STO-3G varsayılanlarında 11.045131, H2'de
    9.557692) ve 1x'te izoyüzey tamamen kameranın içinde kalır.

    İki BAĞIMSIZ sınır yazılıyor, ikisi de kameranın kendi sabitinden değil,
    BAŞKA nesnelerden türüyor:

    1. Izgara kutusunun en uzun kenarı. pyqtgraph'ın varsayılan görüş açısı
       60 derece; odak düzleminde görünen dikey açıklık 2*d*tan(30°) = 1.155*d,
       yani kutunun sığması için d >= 0.866 * kenar. Ölçülen oran 1.016949
       (hem suda hem H2'de; çünkü mesafe 60*spacing, kenar 59*spacing).
       Üst sınır 2.0: 60 -> 120 mutasyonu oranı 2.034 yapar ve yakalanır.
    2. Çizilen ağın kendi yarıçapı. Kamera yüzeyin İÇİNDE kalmamalı; ölçüldü,
       su HOMO'sunun (level=0.05) maks|r| değeri 2.372238 bohr.
    """
    pencere, _ = su_penceresi
    mesafe = float(pencere._view.opts["distance"])

    kenarlar = np.asarray(pencere._grid.spacing) * (np.asarray(pencere._grid.shape) - 1)
    en_uzun = float(kenarlar.max())
    assert mesafe >= 0.87 * en_uzun, (
        f"kamera çok yakın: kutunun {en_uzun:.3f} bohr'luk kenarı kadraja "
        f"sığmaz (mesafe {mesafe:.3f})"
    )
    assert mesafe <= 2.0 * en_uzun, (
        f"kamera çok uzak: molekül kadrajda noktaya döner "
        f"(mesafe {mesafe:.3f}, kenar {en_uzun:.3f})"
    )

    yaricap = max(
        float(np.linalg.norm(np.asarray(mesh.vertices, dtype=float), axis=1).max())
        for mesh in pencere._meshes
        if len(mesh.vertices) > 0
    )
    assert mesafe > yaricap, (
        f"kamera çizilen yüzeyin İÇİNDE ({mesafe:.3f} <= {yaricap:.3f} bohr)"
    )


@requires_offscreen
def test_acilista_secici_gosterilen_orbitali_isaret_ediyor():
    """Açılış durumu, bilerek TAZE bir pencerede -- paylaşılan fixture ONARIYOR.

    BEŞİNCİ MEKANİZMA BURADA NÜKSETMİŞTİ. Kaydırıcının açılış bandı için
    fixture paylaşımı zaten kusur sayılıp taze pencereye taşınmıştı
    (`test_kaydiricinin_acilis_araligi_surukleyebilir_bir_bant`), ama SEÇİCİ
    için aynı şey yapılmamıştı.

    ÖLÇÜLDÜ: `_kur_arayuz`'dan `self._secici.setCurrentIndex(self._orbital_index)`
    satırını silen mutasyon
        - tek başına (`-k`)      -> 1 failed, 33 deselected
        - dosyanın TAMAMINDA     -> 34 passed  (KIRILMA YOK)
    Sebep: modül kapsamlı `su_penceresi` fixture'ında daha erken koşan testler
    `set_orbital_index` çağırıyor, o da `_secici_senkronla`'yı çalıştırıp
    mutasyonu ONARIYOR. Yani açılış iddiası sessizce test SIRASINA bağlanmıştı.

    Su/STO-3G'de yedi orbital var, beşi dolu: HOMO indeksi 4.
    """
    pencere = OrbitalWindow(solve(Molecule.from_string(_SU), "sto-3g"))
    try:
        assert pencere.orbital_index == 4
        assert pencere._secici.currentIndex() == 4, (
            "seçici açılışta gösterilen orbitali işaret etmiyor"
        )
        assert pencere._secici.count() == 7
        assert "(HOMO)" in pencere._secici.currentText()
        # Kaydırıcının açılış DEĞERİ de aynı sınıftan: literal 50 = 0.05*1000.
        assert pencere._kaydirici.value() == 50
        assert pencere._level == 0.05
        # Başlık, ekrandaki resmin HANGİ hesaptan geldiğini söyleyen tek yer.
        # ÖLÇÜLDÜ: başlığı sabit "qlab" yapan mutasyon 42 testin SIFIRINI
        # kırıyordu -- iki farklı bazla açılan iki pencere ayırt edilemezdi.
        assert "RHF/STO-3G" in pencere.windowTitle()
    finally:
        pencere.close()

    baska = OrbitalWindow(solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "6-31g"))
    try:
        assert "RHF/6-31G" in baska.windowTitle()
    finally:
        baska.close()


@requires_offscreen
def test_acilis_boyutu_hem_3b_gorunume_hem_panele_yer_biraliyor():
    """`resize(960, 800)` korumasızdı; ölçüldü, küçük bir boyutta pencere işlevsiz.

    MUTASYON: `self.resize(960, 800)` -> `self.resize(300, 200)`. Tam takımda
    0 test kırılıyordu. Oysa fark ekransız kipte OKUNABİLİR ve yıkıcı:
        960x800 -> 3B görünüm 517 px, panelin metin alanı 190 px
        300x200 -> 3B görünüm   0 px, panelin metin alanı 107 px
    Yani orbitali gösteren widget'ın yüksekliği SIFIR oluyor -- pencerenin tek
    işi ekranda kalmıyor -- ve takım yemyeşil kalıyordu.

    Sınırlar LİTERAL ve ikisi de bir GEREKÇEYE bağlı: 3B görünüm pencerenin
    asıl nesnesi, panel ise en az TDSE penceresinin kutusu (140 px) kadar yer
    almalı, yoksa daha uzun olan beyanı daha küçük bir kutuda göstermiş olurdu.
    """
    pencere = OrbitalWindow(solve(Molecule.from_string(_SU), "sto-3g"))
    try:
        pencere.show()
        _qapp.processEvents()
        assert pencere._view.height() >= 300, (
            f"3B görünüme yer kalmamış ({pencere._view.height()} px); "
            f"pencerenin tek işi o"
        )
        assert pencere._panel._text.viewport().height() >= 140, (
            f"panelin metin alanı TDSE'nin varsayılan kutusundan bile küçük "
            f"({pencere._panel._text.viewport().height()} px)"
        )
    finally:
        pencere.close()


@requires_offscreen
def test_kaydiricinin_izgarasina_oturmayan_seviye_reddediliyor(su_penceresi):
    """`_dogrula_seviye`'nin üçüncü şartı yalnızca ALT sınırı bağlıyordu.

    Dokstring şunu vaat ediyordu: "Kaydırıcının çözünürlüğünde olmalı
    (>= 0.001). Kaydırıcı tam sayı binde-bir taşır; gösteremeyeceği bir seviye,
    denetimin panelden farklı bir sayı göstermesi demektir. Sessiz
    desenkronizasyon burada da reddedilir, kırpılmaz." Kod ise yalnızca
    `deger < 0.001`'i reddediyordu.

    ÖLÇÜLDÜ (mutasyonsuz HEAD, su/STO-3G) -- `set_isosurface_level` sonrası
    kaydırıcı konumu / `_level`:
        0.0155 -> 16 (0.016)   0.0567 -> 57 (0.057)   0.0014 ->   1 (0.001)
        0.1235 -> 124 (0.124)  0.0015 ->  2 (0.002)   0.0504 ->  50 (0.05)
    Altısı da DESENKRON; 0.0011'de kaydırıcı en küçük tıkta kilitli kalırken
    kullanıcı %10 yukarıdaydı, 0.0015'te sapma %33'tü. Eşdeğer de değil:
    0.0155 ve 0.016 FARKLI yüzeyler veriyor (3178 tepe / 3114 tepe).

    Deponun bu sınıfı ZATEN kusur saydığı yer duruyor: ÜST tarafta aynı
    şekilli desenkron (2.0 -> kaydırıcı 0.9) düzeltilmiş ve regresyon testi
    yazılmıştı (`test_kaydirici_seviyeyle_senkron_kaliyor`).

    TOLERANS ŞART, ölçüldü: kaydırıcı 1..5000 konumlarının 82'sinde
    `(n/1000)*1000` tam sayı DEĞİL (1001 -> 1000.9999999999999); en büyük
    yuvarlama artığı 4.547e-13. Katı bir tam-sayı kontrolü o 82 konumu
    reddeder ve sürükleme sırasında istisna Qt'nin sinyal işleyicisinde
    YUTULUR -- düzeltilmek istenenden daha kötü bir sessiz desenkron doğardı.
    """
    from qlab.app.orbital_window import _dogrula_seviye

    pencere, _ = su_penceresi
    once = _pencere_durumu(pencere)

    for oturmayan in (0.0155, 0.0567, 0.0014, 0.1235, 0.0015, 0.0504):
        with pytest.raises(InvalidInputError, match="izoyüzey seviyesi") as hata:
            pencere.set_isosurface_level(oturmayan)
        # Metin "binde-bir" DEĞİL, adımın SAYISINI taşıyor: kaydırıcının ölçeği
        # artık alan başına ayrı (orbital 1/1000, yoğunluk farkı 1/10000) ve
        # sabit bir kelime ikisinden birinde yalan olurdu. Adım aynı zamanda
        # kullanıcının düzeltebileceği şey.
        assert "ızgarasına oturmuyor" in str(hata.value), (
            f"red gerekçesi yanlış: {hata.value}"
        )
        assert "adım 0.001" in str(hata.value), (
            f"orbital alanının adımı basılmıyor: {hata.value}"
        )
    assert _pencere_durumu(pencere) == once, "reddedilen seviye durumu bozmamalı"

    # Kurucu yolu da aynı şartı taşımalı.
    with pytest.raises(InvalidInputError, match="ızgarasına oturmuyor"):
        OrbitalWindow(pencere._result, level=0.0155)

    # Izgaraya OTURAN seviyeler kabul edilir (literal sayılarla).
    for oturan in (0.001, 0.002, 0.016, 0.05, 0.124, 0.9, 2.0, 4.999):
        assert _dogrula_seviye(oturan) == oturan

    # TOLERANS: kaydırıcının ürettiği HER konum kabul edilmeli. 1001 ve
    # kardeşleri kayan noktada tam sayı vermiyor; katı bir kontrol onları
    # reddederdi.
    for konum in (1001, 1003, 1005, 2001, 4999, 5000):
        assert _dogrula_seviye(konum / 1000.0) == konum / 1000.0


@requires_offscreen
def test_kaydirici_surukleme_hicbir_konumda_desenkron_kalmiyor(su_penceresi):
    """Kaydırıcı gerçekten sürüklendiğinde panel ile aynı sayıyı göstermeli.

    Yukarıdaki testin ters yüzü: doğrulayıcı katılaştırılırken kaydırıcının
    kendi yaydığı değerlerin reddedilmediği BURADA sınanır. Qt sinyal
    işleyicisinde fırlayan bir istisna YUTULUR, yani böyle bir gerileme
    testlerde istisna olarak değil sessiz desenkron olarak görünür -- o yüzden
    iddia istisnaya değil, kaydırıcı konumu ile `_level`'ın eşitliğine bakar.

    1001 bilinçli seçim: `(1001/1000)*1000 = 1000.9999999999999`.

    Bant önce `set_isosurface_level(2.0)` ile büyütülür, çünkü kaydırıcının
    açılış üst sınırı 900 ve `setValue` onun ötesini SESSİZCE kırpar --
    kırpılmış bir konumu sınamak, kırpmayı sınamak olurdu (o zaten
    `test_kaydirici_seviyeyle_senkron_kaliyor`'un işi).
    """
    pencere, _ = su_penceresi
    try:
        pencere.set_isosurface_level(2.0)
        assert pencere._kaydirici.maximum() >= 2000
        for konum in (1, 2, 50, 123, 900, 1001, 1003, 1005, 2000):
            pencere._kaydirici.setValue(konum)
            assert pencere._kaydirici.value() == konum, (
                f"kaydırıcı {konum} konumunu reddetti (sinyal içinde yutulmuş "
                f"bir istisna olabilir)"
            )
            assert pencere._level == konum / 1000.0
            assert f"şu an {konum / 1000.0:g}" in pencere.transparency_text()
    finally:
        pencere.set_isosurface_level(0.05)


@requires_offscreen
def test_panel_izgara_cozunurlugunu_ayirt_edici_bir_sayiyla_beyan_ediyor(su_penceresi):
    """Beyan edilen sayı çözünürlüğü BELİRLEMELİ, yoksa hiçbir şey söylemez.

    Modül dokstring'i "Izgara aralığı ... bu pencerenin kendi tanısıdır
    (`grid_points_per_axis`)" diyordu. Ama eksen başına nokta sayısı, kutu
    genişliği olmadan aralığı vermez.

    ÖLÇÜLDÜ: n=60 sabit, yalnızca padding değişerek
        pad= 4.0 -> aralık 0.1841 bohr, ağ (1764, 1764)
        pad=12.0 -> aralık 0.4553 bohr, ağ ( 248,  248)
        pad=20.0 -> aralık 0.7265 bohr, ağ (  86,   86)
    ÜÇ PENCERENİN panel metni BİT BİREBİR AYNIYDI (sha256[:16] üçünde de
    7473027186428447). 4 kat çözünürlük farkı panelde hiç görünmüyordu.

    Kusur varsayılan dışı argüman da gerektirmiyor: yalnızca `--molecule`
    değişerek H2 0.1593 / HF 0.1650 / H2O 0.1841 / C2H4 0.2145 bohr aralık
    veriyor, panelde basılan sayı hep 60.

    Izgara ANİZOTROP olduğu için TEK sayı yetmez: aynı ızgarada eksen
    aralıkları 0.1356 / 0.1841 / 0.1544 bohr. En kaba ve en ince eksen ayrı
    ayrı beyan edilir; en kaba eksen `observable/hf_obs.py`'nin
    `grid_spacing_bohr` alışkanlığıyla aynı adı taşır.
    """
    pencere, _ = su_penceresi
    metinler = []
    for pad in (4.0, 12.0, 20.0):
        baska = OrbitalWindow(pencere._result, padding=pad)
        try:
            metinler.append(baska.transparency_text())
        finally:
            baska.close()
    assert len(set(metinler)) == 3, (
        "4 kat farklı çözünürlükte panel metni aynı kalıyor -- basılan sayı "
        "çözünürlüğü belirlemiyor"
    )

    beyan = pencere._provenance()
    assert beyan.diagnostics["grid_points_per_axis"] == 60
    # Sayılar bağımsız kurulan bir ızgaradan doğrulanır; `pencere._grid`
    # kullanılsaydı iddia totoloji olurdu.
    beklenen = VolumeGrid.around(
        pencere._result.molecule, padding=4.0, points_per_axis=60
    )
    assert beyan.diagnostics["grid_spacing_bohr"] == pytest.approx(
        float(np.max(beklenen.spacing))
    )
    assert beyan.diagnostics["grid_spacing_min_bohr"] == pytest.approx(
        float(np.min(beklenen.spacing))
    )
    # LİTERAL bant: sayı sessizce başka bir eksene ya da başka bir tanıma
    # kayarsa burada görünür (ölçüldü: 0.184086 / 0.135593).
    assert 0.1840 < beyan.diagnostics["grid_spacing_bohr"] < 0.1842
    assert 0.1355 < beyan.diagnostics["grid_spacing_min_bohr"] < 0.1357
    assert (
        beyan.diagnostics["grid_spacing_min_bohr"]
        < beyan.diagnostics["grid_spacing_bohr"]
    ), "ızgara anizotrop; iki eksen aynı sayıyı veriyorsa biri yanlış okunmuş"


@requires_offscreen
def test_mesh_vertex_counts_agsiz_durumu_sessizce_gizlemiyor(su_penceresi):
    """Erişilemez dalın yedeği geçerli bir çıktıdan ayırt edilemiyordu.

    `mesh_vertex_counts` `self._meshes is None` iken `(0, 0)` döndürüyordu ve
    dal `# pragma: no cover` ile erişilemez ilan edilmişti. Erişilemezlik
    DOĞRU (kaydırıcı sinyalleri `_kur_arayuz`'da bloklu, `_yenile`
    `__init__`'in sonuncusu), ama `(0, 0)` bu pencerede GERÇEK bir durumdur:
    ölçüldü, su HOMO'su level=0.90'da tam olarak (0, 0) veriyor. Yani
    `_yenile`'yi `__init__`'ten sonraya taşıyan bir refactor, istisna yerine
    tamamen makul görünen bir sayı döndürürdü.

    Yükseltmek aynı erişilemezlik iddiasını korur ve sessiz yalanı kaldırır;
    dal artık burada koşulduğu için `pragma` da gereksizleşti.
    """
    pencere, _ = su_penceresi
    yedek = pencere._meshes
    try:
        pencere._meshes = None
        with pytest.raises(AssertionError, match="_yenile"):
            pencere.mesh_vertex_counts()
    finally:
        pencere._meshes = yedek
    assert pencere.mesh_vertex_counts() == (1764, 1764)


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
def test_main_basis_argumani_gercekten_SONUCA_ulasiyor(acilan_pencereler):
    """`--basis` yolunun sto-3g DIŞINDAKİ dalı hiç kapsanmıyordu.

    Testlerde geçen tek `--basis` değeri 'sto-3g' idi (grep: tek eşleşme).
    Bayrak sessizce yok sayılsa ve her çağrı sto-3g'ye düşse takım yemyeşil
    kalırdı -- iki dalın AYNI sonucu vermesi mümkün olduğu için 'sto-3g'
    dalını test etmek bayrağın taşındığını göstermez.

    ÖLÇÜLDÜ (H2, aynı geometri, yalnızca bayrak değişiyor):
        --basis sto-3g -> basis_name 'sto-3g', seçicide 2 orbital,
                          başlık 'qlab — orbital görüntüleyici (RHF/STO-3G)'
        --basis 6-31g  -> basis_name '6-31g',  seçicide 4 orbital,
                          başlık 'qlab — orbital görüntüleyici (RHF/6-31G)'
    Sayılar LİTERAL: 2 ve 4 koddaki hiçbir sabite bağlı değil, H2'nin iki
    bazdaki gerçek baz fonksiyonu sayısıdır.
    """
    app_main, acilanlar = acilan_pencereler
    olculen = []
    for baz in ("sto-3g", "6-31g"):
        acilanlar.clear()
        kod = app_main.main(["--molecule", "H 0 0 0; H 0 0 0.74", "--basis", baz])
        assert kod == 0
        pencere = acilanlar[0]
        try:
            assert pencere._result.basis_name == baz, "bayrak sonuca ulaşmıyor"
            olculen.append((pencere._secici.count(), pencere.windowTitle()))
        finally:
            pencere.close()

    (sto_orbital, sto_baslik), (g631_orbital, g631_baslik) = olculen
    assert sto_orbital == 2 and g631_orbital == 4
    assert "STO-3G" in sto_baslik and "6-31G" in g631_baslik
    assert sto_baslik != g631_baslik, "başlık hangi bazın koştuğunu göstermiyor"


@requires_offscreen
def test_main_bilinmeyen_bazi_PENCERE_ACMADAN_reddediyor(acilan_pencereler, capsys):
    """Geçersiz baz, tipli hataya düşmeli ve hiçbir pencere kurulmamalı.

    ÖLÇÜLDÜ: `--basis cc-pvdz` -> kod 1, stderr «Bilinmeyen baz seti:
    'cc-pvdz'. Desteklenenler: ('sto-3g', '6-31g')», traceback YOK. Doğru
    çalışıyordu ama tek bir test bile bunu tutmuyordu.

    KAPSANMAYAN, açıkça: "Qt hiç başlatılmadan" iddiası bu testte
    SINANAMAZ -- bu modül içe aktarılırken zaten bir QApplication kuruyor,
    yani süreçte her hâlükârda bir tane var. Sınanan şey gözlemlenebilir
    olan: pencere KURULMUYOR ve hata kullanıcının düzeltebileceği şeyi
    (desteklenen baz adlarını) adlandırıyor.
    """
    app_main, acilanlar = acilan_pencereler
    kod = app_main.main(["--molecule", "H 0 0 0; H 0 0 0.74", "--basis", "cc-pvdz"])
    hata = capsys.readouterr().err
    assert kod == 1
    assert acilanlar == [], "geçersiz bazda pencere kurulmamalı"
    assert "cc-pvdz" in hata, hata
    # Kullanıcı ne yazabileceğini görmeli; yalnızca "bilinmeyen" demek yetmez.
    assert "sto-3g" in hata and "6-31g" in hata, hata
    assert "Traceback" not in hata


@requires_offscreen
def test_main_sonlu_olmayan_koordinat_DOGRU_teshisle_reddediliyor(
    acilan_pencereler, capsys
):
    """`--molecule` geçersiz geometriyi geçersiz geometri diye reddetmeli.

    ÖLÇÜLDÜ (düzeltmeden önce, üç biçim de): `main(["--molecule",
    "H 0 0 0; H 0 0 nan"])` kod=1 döndürüyordu ama stderr'e şunu basıyordu:

        SCF 100 iterasyonda yakinsamadi (son residual nan). Iterasyon
        sinirini artirmayi veya baslangic tahminini degistirmeyi deneyin.

    Hata TİPİ doğruydu (SCFConvergenceError → QLabError), ama TEŞHİS yanlıştı
    ve önerdiği çare kullanıcıyı yanlış düzeltmeye yönlendiriyordu: sorun
    yakınsama değil, geometrinin kendisi. Bu depo aynı kusur sınıfını
    `_dogrula_seviye` için açıkça reddediyor ("yanlış gerekçe gören kullanıcı
    yanlış düzeltmeyi dener"). Üstelik bedeli 100 SCF iterasyonu.

    `inf` ayrıca ölçüldü ve daha da kötüydü: teşhisten ÖNCE
    `integrals.py:241`'den ham bir `RuntimeWarning: invalid value encountered
    in subtract` sızıyordu.

    KAPSAM, açıkça: bu koruma CLI sınırında. `Molecule.from_string` ve
    `Atom.__post_init__` (types.py) sonlu olmayan konumu HÂLÂ kabul ediyor --
    ölçüldü, kütüphane API'sinden `Molecule.from_string("H 0 0 nan")` sorunsuz
    kuruluyor. Serbest metin geometriyi bir KULLANICIYA açan tek yüzey
    `--molecule` olduğu için kapı burada kapatıldı; types.py'deki daha geniş
    düzeltme ayrı bir iştir ve burada yapılmış SAYILMAZ.
    """
    app_main, acilanlar = acilan_pencereler
    for geometri in (
        "H 0 0 0; H 0 0 nan",
        "H 0 0 0; H 0 0 inf",
        "H 0 0 0; H 0 0 -inf",
        "H 0 0 nan; H 0 0 0.74",
    ):
        kod = app_main.main(["--molecule", geometri])
        hata = capsys.readouterr().err
        assert kod == 1, geometri
        assert acilanlar == [], "geçersiz geometride pencere açılmamalı"
        assert "yakinsamadi" not in hata, (
            f"{geometri!r} yakınsama sorunu diye teşhis edildi: {hata!r}"
        )
        assert "iterasyon" not in hata.lower(), (
            f"{geometri!r} için önerilen çare hâlâ iterasyon sınırı: {hata!r}"
        )
        # Teşhis, kullanıcının DÜZELTEBİLECEĞİ şeyi adlandırmalı.
        assert "koordinat" in hata.lower(), hata
        assert "sonlu" in hata.lower(), hata


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


@requires_offscreen
def test_pencere_gosterilmeden_olay_dongusune_girilmiyor():
    """`window.show()` korumasızdı: pencere hiç gösterilmeden exec()'e girilebilirdi.

    ÖLÇÜLDÜ: `main.py`'den `window.show()` satırını silen mutasyon 58 testin
    SIFIRINI kırıyordu. Sebep yapısal: bütün CLI testleri `_pencereyi_goster`'i
    TAMAMEN yamalıyor (`acilan_pencereler` fixture'ı), yani `show()` çağrısı
    hiçbir testte hiç koşmuyor ve varlığını kimse doğrulamıyor.

    "Olay döngüsü testte koşturulamaz" DOĞRU bir gerekçe, ama yamanın sınırı
    bir satır fazla geniş çizilmişti: `show()` olay döngüsü değil, ekransız
    kipte OKUNABİLİR durumdur -- ölçüldü, `isVisible()` çağrıdan önce False,
    sonra True.

    Bu test o yüzden `_pencereyi_goster`'i DEĞİL yalnızca `QApplication.exec`'i
    yamalar. Böylece gerçek gövde koşar ve iki şey birden sınanır: pencere
    gösteriliyor mu ve olay döngüsünün çıkış kodu geri taşınıyor mu.
    """
    from qlab.app import main as app_main

    pencere = OrbitalWindow(solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g"))
    orijinal = QtWidgets.QApplication.exec
    try:
        assert not pencere.isVisible()
        QtWidgets.QApplication.exec = lambda self: 7
        kod = app_main._pencereyi_goster(pencere)
        assert pencere.isVisible(), "pencere gösterilmeden olay döngüsüne girildi"
        assert kod == 7, "olay döngüsünün çıkış kodu çağırana taşınmıyor"
    finally:
        QtWidgets.QApplication.exec = orijinal
        pencere.close()


# ---------------------------------------------------------------------------
# B2'nin dört public yüzeyi: ÜRETİM YOLU
#
# `density_difference`, `extract_isosurface`, `analyse` ve
# `free_atom_density_matrix` bugüne kadar YALNIZCA testlerden çağrılıyordu.
# Yani modülün manşet iddiaları -- "bağ kodda tanımlanmaz, ölçülür" ve
# "elektronegatiflik yoğunluk farkından okunur" -- hiçbir kullanıcı kodundan
# geçmiyordu; kimse onları GÖREMİYORDU. Aşağıdaki testler o dördünü
# pencerenin içine bağlar.


@requires_offscreen
def test_alan_secici_uc_alan_turunu_de_listeliyor(su_sonucu):
    """Seçici üç ALAN TÜRÜ taşır ve açılışta orbitalde durur.

    TAZE pencere, bilinçli: bu bir AÇILIŞ durumu iddiası ve bu dosyada
    paylaşılan `su_penceresi` fixture'ının açılış mutasyonlarını ONARDIĞI
    ölçülmüştü (bkz. `test_kaydiricinin_acilis_araligi_surukleyebilir_bir_bant`
    ve `test_acilista_secici_gosterilen_orbitali_isaret_ediyor`).
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK

    pencere = OrbitalWindow(su_sonucu)
    try:
        etiketler = [
            pencere._alan_secici.itemText(i)
            for i in range(pencere._alan_secici.count())
        ]
        assert etiketler == [ALAN_ORBITAL, ALAN_YOGUNLUK, ALAN_FARK]
        assert etiketler == ["orbital", "elektron yoğunluğu", "yoğunluk farkı"]
        assert pencere.field_kind == ALAN_ORBITAL
        assert pencere._alan_secici.currentIndex() == 0
    finally:
        pencere.close()


@requires_offscreen
def test_her_alan_turu_KENDI_ureticisinden_geliyor(su_sonucu):
    """Üç alanın üçü de AYRI bir üreticiden gelmeli; ikisi aynıya bağlanamaz.

    Bu testin işi bir sayı kilidi değil, bir EŞLEME kilidi: pencerenin
    gösterdiği değerler, o alan türünün adını taşıyan `observable` fonksiyonunun
    BAĞIMSIZ olarak yeniden hesaplanmış çıktısıyla birebir aynı olmalı.
    Üç alan pairwise farklı olduğu için (aşağıda ölçülüyor) yanlış eşleme
    sessiz kalamaz.

    ÖLÇÜLDÜ (su/STO-3G, pad=4.0, n=60 -- pencerenin kendi varsayılanları):
        orbital (HOMO)     aralık [-0.697796, +0.697796]
        elektron yoğunluğu aralık [ 0.000000, 49.960388]
        yoğunluk farkı     aralık [-0.283034, +0.318152]
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK
    from qlab.observable.hf_obs import (
        ScalarField,
        density_difference,
        electron_density,
        evaluate_orbital,
    )

    izgara = VolumeGrid.around(su_sonucu.molecule, padding=4.0, points_per_axis=60)
    beklenen = {
        ALAN_ORBITAL: evaluate_orbital(su_sonucu, 4, izgara),
        ALAN_YOGUNLUK: electron_density(su_sonucu, izgara).values,
        ALAN_FARK: density_difference(su_sonucu, izgara).values,
    }
    # İddia totoloji OLMASIN: üç alan gerçekten farklı olmalı.
    assert not np.allclose(beklenen[ALAN_ORBITAL], beklenen[ALAN_YOGUNLUK])
    assert not np.allclose(beklenen[ALAN_YOGUNLUK], beklenen[ALAN_FARK])
    assert not np.allclose(beklenen[ALAN_ORBITAL], beklenen[ALAN_FARK])
    assert beklenen[ALAN_YOGUNLUK].min() >= 0.0, "elektron yoğunluğu kesin pozitif"
    assert beklenen[ALAN_FARK].min() < 0.0, "fark haritasında boşalma da olmalı"

    pencere = OrbitalWindow(su_sonucu)
    try:
        for ad, degerler in beklenen.items():
            pencere.set_field_kind(ad)
            assert isinstance(pencere._alan, ScalarField)
            assert np.allclose(pencere._alan.values, degerler), (
                f"{ad!r} alanı başka bir üreticiye bağlanmış"
            )
    finally:
        pencere.close()


@requires_offscreen
def test_yogunluk_ISARETSIZ_cikarimla_geliyor(su_sonucu):
    """`extract_isosurface`'in var olma sebebi tam olarak bu alan.

    Elektron yoğunluğu KESİN POZİTİFTİR (ölçüldü: minimum 0.000000), yani
    `-level` izoyüzeyi hiçbir zaman yoktur. ÖLÇÜLDÜ (su/STO-3G, pad=4.0, n=60,
    level=0.1): `extract_signed_pair` (+2196, -0) veriyor -- ikinci ağ HER
    ZAMAN boş. `extract_isosurface` tek ağ verir ve ±faz renk sözleşmesini hiç
    ima etmez.

    `extract_signed_pair`'e "sadeleştiren" bir mutasyon burada görünür: ağ
    sayısı 1'den 2'ye çıkar ve `mesh_vertex_counts()` (2196,) yerine (2196, 0)
    döner.
    """
    from qlab.app.orbital_window import ALAN_YOGUNLUK
    from qlab.viz.isosurface import extract_signed_pair

    pencere = OrbitalWindow(su_sonucu, field=ALAN_YOGUNLUK)
    try:
        assert pencere._level == 0.1
        assert pencere.mesh_vertex_counts() == (2196,)
        assert len(pencere._meshes) == 1
        assert pencere._meshes[0].sign == 1
        assert pencere._meshes[0].level == 0.1
        # İşaretli çift AYNI alanda ne verirdi: ikinci ağ boş.
        p, n = extract_signed_pair(pencere._alan, 0.1)
        assert (len(p.vertices), len(n.vertices)) == (2196, 0)
        # Çizilen item sayısı: 1 çekirdek + 1 ağ.
        assert len(pencere._mesh_items) == 1
        assert len(pencere._view.items) == 2
    finally:
        pencere.close()


@requires_offscreen
def test_fark_ISARETLI_cift_uretiyor(su_sonucu):
    """Fark haritasında işaret FİZİKSELDİR: + birikim, − boşalma.

    Orbitalin global işareti keyfidir (aynı özvektörün -1 katı aynı çözümdür),
    ama fark haritasının işareti keyfi DEĞİLDİR: `rho_molekul - rho_pro`
    pozitifse orada elektron BİRİKMİŞTİR. İki ağ da gerçekten dolu olmalı,
    yoksa iddia gösterilemez.

    ÖLÇÜLDÜ (su/STO-3G, pad=4.0, n=60, level=0.01): (+1764, -2020).
    """
    from qlab.app.orbital_window import ALAN_FARK

    pencere = OrbitalWindow(su_sonucu, field=ALAN_FARK)
    try:
        assert pencere._level == 0.01
        assert pencere.mesh_vertex_counts() == (1764, 2020)
        assert len(pencere._meshes) == 2
        assert [m.sign for m in pencere._meshes] == [1, -1]
        assert len(pencere._mesh_items) == 2
        assert len(pencere._view.items) == 3
    finally:
        pencere.close()


@requires_offscreen
def test_her_alanin_KENDI_seviye_araligi_var(su_sonucu):
    """Tek bir kaydırıcı aralığı üç alanın ikisinde YANLIŞTIR.

    ÖLÇÜLDÜ (su/STO-3G, pad=4.0, n=60), alanların gerçek genlikleri:
        orbital            maks|psi| = 0.697796
        elektron yoğunluğu aralık [0.000000, 49.960388]  (çekirdek tepesi)
        yoğunluk farkı     aralık [-0.283034, +0.318152]
    Eski tek aralık 0.001..0.900 idi: yoğunlukta üst sınır tepenin 55'te
    biri, farkta ise kullanışlı bant (0.002-0.01) kaydırıcının ilk %1'ine
    sıkışıyordu.

    ÇÖZÜNÜRLÜK de alan başına ölçüldü. Fark haritasında 0.0001 adımlar AYRI
    yüzeyler veriyor -- 0.0020 (+3208,-4704) / 0.0021 (+3196,-4596) /
    0.0022 (+3156,-4544) / 0.0023 (+3108,-4476) -- yani binde-bir çözünürlük o
    banttaki yapıyı GERÇEKTEN gizler; fark alanı bu yüzden on-binde-bir
    taşır. Yoğunlukta binde-bir yetiyor: 0.095 (2260) / 0.096 (2248) /
    0.097 (2228) zaten ayrı.

    Bantların uçları da ölçüm: yoğunluk 0.01'de 4892, 2.0'da 108 tepe
    (2.0'ın üstü yalnızca ızgaranın ÇÖZMEDİĞİ çekirdek tepesidir); fark
    0.0005'te (+4184,-7368), 0.3'te (+20,-0) ve 0.32'de tamamen boş
    (maks 0.318152).

    Sayılar LİTERAL. Alan türünün kendi sabitine karşı yazılan bir iddia o
    sabitle birlikte hareket eder ve hiçbir zaman kırılamaz -- bu depoda
    gerçekten oldu (`maximum() >= _KAYDIRICI_VARSAYILAN_MAKS`).
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK

    beklenen = {
        #  alan            seviye  kaydırıcı (değer, min, maks)
        ALAN_ORBITAL: (0.05, 50, 1, 900),
        ALAN_YOGUNLUK: (0.1, 100, 10, 2000),
        ALAN_FARK: (0.01, 100, 5, 3000),
    }
    for ad, (seviye, deger, en_az, en_cok) in beklenen.items():
        pencere = OrbitalWindow(su_sonucu, field=ad)
        try:
            assert pencere._level == seviye, ad
            assert pencere._kaydirici.value() == deger, ad
            assert pencere._kaydirici.minimum() == en_az, ad
            assert pencere._kaydirici.maximum() == en_cok, ad
        finally:
            pencere.close()

    # Üç aralığın ÜÇÜ DE farklı olmalı; tek bir sabite çökertilirse burası
    # kırılır.
    assert len({(v[1], v[2], v[3]) for v in beklenen.values()}) == 3


@requires_offscreen
def test_her_alanin_KENDI_renk_isaret_beyani_var(su_sonucu):
    """Aynı iki renk üç alanda ÜÇ AYRI şey demek; tek metin ikisinde yalan.

    ÖLÇÜLDÜ: bu test yazılmadan önce, `_gosterim_yaklasimlari`'nın alan
    beyanını sabit olarak ORBİTALİNKİNE çeviren mutasyon dosyanın 68 testinin
    SIFIRINI kırıyordu -- yani fark haritasının panelinde "orbital işareti
    keyfi ... global işaret fiziksel değildir" yazsa hiçbir test görmezdi.

    Oysa üçü birbirinin ZIDDI:
      * orbital -- işaret KEYFİ: `psi` bir çözümse `-psi` de aynı enerjiyle
        aynı çözümdür, renkler yalnızca bağıl fazı gösterir.
      * elektron yoğunluğu -- işaret diye bir şey YOK: alan kesin pozitif
        (ölçülen aralık [0.000000, 49.960388]), ikinci ağ hep boş.
      * yoğunluk farkı -- işaret FİZİKSEL: + birikim, - boşalma. Burada
        "işaret keyfidir" yazmak, modülün manşet iddiasını yalanlardı.

    O yüzden hem BULUNMASI hem BULUNMAMASI gerekenler sınanıyor.
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK

    beklenen = {
        ALAN_ORBITAL: "işareti keyfi",
        ALAN_YOGUNLUK: "kesin pozitiftir",
        ALAN_FARK: "işaret FİZİKSELDİR",
    }
    pencere = OrbitalWindow(su_sonucu)
    try:
        for ad, capa in beklenen.items():
            pencere.set_field_kind(ad)
            metin = pencere.transparency_text()
            assert capa in metin, f"{ad!r} alanının beyanı panelde yok: {capa!r}"
            for baska_ad, baska_capa in beklenen.items():
                if baska_ad != ad:
                    assert baska_capa not in metin, (
                        f"{ad!r} alanında {baska_ad!r} alanının beyanı basılıyor: "
                        f"{baska_capa!r}"
                    )
    finally:
        pencere.close()


@requires_offscreen
def test_alan_degisince_seviye_alanin_KENDI_varsayilanina_donuyor(su_sonucu):
    """Alan değişince eski seviye taşınmaz -- taşınsa çoğu zaman GEÇERSİZ olurdu.

    ÖLÇÜLDÜ: orbitalin 0.9 seviyesi fark haritasında hâlâ geçerli bir sayı
    (aralık 0.3'e kadar boş yüzey verir) ama yoğunlukta 0.9 kaba bir çekirdek
    yüzeyidir; tersine farkın 0.0005'i orbital kaydırıcısının çözünürlüğünün
    ALTINDA kalır ve `_dogrula_seviye` tarafından reddedilirdi. Taşımak yerine
    her alan kendi varsayılanına döner ve kaydırıcı da onunla gider.
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK

    pencere = OrbitalWindow(su_sonucu)
    try:
        pencere.set_isosurface_level(0.5)
        assert pencere._level == 0.5

        pencere.set_field_kind(ALAN_FARK)
        assert pencere._level == 0.01
        assert pencere._kaydirici.value() == 100
        assert pencere._kaydirici.maximum() == 3000

        pencere.set_field_kind(ALAN_YOGUNLUK)
        assert pencere._level == 0.1
        assert pencere._kaydirici.value() == 100
        assert pencere._kaydirici.maximum() == 2000

        pencere.set_field_kind(ALAN_ORBITAL)
        assert pencere._level == 0.05
        assert pencere._kaydirici.value() == 50
        assert pencere._kaydirici.maximum() == 900
    finally:
        pencere.close()


@requires_offscreen
def test_fark_alaninda_seviye_ONBINDE_BIR_izgarasina_baglaniyor(su_sonucu):
    """Fark alanının kaydırıcısı on-binde-bir taşır; doğrulama da öyle olmalı.

    Doğrulayıcı alan türünden BAĞIMSIZ tek bir ölçeğe bağlı kalsaydı iki yönlü
    sessiz desenkron doğardı: fark alanında 0.0007 (kaydırıcının 7. tıkı)
    reddedilir, 0.0155 gibi binde-bir ızgarasına oturmayan bir sayı ise kabul
    edilirdi. İkisi de burada sınanıyor, LİTERAL sayılarla.
    """
    from qlab.app.orbital_window import ALAN_FARK

    pencere = OrbitalWindow(su_sonucu, field=ALAN_FARK)
    try:
        # Kaydırıcının kendi ürettiği her konum kabul edilmeli.
        for konum in (5, 7, 100, 155, 1001, 3000):
            pencere._kaydirici.setValue(konum)
            assert pencere._kaydirici.value() == konum
            assert pencere._level == konum / 10000.0

        # On-binde-bir ızgarasına OTURMAYAN seviye reddedilir. Sayı bandın
        # İÇİNDEN seçildi (0.0005 <= 0.00155 <= 0.3): bandın altındaki bir
        # değer ÖNCEKİ kontrole takılır ve bu şartı hiç sınamazdı -- ilk
        # yazılışında tam olarak öyle oldu ve gerekçe iddiası yakaladı.
        with pytest.raises(InvalidInputError, match="izoyüzey seviyesi") as hata:
            pencere.set_isosurface_level(0.00155)
        assert "ızgarasına oturmuyor" in str(hata.value), hata.value
        assert "0.0001" in str(hata.value), hata.value
        assert "0.0016" in str(hata.value), "en yakın geçerli seviye önerilmiyor"

        # Bandın ALTI reddedilir: kaydırıcı 0.0005'in altını gösteremez.
        with pytest.raises(InvalidInputError, match="çözünürlüğünün") as hata:
            pencere.set_isosurface_level(0.0001)
        assert "0.0005" in str(hata.value), hata.value
        assert pencere._level == 0.3, "reddedilen seviye durumu bozmamalı"
    finally:
        pencere.close()


@requires_offscreen
def test_yogunluk_alaninda_bandin_alti_reddediliyor(su_sonucu):
    """Yoğunluk kaydırıcısının tabanı 0.01; altı gösterilemez, kırpılmaz.

    ÖLÇÜLDÜ: `QSlider.setValue` aralığın dışını SESSİZCE kırpar. 0.005
    istenip kaydırıcı 0.01'de kalsaydı panel ile denetim farklı sayı
    gösterirdi -- deponun `_kaydirici_senkronla`'da zaten reddettiği kusurun
    aynısı.
    """
    from qlab.app.orbital_window import ALAN_YOGUNLUK

    pencere = OrbitalWindow(su_sonucu, field=ALAN_YOGUNLUK)
    try:
        with pytest.raises(InvalidInputError, match="çözünürlüğünün") as hata:
            pencere.set_isosurface_level(0.005)
        assert "0.01" in str(hata.value), hata.value
        assert pencere._level == 0.1, "reddedilen seviye durumu bozmamalı"
        assert pencere._kaydirici.value() == 100
    finally:
        pencere.close()


@requires_offscreen
def test_gecersiz_alan_turu_reddediliyor_ve_durumu_bozmuyor(su_penceresi):
    """Bilinmeyen alan adı sessizce bir alana düşmemeli."""
    pencere, _ = su_penceresi
    once = _pencere_durumu(pencere)
    with pytest.raises(InvalidInputError, match="alan türü") as hata:
        pencere.set_field_kind("elektron yogunlugu")  # diyakritiksiz: YOK
    # Kullanıcı ne yazabileceğini görmeli.
    assert "orbital" in str(hata.value)
    assert "yoğunluk farkı" in str(hata.value)
    assert _pencere_durumu(pencere) == once
    assert pencere.field_kind == "orbital"

    with pytest.raises(InvalidInputError, match="alan türü"):
        OrbitalWindow(pencere._result, field="rho")


@requires_offscreen
def test_orbital_secici_yalnizca_orbital_alaninda_ETKIN(su_sonucu):
    """Yoğunluk ve fark alanında "hangi orbital" sorusunun karşılığı YOKTUR.

    İkisi de yoğunluk MATRİSİNDEN üretilir (`electron_density` P ile kontrakte
    eder), tek bir orbitalden değil. Seçici etkin bırakılsaydı kullanıcı onu
    çevirir ve ekranda hiçbir şey değişmezdi -- "etkin ama etkisiz" bir
    denetim yalan söyler. Aynı karar `TDSEWindow`'da bariyeri olmayan
    senaryolar için zaten verilmiş durumda.
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK

    pencere = OrbitalWindow(su_sonucu)
    try:
        assert pencere._secici.isEnabled()
        for ad in (ALAN_YOGUNLUK, ALAN_FARK):
            pencere.set_field_kind(ad)
            assert not pencere._secici.isEnabled(), ad
        pencere.set_field_kind(ALAN_ORBITAL)
        assert pencere._secici.isEnabled()
    finally:
        pencere.close()


@requires_offscreen
def test_alan_secici_kullanicida_olu_degil(su_sonucu):
    """Sinyal gerçekten yayılmalı; `set_field_kind`'ı doğrudan çağırmak yetmez.

    ÖLÇÜLDÜ (bu deponun kardeş kusuru): `currentIndexChanged` bağlantısını
    kaldırmak, orbital seçicisi için 9/9 test geçiriyordu -- çünkü bütün
    testler ilgili metodu DOĞRUDAN çağırıyordu.
    """
    pencere = OrbitalWindow(su_sonucu)
    try:
        pencere._alan_secici.setCurrentIndex(2)
        assert pencere.field_kind == "yoğunluk farkı", "alan seçici pencereye bağlı değil"
        assert pencere.mesh_vertex_counts() == (1764, 2020)
        pencere._alan_secici.setCurrentIndex(1)
        assert pencere.field_kind == "elektron yoğunluğu"
        assert pencere.mesh_vertex_counts() == (2196,)
    finally:
        pencere.close()


@requires_offscreen
def test_alan_degistikce_ag_itemleri_sizmiyor(su_sonucu):
    """Alan değişiminde de eski ağlar görünümden silinmeli.

    Orbital değişimi için aynı sızıntı ölçülmüştü (13 değişimde item sayısı
    3'ten 27'ye çıkıyordu); alan değişimi ikinci bir yol ve kendi kilidi
    olmalı. Değişmez: 1 çekirdek item'ı + BOŞ OLMAYAN ağ sayısı.
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK

    pencere = OrbitalWindow(su_sonucu)
    try:
        for _tur in range(3):
            for ad in (ALAN_ORBITAL, ALAN_YOGUNLUK, ALAN_FARK):
                pencere.set_field_kind(ad)
                dolu = sum(1 for sayi in pencere.mesh_vertex_counts() if sayi > 0)
                assert len(pencere._mesh_items) == dolu, ad
                assert len(pencere._view.items) == 1 + dolu, ad
        assert len(pencere._view.items) <= 3
    finally:
        pencere.close()


# ---------------------------------------------------------------------------
# `analyse()` -- rapor ekranda


@requires_offscreen
def test_analiz_okuma_satiri_OLCULEN_sayilari_basiyor(su_penceresi):
    """`analyse()` bugüne kadar hiçbir kullanıcı kodundan geçmiyordu.

    ÖLÇÜLDÜ (su/STO-3G):
        Mulliken  [-0.365510, +0.182755, +0.182755]  (toplam 0, molekül nötr)
        Löwdin    [-0.252869, +0.126434, +0.126434]
        dipol     1.725514 D
        HOMO      -0.391274   LUMO 0.605136   aralık 0.996410 hartree
        Mayer     O1-H2 0.954201, O1-H3 0.954201, H2-H3 0.012400

    Metin PANELDEN/WIDGET'tan okunur, ikinci kez üretilerek değil -- kardeş
    `transparency_text()` için ölçülmüştü ki ikinci-üretim deseni, widget'a
    hiç yazmayan bir mutasyonu da yeşil bırakıyor (panel 0 karakter, metot
    1174 karakter, 9/9 geçiyordu).
    """
    pencere, _ = su_penceresi
    metin = pencere.analysis_text()
    assert metin, "okuma satırı boş"

    for parca in (
        "O1 -0.3655",
        "H2 +0.1828",
        "H3 +0.1828",
        "1.7255 D",
        "-0.391274",
        "0.605136",
        "0.996410",
        "O1-H2 0.9542",
        "H2-H3 0.0124",
    ):
        assert parca in metin, f"{parca!r} okuma satırında yok:\n{metin}"

    # YAPISAL SONDA: metot widget'ı OKUYOR mu, yoksa ikinci kez mi üretiyor?
    pencere._okuma.setText("SONDA-42")
    try:
        assert pencere.analysis_text() == "SONDA-42", (
            "analysis_text() widget'ı okumuyor, metni yeniden üretiyor"
        )
    finally:
        pencere._okuma_satirini_guncelle()
    assert pencere.analysis_text() != "SONDA-42"


@requires_offscreen
def test_analiz_okuma_satiri_gercekten_PENCEREDE(su_penceresi):
    """Okuma satırı düzene eklenmiş ve görünür olmalı.

    ÖLÇÜLDÜ (kardeş kusur, aynı dosya): `self._panel.setVisible(False)`
    mutasyonu 9/9 geçiyordu -- widget ekranda yokken testler mutluydu.
    """
    pencere, _ = su_penceresi
    assert not pencere._okuma.isHidden(), "okuma satırı gizlenmiş"
    assert pencere._okuma.parent() is not None, "okuma satırı düzene eklenmemiş"
    assert pencere.centralWidget().isAncestorOf(pencere._okuma)


@requires_offscreen
def test_analiz_beyani_seffaflik_paneline_KATILIYOR(su_penceresi):
    """Ekranda yük görünüyorsa, o yükün üç ölçülmüş uyarısı da görünmeli.

    `analyse()`in `Provenance`'ı üç uyarı taşır ve üçü de bu pencerede
    basılan sayılara doğrudan dokunur:
      1. Mulliken/Löwdin baz duyarlılığı -- LiH/STO-3G'de İŞARET bile ters
         çıkıyor (Li'ye -0.016028455; PySCF aynı sayıyı veriyor).
      2. Mayer bağ mertebesi RHF'de AYRIŞMAYI göstermez: H2'nin mertebesi her
         mesafede tam 1.0 kalıyor.
      3. HOMO/LUMO Koopmans düzeyindedir; iyonlaşma enerjisi -eps_HOMO'ya
         eşit ALINAMAZ.
    Ekranda -0.3655 yazıp bu üçünü yazmamak, sayıya hak etmediği bir kesinlik
    verirdi.
    """
    from qlab.engine.hf.analysis import analyse

    pencere, _ = su_penceresi
    metin = pencere.transparency_text()
    rapor = analyse(pencere._result)

    # Motorun kendi maddeleri ZATEN vardı; eklenmesi gerekenler analizinkiler.
    analiz_ozel = [
        m
        for m in rapor.provenance.approximations + rapor.provenance.neglected
        if m not in pencere._result.provenance.approximations
        and m not in pencere._result.provenance.neglected
    ]
    assert len(analiz_ozel) == 5, analiz_ozel
    for madde in analiz_ozel:
        assert madde in metin, f"analiz beyanı panelde yok: {madde[:70]!r}"

    # Üç uyarının okunabilir izleri (metin sarılabilir; kısa çapalar):
    assert "Mulliken" in metin
    assert "Koopmans" in metin
    assert "Mayer" in metin


@requires_offscreen
def test_kararlilik_hukmu_ALAN_DEGISINCE_de_panelde_kaliyor(su_sonucu):
    """Kararlılık hükmü DALGA FONKSİYONUNUN özelliğidir, alanın değil.

    `electron_density` ve `density_difference` kendi `Provenance`'larında
    `stability` alanını `None` bırakıyor (ölçüldü) -- doğru, çünkü onlar
    kararlılık ölçmüyor. Ama pencere o beyanı OLDUĞU GİBİ taşısaydı, kullanıcı
    alanı değiştirdiğinde panelin "Kararlılık" satırı ORTADAN KALKARDI ve
    bu "sorun yok" diye okunurdu. Aynı gerekçe `Provenance.stability`
    dokstring'inde ve `analyse()`in "provenance'ı sıfırdan kurup stability'yi
    None bırakmak..." notunda zaten yazılı.

    Su/STO-3G'de hüküm KARARLI ve en düşük Hessian özdeğeri 10 dönme üzerinden
    ölçülüyor; satırın metni `StabilityVerdict.as_line()`'dan gelir.
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK

    assert su_sonucu.provenance.stability is not None, (
        "motor kararlılık ölçmediyse bu test totoloji olurdu"
    )
    pencere = OrbitalWindow(su_sonucu)
    try:
        for ad in (ALAN_ORBITAL, ALAN_YOGUNLUK, ALAN_FARK):
            pencere.set_field_kind(ad)
            beyan = pencere._provenance()
            assert beyan.stability == su_sonucu.provenance.stability, ad
            assert "Kararlılık" in pencere.transparency_text(), ad
            assert "10 dönme" in pencere.transparency_text(), ad
    finally:
        pencere.close()


@requires_offscreen
def test_LUMOSUZ_sistemde_okuma_satiri_YOK_diyor(su_sonucu):
    """He/STO-3G'de boş orbital yoktur; 0.0 yazmak "boşluk yok" diye okunurdu.

    ÖLÇÜLDÜ (He/STO-3G): tek orbital, HOMO -0.876036, LUMO None, aralık None.
    `AnalysisReport` bu durumu None ile taşır ve okuma satırı da onu None'a
    sadık basmalı.

    Gerekçenin uzun metni (`_NO_LUMO_NOTE`) 11.10.2026'dan beri okuma
    satırında da basılır (K1); o kilit
    `test_KOSULLU_ANALIZ_NOTU_kosulu_olusunca_OKUMA_SATIRINDA_yoksa_HIC`dedir.
    Burada aranan şey o değil: "yok" kelimesi kendini açıklar,
    `AnalysisReport`'un reddettiği şey 0.000000 BASMAKTI -- çünkü okuyan onu
    "boşluk yok, metalik" diye okurdu. Test o farkı sınıyor.

    Bağ ÇİFTİ olmayan tek atomlu bir sistem aynı zamanda "Mayer bağ
    mertebeleri:" başlığının boş basılmadığını da sınar.
    """
    del su_sonucu  # bu test kendi molekülünü kurar
    pencere = OrbitalWindow(solve(Molecule.from_string("He 0 0 0"), "sto-3g"))
    try:
        metin = pencere.analysis_text()
        assert "HOMO -0.876036" in metin, metin
        assert "LUMO yok" in metin, metin
        assert "aralık yok" in metin, metin
        assert "0.000000" not in metin, f"olmayan bir aralık sayı gibi basılmış:\n{metin}"
        assert "Mayer" not in metin, f"bağ çifti yokken boş liste basılmış:\n{metin}"
    finally:
        pencere.close()


_KOSULLU_NOTLAR = ("_CHARGED_DIPOLE_NOTE", "_NO_LUMO_NOTE", "_NO_HOMO_NOTE")


@requires_offscreen
@pytest.mark.parametrize(
    ("geometri", "yuk", "alan", "beklenen"),
    [
        # `main.py` yolu charge=0 kurar: GUI'den ulaşılan TEK koşullu not bu.
        ("He 0 0 0", 0, "orbital", ("_NO_LUMO_NOTE",)),
        # Aşağıdaki ikisine YALNIZ kurucudan ulaşılır (yüklü `HFResult`).
        ("He 0 0 0; H 0 0 0.774", 1, "orbital", ("_CHARGED_DIPOLE_NOTE",)),
        # Sıfır elektron: orbital alanında açılmaz (HOMO indeksi -1,
        # `InvalidInputError`), yoğunluk alanında açılır.
        (
            "H 0 0 0; H 0 0 0.74",
            2,
            "elektron yoğunluğu",
            ("_CHARGED_DIPOLE_NOTE", "_NO_HOMO_NOTE"),
        ),
        # Nötr ve LUMO'su var: HİÇBİR not basılmaz (koşulsuz uyarı mutantı).
        (_SU, 0, "orbital", ()),
        # Yüklü VE LUMO'suz (tek baz fonksiyonu, iki elektron): iki notun
        # SIRASINI kilitler. Doğrulayıcı (11.10.2026): yüklü-dipol ve
        # LUMO-yok notlarının yerini değiştiren mutant bu durum olmadan
        # sağ kalıyordu (hiçbir vaka ikisini birden taşımıyordu).
        ("H 0 0 0", -1, "orbital", ("_CHARGED_DIPOLE_NOTE", "_NO_LUMO_NOTE")),
    ],
    ids=["He-LUMO-yok", "HeH+-yuklu-dipol", "H2++-HOMO-yok", "su-not-yok", "H--yuklu-LUMO-yok"],
)
def test_KOSULLU_ANALIZ_NOTU_kosulu_olusunca_OKUMA_SATIRINDA_yoksa_HIC(
    geometri, yuk, alan, beklenen
):
    """K1 (Osman kararı 11.10.2026): `analyse()`in koşullu gerekçe notları
    koşulları oluştuğunda okuma satırında basılır.

    ÖLÇÜLDÜ (HEAD a9ec48a, kamu kurucusu): He/STO-3G "LUMO yok" basıyor ama
    `_NO_LUMO_NOTE` ne okuma satırında ne panelde; HeH+ 2.8368 D basıyor ve
    `_CHARGED_DIPOLE_NOTE` hiçbir yerde yok; H2++ (yoğunluk alanı) "HOMO yok"
    basıyor, `_NO_HOMO_NOTE` yok. Üç notun hiçbiri hiçbir pencerede yoktu.

    Testin ÖNCÜLÜ de sınanır: beklenen not kümesi `analyse()`in kendi
    kararıyla (raporun `expected_error`ı) AYNI olmalı -- pencere koşulu
    yeniden türetmez, raporu yansıtır. Sıra raporun sırasıdır.
    """
    from qlab.engine.hf import analysis

    sonuc = solve(Molecule.from_string(geometri, charge=yuk), "sto-3g")
    rapor_notlari = analysis.analyse(sonuc).provenance.expected_error
    for ad in _KOSULLU_NOTLAR:
        assert (getattr(analysis, ad) in rapor_notlari) == (ad in beklenen), (
            f"testin öncülü çürük: {ad} için motorun kararı farklı"
        )

    pencere = OrbitalWindow(sonuc, field=alan)
    try:
        metin = pencere.analysis_text()
        for ad in _KOSULLU_NOTLAR:
            not_ = getattr(analysis, ad)
            if ad in beklenen:
                assert f"Analiz notu: {not_}" in metin, f"{ad} okuma satırında yok:\n{metin}"
            else:
                assert not_ not in metin, f"koşulu olmayan {ad} basılmış:\n{metin}"
        konumlar = [metin.index(getattr(analysis, ad)) for ad in beklenen]
        assert konumlar == sorted(konumlar), "notlar raporun sırasında değil"
    finally:
        pencere.close()


# ---------------------------------------------------------------------------
# `free_atom_density_matrix` -- fark haritasının REFERANSI


@requires_offscreen
def test_fark_alani_serbest_atom_referansini_SAYIYLA_beyan_ediyor(su_sonucu):
    """Fark haritasının bütün anlamı referansına dayanır.

    `rho_molekul - rho_promolekul` haritasında "bağ" diye gösterilen şey,
    promolekülün ne olduğu bilinmeden okunamaz. Provenance bunu bugüne kadar
    yalnızca NESİR olarak söylüyordu ("SAD tipi, küresel ortalamalı serbest
    atom"); `free_atom_density_matrix` aynı şeyi KONTROL EDİLEBİLİR bir
    sayıyla söylemeyi mümkün kılıyor.

    ÖLÇÜLDÜ (STO-3G): O 5 baz fonksiyonu, Tr(PS) = 8.0000000000 (Z=8);
    H 1 baz fonksiyonu, Tr(PS) = 1.0000000000 (Z=1). Maliyet ihmal edilebilir:
    H 0.5 ms, O 19.3 ms.

    Beyan YALNIZCA fark alanında görünür: orbital ya da yoğunluk alanında
    promolekül diye bir şey yoktur ve oraya yazmak yalan olurdu.
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK

    pencere = OrbitalWindow(su_sonucu, field=ALAN_FARK)
    try:
        metin = pencere.transparency_text()
        assert "Tr(PS)" in metin, metin
        assert "O 5 baz fonksiyonu" in metin, metin
        assert "8.000000" in metin, metin
        assert "H 1 baz fonksiyonu" in metin, metin
        assert "1.000000" in metin, metin
        assert "Z=8" in metin and "Z=1" in metin, metin

        # Metnin yanında MAKİNE-OKUNUR bir toplam da durur ve komşusuyla
        # birlikte anlam kazanır: referansın taşıdığı elektron TAM 10.000000
        # (2x1 + 8), ızgara üzerinden integre edilen promolekül ise ölçüldü
        # 9.8907 -- aradaki fark doğrudan ızgaranın kestiği miktardır.
        beyan = pencere._provenance()
        assert beyan.diagnostics["promolecule_reference_electrons"] == pytest.approx(
            10.0, abs=1e-9
        )
        integre = beyan.diagnostics["integrated_electrons_promolecule"]
        assert 9.88 < integre < 9.90, integre
        assert integre < beyan.diagnostics["promolecule_reference_electrons"]

        for ad in (ALAN_ORBITAL, ALAN_YOGUNLUK):
            pencere.set_field_kind(ad)
            assert "Tr(PS)" not in pencere.transparency_text(), (
                f"{ad!r} alanında promolekül referansı beyan edilmemeli"
            )
            assert "promolecule_reference_electrons" not in (
                pencere._provenance().diagnostics
            ), f"{ad!r} alanında promolekül diye bir şey yok"
    finally:
        pencere.close()


def test_serbest_atom_onbellegi_element_VE_BAZ_ile_anahtarlaniyor():
    """Yalnız elemente göre anahtarlanan bir önbellek 6-31G ile sto-3g'yi karıştırır.

    Tr(PS) TEK BAŞINA bu kusuru görmez -- ölçüldü, iki bazda da aynı sayı
    (H 1.0000000000, O 8.0000000000), çünkü nötr atomun elektron sayısı baza
    bağlı değildir. Ayrımı YAPAN şey referansın BOYUTUDUR: ölçüldü,
    H sto-3g 1 / 6-31G 2 baz fonksiyonu, O sto-3g 5 / 6-31G 9.

    Önbelleğin GERÇEKTEN önbellek olduğu da yapısal olarak sınanır: hiç
    önbelleğe bakmayan bir uygulamada anahtar sorusunun bir anlamı kalmazdı.
    Önbellek artık pencereye aittir ve fonksiyona açıkça verilir (bkz.
    `test_serbest_atom_onbellegi_PENCEREYE_ait_pencereler_arasi_sizmiyor`);
    burada çıplak bir sözlük yeterli, global bir şey temizlenmez.
    """
    from qlab.app.orbital_window import _serbest_atom_referansi

    onbellek: dict = {}
    olculen = {
        (sembol, baz): _serbest_atom_referansi(sembol, baz, onbellek)
        for sembol in ("H", "O")
        for baz in ("sto-3g", "6-31g")
    }
    assert olculen[("H", "sto-3g")][0] == 1
    assert olculen[("H", "6-31g")][0] == 2
    assert olculen[("O", "sto-3g")][0] == 5
    assert olculen[("O", "6-31g")][0] == 9
    for anahtar, (_n, iz) in olculen.items():
        beklenen = 1.0 if anahtar[0] == "H" else 8.0
        assert iz == pytest.approx(beklenen, abs=1e-9), anahtar

    assert set(onbellek) == set(olculen), (
        "önbellek element+baz çiftiyle anahtarlanmıyor"
    )

    # YAPISAL SONDA: önbelleğe yabancı bir değer konur. Fonksiyon
    # önbelleği OKUYORSA onu döndürür.
    onbellek[("O", "sto-3g")] = (999, -1.0)
    assert _serbest_atom_referansi("O", "sto-3g", onbellek) == (999, -1.0), (
        "sonuç önbelleğe alınmıyor; aynı element iki kez çözülüyor"
    )


# ---------------------------------------------------------------------------
# B2 küçük açıkları (grup 7): kurucu sırası, alan önbelleği (3), serbest atom
# önbelleği (6), alan seçicisi (7)


_H2 = "H 0 0 0; H 0 0 0.74"


@requires_offscreen
@pytest.mark.parametrize(
    ("yuk", "argumanlar", "gerekce"),
    [
        pytest.param(0, {"points_per_axis": 1}, "points_per_axis", id="nokta-1"),
        pytest.param(0, {"padding": 0.0}, "padding", id="padding-0"),
        pytest.param(0, {"padding": -100.0}, "padding", id="padding-negatif"),
        # Sıfır elektron: HOMO indeksi -1, orbital alanı üretilemez.
        pytest.param(2, {}, "orbital indeksi", id="H2++-orbital"),
        pytest.param(0, {"level": 0.0}, "izoyüzey seviyesi", id="seviye-0"),
        pytest.param(0, {"field": "rho"}, "alan türü", id="alan-rho"),
    ],
)
def test_kurucu_reddettigi_girdide_YARIM_PENCERE_birakmiyor(yuk, argumanlar, gerekce):
    """Reddedilen HER girdi `super().__init__`'ten ÖNCE reddedilmeli.

    ÖLÇÜLDÜ (HEAD e262cab, kamu kurucusu, `parent` verilerek): seviye ve alan
    türü super()'den önce doğrulanıyordu (ebeveynde 0 QMainWindow kalıyor),
    ama `points_per_axis=1`, `padding=0.0`, `padding=-100` ve sıfır elektronlu
    H2++'nın orbital alanı super()'den SONRA reddediliyordu: dördünde de
    istisna fırlıyor AMA ebeveynin altında yarım kurulmuş 1 QMainWindow
    kalıyordu. Kurucunun kendi yorumu ("yarım kurulmuş bir QMainWindow
    ortada bırakılmaz") altı yolun yalnız ikisinde doğruydu.
    """
    sonuc = solve(Molecule.from_string(_H2, charge=yuk), "sto-3g")
    ebeveyn = QtWidgets.QWidget()
    try:
        with pytest.raises(InvalidInputError, match=gerekce):
            OrbitalWindow(sonuc, parent=ebeveyn, **argumanlar)
        kalan = ebeveyn.findChildren(QtWidgets.QMainWindow)
        assert kalan == [], f"reddedilen kurucu yarım pencere bıraktı: {kalan}"
    finally:
        ebeveyn.deleteLater()


@requires_offscreen
def test_alan_onbellegi_DURUMDAN_turuyor_ucuncu_bir_yol_bayatlatamiyor(su_sonucu):
    """(3) `_alan` önbelleği elle yazılmış iki geçersizleştirmeye bağlıydı.

    `set_field_kind` ve `set_orbital_index` alanı AÇIKÇA tazeliyordu; durumu
    değiştiren üçüncü bir yol bunu unutsa `_yenile` bayat alanı çizerdi.
    ÖLÇÜLDÜ (HEAD e262cab): `_orbital_index` 4'ten 0'a çekilip `_yenile`
    çağrılınca ağlar HOMO'nun ağları kalıyor, panel ise "orbital_index 0"
    basıyordu. Önbellek artık (alan türü, orbital indeksi) anahtarından
    türüyor: anahtar değişince alan yeniden üretilir, kimse hatırlamasa da.
    """
    from qlab.app.orbital_window import ALAN_YOGUNLUK, _YOGUNLUK

    pencere = OrbitalWindow(su_sonucu)
    referans = OrbitalWindow(su_sonucu)
    try:
        homo_sayilari = pencere.mesh_vertex_counts()
        referans.set_orbital_index(0)
        assert referans.mesh_vertex_counts() != homo_sayilari  # öncül

        pencere._orbital_index = 0  # açık tazeleme YOK
        pencere._yenile()
        assert pencere.mesh_vertex_counts() == referans.mesh_vertex_counts()

        referans.set_field_kind(ALAN_YOGUNLUK)
        referans.set_isosurface_level(0.05)
        pencere._alan_turu = _YOGUNLUK  # açık tazeleme YOK
        # Beyan da alanı okuyan bir yol: `_yenile`'den ÖNCE de güncel olmalı.
        assert pencere._provenance() == referans._provenance()
        pencere._yenile()
        assert pencere.mesh_vertex_counts() == referans.mesh_vertex_counts()
        assert len(pencere.mesh_vertex_counts()) == 1
    finally:
        pencere.close()
        referans.close()


@requires_offscreen
def test_yogunluk_alaninda_orbital_degisimi_alani_YENIDEN_URETMIYOR(su_sonucu):
    """Yoğunluk ve fark alanı orbital indeksine BAĞLI DEĞİLDİR.

    ÖLÇÜLDÜ (HEAD e262cab): fark alanında `set_orbital_index(3)` alanı
    yeniden üretiyordu (176.5 ms, yeni `ScalarField` nesnesi) ve çizilen
    yüzey bit birebir aynı kalıyordu -- boşa harcanan hesap.
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_YOGUNLUK

    for ad in (ALAN_YOGUNLUK, ALAN_FARK):
        pencere = OrbitalWindow(su_sonucu, field=ad)
        try:
            alan = pencere._alan
            pencere.set_orbital_index(3)
            assert pencere.orbital_index == 3
            assert pencere._alan is alan, f"{ad!r}: alan boşuna yeniden üretildi"
        finally:
            pencere.close()


@requires_offscreen
def test_yogunluk_alaninda_da_gecersiz_orbital_indeksi_reddediliyor(su_sonucu):
    """Sözleşme ALANDAN BAĞIMSIZ: geçersiz indeks reddedilir, durum korunur.

    ÖLÇÜLDÜ (HEAD e262cab): yoğunluk alanında `set_orbital_index(99)`
    KABUL EDİLİYORDU (orbital_index 99). Ardından `set_field_kind("orbital")`
    `InvalidInputError` fırlatıyor ama fırlatmadan önce pencerenin YARISINI
    değiştirmiş oluyordu: field_kind "orbital", seviye 0.05, kaydırıcı 1..900,
    seçici etkin -- ağlar ve alan ise hâlâ yoğunluğun (2196,), alan seçici
    hâlâ "elektron yoğunluğu".
    """
    from qlab.app.orbital_window import ALAN_ORBITAL, ALAN_YOGUNLUK

    pencere = OrbitalWindow(su_sonucu, field=ALAN_YOGUNLUK)
    try:
        once = _pencere_durumu(pencere)
        with pytest.raises(InvalidInputError, match="orbital indeksi"):
            pencere.set_orbital_index(99)
        assert pencere.orbital_index == 4
        assert _pencere_durumu(pencere) == once
        # SINIR (dogrulayici VB1, 12.10.2026): yalniz 99 sinaniyordu; `<=` mutanti
        # geciyordu. Aralik [0, n_orbital): n_orbital ve -1 red, n_orbital - 1 kabul.
        n_orbital = len(su_sonucu.orbital_energies)
        for gecersiz in (n_orbital, -1):
            with pytest.raises(InvalidInputError, match="orbital indeksi"):
                pencere.set_orbital_index(gecersiz)
            assert _pencere_durumu(pencere) == once
        pencere.set_orbital_index(n_orbital - 1)
        assert pencere.orbital_index == n_orbital - 1
        pencere.set_orbital_index(4)
        pencere.set_field_kind(ALAN_ORBITAL)
        assert pencere.mesh_vertex_counts() == (1764, 1764)
    finally:
        pencere.close()


@requires_offscreen
def test_alan_degisimi_URETILEMEYEN_alanda_pencereyi_YARIM_birakmiyor():
    """Sıfır elektronlu sistem orbital alanına GEÇEMEZ -- ve yarım da geçmez.

    ÖLÇÜLDÜ (HEAD e262cab, kamu kurucusu): H2++ yoğunluk alanında açılıyor
    (HOMO indeksi -1). `set_field_kind("orbital")` `InvalidInputError`
    fırlatıyordu ama önce `_alan_turu`, seviye, kaydırıcı aralığı ve seçicinin
    etkinliği değişmiş oluyordu. Yeni alan artık durum DEĞİŞMEDEN üretilir.
    Kullanıcı yolu da (alan seçici) sınanır: Qt istisnayı sinyal işleyicisinde
    yutar, seçici "orbital" gösterip pencere yoğunlukta kalırsa bu sessiz bir
    desenkrondur; seçici pencerenin gerçek alanına geri döner.
    """
    from qlab.app.orbital_window import ALAN_ORBITAL, ALAN_YOGUNLUK

    sonuc = solve(Molecule.from_string(_H2, charge=2), "sto-3g")
    pencere = OrbitalWindow(sonuc, field=ALAN_YOGUNLUK)
    try:
        def durum():
            return (
                *_pencere_durumu(pencere),
                pencere.field_kind,
                pencere._kaydirici.minimum(),
                pencere._kaydirici.maximum(),
                pencere._secici.isEnabled(),
                pencere._alan_secici.currentText(),
            )

        once = durum()
        with pytest.raises(InvalidInputError, match="orbital indeksi"):
            pencere.set_field_kind(ALAN_ORBITAL)
        assert durum() == once

        secici = pencere._alan_secici
        secici.setCurrentIndex(secici.findText(ALAN_ORBITAL))
        assert durum() == once, "alan seçici pencerenin gerçek alanına dönmedi"
    finally:
        pencere.close()


@requires_offscreen
def test_alan_secici_SIRADAN_degil_OGENIN_KENDI_VERISINDEN_okuyor(su_sonucu):
    """(7) Seçici satırı -> alan türü eşlemesi satırın KONUMUNA bağlıydı.

    `list(_ALAN_TURLERI)[index]` seçicideki sıra ile sözlük sırasının aynı
    olmasına dayanıyordu. ÖLÇÜLDÜ (HEAD e262cab): seçicinin modeli
    sıralandığında (Qt'nin kendi API'si) "orbital" yazan satır seçilince
    pencere "elektron yoğunluğu" gösteriyor, "elektron yoğunluğu" yazan satır
    seçilince "orbital". Her satır artık kendi alan türünü veri olarak taşır.
    """
    from qlab.app.orbital_window import ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK

    pencere = OrbitalWindow(su_sonucu)
    try:
        secici = pencere._alan_secici
        for i in range(secici.count()):
            assert secici.itemData(i) == secici.itemText(i)

        model = secici.model()
        secici.blockSignals(True)
        satir = model.takeRow(0)  # "orbital"
        model.insertRow(model.rowCount(), satir)  # ... sona
        secici.setCurrentIndex(secici.findText(ALAN_ORBITAL))
        secici.blockSignals(False)
        etiketler = [secici.itemText(i) for i in range(secici.count())]
        assert etiketler == [ALAN_YOGUNLUK, ALAN_FARK, ALAN_ORBITAL]  # öncül

        for ad in (ALAN_FARK, ALAN_ORBITAL, ALAN_YOGUNLUK):
            secici.setCurrentIndex(secici.findText(ad))
            assert pencere.field_kind == ad, f"{ad!r} satırı başka bir alan açtı"
        for ad in (ALAN_ORBITAL, ALAN_FARK):
            pencere.set_field_kind(ad)
            assert secici.currentText() == ad, "seçici pencereyle senkron değil"
    finally:
        pencere.close()


@requires_offscreen
def test_serbest_atom_onbellegi_PENCEREYE_ait_pencereler_arasi_sizmiyor(su_sonucu):
    """(6) Önbellek modül düzeyinde globaldi; artık her pencerenin kendisinin.

    ÖLÇÜLDÜ (HEAD e262cab): pencere kapandıktan sonra global önbellekte 2
    girdi kalıyor ve sonraki her pencere onları okuyordu; bozuk bir girdi
    (test sondası `(999, -1.0)`) bütün sonraki pencerelerin fark beyanına
    sızardı. Kaybedilen tek şey pencere başına bir kez ödenen hesap (O/STO-3G
    19.6 ms, H 0.4 ms) -- fark alanının kendi 170 ms'sinin yanında küçük.
    """
    from qlab.app import orbital_window
    from qlab.app.orbital_window import ALAN_FARK

    assert not hasattr(orbital_window, "_SERBEST_ATOM_ONBELLEGI")
    a = OrbitalWindow(su_sonucu, field=ALAN_FARK)
    b = OrbitalWindow(su_sonucu, field=ALAN_FARK)
    try:
        assert a._serbest_atom_onbellegi is not b._serbest_atom_onbellegi
        assert set(a._serbest_atom_onbellegi) == {("O", "sto-3g"), ("H", "sto-3g")}

        a._serbest_atom_onbellegi[("O", "sto-3g")] = (999, -1.0)
        a.set_isosurface_level(0.02)
        b.set_isosurface_level(0.02)
        assert "O 999 baz fonksiyonu" in a.transparency_text(), (
            "pencere kendi önbelleğini okumuyor"
        )
        assert "O 5 baz fonksiyonu, Tr(PS)=8.000000" in b.transparency_text()
        assert "O 999 baz fonksiyonu" not in b.transparency_text()
    finally:
        a.close()
        b.close()
