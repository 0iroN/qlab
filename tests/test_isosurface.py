"""İzoyüzey ağı çıkarımı.

Doğrulama fikri: bilinen analitik bir alanın izoyüzeyi bilinen bir şekildir.
`f(x,y,z) = x^2+y^2+z^2` alanının `level = r^2` izoyüzeyi, merkezi orijinde ve
yarıçapı `r` olan bir küredir. `p(x,y,z) = z*exp(-r^2)` alanı ise iki lobludur
ve işareti z ile birlikte döner.

**Neden hem izotropik hem anizotropik kutu var:** izotropik kutuda (üç eksende
aynı `spacing`) indeks->bohr dönüşümünde eksenleri karıştırmak yarıçapı hiç
değiştirmez -- ölçüldü, ortalama ve standart sapma **bit birebir** aynı çıkıyor
(0.9992148566441512 / 0.0003633044245967018, eleman bazında maks fark 1.11e-16).
Yani yalnız izotropik kutu kullanan bir test takımı bu hatayı göremez. Ve hata
kuramsal değil: `VolumeGrid.around` gerçekte anizotropik ızgara üretiyor
(H2, padding=6.0: spacing = [0.15189873, 0.15189873, 0.16959997] bohr, z ekseni
[-6.000, 7.398]). Bu yüzden aşağıda anizotropik bir küre testi var.

Ölçümler bu depoda pyqtgraph 0.14.0 / numpy 2.5.1 ile alındı.

`_kutu_izgarasi` `VolumeGrid`'i kurup eksenleri **ayrıca** `linspace` ile
üretiyor; bu bilinçli bir tekrardır. `grid.axes` kullanılsaydı alan ile dönüşüm
aynı formülü paylaşır, test kendi kendini onaylardı. İki yolun farkı ölçüldü:
maks 3.553e-15 bohr (anizotropik kutuda eksenlere göre 3.553e-15 / 1.776e-15 /
7.105e-15) -- buradaki 0.05 mertebesindeki sınırların yanında sıfırdır.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.observable.hf_obs import ScalarField, VolumeGrid
from qlab.provenance import Provenance
from qlab.viz.isosurface import Mesh, extract_isosurface, extract_signed_pair

_TEST_PROVENANCE = Provenance(
    method="test",
    approximations=(),
    neglected=(),
    expected_error="test",
    converged=True,
)

#: Planın kullandığı izotropik kutu: üç eksende de spacing = 0.1 bohr.
_IZOTROPIK = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))

#: Anizotropik kutu: spacing = 0.1 / 0.05 / 0.2 bohr (aynı nokta sayısı,
#: farklı kenar uzunlukları). Eksen karışıklığını görünür kılan yapılandırma.
_ANIZOTROPIK = ((-2.0, 2.0), (-1.0, 1.0), (-4.0, 4.0))


def _kutu_izgarasi(
    sinirlar: tuple[tuple[float, float], ...] = _IZOTROPIK, npts: int = 41
) -> tuple[VolumeGrid, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    eksenler = tuple(np.linspace(dusuk, yuksek, npts) for dusuk, yuksek in sinirlar)
    grid = VolumeGrid(
        origin=np.array([dusuk for dusuk, _ in sinirlar], dtype=float),
        spacing=np.array([float(eksen[1] - eksen[0]) for eksen in eksenler]),
        shape=(npts, npts, npts),
    )
    return grid, eksenler


def _kure_alani(
    sinirlar: tuple[tuple[float, float], ...] = _IZOTROPIK, npts: int = 41
) -> ScalarField:
    grid, eksenler = _kutu_izgarasi(sinirlar, npts)
    x, y, z = np.meshgrid(*eksenler, indexing="ij")
    return ScalarField(
        values=x * x + y * y + z * z, grid=grid, provenance=_TEST_PROVENANCE
    )


def _p_degeri(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """p tipi analitik alan: z*exp(-r^2). İki loblu, z ile işaret değiştirir."""
    return z * np.exp(-(x * x + y * y + z * z))


def _p_alani(
    sinirlar: tuple[tuple[float, float], ...] = _IZOTROPIK, npts: int = 41
) -> ScalarField:
    grid, eksenler = _kutu_izgarasi(sinirlar, npts)
    x, y, z = np.meshgrid(*eksenler, indexing="ij")
    return ScalarField(
        values=_p_degeri(x, y, z), grid=grid, provenance=_TEST_PROVENANCE
    )


def test_kurenin_izoyuzeyi_dogru_yaricapta():
    """Tepe noktalarının orijinden uzaklığı 1.0 olmalı -- BOHR biriminde.

    İki assert FARKLI hataları yakalar, ikisi de gerekli:

    - `mean == approx(1.0, abs=0.05)` -> **indeks/bohr dönüşümünün atlanmasını**
      yakalar. Ölçüldü: dönüşüm atlanırsa (ham indeks tepe noktaları) bu testin
      gördüğü ortalama **35.600986** (std 5.693233) olur, çünkü küre indeks
      uzayında (20,20,20) merkezlidir ve orijinden uzaklık ~sqrt(3)*20'dir.
      Sınır 35x aşılır. (Küre kendi merkezine göre ölçülseydi indeks yarıçapı
      9.992149 çıkardı -- ama bu testin baktığı büyüklük o değil.)
      Gerçek ayrıklaştırma hatası yalnızca 7.851e-4 (1 - 0.9992148566441512),
      yani sınır bu iş için 64x gevşek; testin işi hassasiyet ölçmek DEĞİL.
    - `std < 0.05` -> **origin'in bir hücre kaymasını** yakalar. Ölçüldü: origin
      her eksende bir hücre kaydırılınca ortalama 1.009220 olur (sınırdan
      GEÇER) ama std 0.099529 olur (sınırda KALIR, 2x marj). Kaymayı yakalayan
      tek assert std'dir.

    Bu testin yakalayamadığı hata: eksen takası. Kutu izotropik olduğu için
    takas her iki büyüklüğü de bit birebir aynı bırakıyor -- bkz. bir alttaki
    anizotropik test.
    """
    mesh = extract_isosurface(_kure_alani(), level=1.0)
    yaricaplar = np.linalg.norm(mesh.vertices, axis=1)
    assert yaricaplar.mean() == pytest.approx(1.0, abs=0.05, rel=0.0)
    assert yaricaplar.std() < 0.05
    assert mesh.level == 1.0
    assert mesh.sign == 1


def test_anizotropik_izgarada_eksenler_karismiyor():
    """Eksen başına farklı `spacing` -- indeks->bohr eşleşmesi burada görünür.

    Kutu: x [-2,2], y [-1,1], z [-4,4], her eksende 41 nokta, yani
    spacing = 0.1 / 0.05 / 0.2 bohr. Aynı küre alanı, aynı level=1.0.

    Ölçülen değerler (2126 tepe, 4248 üçgen):

    | dönüşüm                      | yarıçap ort | std      |
    |------------------------------|-------------|----------|
    | DOĞRU eşleşme                | 0.997814    | 0.001787 |
    | x<->z takası                 | 1.232725    | 0.385088 |
    | origin[2] += spacing[2]      | 1.010128    | 0.123603 |
    | spacing üç eksende spacing[0] | 2.512488   | 0.515653 |
    | origin üç eksende origin[0]  | 2.377641    | 0.585200 |

    İzotropik testte takas hiçbir şeyi değiştirmiyordu; burada ortalama sınırı
    4.7x, std sınırı 7.7x aşıyor. Sınırlar izotropik testle bilinçli olarak
    aynı tutuldu ki iki test doğrudan karşılaştırılabilsin; doğru dönüşümdeki
    paylar ortalamada 23x (sapma 2.186e-3), std'de 28x'tir.
    """
    mesh = extract_isosurface(_kure_alani(_ANIZOTROPIK), level=1.0)
    yaricaplar = np.linalg.norm(mesh.vertices, axis=1)
    assert yaricaplar.mean() == pytest.approx(1.0, abs=0.05, rel=0.0)
    assert yaricaplar.std() < 0.05


def test_izoyuzey_ucgenlerden_olusuyor():
    mesh = extract_isosurface(_kure_alani(), level=1.0)
    assert mesh.faces.shape[1] == 3
    assert mesh.faces.max() < len(mesh.vertices)
    assert len(mesh.faces) > 100


def test_daha_yuksek_seviye_daha_buyuk_kure():
    """Ölçülen yarıçaplar: level 0.25 -> 0.498417, level 2.25 -> 1.499455."""
    kucuk = extract_isosurface(_kure_alani(), level=0.25)
    buyuk = extract_isosurface(_kure_alani(), level=2.25)
    assert np.linalg.norm(kucuk.vertices, axis=1).mean() < np.linalg.norm(
        buyuk.vertices, axis=1
    ).mean()


def test_isaretli_cift_iki_ayri_ag_veriyor():
    """p tipi bir orbital iki lobludur; +faz ve -faz ayrı çizilmelidir.

    Ölçüldü (izotropik kutu, alan aralığı -0.428838 .. +0.428838): her iki ağ
    da 3102 tepe taşıyor, z ortalamaları +0.875716 ve -0.875716.
    """
    pozitif, negatif = extract_signed_pair(_p_alani(), level=0.05)
    assert pozitif.sign == 1
    assert negatif.sign == -1
    assert pozitif.level == 0.05
    assert negatif.level == 0.05
    assert len(pozitif.vertices) > 0
    assert len(negatif.vertices) > 0
    assert pozitif.vertices[:, 2].mean() > 0
    assert negatif.vertices[:, 2].mean() < 0


def test_isaretli_ciftin_tepe_noktalarinda_alan_degeri_sign_carpi_level():
    """Modülün `alan == sign * level` sözünü sayısal olarak sınar.

    Tepe noktaları bohr'a çevrildiğine göre analitik alan onların üzerinde
    doğrudan hesaplanabilir; interpolasyona gerek yok. Anizotropik kutuda
    ölçüldü (3076 tepe): pozitif ağda ortalama +0.05017837 (std 9.7336e-4),
    negatif ağda -0.05017837; hedeften **maksimum** sapma 2.05850e-3.
    Sınır 5e-3 seçildi -- ölçülen sapmanın 2.4 katı.

    Bu, dosyadaki en keskin assert: x<->z takası uygulandığında maksimum sapma
    0.47144081'e fırlıyor, yani sınır 94x aşılıyor. (İzotropik kutuda bile
    yakalıyor: doğru dönüşümde maks sapma 4.7693e-4, takasta 0.47593410 --
    çünkü p alanı, kürenin aksine, küresel simetrik değil.)
    """
    field = _p_alani(_ANIZOTROPIK)
    pozitif, negatif = extract_signed_pair(field, level=0.05)
    for mesh in (pozitif, negatif):
        x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
        sapma = np.abs(_p_degeri(x, y, z) - mesh.sign * mesh.level)
        assert sapma.max() < 5e-3


def test_alanda_bulunmayan_seviye_bos_ag_veriyor():
    """Sessizce çökmek yerine boş ağ dönmeli.

    DÜRÜSTLÜK NOTU: bu davranış bu modülün eklediği bir koruma DEĞİL,
    `pg.isosurface`'in kendi davranışıdır -- alan aralığının dışındaki bir
    seviye için `(0, 3)` şekilli `float32` tepe dizisi ve `(0, 3)` şekilli
    `uint32` yüz dizisi döndürüyor, istisna atmıyor (ölçüldü). Test bu
    bağımlılık davranışını kilitler; pyqtgraph bir gün istisna atmaya
    başlarsa burada haber alırız.
    """
    mesh = extract_isosurface(_kure_alani(), level=1000.0)
    assert len(mesh.vertices) == 0
    assert len(mesh.faces) == 0
    assert mesh.vertices.shape == (0, 3)
    assert mesh.faces.shape == (0, 3)


def test_ag_dizileri_gercekten_salt_okunur():
    """`@dataclass(frozen=True)` numpy dizisinin İÇERİĞİNİ korumaz.

    frozen yalnızca alan yeniden bağlamayı (`mesh.vertices = ...`) engeller;
    `mesh.vertices[0, 0] = ...` frozen bir dataclass'ta da çalışır. `Mesh`
    bu yüzden dizileri kopyalayıp `setflags(write=False)` uyguluyor, ve
    dokstring'i bunu vaat ediyor -- vaat burada kilitleniyor.

    Kopyanın gerekliliği ayrıca ölçüldü: kopyasız salt-okunur bir `view`
    alınsaydı, çağıran taraf hâlâ yazılabilir kaynağı tuttuğu için
    `view` üzerinden okunan değer dışarıdan değiştirilebiliyordu.
    """
    mesh = extract_isosurface(_kure_alani(), level=1.0)
    with pytest.raises(ValueError):
        mesh.vertices[0, 0] = 0.0
    with pytest.raises(ValueError):
        mesh.faces[0, 0] = 0

    disaridan = np.zeros((2, 3), dtype=float)
    elle = Mesh(
        vertices=disaridan, faces=np.zeros((1, 3), dtype=np.uint32), level=1.0, sign=1
    )
    disaridan[0, 0] = 7.0
    assert elle.vertices[0, 0] == 0.0
