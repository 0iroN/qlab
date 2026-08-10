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
maks **7.105e-15** bohr (anizotropik kutuda eksenlere göre 3.553e-15 /
1.776e-15 / 7.105e-15; izotropik kutuda üç eksende de 3.553e-15) -- buradaki
0.05 mertebesindeki sınırların yanında sıfırdır.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.errors import InvalidInputError
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


def _p_gradyani(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """`_p_degeri`nin analitik gradyanı, (n, 3).

    p = z*exp(-r^2) olduğuna göre
    dp/dx = -2xz*exp(-r^2), dp/dy = -2yz*exp(-r^2),
    dp/dz = (1 - 2z^2)*exp(-r^2).

    Sayısal gradyan bilinçli olarak KULLANILMIYOR: `np.gradient` alanı ızgara
    üzerinden türetirdi, yani ağın geldiği aynı ayrıklaştırmayı paylaşırdı.
    Analitik gradyan bağımsız bir referanstır.
    """
    e = np.exp(-(x * x + y * y + z * z))
    return np.stack([-2 * x * z * e, -2 * y * z * e, (1 - 2 * z * z) * e], axis=1)


def _ucgen_normalleri(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Ağın üçgen normallerini ve ağırlık merkezlerini döndürür, ((k,3), (k,3)).

    Normal `(b-a) x (c-a)`; yönü doğrudan `faces` satırındaki tepe SIRASINDAN,
    yani sarımdan gelir. Normalleştirilmez -- uzunluğu üçgen alanının iki
    katıdır ve dejenere (sıfır alanlı) üçgenleri ayıklamak için kullanılıyor.
    """
    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=int)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.cross(b - a, c - a), (a + b + c) / 3.0


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
      gördüğü ortalama **35.600985** (std 5.693233) olur, çünkü küre indeks
      uzayında (20,20,20) merkezlidir ve orijinden uzaklık ~sqrt(3)*20'dir.
      Hedeften sapma 34.600985, sınır 0.05, yani sınır **692x** aşılır --
      dosyanın geri kalanındaki "94x / 4.7x / 7.7x" ile aynı sapma/sınır
      konvansiyonu. (Küre kendi merkezine göre ölçülseydi indeks yarıçapı
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


def test_sarim_normali_alanin_azaldigi_yone_ceviriyor():
    """`pg.isosurface`'in SARIM sözleşmesini kilitler.

    Bir üçgenin normali `faces` satırındaki tepe sırasından gelir; sıra ters
    çevrilirse normal 180 derece döner. `pg.isosurface` sarımı, normal
    **girdi dizisinin AZALDIĞI** yöne bakacak şekilde üretiyor. Burada küre
    alanı (x^2+y^2+z^2) kullanılıyor: alan dışarı doğru ARTAR, dolayısıyla
    doğru sarımda normal MERKEZE bakmalıdır (geometrik olarak "içeri").

    Bu, `extract_signed_pair`'in `-values` numarasının dayandığı sözleşmedir;
    bkz. o fonksiyonun dokstring'i.

    Ölçüldü: normal ile gradyan (2*r) arasındaki kosinüs, izotropik kutuda
    3656 üçgenin ortalaması -0.9865, anizotropikte 4248 üçgenin ortalaması
    -0.9876; ikisinde de dejenere olmayan üçgenlerin **tamamı** negatif.

    DEJENERE ÜÇGENLER: küre ağında 48 üçgenin alanı tam olarak sıfır (izotropik
    kutuda 3656'nın %1.31'i, anizotropikte 4248'in %1.13'ü). Sıfır alanlı bir
    üçgenin normali sıfır vektördür, yani YÖNÜ YOKTUR -- bunlar ayıklanmadan
    "tüm üçgenler" denemez. Ölçüldü: kosinüsü negatif olmayan üçgen kümesi ile
    dejenere üçgen kümesi TAM OLARAK aynı (dejenere olmayan ve d == 0 olan
    üçgen sayısı: 0). Bu üçgenler görüntüyü de bozmaz: sıfır normal, tepe
    normali toplamına sıfır katkı verir. p alanının ağında hiç dejenere üçgen
    yok (0 / 6200 ve 0 / 6040).
    """
    for sinirlar in (_IZOTROPIK, _ANIZOTROPIK):
        mesh = extract_isosurface(_kure_alani(sinirlar), level=1.0)
        normaller, merkezler = _ucgen_normalleri(mesh)
        dejenere = np.linalg.norm(normaller, axis=1) == 0.0
        gradyan = 2.0 * merkezler  # grad(x^2+y^2+z^2)
        izdusum = np.sum(normaller * gradyan, axis=1)

        assert dejenere.mean() < 0.05
        assert np.all(izdusum[~dejenere] < 0.0)


def test_isaretli_ciftte_iki_lobun_da_normali_disari_bakiyor():
    """`-values` numarasının ASIL sebebi: iki lobun da normali DIŞARI baksın.

    `extract_signed_pair` negatif ağı `pg.isosurface(-values, +level)` ile
    çıkarıyor. Bunun `pg.isosurface(values, -level)` ile "eşdeğer" olduğu
    sanılabilir -- TEPELER için doğrudur (ölçüldü: her iki yol da 3076 tepe,
    `np.array_equal` True) ama YÜZLER için sarım terstir (6040 yüzün her biri,
    örneğin [58, 53, 0] karşı [58, 0, 53]).

    Fark görüntüye yansır. Bir lobun içinden dışına çıkarken `sign*p` azalır,
    dolayısıyla normalin dışarı bakması `normal . grad(sign*p) < 0` demektir.
    Ölçüldü:

    | negatif ağ nasıl çıkarılıyor      | dışarı-bakan üçgen oranı |
    |-----------------------------------|--------------------------|
    | `-values, +level` (mevcut kod)    | 1.000                    |
    | `values, -level` (sadeleştirme)   | 0.000                    |

    Yani "sadeleştirme" negatif lobu komple iç-dışa çevirir. Tüketici
    `app/orbital_window.py` yüzleri `gl.MeshData`ya verip `GLMeshItem`'ı
    `smooth=True, shader="balloon"` ile çiziyor; ikisi de sarımdan türetilen
    tepe normallerine bağlı.

    Marj: kosinüs ortalaması -0.9986, **en kötü** üçgende -0.9838
    (anizotropik kutu, 6040 üçgen); izotropikte en kötü -0.9930. Sınır 0.0,
    yani fiziksel olarak anlamlı olan işaret değişimi -- gevşetilmiş bir eşik
    değil. Dejenere üçgen yok, ayıklama gerekmiyor.
    """
    for sinirlar in (_IZOTROPIK, _ANIZOTROPIK):
        pozitif, negatif = extract_signed_pair(_p_alani(sinirlar), level=0.05)
        for mesh in (pozitif, negatif):
            normaller, merkezler = _ucgen_normalleri(mesh)
            assert np.all(np.linalg.norm(normaller, axis=1) > 0.0)
            gradyan = (
                _p_gradyani(merkezler[:, 0], merkezler[:, 1], merkezler[:, 2])
                * mesh.sign
            )
            assert np.all(np.sum(normaller * gradyan, axis=1) < 0.0)


def test_ucgenler_tek_bir_izgara_hucresinin_icinde_kaliyor():
    """Ağın TOPOLOJİSİNİ kilitler: `faces` gerçekten komşu tepeleri bağlıyor mu?

    `test_izoyuzey_ucgenlerden_olusuyor` bu işi YAPMIYOR: onun üç assert'i de
    (`shape[1]==3`, `faces.max() < len(vertices)`, `len(faces)>100`) tepe
    dizisinin herhangi bir PERMÜTASYONU altında değişmez, yani yüzler tamamen
    yanlış tepeleri gösterse bile geçer.

    Buradaki sınır kuramdan geliyor, veriden uydurulmadı: marching cubes her
    tepeyi bir ızgara hücresinin KENARI üzerine koyar ve her üçgen TEK bir
    hücrenin içinde kalır. Dolayısıyla hiçbir üçgen kenarı o hücrenin
    köşegeninden (`norm(spacing)`) uzun olamaz. Bu bir kalibrasyon değil,
    geometrik bir üst sınır.

    Ölçülen maks kenar / köşegen oranı (sınır 1.0):

    | yapılandırma                   | oran  |
    |--------------------------------|-------|
    | p anizotropik, her iki lob     | 0.989 |
    | p izotropik                    | 0.987 |
    | küre anizotropik               | 0.988 |
    | küre izotropik, level=1.00     | 0.978 |
    | küre izotropik, level=0.25     | 0.955 |
    | küre izotropik, level=2.25     | 0.962 |

    Bozuk indekslemeye duyarlılık (p anizotropik, köşegen 0.2291 bohr):
    tepe dizisi `[::-1]` ile ters çevrilip yüzler yeniden indekslenmezse maks
    kenar 0.2267 -> 2.4611 bohr olur, yani sınır **10.7x** aşılır; sadece bir
    adım kaydırmak (`np.roll(v, 1)`) bile 2.9226 bohr, **12.8x** verir.
    """
    olcumler = [
        (_p_alani(_ANIZOTROPIK), 0.05, True),
        (_p_alani(_IZOTROPIK), 0.05, True),
        (_kure_alani(_ANIZOTROPIK), 1.0, False),
        (_kure_alani(_IZOTROPIK), 1.0, False),
        (_kure_alani(_IZOTROPIK), 0.25, False),
        (_kure_alani(_IZOTROPIK), 2.25, False),
    ]
    for field, level, isaretli in olcumler:
        agler = (
            extract_signed_pair(field, level=level)
            if isaretli
            else (extract_isosurface(field, level=level),)
        )
        kosegen = float(np.linalg.norm(np.asarray(field.grid.spacing, dtype=float)))
        for mesh in agler:
            v = np.asarray(mesh.vertices, dtype=float)
            f = np.asarray(mesh.faces, dtype=int)
            a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            kenarlar = np.concatenate(
                [
                    np.linalg.norm(b - a, axis=1),
                    np.linalg.norm(c - b, axis=1),
                    np.linalg.norm(a - c, axis=1),
                ]
            )
            assert kenarlar.max() <= kosegen


def test_c_bitisik_olmayan_alan_da_kabul_ediliyor():
    """`np.ascontiguousarray` koruması -- ve bugün neden LATENT olduğu.

    `pg.isosurface` girdinin c-bitişik olmasını şart koşuyor, aksi halde
    `TypeError: isosurface input data must be c-contiguous.` atıyor (ölçüldü).
    `_ag_cikar` bu yüzden `np.ascontiguousarray` kullanıyor.

    DÜRÜSTLÜK NOTU: bu koruma bu depoda BUGÜN ölü koddur. `ScalarField.values`
    burada her zaman float64 üretiliyor ve float64 -> float32 dökümü her
    halükârda yeni, bitişik bir kopya doğuruyor -- yani `np.asarray` bile
    yeterdi. Koruma yalnızca `values` ZATEN float32 ve bitişik DEĞİLSE iş
    yapar: o zaman `np.asarray(..., dtype=np.float32)` hiçbir kopya almaz ve
    bitişik olmayan diziyi olduğu gibi geçirir (ölçüldü, C_CONTIGUOUS False
    kalıyor ve pyqtgraph TypeError atıyor).

    `ScalarField` bir veri taşıyıcısı; dtype doğrulaması yapmıyor, dolayısıyla
    böyle bir alan kurulabilir ve bu test onu kuruyor. Test, korumanın
    sözleşmesini kilitler; bugünkü üreticilerin onu tetiklememesi, sözleşmenin
    yanlış olduğu anlamına gelmez.
    """
    kure = _kure_alani()
    genis = np.zeros((41, 41, 82), dtype=np.float32)
    genis[:, :, ::2] = np.asarray(kure.values, dtype=np.float32)
    bitisiksiz = genis[:, :, ::2]
    assert bitisiksiz.dtype == np.float32
    assert not bitisiksiz.flags["C_CONTIGUOUS"]

    field = ScalarField(values=bitisiksiz, grid=kure.grid, provenance=_TEST_PROVENANCE)
    mesh = extract_isosurface(field, level=1.0)
    assert len(mesh.vertices) > 0
    assert np.linalg.norm(mesh.vertices, axis=1).mean() == pytest.approx(
        1.0, abs=0.05, rel=0.0
    )


def test_mesh_bozuk_bicimi_reddediyor():
    """`Mesh` dokstring'inin BİÇİM sözünü kilitler.

    Dokstring "vertices: (m,3)", "faces: (k,3)", "sign: +1 ya da -1" diyor.
    Bu bir belge değil sözleşmedir: `app/orbital_window.py` `faces`'i doğrudan
    `gl.MeshData`ya veriyor, yanlış biçim orada OpenGL katmanında patlar --
    yani hatanın kaynaktan uzakta görüldüğü tipik durum. Doğrulama `Mesh`'in
    kendisinde olmalı.
    """
    with pytest.raises(InvalidInputError):
        Mesh(vertices=np.zeros((4, 7)), faces=np.zeros((2, 3)), level=1.0, sign=1)
    with pytest.raises(InvalidInputError):
        Mesh(vertices=np.zeros((4, 3)), faces=np.zeros((2, 9)), level=1.0, sign=1)
    with pytest.raises(InvalidInputError):
        Mesh(vertices=np.zeros(3), faces=np.zeros((2, 3)), level=1.0, sign=1)
    with pytest.raises(InvalidInputError):
        Mesh(vertices=np.zeros((4, 3)), faces=np.zeros((2, 3)), level=1.0, sign=5)
    with pytest.raises(InvalidInputError):
        Mesh(vertices=np.zeros((4, 3)), faces=np.zeros((2, 3)), level=1.0, sign=0)

    # Boş ağ geçerlidir: pg.isosurface bulunmayan seviye için (0,3) döndürüyor.
    bos = Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)), level=1.0, sign=-1)
    assert len(bos.vertices) == 0
