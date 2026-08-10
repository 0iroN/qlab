"""Skaler alanlardan izoyüzey ağı çıkarımı.

Bir orbital ya da yoğunluk alanı ızgara üzerinde sayı yığınıdır; ekranda
görünen şey onun bir `level` değerindeki izoyüzeyidir. Ağ çıkarımı
`pyqtgraph.isosurface` (marching cubes, saf NumPy, OpenGL gerektirmez) ile
yapılır.

**Birim dönüşümü bu modülün asıl işidir.** `pg.isosurface` tepe noktalarını
ızgara **indeks** uzayında döndürür (0..shape-1 arası kesirli indeksler).
Buradaki `Mesh.vertices` ise **bohr** cinsindendir; dönüşüm
`origin + spacing * indeks` ile eksen eksen yapılır. Dönüşüm atlanırsa
görüntü sessizce yanlış ölçekte çıkar: ölçüldü, [-2,2]^3 kutusunda 41 noktalı
bir küre alanında (level=1.0) tepe noktalarının orijinden uzaklığı 1.0 yerine
35.600985 olur.

Dönüşüm eksen eksen yapılmak zorundadır çünkü `VolumeGrid` genelde
anizotropiktir: `VolumeGrid.around(H2, padding=6.0)` ölçüldüğünde
spacing = [0.15189873, 0.15189873, 0.16959997] bohr çıkıyor.

`pg.isosurface` girdinin **c-bitişik** olmasını şart koşuyor (aksi halde
`TypeError: isosurface input data must be c-contiguous.`). Buradaki
`np.ascontiguousarray` koruması BUGÜN LATENT'tir: `ScalarField.values` bu
depoda her zaman float64 ve float64 -> float32 dökümü zaten yeni, bitişik bir
kopya üretiyor, yani `np.asarray` de yeterdi. Koruma yalnızca `values` ZATEN
float32 ve bitişik DEĞİLKEN iş yapar; `ScalarField` dtype doğrulaması
yapmadığı için bu kurulabilir bir durumdur ve testle kilitlidir
(`test_c_bitisik_olmayan_alan_da_kabul_ediliyor`). float32 şartı
DEĞİL -- ölçüldü, c-bitişik float64 de kabul ediliyor. Yine de float32'ye
döküyoruz: büyük ızgaralarda belleği yarıya indiriyor ve ölçülen fark
görselleştirme için önemsiz (aynı küre için yarıçap 0.999215 yerine 0.999225,
1830 yerine 1854 tepe).

Orbital genliği işaretlidir; `extract_signed_pair` `+level` ve `-level` için
iki ayrı ağ verir, böylece arayüz ±fazı iki renkle çizebilir.

**`Mesh` kendi `Provenance`'ını taşımaz, ve bu bir eksiklikti.** Ağ her zaman
bir `ScalarField`den doğar; o alanın beyanı MOTORUNdur ve ağı çıkarırken
yapılan iki seçim (float32 dökümü, marching-cubes'un doğrusal interpolasyonu)
orada bulunmaz. Bir süre bu iki seçim yalnızca bu dokstring'de yaşadı, yani
ekranda hiçbir yerde görünmüyordu. Metinleri artık
`EXTRACTION_APPROXIMATIONS` taşıyor ve tüketici (`app/orbital_window.py`)
onları kendi panelinin beyanına ekliyor. Beyan burada durur çünkü seçim
burada yapılıyor. `Mesh`'e bir `provenance` alanı EKLENMEDİ: ağ hiçbir zaman
tek başına değil, her zaman onu doğuran alanla birlikte tüketiliyor; ağa da
ayrı bir beyan takmak aynı panelde iki `Provenance` doğurur ve hangi maddenin
hangi katmandan geldiği bulanıklaşırdı.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg

from qlab.errors import InvalidInputError
from qlab.observable.hf_obs import ScalarField, VolumeGrid

#: Ağ çıkarımının GÖRÜNTÜYE yansıyan ve motorun beyanında BULUNMAYAN iki
#: seçimi. Tüketici bunları kendi `Provenance`'ına ekler.
#:
#: Metinler bilinçli olarak KISA: şeffaflık paneli sabit yükseklikte ve
#: ölçüldü, uzun yazılmış hâlleri sarılmış satır sayısını 19'dan 25'e
#: çıkarıp gizli kalan satırı 7'den 13'e taşıyordu. Ölçülen büyüklükler
#: yukarıdaki modül dokstring'inde tam hâliyle duruyor; panele yalnızca
#: seçimin ADI ve mertebesi gidiyor.
EXTRACTION_APPROXIMATIONS: tuple[str, ...] = (
    "izoyüzey ağı float32 alandan çıkarılır (kürede yarıçap 0.999215/0.999225)",
    "ağın köşeleri marching cubes'un doğrusal interpolasyonundan gelir",
)


@dataclass(frozen=True)
class Mesh:
    """Bir izoyüzeyin üçgen ağı.

    - `vertices`: (m, 3), **bohr** biriminde tepe konumları.
    - `faces`: (k, 3), her satır `vertices` içine üç indeks (bir üçgen).
      Satırdaki **SIRA anlamlıdır**: üçgenin sarımını, dolayısıyla normalinin
      yönünü belirler. Sözleşme `extract_signed_pair`'in dokstring'inde.
    - `level`: izoyüzeyin seviyesi, alanın kendi biriminde.
    - `sign`: bu ağın hangi fazı temsil ettiği, +1 ya da -1.

    Bu iddialar **doğrulanır**, yalnızca belgelenmez: (n, 3) olmayan bir
    `vertices`/`faces`, ±1 olmayan bir `sign` ya da sonlu olmayan bir `level`
    `InvalidInputError` verir. Doğrulama burada olmalı çünkü tüketici
    `app/orbital_window.py` `faces`'i doğrudan `gl.MeshData`ya veriyor; bozuk
    biçim orada, kaynaktan uzakta, OpenGL katmanında patlardı.

    `level` bir süre doğrulanmıyordu, kardeşi `sign` doğrulanırken: ölçüldü,
    `Mesh(vertices=(2,3), faces=(1,3), level=nan, sign=1)` istisnasız
    kuruluyordu. Boş olmayan böyle bir ağda aşağıdaki sözleşme -- "alan ==
    sign * level" -- anlamsızlaşır, çünkü `nan` ile hiçbir eşitlik doğru
    değildir. `level` ayrıca tüketiciye taşınan ve panele basılan bir sayıdır.
    İşaret kısıtı YOKTUR: negatif `level`'ın belgelenmiş bir anlamı var
    (`extract_signed_pair`, roller yer değiştirir).

    Sözleşme: ağ üzerindeki her nokta için **alan == sign * level**.
    `extract_isosurface` her zaman `sign = +1` bildirir, yani orada
    alan == level'dir. `extract_signed_pair` negatif ağ için `sign = -1`
    verir; o ağ alanın `-level` izoyüzeyidir. Genlik (`level`) ile faz
    (`sign`) böylece ayrı taşınır ve tekrar etmez.

    **Değişmezlik:** `frozen=True` yalnızca alan yeniden bağlamayı engeller,
    numpy dizisinin İÇERİĞİNİ korumaz -- `mesh.vertices[0] = ...` sıradan bir
    frozen dataclass'ta çalışırdı. Bu yüzden `Mesh` verilen dizileri
    **kopyalayıp** salt-okunur işaretler; yazma denemesi `ValueError` verir ve
    çağıranın elindeki dizide yapılan sonraki değişiklikler ağa sızmaz.
    Kopya bilinçli: kopyasız salt-okunur bir `view`, kaynağı hâlâ yazılabilir
    tutan çağıranın değişikliklerini geçiriyordu.
    """

    vertices: np.ndarray
    faces: np.ndarray
    level: float
    sign: int

    def __post_init__(self) -> None:
        for alan in ("vertices", "faces"):
            dizi = np.array(getattr(self, alan), copy=True)
            if dizi.ndim != 2 or dizi.shape[1] != 3:
                raise InvalidInputError(
                    f"Mesh.{alan} (n, 3) seklinde olmali "
                    f"(her satir bir nokta ya da bir ucgen), "
                    f"verilen sekil: {dizi.shape}"
                )
            dizi.setflags(write=False)
            object.__setattr__(self, alan, dizi)
        if self.sign not in (1, -1):
            raise InvalidInputError(
                f"Mesh.sign yalnizca +1 ya da -1 olabilir (agin fazi); "
                f"verilen: {self.sign!r}"
            )
        if not math.isfinite(float(self.level)):
            raise InvalidInputError(
                f"Mesh.level sonlu bir sayi olmali -- 'alan == sign * level' "
                f"sozlesmesi nan/inf ile anlamsizlasir ve level panele "
                f"basilir; verilen: {self.level!r}"
            )


def extract_isosurface(field: ScalarField, level: float) -> Mesh:
    """Alanın `level` izoyüzeyini bohr koordinatlı bir üçgen ağa çevirir.

    Alanda hiç bulunmayan SONLU bir seviye istenirse boş ağ döner (0 tepe,
    0 üçgen); bu `pg.isosurface`'in kendi davranışıdır, burada eklenen bir
    koruma değildir.

    Sonlu OLMAYAN bir seviye ise reddedilir (`InvalidInputError`) -- o bir
    "bulunmayan seviye" değil, GEÇERSİZ bir seviyedir. Ayrım ölçülmüştü:
    `nan`/`±inf` de tam olarak `level=1000.0` gibi boş ağ döndürüyordu ve
    çağıran ikisini tepe sayısından ayırt edemiyordu; uyarı da yoktu
    (`warnings.simplefilter("always")` + `np.errstate(all="raise")` altında
    uyarı sayısı 0).
    """
    return _ag_cikar(field.values, field.grid, level=level, sign=1)


def extract_signed_pair(field: ScalarField, level: float) -> tuple[Mesh, Mesh]:
    """`(pozitif, negatif)` ağ çifti: alanın `+level` ve `-level` izoyüzeyleri.

    Negatif taraf alanın işareti ters çevrilip aynı pozitif seviyede
    çıkarılarak bulunur (`-values`, `+level`). Bu, `pg.isosurface(values,
    -level)` ile **EŞDEĞER DEĞİLDİR**; iddia yalnızca yarısı doğrudur:

    - TEPELER için eşdeğer -- ölçüldü, `pg.isosurface(-p, +0.05)` ile
      `pg.isosurface(p, -0.05)` aynı tepe kümesini veriyor (her ikisi de
      3076 tepe, `np.array_equal` True).
    - YÜZLER için DEĞİL -- 6040 yüzün her birinde sarım terstir
      (örneğin [58, 53, 0] karşı [58, 0, 53]).

    Ve fark önemsiz değil: `pg.isosurface` sarımı, normal **girdi dizisinin
    AZALDIĞI** yöne bakacak şekilde üretiyor. Negatif lobun içinde `-values`
    büyük olduğu için `-values` yolu o lobun normalini DIŞARI çevirir; iki
    lobun da dışarı bakması bu numaranın sonucudur. `pg.isosurface(values,
    -level)`e "sadeleştiren" bir refactor negatif lobu iç-dışa çevirir --
    ölçüldü, dışarı bakan üçgen oranı 1.000'den 0.000'a düşer.

    Tüketici (`app/orbital_window.py`) `faces`'i `gl.MeshData`ya verip
    `GLMeshItem`'ı `smooth=True, shader="balloon"` ile çiziyor; ikisi de
    sarımdan türetilen tepe normallerine bağlı, yani bu doğrudan görüntüye
    yansır. Sarım `tests/test_isosurface.py`'de iki testle kilitlidir.

    `level` mutlak değer olarak yorumlanmaz: negatif bir `level` verilirse iki
    ağın rolleri yer değiştirir. Çağıran, arayüzde ±faz ayrımı istiyorsa
    pozitif bir `level` vermelidir.
    """
    return (
        _ag_cikar(field.values, field.grid, level=level, sign=1),
        _ag_cikar(-np.asarray(field.values), field.grid, level=level, sign=-1),
    )


def _ag_cikar(values: np.ndarray, grid: VolumeGrid, *, level: float, sign: int) -> Mesh:
    # İki public giriş noktasının TEK hunisi; sonluluk kontrolü bu yüzden
    # burada. `Mesh.__post_init__` de aynı şartı taşıyor (doğrudan kurulan
    # ağlar için), ama hata mesajı burada çağıranın verdiği değeri gösterir.
    if not math.isfinite(float(level)):
        raise InvalidInputError(
            f"izoyuzey seviyesi sonlu bir sayi olmali; nan/inf 'alanda "
            f"bulunmayan seviye' ile ayni bos agi verir ve ikisi ayirt "
            f"edilemez, verilen: {level!r}"
        )
    veri = np.ascontiguousarray(values, dtype=np.float32)
    indeksler, yuzler = pg.isosurface(veri, float(level))
    indeksler = np.asarray(indeksler, dtype=float)
    origin = np.asarray(grid.origin, dtype=float)
    spacing = np.asarray(grid.spacing, dtype=float)
    return Mesh(
        vertices=origin + spacing * indeksler,
        faces=np.asarray(yuzler),
        level=float(level),
        sign=int(sign),
    )
