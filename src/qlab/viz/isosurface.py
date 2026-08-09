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
35.600986 olur.

Dönüşüm eksen eksen yapılmak zorundadır çünkü `VolumeGrid` genelde
anizotropiktir: `VolumeGrid.around(H2, padding=6.0)` ölçüldüğünde
spacing = [0.15189873, 0.15189873, 0.16959997] bohr çıkıyor.

`pg.isosurface` girdinin **c-bitişik** olmasını şart koşuyor (aksi halde
`TypeError: isosurface input data must be c-contiguous.`). float32 şartı
DEĞİL -- ölçüldü, c-bitişik float64 de kabul ediliyor. Yine de float32'ye
döküyoruz: büyük ızgaralarda belleği yarıya indiriyor ve ölçülen fark
görselleştirme için önemsiz (aynı küre için yarıçap 0.999215 yerine 0.999225,
1830 yerine 1854 tepe).

Orbital genliği işaretlidir; `extract_signed_pair` `+level` ve `-level` için
iki ayrı ağ verir, böylece arayüz ±fazı iki renkle çizebilir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg

from qlab.observable.hf_obs import ScalarField, VolumeGrid


@dataclass(frozen=True)
class Mesh:
    """Bir izoyüzeyin üçgen ağı.

    - `vertices`: (m, 3), **bohr** biriminde tepe konumları.
    - `faces`: (k, 3), her satır `vertices` içine üç indeks (bir üçgen).
    - `level`: izoyüzeyin seviyesi, alanın kendi biriminde.
    - `sign`: bu ağın hangi fazı temsil ettiği, +1 ya da -1.

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
            dizi.setflags(write=False)
            object.__setattr__(self, alan, dizi)


def extract_isosurface(field: ScalarField, level: float) -> Mesh:
    """Alanın `level` izoyüzeyini bohr koordinatlı bir üçgen ağa çevirir.

    Alanda hiç bulunmayan bir seviye istenirse boş ağ döner (0 tepe, 0 üçgen);
    bu `pg.isosurface`'in kendi davranışıdır, burada eklenen bir koruma
    değildir.
    """
    return _ag_cikar(field.values, field.grid, level=level, sign=1)


def extract_signed_pair(field: ScalarField, level: float) -> tuple[Mesh, Mesh]:
    """`(pozitif, negatif)` ağ çifti: alanın `+level` ve `-level` izoyüzeyleri.

    Negatif taraf alanın işareti ters çevrilip aynı pozitif seviyede
    çıkarılarak bulunur (`-values`, `+level`) -- ölçüldü, `pg.isosurface(-p,
    +0.05)` ile `pg.isosurface(p, -0.05)` aynı tepe kümesini veriyor.

    `level` mutlak değer olarak yorumlanmaz: negatif bir `level` verilirse iki
    ağın rolleri yer değiştirir. Çağıran, arayüzde ±faz ayrımı istiyorsa
    pozitif bir `level` vermelidir.
    """
    return (
        _ag_cikar(field.values, field.grid, level=level, sign=1),
        _ag_cikar(-np.asarray(field.values), field.grid, level=level, sign=-1),
    )


def _ag_cikar(values: np.ndarray, grid: VolumeGrid, *, level: float, sign: int) -> Mesh:
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
