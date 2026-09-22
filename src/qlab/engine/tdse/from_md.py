"""`md -> tdse` KENARI -- MD'nin kuvvet alanını TDSE'nin potansiyeline bağlar.

Bu dosya Modül 7 §7.2'nin ön ölçümünden (22.09.2026, 34 ajan) doğdu. O
ölçüme kadar `tdse-md` çifti kova tablosunda `OLCULMEDI`/`None` idi ve
gerekçesi "hiçbir sayı gelmedi"ydi. Ölçüldü: sayılar geldi, ve ortaya
**kova sözlüğünde bir delik** çıktı -- çiftin durumu *karşılık var + veri
akabilir + çıktı kıpırdar + kod kenarı YOK* idi ve beş kovanın hiçbiri bunu
taşıyamıyordu. Bu dosya o deliği kovayı değiştirerek değil, **kod kenarını
gerçekten kurarak** kapatır.

ÖLÇÜLMÜŞ KARŞILIK (bu kenarın var olma gerekçesi, ön ölçüm):
MD'nin `Harmonic(k)` alanı ile `tdse.potentials.harmonic(omega, mass)` aynı
ızgarada **BİT BİT** aynı diziyi veriyor -- 11 keyfi `(omega, mass)` çifti ×
4 ızgarada `np.array_equal` True. Cebirsel sebebi: `0.5` ile çarpma ikilik
kayan noktada TAM olduğu için `(0.5*m)*w**2` ile `0.5*(m*w**2)` aynı biti
verir. Yani iki motor bir ANALOJİ değil, bir NESNE paylaşıyor.

AKIŞ YÖNÜ TEKTİR: `md -> tdse`. TDSE psi kareleri üretir; MD'nin girdi
alanları konum/hız/kütle ve bir `ForceField` çağrılabiliridir, **bir dalga
fonksiyonunu tüketen MD girdisi YOKTUR**. Ters yön bir köprü değil bir
yeniden tasarımdır ve bu dosyanın kapsamı DIŞINDADIR.

--------------------------------------------------------------------------
İKİ BEYAN -- İKİSİ DE ADIYLA İLAN EDİLİR, ÇÜNKÜ İKİSİ DE SESSİZ KALABİLİRDİ
--------------------------------------------------------------------------

**(1) KOORDİNAT/KÜTLE İNDİRGEMESİ.** İki motorun HİÇBİRİ bunu taşımaz: MD
üç boyutlu iki cisim görür, TDSE tek boyutlu tek parçacık. Köprü ikisi
arasında iki-cisim -> bağıl-koordinat indirgemesi yapar:
    x (ızgara)  =  r - r_e          r = |r_2 - r_1|
    mu          =  m_1 m_2 / (m_1 + m_2)
`mu` bir KONVANSİYON DEĞİL bir TÜREVDİR ve testle kilitlidir: eşit kütlede
`reduced_mass` `m/2`'yi BİT BİT verir (ölçüldü, `test_from_md.py`).

**(2) GÖMME KATLANMASI -- YAPISAL, ÖLÇÜLÜR, GİZLENMEZ.** `ForceField`
sözleşmesi KONUM alır; ayrım içeride `r = |r_2 - r_1|` ile, yani bir NORM
olarak hesaplanır ve **negatif olamaz**. Izgara ise işaretlidir. Bu yüzden
`x < -r_e` olan her ızgara noktasında ikinci atom öbür tarafa geçer ve
potansiyel duvarın tepesinden geri AŞAĞI döner -- sahte bir özellik.

Bu, elle yazılmış bir `V(x) = d_e (1 - exp(-a x))**2` satırının YAPAMAYACAĞI
bir hatadır; köprü onu sözleşmeye saygı duyduğu için MİRAS ALIR. Kutu
`L <= 2 r_e` ise hiç olmaz. Daha büyük kutularda olur ve ölçüm L >= 3 bohr
istediği için (L = 2.0 yakınsamamış TEK noktadır) bu kenar KATLANARAK
sevk edilir.

**BU YÜZDEN FONKSİYON DİZİ DEĞİL BİR KAYIT DÖNER.** Katlanan nokta sayısı
çağıranın okumak ZORUNDA olduğu bir alandır; ayrı bir "beyan" fonksiyonu
olsaydı çağıran onu çağırmayabilirdi. Deponun dersi: *bir yokluk beyanı
bayatlarsa bir yalandır* -- ve okunmayan bir beyan zaten bayattır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.md.state import ForceField
from qlab.engine.tdse.grid import Grid1D
from qlab.errors import InvalidInputError


@dataclass(frozen=True, eq=False)
class MdTdsePotansiyeli:
    """`md -> tdse` kenarının çıktısı: potansiyel VE onun iki beyanı.

    `eq=False`: `potential` bir ndarray; üretilen `__eq__` dizi döndürür ve
    `__hash__` `TypeError` verir. Depo bu tuzağı `engine/hf/integrals.py` ve
    `engine/md/state.py`de zaten çözmüş; aynı gerekçe.
    """

    #: Izgara üzerinde potansiyel enerji (hartree), şekil `(grid.n,)`.
    potential: np.ndarray
    #: İndirgenmiş kütle (elektron kütlesi). `SplitOperatorPropagator`a
    #: verilecek `mass` BUDUR -- atom kütlesi DEĞİL.
    reduced_mass: float
    #: Izgara sıfırının karşılık geldiği ayrım (bohr): `r = r_e + x`.
    r_e: float
    #: `x < -r_e` olduğu için gömmenin KATLANDIĞI ızgara noktası sayısı.
    #: 0 ise kutu `L <= 2 r_e`dir ve kenar katlanmadan taşınmıştır.
    katlanan_nokta: int
    #: Katlanmanın başladığı ızgara koordinatı (bohr) -- her zaman `-r_e`.
    katlanma_esigi: float

    @property
    def katlandi_mi(self) -> bool:
        return self.katlanan_nokta > 0


def reduced_mass(masses: np.ndarray) -> float:
    """İki cismin indirgenmiş kütlesi: `mu = m_1 m_2 / (m_1 + m_2)`.

    EŞİT KÜTLEDE `m/2`Yİ BİT BİT VERİR -- ve bu bir umut değil bir ölçümdür
    (ölçüldü 22.09.2026: H için `918.57632354099849`, `bridge_obs.MU_H2` ile
    de bit bit aynı). Böylece ön ölçümün *"mu = m/2 BENİM koyduğum
    konvansiyon"* beyanı KAPANIR: artık bir türevdir.

    BİÇİM SEÇİMİ DE ÖLÇÜLDÜ, gerekçesiz taşınmıyor. Kitabi
    `m_1*m_2/(m_1+m_2)` biçiminin PAYI TAŞAR: eşit kütlede `m = 1e150`'ye
    kadar iki biçim bit bit aynı, ama `m = 1e155`'te `m*m/(m+m)` tam olarak
    `inf` döner; seçilen `m_1*(m_2/(m_1+m_2))` aynı yerde `5e+154` verir.
    Bu depodaki kütleler (elektron kütlesi cinsinden, ~1e3) o sınırın çok
    altındadır -- yani seçimin BEDELİ SIFIR, kazancı bir kaçış deliğinin
    kapanması. Aynı cebir eşit kütlede sonucu tam `m/2` yapar: payda tam
    `2m`, `m_2/(2m)` tam `0.5`.
    """
    dizi = np.asarray(masses, dtype=float)
    if dizi.shape != (2,):
        raise InvalidInputError(
            f"indirgenmiş kütle iki cisim bekler, verilen şekil: {dizi.shape}"
        )
    if not np.all(np.isfinite(dizi)) or np.any(dizi <= 0.0):
        raise InvalidInputError(f"kütleler pozitif ve sonlu olmalı: {dizi.tolist()}")
    m1, m2 = float(dizi[0]), float(dizi[1])
    return m1 * (m2 / (m1 + m2))


def potential_from_force_field(
    grid: Grid1D,
    field: ForceField,
    *,
    r_e: float,
    masses: np.ndarray,
) -> MdTdsePotansiyeli:
    """MD kuvvet alanından TDSE potansiyeli üretir (`md -> tdse` kenarı).

    Her ızgara noktası için iki atom `r = r_e + x` ayrımıyla x ekseni
    üzerine konur, `field` ÇAĞRILIR ve potansiyel enerjisi okunur. Kuvvet
    bileşeni kullanılmaz: TDSE potansiyeli tüketir, kuvveti değil.

    `r == 0` olan bir ızgara noktası varsa hata fırlatır -- `Morse` orada
    zaten tanımsızdır ve sessizce `nan` taşımak bu depoda yasaktır.
    """
    if not np.isfinite(r_e) or r_e <= 0.0:
        raise InvalidInputError(f"r_e pozitif ve sonlu olmalı: {r_e}")
    mu = reduced_mass(masses)

    x = grid.x
    r = r_e + x
    if np.any(r == 0.0):
        raise InvalidInputError(
            "bir ızgara noktası tam r = 0'a düşüyor (çakışık atomlar); "
            f"r_e = {r_e!r}, dx = {grid.dx!r} -- ızgarayı kaydır ya da n'i değiştir"
        )

    hizalar = np.zeros((grid.n, 2, 3), dtype=float)
    hizalar[:, 1, 0] = r
    V = np.empty(grid.n, dtype=float)
    for i in range(grid.n):
        enerji, _ = field(hizalar[i])
        V[i] = enerji
    if not np.all(np.isfinite(V)):
        raise InvalidInputError(
            "kuvvet alanı ızgarada sonlu olmayan potansiyel üretti "
            f"({int(np.count_nonzero(~np.isfinite(V)))} nokta)"
        )

    return MdTdsePotansiyeli(
        potential=V,
        reduced_mass=mu,
        r_e=float(r_e),
        katlanan_nokta=int(np.count_nonzero(r < 0.0)),
        katlanma_esigi=-float(r_e),
    )
