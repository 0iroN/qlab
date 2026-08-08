"""Molekuler integraller: ortusme (S), kinetik (T), cekirdek-cekim (V), ERI.

Dort tensor de McMurchie-Davidson (MD) semasiyla hesaplanir. Ozyineleme
**genel bicimde yazilmistir** (acisal momentumda bir ust sinir kodlanmaz;
s kabugu semanin i=j=k=0 ozel durumudur, ayri bir kod yolu yoktur) --
ama "genel bicimde yazilmis olmak" ile "dogrulanmis olmak" ayni sey
DEGILDIR. Bu modulde DOGRULANAN kapsam sudur:

  * s ve p (l<=1): dort tensor de PySCF referansiyla dogrulanir --
    PySCF'in kendi bohr sabitiyle kurulan su geometrisinde atol=1e-12,
    depo geometrisinde ve H2/STO-3G'de atol/rtol=1e-10 duzeyinde
    (tests/hf/test_integrals.py).
  * l>=2: YALNIZCA ortusme (S) ve kinetik (T) -- ve dipol momenti --
    sayisal kuadraturla capraz kontrol edilir (test_integrals.py,
    "l >= 2" bolumu).

Dolayisiyla DOGRULANMAYANLAR acikca sunlardir:

  * Coulomb yarisinin tamami l>=2'de test EDILMEZ: ne cekirdek-cekim (V)
    ne de ERI, p kabugunun ustunde hicbir bagimsiz referansa karsi
    kosulmamistir. Bu, Hermite-Coulomb yardimcilarinin (R) yuksek
    t+u+v mertebelerini ve boys(n>=6)'yi kapsamayan bir bosluktur.
  * Kontraksiyon normalizasyonu l>=2'de test EDILMEZ: yukarida anilan
    l>=2 testleri normalizasyonu iki tarafta da ayni formulle hesaplar,
    dolayisiyla yanlis bir normalizasyon her iki tarafta sadelesir.
    (Tek istisna: basis.py'nin (2n-1)!! carpani _self_overlap icin
    bagimsiz kuadraturla ayrica sinanir -- bkz. test_basis.py.)

Bu bosluklar d kabugu eklenmeden kapatilmali; build_basis su an l>=2'yi
zaten reddediyor (basis.py, spec bolum 13), yani uretimde tetiklenmiyor.

Neden MD? Dort integral tipi de ayni iki ozyinelemeyi paylasir (Hermite
genisletme katsayilari E ve Hermite-Coulomb yardimcilari R), yani tek bir
turetme dogrulandiginda dordu birden dogrulanmis olur. Obara-Saika'da
tek-elektron ve iki-elektron dikey/yatay ozyinelemeleri ayri ayri
turetilip ayri ayri dogrulanmak zorundadir. Kaynak: Helgaker/Jorgensen/
Olsen, Molecular Electronic-Structure Theory, Bolum 9.

MD'nin cekirdegi: iki Kartezyen Gauss'un carpimi, ortak merkez P etrafinda
Hermite Gauss'lari cinsinden **sonlu** bir seri olarak yazilabilir
(p = a+b, mu = ab/p, P = (aA+bB)/p, X_AB = A_x - B_x):

    x_A^i x_B^j exp(-a x_A^2) exp(-b x_B^2) = sum_{t=0}^{i+j} E_t^{ij} L_t(x_P)

E katsayilari su ozyinelemeden gelir (E_t = 0 icin t<0 veya t>i+j):

    E_0^{00} = exp(-mu X_AB^2)
    E_t^{i+1,j} = E_{t-1}^{ij}/(2p) + X_PA E_t^{ij} + (t+1) E_{t+1}^{ij}
    E_t^{i,j+1} = E_{t-1}^{ij}/(2p) + X_PB E_t^{ij} + (t+1) E_{t+1}^{ij}

Hermite Gauss'larinin tekil olmayan integralleri kapali oldugundan:

    S_ab = (pi/p)^{3/2} E_0^{i_x j_x} E_0^{i_y j_y} E_0^{i_z j_z}

Kinetik enerji, -1/2 d^2/dx^2'nin ket uzerindeki dogrudan etkisiyle 1B
ortusmelere indirgenir (S_1B = sqrt(pi/p) E_0):

    T^{1B}_{ij} = -1/2 j(j-1) S_1B(i, j-2) + b(2j+1) S_1B(i, j) - 2b^2 S_1B(i, j+2)
    T_ab = T_x S_y S_z + S_x T_y S_z + S_x S_y T_z

Coulomb tipi integraller Hermite-Coulomb yardimcilarini kullanir
(R^n_{000} = (-2 alpha)^n F_n(alpha |D|^2), D vektoru asagida):

    R^n_{t+1,u,v} = t R^{n+1}_{t-1,u,v} + D_x R^{n+1}_{t,u,v}   (u, v icin benzeri)

    V_ab(C) = (2 pi / p) sum_{tuv} E_t E_u E_v R^0_{tuv}(p, P-C)
    (ab|cd) = 2 pi^{5/2} / (p q sqrt(p+q))
              * sum_{tuv} E^{ab}_t E^{ab}_u E^{ab}_v
              * sum_{t'u'v'} (-1)^{t'+u'+v'} E^{cd}_{t'} E^{cd}_{u'} E^{cd}_{v'}
              * R^0_{t+t',u+u',v+v'}(pq/(p+q), P-Q)

V isareti cekirdek yukuyle birlikte tasinir: V = -Z * V_ab(C).

Kontrakte fonksiyon degeri primitif katkilarinin d_i * d_j * N_i * N_j
agirlikli toplamidir; ayrica her fonksiyon
BasisFunction.contraction_normalisation() ile kendi oz-ortusmesinin
kareokune bolunerek yeniden normalize edilir -- bu olmadan PySCF
referansiyla eslesme atol=1e-10 hedefini onlarca kat asar (bkz.
.superpowers/sdd/task-5-brief.md ve basis.py).

F_n icin Boys fonksiyonu (qlab.engine.hf.boys.boys) kullanilir; su/STO-3G'de
(yalniz s ve p kabuklari, l<=1) gereken en yuksek mertebe cekirdek-cekimde
F_2, ERI'de F_4'tur -- bu sadece mevcut p-kabugu kapsami icin gecerlidir,
genel bir ust sinir DEGILDIR. _hermite_coulomb, order = bra.order + ket.order
mertebesine kadar boys(n, ...) cagirir; l=2 (d kabugu) eklenirse bu order=8'e
kadar cikar ve boys.py'deki _LARGE_T_THRESHOLD yalniz n=0..5 icin olculdu
(bkz. boys.py modul docstring'i). build_basis d kabugunu reddettigi icin
(basis.py, spec bolum 13) bu yol su an hic tetiklenmiyor, ama d/f kabugu
eklenirse esigin n=6+ icin yeniden olculmesi gerekir.

ERI tensoru 8-katli permutasyon simetrisine sahiptir:
(ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba).
Bu simetriden yararlanilarak yalnizca a>=b, c>=d, (ab)>=(cd) uclugeni
hesaplanir; kalan elemanlar kopyalanarak doldurulur (bkz. eri_tensor).
Ayrica her benzersiz (a,b) cifti icin MD verisi bir kez kurulup tum
dortlulerde yeniden kullanilir (_pair_data onbellegi).

Task 10, analiz katmani (dipol momenti) icin besinci bir integral tipi
ekler: dipole_matrices. Bu da S/T/V/ERI'nin kullandigi ayni E ozyinelemesi
uzerine kuruludur -- yeni bir turetme degil, mevcut Hermite genislemesinin
(x-C) operatoruyle carpimidir (bkz. _dipole_primitive_block dokstringi).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qlab.engine.hf.basis import BasisFunction
from qlab.engine.hf.boys import boys
from qlab.types import Molecule


def _weights(function: BasisFunction) -> np.ndarray:
    """Primitif basina tam agirlik: d_i * N_i * kontraksiyon-normalizasyonu."""
    coefficients = np.array([p.coefficient for p in function.primitives])
    return (
        coefficients
        * function.normalisation()
        * function.contraction_normalisation()
    )


def _primitive_grids(
    a: BasisFunction, b: BasisFunction
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iki fonksiyonun primitif ciftleri icin uslar, p ve mu izgarasi.

    p ve mu (na, nb) sekilli matrislerdir (na/nb primitif sayisi).
    """
    exp_a = np.array([p.exponent for p in a.primitives])
    exp_b = np.array([p.exponent for p in b.primitives])
    p = exp_a[:, None] + exp_b[None, :]
    mu = exp_a[:, None] * exp_b[None, :] / p
    return exp_a, exp_b, p, mu


def _pair_centre(
    function: BasisFunction,
    p: np.ndarray,
    exp_left: np.ndarray,
    exp_right: np.ndarray,
    other: BasisFunction,
) -> np.ndarray:
    """Bir primitif cifti icin agirlikli merkez P = (a*A + b*B) / p, sekil (na, nb, 3)."""
    return (
        exp_left[:, None, None] * function.center
        + exp_right[None, :, None] * other.center
    ) / p[:, :, None]


@dataclass(frozen=True, eq=False)
class _Axis:
    """Tek bir Kartezyen eksen icin MD genisletme katsayilari E_t^{ij}.

    p, x_pa, x_pb ve e000 primitif cifti izgarasi sekillidir (na, nb).
    e() ozyinelemesi ayni cift icin cok kez cagrildigindan sonuclar
    onbelleklenir; onbelleksiz agac (i+j) buyudukce 3^(i+j) gibi buyur.

    eq=False: alanlar numpy dizisi (ve bir dict) tasidigi icin uretilen
    __eq__ bir dizi doner (bool degil) ve __hash__ TypeError firlatir.
    Hicbir yer bu nesneleri karsilastirmiyor/hashlemiyor, ama frozen=True
    tek basina eq=True varsayilaniyla bu kirik __eq__/__hash__ ciftini
    otomatik uretirdi -- eq=False onu hic uretmez.
    """

    p: np.ndarray
    x_pa: np.ndarray
    x_pb: np.ndarray
    e000: np.ndarray
    _cache: dict[tuple[int, int, int], np.ndarray] = field(
        default_factory=dict, repr=False, compare=False
    )

    def e(self, i: int, j: int, t: int) -> np.ndarray | float:
        """E_t^{ij}; tanim disi (j<0, t<0, t>i+j) durumlarda 0.

        j<0 dali kinetik enerjideki S(i, j-2) teriminden gelir ve j<2 olan
        her fonksiyon icin -- yani her s ve p fonksiyonu icin -- kullanilir.
        i<0 diye bir dal YOKTUR: i yalnizca j==0 dalinda azaltilir, o dal da
        (i, j) == (0, 0) durumunda calismaz, dolayisiyla i asla negatife
        dusmez. Cagiranlar da yalnizca angular bilesenlerini (>=0) verir.
        """
        if j < 0 or t < 0 or t > i + j:
            return 0.0
        key = (i, j, t)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if i == 0 and j == 0:
            value = self.e000
        elif j == 0:
            value = (
                self.e(i - 1, j, t - 1) / (2 * self.p)
                + self.x_pa * self.e(i - 1, j, t)
                + (t + 1) * self.e(i - 1, j, t + 1)
            )
        else:
            value = (
                self.e(i, j - 1, t - 1) / (2 * self.p)
                + self.x_pb * self.e(i, j - 1, t)
                + (t + 1) * self.e(i, j - 1, t + 1)
            )
        self._cache[key] = value
        return value


@dataclass(frozen=True, eq=False)
class _PairData:
    """Bir baz fonksiyonu ciftinin MD verisi; dort integral tipi de bunu kullanir.

    hermite, E_t^x E_u^y E_v^z carpimlarini (t, u, v) anahtariyla tutar;
    t <= i_x+j_x ve benzeri sinirlarin disindaki terimler sifir oldugu icin
    sozlukte yer almaz.

    eq=False: ayni gerekce _Axis icin gecerli -- alanlar numpy dizisi ve
    dict tasiyor, uretilen __eq__/__hash__ kirik olurdu; kullanilmiyor,
    o yuzden hic uretilmiyor.
    """

    p: np.ndarray
    exp_b: np.ndarray
    centre: np.ndarray
    axes: tuple[_Axis, _Axis, _Axis]
    angular_a: tuple[int, int, int]
    angular_b: tuple[int, int, int]
    hermite: dict[tuple[int, int, int], np.ndarray]
    order: int


def _pair_data(a: BasisFunction, b: BasisFunction) -> _PairData:
    """(a, b) cifti icin primitif izgarasi ve Hermite katsayilarini kurar."""
    exp_a, exp_b, p, mu = _primitive_grids(a, b)
    centre = _pair_centre(a, p, exp_a, exp_b, b)

    axes = tuple(
        _Axis(
            p=p,
            x_pa=centre[:, :, k] - a.center[k],
            x_pb=centre[:, :, k] - b.center[k],
            e000=np.exp(-mu * (a.center[k] - b.center[k]) ** 2),
        )
        for k in range(3)
    )

    hermite: dict[tuple[int, int, int], np.ndarray] = {}
    for t in range(a.angular[0] + b.angular[0] + 1):
        ex = axes[0].e(a.angular[0], b.angular[0], t)
        for u in range(a.angular[1] + b.angular[1] + 1):
            ey = axes[1].e(a.angular[1], b.angular[1], u)
            for v in range(a.angular[2] + b.angular[2] + 1):
                ez = axes[2].e(a.angular[2], b.angular[2], v)
                hermite[(t, u, v)] = ex * ey * ez

    return _PairData(
        p=p,
        exp_b=exp_b,
        centre=centre,
        axes=axes,
        angular_a=a.angular,
        angular_b=b.angular,
        hermite=hermite,
        order=sum(a.angular) + sum(b.angular),
    )


def _hermite_coulomb(
    order: int, alpha: np.ndarray, delta: np.ndarray
) -> dict[tuple[int, int, int], np.ndarray]:
    """R^0_{tuv}(alpha, delta) yardimcilari, t+u+v <= order.

    delta son ekseni 3 olan bir vektor izgarasidir (cekirdek-cekimde P-C,
    ERI'de P-Q). Ozyineleme toplam mertebe artarak yurutulur; L mertebesi
    icin L-1 ve L-2'nin bir ustteki n seviyeleri gerekir, bu yuzden
    F_0..F_order'in tamami basta hesaplanir.
    """
    x, y, z = delta[..., 0], delta[..., 1], delta[..., 2]
    T = alpha * (x * x + y * y + z * z)

    table: dict[tuple[int, int, int, int], np.ndarray] = {}
    for n in range(order + 1):
        table[(n, 0, 0, 0)] = (-2 * alpha) ** n * boys(n, T)

    for total in range(1, order + 1):
        for t in range(total + 1):
            for u in range(total - t + 1):
                v = total - t - u
                for n in range(order - total + 1):
                    if t > 0:
                        value = x * table[(n + 1, t - 1, u, v)]
                        if t > 1:
                            value = value + (t - 1) * table[(n + 1, t - 2, u, v)]
                    elif u > 0:
                        value = y * table[(n + 1, t, u - 1, v)]
                        if u > 1:
                            value = value + (u - 1) * table[(n + 1, t, u - 2, v)]
                    else:
                        value = z * table[(n + 1, t, u, v - 1)]
                        if v > 1:
                            value = value + (v - 1) * table[(n + 1, t, u, v - 2)]
                    table[(n, t, u, v)] = value

    return {
        (t, u, v): value for (n, t, u, v), value in table.items() if n == 0
    }


def _overlap_primitive_block(pair: _PairData) -> np.ndarray:
    """S_ab primitif izgarasi, sekil (na, nb)."""
    return (np.pi / pair.p) ** 1.5 * pair.hermite[(0, 0, 0)]


def _kinetic_primitive_block(pair: _PairData) -> np.ndarray:
    """T_ab primitif izgarasi, sekil (na, nb)."""
    root = np.sqrt(np.pi / pair.p)
    beta = pair.exp_b[None, :]

    overlaps = []
    kinetics = []
    for k in range(3):
        i, j = pair.angular_a[k], pair.angular_b[k]
        axis = pair.axes[k]
        overlap = root * axis.e(i, j, 0)
        overlaps.append(overlap)
        kinetics.append(
            -0.5 * j * (j - 1) * root * axis.e(i, j - 2, 0)
            + beta * (2 * j + 1) * overlap
            - 2 * beta**2 * root * axis.e(i, j + 2, 0)
        )

    return (
        kinetics[0] * overlaps[1] * overlaps[2]
        + overlaps[0] * kinetics[1] * overlaps[2]
        + overlaps[0] * overlaps[1] * kinetics[2]
    )


def _dipole_primitive_block(
    pair: _PairData, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """<a|(r-origin)|b> primitif izgaralari, uc eksen icin (na, nb) sekilli uclu.

    Turetme: x_A^i x_B^j = sum_t E_t^{ij} Lambda_t(x_P) (Hermite Gauss
    genislemesi, modul dokstringindeki E ozyinelemesiyle ayni). (x-C) =
    (x-P) + (P_x-C_x) yazip carpinca:

        x_A^i x_B^j (x-C) = sum_t E_t^{ij} [x_P Lambda_t(x_P)]
                             + (P_x-C_x) sum_t E_t^{ij} Lambda_t(x_P)

    x_P Lambda_t(x_P) = (1/(2p)) Lambda_{t+1}(x_P) + t Lambda_{t-1}(x_P)
    kimligi (Hermite Gauss'larin standart yukseltme/alcaltma ozelligi) ve
    integral(Lambda_s dx) = sqrt(pi/p) * delta_{s,0} (overlap formulunde
    zaten kullanilan ayni ozellik) birlikte, integralde yalniz t=1 terimi
    hayatta kalir:

        <a|(x-C)|b>_1B = sqrt(pi/p) * [ E_1^{ij} + (P_x-C_x) * E_0^{ij} ]

    s-tipi (i=j=0) icin E_1^{00}=0 (tanim geregi t>i+j=0 sifir), yani bu
    S_ab*(P_x-C_x)'e indirgenir -- brief'in "s-tipi kapali form basittir"
    notuyla tutarli. p kabugu icin ayni E ozyinelemesi (zaten kinetik/
    coulomb integrallerinde kullanilan _Axis.e) dogrudan calisir, yeni bir
    ozyineleme gerekmez.

    Tam 3B matris elemani, diger iki eksende duz ortusmeyle carpilir
    (kinetik enerjideki T_x*S_y*S_z kalibiyla ayni mantik, farkla ki
    burada UC ayri matris -- Dx, Dy, Dz -- dondurulur, toplanmaz).
    """
    root = np.sqrt(np.pi / pair.p)
    overlaps = []
    moments = []
    for k in range(3):
        i, j = pair.angular_a[k], pair.angular_b[k]
        axis = pair.axes[k]
        overlap = root * axis.e(i, j, 0)
        overlaps.append(overlap)
        shift = pair.centre[:, :, k] - origin[k]
        moments.append(root * axis.e(i, j, 1) + shift * overlap)

    return (
        moments[0] * overlaps[1] * overlaps[2],
        overlaps[0] * moments[1] * overlaps[2],
        overlaps[0] * overlaps[1] * moments[2],
    )


def _nuclear_primitive_block(pair: _PairData, centre_c: np.ndarray) -> np.ndarray:
    """Birim pozitif yuk icin Coulomb integrali izgarasi, sekil (na, nb).

    Donen deger pozitiftir; cekirdek yuku ve cekim isareti cagirana aittir.
    """
    r = _hermite_coulomb(pair.order, pair.p, pair.centre - centre_c)
    total = 0.0
    for key, coefficient in pair.hermite.items():
        total = total + coefficient * r[key]
    return 2 * np.pi / pair.p * total


def _eri_primitive_block(bra: _PairData, ket: _PairData) -> np.ndarray:
    """(ab|cd) primitif izgarasi, sekil (na, nb, nc, nd)."""
    p = bra.p[:, :, None, None]
    q = ket.p[None, None, :, :]
    delta = bra.centre[:, :, None, None, :] - ket.centre[None, None, :, :, :]
    alpha = p * q / (p + q)

    r = _hermite_coulomb(bra.order + ket.order, alpha, delta)

    total = 0.0
    for (t, u, v), bra_coefficient in bra.hermite.items():
        left = bra_coefficient[:, :, None, None]
        for (tt, uu, vv), ket_coefficient in ket.hermite.items():
            sign = -1.0 if (tt + uu + vv) % 2 else 1.0
            total = total + (
                sign
                * left
                * ket_coefficient[None, None, :, :]
                * r[(t + tt, u + uu, v + vv)]
            )

    return 2 * np.pi**2.5 / (p * q * np.sqrt(p + q)) * total


def overlap_matrix(basis: list[BasisFunction]) -> np.ndarray:
    """Ortusme matrisi S_ab, sekil (n, n)."""
    n = len(basis)
    weights = [_weights(f) for f in basis]
    matrix = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            block = _overlap_primitive_block(_pair_data(basis[a], basis[b]))
            matrix[a, b] = weights[a] @ block @ weights[b]
    return matrix


def kinetic_matrix(basis: list[BasisFunction]) -> np.ndarray:
    """Kinetik enerji matrisi T_ab, sekil (n, n)."""
    n = len(basis)
    weights = [_weights(f) for f in basis]
    matrix = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            block = _kinetic_primitive_block(_pair_data(basis[a], basis[b]))
            matrix[a, b] = weights[a] @ block @ weights[b]
    return matrix


def nuclear_matrix(basis: list[BasisFunction], molecule: Molecule) -> np.ndarray:
    """Cekirdek-cekim matrisi V_ab, sekil (n, n).

    Her cekirdek uzerinden toplanir: V = sum_C -Z_C * V_ab(C); isaret
    negatiftir (cekim).
    """
    n = len(basis)
    weights = [_weights(f) for f in basis]
    matrix = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            pair = _pair_data(basis[a], basis[b])
            toplam = 0.0
            for atom in molecule.atoms:
                block = _nuclear_primitive_block(pair, atom.position)
                toplam += -atom.atomic_number * (weights[a] @ block @ weights[b])
            matrix[a, b] = toplam
    return matrix


def core_hamiltonian(basis: list[BasisFunction], molecule: Molecule) -> np.ndarray:
    """Cekirdek Hamiltoniyeni H_core = T + V, sekil (n, n)."""
    return kinetic_matrix(basis) + nuclear_matrix(basis, molecule)


def dipole_matrices(
    basis: list[BasisFunction], origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dipol integral matrisleri (Dx, Dy, Dz): <mu|(r-origin)|nu>, her biri (n, n).

    origin, moment hesabinin merkezidir; molekul yuksuz (norotr) oldugu
    surece toplam dipol momenti origin secimine gore DEGISMEZ (bkz.
    analysis.dipole_moment) -- ama bu fonksiyon her origin icin dogru
    sonucu verir, sadece norotrlik varsayimi cagirana ait degildir.
    """
    n = len(basis)
    weights = [_weights(f) for f in basis]
    origin = np.asarray(origin, dtype=float)
    dx = np.zeros((n, n))
    dy = np.zeros((n, n))
    dz = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            blocks = _dipole_primitive_block(_pair_data(basis[a], basis[b]), origin)
            dx[a, b] = weights[a] @ blocks[0] @ weights[b]
            dy[a, b] = weights[a] @ blocks[1] @ weights[b]
            dz[a, b] = weights[a] @ blocks[2] @ weights[b]
    return dx, dy, dz


def eri_tensor(basis: list[BasisFunction]) -> np.ndarray:
    """Iki-elektron itme integralleri (ab|cd), sekil (n, n, n, n).

    8-katli permutasyon simetrisinden yararlanilir: yalnizca a>=b, c>=d ve
    (ab bilesik indeksi)>=(cd bilesik indeksi) sartlarini saglayan benzersiz
    ucgen hesaplanir; kalan yedi permutasyon dogrudan kopyalanarak doldurulur.
    Her benzersiz (a, b) cifti icin MD verisi bir kez kurulur: su/STO-3G'de
    bu 406 dortlu icin 812 yerine 28 kurulum demektir.
    """
    n = len(basis)
    weights = [_weights(f) for f in basis]
    tensor = np.zeros((n, n, n, n))

    pairs = {
        (a, b): _pair_data(basis[a], basis[b])
        for a in range(n)
        for b in range(a + 1)
    }

    for a in range(n):
        for b in range(a + 1):
            ab_index = a * (a + 1) // 2 + b
            for c in range(n):
                for d in range(c + 1):
                    cd_index = c * (c + 1) // 2 + d
                    if cd_index > ab_index:
                        continue
                    blok = _eri_primitive_block(pairs[(a, b)], pairs[(c, d)])
                    deger = float(
                        np.einsum(
                            "i,j,k,l,ijkl->",
                            weights[a],
                            weights[b],
                            weights[c],
                            weights[d],
                            blok,
                        )
                    )
                    for i, j in ((a, b), (b, a)):
                        for k, l in ((c, d), (d, c)):
                            tensor[i, j, k, l] = deger
                            tensor[k, l, i, j] = deger
    return tensor
