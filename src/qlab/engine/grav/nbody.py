"""Newtonyen N-cisim cozucusu -- VE BU DOSYA METRIGI BILMEZ.

Ikinci cozucu. `geodesic.py` egri uzay-zamanda bir TEST PARCACIGI tasir;
burasi duz uzayda BIRBIRINI CEKEN cisimleri tasir. Ikisi ayri dosyadir cunku
spec 4.1 bunu bir zorunluluk olarak olcmustur, ve spec 4.2'nin
YONTEM/HEDEF UYUSMAZLIGI beyani ancak iki cozucu GERCEKTEN ayriysa
dogrudur: Newtonyen N-body perihel kaymasini yanlis uretmez, HIC uretmez.

BU DOSYA `metric.py`DEN HICBIR SEY IMPORT ETMEZ ve `Theory`yi hic gormez.
Iddia spec 5'in "kaydeden vekil" kilidiyle sayiya cevrilir -- ama vekilin
SINIRI olculmustur: vekil `integrate`in `NBodyState`ten NE OKUDUGUNU sayar,
yani bir cekirdegin bir alani OKUMADIGINI kanitlar; IKI TUKETICININ ORTAK
BIR TUREVDEN gelmedigini kanitlamaz. Buradaki iddia BIRINCI turdendir, o
yuzden vekil gecerlidir -- ve `tests/grav/test_nbody.py` vekile EK OLARAK
kaynak duzeyinde bir import denetimi kosar, cunku vekil bir modul duzeyi
`from qlab.engine.grav.metric import ...` satirini goremez.

BIRIM SECIMI -- VE BURADA IKI KURAL CARPISIYOR, olculdu:
`engine/grav/__init__.py` "birim secimi `G = c = 1`dir ve `metric.py`de BIR
KEZ beyan edilir; buradan tekrar edilmez" diyor. Ama bu dosya `metric.py`yi
IMPORT EDEMEZ. Cozum bir kopya DEGIL, bir YOKLUKTUR: burada `G` diye bir
sabit YOKTUR, cunku Newtonyen kuvvet `G = 1` ile yazilmistir ve disari acilan
hicbir dugme onu degistirmez. Yani "tek kaynaktan beyan" kurali korunur;
bedeli `eight_invariant`in `G^2` paydasinin SAYISAL olarak is yapmamasidir
ve `observable/grav_obs.py` bunu adiyla yazar.

PENCERE -- tam beyan (bu dosyanin kendi olcumleriyle):

- `scipy.integrate.solve_ivp`, varsayilan **DOP853**, `rtol = 1e-13`,
  `atol = 1e-15`. Dort integrator sevk edilir (`NBODY_METHODS`) ve
  BVP cozumu dordunde de ayni cipayi veriyor (asagida olculdu).
- Durum 12 boyutludur: `(3 cisim) x (2 konum + 2 hiz)`, duzlemde.
- Kuvvet ussu `power` ACIK BIR DUGMEDIR (`p = 2` Newton). `power != 2`
  BIR YAKLASIM DEGIL, BASKA BIR KURAMDIR -- `metric.py`nin `Theory`si gibi
  bir KURAM SECICI; ve tam da bu yuzden cipanin YANILABILIRLIK sinavidir.
- Potansiyel genel `p` icin `U = -m_i*m_j / ((p-1) * r^(p-1))`, yani kuvvetin
  integrali. **BU BIR SECIMDIR VE PLAN ONU HIC BEYAN ETMEMISTI**; planin
  `p`-tablosu (1.7 -> 35.47, 2.1 -> 6.7560 ...) bu secim yazilmadigi icin
  yeniden uretilemez ve bu turda uretilEMEDI (olculen: 1.7 -> 23.956761,
  2.1 -> 6.778916). `p = 2`de iki secim CAKISIR, o yuzden cipanin kendisi
  etkilenmez.

SEKIZ YORUNGESININ BASLANGIC KOSULLARI BURADA ARANIR, LITERATURDEN ALINMAZ
(spec 3.6). Alinsaydi cipa DAIRESEL olurdu: `T`yi disaridan alip `|E|^{3/2}T`
raporlamak, girilen sayiyi geri okumak olurdu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from qlab.errors import InvalidInputError

#: Sevk edilen integrator toleranslari. Kareleme borusunun `quad`
#: sabitleriyle AYNI STATUDE: disari acik, ama varsayilanlari sabittir.
NBODY_RTOL: float = 1e-13
NBODY_ATOL: float = 1e-15

#: Sevk edilen integratorler -- KAPALI LISTE. `BDF` ve `Radau` ortulu
#: metotlardir ve bu problem STIF DEGILDIR; `Radau` yine de listede cunku
#: cipanin integratordan bagimsizligi ancak farkli AILELERLE olculebilir.
NBODY_METHODS: tuple[str, ...] = ("DOP853", "RK45", "Radau", "LSODA")

#: Sekizin `T/3` devri. `(2, 0, 1)` bir OLCUM SONUCUDUR, bir varsayim degil:
#: alti permutasyonun tamami sinandi ve besi O(1) artik verdi (bkz.
#: `test_nbody.py`, `test_PERMUTASYON_OLCULEREK_SECILDI...`).
#:
#: SOZLESME: `(P y)_i = y_{perm[i]}` -- yani `perm[i]`, `i`nin YERINE GECEN
#: cismin indisidir. Ters sozlesme secilseydi cevap `(1, 2, 0)` olurdu (bu
#: permutasyonun tersi) ve o da sinanan altinin icindedir -- yani sozlesme
#: bir ISIM DEGIL, sinavin GECTIGI yeri degistiren bir secimdir.
EULER_PERMUTATION: tuple[int, int, int] = (2, 0, 1)

#: BVP'nin sevk edilen baslangic tahmini. UC HANEYE YUVARLANMIS ve bu
#: BILEREKtir: tahminin kendisi bir cevap TASIMAMALIDIR, yoksa "arama"
#: bir geri okumaya doner. Olculdu -- bu tahminden cozume `least_squares`
#: 5 artik degerlendirmesinde iniyor.
FIGURE_EIGHT_GUESS: tuple[float, float, float] = (0.35, 0.53, 6.3)


@dataclass(frozen=True)
class NBodyState:
    """Duzlemde N cismin ani durumu.

    `masses`     : `(N,)`
    `positions`  : `(N, 2)`
    `velocities` : `(N, 2)`

    UC ALAN, VE TAM UCU -- `integrate` bu uc addan baskasina DOKUNMAZ;
    `test_nbody.py`deki kaydeden vekil dokunulan ad kumesini SAYAR.
    """

    masses: "np.ndarray"
    positions: "np.ndarray"
    velocities: "np.ndarray"


@dataclass(frozen=True)
class FigureEight:
    """BVP'nin buldugu sekiz yorungesi.

    `vx`, `vy` : `t = 0`daki Euler diziliminde 1. ve 2. cismin ortak hizi.
    `period`   : tam periyot `T` (BVP `T/3` uzerinden arar).
    `residual` : `max |Phi_{T/3}(y0) - P y0|` -- ARAMANIN KENDI HATA
                 CUBUGU, ve bu sayi `eight_invariant`in yaninda TASINIR.
    `mass`     : uc esit kutlenin ortak degeri. ALAN OLARAK TASINIR cunku
                 cipa `m^{5/2}`ye bolunur ve bolen bir yerden GELMELIDIR;
                 ikinci bir kaynak ikinci bir sessiz kazanan olurdu (Task
                 1'in olcumu).
    `power`    : kuvvet ussu. Ayni gerekce: `U`nun kapali formu `p`ye bagli.
    """

    vx: float
    vy: float
    period: float
    residual: float
    mass: float = 1.0
    power: float = 2.0


# ---------------------------------------------------------------------------
# CEKIRDEK -- kuvvet, akis, ornekleme
# ---------------------------------------------------------------------------
def accelerations(masses, positions, *, power: float = 2.0):
    """Newtonyen ivmeler, `G = 1`. Kuvvet `~ 1/r^power`.

    Kosegen `inf` ile kapatilir (`inf ** negatif = 0.0`, bit bit), yani
    `i = j` terimi bir `if` ile DEGIL aritmetikle elenir -- dallanma
    olsaydi mutant o dali degistirip sessizce gecebilirdi.
    """
    m = np.asarray(masses, dtype=float)
    q = np.asarray(positions, dtype=float)
    d = q[None, :, :] - q[:, None, :]              # d[i, j] = r_j - r_i
    r2 = np.sum(d * d, axis=-1)
    np.fill_diagonal(r2, np.inf)
    inv = r2 ** (-(power + 1.0) / 2.0)             # 1 / r^(power+1)
    return np.sum((m[None, :] * inv)[:, :, None] * d, axis=1)


def acceleration_jacobian(masses, positions, *, power: float = 2.0):
    """`d(ivme)/d(konum)` -- `(2N, 2N)`. Varyasyonel denklemin sag tarafi.

    Cift cift:  `G m_k [ I/d^(p+1) - (p+1) d d^T / d^(p+3) ]`, ve ayni blok
    kosegenden CIKARILIR (etki-tepki). Sonlu farkla DEGIL analitik yazildi,
    cunku monodrominin `lambda = 1` kumesi DEFEKTIFTIR ve defektif bir
    ozdegerin sayisal bozulmaya tepkisi `sqrt(bozulma)`dir: sonlu fark
    `h ~ 1e-6` verseydi ozdeger yayilimi `1e-3` mertebesinde cikardi.
    """
    m = np.asarray(masses, dtype=float)
    q = np.asarray(positions, dtype=float)
    n = len(m)
    A = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = q[j] - q[i]
            r = math.hypot(d[0], d[1])
            blok = m[j] * (np.eye(2) / r ** (power + 1.0)
                           - (power + 1.0) * np.outer(d, d) / r ** (power + 3.0))
            A[2 * i:2 * i + 2, 2 * j:2 * j + 2] += blok
            A[2 * i:2 * i + 2, 2 * i:2 * i + 2] -= blok
    return A


def _kapi(masses, positions, velocities, *, power: float) -> None:
    m = np.asarray(masses, dtype=float)
    q = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    if m.ndim != 1 or m.size < 2:
        raise InvalidInputError(
            f"N-body en az iki cisim ister; verilen sekil {m.shape}.")
    if q.shape != (m.size, 2) or v.shape != (m.size, 2):
        raise InvalidInputError(
            f"konum/hiz sekli (N, 2) olmali; verilen {q.shape} / {v.shape} "
            f"(N = {m.size}).")
    if np.any(m < 0.0):
        raise InvalidInputError(
            f"negatif kutle fiziksel degil: {m.tolist()}")
    if power <= 1.0:
        raise InvalidInputError(
            f"power = {power}: `p <= 1`de potansiyel `r^(p-1)/(p-1)` "
            "tanimsiz ya da bagli yorunge yok. Sevk edilen aralik p > 1.")
    d = q[None, :, :] - q[:, None, :]
    r2 = np.sum(d * d, axis=-1)
    np.fill_diagonal(r2, np.inf)
    if np.any(r2 == 0.0):
        raise InvalidInputError(
            "iki cisim ayni noktada: Newtonyen kuvvet orada tanimsiz.")


def _rhs(t, y, m, power):
    n = m.size
    q = y[:2 * n].reshape(n, 2)
    v = y[2 * n:].reshape(n, 2)
    a = accelerations(m, q, power=power)
    return np.concatenate([v.ravel(), a.ravel()])


def _ivp(state: NBodyState, *, t_span: float, rtol: float, atol: float,
         method: str, power: float, dense: bool):
    """Ortak cozucu govdesi. UC ALANI BURADA OKUR, baska hicbir yerde."""
    masses = state.masses
    positions = state.positions
    velocities = state.velocities
    _kapi(masses, positions, velocities, power=power)
    if method not in NBODY_METHODS:
        raise InvalidInputError(
            f"method={method!r} sevk edilen listede yok: {NBODY_METHODS}.")
    m = np.asarray(masses, dtype=float)
    y0 = np.concatenate([np.asarray(positions, float).ravel(),
                         np.asarray(velocities, float).ravel()])
    sol = solve_ivp(_rhs, (0.0, float(t_span)), y0, args=(m, float(power)),
                    method=method, rtol=rtol, atol=atol, dense_output=dense)
    if not sol.success:
        raise InvalidInputError(
            f"N-body integrasyonu basarisiz (method={method}, "
            f"t_span={t_span}, rtol={rtol}): {sol.message}")
    return m, sol


def integrate(state: NBodyState, *, t_span: float, rtol: float = NBODY_RTOL,
              atol: float = NBODY_ATOL, method: str = "DOP853",
              power: float = 2.0) -> NBodyState:
    """`t_span` kadar akit, son durumu dondur."""
    m, sol = _ivp(state, t_span=t_span, rtol=rtol, atol=atol, method=method,
                  power=power, dense=False)
    n = m.size
    y = sol.y[:, -1]
    return NBodyState(masses=m,
                      positions=y[:2 * n].reshape(n, 2).copy(),
                      velocities=y[2 * n:].reshape(n, 2).copy())


def sample_positions(state: NBodyState, *, times, rtol: float = NBODY_RTOL,
                     atol: float = NBODY_ATOL, method: str = "DOP853",
                     power: float = 2.0):
    """Verilen zamanlarda konumlar -- `(len(times), N, 2)`.

    `dense_output` kullanir: ayri ayri kosumlar YERINE tek kosum, cunku iki
    kosum iki farkli adim dizisi demektir ve o fark bir DUGME olurdu.
    """
    ts = np.asarray(times, dtype=float)
    m, sol = _ivp(state, t_span=float(ts[-1]), rtol=rtol, atol=atol,
                  method=method, power=power, dense=True)
    n = m.size
    Y = sol.sol(ts)
    return Y[:2 * n, :].T.reshape(ts.size, n, 2)


def total_energy(state: NBodyState, *, power: float = 2.0) -> float:
    """`E = T + U`, `U = -m_i m_j / ((p-1) r^(p-1))` (`G = 1`).

    `p = 2`de bilinen `-m_i m_j / r`. `p != 2` kapali formu bir SECIMDIR
    (kuvvetin integrali) ve modul dokstringinde beyan edilmistir.
    """
    m = np.asarray(state.masses, float)
    q = np.asarray(state.positions, float)
    v = np.asarray(state.velocities, float)
    kin = 0.5 * float(np.sum(m * np.sum(v * v, axis=1)))
    pot = 0.0
    n = m.size
    for i in range(n):
        for j in range(i + 1, n):
            r = math.hypot(*(q[j] - q[i]))
            pot -= m[i] * m[j] / ((power - 1.0) * r ** (power - 1.0))
    return kin + pot


def angular_momentum(state: NBodyState) -> float:
    """Duzlemde toplam acisal momentum (skaler `z` bileseni)."""
    m = np.asarray(state.masses, float)
    q = np.asarray(state.positions, float)
    v = np.asarray(state.velocities, float)
    return float(np.sum(m * (q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0])))


# ---------------------------------------------------------------------------
# SEKIZ YORUNGESI -- BVP
# ---------------------------------------------------------------------------
def euler_state(vx: float, vy: float, *, mass: float = 1.0) -> NBodyState:
    """`t = 0` ANSATZI -- Euler dizilimi, ve GIRDI OLARAK KALAN TEK SEY BU.

        q1 = (1, 0)   q2 = (-1, 0)   q3 = (0, 0)
        v1 = v2 = (vx, vy)           v3 = -2 (vx, vy)

    `v3`un katsayisi TOPLAM MOMENTUMU sifirlar (esit kutlelerde), yani bir
    ayar degil bir korunum yasasidir. Uzunluk olcegi `|q1 - q2| = 2` ve
    kutle olcegi `mass` BIRER OLCEK SECIMIDIR, cipa ikisinden de bagimsiz
    (olculdu: `mass = 0.5/1/2/7`de cipa 12 hanede AYNI).

    `L = 0` BURADAN, YORUNGEDEN DEGIL, GELIR: `q1 x v1 + q2 x v2 + q3 x v3`
    ansatzda cebirsel olarak sifirdir -- `euler_state(0.0, 0.0)` bile 0
    verir. Yani `L = 0` BIR OLCUM DEGILDIR.
    """
    q = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    v = np.array([[vx, vy], [vx, vy], [-2.0 * vx, -2.0 * vy]])
    return NBodyState(masses=np.full(3, float(mass)), positions=q, velocities=v)


def permute(state: NBodyState, perm) -> NBodyState:
    """`(P y)_i = y_{perm[i]}` -- cisim etiketlerini devret."""
    idx = list(perm)
    if sorted(idx) != list(range(len(state.masses))):
        raise InvalidInputError(
            f"perm={perm!r} bir permutasyon degil.")
    return NBodyState(masses=np.asarray(state.masses, float)[idx],
                      positions=np.asarray(state.positions, float)[idx],
                      velocities=np.asarray(state.velocities, float)[idx])


def figure_eight_residual(vx: float, vy: float, period: float, *,
                          mass: float = 1.0,
                          perm=EULER_PERMUTATION,
                          rtol: float = NBODY_RTOL, atol: float = NBODY_ATOL,
                          method: str = "DOP853", power: float = 2.0):
    """`Phi_{T/3}(y0) - P y0` -- 12 artik.

    KOSUL `T/3`TEDIR, `T`DE DEGIL, ve bu bir hiz numarasi DEGIL bir
    AYRIM meselesidir: `Phi_T = (Phi_{T/3})^3` oldugu icin `T` kosulu
    `T/3` kosulunun CEBIRSEL SONUCUDUR ve `T/3` cozumlerine ek olarak
    baska (ayrik) cozumleri de kabul ederdi.
    """
    y0 = euler_state(vx, vy, mass=mass)
    son = integrate(y0, t_span=period / 3.0, rtol=rtol, atol=atol,
                    method=method, power=power)
    hedef = permute(y0, perm)
    return np.concatenate([
        (np.asarray(son.positions) - np.asarray(hedef.positions)).ravel(),
        (np.asarray(son.velocities) - np.asarray(hedef.velocities)).ravel(),
    ])


@lru_cache(maxsize=None)
def solve_figure_eight(*, guess: tuple[float, float, float] = FIGURE_EIGHT_GUESS,
                       mass: float = 1.0, rtol: float = NBODY_RTOL,
                       atol: float = NBODY_ATOL, method: str = "DOP853",
                       power: float = 2.0,
                       perm: tuple[int, int, int] = EULER_PERMUTATION
                       ) -> FigureEight:
    """Sekizin baslangic kosullarini ARAR -- LITERATURDEN ALMAZ.

    Bilinmeyen `(vx, vy, T)`, kosul `Phi_{T/3}(y0) = P_{(2,0,1)} y0`
    (12 artik, `scipy.optimize.least_squares`, `xtol = ftol = gtol = 1e-15`).

    LITERATURDEN ALMAK CIPAYI DAIRESEL YAPARDI ve fark OLCULDU: literaturun
    8 haneli IC'si (Chenciner-Montgomery) EN IYI `T`siyle bile
    (`T = 6.325914012110`) tam periyot donusunde
    `max|Phi_T(y0) - y0| = 7.594684e-10` veriyor; BVP'nin bulduguyla ayni
    olcum **8.690826e-13**, yani **873.9 kat** daha iyi. (Plan 7.595e-10 /
    8.646e-13 / 878 diyordu: literatur yarisi BIREBIR, BVP yarisi son
    hanesinde ayriliyor.)

    OLCULEN (DOP853, rtol 1e-13, atol 1e-15, guess (0.35, 0.53, 6.3)):

        vx = 0.347116888119   vy = 0.532724945388   T = 6.325913982926
        max|artik| = 2.498002e-14      0.19 - 0.22 s (uc kosum, onbellek bos)

    Uc sayi da planin uc sayisiyla BIREBIR ayni cikti (12 hane).

    ARAMA NOKTASI BIR CEVAP TASIMIYOR: `guess` uc haneye yuvarlanmistir ve
    havza olculmustur -- ama HAVZA BIR KUTUYA GORE TANIMLIDIR ve plan kutuyu
    HIC YAZMAMISTI, o yuzden planin "%58.2"si yeniden uretilemez. Iki kutu
    ILAN EDILEREK olculdu (3x3x3 esit arali izgara, rtol 1e-9, yakinsama
    olcutu `residual < 1e-7` ve `|T - T_ref| < 1e-5` ve `||vx|-|vx_ref|| < 1e-5`):

        vx [0.25, 0.45] x vy [0.45, 0.62] x T [5.8, 6.8]  ->  26/27 = %96.3
        vx [0.10, 0.60] x vy [0.30, 0.80] x T [4.5, 8.5]  ->  17/27 = %63.0

    Yani sayi kutunun kendisidir; "havza %X" beyani KUTUSUZ bir sayi DEGILDIR.
    """
    if len(guess) != 3:
        raise InvalidInputError(
            f"guess uc bilesenlidir (vx, vy, T); verilen {guess!r}.")
    if guess[2] <= 0.0:
        raise InvalidInputError(
            f"periyot tahmini pozitif olmali; verilen T = {guess[2]}.")

    def artik(p):
        return figure_eight_residual(p[0], p[1], p[2], mass=mass, perm=perm,
                                     rtol=rtol, atol=atol, method=method,
                                     power=power)

    sol = least_squares(artik, np.array(guess, dtype=float),
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return FigureEight(vx=float(sol.x[0]), vy=float(sol.x[1]),
                       period=float(sol.x[2]),
                       residual=float(np.max(np.abs(sol.fun))),
                       mass=float(mass), power=float(power))


def figure_eight_state(fe: FigureEight) -> NBodyState:
    """`fe`nin `t = 0` durumu."""
    return euler_state(fe.vx, fe.vy, mass=fe.mass)


def return_defect(fe: FigureEight, *, rtol: float = NBODY_RTOL,
                  atol: float = NBODY_ATOL, method: str = "DOP853") -> float:
    """`max |Phi_T(y0) - y0|` -- TAM PERIYOT DONUSU.

    BU BIR CIPA DEGILDIR, VE SEBEBI CEBIRSELDIR: `Phi_T = (Phi_{T/3})^3` ve
    `P^3 = I`, dolayisiyla `Phi_{T/3}(y0) = P y0` saglaninca tam periyot
    donusu ZORUNLU olarak saglanir. Bu sayi INTEGRATORU olcer (uc kez
    uygulanan bir haritanin birikmis hatasini), FIZIGI DEGIL.
    """
    y0 = figure_eight_state(fe)
    son = integrate(y0, t_span=fe.period, rtol=rtol, atol=atol, method=method,
                    power=fe.power)
    return float(max(
        np.max(np.abs(np.asarray(son.positions) - np.asarray(y0.positions))),
        np.max(np.abs(np.asarray(son.velocities) - np.asarray(y0.velocities))),
    ))


# ---------------------------------------------------------------------------
# MONODROMI -- varyasyonel denklem
# ---------------------------------------------------------------------------
def _var_rhs(t, z, m, power):
    n = m.size
    d = 4 * n
    y = z[:d]
    Phi = z[d:].reshape(d, d)
    q = y[:2 * n].reshape(n, 2)
    v = y[2 * n:].reshape(n, 2)
    J = np.zeros((d, d))
    J[:2 * n, 2 * n:] = np.eye(2 * n)
    J[2 * n:, :2 * n] = acceleration_jacobian(m, q, power=power)
    dz = np.empty(d + d * d)
    dz[:2 * n] = v.ravel()
    dz[2 * n:d] = accelerations(m, q, power=power).ravel()
    dz[d:] = (J @ Phi).ravel()
    return dz


def variational_flow(state: NBodyState, *, t_span: float,
                     rtol: float = NBODY_RTOL, atol: float = NBODY_ATOL,
                     method: str = "DOP853", power: float = 2.0):
    """`d Phi_t / d y0` -- `(4N, 4N)` DOGRUSAL HARITA.

    VARYASYONEL DENKLEMLE, sonlu farkla DEGIL. Gerekce olculdu ve
    `acceleration_jacobian` dokstringinde yazili: sekizin monodromisinde
    `lambda = 1` kumesi DEFEKTIFTIR, defektif ozdeger bozulmaya `sqrt` ile
    tepki verir, yani Jacobian'daki `h`-mertebesinde bir hata ozdegerde
    `sqrt(h)` olur.

    `monodromy` bunun periyodik yorunge ozel halidir; ayri tutulmasinin
    sebebi olculdu: spec 4.3'un N-body gozlenebiliri ("`m2 = 0` -> dogrusal
    harita birim") PERIYODIK OLMAYAN bir pencerede, sabit `t_span`de okunur
    (`observable/grav_obs.third_body_influence`).

    UC ALANI OKUR, baska hicbir seyi -- kaydeden vekil bunu da sayiyor.
    """
    masses = state.masses
    positions = state.positions
    velocities = state.velocities
    _kapi(masses, positions, velocities, power=power)
    if method not in NBODY_METHODS:
        raise InvalidInputError(
            f"method={method!r} sevk edilen listede yok: {NBODY_METHODS}.")
    m = np.asarray(masses, dtype=float)
    d = 4 * m.size
    z0 = np.concatenate([
        np.asarray(positions, float).ravel(),
        np.asarray(velocities, float).ravel(),
        np.eye(d).ravel(),
    ])
    sol = solve_ivp(_var_rhs, (0.0, float(t_span)), z0,
                    args=(m, float(power)), method=method, rtol=rtol,
                    atol=atol)
    if not sol.success:
        raise InvalidInputError(
            f"varyasyonel integrasyon basarisiz: {sol.message}")
    return sol.y[d:, -1].reshape(d, d)


def monodromy(fe: FigureEight, *, rtol: float = NBODY_RTOL,
              atol: float = NBODY_ATOL, method: str = "DOP853"):
    """Tam periyotta `d Phi_T / d y0` -- `(12, 12)`. `variational_flow`un
    periyodik ozel hali; TEK GOVDE, ikinci bir kopya yok."""
    return variational_flow(figure_eight_state(fe), t_span=fe.period,
                            rtol=rtol, atol=atol, method=method,
                            power=fe.power)
