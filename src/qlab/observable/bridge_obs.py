"""Tabani OLCULEN tek kenarin (`hf -> md`) TABANI -- ve o taban IKI SAYIDIR.

(Bu dosya yazildiginda `hf -> md` tablonun TEK gercek koprusuydu. Artik
degil: 22.09.2026'da `md -> tdse`, 24.09.2026'da `tb -> md` da
GERCEK_KOPRU oldu. Ama bu dosyanin olctugu taban yalniz `hf -> md`nindir:
`md -> tdse`nin sayisi `tests/tdse/test_from_md.py`de canli olculur,
`tb -> md`nin kenar tabani ise HIC OLCULMEDI -- `bridge_map`teki gerekcesi
bunu yazar.)

Modul 6 bir KUSUR BEYAN ETMISTI: taban yalnizca aritmetigi olcer, cunku
kesme hatasi fizikle birlikte kapanir ve sinav onu yapisal olarak goremez.
Orada bu bir kusur olarak KALDI. Burada KAPANIYOR -- cunku burada baskin
hata bileseninin bir ADI (anharmoniklik) ve bir DUGMESI (Morse'un `a`si)
var.

    A1 -- FIZIGI KAPAT. `a -> 0` merdiveninde yuzey tam harmonige duser ve
          dogru cevap kapali formdadir. Bu ARITMETIGI olcer.
    A2 -- FIZIGI ANALITIK BIR IKIZLE DEGISTIR. `a`, borunun KENDI olctugu
          A^2-egimine esitlenir. Bu HATA CUBUGUNU olcer.

OLCULDU (30.08.2026, asagidaki pencerede):

    sinyal    5481.241299201833 cm^-1 (fizik ACIK, ayni boru, ayni pencere)
    A1 taban  3.3986706512223464e-05 cm^-1  S/G 161276035.8886343  SAGLAM
    A2 taban  1.2025992275084718e-02 cm^-1  S/G 455782.87211757083 SAGLAM
    gercek hata 1.4472808607933985e-02 cm^-1

    gercek/A1 = 425.83733739333576 (durust hane farki 2.6292437376870614)
    A2/gercek = 0.8309370075198864 (durust hane farki 0.08043189838610851)

TABAN SEKIZ ANLAMLI HANEDEN FAZLASINA YAZILAMAZ -- olculdu, ve bu bir
tahmin degil bir SADELESME: artik `|omega_B - TAM|` bir cikarmadir ve
`ulp(TAM_CM)/taban = 2.6760307415442646e-08`. Nitekim ayni taban cm^-1'de
cikarilirsa 3.3986706512223464e-05, atomik birimde cikarilip sonra
cevrilirse 3.398670604389558e-05 cikiyor (bagil fark
1.3779737351171661e-08). Sevk edilen deger BIRINCISIDIR ve `EdgeFloor`
cm^-1 tasidigi icin oyle olmak zorundadir; ikisinin farki testte kilitli.

HUKUM: FIZIGI KAPATMAK *TABANI* OLCER; FIZIGI ANALITIK BIR IKIZLE
DEGISTIRMEK *HATA CUBUGUNU* OLCER. IKISI AYRI SAYIDIR VE PANEL IKISINI DE
YAZAR. Yalniz A1 sevk edilseydi panel gercek hatayi 426 KAT IYIMSER
gosterirdi.

BORU: `tests/md/test_aimd.py`nin YOL B'si harfi harfine -- `omega_A` -> `dt`
-> uc genlikte `run` -> sifir gecisi -> tam periyot -> velocity-Verlet
dispersiyon tersi -> `A^2`de `polyfit` kesisimi. Kenar YALNIZ `ForceField`
nesnesi degistirilerek kesilir: `engine/md`nin tek satiri degismez ve
`hf.scf.solve` SIFIR kez cagrilir (testte kilitli). Dondurulan yuzey
`Morse(d_e = K/(2a^2), r_e, a)`; `K` sabit oldugu icin yuzeyin harmonik
frekansi `a`dan BAGIMSIZDIR ve kapali formdadir:
`sqrt(K/mu) = 0.024974427755723092` a.u. = `5481.253325194108` cm^-1.

`Morse` BIR TERCIH DEGIL -- YANLIS IKIZ PENCEREYI SESSIZCE DEGISTIRIR.
Sevk edilen `Harmonic` MERKEZ-TABANLIDIR: her atomu kendi merkezine baglar,
yani bagi indirgenmis kutle uzerinden GORMEZ. Olculdu, ve tehlikeli olan
sudur: iki yuzeyin `omega_A`si BIT BIT AYNI cikiyor
(`5481.253325194112` ikisinde de), cunku statik ikinci turev yalnizca
`V''(r_e)`yi gorur; ayrilan sey DINAMIKTIR -- `omega_B` `3875.8313271219286`
cm^-1, yani `TAM/sqrt(2)`. Statik yol yanlis ikizi GOREMEZ. `Morse` bagi
indirgenmis kutle uzerinden gordugu icin borunun `mu` konvansiyonunu ve
`dt` penceresini AYNEN korur.

TABAN KIPIRDIYOR -- DORT DUGME, HEPSI OLCULDU (`a = 0`, oteki pencere sabit):

  1. `dt` merdiveni, N = 50 -> 1600 (cm^-1):
     4.159394e-03 / 4.418242e-04 / 3.219782e-05 / 2.162532e-06 /
     1.399601e-07 / 8.826032e-09; ardisik oranlar
     9.41 / 13.72 / 14.89 / 15.45 / 15.86 -> `16 = 2^4`.
  2. KARELEMEDEN BAGIMSIZ KAPALI-FORM ONGORU `artik/omega = (omega*dt)^4/144`;
     olculen/ongoru ayni merdivende 0.4382 / 0.7448 / 0.8684 / 0.9332 /
     0.9663 / 0.9750, yani ongoruye YAKINSIYOR.
  3. Kuvvet x1.001: taban 3.219781842439581e-05 -> 2.7398892935271792 cm^-1,
     yani 85095.49490008943 KAT. Kapali form dogruluyor:
     `TAM*(sqrt(1.001) - 1) = 2.73994184829577` cm^-1.
  4. Kutle x4: `dt` DONDURULMUS pencerede taban 1.1232075321703926e-06
     cm^-1'e DUSER (28.66595664843827 kat KUCULUR). Ayni bozulma `dt` boru
     kuralindan yeniden turetilerek yapilirsa carpan TAM 0.5 cikar -- ve bu
     kapali formun ikinci, bagimsiz dogrulamasidir: `omega*dt` degismedigi
     icin `artik = omega*(omega*dt)^4/144` yalnizca `omega` ile, yani tam
     yariya iner.

`a = 0` NOKTASI DEJENERE DEGIL -- MODUL 6'NIN TAM TERSI. Modul 6'da fizigi
kapatmak bir NOKTA olamiyordu (integrand sabite dusuyor, sonuc bit bit 0.0)
ve onarim bir MERDIVENDI. Burada cebirsel `a = 0` noktasi
`1.4670405515393448e-10` a.u. donduruyor, `== 0.0` -> False, ve `a -> 0`
merdiveni tam bu noktaya yakinsiyor (`a = 1e-4` -> `1.4674508136414133e-10`).
Merdiven burada SIGORTADIR, ZORUNLULUK DEGIL -- ve bunu olcerek biliyoruz.

ELENEN UC ADAY, SAYISIYLA:

  (b) `step -> 0` (HF gradyan adimi) -- KIPIRDIYOR AMA FIZIGI KAPATMIYOR.
      Olculdu: `omega_ref`ten sapma h = 1e-4 / 5e-5 / 1e-5 icin
      1.4472299906628905e-02 / 1.4472808607933985e-02 /
      1.4473752252342820e-02 cm^-1 -- limitte SABIT (bagil oynama 1.0e-04).
      Ustelik bu adayin TAM'i YOK: tek TAM adayi ayni HF cozumunun oteki
      tuketicisi olurdu, o da KARISIK BORU YASAGIDIR.
  (c) baz daraltma -- YON BOS. Olculdu: H2'de `sto-3g` 2 fonksiyon (zaten
      minimal), tek alternatif `6-31g` 4 fonksiyon, yani GENISLETIYOR;
      `sto-1g` `BasisNotAvailableError` ile reddediliyor.
  (d) kenari tam kesmek -- boru patliyor. Olculdu: `a = 0`da `K/(2*a*a)`
      Python `float`inda `ZeroDivisionError`, `numpy.float64` ile `inf`;
      `Morse(a=0)` ise `InvalidInputError`.

OLU BASAMAK OLCUTU MODUL 6'DAN GENISLETILEREK DEVRALINDI. Orada olcut
`deger == 0.0` idi; burada yetmiyor ve bu OLCULDU: `nan == 0.0` -> False,
yani bir `nan` OLU sayilmaz ve `max()`e sizar -- `max([nan, 1.0, 2.0])`
`nan`, ama `max([1.0, nan, 2.0])` `2.0` donuyor, yani hukum ARGUMAN
SIRASINA baglanir. Genisletilmis olcut `not math.isfinite(deger) or
deger == 0.0`dir ve `grav_obs.newton_noise_floor` icinde durur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from qlab.engine.md.forcefields import Morse
from qlab.engine.md.integrator import run
from qlab.engine.md.state import ForceField, MDState
from qlab.errors import InvalidInputError
from qlab.observable.grav_obs import newton_noise_floor
from qlab.units import HARTREE_TO_WAVENUMBER, amu_to_electron_mass

#: HF/STO-3G yuzeyinin kendi yay sabiti, Ha/bohr^2. PENCERESI VAR: statik
#: ikinci fark `h = 0.002` ile okundu (`tests/md/test_aimd.py:38` ayni
#: sayiyi tasiyor). `h` bir pencere dugmesidir -- olculdu, h = 0.008 /
#: 0.004 / 0.002 / 0.001 / 0.0005 icin k = 0.5729658789227488 /
#: 0.5729422212219504 / 0.5729363070638982 / 0.5729348293570524 /
#: 0.5729344607630082.
K_HF: float = 0.5729363

#: H2 denge bag uzunlugu, bohr. OLCULDU 30.08.2026: bu dizge Task 2'den
#: once depoda TAM DORT dosyada geciyordu ve DORDU DE TESTTI
#: (`tests/md/test_aimd.py`, `tests/md/test_hf_gradient.py`,
#: `tests/md/test_density_film.py`, `tests/hf/test_density_observable.py`);
#: burasi `src/` agacindaki ILK kaydidir. Bu sayi ON HANEYE YAZILAMAZ --
#: spec 4'un olcumu: yontem penceresi ~5e-08 bohr ve o basamaktan sonrasi
#: okunamaz. Burada bir DENGE NOKTASI degil, DONDURULAN YUZEYIN MERKEZI
#: olarak kullaniliyor; taban icin gereken sey tam minimum olmasi degil,
#: `Morse` ile kapali formun AYNI merkezi gormesidir.
R_E_BOHR: float = 1.3459193602

#: mu = m(1H)/2. KUTLE BORU HATTINDAN GECER, literal DEGIL.
MU_H2: float = float(amu_to_electron_mass(1.00782503207)) / 2.0

#: Dondurulan yuzeyin KAPALI FORMDAKI dogru cevabi. `a`dan bagimsizdir:
#: `V''(r_e) = 2*d_e*a^2 = K` her `a` icin.
TAM_AU: float = math.sqrt(K_HF / MU_H2)
TAM_CM: float = TAM_AU * HARTREE_TO_WAVENUMBER

#: Borunun KENDI olctugu A^2-egimini eslestiren `a`. HF yuzeyinde egim
#: -0.010024824447837994, dondurulmus Morse'ta `a = 0.9137`de
#: -0.010023011049457173 (bagil fark 1.8e-04).
A_ESLESEN: float = 0.9137

#: `a -> 0` merdiveni. En bastaki `0.03` tabani BELIRLER ve artik merdiven
#: ICINDE `a` ile monoton buyur; en sondaki `0.0` CEBIRSEL noktadir.
#: MERDIVENIN DISINDA MONOTONLUK YOK -- olculdu: `a = 0.05`te artik
#: 3.346970967886756e-05 cm^-1'e DUSUYOR ve `a = 0.1`de
#: 1.757641213185146e-05'e. Yani `0.03` bir SINIR degil bir TEPEDIR; taban
#: "en buyuk artik" olarak okundugu icin merdiveni buyutmek onu buyutmez.
A1_MERDIVENI: tuple[float, ...] = (0.03, 0.02, 0.01, 0.003, 0.001, 1e-4, 0.0)

#: YOL B penceresi -- `test_aimd.py`nin kullandigi degerlerin AYNISI.
N_BOLME: int = 200
PERIYOT_SAYISI: int = 6
GENLIKLER: tuple[float, ...] = (0.01, 0.02, 0.04)
H_IKINCI_TUREV: float = 0.002

#: Kenarin GERCEK hatasi, cm^-1 -- ve penceresi sudur: HF borusunun
#: `omega_B`si (`step = 5e-5`) `5481.229456091029`, HF yuzeyinin kendi
#: harmonik frekansi `omega_ref` `5481.243928899637` (statik ikinci farkin
#: Richardson kesisimi, `h` cifti `(0.002, 0.004)`). CIFT BIR PENCERE
#: DUGMESIDIR, olculdu: (0.001, 0.002) -> 1.4478119341220008e-02,
#: (0.004, 0.008) -> 1.4471104747826757e-02, (0.0005, 0.001) ->
#: 1.4483430074506032e-02. Bu sayi `tests/bridge/test_kenar_tabani.py`de
#: HF borusu KOSTURULARAK yeniden uretilir; burada bir literal degil, o
#: kosumun kaydidir.
GERCEK_HATA_CM: float = 0.014472808607933985

#: `EdgeFloor.tur`un iki degeri. AYRI KOVADIR: biri aritmetigi, oteki hata
#: cubugunu olcer ve ikisi birbirinin yerine yazilamaz.
TURLER: tuple[str, ...] = ("aritmetik", "hata-cubugu")


@dataclass(frozen=True)
class EdgeFloor:
    """Bir KENARIN tabani -- ve tabanin HANGI SORUYU cevapladigi.

    `taban`  : cm^-1. Fizik kapali (ya da ikizle degistirilmis) borudan
               cikan en buyuk mutlak sapma.
    `sinyal` : cm^-1. Ayni borunun ayni pencerede fizik ACIKKEN verdigi
               deger.
    `oran`   : `sinyal / taban`.
    `boru`   : hangi boru, hangi pencere, hangi merdiven. KARISIK BORU
               YASAGININ kaydi.
    `hukum`  : `grav_obs.VERDICTS`ten biri.
    `tur`    : `TURLER`den biri. BU ALAN OLMADAN IKI SAYI TEK HUCREYE
               DUSER ve panel hangi soruyu cevapladigini soyleyemez.
    """

    taban: float
    sinyal: float
    oran: float
    boru: str
    hukum: str
    tur: str


@dataclass(frozen=True)
class _BagHarmonik:
    """Morse'un `a -> 0` CEBIRSEL limiti: `V = k/2 (r - r_e)^2`, BAG uzerinden.

    Sevk edilen `Harmonic` BUNUN YERINE KULLANILAMAZ: o her atomu kendi
    merkezine baglar ve bagi indirgenmis kutle uzerinden gormez (modul
    dokstringi, olculdu). Burada bir yuzey degil bir LIMIT sevk ediliyor --
    `Morse` `a = 0`da tanimsizdir (`d_e = K/(2a^2)` sonsuza gider), oysa
    limit yuzeyi tamamen sonludur ve `a = 0` NOKTASININ var olmasini
    saglayan sey budur.
    """

    k: float
    r_e: float

    def __post_init__(self) -> None:
        for ad, deger in (("k", self.k), ("r_e", self.r_e)):
            if not math.isfinite(deger) or deger <= 0.0:
                raise InvalidInputError(f"{ad} pozitif ve sonlu olmali: {deger}")

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        if np.shape(positions) != (2, 3):
            raise InvalidInputError(
                f"iki atom bekleniyor, verilen sekil: {np.shape(positions)}"
            )
        delta = positions[1] - positions[0]
        r = float(np.linalg.norm(delta))
        if r == 0.0:
            raise InvalidInputError("cakisik atomlar: bag yonu tanimsiz")
        x = r - self.r_e
        enerji = 0.5 * self.k * x * x
        dv_dr = self.k * x
        birim = delta / r
        return enerji, np.vstack([dv_dr * birim, -dv_dr * birim])


def _donmus_yuzey(a: float) -> ForceField:
    """Dondurulmus HF yuzeyi. `a = 0` CEBIRSEL limittir, bir yaklasim degil."""
    if not math.isfinite(a) or a < 0.0:
        raise InvalidInputError(f"a sonlu ve negatif olmayan olmali: {a}")
    if a == 0.0:
        return _BagHarmonik(k=K_HF, r_e=R_E_BOHR)
    return Morse(d_e=K_HF / (2.0 * a * a), r_e=R_E_BOHR, a=a)


def _omega_a(
    force_field: ForceField,
    *,
    h: float = H_IKINCI_TUREV,
    mu: float = MU_H2,
    r_e: float = R_E_BOHR,
) -> float:
    """YOL A: statik ikinci turev. Kuvvet KULLANILMAZ. Birim: a.u."""

    def E(r: float) -> float:
        return force_field(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]))[0]

    k = (E(r_e + h) + E(r_e - h) - 2.0 * E(r_e)) / h**2
    return float(np.sqrt(k / mu))


def omega_b(
    force_field: ForceField,
    *,
    mu: float = MU_H2,
    masses: np.ndarray | None = None,
    r_e: float = R_E_BOHR,
    n_bolme: int = N_BOLME,
    periyot_sayisi: int = PERIYOT_SAYISI,
    genlikler: Sequence[float] = GENLIKLER,
    h: float = H_IKINCI_TUREV,
    dt: float | None = None,
) -> float:
    """YOL B: dinamik periyot. IKINCI TUREV ALINMAZ. Birim: a.u.

    `tests/md/test_aimd.py`nin YOL B'sinin AYNISIDIR ve bu bir iddia degil
    bir OLCUMDUR: ayni cagri HF alaniyla yapildiginda `omega_B/omega_A - 1`
    o testin dokstringindeki sayiyi BIT BIT veriyor
    (`-4.360844408468623e-06`); testte kilitli.

    IKI DUZELTME SART, ikisi de o testte olculdu: (1) sonlu genlikte frekans
    duser, o yuzden `A^2`de dogrusal uydurulup `A -> 0`a ekstrapole edilir;
    (2) velocity-Verlet dispersiyonunun TERSI uygulanir,
    `omega = (2/dt) sin(Omega dt/2)`.

    `dt` verilmezse boru kuralindan turetilir: `dt = (2*pi/omega_A)/n_bolme`.
    Verilirse PENCERE DONDURULMUS olur -- kutle dugmesi bunu kullanir ve iki
    kullanim AYNI SAYIYI VERMEZ (modul dokstringi, dorduncu dugme).
    """
    return _a2_uydurmasi(
        force_field, mu=mu, masses=masses, r_e=r_e, n_bolme=n_bolme,
        periyot_sayisi=periyot_sayisi, genlikler=genlikler, h=h, dt=dt,
    )[1]


def a2_egimi(force_field: ForceField, **pencere: Any) -> float:
    """`A^2`de uydurulan dogrunun EGIMI -- A2 ikizini eslestiren buyukluk.

    `omega_b` KESISIMI dondurur; anharmoniklik EGIMDEDIR. A2 probu tam bu
    egim uzerinden secildi: HF yuzeyinde -0.010024824447837994, dondurulmus
    Morse'ta `a = 0.9137`de -0.010023011049457173.
    """
    return _a2_uydurmasi(force_field, **pencere)[0]


def _a2_uydurmasi(
    force_field: ForceField,
    *,
    mu: float = MU_H2,
    masses: np.ndarray | None = None,
    r_e: float = R_E_BOHR,
    n_bolme: int = N_BOLME,
    periyot_sayisi: int = PERIYOT_SAYISI,
    genlikler: Sequence[float] = GENLIKLER,
    h: float = H_IKINCI_TUREV,
    dt: float | None = None,
) -> tuple[float, float]:
    """`(egim, kesisim)` -- `omega_b` ile `a2_egimi` TEK kosumu paylasir.

    Ayri iki govde yazilsaydi ikisi sessizce ayrisabilirdi; ve bu depoda
    ayni sayinin iki govdeden okunmasi zaten bir kez yanlis etiket uretti
    (`grav_obs.newton_noise_floor`, `PIPELINE_KEYS`).
    """
    if masses is None:
        masses = np.full(2, float(amu_to_electron_mass(1.00782503207)))
    if n_bolme < 1 or periyot_sayisi < 1:
        raise InvalidInputError(
            f"n_bolme ve periyot_sayisi pozitif olmali: {n_bolme}, {periyot_sayisi}"
        )
    if len(genlikler) < 2:
        raise InvalidInputError(
            f"A^2 uydurmasi en az iki genlik ister, verilen: {tuple(genlikler)}"
        )
    wa = _omega_a(force_field, h=h, mu=mu, r_e=r_e)
    adim = (2.0 * np.pi / wa) / n_bolme if dt is None else float(dt)

    noktalar: list[tuple[float, float]] = []
    for genlik in genlikler:
        durum = MDState(
            positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r_e + genlik]]),
            velocities=np.zeros((2, 3)),
            masses=masses,
        )
        yorunge = run(
            durum, force_field, dt=adim, n_steps=n_bolme * periyot_sayisi,
            save_every=1,
        )
        bag = np.linalg.norm(
            yorunge.positions[:, 1, :] - yorunge.positions[:, 0, :], axis=1
        )
        sapma = bag - r_e
        t = yorunge.times
        gecis = np.nonzero(np.diff(np.sign(sapma)) != 0)[0]
        anlar = [
            t[i]
            + (t[i + 1] - t[i]) * abs(sapma[i]) / (abs(sapma[i]) + abs(sapma[i + 1]))
            for i in gecis
        ]
        # TAM periyot: ayni yondeki gecisler arasi.
        periyot = float(np.mean(np.diff(np.asarray(anlar)[::2])))
        omega_ayrik = 2.0 * np.pi / periyot
        noktalar.append(
            (genlik**2, (2.0 / adim) * np.sin(omega_ayrik * adim / 2.0))
        )

    egim, kesisim = np.polyfit(
        np.array([n[0] for n in noktalar]), np.array([n[1] for n in noktalar]), 1
    )
    return float(egim), float(kesisim)


def _artik_au(*, a: float, mu: float = MU_H2, **pencere: Any) -> float:
    """`|omega_B - TAM|` ATOMIK BIRIMDE (`EdgeFloor.taban` cm^-1'dedir).

    Iki birim BILEREK ayri adlandirildi: bu fonksiyonun adindaki `_au`
    olmasaydi `3.2e-05` ile `1.5e-10` ayni panelde ayni sutuna duserdi.
    """
    tam = math.sqrt(K_HF / mu)
    return abs(omega_b(_donmus_yuzey(a), mu=mu, **pencere) - tam)


def _boru_kur(*, tur: str, a_acik: float,
              probeler: tuple[tuple[dict[str, Any], float], ...],
              etiket: str) -> Callable[..., float]:
    """`newton_noise_floor`un sozlesmesini TASIYAN boruyu uretir.

    Sozlesme devralindi: cagrilan `PIPELINE` ve `NEWTON_PROBES` beyan eder,
    cagiran bir probe anahtari VEREMEZ. `a` bir probe anahtaridir, yani
    `newton_noise_floor(fn, a=...)` `InvalidInputError` alir -- fizigi
    kapatma noktasi cagirana birakilmaz.
    """

    def fn(
        *,
        a: float = a_acik,
        n_bolme: int = N_BOLME,
        periyot_sayisi: int = PERIYOT_SAYISI,
        genlikler: tuple[float, ...] = GENLIKLER,
    ) -> float:
        return (
            omega_b(
                _donmus_yuzey(a),
                n_bolme=n_bolme,
                periyot_sayisi=periyot_sayisi,
                genlikler=genlikler,
            )
            * HARTREE_TO_WAVENUMBER
        )

    fn.PIPELINE = etiket  # type: ignore[attr-defined]
    fn.NEWTON_PROBES = probeler  # type: ignore[attr-defined]
    fn.PIPELINE_KEYS = ("n_bolme", "periyot_sayisi", "genlikler")  # type: ignore[attr-defined]
    fn.__name__ = f"_kenar_borusu_{tur.replace('-', '_')}"
    return fn


#: `edge_noise_floor`un gecirdigi PENCERE anahtarlari -- `_boru_kur`un
#: `PIPELINE_KEYS`iyle AYNI kume, ve bu bir tekrar degil bir kapidir:
#: senaryo (Task 5) pencereyi buradan gecirir, tanimadigi bir anahtar
#: `InvalidInputError` alir; `a` bu kumede DEGILDIR (A1'de cagirana kapali,
#: A2'de adli parametre).
PENCERE_ANAHTARLARI: tuple[str, ...] = ("n_bolme", "periyot_sayisi", "genlikler")


def edge_noise_floor(
    *, tur: str, a: float | None = None, **pencere: Any
) -> EdgeFloor:
    """Kenarin tabanini olcer -- ve HANGI SORUYU cevapladigini soyler.

    `pencere` (Task 5): `PENCERE_ANAHTARLARI`ndan `n_bolme`, `periyot_sayisi`,
    `genlikler` -- YOL B'nin dugmeleri. Senaryo bunlari YAML'dan gecirir;
    verilmezse `bridge_obs`un sevk edilen varsayilanlari (N = 200, 6 periyot,
    (0.01, 0.02, 0.04)) kullanilir ve etiket onlari yine yazar. Sinyal AYNI
    pencerede okunur (`newton_noise_floor` `fn(**kw)`yi ayni `kw` ile
    cagirir), yani karisik pencere de yasaktir. Taban `dt`ye dorduncu
    mertebeden baglidir (modul dokstringi): N = 200 -> 400 orani 14.89.

    `tur = "aritmetik"` (A1): fizik KAPATILIR. Prob demeti `A1_MERDIVENI`dir
    ve cagiran onu SECEMEZ -- `a` verilirse `InvalidInputError`. Bu, Modul
    6'nin "fizigi kapatma noktasini cagrilan belirler" sozlesmesinin ayni
    kuralidir; A1'de kapatma noktasi bir TASARIM KARARIDIR, bir dugme degil.

    `tur = "hata-cubugu"` (A2): fizik bir ANALITIK IKIZLE degistirilir. `a`
    burada ANLAMLIDIR cunku HANGI ikiz oldugunu soyler; varsayilan
    `A_ESLESEN`dir ve secimi olculmustur (A^2-egimi eslesmesi). `a`nin bir
    dugme olmasi burada bir kacak DEGIL, sinavin kendisidir: ikiz
    degistirilince taban degismelidir. Olculdu: `a = 0.5`te taban
    2.006471427193901e-03 cm^-1, yani sevk edilen degerin
    0.16684456311774626'si.

    Iki cagri AYRI PANEL SATIRIDIR. Birini otekinin yerine yazmak, Modul
    6'nin kusur beyanini onarilmis gibi gostermek olur.
    """
    if tur not in TURLER:
        raise InvalidInputError(f"tur {TURLER} icinden olmali, verilen: {tur!r}")
    yabanci = sorted(set(pencere) - set(PENCERE_ANAHTARLARI))
    if yabanci:
        raise InvalidInputError(
            f"pencere anahtari taninmiyor: {yabanci}; gecerli anahtarlar "
            f"{PENCERE_ANAHTARLARI}. `a` bir pencere anahtari DEGILDIR."
        )

    if tur == "aritmetik":
        if a is not None:
            raise InvalidInputError(
                "A1'de `a` verilemez: fizigi kapatma merdivenini CAGRILAN "
                "belirler. Cagiran secebilseydi 'fizigi kapat' sinavi "
                "cagirana birakilmis bir dugme olurdu."
            )
        probeler = tuple(({"a": x}, TAM_CM) for x in A1_MERDIVENI)
        a_acik = A_ESLESEN
        etiket = (
            "YOL B periyot borusu | dondurulmus Morse yuzeyi | "
            "fizik KAPALI: a -> 0"
        )
    else:
        ikiz = A_ESLESEN if a is None else float(a)
        if not math.isfinite(ikiz) or ikiz <= 0.0:
            raise InvalidInputError(f"ikiz `a` pozitif ve sonlu olmali: {ikiz}")
        probeler = (({"a": ikiz}, TAM_CM),)
        a_acik = ikiz
        etiket = (
            "YOL B periyot borusu | dondurulmus Morse yuzeyi | "
            f"fizik IKIZLE DEGISTIRILDI: a = {ikiz}"
        )

    fn = _boru_kur(tur=tur, a_acik=a_acik, probeler=probeler, etiket=etiket)
    kayit = newton_noise_floor(fn, **pencere)
    return EdgeFloor(
        taban=kayit.value,
        sinyal=kayit.signal,
        oran=kayit.ratio,
        boru=kayit.pipeline,
        hukum=kayit.verdict,
        tur=tur,
    )
