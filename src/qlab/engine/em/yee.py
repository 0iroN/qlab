"""Yee guncellemesi. BU DOSYA ORTAMI BILMEZ.

Gordugu tek sey `medium.eps_r` / `medium.mu_r` ALANLARI ve sinirin `sigma`
DIZISIDIR; arayuz mu, silindir mi, bosluk mu -- hicbiri buraya girmez. Bu bir
stil tercihi degil, Modul 5'in iki fizik cipasinin (Bessel oranlari ve
Rayleigh ussu) AYNI cekirdekten gecmesinin ON KOSULUDUR; ayri kod yollarindan
gecerlerse "ayni cozucu, iki bagimsiz hukum" iddiasi duser. (Modul 4'te
`metropolis.py`nin modeli bilmemesiyle ayni gerekce.)

Ayni gerekcenin ikinci yuzu: `yee_step_1d` ile `run_1d` AYNI ic fonksiyonu
(`_adim`) cagirir. Iki ayri guncelleme yazilsaydi "cipa cekirdegi sinar"
iddiasi yine duserdi -- cipa hangisini sinadigini bilemezdi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from qlab.engine.em.grid import (
    FieldRecord,
    FieldState,
    FieldState2D,
    Window,
    courant_limit,
    dx_kapisi,
)
from qlab.engine.em.medium import Medium
from qlab.errors import InvalidInputError


@dataclass(frozen=True, eq=False)
class YeeCoefficients:
    """Onceden hesaplanmis guncelleme katsayilari (1B).

        E: e <- ca * e + cb * (h[i] - h[i-1])
        H: h <- da * h + db * (e[i+1] - e[i])

    Kayipli ortamda (sogurucu) yari-adim ustel formu:
        d_e = sigma * S / (2 eps),   ca = (1-d_e)/(1+d_e),  cb = (S/eps)/(1+d_e)
    ve manyetik kaybin ESLESME kosulu sigma_m / mu = sigma_e / eps'tir; bu
    yuzden `d_h` yari hucrede (sigma/eps)'in ARITMETIK ortalamasindan kurulur.
    Eslesme kuralinda aritmetik kullanilmasi `medium.ey_kural`dan BAGIMSIZDIR
    ve oyle olmalidir: `ey_kural` teget E'nin sureklilik kuralidir, sogurucu
    eslesmesinin degil.
    """

    ca: np.ndarray
    cb: np.ndarray
    da: np.ndarray
    db: np.ndarray
    dim: int
    courant: float


def _etkin_courant_kapisi(medium: Medium, courant: float, limit: float, yol: str) -> None:
    """ETKIN Courant kapisi -- uc katsayi yolunun (1B, 2B, egik) ORTAK kapisi.

    `S` EN HIZLI dalga hiziyla tanimlidir (grid.courant_limit). Her hucrede
    eps_r*mu_r >= 1 ise surekli ortamda en hizli dalga c'dir, etkin S = S ve bu kapi ham kapidan
    SONRA kostugu icin yeni bir sey reddetmez (taban `min(1, .)`yi silen mutant
    bu yuzden ESDEGERDIR, olculdu); bir hucrede < 1 ise dalga c/sqrt(eps_r*mu_r)
    ile vakumdan HIZLIDIR ve sinanan sayi S/sqrt(min) olur. HUCRE carpiminin
    minimumu, minimumlarin carpimi DEGIL -- olculdu (01.10.2026, public yol,
    tohumlu, PEC, L = 200): eps 0.25|1 & mu 1|0.25'te esik 0.500019, minimumlarin
    carpimi (0.0625) 0.25 derdi. `mu_r < 1` `eps_r < 1` ile AYNI deliktir
    (0.500020 / 0.500020); senaryo semasinda mu olmadigi icin fabrika kapisi
    onu goremez. Carpim alt tasmayla 0'a inerse etkin S sonsuzdur.

    YETMEDIGI YER (olculdu 02.10.2026; ESKI acik, c4c66ad ayni adimda patliyordu):
    egik p + `ey_kural` != aritmetik + konuma bagli mu_r. p'de Hz tam hucrede mu_r
    ile, Ey yari hucrede `eps_yari` ile eslenir; esik hucre carpimiyla da basit bir
    yerel carpimla da BELIRLENMEZ ve her hucrede eps*mu = 1 iken bile izinin altina
    iner -- kontrasta ve geometriye bagli, alttan sinirsiz (eps 4|0.25, mu 0.25|4,
    L = 200'de 0.47-0.66 kati). O bandi `oblique._egik_p_kesin_esik` keser.
    """
    with np.errstate(over="ignore", under="ignore"):  # tasma -> inf (yavas dalga, gecer); alt tasma -> 0 (inf, reddeder)
        cm = float(np.min(medium.eps_r * medium.mu_r))
    etkin = courant / float(np.sqrt(min(1.0, cm))) if cm > 0.0 else float("inf")
    if etkin > limit:
        raise InvalidInputError(
            f"etkin Courant S/sqrt(min(eps_r*mu_r)) = {etkin!r} {yol} kararlilik "
            f"sinirini ({limit!r}) asiyor: min(eps_r*mu_r) = {cm!r} < 1, dalga "
            f"vakumdan hizlidir (S = {courant!r}). Ayrinti: grid.courant_limit."
        )


#: `_sonlu_katsayi_kapisi`nin |a| <= 1 sinadigi sonum yuvalari (1B ve 2B adlari).
_SONUM_YUVALARI = ("ca", "da", "dax", "day")


def _sonlu_katsayi_kapisi(yol: str, **diziler: np.ndarray) -> None:
    """Katsayilar SONLU, sonum yuvalari |a| <= 1 -- 1B ve 2B yolun ortak son kapisi (G13).

    Etkin Courant kapisi hucre CARPIMINI okur; carpim sonlu ve >= S^2 iken
    tek basina bolum tasabilir. OLCULDU (HEAD e262cab, kamu yolu `run_1d`,
    N = 200, PML 15, S = 0.5, 600 adim): bir hucrede eps_r = 2e-309 (alt-normal),
    mu_r = 1.5e308 -> carpim 0.3 >= 0.25, kapi geciyor, cb = S/eps = inf ve
    kaydin 1108 ornegi sessizce nan. Sinirin sigma dizisi sonlu degilse de
    (ozel bir sinir nesnesi) ayni kapi keser.

    SONUM YUVALARI (`ca`, `da`; 2B'de `dax`, `day`) her yerde |a| <= 1 olmali
    (G13 duzeltmesi, 12.10.2026): a = (1 - d)/(1 + d), d = sigma S/(2 eps).
    Kapili sinirlar sigma >= 0 verir ve d >= 0'da |a| <= 1 TAM tutar (|1 - d|
    <= 1 + d, yuvarlama monoton) -- sahte ret yok. Negatif sigma donduren
    ozel bir sinir nesnesi (sinir kapisi yalniz `PML`/`LossyLayer`da kosar)
    KAZANC kurar; OLCULDU (kamu yolu, bu kazanc kapisi yokken): `run_1d`, vakum
    N = 200, PEC, S = 0.5, 600 adim, 95-105. hucrelerde sigma = -0.5 (ca =
    1.286) -> kayit 6.05e+21 (referans 0.509); `run_2d`, 41 x 41, S = 0.5,
    300 adim, tek hucrede ayni sigma -> 841 (referans 0.0723).
    """
    for ad, dizi in diziler.items():
        if not np.all(np.isfinite(dizi)):
            raise InvalidInputError(
                f"{yol} guncelleme katsayisi `{ad}` sonlu degil: ortam olcegi (or. alt-normal "
                "eps_r x dev mu_r) ya da sinirin sigma dizisi tasiyor -- kapi kapali yonde reddeder."
            )
    for ad in _SONUM_YUVALARI:
        if ad in diziler and not np.all(np.abs(diziler[ad]) <= 1.0):
            raise InvalidInputError(
                f"{yol} guncelleme katsayisi `{ad}` KAZANC veriyor: her yerde |{ad}| <= 1 olmali, "
                f"en buyugu {float(np.max(np.abs(diziler[ad])))!r} -- sinirin sigma dizisi negatif "
                "(ozel sinir nesnesi); kapi kapali yonde reddeder."
            )


def yee_coefficients(
    medium: Medium, *, courant: float, boundary
) -> YeeCoefficients:
    """Katsayilar. COURANT DENETIMI BURADADIR, yani her iki yolda da kosar."""
    cells = medium.cells
    if cells < 3:
        raise InvalidInputError(f"1B izgara en az 3 hucre olmali, verilen: {cells}")
    limit = courant_limit(1)
    if not courant > 0.0:
        raise InvalidInputError(f"S > 0 olmali, verilen: {courant}")
    if courant > limit:
        raise InvalidInputError(
            f"S = {courant} kararlilik sinirini asiyor (1B'de S_max = {limit}). "
            "S EN HIZLI dalga hiziyla tanimlidir -- bu bir MODELLEME SECIMIDIR, "
            "olcum degil; ham esik ortama baglidir (eps_r = 1/2.25/4/9 icin "
            "olculen 1.000040301/1.500060657/2.000081036/3.000121823) ve "
            "degismeyen sey S_max/sqrt(eps_r)'dir. Esikteki artik asim iki "
            "parcadir ve ikisi de olculdu: sonlu IZGARA payi (PEC duvarda tam "
            "olarak 1/cos(pi/(2(L-1))), L = 200'de +3.1e-05) ve sonlu KOSUM "
            "payi (periyodik izgarada 4000 adimda +5.99e-06, 20000 adimda "
            "+2.12e-07). Ayrinti: grid.courant_limit."
        )
    _etkin_courant_kapisi(medium, courant, limit, "1B")
    sigma = np.asarray(boundary.sigma_1d(cells), dtype=float)
    if sigma.shape != (cells,):
        raise InvalidInputError(
            f"sinir sigma dizisi (cells,) olmali, verilen: {sigma.shape}"
        )
    eps = medium.eps_r
    mu_h = medium.mu_yari()

    d_e = sigma * courant / (2.0 * eps)
    ca = (1.0 - d_e) / (1.0 + d_e)
    cb = (courant / eps) / (1.0 + d_e)

    d_h = 0.5 * (d_e[:-1] + d_e[1:])
    da = (1.0 - d_h) / (1.0 + d_h)
    db = (courant / mu_h) / (1.0 + d_h)
    _sonlu_katsayi_kapisi("1B", ca=ca, cb=cb, da=da, db=db)
    return YeeCoefficients(ca=ca, cb=cb, da=da, db=db, dim=1, courant=courant)


def _adim(state: FieldState, coeffs: YeeCoefficients) -> FieldState:
    """TEK GUNCELLEME YOLU. `yee_step_1d` de `run_1d` de buradan gecer.

    Sira: once H (yarim adim ileri), sonra E (tam adim). YARIM HUCRE
    KAYDIRMA `h[i] = H(i+1/2)` uzerinden gelir: E'nin curl'u `h[i] - h[i-1]`,
    H'ninki `e[i+1] - e[i]`. Ikisi de MERKEZI farktir ve semanin 2. mertebesi
    tam olarak buradan gelir.

    Kaydirma dusurulunce olan sey MERTEBE DUSMESI DEGIL, IRAKSAMADIR: iki
    curl de AYNI yonlu farki alirsa sema patliyor. PENCERE: bu dosyanin
    katsayilari (vakum + PEC), PEC duvarli izgarada m = 8 duran kip, 400
    adim; buyukluk izdusum serisi `s[n] = <e_n, sin(k i)>`in TEPESI bolu
    `s[0]`:

                  N = 201     N = 401     N = 801     N = 1601
        S = 0.5   3.17e+159   7.99e+144   3.20e+145   7.99e+144
        S = 0.25  1.35e+65    4.38e+65    2.53e+64    1.69e+64
        S = 0.1   5.62e+14    2.70e+14    4.50e+13    1.13e+13

    Bu cekirdek ayni pencerede en fazla 1.000282 veriyor. Yani ortada
    olculebilecek bir mertebe YOK; CIPA 2'nin merdiveni ancak yakinsayan bir
    seride kurulur.

    KARARSIZLIK KOSULSUZDUR ama GORUNMESI ADIM SAYISINA BAGLIDIR, ve ikisi de
    olculdu. Kosulsuz: kaydirmasiz semanin buyutme matrisi
    `[[1 + S^2 a^2, S a], [S a, 1]]`, `a = exp(i theta) - 1`; `det = 1` ama
    `iz` karmasik, ve olculen `|lam|max` S = 0.5 / 0.25 / 0.1 / 0.05 / 0.02 /
    0.001 icin 2.617904 / 1.640341 / 1.220983 / 1.105118 / 1.040805 /
    1.002002 -- hepsi 1'in USTUNDE. Pencere: buyume adim basina gittigi ve
    buyuyen kipi YUVARLAMA ARTIGI tohumladigi icin (ayni kosum float32'de
    daha erken patliyor) S = 0.01'de tepe orani 2000 adimda hala 1.00, 4000
    adimda 1.18e+15. Yani "her S'te patliyor" cumlesi ADIM SAYISI
    soylenmeden yazilamaz -- bu dosyada bir kez oyle yazildi ve CURUDU.

    KILIT: tests/em/test_sema_cipalari.py'de iki test,
    `test_KAYDIRMA_DUSURULUNCE_...` ve `test_KAYDIRMASIZ_SEMA_KOSULSUZ_...`.
    """
    e, h = state.e, state.h
    h = coeffs.da * h + coeffs.db * (e[1:] - e[:-1])
    e_yeni = coeffs.ca * e
    e_yeni[1:-1] += coeffs.cb[1:-1] * (h[1:] - h[:-1])
    # Izgaranin iki ucu PEC duvardir; sogurucu bu duvarin ONUNDE durur.
    e_yeni[0] = 0.0
    e_yeni[-1] = 0.0
    return FieldState(e_yeni, h, 1)


def yee_step_1d(
    state: FieldState, medium: Medium, *, courant: float, boundary
) -> FieldState:
    """Bir Yee adimi. Bu fonksiyon GEOMETRIYI BILMEZ.

    Gordugu tek sey `medium.eps_r` ve `medium.mu_r` ALANLARIDIR; arayuz mu,
    silindir mi, bosluk mu -- hicbiri buraya girmez. Bu bir stil tercihi
    degil, Modul 5'in iki fizik cipasinin (Bessel ve Rayleigh) AYNI
    cekirdekten gecmesinin ON KOSULUDUR; ayri kod yollarindan gecerlerse
    "ayni cozucu, iki bagimsiz hukum" iddiasi duser. (Modul 4'te
    `metropolis.py`nin modeli bilmemesiyle ayni gerekce.)

    Katsayilar her cagride yeniden kurulur; cok adimli kosumlar icin
    `run_1d` kullanilir (o katsayilari BIR KEZ kurar). Olculdu (N = 2400,
    4000 adim, tekrarlarin en iyisi): `yee_step_1d` 0.117-0.118 s, `run_1d`
    0.030-0.031 s -- uc bagimsiz olcumde oran 3.79 / 3.81 / 3.92 kat. Yani
    `yee_step_1d` bir aciklik yuzudur, sicak yol degildir; ama IKISI DE
    `_adim`den gecer, dolayisiyla sicak yolun semasi bu fonksiyonunkiyle
    AYNIDIR.
    """
    if state.dim != 1:
        raise InvalidInputError(
            f"yee_step_1d yalniz 1B durum alir, verilen dim = {state.dim}. "
            "2B icin `yee_step_2d`/`run_2d` var; 3B ise KAPSAM DISIDIR ve "
            "sebebi OLCULDU: Airy sabiti j_1,1/pi uzerinden 2B'de "
            "capalaniyor (16.0 s), ayni adim sayisi ayni dogrusal boyutta "
            "ciplak 3B cekirdekte ~512 dakika."
        )
    if state.cells != medium.cells:
        raise InvalidInputError(
            f"durum ve ortam ayni izgarada olmali: {state.cells} != {medium.cells}"
        )
    return _adim(state, yee_coefficients(medium, courant=courant, boundary=boundary))


def run_1d(
    medium: Medium,
    *,
    courant: float,
    boundary,
    steps: int,
    source: Callable[[float], float] | None,
    source_index: int | None,
    probes: tuple[int, ...],
    wavelength: float,
    sgf: float,
    window_fn: str = "dikdortgen",
    dx: float = 1.0,
    dtype: type = float,
    state: FieldState | None = None,
) -> FieldRecord:
    """Cok adimli 1B kosum -- ve PENCERESINI dondurmek ZORUNDA.

    `wavelength`, `sgf` ve `window_fn` VARSAYILANSIZ ISTENIR (window_fn'in
    varsayilani "dikdortgen"dir cunku FFT penceresi UYGULANMAMASI da bir
    secimdir ve adi vardir). Bu, cagriyi zorlastirmak icin degil: bu depoda
    hukmun pencerenin disina tasinmasi ON KEZ olctu, ve pencereyi imzaya
    koymak onu unutulamaz kilar.

    `source` yumusak NOKTA kaynagidir: `e[source_index] += cb * source(t)`,
    `t = n * S`. Yumusak kaynak IKI YONLU isir ve yansiyan/sacilan alan
    IKI KOSUMUN FARKINDAN okunur.

    `wavelength` HUCRE cinsindendir ve bu artik bir TANIM DEGIL bir TEOREM:
    `dx_kapisi` hucre ile uzunlugu ayni sey yaptigi icin kaynagin zaman
    periyodu DOGRUDAN hucre cinsinden dalga boyudur.

    KAYNAGIN SAATI BIR ZAMANLAR *"t = n * S * dx"* IDI ve `dx` GERCEKTEN
    CARPILIYORDU -- oysa `wavelength` HUCRE cinsinden beyan ediliyordu
    (`Window.as_lines`: "lambda ... hucre"). Ikisi yalniz `dx = 1`de
    ortusuyordu, `dx != 1` SESSIZCE YANLIS calisiyordu, ve kusur Task 6'da
    BEYAN EDILEREK acik birakilmisti. Simdi OLCULEREK kapatildi: `dx` ayri
    bir serbestlik derecesi DEGILDIR, `wavelength`in olcekli bir kopyasidir
    -- davranis yalniz `wavelength/dx` oranina bagli ve bu BIT BIT olculdu
    (bkz. `grid.dx_kapisi`). `dx != 1` artik REDDEDILIYOR, dolayisiyla
    `dt = S`tir ve `* dx` carpani BU SATIRDAN KALDIRILDI. Carpani birakmak
    kapatilmis bir dugmeye 1 ile carpan OLU bir kod yolu birakirdi; kapali
    bir dugmenin dogru yeri kapidir, carpim degil.

    BU SATIR BIR ZAMANLAR *"TF/SF tek yonlu kaynak Task 3'te gelir"*
    DIYORDU ve o soz BAYATLADI: Task 3 geldi gecti, TF/SF sevk EDILMEDI ve
    depoda hicbir karsiligi yok. Task 5 ayni ihtiyaci (tek yonlu, duzlem
    dalga aydinlatma) BASKA bir yoldan cozdu ve o yol daha ucuz cikti:
    kaynak terimi HIC kullanilmiyor, izgara `state=` ile eslesmis bir
    duzlem dalga darbesiyle BASLATILIYOR (`E(x, 0) = f(x)`,
    `Hy(i+1/2, -dt/2) = f(i + 0.75)`). Olculdu: tek yarik cipasi nokta
    kaynakla duzlem dalga sinirina ancak d ~ 500 hucrede yaklasiyor
    (16.5 s/yapilandirma), baslangic kosuluyla ayni pencere 0.3-2 s.
    Kilit: tests/em/test_uzak_alan.py.
    """
    dx_kapisi(dx)
    if steps < 0:
        raise InvalidInputError(f"steps negatif olamaz, verilen: {steps}")
    if (source is None) != (source_index is None):
        raise InvalidInputError(
            "source ve source_index birlikte verilir ya da birlikte None olur"
        )
    cells = medium.cells
    for p in probes:
        if not 0 <= p < cells:
            raise InvalidInputError(f"prob indisi izgara disinda: {p} (cells={cells})")
    if source_index is not None and not 0 < source_index < cells - 1:
        raise InvalidInputError(
            f"kaynak indisi PEC duvarlarin ARASINDA olmali: {source_index}"
        )
    coeffs = yee_coefficients(medium, courant=courant, boundary=boundary)
    st = FieldState.zeros_1d(cells, dtype=dtype) if state is None else state
    if st.dim != 1 or st.cells != cells:
        raise InvalidInputError(
            f"baslangic durumu 1B ve {cells} hucre olmali, verilen: "
            f"dim={st.dim}, cells={st.cells}"
        )
    pidx = np.asarray(probes, dtype=int)
    kayit = np.zeros((len(probes), steps), dtype=st.e.dtype)
    for n in range(steps):
        # KAYIT ONCE: e^n prob'a yazilir, sonra e^{n+1} uretilir. Kaydin
        # adimla hizasi bir SECIMDIR ve burada beyan edilir -- kayit sonra
        # yapilsaydi butun zaman serisi bir adim kayardi ve faz okumalari
        # (Task 3) sessizce 2*pi*S/lambda kadar donerdi.
        kayit[:, n] = st.e[pidx]
        st = _adim(st, coeffs)
        if source is not None:
            # `_adim` her adimda YENI bir `e` dizisi ureti ve o diziyi baska
            # kimse tutmuyor; yerinde degistirmek bu yuzden guvenlidir ve
            # adim basina bir tam dizi kopyasini ortadan kaldirir. Cagirandan
            # gelen `state`in dizisine HICBIR ZAMAN dokunulmaz (ilk yazma
            # `_adim` sonrasidir).
            st.e[source_index] += coeffs.cb[source_index] * source(n * courant)
    return FieldRecord(
        probes=kayit,
        probe_indices=tuple(int(p) for p in probes),
        window=Window(
            cells=cells,
            dx=dx,
            courant=courant,
            steps=steps,
            wavelength=wavelength,
            npml=int(getattr(boundary, "thickness", 0)),
            sgf=sgf,
            window_fn=window_fn,
        ),
        final=st,
    )


# ===========================================================================
# 2B (TM_z) -- AYNI SEMA, IKINCI BOYUT
# ===========================================================================


@dataclass(frozen=True, eq=False)
class YeeCoefficients2D:
    """2B (TM_z) guncelleme katsayilari.

        Hx: hx <- dax * hx - dbx * (e[:, 1:] - e[:, :-1])
        Hy: hy <- day * hy + dby * (e[1:, :] - e[:-1, :])
        Ez: e  <- ca * e + cb * ( (hy[1:,:] - hy[:-1,:])
                                  - (hx[:,1:] - hx[:,:-1]) )

    Hx'teki EKSI kozmetik degil: TM_z'de dHx/dt = -dEz/dy, dHy/dt = +dEz/dx.
    1B'de yalniz Hy vardi ve orada isaret artiydi -- yani 1B guncellemesi bu
    semanin y'den BAGIMSIZ KESITIDIR. Bu bir IDDIADIR ve bit duzeyinde
    olculuyor (bkz. tests/em/test_bessel.py).

    `pec_mask` bir IC iletken maskesidir ve cekirdek onun NE OLDUGUNU bilmez:
    dairesel bosluk mu, kare kutu mu, hicbiri mi -- gordugu tek sey bool bir
    dizidir. `sigma` ile ayni gerekce.
    """

    ca: np.ndarray
    cb: np.ndarray
    dax: np.ndarray
    dbx: np.ndarray
    day: np.ndarray
    dby: np.ndarray
    pec_mask: np.ndarray
    dim: int
    courant: float


def yee_coefficients_2d(medium: Medium, *, courant: float, boundary) -> YeeCoefficients2D:
    """2B katsayilar. COURANT DENETIMI BURADADIR ve SINIRI 2B'NINKIDIR.

    `courant_limit(2)` = 1/sqrt(2), 1B'nin 1'i DEGIL. Devralma tehlikesi
    gercektir ve olculdu: 1B kapisi S = 0.9'u GECIRIR, 2B'de o sema PATLAR.
    """
    shape = medium.shape
    if len(shape) != 2:
        raise InvalidInputError(f"2B cozucu 2B ortam ister, verilen sekil: {shape}")
    nx, ny = shape
    if nx < 3 or ny < 3:
        raise InvalidInputError(
            f"2B izgara her yonde en az 3 hucre olmali, verilen: {nx} x {ny}"
        )
    limit = courant_limit(2)
    if not courant > 0.0:
        raise InvalidInputError(f"S > 0 olmali, verilen: {courant}")
    if courant > limit:
        raise InvalidInputError(
            f"S = {courant} kararlilik sinirini asiyor (2B'de S_max = {limit}, "
            "yani 1/sqrt(2)). Bu sinir 1B'ninkinden FARKLIDIR ve 1B kapisi "
            "DEVRALINAMAZ: S = 0.9 1B'de gecerdi, 2B'de sema patlar. Ayrinti: "
            "grid.courant_limit."
        )
    _etkin_courant_kapisi(medium, courant, limit, "2B")
    sigma = np.asarray(boundary.sigma_2d(shape), dtype=float)
    if sigma.shape != (nx, ny):
        raise InvalidInputError(
            f"sinir sigma dizisi {(nx, ny)} olmali, verilen: {sigma.shape}"
        )
    mask = np.asarray(boundary.pec_mask_2d(shape))
    if mask.shape != (nx, ny) or mask.dtype != bool:
        raise InvalidInputError(
            f"pec_mask_2d {(nx, ny)} sekilli bool dizi olmali, verilen: "
            f"{mask.shape} / {mask.dtype}"
        )
    eps = medium.eps_r
    mu_x = medium.mu_yari_2d(1)  # Hx(i, j+1/2)
    mu_y = medium.mu_yari_2d(0)  # Hy(i+1/2, j)

    d_e = sigma * courant / (2.0 * eps)
    ca = (1.0 - d_e) / (1.0 + d_e)
    cb = (courant / eps) / (1.0 + d_e)

    d_hx = 0.5 * (d_e[:, :-1] + d_e[:, 1:])
    d_hy = 0.5 * (d_e[:-1, :] + d_e[1:, :])
    katsayi = dict(
        ca=ca,
        cb=cb,
        dax=(1.0 - d_hx) / (1.0 + d_hx),
        dbx=(courant / mu_x) / (1.0 + d_hx),
        day=(1.0 - d_hy) / (1.0 + d_hy),
        dby=(courant / mu_y) / (1.0 + d_hy),
    )
    _sonlu_katsayi_kapisi("2B", **katsayi)
    return YeeCoefficients2D(**katsayi, pec_mask=mask, dim=2, courant=courant)


def _adim_2d(state: FieldState2D, co: YeeCoefficients2D) -> FieldState2D:
    """TEK 2B GUNCELLEME YOLU -- `yee_step_2d` de `run_2d` de buradan gecer.

    1B'deki `_adim` ile ayni gerekce ve ayni sira: once H (yarim adim ileri),
    sonra E (tam adim). Dort kenardaki PEC duvar ve IC ILETKEN MASKESI E'nin
    ustune en son uygulanir -- maske, "iletkende alan yok"un en dogrudan
    halidir ve merdivenlenme tam olarak burada olur.
    """
    e, hx, hy = state.e, state.hx, state.hy
    hx = co.dax * hx - co.dbx * (e[:, 1:] - e[:, :-1])
    hy = co.day * hy + co.dby * (e[1:, :] - e[:-1, :])
    e_yeni = co.ca * e
    e_yeni[1:-1, 1:-1] += co.cb[1:-1, 1:-1] * (
        (hy[1:, 1:-1] - hy[:-1, 1:-1]) - (hx[1:-1, 1:] - hx[1:-1, :-1])
    )
    # Izgaranin DORT kenari PEC duvardir -- 1B'deki iki uc ile ayni kural.
    e_yeni[0, :] = 0.0
    e_yeni[-1, :] = 0.0
    e_yeni[:, 0] = 0.0
    e_yeni[:, -1] = 0.0
    e_yeni[co.pec_mask] = 0.0
    return FieldState2D(e_yeni, hx, hy)


def yee_step_2d(
    state: FieldState2D, medium: Medium, *, courant: float, boundary
) -> FieldState2D:
    """Bir 2B Yee adimi. Bu fonksiyon da GEOMETRIYI BILMEZ.

    Ortamdan gordugu adlar `shape`, `eps_r`, `mu_yari_2d`, `mu_r` (yalniz etkin
    Courant kapisi, 01.10.2026); sinirdan gordugu
    adlar `sigma_2d`, `pec_mask_2d`. Dairenin yaricapi, kaynagin yeri,
    dedektorlerin sayisi -- hicbiri buraya girmez. `yee_step_1d` ile ayni
    gerekce; ve `run_2d` ile ayni ic fonksiyon (`_adim_2d`).
    """
    if not isinstance(state, FieldState2D):
        raise InvalidInputError(
            f"yee_step_2d FieldState2D alir, verilen: {type(state).__name__}"
        )
    if state.shape != medium.shape:
        raise InvalidInputError(
            f"durum ve ortam ayni izgarada olmali: {state.shape} != {medium.shape}"
        )
    return _adim_2d(
        state, yee_coefficients_2d(medium, courant=courant, boundary=boundary)
    )


def run_2d(
    medium: Medium,
    *,
    courant: float,
    boundary,
    steps: int,
    source: Callable[[float], float] | None,
    source_index: tuple[int, int] | None,
    probes: tuple[tuple[int, int], ...],
    wavelength: float,
    sgf: float,
    window_fn: str = "dikdortgen",
    dx: float = 1.0,
    dtype: type = float,
    state: FieldState2D | None = None,
) -> FieldRecord:
    """Cok adimli 2B kosum -- ve PENCERESINI dondurmek ZORUNDA (1B ile ayni).

    `probes` (i, j) IKILILERININ dizisidir; `probe_indices` onlari aynen
    tasir, dolayisiyla DEDEKTOR SAYISI kayittan SAYILABILIR. Bu bir kolaylik
    degil: dedektor sayisi bu modulde bir pencere dugmesidir ve
    `mode_spectrum` onu kendi beyanina koyar.

    `source` yumusak NOKTA kaynagidir (1B ile ayni satir):
    `e[i, j] += cb[i, j] * source(t)`, `t = n * S`.

    `dx != 1` BURADA DA REDDEDILIYOR ve gerekce `run_1d`dekinin AYNISIDIR
    (tek kaynak: `grid.dx_kapisi`). Kusurun 2B'deki olcusu Task 6'da
    beyan edilmisti: dairesel bosluk merdiveninde (R = 5/10, step_factor
    120) tepe sayisi dx = 1/0.5/2 icin 20-13 / 4-4 / 31-43 oynarken pencere
    ucunde de lambda'yi 7.257613 HUCRE diye basiyordu. O yuzden `dx`
    `EMScenario` icinde de bir alan DEGILDIR -- ama senaryodan cikarmak
    MOTORU kapatmiyordu; bu kapi onu kapatiyor.

    TE_z AYRI BIR YOL DEGILDIR (G1-p, 10.10.2026): bu cekirdek DUAL ortamla
    (eps_r = 1, mu_r = eps, sogurucuda sigma/eps) kosulunca dual izgarada bir
    2B TE_z Yee'dir -- kayit Hz, `final.hx`/`final.hy` = -Ex/-Ey. Dort kenar
    ve `pec_mask` o zaman PMC'dir (TE_z'de PEC cisim bu yolla kurulamaz).
    Sevk edilen `PML` sigma/eps olcegini VERMEZ ve dort kenara koyar: egik p
    ile BIT-YAKIN kimlik icin sinir nesnesi kullanicidadir (testte yerel).
    Olceksiz sigma da dualde eslesmis bir sogurucudur (H kaybi sigma'/eps'
    ile kurulur); olcek fizik icin degil kimlik icin gerekir.
    Olcum ve kosullar: `oblique` modul dokstringi; kilit
    `tests/em/test_g1p_egik_2b_te.py`.
    """
    dx_kapisi(dx)
    if steps < 0:
        raise InvalidInputError(f"steps negatif olamaz, verilen: {steps}")
    if (source is None) != (source_index is None):
        raise InvalidInputError(
            "source ve source_index birlikte verilir ya da birlikte None olur"
        )
    shape = medium.shape
    if len(shape) != 2:
        raise InvalidInputError(f"2B cozucu 2B ortam ister, verilen: {shape}")
    nx, ny = shape
    for p in probes:
        if len(tuple(p)) != 2:
            raise InvalidInputError(f"prob (i, j) ikilisi olmali, verilen: {p}")
        if not (0 <= p[0] < nx and 0 <= p[1] < ny):
            raise InvalidInputError(
                f"prob indisi izgara disinda: {tuple(p)} (sekil {shape})"
            )
    if source_index is not None:
        si, sj = source_index
        if not (0 < si < nx - 1 and 0 < sj < ny - 1):
            raise InvalidInputError(
                f"kaynak indisi PEC duvarlarin ARASINDA olmali: {source_index}"
            )
    co = yee_coefficients_2d(medium, courant=courant, boundary=boundary)
    st = FieldState2D.zeros_2d(nx, ny, dtype=dtype) if state is None else state
    if not isinstance(st, FieldState2D) or st.shape != shape:
        raise InvalidInputError(
            f"baslangic durumu 2B ve {shape} sekilli olmali, verilen: "
            f"{type(st).__name__} / {getattr(st, 'shape', None)}"
        )
    if source_index is not None and bool(co.pec_mask[source_index]):
        raise InvalidInputError(
            f"kaynak ILETKENIN ICINDE: {source_index}. Maske orada Ez'yi her "
            "adimda sifirlar, yani kaynak sessizce HIC calismaz -- ve kayit "
            "bos donerdi."
        )
    pi = np.asarray([p[0] for p in probes], dtype=int)
    pj = np.asarray([p[1] for p in probes], dtype=int)
    kayit = np.zeros((len(probes), steps), dtype=st.e.dtype)
    for n in range(steps):
        # KAYIT ONCE -- `run_1d` ile ayni hiza secimi ve ayni gerekce.
        kayit[:, n] = st.e[pi, pj]
        st = _adim_2d(st, co)
        if source is not None:
            st.e[source_index] += co.cb[source_index] * source(n * courant)
    return FieldRecord(
        probes=kayit,
        probe_indices=tuple((int(p[0]), int(p[1])) for p in probes),
        window=Window(
            cells=(nx, ny),
            dx=dx,
            courant=courant,
            steps=steps,
            wavelength=wavelength,
            npml=int(getattr(boundary, "thickness", 0)),
            sgf=sgf,
            window_fn=window_fn,
        ),
        final=st,
    )
