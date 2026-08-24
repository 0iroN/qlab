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

from qlab.engine.em.grid import FieldRecord, FieldState, Window, courant_limit
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
    return YeeCoefficients(ca=ca, cb=cb, da=da, db=db, dim=1, courant=courant)


def _adim(state: FieldState, coeffs: YeeCoefficients) -> FieldState:
    """TEK GUNCELLEME YOLU. `yee_step_1d` de `run_1d` de buradan gecer.

    Sira: once H (yarim adim ileri), sonra E (tam adim). YARIM HUCRE
    KAYDIRMA `h[i] = H(i+1/2)` uzerinden gelir: E'nin curl'u `h[i] - h[i-1]`,
    H'ninki `e[i+1] - e[i]`. Ikisi de MERKEZI farktir ve semanin 2. mertebesi
    tam olarak buradan gelir. Kaydirma dusurulunce olan sey MERTEBE DUSMESI
    DEGIL, KARARSIZLIKTIR: iki curl de ayni yonlu farki alirsa sema her
    Courant sayisinda patliyor (N = 201, 600 adim: S = 0.5 -> 3.9e+242,
    S = 0.25 -> 3.0e+125, S = 0.1 -> 4.8e+48).
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
            "2B Task 4'te gelir; 3B ise KAPSAM DISIDIR ve sebebi olculdu: "
            "Airy sabiti j_1,1/pi uzerinden 2B'de capalaniyor, 3B karsiligi "
            "ayni adim sayisinda ciplak cekirdekle bile cok daha pahali."
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
    `t = n * S * dx`. TF/SF tek yonlu kaynak Task 3'te gelir; yumusak kaynak
    iki yonlu isir ve yansiyan alan IKI KOSUMUN FARKINDAN okunur.
    """
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
            st.e[source_index] += coeffs.cb[source_index] * source(
                n * courant * dx
            )
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
