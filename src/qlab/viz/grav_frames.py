"""Yercekimi cizim yuku. FIZIK URETMEZ; okur, cevirir ve toplar.

YORUNGE IZI ODE'YE BAGLIDIR VE IHTIYAC OLCULDU -- ama gerekce ZAMAN DEGIL,
FONKSIYONUN YONUDUR. Kareleme `phi(u)` verir, yani TERS FONKSIYONU: esit
arali `phi` istendiginde her nokta icin ayri bir kok bulma isi cikar. ODE'nin
`dense_output`u bunu bedavaya verir.

MALIYET UC KEZ OLCULDU VE UCU DE FARKLI CIKTI -- cunku "kat" sayisi tamamen
karelemenin NASIL yazildigina bagli (GR, M = 1, p = 3e5, e = 0.3, 2000 nokta,
bu makine):

    ODE + dense_output                                    2.842 ms
    VEKTORLESTIRILMIS birikimli yamuk (psi'de esit arali)  0.050 ms
    NOKTA BASINA `quad` (u'da verilen noktalarda)         28.830 ms
    `quad` + `brentq` ILE TERS CEVIRME (ESIT ARALI phi)   72.325 ms

Ilk iki satir ODE'den UCUZ ya da ona yakin gorunur ama IKISI DE ISTENEN SEYI
URETMEZ: ciktilari `phi`de esit arali degildir. APPLES-TO-APPLES olan tek
satir sonuncusudur ve orada ODE 25 KAT ucuzdur. Plan bu orani 14.4, Task 4 ise
4.8 diye olcmustu; ucu de dogru olcumdur ve UCU DE FARKLI SEYI olcer -- bu
yuzden dordu de burada yazili. Tek bir "kat" sayisi yazmak yaniltici olurdu.

`viz` katmani `observable`i ve `engine`i import EDEBILIR (katman sozlesmesi:
run -> viz -> observable -> engine); bu dosya cipa sayilarini `grav_obs`tan
OKUR, hicbirini yeniden hesaplamaz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.grav.factory import FloorLine, GravSetup
from qlab.engine.grav.geodesic import (
    perihelion_coefficient,
    perihelion_coefficient_ode,
)
from qlab.errors import InvalidInputError
from qlab.observable.grav_obs import (
    NoiseFloor,
    apsidal_exponent,
    circular_frequency_defect,
    critical_impact_parameter,
    deflection_ratio,
    eight_invariant,
    isco_radius,
    lagrange_mu_c,
    monodromy_angles,
    newton_noise_floor,
    photon_sphere,
    second_order_coefficient,
    third_body_influence,
)


@dataclass(frozen=True, eq=False)
class GravPayload:
    """Bir yercekimi kosumunun cizilebilir ve yazilabilir hali.

    `phi`/`u`/`r` ODE yorunge motorundan gelir; `x`/`y` onlarin kartezyen
    izdusumudur ve BU DOSYADA uretilir -- `Trajectory` bir egri saklar, bir
    resim degil.

    `floors` panelin ZORUNLU gurultu tabani satirlarinin kaynagidir ve
    SIRASI anlamlidir: ilk oge senaryonun kendi ayarindaki KARELEME
    borusudur, cunku `factory.provenance_for` panelin canli satirinda
    `floors[0].signal`i "perihel katsayisi" diye basar.

    `eq=False`: alanlar ndarray tasir; uretilen `__eq__` dizi dondurur.
    """

    setup: GravSetup
    phi: np.ndarray
    u: np.ndarray
    r: np.ndarray
    x: np.ndarray
    y: np.ndarray
    coefficient: float
    second_order: float
    deflection: float
    eight: float
    angles: tuple[float, float]
    closed_forms: dict[str, float]
    floors: tuple[FloorLine, ...]


def _satir(label: str, nf: NoiseFloor, *, anchor: bool = True) -> FloorLine:
    """`NoiseFloor` -> `FloorLine`. Katman sinirinin ta kendisi.

    `engine/grav/factory.py` `NoiseFloor`u goremez (`observable` ondan
    yukarida); ceviri bu dosyada, yani ikisini de gorebilen tek katmanda
    yapilir. Hicbir alan yeniden hesaplanmaz, hicbiri yuvarlanmaz.
    """
    return FloorLine(
        label=label,
        value=nf.value,
        signal=nf.signal,
        ratio=nf.ratio,
        pipeline=nf.pipeline,
        verdict=nf.verdict,
        anchor=anchor,
    )


def build_grav_payload(setup: GravSetup) -> GravPayload:
    """Cipa sayilarini ve gurultu tabanlarini OKUR, hicbirini uretmez.

    HER TABAN KENDI BORUSUNDAN OKUNUR -- KARISIK BORU YASAK (spec 4.3).
    Sevk edilen bes satirin besi de `newton_noise_floor`un CAGRILAN
    FONKSIYONU kendi `NEWTON_PROBES`uyla kosturmasiyla gelir; bu dosya
    hicbirine bir `theory` ya da `m2` enjekte etmez, edemez de (cagirana
    kapali).

    BESINCI SATIR BILEREK BURADADIR VE BIR CIPA DEGILDIR: spec 4.3'un kendi
    N-cisim gozlenebiliri (`third_body_influence`) `m2 = 0`da BIT BIT sifir
    veriyor, yani 0/1 canli basamakla DEJENERE. Panelden CIKARILMADI, cunku
    sevk edilen ikamenin (`circular_frequency_defect`) gerekcesi tam olarak
    bu olcumdur; bir dejenerelik saklandiginda ikamenin gerekcesi de
    kaybolur. `anchor=False` ile isaretlenir ve `converged` hukmune girmez.

    KAPALI-FORM TUREVLER DE OKUNUR ama panel onlari FIZIK CIPASI diye
    yazmaz: sinifin kendisi `factory._CIPA_SINIFI_BEYANI`nda yazili.
    """
    s = setup.scenario
    t = setup.trajectory
    theory = setup.theory

    coefficient = perihelion_coefficient(
        theory=theory, M=s.mass, e=s.eccentricity,
        richardson=s.richardson, gauge=s.gauge,
    )
    second_order = second_order_coefficient(
        theory=theory, M=s.mass, p=s.second_order_p, e=s.eccentricity,
    )
    deflection = deflection_ratio(
        theory=theory, M=s.mass, richardson=s.deflection_richardson,
        gauge=s.gauge,
    )
    eight = float(eight_invariant(setup.figure_eight))
    angles = monodromy_angles(
        setup.figure_eight, rtol=s.nbody_rtol, atol=s.nbody_atol,
        method=s.nbody_method,
    )

    # UC KAPALI-FORM TUREV KURAMA BAGLIDIR VE HER KURAM NOKTASINDA YOKTUR.
    # `beta > gamma`da 1PN'de kesilmis A(u) ufuk YAPMAZ: foton kuresi yoktur
    # ve donen ISCO koku fiziksel degildir; `grav_obs` orada INANILIR GORUNEN
    # BIR SAYI DONDURMEK YERINE REDDEDER. Burada o red YUTULMAZ, ANAHTAR HIC
    # YAZILMAZ -- `_run_spin`in isi sigasi kuralinin ve Modul 1B'nin
    # kararlilik alani dersinin aynisi: "olculmedi" ile "sifir cikti" ayni
    # degerle gosterilemez. Panel eksikligi kendi cumlesiyle SOYLER.
    closed_forms: dict[str, float] = {}
    for ad, fn in (
        ("isco_r_over_rs", lambda: isco_radius(theory=theory, M=s.mass) / 2.0),
        ("foton_r_over_rs", lambda: photon_sphere(theory=theory, M=s.mass) / 2.0),
        ("b_krit_over_M",
         lambda: critical_impact_parameter(theory=theory, M=s.mass)),
    ):
        try:
            closed_forms[ad] = float(fn())
        except InvalidInputError:
            continue
    # Bu ikisi KURAMDAN BAGIMSIZDIR (biri Newtonyen kisitli uc-cisim
    # probleminin, oteki merkezi kuvvet ailesinin sayisidir) ve her zaman
    # okunur; ayni sozlukte durmalari onlari ayni sinifa sokmaz -- sinif
    # `factory._CIPA_SINIFI_BEYANI`nda yazili.
    closed_forms["mu_c"] = float(lagrange_mu_c())
    closed_forms["apsidal_us"] = float(apsidal_exponent())

    floors = (
        _satir(
            "perihel katsayisi (kareleme = CIPA YOLU)",
            newton_noise_floor(
                perihelion_coefficient, M=s.mass, e=s.eccentricity,
                richardson=s.richardson, gauge=s.gauge,
            ),
        ),
        _satir(
            "perihel katsayisi (ODE = YORUNGE MOTORU, cipa yolu DEGIL)",
            newton_noise_floor(
                perihelion_coefficient_ode, M=s.mass, e=s.eccentricity,
            ),
        ),
        _satir(
            "bukulme orani",
            newton_noise_floor(
                deflection_ratio, M=s.mass,
                richardson=s.deflection_richardson, gauge=s.gauge,
            ),
        ),
        _satir(
            "N-cisim dairesel frekans kusuru",
            newton_noise_floor(
                circular_frequency_defect, rtol=s.nbody_rtol,
                atol=s.nbody_atol, method=s.nbody_method,
            ),
        ),
        _satir(
            "N-cisim dogrusal harita (spec 4.3'un kendi gozlenebiliri -- "
            "IKAME EDILDI, CIPA DEGIL)",
            newton_noise_floor(
                third_body_influence, rtol=s.nbody_rtol, atol=s.nbody_atol,
                method=s.nbody_method,
            ),
            anchor=False,
        ),
    )

    return GravPayload(
        setup=setup,
        phi=t.phi,
        u=t.u,
        r=t.r,
        x=t.r * np.cos(t.phi),
        y=t.r * np.sin(t.phi),
        coefficient=float(coefficient),
        second_order=float(second_order),
        deflection=float(deflection),
        eight=eight,
        angles=angles,
        closed_forms=closed_forms,
        floors=floors,
    )
