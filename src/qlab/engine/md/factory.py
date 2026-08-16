"""MDScenario -> (durum, kuvvet alanı, termostat) ve Provenance."""

from __future__ import annotations

import numpy as np

from qlab.engine.md.forcefields import Harmonic, LennardJones, Morse
from qlab.engine.md.state import MDState
from qlab.engine.md.thermostats import Langevin, NoseHoover
from qlab.errors import ScenarioError
from qlab.provenance import Provenance
from qlab.scenario.schema import MDScenario
from qlab.units import AMU_TO_ELECTRON_MASS, amu_to_electron_mass

_YAKLASIMLAR = (
    "klasik dinamik: çekirdekler nokta parçacık, elektron yok",
    "sıfır nokta enerjisi ve tünelleme yok (kuantum titreşim klasikleştirildi)",
    "kuvvet alanı analitik bir modeldir, hesaplanmış bir yüzey değildir",
)
_IHMALLER = (
    "kuantum istatistiği (Maxwell-Boltzmann klasik limittir)",
    "elektronik uyarılma ve Born-Oppenheimer yüzeyler arası geçiş",
)


def _fcc_kafes(n_particles: int, density_reduced: float, sigma: float):
    """FCC kafes ve kutu kenarı. L = sigma * (N/rho*)^(1/3).

    ÖLÇÜLDÜ: N=108, rho*=0.8442, sigma=6.4345174543 -> L = 32.422173 bohr.
    Kutu kenarı YOĞUNLUKTAN TÜRETİLİR, YAML'dan okunmaz -- tek doğruluk
    kaynağı olsun diye. (Eski YAML 33.0656 yazıyordu ve o değer beyan edilen
    yoğunlukla çelişiyordu: gerçek rho* 0.795871, -%5.72.)
    """
    hucre = round((n_particles / 4) ** (1 / 3))
    if 4 * hucre**3 != n_particles:
        raise ScenarioError(
            f"n_particles bir FCC sayısı olmalı (4*n^3): {n_particles}. "
            f"En yakınları {4 * hucre**3} ve {4 * (hucre + 1) ** 3}."
        )
    kenar = (n_particles / density_reduced) ** (1 / 3) * sigma
    a = kenar / hucre
    taban = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    konumlar = np.concatenate(
        [
            taban + np.array([i, j, k]) * a
            for i in range(hucre)
            for j in range(hucre)
            for k in range(hucre)
        ]
    )
    return konumlar, float(kenar)


def build(scenario: MDScenario):
    """Senaryodan motoru kurar. Kütle dönüşümü BURADA, sınırda yapılır."""
    tur = scenario.force_field["kind"]
    p = dict(scenario.force_field.get("params", {}))
    ilk = scenario.initial

    # İKİ BAŞLANGIÇ BİÇİMİ. Bu dal eskiden YOKTU ve `argon_nvt.yaml` ham
    # `KeyError: 'positions_bohr'` veriyordu -- Global Constraint "sessiz
    # başarısızlık yok / ham hata tüketiciye ulaşmaz" ihlali.
    if "positions_bohr" in ilk:
        konumlar = np.asarray(ilk["positions_bohr"], dtype=float)
    elif ilk.get("lattice") == "fcc":
        konumlar, kenar = _fcc_kafes(
            int(ilk["n_particles"]), float(ilk["density_reduced"]), float(p["sigma"])
        )
        p["box_length"] = kenar  # yoğunluktan türetilen kenar YAML'ı EZER
    else:
        raise ScenarioError(
            "initial bloğu ya `positions_bohr` ya da "
            "`lattice: fcc` + `n_particles` + `density_reduced` içermeli; "
            f"verilen anahtarlar: {sorted(ilk)}"
        )

    ham_kutle = np.asarray(ilk["masses_amu"], dtype=float)
    if ham_kutle.size == 1 and konumlar.shape[0] > 1:
        ham_kutle = np.full(konumlar.shape[0], ham_kutle.item())
    kutleler = amu_to_electron_mass(ham_kutle)

    hizlar = np.asarray(
        ilk.get("velocities", np.zeros_like(konumlar)), dtype=float
    )
    durum = MDState(positions=konumlar, velocities=hizlar, masses=kutleler)

    if tur == "harmonic":
        alan = Harmonic(k=p["k"], center=p.get("center", np.zeros_like(konumlar)))
        # velocity-Verlet omega*dt = 2.0'da patlar (ölçüldü: 2.001 -> 4.3e+54).
        # Harmonikte omega analitiktir, o yüzden burada denetlenebilir.
        omega = float(np.sqrt(alan.k / kutleler.min()))
        if omega * scenario.dt >= 2.0:
            raise ScenarioError(
                f"dt kararlılık sınırını aşıyor: omega*dt = "
                f"{omega * scenario.dt:.3f} >= 2.0. velocity-Verlet bu sınırda "
                f"patlar. dt < {2.0 / omega:.4g} seçin."
            )
    elif tur == "morse":
        alan = Morse(**p)
    elif tur == "lennard_jones":
        alan = LennardJones(**p)
    else:
        raise ScenarioError(f"bilinmeyen kuvvet alanı: {tur!r}")

    termostat = None
    if scenario.thermostat is not None:
        t_tur = scenario.thermostat["kind"]
        t_p = dict(scenario.thermostat.get("params", {}))
        if t_tur == "langevin":
            termostat = Langevin(**t_p)
        elif t_tur == "nose_hoover":
            termostat = NoseHoover(**t_p)
        else:
            raise ScenarioError(f"bilinmeyen termostat: {t_tur!r}")

    return durum, alan, termostat


#: NVE koşusunda kabul edilen bağıl enerji sürüklenmesi. LİTERAL ve ölçülmüş:
#: dt = 40 a.u.'da 1.739e-03, dt = 70'te 1.342e-02, dt = 90'da 2.397e-01.
#: Eskiden `converged=True` SABİTTİ ve panel %24 kaymış enerjinin yanında bile
#: ✓ basıyordu -- deponun `tdse/factory.py` ve `hf/factory.py` desenine aykırı.
_MD_DRIFT_TOLERANCE: float = 5.0e-03


def provenance_for(
    scenario: MDScenario, *, energy_drift: float, n_constraints: int = 3
) -> Provenance:
    kutleler = amu_to_electron_mass(
        np.asarray(scenario.initial["masses_amu"], dtype=float)
    )
    termostat = scenario.thermostat
    if termostat is None:
        yontem = f"klasik MD / velocity-Verlet / {scenario.force_field['kind']}"
        beklenen = (
            "velocity-Verlet simplektiktir: enerji SÜRÜKLENMEZ, SALINIR. "
            "Salınım genliği (omega*dt)^2/4 mertebesindedir ve bir hata değildir."
        )
        yakinsadi = bool(np.isfinite(energy_drift)) and (
            energy_drift <= _MD_DRIFT_TOLERANCE
        )
    else:
        parametreler = ", ".join(
            f"{ad}={deger}" for ad, deger in sorted(termostat["params"].items())
        )
        yontem = (
            f"klasik MD / velocity-Verlet + {termostat['kind']}({parametreler})"
            f" / {scenario.force_field['kind']}"
        )
        # NVT'DE ENERJİ KORUNMAZ -- bu bir kusur değil, tanımdır. Eskiden
        # termostatlı koşuda da "enerji SÜRÜKLENMEZ" yazıyordu ve ölçülen
        # sürüklenme %22.4 iken panel ✓ basıyordu.
        beklenen = (
            "NVT: toplam enerji KORUNMAZ, sistem banyoyla enerji alışverişi "
            "yapar. Ölçülen büyüklük kinetik sıcaklıktır, enerji sürüklenmesi "
            "değil; aşağıdaki energy_drift bir tanı değeridir, bir hata payı DEĞİL."
        )
        yakinsadi = True

    return Provenance(
        method=yontem,
        approximations=_YAKLASIMLAR,
        neglected=_IHMALLER,
        expected_error=beklenen,
        converged=yakinsadi,
        diagnostics={
            "dt": scenario.dt,
            "n_steps": scenario.n_steps,
            "energy_drift": energy_drift,
            "n_constraints": n_constraints,
            "masses_electron_mass": np.round(kutleler, 4).tolist(),
            "amu_to_electron_mass": AMU_TO_ELECTRON_MASS,
        },
    )
