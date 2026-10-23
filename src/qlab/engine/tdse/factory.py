"""Senaryo tanımından çalışır motor nesneleri üretir.

engine katmanı scenario katmanını import edebilir (katman sırası:
app > viz > observable > engine > scenario); tersi yasaktır.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.initial import gaussian_packet
from qlab.engine.tdse.propagator import SplitOperatorPropagator, UniterlikOlcumu
from qlab.errors import InvalidInputError, ScenarioError
from qlab.provenance import Provenance
from qlab.scenario.schema import TDSEScenario

#: Maskesiz koşunun norm ölçütü |N - 1| ve maskeli koşunun maske öncesi
#: ölçütleri (B5-U) AYNI eşiği okur: maske ≡ 1'de bütçe sapması tam olarak
#: |N - 1|'dir. Ölçüldü (14.10.2026, maskeli 64000 adımlık tünelleme, paket
#: %99.99 soğuruluyor): adım oranı en çok 1.8e-15, bütçe 1.4e-11 — eşiğe
#: ~5e6 ve ~730 kat pay; maskesiz `tunneling.yaml` 8.4e-12.
_NORM_TOLERANCE = 1e-8

# Maskeli koşunun yakınsama tanısı, ÖLÇÜM VERİLMEDİYSE (B5). Maske normu
# kasten düşürür; |N - 1| üniterlik ölçütü olamaz. Ölçüm `evolve_olcumlu`dadır
# (maske öncesi üniter adım normu); `provenance_for`a verilmediyse panel ✓
# basamaz.
_MASKELI_YARGILANMADI = (
    "YARGILANMADI: soğurucu maske normu bilinçli olarak düşürür, |N - 1| "
    "üniterlik ölçütü olamaz — ve bu çağrıya maske öncesi üniter adım normu "
    "ölçümü (evolve_olcumlu) VERİLMEDİ; observed_norm_drop bilgilendiricidir, "
    "hüküm değildir"
)


def _maskeli_tani(olcum: UniterlikOlcumu, gecti: bool) -> str:
    """B5-U: maskeli koşunun tanı satırı — NEYİN yargılandığını adıyla söyler."""
    olculen = (
        f"{olcum.adim_sayisi} adımın her birinde soğurucu maskeden ÖNCE "
        f"‖Uψ‖²/‖ψ‖² ölçüldü: en büyük |oran − 1| = "
        f"{olcum.en_buyuk_adim_sapmasi:.1e}, norm bütçesi |N + soğurulan − N0|/N0 "
        f"en çok {olcum.en_buyuk_butce_sapmasi:.1e} (eşik {_NORM_TOLERANCE:.0e}, "
        f"maskesiz koşunun norm ölçütü)"
    )
    sogurma = (
        f"Maskenin soğurduğu norm ({olcum.sogurulan_norm:.3e}) AYRIDIR: sınır "
        f"koşuludur, fizik değil, ve bu hükme girmez"
    )
    if gecti:
        return (
            f"maske öncesi üniter adım normu yargılandı — {olculen}; ikisi de "
            f"eşiğin altında: üniter adım normu koruyor. {sogurma}"
        )
    return (
        f"maske öncesi üniter adım normu yargılandı ve EŞİĞİ AŞTI — {olculen}: "
        f"üniter adım normu KORUMUYOR (dt çok büyük ya da kod hatası; maskesiz "
        f"koşuda evolve() bu sapmada UnitarityViolationError fırlatırdı). {sogurma}"
    )


def _build_potential(scenario: TDSEScenario, grid: Grid1D) -> np.ndarray:
    kind = scenario.potential.kind
    params = scenario.potential.params
    try:
        if kind == "free":
            return potentials.free(grid)
        if kind == "barrier":
            return potentials.rectangular_barrier(grid, **params)
        if kind == "well":
            return potentials.square_well(grid, **params)
        if kind == "harmonic":
            return potentials.harmonic(grid, mass=scenario.mass, **params)
        if kind == "uniform_field":
            return potentials.uniform_field(grid, **params)
    except TypeError as exc:
        raise ScenarioError(f"{kind!r} potansiyeli için parametreler hatalı: {exc}") from exc
    raise ScenarioError(f"Bilinmeyen potansiyel türü: {kind!r}")


def build(
    scenario: TDSEScenario,
) -> tuple[Grid1D, np.ndarray, np.ndarray, SplitOperatorPropagator]:
    grid = Grid1D(n=scenario.grid.n, length=scenario.grid.length_bohr)
    potential = _build_potential(scenario, grid)
    psi0 = gaussian_packet(
        grid,
        x0=scenario.packet.x0_bohr,
        k0=scenario.packet.k0,
        sigma=scenario.packet.sigma_bohr,
    )

    mask = None
    if scenario.absorbing_width_bohr is not None:
        try:
            mask = potentials.absorbing_mask(grid, width=scenario.absorbing_width_bohr)
        except TypeError as exc:
            raise ScenarioError(
                f"absorbing_width_bohr sayısal olmalı, verilen: {scenario.absorbing_width_bohr!r}"
            ) from exc

    propagator = SplitOperatorPropagator(
        grid=grid,
        potential=potential,
        dt=scenario.dt,
        mass=scenario.mass,
        mask=mask,
    )
    return grid, potential, psi0, propagator


def provenance_for(
    scenario: TDSEScenario,
    norm_deviation: float,
    uniterlik: UniterlikOlcumu | None = None,
) -> Provenance:
    """Panel beyanı. `uniterlik` maskeli koşunun `evolve_olcumlu` ölçümüdür.

    Maskesiz koşuda verilmez (verilirse `InvalidInputError`; orada ölçüt
    `norm_deviation`dır). Maskeli koşuda verilmezse hüküm ✗ YARGILANMADI'dır.

    KARAR (grup 9, B5-U): maskeli koşuda eşiği aşan maske öncesi ölçüm İSTİSNA
    FIRLATMAZ, kareler incelenebilir kalır ve panel ✗ ile neyin aştığını
    söyleyen tanı satırı basar — halka indirgemesinin Δ ≈ 0 kapısı emsali
    (`tb/factory.py`, B7dog-2: parametreler değişmez, hüküm ✗ + tanı); maskesiz
    koşu ihlali `evolve`de `UnitarityViolationError` ile bildirir.
    """
    approximations = [
        "tek parçacık",
        "sabit dış potansiyel",
        "periyodik sınır koşulu (FFT)",
        "Strang bölmesi, O(dt^2)",
    ]
    masked = scenario.absorbing_width_bohr is not None
    if masked:
        approximations.append("kenarlarda soğurucu maske — norm bilinçli olarak azalır")
    elif uniterlik is not None:
        raise InvalidInputError(
            "maskesiz TDSE koşusuna maske öncesi üniterlik ölçümü verilmez: "
            "orada ölçüt |N - 1|'dir (norm_deviation) ve evolve_olcumlu None döndürür"
        )
    if uniterlik is not None and uniterlik.adim_sayisi != scenario.n_steps:
        raise InvalidInputError(
            f"üniterlik ölçümü {uniterlik.adim_sayisi} adımı kapsıyor, senaryo "
            f"{scenario.n_steps} adım; ölçüm bu koşuya ait değil"
        )

    # Soğurucu maske normu bilinçli olarak azaltır; |N - 1| maskeli koşuda
    # üniterlik hakkında hiçbir şey söylemez (ne "yakınsamadı" kanıtıdır ne
    # "yakınsadı"). Eskiden burada converged=True vardı ve panel %99.99 norm
    # kaybının yanına ✓ basıyordu (B5, 11.10.2026); sonra ölçülmeyen hüküm
    # ✗ + YARGILANMADI oldu (EM emsali). B5-U (14.10.2026): ölçülebilen şey
    # maskeden ÖNCEKİ üniter adımdır — `evolve_olcumlu` her adımda ‖Uψ‖²/‖ψ‖²
    # ve norm bütçesini (N + soğurulan) ölçer; ikisi de maskesiz yolun eşiğinin
    # altındaysa ✓, değilse ✗, ve tanı satırı neyin yargılandığını söyler.
    # Ölçüm verilmediyse (`uniterlik` None) hüküm hâlâ ✗ YARGILANMADI'dır
    # (= denenmedi; YARGILANAMAZ'dan bilerek ayrı kelime). Maskesiz koşu
    # değişmedi: eşik `norm_deviation`a tam olarak uygulanır.
    if not masked:
        converged = norm_deviation <= _NORM_TOLERANCE
        convergence_diagnosis = None
        diagnostics = {"norm_deviation": norm_deviation, "steps": scenario.n_steps}
    elif uniterlik is None:
        converged = False
        convergence_diagnosis = _MASKELI_YARGILANMADI
        diagnostics = {"observed_norm_drop": norm_deviation, "steps": scenario.n_steps}
    else:
        # NaN iki karşılaştırmayı da False yapar -> ✗ (bilerek `not >` değil).
        converged = bool(
            uniterlik.en_buyuk_adim_sapmasi <= _NORM_TOLERANCE
            and uniterlik.en_buyuk_butce_sapmasi <= _NORM_TOLERANCE
        )
        convergence_diagnosis = _maskeli_tani(uniterlik, converged)
        diagnostics = {
            "observed_norm_drop": norm_deviation,
            "premask_step_deviation": uniterlik.en_buyuk_adim_sapmasi,
            "premask_budget_deviation": uniterlik.en_buyuk_butce_sapmasi,
            "absorbed_norm": uniterlik.sogurulan_norm,
            "steps": scenario.n_steps,
        }

    return Provenance(
        method=f"TDSE split-operator (Strang), 1B, dt={scenario.dt}",
        approximations=tuple(approximations),
        neglected=("elektron-elektron etkileşimi", "spin", "göreli etkiler"),
        expected_error=f"O(dt^2) ~ {scenario.dt**2:.1e} hartree mertebesi",
        converged=converged,
        diagnostics=diagnostics,
        convergence_diagnosis=convergence_diagnosis,
    )
