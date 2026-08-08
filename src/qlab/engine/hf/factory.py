"""HFScanScenario'dan bir Hartree-Fock parametre taramasi uretir.

engine katmani scenario katmanini import edebilir (katman sirasi:
app > viz > observable > engine > scenario); tersi yasaktir.

Tasarim karari -- bir nokta yakinsamadiginda tarama ne yapar:
SCFConvergenceError yalnizca O NOKTA icin yakalanir; o noktanin enerjisi NaN
olarak isaretlenir ve tarama diger degerlerle DEVAM eder. ScanResult.provenance
o zaman converged=False tasir ve hangi deger(ler)in basarisiz oldugu
diagnostics["failed_values"] icinde raporlanir. Boylece tek bir zor geometri
(ornegin ayrisma siniri yakini, bkz. test_physical.py buyuk-ayrim notu) butun
egriyi yok etmez, ama sonuc hicbir zaman "hepsi yakinsadi" gibi SAHTE bicimde
raporlanmaz -- cagiran taraf NaN'lari ve converged=False'u gormeden egriyi
guvenilir sayamaz (spec ilkesi: yakinsamamis bir sonuc asla normal sonuc gibi
donmez).

Buna karsilik SCFConvergenceError DISINDAKI hatalar (yanlis baz adi, cakisik
cekirdek, sablon hatasi, acik kabuk sistem) TEK BIR NOKTAYA ozgu degildir --
taramanin TUMUNU gecersiz kilan bir yapilandirma hatasidir ve genelde HER
noktada tekrar tekrar olusur. Bunlar yakalanip NaN'a cevrilmez; hemen yukselir
ve tarama bastan durur -- aksi halde yapilandirma hatasi, anlamsiz bir NaN
egrisi olarak sessizce raporlanirdi.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.hf.scf import provenance_for, solve
from qlab.engine.hf.stability import StabilityResult
from qlab.errors import ScenarioError, SCFConvergenceError
from qlab.provenance import Provenance, StabilityVerdict
from qlab.scenario.schema import HFScanScenario
from qlab.types import Molecule


@dataclass(frozen=True)
class ScanResult:
    scenario: HFScanScenario
    values: np.ndarray
    energies: np.ndarray
    # Nokta basina en dusuk orbital Hessian ozdegeri. Olculemeyen noktalarda
    # (SCF yakinsamadi, veya dondurulecek isgal-bos cift yok) NaN'dir --
    # 0.0 gibi bir sayi "sinirda kararli" diye okunurdu.
    lowest_eigenvalues: np.ndarray
    provenance: Provenance


def _molecule_for(scenario: HFScanScenario, value: float) -> Molecule:
    """Tarama degerini molekul sablonuna yerlestirir ve Molecule'e cevirir.

    Sablon {value} disinda bir yer tutucu iceriyorsa (kopyala-yapistir
    kalintisi, yanlis alan adi, ...) str.format() KeyError/IndexError
    firlatir -- bu ciplak haliyle sizdirilmaz, ScenarioError'a cevrilir.
    """
    try:
        spec = scenario.molecule_template.format(value=value)
    except (KeyError, IndexError, ValueError) as exc:
        raise ScenarioError(
            f"'molecule_template' {{value}} disinda gecersiz bir yer tutucu "
            f"iceriyor: {exc}"
        ) from exc
    return Molecule.from_string(spec, unit=scenario.unit)


def run_scan(scenario: HFScanScenario) -> ScanResult:
    """HFScanScenario'daki her deger icin RHF enerjisini hesaplar.

    Yakinsama davranisi icin modul dokstringine bakin: nokta-bazli SCF
    yakinsamama NaN olarak isaretlenip taramaya devam edilir, yapilandirma
    hatalari ise taramanin tumunu durdurur.
    """
    values = np.array(scenario.values, dtype=float)
    energies = np.full(values.shape, np.nan)
    lowest_eigenvalues = np.full(values.shape, np.nan)
    failed_values: list[float] = []
    unstable_values: list[float] = []
    measured: list[StabilityResult] = []

    for index, value in enumerate(scenario.values):
        molecule = _molecule_for(scenario, value)
        try:
            result = solve(molecule, scenario.basis)
        except SCFConvergenceError:
            failed_values.append(value)
            continue
        energies[index] = result.energy_total
        if result.stability is not None:
            measured.append(result.stability)
            if not result.stability.stable:
                unstable_values.append(value)
            if result.stability.lowest_eigenvalue is not None:
                lowest_eigenvalues[index] = result.stability.lowest_eigenvalue

    n_failed = len(failed_values)
    converged = n_failed == 0

    diagnostics: dict = {"n_points": len(scenario.values), "n_failed": n_failed}
    if failed_values:
        diagnostics["failed_values"] = failed_values

    # Kararlilik ozeti yalnizca EN AZ BIR nokta olculebildiyse raporlanir.
    # Hicbir nokta yakinsamadiysa "n_unstable=0" yazmak, hicbir sey
    # olculmemisken "hepsi kararli" gibi okunurdu.
    verdict = None
    if measured:
        diagnostics["n_unstable"] = len(unstable_values)
        if unstable_values:
            diagnostics["unstable_values"] = unstable_values
        verdict = _scan_verdict(measured, unstable_values)

    # Yaklasimlar/ihmaller/beklenen hata metni scf.provenance_for'dan gelir --
    # tek nokta ile tarama ayni RHF'i kosar, dolayisiyla ayni beyani basmak
    # ZORUNDADIR (bkz. scf.py, provenance_for).
    provenance = provenance_for(
        method=f"RHF/{scenario.basis.upper()} tarama ({scenario.parameter})",
        converged=converged,
        diagnostics=diagnostics,
        stability=verdict,
    )

    return ScanResult(
        scenario=scenario,
        values=values,
        energies=energies,
        lowest_eigenvalues=lowest_eigenvalues,
        provenance=provenance,
    )


def _scan_verdict(
    measured: list[StabilityResult], unstable_values: list[float]
) -> StabilityVerdict:
    """Taramanin tek satirlik kararlilik hukmu: EN KOTU nokta belirler.

    Tarama, tek bir nokta degil bir egridir; "kararli" demek ancak HICBIR
    noktada negatif ozdeger yoksa dogrudur. Raporlanan ozdeger de butun
    noktalarin en dusugudur -- ortalama veya son nokta, bir eyer noktasini
    gizlerdi. Hangi degerlerin kararsiz oldugu diagnostics["unstable_values"]
    icinde, nokta bazinda ise ScanResult.lowest_eigenvalues icindedir.
    """
    eigenvalues = [
        stability.lowest_eigenvalue
        for stability in measured
        if stability.lowest_eigenvalue is not None
    ]
    return StabilityVerdict(
        stable=not unstable_values,
        lowest_eigenvalue=min(eigenvalues) if eigenvalues else None,
        n_rotations=max(stability.n_rotations for stability in measured),
    )
