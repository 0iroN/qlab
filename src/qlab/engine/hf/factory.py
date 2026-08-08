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

from qlab.engine.hf.scf import solve
from qlab.errors import ScenarioError, SCFConvergenceError
from qlab.provenance import Provenance
from qlab.scenario.schema import HFScanScenario
from qlab.types import Molecule


@dataclass(frozen=True)
class ScanResult:
    scenario: HFScanScenario
    values: np.ndarray
    energies: np.ndarray
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
    failed_values: list[float] = []

    for index, value in enumerate(scenario.values):
        molecule = _molecule_for(scenario, value)
        try:
            result = solve(molecule, scenario.basis)
        except SCFConvergenceError:
            failed_values.append(value)
            continue
        energies[index] = result.energy_total

    n_failed = len(failed_values)
    converged = n_failed == 0

    diagnostics: dict = {"n_points": len(scenario.values), "n_failed": n_failed}
    if failed_values:
        diagnostics["failed_values"] = failed_values

    provenance = Provenance(
        method=f"RHF/{scenario.basis.upper()} tarama ({scenario.parameter})",
        approximations=(
            "Born-Oppenheimer",
            "tek Slater determinanti",
            "sonlu baz seti",
            "kapali kabuk (cift isgal)",
        ),
        neglected=("elektron korelasyonu", "goreli etkiler", "cekirdek hareketi"),
        expected_error=(
            "korelasyon enerjisi eksik -- bag enerjilerinde tipik olarak birkac eV"
        ),
        converged=converged,
        diagnostics=diagnostics,
    )

    return ScanResult(
        scenario=scenario, values=values, energies=energies, provenance=provenance
    )
