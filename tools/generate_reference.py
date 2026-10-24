"""PySCF'ten referans degerleri uretir. Excalibur'da (Linux) BIR KEZ kosulur.

    ssh <kullanici>@<linux-makine>
    python3 -m venv ~/pyscfenv && ~/pyscfenv/bin/pip install pyscf numpy
    ~/pyscfenv/bin/python generate_reference.py --out reference/

Uretilen JSON dosyalari repoya ISLENIR. Testler calisma zamaninda PySCF'e
BAGIMLI DEGILDIR — bu bilincli: PySCF Windows'ta wheel olarak yok (kaynak
tarball'i geliyor, C derleyici + BLAS gerektiriyor), ve islenmis referanslar hem
testleri hizlandiriyor hem de referans degerleri surum kontrolune sokuyor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyscf import gto, scf

# (etiket, atom dizgisi, baz) — atom dizgisi PySCF formatinda, ANGSTROM
SYSTEMS = [
    ("he", "He 0 0 0", "sto-3g"),
    ("h2_074", "H 0 0 0; H 0 0 0.74", "sto-3g"),
    ("h2_074_631g", "H 0 0 0; H 0 0 0.74", "6-31g"),
    ("h2_100", "H 0 0 0; H 0 0 1.00", "sto-3g"),
    ("h2_200", "H 0 0 0; H 0 0 2.00", "sto-3g"),
    ("lih", "Li 0 0 0; H 0 0 1.60", "sto-3g"),
    ("hf_mol", "H 0 0 0; F 0 0 0.92", "sto-3g"),
    ("h2o", "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", "sto-3g"),
    ("h2o_631g", "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", "6-31g"),
    ("nh3", "N 0 0 0.116; H 0 0.939 -0.271; H 0.813 -0.470 -0.271; H -0.813 -0.470 -0.271", "sto-3g"),
    ("ch4", "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629", "sto-3g"),
]

# Integral matrisleri yalnizca bu kucuk sistemler icin dokulur (dosya boyutu).
INTEGRAL_SYSTEMS = ["h2_074", "h2o"]

ELEMENTS = ["H", "He", "Li", "C", "N", "O", "F"]


def dump_basis(elements: list[str], basis_name: str) -> dict:
    """PySCF'in kendi baz seti verisini doker — elle sayi kopyalamayi kaldirir."""
    out = {}
    for element in elements:
        raw = gto.basis.load(basis_name, element)
        shells = []
        for shell in raw:
            angular = shell[0]
            primitives = [[float(v) for v in row] for row in shell[1:]]
            shells.append({"angular_momentum": angular, "primitives": primitives})
        out[element] = shells
    return out


def run_system(label: str, atom: str, basis: str) -> tuple[dict, dict | None]:
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    energy_total = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"{label}: PySCF SCF yakinsamadi — referans uretilemez")

    energy_nuclear = float(mol.energy_nuc())
    summary = {
        "atom": atom,
        "basis": basis,
        "unit": "Angstrom",
        "n_basis_functions": int(mol.nao),
        "n_electrons": int(mol.nelectron),
        "energy_total": energy_total,
        "energy_nuclear": energy_nuclear,
        "energy_electronic": energy_total - energy_nuclear,
        "orbital_energies": [float(e) for e in mf.mo_energy],
        "dipole_debye": [float(d) for d in mf.dip_moment(verbose=0)],
        "mulliken_charges": [float(q) for q in mf.mulliken_pop(verbose=0)[1]],
    }

    integrals = None
    if label in INTEGRAL_SYSTEMS:
        integrals = {
            "atom": atom,
            "basis": basis,
            "unit": "Angstrom",
            "overlap": mol.intor("int1e_ovlp").tolist(),
            "kinetic": mol.intor("int1e_kin").tolist(),
            "nuclear": mol.intor("int1e_nuc").tolist(),
            "eri": mol.intor("int2e").tolist(),
        }
    return summary, integrals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="reference/ dizini")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import pyscf

    (out / "basis_sto3g.json").write_text(
        json.dumps(dump_basis(ELEMENTS, "sto-3g"), indent=1), encoding="utf-8"
    )
    (out / "basis_631g.json").write_text(
        json.dumps(dump_basis(ELEMENTS, "6-31g"), indent=1), encoding="utf-8"
    )

    energies = {"_pyscf_version": pyscf.__version__}
    for label, atom, basis in SYSTEMS:
        summary, integrals = run_system(label, atom, basis)
        energies[label] = summary
        if integrals is not None:
            (out / f"integrals_{label}.json").write_text(
                json.dumps(integrals), encoding="utf-8"
            )
        print(f"{label:14s} E = {summary['energy_total']:.10f}")

    (out / "hf_energies.json").write_text(
        json.dumps(energies, indent=1), encoding="utf-8"
    )
    print(f"\nPySCF {pyscf.__version__} ile uretildi -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
