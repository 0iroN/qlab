"""Spec bolum 6.5 -- deneysel degerlerle karsilastirma.

Bu dosya modulun ASIL iddiasini sinar: kodda "kovalent bag" diye bir sey yok;
bag, enerjinin bag uzunluguna gore minimum yapmasindan OKUNUR.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from qlab.engine.hf.analysis import dipole_debye
from qlab.engine.hf.scf import solve
from qlab.types import Molecule
from qlab.units import ANGSTROM_TO_BOHR, HARTREE_TO_EV

EXPERIMENTAL_H2_LENGTH_ANGSTROM = 0.741
EXPERIMENTAL_H2_DISSOCIATION_EV = 4.75
EXPERIMENTAL_WATER_ANGLE_DEG = 104.5
EXPERIMENTAL_WATER_DIPOLE_DEBYE = 1.85


def h2_energy(separation_angstrom: float) -> float:
    return solve(
        Molecule.from_string(f"H 0 0 0; H 0 0 {separation_angstrom}"), "sto-3g"
    ).energy_total


def h2_numerical_force(separation_angstrom: float, step: float = 1e-4) -> float:
    """|dE/dr| -- hartree/bohr, merkezi fark. Motorda analitik gradyan yok."""
    slope_per_angstrom = (
        h2_energy(separation_angstrom + step) - h2_energy(separation_angstrom - step)
    ) / (2 * step)
    return abs(slope_per_angstrom) / ANGSTROM_TO_BOHR


@pytest.mark.slow
def test_h2_potential_curve_has_a_minimum_near_experiment():
    """Bag BURADA dogar: enerji egrisinin minimumu denge bag uzunlugudur."""
    separations = np.linspace(0.4, 3.0, 40)
    energies = np.array([h2_energy(r) for r in separations])

    index = int(np.argmin(energies))
    assert 0 < index < len(separations) - 1, "minimum aralikta olmali"

    a, b, _ = np.polyfit(
        separations[index - 1 : index + 2], energies[index - 1 : index + 2], 2
    )
    equilibrium = -b / (2 * a)
    relative = abs(equilibrium - EXPERIMENTAL_H2_LENGTH_ANGSTROM) / EXPERIMENTAL_H2_LENGTH_ANGSTROM
    print(
        f"\nH2 denge bag uzunlugu: {equilibrium:.4f} A "
        f"(deneysel {EXPERIMENTAL_H2_LENGTH_ANGSTROM} A, sapma %{100 * relative:.1f})"
    )
    assert relative < 0.08


@pytest.mark.slow
def test_coarse_scan_fit_sits_above_the_true_sto3g_minimum():
    """Ustteki testin verdigi sayi bir DENGE DEGIL, izgara esidir.

    Tarama 0.4-3.0 A arasini 40 noktayla ortuyor; adim 0.0667 A. Bu kaba
    izgaraya oturtulan uc-nokta parabolu minimumu YUKARI kaydirir. Bu test
    ikisini ayirir: kaydirmanin isareti ve buyuklugu olculur, ve iki
    noktadaki kuvvet karsilastirilir -- fark uc buyukluk mertebesi.

    Neden onemli: README bir zamanlar kaba fiti (0.7156 A) "denge bag
    uzunlugu -- olculen" diye sunuyordu. Modul 2'nin kuvvet testleri
    hangisine demirlerse |F| ya ~0 ya da 3.6e-03 hartree/bohr olur; yani
    yanlis sayiya demirleyen bir MD testi "denge" saydigi noktada
    molekulu iterken bulur kendini.
    """
    separations = np.linspace(0.4, 3.0, 40)
    energies = np.array([h2_energy(r) for r in separations])
    index = int(np.argmin(energies))

    a, b, _ = np.polyfit(
        separations[index - 1 : index + 2], energies[index - 1 : index + 2], 2
    )
    coarse = -b / (2 * a)

    refined = minimize_scalar(
        h2_energy,
        bracket=(separations[index - 1], separations[index], separations[index + 1]),
        method="brent",
        options={"xtol": 1e-12},
    )

    assert refined.x == pytest.approx(0.7122299, abs=2e-5), "STO-3G'nin gercek minimumu"
    assert refined.fun == pytest.approx(-1.117505885116, abs=1e-9)

    bias = coarse - refined.x
    assert bias == pytest.approx(0.0033483, abs=2e-5), (
        "kaba fit gercek minimumun USTUNDE olmali; isaret de buyukluk de sabit"
    )

    force_coarse = h2_numerical_force(coarse)
    force_refined = h2_numerical_force(refined.x)
    print(
        f"\nH2: kaba fit {coarse:.4f} A (|F| {force_coarse:.3e}), "
        f"gercek minimum {refined.x:.4f} A (|F| {force_refined:.3e} hartree/bohr)"
    )

    assert force_coarse > 1e-3, "kaba fitte molekul hala itiliyor"
    assert force_refined < 1e-6, "gercek minimumda kuvvet sayisal gurultu seviyesinde"


@pytest.mark.slow
def test_h2_binding_energy_overshoots_because_rhf_cannot_dissociate():
    """RHF'nin ayrisma felaketi, olculmus bicimde.

    DIKKAT -- bu testin ilk hali YANLISTI ve dispatch oncesi olcumle
    duzeltildi. Once "HF deneyselden DUSUK baglanma verir" deniyordu
    (eksik korelasyon mantigiyla). Olculen bunun TERSI:

        E(0.74 A) = -1.1167593074 hartree
        E(6.0 A)  = -0.5899714207  -> baglanma 14.3346 eV
        E(8.0 A)  = -0.5789343078  -> baglanma 14.6350 eV
        E(10.0 A) = -0.5723195877  -> baglanma 14.8150 eV

    Deneysel ayrisma enerjisi 4.75 eV; RHF ucune katliyor. Sebep: RHF
    ayrisma limitinde iki notr H atomuna GIDEMEZ -- kisitli tek determinant
    %50 iyonik (H- H+) kalir, bu yuzden E(sonsuz) olmasi gerekenden cok
    YUKSEKTIR ve o fazlalik dogrudan baglanma enerjisine eklenir.
    Korelasyon eksikligi ters yonde calisir ama bu hatanin yaninda kucuk
    kalir.

    Yani HF'nin deneyselden sapmasi burada da BUG DEGIL -- ama sebebi
    korelasyon degil, yontemin ayrisma kusurudur. Ayrimi dogru yazmak,
    "HF deneyselden az verir" ezberini tekrarlamaktan yeglenir.
    """
    binding_ev = (h2_energy(8.0) - h2_energy(0.74)) * HARTREE_TO_EV
    print(
        f"\nH2 baglanma enerjisi (RHF/STO-3G, 8 A ayrisma): {binding_ev:.4f} eV, "
        f"deneysel {EXPERIMENTAL_H2_DISSOCIATION_EV} eV -- RHF ucune katliyor"
    )
    assert binding_ev > 0, "bag olusumu enerjiyi dusurmeli"
    assert binding_ev > 2 * EXPERIMENTAL_H2_DISSOCIATION_EV, (
        "RHF ayrisma limiti fazla yuksek oldugu icin baglanma ASIRI cikmali"
    )
    assert binding_ev == pytest.approx(14.635, abs=0.02, rel=0.0), (
        "olculmus deger; degisirse ayrisma davranisi degismis demektir"
    )


@pytest.mark.slow
def test_large_separation_breaks_down_numerically_and_we_say_so():
    """Beyan edilen sinir: cok buyuk ayrimda RHF cozumu sayisal olarak kirilir.

    Olculen (STO-3G, H2):
        R=10 A  S_01=1.59e-14  E=-0.572320   asimptota uyuyor (-0.572359)
        R=12 A  S_01=2.76e-20  E=-0.202656   ASIMPTOTTAN KOPUYOR (-0.567949)
        R=20 A  S_01=8.33e-54  E=-0.185017   (-0.559129)

    Integraller SAGLAM -- h_00, (00|11) ve cekirdek itmesi her mesafede
    duzgun ilerliyor, kontrol edildi. Kirilan ozdeger adimi: iki AO
    arasindaki eslesme ~1e-20'ye dustugunde kosegen elemanlarin yuvarlama
    farkinin (~1e-17) ALTINA iner ve eigh dejenere altuzayda keyfi bir baz
    secer. Simetrik (dogru) cozum yerine yerellesmis H- H+ cozumu bulunur;
    bu gecerli bir duragan noktadir ama cok daha yuksek enerjilidir.

    Bu bir integral hatasi degil, sonlu hassasiyetin siniridir. Gercek
    kuantum kimyasi paketleri bunu "stability analysis" ile yakalar; qlab'da
    O ARTIK VARDIR (engine/hf/stability.py) ve bu noktayi eyer noktasi
    olarak isaretler -- 12 A'da en dusuk orbital Hessian ozdegeri -2.92
    olculur, 10 A'da +2.89 (bkz. tests/hf/test_stability.py). Analiz
    kirilmayi DUZELTMEZ, yalnizca sessizligini kaldirir: enerji hala
    yanlistir, ama artik ciktida yanlis oldugu YAZAR.
    Sinir kilitlenmistir: 10 A'da dogru, 12 A'da bozuk. Biri gelip
    duzeltirse test kirmizi olur ve o zaman bu docstring guncellenir.
    """
    good = h2_energy(10.0)
    broken = h2_energy(12.0)
    assert good == pytest.approx(-0.572320, abs=1e-5, rel=0.0), "10 A saglam olmali"
    assert broken > good + 0.3, (
        "12 A'da simetri kirilmasi bekleniyor; duzelmisse docstring guncellensin"
    )


def water_energy(angle_deg: float, bond_angstrom: float = 0.9572) -> float:
    half = np.radians(angle_deg) / 2
    y = bond_angstrom * np.sin(half)
    z = bond_angstrom * np.cos(half)
    return solve(
        Molecule.from_string(f"O 0 0 0; H 0 {y} {z}; H 0 {-y} {z}"), "sto-3g"
    ).energy_total


@pytest.mark.slow
def test_water_equilibrium_angle_near_experiment():
    angles = np.linspace(90.0, 130.0, 21)
    energies = np.array([water_energy(a) for a in angles])

    index = int(np.argmin(energies))
    assert 0 < index < len(angles) - 1
    a, b, _ = np.polyfit(angles[index - 1 : index + 2], energies[index - 1 : index + 2], 2)
    equilibrium = -b / (2 * a)
    relative = abs(equilibrium - EXPERIMENTAL_WATER_ANGLE_DEG) / EXPERIMENTAL_WATER_ANGLE_DEG
    print(
        f"\nH2O denge acisi: {equilibrium:.2f} derece "
        f"(deneysel {EXPERIMENTAL_WATER_ANGLE_DEG}, sapma %{100 * relative:.1f})"
    )
    assert relative < 0.08


def test_water_dipole_within_hf_expectation(hf_energies):
    measured = dipole_debye(
        solve(Molecule.from_string(hf_energies["h2o"]["atom"]), "sto-3g")
    )
    relative = abs(measured - EXPERIMENTAL_WATER_DIPOLE_DEBYE) / EXPERIMENTAL_WATER_DIPOLE_DEBYE
    print(f"\nH2O dipolu: {measured:.3f} D (deneysel {EXPERIMENTAL_WATER_DIPOLE_DEBYE} D)")
    assert relative < 0.25


def test_basis_set_improves_energy(hf_energies):
    """Variasyonel ilke: daha buyuk baz asla daha yuksek enerji vermez."""
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    assert solve(molecule, "6-31g").energy_total <= solve(molecule, "sto-3g").energy_total
