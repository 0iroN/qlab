"""Yakinsamis RHF dalga fonksiyonundan kimyasal buyukluklerin cikarilmasi.

Atomik yukler (Mulliken, Lowdin), dipol momenti, HOMO/LUMO boslugu ve Mayer
bag mertebesi -- hepsi HFResult.density ve HFResult.molecule/basis_name'den
turetilir. HFResult baz fonksiyonlarinin kendisini saklamaz (yalnizca
molecule + basis_name adi tasir), bu yuzden burada build_basis SCF'nin
kullandigi bazi yeniden kurar.

Spec bolum 6.3'ten tasinan kritik not: iyonik karakter bu projede
BIRINCIL olarak dipol momentiyle (dipole_moment/dipole_debye) olculur.
Mulliken ve Lowdin yukleri ikincil, tanisal metriklerdir -- baz setine
asiri duyarlidirlar ve (LiH/STO-3G'de goruldugu gibi, bkz.
mulliken_charges dokstringi) hatta ISARET olarak bile yanilabilirler.
Bu belirsizlik gizlenmez, olculur ve yazilir.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.hf.basis import BasisFunction, build_basis
from qlab.engine.hf.integrals import dipole_matrices, overlap_matrix
from qlab.engine.hf.scf import HFResult
from qlab.units import AU_DIPOLE_TO_DEBYE


def _basis_and_atom_index(result: HFResult) -> tuple[list[BasisFunction], np.ndarray]:
    """SCF'nin kullandigi bazi yeniden kurar ve her fonksiyonun atomunu dondurur.

    atom_index, basis.py'nin (Task 3) BasisFunction.atom_index alanindan
    dogrudan okunur -- burada yeni bir alan tanimlanmaz.
    """
    basis = build_basis(result.molecule, result.basis_name)
    atom_index = np.array([function.atom_index for function in basis])
    return basis, atom_index


def _nuclear_charges(result: HFResult) -> np.ndarray:
    return np.array([atom.atomic_number for atom in result.molecule.atoms], dtype=float)


def mulliken_charges(result: HFResult) -> np.ndarray:
    """Mulliken atomik yukleri, sekil (n_atoms,): q_A = Z_A - sum_{mu in A} (PS)_mu,mu.

    ONEMLI -- bu projenin iyonik karakter olcumu icin BIRINCIL metrigi
    DEGILDIR: bkz. dipole_moment/dipole_debye. Mulliken analizi, iki atom
    arasindaki paylasilan (cift-merkezli, PS'nin kosegen-disi kismindan
    gelen) yogunluk katkisini KOR bicimde yari yariya boler -- hangi
    atomun gercekten daha elektronegatif oldugunu, veya baz fonksiyonlarinin
    goreli boyutunu/yayilmasini dikkate almaz. Bu yuzden sonuc baz setine
    asiri duyarlidir ve bazen nitel olarak yaniltici cikar.

    Somut, olculmus ornek: LiH/STO-3G'de bu formul Li'ye KUCUK NEGATIF,
    H'ye KUCUK POZITIF yuk atar -- kimyasal sezginin (Li elektropozitif,
    H elektronegatif) tam tersi. Bu bir kod hatasi degildir (PySCF'in
    urettigi referans degerle tam eslesir, bkz. test_analysis.py
    test_lih_mulliken_sign_is_a_documented_basis_artifact); Mulliken
    yonteminin STO-3G gibi dengesiz/minimal bazlarda bilinen bir
    sinirlamasidir. Dogru YONLU (Li+, H-) iyonik karakter dipol
    momentinden okunur.

    Kisacasi: buyukluk (ve LiH gibi kenar durumlarda isaret bile) baza
    gore degisir; bu yuzden Mulliken/Lowdin burada tanisal/ikincil bir
    metrik olarak sunulur, iyonik karakterin olcusu olarak degil.
    """
    basis, atom_index = _basis_and_atom_index(result)
    overlap = overlap_matrix(basis)
    population = np.diag(result.density @ overlap)
    charges = _nuclear_charges(result)
    np.add.at(charges, atom_index, -population)
    return charges


def lowdin_charges(result: HFResult) -> np.ndarray:
    """Lowdin atomik yukleri, sekil (n_atoms,): q_A = Z_A - sum_{mu in A} (S^(1/2) P S^(1/2))_mu,mu.

    Mulliken'in simetrik bir alternatifi: paylasilan yogunlugu once
    S^(1/2) ile "ortogonal" bir baza tasiyip oyle bolusturur, bu da
    Mulliken'e gore baz setine biraz daha az duyarlidir. Ama bu da
    yalnizca tanisal/ikincil bir metriktir -- ayni uyari mulliken_charges
    dokstringindeki gibi gecerlidir: iyonik karakterin BIRINCIL olcusu
    dipole_moment/dipole_debye'dir.
    """
    basis, atom_index = _basis_and_atom_index(result)
    overlap = overlap_matrix(basis)
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    sqrt_overlap = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    population = np.diag(sqrt_overlap @ result.density @ sqrt_overlap)
    charges = _nuclear_charges(result)
    np.add.at(charges, atom_index, -population)
    return charges


def dipole_moment(result: HFResult) -> np.ndarray:
    """Dipol momenti, atomik birim (e*bohr), sekil (3,).

    d = sum_A Z_A R_A - sum_uv P_uv <mu|r|nu>

    Origin olarak molekul koordinat cercevesinin kendisi (0,0,0) kullanilir
    -- Molecule.atoms konumlariyla ayni cerceve. Molekul norotr oldugu
    surece (bu projede tum senaryolar norotr) sonuc origin secimine
    duyarsizdir.

    Bu, iyonik karakterin bu projedeki BIRINCIL olcusudur (bkz. spec 6.3 ve
    mulliken_charges dokstringi) -- Mulliken/Lowdin yuklerinin aksine baz
    setine asiri duyarli degildir ve isaret/yon konusunda guvenilirdir.
    """
    basis = build_basis(result.molecule, result.basis_name)
    origin = np.zeros(3)
    dipole_x, dipole_y, dipole_z = dipole_matrices(basis, origin)
    electronic = np.array(
        [
            np.sum(result.density * dipole_x),
            np.sum(result.density * dipole_y),
            np.sum(result.density * dipole_z),
        ]
    )
    nuclear = np.sum(
        [atom.atomic_number * atom.position for atom in result.molecule.atoms],
        axis=0,
    )
    return nuclear - electronic


def dipole_debye(result: HFResult) -> float:
    """Dipol momenti buyuklugu, Debye. Cikis siniri -- birim donusumu YALNIZ burada."""
    return float(np.linalg.norm(dipole_moment(result)) * AU_DIPOLE_TO_DEBYE)


def homo_lumo(result: HFResult) -> tuple[float, float, float]:
    """(homo, lumo, gap) orbital enerjileri, hartree.

    Kapali kabuk (RHF): en yuksek isgal edilen orbital n_occupied-1
    indeksinde (0-tabanli), en dusuk bos orbital n_occupied indeksindedir
    -- orbital_energies zaten artan sirada (np.linalg.eigh geregi).
    """
    n_occupied = result.molecule.n_electrons // 2
    homo = float(result.orbital_energies[n_occupied - 1])
    lumo = float(result.orbital_energies[n_occupied])
    return homo, lumo, lumo - homo


def mayer_bond_orders(result: HFResult) -> np.ndarray:
    """Mayer bag mertebesi matrisi B_AB, sekil (n_atoms, n_atoms).

    B_AB = sum_{mu in A} sum_{nu in B} (PS)_mu,nu (PS)_nu,mu

    (PS) genelde simetrik degildir (P simetrik, S simetrik, ama carpimlari
    degil), bu yuzden (PS)_mu,nu ve (PS)_nu,mu ayri ayri PS ve PS^T'den
    okunur. B_AB'nin B_BA'ya esitligi (mu,nu) dummy indekslerinin
    (A,B)<->(B,A) altinda yeniden adlandirilmasiyla analitik olarak
    garantidir; sayisal simetri testi bu ozdesligin uygulamada da
    (yuvarlama hatasi seviyesinde) tuttugunu dogrular.
    """
    basis, atom_index = _basis_and_atom_index(result)
    overlap = overlap_matrix(basis)
    population = result.density @ overlap
    population_transposed = population.T

    n_atoms = len(result.molecule.atoms)
    orders = np.zeros((n_atoms, n_atoms))
    for a in range(n_atoms):
        mask_a = atom_index == a
        for b in range(n_atoms):
            mask_b = atom_index == b
            block = population[np.ix_(mask_a, mask_b)]
            block_transposed = population_transposed[np.ix_(mask_a, mask_b)]
            orders[a, b] = np.sum(block * block_transposed)
    return orders
