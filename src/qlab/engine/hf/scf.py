"""Roothaan-Hall SCF dongusu: kapali kabuk (restricted) Hartree-Fock.

Matematik (spec bkz. .superpowers/sdd/b1-task-8-brief.md):

    X        = S^(-1/2)                          (Lowdin simetrik ortogonallestirme)
    P_ls     = 2 sum_i^occ C_li C_si
    F_uv     = Hcore_uv + sum_ls P_ls [ (uv|ls) - 1/2 (ul|vs) ]
    F'       = X^T F X  ->  ozdeger  ->  C = X C'
    E_elec   = 1/2 sum_uv P_uv (Hcore_uv + F_uv)
    E_toplam = E_elec + E_cekirdek

X = S^(-1/2) simetrik ortogonallestirme icin S'nin ozayrisimi kullanilir:
S = U s U^T (s kosegen, ozdegerler pozitif -- S bir ortusme matrisi, tekil
degil surece pozitif tanimlidir), X = U s^(-1/2) U^T.

Baslangic tahmini: core Hamiltoniyeni (Hcore) genellesmis ozdeger problemi
olarak cozulur (P=0 ile Fock = Hcore). Bu "core guess" olarak bilinir ve
DIIS olmadan bile makul bir baslangic noktasidir.

Yakinsama kriteri iki parcalidir (VE): enerji degisimi < energy_tolerance
VE RMS(e) < residual_tolerance, burada e = X^T (F P S - S P F) X ticaretcil
(commutator) hatasidir -- yakinsamis bir cozumde F ve P commute eder (FPS -
SPF = 0), yani e sifira gider.

DIIS (Pulay) hizlandirma: son en fazla _DIIS_MAX_HISTORY iterasyonun
(F_i, e_i) ciftleri tutulur ve

    [ B  1 ] [c]   [0]
    [ 1^T 0] [lam] = [1]      B_ij = tr(e_i^T e_j)

sisteminden c katsayilari cozulup F = sum c_i F_i ile bir sonraki
diyagonalizasyonda kullanilacak Fock kurulur. Enerji ve residual HER ZAMAN
duz (ekstrapole edilmemis) Fock/yogunluk ciftinden hesaplanir -- yalnizca
BIR SONRAKI yogunlugu uretmek icin kullanilan Fock ekstrapole edilir. Alt
uzay dolmadan (ilk 2 iterasyon) duz Fock kullanilir. B tekillesirse (bilinen
DIIS davranisi -- alt uzayda dogrusal bagimli hata vektorleri biriktiginde
olur) en eski cift atilip yerel bir kopya uzerinde tekrar denenir; kalici
gecmis bundan etkilenmez.

Duz Roothaan iterasyonu bazi sistemlerde yavas yakinsayabilir veya
salinabilir -- bu olculmesi gereken bir riskir, gizlenecek bir sorun degil.
Salinim, enerji gecmisindeki ardisik farklarin isaretinin tutarli sekilde
degistigi (bkz. _is_oscillating) durumlarda SCFConvergenceError.oscillating
alaninda raporlanir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor, overlap_matrix
from qlab.errors import SCFConvergenceError, UnsupportedSystemError
from qlab.provenance import Provenance
from qlab.types import Molecule


@dataclass(frozen=True)
class HFResult:
    molecule: Molecule
    basis_name: str
    energy_total: float
    energy_electronic: float
    energy_nuclear: float
    orbital_energies: np.ndarray
    coefficients: np.ndarray
    density: np.ndarray
    n_iterations: int
    provenance: Provenance


def _orthogonalisation_matrix(overlap: np.ndarray) -> np.ndarray:
    """X = S^(-1/2), Lowdin simetrik ortogonallestirme."""
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    inverse_sqrt = np.diag(eigenvalues ** -0.5)
    return eigenvectors @ inverse_sqrt @ eigenvectors.T


def _density_matrix(coefficients: np.ndarray, n_occupied: int) -> np.ndarray:
    """P_ls = 2 sum_i^occ C_li C_si."""
    occupied = coefficients[:, :n_occupied]
    return 2.0 * occupied @ occupied.T


def _fock_matrix(
    core: np.ndarray, eri: np.ndarray, density: np.ndarray
) -> np.ndarray:
    """F_uv = Hcore_uv + sum_ls P_ls [ (uv|ls) - 1/2 (ul|vs) ]."""
    coulomb = np.einsum("uvls,ls->uv", eri, density)
    exchange = np.einsum("ulvs,ls->uv", eri, density)
    return core + coulomb - 0.5 * exchange


def _electronic_energy(density: np.ndarray, core: np.ndarray, fock: np.ndarray) -> float:
    """E_elec = 1/2 sum_uv P_uv (Hcore_uv + F_uv)."""
    return 0.5 * float(np.einsum("uv,uv->", density, core + fock))


def _commutator_residual(
    fock: np.ndarray, density: np.ndarray, overlap: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """e = X^T (F P S - S P F) X, yakinsamada sifira giden ticaretcil hatasi."""
    fps = fock @ density @ overlap
    spf = overlap @ density @ fock
    return x.T @ (fps - spf) @ x


_DIIS_MAX_HISTORY = 8


def _diis_extrapolate(
    fock_history: list[np.ndarray], error_history: list[np.ndarray]
) -> np.ndarray:
    """DIIS (Pulay) ekstrapolasyonu: F = sum_i c_i F_i.

        [ B  1 ] [c]   [0]
        [ 1^T 0] [lam] = [1]      B_ij = tr(e_i^T e_j)

    Cagiran, en az 2 cift oldugundan emin olmalidir (kucuk alt uzaylarda
    ekstrapolasyonun bir anlami yoktur, cagiran taraf bunu duz Fock ile
    atlamalidir).

    B tekillesirse (alt uzayda dogrusal bagimli/tekrar eden hata vektorleri
    biriktiginde olur -- ozellikle yakinsamaya cok yaklasilmis, hata
    vektorleri neredeyse sifira inmis durumlarda) en eski cift atilip
    KUCULTULMUS bir alt uzayla tekrar denenir. Bu deneme yerel kopyalar
    uzerinde yapilir; kalici fock_history/error_history degistirilmez --
    bir sonraki iterasyonda ayni cift yine mevcut olabilir ve o zaman B
    tekil olmayabilir.
    """
    fock_list = list(fock_history)
    error_list = list(error_history)

    while len(fock_list) >= 2:
        m = len(error_list)
        b_matrix = np.array(
            [[np.trace(ei.T @ ej) for ej in error_list] for ei in error_list]
        )
        a_matrix = np.zeros((m + 1, m + 1))
        a_matrix[:m, :m] = b_matrix
        a_matrix[:m, m] = 1.0
        a_matrix[m, :m] = 1.0
        rhs = np.zeros(m + 1)
        rhs[m] = 1.0

        try:
            solution = np.linalg.solve(a_matrix, rhs)
        except np.linalg.LinAlgError:
            fock_list.pop(0)
            error_list.pop(0)
            continue

        coefficients = solution[:m]
        return sum(c * f for c, f in zip(coefficients, fock_list))

    # Butun ciftler B tekilligi yuzunden dusuruldu, tek eleman kaldi:
    # anlamli bir dogrusal kombinasyon kurulamaz -- en son duz Fock'a don.
    return fock_list[0]


_OSCILLATION_WINDOW = 10
_OSCILLATION_MIN_DELTAS = 4
_OSCILLATION_FLIP_RATIO = 0.7


def _is_oscillating(energy_history: list[float]) -> bool:
    """Enerji gecmisinde salinim tespiti.

    Kriter: son en fazla _OSCILLATION_WINDOW ardisik enerji farkinin
    (delta_i = E_i - E_{i-1}) isareti incelenir. Bu farklarin komsu
    ciftlerinin urunu negatifse (isaret degisti) bu bir "flip"tir. Monoton
    yakinsayan (veya monoton yakinsayip erken kesilen) bir SCF'de delta'lar
    hep ayni isarettedir (flip orani 0), gercek bir salinan sistemde (iki
    enerji arasinda gidip gelme) HER ardisik cift isaret degistirir (flip
    orani 1). Flip orani _OSCILLATION_FLIP_RATIO esigini asarsa salinim
    var sayilir.

    Isareti sifir olan delta'lar (tam yakinsama sinirinda, iki enerji
    ayni) flip sayimindan cikarilir -- ne "ayni yonde devam" ne de "isaret
    degisimi" sayilmamalidir.

    Yeterli veri yoksa (en az _OSCILLATION_MIN_DELTAS sifir-olmayan delta,
    yani en az birkac iterasyon gectiginde) temkinli sekilde False donulur:
    "henuz karar verilemez" varsayilan olarak salinim degildir -- bu,
    SCFConvergenceError.oscillating'in varsayilan degeriyle de tutarlidir.
    """
    window = energy_history[-(_OSCILLATION_WINDOW + 1):]
    deltas = np.diff(window)
    signs = deltas[deltas != 0]
    signs = np.sign(signs)
    if len(signs) < _OSCILLATION_MIN_DELTAS:
        return False
    flips = np.sum(signs[1:] * signs[:-1] < 0)
    flip_ratio = flips / (len(signs) - 1)
    return bool(flip_ratio >= _OSCILLATION_FLIP_RATIO)


def solve(
    molecule: Molecule,
    basis_name: str = "sto-3g",
    max_iterations: int = 100,
    energy_tolerance: float = 1e-10,
    residual_tolerance: float = 1e-8,
    use_diis: bool = True,
) -> HFResult:
    """Kapali kabuk RHF enerjisini Roothaan-Hall SCF donguleriyle cozer.

    use_diis=True (varsayilan) ise DIIS (Pulay) hizlandirma uygulanir:
    diyagonalizasyonda kullanilan Fock, son en fazla _DIIS_MAX_HISTORY
    iterasyonun ekstrapole edilmis kombinasyonudur. Bu yalnizca yakinsama
    YOLUNU kisaltir; enerji ve residual her zaman duz (ekstrapole
    edilmemis) Fock/yogunluk ciftinden hesaplanir, dolayisiyla yakinsama
    kriteri ve sonuc DIIS'ten bagimsizdir (bkz. modul dokstringi).
    """
    if molecule.n_electrons % 2 != 0:
        raise UnsupportedSystemError(
            f"acik kabuk sistem desteklenmiyor (n_electrons={molecule.n_electrons}); "
            "yalnizca kapali kabuk (cift elektron) sistemler cozulebilir"
        )

    n_occupied = molecule.n_electrons // 2

    basis = build_basis(molecule, basis_name)
    overlap = overlap_matrix(basis)
    core = core_hamiltonian(basis, molecule)
    eri = eri_tensor(basis)
    nuclear_repulsion = molecule.nuclear_repulsion()

    x = _orthogonalisation_matrix(overlap)

    # Baslangic tahmini: core guess (P=0 ile Fock = Hcore).
    fock_prime = x.T @ core @ x
    _, coefficients_prime = np.linalg.eigh(fock_prime)
    coefficients = x @ coefficients_prime
    density = _density_matrix(coefficients, n_occupied)

    energy_history: list[float] = []
    fock_history: list[np.ndarray] = []
    error_history: list[np.ndarray] = []
    fock = core
    residual_norm = float("inf")

    for iteration in range(1, max_iterations + 1):
        fock = _fock_matrix(core, eri, density)
        energy_electronic = _electronic_energy(density, core, fock)
        energy_history.append(energy_electronic)

        # Residual/enerji DAIMA duz (ekstrapole edilmemis) Fock-yogunluk
        # ciftinden hesaplanir -- yakinsama kriteri boylece DIIS'ten
        # bagimsiz kalir (bkz. modul dokstringi).
        residual = _commutator_residual(fock, density, overlap, x)
        residual_norm = float(np.sqrt(np.mean(residual ** 2)))

        if use_diis:
            fock_history.append(fock)
            error_history.append(residual)
            if len(fock_history) > _DIIS_MAX_HISTORY:
                fock_history.pop(0)
                error_history.pop(0)
            # Alt uzay dolmadan (ilk 2 iterasyon) duz Fock kullan.
            if iteration > 2:
                fock = _diis_extrapolate(fock_history, error_history)

        # Yeni katsayilari/yoguntugu her iterasyonda guncelle (yakinsasa da
        # yakinsamasa da), boylece dongu kirilinca ekstra bir tekrar
        # hesaplamaya gerek kalmaz: donen coefficients/density/orbital_energies
        # tam da bu iterasyonda diyagonalize edilen Fock'tan turetilmis olur
        # (DIIS aktifse bu, ekstrapole edilmis Fock'tur).
        fock_prime = x.T @ fock @ x
        orbital_energies, coefficients_prime = np.linalg.eigh(fock_prime)
        coefficients = x @ coefficients_prime
        density = _density_matrix(coefficients, n_occupied)

        converged = (
            iteration > 1
            and abs(energy_history[-1] - energy_history[-2]) < energy_tolerance
            and residual_norm < residual_tolerance
        )
        if converged:
            break
    else:
        raise SCFConvergenceError(
            iterations=max_iterations,
            energy_history=energy_history,
            final_residual=residual_norm,
            oscillating=_is_oscillating(energy_history),
        )

    energy_electronic = energy_history[-1]
    energy_total = energy_electronic + nuclear_repulsion

    provenance = Provenance(
        method=f"RHF/{basis_name.upper()}",
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
        converged=True,
        diagnostics={"iterations": iteration, "final_residual": residual_norm},
    )

    return HFResult(
        molecule=molecule,
        basis_name=basis_name,
        energy_total=energy_total,
        energy_electronic=energy_electronic,
        energy_nuclear=nuclear_repulsion,
        orbital_energies=orbital_energies,
        coefficients=coefficients,
        density=density,
        n_iterations=iteration,
        provenance=provenance,
    )
