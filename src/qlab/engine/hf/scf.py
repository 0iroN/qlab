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
from qlab.engine.hf.stability import StabilityResult, analyse_stability, orbital_hessian
from qlab.errors import SCFConvergenceError, UnsupportedSystemError
from qlab.provenance import Provenance, StabilityVerdict
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
    # Kararlilik analizi kosulmadiysa (solve(stability=False)) None kalir --
    # "analiz edilmedi" ile "kararli" ayni sey degildir ve ayni degerle
    # gosterilmemelidir.
    stability: StabilityResult | None = None


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


_HF_APPROXIMATIONS: tuple[str, ...] = (
    "Born-Oppenheimer",
    "tek Slater determinanti",
    "sonlu baz seti",
    "kapali kabuk (cift isgal)",
)

_HF_NEGLECTED: tuple[str, ...] = (
    "dinamik elektron korelasyonu",
    "statik korelasyon -- RHF ayrisma kusuru (notr atomlara ayrisamaz)",
    "goreli etkiler",
    "cekirdek hareketi",
    # Kararlilik analizi eklendiginde bu listeye eklendi: analiz YALNIZCA ic
    # (RHF -> RHF, tekil, reel) kararliligi sinar. "Kararli" hukmu, cozumun
    # kuresel minimum oldugunu veya UHF'in daha dusuk enerji vermeyecegini
    # SOYLEMEZ; bu siniri panelde yazmazsak hukum oldugundan genis okunur.
    "dis kararsizlik sinavi (RHF->UHF, triplet, kompleks orbital) -- "
    "yalnizca ic/tekil kararlilik test edilir",
)

# Bu metin, RHF'in EN BUYUK hatasini isimlendirmek zorundadir; bu depoda
# olculen sey sudur (tests/hf/test_physical.py, README "Ne iddia etmiyor"):
# H2 0.74 -> 8.0 A ayrimi icin baglanma enerjisi 14.6350 eV cikar, deneysel
# ayrisma enerjisi 4.75 eV'dir. Fark eksik korelasyondan GELMEZ -- kisitli tek
# determinant ayrisma limitinde %50 iyonik (H- H+) kalir, E(sonsuz) olmasi
# gerekenden yuksektir ve fazlalik dogrudan baglanma enerjisine eklenir.
# Eksik korelasyon gercek bir eksiktir ama ters yonde ve bu hatanin yaninda
# kucuktur. Panel her HF sonucunda TEK bir satir basar; o satirin ikincil
# sebebi baskin sebep gibi gostermesi, projenin ana iddiasiyla celisir.
_HF_EXPECTED_ERROR = (
    "gerilmis geometrilerde baskin hata eksik korelasyon DEGIL, kisitli tek "
    "determinantin ayrisma kusurudur: H2 icin 0.74 -> 8.0 A baglanma enerjisi "
    "14.6350 eV olculdu, deneysel ayrisma enerjisi 4.75 eV (~3 kat). Eksik "
    "korelasyon gercek ama daha kucuk bir etkidir ve ters yonde calisir."
)


def provenance_for(
    method: str,
    converged: bool,
    diagnostics: dict,
    stability: StabilityVerdict | None = None,
) -> Provenance:
    """RHF sonuclarinin ortak seffaflik blogu -- TEK kaynak.

    Hem solve() (tek nokta) hem factory.run_scan() (tarama) ayni
    yaklasimlari, ayni ihmal listesini ve ayni beklenen-hata metnini basar.
    Bu blok daha once iki dosyada birebir kopyalanmisti; bir kopya
    duzeltilip digeri unutuldugunda panel iki farkli seyi iddia ederdi.
    TDSE tarafindaki tdse/factory.py:provenance_for ile ayni desen.

    Yalnizca method/converged/diagnostics/stability cagriya gore degisir.
    stability=None, "kararlilik analizi kosulmadi" demektir ve panelde
    ilgili satir hic basilmaz.
    """
    return Provenance(
        method=method,
        approximations=_HF_APPROXIMATIONS,
        neglected=_HF_NEGLECTED,
        expected_error=_HF_EXPECTED_ERROR,
        converged=converged,
        diagnostics=diagnostics,
        stability=stability,
    )


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
    stability: bool = True,
) -> HFResult:
    """Kapali kabuk RHF enerjisini Roothaan-Hall SCF donguleriyle cozer.

    stability=True (varsayilan) ise yakinsamadan sonra IC (RHF -> RHF)
    kararlilik analizi kosulur: cozumun bir minimum mu yoksa eyer noktasi mi
    oldugu, tekil orbital Hessian'inin ozdegerlerinden belirlenir (bkz.
    stability.py). Kararsizlik YUKSELTMEZ -- eyer noktasi da gecerli bir
    duragan noktadir, duzeltilen kusur sessizliktir.

    Varsayilanin acik olmasi olcume dayanir: analiz, bir AO->MO ERI donusumu
    (N^5) gerektirir ve bu depodaki en buyuk sistemde bile SCF suresinin
    kucuk bir kesridir (rakamlar: .superpowers/sdd/b1-stability-report.md).

    stability=False atlanirsa sonuc SESSIZ KALMAZ ama EKSIK kalir:
    HFResult.stability ve Provenance.stability None olur, panel kararlilik
    satirini hic basmaz. Bu, "kararli bulundu" ile karistirilmamalidir --
    o durumda cozumun minimum mu eyer mi oldugu OLCULMEMISTIR.

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

    # Negatif elektron sayisi: CIFT oldugu icin yukaridaki kontrolden GECIYORDU.
    # Olculdu -- `_density_matrix`'teki `coefficients[:, :n_occupied]` dilimi
    # negatif n_occupied'da `[:, :-1]` olup SONDAN sutun atiyor, ve fonksiyon
    # BASKA bir sistemin dogru cevabini donduruyordu: H2 yuk +4 (-2 elektron)
    # -> E = -1.116759, yani NOTR H2'nin enerjisi; yuk +6 (-4 elektron)
    # -> E = +0.715104, yani sifir elektronlu sistemin enerjisi.
    if molecule.n_electrons < 0:
        raise UnsupportedSystemError(
            f"negatif elektron sayisi anlamsiz (n_electrons="
            f"{molecule.n_electrons}, charge={molecule.charge}); "
            "yuk cekirdek yuklerinin toplamini asamaz"
        )

    basis = build_basis(molecule, basis_name)

    # Baz setinin KAPASITESI. Kapali kabuk RHF her isgalli orbitale iki elektron
    # koyar, yani n_electrons/2 orbital gerekir; baz o kadarini uretmiyorsa
    # sistem bu baz setinde COZULEMEZ.
    #
    # Kontrol olmadan hata SESSIZDI, cunku numpy dilimi kirpiyor: uc isgalli
    # orbital istenip iki tane bulununca fazlasi dusuyor ve hesap, molekulun
    # BEYAN ETTIGINDEN az elektronla yapiliyordu. Olculdu (H2/STO-3G, iki
    # orbital): yuk -2 (4e) E=+0.923179 Tr(PS)=4 -- durust; yuk -4 (6e) ve
    # yuk -6 (8e) de AYNI E=+0.923179 ve AYNI Tr(PS)=4. Uc ayri kimyasal
    # sistem, tek cevap, tek kelime uyari yok.
    #
    # Sinir ASANI reddeder, TAM dolduranı degil: H2(2-) dort elektronu iki
    # orbitale cift isgalle yerlestirir ve Tr(PS)=4.0000000000 ile aritmetigi
    # durusttur (enerjinin pozitif cikmasi hata degil -- sistem bagli degil).
    n_occupied = molecule.n_electrons // 2
    if n_occupied > len(basis):
        raise UnsupportedSystemError(
            f"baz seti bu kadar elektronu tasiyamiyor: {molecule.n_electrons} "
            f"elektron {n_occupied} isgalli orbital ister, ama {basis_name!r} "
            f"bu molekul icin yalnizca {len(basis)} orbital uretiyor. "
            "Daha genis bir baz seti gerekir (ornegin '6-31g')."
        )
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
    # residual_norm sentineli: max_iterations=0 ile dongu hic donmezse
    # SCFConvergenceError yine de bir sayi tasimak zorunda.
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

    stability_result = None
    if stability:
        stability_result = analyse_stability(
            orbital_hessian(core, eri, coefficients, n_occupied)
        )

    provenance = provenance_for(
        method=f"RHF/{basis_name.upper()}",
        converged=True,
        diagnostics={"iterations": iteration, "final_residual": residual_norm},
        stability=None if stability_result is None else stability_result.verdict(),
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
        stability=stability_result,
    )
