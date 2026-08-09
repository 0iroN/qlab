"""Kartezyen izgara ve HF baz fonksiyonu degerlendirmesi.

Bu dosya Modul 1B2'nin temeli: molekul etrafinda bir kutu izgarasi kurar ve
kontrakte bir Kartezyen Gauss baz fonksiyonunu bu izgaranin her noktasinda
degerlendirir. Sonraki adimlar (orbitaller, elektron yogunlugu, yogunluk
farki, izoyuzeyler) hepsi buradan okur.

Dogrulama fikri: tek bir baz fonksiyonunun bir noktadaki degeri icin disaridan
referans yok, ama iki fonksiyonun izgara uzerindeki sayisal ortusmesi
(carpimlarinin izgara uzerinden toplami * hacim elemani) motorun analitik
overlap_matrix'ine (qlab.engine.hf.integrals) oturmak zorunda. Bu, yeni kodu
Modul 1B'nin PySCF'e karsi dogrulanmis motoruna zincirler ve bagimsiz bir
referans veri gerektirmez (bkz. tests/hf/test_hf_obs.py).

Baz fonksiyonu degeri (A merkez, (i,j,k) acisal uclu):
    phi(r) = c_kontraksiyon
             * (x-Ax)^i (y-Ay)^j (z-Az)^k
             * SUM_p d_p N_p exp(-a_p |r-A|^2)

Uc carpanin ucu de gereklidir: primitif katsayisi d_p, primitif
normalizasyonu N_p (BasisFunction.normalisation()) ve kontraksiyon
normalizasyonu c_kontraksiyon (BasisFunction.contraction_normalisation()).
Kontraksiyon normalizasyonunun atlanmasi Plan B1 Task 5'te PySCF'ten 6e-9
sapmaya yol acmisti (bkz. BasisFunction.contraction_normalisation() dokstring'i).

Dosyanin ikinci yarisi promolekuldur: molekulun geometrisinde duran ama
birbirini gormeyen serbest atomlarin yogunluklarinin toplami. RHF serbest
atomlari genel olarak cozemedigi icin (tek elektronlu H acik kabuktur, O'nun
taban durumu ucludur) referans, kuresel ortalamali ve kesirli isgalli bir
serbest atomdur -- gercek taban durum DEGILDIR ve bu Provenance'ta yazilidir.
Turetme ve olcumler promolecule_density'nin hemen ustundeki blokta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.hf.basis import BasisFunction, build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor, overlap_matrix
from qlab.engine.hf.scf import (
    HFResult,
    _electronic_energy,
    _fock_matrix,
    _orthogonalisation_matrix,
)
from qlab.errors import (
    InvalidInputError,
    SCFConvergenceError,
    UnsupportedSystemError,
)
from qlab.provenance import Provenance
from qlab.types import Atom, Molecule


@dataclass(frozen=True)
class VolumeGrid:
    """Molekul uzayi uzerinde esit aralikli bir Kartezyen kutu izgarasi.

    Koordinatlar ve `spacing` bohr birimindedir. `origin`, izgaranin
    (0,0,0) indeksli kosesidir; izgara `origin + spacing * indeks` ile
    her eksende `shape[eksen]` noktaya genisler.
    """

    origin: np.ndarray
    spacing: np.ndarray
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        for eksen, nokta in enumerate(self.shape):
            if nokta < 2:
                raise InvalidInputError(
                    f"izgara her eksende en az 2 nokta icermeli "
                    f"(eksen {eksen}), verilen shape: {self.shape}"
                )
        spacing = np.asarray(self.spacing, dtype=float)
        if not np.all(np.isfinite(spacing)):
            raise InvalidInputError(
                f"spacing her eksende sonlu olmali, verilen: {self.spacing}"
            )
        if np.any(spacing <= 0):
            raise InvalidInputError(
                f"spacing her eksende pozitif olmali (kutu ters veya "
                f"dejenere olmus olabilir), verilen: {self.spacing}"
            )

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Her eksen icin 1B koordinat dizisi (x, y, z)."""
        return tuple(
            self.origin[eksen] + self.spacing[eksen] * np.arange(self.shape[eksen])
            for eksen in range(3)
        )

    @property
    def volume_element(self) -> float:
        """Tek bir izgara hucresinin hacmi: eksen araliklarinin carpimi."""
        return float(np.prod(self.spacing))

    @property
    def points(self) -> np.ndarray:
        """Her izgara noktasinin (x, y, z) konumu, sekil (nx, ny, nz, 3)."""
        x, y, z = self.axes
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        return np.stack([xx, yy, zz], axis=-1)

    @classmethod
    def around(
        cls, molecule: Molecule, padding: float = 6.0, points_per_axis: int = 80
    ) -> VolumeGrid:
        """Molekulun tum cekirdeklerini kapsayan bir kutu izgarasi kurar.

        Her eksende sinir [min - padding, max + padding] araligidir ve bu
        aralik `points_per_axis` esit araliklarli noktaya bolunur (uc
        noktalar dahil, yani spacing = aralik / (points_per_axis - 1)).
        `padding` bohr birimindedir.

        `points_per_axis` en az 2 olmalidir (aksi halde spacing sifira
        bolme veya negatif/tanimsiz cikar) ve `padding` kutuyu ters
        cevirmemeli veya dejenere etmemelidir (bir eksende yuksek sinirin
        dusuk sinira esit ya da kucuk olmasi).
        """
        if points_per_axis < 2:
            raise InvalidInputError(
                f"points_per_axis en az 2 olmali, verilen: {points_per_axis}"
            )
        positions = np.array([atom.position for atom in molecule.atoms])
        low = positions.min(axis=0) - padding
        high = positions.max(axis=0) + padding
        if np.any(high <= low):
            raise InvalidInputError(
                f"padding kutuyu ters cevirdi veya dejenere etti "
                f"(yuksek sinir <= dusuk sinir): dusuk={low}, yuksek={high}, "
                f"padding={padding}"
            )
        spacing = (high - low) / (points_per_axis - 1)
        return cls(origin=low, spacing=spacing, shape=(points_per_axis,) * 3)


def evaluate_basis_function(function: BasisFunction, grid: VolumeGrid) -> np.ndarray:
    """Kontrakte baz fonksiyonunu izgaranin her noktasinda degerlendirir.

    Donen dizinin sekli grid.shape ile ayni, yani (nx, ny, nz). Vektorize
    edilmistir: izgara noktalari uzerinde Python dongusu kurulmaz.
    """
    displacement = grid.points - function.center  # (nx, ny, nz, 3)
    i, j, k = function.angular
    angular_factor = (
        displacement[..., 0] ** i
        * displacement[..., 1] ** j
        * displacement[..., 2] ** k
    )
    r_squared = np.sum(displacement**2, axis=-1)  # (nx, ny, nz)

    exponents = np.array([p.exponent for p in function.primitives])
    coefficients = np.array([p.coefficient for p in function.primitives])
    weights = coefficients * function.normalisation()  # (n_primitives,)

    radial = np.exp(-exponents[:, None, None, None] * r_squared[None, ...])
    radial_sum = np.tensordot(weights, radial, axes=(0, 0))  # (nx, ny, nz)

    return function.contraction_normalisation() * angular_factor * radial_sum


def evaluate_orbital(result: HFResult, index: int, grid: VolumeGrid) -> np.ndarray:
    """Bir molekuler orbitali izgaranin her noktasinda degerlendirir.

    Orbital, baz fonksiyonlarinin MO katsayilariyla (result.coefficients'in
    `index` sutunu) agirlikli toplamidir:

        psi_index(r) = SUM_mu C[mu, index] * phi_mu(r)

    `index` sifir tabanlidir ve result.coefficients'in sutun indeksidir
    (HOMO = molecule.n_electrons // 2 - 1, LUMO = molecule.n_electrons // 2).
    Sinir disi (negatif veya baz boyutunu asan) index icin InvalidInputError.

    Her baz fonksiyonu izgarada tam olarak bir kez degerlendirilir (evaluate_
    basis_function ile) ve agirlikli toplama eklenir -- baz, orbital basina
    yeniden hesaplanmaz.

    Baz, result.molecule ve result.basis_name'den build_basis ile yeniden
    kurulur; solve() de basisi ayni sekilde (build_basis(molecule, basis_name))
    kurdugu icin sutun sirasi result.coefficients ile eslesir.
    """
    n_basis = result.coefficients.shape[0]
    if not (0 <= index < n_basis):
        raise InvalidInputError(
            f"orbital indeksi [0, {n_basis}) araliginda olmali, verilen: {index}"
        )
    basis = build_basis(result.molecule, result.basis_name)
    orbital_coefficients = result.coefficients[:, index]

    orbital = np.zeros(grid.shape, dtype=float)
    for function, coefficient in zip(basis, orbital_coefficients):
        orbital += coefficient * evaluate_basis_function(function, grid)
    return orbital


@dataclass(frozen=True)
class ScalarField:
    """Izgara uzerinde degerlendirilmis bir skaler nicelik ve provenance'i.

    `values` sekli grid.shape ile aynidir, yani (nx, ny, nz). `provenance`,
    bu alanin nasil uretildigini ve izgaranin hangi sinirlari tasidigini
    beyan eder -- bkz. electron_density().
    """

    values: np.ndarray
    grid: VolumeGrid
    provenance: Provenance


def electron_density(result: HFResult, grid: VolumeGrid) -> ScalarField:
    """Elektron yogunlugunu izgaranin her noktasinda degerlendirir.

    rho(r) = SUM_uv P_uv phi_u(r) phi_v(r)

    P, result.density'dir (yogunluk matrisi). Her baz fonksiyonu izgarada
    tam olarak bir kez degerlendirilir (evaluate_basis_function ile) ve
    yogunluk matrisiyle kontrakte edilir -- evaluate_orbital isgal edilmis
    orbitaller uzerinde dongu icinde cagrilmaz.

    Donen ScalarField'in provenance'i izgaranin cozunurluk sinirini acikca
    beyan eder: izgara cekirdek orbitallerini cozmez, bu yuzden hidrojen
    disi atomlarda integre edilen elektron sayisi eksik cikar (degerlik
    orbitalleri etkilenmez). diagnostics, olculen ve beklenen elektron
    sayisini tasir ki sapma sayisal olarak gorunsun.
    """
    basis = build_basis(result.molecule, result.basis_name)
    values = np.stack([evaluate_basis_function(function, grid) for function in basis])
    density = np.einsum("uv,uxyz,vxyz->xyz", result.density, values, values)

    en_buyuk_aralik = float(grid.spacing.max())
    olculen = float(density.sum() * grid.volume_element)

    provenance = Provenance(
        method=f"RHF/{result.basis_name.upper()} elektron yogunlugu, kartezyen izgara",
        approximations=(
            "Born-Oppenheimer",
            "tek Slater determinanti",
            "sonlu baz seti",
            f"duzgun kartezyen izgara, aralik {en_buyuk_aralik:.3f} bohr",
        ),
        neglected=("elektron korelasyonu", "goreli etkiler"),
        expected_error=(
            "izgara cekirdek orbitallerini cozmez: hidrojen disi atomlarda "
            "integre edilen elektron sayisi eksik cikar (su/STO-3G, 0.203 bohr "
            "aralik: 9.70 / 10). Degerlik orbitalleri etkilenmez -- ayni izgarada "
            "HOMO normu 1.000000. Gorsellestirme icin uygundur, integre nicelik "
            "hesabi icin DEGILDIR."
        ),
        converged=result.provenance.converged,
        diagnostics={
            "integrated_electrons": olculen,
            "expected_electrons": result.molecule.n_electrons,
            "grid_spacing_bohr": en_buyuk_aralik,
        },
    )

    return ScalarField(values=density, grid=grid, provenance=provenance)


# --------------------------------------------------------------------------
# Promolekul: kuresel ortalamali, kesirli isgalli SERBEST atomlarin toplami
# --------------------------------------------------------------------------
#
# Promolekul, molekulun geometrisinde duran ama birbirini GORMEYEN serbest
# atomlarin yogunluklarinin toplamidir. Task 6 gercek yogunluktan bunu
# cikararak "bag kurulurken elektron nereye gitti" sorusunu olcer.
#
# SORUN: solve() serbest atomlari genel olarak COZEMEZ. Hidrojenin 1
# elektronu vardir (acik kabuk, UnsupportedSystemError) ve oksijenin gercek
# taban durumu bir uclu (3P) acik kabuktur, kapali kabuk degildir. Yani RHF
# serbest atom yogunlugunu dogrudan veremez.
#
# SECILEN REFERANS: kuresel ortalamali, kesirli isgalli serbest atom.
#   * Atom kendi merkezinde tek basina cozulur (yalnizca kendi cekirdegi).
#   * Baz fonksiyonlari simetri (l) bloklarina ayrilir. Serbest bir atomda
#     Fock matrisi bu bloklar arasinda TAM sifirdir (tek merkezli bazda
#     <s|h|p> = 0 ve kuresel bir yogunluk s-p birlesmesi uretmez), bu yuzden
#     her blok kendi genellesmis ozdeger problemi olarak cozulur.
#   * Radyal orbitaller enerjilerine gore siralanip aufbau ile doldurulur.
#     Bir p radyal kabugu 6 elektron tasir; kismen doluysa PAY UC KARTEZYEN
#     YONE ESIT DAGITILIR (kesirli isgal). Oksijende bu, her 2p yonune 4/3
#     elektron demektir ve sonuc kuresel simetrik olur.
#
# NEDEN KESIRLI ISGAL DOGRU ELEKTRON SAYISINI VERIR -- ve naif yol neden
# vermez: yogunluk P = SUM_i n_i c_i c_i^T ve integrali Tr(P S)'dir. Orbital
# katsayilari S-ORTONORMAL ise (C^T S C = I, yani genellesmis ozdeger
# probleminden gelirse) Tr(P S) = SUM_i n_i olur, yani TAM elektron sayisi.
# STO-3G'de ayni atomun 1s ve 2s fonksiyonlari DIK DEGILDIR (O icin ortusme
# 0.236704); orbitalleri S yerine Oklit normuyla normalize etmek (duz
# np.linalg.eigh) bu ozdesligi bozar ve kismen dolu bloklu atomlarda
# elektron sayisini kacirir (olculdu: Li icin 3.135216).
#
# Ayni dik-olmama, elektron sayisi dogru cikarken bile YOGUNLUGU bozabilir:
# P = diag(2, 2, 4/3, 4/3, 4/3) gibi naif bir kosegen isgal Tr(P S) = 8
# verir (S'nin kosegeni 1'dir) ama bir Slater determinantinin yogunlugu
# degildir -- dogal isgal sayilari (P S'nin ozdegerleri) {1.333, 1.333,
# 1.333, 1.527, 2.473} cikar ve 2.473 > 2 Pauli'yi ihlal eder. Dogru cozum
# dolu s blogunda P_ss = 2 S_ss^-1 verir (O icin kosegen 2.118709, kosegen
# disi -0.501507) -- tam da S^-1'in yaptigi is budur.
#
# ISIN NE KADARINI SCF YAPAR (olculdu): bir simetri blogu TAMAMEN doluysa
# isgal edilen alt uzay tum blogu kaplar ve C C^T = S^-1 olur, yani yogunluk
# ORBITALLERDEN BAGIMSIZ hale gelir. Bu yuzden H, He, C, N, O, F icin
# yogunluk ilk Fock'tan sonra degismez (core guess ile yakinsak cozum
# arasindaki fark O icin 1.1e-16). SCF yalnizca kismen dolu bloklu atomda
# gercekten calisir -- bu baz setinde tek ornek Li'dir (1s2 2s1): core guess
# ile yakinsak cozum arasindaki fark 0.055, yakinsama 12 iterasyon.
#
# BEYAN EDILEN SAPMA: bu referans atom GERCEK taban durum DEGILDIR. Gercek O
# atomu 3P'dir: kuresel degildir ve spin-kutupludur. Ayrica kesirli isgalli
# bir kabukta kapali kabuk Fock ifadesi (J - K/2) oz-etkilesimi tam
# goturmez. Ikisi de Provenance'ta yazilidir; gizlenmez.
#
# TERMINOLOJI: bu yapi literaturdeki "average of configuration" HF DEGILDIR --
# o, Slater'in acik-kabuk eslesme katsayilarini (coupling coefficients)
# kullanir. Burada SAD-tipi (superposition of atomic densities) bir yapi var:
# kapali-kabuk J - K/2 ifadesi dogrudan kesirli isgalli bir yogunluga
# uygulaniyor. Etkisi yukarida zaten olculdu: dolu bloklarda sifir (SCF ilk
# adimdan sonra degismiyor), yalnizca kismen dolu blokta (bu baz setinde
# yalnizca Li) devrede.

_MAX_ANGULAR_MOMENTUM = 1

_FREE_ATOM_MAX_ITERATIONS = 100
# Yakinsama olcutu yogunluk degisimidir (enerji degil): urettigimiz sey
# yogunluktur ve kesirli isgalli bir kabukta "enerji" zaten tam bir HF
# enerjisi degildir. Olculdu: dampingsiz dongu Li'de 12 iterasyonda
# 9.098e-14'e, digerlerinde 2 iterasyonda 8.9e-16'ya iner.
_FREE_ATOM_TOLERANCE = 1e-12


def _symmetry_blocks(basis: list[BasisFunction]) -> list[list[list[int]]]:
    """Serbest atomun baz fonksiyonlarini simetri (l) bloklarina ayirir.

    Her l degeri icin, o l'nin dejenere kartezyen bilesenlerine karsilik gelen
    indeks listeleri donulur: s icin tek liste, p icin (px, py, pz) uc liste.
    Her listenin k. elemani AYNI radyal kabuga aittir -- build_basis her
    kabugu tek seferde ve bilesenleri sabit sirada urettigi icin bu esleme
    kabuk sirasindan gelir. Bu DIGER BIR MODULE (basis.py) ait bir sira
    varsayimidir ve dogrudan sinanir:
    test_simetri_bloklari_p_yonlerini_ayni_radyal_kabuga_esler (O/6-31G'de),
    bkz. tests/hf/test_hf_obs.py.

    d ve ustu REDDEDILIR: ayni l'nin kartezyen bilesenleri d'de birbirine
    simetri geregi denk degildir (6 kartezyen bilesen, ama kabuk 12 degil 10
    elektron tasir), yani buradaki "esit dagit" kurali sessizce yanlis bir
    yogunluk uretirdi. basis.py su an d kabugunu zaten reddediyor; bu koruma
    o kural gevsetilirse diye burada duruyor.
    """
    gruplar: dict[int, dict[tuple[int, int, int], list[int]]] = {}
    for index, function in enumerate(basis):
        momentum = sum(function.angular)
        if momentum > _MAX_ANGULAR_MOMENTUM:
            raise UnsupportedSystemError(
                f"serbest atom yogunlugu icin desteklenmeyen acisal momentum "
                f"{momentum} (fonksiyon {function.angular}); yalnizca s ve p "
                "kabuklari kuresel ortalanabilir"
            )
        gruplar.setdefault(momentum, {}).setdefault(function.angular, []).append(index)
    return [
        [gruplar[momentum][acisal] for acisal in sorted(gruplar[momentum])]
        for momentum in sorted(gruplar)
    ]


def _free_atom_density(
    basis: list[BasisFunction], molecule: Molecule, n_electrons: int
) -> tuple[np.ndarray, int]:
    """Kuresel ortalamali, kesirli isgalli serbest atom yogunluk matrisi.

    `molecule` TEK atomlu olmalidir (serbest atom); `basis` o atomun baz
    fonksiyonlaridir ve donen matrisin satir/sutun sirasi bu listeye eslenir.

    Dongu Roothaan-Hall'un ta kendisidir (Fock kurulumu ve S^(-1/2)
    ortogonallestirme scf.py'den alinir -- ayni formulun ikinci bir kopyasi
    tutulmaz), tek farki isgal atamasidir: her simetri blogu ayri
    diyagonalize edilir, radyal orbitaller enerjiye gore siralanir ve aufbau
    ile doldurulur; kismen dolu bir kabugun payi kartezyen yonlere ESIT
    dagitilir.

    Aufbau siralamasi enerji sirasidir. Bu fonksiyon hem STO-3G hem 6-31G
    icin cagrilir; kapsamdaki en kucuk doldurma-sinniri araligi 6-31G'dedir,
    STO-3G'de degil -- STO-3G'de Li disindaki elementlerin dolu kabugu tam
    baz boyutunu kapladigindan orada bir sinir (dolu/bos gecisi) hic
    olusmaz. Olculdu: 6-31G'de Li'nin 2s -> 2p araligi 0.1330 hartree
    (kapsamdaki en kucuk), C/N/O/F icin 0.8392-2.0330 hartree araliginda.
    Yani siralama belirsiz degildir. Kabuk ICINDEKI dejenerasyon zaten
    kesirli isgalle ele alinir.

    BLOKLAMA VARSAYIMI OLCULDU, farz edilmedi: serbest atomda hem S'nin hem
    F'in s-p blogu TAM sifirdir (Li, C, N, O, F icin maks 0.000e+00), yani
    bloklari ayri diyagonalize etmek bir yaklasim degil, ozdesliktir. Ayni
    l'nin uc kartezyen yonu icin de tek bir temsilci yon (directions[0])
    cozulup diger ikisine uygulanir; uc yonun Fock bloklari arasindaki maks
    fark olculdu ve 1.554e-15'i (F) asmiyor -- yani temsilci secimi makine
    epsilonu mertebesinde farksizdir.

    Doner: (yogunluk matrisi, iterasyon sayisi). Yakinsamazsa
    SCFConvergenceError -- yakinsamamis bir yogunluk normal sonuc gibi
    donmez.
    """
    if n_electrons < 0 or n_electrons > 2 * len(basis):
        raise InvalidInputError(
            f"serbest atom elektron sayisi [0, {2 * len(basis)}] araliginda "
            f"olmali (minimal bazin tasiyabilecegi kadar), verilen: {n_electrons}"
        )

    overlap = overlap_matrix(basis)
    core = core_hamiltonian(basis, molecule)
    eri = eri_tensor(basis)
    blocks = _symmetry_blocks(basis)

    density = np.zeros((len(basis), len(basis)))
    energy_history: list[float] = []
    change = float("inf")

    for iteration in range(1, _FREE_ATOM_MAX_ITERATIONS + 1):
        fock = _fock_matrix(core, eri, density)
        # Kesirli isgalli bir kabukta bu nicelik TAM bir HF enerjisi degildir;
        # yalnizca yakinsamayan bir dongunun tanisi icin tutulur.
        energy_history.append(_electronic_energy(density, core, fock))

        radial: list[tuple[float, np.ndarray, list[list[int]]]] = []
        for directions in blocks:
            indices = directions[0]
            x = _orthogonalisation_matrix(overlap[np.ix_(indices, indices)])
            energies, vectors_prime = np.linalg.eigh(
                x.T @ fock[np.ix_(indices, indices)] @ x
            )
            vectors = x @ vectors_prime
            for column in range(len(indices)):
                radial.append((float(energies[column]), vectors[:, column], directions))
        radial.sort(key=lambda item: item[0])

        remaining = n_electrons
        new_density = np.zeros_like(density)
        for _, vector, directions in radial:
            occupation = min(2 * len(directions), remaining)
            remaining -= occupation
            if occupation == 0:
                continue
            block = (occupation / len(directions)) * np.outer(vector, vector)
            for indices in directions:
                new_density[np.ix_(indices, indices)] += block

        change = float(np.abs(new_density - density).max())
        density = new_density
        if iteration > 1 and change < _FREE_ATOM_TOLERANCE:
            break
    else:
        raise SCFConvergenceError(
            iterations=_FREE_ATOM_MAX_ITERATIONS,
            energy_history=energy_history,
            final_residual=change,
        )

    return density, iteration


def free_atom_density_matrix(
    symbol: str, basis_name: str = "sto-3g", n_electrons: int | None = None
) -> np.ndarray:
    """Tek bir serbest atomun kuresel ortalamali yogunluk matrisi.

    Varsayilan olarak NOTR atom cozulur (n_electrons = atom numarasi).
    `n_electrons` acikca verilebilir; bunun sebebi kapali kabuklu bir iyonun
    (ornegin Li+, 2 elektron) motorun PySCF'e karsi dogrulanmis solve()'uyla
    karsilastirilabilmesidir -- bu kurulumu bagimsiz bir referansa zincirleyen
    tek nokta odur (bkz. tests/hf/test_hf_obs.py).

    Yogunluk matrisi otelemeden bagimsizdir, bu yuzden atom orijine konur.
    """
    atom = Atom(symbol, np.zeros(3))
    molecule = Molecule((atom,))
    basis = build_basis(molecule, basis_name)
    if n_electrons is None:
        n_electrons = atom.atomic_number
    density, _ = _free_atom_density(basis, molecule, n_electrons)
    return density


def promolecule_density(
    molecule: Molecule, basis_name: str, grid: VolumeGrid
) -> ScalarField:
    """Promolekul yogunlugu: etkilesmeyen serbest atomlarin ust uste binmesi.

    rho_pro(r) = SUM_A SUM_{u,v in A} P^A_uv phi_u(r) phi_v(r)

    Yani molekuler yogunluk matrisi yerine, atom bloklarindan olusan BLOK
    KOSEGEN bir matris kullanilir: farkli atomlar arasi capraz terim yoktur,
    cunku serbest atomlar birbirini gormez. Kasitli olarak electron_density
    ile ayni kontraksiyonu kullanir; Task 6'nin fark haritasi iki alani ayni
    izgarada, ayni bazda ve ayni yoldan uretilmis olarak cikarabilsin diye.

    Her blok kuresel ortalamali, kesirli isgalli SERBEST NOTR atomdan gelir
    (bkz. bu bolumun basindaki turetme notu). Molekulun yuku atomlara
    DAGITILMAZ: promolekulun elektron sayisi her zaman SUM Z'dir. Yuklu bir
    molekulde bu, molecule.n_electrons'tan yuk kadar farklidir ve fark
    haritasi net -yuk kadar integre olur -- tanimin sonucu, hata degil.
    Beklenen sayi bu yuzden diagnostics'e SUM Z olarak yazilir.

    Donen ScalarField'in provenance'i iki ayri sapmayi acikca beyan eder:
    referans atomun gercek taban durum olmamasi ve izgaranin cekirdek
    kabuklarini cozememesi.
    """
    basis = build_basis(molecule, basis_name)
    density = np.zeros((len(basis), len(basis)))
    iterations = 0
    for atom_index, atom in enumerate(molecule.atoms):
        indices = [i for i, f in enumerate(basis) if f.atom_index == atom_index]
        atom_basis = [basis[i] for i in indices]
        block, atom_iterations = _free_atom_density(
            atom_basis, Molecule((atom,)), atom.atomic_number
        )
        density[np.ix_(indices, indices)] = block
        iterations = max(iterations, atom_iterations)

    values = np.stack([evaluate_basis_function(function, grid) for function in basis])
    promolecule = np.einsum("uv,uxyz,vxyz->xyz", density, values, values)

    en_buyuk_aralik = float(grid.spacing.max())
    olculen = float(promolecule.sum() * grid.volume_element)
    beklenen = sum(atom.atomic_number for atom in molecule.atoms)

    provenance = Provenance(
        method=(
            f"promolekul yogunlugu / {basis_name.upper()}: kuresel ortalamali, "
            "kesirli isgalli serbest atomlarin toplami, kartezyen izgara"
        ),
        approximations=(
            "Born-Oppenheimer",
            "serbest atomlar birbirini gormez (etkilesimsiz ust uste binme)",
            "kuresel ortalamali atom: kismen dolu kabugun payi kartezyen "
            "yonlere esit ve kesirli dagitilir",
            "notr serbest atom (molekulun yuku atomlara dagitilmaz)",
            "sonlu baz seti",
            f"duzgun kartezyen izgara, aralik {en_buyuk_aralik:.3f} bohr",
        ),
        neglected=(
            "atomlarin GERCEK taban durumu: acik kabuklu atomlarda taban durum "
            "kuresel degildir ve spin-kutupludur (O: 3P)",
            "kesirli isgalli kabukta oz-etkilesimin tam goturulmesi "
            "(kapali kabuk J - K/2 ifadesi kullanilir)",
            "elektron korelasyonu",
            "goreli etkiler",
        ),
        expected_error=(
            "referans atom gercek taban durum DEGILDIR: kuresel ortalama ve "
            "kesirli isgal bilincli bir secimdir, fark dogrudan yogunluk farki "
            "haritasina yazilir. Ayrica izgara cekirdek kabuklarini cozmez ve "
            "hata IKI YONE birden gider: O/STO-3G promolekulu 0.203 bohr "
            "aralikta 7.697/8 (eksik), 0.101 bohr aralikta 7.990/8; ama bir "
            "izgara noktasi tam cekirdegin uzerine duserse Riemann toplami "
            "ASAR (0.250 bohr aralikta 9.923/8). Hidrojende cekirdek/degerlik "
            "ayrimi yoktur ve integral tamdir (1.000000). Gorsellestirme ve "
            "fark haritasi icin uygundur, integre nicelik hesabi icin DEGILDIR."
        ),
        converged=True,
        diagnostics={
            "integrated_electrons": olculen,
            "expected_electrons": beklenen,
            "grid_spacing_bohr": en_buyuk_aralik,
            "atom_scf_iterations": iterations,
        },
    )

    return ScalarField(values=promolecule, grid=grid, provenance=provenance)
