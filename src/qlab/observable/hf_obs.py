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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.hf.basis import BasisFunction, build_basis
from qlab.engine.hf.scf import HFResult
from qlab.errors import InvalidInputError
from qlab.types import Molecule


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
