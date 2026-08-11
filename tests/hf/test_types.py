import numpy as np
import pytest

from qlab.errors import InvalidInputError
from qlab.types import Atom, Molecule


def test_atomic_number_lookup():
    assert Atom("H", np.zeros(3)).atomic_number == 1
    assert Atom("O", np.zeros(3)).atomic_number == 8


def test_unknown_element_rejected():
    with pytest.raises(InvalidInputError, match="Bilinmeyen element"):
        Atom("Xx", np.zeros(3))


def test_position_must_be_three_dimensional():
    with pytest.raises(InvalidInputError, match="uc bilesenli"):
        Atom("H", np.zeros(2))


def test_electron_count_and_charge():
    mol = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    assert mol.n_electrons == 2
    cation = Molecule.from_string("H 0 0 0; H 0 0 0.74", charge=1)
    assert cation.n_electrons == 1


def test_from_string_converts_angstrom_to_bohr():
    mol = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    separation = np.linalg.norm(mol.atoms[1].position - mol.atoms[0].position)
    assert separation == pytest.approx(0.74 * 1.8897261254578281, rel=1e-9)


def test_from_string_accepts_bohr_directly():
    mol = Molecule.from_string("H 0 0 0; H 0 0 1.4", unit="bohr")
    separation = np.linalg.norm(mol.atoms[1].position - mol.atoms[0].position)
    assert separation == pytest.approx(1.4, rel=1e-12)


def test_nuclear_repulsion_matches_pyscf(hf_energies):
    """PySCF'in kendi cekirdek itme enerjisiyle birebir uyusmali.

    Bu test Angstrom->bohr donusumunun VE mesafe hesabinin ayni anda dogru
    oldugunu kanitlar; ikisinden biri yanlissa deger tutmaz.
    """
    record = hf_energies["h2_074"]
    mol = Molecule.from_string(record["atom"])
    assert mol.nuclear_repulsion() == pytest.approx(record["energy_nuclear"], rel=1e-10)


def test_nuclear_repulsion_matches_pyscf_for_water(hf_energies):
    record = hf_energies["h2o"]
    mol = Molecule.from_string(record["atom"])
    assert mol.nuclear_repulsion() == pytest.approx(record["energy_nuclear"], rel=1e-10)


def test_coincident_nuclei_rejected():
    with pytest.raises(InvalidInputError, match="cakisik"):
        Molecule.from_string("H 0 0 0; H 0 0 0")


def test_malformed_atom_line_rejected():
    with pytest.raises(InvalidInputError, match="Sembol"):
        Molecule.from_string("H 0 0")


def test_non_numeric_coordinate_rejected():
    with pytest.raises(InvalidInputError, match="sayiya"):
        Molecule.from_string("H 0 0 abc")


def test_odd_electron_count_is_allowed_by_molecule():
    """Molecule tek elektronu reddetmez — o kisit RHF cozucusune aittir."""
    assert Molecule.from_string("H 0 0 0").n_electrons == 1


# ---------------------------------------------------------------------------
# Sonlu olmayan koordinatlar


@pytest.mark.parametrize("deger", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("eksen", [0, 1, 2], ids=["x", "y", "z"])
def test_atom_sonlu_olmayan_koordinati_reddediyor(deger, eksen):
    """Konum sonlu olmak zorunda; kontrol EN ERKEN sınırda.

    ÖLÇÜLDÜ (düzeltmeden önce): `Atom` nan/±inf'i sessizce kabul ediyordu ve
    hata çok sonra, YANLIŞ YERDEN geliyordu — `solve()` yüz SCF iterasyonu
    koşup `SCFConvergenceError: SCF 100 iterasyonda yakinsamadi (son residual
    nan)` fırlatıyordu (scf.py:406). Tanı yakınsamayı suçluyordu, oysa kusur
    girdideydi. Yol boyunca `integrals.py`'nin beş ayrı satırından
    `RuntimeWarning` yağıyordu.

    `Atom` seçildi çünkü HER kurulum yolu (doğrudan, `from_string`,
    `dataclasses.replace`) buradan geçer.
    """
    konum = np.zeros(3)
    konum[eksen] = deger
    with pytest.raises(InvalidInputError, match="sonlu"):
        Atom("H", konum)


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param("H nan 0 0; H 0 0 0.74", id="nan"),
        pytest.param("H inf 0 0; H 0 0 0.74", id="inf"),
        pytest.param("H -inf 0 0; H 0 0 0.74", id="-inf"),
        pytest.param("H 1e400 0 0; H 0 0 0.74", id="1e400-tasma"),
    ],
)
def test_from_string_sonlu_olmayan_koordinati_reddediyor(spec):
    """`float()` "nan"/"inf" dizgelerini kabul eder; `1e400` sessizce taşar.

    Sonuncusu en sinsisi: girdide sonsuz diye bir şey YAZMIYOR, sonlu görünen
    bir sayı var, ama `float("1e400")` `inf` veriyor. Ölçüldü, dördü de
    düzeltmeden önce kabul ediliyordu.
    """
    with pytest.raises(InvalidInputError, match="sonlu"):
        Molecule.from_string(spec)


def test_cakisik_cekirdek_kontrolu_artik_nan_ile_atlatilamiyor():
    """REGRESYON: çakışıklık kontrolü nan/inf ile SESSİZCE atlatılabiliyordu.

    ÖLÇÜLDÜ (düzeltmeden önce): `distance < _MIN_SEPARATION_BOHR` karşılaştırması
    nan'da **False** döner, yani BİREBİR AYNI nan konumundaki iki atom
    "çakışık değil" sayılıyordu. İki atom da +inf'teyken de aynı: `inf - inf`
    = nan, yine False. Yani mevcut korumanın kendisi delinebiliyordu.

    Düzeltme `Atom` seviyesinde olduğu için bu yol artık daha kurulumda
    kapanıyor — çakışıklık kontrolü hiç sınanmadan önce.
    """
    for deger in (np.nan, np.inf):
        with pytest.raises(InvalidInputError, match="sonlu"):
            Molecule(
                (
                    Atom("H", np.array([deger, 0.0, 0.0])),
                    Atom("H", np.array([deger, 0.0, 0.0])),
                )
            )


def test_cekirdek_itmesi_artik_sessizce_sifir_ya_da_nan_donemiyor():
    """En sinsi yol: sonsuzdaki bir atom MEŞRU GÖRÜNEN bir sayı üretiyordu.

    ÖLÇÜLDÜ (düzeltmeden önce):
        H[nan] + H[0,0,1.4] -> nuclear_repulsion() = **nan**
        H[inf] + H[0,0,1.4] -> nuclear_repulsion() = **0.0**

    İkincisi tehlikeli olanı: `Z₁Z₂/inf` sıfırdır ve sıfır, bir enerji
    dökümünde tamamen makul görünür. nan en azından kendini belli eder.
    """
    for deger in (np.nan, np.inf):
        with pytest.raises(InvalidInputError, match="sonlu"):
            Molecule(
                (
                    Atom("H", np.array([deger, 0.0, 0.0])),
                    Atom("H", np.array([0.0, 0.0, 1.4])),
                )
            )
