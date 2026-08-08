"""Referans dosyalari var, okunabiliyor ve beklenen anahtarlari tasiyor."""

import numpy as np


def test_basis_data_has_expected_elements(sto3g_basis_data):
    for element in ("H", "He", "Li", "C", "N", "O", "F"):
        assert element in sto3g_basis_data, f"{element} STO-3G verisi eksik"


def test_hydrogen_sto3g_has_three_primitives(sto3g_basis_data):
    shells = sto3g_basis_data["H"]
    assert len(shells) == 1, "H'nin STO-3G'de tek kabugu olmali"
    assert shells[0]["angular_momentum"] == 0
    assert len(shells[0]["primitives"]) == 3


def test_oxygen_sto3g_has_s_and_p_shells(sto3g_basis_data):
    angulars = sorted(s["angular_momentum"] for s in sto3g_basis_data["O"])
    assert 0 in angulars and 1 in angulars, "O'da hem s hem p kabugu olmali"


def test_energies_reference_records_pyscf_version(hf_energies):
    assert "_pyscf_version" in hf_energies
    assert hf_energies["_pyscf_version"]


def test_h2_reference_energy_is_physically_sensible(hf_energies):
    record = hf_energies["h2_074"]
    assert record["n_electrons"] == 2
    assert record["n_basis_functions"] == 2
    assert -1.2 < record["energy_total"] < -1.0
    assert record["energy_nuclear"] > 0


def test_integral_matrices_are_square_and_consistent(h2_sto3g_integrals):
    overlap = np.array(h2_sto3g_integrals["overlap"])
    kinetic = np.array(h2_sto3g_integrals["kinetic"])
    eri = np.array(h2_sto3g_integrals["eri"])
    n = overlap.shape[0]
    assert overlap.shape == kinetic.shape == (n, n)
    assert eri.shape == (n, n, n, n)
    # rtol=0.0 acikca: varsayilan rtol=1e-7, beklenen deger 1.0 oldugu icin
    # "atol=1e-10" yazan iddiayi gercekte 1e-7'ye gevsetiyordu. Referans
    # dosyasindaki olculen sapma 2.22e-16; sinir 2e-15, ~9 kat pay.
    np.testing.assert_allclose(np.diag(overlap), 1.0, atol=2e-15, rtol=0.0)
