"""Atomik birimler ve SI dönüşümleri.

Motorların içi baştan sona atomik birimlerdedir (hbar = m_e = e = 4*pi*eps0 = 1).
Dönüşüm YALNIZCA giriş (senaryo okuma) ve çıkış (görselleştirme, rapor)
sınırlarında yapılır. Bu dosya projenin tek dönüşüm noktasıdır; başka hiçbir
yerde çıplak dönüşüm katsayısı bulunmamalıdır.

Kaynak: CODATA 2018.
"""

from __future__ import annotations

import numpy as np

BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM

HARTREE_TO_EV: float = 27.211386245988
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV

AU_TIME_TO_SECOND: float = 2.4188843265857e-17
AU_TIME_TO_FEMTOSECOND: float = AU_TIME_TO_SECOND * 1e15

AU_DIPOLE_TO_DEBYE: float = 2.541746473

# --- Moleküler dinamik (Modül 2) ---------------------------------------------
# Bu üçü ÇIPLAK KATSAYI olarak hiçbir yerde tekrarlanmaz; motorların içi
# atomik birimdedir ve dönüşüm yalnızca burada yapılır.

HARTREE_TO_JOULE: float = 4.3597447222071e-18
BOLTZMANN_JOULE_PER_KELVIN: float = 1.380649e-23

KB_HARTREE_PER_KELVIN: float = BOLTZMANN_JOULE_PER_KELVIN / HARTREE_TO_JOULE

#: Atomik kütle birimi ELEKTRON KÜTLESİ cinsinden. Atomik birimde kütlenin
#: birimi m_e'dir; bir protonu "1" yazmak 1822.888486 kat hafif bir çekirdek
#: demektir ve o hata sessizdir -- periyot 42.695298 kat kısalır ama enerji
#: korunmaya devam eder.
ATOMIC_MASS_UNIT_KG: float = 1.66053906660e-27
ELECTRON_MASS_KG: float = 9.1093837015e-31
AMU_TO_ELECTRON_MASS: float = ATOMIC_MASS_UNIT_KG / ELECTRON_MASS_KG

#: Hartree -> dalga sayısı (cm^-1). Titreşim frekanslarını spektroskopik
#: referanslarla karşılaştırmak için tek dönüşüm noktası.
#: TÜRETİLİR, elle yazılmaz: E_h / (h c). ÖLÇÜLDÜ -- türetilen değer
#: 219474.63136319647, CODATA tablo değerinden bağıl farkı 1.60e-14.
PLANCK_JOULE_SECOND: float = 6.62607015e-34
SPEED_OF_LIGHT_CM_PER_S: float = 2.99792458e10
HARTREE_TO_WAVENUMBER: float = HARTREE_TO_JOULE / (
    PLANCK_JOULE_SECOND * SPEED_OF_LIGHT_CM_PER_S
)


def bohr_to_angstrom(x: float | np.ndarray) -> float | np.ndarray:
    return x * BOHR_TO_ANGSTROM


def angstrom_to_bohr(x: float | np.ndarray) -> float | np.ndarray:
    return x * ANGSTROM_TO_BOHR


def hartree_to_ev(x: float | np.ndarray) -> float | np.ndarray:
    return x * HARTREE_TO_EV


def ev_to_hartree(x: float | np.ndarray) -> float | np.ndarray:
    return x * EV_TO_HARTREE


def au_time_to_femtosecond(x: float | np.ndarray) -> float | np.ndarray:
    return x * AU_TIME_TO_FEMTOSECOND


def amu_to_electron_mass(x: float | np.ndarray) -> float | np.ndarray:
    """Atomik kütle biriminden elektron kütlesine.

    MD senaryoları kütleyi amu olarak alır (insanın yazdığı birim); motor
    içi m_e'dir. Dönüşüm burada, sınırda yapılır.
    """
    return x * AMU_TO_ELECTRON_MASS
