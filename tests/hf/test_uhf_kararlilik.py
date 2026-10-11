"""B7 kapanisi: UHF -> UHF kararlilik analizi ve kararlilik IZLEYEN arama.

NE KAPANDI (09.10.2026). `search_uhf`in tohum ailesi HF molekulunde (H + F,
20 bohr, STO-3G) parca toplamini HIC bulmuyordu ve depo bunu GORMUYORDU: ailenin
buldugu iki durum da UHF EYERIdir ama UHF -> UHF Hessian'i yoktu. Simdi
(1) `uhf_orbital_hessian` analitik UHF Hessian'ini verir, (2) `solve_uhf(
stability=True)` her tek kosunun hukmunu basar, (3) `search_uhf` tohumlarin
en iyisine kararlilik izleme uygular (varsayilan) ve DONEN cozumun hukmunu
tasir; izleme kapaliysa sayilar B7 oncesiyle bit bit aynidir ama eyer "✗"
basar.

NE KAPANMADI (beyanla, testle): kuresel minimum garantisi YOK (T7), kompleks
kanal sinanmaz, |lambda| < 1e-8 yumusak modlar ayirt edilmez (T10'un sifir
modu bu bandin ICINDE kararli sayilir), ve izlemenin kabul esiginin altinda
kalan gercek inisler reddedilir -- catallanmanin hemen yaninda izleme eyerde
kalir ve "✗" basar (KOR_BOLGE testi).

Parametrelendirme (sonlu fark ile analitik ifade AYNI olmak zorunda):

    C_s(kappa) = C_s exp(K_s),  K_s[a, i] = kappa_ia,  K_s[i, a] = -kappa_ia
    indeks: once alfa (i * n_v + a), sonra beta.

Sonlu fark makinesi asagida URETIM KODUNDAN BAGIMSIZ yazilmistir: analitik
gradyan g_s[i, a] = 2 (C_s^T F_s C_s)[i, a] (Hessian formulunu KULLANMAZ),
merkezi fark h = 1e-4. Fark O(h^2) KESME hatasidir, gurultu degil (u04: bes
durumda bagil 6.667e-9 * max|H|, h = 3e-4'te 6.000e-8 -- oran 9 = 3^2).

Sayilarin hepsi bu depoda kamu yolundan (`search_uhf`, `solve_uhf`; izleme
parcalarinin, yumusak modun ve T7'nin eyerlerinde depodaki OZEL
`_follow_instability`) OLCULDU; olcum betikleri
`.superpowers/sdd/b7-olcum/uygulama/` ve `.superpowers/sdd/b7-olcum/duzeltme/`.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from scipy.linalg import expm

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor, overlap_matrix
from qlab.engine.hf.multistart import (
    _IZLEME_KABUL,
    _follow_instability,
    _uhf_enerjisi,
    search_uhf,
)
from qlab.engine.hf.scf import solve
from qlab.engine.hf.stability import (
    EIGENVALUE_TOLERANCE,
    orbital_hessian,
    triplet_orbital_hessian,
    uhf_orbital_hessian,
)
from qlab.engine.hf.uhf import energy_from_density, solve_uhf
from qlab.errors import InvalidInputError, SCFConvergenceError
from qlab.provenance import StabilityVerdict
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM

UZAK_A = 20.0 * BOHR_TO_ANGSTROM  # 20 bohr; test_uhf_cipalar ile AYNI dizgi
HF20 = f"H 0 0 0; F 0 0 {UZAK_A:.9f}"
E_H_STO3G = -0.466581849557276  # Hcore/S, tek atomdan (test_uhf_cipalar)


def _mol(metin: str, unit: str = "angstrom") -> Molecule:
    return Molecule.from_string(metin, unit=unit)


def _halka(n: int, d: float) -> Molecule:
    """Testin KENDI halka kurucusu (tests/tb/test_downfold_yasa.py ile ayni dizgi)."""
    a = d / (2 * np.sin(np.pi / n)) * BOHR_TO_ANGSTROM
    return Molecule.from_string(
        "; ".join(
            f"H {a*np.cos(2*np.pi*k/n):.17g} {a*np.sin(2*np.pi*k/n):.17g} 0"
            for k in range(n)
        )
    )


def _alternan(n: int, olcek: float) -> np.ndarray:
    return olcek * np.array([(-1.0) ** i for i in range(n)])


@functools.lru_cache(maxsize=None)
def _hf20(seeds: int, follow: bool):
    return search_uhf(_mol(HF20), "sto-3g", seeds=seeds, follow_instability=follow)


@functools.lru_cache(maxsize=None)
def _parca_toplami() -> float:
    """E(H) + E(F), parcalar AYRI hesaplanir (H tam: Hcore/S; F: search_uhf)."""
    flor = search_uhf(_mol("F 0 0 0"), "sto-3g", seeds=12, n_alpha=5)
    return E_H_STO3G + flor.energy


# --- Bagimsiz sonlu fark makinesi -----------------------------------------


def _integraller(molecule: Molecule):
    basis = build_basis(molecule, "sto-3g")
    return core_hamiltonian(basis, molecule), eri_tensor(basis)


def _dondur(c: np.ndarray, kappa: np.ndarray, n_occ: int) -> np.ndarray:
    n = c.shape[1]
    if n_occ == 0 or n_occ == n:
        return c
    k = np.zeros((n, n))
    blok = kappa.reshape(n_occ, n - n_occ)
    k[n_occ:, :n_occ] = blok.T
    k[:n_occ, n_occ:] = -blok
    return c @ expm(k)


def _gradyan(core, eri, ca, cb, na, nb) -> np.ndarray:
    pa, pb = ca[:, :na] @ ca[:, :na].T, cb[:, :nb] @ cb[:, :nb].T
    j = np.einsum("uvls,ls->uv", eri, pa + pb)
    fa = core + j - np.einsum("ulvs,ls->uv", eri, pa)
    fb = core + j - np.einsum("ulvs,ls->uv", eri, pb)
    ga = 2.0 * (ca.T @ fa @ ca)[:na, na:]
    gb = 2.0 * (cb.T @ fb @ cb)[:nb, nb:]
    return np.concatenate([ga.ravel(), gb.ravel()])


def _fd_hessian(core, eri, ca, cb, na, nb, h=1e-4) -> np.ndarray:
    ma = na * (ca.shape[1] - na)
    m = ma + nb * (cb.shape[1] - nb)
    hess = np.zeros((m, m))
    for q in range(m):
        e = np.zeros(m)
        e[q] = h
        arti = _gradyan(core, eri, _dondur(ca, e[:ma], na), _dondur(cb, e[ma:], nb), na, nb)
        eksi = _gradyan(core, eri, _dondur(ca, -e[:ma], na), _dondur(cb, -e[ma:], nb), na, nb)
        hess[:, q] = (arti - eksi) / (2 * h)
    return 0.5 * (hess + hess.T)


# --- T1 / T2: Hessian'in kendisi --------------------------------------------

_T1_DURUMLAR = {
    # ad: (molekul, n_alpha, bias)
    "HF20_S2_0_eyer": (HF20, None, np.zeros(6)),
    "HF20_S2_1_eyer": (HF20, None, _alternan(6, 1.0)),
    "N_atom_zemin": ("N 0 0 0", 5, np.zeros(5)),
    "N_atom_eyer": ("N 0 0 0", 5, _alternan(5, 0.35)),
    "H2_5bohr_BS": (f"H 0 0 0; H 0 0 {5.0 * BOHR_TO_ANGSTROM:.9f}", None, _alternan(2, 0.35)),
}


@pytest.mark.parametrize("ad", sorted(_T1_DURUMLAR))
def test_T1_ANALITIK_UHF_HESSIAN_SONLU_FARKLA_uyusuyor(ad):
    """T1. Acik kabukta TEK bagimsiz denetim (kapali kabuk ozdesligi Fa = Fb
    oldugu icin beta blogunda Fa kullanan bir mutanti GOREMEZ -- M4).

    OLCULDU: bes durumun besinde max|An - FD| / max|H| = 6.667e-9 (h = 1e-4)
    ve 6.000e-8 (h = 3e-4) -- oran 9 = 3^2, yani fark O(h^2) KESMEDIR,
    gurultu degil. Esik 1e-7 olculenin 15 kati. N atomu n_alpha = 5,
    n_beta = 2: alfa blogu BOS (5 orbitalin 5'i dolu), beta 2 x 3 -- na/nb
    takasini (M5) yakalayan durum budur.
    """
    metin, na, bias = _T1_DURUMLAR[ad]
    mol = _mol(metin)
    u = solve_uhf(mol, "sto-3g", n_alpha=na, initial_bias=bias)
    core, eri = _integraller(mol)
    an = uhf_orbital_hessian(
        core, eri, u.coefficients_alpha, u.coefficients_beta, u.n_alpha, u.n_beta
    )
    fd = _fd_hessian(
        core, eri, u.coefficients_alpha, u.coefficients_beta, u.n_alpha, u.n_beta
    )
    assert an.shape == fd.shape and an.size > 0
    assert float(np.abs(an - fd).max() / np.abs(an).max()) < 1e-7, ad


@pytest.mark.parametrize(
    "metin",
    ["H 0 0 0; H 0 0 1.4", "H 0 0 0; H 0 0 3.0", "O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11"],
)
def test_T2_KAPALI_KABUKTA_IC_ve_DIS_kanali_OZEL_HAL_olarak_iceriyor(metin):
    """T2. C_alfa = C_beta'da: Haa + Hbb + Hab + Hba = `orbital_hessian` ve
    Haa + Hbb - Hab - Hba = `triplet_orbital_hessian`.

    OLCULDU: bagil fark <= 3.9e-16 (H2 1.4 / 3.0 bohr, su). Uc ayri kod yolu
    (iki eski kanal FD ile ayri ayri dogrulanmisti) ayni nesneye iniyor.
    """
    mol = _mol(metin, unit="bohr")
    r = solve(mol, "sto-3g", stability=False)
    core, eri = _integraller(mol)
    n_occ = mol.n_electrons // 2
    c = r.coefficients
    h = uhf_orbital_hessian(core, eri, c, c, n_occ, n_occ)
    m = h.shape[0] // 2
    aa, ab, ba, bb = h[:m, :m], h[:m, m:], h[m:, :m], h[m:, m:]
    tekil = orbital_hessian(core, eri, c, n_occ)
    uclu = triplet_orbital_hessian(core, eri, c, n_occ)
    assert float(np.abs(aa + bb + ab + ba - tekil).max() / np.abs(tekil).max()) < 1e-12
    assert float(np.abs(aa + bb - ab - ba - uclu).max() / np.abs(uclu).max()) < 1e-12


# --- T3 / T4 / T9: HF molekulu, izlemeli ve izlemesiz ----------------------


@pytest.mark.parametrize(
    ("seeds", "kazanc"),
    [(1, 0.8824), (8, 0.8824), pytest.param(24, 0.7749, marks=pytest.mark.slow)],
)
def test_T3_HF_MOLEKULU_IZLEMEYLE_PARCA_TOPLAMINA_iniyor(seeds, kazanc):
    """T3. Varsayilan (izlemeli) arama HF@20'de parca toplamini BULUYOR.

    OLCULDU: seeds = 1 / 8 / 12 / 24 / 40 / 64 -> -98.45308680829324 /
    ...325 / ...325 / ...324 / ...325 / ...325; parca toplamina (E_H tam, E_F
    search_uhf(12)) 0 ... +1.4e-14 (<= 1 ulp). Hukum ✓, kabul edilen inis 1;
    kazanc 0.882405 (1, 8) ve 0.774865 (24). seeds = 24'te tohumlar 3 ayri
    cozum veriyor (izleme bu sayiyi DEGISTIRMEZ).
    Son haneler izleme ayrintisina 1e-14 duzeyinde bagli -- `==` ile KILITLENMEZ.
    """
    r = _hf20(seeds, True)
    assert abs(r.energy - _parca_toplami()) < 1e-11
    assert r.stability is not None and r.stability.stable
    assert r.stability.channel == "uhf"
    assert r.follow_cycles >= 1
    assert r.follow_gain == pytest.approx(kazanc, abs=1e-3)
    assert r.follow_gain == pytest.approx(r.seed_energy - r.energy, abs=0.0)
    if seeds == 24:
        assert r.n_distinct == 3


@pytest.mark.parametrize(
    ("seeds", "enerji", "lam"),
    [
        (8, -97.570681531688, -1.764811),
        pytest.param(24, -97.678222271884, -1.549729, marks=pytest.mark.slow),
    ],
)
def test_T4_IZLEMESIZ_SINIR_bit_bit_ESKI_SAYI_ama_HUKUM_EYER(seeds, enerji, lam):
    """T4. DURUST OLUMSUZ, korunur: tohum ailesi TEK BASINA parca toplamini bulmaz.

    OLCULDU (follow_instability=False): seeds = 8 -> -97.570681531688
    (S^2 = 0), seeds = 24 -> -97.678222271884 (S^2 = 1); B7 oncesinin
    literalleri. Ikisi de EYER: lambda_min -1.764811 / -1.549729 (iki katli).
    Sayi degismez, SESSIZLIK biter: hem `as_line` hem PANEL "✗ ... UHF EYERI"
    basar. Panelin hukmu UHF KANALINDAN gelmeli: izlemesiz yolun hukmunu "ic"
    kanalla kuran bir mutant (X9) paneli RHF cumlesiyle ("yakininda daha dusuk
    enerjili bir RHF cozumu var") basardi ve `as_line` kanala bakmadigi icin
    onu goremezdi. Yaklasim beyani da KOSULLU olmali: ayni panelde kosulsuz
    "search_uhf'in cozumu YEREL minimumdur" cumlesi "✗ UHF EYERI"yle celisirdi.
    """
    r = _hf20(seeds, False)
    assert r.energy == pytest.approx(enerji, abs=1e-9)
    assert r.follow_cycles == 0 and r.follow_gain == 0.0
    assert r.seed_energy == r.energy
    assert r.stability is not None and not r.stability.stable
    assert r.stability.channel == "uhf"
    assert r.stability.lowest_eigenvalue == pytest.approx(lam, abs=1e-5)
    assert "✗ donen cozum UHF EYERI" in r.as_line()
    panel = dict(r.result.provenance.as_lines())
    assert panel["Kararlılık"].startswith("✗ UHF EYERİ")
    assert "minimumdur" not in panel["Yaklaşımlar"]
    assert "KURESEL minimum oldugu hicbir durumda iddia EDILMEZ" in panel["Yaklaşımlar"]


@pytest.mark.parametrize("seeds", [1, pytest.param(24, marks=pytest.mark.slow)])
def test_T9_HUKUM_DONEN_COZUME_ait_tohumun_en_iyisine_DEGIL(seeds):
    """T9. Ayni tohumlar, iki yol: izlemeli hukum ✓ (lambda > -1e-8), izlemesiz
    ✗ (lambda < -1.5). Hukmu tohumun en iyisinden hesaplayan bir mutant
    (M8) izlemeli raporda ✗ basardi.

    OLCULDU: seeds = 1 izlemeli lambda_min -1.7e-15, izlemesiz -1.764811;
    seeds = 24 izlemeli -3.8e-15, izlemesiz -1.549729.
    """
    izlemeli, izlemesiz = _hf20(seeds, True), _hf20(seeds, False)
    assert izlemeli.stability.lowest_eigenvalue > -EIGENVALUE_TOLERANCE
    assert izlemesiz.stability.lowest_eigenvalue < -1.5
    assert izlemeli.result.stability is izlemeli.stability
    assert izlemeli.result.provenance.stability is izlemeli.stability
    assert izlemeli.seed_energy == izlemesiz.energy


# --- T5 / T8 / T10: N atomu, iki dongu, sifir mod -------------------------


def test_T5_N_ATOMU_tek_kosu_EYER_der_arama_YEREL_MINIMUM_dondurur():
    """T5. Bias bagimliligi TEK KOSU icin tasarim geregidir (bir duragan nokta
    doner) ama artik SESSIZ degildir.

    OLCULDU: solve_uhf(n_alpha=5, alternan bias 0.35, stability=True) ->
    -53.259271159, ✗, lambda_min -0.919478. search_uhf(12) -> -53.719010162594,
    ✓, lambda_min +0.919478, kabul edilen inis 0 (12 tohumun en iyisi zaten
    minimum; tohumlar 2 ayri cozum veriyor).
    """
    mol = _mol("N 0 0 0")
    tek = solve_uhf(mol, "sto-3g", n_alpha=5, initial_bias=_alternan(5, 0.35), stability=True)
    assert tek.energy_total == pytest.approx(-53.259271159, abs=1e-9)
    assert tek.stability is not None and not tek.stability.stable
    assert tek.stability.lowest_eigenvalue == pytest.approx(-0.919478, abs=1e-5)
    assert tek.provenance.stability is tek.stability

    arama = search_uhf(mol, "sto-3g", seeds=12, n_alpha=5)
    assert arama.energy == pytest.approx(-53.719010162594, abs=1e-9)
    assert arama.stability.stable
    assert arama.stability.lowest_eigenvalue == pytest.approx(0.919478, abs=1e-5)
    assert arama.follow_cycles == 0 and arama.follow_gain == 0.0


def test_T8_IZLEME_IKI_DONGU_gerektiren_vakayi_bitiriyor():
    """T8. H8 halkasi d = 2.0 bohr, seeds = 1 (sifir bias): tohum S^2 = 0'lik
    bir eyer (-3.9604335580), ilk inis baska bir eyere, ikinci inis minimuma.

    OLCULDU: kabul edilen inis 2, E -4.093600816789214, lambda_min +0.311217,
    kazanc 1.332e-01. Tavani 1 olan bir izleme (M10) eyerde kalirdi.
    """
    r = search_uhf(_halka(8, 2.0), "sto-3g", seeds=1)
    assert r.follow_cycles == 2
    assert r.energy == pytest.approx(-4.0936008168, abs=1e-9)
    assert r.stability.stable


def test_T10_SIFIR_MOD_kararsizlik_SAYILMIYOR():
    """T10. Ozgulluk: F atomunun p-deligi dejeneredir, dejenere donme bir SIFIR
    moddur ve kararsizlik DEGILDIR.

    OLCULDU: search_uhf(12, n_alpha=5) -> -97.98650495873598, lambda_min
    -5.33e-15 (sifirin ICINDE; esik 1e-8 bunun 1.9e6 kati), ✓, inis 0.
    Esigin isareti ters cevrilse (M9) bu sifir mod "kararsiz" sayilir.
    """
    r = search_uhf(_mol("F 0 0 0"), "sto-3g", seeds=12, n_alpha=5)
    assert r.stability.stable
    assert abs(r.stability.lowest_eigenvalue) < 1e-12
    assert r.follow_cycles == 0


# --- T11: panel metni -------------------------------------------------------


def test_T11_UHF_KANALI_KENDI_CUMLESINI_kuruyor_bilinmeyen_kanal_HATA():
    """T11. B7'den once bilinmeyen kanal sessizce "iç (RHF->RHF) minimum"
    cumlesine dusuyordu. Simdi "uhf" kendi cumlesini kurar ("YEREL", kuresel
    iddia yok), bilinmeyen ad ValueError verir.
    """
    kararli = StabilityVerdict(stable=True, lowest_eigenvalue=0.92, n_rotations=6, channel="uhf")
    eyer = StabilityVerdict(stable=False, lowest_eigenvalue=-0.92, n_rotations=6, channel="uhf")
    satir = kararli.as_line()
    assert satir.startswith("✓") and "UHF" in satir and "YEREL" in satir
    assert "küresel minimum iddia EDİLMEZ" in satir
    assert "RHF->RHF" not in satir and "RHF->UHF" not in satir
    assert eyer.as_line().startswith("✗") and "UHF EYERİ" in eyer.as_line()
    with pytest.raises(ValueError):
        StabilityVerdict(stable=True, lowest_eigenvalue=1.0, n_rotations=1, channel="xyz").as_line()

    izlemeli, izlemesiz = _hf20(1, True), _hf20(1, False)
    assert "✗" not in izlemeli.as_line()
    assert "kararlilik izlemesi 1 dongude" in izlemeli.as_line()
    # B7-a: arama satiri da "YEREL" der ve kuresel iddia etmez (downfold
    # paneli Kararlilik satirini basmaz, YALNIZ bu satiri basar).
    kararli_satir, eyer_satir = izlemeli.as_line(), izlemesiz.as_line()
    assert "donen cozum UHF YEREL minimum" in kararli_satir
    assert "KURESEL minimum iddia EDILMEZ" in kararli_satir
    assert "|lambda| < 1e-8 yumusak modlar ayirt edilmez" in kararli_satir
    assert "UHF-kararli" not in kararli_satir
    # CLI testi arama satirinda "tohum" SAYIYOR (mesafe basina bir kez).
    assert kararli_satir.count("tohum") == 1
    assert "✗ donen cozum UHF EYERI" in eyer_satir
    assert "YEREL" not in eyer_satir and "KURESEL" not in eyer_satir
    panel = dict(izlemeli.result.provenance.as_lines())
    assert "UHF YEREL minimum" in panel["Kararlılık"]


# --- T12: izleme SCF'li cozum dondurur -------------------------------------


def test_T12_IZLEMENIN_DONDURDUGU_COZUM_SCF_ile_yakinsamis():
    """T12. Izleme dondurulmus orbitalleri DEGIL, onlardan yeniden kosulmus
    kamu `solve_uhf`in sonucunu dondurur.

    OLCULDU (HF@20, seeds = 1): komutator artigi (Lowdin bazinda, testin kendi
    hesabi) max 1.13e-9 alfa / 9.50e-9 beta -- `solve_uhf`in kendi durma
    olcutunun (1e-8) altinda, tanim geregi; `energy_from_density` farki 0.0.
    Ayni eyerin en negatif yonde dondurulmus-ama-SCF'siz orbitalleri (M14)
    0.05 ... pi/2 acilarinda artigi 9.7e-3 ... 0.37 birakir.
    """
    mol = _mol(HF20)
    r = _hf20(1, True).result
    core, eri = _integraller(mol)
    s = overlap_matrix(build_basis(mol, "sto-3g"))
    w, v = np.linalg.eigh(s)
    x = v @ np.diag(w**-0.5) @ v.T
    pa, pb = r.density_alpha, r.density_beta
    j = np.einsum("uvls,ls->uv", eri, pa + pb)
    for p, k in ((pa, pa), (pb, pb)):
        f = core + j - np.einsum("ulvs,ls->uv", eri, k)
        artik = x.T @ (f @ p @ s - s @ p @ f) @ x
        assert float(np.abs(artik).max()) < 1e-8
    assert abs(energy_from_density(r, mol, "sto-3g") - r.energy_total) < 1e-10


# --- Izlemenin PARCALARI yuk tasiyor mu (M15a / M16 / M17 / M18) ----------
#
# Ilk mutasyon turunda dort parca HAYATTA KALDI: kabul denetimi yok (M15a),
# yalniz + yon (M16), DIIS'siz geri donus yok (M17), yalniz pi/2 (M18). NEDEN
# olculdu (`.superpowers/sdd/b7-olcum/uygulama/u08_izleme_parcalari.log`, 50
# eyer, 54 kabul edilen inis): kabul edilenlerin HEPSI siradaki ILK adaydan
# geliyor ve T1-T12'nin vakalarinda ilk DIIS'li yeniden baslatma zaten iniyor --
# yani o parcalar hic sinanmiyordu. Asagidaki vakalar her parcanin TEK BASINA
# fark yarattigi eyerlerdir: eyerler kamu `solve_uhf`ten uretilir, izleme
# depodaki OZEL `_follow_instability` ile kosulur. Iki parca SIGORTADIR ve
# burada KILITLI DEGIL: aday sayisi (`_IZLEME_ADAY = 1` mutanti) ve kucuk acilar
# 0.05/0.1/0.2 (onlarsiz mutant) testlerin hepsini geciyor; ikincisinin 22
# vakada fark yaratmadigi olculdu (duzeltme olcumu f03, multistart yorumu).


def _haar_ilk(mol: Molecule) -> tuple[np.ndarray, np.ndarray]:
    """rng 72026'nin ILK Haar cifti, S^-1/2 ile ortonormal (T7 ile ayni kurulum)."""
    s = overlap_matrix(build_basis(mol, "sto-3g"))
    w, v = np.linalg.eigh(s)
    c0 = v @ np.diag(w**-0.5) @ v.T
    rng = np.random.default_rng(72026)
    cift = []
    for _ in range(2):
        q, r = np.linalg.qr(rng.normal(size=(s.shape[0], s.shape[0])))
        cift.append(c0 @ (q * np.sign(np.diag(r))))
    return cift[0], cift[1]


def _blok(n: int, olcek: float) -> np.ndarray:
    b = np.ones(n)
    b[: n // 2] = -1.0
    return olcek * b


_PARCA_VAKALARI = {
    # ad: (molekul, baslangic turu, beklenen eyer, beklenen varis)
    "M16_EKSI_YON_H8_d3": (_halka(8, 3.0), "haar", -3.7539852603, -3.8572719930),
    "M17_DIISSIZ_H8_d2.6": (_halka(8, 2.6), "blok0.1", -3.8008561172, -3.9573119765),
    "M18_ARA_ACI_HF20_S2_1": (_mol(HF20), "haar", -97.6782222719, -98.4530868083),
}


@pytest.mark.parametrize("ad", sorted(_PARCA_VAKALARI))
def test_IZLEMENIN_PARCALARI_tek_tek_YUK_TASIYOR(ad):
    """Her vaka izlemenin BIR parcasi olmadan baska yere iner (olculdu, u08/u09/u10):

    M16 H8 d = 3.0, ilk Haar cifti, DIIS: eyer -3.7539852603 (lambda_0
        -0.3976, lambda_1 +0.635 -- dejenere DEGIL). Dogru izleme - yonde
        kuresele -3.8572719930 iner; yalniz + yon kuresel OLMAYAN -3.7773306557'ye.
    M17 H8 d = 2.6, blok bias 0.1 (tohum ailesinin 3. deseni): eyer -3.8008561172.
        DIIS'li yeniden baslatma 2000 iterasyonda YAKINSAMIYOR, DIIS'siz
        -3.9573119765'e iniyor; geri donus olmadan dort aday da duser ve eyer
        "✗" ile kalir.
    M18 HF@20, ilk Haar cifti, DIIS: S^2 = 1 eyeri -97.6782222719. Kabul 1.3 rad'da;
        yalniz pi/2 ile SCF eyere geri coker. DURUST SINIR: lambda_0 = lambda_1 =
        -1.549729 (iki katli), yani en negatif yon 2B ozuzayda BLAS'in secimidir.
        Ozuzayda 13 yon denendi (u10): dogru izleme 13/13'te kuresele iniyor --
        bu testin dogru kod icin SAGLAM oldugunun olcusu --, yalniz-pi/2 ise
        5/13'te eyerde kaliyor (bu makinenin secimi phi = 0 onlardan biri).
        Baska bir BLAS'ta M18 bu testi gecebilir; dogru kod gecmeye devam eder.
    """
    mol, tur, eyer_e, varis_e = _PARCA_VAKALARI[ad]
    if tur == "haar":
        eyer = solve_uhf(mol, "sto-3g", initial_orbitals=_haar_ilk(mol), max_iterations=3000, stability=True)
    else:
        eyer = solve_uhf(mol, "sto-3g", initial_bias=_blok(8, 0.1), stability=True)
    assert eyer.energy_total == pytest.approx(eyer_e, abs=1e-9)
    assert not eyer.stability.stable
    son, hukum, inis = _follow_instability(mol, "sto-3g", eyer, 2000)
    assert hukum.stable and inis == 1
    assert son.energy_total == pytest.approx(varis_e, abs=1e-9)


def test_YUMUSAK_MOD_HF_R8_bandin_UST_KENARINDA_yakalaniyor():
    """B7-b'nin (yumusak mod bandi) olculen kenari -- DURUST SINIR, kapanmadi.

    OLCULDU (eyer kamu `solve_uhf`ten, izleme depodaki OZEL
    `_follow_instability` ile): HF R = 8 bohr, ilk Haar cifti, DIIS (2083
    iterasyon): pi-delik eyeri -98.45308681061466, lambda_min -2.116e-7 --
    esigin (1e-8) yalniz 21 kati ustunde. Izleme TEK iniste sigma-delik
    minimumuna iner: -98.45308691601424 (lambda_min +2.099e-7), inis
    1.054e-7 Ha -- kabul esiginin (1e-9) 105 kati ustunde. Kamu yolu da ayni:
    izlemesiz search_uhf(12)'nin en iyisi lambda -2.116e-7 (12 tohumun YALNIZ
    1'i yakinsiyor), izlemeli search_uhf(12) 1.054e-7 indiriyor (duzeltme
    olcumu f02). Daha buyuk R'de lambda bandin ICINE girer ve ayrim kaybolur:
    R = 9 bohr'da search_uhf(12)'nin en iyisi lambda -5.189e-9 ile ✓ basiliyor,
    R = 10'da -7.571e-11; R = 9 noktasinin gercekten eyer olup olmadigi
    OLCULMEDI.
    """
    mol = _mol(f"H 0 0 0; F 0 0 {8.0 * BOHR_TO_ANGSTROM:.9f}")
    eyer = solve_uhf(mol, "sto-3g", initial_orbitals=_haar_ilk(mol), max_iterations=3000, stability=True)
    assert not eyer.stability.stable
    assert eyer.stability.lowest_eigenvalue == pytest.approx(-2.116e-7, abs=5e-10)
    son, hukum, inis = _follow_instability(mol, "sto-3g", eyer, 2000)
    assert hukum.stable and inis == 1
    assert hukum.lowest_eigenvalue == pytest.approx(2.099e-7, abs=5e-10)
    assert eyer.energy_total - son.energy_total == pytest.approx(1.054e-7, abs=1e-9)


def test_GERI_COKME_INIS_SAYILMAZ_kabul_esigi_yuk_tasiyor():
    """M15a (kabul denetimi YOK) icin vaka: H2, R = Rc + 0.1 bohr, seeds = 1.

    OLCULDU (u08): sifir bias'li tohum RHF eyeridir (-1.0030763101, lambda
    -0.0750). En dusuk aday (+0.2 rad) DIIS'le yeniden kosulunca SCF AYNI
    eyere geri cokuyor (dE = 0.0); dogru izleme onu reddedip DIIS'siz kosar ve
    TEK iniste -1.0046971069'a varir (kazanc 1.621e-03). Kabul denetimi
    olmasaydi geri cokme bir "inis" sayilir ve rapor 2 dongu derdi -- enerji
    ayni kalsa bile "kararlilik izlemesi N dongude" satiri yalan olurdu.
    """
    r = search_uhf(_h2(RC_BOHR + 0.1), "sto-3g", seeds=1)
    assert r.stability.stable
    assert r.follow_cycles == 1
    assert r.energy == pytest.approx(-1.0046971069, abs=1e-9)
    assert r.follow_gain == pytest.approx(1.621e-3, abs=1e-6)


def test_KOR_BOLGE_kabul_esiginin_altindaki_GERCEK_inis_reddediliyor_hukum_EYER():
    """DURUST OLUMSUZ, KAPANMADI (beyanli: README, multistart `_IZLEME_KABUL`).

    Catallanmanin hemen yaninda gercek inis dE ~ d^2 ile keyfi kucuktur ve
    kabul esiginin (1e-9) altinda kalir. OLCULDU (kamu yolu, duzeltme olcumu
    f02/f06): H2, R = Rc + 1e-5 bohr, varsayilan search_uhf(seeds=1) -> inis 0,
    RHF eyerinde kaliyor (RHF - E = -2.2e-16), lambda_min -7.739e-6 (yumusak
    mod bandinin 774 kati DISINDA), hukum ✗ ve panel "✗ UHF EYERI" basiyor.
    Ayni noktada search_uhf(12)'nin tohum ailesi kirik simetrili minimumu
    buluyor: ✓, lambda_min +1.781e-5, RHF - E = 1.730127e-11 -- esigin
    altinda. Kor bolge H2'de d <= 6e-5 (d = 8e-5'te inis 1.12e-9 kabul
    ediliyor). Engel esiktir: depodaki ozel yolda esik 1e-13 olunca ayni
    noktada tek inisle minimuma iniliyor.

    Bu test SINIRI kilitler: hukmun sessiz kalmadigini (✗) ve inisin gercekten
    esigin altinda oldugunu. Esik degisirse ya da izleme bu bolgeyi kapatirsa
    kirmizi olur -- o zaman README'deki kor bolge maddesi guncellenir.
    """
    m = _h2(RC_BOHR + 1e-5)
    rhf = solve(m, "sto-3g", stability=False).energy_total
    tek = search_uhf(m, "sto-3g", seeds=1)
    assert tek.follow_cycles == 0 and not tek.stability.stable
    assert abs(rhf - tek.energy) < 1e-13
    assert tek.stability.lowest_eigenvalue == pytest.approx(-7.739e-6, abs=1e-9)
    assert dict(tek.result.provenance.as_lines())["Kararlılık"].startswith("✗ UHF EYERİ")
    on_iki = search_uhf(m, "sto-3g", seeds=12)
    assert on_iki.stability.stable and on_iki.follow_cycles == 0
    assert 0.0 < rhf - on_iki.energy < _IZLEME_KABUL
    assert rhf - on_iki.energy == pytest.approx(1.730e-11, abs=1e-13)


def test_CIZGI_ARAMASI_ENERJISI_solve_uhf_ile_OZDES():
    """Izlemenin aday siralamasi `_uhf_enerjisi` ile yapilir; yakinsamis
    orbitallerde kamu `solve_uhf`in energy_total'ini vermeli. Acik kabuk, iki
    spin blogunda da hem dolu hem bos orbital var: OH radikali, n_alpha = 5,
    n_beta = 4, 6 baz fonksiyonu.

    OLCULDU (duzeltme olcumu f04): fark 0.0 (OH, Li, N). Degis-tokussuz ifade
    (X7) ayni orbitallerde OH'de +8.72 Ha sapiyor ama testlerin hepsini
    geciyordu: siralama yuk tasir (siralamasiz mutant T3'te oluyor), yanlis
    bir enerji ifadesi ise baska bir "ise yarayan" sira uretebiliyor.
    """
    mol = _mol("O 0 0 0; H 0 0 0.97")
    u = solve_uhf(mol, "sto-3g", n_alpha=5)
    core, eri = _integraller(mol)
    e = _uhf_enerjisi(
        core, eri, mol.nuclear_repulsion(),
        u.coefficients_alpha, u.coefficients_beta, u.n_alpha, u.n_beta,
    )
    assert (u.n_alpha, u.n_beta) == (5, 4)
    assert abs(e - u.energy_total) < 1e-12


# --- Girdi denetimi ve varsayilan yolun bit bit korunmasi -----------------


def test_SOLVE_UHF_varsayilanla_HUKUM_YOK_ve_baslangic_girdisi_denetleniyor():
    """`solve_uhf`in varsayilanlari B7 oncesini korur: hukum alani None ve
    provenance'ta "Kararlılık" satiri YOK. initial_orbitals ile initial_bias
    birlikte, yanlis sekil ya da ortonormal olmayan orbitaller REDDEDILIR."""
    mol = _mol("H 0 0 0; H 0 0 2.0")
    u = solve_uhf(mol, "sto-3g", initial_bias=_alternan(2, 0.35))
    assert u.stability is None and u.provenance.stability is None
    assert "Kararlılık" not in dict(u.provenance.as_lines())

    ca, cb = u.coefficients_alpha, u.coefficients_beta
    ayni = solve_uhf(mol, "sto-3g", initial_orbitals=(ca, cb))
    assert ayni.energy_total == pytest.approx(u.energy_total, abs=1e-12)
    with pytest.raises(InvalidInputError):
        solve_uhf(mol, "sto-3g", initial_orbitals=(ca, cb), initial_bias=np.zeros(2))
    with pytest.raises(InvalidInputError):
        solve_uhf(mol, "sto-3g", initial_orbitals=(ca[:, :1], cb))
    with pytest.raises(InvalidInputError):
        solve_uhf(mol, "sto-3g", initial_orbitals=(2.0 * ca, cb))


# --- T6 (hizli serit, olculdu 0.06 s) / T7 (yavas serit) ---------------------

RC_BOHR = 2.179694793840


def _h2(r_bohr: float) -> Molecule:
    yari = r_bohr * BOHR_TO_ANGSTROM / 2
    return Molecule.from_string(f"H 0 0 {-yari}; H 0 0 {yari}")


def test_T6_CATALLANMA_seeds1_IZLEMEYLE_cokmuyor_izlemesiz_HER_NOKTADA_cokuyor():
    """T6. Coulson-Fischer catallanmasi, bes noktali pencere (d <= 1.6e-2).

    OLCULDU: seeds = 1 + izleme (varsayilan) -> |dE| / spin / S^2 usleri
    1.996042 / 0.497009 / 0.995426, hicbir noktada |dE| < 1e-12 yok (her
    noktada 1 inis). Izlemesiz seeds = 1 BES NOKTANIN BESINDE RHF'e cokuyor
    ve hukum bes noktada da ✗. S^2 usunun 4. hanesi protokole bagli:
    search_uhf(12) 0.995332, burada 0.995426 (Rc yakininda lambda kucuk, SCF
    artigi S^2'yi ~4e-7 oynatiyor). Sabit bias 0.35'in IZLEMESIZ ussu 0.995360
    ama cokmeye DUYARSIZ (duzeltme olcumu f05): cokme d = 4e-3'te, log-izgaranin
    tam ORTASINDA, en kucuk kareler egiminde agirligi sifir -- cokmeyi us degil
    asagidaki |dE| >= 1e-12 denetimi yakalar. Pencere d >= 1e-3: Rc'ye
    d <= 6e-5 yakinlikta izleme de eyerde kalir (KOR_BOLGE testi).
    """
    d = np.array([1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2])
    farklar, spinler, s_kareler = [], [], []
    for adim in d:
        molecule = _h2(RC_BOHR + adim)
        rhf = solve(molecule, "sto-3g", stability=False).energy_total
        izlemesiz = search_uhf(molecule, "sto-3g", seeds=1, follow_instability=False)
        assert abs(rhf - izlemesiz.energy) < 1e-12 and not izlemesiz.stability.stable
        rapor = search_uhf(molecule, "sto-3g", seeds=1)
        assert rapor.stability.stable
        uhf = rapor.result
        farklar.append(rhf - uhf.energy_total)
        assert abs(farklar[-1]) >= 1e-12, f"d={adim}: izlemeye ragmen RHF'e coktu"
        overlap = overlap_matrix(build_basis(molecule, "sto-3g"))
        p_alpha = uhf.coefficients_alpha[:, :1] @ uhf.coefficients_alpha[:, :1].T
        p_beta = uhf.coefficients_beta[:, :1] @ uhf.coefficients_beta[:, :1].T
        spinler.append(float(np.abs(np.diag((p_alpha - p_beta) @ overlap)).max()))
        s_kareler.append(uhf.s_squared)

    def us(y) -> float:
        return float(np.polyfit(np.log(d), np.log(np.abs(np.array(y))), 1)[0])

    assert us(farklar) == pytest.approx(1.996042, abs=0.01)
    assert us(spinler) == pytest.approx(0.497009, abs=0.01)
    assert us(s_kareler) == pytest.approx(0.995426, abs=0.01)


@pytest.mark.slow
def test_T7_KARARLI_KURESEL_DEGILDIR_H8_halkasinda():
    """T7. UHF-kararlilik = YEREL minimum, KURESEL degil -- olculerek.

    H8 halkasi d = 3.0 bohr, STO-3G. Baslangiclar KAMU yolundan:
    solve_uhf(initial_orbitals = (S^-1/2 Q_a, S^-1/2 Q_b), stability=True),
    Q Haar-rastgele (QR, isaret duzeltilmis), rng 72026, ilk 10 baslangic,
    DIIS'li ve DIIS'siz.

    OLCULDU (20 baslangicta 40 SCF; 32 yakinsadi, 8 ayri duragan nokta, 3
    yerel minimum): kararli -3.8572719930 (kuresel), -3.7773306557 (+0.0799)
    ve -3.7762350559 (+0.0810). Izleme bes eyerin IKISINDEN kuresel OLMAYAN
    -3.7773306557'ye iniyor (-3.6757093681 ve -3.6604641385'ten). Ayni betik
    tek is parcacikli OpenBLAS'ta ve OPENBLAS_CORETYPE=Sandybridge ile de her
    baslangicta ayni durumu verdi (yalniz esit enerjililerin siralamasi farkli).
    search_uhf(>= 4) bugun kureseli buluyor -- ama bu bir garanti DEGIL.
    """
    mol = _halka(8, 3.0)
    s = overlap_matrix(build_basis(mol, "sto-3g"))
    w, v = np.linalg.eigh(s)
    c0 = v @ np.diag(w**-0.5) @ v.T
    rng = np.random.default_rng(72026)

    def haar() -> np.ndarray:
        q, r = np.linalg.qr(rng.normal(size=(8, 8)))
        return q * np.sign(np.diag(r))

    kararli, eyerler = set(), []
    for _ in range(10):
        ca, cb = c0 @ haar(), c0 @ haar()
        for diis in (True, False):
            try:
                u = solve_uhf(mol, "sto-3g", initial_orbitals=(ca, cb), use_diis=diis,
                              max_iterations=3000, stability=True)
            except SCFConvergenceError:  # bazi Haar baslangiclari yakinsamaz
                continue
            if u.stability.stable:
                kararli.add(round(u.energy_total, 8))
            else:
                eyerler.append(u)
    kuresel, yerel = -3.8572719930, -3.7773306557
    assert any(abs(e - kuresel) < 1e-8 for e in kararli)
    assert any(abs(e - yerel) < 1e-8 for e in kararli)
    inisler = []
    for eyer in eyerler:
        son, hukum, _ = _follow_instability(mol, "sto-3g", eyer, 2000)
        assert hukum.stable
        inisler.append(son.energy_total)
    assert any(abs(e - yerel) < 1e-8 for e in inisler), inisler
    assert search_uhf(mol, "sto-3g", seeds=4).energy == pytest.approx(kuresel, abs=1e-9)
