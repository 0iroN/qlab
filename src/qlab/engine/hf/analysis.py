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

Bu hukmun bir on kosulu vardir: molekul NOTR olmali. Yuklu bir sistemde
dipol momenti koordinat orijinine baglidir (bkz. dipole_moment
dokstringi), yani iyonik karakterin olcusu olarak kullanilamaz. analyse()
charge != 0 gordugunde bunu provenance'ta ACIKCA beyan eder; bu on kosul
bir zamanlar yalnizca "bu projede tum senaryolar notrdur" varsayimiyla
tutuluyordu, artik tutmuyor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.hf.basis import BasisFunction, build_basis
from qlab.engine.hf.integrals import (
    core_hamiltonian,
    dipole_matrices,
    eri_tensor,
    overlap_matrix,
)
from qlab.engine.hf.scf import HFResult
from qlab.engine.hf.stability import (
    StabilityResult,
    analyse_stability,
    orbital_hessian,
)
from qlab.provenance import Provenance
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
    -- Molecule.atoms konumlariyla ayni cerceve.

    ORIGIN BAGIMLILIGI (olgu, kod hatasi degil): sistem a kadar otelenirse
    d -> d + q*a olur, q = molekulun net yuku. Molekul NOTR ise (q = 0) bu
    terim yok olur ve sonuc origin secimine duyarsizdir. Yuklu bir sistemde
    ise bildirilen sayi molekulun uzayda NEREDE durduguna baglidir ve tek
    basina molekulun bir ozelligi degildir. Olculdu (LiH/STO-3G, molekul
    +10 A z'de otelendi): notr 4.858750745324352 -> 4.8587507453240795 D
    (fark 2.7e-13, kayan nokta gurultusu); charge=2 icin 7.667440123884034
    -> 103.73153436795428 D. Yuklu durumdaki kayma tam olarak q*a'dir
    (olculdu, goreli sapma 5.6e-16). Bkz.
    test_dipol_OTELEME_altinda_yalnizca_NOTR_sistemde_degismez.

    Burada origin SESSIZCE kaydirilmaz (ornegin yuk merkezine): o secim de
    bir yaklasim olurdu ve beyan edilmesi gerekirdi. Bunun yerine sayi
    oldugu gibi verilir ve analyse() charge != 0 durumunda ne oldugunu
    provenance'a yazar.

    Bu, iyonik karakterin bu projedeki BIRINCIL olcusudur (bkz. spec 6.3 ve
    mulliken_charges dokstringi) -- Mulliken/Lowdin yuklerinin aksine baz
    setine asiri duyarli degildir ve isaret/yon konusunda guvenilirdir.
    Ancak bu hukum NOTR sistemler icindir; yukaridaki origin bagimliligi
    yuzunden yuklu bir sistemde dipol iyonik karakterin olcusu olamaz.
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


def stability(result: HFResult) -> StabilityResult:
    """Ic (RHF -> RHF) kararlilik analizi, HFResult uzerinden.

    solve() bunu varsayilan olarak zaten kosar ve sonucu HFResult.stability
    icinde tasir; bu fonksiyon solve(stability=False) ile kosulmus (veya
    baska bir yoldan elde edilmis) bir sonuc icin SONRADAN hesaplama
    yoludur. HFResult integralleri saklamadigi icin (yalnizca molecule +
    basis_name adi) baz burada yeniden kurulur -- analysis.py'nin geri
    kalaniyla ayni desen.

    Yontem ve sinirlar icin stability.py modul dokstringine bakin: yalnizca
    IC (tekil, reel) kararlilik sinanir, dis (RHF->UHF) kararsizlik degil.
    """
    basis = build_basis(result.molecule, result.basis_name)
    core = core_hamiltonian(basis, result.molecule)
    eri = eri_tensor(basis)
    n_occupied = result.molecule.n_electrons // 2
    return analyse_stability(
        orbital_hessian(core, eri, result.coefficients, n_occupied)
    )


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


# ---------------------------------------------------------------------------
# Tek Provenance tasiyan rapor
#
# Yukaridaki yedi public fonksiyon ciplak np.ndarray/float dondurur, yani
# projenin "her sonuc Provenance tasimak zorundadir" ilkesini (spec bolum 3,
# Ilke 2) tek baslarina karsilamazlar. Yedi imzayi ayri ayri degistirmek
# yerine tek rapor toplanir: tuketici (panel, CLI) zaten hepsini birlikte
# ister ve yedi ayri Provenance gereksiz tekrardir. Fonksiyonlar dogrudan
# kullanim icin public KALIR.
# ---------------------------------------------------------------------------

_ANALYSIS_APPROXIMATIONS: tuple[str, ...] = (
    "Mulliken/Lowdin yuk bolusturmesi: paylasilan yogunluk atomlara KOR "
    "bicimde bolunur, sonuc baz setine asiri duyarlidir -- LiH/STO-3G'de "
    "ISARET bile ters cikar (Li'ye -0.016028455; PySCF ayni sayiyi verir, "
    "bkz. test_lih_mulliken_sign_is_a_documented_basis_artifact)",
    "iyonik karakter BIRINCIL olarak dipol momentiyle olculur, yuklerle "
    "DEGIL (bkz. test_lih_dipole_shows_correct_ionic_direction ve "
    "test_ionic_character_ordering: H2 < HF < LiH) -- ama bu hukmun on "
    "kosulu molekulun NOTR olmasidir: yuklu bir sistemde dipol koordinat "
    "orijininin secimine baglidir ve iyonik karakterin olcusu OLAMAZ "
    "(o durumda expected_error ayrica uyarir)",
    "HOMO/LUMO kanonik RHF orbital enerjileridir (Koopmans duzeyi)",
)

_ANALYSIS_NEGLECTED: tuple[str, ...] = (
    "Koopmans'ta orbital gevsemesi ve elektron korelasyonu: iyonlasma "
    "enerjisi -eps_HOMO'ya esit ALINAMAZ, iki ihmal ters yonde calisir ve "
    "burada olculmemistir",
    "Mayer bag mertebesi RHF'de ayrismayi GOSTERMEZ: H2'nin bag mertebesi "
    "her mesafede tam 1.0 kalir (olculdu 0.74/1.00/2.00/3.00 A, bkz. "
    "test_rhf_bond_order_of_h2_stays_one_at_any_separation)",
)

_ANALYSIS_EXPECTED_ERROR = (
    "Bu raporun en buyuk hatasi bir sayida degil YORUMDA'dir: Mulliken ve "
    "Lowdin yukleri baz setine asiri duyarlidir ve LiH/STO-3G'de ISARET "
    "olarak yanilirlar (Li: -0.016028455, yani kimyasal sezginin tersi). "
    "Iyonik karakter bu projede BIRINCIL olarak dipol momentiyle okunur. "
    "Mayer bag mertebesi RHF'de bag KIRILISINI gostermez ve HOMO/LUMO "
    "Koopmans duzeyinde kalir."
)

# Bos orbital yoksa LUMO diye bir sey yoktur. gap=0.0 yazmak "bosluk yok,
# metalik" diye okunurdu; ayni karar StabilityVerdict.lowest_eigenvalue'da
# (n_rotations == 0) zaten verilmis durumda.
_NO_LUMO_NOTE = (
    "BU SISTEMDE LUMO YOK: butun orbitaller isgal edilmis (n_occupied == "
    "n_orbitals; olculdu: He/STO-3G ve He2/STO-3G), bu yuzden lumo ve gap "
    "None birakildi -- gap=0.0 yazmak 'bosluk yok, metalik' diye okunurdu."
)

# Simetrigi: isgal edilmis orbital yoksa HOMO da yoktur. Bu durumda
# homo_lumo() patlamaz, SESSIZCE yanlis doner (n_occupied-1 = -1 son orbitale
# sarar) -- olculdu, Li/STO-3G charge=3: gap -3.5303736858323314, yani
# negatif bir "bosluk".
_NO_HOMO_NOTE = (
    "BU SISTEMDE ISGAL EDILMIS ORBITAL YOK (n_occupied == 0): HOMO diye bir "
    "sey yoktur, homo ve gap None birakildi."
)

# Yuklu sistemde dipol orijine baglidir. Bu bir kod hatasi degil bir olgudur;
# kusur onun SESSIZ kalmasi olurdu. dipole_moment()'in dokstringi bir zamanlar
# "bu projede tum senaryolar norotr" diyordu -- charge artik raporun birinci
# sinif niceligi (diagnostics["expected_charge"]) oldugu icin o on kosul
# gecerli degil, ve bu not onun yerini alir.
_CHARGED_DIPOLE_NOTE = (
    "BU SISTEM YUKLU (charge != 0) ve dipol momenti bu durumda koordinat "
    "ORIJININE BAGLIDIR: sistem a kadar otelenirse d -> d + q*a olur, yani "
    "bildirilen sayi molekulun uzayda nerede durduguna baglidir ve tek basina "
    "molekulun bir ozelligi DEGILDIR. Kullanilan orijin, molekulun kendi "
    "koordinat cercevesinin (0,0,0) noktasidir; baska bir orijin (yuk merkezi, "
    "kutle merkezi) baska bir sayi verirdi ve o secim de beyan edilmesi gereken "
    "bir yaklasim olurdu -- bu yuzden sessizce yapilmadi. Olculdu (LiH/STO-3G, "
    "charge=2, molekul +10 A otelendi): 7.667440123884034 D -> "
    "103.73153436795428 D; ayni oteleme notr LiH'te sayiyi degistirmez. Sonuc: "
    "iyonik karakteri dipolden okuma hukmu BU SISTEMDE GECERSIZDIR, bkz. "
    "test_dipol_OTELEME_altinda_yalnizca_NOTR_sistemde_degismez."
)


@dataclass(frozen=True)
class AnalysisReport:
    """Yakinsamis bir RHF sonucundan cikarilan kimyasal buyukluklerin tamami.

    Alanlar yukaridaki fonksiyonlarin ciktilarini BIREBIR tasir; bu sinif
    hicbir seyi yeniden hesaplamaz (bkz. analyse()).

    frozen=True alanlarin yeniden BAGLANMASINI engeller; dizi alanlari
    (mulliken, lowdin, dipole_au, bond_orders) ayrica SALT-OKUNUR isaretlenir
    (bkz. _read_only), cunku frozen tek basina dizi icerigini korumaz ve
    korumasiz haliyle bir tuketici rapor.mulliken[0] = -99.0 yazip "olculmus"
    gorunumlu bir sayi enjekte edebiliyordu.

    dipole_au/dipole_debye yuklu bir sistemde koordinat orijinine baglidir;
    provenance bunu beyan eder (diagnostics["dipole_origin_dependent"] ve
    expected_error). Ayrinti icin dipole_moment dokstringine bakin.

    homo/lumo/gap `float | None`'dir. None, "olculemedi" demektir ve sebebi
    provenance.expected_error'da yazilidir: bos orbitali olmayan bir sistemde
    LUMO, isgal edilmis orbitali olmayan bir sistemde HOMO yoktur. Bu alanlara
    0.0 yazmak "bosluk yok, metalik" diye okunurdu -- ayni gerekce
    StabilityVerdict.lowest_eigenvalue'da da yazili.
    """

    mulliken: np.ndarray
    lowdin: np.ndarray
    dipole_au: np.ndarray
    dipole_debye: float
    homo: float | None
    lumo: float | None
    gap: float | None
    bond_orders: np.ndarray
    provenance: Provenance


def _merged(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Birden cok kaynagin beyanlarini sirayi bozmadan, tekrarsiz birlestirir.

    Ayni desen observable/hf_obs.py:_merged_declarations'ta da var (fark
    haritasi iki alanin beyanini birden tasir). Katman kurali geregi (engine,
    observable'i import EDEMEZ) o yardimci buraya cagrilamaz; uc satirlik
    kopya, katmanlarin ters yone bagimli olmasindan iyidir.
    """
    return tuple(dict.fromkeys(item for group in groups for item in group))


def _read_only(array: np.ndarray) -> np.ndarray:
    """Diziyi salt-okunur isaretler ve AYNI diziyi geri dondurur.

    Gerekce: dataclass(frozen=True) yalnizca alan YENIDEN BAGLAMAYI engeller
    (rapor.gap = 0.0 -> FrozenInstanceError), dizi ICERIGINI korumaz. Olculdu
    (bu koruma eklenmeden once, su/STO-3G): rapor.mulliken[0] = -99.0 sorunsuz
    calisiyordu ve mulliken [-0.36550977 0.18275488 0.18275488]'den
    [-99.0 0.18275488 0.18275488]'e donuyordu; lowdin, dipole_au ve bond_orders
    de ayni sekilde acikti. test_analiz_raporu_dondurulmus'un dokstringi
    tuketicinin "yerinde degistirip olculmus gorunumlu bir sayi enjekte
    edememesini" vaat ediyor -- bu, o vaadin tutuldugu yer.

    Bayragi burada degistirmek guvenlidir: mulliken_charges/lowdin_charges/
    dipole_moment/mayer_bond_orders her cagrida YENI bir dizi uretir (hicbiri
    HFResult'in ic dizilerine view dondurmez), yani salt-okunurluk yalnizca
    raporun kendi kopyasini baglar. Tuketici degistirmek isterse .copy() alir.
    """
    array.setflags(write=False)
    return array


def analyse(result: HFResult) -> AnalysisReport:
    """Yedi analiz fonksiyonunu bir kez toplar ve TEK bir Provenance kurar.

    Kendisi fizik HESAPLAMAZ: her alan yukaridaki fonksiyonlarin ciktisidir.
    Tek istisna, homo/lumo/gap'in "yok" durumlaridir -- homo_lumo() bos
    orbitali olmayan sistemde ham IndexError firlatir (olculdu, He/STO-3G:
    "index 1 is out of bounds for axis 0 with size 1") ve isgal edilmis
    orbitali olmayan sistemde sessizce negatif bir bosluk dondurur (olculdu,
    Li/STO-3G charge=3: gap = -3.5303736858323314). Public bir API'nin ham
    bir IndexError ile patlamasi da, negatif bir boslugu olculmus gibi
    sunmasi da kabul edilemez; iki durumda da ilgili alanlar None kalir ve
    provenance bunu ACIKCA beyan eder. homo_lumo()'nun KENDI davranisi
    degistirilmemistir -- o ayri, onceden var olan bir kusurdur.

    NE TASINIR: rapor kendi beyanlarini (Mulliken'in baz duyarliligi,
    Koopmans, Mayer'in ayrisma korlugu) dalga fonksiyonunun beyanlarinin
    USTUNE ekler; SCF'in yaklasimlari, ihmalleri, diagnostics'i, `converged`
    ve `stability` alanlari AYNEN tasinir. Gerekce: analiz, analiz ettigi
    dalga fonksiyonu kadar guvenilirdir. SCF bir eyer noktasina oturduysa
    rapordaki her Mulliken yuku ve her bag mertebesi bir eyer noktasi dalga
    fonksiyonundan hesaplanmistir; provenance'i sifirdan kurup stability'yi
    None birakmak panelin kararlilik satirini hic basmamasina, yani okuyucunun
    bunu "sorun yok" diye okumasina yol acardi.

    YUKLU SISTEMLER: charge != 0 oldugunda dipol momenti koordinat orijinine
    baglidir (olgu, bkz. dipole_moment dokstringi). analyse() bu durumda
    _CHARGED_DIPOLE_NOTE'u expected_error'a ekler ve
    diagnostics["dipole_origin_dependent"] bayragini kaldirir. Orijin sessizce
    kaydirilmaz: bir orijin secmek de bir yaklasimdir ve beyan edilmesi
    gerekirdi. Notr sistemde bu not EKLENMEZ -- kosulsuz eklenmis bir uyari
    dogru sayiya duyulan guveni bosa harcatirdi.

    stability(result) BILEREK cagrilmaz: kararlilik hukmu dalga
    fonksiyonunun ozelligidir ve SCF onu ya olcmustur ya olcmemistir
    (solve(stability=False)). Burada yeniden hesaplamak, kaynagin OLCMEDIGI
    bir hukmu rapora yazmak olurdu.

    Maliyet notu: dipole_debye() ic dipole_moment()'i yeniden kurar, yani
    dipol integralleri iki kez hesaplanir. Birim donusumunu burada tekrar
    etmemek icin bilincli tercih -- au -> Debye donusumu YALNIZCA
    dipole_debye()'de yasar (bkz. modul dokstringi). Ayni tur tekrar
    modulde zaten var: _basis_and_atom_index uc fonksiyon tarafindan ayri
    ayri cagrilir.
    """
    mulliken = _read_only(mulliken_charges(result))
    lowdin = _read_only(lowdin_charges(result))
    dipole_au = _read_only(dipole_moment(result))
    debye = dipole_debye(result)
    bond_orders = _read_only(mayer_bond_orders(result))

    n_occupied = result.molecule.n_electrons // 2
    n_orbitals = int(result.orbital_energies.shape[0])
    has_homo = n_occupied >= 1
    has_lumo = n_occupied < n_orbitals

    if has_homo and has_lumo:
        homo, lumo, gap = homo_lumo(result)
    else:
        # homo_lumo() bu iki durumda cagrilamaz (biri patlar, digeri sessizce
        # yanlis doner), bu yuzden indeks konvansiyonu -- isgal edilmis en
        # yuksek orbital n_occupied-1, en dusuk bos orbital n_occupied --
        # burada TEK SEFER tekrar edilir. Konvansiyonun dogrulugu
        # test_lumo_yoksa_homo_hala_EN_YUKSEK_isgal_edilmis_orbital ile
        # baglanmistir (He2/STO-3G: iki isgal orbitali, LUMO yok, enerjileri
        # farkli -- yanlis indeks orada gorunur).
        homo = float(result.orbital_energies[n_occupied - 1]) if has_homo else None
        lumo = float(result.orbital_energies[n_occupied]) if has_lumo else None
        gap = None

    charged = bool(result.molecule.charge != 0)

    expected_error = [_ANALYSIS_EXPECTED_ERROR]
    if charged:
        expected_error.append(_CHARGED_DIPOLE_NOTE)
    if not has_lumo:
        expected_error.append(_NO_LUMO_NOTE)
    if not has_homo:
        expected_error.append(_NO_HOMO_NOTE)
    expected_error.append(
        "Bunlarin uzerine dalga fonksiyonunun kendi hatasi biner: "
        + result.provenance.expected_error
    )

    provenance = Provenance(
        method=(
            f"RHF/{result.basis_name.upper()} dalga fonksiyonu analizi: "
            "Mulliken/Lowdin yukleri, dipol, HOMO/LUMO, Mayer bag mertebeleri"
        ),
        approximations=_merged(
            result.provenance.approximations, _ANALYSIS_APPROXIMATIONS
        ),
        neglected=_merged(result.provenance.neglected, _ANALYSIS_NEGLECTED),
        expected_error=" ".join(expected_error),
        converged=result.provenance.converged,
        diagnostics={
            **result.provenance.diagnostics,
            "n_occupied": n_occupied,
            "n_orbitals": n_orbitals,
            # Mulliken yuklerinin toplami molekulun yukune esit olmak
            # zorundadir (bolusturme yalnizca dagitir, uretmez/yok etmez);
            # olculen/beklenen cifti panelde yan yana durur.
            "mulliken_charge_sum": float(np.sum(mulliken)),
            "expected_charge": result.molecule.charge,
            # Yuklu sistemde dipol koordinat orijinine baglidir (bkz.
            # _CHARGED_DIPOLE_NOTE ve dipole_moment dokstringi). Metin
            # beyaninin makine-okunur esi: panel/CLI dipol satirini bu
            # bayraga bakarak isaretleyebilir.
            "dipole_origin_dependent": charged,
        },
        stability=result.provenance.stability,
    )

    return AnalysisReport(
        mulliken=mulliken,
        lowdin=lowdin,
        dipole_au=dipole_au,
        dipole_debye=debye,
        homo=homo,
        lumo=lumo,
        gap=gap,
        bond_orders=bond_orders,
        provenance=provenance,
    )
