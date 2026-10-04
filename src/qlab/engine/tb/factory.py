"""Senaryo -> TBModel. BIRIM SINIRI ve DAIRESELLIK BEYANI burada.

Bu dosyanin en onemli isi teknik degil ETIKTIR. `_YAKLASIMLAR` metni,
kullanicinin ekraninda "bu modulun urettigi bant araligi bir tahmin
degildir" diye yazan yerdir. Olculdu (Vogl sp3s*, silisyum): E_gap =
1.17 eV elde etmek icin gereken tek parametre V(s*a,pc) = 5.376670 eV;
yayinlanan deger 5.3749, fark %0.033; kazanc -0.7561 eV/eV; AYNI kadran
Ge'yi (0.66 eV) ve V = 7.0'da METALI veriyor.

BIRIM SINIRI. TB literaturu eV'de yayinlanir, depo ici hartree'dir.
Donusum tam BURADA yapilir -- tipki `md/factory.py`nin girdide
`amu_to_electron_mass` cagirmasi gibi. `units.py`ye yeni sabit EKLENMEDI;
`EV_TO_HARTREE` zaten oradaydi ve `src/` icinde ilk kez burada kullaniliyor.
"""

from __future__ import annotations

import numpy as np

from qlab.engine.tb.bands import grid_extremum_bound
from qlab.engine.tb.downfold import (
    STAGGERING_TOLERANCE,
    DownfoldedParameters,
    downfold_uhf,
    hydrogen_ring,
)
from qlab.engine.tb.hamiltonian import Hopping, TBModel
from qlab.engine.tb.lattice import Lattice
from qlab.provenance import Provenance
from qlab.scenario.schema import DownfoldScenario, TBScenario
from qlab.units import EV_TO_HARTREE

# k-izgarasi yakinsama esigi (hartree), Osman'in karari (04.10.2026): 1e-4
# hartree = 2.7 meV. Olcut OLCUM DEGIL KANITTIR: B2 (`grid_extremum_bound`)
# izgaranin spektrum uclarini en cok ne kadar kacirabilecegini hoplama
# listesinden hesaplar; `converged` = B2 <= esik.
#
# ESKI OLCUT (n ile 2n izgarasinin uc farki) n'nin HATASINI degil n ile 2n
# arasindaki IYILESMEYI olcuyordu: ikisi gercek uca en yakin AYNI noktayi
# paylasinca fark TAM 0 cikar, hata sifir olmasa bile. Olculdu (dogrulama
# turu): ucu kesirli 0.26'da olan ikinci komsulu zincir (t2 = 0.05, t1 =
# -4 t2 cos(2 pi 0.26)) n=4 ve n=12'de eski ✓ veriyordu, gercek uc hatasi
# 7.885e-04 hartree (eski esigin 788 kati); B2(4) = 1.31e-01 (✗). Zincir
# n=4'teki ve grafendeki eski ✓ ise DOGRUYDU (uclar izgaradaydi) -- ama
# kanit degildi. B2 a priori'dir:
# sevk edilen zincirde n=4 -> 3.854e-02 (✗), n=78 -> 1.013e-04 (✗),
# n=79 -> 9.880e-05 (✓), n=100 -> 6.166e-05 (✓). 1e-6 secilseydi zincir
# n >= 786 isterdi ve 2B/3B'de hicbir makul izgara ✓ almazdi. Esik bir
# GARANTI oldugu icin gercek hata genelde cok daha kucuktur; TB
# parametrelerinin kendi belirsizligi eV mertebesindedir.
#
# Bu olcut ILETKENLIK hukmunu sertifikalamaz -- o ic kenarlara (VBM/CBM)
# bakar ve `certified_conduction` (B1) ile kendi satirinda basilir.
_K_YAKINSAMA_ESIGI = 1e-4

_YAKLASIMLAR: tuple[str, ...] = (
    "tight-binding: sonlu menzilli, ortogonal (S = I) tek-parcacik modeli",
    "parametreler DISARIDAN verilir; bu modul onlari turetmez",
    "BANT ARALIGI ve BANT GENISLIGI DEGERLERI DOGRULAMA DEGILDIR -- girilen "
    "fit parametresinin geri okunmasidir (olculdu: silisyumda 1.17 eV icin "
    "gereken tek parametre 5.376670 eV, ayni kadran Ge'yi ve metali de "
    "veriyor)",
    "ILETKEN hukmu elektron SAYIMINDAN gelir ve parametrelerden bagimsizdir; "
    "YALITKAN hukmu gelmez -- cift sayim yalitkanligi garanti etmez "
    "(olculdu: 1B %67, 2B %34, 3B %20 metal cikiyor)",
    "iletkenlik hukmu izgaradan KANITLIDIR: metal yalniz izgarada tanik "
    "varken (tek sayimda da: yari dolu bant izgarada duz gorunurse "
    "'belirsiz'), yalitkan yalniz izgara araligi 2*B1'i asarken ya da butun "
    "bantlar doluyken ya da H(k) k'dan bagimsizken, aksi halde 'belirsiz' "
    "(B1 = (pi/n) max_i sum|t| ||cell||_1, Weyl siniri; dokunan bantlar -- "
    "yari metal -- hicbir n'de metal kanitlanamaz)",
    "cekirdekler sabit: fonon yok, Peierls carpilmasinin enerjetigi yok",
)

_IHMAL_EDILEN: tuple[str, ...] = (
    "ortusme matrisi: ortogonal taban varsayildi (S = I). Olculdu, ihmalin "
    "bedeli bant genisliginin %45-65'i olabiliyor",
    "spin, spin-yorunge etkilesimi, manyetik alan (Peierls fazi)",
    "elektron-elektron etkilesimi -- bu bir ortalama-alan modeli BILE degil",
    "sicaklik: doluluk tam sayi elektron sayimidir, Fermi-Dirac yoktur",
)


def build_tb(scenario: TBScenario) -> tuple[TBModel, Provenance]:
    """Senaryodan model kurar; eV -> hartree donusumu BURADA yapilir."""
    orgu = Lattice(
        vectors=np.array(scenario.lattice_vectors_bohr, dtype=float),
        basis=np.array(scenario.basis_bohr, dtype=float),
    )
    model = TBModel(
        lattice=orgu,
        onsite=np.array(scenario.onsite_ev, dtype=float) * EV_TO_HARTREE,
        hoppings=tuple(
            Hopping(i=h.i, j=h.j, cell=tuple(h.cell), t=h.t_ev * EV_TO_HARTREE)
            for h in scenario.hoppings
        ),
    )

    # `converged` OLCULEBILIR bir seye baglanir. Olculmus tuzak: `eigvalsh`
    # dort patolojik girdide de istisna atmadan sonlu deger dondurdu, yani
    # "SCF yok, True" yazmak alani anlamsizlastirirdi. Burada olcut, bant
    # KENARLARININ izgarada kacirilabilecegi en buyuk miktarin KANITLI
    # siniridir (B2) -- kosegenlestirme gerektirmez (eskiden n ve 2n
    # izgaralari hesaplaniyordu: grafen k=100'de ~1.3 s).
    n = scenario.k_points
    sinir = grid_extremum_bound(model, n)

    kaynak = Provenance(
        method=f"tight-binding ({scenario.name})",
        approximations=_YAKLASIMLAR,
        neglected=_IHMAL_EDILEN,
        expected_error=(
            "parametrelerin kendi belirsizligi kadar; bu modul parametreleri "
            "turetmedigi icin MUTLAK bir hata siniri VEREMEZ. Yapisal "
            "dogruluk (gercek-uzay <-> Bloch capraz kontrolu) hata/||H|| "
            "olcusunde 1e-13 altindadir."
        ),
        converged=bool(sinir <= _K_YAKINSAMA_ESIGI),
        diagnostics={
            "k_edge_bound_hartree": sinir,
            "k_points": float(n),
            "n_electron_per_cell": float(scenario.n_electron_per_cell),
        },
    )
    return model, kaynak


# --------------------------------------------------------------------------
# MODUL 3H -- UHF INDIRGEMESI. Panel beyani TEK KAYNAKTAN, yani BURADAN gelir.
#
# `run.py::_run_downfold` bu metinlerin hicbirini yeniden yazmaz;
# `Provenance.as_lines()` ciktisini oldugu gibi basar. Gerekcesi bu depoda
# OLCULEREK ogrenildi: ayni beyan blogu bir kez `scf.py` ve `factory.py`de
# birebir KOPYALANMISTI ve biri duzeltilip digeri unutuldugunda panel iki
# farkli sey iddia ediyordu (bkz. tests/hf/test_hf_scenario.py::
# test_scan_and_single_point_share_one_provenance_source). Buradaki kopya
# sayisi testle kilitlidir: her cumle `src/` agacinda TAM BIR KEZ gecer.
# --------------------------------------------------------------------------

_INDIRGEME_YAKLASIMLAR: tuple[str, ...] = (
    "UHF (tek determinant, spin ozdurumu DEGIL)",
    "sonlu baz seti (STO-3G)",
    "iki-siteli antiferromanyetik hucre",
    "Lowdin ortogonallestirmesi",
    "MODELIN SEKLI VARSAYIM DEGIL OLCUMDUR: alternasyon, iki-siteli hucre ve "
    "hoplamanin spinden bagimsizligi her kosumda YENIDEN olculur "
    "(kaydirma artigi asagida raporlanir)",
    "BU MODUL HIZLANMA IDDIA ETMEZ: HF/TB maliyet orani OLCULMEDI. Iddia "
    "edilen tek sey parametrelerin TURETILEBILIR ve CAPALANABILIR oldugudur",
)

_INDIRGEME_IHMAL: tuple[str, ...] = (
    "spin projeksiyonu (PUHF) yok -- <S^2> raporlanir, DUZELTILMEZ",
    "kenar etkileri: yalnizca HALKA; acik zincir kapsam DISINDADIR",
    "6-31G ve otesi -- atom basina iki baz fonksiyonu tasir, site-bazli "
    "okuma orada TANIMSIZDIR",
    "t3 ve otesi hoplamalar (menzil kesmesi)",
    "elektron korelasyonu -- UHF bir ortalama-alan yontemidir",
)

# Kesme hatasinin OLCULMUS beyani. Sayilar tests/tb/test_downfold_kesme.py'de
# kilitlidir ve oradan gelir; burada YENIDEN OLCULMEZ, ALINTILANIR.
_INDIRGEME_BEKLENEN_HATA = (
    "iki-siteli model UHF alfa tayfini N=8 d=2.0 bohr'da 0.066287 hartree "
    "kaciriyor (yayilimin, 1.693171, %3.9'u); t2 olmadan 0.153603 (%9.1). "
    "Alti sistemde olculen aralik 0.003925 ... 0.153603 hartree ve t2 hatayi "
    "IKI ILA UC KAT dusuruyor (2.25 ... 3.26). Hata menzille SIFIRLANMAZ: "
    "tam menzilde bile 0.063140 kaliyor, cunku tavani menzil degil KABUK "
    "ORTALAMASI belirler."
)


def build_downfold(
    scenario: DownfoldScenario,
) -> tuple[tuple[DownfoldedParameters, ...], Provenance]:
    """Senaryodaki her mesafede UHF indirgemesini kosar.

    BIRIM SINIRI YOKTUR ve bu bir eksiklik degil bir OLCUM sonucudur:
    `TBScenario` eV'den hartree'ye cevirir cunku TB literaturu eV'de
    yayinlanir; burada girdi bir GEOMETRIDIR ve geometrinin atomik birimi
    zaten bohr'dur. Cikti (eps_bar, Delta, t_m) hartree'dir.

    `converged` OLCULEBILIR bir seye baglanir ve o sey KADEMELI YAPININ
    DOGRULANMASIDIR: her mesafede |diag(H^a) - roll(diag(H^b), 1)| olculur
    ve hepsi `STAGGERING_TOLERANCE`in altinda kalmalidir. Yani panel
    "yakinsadi" derken "SCF durdu" degil, "modelin varsaydigi iki-siteli
    antiferromanyetik yapi GERCEKTEN bulundu" demis olur.
    """
    parametreler = tuple(
        downfold_uhf(
            hydrogen_ring(scenario.ring_size, d),
            scenario.basis,
            n_neighbour=scenario.n_neighbour,
            seeds=scenario.seeds,
        )
        for d in scenario.distances_bohr
    )

    artik_max = max(p.staggering_residual for p in parametreler)
    s2_max = max(abs(p.s_squared) for p in parametreler)

    # ARAMA SATIRI. Her mesafe AYRI bir arama demektir. Hepsi ayni ozeti
    # verirse tek satir yazilir; vermezse HEPSI mesafesiyle birlikte yazilir.
    # "Ilkini yaz" secenegi REDDEDILDI: bir kosumun ozeti otekilerin adina
    # konusamaz, ve bu modulun butun sayilari bir ARAMA protokolunden gelir.
    satirlar = tuple(p.search_line for p in parametreler)
    if len(set(satirlar)) == 1:
        arama = f"{len(parametreler)} mesafenin hepsinde: {satirlar[0]}"
    else:
        arama = "; ".join(
            f"d={d:.3f}: {s}"
            for d, s in zip(scenario.distances_bohr, satirlar)
        )

    kaynak = Provenance(
        method=(
            f"UHF indirgemesi ({scenario.name}); N={scenario.ring_size} halka, "
            f"{len(parametreler)} mesafe, baz {scenario.basis}"
        ),
        approximations=_INDIRGEME_YAKLASIMLAR
        + (f"menzil {scenario.n_neighbour} kesmesi",),
        neglected=_INDIRGEME_IHMAL,
        expected_error=_INDIRGEME_BEKLENEN_HATA,
        converged=bool(artik_max < STAGGERING_TOLERANCE),
        diagnostics={
            "staggering_residual_max": artik_max,
            "s_squared_max": s2_max,
            "n_distance": float(len(parametreler)),
            "n_neighbour": float(scenario.n_neighbour),
            "arama": arama,
        },
    )
    return parametreler, kaynak
