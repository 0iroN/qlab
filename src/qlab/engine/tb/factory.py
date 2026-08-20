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

from qlab.engine.tb.bands import band_structure, uniform_k_grid
from qlab.engine.tb.hamiltonian import Hopping, TBModel
from qlab.engine.tb.lattice import Lattice
from qlab.provenance import Provenance
from qlab.scenario.schema import TBScenario
from qlab.units import EV_TO_HARTREE

# k-izgarasi yakinsama esigi (hartree). Olcut: n ve 2n izgaralarinda bant
# KENARLARININ (spektrumun alt ve ust ucu) max farki. Esik olculmus degil
# SECILMISTIR; `Provenance.diagnostics` ham sapmayi da tasir, yani okuyucu
# kendi esigini uygulayabilir.
#
# OLCULMUS SINIR -- n vs 2n olcutu KANDIRILABILIR. Bu olcut "iki izgara
# ayni sonucu veriyor" der, "sonuc dogru" DEMEZ; ikisi de AYNI sistematik
# hatayi tasiyorsa fark sifir cikar. 1B zincirde tam bu oluyor: cift n'de
# sapma TAM 0.000000e+00, cunku hem n hem 2n cift ve ikisi de bant
# kenarlarini TAM yakaliyor -- yani sifir, yogunluktan degil PARITEDEN
# geliyor. Tek n'de olculdu: n=3 -> 6.247e-02, n=11 -> 5.061e-03,
# n=101 -> 6.044e-05 (hepsi esigi gecmiyor, ve HAKLI OLARAK gecmiyor:
# tek izgarada bant kenari n=101'de bile %0.024 eksik).
_K_YAKINSAMA_ESIGI = 1e-6

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
    # "SCF yok, True" yazmak alani anlamsizlastirirdi. Burada olcut,
    # k-izgarasi ikiye katlandiginda bant KENARLARININ ne kadar oynadigi.
    n = scenario.k_points
    kaba = band_structure(model, uniform_k_grid(orgu, n))
    ince = band_structure(model, uniform_k_grid(orgu, 2 * n))
    sapma = float(
        max(
            abs(float(kaba.min()) - float(ince.min())),
            abs(float(kaba.max()) - float(ince.max())),
        )
    )

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
        converged=bool(sapma < _K_YAKINSAMA_ESIGI),
        diagnostics={
            "k_convergence_hartree": sapma,
            "k_points": float(n),
            "n_electron_per_cell": float(scenario.n_electron_per_cell),
        },
    )
    return model, kaynak
