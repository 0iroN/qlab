"""Senaryo -> TB modeli, birim sinirinda donusum ve DAIRESELLIK BEYANI.

Bu dosyanin en onemli testi bir sayi degil bir CUMLE denetler: seffaflik
panelinde "bant araligi degeri fit parametresinin geri okunmasidir" yaziyor
mu? Olculdu, yazmazsa modul kendi olctugumuz hatayi yeniden uretir.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from qlab.engine.tb.factory import build_tb
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.units import EV_TO_HARTREE

ZINCIR_YAML = """
kind: tb
name: tek boyutlu zincir
lattice_vectors_bohr: [[2.5, 0.0, 0.0]]
basis_bohr: [[0.0, 0.0, 0.0]]
onsite_ev: [-5.3]
hoppings:
  - {i: 0, j: 0, cell: [1, 0, 0], t_ev: -1.7}
  - {i: 0, j: 0, cell: [-1, 0, 0], t_ev: -1.7}
n_electron_per_cell: 1
k_points: 100
"""


def _yaz(tmp_path, metin: str):
    yol = tmp_path / "tb.yaml"
    yol.write_text(metin, encoding="utf-8")
    return yol


def test_senaryo_yukleniyor_ve_BIRIM_DONUSUYOR(tmp_path):
    """eV -> hartree donusumu SENARYO SINIRINDA yapilir, motorda degil.

    OLCULDU: -5.3 eV -> -0.19477140753097144 hartree,
    -1.7 eV -> -0.062473847698613486 hartree.
    """
    senaryo = load_scenario(_yaz(tmp_path, ZINCIR_YAML))
    model, _ = build_tb(senaryo)
    assert model.onsite[0] == pytest.approx(-5.3 * EV_TO_HARTREE, abs=1e-15)
    assert model.hoppings[0].t == pytest.approx(-1.7 * EV_TO_HARTREE, abs=1e-15)
    assert model.onsite[0] == pytest.approx(-0.19477140753097144, abs=1e-15)


def test_SEFFAFLIK_daireselligi_BEYAN_EDIYOR(tmp_path):
    """Bu modulun en onemli testi.

    Provenance metni, bant araligi/genisligi DEGERLERININ dogrulama
    olmadigini ACIKCA soylemeli. Olculdu: silisyumda 1.17 eV elde etmek
    icin gereken tek parametre 5.376670 eV ve ayni kadran Ge'yi de metali
    de veriyor -- yani deger bir tahmin degil, geri okuma.
    """
    senaryo = load_scenario(_yaz(tmp_path, ZINCIR_YAML))
    _, kaynak = build_tb(senaryo)
    metin = " ".join(deger for _, deger in kaynak.as_lines())

    # ANAHTAR KELIME DEGIL, TAM CUMLE aranir. Olculdu: kelime bazli bir
    # test, bitisik dize sabitlerinin yalnizca ILK satirini degistiren
    # mutantlari KACIRIYOR -- "fit", "geri okun" ve "5.376670" sonraki
    # satirlarda kaldigi icin beyan sacmalasirken test yesil kaliyordu.
    # Uc beyan da bu modulun etik cekirdegidir ve sessizce yeniden
    # yazilmamalidir.
    for cumle in (
        "BANT ARALIGI ve BANT GENISLIGI DEGERLERI DOGRULAMA DEGILDIR",
        "fit parametresinin geri okunmasidir",
        "5.376670",
        "ILETKEN hukmu elektron SAYIMINDAN gelir",
        "cift sayim yalitkanligi garanti etmez",
        "ortusme matrisi: ortogonal taban varsayildi (S = I)",
        "spin, spin-yorunge etkilesimi",
    ):
        assert cumle in metin, f"beyan kayip: {cumle!r}"


def test_converged_K_IZGARASI_YAKINSAMASINA_bagli(tmp_path):
    """OLCULMUS TUZAK: `eigvalsh` dort patolojik girdide de istisna atmadan
    sonlu ozdeger dondurdu. "SCF yok, converged=True" yazmak alani
    anlamsizlastirir. Burada `converged`, k sayisi ikiye katlandiginda bant
    KENARLARININ degisiminin esigin altinda kalmasidir.

    OLCULDU (1B zincir): tek n -> n=3 6.247e-02, n=11 5.061e-03,
    n=101 6.044e-05 (hicbiri 1e-6 esigini gecmiyor ve HAKLI OLARAK
    gecmiyor -- tek izgarada bant kenari n=101'de bile %0.024 eksik);
    cift n -> TAM 0.000000e+00.
    """
    senaryo = load_scenario(_yaz(tmp_path, ZINCIR_YAML))
    _, kaynak = build_tb(senaryo)
    assert kaynak.converged is True
    assert kaynak.diagnostics["k_convergence_hartree"] == 0.0

    for n, beklenen in ((3, 6.247385e-02), (11, 5.061260e-03), (101, 6.043941e-05)):
        _, kaba = build_tb(dataclasses.replace(senaryo, k_points=n))
        assert kaba.converged is False
        assert kaba.diagnostics["k_convergence_hartree"] == pytest.approx(
            beklenen, rel=1e-5
        )


def test_n_vs_2n_olcutunun_SINIRI_beyan_edilmistir(tmp_path):
    """DURUST OLUMSUZ: "iki izgara ayni sonucu veriyor" != "sonuc dogru".

    OLCULDU: cift n'de sapma TAM 0.000000e+00 cikiyor ama bu YOGUNLUKTAN
    degil PARITEDEN geliyor -- hem n hem 2n cift ve ikisi de bant
    kenarlarini tam yakaliyor. Yani olcut, ayni sistematik hatayi tasiyan
    iki izgarayi "yakinsamis" ilan edebilir.

    Bu test o sinirin gercek oldugunu gosterir: n=4 gibi COK KABA bir cift
    izgara bile `converged=True` veriyor.
    """
    senaryo = load_scenario(_yaz(tmp_path, ZINCIR_YAML))
    _, kaba = build_tb(dataclasses.replace(senaryo, k_points=4))
    assert kaba.converged is True
    assert kaba.diagnostics["k_convergence_hartree"] == 0.0
    assert kaba.diagnostics["k_points"] == 4.0


def test_bilinmeyen_kind_hala_reddediliyor(tmp_path):
    with pytest.raises(ScenarioError, match="Bilinmeyen senaryo"):
        load_scenario(_yaz(tmp_path, "kind: yok\nname: x\n"))


def test_bozuk_hoplama_reddediliyor(tmp_path):
    eksik = ZINCIR_YAML.replace("{i: 0, j: 0, cell: [1, 0, 0], t_ev: -1.7}", "{i: 0, j: 0}")
    with pytest.raises(ScenarioError, match="eksik anahtar"):
        load_scenario(_yaz(tmp_path, eksik))
    kisa = ZINCIR_YAML.replace("cell: [1, 0, 0]", "cell: [1, 0]")
    with pytest.raises(ScenarioError, match="uc tam sayi|üç tam sayı"):
        load_scenario(_yaz(tmp_path, kisa))
    fazla = ZINCIR_YAML.replace("k_points: 100", "k_points: 100\nbilinmeyen: 3")
    with pytest.raises(ScenarioError, match="Bilinmeyen anahtar"):
        load_scenario(_yaz(tmp_path, fazla))


def test_gecersiz_hoplama_indisi_MOTOR_tarafinda_yakalaniyor(tmp_path):
    bozuk = ZINCIR_YAML.replace("{i: 0, j: 0, cell: [1, 0, 0]", "{i: 0, j: 9, cell: [1, 0, 0]")
    from qlab.errors import InvalidInputError

    with pytest.raises(InvalidInputError, match="orbital indisi"):
        build_tb(load_scenario(_yaz(tmp_path, bozuk)))


def test_CLI_uctan_uca(tmp_path):
    """`python -m qlab.run scenarios/zincir_tb.yaml -o cikti.npz`.

    Suresi OLCULDU ve saniye altinda; ayri surec baslatmasina ragmen
    `@pytest.mark.slow` gerekmiyor.
    """
    cikti = tmp_path / "cikti.npz"
    sonuc = subprocess.run(
        [sys.executable, "-m", "qlab.run", "scenarios/zincir_tb.yaml", "-o", str(cikti)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert sonuc.returncode == 0, sonuc.stderr
    assert cikti.exists()
    with np.load(cikti) as veri:
        assert "energies_ev" in veri
        assert "orbital_weights" in veri
        assert bool(veri["is_metal"]) is True
    assert "Yaklaşımlar" in sonuc.stdout
    assert "geri okun" in sonuc.stdout  # dairesellik beyani EKRANA basiliyor
    assert "metal" in sonuc.stdout
