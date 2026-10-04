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

from qlab.engine.tb.bands import band_structure, uniform_k_grid
from qlab.engine.tb.factory import build_tb
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.units import EV_TO_HARTREE, HARTREE_TO_EV

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


def test_converged_KANITLI_kenar_siniri_B2(tmp_path):
    """E1 (04.10.2026): `converged` = B2 <= 1e-4 hartree (Osman'in esigi).

    B2 (`grid_extremum_bound`) izgaranin spektrum uclarini en cok ne kadar
    kacirabilecegini hoplama listesinden KANITLAR; olcum degil, kosegenlestirme
    de yok. Zincirde SIKIDIR: tek n'de pi noktasi kacirilir, izgaranin gercek
    kenar hatasi B2 = |t| pi^2 / n^2'ye n buyudukce 1 oraniyla yaklasir. Esik
    sevk edilen zinciri (n = 100) ✓ birakir; ✓ icin n >= 79.
    """
    senaryo = load_scenario(_yaz(tmp_path, ZINCIR_YAML))
    t, eps = 1.7 * EV_TO_HARTREE, -5.3 * EV_TO_HARTREE
    for n, yakinsadi in ((4, False), (78, False), (79, True), (100, True), (101, True)):
        model, kaynak = build_tb(dataclasses.replace(senaryo, k_points=n))
        sinir = kaynak.diagnostics["k_edge_bound_hartree"]
        assert sinir == pytest.approx(t * np.pi**2 / n**2, rel=1e-12)
        assert kaynak.converged is yakinsadi
        izgara = band_structure(model, uniform_k_grid(model.lattice, n))
        hata = max(float(izgara.min()) - (eps - 2 * t), (eps + 2 * t) - float(izgara.max()))
        assert -1e-15 <= hata <= sinir
        if n == 101:
            assert hata / sinir > 0.9999  # siki: tek n'de pi kacirilir
        if n % 2 == 0:
            assert hata < 1e-15  # cift n'de uclar izgarada


def test_ESKI_OLCUT_IYILESMEYI_OLCUYORDU_HATAYI_DEGIL(tmp_path):
    """Eski olcut (n ile 2n izgarasinin uc farki < 1e-6) n'nin HATASINI degil
    n -> 2n IYILESMESINI olcuyordu: ikisi gercek uca en yakin AYNI noktayi
    paylasinca fark TAM 0 olur. Dogrulama turunun buldugu gercek ornek: ucu
    kesirli 0.26'da olan ikinci komsulu zincir. n=4 (ve 12) ile 2n en yakin
    nokta olarak 0.25'i paylasir -> eski fark 0 (✓), gercek uc hatasi
    t1^2 / (4 t2) = 7.885e-04 hartree. B2 bunu ✗ yapar. (Zincir n=4'teki eski
    ✓ ise DOGRUYDU -- uclar izgaradaydi -- ama kanit degildi; B2 onu da ✗
    yapar: ✗ "yanlis" degil "KANITLANAMADI" demektir.)
    """
    t2 = 0.05
    t1 = -4 * t2 * np.cos(2 * np.pi * 0.26)
    model_026 = dict(
        vektorler=[[2.5, 0, 0]], taban=[[0, 0, 0]], onsite_ha=[0.0],
        hop_ha=[(0, 0, (1, 0, 0), t1), (0, 0, (-1, 0, 0), t1),
                (0, 0, (2, 0, 0), t2), (0, 0, (-2, 0, 0), t2)],
        n_e=1,
    )
    gercek_min = -(t1**2) / (4 * t2) - 2 * t2  # cos(2 pi f*) = -t1 / (4 t2), f* = 0.26
    for n in (4, 12):
        model, kaynak = build_tb(load_scenario(_yaz(tmp_path, _tb_yaml("u026", k=n, **model_026))))
        kaba = band_structure(model, uniform_k_grid(model.lattice, n))
        ince = band_structure(model, uniform_k_grid(model.lattice, 2 * n))
        eski_fark = max(abs(kaba.min() - ince.min()), abs(kaba.max() - ince.max()))
        assert eski_fark == 0.0  # eski olcut: ✓
        hata = float(kaba.min()) - gercek_min
        assert hata == pytest.approx(t1**2 / (4 * t2), rel=1e-9)
        assert hata == pytest.approx(7.885e-04, rel=1e-3)
        assert hata <= kaynak.diagnostics["k_edge_bound_hartree"]
        assert kaynak.converged is False
    senaryo = load_scenario(_yaz(tmp_path, ZINCIR_YAML))
    _, kaba = build_tb(dataclasses.replace(senaryo, k_points=4))
    assert kaba.converged is False
    assert kaba.diagnostics["k_edge_bound_hartree"] == pytest.approx(3.854e-02, rel=1e-3)
    assert kaba.diagnostics["k_points"] == 4.0


def _tb_yaml(ad, vektorler, taban, onsite_ha, hop_ha, n_e, k):
    """Hartree'de verilen modelden YAML (eV) -- kamu yolu testleri icin."""
    satir = [
        "kind: tb",
        f"name: {ad}",
        f"lattice_vectors_bohr: {vektorler}",
        f"basis_bohr: {taban}",
        f"onsite_ev: {[float(e) * HARTREE_TO_EV for e in onsite_ha]}",
        "hoppings:",
    ]
    for i, j, c, t in hop_ha:
        satir.append(
            f"  - {{i: {i}, j: {j}, cell: {list(c)}, t_ev: {float(t) * HARTREE_TO_EV!r}}}"
        )
    satir += [f"n_electron_per_cell: {n_e}", f"k_points: {k}", ""]
    return "\n".join(satir)


_T_GRAFEN = -2.7 * EV_TO_HARTREE
_GRAFEN = dict(
    vektorler=[[4.65, 0.0, 0.0], [2.325, 4.027018, 0.0]],
    taban=[[0.0, 0.0, 0.0], [2.325, 1.342339, 0.0]],
    onsite_ha=[0.0, 0.0],
    hop_ha=[
        (0, 1, (0, 0, 0), _T_GRAFEN), (0, 1, (-1, 0, 0), _T_GRAFEN),
        (0, 1, (0, -1, 0), _T_GRAFEN), (1, 0, (0, 0, 0), _T_GRAFEN),
        (1, 0, (1, 0, 0), _T_GRAFEN), (1, 0, (0, 1, 0), _T_GRAFEN),
    ],
    n_e=2,
)
# Iki orbitalli METAL (gercek ortusme -1e-3 hartree, referans E1 on olcumu):
# orbital 0'in tepesi yuksek simetri noktasi DISINDA (k* ~ 0.20).
_IKI_ORBITAL = dict(
    vektorler=[[2.5, 0, 0]], taban=[[0, 0, 0], [1.2, 0, 0]],
    onsite_ha=[0.0, 0.17763795853762832],
    hop_ha=[
        (0, 0, (1, 0, 0), 0.0625), (0, 0, (-1, 0, 0), 0.0625),
        (0, 0, (2, 0, 0), -0.05), (0, 0, (-2, 0, 0), -0.05),
        (1, 1, (1, 0, 0), -0.03), (1, 1, (-1, 0, 0), -0.03),
        (0, 1, (0, 0, 0), 0.005), (1, 0, (0, 0, 0), 0.005),
    ],
    n_e=2,
)
_SSH = dict(
    vektorler=[[4.0, 0, 0]], taban=[[0, 0, 0], [1.8, 0, 0]], onsite_ha=[0.0, 0.0],
    hop_ha=[
        (0, 1, (0, 0, 0), -0.10), (1, 0, (0, 0, 0), -0.10),
        (1, 0, (1, 0, 0), -0.04), (0, 1, (-1, 0, 0), -0.04),
    ],
    n_e=2,
)
# Yalniz +-2 hucreye hoplayan zincir: k=2 izgarasinda bant DUZ gorunur.
_ARTI_IKI = dict(
    vektorler=[[2.5, 0, 0]], taban=[[0, 0, 0]], onsite_ha=[-0.1948],
    hop_ha=[(0, 0, (2, 0, 0), -0.0625), (0, 0, (-2, 0, 0), -0.0625)], n_e=1,
)


def _panel(tmp_path, capsys, metin):
    import qlab.run as run_modulu

    cikti = tmp_path / "panel.npz"
    assert run_modulu.main([str(_yaz(tmp_path, metin)), "-o", str(cikti)]) == 0
    satirlar = capsys.readouterr().out.splitlines()
    hukum = next(s for s in satirlar if "İletkenlik" in s).split(":", 1)[1].strip()
    yakinsama = next(s for s in satirlar if s.startswith("Yakınsama"))[16:]
    with np.load(cikti) as veri:
        npz = (str(veri["iletkenlik"]), bool(veri["is_metal"]))
    return hukum, yakinsama, npz


def test_E13_ILETKENLIK_HUKMU_KANITLI_KAMU_YOLUNDA(tmp_path, capsys):
    """E13 (04.10.2026): panel "yalıtkan" yalniz KANITLIYKEN der.

    Eskiden (olculdu, kamu yolu): grafen her k_points'te "yalıtkan ✓"
    (Dirac noktasi izgarada yoksa sahte aralik, varsa yuvarlamayla +5e-17
    ayrik bantlar); iki orbitalli metal k=4/16'da "yalıtkan ✓"; yalniz +-2
    hucreli zincir k=2'de "yalıtkan". Uc degerli hukum Osman'in karari.
    Iki bayrak BAGIMSIZDIR: grafen k=100'de kenarlar kanitli (✓) ama
    hukum "belirsiz" -- dokunan bantlar hicbir n'de metal kanitlanamaz.
    """
    vakalar = (
        (_GRAFEN, 4, "belirsiz", "✗"),
        (_GRAFEN, 12, "belirsiz", "✗"),
        (_GRAFEN, 100, "belirsiz", "✓"),
        (_IKI_ORBITAL, 4, "belirsiz", "✗"),
        (_IKI_ORBITAL, 16, "belirsiz", "✗"),
        (_IKI_ORBITAL, 10, "metal", "✗"),
        (_SSH, 4, "yalıtkan", "✗"),
        (_SSH, 100, "yalıtkan", "✓"),
        (_ARTI_IKI, 2, "belirsiz", "✗"),
        (_ARTI_IKI, 3, "metal", "✗"),
    )
    for model, k, beklenen, yakinsama_isareti in vakalar:
        hukum, yakinsama, npz = _panel(tmp_path, capsys, _tb_yaml("vaka", k=k, **model))
        assert hukum.startswith(beklenen + " —"), (k, hukum)
        assert yakinsama.startswith(yakinsama_isareti), (k, yakinsama)
        assert npz == (beklenen, beklenen == "metal"), (k, npz)
    hukum, _, _ = _panel(tmp_path, capsys, _tb_yaml("vaka", k=100, **_SSH))
    assert "> 2·B1" in hukum and "eV" in hukum
    # Dogrulama turu: ayni anahtarda birbirini goturen parcalar (1.23 + 1.6 -
    # 2.83 eV = 0) DUZ bant verir; ilk surumun payi yalniz |E|max'a bagliydi
    # ve toplama kirintisini (7.6e-16 eV) "metal (tanikli)" basiyordu.
    parcali = ZINCIR_YAML.replace(
        "  - {i: 0, j: 0, cell: [1, 0, 0], t_ev: -1.7}\n  - {i: 0, j: 0, cell: [-1, 0, 0], t_ev: -1.7}\n",
        "".join(
            f"  - {{i: 0, j: 0, cell: [{s}, 0, 0], t_ev: {t}}}\n"
            for s in (1, -1) for t in (1.23, 1.6, -2.83)
        ),
    ).replace("k_points: 100", "k_points: 50").replace("onsite_ev: [-5.3]", "onsite_ev: [0.0]")
    # onsite 0: |E|max yuvarlama mertebesinde kalir, pay YALNIZ H olcegiyle
    # (`hamiltonian_rounding_scale`) yeterli olur -- onsite -5.3 eV iken |E|max
    # payi tek basina kurtariyordu ve run.py'nin olcegi gecirmemesi (mutant)
    # hayatta kaliyordu.
    assert "1.23" in parcali and "onsite_ev: [0.0]" in parcali
    hukum, _, npz = _panel(tmp_path, capsys, parcali)
    assert hukum.startswith("belirsiz —") and npz == ("belirsiz", False)


def test_bilinmeyen_kind_hala_reddediliyor(tmp_path):
    with pytest.raises(ScenarioError, match="Bilinmeyen senaryo"):
        load_scenario(_yaz(tmp_path, "kind: yok\nname: x\n"))


def test_bozuk_hoplama_reddediliyor(tmp_path):
    eksik = ZINCIR_YAML.replace("{i: 0, j: 0, cell: [1, 0, 0], t_ev: -1.7}", "{i: 0, j: 0}")
    with pytest.raises(ScenarioError, match="eksik anahtar"):
        load_scenario(_yaz(tmp_path, eksik))
    # Eskiden int() KESIYORDU: [1.7, 0, 0] sessizce (1, 0, 0) oluyordu.
    kesirli = ZINCIR_YAML.replace("cell: [1, 0, 0]", "cell: [1.7, 0, 0]")
    with pytest.raises(ScenarioError, match="tam sayı"):
        load_scenario(_yaz(tmp_path, kesirli))
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


def test_H0_TEK_YONLU_hoplama_KAMU_YOLUNDA_reddediliyor(tmp_path, capsys):
    """H0 (04.10.2026): eskiden rc=0, "metal ✓" ve bant genisligi 3.4 eV
    (dogrusu 6.8) basiyordu. Artik `run.main` rc=1 ve nedeni soyler."""
    import qlab.run as run_modulu

    tek_yon = ZINCIR_YAML.replace("  - {i: 0, j: 0, cell: [-1, 0, 0], t_ev: -1.7}\n", "")
    assert tek_yon != ZINCIR_YAML
    rc = run_modulu.main([str(_yaz(tmp_path, tek_yon)), "-o", str(tmp_path / "x.npz")])
    assert rc == 1
    assert "Hermityen degil" in capsys.readouterr().err
    assert not (tmp_path / "x.npz").exists()


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
        assert str(veri["iletkenlik"]) == "metal"
    assert "Yaklaşımlar" in sonuc.stdout
    assert "geri okun" in sonuc.stdout  # dairesellik beyani EKRANA basiliyor
    assert "metal" in sonuc.stdout
