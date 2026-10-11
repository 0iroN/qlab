import io
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from qlab import run

SMALL_SCENARIO = textwrap.dedent(
    """
    name: Küçük serbest paket
    grid:
      n: 256
      length_bohr: 100.0
    potential:
      kind: free
    packet:
      x0_bohr: -10.0
      k0: 0.5
      sigma_bohr: 5.0
    dt: 0.05
    n_steps: 40
    save_every: 10
    """
)


@pytest.fixture
def scenario_file(tmp_path):
    path = tmp_path / "small.yaml"
    path.write_text(SMALL_SCENARIO, encoding="utf-8")
    return path


SMALL_HF_SCENARIO = textwrap.dedent(
    """
    kind: hf_scan
    name: Küçük H2 taraması
    molecule_template: "H 0 0 0; H 0 0 {value}"
    basis: sto-3g
    parameter: bond_length
    values: [0.7, 0.74]
    unit: angstrom
    """
)


@pytest.fixture
def hf_scenario_file(tmp_path):
    path = tmp_path / "hf_small.yaml"
    path.write_text(SMALL_HF_SCENARIO, encoding="utf-8")
    return path


def test_run_scenario_returns_frames_and_provenance(scenario_file):
    result = run.run_scenario(scenario_file)
    assert result.frames.shape == (5, 256)  # t=0 dahil
    assert result.times[-1] == pytest.approx(2.0)
    assert result.provenance.converged is True
    assert "periyodik" in " ".join(result.provenance.approximations).lower()


def test_run_scenario_preserves_norm(scenario_file):
    result = run.run_scenario(scenario_file)
    final_norm = np.sum(np.abs(result.frames[-1]) ** 2) * result.grid.dx
    # rel=0.0: varsayılan rel=1e-6 yazılan abs=1e-10'u ezmesin. Ölçülen sapma
    # 3.55e-15; abs=2e-14 ~5.6x payla bağlayıcı.
    assert final_norm == pytest.approx(1.0, abs=2e-14, rel=0.0)


def test_save_writes_readable_npz(scenario_file, tmp_path):
    result = run.run_scenario(scenario_file)
    output = tmp_path / "out.npz"
    run.save(result, output)

    with np.load(output, allow_pickle=False) as data:
        assert data["frames"].shape == (5, 256)
        assert data["x"].shape == (256,)
        assert data["potential"].shape == (256,)
        assert data["times"].shape == (5,)
        provenance = json.loads(str(data["provenance_json"]))

    assert provenance["method"].startswith("TDSE split-operator")
    assert provenance["converged"] is True


def test_main_returns_zero_and_writes_output(scenario_file, tmp_path):
    output = tmp_path / "cli.npz"
    assert run.main([str(scenario_file), "-o", str(output)]) == 0
    assert output.is_file()


def test_main_reports_missing_scenario(capsys):
    assert run.main(["yok.yaml", "-o", "x.npz"]) == 1
    assert "bulunamadı" in capsys.readouterr().err


def test_run_scenario_dispatches_hf_scan(hf_scenario_file):
    """run_scenario, HFScanScenario için TDSE fabrikasına değil run_scan'e
    gitmeli — aksi halde HF senaryosu grid/potential arayan TDSE koduna
    düşer ve AttributeError ile çöker."""
    from qlab.engine.hf.factory import ScanResult

    result = run.run_scenario(hf_scenario_file)
    assert isinstance(result, ScanResult)
    assert result.energies.shape == (2,)
    assert np.all(np.isfinite(result.energies))


def test_save_writes_readable_npz_for_hf_scan(hf_scenario_file, tmp_path):
    result = run.run_scenario(hf_scenario_file)
    output = tmp_path / "hf_out.npz"
    run.save(result, output)

    with np.load(output, allow_pickle=False) as data:
        assert set(data.files) == {
            "values",
            "energies",
            "lowest_eigenvalues",
            # Modül 1C'de eklendi: dış (RHF→UHF, üçlü) kanal da nokta bazında
            # yazılır. Aynı gerekçe, daha güçlü: bir ayrışma taramasında işaret
            # değiştiren kanal tipik olarak budur.
            "external_lowest_eigenvalues",
            "provenance_json",
        }
        assert data["values"].shape == (2,)
        assert data["energies"].shape == (2,)
        # Kararlılık bilgisi nokta bazında da makine-okunur olmalı: tek bir
        # özet bayrağı, eğrinin HANGİ noktasının eyer noktası olduğunu söylemez.
        lowest_eigenvalues = data["lowest_eigenvalues"]
        dis_eigenvalues = data["external_lowest_eigenvalues"]
        assert lowest_eigenvalues.shape == (2,)
        assert dis_eigenvalues.shape == (2,)
        provenance = json.loads(str(data["provenance_json"]))

    assert provenance["method"].startswith("RHF/")
    assert provenance["converged"] is True
    assert provenance["stability"] == {
        "stable": True,
        "lowest_eigenvalue": pytest.approx(
            float(lowest_eigenvalues.min()), abs=0.0, rel=0.0
        ),
        "n_rotations": 1,
        "channel": "ic",
    }
    # İki kanal AYRI kaydedilmeli ve birbirinin kopyası OLMAMALI -- aynı
    # sayıyı iki alana yazan bir hata, tek kanala bakan testten kaçardı.
    assert provenance["external_stability"]["channel"] == "dis"
    assert provenance["external_stability"]["lowest_eigenvalue"] == pytest.approx(
        float(dis_eigenvalues.min()), abs=0.0, rel=0.0
    )
    assert not np.allclose(lowest_eigenvalues, dis_eigenvalues)


def test_main_prints_stability_verdict_for_hf_scan(hf_scenario_file, tmp_path, capsys):
    """Şeffaflık bloğu kararlılık hükmünü GÖRÜNÜR biçimde basmalı — bu
    özelliğin varlık sebebi, sonucun kullanıcıya ulaşmasıdır."""
    output = tmp_path / "hf_stab.npz"
    assert run.main([str(hf_scenario_file), "-o", str(output)]) == 0
    out = capsys.readouterr().out
    assert "Kararlılık" in out
    assert "✓" in out


def test_main_does_not_print_stability_for_tdse(scenario_file, tmp_path, capsys):
    """TDSE koşusunda kararlılık analizi diye bir şey yoktur; panel o satırı
    hiç basmamalı (boş bir satır "sorun yok" gibi okunurdu)."""
    output = tmp_path / "tdse.npz"
    assert run.main([str(scenario_file), "-o", str(output)]) == 0
    assert "Kararlılık" not in capsys.readouterr().out


def _yakinsama_satirlari(out: str) -> tuple[str, str | None]:
    """TDSE CLI panelinden "Yakınsama" ve "Yakınsama tanısı" değerleri."""
    yakinsama = tani = None
    for satir in out.splitlines():
        govde = satir.strip()
        if govde.startswith("Yakınsama tanısı"):
            tani = govde[len("Yakınsama tanısı"):].strip()
        elif govde.startswith("Yakınsama"):
            yakinsama = govde[len("Yakınsama"):].strip()
    return yakinsama, tani


def test_main_MASKELI_TDSE_paneli_X_ve_YARGILANMADI_basar(tmp_path, capsys):
    """KAMU YOLU (B5, 11.10.2026): maskeli bir TDSE koşusu CLI panelinde ✗
    basar ve "Yakınsama tanısı" satırı YARGILANMADI ile başlar.

    Maske normu kasten düşürür ve evolve() o koşuda norm kontrolünü atlar,
    yani üniterlik HİÇ ÖLÇÜLMEZ. Eskiden panel maskeli koşuda her zaman ✓
    basıyordu (ölçüldü: 64000 adımlık maskeli tünelleme koşusunda %99.99
    norm kaybının yanına ✓). Aynı küçük senaryonun MASKESİZ hâli aynı yoldan
    ✓ basar ve tanı satırı taşımaz -- yani ✗'i maskeden başka bir şey
    üretmiyor (her koşuyu ✗ yapan bir mutant da kırmızıya döner)."""
    maskeli = tmp_path / "maskeli.yaml"
    maskeli.write_text(SMALL_SCENARIO + "absorbing_width_bohr: 10.0\n", encoding="utf-8")
    maskesiz = tmp_path / "maskesiz.yaml"
    maskesiz.write_text(SMALL_SCENARIO, encoding="utf-8")

    assert run.main([str(maskeli), "-o", str(tmp_path / "m.npz")]) == 0
    yakinsama, tani = _yakinsama_satirlari(capsys.readouterr().out)
    assert yakinsama.startswith("✗"), yakinsama
    assert "observed_norm_drop=" in yakinsama
    assert tani is not None and tani.startswith("YARGILANMADI:"), tani

    assert run.main([str(maskesiz), "-o", str(tmp_path / "s.npz")]) == 0
    yakinsama, tani = _yakinsama_satirlari(capsys.readouterr().out)
    assert yakinsama.startswith("✓"), yakinsama
    assert "norm_deviation=" in yakinsama
    assert tani is None


def test_main_reports_point_count_for_hf_scan(hf_scenario_file, tmp_path, capsys):
    output = tmp_path / "hf_cli.npz"
    assert run.main([str(hf_scenario_file), "-o", str(output)]) == 0
    out = capsys.readouterr().out
    assert "2 nokta" in out
    assert "Yöntem" in out


def test_main_outputs_turkish_text_through_cp1252_stream(scenario_file, tmp_path):
    """Ana fonksiyon, CP1252 kodlamalı (Türkçe 'ı' karakterini taşıyamayan) bir
    akış üzerinden bile Türkçe metni başarıyla çıkış yapabilmelidir."""
    output = tmp_path / "cli.npz"

    # CP1252 akışı oluştur; 'ı' karakteri bu kodlamada yapılamaz.
    # Başarılı bir çalıştırma, main() içindeki UTF-8 yeniden yapılandırmasını kanıtlar.
    cp1252_buffer = io.BytesIO()
    cp1252_stream = io.TextIOWrapper(cp1252_buffer, encoding="cp1252", errors="strict")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = cp1252_stream
        sys.stderr = io.TextIOWrapper(
            io.BytesIO(), encoding="cp1252", errors="strict"
        )

        # main() Türkçe çıktı yapmak istediğinde, fix olmadan UnicodeEncodeError
        # ile başarısız olmalı (scenario adında ve provenance'da 'ı' vardır).
        # Fix ile, main() stdout'u UTF-8'e yapılandırır ve başarılı olur.
        result = run.main([str(scenario_file), "-o", str(output)])
        assert result == 0
        assert output.is_file()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def test_main_outputs_turkish_text_through_cp1252_stream_for_hf_scan(
    hf_scenario_file, tmp_path
):
    """HF tarama dalı da, TDSE dalı gibi, CP1252 kodlamalı bir akış üzerinden
    Türkçe metni başarıyla çıkış yapabilmelidir.

    main() akışları tek bir yerde (dallanmadan önce) yeniden yapılandırır, ama
    bu korumayı sınayan tek regresyon testi şimdiye kadar yalnızca TDSE dalını
    (isinstance(result, ScanResult) False olan yol) kapsıyordu. HF dalı hem
    senaryo adında ("Küçük H2 taraması") hem de şeffaflık bloğu etiketlerinde
    ("Yöntem", "Yaklaşımlar", "İhmal edilen", "Yakınsama") Türkçe karakter
    taşıdığı için, fix olmadan aynı UnicodeEncodeError'a düşebilir.
    """
    output = tmp_path / "hf_cli.npz"

    # CP1252 akışı oluştur; 'ı'/'ş' gibi karakterler bu kodlamada yapılamaz.
    cp1252_buffer = io.BytesIO()
    cp1252_stream = io.TextIOWrapper(cp1252_buffer, encoding="cp1252", errors="strict")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = cp1252_stream
        sys.stderr = io.TextIOWrapper(
            io.BytesIO(), encoding="cp1252", errors="strict"
        )

        # main() Türkçe çıktı yapmak istediğinde, fix olmadan UnicodeEncodeError
        # ile başarısız olmalı (scenario adında ve provenance etiketlerinde
        # Türkçe karakter vardır). Fix ile, main() stdout'u UTF-8'e yapılandırır
        # ve başarılı olur.
        result = run.main([str(hf_scenario_file), "-o", str(output)])
        assert result == 0
        assert output.is_file()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def test_md_senaryosu_CLI_dan_kosuyor_ve_npz_yaziyor(tmp_path):
    """MD dalı `main`e BAĞLI mı -- B08'in bıraktığı iş.

    ÖLÇÜLDÜ (onarım öncesi): `_run_md` hiçbir yerden çağrılmıyordu ve
    `python -m qlab.run scenarios/h2_morse.yaml -o x.npz` yakalanmayan bir
    `AttributeError: 'MDScenario' object has no attribute 'grid'` veriyordu
    (main yalnız QLabError yakalar). Bu test o dalı uçtan uca kilitler.

    Sayılar LİTERAL: h2_morse.yaml 4000 adım / save_every 4 -> 1001 kayıt,
    ve dt = 0.783307192 a.u. ile son an 75.78917958522622 fs.
    """
    from qlab.run import main

    cikti = tmp_path / "md.npz"
    kok = Path(__file__).resolve().parents[1]
    assert main([str(kok / "scenarios" / "h2_morse.yaml"), "-o", str(cikti)]) == 0

    veri = np.load(cikti)
    assert sorted(veri) == [
        "kinetic", "positions", "potential", "temperature_kelvin",
        "times_fs", "total", "velocities",
    ]
    assert veri["total"].shape == (1001,)
    assert veri["times_fs"][-1] == pytest.approx(75.78917958522622, rel=1e-9, abs=0.0)
    # ilk karede v0 = 0 -> kinetik ve sıcaklık TAM sıfır (yön testi)
    assert veri["kinetic"][0] == 0.0
    assert veri["temperature_kelvin"][0] == 0.0
    assert veri["potential"][0] > 0.0


def test_MD_SURUKLEME_dali_DUZGUN_ISTISNA_firlatiyor(tmp_path):
    """`_run_md` yeterli kayit yoksa `InvalidInputError` firlatmali.

    BULUNDU (Modul 3H incelemesi, 21.08.2026): `run.py` bu istisnayi
    firlatiyordu ama IMPORT ETMEMISTI (`from qlab.errors import QLabError`
    tek basina duruyordu). Dal tetiklendiginde yakalanmayan bir `NameError`
    doguyordu, yani "surukleme olcmek icin en az 2 kayit gerekir" mesaji
    kullaniciya HIC ulasmiyordu -- `main()` yalnizca `QLabError` yakalar ve
    `NameError` onun altinda degildir.

    Depodaki hicbir test bu dala ugramiyordu; hata ancak kaynak OKUNARAK
    goruldu. Bu test o dali gercekten tetikler.
    """
    from qlab.errors import InvalidInputError
    from qlab.run import _run_md
    from qlab.scenario.schema import MDScenario

    scenario = MDScenario(
        name="tek kayitlik kosu",
        force_field={"kind": "harmonic", "params": {"k": 0.5}},
        initial={
            "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            "masses_amu": [1.00782503207, 1.00782503207],
        },
        thermostat=None,
        dt=1.0,
        # n_steps=1 + save_every=5 -> YALNIZ t=0 kaydediliyor, yani TEK kayit.
        # (n_steps=0 ise daha ERKEN reddediliyor: "n_steps en az 1 olmali" --
        #  yani o deger bu dala hic ulasmiyor ve dali sinamiyordu.)
        n_steps=1,
        save_every=5,
    )
    with pytest.raises(InvalidInputError, match="en az 2"):
        _run_md(scenario, str(tmp_path / "olmayacak.npz"))
