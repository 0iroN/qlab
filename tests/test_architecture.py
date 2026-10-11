"""Katman kuralının makine tarafından zorlanması.

Spec §4.1: engine paketi viz/app import edemez. Bu kural yorumla değil,
import-linter ile korunur — ihlal edildiğinde test kırılır.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_lint_imports() -> str | None:
    """`lint-imports` konsol betiğinin tam yolunu bul.

    `python -m importlinter.cli lint-imports` KULLANILMAZ: importlinter.cli
    modülünde `if __name__ == "__main__":` koruması yok, bu yüzden `-m` ile
    çağrıldığında modül sadece import edilir, "lint-imports" argümanı sessizce
    yok sayılır ve süreç 0 çıkış koduyla hiçbir çıktı üretmeden biter — yani
    `assert result.returncode == 0` hiçbir zaman kırılmaz. Bunun yerine gerçek
    konsol betiği (`lint-imports`) doğrudan çalıştırılır. Önce PATH üzerinde
    aranır; bulunamazsa çalışan yorumlayıcının kendi Scripts/bin dizinine
    bakılır (Windows'ta kullanıcı bazlı pip kurulumlarında PATH'e otomatik
    eklenmeyebiliyor). Hiçbir mutlak yol koda gömülmez.
    """
    exe = shutil.which("lint-imports")
    if exe:
        return exe

    interpreter_dir = Path(sys.executable).parent
    candidates = [
        interpreter_dir / "Scripts" / "lint-imports.exe",  # Windows
        interpreter_dir / "Scripts" / "lint-imports",
        interpreter_dir / "bin" / "lint-imports",  # POSIX
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return None


def test_import_contracts_hold():
    lint_imports_exe = _resolve_lint_imports()
    assert lint_imports_exe is not None, (
        "'lint-imports' konsol betiği bulunamadı (ne PATH'te ne de "
        f"yorumlayıcının ({sys.executable}) Scripts/bin dizininde). "
        "import-linter kurulu değilse mimari kuralının doğrulanamadığı "
        "anlamına gelir — bu durum 'mimari sağlam' olarak YORUMLANAMAZ, "
        "test burada bilinçli olarak kırılır."
    )

    # pyproject.toml'daki sözleşme adları Türkçe karakter içeriyor (örn. "Katmanlı
    # mimari"). Windows'ta subprocess çıktısı varsayılan olarak konsol kod
    # sayfasıyla (örn. cp1252) kodlanıp çözülür — bu kod sayfası 'ş'/'ı' gibi
    # karakterleri temsil edemediğinden, UTF-8 zorlanmazsa import-linter'ın
    # kendisi rapor basarken UnicodeEncodeError ile çöker (ihlal olsun ya da
    # olmasın). Alt süreç ve çözümleme tarafını UTF-8'e sabitleyerek bu
    # platforma özgü çökmeyi önlüyoruz.
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [lint_imports_exe],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        env=env,
    )
    assert result.returncode == 0, f"Import sözleşmesi ihlal edildi:\n{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize(
    ("hedef", "ihlal", "sozlesme"),
    [
        pytest.param(
            "src/qlab/engine/hf/scf.py",
            "from PySide6 import QtWidgets",
            "engine görselleştirmeyi bilmez",
            id="engine-PySide6",
        ),
        pytest.param(
            "src/qlab/engine/hf/scf.py",
            "import pyqtgraph.opengl",
            "engine görselleştirmeyi bilmez",
            id="engine-pyqtgraph",
        ),
        pytest.param(
            "src/qlab/engine/hf/scf.py",
            "import OpenGL",
            "engine görselleştirmeyi bilmez",
            id="engine-OpenGL",
        ),
        # Modül 3 ön ölçümünde bulundu ve 20.08.2026'da kapatıldı: bu ihlal
        # ÖLÇÜLEREK rc=0 ile geçiyordu (üç sözleşme de KEPT), aynı dosyaya
        # konan PySide6 kontrolü ise rc=1 veriyordu. `matplotlib` yasak
        # listesine eklendikten sonra rc=1 ve taranan dosya sayısı 67 -> 68'e
        # çıkıyor, yani enjeksiyon gerçekten görülüyor.
        pytest.param(
            "src/qlab/engine/tb/bands.py",
            "import matplotlib.pyplot",
            "engine görselleştirmeyi bilmez",
            id="engine-tb-matplotlib",
        ),
        pytest.param(
            "src/qlab/engine/tb/lattice.py",
            "from qlab.viz import band_frames",
            "engine görselleştirmeyi bilmez",
            id="engine-tb-viz",
        ),
        pytest.param(
            "src/qlab/observable/tb_obs.py",
            "import pyqtgraph",
            "observable görselleştirmeyi ve GUI'yi bilmez",
            id="observable-tb-pyqtgraph",
        ),
        pytest.param(
            "src/qlab/observable/hf_obs.py",
            "from PySide6 import QtWidgets",
            "observable görselleştirmeyi ve GUI'yi bilmez",
            id="observable-PySide6",
        ),
        pytest.param(
            "src/qlab/observable/hf_obs.py",
            "import pyqtgraph.opengl",
            "observable görselleştirmeyi ve GUI'yi bilmez",
            id="observable-pyqtgraph",
        ),
        pytest.param(
            "src/qlab/observable/hf_obs.py",
            "import OpenGL",
            "observable görselleştirmeyi ve GUI'yi bilmez",
            id="observable-OpenGL",
        ),
        pytest.param(
            "src/qlab/units.py",
            "import qlab.engine.hf.scf",
            "Katmanlı mimari",
            id="units-engine",
        ),
        pytest.param(
            "src/qlab/engine/hf/analysis.py",
            "import qlab.run",
            "Katmanlı mimari",
            id="engine-run",
        ),
        # Modül 7 / Task 5 (03.09.2026): `bridge_map` saf tanım katmanıdır ve
        # panel metninin TEK KAYNAĞIDIR; beyan ettiği şeye (observable'daki
        # kenar tabanına) bağlanamaz. Ölçüldü: bu enjeksiyon rc=1, "Katmanlı
        # mimari BROKEN". Task 1'de aynı satır `engine.hf.scf` ile ölçülmüştü;
        # burada hedef bir katman YUKARI (observable) — sözleşme yalnız
        # motoru değil gözlenebilir katmanını da kapsıyor.
        pytest.param(
            "src/qlab/bridge_map.py",
            "import qlab.observable.bridge_obs",
            "Katmanlı mimari",
            id="bridge_map-observable",
        ),
        # `viz/bridge_frames.py` `Provenance`ı kuran katmandır ve tepe-seviye
        # `run`a bağlanamaz. Ölçüldü: rc=1, "Katmanlı mimari BROKEN"; kontrol
        # olarak dosyanın meşru import'ları (bridge_map, observable,
        # provenance, scenario) temiz ağaçta rc=0.
        pytest.param(
            "src/qlab/viz/bridge_frames.py",
            "import qlab.run",
            "Katmanlı mimari",
            id="viz-bridge-run",
        ),
        # 11.10.2026: `viz` yük modüllerinin GUI yasağı. Sözleşmeden ÖNCE
        # ölçüldü: `band_frames -> PySide6` rc=0 (üç sözleşme de KEPT). Son
        # iki satır muafiyetin DAR olduğunu gösterir: `density_frames`in
        # DOĞRUDAN pyqtgraph importu ve `band_frames`in isosurface üzerinden
        # DOLAYLI yolu muaf DEĞİL — yok sayılan tek şey
        # `density_frames -> isosurface` kenarıdır.
        pytest.param(
            "src/qlab/viz/band_frames.py",
            "from PySide6 import QtWidgets",
            "viz yükleri GUI'yi bilmez",
            id="viz-band-PySide6",
        ),
        pytest.param(
            "src/qlab/viz/spin_frames.py",
            "import matplotlib.pyplot",
            "viz yükleri GUI'yi bilmez",
            id="viz-spin-matplotlib",
        ),
        pytest.param(
            "src/qlab/viz/em_frames.py",
            "import OpenGL",
            "viz yükleri GUI'yi bilmez",
            id="viz-em-OpenGL",
        ),
        pytest.param(
            "src/qlab/viz/density_frames.py",
            "import pyqtgraph",
            "viz yükleri GUI'yi bilmez",
            id="viz-density-pyqtgraph-dogrudan",
        ),
        pytest.param(
            "src/qlab/viz/band_frames.py",
            "from qlab.viz import isosurface",
            "viz yükleri GUI'yi bilmez",
            id="viz-band-isosurface-dolayli",
        ),
    ],
)
def test_import_sozlesmesi_gercekten_REDDEDIYOR(tmp_path, hedef, ihlal, sozlesme):
    """Sözleşmenin DİŞİ var mı: ihlal enjekte edilince gerçekten kırılıyor mu?

    `test_import_contracts_hold` yalnızca temiz ağaçta rc==0 olduğunu görür ve
    bu TEK BAŞINA hiçbir şey kanıtlamaz: `forbidden_modules` boşaltılsa, yanlış
    yazılsa ya da `include_external_packages` düşürülse o test yine yeşil
    kalırdı. Bu deponun tekrar eden kusuru tam olarak budur -- yakalaması
    gerekeni yakalamayan bir test.

    ÖLÇÜLDÜ (düzeltmeden önce, `lint-imports` ile, her biri ayrı ayrı enjekte
    edilip geri alındı): aşağıdaki BEŞ ihlalin BEŞİ DE rc=0 ile KAÇIYORDU --
    `engine` doğrudan `PySide6`/`pyqtgraph`/`OpenGL` import edebiliyor,
    `observable` PySide6 import edebiliyor, `units` engine'e, `engine.hf.analysis`
    tepe-seviye `qlab.run`'a bağlanabiliyordu. Sözleşmenin ilan edilen amacı
    "engine görselleştirmeyi bilmez" iken yasak listesi yalnızca `qlab.viz` ve
    `qlab.app` içeriyordu; dış GUI paketleri ve tepe-seviye modüller hiç
    adlandırılmamıştı. Kontrol olarak enjekte edilen `engine -> qlab.viz`,
    `engine -> qlab.app` ve `scenario -> qlab.engine` o zaman da rc=1 ile
    yakalanıyordu, yani araç çalışıyordu; eksik olan KAPSAMDI.

    Kopya bir ağaçta koşar: gerçek `src/` mutasyona uğratılıp geri alınsaydı
    eşzamanlı bir koşu yanlış sonuç görebilirdi.
    """
    lint_imports_exe = _resolve_lint_imports()
    assert lint_imports_exe is not None

    kopya = tmp_path / "agac"
    shutil.copytree(PROJECT_ROOT / "src", kopya / "src")
    shutil.copy2(PROJECT_ROOT / "pyproject.toml", kopya / "pyproject.toml")

    dosya = kopya / hedef
    metin = dosya.read_text(encoding="utf-8")
    isaret = "from __future__ import annotations\n"
    assert isaret in metin, f"{hedef} beklenen çapayı taşımıyor"
    dosya.write_text(metin.replace(isaret, isaret + "\n" + ihlal + "\n", 1), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(kopya / "src")

    sonuc = subprocess.run(
        [lint_imports_exe],
        cwd=kopya,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    cikti = (sonuc.stdout or "") + (sonuc.stderr or "")
    assert sonuc.returncode != 0, (
        f"{hedef} içine enjekte edilen {ihlal!r} sözleşmeden KAÇTI:\n{cikti}"
    )
    # "BROKEN" ŞART. Sade `sozlesme in cikti` bir TOTOLOJİYDİ ve bunu ölçerek
    # gördüm: import-linter her sözleşmenin adını "<ad> KEPT" / "<ad> BROKEN"
    # diye HER KOŞUDA basar, yani ad her zaman çıktıda olur. Somut kaçış:
    # `forbidden_modules`'tan dış GUI paketlerini SİLEN mutasyon uygulandığında
    # bu testin beş parametresi de GEÇİYORDU (0/6 kırık) -- çünkü enjekte edilen
    # `engine.hf.scf -> PySide6` rc=1'i yine üretiyor, ama BAŞKA bir sözleşmeden:
    # `qlab.observable.hf_obs -> qlab.engine.hf.scf -> PySide6` zinciri
    # "observable ... bilmez" sözleşmesini kırıyor. Yani ihlal yakalanıyor gibi
    # görünürken ADI GEÇEN sözleşme hâlâ KEPT'ti. Aşağıdaki satır o farkı
    # okuyabilen tek iddiadır.
    assert f"{sozlesme} BROKEN" in cikti, (
        f"ihlal rc!=0 üretti ama {sozlesme!r} sözleşmesi KIRILMADI "
        f"(başka bir sözleşme yakalamış olabilir):\n{cikti}"
    )


def test_viz_SOZLESMESI_KAPSAMI_DIZINLE_AYNI_ve_MUAFIYET_TEK_KENAR():
    """`viz yükleri GUI'yi bilmez` sözleşmesinin kaynak listesi `viz/`
    altındaki `*frames.py` modüllerinin TAMAMIDIR (sekiz). import-linter
    listede olmayan bir modülü denetlemez: listeye yazılmadan eklenen yeni
    bir yük modülü GUI import etse bütün sözleşmeler KEPT kalırdı — bu test
    o sessiz kaçışı kapatır. Muafiyet tek bir KENARDIR (`density_frames ->
    isosurface`) ve `isosurface` kaynak listesinde değildir. Alt süreç
    koşmaz; `pyproject.toml` `tomllib` ile okunur."""
    import tomllib

    ayar = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sozlesmeler = ayar["tool"]["importlinter"]["contracts"]
    (viz,) = [s for s in sozlesmeler if s["name"] == "viz yükleri GUI'yi bilmez"]
    yukler = {f"qlab.viz.{p.stem}" for p in (PROJECT_ROOT / "src/qlab/viz").glob("*frames.py")}
    assert len(yukler) == 8, sorted(yukler)
    assert viz["type"] == "forbidden"
    assert set(viz["source_modules"]) == yukler
    assert set(viz["forbidden_modules"]) == {"PySide6", "pyqtgraph", "OpenGL", "matplotlib"}
    assert viz["ignore_imports"] == ["qlab.viz.density_frames -> qlab.viz.isosurface"]
    assert not viz.get("allow_indirect_imports", False)
    assert len(sozlesmeler) == 4
