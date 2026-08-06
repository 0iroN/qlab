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


@pytest.mark.skip(reason="sözleşmedeki modüller Task 3'te oluşacak")
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
