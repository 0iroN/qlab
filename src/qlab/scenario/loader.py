"""YAML senaryo dosyalarını TDSEScenario'ya çevirir.

Bilinmeyen anahtarlar sessizce yutulmaz, hata verir: "sigma_bohrr" gibi bir
yazım hatası sessizce varsayılana düşseydi simülasyon yanlış parametrelerle
koşar ve kimse fark etmezdi.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from qlab.errors import ScenarioError
from qlab.scenario.schema import (
    POTENTIAL_KINDS,
    GridSpec,
    PacketSpec,
    PotentialSpec,
    TDSEScenario,
)

_TOP_LEVEL = {
    "name", "grid", "potential", "packet", "dt",
    "n_steps", "save_every", "mass", "absorbing_width_bohr",
}
_REQUIRED = {"name", "grid", "potential", "packet", "dt", "n_steps"}


def load_scenario(path: str | Path) -> TDSEScenario:
    file_path = Path(path)
    if not file_path.is_file():
        raise ScenarioError(f"Senaryo dosyası bulunamadı: {file_path}")

    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ScenarioError(f"Dosya UTF-8 kodlanmadı: {file_path}") from exc
    except OSError as exc:
        raise ScenarioError(f"Dosya okunamadı ({file_path}): {exc}") from exc
    except yaml.YAMLError as exc:
        raise ScenarioError(f"YAML okunamadı ({file_path}): {exc}") from exc

    if not isinstance(raw, dict):
        raise ScenarioError(f"Senaryo dosyası bir sözlük olmalı: {file_path}")

    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        raise ScenarioError(
            f"Bilinmeyen anahtar(lar): {sorted(unknown)}. "
            f"Geçerli anahtarlar: {sorted(_TOP_LEVEL)}"
        )

    missing = _REQUIRED - set(raw)
    if missing:
        raise ScenarioError(f"Eksik zorunlu anahtar(lar): {sorted(missing)}")

    potential_raw = raw["potential"]
    if not isinstance(potential_raw, dict):
        raise ScenarioError(
            f"'potential' anahtar bir sözlük olmalı, {type(potential_raw).__name__} verildi"
        )

    kind = potential_raw.get("kind")
    if kind not in POTENTIAL_KINDS:
        raise ScenarioError(
            f"Bilinmeyen potansiyel türü: {kind!r}. Geçerli türler: {list(POTENTIAL_KINDS)}"
        )

    # Sayısal alanları tek tek dönüştürerek hata tanısı yapabilelim
    try:
        dt_val = float(raw["dt"])
    except (ValueError, TypeError) as exc:
        raise ScenarioError(f"'dt' sayıya dönüştürülemedi: {raw['dt']!r}") from exc

    try:
        n_steps_val = int(raw["n_steps"])
    except (ValueError, TypeError) as exc:
        raise ScenarioError(f"'n_steps' sayıya dönüştürülemedi: {raw['n_steps']!r}") from exc

    try:
        save_every_val = int(raw.get("save_every", 1))
    except (ValueError, TypeError) as exc:
        raise ScenarioError(f"'save_every' sayıya dönüştürülemedi: {raw.get('save_every')!r}") from exc

    try:
        mass_val = float(raw.get("mass", 1.0))
    except (ValueError, TypeError) as exc:
        raise ScenarioError(f"'mass' sayıya dönüştürülemedi: {raw.get('mass')!r}") from exc

    try:
        return TDSEScenario(
            name=raw["name"],
            grid=GridSpec(**raw["grid"]),
            potential=PotentialSpec(kind=kind, params=dict(potential_raw.get("params", {}))),
            packet=PacketSpec(**raw["packet"]),
            dt=dt_val,
            n_steps=n_steps_val,
            save_every=save_every_val,
            mass=mass_val,
            absorbing_width_bohr=raw.get("absorbing_width_bohr"),
        )
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"Senaryo alanları hatalı ({file_path}): {exc}") from exc
