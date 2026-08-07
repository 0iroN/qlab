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
    kind = potential_raw.get("kind")
    if kind not in POTENTIAL_KINDS:
        raise ScenarioError(
            f"Bilinmeyen potansiyel türü: {kind!r}. Geçerli türler: {list(POTENTIAL_KINDS)}"
        )

    try:
        return TDSEScenario(
            name=raw["name"],
            grid=GridSpec(**raw["grid"]),
            potential=PotentialSpec(kind=kind, params=dict(potential_raw.get("params", {}))),
            packet=PacketSpec(**raw["packet"]),
            dt=float(raw["dt"]),
            n_steps=int(raw["n_steps"]),
            save_every=int(raw.get("save_every", 1)),
            mass=float(raw.get("mass", 1.0)),
            absorbing_width_bohr=raw.get("absorbing_width_bohr"),
        )
    except TypeError as exc:
        raise ScenarioError(f"Senaryo alanları hatalı ({file_path}): {exc}") from exc
