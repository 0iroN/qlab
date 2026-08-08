"""YAML senaryo dosyalarını TDSEScenario ya da HFScanScenario'ya çevirir.

Bilinmeyen anahtarlar sessizce yutulmaz, hata verir: "sigma_bohrr" gibi bir
yazım hatası sessizce varsayılana düşseydi simülasyon yanlış parametrelerle
koşar ve kimse fark etmezdi.

Senaryo türü isteğe bağlı bir "kind" anahtarıyla seçilir: yoksa veya "tdse"
ise mevcut TDSEScenario yolu izlenir (geriye uyumluluk — Task 1A'dan kalan
mevcut YAML dosyaları "kind" taşımıyor), "hf_scan" ise HFScanScenario yolu.
Bilinmeyen bir "kind" değeri ScenarioError üretir.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from qlab.errors import ScenarioError
from qlab.scenario.schema import (
    POTENTIAL_KINDS,
    GridSpec,
    HFScanScenario,
    PacketSpec,
    PotentialSpec,
    TDSEScenario,
)

_TOP_LEVEL = {
    "kind", "name", "grid", "potential", "packet", "dt",
    "n_steps", "save_every", "mass", "absorbing_width_bohr",
}
_REQUIRED = {"name", "grid", "potential", "packet", "dt", "n_steps"}

_HF_TOP_LEVEL = {
    "kind", "name", "molecule_template", "basis", "parameter", "values", "unit",
}
_HF_REQUIRED = {"name", "molecule_template", "basis", "parameter", "values"}

_KNOWN_KINDS = ("tdse", "hf_scan")


def load_scenario(path: str | Path) -> TDSEScenario | HFScanScenario:
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

    kind = raw.get("kind", "tdse")
    if kind == "tdse":
        return _load_tdse_scenario(raw, file_path)
    if kind == "hf_scan":
        return _load_hf_scan_scenario(raw, file_path)
    raise ScenarioError(
        f"Bilinmeyen senaryo türü (kind): {kind!r}. Geçerli türler: {list(_KNOWN_KINDS)}"
    )


def _load_tdse_scenario(raw: dict, file_path: Path) -> TDSEScenario:
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


def _load_hf_scan_scenario(raw: dict, file_path: Path) -> HFScanScenario:
    unknown = set(raw) - _HF_TOP_LEVEL
    if unknown:
        raise ScenarioError(
            f"Bilinmeyen anahtar(lar): {sorted(unknown)}. "
            f"Geçerli anahtarlar: {sorted(_HF_TOP_LEVEL)}"
        )

    missing = _HF_REQUIRED - set(raw)
    if missing:
        raise ScenarioError(f"Eksik zorunlu anahtar(lar): {sorted(missing)}")

    molecule_template = raw["molecule_template"]
    if not isinstance(molecule_template, str):
        raise ScenarioError(
            "'molecule_template' bir metin olmalı, verilen: "
            f"{type(molecule_template).__name__}"
        )
    if "{value}" not in molecule_template:
        raise ScenarioError(
            "'molecule_template' içinde {value} yer tutucusu bulunmalı — "
            "aksi halde tarama aynı molekülü m kez çözerdi"
        )

    # values'ı tek tek dönüştürerek hangi girdinin bozuk olduğunu tanılayabilelim.
    # Liste-olmayan (skaler, None, ...) bir değer TypeError, sayıya çevrilemeyen
    # bir eleman ValueError fırlatır -- ikisi de burada yakalanır.
    values_raw = raw["values"]
    try:
        values_tuple = tuple(float(v) for v in values_raw)
    except TypeError as exc:
        raise ScenarioError(f"'values' bir sayı listesi olmalı: {exc}") from exc
    except ValueError as exc:
        raise ScenarioError(
            f"'values' listesindeki bir eleman sayıya dönüştürülemedi: {exc}"
        ) from exc

    if len(values_tuple) == 0:
        raise ScenarioError(
            "'values' listesi boş olamaz — en az bir tarama noktası gerekli"
        )

    # Not: HFScanScenario __post_init__ taşımayan sade bir frozen dataclass'tır
    # ve tüm alanlar burada anahtar-kelime ile veriliyor, bu yüzden aşağıdaki
    # construction hiçbir zaman TypeError/ValueError fırlatamaz — dal, tasarım
    # gereği erişilemez. _load_tdse_scenario'daki kardeş dalla simetriyi
    # korumak ve HFScanScenario ileride __post_init__ doğrulaması kazanırsa
    # (örn. alan-bazlı bir kısıt eklenirse) sessizce sızmasını önlemek için
    # savunma amaçlı bırakıldı.
    try:
        return HFScanScenario(
            name=raw["name"],
            molecule_template=molecule_template,
            basis=raw["basis"],
            parameter=raw["parameter"],
            values=values_tuple,
            unit=raw.get("unit", "angstrom"),
        )
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"Senaryo alanları hatalı ({file_path}): {exc}") from exc
