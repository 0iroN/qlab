"""YAML senaryo dosyalarını dokuz senaryo türünden birine çevirir (aşağıda).

Bilinmeyen anahtarlar sessizce yutulmaz, hata verir: "sigma_bohrr" gibi bir
yazım hatası sessizce varsayılana düşseydi simülasyon yanlış parametrelerle
koşar ve kimse fark etmezdi.

Senaryo türü isteğe bağlı bir "kind" anahtarıyla seçilir: yoksa veya "tdse"
ise mevcut TDSEScenario yolu izlenir (geriye uyumluluk — Task 1A'dan kalan
mevcut YAML dosyaları "kind" taşımıyor). Diğerleri: "hf_scan" → HFScanScenario,
"md" → MDScenario, "tb" → TBScenario, "downfold" → DownfoldScenario,
"spin" → SpinScenario, "em" → EMScenario, "grav" → GravScenario,
"bridge" → BridgeScenario.
Bilinmeyen bir "kind" değeri ScenarioError üretir ve geçerli türleri sayar.

ALT TÜRLER de aynı kuralla denetlenir, tüketiciye ham hata sızmadan:
"potential.kind" → POTENTIAL_KINDS, "model"/"algorithm" → SPIN_*_KINDS,
ve "force_field.kind"/"thermostat.kind" → MD_*_KINDS; `tb_ss_sigma` türünün
parametre anahtarları → TB_SS_SIGMA_* ve RADIAL_LAW_FIELDS.
"""

from __future__ import annotations

import math
import re
import string
import sys
from pathlib import Path

import yaml

from qlab.errors import ScenarioError
from qlab.scenario.schema import (
    EM_MIN_DEPTH,
    EM_WINDOW_FNS,
    GRAV_GAUGES,
    GRAV_NBODY_METHODS,
    MD_FORCE_FIELD_KINDS,
    MD_THERMOSTAT_KINDS,
    POTENTIAL_KINDS,
    RADIAL_LAW_FIELDS,
    TB_SS_SIGMA_LAWS,
    TB_SS_SIGMA_PARAMS,
    TB_SS_SIGMA_REQUIRED,
    SPIN_ALGORITHM_KINDS,
    SPIN_MODEL_KINDS,
    BridgeScenario,
    DownfoldScenario,
    EMScenario,
    GravScenario,
    GridSpec,
    HFScanScenario,
    PacketSpec,
    MDScenario,
    PotentialSpec,
    SpinScenario,
    TBHopping,
    TBScenario,
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

_MD_TOP_LEVEL = {
    "kind", "name", "force_field", "initial", "thermostat",
    "dt", "n_steps", "save_every",
}
_MD_REQUIRED = {"name", "force_field", "initial", "dt", "n_steps"}

_TB_TOP_LEVEL = {
    "kind", "name", "lattice_vectors_bohr", "basis_bohr", "onsite_ev",
    "hoppings", "n_electron_per_cell", "k_points",
}
_TB_REQUIRED = {
    "name", "lattice_vectors_bohr", "basis_bohr", "onsite_ev", "hoppings",
    "n_electron_per_cell", "k_points",
}

_DOWNFOLD_TOP_LEVEL = {
    "kind", "name", "ring_size", "distances_bohr", "basis", "n_neighbour",
    "seeds",
}
_DOWNFOLD_REQUIRED = {"name", "ring_size", "distances_bohr"}

_SPIN_TOP_LEVEL = {
    "kind", "name", "model", "algorithm", "lattice_size", "temperatures",
    "n_sweep", "n_burn", "seed", "coupling", "field", "heisenberg_step",
}
_SPIN_REQUIRED = {
    "name", "model", "algorithm", "lattice_size", "temperatures", "n_sweep",
    "n_burn", "seed",
}

_EM_TOP_LEVEL = {
    "kind", "name", "radii", "ratio", "courant", "step_factor",
    "wavelength_per_radius", "sgf", "detectors", "pad", "window_fn",
    "peak_floor", "eps_r", "margin",
}
# HEPSI ZORUNLU (kind/name dahil, eps_r/margin haric). Bu, TDSE ve MD
# semalarindan bilincli bir SAPMADIR: orada varsayilanlar var, burada
# pencerenin her dugmesi yazilmak zorunda -- cunku bu modulde tekrarlanabilirlik
# tohumla degil PENCERE BEYANIYLA saglanir.
_EM_REQUIRED = {
    "name", "radii", "ratio", "courant", "step_factor",
    "wavelength_per_radius", "sgf", "detectors", "pad", "window_fn",
    "peak_floor",
}

#: 2B Courant siniri. `engine/em/grid.py::courant_limit(2)` ile AYNI sayi,
#: ayni katman ayrimiyla tekrar edilir: yukleyici YAML'in hatasini FDTD
#: kosmadan once yakalar, motor (`yee_coefficients_2d`) cagrinin hatasini
#: yakalar ve ikisi ayri tur firlatir.
_EM_COURANT_LIMIT_2D = 2.0 ** -0.5

_GRAV_TOP_LEVEL = {
    "kind", "name", "gauge", "gamma", "beta", "mass", "eccentricity",
    "richardson", "second_order_p", "deflection_richardson", "trajectory_p",
    "trajectory_points", "nbody_rtol", "nbody_atol", "nbody_method",
}
# HEPSI ZORUNLU (kind haric). `_EM_REQUIRED` ile ayni gerekce ve ondan bir
# adim daha ileri: EM'de `eps_r`/`margin` varsayilanliydi, burada VARSAYILANLI
# ALAN YOK. Bu modulde tekrarlanabilirligin karsiligi pencerenin tam
# beyanidir ve pencerenin her dugmesi -- Richardson ciftleri, `p`/`b`
# basamaklari, N-cisim toleranslari, integrator adi ve KURAM SECICI (gamma,
# beta) -- yazilmak zorundadir.
_GRAV_REQUIRED = _GRAV_TOP_LEVEL - {"kind"}

_BRIDGE_TOP_LEVEL = {
    "kind", "name", "a_ikiz", "n_bolme", "periyot_sayisi", "genlikler",
}
# HEPSİ ZORUNLU (kind hariç) — `_GRAV_REQUIRED` ile aynı gerekçe: bu modülde
# tekrarlanabilirliğin karşılığı pencerenin tam beyanıdır. `a` (A1'in
# fizik-kapatma merdiveni) BİLEREK bu kümede DEĞİLDİR ve "bilinmeyen anahtar"
# hatası verir; gerekçesi `BridgeScenario` dokstringinde.
_BRIDGE_REQUIRED = _BRIDGE_TOP_LEVEL - {"kind"}

_KNOWN_KINDS = (
    "tdse", "hf_scan", "md", "tb", "downfold", "spin", "em", "grav", "bridge",
)


def load_scenario(
    path: str | Path,
) -> (
    TDSEScenario | HFScanScenario | MDScenario | TBScenario | DownfoldScenario
    | SpinScenario | EMScenario | GravScenario | BridgeScenario
):
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
    if kind == "md":
        return _load_md_scenario(raw, file_path)
    if kind == "tb":
        return _load_tb_scenario(raw, file_path)
    if kind == "downfold":
        return _load_downfold_scenario(raw, file_path)
    if kind == "spin":
        return _load_spin_scenario(raw, file_path)
    if kind == "em":
        return _load_em_scenario(raw, file_path)
    if kind == "grav":
        return _load_grav_scenario(raw, file_path)
    if kind == "bridge":
        return _load_bridge_scenario(raw, file_path)
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


def _sablon_kapisi(sablon: str) -> None:
    """`molecule_template`in yer tutucu kapısı — `str.format`ın KENDİ ayrıştırıcısıyla.

    Fabrika şablonu `str.format(value=...)` ile doldurur; kapı aynı dilbilgisini
    okur. Eskiden düz alt-dize testiydi (`"{value}" in şablon`) ve İKİ YÖNDE
    yanılıyordu (ölçüldü 11.10.2026, kamu yolu): `{value:.2f}` (geçerli: 0.7412
    → `0.74`) REDDEDİLİYORDU; `{{value}}` (yer tutucu DEĞİL, düz metin) ve
    `{value} {oops}` KABUL ediliyordu — ikincisi ancak tarama başlayınca düşüyordu.

    Kural: en az bir alan ve her alanın adı TAM OLARAK `value` (biçim belirteci
    ve dönüşüm serbest: `{value:.2f}`, `{value!r}`); `{val}`, `{}`, `{value.real}`
    reddedilir. Sonra şablon bir kez `value=1.0` ile biçimlenir: geçersiz bir
    biçim belirteci (`{value:d}`) ya da iç içe bir alan (`{value:{w}}`,
    `{value:{value[0]}}`, `{value:{value.foo}}`) da YAML okunurken düşer —
    sonuncu ikisi `TypeError`/`AttributeError` fırlatır ve ham sızıyordu
    (doğrulayıcı, 11.10.2026). Fabrikanın `_molecule_for` kapısı Python yolu
    (`HFScanScenario(...)` doğrudan) için DURUR — katman ayrımı.

    Bu kapı tek bir şablonu sınar; FARKLI değerlerin AYNI moleküle
    biçimlenmesini (`{value:.0f}`) `_load_hf_scan_scenario`daki çarpışma
    kapısı yakalar, çünkü o değerleri görmeyi gerektirir.
    """
    try:
        adlar = [ad for _, ad, _, _ in string.Formatter().parse(sablon) if ad is not None]
    except ValueError as exc:
        raise ScenarioError(f"'molecule_template' ayrıştırılamadı: {exc}") from exc
    if "value" not in adlar:
        raise ScenarioError(
            "'molecule_template' içinde {value} yer tutucusu bulunmalı — "
            "aksi halde tarama aynı molekülü m kez çözerdi"
        )
    yabanci = sorted({ad for ad in adlar if ad != "value"})
    if yabanci:
        raise ScenarioError(
            f"'molecule_template' {{value}} dışında yer tutucu içeriyor: {yabanci}. "
            "Fabrika şablonu yalnız value= ile doldurur."
        )
    try:
        sablon.format(value=1.0)
    except (KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
        raise ScenarioError(
            f"'molecule_template' biçimlenemiyor (value = 1.0 ile sınandı): {exc!r}"
        ) from exc


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
    _sablon_kapisi(molecule_template)

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

    # ÇARPIŞMA KAPISI (B4, 11.10.2026): biçim belirteci FARKLI değerleri AYNI
    # moleküle indirebilir. ÖLÇÜLDÜ (doğrulayıcı, kamu yolu, kapı yokken):
    # `{value:.0f}` ile values [0.6, 0.7, 0.9, 1.4] dört ÖZDEŞ enerji
    # (-1.0661086493089347) verdi; CLI "4 nokta", her denetim ✓, rc = 0 —
    # yer tutucu kapısının önlemek için var olduğu sessiz yolun ta kendisi.
    # Biçimlenmiş moleküller, farklı değerler kadar farklı olmalı. Değerler
    # biçimlenmiş METNE göre gruplanır, küme boylarının eşitliğine göre değil:
    # `0.0` ile `-0.0` eşit değer ama farklı metindir ve çarpışma DEĞİLDİR.
    # Fabrikanın `run_scan`i aynı kapıyı Python yolu için tekrarlar. Biçimleme
    # burada fırlatamaz: `_sablon_kapisi` şablonu bir float ile sınadı ve bir
    # float biçim belirtecinin geçerliliği değere bağlı değildir.
    gruplar: dict[str, list[float]] = {}
    for deger in values_tuple:
        grup = gruplar.setdefault(molecule_template.format(value=deger), [])
        if deger not in grup:
            grup.append(deger)
    carpisan = {metin: grup for metin, grup in gruplar.items() if len(grup) > 1}
    if carpisan:
        raise ScenarioError(
            "'molecule_template' farklı 'values' değerlerini AYNI moleküle "
            f"biçimliyor: {carpisan}. Tarama aynı molekülü birden çok kez çözer "
            "ve eğri sessizce düzleşirdi; biçim belirtecinin hassasiyetini "
            "artırın (ör. {value:.0f} yerine {value:.2f}) ya da değerleri ayırın."
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


def _tb_ss_sigma_parametre_kapisi(params: object) -> None:
    """`tb_ss_sigma` parametrelerinin ANAHTAR kapısı (değerleri motor denetler).

    Öteki MD türlerinin parametreleri `**p` ile sınıfa gider ve yazım
    hatası orada ham `TypeError` verirdi; bu tür iki iç içe yasa ve bir
    birim dönüşümü taşıdığı için anahtarlar burada, YAML okunurken ve
    geçerli adlar SAYILARAK reddedilir. Değer kapıları (sonluluk,
    `r_on < r_cut`, kT > 0, L/2) motordadır ve `InvalidInputError` fırlatır;
    iki denetim ayrı türdür.
    """
    if not isinstance(params, dict):
        raise ScenarioError(
            "'force_field.params' tb_ss_sigma için bir sözlük olmalı, verilen: "
            f"{type(params).__name__}"
        )
    bilinmeyen = set(params) - set(TB_SS_SIGMA_PARAMS)
    if bilinmeyen:
        raise ScenarioError(
            f"tb_ss_sigma: bilinmeyen parametre(ler) {sorted(bilinmeyen)}. "
            f"Geçerli parametreler: {list(TB_SS_SIGMA_PARAMS)}"
        )
    eksik = set(TB_SS_SIGMA_REQUIRED) - set(params)
    if eksik:
        raise ScenarioError(
            f"tb_ss_sigma: eksik parametre(ler) {sorted(eksik)} "
            f"(zorunlu: {list(TB_SS_SIGMA_REQUIRED)})"
        )
    for rol in TB_SS_SIGMA_LAWS:
        yasa = params[rol]
        if not isinstance(yasa, dict):
            raise ScenarioError(
                f"tb_ss_sigma: '{rol}' bir sözlük olmalı "
                f"({list(RADIAL_LAW_FIELDS)}), verilen: {type(yasa).__name__}"
            )
        if set(yasa) != set(RADIAL_LAW_FIELDS):
            raise ScenarioError(
                f"tb_ss_sigma: '{rol}' yasasının anahtarları "
                f"{sorted(yasa)}; beklenen tam olarak {list(RADIAL_LAW_FIELDS)} "
                f"(bilinmeyen {sorted(set(yasa) - set(RADIAL_LAW_FIELDS))}, "
                f"eksik {sorted(set(RADIAL_LAW_FIELDS) - set(yasa))})"
            )


def _load_md_scenario(raw: dict, file_path: Path) -> MDScenario:
    unknown = set(raw) - _MD_TOP_LEVEL
    if unknown:
        raise ScenarioError(
            f"Bilinmeyen anahtar(lar): {sorted(unknown)}. "
            f"Geçerli anahtarlar: {sorted(_MD_TOP_LEVEL)}"
        )
    missing = _MD_REQUIRED - set(raw)
    if missing:
        raise ScenarioError(f"Eksik anahtar(lar): {sorted(missing)}")

    # Kuvvet alanı türü BURADA denetlenir, `engine/md/factory.py`ye varmadan.
    # Factory'nin `else: raise` dalı DURUYOR ve kaldırılmamalı: o dal
    # `MDScenario(...)` doğrudan kurulduğunda (testler bunu yapıyor) tek
    # bekçidir. Buradaki kapının kazancı ayrı: hata YAML'ı okurken çıkar ve
    # geçerli türleri SAYAR -- `POTENTIAL_KINDS`/`SPIN_MODEL_KINDS`
    # kapılarının deseni.
    force_field = raw["force_field"]
    if not isinstance(force_field, dict):
        raise ScenarioError(
            f"'force_field' bir sözlük olmalı, verilen: "
            f"{type(force_field).__name__}"
        )
    ff_kind = force_field.get("kind")
    if ff_kind not in MD_FORCE_FIELD_KINDS:
        raise ScenarioError(
            f"Bilinmeyen kuvvet alanı: {ff_kind!r}. "
            f"Geçerli kuvvet alanları: {list(MD_FORCE_FIELD_KINDS)}"
        )
    if ff_kind == "tb_ss_sigma":
        _tb_ss_sigma_parametre_kapisi(force_field.get("params"))

    # Termostat türü de BURADA denetlenir -- yukarıdaki kapıyla aynı gerekçe.
    # FARK ÖLÇÜLDÜ (30.08.2026): `MD_THERMOSTAT_KINDS` ÖLÜydü ama BAYAT
    # DEĞİLDİ. Yani bu kapı mevcut bir kusuru değil, aynı tuzağın ikinci
    # kopyasını kapatıyor -- ve asıl işi türetme kilidi yapıyor.
    #
    # Termostat İSTEĞE BAĞLIDIR: yoksa ya da boşsa NVE koşusudur ve kapı
    # ateşlenmez. Bu `else None` semantiği KORUNUYOR, daraltılmıyor.
    thermostat = raw.get("thermostat")
    if thermostat:
        if not isinstance(thermostat, dict):
            raise ScenarioError(
                f"'thermostat' bir sözlük olmalı, verilen: "
                f"{type(thermostat).__name__}"
            )
        t_kind = thermostat.get("kind")
        if t_kind not in MD_THERMOSTAT_KINDS:
            raise ScenarioError(
                f"Bilinmeyen termostat: {t_kind!r}. "
                f"Geçerli termostatlar: {list(MD_THERMOSTAT_KINDS)}"
            )

    # C6 (11.10.2026, ölçüldü `python -m qlab.run`): `initial: [1, 2]` ve
    # `params: [1, 2]` fabrikadaki `dict(...)`te ham TypeError veriyordu;
    # `dt: abc` ve `n_steps: dort` ham ValueError, `n_steps: 40.7` ise SESSİZCE
    # 40'a kesiliyordu. `_yaml_sayisi`/`_yaml_tam_sayisi` spin yükleyicisinin kapısı.
    for ad, blok in (
        ("initial", raw["initial"]),
        ("force_field.params", force_field.get("params", {})),
        ("thermostat.params", thermostat.get("params", {}) if thermostat else {}),
    ):
        if not isinstance(blok, dict):
            raise ScenarioError(
                f"'{ad}' bir sözlük olmalı, verilen: {type(blok).__name__}"
            )
    return MDScenario(
        name=str(raw["name"]),
        force_field=dict(force_field),
        initial=dict(raw["initial"]),
        thermostat=dict(thermostat) if thermostat else None,
        dt=_yaml_sayisi("dt", raw["dt"]),
        n_steps=_yaml_tam_sayisi("n_steps", raw["n_steps"]),
        save_every=_yaml_tam_sayisi("save_every", raw.get("save_every", 1)),
    )


def _load_tb_scenario(raw: dict, file_path: Path) -> TBScenario:
    unknown = set(raw) - _TB_TOP_LEVEL
    if unknown:
        raise ScenarioError(
            f"Bilinmeyen anahtar(lar): {sorted(unknown)}. "
            f"Geçerli anahtarlar: {sorted(_TB_TOP_LEVEL)}"
        )
    missing = _TB_REQUIRED - set(raw)
    if missing:
        raise ScenarioError(f"Eksik anahtar(lar): {sorted(missing)}")

    hoplamalar = []
    for indis, ham in enumerate(_tb_listesi("hoppings", raw["hoppings"])):
        if not isinstance(ham, dict):
            raise ScenarioError(
                f"'hoppings[{indis}]' bir sözlük olmalı, "
                f"{type(ham).__name__} verildi"
            )
        eksik = {"i", "j", "cell", "t_ev"} - set(ham)
        if eksik:
            raise ScenarioError(
                f"'hoppings[{indis}]' eksik anahtar(lar): {sorted(eksik)}"
            )
        # Eskiden int(x) KESIYORDU: [1.7, 0, 0] sessizce (1, 0, 0) oluyordu
        # (dogrulama turu, 04.10.2026). Tam sayi olmayan bilesen reddedilir.
        if not isinstance(ham["cell"], (list, tuple)) or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not float(x).is_integer()
            for x in ham["cell"]
        ):
            raise ScenarioError(
                f"'hoppings[{indis}].cell' üç tam sayı olmalı: {ham['cell']!r}"
            )
        hucre = tuple(int(x) for x in ham["cell"])
        if len(hucre) != 3:
            raise ScenarioError(
                f"'hoppings[{indis}].cell' üç tam sayı olmalı: {ham['cell']!r}"
            )
        hoplamalar.append(
            TBHopping(
                i=_yaml_tam_sayisi(f"hoppings[{indis}].i", ham["i"]),
                j=_yaml_tam_sayisi(f"hoppings[{indis}].j", ham["j"]),
                cell=hucre,
                t_ev=_yaml_sayisi(f"hoppings[{indis}].t_ev", ham["t_ev"]),
            )
        )

    # LOADER-VE (11.10.2026, ölçüldü `python -m qlab.run`): sayı olmayan
    # girdi (`lattice_vectors_bohr: [[abc, ...]]`, `basis_bohr`, `onsite_ev`,
    # `n_electron_per_cell: bir`, `k_points`, `t_ev`, `i`) ham ValueError,
    # liste yerine sayı ham TypeError veriyordu; `n_electron_per_cell: 1.5`
    # SESSİZCE 1'e, `k_points: 100.7` 100'e kesiliyor, `true` 1.0 ve tırnaklı
    # `'-5.3'` sayı oluyordu. Kapılar spin yükleyicisininkiler.
    def _vektorler(ad: str) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(
                _yaml_sayisi(f"{ad}[{s}][{b}]", x)
                for b, x in enumerate(_tb_listesi(f"{ad}[{s}]", v))
            )
            for s, v in enumerate(_tb_listesi(ad, raw[ad]))
        )

    return TBScenario(
        name=str(raw["name"]),
        lattice_vectors_bohr=_vektorler("lattice_vectors_bohr"),
        basis_bohr=_vektorler("basis_bohr"),
        onsite_ev=tuple(
            _yaml_sayisi(f"onsite_ev[{s}]", x)
            for s, x in enumerate(_tb_listesi("onsite_ev", raw["onsite_ev"]))
        ),
        hoppings=tuple(hoplamalar),
        n_electron_per_cell=_yaml_tam_sayisi(
            "n_electron_per_cell", raw["n_electron_per_cell"]
        ),
        k_points=_yaml_tam_sayisi("k_points", raw["k_points"]),
    )


def _tb_listesi(ad: str, deger: object) -> list:
    """TB senaryosunda liste beklenen alan; sayı/dize/sözlük `ScenarioError`."""
    if not isinstance(deger, (list, tuple)):
        raise ScenarioError(
            f"'{ad}' bir liste olmalı, verilen: {deger!r} ({type(deger).__name__})"
        )
    return list(deger)


def _load_downfold_scenario(raw: dict, file_path: Path) -> DownfoldScenario:
    """UHF indirgemesi senaryosu.

    Yükleyici, motorun iki kontrolünü (çift `ring_size`, menzil aralığı)
    BİLEREK tekrarlar ve bu bir kopya değil bir KATMAN AYRIMIDIR: burada
    yakalanan şey YAML'ın hatasıdır ve hata SCF koşmadan ÖNCE verilir;
    motorda (`downfold_uhf`) yakalanan şey ÇAĞRININ hatasıdır. İkisi ayrı
    tür fırlatır — `ScenarioError` ve `UnsupportedSystemError` — ve testler
    bu farkı kilitler, yani biri kaldırılırsa öteki sessizce yerini almaz.

    Yükleyiciye ÖZGÜ olanlar: boş / tekrarlı / pozitif olmayan mesafe
    listesi ve `seeds < 1`. Bunların motorda karşılığı yoktur çünkü motor
    tek bir molekülle çağrılır, mesafe listesiyle değil.
    """
    unknown = set(raw) - _DOWNFOLD_TOP_LEVEL
    if unknown:
        raise ScenarioError(
            f"Bilinmeyen anahtar(lar): {sorted(unknown)}. "
            f"Geçerli anahtarlar: {sorted(_DOWNFOLD_TOP_LEVEL)}"
        )
    missing = _DOWNFOLD_REQUIRED - set(raw)
    if missing:
        raise ScenarioError(f"Eksik anahtar(lar): {sorted(missing)}")

    try:
        ring_size = int(raw["ring_size"])
    except (TypeError, ValueError) as exc:
        raise ScenarioError(
            f"'ring_size' tam sayıya dönüştürülemedi: {raw['ring_size']!r}"
        ) from exc
    if ring_size < 4 or ring_size % 2 != 0:
        raise ScenarioError(
            f"'ring_size' EN AZ 4 ve ÇİFT olmalı, verilen: {ring_size}. "
            "İki-siteli antiferromanyetik hücre tek sayıda sitede kurulamaz."
        )

    try:
        distances = tuple(float(x) for x in raw["distances_bohr"])
    except TypeError as exc:
        raise ScenarioError(f"'distances_bohr' bir sayı listesi olmalı: {exc}") from exc
    except ValueError as exc:
        raise ScenarioError(
            f"'distances_bohr' listesindeki bir eleman sayıya dönüştürülemedi: {exc}"
        ) from exc
    if not distances:
        raise ScenarioError("'distances_bohr' listesi boş olamaz")
    if any(d <= 0.0 for d in distances):
        raise ScenarioError(f"'distances_bohr' değerleri pozitif olmalı: {distances}")
    if len(set(distances)) != len(distances):
        raise ScenarioError(
            f"'distances_bohr' içinde tekrarlanan mesafe var: {distances}. "
            "Aynı geometriyi iki kez çözmek yalnızca süre harcar."
        )

    try:
        n_neighbour = int(raw.get("n_neighbour", 2))
        seeds = int(raw.get("seeds", 12))
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"'n_neighbour'/'seeds' tam sayı olmalı: {exc}") from exc
    if not 1 <= n_neighbour <= ring_size // 2:
        raise ScenarioError(
            f"'n_neighbour' 1 ile {ring_size // 2} arasında olmalı, verilen: "
            f"{n_neighbour}. Halka mesafesi {ring_size // 2}'de doyar."
        )
    if seeds < 1:
        raise ScenarioError(f"'seeds' en az 1 olmalı, verilen: {seeds}")

    return DownfoldScenario(
        name=str(raw["name"]),
        ring_size=ring_size,
        distances_bohr=distances,
        basis=str(raw.get("basis", "sto-3g")),
        n_neighbour=n_neighbour,
        seeds=seeds,
    )


#: YAML 1.2'nin ÜSTEL kayan nokta yazımı. PyYAML YAML 1.1 çözücüsüyle bunların
#: noktasız ya da işaretsiz olanlarını (`1e-6`, `5e-324`, `1.0e308`, `1e+100`)
#: TIRNAKSIZ yazılsa bile DİZE döndürür; Python'un `repr(float)`u da tam bu
#: biçimi üretir. Bunlar kullanıcının metni değil çözücünün açığıdır.
_YAML12_USTEL = re.compile(r"[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)[eE][-+]?[0-9]+")


def _yaml_sayisi(ad: str, deger: object) -> float:
    """YAML'dan gelen TEK bir gerçel sayı — bool ve TIRNAKLI metin REDDEDİLİR.

    `float()` ikisini de SESSİZCE çevirir; ölçüldü (11.10.2026, kamu yolu):
    `coupling: true` → 1.0 ve panel "J = 1 (girdi)" basıyordu, `coupling: '2'`
    → 2.0, `temperatures: [true]` → T = 1.0. YAML'da `true` (ve `yes`/`on`)
    bir bayraktır, tırnaklı `'2'` bir metindir.

    TEK İSTİSNA bilinçlidir: PyYAML'ın tırnaksız olsa da dize döndürdüğü üstel
    yazım (`_YAML12_USTEL`; `coupling: 1e-6`, F7 kilidinin `field: -5e-324`ü)
    sayı olarak okunur — o dizenin tırnaklı mı tırnaksız mı yazıldığı
    `safe_load`dan sonra ayırt edilemez ve reddetmek, eskiden doğru okunan
    YAML'ları kırardı. Dize tırnaksız yazılınca çözücünün ZATEN sayı yapacağı
    bir biçimdeyse (`'2'`, `'0.1'`, `'1.0e-6'`) tırnak bilinçlidir ve reddedilir.
    Sonluluk BURADA denetlenmez: her alanın adını anan kendi kapısı vardır.
    Motor katmanındaki eşi `engine/spin/factory.py::_skaler`dır.
    """
    if (
        isinstance(deger, str)
        and _YAML12_USTEL.fullmatch(deger)
        and isinstance(yaml.safe_load(deger), str)
    ):
        return float(deger)
    if isinstance(deger, bool) or not isinstance(deger, (int, float)):
        raise ScenarioError(
            f"'{ad}' bir sayı olmalı, verilen: {deger!r} ({type(deger).__name__}). "
            "bool ve tırnaklı metin sessizce sayıya çevrilmez (true → 1.0, '2' → 2.0)."
        )
    try:
        return float(deger)
    except OverflowError as exc:
        raise ScenarioError(f"'{ad}' float aralığını aşıyor: {deger!r}") from exc


def _yaml_tam_sayisi(ad: str, deger: object) -> int:
    """YAML'dan gelen bir TAM SAYI — bool, metin ve kesirli sayı REDDEDİLİR.

    `int()` `true`yu 1'e, `'5'`i 5'e çevirir ve `4.7`yi 4'e KESER; ölçüldü
    (11.10.2026, kamu yolu): `n_sweep: true` → 1 ölçüm, `seed: '5'` → 5,
    `lattice_size: 4.7` → 4. Tam değerli bir float (`1000.0`) kabul edilir —
    `hoppings[].cell` kapısının kuralı.
    """
    if (
        isinstance(deger, bool)
        or not isinstance(deger, (int, float))
        or (isinstance(deger, float) and not deger.is_integer())
    ):
        raise ScenarioError(
            f"'{ad}' tam sayı olmalı, verilen: {deger!r} ({type(deger).__name__})"
        )
    return int(deger)


def _load_spin_scenario(raw: dict, file_path: Path) -> SpinScenario:
    """Spin Monte Carlo senaryosu.

    Yükleyici, motorun ÜÇ kontrolünü (çift `L`, Wolff'un `h = 0` ve
    `n_component = 1` sınırları) BİLEREK tekrarlar ve bu bir kopya değil bir
    KATMAN AYRIMIDIR — `_load_downfold_scenario`daki gerekçenin aynısı:
    burada yakalanan şey YAML'ın hatasıdır ve hata hiçbir sweep koşmadan
    ÖNCE verilir; motorda yakalanan şey ÇAĞRININ hatasıdır. İkisi ayrı tür
    fırlatır (`ScenarioError` ile `InvalidInputError`/`UnsupportedSystemError`)
    ve testler bu farkı kilitler.

    Yükleyiciye ÖZGÜ olanlar: boş / tekrarlı / pozitif olmayan sıcaklık
    listesi, `seed`in zorunluluğu, ve `heisenberg_step`in Ising ile birlikte
    verilemeyeceği. Bunların motorda karşılığı yoktur çünkü motor tek bir
    sıcaklıkla ve zaten kurulmuş bir modelle çağrılır.

    `J <= 0` REDDEDİLİYOR VE BU BİR KAPSAM KARARIDIR, bir sınır değil.
    Modülün dört çıpasının dördü de (Onsager, kritik üs, Mermin-Wagner,
    modüller arası bağ) FERROMANYETİKTİR. Metropolis çekirdeği `J < 0`
    koşabilir — antiferromanyet meşru bir fiziktir — ama bu modül onun için
    hiçbir doğrulama taşımaz, dolayısıyla YAML'dan ulaşılabilir olması
    doğrulanmamış bir yolu sevk etmek olurdu. Wolff'ta ise sınır fiziksel:
    `p = 1 - exp(-2J/T)` negatife düşer ve algoritma sessizce SONSUZ
    SICAKLIĞI örnekler. Aynı karar `build_spin`de Python yolu için de
    uygulanır. `coupling: .inf` 27.09.2026'ya kadar bu kapıdan GEÇİYORDU
    (`not coupling > 0` +inf'i kabul eder) ve hata ancak enerji serisinde,
    coupling adını anmadan geliyordu; artık `coupling`, `field` ve
    `heisenberg_step`in sonluluğu da denetlenir. 27.09.2026'da AÇIK KALAN
    üç kapı 11.10.2026'da KAPANDI (her biri önce kamu yolundan yeniden
    ölçüldü): (1) bool/tırnaklı metin artık sayıya çevrilmez — `coupling:
    true` → 1.0 ve `'2'` → 2.0 geçiyordu, aynısı her sayısal alanda
    (`_yaml_sayisi`, `_yaml_tam_sayisi`; `md/factory.py`deki Kelvin reddinin
    karşılığı; PyYAML'ın üstel yazım açığı bilinçli istisna);
    (2) sonlu ama TAŞAN `coupling`/`field` adıyla düşer — Metropolis'te
    E_max TAŞMA KAPISINDA (n_sweep · E_max² sonlu), HER algoritmada 2J
    kapısında (Wolff'un bağ olasılığı 2J'yi hesaplar; kapının ilk sürümü
    Wolff'u "enerji ölçmez" diye dışarıda bırakmıştı ve coupling >= 9e307
    orada <|m|> = 1, tau = inf basıyordu); (3) Python yolunda (`build_spin`, `provenance_for`) `field` ve
    `heisenberg_step` sonluluğu, sayısal alanların bool/metin reddi ve taşma
    kapısı da var (`InvalidInputError`). AYRI ve AÇIK (taşma kapısı YAKALAMAZ,
    ölçüldü 11.10.2026, L = 8, T = 2, 64 ölçüm): enerji serisi SABİT iken ısı
    sığasının <E²> − <E>² farkı iptal hatası verir — `field` = 1e10'da
    C = 262144, 1e100'de C = −2.7e185 (fizik: 0); kusur `heat_capacity`dedir.
    """
    unknown = set(raw) - _SPIN_TOP_LEVEL
    if unknown:
        raise ScenarioError(
            f"Bilinmeyen anahtar(lar): {sorted(unknown)}. "
            f"Geçerli anahtarlar: {sorted(_SPIN_TOP_LEVEL)}"
        )
    missing = _SPIN_REQUIRED - set(raw)
    if missing:
        raise ScenarioError(f"Eksik anahtar(lar): {sorted(missing)}")

    model = str(raw["model"])
    if model not in SPIN_MODEL_KINDS:
        raise ScenarioError(
            f"Bilinmeyen spin modeli: {model!r}. "
            f"Geçerli modeller: {list(SPIN_MODEL_KINDS)}"
        )
    algorithm = str(raw["algorithm"])
    if algorithm not in SPIN_ALGORITHM_KINDS:
        raise ScenarioError(
            f"Bilinmeyen örnekleme algoritması: {algorithm!r}. "
            f"Geçerli algoritmalar: {list(SPIN_ALGORITHM_KINDS)}"
        )

    lattice_size = _yaml_tam_sayisi("lattice_size", raw["lattice_size"])
    n_sweep = _yaml_tam_sayisi("n_sweep", raw["n_sweep"])
    n_burn = _yaml_tam_sayisi("n_burn", raw["n_burn"])
    seed = _yaml_tam_sayisi("seed", raw["seed"])

    sicakliklar = raw["temperatures"]
    if not isinstance(sicakliklar, (list, tuple)):
        raise ScenarioError(
            f"'temperatures' bir sayı listesi olmalı, verilen: {sicakliklar!r}"
        )
    temperatures = tuple(
        _yaml_sayisi(f"temperatures[{i}]", x) for i, x in enumerate(sicakliklar)
    )

    coupling = _yaml_sayisi("coupling", raw.get("coupling", 1.0))
    field = _yaml_sayisi("field", raw.get("field", 0.0))
    heisenberg_step = _yaml_sayisi("heisenberg_step", raw.get("heisenberg_step", 0.6))
    if not (math.isfinite(field) and math.isfinite(heisenberg_step)):
        raise ScenarioError(
            f"'field' ve 'heisenberg_step' sonlu olmalı, verilen: field = {field}, "
            f"heisenberg_step = {heisenberg_step}. Sonlu olmayan 'field' hatayı ancak "
            "enerji serisinde, adını anmadan verirdi; 'heisenberg_step' ise öneriyi "
            "SESSİZCE dondururdu (<|m|> = 1.0)."
        )

    if lattice_size < 2:
        raise ScenarioError(
            f"'lattice_size' en az 2 olmalı, verilen: {lattice_size}. "
            "L = 1'de bir site kendi komşusudur."
        )
    if not temperatures:
        raise ScenarioError("'temperatures' listesi boş olamaz")
    if any(not math.isfinite(t) or t <= 0.0 for t in temperatures):
        raise ScenarioError(
            f"'temperatures' değerleri pozitif ve sonlu olmalı: {temperatures}. "
            "Sıcaklık enerji birimindedir (k_B = 1) ve sıfır bölen olarak Metropolis "
            "üstelinin içine girer."
        )
    if len(set(temperatures)) != len(temperatures):
        raise ScenarioError(
            f"'temperatures' içinde tekrarlanan sıcaklık var: {temperatures}. "
            "Aynı noktayı iki kez örneklemek yalnızca süre harcar."
        )
    if n_sweep < 1:
        raise ScenarioError(f"'n_sweep' en az 1 olmalı, verilen: {n_sweep}")
    if n_burn < 0:
        raise ScenarioError(f"'n_burn' negatif olamaz, verilen: {n_burn}")
    if not (math.isfinite(coupling) and coupling > 0.0):
        raise ScenarioError(
            f"'coupling' (J) sonlu ve pozitif olmalı, verilen: {coupling}. Bu modülün "
            "dört çıpası da ferromanyetiktir ve J <= 0 için hiçbir doğrulama "
            "taşımaz; Wolff'ta ayrıca p = 1 - exp(-2J/T) negatife düşer ve "
            "algoritma sessizce sonsuz sıcaklığı örnekler."
        )
    # TAŞMA KAPISI (11.10.2026): sonlu ama TAŞAN coupling/field. Metropolis'in
    # enerji serisi panelde ısı sığasının KARE ortalamasına girer, yani
    # n_sweep · E_max² sonlu kalmalıdır; E_max = L²(2J + |h|) bir konfigürasyonun
    # alabileceği en büyük |E|'dir (2L² bağ, L² site, birim spin) ve adımın
    # |ΔE| <= 8J + 2|h| sınırını da kapsar. ÖLÇÜLDÜ (kapı yokken, kamu yolu,
    # L = 8, 64 ölçüm, T = 2J): J = 1e100 / 1e150'de <|m|>, U, C, tau J = 1 ile
    # aynı; 1e152'de C = inf, 1e153 ... 1e300'de C = nan (rc = 0, uyarısız);
    # 1e307'de "enerji serisi sonlu olmali" (adsız, sweep'lerden SONRA). `field`
    # (T = 2) 1e152 / 1e153 / 1e307'de aynı üç biçimde düşüyordu. Bu E_max
    # kapısı yalnız Metropolis'tedir (Wolff enerji serisi kaydetmez); Wolff'un
    # kendi taşması aşağıdaki 2J kapısındadır. Tam sayı alanları float'a
    # çevrilirken taşarsa (`n_sweep` > ~1.8e308, `lattice_size` > ~1.3e154)
    # ham `OverflowError` yerine alanın adı söylenir.
    if algorithm == "metropolis":
        try:
            tavan = math.sqrt(sys.float_info.max / n_sweep)
        except OverflowError as exc:
            raise ScenarioError(
                f"'n_sweep' float aralığını aşıyor (~2^{n_sweep.bit_length()}); "
                "taşma kapısı ölçüm sayısını float olarak kullanır."
            ) from exc
        try:
            e_max = lattice_size * lattice_size * (2.0 * coupling + abs(field))
        except OverflowError as exc:
            raise ScenarioError(
                f"'lattice_size' çok büyük: L² float aralığını aşıyor "
                f"(L ~2^{lattice_size.bit_length()})."
            ) from exc
        if not e_max <= tavan:
            raise ScenarioError(
                f"'coupling'/'field' çok büyük, enerji ölçeği taşıyor: E_max = L²(2J + |h|) = "
                f"{e_max!r} (L = {lattice_size}, coupling = {coupling!r}, field = "
                f"{field!r}); ısı sığası {n_sweep} ölçümde E²'yi toplar ve E_max en "
                f"çok {tavan:.6g} olabilir. Taşan bir koşum C'yi sessizce inf/nan "
                "basar ya da sweep'lerden SONRA adsız bir hatayla düşer."
            )
    # 2J KAPISI (HER algoritma, 11.10.2026): Wolff'un bağ olasılığı
    # p = 1 - exp(-2J/T) 2J'yi hesaplar. ÖLÇÜLDÜ (doğrulayıcı, kamu yolu, kapı
    # yokken, kT/J = 1.7): coupling >= 9e307'de 2J = inf, p = 1 ve koşum rc = 0
    # ile <|m|> = 1.000000, tau = inf basıyordu; 8e307'de sonuç J = 1 ile aynı
    # (0.964844). 2J/T'nin taşması (çok küçük T) ise p = 1'i verir ki o doğru
    # limittir; kapı yalnız 2J'yi ister. Metropolis'te yukarıdaki E_max kapısı
    # bunu zaten kapsar.
    if not math.isfinite(2.0 * coupling):
        raise ScenarioError(
            f"'coupling' çok büyük: 2J = {2.0 * coupling} taşıyor (coupling = "
            f"{coupling!r}, en çok {sys.float_info.max / 2.0!r}). Wolff'un bağ "
            "olasılığı p = 1 - exp(-2J/T) o zaman sessizce 1 olur: her adımda "
            "bütün küme döner, <|m|> = 1 ve tau = inf basılır."
        )
    if "heisenberg_step" in raw and model != "heisenberg":
        raise ScenarioError(
            "'heisenberg_step' yalnızca model 'heisenberg' iken verilebilir; "
            f"verilen model: {model!r}. Ising'de öneri işaret çevirmedir ve "
            "adım parametresi hiçbir şeyi değiştirmez."
        )
    if heisenberg_step <= 0.0:
        raise ScenarioError(
            f"'heisenberg_step' pozitif olmalı, verilen: {heisenberg_step}"
        )

    if algorithm == "metropolis" and lattice_size % 2 != 0:
        raise ScenarioError(
            f"'metropolis' ÇİFT 'lattice_size' gerektirir, verilen: "
            f"{lattice_size}. Tek L'de periyodik kare kafes iki parçalı "
            "değildir ve checkerboard maskesi ayrıntılı dengeyi bozar."
        )
    if algorithm == "wolff":
        if model != "ising":
            raise ScenarioError(
                f"'wolff' yalnızca 'ising' modelini koşturur, verilen model: "
                f"{model!r}. Küme kurulumu 'aynı işaret' kavramına dayanır ve "
                "sürekli spin uzayında karşılığı (yansıtma düzlemi) bu depoda "
                "yazılmadı."
            )
        if field != 0.0:
            raise ScenarioError(
                f"'wolff' yalnızca 'field' = 0 ile koşar, verilen: {field}. "
                "Küme çevirme alan terimini ihlal eder: çevrilen kümenin alan "
                "enerjisi değişir ama kabul olasılığı 1 kalır."
            )

    return SpinScenario(
        name=str(raw["name"]),
        model=model,
        algorithm=algorithm,
        lattice_size=lattice_size,
        temperatures=temperatures,
        n_sweep=n_sweep,
        n_burn=n_burn,
        seed=seed,
        coupling=coupling,
        field=field,
        heisenberg_step=heisenberg_step,
    )


def _load_em_scenario(raw: dict, file_path: Path) -> EMScenario:
    """Maxwell FDTD senaryosu — ve PENCERENİN TAM BEYANI.

    Bu yükleyici, motorun iki kontrolünü (2B Courant sınırı, FFT penceresi
    adı) BİLEREK tekrarlar ve bu bir kopya değil bir KATMAN AYRIMIDIR —
    `_load_downfold_scenario` ve `_load_spin_scenario`daki gerekçenin aynısı:
    burada yakalanan şey YAML'ın hatasıdır ve hata hiçbir FDTD adımı
    koşmadan ÖNCE verilir; motorda yakalanan şey ÇAĞRININ hatasıdır. İkisi
    ayrı tür fırlatır (`ScenarioError` ile `InvalidInputError`).

    Yükleyiciye ÖZGÜ olanlar — motorda karşılıkları YOKTUR, çünkü motor tek
    bir yarıçapla ve zaten kurulmuş bir ortamla çağrılır:

    * **`radii` en az dört basamak.** Sayım kimliği: n değer → n−3 Richardson
      noktası. Üç basamaklı bir merdiven `p_inf = None` döndürür, yani panelin
      yakınsama satırı YAPISAL OLARAK "YARGILANAMAZ" olurdu. Böyle bir
      senaryoyu sevk etmek, hiçbir zaman hüküm veremeyecek bir kadranı
      göstermek olurdu.
    * **`ratio` `radii` ile tutarlı olmalı.** `p` tam olarak
      `log(d_i/d_{i+1})/log(ratio)` ile okunur; oran uymuyorsa çıkan sayı bir
      mertebe DEĞİLDİR. Aynı kapı `observable/em_obs.py::_merdiven`de de var
      ama orası `spacings` (= 1/R) görür, burası `radii` görür.
    * **`radii` KESİN ARTAN.** Merdiven kabadan inceye sıralanmak zorundadır
      (`spacings = 1/R` kesin azalan); ters sıralı bir merdiven `p`nin
      işaretini çevirir ve imkân kapılarından SESSİZCE geçerdi.

    `dx` VE `npml` BURADA GEÇERLİ ANAHTAR DEĞİLDİR ve ikisi de "bilinmeyen
    anahtar" hatası verir. Gerekçeleri `EMScenario` dokstringinde ölçülmüş
    hâlde yazılı: `npml` bağlı değildir, `dx` ise bağlıydı ama SESSİZCE
    YANLIŞ çalışıyordu (tepe sayısı `Δx` = 1/0.5/2 için 20/13 → 4/4 → 31/43)
    ve bugün motorda da kapalı (`engine/em/grid.py::dx_kapisi`).
    """
    unknown = set(raw) - _EM_TOP_LEVEL
    if unknown:
        raise ScenarioError(
            f"Bilinmeyen anahtar(lar): {sorted(unknown)}. "
            f"Geçerli anahtarlar: {sorted(_EM_TOP_LEVEL)}"
        )
    missing = _EM_REQUIRED - set(raw)
    if missing:
        raise ScenarioError(f"Eksik anahtar(lar): {sorted(missing)}")

    try:
        radii = tuple(float(x) for x in raw["radii"])
    except TypeError as exc:
        raise ScenarioError(f"'radii' bir sayı listesi olmalı: {exc}") from exc
    except ValueError as exc:
        raise ScenarioError(
            f"'radii' listesindeki bir eleman sayıya dönüştürülemedi: {exc}"
        ) from exc

    try:
        ratio = float(raw["ratio"])
        courant = float(raw["courant"])
        step_factor = float(raw["step_factor"])
        wavelength_per_radius = float(raw["wavelength_per_radius"])
        sgf = float(raw["sgf"])
        peak_floor = float(raw["peak_floor"])
        eps_r = float(raw.get("eps_r", 1.0))
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"EM senaryosunun sayısal alanları hatalı: {exc}") from exc

    try:
        detectors = int(raw["detectors"])
        pad = int(raw["pad"])
        margin = int(raw.get("margin", 3))
    except (TypeError, ValueError) as exc:
        raise ScenarioError(
            f"'detectors'/'pad'/'margin' tam sayı olmalı: {exc}"
        ) from exc

    window_fn = str(raw["window_fn"])
    if window_fn not in EM_WINDOW_FNS:
        raise ScenarioError(
            f"Bilinmeyen FFT penceresi: {window_fn!r}. "
            f"Geçerli pencereler: {list(EM_WINDOW_FNS)}"
        )

    if len(radii) < EM_MIN_DEPTH:
        raise ScenarioError(
            f"'radii' en az {EM_MIN_DEPTH} basamak olmalı, verilen: {len(radii)}. "
            "n değer n−1 fark, n−2 mertebe ve n−3 Richardson noktası verir; "
            "dört, belirsizliğin ölçülebildiği EN KÜÇÜK derinliktir ve daha "
            "sığ bir merdiven panelde yapısal olarak YARGILANAMAZ basardı."
        )
    if any(not math.isfinite(r) or r <= 0.0 for r in radii):
        raise ScenarioError(f"'radii' değerleri pozitif ve sonlu olmalı: {radii}")
    if any(b <= a for a, b in zip(radii, radii[1:])):
        raise ScenarioError(
            f"'radii' KESİN ARTAN olmalı (kabadan inceye), verilen: {radii}. "
            "Ters sıralı bir merdiven `p`nin işaretini çevirir ve imkân "
            "kapılarından sessizce geçerdi."
        )
    if not ratio > 1.0 or not math.isfinite(ratio):
        raise ScenarioError(f"'ratio' 1'den büyük ve sonlu olmalı, verilen: {ratio}")
    olculen = [b / a for a, b in zip(radii, radii[1:])]
    if any(abs(o - ratio) > 1e-9 * ratio for o in olculen):
        raise ScenarioError(
            f"'radii' oranı 'ratio' ile tutarlı değil: ölçülen {olculen}, "
            f"beyan edilen {ratio}. `p` tam olarak log(d_i/d_(i+1))/log(ratio) "
            "ile okunur; oran uymuyorsa çıkan sayı bir MERTEBE DEĞİLDİR."
        )
    if not courant > 0.0:
        raise ScenarioError(f"'courant' (S) pozitif olmalı, verilen: {courant}")
    if courant > _EM_COURANT_LIMIT_2D:
        raise ScenarioError(
            f"'courant' 2B kararlılık sınırını aşıyor: {courant} > "
            f"{_EM_COURANT_LIMIT_2D} (= 1/√2). Bu sınır 1B'ninkinden FARKLIDIR "
            "ve 1B kapısı DEVRALINAMAZ — ölçüldü: S = 0.9 1B'de geçiyor, "
            "2B'de şema patlıyor."
        )
    if not step_factor > 0.0 or not math.isfinite(step_factor):
        raise ScenarioError(
            f"'step_factor' pozitif ve sonlu olmalı, verilen: {step_factor}"
        )
    if not wavelength_per_radius > 0.0 or not math.isfinite(wavelength_per_radius):
        raise ScenarioError(
            "'wavelength_per_radius' pozitif ve sonlu olmalı, verilen: "
            f"{wavelength_per_radius}"
        )
    if not sgf > 0.0 or not math.isfinite(sgf):
        raise ScenarioError(f"'sgf' pozitif ve sonlu olmalı, verilen: {sgf}")
    if detectors < 2:
        raise ScenarioError(
            f"'detectors' en az 2 olmalı, verilen: {detectors}. Tek dedektör "
            "ölçüldü ve merdiveni DEĞİL paneli düşürüyor: tepe sayısı 16'dan "
            "11'e iniyor ve bir oran işaret değiştiriyor."
        )
    if pad < 1:
        raise ScenarioError(f"'pad' en az 1 olmalı, verilen: {pad}")
    if not 0.0 < peak_floor < 1.0:
        raise ScenarioError(
            f"'peak_floor' (0, 1) aralığında olmalı, verilen: {peak_floor}"
        )
    if not eps_r > 0.0 or not math.isfinite(eps_r):
        raise ScenarioError(f"'eps_r' pozitif ve sonlu olmalı, verilen: {eps_r}")
    # ETKİN COURANT (28.09): ham S kapısı yalnız eps_r >= 1'de yeter; eps_r < 1'de
    # dalga c/√eps_r ile vakumdan HIZLIDIR. Kapı yokken eps_r = 0.25, S = 0.5
    # yükleniyor, alanlar inf/nan'a taşıyor ve CLI nedeni anmadan "iki tepe
    # bulunamadı" diyordu. Python yolundaki eşi: `factory._courant_kapisi`.
    # min(eps_r, 1) EŞDEĞER MUTANTTIR (silinse davranış bit bit aynı: ham kapı önce
    # koşar); kapıyı sıradan bağımsız kılmak için tutuldu (fabrika `0 < eps < 1` yazar).
    etkin = courant / math.sqrt(min(eps_r, 1.0))
    if etkin > _EM_COURANT_LIMIT_2D:
        k = 5 - math.floor(math.log10(tavan := _EM_COURANT_LIMIT_2D * math.sqrt(eps_r)))
        s_max = math.floor(tavan * 10**k) / 10**k  # 6 haneye AŞAĞI kesilir: öneri geçmeli
        raise ScenarioError(
            f"eps_r = {eps_r!r} < 1 ortamda dalga vakumdan hızlıdır ve etkin Courant "
            f"S/√eps_r = {etkin!r} 2B sınırını ({_EM_COURANT_LIMIT_2D!r}) aşıyor "
            f"(courant = {courant!r}); en çok S = {s_max!r} olabilir."
        )
    if margin < 1:
        raise ScenarioError(
            f"'margin' en az 1 olmalı, verilen: {margin}. Merkez düğümü ile "
            "dış ızgara kenarı arasında en az bir hücre kalmalı."
        )

    return EMScenario(
        name=str(raw["name"]),
        radii=radii,
        ratio=ratio,
        courant=courant,
        step_factor=step_factor,
        wavelength_per_radius=wavelength_per_radius,
        sgf=sgf,
        detectors=detectors,
        pad=pad,
        window_fn=window_fn,
        peak_floor=peak_floor,
        eps_r=eps_r,
        margin=margin,
    )


def _grav_richardson(raw: dict, anahtar: str) -> tuple[float, float]:
    """Bir Richardson çiftini okur ve KENDİ KAPILARINI koşar.

    Çift ARTAN olmak zorundadır ve bu bir biçim tercihi değildir: dışdeğerleme
    `(f1*h2 - f2*h1)/(h2 - h1)` ile `h = 1/basamak` üzerinden yapılır, yani
    eşit basamaklarda sıfıra bölmedir ve ters sıralı bir çift kaba basamağı
    "ince" sayar. Motor eşitliği zaten reddediyor (`InvalidInputError`), ama
    buradaki kapı YAML'ın hatasını hiçbir kareleme koşmadan önce yakalar —
    `_load_em_scenario`ın Courant kapısıyla aynı katman ayrımı.
    """
    deger = raw[anahtar]
    if not isinstance(deger, (list, tuple)) or len(deger) != 2:
        raise ScenarioError(
            f"'{anahtar}' iki elemanlı bir liste olmalı (alt, üst), "
            f"verilen: {deger!r}"
        )
    try:
        alt, ust = float(deger[0]), float(deger[1])
    except (TypeError, ValueError) as exc:
        raise ScenarioError(
            f"'{anahtar}' basamakları sayıya dönüştürülemedi: {exc}"
        ) from exc
    if not math.isfinite(alt) or not math.isfinite(ust) or alt <= 0.0 or ust <= 0.0:
        raise ScenarioError(
            f"'{anahtar}' basamakları pozitif ve sonlu olmalı, verilen: "
            f"({alt}, {ust})"
        )
    if not alt < ust:
        raise ScenarioError(
            f"'{anahtar}' KESİN ARTAN olmalı (alt < üst), verilen: "
            f"({alt}, {ust}). Eşit basamaklarda `1/p`de dışdeğerleme sıfıra "
            "bölmedir; ters sıralı bir çift ise kaba basamağı ince sayar."
        )
    return alt, ust


def _load_grav_scenario(raw: dict, file_path: Path) -> GravScenario:
    """Yerçekimi senaryosu — ve PENCERENİN TAM BEYANI.

    Bu yükleyici motorun üç kontrolünü (ayar adı, integratör adı, `e > 0`
    kapısı) BİLEREK tekrarlar ve bu bir kopya değil bir KATMAN AYRIMIDIR —
    `_load_em_scenario`, `_load_spin_scenario` ve `_load_downfold_scenario`
    ile aynı gerekçe: burada yakalanan şey YAML'ın hatasıdır ve hata hiçbir
    kareleme, hiçbir `solve_ivp` adımı koşmadan ÖNCE verilir; motorda
    yakalanan şey ÇAĞRININ hatasıdır. İkisi ayrı tür fırlatır
    (`ScenarioError` ile `InvalidInputError`) ve testler bu farkı kilitler.

    Yükleyiciye ÖZGÜ olanlar — motorda karşılıkları YOKTUR:

    * **`e = 0` ile `e = 1` İKİ AYRI SEBEPLE reddedilir.** `e = 0`da `u₁ = u₂`
      bir DAİREDİR ve orada ölçülecek bir perihel kayması yoktur (motorun
      kapısı da sözleşmeseldir, sayısal değil); `e ≥ 1` ise bağlı bir yörünge
      değildir ve dönüş noktaları gerçek kalmaz.
    * **`gamma`/`beta` REDDEDİLMEZ.** GR olmayan bir seçim geçerli bir
      senaryodur — reddedilseydi §4.3'ün hata çubuğu makinesi (`λ = 0`, yani
      `β = 2 + 2γ`) senaryo yolundan hiç görünmezdi. Panel bunun yerine
      `Kuram seçici` satırında ne çözüldüğünü YAZAR.
    * **`richardson` ve `deflection_richardson` AYRI kapılardan geçer**
      (`_grav_richardson`), çünkü ikisi ayrı borunun penceresidir.

    `lam` BURADA GEÇERLİ BİR ANAHTAR DEĞİLDİR ve "bilinmeyen anahtar" hatası
    verir: tek kaynak `(gamma, beta)`dır, `λ` motorda türetilmiş bir
    `@property`dir. İki alan aynı şeyi belirlerse sessiz bir kazanan doğar.
    """
    unknown = set(raw) - _GRAV_TOP_LEVEL
    if unknown:
        raise ScenarioError(
            f"Bilinmeyen anahtar(lar): {sorted(unknown)}. "
            f"Geçerli anahtarlar: {sorted(_GRAV_TOP_LEVEL)}"
        )
    missing = _GRAV_REQUIRED - set(raw)
    if missing:
        raise ScenarioError(f"Eksik anahtar(lar): {sorted(missing)}")

    try:
        gamma = float(raw["gamma"])
        beta = float(raw["beta"])
        mass = float(raw["mass"])
        eccentricity = float(raw["eccentricity"])
        second_order_p = float(raw["second_order_p"])
        trajectory_p = float(raw["trajectory_p"])
        nbody_rtol = float(raw["nbody_rtol"])
        nbody_atol = float(raw["nbody_atol"])
    except (TypeError, ValueError) as exc:
        raise ScenarioError(
            f"Yerçekimi senaryosunun sayısal alanları hatalı: {exc}"
        ) from exc

    try:
        trajectory_points = int(raw["trajectory_points"])
    except (TypeError, ValueError) as exc:
        raise ScenarioError(
            f"'trajectory_points' tam sayı olmalı: {exc}"
        ) from exc

    gauge = str(raw["gauge"])
    if gauge not in GRAV_GAUGES:
        raise ScenarioError(
            f"Bilinmeyen ayar (gauge): {gauge!r}. "
            f"Geçerli ayarlar: {list(GRAV_GAUGES)}"
        )
    nbody_method = str(raw["nbody_method"])
    if nbody_method not in GRAV_NBODY_METHODS:
        raise ScenarioError(
            f"Bilinmeyen N-cisim integratörü: {nbody_method!r}. "
            f"Geçerli integratörler: {list(GRAV_NBODY_METHODS)}"
        )

    for ad, deger in (("gamma", gamma), ("beta", beta)):
        if not math.isfinite(deger):
            raise ScenarioError(f"'{ad}' sonlu olmalı, verilen: {deger}")
    if not math.isfinite(mass) or mass <= 0.0:
        raise ScenarioError(f"'mass' pozitif ve sonlu olmalı, verilen: {mass}")
    if not 0.0 < eccentricity < 1.0:
        raise ScenarioError(
            f"'eccentricity' (0, 1) aralığında olmalı, verilen: {eccentricity}. "
            "e = 0'da u₁ = u₂ bir DAİREDİR ve orada ölçülecek bir perihel "
            "kayması yoktur; e ≥ 1 bağlı bir yörünge değildir."
        )
    for ad, deger in (("second_order_p", second_order_p),
                      ("trajectory_p", trajectory_p)):
        if not math.isfinite(deger) or deger <= 0.0:
            raise ScenarioError(f"'{ad}' pozitif ve sonlu olmalı, verilen: {deger}")
    for ad, deger in (("nbody_rtol", nbody_rtol), ("nbody_atol", nbody_atol)):
        if not math.isfinite(deger) or not 0.0 < deger < 1.0:
            raise ScenarioError(
                f"'{ad}' (0, 1) aralığında ve sonlu olmalı, verilen: {deger}"
            )
    if trajectory_points < 2:
        raise ScenarioError(
            f"'trajectory_points' en az 2 olmalı, verilen: {trajectory_points}. "
            "Bu bir ÖRNEKLEME yoğunluğudur, doğruluk düğmesi DEĞİL; doğruluğu "
            "çözücünün kendi `rtol`/`atol`ü belirler."
        )

    return GravScenario(
        name=str(raw["name"]),
        gauge=gauge,
        gamma=gamma,
        beta=beta,
        mass=mass,
        eccentricity=eccentricity,
        richardson=_grav_richardson(raw, "richardson"),
        second_order_p=second_order_p,
        deflection_richardson=_grav_richardson(raw, "deflection_richardson"),
        trajectory_p=trajectory_p,
        trajectory_points=trajectory_points,
        nbody_rtol=nbody_rtol,
        nbody_atol=nbody_atol,
        nbody_method=nbody_method,
    )


def _load_bridge_scenario(raw: dict, file_path: Path) -> BridgeScenario:
    """Modüller arası yokluk haritası senaryosu — kenar tabanının PENCERESİ.

    Bu yükleyici motorun üç kontrolünü (`n_bolme`/`periyot_sayisi` pozitif,
    en az iki genlik, ikiz `a` pozitif ve sonlu) BİLEREK tekrarlar ve bu bir
    kopya değil bir KATMAN AYRIMIDIR — `_load_grav_scenario` ile aynı gerekçe:
    burada yakalanan şey YAML'ın hatasıdır ve hata hiçbir MD adımı koşmadan
    ÖNCE verilir; motorda (`bridge_obs.edge_noise_floor`/`omega_b`) yakalanan
    şey ÇAĞRININ hatasıdır. İkisi ayrı tür fırlatır (`ScenarioError` ile
    `InvalidInputError`) ve testler bu farkı kilitler.

    Yükleyiciye ÖZGÜ olanlar — motorda karşılıkları YOKTUR:

    * **`genlikler` KESİN ARTAN ve tekrarsız.** `A²`'de doğrusal uydurma
      iki farklı `A²` ister; tekrarlı bir liste `polyfit`e tekil bir
      tasarım matrisi verir ve uyarıyla "geçer".
    * **`a` GEÇERLİ BİR ANAHTAR DEĞİLDİR** ve "bilinmeyen anahtar" hatası
      verir: A1'in fizik-kapatma merdiveni çağrılanın tasarım kararıdır,
      YAML'dan seçilemez (Modül 6'nın `lam` anahtarı dersinin aynısı: iki
      yer aynı şeyi belirlerse sessiz bir kazanan doğar).
    """
    unknown = set(raw) - _BRIDGE_TOP_LEVEL
    if unknown:
        raise ScenarioError(
            f"Bilinmeyen anahtar(lar): {sorted(unknown)}. "
            f"Geçerli anahtarlar: {sorted(_BRIDGE_TOP_LEVEL)}"
        )
    missing = _BRIDGE_REQUIRED - set(raw)
    if missing:
        raise ScenarioError(f"Eksik anahtar(lar): {sorted(missing)}")

    try:
        a_ikiz = float(raw["a_ikiz"])
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"'a_ikiz' sayıya dönüştürülemedi: {exc}") from exc
    try:
        n_bolme = int(raw["n_bolme"])
        periyot_sayisi = int(raw["periyot_sayisi"])
    except (TypeError, ValueError) as exc:
        raise ScenarioError(
            f"'n_bolme'/'periyot_sayisi' tam sayı olmalı: {exc}"
        ) from exc
    try:
        genlikler = tuple(float(x) for x in raw["genlikler"])
    except TypeError as exc:
        raise ScenarioError(f"'genlikler' bir sayı listesi olmalı: {exc}") from exc
    except ValueError as exc:
        raise ScenarioError(
            f"'genlikler' listesindeki bir eleman sayıya dönüştürülemedi: {exc}"
        ) from exc

    if not math.isfinite(a_ikiz) or a_ikiz <= 0.0:
        raise ScenarioError(
            f"'a_ikiz' pozitif ve sonlu olmalı, verilen: {a_ikiz}. `a = 0` "
            "A1'in cebirsel noktasıdır ve A2'nin ikizi olamaz; A1 merdiveni "
            "zaten çağrılanın elindedir."
        )
    if n_bolme < 1:
        raise ScenarioError(f"'n_bolme' en az 1 olmalı, verilen: {n_bolme}")
    if periyot_sayisi < 1:
        raise ScenarioError(
            f"'periyot_sayisi' en az 1 olmalı, verilen: {periyot_sayisi}"
        )
    if len(genlikler) < 2:
        raise ScenarioError(
            f"'genlikler' en az iki genlik taşımalı, verilen: {genlikler}. "
            "A² uydurması tek noktadan doğru geçiremez."
        )
    if any(not math.isfinite(g) or g <= 0.0 for g in genlikler):
        raise ScenarioError(
            f"'genlikler' değerleri pozitif ve sonlu olmalı: {genlikler}"
        )
    if any(b <= a for a, b in zip(genlikler, genlikler[1:])):
        raise ScenarioError(
            f"'genlikler' KESİN ARTAN olmalı, verilen: {genlikler}. Tekrarlı "
            "bir liste `polyfit`e tekil bir tasarım matrisi verir."
        )

    return BridgeScenario(
        name=str(raw["name"]),
        a_ikiz=a_ikiz,
        n_bolme=n_bolme,
        periyot_sayisi=periyot_sayisi,
        genlikler=genlikler,
    )
