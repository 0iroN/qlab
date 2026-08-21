import dataclasses
import pytest

from qlab.errors import InvalidInputError, QLabError, UnitarityViolationError
from qlab.provenance import Provenance, StabilityVerdict


def make_provenance(**overrides) -> Provenance:
    defaults = dict(
        method="TDSE split-operator (Strang), 1B",
        approximations=("tek parçacık", "periyodik sınır koşulu"),
        neglected=("elektron-elektron etkileşimi", "spin"),
        expected_error="O(dt^2) ~ 4.0e-04 hartree",
        converged=True,
        diagnostics={"norm_deviation": 3.2e-13, "steps": 8000},
    )
    defaults.update(overrides)
    return Provenance(**defaults)


def test_provenance_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_provenance().method = "başka bir şey"


def test_as_lines_returns_expected_labels():
    labels = [label for label, _ in make_provenance().as_lines()]
    assert labels == ["Yöntem", "Yaklaşımlar", "İhmal edilen", "Beklenen hata", "Yakınsama"]


def test_as_lines_joins_lists_with_separator():
    assert dict(make_provenance().as_lines())["Yaklaşımlar"] == "tek parçacık · periyodik sınır koşulu"


def test_as_lines_marks_converged_state():
    assert dict(make_provenance(converged=True).as_lines())["Yakınsama"].startswith("✓")
    assert dict(make_provenance(converged=False).as_lines())["Yakınsama"].startswith("✗")


def test_as_lines_omits_stability_when_it_was_not_analysed():
    """TDSE sonuçları (ve stability=False ile koşulan HF sonuçları) kararlılık
    satırı BASMAMALI — analiz edilmemiş bir şeyi panelde göstermek, boş bir
    değeri "sorun yok" gibi okutur."""
    labels = [label for label, _ in make_provenance().as_lines()]
    assert "Kararlılık" not in labels


def test_as_lines_reports_a_stable_verdict():
    prov = make_provenance(
        stability=StabilityVerdict(
            stable=True, lowest_eigenvalue=2.93, n_rotations=1
        )
    )
    labels = [label for label, _ in prov.as_lines()]
    assert labels == [
        "Yöntem",
        "Yaklaşımlar",
        "İhmal edilen",
        "Beklenen hata",
        "Yakınsama",
        "Kararlılık",
    ]
    line = dict(prov.as_lines())["Kararlılık"]
    assert line.startswith("✓")
    assert "2.930e+00" in line


def test_as_lines_names_the_saddle_point_for_an_unstable_verdict():
    """Kararsızlık sessiz kalmamalı: panel satırı sonucun bir EYER NOKTASI
    olduğunu açıkça söylemeli, yalnızca bir sayı basmamalı."""
    prov = make_provenance(
        stability=StabilityVerdict(
            stable=False, lowest_eigenvalue=-2.93, n_rotations=1
        )
    )
    line = dict(prov.as_lines())["Kararlılık"]
    assert line.startswith("✗")
    assert "eyer" in line.lower()
    assert "-2.930e+00" in line


def test_as_lines_says_when_there_is_nothing_to_rotate():
    """Bos orbitali olmayan sistemde (He/STO-3G) döndürülecek çift yoktur;
    "kararlı" demek doğru ama bir özdeğer basmak yanlış olurdu."""
    prov = make_provenance(
        stability=StabilityVerdict(
            stable=True, lowest_eigenvalue=None, n_rotations=0
        )
    )
    line = dict(prov.as_lines())["Kararlılık"]
    assert line.startswith("✓")
    assert "dönme yok" in line


def test_stability_verdict_survives_asdict_for_json():
    """run.save() provenance'ı dataclasses.asdict + json.dumps ile yazar;
    iç içe dataclass düz sözlüğe dönüşmeli ve JSON'a yazılabilmeli."""
    import json

    prov = make_provenance(
        stability=StabilityVerdict(
            stable=False, lowest_eigenvalue=-2.93, n_rotations=1
        )
    )
    payload = json.loads(json.dumps(dataclasses.asdict(prov), ensure_ascii=False))
    assert payload["stability"] == {
        "stable": False,
        "lowest_eigenvalue": -2.93,
        "n_rotations": 1,
        # Modül 1C'de eklendi: kayıt HANGİ kanalın hükmü olduğunu taşımalı.
        # İki kanal aynı alanları döker; kanal adı olmadan diskteki iki kayıt
        # birbirinden AYIRT EDİLEMEZDİ.
        "channel": "ic",
    }


def test_asdict_carries_BOTH_stability_channels():
    """İki kanal ayrı alanlar olarak yazılmalı — birinin varlığı diğerini
    gizlememeli. `stability` "kararlı" derken `external_stability`in "eyer"
    demesi tipik durumdur (ölçülen örnek: H2 @ 2.00 Å)."""
    import json

    prov = make_provenance(
        stability=StabilityVerdict(
            stable=True, lowest_eigenvalue=2.55, n_rotations=1
        ),
        external_stability=StabilityVerdict(
            stable=False, lowest_eigenvalue=-1.60, n_rotations=1, channel="dis"
        ),
    )
    payload = json.loads(json.dumps(dataclasses.asdict(prov), ensure_ascii=False))
    assert payload["stability"]["channel"] == "ic"
    assert payload["stability"]["stable"] is True
    assert payload["external_stability"]["channel"] == "dis"
    assert payload["external_stability"]["stable"] is False


def test_as_lines_prints_the_external_channel_with_its_OWN_sentence():
    """Dış kanalın satırı iç kanalınkinden AYRI cümle kurmalı: aynı metni iki
    kez basmak, okuyucuya iki kanalı tek hüküm gibi gösterirdi."""
    prov = make_provenance(
        stability=StabilityVerdict(
            stable=True, lowest_eigenvalue=2.55, n_rotations=1
        ),
        external_stability=StabilityVerdict(
            stable=False, lowest_eigenvalue=-1.60, n_rotations=1, channel="dis"
        ),
    )
    satirlar = dict(prov.as_lines())
    assert satirlar["Kararlılık"].startswith("✓")
    assert "iç (RHF->RHF)" in satirlar["Kararlılık"]
    assert satirlar["Dış kararlılık"].startswith("✗")
    assert "SPİN EYERİ" in satirlar["Dış kararlılık"]
    assert "-1.600e+00" in satirlar["Dış kararlılık"]
    assert satirlar["Kararlılık"] != satirlar["Dış kararlılık"]


def test_as_lines_omits_the_external_channel_when_it_was_not_analysed():
    """"Analiz edilmedi" ile "kararlı" ayrımı ikinci kanalda da geçerli."""
    prov = make_provenance(
        stability=StabilityVerdict(
            stable=True, lowest_eigenvalue=2.55, n_rotations=1
        )
    )
    assert "Dış kararlılık" not in [label for label, _ in prov.as_lines()]


def test_unitarity_error_carries_diagnosis():
    err = UnitarityViolationError(step=412, norm_deviation=2.5e-6, dt=0.5)
    assert err.step == 412
    assert err.norm_deviation == pytest.approx(2.5e-6)
    assert err.dt == pytest.approx(0.5)
    assert "412" in str(err)
    assert "0.5" in str(err)


def test_error_hierarchy():
    assert issubclass(UnitarityViolationError, QLabError)
    assert issubclass(InvalidInputError, QLabError)


def test_unmasked_convergence_shows_norm_deviation():
    """Maskesiz (üniterlik kontrolü yapılan) koşular norm_deviation etiketi
    göstermelidir, bu kontrol kanıtıdır."""
    prov = make_provenance(
        converged=False,
        diagnostics={"norm_deviation": 1.5e-2, "steps": 100},
    )
    yakinsama_line = dict(prov.as_lines())["Yakınsama"]
    assert "norm_deviation=" in yakinsama_line
    assert "observed_norm_drop=" not in yakinsama_line


def test_masked_convergence_shows_observed_norm_drop():
    """Maskeli (üniterlik kontrolü yapılmayan) koşular observed_norm_drop
    etiketi göstermelidir, bu bilgilendirici ölçüdür ve kontrol kanıtı değildir."""
    prov = make_provenance(
        converged=True,
        diagnostics={"observed_norm_drop": 7.0e-2, "steps": 100},
    )
    yakinsama_line = dict(prov.as_lines())["Yakınsama"]
    assert "observed_norm_drop=" in yakinsama_line
    assert "norm_deviation=" not in yakinsama_line


def test_masked_case_cannot_be_misread_as_convergence_evidence():
    """Maskeli koşuda kontrol yapılmadığı için, panel okuyucu norm değerini
    yakınsama kanıtı olarak yanlış okumamalı. observed_norm_drop etiketinin
    varlığı ve norm_deviation'ın YOKLUĞU bunu garanti eder."""
    prov = make_provenance(
        converged=True,  # Maske varken always True (kontrol yapılmadığı için)
        diagnostics={"observed_norm_drop": 7.0e-2, "steps": 100},
    )
    yakinsama_line = dict(prov.as_lines())["Yakınsama"]
    # Başlangıç ✓ kontrol geçti anlamı değildir maskede, ama norm_deviation
    # görmemek kontrol yapılmadığını kanıtlar.
    assert yakinsama_line.startswith("✓")
    assert "norm_deviation" not in yakinsama_line
