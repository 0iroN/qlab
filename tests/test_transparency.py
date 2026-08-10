"""Şeffaflık paneli ve TDSE penceresinin testleri.

`format_panel` saf bir fonksiyon olduğu için GUI'siz doğrudan test edilir
(aşağıdaki ilk dört test). Ardından gelen testler ise Qt'nin ekransız
("offscreen") platform eklentisini kullanarak gerçek `TDSEWindow`'u bellekte
kurar, kaydırıcıyı programatik olarak hareket ettirir ve gerçek widget
durumunu (readout metni, panel metni) okur — yalnızca "içe aktarma hatasız"
gibi zayıf bir kanıtla yetinilmez.
"""

import dataclasses
import os
import re

import pytest

from qlab.app.transparency import format_panel
from qlab.provenance import Provenance

PROVENANCE = Provenance(
    method="TDSE split-operator (Strang), 1B, dt=0.02",
    approximations=("tek parçacık", "periyodik sınır koşulu (FFT)"),
    neglected=("elektron-elektron etkileşimi", "spin"),
    expected_error="O(dt^2) ~ 4.0e-04 hartree mertebesi",
    converged=True,
    diagnostics={"norm_deviation": 3.2e-13, "steps": 8000},
)


def test_panel_contains_every_label():
    text = format_panel(PROVENANCE)
    for label in ("Yöntem", "Yaklaşımlar", "İhmal edilen", "Beklenen hata", "Yakınsama"):
        assert label in text


def test_panel_contains_method_and_error():
    text = format_panel(PROVENANCE)
    assert "split-operator" in text
    assert "4.0e-04" in text


def test_panel_marks_failure_visibly():
    failed = Provenance(
        method="x",
        approximations=("a",),
        neglected=("b",),
        expected_error="c",
        converged=False,
        diagnostics={"norm_deviation": 1.0},
    )
    assert "✗" in format_panel(failed)


def test_panel_renders_the_stability_verdict_when_there_is_one():
    """Panel Provenance'ı olduğu gibi basar; kararlılık satırı eklendiğinde
    ayrıca bir uyarlama gerekmemeli. Bugün Qt penceresine yalnızca TDSE
    sonucu besleniyor (HF sonucu GUI'ye hiç gitmiyor), ama `format_panel`
    genel bir fonksiyondur ve bu yol testsiz kalmamalı."""
    from qlab.provenance import StabilityVerdict

    text = format_panel(
        dataclasses.replace(
            PROVENANCE,
            stability=StabilityVerdict(
                stable=False, lowest_eigenvalue=-2.92, n_rotations=1
            ),
        )
    )
    assert "Kararlılık" in text
    assert "-2.920e+00" in text
    assert "EYER NOKTASI" in text


def test_panel_lines_are_aligned():
    lines = [line for line in format_panel(PROVENANCE).splitlines() if line.strip()]
    columns = {line.index(":") for line in lines if ":" in line}
    assert len(columns) == 1, "etiket sütunu hizalı olmalı"


# ---------------------------------------------------------------------------
# Ekransız (offscreen) Qt testleri
#
# QT_QPA_PLATFORM burada, PySide6 içe aktarılmadan ÖNCE zorlanır: gösterge
# yoksa (CI, SSH oturumu) gerçek platform eklentisi başlatma sırasında çöker.
# `setdefault` kullanılır ki çağıran zaten farklı bir platform istemişse ona
# saygı duyulsun; aksi halde ekransız kurulum belirleyicidir (interaktif
# oturumda bile aynı davranır).
#
# Bu ortamda offscreen platformunun gerçekten çalıştığı önceden elle
# doğrulandı (bkz. görev raporu); bu yüzden aşağıdaki testlerin atlanması
# BEKLENMEZ — `pytest.mark.skipif` yalnızca platform gerçekten kurulamazsa
# devreye giren bir güvenlik ağıdır, her zaman atlayan bir test testten daha
# kötüdür.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# İçe aktarmalar bilinçli olarak `try` DIŞINDA. Bir süre hepsi probe'un
# içindeydi ve o hâliyle ÖLÇÜLDÜ: `qlab.app.main` modülü kaybolduğunda bu
# dosyanın 11 testinden 6'sı KIRMIZI olmuyor, sessizce ATLANIYORDU (takım
# yemyeşil kalıyor). Eksik bir kendi modülümüz "platform yok" değil, BOZUK BİR
# YAPI'dır ve toplama sırasında patlamalıdır. `try` yalnızca gerçek ortam
# sorusunu sarmalı: offscreen platformu kurulabiliyor mu?
# (Aynı tuzak `tests/test_orbital_window.py`'de de vardı, orada da böyle kapandı.)
from PySide6 import QtWidgets

from qlab.app.main import TDSEWindow, main
from qlab.scenario.schema import GridSpec, PacketSpec, PotentialSpec, TDSEScenario

try:
    _qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _OFFSCREEN_OK = True
except Exception:  # pragma: no cover - bu ortamda offscreen platformu kurulamazsa
    _OFFSCREEN_OK = False

requires_offscreen = pytest.mark.skipif(
    not _OFFSCREEN_OK, reason="Qt ekransız (offscreen) platformu bu ortamda kurulamadı"
)


def _small_scenario(height: float = 0.8) -> "TDSEScenario":
    """`scenarios/tunneling.yaml` KULLANILMAZ — dt=0.005, n_steps=32000 ile bir
    test için çok yavaştır. Bunun yerine birkaç yüz adımlık küçük bir senaryo
    kurulur; bariyer yüksekliğine göre iletim oranı yine de belirgin biçimde
    değişir (bkz. aşağıdaki testler), bu yüzden gözlemlenebilirlik kaybolmaz.
    """
    return TDSEScenario(
        name="test — küçük tünelleme",
        grid=GridSpec(n=256, length_bohr=80.0),
        potential=PotentialSpec(
            kind="barrier", params={"height": height, "width": 2.0, "center": 0.0}
        ),
        packet=PacketSpec(x0_bohr=-15.0, k0=1.5, sigma_bohr=2.5),
        dt=0.05,
        n_steps=300,
        save_every=300,
    )


@requires_offscreen
def test_window_recompute_populates_readout():
    window = TDSEWindow(_small_scenario())
    text = window._readout.text()

    transmit_match = re.search(r"iletim = ([0-9.eE+-]+)", text)
    norm_match = re.search(r"norm = ([0-9.eE+-]+)", text)
    assert transmit_match is not None, text
    assert norm_match is not None, text

    transmitted = float(transmit_match.group(1))
    norm = float(norm_match.group(1))
    assert 0.0 <= transmitted <= 1.0
    assert norm == pytest.approx(1.0, abs=1e-6)
    window.close()


@requires_offscreen
def test_window_transparency_panel_has_all_labels_and_plausible_deviation():
    window = TDSEWindow(_small_scenario())
    text = window._panel._text.toPlainText()

    for label in ("Yöntem", "Yaklaşımlar", "İhmal edilen", "Beklenen hata", "Yakınsama"):
        assert label in text

    deviation_match = re.search(r"norm_deviation=([0-9.eE+-]+)", text)
    assert deviation_match is not None, text
    deviation = float(deviation_match.group(1))
    # Split-operator üniterdir; bu kadar az adımda sapma makine hassasiyeti
    # mertebesinde olmalı — 1e-6 gevşek bir üst sınır, gerçek sapma çok daha
    # küçük çıkar (bkz. görev raporu).
    assert 0.0 <= deviation < 1e-6
    window.close()


@requires_offscreen
def test_slider_move_changes_readout():
    window = TDSEWindow(_small_scenario(height=0.8))
    baseline = window._readout.text()

    window._slider.setValue(0)
    window._recompute()
    zero_height_text = window._readout.text()
    assert zero_height_text != baseline

    window._slider.setValue(200)
    window._recompute()
    high_height_text = window._readout.text()
    assert high_height_text != zero_height_text
    assert high_height_text != baseline

    # Fizik yönü de doğru olmalı: bariyer yükseldikçe iletim azalır.
    def transmitted_of(text: str) -> float:
        return float(re.search(r"iletim = ([0-9.eE+-]+)", text).group(1))

    assert transmitted_of(zero_height_text) > transmitted_of(high_height_text)
    window.close()


@requires_offscreen
def test_slider_extremes_do_not_crash():
    """Kaydırıcı uçları: 0 (V0=0.00) ve 200 (V0=2.00). factory.build + evolve
    QLabError'a göre totaldir, ama yine de ham bir hata sızdırmadığını burada
    doğrudan doğruluyoruz.
    """
    window = TDSEWindow(_small_scenario())

    window._slider.setValue(0)
    window._recompute()
    assert "V0 = 0.00" in window._readout.text()

    window._slider.setValue(200)
    window._recompute()
    assert "V0 = 2.00" in window._readout.text()
    window.close()


@requires_offscreen
def test_debounce_delays_recompute_until_timeout():
    """Kaydırıcı sürüklenirken her ara değer için tam simülasyon koşulursa
    arayüz kilitlenir — bu yüzden `_recompute` doğrudan `valueChanged`'e değil,
    250 ms'lik tek atımlık bir QTimer'ın `timeout`'una bağlanır. Burada
    kaydırıcı hareket ettirilir ve `_recompute`'un HENÜZ çağrılmadığı,
    yalnızca zamanlayıcı ateşlendiğinde çağrıldığı doğrudan doğrulanır.
    """
    window = TDSEWindow(_small_scenario(height=0.8))
    before = window._readout.text()

    window._slider.setValue(150)
    assert window._debounce.isActive(), "kaydırıcı hareketi zamanlayıcıyı başlatmalı"
    assert window._readout.text() == before, "zamanlayıcı dolmadan yeniden hesap olmamalı"

    window._debounce.timeout.emit()
    assert window._readout.text() != before, "zamanlayıcı dolunca yeniden hesap tetiklenmeli"
    window.close()


@requires_offscreen
def test_main_rejects_hf_scan_scenario_before_building_the_window(tmp_path, capsys):
    """Modül 1B `load_scenario`'yu genişletti: dönüş türü artık TDSEScenario ya
    da HFScanScenario. `run.py` türe göre dallanmayı öğrendi, `app/main.py`
    öğrenmemişti — bir HF tarama senaryosuyla açıldığında `TDSEWindow`
    `scenario.potential`a dokunup çıplak AttributeError ile çöküyordu
    (`scenarios/h2_bond_scan.yaml` ile doğrulandı).

    Beklenen davranış: pencere HİÇ kurulmadan, çıkış kodu 1 ve GUI'nin neyi
    desteklediğini + HF taramalarının nereden koşulacağını söyleyen temiz bir
    mesaj. Metnin kendisi de sınanır: kullanıcıyı `python -m qlab.run`'a
    yönlendirmeyen bir hata, yarısı eksik bir hatadır.
    """
    scenario_path = tmp_path / "hf.yaml"
    scenario_path.write_text(
        "kind: hf_scan\n"
        "name: H2 test taramasi\n"
        'molecule_template: "H 0 0 0; H 0 0 {value}"\n'
        "basis: sto-3g\n"
        "parameter: bond_length\n"
        "values: [0.74]\n",
        encoding="utf-8",
    )

    exit_code = main([str(scenario_path)])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "TDSE" in stderr
    assert "qlab.run" in stderr
    assert "Traceback" not in stderr


@requires_offscreen
def test_panel_yuksekligi_parametreden_geliyor_ve_varsayilani_140():
    """`max_height` gerçekten widget'a ulaşmalı; sabit yazılırsa kimse görmezdi.

    Bu parametre kabul kriteri 3'ün GERÇEK EKRAN doğrulamasının bulduğu tek
    kusurun düzeltmesidir (README: "12 satırlık beyan 140 px'lik panelde
    katlanın altında kalıyordu"). ÖLÇÜLDÜ: düzeltmenin iki ucu da testsizdi --
    `setMaximumHeight(max_height)` → `setMaximumHeight(140)` mutasyonu
    tests/test_orbital_window.py + test_isosurface.py + test_transparency.py
    toplamının 58 testinden SIFIRINI kırıyordu.

    Sayılar LİTERAL: `_VARSAYILAN_YUKSEKLIK`'e karşı yazılsaydı iddia sabitle
    birlikte hareket eder ve hiçbir zaman kırılamazdı (bu depoda ölçülmüş bir
    tuzak, bkz. `test_kaydiricinin_acilis_araligi_surukleyebilir_bir_bant`).
    """
    from qlab.app.transparency import TransparencyPanel

    varsayilan = TransparencyPanel()
    assert varsayilan._text.maximumHeight() == 140

    # Parametre ile verilen değer widget'a AYNEN geçmeli. İki farklı değer:
    # tek değerle sınanan bir sabit-yazım mutasyonu kaçabilirdi.
    for istenen in (260, 300):
        panel = TransparencyPanel(max_height=istenen)
        assert panel._text.maximumHeight() == istenen


@requires_offscreen
def test_daha_yuksek_kutu_ayni_beyanin_daha_cogunu_gosteriyor():
    """`max_height`'ın AMACI: sayının kendisi değil, okunabilen satır sayısı.

    "Okunmayan bir beyan, beyan etmiş sayılmaz" (sınıf dokstring'i). O gerekçe
    makine tarafından hiç sınanmıyordu. Burada aynı uzun beyan iki farklı
    kutuda gösterilir ve gizli kalan satır sayısı karşılaştırılır.

    ÖLÇÜLDÜ (ekransız, 960 px genişlikte, aşağıdaki beyan 33 sarılmış satır
    sürüyor, satır aralığı 14 px):
        kutu 140 px -> görünen alan 135 px, kaydırma çubuğu 24 satır gizliyor
        kutu 260 px -> görünen alan 255 px, kaydırma çubuğu 15 satır gizliyor
        kutu 400 px -> görünen alan 395 px, kaydırma çubuğu  5 satır gizliyor
    İddia MUTLAK sayılara değil SIRALAMAYA bağlanıyor: platformun yazı tipi
    ölçüleri değişse de "daha yüksek kutu daha az satır gizler" bozulmaz.
    Beyan bilerek uzun tutuldu; kısa bir beyanda üç kutu da her şeyi gösterir
    ve test hiçbir şey ölçmezdi (ilk taslakta tam bu oldu: üçü de 0 gizledi).
    """
    from qlab.app.transparency import TransparencyPanel

    dolgu = tuple(
        f"yaklaşım {i}: ölçülmüş bir sapmanın uzun uzun anlatıldığı bir madde"
        for i in range(12)
    )
    uzun = dataclasses.replace(
        PROVENANCE,
        approximations=PROVENANCE.approximations + dolgu,
        neglected=PROVENANCE.neglected + dolgu,
    )

    gizli = {}
    for kutu in (140, 260, 400):
        panel = TransparencyPanel(max_height=kutu)
        panel.set_provenance(uzun)
        panel.resize(960, kutu + 40)
        panel.show()
        _qapp.processEvents()
        gizli[kutu] = panel._text.verticalScrollBar().maximum()
        panel.close()

    assert gizli[140] > gizli[260] > gizli[400] >= 0, (
        f"daha yüksek kutu daha az satır gizlemeli, ölçülen: {gizli}"
    )
