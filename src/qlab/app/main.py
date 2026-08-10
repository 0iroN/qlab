"""Etkileşimli TDSE penceresi.

Kaydırıcıyla bariyer yüksekliği değişir, simülasyon yeniden koşar, iletim oranı
ve şeffaflık paneli anlık güncellenir (kabul kriteri 4 ve 5).

Yeniden hesap 250 ms geciktirilir: kaydırıcı sürüklenirken her ara değer için
8000 adım koşmak arayüzü kilitler.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from qlab.app.transparency import TransparencyPanel
from qlab.engine.tdse import factory
from qlab.engine.tdse.propagator import evolve
from qlab.errors import QLabError, ScenarioError
from qlab.run import reconfigure_stdio_utf8
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import TDSEScenario
from qlab.viz.frames import build_payload

_MEASUREMENT_BOUNDARY = 5.0
_RECOMPUTE_DELAY_MS = 250


class TDSEWindow(QtWidgets.QMainWindow):
    def __init__(self, scenario) -> None:
        super().__init__()
        self.setWindowTitle(f"qlab — {scenario.name}")
        self._scenario = scenario

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "x (bohr)")
        self._plot.setLabel("left", "|psi|^2  (potansiyel ölçekli)")
        self._density_curve = self._plot.plot(pen=pg.mkPen(width=2))
        self._potential_curve = self._plot.plot(
            pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine, width=1)
        )

        # Kaydırıcı bir bariyer yüksekliğini döndürür; senaryoda öyle bir
        # parametre yoksa döndürecek bir şey de yoktur (bkz. `_recompute`).
        self._bariyerli = "height" in scenario.potential.params

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(200)
        self._slider.setValue(
            int(round(100 * scenario.potential.params.get("height", 0.8)))
        )
        self._slider.setEnabled(self._bariyerli)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_RECOMPUTE_DELAY_MS)
        self._debounce.timeout.connect(self._recompute)
        self._slider.valueChanged.connect(lambda _: self._debounce.start())

        self._readout = QtWidgets.QLabel()
        self._panel = TransparencyPanel()

        controls = QtWidgets.QFormLayout()
        controls.addRow(
            "Bariyer yüksekliği (hartree × 100)"
            if self._bariyerli
            else "Bariyer yüksekliği — bu potansiyelde yok",
            self._slider,
        )
        controls.addRow("Ölçüm", self._readout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._plot, stretch=1)
        layout.addLayout(controls)
        layout.addWidget(self._panel)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self._recompute()

    def _recompute(self) -> None:
        height = self._slider.value() / 100.0
        params = dict(self._scenario.potential.params)
        # Kaydırıcı YALNIZCA senaryonun zaten taşıdığı bir `height`'ı değiştirir.
        # Koşulsuz enjekte edilince iki ayrı hata çıkıyordu (ölçüldü):
        #   - `harmonic` bu parametreyi hiç almıyor, pencere ScenarioError ile
        #     ÇÖKÜYORDU -- yani `qlab.app.main scenarios/harmonic_trap.yaml`
        #     hiç çalışmamıştı;
        #   - `double_slit` ise ALIYOR (varsayılan 10000.0 hartree, duvarı
        #     geçirimsiz yapan değer), yani sessizce 0.80'e düşürülür ve
        #     yarıklar anlamını yitirirdi -- çökme bile olmadan.
        # Bariyeri olmayan senaryoda kaydırıcı kurulumda kapatılır; "etkin ama
        # etkisiz" bir denetim yalan söylerdi.
        if self._bariyerli:
            params["height"] = height
        scenario = dataclasses.replace(
            self._scenario,
            potential=dataclasses.replace(self._scenario.potential, params=params),
        )

        grid, potential, psi0, propagator = factory.build(scenario)
        frames, _ = evolve(
            propagator, psi0, n_steps=scenario.n_steps, save_every=scenario.n_steps
        )

        payload = build_payload(frames[-1], grid, potential, _MEASUREMENT_BOUNDARY)
        self._density_curve.setData(payload.x, payload.density)
        self._potential_curve.setData(payload.x, payload.potential_scaled)
        # Bariyeri olmayan senaryoda "V0" diye bir sayı basmak yalan olurdu:
        # kaydırıcının değeri hiçbir yere gitmiyor.
        bariyer = f"V0 = {height:.2f} hartree   ·   " if self._bariyerli else ""
        self._readout.setText(
            f"{bariyer}iletim = {payload.transmitted:.4f}"
            f"   ·   norm = {payload.norm:.10f}"
        )
        self._panel.set_provenance(
            factory.provenance_for(scenario, norm_deviation=abs(payload.norm - 1.0))
        )


def _pencereyi_goster(window: QtWidgets.QWidget) -> int:
    """Pencereyi gösterir ve Qt olay döngüsünü koşturur; çıkış kodunu döndürür.

    Bu, testlerin yamaladığı TEK gösterim noktasıdır: olay döngüsü bir testte
    koşturulamaz. Ayrı bir işlev olması, `main()`'in hangi pencereyi kurduğunun
    olay döngüsünü açmadan doğrulanabilmesini sağlar.

    `QApplication.instance()` önce denenir. Süreçte zaten bir uygulama varken
    ikincisini kurmak Qt'yi düşürür; test oturumunda `tests/test_transparency.py`
    ve `tests/test_orbital_window.py` modül düzeyinde birer tane kuruyor.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window.show()
    return app.exec()


def _orbital_penceresi(molecule_text: str, basis: str) -> QtWidgets.QWidget:
    """`--molecule` yolu: SCF koşar ve orbital penceresini kurar.

    İçe aktarma işlev gövdesinde: `orbital_window` PyOpenGL'e bağlı ve TDSE
    yolunu kullanan biri için o bağımlılığı modül yüklenirken zorunlu kılmak
    gereksiz olurdu.
    """
    from qlab.app.orbital_window import OrbitalWindow
    from qlab.engine.hf.scf import solve
    from qlab.types import Molecule

    molecule = Molecule.from_string(molecule_text)
    result = solve(molecule, basis)

    # QApplication, HERHANGİ bir widget'tan ÖNCE var olmak zorunda -- TDSE dalı
    # `TDSEWindow`'dan hemen önce aynısını yapıyor. Yoksa Qt `qFatal` ile süreci
    # ÖLDÜRÜR: ölçüldü, Windows'ta çıkış kodu 0xC0000409, Python istisnası YOK,
    # traceback YOK, stderr BOŞ. Aynı süreçte koşan hiçbir test bunu
    # yakalayamaz çünkü her Qt test modülü içe aktarılırken kendi
    # QApplication'ını kuruyor -- ortam hatayı maskeliyor. Yakalayan test
    # `test_main_taze_surecte_qapplication_olmadan_cokmuyor`, ALT SÜREÇTE koşar.
    #
    # Molekül ayrıştırma ve SCF'ten SONRA: geçersiz bir geometri ya da
    # yakınsamayan bir hesap Qt hiç başlatılmadan raporlanır.
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication(sys.argv[:1])

    return OrbitalWindow(result, molecule)


def main(argv: list[str] | None = None) -> int:
    # `qlab.run.main()` ile aynı desen, aynı yerden (iş başlamadan önce)
    # çağrılır — bkz. reconfigure_stdio_utf8 docstring'i. CP1252 konsolda
    # Türkçe diyakritikler artık kaçış dizisi olarak değil doğrudan basılır.
    reconfigure_stdio_utf8()

    parser = argparse.ArgumentParser(prog="qlab.app.main")
    parser.add_argument(
        "scenario", nargs="?", default=None, help="YAML senaryo dosyası (TDSE)"
    )
    parser.add_argument(
        "--molecule",
        default=None,
        help='Molekül geometrisi, ör. "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587" '
        "(ångström). Verilirse RHF koşulur ve orbital penceresi açılır.",
    )
    parser.add_argument(
        "--basis", default="sto-3g", help="Baz seti (yalnızca --molecule ile)"
    )
    args = parser.parse_args(argv)

    # İki giriş yolu birbirini DIŞLAR. Biri seçilip diğeri sessizce yok
    # sayılsaydı kullanıcı hangisine baktığını tahmin etmek zorunda kalırdı.
    if (args.scenario is None) == (args.molecule is None):
        print(
            "tam olarak birini verin: bir YAML senaryo dosyası (TDSE) ya da "
            "--molecule (orbital penceresi).\n"
            'örnek: python -m qlab.app.main --molecule "H 0 0 0; H 0 0 0.74"',
            file=sys.stderr,
        )
        return 1

    if args.molecule is not None:
        try:
            window = _orbital_penceresi(args.molecule, args.basis)
        except QLabError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return _pencereyi_goster(window)

    try:
        scenario = load_scenario(args.scenario)
        # `load_scenario` artık TDSEScenario ya da HFScanScenario döndürebiliyor
        # (Modül 1B), ama bu pencere yalnızca TDSE'yi çizer: `TDSEWindow`
        # doğrudan `scenario.potential`a dokunur. Tür kontrolü QApplication
        # KURULMADAN ÖNCE yapılır — aksi halde kullanıcı bir Qt penceresi
        # açılırken çıplak bir AttributeError görürdü. Hata, projenin kendi
        # hiyerarşisinden (ScenarioError -> QLabError) seçilir ki aşağıdaki
        # `except` bloğu onu `run.py` ile aynı biçimde, traceback dökmeden
        # raporlasın.
        if not isinstance(scenario, TDSEScenario):
            raise ScenarioError(
                "GUI kabuğu şu an yalnızca TDSE senaryolarını görselleştirir; "
                f"{args.scenario!r} bir "
                f"{type(scenario).__name__} (HF taraması). "
                "HF taramaları GUI'siz koşucudan çalıştırılır: "
                f"python -m qlab.run {args.scenario} -o cikti.npz"
            )
    except QLabError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # QApplication, pencere kurulmadan ÖNCE var olmalı (widget'lar ona bağlanır);
    # kurulumu `_pencereyi_goster` ile paylaşan tek yol `instance()` kontrolüdür.
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication(sys.argv[:1])
    window = TDSEWindow(scenario)
    window.resize(1100, 780)
    return _pencereyi_goster(window)


if __name__ == "__main__":
    raise SystemExit(main())
