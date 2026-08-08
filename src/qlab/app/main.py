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

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(200)
        self._slider.setValue(
            int(round(100 * scenario.potential.params.get("height", 0.8)))
        )

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_RECOMPUTE_DELAY_MS)
        self._debounce.timeout.connect(self._recompute)
        self._slider.valueChanged.connect(lambda _: self._debounce.start())

        self._readout = QtWidgets.QLabel()
        self._panel = TransparencyPanel()

        controls = QtWidgets.QFormLayout()
        controls.addRow("Bariyer yüksekliği (hartree × 100)", self._slider)
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
        self._readout.setText(
            f"V0 = {height:.2f} hartree   ·   iletim = {payload.transmitted:.4f}"
            f"   ·   norm = {payload.norm:.10f}"
        )
        self._panel.set_provenance(
            factory.provenance_for(scenario, norm_deviation=abs(payload.norm - 1.0))
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qlab.app.main")
    parser.add_argument("scenario", help="YAML senaryo dosyası")
    args = parser.parse_args(argv)

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

    app = QtWidgets.QApplication(sys.argv[:1])
    window = TDSEWindow(scenario)
    window.resize(1100, 780)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
