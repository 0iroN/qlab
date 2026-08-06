"""Hesabın hangi yaklaşımlar altında yapıldığını taşıyan veri yapısı.

Bu bir arayüz süsü değildir: her sonuç bir Provenance taşımak zorundadır
(spec §3, İlke 2). Şeffaflık paneli doğrudan as_lines() çıktısını basar;
panel kendi başına hiçbir fizik bilgisi üretmez.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEPARATOR = " · "


@dataclass(frozen=True)
class Provenance:
    method: str
    approximations: tuple[str, ...]
    neglected: tuple[str, ...]
    expected_error: str
    converged: bool
    diagnostics: dict = field(default_factory=dict)

    def as_lines(self) -> list[tuple[str, str]]:
        """Şeffaflık panelinin doğrudan bastığı etiket/değer çiftleri."""
        mark = "✓" if self.converged else "✗"
        return [
            ("Yöntem", self.method),
            ("Yaklaşımlar", SEPARATOR.join(self.approximations)),
            ("İhmal edilen", SEPARATOR.join(self.neglected)),
            ("Beklenen hata", self.expected_error),
            ("Yakınsama", f"{mark} {self._diagnostics_summary()}".strip()),
        ]

    def _diagnostics_summary(self) -> str:
        if not self.diagnostics:
            return ""
        return ", ".join(f"{k}={self._format(v)}" for k, v in self.diagnostics.items())

    @staticmethod
    def _format(value: object) -> str:
        return f"{value:.3e}" if isinstance(value, float) else str(value)
