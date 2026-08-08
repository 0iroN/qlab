"""Hesabın hangi yaklaşımlar altında yapıldığını taşıyan veri yapısı.

Bu bir arayüz süsü değildir: her sonuç bir Provenance taşımak zorundadır
(spec §3, İlke 2). Şeffaflık paneli doğrudan as_lines() çıktısını basar;
panel kendi başına hiçbir fizik bilgisi üretmez.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEPARATOR = " · "


@dataclass(frozen=True)
class StabilityVerdict:
    """Bir çözümün kararlılık analizinin makine-okunur özeti.

    `converged`'den AYRI bir kavramdır ve onun yerine geçmez: yakınsamış bir
    çözüm enerjinin bir durağan noktasıdır, ama minimum olmak zorunda
    değildir. `stable=False`, o noktanın yakınında daha düşük enerjili bir
    çözüm bulunduğunu (yani sonucun bir EYER NOKTASI olduğunu) söyler —
    sonucun geçersiz olduğunu değil.

    `lowest_eigenvalue`, orbital Hessian'ının en küçük özdeğeridir (hartree
    / radyan²). Döndürülecek çift yoksa (`n_rotations == 0`, örneğin boş
    orbitali olmayan He/STO-3G) ölçülecek bir özdeğer de yoktur ve alan
    None kalır — 0.0 yazmak "sınırda kararlı" diye okunurdu.

    Alanların hepsi JSON'a yazılabilir türlerdir; `run.save()` provenance'ı
    `dataclasses.asdict` + `json.dumps` ile diske yazar.
    """

    stable: bool
    lowest_eigenvalue: float | None
    n_rotations: int

    def as_line(self) -> str:
        """Şeffaflık panelinin bastığı tek satırlık hüküm."""
        mark = "✓" if self.stable else "✗"
        if self.lowest_eigenvalue is None:
            return f"{mark} işgal-boş dönme yok, sınanacak yön yok"
        eigenvalue = f"{self.lowest_eigenvalue:.3e}"
        if self.stable:
            return (
                f"{mark} iç (RHF->RHF) minimum: negatif Hessian özdeğeri yok "
                f"(en düşük {eigenvalue}, {self.n_rotations} dönme)"
            )
        return (
            f"{mark} EYER NOKTASI: negatif Hessian özdeğeri var "
            f"(en düşük {eigenvalue}, {self.n_rotations} dönme) — yakınında "
            f"daha düşük enerjili bir RHF çözümü var"
        )


@dataclass(frozen=True)
class Provenance:
    method: str
    approximations: tuple[str, ...]
    neglected: tuple[str, ...]
    expected_error: str
    converged: bool
    diagnostics: dict = field(default_factory=dict)
    # Yalnızca kararlılık analizi GERÇEKTEN koşulduğunda doldurulur. TDSE
    # sonuçlarında ve stability=False ile koşulan HF sonuçlarında None kalır;
    # panel o zaman satırı hiç basmaz. Analiz edilmemiş bir şeyi boş/nötr bir
    # değerle göstermek, okuyucuya "sorun yok" diye okunurdu.
    stability: StabilityVerdict | None = None

    def as_lines(self) -> list[tuple[str, str]]:
        """Şeffaflık panelinin doğrudan bastığı etiket/değer çiftleri."""
        mark = "✓" if self.converged else "✗"
        lines = [
            ("Yöntem", self.method),
            ("Yaklaşımlar", SEPARATOR.join(self.approximations)),
            ("İhmal edilen", SEPARATOR.join(self.neglected)),
            ("Beklenen hata", self.expected_error),
            ("Yakınsama", f"{mark} {self._diagnostics_summary()}".strip()),
        ]
        if self.stability is not None:
            lines.append(("Kararlılık", self.stability.as_line()))
        return lines

    def _diagnostics_summary(self) -> str:
        if not self.diagnostics:
            return ""
        return ", ".join(f"{k}={self._format(v)}" for k, v in self.diagnostics.items())

    @staticmethod
    def _format(value: object) -> str:
        return f"{value:.3e}" if isinstance(value, float) else str(value)
