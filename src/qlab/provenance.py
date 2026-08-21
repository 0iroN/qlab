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
    # Hangi kanalın hükmü: "ic" (tekil, RHF→RHF) veya "dis" (üçlü, RHF→UHF).
    # İki kanal FARKLI şeyler söyler ve biri diğerinin yerine geçmez — bir
    # çözüm iç kanalda minimum, dış kanalda eyer olabilir (ölçülmüş örnek:
    # H2 @ 2.00 Å, iç +2.5467 "kararlı", dış −1.5995 eyer). Alan varsayılanla
    # "ic"tir; Modül 1C'den önce tek kanal vardı ve o kanal buydu, dolayısıyla
    # eski kayıtlar doğru okunur.
    channel: str = "ic"

    def as_line(self) -> str:
        """Şeffaflık panelinin bastığı tek satırlık hüküm."""
        mark = "✓" if self.stable else "✗"
        if self.lowest_eigenvalue is None:
            return f"{mark} işgal-boş dönme yok, sınanacak yön yok"
        eigenvalue = f"{self.lowest_eigenvalue:.3e}"
        nicelik = f"(en düşük {eigenvalue}, {self.n_rotations} dönme)"
        if self.channel == "dis":
            if self.stable:
                return (
                    f"{mark} dış (RHF->UHF) yönde de minimum: spin simetrisini "
                    f"kırmak enerjiyi düşürmüyor {nicelik}"
                )
            return (
                f"{mark} SPİN EYERİ: dış (RHF->UHF) yönde negatif Hessian "
                f"özdeğeri var {nicelik} — spin simetrisini kıran bir UHF "
                f"çözümü daha düşük enerjilidir"
            )
        if self.stable:
            return (
                f"{mark} iç (RHF->RHF) minimum: negatif Hessian özdeğeri yok "
                f"{nicelik}"
            )
        return (
            f"{mark} EYER NOKTASI: negatif Hessian özdeğeri var "
            f"{nicelik} — yakınında daha düşük enerjili bir RHF çözümü var"
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
    # Dış (RHF→UHF, üçlü) kanal — Modül 1C'de eklendi. AYRI bir alan olmasının
    # sebebi, iki hükmün birbirinin yerine geçmemesidir: `stability` "kararlı"
    # derken bunun "eyer" demesi TİPİK durumdur, çelişki değil. Tek alanda
    # birleştirilse hangi kanalın konuştuğu kaybolurdu.
    external_stability: StabilityVerdict | None = None

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
        if self.external_stability is not None:
            lines.append(("Dış kararlılık", self.external_stability.as_line()))
        return lines

    def _diagnostics_summary(self) -> str:
        if not self.diagnostics:
            return ""
        return ", ".join(f"{k}={self._format(v)}" for k, v in self.diagnostics.items())

    @staticmethod
    def _format(value: object) -> str:
        return f"{value:.3e}" if isinstance(value, float) else str(value)
