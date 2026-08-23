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
    # DAİRESELLİK BEYANI — Modül 4'te eklendi, ve `approximations`ın İÇİNE
    # KONMADI. Gerekçe bir kavram ayrımıdır: bir yaklaşım YAPILAN bir şeydir
    # ("kare kafes kullandık"), dairesellik beyanı ise çıktının NE OLMADIĞINI
    # söyler ("bu sayı bir tahmin değil, girdinin geri okunması"). `expected_error`
    # da olamaz, çünkü orada bir hata sınırı verilir; burada sınırlanacak bir
    # hata YOKTUR — SI değeri hiç tahmin edilmemiştir ve ona yüzde yazmak
    # kategori hatası olurdu. Ayrı alan olmasının pratik bedeli de ölçüldü:
    # `as_lines()` yaklaşımları " · " ile birleştirir, yani listeye konsaydı
    # modülün varlık sebebi beş maddeden biri olarak akıp giderdi. Panelde
    # kendi satırını alır. (`external_stability`ın ayrı alan olma gerekçesiyle
    # aynı aile: iki hüküm birbirinin yerine geçemez.)
    circularity: str | None = None
    # ÖRNEKLEME TANISI — Modül 4'te eklendi. `diagnostics` sözlüğüne KONMADI
    # ve gerekçesi ölçülmüş bir yanlış-okuma riskidir: `_diagnostics_summary`
    # çıktısı "Yakınsama" satırının İÇİNE, ✓/✗ işaretinin HEMEN ARDINA
    # basılır. Hata çubuğunun bir alt sınır olduğu uyarısı oraya konsaydı
    # panelde "✓ ... ALT SINIR" diye görünürdü — yani uyarının yanında bir
    # onay işareti. `stability` alanının ayrı tutulma gerekçesi buydu:
    # analiz edilmemiş ya da güvenilmez bir şeyi nötr/olumlu bir bağlamda
    # göstermek okuyucuya "sorun yok" diye okunur. İki satır iki AYRI soruyu
    # cevaplar: "Yakınsama" hesabın bitip bitmediğini, "Örnekleme" raporlanan
    # belirsizliğin belirsizliğin KENDİSİ mi yoksa yalnızca ALT SINIRI mı
    # olduğunu.
    sampling: str | None = None

    def as_lines(self) -> list[tuple[str, str]]:
        """Şeffaflık panelinin doğrudan bastığı etiket/değer çiftleri."""
        mark = "✓" if self.converged else "✗"
        lines = [
            ("Yöntem", self.method),
            ("Yaklaşımlar", SEPARATOR.join(self.approximations)),
            ("İhmal edilen", SEPARATOR.join(self.neglected)),
            ("Beklenen hata", self.expected_error),
        ]
        if self.circularity is not None:
            lines.append(("Dairesellik", self.circularity))
        lines.append(("Yakınsama", f"{mark} {self._diagnostics_summary()}".strip()))
        if self.sampling is not None:
            lines.append(("Örnekleme", self.sampling))
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
