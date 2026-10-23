"""Hesabın hangi yaklaşımlar altında yapıldığını taşıyan veri yapısı.

Bu bir arayüz süsü değildir: her sonuç bir Provenance taşımak zorundadır
(spec §3, İlke 2). Şeffaflık paneli doğrudan as_lines() çıktısını basar;
panel kendi başına hiçbir fizik bilgisi üretmez.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEPARATOR = " · "

# Kararlılık hükmünün tanınan kanalları: iç (RHF→RHF), dış (RHF→UHF), uhf
# (UHF→UHF). Bu kümenin dışındaki bir ad `StabilityVerdict.as_line`da hata verir.
_KANALLAR = ("ic", "dis", "uhf")


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
    #
    # ÜÇÜNCÜ kanal "uhf" (UHF→UHF, reel α/β dönmeleri) — B7'de eklendi
    # (09.10.2026). İç ve dış kanal bir RHF çözümünü sınar; bu kanal bir UHF
    # çözümünün KENDİSİNİ sınar. Bilinmeyen bir kanal adı artık ValueError
    # verir: B7'den önce bilinmeyen her ad sessizce iç kanalın "iç (RHF->RHF)
    # minimum" cümlesine düşüyordu — bir UHF hükmü RHF cümlesiyle basılırdı.
    channel: str = "ic"

    def as_line(self) -> str:
        """Şeffaflık panelinin bastığı tek satırlık hüküm."""
        if self.channel not in _KANALLAR:
            raise ValueError(
                f"bilinmeyen kararlılık kanalı {self.channel!r}; "
                f"tanınanlar: {', '.join(_KANALLAR)}"
            )
        mark = "✓" if self.stable else "✗"
        if self.lowest_eigenvalue is None:
            return f"{mark} işgal-boş dönme yok, sınanacak yön yok"
        eigenvalue = f"{self.lowest_eigenvalue:.3e}"
        nicelik = f"(en düşük {eigenvalue}, {self.n_rotations} dönme)"
        if self.channel == "uhf":
            if self.stable:
                return (
                    f"{mark} UHF YEREL minimum: reel α/β dönmelerinde negatif "
                    f"Hessian özdeğeri yok {nicelik} — küresel minimum iddia "
                    f"EDİLMEZ, |λ| < 1e-8 yumuşak modlar ayırt edilmez"
                )
            return (
                f"{mark} UHF EYERİ: reel α/β dönmelerinde negatif Hessian "
                f"özdeğeri var {nicelik} — yakınında daha düşük enerjili bir "
                f"UHF çözümü var"
            )
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
    # ÇIPA SINIFI — Modül 5'te eklendi. Bir çıpanın SINIFI (şema / fizik /
    # çeviri denetimi) ne bir yaklaşımdır ne de bir ihmal: "Courant sınırı
    # şemanın kendi özelliğidir" cümlesi yapılan bir şeyi de, yapılmayan bir
    # şeyi de anlatmaz — okunan sayının NEYİ doğruladığını söyler. Bu depoda
    # bir şema çıpasını fizik çıpası diye sunmak ÜÇ kez düzeltilmiş bir
    # hatadır ve düzeltmenin panel tarafındaki karşılığı budur. `approximations`
    # listesine konamaz çünkü `as_lines()` o listeyi " · " ile birleştirir ve
    # dört sınıf etiketi yaklaşımların arasında akıp giderdi (dairesellik
    # alanının ayrı tutulma gerekçesinin aynısı).
    anchor_class: str | None = None
    # PENCERE — Modül 5'te eklendi. Bu modülde RNG yoktur; tekrarlanabilirliğin
    # karşılığı PENCERENİN TAM BEYANIDIR (ızgara, Δx, S, adım, λ, npml, sgf,
    # FFT penceresi). Ayrı alan olmasının gerekçesi ölçülmüş: bu depoda bir
    # hükmün kendi penceresinin dışına taşınması ON KEZ oldu, ve pencere bir
    # `Yaklaşımlar` maddesi olsaydı tam da o cümlelerin arasında görünmez
    # olurdu. `expected_error`a da konamaz: orada bir hata SINIRI verilir,
    # pencere ise sınırın hangi bölgede geçerli olduğunu söyler.
    window: str | None = None
    # YAKINSAMA TANISI — Modül 5'te eklendi, ve `diagnostics` sözlüğüne
    # KONMADI. Gerekçe Modül 4'ün `sampling` alanınınkiyle AYNI ve ondan daha
    # sert: `_diagnostics_summary()` çıktısı "Yakınsama" satırının İÇİNE,
    # ✓/✗ işaretinin HEMEN ARDINA basılır. Üç değerli hüküm oraya konsaydı
    # panelde "✓ ... verdict=YARGILANAMAZ" diye görünürdü — yani yargılanamadığı
    # söylenen bir sayının yanında bir onay işareti. Modül 5'in kendi kuralı
    # tam olarak bunu yasaklar: YARGILANAMAZ sessizce "oturdu" diye basılmaz.
    convergence_diagnosis: str | None = None
    # KURAM SEÇİCİ — Modül 6'da eklendi, ve `diagnostics` sözlüğüne KONMADI.
    # Gerekçe Modül 4'ün `sampling`i ve Modül 5'in `convergence_diagnosis`i ile
    # AYNI aile, ve bu modülde en serti: `_diagnostics_summary()` çıktısı
    # "Yakınsama" satırının İÇİNE, ✓/✗ işaretinin HEMEN ARDINA basılır. `γ`, `β`
    # ve `λ` oraya konsaydı panelde **"✓ … theory=NOT-GR"** diye görünürdü —
    # yani "çözülen kuram genel görelilik DEĞİL" uyarısının yanında bir onay
    # işareti. `approximations` listesine de konamaz ve bu bir kategori
    # ayrımıdır: bir kuram seçicisi YAPILAN bir şey (yaklaşım) değildir, bir
    # BİRİM SEÇİMİ de değildir (`G = c = 1` odur) — HANGİ KURAMIN çözüldüğünü
    # seçer. `anchor_class`tan da ayrıdır: çıpa sınıfı okunan sayının NEYİ
    # doğruladığını söyler, kuram seçici ise sayının HANGİ KURAMA ait olduğunu.
    theory_selector: str | None = None

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
        if self.anchor_class is not None:
            lines.append(("Çıpa sınıfı", self.anchor_class))
        if self.theory_selector is not None:
            lines.append(("Kuram seçici", self.theory_selector))
        if self.window is not None:
            lines.append(("Pencere", self.window))
        lines.append(("Yakınsama", f"{mark} {self._diagnostics_summary()}".strip()))
        if self.convergence_diagnosis is not None:
            lines.append(("Yakınsama tanısı", self.convergence_diagnosis))
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


def etiket_genisligi() -> int:
    """Şeffaflık bloğunun etiket sütunu: `as_lines()`in basabileceği EN UZUN etiket.

    TEK KAYNAK (C5, grup 9, 14.10.2026): GUI (`app/transparency.format_panel`) ve
    CLI (`run.py`nin dalları) sütunu buradan okur. Genişlik panelin içeriğinden
    değil `as_lines()`in SÖZ DAĞARCIĞINDAN türer: bütün isteğe bağlı alanları
    dolu bir örneğin satırları taranır (alanlar `dataclasses.fields`ten, yani
    yeni bir alan ve daha uzun bir etiket eklenince sütun kendiliğinden
    genişler). Bir tanı satırının gelip gitmesi bu yüzden hizayı oynatmaz.
    ESKİDEN sabit 14'tü (GUI `_LABEL_WIDTH`, CLI `<14`; üç dal elle `<16`) ve
    16 karakterlik "Yakınsama tanısı" sütunu taşırıyordu: o satırın ':'si
    ötekilerden 2 sütun sağda kalıyordu (ölçüldü, HEAD 448f808: MD/HF NVE
    paneli, 9 satırın 8'i 14'te, tanı satırı 16'da).
    """
    from dataclasses import fields

    ornek = {
        f.name: StabilityVerdict(stable=True, lowest_eigenvalue=None, n_rotations=0)
        if "StabilityVerdict" in str(f.type) else ""
        for f in fields(Provenance)
        if f.default is None
    }
    tam = Provenance(
        method="", approximations=(), neglected=(), expected_error="", converged=True, **ornek
    )
    return max(len(etiket) for etiket, _ in tam.as_lines())
