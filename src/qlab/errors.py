"""qlab istisna hiyerarşisi.

İlke (spec §3, İlke 3): güvenilmez veya fiziksel olarak anlamsız bir sonuç
asla normal sonuç gibi dönmez. Her istisna, tanıyı yapabilmek için gereken
sayısal bağlamı taşır.
"""

from __future__ import annotations


class QLabError(Exception):
    """qlab kaynaklı tüm hataların ortak atası."""


class InvalidInputError(QLabError):
    """Fiziksel veya sayısal olarak geçersiz girdi."""


class ScenarioError(QLabError):
    """Senaryo dosyası okunamadı veya geçersiz."""


class UnitarityViolationError(QLabError):
    """Gerçek zamanlı evrimde norm korunumu bozuldu.

    Split-operator ünitertir; norm korunumu yöntemin garantisidir. Sapma varsa
    ya dt çok büyüktür ya da kodlama hatası vardır. İki durumda da sonuç
    kullanılamaz.
    """

    def __init__(self, step: int, norm_deviation: float, dt: float) -> None:
        self.step = step
        self.norm_deviation = norm_deviation
        self.dt = dt
        super().__init__(
            f"Norm korunumu {step}. adımda bozuldu "
            f"(sapma {norm_deviation:.3e}, dt={dt}). "
            "Muhtemel sebep: dt çok büyük. Yarıya indirip tekrar deneyin."
        )
