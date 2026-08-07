"""Boys fonksiyonu: F_n(T) = integral_0^1 t^{2n} exp(-T t^2) dt.

Tum cekirdek-cekim ve iki-elektron integrallerinin sayisal temeli. Iki
rejim ayri ele alinir:

- Kucuk T: kapali form F_n(T) = hyp1f1(n+1/2, n+3/2, -T) / (2n+1)
  (Kummer'in birinci turden birlesik hipergeometrik fonksiyonu).
- Buyuk T: asimptotik seri F_n(T) ~ (2n-1)!! / 2^{n+1} * sqrt(pi / T^{2n+1}).

Gecis esigi olculerek secildi (bkz. .superpowers/sdd/task-4-report.md):
hyp1f1 formu, sayisal integrasyon referansina karsi, test kapsamindaki
T <= 120 araliginda n=0..5 icin daima ~1e-15 bagil hatada kaldi (hicbir
bozulma yok; daha genis bir kesif taramasinda T=500'e kadar da bozulma
yalnizca ~5e-12 mertebesinde kaldi). Asimptotik formun hyp1f1'e yakinsamasi
n arttikca gecikiyor: test suite'teki en yuksek mertebe olan n=5 icin iki
form T=40'ta 1.5e-12, T=41'de 6.0e-13 farkla ayrisiyor, T=45'te fark
1.7e-14'e iniyor; n=0..4 icin yakinsama cok daha erken (T~25-40)
gerceklesiyor. _LARGE_T_THRESHOLD = 45.0 secildi: en zorlu durumda (n=5)
bile 1e-12 esiginin rahatca guvenli tarafinda, ve hyp1f1 o noktada hala
tam hassasiyetle dogru oldugu icin kesintide sureklilik kaybi olmuyor.

Esik yalnizca n=0..5 icin olculdu (bu gorevin test kapsamiyla sinirli).
Cok daha yuksek n icin (ileri gorevlerde d/f kabuklari gerekirse) esigin
yeniden olculmesi gerekebilir.

Ozyineleme kullanilmiyor: her iki formul de kendi rejiminde dogrudan makine
hassasiyetine yakin sonuc verdigi icin asagi yonlu ozyinelemeye ihtiyac
kalmiyor. F_n(T) = (2T F_{n+1}(T) + exp(-T)) / (2n+1) ozdesligi (bagimsiz
dogrulama olarak test dosyasinda sinanir) dogru hesaplanan herhangi bir
F degerinde otomatik saglanir; ozyineleme kullanilsaydi yalnizca asagi
yonlu (F_{n_max}'tan asagi) olmasi gerekirdi -- yukari yon buyuk n'de
kararsizdir.
"""

from __future__ import annotations

import numpy as np
from scipy.special import factorial2, hyp1f1

_LARGE_T_THRESHOLD = 45.0


def _double_factorial_odd(n: int) -> float:
    """(2n-1)!! -- n=0 icin (-1)!! = 1 kuraldir.

    scipy.special.factorial2(-1) 0.0 doner (scipy 1.18'de olculdu), bu
    yuzden n=0 durumu elle verilir.
    """
    return 1.0 if n == 0 else float(factorial2(2 * n - 1))


def _small_T(n: int, T: np.ndarray) -> np.ndarray:
    """Kapali form: F_n(T) = hyp1f1(n+1/2, n+3/2, -T) / (2n+1)."""
    return hyp1f1(n + 0.5, n + 1.5, -T) / (2 * n + 1)


def _large_T(n: int, T: np.ndarray) -> np.ndarray:
    """Asimptotik form: F_n(T) ~ (2n-1)!! / 2^{n+1} * sqrt(pi / T^{2n+1})."""
    return _double_factorial_odd(n) / 2 ** (n + 1) * np.sqrt(np.pi / T ** (2 * n + 1))


def boys(n: int, T: float | np.ndarray) -> float | np.ndarray:
    """Boys fonksiyonu F_n(T) = integral_0^1 t^{2n} exp(-T t^2) dt.

    T skaler veya numpy dizisi olabilir; donen deger girdiyle ayni sekli
    tasir (skaler icin float, dizi icin ayni sekilli ndarray).
    """
    T_arr = np.asarray(T, dtype=float)
    scalar_input = T_arr.ndim == 0
    T_flat = np.atleast_1d(T_arr)

    result = np.empty_like(T_flat)
    small_mask = T_flat < _LARGE_T_THRESHOLD
    large_mask = ~small_mask

    if np.any(small_mask):
        result[small_mask] = _small_T(n, T_flat[small_mask])
    if np.any(large_mask):
        result[large_mask] = _large_T(n, T_flat[large_mask])

    if scalar_input:
        return float(result[0])
    return result.reshape(T_arr.shape)
