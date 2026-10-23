"""HC-SABIT (14.10.2026): ısı sığası ve süseptibilite MERKEZLİ varyansla.

`<x²> − <x>²` iki büyük ve neredeyse eşit sayının farkıdır. HEAD'de (448f808)
kamu yolundan ölçüldü (Ising L = 8, T = 2, 64 ölçüm, enerji serisi tümüyle
sabit, fizik C = 0): `field` = 1e10 → C = 262144, 1e100 → C = −2.7e185.
Düzeltme `spin_obs._merkezli_varyans`: ilk örneğe kaydırılmış iki geçiş.

SINIFLAR (dürüst etiket):
- FİZİK ÇIPASI: donmuş, alanla hizalı kafeste enerji serisi SABİTTİR
  (her çevirme ΔE ≥ 2|h|, kabul exp(−2|h|/T) = 0) ve C = 0'dır; ısı sığası
  NEGATİF olamaz. Ayrıca enerjinin sıfırı keyfidir: C sabit ofsetle değişmez.
- BAĞIMSIZ YÖNTEM ÇIPASI: büyük ortalama çevresinde küçük dalgalanmanın
  varyansı tam kesirli aritmetikle (`fractions.Fraction`) hesaplanan
  değerle karşılaştırılır — yöntemin kendisine hiç sorulmadan.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from qlab import run
from qlab.engine.spin.factory import build_spin
from qlab.observable.spin_obs import heat_capacity, susceptibility
from qlab.scenario.loader import load_scenario

_YAML = """\
kind: spin
name: donmus hizali kafes
model: ising
algorithm: metropolis
lattice_size: 8
temperatures: [2.0]
n_sweep: 1000
n_burn: 10
seed: 7
coupling: 1.0
field: {h}
"""


@pytest.mark.parametrize("h", ["1e10", "1e100", "-1e100"])
def test_KAMU_YOLU_donmus_kafeste_C_SIFIR_ve_NEGATIF_DEGIL(tmp_path, h):
    """FİZİK ÇIPASI, kamu yolu (`qlab.run` → npz): C = 0, χ = 0, ikisi de ≥ 0.

    HEAD'de aynı yol h = 1e10'da C = 262144, h = 1e100'de C = −2.7e185
    yazıyordu (64 ölçümle; burada 1000 ölçüm: düz iki geçişli — kaydırmasız —
    bir varyans da bu uzunlukta sabit −6.4e101 serisinde ortalamanın kendi
    yuvarlamasından 1.5e172 bırakır, bu yüzden 64 değil 1000).
    Eşik 1e-12: fizik 0, düzeltilmiş yolda ölçülen tam 0.0 (sabit seride
    x − x = 0); son-ulp kilitlenmez."""
    yol = tmp_path / "donmus.yaml"
    yol.write_text(_YAML.format(h=h), encoding="utf-8")

    seri = build_spin(load_scenario(yol)).samples[0].energy_series
    assert np.ptp(seri) == 0.0, "öncül: donmuş kafeste enerji serisi sabit olmalı"

    assert run.main([str(yol), "-o", str(tmp_path / "donmus.npz")]) == 0
    with np.load(tmp_path / "donmus.npz", allow_pickle=False) as v:
        C = float(v["heat_capacity"][0])
        chi = float(v["susceptibility"][0])
    assert 0.0 <= C <= 1e-12, C
    assert 0.0 <= chi <= 1e-12, chi


@pytest.mark.parametrize("ofset", [0.0, 1e6, 1e10, 1e12, -1e12])
def test_ENERJI_SIFIRI_KEYFI_C_ofsetle_DEGISMEZ(ofset):
    """FİZİK ÇIPASI: enerji sıfırının seçimi C'yi değiştiremez.

    Seri −50 ± 3 (tam ikili: varyans TAM 9); ofsetli değerler tam sayı
    olduğu için temsil kaybı yok. Ölçüldü (HEAD formülü): ofset 1e10 → 16384,
    1e12 → −134217728, −1e12 → −268435456 (beklenen 9). Düzeltilmiş yol beş
    ofsette de 9.0. Tolerans bağıl 1e-12."""
    e = -50.0 + 3.0 * np.tile([1.0, -1.0], 500) + ofset
    assert heat_capacity(e, L=4, T=2.0) == pytest.approx(9.0 / (16 * 4.0), rel=1e-12, abs=0.0)


def _tam_varyans(a: np.ndarray) -> float:
    kesirler = [Fraction(float(x)) for x in a]
    ort = sum(kesirler) / len(kesirler)
    return float(sum((k - ort) ** 2 for k in kesirler) / len(kesirler))


def test_BUYUK_ORTALAMA_KUCUK_DALGALANMA_tam_kesirli_varyansla_AYNI():
    """BAĞIMSIZ YÖNTEM ÇIPASI: |m| = 0.9 çevresinde 1e-9'luk dalgalanma.

    Tam kesirli varyans 6.25e-19. Ölçüldü: HEAD formülü bağıl 179 kat hatalı
    (iptal tabanı ~1e-16 ≫ 6e-19), kaydırmasız iki geçiş 7.9e-14, düzeltilmiş
    yol 1.5e-16. Tolerans bağıl 1e-9 (yuvarlamanın ~1e7 katı, HEAD hatasının
    ~1e-11'i). Aynı seri ısı sığasından da geçer (aynı yardımcı, L²T² bölen)."""
    a = 0.9 + 1e-9 * np.tile([1.0, -1.0, 0.5, -0.5], 250)
    tam = _tam_varyans(a)
    assert tam > 6e-19, "öncül: dalgalanma float'ta temsil edilmiş olmalı"
    assert susceptibility(a, L=4, T=2.0) == pytest.approx(16 * tam / 2.0, rel=1e-9, abs=0.0)
    assert heat_capacity(a, L=4, T=2.0) == pytest.approx(tam / (16 * 4.0), rel=1e-9, abs=0.0)
