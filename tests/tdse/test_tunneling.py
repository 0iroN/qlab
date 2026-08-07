"""Spec §5.3 / §5.4 — tünelleme oranının paket-integre analitikle karşılaştırması."""

import numpy as np
import pytest

from qlab.engine.tdse import potentials
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.initial import gaussian_packet
from qlab.engine.tdse.propagator import SplitOperatorPropagator, evolve
from qlab.observable import tdse_obs

from .analytic_reference import packet_transmission, transmission_coefficient

BARRIER_HEIGHT = 0.8
BARRIER_WIDTH = 2.0


def test_transmission_is_between_zero_and_one():
    for energy in (0.1, 0.5, 0.8, 1.5, 3.0):
        assert 0.0 <= transmission_coefficient(energy, BARRIER_HEIGHT, BARRIER_WIDTH) <= 1.0


def test_transmission_increases_with_energy_below_barrier():
    low = transmission_coefficient(0.2, BARRIER_HEIGHT, BARRIER_WIDTH)
    high = transmission_coefficient(0.7, BARRIER_HEIGHT, BARRIER_WIDTH)
    assert high > low


def test_transmission_is_continuous_at_barrier_top():
    below = transmission_coefficient(0.8 - 1e-7, BARRIER_HEIGHT, BARRIER_WIDTH)
    at_top = transmission_coefficient(0.8, BARRIER_HEIGHT, BARRIER_WIDTH)
    above = transmission_coefficient(0.8 + 1e-7, BARRIER_HEIGHT, BARRIER_WIDTH)
    assert below == pytest.approx(at_top, rel=1e-5)
    assert above == pytest.approx(at_top, rel=1e-5)


def test_wide_barrier_blocks_more_than_narrow():
    narrow = transmission_coefficient(0.5, BARRIER_HEIGHT, 1.0)
    wide = transmission_coefficient(0.5, BARRIER_HEIGHT, 4.0)
    assert wide < narrow


@pytest.mark.slow
def test_simulated_transmission_matches_packet_integrated_analytic():
    """Sayısal iletim, paket-integre analitik değerle uyuşmalı.

    Parametreler bilinçli seçildi:
      - Kutu (400 bohr) yeterince büyük: 160 a.u. sonunda hiçbir bileşen kenara
        ulaşmaz, bu yüzden soğurucu maske GEREKMEZ ve norm tam korunur.
      - sigma = 10 bohr, bariyer genişliğinden (2 bohr) çok büyük: durağan hal
        analizi geçerli.
      - E0 = 0.5 < V0 = 0.8: gerçek tünelleme rejimi, T ~ %16.

    İki ölçülmüş düzeltme (ölçümle doğrulandı, tolerans gevşetilmedi):

    1) dt = 0.005 (0.02 değil). n=4096, length=400 ızgarasında dt=0.02 ile
       split-operator'ın Trotter adımı k^2*dt/(2m) fazını 2π'nin katına
       getiren tek bir k'da (k^2 = 4π/dt => k ≈ 25.07; ölçülen zirve
       k ≈ 25.09) rezonans kuruyor: bariyerin keskin kenarlarının saçtığı
       ihmal edilebilir enerji orada uyumlu birikip tek bir düzlem-dalga
       moduna dönüşüyor (bu yüzden ızgara boyunca DÜZ yayılıyor — bu bir
       sarma değil, tek modlu bir Fourier bileşeni). Bu, edge_probability'yi
       1.78e-4'e (eşiğin 17800 katı) şişiriyor. dt=0.005'te rezonans k'sı
       (≈50.1) ızgaranın Nyquist sınırının (≈32.2) üstüne çıkıyor ve
       edge_probability serbest-yayılım gürültü tabanına (~1.8e-10) düşüyor
       — eşik (1e-8) GEVŞETİLMEDİ, çünkü artık gerçekten karşılanıyor.

    2) Analitik karşılaştırmada bariyerin İZGARADA GERÇEKTEN VAR OLAN
       genişliği kullanılıyor, nominal width=2.0 değil. rectangular_barrier
       np.where(|x|<=width/2,...) ile ayrıklaştırıyor; n=4096, length=400
       için bu tam olarak 21 nokta = 2.05078125 bohr veriyor (nominalden
       ~%2.5 farklı). Tünelleme olasılığı genişliğe ÜSTEL bağımlı olduğundan
       (sinh(kappa*width)^2), bu küçük fark sigma'dan bağımsız, kalıcı bir
       %6.9 sapma üretiyor (sigma 10->30 taranarak doğrulandı: sapma
       küçülmüyor, bu yüzden enerji-yayılımı/Jensen etkisi değil). Analitik
       modelin ızgarada GERÇEKTEN simüle edilen bariyeri tanımlaması daha
       dürüst bir karşılaştırmadır; bu nominal genişliği "düzeltmek" değil,
       neyin ölçüldüğünü doğru tarif etmektir.
    """
    grid = Grid1D(n=4096, length=400.0)
    psi0 = gaussian_packet(grid, x0=-80.0, k0=1.0, sigma=10.0)
    potential = potentials.rectangular_barrier(
        grid, height=BARRIER_HEIGHT, width=BARRIER_WIDTH, center=0.0
    )
    prop = SplitOperatorPropagator(grid=grid, potential=potential, dt=0.005)

    frames, _ = evolve(prop, psi0, n_steps=32000, save_every=32000)
    final = frames[-1]

    # Sarma olmadığını kanıtla — bu kontrol olmadan ölçüm anlamsız
    assert tdse_obs.edge_probability(final, grid, width=40.0) < 1e-8
    assert tdse_obs.norm(final, grid) == pytest.approx(1.0, abs=1e-10)

    measured = tdse_obs.probability_beyond(final, grid, boundary=5.0)
    effective_width = float(np.count_nonzero(potential)) * grid.dx
    predicted = packet_transmission(psi0, grid, BARRIER_HEIGHT, effective_width)
    relative_error = abs(measured - predicted) / predicted

    print(
        f"\nÖlçülen T = {measured:.5f}, paket-integre analitik T = {predicted:.5f} "
        f"(izgara genişliği={effective_width:.5f}), bağıl fark = "
        f"{100 * relative_error:.2f}%"
    )
    assert relative_error < 0.03
