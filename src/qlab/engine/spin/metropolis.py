"""Checkerboard Metropolis -- KOSTURDUGU MODELI BILMEZ.

Bu modul `IsingModel` ve `HeisenbergModel` adlarini HIC gecirmez. Gordugu tek
sey `SpinModel` protokoludur: `n_component` ve `propose`. Kabul/red karari
buradadir ve `dE` MODELDEN BAGIMSIZ hesaplanir --

    dE = -J sum_k (s'_k - s_k) . komsu_k  -  h (s'^z - s^z)

Bu ayrim bir stil tercihi degil, Modul 4'un Mermin-Wagner cipasinin ON
KOSULUDUR (spec §3.4): iki model ayri ornekleyiciden gecerse "ayni kod, zit
hukum" iddiasi duser. Cekirdegin modele yalnizca protokolden baktigi
`tests/spin/test_metropolis.py`de UC ayri testle kilitlenir (casus modelin
cagri sayimi, testte tanimlanan `n_component = 2` XY modeli, ve J = 0'da iki
modelin iki FARKLI kapali formu vermesi).

CHECKERBOARD, VE NEDEN CIFT L ZORUNLU:
Bir yari-adimda yalniz TEK bir alt-orgu guncellenir; o alt-orgunun hicbir iki
sitesi komsu olmadigi icin butun oneriler ayni `komsu` alaniyla dogru sekilde
degerlendirilir. Periyodik kare kafes bunu YALNIZ CIFT L'de saglar. Sayildi:
ayni alt-orgudeki komsu bag sayisi L=5'te 50 bagin 10'u (%20.0), L=9'da
162'nin 18'i (%11.1), L=17'de 578'in 34'u (%5.9); cift L'de 0. Tek L bu
yuzden reddedilir -- bozulma sessizdir ve sonuc makul gorunur.

Maskenin gercekten calistigi da olculdu: maske kaldirilip tum siteler ayni
anda guncellenince L=4, T=2.0, h=0 icin `<|m|>` 0.918943 yerine 0.000000 ve
`<E>` -28.086085 yerine +29.0 cikiyor -- paralel guncelleme bir limit
dongusune dusuyor ve enerjinin ISARETI bile ters donuyor.

TASMA KORUMASI -- SINIR OLCULEREK SECILDI:
`np.exp`in argumani `-dE/T` KIRPILIR (`_USTEL_SINIR = 50`). Kirpma dE'ye
DEGIL usse uygulanir; plan taslagi dE'yi kirpiyordu ve o, kucuk T'de tasmayi
ONLEMEZ (dE = -50, T = 0.005 icin `exp(10000)` = inf). Ustel kirpildiginda
tasma her T icin imkansizdir ve fizik degismez: `exp(+50) = 5.2e21` zaten
"her zaman kabul", `exp(-50) = 1.9e-22` ise `rng.random()`in cozunurlugunun
(2^-53 ~ 1.1e-16) cok altinda, yani "asla kabul".

Sinirin GEVSETILEMEZ oldugu olculdu (planin "olc" dedigi mutant): sinir
+-5'e indirilirse Ising'in `<|m|>`si Onsager-Yang'in tam degerinden
-1.39e-02 (T=0.8) / -1.40e-02 (T=1.0) / -9.24e-03 (T=1.4) sapar; +-1'de
miknatislanma tamamen coker (0.040389). Yani +-5 FIZIGI BOZAR, "test
gormuyor" degil. Ilginc olan su: ayni mutantta Heisenberg'in Mermin-Wagner
egimi -1.0062'den yalnizca -0.9864'e gidiyor ve sinirin icinde kaliyor --
yani bu mutanti Mermin-Wagner testi DEGIL, Onsager-Yang testi yakalar.

BEYAN EDILEN MALIYET: `model.propose` HER site icin oneri uretir, ama bir
yari-adimda sitelerin yarisi kullanilir. Yani Heisenberg'de rastgele sayilarin
yarisi bosa gider. Bu bilincli: protokolu "yalniz su siteler icin oneri uret"
diye daraltmak, modele maske bilgisini gecirmek olurdu ve cekirdegin modeli
bilmemesi iddiasini zayiflatirdi. Olculen bedel iki kat rastgele sayi; L=32
Heisenberg'de 2000 sweep 0.54 s suruyor.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from qlab.engine.spin.lattice import (
    SpinState,
    magnetisation,
    neighbour_sum,
    total_energy,
)
from qlab.engine.spin.models import SpinModel
from qlab.errors import InvalidInputError

_USTEL_SINIR = 50.0


@dataclass(frozen=True, eq=False)
class SamplingResult:
    """Bir ornekleme kosumunun sonucu.

    `magnetisation_series`: olcum basina `|m|` (bilesen vektorunun NORMU;
    n_component = 1'de mutlak deger). Burn-in ATILMIS haldedir, uzunlugu
    `n_sweep`tir.

    `energy_series` ve `acceptance_rate` OPSIYONELDIR ve `None` "OLCULMEDI"
    demektir -- SIFIR DEGIL. Wolff (Task 3) kume cevirdigi icin enerjiyi adim
    basina ucuza guncelleyemez; her olcumde bastan hesaplamak O(L^2) ekler ve
    bu modulun hicbir cipasi Wolff'un enerjisine ihtiyac duymaz (isi sigasi
    Metropolis'ten gelir). `observable` tarafi `None`u sifir SANMAMALIDIR --
    bu depoda "analiz edilmedi ile kararli ayni degildir" dersinin (Modul 1B
    kararlilik alani) aynisi. Wolff'ta `acceptance_rate` de anlamsizdir
    (kume her zaman cevrilir) ve `None` kalir.

    `eq=False`: alanlar ndarray oldugu icin uretilen `__eq__` bir dizi
    dondurur ve uretilen `__hash__` `TypeError` verir. Depo bu tuzagi
    `engine/hf/integrals.py`, `engine/md/state.py` ve `spin/lattice.py`de
    zaten odedi; ayni gerekce.
    """

    magnetisation_series: np.ndarray
    final_state: SpinState
    n_sweep: int
    algorithm: str
    energy_series: np.ndarray | None = None
    acceptance_rate: float | None = None


@functools.lru_cache(maxsize=None)
def _alt_orgu_maskeleri(L: int) -> tuple[np.ndarray, np.ndarray]:
    """`(i + j) % 2` paritesine gore iki alt-orgu maskesi.

    Onbelleklenir cunku her sweep'te iki kez gerekir ve L kosum boyunca
    sabittir. Donen diziler YAZILAMAZ isaretlenir; onbellekteki bir diziyi
    cagiran degistirirse butun kosum sessizce bozulurdu.
    """
    i, j = np.indices((L, L))
    parite = (i + j) % 2
    maskeler = (parite == 0, parite == 1)
    for maske in maskeler:
        maske.flags.writeable = False
    return maskeler


def _dogrula(state: SpinState, model: SpinModel, temperature: float) -> None:
    if state.L % 2 != 0:
        raise InvalidInputError(
            f"checkerboard Metropolis CIFT L gerektirir, verilen: L = {state.L}. "
            "Tek L'de periyodik kare kafes iki-parcali degildir ve ayni "
            "alt-orgudeki baglarin iki ucu ayni anda guncellenir "
            f"(L = {state.L} icin {2 * state.L} bag), ayrintili denge bozulur."
        )
    if state.n_component != model.n_component:
        raise InvalidInputError(
            f"model ve kafes n_component'i uyusmuyor: model {model.n_component}, "
            f"kafes {state.n_component}. Uyusmazlik SESSIZCE yanlis fizik verir "
            "(ornegin Heisenberg onerisi (L, L, 1) dizide +-1 dondurur)."
        )
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise InvalidInputError(
            f"sicaklik pozitif ve sonlu olmali, verilen: {temperature}"
        )


def metropolis_sweep(
    state: SpinState,
    model: SpinModel,
    *,
    temperature: float,
    coupling: float,
    field: float,
    rng: np.random.Generator,
) -> SpinState:
    """Bir tam sweep: iki alt-orgu sirayla, her site icin TEK oneri.

    Girdiyi DEGISTIRMEZ; yeni bir `SpinState` dondurur.

    `komsu` her yari-adimda YENIDEN hesaplanir. Bir kez hesaplayip iki
    yari-adimda kullanmak, ikinci alt-orgunun bayat komsu alanini gormesi
    demek olurdu ve ayrintili dengeyi bozardi.
    """
    _dogrula(state, model, temperature)
    spins = state.spins
    L = state.L

    for maske in _alt_orgu_maskeleri(L):
        komsu = neighbour_sum(spins)
        aday = model.propose(spins, rng)
        fark = aday - spins
        delta = -coupling * np.sum(fark * komsu, axis=2) - field * fark[..., -1]
        ustel = np.exp(
            -np.clip(delta / temperature, -_USTEL_SINIR, _USTEL_SINIR)
        )
        kabul = rng.random((L, L)) < ustel
        spins = np.where((kabul & maske)[..., None], aday, spins)

    return SpinState(spins)


def run_metropolis(
    state: SpinState,
    model: SpinModel,
    *,
    temperature: float,
    coupling: float,
    field: float,
    rng: np.random.Generator,
    n_sweep: int,
    n_burn: int,
) -> SamplingResult:
    """`n_burn` sweep at, sonraki `n_sweep` sweep'i olc.

    Olcum sweep'ten SONRA alinir, yani seri denge dagilimindan `n_sweep`
    ornek tasir. Ardisik ornekler ILISKILIDIR; hata cubugu Task 4'un
    `binning_error`iyla hesaplanir ve NAIF SEM bu depoda hicbir yerde
    raporlanmaz.

    KABUL ORANI NASIL OLCULUYOR: `metropolis_sweep` yalniz durum dondurur
    (imza plan tarafindan sabitlenmistir), bu yuzden kabul orani DEGISEN SITE
    SAYISINDAN turetilir. Ising'de bu tamdir (kabul edilen oneri her zaman
    isareti cevirir). Heisenberg ve XY gibi surekli modellerde "kabul edildi
    ama spin ayni kaldi" olayinin olcusu SIFIRDIR (oneri surekli bir
    dagilimdan gelir), yani orada da tamdir. Yalniz burn-in SONRASI sweep'ler
    sayilir: burn-in'deki kabul orani denge degeri DEGILDIR.
    """
    _dogrula(state, model, temperature)
    if n_sweep < 1:
        raise InvalidInputError(f"n_sweep en az 1 olmali, verilen: {n_sweep}")
    if n_burn < 0:
        raise InvalidInputError(f"n_burn negatif olamaz, verilen: {n_burn}")

    m_serisi: list[float] = []
    e_serisi: list[float] = []
    degisen_toplam = 0

    for adim in range(n_burn + n_sweep):
        onceki = state.spins
        state = metropolis_sweep(
            state,
            model,
            temperature=temperature,
            coupling=coupling,
            field=field,
            rng=rng,
        )
        if adim >= n_burn:
            m_serisi.append(float(np.linalg.norm(magnetisation(state))))
            e_serisi.append(total_energy(state, coupling, field))
            degisen_toplam += int(
                np.count_nonzero(np.any(state.spins != onceki, axis=2))
            )

    return SamplingResult(
        magnetisation_series=np.array(m_serisi),
        final_state=state,
        n_sweep=n_sweep,
        algorithm="metropolis",
        energy_series=np.array(e_serisi),
        acceptance_rate=degisen_toplam / (n_sweep * state.L * state.L),
    )
