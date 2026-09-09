"""Kopru haritasi yuku. FIZIK URETMEZ; okur, cevirir ve toplar.

`viz` katmani `observable`i, `bridge_map`i, `provenance`i ve `scenario`yu
import EDEBILIR (katman sozlesmesi: run -> viz -> observable -> engine ->
scenario -> provenance -> types : bridge_map). Bu dosya tam da o yuzden
buradadir: panelin METNI `qlab.bridge_map`te (saf tanim katmani, hicbir
motoru goremez), SAYILARI `observable/bridge_obs.py`de (kenar tabani) ve
`bridge_map.olculen_*` tarayicilarinda durur; ikisini de goren ve
`Provenance`i kurabilen ilk katman bu katmandir. `engine/grav/factory.py`nin
"sayilar disaridan gelir, metin fabrikada kalir" deseninin ayni, tek farkla:
burada bir motor yok, o yuzden fabrika `engine`de degil `bridge_map`tedir ve
`Provenance` nesnesini `viz` kurar.

BU DOSYA HICBIR BEYAN METNI YAZMAZ. Yontem / Yaklasimlar / Ihmal edilen /
Beklenen hata / Dairesellik / Cipa sinifi / Kuram secici / Pencere /
Yakinsama tanisi satirlarinin tek kaynagi `bridge_map.panel_metinleri()`dir;
burada yalnizca `EdgeFloor` duz demete cevrilir ve sozluk `Provenance`a
acilir. Kopya sayisi `tests/bridge/test_bridge_senaryo.py` ile kilitlidir
ve kilit PARCA duzeyindedir.

KATMAN SOZLESMESI OLCULDU (Task 5): `bridge_map -> qlab.observable.bridge_obs`
enjeksiyonu `lint-imports` ile rc=1 ("Katmanli mimari BROKEN"), bu dosyaya
`import qlab.run` enjeksiyonu da rc=1; kontrol olarak bu dosyanin mesru
import'lari rc=0. Ikisi de `tests/test_architecture.py`de parametre olarak
duruyor.

MALIYET OLCULDU (bu makine, 03.09.2026): A1 tabani 0.495 s (8 YOL B kosumu,
Morse/harmonik), A2 tabani 0.123 s (2 kosum), kenar taramasi 0.047 s
(48 dosya, `ast`); toplam ~0.7 s. HF hic cozulmez.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from qlab.bridge_map import (
    KOVA_TABLOSU,
    KOVALAR,
    olculen_kenarlar,
    olculen_units_ekseni,
    panel_metinleri,
    tablo_tutarsizliklari,
)
from qlab.observable.bridge_obs import GERCEK_HATA_CM, EdgeFloor, edge_noise_floor
from qlab.provenance import Provenance
from qlab.scenario.schema import BridgeScenario


@dataclass(frozen=True)
class BridgePayload:
    """Bir kopru-haritasi kosumunun yazilabilir hali -- HICBIRI YUVARLANMAMIS.

    `kenarlar` ve `eksen` bu kosumda `ast` ile TARANIR (elle yazilmaz),
    `kova_sayim` tablodan SAYILIR, `tutarsizliklar` ikisinin tabloyla
    celistigi noktalardir ve bos olmasi panelin tik kosuludur. `a1`/`a2`
    kenarin iki tabanidir ve iki ayri `EdgeFloor`dur: tek alana
    dusurulmezler. `gercek_hata` `bridge_obs.GERCEK_HATA_CM`nin kaydidir;
    HF borusu burada KOSMAZ (slow testte canli uretilir).
    """

    scenario: BridgeScenario
    kenarlar: frozenset[tuple[str, str]]
    eksen: dict[str, int]
    kova_sayim: dict[str, int]
    tutarsizliklar: tuple[str, ...]
    a1: EdgeFloor
    a2: EdgeFloor
    gercek_hata: float


def build_bridge_payload(
    scenario: BridgeScenario, *, motor_koku: Path | str | None = None
) -> BridgePayload:
    """Tabloyu koda karsi tarar ve kenarin IKI tabanini olcer.

    `motor_koku` YALNIZ sinama icindir (sentetik bir `engine/` agaci verilir
    ki "tablo bayatlarsa tik duser" iddiasi olculebilsin); varsayilan gercek
    agactir ve senaryo bu dugmeyi TASIMAZ -- taranan agac bir pencere
    dugmesi degil, deponun kendisidir.

    Iki taban ayni pencerede okunur: senaryonun `n_bolme` / `periyot_sayisi`
    / `genlikler`i `edge_noise_floor`a gecer ve etikete islenir; A1'in
    merdiveni cagirana kapalidir, A2'nin ikizi `a_ikiz`dir.
    """
    kenarlar = olculen_kenarlar(motor_koku)
    eksen = olculen_units_ekseni(motor_koku)
    sayim = Counter(f.kova for f in KOVA_TABLOSU)
    kova_sayim = {k: int(sayim.get(k, 0)) for k in KOVALAR}
    pencere = dict(
        n_bolme=scenario.n_bolme,
        periyot_sayisi=scenario.periyot_sayisi,
        genlikler=tuple(scenario.genlikler),
    )
    return BridgePayload(
        scenario=scenario,
        kenarlar=kenarlar,
        eksen=eksen,
        kova_sayim=kova_sayim,
        tutarsizliklar=tablo_tutarsizliklari(kenarlar, eksen),
        a1=edge_noise_floor(tur="aritmetik", **pencere),
        a2=edge_noise_floor(tur="hata-cubugu", a=scenario.a_ikiz, **pencere),
        gercek_hata=GERCEK_HATA_CM,
    )


def _demet(f: EdgeFloor) -> tuple[float, float, float, str, str]:
    """`EdgeFloor` -> duz demet. Katman sinirinin ta kendisi.

    `bridge_map` `EdgeFloor`u goremez (`observable` ondan yukarida); ceviri
    bu dosyada, yani ikisini de gorebilen katmanda yapilir. Hicbir alan
    yeniden hesaplanmaz, hicbiri yuvarlanmaz.
    """
    return (f.taban, f.sinyal, f.oran, f.hukum, f.boru)


def provenance_for(yuk: BridgePayload) -> Provenance:
    """Olculmus sayilardan panel beyanini kurar. HICBIR METIN YAZMAZ.

    Butun metin `bridge_map.panel_metinleri()`nden gelir; bu fonksiyon
    sozlugu `Provenance`a acar. `converged` orada baglanir: tablo kodla
    tutarli VE iki taban da SAGLAM.
    """
    s = yuk.scenario
    metin = panel_metinleri(
        ad=s.name,
        a_ikiz=s.a_ikiz,
        n_bolme=s.n_bolme,
        periyot_sayisi=s.periyot_sayisi,
        genlikler=tuple(s.genlikler),
        kova_sayim=yuk.kova_sayim,
        kenarlar=yuk.kenarlar,
        eksen=yuk.eksen,
        a1=_demet(yuk.a1),
        a2=_demet(yuk.a2),
        gercek_hata=yuk.gercek_hata,
        tutarsizliklar=yuk.tutarsizliklar,
    )
    return Provenance(**metin)  # type: ignore[arg-type]
