"""D11 (Osman kararı 11.10.2026) — ön ölçüm P6'nın `md-tb` yeniden koşusu KONUSUZ.

P6 (modül 7 ön ölçümü, 22.09.2026) `md-tb` çiftinin KOVASINI seçmek için üç
merceği aynı gözlemlenebilir (`observable/tb_obs.py`) üzerinde yeniden koşmayı
öneriyordu; mercekler TAUTOLOJI (0.0) / YAPISAL_YOK / kovayı düşüren
tekrar-üretim diye çatışıyordu. (P6'nın YAPISAL_YOK için andığı 17023.67 GERİ
ÇEKİLDİ: aynı ön ölçümün md-tb bölümü onu TAŞINAMAZ diye sildi — h → 0'da
ıraksayan bir oran.) `md -> tb` kod kenarı 25.09.2026'da (648f03b) işlendi ve o
günden beri kova ÖLÇÜMLE DEĞİL KİLİTLE belirlenir: kilit 1'in OLCULMEDI ve
YAPISAL_YOK kolları 09.09'dan (ab7d68c) beri vardır, TAUTOLOJI kolu ise ancak
11.10.2026'da (80eb21c, grup 4) eklendi — arada md-tb'yi TAUTOLOJI'ye taşımak
kilit 1'den geçerdi. README'deki kapalı not bu cümleyi kurar; bu dosya onu
BUGÜNKÜ GERÇEK taramada sınar: P6'nın varabileceği üç kovanın her biri
`bridge_map.tablo_tutarsizliklari`nin KİLİT 1'ine çarpar — kova sayımı
(kilit 3) bilerek uydurulduğu hâlde, yani kırmızıyı sayım değil kenar üretir.
"""

from __future__ import annotations

import dataclasses

import pytest

from qlab import bridge_map as bm

#: P6'nın üç merceğinin vardığı (ya da düşürdüğü) kovalar.
P6_KOVALARI = (bm.Kova.TAUTOLOJI, bm.Kova.YAPISAL_YOK, bm.Kova.OLCULMEDI)


@pytest.fixture(scope="module")
def gercek_tarama():
    return (
        bm.olculen_kenarlar(),
        bm.olculen_units_ekseni(),
        bm.taranamayanlar(),
        bm.olculen_kenar_siniflari(),
    )


def _sorunlar(gercek_tarama) -> tuple[str, ...]:
    kenarlar, eksen, taranamayan, siniflar = gercek_tarama
    return bm.tablo_tutarsizliklari(
        kenarlar, eksen, taranamayan=taranamayan, siniflar=siniflar
    )


def test_bugun_md_tb_KENARLI_GERCEK_KOPRU_ve_tablo_YESIL(gercek_tarama):
    """Öncül: kenar taranıyor, çift GERCEK_KOPRU'da, tablo tutarlı."""
    (girdi,) = [f for f in bm.KOVA_TABLOSU if f.cift == ("md", "tb")]
    assert girdi.kova == bm.Kova.GERCEK_KOPRU
    assert ("md", "tb") in gercek_tarama[0]
    assert _sorunlar(gercek_tarama) == ()


@pytest.mark.parametrize("kova", P6_KOVALARI)
def test_P6nin_HICBIR_kovasi_KILIT_1den_gecemez(monkeypatch, gercek_tarama, kova):
    """Çift P6'nın kovasına taşınır, sayım ona göre UYDURULUR; kilit 1 yine ateşler."""
    tablo = tuple(
        dataclasses.replace(f, kova=kova) if f.cift == ("md", "tb") else f
        for f in bm.KOVA_TABLOSU
    )
    sayim = dict(bm.KOVA_SAYILARI)
    sayim[bm.Kova.GERCEK_KOPRU] -= 1
    sayim[kova] += 1
    monkeypatch.setattr(bm, "KOVA_TABLOSU", tablo)
    monkeypatch.setattr(bm, "KOVA_SAYILARI", sayim)

    sorunlar = _sorunlar(gercek_tarama)
    assert not [s for s in sorunlar if s.startswith("kova sayimi")], sorunlar
    kilit_1 = [s for s in sorunlar if s.startswith("taranan kenar ('md', 'tb') tabloda")]
    assert kilit_1, f"{kova}: kilit 1 md-tb'yi reddetmedi: {sorunlar}"
