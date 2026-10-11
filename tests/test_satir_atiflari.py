"""Depodaki her `dosya.py:N` / `dosya.yaml:N` atfı BUGÜNÜN satırını gösterir — kilit.

ÖLÇÜLDÜ 01.10.2026: 49 `x.py:N` atfının 6'sı yanlış satırı gösteriyordu; hepsi
yazıldığı gün doğruydu ve hedef dosyaya satır eklenince SESSİZCE kaymıştı
(`scf.py:395` → 478, …). Düzyazıdaki "296-303. satırlar" baştan yanlıştı.
Kilitli atıfların (`ast` `lineno`su ya da anahtar kelime) HİÇBİRİ kaymamıştı ve
depo kaymayı dört ayrı olayda YEREL olarak yakalamıştı (`:126`, `:10` → `:13`,
`h2_aimd.yaml:13`, `:16`/`:18`). Bu test kuralı genelleştirir: her atıf bir
SİMGEYE bağlanır, simge kayarsa test bugünkü satırı söyler. Göreli biçim
(`` `:174-178` ``), `satır N`, `#L`, en-tire aralık görünmez (02.10 doğrulayıcıyla
ölçüldü); tam biçim yazılır. İlk koşu düzeltme turunun KENDİ
hatasını yakaladı (`spin_obs.py:212` doğruydu, 213'e "düzeltilmişti").
"""

import re
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
ATIF = re.compile(r"(?<![\w/.])((?:\w+/)*\w+\.(?:py|yaml|toml)):(\d+)(?:-(\d+))?")
TARANAN = ("scenarios", "reference", "tools", "src", "tests")
UZANTI = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json"}

#: hedef -> (ilk satırın simgesi,) ya da (ilk satırın, son satırın simgesi)
ATIFLAR = {
    "scenarios/h2_aimd.yaml:3": ("k = 0.5729363",),
    "scenarios/h2_aimd.yaml:10": ("kind: hartree_fock",),
    "scenarios/h2_aimd.yaml:13": ("basis: sto-3g",),
    "src/qlab/engine/hf/integrals.py:241": ("x_pb=centre",),
    "src/qlab/engine/hf/scf.py:129": ('"Born-Oppenheimer"',),
    "src/qlab/engine/hf/scf.py:445": ("raise SCFConvergenceError(",),
    "src/qlab/engine/hf/scf.py:478": ("return HFResult(",),
    "src/qlab/engine/md/forcefields.py:13": ("from qlab.engine.hf.scf import",),
    "src/qlab/engine/spin/metropolis.py:189-191": ("delta = -coupling *", "delta / temperature"),
    "src/qlab/engine/tb/downfold.py:20": ("ANTIFERROMANYETIKTIR",),
    "src/qlab/engine/tb/downfold.py:130-132": ("qlab.engine.hf.basis", "qlab.engine.hf.multistart"),
    "pyproject.toml:27": ('markers = ["slow',),
    "src/qlab/engine/tb/downfold.py:259-271": ("hoppings = tuple(",),  # son satır `)`: ucu YALNIZ bridge_map atfı için ast kilitli
    "src/qlab/engine/tb/downfold.py:287": ("def model_from_downfold(",),
    "src/qlab/engine/tb/factory.py:143": ("iki-siteli antiferromanyetik hucre",),
    "src/qlab/engine/tb/factory.py:183-187": ("`converged` OLCULEBILIR", "yapi GERCEKTEN bulundu"),
    "src/qlab/observable/hf_obs.py:371-377": ("Ayni dik-olmama", "S^-1'in yaptigi is budur"),
    "src/qlab/observable/spin_obs.py:172": ('_seri(m, "miknatislanma serisi")',),
    "src/qlab/observable/spin_obs.py:212": ('_seri(m, "miknatislanma serisi")',),
    "src/qlab/scenario/schema.py:73": ("MD_FORCE_FIELD_KINDS = (",),
    "src/qlab/viz/frames.py:23-24": ("@dataclass(frozen=True)", "class FramePayload"),
    "src/qlab/viz/isosurface.py:55": ("import pyqtgraph",),
    "tests/md/test_aimd.py:38": ("MU_H2 / 0.5729363",),
    "tests/spin/test_spin_obs.py:572-574": ("(2.0, 0.644810791", "(3.5, 0.380777825"),
}

#: (atıf yapan dosya, atıf) -- gerçek dosya DEĞİL, tarayıcı testlerinin sentetik ağaçları.
SENTETIK = {("tests/bridge/test_kenar_tarayici.py", a) for a in ("__init__.py:1", "ortak.py:1", "q/r.py:3", "q/s.py:0")}
SENTETIK.add(("tests/bridge/test_bridge_senaryo.py", "spin/gizli.py:1"))


def _gorulen() -> tuple[dict[str, list[str]], set[tuple[str, str]]]:
    """(çözülen hedef -> atıf yerleri, görülen sentetik çiftler); kendisi taranmaz."""
    adaylar = [*(p for p in KOK.glob("*") if p.is_file()), *(p for d in TARANAN for p in (KOK / d).rglob("*"))]
    ben = Path(__file__).resolve()
    dosyalar = sorted(p.relative_to(KOK).as_posix() for p in adaylar if p.suffix in UZANTI and p.resolve() != ben)
    hedefler, sentetik = {}, set()
    for kaynak in dosyalar:
        for no, satir in enumerate((KOK / kaynak).read_text(encoding="utf-8").splitlines(), 1):
            for m in ATIF.finditer(satir):
                if (kaynak, m.group(0)) in SENTETIK:
                    sentetik.add((kaynak, m.group(0)))
                    continue
                yol = [d for d in dosyalar if d == m.group(1) or d.endswith("/" + m.group(1))]
                assert len(yol) == 1, f"{kaynak}:{no}: `{m.group(0)}` tek dosyaya çözülmüyor: {yol}"
                aralik = m.group(2) if m.group(3) is None else f"{m.group(2)}-{m.group(3)}"
                hedefler.setdefault(f"{yol[0]}:{aralik}", []).append(f"{kaynak}:{no}")
    return hedefler, sentetik


def test_her_atif_kayitli_ve_kayit_defteri_depoyla_BIREBIR():
    hedefler, sentetik = _gorulen()
    kayitsiz = {h: y for h, y in hedefler.items() if h not in ATIFLAR}
    assert not kayitsiz, f"kayıtsız atıf (ATIFLAR'a simgesiyle ekle): {kayitsiz}"
    assert set(ATIFLAR) == set(hedefler), f"artık atıf yapılmayan kayıt: {set(ATIFLAR) - set(hedefler)}"
    assert sentetik == SENTETIK, f"artık görülmeyen sentetik çift: {SENTETIK - sentetik}"


def test_her_atifin_simgesi_BUGUN_o_satirda():
    bozuk = []
    for hedef, simgeler in ATIFLAR.items():
        yol, aralik = hedef.rsplit(":", 1)
        bas, _, son = aralik.partition("-")
        satirlar = (KOK / yol).read_text(encoding="utf-8").splitlines()
        for no, simge in zip((int(bas), int(son or bas)), simgeler):
            if not 1 <= no <= len(satirlar) or simge not in satirlar[no - 1]:
                nerede = [i for i, s in enumerate(satirlar, 1) if simge in s]
                bozuk.append(f"{hedef}: `{simge}` {no}. satırda değil, bugün {nerede}")
    assert not bozuk, "atıf kaydı (atfı ve ATIFLAR'ı güncelle):\n" + "\n".join(bozuk)
