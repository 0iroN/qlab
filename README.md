# qlab

Kuantum mekaniğini gerçek yöntemlerle çözen, sonucu görselleştiren ve **hangi
yaklaşımın yapıldığını her ekranda açıkça beyan eden** bir laboratuvar.

Tasarım belgesi: `../docs/superpowers/specs/2026-08-05-qlab-kuantum-cekirdegi-design.md`

## Durum

- **Modül 1A — TDSE motoru: tamam.** Tek parçacık, split-operator FFT.
  Motor çekirdeği — ızgara, potansiyeller, başlangıç paketi, propagatör ve
  observable katmanı — 1B/2B'dir ve ikisi de test edilmiştir (2B: `Grid2D`,
  `gaussian_packet_2d`, `double_slit`, `tdse_obs_2d`, çift yarık girişim
  testi). Senaryo şeması/yükleyici/fabrika, headless koşucu, görselleştirme
  veri katmanı, Qt kabuğu ve şeffaflık paneli ise yalnızca 1B'dir: `GridSpec`
  hiçbir 2B alan taşımaz, `factory.build()` sabit biçimde `Grid1D` üretir, Qt
  penceresi yalnızca 1B çizer. 2B'ye bu YAML → fabrika → koşucu/GUI hattından
  ulaşılamaz — yalnızca engine/observable modülleri doğrudan import edilerek
  kullanılabilir.
- **Modül 1B — Hartree-Fock: henüz yazılmadı, ayrı bir plana (Plan B) ait.**
  Bu depoda `engine/hf/`, `observable/hf_obs.py`, `viz/volume.py`, PySCF
  referans testleri ve H₂/H₂O senaryoları YOKTUR. Aşağıdaki "Kabul kriterleri
  durumu" bölümü hangi maddenin bu yüzden kapanmadığını tek tek listeler.

## Ne iddia etmiyor

Bu yazılım "tam kuantum, tamamen gerçekçi" bir simülatör değildir. Çok-elektronlu
Schrödinger denklemi 3N boyutludur; tam çözüm ~20 elektronda tıkanır. Her yöntem
bir yaklaşımdır. Buradaki duruş yaklaşımı gizlemek değil, ölçüp ekranda göstermektir.

## Kurulum

```bash
python -m pip install -e ".[dev]"
```

## Kullanım

GUI olmadan senaryo koştur:

```bash
python -m qlab.run scenarios/tunneling.yaml -o out.npz
```

Etkileşimli pencere:

```bash
python -m qlab.app.main scenarios/tunneling.yaml
```

Testler:

```bash
pytest tests/ -v
pytest tests/ -v -m slow
```

## Doğrulama

Her fizik iddiası bilinen bir referansa karşı test edilir. Aşağıdaki sayılar
bu görevde (Task 15) gerçekten çalıştırılıp ölçülmüştür — plandaki hedef
değerler değil, ölçülen değerlerdir:

| Test | Referans / tolerans | Ölçülen |
|---|---|---|
| Norm korunumu | Üniterlik garantisi, eşik 1e-8 | 8.427e-12 (headless CLI koşusu, `tunneling.yaml`, 32000 adım) |
| Kutuda parçacık, taban durum (motor seviyesi) | Sonlu-duvar transandant denklemi, tolerans %0.05 | sapma **%-0.017** (measured=0.17762085, referans=0.17765102) |
| Harmonik taban durum | E₀ = ω/2, tolerans %0.01 | pratikte tam örtüşme (measured=0.500000000002) |
| Serbest paket yayılımı | σ₀√(1+(t/2mσ₀²)²), tolerans %1 | pratikte tam örtüşme (measured=7.0710678119, analitik=7.0710678119 — Gauss paketinin yayılımı analitik olarak da tam bu formülü izler) |
| dt yakınsaması | O(Δt²), beklenen oran 4.0 | ölçülen oran **4.0001** (dt=0.02→0.01, hata 9.093e-05→2.273e-05) |
| Tünelleme oranı | Paket-integre analitik T(E), disiplin hedefi %2 | ölçülen fark **%0.07** (T_ölçülen=0.14782, T_analitik=0.14791) |
| Çift yarık girişimi | ≥3 girişim tepesi (%5 eşik) | 3 tepe (y=±9.77, 0); tek-yarık kontrolüyle ayrım oranı 2.94× |

Not: "Kutuda parçacık" satırı motor seviyesinde ölçülür (`SplitOperatorPropagator`
imajiner zamanda gerçekten evrimletilir); saf analitik formülün ikinci, bağımsız
bir sayısal yöntemle (merkezi-fark köşegenleştirme) çapraz doğrulaması ayrıca
`tests/tdse/test_analytic.py::test_particle_in_box_eigenvalues` içinde vardır
ama qlab.engine'i hiç çağırmaz — motorun kanıtı değildir.

## Mimari kuralı

`engine` paketi `viz` veya `app` import **edemez**. Kural yorumla değil
import-linter ile korunur: `pytest tests/test_architecture.py`.

## Kabul kriterleri durumu

Spec §15'teki yedi maddenin gerçek durumu (ölçüm ayrıntıları
`.superpowers/sdd/task-15-report.md` içindedir):

| # | Kriter | Durum |
|---|---|---|
| 1 | `pytest` tam yeşil, PySCF referans testleri dahil | **Kısmen** — 125/125 test yeşil, ama PySCF referans testi bu depoda yok (Plan B) |
| 2 | Headless koşucu çalışır + H₂ eğrisi bağ uzunluğu ±%8 | **Kısmen** — headless koşucu tamam; H₂ eğrisi Plan B'ye ait (Hartree-Fock gerektirir) |
| 3 | GUI'de H₂O HOMO izoyüzeyi döndürülebilir | **Kapsam dışı** — Hartree-Fock gerektirir, Plan B |
| 4 | Kaydırıcı ile canlı iletim, analitiğin ±%2'si içinde | **Kapandı** — ölçülen fark %0.07 |
| 5 | Şeffaflık paneli her hesapta dolu | **Kapandı** (TDSE tarafı) |
| 6 | Yakınsamayan SCF sessizce sayı döndürmez, test edilmiş | **TDSE eşdeğeri kapandı** — üniterlik koruması (`test_raises_when_norm_breaks`) test edilmiştir; SCF'nin kendisi Plan B'ye ait |
| 7 | `engine`'in `viz`/`app` bağımlılığı yok, import-linter ile doğrulandı | **Kapandı** |

Bu modülün (Modül 1A — TDSE) kapsamına giren her şey kapalıdır. 1, 2 (ikinci
yarı), 3 ve 6'nın SCF kısmı Hartree-Fock motorunu gerektirdiği için Plan B'ye
aittir; burada "geçti" gibi gösterilmemiştir.
