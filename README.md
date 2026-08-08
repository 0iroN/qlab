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
- **Modül 1B — Hartree-Fock: tamam.** Kısıtlı Hartree-Fock (RHF), keyfi s/p
  kabuğu için McMurchie-Davidson (MD) özyinelemesiyle: Boys fonksiyonu, tek-
  ve iki-elektron integralleri (örtüşme, kinetik, çekirdek-çekim, elektron
  itmesi), DIIS ile hızlandırılmış SCF, analiz katmanı (Mulliken/Löwdin
  yükleri, dipol, Mayer bağ mertebesi, HOMO/LUMO), PySCF referans testleri
  (`reference/*.json`, oturum kapsamlı fixture, koşarken PySCF'e bağımlı
  değil) ve YAML senaryo entegrasyonu (`h2_bond_scan.yaml`,
  `water_angle_scan.yaml`, headless koşucu üzerinden). MD, Obara-Saika
  yerine seçildi: dört integral tipi de aynı iki özyinelemeyi (Hermite
  genişletme katsayıları E, Hermite-Coulomb yardımcıları R) paylaşır, yani
  tek bir türetme doğrulandığında dördü birden doğrulanmış olur — gerekçe
  `engine/hf/integrals.py` modül dokstring'inde. Ölçülen sayılar aşağıdaki
  Doğrulama tablosundadır.
- **Modül 1B2 — orbital görselleştirme: henüz yazılmadı, ayrı bir plana
  (Plan B2) ait.** `observable/hf_obs.py` (moleküler orbitali kartezyen
  ızgarada değerlendirme, elektron yoğunluğu, yoğunluk farkı haritası) ve
  `viz/volume.py` (PyVista marching cubes izoyüzeyleri, GUI entegrasyonu)
  bu depoda YOKTUR. Kabul kriteri 3 (GUI'de döndürülebilir H₂O HOMO
  izoyüzeyi) bu yüzden kapanmamıştır — bkz. "Kabul kriterleri durumu".
- **Modül 2 — moleküler dinamik: planlanıyor.** Henüz başlamadı.

## Ne iddia etmiyor

Bu yazılım "tam kuantum, tamamen gerçekçi" bir simülatör değildir. Çok-elektronlu
Schrödinger denklemi 3N boyutludur; tam çözüm ~20 elektronda tıkanır. Her yöntem
bir yaklaşımdır. Buradaki duruş yaklaşımı gizlemek değil, ölçüp ekranda göstermektir.

Hartree-Fock (RHF) de bir yaklaşımdır, TDSE motoru kadar. Yaklaşımları: Born-
Oppenheimer (çekirdekler sabit), tek Slater determinantı, sonlu baz seti
(STO-3G/6-31G), kapalı kabuk (çift işgal). Korelasyon enerjisi eksiktir — bu
RHF'nin bilinen temel eksiğidir — ama bu depoda ölçülmüş bir korelasyon-enerjisi
referansı (örneğin bir CI/CCSD karşılaştırması) YOKTUR, dolayısıyla buraya bir
sayı yazılmıyor.

RHF ayrışmayı doğru tanımlayamaz, ve bu ölçülmüştür: H₂'yi 0.74 Å'dan 8.0 Å'ya
ayırmanın "bağlanma enerjisi" 14.6350 eV çıkar — deneysel ayrışma enerjisi
4.75 eV'nin yaklaşık üç katı (bkz. Doğrulama tablosu). Bunun nedeni eksik
korelasyon DEĞİLDİR; kısıtlı tek determinant H₂'yi iki nötr atoma ayrıştıramaz,
ayrışma limitinde %50 iyonik (H⁻ H⁺) kalır, bu yüzden E(sonsuz) olması
gerekenden çok yüksektir ve fazlalık doğrudan bağlanma enerjisine eklenir —
eksik korelasyon ters yönde çalışır ama bu hatanın yanında küçük kalır. Aynı
kusur ikinci, bağımsız bir gözlemlenebilirde de görülür: H₂ için Mayer bağ
mertebesi her ayrım mesafesinde tam 1.0'dir (çekirdekler ne kadar ayrılırsa
ayrılsın RHF bağ orbitalini çift işgal etmeye zorlar).

Çok büyük ayrımlarda (H₂, ~10-12 Å üstü) RHF çözümü sayısal olarak kırılır:
10 Å'da enerji asimptota uyar, 12 Å'da kopar. İntegraller sağlamdır (kontrol
edildi); kırılan adım özdeğer çözümüdür — iki atomik orbital arasındaki eşleşme
yuvarlama farkının altına düşünce `eigh` simetrik yerine yerelleşmiş, çok daha
yüksek enerjili bir çözüm seçer. Gerçek kuantum kimyası paketleri bunu
"stability analysis" ile yakalar; qlab'da o yoktur.

Qt penceresi bu depoda hiçbir gerçek ekranda henüz açılıp gözle doğrulanmadı.
Projedeki her GUI testi `QT_QPA_PLATFORM=offscreen` ile ekransız çalışır
(bkz. `tests/test_transparency.py`) — widget durumu programatik olarak okunur,
ama kimse pencereyi fiilen görmedi.

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

Her fizik iddiası bilinen bir referansa karşı test edilir.

### Modül 1A — TDSE

Aşağıdaki sayılar bu görevde (Task 15) gerçekten çalıştırılıp ölçülmüştür —
plandaki hedef değerler değil, ölçülen değerlerdir:

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

### Modül 1B — Hartree-Fock

Aşağıdaki sayılar bu görevde (B1 Task 13) gerçekten çalıştırılıp ölçülmüştür,
`reference/hf_energies.json` içindeki `_pyscf_version` alanına göre PySCF
2.14.0'a karşı:

| Test | Referans / tolerans | Ölçülen |
|---|---|---|
| PySCF toplam enerji uyumu (9 sistem: He, H₂ @0.74/1.00/2.00 Å, LiH, HF, H₂O, NH₃, CH₄; STO-3G) | `tests/hf/test_scf.py::test_total_energy_matches_pyscf` | en kötü sapma **1.32e-11 hartree** (H₂, R=2.00 Å); su -74.9630631297 hartree (fark 4.56e-12) |
| İntegral uyumu (S, T, V, ERI; H₂ ve H₂O, STO-3G) | `tests/hf/test_integrals.py`, atol=1e-10..3e-10 | en kötü mutlak sapma **2.129e-10** (su çekirdek-çekim matrisi, [5,1]) |
| H₂ denge bağ uzunluğu | Deneysel 0.741 Å, tolerans ±%8 | **0.7156 Å**, sapma %3.4 |
| H₂ bağlanma enerjisi (RHF/STO-3G, 0.74→8.0 Å) | Deneysel ayrışma enerjisi 4.75 eV | **14.6350 eV** — RHF ayrışma kusuru yüzünden ~3 kat fazla, korelasyon eksikliği DEĞİL (bkz. "Ne iddia etmiyor") |
| H₂O denge açısı | Deneysel 104.5°, tolerans ±%8 | **100.97°**, sapma %3.4 |
| H₂O dipolü | Deneysel 1.85 D | **1.7255 D**, sapma %6.7 |
| SCF yakınsama hızlandırma (DIIS vs düz Roothaan-Hall, iterasyon sayısı) | `tests/hf/test_scf.py::test_diis_reduces_iteration_count` | H₂O 19→8, LiH 23→7, NH₃ 18→8, CH₄ 12→11 |
| Yakınsamayan SCF sessizce sayı döndürmemeli | `tests/hf/test_scf.py::test_non_convergence_raises_rather_than_returning` | `SCFConvergenceError` fırlatılır, iterasyon/enerji geçmişi hatanın içinde taşınır |
| Referans koruması gerçekten korur mu | `reference/hf_energies.json` geçici kaldırılıp `pytest tests/hf/ -v` koşuldu | **kırmızı**, atlanmadı: 134 passed, 64 error — hepsi `pytest.fail`: "Referans dosyasi bulunamadi" |

Not: İntegral satırındaki 2.129e-10'luk artık özyinelemenin kendisinden değil,
referansı üreten PySCF derlemesinin bohr sabitinden (CODATA 2010,
0.52917721092 Å) qlab'ın kullandığı sabitten (CODATA 2018, 0.529177210903 Å)
gelir; bu fark çekirdek konumlarına geçer ve oradan her integrale kendi
çekirdek-konumu türevi üzerinden sızar. PySCF'in kendi bohr sabitiyle kurulan
aynı geometride dört tensör de rtol=0, atol=1e-12 ile eşleşir
(`test_water_matches_pyscf_to_machine_precision_at_pyscf_geometry`) — bu,
özyinelemeyi kilitleyen asıl kapıdır; yukarıdaki 1e-10 seviyesi birim-sabiti
farkını ölçer, integral hassasiyetini değil.

## Mimari kuralı

`engine` paketi `viz` veya `app` import **edemez**. Kural yorumla değil
import-linter ile korunur: `pytest tests/test_architecture.py`.

## Kabul kriterleri durumu

Spec §15'teki yedi maddenin gerçek durumu (ölçüm ayrıntıları
`.superpowers/sdd/task-15-report.md` — Modül 1A — ve
`.superpowers/sdd/b1-task-13-report.md` — Modül 1B — içindedir):

| # | Kriter | Durum |
|---|---|---|
| 1 | `pytest` tam yeşil, PySCF referans testleri dahil | **Kapandı** — 335/335 test yeşil (`pytest tests/ -v`), PySCF referans testleri dahil; referans dosyası kaldırıldığında testler ATLANMIYOR, KIRMIZI oluyor — kanıtlandı (134 passed, 64 error) |
| 2 | Headless koşucu çalışır + H₂ eğrisi bağ uzunluğu ±%8 | **Kapandı** — `python -m qlab.run scenarios/h2_bond_scan.yaml`; denge bağ uzunluğu 0.7156 Å, sapma %3.4 |
| 3 | GUI'de H₂O HOMO izoyüzeyi döndürülebilir | **AÇIK** — Hartree-Fock artık var ama orbital görselleştirme yok; `observable/hf_obs.py` ve `viz/volume.py` Plan B2'ye ait, henüz yazılmadı |
| 4 | Kaydırıcı ile canlı iletim, analitiğin ±%2'si içinde | **Kapandı** — ölçülen fark %0.07 |
| 5 | Şeffaflık paneli her hesapta dolu | **Kapandı** — TDSE tarafında pencere paneli; HF tarafında headless koşucu her senaryoda Yöntem/Yaklaşımlar/İhmal edilen/Beklenen hata/Yakınsama satırlarını basar (`h2_bond_scan.yaml` çıktısı) |
| 6 | Yakınsamayan SCF sessizce sayı döndürmez, test edilmiş | **Kapandı** — `SCFConvergenceError`, `test_non_convergence_raises_rather_than_returning` ile test edilmiştir |
| 7 | `engine`'in `viz`/`app` bağımlılığı yok, import-linter ile doğrulandı | **Kapandı** |

6/7 madde kapalıdır. Kapanmayan tek kriter 3'tür: GUI'de döndürülebilir H₂O
HOMO izoyüzeyi, orbital görselleştirme gerektirir ve bu Plan B2'nin kapsamıdır
(`observable/hf_obs.py`, `viz/volume.py`, GUI entegrasyonu) — burada "geçti"
gibi gösterilmemiştir. Ayrıca: Qt penceresinin kendisi bu depoda hiçbir gerçek
ekranda henüz açılmadı; tüm GUI testleri ekransız (`QT_QPA_PLATFORM=offscreen`)
koşar.
