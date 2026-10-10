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
- **Modül 1B2 — orbital görselleştirme: tamam.** `observable/hf_obs.py`
  (moleküler orbitali kartezyen ızgarada değerlendirme, elektron yoğunluğu,
  promolekül ve yoğunluk farkı haritası), `engine/hf/analysis.py::analyse`
  (yedi analiz fonksiyonunu tek `Provenance` altında toplayan rapor),
  `viz/isosurface.py` (marching cubes ağ çıkarımı, ±faz için işaretli çift) ve
  `app/orbital_window.py` (döndürülebilir 3B pencere). Görselleştirme yığını
  **PyVista değil `pyqtgraph.opengl` + PyOpenGL**: ~150 MB yerine ~2 MB ve
  depoda zaten kullanılan tek render yığını — spec'ten bilinçli sapma.
  Kabul kriteri 3 kapandı ve bu, depodaki **ilk gerçek-ekran doğrulamasıdır**.
- **Modül 1C — kısıtlanmamış Hartree-Fock (UHF): tamam.** `engine/hf/uhf.py`
  (α/β Fock, birleşik DIIS, ⟨S²⟩), `engine/hf/multistart.py` (çok başlangıçlı
  RHF/UHF arama protokolü), `solve()`'a `initial_bias`, ve
  `stability.py::triplet_orbital_hessian` ile **dış (RHF→UHF, üçlü)
  kararlılık** analizi — panel ve kayıt artık iki kanalı ayrı ayrı basıyor.
  Modül bir özellik olarak değil **ön koşul** olarak yazıldı: Plan G'nin ön
  ölçümü referans sisteminin RHF çözümünün taban durum olmadığını gösterdi
  (çekirdek spec §13'ün "kapsam dışı" maddesi bu yüzden düzeltildi). Sevk
  edilmiş kodda **iki kusur** bulundu (N₂'de DIIS 19.78 eV, gerilmiş
  geometrilerde başlangıç tahmini 0.28–0.50 Ha) ve deponun kendi PySCF
  referans kaydı `h2_200`'ün gerçek HF taban durumundan **0.153420178832
  hartree** yukarıda olduğu ölçüldü. Modül 2D'nin etkilenmediği ölçülerek
  gösterildi. Ölçülen sayılar aşağıdaki Doğrulama bölümündedir.
- **Modül 2 (Plan C) — klasik moleküler dinamik: tamam.** Kuvvetten bağımsız
  bir MD çekirdeği: `MDState` + `ForceField` protokolü, velocity-Verlet,
  üç analitik kuvvet alanı (harmonik, Morse, Lennard-Jones — periyodik kutu
  ve kaydırılmış kesmeyle), iki termostat (Langevin BAOAB, Nosé-Hoover +
  zincir), sıcaklık/hız-dağılımı gözlemleri, YAML senaryo katmanı ve headless
  koşucu (`h2_morse.yaml`, `argon_nvt.yaml`), zaman serisi çizim yükü.
  Ölçülen sayılar aşağıdaki Doğrulama tablosundadır.
- **Modül 2D (Plan D) — ab initio moleküler dinamik: tamam.** Hartree-Fock
  enerjisinin sayısal gradyanı **dördüncü kuvvet alanı** olarak takıldı
  (`HartreeFockGradient`); çekirdekler artık Born-Oppenheimer yüzeyinde
  hareket ediyor ve modül **dairesel olmayan** bir fiziksel çıpa kazandı —
  titreşim frekansı yüzeyin kendisinden, iki bağımsız yoldan çıkıyor.
  Mimari sözü tutuldu ve git ile ölçüldü: `integrator.py`, `thermostats.py`
  ve `md_obs.py` üçlüsünde **0 satır** değişti; `factory.py` ise değişmek
  zorundaydı (dağıtım dalı + HF'ye özgü şeffaflık beyanı). Ölçülen sayılar
  aşağıdaki Doğrulama tablosundadır.
  **Kendi planına bırakılan** zamana bağlı yoğunluk farkı filmi Modül 2E'de
  yapıldı (aşağıda).
- **Modül 2E — zamana bağlı yoğunluk farkı: tamam.** Bağ orta noktasındaki
  yoğunluk farkı (`bond_midpoint_difference`) ve kare dizisi yükü
  (`viz/density_frames.py`). Modülün asıl kazancı bir kod değil bir **eleme**:
  en doğal görünen aday `max|fark|` ölçülerek **elendi** — bağ zayıflarken
  ters yöne gidiyor, değeri ızgaraya bağlı, ve asimetrik bağda orta noktayla
  işareti bile zıt. Kalan gözlemlenebilir çözünürlükten de kutu boyutundan da
  bağımsız (0.043821, dört kutuda altı hane aynı) ve kopuş boyunca sayısal
  sıfıra tek yönlü oturuyor. Yük **alan küpü taşımaz** (60×–269× tasarruf) ve
  bu test edilir. **Filmden enerji okunmaz:** RHF'in ayrışma enerjisi 14.875 eV
  (deneysel 4.75) ile felakettir, yoğunluğu ise doğrudur. Ölçülen sayılar
  aşağıdaki Doğrulama tablosundadır.
- **Modül 3F — tight-binding çekirdeği: tamam.** Örgü/komşu kurulumu,
  `H(k)`, bant yapısı, elektron sayımı, bant yükü ve `kind: tb` senaryosu.
  Modülün asıl çıktısı bir sayı değil bir **hüküm**: yol haritasının
  "bakır/silisyum bant aralığı" hedefinin ilk yarısı ölçümle **geçersiz
  kılındı** — TB'de bant aralığının *değeri* bir tahmin değil, girilen fit
  parametresinin geri okunmasıdır (silisyumda 1.17 eV için gereken tek
  parametre 5.376670 eV; aynı kadran Ge'yi ve V=7.0'da metali de veriyor).
  Geriye kalan gerçek: **iletken yarısı sayımdan** gelir ve parametrelerden
  bağımsızdır (tek elektron → 300/300 metal), **yalıtkan yarısı fit'ten**
  gelir (çift sayım yalıtkanlığı garanti etmez: 1B %67, 2B %34, 3B %20 metal).
  Bant kenarları ızgaradan **okunmuyor** — ölçüldü, ızgarayı sıklaştırmak
  hatayı monoton küçültmüyor. Parametreleri qlab'ın kendi Hartree-Fock'undan
  türetmek Plan G'ye bırakıldı. Ölçülen sayılar aşağıdaki Doğrulama
  tablosundadır.
- **Modül 3H — UHF indirgemesi (downfolding): tamam.** Modül 3F'nin açıkça
  yazılmış en büyük açığı kapandı: **parametreler artık dışarıdan girmiyor**,
  qlab'ın kendi UHF'inden türetiliyor. `engine/tb/downfold.py` spin-kırık UHF
  çözümünün α/β Fock matrislerini Löwdin ile ortogonalleştirip site bazında
  okuyor ve dört sayı üretiyor (`ε̄`, `Δ`, `t₁`, `t₂`); `model_from_downfold`
  onları TB motoruna besliyor, `fit_distance_laws` mesafe yasalarını
  uyduruyor, `kind: downfold` senaryosu uçtan uca koşuyor. **RHF üzerinde
  yapılamazdı ve gerekçesi Hückel'dir:** hidrojen halkalarında RHF çoğu
  geometride HF'in taban durumu değil (4n halkalarında HOMO/LUMO tam
  dejenere; 7/7 ölçüldü, N=8 ve N=12 test edilen hiçbir geometride kararlı
  değil). Modülün asıl kazancı bir sayı değil bir **çıpa**: kademeli alanın
  çatallanma üssü üç halkada bağımsız olarak **0.496190 / 0.497600 /
  0.497811** çıkıyor (beklenti ½) ve Modül 1C'nin H₂'de bambaşka bir
  yordamla ölçtüğü 0.4970 ile aynı yere düşüyor — `d_c` triplet Hessian'ın
  kökünden, `Δ` indirgemeden gelir, iki hesap birbirini hiç kullanmaz.
  Sahte-çıpa sınavı **geçti ama DAR**: 40 fizik-dışı bazın 0'ı gerçeği
  geçemedi, buna karşılık kuyruk marjı RHF tarafının üçte biri (1.49× vs
  4.29×) ve sınav hedef kümesine koşullu. **Hızlanma İDDİA EDİLMİYOR** —
  HF/TB maliyet oranı ölçülmedi. Ölçülen sayılar ve dört dürüst olumsuz
  aşağıdaki Doğrulama bölümündedir.
- **Modül 4 — spin Monte Carlo: tamam.** `engine/spin/` (kare kafes, iki
  model, checkerboard Metropolis, Wolff kümesi), `observable/spin_obs.py`
  (Binder, χ, C, ξ ve **hata çubuğunun kendisi** — plato denetimli binning),
  `viz/spin_frames.py` ve `kind: spin` senaryosu. Modülün asıl çıktısı yine
  bir sayı değil bir **hüküm**: yol haritasının "demir Curie sıcaklığı
  1043 K" hedefi ölçümle **dairesel çıktı** — Ising tek bir enerji ölçeği
  taşır ve `T_c ∝ J`, yani 1043 K için gereken `J = 39.608394 meV` geri
  konulduğunda `T_c` **birebir 1043.0000 K** okunur. Hedef düzeltildi ve
  yerine **dört dairesel olmayan çıpa** kondu: Onsager'in `kT_c/J`'si,
  kritik üs `β/ν = 0.122941 ± 0.003225` (ortalama alanı **272 kat** marjla
  dışlar), **Mermin-Wagner** (aynı kafes, aynı örnekleyici, zıt hüküm — ve
  sebebi bir teorem), ve Modül 3H'nin mean-field üssüyle (0.4962) karşılaştırma.
  Bu modülde depoya ilk kez bir **hata çubuğu** girdi ve ilk kuralı naif
  SEM'in hiçbir yerde raporlanmamasıdır (ölçüldü: 8.81 kat küçük gösteriyor).
  **Modül hiçbir malzemenin T_c'sini iddia etmez.** Ölçülen sayılar ve
  düzeltilen dört plan iddiası aşağıdaki Doğrulama bölümündedir.
- **Modül 5 — elektromanyetik alan (Maxwell FDTD): tamam.** `engine/em/`
  (Yee ızgarası 1B **ve** 2B, `ε`/`μ` haritaları, PML/PEC/dairesel PEC/kayıplı
  katman, Ricker ve Gauss×cos kaynakları, eğik gelişin 1B indirgemesi),
  `observable/em_obs.py` (**yakınsama merdiveni**, Fresnel katsayıları, TİY faz
  farkı, kip spektrumu, iki ayrı uzak alan, saçılma genişliği),
  `viz/em_frames.py` ve `kind: em` senaryosu. Yol haritasının hedefi burada
  **iki ayrı sebeple** düştü: *"bilinen yansıma/kırılma katsayıları"* bir
  **tautolojidir** (`n` girdidir, `R` onun cebirsel sonucudur ve `n`'e bire bir
  geri çevrilir), *"Zeeman yarılması"* ise bir **yöntem/hedef uyuşmazlığıdır**
  (Maxwell FDTD klasiktir; ürettiği şey yörüngesel `g = 1`, spin Zeeman'ı
  `g = 2` ister). Yerine **dört fizik çıpası** kondu — Bessel oranları
  (`j₁,₁/j₀,₁` **1.593293002**, merdiven OTURDU(2)), Rayleigh üssü (2B'de
  **p ∈ [2.924895, 3.243596]**, 3B'nin 4'ünü aralıkla dışlar), tek yarık
  (`a·sin θ₁/λ` = **0.998610 ± 0.019357**) ve TİY faz farkı (`δ_ps` merdiveni
  OTURDU(2)) — ve dördü de `n`'den, `R`'den, `c`'den ve birim seçiminden
  bağımsızdır. Bu modülde depoya ilk kez bir **yakınsama ölçütü** girdi ve ilk
  kuralı üç değerli olmasıdır: `OTURDU` / `OTURMADI` / **`YARGILANAMAZ`**, ve
  sonuncusu panelde sessizce "oturdu" diye basılamaz. **Modül hiçbir malzemenin
  optik sabitini iddia etmez.** Ölçülen sayılar ve düzeltilen otuz altı plan
  iddiası aşağıdaki Doğrulama bölümündedir.
- **Modül 6 — yerçekimi ve uzay-zaman: tamam.** `engine/grav/` (Schwarzschild
  **ve** izotropik PPN metrikleri, `w(u)` formu, kareleme **çıpa yolu**, ODE
  **yörünge motoru**, ve metriği HİÇ görmeyen ayrı bir Newtonyen N-cisim
  çözücüsü), `observable/grav_obs.py` (perihel katsayısı, ikinci mertebe,
  bükülme oranı, **`newton_noise_floor`**, beş kapalı-form türev, sekiz
  yörüngesinin sabiti ve monodromi açıları), `viz/grav_frames.py` ve
  `kind: grav` senaryosu. **Dört fizik çıpası** — `C = 2π(2+2γ−β)`
  (**18.84955591968662**, `6π`'den 1.85e-09), `K = πλ²(3+e²/6)`, bükülme oranı
  `1+γ` (**1.9999999996887523**) ve sekizin `|E|^{3/2}T/(G²m^{5/2})`
  (**9.237681250723917**) — ve dördü de girilen `M`'den, `a`'dan, `e`'den ve
  birim seçiminden **bağımsızdır**. Bu modülde depoya ilk kez bir **hata
  çubuğu makinesi** girdi: her fizik sayısının yanında **kendi borusundan**
  okunan `λ = 0` (ya da `m2 = 0`) tabanı ve S/G oranı — **karışık boru
  yasak**. `λ`, `γ`, `β` ve N-cisimdeki `power` panelde yeni bir kutuya,
  **KURAM SEÇİCİ**ye yazılır: bunlar bir yaklaşım da bir birim seçimi de
  değildir, **hangi kuramın çözüldüğünü** seçerler. **Modül Merkür'ün perihel
  kaymasını ÖNGÖRMEZ** — 43″ `M`, `a` ve `e`'den bire bir geri okunur ve
  değeri hangi efemeris tablosunun seçildiğine bağlıdır. Ölçülen sayılar ve
  düzeltilen **otuz dört** plan iddiası aşağıdaki Doğrulama bölümündedir.
- **Modül 7 — modüller arası yokluk haritası ve kenar ölçümü: tamam.**
  `bridge_map.py` (saf tanım katmanı: 21 çiftin kova tablosu **koddan
  türetilir** — `olculen_kenarlar()` ve `olculen_units_ekseni()` `engine/`
  ağacını `ast` ile tarar, tablo her koşumda ona karşı sınanır ve çelişirse
  panel tik basmaz), `observable/bridge_obs.py` (tabanı ölçülen tek kenar
  `hf → md`'nin tabanı — **iki sayı**), `viz/bridge_frames.py` ve `kind: bridge`
  senaryosu. **Sıfır yeni fizik.** Kova tablosu bugün **3 / 8 / 8 / 0 / 2**
  (gerçek köprü / tautoloji / yapısal yokluk / benzerlik / **ölçülmedi** — ayrı
  kova, §7.2 iki çiftte açık). 27.09.2026'da `hf → spin` **kenar yazılmadan, hüküm
  verilerek** `OLCULMEDI`den `TAUTOLOJI`ye geçti: H2 Yamaguchi `J`sinin
  büyüklüğü oran protokolünde bit bit sadeleşir, işareti `h = 0` çift L'de bir
  **ayar etiketidir** (AFM kademeli seri FM tekbiçim seriyle bit bit aynı) ve
  şartlıdır (izlemesiz `search_uhf(seeds=1, follow_instability=False)` kırık
  simetriyi bulamaz, sessizce +J verir; 09.10.2026'dan beri varsayılan
  kararlılık izlemesi onu buluyor, J < 0);
  yeni kova açılmadı. 28.09.2026'da `tdse → spin`, `md → em` ve `tb → em` de
  kenarsız, hükümle `TAUTOLOJI`ye geçti (TDSE'nin 1B soft-Coulomb H2 `J`si gerçek
  ama spin hedefi büyüklüğü siler, 300 K'de R = 3/4/5 donar ya da doyar; EM çiftlerinde
  tekbiçim ortam bir korunumun sıfırı ya da bir eps koşumunun etiketidir,
  konuma bağlı ortam yolu TÜR B ve hükmü yok); `tdse → tb` ile `spin → em`
  ölçülmedi kovasında kaldı ama nedenleri artık ölçülmüş: bir tanım seçimi (kovayı
  belirleyecek sevk-yolu sınavı koşturulmadı) ve bir eksen/Λ beyanı. Modül 7 kapandığında **1 / 4 / 8 / 0 / 8**'di; iki çift
  sonradan `OLCULMEDI`den `GERCEK_KOPRU`ya geçti ve ikisinde de **kova
  sözlüğü bükülmedi, kod kenarı kuruldu**: 22.09.2026'da `md → tdse`
  (`engine/tdse/from_md.py`; kenar sayısı **0.13167174 bohr**, pencere
  L = 4.0 / n = 512 / A = 0.1 / N_BOLME = 200, periyot 18 ve 24'te bit bit
  aynı) ve 24.09.2026'da `tb → md` (`engine/md/tb_kuvvet.py`, ss-σ sıkı-bağ kuvvet
  alanı — **kenar tabanı ölçülmedi, doğa çıpası yok**; aşağıda Modül 2 ve
  3F bölümleri). Çapraz-motor import bugün **dört kenar** — `md → hf`,
  `md → tb`, `tb → hf`, `tdse → md` — ve yapı bir merdiven değil:
  `hf - md - tb` **üçgeni** artı `md - tdse` **kuyruğu** (Modül 7
  kapandığında iki çubuklu bir yıldızdı, 22.09'da dört düğümlü bir zincir
  oldu). Üçgen import grafiğinde bir döngüdür, **veri akışında değil**:
  `tb → hf` kenarında veri akmaz (`model_from_downfold` `src/`de hiç
  çağrılmıyor), yani hf → tb → md diye bir veri yolu yoktur. `qlab.units`e
  dokunan küme hâlâ `{hf, md, tb}`, ama artık kenar düğümlerinin **alt
  kümesidir**: `tdse` bağlı ama ölçeksiz — Modül 7'nin ölçtüğü "units kümesi =
  bağlı küme" özdeşliği bir yasa değil eski topolojinin rastlantısıymış.
  Kenar sayısı, dereceler ve sıfır bağlı çift sayısı beyanda yazılmaz, panelin
  canlı satırı onları her koşumda taramadan basar.
  **Kenar tarayıcısı — yalın sürüm (26.09.2026).** Tarayıcı yalnız **düz**
  import biçimlerini çözer: `import qlab.engine.X…`, `from qlab.engine.X…
  import …`, `from qlab.engine import X`, göreli importlar (alt pakette de),
  `qlab.engine.X` zinciri ve alt paketler (dosya üst düzey motora atfedilir).
  **Adı sayılan** yükleme ve erişim biçimleri motor ağacında (`engine/`
  kökündeki dosyalar dahil) **yasaktır** (`importlib`/`runpy`/`pkgutil`/
  `pydoc`/`imp`/`zipimport`/`builtins`/`sys` importu; `__import__`, `exec`,
  `eval`, `compile`, `vars`, `globals`, `locals`, `__builtins__`;
  `TYPE_CHECKING`; çıplak `import qlab`, `from qlab import engine`,
  `from qlab.engine import *`; bir zincirin başı olmayan `qlab` adı ve
  zincirsiz `qlab.engine` — takma ad, `getattr(qlab…)`; tam liste
  `taranamayanlar()` dokstringinde): `taranamayanlar()` her geçişi
  `dosya:satır biçim` diye adıyla raporlar ve panel tik basmaz. Bu bir analiz
  değil bir **yasaklama politikasıdır** ve liste **sonludur**: izinli bir kenar
  üzerinden yeniden aktarım (aklama: tdse, hf'yi md'nin bir özniteliğiyle
  kullanır, `tdse → hf` kenarı çıkmaz), `os.sys.modules`, `mock.patch` dizgesi,
  `.pth` ve C uzantısı görünmez. Gerçek ağaçta yasakların sayısı sıfır, tablo
  değişmedi (648f03b'nin tarayıcısıyla aynı kenar ve eksen, testte kilitli).
  Canlı satır ayrıca `taranamayan N` (sıfırken de basılır: ölçülmüş bir sıfır)
  ve `kenar sınıfları` parçalarını basar. Her kenar ayrıca **çalışma zamanı**
  ya da **yalnız tip** diye
  sınıflanır (`olculen_kenar_siniflari()`): `md → hf`, `md → tb`, `tb → hf`
  çalışma zamanı; `tdse → md` **yalnız tip** — `ForceField` from_md.py'de yalnız
  bir argüman annotasyonu, veri akışı ördek tiplemeyle, import satırı silinince
  modül **bit bit aynı** çalışıyor (ölçüldü, testte kilitli). GERÇEK_KÖPRÜ
  kenarlarının sınıfı `GERCEK_KOPRU_KENAR_SINIFLARI`nda beyanlıdır ve kilit 5
  onu ölçüme karşı sınar. Hüküm koşulludur: annotasyonu **değerlendiren** bir
  tüketici (`get_type_hints`, `get_annotations`, `eval_str`, `singledispatch`
  ve benzerleri; tam liste `tests/bridge/test_kenar_tarayici.py` `TUKETICILER`)
  `src/qlab` çapında yasak ve sıfır (testle kilitli). Kilit 2 yalnız tip
  kenarını da kod kenarı sayar; kilit 2'nin çalışma zamanı kenarı isteyip
  istemeyeceği Osman'ın kararı: **hayır** (08.10.2026; Osman'ın otonom devam yetkisiyle varsayılan uygulandı). Evet denilseydi `tdse → md`
  bugünkü kaynakla GERÇEK_KÖPRÜ'de **kalamazdı**: kova sayımı değişir, 22.09'un
  "delik kenar kurularak kapatıldı" hükmü yeniden açılırdı. Sınıfın anlamı **tek
  yönlüdür**: `yalnız tip` yükün yokluğunu iddia eder ve yük taşıma çıpasıyla
  kilitlidir; `çalışma zamanı` muhafazakâr bir statik sınıftır, yükü garanti
  etmez — annotasyonu dizge yazmak, future importunu silmek ya da adı modül
  düzeyinde bir yuvaya koymak (`_ = ForceField`) sınıfı veri akışını
  değiştirmeden çevirir (ölçüldü, sentetik ağaç); sonuncusu import silme
  ölçüsünü de çevirir. "Evet" kalıcı bir şart olacaksa bu kaçışları kapatan
  bir ölçüte bağlanmalıdır. `md → tb` gerekçesindeki
  ölçüt kayması da düzeltildi: GERÇEK_KÖPRÜ ölçütü "karşılık var + veri
  **akabilir** + çıktı kıpırdar + kod kenarı **var**"dır; "veri akar" ve "yük
  taşır" o kenarda ölçülen daha güçlü olgulardır, ölçütün parçası değil.
  `hf → md`'nin kenar tabanı **A1** fizik-kapalı limit **3.3986707e-05 cm⁻¹** (*aritmetiği*
  ölçer, S/G 1.61e+08) ve **A2** anharmoniklik-eşleştirilmiş prob
  **1.2025992e-02 cm⁻¹** (*hata çubuğunu* ölçer, S/G 4.56e+05); gerçek hata
  1.4472809e-02 → **A1 425.84 kat iyimser, A2/gerçek 0.8309** — Modül 6'nın
  "taban yalnız aritmetiği ölçer" kusur beyanı burada ikinci probla **kapandı**,
  ve iki sayı iki ayrı düğmeye tepki veriyor (N = 200 → 400: A1 15.18 kat,
  A2 1.0008 kat). `hf → md` bir **şema çıpasıdır** (iki tüketici de aynı HF
  çözümünün soyundan); mevcut `ω_B/ω_A` çıpası kütleye **bit bit kör** ve kütle
  borusu kendi mutlak sınavını aldı; düşen iki aday (`hf → tb`, `tb → spin`)
  **tautoloji olarak, penceresiyle** sevk edildi (28.09.2026'dan beri altı: +
  `hf → spin`, `tdse → spin`, `md → em`, `tb → em`, kenarsız hükümle). Planın/spec'in **yirmi beş**
  sayısı ölçümle düzeltildi (Task 5'te bir: 6-31G'nin `ω_A`sı **penceresizdi**).
  Modül 7'nin Doğrulama bölümünde ayrı bir başlığı YOKTUR (bu satır eskiden
  öyle bir bölüme gönderme yapıyordu): ölçülen sayılar ve dürüst olumsuzlar
  panelde ve `tests/bridge/`in dosya dokstringlerindedir; ss-σ kenarının
  sayıları Doğrulama'nın Modül 2 ve Modül 3F bölümlerindedir.

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
yüksek enerjili bir çözüm seçer. Bu çözüm gerçek bir durağan noktadır
(`converged=True` yalan değildir) ama bir **eyer noktasıdır**. qlab bunu artık
iç (RHF→RHF) kararlılık analiziyle yakalar: yakınsamadan sonra tekil orbital
Hessian'ının özdeğerleri hesaplanır ve sonuç şeffaflık panelinde
"Kararlılık" satırı olarak basılır (H₂ 0.74 Å: +4.52, 10 Å: +2.89, **12 Å:
−2.92 → eyer noktası**). Analiz sonucu DÜZELTMEZ, yalnızca sessizliği
kaldırır — kırılma hâlâ oradadır ve kararsız bir nokta hata yükseltmez, çünkü
kullanıcı o çözümü meşru olarak isteyebilir.

**Modül 1C'de bu cümle DEĞİŞTİ.** Eskiden burada *"Sınanmayan: dış
kararsızlıklar (RHF→UHF, triplet, kompleks orbital)"* yazıyordu. **Dış
(RHF→UHF, üçlü) kanal artık sınanıyor:** `solve(stability=True)` iki Hessian'ı
da kurar, panel iki ayrı satır basar ve tarama nokta bazında ikisini de
kaydeder. Bunun neden gerektiği ölçülmüştür — deponun kendi PySCF referans
kaydı `h2_200` iç kanalda **+2.5467 ile "kararlı"** derken dış kanalda
**−1.5995** ile eyerdir ve gerçek HF taban durumu **0.153420178832 hartree
aşağıdadır** (Modül 1C bölümü). **Hâlâ sınanmayan: kompleks (reel olmayan)
orbital kararsızlığı** — ve iki kanalda da "kararlı" hükmü çözümün küresel
minimum olduğunu söylemez.

RHF'nin yukarıda anlatılan ayrışma kusuru da artık yalnız anlatılmıyor,
**ölçülüyor**: UHF aynı limitte H₂'yi iki nötr hidrojene ayırıyor (20 bohr'da
2·E_H'den sapma **4.441e-16**) ve RHF'in aynı noktada **9.85877 eV** yukarıda
kaldığı ölçülüyor. UHF de bir yaklaşımdır ve kendi kusurunu taşır: spin
özdurumu değildir (`h2_200` çözümünde ⟨S²⟩ = 0.945862, singlet için 0
olmalıydı) ve bu depoda spin projeksiyonu (PUHF) yoktur.

Görselleştirme ızgarası çekirdek orbitallerini ÇÖZMEZ, ve bu ölçülmüştür:
su/STO-3G'de HOMO'nun ızgara üzerindeki normu 1.00000000 çıkarken çekirdek
1s orbitalininki **0.98760196**'dır — analitik değeri 1 olan bir büyüklükte
%1.24 eksik. Aynı orbitalin ızgaradaki maksimum genliği de yakınsamamıştır
(n=60'ta 4.9041, n=80'de 5.9271). Çekirdeğe yakın gösterilen her yüzey bu
yüzden niteldir; oradan sayı okunmaz.

Yoğunluk farkı haritasının promolekül referansı **gerçek taban durum
değildir**. RHF serbest atomları genel olarak çözemez (tek elektronlu H açık
kabuktur, oksijenin taban durumu üçlüdür), bu yüzden referans küresel
ortalamalı, kesirli işgalli bir serbest atomdur (SAD tipi: simetri blokları
ayrı köşegenleştirilir, kısmi dolu kabuk üç kartezyen yöne eşit bölünür —
oksijende 2p başına 4/3 elektron). Bu bir seçimdir ve `Provenance`'ta
yazılıdır; bağımsız bir dış referansı YOKTUR, çünkü karşılaştırılacak "doğru"
serbest atomu aynı yöntem üretemez.

Orbital penceresindeki **±faz renkleri mutlak bir anlam taşımaz**. Bir
özvektörün global işareti keyfidir; başka bir kütüphane aynı orbitali ters
işaretle verse kırmızı ile mavi yer değiştirirdi ve hiçbir fizik değişmezdi.
İki renk yalnızca aynı orbitalin kendi içindeki bağıl fazı gösterir. Aynı
şekilde izoyüzey seviyesi hesabın bir sonucu değil bakanın bir seçimidir:
aynı su HOMO'su 0.02'de 2850, 0.50'de 114 tepe verir. Pencere üçünü de
şeffaflık panelinde beyan eder.

**Modül 4 hiçbir malzemenin Curie sıcaklığını iddia etmez, ve bu bir alçak
gönüllülük değil bir cebir sonucudur.** 2B Ising modeli tek bir enerji ölçeği
taşır (`J`) ve `T_c ∝ J`; Onsager'in tam çözümü oranı sabitler
(`kT_c/J = 2/ln(1+√2) = 2.269185314`). Dolayısıyla SI birimindeki her `T_c`,
girilen `J`'nin geri okunmasıdır — ölçüldü: demirin 1043 K'i için gereken
`J = 39.608394 meV` konulduğunda `T_c` **birebir 1043.0000 K** çıkar (fark
0.000e+00). Aynı kadran her malzemeyi verir (Ni 627 K → 23.810607 meV,
Co 1388 K → 52.709924 meV, Gd 292 K → 11.088831 meV). Bu modülün ölçtüğü
şeyler `kT_c/J` oranı ve kritik üslerdir; ikisi de `J`'den, kafes sabitinden
ve birim seçiminden bağımsızdır. Beyan şeffaflık panelinde kendi satırında
(`Dairesellik`) basılır.

Aynı modeldeki başka iddialar da kurulmuyor: **spin kuantum değil klasiktir**
(sıfır noktası dalgalanması ve komütasyon bağıntıları yoktur), **kafes
gerçek bir kristal değildir** (koordinasyon sayısı 4'te sabit, örgü
geometrisi girdide yok), **3B kapsam dışıdır** (3B Ising'in tam çözümü yoktur
ve çıpa Onsager yerine literatürün sayısal değerlerine dayanırdı), ve
denge-dışı süreçler (histerezis, tavlama) yazılmadı. Korelasyon uzunluğu
sevk edilen senaryoda **raporlanmaz** — gerekçesi ölçüldü ve panelde yazılı.

**Modül 7 hiçbir modüller arası fiziksel bağ kurmaz ve bir "ölçek merdiveni"
olduğunu iddia etmez — ölçüldü, yok.** Yedi motorun 21 çiftinde bugün üç
gerçek köprü var ve **hiçbiri bir doğa çıpası değildir**: `hf → md` bir
**şema çıpasıdır** (iki tüketici — statik ikinci türev ve MD periyodu — aynı
HF çözümünün soyundan gelir, uyum doğayı değil iki uygulamayı ölçer),
`md → tdse` tek bir yüzeyin iki ayrı motorda koşulmasının **ayrışmasını**
ölçer, `tb → md` ise **seçilmiş** bir radyal yasayı taşır (çıpaları modelin
kendi kuramına karşıdır; kenar tabanı ölçülmedi). `hf → tb`, `tb → spin`,
(27.09.2026'dan beri) `hf → spin` ve (28.09.2026'dan beri) `tdse → spin`,
`md → em`, `tb → em` birer tautolojidir (ilki `np.mean` tanımının
`==` ile geri okunması, spin'e gidenler `J`'nin yalnız `J/T` oranı olarak girmesi,
EM'e gidenler tekbiçim ortamda bir korunumun sıfırı ya da bir eps koşumunun etiketi —
`hf → spin`de işaret de `h = 0`'da bir ayar etiketi; `tb → spin`in `+J`si
bir ayar seçimidir — kaynak antiferromanyetiktir ve bu işaret dönmesi
27.09.2026'ya kadar hiçbir yerde yazılmamıştı; beyan o gün eklendi, işaret
çevrilmedi, çünkü çevrilse kilitli 0.687930 bozulurdu); `grav`
atom ölçeğine **bağlanmaz** (`F_g/F_e = 8.09e-37`); iki çift ise **hükümsüz**
kalır — nedeni ölçüldü, hüküm bir tanım seçimine (sevk-yolu sınavı koşturulmadı)
ve bir eksen/Λ beyanına bağlı — ve panel bunu "ölçüldü, sıfır çıktı" ile aynı hücreye yazmaz.
`hf → md`nin tabanı **iki sayıdır** ve yalnız birincisi (fizik-kapalı,
3.40e-05 cm⁻¹) yazılsaydı panel gerçek hatayı **425.84 kat iyimser**
gösterirdi — ikincisi (analitik ikiz, 1.20e-02 cm⁻¹) gerçek hatanın 0.83'ü.
Kova tablosu elle listelenmez, koddan türetilir; bayat bir yokluk beyanı bir
yalandır ve yalanın yanında tik basılmaz.

Qt penceresi bu depoda **bir kez** gerçek ekranda açılıp gözle doğrulandı:
orbital penceresi (su/STO-3G HOMO), Windows platformunda, kabul kriteri 3
kapsamında — ayrıntı "Kabul kriterleri durumu" tablosunda. Onun DIŞINDA
projedeki her GUI testi `QT_QPA_PLATFORM=offscreen` ile ekransız çalışır
(bkz. `tests/test_transparency.py`, `tests/test_orbital_window.py`) — widget
durumu programatik olarak okunur, çizimin kendisi doğrulanmaz. TDSE penceresi
hâlâ hiçbir gerçek ekranda görülmedi.

## Kurulum

```bash
python -m pip install -e ".[dev]"
```

Orbital penceresi `pyqtgraph.opengl` üzerinden **PyOpenGL**'e bağlıdır; ikisi de
bağımlılık listesindedir. Eksikse `tests/test_orbital_window.py` sessizce
atlamaz, KIRMIZI olur — kurulu olmayan bir bağımlılık bozuk bir ortamdır,
atlanacak bir koşul değil.

## Kullanım

GUI olmadan senaryo koştur:

```bash
python -m qlab.run scenarios/tunneling.yaml -o out.npz
```

Klasik moleküler dinamik (Modül 2) — H₂'nin Morse potansiyelinde titreşimi
ve 108 argon atomunun Langevin termostatıyla dengelenmesi:

```bash
python -m qlab.run scenarios/h2_morse.yaml -o h2md.npz
```

```bash
python -m qlab.run scenarios/argon_nvt.yaml -o argon.npz
```

Çıktı `.npz`'si `times_fs`, `positions`, `velocities`, `kinetic`, `potential`,
`total` ve `temperature_kelvin` taşır; şeffaflık paneli koşunun yöntemini,
yaklaşımlarını ve enerji sürüklenmesini stdout'a basar.

Ab initio MD (Modül 2D) — H₂ Hartree-Fock yüzeyinde titreşiyor, kuvvetler
her adımda SCF ile yeniden çözülüyor (adım başına 7 SCF, ~8.5 ms):

```bash
python -m qlab.run scenarios/h2_aimd.yaml -o h2aimd.npz
```

Sıkı-bağ MD (md ↔ tb ss-σ kenarı, 24.09.2026) — 16 atomlu yarı dolu bir
zincirde (kütle ¹H; atomlar bir element DEĞİL, yasası seçilmiş bir modelin
siteleridir) kısa NVE; kuvvet, tek orbitalli ss-σ modelinin Mermin serbest
enerjisinin gradyanıdır (ölçüldü, 24.09.2026, bu makine: CLI 0.72 s,
`energy_drift` 4.513e-04, eşik 1e-02; salınım S 1.005e-02, eşik 5e-02 —
D2, 09.10.2026; kare-0 ω0·dt 0.2555 ve yol kareleri 0.3174, eşik 0.67 — D2-i ve
D2-vi, 10.10.2026; pencere
senaryonun kendisi — a = 2
bohr zincir + 0.05 bohr bozulma, 631.6 K, L = 32 bohr, dt = 10, 400 adım):

```bash
python -m qlab.run scenarios/tb_zincir_md.yaml -o tbmd.npz
```

Elektron sıcaklığı senaryoda **Kelvin** (`electron_temperature_kelvin`),
kuvvet sınıfında hartree'dir; dönüşüm fabrikada yapılır ve panel ikisini de
basar. Senaryonun iki radyal yasası (`ss_sigma`, `itme`) **seçimdir**: bu
kenarda doğa çıpası yoktur.

UHF indirgemesi (Modül 3H) — H₈ halkasının tight-binding parametreleri beş
mesafede qlab'ın kendi UHF'inden türetiliyor (ölçüldü, 5.5 s):

```bash
python -m qlab.run scenarios/halka_indirgeme.yaml -o indirgeme.npz
```

Çıktı `.npz`'si `distances_bohr`, `epsilon_bar`, `delta`, `hoppings`,
`s_squared`, `energy_uhf`, `staggering_residual` ve `provenance_json` taşır.
Panelin `Yakınsama` satırı burada "SCF durdu" demez — modelin varsaydığı
iki-siteli antiferromanyetik yapının her mesafede gerçekten bulunduğunu söyler.

Spin Monte Carlo (Modül 4) — 2B Ising kafesinde altı sıcaklıklık bir tarama
(ölçüldü, 0.56 s):

```bash
python -m qlab.run scenarios/spin_tarama.yaml -o spin.npz
```

Çıktı `.npz`'si `temperatures`, `magnetisation`, `magnetisation_error`,
`plateau_reached`, `tau_int`, `binder`, `susceptibility`, `heat_capacity`,
`coupling` ve `provenance_json` taşır (`susceptibility` enerji⁻¹ birimindedir:
coupling'ler arası karşılaştırılacak sayı `coupling · susceptibility`'dir). **Sıcaklıklar enerji birimindedir, kelvin değil** —
`k_B = 1` bir birim seçimidir; `coupling` (J) bir girdidir ve Onsager'le
karşılaştırılacak sayı `kT/J = T/coupling`'dir (`Yöntem` satırı onu basar;
27.09.2026'ya kadar "`J = k_B = 1`, sıcaklık `kT/J`" yazıyordu ve coupling ≠ 1'de
yanlıştı). `coupling` sonlu ve pozitif olmalı (YAML'da da Python yolunda da);
Ising checkerboard Metropolis `T/J → ∞`'da donar (ölçülen pencerelerde
`T/J ≳ 1e3`) ve panel o zaman ✗ basar.
Panel iki ek satır basar: `Dairesellik`
(SI'daki `T_c` bir tahmin değildir) ve `Örnekleme` (kaç sweep, kaç ölçüm,
hangi algoritma, `τ_int`, ve binning platosunun oluşup oluşmadığı). Plato
oluşmamış sıcaklıklarda hata çubuğu **"ALT SINIR" diye işaretlenir** ve
`Yakınsama` satırı ✗ basar — bu "hesap bitmedi" demek değil, "raporladığım
belirsizliklerin hepsi belirsizliğin kendisi değil" demektir.

Maxwell FDTD (Modül 5) — 2B dairesel PEC boşlukta dört basamaklı bir yarıçap
merdiveni (ölçüldü, 1.49 s):

```bash
python -m qlab.run scenarios/em_tarama.yaml -o em.npz
```

Çıktı `.npz`'si `radii`, `spacings`, `ratios`, `noise`, `peaks`, `orders`,
`verdict`, `field`, `field_conductor` ve `provenance_json` taşır; `p_inf` ile
`uncertainty` **yalnızca okunabildiklerinde** yazılır — okunamamış bir mertebeyi
`0.0` ile göstermek "mertebe sıfır çıktı" diye okunurdu. **Yarıçaplar ve dalga
boyu hücre cinsindendir, metre değil** — `c = ε₀ = μ₀ = 1` bir birim seçimidir
ve senaryo bir **tohum taşımaz**: FDTD deterministtir, tekrarlanabilirliğin
karşılığı pencerenin tam beyanıdır. `npml` ve `Δx` bilinçli olarak **anahtar
değildir** — biri bağlı değil (dairesel PEC boşluk tam yansıtıcıdır), öteki
bağlıydı ama **sessizce yanlış** çalışıyordu (ölçüldü: tepe sayısı `Δx` = 1/0.5/2
için 20/13 → 4/4 → 31/43, `λ` üçünde de aynı kalarak); 26.08.2026'dan beri
motor `Δx ≠ 1`'i reddediyor (`engine/em/grid.dx_kapisi`). Panel ikisini de yine
basar, değerleri koşumdan gelir. Panel dört ek satır basar: `Dairesellik`
(bir malzemenin `R`'si bir tahmin değildir), `Çıpa sınıfı` (hangi sayı şema,
hangisi fizik, hangisi çeviri denetimi), `Pencere` (ızgara/`Δx`/`S`/adım/`λ`/
`npml`/`sgf`/FFT penceresi + hükmün nerede geçerli olduğu) ve `Yakınsama tanısı` (merdiven,
`p_inf ± u`, ve **üç değerli hüküm**). `YARGILANAMAZ` sessizce "oturdu" diye
basılmaz.

Modüller arası yokluk haritası (Modül 7) — 21 çiftin kova tablosu koddan
türetilir ve tabanı ölçülen tek kenarın (`hf → md`) tabanı iki sayı olarak
okunur (03.09.2026'da ölçüldü, 0.7 s; 26.09.2026'dan beri tarama dört kez
koşuyor — kenar, eksen, taranamayan, sınıf — ve bütün yük bu makinede ~1.0 s;
HF hiç çözülmez):

```bash
python -m qlab.run scenarios/kopru_haritasi.yaml -o kopru.npz
```

Çıktı `.npz`'si `pair_a`/`pair_b`/`kova`/`magnitude` (21 çift), `edges`,
`units_axis_*`, `floor_*` (A1 ve A2 ayrı ayrı), `real_error`, `inconsistencies`,
`converged` ve `provenance_json` taşır. **`magnitude` `nan` taşıdığında yanında
`magnitude_is_none` maskesi vardır** — ölçülmemiş bir büyüklük (`None`) ile
bit bit sıfır ölçülmüş bir büyüklük (`tb → spin`, `0.0`) aynı değerle
gösterilemez. Senaryo bir tablo taşımaz: tablo koddan gelir, YAML yalnız kenar
tabanının **penceresini** (`n_bolme`, `periyot_sayisi`, `genlikler`, `a_ikiz`)
taşır ve hepsi zorunludur; `a` (A1'in fizik-kapatma merdiveni) bir anahtar
**değildir**. Panel on satır basar; `Yakınsama` burada "hesap bitti" değil,
"tablo koda karşı tarandı ve tutarlı **ve** kenarın iki tabanı da okunabildi"
demektir. `Kuram seçici` satırı baz setidir: spec'in 6-31G sayısı yalnız
6-31G'nin kendi `r_e`sinde çıkar, sevk edilen `r_e`de 275 cm⁻¹ başkadır.

Etkileşimli TDSE penceresi:

```bash
python -m qlab.app.main scenarios/tunneling.yaml
```

Döndürülebilir orbital penceresi (RHF koşar, HOMO ile açılır):

```bash
python -m qlab.app.main --molecule "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587" --basis sto-3g
```

İki giriş yolu birbirini dışlar; ikisi birden ya da hiçbiri verilmezse komut
hangisini kastettiğinizi tahmin etmeden hata verir.

Testler:

```bash
pytest tests/ -v
pytest tests/ -v -m slow
```

`pyproject.toml` bir `addopts` taşımaz, yani `pytest tests/` yavaş işaretli
testleri de KOŞAR — ikinci komut ayrı bir küme değil, birincinin alt kümesidir.

**Bu makinede ölçüldü** (ilk iki satır 02.10.2026, sonraki dördü Modül 5, kalanı Modül 4):

| Koşu | Test | Süre |
|---|---|---|
| tam süite (`87337f0`, 02.10.2026) | **2248** | **1050.45 s** |
| `-m "not slow"` (`87337f0`) | **2108** | **191.01 s** |
| *tam süite (Modül 5'in iki kapısından SONRA)* | *1397* | *770.02 s* |
| *yalnız Modül 5 (`tests/em/`, aynı hâl)* | *237* | *197.39 s* |
| *tam süite (Modül 5 kapanışı, `cda2871`)* | *1389* | *939.68 s* |
| *yalnız Modül 5 (`tests/em/`, `cda2871`)* | *229* | *237.32 s* |
| *tam süite (Modül 4 kapanışı)* | *1159* | *585.36 s* |
| *`-m "not slow"` (Modül 4 kapanışı)* | *1088* | *107.30 s* |
| *yalnız `-m slow` (Modül 4 kapanışı)* | *71* | *475.33 s* |
| *yalnız Modül 4 (`tests/spin/`)* | *186* | *187.23 s* |
| *aynısı, `-m "not slow"`* | *166* | *29.82 s* |

**Eğik yazılan dokuz satır ESKİ ölçümlerdir ve YENİDEN ÖLÇÜLMEDİ** (ikisi
Modül 5'in iki kapısından sonra, ikisi Modül 5 kapanışı `cda2871`, beşi Modül 4
kapanışı); bugünkü toplama uydurmak
ölçüm değil uydurma olurdu. `cda2871` satırı yine de 26.08.2026'da BİR KEZ yeniden
koşuldu ve **1389 passed / 738.92 s** verdi — 939.68 s ile arasındaki fark
aşağıda beyan edilen makine gürültüsündendir, süite değişmedi.

`Δx` kapısı (`engine/em/grid.dx_kapisi`) süiteye `tests/em/test_hucre_birimi.py`
üzerinden **8 test** ekledi (1389 → 1397, `tests/em` 229 → 237); dosyanın kendi
süresi **1.51 s** ve içinde hiç `slow` işareti YOK. Eğik Courant kapısı
(`engine/em/oblique.oblique_courant_limit`) ise **hiç test EKLEMEDİ** — mevcut
kararlılık kilidini yeniden yazdı, çünkü kapı konunca eski ölçüm kendi kapısını
ölçer hâle gelirdi.

29.09–02.10.2026 arasında Modül 5'in kararlılık kapılarına üçü daha eklendi:
`eps_r < 1` ortamında **etkin Courant** `S/√eps_r` yükleyici ve fabrika kapısı
(`5d9a0bf`); aynı kapı **motor düzeyinde**, hücre başına
`S/√min(1, eps_r·mu_r)` ile üç katsayı yolunda (`yee._etkin_courant_kapisi`,
`c4c66ad`, 2242 → 2247 test — motor `eps_r·mu_r < 1` ortamında patlayan bir S
bandını kabul ediyordu ve `mu_r < 1` aynı delikti, fabrika onu göremezdi); ve
eğik p'nin **kesin** kararlılık eşiği `S = 2/√λmax`, ayrık operatörün en büyük
özdeğerinden (`oblique._egik_p_kesin_esik`, `87337f0`, 2247 → 2248 test —
`ey_kural` ≠ aritmetik ve konuma bağlı `mu_r`'de hücre çarpımı kapısı
yetmiyordu; model tek adım büyütme matrisi ve zaman adımlamasıyla iki bağımsız
yoldan doğrulandı ve kayıplı sınırda — PML, LossyLayer — da kesin). Ayrıca depodaki her `dosya.py:N`
atfı artık bugünün satırına kilitli (`tests/test_satir_atiflari.py`, `f1cf900`,
2240 → 2242 test — 49 atfın 6'sı hedef dosyaya satır eklendikçe sessizce
kaymıştı). Ölçümler ilgili docstringlerde.

MAKİNE GÜRÜLTÜSÜ: **aynı** süit aynı gün 727.75 s ve 770.02 s (`tests/em`:
191.01 s ve 197.39 s), `cda2871` ise 939.68 s ve 738.92 s koştu. Yani
tablodaki saniyeler ±%6-25 oynar ve **test sayıları dışında** karşılaştırma
için kullanılamaz.

Modül 5
senaryo/panel görevi süiteye **50 test** ve `tests/em` tarafında
**200.9 s → 237.32 s** (yani **+36.4 s**) ekledi; bunun **1.49 s**'i sevk
edilen senaryonun kendi koşumu, geri kalanı `slow` işaretli iki
karşılaştırma koşumu ve uçtan uca CLI alt sürecidir.

Modül 4 kapanışında sevk edilen hâl ayrıca bir kez daha doğrulanmıştı:
**1159 passed / 583.70 s**; tablodaki 585.36 s ile arasındaki %0.3, aşağıda
beyan edilen makine gürültüsünün (%16'ya kadar) çok içindedir.

Modül 4'ün senaryo/panel görevi süiteye **44 test** ve **4.54 s** ekledi;
dosyanın en yavaş çağrısı 0.95 s, yani deponun 2 s eşiğinin altında ve
dosyada **hiç `slow` işareti yok**. Modül 4'ün toplam maliyeti 1114 → 1159
(+45; biri `tests/test_scenario.py`nin keşfe dayalı parametrizasyonundan
kendiliğinden geldi).

Aşağıdaki blok **Modül 3H kapanışının** ölçümüdür ve Modül 4'ü içermez;
yeniden koşulmadı, sayılarını bugüne uydurmak ölçüm değil uydurma olurdu:

| Koşu (Modül 3H oturumu) | Test | Süre |
|---|---|---|
| tam süite | **971** | **432.22 s** |
| `-m "not slow"` | **920** | **85.90 s** |
| yalnız `-m slow` | **51** | **352.41 s** |
| yalnız Modül 3F + 3H (`tests/tb/`) | **185** | **138.55 s** |
| aynısı, `-m "not slow"` | **170** | **16.73 s** |

Sevk edilen son hâl ayrıca bir kez daha doğrulandı: **971 passed / 424.45 s**.
Tablodaki 432.22 s ile arasındaki %1.8, aşağıda beyan edilen makine
gürültüsünün içindedir; tablo tek bir oturumun beş koşusu olduğu için o
oturumun sayısıyla bırakıldı.

Aşağıdaki satırlar **864 testlik önceki oturumun** ölçümleridir ve Modül 3H'yi
içermez; yeniden koşulmadılar ve sayılarını bugüne uydurmak ölçüm değil
uydurma olurdu. (O oturumun kendi toplamları: tam süite 864 / 264.98 s,
`-m "not slow"` 828 / 67.97 s, `-m slow` 36 / 210.65 s.)

| Koşu (önceki oturum) | Test | Süre |
|---|---|---|
| yalnız Modül 1C'nin beş dosyası (`test_uhf`, `test_uhf_cipalar`, `test_rhf_sikismasi`, `test_coulson_fischer`, `test_dis_kararlilik`) | 60 | 31.46 s |
| yalnız `tests/hf/` (1B + 1B2 + 1C) | 420 | 59.65 s |
| aynısı, `-m "not slow"` | 399 | 34.68 s |
| yalnız Modül 3F (`tests/tb/`) | 80 | 4.19 s |
| yalnız B2'nin üç dosyası (`test_orbital_window`, `test_isosurface`, `test_hf_obs`) | 165 | 29.84 s |
| yalnız Modül 2 + 2D + 2E (`tests/md/`) | 100 | 69.23 s |
| aynısı, `-m "not slow"` | 93 | 7.07 s |

Yavaş alt küme toplam sürenin **~%81'ini** yiyor (Modül 4 kapanışında
475.33 / 585.36 s); Modül 3H kapanışında oran %82, ondan önce %79'du — yani
Modül 4 yavaş şeridi **oransal olarak büyütmedi** (yavaş şerit 51 → 71,
ve eklenen 20 işaretin hepsi Task 2–5'ten gelir; Task 6 hiç eklemedi).
Mutlak saniyeler bu makinede eşzamanlı yüke
duyarlıdır ve bu ölçüldü: 864 testlik oturumda aynı tam süite yalıtılmış üç
koşuda **255.42 / 264.98 / 270.09 s**, makinede başka bir ölçüm koşarken ise
**296.58 s** verdi (%16 fark). Aşağıdaki yapısal gözlemler bu duyarlılıktan
etkilenmez.

**Modül 3F hiç yavaş test eklememişti** — eklenen **84 testin 84'ü de** hızlı
şeritte (692 → 776) ve yavaş şerit 25'te sabit kalmıştı. En yavaş yeni test
0.80 s; dört ayrı yerde plan `@pytest.mark.slow` öneriyordu, dördü de ölçümle
kaldırıldı. Bir kez de tersi oldu: 2.90 s süren bir test referansı ızgaradan
**algebraya** çevrilerek 0.32 s'ye indi — yavaş bir testin çaresi her zaman
markerlemek değildir.

**Modül 1C ise ekledi, ve gerekçesi ölçüldü:** yavaş şerit 25 → **36**
(+11), hepsi Task 1 ve Task 3'ten — N₂/F₂ üzerinde çok başlangıçlı arama
tanımı gereği pahalıdır (bir geometri × 8–12 tohum × tam SCF) ve ucuzlatmanın
tek yolu aramayı küçültmekti, ki o zaman test **ölçtüğü şeyi ölçmezdi**.
Task 4+5'in eklediği **20 testin 20'si** hızlı şeritte: iki dosya birlikte
2.76 s. Modül 1C'nin beş dosyası toplam 31.46 s'nin 60 testini taşıyor.

**Modül 2E hiç yavaş test eklemedi.** Plan E'nin planı üç testi
`@pytest.mark.slow` işaretliyordu; ölçüldü ve kaldırıldı — en yavaşı 0.35 s,
yani hepsi "saniye altı çıpa" kuralının içine düşüyor. Sonuç: yavaş şerit
25 testte sabit kaldı, eklenen 17 testin **17'si de** hızlı şeritte.

**Modül 3H, deponun en pahalı modülü oldu — ve gerekçesi ölçüldü.** Süite
864 → **971** (+107): hızlı şerit 828 → **920** (+92), yavaş şerit 36 → **51**
(+15). Yavaş şeridin süresi 210.65 → **352.41 s**; aradaki ~142 s'nin tamamını
Modül 3H'ye yazmak **iki ayrı oturumun** karşılaştırması olur ve bu makine
eşzamanlı yüke %16'ya kadar duyarlıdır — test SAYILARI kesin, saniye farkı
yaklaşıktır. Sebep ucuzlatılamaz: bu modülün her fiziksel testi bir
**arama protokolü** koşar (bir geometri × 12–20 tohum × tam UHF), çıpa testi
üstüne `brentq` ile bir kök araması ekler ve sahte-çıpa sınavı altı ayrı baz
seti üzerinde tam bir yasa uydurması yapar (36 `search_uhf` çağrısı). Aramayı
küçültmek testin **ölçtüğü şeyi ölçmemesi** demek olurdu.

Task 5'in (senaryo + panel + README) kendi payı küçük: eklenen **25 testin
24'ü** hızlı şeritte, tek yavaş test uçtan uca CLI koşumu (5.02 s). `tests/tb`
artık 185 test / 138.55 s; hızlı şeridi ise 170 test / **16.73 s** — yani
günlük çalışma döngüsü hâlâ ucuz, pahalı olan yalnızca işaretli şerit.

**Yavaş testlerin dağılımı (ölçüldü, 864 testlik oturum — Modül 3H'nin
eklediği 15 yavaş test bu listede YOKTUR):** `hf/test_physical.py` ×5,
`hf/test_stability.py` ×5, `tdse/test_double_slit.py` ×4, `md/test_aimd.py`
×3, `md/test_maxwell_boltzmann.py` ×2, `md/test_thermostats.py` ×2,
`tdse/test_analytic.py` ×2, `tdse/test_tunneling.py` ×1,
`test_scenario.py` ×1.

**Süreyi kim yiyor:** en yavaş beş test tek başına ~169 s ediyor ve
**dördü MD değil TDSE**: `test_particle_in_box_ground_state_via_engine`
44.56 s, üç çift-yarık testi ~28.6 s'er; Nosé-Hoover ergodiklik fixture'ı
38.94 s ile ikinci sırada. Yani modül eklemek süreyi yiyen şey olmadı —
Plan D'nin **üç** yavaş testi ve tüm hızlı testleri toplam ~3 s ekledi.

**Düzeltme — "B2 süreye neredeyse hiçbir şey eklemedi" iddiası yanlıştı.**
Ölçüldü: B2'nin üç dosyasında **sıfır** yavaş test var, yani maliyeti tamamen
hızlı kısma düşüyor ve orada küçük değil: hızlı kısım 53.62 s, bunun
29.84 s'i (~%56) B2'nin üç dosyası. Eski iddia toplam sürenin sabit kalmasına
dayanıyordu (127.3 → 127.96 s), ama o sabitlik bir gürültü iptaliydi:
B2'nin hiç dokunmadığı yavaş alt küme aynı iki ölçüm arasında
112.6 → 106.67 s "hızlanmıştı".

**İlan edilmiş bir test bütçesi YOKTUR.** Plan D'nin ilk taslağı "`tests/md`
≤ 50 s, tam süite ≤ 240 s" diye bir kısıt varsayıyordu; `README.md` ve
`pyproject.toml` tarandı, öyle bir sınır **hiç tanımlı değil** — o sayılar
bu tablonun *ölçümleriydi* ve yanlışlıkla limite çevrilmişti. Yerine geçen
kural ölçüme dayanıyor: yeni testler `@pytest.mark.slow` şeridine girer,
varsayılan şeride yalnız saniye altı çıpalar konur; çünkü hızlı şerit
(Plan D'de 53.62 s, 02.10.2026'da 191.01 s) günlük çalışma döngüsüdür ve korunması tam süiteden daha
değerlidir. Kural Modül 2E'de iki kez iş gördü: plan dört testi yavaş
işaretlemişti, ölçüm dördünü de hızlı şeride geri gönderdi.

**Hızlı şeridin fizik ısırığı ölçülüdür (G7, 09.10.2026).** Envanterin "hızlı
şerit bütün modüllerde fizik çıpalarını atlıyor" iddiası ölçümle çöktü: her
modülün hızlı şeridinde bağımsız referanslı çıpalar vardı (HF'de hiç açık
yoktu). Ama modül başına fizik mutantları iki şeritte ayrı koşulunca dar
açıklar çıktı — yalnız yavaş şeridin yakaladığı, HİÇBİR şeridin
yakalamadığı ya da hızlı şeritte yalnız regresyon iğnesiyle (kendi önceki
çıktısına rel 1e-9) ölen mutantlar: tdse 2B dinamik ve m ≠ 1, md Nose-Hoover ve termostat
tekmelerinde kütle, em 2B kırılma indisi, spin M13, tb UHF spin takası ve
Δ işareti, grav sekiz BVP, bridge anharmonik A² yolu. Her biri için
`tests/<modül>/test_g7_hizli_cipa.py` (7 dosya, 18 test, tek başına 1.73 s,
en yavaşı 0.17 s) kapalı form referanslı ucuz bir ısırık taşır; her dosya bir
şüpheci denetiminden geçti ve hedef mutantları öldürdüğü ölçüldü. Bu dosyalar
yavaş şeridin doğa çıpalarının YERİNİ TUTMAZ; hızlı şeridin neyi
göremeyeceğini daraltır.

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
| PySCF toplam enerji uyumu (9 sistem: He, H₂ @0.74/1.00/2.00 Å, LiH, HF, H₂O, NH₃, CH₄; STO-3G) | `tests/hf/test_scf.py::test_total_energy_matches_pyscf`, sınır `abs=1e-10, rel=0.0` (kabul kriteri 1e-6'dır; test ölçülen sapmanın ~7.6 katında bağlayıcı tutulur) | en kötü sapma **1.32e-11 hartree** (H₂, R=2.00 Å); su -74.9630631297 hartree (fark 4.56e-12) |
| İntegral uyumu (S, T, V, ERI; H₂ ve H₂O, STO-3G) | `tests/hf/test_integrals.py`, atol=1e-10..3e-10 | en kötü mutlak sapma **2.129e-10** (su çekirdek-çekim matrisi, [5,1]) |
| H₂ denge bağ uzunluğu | Deneysel 0.741 Å, tolerans ±%8 | Yöntemin gerçek minimumu **0.71223 Å** (sapma %3.9), E = -1.117505885116 hartree. Parabol fiti kullanan taramalar bunun biraz üstünü verir: dağıtılan senaryo **0.71348 Å** (%3.7), `test_physical`'ın 40 noktalı düzgün ızgarası **0.71558 Å** (%3.4). İkisi de denge değil, ızgara eseridir — aşağıdaki nota bakın |
| H₂ bağlanma enerjisi (RHF/STO-3G, 0.74→8.0 Å) | Deneysel ayrışma enerjisi 4.75 eV | **14.6350 eV** — RHF ayrışma kusuru yüzünden ~3 kat fazla, korelasyon eksikliği DEĞİL (bkz. "Ne iddia etmiyor") |
| H₂O denge açısı | Deneysel 104.5°, tolerans ±%8 | **100.97°**, sapma %3.4 |
| H₂O dipolü | Deneysel 1.85 D | **1.7255 D**, sapma %6.7 |
| SCF yakınsama hızlandırma (DIIS vs düz Roothaan-Hall, iterasyon sayısı) | `tests/hf/test_scf.py::test_diis_reduces_iteration_count` | H₂O 19→8, LiH 23→7, NH₃ 18→8, CH₄ 12→11 |
| Yakınsamayan SCF sessizce sayı döndürmemeli | `tests/hf/test_scf.py::test_non_convergence_raises_rather_than_returning` | `SCFConvergenceError` fırlatılır, iterasyon/enerji geçmişi hatanın içinde taşınır |
| Referans koruması gerçekten korur mu | `reference/hf_energies.json` geçici kaldırılıp `pytest tests/hf/` koşuldu | **kırmızı**, atlanmadı: **92 error + 2 failed, 234 passed, 0 skipped** (toplam 328). 92 hatanın hepsi `pytest.fail`: "Referans dosyasi bulunamadi"; 2 failed, alt süreçte pytest koşan yol-sırası testidir ve o alt süreç de aynı sebeple kırmızıdır |

Not (bağ uzunluğu — neden üç sayı): Bu README bir süre yalnızca 0.7156 Å'yı
bildirdi ve onu "denge bağ uzunluğu — ölçülen" diye sundu. İki ayrı hata
vardı. Birincisi: o sayı bir taramanın çıktısıdır, yöntemin dengesi değil.
İkincisi: kabul kriteri 2 kanıt olarak `h2_bond_scan.yaml` komutunu gösterip
o komutun **üretmediği** bir sayıyı bildiriyordu — 0.7156 Å, testteki
`linspace(0.4, 3.0, 40)` ızgarasından (adım 0.0667 Å) geliyordu.

Parabol fiti, minimumun etrafındaki üç ızgara noktasına oturtulur ve kaba
ızgarada minimumu yukarı kaydırır. Ölçülen üç sayı:

| Kaynak | Minimum civarındaki örnekleme | Fit | Gerçek minimumdan yanlılık |
|---|---|---|---|
| Brent (ızgarasız arama) | — | **0.7122299 Å** | 0 (tanım gereği) |
| `scenarios/h2_bond_scan.yaml` | 0.65 / 0.70 / 0.74 Å | **0.7134824 Å** | **+0.0012525 Å** |
| `test_h2_potential_curve_has_a_minimum_near_experiment` | 0.0667 Å düzgün adım | **0.7155781 Å** | **+0.0033483 Å** |

Sıralama iddiayı ölçüyor: minimum civarında daha sık örnekleyen ızgara daha
küçük yanlılık veriyor — yani eser gerçekten ızgaradan geliyor, yöntemden
değil. Bu sıralamanın kendisi
`test_shipped_h2_scan_fit_is_grid_biased_but_less_than_the_uniform_grid`
ile kilitli.

Fark akademik değil, çünkü kuvvet üç büyüklük mertebesi farklı: sayısal
|dE/dr| gerçek minimumda **8.7e-09 hartree/bohr** (yani sıfır, gürültü
seviyesi), 0.7156 Å'da **3.589e-03 hartree/bohr**. Modül 2'nin moleküler
dinamik testleri "denge" diye kaba fite demirlerse, molekülü dengede
sandıkları noktada iterken bulurlar. Gerçek minimum ve kaba fit
`test_coarse_scan_fit_sits_above_the_true_sto3g_minimum` ile ayrı ayrı
kilitli: hangisinin hangisi olduğu artık nesirle değil testle korunuyor.

Not: İntegral satırındaki 2.129e-10'luk artık özyinelemenin kendisinden değil,
referansı üreten PySCF derlemesinin bohr sabitinden (CODATA 2010,
0.52917721092 Å) qlab'ın kullandığı sabitten (CODATA 2018, 0.529177210903 Å)
gelir; bu fark çekirdek konumlarına geçer ve oradan her integrale kendi
çekirdek-konumu türevi üzerinden sızar. PySCF'in kendi bohr sabitiyle kurulan
aynı geometride dört tensör de rtol=0, atol=1e-12 ile eşleşir
(`test_water_matches_pyscf_to_machine_precision_at_pyscf_geometry`) — bu,
özyinelemeyi kilitleyen asıl kapıdır; yukarıdaki 1e-10 seviyesi birim-sabiti
farkını ölçer, integral hassasiyetini değil.

### Modül 1B2 — orbital görselleştirme

Bu tablodaki sayılar bu görevde (B2 Task 10) gerçekten koşturulup ölçülmüştür.
Modül 1B'nin aksine burada **bağımsız bir dış referans yoktur**: doğrulama
fikri, ızgara üzerinde sayısal olarak hesaplanan büyüklüğün motorun kendi
ANALİTİK sonucuna oturmasıdır. Bu, yeni kodu Modül 1B'nin PySCF'e karşı
doğrulanmış motoruna zincirler; "PySCF'ten daha doğru" gibi bir iddia
taşımaz.

| Test | Referans / tolerans | Ölçülen |
|---|---|---|
| Izgara örtüşmesi ↔ analitik `overlap_matrix` (H₂/STO-3G, padding 7 bohr, n=80, `np.einsum`) | Motorun kendi analitik integrali | en kötü fark **2.700e-09** |
| Elektron sayısı: ∫ρ dV (H₂/STO-3G, padding 6 bohr, n=120) | Tam olarak 2 | **1.9999993405** (kalan kutu kuyruğunun kesilmesinden) |
| Su HOMO'sunun ızgara normu | Analitik 1 | **1.00000000** |
| Su çekirdek 1s orbitalinin ızgara normu | Analitik 1 | **0.98760196** — %1.24 eksik; ızgaranın çekirdek tepesini çözemediğinin sayısal kanıtı |
| Yoğunluk farkı sıfıra integre oluyor mu (H₂) | Molekül ve promolekül aynı elektron sayısını taşır | **−1.411e-08** |
| **Kovalent bağda elektron birikimi** (H₂, bağ bölgesi \|x\|<1, \|y\|<1, 0.2<z<1.2 bohr) | Yön iddiası: çekirdekler arasında ARTMALI | bağ bölgesinde **+0.055005 elektron**, dış bölgede (z<−1.0 veya z>2.4) **−0.027068 elektron** (ikisi de bölge üzerinden İNTEGRE); haritanın ham tepe değeri ayrı bir büyüklüktür: **+0.042281 e/bohr³** (z=+0.643 bohr), ham dip **−0.064390 e/bohr³** |
| **Polar bağda yük transferi** (HF, r=0.917 Å, orta düzlem z=0.8664 bohr) | Yön iddiası: elektronegatif atoma GİTMELİ | H yarısı **−0.1841**, F yarısı **+0.1841** — üç ayrı ızgarada ±%0.2 |
| İzoyüzey yarıçapı (analitik küre `x²+y²+z²`, level=1.0, bohr) | Yarıçap 1.0 | ort **0.999215**, std 0.000363; indeks→bohr dönüşümü atlanırsa aynı büyüklük 35.600985 çıkar |
| İzoyüzey ağının yön tutarlılığı | Her iki fazın normali DIŞARI bakmalı | dışarı-bakan üçgen oranı **1.000 / 1.000** |

"Bağ", bu tabloda hiçbir yerde koda yazılmamıştır. `+0.055005` ve `0.184`
elektron, iki ayrı hesabın (molekül ve etkileşmeyen serbest atomlar) farkından
OKUNMUŞTUR. Elektronegatiflik de öyle: HF'nin hangi ucuna elektron gittiği
hiçbir tabloda tanımlı değildir, haritadan çıkar.

**İki birim karışmasın.** Bölge sayıları (`+0.055005`, `−0.027068`) bölge
üzerinden integre edilmiş ELEKTRON sayılarıdır (`values[maske].sum() *
volume_element`); `+0.042281` ve `−0.064390` ise haritanın tek bir noktadaki
HAM yoğunluk farkıdır (e/bohr³). Bunlar aynı büyüklüğün iki okuması değildir:
ölçüldü, tepe noktasının kendi hücresi üzerinden integrali 4.841e-05
elektrondur — bağ bölgesi sayısının ~1/1136'sı. Tablo bir süre üçünü aynı
hücrede, birim ayrımı yapmadan yan yana koyuyordu.

**Izgara örtüşmesi satırının kesinliği hakkında.** Orada `np.einsum` yolunun
yazılı olması gereksiz bir ayrıntı değil: bu büyüklükteki artıklar float64
toplama SIRASINA duyarlıdır. Ölçüldü — daha ince bir ızgarada (padding 8 bohr,
n=120) aynı hesap `np.einsum` ile 1.669e-11, `np.tensordot` ile 1.610e-11
verir; **%3.6 yayılım**, yani 4. anlamlı hane indirgeme yolunun eseridir.
Tablodaki satır bu yüzden deponun kendi testinin
(`test_izgara_ortusmesi_analitik_ortusmeye_oturuyor`) gerçekten koştuğu
ızgarayı ve kod yolunu bildirir; o ızgarada iki yol 5 haneye kadar örtüşür
(2.699635e-09 / 2.699516e-09).

### Modül 1C — kısıtlanmamış Hartree-Fock (UHF)

**Bu modülün kazancı enerji değil, DÜRÜSTLÜK.** UHF buraya bir "özellik"
olarak değil, bir **ön koşul** olarak girdi: Plan G (tight-binding
parametrelerini deponun kendi HF'inden türetme) yazılacaktı, ön ölçüm
referans sisteminin RHF çözümünün HF'in taban durumu **olmadığını** gösterdi
ve downfolding'in çıpası yanlış bir yüzeye oturacaktı. Plan ertelendi, yerine
bu modül yazıldı. (Çekirdek spec §13'te "kapsam dışı" yazan madde bu yüzden
üstü çizilerek düzeltildi.)

#### Deponun kendi referans kaydı taban durum değildi

`reference/hf_energies.json`'daki **11 PySCF kaydının 10'u** çok başlangıçlı
UHF aramasıyla **≤ 9.001e-12 hartree** uyuşuyor. Ayrılan tek kayıt `h2_200`
(H₂ @ 2.00 Å):

| büyüklük | değer |
|---|---|
| E_RHF (referans kaydı) | **−0.783792654277** |
| E_UHF (12 tohumlu arama) | **−0.937212833096** |
| fark | **0.153420178832 hartree** |
| iç (RHF→RHF) kararlılık | **+2.546683038854** → *"kararlı"* |
| dış (RHF→UHF) kararlılık | **−1.599532559297** → **spin eyeri** |

Yani depo, gerçek HF taban durumundan 0.153 hartree yukarıdaki bir noktaya
"iç minimum" diyordu ve bu çıktının **hiçbir yerinde** görünmüyordu. Modül 1C
o sessizliği kaldırıyor: `solve(stability=True)` artık iki kanalı da koşuyor
ve panel iki ayrı satır basıyor.

#### Coulson-Fischer noktası — modülün en güçlü çıpası

RHF çözümünün spin kanalında kararlı olmaktan çıktığı R. Fiziksel bir
**çatallanma** noktasıdır; deneysel hiçbir sayı içermez, hiçbir yerden
verilmez, **okunur**.

| yol | ne KULLANMIYOR | Rc (bohr) |
|---|---|---|
| (a) triplet Hessian'ın kökü | — | **2.179694793840** |
| (b) simetriden kurulmuş orbitaller + sonlu fark | **SCF yok, analitik Hessian yok** | **2.179694793998** |
| (c) SCF'in kendi davranışı (UHF/RHF ayrımı) | Hessian yok | aynı sınır |

Fark **1.578e-10**, ve bu fark **kesme değil YUVARLAMA** — ölçüldü: h
küçüldükçe fark *büyüyor* (3e-3 → 3.736e-10, 1e-3 → 1.275e-09, 3e-4 →
3.734e-09), yani h=1e-3 zaten yuvarlamanın hâkim olduğu bölgede.
Rc = 1.153444811624 Å.

Çatallanma üsleri hiçbir yerden verilmedi, uyumdan çıktı:

| pencere | \|ΔE\| | \|spin\| | ⟨S²⟩ |
|---|---|---|---|
| d ≤ 2e-1 | 1.9767 | 0.4825 | 0.9727 |
| **d ≤ 1.6e-2** | **1.9960** | **0.4970** | **0.9953** |
| çatallanma beklentisi | 2 | 1/2 | 1 |

Zincirlerde Rc kayıyor: n=2 **2.179694794**, n=4 1.951986163, n=6 1.804435695
bohr (ölçüldü ama iddia edilmiyor: n=8 1.695323976, n=10 1.609356683, n=12
1.538971647 — 1/n uyumu artığı 0.089'a çıkıyor, altı nokta n→∞ limitini
seçmeye **yetmez**).

#### Çıpa tablosu — SINIFIYLA birlikte

Tautolojiler gizlenmiyor; sınıflarıyla yazılıyor. Sınıflar:
*dairesel değil* (bağımsız ikinci yolu var) · *transkripsiyon denetimi*
(özdeşlik, ama mutasyona duyarlı) · *tautoloji* (fizik-dışı girdide de
tutuyor).

| iddia | ölçülen | sınıf |
|---|---|---|
| **Coulson-Fischer noktası** | üç bağımsız yol, fark 1.578e-10 | **dairesel değil — en güçlü** |
| **Çatallanma üsleri** | 1.9960 / 0.4970 / 0.9953 (beklenti 2 / ½ / 1) | **dairesel değil** |
| Parça toplamı (boyut tutarlılığı) | H₂ +5.551e-16, LiH −9.592e-14, N₂ +1.421e-14 | dairesel değil, **güçlü** |
| Ayrışma limiti | UHF(H₂, 20 bohr) − 2E_H = **4.441e-16**; RHF aynı yerde **9.85877 eV** sapıyor | dairesel değil, **zayıf** (iki mutant geçiyor) |
| Triplet Hessian = enerjinin 2. türevi | bağıl fark 1.305e-10 … 6.659e-10, beş sistemde | transkripsiyon denetimi (**taşıyıcı**) |
| λ_triplet < 0 ⟺ UHF kazancı var | gerçek sistemlerde **18/18**; fizik-dışı girdide de **51/53** | **muhafız** — sahte-çıpa sınavını geçiyor, çıpa DEĞİL |
| E_UHF ≤ E_RHF | 0 karşı örnek | transkripsiyon denetimi, **zayıf** (10 mutantın 7'sini kaçırıyor) |
| RHF kararlıyken UHF özdeş | fark < 1e-12 | **TAUTOLOJİ** — muhafız, "çıpa" değil |

Son satırdaki sahte-çıpa sınavı **iki ayrı soruyu** ayırıyor ve ikisinin
cevabı farklı:

| ölçüt | gerçek H₂ (18 ayrım) | fizik-dışı rastgele integraller (53 yakınsayan) |
|---|---|---|
| **triplet** kanal | **18/18** | 51/53 |
| **singlet** kanal | **8/18** | 50/53 |

Fizik-dışı girdide tuttuğu için hüküm **fizik değil** (cebirsel bir sonuç:
Hessian, SCF'in indiği aynı fonksiyonelin ikinci türevidir). Ama sahte girdi
iki kanalı **ayıramıyor**; ayrım gerçek moleküllerde çıkıyor. Yani hüküm
fizik değil ama **doğru cebir** — yanlış kanalla verildiğinde gerçek
moleküllerde 10/18 düşüyor.

#### Sevk edilmiş kodda bulunan kusur — ve Modül 2D'ye etkisi

Task 1 deponun **zaten sevk edilmiş** `solve()`'unda iki ayrı kusur ölçtü;
ikisi de `converged=True` diyordu:

| kusur | nerede | ölçülen |
|---|---|---|
| **DIIS** | N₂ R=1.10 Å | DIIS'li −106.769673858 vs DIIS'siz **−107.496500512** → **0.726827 Ha = 19.78 eV** |
| **başlangıç tahmini** | N₂ R=1.30–2.00 Å | 2–5 ayrı çözüm; en düşüğü **0.278–0.498 Ha** aşağıda (DIIS masum) |
| aynı kusur, ikinci molekül | F₂ R=1.80 / 2.00 / 2.40 Å | **0.4094 / 0.2513 / 0.0812 Ha**; 3–4 ayrı çözüm |

F₂'nin **kendi dengesi** etkilenmiyor, ve o denge deneysel bir sayı değil
**ölçüldü**: yöntemin STO-3G minimumu **R = 1.314612 Å**,
E = −195.981624421. Orada 12 tohum tek çözüme iniyor (fark 1.705e-13) ve iki
kanalda da kararlı (iç **+1.304533**, dış **+0.228957**). Kusur gerilmiş
geometrilerin kusurudur.

**Modül 2D bu `solve()` üzerine kuruludur, dolayısıyla etki ÖLÇÜLDÜ:**

| R (bohr) | çok başlangıçlı RHF farkı | ayrı çözüm | iç λ | dış λ | UHF kazancı |
|---|---|---|---|---|---|
| 1.2960 (MD dönüş noktası) | 4.441e-16 | **1** | +4.77335 | **+1.92166** | **0.000e+00** |
| 1.3459 (denge) | 4.441e-16 | **1** | +4.64526 | **+1.77054** | **0.000e+00** |
| 1.3960 (MD dönüş noktası) | 0.000e+00 | **1** | +4.52409 | **+1.62586** | **0.000e+00** |
| 2.1797 (= Rc) | 2.220e-16 | 1 | +3.30753 | −0.00001 | 4.733e-12 |
| 2.5000 | 6.661e-16 | 1 | +3.03687 | −0.45013 | 1.415e-02 |
| 4.0000 | 0.000e+00 | 1 | +2.51654 | −1.72551 | **1.748e-01** |

**Hüküm: Modül 2D'nin sonuçları etkilenmiyor, ve bu bir varsayım değil
ölçüm.** Ab initio MD H₂'yi 1.3959 bohr'dan başlatıp ~0.05 bohr genlikle
titreştiriyor; gezdiği [1.296, 1.396] aralığının her noktasında (i) çok
başlangıçlı RHF, **12 tohumun 12'sinde de aynı** çözüme iniyor (`n_distinct`
= 1, enerji farkı ≤ 6.7e-16) ve (ii) dış kararlılık özdeğeri **pozitif**,
UHF kazancı **tam sıfır**. Yani ω_A / ω_B uyuşması (milyonda 4.4) gerçek HF
taban durumu üzerindedir.

*(Bu, "H₂/STO-3G'de tek bir RHF durağan noktası vardır" demek DEĞİLDİR —
σu² işgali de bir durağan noktadır. Ölçülen şey, 12 tohumluk bu arama
protokolünün bu geometrilerde daha düşük bir çözüm bulamadığıdır.)*

**Ama modülün geometri kapısı 21.1 bohr'a kadar açık** ve R > 2.1797 bohr'da
aynı yüzey bir spin eyeridir — 4.0 bohr'da gerçek HF taban durumu
**0.1748 hartree aşağıdadır**. Modül 2D'nin BO yüzeyi bu yüzden yalnızca
Rc'nin altında HF taban durumudur; üstünde "RHF yüzeyi"dir ve öyle
okunmalıdır. *03.10.2026'dan beri panel bunu kendisi söyler (B10):* her adımın
merkez SCF'i iç ve dış kararlılık analiziyle çözülür ve en kötü hüküm
"Kararlılık" / "Dış kararlılık" satırlarına girer — durgun R₀ = 2.5 bohr
koşusunda "✗ SPİN EYERİ" (λ = −0.4501), sevk edilen `h2_aimd`'de iki satır da
✓. Yakınsama hükmü değişmez; kapı Rc'ye daraltılmadı (Rc moleküle özgüdür,
kararlı H₂O/LiH/He₂ dengelerini reddederdi).

#### M9 deliği — tasarlanmış bir mutant dokuz denetimi de geçti

Task 3'ün kabul ölçütü şu mutanttı: doğru Fock, doğru enerji, ama artık
`E += 1e-3 · Σ(Mulliken spin popülasyonu)²`, ve yalnız N > 2'ye kapılı. Ön
ölçümde önerilen **dokuz denetimin dokuzunu da geçiyordu** — çünkü önerilen
muhafızın dört vakasının dördünde de spin popülasyonu sıfırdı.

Delik gerilmiş H₄ eklenerek kapatıldı, ve muhafızın **taşıyıcı** olduğu δ
küçültülerek kanıtlandı:

| δ (ceza katsayısı) | kırılan test |
|---|---|
| 1e-3 | yedi (atom enerjileri, parça toplamı, HF sınırı ve muhafız) |
| 1e-10 | iki: LiH parça toplamı ve muhafız |
| **1e-12** | **yalnız muhafız** |
| 1e-13 | hiçbiri (eşiğin altında) |

*(Protokol etiketi: bu tablo B7 öncesi, **izlemesiz** `search_uhf` ile
ölçüldü; 09.10.2026'da varsayılan kararlılık izlemesine geçildikten sonra
yeniden koşulmadı. "HF sınırı" bugünkü izlemesiz sınır testidir.)*

**Ders:** bir mutantın ölmesi, onu öldürmek için yazdığın testin çalıştığı
anlamına gelmez. *Hangi* testin öldürdüğünü sor.

#### Kök okuyan çıpa ÖLÇEĞE KÖRDÜR

Bu depoda ilk kez ölçüldü ve çıpa tasarımı hakkında geneldir. Triplet
Hessian'ın `4.0` çarpanı `2.0` yapıldığında **Rc hiç kıpırdamıyor** — küresel
bir çarpan bir sıfır geçişini yerinden oynatmaz:

| test | çarpan mutantı |
|---|---|
| `test_COULSON_FISCHER_noktasi` | **yeşil** |
| `test_ZINCIRLERDE_Rc_n_ile_KAYIYOR` | **yeşil** |
| `test_TRIPLET_OLCUTU_...SINGLETTEN_iyi` | **yeşil** |
| `test_Rc_IKINCI_BAGIMSIZ_YOLDAN_SCF_SUZ` (aralık boyu) | kırmızı |
| `test_TRIPLET_Hessian_SONLU_FARKLA_uyusuyor` | kırmızı |
| `test_TRIPLET_SINGLETTEN_AYRI_HUKUM_veriyor` | kırmızı |

**Kök = FİZİK, ölçek = TRANSKRİPSİYON.** İki ayrı test sınıfı gerekir ve biri
diğerinin yerine geçmez; yalnız çıpalarla yetinilseydi çarpanı yanlış yazılmış
bir Hessian sevk edilirdi.

Diğer mutantlarda Rc nereye gidiyor: Coulomb terimi **bırakılırsa** (yani
singlet yazılırsa) kök **hiç yok** — H₂'nin singlet Hessian'ı 0.6–12 bohr
arasında negatife inmiyor; yarım bırakılırsa yine kök yok; Fock bloklarının
işareti takas edilirse yine kök yok; `(ab|ij)` terimi düşerse kök
**4.570719109 bohr**'a kayıyor (2.39 bohr hata — ama VAR, yani "kök var mı"
diye soran bir test bunu kaçırırdı).

#### Dürüst olumsuzlar

- **⟨S²⟩ enerjiden ~6 mertebe yavaş yakınsıyor.** Ölçüldü: SCF toleransı
  1e-6'da enerji zaten 2.585e-12'ye oturmuşken ⟨S²⟩ hâlâ **1.457e-06**
  oynuyor. Sebebi yapısaldır — enerji durağan noktada ikinci mertebeden
  duyarlıdır, ⟨S²⟩ değildir. ⟨S²⟩'den yedi anlamlı haneden fazlası
  okunmamalıdır.
- **UHF bir spin özdurumu değildir.** `h2_200`'ün UHF çözümünde ⟨S²⟩ =
  **0.945862370628**, oysa bir singlet için 0 olmalıdır. Bu spin kirlenmesi
  ölçülüp raporlanıyor ama **düzeltilmiyor**: spin projeksiyonu (PUHF) yok.
- **Her UHF enerjisi bir ARAMA protokolüyle gelir** ve çıpanın gücü kodun
  değil aramanın özelliğidir. B7 (09.10.2026) bunu **kararlılık izleyen
  aramayla** kapattı: `search_uhf`in tohum ailesi aynen kaldı, tohumların en
  iyisine analitik UHF orbital Hessian'ıyla (`uhf_orbital_hessian`, reel α/β
  dönmeleri) kararlılık izleme uygulanıyor ve **varsayılan** budur
  (`follow_instability=True`). HF molekülünde (H + F, 20 bohr, STO-3G) parça
  toplamı −98.453086808293:

  | tohum | izlemesiz (`follow_instability=False`) | izlemeli (varsayılan) |
  |---|---|---|
  | 1 | −97.570681531688 (+0.8824, ✗ eyer) | −98.453086808293 (1 iniş) |
  | 8 / 12 | −97.570681531688 (**+0.8824**, ✗ λ −1.764811) | −98.453086808293 (1 iniş) |
  | 24 / 40 / 64 | −97.678222271884 (**+0.7749**, ✗ λ −1.549729, 3 ayrı çözüm) | −98.453086808293 (1 iniş) |

  İzlemeli sütunun parça toplamına uzaklığı 0 … +1.4e-14 (≤ 1 ulp; son hane
  izleme ayrıntısına bağlı, `==` ile kilitlenmez). **Kök neden tohum SAYISI
  değil ÖLÇEĞİ:** ailenin en büyük |bias|'ı 1.3008, H 1s'i doldurmak için
  gereken tek bileşenli bias ise 7.33 < B* ≤ 7.34 (Hcore köşegeninde F 2p ile
  H 1s arası 7.334 Ha). Ailenin bulduğu iki durum da UHF **eyeridir** ve depo
  bunu görmüyordu — UHF → UHF kararlılık analizi yoktu; izlemesiz yol bugün
  aynı sayıları bit bit verir ama panelde "✗ dönen çözüm UHF EYERİ" basar.
  **N atomu:** tek `solve_uhf` bir durağan nokta döndürür ve bias'a bağlıdır
  (tekbiçim −53.719010163, alternan −53.259271159, **0.46 Ha** fark) — bu
  tasarım gereğidir ama artık sessiz değildir: `stability=True` alternan
  sonucu ✗ λ −0.919478 diye basar; `search_uhf(12)` −53.719010162594 (✓ λ
  +0.919478, iniş 0) döndürür. **Çatallanma:** tek sabit bias (0.35) ile UHF
  bu makinede 7 noktalı ızgarada d = 1e-2'de, 5 noktalıda d = 4e-3'te RHF'e
  geri çöküyor ve |ΔE| üssü `nan` çıkıyor (çöküş yeri makineye bağlı,
  kilitlenmez); `stability=True` tam o noktalarda ✗ basar. 5 noktalı
  ızgarada (d = 1e-3 … 1.6e-2) izlemesiz `seeds=1` her noktada çöküyor,
  izlemeli `seeds=1` hiçbir noktada çökmüyor (1.996042 / 0.497009 /
  0.995426); Rc'ye daha yakında (d ≤ 6e-5) izlemeli `seeds=1` de eyerde
  kalıyor — aşağıdaki kör bölge maddesi. Yukarıdaki tablo `search_uhf`
  (12 tohum) ile ölçülmüştür ve izleme ona dokunmaz (bit bit aynı). ⟨S²⟩
  üssünün 4. hanesi protokole bağlıdır: 12 tohum 0.995332, `seeds=1` +
  izleme 0.995426. Sabit bias 0.35'in izlemesiz üssü (0.995360) ise çöküşe
  **duyarsızdır**: çöken d = 4e-3 log-ızgaranın tam ortasında, en-küçük-kareler
  eğiminde ağırlığı sıfır — ⟨S²⟩ orada 1.3e-15'e düştüğü hâlde üs aynı
  çıkıyor (orta nokta atılınca da 0.995360). Çöküşü üs değil ayrı |ΔE|
  denetimi yakalar.
- **"UHF-kararlı" YEREL minimum demektir, KÜRESEL değil — ölçüldü.** H8
  halkası d = 3.0 bohr STO-3G'de Haar-rastgele başlangıçlardan (kamu
  `solve_uhf(initial_orbitals=…, stability=True)`, 20 başlangıç × DIIS'li /
  DIIS'siz) 8 ayrı durağan nokta, **3 yerel minimum** bulundu: −3.8572719930
  (küresel), −3.7773306557 (+0.0799) ve −3.7762350559 (+0.0810); izleme
  (depodaki özel `_follow_instability` ile, eyerlerden doğrudan) beş eyerin
  ikisinden küresel olmayan minimuma iniyor. `search_uhf(≥ 4)` bugün
  küreseli buluyor ama bu bir garanti değildir; panel "UHF YEREL minimum —
  küresel minimum iddia EDİLMEZ" der.
- **Yumuşak mod bandı: |λ| < 1e-8'de kararlılık ayırt edilmez.** Ölçülen
  bant [2.6e-10, 2.1e-7]: kararlı bir çözümde sıfır moda bulaşan SCF artığı
  en çok −2.579e-10 (HF/6-31G, 20 bohr), en küçük gerçek kararsızlık
  −2.116e-7 (HF/STO-3G, 8 bohr, π-delik eyeri: izlemesiz `search_uhf(12)`'nin
  en iyisi; varsayılan izlemeli arama 1.054e-7 Ha aşağıdaki σ-delik
  minimumuna iniyor, λ +2.099e-7). Daha büyük R'de λ bandın içine giriyor:
  R = 9 bohr'da `search_uhf(12)`'nin en iyisi λ = −5.189e-9 ile **zaten bant
  içinde** ve ✓ basılıyor (izleme dokunmuyor, iniş 0), R = 10'da −7.571e-11.
  R = 9 noktasının gerçekten bir eyer olup olmadığı (altında σ-delik minimumu
  var mı) ölçülmedi. İzlemesiz `search_uhf(12)`'nin 12 tohumundan R = 8'de
  yalnız 1'i, R = 9'da 3'ü, R = 10'da 4'ü yakınsıyor ve raporda bunun için
  bir uyarı yok.
- **Kabul eşiğinin kör bölgesi: izleme her eyerden inmez — ölçüldü.** İzleme
  yeni çözümü yalnız `_IZLEME_KABUL` = 1e-9 Ha'dan fazla inerse kabul eder;
  bir çatallanmanın hemen yanında gerçek iniş ΔE ∝ d² ile keyfi küçüktür.
  H2/STO-3G, R = Rc + d, varsayılan `search_uhf(seeds=1)`: d ≤ 6e-5'te (λ =
  −7.7e-7 … −4.6e-5, yumuşak mod bandının 77–4600 katı **dışında**) kırık
  simetrili minimum RHF eyerinin yalnız 1.7e-13 … 6.3e-10 Ha altında; izleme
  bu inişi reddediyor (iniş 0) ve panel "✗ dönen çözüm UHF EYERİ" basıyor.
  Aynı noktalarda `search_uhf(12)` minimumu buluyor (✓); d = 8e-5'te iniş
  1.12e-9 ve kabul ediliyor. Engel SCF değil eşiktir: depodaki özel yolda
  eşik 1e-13'e indirilince d = 1e-6 / 1e-5 / 6e-5'te tek inişle minimuma
  iniliyor. Eşik düşürülmedi — alt sınırı yumuşak yüzeyde SCF durma
  ölçütünün enerji izi (kuramsal ≈2.5e-10) ve geri çöküşü iniş saymama
  görevi. Hüküm dürüst (sessizlik yok) ama sınır açıktır; genişliği yalnız
  H2 çatallanmasında ölçüldü (halkalar, LiF, 6-31G ölçülmedi).
- **PySCF bu makinede kurulu değil** (yalnız kaynak tarball); `reference/`
  kayıtları PySCF **2.14.0** ile başka bir yerde üretilmiştir. Dahası 11
  referans kaydının **hepsi kapalı kabuktur**, yani UHF'in **spin kırılan
  dalını hiç sınamıyor** — o dalın doğrulaması bu depoda yalnız iç tutarlılık
  (ayrışma limiti, boyut tutarlılığı, çatallanma) üzerinden yapılır.
- **Deneysel bağ uzunluğu hiçbir yerde girdi değildir.** F₂ için yapılan
  taramada yöntemin kendi dengesi (1.30 Å) kullanılmıştır.
- **Kompleks (reel olmayan) orbital kanalı hâlâ sınanmıyor.** İç, dış ve
  UHF kanalının üçünde de "kararlı" hükmü, çözümün **küresel minimum**
  olduğunu söylemez; UHF Hessian'ı yalnız reel α/β dönmelerini içerir.

### Modül 2 — klasik moleküler dinamik

Aşağıdaki sayılar bu görevde gerçekten çalıştırılıp ölçülmüştür. Modülün
iddiası tek cümleyle: **integratör kuvvetin nereden geldiğini bilmez** —
tek sözleşme `(konumlar) -> (enerji, kuvvetler)`, ve harmonik/Morse/
Lennard-Jones onun üç uygulamasıdır.

| Test | Referans / sınır | Ölçülen |
|---|---|---|
| Periyot analitiğe oturuyor | T = 2π√(m/k), sınır \|T/T_analitik − 1\| < 1e-3, N ≥ 200 şart | bağıl hata **4.11e-05**, kapalı form −π²/(6N²) ile 4 hanede uyuşuyor |
| Enerji SÜRÜKLENMİYOR | blok-ortalama baş-son farkı < 1e-6 (N=200, 1e5 adım, save_every=10) | **2.32e-10**, marj 4305× |
| Enerji SALINIMI kapalı forma uyuyor | tepe-tepe/E₀ = π²/N², sınır \|oran − 1\| < 0.05 | oran **1.0000** (üç N değerinde) |
| Global hata ikinci mertebeden | dt yarılanınca oran 4.0 ± %35, beş dt | **4.0018 → 4.0000** |
| Fiziksel çıpa — H₂ titreşimi | Deneysel ω_e = 4401.21 cm⁻¹, sınır ±5.0 | **4401.39 cm⁻¹** (ölçülen periyottan; ayrıklaştırma yanlılığı +π²/6N²) |
| Anharmoniklik: ω(E) = ω₀√(1−E/Dₑ) | sınır 2e-3, 800 adım/periyot | E/Dₑ=0.10 **0.948686** (marj 590×), 0.50 **0.707120** (155×), 0.90 **0.316508** (7.1×) |
| Morse dış dönüş noktası | analitik 4.287169 bohr, bant 4.2–4.4 | **4.280174** |
| LJ analitik çıpaları | r_min = 2^(1/6)σ, V(r_min) = −ε | **7.2225016416 bohr**, oran **1.00000000000000** |
| LJ kesme sürekliliği | sınır 1e-9 (prob 1e-6) | **2.299439e-12**, marj 435× |
| Langevin hedef sıcaklığa oturuyor | 300 K, ±%2 (400 replika, seed=7) | **297.785 K** (−%0.738), marj 2.71× |
| BAOAB sürtünmesiz limitte velocity-Verlet | γ=1e-14, 40 adım, sınır 1e-5 bohr | **1.08e-07** |
| Nosé-Hoover ergodiklik kusuru (1 dof) | düz NH < −4 %, zincirli \|·\| < 3 % | düz **−8.18 %**, zincirli **+0.15 %** (40 tohum) |
| Kusur KONUMDA, sıcaklıkta değil | \|düz ⟨v²⟩ sapması\| < 1 % | **+0.019 %** (marj 8.8×) |
| Maxwell-Boltzmann kendiliğinden çıkıyor | R4 = 5/3, 4σ; artı ⟨T⟩ = 87.2 K ±%10 | R4 sınır içinde; ⟨T⟩ **87.295 ± 1.763 K** (16 tohum) |
| MD senaryosu CLI'dan uçtan uca | `python -m qlab.run scenarios/h2_morse.yaml -o md.npz` | çıkış 0, 1001 kayıt, son an **75.78917958522622 fs** |

**Bu tablodaki her satır bir mutasyon sınavından geçti.** velocity-Verlet'in
yedi mutantı (forward Euler, simplektik Euler, yarım tekme eksik, a=F,
kuvvet işareti ters, a=F/m², bayat kuvvet) tek tek koşuldu ve **yedisi de**
en az bir testi kırmızı yapıyor.

**Kütle boru hattı — modülün en tehlikeli hatası.** Atomik birimde kütlenin
birimi elektron kütlesidir; "1 amu = 1" yazan kod 1822.888486× hafif bir
molekül simüle eder. Ölçüldü: periyot 313.322877 → 7.338580 a.u.
(çarpan 42.695298 = √AMU_TO_ELECTRON_MASS) ve **hiçbir şey patlamaz** —
enerji bile korunur, çünkü N=200 adım/periyotta bağıl salınım
π²/N² = 2.467401e-04'tür ve bu sayı doğru ile hatalı kütlede **birebir
aynıdır**. Yakalayan tek şey fiziksel çıpadır: 4401.39 yerine 187918.70 cm⁻¹.
Bu yüzden `MU_H2` testte literal DEĞİL, `amu_to_electron_mass`'ten geçer —
literal olduğu sürece `AMU_TO_ELECTRON_MASS = 1.0` mutantı Task 4'ün altı
testini de geçiyordu (ölçüldü).

**Nosé-Hoover'ın ergodiklik kusuru nerede DEĞİL.** 200–400 bağımsız osilatörü
tek global ζ'ya bağlayan kurulumda kusur ölçüm çözünürlüğünün altındadır
(Langevin kontrolü aynı kurulumda +0.017 ± 0.096 %). Kusur **tek** serbestlik
dereceli osilatörün özelliğidir ve testler orada kurulur. Ergodikliğin asıl
imzası sapmanın büyüklüğü bile değil, zaman ortalamalarının saçılımının
pencereyle küçülmemesidir:

| ölçüm penceresi | düz NH | zincirli (M=3) |
|---|---|---|
| 8 P | 21.203 % | 18.941 % |
| 40 P | 21.326 % | 8.824 % |
| 160 P | **21.170 %** | **4.186 %** |

20 kat pencerede düz NH'de saçılım hiç küçülmüyor; zincirlide 4.52 kat
küçülüyor (ideal √20 = 4.472).

**Dürüst olumsuzlar (ölçüldü, gizlenmiyor):**

- Maxwell-Boltzmann bir **kuvvet alanı sınavı değildir**. ε×10, ε×0.1, σ×0.9
  ve WCA kusursuz MB veriyor; dahası kuvvet alanı TAMAMEN kaldırılsa (ideal
  gaz) bile test geçiyor. Ölçek çıpası (kinetik sıcaklık) bu yüzden şart:
  termostat kaldırılınca ⟨T⟩ = **6.24e-26 K** ve test ancak o çıpayla kırmızı olur.
- Morse'un V(∞) testi `a`yı yalnızca **alttan** sınırlar; büyümesini asla
  yakalamaz (a=2.058 ve a=1e6 için sapma tam 0.0). Üst sınır V″(r_e)
  testinden gelir.
- Termostat kütlesi Q → 2Q hatası fiziksel gözlemlerin hiçbirinde eşiği
  aşmıyor; onu yalnızca `thermostat_mass`ın literal testi kırıyor.

#### Beşinci kuvvet alanı: ss-σ sıkı-bağ (md ↔ tb kenarı, 24.09.2026)

`engine/md/tb_kuvvet.py::SsSigmaTBKuvveti` `ForceField` sözleşmesinin beşinci
uygulamasıdır ve integratöre de termostatlara da dokunmadan eklendi; senaryo
türü `kind: tb_ss_sigma` (`scenarios/tb_zincir_md.yaml`). TB tarafı (yasa,
Mermin dolumu, model çıpaları) Modül 3F bölümündedir. Döndürdüğü enerji
**Mermin serbest enerjisi Ω = E_band − kT·S + E_rep**'dir ve kuvvet −∇Ω'dır,
E'nin gradyanı değil: integratör KE + Ω'yı korur, **KE + E'yi korumaz**
(fark kT·dS, bir hata değil tanım). **MD'de kT > 0 zorunludur** — kT = 0'da
Fermi seviyesindeki dejenere noktada gradyan yoktur. Sınıf **hartree** alır,
senaryo **Kelvin** taşır; dönüşüm fabrikadadır. **Doğa çıpası yoktur**:
yasalar seçildi, yasadan geri okunan bağ, genlik ve eşik fizik iddiası değildir.

| Test | Pencere | Ölçülen |
|---|---|---|
| KE + Ω salınımı O(dt²) | N = 16, a = 2, L = 32, n_e = 16, kT = 0.002, x'te gevşetilmiş Peierls minimumu + 100 K x-hızları (tohum 7), ω_max·dt ∈ {0.2, 0.1, 0.05}, 1.5 T_min | oranlar **4.016312** ve **4.000702**, tolerans 4 ± 0.1 |
| KE + E korunmuyor, kT·ΔS'de doyuyor | aynı zincir düzgün, **metalik kT = 0.02** (M11 penceresinde kT·ptp(S) yalnız 2.6e-08 ve E ile Ω ayırt edilemiyor) | tt(KE+Ω) oranları **4.0178 / 4.0044**, tt(KE+E) oranları **1.0881 / 1.0215**; ω_max·dt = 0.05'te tt(KE+E)/(kT·ptp S) = **1.0072** |
| Kelvin tuzağı | zincir(16, 2, 0.05, 3), n_e = 16, L = 32, yalnız bant | max\|F\| kT = 0.002'de **1.054e-01**, kT = 300 "hartree"de **2.803e-05** (3759 kat, sessizce) — sınıf kT ≥ 1 hartree'yi reddeder; 1 K altındaki bir Kelvin sayısı ve ters tuzak yakalanmaz (beyanlı) |
| HF'e dokunmuyor | alt süreçte yalnız `import qlab.engine.md.tb_kuvvet`; fabrikadan kurulan TB alanında 20 adım NVE | yüklenen hf modülü **0**; `hf.scf.solve` çağrısı **0** (pozitif kontrol: aynı sayaçla HF alanında 1 adım **14**) |
| Sürüklenme hükmü enerjinin sıfırından bağımsız | zincir(16, 2, 0.05, 3), 631.6 K, L = 32, dt = 15, 267 adım; Ω + c (D2'den önce dt = 20, 200 adım: orada S 4.138e-02, eşiğin yalnız 1.2 katı altı) | eski ölçüt (\|blok[0]\|'a bölen) c = 0'da **1.09e-05** ✓, c = −Ω₀'da **5.35e-02** ✗ (dt = 20'de 3.26e-05 / 9.13e-02); TB ölçütü (max KE'ye bölen) c boyunca bağıl **≤ 5.1e-11**, S **≤ 2.9e-13** aynı; S **2.297e-02** ✓ |
| Eşik TB'ye özgü sürüklenmeyi yakalıyor | aynı zincir, n_e = 15, v₀ = 0, dt = 10, 400 adım | T_e = 3.16 K'de **5.48e-02** ✗ (Fermi seviyesindeki geçiş adımda çözülmüyor), 631.6 K'de **5.97e-04** ✓; eşik **1e-02**. D2'den beri 3.16 K'de S de ✗ (7.30e-02); test sürüklenme hükmünü S'den ayrı okur |
| Salınım hükmü iri dt'yi yakalıyor (D2) | zincir(16, 2, 0.05, 4), n_e = 16, 631.6 K, v₀ = 0, dt = 40, 100 adım (kamu yolu) | save_every 1'de sürüklenme **8.788e-03**, 2'de 1.401e-03 (ikisi de eşik altı; D2 öncesi panel ✓ basıyordu), S = tepe-tepe(KE + Ω)/max(KE) **2.310e-01** ✗; save_every 1/2/5/20'de S bit bit aynı (karelerden okunan S 2.310e-01 → 2.081e-01 kayıyor); sevk edilen senaryoda S **1.0054e-02** ✓; eşik **5e-02** |
| Seyrelmeyi a priori tanı yakalıyor (D2-i) | kütle safsızlığı (15 × 12 amu zincirde tek H, Peierls minimumu + 300 K, dt = 52.35, 57 adım) ve ayrışan iki zincir (T1 zinciri iki kopya, ±v_y, dt = 40, 100 adım); kamu yolu | S **3.629e-02** / **2.100e-02** (eşiğin altında: titreşime katılmayan KE seyreltiyor; D2 paneli ✓ basıyordu) ama kare-0 ω0·dt **0.927** / **0.945** > 0.67 → ✗ (yol kareleri 1.001 / 1.241); aynı pencereler küçük dt'de ✓ (ω0·dt 0.463 / 0.473; yol kareleri 0.500 / 0.628); eşik **0.67** |
| Yol körlüğünü yol kareleri yakalıyor (D2-vi) | Peierls minimumu + 6000 K ısıl hız (dt = 23.02, 130 adım) ve kare 0'da 10 bohr ayrı iki 8'li zincir ±0.004 ile çarpışıyor (L = 64, dt = 21.61, 139 adım); kamu yolu | kare-0 ω0·dt **0.533** / **0.437**, S **4.511e-02** / **3.961e-02**, sürüklenme eşiğin altında (D2-i paneli ✓ basıyordu); izleyicinin seçtiği karelerde ω·dt **0.943** / **0.955** > 0.67 → ✗; dt yarıda ✓ (yol kareleri 0.469 / 0.469); save_every 5'te seçilen adımlar ve ω aynı |

Çağrı maliyeti (bu makine, 1B zincir a = 2, kT = 2e-3, L = 2N, 5 tekrarın en
küçüğü): N = 16'da **0.62 ms**, 128'de **4.31 ms**. `energy_drift` eşiği
iri dt'nin salınımına kördür (M11 minimumunun ω_max'ıyla ω_max·dt ≥ 0.93
olan 33 hücrenin 8'i eşiğin altında kaldı). **D2'den (09.10.2026) beri o yarıyı
salınım hükmü yargılar:** S = [max(KE + Ω) − min(KE + Ω)] / max(KE), uçlar
integratörün **her adımından** (kayıtlı karelerden okunan S save_every'ye göre
hem alta hem üste kayıyordu), eşik **5e-02** (ön ölçümde, her pencerenin kendi
yol boyu ω_max'ıyla, ω·dt ≤ 0.463 sınıfının en büyük S'i 0.0356, ω·dt ≥ 0.925
sınıfının en küçüğü 0.0729 — ölçülen ailelerin değeri, sınır değil: tek
harmonik modda S = x²/(4 − x²) analitiktir, x = ω·dt, ve eşik x = 0.436'ya
düşer); TB NVE'de Yakınsama = sürüklenme **VE** S, ve "Yakınsama tanısı"
satırı ikisinin ayrı hükmünü basar. Koşuyu reddeden bir dt kararlılık
denetimi TB dalında yoktur, dt'yi kullanıcı seçer; seçilen dt'yi koşudan sonra
S yargılar. NVT'de S hesaplanmaz. **D2-i'den (10.10.2026) beri koşudan ÖNCE de
bir tanı hükme girer, D2-vi'den beri koşudan SONRA da:** kare-0 geometrisinin
ve yolun iki karesinin tam 3N kütle ağırlıklı Hessian'ından (kuvvetin merkezi
farkı, 6N kuvvet çağrısı — N = 16'da ≈ 100 MD adımı) en hızlı mod; iki kare,
koşu boyunca alanı saran bir izleyiciyle **her adımdan** seçilir (kayıtlı
karelerden değil): en kısa çift mesafesinin ve kütle ağırlıklı köşegen
sertlik vekilinin tepe yaptığı anlar. Eşik ikisine de ω·dt ≤ **0.67** (ön
ölçümde, D2'nin 1170 hücresi bit bit yeniden üretilerek: küçük-dt sınıfının
en büyük ω·dt'si 0.478, iri sınıfın en küçüğü 0.930, geometrik orta 0.667;
seçilen kareler yolun kareleri olduğu için küçük-dt sınıfında ✗ vermez —
0/552 —, iri sınıfın 302'sinin 302'sini tek başına yakalar; bedeli gri
bölgede S'nin ✓ verdiği 53 hücrenin 4'ünün ✗ olması). Yalnız son kareye ya da
eşit aralıklı karelere bakmak ölçüldü ve reddedildi: kısa süren olayı
(çarpışıp ayrılan parçalar) kaçırıyorlardı; yalnız en kısa mesafe kütleyi
görmüyordu (içinde H olan ağır parçalar). TB NVE'de Yakınsama = sürüklenme
**VE** S **VE** kare-0 ω0·dt **VE** yol karelerinin ω·dt'si; tanı satırı
dördünün ayrı hükmünü basar; bedel en çok üç Hessian ve adım başına ≈ %10
(sevk edilen senaryonun CLI süresi, yorumlayıcı açılışı dahil, 0.76 s'den
0.90 s'ye). **Beyanlı sınırlar:** gri bölgede
(ölçülen ailelerde ω·dt ≈ 0.46–0.93) hükmü tohum, aile ve koşu uzunluğu
değiştirir; büyüme evresindeki ✗ "korunum doğrulanamadı" diye okunur; max KE
yuvarlama düzeyindeyse S gürültüdür. Payda **tüm** kinetik enerji olduğu için
titreşime katılmayan ya da az katılan her KE S'yi seyreltir — **kör aileler**,
liste kapsayıcı değil: ayrışan parçaların ötelemesi (T1 zincirinin iki kopyası
±v_y ile ayrışırken yol ω_max·dt 1.09'da S = 0.021, panel ✓; ötelemesiz ✗),
kütle safsızlığı (15 × 12 amu zincirde tek H, ω·dt = 1.0'da S = 0.036, ✓),
yalnız düşük modun uyarıldığı koşu, yüksek T_ion (ön ölçümde 3000–6000 K),
enine yumuşak modlu geometri, sonradan çarpışan parçalar. S'nin paydası
değişmedi; seyrelmeyi ω·dt tanısı yakalar — ölçülen kör ailelerde (D2
şüphecisi ve doğrulaması, D2-i ve D2-vi eleştirmenleri) D2 panelinin sahte ✓
verdiği 95 iri-dt hücresinin **hepsi** artık ✗ (yalnız kare 0'la, eşik
0.54'te, 40'ı kaçıyordu; dördü kapanış testiyle kilitli). Parça başına COM
paydası da ölçüldü, yalnız öteleme ailesini yakalıyordu; uygulanmadı. **Kalan
körlük** (beyanlı): tanı yolun en çok üç karesine bakar — en hızlı mod bu
karelerde değilse (sertliği çift mesafesinden ya da köşegenden okunmayan,
ör. elektronik ya da toplu bir mod başka bir anda tepe yaparsa) ve S
seyreltilmişse panel ✓ basar; sertlik vekili ss-σ'nın iki üstel yasasına
özgüdür; güvence koşunun kendi yoluna göredir.

**Dürüst olumsuzlar:** termostatlı TB koşusu hiç koşulmadı (panel yalnız NVT'de
eşiğin `None` olduğunu söyler; korunum cümlesi — KE + Ω korunur, KE + E
korunmaz — yalnız NVE'nin beklenen hatasında basılır, NVT panelinde basılmaz);
enine (y, z) kararsızlık ölçülmedi — 1B pencereler y = z = 0 alt-uzayında
kalır; termal bir MD koşusunun Fermi geçişine varıp varmadığı **ölçülmedi**
(G5'in n_e = 15 penceresi v₀ = 0'dan başlar; ön ölçümün yörüngesi tek bir
moda 1.5·ENGEL veriyordu, ≈26600 K eşdeğeri — kurgulanmış bir pencere); N ≥ 48
zincirde birinci derece polaron ön ölçümde görüldü (N = 48/64, r_on = 2.6 /
r_cut = 3.3, N_e ∈ {2, 4}'te 8/114 hücrede q = 1), bu depoda sınanmadı;
kenarın tabanı (`hf → md`'nin A1/A2'sinin karşılığı)
**ölçülmedi** ve kova girdisinin `buyukluk`u bu yüzden `None`dur; makineler
arası tekrarlanabilirlik ölçülmedi.

### Modül 2D — ab initio moleküler dinamik (Plan D)

Modülün asıl kazancı burada: **çıpa artık dairesel değil.** Plan C'nin Morse
alanında `a` parametresi zaten deneysel ω_e'den türetilmişti, yani 4401.21
cm⁻¹ çıkması H2 fiziğini doğrulamıyordu — yalnız formüllerin tutarlılığını.
Plan D'de Hartree-Fock **kendi eğrisini üretir**; hiçbir deneysel sayı girdi
değildir.

| Test | Referans / sınır | Ölçülen |
|---|---|---|
| Kuvvet ↔ bağımsız referans | Richardson ekstrapolasyonu (h=1e-3/5e-4, 4. mertebe), sınır 1e-9 | hata **1.85e-12** (r_e), **2.52e-11** (1.6 bohr), **1.43e-11** (2.0 bohr) — marj 540× / 39.6× / 70× |
| Dengede kuvvet h ile sıfıra gidiyor | `\|F\|` < 5h² (sabit eşik YASAK — h'ye bağlı) | h=1e-3 → **3.06e-07**, h=1e-4 → **3.84e-09**; marj 16.3× / 13.0× |
| Newton III (toplam kuvvet sıfır) | atol 1e-9 | **tam 0.000e+00** (öteleme kısayolu sayesinde) |
| Adım maliyeti | sözleşme 6(N−1)+1 SCF | H2'de **7 SCF**, adım **8.5 ms** (tam alan 13 SCF / 15.9 ms olurdu) |
| BO yüzeyinde enerji SALINIMI | (tepe-tepe/E_titreşim)/(π²/N²) = 1.0833 ± 0.01 | **1.0836** (N=50), 1.0829–1.0836 dört konfigürasyonda; saçılım 0.0007, marj 14× |
| **ω — YOL A (statik)** | enerji eğrisinin ikinci türevi, kuvvet KULLANILMAZ | **5481.25336 cm⁻¹** |
| **ω — YOL B (dinamik)** | MD periyodu, ikinci türev ALINMAZ | **5481.22946 cm⁻¹** |
| **İki yolun uyuşması** | \|ω_B/ω_A − 1\| < 1e-4 | **−4.36e-06** — milyonda 4.4; marj 23× |
| Geometri kapısı | R > 21.1 bohr'da hata fırlatır | E(21.10) = −0.5695574096 → E(21.20) = −0.2057275665, **ΔE = +0.36383 hartree** |
| MD senaryosu CLI'dan uçtan uca | `python -m qlab.run scenarios/h2_aimd.yaml -o h2aimd.npz` | çıkış 0, `energy_drift=1.465e-04` (03.10.2026'dan beri payda max(KE); eskiden \|blok[0]\| ile 8.78e-08), panelde `gradient_step_bohr=1e-05`, `energy_drift_scale=max_kinetic` |

**İki yolun bağımsızlığı iddia değil, ölçüm.** Kuvvetleri ×1.001 ile bozan
bir mutant çalıştırıldı:

| | bozulmamış | kuvvet ×1.001 |
|---|---|---|
| ω_A (statik) | 5481.25336 | **5481.25336** — birebir aynı |
| ω_B (dinamik) | 5481.22946 | 5483.96942 |
| ω_B/ω_A − 1 | −4.36e-06 | **+4.96e-04** → test kırmızı |

ω_A gradyan kodundan **yapısal olarak** etkilenemiyor (yalnız enerjileri
kullanır), ω_B ise ikinci türevi hiç almaz. İkisinin aynı hesap olması
imkânsızdır. Gradyan adımı değiştirildiğinde de aynı desen: h = 5e-5 / 1e-3 /
1e-2 için ω_A üçünde de 5481.25336, ω_B ise 5481.22946 / 5481.22952 /
5481.23607.

**İki düzeltme olmadan bu uyuşma elde edilemez** ve ikisi de ölçüldü.
(a) *Anharmoniklik:* sonlu genlikte frekans düşer — A=0.01 → 5481.24125,
0.02 → 5480.56669, 0.04 → 5477.93574 cm⁻¹. Tek genlikle karşılaştırmak
0.687 cm⁻¹ sahte fark üretir; A² → 0 ekstrapolasyonu şart.
(b) *velocity-Verlet dispersiyonu:* ayrık integratörün frekansı sürekli
olandan yüksektir, Ω = (2/dt)·arcsin(ω dt/2); tersi uygulanır. Etki 0.225 cm⁻¹.

**HF/STO-3G deneyselden %24.54 sapıyor — ve bu bir KUSUR değil, YÖNTEMİN
BEYANIDIR.** Ölçülen 5481.25 cm⁻¹, deneysel ω_e 4401.21 cm⁻¹. Bu depoda
deneysel sayıya yaklaşma hedefi ya da toleransı **yoktur**. Sapmanın
yöntemden geldiğinin bağımsız kanıtı baz setiyle gelir: 6-31G ile sapma
**+%5.57**'ye iner, yani hata 4.4 kat azalır — bir kodlama hatası baz
setiyle böyle davranmazdı.

**432× payda tuzağı — bir sonraki okuyucu için.** BO yüzeyinde toplam enerji
≈ −1.1175 hartree, titreşim enerjisi ise 2.582381e-03; oran **431.75**. Plan
C'nin enerji testleri paydaya `|E_toplam|` yazar. O satır buraya kopyalanırsa
her enerji ölçütü 432 kat zayıflar. Payda **E_titreşim**'dir.

**Sürüklenme bir kabul ölçütü DEĞİLDİR** ve bu ölçülerek anlaşıldı. Planın
ilk hâli `sürüklenme < 1e-8` yazıyordu; ölçülen değer **9.42e-05** (N=50) ve
**1.94e-05** (N=100) — yani dt ile ölçekleniyor, bir gürültü tabanı değil
ayrıklaştırma eseri. O sınır 9421 kat dışarıda kalırdı. Ölçüt salınım
oranıdır; salınım temiz π²/N² yasasına uyuyor (N ikiye katlanınca 4.0016).

**Dürüst olumsuzlar (ölçüldü, gizlenmiyor):**

- **Langevin 300 K'ya OTURMADI.** 200 adımda ⟨T⟩ = 218.4 K, std 106.1 K. Bu
  bir kod hatası değil: tek titreşim serbestliği olan iki atomlu sistem
  ergodik değil ve 200 adım istatistik için yetersiz. **Termostatın HF ile
  KOŞTUĞU kanıtlandı; hedef sıcaklığa OTURDUĞU kanıtlanmadı** ve iddia
  edilmiyor. (Nosé-Hoover'da ⟨T⟩ = 289.2 K ama T_max 974.3 K — aynı uyarı.)
- **"Isı bağı kırar" hikâyesi RHF'e TAŞINMAZ.** Ölçüldü: RHF/STO-3G ayrışma
  limiti −0.5709 hartree, doğrusu −1.000000. Plan C'nin Morse ayrışması nicel
  bir iddiaydı; RHF'te aynı iddia kurulamaz.
- **Varsayılan gradyan adımını hiçbir FİZİK testi korumuyor.** Ölçüldü:
  h=1e-3 yapan mutant bütün AIMD testlerini geçiyor (salınım oranı 1.0891,
  sınırın içinde) oysa gradyan hatası 1.34e-07 — 5e-5'tekinin 10⁴ katı. Hata
  sistematik ve kuvvete göre küçük olduğu için dinamik onu görmüyor. Bu
  yüzden varsayılanı **doğrudan kilitleyen** ayrı bir test var.
- **Analitik HF gradyanı (Pulay kuvvetleri) ÖLÇÜLEMEDİ** — bu depoda yok.
  "Analitik gradyan 6N+1 SCF'yi 1'e indirir" bir tahmindir ve buraya
  yazılmıyor. Aynı şekilde **sıcak başlangıç** (önceki yoğunluğu tohum verme)
  kazancı da ölçülemedi: `scf.solve` imzasında başlangıç yoğunluğu yok.
- **Maliyet kapsamı dar.** Ölçüldü: H2 1000 adım ~0.14 dk, H2O ~15.4 dk,
  CH4 ~77 dk. Test süitesine **yalnız H2** girer.

**Mimari sözü tutuldu ve git ile ölçüldü.** Plan D dördüncü kuvvet alanını
ekledi; `engine/md/integrator.py`, `engine/md/thermostats.py` ve
`observable/md_obs.py` üçlüsünde **0 satır** değişti (`git diff --stat
82c65c4..HEAD` boş). Ama iddia dardır: `engine/md/factory.py` değişmek
**zorundaydı** — dağıtım kapalı bir sayımdı ve `_YAKLASIMLAR` listesinin üç
satırından ikisi ("elektron yok", "analitik bir modeldir") HF dalında düz
yalan olurdu ve şeffaflık panelinde basılırdı.

**Ayrı plana bırakılan — zamana bağlı yoğunluk filmi.** Plan B2'nin kapanış
notu bunu istiyordu ama ölçüm onu Plan D'den ayırdı: bir karenin maliyeti
ızgarada baz değerlendirmesinden geliyor, kuvvet alanından değil; yörünge için
**sıfır SCF** gerekiyor. Yani film ab initio kuvvet gerektirmiyor. Modül 2E'de
yapıldı ve oradaki uçtan uca ölçüm bunu doğruladı: kare maliyetinin ~%64'ü
yoğunluk alanı, %31–35'i izoyüzey, SCF **%6'nın altında** ve n büyüdükçe %1'in
altına iniyor.

### Modül 2E — zamana bağlı yoğunluk farkı (Plan E)

Plan D'nin kapanışı bu filmi kapsam dışı bırakırken bir ölçüme dayanmıştı:
maliyet kuvvet alanından değil ızgaradan geliyor, dolayısıyla film ab initio
kuvvet **gerektirmiyor**. Plan E o ayrımı doğruladı ve filmin asıl sorusunu
değiştirdi: mesele kareleri üretmek değil, **hangi sayının bağın zayıflamasını
ölçtüğü**.

| Test | Referans / sınır | Ölçülen |
|---|---|---|
| Orta nokta değeri çözünürlükten bağımsız | n = 41 / 61 / 81, tolerans 1e-6 | üçünde de **0.043821** — altı hanede aynı |
| …ve kutu boyutundan da bağımsız | yarı genişlik 6.0 vs 7.0 | **0.043821** (7.0 / n=51'de de) |
| Gerilme serisi tek yönlü çöküyor | monoton azalan | 1.3459 → **0.043821**, 1.8 → **0.039467**, 2.5 → **0.028001**, 4.0 → **0.006847** |
| Kopuş serisi (sevk edilen modül) | sayısal sıfıra oturmalı | r = 1.3459 → 0.043821, 4.1649 → 0.005686, 6.2559 → **0.000541**, 8.2587 → 0.000044, 10.2504 → 0.000002, 12.24 ve ötesi → **0.000000** |
| Çift ızgara reddediliyor | `InvalidInputError` | n = 40 / 60 / 80 → 0.033640 / 0.038969 / 0.041031, yani **%23 / %11 / %6 eksik** |
| Kutu tüm kareler boyunca sabit | `origin` ve `shape` değişmez | `[-8,-8,-8]`, `(41,41,41)` |
| Kayan kütle merkezi kutuya taşınıyor | ağ ağırlık merkezi orijinde | yörünge COM'u 5 karede **1.92 bohr** kayarken ağ merkezi max \|z\| ≈ **9e-09** |
| Yük alan küpü taşımıyor | hiçbir dizi `ndim > 2`, toplam < 300 KB | 4 karede **36.792 bayt** (küple 2.205.472 → **60×**) |
| Alan kare başına bir kez hesaplanıyor | çağrı sayısı = kare sayısı | **4 / 4** (düzeltmeden önce 8) |

**Neden `max|fark|` değil.** En doğal görünen aday — fark haritasının mutlak
maksimumu — bir bağ gözlemlenebiliri **değildir**, ve bu iddia iki bağımsız
ölçümle kilitlenmiştir:

| | r = 1.3459 | r = 1.8 |
|---|---|---|
| bağ orta noktası | 0.043821 | **0.039467** (düştü) |
| `max\|fark\|` (n=61) | 0.064559 | **0.073424** (YÜKSELDİ) |

Yön yanlış olmakla kalmıyor, **değer ızgaraya bağlı**: aynı iki mesafede
`max|fark|` n = 41 / 61 / 81'de 0.061516 / 0.064559 / 0.063294 — n ile tek
yönlü bile değil. Orta nokta üç ızgarada da 0.043821'dir. Sebep fiziksel:
mutlak maksimum çekirdek yakınındaki keskin yapıdadır, bağ bölgesini hiç
görmez, ve kovaladığı çekirdek sivriliği ızgara inceldikçe sınırsız
keskinleşir. **Çözünürlüğe bağlı bir büyüklük gözlemlenebilir değildir.**

Asimetrik bağda ayrım daha da sert. HF molekülünde orta nokta **negatiftir**
(−0.194841, n = 41/61/81'in üçünde de altı hane sabit) — iyonik bağda orta
nokta zayıflar, yoğunluk F'ye kaçar — oysa `max` **+0.235735**, işaretleri
bile zıttır. Simetrik H2 bu ayrımı yapamaz: ölçüldü, orada işaretli maksimum
**tam olarak** bağ orta noktasındadır (argmax indisi (30,30,30), fark
0.00e+00, dört germe mesafesinin dördünde de).

**Izgara paritesi tuzağı.** Bağ orta noktası ancak `n` **tekse** gerçek bir
ızgara düğümüdür. Çift `n`'de en yakın düğüm okunur ve değer sessizce eksik
çıkar: n=40'ta %23, n=60'ta %11, n=80'de %6. Tuzağın asıl tehlikesi sayının
büyüklüğü değil **şekli**: hata n büyüdükçe küçüldüğü için bir *yakınsama
sorunu gibi görünür* ve insanı ızgarayı büyütmeye iter — maliyet n³ ile artar.
Oysa doğru düzeltme `n`'i tek yapmaktır ve bedavadır. Bu yüzden `symmetric_box`
çift `n`'i sessizce kabul etmez, hata fırlatır.

**Bellek kararı: ağ sakla, alan saklama.** (M,n,n,n) float64 bir küp M=100 /
n=80 için **0.410 GB** tutar; bir film için kabul edilemez. İzoyüzey ağları
ise kare başına yalnız **50–410 tepe**. Aynı 9 karenin ağ yükü 29.5 / 78.3 /
142.0 KB, alan küpleri 4.96 / 16.3 / 38.3 MB — **168× / 209× / 269×**. Alan
üretilir, ağ çıkarılır, alan atılır. Bu bir yorum değil test edilen bir
sözleşmedir: yükteki hiçbir dizi `ndim > 2` olamaz.

**Kare maliyeti** (uçtan uca, ölçüldü): n=41 → 21.6 ms, n=61 → 63.1 ms,
n=81 → 141.5 ms; 100 karelik film 2.16 / 6.31 / 14.15 s. Dağılım: yoğunluk
alanı ~%64, izoyüzey %31–35, SCF %6'dan az ve n büyüdükçe %1'in altına iner.
**Pahalı olan elektronik yapıyı çözmek değil, onu örneklemektir** — Plan D'nin
"film ab initio kuvvet gerektirmiyor" hükmünün sayısal karşılığı budur.

#### Dürüst olumsuzlar

- **RHF'in enerjisi ayrışmada felakettir, yoğunluğu doğrudur.** Ölçüldü:
  D_e = E(20 bohr) − E(r_e) = −0.570861 − (−1.117506) = **0.546645 hartree =
  14.875 eV**, deneysel değer **4.75 eV** — üç katından fazla. Doğru ayrışma
  limiti iki H atomu için −1.000000 hartree'dir; RHF 10 bohr'da −0.595971
  verir. Sebep yapısaldır: tek determinant kısıtlı dalga fonksiyonu H2'yi
  doğru ayrıştıramaz. Buna karşın *yoğunluk farkı* orta noktada doğru
  davranır ve sayısal sıfıra oturur. **Bu yüzden filmden enerji okunmaz** —
  film bir yoğunluk anlatısıdır, bir enerjetik iddia değil.
- **Video/görüntü üretimi kapsam dışıdır** ve bu bir eksiklik değil bir
  sınır. Deponun `viz/` katmanı hep *yük* üretir (`frames.py`,
  `md_frames.py`); `.mp4`/PNG yazmak yeni bir bağımlılık ve yeni bir katman
  demektir. `build_density_film` kare dizisini verir, onu neyin çizeceği
  çağıranın işidir.
- **H2 üstü ölçülmedi.** Film yalnız iki atomlu bir bağ için sınandı;
  `bond_midpoint_difference` iki atom indisi alır ve çok-bağlı bir molekülde
  hangi bağın anlatıldığı çağıranın kararıdır. H2O ve üstü için ne maliyet
  ne doğruluk ölçülmüştür.
- **Dış kararlılık analizi bu yolda koşulmaz** (`stability=False`). RHF
  çözümünün kısıtlamayı kaldırınca kararlı kalıp kalmadığı — ki gerilmiş H2'de
  kalmaz — sorulmamıştır.
- **Film ab initio kuvvet gerektirmiyor.** Buradaki yörüngeler Morse alanıyla
  üretildi ve kare başına SCF maliyeti %6'nın altında. Yani "kopuş filmi"
  görüntüsü, kuvvetlerin Hartree-Fock'tan geldiğini **kanıtlamaz**; onu
  kanıtlayan Modül 2D'nin çıpasıdır.

### Modül 3F — tight-binding çekirdeği (Plan F)

Bu modülün en önemli çıktısı bir sayı değil bir **hüküm**: yol haritasının
Modül 3 için ilan ettiği doğrulama hedefinin ilk yarısı — "bakır/silisyum bant
aralığı" — ölçümle **geçersiz kılındı**. Bir tight-binding modelinde bant
aralığının *değeri* bir tahmin değil, girilen fit parametresinin geri
okunmasıdır.

| Test | Referans / sınır | Ölçülen |
|---|---|---|
| **Gerçek-uzay ↔ Bloch çapraz kontrolü** (çekirdek çıpa) | hata/‖H‖ < 1e-13, N = 4…256 | N=12'de **8.882e-16**; hata √N ile büyüyor (log-log eğim **0.5025**), hata/‖H‖ N=4→4096 boyunca 3.165e-15 |
| Ters örgü konvansiyonu | b_i·a_j = 2π δ_ij, atol 1e-12 | eğik örgüde artık **1.178e-16** (yanlış sürümde 2.173) |
| Kesme yarıçapı, fcc | 12 komşu, 11 örgü sabitinde | **11/11** (bağıl toleranssız 5'inde **0** komşu seçiliyordu) |
| `eigh` UPLO tuzağı | üst üçgene +1000 | özdeğer değişimi **tam 0.000e+00** |
| Gauge (hücre vs atom) | özdeğerler eşit kalmalı | H elemanları farkı > 1e-3, özdeğerler 1e-14'te aynı |
| Faz konvansiyonu | H[0,1] işaretli | **−0.10362686 + 0.03983523j** (ters işarette özdeğerler **aynı**) |
| **Tek elektron sayımı → metal** | parametre-bağımsız | **300/300** (1B/2B/3B × 100 rastgele model) |
| Yapı oranı W²/μ₂ = 4z | ε ∈ (−500,500), t ∈ (−50,50) | 1B **8.000000000** (2.22e-16), 2B **16**, 3B **24** |
| Bant genişliği, tek n | kapalı form 2\|t\|(1+cos(π/n)) | fark **2.78e-17**; oranlar 0.750000 / 0.950484 / 0.991487 / 0.999758 |
| Ağırlıklar gauge-bağımsız | özvektörler değil | özvektör farkı **0.814887**, ağırlık farkı **3.331e-16** |
| Şeffaflık paneli daireselliği beyan ediyor | tam cümle | CLI çıktısında doğrulandı |
| Süite | — | Plan F kapanışında **801 test yeşil**, `tests/tb/` 80 test / 4.19 s (o günün sayısı; bugünkü toplam için yukarıdaki süite tablosuna bakın) |

#### Bu modül bant aralığı TAHMİN ETMEZ

Vogl-Hjalmarson-Dow (1983) sp³s\* parametrizasyonuyla silisyum kuruldu ve
bağımsız bir uygulamayla yeniden üretildi:

| ölçüm | değer |
|---|---|
| dolaylı bant aralığı | 1.171338 eV (deneysel 1.12) |
| E_gap = 1.17 eV için gereken `V(s*a,pc)` | **5.376670 eV** — yayınlanan 5.3749, fark %0.033 |
| kazanç dE_gap/dV | −0.7561 eV/eV |
| aynı kadran | Ge'yi (0.66 eV), GaAs'ı ve V=7.0'da **metali** veriyor |
| Γ noktası özdeğerleri | analitik olarak E_s ± \|V_ss\| — **doğrudan girdiler** |

Aynı şey 1B'de doğrudan gösterildi: hedef aralık seçilip `t` ondan cebirsel
türetildiğinde çözücü hedefi geri veriyor (1.12 → 1.120000000000000, 0 ULP).

**Fit edilmeyen tek nicelik yanlış çıkıyor:** iletim bandı minimumunun konumu
**0.731066 X**, deneysel **0.85 X** — **%14.0 hata**. Model onu tahmin etmeye
çalıştı ve ıskaladı. Doğrulanabilir olan tek şey buydu ve doğrulanmadı.

#### İletken yarısı sayımdan, yalıtkan yarısı fit'ten

Modülün ölçülmüş asimetrisi budur. **Tek** elektron sayımı her zaman metal
verir — bir bandı ancak yarı doldurabilir, dolayısıyla hüküm bir hesap değil
bir **sayımdır** (300/300, iki bağımsız uygulamayla). Ama **çift** sayım
yalıtkanlığı **garanti etmez**: bant örtüşmesi yüzünden metal çıkma oranı
1B %67, 2B %34, 3B %20 ölçüldü. Silisyumun yarıiletkenliği sayımdan çıkmıyor.

**Panel hükmü 04.10.2026'dan beri kanıtlıdır (E13).** Izgaranın kendi hükmü
(`is_metal`) iki yerde yanlış "yalıtkan" diyordu. Bunu kamu yolundan ölçtük:
- Grafen **her** `k_points`'te "yalıtkan ✓" çıkıyordu. Dirac noktası ızgarada
  yoksa aralık sahte; ızgaradaysa dokunan bantlar yuvarlamayla +5e-17 ayrık
  çıkıyor ve `>=` bunu aralık sayıyor.
- İki orbitalli bir metal k = 4 ve 16'da "yalıtkan ✓" çıkıyordu.

Panel artık `certified_conduction`'ı basar:
- **metal**: yalnız ızgarada tanık varken (bant örtüşmesi ya da tek sayımda
  yuvarlamadan geniş yarı dolu bant). Tek sayımda yarı dolu bant ızgarada
  düz görünürse hüküm "belirsiz"dir.
- **yalıtkan**: yalnız ızgara aralığı 2·B1'i aşarken, bütün bantlar doluyken
  ya da H(k) k'dan bağımsızken. B1 = (π/n)·max_i Σ|t|·‖cell‖₁, Weyl sınırı.
- **belirsiz**: geri kalan her durum. Gerçek aralık [g − 2·B1, g] içindedir;
  metal, yarı metal ya da dar aralıklı yalıtkan olabilir.

Yuvarlama payı 1e3·eps·(|E|max + max_i(|ε_i| + Σ|t|(1 + 2π‖cell‖₁))).
İlk sürümde pay yalnız |E|max'a bağlıydı. Doğrulama turu bu yüzden kamu
yolundan iki sahte "metal (tanıklı)" üretti: birbirini götüren parçalı
hoplama (1.23 + 1.6 − 2.83 eV) ve |R| = 1000 hücrede faz yuvarlaması.

Dokunan bantlar (yarı metal) hiçbir n'de metal kanıtlanamaz; grafen her
ızgarada "belirsiz"dir. Ön ölçüm ajanlarının kurduğu ~880 karşı örnekte sahte
"yalıtkan" sıfır. "yalıtkan" denen 23 dosyanın 23'ü 200 000 noktalık ızgarada
kanıtlı. İki bayrak birbirinden bağımsızdır: grafen k=100'de kenarlar kanıtlı
(✓) ama hüküm "belirsiz". npz'de `iletkenlik` tam hükmü taşır; `is_metal`
yalnız kanıtlı metalde True'dur ve False "yalıtkan" demek **değildir**.

**Hoplama listesi Hermityen olmalıdır (H0, 04.10.2026).** Eskiden tek yönlü
yazılmış bir zincir rc=0 ile "metal ✓" basıyor ve bant genişliği yarıya
iniyordu (3.4 eV, doğrusu 6.8), çünkü `eigvalsh` yalnız alt üçgeni okur.
`TBModel` artık bunu reddediyor. Tolerans ölçeğe göre 1e-12; testlerin
kurduğu 1904 modelde en büyük yuvarlama sapması 7.22e-16.

#### Tautoloji-çıpa — bu depoda üçüncü kez

Ön ölçümde "çıpa" ilan edilenlerin çoğu karşıt doğrulamada düştü:

| iddia edilen çıpa | hüküm |
|---|---|
| "sayısal = analitik, 1 ULP" | **tautoloji** — `eigvalsh(1×1)` ile `H[0,0].real` arası 0 ULP; 1 ULP'in tamamı birleşme sırası |
| "bant katlanması, 0 ULP" | **tautoloji** — iki matris bit bit aynı |
| "durum sayımı" | 6 kasıtlı bozulmanın **0'ını** yakaladı |
| grafende K'de sıfır aralık "korunuyor" | **çürütüldü** — Haldane terimi A,B özdeş kalırken 0.519615242 eV açıyor |
| `E(Γ)/E(M) = 3` gibi oranlar | model kesiminin geri okunması |

Körlük doğrudan sergilendi ve testle kilitlendi: tek orbitalli zincirde
`eigvalsh` 1×1 matriste **yalnız gerçek kısmı** okur, dolayısıyla üç farklı
Hamiltoniyen aynı özdeğeri verir — doğrusu (−0.17251924+0j), hücre çarpanı
düşürülmüş olan (**−0.17251924 − 0.12299824j**, Hermityen bile değil) ve fazı
tamamen kaybetmiş olan; üçünün de `eigvalsh`'i −0.172519243044.

**Ayırıcı sınav ölçüldü:** aynı altı mutanta karşı çekirdek çıpa **6/6**,
tautoloji-çıpa **1/6** yakaladı.

> **Kural.** 1e-12 altında uyuşan iki "bağımsız yol" kanıt değil şüphedir.
> Sahte-çıpa sınavı (fiziğe ait olmayan rastgele girdi) şüpheyi *işaret eder*
> ama tek başına ayırmaz — **ayıran şey mutasyondur**: bir yolu bozunca öteki
> yakalıyor mu?

#### k-uzayı parite tuzağı — Modül 2E'nin kardeşi, ama daha kötü

Grafende ölçüldü: 601×601 ızgara Dirac noktasını yakalıyor (**2.471886e-15 eV**),
600×600 sahte bir **1.635969e-02 eV** aralık uyduruyor — ve **daha sık** olan
1000×1000 **1.698157e-02 eV**, yani 601'den **13 mertebe kötü**. Sıklaştırmak
düzeltmiyor.

1B'de iki ayrı biçimde çıkıyor. Birincisi parite: bant genişliği tek n'de
sistematik eksik, çift n'de tam. İkincisi daha sinsi — **hata monoton bile
değil**:

| n | bant minimumu hatası |
|---|---|
| 5 | **2.426e-06** |
| 7 | 1.934e-02 |
| 12 | 7.031e-03 |
| 13 | **7.370e-03** (n=12'den kötü) |
| 2001 | 1.936e-09 |
| 20001 | **3.849e-09** (on kat sık ızgara iki kat kötü) |

n=5, n=13'ten ~3000 kat iyi. Bu yüzden `band_edges` uçları ızgaradan
**okumuyor**: menzille ölçeklenen kaba tohum ızgarası yalnız başlangıç seçiyor,
ucu sürekli arama (L-BFGS-B + kompas cilası) buluyor. (Eskiden burada
"201×201 ızgaradan hem daha doğru hem 8× hızlı" yazıyordu; o tek modelin
minimumu (0, 1/2) TRIM noktasındaydı ve her çift ızgarada vardı, 201 ise
tek — yani karşılaştırma parite tuzağının yan ürünüydü.)

Parite kodda **sihirli sayıyla** sabitlenmiyor: ön ölçümde iki ajan iki farklı
ızgara konvansiyonunda iki farklı "doğru N" ölçtü (601 ve N%3==0), ikisi de
kendi konvansiyonunda doğru. Bunun yerine çağıran hangi k noktalarının ızgarada
bulunması gerektiğini söylüyor, kod **denetliyor** (`must_include`).

#### Dürüst olumsuzlar

- **`band_structure` çıktısı sıralamadır, bant kimliği değildir.** Ölçüldü
  (iki bağımsız zincir): kesişimin 0.05 bohr⁻¹ sağında sıralı "bant 0"ın
  etkin kütlesi **+1.6735** (öbür zincirinki), gerçek zincir 0'ınki
  **−0.6276**. İşaret bile ters ve sonlu fark adımından (1e-3…1e-5)
  bağımsız. Eski "12457 kat" sayısı **yeniden üretilemedi**: kesişimin tam
  üzerinde sıralı kütle adım h ile ölçeklenir (oran 1/h), yani sayı büyük
  olasılıkla orada alınmıştı; ama eski "−0.800" bu modelin kesişimdeki
  −0.859'uyla tutmuyor (04.10.2026, E3). Çizim için doğru, bant-çözümlü
  nicelikler için değil. Depo bant
  çözümlü nicelik (etkin kütle vb.) **üretmez**. Bant takibi bilerek
  eklenmedi: ön ölçümde kaçınılmaz kesişimde bulunan "kimlik" adım
  yoğunluğuna bağlıydı.
- **Dejenere alt uzayda bant başına orbital ağırlığı tanımlı değildir.**
  Doğrudan sergilendi: 2B'de iki bant tam dejenere (E farkı 0.0); numpy
  `[1,0]/[0,1]` döndürüyor ama 45° döndürülmüş çift de aynı ölçüde geçerli ve
  ağırlıkları `[0.5,0.5]`. Alt uzay toplamı iki tabanda da aynı. **E3a,
  04.10.2026:** bant yükü (ve npz) dejenere kümede artık **alt uzay
  ortalamasını** yazar ve `degenerate` bayrağıyla işaretler. "Dejenere"
  ardışık özdeğerlerin 1e3·eps·(|E|max + H ölçeği) içinde kalması demektir;
  kümeleme geçişlidir. Süreksizlik yok olmaz, ölçeği küçülür: eşiğin hemen
  altı ve üstü yine farklı ağırlık verir. Sabit 1e-8 Ha eşiği (ön ölçüm)
  bu süreksizliği fiziksel olarak çözülebilen ayrışmalara taşıyordu.
  Eşiğin içinde `eigh`in ağırlığı ~1e-3 doğrulukla belirli olabilir; yine
  de bilerek ortalanır (muhafazakâr seçim). Bağımsız doğrulamada eigh'in
  gerçek ayrışma kırıntısı en çok 8e-3·eşik çıktı (240 orbital, |E| 300 Ha).
- **`band_edges` bulunan en iyi uç değerdir, kanıtlanmış küresel uç değer
  değil; ama dönen aralık HER ZAMAN gerçek bandın İÇİNDEDİR** (her değer
  gerçek bir k'daki gerçek özdeğer: min üst sınır, max alt sınır). Eski
  sürüm (8×8 başlangıç, eksen-eksen iniş) ölçüldü: 1e-9 eşiğinde orijinal
  2B kümede 14/50, 3B'de 8/30, grafende 24/48, uzun menzilli eğik 2B'de
  39/40 arama başarısız, en kötüsü bant genişliğinin **%47**'si; grafende
  0.24–0.60 eV sahte aralık. Başlangıçların 7/8'i ölü koddu (bounded Brent
  ilk ekseni okumuyor). **E2, 04.10.2026:** eksen başına 8·R_a
  noktalı (R_a: o eksendeki sıfır olmayan hoplama menzili) Γ-merkezli
  tohum ızgarası, en iyi 3 yerel uçtan sarılı L-BFGS-B ve gürültü eşikli
  kompas cilası; H(f) kesirli koordinatta kuruluyor. Eski (1e-9) / yeni
  (1e-12): orijinal 2B 14/50 / 0/50, 3B 8/30 / 0/30, grafen 24/48 / 0/48,
  PT simetrik iki orbitalli 20/40 / 0/40, uzun menzilli 2B 39/40 / 0/40,
  eşit genlikli "pürüzlü" 1B (menzil 6) 133/298 / 0/298, pürüzlü 2B
  153/160 / 0/160, pürüzlü 3B 19/20 / 0/20; eskinin 0.94× ile 176× arası
  hızında (3B kümede 72×). Kesin olan yön, yuvarlama payı içinde: dönen
  aralık gerçek bandın içindedir, örgü tabanının koşulundan bağımsız.
  Bağımsız doğrulama, eski k·R kurulumunun indirgenmemiş tabanda gerçek
  minimumun 1.9e-8 W altına indiğini gösterdi; bu yüzden kesirli kurucu
  eklendi. **Küresel uç kanıtlanmaz** ve bilinen başarısızlık sınıfları
  var:
  - 2B/3B'de kesişim dikişi olan bantların iç kenarları %6–10
    kaçabiliyor. Uç tohumda yerel uç olarak görünmüyor. Bu durum testle
    kilitli bir dürüst olumsuzdur.
  - Zayıf eşleşmeli kaçınılmış kesişimin dar havzası %7.5'e kadar
    kaçabiliyor.
  Kanıtlı bilgi gerekiyorsa B1 / `certified_conduction` kullanılmalı.
  Ölçümle seçilen ya da atılan parçalar:
  - Kompassız grafen 20/48 başarısızdı.
  - Γ-merkezli sabit 8 tohum pürüzlü 1B'de 22/149 modeli bozuyordu.
  - Tek aday PT'de %5 kaçırıyordu.
  - Ön ölçümün 0.381966 kaydırması dar bir tepeyi kaçırdığı için
    kaldırıldı.
  Hepsi testle kilitli. (İlk ölçümdeki "tek başlangıç 31/50, 5×5 8/50"
  sayıları kaynak kümesiyle birebir yeniden üretildi.)
- **`converged` eskiden kandırılıyordu; 04.10.2026'dan beri KANITLI (E1).**
  n vs 2n karşılaştırması n'nin **hatasını** değil n ile 2n arasındaki
  **iyileşmeyi** ölçüyordu. İki ızgara gerçek uca en yakın aynı noktayı
  paylaşınca fark tam 0 çıkıyordu. Ölçüldü: ucu kesirli 0.26'da olan ikinci
  komşulu zincir n=4 ve 12'de ✓ alıyordu, gerçek uç hatası 7.885e-04 Ha
  (eşiğin 788 katı). Zincir n=4'teki ✓ ise doğruydu, uçlar ızgaradaydı, ama
  bir kanıt değildi. Artık `converged` = B2 ≤ 1e-4 Ha
  (Osman'ın eşiği). B2 = (π²/2n²)·max_i Σ|t|·‖cell‖₁², ızgaranın spektrum
  uçlarını en çok ne kadar kaçırabileceğinin **kanıtıdır** (Rayleigh bölümü;
  zincirde sıkı, oran → 1). Sevk edilen zincir n=100'de 6.17e-05 ✓, n=4'te ✗.
  Köşegenleştirme gerekmediği için `build_tb` artık iki ızgara hesaplamıyor.
- **Ortogonal olmayan taban yoktur** (S = I). Bedeli ölçüldü: bant genişliğinin
  %45–65'i.
- **`viz` katmanının GUI yasağı YOKTUR** — ölçüldü, `viz/band_frames.py →
  PySide6` rc=0 ile geçiyor ve `viz/isosurface.py:55` zaten pyqtgraph import
  ediyor. "Bant yükü çizim yapmaz" bir **disiplindir, sözleşme değildir**.
  (`engine` tarafındaki `matplotlib` kaçış deliği ise bu modülde kapatıldı:
  enjeksiyon artık rc=1 veriyor ve taranan dosya 67 → 68'e çıkıyor.)
- **Büyük k'de argument reduction sessizce bozuluyor:** k + 1e12·G →
  özdeğer farkı 2.387652e-03 eV mertebesinde, hiçbir uyarı yok.
- **Bu ULP sayıları taşınabilir değildir** — tek makine, tek yığın
  (Python 3.14.7, numpy 2.5.1, scipy 1.18.0, Windows 11).

#### Ne yapılmadı

Silisyum ve bakır **sevk edilmedi** (ön ölçümde kuruldular, depoya girmediler);
spin, spin-yörünge etkileşimi ve manyetik alan yok; Peierls çarpılmasının
enerjetiği hesaplanmıyor; sıcaklık/Fermi-Dirac doluluğu yok. Ve en önemlisi:
**parametreler türetilmiyor**, dışarıdan giriyor. Onları qlab'ın kendi
Hartree-Fock'undan türetmek — daireselliği kırma hamlesinin TB karşılığı —
Plan G'ye bırakıldı ve ön ölçümü "KISMEN" dedi.
*(Plan F kapanışının kaydı. O günden bu yana üç madde değişti: (1) parametre
türetimi Modül 3H'de geldi — H halkalarında UHF indirgemesi (`downfold_uhf`,
`model_from_downfold`, `fit_distance_laws`); Plan F'nin kendi malzemeleri için
türetim yok; ss-σ kenarı ise BİLEREK seçilmiş bir yasa taşır ve downfold yasası
orada yasaktır (M10 `ast` kilidi). (2)–(3) 24.09.2026 itibarıyla: Fermi-Dirac
doluluğu `tb/mermin.py`de var ve 1B zincirde Peierls/Kohn enerjetiği model
çıpası olarak sınanıyor — aşağıdaki ss-σ bölümü. Spin ve silisyum/bakır hâlâ
yok.)*

#### md ↔ tb ss-σ kenarı: mesafeye bağlı hoplama ve Mermin dolumu (24.09.2026)

Modül 7'nin `OLCULMEDI` kovasındaki `md-tb` çifti **kenarı kurularak**
`GERCEK_KOPRU`ya geçti (spec `.superpowers/sdd/md-tb-ss-sigma-spec-2026-09-23.md`;
MD tarafı Modül 2 bölümünde). TB tarafında üç parça: `tb/radyal.py` (üstel
radyal yasa + iptalsiz sin² kuyruğu; aynı biçim hoplama rolünde `ss_sigma`,
itme rolünde `itme` adını taşır), `tb/mermin.py` (Mermin dolumu, g_s = 2;
enerji Ω = E_band − kT·S + E_rep, kuvvet −∇Ω) ve `hamiltonian.distance_model`
(kafes yolu; kesme yasadan, `max_cell` ters örgüden türetilir). **Atom başına
tek s orbitali: Slater-Koster tablosunun yalnız ss-σ satırıdır, açı çarpanı
özdeş 1 — bu bir SK modeli değildir**; p/d orbitalleri yok. Parametreler
**seçimdir ve bu kenarda doğa çıpası yoktur**: aşağıdaki MODEL ÇIPALARI modelin
kendi kuramına karşı sınanır, doğaya karşı değil. Ortak pencere: hoplama yasası
−0.1 / d0 = 2 / üs 2, itme 0.05 / 2 / 6, r_on = 3.0, r_cut = 3.8 bohr.

| Test | Pencere | Ölçülen |
|---|---|---|
| F = −∇Ω, 3N bileşenin hepsi (makine) | 3B küme N = 8 kutusuz / bozulmuş periyodik zincir N = 16, L = 32; kT = 1e-3, merkezi h = 1e-5, her çift dikişten ≥ 1e-3 uzak | max\|F + ∇Ω\| **1.32e-09 / 7.78e-11**, tolerans 1e-8 |
| Kesme merdiveni kapanıyor | dimer, n_e = 2, kT = 1e-3; ΔE(ε) = Ω(r_cut − ε) − Ω(r_cut + ε) | ΔE(1e-3)/ΔE(1e-4) = **100.0389** (basamak ε² ile kapanıyor); sert kesmeli `uniform_model` aynı dimerde (40 bohr kutu, Γ, kT = 0) sabit **0.2** sıçrar — 2\|t\|, girdiden geri okuma |
| Ω−E farkının işareti | zincir(16, 2, 0.05, 3), n_e = 16, L = 32, x = gap/kT ∈ {0.2, 1, 3.09, 10, 20} | (F − F_FD(E)) / (−P(x)·∇gap) = 1, en kötü sapma **6.1e-07**: smearing boşluğu **kapatan** yöne iter |
| Kafes yolu ≡ MD yolu, aynı yasa nesnesi | zincir(16, 1.8, 0.05, {3, 5}), L = 28.8, kT = 2e-3 | \|ΔΩ\| ≤ **2.2e-16**, max\|ΔF\| ≤ **2.6e-16**, tolerans 1e-14 |
| `max_cell` ters örgüden | 15° eğik 2B örgü, kesme 0.79 | naif formül **4** komşu, ters örgü **6**, bağımsız referans **6** |
| Parçacık-boşluk simetrisi (model çıpası) | iki parçalı zincir(16, a, …), N_e = 8 ↔ 24, kT = 2e-3, dört geometri | \|ΔΩ\| ≤ **8.9e-16**, max\|ΔF\| ≤ **9.2e-16** |
| Kohn argmin = q_F (model çıpası) | düzgün N = 16, 15 çift N_e, kT ∈ {1e-4, 2e-3}, yalnız en yakın komşu | **30/30**, en dar marj 5.5221e-02 |
| Hückel 4n+2 eşiği Λ_c·S(N) = 1 (model çıpası) | yarı dolu halka N ∈ {10, 14, 18, 22}, a = 2 ≤ r_on, kT = 0 | Λ_c·S − 1 = **+1.95e-07 … +5.21e-07**, tolerans 2e-6 |

Maliyet (bu makine, x'te N atomlu süperhücre, a = 2): kafes yolu N = 16 … 128
boyunca MD yolundan **9.5 – 51 kat** yavaş; MD bu yüzden kafes yolundan
geçmez ve iki yolun eşitliği yukarıdaki satırla kilitlidir. Çakışık atom artık
sessizce atılmıyor: `neighbour_shells` d ≤ 1e-9 bohr çiftinde
`InvalidInputError` verir (eskiden çifti atıyordu).

**Açık kalanlar (beyanlı, ayrı iş):** tek N_e ve tek N'de Kohn seçim kuralı
farklıdır ve pencere dışıdır; `uniform_model`in `max_cell = 2` kabuk kaybı (1B'de
kesme 3\|a\|'da) *03.10.2026'da düzeltildi: max_cell `max_cell_for_cutoff`tan, patolojik ölçek tarama yükü kapısında*; enine (y, z) hareket ve 3B periyodik kristal
sınanmadı; p orbitalleri ve açıya bağlı Slater-Koster tablosu yok; kenar
tarayıcısının kör noktaları (alt paket, `importlib`, `TYPE_CHECKING`) açık
*(26.09.2026: kapandı — alt paket ve göreli import çözülüyor, `importlib` ve
`TYPE_CHECKING` motor ağacında yasak ve adıyla kırmızı; Modül 7 özetindeki
"yalın sürüm" notu. Yasak listesi sonlu; açık kalan artık yollar: izinli bir
kenar üzerinden yeniden aktarım (aklama), `os.sys.modules`, `mock.patch`
dizgesi, `.pth`, C uzantısı)*.
Ayrıca (24.09.2026, bulgu düzeltme turu):

- Ω−E farkının P(x) = x/(4cosh²(x/4)) yasası yalnız Fermi çiftinde **2
  elektron** varken geçerli (çift N_e). Tek N_e'de (çiftte 1 ya da 3
  elektron) yasa başka: P₁(x), iki seviyeli kuramda P_max 0.2397, x ≈ 2.087'de.
  zincir(16, 2, 0.05, 3)'te n_e = 15 ve 17 için P_ölçülen/P₁ = 1.000000
  ölçüldü (x = 1 / 2.087 / 3.087) — **testi yok** (`tb/mermin.py` dokstringi).
- Hückel C3 (4n+2) ve C4 (4n) eşikleri yalnız çift N'de tanımlı; tek N'de yarı
  dolu halka bir soliton verir (eleştirmen E2). Eşik tek N'de **hiç ölçülmedi**.
- Kuyruk genişliği r_cut − r_on bütün dinamik pencerelerde 0.8 bohr'a sabit
  kaldı (yalnız C3'ün negatif testleri r_on'u statik olarak oynatır); konuma
  bağlı onsite (düzensizlik) sınıf onsite almadığı için denenemez.
- `observable/tb_obs.py` üzerinden hiçbir md-tb ölçümü yapılmadı; ön ölçümdeki
  P6'nın üç mercekli yeniden koşusu ayrı iş (kova artık kilitle belirlendiği
  için büyük ölçüde eskidi).
- kT > 0'da raporlanan μ kayan-nokta platosunun ortasıdır: kuyruklar yuvarlamanın
  altındayken (boşluk/kT ≳ 70) fiziksel kanonik kökten kT'nin bir kesri kadar
  sapar (ölçüldü: en çok 0.394 kT; Ω, F ve n etkilenmez). Log uzayında kök adımı
  yazılmadı.
- `mermin_dolum` çok büyük kT'de nan döner (kT = 1e308 ölçüldü; fizik penceresi
  dışı, kapı yok).

### Modül 3H — UHF indirgemesi / downfolding (Plan H)

Modül 3F'nin açıkça yazılmış en büyük açığı kapandı: **parametreler artık
dışarıdan girmiyor, qlab'ın kendi UHF'inden türetiliyor.**
`engine/tb/downfold.py`, `search_uhf`'in bulduğu spin-kırık çözümün α/β Fock
matrislerini Löwdin ile ortogonalleştirip site bazında okur

    H_eff^σ = S^(-1/2) F^σ S^(-1/2)

ve dört sayı üretir: `ε̄`, `Δ`, `t₁`, `t₂`. Sevk edilenler: `downfold_uhf`,
`model_from_downfold`, `fit_distance_laws` / `DistanceLaws`, `hydrogen_ring`;
`kind: downfold` senaryosu (`scenarios/halka_indirgeme.yaml`); testler
`test_downfold.py`, `test_downfold_cipa.py`, `test_downfold_kesme.py`,
`test_downfold_yasa.py`, `test_downfold_senaryo.py`.

**Modülün asıl kazancı bir sayı değil bir çıpa:** kademeli alanın çatallanma
üssü, üç halkada bağımsız olarak ½ çıkıyor. **Modülün asıl sınırı da yazılı:
hızlanma İDDİA EDİLMİYOR** — HF/TB maliyet oranı bu modülde hiç ölçülmedi.

#### Neden UHF, neden RHF değil — hüküm Hückel'den geliyor

Plan G'nin indirgemeyi RHF üzerine kurma denemesi düştü, çünkü hidrojen
halkalarında RHF çözümü çoğu geometride HF'in **taban durumu değildir**.
Desen 4n / 4n+2'dir ve **7/7 ölçüldü** (d = 1.6 bohr, dış — RHF→UHF, üçlü —
kararlılık özdeğeri ve UHF'in kazandırdığı enerji):

| N | tür | dış λ | UHF kazancı (hartree) |
|---|---|---|---|
| 6 | 4n+2 | **+1.26984** | 1.78e-15 |
| 8 | 4n | **−0.38237** | **0.099079** |
| 10 | 4n+2 | **+0.59956** | 7.11e-15 |
| 12 | 4n | **−0.30153** | **0.082805** |
| 14 | 4n+2 | **+0.22768** | 1.07e-14 |
| 16 | 4n | **−0.26614** | **0.077927** |
| 18 | 4n+2 | **+0.00983** | 2.84e-14 |

4n halkalarında TB spektrumunun HOMO/LUMO'su tam dejeneredir; RHF dejenere
seviyeyi çift işgale zorlayınca spin kararsızlığı doğar ve UHF gerçekten
aşağı iniyor (yaklaşık 0.08–0.10 hartree). 4n+2'de kazanç sayısal sıfır.

**N=8 ve N=12 test edilen hiçbir geometride kararlı değil** — altı mesafede
(1.6 / 1.8 / 2.0 / 2.2 / 2.6 / 3.0 bohr) **12/12** nokta negatif λ verdi.
N=8'in altı değeri: −0.382368, −0.390441, −0.399690, −0.409562, −0.428241,
−1.203061.

> **Plan H'nin "45 noktanın 35'i spin kararsız" sayısı tekrarlanamadı ve
> sebebi bu deponun kendi tekrar eden dersi:** ızgara yazılmamıştı. Adı konmuş
> bir ızgarayla yeniden ölçüldü — N ∈ {6, 8, 10, 12, 14} × d ∈ {1.6, 1.8, 2.0,
> 2.2, 2.6, 3.0, 3.4, 3.8, 4.2} bohr, yine 45 nokta — ve çıkan sayı
> **38/45**. Nitel hüküm (noktaların büyük çoğunluğu spin kararsız) aynı;
> kesin sayı **ızgaraya bağlı** ve ızgara adlandırılmadan tekrarlanamaz.
> Yukarıdaki 38/45 bu ızgaranın sayısıdır.
>
> Tamamlık için: özgün 35/45 **yanlış değildi**, ızgarası yazılmamıştı. O
> ölçüm N ∈ {6, 8, 10, 12, 14} × d ∈ {1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6,
> 3.2, 3.8} bohr üzerinde yapılmıştı ve o ızgarada tekrarlanıyor. İki sayının
> farkı, iki ızgaranın uzak mesafelerde farklı sıklıkta olmasıdır — 4.2
> bohr'a kadar giden ızgara daha çok kararsız nokta içerir. **Kusur sayıda
> değil, sayının yanında ızgaranın olmamasındaydı.**

#### Modelin şekli varsayım değil ÖLÇÜM

Spin-kırık halka çözümü antiferromanyetiktir ve bu her çağrıda **yeniden
doğrulanır**, varsayılmaz:

- Mulliken spin popülasyonu alternan (N=8, d=2.0: ∓0.517253)
- `diag(H_eff^α)` iki değer arasında alternan: **+0.250498 / −0.135858**
- `diag(H_eff^β)`, α'nın **bir site kaydırılmışı**. Kaydırma artığı beş
  sistemde ölçüldü ve sistemler **adlıdır** — aralık, hangi sistemlerden
  geldiği yazılmadan tekrarlanamaz:

  | sistem | kaydırma artığı |
  |---|---|
  | N=6, d=2.6 | 2.220446e-15 |
  | N=6, d=3.0 | 1.942890e-15 |
  | N=8, d=2.0 | 5.558044e-09 |
  | N=8, d=2.6 | 3.951947e-09 |
  | N=10, d=2.6 | 9.087647e-09 |

- Hoplama **spinden bağımsız ve tekbiçim**: N=8 d=2.0'da t = −0.407145,
  sekiz bağın sekizinde de aynı. "Aynı"nın ölçüsü:
  max|tᵅ − tᵅ[0]| = **3.003514e-09**, max|tᵅ − tᵝ| = **8.213723e-10**. Bu
  sayılar sıfır değildir ve olmaları da beklenmez — `solve_uhf`'in artık
  toleransı 1e-8'dir, Fock elemanlarının gürültü tabanı oradadır.

Dolayısıyla bağımsız parametre **dörttür**. `Δ`'nın **işareti bir ayar
seçimidir** — hangi alt-örgünün α fazlası taşıdığına UHF araması karar verir
ve karar d'den d'ye döner. Ölçüldü (N=8, d = 1.6 … 3.6, on nokta):
`+ − + − + − + + + +`.

#### ÇIPA — kademeli alanın çatallanma üssü · **SINIF: dairesel değil**

İki büyüklük **iki ayrı hesaptan** gelir ve birbirini hiç kullanmaz. `d_c`,
RHF çözümünün üçlü orbital Hessian'ının sıfır kökünden (Modül 1C,
`analysis.external_stability`); `Δ` indirgemeden. Üs hiçbir yerden verilmez,
log-log eğiminden okunur.

| N | d_c (bohr) | ölçülen üs | Δ(d_c + 0.032) | log-artık (max) |
|---|---|---|---|---|
| 6 | **2.233818500** | **0.496190** | 8.115373e-02 | 1.892e-03 |
| 10 | **1.957853161** | **0.497600** | 6.145320e-02 | 1.169e-03 |
| 14 | **1.757512489** | **0.497811** | 4.948334e-02 | 1.034e-03 |

Üçü de çatallanma beklentisi **½**'nin hemen altında: sapmalar 0.003810
(N=6), 0.002400 (N=10), 0.002189 (N=14) — ve N büyüdükçe küçülüyor. Ve
**üçüncü bir bağımsız okuma**: Modül 1C, H₂ molekülünde — başka bir sistem,
başka bir yordam, Coulson-Fischer noktasında |spin| ölçüsü — üssü **0.4970**
ölçmüştü (`tests/hf/test_coulson_fischer.py`). Aynı yere düşüyor.

> Task 2'nin dokstringi bu satırda *"üs ½'ye 0.0022'den yakın"* diyordu ve o
> sayı **yanlıştı** — N=6'nın sapması 0.003810, yani iddia edilenin 1.7 katı.
> Doğrulama turunda yakalandı ve dokstring düzeltildi; testin toleransı
> (abs=0.02) zaten gerçek sapmanın çok üstündeydi, yani hata testi kırmıyordu
> ve ancak **yeniden ölçülünce** görülebilirdi.

Çıpa **iki taraflıdır**: geçişin altında `Δ` sönüyor — N=6 d_c−0.20'de
2.886867e-09, d_c−0.02'de 1.539807e-08; N=10'da 9.908740e-16 ve
1.998401e-16. Bu artıklar fizik değil `solve_uhf`'in artık tabanıdır ve eşik
(1e-7) tam da bu yüzden planın önerdiği 1e-8 **değildir**: N=6 d_c−0.02
noktası 1.539807e-08 ile 1e-8'i aşıyor.

> **Üs testi tek başına ÖLÇEĞE KÖRDÜR** ve bu delik ayrıca kapatıldı: log-log
> eğimi `Δ → c·Δ` dönüşümü altında değişmez, yani üs testi `Δ`'yı sabit bir
> sayıyla çarpan bir kusuru göremez. Aynı testte mutlak bir değer de
> kilitlenir (yukarıdaki `Δ(d_c+0.032)` sütunu) ve o satır **transkripsiyon
> denetimidir**, çıpa değil.

#### Kesme hatasının dürüst tablosu

Hata = max|iki-siteli TB özdeğerleri − UHF α özdeğerleri|, hartree, STO-3G,
seeds=16. **Kesme hatası küçük DEĞİLDİR ve büyük olduğu yazılıdır.**

| N | d | UHF yayılımı | NN-only | +t₂ | oran | %NN | %t₂ |
|---|---|---|---|---|---|---|---|
| 6 | 2.60 | 1.113021 | 0.039420 | **0.012078** | 3.26 | 3.5% | 1.1% |
| 6 | 3.00 | 0.947273 | 0.023748 | **0.009107** | 2.61 | 2.5% | 1.0% |
| 6 | 3.60 | 0.834770 | 0.010104 | **0.003925** | 2.57 | 1.2% | 0.5% |
| 8 | 2.00 | 1.693171 | 0.153603 | **0.066287** | 2.32 | 9.1% | 3.9% |
| 8 | 2.60 | 1.134138 | 0.073229 | **0.032485** | 2.25 | 6.5% | 2.9% |
| 10 | 2.60 | 1.140294 | 0.059631 | **0.020470** | 2.91 | 5.2% | 1.8% |

Planın (ve ilk dokstringin) "t₂ hatayı ~3 kat düşürür" cümlesi **düzeltildi**:
ölçülen oran 2.25 … 3.26, ortalaması 2.65 — "üç kat" altı sistemin yalnızca
**birinde** tutuyor. Doğru cümle *"iki ila üç kat"*tır.

#### Modelin tavanı MENZİL değil KABUK ORTALAMASI

"Menzili büyütünce hata sıfıra gider" beklentisi **yanlış** ve ölçüldü:
N=8 d=2.0'da menzil tamamen açıkken (m=4, hiçbir şey kesilmemişken) hata hâlâ
**0.063140** — menzil 2'nin 0.066287'sinden yalnızca %5 aşağıda.

Sebep iki parçalı ve ikisi de ölçüldü. **(a) İndirgemenin kendisi kayıpsız:**
`H_eff^α`'nın özdeğerleri UHF α orbital enerjileriyle **7.194e-10**'da aynı.
**(b) Kayıp kabuk ortalamasında:** model her halka mesafesi m için tek bir
`t_m` taşır, ama `H_eff^α` **çift** m'de A-A ile B-B'yi ayırır (α kanalında A
alt-örgüsü α fazlası, B β fazlası taşır ve ikisi eşdeğer değildir). Kabuk içi
yayılım, N=8 d=2.0:

| kabuk | yayılım |
|---|---|
| m=1 | 3.003515e-09 (tekbiçim — SCF artık tabanı) |
| m=2 | **5.106121e-02** |
| m=3 | 1.007388e-09 (tekbiçim) |
| m=4 | **2.415794e-02** |

Tek m'de kabuk tekbiçim, çünkü o hoplamalar A↔B'yi bağlar ve AFM işlemi
altında simetriktir. **Hükmü:** hatayı gerçekten düşürmek için gereken şey t₃
değil, çift menzilde alt-örgüye bağlı **iki ayrı** hoplamadır (t₂^AA, t₂^BB) —
ve o, dört parametreli modelin **kapsamı dışındadır**.

#### Sahte-çıpa sınavı · **SINIF: dairesel değil, ama DAR**

Sınav: fiziği çıkar, **boru hattını yerinde bırak**; sonra aynı yasa uydurma
makinesini fizik-dışı girdiyle koştur. Ölçülen metrik **ekstrapolasyon
oranı** = (pencere dışı log artık RMS) / (pencere içi log artık RMS) — yani
"yasa, uydurduğu yerden ne kadar uzağa taşıyor".

N=8 halkası, pencere 1.8/2.0/2.2 bohr, hedef 2.4/2.6/3.0 bohr, seeds=12:

| kurulum | oran |
|---|---|
| **GERÇEK (STO-3G)** | **4.160383** |
| 40 rastgele düğümsüz baz — medyan | 29.902685 |
| aynı 40 — **en iyisi** | 6.216006 |
| aynı 40 — en kötüsü | 413.126243 |
| **gerçeği geçen** | **0/40** |
| F3 saf geometri çekirdeği (`S=exp(−a r)`, `H=S⁻¹`), beş a | 9.9647 … 29.3917, **0/5** |

45 fizik-dışı kurulumun 45'i de gerçekten kötü. **F2 (`eri*0`) koşulamadı ve
bu bir eksiklik değil bir ölçüm:** elektron-elektron kapanınca değiştokuş da
yok olur, UHF RHF'e çöker ve |Δ| = 1.110223e-16 çıkar; `fit_distance_laws`
bunu görüp `UnsupportedSystemError` fırlatıyor. İkame **beyanla değil kodun
kendi kapsam kontrolüyle** elendi. Aynı kontrol F1'de de gerçekten ateşledi:
41 çizimin 1'i "t₂ pencere içinde işaret değiştiriyor" diye reddedildi.

#### Dürüst olumsuzlar — dördü de ölçüldü

1. **Kuyruk marjı RHF tarafının üçte biri.** Burada gerçek 4.160383, en iyi
   sahte 6.216006 — yalnızca **1.49 kat**. Karşılaştırma noktası Plan H'nin
   RHF yolu üzerindeki ön ölçümüdür (2.80'e karşı en iyi sahte 12.01, yani
   **4.29 kat**); o rakam bu depoda **yeniden ölçülemez** çünkü RHF
   indirgemesi sevk edilmedi — kaynağı belirtilerek aktarılıyor. Sınav
   geçiyor ama ayırma gücü düşmüş durumda ve bu yazılmadan geçilemez.
2. **İkinci metrik aynı hükmü vermiyor.** "Yasa pencere dışını iyi öngörüyor"
   cümlesi tek başına fizik kanıtı **değildir**: ham bağıl öngörü hatasında
   40 sahtenin **2'si** (0.005409 ve 0.005571; gerçek 0.010811) ve saf
   geometri çekirdeklerinin **5'inden 2'si** (0.000997 ve 0.000297) gerçekten
   **daha iyi** öngörüyor. `exp(−a r)` tanımı gereği üsteldir; "üstel bir
   yasaya uyuyor" olmak bir çekirdeği fiziksel yapmaz.
3. **Mutlak hata tek başına da ayırmıyor, ters yönden.** 40 sahtenin **7'si**
   gerçekten küçük mutlak hata veriyor — ikisinin sebebi t₁'lerinin neredeyse
   **sıfır** olması. Bu yüzden her iki sınır da (bağıl VE mutlak) kilitlenir.
4. **Sınav HEDEF KÜMESİNE KOŞULLU.** Aynı yasa, pencere hiç değişmeden,
   yalnızca hedef kümesi uzatılarak:

   | hedef kümesi (bohr) | oran |
   |---|---|
   | (2.4, 2.6, 3.0) | **4.160383** |
   | + 3.2 | 5.997079 |
   | + 3.6 | **15.479892** |

   d = 3.6 hedefe konsaydı **gerçek sistem en iyi fizik-dışı bazdan
   (6.216006) daha kötü görünürdü ve sınav düşerdi.** Çıpanın geçtiği hüküm
   bu yüzden "yasa ekstrapole eder" değil, şu dar hükümdür: *1.8–2.2 bohr
   penceresinden uydurulan t₁ yasası, 3.0 bohr'a kadar fizik-dışı bazların
   üretemediği bir doğrulukla öngörüyor.* 3.0 bohr'un ötesi için bu modül bir
   şey iddia etmez.

#### `p = 2` ölçülmedi, SEÇİLDİ — ve bedeli ölçüldü

`ε̄ = ε_∞ + C·t₁²` yasasındaki üs Löwdin'in ikinci mertebe açılımından gelir
(yerinde düzeltme örtüşmede ikinci mertebedir), yani **teoriden**. Serbest
bırakıldığında N=8 halkasının on noktalı gridinde ölçülen üs **1.8887** —
yani serbest uyum 2'ye yakın ama onu **vermiyor**. (Plan H'nin RHF yolundaki
ön ölçümü 1.9869 diyordu; o sayı bu depoda yeniden ölçülemez, kaynağı
belirtilerek aktarılıyor. UHF tarafında serbest üs 2'den **daha uzak**.) p=2
zorlamasının pencere içi artığı 5.737638e-04 RMS.

#### `Δ` için temiz bir mesafe yasası YOK — ve sebebi ölçüldü

`ln|t_m|` doğrusal oturuyor (artık RMS 1.855587e-03 ve 4.104227e-04), ama
`|Δ| = A·exp(b·d)` **yalnızca pencere içinde** geçerli; dışarıda hızla
bozuluyor: bağıl hata 0.028079 (d=2.4) → 0.055498 (2.6) → 0.069184 (3.0),
yani t₁'in maksimumunun (0.010811) **altı katı** ve d ile monoton artıyor.
Sebep: **`Δ` doyuyor.** N=8'de |Δ| d=1.6'da 0.170033 iken d=3.6'da
0.363453'te doyuyor; pencere yasası orada %5.0 yüksek tahmin veriyor
(0.381747).

Doymanın sebebi **bağımsız bir hesapla** doğrulandı. Hubbard ortalama alan
boşluk denklemi, ölçülen (t₁, Δ) çiftinden U'yu çözdürür; çıkan U neredeyse
sabittir ve Löwdin bazında **bağımsız olarak** hesaplanan yerinde Coulomb
integrali (00|00) ile aynı yere düşer:

| d (bohr) | t₁ | \|Δ\| | U (boşluk denklemi) | (00\|00) | fark |
|---|---|---|---|---|---|
| 1.6 | −0.573180 | 0.170033 | 0.873471 | 0.946171 | 7.7% |
| 2.0 | −0.407145 | 0.193178 | 0.827766 | 0.883907 | 6.4% |
| 2.6 | −0.255455 | 0.265159 | 0.802284 | 0.828145 | 3.1% |
| 3.0 | −0.183164 | 0.318473 | 0.794279 | 0.806566 | 1.5% |
| 3.6 | −0.106972 | 0.363453 | 0.784102 | 0.788356 | **0.5%** |
| 6.0 | (ayrışmış atom sınırı) | — | — | 0.774783 | — |

U uçtan uca 0.873471 → 0.784102 (oran 1.114) oynuyor; uyum d büyüdükçe
iyileşiyor — ortalama alanın ve atomik sınırın en iyi çalıştığı yerde. Yani
`Δ` doyar çünkü U mesafeye kabaca duyarsızdır ve yerel moment 1'e doyar;
Δ = U·m/2.

#### Sınırlar — modülün beyan ettiği kapsam

- **Yalnız STO-3G.** 6-31G atom başına iki fonksiyon taşır ve site-bazlı okuma
  orada **tanımsızdır** (`H_eff[0,1]` atom-*içi* elemanı döndürürdü). Kod bunu
  sessizce geçmez, `UnsupportedSystemError` fırlatır.
- **Yalnız halka.** Açık zincir kapsam dışıdır ve gerekçesi ölçüldü: N=8 açık
  zincirde, halkadan gelen **bütün** menzillerle (m = 1…4; halka mesafesi
  orada doyar, bulk parametrizasyonu daha uzağını zaten tanımlamaz) kurulan
  modelin hatası d=1.8 bohr'da **0.095563 Ha**. Plan H'nin ön ölçümü aynı
  büyüklüğü "0.096 Ha'lık bir taban" diye kaydetmişti ve sayı tutuyor; ama
  **taban olduğu bu protokolde görünmüyor** — hata d ile düzgün küçülüyor
  (0.086963 / 0.058007 / 0.024325 @ 2.0 / 2.6 / 3.2 bohr). Yani o taban, eğer
  varsa, **menzil ekseninde** bir tabandır, mesafe ekseninde değil. Kenar
  etkisi bu modülde modellenmez ve bu satır onu modellemeye çalışmaz.
- **UHF bir spin özdurumu DEĞİLDİR.** ⟨S²⟩ sonuca iliştirilir ama
  **düzeltilmez** (spin projeksiyonu / PUHF yoktur). Ölçüldü: N=8 d=2.0'da
  ⟨S²⟩ = **1.220731** — ne singlet ne triplet. Sevk edilen senaryonun en uzak
  noktasında (d=3.2) **2.948899**'a çıkıyor.
- **Parametreler bir ARAMA protokolünden gelir**, tek bir SCF'ten değil.
  Arama özeti panelde yazılıdır ve ayrışma gerçektir: sevk edilen senaryoda
  d=1.8/2.0/2.2'de 12 tohumun 12'si yakınsıyor, d=2.6'da 11'i, **d=3.2'de
  yalnızca 9'u**.
- **`t`'nin işareti bir AYAR SEÇİMİDİR** ve bu modül onu sabitlemez
  (`tests/tb/test_ayar_ve_isaret.py`). Aynı şey `Δ`'nın işareti için de
  geçerlidir; `DistanceLaws.at()` bu yüzden her zaman pozitif `Δ` döndürür.
- **HIZLANMA İDDİA EDİLMİYOR.** Bu modül HF/TB maliyet oranını **ölçmedi**.
  İddia edilen tek şey parametrelerin türetilebilir ve çapalanabilir
  olduğudur — ve bu cümle şeffaflık panelinde de yazılıdır.

#### Panel ve senaryo

```bash
python -m qlab.run scenarios/halka_indirgeme.yaml -o indirgeme.npz
```

Çıkış 0; panel Yöntem / Yaklaşımlar / İhmal edilen / Beklenen hata / Yakınsama
satırlarını basar. `Yakınsama` burada "SCF durdu" **demek değildir**: ölçüt,
modelin varsaydığı iki-siteli antiferromanyetik yapının her mesafede
gerçekten bulunmasıdır (`staggering_residual_max` < 1e-7; sevk edilen
senaryoda ölçülen **7.526e-09**).

Beyan metninin **tek kaynağı** `engine/tb/factory.py`dir ve bu bir disiplin
değil bir **test**tir: her beyan cümlesi `src/` ağacında tam bir kez geçer;
`tests/` tarafında da — kilidi kuran tek dosya ve dört standart terim dışında
— hiç geçmez. `run.py::_run_downfold` metnin bir harfini bile yeniden yazmaz,
yalnızca `provenance.as_lines()` basar. Gerekçesi bu depoda ölçülmüştü: aynı
blok bir kez `scf.py` ile `factory.py`de kopyalanmış, biri düzeltilip öteki
unutulunca panel iki farklı şey iddia etmişti.

Sayım denetiminin kendisi de bir tuzağı kapatıyor ve o tuzak ölçüldü: bitişik
dize sabitleri (`"birinci parca "` + `"ikinci parca"`) kaynakta **iki**
parçadır ama çalışma zamanında **tek** dizedir; ham metinde aramak 0 eşleşme
verir ve "kopya yok" diye okunurdu — yani test kendi iddiasının tersini
kanıtlardı. Denetim bu yüzden metni önce birleştiriyor.

### Modül 4 — spin Monte Carlo / manyetizma (Plan I)

#### Yol haritasının hedefi ölçümle geçersiz kılındı: **demir tautolojisi**

Yol haritası "demirin Curie sıcaklığı 1043 K" diyordu. Bu bir doğrulama
değil, bir **tautoloji**: 2B Ising modeli **tek** bir enerji ölçeği taşır ve
`T_c ∝ J`. Onsager'in 1944 tam çözümü oranı sabitler:

    kT_c / J = 2 / ln(1 + √2) = 2.269185314

Dolayısıyla SI birimindeki her `T_c` girilen `J`'nin geri okunmasıdır. Modül 4
kapanışında **yeniden ölçüldü** (`k_B = 8.617333262e-2 meV/K`, CODATA 2018):

| hedef | gereken `J` | o `J` ile geri okunan `T_c` | fark |
|---|---|---|---|
| **demir, 1043 K** | **39.608394 meV** | **1043.0000 K** | **0.000e+00** |
| nikel, 627 K | 23.810607 meV | 627.0000 K | 0.000e+00 |
| kobalt, 1388 K | 52.709924 meV | 1388.0000 K | 0.000e+00 |
| gadolinyum, 292 K | 11.088831 meV | 292.0000 K | 5.684e-14 |

Aynı kadran her malzemeyi verir; hedef seçilir, parametre ondan türetilir,
sonra parametre "sonuç" diye geri okunur. (Gadolinyum satırındaki 5.684e-14
kayan nokta yuvarlamasıdır, fizik değil. Tabloda gösterilen **altı haneye
yuvarlanmış** `J`'ler kullanılırsa fark 7.2e-07 … 7.8e-06 K olur — yani
"1043.0000 K" hanesini bozan tek şey, sayının kaç basamağının yazıldığıdır.) **Modül 3F'nin silisyum bant
aralığı hedefiyle aynı aile** ve çekirdek spec'in aynı tablosundaki
ölçümle-düzeltilmiş **ikinci** hedef. Hedef düzeltildi; yerine aşağıdaki
dört çıpa kondu ve **dördü de girdilerden geri okunamaz**.

#### Dört doğrulama hedefi ve ÖLÇÜLEN sayıları

| # | Çıpa | Ölçülen | Referans | Sınıf |
|---|---|---|---|---|
| 1 | Binder kesişimi → `kT_c/J` | **2.265050 / 2.265253 / 2.265570** (üç ikili) | Onsager **2.269185314**, simülasyona **söylenmedi** | dairesel değil |
| 2 | Kritik üs `β/ν` | **0.122941 ± 0.003225** | tam **0.125** (−0.64σ); ortalama alan **1.0** → **272.0 kat** marjla dışlandı | dairesel değil (evrensel) |
| 3 | Mermin-Wagner | Ising eğimi **+0.000002**, Heisenberg **−1.0545** (T = 1.0) | teorem: 2B'de sürekli simetri düzenlenmez, ayrık düzenlenir | dairesel değil (teorem) |
| 4 | Modüller arası bağ | 2B Ising **0.122941** ↔ Modül 3H UHF **0.4962** | mean-field ½ ile tam çözümün ⅛'i **dört kat** farklı | dairesel değil |

#### Onsager — çözünürlük tuzağı ve gerçek sistematik kayma

Planın "kesişim 2.260, sapma %0.4" iddiası **yeniden ölçüldü ve çürüdü**:
%0.4 mutlak olarak 0.0092'dir, yani planın kendi ızgara adımının (0.04)
**dörtte biri**. O sayı bir kesişim değil, bir **ızgara noktasıydı**.

Bu modülde kesişim **doğrusal ara değerlemeyle** bulunur ve ara değerlemenin
ızgara sistematiği **gürültüsüz** (tam sayımla) ölçülür: adım 0.04 →
−5.950e-04, **0.02 → −2.156e-04**, 0.001 → −4.2e-07. Sistematik, istatistiksel
σ'nın (3.0–5.2e-03) **on dörtte biri** — yani sınırlayıcı olan ızgara değil,
istatistik.

| çift (L × L') | kesişim | σ | Onsager'den sapma | 3σ |
|---|---|---|---|---|
| 8 × 16 | 2.265050 | 0.004118 | −0.004135 (−1.00σ) | 0.01235 |
| 8 × 24 | 2.265253 | 0.003020 | −0.003932 (−1.30σ) | 0.00906 |
| 16 × 24 | 2.265570 | 0.005206 | −0.003616 (−0.69σ) | 0.01562 |

Üçünün yayılımı **0.000519**, en dar 3σ'nın on yedide biri: **bu istatistikle
üçü ayırt edilemiyor**. Ama üçü de **aynı yönde** sapıyor ve tek bir tohum
bloğu bunu ayıramaz. Üç bağımsız tohum bloğu (20260824 / 20260825 / 19990101)
havuzlanınca:

    blok ortalamalari  2.265291 / 2.266747 / 2.267474
    havuz              2.266504 ± 0.000642
    Onsager sapmasi    -0.002681 = **-4.18σ**

Yani kayma **gerçek ve sistematiktir** (sonlu boyut), tek blokta gürültünün
içinde kalır. Kaymanın gerçekliği ayrıca gürültüsüz de gösteriliyor: `L = 3` ×
`L = 4` kafeslerin **tam sayımında** (512 ve 65536 durum, hata çubuğu yok)
kesişim **2.1692678**, Onsager'den **−0.0999 (−%4.40)**. Küçük `L`'de kayma
devleşiyor, `L ≥ 8`'de bu istatistiğin çözünürlüğünün altına iniyor.

#### Kritik üs — ve "β = 1/8 çıktı" yazmanın neden yasak olduğu

Pencere: `L` = 8, 12, 16, 24, 32 · `T = T_c` · Wolff · `n_burn = 300`,
`n_sweep = 1500` · K = 24 bağımsız tohumlu zincir · 3000 çekilişlik bootstrap.

    <|m|>: 0.775523 / 0.740615 / 0.715627 / 0.678801 / 0.654571
    β/ν  = 0.122941 ± 0.003225      (3σ = 0.009674)

Doğru cümle: **β/ν = 0.122941 ± 0.003225 ölçüldü; tam değer 0.125 (sapma
−0.64σ, yani belirsizliğin içinde); ortalama alan (β/ν = 1.0) 272.0 kat marjla
dışlandı.** Kalan sapma bir sonlu-boyut düzeltmesidir ve **sıfırlanmaz** — bu,
`L` alt kümeleri ayrı ayrı uydurulunca görünür: (8,12,16) → 0.121346,
(16,24,32) → 0.129350, (24,32,48) → 0.123540. Yani log-log doğrusu **tam doğru
değildir**; altı noktanın 0.125'e bu kadar yakın düşmesi düzeltmelerin bu `L`
penceresinde birbirini götürmesindendir.

`β/ν` ile `β` sayısal olarak aynıdır çünkü 2B Ising'de `ν = 1`. **Bu depo
`ν`yü ÖLÇMEZ**; `ν = 1` bağı literatürdendir ve beyan edilir.

#### Mermin-Wagner — aynı kod, zıt hüküm, **ve penceresi**

Pencere: `⟨|m|⟩`, `L` = 8, 12, 16, 24, 32 · `J = 1`, `h = 0` · **aynı**
Metropolis çekirdeği · `n_burn = 500`, `n_sweep = 1500` · tohum 20260822 ·
soğuk (tekbiçim) başlangıç.

| T | Ising eğimi | Heisenberg eğimi | m_I(32)/m_H(32) |
|---|---|---|---|
| **0.8** | +0.000033 | **−0.9196** | 6.99 |
| 1.0 | +0.000002 | **−1.0545** | 11.35 |
| 1.4 | +0.000174 | **−0.9580** | 16.04 |

Ising'in mıknatıslanması **dört hanede `L`'den bağımsız** (T = 1.0'da L = 8
için 0.999208, L = 32 için 0.999250); Heisenberg'inki güç yasasıyla sönüyor.
Farkı yaratan şey **kod değil simetridir** ve bunu `metropolis.py`nin
koşturduğu modeli **bilmemesi** garanti eder (üç ayrı testle kilitli).

**GEÇERLİLİK PENCERESİ — beyan ediliyor, gevşetme değil: çıpa `T ≳ 1.0`'da
kurulur.** `T = 0.8` ayrı bir parametre olarak ve gevşek sınırla girer, oran
iddiası orada **hiç kurulmaz**. Sebebi ölçüldü: 2B Heisenberg'de `ξ ~
exp(2πJ/T)` ile patlar ve `L ≫ ξ` koşulu `L = 32`'de sağlanmaz. İmzası da bu —
8 tohumda eğim `T = 0.8`'de **−0.8690 ± 0.0754** ile −1'e ulaşamıyor ve
yayılım `T = 1.4`'ünkinin (±0.0422) neredeyse iki katı; oran orada **5.52**'ye
kadar iniyor, yani "> 8.0" iddiası **yanlış olurdu ve kurulmadı**.

#### Hata çubuğu — deponun ilk hata çubuğu, ve ilk kuralı

**Naif SEM bu depoda hiçbir yerde raporlanmaz.** MC örnekleri ilişkilidir ve
`std/√N` gerçek hatayı sistematik olarak küçük gösterir. Ölçüldü (L = 32,
T = T_c, 8000 ölçüm; bağımsız referans = 200 zincirin (Metropolis) ve 100
zincirin (Wolff) ortalamalarının std'si):

| algoritma | naif SEM | bağımsız referans | naif kaç kat küçük | referansın kendi belirsizliği |
|---|---|---|---|---|
| Metropolis | 1.946079e-03 | 1.713639e-02 | **8.81×** | %5.0 |
| Wolff | 1.955908e-03 | 4.642595e-03 | 2.37× | %7.1 |

Planın **"10.4 kat"**ı bayattı; 200 zincirlik referansla **8.81 kat** çıkıyor.
Ve aynı ölçüm 24 zincirle yapılınca gerçeğin **0.80 katı** oluyor
(1.3666e-02): **referansın kendisi de bir tahmindir** — bu ders bu modülde
üç kez ayrı ayrı ödendi.

**Plato bayrağı ALGORİTMANIN DEĞİL KOŞUMUN özelliğidir.** Planın "Metropolis
plato yapmaz, Wolff yapar" anlatısı bir algoritma cümlesi olarak **yanlış**;
ölçüldü (T = T_c, 24 zincir):

| algoritma | pencere | plato |
|---|---|---|
| Metropolis | L = 8, N = 4000 | **24/24** |
| Metropolis | L = 16, N = 8000 | 23/24 |
| Metropolis | L = 32, N = 8000 | **0/24** |

Bayrak "bu algoritma iyi mi" değil, "**bu koşum** kendi otokorelasyonunu
çözecek kadar uzun mu" sorusunu cevaplar. Plato oluşmamışsa dönen sayı hatanın
kendisi değil bir **ALT SINIRIDIR** ve panel onu öyle işaretler.

Ölçütün kendisi de düzeltildi: planın "son üç blok boyutunun hataları %5
içinde mi" ölçütü, kendi gürültüsü %6.35–12.91 olan sayıları %5'lik bir
eşikle karşılaştırıyordu ve Wolff'un plato **yaptığı** pencerede 24 tohumun
yalnız **3'ünde** `True` diyordu — yani planın kendi zorunlu testi kendi
koduyla **21/24 kırmızı** olurdu. Yerine konan ölçüt bir **seçimdir**
(iki düğme taranarak bulundu) ve daireyi kıran şey AR(1) sınavıdır:
`τ_int` orada kapalı formdan bilinir ve ölçüt hiç ayarlanmadan doğru hükmü
veriyor —

    φ       tam τ     binning τ   oran    plato
    0.0       0.500       0.541   1.082   24/24
    0.5       1.500       1.552   1.035   24/24
    0.8       4.500       4.446   0.988   24/24
    0.95     19.500      16.333   0.838   10/24   (geçiş bölgesi)
    0.995   199.500      52.833   0.265    0/24

#### ξ'nin kırılması — ve düzeltilmiş hüküm

Plan `ξ`nin `T ≥ 1.0`'da kırıldığını üç negatif oranla (−0.0511 / −0.0125 /
−0.0383) yazmıştı. **Yeniden ölçüldü: işaret tohumdan tohuma değişiyor** (her
sıcaklıkta 2/5). Doğru hüküm *"orada oran negatiftir"* değil,
***"orada tahminci KARARSIZDIR"***.

Ve `None` hikâyenin yalnızca yarısıdır: tamamen **ilişkisiz** bir ±1 kafeste
(gerçek `ξ = 0`) 200 tohumun `L = 16`'da 123'ü, `L = 32`'de 124'ü `None`
veriyor — kalan **~%38'i sahte bir `ξ` üretiyor** (4.418 ve 7.791). Yani
pozitif dönüş de orada güvenilir değil. `max(oran, 0)` ile kırpmak **yasak**:
sıfır "korelasyon yok" diye okunur, oysa gerçek hüküm "bu tahminci burada bir
şey söylemez".

İki ayrı geçerlilik sınırı daha ölçüldü ve beyan ediliyor:

- **Tek konfigürasyon yetmez.** T_c, L = 32, 5 tohum: `ξ` = 3.644 / 17.735 /
  30.337 / 36.768 / 150.562 — **41 kat** yayılım. Aynı 5 tohumun 200'er
  konfigürasyonluk topluluğunda 26.668 … 33.078 (%11).
- **Düzenli fazda ham tahminci mıknatıslanmayı okuyor** ve topluluk bunu
  **çözmüyor**: T = 1.5, L = 32'de `ξ` = 613.966 … 1431.310, yani `ξ/L` 16–45.
  L = 32'lik bir kafeste `ξ = 800` bir korelasyon uzunluğu değildir.
- **"Bağlı" (connected) sürüm cebirsel olarak ölüdür**: tek konfigürasyondan
  site ortalamasını çıkarmak `Σ(s−s̄) = 0` yapar, `S(0)` tam sıfır olur ve oran
  **her zaman tam −1** çıkar (15/15 ölçüldü).

Bu yüzden sevk edilen senaryo `ξ`yi **raporlamaz** ve gerekçesini panelde
yazar.

#### İki algoritma, tek hüküm — ve çapraz kontrolün kendi kör noktası

Metropolis ile Wolff'un ortak hiçbir karar noktası yoktur (biri kabul/red
verir, öteki hiç vermez), ve altı pencerede birbirini doğruluyorlar (fark /
3σ oranları 0.109 … 0.481). Ama çapraz kontrol *"ikisi de aynı şekilde
bozuk"* ihtimalini dışlayamaz; onu **tam sayım** dışlıyor: `L = 3` (512
durum) ve `L = 4` (65536), enerji bağımsız yazılmış, sapmalar
**0.08 / 0.66 / 0.78 / 1.05 × SE**. `L = 3` ayrıca bir kapsam iddiasıdır —
Metropolis tek `L`'yi reddeder, Wolff koşar; o çıpa çapraz kontrolle
**kurulamazdı**.

#### Modüller arası bağ

Modül 3H, UHF indirgemesinde kademeli alanın çatallanma üssünü üç halkada
bağımsız olarak **0.496190 / 0.497600 / 0.497811** ölçtü ve **UHF bir
ortalama-alan yöntemidir**, yani o üs ½'dir. Modül 4 aynı depoda, aynı tür
geçişte, gerçek 2B Ising'i koşuyor ve **0.122941** ölçüyor. İki sayı birbirini
bağlamlandırıyor: **mean-field ile tam çözümün farkı artık bu depoda
ölçülmüştür** ve ikisi de dairesel olmaktan çıkıyor.

#### Panel ve senaryo

```bash
python -m qlab.run scenarios/spin_tarama.yaml -o spin.npz
```

Çıkış 0; panel **yedi** satır basar — Yöntem / Yaklaşımlar / İhmal edilen /
Beklenen hata / **Dairesellik** / Yakınsama / **Örnekleme**. Son ikisi bu
modülde `Provenance`a eklendi ve ayrı alan olmalarının gerekçesi
`provenance.py`de yazılı: dairesellik beyanı `Yaklaşımlar` listesine konsaydı
modülün varlık sebebi beş maddeden biri olarak akıp giderdi, örnekleme tanısı
`diagnostics`e konsaydı panelde **"✓ … ALT SINIR"** diye görünürdü — yani
uyarının yanında bir onay işareti.

`Yakınsama` burada "hesap bitti" **demek değildir**: ölçüt, raporlanan **her**
hata çubuğunun gerçekten bir hata çubuğu olup olmadığıdır. Sevk edilen
senaryoda ✗ basar ve bu doğrudur — T = 2.2 ve T = 2.4'te binning platosu
oluşmuyor. Ölçüldü (24 tohum, aynı pencere): T = 1.6 → 23/24, T = 2.0 →
16/24, **T = 2.2 → 0/24**, **T = 2.4 → 0/24**, T = 2.8 → 13/24,
T = 3.4 → 24/24. Yani işaretin **iki yönü de** aynı sevk edilmiş senaryodan
kilitlenebiliyor.

Beyan metninin **tek kaynağı** `engine/spin/factory.py`dir ve bu bir disiplin
değil bir **test**tir: on iki beyan cümlesinin her biri `src/` ağacında tam bir
kez geçer, `tests/` tarafında da kilidi kuran tek dosya dışında hiç geçmez.
`run.py::_run_spin` metnin bir harfini bile yeniden yazmaz. Tek kelimelik
istisna `ALT SINIR` işaretidir ve o da yazılmaz, fabrikadan **ithal edilir**.

#### Dürüst olumsuzlar

- **`ν` ölçülmedi.** `β/ν = β` yalnızca literatürden gelen `ν = 1` bağı
  yüzündendir.
- **Plato bayrağının gerçek bir geçiş bölgesi var** (AR(1)'de φ = 0.95 →
  10/24; farklı tohumlarla bağımsız yeniden koşumda 8/24 — tohum gürültüsü,
  aynı hüküm). Kusur değil, çözünürlük. Ve bayrak **muhafazakârdır**: `False`
  derken binning değeri gerçeğin 0.854 katı — yani "plato yok", *serideki
  kanıtın eksikliğini* söylüyor, sayının yanlış olduğunun ispatını değil.
- **Hiçbir tahminci `τ = 199.5`'i temiz okumuyor.** ACF de kırılıyor
  (φ = 0.995'te 0.85× ve ±%48 saçılım); N = 8000 o otokorelasyona yetmiyor.
- **Planın "naif Metropolis + doğrudan fit → 0.0973" iddiası SINANMADI** —
  kritik üs her yerde Wolff ile ölçüldü.
- **Spec ön ölçümünün `τ_int` = 53.78 (Metropolis) / 3.07 (Wolff)
  karşılaştırması bu modülde YENİDEN ÖLÇÜLMEDİ.** Sevk edilen senaryonun
  kendi aralığı ölçüldü (L = 16, 24 tohum: 0.68 … 6.44); "17.5 kat kritik
  yavaşlama" cümlesi spec'in penceresine aittir, buraya taşınmadı.
- **Bir test tam sayı (123/200) sabitliyor**; deterministiktir ama numpy'nin
  RNG akışı değişirse kırılır.
- **Sevk edilen senaryonun T = 2.0 ve T = 2.8 satırları testlerde
  kilitlenmez** (16/24 ve 13/24 — geçiş bölgesi). Tek akıştan okunan bir
  oranı fiziğe yazmak bu depoda üç kez düzeltilmiş bir hatadır.

#### Planın bayat çıkan iddiaları

| planın dediği | ölçülen |
|---|---|
| `β/ν = 0.1152` | **yeniden üretilemedi**; 0.122941 ± 0.003225 |
| Binder kesişimi **2.260**, sapma **%0.4** | o bir **ızgara noktası**; ara değerlemeyle 2.265050 / 2.265253 / 2.265570, havuzlanınca −4.18σ |
| "Metropolis plato yapmaz, Wolff yapar" | algoritma cümlesi olarak **yanlış**; Metropolis L = 8 / N = 4000'de 24/24 plato yapıyor |
| naif SEM **10.4 kat** | **8.81 kat** (200 zincirlik referans; referansın kendi belirsizliği %5.0) |
| `ξ` `T ≥ 1.0`'da kırılıyor (−0.0511 / −0.0125 / −0.0383) | işaret **tohumdan tohuma değişiyor**; doğru hüküm "tahminci **kararsızdır**", artı ~%38 sahte pozitif |
| plato ölçütü: son üç hata %5 içinde, `min_blocks = 16` | kendi zorunlu testini **21/24 kırmızı** yapardı |
| demir 1043 K için `J = 39.608394 meV` → birebir 1043.0000 K | **doğrulandı** (fark 0.000e+00) — planın yeniden üretilebilen tek sayısı |

### Modül 5 — elektromanyetik alan / Maxwell FDTD (Plan J)

#### Yol haritasının hedefi İKİ AYRI sebeple düştü

Çekirdek spec bir zamanlar *"bilinen yansıma/kırılma katsayıları, Zeeman
yarılması"* diyordu. **Hedefin iki yarısı da çürüdü, ama farklı sebeplerle** —
ve ikincisi bu depoda ilk kez görülen bir arıza türü.

**Birinci yarı — TAUTOLOJİ (tanıdık aile, üçüncü üye).** Dik gelişte
`R = ((n₁−n₂)/(n₁+n₂))²` ve `n` **girdidir**. Yeniden ölçüldü (float64,
`eps = 2.22e-16`):

| girilen `n₂` | çıkan `R` | `R`'den geri okunan `n₂` | fark |
|---|---|---|---|
| 1.33 | 0.020059312200 | 1.330000000000 | 2.220e-16 |
| **1.5** | **0.040000000000** | **1.500000000000** | **2.220e-16** |
| 3.4 | 0.297520661157 | 3.400000000000 | 4.441e-16 |

Modül 3F'nin silisyum bant aralığı ve Modül 4'ün demir `T_c`'siyle **aynı
ailenin üçüncü üyesi**. Fresnel atılmadı; **sınıfı değişti** — fizik çıpası
değil **çeviri denetimi** olarak sevk edildi (aşağıya bakın).

**İkinci yarı — YÖNTEM/HEDEF UYUŞMAZLIĞI.** Maxwell FDTD **klasiktir**: durum
vektörü `(E, H)`, güncelleme sabitleri `ε`, `μ`, `Δx`, `Δt`. İçinde ne enerji
seviyesi, ne spin, ne `ħ` vardır — bir meV yarılması bu yöntemin **çıktı
uzayında değildir**. Yöntemin gerçekten ürettiği şey manyetize Lorentz
osilatörüdür ve normal kipleri `ω± = √(ω₀² + ω_c²/4) ∓ ω_c/2`, yani yarılma
**tam `ω_c`** (üç `(ω₀, ω_c)` ikilisinde `yarılma/ω_c` = 1.000000000000).
Bu **yörüngesel `g = 1`**'dir; spin Zeeman'ı `g = 2` ister. Yani hedef,
yöntemin ürettiği şeyin **tam iki kat yanlış** bir sürümüdür. **Zeeman kapsam
dışıdır** — "yapılmadı" değil, **yapılamaz**.

#### İKİ SINIF, ve karıştırılamazlar

Bu modül iki farklı şeyi doğrular ve panel ikisini **ayrı satırda** beyan eder.
Bir şema çıpasını fizik çıpası diye sunmak bu depoda **üç kez** düzeltilmiş bir
hatadır; `Provenance.anchor_class` alanı onun panel tarafındaki karşılığıdır.

**ŞEMA ÇIPALARI — çözücüyü doğrular, fiziği DEĞİL.** Sayıları şemanın kendi
özelliğidir, girilmez, ölçülür.

| Şema çıpası | Ölçülen | Not |
|---|---|---|
| Courant `S·√d` | 1B PEC duvarda `S_max` **3.115408e-05** payla platoya oturuyor; pay **iki parçalı** — ızgara payı tam `1/cos(π/(2(L−1)))` (`L⁻²`), koşum payı `L`'den bağımsız | **Sınır tipi adlandırılmadan yazılamaz**; periyodikte 1.000005 |
| 2B Courant | `S·√2` = 1.000786 … 1.002102 (L = 32/64/128), aynı iki parçalı yapı | **1B kapısı devralınamaz:** `S = 0.9` 1B'de geçiyor, 2B'de şema patlıyor |
| Yee mertebesi | bu pencerede 4'e en yakın değer **3.999995**, sonrası 4'ten uzaklaşıyor (3.999809) | tahmincinin `arccos` tabanı |
| Dispersiyon katsayısı | `(1−S²)/C` = **24.000002 / 24.000000 / 23.999999 / 24.000001** (S = 0.3/0.5/0.7/0.9) | bant **`m` çiftine bağlı** (2,4 → 24.00033) |
| 2B köşegen | tek sayı değil **bir yasa**: `24·C(θ) = cos⁴θ + sin⁴θ − S²`, Richardson `k→0` ile **4.6e-07** içinde | 1B'nin 24'ü yasanın θ = 0 noktası |
| Sihirli adım `S = 1` | **1.332e-15** (m = 128) | sıfır **değil**, makine hassasiyeti |

**FİZİK ÇIPALARI — sürekli-ortam elektrodinamiğinin ızgaradan ORTAYA
ÇIKTIĞINI gösterir.** Dördü de `n`'den, `R`'den, `c`'den ve birim seçiminden
bağımsızdır.

| # | Çıpa | Ölçülen | Referans | Yakınsama hükmü |
|---|---|---|---|---|
| 1 | Bessel oranı `j₁,₁/j₀,₁` | **1.593293002** (−4.7504e-05) | tam **1.593340506**, simülasyona **söylenmedi** | **OTURDU(2)**, `p_inf` = 1.9625 ± 0.0853 |
| 2 | Rayleigh üssü (2B) | **p ∈ [2.924895, 3.243596]** (beş pencere) | 2B'de 3, 3B'de 4 → **4 dışlandı** | **mertebe sertifikası YOK** |
| 3 | Tek yarık `a·sin θ₁/λ` | **0.998610 ± 0.019357** (n = 14, [0.964916, 1.029080]) | tam **1** | **YARGILANAMAZ** (bilerek sevk edildi) |
| 4 | TİY faz farkı `δ_ps` | `n₂` 1.00→1.35'te yayılım **37.606652°** | kapalı form **37.350204°** (oran 1.006866) | **OTURDU(2)**, `p_inf` = 2.1028 ± 0.3364 |

**ÇEVİRİ DENETİMİ — Fresnel `|R|`.** Çözücünün arayüzü gerçekten çözdüğünü
gösterir, ama bir fizik çıpası **değildir** ve ayırma gücünün zayıflığı
ölçüldü: arayüzü **bir hücre kaydırmak** dik gelişte `|R|`'yi yalnız bağıl
**2.4e-06** oynatıyor — Fresnel bu bozmayı pratikte **görmez**. (Spec §2.3
"7 hanede birebir aynı" diyordu; ölçülen **6 hane**.)

**ÖLÇÜM ALETİ — çıpa değil.** `mode_spectrum` adsız tepe döndürür ve Bessel
köklerini **hiç görmez**; `convergence_ladder` ise çıpaların hükmünü veren
ölçüttür ve kendi karnesini taşır (426 gerçek merdiven: doğru pozitif %84.1,
yanlış pozitif %0.0 — `noise` olmadan %16.4).

#### Bessel oranları — modülün en güçlü çıpası, ve İKİ penceresi

Dairesel PEC boşluğun kip frekansları `j_{m,n}/(2πR√ε)`'dir. Modül **oranları**
okur, tekil frekansları değil — ve bu bir tercih değil bir **ölçüm sonucudur**.

Pencere: `dx = 1`, `S = 0.5`, adım `1680·R`, Ricker (`f_p = 1.8 f₀₁`), 8
dedektör, **blackman**, `pad = 8`, tepe tabanı 0.01, `ε_r = 1`, R = 80.

| oran | ölçülen | tam | sapma | genlik/max |
|---|---|---|---|---|
| j11/j01 | 1.593293002 | 1.593340506 | **−4.7504e-05** | 1.000 |
| j21/j01 | 2.135392329 | 2.135548787 | −1.5646e-04 | 0.952 |
| j02/j01 | 2.295226545 | 2.295417267 | −1.9072e-04 | 0.298 |
| j31/j01 | 2.652755373 | 2.653066405 | −3.1103e-04 | 0.519 |
| j12/j01 | 2.916874952 | 2.917295455 | −4.2050e-04 | **0.0214** |

**BİRİNCİ PENCERE — hane sayısı KİPE BAĞLIDIR: 8.85 kat.** Tek bir "beş hane"
cümlesi yazılamaz; beş sapmanın beşi de **aynı işaretlidir** ve bu
merdivenlenmenin imzasıdır.

**İKİNCİ PENCERE — merdivenlenme oranda götürüşüyor, mutlak `j`'de KALIYOR.**
Mutlak `j₀,₁` merdiveni **OTURMADI** (`p_inf` = 0.4863 ± 0.3574) — plan ve spec
`OTURDU(1)` yazıyordu ve çürüdü. Sebep bir kuvvet yasası değil **kafes
aritmetiğidir**:

    δ(R) = 0.9763·(√(N(R)/π) − R) + 0.3651     Pearson r = 0.999041, R² = 0.9981

Boşluğun gördüğü etkin yarıçap **merdivenlenmiş bölgenin alanından** okunuyor;
Gauss'un daire problemi bir kuvvet yasası olmadığı için mutlak `j` **hiçbir
tamsayı mertebeyi sertifika edemez**. Oranda `δ` tam sadeleşir. Kontrast
plandakinden keskin: **OTURMADI vs OTURDU(2)**.

Ve bu, çıpanın **körlüğünü kapatan tek testtir**: on altı mutandan ikisi
(`frekans_ekseni_dt_yok`, `maske_sinir_esit`) **yalnız** mutlak `j` testinden
düşüyor, çünkü ikisi de etkin ölçeği oynatıyor ve oranda sadeleşiyor. Oran ve
mutlak, aynı ölçümün iki yüzü olarak **birlikte** sevk edilmek zorundadır.

#### Airy sabiti — 2B'de çapalanan bir 3B sayısı, ve bir İTHALİN beyanı

`j₁,₁/π` = **1.2196335** (tam 1.2196699, bağıl sapma **2.981e-05**). Bu, dairesel
bir açıklığın Airy deseninin birinci karanlık halkasının açısal yarıçapını veren
sayıdır ve **2B boşluğun kip oranından** okunuyor.

**OKUMA BİR İTHAL TAŞIYOR VE SAKLANMIYOR:** ölçülen şey **orandır**;
`j₀,₁ = 2.4048256` teoriden gelir, bu modül onu **ölçmez** (mutlak merdiven
"OTURMADI" diyor). İthal ek hata getirmiyor — Airy'nin bağıl sapması oranın
bağıl sapmasının birebir kendisidir.

#### Rayleigh üssü — σ CİNSİNDEN MARJ YAZILAMAZ, ve sebebi ölçüldü

İnce bir silindirden saçılan genişlik `σ ∝ (a/λ)^p`, 2B'de `p = 3`, 3B'de
`p = 4`. Modül `p`'yi **fit'ten** okur ve 4'ü dışlar. Ama *"3'ten 15.1σ"*
**yazılamaz**:

    FDTD'nin σ_fit'i          0.016080
    ANALİTİK tam çözümün      0.015423      (aynı ızgara, aynı pencere)
    oran                      1.0426        artıkların işaret dizisi BİREBİR AYNI

Yani `σ_fit` bir belirsizlik değil **pencerenin eğriliğidir**; artığın %96'sı
gürültüsüz bir eğriden geliyor ve tam çözüm de 3'ten 15.5σ uzakta. Hüküm bu
yüzden **aralık** olarak yazıldı: beş pencerede `p ∈ [2.924895, 3.243596]`,
**aralık genişliği 0.319 < 4'e uzaklık 0.756**.

**λ ETİKETLEME — merdivenin göremediği bir arıza.** Nominal `λ` istendiğinde
FFT binleri kayıyor (en büyük %4.17) ve üs 3.290854 yerine 3.248311 çıkıyor.
Merdiven bunu **görmüyor**: dört basamakta hata +0.0294/+0.0324/+0.0318/+0.0316
ve mertebeler −0.4295/+0.0854/+0.0225 — sıfıra yakınsamıyor, **ilk adımda
büyüyor**. Bu yüzden **ayrı bir test** kilitliyor.

#### İki uzak alan — "iki yöntem" değil, İKİ AYRI GÖZLEMLENEBİLİR

Ayrım ortalamada değil **kontur bağımlılığındadır** ve iki yönde de ölçüldü:

- **Yarıkta** kutu yarı yüksekliği 40 → 160 iken kapalı kutu yayılımı
  **1.743e-02**, `aperture_far_field` **4.460e-04** → **39 kat**.
- **Saçılmada tersi:** NTFF'in tam çözümden RMS farkı 4.0e-04 … 2.5e-03, tek
  satırın FT'sininki **2.121e-01 … 4.678e-01** → **100–500 kat**.

**Fraunhofer kapısı düzeldi:** `R ≥ a²/λ` **değil**, `R ≫ λ`. `R = 10λ`'da eski
kapı 4000–25000 kat rahat geçiyor ama desen **%14.5–37.4 yanlış**. Sevk edilen
eşik `R ≥ 100λ` ve eşikteki artık hata bir tolerans değil **eşiğin bedelidir**.
Kapı ihlalinde sessizce yanlış açı basılmaz, **`None`** döner.

#### Yakınsama merdiveni — deponun ilk yakınsama ölçütü, ve ilk kuralı

Her fizik sayısı değeriyle birlikte bir **yakınsama kaydı** döndürmek
zorundadır (pazarlık edilmez). Ölçüt üç değerlidir — `OTURDU(p)` / `OTURMADI` /
**`YARGILANAMAZ`** — ve sonuncusu Modül 4'ün `correlation_length → None`'ıyla
aynı statüdedir: **yargılayamadığını yargılamış gibi gösterme.**

**DERİNLİK EN AZ DÖRT, ve bu bir SAYIM KİMLİĞİDİR:** n değer → n−1 fark → n−2
mertebe → **n−3 Richardson noktası**. Dört, `u`'nun ölçülebildiği en küçük
derinliktir (426 merdivende doğrulandı: derinlik 3'te 426/426 `p_inf = None`).

**KARNE — 426 GERÇEK merdiven, sentetik yok:**

| | doğru pozitif | yanlış pozitif |
|---|---|---|
| `noise` ile (sevk edilen) | **%84.1** (201/239) | **%0.0** (0/426) |
| `noise = 0` | %84.1 | **%16.4** (70/426) |

Yer gerçeği bilinen sürekli limitten okunuyor ve **tahminci limiti hiç
görmüyor** — karne dairesel değil. `κ = 0.5` bir tolerans değil **iki tamsayı
mertebe arasındaki karar sınırıdır** ve 0.40–0.60 arasında düz bir plato var.

**σ TEK BAŞINA ÖLÇÜT DEĞİLDİR** — ve bu iki bağımsız olayda görüldü: doğru
çözücünün kendi sistematiği `δ_ps`'te **71σ**, ve λ = 16'da bir mutant doğru
çözücüden **daha küçük** sapma veriyor. Aynısı Bessel'de: `pad = 1` mutantı tek
ızgarada beş oranın **dördünde** doğru borudan daha yakın çıkıyor (j31'de
**6.6 kat**), ama merdiven onu **YARGILANAMAZ** ile yakalıyor. Tersi de ölçüldü:
`dedektör = 1`'de **merdiven hiç değişmiyor** (1e-09) ama **panel düşüyor**
(tepe 16 → 11). **Merdiven ve panel farklı şeyler yakalıyor; ikisi de gerekli.**

#### Modüller arası bağ — VAR, ama iki bağın ikisi aynı güçte DEĞİL

**BİRİNCİL — modülün KENDİ BOYUT MERDİVENİ, ve tamamen kendi içinde
ölçülebilir.** İki sayı da `d`'ye bağlıdır ve `d` **ayrıktır** — sürekli bir
düğmeye çevrilemez, yani bu bağ bir parametre taraması değil bir **sınıf
ayrımıdır** (Modül 4'ün evrensellik-sınıfı kontrastının akrabası):

- `Courant = 1/√d`: 1B'de 1, 2B'de 1/√2 — **ölçüldü ve 1B kapısının 2B'de
  patladığı gösterildi**.
- Rayleigh üssü: 2B'de **3** (ölçülen aralık [2.924895, 3.243596]), 3B'de 4 —
  ve **4 dışlandı**.

**İKİNCİL — Maxwell ↔ Schrödinger `v_g/v_p`, ve BAĞIN YARISI ÇEVİRİ
DENETİMİDİR.** Aynı gerçek-uzay tahmincisiyle Maxwell (kütlesiz) 1, deponun
split-operator'ü (kütleli) 2 verir. **Ama bağ simetrik değildir ve bu bir
inceleme sonucudur:** Yee'nin `ω(k)`'sı **ortaya çıkıyor** — `engine/em/`
ağacının hiçbir yerinde bir dispersiyon bağıntısı yazılı değil, çekirdek yalnız
curl güncellemesi yapıyor; split-operator'ünki ise `exp(−i k²Δt/2m)` olarak
**koda yazılıdır**. Yani Schrödinger tarafında `v_g` bir ölçüm değil bir geri
okumadır; **yalnız `v_p` (tepe izleme) gerçek bir ölçümdür.**

**DÜRÜST OLUMSUZ:** bu bağın sayıları Plan J'nin ön ölçümünden gelir ve **bu
modülde YENİDEN ÖLÇÜLMEDİ**; test olarak da sevk edilmedi (Modül 1'in
çekirdeğine bağımlılık kurmak katman sözleşmesini zorlar ve kazancı yarım).
Bu yüzden yukarıda **sayı yazılmadı** — yazılsaydı ölçülmemiş bir sayı
ölçülmüş gibi görünürdü.

#### Panel ve senaryo

```bash
python -m qlab.run scenarios/em_tarama.yaml -o em.npz
```

Çıkış 0; panel **dokuz** satır basar — Yöntem / Yaklaşımlar / İhmal edilen /
Beklenen hata / **Dairesellik** / **Çıpa sınıfı** / **Pencere** / Yakınsama /
**Yakınsama tanısı**. Ortadaki üçü bu modülde `Provenance`a eklendi ve ayrı
alan olmalarının gerekçesi `provenance.py`de yazılı:

- **Çıpa sınıfı** `Yaklaşımlar` listesine konsaydı, `as_lines()` o listeyi
  `" · "` ile birleştirdiği için dört sınıf etiketi yaklaşımların arasında akıp
  giderdi.
- **Pencere** bir yaklaşım da bir hata sınırı da değildir: hata sınırının
  **nerede geçerli olduğunu** söyler, ve bu depoda bir hükmün penceresinin
  dışına taşınması **on kez** ölçüldü.
- **Yakınsama tanısı** `diagnostics`e konsaydı panelde **"✓ …
  verdict=YARGILANAMAZ"** diye görünürdü — yani yargılanamadığı söylenen bir
  sayının yanında bir onay işareti. Modül 4'ün `sampling` alanıyla aynı
  gerekçe, ondan daha sert.

`Yakınsama` burada "hesap bitti" **demek değildir** — FDTD'de böyle bir ölçüt
yoktur. Bağlandığı şey şudur: **bu koşumun merdiveni bir tamsayı mertebeye
oturdu mu?** Sevk edilen senaryoda ✓ basar ve bu doğrudur.

**TOHUM YERİNE PENCERE.** FDTD deterministtir; RNG hiç kullanılmaz. Bu yüzden
`em_tarama.yaml` bir tohum taşımaz ama pencerenin **her düğmesini** taşır:
ızgara (`radii` + `margin`), `S`, adım (`step_factor`), `λ`
(`wavelength_per_radius`), `sgf`, FFT penceresi, `pad`, tepe tabanı, dedektör
sayısı, `eps_r`.

**İKİ DÜĞME BİLİNÇLİ OLARAK ANAHTAR DEĞİLDİR — VE SEBEPLERİ FARKLI.** `npml`
**bağlı değildir**: dairesel PEC boşluk tam yansıtıcıdır ve
`CircularPEC.thickness` sabit sıfırdır. `Δx` ise **bağlıydı ama sessizce
yanlış çalışıyordu ve bu ölçüldü**: `run_2d` kaynağı `t = n·S·Δx` zamanında
besliyordu, oysa `wavelength` **hücre** cinsinden verilir ve ikisi yalnız
`Δx = 1`'de örtüşür. R = 5/10 penceresinde tepe sayısı `Δx` = 1'de **20/13**,
`Δx` = 0.5'te **4/4**, `Δx` = 2'de **31/43** çıkıyordu — `λ` üçünde de 7.257613
kalarak. Yani kaynağın bandı kipin frekansına göre kayıyordu (26.08.2026'da
kapandı: `dx_kapisi`, bkz. aşağıdaki kapanış maddesi). Bağlanmamış ya da sessizce yanlış
çalışan bir düğmeyi kabul etmek bu depoda düzeltilmiş bir hatadır
(`heisenberg_step` + Ising). Panel ikisini de yine basar (`npml = 0`,
`dx = 1`), değerleri `Window`dan gelerek.

Sevk edilen koşum (ölçüldü, 26.08.2026: FDTD 0.95 s + analiz 0.54 s = **1.49 s**):

    R      ızgara    adım    oran            tepe   gürültü
    5      17 x 17   2100    1.578632448506  28     5.04e-06
    10     27 x 27   4200    1.589868261123  21     7.08e-06
    20     47 x 47   8400    1.592538328891  17     1.95e-06
    40     87 x 87   16800   1.593146492961  15     2.71e-06
    p_inf = 2.195532 ± 0.061188  ->  OTURDU(2)

Hüküm **adım çarpanına duyarsız** (420/840/1260/1680 → 2.195532 / 2.184666 /
2.191308 / 2.191648, dördünde de OTURDU(2)), yani ölçülen şey kaydın uzunluğu
değil **merdivenlenmenin mertebesidir**.

**KABA BASAMAK PANELİN KENDİ VİTRİNİ:** R = 5'te beş Bessel oranından yalnız
**ikisi** %2'lik eşleme penceresine giriyor (j02/j31/j12 sırasıyla −2.82% /
−3.23% / +2.80% ile dışarıda), R = 40'ta **beşi de** giriyor. Yani "hane sayısı
kipe bağlıdır" cümlesi aynı senaryonun iki ucundan okunuyor.

**SEVK EDİLEN BORU HATTI TEORİSİZDİR.** `viz/em_frames.py` `j₁,₁/j₀,₁` diye bir
şey bilmez; okuduğu şey **ikinci en düşük tepenin en düşüğe oranıdır** ve `src/`
ağacında hiçbir Bessel kökü geçmez. İki tanımın sevk edilen pencerede **birebir
aynı** sayıyı verdiği testle ölçülüyor.

Beyan metninin **tek kaynağı** `engine/em/factory.py`dir ve bu bir disiplin
değil bir **test**tir: yirmi beyan girdisinin her biri `src/` ağacında tam bir
kez geçer, `tests/` tarafında da kilidi kuran tek dosya dışında hiç geçmez.
`run.py::_run_em` metnin bir harfini bile yeniden yazmaz; tek kelimelik istisna
`HUKUM_VERILMEDI` işaretidir ve o da yazılmaz, fabrikadan **ithal edilir**.

#### Dokstring denetimi — ALTINCI kez ürün verdi

Bu depoda "sevk edilmiş dokstringde çürütülmüş iddia ara" adımı beş modülde beş
kez ürün vermişti; Task 6'da altıncı kez verdi ve **ikisi de senaryo/panel sevk
edilince sınanabilir hâle gelen iddialardı**:

1. **`test_bessel.py`: "kaba uçta sınır FİZİKSELDİR (R = 5'te tepe kapısı kipi
   bulamıyor)".** Senaryonun merdiveni R = 5'e uzatılınca ilk kez ölçülebildi ve
   **çürüdü**: R = 5'te aynı boruyla **28 tepe** bulunuyor (R = 40'takinden
   fazla — merdivenlenme dejenere ikilileri yarıyor) ve merdivenin okuduğu sayı
   `j11/j01` **bulunuyor** (1.578632448506). Düşen şey tepe kapısı değil
   **%2'lik eşleme penceresidir**; üç oranın genlikleri 0.57/0.82/0.30 ile
   tabandan çok yukarıda. Düzeltildi ve `test_em_senaryo.py`de kilitlendi.
2. **`mode_spectrum`: "ara değerleme kalkınca … merdiven mertebesi düşer".**
   Bu cümle Task 4'te **kendi testi tarafından** çürütülmüştü
   (`test_ARA_DEGERLEME_KALKINCA_HEM_MERDIVEN_HEM_PANEL_DUSUYOR`: mertebeler
   düşmüyor, **oynuyor** — p = 3.3411 ve 1.4245, biri 2'nin **üstünde** — ve
   belirsizlik patlıyor, u = 10.0255 → **YARGILANAMAZ**) ama dokstring
   düzeltilmemişti ve **çürüten testin kendisine atıf yapıyordu**. Düzeltildi.
3. **`DEFAULT_KAPPA`'nın "panelde öyle yazılır" sözü ÖDENDİ.** `κ = 0.5`'in bir
   tolerans değil bir modelleme seçimi olduğu artık panelin `Pencere` satırında
   yazılı ve testle kilitli.
4. **`Δx` SESSİZCE YANLIŞ ÇALIŞAN BİR DÜĞMEDİR** — ve bu, senaryo şeması
   yazılırken bulundu. `run_2d` kaynağı `t = n·S·Δx` zamanında besler ama
   `wavelength` **hücre** cinsinden verilir; ikisi yalnız `Δx = 1`'de örtüşür.
   Ölçüldü (R = 5/10): tepe sayısı **20/13 → 4/4 → 31/43** (`Δx` = 1/0.5/2),
   `λ` üçünde de 7.257613 kalarak. Düğme senaryodan **çıkarıldı** (`npml` ile
   aynı muamele, farklı gerekçe) ve reddedildiği testle kilitlendi. O turda
   **motor düzeltilmemişti** ve dürüst olumsuz olarak burada duruyordu;
   **26.08.2026'da kapatıldı.** Ölçüldüğünde `Δx` bir düğme ÇIKMADI:
   davranış yalnız `wavelength/Δx` oranına bağlı ve bu 1B'de beş, 2B'de
   dört, eğikte dört `(Δx, W)` çiftinde **bit bit** aynı; kaynaksız koşumda
   ise `Δx` hiçbir şeye dokunmuyor. `engine/em/grid.dx_kapisi` artık
   `Δx ≠ 1`'i reddediyor, `t = n·S·Δx` üç koşum yolunda da `t = n·S` oldu ve
   "lambda … hucre" beyanı ilk kez sınanabilir hâle geldi
   (`tests/em/test_hucre_birimi.py`).
5. **Kararlılık kapıları motor yolunda eksikti — 29.09–02.10.2026'da kapandı.**
   Yükleyici ve fabrika `eps_r < 1`'de etkin `S/√eps_r`'yi denetliyordu
   (`5d9a0bf`), ama üç katsayı yolu (`yee_coefficients`,
   `yee_coefficients_2d`, `oblique_coefficients`) `eps_r·mu_r < 1` ortamında
   patlayan bir bandı KABUL ediyordu — public yolla ölçüldü: 1B (0.5, 1], 2B
   (0.354, 0.707], eğik `ky = 0.8` (0.464, 0.928] (`eps` ya da `mu` 0.25).
   `yee._etkin_courant_kapisi` hücre başına `eps_r·mu_r`yi okur (`c4c66ad`);
   karışık ortamda eşik 0.500019 ölçüldü, minimumların çarpımı 0.25 derdi.
   Eğik p'de `ey_kural` ≠ aritmetik ve konuma bağlı `mu_r` ile o kapı da
   yetmiyordu (her hücrede `eps·mu = 1` iken `eps 4|0.25`, `L = 200`: harmonik
   0.627587, sağ 0.470588); `oblique._egik_p_kesin_esik` kesin eşiği ayrık
   operatörün en büyük özdeğerinden okur (`87337f0`).

#### Dürüst olumsuzlar

- **Tek yarık çıpası iki hanelik** (±%1.9) **ve eğiklik çarpanına KÖR:**
  `cos θ` kaldırılınca tek tek fark yalnız 5.2551e-05 … 2.9382e-03, yani
  çıpanın kendi yayılımının (0.019357) altında. Çarpanın doğruluğu **beyan
  ediliyor, ölçülmüyor** — RS-II'nin 1'i ya da Kirchhoff'un `(1+cos θ)/2`'si
  konsaydı bu test görmezdi.
- **Rayleigh üssünün MERTEBE SERTİFİKASI YOK** (merdiven YARGILANAMAZ) ve σ
  cinsinden marj yazılamıyor; hüküm yalnızca aralık olarak duruyor. Üs `a` ve
  `ε_r`'den bu pencerede tam bağımsız da değil (3.2429/3.0933/2.9249 — üçünde de
  tam çözümle 0.0185 içinde, yani oynayan çözücü değil pencere).
- ~~**`courant_limit` EĞİK YOLU SINAMIYOR.**~~ **KAPANDI 26.08.2026.** Eğik
  şemanın kararlılık sınırı `k_y` ile düşüyor ve `courant_limit(1)` bunu
  görmüyordu: `k_y = 1.2`'de `S = 0.9` kapıyı **geçiyordu** ve şema
  patlıyordu (ölçüldü: 800 adımda `|e|` **4.22e+216** katına çıkıyor,
  başlangıcın 1e12 katını **46. adımda** aşıyor — oysa aynı `S`, `k_y = 0`'da
  KARARLI, büyüme 1.163606). Artık `oblique.oblique_courant_limit(k_y)` =
  `1/√(1+(k_y/2)²)` ikinci bir kapı olarak `oblique_coefficients`te duruyor;
  `k_y = 0`'da tam olarak `courant_limit(1)` = 1 döndürüyor, yani izotropik
  kapıyı **genelliyor**. Kapalı formun sonlu-PEC hâli
  `1/√(cos(π/(2(L−1)))² + (k_y/2)²)`'dir ve ölçülenin ona oranı yedi `k_y`
  değerinde **1.00015-1.00016**; artan pay `adım^-2.08` ile sönüyor
  (`grid.courant_limit`in izotropik ölçümüyle aynı: 5× adımda 28.2×).
  Not: kapalı formun etiketi bir zamanlar `k_y Δx` idi — `Δx` o terime hiç
  girmez, `k_y` **radyan/hücre**dir.
- ~~**`oblique.py` TAM 2B İLE KARŞILAŞTIRILMADI.**~~ **s (TM_z) kolu KAPANDI
  09.10.2026** — kod düzeltmesi yok; ölçülmüş bedel beyanı (`oblique.py` modül
  dokstringi) ve üç kilit testi (`tests/em/test_g1_egik_2b.py`: T1 kimlik ve T2
  kapalı-form bedel hızlı şeritte, T3 kritik bant `slow`). 1B-açılı indirgeme
  bir modelleme seçimi olarak kalıyor (`k_y` analitik dayatılıyor); bedel
  tanımı: aynı `ω`, aynı ANALİTİK `k_y = k0·n1·sin θ`. Ölçülen:
  - **Kimlik:** y-değişmez ortamda motorun 2B TM_z Yee'si, eğik s'nin
    `k_y → 2 sin(k_y/2)` konmuş hâlidir, bit-yakın. Yayılan kiplerde
    (taşıyıcıda `K < √ε_kaynak·Ω`) zaman kaydında tepeye göre ≤ 1.5e-14:
    ε 1→2.25, S 0.4 en büyük 1.414e-14; 4→1, S 0.6, m = 1, 2'de 5.64e-15.
    ε 1→4 (S 0.5, 50°) yalnız **tek frekansta** ölçüldü:
    `|ρ_2B − ρ_eğik(K)|` = 6.6e-16 (tepeye göre değil). 4→1'de m = 3 kısmen
    sönümlü (taşıyıcıda K 1.111 > 0.895, ama geniş bantlı darbenin üst
    kuyruğu yayılıyor; tepe 1.5e-4): 2.9e-13; tam sönümlü m ≥ 4 (tepe ~1e-8)
    bağıl 2.8e-9.
  - **Kapalı form:** genlik bedeli = `|r(k_y)| − |r(2 sin(k_y/2))|` (karmaşık
    fark değil; faz bedeli ayrı bir büyüklüktür ve yalnız referans düzlemiyle
    yazılır, aşağıda TİY maddesi), arayüz yüzüne (I−½)
    referanslı ayrık Fresnel `r_s = sin((k1−k2)/2) / sin((k1+k2)/2)`. Sabit
    açıda O(Δx²); sevk edilen aletle (ε 1→2.25, λ = 20, S 0.4) 30/45/60°:
    **1.00e-4 / 6.62e-4 / 3.13e-3** (kapalı formla ≤ 1.7e-6 içinde). 75°'de
    aletin kendi protokol hatası (2.0e-2) bedelden (1.47e-2) büyük; orada
    bedeli bu alet çözemiyor. **10.10.2026:** alet orada artık `None`
    döndürüyor (kuyruk kapısı, aşağıdaki madde).
  - **Kritik açı bandı (n1 > n2):** iki şemanın kritik açısı ayrılıyor; bantta
    eğik `|r| = 1`, 2B kısmi yansıma veriyor. ε 2.25→1, λ = 20'de bant
    **0.21°**, bedel kapalı formda **0.147**'ye kadar (çözücüyle 41.8°'de
    0.140) ve yalnız **O(Δx)** iniyor (λ 10→160 üsleri 0.82→0.98). "O(Δx²)"
    noktasaldır, açıda düzgün değildir.
  - **TİY faz bedeli yalnız referans düzlemiyle yazılır:** ε 2.25→1, 60°,
    λ = 20.78'de `|arg(r_eğik / r_2B)|` I−½'de **1.99°**, prob düzleminde
    **4.73°**. `reflection_coefficient` prob referanslı okuyor (dokstringinde
    beyan edildi). **Faz sayıları kilitsiz:** T2 ve T3 yalnız `|r|`'ye bakıyor;
    kapalı formun işareti de kilitsiz (kapalı formun payında işaret
    çevrilince üç test de geçiyor, doğrulayıcı ölçtü).
  - Sürekli Fresnel'e hangi şemanın daha yakın olduğu açıyla işaret
    değiştiriyor (kapalı form: 1→2.25'te ~75°, 1→4'te ~80°); bu yüzden bir
    "daha doğru şema" hükmü verilmiyor.
  - Beş mutantın (A eğik→`2 sin(k_y/2)`, B yardımcı alan sırası, C yardımcı
    alanın PML kaybı, D 2B Hx işareti, G enine terimde komşu `cb`) hepsini T1
    öldürüyor (A, B, G'yi T2 ve T3 de). G1'den önce G'yi hızlı şeritte yalnız
    bir kararlılık kapısı görüyordu; fizik referanslı hızlı test yoktu.
  - **AÇIK ALT MADDE — p (TE_z) tam 2B ile KARŞILAŞTIRILMADI.** Motorun 2B
    yolu yalnız TM_z (`yee._adim_2d`: Ez, Hx, Hy); G1'in p kolu açıktır.
    Kontrast 1↔2.25, 1→4, 4→1 ve S 0.4/0.5/0.6 dışı da ölçülmedi.
- ~~**`reflection_coefficient` sıyırma / kritik-yakını açılarda SESSİZ
  protokol hatası.**~~ **KAPANDI 10.10.2026 (sevk geometrisinde) — kuyruk
  kapısı** (kayıt tamlığı); kayıt içi yankı ayrı bir açık madde (aşağıda). Kaydın son periyodundaki genlik (gelen ya da yansıyan seride)
  tepenin `tail_gate` = 1.6e-3 katını aşarsa alet `None` döndürüyor; yeni
  koşum yok, geçen her sayı eski sayının bit bit aynısı. Ölçülen
  (`.superpowers/sdd/rc-olcum/`, ayrık-tam referansa karşı: s'te G1 kapalı
  formu, p'de transfer özyinelemesi):
  - **Kaynak ayrıldı** (G1'in iki durumunda): FFT bini ve pencere suçsuz
    (bin = tam-f DTFT, ≤ 2.9e-15; Hann her yerde daha kötü), PML suçsuz
    (npml 60→200, r0 1e-6→1e-12 T_min'de ρ'yu ≤ 2.3e-7 oynatıyor). Suç kayıt uzunluğunda:
    darbe kayıt bittiğinde hâlâ geliyor. Sebepler: sabit-`k_y`
    indirgemesinin grup hızı dağılımı ve kesim frekansına yakınlık
    (sıyırma; 1→2.25, λ = 20, 75°'de darbe 4.9 kat genişliyor), dal
    noktasının uzun yansıma kuyruğu (kritik açı), kaba ızgarada şemanın
    yavaş grup hızı (ε = 4, λ = 10'da 0.393, sürekli 0.5 — **θ = 0'da bile**
    hata 1.09e-1) ve kaynağın sola giden dalgasının soğurucudan dönüşü.
    `_record_steps` bunları düzeltmiyor; sürekli `vx` ve dağılmamış `sg`
    ile hesaplıyor (dokstringinde beyanlı).
  - **Izgara (sevk penceresi, 4 kontrast × 56 açı × λ 10/20/40 × s/p = 336
    durum):** eski alet 253 sayı döndürüyordu, 157'sinin hatası > 1e-4,
    99'unun > 1e-2 (en kötü 0.395). Kapıyla 89 sayı, en kötü hata
    **8.28e-5** (ızgara noktalarında; bağımsız doğrulayıcı aynı pencerede
    ızgara dışında 1.49e-4 ölçtü, 2.25→1, 44°, λ = 30). 164 yeni `None`'un bir kısmı aslında temizdi (eski hata
    2.96e-6'ya kadar): bedel yanlış alarm, sessiz hata değil.
  - **Eşik banttan:** depodaki sabitlenmiş çağrıların en büyük kuyruğu
    9.44e-4, sessiz hatası > 1e-4 olanların en küçüğü 2.78e-3; eşik
    geometrik orta. **Kapı bir kanıt değil, ölçülmüş bir oran — ve yalnız
    prob arayüze yakınken (sevk ofseti 6):** hata/kuyruk ızgarada ≤ 0.11
    (ızgara dışı 0.15), depoda en kötü 0.54 (bir yankı kayda tepesinden
    giriyor) → orada geçen sayının protokol hatası ≲ 0.54·τ ≈ 8.6e-4 diye
    okunur. Küçük T2 penceresinde geçen iki sayı 1e-4'ü aşıyor
    (2.25→1, 45°: 2.05e-4 / 1.82e-4).
  - **Açı ve çözünürlük** (sevk penceresinde `None`): n1 = 1'de λ = 10
    θ ≥ 60 (1→2.25 p'de 55° de, Brewster yakınında yanlış alarm), λ = 20
    θ ≥ 65; kritik bant (2.25→1: 40–43°, 4→1: 29.5–30.5°);
    ε1 = 4, λ = 10'da her açı; λ = 40, n1 = 1'de θ ≥ 45 (kaynak yankısı).
    Geometri ve `sgf` değişince sınırlar kayıyor.
  - **Elenen düzeltme adayı:** yankıya kadar uzun koşup kuyruğu eşiğin
    altına inen ilk kaydı almak. Cıvıltılı ve dal noktası kuyrukları bir
    periyot susup geri geliyor (τ = 3e-4'te kabul edilen bir sayının hatası
    3.35e-3). **Açık:** dağılım / ayrık grup hızıyla düzeltilmiş bir
    `_record_steps` ölçülmedi.
  - Kilitler: `tests/em/test_kayit_tamligi.py` (hızlı şeritte 13 test,
    her biri ≤ 0.5 s; sevk penceresindeki G1 başlıkları `slow`); birleşik
    mutasyon 23/24 (hayatta kalan `>` → `>=`, eşdeğer). Kilit dosyası
    olmadan em hızlı şeridi kapı mutantlarının hiçbirini görmüyor. Kapı
    kapalıyken ham sayıları gösteren `test_arayuz.py` testleri artık iki
    kapıyı da açıkça kapatıyor (`unity_gate` 1e9, `tail_gate` 1). Körlük 2
    yarıya indi: λ = 54'te soğurucusuz kayıt kapıda `None`.
- **AÇIK — kayıt içi soğurucu yankısı kuyruk kapısına görünmüyor** (orta,
  10.10.2026, bağımsız doğrulayıcı buldu). Prob arayüzden uzaksa kaynağın
  sol PML yankısı gelen kayda TAMAMEN giriyor, yansıması kayıt bitmeden
  yansıyana varmıyor; kaydın sonu sessiz, kapı sayıyı geçiriyor. 9→1, 30°,
  λ = 40: sevk penceresinde `probe_offset` 590'da hata **5.45e-3**, T2
  penceresinde ofset 400'de **6.20e-3**; r0 1e-6→1e-12 ile 8.1e-5'e iniyor.
  Sağ taraf kısaysa geçen dalganın sağ PML yankısı da giriyor (cells 1600,
  2.25→9: 8.21e-3). Testle kilitli dürüst olumsuz (kapı düzeltilince
  kırmızıya döner). Ölçülmüş aday, uygulanmadı: gelen kayıtta ana darbeden
  sonraki ikinci varışın genliği (hatayla aynı mertebe).
- **`-m "not slow"` koşumunda modülün doğa çıpalarının TAMAMI atlanıyor.**
  İşaret fixture yüzünden geniş (Bessel tarafında `kayitlar`ı tüketen 11
  test), ve Rayleigh/yarık tarafı da `slow`. G7 (09.10.2026) hızlı şeride
  kapalı form referanslı iki ucuz ısırık ekledi (`tests/em/test_g7_hizli_cipa.py`:
  dielektrik dolu dikdörtgen kovuk ve dik geliş Fresnel |r| + işareti); bütün
  hızlı 2B koşumları eps_r = 1'de olduğu için 2B kırılma indisi mutantı
  (eps → √eps) ancak böyle görünür. Modül 5'in sözlüğünde bunlar şema çıpası /
  çeviri denetimidir, doğa çıpalarının yerini TUTMAZ. Karşılığında aletin kendi kapıları ucuz
  fixture'lara ayrıldı ve `kind: em` senaryosunun uçtan uca dalı **bilinçli
  olarak `slow` bırakılmadı** — yoksa "not slow" koşumu senaryo sınırını hiç
  sınamazdı.
- **`m = 2` ikilisinin merdivenlenmeden gelen YARILMASI kovalanmadı.**
  Merdivenlenmiş daire C4v'dir ve `m = 2` ikilisi ayrılıyor (R = 20'de
  2.126400 / 2.139935, tam değer **ikisinin arasında**); sevk edilen ölçüm
  ikisinden birini seçiyor.
- **Beş oranın bantları (1e-4 … 7e-4) birer TOLERANSTIR** (ölçülenin ~2 katı);
  hüküm onlarda değil merdivendedir.
- **Eşleme (%2) TEORİYİ KULLANIR.** `mode_spectrum` kökleri görmüyor ama hangi
  tepenin hangi kip olduğu Bessel köklerine en yakınlıkla seçiliyor. Eşleme
  penceresi hükmün 200 katı olduğu için değer üretemez — ama bu bir
  **argümandır**, ölçüm değil.
- **`δ_ps`'in ÜÇ KÖRLÜĞÜ ölçüldü ve kilitlendi:** `kaydir` (1.332e-06°),
  `sinir_pec` (λ = 40'ta 0.033σ) ve `kaynak_cift` (birebir no-op, çünkü `ρ` bir
  orandır). Ve "θ ≥ 75°'de R > 1" cümlesi **yeniden üretilemedi**: bozulma θ'da
  monoton değil, `sgf`'nin fonksiyonu.
- **Tam çözümün ÜÇÜNCÜ bir bağımsız sertifikası yok**; optik teorem kapısı
  bilinçli kullanılmadı (tautoloji olduğu testle gösterildi).
- **2B soğurucuda `d = 40`'ta 45°/0° oranı tersine dönüyor** (0.11) ve sebebi
  kovalanmadı. Ayrıca "d = 15 yeter" bir **1B hükmüdür**: 2B'de aynı kalan
  genliğe inmek için `d = 40` gerekiyor (**10 kat kötü**).
- **Yalnız `r = 2` ve `r = 1.5` sınandı**; başka bir ızgara oranı denenmedi.

#### Planın bayat çıkan iddiaları

Plan J'nin **otuz altı** sayısı ölçümle düzeltildi (Task 1'de dokuz, 2'de on,
3'te altı, 4'te altı, 5'te beş). En öğreticileri:

| planın/spec'in dediği | ölçülen |
|---|---|
| Courant aşımı **+2.1e-7**, "sonlu koşum artefaktı" | aşım **iki parçalı**; o sayı yalnız periyodik ölçümdü ve **sınır tipini adlandırmıyordu** |
| Yee mertebesi **4.000009** | bu pencerede en yakın **3.999995**, sonrası 4'ten uzaklaşıyor |
| Sihirli adımda **0.000e+00** | **1.332e-15** — sıfır değil, makine hassasiyeti |
| `ricker(..., sgf)` bir bant genişliği düğmesi | Ricker'ın bağıl bandı `f_p`'den **bağımsız** (dört λ'da 0.822754, değişim 0.0e+00); 1560 katlık ölçüm **Gauss×cos**'a ait |
| PML `d = 12` geçmiyor | hüküm **`m` ile birlikte** verilmiş; `m = 1,2`'de d = 12 **geçiyor**, sevk edilen sayı `d = 15` |
| `std(p_inf'ler)` terimi belirsizliğe katkı verir | derinlik 4'te **cebirsel olarak ölü** (tek Richardson noktası) |
| yarım-tamsayı kör noktası ölçütün sınırının üstünde | `noise = 0`da bir **bıçak sırtı**; `noise`'un zorunlu olmasının **üçüncü** gerekçesi |
| θ ≥ 75°'de R > 1 (1.006/1.306/1.933) | **yeniden üretilemedi**; bölge `sgf`'nin fonksiyonu ve θ'da **monoton değil** |
| PEC'te `dt_carpan` 116σ, "iki kat daha kör" | sevk edilen aynada **0.0σ** — iki kat değil **sonsuz kat** kör |
| `abs(r) = 1` çıpası `sinir_pec`i 4.2σ ile yakalar | ölçülen **0.11σ** |
| `kaydir` `abs(R)`'de 7 hanede birebir aynı | **6 hane** (bağıl 2.4e-06) |
| mutlak `j` için **OTURDU(1)** | **OTURMADI** (p_inf = 0.4863 ± 0.3574); sebep kafes aritmetiği |
| `24·C_köşegen` = 0.0000000 **tam**, eksende 0.5005 | tek sayı değil **bir yasa**; "tam sıfır" bir **ızgara noktası**, eksen değeri sevk edilen çekirdekle **ölçülemez** |
| Hann + `pad = 8` penceresi | **sevk edilemez** — Hann'ın yan lobu 0.026707 > en zayıf kipin genliği 0.0214; blackman sevk edildi |
| ara değerleme kalkarsa **mertebe düşer** | mertebe düşmüyor, **oynuyor**; belirsizlik patlıyor → **YARGILANAMAZ** |
| 3B maliyeti **0.60 GiB** | üretilemedi; D = 80'de altı alanın ayak izi **23.4 MiB** (aynı pencerede D = 167 → 213.2 MiB, ~1920 kat süre) |
| kapalı kutu "dört kat kötü" (std 0.065/0.077) | sevk edilen köşe döşemesiyle **0.016353**; planın sayısına yakın olan şey **köşesi düşürülmüş** kutu (0.302356) — sayı çürütülmekle kalmadı, **nereden geldiği teşhis edildi** |
| Fraunhofer kapısı `R ≥ a²/λ` | **yetersiz**; `R = 10λ`'da kapı 4000–25000 kat rahat geçerken desen %14.5–37.4 yanlış → `R ≥ 100λ` |
| `ε_r` 1.0 → 2.25'te oranlar 8e-6'da aynı | kayma R ile **dört kat** iniyor → bu bir fizik çıpası değil **şema ölçümü** |
| **R = 5'te tepe kapısı kipi bulamıyor** (sevk edilmiş test dokstringi) | **28 tepe** bulunuyor ve `j11/j01` eşleşiyor; düşen şey **%2'lik eşleme penceresi** |

### Modül 6 — yerçekimi ve uzay-zaman (Plan K)

#### Yol haritasının hedefi bir TAUTOLOJİYDİ — ve ölçüldü

Yol haritası *"Merkür'ün perihel kayması (43″/yüzyıl)"* diyordu. **Bu bir
öngörü değildir.** Kayma `Δϖ = 6πGM/(c²a(1−e²))` ile verilir ve `M`, `a`, `e`
GİRDİDİR; dolayısıyla 43″ üç düğmenin **her birinden bire bir geri okunur.**
Ölçüldü (float64, bu depoda):

| bozulma | sonuç oranı | kapalı form | bağıl fark |
|---|---|---|---|
| `M` × 1.01 | 1.0099999999999996 | 1.01 | 4.40e-16 |
| `a` × 1.01 | 0.99009900990099 | 1/1.01 | 1.12e-16 |
| `e`: 0.20563069 → 0.3 | 1.0524351860770593 | (1−e₀²)/(1−e²) = 1.052435186077059 | 2.11e-16 |

Modül 3F'nin silisyum bant aralığı, Modül 4'ün demir `T_c`'si ve Modül 5'in
Fresnel `R`'si ile **aynı ailenin dördüncü üyesi**.

**VE DEĞERİN KENDİSİ HANGİ TABLONUN SEÇİLDİĞİNE BAĞLIDIR** — dört **adı
konmuş** seçim ölçüldü (IAU 2015 `GM_☉`, JPL J2000 elemanları):

    A  (a, e, T) tablodan + Julyen yüzyıl        42.980644"
    B  T Kepler III'ten (a, e, GM tablodan)      42.980473"
    C  a Kepler III'ten (T, e, GM tablodan)      42.980758"
    D  A ile aynı, ama TROPİK yüzyıl             42.979726"

yayılım **0.0010320″**, yani 43'ten ölçülen farkın (0.019600″) **%5.27**'si.
Planın *"dört seçim 42.979758…42.991703, yayılım 0.0119″ = farkın %59'u"*
satırı **yeniden üretilemedi** ve sebebi tanıdık: plan hangi dört tablo
olduğunu **hiç yazmamıştı**. Alt uç yakın çıkıyor (42.979726 ≈ 42.979758) ama
üst uç (42.991703) standart hiçbir tablodan gelmiyor — denenen sekiz varyantın
hepsi 42.9797–42.9808 bandına düşüyor.

Yerine **dört fizik çıpası** kondu ve dördü de `M`'den, `a`'dan, `e`'den ve
birim seçiminden bağımsızdır (`G = c = 1` bir birim seçimidir):

| çıpa | ölçülen | hedef | dürüst hane |
|---|---|---|---|
| perihel katsayısı `C = 2π(2+2γ−β)` | **18.84955591968662** | `6π` = 18.84955592153876 | 10.01 |
| aynısı, **izotropik ayarda** | **18.849555921564487** | `6π` | 11.86 |
| ikinci mertebe `K = πλ²(3+e²/6)` | **85.24858543879077** | `27π + 1.5πe²` (λ = 3 dilimi) | en kötü bağıl 4.086e-05 |
| bükülme oranı `1+γ` | **1.9999999996887523** | 2 | 9.81 |
| aynısı, **izotropik ayarda** | **1.999999999804698** | 2 | 10.01 |
| sekiz sabiti `|E|^{3/2}T/(G²m^{5/2})` | **9.237681250723917** | — (kapalı formu yok) | 4 integratörde 1.972e-12 |

#### Hata çubuğu makinesi — deponun ilk "fiziği kapat, aynı boruyu koştur"u

Her fizik sayısının yanında **kendi borusundan** okunan bir taban vardır:
fiziği kapat (`λ = 0`, yani `β = 2+2γ`; N-cisimde `m2 = 0`), **aynı** boruyu
**aynı** pencerede koştur, çıkan sapmayı taban say. **KARIŞIK BORU YASAK** —
spec §4.3'ün ilk tablosu gürültüyü ODE'den sinyali karelemeden alıyordu ve o
tablo geçersizdi.

Sevk edilen koşumda beş boru, beş etiket:

    boru                         taban          S/G          hüküm     canlı
    kareleme | gauge=schw    1.594187e-09   1.1824e+10     SAGLAM      6/6
    ode | method=DOP853      1.921807e-06   9.8082e+06     SAGLAM      6/6
    bukulme | gauge=schw     3.651917e-10   5.4766e+09     SAGLAM      6/6
    nbody | method=DOP853    3.996803e-14   3.0926e+13     SAGLAM      7/7
    nbody-dogrusal-harita    0.000000e+00   OKUNAMADI      DEJENERE    0/1

**ÜÇ BORU ÜÇ AYRI ŞEKİLDE DAVRANIYOR VE PANEL BUNU YAZAR** — "S/G büyük"
cümlesi bu farklar olmadan yanıltır:

- **Kareleme**: `w−1` ile `dφ−2π` aynı büyüklüğün iki yüzü olduğu için hüküm
  `SAGLAM`'dan doğrudan `DEJENERE`'ye geçer; **`CURUK`'e ERİŞİLEMEZ**.
- **ODE**: hiçbir basamak ölmez; **`CURUK` erişilebilir, `DEJENERE` değil** —
  karelemenin **tam tersi**.
- **N-cisim**: iki gözlemlenebilir **iki ayrı dejenerelik yüzeyi** taşır.

Ve `CURUK` **yapısal olarak dar bir olaydır** (üç boruda üç kez ölçüldü):
sinyal her zaman varsayılan fizik noktasının O(1) değeridir ve taban ona ancak
boru çöpe döndüğünde yetişir.

#### Kuram seçici — panelde YENİ BİR KUTU, ve İKİ üyesi var

`λ`, `γ`, `β` bir **yaklaşım değildir** (yaklaşım YAPILAN bir şeydir: "test
parçacığı aldık") ve bir **birim seçimi de değildir** (`G = c = 1` odur).
**Hangi kuramın çözüldüğünü** seçerler: GR `γ = β = 1` ⇔ `λ = 3`, ve **herhangi
başka değer GR DEĞİLDİR**. `Provenance.theory_selector` bu yüzden ayrı bir
alandır ve `diagnostics`e **konmadı**: `_diagnostics_summary()` çıktısı
"Yakınsama" satırının **içine**, ✓ işaretinin hemen ardına basılır — kuram
seçici oraya konsaydı panelde **"✓ … theory=NOT-GR"** görünürdü.

**İKİNCİ ÜYE TASK 5'TE BULUNDU:** N-cisim tarafındaki `power` (kuvvet üssü) de
hangi kuramın çözüldüğünü seçer, ve `p ≠ 2`'de **potansiyelin kapalı formu bir
SEÇİMDİR** (sevk edilen seçim `U = −mᵢmⱼ/((p−1)r^(p−1))`). O seçim
adlandırılmadığı için planın bütün bir üs tablosu yeniden üretilemedi.
`p = 2`'de iki seçim çakışır, o yüzden çıpanın kendisi etkilenmez.

#### Sekiz yörüngesi — ARAMADAN SONRA daireselleşen ÜÇ iddia

Başlangıç koşulları **depo içinde aranır**, literatürden alınmaz: bilinmeyen
`(vx, vy, T)`, koşul `Φ_{T/3}(y₀) = P_{(2,0,1)} y₀`. Çözüm
`vx = 0.347116888119`, `vy = 0.532724945388`, `T = 6.325913982926`; BVP artığı
**2.498002e-14** ve literatür IC'sinden **874 kat** daha iyi.

Ama aramadan **sonra** üç iddia daireselleşir ve panel bunu böyle yazar:

1. *"üçü de aynı eğride `T/3` kaymayla dolaşıyor"* → **BVP koşulunun ta
   kendisi** (arama tam olarak onu sıfırlar);
2. tam periyot dönüşü → `Φ_T = (Φ_{T/3})³` **cebirsel sonucu**; integratörü
   ölçer, fiziği değil;
3. **`L = 0` bir ÖLÇÜM DEĞİL** — simetrik ansatz'ta `build(0, 0)` bile sıfır
   verir.

**Girdi olarak kalan tek şey SİMETRİ SINIFIDIR** (Euler dizilimi + `T/3`
devri) ve o bir sayı değildir. Bu üç madde `Provenance.circularity`nin
**ikinci parçasıdır**; birinci parça 43″/efemeristir. Tek parçalı bir
dairesellik beyanı §9 kriteri 5'in ikinci yarısını **karşılamaz**.

**KARARLILIK `max|λ|` İLE RAPORLANMAZ:** `λ = 1` sekiz katlı ve **defektiftir**
(cebirsel kat 8, geometrik kat 4 → dört tane 2×2 Jordan bloğu) ve defektif bir
özdeğer bozulmaya **karekökle** tepki verir; `max|λ| − 1` yakınsamaz. Hüküm
**eliptik çiftin açısıyla** verilir: **±1.872970598421** ve
**±0.052921540137** rad.

#### Üç sınıf, "çıpa değil" rafı, ve Task 5'in düzeltmesi

- **FİZİK ÇIPASI** (statik kökü yok, yörünge çözülmek zorunda): `C`, `K`,
  bükülme oranı, sekiz sabiti.
- **KAPALI-FORM TÜREV** (fizik çıpası **değil**): ISCO `r/r_s` = 3, foton
  küresi 1.5, `b_krit/M` = `3√3` = 5.196152422706632,
  `μ_c` = 0.038520898328784976, apsidal üs −0.4999999923696353.
  Statik/dinamik oranları **ISCO'da 40418×**, **`b_krit`'te 483×** — dinamik
  yol her ikisinde de daha pahalı **ve** daha az hassas. Üçü yalnız `β−γ`'ya
  bağlıdır ve **`β > γ`'da yoktur**; orada motor inanılır görünen bir sayı
  döndürmek yerine **reddeder** ve panel yokluğu yazar.
- **ŞEMA ÇIPASI**: Richardson merdiveninin kendi optimumu, `t0 = 0`
  sözleşmesi.
- **ÇIPA DEĞİL — rafta ÜÇ üye ve üçünün sebebi ayrı:** Kepler 3/2
  (`[GM] = L³T⁻²` onu **zorunlu** kılar) · `⟨U⟩/|E| = 2` (viryal) · **enerji
  korunumu** (`n = −2, −2.5, −1.3, 0, +3`'ün hepsinde geçiyor → bir
  **integratör** kapısı, tautoloji).

**VE TASK 5'İN DÜZELTMESİ PANELDE:** bir çıpa olarak **tautoloji olmak,
mutasyon kapsaması olmamak demek değildir** — o test bir mutantı öldürdü. Bir
çıpanın **fizik değeri** ile bir testin **ayırt etme gücü** iki ayrı
büyüklüktür.

#### Yöntem/hedef uyuşmazlığı — İKİ TÜR, AYRI SATIRDA

| tür | anlamı | bu modüldeki üye |
|---|---|---|
| **A — yöntem hedefi KAPSAMIYOR** | düzeltilemez; başka bir **kuram** gerekir | Newtonyen N-cisim perihel kaymasını **yanlış üretmez, HİÇ üretmez** (`λ = 0`'da kapalı form tam sıfır); jeodezik çözücü **test parçacığı** taşıdığı için geri-tepkiyi üreten bir denklem taşımaz |
| **B — hedef bu modülün DIŞINDA** | ertelendi, engel yok | Kerr (tüm `ISCO(a)` tablosu **0.27 ms**, ODE'siz — maliyeti sıfır, **çıpa değeri de sıfır**), ışıma sönümü, gelgit deformasyonu |

**A'yı B sanmak "sonraki modülde ekleriz" yalanını üretir; B'yi A sanmak
yapılabilir işi gereksiz yere gömer.** Modül 5'in tam sayısı buydu (Zeeman'ın
yörüngesel `g = 1`'i) ve o zaman iki tür ayrılmamıştı.

#### Modüller arası bağ — YOKLUK beyanı, ölçülmüş sayılarla

Bu modül atom ölçeğindeki modüllerle **fiziksel olarak bağlanmaz**, ve engel
**boyut değil ENERJİDİR** (CODATA 2018 / SI 2019, bu depoda):

- **`F_g/F_e = 8.093550856608016e-37`** — ve bu **iki PROTON** içindir. Çift
  seçimi bir düğmedir: elektron-proton için aynı oran **4.407886e-40**, yani
  **üç mertebe** başka. Plan hangi çift olduğunu yazmamıştı; ölçüm iki protonu
  gösterdi ve panel bunu adıyla yazıyor.
- Yerçekimsel "atom"un (`α_g = Gm_p²/(ħc)` = 5.906149e-39) taban durum enerjisi
  **2.621912e-87 J**, yani KAF fotonlarının `k_BT` = 3.762931e-23 J'unun
  **6.967737e-65** katı. **İKİ DÜĞME DAHA:** indirgenmiş kütle kullanılsaydı
  sayı **tam yarıya** düşerdi (3.483869e-65), ve `T = 2.725 K` yuvarlamasıyla
  **6.968950e-65** çıkardı — planın **6.969e-65**'i tam olarak budur. Yani o
  sayı yanlış değildi, **penceresi yazılmamıştı**.
- Protonun `r_s`'ine (`2Gm_p/c²` = 2.484232e-54 m) inmek, kenarı bir Bohr
  yarıçapı olan bir küpte **9.665562e+129** ızgara noktası ister. Planın
  **~6.5e127**'si **yeniden üretilemedi** ve sebebi yine aynı: **span bir
  düğmedir ve adlandırılmamıştı** — kenar 1 Å seçilseydi **6.522644e+130**,
  planın sayısı için kenarın ≈ 1e-11 m olması gerekirdi. Bir dekat span üç
  mertebe nokta demek.

Kurulabilen tek bağ `1/r`'nin **yapı ortaklığıdır** ve o da bir **kapı**, çıpa
değil.

#### Panel ve senaryo

```bash
python -m qlab.run scenarios/grav_tarama.yaml -o grav.npz
```

Çıkış 0; panel **on** satır basar — Yöntem / Yaklaşımlar / İhmal edilen /
Beklenen hata / Dairesellik / Çıpa sınıfı / **Kuram seçici** / Pencere /
Yakınsama / Yakınsama tanısı. Ortadaki yeni satır bu modülde `Provenance`a
eklendi.

`Yakınsama` burada "hesap bitti" **demek değildir** ve bir merdivenin oturması
da değildir. Bağlandığı şey şudur: **bu koşumun her ÇIPA borusu kendi hata
çubuğunu okuyabildi mi?** Yani tik "hesap bitti" değil, *"bastığım her çıpa
sayısının yanında geçerli bir hata çubuğu var"* demektir. `CURUK` ya da
`DEJENERE` bir çıpa borusu tiki düşürür — ve bu **ölçülerek** gösterildi
(`test_CIPA_BORUSU_CURURSE_PANEL_TIK_BASMAZ`).

**DEJENERE BORU PANELDEN ÇIKARILMADI.** Spec §4.3'ün kendi N-cisim
gözlemlenebiliri (`third_body_influence`) `m2 = 0`'da **bit bit sıfır** verir
(0/1 canlı basamak) ve limit de kurtarmaz — gürültü sinyalle birlikte
ölçeklenir. Sevk edilen `circular_frequency_defect` bunun bir **İKAMESİDİR** ve
gerekçesi tam olarak bu ölçümdür; dejenerelik saklansaydı ikamenin gerekçesi de
kaybolurdu. Satır basılır, `converged` hükmüne girmez, ve ikisi de testte
kilitli.

**TOHUM YERİNE PENCERE.** Bu modülde RNG hiç kullanılmaz. `grav_tarama.yaml`
bir tohum taşımaz ama pencerenin **her düğmesini** taşır ve **varsayılanlı
alanı YOKTUR** (`em_tarama.yaml`den bir adım daha ileri): ayar, `γ`, `β`, `M`,
`e`, iki ayrı Richardson çifti, ikinci mertebenin basamağı, yörünge motorunun
`p`'si ve nokta sayısı, N-cisim `rtol`/`atol` ve integratör adı.

**`lam` BİR ANAHTAR DEĞİLDİR** — tek kaynak `(γ, β)`dır ve `λ` motorda
türetilmiş bir `@property`dir. İki alan aynı şeyi belirlerse **sessiz bir
kazanan** doğar ve o kazanan adlandırılmamış yeni bir pencere düğmesidir.

**`trajectory_p` = 100 ile `richardson` = (3e5, 1e6) AYNI DOSYADA, ve bu bir
tutarsızlık değil:** çıpa `p = 3e5`'te okunur (perihel kayması 6.28e-05 rad,
gözle görünmez), yörünge izi `p = 100`'de çizilir (bir radyal periyot 2π'den
**0.197475 rad** büyük, gül deseni ekranda görünür). İki ayrı işin iki ayrı
penceresi; panel ikisini de adıyla basar.

Beyan metninin **tek kaynağı** `engine/grav/factory.py`dir ve bu bir disiplin
değil bir **test**tir — üstelik **iki katmanlı**: yirmi dört beyan girdisinin
her biri `src/` ağacında tam bir kez geçer, **ve** hiçbir dize sabiti bir beyan
cümlesinin **içinde** geçemez. İkinci katman Modül 5'in ölçülmüş deliğidir: bir
mutant `_PENCERE_BEYANI`nin **55 karakterlik bir parçasını** kopyalayarak
tam-cümle sayımından kaçmıştı. Eşik bu modülde **yeniden ölçüldü**
(4 → 92, 6 → 20, 8 → 13, 12 → 2, 16 → **0**) ve yine **16** çıktı — bir
devralma değil, bağımsız bir ölçümün aynı yere düşmesi.

#### Dokstring denetimi — YEDİNCİ kez ürün verdi

Desen sabit: *yeni bir parça sevk edilince ilk kez sınanabilir hâle gelen bir
iddia düşer.* Bu turda düşen, `geodesic.run_trajectory`nin **maliyet
tablosuydu**. Sevk edilen dokstring şunu yazıyordu:

    ODE + dense_output, 2000 nokta          2.482 ms
    BIRIKIMLI kareleme, 2000 nokta         12.026 ms   ( 4.8 kat)
    NOKTA BASINA ayri quad, 2000 nokta    207.568 ms   (83.6 kat)

Tablo üç satırında da ODE'yi **en ucuz** gösteriyordu. Ölçüldü — ve
**VEKTÖRLEŞTİRİLMİŞ** bir birikimli kareleme (aynı `ψ` ikamesi, yamuk kuralı)
2000 noktayı **0.050 ms**'de veriyor, yani ODE'den **57 kat UCUZ**; dahası
`φ(π)` değeri `quad`'ınkiyle **bit bit aynı** (bağıl fark 0.0). Yani "kat"
sayısı bir yöntem farkı değil bir **yazım** farkıydı.

Ama düşen şey tablonun **ikincil** yarısıdır; dokstringin asıl gerekçesi
**güçlenerek** ayakta kaldı: kareleme `φ(u)` verir, yani **ters fonksiyonu**,
ve **eşit aralıklı `φ`** istendiğinde her nokta ayrı bir kök bulma işidir.
Elmayla elma karşılaştırması ancak orada yapılır ve ölçüldü:

    ODE + dense_output (eşit aralıklı φ)                  2.842 ms
    `quad` + `brentq` ile ters çevirme (eşit aralıklı φ)  72.325 ms   (25 kat)

Yani planın **14.4 kat**'ı da Task 4'ün **4.8 kat**'ı da bu pencerede
üretilemiyor; doğru sayı **25 kat**tır ve karşılaştırılan şey ilk kez aynı
çıktıdır.

#### Dürüst olumsuzlar

Beşi de **panelde ve `.npz` kaydında** yazılıdır — saklanmıyorlar:

- **Taban YALNIZ ARİTMETİĞİ ölçer.** Kesme hatası fizikle birlikte kapandığı
  için "fiziği kapat, aynı boruyu koştur" sınavı onu **yapısal olarak
  göremez**: tek basamaklı bir pencerede gerçek hata 2.842e-04 iken taban
  2.453e-10, yani gerçeğin **1/1 158 000**'i. **S/G ancak YUVARLAMA SINIRLI bir
  pencerede bir hata çubuğudur.**
- **`K` ayar-bağımlıdır** (Schwarzschild 85.248585 ↔ izotropik 26.577757),
  dolayısıyla iki ayarlı çapraz kontrol **yalnız birinci mertebeyi** belgeler.
- **Çapraz kontrolün görmediği yüzey ölçüldü:** iki ayar dönüş noktalarını,
  `quad` düğmelerini, `ψ` ikamesini ve Richardson çiftini **paylaşır**; o
  yüzeye sokulan bir bozulma iki tarafı **birlikte** kaydırır ve farkı
  **büyütmez, KÜÇÜLTÜR** — orada çapraz kontrol kör değil **yanıltıcıdır**.
  Öldüren şey `6π` çıpasıdır.
- **Spec §4.3'ün kendi N-cisim gözlemlenebilirinde taban ve kesişim
  üretilemiyor**; sevk edilen `circular_frequency_defect` beyan edilmiş bir
  **ikamedir** ve gerekçesi bu ölçümdür.
- **İzotropik `K`'nın kapalı formu türetilmedi**, ve izotropik `λ = 0`
  çizgisinin model artığı **`−24π(1+γ)²` ÖLÇÜLDÜ ama TÜRETİLMEDİ**.

Bunlara ek olarak, panelde değil ama burada yazılı olan üç beyan (ikincisi ölçülerek kapandı):

- **`slow` eşiğinden sapma beyan edildi:** modülün ölçülmüş eşiği 1.0 s,
  `test_CLI_UCTAN_UCA` **1.43 s** sürüyor ve bilerek işaretlenmedi — yoksa
  `-m "not slow"` koşumu senaryo sınırını hiç sınamazdı.
- **Sekiz yörüngesinin havza yüzdesi bir TANIMSIZLIKTI** ve iki kutu ilan
  edilerek ölçüldü (dar %96.3, geniş %63.0); planın "%58.2"si kutusuz
  yazılmıştı.
- **`b = 1.1 b_krit` güçlü alan noktası sevk edildi, dinamik `b_krit`
  edilmedi** — Task 4'te ölçüldü: **168 kat ucuz** (0.0386 ms ↔ 6.50 ms) ve
  duyarlılık eşiği `ε ≈ 3.5e-16` ↔ `2.5e-5`, yani ~**1e11 kat** daha duyarlı;
  planın *90 kat / 8-10 kat* sayıları o turda **üretilemedi** ama hüküm aynı
  yönde çıktı. Dinamik yolun yakalayıp güçlü-alan bükülmesinin kaçırdığı bir
  `ε` **arandı, bulunamadı**.

#### Planın bayat çıkan iddiaları

Plan K'nın **otuz dört** sayısı ölçümle düzeltildi (Task 1–5'te otuz, Task
6'da dört). Task 6'nın dördü:

| planın/spec'in dediği | ölçülen |
|---|---|
| efemeris yayılımı **0.0119″** = farkın **%59**'u (dört seçim 42.979758…42.991703) | dört **adı konmuş** seçimle **0.0010320″** = **%5.27**; üst uç (42.991703) standart hiçbir tablodan gelmiyor — plan tabloları **hiç adlandırmamıştı** |
| yerçekimsel atom / KAF oranı **6.969e-65** | `T = 2.72548 K` ile **6.967737e-65**; planın sayısı `T = 2.725 K` yuvarlamasınındır (6.968950e-65) — yanlış değil, **penceresi yazılmamış** |
| protonun `r_s`'ine **~6.5e127** ızgara noktası | **span adlandırılmamıştı**; kenar `a₀` ile **9.665562e+129**, kenar 1 Å ile **6.522644e+130**, planın sayısı için kenar ≈ 1e-11 m gerekir |
| yörünge maliyeti **14.4 kat** (Task 4: **4.8 kat**) | elmayla elma (eşit aralıklı `φ`) **25 kat**; vektörleştirilmiş birikimli kareleme ODE'den **57 kat UCUZ** ve `φ(π)`de `quad` ile **bit bit aynı** |

Ve **doğrulanan** sayılar da yazılı: `F_g/F_e = 8.093551e-37` birebir çıktı
(8.093550856608016e-37 — ama **iki proton** için; plan çifti yazmamıştı), ve
Task 1–5'in bütün çıpa değerleri ile beş gürültü tabanı senaryo yolundan
**yeniden okunduğunda birebir aynı** geldi.

#### Ne yapılmadı

Kerr metriği (tüm `ISCO(a)` tablosu **0.27 ms**) · Kozai-Lidov salınımı
(ölçümle reddedildi: `cos²i_c = 0.602 ± 0.003`, `|Δ| < 1e-4` için maliyet
**4.5 gün**) — **bu dört sayı tasarım belgesinin ön ölçümünden ALINTIDIR ve bu
depoda yeniden üretilmedi**, çünkü ne Kerr ne Kozai gövdesi sevk edildi; panel
alıntı olduklarını kendi cümlesiyle yazar · kuantum yerçekimi · kozmoloji
ve kozmolojik sabit · ışıma sönümü · gelgit deformasyonu · `N > 3` kaotik
sistemler, Lyapunov üsleri, 3B yörüngeler, Simó'nun öteki koreografileri ·
**gözlem verisiyle karşılaştırma**.

## Mimari kuralı

`engine` paketi `viz` veya `app` import **edemez** — ve GUI'nin kendisini de,
yani `PySide6` / `pyqtgraph` / `OpenGL` / `matplotlib`'i de. (`matplotlib`
Modül 3'te ölçülerek eklendi: `engine/tb/bands.py`ye enjekte edilen
`import matplotlib.pyplot` rc=0 ile **geçiyordu**, aynı dosyaya konan PySide6
kontrolü ise rc=1 veriyordu — yani araç çalışıyordu, eksik olan kapsamdı.) Aynısı `observable` için geçerli.
Katman sırası `run/app → viz → observable → engine → scenario →
provenance → types → units/errors`. Kural yorumla değil import-linter ile
korunur: `pytest tests/test_architecture.py`.

Sözleşmenin kapsamı ÖLÇÜLEREK genişletildi. Önceki hâlinde yasak listesi
yalnızca `qlab.viz` ve `qlab.app` idi ve şu altı ihlalin altısı da rc=0 ile
KAÇIYORDU: `engine → PySide6`, `engine → pyqtgraph`, `engine → OpenGL`,
`observable → PySide6`, `engine.hf.analysis → qlab.run`, `units → engine`.
Yani "engine görselleştirmeyi bilmez" diyen sözleşme, engine'in doğrudan Qt
import etmesine izin veriyordu.

`test_import_contracts_hold` tek başına bunu göremezdi: o yalnızca temiz
ağaçta rc==0 olduğunu görür ve sözleşme boşaltılsa bile yeşil kalırdı. Bu
yüzden yanına `test_import_sozlesmesi_gercekten_REDDEDIYOR` eklendi — kopya
bir ağaca ihlal enjekte eder ve **adı geçen sözleşmenin BROKEN raporlandığını**
şart koşar. O "BROKEN" şartı da ölçümle geldi: yalnızca sözleşme adının
çıktıda geçmesini aramak totolojiydi, çünkü import-linter her sözleşmenin adını
her koşuda basar.

## Kabul kriterleri durumu

Spec §15'teki yedi maddenin gerçek durumu (ölçüm ayrıntıları
`.superpowers/sdd/task-15-report.md` — Modül 1A — ve
`.superpowers/sdd/b1-task-13-report.md` — Modül 1B — içindedir):

| # | Kriter | Durum |
|---|---|---|
| 1 | `pytest` tam yeşil, PySCF referans testleri dahil | **Kapandı** — **2248/2248 test yeşil** (`pytest tests/`, 1050.45 s; 02.10.2026, `87337f0`). Önceki ölçümler 1815/1815 (1203.40 s, Modül 7 kapanışı), 1705/1705 (766.92 s, Modül 6 kapanışı), 1389/1389 (939.68 s, Modül 5 kapanışı), 1159/1159 (585.36 s, Modül 4 kapanışı) ve 971/971 (432.22 s, Modül 3H kapanışı) idi; süre 1389 → 1705'te DÜŞTÜ ve bu bir hızlanma değil bir KARIŞIM değişimidir — Modül 6'nın 307 testinin çoğu mikrosaniyelik kapalı-form ve panel denetimidir; aşağıdaki `reference/` deneyi o günden de eskidir. Ayrıca `reference/` geçici olarak kaldırılıp tam süite yeniden koşuldu: testler ATLANMIYOR, KIRMIZI oluyor — ölçüldü, **128 failed + 132 error + 291 passed, 0 skipped** (toplam 551, 154.20 s). Yani 260 test kırmızıya dönüyor ve hiçbiri sessizce atlanmıyor. Dikkat: bu ikinci deney süite 551 testliyken yapıldı ve o günün sayılarını taşır; yeniden koşulmadı, sayıları bugünkü toplama uydurmak ölçüm değil uydurma olurdu. Not: PySCF bu makinede **kurulu değildir** — `reference/*.json` kayıtları PySCF 2.14.0 ile başka bir yerde üretilmiştir ve testler koşarken PySCF'e bağımlı değildir; "PySCF referans testleri dahil" ifadesi kayıtlarla karşılaştırmayı anlatır, canlı bir PySCF çağrısını değil |
| 2 | Headless koşucu çalışır + H₂ eğrisi bağ uzunluğu ±%8 | **Kapandı** — `python -m qlab.run scenarios/h2_bond_scan.yaml -o h2_curve.npz`; taramanın parabol fiti **0.71348 Å**, deneyselden sapma %3.7, kriterin istediği ±%8 sağlanıyor. Bu sayı yöntemin denge uzunluğu DEĞİL: STO-3G'nin gerçek minimumu 0.71223 Å, fark taramanın ızgara eseridir (Modül 1B tablosunun altındaki nota bakın). Bu satır bir zamanlar 0.7156 Å diyordu — o sayı bu komuttan değil, `test_physical`'ın ayrı ızgarasından geliyordu; artık `test_shipped_h2_scan_fit_is_grid_biased_but_less_than_the_uniform_grid` senaryonun kendi sayısını kilitliyor |
| 3 | GUI'de H₂O HOMO izoyüzeyi döndürülebilir | **Kapandı — bu depoda ilk gerçek-ekran doğrulaması.** `python -m qlab.app.main --molecule "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"` gerçek Windows platformunda açıldı; su HOMO'su (1b₁) çizildi, ±faz iki renkte, 1764/1764 tepe, çekirdekler ve şeffaflık paneli görüldü. Döndürme ölçüldü: kamera azimutu değişince kare piksel bileşenlerinin **%14.1'i**, elevasyon değişince **%7.2'si** değişiyor — sahne gerçekten 3B ve görünüm tepki veriyor. **Ne ölçülmedi:** fareyle sürükleyerek döndürme; kamera programatik değiştirildi. Kriterin "döndürülebilir" kelimesi bu yüzden yalnızca sahne tarafında kapalıdır, girdi tarafında bir insanın bakması gerekir. Doğrulamanın kendisi bir kusuru da GÖRÜNÜR KILDI: 12 satırlık beyan 140 px'lik panelde katlanın altında kalıyordu (düzeltildi, `TransparencyPanel(max_height=…)`) |
| 4 | Kaydırıcı ile canlı iletim, analitiğin ±%2'si içinde | **Kapandı** — ölçülen fark %0.07 |
| 5 | Şeffaflık paneli her hesapta dolu | **Kapandı** — TDSE tarafında pencere paneli; HF tarafında headless koşucu her senaryoda Yöntem/Yaklaşımlar/İhmal edilen/Beklenen hata/Yakınsama satırlarını basar (`h2_bond_scan.yaml` çıktısı) |
| 6 | Yakınsamayan SCF sessizce sayı döndürmez, test edilmiş | **Kapandı** — `SCFConvergenceError`, `test_non_convergence_raises_rather_than_returning` ile test edilmiştir |
| 7 | `engine`'in `viz`/`app` bağımlılığı yok, import-linter ile doğrulandı | **Kapandı** — üç sözleşme (`engine görselleştirmeyi bilmez`, `observable görselleştirmeyi ve GUI'yi bilmez`, `Katmanlı mimari`), hepsi KEPT. Sözleşmenin kendisinin diş taşıdığı da ölçülüyor: enjekte edilen 13 ihlalin 13'ü adı geçen sözleşmeyi BROKEN yapıyor (11.08'de 8'di) (bkz. "Mimari kuralı") |

**7/7 madde kapalıdır.** Kriter 3 Modül 1B2 ile kapandı; yukarıdaki tablonun
yedi satırı da "Kapandı" der.

(Bu paragraf bir süre kendi tablosuyla çelişti: "6/7 madde kapalıdır, kapanmayan
tek kriter 3'tür" diyor ve "Qt penceresi bu depoda hiçbir gerçek ekranda henüz
açılmadı" diye ekliyordu — oysa aynı belgenin üç ayrı yeri, tablonun 3. satırı
dahil, kriterin B2'de kapandığını ve pencerenin gerçek ekranda **bir kez**
açıldığını yazıyordu. B2'den kalma güncellenmemiş bir özetti.)

Kriter 3'ün kapanışı kendi sınırını taşıyor ve o sınır hâlâ geçerlidir: sahne
tarafı ölçüldü, **girdi tarafı ölçülmedi** — fareyle sürükleyerek döndürme
denenmedi, kamera programatik değiştirildi. TDSE penceresi ise hâlâ hiçbir
gerçek ekranda görülmedi; onun dışındaki tüm GUI testleri ekransız
(`QT_QPA_PLATFORM=offscreen`) koşar.
