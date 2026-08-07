# Referans değerler

Bu dizindeki JSON dosyaları **PySCF'ten bir kez üretilmiştir** ve elle
düzenlenmemelidir. Üreten betik: `tools/generate_reference.py`.

Yeniden üretmek için (Linux gerekir — PySCF'in Windows wheel'i yoktur, kaynak
tarball'ı gelir ve C derleyici + BLAS ister):

```bash
python3 -m venv ~/pyscfenv && ~/pyscfenv/bin/pip install pyscf numpy
~/pyscfenv/bin/python tools/generate_reference.py --out reference/
```

Testler çalışma zamanında PySCF'e **bağımlı değildir** — yalnızca bu JSON
dosyalarını okurlar. Bu bilinçli: testler hızlı kalır, referanslar sürüm
kontrolüne girer ve hangi PySCF sürümüyle üretildikleri
(`hf_energies.json` içindeki `_pyscf_version`) kayıtlıdır.

Enerjiler hartree, uzunluklar Ångström (JSON içinde `unit` alanı), dipol Debye.
