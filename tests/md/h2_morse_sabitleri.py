"""H2'nin Morse sabitleri -- testlerin TEK kaynagi (C6, 11.10.2026).

Eskiden `test_dissociation.py` ve `test_forcefields.py` ayni dort sayiyi ayri
ayri literal yaziyordu; biri degisince oteki sessizce eski degerde kalirdi.
`scenarios/h2_morse.yaml`in params'i da bu sayilardir ve
`test_md_girdi_kapilari.py` ikisinin esitligini kilitler.

Kaynak: Huber & Herzberg 1979 / NIST WebBook (tasarim §4.2). `a` omega_e'den
TURETILMISTIR (dairesellik, tasarim §4.3): bu sayilarla olculen hicbir sey H2
fizigini dogrulamaz.
"""

H2_DE = 0.174434333  # hartree (4.7466 eV)
H2_RE = 1.401118538  # bohr (0.74144 A)
H2_A = 1.028997701  # bohr^-1, a = omega_e*sqrt(mu/(2*De))
H2_MU = 918.576324  # m_e, indirgenmis kutle
