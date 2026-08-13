"""Klasik moleküler dinamik motoru (Modül 2, Plan C).

İntegratör kuvvetin nereden geldiğini bilmez; tek sözleşme
`(konumlar) -> (enerji, kuvvetler)` (bkz. `state.ForceField`). Harmonik,
Morse ve Lennard-Jones bu sözleşmenin üç uygulamasıdır; Plan D dördüncüsünü
(Hartree-Fock sayısal gradyanı) ekleyecek ve buraya hiç dokunmayacak.

Birimler: konum bohr, enerji hartree, zaman atomik zaman birimi, KÜTLE
ELEKTRON KÜTLESİ. Son madde bu modülün en tehlikeli hatasıdır -- bkz.
`qlab.units.amu_to_electron_mass`.
"""
