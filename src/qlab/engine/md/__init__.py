"""Klasik moleküler dinamik motoru (Modül 2, Plan C).

İntegratör kuvvetin nereden geldiğini bilmez; tek sözleşme
`(konumlar) -> (enerji, kuvvetler)` (bkz. `state.ForceField`). Bu
sözleşmenin beş uygulaması var: Harmonik, Morse ve Lennard-Jones (Plan C),
Hartree-Fock sayısal gradyanı (Plan D, `forcefields.HartreeFockGradient`) ve
24.09.2026'dan beri `tb_kuvvet.SsSigmaTBKuvveti` (tek orbitalli ss-sigma
sıkı-bağ, Mermin serbest enerjisi). Son ikisi integratöre ve termostatlara
dokunmadan eklendi.

Birimler: konum bohr, enerji hartree, zaman atomik zaman birimi, KÜTLE
ELEKTRON KÜTLESİ. Son madde bu modülün en tehlikeli hatasıdır -- bkz.
`qlab.units.amu_to_electron_mass`.
"""
