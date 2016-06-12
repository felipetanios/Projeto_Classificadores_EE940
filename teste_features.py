# -*- coding: utf-8 -*-

import mir3.modules.tool.wav2spectrogram as spectrogram
import mir3.modules.features.flatness as flatness
import mir3.modules.features.centroid as cent
import mir3.modules.features.rolloff as roll
import mir3.modules.features.energy as energ
import mir3.modules.features.flux as specfl
import os
from scipy import *
from pylab import *
from scipy.io import wavfile
import numpy as np




##vetor com todos os nomes dos arquivos .wav
filenames = []

##vetor com todos os generos dos arquivos .wav
filegenders = []

for subdir, dirs, files in os.walk('./'):
	for file in files:
		if file.endswith(".wav"):
			filenames.append(file)

# for i in xrange(len(filenames)):
# 	filegenders.append(filenames[i][0])


rock = []
pop = []
for i in xrange(len(filenames)):
	if filenames[i][0] == "0":
		rock.append(filenames[i])
	else:
		pop.append(filenames[i])



##CALCULO DE FLATNESS E CENTROIDE

flat_rock = []
cent_rock = []
rolloff_rock = []
energy_rock = []
sflux_rock = []
for fname in rock:
    wav2spec = spectrogram.Wav2Spectrogram() # Objeto que converte arquivos wav para espectrogramas
    s = wav2spec.convert(open(fname, 'rb'), window_length=1024, window_step=512, spectrum_type='magnitude')

    fness = flatness.Flatness()
    f = fness.calc_track(s)
    f1 = np.average(f.data)
    flat_rock.append(f1)
    centr = cent.Centroid()
    centroid = centr.calc_track(s)
    centroid1 = np.average(centroid.data)
    cent_rock.append(centroid1)
    # roff = roll.Rolloff()
    # roll_off = roff.calc_track(s)
    # roll_off1 = np.average(roll_off)
    # rolloff_rock.append(roll_off1)
    en = energ.Energy()
    energy = en.calc_track(s)
    energy1 = np.average(energy.data)
    energy_rock.append(energy1)
    fl = specfl.Flux()
    flux = fl.calc_track(s)
    flux1 = np.average(flux.data)
    sflux_rock.append(flux1)

avg_flatrock= 0
avg_centrock = 0
avg_rollrock = 0
avg_energyrock = 0
avg_sfluxrock = 0
for i in xrange(len(flat_rock)):
	avg_flatrock += flat_rock[i]
	avg_centrock += cent_rock[i]
	# avg_rollrock += rolloff_rock[i]
	avg_energyrock += energy_rock[i]
	avg_sfluxrock += sflux_rock[i]
avg_flatrock= avg_flatrock/len(flat_rock)
avg_centrock = avg_centrock/len(cent_rock)
# avg_rollrock = avg_rollrock/len(rolloff_rock)
avg_energyrock = avg_energyrock/len(energy_rock)
avg_sfluxrock = avg_sfluxrock/len(sflux_rock)


print ("FLATNESS ROCK: ")
print (avg_flatrock)
print("\n")

print ("CENTROID ROCK: ")
print (avg_centrock)
print("\n")

print ("ROLL OFF ROCK: ")
print (avg_rollrock)
print("\n")

print ("ENERGY ROCK: ")
print (avg_energyrock)
print("\n")

print ("SPECTRAL FLUX ROCK: ")
print (avg_sfluxrock)
print("\n")


flat_pop = []
cent_pop = []
rolloff_pop = []
energy_pop = []
sflux_pop = []
for fname in pop:
    wav2spec = spectrogram.Wav2Spectrogram() # Objeto que converte arquivos wav para espectrogramas
    s = wav2spec.convert(open(fname, 'rb'), window_length=1024, window_step=512, spectrum_type='magnitude')

    fness = flatness.Flatness()
    f = fness.calc_track(s)
    f1 = np.average(f.data)
    flat_pop.append(f1)
    centr = cent.Centroid()
    centroid = centr.calc_track(s)
    centroid1 = np.average(centroid.data)
    cent_pop.append(centroid1)
    # roff = roll.Rolloff()
    # roll_off = roff.calc_track(s)
    # roll_off1 = np.average(roll_off)
    # rolloff_pop.append(roll_off1)
    en = energ.Energy()
    energy = en.calc_track(s)
    energy1 = np.average(energy.data)
    energy_pop.append(energy1)
    fl = specfl.Flux()
    flux = fl.calc_track(s)
    flux1 = np.average(flux.data)
    sflux_pop.append(flux1)

avg_flatpop= 0
avg_centpop = 0
avg_rollpop = 0
avg_energypop = 0
avg_sfluxpop = 0
for i in xrange(len(flat_pop)):
	avg_flatpop += flat_pop[i]
	avg_centpop += cent_pop[i]
	# avg_rollpop += rolloff_pop[i]
	avg_energypop += energy_pop[i]
	avg_sfluxpop += sflux_pop[i]
avg_flatpop= avg_flatpop/len(flat_pop)
avg_centpop = avg_centpop/len(cent_pop)
# avg_rollpop = avg_rollpop/len(rolloff_pop)
avg_energypop = avg_energypop/len(energy_pop)
avg_sfluxpop = avg_sfluxpop/len(sflux_pop)


print ("FLATNESS pop: ")
print (avg_flatpop)
print("\n")

print ("CENTROID pop: ")
print (avg_centpop)
print("\n")

print ("ROLL OFF pop: ")
print (avg_rollpop)
print("\n")

print ("ENERGY pop: ")
print (avg_energypop)
print("\n")

print ("SPECTRAL FLUX pop: ")
print (avg_sfluxpop)
print("\n")