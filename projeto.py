
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
from scipy.stats import f_oneway
import scipy.stats as st



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



##CALCULO DE FLATNESS, CENTROIDE, ENERGIA E FLUXO ESPECTRAL PARA AS FAIXAS DE POP E ROCK

flat_rock = [[]]
cent_rock = [[]]
rolloff_rock = [[]]
energy_rock = [[]]
sflux_rock = [[]]
cent_en_rock = [[]]
for fname in rock:
    wav2spec = spectrogram.Wav2Spectrogram() # Objeto que converte arquivos wav para espectrogramas
    s = wav2spec.convert(open(fname, 'rb'), window_length=1024, window_step=512, spectrum_type='magnitude')

    fness = flatness.Flatness()
    f = fness.calc_track(s)
    f1 = [np.average(f.data)]
    flat_rock.append(f1)
    centr = cent.Centroid()
    centroid = centr.calc_track(s)
    centroid1 = [np.average(centroid.data)]
    cent_rock.append(centroid1)
    # roff = roll.Rolloff()
    # roll_off = roff.calc_track(s)
    # roll_off1 = [np.average(roll_off)]
    # rolloff_rock.append(roll_off1)
    en = energ.Energy()
    energy = en.calc_track(s)
    energy1 = [np.average(energy.data)]
    energy_rock.append(energy1)
    fl = specfl.Flux()
    flux = fl.calc_track(s)
    flux1 = [np.average(flux.data)]
    sflux_rock.append(flux1)
    aux = [np.average(f.data)*np.average(centroid.data)*np.average(energy.data)*np.average(flux.data)]
    cent_en_rock.append(aux)

flat_pop = [[]]
cent_pop = [[]]
rolloff_pop = [[]]
energy_pop = [[]]
sflux_pop = [[]]
cent_en_pop = [[]]
for fname in pop:
    wav2spec = spectrogram.Wav2Spectrogram() # Objeto que converte arquivos wav para espectrogramas
    s = wav2spec.convert(open(fname, 'rb'), window_length=1024, window_step=512, spectrum_type='magnitude')

    fness = flatness.Flatness()
    f = fness.calc_track(s)
    f1 = [np.average(f.data)]
    flat_pop.append(f1)
    centr = cent.Centroid()
    centroid = centr.calc_track(s)
    centroid1 = [np.average(centroid.data)]
    cent_pop.append(centroid1)
    # roff = roll.Rolloff()
    # roll_off = roff.calc_track(s)
    # roll_off1 = [np.average(roll_off)]
    # rolloff_pop.append(roll_off1)
    en = energ.Energy()
    energy = en.calc_track(s)
    energy1 = [np.average(energy.data)]
    energy_pop.append(energy1)
    fl = specfl.Flux()
    flux = fl.calc_track(s)
    flux1 = [np.average(flux.data)]
    sflux_pop.append(flux1)
    aux = [np.average(f.data)*np.average(centroid.data)*np.average(energy.data)*np.average(flux.data)]
    cent_en_pop.append(aux)


rock_ = []
pop_ = []
for i in xrange(len(rock)):
	rock_.append(0)
for i in xrange(len(pop)):
	pop_.append(1)


flat_rock = flat_rock[1:]
cent_rock = cent_rock[1:]
#rolloff_rock = rolloff_rock[1:]
energy_rock = energy_rock[1:]
sflux_rock = sflux_rock[1:]

flat_pop = flat_pop[1:]
cent_pop = cent_pop[1:]
#rolloff_pop = rolloff_pop[1:]
energy_pop = energy_pop[1:]
sflux_pop = sflux_pop[1:]
cent_en_rock = cent_en_rock[1:]
cent_en_pop = cent_en_pop[1:]


##teste com KNN


# Parametros para executar busca exaustiva
train_size_min = 0.2
train_size_max = 0.95
train_size_step = 0.05

# Numero de iteracoes para cada tamanho de conjunto de treino
n_iter = 100

#Listas que armazenarao os resultados
steps = []
medias_flat = []
variancias_flat = []
medias_cent = []
variancias_cent = []
medias_en = []
variancias_en = []

train_size_atual = train_size_min
while train_size_atual <= train_size_max: # para cada tamanho do conjunto de treino
    acertos_flat = []
    acertos_cent = []
    acertos_energy = []
    for k in xrange(n_iter): # para cada iteracao do processo Monte Carlo
        dados_treino, dados_teste, rotulos_treino, rotulos_teste = train_test_split(flat_rock+ flat_pop, rock_ + pop_, train_size=train_size_atual)
        dados_treino1, dados_teste1, rotulos_treino1, rotulos_teste1 = train_test_split(cent_rock+ cent_pop, rock_ + pop_, train_size=train_size_atual)
        dados_treino2, dados_teste2, rotulos_treino2, rotulos_teste2 = train_test_split(energy_rock+ energy_pop, rock_ + pop_, train_size=train_size_atual)


        classificador = KNeighborsClassifier(n_neighbors=5) # n_neighbors = K
        classificador.fit(dados_treino, rotulos_treino)
        classificador1 = KNeighborsClassifier(n_neighbors=5)
        classificador1.fit(dados_treino1, rotulos_treino1)
        classificador2 = KNeighborsClassifier(n_neighbors=5)
        classificador2.fit(dados_treino2, rotulos_treino2)

        score = classificador.score(dados_teste, rotulos_teste)
        score1 = classificador1.score(dados_teste1, rotulos_teste1)
        score2 = classificador2.score(dados_teste2, rotulos_teste2)

        acertos_flat.append(score)
        acertos_cent.append(score1)
        acertos_energy.append(score2)
    
    steps.append(train_size_atual)
    medias_flat.append(np.mean(np.array(acertos_flat)))
    variancias_flat.append(np.std(np.array(acertos_flat)))
    medias_cent.append(np.mean(np.array(acertos_cent)))
    variancias_cent.append(np.std(np.array(acertos_cent)))
    medias_en.append(np.mean(np.array(acertos_energy)))
    variancias_en.append(np.std(np.array(acertos_energy)))
    
    train_size_atual += train_size_step

plt.figure();

color=['red', 'blue', 'green']
label=['Centroid', 'Flatness', 'Energy']
plt.errorbar(steps, medias_flat, yerr=variancias_flat)
plt.errorbar(steps, medias_cent, yerr=variancias_cent)
plt.errorbar(steps, medias_en, yerr=variancias_en)
plt.ylabel('Indice de acertos em %')
plt.xlabel('Tamanho do conjunto de treino em %')
plt.legend(label, loc = 2)

plt.show()


# sample_size = len(medias_flat)

acertos_flat = []
for k in xrange(n_iter): # para cada iteracao do processo Monte Carlo
    dados_treino, dados_teste, rotulos_treino, rotulos_teste = train_test_split(flat_rock+ flat_pop, rock_ + pop_, train_size=0.4)
    
    classificador = KNeighborsClassifier(n_neighbors=5) # n_neighbors = K
    classificador.fit(dados_treino, rotulos_treino)
    score = classificador.score(dados_teste, rotulos_teste)
    acertos_flat.append(score)
    


acertos_cent = []
for k in xrange(n_iter): # para cada iteracao do processo Monte Carlo
    dados_treino, dados_teste, rotulos_treino, rotulos_teste = train_test_split(cent_rock+ cent_pop, rock_ + pop_, train_size=0.4)
    
    classificador = KNeighborsClassifier(n_neighbors=5) # n_neighbors = K
    classificador.fit(dados_treino, rotulos_treino)
    score = classificador.score(dados_teste, rotulos_teste)
    acertos_cent.append(score)


acertos_energy = []
for k in xrange(n_iter): # para cada iteracao do processo Monte Carlo
    dados_treino, dados_teste, rotulos_treino, rotulos_teste = train_test_split(energy_rock+ energy_pop, rock_ + pop_, train_size=0.4)
    
    classificador = KNeighborsClassifier(n_neighbors=5) # n_neighbors = K
    classificador.fit(dados_treino, rotulos_treino)
    score = classificador.score(dados_teste, rotulos_teste)
    acertos_energy.append(score)

A = np.asarray(acertos_flat)
B = np.asarray(acertos_cent)
C = np.asarray(acertos_energy)

# print "Realizando t-testes dois-a-dois em 30%:"
# print "um p-value < 0.05 indica diferenca significativa"
# print st.ttest_ind(A, B)
# print st.ttest_ind(A, C)
# print st.ttest_ind(B, C)
