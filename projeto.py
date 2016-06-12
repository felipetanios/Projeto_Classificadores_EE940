
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

flat_rock_ = zeros(len(flat_rock))
cent_rock_ = zeros(len(cent_rock))
#rolloff_rock_ = zeros(len(rolloff_rock))
energy_rock_ = zeros(len(energy_rock))
sflux_rock_ = zeros(len(sflux_rock))

flat_pop_ = zeros(len(flat_pop))
cent_pop_ = zeros(len(cent_pop))
#rolloff_pop_ = zeros(len(rolloff_pop))
energy_pop_ = zeros(len(energy_pop))
sflux_pop_ = zeros(len(sflux_pop))

#for i in xrange(len(flat_rock)):
flat_rock_ = np.asarray(flat_rock)
cent_rock_ = np.asarray(cent_rock)
#rolloff_rock_ = np.asarray(rolloff_rock)
energy_rock_ = np.asarray(energy_rock)
sflux_rock_ = np.asarray(sflux_rock)

#for i in xrange(len(flat_pop)):
flat_pop_ = np.asarray(flat_pop)
cent_pop_ = np.asarray(cent_pop)
#rolloff_pop_ = np.asarray(rolloff_pop)
energy_pop_ = np.asarray(energy_pop)
sflux_pop_= np.asarray(sflux_pop)


rock_ = []
pop_ = []
for i in xrange(len(rock)):
	rock_.append(0)
for i in xrange(len(pop)):
	pop_.append(1)

# ##teste com KNN


# Parametros para executar busca exaustiva
train_size_min = 0.2
train_size_max = 0.95
train_size_step = 0.05

# Numero de iteracoes para cada tamanho de conjunto de treino
n_iter = 100

# Listas que armazenarao os resultados
steps = []
medias = []
variancias = []

train_size_atual = train_size_min
while train_size_atual <= train_size_max: # para cada tamanho do conjunto de treino
    acertos = []
    for k in xrange(n_iter): # para cada iteracao do processo Monte Carlo
        dados_treino, dados_teste, rotulos_treino, rotulos_teste = train_test_split(energy_rock+ energy_pop, rock_ + pop_, train_size=train_size_atual)
        print(dados_treino)
        print(rotulos_treino)
        
        # classificador = KNeighborsClassifier(n_neighbors=5) # n_neighbors = K
        # classificador.fit(dados_treino, rotulos_treino)
        # score = classificador.score(dados_teste, rotulos_teste)
        # acertos.append(score)
    
    steps.append(train_size_atual)
    medias.append(np.mean(np.array(acertos)))
    variancias.append(np.std(np.array(acertos)))
    
    train_size_atual += train_size_step


plt.figure();
plt.errorbar(steps, medias, yerr=variancias);
plt.ylabel('Indice de acertos');
plt.xlabel('Tamanho do conjunto de treino');

classificador = KNeighborsClassifier(n_neighbors=5)
classificador.fit(energy_rock + energy_pop, rock + pop)
