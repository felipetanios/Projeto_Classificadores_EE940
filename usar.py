##ARQUIVO PARA DESCREVER OS METODOS PARA CLASSIFICADORES##



## imports ##


import warnings
warnings.filterwarnings('ignore')

# numerical processing and scientific libraries
import numpy as np
import scipy

# signal processing
from scipy.io                     import wavfile
from scipy                        import stats, signal
from scipy.fftpack                import fft

from scipy.signal                 import lfilter, hamming
from scipy.fftpack.realtransforms import dct
from scikits.talkbox              import segment_axis
from scikits.talkbox.features     import mfcc

# general purpose
import collections

# plotting
import matplotlib.pyplot as plt
from numpy.lib                    import stride_tricks

from IPython.display              import HTML
from base64                       import b64encode

# Classification and evaluation
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


## ANALISE DE FLATNESS ## ## serve para pitch##

import mir3.modules.tool.wav2spectrogram as spectrogram
import mir3.modules.features.flatness as flatness

fnames = ['audio/tabla.wav', 'audio/bbking.wav', 'audio/chorus.wav']
flat_samples = []
for fname in fnames:
    wav2spec = spectrogram.Wav2Spectrogram() # Objeto que converte arquivos wav para espectrogramas
    s = wav2spec.convert(open(fname, 'rb'), window_length=1024, window_step=512, spectrum_type='magnitude')

    fness = flatness.Flatness()
    f = fness.calc_track(s)
    flat_samples.append(f.data)
    

## DETECÇAÕ DE ANDAMENTO ##

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import IPython.lib.display as display 

import mir3.modules.tool.wav2spectrogram as spec
import mir3.modules.features.energy as energy

fname = 'audio/tabla.wav'
wav2spec = spec.Wav2Spectrogram() # Objeto que converte arquivos wav para espectrogramas
s = wav2spec.convert(open(fname, 'rb'), window_length=1024, window_step=512, spectrum_type='magnitude')

en = energy.Energy()
f = en.calc_track(s)
T = f.metadata.sampling_configuration.ofs

t = np.linspace(0, len(f.data)/T, len(f.data))

plt.figure(figsize=(10, 6))
plt.plot(t, np.log10(f.data/np.max(f.data)))
plt.xlabel('Tempo (s)')
plt.ylabel('Energia (dB)')
plt.show()

display.Audio(fname)

import mir3.modules.features.flux as flux
fname = 'audio/tabla.wav'
wav2spec = spec.Wav2Spectrogram() # Objeto que converte arquivos wav para espectrogramas
s = wav2spec.convert(open(fname, 'rb'), window_length=1024, window_step=512, spectrum_type='magnitude')

fx = flux.Flux()
f = fx.calc_track(s)
T = f.metadata.sampling_configuration.ofs

t = np.linspace(0, len(f.data)/float(T), len(f.data))

f.data += 10**(-2) # Isto evita divisoes por zero ao calcular o log10 a seguir 
plt.figure(figsize=(10, 6))
plt.plot(t, np.log10(f.data/np.max(f.data)))
plt.xlabel('Tempo (s)')
plt.ylabel('Fluxo espectral (dB)')
plt.show()

display.Audio(fname)

##Vemos um fenômeno interessante, desta vez. As batidas principais
##(correspondentes aos upbeats) estão bem marcadas, mas ainda assim temos uma
##certa dificuldade em visualizar as batidas correspondentes às batidas mais
##fracas. Mesmo assim, este processo já permite encontrar um algoritmo para
##detecção de onsets, que parte da idéia de que todo pico do fluxo espectral
##acima do limiar  −0.5−0.5  dB deve corresponder a um evento:

h = np.log10(f.data/np.max(f.data))
tg = []
g = []
for i in xrange(len(h)-2):
    if (h[i+1] > h[i]) and (h[i+1] > h[i+2]): # Condicao 1: eh um pico
        if (h[i+1] > -.5): # Condicao 2: magnitude acima de um limiar
            g.append(h[i+1])
            tg.append(t[i+1])

plt.figure(figsize=(10, 6))
plt.plot(t, h)
plt.plot(tg, g, 'ro')
plt.xlabel('Tempo (s)')
plt.ylabel('Fluxo espectral (dB)')
plt.show()

def t0_acc(x):
    """Retorna o periodo fundamental de x, em amostras"""
    X = np.abs(np.fft.fft(x))
    r = np.real(np.fft.ifft(X*X))
    r2 = np.zeros(len(r)/2)
    for n in xrange(len(r2)):
        r2[n] = r[n]-r[n/2]
    r2 *= np.linspace(1, 0, len(r2))
    t0 = np.argmax(r2)
    return t0, r, r2


t0,r,r2 = t0_acc(h) # T0 em amostras

#plt.plot(t[0:len(r2)],h[0:len(r2)])
#plt.show()
#plt.plot(t[0:len(r2)],r[0:len(r2)])
#plt.show()
plt.figure(figsize=(10, 6))
plt.plot(t[0:len(r2)], r2)
plt.plot(t[t0], r2[t0], 'ro')
plt.title('Pico de autocorrelacao encontrado')
plt.xlabel('Tempo (s)')
plt.show()
f0 = 1/float(t[t0]) # f0 em Hz
bpm = f0 * 60
print "Andamento: ", bpm, "BPM"



## ANALISE DE BRIGHTNESS ## ## centroid##

def spectral_centroid(wavedata, window_size, sample_rate):
    
    magnitude_spectrum = stft(wavedata, window_size)
    
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))
    
    sc = []

    for t in range(timebins-1):
        
        power_spectrum = np.abs(magnitude_spectrum[t])**2
        
        sc_t = np.sum(power_spectrum * np.arange(1,freqbins+1)) / np.sum(power_spectrum)
        
        sc.append(sc_t)
        
    
    sc = np.asarray(sc)
    sc = np.nan_to_num(sc)
    
    return sc, np.asarray(timestamps)


## ZERO CROSSING RATE ## ## usa força bruta, nao sei se é ideal

def zero_crossing_rate_BruteForce(wavedata):
    
    zero_crossings = 0
    
    for i in range(1, number_of_samples):
        
        if ( wavedata[i - 1] <  0 and wavedata[i] >  0 ) or \
           ( wavedata[i - 1] >  0 and wavedata[i] <  0 ) or \
           ( wavedata[i - 1] != 0 and wavedata[i] == 0):
                
                zero_crossings += 1
                
    zero_crossing_rate = zero_crossings / float(number_of_samples - 1)

    return zero_crossing_rate

## ZERO CROSSING RATE ##  ## sem força bruta ## acredito que block lenght é o tamanho da janela de hanning

def zero_crossing_rate(wavedata, block_length, sample_rate):
    
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(wavedata)/block_length))
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(samplerate)))
    
    zcr = []
    
    for i in range(0,num_blocks-1):
        
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])
        
        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:stop]))))
        zcr.append(zc)
    
    return np.asarray(zcr), np.asarray(timestamps)


## Root Mean Square ## 

def root_mean_square(wavedata, block_length, sample_rate):
    
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(wavedata)/block_length))
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(samplerate)))
    
    rms = []
    
    for i in range(0,num_blocks-1):
        
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])
        
        rms_seg = np.sqrt(np.mean(wavedata[start:stop]**2))
        rms.append(rms_seg)
    
    return np.asarray(rms), np.asarray(timestamps)

## Spectral Rolloff ## indication of how much energy is in the lower frequencies
##It is used to distinguish voiced from unvoiced speech or music

def spectral_rolloff(wavedata, window_size, sample_rate, k=0.85):
    
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    power_spectrum     = np.abs(magnitude_spectrum)**2
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))
    
    sr = []

    spectralSum    = np.sum(power_spectrum, axis=1)
    
    for t in range(timebins-1):
        
        # find frequency-bin indeces where the cummulative sum of all bins is higher
        # than k-percent of the sum of all bins. Lowest index = Rolloff
        sr_t = np.where(np.cumsum(power_spectrum[t,:]) >= k * spectralSum[t])[0][0]
        
        sr.append(sr_t)
        
    sr = np.asarray(sr).astype(float)
    
    # convert frequency-bin index to frequency in Hz
    sr = (sr / freqbins) * (sample_rate / 2.0)
    
    return sr, np.asarray(timestamps)


## KNN ##