# -*- coding: utf-8 -*-

import mir3.modules.tool.wav2spectrogram as spectrogram
import mir3.modules.features.flatness as flatness
import os

##vetor com todos os nomes dos arquivos .wav
counter = 0
filenames = []
for subdir, dirs, files in os.walk('./'):
	for file in files:
		if file.endswith(".wav"):
			name = file
			#print (name)
			counter += 1
			filenames.append(name)
print (counter)
print (filenames)