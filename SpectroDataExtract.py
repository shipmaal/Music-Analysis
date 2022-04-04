from pprint import pprint

import librosa

import torch

import numpy as np
import matplotlib.pyplot as plt
import torchaudio.transforms

y, sr = librosa.load("Projects/Who Dat Boy/Audio/Who Dat Boy.ogg", offset=45, duration=30)
mfccs = librosa.feature.melspectrogram(y=y, sr=sr)

fig, ax = plt.subplots()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
fig.show()

