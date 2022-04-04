import librosa as lr
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch

path = "Projects/Who Dat Boy/Audio/Who Dat Boy.ogg"

y, sr = lr.load(path)
hop_length = 256
S = np.abs(lr.stft(y))

# mel = lr.feature.melspectrogram(y=y, sr=sr, n_mels=256, hop_length=hop_length)
# mel_dB = lr.power_to_db(mel, ref=np.max)

fig, ax = plt.subplots(nrows=1, ncols=1)
img = lr.display.specshow(mel_dB, y_axis='mel', ax=ax)

fig.show()

# tensor_mel_dB = torch.from_numpy(mel_dB)
# print(tensor_mel_dB.size())

