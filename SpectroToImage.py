import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

import cv2
import PIL

import torch
from torch import nn
import torchaudio
import torchvision
import torchaudio.functional as F
import torchaudio.transforms as T

# from ImageEnhancer import enhance

import ffmpeg


# bottleneck is in enhancement. Try to find a way to enhance less? or find way to improve enhancement
# code is fucking messy, clean up once I figure out optimal way to generate images


def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1, figsize=(80, 20))
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


path = "Projects/Who Dat Boy/Audio/Who Dat Boy.ogg"

metadata = (torchaudio.info(path))

waveform, sample_rate = torchaudio.load(path)

n_fft = 8192
win_length = None
hop_length = 512
n_mels = 512

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk",
)
# Perform transformation
spec = mel_spectrogram(waveform)

print_stats(spec)
plot_spectrogram(spec[0], title='torchaudio')

interval = 3
spec_frames = round(metadata.num_frames / hop_length)
print(spec_frames)
frame_rate = (spec_frames / (metadata.num_frames / metadata.sample_rate)) / interval
print(frame_rate)

flatten = nn.Flatten()
linear = nn.Linear(4097, 4096)

# linearize from 4096 to 4080

os.mkdir("spec-img")

# Transforms into rgb color format
# don't like zipper format, find way include both channels? seems easy enough to fix, maybe go forward with it
# maybe it'll look good
img_tensor = (librosa.power_to_db(spec))
img_tensor = np.reshape(img_tensor, (2, n_mels * (spec_frames - 1)))
img_tensor = np.ravel(img_tensor, 'F')
img_tensor = torch.from_numpy(img_tensor)
img_tensor = torch.reshape(img_tensor, (n_mels * 2, (spec_frames - 1)))
img_tensor = img_tensor.add(abs(img_tensor.amin()))
img_tensor = img_tensor.div(80).multiply(255)
print(img_tensor.size())
print(img_tensor[200].size())

# check three frames at a time, with manipulation at the end and beginning.
for i in range(round(spec_frames / interval)):
    print(i)
    r = torch.reshape(img_tensor[:, i], (1, 32, 32))
    g = torch.reshape(img_tensor[:, i+1], (1, 32, 32))
    b = torch.reshape(img_tensor[:, i+2], (1, 32, 32))

    rgb = torch.cat((r, g, b), dim=0).to(torch.uint8)
    print(rgb.size())

    # try:
    #     # # img_tensor = linear(img_tensor)
    #     # unflatten = nn.Unflatten(2, (64, 64))
    #     # img_tensor = unflatten(img_tensor).to(torch.uint8)
    # except Exception as e:
    #     print(e)
    #     break

    torchvision.io.write_jpeg(input=rgb, filename=f'spec-img\\{i}-0.jpeg')

    # torchvision.io.write_jpeg(input=img_tensor[1], filename=f'spec-img\\{i}-1.jpeg')

#
# width = 1280
# height = 720
# channel = 3
#
# fps = frame_rate
#
# fourcc = cv2.VideoWriter_fourcc(*'MP42')
#
# video = cv2.VideoWriter("whodatboy.avi", fourcc, float(fps), (width, height))
#
# directory = 'spec-img'
#
# img_name_list = os.listdir(directory)
#
# for frame_count in range(len(img_name_list)):
#     print(frame_count)
#     img_name = img_name_list[frame_count]
#     img_path = os.path.join(directory, img_name)
#     enhance(img_path, frame_count)
#     # os.remove(f"Super Resolution\\Super Resolution{frame_count}.jpg")
#     # os.remove(f"Super Resolution{frame_count}-{frame_count}.jpg")
#
#     img = cv2.imread(f"Super Resolution\\Super Resolution{frame_count}.jpg")
#     img_resize = cv2.resize(img, (width, height))
#
#     video.write(img_resize)
#
# video.release()


# cv2.imwrite()
