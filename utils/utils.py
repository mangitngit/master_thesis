from math import ceil
from os import path, makedirs
from matplotlib import pyplot as plt
from matplotlib import use as mt_use
from librosa import display, stft, istft

import numpy as np

mt_use('Agg')


def create_folder(output):
    if not path.exists(output):
        makedirs(output)


def plot_spec(spectrogram, name, sr=44100, hop_length=512):
    fig = plt.figure(figsize=(15, 7))
    plt.title(name)
    display.specshow(spectrogram,
                     sr=sr,
                     hop_length=hop_length,
                     x_axis="time",
                     y_axis="linear")
    plt.clim(0, 1)
    plt.set_cmap("inferno")
    plt.colorbar()
    plt.savefig(name+".png")
    plt.clf()
    plt.close(fig)


def plot_history(history, name):
    fig = plt.figure()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name + "/accu.png")

    plt.close(fig)
    fig2 = plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name + "/loss.png")

    plt.close(fig2)


def calc_total_size(file_list, slice_size):
    slices_sum = 0
    for specto in file_list:
        file = np.load(specto)["mix"]
        slices_sum += (file.shape[1] // slice_size) * (file.shape[0] // slice_size)
    return slices_sum


# chop func for spectrogram
def chop(matrix, scale):
    slices = []
    for time_axis in range(0, matrix.shape[1] // scale):
        for freq_axis in range(0, matrix.shape[0] // scale):
            s = matrix[freq_axis * scale: (freq_axis + 1) * scale, time_axis * scale: (time_axis + 1) * scale]
            slices.append(s)
    return slices


# expand spectrogram to fit a predict function
def expand_spectrogram(spectrogram, slice_size):
    new_x = ceil(spectrogram.shape[0] / slice_size) * slice_size
    new_y = ceil(spectrogram.shape[1] / slice_size) * slice_size
    new_spectrogram = np.zeros((new_x, new_y))
    new_spectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram
    return new_spectrogram


# make a log(x+1) normalized spectrogram
def audio_2_spectrogram(audio, fft_size=1024, hop_length=512):
    spectrogram = stft(audio, n_fft=fft_size, hop_length=hop_length)
    magnitude = np.log1p(np.abs(spectrogram).astype(np.float32))
    phase = np.exp(1.j * np.angle(spectrogram))

    coef = magnitude.max()
    return magnitude / coef, phase, coef


def spectrogram_2_audio(spectro, phase, coef, hop_size, length):
    log_magnitude_again = spectro * coef
    magnitude_again = np.expm1(log_magnitude_again)
    spectro_again = magnitude_again * phase

    mix_again = istft(spectro_again, hop_length=hop_size, length=length)
    return mix_again
