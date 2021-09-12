from utils.utils import chop
from numpy import array, load, random, newaxis
from sklearn.utils import shuffle


# generator for npz files of whole songs
def npz_generator(spectrogram_list, chop_size=128, batch=1):
    while True:
        for spectrogram in spectrogram_list:
            file = load(spectrogram)

            mix = chop(file["mix"], chop_size)
            vocal = chop(file["vocal"], chop_size)
            mix, vocal = shuffle(mix, vocal)

            mix = array(mix)[:, :, :, newaxis]
            vocal = array(vocal)[:, :, :, newaxis]

            for i in range(0, len(mix) - 1, batch):
                yield mix[i:i + batch], vocal[i:i + batch]

        # shuffle after epoch
        random.shuffle(spectrogram_list)


# generator for npz files of chopped songs
def chop_npz_generator(chopped_spectrogram_list, batch=1):
    while True:
        mix_batch = []
        vocal_batch = []
        for i, spectrogram in enumerate(chopped_spectrogram_list):
            file = load(spectrogram)
            mix_batch.append(file["mix"])
            vocal_batch.append(file["vocal"])

            # yield batch size package
            if (i+1) % batch == 0:
                mix_batch = array(mix_batch)[:, :, :, newaxis]
                vocal_batch = array(vocal_batch)[:, :, :, newaxis]
                yield mix_batch, vocal_batch
                mix_batch = []
                vocal_batch = []

        # shuffle after epoch
        random.shuffle(chopped_spectrogram_list)
