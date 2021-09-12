from utils.utils import create_folder, audio_2_spectrogram, chop, plot_spec
from tqdm import tqdm
from argparse import ArgumentParser
import soundfile as sf
import numpy as np
import os

"""
output npz:
"mix" - mix array
"vocal" - vocal array
"""


class Data:
    def __init__(self, input_data_path="musdb18hq", arrays_to_save_path="musdb18hq_npz",
                 fft_size=1024, hop_length=512):
        self.list_mix_dir = [os.path.join(input_data_path, f) for f in os.listdir(input_data_path)]
        self.data_to_save_path = arrays_to_save_path
        self.fft_size = fft_size
        self.hop_length = hop_length

    """ for MUSDB18hq dataset """
    def get_train_data(self, chopping, scale=128):
        file_mix = "mixture.wav"
        file_vocal = "vocals.wav"

        for mix_dir in tqdm(self.list_mix_dir):
            # Split for Windows path
            f_name = mix_dir.split("\\")[-1]

            y_mix, sr = sf.read(os.path.join(mix_dir, file_mix), dtype='float32', always_2d=True)
            y_vocal, _ = sf.read(os.path.join(mix_dir, file_vocal), dtype='float32', always_2d=True)

            assert (y_mix.shape == y_vocal.shape)

            y_mix_specto, _, _ = audio_2_spectrogram(y_mix.T[1], self.fft_size, self.hop_length)
            y_vocal_specto, _, _ = audio_2_spectrogram(y_vocal.T[1], self.fft_size, self.hop_length)
            if not chopping:
                self.save_as_npz_array(y_mix_specto, y_vocal_specto, f_name + "_right")
            else:
                chopped_mix = chop(matrix=y_mix_specto, scale=scale)
                chopped_vocal = chop(matrix=y_vocal_specto, scale=scale)
                for i in range(len(chopped_mix)):
                    self.save_as_npz_array(chopped_mix[i], chopped_vocal[i], f_name + f"_{i}")

            # y_mix_specto, _, _ = audio_2_spectrogram(y_mix.T[0], self.fft_size, self.hop_length)
            # y_vocal_specto, _, _ = audio_2_spectrogram(y_vocal.T[0], self.fft_size, self.hop_length)
            # if not chopping:
            #     self.save_as_npz_array(y_mix_specto, y_vocal_specto, f_name + "_left")
            # else:
            #     chopped_mix = chop(matrix=y_mix_specto, scale=scale)
            #     chopped_vocal = chop(matrix=y_vocal_specto, scale=scale)
            #     for i in range(len(chopped_mix)):
            #         self.save_as_npz_array(chopped_mix[i], chopped_vocal[i], f_name + f"_left_{i}")

    def save_as_npz_array(self, mix, vocal, f_name):
        np.savez(os.path.join(self.data_to_save_path, f_name + ".npz"), mix=mix, vocal=vocal)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default="musdb18hq", type=str, help="Path for input data")
    parser.add_argument('--output', '-o', default="musdb18hq_npz", type=str, help="Path for created data")
    parser.add_argument('--fft_size', '-f', default=1024, type=int, help="Windows size for STFT")
    parser.add_argument('--hop_length', '-l', default=512, type=int, help="Hop length for STFT")
    parser.add_argument('--chop', '-c', action="store_true")
    parser.add_argument('--scale', '-s', default=128, type=int, help="Slice size after chop")
    args = parser.parse_args()

    create_folder(args.output)

    data = Data(input_data_path=args.input, arrays_to_save_path=args.output)
    data.get_train_data(chopping=args.chop, scale=args.scale)
    print("Done")


if __name__ == '__main__':
    main()
