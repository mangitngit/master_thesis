from utils.unet import unet, unet_short
from utils.utils import audio_2_spectrogram, spectrogram_2_audio,\
    create_folder, plot_spec, expand_spectrogram

from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser

import soundfile as sf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Isolator:
    def __init__(self, input_path="files/mixture.wav", output_path="files",
                 frame_size=1024, hop_size=512, slice_size=128, hard_mask=0.0, loop=1, plot=False):

        self.y_mix, self.sr = sf.read(input_path, dtype='float32', always_2d=True)

        tmp = input_path.split('/')[-1]
        self.wave_name = output_path + "/" + tmp.split('.')[0]+"_vocal"
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.slice_size = slice_size

        self.hard_mask = hard_mask
        self.loops = loop
        self.do_plot = plot

        self.model = unet_short()
        self.model.load_weights("train_results/weights_short.h5")

        self.out_specto = np.zeros(self.y_mix.shape)

        self.execute()

    def execute(self):
        for channel in range(2):
            spectro, phase, coef = audio_2_spectrogram(self.y_mix.T[channel],
                                                       self.frame_size,
                                                       self.hop_size)

            expand_spectro = expand_spectrogram(spectro[:, :], self.slice_size)
            spectro_predict_ready = expand_spectro[np.newaxis, :, :, np.newaxis]

            for _ in range(self.loops):
                predict = self.model.predict(spectro_predict_ready[:, :, 0:self.slice_size, :])
                for part in tqdm(range(self.slice_size, spectro_predict_ready.shape[2], self.slice_size)):
                    slice_predict = self.model.predict(spectro_predict_ready[:, :, part:part+self.slice_size, :])
                    predict = np.append(predict, slice_predict, 2)
                spectro_predict_ready = deepcopy(predict)

            vocal_predict = spectro_predict_ready[0, :spectro.shape[0], :spectro.shape[1], 0]

            if self.hard_mask != 0.0:
                vocal_predict = np.where(vocal_predict > self.hard_mask, spectro, vocal_predict)

            if self.do_plot:
                plot_spec(vocal_predict, self.wave_name, self.sr, self.hop_size)
            out_audio = spectrogram_2_audio(vocal_predict, phase, coef, self.hop_size, self.y_mix.shape[0])

            self.out_specto[:, channel] = out_audio

        sf.write(self.wave_name+".wav", self.out_specto, self.sr)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default="files/mixture.wav", type=str)
    parser.add_argument('--output', '-o', default="predict_results/", type=str, help="Path to train results")
    parser.add_argument('--fft_size', '-f', default=1024, type=int, help="Windows size for STFT")
    parser.add_argument('--hop_length', '-l', default=512, type=int, help="Hop length for STFT")
    parser.add_argument("--slice_size", '-s', default=128, type=int, help="Size of slices from spectrogram.")
    parser.add_argument('--hard_mask', '-m', default=0.0, type=float, help="Set threshold for hard mask if used")
    parser.add_argument("--loop", '-L', default=1, type=int, help="Amount of loops in predict func")
    parser.add_argument('--do_plot', '-d', action='store_true', help="Plot spectrogram")
    parser.add_argument('--gpu', '-g', action='store_true', help="Run with GPU")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if args.gpu else '-1'
    create_folder(args.output)

    Isolator(args.input, args.output, args.fft_size, args.hop_length,
             args.slice_size, args.hard_mask, args.loop, args.do_plot)


if __name__ == '__main__':
    main()
