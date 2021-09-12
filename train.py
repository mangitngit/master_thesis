from utils.unet import unet, unet_short
from utils.utils import create_folder, plot_history, calc_total_size
from utils.generators import chop_npz_generator, npz_generator

from argparse import ArgumentParser
from librosa.util import find_files
from keras.callbacks import ModelCheckpoint, CSVLogger
from math import ceil
from numpy import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Train:
    def __init__(self, input_path="musdb18hq_npz", chopped=False, results_path="train_results",
                 val_split=0.1, epochs=10, batch=16, slice_size=128):
        self.chopped = chopped
        self.filepath_accu = ""
        self.filepath_val_accu = ""
        self.filepath_loss = ""
        self.model = unet()

        self.file_list = find_files(input_path, ext="npz")
        random.shuffle(self.file_list)

        self.val_split = ceil(val_split * len(self.file_list))

        self.slice_size = slice_size
        self.total_size = calc_total_size(self.file_list, self.slice_size) if not self.chopped else len(self.file_list)
        # print(self.total_size)
        # self.total_size = 94892

        self.batch = batch
        self.epochs = epochs

        self.val_steps = self.total_size * val_split
        self.epoch_steps = ceil((self.total_size - self.val_steps) / self.batch)
        self.val_steps = ceil(self.val_steps / self.batch)

        self.filepath_val_accu = results_path + "/val_acc-{epoch:02d}-{val_accuracy:.2f}.h5"
        self.filepath_accu = results_path + "/acc-{epoch:02d}-{accuracy:.2f}.h5"
        self.filepath_loss = results_path + "/loss-{epoch:02d}-{accuracy:.2f}.h5"

        self.run(results_path)

    def run(self, name):
        csv_logger = CSVLogger(name + "/model_history.csv", append=True)
        checkpoint_val_accu = ModelCheckpoint(self.filepath_val_accu, monitor='val_accuracy', verbose=1,
                                              save_best_only=True, save_weights_only=True, mode='max')
        checkpoint_accu = ModelCheckpoint(self.filepath_accu, monitor='accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, mode='max')
        checkpoint_loss = ModelCheckpoint(self.filepath_loss, monitor='val_loss', verbose=1, save_best_only=True,
                                          save_weights_only=True, mode='min')

        callbacks_list = [checkpoint_val_accu, checkpoint_accu, checkpoint_loss, csv_logger]

        if self.chopped:
            gener = chop_npz_generator(self.file_list[:-self.val_split], self.batch)
            valider = chop_npz_generator(self.file_list[-self.val_split:], self.batch)
        else:
            gener = npz_generator(self.file_list[:-self.val_split], self.slice_size, self.batch)
            valider = npz_generator(self.file_list[-self.val_split:], self.slice_size, self.batch)

        history = self.model.fit(x=gener,
                                 epochs=self.epochs,
                                 steps_per_epoch=self.epoch_steps,
                                 validation_data=valider,
                                 validation_steps=self.val_steps,
                                 callbacks=callbacks_list)
        plot_history(history, name)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default="musdb18hq_npz", type=str)
    parser.add_argument('--chopped', '-c', action='store_true', help="Spectrograms in database are already chopped")
    parser.add_argument('--output', '-o', default="train_results", type=str, help="Path to train results")
    parser.add_argument("--val_split", '-v', default=0.1, type=float, help="Proportion of the data to train on")
    parser.add_argument("--batch", '-b', default=32, type=int, help="Batch size for training")
    parser.add_argument("--epochs", '-e', default=100, type=int, help="Number of epochs to train.")
    parser.add_argument("--slice_size", '-s', default=128, type=int, help="Size of slices from spectrogram.")
    parser.add_argument('--gpu', '-g', action='store_true', help="Run with GPU")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if args.gpu else '-1'
    create_folder(args.output)

    Train(args.input, args.chopped, args.output, args.val_split, args.epochs, args.batch, args.slice_size)


if __name__ == '__main__':
    main()
